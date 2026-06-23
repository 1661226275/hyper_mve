"""Worker — v4 episode collection (Pkg-05 spec 02, Ch5.6).

Rewrite of the v4.7 worker for the v4 stack:
    - BeliefNet.step online inference (C5-W2) replaces v4.7 set_context_from_history.
    - 7-API per-agent context (set_context_objective once + set_context_subjective
      per agent), no model.update_step (C5-W1: worker is always no_grad inference).
    - emits v4 TimeStepRecord (12 fields) + a parallel c_t scalar sequence.
    - holds a single MVEPlanner instance so the CRN seed persists across episodes
      (review 修订 5).

[v4-opt 2026-06] Vectorized collection (2agent run diagnosis): the MVE planner is
already batched over B, but the old worker fed it B=1 — ~70 tiny GPU calls per env
step, kernel-launch bound (measured 0.08 train-steps/s). ``collect_episodes`` now
steps ``n_envs`` ResourceCommons environments in lockstep (the env never terminates
before T_max, Pkg-02 spec 08) and runs the planner once per env-step with
B=n_envs. ``collect_episode`` keeps the old single-env API by delegating with B=1.
The batched path also powers deterministic evaluation (epsilon=0 + argmax actions)
and surfaces per-episode returns + planner diagnostics (q_std / q_gap /
uniform_frac) that were previously discarded.

Self-Info strictness (C11): set_context_subjective only ever gets the RAW (B, 4)
capability vector — never oracle types. Own type is resolved inside the model from
cfg.env.type_assignment, not passed here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from hyper_mve.configs import V4Config
from hyper_mve.models import HyperMuZeroModel
from hyper_mve.planning.mve_planner import MVEPlanner
from hyper_mve.schemas import TimeStepRecord
from hyper_mve.utils.utils import inverse_scalar_transform


@dataclass
class CollectResult:
    """One collected episode + the per-episode observability scalars [v4-opt 2026-06].

    ``records`` / ``c_t_seq`` are exactly the two ``store_episode`` arguments; the
    rest feed the ``collect/*`` TensorBoard family (returns) and the planner-target
    health probes (q_std / q_gap / uniform_frac are NaN when the planner was off).
    """

    records: list[TimeStepRecord]
    c_t_seq: torch.Tensor          # (T,) float32
    returns: np.ndarray            # (N,) per-agent undiscounted episode return (ΣR, subjective)
    pi_entropy_mean: float         # mean entropy of the stored pi_mve targets
    q_std_mean: float              # planner candidate-return std (raw, pre-floor)
    q_gap_mean: float              # planner candidate-return max-min spread
    uniform_frac: float            # fraction of rows hit by the qstd noise guard
    # [2026-06 thesis welfare] per-agent cumulative PHYSICAL harvest Σ_t u_{i,t}
    # (from info['harvests']) and end-of-episode resource sustainability
    # S = Σ_k q_{k,Tmax} / (K·Q_max). Optional/NaN defaults keep any other
    # CollectResult construction valid; collect_episodes always populates them.
    phys_returns: Optional[np.ndarray] = None   # (N,) float32 — social-physical-welfare signal
    sustainability: float = float("nan")        # S ∈ [0, 1]; feeds fairness/tragedy too


class Worker:
    """Collects episodes into v4 TimeStepRecords (single-env or vectorized)."""

    def __init__(
        self,
        cfg: V4Config,
        model: HyperMuZeroModel,
        env=None,
        planner: Optional[MVEPlanner] = None,
        envs: Optional[list] = None,
    ):
        """Either ``env`` (single, legacy) or ``envs`` (vectorized) must be given.

        All envs must share ``cfg.env`` dimensions (N/A/T_max); they are stepped in
        lockstep, which is safe because ResourceCommons only terminates at T_max.
        """
        self.cfg = cfg
        self.model = model
        if envs is None:
            assert env is not None, "Worker needs `env` or `envs`"
            envs = [env]
        elif env is not None:
            raise ValueError("pass either `env` or `envs`, not both")
        self.envs = envs
        self.env = envs[0]  # legacy single-env attribute (tests / scripts)
        # review 修订 5: hold one planner so CRN rng persists across episodes.
        self.planner = planner if planner is not None else MVEPlanner(cfg)
        self.model.eval()

    @torch.no_grad()
    def collect_episode(
        self,
        epsilon: float = 0.1,
        use_planner: bool = True,
    ) -> tuple[list[TimeStepRecord], torch.Tensor]:
        """Collect a full episode on ``self.env`` (legacy B=1 API).

        Returns:
            (records, c_t_seq): T TimeStepRecords and the parallel (T,) c_t series
            (the second arg to ``EpisodeReplayBuffer.store_episode``).
        """
        result = self.collect_episodes(
            n_envs=1, epsilon=epsilon, use_planner=use_planner,
        )[0]
        return result.records, result.c_t_seq

    @torch.no_grad()
    def collect_episodes(
        self,
        n_envs: Optional[int] = None,
        epsilon: float = 0.1,
        use_planner: bool = True,
        deterministic: bool = False,
        reset_seeds: Optional[list[Optional[int]]] = None,
        reset_options: Optional[list[Optional[dict]]] = None,
        tiebreak_rng: Optional[np.random.Generator] = None,
    ) -> list[CollectResult]:
        """Collect one episode from each of the first ``n_envs`` envs, in lockstep.

        Args:
            n_envs: how many of ``self.envs`` to run (default: all).
            epsilon: ε-greedy mix-in (ignored when ``deterministic``).
            use_planner: planner-on collection (Alg 5.2 a11-a14). With the planner
                OFF the stored pi_mve is the model's own prior — a self-distillation
                target; the caller must mark such episodes ``planner_on=False`` so
                the policy loss can mask them (Ch5.9.1b).
            deterministic: evaluation mode — actions = argmax(pi_mve), no ε, no
                sampling RNG consumed.
            reset_seeds / reset_options: optional per-env ``env.reset`` arguments
                (evaluation uses them to pin seeds and force a static c).
            tiebreak_rng: optional reproducible RNG used to break argmax ties in
                deterministic mode [v4-opt 2026-06b]. Without it, deterministic
                actions on a uniform pi_mve (e.g. rows the noise guard fell back
                to uniform) collapse to action 0 — which in ResourceCommons is
                NOOP, masking the planner's true performance. With it, the choice
                is uniform-random over the tied argmax set and still reproducible
                across run_eval calls (same rng state → same picks).

        Returns:
            list of ``CollectResult``, one per env, in env order.
        """
        if n_envs is None:
            n_envs = len(self.envs)
        assert 1 <= n_envs <= len(self.envs), f"n_envs={n_envs} not in [1, {len(self.envs)}]"
        envs = self.envs[:n_envs]
        B = n_envs
        N = self.cfg.env.N
        A = self.cfg.env.A
        device = next(self.model.parameters()).device

        obs_list, infos = [], []
        for b, env in enumerate(envs):
            seed_b = reset_seeds[b] if reset_seeds is not None else None
            opts_b = reset_options[b] if reset_options is not None else None
            obs_b, info_b = env.reset(seed=seed_b, options=opts_b)
            obs_list.append(np.asarray(obs_b, dtype=np.float32))
            infos.append(info_b)

        prev_hidden = self.model.belief_net.init_hidden(B, N, device=device)

        records: list[list[TimeStepRecord]] = [[] for _ in range(B)]
        c_t_lists: list[list[float]] = [[] for _ in range(B)]
        returns = np.zeros((B, N), dtype=np.float64)
        # [2026-06 thesis welfare] per-agent cumulative PHYSICAL harvest Σ_t u
        # (info['harvests'] is the public per-step harvest, untouched by the
        # Fehr-Schmidt φ·ψ term that makes `returns` subjective).
        phys_returns = np.zeros((B, N), dtype=np.float64)
        ent_acc = torch.zeros(B, device=device)
        qstd_acc = torch.zeros(B, device=device)
        qgap_acc = torch.zeros(B, device=device)
        uniform_acc = 0.0
        t_steps = 0

        for t in range(self.cfg.env.T_max):
            obs_np = np.stack(obs_list, axis=0)                # (B, N, obs_dim)
            obs_t = torch.from_numpy(obs_np).to(device)

            # --- BeliefNet online inference (C5-W2) ---
            prev_hidden, c_hat, z_hat = self.model.belief_net.step(obs_t, prev_hidden)
            # c_hat (B, N), z_hat (B, N, N-1, 2)

            s = self.model.encode(obs_t)                       # (B, latent_dim)
            c_t_scalars = [float(info["c_true"]) for info in infos]
            c_t_tensor = torch.tensor(c_t_scalars, dtype=torch.float32, device=device)
            self.model.set_context_objective(c_t_tensor)

            cap_np = np.stack([
                np.stack([np.asarray(info["caps"][k].to_array(), dtype=np.float32)
                          for k in range(N)], axis=0)
                for info in infos
            ], axis=0)                                          # (B, N, 4) RAW caps
            cap_t = torch.from_numpy(cap_np).to(device)

            cap_dict: dict[int, torch.Tensor] = {}
            belief_dict: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            value_estimates = torch.zeros(B, N, device=device)
            prior_pi = torch.zeros(B, N, A, device=device)

            for k in range(N):
                cap_dict[k] = cap_t[:, k]                       # (B, 4) (no type leak)
                belief_dict[k] = (c_hat[:, k], z_hat[:, k])

                self.model.set_context_subjective(k, cap_dict[k], belief_dict[k])
                logits_k, v_k = self.model.predict(s)
                value_estimates[:, k] = inverse_scalar_transform(v_k).squeeze(-1)
                if not use_planner:
                    prior_pi[:, k] = torch.softmax(logits_k, dim=-1)

            if use_planner:
                pi_t, diag = self.planner.sample_mve_plan(
                    self.model, s, cap_dict, belief_dict, c_t_tensor,
                    return_diagnostics=True,
                )                                               # (B, N, A)
                qstd_acc += diag["q_std"].mean(dim=1)
                qgap_acc += diag["q_gap"].mean(dim=1)
                uniform_acc += float(diag["uniform_frac"])
            else:
                pi_t = prior_pi

            ent_acc += -(pi_t.clamp_min(1e-9).log() * pi_t).sum(-1).mean(dim=1)
            t_steps += 1

            pi_np = pi_t.cpu().numpy().astype(np.float32)       # (B, N, A)
            v_np = value_estimates.cpu().numpy().astype(np.float32)
            c_hat_np = c_hat.cpu().numpy().astype(np.float32)
            z_hat_np = z_hat.cpu().numpy().astype(np.float32)

            # --- action selection: argmax (eval) or epsilon-greedy (per env, agent) ---
            joint_actions = np.zeros((B, N), dtype=np.int64)
            if deterministic:
                # Tie-broken argmax: deterministic with an explicit rng, plain argmax
                # without. NOOP=action 0 in ResourceCommons, so a tied row (e.g. one
                # the planner's noise guard set to uniform) would otherwise always
                # pick 0 and trivially underestimate the planner [v4-opt 2026-06b].
                if tiebreak_rng is None:
                    joint_actions = pi_np.argmax(axis=-1)
                else:
                    for b in range(B):
                        for k in range(N):
                            p = pi_np[b, k]
                            best = np.flatnonzero(p >= p.max() - 1e-9)
                            joint_actions[b, k] = (int(best[0]) if best.size == 1
                                                   else int(tiebreak_rng.choice(best)))
            else:
                for b in range(B):
                    for k in range(N):
                        if np.random.rand() < epsilon:
                            joint_actions[b, k] = np.random.randint(A)
                        else:
                            p = pi_np[b, k].astype(np.float64)
                            p = p / p.sum()
                            joint_actions[b, k] = int(np.random.choice(A, p=p))

            # --- step every env + write TimeStepRecords (C5-W3) ---
            all_done = True
            for b, env in enumerate(envs):
                next_obs, reward, done, _truncated, next_info = env.step(joint_actions[b])
                reward = np.asarray(reward, dtype=np.float32)
                returns[b] += reward
                harv = next_info.get("harvests") if isinstance(next_info, dict) else None
                if harv is not None:
                    phys_returns[b] += np.asarray(harv, dtype=np.float64)

                records[b].append(TimeStepRecord(
                    o=obs_np[b],
                    a=joint_actions[b],
                    r=reward,
                    delta=np.asarray(next_info["deltas"], dtype=np.float32),
                    pi_mve=pi_np[b],
                    v=v_np[b],
                    tau=np.asarray(infos[b]["types"], dtype=np.int8),
                    cap=cap_np[b],
                    c_hat=c_hat_np[b],
                    z_hat=z_hat_np[b],
                    t=t,
                    done=bool(done),
                ))
                c_t_lists[b].append(c_t_scalars[b])

                obs_list[b] = np.asarray(next_obs, dtype=np.float32)
                infos[b] = next_info
                all_done = all_done and bool(done)

            # Envs share T_max and never terminate early (Pkg-02 spec 08), so they
            # finish together; the guard keeps the lockstep invariant explicit.
            if all_done:
                break

        ent_mean = (ent_acc / max(1, t_steps)).cpu().numpy()
        if use_planner:
            qstd_mean = (qstd_acc / max(1, t_steps)).cpu().numpy()
            qgap_mean = (qgap_acc / max(1, t_steps)).cpu().numpy()
            uniform_mean = uniform_acc / max(1, t_steps)
        else:
            qstd_mean = np.full(B, np.nan)
            qgap_mean = np.full(B, np.nan)
            uniform_mean = float("nan")

        # [2026-06 thesis welfare] sustainability S = Σ_k q_{k,Tmax} / (K·Q_max)
        # from the final info's resource_state (shape (K, 3) = [pos_x, pos_y,
        # stock]); the stock is the last column. NaN if the env didn't surface it.
        denom = int(self.cfg.env.K) * float(self.cfg.env.Q_max)
        sustain = np.full(B, np.nan, dtype=np.float64)
        if denom > 0:
            for b in range(B):
                res = infos[b].get("resource_state") if isinstance(infos[b], dict) else None
                if res is None:
                    continue
                arr = np.asarray(res, dtype=np.float64)
                stocks = arr[:, -1] if arr.ndim == 2 else arr
                sustain[b] = float(stocks.sum()) / denom

        return [
            CollectResult(
                records=records[b],
                c_t_seq=torch.tensor(c_t_lists[b], dtype=torch.float32),
                returns=returns[b].astype(np.float32),
                pi_entropy_mean=float(ent_mean[b]),
                q_std_mean=float(qstd_mean[b]),
                q_gap_mean=float(qgap_mean[b]),
                uniform_frac=float(uniform_mean),
                phys_returns=phys_returns[b].astype(np.float32),
                sustainability=float(sustain[b]),
            )
            for b in range(B)
        ]
