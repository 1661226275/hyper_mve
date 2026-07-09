"""Worker — v5 episode collection (Pkg-09; base: Pkg-05 spec 02).

v5 changes (relative to the v4 worker):
    - BeliefNet.step returns (hidden, g_hat) — the |G|-way regime posterior.
    - No ``set_context_objective`` (6-API v5): transition is a plain shared
      module; the per-agent loop only calls ``set_context_subjective(k,
      row_k, g_hat_k)``.
    - Oracle discipline: ``info["rows"]`` is an Oracle field; the worker
      applies **row-i-only-for-agent-i** — agent k's conditioning gets
      ``rows[:, k]`` only (the same information its own observation carries).
      ``info["g_true"]`` is consumed solely as the supervision label stored
      in the record.
    - emits v5 TimeStepRecords (o, a, r, pi_mve, v, row, g_hat, g).

[v4-opt 2026-06] Vectorized collection retained: ``collect_episodes`` steps
``n_envs`` RelationCommons environments in lockstep (the env never terminates
before T_max) and runs the planner once per env-step with B=n_envs.
``collect_episode`` keeps the single-env API by delegating with B=1. The
batched path also powers deterministic evaluation (epsilon=0 + argmax
actions) and surfaces per-episode returns + planner diagnostics.
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
    """One collected episode + per-episode observability scalars.

    ``records`` is the ``store_episode`` argument; the rest feed the
    ``collect/*`` TensorBoard family (returns) and the planner-target health
    probes (q_std / q_gap / uniform_frac are NaN when the planner was off).
    """

    records: list[TimeStepRecord]
    returns: np.ndarray            # (N,) per-agent undiscounted episode return (ΣR, subjective)
    pi_entropy_mean: float         # mean entropy of the stored pi_mve targets
    q_std_mean: float              # planner candidate-return std (raw, pre-floor)
    q_gap_mean: float              # planner candidate-return max-min spread
    uniform_frac: float            # fraction of rows hit by the qstd noise guard
    # [2026-06 thesis welfare] per-agent cumulative PHYSICAL harvest Σ_t u_{i,t}
    # (from info['harvests']) and end-of-episode resource sustainability
    # S = Σ_k q_{k,Tmax} / (K·Q_max).
    phys_returns: Optional[np.ndarray] = None   # (N,) float32 — social-physical-welfare signal
    sustainability: float = float("nan")        # S ∈ [0, 1]; feeds fairness/tragedy too


class Worker:
    """Collects episodes into v5 TimeStepRecords (single-env or vectorized)."""

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
        lockstep, which is safe because RelationCommons only terminates at T_max.
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
    ) -> list[TimeStepRecord]:
        """Collect a full episode on ``self.env`` (single-env API).

        Returns:
            records: T TimeStepRecords (the ``store_episode`` argument).
        """
        result = self.collect_episodes(
            n_envs=1, epsilon=epsilon, use_planner=use_planner,
        )[0]
        return result.records

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
            use_planner: planner-on collection. With the planner OFF the stored
                pi_mve is the model's own prior — a self-distillation target;
                the caller must mark such episodes ``planner_on=False`` so the
                policy loss can mask them (Ch5.9.1b).
            deterministic: evaluation mode — actions = argmax(pi_mve), no ε, no
                sampling RNG consumed.
            reset_seeds / reset_options: optional per-env ``env.reset`` arguments
                (evaluation uses them to pin seeds and force a regime g).
            tiebreak_rng: optional reproducible RNG used to break argmax ties in
                deterministic mode [v4-opt 2026-06b].

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
        returns = np.zeros((B, N), dtype=np.float64)
        # [2026-06 thesis welfare] per-agent cumulative PHYSICAL harvest Σ_t u
        # (info['harvests'] is the public per-step harvest, untouched by the
        # relational mixing that makes `returns` subjective).
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
            prev_hidden, g_hat = self.model.belief_net.step(obs_t, prev_hidden)
            # g_hat (B, N, |G|)

            s = self.model.encode(obs_t)                       # (B, latent_dim)

            # Oracle rows, row-i-only-for-agent-i discipline: agent k's
            # conditioning receives rows[:, k] only.
            rows_np = np.stack(
                [np.asarray(info["rows"], dtype=np.float32) for info in infos],
                axis=0,
            )                                                   # (B, N, N-1)
            rows_t = torch.from_numpy(rows_np).to(device)

            row_dict: dict[int, torch.Tensor] = {}
            belief_dict: dict[int, torch.Tensor] = {}
            value_estimates = torch.zeros(B, N, device=device)
            prior_pi = torch.zeros(B, N, A, device=device)

            for k in range(N):
                row_dict[k] = rows_t[:, k]                      # (B, N-1) own row only
                belief_dict[k] = g_hat[:, k]                    # (B, |G|)

                self.model.set_context_subjective(k, row_dict[k], belief_dict[k])
                logits_k, v_k = self.model.predict(s)
                value_estimates[:, k] = inverse_scalar_transform(v_k).squeeze(-1)
                if not use_planner:
                    prior_pi[:, k] = torch.softmax(logits_k, dim=-1)

            if use_planner:
                pi_t, diag = self.planner.sample_mve_plan(
                    self.model, s, row_dict, belief_dict,
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
            g_hat_np = g_hat.cpu().numpy().astype(np.float32)

            # --- action selection: argmax (eval) or epsilon-greedy (per env, agent) ---
            joint_actions = np.zeros((B, N), dtype=np.int64)
            if deterministic:
                # Tie-broken argmax [v4-opt 2026-06b]: NOOP=action 0, so a tied
                # row (e.g. one the planner's noise guard set to uniform) would
                # otherwise always pick 0 and underestimate the planner.
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
                    pi_mve=pi_np[b],
                    v=v_np[b],
                    row=rows_np[b],
                    g_hat=g_hat_np[b],
                    g=int(infos[b]["g_true"]),
                    t=t,
                    done=bool(done),
                ))

                obs_list[b] = np.asarray(next_obs, dtype=np.float32)
                infos[b] = next_info
                all_done = all_done and bool(done)

            # Envs share T_max and never terminate early, so they finish
            # together; the guard keeps the lockstep invariant explicit.
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
