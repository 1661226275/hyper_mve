"""Per-Agent Coordinate Descent MVE Planner (v4.6 algorithm, v5 API — Pkg-09).

For each agent in randomised order (coordinate descent):
    1. Pre-sample other agents' step-0 actions once per scenario (CRN)
    2. Enumerate all A candidate first-actions for this agent
    3. Replicate other agents' step-0 actions across all A candidates (noise cancels)
    4. For step>0: all agents sample from policy network (independent)
    5. Rollout K steps, accumulate agent j's discounted returns + terminal value
    6. pi_mve[j] = softmax(mean_return_per_action / temperature)

v4.6 Common Random Numbers (CRN) — load-bearing: the other agents' step-0
actions are sampled ONCE per scenario and SHARED across all A candidate
actions, cancelling the dominant noise source so the planner can detect the
true per-action signal. **The 4-phase compute below is preserved verbatim.**

v5 changes (Pkg-09):
    - ``sample_mve_plan(model, root_s, row, belief)``: the ``cap`` dict becomes
      the per-agent own-row dict ``{k: (B, N-1)}``; ``belief`` values are single
      regime-posterior tensors ``(B, |G|)``; the ``c_t`` argument is gone.
    - ``set_context_objective`` no longer exists: ``model.transition`` is a
      plain shared module (batch-agnostic), so the objective θ-cache/install
      machinery is deleted. The subjective θ-cache is retained verbatim.
    - **p > 0 approximation (research point 2)**: within the imagined K-step
      rollout the relationship regime is FROZEN at the current belief — the
      world model does not simulate regime switches. Real-step belief updates
      (worker) handle switches between plans. Under p = 0 (research point 1)
      this is exact; under p > 0 it biases plans over horizons ≳ 1/p, which is
      the documented trade-off (see sdd/pkg-09-dynamic-relations/design.md).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from hyper_mve.configs import V4Config
from hyper_mve.utils.utils import actions_to_one_hot, inverse_scalar_transform


def _is_hyper_model(model) -> bool:
    """v5 HyperMuZeroModel/BaselineModel expose set_context_subjective (vs legacy get_id_emb)."""
    return hasattr(model, "set_context_subjective")


def _expand_dim0(t, repeats):
    """Expand a tensor (or None) along dim 0 for parallel sampling (CRN layout)."""
    if t is None:
        return None
    return t.repeat_interleave(repeats, dim=0)


def _set_subjective(model, agent_idx, row_b, belief_b):
    """v5: set per-agent subjective context (own row + regime posterior)."""
    model.set_context_subjective(
        agent_idx,
        row_b[:, agent_idx],                              # (batch, N-1)
        belief_b[:, agent_idx],                           # (batch, |G|)
    )


def _multinomial_sample(probs, generator):
    """Deterministic categorical sample (given generator); probs (BM, A) -> (BM,)."""
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


class MVEPlanner:
    """MVE planner — CRN + coordinate descent (Pkg-05 spec 06, v5 API)."""

    def __init__(self, cfg: V4Config):
        self.cfg = cfg
        self.N = cfg.env.N
        self.A = cfg.env.A
        self.mve_samples = cfg.train.mve_samples
        self.mve_depth = cfg.train.mve_depth
        self.mve_temperature = cfg.train.mve_temperature
        # [v4-opt 2026-06] uniform fallback when candidate returns are noise-level flat.
        self.mve_qstd_floor = getattr(cfg.train, "mve_qstd_floor", 0.0)
        self.gamma = cfg.train.gamma
        self.use_crn = cfg.train.use_crn
        # Planner-side coordinate-descent agent ordering (pkg-08 spec 06 §4).
        self.randomize_order = cfg.train.randomize_order

        # CRN state — persists across episodes (review 修订 5); arbitrary seed.
        self.crn_rng = np.random.default_rng(seed=cfg.train.epsilon_decay_steps)

    def _sample_policy_action(self, model, curr_s, agent_idx, batch_size, device,
                              is_hyper, row_b, belief_b, generator, install_subj=None):
        """Sample action for ``agent_idx`` from the model policy (deterministic).

        ``install_subj`` (θ-cache fast path): an optional ``(agent_idx, batch)``
        callable that installs the agent's pre-generated subjective θ instead of
        regenerating it via ``set_context_subjective``. Numerically identical;
        see ``sample_mve_plan``'s θ-cache block.
        """
        if is_hyper:
            if install_subj is not None:
                install_subj(agent_idx, batch_size)
            else:
                _set_subjective(model, agent_idx, row_b, belief_b)
            logits_i, _ = model.predict(curr_s)
        else:
            id_i = torch.full((batch_size,), agent_idx, dtype=torch.long, device=device)
            id_emb_i = model.get_id_emb(id_i)
            logits_i, _ = model.predict(curr_s, id_emb_i)
        return _multinomial_sample(F.softmax(logits_i, dim=-1), generator)

    @torch.no_grad()
    def sample_mve_plan(self, model, root_s, row: dict, belief: dict,
                        return_diagnostics: bool = False,
                        eval_use_crn: Optional[bool] = None,
                        eval_randomize_order: Optional[bool] = None):
        """Per-agent coordinate-descent MVE planning with CRN.

        Args:
            model:  HyperMuZeroModel (or baseline via duck-typing).
            root_s: (B, latent_dim) current latent state.
            row:    {agent_id: (B, N-1)} own relationship row per agent (D7;
                    row-i-only-for-agent-i discipline upheld by the caller).
            belief: {agent_id: (B, |G|)} regime posterior per agent (D7).
            return_diagnostics: if True, also return the per-agent per-action
                expected returns and their normalised scores.
            eval_use_crn / eval_randomize_order: pkg-08 spec 03 §4.2 per-call
                overrides; ``None`` (default) preserves training-time flags.

        Returns:
            pi_mve: (B, N, A) per-agent search policy.
            If ``return_diagnostics``: ``(pi_mve, diag)`` with keys
            ``returns_per_action`` (B, N, A); ``q_normalized`` (B, N, A);
            ``q_std`` (B, N) raw per-candidate return std (pre-floor);
            ``q_gap`` (B, N); ``uniform_frac`` scalar [v4-opt 2026-06].
        """
        B = root_s.shape[0]
        N, A = self.N, self.A
        device = root_s.device

        # --- D7 / R5-10: validate dict inputs (shape drift / info-leak guard) ---
        for k in range(N):
            assert k in row and k in belief, f"row/belief missing agent_id {k}"
            assert row[k].dim() == 2 and row[k].shape[-1] == N - 1, (
                f"row[{k}].shape must be (B, {N - 1}), got {tuple(row[k].shape)}"
            )
            assert torch.is_tensor(belief[k]) and belief[k].dim() == 2, (
                f"belief[{k}] must be a (B, |G|) tensor (v5), got "
                f"{type(belief[k]).__name__}"
            )

        # --- pack dicts -> tensors for the (verbatim) CRN compute below ---
        row_t = torch.stack([row[k] for k in range(N)], dim=1)            # (B, N, N-1)
        belief_t = torch.stack([belief[k] for k in range(N)], dim=1)      # (B, N, |G|)

        K = self.mve_depth
        S = self.mve_samples
        gamma = self.gamma
        temperature = self.mve_temperature
        is_hyper = _is_hyper_model(model)

        # ── Subjective θ-cache fast path (numerically equivalent; HyperMuZeroModel only) ──
        # Each agent's θ_rew^i / θ_pred^i depends only on (row_i, belief_i) — both
        # INVARIANT across the K rollout steps AND across every coordinate-descent
        # iteration. The baseline path regenerates them on every step at the
        # expanded B*M batch. Here we generate them ONCE at base batch B and reuse
        # the identical tensors, tiling by repeat_interleave to whatever batch each
        # call site needs.
        #
        # Equivalence: (1) functional nets are per-row (bmm + per-sample norm),
        # so tile-then-consume == consume-tiled-context; (2) hyper forward is a
        # deterministic map in eval (no dropout / no RNG); (3) the planner's
        # batch growth is pure repeat_interleave (B -> B*spa -> B*M). It consumes
        # NO RNG, so ``gen`` / ``crn_rng`` draw in the same order → the same
        # sampling decisions. Bit-identical on CPU; ~1e-6 rel on CUDA (kernel
        # selection), far below the planner's own noise floor.
        # v5: transition is a plain shared module (batch-agnostic) — the v4
        # objective θ-cache/install machinery no longer exists.
        theta_cache_on = is_hyper and hasattr(model, "install_subjective_theta")
        cached_subj: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}  # agent -> (θ_rew, θ_pred) @ B
        install_subj = None
        if theta_cache_on:
            for i in range(N):
                _set_subjective(model, i, row_t, belief_t)         # θ_rew^i/θ_pred^i @ B
                cached_subj[i] = model.current_subjective_thetas()

            def _install_subjective(agent_idx, batch):
                """Install agent θ_rew/θ_pred tiled B -> batch (== set_context_subjective)."""
                repeats = batch // B
                tr, tp = cached_subj[agent_idx]
                model.install_subjective_theta(
                    agent_idx, _expand_dim0(tr, repeats), _expand_dim0(tp, repeats)
                )

            install_subj = _install_subjective

        # [pkg-08 spec 03 §4.2] Resolve per-call eval-mode overrides. ``None``
        # → fall back to the training-time flag stored on ``self``.
        active_use_crn = self.use_crn if eval_use_crn is None else bool(eval_use_crn)
        active_randomize_order = (
            self.randomize_order if eval_randomize_order is None else bool(eval_randomize_order)
        )

        spa = S // A          # scenarios per candidate action
        M = A * spa           # trajectories per batch element per agent

        # Seedable torch generator (derived from crn_rng) -> reproducible sampling.
        gen = torch.Generator(device=device)
        gen.manual_seed(int(self.crn_rng.integers(0, 2**31 - 1)))

        # Coordinate-descent agent ordering (randomize_order=False -> fixed order).
        if active_randomize_order:
            agent_order = self.crn_rng.permutation(N).tolist()
        else:
            agent_order = list(range(N))

        pi_mve = torch.zeros(B, N, A, device=device)
        diag_returns = torch.zeros(B, N, A, device=device)   # probe: expected return per action
        diag_qnorm = torch.zeros(B, N, A, device=device)     # probe: z-scored return per action
        diag_qstd = torch.zeros(B, N, device=device)         # probe: raw candidate-return std
        optimised = set()

        for j in agent_order:
            # ── Phase 1: pre-sample other agents' step-0 actions (CRN) ─────
            s_scenarios = root_s.repeat_interleave(spa, dim=0)  # (B*spa, latent)
            B_spa = B * spa
            if is_hyper:
                row_scenarios = _expand_dim0(row_t, spa)        # (B*spa, N, N-1)
                belief_scenarios = _expand_dim0(belief_t, spa)  # (B*spa, N, |G|)
            else:
                row_scenarios = belief_scenarios = None

            step0_actions_per_scenario = {}  # agent_i -> (B*spa,)
            for i in range(N):
                if i == j:
                    continue
                if i in optimised:
                    probs_i = pi_mve[:, i].repeat_interleave(spa, dim=0)  # (B*spa, A)
                    step0_actions_per_scenario[i] = _multinomial_sample(probs_i, gen)
                else:
                    step0_actions_per_scenario[i] = self._sample_policy_action(
                        model, s_scenarios, i, B_spa, device, is_hyper,
                        row_scenarios, belief_scenarios, gen, install_subj=install_subj,
                    )

            # ── Phase 2: expand to (B*M,) = (B*spa*A,) — scenario outer, candidate inner
            s_exp = s_scenarios.repeat_interleave(A, dim=0)  # (B*M, latent)
            BM = B * M
            if is_hyper:
                row_exp = _expand_dim0(row_scenarios, A)        # (B*M, N, N-1)
                belief_exp = _expand_dim0(belief_scenarios, A)  # (B*M, N, |G|)
            else:
                row_exp = belief_exp = None

            cum_return_j = torch.zeros(BM, device=device)
            discount = 1.0
            curr_s = s_exp

            # Agent j's candidate first-actions: [0..A-1, 0..A-1, ...]
            first_action_j = torch.arange(A, device=device).repeat(B * spa)  # (B*M,)

            # Replicate other agents' step-0 actions across the A candidates.
            step0_actions_expanded = {}
            for i, a_scenario in step0_actions_per_scenario.items():
                step0_actions_expanded[i] = a_scenario.repeat_interleave(A, dim=0)  # (B*M,)

            # ── Phase 3: rollout K steps ───────────────────────────────────
            for step in range(K):
                all_actions = []
                for i in range(N):
                    if step == 0 and i == j:
                        a_i = first_action_j                       # enumerated candidate
                    elif step == 0 and i in step0_actions_expanded and active_use_crn:
                        a_i = step0_actions_expanded[i]            # CRN: shared per scenario
                    else:
                        # step>0, or CRN disabled: independent sampling at BM granularity
                        a_i = self._sample_policy_action(
                            model, curr_s, i, BM, device, is_hyper,
                            row_exp, belief_exp, gen, install_subj=install_subj,
                        )
                    all_actions.append(a_i)

                joint_actions = torch.stack(all_actions, dim=-1)        # (B*M, N)
                action_onehot = actions_to_one_hot(joint_actions, A)    # (B*M, N*A)

                # Objective state transition (v5: plain shared module).
                s_next = model.transition(curr_s, action_onehot)

                # Subjective reward for agent j.
                if is_hyper:
                    if theta_cache_on:
                        install_subj(j, BM)
                    else:
                        _set_subjective(model, j, row_exp, belief_exp)
                    r_j_scaled = model.predict_reward(curr_s, action_onehot)
                else:
                    id_j = torch.full((BM,), j, dtype=torch.long, device=device)
                    id_emb_j = model.get_id_emb(id_j)
                    r_j_scaled = model.predict_reward(curr_s, action_onehot, id_emb_j)

                r_j = inverse_scalar_transform(r_j_scaled).squeeze(-1)  # (B*M,)
                cum_return_j = cum_return_j + discount * r_j

                curr_s = s_next
                discount *= gamma

            # ── Phase 3b: terminal value for agent j ───────────────────────
            if is_hyper:
                if theta_cache_on:
                    install_subj(j, BM)
                else:
                    _set_subjective(model, j, row_exp, belief_exp)
                _, v_j_scaled = model.predict(curr_s)
            else:
                id_j = torch.full((BM,), j, dtype=torch.long, device=device)
                id_emb_j = model.get_id_emb(id_j)
                _, v_j_scaled = model.predict(curr_s, id_emb_j)

            v_j = inverse_scalar_transform(v_j_scaled).squeeze(-1)  # (B*M,)
            cum_return_j = cum_return_j + discount * v_j

            # ── Phase 4: aggregate with CRN layout (B*M,) -> (B, spa, A) -> (B, A)
            returns_per_action = cum_return_j.view(B, spa, A).mean(dim=1)
            q_mean = returns_per_action.mean(dim=-1, keepdim=True)
            q_std_raw = returns_per_action.std(dim=-1, keepdim=True)     # (B, 1) pre-floor
            q_normalized = (returns_per_action - q_mean) / (q_std_raw + 1e-8)
            pi_j = F.softmax(q_normalized / temperature, dim=-1)
            # [v4-opt 2026-06] noise guard: when the A candidates' returns are nearly
            # equal, the z-score amplifies spa-scenario sampling noise to unit scale
            # and softmax emits a confident-but-arbitrary target. Below the floor the
            # honest target is "no information" = uniform.
            if self.mve_qstd_floor > 0.0:
                noise_rows = q_std_raw < self.mve_qstd_floor              # (B, 1)
                pi_j = torch.where(noise_rows, torch.full_like(pi_j, 1.0 / A), pi_j)
            pi_mve[:, j] = pi_j
            diag_returns[:, j] = returns_per_action
            diag_qnorm[:, j] = q_normalized
            diag_qstd[:, j] = q_std_raw.squeeze(-1)

            optimised.add(j)

        if return_diagnostics:
            diag_qgap = diag_returns.max(dim=-1).values - diag_returns.min(dim=-1).values
            if self.mve_qstd_floor > 0.0:
                uniform_frac = (diag_qstd < self.mve_qstd_floor).float().mean()
            else:
                uniform_frac = torch.zeros((), device=device)
            return pi_mve, {
                "returns_per_action": diag_returns,
                "q_normalized": diag_qnorm,
                "q_std": diag_qstd,            # (B, N) raw (pre-floor)
                "q_gap": diag_qgap,            # (B, N) max-min candidate return spread
                "uniform_frac": uniform_frac,  # scalar in [0, 1]
            }
        return pi_mve  # (B, N, A)
