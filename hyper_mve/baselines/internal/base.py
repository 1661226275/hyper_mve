"""BaselineModel — 7-API base class shared by the 5 internal variants (pkg-07 spec 03 §3).

All 5 internal model classes subclass ``BaselineModel`` and override two hooks:

  * ``_build_conditioning_subsystem(cfg)`` — called from ``__init__``;
    constructs the functional nets, hypernet generator (when present),
    RewardHead, and any per-variant capacity knobs.

  * ``_build_conditioning_state(agent_id, cap_i, belief_gated)`` — called
    from ``set_context_subjective``; computes per-call conditioning state
    (``ctx_aug`` for input-conditioned variants, ``id_onehot`` for ma_muzero,
    generated ``θ`` for no_belief / rewardhead_explicit_type).

  * ``_apply_trans(s, action) -> Δs`` — returns the *delta* state (the base
    class adds the residual ``s + Δs``).

  * ``_apply_reward(s, action) -> (B, 1)`` — subjective reward for the
    last-set ``agent_id``.

  * ``_apply_pred(s) -> (policy_logits, value)`` — subjective policy + value
    for the last-set ``agent_id``.

The base class itself enforces the call-order contract, the Self-Info strict
``cap_i.shape[-1] == 4`` assertion, and the
:class:`hyper_mve.models.BeliefGradGating` apply hook (pkg-04 spec 04).
"""
from __future__ import annotations

import time
from dataclasses import replace
from types import MappingProxyType
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from hyper_mve.baselines.shared_backbones import (
    SHARED_BACKBONE_PREFIXES,
    create_belief_net,
    create_rep_net,
    create_tri_context_encoder,
)
from hyper_mve.configs import V4Config
from hyper_mve.eval.eval_report import EvalReport
from hyper_mve.models import BeliefGradGating


class BaselineModel(nn.Module):
    """7-API base class for the 5 internal baselines (pkg-07 spec 03)."""

    #: pkg-07 spec 02 §3 SB5 — backbone attribute names. Used by
    #: :func:`hyper_mve.baselines.shared_backbones.count_conditioning_params`
    #: to exclude shared-backbone parameters from the fairness count.
    SHARED_BACKBONE_PREFIXES: tuple[str, ...] = SHARED_BACKBONE_PREFIXES

    def __init__(self, cfg: V4Config) -> None:
        super().__init__()
        self.cfg = cfg

        # 3 shared backbones — independent instances (SB3 / pkg-07 design D4).
        self.rep_net = create_rep_net(cfg.env, cfg.model)
        self.belief_net = create_belief_net(cfg.env, cfg.model)
        self.tri_context_encoder = create_tri_context_encoder(cfg.env, cfg.model)

        # BeliefGradGating shared instance contract (C7-INT-GRAD1 / design D6).
        self.grad_gating = BeliefGradGating(
            num_warmup_steps=cfg.train.belief_grad_gating_steps,
        )

        # Per-variant conditioning subsystem (hook).
        self._build_conditioning_subsystem(cfg)

        # Stateful caches (pkg-07 spec 03 §4.1) — five public field names that
        # every variant must keep.
        self._step: int = 0
        self._ctx_obj: Optional[torch.Tensor] = None
        self._agent_id: Optional[int] = None
        self._cap_i: Optional[torch.Tensor] = None
        self._belief_gated: Optional[tuple[torch.Tensor, torch.Tensor]] = None

    # ---------------------------------------------------------------- hooks

    def _build_conditioning_subsystem(self, cfg: V4Config) -> None:
        """Build per-variant functional nets / hypernet / RewardHead.

        Subclasses override.
        """
        raise NotImplementedError("Subclasses must implement _build_conditioning_subsystem")

    def _build_conditioning_state(
        self,
        agent_id: int,
        cap_i: torch.Tensor,
        belief_gated: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Compute per-call conditioning state from the latest subjective inputs.

        Subclasses override. Called from ``set_context_subjective`` *after*
        the Self-Info / shape-strict assertions and the BeliefGradGating
        ``apply``. The default implementation is a no-op; variants that need
        ``ctx_aug`` / ``θ`` / ``id_onehot`` overwrite it.
        """
        # No-op default; concrete variants override.
        return None

    def _build_objective_state(self, c_t: torch.Tensor) -> None:
        """Build agent-agnostic *transition* conditioning from the rule ``c_t``.

        Called from :meth:`set_context_objective`. Transition is objective (rule
        only) — it must be well-defined after ``set_context_objective`` alone,
        because the trainer/planner call ``transition`` once per step before any
        ``set_context_subjective``. The input/id-conditioned variants build a
        c_ctx-only conditioning (role/belief zeroed); the theta-based variants
        generate ``theta_state`` from c_ctx here. Default no-op.
        """
        return None

    def _apply_trans(self, s: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return Δs (the *delta* state). Base class wraps with residual."""
        raise NotImplementedError

    def _apply_reward(self, s: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return per-batch scalar reward (shape ``(B, 1)``)."""
        raise NotImplementedError

    def _apply_pred(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(policy_logits (B, A), value (B, 1))``."""
        raise NotImplementedError

    # ---------------------------------------------------------------- 7-API

    def update_step(self, global_step: int) -> None:  # pkg-04 spec 02 line 169
        self._step = int(global_step)

    def set_context_objective(self, c_t: torch.Tensor) -> None:  # pkg-04 spec 02 line 181
        self._ctx_obj = c_t
        # Build the OBJECTIVE transition conditioning now (mirrors
        # HyperMuZeroModel.set_context_objective regenerating theta_state). The
        # trainer + planner call transition() once per step BEFORE any per-agent
        # set_context_subjective, so the conditioning transition consumes must be
        # established here, not in set_context_subjective.
        self._build_objective_state(c_t)

    def set_context_subjective(
        self,
        agent_id: int,
        cap_i: torch.Tensor,
        belief: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Cache (agent_id, cap_i, belief) for the next ``transition`` /
        ``predict_reward`` / ``predict`` call (pkg-04 spec 02 line 202)."""
        # Self-Info strict (C7-INT-SELF1).
        assert cap_i.dim() == 2 and cap_i.shape[-1] == 4, (
            "Self-Info strict: cap_i must be (B, 4) RAW CapabilityVector; "
            "forbidden to concat opponent oracle types etc. into cap_i "
            f"(C7-INT-SELF1). Got shape {tuple(cap_i.shape)}."
        )
        c_hat, z_hat = belief
        # BeliefGradGating apply (C7-INT-GRAD1 — pre-5K detach, post-5K passthrough).
        c_hat_g, z_hat_g = self.grad_gating.apply_raw(c_hat, z_hat, self._step)
        self._agent_id = int(agent_id)
        self._cap_i = cap_i
        self._belief_gated = (c_hat_g, z_hat_g)
        self._build_conditioning_state(self._agent_id, cap_i, (c_hat_g, z_hat_g))

    def encode(self, obs: torch.Tensor) -> torch.Tensor:  # pkg-04 spec 02 line 277
        return self.rep_net(obs)

    def transition(self, s: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """s + Δs (residual, pkg-04 spec 02 line 284).

        OBJECTIVE: depends on the rule only, so it requires
        ``set_context_objective`` (NOT ``set_context_subjective``) — the
        trainer/planner call it once per step before the per-agent subjective
        loop, exactly as for ``HyperMuZeroModel`` (whose ``theta_state`` is
        objective).
        """
        self._assert_objective_set("transition")
        return s + self._apply_trans(s, action)

    def predict_reward(self, s: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Subjective reward (pkg-04 spec 02 line 298)."""
        self._assert_subjective_set("predict_reward")
        return self._apply_reward(s, action)

    def predict(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Subjective (policy_logits, value) (pkg-04 spec 02 line 318)."""
        self._assert_subjective_set("predict")
        return self._apply_pred(s)

    # ---------------------------------------------------------------- helpers

    def _assert_subjective_set(self, method_name: str) -> None:
        if self._agent_id is None:
            raise AssertionError(
                f"{method_name}() called before set_context_subjective(). "
                "Pkg-04 spec 02 §3.4: must call set_context_objective then "
                "set_context_subjective before predict_reward/predict."
            )

    def _assert_objective_set(self, method_name: str) -> None:
        if self._ctx_obj is None:
            raise AssertionError(
                f"{method_name}() called before set_context_objective(). "
                "Pkg-04 spec 02 §3.4: transition is objective — call "
                "set_context_objective(c_t) first."
            )

    @staticmethod
    def _match_batch(cond: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Tile a per-call conditioning tensor along dim 0 to ``ref``'s batch.

        The MVE planner (``planning/mve_planner.py``) grows the rollout batch by
        ``repeat_interleave`` *between* ``set_context_subjective`` and the
        Phase-3 ``transition``: the subjective context is built at ``B_spa``
        (Phase 1 action sampling) while the rollout state is expanded to
        ``B*M = B_spa * A`` (Phase 2), and the planner only refreshes the
        subjective context again right before ``predict_reward`` / ``predict``.
        So ``transition`` is handed a state at the larger batch while the cached
        conditioning tensor (``_ctx_aug`` / ``θ`` / ``_id_onehot``) still sits at
        the smaller one.

        ``HyperMuZeroModel`` is immune because its ``set_context_objective``
        regenerates ``θ_state`` at the expanded batch (hyper_muzero_model.py:170);
        the five input/θ/id-conditioned baselines defer *all* conditioning to
        ``set_context_subjective``, so they tile the cached tensor here. The
        planner's expansion is a pure ``repeat_interleave`` and the per-scenario
        context rows are identical within a batch element (same c_t / cap /
        belief), so interleave-tiling reproduces the correct per-candidate
        conditioning exactly. Returns ``cond`` untouched on the equal-batch path
        (the training loss and the reward/predict planner calls), so this is a
        no-op everywhere except the planner's transition step.
        """
        b_ref, b_cond = ref.shape[0], cond.shape[0]
        if b_cond == b_ref:
            return cond
        if b_ref > b_cond and b_ref % b_cond == 0:
            return cond.repeat_interleave(b_ref // b_cond, dim=0)
        raise RuntimeError(
            f"BaselineModel conditioning batch {b_cond} is not an integer "
            f"divisor of operand batch {b_ref}; cannot tile to match. This "
            "indicates an unexpected MVE-planner batch layout."
        )

    # ---------------------------------------------------------------- evaluate

    def evaluate(
        self,
        env_fn: Callable[[], Any],
        c_grid: tuple[float, ...],
        episodes: int,
    ) -> EvalReport:
        """Internal-variant ``.evaluate`` (pkg-07 spec 08 §4.1 signature lock).

        For internal variants the unified evaluator (pkg-08 spec 01 §4.2)
        drives the real eval through ``training/evaluation.py:run_eval``.
        This method exists so that the signature contract is satisfied and so
        that callers using a baseline standalone (no unified evaluator) still
        get an :class:`EvalReport` — but the per-c loop here is intentionally
        minimal; the heavy lifting belongs in pkg-08 spec 01.
        """
        # pkg-07 spec 08 §4.1: structural placeholder. The unified evaluator
        # (pkg-08 spec 01 §4.2 internal branch) calls run_eval directly and
        # builds the EvalReport; this method is the spec-08 contract slot for
        # any caller that hits ``.evaluate`` on an internal runner without
        # going through the unified evaluator.
        t0 = time.time()
        empty_per_c: dict[float, float] = {c: 0.0 for c in c_grid}
        empty_int_per_c: dict[float, int] = {c: 0 for c in c_grid}
        empty_bool_per_c: dict[float, bool] = {c: False for c in c_grid}
        segments = tuple(self.cfg.eval.c_segments)
        ratios = tuple(self.cfg.eval.bell_curve_type_ratios)
        return EvalReport(
            variant=type(self).__name__,
            seed=0,
            config_hash="0" * 40,
            eval_mode="planner",
            eval_planner_mode=self.cfg.eval.eval_planner_mode,
            c_visible=bool(self.cfg.env.c_visible),
            return_mean=0.0,
            return_sem=0.0,
            return_zero_shot_seen=0.0,
            return_zero_shot_unseen=0.0,
            return_zero_shot_gap=0.0,
            return_per_c=MappingProxyType(dict(empty_per_c)),
            return_per_c_sem=MappingProxyType(dict(empty_per_c)),
            episodes_per_c=MappingProxyType(dict(empty_int_per_c)),
            return_per_segment=MappingProxyType({seg: 0.0 for seg in segments}),
            return_per_segment_sem=MappingProxyType({seg: 0.0 for seg in segments}),
            return_per_type_ratio=MappingProxyType({r: 0.0 for r in ratios}),
            return_per_type_ratio_sem=MappingProxyType({r: 0.0 for r in ratios}),
            regret_per_c=MappingProxyType(dict(empty_per_c)),
            regret_mean=0.0,
            oracle_ceiling_per_c=MappingProxyType(dict(empty_per_c)),
            oracle_ceiling_cache_hit=MappingProxyType(dict(empty_bool_per_c)),
            planner_prior_return_gap=0.0,
            direct_inference_return_mean=0.0,
            planner_full_return_mean=0.0,
            walltime_seconds=float(time.time() - t0),
            env_steps_evaluated=0,
            episodes_total=0,
            info_gating_strict=True,
            set_context_subjective_oracle_leak=False,
        )


__all__ = ["BaselineModel"]
