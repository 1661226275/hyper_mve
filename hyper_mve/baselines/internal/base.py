"""BaselineModel — v5 6-API base class shared by the internal variants (pkg-07 spec 03 §3).

All internal model classes subclass ``BaselineModel`` and override hooks:

  * ``_build_conditioning_subsystem(cfg)`` — called from ``__init__``;
    constructs the nets, hypernet generator (when present), RewardHead, and
    any per-variant capacity knobs.

  * ``_build_conditioning_state(agent_id, row_i, belief_gated)`` — called
    from ``set_context_subjective``; computes per-call conditioning state
    (``ctx_aug`` for input-conditioned variants, ``id_onehot`` for ma_muzero,
    generated ``θ`` for no_belief).

  * ``_apply_trans(s, action) -> Δs`` — returns the *delta* state (the base
    class adds the residual ``s + Δs``). v5: transition is objective and
    UNCONDITIONED (c_t removed) — variants must not consume subjective state
    here.

  * ``_apply_reward(s, action) -> (B, 1)`` — subjective reward for the
    last-set ``agent_id``.

  * ``_apply_pred(s) -> (policy_logits, value)`` — subjective policy + value
    for the last-set ``agent_id``.

The base class itself enforces the Self-Info strict ``row_i.shape ==
(B, N-1)`` assertion and the :class:`hyper_mve.models.BeliefGradGating`
apply hook (pkg-04 spec 04). v5 (Pkg-09): ``set_context_objective`` is
deleted from the API — see the amendment note in ``hyper_muzero_model.py``.
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

        # Stateful caches (pkg-07 spec 03 §4.1, v5 field set).
        self._step: int = 0
        self._agent_id: Optional[int] = None
        self._row_i: Optional[torch.Tensor] = None
        self._belief_gated: Optional[torch.Tensor] = None

    # ---------------------------------------------------------------- hooks

    def _build_conditioning_subsystem(self, cfg: V4Config) -> None:
        """Build per-variant functional nets / hypernet / RewardHead.

        Subclasses override.
        """
        raise NotImplementedError("Subclasses must implement _build_conditioning_subsystem")

    def _build_conditioning_state(
        self,
        agent_id: int,
        row_i: torch.Tensor,
        belief_gated: torch.Tensor,
    ) -> None:
        """Compute per-call conditioning state from the latest subjective inputs.

        Subclasses override. Called from ``set_context_subjective`` *after*
        the Self-Info / shape-strict assertions and the BeliefGradGating
        ``apply``. The default implementation is a no-op; variants that need
        ``ctx_aug`` / ``θ`` / ``id_onehot`` overwrite it.
        """
        # No-op default; concrete variants override.
        return None

    def _apply_trans(self, s: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return Δs (the *delta* state). Base class wraps with residual.

        v5: transition is objective and unconditioned — no context input.
        """
        raise NotImplementedError

    def _apply_reward(self, s: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return per-batch scalar reward (shape ``(B, 1)``)."""
        raise NotImplementedError

    def _apply_pred(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(policy_logits (B, A), value (B, 1))``."""
        raise NotImplementedError

    # ---------------------------------------------------------------- 6-API (v5)

    def update_step(self, global_step: int) -> None:  # pkg-04 spec 02 line 169
        self._step = int(global_step)

    def set_context_subjective(
        self,
        agent_id: int,
        row_i: torch.Tensor,
        belief: torch.Tensor,
    ) -> None:
        """Cache (agent_id, row_i, belief) for the next ``predict_reward`` /
        ``predict`` call (v5 amendment of pkg-04 spec 02 line 202)."""
        # Self-Info strict (C7-INT-SELF1, v5 form).
        N = self.cfg.env.N
        assert row_i.dim() == 2 and row_i.shape[-1] == N - 1, (
            f"Self-Info strict: row_i must be (B, {N - 1}) — agent's OWN "
            "diagonal-free relationship row only (C7-INT-SELF1). "
            f"Got shape {tuple(row_i.shape)}."
        )
        # BeliefGradGating apply (C7-INT-GRAD1 — pre-5K detach, post-5K passthrough).
        g_hat_g = self.grad_gating.apply_raw(belief, self._step)
        self._agent_id = int(agent_id)
        self._row_i = row_i
        self._belief_gated = g_hat_g
        self._build_conditioning_state(self._agent_id, row_i, g_hat_g)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:  # pkg-04 spec 02 line 277
        return self.rep_net(obs)

    def transition(self, s: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """s + Δs (residual, pkg-04 spec 02 line 284).

        v5: OBJECTIVE and unconditioned — callable at any time, no
        set_context_* prerequisite (mirrors HyperMuZeroModel's plain
        TransitionNet).
        """
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
                "v5 API: must call set_context_subjective before "
                "predict_reward/predict."
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
        regime_grid: tuple[int, ...],
        episodes: int,
    ) -> EvalReport:
        """Internal-variant ``.evaluate`` (pkg-07 spec 08 §4.1 signature lock, v5).

        For internal variants the unified evaluator (pkg-08 spec 01 §4.2)
        drives the real eval through ``training/evaluation.py:run_eval``.
        This method exists so that the signature contract is satisfied and so
        that callers using a baseline standalone (no unified evaluator) still
        get an :class:`EvalReport` — the per-regime loop here is intentionally
        a zero-valued placeholder.
        """
        t0 = time.time()
        grid = tuple(int(g) for g in regime_grid)
        return EvalReport(
            variant=type(self).__name__,
            seed=0,
            config_hash="0" * 40,
            eval_mode="planner",
            eval_planner_mode=self.cfg.eval.eval_planner_mode,
            return_mean=0.0,
            return_sem=0.0,
            return_zero_shot_seen=0.0,
            return_zero_shot_unseen=0.0,
            return_zero_shot_gap=0.0,
            return_per_regime=MappingProxyType({g: 0.0 for g in grid}),
            return_per_regime_sem=MappingProxyType({g: 0.0 for g in grid}),
            episodes_per_regime=MappingProxyType({g: 0 for g in grid}),
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
