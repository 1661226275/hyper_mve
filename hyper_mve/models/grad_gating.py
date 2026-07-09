"""Belief gradient gating helper (Pkg-04 spec 04, defence line 5, Ch4.6.5; v5 form).

For the first ``num_warmup_steps`` steps (default 5000) the main-task loss is
cut off from BeliefNet / BeliefEncoder gradients, so both train only from the
independent L_belief loss (Pkg-09 belief_loss), avoiding the
chicken-and-egg noise feedback loop.

Two-layer detach (review revision 3, v5 shapes):
    cut 1 (apply_raw): raw regime posterior ``g_hat`` -> blocks main-loss
        backprop into the BeliefNet GRU/head.
    cut 2 (apply_ctx): the belief sub-segment of ctx_aug (v5: [32:64]) ->
        blocks main-loss backprop into the BeliefEncoder projection.

The independent L_belief path is unaffected by gating (the trainer backprops
belief_net parameters directly). D3: explicit ``.detach()`` rather than
nn.Hook (readable/debuggable, maps directly to graph semantics).
"""
from __future__ import annotations

import torch


class BeliefGradGating:
    """Belief gradient gating helper (no learnable params; model-internal)."""

    def __init__(self, num_warmup_steps: int = 5000):
        """
        Args:
            num_warmup_steps: number of initial steps to cut belief gradients.
                Read from cfg.train.belief_grad_gating_steps (default 5000).
        """
        self.num_warmup_steps = num_warmup_steps

    def apply_raw(
        self,
        g_hat: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        """Cut 1: raw regime posterior -> blocks main loss into BeliefNet.

        Args:
            g_hat: (B, |G|) or (B, N, |G|) softmax regime posterior
            step:  current global_step (from ``update_step``)
        Returns:
            g_hat detached when ``step < num_warmup_steps``, else passthrough.
        """
        if step < self.num_warmup_steps:
            return g_hat.detach()
        return g_hat

    def apply_ctx(
        self,
        ctx_aug: torch.Tensor,
        step: int,
        belief_slice: tuple[int, int] = (32, 64),
    ) -> torch.Tensor:
        """Cut 2: ctx_aug belief sub-segment -> blocks main loss into BeliefEncoder.

        v5 ctx_aug structure: [role (0:32), belief_vec (32:64)].
        Only the belief segment is detached; the role path keeps gradients.

        Args:
            ctx_aug:      (..., 64) ctx_aug tensor
            step:         current global_step
            belief_slice: (start, end), default (32, 64) = belief_vec segment
        Returns:
            ctx_aug_gated: same shape; belief segment detached during warmup.
        """
        if step >= self.num_warmup_steps:
            return ctx_aug

        start, end = belief_slice
        return torch.cat([
            ctx_aug[..., :start],
            ctx_aug[..., start:end].detach(),
            ctx_aug[..., end:],
        ], dim=-1)

    # Legacy call-name compatibility (``.apply`` == ``apply_raw``).
    apply = apply_raw

    @property
    def is_gating_active(self) -> bool:
        """Debug helper; whether gating applies is decided by ``apply(step)``."""
        return True
