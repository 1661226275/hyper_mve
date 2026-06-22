"""ExplicitTypeRewardBaselineModel (pkg-07 spec 03 §7.5).

Reward-head ablation:
  * Keeps ``hyper_trans`` + ``hyper_pred`` (pkg-07 spec 03 §7.5 + C7-INT-STRUCT3).
  * **No** ``hyper_rew`` — the RewardHead is replaced by a type-branched
    module whose forward indexes by ``own_type``.

The conditioning failure mode: discrete type branches cannot absorb the
continuous capability heterogeneity (cap is 4-dim continuous; type is 2-dim
discrete). This is the strong Assertion-A counterfactual to ``ma_muzero``'s
weaker "completely shared RewardHead" mode.

The hypernet skeleton is constructed via a thin ``DualHyperNetwork`` that
generates only the ``trans`` + ``pred`` payloads (the ``rew`` payload is set
to size 0; ``hyper_rew`` is intentionally absent). ``_apply_reward`` skips
the hypernet entirely and dispatches the type-branched MLP.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from hyper_mve.baselines.internal.base import BaselineModel
from hyper_mve.baselines.internal.input_conditioned import _mlp
from hyper_mve.configs import V4Config
from hyper_mve.models import (
    FunctionalPredictionNet,
    FunctionalStateTransNet,
)


class _TypeBranchedRewardHead(nn.Module):
    """Discrete type branches: ``n_branch`` parallel MLPs, indexed by branch arg."""

    def __init__(self, n_branch: int, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.n_branch = int(n_branch)
        self.branches = nn.ModuleList([
            _mlp(in_dim, hidden_dim, 1, n_layers=2) for _ in range(self.n_branch)
        ])

    def forward(self, x: torch.Tensor, branch: int) -> torch.Tensor:
        if not (0 <= branch < self.n_branch):
            raise IndexError(f"branch={branch} outside [0, {self.n_branch})")
        return self.branches[branch](x)


class ExplicitTypeRewardBaselineModel(BaselineModel):
    """RewardHead replaced by a per-type branched module; no hyper_rew."""

    def _build_conditioning_subsystem(self, cfg: V4Config) -> None:
        latent_dim = cfg.model.latent_dim
        joint_action_dim = cfg.env.N * cfg.env.A
        hidden_dim = int(cfg.model.hidden_dim)
        gen_scope = cfg.model.hyper_gen_scope
        # LoRA fc2 rank only takes effect for that scope (mirrors hyper_muzero_model.py:93).
        lora_rank = cfg.model.lora_fc2_rank if gen_scope == "lora_fc2" else None

        # Functional nets that consume generator-emitted theta_state / theta_pred.
        self.state_trans_net = FunctionalStateTransNet(
            latent_dim, joint_action_dim, hidden_dim, gen_scope, lora_rank,
        )
        self.prediction_net = FunctionalPredictionNet(
            latent_dim, cfg.env.A, hidden_dim, gen_scope, lora_rank,
        )

        # Two independent HyperNetMLPs. We construct them as top-level
        # attributes ``hyper_trans`` and ``hyper_pred`` so that
        # ``test_explicit_type_no_hyper_rew`` (spec 03 §9.5) finds the
        # ``hyper_trans`` / ``hyper_pred`` substrings in
        # ``model.named_modules()`` while ``hyper_rew`` is genuinely absent.
        from hyper_mve.models import HyperNetMLP

        self.hyper_trans = HyperNetMLP(
            input_dim=cfg.model.d_c,
            output_dim=self.state_trans_net.generated_param_count,
            hidden_dims=cfg.model.hyper_hidden_dims,
            output_scale_init=cfg.model.trans_output_scale_init,
            output_groups=self.state_trans_net.gen_groups,
            output_rank=cfg.model.hyper_output_rank,
        )
        self.hyper_pred = HyperNetMLP(
            input_dim=cfg.model.d_ctx_aug,
            output_dim=self.prediction_net.generated_param_count,
            hidden_dims=cfg.model.hyper_hidden_dims,
            output_scale_init=cfg.model.pred_output_scale_init,
            output_groups=self.prediction_net.gen_groups,
            output_rank=cfg.model.hyper_output_rank,
        )

        # Type-branched RewardHead — replaces hyper_rew + FunctionalRewardHead.
        n_branches = int(cfg.baselines.internal_explicit_type_branches)
        self.reward_head = _TypeBranchedRewardHead(
            n_branches, latent_dim + joint_action_dim, hidden_dim,
        )

        # Pre-cached belief slice for grad gating (mirrors HyperMuZeroModel).
        self._belief_slice: tuple[int, int] = (
            cfg.model.d_c + cfg.model.d_role, cfg.model.d_ctx_aug,
        )
        self._theta_state: Optional[torch.Tensor] = None
        self._theta_pred: Optional[torch.Tensor] = None
        self._own_type: Optional[int] = None

    def _build_conditioning_state(self, agent_id, cap_i, belief_gated):
        """Generate theta_state (objective) + theta_pred (subjective) per call."""
        c_hat_g, z_hat_g = belief_gated
        tce = self.tri_context_encoder
        c_t = self._ctx_obj if self._ctx_obj is not None else torch.zeros(
            cap_i.shape[0], device=cap_i.device,
        )
        c_ctx = tce.forward_c_ctx_only(c_t)

        # ctx_aug assembly (mirrors HyperMuZeroModel.set_context_subjective:210-233).
        B = cap_i.shape[0]
        device = cap_i.device
        agent_ids_one = torch.full((B, 1), int(agent_id), dtype=torch.long, device=device)
        own_type_tensor = torch.full(
            (B, 1),
            int(self.cfg.env.type_assignment[int(agent_id)].value),
            dtype=torch.long, device=device,
        )
        role = tce.role_encoder(agent_ids_one, own_type_tensor, cap_i.unsqueeze(1))
        role = tce.ln_role(role).squeeze(1)
        be = tce.belief_encoder
        c_hat_proj = be.proj_c_hat(c_hat_g.reshape(B, 1, 1))
        z_pooled = be.z_pool(z_hat_g.unsqueeze(1))
        z_pooled_proj = be.proj_z_pooled(z_pooled)
        belief_vec = torch.cat([c_hat_proj, z_pooled_proj], dim=-1)
        belief_vec = tce.ln_belief(belief_vec).squeeze(1)
        ctx_aug = torch.cat([c_ctx, role, belief_vec], dim=-1)

        # ctx_aug belief-segment gating (BeliefGradGating apply_ctx mirrors
        # HyperMuZeroModel:236).
        ctx_aug = self.grad_gating.apply_ctx(
            ctx_aug, self._step, belief_slice=self._belief_slice,
        )

        # Generate theta_state (from c_ctx only) + theta_pred (from ctx_aug).
        # NOTE: pred-side context detach honours cfg.train.detach_pred_context
        # (pkg-04 spec 02 D5) just like HyperMuZeroModel.
        self._theta_state = self.hyper_trans(c_ctx)
        if self.cfg.train.detach_pred_context:
            self._theta_pred = self.hyper_pred(ctx_aug.detach())
        else:
            self._theta_pred = self.hyper_pred(ctx_aug)
        self._own_type = int(self.cfg.env.type_assignment[int(agent_id)].value)

    def _apply_trans(self, s: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # FunctionalStateTransNet returns s' (= s + Δs internally per pkg-04
        # spec 04). Base class adds another residual ``s + ``. To keep the
        # base contract (``transition`` returns ``s + Δs``) we have
        # FunctionalStateTransNet emit Δs only — but the existing v4 net emits
        # s' directly (residual built in). We compute Δs = s' - s here.
        s_next = self.state_trans_net(s, action, self._theta_state)
        return s_next - s

    def _apply_reward(self, s: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # No theta consumed — pure type-branched MLP.
        return self.reward_head(torch.cat([s, action], dim=-1), branch=self._own_type)

    def _apply_pred(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.prediction_net(s, self._theta_pred)


__all__ = ["ExplicitTypeRewardBaselineModel"]
