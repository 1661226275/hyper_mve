"""NoBeliefBaselineModel (pkg-07 spec 03 §7.4).

Information ablation (not parameter ablation): keeps the **full hypernet
skeleton** (``hyper_trans`` + ``hyper_rew`` + ``hyper_pred`` mirroring
``HyperMuZeroModel``) and zeros out the belief tuple **before it enters**
``TriContextEncoder``. The ``set_context_subjective`` -> ``apply_raw``
BeliefGradGating path is still walked so that the pre-5K-grad-zero invariant
(spec 03 §6.3) holds uniformly across all 5 variants.

The conditioning subsystem is structurally isomorphic to ``HyperMuZeroModel``
— the **only** difference is that ``ctx_aug`` is built from a zero-belief
tuple. This preserves the spec 02 §3.2 + spec 07 §10.4 "isomorphic to hyper"
equal-param premise behind Assertion C: the parameter count matches hyper
exactly, so the comparison isolates "belief information path on/off" rather
than "extra params off".
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from hyper_mve.baselines.internal.base import BaselineModel
from hyper_mve.configs import V4Config
from hyper_mve.models import (
    DualHyperNetwork,
    FunctionalPredictionNet,
    FunctionalRewardHead,
    FunctionalStateTransNet,
)


class NoBeliefBaselineModel(BaselineModel):
    """no_belief variant — belief tuple is zeroed before TriContextEncoder.

    Structurally isomorphic to ``HyperMuZeroModel``: full DualHyperNetwork
    (3 hypernets) + 3 functional nets driven by generated theta. The single
    deviation is the ``zero_belief`` substitution in ``_build_conditioning_state``.
    """

    def _build_conditioning_subsystem(self, cfg: V4Config) -> None:
        latent_dim = cfg.model.latent_dim
        joint_action_dim = cfg.env.N * cfg.env.A
        hidden_dim = int(cfg.model.hidden_dim)
        gen_scope = cfg.model.hyper_gen_scope
        lora_rank = cfg.model.lora_fc2_rank if gen_scope == "lora_fc2" else None

        # 3 functional nets — same factory pattern as HyperMuZeroModel:94-96.
        self.state_trans_net = FunctionalStateTransNet(
            latent_dim, joint_action_dim, hidden_dim, gen_scope, lora_rank,
        )
        self.reward_head = FunctionalRewardHead(
            latent_dim, joint_action_dim, hidden_dim, gen_scope, lora_rank,
        )
        self.prediction_net = FunctionalPredictionNet(
            latent_dim, cfg.env.A, hidden_dim, gen_scope, lora_rank,
        )

        # Full DualHyperNetwork (mirrors HyperMuZeroModel:101-118 byte-for-byte).
        self.hyper_net = DualHyperNetwork(
            c_ctx_dim=cfg.model.d_c,
            ctx_aug_dim=cfg.model.d_ctx_aug,
            trans_param_count=self.state_trans_net.generated_param_count,
            rew_param_count=self.reward_head.generated_param_count,
            pred_param_count=self.prediction_net.generated_param_count,
            hidden_dims=cfg.model.hyper_hidden_dims,
            rew_hidden_dims=cfg.model.hyper_rew_hidden_dims,
            trans_output_scale_init=cfg.model.trans_output_scale_init,
            rew_output_scale_init=cfg.model.rew_output_scale_init,
            pred_output_scale_init=cfg.model.pred_output_scale_init,
            detach_pred_context=cfg.train.detach_pred_context,
            trans_output_groups=self.state_trans_net.gen_groups,
            rew_output_groups=self.reward_head.gen_groups,
            pred_output_groups=self.prediction_net.gen_groups,
            share_subjective_trunk=cfg.model.share_subjective_trunk,
            hyper_output_rank=cfg.model.hyper_output_rank,
        )

        self._belief_slice: tuple[int, int] = (
            cfg.model.d_c + cfg.model.d_role, cfg.model.d_ctx_aug,
        )
        self._theta_state: Optional[torch.Tensor] = None
        self._theta_rew: Optional[torch.Tensor] = None
        self._theta_pred: Optional[torch.Tensor] = None

    def _build_conditioning_state(self, agent_id, cap_i, belief_gated):
        # Run the gating path *and* then zero the belief tuple before it
        # reaches TriContextEncoder. This is the "information consumption,
        # not parameter" distinction (pkg-07 spec 03 §7.4 lines 358-368).
        c_hat_g, z_hat_g = belief_gated
        zero_c = torch.zeros_like(c_hat_g)
        zero_z = torch.zeros_like(z_hat_g)

        tce = self.tri_context_encoder
        c_t = self._ctx_obj if self._ctx_obj is not None else torch.zeros(
            cap_i.shape[0], device=cap_i.device,
        )
        c_ctx = tce.forward_c_ctx_only(c_t)

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
        # Belief sub-encoder fed with zero tensors — the structural test
        # ``test_no_belief_zeros_belief_path`` asserts that varying the input
        # belief leaves the output unchanged.
        c_hat_proj = be.proj_c_hat(zero_c.reshape(B, 1, 1))
        z_pooled = be.z_pool(zero_z.unsqueeze(1))
        z_pooled_proj = be.proj_z_pooled(z_pooled)
        belief_vec = torch.cat([c_hat_proj, z_pooled_proj], dim=-1)
        belief_vec = tce.ln_belief(belief_vec).squeeze(1)

        ctx_aug = torch.cat([c_ctx, role, belief_vec], dim=-1)
        ctx_aug = self.grad_gating.apply_ctx(
            ctx_aug, self._step, belief_slice=self._belief_slice,
        )

        # Generate theta_state (objective) from c_ctx alone, theta_rew/pred
        # from the (zero-belief) ctx_aug.
        self._theta_state = self.hyper_net.forward_trans(c_ctx)
        self._theta_rew, self._theta_pred = self.hyper_net.forward_subjective(ctx_aug)

    def _apply_trans(self, s, action):
        # FunctionalStateTransNet emits s' (with internal residual). Base class
        # expects Δs (it adds another ``s + ``); subtract back to keep the
        # contract. _match_batch tiles theta to the planner-expanded batch
        # (functional_linear does per-sample bmm → batch must match s).
        s_next = self.state_trans_net(s, action, self._match_batch(self._theta_state, s))
        return s_next - s

    def _apply_reward(self, s, action):
        return self.reward_head(s, action, self._match_batch(self._theta_rew, s))

    def _apply_pred(self, s):
        return self.prediction_net(s, self._match_batch(self._theta_pred, s))


__all__ = ["NoBeliefBaselineModel"]
