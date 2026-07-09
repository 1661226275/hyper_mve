"""NoBeliefBaselineModel (pkg-07 spec 03 §7.4, v5 form).

Information ablation (not parameter ablation): keeps the **full v5 hyper
skeleton** (plain shared TransitionNet + ``hyper_rew`` + ``hyper_pred``
mirroring ``HyperMuZeroModel``) and zeros out the regime posterior **before
it enters** the belief encoder. The ``set_context_subjective`` ->
``apply_raw`` BeliefGradGating path is still walked so the pre-5K-grad-zero
invariant (spec 03 §6.3) holds uniformly across the internal variants.

The conditioning subsystem is structurally isomorphic to
``HyperMuZeroModel`` — the **only** difference is that ``ctx_aug`` is built
from a zero posterior. Under the v5 hidden-regime design this is the
headline "no theory-of-mind" ablation: the model keeps its own row (public
Self-Info) but can never infer the opponents' side of the regime.
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
    TransitionNet,
)


class NoBeliefBaselineModel(BaselineModel):
    """no_belief variant — regime posterior is zeroed before the belief encoder.

    Structurally isomorphic to ``HyperMuZeroModel`` (v5): plain shared
    TransitionNet + subjective DualHyperNetwork (hyper_rew/hyper_pred) + 2
    functional nets driven by generated theta. The single deviation is the
    ``zero_belief`` substitution in ``_build_conditioning_state``.
    """

    def _build_conditioning_subsystem(self, cfg: V4Config) -> None:
        latent_dim = cfg.model.latent_dim
        joint_action_dim = cfg.env.N * cfg.env.A
        hidden_dim = int(cfg.model.hidden_dim)
        gen_scope = cfg.model.hyper_gen_scope
        lora_rank = cfg.model.lora_fc2_rank if gen_scope == "lora_fc2" else None

        # Same factory pattern as HyperMuZeroModel (v5).
        self.state_trans_net = TransitionNet(latent_dim, joint_action_dim, hidden_dim)
        self.reward_head = FunctionalRewardHead(
            latent_dim, joint_action_dim, hidden_dim, gen_scope, lora_rank,
        )
        self.prediction_net = FunctionalPredictionNet(
            latent_dim, cfg.env.A, hidden_dim, gen_scope, lora_rank,
        )

        # Subjective DualHyperNetwork (mirrors HyperMuZeroModel byte-for-byte).
        self.hyper_net = DualHyperNetwork(
            ctx_aug_dim=cfg.model.d_ctx_aug,
            rew_param_count=self.reward_head.generated_param_count,
            pred_param_count=self.prediction_net.generated_param_count,
            hidden_dims=cfg.model.hyper_hidden_dims,
            rew_hidden_dims=cfg.model.hyper_rew_hidden_dims,
            rew_output_scale_init=cfg.model.rew_output_scale_init,
            pred_output_scale_init=cfg.model.pred_output_scale_init,
            detach_pred_context=cfg.train.detach_pred_context,
            rew_output_groups=self.reward_head.gen_groups,
            pred_output_groups=self.prediction_net.gen_groups,
            share_subjective_trunk=cfg.model.share_subjective_trunk,
            hyper_output_rank=cfg.model.hyper_output_rank,
        )

        self._belief_slice: tuple[int, int] = (
            cfg.model.d_role, cfg.model.d_ctx_aug,
        )
        self._theta_rew: Optional[torch.Tensor] = None
        self._theta_pred: Optional[torch.Tensor] = None

    def _build_conditioning_state(self, agent_id, row_i, belief_gated):
        # Run the gating path *and* then zero the posterior before it reaches
        # the belief encoder. This is the "information consumption, not
        # parameter" distinction (pkg-07 spec 03 §7.4).
        zero_g = torch.zeros_like(belief_gated)

        tce = self.tri_context_encoder
        B = row_i.shape[0]
        device = row_i.device
        agent_ids_one = torch.full((B, 1), int(agent_id), dtype=torch.long, device=device)
        role = tce.role_encoder(agent_ids_one, row_i.unsqueeze(1))
        role = tce.ln_role(role).squeeze(1)
        # Belief sub-encoder fed with a zero tensor — the structural test
        # asserts that varying the input posterior leaves the output unchanged.
        belief_vec = tce.belief_encoder(zero_g.unsqueeze(1))
        belief_vec = tce.ln_belief(belief_vec).squeeze(1)

        ctx_aug = torch.cat([role, belief_vec], dim=-1)
        ctx_aug = self.grad_gating.apply_ctx(
            ctx_aug, self._step, belief_slice=self._belief_slice,
        )

        self._theta_rew, self._theta_pred = self.hyper_net.forward_subjective(ctx_aug)

    def _apply_trans(self, s, action):
        # TransitionNet emits s' (with internal residual). Base class expects
        # Δs (it adds another ``s + ``); subtract back to keep the contract.
        return self.state_trans_net(s, action) - s

    def _apply_reward(self, s, action):
        return self.reward_head(s, action, self._match_batch(self._theta_rew, s))

    def _apply_pred(self, s):
        return self.prediction_net(s, self._match_batch(self._theta_pred, s))


__all__ = ["NoBeliefBaselineModel"]
