"""MAMuZeroBaselineModel (pkg-07 spec 03 §7.3).

Vanilla MA-MuZero baseline:
  * **No hypernet** (C7-INT-STRUCT1).
  * **Shared single RewardHead** (C7-INT-STRUCT2) — not a per-agent ModuleList.
  * Conditioning via ``id_onehot ⊕ own_type_onehot`` concatenated onto the
    input. Belief tuple is consumed by ``set_context_subjective`` (to keep
    the BeliefGradGating apply path uniform across all 5 variants) but
    *not* fed into the conditioning state — the failure mode.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from hyper_mve.baselines.internal.base import BaselineModel
from hyper_mve.baselines.internal.input_conditioned import _PredHead, _mlp
from hyper_mve.configs import V4Config


class MAMuZeroBaselineModel(BaselineModel):
    """Vanilla MA-MuZero baseline — single shared RewardHead, no hypernet."""

    def _build_conditioning_subsystem(self, cfg: V4Config) -> None:
        latent_dim = cfg.model.latent_dim
        joint_action_dim = cfg.env.N * cfg.env.A
        hidden_dim = int(cfg.model.hidden_dim)
        # id_onehot dim = N agents + 2 own-type bits.
        self._id_dim = cfg.env.N + 2
        in_trans = latent_dim + joint_action_dim + self._id_dim
        in_reward = latent_dim + joint_action_dim + self._id_dim
        in_pred = latent_dim + self._id_dim
        self.trans_net = _mlp(in_trans, hidden_dim, latent_dim, n_layers=2)
        # Single shared RewardHead (C7-INT-STRUCT2). Not a ModuleList; not
        # per-agent. The structural-marker test asserts ``isinstance(...,
        # nn.Module) and not isinstance(..., nn.ModuleList)``.
        self.reward_head = _mlp(in_reward, hidden_dim, 1, n_layers=2)
        # pred head always constructed (predict() must not AttributeError;
        # see pkg-07 spec 03 §7.3 Edge Cases).
        self._share_pred_head: bool = bool(cfg.baselines.internal_ma_muzero_share_pred_head)
        if not self._share_pred_head:
            # Field reserved for future per-agent pred heads; not implemented
            # in this scaffold (pkg-07 spec 03 §7.3 + pkg-06 spec 04 §4
            # Edge Cases).
            raise NotImplementedError(
                "MAMuZeroBaselineModel(internal_ma_muzero_share_pred_head=False) "
                "is not implemented (pkg-07 spec 03 §7.3 Edge Cases). "
                "Set cfg.baselines.internal_ma_muzero_share_pred_head=True."
            )
        self.pred_net = _PredHead(in_pred, hidden_dim, cfg.env.A, n_layers=2)
        self._id_onehot: Optional[torch.Tensor] = None

    def _build_conditioning_state(self, agent_id, cap_i, belief_gated):
        # belief is consumed by grad_gating.apply_raw (already done in base.
        # set_context_subjective) but NOT fed into id_onehot. ma_muzero's
        # failure mode is "shared RewardHead learns avg gradient across α/β";
        # belief flow is irrelevant to the conditioning.
        del belief_gated  # intentionally unused
        B = cap_i.shape[0]
        device = cap_i.device
        n_agents = self.cfg.env.N
        agent_one_hot = torch.zeros((B, n_agents), dtype=torch.float32, device=device)
        agent_one_hot[:, int(agent_id)] = 1.0
        # own_type derived from type_assignment (Self-Info strict; pkg-07
        # spec 03 §5.3 — *not* from env.info["types"]).
        own_type_idx = int(self.cfg.env.type_assignment[int(agent_id)].value)
        type_one_hot = torch.zeros((B, 2), dtype=torch.float32, device=device)
        type_one_hot[:, own_type_idx] = 1.0
        self._id_onehot = torch.cat([agent_one_hot, type_one_hot], dim=-1)

    def _apply_trans(self, s: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        idh = self._match_batch(self._id_onehot, s)
        return self.trans_net(torch.cat([s, action, idh], dim=-1))

    def _apply_reward(self, s: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        idh = self._match_batch(self._id_onehot, s)
        return self.reward_head(torch.cat([s, action, idh], dim=-1))

    def _apply_pred(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        idh = self._match_batch(self._id_onehot, s)
        return self.pred_net(torch.cat([s, idh], dim=-1))


__all__ = ["MAMuZeroBaselineModel"]
