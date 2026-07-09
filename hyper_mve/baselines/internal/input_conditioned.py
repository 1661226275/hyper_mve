"""input_wide + input_deep variants (pkg-07 spec 03 §7.1 + §7.2).

Both variants have **no DualHyperNetwork**; conditioning is delivered by
concatenating ``ctx_aug`` (from :class:`TriContextEncoder`) onto the input of
the functional nets.

  * ``InputWideBaselineModel`` — fixed-depth MLPs at ``cfg.baselines.internal_wide_hidden_dim``
    width (knob for the spec 07 5%/10% equal-param sweep).
  * ``InputDeepBaselineModel`` — width matches hyper (``cfg.model.hidden_dim``)
    but layer count is ``cfg.baselines.internal_deep_layers``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from hyper_mve.baselines.internal.base import BaselineModel
from hyper_mve.configs import V4Config


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int) -> nn.Sequential:
    """Build a fully-connected MLP with ``n_layers`` hidden layers + 1 output."""
    if n_layers <= 0:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class _PredHead(nn.Module):
    """Two-headed predictor: shared trunk -> (policy_logits, value)."""

    def __init__(self, in_dim: int, hidden_dim: int, n_actions: int, n_layers: int) -> None:
        super().__init__()
        self.trunk = _mlp(in_dim, hidden_dim, hidden_dim, n_layers)
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h)


class _InputConditionedBase(BaselineModel):
    """Common scaffolding for input_wide + input_deep.

    Concrete subclasses set ``_hidden_dim`` and ``_n_layers``.
    """

    _hidden_dim: int  # set by subclass
    _n_layers: int    # set by subclass

    def _build_conditioning_subsystem(self, cfg: V4Config) -> None:
        latent_dim = cfg.model.latent_dim
        joint_action_dim = cfg.env.N * cfg.env.A
        ctx_dim = cfg.model.d_ctx_aug
        # v5: transition is objective and unconditioned (c_t removed) — no ctx
        # concat on the transition input.
        in_trans = latent_dim + joint_action_dim
        in_reward = latent_dim + joint_action_dim + ctx_dim
        in_pred = latent_dim + ctx_dim
        self.trans_net = _mlp(in_trans, self._hidden_dim, latent_dim, self._n_layers)
        self.reward_head = _mlp(in_reward, self._hidden_dim, 1, self._n_layers)
        self.pred_net = _PredHead(in_pred, self._hidden_dim, cfg.env.A, self._n_layers)
        self._ctx_aug: Optional[torch.Tensor] = None       # subjective (reward/pred)

    def _build_conditioning_state(self, agent_id, row_i, belief_gated):
        # Build ctx_aug via TriContextEncoder's sub-encoders (single-agent path
        # — same shape-circumvention HyperMuZeroModel uses).
        tce = self.tri_context_encoder
        B = row_i.shape[0]
        device = row_i.device
        agent_ids_one = torch.full((B, 1), int(agent_id), dtype=torch.long, device=device)
        role = tce.role_encoder(agent_ids_one, row_i.unsqueeze(1))
        role = tce.ln_role(role).squeeze(1)
        belief_vec = tce.belief_encoder(belief_gated.unsqueeze(1))
        belief_vec = tce.ln_belief(belief_vec).squeeze(1)
        self._ctx_aug = torch.cat([role, belief_vec], dim=-1)

    def _apply_trans(self, s: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.trans_net(torch.cat([s, action], dim=-1))

    def _apply_reward(self, s: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        ctx = self._match_batch(self._ctx_aug, s)
        return self.reward_head(torch.cat([s, action, ctx], dim=-1))

    def _apply_pred(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ctx = self._match_batch(self._ctx_aug, s)
        return self.pred_net(torch.cat([s, ctx], dim=-1))


class InputWideBaselineModel(_InputConditionedBase):
    """pkg-07 spec 03 §7.1 — no hypernet, concat ``ctx_aug`` to input, widened hidden."""

    def _build_conditioning_subsystem(self, cfg: V4Config) -> None:
        self._hidden_dim = int(cfg.baselines.internal_wide_hidden_dim)
        self._n_layers = 2
        super()._build_conditioning_subsystem(cfg)


class InputDeepBaselineModel(_InputConditionedBase):
    """pkg-07 spec 03 §7.2 — no hypernet, concat ``ctx_aug`` to input, deepened layer count."""

    def _build_conditioning_subsystem(self, cfg: V4Config) -> None:
        self._hidden_dim = int(cfg.model.hidden_dim)
        self._n_layers = int(cfg.baselines.internal_deep_layers)
        super()._build_conditioning_subsystem(cfg)


__all__ = ["InputWideBaselineModel", "InputDeepBaselineModel"]
