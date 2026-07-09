"""TransitionNet — v5 plain shared state-transition network (Pkg-09).

v5 replaces the hypernet-generated ``FunctionalStateTransNet``: with c_t
removed, nothing objective varies between contexts, so the transition
function is a single weight-shared SGD ``nn.Module``. Perspective invariance
(one physical model for all agents — the objective half of the Harsanyi
split) is carried by weight sharing itself; hypernet generation remains only
on the subjective path (reward / prediction heads).

Architecture (mirrors the FULL functional net shape, ordinary parameters):

    [s, A_joint] -> FC1 -> LN -> ReLU
                  -> FC2 -> LN -> ReLU
                  -> FC3 -> LayerNorm -> delta_s
    s' = s + delta_s  (residual)

Future transfer hook: reintroducing objective conditioning (e.g. physics
parameters for cross-environment transfer) would swap this module back for a
generated functional net — the ``transition(s, a)`` API is unchanged either
way.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyper_mve.utils.utils import orthogonal_init


class TransitionNet(nn.Module):
    """Plain shared transition net: ``(s, A_joint_onehot) -> s'`` with residual."""

    def __init__(self, latent_dim: int, joint_action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        input_dim = latent_dim + joint_action_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)
        self.ln_out = nn.LayerNorm(latent_dim)

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)

    def forward(self, state: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state:         (B, latent_dim)
            action_onehot: (B, joint_action_dim) one-hot N*A flat
        Returns:
            next_state: (B, latent_dim) = state + Δs (residual)
        """
        x = torch.cat([state, action_onehot], dim=-1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        delta_s = self.ln_out(self.fc3(x))
        return state + delta_s
