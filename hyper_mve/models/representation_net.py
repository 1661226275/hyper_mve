"""Representation Network (Pkg-04, v4 cfg-driven).

Encodes the joint observation (all agents' obs) into an objective latent state.
RepNet is objective: shared across all agents, independent of any set_context.

v4 change (relative to v4.7):
    v4.7: RepresentationNet(joint_obs_dim, hidden_dim=256, latent_dim=64)
    v4:   RepresentationNet(cfg) -- derives obs_dim from ObservationLayout.total_dim
          and accepts obs of shape (B, N, obs_dim), flattening to (B, N*obs_dim).

Input:  obs (B, N, obs_dim)  OR  (B, N*obs_dim)
Output: s   (B, latent_dim)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from hyper_mve.schemas import ObservationLayout
from hyper_mve.utils.utils import orthogonal_init


class RepresentationNet(nn.Module):
    """Encode joint observation into objective latent state (Ch4.4).

    obs_dim is derived from ``ObservationLayout.total_dim(N, K)`` (Pkg-02); the
    joint input dim is ``N * obs_dim`` (all agents concatenated).
    """

    def __init__(self, cfg, hidden_dim: int = 256):
        super().__init__()
        self.cfg = cfg
        self.N = cfg.env.N
        # Per-agent observation dim (Pkg-02 ObservationLayout).
        self.obs_dim = ObservationLayout.total_dim(cfg.env.N, cfg.env.K)
        self.joint_obs_dim = self.N * self.obs_dim
        self.latent_dim = cfg.model.latent_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(self.joint_obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.latent_dim)
        self.ln = nn.LayerNorm(self.latent_dim)

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)

    def forward(self, obs):
        """
        Args:
            obs: (B, N, obs_dim) per-agent stacked, or (B, N*obs_dim) pre-flattened.
        Returns:
            latent_state: (B, latent_dim)
        """
        if obs.dim() == 3:
            # (B, N, obs_dim) -> (B, N*obs_dim)
            obs = obs.reshape(obs.shape[0], -1)
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.ln(x)
        return x


class Projector(nn.Module):
    """Non-linear projection head for BYOL consistency loss (v4.0 / Ch5.8.2).

    Maps latent states to a projection space where cosine similarity is used
    instead of MSE. Shared by predicted and (detached) target sides.

    Architecture: 2-layer MLP (latent_dim -> proj_dim -> proj_dim).
    """

    def __init__(self, latent_dim, proj_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, proj_dim)
        self.fc2 = nn.Linear(proj_dim, proj_dim)

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)

    def forward(self, x):
        """
        Args:
            x: (B, latent_dim)
        Returns:
            proj: (B, proj_dim)
        """
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def cosine_similarity_loss(p, z):
    """Cosine similarity loss for consistency (v4.0).

    Uses (1 - cos) formulation, range [0, 2]; lower = more similar.

    Args:
        p: (B, proj_dim) predicted projection (receives gradient)
        z: (B, proj_dim) target projection (should be detached)
    Returns:
        loss: scalar in [0, 2].
    """
    p = F.normalize(p, p=2, dim=-1)
    z = F.normalize(z, p=2, dim=-1)
    return 1.0 - (p * z).sum(dim=-1).mean()


# Backward-compatible alias
negative_cosine_similarity = cosine_similarity_loss
