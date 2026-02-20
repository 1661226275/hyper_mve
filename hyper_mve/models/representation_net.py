"""
Representation Network

Encodes joint observations (concatenation of all agents' obs) into a latent state.
Used as the first stage in all three experiments.

Input:  o_joint (batch_size, joint_obs_dim)  — concatenation of all agents' obs
Output: s       (batch_size, latent_dim)     — objective latent state
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import orthogonal_init


class RepresentationNet(nn.Module):
    """
    Encode joint observation into latent state.

    Input:  o_joint (batch_size, joint_obs_dim)
    Output: s       (batch_size, latent_dim)
    """

    def __init__(self, joint_obs_dim, hidden_dim=256, latent_dim=64):
        super(RepresentationNet, self).__init__()
        self.fc1 = nn.Linear(joint_obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)
        self.ln = nn.LayerNorm(latent_dim)

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)

    def forward(self, joint_obs):
        """
        Args:
            joint_obs: (batch_size, joint_obs_dim)
        Returns:
            latent_state: (batch_size, latent_dim)
        """
        x = F.relu(self.fc1(joint_obs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.ln(x)
        return x


class Projector(nn.Module):
    """
    Non-linear projection head for Consistency Loss (v4.0).

    Maps latent states to a projection space where Cosine Similarity
    is used instead of MSE. Shared by both predicted (s_pred) and
    target (s_target) sides (target side is detached).

    Used by all three experiments (Exp1/2/3) for fair comparison.

    Architecture: 2-layer MLP (latent_dim -> proj_dim -> proj_dim)
    """

    def __init__(self, latent_dim, proj_dim=64):
        super(Projector, self).__init__()
        self.fc1 = nn.Linear(latent_dim, proj_dim)
        self.fc2 = nn.Linear(proj_dim, proj_dim)

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)

    def forward(self, x):
        """
        Args:
            x: (batch_size, latent_dim)
        Returns:
            proj: (batch_size, proj_dim)
        """
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def cosine_similarity_loss(p, z):
    """
    Cosine similarity loss for consistency (v4.0).

    Uses (1 - cos) formulation:
      - Range [0, 2]. Perfect alignment = 0, orthogonal = 1, opposite = 2.
      - Always non-negative, visually consistent with other losses.
      - Mathematically equivalent gradient to -cos (differs only by constant).

    Args:
        p: (B, proj_dim) predicted projection (receives gradient)
        z: (B, proj_dim) target projection (should be detached)
    Returns:
        loss: scalar, in range [0, 2]. Lower = more similar.
    """
    p = F.normalize(p, p=2, dim=-1)
    z = F.normalize(z, p=2, dim=-1)
    return 1.0 - (p * z).sum(dim=-1).mean()


# Backward-compatible alias
negative_cosine_similarity = cosine_similarity_loss
