"""Gym spaces constructors for RelationCommons (Pkg-09)."""
from __future__ import annotations

import gym
import numpy as np

from hyper_mve.schemas import RelationObservationLayout


def make_observation_space(N: int, K: int) -> gym.spaces.Box:
    """Joint observation: ``(N, RelationObservationLayout.total_dim(N, K))`` float32."""
    obs_dim = RelationObservationLayout.total_dim(N, K)
    return gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(N, obs_dim),
        dtype=np.float32,
    )


def make_action_space(N: int, A: int = 6) -> gym.spaces.MultiDiscrete:
    """Joint action: ``MultiDiscrete([A] * N)``.

    Encoding (fixed): 0=NOOP, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=HARVEST.
    """
    return gym.spaces.MultiDiscrete([A] * N)
