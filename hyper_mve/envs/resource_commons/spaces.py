"""Gym spaces constructors for ResourceCommons (Pkg-02 spec 08 §2.4)."""
from __future__ import annotations

import gym
import numpy as np

from hyper_mve.schemas import ObservationLayout


def make_observation_space(N: int, K: int) -> gym.spaces.Box:
    """Joint observation: ``(N, ObservationLayout.total_dim(N, K))`` float32.

    Bounds are ``(-inf, +inf)`` because some block fields (relative deltas,
    normalised counters) can take any real value within the episode.
    """
    obs_dim = ObservationLayout.total_dim(N, K)
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
