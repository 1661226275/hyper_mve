"""ResourceCommonsState — env internal state (Ch3.2 formal tuple).

Pkg-02 spec 01. Mutable dataclass; ``env.step`` updates fields in place.
Use :meth:`copy` when a snapshot is required (e.g. evaluation, debugging).

Field categories:

- **Dynamic** (mutate every step): agent positions, resource stocks,
  cumulative_harvests, steps_since_harvest, c_t, c_history, step_idx, done,
  last_actions.
- **Episode-fixed** (resampled at reset, constant within episode):
  agent_caps, agent_types, hotspot_centers, resource_positions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from hyper_mve.schemas import CapabilityVector


@dataclass
class ResourceCommonsState:
    """Mutable container for ResourceCommons env state (Ch3.2).

    See pkg-02 spec 01 for the mapping between the 14-tuple components and
    these fields. Static components (N, L, T_max, ...) live on the env
    object, not here.
    """

    # --- Agent dynamic state ---
    agent_positions: np.ndarray         # (N, 2) int32, (x, y) ∈ [0, L)
    cumulative_harvests: np.ndarray     # (N,)   float32, Σ_t u_i^t
    steps_since_harvest: np.ndarray     # (N,)   int32, since last successful HARVEST
    last_actions: np.ndarray            # (N,)   int64, previous step's actions

    # --- Resource field ---
    resource_positions: np.ndarray      # (K, 2) int32, fixed within episode
    resource_stocks: np.ndarray         # (K,)   float32, q_k ∈ [0, Q_max]

    # --- Shared context c_t (Ch3.2 component 10) ---
    c_t: float
    c_history: np.ndarray               # (T_max + 1,) float32; index t holds c_t after step t

    # --- Episode-fixed identity / topology ---
    agent_caps: Tuple[CapabilityVector, ...]   # length N (frozen instances)
    agent_types: np.ndarray             # (N,) int8 (AgentType.value)
    hotspot_centers: np.ndarray         # (M, 2) int32, fixed within episode

    # --- Timing ---
    step_idx: int
    done: bool

    def copy(self) -> "ResourceCommonsState":
        """Return a deep copy. ``agent_caps`` is a tuple of frozen dataclasses,
        so it is safely shared by reference.
        """
        return ResourceCommonsState(
            agent_positions=self.agent_positions.copy(),
            cumulative_harvests=self.cumulative_harvests.copy(),
            steps_since_harvest=self.steps_since_harvest.copy(),
            last_actions=self.last_actions.copy(),
            resource_positions=self.resource_positions.copy(),
            resource_stocks=self.resource_stocks.copy(),
            c_t=self.c_t,
            c_history=self.c_history.copy(),
            agent_caps=self.agent_caps,          # frozen → share by reference
            agent_types=self.agent_types.copy(),
            hotspot_centers=self.hotspot_centers.copy(),
            step_idx=self.step_idx,
            done=self.done,
        )
