"""RelationCommonsState — v5 env internal state (Pkg-09).

Mutable dataclass; ``env.step`` updates fields in place. Compared to the v4
``ResourceCommonsState``: capability / type / hotspot / c_t machinery is
gone; the regime ``g`` and its relationship matrix ``W`` are the only
latent identity state.

Field categories:

- **Dynamic** (mutate every step): agent positions, resource stocks,
  cumulative_harvests, steps_since_harvest, last_actions, step_idx, done —
  plus ``g`` / ``W`` / ``rows`` when ``regime_switch_prob > 0`` (research
  point 2; static within episode under point 1).
- **Episode-fixed**: resource_positions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RelationCommonsState:
    """Mutable container for RelationCommons env state."""

    # --- Agent dynamic state ---
    agent_positions: np.ndarray         # (N, 2) int32, (x, y) ∈ [0, L)
    cumulative_harvests: np.ndarray     # (N,)   float32, Σ_t u_i^t
    steps_since_harvest: np.ndarray     # (N,)   int32, since last successful HARVEST
    last_actions: np.ndarray            # (N,)   int64, previous step's actions

    # --- Resource field ---
    resource_positions: np.ndarray      # (K, 2) int32, fixed within episode
    resource_stocks: np.ndarray         # (K,)   float32, q_k ∈ [0, Q_max]

    # --- Relationship regime (Pkg-09) ---
    g: int                              # current regime id ∈ [0, |G|)
    W: np.ndarray                       # (N, N) float32, diagonal = 1
    rows: np.ndarray                    # (N, N-1) float32, diagonal-free rows w_i·

    # --- Timing ---
    step_idx: int
    done: bool

    def copy(self) -> "RelationCommonsState":
        """Return a deep copy (evaluation / debugging snapshots)."""
        return RelationCommonsState(
            agent_positions=self.agent_positions.copy(),
            cumulative_harvests=self.cumulative_harvests.copy(),
            steps_since_harvest=self.steps_since_harvest.copy(),
            last_actions=self.last_actions.copy(),
            resource_positions=self.resource_positions.copy(),
            resource_stocks=self.resource_stocks.copy(),
            g=self.g,
            W=self.W.copy(),
            rows=self.rows.copy(),
            step_idx=self.step_idx,
            done=self.done,
        )
