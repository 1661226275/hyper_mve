"""Five-block v5 observation construction (Pkg-09).

Follows :class:`hyper_mve.schemas.RelationObservationLayout` exactly and
reuses the Pkg-01 padding helpers. Differences from v4: no FOV filtering
(all resource cells in index order; all neighbors visible with
``presence_flag = 1``), ``global`` is ``(time_remaining_ratio,)`` only, and
the ``capability``/``type`` blocks are replaced by the agent's **own**
relationship row ``w_i·`` (Self-Info: others' rows are never observed).
"""
from __future__ import annotations

import numpy as np

from hyper_mve.schemas import (
    RelationObservationLayout,
    pad_neighbor_block,
    pad_resource_block,
)
from hyper_mve.schemas._constants import Q_MAX

from .dynamics import ETA
from .state import RelationCommonsState

_ACTION_ONEHOT_DIM = RelationObservationLayout.NEIGHBOR_ACTION_DIM


def build_observation(
    state: RelationCommonsState,
    agent_id: int,
    L: int,
    T_max: int,
    K: int,
) -> np.ndarray:
    """One agent's flattened observation, ``(RelationObservationLayout.total_dim(N, K),)``."""
    N = state.agent_positions.shape[0]
    blocks: list[np.ndarray] = [
        _self_block(state, agent_id, L, T_max),
        _resource_block(state, agent_id, K),
        _neighbor_block(state, agent_id, N),
        _global_block(state, T_max),
        state.rows[agent_id].astype(np.float32, copy=False),
    ]
    obs = np.concatenate(blocks).astype(np.float32, copy=False)
    expected = RelationObservationLayout.total_dim(N, K)
    if obs.shape != (expected,):
        raise AssertionError(f"obs dim {obs.shape} != ({expected},)")
    return obs


def build_joint_observation(
    state: RelationCommonsState,
    L: int,
    T_max: int,
    K: int,
) -> np.ndarray:
    """All-agent joint observation, ``(N, obs_dim)`` float32."""
    N = state.agent_positions.shape[0]
    obs_dim = RelationObservationLayout.total_dim(N, K)
    joint = np.zeros((N, obs_dim), dtype=np.float32)
    for i in range(N):
        joint[i] = build_observation(state, i, L, T_max, K)
    return joint


# --------------------------------------------------------------- private blocks


def _self_block(
    state: RelationCommonsState,
    agent_id: int,
    L: int,
    T_max: int,
) -> np.ndarray:
    """``(4,)``: ``[x_norm, y_norm, cum_harvest_norm, steps_since_harvest_norm]``.

    Cumulative harvest is normalized by the episode ceiling ``T_max · η``
    (η = 1 homogeneous) — the v4 ζ carry-capacity normalizer is gone.
    """
    pos = state.agent_positions[agent_id]
    return np.array(
        [
            float(pos[0]) / max(L - 1, 1),
            float(pos[1]) / max(L - 1, 1),
            float(state.cumulative_harvests[agent_id]) / max(T_max * ETA, 1.0),
            float(state.steps_since_harvest[agent_id]) / max(T_max, 1),
        ],
        dtype=np.float32,
    )


def _resource_block(
    state: RelationCommonsState,
    agent_id: int,
    K: int,
) -> np.ndarray:
    """``(K * 3,)``: **all** cells ``(rel_dx, rel_dy, q_norm)`` in index order."""
    pos = state.agent_positions[agent_id]
    cells = [
        (
            float(state.resource_positions[k][0] - pos[0]),
            float(state.resource_positions[k][1] - pos[1]),
            float(state.resource_stocks[k]) / float(Q_MAX),
        )
        for k in range(K)
    ]
    return pad_resource_block(cells, K)


def _neighbor_block(
    state: RelationCommonsState,
    agent_id: int,
    N: int,
) -> np.ndarray:
    """``((N-1) * 9,)``: all other agents, ascending ``j`` skipping self,
    each ``(rel_dx, rel_dy, last_action_onehot[6], presence_flag=1)``."""
    pos = state.agent_positions[agent_id]
    visible: list[tuple[float, float, np.ndarray, bool]] = []
    for j in range(N):
        if j == agent_id:
            continue
        rel = state.agent_positions[j] - pos
        act_oh = np.zeros(_ACTION_ONEHOT_DIM, dtype=np.float32)
        last_a = int(state.last_actions[j])
        if 0 <= last_a < _ACTION_ONEHOT_DIM:
            act_oh[last_a] = 1.0
        visible.append((float(rel[0]), float(rel[1]), act_oh, True))
    return pad_neighbor_block(visible, N)


def _global_block(
    state: RelationCommonsState,
    T_max: int,
) -> np.ndarray:
    """``(1,)``: ``(time_remaining_ratio,)``."""
    time_remaining = float(T_max - state.step_idx) / max(T_max, 1)
    return np.array([time_remaining], dtype=np.float32)
