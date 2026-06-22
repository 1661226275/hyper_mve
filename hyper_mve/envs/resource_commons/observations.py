"""Six-block observation construction — Ch3.7 + Pkg-02 spec 04.

Strictly follows :class:`hyper_mve.schemas.ObservationLayout` dimensions and
reuses :func:`pad_resource_block` / :func:`pad_neighbor_block` from Pkg-01
(no duplicated padding logic).

**Self-Info principle** (Ch3.7.4 + Ch4.2.2):

- ``type`` block: own one-hot only, 2 dims (not 2N).
- ``capability`` block: own 4-tuple only, not other agents'.
- ``neighbor`` block: relative positions + last action + presence flag,
  no type / capability of others.

Opponent types are inferred by Pkg-03 BeliefNet (Oracle-supervised at train
time via ``info["types"]``).

[pkg-08 spec 02 §5.1 / spec 08 §6.1 patch A.1] ``build_joint_observation``
accepts an optional ``env_cfg`` kwarg; when ``env_cfg.c_visible is False``
the c_t slot in each agent's ``global`` block is overwritten with
``env_cfg.c_hidden_constant``. ``info["c_true"]`` is unaffected (Oracle
field). The signature default ``None`` preserves backward compatibility for
any caller that does not yet thread ``env_cfg``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from hyper_mve.configs.env_config import EnvConfig
from hyper_mve.schemas import (
    AgentType,
    ObservationLayout,
    one_hot as type_one_hot,
    pad_neighbor_block,
    pad_resource_block,
)

from .state import ResourceCommonsState


# Action one-hot dim (NOOP, UP, DOWN, LEFT, RIGHT, HARVEST = 6 slots).
_ACTION_ONEHOT_DIM = ObservationLayout.NEIGHBOR_ACTION_DIM


def build_observation(
    state: ResourceCommonsState,
    agent_id: int,
    L: int,
    T_max: int,
    K: int,
) -> np.ndarray:
    """Construct one agent's flattened observation (Ch3.7 six-block layout).

    Returns:
        ``np.ndarray`` shape ``(ObservationLayout.total_dim(N, K),)`` float32.
    """
    N = state.agent_positions.shape[0]
    blocks: list[np.ndarray] = [
        _self_block(state, agent_id, L, T_max),
        _resource_block(state, agent_id, K),
        _neighbor_block(state, agent_id, N),
        _global_block(state, T_max),
        state.agent_caps[agent_id].to_array(dtype=np.float32),
        type_one_hot(
            AgentType(int(state.agent_types[agent_id])), dtype=np.float32,
        ),
    ]
    obs = np.concatenate(blocks).astype(np.float32, copy=False)
    expected = ObservationLayout.total_dim(N, K)
    if obs.shape != (expected,):
        raise AssertionError(f"obs dim {obs.shape} != ({expected},)")
    return obs


def build_joint_observation(
    state: ResourceCommonsState,
    L: int,
    T_max: int,
    K: int,
    env_cfg: Optional[EnvConfig] = None,
) -> np.ndarray:
    """Construct all-agent joint observation, ``(N, obs_dim)`` float32.

    [pkg-08 spec 02 §5.1] When ``env_cfg.c_visible is False``, the ``c_t``
    slot in each agent's ``global`` block is overwritten with
    ``env_cfg.c_hidden_constant``; ``info["c_true"]`` (Oracle field, owned
    by ``ResourceCommonsEnv._build_info``) is unaffected.
    """
    N = state.agent_positions.shape[0]
    obs_dim = ObservationLayout.total_dim(N, K)
    joint = np.zeros((N, obs_dim), dtype=np.float32)
    for i in range(N):
        joint[i] = build_observation(state, i, L, T_max, K)
    if env_cfg is not None and not env_cfg.c_visible:
        c_slot = ObservationLayout.block_offset("global", N, K)[0]
        joint[:, c_slot] = np.float32(env_cfg.c_hidden_constant)
    return joint


# --------------------------------------------------------------- private blocks


def _self_block(
    state: ResourceCommonsState,
    agent_id: int,
    L: int,
    T_max: int,
) -> np.ndarray:
    """``(4,)``: ``[x_norm, y_norm, cum_harvest_norm, steps_since_harvest_norm]``."""
    pos = state.agent_positions[agent_id]
    cap_zeta = float(state.agent_caps[agent_id].zeta)
    return np.array(
        [
            float(pos[0]) / max(L - 1, 1),
            float(pos[1]) / max(L - 1, 1),
            float(state.cumulative_harvests[agent_id]) / max(cap_zeta, 1.0),
            float(state.steps_since_harvest[agent_id]) / max(T_max, 1),
        ],
        dtype=np.float32,
    )


def _resource_block(
    state: ResourceCommonsState,
    agent_id: int,
    K: int,
) -> np.ndarray:
    """``(K * 3,)``: FOV-visible ``(rel_dx, rel_dy, q_norm)`` padded to K slots.

    ``q_norm = q_k / Q_max`` with ``Q_max = 10`` matching :mod:`schemas._constants`.
    """
    from hyper_mve.schemas._constants import Q_MAX

    pos = state.agent_positions[agent_id]
    fov = state.agent_caps[agent_id].fov_int
    visible: list[tuple[float, float, float]] = []
    for k in range(K):
        rel = state.resource_positions[k] - pos
        if int(np.max(np.abs(rel))) <= fov:
            visible.append(
                (float(rel[0]), float(rel[1]),
                 float(state.resource_stocks[k]) / float(Q_MAX))
            )
    return pad_resource_block(visible, K)


def _neighbor_block(
    state: ResourceCommonsState,
    agent_id: int,
    N: int,
) -> np.ndarray:
    """``((N-1) * 9,)``: FOV-visible neighbours, each entity:
    ``(rel_dx, rel_dy, last_action_onehot[6], presence_flag)``.

    Iteration order is by ``j`` ascending (skipping ``agent_id``) — the
    same convention BeliefNet ``z_hat`` uses, so any downstream code that
    cross-references this block stays consistent.
    """
    pos = state.agent_positions[agent_id]
    fov = state.agent_caps[agent_id].fov_int
    visible: list[tuple[float, float, np.ndarray, bool]] = []
    for j in range(N):
        if j == agent_id:
            continue
        rel = state.agent_positions[j] - pos
        if int(np.max(np.abs(rel))) <= fov:
            act_oh = np.zeros(_ACTION_ONEHOT_DIM, dtype=np.float32)
            last_a = int(state.last_actions[j])
            if 0 <= last_a < _ACTION_ONEHOT_DIM:
                act_oh[last_a] = 1.0
            visible.append((float(rel[0]), float(rel[1]), act_oh, True))
    return pad_neighbor_block(visible, N)


def _global_block(
    state: ResourceCommonsState,
    T_max: int,
) -> np.ndarray:
    """``(2,)``: ``[c_t, time_remaining_ratio]``."""
    time_remaining = float(T_max - state.step_idx) / max(T_max, 1)
    return np.array([float(state.c_t), time_remaining], dtype=np.float32)
