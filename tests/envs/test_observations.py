"""Unit tests for six-block observations (Pkg-02 spec 04 §5.1)."""
from __future__ import annotations

import numpy as np
import pytest

from hyper_mve.envs.resource_commons.observations import (
    build_joint_observation,
    build_observation,
)
from hyper_mve.envs.resource_commons.state import ResourceCommonsState
from hyper_mve.schemas import (
    AgentType,
    CapabilityVector,
    ObservationLayout,
)


def _make_state(
    *,
    N: int = 4,
    K: int = 20,
    L: int = 16,
    T_max: int = 200,
    agent_positions: np.ndarray | None = None,
    resource_positions: np.ndarray | None = None,
    resource_stocks: np.ndarray | None = None,
    agent_caps: tuple[CapabilityVector, ...] | None = None,
    agent_types: np.ndarray | None = None,
    last_actions: np.ndarray | None = None,
    c_t: float = 0.5,
) -> ResourceCommonsState:
    if agent_positions is None:
        agent_positions = np.zeros((N, 2), dtype=np.int32)
    if resource_positions is None:
        resource_positions = np.zeros((K, 2), dtype=np.int32)
    if resource_stocks is None:
        resource_stocks = np.full(K, 5.0, dtype=np.float32)
    if agent_caps is None:
        agent_caps = tuple(
            CapabilityVector(eta=1.0, phi_fov=3.0, nu=0.9, zeta=20.0)
            for _ in range(N)
        )
    if agent_types is None:
        agent_types = np.zeros(N, dtype=np.int8)
    if last_actions is None:
        last_actions = np.zeros(N, dtype=np.int64)
    return ResourceCommonsState(
        agent_positions=agent_positions,
        cumulative_harvests=np.zeros(N, dtype=np.float32),
        steps_since_harvest=np.zeros(N, dtype=np.int32),
        last_actions=last_actions,
        resource_positions=resource_positions,
        resource_stocks=resource_stocks,
        c_t=c_t,
        c_history=np.zeros(T_max + 1, dtype=np.float32),
        agent_caps=agent_caps,
        agent_types=agent_types,
        hotspot_centers=np.zeros((1, 2), dtype=np.int32),
        step_idx=0,
        done=False,
    )


# ---------- total dim / dtype ----------

def test_obs_total_dim_matches_layout():
    s = _make_state(N=4, K=20)
    obs = build_observation(s, agent_id=0, L=16, T_max=200, K=20)
    expected = ObservationLayout.total_dim(N=4, K=20)
    assert expected == 99
    assert obs.shape == (expected,)


def test_obs_dtype_float32():
    s = _make_state(N=4, K=20)
    obs = build_observation(s, agent_id=0, L=16, T_max=200, K=20)
    assert obs.dtype == np.float32


def test_joint_observation_shape():
    s = _make_state(N=4, K=20)
    joint = build_joint_observation(s, L=16, T_max=200, K=20)
    assert joint.shape == (4, 99)
    assert joint.dtype == np.float32


# ---------- Self-Info (key v4 invariant) ----------

def test_type_block_is_own_one_hot():
    types = np.array([0, 0, 1, 1], dtype=np.int8)
    s = _make_state(N=4, K=20, agent_types=types)
    start, end = ObservationLayout.block_offset("type", N=4, K=20)

    obs0 = build_observation(s, 0, L=16, T_max=200, K=20)
    assert np.allclose(obs0[start:end], [1.0, 0.0])     # ALPHA

    obs2 = build_observation(s, 2, L=16, T_max=200, K=20)
    assert np.allclose(obs2[start:end], [0.0, 1.0])     # BETA


def test_capability_block_is_own_cap():
    custom_cap = CapabilityVector(eta=0.7, phi_fov=2.5, nu=0.85, zeta=18.0)
    caps = (custom_cap,) + tuple(
        CapabilityVector(eta=1.0, phi_fov=3.0, nu=0.9, zeta=20.0)
        for _ in range(3)
    )
    s = _make_state(N=4, K=20, agent_caps=caps)
    start, end = ObservationLayout.block_offset("capability", N=4, K=20)
    obs = build_observation(s, 0, L=16, T_max=200, K=20)
    assert np.allclose(obs[start:end], custom_cap.to_array())


def test_other_agents_type_does_not_leak_into_obs():
    """v4 key — changing another agent's type must not change my observation."""
    pos = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.int32)
    s1 = _make_state(
        N=4, K=20, agent_positions=pos.copy(),
        agent_types=np.array([0, 0, 1, 1], dtype=np.int8),
    )
    s2 = _make_state(
        N=4, K=20, agent_positions=pos.copy(),
        agent_types=np.array([0, 1, 0, 1], dtype=np.int8),
    )
    # agent 0 stays ALPHA in both
    obs1 = build_observation(s1, 0, L=16, T_max=200, K=20)
    obs2 = build_observation(s2, 0, L=16, T_max=200, K=20)
    assert np.allclose(obs1, obs2)


def test_other_agents_cap_does_not_leak_into_obs():
    """Same Self-Info check on capability: my obs is invariant to others' caps."""
    pos = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.int32)
    own = CapabilityVector(eta=1.0, phi_fov=3.0, nu=0.9, zeta=20.0)
    s1 = _make_state(
        N=4, K=20, agent_positions=pos.copy(),
        agent_caps=(own,) + tuple(
            CapabilityVector(eta=1.0, phi_fov=3.0, nu=0.9, zeta=20.0)
            for _ in range(3)
        ),
    )
    s2 = _make_state(
        N=4, K=20, agent_positions=pos.copy(),
        agent_caps=(own,) + tuple(
            CapabilityVector(eta=0.5, phi_fov=4.0, nu=1.0, zeta=30.0)
            for _ in range(3)
        ),
    )
    obs1 = build_observation(s1, 0, L=16, T_max=200, K=20)
    obs2 = build_observation(s2, 0, L=16, T_max=200, K=20)
    assert np.allclose(obs1, obs2)


# ---------- FOV filter ----------

def test_resource_outside_fov_is_zero_padded():
    s = _make_state(
        N=1, K=2,
        agent_positions=np.array([[0, 0]], dtype=np.int32),
        resource_positions=np.array([[0, 1], [10, 10]], dtype=np.int32),
        resource_stocks=np.array([5.0, 5.0], dtype=np.float32),
        agent_caps=(CapabilityVector(eta=1.0, phi_fov=2.0, nu=1.0, zeta=20.0),),
    )
    obs = build_observation(s, 0, L=16, T_max=200, K=2)
    start, end = ObservationLayout.block_offset("resource", N=1, K=2)
    res = obs[start:end].reshape(2, 3)
    # Near resource (0, 1) is visible — at least one component nonzero
    assert not np.allclose(res[0], 0.0)
    # Far resource (10, 10) is outside FOV — padded zero
    assert np.allclose(res[1], 0.0)


def test_neighbor_presence_flag_distinguishes_visible_from_padded():
    s = _make_state(
        N=3, K=20,
        agent_positions=np.array([[0, 0], [1, 1], [10, 10]], dtype=np.int32),
        agent_caps=tuple(
            CapabilityVector(eta=1.0, phi_fov=2.0, nu=1.0, zeta=20.0)
            for _ in range(3)
        ),
    )
    obs = build_observation(s, 0, L=16, T_max=200, K=20)
    start, end = ObservationLayout.block_offset("neighbor", N=3, K=20)
    nb = obs[start:end].reshape(2, 9)
    # neighbour 1 at (1,1) is in FOV ⇒ presence=1
    assert nb[0, 8] == 1.0
    # neighbour 2 at (10,10) is out of FOV ⇒ padded ⇒ presence=0
    assert nb[1, 8] == 0.0


# ---------- global block ----------

def test_global_block_carries_c_t():
    s = _make_state(N=4, K=20, c_t=0.73)
    obs = build_observation(s, 0, L=16, T_max=200, K=20)
    start, end = ObservationLayout.block_offset("global", N=4, K=20)
    g = obs[start:end]
    assert g[0] == pytest.approx(0.73)
    # time_remaining at step 0 should be 1.0
    assert g[1] == pytest.approx(1.0)
