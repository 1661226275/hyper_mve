"""Unit tests for ResourceCommonsState (Pkg-02 spec 01 §5.1)."""
from __future__ import annotations

import numpy as np
import pytest

from hyper_mve.envs.resource_commons.state import ResourceCommonsState
from hyper_mve.schemas import CapabilityVector


def _make_minimal_state(N: int = 4, K: int = 20, T_max: int = 200) -> ResourceCommonsState:
    caps = tuple(
        CapabilityVector(eta=1.0, phi_fov=3.0, nu=0.9, zeta=20.0)
        for _ in range(N)
    )
    return ResourceCommonsState(
        agent_positions=np.zeros((N, 2), dtype=np.int32),
        cumulative_harvests=np.zeros(N, dtype=np.float32),
        steps_since_harvest=np.zeros(N, dtype=np.int32),
        last_actions=np.zeros(N, dtype=np.int64),
        resource_positions=np.zeros((K, 2), dtype=np.int32),
        resource_stocks=np.ones(K, dtype=np.float32),
        c_t=0.5,
        c_history=np.zeros(T_max + 1, dtype=np.float32),
        agent_caps=caps,
        agent_types=np.zeros(N, dtype=np.int8),
        hotspot_centers=np.zeros((3, 2), dtype=np.int32),
        step_idx=0,
        done=False,
    )


def test_state_construct_minimal():
    s = _make_minimal_state()
    assert s.c_t == 0.5
    assert s.step_idx == 0
    assert s.done is False


def test_state_copy_independent():
    s1 = _make_minimal_state()
    s2 = s1.copy()
    # Mutate copy; original must stay intact.
    s2.agent_positions[0, 0] = 99
    s2.resource_stocks[0] = 99.0
    s2.c_t = 0.9
    s2.step_idx = 50
    assert s1.agent_positions[0, 0] != 99
    assert s1.resource_stocks[0] != 99.0
    assert s1.c_t == 0.5
    assert s1.step_idx == 0


def test_state_dtype_strict():
    s = _make_minimal_state()
    assert s.agent_positions.dtype == np.int32
    assert s.cumulative_harvests.dtype == np.float32
    assert s.steps_since_harvest.dtype == np.int32
    assert s.last_actions.dtype == np.int64
    assert s.resource_positions.dtype == np.int32
    assert s.resource_stocks.dtype == np.float32
    assert s.agent_types.dtype == np.int8
    assert s.c_history.dtype == np.float32
    assert s.hotspot_centers.dtype == np.int32


def test_state_caps_shared_by_reference():
    """agent_caps is a tuple of frozen dataclasses — share by reference is intentional."""
    s1 = _make_minimal_state()
    s2 = s1.copy()
    assert s1.agent_caps is s2.agent_caps  # same tuple object


def test_state_c_history_size_t_max_plus_one():
    """c_history must be sized T_max+1 (step_idx reaches T_max on the final step)."""
    s = _make_minimal_state(T_max=200)
    assert s.c_history.shape == (201,)
