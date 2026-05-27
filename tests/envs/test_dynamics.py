"""Unit tests for ResourceCommons dynamics (Pkg-02 spec 02 §5.1)."""
from __future__ import annotations

import numpy as np
import pytest

from hyper_mve.envs.resource_commons.dynamics import (
    compute_alpha,
    compute_neighbor_factor,
    fair_share_harvest,
    step_dynamics,
)
from hyper_mve.envs.resource_commons.state import ResourceCommonsState
from hyper_mve.schemas import CapabilityVector


# Default physical constants (matching EnvConfig defaults / _constants.py)
ALPHA_MIN = 0.02
ALPHA_MAX = 0.20
Q_MAX = 10.0
KAPPA_F = 6.0
THETA_F = 0.3
D_NBR = 3


def _make_state(
    *,
    N: int,
    K: int,
    agent_positions: np.ndarray | None = None,
    resource_positions: np.ndarray | None = None,
    resource_stocks: np.ndarray | None = None,
    agent_caps: tuple[CapabilityVector, ...] | None = None,
    c_t: float = 1.0,
    T_max: int = 200,
) -> ResourceCommonsState:
    if agent_positions is None:
        agent_positions = np.zeros((N, 2), dtype=np.int32)
    if resource_positions is None:
        resource_positions = np.zeros((K, 2), dtype=np.int32)
    if resource_stocks is None:
        resource_stocks = np.full(K, Q_MAX, dtype=np.float32)
    if agent_caps is None:
        agent_caps = tuple(
            CapabilityVector(eta=1.0, phi_fov=3.0, nu=0.9, zeta=20.0)
            for _ in range(N)
        )
    return ResourceCommonsState(
        agent_positions=agent_positions,
        cumulative_harvests=np.zeros(N, dtype=np.float32),
        steps_since_harvest=np.zeros(N, dtype=np.int32),
        last_actions=np.zeros(N, dtype=np.int64),
        resource_positions=resource_positions,
        resource_stocks=resource_stocks,
        c_t=c_t,
        c_history=np.zeros(T_max + 1, dtype=np.float32),
        agent_caps=agent_caps,
        agent_types=np.zeros(N, dtype=np.int8),
        hotspot_centers=np.zeros((1, 2), dtype=np.int32),
        step_idx=0,
        done=False,
    )


# ---------- α(c) ----------

def test_alpha_linear_endpoints():
    assert compute_alpha(0.0, ALPHA_MIN, ALPHA_MAX) == pytest.approx(0.02)
    assert compute_alpha(1.0, ALPHA_MIN, ALPHA_MAX) == pytest.approx(0.20)
    assert compute_alpha(0.5, ALPHA_MIN, ALPHA_MAX) == pytest.approx(0.11)


def test_alpha_out_of_range():
    with pytest.raises(AssertionError):
        compute_alpha(1.5, ALPHA_MIN, ALPHA_MAX)
    with pytest.raises(AssertionError):
        compute_alpha(-0.1, ALPHA_MIN, ALPHA_MAX)


# ---------- neighbor factor ----------

def test_neighbor_factor_full_neighbors():
    stocks = np.full(5, Q_MAX, dtype=np.float32)
    positions = np.array([[0, 0], [1, 0], [0, 1], [2, 2], [3, 3]], dtype=np.int32)
    f = compute_neighbor_factor(
        stocks, positions, k=0,
        q_max=Q_MAX, kappa_f=KAPPA_F, theta_f=THETA_F, d_nbr=D_NBR,
    )
    # mean_q_norm = 1.0 → sigmoid(6 - 0.3) ≈ 0.997
    assert f > 0.99


def test_neighbor_factor_empty_neighbors_fallback():
    """Isolated cell → fall back to σ(-θ_f) ≈ 0.426."""
    stocks = np.array([10.0], dtype=np.float32)
    positions = np.array([[0, 0]], dtype=np.int32)
    f = compute_neighbor_factor(
        stocks, positions, k=0,
        q_max=Q_MAX, kappa_f=KAPPA_F, theta_f=THETA_F, d_nbr=D_NBR,
    )
    expected = 1.0 / (1.0 + np.exp(THETA_F))
    assert f == pytest.approx(expected, abs=1e-6)


def test_neighbor_factor_all_neighbors_depleted():
    stocks = np.array([10.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    positions = np.array([[0, 0], [1, 0], [0, 1], [2, 2], [3, 3]], dtype=np.int32)
    f = compute_neighbor_factor(
        stocks, positions, k=0,
        q_max=Q_MAX, kappa_f=KAPPA_F, theta_f=THETA_F, d_nbr=D_NBR,
    )
    # mean_q_norm = 0 → sigmoid(-0.3) ≈ 0.426
    assert 0.4 < f < 0.45


# ---------- logistic regen ----------

def test_logistic_regen_5steps_against_analytic():
    """Single isolated cell, q_0 = 0, c = 1.0 (α = 0.2): closed-form recurrence.

    Per step:  q_{t+1} = q_t + α · f · (Q_max - q_t)
    with f = σ(-θ_f) for an isolated cell.
    """
    state = _make_state(
        N=1, K=1, c_t=1.0,
        resource_stocks=np.array([0.0], dtype=np.float32),
    )
    f_isolated = 1.0 / (1.0 + np.exp(THETA_F))   # ≈ 0.4256
    expected = 0.0
    for _ in range(5):
        step_dynamics(
            state, harvests_per_resource=np.zeros(1, dtype=np.float32),
            alpha_min=ALPHA_MIN, alpha_max=ALPHA_MAX, q_max=Q_MAX,
            kappa_f=KAPPA_F, theta_f=THETA_F, d_nbr=D_NBR,
        )
        expected = expected + 0.20 * f_isolated * (Q_MAX - expected)
    assert state.resource_stocks[0] == pytest.approx(expected, abs=1e-4)


def test_step_dynamics_clip_no_negative():
    """Consumption greater than stock leaves stock ≥ 0 (clip safety)."""
    state = _make_state(
        N=1, K=1, c_t=0.0,
        resource_stocks=np.array([1.0], dtype=np.float32),
    )
    step_dynamics(
        state, harvests_per_resource=np.array([5.0], dtype=np.float32),
        alpha_min=ALPHA_MIN, alpha_max=ALPHA_MAX, q_max=Q_MAX,
        kappa_f=KAPPA_F, theta_f=THETA_F, d_nbr=D_NBR,
    )
    assert state.resource_stocks[0] >= 0.0


def test_step_dynamics_clip_no_overflow():
    """Regen + 0 consumption cannot push stock above Q_max."""
    state = _make_state(
        N=1, K=1, c_t=1.0,
        resource_stocks=np.array([Q_MAX], dtype=np.float32),
    )
    step_dynamics(
        state, harvests_per_resource=np.zeros(1, dtype=np.float32),
        alpha_min=ALPHA_MIN, alpha_max=ALPHA_MAX, q_max=Q_MAX,
        kappa_f=KAPPA_F, theta_f=THETA_F, d_nbr=D_NBR,
    )
    assert state.resource_stocks[0] <= Q_MAX


# ---------- fair-share harvest ----------

def test_fair_share_three_on_one_cell():
    state = _make_state(
        N=3, K=1,
        agent_positions=np.array([[0, 0], [0, 0], [0, 0]], dtype=np.int32),
        resource_positions=np.array([[0, 0]], dtype=np.int32),
        resource_stocks=np.array([3.0], dtype=np.float32),
        agent_caps=tuple(
            CapabilityVector(eta=1.0, phi_fov=3.0, nu=1.0, zeta=20.0)
            for _ in range(3)
        ),
    )
    mask = np.array([True, True, True])
    per_agent, per_resource = fair_share_harvest(state, state.agent_positions, mask)
    # share = 3/3 = 1.0; each takes min(eta=1.0, 1.0) = 1.0
    assert np.allclose(per_agent, [1.0, 1.0, 1.0])
    assert per_resource[0] == pytest.approx(3.0)


def test_fair_share_eta_caps_when_share_exceeds_eta():
    state = _make_state(
        N=1, K=1,
        agent_positions=np.array([[0, 0]], dtype=np.int32),
        resource_positions=np.array([[0, 0]], dtype=np.int32),
        resource_stocks=np.array([10.0], dtype=np.float32),
        agent_caps=(CapabilityVector(eta=0.5, phi_fov=3.0, nu=1.0, zeta=20.0),),
    )
    per_agent, _ = fair_share_harvest(
        state, state.agent_positions, np.array([True]),
    )
    assert per_agent[0] == pytest.approx(0.5)   # min(0.5, 10) = 0.5


def test_fair_share_off_cell_no_harvest():
    state = _make_state(
        N=1, K=1,
        agent_positions=np.array([[5, 5]], dtype=np.int32),
        resource_positions=np.array([[0, 0]], dtype=np.int32),
        resource_stocks=np.array([10.0], dtype=np.float32),
    )
    per_agent, _ = fair_share_harvest(
        state, state.agent_positions, np.array([True]),
    )
    assert per_agent[0] == 0.0


def test_fair_share_harvest_mask_off():
    state = _make_state(
        N=1, K=1,
        agent_positions=np.array([[0, 0]], dtype=np.int32),
        resource_positions=np.array([[0, 0]], dtype=np.int32),
        resource_stocks=np.array([10.0], dtype=np.float32),
    )
    per_agent, _ = fair_share_harvest(
        state, state.agent_positions, np.array([False]),
    )
    assert per_agent[0] == 0.0
