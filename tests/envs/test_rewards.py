"""Hard-gate tests for Fehr-Schmidt reward — Table 3.5.4 + Ch4.1.1.

Pkg-02 spec 03 §5.1. Both tables are numerical contracts that block PR
merge if violated.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hyper_mve.envs.resource_commons.rewards import (
    compute_delta,
    compute_phi,
    compute_psi,
    compute_rewards,
    compute_rewards_torch,
)
from hyper_mve.schemas import AgentType


# Default Ch3.5 scalars (matching EnvConfig defaults).
KAPPA = 0.5
LAMBDA_DISADV = 2.0
LAMBDA_ADV = 0.6
EPSILON_MOVE = 0.01


# ---------- φ(c) ----------

def test_phi_endpoints():
    assert compute_phi(0.0, KAPPA) == pytest.approx(0.5)
    assert compute_phi(1.0, KAPPA) == pytest.approx(-0.5)
    assert compute_phi(0.5, KAPPA) == pytest.approx(0.0)


# ---------- ψ(Δ) ----------

def test_psi_zero_at_origin():
    assert compute_psi(0.0, LAMBDA_DISADV, LAMBDA_ADV) == 0.0


def test_psi_advantage_uses_lambda_adv():
    # Δ > 0 ⇒ ψ = -λ_adv · Δ
    assert compute_psi(1.0, LAMBDA_DISADV, LAMBDA_ADV) == pytest.approx(-0.6)
    assert compute_psi(2.0, LAMBDA_DISADV, LAMBDA_ADV) == pytest.approx(-1.2)


def test_psi_disadvantage_uses_lambda_disadv():
    # Δ < 0 ⇒ ψ = -λ_disadv · |Δ|
    assert compute_psi(-1.0, LAMBDA_DISADV, LAMBDA_ADV) == pytest.approx(-2.0)
    assert compute_psi(-0.5, LAMBDA_DISADV, LAMBDA_ADV) == pytest.approx(-1.0)


# ---------- Table 3.5.4 (φ × ψ four quadrants) — HARD GATE ----------

def test_table_3_5_4_lean_advantage():
    """Scene 1 — c=0, Δ=+1: φ=+0.5, ψ=-0.6, product=-0.3."""
    phi = compute_phi(0.0, KAPPA)
    psi = compute_psi(1.0, LAMBDA_DISADV, LAMBDA_ADV)
    assert phi * psi == pytest.approx(-0.3, abs=1e-6)


def test_table_3_5_4_lean_disadvantage():
    """Scene 2 — c=0, Δ=-1: φ=+0.5, ψ=-2.0, product=-1.0."""
    phi = compute_phi(0.0, KAPPA)
    psi = compute_psi(-1.0, LAMBDA_DISADV, LAMBDA_ADV)
    assert phi * psi == pytest.approx(-1.0, abs=1e-6)


def test_table_3_5_4_abundance_advantage():
    """Scene 3 — c=1, Δ=+1: φ=-0.5, ψ=-0.6, product=+0.3."""
    phi = compute_phi(1.0, KAPPA)
    psi = compute_psi(1.0, LAMBDA_DISADV, LAMBDA_ADV)
    assert phi * psi == pytest.approx(+0.3, abs=1e-6)


def test_table_3_5_4_abundance_disadvantage():
    """Scene 4 — c=1, Δ=-1: φ=-0.5, ψ=-2.0, product=+1.0."""
    phi = compute_phi(1.0, KAPPA)
    psi = compute_psi(-1.0, LAMBDA_DISADV, LAMBDA_ADV)
    assert phi * psi == pytest.approx(+1.0, abs=1e-6)


# ---------- Δ ----------

def test_delta_zero_when_all_equal():
    harvests = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    assert np.allclose(compute_delta(harvests), 0.0)


def test_delta_unit_advantage_four_agents():
    harvests = np.array([2.0, 1.0, 1.0, 1.0], dtype=np.float32)
    deltas = compute_delta(harvests)
    # u_0 = 2, mean_others = (1+1+1)/3 = 1 ⇒ Δ_0 = 1
    assert deltas[0] == pytest.approx(1.0, abs=1e-5)
    # u_1 = 1, mean_others = (2+1+1)/3 ≈ 1.333 ⇒ Δ_1 ≈ -0.333
    assert deltas[1] == pytest.approx(-1 / 3, abs=1e-5)


def test_delta_single_agent_degenerate():
    """N=1 ⇒ mean_others uses denom = max(N-1, 1) = 1 ⇒ Δ = u_0 - 0 = u_0."""
    deltas = compute_delta(np.array([3.5], dtype=np.float32))
    assert deltas[0] == pytest.approx(3.5)


# ---------- compute_rewards (numpy) ----------

def test_alpha_reward_has_no_preference_term():
    harvests = np.array([2.0, 1.0], dtype=np.float32)
    moved = np.array([False, False])
    types = np.array([int(AgentType.ALPHA)] * 2, dtype=np.int8)
    rewards, _ = compute_rewards(
        harvests=harvests, moved_mask=moved, agent_types=types,
        c_t=0.0, kappa=KAPPA, lambda_disadv=LAMBDA_DISADV,
        lambda_adv=LAMBDA_ADV, epsilon_move=EPSILON_MOVE,
    )
    # R^α = u_i (no movement cost in this test)
    assert np.allclose(rewards, [2.0, 1.0], atol=1e-5)


def test_beta_reward_includes_preference_term():
    harvests = np.array([2.0, 1.0], dtype=np.float32)
    moved = np.array([False, False])
    types = np.array([int(AgentType.BETA)] * 2, dtype=np.int8)
    rewards, _ = compute_rewards(
        harvests=harvests, moved_mask=moved, agent_types=types,
        c_t=0.0, kappa=KAPPA, lambda_disadv=LAMBDA_DISADV,
        lambda_adv=LAMBDA_ADV, epsilon_move=EPSILON_MOVE,
    )
    # u_0=2, Δ_0=+1, φ(0)=+0.5, ψ(+1)=-0.6 ⇒ term=-0.3 ⇒ R^β_0 = 1.7
    assert rewards[0] == pytest.approx(1.7, abs=1e-5)
    # u_1=1, Δ_1=-1, φ(0)=+0.5, ψ(-1)=-2.0 ⇒ term=-1.0 ⇒ R^β_1 = 0.0
    assert rewards[1] == pytest.approx(0.0, abs=1e-5)


def test_mixed_types_alpha_unaffected_beta_modulated():
    harvests = np.array([2.0, 1.0], dtype=np.float32)
    moved = np.array([False, False])
    types = np.array(
        [int(AgentType.ALPHA), int(AgentType.BETA)],
        dtype=np.int8,
    )
    rewards, _ = compute_rewards(
        harvests=harvests, moved_mask=moved, agent_types=types,
        c_t=0.0, kappa=KAPPA, lambda_disadv=LAMBDA_DISADV,
        lambda_adv=LAMBDA_ADV, epsilon_move=EPSILON_MOVE,
    )
    assert rewards[0] == pytest.approx(2.0, abs=1e-5)        # α
    assert rewards[1] == pytest.approx(0.0, abs=1e-5)        # β


def test_move_cost_applied_to_all_types():
    """ε·1[moved] is a physical-layer cost — applies to α and β alike."""
    harvests = np.array([1.0], dtype=np.float32)
    moved = np.array([True])
    types = np.array([int(AgentType.ALPHA)], dtype=np.int8)
    rewards, _ = compute_rewards(
        harvests=harvests, moved_mask=moved, agent_types=types,
        c_t=0.5, kappa=KAPPA, lambda_disadv=LAMBDA_DISADV,
        lambda_adv=LAMBDA_ADV, epsilon_move=EPSILON_MOVE,
    )
    assert rewards[0] == pytest.approx(0.99, abs=1e-6)


def test_returns_deltas_matching_compute_delta():
    harvests = np.array([3.0, 1.0, 2.0, 0.5], dtype=np.float32)
    moved = np.zeros(4, dtype=bool)
    types = np.zeros(4, dtype=np.int8)
    _, deltas = compute_rewards(
        harvests=harvests, moved_mask=moved, agent_types=types,
        c_t=0.5, kappa=KAPPA, lambda_disadv=LAMBDA_DISADV,
        lambda_adv=LAMBDA_ADV, epsilon_move=EPSILON_MOVE,
    )
    assert np.allclose(deltas, compute_delta(harvests))


# ---------- Ch4.1.1 partial-derivative table (autograd) — HARD GATE ----------

def test_alpha_grad_constant_one():
    """∂R^α/∂u_i = 1 always, regardless of c or Δ (100 random seeds)."""
    rng = np.random.default_rng(0)
    for _ in range(100):
        harvests = torch.tensor(
            rng.uniform(0.1, 3.0, size=4), requires_grad=True, dtype=torch.float64,
        )
        moved = torch.zeros(4, dtype=torch.bool)
        types = torch.full((4,), int(AgentType.ALPHA), dtype=torch.int8)
        c_t = float(rng.uniform(0.0, 1.0))
        rewards = compute_rewards_torch(
            harvests, moved, types, c_t,
            kappa=KAPPA, lambda_disadv=LAMBDA_DISADV,
            lambda_adv=LAMBDA_ADV, epsilon_move=EPSILON_MOVE,
        )
        for i in range(4):
            (grad,) = torch.autograd.grad(rewards[i], harvests, retain_graph=True)
            assert abs(grad[i].item() - 1.0) < 1e-9


def test_beta_grad_four_quadrants():
    """∂R^β/∂u_i ∈ {0.7, 2.0, 1.3, 0.0} across (c, Δ) quadrants.

    Derivation: ∂R^β/∂u_i = 1 + φ(c) · ψ'(Δ_i), where
        ψ'(Δ>0) = -λ_adv,  ψ'(Δ<0) = +λ_disadv.
    """
    moved = torch.zeros(4, dtype=torch.bool)
    types = torch.full((4,), int(AgentType.BETA), dtype=torch.int8)

    cases = [
        # (harvests, c_t, expected ∂R^β/∂u_0)
        ([2.0, 1.0, 1.0, 1.0], 0.0, 0.7),   # lean + advantage
        ([1.0, 2.0, 2.0, 2.0], 0.0, 2.0),   # lean + disadvantage
        ([2.0, 1.0, 1.0, 1.0], 1.0, 1.3),   # abundance + advantage
        ([1.0, 2.0, 2.0, 2.0], 1.0, 0.0),   # abundance + disadvantage
    ]
    for vals, c_t, expected in cases:
        harvests = torch.tensor(vals, dtype=torch.float64, requires_grad=True)
        rewards = compute_rewards_torch(
            harvests, moved, types, c_t,
            kappa=KAPPA, lambda_disadv=LAMBDA_DISADV,
            lambda_adv=LAMBDA_ADV, epsilon_move=EPSILON_MOVE,
        )
        (grad,) = torch.autograd.grad(rewards[0], harvests)
        assert abs(grad[0].item() - expected) < 1e-9, (
            f"c={c_t}, vals={vals}: got {grad[0].item()}, expected {expected}"
        )
