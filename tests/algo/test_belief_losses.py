"""Unit tests for ``hyper_mve.algo.modules.belief_losses`` (v5 Pkg-09: L_regime + L_div)."""
from __future__ import annotations

import inspect
import math

import pytest
import torch

from hyper_mve.algo.modules.belief_losses import (
    l_regime,
    l_div,
    belief_loss,
    build_oracle_g_seq,
)

_G = 5  # |G| for g2


def _perfect_g_hat(g_true: torch.Tensor, N: int, eps: float = 1e-3) -> torch.Tensor:
    """Smoothed one-hot posterior matching g_true, (B, T, N, |G|)."""
    onehot = build_oracle_g_seq(g_true, N=N, n_regimes=_G)
    return onehot * (1.0 - eps) + eps / _G


def test_l_regime_zero_loss_at_perfect_pred():
    B, T, N = 2, 3, 4
    g_true = torch.randint(0, _G, (B, T))
    loss = l_regime(_perfect_g_hat(g_true, N), g_true)
    assert loss.item() < 0.01


def test_l_regime_uniform_pred_baseline():
    """Uniform posterior ⇒ L_regime = ln|G| ≈ 1.609."""
    B, T, N = 1, 2, 4
    g_true = torch.randint(0, _G, (B, T))
    g_hat = torch.full((B, T, N, _G), 1.0 / _G)
    loss = l_regime(g_hat, g_true)
    assert torch.allclose(loss, torch.tensor(math.log(_G)), atol=1e-4)


def test_l_regime_broadcast_correctness():
    """g_true (B, T) is broadcast over N — hand-computed scalar check."""
    B, T, N = 1, 2, 2
    g_true = torch.tensor([[0, 1]])
    g_hat = torch.full((B, T, N, _G), 1.0 / _G)
    # only agent 0 at t=0 predicts the true regime with prob 0.5
    g_hat[0, 0, 0] = torch.tensor([0.5, 0.125, 0.125, 0.125, 0.125])
    loss = l_regime(g_hat, g_true)
    # nll entries: -(log 0.5) at (0,0,0); -(log 0.2) at the other 3
    expected = (-math.log(0.5) + 3 * -math.log(1.0 / _G)) / (B * T * N)
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-5)


def test_l_regime_gradient_direction():
    """Gradient pushes probability mass toward the true regime."""
    B, T, N = 1, 1, 1
    g_true = torch.tensor([[2]])
    g_hat = torch.full((B, T, N, _G), 1.0 / _G, requires_grad=True)
    loss = l_regime(g_hat, g_true)
    loss.backward()
    # ∂L/∂g_hat is negative only at the true-class entry
    assert g_hat.grad[0, 0, 0, 2].item() < 0
    assert (g_hat.grad[0, 0, 0, [0, 1, 3, 4]] == 0).all()


def test_l_regime_wrong_input_shape_rejected():
    """g_true_seq with an N dim is rejected (broadcast contract)."""
    B, T, N = 1, 2, 4
    g_hat = torch.full((B, T, N, _G), 1.0 / _G)
    with pytest.raises(AssertionError, match="g_true_seq shape"):
        l_regime(g_hat, torch.zeros(B, T, N, dtype=torch.long))


def test_l_regime_mask():
    """Masked steps are excluded from the mean."""
    B, T, N = 1, 2, 2
    g_true = torch.tensor([[0, 0]])
    g_hat = torch.full((B, T, N, _G), 1.0 / _G)
    g_hat[0, 1] = torch.tensor([0.999, 0.00025, 0.00025, 0.00025, 0.00025])
    # mask out t=1 (the near-perfect step) -> loss = uniform CE
    mask = torch.tensor([[True, False]])
    loss = l_regime(g_hat, g_true, mask=mask)
    assert torch.allclose(loss, torch.tensor(math.log(_G)), atol=1e-4)


def test_l_div_hinge_zero_when_high_variance():
    B, T, N, h = 1, 1, 4, 8
    hidden = torch.randn(B, T, N, h) * 5
    loss = l_div(hidden, target_std=0.1)
    assert loss.item() < 1e-6


def test_l_div_hinge_positive_when_low_variance():
    B, T, N, h = 1, 1, 4, 8
    base = torch.randn(B, T, 1, h)
    hidden = base.expand(B, T, N, h).clone() + torch.randn(B, T, N, h) * 0.001
    loss = l_div(hidden, target_std=0.1)
    assert loss.item() > 0


def test_l_div_collapse_protection():
    B, T, N, h = 1, 1, 4, 8
    hidden = torch.ones(B, T, N, h) * 0.5
    loss = l_div(hidden, target_std=0.1)
    assert torch.allclose(loss, torch.tensor(0.01), atol=1e-6)


def test_belief_loss_combination():
    """belief_loss combines the two losses with weights; breakdown detached."""
    B, T, N = 2, 3, 4
    g_hat = torch.softmax(torch.randn(B, T, N, _G, requires_grad=True), dim=-1)
    hidden = torch.randn(B, T, N, 128, requires_grad=True)
    g_true = torch.randint(0, _G, (B, T))

    total, breakdown = belief_loss(
        g_hat, hidden, g_true, weights=(1.0, 0.01),
    )

    assert total.requires_grad
    assert "l_regime" in breakdown
    assert "l_div" in breakdown
    assert "total" in breakdown
    for v in breakdown.values():
        assert not v.requires_grad

    expected = 1.0 * breakdown["l_regime"] + 0.01 * breakdown["l_div"]
    assert torch.allclose(total.detach(), expected, atol=1e-5)


def test_l_regime_uses_oracle_field_name():
    """document: the label arg is g_true_seq (matches env.info['g_true'])."""
    sig = inspect.signature(l_regime)
    assert "g_true_seq" in sig.parameters


# ====== build_oracle_g_seq (curriculum oracle blend source) ======

def test_build_oracle_g_seq_one_hot():
    g_true = torch.tensor([[0, 2, 4]])
    g = build_oracle_g_seq(g_true, N=4, n_regimes=_G)
    assert g.shape == (1, 3, 4, _G)
    sums = g.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums))
    # all agents share the same one-hot
    assert (g[0, 1, :, 2] == 1.0).all()
    assert (g[0, 1, :, [0, 1, 3, 4]] == 0.0).all()


def test_build_oracle_g_seq_matches_l_regime():
    """Feeding the oracle one-hot into l_regime yields ~0 loss (end-to-end)."""
    g_true = torch.randint(0, _G, (2, 3))
    oracle_g = build_oracle_g_seq(g_true, N=4, n_regimes=_G)
    loss = l_regime(oracle_g * 0.999 + 0.001 / _G, g_true)
    assert loss.item() < 0.01
