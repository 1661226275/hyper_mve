"""Q-based policy target (stage B of the prior-collapse fix).

The upstream target is the normalized root VISIT count. Visits are allocated by
UCB, whose prior term dominates its [0,1]-clipped value term when the prior is
peaked — so under a collapsed policy the target largely echoes the prior back
and the loop closes on itself. ``--policy_target_type q_softmax`` reads the
target from the search's own per-agent advantages instead.

These test the pure helpers, so no model, env, buffer or Ray is involved.
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest
import torch

_FORK = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "hyper_mve", "algo", "mazero_mixed",
)
if _FORK not in sys.path:
    sys.path.insert(0, _FORK)

from core.train import policy_loss_step, policy_target_weights  # noqa: E402

BATCH, C, N, A = 4, 13, 2, 6


def _cfg(target_type="q_softmax", temperature=1.0):
    return types.SimpleNamespace(
        policy_target_type=target_type,
        policy_target_temperature=temperature,
    )


def _inputs(seed=0, n_valid=C):
    g = torch.Generator().manual_seed(seed)
    log_prob = torch.randn(BATCH, N, C, generator=g)
    mask = torch.zeros(BATCH, C)
    mask[:, :n_valid] = 1.0
    visits = torch.rand(BATCH, C, generator=g) * mask
    visits = visits / visits.sum(1, keepdim=True).clamp_min(1e-8)
    adv = torch.randn(BATCH, C, N, generator=g)
    return log_prob, log_prob.sum(dim=1), visits, adv, mask


def test_target_is_prior_independent():
    """The precise encoding of stage B: the q_softmax loss must not depend on
    the visit counts at all, and must depend on the advantages."""
    cfg = _cfg()
    lp, slp, visits, adv, mask = _inputs()

    other_visits = torch.rand(BATCH, C) * mask
    other_visits = other_visits / other_visits.sum(1, keepdim=True).clamp_min(1e-8)
    a = policy_loss_step(cfg, lp, slp, visits, adv, mask)
    b = policy_loss_step(cfg, lp, slp, other_visits, adv, mask)
    torch.testing.assert_close(a, b), "q_softmax target leaked visit counts"

    c = policy_loss_step(cfg, lp, slp, visits, adv + 1.7 * torch.randn(BATCH, C, N), mask)
    assert not torch.allclose(a, c), "q_softmax target ignored the advantages"


def test_visit_target_is_bit_identical_to_upstream():
    """Default config must reproduce the pre-change loss exactly, so the A/B is
    honest and the flag is a true no-op when off."""
    lp, slp, visits, adv, mask = _inputs()
    got = policy_loss_step(_cfg("visit"), lp, slp, visits, adv, mask)
    want = -(slp * visits * mask).sum(dim=1)
    assert torch.equal(got, want)


def test_target_rows_sum_to_one_per_agent():
    _, _, _, adv, mask = _inputs()
    w = policy_target_weights(adv, mask, 1.0)
    assert w.shape == (BATCH, C, N)
    torch.testing.assert_close(w.sum(dim=1), torch.ones(BATCH, N))


def test_padding_gets_exactly_zero_weight():
    _, _, _, adv, mask = _inputs(n_valid=5)
    w = policy_target_weights(adv, mask, 1.0)
    assert torch.equal(w[:, 5:, :], torch.zeros(BATCH, C - 5, N))
    torch.testing.assert_close(w.sum(dim=1), torch.ones(BATCH, N))


def test_out_of_trajectory_row_is_finite_and_zero():
    """Landmine 1: the reanalyze worker gives out-of-trajectory rows an
    ALL-False mask. softmax over an all -inf row is NaN, and 0*NaN=NaN would
    then poison total_loss even though the row's weight is zero."""
    lp, slp, visits, adv, mask = _inputs()
    mask[2] = 0.0                                  # fully masked row
    w = policy_target_weights(adv, mask, 1.0)
    assert torch.isfinite(w).all()
    assert torch.equal(w[2], torch.zeros(C, N))

    loss = policy_loss_step(_cfg(), lp, slp, visits, adv, mask)
    assert torch.isfinite(loss).all()
    assert loss[2].item() == 0.0


def test_fp16_autocast_does_not_nan():
    """Landmine 2: under autocast a -1e9 sentinel overflows fp16 (max 65504) to
    -inf, reintroducing landmine 1 a few hundred steps in once GradScaler
    settles. Guarded by computing the target in fp32."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA for fp16 autocast")
    dev = "cuda"
    lp, slp, visits, adv, mask = _inputs()
    lp, slp = lp.to(dev), slp.to(dev)
    visits, adv, mask = visits.to(dev), adv.to(dev), mask.to(dev)
    mask[1] = 0.0
    with torch.autocast("cuda", dtype=torch.float16):
        loss = policy_loss_step(_cfg(), lp, slp, visits, adv, mask)
    assert torch.isfinite(loss).all(), f"non-finite under fp16 autocast: {loss}"


def test_gradient_flows_and_is_finite():
    lp, _, visits, adv, mask = _inputs()
    mask[3] = 0.0
    lp = lp.clone().requires_grad_(True)
    loss = policy_loss_step(_cfg(), lp, lp.sum(dim=1), visits, adv, mask)
    loss.mean().backward()
    assert torch.isfinite(lp.grad).all()
    # a fully-masked row must contribute no gradient
    assert torch.equal(lp.grad[3], torch.zeros(N, C))


def test_temperature_sharpens_the_target():
    _, _, _, adv, mask = _inputs()
    hot = policy_target_weights(adv, mask, 4.0).max(dim=1).values
    cold = policy_target_weights(adv, mask, 0.25).max(dim=1).values
    assert (cold > hot).all(), "lower temperature must produce a peakier target"
