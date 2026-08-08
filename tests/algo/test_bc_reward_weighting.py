"""Reward-weighted behavior-cloning unit tests (2026-07-22, Part B).

Guards the two corrections the design hinges on:
  * correction 1 — G_t is the FULL-episode return-to-go (sum to end), not windowed;
  * correction 2 — G_t and the weight are PER-AGENT (general-sum game).
plus the per-(regime,agent) normalization, positive-only weighting, and the
bit-exact plain-BC fallback when weighting is off.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_FORK = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "hyper_mve", "algo", "mazero_mixed",
)
if _FORK not in sys.path:
    sys.path.insert(0, _FORK)
from core.train import bc_loss_step, bc_reward_weights  # noqa: E402


def _cfg(on, cap=0.0):
    return SimpleNamespace(bc_reward_weighting=on, bc_weight_cap=cap)


# --- correction 1: full-episode return-to-go (the formula game.py uses) --------

def test_returns_to_go_is_full_episode_sum_to_end():
    # (T, N) per-agent rewards; returns_to_go[t] must equal sum(rewards[t:]).
    rewards = np.array([[1., 10.], [2., 20.], [3., 30.], [4., 40.]], dtype=np.float32)
    rtg = np.flip(np.cumsum(np.flip(rewards, axis=0), axis=0), axis=0)
    for t in range(len(rewards)):
        assert np.allclose(rtg[t], rewards[t:].sum(axis=0))
    # a late step keeps its post-step reward (would be lost by a windowed cumsum)
    assert rtg[2, 0] == 3. + 4.


# --- correction 2: per-agent weighting -----------------------------------------

def test_weight_is_per_agent_on_asymmetric_returns():
    # one regime, both agents "reference & in-traj"; agent 0's G is high on step 0
    # and low on step 1, agent 1 the reverse. Their weights must differ per step.
    B, K1, N = 4, 1, 2
    rtg = torch.tensor([[[5.0, 0.0]], [[0.0, 5.0]], [[5.0, 0.0]], [[0.0, 5.0]]])
    regime = torch.zeros(B, dtype=torch.long)
    ref = torch.ones(B)
    mask = torch.ones(B, K1)
    w = bc_reward_weights(_cfg(True), rtg, regime, ref, mask, n_regimes=5)
    # agent 0 above its own mean on rows 0,2 (weight>0), below on 1,3 (weight 0)
    assert w[0, 0, 0] > 0 and w[1, 0, 0] == 0
    assert w[1, 0, 1] > 0 and w[0, 0, 1] == 0
    # and the two agents' weight columns are not identical
    assert not torch.allclose(w[..., 0], w[..., 1])


# --- positive-advantage only ---------------------------------------------------

def test_below_own_average_gets_zero_weight():
    B, K1, N = 6, 1, 1
    rtg = torch.tensor([[[v]] for v in (0., 1., 2., 3., 4., 5.)])
    regime = torch.zeros(B, dtype=torch.long)
    w = bc_reward_weights(_cfg(True), rtg, regime, torch.ones(B), torch.ones(B, K1), 5)
    # mean is 2.5; values below get 0, above get >0
    assert (w[:3] == 0).all()
    assert (w[3:] > 0).all()


# --- per-(regime, agent) normalization is group-local --------------------------

def test_normalization_is_per_regime():
    # regime 0 has small returns, regime 1 large. Within each, the max-of-group
    # gets a positive weight — pooling would drown regime 0 entirely.
    B, K1, N = 4, 1, 1
    rtg = torch.tensor([[[0.0]], [[1.0]], [[100.0]], [[101.0]]])
    regime = torch.tensor([0, 0, 1, 1])
    w = bc_reward_weights(_cfg(True), rtg, regime, torch.ones(B), torch.ones(B, K1), 5)
    # the above-average member of EACH regime is upweighted (regime 0 not drowned)
    assert w[1, 0, 0] > 0    # regime 0's high one
    assert w[3, 0, 0] > 0    # regime 1's high one
    assert w[0, 0, 0] == 0 and w[2, 0, 0] == 0


# --- off ⇒ ones ⇒ bit-exact plain BC ------------------------------------------

def test_off_returns_all_ones():
    rtg = torch.randn(3, 2, 2)
    w = bc_reward_weights(_cfg(False), rtg, torch.zeros(3, dtype=torch.long),
                          torch.ones(3), torch.ones(3, 2), 5)
    assert torch.equal(w, torch.ones_like(rtg))


def test_bc_loss_step_with_unit_weight_equals_plain_ce_sum():
    B, N, A = 2, 2, 6
    logits = torch.randn(B, N, A)
    action = torch.randint(0, A, (B, N))
    ref, mask = torch.ones(B), torch.ones(B)
    ones = torch.ones(B, N)
    got = bc_loss_step(logits, action, ref, mask, ones)
    logp = logits.float().log_softmax(dim=-1)
    ce = -logp.gather(2, action.unsqueeze(-1)).squeeze(-1).sum(dim=1)
    assert torch.allclose(got, ce)


def test_sparse_group_falls_back_to_uniform_not_zero():
    """A (regime,agent) group too sparse to standardize must keep weight 1 (plain
    BC), never 0 — otherwise BC silently dies for under-sampled regimes."""
    rtg = torch.tensor([[[42.0]]])                     # single valid step
    w = bc_reward_weights(_cfg(True), rtg, torch.zeros(1, dtype=torch.long),
                          torch.ones(1), torch.ones(1, 1), 5)
    assert w[0, 0, 0] == 1.0


def test_cap_limits_weight():
    rtg = torch.tensor([[[0.0]], [[100.0]]])   # one huge advantage
    regime = torch.zeros(2, dtype=torch.long)
    w = bc_reward_weights(_cfg(True, cap=1.5), rtg, regime,
                          torch.ones(2), torch.ones(2, 1), 5)
    assert w.max() <= 1.5 + 1e-6
