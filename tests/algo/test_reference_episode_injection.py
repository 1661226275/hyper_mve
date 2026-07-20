"""2026-07-20 harvest-collapse fix package: unit tests for the two pure
schedule/weighting functions, exercised in isolation from the full training
step (no model, no batch, no ray) — same style as test_grad_gating.py.

``reference_episode_prob`` (core/selfplay_worker.py) and
``reward_nonzero_weight`` (core/train.py) are the mechanisms; see their
docstrings and hyper_mve/envs/relation_commons/reference_policies.py for the
harvest-collapse context they were written against.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from hyper_mve.algo.runner import _ensure_fork_on_path

_ensure_fork_on_path()

from core.selfplay_worker import reference_episode_prob  # noqa: E402
from core.train import reward_nonzero_weight  # noqa: E402


def _cfg(**kw):
    return SimpleNamespace(**kw)


# ------------------------------------------------------- reference_episode_prob


def test_reference_episode_prob_off_by_default():
    cfg = _cfg()  # no schedule attrs at all -> getattr defaults to 0.0/0
    assert reference_episode_prob(cfg, 0) == 0.0
    assert reference_episode_prob(cfg, 999_999) == 0.0


def test_reference_episode_prob_held_at_start_without_anneal():
    """anneal_steps<=0 means 'stay at start' (used to hold a fixed ratio)."""
    cfg = _cfg(reference_episode_prob_start=0.4, reference_episode_prob_end=0.1,
               reference_episode_anneal_steps=0)
    assert reference_episode_prob(cfg, 0) == 0.4
    assert reference_episode_prob(cfg, 500_000) == 0.4


def test_reference_episode_prob_linear_decay():
    cfg = _cfg(reference_episode_prob_start=0.5, reference_episode_prob_end=0.1,
               reference_episode_anneal_steps=1000)
    assert reference_episode_prob(cfg, 0) == pytest.approx(0.5)
    assert reference_episode_prob(cfg, 250) == pytest.approx(0.4)
    assert reference_episode_prob(cfg, 500) == pytest.approx(0.3)
    assert reference_episode_prob(cfg, 1000) == pytest.approx(0.1)


def test_reference_episode_prob_holds_at_end_past_anneal():
    cfg = _cfg(reference_episode_prob_start=0.5, reference_episode_prob_end=0.1,
               reference_episode_anneal_steps=1000)
    assert reference_episode_prob(cfg, 5000) == pytest.approx(0.1)


def test_reference_episode_prob_supports_rising_schedule():
    """Not just decay: start < end must also ramp up correctly (frac is
    unsigned, only the direction of (p1 - p0) matters)."""
    cfg = _cfg(reference_episode_prob_start=0.0, reference_episode_prob_end=0.3,
               reference_episode_anneal_steps=100)
    assert reference_episode_prob(cfg, 0) == pytest.approx(0.0)
    assert reference_episode_prob(cfg, 50) == pytest.approx(0.15)
    assert reference_episode_prob(cfg, 100) == pytest.approx(0.3)


# ----------------------------------------------------------- reward_nonzero_weight


def test_reward_nonzero_weight_off_by_default():
    t = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, -0.01]])
    w = reward_nonzero_weight(t, upweight=0.0, eps=1e-6)
    assert torch.equal(w, torch.ones(3))


def test_reward_nonzero_weight_upweights_nonzero_team_reward():
    # rows: all-zero, positive harvest payoff, negative move cost (ε),
    # below-threshold noise that should NOT count as nonzero.
    t = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, -0.01],
        [1e-9, 0.0],
    ])
    w = reward_nonzero_weight(t, upweight=5.0, eps=1e-6)
    assert torch.allclose(w, torch.tensor([1.0, 6.0, 6.0, 1.0]))


def test_reward_nonzero_weight_sums_across_agents_before_thresholding():
    """Two agents whose rewards cancel in sum but are individually nonzero
    still count as an informative (nonzero-|reward|) transition, since the
    threshold sums absolute values, not the signed team total."""
    t = torch.tensor([[1.0, -1.0]])
    w = reward_nonzero_weight(t, upweight=3.0, eps=1e-6)
    assert torch.allclose(w, torch.tensor([4.0]))


def test_reward_nonzero_weight_shape_matches_batch():
    t = torch.zeros(7, 2)
    w = reward_nonzero_weight(t, upweight=2.0, eps=1e-6)
    assert w.shape == (7,)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
