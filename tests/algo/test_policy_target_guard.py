"""Zero-information guard (stage C of the prior-collapse fix).

A flat-advantage root produces a near-UNIFORM q_softmax target, which is not a
weak gradient — it is a gradient actively pushing the policy toward uniform.
This is the modern analogue of the retired architecture's ``planner_on`` mask
and its ``mve_qstd_floor``.

The threshold is deliberately RELATIVE (a fraction of the batch-pooled
advantage spread, which the caller has already divided out). Measured live: the
Bayes-averaged value head runs 2.6x wider than the single head, so any absolute
threshold masks g1 alone on one arm and every regime on the other.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

_FORK = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "hyper_mve", "algo", "mazero_mixed",
)
if _FORK not in sys.path:
    sys.path.insert(0, _FORK)

from core.utils import policy_target_informative  # noqa: E402

ROWS, C, N = 6, 13, 2


def _row(spread, n_live=C, n_visited=C, seed=0):
    """One (adv, mask, visits) row with a controlled advantage spread."""
    rng = np.random.RandomState(seed)
    adv = rng.randn(C, N) * spread
    mask = np.zeros(C, dtype=bool)
    mask[:n_live] = True
    visits = np.zeros(C)
    visits[:n_visited] = 1.0 / max(n_visited, 1)
    return adv, mask, visits


def _stack(rows):
    adv = np.stack([r[0] for r in rows])
    mask = np.stack([r[1] for r in rows])
    vis = np.stack([r[2] for r in rows])
    return adv, mask, vis


def test_guard_is_off_by_default():
    """min_qstd <= 0 must return all-ones, bit-exact with not applying it."""
    adv, mask, vis = _stack([_row(0.001), _row(1.0)])
    out = policy_target_informative(adv, mask, vis, min_qstd=0.0)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, np.ones(2, dtype=np.float32))


def test_flat_advantage_row_is_masked_out():
    adv, mask, vis = _stack([_row(0.02, seed=1), _row(1.0, seed=2)])
    out = policy_target_informative(adv, mask, vis, min_qstd=0.4)
    assert out[0] == 0.0, "noise-level-flat row was not guarded out"
    assert out[1] == 1.0, "informative row was wrongly guarded out"


def test_threshold_is_relative_to_the_normalized_scale():
    """The same relative threshold must behave identically on two batches whose
    raw advantage scales differ — the caller has already divided by the pooled
    std, so only the RATIO matters. This is the property an absolute threshold
    lacks."""
    for scale in (0.25, 1.0, 4.0):
        rows = [_row(0.1 * scale, seed=3), _row(1.0 * scale, seed=4)]
        adv, mask, vis = _stack(rows)
        # emulate the caller's pooled normalization
        pooled = adv[mask].std()
        out = policy_target_informative(adv / pooled, mask, vis, min_qstd=0.4)
        assert out.tolist() == [0.0, 1.0], f"scale {scale} changed the verdict"


def test_single_child_root_is_masked_out():
    """The degenerate root the upstream sampler produces under a collapsed
    prior: one child, nothing to prefer."""
    adv, mask, vis = _stack([_row(1.0, n_live=1, n_visited=1, seed=5)])
    out = policy_target_informative(adv, mask, vis, min_qstd=0.4, min_children=2)
    assert out[0] == 0.0


def test_unvisited_children_do_not_count_as_evaluated():
    adv, mask, vis = _stack([_row(1.0, n_live=C, n_visited=1, seed=6)])
    out = policy_target_informative(adv, mask, vis, min_qstd=0.4)
    assert out[0] == 0.0, "row with only one EVALUATED child passed the guard"


def test_fully_masked_row_is_finite_and_zero():
    adv, mask, vis = _stack([_row(1.0, n_live=0, n_visited=0, seed=7)])
    out = policy_target_informative(adv, mask, vis, min_qstd=0.4)
    assert np.isfinite(out).all()
    assert out[0] == 0.0


def test_renormalization_preserves_loss_magnitude():
    """Masking half the batch must not halve the effective policy LR: the
    rescale should recover the mean over the SURVIVING rows."""
    per_row = torch.tensor([2.0, 2.0, 2.0, 2.0])
    informative = torch.tensor([1.0, 1.0, 0.0, 0.0])
    mask_batch = torch.ones(4, 3)                    # 3 in-trajectory steps each

    guarded = per_row * informative
    num = mask_batch.sum()
    den = (informative.unsqueeze(1) * mask_batch).sum()
    scale = torch.clamp(num / den, max=4.0)
    got = (guarded * scale).mean()

    # equals the mean over unmasked rows alone, NOT half of it
    torch.testing.assert_close(got, per_row.mean())


def test_renormalization_is_capped():
    """A batch where almost nothing survives must not blow up the gradient."""
    informative = torch.zeros(64)
    informative[0] = 1.0
    mask_batch = torch.ones(64, 1)
    num = mask_batch.sum()
    den = (informative.unsqueeze(1) * mask_batch).sum()
    scale = torch.clamp(num / den, max=4.0)
    assert scale.item() == 4.0, "uncapped rescale would be 64x here"


def test_all_masked_batch_gives_zero_not_nan():
    informative = torch.zeros(8)
    mask_batch = torch.ones(8, 2)
    den = (informative.unsqueeze(1) * mask_batch).sum()
    scale = (torch.clamp(mask_batch.sum() / den, max=4.0)
             if den > 0 else torch.zeros(()))
    loss = (torch.tensor([3.0] * 8) * informative * scale)
    assert torch.isfinite(loss).all()
    assert loss.sum().item() == 0.0


def test_g1_like_row_is_the_case_this_guards():
    """Reproduces the measured relation/g1 situation: advantage spread ~0.29 of
    the batch-pooled spread on the main method. That row must be guarded out at
    the recommended threshold, while a g0-like row (~1.4x) must survive."""
    g1, g0 = _row(0.29, seed=11), _row(1.40, seed=12)
    adv, mask, vis = _stack([g1, g0])
    out = policy_target_informative(adv, mask, vis, min_qstd=0.4)
    assert out.tolist() == [0.0, 1.0]
