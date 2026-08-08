"""Root candidate construction in the sampled-MCTS tree (stage A of the
prior-collapse fix).

Drives the compiled ctree directly with numpy — no torch model, no env, no
Ray — so these run in milliseconds and pin the C++ contract that the Python
side depends on.

Background: upstream samples the root's children from ``beta`` (the policy
prior blended with Dirichlet noise) with replacement and dedups them through a
map. Once the prior collapses to one action that yields a ONE-CHILD root, and
``select_child_decoupled`` skips any action not ``present`` among the children,
so the search cannot consider the actions it would need to correct the prior.
``--root_cover star`` enumerates, per agent, every action against one shared
CRN anchor instead.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_FORK = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "hyper_mve", "algo", "mazero_mixed",
)
if _FORK not in sys.path:
    sys.path.insert(0, _FORK)

cytree = pytest.importorskip(
    "core.mcts.ctree.ctree_sampled.cytree",
    reason="compiled ctree extension not built; run core/mcts/ctree/make.sh",
)

B, N, A = 1, 2, 6
COVER_MAX = 1 + N * A          # 13
SIMS = 25
SEED = 7


def _collapsed_prior(peak: int = 5):
    """The exact pathology observed in results/: prior_action_fractions
    [0,0,0,0,0,1.0]."""
    p = np.full((B, N, A), 1e-8, dtype=np.float32)
    p[:, :, peak] = 1.0
    return p / p.sum(-1, keepdims=True)


def _make(mode, probs, beta=None, sampled_times=COVER_MAX, noise_eps=0.25):
    beta = probs.copy() if beta is None else beta
    noises = np.full((B, N, A), 1.0 / A, dtype=np.float32)
    rew = np.zeros((B, N), dtype=np.float32)
    val = np.zeros((B, N), dtype=np.float32)
    t = cytree.Tree_batch(B, N, A, sampled_times, SIMS, 0.01, SEED, 0.75, 0.8, 1, mode)
    t.prepare(rew.reshape(-1), val.reshape(-1), probs.reshape(-1),
              beta.reshape(-1), sampled_times, noise_eps, noises.reshape(-1))
    return t


def _run_sims(tree, value_fn=None, n=SIMS):
    """Run n simulations. value_fn(last_actions) -> (B, N) leaf values."""
    for i in range(n):
        _, _, last_actions = tree.batch_selection(19652.0, 1.25, 0.997)
        rew = np.zeros((B, N), dtype=np.float32)
        val = (np.zeros((B, N), dtype=np.float32) if value_fn is None
               else value_fn(last_actions).astype(np.float32))
        probs = np.full((B, N, A), 1.0 / A, dtype=np.float32)
        tree.batch_expansion_and_backup(i + 1, 0.997, 5, rew.reshape(-1),
                                        val.reshape(-1), probs.reshape(-1),
                                        probs.reshape(-1))


def test_upstream_sampling_collapses_to_one_child():
    """The bug, pinned. Not an assertion about desired behaviour — this is the
    baseline the fix has to beat, and it documents why."""
    acts = _make(0, _collapsed_prior()).get_roots_sampled_actions()[0]
    assert len(acts) == 1
    assert set(acts[:, 0]) == {5} and set(acts[:, 1]) == {5}


def test_root_covers_every_action_per_agent():
    acts = _make(1, _collapsed_prior()).get_roots_sampled_actions()[0]
    for i in range(N):
        assert set(acts[:, i].tolist()) == set(range(A)), (
            f"agent {i} missing actions: {set(range(A)) - set(acts[:, i].tolist())}")


def test_root_child_count_within_buffer_width():
    """Root children must fit sampled_action_times — the replay-buffer width.
    Exceeding it makes concat_with_zero_padding raise deep in the reanalyze
    worker, far from the cause."""
    for probs in (_collapsed_prior(), np.full((B, N, A), 1.0 / A, dtype=np.float32)):
        acts = _make(1, probs).get_roots_sampled_actions()[0]
        assert len(acts) <= COVER_MAX


def test_greedy_joint_is_always_present():
    """The action the policy would actually execute must be evaluated."""
    acts = _make(1, _collapsed_prior(peak=3)).get_roots_sampled_actions()[0]
    assert any(tuple(row) == (3, 3) for row in acts)


def test_enumerated_children_have_unit_importance_ratio():
    """IS-math guard: enumerated children are not sampled, so there is no
    sampling frequency to correct for. beta_hat == beta makes the exported
    ratio exactly pred_prob."""
    t = _make(1, _collapsed_prior())
    beta = np.asarray(t.get_roots_sampled_beta()[0], dtype=np.float64)
    beta_hat = np.asarray(t.get_roots_sampled_beta_hat()[0], dtype=np.float64)
    np.testing.assert_array_equal(beta_hat, beta)
    imp = np.asarray(t.get_roots_sampled_imp_ratio()[0], dtype=np.float64)
    pred = np.asarray(t.get_roots_sampled_pred_probs()[0], dtype=np.float64)
    np.testing.assert_allclose(imp, pred, rtol=1e-5, atol=1e-8)


def test_underflowed_beta_does_not_produce_nan():
    """Sampling could never draw a zero-probability action; enumeration can.
    Dividing prior by such a beta would emit inf/NaN into every visit count."""
    probs = _collapsed_prior()
    beta = probs.copy()
    beta[:, :, 0] = 0.0                       # exactly 0, as after fp32 underflow
    beta /= beta.sum(-1, keepdims=True)
    t = _make(1, probs, beta=beta)
    for name in ("get_roots_sampled_priors", "get_roots_sampled_imp_ratio",
                 "get_roots_sampled_beta", "get_roots_sampled_beta_hat"):
        v = np.asarray(getattr(t, name)()[0], dtype=np.float64)
        assert np.all(np.isfinite(v)), f"{name} produced non-finite values: {v}"
    # a zero-beta action is illegal-by-convention and must not be enumerated
    acts = t.get_roots_sampled_actions()[0]
    assert 0 not in set(acts[:, 0].tolist())


def test_every_root_child_gets_at_least_one_simulation():
    """The pre-existing root round-robin (cnode.cpp) is what makes the cover
    worth building: it evaluates each child exactly once before UCB engages.
    The Q-based policy target depends on this holding."""
    t = _make(1, _collapsed_prior())
    _run_sims(t)
    visits = np.asarray(t.get_roots_sampled_visit_count()[0])
    assert visits.min() >= 1, f"unvisited root children: {visits}"


def test_value_can_outvote_a_onehot_prior():
    """The test that would have caught the whole failure.

    A leaf value that strongly favours action 2 must show up in the search's Q
    estimates even though the prior is one-hot on 5. Under the visit-count
    target this is exactly what UCB's prior term suppresses.
    """
    def value_fn(last_actions):
        v = np.zeros((B, N), dtype=np.float32)
        for b in range(B):
            if 2 in last_actions[b]:
                v[b, :] = 10.0
        return v

    t = _make(1, _collapsed_prior())
    _run_sims(t, value_fn=value_fn)
    acts = t.get_roots_sampled_actions()[0]
    q = np.asarray(t.get_roots_sampled_qvalues(0.997)[0], dtype=np.float64)
    best = acts[int(np.argmax(q))]
    assert 2 in set(best.tolist()), (
        f"highest-Q root child {best.tolist()} does not contain the "
        f"high-value action 2; q={q}")


def test_leaf_sampling_is_independent_of_root_cover():
    """sampled_action_times is forced up by the cover (it is the buffer width);
    leaves must not inherit that branching factor."""
    t = _make(1, _collapsed_prior())
    _run_sims(t, n=3)      # leaves expand with sampled_times=5 in _run_sims
    visits = np.asarray(t.get_roots_sampled_visit_count()[0])
    assert visits.sum() == 3
