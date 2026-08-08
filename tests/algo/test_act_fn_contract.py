"""``MAZeroMixedRunner.make_act_fn`` must be the same policy ``evaluate()`` runs.

The act_fn exists so best-response training (NashConv) and cross-play between
checkpoints can *drive* this policy one step at a time. Everything those measure
is meaningless if the act_fn is a slightly different policy from the one the
headline numbers come from, and the difference would be invisible — both would
produce plausible returns.

Two things make the equivalence non-obvious and are pinned here:

* ``_rollout_planner`` runs a whole regime's episodes as **one lockstep search
  batch**, while the act_fn runs at batch 1. That is only sound because the
  compiled tree is batch-size invariant.
* The search draws a tree seed from ``np_random`` per ``batch_search`` call, and
  the lockstep batch shares one draw across every episode at a given step. The
  act_fn therefore re-seeds at ``t == 0`` rather than continuing the stream.

The comparison is on the episode return and the action histogram rather than a
recorded action sequence: the env is deterministic under a pinned seed, so an
exact return match over 100 steps is a strong signature that the two agreed at
every step, and the rollout helpers already return both.

Scope: these compare at MATCHED batch size (1 vs 1), which is the equivalence
that has to hold. Against ``evaluate()``'s real 16-episode lockstep batch the
two drift by a few percent — the tree is batch-invariant but the model forward
is not bit-identical across batch sizes on GPU. That is measured and recorded in
``results/analysis/v6_reporting_protocol.md``; it is why act_fn-driven numbers
and ``eval_report`` numbers must not share a table.
"""
from __future__ import annotations

import dataclasses
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

pytest.importorskip("torch")
cytree = pytest.importorskip(
    "core.mcts.ctree.ctree_sampled.cytree",
    reason="compiled ctree extension not built; run core/mcts/ctree/make.sh",
)

from hyper_mve.utils.configs import V4Config                              # noqa: E402
from hyper_mve.algo.runner import MAZeroMixedRunner, _EVAL_SEARCH_SEED    # noqa: E402
from hyper_mve.envs.adapters.pettingzoo_wrapper import (                  # noqa: E402
    RelationCommonsPettingZooEnv,
)

REGIME = 2          # asymmetric: the belief pathway is actually doing something


@pytest.fixture(scope="module")
def runner_and_env():
    """A runner on a short-episode rel_recip. Weights are random — determinism
    is what is under test, not quality."""
    base = V4Config.from_preset("rel_recip")
    cfg = dataclasses.replace(
        base, env=dataclasses.replace(base.env, T_max=12))

    def env_fn():
        return RelationCommonsPettingZooEnv(
            cfg.env, oracle_mode=False, eval_info_mode=False)

    runner = MAZeroMixedRunner(cfg)
    runner._lazy_model()
    return runner, env_fn, cfg


def _drive_with_act_fn(act_fn, env_fn, g, cfg):
    """Roll one episode under act_fn, mirroring the rollout helpers' bookkeeping."""
    env = env_fn()
    agents = list(env.possible_agents)
    obs_dict, _ = env.reset(seed=10_000 + 97 * int(g) + 0, options={"g": int(g)})
    counts = np.zeros(int(cfg.env.A), dtype=np.int64)
    ep_ret, t, done = 0.0, 0, False
    while not done:
        obs = np.stack([obs_dict[a] for a in agents]).astype(np.float32)
        joint = act_fn(obs, t)
        acts = {a: int(joint[k]) for k, a in enumerate(agents)}
        for a in acts.values():
            counts[a] += 1
        obs_dict, rew, term, trunc, _ = env.step(acts)
        ep_ret += float(sum(rew.values()))
        t += 1
        done = bool(any(term.values()) or any(trunc.values()))
    env.close()
    return ep_ret, counts


def test_prior_act_fn_matches_the_prior_rollout(runner_and_env):
    runner, env_fn, cfg = runner_and_env
    device = runner._device_of(runner._model)

    want_returns, want_counts, _, _ = runner._rollout_prior(
        env_fn, REGIME, 1, device)
    got_ret, got_counts = _drive_with_act_fn(
        runner.make_act_fn("prior"), env_fn, REGIME, cfg)

    assert got_ret == pytest.approx(want_returns[0], abs=1e-9)
    np.testing.assert_array_equal(got_counts, want_counts)


def test_planner_act_fn_matches_the_planner_rollout(runner_and_env):
    """The load-bearing one: batch-1 act_fn vs the lockstep search batch."""
    runner, env_fn, cfg = runner_and_env
    device = runner._device_of(runner._model)

    want_returns, want_counts, _, _, _ = runner._rollout_planner(
        env_fn, REGIME, 1, device, np.random.RandomState(_EVAL_SEARCH_SEED))
    got_ret, got_counts = _drive_with_act_fn(
        runner.make_act_fn("planner"), env_fn, REGIME, cfg)

    assert got_ret == pytest.approx(want_returns[0], abs=1e-9)
    np.testing.assert_array_equal(got_counts, want_counts)


def test_act_fn_resets_belief_state_on_new_episode(runner_and_env):
    """``t == 0`` must mean "new episode" — otherwise the belief GRU carries
    across episode boundaries and BR training silently measures a policy with
    leaked history."""
    runner, env_fn, cfg = runner_and_env
    act_fn = runner.make_act_fn("planner")

    first = _drive_with_act_fn(act_fn, env_fn, REGIME, cfg)
    second = _drive_with_act_fn(act_fn, env_fn, REGIME, cfg)   # same env seed
    assert second[0] == pytest.approx(first[0], abs=1e-9)
    np.testing.assert_array_equal(second[1], first[1])


def test_prior_and_planner_are_different_policies(runner_and_env):
    """Guards against the act_fn silently ignoring ``mode``.

    On v5 these score 14.28 (prior) vs 61.94 (planner), so which one a metric
    freezes changes what it means — the two must not be the same function.
    """
    runner, env_fn, cfg = runner_and_env
    prior = _drive_with_act_fn(runner.make_act_fn("prior"), env_fn, REGIME, cfg)
    plan = _drive_with_act_fn(runner.make_act_fn("planner"), env_fn, REGIME, cfg)
    assert not np.array_equal(prior[1], plan[1])


def test_rejects_unknown_mode(runner_and_env):
    runner, _, _ = runner_and_env
    with pytest.raises(ValueError, match="mode"):
        runner.make_act_fn("greedy")


def test_eval_diagnostics_carry_unsummed_per_agent_returns(runner_and_env):
    """``return_mean`` sums over agents and is structurally blind in 3 of 5
    regimes — g1's harvests cancel to ``-ε·(moves)`` exactly, g2/g3 reduce to one
    agent's harvest. Anything comparing arms on those regimes has to read the
    per-agent numbers, so they must actually reach the diagnostics file.
    """
    runner, env_fn, cfg = runner_and_env
    # evaluate() refuses to run on a lazily-built random model, because
    # evaluating noise returns plausible-looking numbers. This test asserts
    # plumbing, not quality, so opt out deliberately rather than by accident.
    runner._weights_source = "test:random-init (structure only)"
    try:
        runner.evaluate(env_fn, regime_grid=(1, 2), episodes=1)
    finally:
        runner._weights_source = None
    diag = runner._eval_diagnostics

    per_agent = diag["return_per_regime_planner_per_agent"]
    assert set(per_agent) == {1, 2}
    for g, vec in per_agent.items():
        assert len(vec) == cfg.env.N
        # the summed number the report carries must be the sum of these
        assert sum(vec) == pytest.approx(
            diag["return_per_regime_planner"][g], abs=1e-6)

    # g1 is zero-sum: the per-agent returns cancel, which is exactly why the
    # summed metric cannot see anything there.
    assert sum(per_agent[1]) == pytest.approx(0.0, abs=0.5)


# ---------------------------------------------------------------------------
# The assumption the planner equivalence rests on, pinned at the C++ boundary.
# ---------------------------------------------------------------------------

_N, _A, _SIMS, _SEED = 2, 6, 25, 7


def _search_element0(B, root_cover_mode, sampled_times):
    """Element 0's sampled actions + visit counts after a full search at batch B."""
    rng = np.random.default_rng(0)
    probs = np.empty((B, _N, _A), dtype=np.float32)
    probs[0] = np.array([[0.4, 0.1, 0.1, 0.2, 0.1, 0.1],
                         [0.1, 0.3, 0.2, 0.1, 0.2, 0.1]], dtype=np.float32)
    for b in range(1, B):                       # distinct neighbours: any
        r = rng.random((_N, _A)).astype(np.float32)   # cross-element RNG bleed
        probs[b] = r / r.sum(-1, keepdims=True)       # would show up as drift
    noises = np.full((B, _N, _A), 1.0 / _A, dtype=np.float32)
    zeros = np.zeros((B, _N), dtype=np.float32)

    t = cytree.Tree_batch(B, _N, _A, sampled_times, _SIMS, 0.01, _SEED,
                          0.75, 0.8, 1, root_cover_mode)
    t.prepare(zeros.reshape(-1), zeros.reshape(-1), probs.reshape(-1),
              probs.reshape(-1), sampled_times, 0.25, noises.reshape(-1))
    for i in range(_SIMS):
        t.batch_selection(19652.0, 1.25, 0.997)
        flat = np.full((B, _N, _A), 1.0 / _A, dtype=np.float32)
        vals = np.tile(np.arange(_N, dtype=np.float32), (B, 1)) * 0.1
        t.batch_expansion_and_backup(i + 1, 0.997, 5, zeros.reshape(-1),
                                     vals.reshape(-1), flat.reshape(-1),
                                     probs.reshape(-1))
    return (np.asarray(t.get_roots_sampled_actions()[0]).reshape(-1, _N),
            np.asarray(t.get_roots_sampled_visit_count()[0]).reshape(-1))


@pytest.mark.parametrize("mode,sampled_times", [(0, 5), (1, 1 + _N * _A)])
@pytest.mark.parametrize("B", [2, 4, 16])
def test_tree_search_is_batch_size_invariant(mode, sampled_times, B):
    """Element 0's search must not depend on how many trees share the batch.

    This is what lets the batch-1 act_fn stand in for ``_rollout_planner``'s
    lockstep batch. If it ever fails, ``make_act_fn("planner")`` stops being
    equivalent to ``evaluate()`` and every NashConv / cross-play number built on
    it is measuring a different policy.
    """
    a1, v1 = _search_element0(1, mode, sampled_times)
    aB, vB = _search_element0(B, mode, sampled_times)
    np.testing.assert_array_equal(a1, aB)
    np.testing.assert_array_equal(v1, vB)
