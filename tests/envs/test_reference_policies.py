"""Reference-policy scale for RelationCommons (``rel_duo``).

These tests exist to make a collapsed policy fail loudly. The 2026-07-18 grid
trained four architecturally different mazero_mixed variants for ~7 GPU-hours
each and every one reported ``return_mean = 15.10890765041113`` — which is
exactly what a constant-HARVEST policy scores. Nothing in the suite could say
so, because the repo had no reference scale at all.

``test_harvest_only_reproduces_the_collapse_number`` pins that constant so the
signature is recognizable on sight; the ordering test pins the scale a learned
policy has to be read against.
"""
from __future__ import annotations

import pytest

from hyper_mve.utils.configs import V4Config
from hyper_mve.envs.adapters.pettingzoo_wrapper import RelationCommonsPettingZooEnv
from hyper_mve.envs.relation_commons.reference_policies import (
    evaluate_reference_policy,
    harvest_only_policy,
    make_scripted_greedy_policy,
    noop_policy,
    reference_policy_suite,
)

GRID = (0, 1, 2, 3, 4)
EPISODES = 16

# The exact float the collapsed grid produced. Bit-exact because the eval env
# is deterministic under the canonical per-episode seeding and the policy is a
# constant action.
COLLAPSE_RETURN_MEAN = 15.10890765041113
COLLAPSE_PER_REGIME = {
    0: 23.244473308324814,
    1: 0.0,
    2: 17.43335498124361,
    3: 11.622236654162407,
    4: 23.244473308324814,
}


@pytest.fixture(scope="module")
def rel_duo():
    cfg = V4Config.from_preset("rel_duo")

    def env_fn():
        return RelationCommonsPettingZooEnv(
            cfg.env, oracle_mode=False, eval_info_mode=False
        )

    return cfg, env_fn


def test_noop_scores_exactly_zero(rel_duo):
    _, env_fn = rel_duo
    r = evaluate_reference_policy(env_fn, noop_policy, GRID, 4)
    assert r["return_mean"] == 0.0
    assert all(v == 0.0 for v in r["return_per_regime"].values())


def test_harvest_only_reproduces_the_collapse_number(rel_duo):
    """Constant HARVEST == the number every arm of the 2026-07-18 grid reported.

    If this drifts, the env's physics or the eval seeding changed and the
    archived grid results are no longer comparable to new ones.
    """
    _, env_fn = rel_duo
    r = evaluate_reference_policy(env_fn, harvest_only_policy, GRID, EPISODES)

    assert r["return_mean"] == pytest.approx(COLLAPSE_RETURN_MEAN, abs=1e-9)
    for g, expected in COLLAPSE_PER_REGIME.items():
        assert r["return_per_regime"][g] == pytest.approx(expected, abs=1e-9)

    # The whole action budget goes to HARVEST (index 5) and nothing moves,
    # which is why the competitive regime lands on exactly 0.
    assert r["action_fractions"][5] == 1.0
    assert r["return_per_regime"][1] == 0.0


def test_scripted_greedy_beats_the_collapse_by_a_wide_margin(rel_duo):
    """A 20-line heuristic must be worth many times the degenerate attractor."""
    cfg, env_fn = rel_duo
    policy = make_scripted_greedy_policy(
        cfg.env.N, cfg.env.K, distinct_targets=True
    )
    r = evaluate_reference_policy(env_fn, policy, GRID, EPISODES)

    assert r["return_mean"] > 90.0
    # Cooperative and neutral both approach the ~186 two-agent ceiling.
    assert r["return_per_regime"][0] > 150.0
    assert r["return_per_regime"][4] > 150.0
    # Competitive is zero-sum by construction; movement cost keeps it just
    # below zero rather than at it.
    assert -1.0 < r["return_per_regime"][1] <= 0.0


def test_reference_suite_is_strictly_ordered(rel_duo):
    """noop < random < harvest_only < scripted_greedy — the interpretive scale."""
    cfg, env_fn = rel_duo
    means = {
        name: evaluate_reference_policy(env_fn, pol, GRID, 4)["return_mean"]
        for name, pol in reference_policy_suite(cfg.env.N, cfg.env.K).items()
    }
    assert means["noop"] < means["random"] < means["harvest_only"]
    assert means["harvest_only"] < means["scripted_greedy"]
    assert means["scripted_greedy"] < means["scripted_greedy_distinct"]
