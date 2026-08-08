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
    make_relational_greedy_policy,
    make_scripted_greedy_policy,
    noop_policy,
    reference_policy_suite,
)
from hyper_mve.utils.schemas.relation import get_regime_family

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


# --------------------------------------------------------------------------
# What regime knowledge is worth (2026-08-05). These pin the structural facts
# behind the decision to stop treating the belief posterior as a return lever;
# see results/analysis/regime_knowledge_ceiling.md.
# --------------------------------------------------------------------------


def _make_levels(cfg):
    """``(self_info, oracle)`` controllers configured for this env's physics.

    The coupling MUST come from the env: which entry of W sets an agent's regard
    for its neighbour is exactly what v6 changes, and a controller given the
    wrong rule optimises the wrong objective. Likewise the conservation
    threshold — under constant regrowth restraint is strictly harmful, under
    logistic it is where most of the regime's value lives.
    """
    kw = dict(
        coupling=cfg.env.reward_coupling,
        reciprocity_lambda=cfg.env.reciprocity_lambda,
        conserve_threshold=0.3 if cfg.env.regrowth_law == "logistic" else 0.0,
    )
    return (
        make_relational_greedy_policy(
            cfg.env.N, cfg.env.K, level="self_info", **kw),
        make_relational_greedy_policy(
            cfg.env.N, cfg.env.K, level="oracle",
            family=get_regime_family(cfg.env), **kw),
    )


def _levels(cfg, env_fn, episodes=8):
    si, orc = _make_levels(cfg)
    return (
        evaluate_reference_policy(env_fn, si, GRID, episodes),
        evaluate_reference_policy(env_fn, orc, GRID, episodes),
    )


def _unilateral_effect(cfg, env_fn, episodes=8):
    """Mean |Δ agent 0's own return| on g2/g3 when agent 0 alone learns the truth.

    Unilateral because the value of information is what ONE agent gains by
    deviating while the others hold still — hand both agents the oracle and in a
    social dilemma the team return can fall while each agent's own objective
    rises. Absolute value because this compares two fixed heuristics, so it is
    not a VoI and is not bounded below by zero; the sign says whether the
    heuristic responds well, the magnitude says whether the regime matters.
    """
    si, orc = _make_levels(cfg)
    partner, _ = _make_levels(cfg)
    base = evaluate_reference_policy(env_fn, si, GRID, episodes,
                                     policies=[si, partner])
    dev = evaluate_reference_policy(env_fn, orc, GRID, episodes,
                                    policies=[orc, partner])
    per = [abs(dev["return_per_agent_per_regime"][g][0]
               - base["return_per_agent_per_regime"][g][0]) for g in (2, 3)]
    return sum(per) / len(per)


def test_oracle_matches_self_info_exactly_on_symmetric_regimes(rel_duo):
    """g0/g1/g4 have w_j· == w_i·, so the mirror prior is already the truth.

    The two controllers are identical apart from that one scalar, so agreeing
    bit-for-bit here is what proves the oracle channel changes *only* what it
    is supposed to. Any drift means set_regime is leaking somewhere else.
    """
    cfg, env_fn = rel_duo
    si, orc = _levels(cfg, env_fn)
    for g in (0, 1, 4):
        assert orc["return_per_regime"][g] == pytest.approx(
            si["return_per_regime"][g], abs=1e-9
        ), f"oracle and self_info diverged on symmetric regime g{g}"


def test_knowing_the_opponent_row_is_worth_little(rel_duo):
    """The measured ceiling on regime knowledge, as a regression guard.

    At 64 episodes the gap is +0.81 mean / +2.04 over g2-g3 — the same size as
    the trained model's oracle gap (UB - A1 = +0.67 +/- 2.88) and well inside
    the +/-2.29 seed sd of the method. That is why the belief posterior cannot
    pay for itself on rel_duo at any architecture.

    This asserts the *finding*, not the exact number. If someone raises the
    coupling (scarcer resources, larger lambda, more agents per cell) this test
    should fail — that is the signal that the belief channel became worth
    building, and the analysis doc needs revisiting.
    """
    cfg, env_fn = rel_duo
    si, orc = _levels(cfg, env_fn)
    gap = orc["return_mean"] - si["return_mean"]
    assert -1.0 < gap < 5.0, (
        f"regime-knowledge gap {gap:.2f} left the measured band; the env's "
        "strategic coupling changed and results/analysis/regime_knowledge_"
        "ceiling.md no longer applies"
    )


@pytest.fixture(scope="module")
def rel_recip():
    cfg = V4Config.from_preset("rel_recip")

    def env_fn():
        return RelationCommonsPettingZooEnv(
            cfg.env, oracle_mode=False, eval_info_mode=False
        )

    return cfg, env_fn


def test_v6_makes_the_hidden_row_matter(rel_duo, rel_recip):
    """The whole point of rel_recip, as a single controlled comparison.

    Same agents, same regimes, same controller — only ``reward_coupling``,
    ``regrowth_law`` and the grid differ. On rel_duo, knowing the hidden half of
    W barely moves agent 0's own return on the asymmetric regimes; on rel_recip
    it moves it by an order of magnitude more.

    If this fails, either rel_recip drifted off its design point or rel_duo
    stopped being the null control — check ``scripts/probes/regime_voi_probe.py``
    on both presets before touching the test.
    """
    duo_effect = _unilateral_effect(*rel_duo)
    recip_effect = _unilateral_effect(*rel_recip)
    assert duo_effect < 3.0, (
        f"rel_duo effect {duo_effect:.2f} is no longer ~null; it is the control "
        "for the whole regime_knowledge_ceiling analysis"
    )
    assert recip_effect > 5.0, (
        f"rel_recip effect {recip_effect:.2f} fell below the design gate"
    )
    assert recip_effect > 3 * duo_effect


def test_team_return_is_structurally_capped_per_regime(rel_duo):
    """``R_i = (u_i + w_ij u_j)/(1+|w_ij|)`` fixes the team return per regime.

    Summing over agents: g1 cancels to exactly 0, g2/g3 collapse to a single
    agent's harvest, g0/g4 keep the full pair. So the headline metric is
    ~0 / ~half / ~full by construction, independent of how well any agent
    infers the regime — the reason `return_mean` can barely respond to belief
    quality.
    """
    cfg, env_fn = rel_duo
    _, orc = _levels(cfg, env_fn)
    pr = orc["return_per_regime"]

    # g1 is zero-sum: only the epsilon move cost survives the cancellation.
    assert -1.0 < pr[1] <= 0.0

    # g2/g3 are one agent's harvest; g0/g4 are both. Generous band because the
    # two halves are different episodes under the 97*g+ep seeding.
    full = (pr[0] + pr[4]) / 2.0
    for g in (2, 3):
        assert 0.35 * full < pr[g] < 0.65 * full, (
            f"g{g}={pr[g]:.1f} is not ~half the g0/g4 ceiling {full:.1f}"
        )
