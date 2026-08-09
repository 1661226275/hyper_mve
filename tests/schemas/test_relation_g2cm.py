"""Contracts for the ``g2cm`` regime family (2026-08-09 scope change).

``g2cm`` is ``g2`` with ``mutual_comp`` — the only purely adversarial regime —
removed, and ``asym_exploit_mild`` added in its place. These tests pin the three
things that make it correct rather than merely different:

* nothing in it is zero-sum, which is the point of the change;
* ``mutual_coop`` keeps id 0 and ``neutral`` keeps id 4, which several consumers
  depend on positionally;
* **every own-row observation stays ambiguous**, which is what stops the removal
  from silently halving value of information.

``g2`` itself is asserted to be untouched throughout — this family is additive.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest

from hyper_mve.utils.schemas import build_g2, compute_relational_rewards
from hyper_mve.utils.schemas.relation import build_g2cm, effective_coupling_matrix


def _reward(u, W, coupling="reciprocal"):
    return compute_relational_rewards(
        harvests=np.asarray(u, dtype=np.float32),
        moved_mask=np.zeros(len(u), dtype=bool),
        W=W, epsilon_move=0.0, coupling=coupling,
    )


def test_g2cm_exact_w_values():
    fam = build_g2cm(lam=1.0)
    assert fam.size == 5
    assert fam.N == 2
    assert fam.names() == (
        "mutual_coop", "asym_exploit", "asym_exploited", "asym_exploit_mild",
        "neutral",
    )
    expected = {
        "mutual_coop": [[1.0, 1.0], [1.0, 1.0]],
        "asym_exploit": [[1.0, -1.0], [1.0, 1.0]],       # agent 0 hostile, 1 supportive
        "asym_exploited": [[1.0, 1.0], [-1.0, 1.0]],     # mirror
        "asym_exploit_mild": [[1.0, -1.0], [0.0, 1.0]],  # agent 0 hostile, 1 indifferent
        "neutral": [[1.0, 0.0], [0.0, 1.0]],
    }
    for reg in fam.regimes:
        np.testing.assert_array_equal(
            reg.w_array(), np.array(expected[reg.name], np.float32))


def test_g2cm_intensity_scales_off_diagonal_only():
    fam = build_g2cm(lam=0.5)
    for reg in fam.regimes:
        W = reg.w_array()
        assert W[0, 0] == W[1, 1] == 1.0
        assert set(np.abs(W[~np.eye(2, dtype=bool)]).tolist()) <= {0.0, 0.5}


def test_g2cm_has_no_purely_adversarial_regime():
    """The point of the family: no regime may have BOTH weights negative.

    Both-negative is zero-sum, which changes the nature of the game and makes no
    single training objective correct across the family.
    """
    for reg in build_g2cm(lam=1.0).regimes:
        W = reg.w_array()
        assert not (W[0, 1] < 0 and W[1, 0] < 0), (
            f"{reg.name} has both weights negative -- that is a zero-sum regime")

    # ...and g2 is untouched, still carrying exactly the one that was removed.
    both_neg = [r.name for r in build_g2(lam=1.0).regimes
                if r.w_array()[0, 1] < 0 and r.w_array()[1, 0] < 0]
    assert both_neg == ["mutual_comp"]


def test_g2cm_keeps_mutual_coop_at_0_and_neutral_at_4():
    """Load-bearing ids, not cosmetics.

    The empirical price-of-anarchy denominator reads ``welfare_physical["0"]`` as
    the all-cooperative regime, and probes that need ``W = 0`` to read physical
    harvests back pin the zero-coupling regime. Both must keep agreeing with g2.
    """
    cm, g2 = build_g2cm(lam=1.0), build_g2(lam=1.0)
    assert cm.names()[0] == g2.names()[0] == "mutual_coop"
    assert cm.names()[4] == g2.names()[4] == "neutral"
    np.testing.assert_array_equal(cm.regimes[4].w_array(), np.eye(2, dtype=np.float32))


def test_g2cm_every_own_row_observation_stays_ambiguous():
    """The test that guards the reason ``asym_exploit_mild`` exists at all.

    Under ``reciprocal`` coupling an agent observes its own row ``w_01`` and must
    infer the hidden ``w_10``. Value of information comes ONLY from observations
    consistent with more than one regime, so an own-row value matched by exactly
    one regime contributes a hard zero. Removing ``mutual_comp`` without a
    replacement strands ``w_01 = -lam`` as such a singleton and halves measured
    VoI — 3.99 on g2 versus 2.00 for the bare removal, both measured by
    ``scripts/probes/regime_voi_probe.py``. This pins the structure so a future
    regime edit cannot quietly do it again.
    """
    buckets = defaultdict(set)
    for reg in build_g2cm(lam=1.0).regimes:
        W = reg.w_array()
        buckets[float(W[0, 1])].add(float(W[1, 0]))

    assert buckets[+1.0] == {+1.0, -1.0}, "the +lam branch lost its ambiguity"
    assert buckets[-1.0] == {+1.0, 0.0}, "the -lam branch lost its ambiguity"
    # The 0 branch is a KNOWN singleton and the remaining headroom: a partner
    # would need w01 = 0 with w10 != 0, which is admissible (only one negative
    # weight) but was deliberately out of scope for this change.
    assert buckets[0.0] == {0.0}


def test_g2cm_asym_exploit_mild_is_not_blind_to_summed_return():
    """``mutual_comp`` made ``R_0 + R_1`` cancel regardless of policy.

    Its replacement must not: under reciprocal coupling ``asym_exploit_mild``
    gives ``ŵ_01 = w_10 = 0`` and ``ŵ_10 = w_01 = -lam``, so the pair sums to
    ``(u_0 + u_1)/2`` and still responds to both agents' harvests.
    """
    fam = build_g2cm(lam=1.0)
    mild = fam.regimes[3]
    assert mild.name == "asym_exploit_mild"

    W_hat = effective_coupling_matrix(mild.w_array(), "reciprocal", 0.0)
    assert W_hat[0, 1] == 0.0 and W_hat[1, 0] == -1.0

    u = [4.0, 6.0]
    r = _reward(u, mild.w_array())
    np.testing.assert_allclose(r, [4.0, 1.0], rtol=1e-6)     # u0 , (u1-u0)/2
    assert float(r.sum()) == pytest.approx(sum(u) / 2.0)

    # The contrast that motivated the removal: g2's mutual_comp sums to exactly
    # zero for ANY harvests, so summed return cannot see policy there at all.
    comp = build_g2(lam=1.0).regimes[1]
    assert comp.name == "mutual_comp"
    assert float(_reward(u, comp.w_array()).sum()) == pytest.approx(0.0)


def test_asymmetric_ids_are_derived_not_written_down():
    """(2, 3) under g2 but (1, 2, 3) under g2cm — which is why no caller may
    hardcode them."""
    assert build_g2(lam=1.0).asymmetric_ids() == (2, 3)
    assert build_g2(lam=1.0).symmetric_ids() == (0, 1, 4)
    assert build_g2cm(lam=1.0).asymmetric_ids() == (1, 2, 3)
    assert build_g2cm(lam=1.0).symmetric_ids() == (0, 4)


def test_g2cm_is_registered_and_resolves_through_env_cfg():
    from hyper_mve.utils.configs.env_config import EnvConfig
    from hyper_mve.utils.schemas import get_regime_family

    cfg = EnvConfig(N=2, L=3, K=2, T_max=100, relation_family="g2cm")
    fam = get_regime_family(cfg)
    assert fam.name == "g2cm" and fam.size == 5
