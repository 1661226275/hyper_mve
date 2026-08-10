"""Contracts for the ``rel_coopmix`` / ``rel_coopmix_holdout`` presets.

Kept separate from ``test_presets.py`` so the v5/v6 preset assertions there stay
literally untouched — this scope change is additive, and a diff on that file
would mean it was not.
"""
from __future__ import annotations

from dataclasses import fields, replace

import pytest

from hyper_mve.utils.configs import V4Config


def test_rel_coopmix_preset_table():
    cfg = V4Config.from_preset("rel_coopmix")
    assert cfg.preset_name == "rel_coopmix"
    env = cfg.env
    assert env.N == 2
    assert env.L == 3
    assert env.K == 2
    assert env.relation_family == "g2cm"
    assert env.relation_intensity == 1.0
    assert env.regime_switch_prob == 0.0
    assert env.regime_kernel == "uniform"
    assert env.alpha == 0.30
    assert env.regrowth_law == "logistic"
    assert env.reward_coupling == "reciprocal"
    assert env.train_regime_ids is None          # full family


def test_rel_coopmix_is_rel_recip_with_only_the_family_changed():
    """The controlled comparison. If any other physics knob drifts, the two
    presets stop being a one-variable contrast and no measurement against
    rel_recip means anything."""
    recip = V4Config.from_preset("rel_recip").env
    coopmix = V4Config.from_preset("rel_coopmix").env
    differing = [f.name for f in fields(recip)
                 if getattr(recip, f.name) != getattr(coopmix, f.name)]
    assert differing == ["relation_family"]


def test_rel_coopmix_holdout_trains_on_one_regime_of_each_character():
    """Asserted on RESOLVED NAMES, never on the id tuple.

    ``train_regime_ids`` is ``(0, 1, 4)`` here, which is byte-identical to
    ``rel_recip_holdout``'s value while meaning something different: id 1 is
    ``mutual_comp`` under ``g2`` and ``asym_exploit`` under ``g2cm``. An
    ``== (0, 1, 4)`` assertion would pass against both and catch neither.
    """
    from hyper_mve.utils.schemas.relation import get_regime_family

    base = V4Config.from_preset("rel_coopmix")
    hold = V4Config.from_preset("rel_coopmix_holdout")
    assert hold.preset_name == "rel_coopmix_holdout"
    assert hold.env == replace(base.env, train_regime_ids=hold.env.train_regime_ids)

    fam = get_regime_family(hold.env)
    trained = tuple(fam.names()[g] for g in hold.env.train_regime_ids)
    held = tuple(fam.names()[g] for g in range(fam.size)
                 if g not in hold.env.train_regime_ids)

    # cooperative + BOTH asymmetric mirrors; the mild form and the independent
    # regime are held out.
    assert trained == ("mutual_coop", "asym_exploit", "asym_exploited")
    assert held == ("asym_exploit_mild", "neutral")

    # ...and the tuple this preset used to carry, (0, 1, 4), resolves to
    # something else again under g2 -- the trap this test exists to catch.
    recip_fam = get_regime_family(V4Config.from_preset("rel_recip_holdout").env)
    assert tuple(recip_fam.names()[g] for g in (0, 1, 4)) == (
        "mutual_coop", "mutual_comp", "neutral")


def test_rel_coopmix_holdout_two_held_regimes_probe_different_things():
    """The two held-out regimes are NOT one uniform generalization claim.

    Under ``train_regime_ids=(0, 1, 2)`` training covers own-row values
    ``{+lam, -lam}`` but never ``0``. So:

    * g4 ``neutral`` presents an own-row VALUE never trained on -- the sharp,
      out-of-distribution test;
    * g3 ``asym_exploit_mild`` presents an own-row value that IS familiar
      (``-lam``, from g1) in an unseen COMBINATION (partner row ``0`` rather
      than ``+lam``) -- the milder, in-distribution test.

    Reporting them as a single "held-out score" averages two different
    questions, which is what this test exists to prevent. The predecessor of
    this test asserted the training set covered all three own-row values; that
    was true of the old ``(0, 1, 4)`` partition and is deliberately false now.
    """
    from hyper_mve.utils.schemas.relation import get_regime_family

    hold = V4Config.from_preset("rel_coopmix_holdout")
    fam = get_regime_family(hold.env)
    names = fam.names()
    trained_ids = hold.env.train_regime_ids

    def own_row(g):
        return float(fam.regimes[g].w_array()[0, 1])

    def partner_row(g):
        return float(fam.regimes[g].w_array()[1, 0])

    seen_own = {own_row(g) for g in trained_ids}
    assert seen_own == {+1.0, -1.0}, "training must not cover the zero own-row"

    held = [g for g in range(fam.size) if g not in trained_ids]
    by_name = {names[g]: g for g in held}
    assert set(by_name) == {"asym_exploit_mild", "neutral"}

    # g4: unseen own-row value -> the model has never observed this row at all.
    g_neutral = by_name["neutral"]
    assert own_row(g_neutral) == 0.0
    assert own_row(g_neutral) not in seen_own

    # g3: seen own-row value, unseen pairing. Pin the source of the familiarity
    # (g1) so the claim cannot silently rot if the family is reordered.
    g_mild = by_name["asym_exploit_mild"]
    assert own_row(g_mild) in seen_own
    assert own_row(g_mild) == own_row(names.index("asym_exploit"))
    seen_pairs = {(own_row(g), partner_row(g)) for g in trained_ids}
    assert (own_row(g_mild), partner_row(g_mild)) not in seen_pairs


@pytest.mark.parametrize("name", ["rel_coopmix", "rel_coopmix_holdout"])
def test_rel_coopmix_presets_round_trip_through_from_preset(name):
    import json

    cfg = V4Config.from_preset(name)
    assert cfg.preset_name == name
    json.dumps(cfg.to_dict())          # must stay JSON-serialisable
