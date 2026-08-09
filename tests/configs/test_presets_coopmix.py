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

    # one cooperative, one mixed, one independent
    assert trained == ("mutual_coop", "asym_exploit", "neutral")
    assert held == ("asym_exploited", "asym_exploit_mild")

    # ...and the same literal tuple resolves to something else under g2, which
    # is the trap this test exists to catch.
    recip_fam = get_regime_family(V4Config.from_preset("rel_recip_holdout").env)
    assert tuple(recip_fam.names()[g] for g in (0, 1, 4)) == (
        "mutual_coop", "mutual_comp", "neutral")


def test_rel_coopmix_holdout_training_set_covers_every_own_row_value():
    """Coverage claim in the preset docstring, pinned.

    Every own-row value an agent can observe (+lam, -lam, 0) must occur among
    the trained regimes, or the model meets an unseen observation rather than an
    unseen partner row at eval time.
    """
    from hyper_mve.utils.schemas.relation import get_regime_family

    hold = V4Config.from_preset("rel_coopmix_holdout")
    fam = get_regime_family(hold.env)
    seen_rows = {float(fam.regimes[g].w_array()[0, 1])
                 for g in hold.env.train_regime_ids}
    all_rows = {float(r.w_array()[0, 1]) for r in fam.regimes}
    assert seen_rows == all_rows == {+1.0, -1.0, 0.0}


@pytest.mark.parametrize("name", ["rel_coopmix", "rel_coopmix_holdout"])
def test_rel_coopmix_presets_round_trip_through_from_preset(name):
    import json

    cfg = V4Config.from_preset(name)
    assert cfg.preset_name == name
    json.dumps(cfg.to_dict())          # must stay JSON-serialisable
