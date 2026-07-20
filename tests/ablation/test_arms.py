"""Ablation-arm wiring gates (phase-7).

The arm set + the two application seams (V4Config-level, fork-argv-level).
Fork construction under each arm is covered by the mazero_mixed smoke; this
locks the argv transforms so an arm can't silently become a no-op.
"""
from __future__ import annotations

import pytest

from hyper_mve.ablation.arms import ARMS, apply_arm, apply_arm_argv


_BASE_ARGV = [
    "--opr", "train_sync", "--case", "relation",
    "--subjective_model", "--decoupled_selection",
]


def test_arm_set_is_the_locked_five_plus_diagnostics():
    # The five thesis arms are user-locked (phase 7). ``oracle_belief`` is a
    # DIAGNOSTIC control (2026-07-20 harvest-collapse triage), not a thesis
    # ablation — it must never appear in the formal grid manifest.
    locked = {
        "point_estimate_leaf", "joint_selection", "no_subjective",
        "moe_router", "film",
    }
    assert locked <= set(ARMS)
    assert set(ARMS) - locked == {"oracle_belief"}


def test_unknown_arm_rejected():
    for bad in ("nope", "belief", "hyper", ""):
        with pytest.raises(ValueError):
            apply_arm_argv(list(_BASE_ARGV), bad)
    with pytest.raises(ValueError):
        apply_arm(object(), "nope")


def test_point_estimate_leaf_adds_flag():
    out = apply_arm_argv(list(_BASE_ARGV), "point_estimate_leaf")
    assert "--belief_point_estimate" in out
    assert "--subjective_model" in out          # still subjective
    assert "--decoupled_selection" in out


def test_joint_selection_drops_decoupled():
    out = apply_arm_argv(list(_BASE_ARGV), "joint_selection")
    assert "--decoupled_selection" not in out
    assert "--subjective_model" in out


def test_no_subjective_drops_subjective_model():
    out = apply_arm_argv(list(_BASE_ARGV), "no_subjective")
    assert "--subjective_model" not in out
    # plain MAZero keeps decoupled search off the subjective pathway; the
    # arm only removes the subjective head, nothing else added
    assert "--belief_point_estimate" not in out


@pytest.mark.parametrize("arm", ["moe_router", "film"])
def test_conditioning_arms_set_flag(arm):
    out = apply_arm_argv(list(_BASE_ARGV), arm)
    assert "--conditioning" in out
    assert out[out.index("--conditioning") + 1] == arm
    assert "--subjective_model" in out          # conditioning swap, not drop


def test_apply_arm_is_identity_at_cfg_level():
    sentinel = object()
    assert apply_arm(sentinel, "moe_router") is sentinel


def test_argv_transform_does_not_mutate_input():
    argv = list(_BASE_ARGV)
    apply_arm_argv(argv, "no_subjective")
    assert argv == _BASE_ARGV       # returned a new list
