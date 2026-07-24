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
    # ``mcts_fix*`` are DIAGNOSTIC arms (2026-07-20 prior-collapse fix): the fix
    # is OPT-IN because it is net-negative, so these enable it for continued
    # investigation and must never appear in the formal grid manifest.
    assert set(ARMS) - locked == {"oracle_belief", "mcts_fix", "mcts_fix_cover",
                                  "mcts_fix_qtarget", "mcts_fix_oracle", "ref_bc",
                                  "ref_bc_mcts_fix", "ref_bc_no_subjective",
                                  "ref_bc_belief_blind", "ref_bc_rw", "ref_bc_hardval",
                                  "ref_bc_anneal_scaled"}


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


# --- prior-collapse-fix controls -------------------------------------------
# These assert on the PARSED config, not on argv token membership. A value
# override works by appending a second occurrence of the flag and relying on
# argparse taking the last one; a token-presence check passes even when that
# override is broken (e.g. if _ARGV_REMOVE stripped the flag and orphaned its
# value), so it would not actually gate anything.

# the BASELINE fork argv, matching what runner.py now emits by default
_BASELINE_ARGV = [
    "--opr", "train_sync", "--case", "relation", "--env", "rel_duo",
    "--exp_name", "t", "--seed", "0",
    "--num_simulations", "25",
    "--sampled_action_times", "5",
    "--subjective_model", "--decoupled_selection",
]


def _parse(argv):
    import os
    import sys
    fork = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
        "hyper_mve", "algo", "mazero_mixed")
    if fork not in sys.path:
        sys.path.insert(0, fork)
    import core.config as core_config
    return core_config.parse_args(argv)


def test_default_is_baseline_not_the_fix():
    """The fix is net-negative, so the DEFAULT must be byte-for-byte upstream:
    prior-sampled root, visit-count target. New runs must reproduce the
    comparable 22.1 baseline, not the 13.3 fix."""
    a = _parse(list(_BASELINE_ARGV))
    assert a.root_cover == "none"
    assert a.sampled_action_times == 5
    assert a.policy_target_type == "visit"


def test_mcts_fix_enables_both_stages():
    a = _parse(apply_arm_argv(list(_BASELINE_ARGV), "mcts_fix"))
    assert a.root_cover == "star"
    assert a.sampled_action_times == 13      # >= 1 + num_agents*action_space_size
    assert a.leaf_sampled_times == 5         # leaves must NOT inherit the cover width
    assert a.policy_target_type == "q_softmax"


def test_mcts_fix_cover_is_stage_a_only():
    a = _parse(apply_arm_argv(list(_BASELINE_ARGV), "mcts_fix_cover"))
    assert a.root_cover == "star"
    assert a.sampled_action_times == 13
    assert a.policy_target_type == "visit"       # stage B stays off


def test_mcts_fix_qtarget_is_stage_b_only():
    a = _parse(apply_arm_argv(list(_BASELINE_ARGV), "mcts_fix_qtarget"))
    assert a.policy_target_type == "q_softmax"
    assert a.root_cover == "none"                # stage A stays off
    assert a.sampled_action_times == 5


def test_ref_bc_enables_behavior_cloning_and_extends_guidance():
    """The competence arm: BC toward the executed scripted action, plus a
    stronger/longer reference-episode schedule. The overrides must WIN over the
    base runner emission (argparse takes the last occurrence)."""
    base = list(_BASELINE_ARGV) + [
        "--reference_episode_prob_start", "0.4",     # what runner._build_game_config emits
        "--reference_episode_prob_end", "0.08",
        "--reference_episode_anneal_steps", "9000",
    ]
    a = _parse(apply_arm_argv(base, "ref_bc"))
    assert a.bc_loss_coeff == 1.0                     # BC on
    assert a.reference_episode_prob_start == 0.5      # override won over 0.4
    assert a.reference_episode_prob_end == 0.1
    assert a.reference_episode_anneal_steps == 28000  # extended guidance


def test_ref_bc_mcts_fix_is_the_union():
    """Competence follow-up: BC guidance AND the prior-collapse fix, so the
    de-collapsed prior can track the now-reliable value."""
    a = _parse(apply_arm_argv(list(_BASELINE_ARGV), "ref_bc_mcts_fix"))
    assert a.bc_loss_coeff == 1.0                    # BC on (from ref_bc)
    assert a.reference_episode_prob_start == 0.5     # guidance (from ref_bc)
    assert a.root_cover == "star"                    # de-collapse (from mcts_fix)
    assert a.sampled_action_times == 13
    assert a.policy_target_type == "q_softmax"


def test_ref_bc_no_subjective_is_the_regime_blind_control():
    """The learned regime-blind control: ref_bc competence with the subjective
    head removed."""
    a = _parse(apply_arm_argv(list(_BASELINE_ARGV), "ref_bc_no_subjective"))
    assert a.bc_loss_coeff == 1.0                    # BC on
    assert a.subjective_model is False               # regime-blind (no belief pathway)


def test_ref_bc_belief_blind_keeps_subjective_but_blinds_belief():
    """Capacity-matched control: full subjective architecture (same params) with
    a forced uniform posterior — isolates the belief posterior's marginal value."""
    a = _parse(apply_arm_argv(list(_BASELINE_ARGV), "ref_bc_belief_blind"))
    assert a.bc_loss_coeff == 1.0
    assert a.belief_blind is True
    assert a.subjective_model is True    # architecture kept; only the posterior is blinded


def test_ref_bc_rw_enables_reward_weighting():
    """Reward-weighted BC: same as ref_bc plus per-(regime,agent) return-to-go
    weighting of the BC target."""
    a = _parse(apply_arm_argv(list(_BASELINE_ARGV), "ref_bc_rw"))
    assert a.bc_loss_coeff == 1.0
    assert a.bc_reward_weighting is True


def test_ref_bc_hardval_enables_value_hard_select():
    a = _parse(apply_arm_argv(list(_BASELINE_ARGV), "ref_bc_hardval"))
    assert a.bc_loss_coeff == 1.0
    assert a.value_hard_select is True
    assert _parse(list(_BASELINE_ARGV)).value_hard_select is False   # off by default


def test_reward_weighting_is_off_by_default():
    assert _parse(list(_BASELINE_ARGV)).bc_reward_weighting is False
    assert _parse(apply_arm_argv(list(_BASELINE_ARGV), "ref_bc")).bc_reward_weighting is False


def test_bc_loss_is_off_by_default():
    """bc_loss_coeff defaults to 0.0 so every non-ref_bc config is a bit-exact
    no-op on the BC path."""
    assert _parse(list(_BASELINE_ARGV)).bc_loss_coeff == 0.0
    assert _parse(apply_arm_argv(list(_BASELINE_ARGV), "mcts_fix")).bc_loss_coeff == 0.0


def test_ref_bc_anneal_scaled_is_ref_bc_at_the_argv_seam():
    """At the apply_arm_argv seam alone (before runner.py's budget-aware
    override), ref_bc_anneal_scaled is byte-identical to ref_bc -- the scaling
    itself needs training_steps, which only MAZeroMixedRunner._build_game_config
    knows (see test_ref_bc_anneal_scaled_preserves_fraction_not_absolute_count
    in tests/algo/)."""
    a = _parse(apply_arm_argv(list(_BASELINE_ARGV), "ref_bc_anneal_scaled"))
    assert a.bc_loss_coeff == 1.0
    assert a.reference_episode_prob_start == 0.5
    assert a.reference_episode_anneal_steps == 28000  # placeholder; runner.py overrides it


def test_mcts_fix_oracle_is_fix_plus_pinned_belief():
    a = _parse(apply_arm_argv(list(_BASELINE_ARGV), "mcts_fix_oracle"))
    assert a.root_cover == "star" and a.policy_target_type == "q_softmax"
    # belief pinned to the true regime for the whole run
    assert a.belief_oracle_steps >= 100000000
    assert a.belief_anneal_steps == 0


def test_target_guard_is_not_enabled_in_any_shipped_config():
    """TRIPWIRE. Stage C (the flat-Q guard) is implemented but deliberately not
    enabled: the approved sequencing is measure A+B first. It defaults to 0 in
    both the baseline and every mcts_fix arm.

    When stage C is turned on, whoever enables it must revisit this test AND
    add the ~0.4 threshold (per the advantage-scale probe: relation/g1 sits at
    ~0.29 of the batch-pooled spread) to the relevant arm.
    """
    import pathlib
    runner = pathlib.Path(__file__).resolve().parents[2] / "hyper_mve" / "algo" / "runner.py"
    assert "--policy_target_min_qstd" not in runner.read_text(encoding="utf-8"), (
        "runner.py now enables the flat-Q guard — update this tripwire."
    )
    for argv in (list(_BASELINE_ARGV),
                 apply_arm_argv(list(_BASELINE_ARGV), "mcts_fix")):
        a = _parse(argv)
        assert a.policy_target_min_qstd == 0.0    # guard off
        assert a.policy_target_renorm_cap == 4.0  # rescale cap still armed


def _validators():
    _parse(list(_BASELINE_ARGV))         # ensures the fork is importable
    import core.config as core_config
    return core_config.validate_root_cover, core_config.root_cover_size


def test_root_cover_rejects_too_narrow_buffer():
    """The cover can build 1 + N*A children and every one must fit the replay
    buffer width; exceeding it otherwise raises deep inside the reanalyze
    worker, far from the cause."""
    validate, size = _validators()
    assert size(2, 6) == 13
    with pytest.raises(AssertionError, match="sampled_action_times"):
        validate(1, 5, 25, 2, 6)         # buffer width 5 < cover 13
    validate(1, 13, 25, 2, 6)            # the shipped configuration is legal


def test_root_cover_rejects_too_few_simulations():
    """Every root child needs its forced round-robin simulation, else the tail
    of the cover is enumerated but never evaluated."""
    validate, _ = _validators()
    with pytest.raises(AssertionError, match="num_simulations"):
        validate(1, 13, 8, 2, 6)         # 8 sims < cover 13


def test_root_cover_validation_is_inert_when_disabled():
    """root_cover_off must not inherit the cover's sizing constraints."""
    validate, _ = _validators()
    validate(0, 5, 25, 2, 6)             # upstream settings, no raise
