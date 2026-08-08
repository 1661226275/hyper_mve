"""Ablation arms (phase 7) — OUR method's variations, evaluated on the same
3 metrics as the main method (fidelity / global reward / generalization).

An arm is a named transformation applied at two seams:

* ``apply_arm(cfg, arm)`` — V4Config-level overrides (scripts/train.py calls
  this while building the run config). Currently every arm is argv-level,
  so this validates the name and returns the cfg unchanged.
* ``apply_arm_argv(argv, arm)`` — fork-argv overrides consumed by
  ``MAZeroMixedRunner._build_game_config`` (the arms toggle fork flags).

Arms (user-locked set):
  * ``point_estimate_leaf`` — leaf values from the single posterior-blended
    head instead of the Bayes average over the regime family
    (``--belief_point_estimate``).
  * ``joint_selection``     — joint team-UCB child selection instead of the
    vectorized decoupled per-agent selection (drop ``--decoupled_selection``).
  * ``no_subjective``       — plain MAZero (drop ``--subjective_model``):
    the cooperative-sanity arm; no belief / hypernet / subjective heads.
  * ``moe_router`` / ``film`` — θ-generation mechanism swaps for the
    DualHyperNetwork on the SAME ctx input (``--conditioning <arm>``):
    the controlled Direction-1 mechanism comparison. (Distinct from the
    ``m3w_adapted`` BASELINE, which is a different training paradigm.)
  * ``oracle_belief``       — diagnostic control: belief pinned to the oracle
    one-hot g for the entire training run (removes belief quality as a
    confound; eval remains oracle-free, so ``regime_accuracy`` still
    measures the concurrently-trained BeliefNet).
"""
from __future__ import annotations

ARMS: tuple[str, ...] = (
    "point_estimate_leaf",
    "joint_selection",
    "no_subjective",
    "moe_router",
    "film",
    "oracle_belief",
    "mcts_fix",
    "mcts_fix_cover",
    "mcts_fix_qtarget",
    "mcts_fix_oracle",
    "ref_bc",
    "ref_bc_mcts_fix",
    "ref_bc_no_subjective",
    "ref_bc_belief_blind",
    "ref_bc_rw",
    "ref_bc_hardval",
    "ref_bc_anneal_scaled",
    "ref_bc_anneal_scaled_no_subjective",
    "ref_bc_anneal_scaled_decoupled",
    "ref_bc_anneal_scaled_hardval_decoupled",
    "ref_bc_anneal_scaled_hardval",
    "ref_bc_anneal_scaled_no_subjective_decoupled",
    "ref_bc_anneal_scaled_hardval_decoupled_big",
    "ref_bc_anneal_scaled_hardval_decoupled_big2",
    "ref_bc_anneal_scaled_hardval_decoupled_lrctl",
    # v6 policy-target 2x2 (the fourth cell is the bare method of record)
    "ref_bc_anneal_scaled_hardval_decoupled_cover",
    "ref_bc_anneal_scaled_hardval_decoupled_qtarget",
    "ref_bc_anneal_scaled_hardval_decoupled_mctsfix",
    # v6 visit_q_blend tau sweep (replaces q_softmax; see below)
    "ref_bc_anneal_scaled_hardval_decoupled_blend_t025",
    "ref_bc_anneal_scaled_hardval_decoupled_blend_t05",
    "ref_bc_anneal_scaled_hardval_decoupled_blend_t1",
    "ref_bc_anneal_scaled_hardval_decoupled_blend_t2",
)

# arm -> (flags to add, flags to remove); value-flags are (name, value) adds.
_ARGV_ADD: dict[str, tuple[str, ...]] = {
    "point_estimate_leaf": ("--belief_point_estimate",),
    "joint_selection": (),
    "no_subjective": (),
    "moe_router": ("--conditioning", "moe_router"),
    "film": ("--conditioning", "film"),
    "oracle_belief": ("--belief_oracle_steps", "100000000",
                      "--belief_anneal_steps", "0"),
    # 2026-07-20 prior-collapse fix, OPT-IN (the default is byte-for-byte
    # upstream because a budget-matched A/B showed the full fix REDUCES return
    # 22.1 -> 13.3 — it de-collapses the prior but the value head is unreliable
    # in the asymmetric regimes, so the policy follows it into bad actions
    # there). These arms enable the fix for continued investigation.
    #   mcts_fix          — both stages: root-cover enumeration + Q-softmax target.
    #   mcts_fix_cover    — stage A only (enumeration; target stays visit-count).
    #   mcts_fix_qtarget  — stage B only (Q target; root stays prior-sampled).
    #   mcts_fix_oracle   — full fix with belief pinned to the true regime; the
    #                       diagnostic that localizes the g2/g3 failure to the
    #                       belief net (if it recovers) vs the value head (if
    #                       not).
    # NOTE: value-flag overrides must be ADDED, never removed. _ARGV_REMOVE
    # filters by exact token, so removing "--sampled_action_times" would orphan
    # its "5" and argparse would reject the bare positional. argparse takes the
    # LAST occurrence of a repeated optional, so appending wins.
    "mcts_fix": ("--root_cover", "star", "--sampled_action_times", "13",
                 "--leaf_sampled_times", "5", "--policy_target_type", "q_softmax"),
    "mcts_fix_cover": ("--root_cover", "star", "--sampled_action_times", "13",
                       "--leaf_sampled_times", "5"),
    "mcts_fix_qtarget": ("--policy_target_type", "q_softmax",),
    "mcts_fix_oracle": ("--root_cover", "star", "--sampled_action_times", "13",
                        "--leaf_sampled_times", "5", "--policy_target_type", "q_softmax",
                        "--belief_oracle_steps", "100000000", "--belief_anneal_steps", "0"),
    # 2026-07-21 competence fix. The scripted-reference bootstrap was already on
    # by default (runner emits 0.4->0.08) but the policy still collapsed: the
    # reanalyze policy target is the collapsible MCTS visit distribution, so the
    # demonstrated actions never became a policy target (confirmed:
    # workers/eps_reward_max ~91 = demos reached replay, mean stayed flat ~10).
    # This arm (a) adds a behavior-cloning CE toward the executed scripted action
    # on reference steps, and (b) strengthens + EXTENDS the guidance (0.5->0.1
    # over 28000 train steps ~= 3/4 of a 600k-env-step run). Overrides win because
    # argparse takes the last occurrence of a repeated optional.
    "ref_bc": ("--bc_loss_coeff", "1.0",
               "--reference_episode_prob_start", "0.5",
               "--reference_episode_prob_end", "0.1",
               "--reference_episode_anneal_steps", "28000"),
}
_ARGV_REMOVE: dict[str, tuple[str, ...]] = {
    "point_estimate_leaf": (),
    "joint_selection": ("--decoupled_selection",),
    "no_subjective": ("--subjective_model",),
    "moe_router": (),
    "film": (),
    "oracle_belief": (),
    "mcts_fix": (),
    "mcts_fix_cover": (),
    "mcts_fix_qtarget": (),
    "mcts_fix_oracle": (),
    "ref_bc": (),
}

# 2026-07-21 competence follow-ups, composed from the single-arm definitions so
# they can't drift. ref_bc lifted return 22.1->66.6 but only via the SEARCH
# (distilled prior still 15.7); these two isolate the remaining questions:
#   ref_bc_mcts_fix       — de-collapse the prior now that the value is reliable
#                           (mcts_fix was net-negative on the OLD bad value).
#   ref_bc_no_subjective  — the regime-BLIND learned control: does the subjective
#                           architecture beat it at equal competence?
_ARGV_ADD["ref_bc_mcts_fix"] = _ARGV_ADD["ref_bc"] + _ARGV_ADD["mcts_fix"]
_ARGV_ADD["ref_bc_no_subjective"] = _ARGV_ADD["ref_bc"] + _ARGV_ADD["no_subjective"]
_ARGV_REMOVE["ref_bc_mcts_fix"] = _ARGV_REMOVE["ref_bc"] + _ARGV_REMOVE["mcts_fix"]
_ARGV_REMOVE["ref_bc_no_subjective"] = _ARGV_REMOVE["ref_bc"] + _ARGV_REMOVE["no_subjective"]

# 2026-07-22 capacity-matched control: full subjective architecture, uniform
# posterior (--belief_blind). Isolates the belief posterior's marginal value.
_ARGV_ADD["ref_bc_belief_blind"] = _ARGV_ADD["ref_bc"] + ("--belief_blind",)
_ARGV_REMOVE["ref_bc_belief_blind"] = _ARGV_REMOVE["ref_bc"]

# 2026-07-22 reward-weighted BC: weight each reference step's per-agent BC by the
# agent's own per-(regime,agent)-normalized return-to-go. A/B vs plain ref_bc to
# see the prior de-collapse.
_ARGV_ADD["ref_bc_rw"] = _ARGV_ADD["ref_bc"] + ("--bc_reward_weighting",)
_ARGV_REMOVE["ref_bc_rw"] = _ARGV_REMOVE["ref_bc"]

# 2026-07-23 gradient-diffusion diagnostic: hard-select the true-regime value head
# during training (Bayes-average at deploy). If head_diversity rises, the per-regime
# hypernet collapse was gradient diffusion, not the row leak.
_ARGV_ADD["ref_bc_hardval"] = _ARGV_ADD["ref_bc"] + ("--value_hard_select",)
_ARGV_REMOVE["ref_bc_hardval"] = _ARGV_REMOVE["ref_bc"]

# 2026-07-24 budget-unaware-anneal fix. ref_bc's --reference_episode_anneal_steps
# is a hardcoded 28000 ("~3/4 of a 600k-env-step run", i.e. 28000/37500 of that
# budget's training_steps) that argparse-overrides the base config's own
# proportional default (training_steps // 2) at EVERY budget. At 1M env steps
# (training_steps=62500) this leaves BC guidance annealed by 44.8% of training
# instead of 600K's 74.7%, i.e. 3.6x more "unsupervised" steps in absolute terms
# -- a leading suspect for the 600K->1M ref_bc regression (66.6 -> 49.8 return).
# The 28000 placeholder below is inert: MAZeroMixedRunner._build_game_config
# recomputes it as round(training_steps * 28000 / 37500) -- preserving the
# VALIDATED 74.7% fraction, not the raw absolute count -- and appends the
# override after apply_arm_argv runs, so it wins regardless of what's here.
_ARGV_ADD["ref_bc_anneal_scaled"] = _ARGV_ADD["ref_bc"]
_ARGV_REMOVE["ref_bc_anneal_scaled"] = _ARGV_REMOVE["ref_bc"]

# 2026-07-24 Module-1 ablation matched to the anneal-scaled MAIN method: plain
# MAZero (drop --subjective_model) carrying the SAME budget-proportional BC
# anneal (applied in runner._build_game_config for any arm whose name contains
# "anneal_scaled"), so the ablation changes ONLY the subjective module, not the
# BC schedule. This is the fair "revert to baseline MAZero" cell at 1M.
_ARGV_ADD["ref_bc_anneal_scaled_no_subjective"] = _ARGV_ADD["ref_bc"]
_ARGV_REMOVE["ref_bc_anneal_scaled_no_subjective"] = (
    _ARGV_REMOVE["ref_bc"] + _ARGV_REMOVE["no_subjective"])

# 2026-07-26 deployment-cost cell: the anneal-scaled main method with the
# ORIGINAL per-agent decoupled selection re-enabled (the base argv no longer
# passes --decoupled_selection, since centralized/joint search is now the
# user-locked default). Measures what the complexity reduction of centralized
# action broadcasting actually costs, on the same budget/logging as the main
# run. First measurement (seed0, 1M): centralized 56.51 vs decoupled 67.49
# under the older logging -- this arm reproduces the pair apples-to-apples.
_ARGV_ADD["ref_bc_anneal_scaled_decoupled"] = (
    _ARGV_ADD["ref_bc"] + ("--decoupled_selection",))
_ARGV_REMOVE["ref_bc_anneal_scaled_decoupled"] = _ARGV_REMOVE["ref_bc"]

# 2026-07-26 (user request): the hardval configuration carried to 1M with the
# budget-proportional anneal and the original decoupled selection -- i.e. the
# combination that has never been run. Its two ingredients were each validated
# separately and are orthogonal:
#   value_hard_select : trains the TRUE-regime value head (train-time g_true,
#                       same disclosed CTDE envelope as belief supervision;
#                       deploy still Bayes-averages). At 600K it lifted
#                       head_diversity 0.002 -> 0.318 but NOT return
#                       (62.56 vs plain ref_bc's 66.59).
#   anneal_scaled     : preserves the validated 74.7% BC-anneal fraction at any
#                       budget (recovered 49.82 -> 67.49 at 1M).
# Registry-protocol reference points at 1M: anneal_scaled decoupled 67.49,
# centralized 56.51. (The ~72-74 figure sometimes quoted for hardval comes from
# value_deploy_probe.py's own eval protocol, not the registry protocol.)
_ARGV_ADD["ref_bc_anneal_scaled_hardval_decoupled"] = (
    _ARGV_ADD["ref_bc"] + ("--value_hard_select", "--decoupled_selection"))
_ARGV_REMOVE["ref_bc_anneal_scaled_hardval_decoupled"] = _ARGV_REMOVE["ref_bc"]

# 2026-07-26 the CENTRALIZED half of the hardval pair -- completes the version
# selection 2x2 {plain, hardval} x {centralized, decoupled} at 1M/seed0. The
# base argv is already centralized (no --decoupled_selection), so this arm is
# the decoupled one minus that single flag.
#   selection rule: gate on head_diversity (the per-regime heads must actually
#   be differentiated -- belief_blind scores 62.94 at head_diversity 0.0000, so
#   return alone cannot identify the right version), then max return.
_ARGV_ADD["ref_bc_anneal_scaled_hardval"] = (
    _ARGV_ADD["ref_bc"] + ("--value_hard_select",))
_ARGV_REMOVE["ref_bc_anneal_scaled_hardval"] = _ARGV_REMOVE["ref_bc"]

# 2026-07-27 Module-1 ablation control MATCHED to the selected method
# (ref_bc_anneal_scaled_hardval_decoupled): plain MAZero -- no subjective heads --
# carrying the same scaled BC anneal AND the same decoupled selection, so the only
# difference from the method is Module 1 itself.
# Disclosed asymmetry: removing the subjective module deletes the per-regime value
# heads, so --value_hard_select has nothing to act on and is necessarily absent
# here. The control is "plain MAZero + same anneal + same selection mode", not
# "the method minus one flag".
_ARGV_ADD["ref_bc_anneal_scaled_no_subjective_decoupled"] = (
    _ARGV_ADD["ref_bc"] + ("--decoupled_selection",))
_ARGV_REMOVE["ref_bc_anneal_scaled_no_subjective_decoupled"] = (
    _ARGV_REMOVE["ref_bc"] + _ARGV_REMOVE["no_subjective"])

# 2026-07-27 high-parameter version of the method for the capacity comparison.
# Identical to the method of record except --model_scale 3.0, which multiplies
# every network width (and the hypernet/context widths): 1.26M -> 5.25M params
# (4.18x), i.e. roughly the capacity mamba originally had (8.4M net) while the
# parameter-matched baselines sit at ~1.2-1.3M alongside the method.
# lr 0.005 is REQUIRED, not a tuning choice: at the inherited lr=0.02 every
# scale>1 model drives transient extremes into the vendored C++ tree and the
# process dies of SIGFPE (exit 136, no traceback) inside trees.batch_selection
# within ~10 min. Measured: scale 2.0 @0.02 crashes with FINITE losses (so NaN
# is not the trigger), while scale 2.0 and 3.0 @0.005 both survive with finite
# losses. See results/analysis/parameter_matching.md.
# Because this changes lr as well as capacity, the *_lrctl arm below runs the
# method's own width at the same lr, so capacity is the only difference between
# the control and this row.
_ARGV_ADD["ref_bc_anneal_scaled_hardval_decoupled_big"] = (
    _ARGV_ADD["ref_bc_anneal_scaled_hardval_decoupled"]
    + ("--model_scale", "3.0", "--lr", "0.005"))
_ARGV_REMOVE["ref_bc_anneal_scaled_hardval_decoupled_big"] = _ARGV_REMOVE["ref_bc"]

# lr control: the method of record's architecture (scale 1.0) at the
# high-parameter row's lr. Isolates the lr change from the capacity change.
_ARGV_ADD["ref_bc_anneal_scaled_hardval_decoupled_lrctl"] = (
    _ARGV_ADD["ref_bc_anneal_scaled_hardval_decoupled"] + ("--lr", "0.005"))
_ARGV_REMOVE["ref_bc_anneal_scaled_hardval_decoupled_lrctl"] = _ARGV_REMOVE["ref_bc"]

# 2026-07-27 scale 3.0 (5.25M) crashes with SIGFPE inside the vendored C++ tree
# (mcts_sampled.py:131 trees.batch_selection) about 60 s into selfplay: the wider
# net at the inherited lr=0.02 produces a degenerate/NaN root distribution, and
# the C++ selection then divides by a zero visit count. (The Python guard at
# mcts_sampled.py:111 is malformed -- `assert ~(...).sum()` is truthy for an
# all-zero row -- so it does not catch it.) This 2x variant (2.78M, 2.21x the
# method) is the fallback high-parameter point.
_ARGV_ADD["ref_bc_anneal_scaled_hardval_decoupled_big2"] = (
    _ARGV_ADD["ref_bc_anneal_scaled_hardval_decoupled"] + ("--model_scale", "2.0"))
_ARGV_REMOVE["ref_bc_anneal_scaled_hardval_decoupled_big2"] = _ARGV_REMOVE["ref_bc"]

# ---------------------------------------------------------------------------
# 2026-08-05 policy-target 2x2 on the method of record, for the v6 env.
#
# The prior-collapse fix was screened once on v5 (seed 0, 200k) and REJECTED:
# it de-collapses the prior but took return 22.1 -> 13.3, so the canonical argv
# still passes neither flag and runs the upstream defaults (root_cover=none,
# policy_target_type=visit). Two reasons that verdict does not transfer to v6:
# it moved BOTH stages at once, and it was scored on the summed team return,
# which on this reward cannot respond in g1 (the harvests cancel to
# -eps*(moves) exactly) and sees only one agent in g2/g3 -- precisely the
# regimes v6 changes. See results/analysis/regime_knowledge_ceiling.md.
#
# The cells reuse the stage tuples rather than retyping them, so the root-cover
# width stays consistent with validate_root_cover (sampled_action_times must be
# 1 + N*A = 13, and num_simulations >= that; the canonical argv passes 25).
#
#   cell        root_cover  policy_target   arm
#   baseline    none        visit           ref_bc_anneal_scaled_hardval_decoupled
#   A only      star        visit           ..._cover
#   B only      none        q_softmax       ..._qtarget
#   A+B         star        q_softmax       ..._mctsfix
#
# B-alone is a real cell, not a formality: with root_cover=none the root's
# children are sampled from the prior and can collapse to one child
# (tests/algo/test_root_action_enumeration.py), and a softmax target over a
# single child is degenerate. If B-alone underperforms, that cell says whether
# the target or the collapsed root is responsible.
#
# NOTE the names keep the "anneal_scaled" substring, which runner.py keys on to
# rescale the BC anneal to the run's budget.
_ARGV_ADD["ref_bc_anneal_scaled_hardval_decoupled_cover"] = (
    _ARGV_ADD["ref_bc_anneal_scaled_hardval_decoupled"] + _ARGV_ADD["mcts_fix_cover"])
_ARGV_REMOVE["ref_bc_anneal_scaled_hardval_decoupled_cover"] = _ARGV_REMOVE["ref_bc"]

_ARGV_ADD["ref_bc_anneal_scaled_hardval_decoupled_qtarget"] = (
    _ARGV_ADD["ref_bc_anneal_scaled_hardval_decoupled"] + _ARGV_ADD["mcts_fix_qtarget"])
_ARGV_REMOVE["ref_bc_anneal_scaled_hardval_decoupled_qtarget"] = _ARGV_REMOVE["ref_bc"]

_ARGV_ADD["ref_bc_anneal_scaled_hardval_decoupled_mctsfix"] = (
    _ARGV_ADD["ref_bc_anneal_scaled_hardval_decoupled"] + _ARGV_ADD["mcts_fix"])
_ARGV_REMOVE["ref_bc_anneal_scaled_hardval_decoupled_mctsfix"] = _ARGV_REMOVE["ref_bc"]

# --- visit_q_blend: q_softmax with the visit allocation restored as a prior ---
#
# The 2x2 (wave 1, n=2) put B-alone (none / q_softmax) first on every instrument
# -- return, the last-20% trace, and g1 NashConv (5.43 vs the baseline's 12.83)
# -- but at ~5x the baseline's seed spread. The blend targets exactly that
# variance: q_softmax weights a 1-visit Q like a 20-visit Q, and at
# root_cover=star the budget is only ~1.9 visits/child. See
# core/train.py:policy_target_weights.
#
# tau is the only knob and is UNKNOWN, so it is swept rather than guessed --
# this codebase is brutally hyperparameter-sensitive (the lr sweep separated
# 46.68 from 23.78 between 0.02 and 0.01), and screening one guessed tau is the
# same mistake that produced the wrong v5 verdict. tau -> inf is the visit
# target and tau -> 0 is q_softmax, so the bracket must straddle 1.0, where the
# std-normalized advantage and the log-visit span are comparable.
for _tau_name, _tau in (("t025", "0.25"), ("t05", "0.5"),
                        ("t1", "1.0"), ("t2", "2.0")):
    _blend_arm = f"ref_bc_anneal_scaled_hardval_decoupled_blend_{_tau_name}"
    _ARGV_ADD[_blend_arm] = (
        _ARGV_ADD["ref_bc_anneal_scaled_hardval_decoupled"]
        + ("--policy_target_type", "visit_q_blend",
           "--policy_target_temperature", _tau))
    _ARGV_REMOVE[_blend_arm] = _ARGV_REMOVE["ref_bc"]
del _tau_name, _tau, _blend_arm


def _validate(arm: str) -> str:
    if arm not in ARMS:
        raise ValueError(f"Unknown ablation arm {arm!r}. Valid: {list(ARMS)}")
    return arm


def apply_arm(cfg, arm: str):
    """V4Config-level seam: validate; all current arms are argv-level."""
    _validate(arm)
    return cfg


def apply_arm_argv(argv: list[str], arm: str) -> list[str]:
    """Fork-argv seam: returns a NEW argv with the arm's flag edits applied."""
    _validate(arm)
    out = [a for a in argv if a not in _ARGV_REMOVE[arm]]
    out.extend(_ARGV_ADD[arm])
    return out


__all__ = ["ARMS", "apply_arm", "apply_arm_argv"]
