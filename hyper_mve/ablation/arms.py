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
