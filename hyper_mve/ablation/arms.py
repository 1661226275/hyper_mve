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
"""
from __future__ import annotations

ARMS: tuple[str, ...] = (
    "point_estimate_leaf",
    "joint_selection",
    "no_subjective",
    "moe_router",
    "film",
)

# arm -> (flags to add, flags to remove); value-flags are (name, value) adds.
_ARGV_ADD: dict[str, tuple[str, ...]] = {
    "point_estimate_leaf": ("--belief_point_estimate",),
    "joint_selection": (),
    "no_subjective": (),
    "moe_router": ("--conditioning", "moe_router"),
    "film": ("--conditioning", "film"),
}
_ARGV_REMOVE: dict[str, tuple[str, ...]] = {
    "point_estimate_leaf": (),
    "joint_selection": ("--decoupled_selection",),
    "no_subjective": ("--subjective_model",),
    "moe_router": (),
    "film": (),
}


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
