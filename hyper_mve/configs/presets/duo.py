"""Duo preset — 2-agent (1α+1β) drift experiment.

Medium-scale grid/dynamics/training budget, but N=2 (one ALPHA, one BETA) with a
drifting context (``c_mode="random_walk"``). Built for the diagnostic experiment:
verify the MVE planner produces a differentiated action distribution and that the
prediction net learns toward it, with a non-trivial belief target (drifting c_t)
that does not collapse the way a static context does.

Differs from ``easy`` (also N=2, 1α+1β) by keeping Medium's L/K/M/T_max and the
1M-step budget — so results are comparable to the Medium baseline, only the agent
count and context dynamics change.
"""
from __future__ import annotations

from dataclasses import replace

from hyper_mve.schemas import AgentType

from ..v4_config import V4Config
from .medium import build_medium_config


def build_duo_config() -> V4Config:
    """Construct the 2-agent drift V4Config (Medium-scale, N=2, random_walk)."""
    base = build_medium_config()

    # N and type_assignment must change together: ``replace`` applies all kwargs
    # atomically before EnvConfig.__post_init__ re-validates len(type_assignment)==N.
    env = replace(
        base.env,
        N=2,
        type_assignment=(AgentType.ALPHA, AgentType.BETA),
        c_mode="random_walk",
    )

    return replace(base, env=env, preset_name="duo")
