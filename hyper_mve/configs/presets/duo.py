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
    """Construct the 2-agent drift V4Config (Medium-scale, N=2, random_walk).

    Uses the ``film_head`` partial-generation hypernet (shared SGD fc1/fc2 trunk +
    generated FiLM gamma/beta + output head, grouped-RMS-normed) plus the companion
    knobs that unstarve per-agent prediction differentiation: ``detach_pred_context=
    False`` (let policy/value gradient reach role/belief encoders, now that the trunk
    is a stable shared SGD net) and all three ``*_output_scale_init=0.1`` (symmetric;
    with grouped RMS norm each generated element starts at ~0.1, so FiLM gamma actually
    modulates and the generated head starts near fan-in standard init). These target
    the cos_pred_cross->0.998 collapse seen in the FULL-generation run.
    """
    base = build_medium_config()

    # N and type_assignment must change together: ``replace`` applies all kwargs
    # atomically before EnvConfig.__post_init__ re-validates len(type_assignment)==N.
    env = replace(
        base.env,
        N=2,
        type_assignment=(AgentType.ALPHA, AgentType.BETA),
        c_mode="random_walk",
    )

    model = replace(
        base.model,
        hyper_gen_scope="film_head",
        trans_output_scale_init=0.1,   # symmetric with rew (already 0.1) under grouped RMS norm
        pred_output_scale_init=0.1,
    )

    train = replace(base.train, detach_pred_context=False)

    return replace(base, env=env, model=model, train=train, preset_name="duo")
