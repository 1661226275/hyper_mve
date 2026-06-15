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

    train = replace(
        base.train,
        detach_pred_context=False,
        # [v4-opt 2026-06] N=2 belief losses are trivial (one fixed opponent label;
        # 2-agent variance hinge) and the gate opening at 5000 coincided with the
        # LR-warmup end — two confounded regime changes right at the observed
        # policy-loss U-turn. Keep the belief path permanently detached from the
        # main loss in duo runs (L_belief still trains the BeliefNet).
        belief_grad_gating_steps=1_000_000_000,
        # [v4-opt 2026-06c] P0.4: 2agent runs showed diag/target_age_steps ≈ 2500 at
        # the tail with the default 5000-episode buffer. After P0.2 sharpens π_mve,
        # batching against 2500-step-old (= old-aggregation) targets is the next
        # bottleneck; halving the buffer brings target age to ~750 steps.
        buffer_size=1500,
        # [v4-opt 2026-06c] P0.2(a): the planner aggregation softmax(z_score(Ḡ)/τ)
        # structurally floors π_mve entropy at ≈ 1.39 (verified by 20k-draw MC),
        # which matched the observed plateau exactly across all three 2agent cells.
        # Dropping τ from 1.0 → 0.5 is the cheapest test of whether the entropy
        # plateau is the z-score's fault. Side effect to monitor: π_mve also drives
        # collection sampling (worker.py:110-112) so sharper targets reduce
        # exploration — fall back is ε-greedy at ε_min=0.05.
        mve_temperature=0.5,
    )

    return replace(base, env=env, model=model, train=train, preset_name="duo")
