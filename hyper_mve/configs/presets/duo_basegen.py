"""Duo-BaseGen preset — duo env + base_gen hypernet + shared subjective trunk.

Same 2-agent (1α+1β) drift env as ``duo`` (Medium-scale, N=2, random_walk), but swaps the
hypernet architecture to test the two optimizations together:
  - ``hyper_gen_scope="base_gen"``: plain SGD fc1 base (Linear+LN+ReLU, no FiLM) + fully
    HyperNet-generated fc2 (weight + FiLM gamma/beta) + generated head (CCWM
    fc_dynamics_1/fc_dynamics_2 style; more capacity than ``duo``'s film_head, more stable
    than full).
  - ``share_subjective_trunk=True``: hyper_rew + hyper_pred merge into one shared trunk
    with two heads (theta_rew / theta_pred); hyper_trans (objective) stays separate.

Companion knobs (mirror ``duo``): all three ``*_output_scale_init=0.1`` (symmetric under
grouped RMS norm — each generated element starts ~0.1 so FiLM gamma actually modulates and
the generated weight starts near fan-in standard init) and ``detach_pred_context=False``
(let policy/value gradient reach role/belief encoders + the now-shared subjective trunk).
Keep ``duo`` (film_head, independent trunks) as the A/B baseline.
"""
from __future__ import annotations

from dataclasses import replace

from hyper_mve.schemas import AgentType

from ..v4_config import V4Config
from .medium import build_medium_config


def build_duo_basegen_config() -> V4Config:
    """Construct the duo-env base_gen + shared-subjective-trunk V4Config."""
    base = build_medium_config()

    # N and type_assignment must change together (EnvConfig.__post_init__ re-validates
    # len(type_assignment)==N); ``replace`` applies all kwargs atomically.
    env = replace(
        base.env,
        N=2,
        type_assignment=(AgentType.ALPHA, AgentType.BETA),
        c_mode="random_walk",
    )

    model = replace(
        base.model,
        hyper_gen_scope="base_gen",
        share_subjective_trunk=True,
        trans_output_scale_init=0.1,   # symmetric with rew (already 0.1) under grouped RMS norm
        pred_output_scale_init=0.1,
    )

    train = replace(
        base.train,
        detach_pred_context=False,
        # [v4-opt 2026-06] duo family: belief gate never opens (see duo.py rationale).
        belief_grad_gating_steps=1_000_000_000,
    )

    return replace(base, env=env, model=model, train=train, preset_name="duo_basegen")
