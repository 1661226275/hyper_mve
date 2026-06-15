"""Duo-Film-LoRA preset — duo env + film_head + 3-way output-layer LoRA(r=32).

PRIMARY film-line cell of the LoRA experiment. Same 2-agent (1alpha+1beta) drift env as
``duo`` (Medium-scale, N=2, random_walk) with the ``film_head`` partial-generation hypernet,
plus output-layer LoRA (``hyper_output_rank=32``) applied UNIFORMLY to all three hypernets
(hyper_trans / hyper_rew / hyper_pred). In film_head the hypernet is ~92% output-layer
params, so factorizing ``Linear(256 -> theta_pc)`` collapses the hypernet from ~3.09M to
~694k params.

Companion knobs (mirror ``duo``): all three ``*_output_scale_init=0.1`` and
``detach_pred_context=False``. ``share_subjective_trunk=False`` (dropped from the LoRA line;
LoRA on the shared SubjectiveHyperNet is not yet wired).
"""
from __future__ import annotations

from dataclasses import replace

from hyper_mve.schemas import AgentType

from ..v4_config import V4Config
from .medium import build_medium_config


def build_duo_film_lora_config() -> V4Config:
    """Construct the duo-env film_head + output-LoRA(r=32) V4Config."""
    base = build_medium_config()

    env = replace(
        base.env,
        N=2,
        type_assignment=(AgentType.ALPHA, AgentType.BETA),
        c_mode="random_walk",
    )

    model = replace(
        base.model,
        hyper_gen_scope="film_head",
        hyper_output_rank=32,
        share_subjective_trunk=False,
        trans_output_scale_init=0.1,   # symmetric with rew (already 0.1) under grouped RMS norm
        pred_output_scale_init=0.1,
    )

    train = replace(
        base.train,
        detach_pred_context=False,
        # [v4-opt 2026-06] duo family: belief gate never opens (see duo.py rationale).
        belief_grad_gating_steps=1_000_000_000,
        # [v4-opt 2026-06c] duo family P0.4 + P0.2(a) — see duo.py docstring for rationale.
        buffer_size=1500,
        mve_temperature=0.5,
    )

    return replace(base, env=env, model=model, train=train, preset_name="duo_film_lora")
