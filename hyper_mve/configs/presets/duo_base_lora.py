"""Duo-Base-LoRA preset — duo env + base_gen + 3-way output-layer LoRA(r=32).

PRIMARY base-line cell of the LoRA experiment. Same 2-agent (1alpha+1beta) drift env as
``duo_basegen`` but with separate hypernets (``share_subjective_trunk=False``) and output-layer
LoRA (``hyper_output_rank=32``) applied uniformly to all three hypernets, shrinking the
base_gen hypernet from ~15.6M to ~2.30M params. ``lora_fc2`` is forbidden on base_gen (it
already generates fc2 fully, so a rank-r Delta_W would be redundant).

Companion knobs (mirror ``duo_basegen``): all three ``*_output_scale_init=0.1`` and
``detach_pred_context=False``.
"""
from __future__ import annotations

from dataclasses import replace

from hyper_mve.schemas import AgentType

from ..v4_config import V4Config
from .medium import build_medium_config


def build_duo_base_lora_config() -> V4Config:
    """Construct the duo-env base_gen + output-LoRA(r=32) V4Config."""
    base = build_medium_config()

    env = replace(
        base.env,
        N=2,
        type_assignment=(AgentType.ALPHA, AgentType.BETA),
        c_mode="random_walk",
    )

    model = replace(
        base.model,
        hyper_gen_scope="base_gen",
        hyper_output_rank=32,
        share_subjective_trunk=False,
        trans_output_scale_init=0.1,
        pred_output_scale_init=0.1,
    )

    train = replace(base.train, detach_pred_context=False)

    return replace(base, env=env, model=model, train=train, preset_name="duo_base_lora")
