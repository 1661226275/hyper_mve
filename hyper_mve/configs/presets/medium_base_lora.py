"""Medium-Base-LoRA preset — 4-agent (medium env) + base_gen + 3-way output-LoRA(r=32).

The N=4 counterpart of ``duo_base_lora``: base_gen (fully generated fc2) + uniform output-layer
LoRA (``hyper_output_rank=32``), separate hypernets. Keeps medium's env (N=4, 2alpha+2beta,
static). ``lora_fc2`` is forbidden on base_gen (it already generates fc2 fully). Companion knobs
mirror the duo LoRA line: all three ``*_output_scale_init=0.1`` and ``detach_pred_context=False``.
"""
from __future__ import annotations

from dataclasses import replace

from ..v4_config import V4Config
from .medium import build_medium_config


def build_medium_base_lora_config() -> V4Config:
    """Construct the medium-env (N=4) base_gen + output-LoRA(r=32) V4Config."""
    base = build_medium_config()

    model = replace(
        base.model,
        hyper_gen_scope="base_gen",
        hyper_output_rank=32,
        share_subjective_trunk=False,
        trans_output_scale_init=0.1,
        pred_output_scale_init=0.1,
    )

    train = replace(base.train, detach_pred_context=False)

    return replace(base, model=model, train=train, preset_name="medium_base_lora")
