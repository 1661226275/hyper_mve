"""Medium-Film-LoRA preset — 4-agent (medium env) + film_head + 3-way output-LoRA(r=32).

The N=4 (2alpha+2beta, static) main-comparison counterpart of ``duo_film_lora``. Keeps medium's
env untouched; only swaps the hypernet to film_head partial generation + uniform output-layer
LoRA (``hyper_output_rank=32``). Companion knobs mirror the duo LoRA line: all three
``*_output_scale_init=0.1`` and ``detach_pred_context=False``. ``share_subjective_trunk=False``.
"""
from __future__ import annotations

from dataclasses import replace

from ..v4_config import V4Config
from .medium import build_medium_config


def build_medium_film_lora_config() -> V4Config:
    """Construct the medium-env (N=4) film_head + output-LoRA(r=32) V4Config."""
    base = build_medium_config()

    model = replace(
        base.model,
        hyper_gen_scope="film_head",
        hyper_output_rank=32,
        share_subjective_trunk=False,
        trans_output_scale_init=0.1,
        pred_output_scale_init=0.1,
    )

    train = replace(base.train, detach_pred_context=False)

    return replace(base, model=model, train=train, preset_name="medium_film_lora")
