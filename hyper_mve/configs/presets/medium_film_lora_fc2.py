"""Medium-Film-LoRA-fc2 preset — 4-agent (medium env) + lora_fc2(r=8) + output-LoRA(r=32).

The N=4 counterpart of ``duo_film_lora_fc2``: film_head PLUS a per-context rank-8 weight delta
on the shared fc2 (``W2_eff = W2_base + Bf @ Af``), giving per-context feature mixing. Keeps
medium's env (N=4, 2alpha+2beta, static). The companion ``*_output_scale_init=0.1`` is REQUIRED
(asserted in ModelConfig) so Delta_W stays expressive (~0.028 at 0.1/r=8 vs ~3e-4 at the default
0.01). ``share_subjective_trunk=False``.
"""
from __future__ import annotations

from dataclasses import replace

from ..v4_config import V4Config
from .medium import build_medium_config


def build_medium_film_lora_fc2_config() -> V4Config:
    """Construct the medium-env (N=4) lora_fc2(r=8) + output-LoRA(r=32) V4Config."""
    base = build_medium_config()

    model = replace(
        base.model,
        hyper_gen_scope="lora_fc2",
        lora_fc2_rank=8,
        hyper_output_rank=32,
        share_subjective_trunk=False,
        trans_output_scale_init=0.1,   # REQUIRED >= 0.05 for lora_fc2 (Delta_W expressive)
        pred_output_scale_init=0.1,
    )

    train = replace(base.train, detach_pred_context=False)

    return replace(base, model=model, train=train, preset_name="medium_film_lora_fc2")
