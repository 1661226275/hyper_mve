"""Duo-Film-LoRA-fc2 preset — duo env + lora_fc2(r=8) + 3-way output-layer LoRA(r=32).

Film-line cell WITH the expressivity booster. Same env/knobs as ``duo_film_lora`` but swaps
the gen_scope to ``lora_fc2``: film_head PLUS a per-context rank-r weight delta on the shared
fc2 (``W2_eff = W2_base + Bf @ Af``), giving the per-context feature MIXING that plain
film_head lacks. ``lora_fc2_rank=8`` (default rank). Output-layer LoRA(r=32) is also on, so
the hypernet is ~896k params.

The companion ``*_output_scale_init=0.1`` is REQUIRED here (asserted in ModelConfig): under
grouped RMS norm Delta_W ~ output_scale^2 * sqrt(r) ~ 0.028 at 0.1/r=8 (~32% of the kaiming
fc2 base); at the default 0.01 it collapses to ~3e-4 (dead). ``share_subjective_trunk=False``.
"""
from __future__ import annotations

from dataclasses import replace

from hyper_mve.schemas import AgentType

from ..v4_config import V4Config
from .medium import build_medium_config


def build_duo_film_lora_fc2_config() -> V4Config:
    """Construct the duo-env lora_fc2(r=8) + output-LoRA(r=32) V4Config."""
    base = build_medium_config()

    env = replace(
        base.env,
        N=2,
        type_assignment=(AgentType.ALPHA, AgentType.BETA),
        c_mode="random_walk",
    )

    model = replace(
        base.model,
        hyper_gen_scope="lora_fc2",
        lora_fc2_rank=8,
        hyper_output_rank=32,
        share_subjective_trunk=False,
        trans_output_scale_init=0.1,   # REQUIRED >= 0.05 for lora_fc2 (Delta_W expressive)
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

    return replace(base, env=env, model=model, train=train, preset_name="duo_film_lora_fc2")
