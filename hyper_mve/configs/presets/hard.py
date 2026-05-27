"""Hard preset (Ch3.9).

N=8, L=24, K=40, T_max=300, 4α+4β with oscillating context. Used for the
online-adaptation experiments (Roadmap §6.10) and the scale stress test.
"""
from __future__ import annotations

from dataclasses import replace

from hyper_mve.schemas import AgentType

from ..v4_config import V4Config
from .medium import build_medium_config


def build_hard_config() -> V4Config:
    """Construct the Hard-difficulty V4Config (Ch3.9)."""
    base = build_medium_config()

    env = replace(
        base.env,
        N=8,
        L=24,
        K=40,
        M=5,
        T_max=300,
        type_assignment=(
            (AgentType.ALPHA,) * 4 + (AgentType.BETA,) * 4
        ),
        c_mode="oscillate",
    )

    train = replace(
        base.train,
        max_train_steps=2_000_000,
    )

    return replace(base, env=env, train=train, preset_name="hard")
