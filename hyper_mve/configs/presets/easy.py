"""Easy preset (Ch3.9).

N=2, L=8, K=8, T_max=100, 1α+1β. Used for LR sweeps and the Decision-Point
ablations (1 & 3) since 200K-step training is materially faster than the
1M-step Medium baseline.
"""
from __future__ import annotations

from dataclasses import replace

from hyper_mve.schemas import AgentType

from ..v4_config import V4Config
from .medium import build_medium_config


def build_easy_config() -> V4Config:
    """Construct the Easy-difficulty V4Config (Ch3.9)."""
    base = build_medium_config()

    env = replace(
        base.env,
        N=2,
        L=8,
        K=8,
        M=1,
        T_max=100,
        type_assignment=(AgentType.ALPHA, AgentType.BETA),
    )

    train = replace(
        base.train,
        max_train_steps=200_000,
    )

    return replace(base, env=env, train=train, preset_name="easy")
