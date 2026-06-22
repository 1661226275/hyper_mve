"""v4 configuration package (Pkg-01 output).

Public surface:

    >>> from hyper_mve.configs import V4Config
    >>> cfg = V4Config.from_preset("medium")
    >>> cfg.env.N, cfg.env.K, cfg.train.lr
    (4, 20, 0.0001)

Sub-config dataclasses (``EnvConfig``, ``ModelConfig``, ``TrainConfig``,
``MupConfig``, ``EvalConfig``, ``LegacyConfig``) are also re-exported for
typed function signatures in downstream packages.
"""
from __future__ import annotations

from .baselines_config import BaselinesConfig
from .env_config import EnvConfig
from .eval_config import EvalConfig
from .legacy_config import LegacyConfig
from .model_config import ModelConfig
from .mup_config import MupConfig
from .train_config import TrainConfig
from .v4_config import V4Config

__all__ = [
    "BaselinesConfig",
    "EnvConfig",
    "EvalConfig",
    "LegacyConfig",
    "ModelConfig",
    "MupConfig",
    "TrainConfig",
    "V4Config",
]
