"""v5 configuration package (Pkg-01/09 output).

Public surface:

    >>> from hyper_mve.utils.configs import V4Config
    >>> cfg = V4Config.from_preset("rel_duo")
    >>> cfg.env.N, cfg.env.K, cfg.train.lr
    (2, 8, 0.0001)

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
