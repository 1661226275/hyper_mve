"""Backward-compat shim (v4.7 → v4).

The v4.7 ``BaseConfig`` flat class has been replaced by the
``hyper_mve.configs.V4Config`` 5-layer structure. This module re-exports the
Medium-preset values as a flat ``BaseConfig`` so legacy call-sites that did
``from hyper_mve.config import BaseConfig`` keep working — with a
``DeprecationWarning`` to flag the migration.

The full v4.7 ``BaseConfig`` definition lives at
``hyper_mve._legacy_v4_7.config`` and is what the archived v4.7 training
scripts actually import.

Use ``from hyper_mve.configs import V4Config`` in new code.
"""
from __future__ import annotations

import warnings

from hyper_mve.configs import V4Config

warnings.warn(
    "hyper_mve.config.BaseConfig is deprecated. "
    "Use 'from hyper_mve.configs import V4Config' (then V4Config.from_preset(...)) "
    "in new code.",
    DeprecationWarning,
    stacklevel=2,
)


def _flatten_sub_configs(cfg: V4Config) -> dict:
    """Collapse sub-config dataclasses into a single flat namespace."""
    flat: dict = {}
    for sub_name in ("env", "model", "train", "eval", "legacy"):
        sub = getattr(cfg, sub_name)
        for key, value in vars(sub).items():
            flat[key] = value
    flat["preset_name"] = cfg.preset_name
    return flat


def _build_base_config_shim() -> type:
    """Return a class whose attributes mirror v4.7-style flat field access."""
    cfg = V4Config.from_preset("medium")
    flat = _flatten_sub_configs(cfg)

    # Compose a class object so callers can do BaseConfig.lr like before.
    attrs = dict(flat)
    attrs["__doc__"] = (
        "Flat backward-compat view of V4Config.from_preset('medium'). "
        "Deprecated; use V4Config directly."
    )
    attrs["_v4_config"] = cfg
    return type("BaseConfig", (), attrs)


BaseConfig = _build_base_config_shim()
