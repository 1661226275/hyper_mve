"""v5 RelationCommons presets (Pkg-09): rel_duo / rel_duo_holdout.

Each module exposes ``build_<name>_config()`` factories returning a fully
populated ``V4Config``. Importers should normally use
``V4Config.from_preset(name)`` instead of touching these factories directly.
"""
from __future__ import annotations

from .rel_duo import build_rel_duo_config, build_rel_duo_holdout_config

__all__ = [
    "build_rel_duo_config",
    "build_rel_duo_holdout_config",
]
