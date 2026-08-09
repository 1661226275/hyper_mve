"""RelationCommons presets: rel_duo (v5) / rel_recip (v6), each with a holdout.

Each module exposes ``build_<name>_config()`` factories returning a fully
populated ``V4Config``. Importers should normally use
``V4Config.from_preset(name)`` instead of touching these factories directly.

``rel_duo`` is the v5 environment and stays frozen — it is the control in which
the hidden regime is provably worthless to infer. ``rel_recip`` is the v6
environment built so that it is not, and stays frozen too now that runs are
archived against it. ``rel_coopmix`` is ``rel_recip`` on the ``g2cm`` family, which
carries no purely adversarial regime.
"""
from __future__ import annotations

from .rel_coopmix import build_rel_coopmix_config, build_rel_coopmix_holdout_config
from .rel_duo import build_rel_duo_config, build_rel_duo_holdout_config
from .rel_recip import build_rel_recip_config, build_rel_recip_holdout_config

__all__ = [
    "build_rel_coopmix_config",
    "build_rel_coopmix_holdout_config",
    "build_rel_duo_config",
    "build_rel_duo_holdout_config",
    "build_rel_recip_config",
    "build_rel_recip_holdout_config",
]
