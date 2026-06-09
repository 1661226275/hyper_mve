"""Difficulty presets (Ch3.9): easy / medium / hard.

Each module exposes a single ``build_<name>_config()`` factory returning a
fully populated ``V4Config``. Importers should normally use
``V4Config.from_preset(name)`` instead of touching these factories directly.
"""
from __future__ import annotations

from .duo import build_duo_config
from .duo_base_lora import build_duo_base_lora_config
from .duo_basegen import build_duo_basegen_config
from .duo_film_lora import build_duo_film_lora_config
from .duo_film_lora_fc2 import build_duo_film_lora_fc2_config
from .easy import build_easy_config
from .hard import build_hard_config
from .medium import build_medium_config
from .medium_base_lora import build_medium_base_lora_config
from .medium_film_lora import build_medium_film_lora_config
from .medium_film_lora_fc2 import build_medium_film_lora_fc2_config

__all__ = [
    "build_easy_config",
    "build_medium_config",
    "build_hard_config",
    "build_duo_config",
    "build_duo_basegen_config",
    "build_duo_film_lora_config",
    "build_duo_film_lora_fc2_config",
    "build_duo_base_lora_config",
    "build_medium_film_lora_config",
    "build_medium_film_lora_fc2_config",
    "build_medium_base_lora_config",
]
