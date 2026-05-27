"""MupConfig — μP base-shape placeholder (Pkg-07 fills concrete values)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MupConfig:
    """μP coordinate-check / scaling configuration (Pkg-07 owns the fill-in).

    For Pkg-01 this exists only to lock the import surface so downstream
    packages can take a ``MupConfig`` parameter without coupling to Pkg-07.
    """

    enabled: bool = False
    base_shape_path: Optional[str] = None
    lr_scaling_enabled: bool = False
