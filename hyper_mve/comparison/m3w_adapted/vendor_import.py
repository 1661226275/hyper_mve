"""Unmodified vendor imports from the m3w-marl clone.

The upstream package inits (``m3w/__init__.py``, ``m3w/models/__init__.py``)
are empty, so a sys.path shim + plain import loads exactly
``m3w/models/world_models.py`` (torch + einops only). The vendor-integrity
unit test asserts via ``inspect.getsourcefile`` that these classes come from
the clone and that the clone file is unmodified in its nested git repo.
"""
from __future__ import annotations

import sys
from pathlib import Path

M3W_DIR = Path(__file__).resolve().parents[1] / "vendor" / "m3w-marl"


def ensure_m3w_on_path() -> None:
    p = str(M3W_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def load_vendor_modules():
    """Return (CenMoEDynamicsModel, CenMoERewardModel, NoisyTopKRouter)."""
    ensure_m3w_on_path()
    from m3w.models.world_models import (  # noqa: PLC0415
        CenMoEDynamicsModel,
        CenMoERewardModel,
        NoisyTopKRouter,
    )
    return CenMoEDynamicsModel, CenMoERewardModel, NoisyTopKRouter
