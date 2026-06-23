"""Modular thesis experiment suite.

A *cell* is a declarative SweepConfig YAML + a ``meta:`` block (id, tier,
deliverables, blocked_on, …). The :data:`manifest.yaml` lists every cell in
decision-gate order. ``scripts/run_suite.py`` loads cells, narrows them
(``--only/--variants/--seeds/--size``), and drives ``run_sweep`` with per-cell
registry isolation + ``--force`` invalidation.
"""
from __future__ import annotations

from .cell import (
    SuiteCell,
    cell_from_mapping,
    default_manifest_path,
    load_cell,
    load_manifest,
    TIERS,
)

__all__ = [
    "SuiteCell",
    "cell_from_mapping",
    "load_cell",
    "load_manifest",
    "default_manifest_path",
    "TIERS",
]
