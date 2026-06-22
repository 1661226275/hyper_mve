"""pkg-08 experiments package — sweep + ablate + stats + compare.

Exports the public symbols Phase 5 declares; downstream callers should import
from this package surface rather than reaching into the submodules:

    >>> from hyper_mve.experiments import (
    ...     SweepConfig, run_sweep,
    ...     RegistryRow, RunRegistry,
    ...     ABLATION_IDS,
    ... )

Submodule contracts:
- ``sweep.py``           — pkg-08 spec 05 §3 / §5 / §7 / §8 (sweep harness).
- ``run_registry.py``    — pkg-08 spec 05 §4 (23-key JSONL append-only).
- ``_sweep_worker.py``   — pkg-08 spec 05 §5.3 (subprocess-per-row entry).
- ``ablate.py``          — pkg-08 spec 06 §2 (canned-YAML dispatcher).
- ``stats.py``           — pkg-08 spec 07 §2 (Welch t + Holm-Bonferroni).
- ``compare.py``         — pkg-08 spec 07 §6 (CLI + plot + disclosure).

This module deliberately does NOT eagerly import its submodules — pulling in
matplotlib/scipy/torch up front would slow every ``from hyper_mve.experiments
import ...`` call that only needs the registry types. Submodule imports are
on-demand via ``__getattr__`` (PEP 562).
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "SweepConfig",
    "SweepRow",
    "run_sweep",
    "enumerate_cartesian",
    "compute_config_hash",
    "GpuSemaphore",
    "RegistryRow",
    "RunRegistry",
    "ABLATION_IDS",
]


def __getattr__(name: str) -> Any:
    if name in {"RegistryRow", "RunRegistry"}:
        from . import run_registry as _rr
        return getattr(_rr, name)
    if name in {
        "SweepConfig", "SweepRow", "run_sweep",
        "enumerate_cartesian", "compute_config_hash", "GpuSemaphore",
    }:
        from . import sweep as _sw
        return getattr(_sw, name)
    if name == "ABLATION_IDS":
        from . import ablate as _ab
        return _ab.ABLATION_IDS
    raise AttributeError(f"module 'hyper_mve.experiments' has no attribute {name!r}")
