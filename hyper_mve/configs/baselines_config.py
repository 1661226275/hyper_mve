"""BaselinesConfig — 5-field cross-spec contract for pkg-07 baselines.

Pkg-07 design §D10: `cfg.baselines.*` namespace (4 internal capacity knobs +
1 external LR-sweep grid mapping). Per-impl tuning constants (e.g.
``external_smoke_max_env_steps``, ``external_qmix_mixer_hidden_dim``) live
inside the runner module defaults — they are impl-internal, not cross-spec
contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


def _default_external_lr_sweep_grid() -> Mapping[str, tuple[float, ...]]:
    return MappingProxyType({
        "external_mappo":         (1e-4, 3e-4, 1e-3),
        "external_qmix":          (1e-4, 3e-4, 1e-3),
        "external_ma_muzero_gh":  (1e-4, 3e-4, 1e-3),
        # external_mamba added at sourcing time; external_marie/ga not swept (stubs).
    })


@dataclass(frozen=True)
class BaselinesConfig:
    """5-field exhaustive cfg namespace (4 internal from pkg-06 D10 + 1 external new)."""

    # Internal (4)
    internal_wide_hidden_dim: int = 512
    internal_deep_layers: int = 8
    internal_ma_muzero_share_pred_head: bool = True
    internal_explicit_type_branches: int = 2          # α / β

    # External (1) — per-baseline LR sweep mapping (NOT a single shared tuple).
    external_lr_sweep_grid: Mapping[str, tuple[float, ...]] = field(
        default_factory=_default_external_lr_sweep_grid
    )
