"""BaselinesConfig — 5-field cross-spec contract for pkg-07 baselines.

Pkg-07 design §D10: `cfg.baselines.*` namespace (4 internal capacity knobs +
1 external LR-sweep grid mapping). Per-impl tuning constants (e.g.
``external_smoke_max_env_steps``, ``external_qmix_mixer_hidden_dim``) live
inside the runner module defaults — they are impl-internal, not cross-spec
contract.

Note on the LR-sweep grid type: stored as a plain ``dict`` (not
``MappingProxyType``) because ``MappingProxyType`` is not picklable and
breaks ``dataclasses.asdict(cfg)`` / ``copy.deepcopy(model)`` once
``BaselinesConfig`` is embedded in ``V4Config``. The dataclass is
``frozen``, so the field reference is still immutable.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def _default_external_lr_sweep_grid() -> dict[str, tuple[float, ...]]:
    return {
        "external_mappo":         (1e-4, 3e-4, 1e-3),
        "external_qmix":          (1e-4, 3e-4, 1e-3),
        "external_ma_muzero_gh":  (1e-4, 3e-4, 1e-3),
        # [2026-07-10] MAMBA sourced (spec 06 §4.4 amendment): actor/value LR grid.
        "external_mamba":         (1e-4, 3e-4, 1e-3),
        # external_marie/ga not swept (stubs).
    }


@dataclass(frozen=True)
class BaselinesConfig:
    """5-field exhaustive cfg namespace (4 internal from pkg-06 D10 + 1 external new)."""

    # Internal (4)
    internal_wide_hidden_dim: int = 512
    internal_deep_layers: int = 8
    internal_ma_muzero_share_pred_head: bool = True
    internal_explicit_type_branches: int = 2          # α / β

    # External (1) — per-baseline LR sweep mapping (NOT a single shared tuple).
    external_lr_sweep_grid: dict[str, tuple[float, ...]] = field(
        default_factory=_default_external_lr_sweep_grid
    )
