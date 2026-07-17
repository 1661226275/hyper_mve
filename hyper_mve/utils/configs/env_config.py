"""EnvConfig — RelationCommons dimensions and dynamics parameters (v5 Pkg-09).

The v4 fields (type_assignment, c_t evolution, α(c_t) regen range, Fehr-Schmidt
scalars, capability sampling ranges, neighbor-coupling knobs) were deleted with
the resource_commons environment in the Stage-6 cleanup. Physics is fixed:
constant regen rate ``alpha``; roles live entirely in the relationship matrix
``W(g)`` (``schemas.relation``).
"""
from __future__ import annotations

from dataclasses import dataclass

from hyper_mve.utils.schemas._constants import EPSILON_MOVE, Q_MAX


_VALID_RELATION_FAMILIES: tuple[str, ...] = ("g2", "g4", "g4_ext")
_VALID_REGIME_KERNELS: tuple[str, ...] = ("uniform",)


@dataclass(frozen=True)
class EnvConfig:
    """RelationCommons environment configuration (rel_* presets fill these)."""

    # Dimensions
    N: int
    L: int
    K: int
    T_max: int
    A: int = 6                          # NOOP + 4 directions + HARVEST

    # Resource dynamics (fixed physics)
    Q_max: float = Q_MAX
    alpha: float = 0.10                 # constant regen rate (replaces v4 α(c_t))
    epsilon_move: float = EPSILON_MOVE  # per-move cost in the relational reward

    # Relationship regimes (Pkg-09, `schemas.relation`)
    relation_family: str = "g2"
    relation_intensity: float = 1.0     # λ: ±λ off-diagonal weights
    regime_prior: tuple[float, ...] | None = None   # ρ over full G (None = uniform)
    regime_switch_prob: float = 0.0     # p: 0 = point 1 (episodic), >0 = point 2
    regime_kernel: str = "uniform"      # κ
    train_regime_ids: tuple[int, ...] | None = None  # holdout restriction at reset

    def __post_init__(self) -> None:
        if self.N < 1:
            raise ValueError(f"N must be ≥ 1, got {self.N}")
        if self.K < 0 or self.L < 1 or self.T_max < 1:
            raise ValueError(
                f"L({self.L}), K({self.K}), T_max({self.T_max}) must be positive"
            )
        if self.relation_family not in _VALID_RELATION_FAMILIES:
            raise ValueError(
                f"Unknown relation_family: {self.relation_family!r} "
                f"(valid: {_VALID_RELATION_FAMILIES})"
            )
        if not (0.0 < self.relation_intensity <= 1.0):
            raise ValueError(
                f"relation_intensity={self.relation_intensity} ∉ (0, 1]"
            )
        if not (0.0 <= self.regime_switch_prob <= 1.0):
            raise ValueError(
                f"regime_switch_prob={self.regime_switch_prob} ∉ [0, 1]"
            )
        if self.regime_kernel not in _VALID_REGIME_KERNELS:
            raise ValueError(
                f"Unknown regime_kernel: {self.regime_kernel!r} "
                f"(valid: {_VALID_REGIME_KERNELS})"
            )
        if self.regime_prior is not None:
            if any(p < 0 for p in self.regime_prior):
                raise ValueError(
                    f"regime_prior has negative entries: {self.regime_prior}"
                )
            if abs(sum(self.regime_prior) - 1.0) > 1e-6:
                raise ValueError(
                    f"regime_prior must sum to 1, got {sum(self.regime_prior)}"
                )
        if self.train_regime_ids is not None and len(self.train_regime_ids) == 0:
            raise ValueError("train_regime_ids must be None or non-empty")
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError(f"alpha={self.alpha} ∉ (0, 1]")
