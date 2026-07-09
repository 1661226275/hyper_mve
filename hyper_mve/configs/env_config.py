"""EnvConfig — environment dimensions and dynamics parameters (Ch3.3-3.6, 3.9)."""
from __future__ import annotations

from dataclasses import dataclass, field

from hyper_mve.schemas import AgentType
from hyper_mve.schemas._constants import (
    ALPHA_MAX,
    ALPHA_MIN,
    EPSILON_MOVE,
    ETA_RANGE,
    KAPPA,
    LAMBDA_ADV,
    LAMBDA_DISADV,
    NU_RANGE,
    PHI_FOV_RANGE,
    Q_MAX,
    ZETA_RANGE,
)


_VALID_C_MODES: tuple[str, ...] = ("static", "oscillate", "random_walk")
_VALID_RELATION_FAMILIES: tuple[str, ...] = ("g2", "g4", "g4_ext")
_VALID_REGIME_KERNELS: tuple[str, ...] = ("uniform",)


@dataclass(frozen=True)
class EnvConfig:
    """ResourceCommons environment configuration (Ch3.9 presets fill these).

    Field categories: grid dims, type assignment, context evolution,
    resource dynamics (Ch3.3), type-mechanism scalars (Ch3.5), and
    capability sampling ranges (Ch3.6, defaults from `_constants`).
    """

    # Dimensions
    N: int
    L: int
    K: int
    M: int
    T_max: int
    A: int = 6                          # NOOP + 4 directions + HARVEST

    # Type assignment (Ch3.5.1); length must equal N
    type_assignment: tuple[AgentType, ...] = (
        AgentType.ALPHA, AgentType.ALPHA,
        AgentType.BETA, AgentType.BETA,
    )

    # Context evolution (Ch3.4)
    c_mode: str = "static"
    c_oscillate_period: int = 50
    c_random_walk_sigma: float = 0.05
    c_shock_prob: float = 0.2
    c_shock_range: float = 0.3

    # [v4-opt 2026-06c] P0.3: c_t observability switch (Review_v4_TheoryAudit §2.3,
    # Ch3.7 "c_hidden" mode). When False, the c_t slot in each agent's observation
    # `global` block is overwritten with `c_hidden_constant` so the ĉ-head becomes a
    # real inference target instead of identity-readback. ẑ-head behaviour is
    # unaffected (see §9-3 of the diagnosis: under duo N=2 with fixed type assignment
    # ẑ is structurally trivial regardless of c visibility).
    # Notes:
    #   - The Oracle field info["c_true"] is NOT touched; only the model-facing
    #     observation is masked.
    #   - context_evolution still drives c_t dynamics internally, so resource
    #     dynamics and reward computations remain consistent with the rule.
    c_visible: bool = True
    c_hidden_constant: float = 0.5

    # Resource dynamics (Ch3.3)
    Q_max: float = Q_MAX
    alpha_min: float = ALPHA_MIN
    alpha_max: float = ALPHA_MAX
    kappa_f: float = 6.0
    theta_f: float = 0.3
    d_nbr: int = 3

    # Patchy resource generation (Ch3.3.2)
    sigma_patch: float = 2.0

    # Type mechanism (Ch3.5)
    kappa: float = KAPPA
    lambda_disadv: float = LAMBDA_DISADV
    lambda_adv: float = LAMBDA_ADV
    epsilon_move: float = EPSILON_MOVE

    # Capability sampling ranges (Ch3.6)
    eta_range: tuple[float, float] = ETA_RANGE
    phi_fov_range: tuple[float, float] = PHI_FOV_RANGE
    nu_range: tuple[float, float] = NU_RANGE
    zeta_range: tuple[float, float] = ZETA_RANGE

    # ------------------------------------------------------------------
    # v5 relationship regimes (Pkg-09, `schemas.relation`) — consumed only
    # by `envs.relation_commons`; legacy ResourceCommons ignores them.
    # Family↔N consistency is validated in `get_regime_family`, not here,
    # so legacy presets (any N) stay constructible with these defaults.
    # ------------------------------------------------------------------
    relation_family: str = "g2"
    relation_intensity: float = 1.0     # λ: ±λ off-diagonal weights
    regime_prior: tuple[float, ...] | None = None   # ρ over full G (None = uniform)
    regime_switch_prob: float = 0.0     # p: 0 = point 1 (episodic), >0 = point 2
    regime_kernel: str = "uniform"      # κ
    train_regime_ids: tuple[int, ...] | None = None  # holdout restriction at reset
    alpha: float = 0.10                 # v5 constant regen rate (replaces α(c_t))

    def __post_init__(self) -> None:
        if len(self.type_assignment) != self.N:
            raise ValueError(
                f"type_assignment length {len(self.type_assignment)} != N {self.N}"
            )
        if self.c_mode not in _VALID_C_MODES:
            raise ValueError(
                f"Unknown c_mode: {self.c_mode!r} (valid: {_VALID_C_MODES})"
            )
        if not (0.0 < self.alpha_min < self.alpha_max <= 1.0):
            raise ValueError(
                f"alpha range invalid: ({self.alpha_min}, {self.alpha_max})"
            )
        if self.N < 1:
            raise ValueError(f"N must be ≥ 1, got {self.N}")
        if self.K < 0 or self.L < 1 or self.T_max < 1:
            raise ValueError(
                f"L({self.L}), K({self.K}), T_max({self.T_max}) must be positive"
            )
        # v5 relationship-regime fields (Pkg-09)
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
