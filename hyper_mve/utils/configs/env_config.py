"""EnvConfig — RelationCommons dimensions and dynamics parameters (v5 Pkg-09).

The v4 fields (type_assignment, c_t evolution, α(c_t) regen range, Fehr-Schmidt
scalars, capability sampling ranges, neighbor-coupling knobs) were deleted with
the resource_commons environment in the Stage-6 cleanup. Roles live entirely in
the relationship matrix ``W(g)`` (``schemas.relation``).

v6 makes two pieces of physics selectable rather than fixed —
``regrowth_law`` and ``reward_coupling`` — because the v5 choices between them
made the hidden relationship provably worthless to infer (VoI ≡ 0; see
``results/analysis/regime_knowledge_ceiling.md``). Both default to the v5
behaviour, so existing presets are unchanged bit-for-bit.
"""
from __future__ import annotations

from dataclasses import dataclass

from hyper_mve.utils.schemas._constants import EPSILON_MOVE, Q_MAX
from hyper_mve.utils.schemas.relation import VALID_REWARD_COUPLINGS


_VALID_RELATION_FAMILIES: tuple[str, ...] = ("g2", "g2cm", "g4", "g4_ext", "tag4")
_VALID_REGIME_KERNELS: tuple[str, ...] = ("uniform",)
_VALID_ENV_KINDS: tuple[str, ...] = ("relation", "mpe_tag")
_VALID_REGROWTH_LAWS: tuple[str, ...] = ("constant", "logistic")


@dataclass(frozen=True)
class EnvConfig:
    """RelationCommons environment configuration (rel_* presets fill these)."""

    # Dimensions
    N: int
    L: int
    K: int
    T_max: int
    A: int = 6                          # NOOP + 4 directions + HARVEST

    # Resource dynamics
    Q_max: float = Q_MAX
    alpha: float = 0.10                 # regen rate (replaces v4 α(c_t))
    epsilon_move: float = EPSILON_MOVE  # per-move cost in the relational reward

    # v6: which regrowth law. "constant" (v5 default) regrows at α(Q_max − q),
    # i.e. FASTEST when the cell is empty — restraint has negative option value
    # and there is no commons dilemma at all. "logistic" regrows at
    # α·q·(1 − q/Q_max), so stock left in the ground compounds and restraint
    # becomes a real strategic dimension. See
    # results/analysis/regime_knowledge_ceiling.md.
    regrowth_law: str = "constant"

    # Relationship regimes (Pkg-09, `schemas.relation`)
    relation_family: str = "g2"
    relation_intensity: float = 1.0     # λ: ±λ off-diagonal weights
    regime_prior: tuple[float, ...] | None = None   # ρ over full G (None = uniform)
    regime_switch_prob: float = 0.0     # p: 0 = point 1 (episodic), >0 = point 2
    regime_kernel: str = "uniform"      # κ
    train_regime_ids: tuple[int, ...] | None = None  # holdout restriction at reset

    # v6: which row of W weights agent i's regard for u_j.
    # "own_row" (v5 default) uses w_ij — the agent's OWN row, which the
    # observation already carries, so the hidden w_ji never enters any reward
    # and inferring it is worth exactly nothing. "reciprocal" uses w_ji (Ŵ = Wᵀ)
    # — reciprocal altruism, and the weight is now hidden. "levine" interpolates
    # via (W + λWᵀ)/(1+λ); λ=0 is own_row exactly. See
    # schemas.relation.effective_coupling_matrix.
    reward_coupling: str = "own_row"
    reciprocity_lambda: float = 0.0     # λ, consumed only by "levine"

    # Environment kind (phase-3 realignment): "relation" = RelationCommons
    # (the fields above are its physics); "mpe_tag" = regime-ified MPE
    # simple_tag (L/K/Q_max/alpha unused; W(g) re-weights the raw per-agent
    # physical tag rewards). fixed_regime pins one regime for the entire run
    # (the mpe_tag_fixed calibration config uses the W=I regime).
    env_kind: str = "relation"
    fixed_regime: int | None = None

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
        if self.env_kind not in _VALID_ENV_KINDS:
            raise ValueError(
                f"Unknown env_kind: {self.env_kind!r} (valid: {_VALID_ENV_KINDS})"
            )
        if self.fixed_regime is not None and self.fixed_regime < 0:
            raise ValueError(f"fixed_regime must be ≥ 0, got {self.fixed_regime}")
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError(f"alpha={self.alpha} ∉ (0, 1]")
        if self.regrowth_law not in _VALID_REGROWTH_LAWS:
            raise ValueError(
                f"Unknown regrowth_law: {self.regrowth_law!r} "
                f"(valid: {_VALID_REGROWTH_LAWS})"
            )
        if self.reward_coupling not in VALID_REWARD_COUPLINGS:
            raise ValueError(
                f"Unknown reward_coupling: {self.reward_coupling!r} "
                f"(valid: {VALID_REWARD_COUPLINGS})"
            )
        if self.reciprocity_lambda < 0.0:
            raise ValueError(
                f"reciprocity_lambda must be ≥ 0, got {self.reciprocity_lambda}"
            )
