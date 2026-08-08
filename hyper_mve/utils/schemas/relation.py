"""Relationship regimes (v5 Dynamic-Relationship Markov Game, Pkg-09).

Single source of truth for the v5 formalization:

* **Regime / RegimeFamily** — the finite named family ``G`` of relationship
  matrices ``W(g) ∈ [-1, 1]^{N×N}`` with ``w_ii = 1``. Agent *i*'s role is
  its row ``w_i·``.
* **Sampling kernel** — ``g_0 ~ ρ`` at reset (:func:`sample_initial_regime`);
  per step ``g_{t+1} = g_t`` w.p. ``1 - p`` else ``~ κ(·|g_t)``
  (:func:`step_regime`). ``p = 0`` ⇒ research point 1 (per-episode static,
  Bayesian game); ``p > 0`` ⇒ research point 2 (hidden Markov switching).
* **Relational reward** (:func:`compute_relational_rewards`) — replaces the
  v4 Fehr-Schmidt mechanism entirely:

      ``R_i = (u_i + Σ_{j≠i} w_ij·u_j) / (1 + Σ_{j≠i} |w_ij|) - ε·1[moved_i]``

  Row-normalized so return scales are comparable across regimes. Special
  cases: ``W = I`` ⇒ selfish ``R_i = u_i - ε·moved``; all-ones ``W`` ⇒ every
  agent receives ``mean(u) - ε·moved`` (team reward).

Pure functions + frozen dataclasses, numpy only — no torch, no config
imports (``get_regime_family`` duck-types the config object).
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


_VALID_FAMILIES: tuple[str, ...] = ("g2", "g4", "g4_ext")
_VALID_KERNELS: tuple[str, ...] = ("uniform",)


# ---------------------------------------------------------------------------
# Regime / RegimeFamily
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Regime:
    """One relationship regime: a named ``W`` matrix.

    ``W`` is stored as a nested tuple (hashable / frozen); use
    :meth:`w_array` / :meth:`row` for numpy views.
    """

    id: int
    name: str
    W: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        n = len(self.W)
        for r, row in enumerate(self.W):
            if len(row) != n:
                raise ValueError(
                    f"Regime {self.name!r}: W row {r} has length {len(row)} != {n}"
                )
            for c, w in enumerate(row):
                if r == c and w != 1.0:
                    raise ValueError(
                        f"Regime {self.name!r}: diagonal w_{r}{c}={w} != 1.0"
                    )
                if not (-1.0 <= w <= 1.0):
                    raise ValueError(
                        f"Regime {self.name!r}: w_{r}{c}={w} ∉ [-1, 1]"
                    )

    def w_array(self, dtype=np.float32) -> np.ndarray:
        """``(N, N)`` numpy copy of ``W``."""
        return np.array(self.W, dtype=dtype)

    def row(self, i: int, dtype=np.float32) -> np.ndarray:
        """Agent *i*'s row ``w_i·`` **excluding** the diagonal — ``(N-1,)``,
        ascending ``j`` skipping ``j = i`` (same ordering convention as the
        v4 ẑ block)."""
        full = self.w_array(dtype=dtype)[i]
        return np.concatenate([full[:i], full[i + 1:]])


@dataclass(frozen=True)
class RegimeFamily:
    """The finite family ``G`` for a fixed agent count ``N``."""

    name: str
    N: int
    regimes: tuple[Regime, ...]

    def __post_init__(self) -> None:
        if not self.regimes:
            raise ValueError(f"RegimeFamily {self.name!r} is empty")
        for k, reg in enumerate(self.regimes):
            if reg.id != k:
                raise ValueError(
                    f"RegimeFamily {self.name!r}: regime ids must be contiguous "
                    f"0..{len(self.regimes) - 1}; got id={reg.id} at index {k}"
                )
            if len(reg.W) != self.N:
                raise ValueError(
                    f"RegimeFamily {self.name!r}: regime {reg.name!r} has "
                    f"N={len(reg.W)} != family N={self.N}"
                )

    @property
    def size(self) -> int:
        """``|G|``."""
        return len(self.regimes)

    def W_stack(self, dtype=np.float32) -> np.ndarray:
        """``(|G|, N, N)`` stack of all W matrices."""
        return np.stack([r.w_array(dtype=dtype) for r in self.regimes])

    def rows_stack(self, dtype=np.float32) -> np.ndarray:
        """``(|G|, N, N-1)`` — every agent's diagonal-free row per regime."""
        return np.stack(
            [
                np.stack([r.row(i, dtype=dtype) for i in range(self.N)])
                for r in self.regimes
            ]
        )

    def names(self) -> tuple[str, ...]:
        return tuple(r.name for r in self.regimes)


# ---------------------------------------------------------------------------
# Family builders
# ---------------------------------------------------------------------------

def _w2(w01: float, w10: float) -> tuple[tuple[float, ...], ...]:
    return ((1.0, w01), (w10, 1.0))


def build_g2(lam: float = 1.0) -> RegimeFamily:
    """``G2`` (N=2), 5 regimes. Asymmetric names are from agent 0's
    perspective: in ``asym_exploit`` agent 0 is hostile (``w_01 = -λ``)
    while agent 1 is supportive (``w_10 = +λ``) — agent 0 exploits agent 1;
    ``asym_exploited`` is the mirror."""
    _validate_intensity(lam)
    return RegimeFamily(
        name="g2",
        N=2,
        regimes=(
            Regime(0, "mutual_coop", _w2(+lam, +lam)),
            Regime(1, "mutual_comp", _w2(-lam, -lam)),
            Regime(2, "asym_exploit", _w2(-lam, +lam)),
            Regime(3, "asym_exploited", _w2(+lam, -lam)),
            Regime(4, "neutral", _w2(0.0, 0.0)),
        ),
    )


def _block_w(n: int, groups: Sequence[Sequence[int]], lam: float) -> tuple[tuple[float, ...], ...]:
    """W from a partition: +λ within a group, -λ across groups, 1 on the
    diagonal. ``groups`` must partition ``range(n)``."""
    member = {}
    for gi, grp in enumerate(groups):
        for a in grp:
            member[a] = gi
    if sorted(member) != list(range(n)):
        raise ValueError(f"groups {groups} do not partition range({n})")
    W = np.full((n, n), -lam, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if member[i] == member[j]:
                W[i, j] = +lam
    np.fill_diagonal(W, 1.0)
    return tuple(tuple(float(w) for w in row) for row in W)


def build_g4(lam: float = 1.0) -> RegimeFamily:
    """``G4`` (N=4), 5 regimes: all-coop, all-comp, three 2+2 pairings."""
    _validate_intensity(lam)
    return RegimeFamily(
        name="g4",
        N=4,
        regimes=(
            Regime(0, "all_coop", _block_w(4, [[0, 1, 2, 3]], lam)),
            Regime(1, "all_comp", _block_w(4, [[0], [1], [2], [3]], lam)),
            Regime(2, "pair_01_23", _block_w(4, [[0, 1], [2, 3]], lam)),
            Regime(3, "pair_02_13", _block_w(4, [[0, 2], [1, 3]], lam)),
            Regime(4, "pair_03_12", _block_w(4, [[0, 3], [1, 2]], lam)),
        ),
    )


def build_g4_ext(lam: float = 1.0) -> RegimeFamily:
    """``g4_ext`` (N=4), 9 regimes: :func:`build_g4` + the four 3+1
    coalitions (trio +λ within, -λ trio↔singleton)."""
    base = build_g4(lam)
    trios = (
        ("trio_012_3", [[0, 1, 2], [3]]),
        ("trio_013_2", [[0, 1, 3], [2]]),
        ("trio_023_1", [[0, 2, 3], [1]]),
        ("trio_123_0", [[1, 2, 3], [0]]),
    )
    extra = tuple(
        Regime(base.size + k, name, _block_w(4, groups, lam))
        for k, (name, groups) in enumerate(trios)
    )
    return RegimeFamily(name="g4_ext", N=4, regimes=base.regimes + extra)


def build_tag4(lam: float = 1.0) -> RegimeFamily:
    """``tag4`` (N=4, regime-ified MPE simple_tag; phase-3 realignment).

    Index convention: agents 0–2 are the predators (MPE ``adversary_*``),
    agent 3 is the prey (MPE ``agent_0``). Five predator-coalition regimes
    re-weight the raw per-agent physical tag rewards through ``W(g)``:

    * 0 ``pred_full_coalition`` — +λ among all three predators
    * 1 ``pred_pair_coalition`` — +λ between predators 0,1; predator 2 solo
    * 2 ``all_solo``            — W = I ⇒ **raw simple_tag rewards exactly**
      (the fixed-role calibration regime)
    * 3 ``pred_rivalry``        — −λ among all three predators
    * 4 ``prey_sympathizer``    — predators 0,1 keep +λ; predator 2 shares
      the prey's utility (+0.8λ symmetric) — a defecting predator
    """
    _validate_intensity(lam)

    def _w(entries: dict[tuple[int, int], float]) -> tuple[tuple[float, ...], ...]:
        W = np.eye(4, dtype=np.float64)
        for (i, j), w in entries.items():
            W[i, j] = w
            W[j, i] = w
        return tuple(tuple(float(x) for x in row) for row in W)

    lam = float(lam)
    return RegimeFamily(
        name="tag4",
        N=4,
        regimes=(
            Regime(0, "pred_full_coalition",
                   _w({(0, 1): +lam, (0, 2): +lam, (1, 2): +lam})),
            Regime(1, "pred_pair_coalition", _w({(0, 1): +lam})),
            Regime(2, "all_solo", _w({})),
            Regime(3, "pred_rivalry",
                   _w({(0, 1): -lam, (0, 2): -lam, (1, 2): -lam})),
            Regime(4, "prey_sympathizer",
                   _w({(0, 1): +lam, (2, 3): +0.8 * lam})),
        ),
    )


def _validate_intensity(lam: float) -> None:
    if not (0.0 < lam <= 1.0):
        raise ValueError(f"relation_intensity λ={lam} ∉ (0, 1]")


_BUILDERS = {"g2": build_g2, "g4": build_g4, "g4_ext": build_g4_ext,
             "tag4": build_tag4}


@functools.lru_cache(maxsize=None)
def _build_cached(family: str, lam: float) -> RegimeFamily:
    return _BUILDERS[family](lam)


def get_regime_family(env_cfg) -> RegimeFamily:
    """Resolve the family from an :class:`EnvConfig`-like object
    (needs ``.relation_family``, ``.relation_intensity``, ``.N``).

    Family↔N consistency is enforced *here* (not in ``EnvConfig``) so that
    legacy configs that never consume regimes stay valid.
    """
    family = env_cfg.relation_family
    if family not in _BUILDERS:
        raise ValueError(
            f"Unknown relation_family: {family!r} (valid: {_VALID_FAMILIES})"
        )
    fam = _build_cached(family, float(env_cfg.relation_intensity))
    if fam.N != env_cfg.N:
        raise ValueError(
            f"relation_family {family!r} requires N={fam.N}, got N={env_cfg.N}"
        )
    return fam


# ---------------------------------------------------------------------------
# Sampling kernel (ρ at reset, κ per step)
# ---------------------------------------------------------------------------

def _resolve_allowed(
    family: RegimeFamily,
    allowed_ids: Optional[Sequence[int]],
) -> tuple[int, ...]:
    if allowed_ids is None:
        return tuple(range(family.size))
    ids = tuple(int(i) for i in allowed_ids)
    if not ids:
        raise ValueError("allowed_ids must be non-empty")
    for i in ids:
        if not (0 <= i < family.size):
            raise ValueError(
                f"allowed regime id {i} out of range [0, {family.size})"
            )
    if len(set(ids)) != len(ids):
        raise ValueError(f"allowed_ids contains duplicates: {ids}")
    return ids


def sample_initial_regime(
    family: RegimeFamily,
    rng: np.random.Generator,
    *,
    prior: Optional[Sequence[float]] = None,
    allowed_ids: Optional[Sequence[int]] = None,
) -> int:
    """``g_0 ~ ρ`` — uniform over ``allowed_ids`` (default: all of G) unless
    an explicit ``prior`` over the *full* family is given (renormalized over
    the allowed subset)."""
    allowed = _resolve_allowed(family, allowed_ids)
    if prior is None:
        return int(rng.choice(allowed))
    p = np.asarray(prior, dtype=np.float64)
    if p.shape != (family.size,):
        raise ValueError(
            f"regime_prior length {p.shape} != |G|={family.size}"
        )
    if (p < 0).any():
        raise ValueError(f"regime_prior has negative entries: {prior}")
    sub = p[list(allowed)]
    total = sub.sum()
    if total <= 0:
        raise ValueError(
            f"regime_prior mass on allowed_ids {allowed} is zero"
        )
    return int(rng.choice(allowed, p=sub / total))


def step_regime(
    g: int,
    family: RegimeFamily,
    rng: np.random.Generator,
    *,
    switch_prob: float,
    kernel: str = "uniform",
    allowed_ids: Optional[Sequence[int]] = None,
) -> int:
    """One step of the regime chain: stay w.p. ``1 - switch_prob``, else jump
    per ``κ``. ``kernel="uniform"``: uniform over the allowed set minus the
    current regime (stays put if no other regime is allowed).

    ``switch_prob = 0`` (research point 1) never draws from ``rng`` — the
    episode's random stream is identical to a chain-free env.
    """
    if kernel not in _VALID_KERNELS:
        raise ValueError(f"Unknown regime_kernel: {kernel!r} (valid: {_VALID_KERNELS})")
    if not (0.0 <= switch_prob <= 1.0):
        raise ValueError(f"switch_prob={switch_prob} ∉ [0, 1]")
    if switch_prob == 0.0:
        return int(g)
    if rng.random() >= switch_prob:
        return int(g)
    allowed = _resolve_allowed(family, allowed_ids)
    others = [i for i in allowed if i != g]
    if not others:
        return int(g)
    return int(rng.choice(others))


# ---------------------------------------------------------------------------
# Relational reward (the citable formula — thesis Ch3, Pkg-09)
# ---------------------------------------------------------------------------

VALID_REWARD_COUPLINGS: tuple[str, ...] = ("own_row", "levine", "reciprocal")


def effective_coupling_matrix(
    W: np.ndarray,
    coupling: str = "own_row",
    reciprocity_lambda: float = 0.0,
) -> np.ndarray:
    """The matrix the reward actually mixes with — ``W`` under some coupling rule.

    Agent ``i``'s reward weight on ``u_j`` comes from row ``i`` of the result:

    * ``own_row`` (default, v5): ``Ŵ = W``. The weight is ``w_ij`` — the agent's
      **own** row, which the observation already carries
      (``observations.py`` block 5). The hidden ``w_ji`` never enters any
      agent's reward, so the best response cannot depend on it and the value of
      inferring it is identically zero. Measured, three ways:
      ``results/analysis/regime_knowledge_ceiling.md``.
    * ``reciprocal`` (v6): ``Ŵ = Wᵀ``. Agent ``i``'s weight on ``u_j`` is
      ``w_ji`` — how much ``j`` values ``i``. Reciprocal altruism: I internalise
      your welfare exactly as much as you internalise mine. The weight is now a
      *hidden* variable, which is what makes inference pay.
    * ``levine``: ``Ŵ = (W + λ·Wᵀ)/(1+λ)`` — Levine (1998) adjusted altruism,
      interpolating the two. ``λ=0`` is ``own_row`` exactly and ``λ→∞`` tends to
      ``reciprocal``, so λ is a continuous knob for a sweep. Note the weight
      only changes **sign** with the hidden row once ``λ > 1``, and that sign
      flip is what carries essentially all of the value of information.

    Every rule preserves the unit diagonal automatically (``Wᵀ`` shares it, and
    ``(1 + λ·1)/(1 + λ) = 1``), so the caller's diagonal assert still holds.
    """
    if coupling not in VALID_REWARD_COUPLINGS:
        raise ValueError(
            f"Unknown reward coupling {coupling!r} "
            f"(valid: {VALID_REWARD_COUPLINGS})"
        )
    W = np.asarray(W, dtype=np.float32)
    if coupling == "own_row":
        return W
    if coupling == "reciprocal":
        return W.T
    lam = float(reciprocity_lambda)
    if lam < 0.0:
        raise ValueError(f"reciprocity_lambda must be ≥ 0, got {lam}")
    return ((W + lam * W.T) / (1.0 + lam)).astype(np.float32)


def compute_relational_rewards(
    *,
    harvests: np.ndarray,
    moved_mask: np.ndarray,
    W: np.ndarray,
    epsilon_move: float,
    coupling: str = "own_row",
    reciprocity_lambda: float = 0.0,
) -> np.ndarray:
    """``R_i = (u_i + Σ_{j≠i} ŵ_ij·u_j) / (1 + Σ_{j≠i} |ŵ_ij|) - ε·1[moved_i]``.

    ``Ŵ = effective_coupling_matrix(W, coupling, reciprocity_lambda)``; the
    default ``own_row`` leaves ``Ŵ = W``, i.e. the v5 formula unchanged.

    Args:
        harvests:   ``(N,)`` per-agent physical harvest ``u_i`` (≥ 0).
        moved_mask: ``(N,) bool`` — move *intent* (cost independent of
                    execution success, same contract as v4).
        W:          ``(N, N)`` relationship matrix, diagonal exactly 1.
        epsilon_move: per-move cost ε.
        coupling:   which row of ``W`` weights agent ``i``'s regard for ``u_j``
                    — see :func:`effective_coupling_matrix`.
        reciprocity_lambda: λ, consumed only by ``coupling="levine"``.

    Returns:
        ``(N,) float32`` rewards. Bound: ``|R_i + ε·moved_i| ≤ max_j u_j``
        (convex-combination property of the row normalization; the coupling
        rules preserve it because they preserve the unit diagonal).
    """
    u = np.asarray(harvests, dtype=np.float32)
    N = u.shape[0]
    W = np.asarray(W, dtype=np.float32)
    if W.shape != (N, N):
        raise AssertionError(f"W shape {W.shape} != ({N}, {N})")
    if not np.allclose(np.diagonal(W), 1.0):
        raise AssertionError(f"W diagonal must be 1.0, got {np.diagonal(W)}")

    W_eff = effective_coupling_matrix(W, coupling, reciprocity_lambda)
    mixed = W_eff @ u                            # u_i + Σ_{j≠i} ŵ_ij·u_j
    denom = np.abs(W_eff).sum(axis=1)            # 1 + Σ_{j≠i} |ŵ_ij|
    move_cost = epsilon_move * np.asarray(moved_mask, dtype=np.float32)
    return (mixed / denom - move_cost).astype(np.float32)
