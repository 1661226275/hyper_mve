"""Frozen EvalReport — rel-v1 (v5 Pkg-09; base: pkg-08 spec 01 §3.1).

rel-v1 [2026-07]: the per-c / c-segment / type-ratio / regret machinery of the
v4 schema is gone with c_t and the type system; the per-regime breakdown and
the belief regime-quality fields replace them. 29 payload fields + 1
``schema_version`` sentinel = 30 total. The dataclass-field-count + sentinel
lock lives in ``tests/integration/test_pkg08_drift_detectors.py``.

NOT in this schema (deliberately): NashConv / Price-of-Anarchy — the two
game-theoretic metrics are post-hoc deliverables (``eval/game_metrics.py``,
schema ``game-metrics-v1``) because best-response training is too expensive
to run at every eval.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Mapping

# Shared empty default for the optional per-agent map. dataclasses rejects any
# unhashable default (mappingproxy included), so it is handed out by a
# default_factory rather than used as a bare default.
_EMPTY_PER_AGENT: Mapping[int, tuple[float, ...]] = MappingProxyType({})


def regime_names_for(cfg) -> tuple[str, ...]:
    """Regime names in id order for a ``V4Config``-like object.

    Every :class:`EvalReport` producer passes this to ``regime_names`` so the
    integer keys of the per-regime dicts are self-describing. Returns ``()`` for
    configs with no resolvable relation family rather than raising: this field is
    provenance, and a whole evaluation should not be lost because the lookup
    failed.
    """
    try:
        from hyper_mve.utils.schemas.relation import get_regime_family
        return tuple(get_regime_family(cfg.env).names())
    except (AttributeError, ValueError, KeyError):
        return ()


@dataclass(frozen=True)
class EvalReport:
    """Single-run evaluation report, produced by:
      - BaselineModel.evaluate(env_fn, regime_grid, episodes) -> EvalReport
      - ExternalBaselineRunner.evaluate(env_fn, regime_grid, episodes) -> EvalReport
      - unified_evaluator.evaluate(runner, env_fn, cfg) -> EvalReport

    Schema is frozen here; drift trips the integration lock test.
    """

    # === Identity (5) — who/what was evaluated ===
    variant: str
    seed: int
    config_hash: str
    eval_mode: Literal["prior", "planner"]
    eval_planner_mode: Literal[
        "direct_inference",
        "planner_no_crn",
        "planner_no_coord_desc",
        "planner_full",
    ]

    # === Headline scalars (5) ===
    # return_mean is social TOTAL welfare (ΣR, subjective relational reward).
    # zero-shot semantics (v5): "seen" = regimes in cfg.env.train_regime_ids
    # (all regimes when None), "unseen" = evaluated regimes outside it.
    return_mean: float
    return_sem: float
    return_zero_shot_seen: float
    return_zero_shot_unseen: float
    return_zero_shot_gap: float

    # === Per-regime breakdown (3) — keyed by regime id ===
    return_per_regime: Mapping[int, float]
    return_per_regime_sem: Mapping[int, float]
    episodes_per_regime: Mapping[int, int]

    # === Planner-prior gap (3) — in-training eval semantic, retained ===
    planner_prior_return_gap: float
    direct_inference_return_mean: float
    planner_full_return_mean: float

    # === Diagnostics / provenance (5) ===
    walltime_seconds: float
    env_steps_evaluated: int
    episodes_total: int
    info_gating_strict: bool
    set_context_subjective_oracle_leak: bool

    # === Belief regime-quality (2) — belief-carrying variants; None otherwise ===
    # regime_accuracy: step-mean argmax accuracy of the BeliefNet posterior vs
    # oracle g. regime_nll is reserved (None until the analysis stage computes
    # it from stored posteriors); keeping the slot avoids a schema bump later.
    regime_accuracy: float | None = None
    regime_nll: float | None = None

    # === v5-thesis welfare metrics (4) ===
    # Physical welfare (Σu, regime-comparable), sustainability
    # (S = Σ_k q_k,Tmax / (K·Q_max)), fairness (F = 1 − N·σ(W_phys)/Σ W_phys),
    # tragedy indicator mean (T = 1[S < 0.2]). Default 0.0: runners that don't
    # surface them report a placeholder.
    welfare_physical_mean: float = 0.0
    sustainability_mean: float = 0.0
    fairness_mean: float = 0.0
    tragedy_index_mean: float = 0.0

    # === Regime identity (1) ===
    # Regime names in id order, i.e. regime_names[g] names the regime that the
    # per-regime dicts above key on g. Present because the integer ids are only
    # meaningful relative to a family: `g1` is `mutual_comp` under `g2` and
    # `asym_exploit` under `g2cm`, so an id-keyed report is not self-describing.
    # Empty tuple means the producer predates rel-v2 or has no regime family.
    regime_names: tuple[str, ...] = ()

    # === Per-agent return (1) ===
    # return_per_regime_per_agent[g][i] is agent i's mean subjective return in
    # regime g, so sum(return_per_regime_per_agent[g]) == return_per_regime[g].
    # The scalar alone cannot express individual optimality: it is a sum over
    # subjective rewards, and that sum cancels wherever the agents' rewards
    # oppose -- identically zero in the old g2 `mutual_comp`, and a single
    # agent's harvest in the asymmetric regimes. Empty mapping means the
    # producer predates rel-v3.
    return_per_regime_per_agent: Mapping[int, tuple[float, ...]] = field(
        default_factory=lambda: _EMPTY_PER_AGENT
    )

    # === Schema version sentinel (1) ===
    # rel-v2 (2026-08-09): added regime_names, and the g2cm family made regime
    # ids family-relative. An old and a new report must not be compared without
    # noticing, which is what this bump is for.
    # rel-v3 (2026-08-09): added return_per_regime_per_agent. The v7 re-run
    # scores on per-agent return + NashConv because the summed return cannot
    # express the individual optimality the method claims.
    schema_version: str = "rel-v3"

    def to_dict(self) -> dict:
        """JSON-safe plain-dict view of the report.

        The ``Mapping`` fields are stored as :class:`types.MappingProxyType`
        (not deep-copyable by ``dataclasses.asdict``), so we convert manually;
        int keys pass through (``json.dumps`` coerces them to strings).
        """
        from dataclasses import fields

        out: dict = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if hasattr(value, "items") and not isinstance(value, (str, bytes)):
                out[f.name] = dict(value.items())
            else:
                out[f.name] = value
        return out
