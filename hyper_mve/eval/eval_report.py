"""Frozen 32-field EvalReport — pkg-08 spec 01 §3.1 (mother-doc).

Byte-faithful to the spec block. Any drift triggers
``test_eval_report_schema_lock_matches_spec_08.py`` (pkg-08 spec 01 §10.8) and
``test_eval_report_32_field_dataclass_lock`` (pkg-08 spec 08 §9.1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping


@dataclass(frozen=True)
class EvalReport:
    """Single-run evaluation report, produced by:
      - BaselineModel.evaluate(env_fn, c_grid, episodes) -> EvalReport     [pkg-07 spec 01 §3.2]
      - ExternalBaselineRunner.evaluate(env_fn, c_grid, episodes) -> EvalReport  [pkg-07 spec 04 §10]
      - unified_evaluator.evaluate(runner, env_fn, cfg) -> EvalReport     [pkg-08 spec 01]

    Schema is frozen here AND in pkg-08 spec 08 §3 (byte-identical). Any drift triggers
    test_eval_report_schema_lock_matches_spec_08.py failure.
    """

    # === Identity (6) — who/what was evaluated ===
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
    c_visible: bool

    # === Headline scalars (5) ===
    return_mean: float
    return_sem: float
    return_zero_shot_seen: float
    return_zero_shot_unseen: float
    return_zero_shot_gap: float

    # === Per-c breakdown (3) — zero-shot full grid ===
    return_per_c: Mapping[float, float]
    return_per_c_sem: Mapping[float, float]
    episodes_per_c: Mapping[float, int]

    # === c-segment aggregation (2) — Ch6.2.4 ===
    return_per_segment: Mapping[tuple[float, float], float]
    return_per_segment_sem: Mapping[tuple[float, float], float]

    # === Bell-curve type-ratio sweep (2) — Ch6.6 ===
    return_per_type_ratio: Mapping[tuple[int, int], float]
    return_per_type_ratio_sem: Mapping[tuple[int, int], float]

    # === Regret vs oracle ceiling (4) — spec 02 §3 ===
    regret_per_c: Mapping[float, float]
    regret_mean: float
    oracle_ceiling_per_c: Mapping[float, float]
    oracle_ceiling_cache_hit: Mapping[float, bool]

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

    # === Optional belief diagnostics (2) — hyper-only; None for every other variant ===
    belief_c_mae: float | None = None
    belief_c_calibration: float | None = None

    # === Schema version sentinel (1) — for forward migration ===
    schema_version: str = "pkg08-spec01-v1"

    def to_dict(self) -> dict:
        """JSON-safe plain-dict view of the report.

        Used by the sweep worker (``_sweep_worker.py`` prefers ``to_dict`` over
        ``dataclasses.asdict``). Two reasons we cannot use ``asdict`` here:

          * ``asdict`` deep-copies every field, and the ``Mapping`` fields are
            stored as :class:`types.MappingProxyType`, which is not
            deep-copyable (``TypeError: cannot pickle 'mappingproxy' object``).
          * The tuple-keyed maps (``return_per_segment``,
            ``return_per_type_ratio`` and their ``_sem`` twins) have tuple keys,
            which ``json.dumps`` rejects. We stringify those keys here.

        Float / int / bool / None keys pass through unchanged (``json.dumps``
        coerces numeric keys to strings on its own). The conversion is one-way;
        nothing in the pipeline reads these maps back by key.
        """
        from dataclasses import fields

        def _key(k):
            if isinstance(k, tuple):
                return ",".join(str(x) for x in k)
            return k

        out: dict = {}
        for f in fields(self):
            value = getattr(self, f.name)
            # Detect mapping-like fields (MappingProxyType included) without
            # importing the concrete proxy type.
            if hasattr(value, "items") and not isinstance(value, (str, bytes)):
                out[f.name] = {_key(k): v for k, v in value.items()}
            else:
                out[f.name] = value
        return out
