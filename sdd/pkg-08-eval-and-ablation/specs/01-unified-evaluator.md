# Spec 01 — Unified Evaluator (`unified_evaluator.evaluate`) + `@dataclass(frozen=True) EvalReport`

> **Parent docs**: [`../proposal.md`](../proposal.md) §2.1 · [`../design.md`](../design.md) §3.2 / §3.3 / §4 D2 / §4 D3 / §4 D7 / §4 D10 · [`../README.md`](../README.md) C8-EVAL-REUSE1 / C8-EVAL-SCHEMA1 / C8-EVAL-MODE1 / C8-EVAL-SELF1 / C8-EVAL-SEG1 / C8-EVAL-BELL1
> **Anchors**: this spec is the **schema mother-doc** of pkg-08. `@dataclass(frozen=True) EvalReport` defined here is consumed verbatim by every other pkg-08 spec (02 zero-shot/regret, 03 four-mode planner, 04 μP, 05 sweep harness, 07 stats/compare) and by the two reverse-consumed pkg-07 contracts (spec 01 `BaselineLike Union`, spec 04 `ResourceCommonsPettingZooEnv`, spec 08 `evaluate(env_fn, c_grid, episodes) -> EvalReport`).
> **Status**: SDD only — describes the contract for `hyper_mve/eval/unified_evaluator.py` + `hyper_mve/eval/eval_report.py`, not the implementation.

---

## ⚠️ Header — three hard locks

### Lock 1 — `training/evaluation.py:run_eval` is NOT replaced

`hyper_mve/training/evaluation.py:run_eval(model, cfg, global_step=0) -> dict[str, float]` (the existing in-training periodic eval introduced in v4-opt 2026-06) is **wrapped, not replaced**. The unified evaluator imports it and calls it as an inner subroutine for the `direct_inference` (≡ legacy `prior`) and `planner_full` (≡ legacy `planner`) modes; the c-grid loop, dual-mode dispatch, CRN seed `_EVAL_PLANNER_SEED`, env seed base `_EVAL_ENV_SEED_BASE`, and tie-break seed `_EVAL_TIEBREAK_SEED` defined in that file are reused verbatim. All existing TB tags (`eval/prior/return_total_c0.5`, `eval/planner/return_total`, `eval/planner_prior_gap`, …) keep emitting unchanged. A named unit test `test_run_eval_not_modified.py` (§7) asserts the source of `run_eval` is byte-identical to a snapshot in `tests/eval/fixtures/run_eval_baseline.py` (regenerated only on intentional updates, with a paper-trail commit), so that any drift triggers a CI failure.

### Lock 2 — `EvalReport` is the unique evaluator output schema

Every entry point on a baseline (internal `BaselineModel.evaluate(env_fn, c_grid, episodes) -> EvalReport`; external `ExternalBaselineRunner.evaluate(env_fn, c_grid, episodes) -> EvalReport`; pkg-08 `unified_evaluator.evaluate(runner, env_fn, cfg) -> EvalReport`) returns the **exact same** `@dataclass(frozen=True) EvalReport` defined in §3 of this spec. Spec 08 §3 mirrors this schema as the "integration contract lock" and is byte-identical (drift caught by `test_eval_report_schema_lock_matches_spec_08.py`). Pkg-07 spec 01 §2 `BaselineLike Union` and spec 08 `evaluate()` signature **reverse-consume** this lock — any addition / removal / type change to `EvalReport` requires a synchronous edit of pkg-07 spec 08 (drift detector in pkg-08 spec 08 §6 forces reviewer reconciliation).

### Lock 3 — Self-Info strict at eval time (C8-EVAL-SELF1 inherits pkg-04 spec 02 + pkg-07 C7-INT-SELF1)

The unified evaluator **MUST NOT** pass oracle types (`info["types"]`) or oracle context (`info["c_true"]`) into `set_context_subjective(agent_id, cap_i, belief)`. Two distinct flags govern this — declared separately to prevent conflation:

1. **`oracle_mode: bool`** (env-level adapter flag, owned by pkg-07 spec 04 §7): set on the `ResourceCommonsPettingZooEnv` constructor. Default `False`. When `False`, the adapter's `_filter_info` strips `_oracle_fields` (= `("c_true", "types")`) from every per-agent info dict. The unified evaluator's layer-1 guard (§7.1) asserts `env._oracle_mode is False` at entry — populates `EvalReport.info_gating_strict`.
2. **`cfg.eval.use_oracle_types: bool`** (evaluator-level config flag, owned by pkg-01 spec 05 + pkg-08 design D10): governs whether the evaluator is **allowed** to construct `env_fn` with `oracle_mode=True` at all. Default `False`. The only legal `True` callers are the curriculum-override `oracle_only` variant (spec 06 D9 + pkg-07 spec 01) — and even then the oracle types are consumed by the curriculum override, not by `set_context_subjective`.

A named test `test_self_info_strict_at_eval.py` (§7) parametrises every variant in `REGISTRY ∪ {"hyper"}` and asserts no oracle leak ever crosses the evaluator boundary via either flag's escape hatch.

---

## 1. Purpose

The unified evaluator exists to:

1. **Extend** `training/evaluation.py:run_eval` from "single in-training periodic eval scalar dict" to a full **`EvalReport` producer** consumed by the paper's Ch6 main table, four ablation tables, zero-shot table, and comparison plots — **without** modifying `run_eval` itself. The c-grid loop and TB tag family in `run_eval` are stable and battle-tested in optimisation-phase runs; reusing them as a subroutine (rather than re-implementing) is the cheapest path to a stable schema (design D3 rationale).
2. **Unify** internal (`BaselineModel`-style; in-process eval reusing trainer model state) and external (`ExternalBaselineRunner`-style; runner self-contained eval with its own trainer/critic state) eval paths behind a single dispatch signature, so that sweep harness (spec 05) can iterate `for v in REGISTRY: report = unified_evaluator.evaluate(runner, env_fn, cfg)` without `if isinstance(runner, BaselineModel)` branches.
3. **Add the aggregation axes** the paper needs but the in-training eval does not have:
   - **c-grid aggregation** over `cfg.eval.zero_shot_test_c` (7 values: train ∪ unseen),
   - **c-segment three-way aggregation** over `cfg.eval.c_segments` (`[0,0.3]` / `[0.3,0.7]` / `[0.7,1.0]`, Ch6.2.4),
   - **bell-curve type-ratio sweep** over `cfg.eval.bell_curve_type_ratios` (`(n_alpha, n_beta)` pairs, Ch6.6),
   - **four-mode planner eval dispatch** (`direct_inference` / `planner_no_crn` / `planner_no_coord_desc` / `planner_full`, spec 03),
   - **regret vs oracle ceiling** with cache (spec 02 §3),
   - **zero-shot seen / unseen / gap** headline scalars (spec 02 §2),
   - **planner-prior gap** scalar (carried verbatim from `run_eval` output `planner_prior_gap`),
   - **belief-head diagnostics** (`belief_c_mae` / `belief_c_calibration`, populated only for `hyper` variant; `None` for every internal / external baseline).
4. **Lock a hashable + JSON-serialisable `EvalReport`** so sweep harness (spec 05) can write each report to `runs/<run_tag>/eval_report.json` (separate file) and to `runs/registry.jsonl` (summary columns only) without surprise `set` / `list` / `np.ndarray` fields breaking serialisation.

The deliverable is two files (no others added in this spec):
- `hyper_mve/eval/eval_report.py` — the `@dataclass(frozen=True) EvalReport` class.
- `hyper_mve/eval/unified_evaluator.py` — `evaluate(runner, env_fn, cfg) -> EvalReport`.

---

## 2. Unified evaluator signature

### 2.1 File path and import surface

```python
# hyper_mve/eval/unified_evaluator.py

from __future__ import annotations

from typing import Callable

from hyper_mve.baselines import BaselineLike            # pkg-07 spec 01 §3.2
from hyper_mve.configs import V4Config
from hyper_mve.envs.adapters.pettingzoo_wrapper import ResourceCommonsPettingZooEnv
from hyper_mve.eval.eval_report import EvalReport


def evaluate(
    runner: BaselineLike,
    env_fn: Callable[[], ResourceCommonsPettingZooEnv],
    cfg: V4Config,
) -> EvalReport: ...
```

### 2.2 Argument contract (locked)

| Arg | Type | Meaning |
|----|------|--------|
| `runner` | `BaselineLike = BaselineModel \| ExternalBaselineRunner` (pkg-07 spec 01 §3.2) | The trained baseline being evaluated. For `BaselineModel` (5 internal variants + the curriculum-override `hyper` / `oracle_only` / `infer_only` constructed from `HyperMuZeroModel(cfg)`), evaluation is in-process and reuses the trainer's model state (no checkpoint round-trip). For `ExternalBaselineRunner` (3 Tier-1 + MAMBA), evaluation defers to the runner's own `.evaluate(env_fn, c_grid, episodes) -> EvalReport` and the unified evaluator wraps the returned report (no schema rewrite). |
| `env_fn` | `Callable[[], ResourceCommonsPettingZooEnv]` (pkg-07 spec 04 §10) | Eval-time env factory. Constructed by the caller (typically pkg-08 sweep harness) with **the eval-time flag pair**: `oracle_mode=False` AND `eval_info_mode=True` (the only place in the codebase where `eval_info_mode=True` is legal, per pkg-07 spec 04 §10). The unified evaluator **MUST NOT** override these flags. Spec 02 (c_hidden) is the unique downstream consumer that additionally inspects `env_fn()._env.cfg.c_visible`. |
| `cfg` | `V4Config` | Full resolved config. The unified evaluator reads `cfg.eval.*` (existing reserved fields **plus the 2 new pkg-08 fields landing on `EvalConfig`**: `eval_planner_mode` + `eval_use_planner_direct_inference`; the other 3 of design D10's 5 new fields land on `EnvConfig`/`TrainConfig`, not `EvalConfig`), `cfg.env.c_visible`, `cfg.env.type_assignment`, and `cfg.baselines.*` (read-only — only for `config_hash` computation, never for eval-time mutation). Exact `cfg.eval` field list is the authoritative source at `hyper_mve/configs/eval_config.py` after pkg-01 spec 05 sync. |

### 2.3 Return contract

Returns a single `EvalReport` (§3). The instance is `frozen=True` — the evaluator constructs it once at the end of `evaluate()` and never mutates. Every `Mapping[K, V]` field is wrapped in `types.MappingProxyType` (immutable view). Every tuple-keyed mapping uses native `tuple` (hashable). Every numeric field is `float` or `int` (no `np.float32` / `np.int64` — coerced to Python scalars at construction so JSON serialisation in spec 05 is one `json.dumps(asdict(report))` away).

### 2.4 What the unified evaluator does NOT take

- **No `c_grid` arg**. The c-grid is sourced from `cfg.eval.zero_shot_test_c` (the canonical 7-value train-∪-unseen grid). Spec 02 locks this to be the same grid as `cfg.eval.eval_c_grid` extended to 7 values; in-training eval continues to use the smaller `eval_c_grid` (3 values) via the unchanged `run_eval`.
- **No `episodes` arg**. Episode count per c is sourced from `cfg.eval.evaluate_episodes` (the reserved field at `eval_config.py:19`, default 30). The 4-mode planner eval (spec 03) sub-divides this budget across modes; spec 03 §3 owns the budget split table.
- **No `eval_mode` arg**. The eval mode is sourced from `cfg.eval.eval_planner_mode: Literal[...]` (the new field at design D10, default `"planner_full"`). For backward compatibility with the legacy `run_eval` two-mode semantics, the unified evaluator additionally emits the `direct_inference` mode's mean (≡ legacy `prior/return_total`) and the `planner_full` mode's mean (≡ legacy `planner/return_total`) as separate scalars in the report (§3 `direct_inference_return_mean` / `planner_full_return_mean`), even when `cfg.eval.eval_planner_mode` is set to a single mode.

---

## 3. `EvalReport` schema — verbatim field enumeration

> Every field is declared `@dataclass(frozen=True)`. Total field count: **32** (6 identity + 5 headline + 3 per-c + 2 c-segment + 2 bell-curve + 4 regret + 3 planner-prior + 3 diagnostics + 2 nullable belief + 1 schema_version sentinel + 1 oracle-leak flag = 32). Spec 08 §3 mirrors this list byte-identically. **Authoritative source**: the `@dataclass` body in §3.1 below — count by reading it; the bucket breakdown is a navigational aid, the body is the truth.

### 3.1 The full `@dataclass`

```python
# hyper_mve/eval/eval_report.py

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping


@dataclass(frozen=True)
class EvalReport:
    """Single-run evaluation report, produced by:
      - BaselineModel.evaluate(env_fn, c_grid, episodes) -> EvalReport     [pkg-07 spec 01 §3.2]
      - ExternalBaselineRunner.evaluate(env_fn, c_grid, episodes) -> EvalReport  [pkg-07 spec 04 §10]
      - unified_evaluator.evaluate(runner, env_fn, cfg) -> EvalReport     [this spec]

    Schema is frozen here AND in pkg-08 spec 08 §3 (byte-identical). Any drift triggers
    test_eval_report_schema_lock_matches_spec_08.py failure.
    """

    # === Identity (6) — who/what was evaluated ===
    variant: str                                       # CLI string ("hyper" / "baseline_input_wide" / "external_mappo" / ...; 14 legal values from pkg-07 spec 01 §2)
    seed: int                                          # the seed the run was trained under
    config_hash: str                                   # SHA1 of (variant, env_cfg, model_cfg, train_cfg); used as oracle-ceiling cache key (spec 02 §3)
    eval_mode: Literal["prior", "planner"]             # legacy 2-mode label; kept for run_eval back-compat
    eval_planner_mode: Literal[                        # spec 03 four-mode label (design D7)
        "direct_inference",
        "planner_no_crn",
        "planner_no_coord_desc",
        "planner_full",
    ]
    c_visible: bool                                    # spec 02 c_hidden flag (design D4); True default

    # === Headline scalars (5) ===
    return_mean: float                                 # mean across all eval episodes (all c, all modes)
    return_sem: float                                  # standard error of the mean
    return_zero_shot_seen: float                       # mean over c in cfg.eval.zero_shot_train_c
    return_zero_shot_unseen: float                     # mean over c in cfg.eval.zero_shot_unseen_c
    return_zero_shot_gap: float                        # seen - unseen (headline gap; spec 02 §2)

    # === Per-c breakdown (3) — zero-shot full grid ===
    return_per_c: Mapping[float, float]                # c -> mean return; 7 c-vals (cfg.eval.zero_shot_test_c)
    return_per_c_sem: Mapping[float, float]            # c -> SEM
    episodes_per_c: Mapping[float, int]                # c -> episode count (for SEM correctness)

    # === c-segment aggregation (2) — Ch6.2.4 ===
    return_per_segment: Mapping[tuple[float, float], float]      # (low, high) -> mean (segments from cfg.eval.c_segments)
    return_per_segment_sem: Mapping[tuple[float, float], float]  # (low, high) -> SEM

    # === Bell-curve type-ratio sweep (2) — Ch6.6 ===
    return_per_type_ratio: Mapping[tuple[int, int], float]       # (n_alpha, n_beta) -> mean (ratios from cfg.eval.bell_curve_type_ratios)
    return_per_type_ratio_sem: Mapping[tuple[int, int], float]

    # === Regret vs oracle ceiling (4) — spec 02 §3 ===
    regret_per_c: Mapping[float, float]                # c -> ceiling(c) - method(c)
    regret_mean: float                                 # mean across c grid
    oracle_ceiling_per_c: Mapping[float, float]        # c -> ceiling value (diagnostic; from runs/_oracle_ceilings/<config_hash>/<c>.json)
    oracle_ceiling_cache_hit: Mapping[float, bool]     # c -> True if cache hit; False if recomputed via oracle_only

    # === Planner-prior gap (3) — in-training eval semantic, retained ===
    planner_prior_return_gap: float                    # planner_full - direct_inference (matches run_eval's planner_prior_gap; spec 03 §4)
    direct_inference_return_mean: float                # exported separately for 4-mode comparison
    planner_full_return_mean: float                    # exported separately

    # === Diagnostics / provenance (5) ===
    walltime_seconds: float                            # total eval wall time
    env_steps_evaluated: int                           # cumulative env steps (for normalisation across baselines)
    episodes_total: int
    info_gating_strict: bool                           # CANONICAL DEFINITION (single source of truth, §6.2/§7.1 reference verbatim):
                                                       #   info_gating_strict = (env_fn().()._oracle_mode is False)
                                                       # I.e. the runner did NOT flip oracle_mode to True. eval_info_mode may be True
                                                       # (unique caller permitted; pkg-07 spec 04 §10 + §7.1 layer 1 below).
                                                       # Equivalent test expression: `env = env_fn(); assert env._oracle_mode is False`.
    set_context_subjective_oracle_leak: bool           # hyper internal: True iff any oracle type ever crossed into set_context_subjective during this eval (MUST be False)

    # === Optional belief diagnostics (2) — hyper-only; None for every other variant ===
    belief_c_mae: float | None = None                  # mean abs err of ĉ vs c_true (c_hidden authenticity check, Theory Audit Q7)
    belief_c_calibration: float | None = None          # |E[ĉ] - c_true| over the eval window

    # === Schema version sentinel (1) — for forward migration ===
    schema_version: str = "pkg08-spec01-v1"            # bumped only by synchronous spec 01 + spec 08 §3 edit
```

### 3.2 Why every field is here (one-line rationale per field group)

| Group | Why these fields | Where consumed |
|-------|------------------|----------------|
| Identity (6) | Distinguish two `EvalReport`s in the registry (variant + seed + hash) and label the mode axis (legacy 2 + new 4) + `c_visible` flag. | spec 05 sweep harness, spec 07 stats |
| Headline (5) | The four numbers every Ch6 table needs: overall mean, SEM, zero-shot seen, zero-shot unseen, and the gap. SEM is for Welch t (spec 07). | Ch6 main table, ablation tables |
| Per-c (3) | Per-c breakdown for the zero-shot table (Ch6.9) and the c-grid plot (Ch6.2). Episodes count is needed because SEM = std / sqrt(N). | spec 02 zero-shot, Ch6 zero-shot table |
| c-segment (2) | Ch6.2.4 three-segment table. Segments are configurable but locked to three for the paper. | Ch6.2.4 segment table |
| Bell-curve (2) | Ch6.6 type-ratio bell curve. 6 ratios cover the spectrum (0,4) → (4,0). | Ch6.6 bell-curve plot |
| Regret (4) | Ch6.7 (Assertion D regret-vs-ceiling) table; cache_hit diagnostic exposes when ceiling was recomputed. | spec 02 regret, Ch6.7 |
| Planner-prior gap (3) | Distillation residual — the original `run_eval` headline. Retained so existing TB curves still parse. | spec 03 mode dispatch, existing TB |
| Diagnostics (5) | walltime/env_steps for normalisation; `info_gating_strict` / `set_context_subjective_oracle_leak` are integrity verification bits the reviewer can grep for. | spec 05 registry, integrity tests |
| Belief diagnostics (2) | hyper-only; nullable for non-hyper variants. Authenticates ĉ in c_hidden runs (Theory Audit Q7). | spec 02 c_hidden, Theory Audit |
| Schema version (1) | Forward migration sentinel. Bumped only on synchronous spec 01 + spec 08 §3 edit. | spec 08 §6 drift detector |

### 3.3 Hashability and JSON-serialisability

- **Hashable**: every `Mapping[float, float]` and `Mapping[tuple[int,int], float]` uses native `tuple` keys (floats are hashable as long as they are not NaN; the evaluator rejects any NaN-keyed c-value at construction). The dataclass itself is `frozen=True`, so `hash(report)` works.
- **JSON-serialisable**: spec 05 writes `runs/<run_tag>/eval_report.json` via `json.dumps(dataclasses.asdict(report), default=_json_default)` where `_json_default` (a helper at `eval/eval_report.py`) (i) flattens `MappingProxyType` to plain dict, (ii) converts tuple keys via `json.dumps(list(key))` for the dict key (round-trip via a custom `_json_object_hook` that parses tuple keys back), and (iii) coerces `np.float32` to `float` (defensive). Spec 05 §4 owns the serialisation contract; this spec only guarantees the dataclass is amenable to it.

### 3.4 What is NOT in `EvalReport`

By design, per-episode raw returns (the `Sequence[float]` of every episode's total return) are **not** in the report. They are written to a sibling file `runs/<run_tag>/eval_episodes.csv` by spec 05 §4 so that downstream stats (Welch t, KS test, bootstrap) can reload them, but the report itself stays small enough to live in `runs/registry.jsonl` as a single line (no megabyte rows). The fields `return_mean`, `return_sem`, and `episodes_per_c` are sufficient summary statistics for every Ch6 table the paper plans (Welch t input is `(mean, sem, n)` per group, all three present).

---

## 4. Wrapping (NOT replacing) `training/evaluation.py:run_eval`

### 4.1 The "extension not replacement" pattern

The unified evaluator's inner loop for the `direct_inference` and `planner_full` modes calls `run_eval(model, cfg_with_c_pinned, global_step=0)` once per c-value, harvests the returned flat dict, and aggregates into the `EvalReport.return_per_c` field. The remaining 2 modes (`planner_no_crn`, `planner_no_coord_desc`) require planner-state mutation that the current `run_eval` does not expose; spec 03 §3 owns the dispatch table for these 2 modes (in-process planner construction with explicit `use_crn=False` / `randomize_order=False` overrides, bypassing the eval-time defaults but reusing the same env seed base and tie-break seed for CRN-across-evaluations parity).

### 4.2 Mechanism (pseudocode)

```python
# hyper_mve/eval/unified_evaluator.py (sketch)

from dataclasses import replace
from hyper_mve.training.evaluation import run_eval


def _evaluate_internal_hyper_or_baseline(
    runner,                          # BaselineModel (in-process)
    env_fn,                          # Callable[[], ResourceCommonsPettingZooEnv]
    cfg,                             # V4Config
) -> EvalReport:
    model = runner if hasattr(runner, "set_context_subjective") else runner.model
    # 1. Run legacy two-mode eval first (covers direct_inference + planner_full).
    #    Output is the flat dict run_eval has always emitted.
    legacy_results = run_eval(model, cfg, global_step=0)
    # 2. For each c in cfg.eval.zero_shot_test_c, harvest legacy_results into
    #    return_per_c. (run_eval already does the c-grid loop over
    #    cfg.eval.eval_c_grid; for the 4 extra c values the unified evaluator
    #    runs run_eval again with cfg.eval.eval_c_grid swapped via dataclasses.replace.)
    return_per_c = {}
    sem_per_c = {}
    episodes_per_c = {}
    for c in cfg.eval.zero_shot_test_c:
        if c in cfg.eval.eval_c_grid:
            return_per_c[c] = legacy_results[f"planner/return_total_c{c:g}"]
            # SEM derived from raw episode returns: see §4.3.
        else:
            cfg_c = replace(cfg, eval=replace(cfg.eval, eval_c_grid=(c,)))
            extra = run_eval(model, cfg_c, global_step=0)
            return_per_c[c] = extra[f"planner/return_total_c{c:g}"]
    # 3. For the 2 extra planner modes (no_crn, no_coord_desc), spec 03 §3
    #    dispatch is called here (returns its own per-c dict).
    no_crn_per_c, no_coord_desc_per_c = _spec03_dispatch(model, cfg, env_fn)
    # 4. Aggregate into c-segments and bell-curve (§5, §6 of this spec).
    # 5. Compute regret_per_c via spec 02 §3 (oracle ceiling cache lookup).
    # 6. Assemble the frozen EvalReport.
    return EvalReport(
        variant=...,
        seed=...,
        config_hash=...,
        eval_mode="planner",                 # legacy semantic kept
        eval_planner_mode=cfg.eval.eval_planner_mode,
        c_visible=cfg.env.c_visible,
        return_mean=...,
        return_sem=...,
        return_zero_shot_seen=...,
        return_zero_shot_unseen=...,
        return_zero_shot_gap=...,
        return_per_c=MappingProxyType(return_per_c),
        ...,
        planner_prior_return_gap=legacy_results["planner_prior_gap"],
        direct_inference_return_mean=mean(legacy_results, prefix="prior/"),
        planner_full_return_mean=mean(legacy_results, prefix="planner/"),
        ...,
        info_gating_strict=_verify_env_fn_flags(env_fn),       # §7
        set_context_subjective_oracle_leak=False,              # asserted to be False
        belief_c_mae=_belief_mae_if_hyper(runner, cfg, env_fn),
        belief_c_calibration=_belief_calib_if_hyper(runner, cfg, env_fn),
    )


def _evaluate_external(
    runner,                          # ExternalBaselineRunner
    env_fn,
    cfg,
) -> EvalReport:
    # Delegate to the runner's own .evaluate (pkg-07 spec 04 §10); it returns
    # an EvalReport directly. The unified evaluator does NOT rewrite the schema.
    return runner.evaluate(
        env_fn=env_fn,
        c_grid=tuple(cfg.eval.zero_shot_test_c),
        episodes=int(cfg.eval.evaluate_episodes),
    )


def evaluate(runner, env_fn, cfg) -> EvalReport:
    if hasattr(runner, "evaluate") and not hasattr(runner, "set_context_subjective"):
        # External runner protocol path (pkg-07 spec 04 §10).
        return _evaluate_external(runner, env_fn, cfg)
    # Internal (hyper + 5 baseline + curriculum overrides) path.
    return _evaluate_internal_hyper_or_baseline(runner, env_fn, cfg)
```

### 4.3 SEM extraction — why `run_eval` needs a sidecar but no edit

`run_eval` currently returns only **mean** per c (no SEM). To compute SEM without modifying `run_eval`, the unified evaluator calls `Worker.collect_episodes` indirectly via `run_eval`, then re-reads the per-episode returns from the captured `Worker` instance through a **read-only side accessor**: a helper `_capture_worker_episode_returns(model, cfg, c) -> np.ndarray` that re-runs the inner `Worker.collect_episodes` call with the same CRN seeds and harvests `outs[i].returns` directly. This is a **separate code path** (in `eval/unified_evaluator.py`) that does not edit `run_eval` — it merely re-runs the same deterministic eval. Because every seed (env, planner, tie-break) is constant across evaluations, the re-run is bit-identical to what `run_eval` produced internally, and SEM = std(returns) / sqrt(n) is computed on the harvested array. The cost is one extra eval per c (negligible vs the existing main-table budget); the benefit is zero edits to the locked `run_eval`.

### 4.4 Lock 1 self-test: `test_run_eval_not_modified.py`

Located at `tests/eval/test_run_eval_not_modified.py`. Three independent integrity checks:

```python
def test_run_eval_source_unchanged_vs_baseline_snapshot():
    """Snapshot test: bytes of training/evaluation.py:run_eval must match the
    pinned baseline at tests/eval/fixtures/run_eval_baseline.py. Drift triggers
    a CI failure; intentional updates require a commit that also updates the
    fixture, leaving a paper-trail diff."""
    import inspect
    from hyper_mve.training.evaluation import run_eval
    from tests.eval.fixtures.run_eval_baseline import RUN_EVAL_BASELINE_SOURCE
    actual = inspect.getsource(run_eval)
    assert actual == RUN_EVAL_BASELINE_SOURCE, (
        "training/evaluation.py:run_eval has been modified. pkg-08 spec 01 Lock 1 "
        "REQUIRES wrapping, not replacing. If this change is intentional, update "
        "tests/eval/fixtures/run_eval_baseline.py in the same commit (and tag the "
        "PR with [pkg-08 spec 01 Lock 1 update])."
    )


def test_run_eval_signature_unchanged():
    """Defensive: even if someone updates the fixture, the signature
    (model, cfg, global_step=0) must remain. Other args added => integration risk."""
    import inspect
    from hyper_mve.training.evaluation import run_eval
    sig = inspect.signature(run_eval)
    assert list(sig.parameters) == ["model", "cfg", "global_step"]
    assert sig.parameters["global_step"].default == 0


def test_run_eval_tb_tag_keys_subset_of_legacy():
    """run_eval must still emit the legacy TB tag family; pkg-08 spec 01 §4.2
    consumes 'planner/return_total_c{c:g}' / 'prior/return_total_c{c:g}' /
    'planner_prior_gap' by name."""
    import torch
    from hyper_mve.configs import V4Config
    cfg = V4Config()                          # default preset
    model = _make_dummy_hyper_model(cfg)      # fixture
    result = run_eval(model, cfg, global_step=0)
    expected_prefixes = {"prior/return_total", "planner/return_total", "planner_prior_gap"}
    actual_keys = set(result)
    assert any(k.startswith("prior/return_total") for k in actual_keys)
    assert any(k.startswith("planner/return_total") for k in actual_keys)
    assert "planner_prior_gap" in actual_keys
```

The fixture file `tests/eval/fixtures/run_eval_baseline.py` is generated by a one-time helper script `tests/eval/fixtures/regenerate.py` (manually invoked; not in CI). The CI surface is just the three asserts above.

---

## 5. Internal runner eval flow (`BaselineModel` + `HyperMuZeroModel` curriculum overrides)

### 5.1 Dispatch boundary

The unified evaluator routes `runner` to the internal path iff `hasattr(runner, "set_context_subjective")` — this is the load-bearing API on `HyperMuZeroModel` (pkg-04 spec 02) and is what differentiates the in-process eval path from the external runner's self-contained eval. 5 internal variants (`input_wide`, `input_deep`, `ma_muzero`, `no_belief`, `rewardhead_explicit_type`) inherit `BaselineModel.set_context_subjective` from pkg-06's `BaselineModel` abstract base (or its pkg-07 spec 03 implementation); 3 curriculum-override variants (`hyper`, `oracle_only`, `infer_only`) use `HyperMuZeroModel` directly.

### 5.2 In-process state reuse

The internal eval path does **not** reload checkpoints — it consumes the live `model` object as passed in by sweep harness (`HyperMuZeroModel(cfg)` or `BaselineModel(cfg)`, post-training). This avoids the ckpt → load round trip's roughly 200 ms overhead per c-value × 7 c-values = 1.4 s per eval, multiplied by spec 03 four modes = 5.6 s, multiplied by spec 05 row count.

### 5.3 Belief-head diagnostics population

For the `hyper` variant only (detected via `isinstance(runner, HyperMuZeroModel)`), the unified evaluator populates `belief_c_mae` and `belief_c_calibration`. The computation reads the belief head's ĉ output from the model's forward pass during the eval rollout (planner mode), and compares against `c_true` (sourced from `info["c_true"]` — but **only** under the unified evaluator's eval-time path, where `eval_info_mode=True` exposes `c_true` for metric collection but never feeds it back into the policy / planner). For every other variant the two fields are populated with `None` (Python literal); JSON serialisation writes them as `null` and downstream filters skip them.

### 5.4 Planner-prior gap population

Sourced verbatim from the legacy `run_eval` output `planner_prior_gap`. This scalar is the headline distillation residual the paper has been tracking since v4-opt 2026-06. The unified evaluator adds two sibling scalars `direct_inference_return_mean` (= mean over c of `prior/return_total_c*`) and `planner_full_return_mean` (= mean over c of `planner/return_total_c*`) so that the 4-mode comparison plot (spec 07 §4) can plot all four modes' means on the same axis.

### 5.5 Self-Info strict enforcement (C8-EVAL-SELF1)

The internal eval path's call to `set_context_subjective(agent_id, cap_i, belief)` is monitored by a wrapper that intercepts any oracle-type leak. Mechanism (sketch):

```python
def _wrap_set_context_strict(model):
    """Wraps model.set_context_subjective with a guard that asserts no oracle types
    cross the boundary. Returns a context manager that records leak attempts."""
    original = model.set_context_subjective
    leak_state = {"leaked": False}

    def guarded(agent_id, cap_i, belief, *args, **kwargs):
        # cap_i is public (per-agent capability vector); belief is the inferred
        # context posterior. Neither may be the env's c_true scalar.
        if isinstance(belief, dict) and "c_true" in belief:
            leak_state["leaked"] = True
        return original(agent_id, cap_i, belief, *args, **kwargs)

    model.set_context_subjective = guarded
    return leak_state
```

The `EvalReport.set_context_subjective_oracle_leak` field is populated from `leak_state["leaked"]` and is asserted to be `False` in spec 01's test `test_self_info_strict_at_eval.py` (§7).

---

## 6. External runner eval flow (`ExternalBaselineRunner`)

### 6.1 Delegation contract

The unified evaluator's external branch (`_evaluate_external` in §4.2 pseudocode) calls `runner.evaluate(env_fn, c_grid, episodes) -> EvalReport` directly (pkg-07 spec 04 §10 + spec 08 signature). The runner is expected to construct its own `EvalReport` with the same schema. The unified evaluator **does not rewrite** the returned `EvalReport` — it returns it as-is.

### 6.2 Populated vs. nullable fields under external runners

External runners do **not** have an MVE planner (they have their own policy + critic). Therefore:

- `planner_prior_return_gap`: set to `0.0` (no planner; no distillation residual). Caller (spec 07 stats) treats `0.0` here as "not applicable" for the variant.
- `direct_inference_return_mean`: set equal to `return_mean` (no planner; argmax of policy is the only mode).
- `planner_full_return_mean`: set equal to `return_mean` (same).
- `belief_c_mae` / `belief_c_calibration`: `None` (no belief head on external runners; even `external_mamba` which has a belief over latent z does not align with the ĉ semantic).
- `info_gating_strict`: per the **canonical definition** in §3.1 = `env_fn()._oracle_mode is False` (eval_info_mode is allowed to be True; only oracle_mode flip is forbidden); verified by `test_external_eval_contract.py` (§7).
- `set_context_subjective_oracle_leak`: `False` (external runners do not call `set_context_subjective`; field is structurally not applicable but encoded as `False` for schema uniformity).

### 6.3 Per-c / segment / bell-curve aggregation responsibility

For external runners, the `runner.evaluate()` implementation is responsible for producing `return_per_c` / `return_per_segment` / `return_per_type_ratio`. Pkg-07 spec 04 §10 locks the c-grid aggregation contract on the runner side; pkg-07 spec 04 itself does not lock segment/bell-curve aggregation (because those depend on `cfg.eval.c_segments` / `cfg.eval.bell_curve_type_ratios`, which are pkg-08 concerns). The unified evaluator's external branch therefore performs a **post-hoc aggregation pass** on the returned `EvalReport.return_per_c` to compute `return_per_segment` and `return_per_type_ratio` if the runner left them as empty `MappingProxyType({})` (see §8 below). This keeps pkg-07 spec 04 free of pkg-08-specific aggregation logic.

---

## 7. Self-Info strict at eval time (C8-EVAL-SELF1)

### 7.1 The four-layer guard

The unified evaluator enforces "no oracle leak at eval time" through four layers:

1. **env_fn flag check** (`_verify_env_fn_flags`): constructs a single `env = env_fn()` and evaluates the **canonical expression** `env._oracle_mode is False` (verbatim from `EvalReport.info_gating_strict` definition in §3.1). The `eval_info_mode` flag is allowed to be `True` (this is the unique caller permitted to flip it; pkg-07 spec 04 §10). The boolean result populates `EvalReport.info_gating_strict`.
2. **`set_context_subjective` wrap** (§5.5): wraps the model's API to intercept any belief dict that carries `"c_true"`. Populates `EvalReport.set_context_subjective_oracle_leak`.
3. **`cfg.eval.use_oracle_types` default**: at spec 02 + design D10 + Pkg-01 spec 05 consumption, `cfg.eval.use_oracle_types: bool = False` is the locked default. The unified evaluator asserts this is `False` at entry (raises `RuntimeError` if `True`); the only caller permitted to set `True` is the curriculum-override `oracle_only` variant (which constructs `HyperMuZeroModel` directly with curriculum stage 1 end = 1.0, and the oracle types are consumed by the curriculum, not by the evaluator).
4. **Info-dict introspection**: at every `step()` call inside the rollout, the eval-time wrapper asserts that `info["types"]` is absent (because `oracle_mode=False`); this is a defensive check that the adapter's `_filter_info` is doing its job. Failure raises `AssertionError` (loud and immediate, not silent).

### 7.2 Named tests

`test_self_info_strict_at_eval.py` (in §7's list) parametrises every variant in `REGISTRY ∪ {"hyper", "oracle_only", "infer_only"}` (14 strings) and asserts:
- `EvalReport.set_context_subjective_oracle_leak is False` for every variant.
- `EvalReport.info_gating_strict is True` for every variant.
- For `variant == "oracle_only"`, the curriculum override is verified separately (the oracle types are consumed by the trainer's curriculum, not by the evaluator's rollout; this test asserts the evaluator did NOT consume them).

---

## 8. c-grid + c-segment + bell-curve aggregation logic

### 8.1 c-grid aggregation (the 7-value zero-shot grid)

The unified evaluator's `return_per_c` field is populated over `cfg.eval.zero_shot_test_c` (default `(0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)` — 7 values). For each c:
- Episodes count = `cfg.eval.evaluate_episodes` (default 30, the reserved field at eval_config.py:19).
- The unified evaluator runs `Worker.collect_episodes` indirectly through `run_eval` (for `direct_inference` + `planner_full`) and through the spec 03 §3 dispatch path (for `planner_no_crn` + `planner_no_coord_desc`).
- Mean and SEM per c are computed from the harvested raw returns (per §4.3).

### 8.2 c-segment three-way aggregation (Ch6.2.4)

For each segment `(low, high)` in `cfg.eval.c_segments` (default `((0.0, 0.3), (0.3, 0.7), (0.7, 1.0))`):

```python
def _aggregate_segments(
    return_per_c: dict[float, float],
    episodes_per_c: dict[float, int],
    segments: tuple[tuple[float, float], ...],
) -> tuple[dict[tuple[float, float], float], dict[tuple[float, float], float]]:
    """For each segment, compute episode-count-weighted mean and SEM.
    Episode-count-weighted to handle uneven episode budgets per c (rare but
    possible if a c-value is added partway through a sweep)."""
    mean_per_seg = {}
    sem_per_seg = {}
    for low, high in segments:
        c_in_seg = [c for c in return_per_c if low <= c <= high]
        if not c_in_seg:
            mean_per_seg[(low, high)] = float("nan")
            sem_per_seg[(low, high)] = float("nan")
            continue
        weights = np.array([episodes_per_c[c] for c in c_in_seg])
        means = np.array([return_per_c[c] for c in c_in_seg])
        mean_per_seg[(low, high)] = float(np.average(means, weights=weights))
        # Segment SEM: pooled std / sqrt(sum of episode counts).
        sem_per_seg[(low, high)] = float(
            np.sqrt(np.average((means - mean_per_seg[(low, high)]) ** 2, weights=weights))
            / np.sqrt(weights.sum())
        )
    return mean_per_seg, sem_per_seg
```

Boundary semantic (closed-closed `[low, high]`): the value c = 0.3 is in both `(0.0, 0.3)` and `(0.3, 0.7)`. By convention the unified evaluator uses **half-open intervals on the lower bound** for segments after the first one (i.e., `[0.0, 0.3]`, `(0.3, 0.7]`, `(0.7, 1.0]`) — the implementation matches Ch6.2.4 prose.

### 8.3 Bell-curve type-ratio sweep (Ch6.6)

For each ratio `(n_alpha, n_beta)` in `cfg.eval.bell_curve_type_ratios` (default 6 ratios):

```python
def _aggregate_type_ratios(
    runner,                                  # BaselineLike
    env_fn,
    cfg,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    """For each (n_alpha, n_beta), construct a type_assignment vector and re-run
    eval with that override. Uses options['types'] = tuple([ALPHA]*n_alpha +
    [BETA]*n_beta) to thread through ResourceCommonsEnv.reset's options arg
    (env.py:128-149, pkg-02 spec)."""
    means = {}
    sems = {}
    for n_alpha, n_beta in cfg.eval.bell_curve_type_ratios:
        types_override = tuple([AgentType.ALPHA] * n_alpha + [AgentType.BETA] * n_beta)
        # Run a sub-eval with options['types'] = types_override at every reset.
        # Implementation: the env_fn is wrapped to inject options['types'] into reset.
        returns = _eval_one_type_ratio(runner, env_fn, cfg, types_override)
        means[(n_alpha, n_beta)] = float(returns.mean())
        sems[(n_alpha, n_beta)] = float(returns.std() / np.sqrt(len(returns)))
    return means, sems
```

The bell-curve sweep budget is `len(cfg.eval.bell_curve_type_ratios)` × `cfg.eval.evaluate_episodes` = 6 × 30 = 180 extra episodes per evaluation; this is the largest single cost component and is documented in pkg-08 README's 850 GPU-hr estimate.

### 8.4 External runner post-hoc aggregation

If `_evaluate_external` returns an `EvalReport` with `return_per_segment` or `return_per_type_ratio` as empty `MappingProxyType({})`, the unified evaluator runs `_aggregate_segments` and `_aggregate_type_ratios` post-hoc on the returned `return_per_c` and produces a new `EvalReport` with these fields filled in (using `dataclasses.replace`, which respects `frozen=True`). This keeps pkg-07 spec 04 free of pkg-08-specific aggregation logic.

---

## 9. Bell-curve type-ratio sweep — design note

The default ratios in `cfg.eval.bell_curve_type_ratios` are `((0,4), (1,3), (2,2), (3,1), (4,0), (1,3))` — note `(1,3)` appears twice (a placeholder for a future asymmetric ratio, see eval_config.py:44). The unified evaluator deduplicates by ratio at aggregation time (the second `(1,3)` produces a second sub-eval whose results are averaged with the first; the resulting mean is stored under the same key, so `Mapping[(1,3), float]` carries the average of both sub-evals). A test `test_bell_curve_dedup_handles_repeated_ratios.py` (§10) asserts this behaviour.

The c-grid for the bell-curve sweep is fixed at `c = 0.5` (the middle of the spectrum) to keep the budget tractable. This is documented in Ch6.6 and is locked in spec 02 §4.

---

## 10. Test contract — C8-EVAL-* enforcement

Located under `tests/eval/`. Each named below corresponds to a specific C8-EVAL constraint (README).

### 10.1 `test_run_eval_not_modified.py` — C8-EVAL-REUSE1

Locks 1.1 + 1.2 + 1.3 above (3 asserts in §4.4). Triggered every CI run.

### 10.2 `test_unified_evaluator_schema.py` — C8-EVAL-SCHEMA1

Round-trip test: constructs a dummy `HyperMuZeroModel(cfg)`, calls `evaluate(model, env_fn, cfg)`, asserts the returned `EvalReport` has every one of the 32 fields populated with the correct type (test name kept generic with the literal count to make off-by-one regressions visible at CI time):

```python
def test_unified_evaluator_returns_evalreport_with_all_32_fields():
    from hyper_mve.eval.unified_evaluator import evaluate
    from hyper_mve.eval.eval_report import EvalReport
    cfg = V4Config()
    model = _make_dummy_hyper_model(cfg)
    env_fn = lambda: ResourceCommonsPettingZooEnv(cfg.env, oracle_mode=False, eval_info_mode=True)
    report = evaluate(model, env_fn, cfg)
    assert isinstance(report, EvalReport)
    # 5 identity fields
    assert isinstance(report.variant, str)
    assert isinstance(report.seed, int)
    assert isinstance(report.config_hash, str) and len(report.config_hash) == 40   # SHA1
    assert report.eval_mode in ("prior", "planner")
    assert report.eval_planner_mode in (
        "direct_inference", "planner_no_crn", "planner_no_coord_desc", "planner_full",
    )
    assert isinstance(report.c_visible, bool)
    # 5 headline scalars
    for f in ("return_mean", "return_sem", "return_zero_shot_seen",
              "return_zero_shot_unseen", "return_zero_shot_gap"):
        assert isinstance(getattr(report, f), float)
    # 3 per-c mappings: keys = float, values = float/int, len matches zero_shot_test_c
    assert set(report.return_per_c) == set(cfg.eval.zero_shot_test_c)
    assert set(report.return_per_c_sem) == set(cfg.eval.zero_shot_test_c)
    assert set(report.episodes_per_c) == set(cfg.eval.zero_shot_test_c)
    # 2 c-segment mappings: keys = tuple[float, float], len matches c_segments
    assert set(report.return_per_segment) == set(cfg.eval.c_segments)
    assert set(report.return_per_segment_sem) == set(cfg.eval.c_segments)
    # 2 bell-curve mappings: keys = tuple[int, int], len matches bell_curve_type_ratios (after dedup)
    expected_ratios = set(tuple(r) for r in cfg.eval.bell_curve_type_ratios)
    assert set(report.return_per_type_ratio) == expected_ratios
    assert set(report.return_per_type_ratio_sem) == expected_ratios
    # 4 regret fields
    assert set(report.regret_per_c) == set(cfg.eval.zero_shot_test_c)
    assert isinstance(report.regret_mean, float)
    assert set(report.oracle_ceiling_per_c) == set(cfg.eval.zero_shot_test_c)
    assert set(report.oracle_ceiling_cache_hit) == set(cfg.eval.zero_shot_test_c)
    # 3 planner-prior gap fields
    for f in ("planner_prior_return_gap", "direct_inference_return_mean", "planner_full_return_mean"):
        assert isinstance(getattr(report, f), float)
    # 5 diagnostics fields
    assert isinstance(report.walltime_seconds, float)
    assert isinstance(report.env_steps_evaluated, int)
    assert isinstance(report.episodes_total, int)
    assert report.info_gating_strict is True
    assert report.set_context_subjective_oracle_leak is False
    # 2 nullable belief fields
    assert report.belief_c_mae is None or isinstance(report.belief_c_mae, float)
    assert report.belief_c_calibration is None or isinstance(report.belief_c_calibration, float)
    # 1 schema version sentinel
    assert report.schema_version == "pkg08-spec01-v1"
    # Frozen check
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.return_mean = 999.0
```

### 10.3 `test_external_eval_contract.py` — C8-EVAL-SCHEMA1 (external branch)

Asserts that both internal and external branches return the same schema:

```python
@pytest.mark.parametrize("variant", [
    "input_wide",         # internal
    "external_mappo",     # external Tier-1
])
def test_internal_and_external_return_same_schema(variant):
    cfg = V4Config()
    runner = create_baseline(cfg, variant)
    env_fn = lambda: ResourceCommonsPettingZooEnv(cfg.env, oracle_mode=False, eval_info_mode=True)
    report = evaluate(runner, env_fn, cfg)
    # Same dataclass type, same fields
    assert isinstance(report, EvalReport)
    assert dataclasses.fields(report) == dataclasses.fields(EvalReport(...))
    if variant.startswith("external_"):
        # External runners populate planner-prior gap with 0.0 (§6.2)
        assert report.planner_prior_return_gap == 0.0
        assert report.belief_c_mae is None
```

### 10.4 `test_self_info_strict_at_eval.py` — C8-EVAL-SELF1

Per §7.2 above. Parametrises 14 variants.

### 10.5 `test_c_segment_aggregation.py` — C8-EVAL-SEG1

Constructs a fake `return_per_c` with known values (e.g., `{0.0: 10, 0.2: 12, 0.35: 14, 0.5: 16, 0.65: 18, 0.8: 20, 1.0: 22}`) and asserts the segment means match the documented half-open-interval logic (§8.2):

```python
def test_c_segment_aggregation_correctness():
    from hyper_mve.eval.unified_evaluator import _aggregate_segments
    return_per_c = {0.0: 10.0, 0.2: 12.0, 0.35: 14.0, 0.5: 16.0, 0.65: 18.0, 0.8: 20.0, 1.0: 22.0}
    episodes_per_c = {c: 30 for c in return_per_c}
    segments = ((0.0, 0.3), (0.3, 0.7), (0.7, 1.0))
    mean_per_seg, _ = _aggregate_segments(return_per_c, episodes_per_c, segments)
    # [0.0, 0.3]: c in {0.0, 0.2}, mean = 11.0
    assert mean_per_seg[(0.0, 0.3)] == pytest.approx(11.0)
    # (0.3, 0.7]: c in {0.35, 0.5, 0.65}, mean = 16.0
    assert mean_per_seg[(0.3, 0.7)] == pytest.approx(16.0)
    # (0.7, 1.0]: c in {0.8, 1.0}, mean = 21.0
    assert mean_per_seg[(0.7, 1.0)] == pytest.approx(21.0)
```

### 10.6 `test_bell_curve_in_report.py` — C8-EVAL-BELL1

Asserts the bell-curve mapping is populated with the expected tuple keys.

### 10.7 `test_bell_curve_dedup_handles_repeated_ratios.py` — Edge case (§9)

Constructs `cfg.eval.bell_curve_type_ratios = ((1,3), (1,3), (2,2))` and asserts the report has exactly 2 keys in `return_per_type_ratio`.

### 10.8 `test_eval_report_schema_lock_matches_spec_08.py` — Lock 2

Compares the `@dataclass` field list in `hyper_mve/eval/eval_report.py` against the verbatim field list in `sdd/pkg-08-eval-and-ablation/specs/08-integration-contracts.md` §3 (parsed from markdown). Drift fails immediately.

---

## 11. Integration hooks (downstream specs consume)

| Consumer | Consumed | Use |
|----------|----------|-----|
| **spec 02** zero-shot + c_hidden + regret | `EvalReport.return_zero_shot_seen` / `return_zero_shot_unseen` / `return_zero_shot_gap` / `return_per_c` (for `cfg.eval.zero_shot_test_c`) / `regret_per_c` / `oracle_ceiling_per_c` / `oracle_ceiling_cache_hit` / `belief_c_mae` / `belief_c_calibration` (c_hidden authenticity) / `c_visible` flag | spec 02 extends the report by populating `regret_*` via the ceiling cache lookup and by gating belief diagnostics on `c_visible=False` runs (Theory Audit Q7). |
| **spec 03** four planner eval mode | `EvalReport.eval_planner_mode` (Literal 4 values) / `direct_inference_return_mean` / `planner_full_return_mean` / `planner_prior_return_gap` | spec 03 owns the dispatch table that fills `eval_planner_mode` and the two extra mode means; spec 01 just declares the schema slot. |
| **spec 04** μP self-check | `EvalReport.return_mean` per (width, lr, seed) row | spec 04 sweeps 18 configurations and uses the headline mean for the μP doubling check; no new fields. |
| **spec 05** sweep harness + RunRegistry | `EvalReport.config_hash` / `variant` / `seed` (for row identity) / `return_mean` / `return_zero_shot_unseen` / `regret_mean` (for summary metrics column on `runs/registry.jsonl`) | spec 05 §4 owns the JSONL row schema; this spec only guarantees the `EvalReport` is JSON-serialisable. |
| **spec 07** statistics + comparison | per-episode raw returns from `runs/<run_tag>/eval_episodes.csv` (sidecar; not in `EvalReport`) + `EvalReport.return_mean` / `return_sem` / `episodes_per_c` (for Welch t input) | spec 07 §3 Welch t consumes `(mean, sem, n)` triples per group; pkg-08 spec 01 guarantees these three are present. |
| **pkg-07 spec 01** §2 `BaselineLike Union` | Reverse-consumed: pkg-07 spec 01's factory returns `BaselineLike`, which pkg-08 spec 01 §2.2 consumes as the `runner` arg. | pkg-07 → pkg-08 contract anchor. |
| **pkg-07 spec 04** `ResourceCommonsPettingZooEnv` | Reverse-consumed: pkg-08 spec 01 §2.2 `env_fn` returns this type. The two-flag info gate (`oracle_mode=False`, `eval_info_mode=True` for eval) is verified by `_verify_env_fn_flags` (§7.1). | pkg-07 → pkg-08 contract anchor. |
| **pkg-07 spec 08** `evaluate(env_fn, c_grid, episodes) -> EvalReport` | Reverse-consumed: pkg-08 spec 01 §3 (this section) defines the `EvalReport` schema that pkg-07 spec 08's signature locks. Drift on either side triggers `test_eval_report_schema_lock_matches_spec_08.py` (§10.8). | Two-way schema lock. |

---

## 12. Edge cases

| Case | Behaviour |
|------|-----------|
| `runner` is `BaselineModel` but missing `set_context_subjective` (mis-implemented internal variant) | `AttributeError` at the dispatch boundary (§5.1); not silently treated as external. |
| `env_fn()` returns adapter with `oracle_mode=True` | `RuntimeError("eval-time env_fn must have oracle_mode=False; got True. Pkg-07 spec 04 §7 violated.")` raised at entry. |
| `env_fn()` returns adapter with `eval_info_mode=False` | Allowed (the eval-time evaluator is the unique caller permitted to flip it, but not required to); `belief_c_mae` and `belief_c_calibration` are populated as `None` (since `c_true` is not accessible). |
| `cfg.eval.zero_shot_test_c` is empty | `RuntimeError("zero_shot_test_c must be non-empty")` raised at entry. |
| `cfg.eval.zero_shot_test_c` contains NaN | Filtered out with a `warnings.warn(...)`; eval proceeds on the remaining values. (Defensive — should never happen, but `Mapping[float, float]` rejects NaN keys at insertion.) |
| `cfg.eval.bell_curve_type_ratios` is empty | `return_per_type_ratio` is empty `MappingProxyType({})`; eval proceeds without bell-curve sweep. |
| External runner returns `EvalReport` with `eval_planner_mode` set to a value other than `"planner_full"` | Allowed (the runner is the authority on its own mode); spec 03 §4 documents this. |
| `runner` is a stub (e.g. `MARIEStub` / `GAStub`) | `NotImplementedError` raised by the stub's `__init__` before `evaluate()` is ever called (pkg-07 spec 01 §3.3); spec 05 sweep harness catches and writes `status="skipped"` row to `registry.jsonl`. |
| `EvalReport.regret_per_c` cache miss | Spec 02 §3 handles — triggers `oracle_only` re-run and writes new ceiling JSON; this spec's schema slot is unchanged. |
| Belief head present on non-`hyper` variant (e.g., a future `no_belief_v2` that adds back a belief head) | Variant-name-based detection (`isinstance(runner, HyperMuZeroModel)`) still gates population to `hyper` only; the new variant must propose a spec edit to extend the gate. |

---

## 13. Cross-references

### Upstream anchors

- **pkg-08 design.md §3.2** — `@dataclass(frozen=True) EvalReport` field enumeration (this spec §3 inherits verbatim with one additional `schema_version` sentinel).
- **pkg-08 design.md §3.3** — four planner eval mode truth table (this spec §3 `eval_planner_mode` literal + §11 spec 03 hook).
- **pkg-08 design.md §4 D2** — `EvalReport` schema lock; internal + external double-consumption.
- **pkg-08 design.md §4 D3** — "wrap, not replace" `run_eval`; spec 01 single test `test_run_eval_not_modified.py`.
- **pkg-08 design.md §4 D7** — four planner eval mode decoupling from Abl4 training-time CRN × CoordDesc cells.
- **pkg-08 design.md §4 D10** — 5 new cfg fields, of which `cfg.eval.eval_planner_mode: Literal[...]` is consumed in §3 and `cfg.eval.eval_use_planner_direct_inference: bool` is consumed as the "mode short-circuit" detected at the §2.2 entry.
- **pkg-08 README C8-EVAL-***: REUSE1 / SCHEMA1 / SELF1 / SEG1 / BELL1 / MODE1 — each surfaced as a named test in §10.
- **pkg-07 spec 01 §2 / §3.2** — `REGISTRY` 11 keys + `BaselineLike Union` consumed in §2.2; `create_baseline(cfg, variant)` factory consumed implicitly through `runner`.
- **pkg-07 spec 04 §6 / §7 / §10** — `ResourceCommonsPettingZooEnv` API + two-flag info gate (`oracle_mode=False`, `eval_info_mode=True` for eval) + `env_fn` contract; consumed in §2.2 and §7.1.
- **pkg-07 spec 08** (drafting) — `BaselineLike.evaluate(env_fn, c_grid, episodes) -> EvalReport` signature mirror; drift caught by `test_eval_report_schema_lock_matches_spec_08.py`.

### Downstream consumption (§11 table)

- spec 02 (zero-shot + c_hidden + regret) extends `regret_*` / `belief_*` fields.
- spec 03 (four planner mode) extends `eval_planner_mode` + `direct_inference_return_mean` + `planner_full_return_mean`.
- spec 04 (μP) consumes `return_mean` only.
- spec 05 (sweep harness + RunRegistry) consumes summary metrics columns.
- spec 07 (stats + compare) consumes `(mean, sem, n)` triples + sidecar `eval_episodes.csv`.
- spec 08 (integration contracts) is the schema lock partner — `EvalReport` schema must match spec 08 §3 byte-identically.

### Existing repo ground-truth anchors

- `hyper_mve/training/evaluation.py:run_eval(model, cfg, global_step=0) -> dict[str, float]` — Lock 1 anchor; modified-detection via `tests/eval/fixtures/run_eval_baseline.py` snapshot.
- `hyper_mve/configs/eval_config.py` — pre-existing reserved fields (`eval_c_grid` / `c_segments` / `zero_shot_train_c` / `zero_shot_test_c` / `zero_shot_unseen_c` / `bell_curve_type_ratios`) consumed in §2.2 and §8.
- `hyper_mve/configs/v4_config.py` — `V4Config` top-level dataclass; pkg-08 spec 01 reads only `cfg.eval`, `cfg.env`, `cfg.baselines`.
- `hyper_mve/baselines/__init__.py` (pkg-07 spec 01 §3.2) — `BaselineLike` type alias source.
- `hyper_mve/envs/adapters/pettingzoo_wrapper.py` (pkg-07 spec 04 §2) — `ResourceCommonsPettingZooEnv` source.

---

## Anchors (verbatim grep targets for spec 08 §6 drift detector)

- `pkg-08 spec 01 Lock 1: training/evaluation.py:run_eval is NOT replaced`
- `pkg-08 spec 01 Lock 2: EvalReport is the unique evaluator output schema`
- `pkg-08 spec 01 Lock 3: Self-Info strict at eval time (C8-EVAL-SELF1)`
- `pkg-08 spec 01 §3: EvalReport @dataclass(frozen=True) — 32 fields, schema_version="pkg08-spec01-v1"`
- `pkg-08 spec 01 §4.4: test_run_eval_not_modified.py — 3 asserts`
- `pkg-08 spec 01 §10.8: test_eval_report_schema_lock_matches_spec_08.py`
- `pkg-08 spec 01 §11: integration hooks table — 7 downstream consumers`
