# Spec 03 — Direct-Inference Toggle + Four Planner Eval Modes

> **Parent docs**: [`../proposal.md`](../proposal.md) · [`../design.md`](../design.md) §3.3 / §4 D7 / §4 D10 · [`../README.md`](../README.md) C8-EVAL-MODE1 + §"5 新 cfg 字段穷举" rows 4 + 5
> **Upstream consumed**: [`./01-unified-evaluator.md`](./01-unified-evaluator.md) §3 (`@dataclass(frozen=True) EvalReport` 32-field schema — slots `eval_planner_mode`, `direct_inference_return_mean`, `planner_full_return_mean`, `planner_prior_return_gap`) + §4 (dispatch flow inside `evaluate(runner, env_fn, cfg)`) · [`./02-zero-shot-and-c-hidden.md`](./02-zero-shot-and-c-hidden.md) §7 ("Orthogonal — apply per-c independently of zero-shot grid composition. The four modes … each populate `return_per_c` over the full 7-value test grid")
> **Cross-refs**: [`./06-ablation-cli-and-cells.md`](./06-ablation-cli-and-cells.md) (Ablation 4 training-time CRN × CoordDesc cell; orthogonal axis) · [`./05-sweep-harness-and-run-registry.md`](./05-sweep-harness-and-run-registry.md) §"paired-mode protocol" (§6 below) · [`./07-statistics-and-comparison.md`](./07-statistics-and-comparison.md) (post-hoc `planner_prior_return_gap` stats join) · [`./08-integration-contracts.md`](./08-integration-contracts.md) §5 (canonical cfg-field declarations + Pkg-05 downstream patches) · [`../../pkg-07-baselines/specs/04-external-runner-and-adapter.md`](../../pkg-07-baselines/specs/04-external-runner-and-adapter.md) (`ResourceCommonsPettingZooEnv` consumed identically by all 4 modes via `env_fn`)
> **Status**: SDD only — describes the contract for `hyper_mve/eval/planner_modes.py` and the dispatch surface added to `hyper_mve/eval/unified_evaluator.py::evaluate`. No production code changes here; the only file-level surface this spec touches is `hyper_mve/configs/eval_config.py` (+2 fields, declared in §3 + §8.1 and codified in spec 08 §5).

---

## Hard locks (3, before §1)

### Lock 1 — Eval-time vs training-time axes are orthogonal and never collapsed

The four modes this spec defines (`direct_inference`, `planner_no_crn`, `planner_no_coord_desc`, `planner_full`) are **evaluation-time** dispatch modes. They govern how the `unified_evaluator.evaluate(runner, env_fn, cfg)` function constructs the per-c rollout when computing `EvalReport.return_per_c`. They have **nothing to do** with how the underlying model was trained — they operate on a model the trainer has already produced, optionally re-routing the MVE planner's internal flags at evaluator entry.

The training-time CRN / coordinate-descent flags (`cfg.train.use_crn: bool` and `cfg.train.randomize_order: bool`, the latter renamed from `cfg.train.use_coord_desc` in spec 06 per Theory Audit Q2) are owned by pkg-08 spec 06 under the Ablation 4 (CRN × Joint/CoordDesc) training-time cell matrix. They govern the worker's rollout policy during episode collection. They are **not** read by `unified_evaluator.evaluate` and they are **not** synonyms of the eval-time fields declared in this spec. A model trained with `cfg.train.use_crn=False` can still be evaluated under `cfg.eval.eval_planner_mode="planner_full"` (with CRN enabled at eval) — this is in fact the recommended setup for Ablation 4 (training-time variation, identical fair evaluation).

The two axes form a 4 × 4 product space at the reporting level (4 training-time cells × 4 evaluation-time modes), but in practice the sweep harness (spec 05) emits exactly one row per (training-cell × seed) at the default `eval_planner_mode="planner_full"`, and the eval-mode axis is exercised only when the paired-mode protocol (§6 below) is enabled or when a dedicated eval-mode ablation row is requested. The orthogonality lock prevents future readers from collapsing the axes into a single 4-cell matrix.

### Lock 2 — `cfg.eval.eval_planner_mode` is the canonical single source of truth; the short-circuit alias is an inconsistency-detector, not a silent override

The canonical field is `cfg.eval.eval_planner_mode: Literal["direct_inference", "planner_no_crn", "planner_no_coord_desc", "planner_full"] = "planner_full"`. The unified evaluator (spec 01 §4.1) reads this field at entry and dispatches on its value.

A second field `cfg.eval.eval_use_planner_direct_inference: bool = False` exists as a back-compat short-circuit for legacy callers that toggle "use direct inference" as a boolean. **It is deprecated on arrival.** Its only legal use is to assert equivalence with the canonical field:

- If `eval_use_planner_direct_inference == True` AND `eval_planner_mode == "direct_inference"` → **consistent**, no error, dispatch proceeds to `direct_inference`.
- If `eval_use_planner_direct_inference == False` AND `eval_planner_mode != "direct_inference"` → **consistent**, no error, dispatch proceeds to whichever mode the canonical field selects.
- If `eval_use_planner_direct_inference == True` AND `eval_planner_mode != "direct_inference"` → **inconsistent**, the evaluator MUST raise `ValueError("eval_use_planner_direct_inference=True conflicts with eval_planner_mode={mode!r}; set eval_planner_mode='direct_inference' or eval_use_planner_direct_inference=False")` at entry, before any rollout.
- If `eval_use_planner_direct_inference == False` AND `eval_planner_mode == "direct_inference"` → **consistent** (the short-circuit is opt-in; absence does not contradict the canonical field), no error, dispatch proceeds to `direct_inference`.

The evaluator must **not** silently prefer one field over the other in the inconsistent case. This rules out the failure mode where a future caller adds the alias for ergonomic reasons and the canonical field is forgotten, producing a silent mode flip undetectable from the report.

### Lock 3 — All 4 modes populate the full `EvalReport` schema; the unified evaluator handles all 4 modes via dispatch; spec 03 does NOT introduce a parallel evaluator path

The `EvalReport` `@dataclass(frozen=True)` is locked at spec 01 §3.1 with 32 fields. All 4 modes produce reports with the same 32 fields, the same key sets on the per-c mappings (`return_per_c` keyed by `cfg.eval.zero_shot_test_c`), and the same `MappingProxyType` wrapping. The modes differ in exactly two places:

1. The value of `EvalReport.eval_planner_mode` slot (one of the 4 literal strings, set to the dispatched mode).
2. The numerical content of `return_per_c`, `return_mean`, `return_sem`, `return_per_segment`, `return_per_type_ratio`, and the zero-shot headline scalars (which all derive from the dispatched mode's per-episode returns).

**The diagnostic slots `direct_inference_return_mean` and `planner_full_return_mean` (spec 01 §3.1 slots 25 + 26) are populated in EVERY single-mode call**, sourced from the legacy `run_eval`'s `prior/*` and `planner/*` tag families (spec 01 §4.2 lines 273-274 + §5.4 verbatim contract): `direct_inference_return_mean = mean(legacy_results, prefix="prior/")`, `planner_full_return_mean = mean(legacy_results, prefix="planner/")`. The `planner_prior_return_gap` slot is sourced directly from `legacy_results["planner_prior_gap"]` (spec 01 §4.2 line 272), NOT computed as a sentinel inside spec 03. The full per-mode population semantic is locked at §5 below — the four-mode taxonomy lives in `eval_planner_mode`; the in-row gap continues to exist because `run_eval` runs BOTH the prior and the planner inner branches in a single call regardless of which mode dispatched.

The unified evaluator (spec 01) handles ALL 4 modes via dispatch on `cfg.eval.eval_planner_mode`. Spec 03 introduces **no parallel `evaluate()` function**, no alternative evaluator entry point, and no second copy of the c-grid loop. The only new file this spec introduces is `hyper_mve/eval/planner_modes.py`, which is a thin dispatch helper consumed by `unified_evaluator.evaluate` at its dispatch boundary (§4 below). The boundary is a single helper function `_dispatch_planner_mode(model, cfg, env_fn) -> _PlannerInvocation` returning the keyword args (`use_crn`, `randomize_order`, `skip_planner`) that the evaluator threads into the existing `Worker.collect_episodes` / `run_eval` machinery.

> **Disclosure on cfg.train transport.** The dispatch helper threads its eval-time overrides for `planner_no_crn` / `planner_no_coord_desc` modes by constructing a shallow-copy override of `cfg.train` via `dataclasses.replace`, because the existing `MVEPlanner.__init__` reads `self.use_crn = cfg.train.use_crn` and `self.use_coord_desc = cfg.train.use_coord_desc` constructor flags from `cfg.train` (the original SoT for training-time flags). This `dataclasses.replace` is a **transport mechanism only, not a semantic statement about training-time settings**: the override is consumed by the planner constructor inside one `run_eval` invocation and the original `cfg.train` is preserved everywhere else (the V4Config root object is unchanged). Future maintainers reading `mve_planner.py` should grep for the literal `cfg_eval_with_overrides = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, ...))` pattern in `unified_evaluator.evaluate` to see the eval-time override site. The cleaner long-term fix is to add `MVEPlanner.__init__(..., use_crn_override=None, randomize_order_override=None)` kwargs that bypass `cfg.train` entirely, but that is a spec-06-scope edit (it touches `mve_planner.py`); spec 03 deliberately stays at the transport-mechanism level to avoid scope creep.

---

## 1. Purpose

Two reviewer-facing motivations make this spec necessary, and a third operational motivation justifies the introduction of the two new cfg fields rather than relying on the existing `cfg.train.use_crn` / `cfg.train.use_coord_desc` axes.

**Motivation 1 — distillation residual decomposition.** Ch6 reviewers want to see how much of the hyper-MuZero variant's measured performance comes from the prediction head's policy prior (`π̂`, the raw policy distribution emitted by `PredictionNet`) versus the MVE planner's additional rollout-based refinement. The headline number the paper needs is:

```
planner_prior_return_gap = planner_full_return_mean - direct_inference_return_mean
```

This is the same scalar `run_eval` (in `hyper_mve/training/evaluation.py`) currently emits as `planner_prior_gap` for the in-training periodic eval. Spec 01 retains this scalar in `EvalReport.planner_prior_return_gap` (schema slot 27) verbatim. But spec 03 generalises the decomposition: instead of being a single scalar inside one eval call, the gap becomes a derivable statistic from any two paired reports — one with `eval_planner_mode="direct_inference"` and one with `eval_planner_mode="planner_full"` — joined by `(variant, seed, config_hash)` keys. The paired-mode protocol (§6 below) is what produces these row pairs; spec 07 (stats) is what joins them and computes Welch t-tests on the gap.

**Motivation 2 — decoupling Ablation 4 (training-time) from the eval-mode axis.** Ablation 4 (spec 06) trains 4 cells in a 2 × 2 grid: CRN-on/off × coord-desc-on/off. Under the legacy field naming (`cfg.train.use_crn` / `cfg.train.use_coord_desc`), if the evaluator reuses the training-time flags at eval time, the training-time CRN-off cell would also evaluate with CRN-off — which would entangle "training-time fragility" with "evaluation-time fragility" and falsely amplify the apparent loss from CRN ablation. Spec 03's dedicated `cfg.eval.eval_planner_mode` field ensures all four Ablation 4 cells are evaluated under a uniform, fair `planner_full` eval mode. Conversely, a `planner_full`-trained model can be evaluated under `eval_planner_mode="planner_no_crn"` to surface eval-time fragility on a strong-training-time baseline — this is a separate sweep cell that the eval-mode axis enables. Without the dedicated field, this decoupling is not expressible.

**Motivation 3 — short-circuit ergonomics for legacy callers.** Some existing in-training eval call sites (and a number of unit tests) toggle a single boolean "skip the planner, just use the prior" — the legacy `eval_mode="prior"` literal in `run_eval`. The `cfg.eval.eval_use_planner_direct_inference: bool` field exists as a back-compat door for these callers; they can set it to `True` without learning the four-mode literal taxonomy, and the unified evaluator will dispatch to `direct_inference`. Lock 2 specifies the consistency-check semantics that prevent this door from silently overriding the canonical field.

The deliverable of this spec is:

```
hyper_mve/eval/planner_modes.py    — thin dispatch helper consumed by unified_evaluator.evaluate
hyper_mve/configs/eval_config.py   — +2 fields (eval_planner_mode + eval_use_planner_direct_inference); declared in §3 + §8.1
```

No other files are touched by this spec. The unified evaluator's `evaluate(runner, env_fn, cfg)` function (owned by spec 01 §4) reads the new fields from `cfg.eval` and threads them into the dispatch helper; no spec-01 code surface is rewritten by spec 03.

---

## 2. The 4 modes — truth table

| Mode | `use_crn` (eval) | `randomize_order` (eval) ≡ `use_coord_desc` (eval) | Uses MVE rollouts | Direct-inference short-circuit |
|------|------------------|----------------------------------------------------|-------------------|--------------------------------|
| `direct_inference` | N/A | N/A | **NO** | **YES** (planner is skipped entirely) |
| `planner_no_crn` | **False** | True | YES | NO |
| `planner_no_coord_desc` | True | **False** | YES | NO (planner runs but coord descent disabled; single agent-order pass) |
| `planner_full` | True | True | YES | NO (default; all eval-time enhancements on) |

**Row 1 — `direct_inference`.** The unified evaluator skips the MVE planner entirely. The model's forward pass produces a raw policy distribution `π̂(s)` from the `PredictionNet` (for hyper variants) or the variant-specific policy head (for baselines), and the agent samples / argmaxes from this distribution under the standard deterministic-eval convention (epsilon = 0, argmax). This is byte-equivalent to the legacy `run_eval` `eval_mode="prior"` path. The flags `use_crn` and `randomize_order` (and their eval-time counterparts) are not consulted because no planner is invoked. The `EvalReport.direct_inference_return_mean` slot is populated; `EvalReport.planner_full_return_mean` is `float("nan")` (see §5 below for the population table).

**Row 2 — `planner_no_crn`.** The unified evaluator invokes the MVE planner exactly as in `planner_full`, but with the planner's internal `use_crn` flag forced to `False` at the dispatch boundary (§4.2). This means the planner does **not** apply Common Random Numbers at step 0 of its rollout — every candidate action is evaluated with independent randomness, increasing the per-candidate noise floor. The expected effect (from v4.6 design rationale and Theory Audit) is that `π_mve` degenerates toward uniform on this row, the planner's distilled signal is noisier, and `return_per_c` drops measurably. Coordinate descent is still enabled (`randomize_order=True`), so the agent-order permutation behaviour is identical to `planner_full`.

**Row 3 — `planner_no_coord_desc`.** The unified evaluator invokes the MVE planner with the planner's `use_coord_desc` flag forced to `False` (equivalently, `randomize_order=False` after the spec 06 rename). This disables the coordinate descent permutation step: the planner does a single agent-order pass instead of multiple permuted passes. CRN is still enabled (`use_crn=True`), so step-0 noise is still controlled, but the joint-action search is restricted to a single ordering. The expected effect is a measurable but smaller drop than `planner_no_crn` (because CRN dominates as the noise control, per v4.6 analysis), with the gap revealing how much joint-action search the planner does over and above sequential decoding.

**Row 4 — `planner_full`.** The default. The unified evaluator invokes the MVE planner with all eval-time enhancements on. Internally, the planner reads its own `use_crn` and `use_coord_desc` from `cfg.train` (the existing constructor pathway), but the dispatch boundary explicitly overrides both to `True` at eval time regardless of the training-time setting — this is the orthogonality guarantee from Lock 1. The `EvalReport.planner_full_return_mean` slot is populated; `EvalReport.direct_inference_return_mean` is `float("nan")` (see §5).

**N.B.** The two intermediate rows (`planner_no_crn`, `planner_no_coord_desc`) are deliberately one-axis-off-`planner_full` rather than full ablation of the eval-time planner machinery. The motivation is that the headline 4-mode comparison plot (spec 07 §4) shows monotonic degradation along a known axis: `direct_inference` < `planner_no_crn` ≤ `planner_no_coord_desc` < `planner_full` (the inequality between the two middle rows is the empirically interesting one). Two-axes-off would compound effects and obscure the per-component contribution.

---

## 3. New cfg fields

This spec adds **two fields** to `EvalConfig` (located at `hyper_mve/configs/eval_config.py`). Both are declared here and codified canonically in spec 08 §5 of pkg-08 as part of the "config-additions block" alongside the other three new cfg fields from sibling specs.

```python
# hyper_mve/configs/eval_config.py  — additions declared by pkg-08 spec 03 §3

from typing import Literal

@dataclass(frozen=True)
class EvalConfig:
    # ... existing fields unchanged (evaluate_freq, evaluate_episodes,
    #     eval_c_grid, eval_episodes_prior, eval_episodes_planner, c_segments,
    #     zero_shot_train_c, zero_shot_test_c, zero_shot_unseen_c,
    #     bell_curve_type_ratios) ...

    # === pkg-08 spec 03 additions ===

    # Canonical eval-time planner mode dispatch (spec 03 §2 truth table).
    # Default "planner_full" preserves all existing behaviour (the current code
    # path runs the planner with CRN+CoordDesc both on).
    eval_planner_mode: Literal[
        "direct_inference",
        "planner_no_crn",
        "planner_no_coord_desc",
        "planner_full",
    ] = "planner_full"

    # Back-compat short-circuit alias (spec 03 Lock 2). Deprecated on arrival.
    # False default preserves current behaviour. The only legal True usage is
    # alongside eval_planner_mode="direct_inference" (consistency check).
    eval_use_planner_direct_inference: bool = False
```

**Type + default summary:**

| Field | Type | Default | Owner | Consumed by |
|-------|------|---------|-------|-------------|
| `cfg.eval.eval_planner_mode` | `Literal["direct_inference", "planner_no_crn", "planner_no_coord_desc", "planner_full"]` | `"planner_full"` | `EvalConfig` | spec 03 §4 dispatch · spec 01 evaluator entry · spec 05 sweep harness override · spec 07 stats key |
| `cfg.eval.eval_use_planner_direct_inference` | `bool` | `False` | `EvalConfig` | spec 03 §4 dispatch (consistency check only) |

**Default-preserves-behaviour rationale.** `"planner_full"` ≡ the current default in-training eval path (planner with CRN + CoordDesc both on). The legacy `cfg.train.use_crn=True` / `cfg.train.use_coord_desc=True` defaults remain unchanged; the new fields do not shadow them. The short-circuit alias defaults to `False`, which combined with `eval_planner_mode="planner_full"` is the consistent "no override" state.

**Pkg-01 spec 05 sync.** Pkg-01 spec 05 (foundation schema) registers `eval_planner_mode` and `eval_use_planner_direct_inference` in its consumption table synchronously with spec 08 finalisation, so the fields are discoverable from the canonical V4Config schema view. The actual field additions land in `hyper_mve/configs/eval_config.py` (the pkg-01-owned file), counted in spec 08 §5 as part of the 5-field configurations-block patch.

---

## 4. Dispatch path (where spec 03 plugs into spec 01)

Per pkg-08 spec 01 §4.1 ("Wrapping (NOT replacing) `training/evaluation.py:run_eval`"), the unified evaluator `evaluate(runner, env_fn, cfg)` reads its eval-time config at entry and dispatches per mode. Spec 03 plugs into this flow at two points: the **consistency check** at entry, and the **dispatch boundary** before the per-c rollout loop.

### 4.1 Consistency check at evaluator entry (Lock 2)

Immediately after spec 01's existing entry guards (Self-Info strict check, env_fn flag verification, zero-shot disjoint-union check), the unified evaluator calls:

```python
# hyper_mve/eval/planner_modes.py

def validate_planner_mode_config(cfg: V4Config) -> None:
    """Lock 2 consistency check. Raises ValueError on inconsistency.
    Called by unified_evaluator.evaluate at entry, before any rollout."""
    mode = cfg.eval.eval_planner_mode
    short_circuit = cfg.eval.eval_use_planner_direct_inference
    if short_circuit and mode != "direct_inference":
        raise ValueError(
            f"eval_use_planner_direct_inference=True conflicts with "
            f"eval_planner_mode={mode!r}; set eval_planner_mode='direct_inference' "
            f"or eval_use_planner_direct_inference=False."
        )
    # Note: the converse (short_circuit=False, mode="direct_inference") is
    # consistent — the alias is opt-in and absence does not contradict the
    # canonical field. See Lock 2 for the full 4-case table.
```

This function is called unconditionally at evaluator entry. It does not allocate any environment, model, or planner state — it is a pure config-validation call.

### 4.2 Dispatch boundary (the mode → planner-flags translation)

Immediately before the per-c rollout loop (spec 01 §4.2 pseudocode line `for c in cfg.eval.zero_shot_test_c:`), the unified evaluator computes the dispatch invocation:

```python
# hyper_mve/eval/planner_modes.py

from dataclasses import dataclass

@dataclass(frozen=True)
class _PlannerInvocation:
    """Returned by _dispatch_planner_mode. The unified evaluator threads these
    three flags into the existing Worker.collect_episodes / run_eval machinery
    by overriding the planner's internal state at construction time (a
    sub-config replace using dataclasses.replace on a shallow copy of
    cfg.train, NOT a behaviour change on the underlying TrainConfig)."""
    skip_planner: bool          # True only for direct_inference; if True, the model's
                                # raw policy distribution is sampled (no MVE rollouts).
    use_crn: bool               # planner's use_crn override at this eval invocation
    randomize_order: bool       # planner's use_coord_desc / randomize_order override

def _dispatch_planner_mode(cfg: V4Config) -> _PlannerInvocation:
    """Translate the eval_planner_mode literal into the planner's three flags.
    Pure function of cfg.eval.eval_planner_mode; returns one of 4 invocations."""
    mode = cfg.eval.eval_planner_mode
    if mode == "direct_inference":
        return _PlannerInvocation(skip_planner=True, use_crn=False, randomize_order=False)
    if mode == "planner_no_crn":
        return _PlannerInvocation(skip_planner=False, use_crn=False, randomize_order=True)
    if mode == "planner_no_coord_desc":
        return _PlannerInvocation(skip_planner=False, use_crn=True, randomize_order=False)
    if mode == "planner_full":
        return _PlannerInvocation(skip_planner=False, use_crn=True, randomize_order=True)
    raise AssertionError(f"unreachable: unknown eval_planner_mode {mode!r}")
```

The `_PlannerInvocation.use_crn` and `_PlannerInvocation.randomize_order` fields for `direct_inference` are set to `False` rather than `None` purely as documentation values; they are not consulted because `skip_planner=True` short-circuits the planner construction entirely.

### 4.3 Internal vs external runner dispatch

Per spec 01 §6, the unified evaluator's dispatch boundary differs between internal runners (in-process `BaselineModel` / `HyperMuZeroModel` with `set_context_subjective`) and external runners (`ExternalBaselineRunner` with its own self-contained `.evaluate()` method).

**Internal runners (5 internal baselines + 3 curriculum overrides: hyper, oracle_only, infer_only).** The unified evaluator's `_evaluate_internal_hyper_or_baseline` path (spec 01 §4.2 pseudocode) is augmented to consult `_PlannerInvocation`. This branch is reached for **both** `HyperMuZeroModel` (the hyper-class variants — `hyper`, `oracle_only`, `infer_only`) and `BaselineModel` subclasses (the 5 internal baselines — `input_wide`, `input_deep`, `ma_muzero`, `no_belief`, `rewardhead_explicit_type`); routing is by `hasattr(runner, "set_context_subjective")` per spec 01 §6.1, which both classes satisfy because `BaselineModel` inherits `set_context_subjective` from pkg-04 spec 02's 7-API contract. Both runner types own their MVE planner through the shared `MuZeroTrainer` infrastructure (pkg-07 spec 03 §1.4 internal reuse contract), so the dispatch helper applies invariant of model type. The §7 test contract therefore includes a `BaselineModel`-specific dispatch test (`test_baseline_model_dispatches_4_modes`, §7.10) parametrised over the 5 internal baseline keys with the dispatch invocation asserted on a stubbed `BaselineModel` factory.

```python
# Inside spec 01 §4.2 _evaluate_internal_hyper_or_baseline, just before the
# per-c rollout loop:
invocation = _dispatch_planner_mode(cfg)
if invocation.skip_planner:
    # direct_inference path: skip MVE planner construction entirely.
    # The Worker / run_eval inner loop uses the policy distribution directly.
    legacy_results = run_eval(
        model, cfg, global_step=0,
        # spec 01 §4.2 path 1: cfg.eval.eval_planner_mode forces "prior" semantics
        # via an explicit override of eval_episodes_planner=0 at the sub-config
        # boundary. run_eval already supports this via its existing eval_mode dispatch.
    )
else:
    # planner_no_crn / planner_no_coord_desc / planner_full: invoke the planner
    # with the three overrides from invocation.
    cfg_eval_with_overrides = dataclasses.replace(
        cfg,
        train=dataclasses.replace(
            cfg.train,
            use_crn=invocation.use_crn,
            use_coord_desc=invocation.randomize_order,
        ),
    )
    legacy_results = run_eval(model, cfg_eval_with_overrides, global_step=0)
```

The `dataclasses.replace` call constructs a **shallow copy** of `cfg.train` with the two overrides applied. The original `cfg.train` is unchanged (the V4Config root object is not mutated). The copy is consumed by `run_eval`, which threads the overrides into the planner construction at its `MVEPlanner(...)` call site. This is **not** a behaviour change to the planner's training-time semantics — the original `cfg.train` is preserved everywhere else.

**External runners (3 Tier-1 external + MAMBA + 2 stubs).** External runners have no MVE planner; they have their own policy / critic and their own eval-time argmax routine. The unified evaluator's `_evaluate_external` path (spec 01 §4.2 pseudocode) **delegates** the entire eval to `runner.evaluate(env_fn, c_grid, episodes)`, which returns its own `EvalReport`. Per spec 01 §6.1 verbatim ("**does not rewrite** the returned `EvalReport` — it returns it as-is") and spec 01 §11 risk-table row ("External runner returns `EvalReport` with `eval_planner_mode` set to a value other than `'planner_full'` — Allowed (the runner is the authority on its own mode)"), the unified evaluator **does NOT mutate the returned report**. The contract is:

- The external runner's returned `EvalReport.eval_planner_mode` is **whatever literal the runner chose** and is preserved by the unified evaluator (no `dataclasses.replace` rewrite). Typically external runners set their slot to `"planner_full"` because they treat themselves as the "planner-equivalent" of their own family.
- **Semantic mapping.** External runners always behave as if `eval_planner_mode="direct_inference"` because they have no MVE planner — the runner's policy argmax is the only inference path, and CRN / coord-desc concepts don't apply. The four-mode literal taxonomy is structurally preserved on the report to enable `groupby` joins in spec 07 stats, but it has **no behavioural effect** on external rollouts: changing `cfg.eval.eval_planner_mode` from `"planner_full"` to `"planner_no_crn"` does not change the external runner's returned `return_per_c` values.
- The diagnostic fields `direct_inference_return_mean`, `planner_full_return_mean`, and `planner_prior_return_gap` are populated by the external runner's own internal logic per pkg-07 spec 04 §10. Per spec 01 §6.2 verbatim, external runners populate `direct_inference_return_mean = planner_full_return_mean = return_mean` and `planner_prior_return_gap = 0.0` (no prior-vs-planner decomposition exists). Spec 03 imposes no additional contract on these slots for external runners beyond what spec 01 §6.2 already locks.

**Stubs (`MARIEStub`, `GAStub`).** Stubs raise `NotImplementedError` at construction time per pkg-07 spec 01 §3.3 and never reach the dispatch boundary. The sweep harness (spec 05) catches the exception and writes a `status="skipped"` row to `runs/registry.jsonl`; no `EvalReport` is produced and the mode dispatch is irrelevant.

---

## 5. EvalReport population per mode

Each of the 4 modes populates the 32-field `EvalReport` (spec 01 §3.1) under the contract spec 01 §4.2 lines 272-274 + §5.4 lock: **the legacy `run_eval` runs BOTH the prior and the planner inner branches in a single call regardless of which mode dispatched**, so the diagnostic slots `direct_inference_return_mean`, `planner_full_return_mean`, and `planner_prior_return_gap` are populated from each row's own `run_eval` output. The table below enumerates the per-mode population contract for those three slots; all other 29 fields are populated identically regardless of mode (per Lock 3).

| EvalReport slot | `direct_inference` | `planner_no_crn` | `planner_no_coord_desc` | `planner_full` |
|-----------------|---------------------|------------------|--------------------------|----------------|
| `eval_planner_mode` | `"direct_inference"` | `"planner_no_crn"` | `"planner_no_coord_desc"` | `"planner_full"` |
| `direct_inference_return_mean` | = `mean(legacy_results, prefix="prior/")` | = `mean(legacy_results, prefix="prior/")` (prior branch is unaffected by `use_crn` / `randomize_order` overrides) | = `mean(legacy_results, prefix="prior/")` | = `mean(legacy_results, prefix="prior/")` |
| `planner_full_return_mean` | = `mean(legacy_results, prefix="planner/")` (under the default `use_crn=True` / `randomize_order=True` pass; this is `planner_full`-equivalent because no override is applied for this mode) | = `mean(legacy_results, prefix="planner/")` (under `use_crn=False` override; this is the **mode-overridden planner mean**, NOT the strict `planner_full` mean) | = `mean(legacy_results, prefix="planner/")` (under `randomize_order=False` override) | = `mean(legacy_results, prefix="planner/")` (standard planner pass) |
| `planner_prior_return_gap` | = `legacy_results["planner_prior_gap"]` (in-row gap, sourced directly from `run_eval`) | = `legacy_results["planner_prior_gap"]` (in-row gap under no-CRN override) | = `legacy_results["planner_prior_gap"]` (in-row gap under no-coord-desc override) | = `legacy_results["planner_prior_gap"]` |
| `return_per_c` source key | `prior/return_total_c*` (spec 01 §4.2 path 1) | `planner/return_total_c*` under the override (spec 01 §4.1 line 217 — `run_eval` is re-invoked with the override) | `planner/return_total_c*` under the override | `planner/return_total_c*` (spec 01 §4.2 path 2) |

**Single-mode-evaluation semantic.** In every single-mode call, both `direct_inference_return_mean` and `planner_full_return_mean` are populated from the same `run_eval` output (they are the mean over c of `prior/*` and `planner/*` TB tags respectively). The `planner_prior_return_gap` slot is sourced verbatim from `legacy_results["planner_prior_gap"]` (spec 01 §4.2 line 272 + §5.4 contract); it is **not** a `0.0` sentinel. For modes `direct_inference` and `planner_full`, this in-row gap is the canonical `pf_mean - di_mean` decomposition of distillation residual (the single scalar the paper has been tracking since v4-opt 2026-06). For modes `planner_no_crn` and `planner_no_coord_desc`, the in-row gap reflects the mode-overridden planner: `legacy_results["planner_prior_gap"]` under that mode's override is `(planner-with-this-override-mean - prior-mean)`, NOT `(planner_full-mean - prior-mean)`. Spec 07 stats reads the in-row gap directly from the `planner_full` row when computing the Ch6 headline planner-prior decomposition; it does NOT re-compute the gap by joining rows for that decomposition.

**Paired-mode-evaluation semantic.** The paired-mode protocol (§6 below) emits two rows for the same `(variant, seed, config_hash)`: one with `eval_planner_mode="direct_inference"` and one with `eval_planner_mode="planner_full"`. Although each row already carries its own in-row `planner_prior_return_gap`, the paired protocol is still useful for two distinct ends:

1. **Cross-mode comparison for the 4-mode plot (spec 07 §4).** The Ch6 4-mode comparison plot needs `return_mean` values from rows under all 4 distinct `eval_planner_mode` literals (so the X-axis has 4 categories); the paired protocol's row-emission contract guarantees these rows exist at the same `(variant, seed, config_hash)` tuple for clean per-seed pairing.
2. **Eval-time CRN / coord-desc isolation.** Joining a `planner_no_crn` row's `planner_full_return_mean` with a `planner_full` row's `planner_full_return_mean` (at the same `(variant, seed, config_hash)`) gives the CRN-isolated eval-time delta — which is NOT extractable from any single row's `planner_prior_return_gap` because that gap is always `(planner-this-mode - prior)`, never `(planner-mode-A - planner-mode-B)`.

The stats layer (spec 07) joins the pair by `(variant, seed, config_hash)` and exposes both kinds of comparison. The `planner_prior_return_gap` slot is NOT re-written by the stats layer (reports are frozen); cross-mode deltas live in spec 07's derived statistics table.

**External runner exception.** Per §4.3, external runners populate `direct_inference_return_mean = planner_full_return_mean = return_mean` and `planner_prior_return_gap = 0.0` (no prior-vs-planner decomposition exists; spec 01 §6.2 verbatim). Downstream stats treat this as "external runner — gap = 0 by construction".

**No `float("nan")` sentinels in single-mode internal-runner rows.** Earlier drafts of this spec used `nan` placeholders for the unfilled mean; spec 01's contract (verified at §4.2 lines 273-274) supersedes that draft. The only `nan` semantic in `EvalReport`'s gap-domain slots is **never realized for internal-runner rows** — both means and the gap are always populated. The `nan` convention is retained for `regret_per_c` on oracle ceiling cache miss (spec 02 §4.7), which is the only canonical sentinel use in the schema.

---

## 6. Paired-mode protocol (for the 4-mode comparison plot and cross-mode deltas)

The Ch6 headline "planner vs prior gap" scalar already lives in each row's `planner_prior_return_gap` slot (per §5 above — `run_eval` runs both prior and planner branches in a single call, so the in-row gap is always computable). The paired-mode protocol therefore exists for two distinct ends that the in-row gap does NOT support:

1. **The 4-mode comparison plot (Ch6.7 / spec 07 §4)**: needs `return_mean` from rows under all 4 distinct `eval_planner_mode` literals at the same `(variant, seed, config_hash)` tuple so the X-axis has 4 paired-by-seed categories.
2. **Cross-mode eval-time deltas** (e.g., `planner_full.return_mean - planner_no_crn.return_mean` = the CRN-isolated eval-time delta): not extractable from any single row's `planner_prior_return_gap` (which is `planner-this-mode - prior`, never `planner-mode-A - planner-mode-B`).

The sweep harness (spec 05) emits **at least two rows per (variant, seed) tuple** for every hyper-class variant in `REGISTRY` (one in `direct_inference` mode and one in `planner_full` mode minimum; the full 4-mode set is emitted only for spec 06's dedicated eval-mode ablation). The stats layer (spec 07) joins the pair and exposes the cross-mode deltas.

### 6.1 Row emission contract

For every hyper-class variant `v` (i.e., `v in {"hyper", "oracle_only", "infer_only", "input_wide", "input_deep", "ma_muzero", "no_belief", "rewardhead_explicit_type"}` — every variant that has an MVE planner; the 3 Tier-1 external + MAMBA are excluded because they have no prior-vs-planner gap) and every seed `s in {0, 1, 2, 3, 4}` (the default 5-seed protocol), the sweep harness emits:

```
row 2k:   variant=v, seed=s, eval_planner_mode="direct_inference"
row 2k+1: variant=v, seed=s, eval_planner_mode="planner_full"
```

These two rows share the same `config_hash` because the training config is identical — only the eval-time mode differs. Spec 05 §3 (`SweepConfig` cartesian) is responsible for the row enumeration; spec 03 only specifies the contract that, **for any hyper-class variant, both modes must be emitted at the same (seed, config_hash) tuple** so the gap is computable.

### 6.2 Stats-layer join

Spec 07 (`hyper_mve/experiments/stats.py`) reads the in-row gap from `planner_full` rows for the canonical Ch6 headline (it does NOT need to join `direct_inference` rows for that scalar — the gap is already in-row per §5). It joins rows under different `eval_planner_mode` literals only for cross-mode deltas:

```python
# Sketch — owned by spec 07 §3, included here only to document the contract spec 03 enables.
def read_planner_prior_gap_per_seed(registry_rows: list[RegistryRow]) -> dict[tuple[str, int], float]:
    """Returns (variant, seed) -> in-row planner_prior_return_gap from planner_full rows.

    NOT a cross-row join — the gap is sourced directly from each planner_full
    row's own EvalReport.planner_prior_return_gap slot (spec 03 §5 + spec 01
    §4.2 line 272).
    """
    gaps = {}
    for row in registry_rows:
        if row.variant in EXTERNAL_VARIANTS:
            continue
        if row.eval_planner_mode != "planner_full":
            continue
        gaps[(row.variant, row.seed)] = row.planner_prior_return_gap
    return gaps


def compute_cross_mode_deltas(registry_rows: list[RegistryRow]) -> dict[tuple[str, int, str, str], float]:
    """Returns (variant, seed, mode_a, mode_b) -> mode_a.return_mean - mode_b.return_mean.

    Cross-mode comparison join. Used for the 4-mode plot (spec 07 §4) and for
    isolating CRN / coord-desc eval-time effects (e.g., "planner_full" minus
    "planner_no_crn" = CRN-isolated delta).
    """
    by_key = defaultdict(dict)
    for row in registry_rows:
        if row.variant in EXTERNAL_VARIANTS:
            continue
        key = (row.variant, row.seed, row.config_hash)
        by_key[key][row.eval_planner_mode] = row
    deltas = {}
    for (variant, seed, _hash), modes in by_key.items():
        for mode_a in modes:
            for mode_b in modes:
                if mode_a == mode_b:
                    continue
                deltas[(variant, seed, mode_a, mode_b)] = (
                    modes[mode_a].return_mean - modes[mode_b].return_mean
                )
    return deltas
```

The contract spec 03 enables (and that spec 07 relies on for the 4-mode plot specifically) is: **a hyper-class variant in `runs/registry.jsonl` has rows in BOTH `direct_inference` and `planner_full` modes at every (seed, config_hash) tuple at minimum, or the sweep harness has logged a `[WARN]` row indicating which mode is missing.** Spec 05 enforces this minimum at sweep planning time; the full 4-mode set is emitted only when spec 06's dedicated eval-mode ablation cell is selected.

### 6.3 Why this lives in the stats layer, not in spec 03

Spec 03 deliberately does **not** introduce a third evaluator path that runs both modes and emits a single combined report. Three reasons:

- **Schema simplicity.** The `EvalReport` schema is a single-row schema; adding a "paired report" type would double the row size and break the JSONL append-only registry's one-row-per-eval invariant (spec 05 §4).
- **Sweep harness compositionality.** The sweep harness (spec 05) already emits multi-row sweeps; emitting two rows per pair is a one-line extension to its cartesian enumeration, whereas extending the evaluator would require nested-eval state.
- **Re-runnability.** If one mode's row fails (e.g., GPU OOM on the planner_full row), the other row is still valid; the sweep harness can re-dispatch only the failing row. A combined report would require both modes to succeed atomically.

The paired-mode protocol is therefore a **sweep-harness pattern**, not a spec-03 evaluator pattern. Spec 03's contract is: each row has a single mode, the schema is single-row, and the gap is derived in stats.

---

## 7. Test contract — 10 named tests

Located under `tests/eval/`. Each name below corresponds to a specific Lock or §2/§5/§6 contract.

### 7.1 `test_all_4_planner_modes_dispatch` — C8-EVAL-MODE1 (§2 + Lock 3)

For each of the 4 literal mode strings, call `evaluate(runner, env_fn, cfg)` with `cfg.eval.eval_planner_mode` set to that string. Assert (i) no exception, (ii) the returned `EvalReport.eval_planner_mode` equals the requested literal, (iii) `EvalReport` is a proper `EvalReport` dataclass instance with all 32 fields populated:

```python
@pytest.mark.parametrize("mode", [
    "direct_inference",
    "planner_no_crn",
    "planner_no_coord_desc",
    "planner_full",
])
def test_all_4_planner_modes_dispatch(mode):
    cfg = V4Config()
    cfg = dataclasses.replace(cfg, eval=dataclasses.replace(cfg.eval, eval_planner_mode=mode))
    runner = _make_dummy_hyper_runner(cfg)
    env_fn = lambda: ResourceCommonsPettingZooEnv(cfg.env, oracle_mode=False, eval_info_mode=True)
    report = evaluate(runner, env_fn, cfg)
    assert isinstance(report, EvalReport)
    assert report.eval_planner_mode == mode
    # Lock 3: full schema population invariant
    assert set(report.return_per_c) == set(cfg.eval.zero_shot_test_c)
```

### 7.2 `test_consistency_check_raises_on_inconsistent_short_circuit` — Lock 2

```python
def test_consistency_check_raises_on_inconsistent_short_circuit():
    cfg = V4Config()
    cfg = dataclasses.replace(
        cfg, eval=dataclasses.replace(
            cfg.eval,
            eval_planner_mode="planner_full",
            eval_use_planner_direct_inference=True,
        ),
    )
    runner = _make_dummy_hyper_runner(cfg)
    env_fn = lambda: ResourceCommonsPettingZooEnv(cfg.env, oracle_mode=False, eval_info_mode=True)
    with pytest.raises(ValueError, match="conflicts with eval_planner_mode"):
        evaluate(runner, env_fn, cfg)
```

### 7.3 `test_consistency_check_passes_on_consistent_short_circuit` — Lock 2 positive path

```python
def test_consistency_check_passes_on_consistent_short_circuit():
    cfg = V4Config()
    cfg = dataclasses.replace(
        cfg, eval=dataclasses.replace(
            cfg.eval,
            eval_planner_mode="direct_inference",
            eval_use_planner_direct_inference=True,
        ),
    )
    runner = _make_dummy_hyper_runner(cfg)
    env_fn = lambda: ResourceCommonsPettingZooEnv(cfg.env, oracle_mode=False, eval_info_mode=True)
    report = evaluate(runner, env_fn, cfg)
    assert report.eval_planner_mode == "direct_inference"
```

### 7.4 `test_direct_inference_skips_planner` — §2 row 1 + §4.2

Asserts the planner public entry is **not** called when `eval_planner_mode="direct_inference"`:

```python
def test_direct_inference_skips_planner(monkeypatch):
    cfg = V4Config()
    cfg = dataclasses.replace(cfg, eval=dataclasses.replace(cfg.eval, eval_planner_mode="direct_inference"))
    runner = _make_dummy_hyper_runner(cfg)
    env_fn = lambda: ResourceCommonsPettingZooEnv(cfg.env, oracle_mode=False, eval_info_mode=True)
    planner_calls = []
    original_planner = MVEPlanner.__init__
    def mock_init(self, *a, **kw):
        planner_calls.append((a, kw))
        return original_planner(self, *a, **kw)
    monkeypatch.setattr(MVEPlanner, "__init__", mock_init)
    evaluate(runner, env_fn, cfg)
    assert planner_calls == [], "MVEPlanner must NOT be constructed in direct_inference mode"
```

### 7.5 `test_planner_no_crn_passes_use_crn_false` — §2 row 2 + §4.2

Asserts the dispatch boundary correctly forwards `use_crn=False` to the planner:

```python
def test_planner_no_crn_passes_use_crn_false():
    cfg = V4Config()
    cfg = dataclasses.replace(cfg, eval=dataclasses.replace(cfg.eval, eval_planner_mode="planner_no_crn"))
    invocation = _dispatch_planner_mode(cfg)
    assert invocation.skip_planner is False
    assert invocation.use_crn is False
    assert invocation.randomize_order is True
```

### 7.6 `test_planner_no_coord_desc_passes_randomize_order_false` — §2 row 3 + §4.2

```python
def test_planner_no_coord_desc_passes_randomize_order_false():
    cfg = V4Config()
    cfg = dataclasses.replace(cfg, eval=dataclasses.replace(cfg.eval, eval_planner_mode="planner_no_coord_desc"))
    invocation = _dispatch_planner_mode(cfg)
    assert invocation.skip_planner is False
    assert invocation.use_crn is True
    assert invocation.randomize_order is False
```

### 7.7 `test_external_runner_collapses_all_4_modes` — §4.3 external runner contract

For an external runner, all 4 modes produce reports with `direct_inference_return_mean == planner_full_return_mean == return_mean` and the unified evaluator overrides `eval_planner_mode` to match the requested mode:

```python
@pytest.mark.parametrize("mode", [
    "direct_inference", "planner_no_crn", "planner_no_coord_desc", "planner_full",
])
def test_external_runner_collapses_all_4_modes(mode):
    cfg = V4Config()
    cfg = dataclasses.replace(cfg, eval=dataclasses.replace(cfg.eval, eval_planner_mode=mode))
    runner = _make_mock_external_mappo_runner(cfg)
    env_fn = lambda: ResourceCommonsPettingZooEnv(cfg.env, oracle_mode=False, eval_info_mode=True)
    report = evaluate(runner, env_fn, cfg)
    assert report.eval_planner_mode == mode    # unified evaluator overrides the slot
    assert report.direct_inference_return_mean == report.return_mean
    assert report.planner_full_return_mean == report.return_mean
```

### 7.8 `test_paired_protocol_gap_computation` — §6.2 stats join

Given two report objects (one `direct_inference`, one `planner_full`) at the same `(variant, seed, config_hash)`, the stats layer computes the gap correctly:

```python
def test_paired_protocol_gap_computation():
    di_report = _make_dummy_report(
        variant="hyper", seed=0, config_hash="abc",
        eval_planner_mode="direct_inference",
        direct_inference_return_mean=10.0,
        planner_full_return_mean=float("nan"),
        return_mean=10.0,
    )
    pf_report = _make_dummy_report(
        variant="hyper", seed=0, config_hash="abc",
        eval_planner_mode="planner_full",
        direct_inference_return_mean=float("nan"),
        planner_full_return_mean=12.5,
        return_mean=12.5,
    )
    gaps = compute_planner_prior_gap([di_report, pf_report])
    assert gaps[("hyper", 0)] == pytest.approx(2.5)
```

### 7.9 `test_population_contract_per_mode` — §5 population table

Asserts each mode's `direct_inference_return_mean` / `planner_full_return_mean` / `planner_prior_return_gap` slots are populated per the §5 table (per spec 01 §4.2 lines 272-274 contract — all three slots are populated in every single-mode internal-runner call; the gap is sourced from `run_eval`'s `planner_prior_gap` output, NOT a `0.0` sentinel):

```python
@pytest.mark.parametrize("mode", [
    "direct_inference",
    "planner_no_crn",
    "planner_no_coord_desc",
    "planner_full",
])
def test_population_contract_per_mode(mode):
    cfg = V4Config()
    cfg = dataclasses.replace(cfg, eval=dataclasses.replace(cfg.eval, eval_planner_mode=mode))
    runner = _make_dummy_hyper_runner(cfg)
    env_fn = lambda: ResourceCommonsPettingZooEnv(cfg.env, oracle_mode=False, eval_info_mode=True)
    report = evaluate(runner, env_fn, cfg)
    # Internal-runner contract per spec 01 §4.2 + §5.4: both diagnostic means
    # AND the in-row gap are populated for ALL modes. No nan sentinels.
    assert not math.isnan(report.direct_inference_return_mean)
    assert not math.isnan(report.planner_full_return_mean)
    assert not math.isnan(report.planner_prior_return_gap)
    # In-row gap is sourced from run_eval's planner_prior_gap (≠ 0 in general).
    # Algebraic identity: report.planner_prior_return_gap == report.planner_full_return_mean - report.direct_inference_return_mean
    # (up to floating-point rounding; equality verified within rel=1e-6).
    assert report.planner_prior_return_gap == pytest.approx(
        report.planner_full_return_mean - report.direct_inference_return_mean,
        rel=1e-6,
    )
```

### 7.10 `test_baseline_model_dispatches_4_modes` — §4.3 BaselineModel branch

Mirrors §7.1 (`test_all_4_planner_modes_dispatch`) but parametrized over the 5 internal `BaselineModel` subclasses (5 internal × 4 modes = 20 cells). Asserts the unified evaluator routes `BaselineModel` instances through the same `_evaluate_internal_hyper_or_baseline` path as `HyperMuZeroModel` (per §4.3 BaselineModel-routing clarification), and that the dispatch helper returns the expected `_PlannerInvocation` per the §4.2 truth table for every internal baseline:

```python
@pytest.mark.parametrize("variant", [
    "input_wide", "input_deep", "ma_muzero", "no_belief", "rewardhead_explicit_type",
])
@pytest.mark.parametrize("mode", [
    "direct_inference", "planner_no_crn", "planner_no_coord_desc", "planner_full",
])
def test_baseline_model_dispatches_4_modes(variant, mode):
    cfg = V4Config()
    cfg = dataclasses.replace(cfg, eval=dataclasses.replace(cfg.eval, eval_planner_mode=mode))
    runner = _make_dummy_baseline_runner(cfg, variant)
    env_fn = lambda: ResourceCommonsPettingZooEnv(cfg.env, oracle_mode=False, eval_info_mode=True)
    # BaselineModel has set_context_subjective per pkg-04 spec02 7-API contract.
    assert hasattr(runner, "set_context_subjective")
    report = evaluate(runner, env_fn, cfg)
    assert isinstance(report, EvalReport)
    assert report.eval_planner_mode == mode
    # Lock 3: full schema population invariant for ALL baseline variants.
    assert set(report.return_per_c) == set(cfg.eval.zero_shot_test_c)
    # §5 contract: both diagnostic means populated even for baseline variants.
    assert not math.isnan(report.direct_inference_return_mean)
    assert not math.isnan(report.planner_full_return_mean)
```

---

## 8. Downstream patches (declared here, codified in spec 08 §5)

This spec produces **one config-additions patch** and explicitly **no behaviour patches** to existing production files outside `eval_config.py`. Honest patch accounting:

### 8.1 `hyper_mve/configs/eval_config.py` — +2 fields (config-additions; no behaviour on its own)

- **Size**: +2 lines on the `EvalConfig` dataclass field list (the two new fields declared in §3), plus 1 line of `from typing import Literal` import if not already present, plus optional 4 lines of inline comments. Net behaviour-bearing edit: 2 lines.
- **Behaviour**: The fields are read-only data. Behaviour comes from §4 (the dispatch path in `unified_evaluator.evaluate` + the helper in `eval/planner_modes.py`); the fields themselves are inert until consumed.
- **Default**: `eval_planner_mode="planner_full"` and `eval_use_planner_direct_inference=False`. Both defaults preserve the current code path's behaviour exactly.
- **Tag**: `[config-additions]`.
- **Test gate**: §7.1 (all 4 modes dispatch) + §7.3 (consistent short-circuit).

### 8.2 `hyper_mve/training/evaluation.py` — NO changes from spec 03

`[no-behaviour]` — spec 03 does not modify `training/evaluation.py:run_eval`. Spec 01 Lock 1 already locks `run_eval` as wrapped-not-replaced; spec 03 inherits this lock. The dispatch boundary lives in the wrapper (`unified_evaluator.evaluate`), not in `run_eval`.

### 8.3 `hyper_mve/planning/mve_planner.py` — NO changes from spec 03

`[no-behaviour]` — spec 03 does not modify `mve_planner.py`. The planner's existing `use_crn` and `use_coord_desc` constructor flags (read from `cfg.train.use_crn` and `cfg.train.use_coord_desc` at planner construction time) are sufficient for spec 03's dispatch — the unified evaluator threads its eval-time overrides via `dataclasses.replace(cfg.train, use_crn=..., use_coord_desc=...)` on the path to `run_eval`. The planner itself sees the (overridden) `cfg.train` and obeys it. Spec 06 of pkg-08 separately adds the `mve_joint_enumerate` flag to `mve_planner.py` (+1 line for the Joint-cell branch); that patch is governed by spec 06 and is not in spec 03's scope.

### 8.4 `hyper_mve/eval/planner_modes.py` — NEW file (declared in §4)

`[new-file]` — spec 03's only new production file. Contains `validate_planner_mode_config(cfg)`, `_PlannerInvocation` dataclass, and `_dispatch_planner_mode(cfg)`. ~30 lines of code. Consumed by spec 01's `unified_evaluator.evaluate` at the entry-check and dispatch-boundary points; not consumed by any other production file in the codebase.

---

## 9. Integration hooks (cross-spec)

| Consumer | Consumed from this spec | Use |
|----------|--------------------------|-----|
| **spec 01** unified evaluator | (i) `validate_planner_mode_config(cfg)` at entry (Lock 2 check); (ii) `_dispatch_planner_mode(cfg) -> _PlannerInvocation` at the dispatch boundary before the per-c rollout loop; (iii) the 3-slot population contract from §5 (`eval_planner_mode`, `direct_inference_return_mean`, `planner_full_return_mean`) | Spec 01 §4.2 pseudocode threads the dispatch helper output into the existing `run_eval` call by overriding `cfg.train.use_crn` / `cfg.train.use_coord_desc` via `dataclasses.replace`. The 3 diagnostic slots are populated per §5 table. |
| **spec 02** zero-shot + c_hidden + regret | Orthogonality — each of the 4 modes runs the full 7-value zero-shot test grid independently. The `return_per_c` mapping is keyed identically across modes (`set(return_per_c) == set(cfg.eval.zero_shot_test_c)`). | Spec 02 does not alter the mode dispatch; spec 03 does not alter the zero-shot partition logic. The two specs compose multiplicatively at the report level (4 modes × 7 c-vals × 30 episodes per c). |
| **spec 05** sweep harness + RunRegistry | The paired-mode protocol (§6) — sweep harness emits two rows per (variant, seed) for hyper-class variants, one in `direct_inference` mode and one in `planner_full`. The 4 eval-time modes themselves are a sweep axis the harness can override via `cfg.eval.eval_planner_mode`. | Spec 05 §3 cartesian enumeration includes `eval_planner_mode` as one of the override axes. The default sweep enumerates only `planner_full`; the paired protocol adds `direct_inference` rows for hyper-class variants; the eval-mode ablation (if enabled) enumerates all 4 modes for selected variants. |
| **spec 06** ablation CLI + cells (Ablation 4) | Decoupling from the training-time CRN × CoordDesc axis (Lock 1). Ablation 4 trains 4 cells with varied `cfg.train.use_crn` / `cfg.train.randomize_order` and evaluates all 4 cells uniformly at `cfg.eval.eval_planner_mode="planner_full"`. | Spec 06 §3 (Abl4 cell definition) explicitly pins `cfg.eval.eval_planner_mode="planner_full"` across all 4 training cells. The two axes form a 4 × 4 product in principle but the default sweep does not exercise it. |
| **spec 07** statistics + comparison | The paired-mode join (§6.2) — stats layer joins `(variant, seed, config_hash)` pairs to compute `planner_prior_return_gap` post-hoc. The 4-mode comparison plot (Ch6.7) consumes per-mode `return_mean` from 4 rows. | Spec 07 §3 implements `compute_planner_prior_gap(registry_rows)` per §6.2 sketch. Welch t-test is applied to the gap distribution across seeds (5 seeds → 5 gaps → t-test vs null hypothesis "gap = 0"). |
| **spec 08** integration contracts | (a) Canonical declaration of the two new cfg fields (§3); (b) Pkg-05 downstream-patch declarations — spec 03 has none beyond `eval_config.py` (the `[no-behaviour]` non-edits to `mve_planner.py` and `training/evaluation.py` are declared in §8.2 + §8.3 above). | Spec 08 §5 mirrors §3 + §8 verbatim. Drift caught by spec 08 §6 drift detector. |
| **pkg-07 spec 04** ResourceCommonsPettingZooEnv | `env_fn` consumed identically by all 4 modes (per Lock 3). The two-flag info gate (`oracle_mode=False`, `eval_info_mode=True`) is invariant across modes. | Pkg-07 spec 04 §6 + §10 — env is constructed once per evaluation regardless of mode; the mode does not affect env behaviour. |

---

## 10. Cross-references

### Upstream anchors (consumed by this spec)

- **pkg-08 design.md §3.3** — four planner eval mode truth table (this spec §2 inherits verbatim).
- **pkg-08 design.md §4 D7** — decoupling of eval-mode axis from Ablation 4 training-time axis; default `"planner_full"`; alias-short-circuit semantics.
- **pkg-08 design.md §4 D10** — 5 new cfg fields; rows 4 + 5 (`eval_planner_mode` + `eval_use_planner_direct_inference`) are the two spec 03 owns.
- **pkg-08 README C8-EVAL-MODE1** — surfaced as `test_all_4_planner_modes_dispatch` in §7.1.
- **pkg-08 README §"5 新 cfg 字段穷举" rows 4 + 5** — the canonical declaration site for the two new fields (default values match §3 here).
- **pkg-08 spec 01 §3.1** — `EvalReport` `@dataclass` schema; spec 03 populates 3 slots (`eval_planner_mode`, `direct_inference_return_mean`, `planner_full_return_mean`) and respects all 32 fields.
- **pkg-08 spec 01 §4.2** — `_evaluate_internal_hyper_or_baseline` pseudocode; spec 03's dispatch helper is consumed at the line immediately before the per-c rollout loop.
- **pkg-08 spec 01 §6** — external runner delegation (`_evaluate_external`); spec 03 §4.3 specifies the `eval_planner_mode` slot override on the external-returned report.
- **pkg-08 spec 02 §7 integration hooks row "spec 03 four planner eval mode"** — orthogonality guarantee with zero-shot partition.
- **pkg-08 spec 06** (drafting) — Ablation 4 training-time CRN × CoordDesc cell matrix; spec 03's Lock 1 decoupling is the contract spec 06 relies on.
- **pkg-07 spec 04 §6** — `reset(options={"c": c})` signature; consumed identically by all 4 modes via `env_fn`.
- **pkg-07 spec 04 §10** — `evaluate(env_fn, c_grid, episodes) -> EvalReport` signature lock; external runners' returned report has its `eval_planner_mode` slot overridden by the unified evaluator per §4.3.

### Downstream consumption (spec 03 → others)

- **spec 01** consumes `validate_planner_mode_config` (entry check) + `_dispatch_planner_mode` (dispatch boundary).
- **spec 05** consumes the paired-mode protocol contract (§6.1) — emits two rows per hyper-class (variant, seed) pair.
- **spec 07** consumes the paired-mode join (§6.2) — computes `planner_prior_return_gap` from joined rows.
- **spec 08** mirrors §3 (cfg fields) and §8 (downstream patches) as part of the canonical integration-contract enumeration.

### Existing repo ground-truth anchors

- `hyper_mve/configs/eval_config.py` — `EvalConfig @dataclass(frozen=True)` body; spec 03 adds 2 fields after the existing `bell_curve_type_ratios` field. (Symbol-anchored; line numbers omitted because pkg-08 spec 02 §5.2 also lands a `c_visible: bool` addition on `EnvConfig` in the same window and the relative ordering of edits across specs may shift line counts.)
- `hyper_mve/configs/train_config.py` — `TrainConfig` fields `use_crn: bool = True` + `use_coord_desc: bool = True` (the training-time flags Lock 1 keeps orthogonal). Spec 06 of pkg-08 renames `use_coord_desc → randomize_order` with alias-with-deprecation; spec 03 is invariant to the rename because it operates on the eval-time field `eval_planner_mode`, not on the training-time field.
- `hyper_mve/planning/mve_planner.py` — symbol-anchored references (line numbers omitted because spec 06 of pkg-08 adds `mve_joint_enumerate` +1 line elsewhere in this file, which will shift line numbers below the edit site):
  - `MVEPlanner.__init__`: the constructor reads `self.use_crn = cfg.train.use_crn` and `self.use_coord_desc = cfg.train.use_coord_desc` from the threaded cfg. Spec 03 threads its overrides via `dataclasses.replace(cfg.train, ...)` on the path into `MVEPlanner.__init__`, so the planner's behaviour is correctly modulated without modifying the planner itself.
  - `if self.use_coord_desc:` branch within the agent-order-permutation block; spec 03 controls whether this branch is entered at eval time via the `randomize_order` override.
  - `elif step == 0 and i in step0_actions_expanded and self.use_crn:` branch within the step-0 CRN block; spec 03 controls whether this branch is entered at eval time via the `use_crn` override.
- `hyper_mve/training/evaluation.py` — `run_eval(model, cfg, global_step=0)` function; spec 03 does not modify it (Lock 1 of spec 01 inherited). Symbol-anchor only; line numbers omitted because the file is the lock-1 target of `test_run_eval_not_modified` (spec 01 §4.4) which byte-pins the source.

---

## 11. Anchors (verbatim grep targets for spec 08 §6 drift detector)

- `pkg-08 spec 03 Lock 1: Eval-time vs training-time axes are orthogonal and never collapsed`
- `pkg-08 spec 03 Lock 2: cfg.eval.eval_planner_mode is the canonical single source of truth; the short-circuit alias is an inconsistency-detector`
- `pkg-08 spec 03 Lock 3: All 4 modes populate the full EvalReport schema; the unified evaluator handles all 4 modes via dispatch`
- `pkg-08 spec 03 §2: 4-mode truth table — direct_inference / planner_no_crn / planner_no_coord_desc / planner_full`
- `pkg-08 spec 03 §3: cfg.eval.eval_planner_mode: Literal[...] = "planner_full"`
- `pkg-08 spec 03 §3: cfg.eval.eval_use_planner_direct_inference: bool = False`
- `pkg-08 spec 03 §4.1: validate_planner_mode_config(cfg) — Lock 2 consistency check at evaluator entry`
- `pkg-08 spec 03 §4.2: _dispatch_planner_mode(cfg) -> _PlannerInvocation — pure dispatch helper`
- `pkg-08 spec 03 §4.3: external runner preserves its own eval_planner_mode literal (no override); spec 01 §6.1 + §11 risk table`
- `pkg-08 spec 03 §4.3: external runners semantically behave as direct_inference-equivalent (no MVE planner)`
- `pkg-08 spec 03 §4.3: BaselineModel routes through same _evaluate_internal_hyper_or_baseline path as HyperMuZeroModel`
- `pkg-08 spec 03 §5: per-mode population — BOTH direct_inference_return_mean AND planner_full_return_mean populated in every single-mode internal-runner call; gap sourced from run_eval (NOT a 0.0 sentinel)`
- `pkg-08 spec 03 §6: paired-mode protocol — emits ≥2 rows per (variant, seed) for hyper-class variants; stats layer joins for 4-mode plot + cross-mode deltas`
- `pkg-08 spec 03 §7: 10 named tests — test_all_4_planner_modes_dispatch / test_consistency_check_raises_on_inconsistent_short_circuit / etc. + test_baseline_model_dispatches_4_modes`
- `pkg-08 spec 03 §8.1: hyper_mve/configs/eval_config.py +2 fields [config-additions]`
- `pkg-08 spec 03 §8.2: hyper_mve/training/evaluation.py NO changes [no-behaviour]`
- `pkg-08 spec 03 §8.3: hyper_mve/planning/mve_planner.py NO changes [no-behaviour]`
- `pkg-08 spec 03 §8.4: hyper_mve/eval/planner_modes.py NEW file [new-file]`
