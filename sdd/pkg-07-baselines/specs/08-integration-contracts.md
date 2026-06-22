# Spec 08 — Integration Contracts (Hard Contracts + Downstream Patch Declarations + Drift Detector)

> **Anchors**: design.md §3.3 (CLI ↔ factory-arg ↔ model-class 14-row map) · §3.4 (vendoring sources) · §3.5 (N-parametric adapter + two-flag info gating + CTDE legitimacy footnote) · §4 D2 (`create_baseline(cfg, variant)` 工厂归属 + alias-with-DeprecationWarning) · §4 D3 (Internal vs External 两 namespace 分派 + union return type) · §4 D10 (`cfg.baselines.*` 5-字段穷举) · README §"输出清单" (`新增` block: 3 new modules + tests subtree) + README §"修改" (2-patch table for existing-file modifications: `train_main.py` + `v4_config.py`) + README §"待声明 cfg 字段（5 个）" (verbatim 5-field enumeration) + README §"spec 间引用表" (M6 ref-matrix) · pkg-06 spec 08 `08-integration-contracts.md` (supersede source — pkg-07 spec 08 carries forward its "contract mother-doc" role and absorbs its 5 复用约束 verbatim as §3.1).
> **Status**: SDD only — describes the **contract surface** assembled from spec 01-07 and re-published for pkg-08 and pkg-02/pkg-05 downstream consumers. No new behaviour, no new model class, no new test logic — every contract in this document is sourced from a sibling spec and re-anchored here for single-document review.
> **Cross-refs**: spec 01 (`01-baseline-registry-and-cli.md` — factory + 11-key REGISTRY + `cli_to_factory_arg` + DeprecationWarning alias + 5 cfg.baselines fields) · spec 02 (`02-shared-backbones-internal.md` — `count_conditioning_params` SoT) · spec 03 (`03-internal-variants.md` — 7-API + Self-Info + stateful + grad-gating common surface) · spec 04 (`04-pettingzoo-adapter.md` — N-parametric adapter + two-flag info gating + `_filter_info` + `env_fn` factory) · spec 05 (`05-external-mappo.md` §13 §14 integration hooks + MAPPO `EvalReport` field-population matrix) · spec 06 (`06-external-qmix-mamuzero-mamba.md` — QMIX/MA-MuZero-GH `EvalReport` mirror + MAMBA `IS_SOURCED` toggle + MARIE/GA stub constructors) · spec 07 (`07-fairness-protocol.md` §8 cross-spec consumer table + §10 anchors) · pkg-08 spec 01 (`01-unified-evaluator.md` — `EvalReport` schema mother-doc; reverse-consumed by §4 below) · pkg-08 spec 05 (sweep harness consumes `REGISTRY` + `cfg.baselines.external_lr_sweep_grid`) · pkg-08 spec 07 (stats reduces external 披露式 disclosure table per §5 below) · pkg-08 spec 08 (`08-integration-contracts.md` — the symmetric peer; drift detection across the pkg-07 ↔ pkg-08 contract bridge).

---

## ⚠️ Header — three hard locks

### Lock 1 — Spec 08 is a CONTRACT MIRROR ONLY (sibling-precedes-mirror invariant)

Every contract listed in this document is **sourced from a sibling spec** (spec 01-07 of pkg-07). If a contract appears here without an upstream sibling source, that is an authoring bug, not a new contract. The drift detector in §6 treats spec 08 as **derivative**: its source-of-truth check is **unilateral** — spec 08 cites the sibling, but the sibling does NOT cite spec 08 back. This is the same shape pkg-06 spec 08 used (and which this spec supersedes; see §0.1).

Editorial workflow: **edit the sibling spec first, then mirror here**. Mirror-first-then-fix-sibling is a CI failure (the §6.3 grep table will fail). Spec 08 is read as the "one-document overview" for reviewers of pkg-08, pkg-02, and pkg-05; reviewers approving spec 08 in isolation are approving the **derivative**, not the truth — the truth lives in spec 01-07.

### Lock 2 — Three categories of contract are exhaustive

This spec locks exactly three categories of contract. There is no fourth.

- **(A) Python API surface**: `create_baseline(cfg, variant)` + `BaselineLike Union` + `REGISTRY` + `cli_to_factory_arg` + `evaluate(env_fn, c_grid, episodes) -> EvalReport` signatures. §2 owns the API; §3 expands the API into the union type discipline + alias shim + supersede declarations.
- **(B) Four downstream code patches**: `cfg.baselines` dataclass declaration + PettingZoo adapter module + `hyper_mve/baselines/*` new module tree + `train_main.py` CLI extension. **Pkg-02 obs-mask is NOT in this list** — that 3-line patch is governed by pkg-08 spec 02 §5.1 + pkg-08 spec 08 §5, and `[pkg-08 spec 02 §5.1]` is the only legal cite for it. §7 owns the four-patch table.
- **(C) `BaselinesConfig` 5-field exhaustive enumeration**: design D10 lock — 4 internal capacity-tuning knobs + 1 external LR-sweep grid mapping. Per-impl tuning constants (spec 05 §11 + spec 06 §2.x / §3.x / §4.x) are explicitly excluded. §5 owns the cfg lock.

Adding a fourth category to this list requires a **synchronous edit to spec 01-07**: every category here corresponds to at least one upstream sibling §-block. A category that has no upstream §-block cannot exist here.

### Lock 3 — Pkg-08 reverse-consumption contracts mirror pkg-08 README §"🔁" 5 anchors verbatim

§8 below mirrors the 5-anchor list from pkg-08 README §"🔁 pkg-07 → pkg-08 契约对账". The 5 anchors are: (1) pkg-07 spec 01 §2.1 REGISTRY 11-keys → pkg-08 spec 05 §3; (2) pkg-07 spec 01 §2.3 量词 canonical → pkg-08 spec 05 §3 + spec 08 §5; (3) pkg-07 spec 04 N-parametric adapter + 两 flag → pkg-08 spec 01 §4 external runner eval 通路; (4) pkg-07 spec 08 `evaluate()` → `EvalReport` 签名 → pkg-08 spec 01 §3 + spec 08 §3; (5) pkg-07 spec 08 `BaselineLike` Union → pkg-08 spec 05 sweep harness type sig. Any drift in pkg-08 README (or pkg-08 spec 08 mirror) triggers drift here. §6.3 grep table includes these 5 anchors as canonical cross-package targets.

---

## 0.1 Supersede statement (pkg-06 spec 08)

Pkg-06 spec 08 (`sdd/pkg-06-baselines/specs/08-integration-contracts.md`) is **superseded** by this spec. Pkg-06 spec 08's 5 复用约束 table (`MuZeroTrainer / Worker / EpisodeReplayBuffer / compose_total_loss / model-class-only-variant`) is **inherited verbatim** here as §3.1 and remains the canonical "internal namespace shared trainer" contract. Pkg-06 spec 08's `create_baseline_model(cfg, variant)` signature is **renamed** to `create_baseline(cfg, variant)` per design D2 + spec 01 §3.2; the old name is preserved as an alias with `DeprecationWarning(stacklevel=2)` (one-release window). Pkg-06 spec 08's "5 variant" Methods-table block expands to **9 main-table columns = hyper + 5 internal + 3 Tier-1** (canonical §5.5 量词; the column total is what appears in the paper main table). MAMBA-if-sourced is a conditional 10th column (per design D8). MARIE/GA are 2 permanent stubs (per design D9) NOT in the main table. **Note on quantifier framings**: §5.5 below uses "9 = hyper + 5 internal + 3 Tier-1" (with hyper, the paper main-table accounting); the alternate framing "9 = 5 internal + 3 Tier-1 + 1 MAMBA-if-sourced" (without hyper) is the algorithm-comparison framing used in some places — both are valid for their context, but §5.5's "with hyper" version is the canonical paper-table accounting and the one §6.3 grep target enforces.

---

## 1. Purpose

Spec 08 exists for **single-document review**. A reviewer approving pkg-07 → pkg-08 integration should be able to read **one file** and answer five questions:

1. **What Python API does pkg-08 consume from pkg-07?** (§2 + §3 — `create_baseline` factory, `BaselineLike Union`, 11-key `REGISTRY`, `cli_to_factory_arg` CLI converter, `.evaluate(env_fn, c_grid, episodes) -> EvalReport` on both `BaselineModel` and `ExternalBaselineRunner`.)
2. **What 5 cfg fields does pkg-07 declare for downstream consumption?** (§5 — `BaselinesConfig` 5-field exhaustive enumeration with types + defaults + consumer spec.)
3. **What 4 code patches does pkg-07 introduce in repo files?** (§7 — `configs/v4_config.py` dataclass addition, `envs/adapters/` new module tree, `baselines/` new module tree, `scripts/train_main.py` CLI extension. **Pkg-02 obs-mask is NOT here.**)
4. **What 5 anchors does pkg-08 reverse-consume from pkg-07?** (§8 — verbatim from pkg-08 README §"🔁".)
5. **How is drift between spec 08 and its sibling sources detected and fixed?** (§6 — drift detector regex + 25-anchor grep table + repair workflow.)

The payoff is concrete: pkg-08 reviewers do not need to read spec 01-07 of pkg-07 to verify the integration boundary is honest; they read spec 08, then cross-check spec 08 vs. sibling specs via the §6 grep table. Pkg-02 and pkg-05 reviewers do not need to read spec 04 / spec 05 / spec 06 to verify the downstream code-patch list is complete; they read §7. Future spec edits trigger drift via §6's regex + CI script `sdd/pkg-07-baselines/scripts/check_ref_matrix.ps1`.

---

## 2. `create_baseline` factory signature (mirror of spec 01 §2 + §3.2)

### 2.1 Locked signature

The factory is the **single entry point** for instantiating any baseline in pkg-07. Its signature is locked in spec 01 §3.2 and mirrored here verbatim:

```python
# hyper_mve/baselines/__init__.py (mirrored from spec 01 §3.2)

from typing import Union, TypeAlias
from hyper_mve.configs import V4Config

# BaselineLike Union type — the load-bearing return-type contract (Lock 1 below).
BaselineLike: TypeAlias = Union["BaselineModel", "ExternalBaselineRunner"]


def create_baseline(cfg: V4Config, variant: str) -> BaselineLike:
    """Unified 11-key factory (pkg-07; supersedes pkg-06 create_baseline_model).

    Args:
        cfg:     V4Config from preset. Consumes cfg.baselines.* 5 fields only (§5 below).
        variant: factory arg ∈ INTERNAL_REGISTRY ∪ EXTERNAL_REGISTRY (11 keys total).

    Returns:
        BaselineLike. INTERNAL_REGISTRY hit → BaselineModel (pkg-04 spec 02 7-API).
                      EXTERNAL_REGISTRY hit → ExternalBaselineRunner (Protocol).

    Raises:
        ValueError on "hyper" / "oracle_only" / "infer_only" (curriculum-override
            row, design §3.3; not factory-dispatched).
        ValueError on unknown variant.
        NotImplementedError on MARIE/GA stubs and MAMBA-if-not-sourced (spec 06).
    """
```

### 2.2 `BaselineLike Union` type (Lock 1 of cross-package consumption)

`BaselineLike = Union[BaselineModel, ExternalBaselineRunner]` is **the** load-bearing cross-spec type alias. It is consumed by:

- **pkg-08 spec 01 §2** (unified evaluator's `evaluate(runner: BaselineLike, env_fn, cfg) -> EvalReport`).
- **pkg-08 spec 05** (sweep harness's `for variant in REGISTRY: runner: BaselineLike = create_baseline(cfg, variant)`).
- **pkg-08 spec 08 §3** (mirror of `EvalReport` schema; the 32-field dataclass is structurally tied to `BaselineLike` having a `.evaluate()` method).
- **pkg-07 spec 03 §10.1** (internal `BaselineModel` is one branch of the union).
- **pkg-07 spec 04 §10** (external `ExternalBaselineRunner` is the other branch).

Drift in this alias name (`BaselineLike`) is the highest-priority anchor in §6.3.

### 2.3 Alias shim `create_baseline_model` with `DeprecationWarning`

Per design D2 + spec 01 §3.2 lines 193-201, the pkg-06 name `create_baseline_model(cfg, variant)` is preserved as an alias:

```python
def create_baseline_model(cfg: V4Config, variant: str) -> BaselineLike:
    """DEPRECATED: pkg-06 alias. Use create_baseline (pkg-07)."""
    warnings.warn(
        "create_baseline_model is renamed to create_baseline (pkg-07). "
        "This alias will be removed in the next release.",
        DeprecationWarning, stacklevel=2,
    )
    return create_baseline(cfg, variant)
```

The alias is preserved for **one release** window — pkg-05 spec 08 §3.1 line 136 still references the old name; the alias keeps that SDD-level reference stable without a synchronous Pkg-05 SDD edit. Any caller in pkg-08 (or in `scripts/train_main.py`) **must** use `create_baseline`; the alias is for backward-compat only and is enforced by spec 01 §7.6 test `test_create_baseline_model_alias_warns`.

### 2.4 What does not go through the factory

Three CLI strings (`hyper` / `oracle_only` / `infer_only`) do NOT enter the factory; they go through `HyperMuZeroModel(cfg) + curriculum_stage_1_end_frac override` per design §3.3 row 1-3. The factory raises `ValueError` on these strings to fail-fast misrouted callers (spec 01 §3.3 + §7.2). The four-cell category boundary is:

| CLI string | Dispatch path | Reason |
|------------|--------------|--------|
| `hyper` | `HyperMuZeroModel(cfg)` direct | A/B′/C primary; curriculum-override row |
| `oracle_only` | `HyperMuZeroModel(cfg)` + `curriculum_stage_1_end_frac=1.0` | regret upper bound (pkg-08 spec 02 §3 ceiling cache) |
| `infer_only` | `HyperMuZeroModel(cfg)` + `curriculum_stage_1_end_frac=0.0` | zero-shot belief inference probe |
| 11 registry keys | `create_baseline(cfg, factory_arg)` | All 5 internal + 3 Tier-1 + MAMBA + 2 stubs |

The split exists because `hyper`/`oracle_only`/`infer_only` share the same model class (`HyperMuZeroModel`) but differ in **trainer-side curriculum override**, not in model construction. A unified factory dispatch would force `HyperMuZeroModel` into the 11-key registry, breaking the structural assertion in spec 01 §2.1 that "REGISTRY enumerates baseline-variants, not curriculum-overrides".

---

## 3. REGISTRY 11-key + sub-registries (mirror of spec 01 §2.1)

### 3.1 `INTERNAL_REGISTRY` (5 keys) + `EXTERNAL_REGISTRY` (6 keys) + 5 复用约束 (pkg-06 spec 08 carryforward)

Per spec 01 §3.2 (verbatim):

```python
# hyper_mve/baselines/__init__.py (mirrored)

INTERNAL_REGISTRY: Mapping[str, Callable[[V4Config], BaselineModel]] = MappingProxyType({
    "input_wide":                InputWideBaselineModel,
    "input_deep":                InputDeepBaselineModel,
    "ma_muzero":                 MAMuZeroBaselineModel,
    "no_belief":                 NoBeliefBaselineModel,
    "rewardhead_explicit_type":  ExplicitTypeRewardBaselineModel,
})

EXTERNAL_REGISTRY: Mapping[str, Callable[[V4Config], ExternalBaselineRunner]] = MappingProxyType({
    "external_mappo":         MAPPOAlgorithm,
    "external_qmix":          QMIXAlgorithm,
    "external_ma_muzero_gh":  MAMuZeroGHAlgorithm,
    "external_mamba":         MAMBAAlgorithm,    # IS_SOURCED toggle → stub if False
    "external_marie":         MARIEStub,         # always NotImplementedError
    "external_ga":            GAStub,            # always NotImplementedError
})

REGISTRY: Mapping[str, Callable[[V4Config], BaselineLike]] = MappingProxyType(
    {**INTERNAL_REGISTRY, **EXTERNAL_REGISTRY}
)
```

Three properties locked here (sourced from spec 01 §4):

- **Disjoint namespaces**: `INTERNAL_REGISTRY.keys() & EXTERNAL_REGISTRY.keys() == set()` (spec 01 §4.1, test `test_two_namespaces_disjoint`).
- **`MappingProxyType` read-only**: each of three registries is wrapped, preventing accidental `REGISTRY["new_baseline"] = ...` at sweep-time (spec 01 §4.2).
- **Merged view consumed by pkg-08**: `REGISTRY` is the merged view; `for v in REGISTRY` covers all 11 keys without internal/external branching (spec 01 §4.3). Trainer-side dispatch (pkg-05 `create_trainer_for_baseline`) still branches on `variant in INTERNAL_REGISTRY` because internal needs `MuZeroTrainer` while external uses `runner.train()`.

The pkg-06 spec 08 §2 5 复用约束 table is **inherited verbatim** for the 5 internal variants:

```
1. 同一 MuZeroTrainer 类                  — no baseline-specific train_step
2. 同一 Worker 类                          — same epsilon-greedy + MVE planner collection
3. 同一 EpisodeReplayBuffer 类
4. 同一 compose_total_loss 函数            — policy/value/reward/projection 4-loss composer
5. 仅 model 类不同                         — the single variable across 5 internal variants
```

External runners (`MAPPOAlgorithm`, `QMIXAlgorithm`, etc.) do **not** subscribe to constraints 1-4 — they carry their own trainer + buffer + loss, per design D3 + spec 05 §3 / spec 06 §2-§4. The 5th constraint ("only model class differs") cleanly does not apply to external — it is replaced by the **披露式 fairness contract** (spec 07 §4.2 + §5 below). The 5 复用约束 self-test continues to live at `tests/baselines/internal/test_reuse.py::test_baseline_reuses_same_trainer_class` per pkg-06 spec 08 §2.1 (inheritance noted in spec 03 §10.2).

### 3.2 11-key dispatch matrix (variant → class → consumer)

| factory arg | class | base type | spec | consumer |
|-------------|-------|-----------|------|----------|
| `input_wide` | `InputWideBaselineModel` | `BaselineModel` | spec 03 §7.1 | `MuZeroTrainer` (pkg-05) |
| `input_deep` | `InputDeepBaselineModel` | `BaselineModel` | spec 03 §7.2 | `MuZeroTrainer` (pkg-05) |
| `ma_muzero` | `MAMuZeroBaselineModel` | `BaselineModel` | spec 03 §7.3 | `MuZeroTrainer` (pkg-05) |
| `no_belief` | `NoBeliefBaselineModel` | `BaselineModel` | spec 03 §7.4 | `MuZeroTrainer` (pkg-05) |
| `rewardhead_explicit_type` | `ExplicitTypeRewardBaselineModel` | `BaselineModel` | spec 03 §7.5 | `MuZeroTrainer` (pkg-05) |
| `external_mappo` | `MAPPOAlgorithm` | `ExternalBaselineRunner` | spec 05 §4 | `runner.train(cfg, env_fn, total_env_steps, lr, seed)` |
| `external_qmix` | `QMIXAlgorithm` | `ExternalBaselineRunner` | spec 06 §2 | `runner.train(...)` |
| `external_ma_muzero_gh` | `MAMuZeroGHAlgorithm` | `ExternalBaselineRunner` | spec 06 §3 | `runner.train(...)` |
| `external_mamba` | `MAMBAAlgorithm` if `IS_SOURCED` else stub | `ExternalBaselineRunner` | spec 06 §4 | conditional; stub `__init__` raises |
| `external_marie` | `MARIEStub` | (stub) | spec 06 §5 | `__init__` raises `NotImplementedError` |
| `external_ga` | `GAStub` | (stub) | spec 06 §5 | `__init__` raises `NotImplementedError` |

The stubs' `__init__` raises (factory itself does not raise on stub keys; the stub class's constructor is what propagates the error). This is the spec 01 §3.3 reject-semantics decision: the factory contains exactly **two `raise ValueError`** paths (hyper + curriculum-override + unknown-variant), no special-cased stub branches; the stubs are responsible for their own failure mode (spec 06 §5).

---

## 4. `evaluate(env_fn, c_grid, episodes) -> EvalReport` signature — the cross-spec consumer contract

### 4.1 Uniform signature on both `BaselineModel` and `ExternalBaselineRunner`

Both branches of `BaselineLike` expose an `.evaluate` method with the **identical signature**:

```python
def evaluate(
    self,
    env_fn: Callable[[], ResourceCommonsPettingZooEnv],
    c_grid: tuple[float, ...],
    episodes: int,
) -> EvalReport: ...
```

This signature is the load-bearing **cross-spec consumer contract** — it is what allows pkg-08 spec 01's unified evaluator (`evaluate(runner, env_fn, cfg) -> EvalReport`) to write `for variant in REGISTRY: report = runner.evaluate(env_fn, c_grid, episodes)` without `isinstance(runner, BaselineModel)` branches in the call site. The signature is locked in:

- **Internal (`BaselineModel`)**: pkg-07 spec 03 §3 (7-API base) + pkg-08 spec 01 §4.2 (`_evaluate_internal_hyper_or_baseline` implementation path; the unified evaluator wraps the in-process model eval).
- **External (`ExternalBaselineRunner`)**: pkg-07 spec 05 §4 + §8 (MAPPO) + spec 06 §2-§4 (QMIX/MA-MuZero-GH/MAMBA) + pkg-08 spec 01 §4.2 `_evaluate_external` (delegation pattern: unified evaluator calls `runner.evaluate(...)` and returns the report as-is, no schema rewrite).

### 4.2 Internal eval routing (pkg-08 spec 01 §4.2 `_evaluate_internal_hyper_or_baseline`)

When `runner: BaselineModel` (`hasattr(runner, "set_context_subjective")` is true), pkg-08 spec 01 §4.2 routes through `_evaluate_internal_hyper_or_baseline`. The path is:

1. `model = runner if hasattr(runner, "set_context_subjective") else runner.model`.
2. Invoke `training/evaluation.py:run_eval(model, cfg, global_step=0)` (the legacy in-training eval; pkg-08 spec 01 Lock 1 says **not modified**).
3. Loop the 7-value c-grid (`cfg.eval.zero_shot_test_c`) and aggregate into `EvalReport.return_per_c` via the `replace(cfg, eval=replace(cfg.eval, eval_c_grid=(c,)))` trick (no `run_eval` edit).
4. For the 2 extra planner modes (`planner_no_crn`, `planner_no_coord_desc`), dispatch via spec 03 §3 (pkg-08); these are NOT pkg-07's concern.
5. Compute regret via spec 02 §3 oracle ceiling cache lookup.
6. Populate `EvalReport` with the 32-field schema (pkg-08 spec 01 §3).

Pkg-07's contribution to this routing is the `BaselineModel` class (spec 03) — pkg-07 owns the API surface that pkg-08 spec 01 §4.2 wraps. Drift between pkg-07 spec 03 §3 7-API surface and pkg-08 spec 01 §4.2 expected `set_context_subjective` / `set_context_objective` calls is detected by spec 03 §9.1 `test_7api_signatures` parametrized over 5 internal variants (5 × 7 method checks = 35 cells).

### 4.3 External eval routing (pkg-08 spec 01 §4.2 `_evaluate_external`)

When `runner: ExternalBaselineRunner` (`hasattr(runner, "evaluate")` AND `not hasattr(runner, "set_context_subjective")`), pkg-08 spec 01 §4.2 routes through `_evaluate_external`:

```python
def _evaluate_external(runner, env_fn, cfg) -> EvalReport:
    return runner.evaluate(
        env_fn=env_fn,
        c_grid=tuple(cfg.eval.zero_shot_test_c),
        episodes=int(cfg.eval.evaluate_episodes),
    )
```

The unified evaluator does **not** rewrite the returned `EvalReport`; pkg-07's external runner is responsible for constructing a schema-conformant report. Spec 05 §8.1 + §8.2 (MAPPO) is the **reference implementation** of the field-population matrix; spec 06 §2.7 / §3.7 / §4.6 inherit it verbatim for QMIX / MA-MuZero-GH / MAMBA. The post-hoc aggregation (segment + bell-curve) the unified evaluator runs on the returned report when external runners leave them as `MappingProxyType({})` is governed by pkg-08 spec 01 §6.3 — pkg-07 specs are not responsible for it.

### 4.4 Field-population matrix for external runners (mirror of spec 05 §8.1 + spec 06 §2.7 / §3.7 / §4.6)

| `EvalReport` field group | Internal (`BaselineModel`) populates | External (`ExternalBaselineRunner`) populates | Source |
|--------------------------|--------------------------------------|------------------------------------------------|--------|
| Identity (6) | All 6 | All 6 (`eval_planner_mode="planner_full"` is the conventional setting) | spec 05 §8.1 |
| Headline (5) | All 5 | All 5 | spec 05 §8.1 |
| Per-c (3) | All 3 over 7-grid | All 3 over `c_grid` arg | spec 05 §8.1 |
| c-segment (2) | All 2 via spec 02 segment aggregation | `MappingProxyType({})` — unified evaluator post-aggregates | spec 05 §8.1 + pkg-08 spec 01 §6.3 |
| Bell-curve (2) | All 2 via spec 02 bell aggregation | `MappingProxyType({})` — unified evaluator post-aggregates | spec 05 §8.1 + pkg-08 spec 01 §6.3 |
| Regret (4) | All 4 via spec 02 §3 oracle ceiling cache | Empty / `0.0` — unified evaluator runs post-hoc regret pass | spec 05 §8.1 |
| Planner-prior gap (3) | All 3 (real values from `run_eval`) | `planner_prior_return_gap=0.0`; `direct_inference_return_mean=return_mean`; `planner_full_return_mean=return_mean` | spec 05 §8.1 |
| Diagnostics (3) | All 3 (`walltime_seconds`, `env_steps_evaluated`, `episodes_total`) | All 3 (same) | pkg-08 spec 01 §3.1 |
| Oracle-leak / info-gating flags (2) | `info_gating_strict = env_fn()._oracle_mode is False`; `set_context_subjective_oracle_leak` per spec 03 §5.1 | `info_gating_strict` same; `set_context_subjective_oracle_leak=False` (external never calls set_context_subjective) | pkg-08 spec 01 §3.1 canonical definition |
| Belief diagnostics (2) | `belief_c_mae` / `belief_c_calibration` populated for `hyper` only; `None` for 5 internal baselines | Both `None` (no belief head) | pkg-08 spec 01 §5.3 + spec 05 §8.1 |
| Schema version (1) | `"pkg08-spec01-v1"` | `"pkg08-spec01-v1"` | spec 05 §8.1 |

**Sum check** (cleanest framing — 10 payload buckets + 1 sentinel): 6 + 5 + 3 + 2 + 2 + 4 + 3 + 3 + 2 + 2 = **32 payload fields** + 1 `schema_version` sentinel = **33 total @dataclass fields**. The canonical "32-field headline" cited throughout pkg-07/pkg-08 SDD is the payload count (excludes the forward-migration sentinel `schema_version`). Spec 01 §3 line 99 uses an equivalent but differently-bucketed narration ("6 identity + 5 headline + 3 per-c + 2 c-segment + 2 bell-curve + 4 regret + 3 planner-prior + 3 diagnostics + 2 nullable belief + 1 schema_version sentinel + 1 oracle-leak flag = 32") which lumps `schema_version` into the bucket-sum-of-32 — both reach 32; this table uses the cleaner "payload-only" framing so the sum is internally consistent. The 32-field count is **locked** in pkg-08 spec 01 §3 (the @dataclass body — authoritative source; this table is a navigational aid; the 11-row table merges `info_gating_strict` + `set_context_subjective_oracle_leak` into one row labelled "Oracle-leak / info-gating flags (2)" for table compactness, but they remain 2 distinct @dataclass fields). Any drift in the schema (added/removed/renamed field) requires a **synchronous edit** of pkg-08 spec 01 §3 + pkg-08 spec 08 §3 + this §4.4 table + spec 05 §8.1 + spec 06 §2.7 / §3.7 / §4.6 — caught by drift detector §6.3 anchor `EvalReport 32 fields`.

---

## 5. `BaselinesConfig` 5-field exhaustive enumeration (the cfg lock)

### 5.1 Dataclass declaration (mirror of design §D10 + spec 01 §6.1)

```python
# hyper_mve/configs/v4_config.py (consumption-side declaration; Pkg-01 spec 05 sync)

from types import MappingProxyType
from typing import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BaselinesConfig:
    """5-field exhaustive cfg namespace (4 internal capacity knobs + 1 external LR-sweep grid).

    Per-impl tuning constants (MAPPO _DEFAULT_*, QMIX _DEFAULT_*, MA-MuZero-GH _DEFAULT_*)
    are NOT here — they live as module-level constants in spec 05 §11 / spec 06 §2.x / §3.x / §4.x.
    """

    # === Internal capacity knobs (4) — etc. equal-param bisect targets per spec 07 §2.4 ===
    internal_wide_hidden_dim: int = 512
    internal_deep_layers: int = 8
    internal_ma_muzero_share_pred_head: bool = True
    internal_explicit_type_branches: int = 2          # default = env.num_types (α/β)

    # === External LR sweep (1) — per-variant grid mapping ===
    external_lr_sweep_grid: Mapping[str, tuple[float, ...]] = field(
        default_factory=lambda: MappingProxyType({
            "external_mappo":         (1e-4, 3e-4, 1e-3),
            "external_qmix":          (1e-4, 3e-4, 1e-3),
            "external_ma_muzero_gh":  (1e-4, 3e-4, 1e-3),
            # external_mamba added at sourcing time iff IS_SOURCED is True (spec 06 §4)
            # external_marie / external_ga not present (permanent stubs do not sweep)
        })
    )
```

### 5.2 Field consumption table

| Field | Type | Default | Consumer spec | Spec § | Test |
|-------|------|---------|---------------|--------|------|
| `internal_wide_hidden_dim` | `int` | `512` | spec 03 (`InputWideBaselineModel._build_conditioning_subsystem`) | §7.1 | spec 01 §7.4 mutation test cell 1 |
| `internal_deep_layers` | `int` | `8` | spec 03 (`InputDeepBaselineModel._build_conditioning_subsystem`) | §7.2 | spec 01 §7.4 mutation test cell 2 |
| `internal_ma_muzero_share_pred_head` | `bool` | `True` | spec 03 (`MAMuZeroBaselineModel.__init__`) | §7.3 | spec 01 §7.4 mutation test cell 3 |
| `internal_explicit_type_branches` | `int` | `2` (= `env.num_types`) | spec 03 (`ExplicitTypeRewardBaselineModel._build_conditioning_subsystem`) | §7.5 | spec 01 §7.4 mutation test cell 4 |
| `external_lr_sweep_grid` | `MappingProxyType[str, tuple[float, ...]]` | 3-grid for each Tier-1 | spec 05 §9 (MAPPO) / spec 06 §2.8 / §3.8 / §4.7 (QMIX / MA-MuZero-GH / MAMBA) | various | spec 01 §7.4 mutation test cell 5 + spec 07 §3 sweep enumeration |

### 5.3 Defaults rationale (mirror of spec 07 §2.4 + §4.4)

The 4 internal capacity knobs' defaults are **starting points** for the spec 07 §2.4 5%/10% equal-param bisection. Implementation-time bisection adjusts these defaults to land conditioned-subsystem parameter delta inside [-5%, +5%] (PASS) or worst-case [-10%, +10%] (WARN). The defaults shipped in pkg-07 SDD are pre-bisection placeholders — they will likely be tuned at Phase A Day 1 of implementation. Drift between the declared default and the post-bisection value is acceptable because spec 07 owns the convergence criterion (delta-based), not spec 08 (which only locks the declaration site).

The external LR sweep grid default `(1e-4, 3e-4, 1e-3)` is a 1-decade span centered on the canonical PPO learning rate. C7-EXT-FAIR1 requires ≥3 LRs; the default satisfies the floor. Implementation may extend the tuple to 4-5 entries per variant if needed (pkg-08 spec 05 sweep harness enumerates whatever is in the mapping).

### 5.4 What is NOT in `BaselinesConfig` (per-impl tuning constants exclusion)

Per design §D10 last paragraph + spec 01 §6.4: **per-impl tuning constants live in module defaults, not on `cfg.baselines`**. Examples:

| Per-impl constant | Lives in | Excluded because |
|-------------------|----------|-------------------|
| MAPPO `_DEFAULT_SHARE_POLICY` / `_DEFAULT_GAMMA` / `_DEFAULT_K_EPOCHS` / … (20 defaults) | `hyper_mve/baselines/external/mappo.py` (spec 05 §11) | These are MAPPO-internal verbatim ports from lzj; not cross-spec contracts |
| QMIX `_DEFAULT_RNN_HIDDEN_DIM=64` / `_DEFAULT_MIXER_HIDDEN_DIM=32` / target update interval / … | `hyper_mve/baselines/external/qmix.py` (spec 06 §2.x) | Same; PyMARL-internal port |
| MA-MuZero-GH `_DEFAULT_MCTS_SIMULATIONS` / `_DEFAULT_NUM_UNROLL_STEPS` / … | `hyper_mve/baselines/external/ma_muzero_gh.py` (spec 06 §3.x) | muzero-general-internal port |
| MAMBA per-impl defaults (if sourced) | `hyper_mve/baselines/external/mamba.py` (spec 06 §4.x) | Conditional sourcing; impl-internal |
| Smoke gate budget `_DEFAULT_SMOKE_BUDGET_ENV_STEPS=20_000` | spec 05 §10.3 + spec 06 §6 | Test-side constant, not training-time tuning |

The rule of thumb: `cfg.baselines.*` is for **fields consumed by ≥ 2 specs OR by the sweep harness**. Single-spec internal parameters do not promote to cfg; they stay as module constants. This rule prevents `cfg.baselines.*` from accreting into a "all tunable parameters" dictionary, which would defeat the purpose of having a frozen 5-field cross-spec contract.

### 5.5 量词 canonical (mirror of spec 01 §2.3)

For grep stability across pkg-07/pkg-08/Ch6:

- **REGISTRY = 11 keys** (5 internal + 3 Tier-1 + MAMBA + 2 stubs).
- **CLI choices = 14 strings** (REGISTRY ∪ {`hyper`, `oracle_only`, `infer_only`}).
- **Methods main table = 9 columns** (hyper + 5 internal + 3 Tier-1). MAMBA-if-sourced is the 10th column. MARIE/GA stubs are NOT in the main table (design D9 epilogue).
- **`BaselinesConfig` = 5 fields** (this §5).
- **`EvalReport` = 32 fields** (pkg-08 spec 01 §3; pkg-07 spec 05 §8 and spec 06 §2.7 / §3.7 / §4.6 populate verbatim).
- **`EXTERNAL_REGISTRY` = 6 keys** (3 Tier-1 + MAMBA + 2 stubs).
- **`INTERNAL_REGISTRY` = 5 keys**.
- **§8 ref_matrix = 8-spec** (per-spec ↔ per-spec graph, NOT variant matrix; spec 01-08 of pkg-07).
- **pkg-07 → pkg-08 reverse-consumption anchors = 5** (§8 below + pkg-08 README §"🔁").
- **§7 downstream code patches = 4** (4 pkg-07-owned; Pkg-02 obs-mask is the 5th but belongs to pkg-08 spec 08).

Any drift in the bolded numbers triggers §6.3 grep failure.

---

## 6. Drift detector (the regex + verbatim grep targets)

### 6.1 Drift detector philosophy

Spec 08 is **derivative**: it mirrors contracts sourced from spec 01-07. The sibling spec is the truth; spec 08 is the one-document summary. Drift = sibling spec changed but spec 08 did not catch up.

The CI script `sdd/pkg-07-baselines/scripts/check_ref_matrix.ps1` (Day 8 of pkg-07 SDD calendar; README §"实施顺序" Phase 8) runs two checks:

- **Forward ref check**: every spec 01-07 references spec 08 only via the cross-references section (this is enforced by the existing pkg-07 README §"spec 间引用表" matrix). Spec 08 references all of spec 01-07 (the spec-08-row of the matrix has 7 entries).
- **Anchor consistency check**: every verbatim anchor listed in §6.3 below appears in BOTH spec 08 AND the cited sibling spec. The check uses a literal grep for the anchor string; mismatch = sibling edited, spec 08 did not catch up. The check produces `[FAIL]` + a diff of present-in-sibling-absent-in-spec-08 anchors.

The asymmetry (sibling does not cite spec 08; spec 08 cites sibling) prevents circular ref-chain pings and makes drift unambiguously assignable to one side: if the anchor is in the sibling but not in spec 08, spec 08 must update. There is no "spec 08 changed first, sibling needs updating" case — spec 08 has no contract content of its own to change first.

### 6.2 Regex check (the actual PowerShell pattern)

`check_ref_matrix.ps1` runs the following PowerShell regex against every spec file in `sdd/pkg-07-baselines/specs/`:

```powershell
$pat = '(?<![Pp]kg-\d{2} )spec\s+0([1-8])|\b0([1-8])-[a-z]'
#   - Bare "spec 0X"   but excludes "Pkg-NN spec 0X" / "pkg-NN spec 0X" via case-insensitive negative look-behind
#   - File name "0X-name"  spec-internal references in cross-ref sections
```

The regex matches two intra-package reference forms:
1. **Bare `spec 0X`** (e.g., `spec 05 §8.1`) — but excludes cross-package mentions like `pkg-04 spec 02` / `Pkg-04 spec 02` / `pkg-08 spec 01` / `Pkg-08 spec 01` via the **case-insensitive** negative look-behind `(?<![Pp]kg-\d{2} )`. The `[Pp]` character class handles both common casings observed in body text; if a third casing (`PKG-NN`) ever appears, expand to `[PpK]kg` or use the PowerShell `(?i)` mode for the whole pattern.
2. **Filename `0X-...md`** (e.g., `02-shared-backbones-internal.md`) — captures cross-ref-section file-link references.

The captured digit (group 1 or group 2) yields the cited spec number. The script tabulates per-file citations into `sdd/pkg-07-baselines/ref_matrix.csv` and compares against the **expected matrix** in README §"spec 间引用表". Spec 08's expected row is: cites spec 01, 02, 03, 04, 05, 06, 07 (all 7). Deviation = `[FAIL]`.

**Self-test on this file**: a smoke test for the regex correctness runs at Day 8 of the SDD calendar against this spec 08 file specifically — both casings should be excluded; only the bare intra-package `spec 0X` references should match.

### 6.3 List of verbatim grep targets (28 anchors)

The CI check additionally greps for the following verbatim strings; each must appear in **both** spec 08 AND its cited sibling spec.

| # | Anchor ID | Verbatim string | Sibling source |
|---|-----------|----------------|----------------|
| 1 | `spec-01-factory-signature` | `create_baseline(cfg: V4Config, variant: str) -> BaselineLike` | spec 01 §3.2 |
| 2 | `spec-01-registry-11-keys` | `INTERNAL_REGISTRY` (5 keys) and `EXTERNAL_REGISTRY` (6 keys) | spec 01 §3.2 |
| 3 | `spec-01-baseline-like-union` | `BaselineLike: TypeAlias = Union["BaselineModel", "ExternalBaselineRunner"]` | spec 01 §3.2 |
| 4 | `spec-01-alias-deprecation` | `DeprecationWarning(stacklevel=2)` on `create_baseline_model` | spec 01 §3.2 lines 193-201 |
| 5 | `spec-01-cli-to-factory-arg` | `cli_to_factory_arg(cli: str) -> str` derives factory arg from CLI | spec 01 §5.2 |
| 6 | `spec-02-count-conditioning-params` | `count_conditioning_params(model) -> int` SoT | spec 02 §2.2 |
| 7 | `spec-02-shared-backbone-prefixes` | `SHARED_BACKBONE_PREFIXES = ('rep_net', 'belief_net', 'tri_context_encoder')` | spec 02 §2.3 SB5 |
| 8 | `spec-02-c7-int-fair2-bitexact` | C7-INT-FAIR2 bit-exact backbone equal-param across 5 internal + hyper | spec 02 §5.1 |
| 9 | `spec-02-no-belief-keeps-full-belief-net` | no_belief still constructs full BeliefNet (information ablation, not parameter ablation) | spec 02 §3.2 |
| 10 | `spec-02-design-d4-anchor` | design D4 "5 internal variant 各自实例化 RepNet/BeliefNet/TriContextEncoder" | spec 02 §3.1 |
| 11 | `spec-03-7-api-surface` | 7-API method names: update_step / set_context_objective / set_context_subjective / encode / transition / predict_reward / predict | spec 03 §3.1 |
| 12 | `spec-03-self-info-strict` | `assert cap_i.shape[-1] == 4` (C7-INT-SELF1) | spec 03 §5.1 |
| 13 | `spec-03-belief-grad-gating-shared` | 5 internal variants share `BeliefGradGating(cfg.train.belief_grad_gating_steps)` | spec 03 §6.1 |
| 14 | `spec-03-stateful-last-subjective` | last `set_context_subjective` agent_id semantic (C7-INT-API2) | spec 03 §4.2 |
| 15 | `spec-04-n-parametric-adapter` | `self._N = self._env.N; self.agents = [f"agent_{i}" for i in range(self._N)]` | spec 04 §3 |
| 16 | `spec-04-two-flag-info-gate` | `oracle_mode=False AND eval_info_mode=False` default; 4 leak surfaces (`c_true`/`types`/`hotspot_centers`/`resource_state`) | spec 04 §7 |
| 17 | `spec-04-filter-info-marker-driven` | `_filter_info` reads `info["_oracle_fields"]` / `info["_eval_only_fields"]` from env-published markers | spec 04 §8 |
| 18 | `spec-04-ctde-legitimate-state` | LEGAL global state = `concat([obs_i for i in agents])` ± action one-hots ± `caps` (public) | spec 04 §7 footnote |
| 19 | `spec-04-env-fn-factory` | external runners construct env via `env_fn: Callable[[], ResourceCommonsPettingZooEnv]` only | spec 04 §10 |
| 20 | `spec-05-forbidden-info-keys` | `_FORBIDDEN_INFO_KEYS = frozenset({"c_true", "types", "resource_state", "hotspot_centers"})` | spec 05 §5.2 |
| 21 | `spec-05-evaluate-evalreport-mapping` | MAPPO `evaluate()` populates 32-field `EvalReport` per pkg-08 spec 01 §3 schema | spec 05 §8 |
| 22 | `spec-06-mamba-is-sourced-toggle` | `IS_SOURCED: Final[bool]` toggle; `MAMBAAlgorithm = _RealMAMBA if IS_SOURCED else _MAMBAStub` at import-time | spec 06 §4 + spec 01 §3.2 line 139 |
| 23 | `spec-06-marie-ga-stub-init-raises` | `MARIEStub.__init__` and `GAStub.__init__` each raise `NotImplementedError` on construction | spec 06 §5 |
| 24 | `spec-07-double-axis-fairness` | Internal = Axis A (bit-exact, spec 02) + Axis B (5%/10% double-threshold, spec 07 §2); External = disclosure-only | spec 07 §"Lock 1" + §"Lock 2" |
| 25 | `spec-07-lr-sweep-grid-source` | `cfg.baselines.external_lr_sweep_grid` is the SoT for LR sweep grid; ≥3 LR × ≥5 seeds; no hard-coded grid in runner | spec 07 §"Lock 3" |
| 26 | `pkg-08-eval-report-32-fields` | `@dataclass(frozen=True) EvalReport` 32-field schema (6+5+3+2+2+4+3+3+2+2+1 = 32) | pkg-08 spec 01 §3 + pkg-08 spec 08 §3 |
| 27 | `pkg-08-info-gating-strict-canonical` | `info_gating_strict = env_fn()._oracle_mode is False` (canonical definition) | pkg-08 spec 01 §3.1 + spec 05 §8.1 |
| 28 | `pkg-08-external-delegation` | `_evaluate_external` returns `runner.evaluate(env_fn, c_grid, episodes)` as-is, no schema rewrite | pkg-08 spec 01 §4.2 + §6 |

Anchors 1-5 are spec 01; 6-10 are spec 02; 11-14 are spec 03; 15-19 are spec 04; 20-21 are spec 05; 22-23 are spec 06; 24-25 are spec 07; 26-28 are cross-package pkg-08 anchors (subset of the 5 reverse-consumption mirrors; the other 2 reverse-consumption anchors are §8 below). Total = **28 verbatim grep targets**.

### 6.4 Drift handling workflow

The repair procedure when `check_ref_matrix.ps1` outputs `[FAIL]`:

1. **Identify the failing anchor** — the diff output names which anchor ID is missing or whose string mismatches.
2. **Locate the sibling source** — column "Sibling source" in §6.3 gives the canonical location.
3. **Determine the drift direction**:
   - If the anchor appears in the sibling but not in spec 08: **spec 08 must catch up**. The sibling was edited (intentionally or unintentionally); spec 08 mirrors the new wording.
   - If the anchor appears in spec 08 but not in the sibling: this is the **forbidden direction** (Lock 1: spec 08 is derivative, not source). Spec 08 has gone ahead of the sibling — revert the spec 08 edit and edit the sibling first instead.
4. **Synchronous fix** — once the drift direction is determined, edit both files in the same commit. Two separate commits (sibling first, then spec 08) is acceptable; spec 08 first then sibling is NOT (CI will fail in the interim).
5. **Re-run** `pwsh sdd/pkg-07-baselines/scripts/check_ref_matrix.ps1` until `[PASS]`.

The §6.3 anchor strings are intentionally verbose ("verbatim grep targets") so that copy-paste edits are reliable. A reviewer should not be inventing wording differences during the fix — they should copy the sibling spec's anchor string into spec 08 verbatim.

---

## 7. Four downstream code patches (Pkg-02 obs-mask is NOT pkg-07 territory)

Per Lock 2 + README §"输出清单" (`新增` block) + README §"修改" (existing-file modifications block), pkg-07 ships **four** code patches grouped as:

- **2 new modules** (§"新增" of README; sourced from spec 04 §2-§8 and spec 01/02/03/05/06): `envs/adapters/` (Patch 2 — sourced from spec 04 — adapter contract owner) + `baselines/` tree (Patch 3 — sourced from spec 01 §3 factory + spec 02 §2 backbones + spec 03 §2 model files + spec 05 §4 + spec 06 §2/§3/§4/§5).
- **2 existing-file modifications** (§"修改" of README; sourced from spec 01 §6 + §5.1 / spec 01 §3.3 + §5.1): `configs/v4_config.py` (Patch 1 — cfg.baselines dataclass) + `scripts/train_main.py` (Patch 4 — 6 CLI strings).

Two potential downstream patches are explicitly **NOT pkg-07's responsibility** and live in pkg-08's "下游补丁声明" list:

- `hyper_mve/envs/resource_commons/observations.py` +3 lines for `c_visible` masking — governed by **pkg-08 spec 02 §5.1 + pkg-08 spec 08 §5** (zero-shot c_hidden evaluation-time concern).
- `hyper_mve/planning/mve_planner.py` +1 line `mve_joint_enumerate` branch — governed by **pkg-08 spec 06 §5.1 + pkg-08 spec 08 §5** (Ablation 4 Joint cell training-time toggle).

The split is intentional: pkg-07 is "baselines + adapter (the algorithmic surface)", pkg-08 is "evaluation + c_hidden + ablation (the experimental harness)". A patch lives in pkg-07 iff its target file is on the algorithm-construction or factory-dispatch axis (cfg dataclass / adapter module / model classes / CLI variant registration); it lives in pkg-08 iff its target file modulates evaluation-time or ablation-time behaviour (obs-mask for c_hidden; planner-enumeration flag for Abl4 Joint cell).

### 7.1 Four-patch table

| Patch # | File path | Size | Tag | Declared in spec | Test gate |
|---------|-----------|------|-----|------------------|-----------|
| 1 | `hyper_mve/configs/v4_config.py` | +1 field on `V4Config` (`baselines: BaselinesConfig = field(default_factory=BaselinesConfig)`) + `BaselinesConfig` dataclass body (~25 lines) | `[config-additions]` | spec 01 §6 + this §5 | `tests/baselines/test_factory.py::test_cfg_baselines_5_fields_consumed_no_defaults` (spec 01 §7.4) + `test_v4config_has_baselines_subconfig` (§9.5 below) |
| 2 | `hyper_mve/envs/adapters/__init__.py` + `hyper_mve/envs/adapters/pettingzoo_wrapper.py` | NEW directory `envs/adapters/` (1 line `__init__.py`) + `pettingzoo_wrapper.py` ~200 lines (class body + `_filter_info` + `reset` / `step` + `observation_space` / `action_space`) | `[new-module]` | spec 04 §2-§8 | `tests/baselines/external/test_adapter_info_gating.py` (spec 04 §9 — 6 named test functions covering C7-EXT-ADPT1 + C7-EXT-ADPT2) |
| 3 | `hyper_mve/baselines/` directory tree | NEW directory tree: 1 `__init__.py` (~30 lines factory + REGISTRY) + 1 `_runner_protocol.py` (~50 lines protocol) + 1 `shared_backbones.py` (~150 lines factories + `count_conditioning_params`) + `internal/` subdir (5 files × ~150 lines each = ~750 lines) + `external/` subdir (1 `__init__.py` + 3 Tier-1 files × ~200 lines = ~600 lines + 2 stubs × ~10 lines = ~20 lines + MAMBA file conditional ~200 if sourced else ~10 lines stub) — total ~1800-2000 lines | `[new-module]` | spec 01 §3 (factory + REGISTRY) + spec 02 §2 (backbone factories) + spec 03 §2 (5 internal model files) + spec 05 §4 (MAPPO) + spec 06 §2 / §3 / §4 / §5 (QMIX / MA-MuZero-GH / MAMBA / 2 stubs) | `tests/baselines/test_factory.py::test_factory_dispatches_11_keys` (spec 01 §7.1) + 10 other sibling-spec tests (spec 02 §5.1 / spec 03 §9 / spec 05 §12 / spec 06 §12) |
| 4 | `hyper_mve/scripts/train_main.py` | +6 new `--variant` CLI strings: `external_mappo` / `external_qmix` / `external_ma_muzero_gh` / `external_mamba` / `external_marie` / `external_ga`. The 8 existing CLI strings (3 curriculum-override + 5 internal) remain unchanged. The `_build_cli_variant_choices()` derivation function (spec 01 §5.1) is added if not already present. The actual diff is +~20 lines (6 strings + helper function + import of `REGISTRY`). | `[downstream-cli]` | spec 01 §3.3 + spec 01 §5.1 | `tests/baselines/external/test_registry.py::test_train_main_variant_cli_round_trip` (§9.7 below) + `test_registry_keys_equal_cli_choices_and_factory_args` (spec 01 §7.5) |

### 7.2 Two pkg-08-territory patches explicitly NOT in this list

For clarity (and reviewer expectations): **two** potential downstream patches are NOT in this 4-patch list because they live in pkg-08:

**Excluded patch A — Pkg-02 obs-mask** (`hyper_mve/envs/resource_commons/observations.py` +3 lines for `c_visible` masking):
- **Declaration**: `sdd/pkg-08-eval-and-ablation/specs/02-zero-shot-and-c-hidden.md §5.1` (pkg-08 spec 02 owns the patch).
- **Aggregation**: `sdd/pkg-08-eval-and-ablation/specs/08-integration-contracts.md §5` (pkg-08 spec 08 aggregates into pkg-08's "下游补丁声明" list).
- Rationale: zero-shot c_hidden is an evaluation-time concern, not an algorithm-construction concern.

**Excluded patch B — Pkg-05 mve_joint_enumerate** (`hyper_mve/planning/mve_planner.py` +1 line `mve_joint_enumerate` branch):
- **Declaration**: `sdd/pkg-08-eval-and-ablation/specs/06-ablation-cli-and-cells.md §5.1` (pkg-08 spec 06 owns the patch).
- **Aggregation**: `sdd/pkg-08-eval-and-ablation/specs/08-integration-contracts.md §5` (pkg-08 spec 08 aggregates).
- Rationale: Ablation 4 Joint cell is an ablation-time concern (training-time enumeration toggle), governed by the ablation harness, not by pkg-07 baselines.

Pkg-07 spec 08 does not aggregate either patch, does not declare either patch, and does not test either patch — both are pkg-08's concern. The drift detector §6.3 anchor `pkg-02-obs-mask-and-pkg-05-mve-joint-enumerate-not-in-pkg-07` (informally; no entry in §6.3 table because absence-checks are vacuous) ensures spec 08 of pkg-07 does not accidentally absorb either patch. The pkg-07 README §"修改" block has been amended to match (lists only the 2 existing-file patches that pkg-07 owns: `train_main.py` CLI + `v4_config.py` dataclass).

### 7.3 Patch ordering for implementation

Implementation order (pkg-07 README §"实施期" Phase A-C):

1. **Patch 1 (cfg dataclass)** — Phase A Day 1. Without `cfg.baselines.*`, the 5 internal model classes' `__init__` raises `AttributeError`. Must land before any model class.
2. **Patch 3 (module tree)** — Phase A wk 1-2 (internal) + Phase B wk 2-3 (MAPPO + adapter import) + Phase C wk 3-4 (QMIX/MA-MuZero-GH) + Phase C' wk 4 (MAMBA sourcing or stub).
3. **Patch 2 (adapter)** — Phase B wk 2-3 (lands before MAPPO; spec 04 §10 lint requires it as the unique env construction site).
4. **Patch 4 (CLI extension)** — Phase B or later. Cannot land before Patch 3's `REGISTRY` is exported (because the helper function `_build_cli_variant_choices` imports `REGISTRY`).

Total estimated implementation lines: ~2000 (Patch 3) + ~200 (Patch 2) + ~25 (Patch 1) + ~20 (Patch 4) ≈ **2245 lines** of new code. Plus ~1500-2000 lines of test files under `tests/baselines/` (per README §"输出清单" tests subtree).

---

## 8. Pkg-08 reverse-consumption contracts (5 anchors)

Per Lock 3, this section mirrors **verbatim** the 5-anchor table from pkg-08 README §"🔁 pkg-07 → pkg-08 契约对账":

```
pkg-07 spec 01 §2.1  REGISTRY 11 keys                 ──→ pkg-08 spec 05 §3 sweep enumeration
pkg-07 spec 01 §2.3  量词 canonical (11 keys)          ──→ pkg-08 spec 05 §3 + spec 08 §5
pkg-07 spec 04       N-parametric adapter + 两 flag    ──→ pkg-08 spec 01 §4 external runner eval 通路
pkg-07 spec 08       evaluate() → EvalReport 签名      ──→ pkg-08 spec 01 §3 EvalReport schema + spec 08 §3
pkg-07 spec 08       BaselineLike Union type           ──→ pkg-08 spec 05 sweep harness type sig
```

### 8.1 Anchor-by-anchor consumption

**Anchor 1 — REGISTRY 11 keys → pkg-08 spec 05 §3 sweep enumeration**: pkg-08 spec 05 sweep harness iterates `for variant in REGISTRY: rows.append(SweepRow(variant=variant, ...))`. The 11-key cardinality is load-bearing because pkg-08 spec 05 sizes per-GPU semaphore + JSONL append-only `RunRegistry` budget against the cartesian product (11 variants × ≥5 seeds × 2 presets × LR-sweep-applicable rows). Drift in `REGISTRY` cardinality (e.g., pkg-07 adds a 12th key without pkg-08 sync) would cause sweep harness to silently miss the new variant in `--sweep-variants all` mode.

**Anchor 2 — 量词 canonical → pkg-08 spec 05 §3 + spec 08 §5**: §5.5 of this spec re-states the canonical 量词 table (11 REGISTRY / 14 CLI / 9 main table / 5 BaselinesConfig / 32 EvalReport / 6 EXTERNAL / 5 INTERNAL). Pkg-08 spec 05 (sweep harness) and pkg-08 spec 08 §5 (cfg field count) reverse-consume these numbers. Drift in any bolded number triggers §6.3 grep failure on both spec 08's. The grep target is the canonical wording itself.

**Anchor 3 — N-parametric adapter + 两 flag → pkg-08 spec 01 §4 external runner eval 通路**: pkg-08 spec 01 §4.2 `_evaluate_external` constructs `env_fn` with `oracle_mode=False, eval_info_mode=True` (where eval_info_mode=True is the unique caller permitted to flip it; pkg-07 spec 04 §10). The N-parametric `agent_0..agent_{N-1}` IDs are consumed by every external runner's eval loop (spec 05 §8.2, spec 06 §2.7 / §3.7 / §4.6). Drift in pkg-07 spec 04's `agent_{i}` naming or in the two-flag default `(False, False)` would break every external runner's eval loop simultaneously.

**Anchor 4 — `evaluate()` → `EvalReport` signature → pkg-08 spec 01 §3 schema + spec 08 §3**: the 32-field schema in pkg-08 spec 01 §3 is the contract; pkg-08 spec 08 §3 mirrors it byte-identically (drift detector pkg-08 spec 08 §6 enforces). Pkg-07 spec 05 §8.1 (MAPPO) and spec 06 §2.7 / §3.7 / §4.6 (QMIX / MA-MuZero-GH / MAMBA) reverse-consume the field-population matrix (§4.4 above is the mirror). Drift in any field name / type / default forces a synchronous edit of: pkg-08 spec 01 §3 + pkg-08 spec 08 §3 + this §4.4 table + spec 05 §8.1 + spec 06 (3 sections). Five-document drift is the highest cost in the entire pkg-07/pkg-08 SDD; this is why §3.1 of pkg-08 spec 01 owns the schema (mother-doc) and everywhere else mirrors.

**Anchor 5 — `BaselineLike Union` → pkg-08 spec 05 sweep harness type sig**: pkg-08 spec 05's sweep harness type-annotates the inner loop as `runner: BaselineLike = create_baseline(cfg, variant)`. The `BaselineLike` alias is what allows pkg-08 spec 01 unified evaluator to have a single dispatch signature for both internal and external runners; without the union type, pkg-08 spec 01 would need two parallel evaluator functions. Drift in the alias name (`BaselineLike` → `BaselineUnion` or `RunnerLike`) would require synchronous edit of pkg-08 spec 01 §2.1 + spec 05 + this §2.2 + spec 01 §3.2 of pkg-07.

### 8.2 Reverse drift handling

If a sibling in pkg-08 (e.g., pkg-08 spec 01 §3) edits the `EvalReport` schema **without** synchronous edit to pkg-07 spec 08 §4.4 (this spec) + spec 05 §8.1 + spec 06 §2.7 / §3.7 / §4.6, the drift is caught by **pkg-08's own §6 drift detector** (pkg-08 spec 08 §6). The pkg-07 spec 08 §6.3 grep table includes anchors 26-28 (cross-package pkg-08 anchors) as a redundancy — both packages independently verify the contract is mirrored. A two-sided drift detector is honest: either side's CI failure surfaces the same drift.

---

## 9. Test contract — 7 named tests

The test files locked here are **new** tests that verify spec 08's contract surface. They do not duplicate spec 01-07's sibling tests (which already exist under `tests/baselines/`); they verify the **integration boundary** that spec 08 mirrors.

### 9.1 `test_factory_dispatches_11_keys` (mirror of spec 01 §7.1)

Located at `tests/baselines/test_factory.py`. Spec 08 mirrors the test name for cross-document grep; the test body is owned by spec 01 §7.1. Asserted invariants:

- 5 internal variants → `BaselineModel` instance.
- 3 Tier-1 external → `ExternalBaselineRunner` instance.
- `external_mamba` → `MAMBAAlgorithm` if `IS_SOURCED` else `NotImplementedError`.
- `external_marie` / `external_ga` → `pytest.raises(NotImplementedError)`.

### 9.2 `test_factory_rejects_hyper_with_value_error` (mirror of spec 01 §7.2)

Located at `tests/baselines/test_factory.py`. Asserted invariants:

- `pytest.raises(ValueError, match="HyperMuZeroModel")` on `variant="hyper"`.
- `pytest.raises(ValueError, match="curriculum")` on `variant="oracle_only"` and `variant="infer_only"`.
- `pytest.raises(ValueError, match="Unknown")` on CLI-prefix mis-route (e.g., `variant="baseline_input_wide"` not converted).

### 9.3 `test_create_baseline_model_alias_warns_deprecation` (mirror of spec 01 §7.6)

Located at `tests/baselines/test_factory.py`. Asserted invariants:

- `with pytest.warns(DeprecationWarning, match="renamed to create_baseline"): create_baseline_model(cfg, "input_wide")`.
- The returned object is identical to what `create_baseline(cfg, "input_wide")` would return.
- `stacklevel=2` (the warning blames the caller, not the alias's `warnings.warn` line).

### 9.4 `test_evaluate_signature_uniform_across_baseline_types` (NEW — spec 08-specific)

Located at `tests/baselines/test_integration.py` (new file owned by spec 08). Parametrized over 8 instantiable variants (5 internal + 3 Tier-1; MAMBA omitted unless sourced; stubs omitted because they raise on `__init__`). Asserted invariants:

```python
@pytest.mark.parametrize("variant", [
    "input_wide", "input_deep", "ma_muzero", "no_belief", "rewardhead_explicit_type",
    "external_mappo", "external_qmix", "external_ma_muzero_gh",
])
def test_evaluate_signature_uniform(cfg, variant):
    runner = create_baseline(cfg, variant)
    import inspect
    sig = inspect.signature(runner.evaluate)
    # Three positional/keyword parameters: env_fn, c_grid, episodes
    params = list(sig.parameters)
    assert params[:3] == ["env_fn", "c_grid", "episodes"] or params[:4] == ["self", "env_fn", "c_grid", "episodes"]
    # Return type annotation is EvalReport (or "EvalReport" string under PEP-563)
    ret_ann = sig.return_annotation
    assert ret_ann is EvalReport or str(ret_ann).endswith("EvalReport")
```

This test verifies that pkg-07's `BaselineLike Union` is structurally honest — both branches expose `.evaluate` with the same shape. Drift in spec 03 §3 (internal `evaluate` signature) or spec 05 §4 (external `evaluate` signature) is caught here.

### 9.5 `test_baselines_config_has_5_fields_exact` (NEW — spec 08-specific)

Located at `tests/baselines/test_integration.py`. Asserted invariants:

```python
def test_baselines_config_has_5_fields_exact():
    from dataclasses import fields
    from hyper_mve.configs.v4_config import BaselinesConfig
    field_names = {f.name for f in fields(BaselinesConfig)}
    assert field_names == {
        "internal_wide_hidden_dim",
        "internal_deep_layers",
        "internal_ma_muzero_share_pred_head",
        "internal_explicit_type_branches",
        "external_lr_sweep_grid",
    }
    # Frozen dataclass invariant
    cfg = BaselinesConfig()
    with pytest.raises(Exception):  # FrozenInstanceError
        cfg.internal_wide_hidden_dim = 1024

def test_v4config_has_baselines_subconfig():
    from hyper_mve.configs.v4_config import V4Config, BaselinesConfig
    cfg = V4Config()  # default preset
    assert isinstance(cfg.baselines, BaselinesConfig)
```

Drift in §5.1 dataclass declaration (added 6th field, removed any field, renamed field) is caught here.

### 9.6 `test_drift_detector_finds_all_28_anchors` (NEW — spec 08-specific)

Located at `tests/baselines/test_integration.py`. Verifies the §6.3 grep table:

```python
def test_drift_detector_finds_all_28_anchors():
    """For every anchor in spec 08 §6.3, verify the verbatim string is present
    in BOTH spec 08 AND its cited sibling. Mismatch = drift."""
    import pathlib
    spec_08_text = pathlib.Path("sdd/pkg-07-baselines/specs/08-integration-contracts.md").read_text(encoding="utf-8")
    anchors = [
        # (anchor_id, sibling_path, verbatim_string_or_pattern)
        ("spec-01-factory-signature", "01-baseline-registry-and-cli.md", "create_baseline(cfg: V4Config, variant: str)"),
        ("spec-01-baseline-like-union", "01-baseline-registry-and-cli.md", "BaselineLike"),
        ("spec-02-count-conditioning-params", "02-shared-backbones-internal.md", "count_conditioning_params"),
        ("spec-02-shared-backbone-prefixes", "02-shared-backbones-internal.md", "SHARED_BACKBONE_PREFIXES"),
        ("spec-03-7-api-surface", "03-internal-variants.md", "set_context_subjective"),
        ("spec-03-self-info-strict", "03-internal-variants.md", "cap_i.shape[-1] == 4"),
        ("spec-04-n-parametric-adapter", "04-pettingzoo-adapter.md", "self._N = self._env.N"),
        ("spec-04-two-flag-info-gate", "04-pettingzoo-adapter.md", "oracle_mode"),
        ("spec-05-forbidden-info-keys", "05-external-mappo.md", "_FORBIDDEN_INFO_KEYS"),
        ("spec-06-mamba-is-sourced-toggle", "06-external-qmix-mamuzero-mamba.md", "IS_SOURCED"),
        ("spec-07-double-axis-fairness", "07-fairness-protocol.md", "Axis A"),
        # ... (full 28-anchor enumeration; abbreviated here for brevity)
    ]
    for anchor_id, sibling_filename, verbatim in anchors:
        sibling_text = pathlib.Path(f"sdd/pkg-07-baselines/specs/{sibling_filename}").read_text(encoding="utf-8")
        assert verbatim in spec_08_text, f"Anchor {anchor_id!r}: verbatim {verbatim!r} absent from spec 08"
        assert verbatim in sibling_text, f"Anchor {anchor_id!r}: verbatim {verbatim!r} absent from sibling {sibling_filename}"
```

This is a **meta-test** — it does not exercise any runtime code; it verifies the SDD document set is consistent. Run on Day 8 of the pkg-07 SDD calendar (Phase 8 alongside `check_ref_matrix.ps1`).

### 9.7 `test_train_main_variant_cli_round_trip` (NEW — spec 08-specific Patch 4 gate)

Located at `tests/baselines/external/test_registry.py`. Asserted invariants:

```python
def test_train_main_variant_cli_round_trip(cfg):
    """Patch 4 gate: train_main.py CLI choices include all 11 REGISTRY keys
    + 3 curriculum-override = 14 strings total."""
    from hyper_mve.scripts.train_main import _build_cli_variant_choices
    from hyper_mve.baselines import REGISTRY, cli_to_factory_arg

    cli_choices = _build_cli_variant_choices()
    assert len(cli_choices) == 14, "CLI = 14 (11 REGISTRY + 3 curriculum-override)"
    assert {"hyper", "oracle_only", "infer_only"} <= set(cli_choices)

    for cli in cli_choices:
        if cli in ("hyper", "oracle_only", "infer_only"):
            with pytest.raises(ValueError, match="curriculum-override"):
                cli_to_factory_arg(cli)
        else:
            arg = cli_to_factory_arg(cli)
            assert arg in REGISTRY

    # Reverse: every REGISTRY key reachable from one CLI string
    seen = {cli_to_factory_arg(c) for c in cli_choices if c not in {"hyper", "oracle_only", "infer_only"}}
    assert seen == set(REGISTRY), "REGISTRY ↔ CLI bijective on 11 in-registry rows"
```

Drift in Patch 4 (e.g., a 7th external CLI string added without REGISTRY update) is caught here.

### 9.8 Test file layout summary

```
tests/baselines/
├── test_factory.py                      # spec 01 §7 (tests 1, 2, 3 above mirrored) + spec 01-owned tests
├── test_integration.py                  # NEW: spec 08-specific (tests 4, 5, 6 above)
├── internal/                            # spec 02/03 tests (not owned by spec 08)
│   ├── test_shared_backbones.py         # spec 02 §5
│   ├── test_7api_conformance.py         # spec 03 §9.1
│   ├── test_stateful.py                 # spec 03 §9.2
│   ├── test_self_info.py                # spec 03 §9.3
│   ├── test_grad_gating.py              # spec 03 §9.4
│   ├── test_structural_markers.py       # spec 03 §9.5
│   ├── test_reuse.py                    # pkg-06 spec 08 §2.1 inheritance (5 复用约束)
│   └── test_param_fairness.py           # spec 07 §2 (5%/10% double-threshold)
└── external/
    ├── test_registry.py                 # spec 01 §7 + spec 08-owned test 7 above
    ├── test_adapter_info_gating.py      # spec 04 §9
    ├── test_mappo_smoke.py              # spec 05 §10
    ├── test_qmix_smoke.py               # spec 06 §6
    ├── test_ma_muzero_gh_smoke.py       # spec 06 §6
    ├── test_external_eval_contract.py   # spec 05 §12.3 (shared)
    ├── test_mappo_adapter_consumption.py # spec 05 §12.2
    ├── test_mappo_lr_sweep.py           # spec 05 §12.5
    └── test_stub_external_baselines.py  # spec 06 §5
```

Spec 08 owns **3 new test functions** (`test_evaluate_signature_uniform_across_baseline_types`, `test_baselines_config_has_5_fields_exact`, `test_drift_detector_finds_all_28_anchors`) in 1 new file (`test_integration.py`), and mirrors **4 sibling test names** (`test_factory_dispatches_11_keys`, `test_factory_rejects_hyper_with_value_error`, `test_create_baseline_model_alias_warns_deprecation`, `test_train_main_variant_cli_round_trip`) for the §9 contract enumeration. Total **7 named tests** — matching the locked count in the spec authoring requirements.

---

## 10. Cross-references

### 10.1 Upstream anchors (pkg-07 internal)

- **design.md §3.3** (CLI ↔ factory-arg ↔ model-class 14-row map) — §2.4 quadrant table + §5.5 量词 canonical mirrors design's row split.
- **design.md §3.4** (vendoring sources) — §3.2 dispatch matrix and spec 05/06 anchors trace back to design §3.4.
- **design.md §3.5** (N-parametric adapter + two-flag info gating + CTDE legitimacy) — §6.3 anchors 15-19 (spec 04 mirrors) trace to design §3.5.
- **design.md §4 D2** (`create_baseline(cfg, variant)` 工厂归属 + alias) — §2.1 + §2.3 mirror.
- **design.md §4 D3** (Internal vs External 两 namespace + union type) — §2.2 + §3.1 mirror.
- **design.md §4 D7** (Tier-1 external selection + adapter contract) — §3.2 dispatch matrix rows 6-9.
- **design.md §4 D8** (MAMBA sourcing protocol + fallback) — §6.3 anchor 22.
- **design.md §4 D9** (MARIE/GA stub strategy) — §6.3 anchor 23.
- **design.md §4 D10** (`cfg.baselines.*` 5-字段穷举) — §5 entire section mirrors.

### 10.2 Sibling spec anchors (pkg-07 spec 01-07)

- **spec 01 §3.2** (factory signature + REGISTRY) — §2.1 + §3.1 mirror; anchors 1-5 in §6.3.
- **spec 01 §5.2** (`cli_to_factory_arg`) — §2.4 + Patch 4 in §7.1 + §9.7.
- **spec 01 §6** (5 cfg.baselines fields) — §5 entire section mirror.
- **spec 01 §7** (5 named tests) — §9 mirrors 4 of 5.
- **spec 02 §2.2** (`count_conditioning_params` SoT) — §6.3 anchor 6.
- **spec 02 §3.1** (design D4 三联约束) — §6.3 anchor 10.
- **spec 02 §5.1** (C7-INT-FAIR2 bit-exact test) — §6.3 anchor 8.
- **spec 03 §3** (7-API surface) — §4.2 internal eval routing trace; §6.3 anchor 11.
- **spec 03 §5.1** (Self-Info strict cap_i.shape[-1] == 4) — §6.3 anchor 12.
- **spec 03 §6** (BeliefGradGating shared) — §6.3 anchor 13.
- **spec 03 §10** (BaselineModel + MuZeroTrainer 5 复用约束 inheritance) — §3.1 5 复用约束 inheritance trace.
- **spec 04 §3** (N-parametric agents) — §6.3 anchor 15.
- **spec 04 §7** (two-flag info gating + CTDE footnote) — §6.3 anchors 16, 18.
- **spec 04 §8** (`_filter_info` marker-driven) — §6.3 anchor 17.
- **spec 04 §10** (`env_fn` factory contract + lint) — §6.3 anchor 19 + §3.2 dispatch matrix consumer column.
- **spec 05 §4** (MAPPO `ExternalBaselineRunner` shape) — §4.4 field-population matrix mirror; §6.3 anchor 21.
- **spec 05 §5.2** (`_FORBIDDEN_INFO_KEYS`) — §6.3 anchor 20.
- **spec 05 §8** (`evaluate()` 32-field population) — §4.4 field-population matrix mirror.
- **spec 05 §9** (LR sweep contract) — §5.2 field consumer for `external_lr_sweep_grid`.
- **spec 06 §4** (MAMBA `IS_SOURCED` toggle) — §6.3 anchor 22.
- **spec 06 §5** (MARIE/GA stubs) — §6.3 anchor 23.
- **spec 07 §"Lock 1"** + §"Lock 2"** (double-axis fairness) — §6.3 anchor 24.
- **spec 07 §"Lock 3"** (LR sweep grid SoT) — §6.3 anchor 25.

### 10.3 Downstream pkg-08 anchors

- **pkg-08 spec 01 §3** (`EvalReport` 32-field schema mother-doc) — §4.4 mirror; §6.3 anchor 26.
- **pkg-08 spec 01 §3.1** (info_gating_strict canonical definition) — §4.4 mirror; §6.3 anchor 27.
- **pkg-08 spec 01 §4.2** (`_evaluate_internal_hyper_or_baseline` + `_evaluate_external`) — §4.2 + §4.3 mirror; §6.3 anchor 28.
- **pkg-08 spec 01 §6** (external delegation + post-hoc aggregation) — §4.3 + §4.4 c-segment/bell-curve mirror.
- **pkg-08 spec 05** (sweep harness consumes REGISTRY + `cfg.baselines.external_lr_sweep_grid`) — §8.1 anchor 1 + §8.1 anchor 5.
- **pkg-08 spec 07** (stats / disclosure markdown) — §5.5 量词 canonical row "Methods main table = 9 columns".
- **pkg-08 spec 08** (mirror of `EvalReport` + RunRegistry schema) — §6.3 anchor 26 + §8.2 reverse-drift handling.
- **pkg-08 README §"🔁"** (5-anchor reverse-consumption table) — §8 entire section mirrors verbatim.

### 10.4 Supersede anchor

- **pkg-06 spec 08** `08-integration-contracts.md` — §0.1 carryforward declaration; 5 复用约束 inherited as §3.1; `create_baseline_model` renamed per design D2.

### 10.5 Ground-truth anchors (repo files)

- `hyper_mve/scripts/train_main.py:45-52` `_DEFERRED_VARIANTS` — Patch 4 ground truth (the 6 of 14 CLI strings already in the repo; the remaining 8 added by Patch 4).
- `hyper_mve/configs/v4_config.py:23-37` — V4Config sub-config layout (Patch 1 lands here; `cfg.baselines` is added at this level, no `.v4` intermediate per design D10 last paragraph).
- `hyper_mve/envs/resource_commons/env.py:84` — `self.N: int = cfg.N` (N-parametric source consumed by spec 04 §3).
- `hyper_mve/envs/resource_commons/env.py:295-323` — `_build_info` schema markers `_oracle_fields` / `_eval_only_fields` consumed by spec 04 §8 `_filter_info`.

---

## 11. Anchors (verbatim grep targets for spec 08 §6 drift detector)

- `pkg-07 spec 08 §0.1: supersedes pkg-06 spec 08; 5 复用约束 inherited verbatim as §3.1`
- `pkg-07 spec 08 §2.1: create_baseline(cfg: V4Config, variant: str) -> BaselineLike — locked signature`
- `pkg-07 spec 08 §2.2: BaselineLike = Union[BaselineModel, ExternalBaselineRunner] — load-bearing type alias`
- `pkg-07 spec 08 §2.3: create_baseline_model alias with DeprecationWarning(stacklevel=2), one-release window`
- `pkg-07 spec 08 §2.4: 14 CLI strings = 11 REGISTRY + 3 curriculum-override (hyper / oracle_only / infer_only)`
- `pkg-07 spec 08 §3.1: INTERNAL_REGISTRY 5 keys + EXTERNAL_REGISTRY 6 keys + MappingProxyType read-only`
- `pkg-07 spec 08 §3.1: pkg-06 spec 08 §2 5 复用约束 inherited verbatim for 5 internal variants`
- `pkg-07 spec 08 §3.2: 11-key dispatch matrix (variant → class → consumer)`
- `pkg-07 spec 08 §4.1: .evaluate(env_fn, c_grid, episodes) -> EvalReport uniform signature on both BaselineLike branches`
- `pkg-07 spec 08 §4.4: external runner EvalReport field-population matrix mirrors spec 05 §8.1 + spec 06 §2.7 / §3.7 / §4.6`
- `pkg-07 spec 08 §5.1: BaselinesConfig @dataclass(frozen=True) 5-field exhaustive enumeration`
- `pkg-07 spec 08 §5.2: 5-field consumption table (internal_wide_hidden_dim / internal_deep_layers / internal_ma_muzero_share_pred_head / internal_explicit_type_branches / external_lr_sweep_grid)`
- `pkg-07 spec 08 §5.4: per-impl tuning constants live in module defaults (spec 05 §11 / spec 06 §2.x), NOT in cfg.baselines`
- `pkg-07 spec 08 §5.5: 量词 canonical (REGISTRY=11 / CLI=14 / methods main table=9 / BaselinesConfig=5 / EvalReport=32)`
- `pkg-07 spec 08 §6.2: PowerShell regex (?<!Pkg-\d{2} )spec\s+0([1-8])|\b0([1-8])-[a-z] for intra-package ref scan`
- `pkg-07 spec 08 §6.3: 28 verbatim grep anchors (spec 01 ×5 / spec 02 ×5 / spec 03 ×4 / spec 04 ×5 / spec 05 ×2 / spec 06 ×2 / spec 07 ×2 / pkg-08 ×3)`
- `pkg-07 spec 08 §6.4: drift handling — sibling-precedes-mirror; spec 08 catches up, never leads`
- `pkg-07 spec 08 §7.1: 4 downstream code patches (v4_config / adapters / baselines tree / train_main CLI) — Pkg-02 obs-mask NOT here`
- `pkg-07 spec 08 §7.2: Pkg-02 obs-mask explicitly excluded — owned by pkg-08 spec 02 §5.1 + pkg-08 spec 08 §5`
- `pkg-07 spec 08 §8: pkg-08 reverse-consumption 5 anchors mirror pkg-08 README §"🔁" verbatim`
- `pkg-07 spec 08 §9: 7 named tests (4 sibling mirrors + 3 spec-08-owned in test_integration.py)`

**End of spec 08 — integration-contracts.**
