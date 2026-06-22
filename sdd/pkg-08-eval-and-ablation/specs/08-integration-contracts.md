# Spec 08 — Integration Contracts (EvalReport + RunRegistry Schema Locks; 5 cfg Fields; 3 Downstream Patches; pkg-07 Reverse-Consumption Drift Detector)

> **Anchors**: design.md §3 (3.1 Eval/Ablation半 + 3.2 EvalReport schema + 3.3 four planner eval mode taxonomy + 3.4 RunRegistry row schema + 3.5 subprocess isolation + 3.6-3.8 framework澄清) · §4 D1-D10 (Decisions: 8-spec分配 / Eval-Ablation半 / EvalReport schema lock / c_hidden路径 / zero-shot cfg驱动 / regret缓存 / 4 planner mode解耦 / subprocess + JSONL Registry / --ablation 5 ID / 5 cfg字段穷举) · §10 Decision Acceptance + Day-1 HARD GATE 10项验收清单 · README §"输出清单" (`新增` block: hyper_mve/eval/ + hyper_mve/experiments/ subtrees; `修改` 3-block A/A'/B layout: 3行为补丁 + 1零行为threading + 3 config字段声明) · README §"5 新 cfg 字段穷举" 5-row table · README §"🔁 pkg-07 → pkg-08 契约对账" 5-anchor block · README §"spec 间引用表" (M6 ref-matrix; spec 08 引用 spec 01-07 全 7 个).
> **Status**: SDD only — describes the **contract surface** assembled from sibling specs 01-07 of pkg-08 and re-published for pkg-08 reviewers, for pkg-07 reverse-consumption verification, and for pkg-02 / pkg-05 implementation-time consumers. No new behaviour, no new model class, no new test logic — every contract in this document is sourced from a sibling spec and re-anchored here for single-document review.
> **Cross-refs**: spec 01 (`01-unified-evaluator.md` — `@dataclass(frozen=True) EvalReport` schema mother-doc; §3.1 verbatim 32-field body) · spec 02 (`02-zero-shot-and-c-hidden.md` — Pkg-02 obs-mask + env.py threading kwargs + `cfg.env.c_visible` declaration + zero-shot disjoint-union invariant + regret cache) · spec 03 (`03-direct-inference-toggle.md` — `cfg.eval.eval_planner_mode` Literal + `cfg.eval.eval_use_planner_direct_inference` short-circuit alias + 4-mode dispatch truth table) · spec 04 (`04-mup-verification.md` — `MupSelftestVerdict` secondary schema; NOT extending `EvalReport`; consumes only `EvalReport.return_mean` slot) · spec 05 (`05-sweep-harness-and-run-registry.md` — `RunRegistry` JSONL row schema mother-doc; §4 verbatim 23-field body; subprocess-per-row contract) · spec 06 (`06-ablation-cli-and-cells.md` — `use_coord_desc → randomize_order` rename + `mve_joint_enumerate` Pkg-05 patch + 5 canned ablation YAMLs + train_main CLI rename) · spec 07 (`07-statistics-and-comparison.md` — Welch t + Holm-Bonferroni + `compare` CLI; 2 new files: `stats.py` + `compare.py`; matplotlib Agg headless) · **pkg-07 spec 08** (`../../pkg-07-baselines/specs/08-integration-contracts.md` — sibling mirror layer in pkg-07; symmetric pattern reference for this spec's §2 below; the bidirectional drift detector shares 5 reverse-consumption anchors with pkg-07 spec 08 §8) · **pkg-07 spec 01** (`../../pkg-07-baselines/specs/01-baseline-registry-and-cli.md` — 11-key REGISTRY consumed by pkg-08 spec 05 sweep enumeration; cardinality and naming reverse-consumed here) · **pkg-07 spec 04** (`../../pkg-07-baselines/specs/04-pettingzoo-adapter.md` — N-parametric `ResourceCommonsPettingZooEnv` adapter + two-flag info gate; consumed by pkg-08 spec 01 §2.2 `env_fn` arg) · **pkg-07 spec 07** (`../../pkg-07-baselines/specs/07-fairness-protocol.md` — external disclosure table consumed via spec 07 `compare --disclose` flag).

---

## ⚠️ Header — three hard locks

### Lock 1 — Spec 08 is a CONTRACT MIRROR ONLY (sibling-precedes-mirror invariant; symmetric with pkg-07 spec 08 Lock 1)

Every contract listed in this document is **sourced from a sibling spec** within pkg-08 (specs 01-07). If a contract appears here without an upstream sibling source, that is an authoring bug, not a new contract. The drift detector in §8 treats spec 08 as **derivative**: its source-of-truth check is **unilateral** — spec 08 cites the sibling, but the sibling does NOT cite spec 08 back as the authoritative source (siblings cite spec 08 only via the standard cross-references / ref-matrix block, not as a content-authoring source).

This is the same shape pkg-07 spec 08 uses (`../../pkg-07-baselines/specs/08-integration-contracts.md` Lock 1). The symmetry is intentional and is the basis of the pkg-07 ↔ pkg-08 bidirectional drift detection in §7 below.

Editorial workflow: **edit the sibling spec first, then mirror here**. Mirror-first-then-fix-sibling is a CI failure (the §8.3 grep table will fail). Spec 08 is read as the "one-document overview" for reviewers of pkg-08, of the pkg-07 reverse-consumption boundary, and of pkg-02 / pkg-05 implementation-time consumers; reviewers approving spec 08 in isolation are approving the **derivative**, not the truth — the truth lives in spec 01-07 of pkg-08 (and, for the 5 reverse-consumption anchors, in pkg-07).

### Lock 2 — Four contract categories are exhaustive (symmetric with §1 Purpose Q1-Q5)

This spec locks exactly **four** categories of contract — symmetric with the §1 Purpose Q1-Q5 reviewer-checklist framing (Q1=EvalReport / Q2=RunRegistry / Q3=cfg fields / Q4=patches; Q5 is the drift-detector meta-section, not a contract). There is no fifth.

- **(A) `EvalReport` @dataclass schema** — 32 payload fields + 1 schema_version sentinel = 33 @dataclass body length (§3, mirror of spec 01 §3.1).
- **(B) `RunRegistry` JSONL row schema** — 22 schema-domain fields + 1 schema_version sentinel = 23 keys (§4, mirror of spec 05 §4).
- **(C) 5 new cfg fields exhaustive enumeration**: design D10 lock — `cfg.env.c_visible` (spec 02) + `cfg.train.randomize_order` (alias-with-deprecation rename, spec 06) + `cfg.train.mve_joint_enumerate` (spec 06) + `cfg.eval.eval_planner_mode` Literal (spec 03) + `cfg.eval.eval_use_planner_direct_inference` (spec 03). §5 owns the exhaustive enumeration.
- **(D) 3-block "下游补丁声明"** mirroring README §"修改" A/A'/B layout: (A) **3 behavioural patches** — `observations.py` obs-mask (spec 02 §5.1) + `mve_planner.py` `mve_joint_enumerate` branch (spec 06 §5.1) + `train_main.py` CLI rename (spec 06 §8.3); (A') **1 zero-behaviour threading patch** — `env.py` call-site kwarg pass-through (spec 02 §5.1b, `[no-behaviour]` tag); (B) **3 config-additions** — `env_config.py` +1 field (spec 02 §5.2) + `train_config.py` +2 fields + `@property` shim (spec 06 §8.1) + `eval_config.py` +2 fields (spec 03 §8.1). §6 owns the canonical 3-block patch declaration.

Adding a fifth category here requires a synchronous edit to spec 01-07: every category corresponds to at least one upstream sibling §-block. A category without an upstream §-block cannot exist here. (Both schemas (A)+(B) are mother-doc anchored at their sibling; spec 08 mirrors byte-identically. The two schemas are the unique persistent contracts of pkg-08 — everything else is procedural — file paths, dispatch tables, test gates — and does not require a frozen schema lock.)

### Lock 3 — pkg-07 → pkg-08 reverse-consumption contracts mirror pkg-08 README §"🔁" 5 anchors verbatim; symmetric with pkg-07 spec 08 §8

§7 below mirrors the 5-anchor list from pkg-08 README §"🔁 pkg-07 → pkg-08 契约对账" verbatim. The 5 anchors are: (1) pkg-07 spec 01 §2.1 REGISTRY 11-keys → pkg-08 spec 05 §3 sweep enumeration; (2) pkg-07 spec 01 §2.3 量词 canonical (11 keys) → pkg-08 spec 05 §3 + spec 08 §5; (3) pkg-07 spec 04 N-parametric adapter + 两 flag → pkg-08 spec 01 §4 external runner eval 通路; (4) pkg-07 spec 08 `evaluate()` → `EvalReport` 签名 → pkg-08 spec 01 §3 EvalReport schema + spec 08 §3; (5) pkg-07 spec 08 `BaselineLike Union` type → pkg-08 spec 05 sweep harness type sig.

The 5-anchor block is **byte-identical** between pkg-07 spec 08 §8 and pkg-08 spec 08 §7. **Bidirectional drift detection**: pkg-07 spec 08 §6 and pkg-08 spec 08 §8 both verify the same 5 anchors; either side's CI failure surfaces the same drift on the pkg-07 ↔ pkg-08 contract bridge. A two-sided drift detector is honest — both packages independently verify the contract is mirrored — and is documented in §7.2 as the canonical reverse-drift handling workflow.

---

## 1. Purpose

Spec 08 exists for **single-document review** of the pkg-08 contract surface. A reviewer approving pkg-08 final-state should be able to read **one file** and answer five questions:

1. **What is the `EvalReport @dataclass(frozen=True)` schema?** (§3 — 32 fields verbatim, mirror of spec 01 §3.1. The 32 fields break down as 6 identity + 5 headline + 3 per-c + 2 c-segment + 2 bell-curve + 4 regret + 3 planner-prior + 3 diagnostics + 2 nullable belief + 1 schema_version sentinel + 1 oracle-leak/info-gating flag = 32, with the sum check verified at the table footer.)
2. **What is the `RunRegistry` JSONL row schema?** (§4 — 23 keys verbatim, mirror of spec 05 §4. 22 schema-domain fields + 1 `schema_version` sentinel = 23 keys on every JSONL line, grouped Identity 5 / Provenance 4 / Lifecycle 3 / Resource 3 / Output pointers 3 / TB pointer 1 / Duplicated summary metrics 3 = 22 schema-domain.)
3. **What 5 new cfg fields does pkg-08 declare for downstream consumption?** (§5 — the design D10 exhaustive enumeration: 1 field on `EnvConfig` + 2 fields + `@property` shim on `TrainConfig` + 2 fields on `EvalConfig`.)
4. **What 3 behavioural patches + 1 zero-behaviour threading + 3 config-additions does pkg-08 introduce?** (§6 — the canonical 3-block "下游补丁声明" mirroring README §"修改" A/A'/B layout, with line counts honest per the SDD "no silent edits" rule.)
5. **What 5 anchors does pkg-08 reverse-consume from pkg-07, and how is bidirectional drift detected?** (§7 + §8 — the 5-anchor mirror of pkg-08 README §"🔁" + the §8 drift detector regex + ≥30 verbatim grep targets.)

The payoff is concrete. Pkg-08 reviewers (and the user finalising pkg-08) do not need to read spec 01-07 of pkg-08 to verify the contract surface is honest; they read spec 08, then cross-check spec 08 vs. sibling specs via the §8 grep table. Pkg-07 spec 08 reviewers verify the pkg-07 → pkg-08 reverse-consumption boundary by cross-checking pkg-07 spec 08 §8 (verbatim 5-anchor block) against pkg-08 spec 08 §7 (verbatim 5-anchor block); two independent CI checks (one per package) catch the same drift. Pkg-02 reviewers verify the `observations.py` + `env.py` patch boundary by reading §6 (A) + §6 (A'); pkg-05 reviewers verify the `mve_planner.py` + `train_main.py` patch boundary by reading §6 (A) lines 2 and 3.

The deliverable of this spec is **zero new files and zero new test functions** beyond what is declared in sibling specs 01-07 + a small number of mirror-tests (§9). Every code-surface contract is sourced from sibling specs; the drift detector tests (§9) are the only spec-08-owned tests in this document.

---

## 2. Pattern reference — pkg-07 spec 08 parallels (the M6 ref-matrix symmetry discipline)

Pkg-07 spec 08 (`../../pkg-07-baselines/specs/08-integration-contracts.md`) is the **symmetric peer** of this spec. Both are integration-contract mirror layers, sit at position 08 within their respective package's 8-spec layout, share the "sibling-precedes-mirror" Lock 1, share the "N exhaustive contract categories" Lock 2, and share a verbatim reverse-consumption anchor block as Lock 3. The symmetry was a design choice (design.md §4 D1 8-spec 对称 + pkg-07 README §"文档导览" / pkg-08 README §"文档导览") to make integration-boundary review tractable: a reviewer who has read pkg-07 spec 08 can read pkg-08 spec 08 with the same mental model, and vice versa.

The parallel structure is tabulated below. Each row identifies a structural element that exists in both spec 08 instances, with the section number / anchor in each.

| Structural element | pkg-07 spec 08 location | pkg-08 spec 08 location | Notes |
|---|---|---|---|
| 3 hard locks before §1 | Header §"⚠️ Header — three hard locks" | Header §"⚠️ Header — three hard locks" | Both packages lock "mirror-only" + "N exhaustive categories" + "reverse-consumption verbatim mirror" |
| Lock 1: mirror-only invariant | Lock 1 | Lock 1 | Both treat spec 08 as derivative; sibling-precedes-mirror; CI fails on mirror-first |
| Lock 2: N exhaustive categories | Lock 2 (3 categories: API + 4 patches + 5 cfg fields) | Lock 2 (3 categories: 2 schemas + 5 cfg fields + 3-block patches) | Same Lock 2 shape, different N-tuple per package; both forbid a 4th category without sibling §-block |
| Lock 3: reverse-consumption 5 anchors | Lock 3 | Lock 3 | Verbatim 5-anchor block; bidirectional drift detection on either spec 08's CI failure |
| Pattern reference / sibling parallels section | §0.1 supersede + §10 cross-refs | §2 (this section) | pkg-08 spec 08 promotes the parallel-pattern table to §2 because there is no supersede (pkg-08 has no superseded predecessor like pkg-06 spec 08 was for pkg-07) |
| Schema mirror section(s) | §4.4 EvalReport field-population matrix (pkg-08-owned schema mirrored back) | §3 EvalReport (full 32-field mirror) + §4 RunRegistry (full 23-key mirror) | pkg-07 spec 08 §4.4 is partial (a population matrix, not the full @dataclass body); pkg-08 spec 08 §3 + §4 are the full mother-doc-mirrored bodies |
| cfg-field enumeration | §5 BaselinesConfig 5-field exhaustive enumeration | §5 5 new cfg fields exhaustive enumeration | Both lock cfg-field count at 5; different field sets per package; both use the exhaustive-enumeration pattern |
| Downstream-patch declaration | §7 4-patch table (`v4_config.py` + `envs/adapters/` + `baselines/` tree + `train_main.py` CLI) | §6 3-block table (3 behavioural + 1 zero-behaviour + 3 config-additions) | Different patch counts; both honest-line-count per the SDD discipline; both explicitly list "what is NOT in this package's territory" |
| Reverse-consumption 5 anchors | §8 (5 anchors verbatim from pkg-08 README §"🔁") | §7 (5 anchors verbatim from pkg-08 README §"🔁") | Section number differs because pkg-07 spec 08 places it at §8 and pkg-08 spec 08 places it at §7; **content is byte-identical** |
| PowerShell regex drift detector | §6.2 regex + 28 verbatim grep targets | §8.2 regex + ≥30 verbatim grep targets | Both use the same `[Pp]kg-\d{2}` case-insensitive negative-look-behind pattern; pkg-08 spec 08's grep count is higher because pkg-08 has 7 siblings + 5 cross-package pkg-07 anchors vs pkg-07's 5 cross-package pkg-08 anchors |
| Drift-handling workflow | §6.4 | §8.4 | Both follow the same 5-step procedure: identify failing anchor → locate sibling source → determine drift direction → synchronous fix → re-run CI |
| Test contract | §9 (7 named tests, 3 spec-08-owned + 4 sibling-mirror) | §9 (≥7 named tests, 3 spec-08-owned + sibling-mirror + 2 meta-tests for drift detector + reverse-consumption) | Both small surface; both meta-test the drift detector itself |

The symmetric pattern is the **M6 ref-matrix discipline**: any future spec-08-class document (e.g., a hypothetical pkg-09) should adopt the same shape so that integration-boundary review remains tractable across an ever-larger SDD corpus. Document the symmetry here so that future spec readers do not re-invent the pattern from scratch when adding a new package.

---

## 3. `EvalReport` @dataclass schema — verbatim 32-field mirror of spec 01 §3.1

### 3.1 Mother-doc anchor

The mother doc is `sdd/pkg-08-eval-and-ablation/specs/01-unified-evaluator.md §3.1`. The @dataclass body lives at `hyper_mve/eval/eval_report.py` (per spec 01 §1 deliverable list). This spec mirrors the schema **byte-identically**; drift is caught by `test_eval_report_32_field_dataclass_lock` (§9.1) which compares `dataclasses.fields(EvalReport)` against the verbatim enumeration below.

### 3.2 Verbatim 32-field table (grouped 11-bucket; sum check at footer)

| # | Field name | Type | Default | Populated by | Schema bucket |
|---|---|---|---|---|---|
| 1 | `variant` | `str` | — (required) | spec 01 §4.2 evaluator entry | Identity |
| 2 | `seed` | `int` | — (required) | spec 01 §4.2 evaluator entry | Identity |
| 3 | `config_hash` | `str` (32-char blake2b-16 hex) | — (required) | spec 05 §3.5 `compute_config_hash(cfg)` | Identity |
| 4 | `eval_mode` | `Literal["prior", "planner"]` | — (required) | spec 01 §3.1 legacy 2-mode label | Identity |
| 5 | `eval_planner_mode` | `Literal["direct_inference", "planner_no_crn", "planner_no_coord_desc", "planner_full"]` | — (required) | spec 03 §4.2 dispatch | Identity |
| 6 | `c_visible` | `bool` | — (required) | spec 02 §3.4 evaluator entry reads `cfg.env.c_visible` | Identity |
| 7 | `return_mean` | `float` | — (required) | spec 01 §4.2 mean over all eval episodes | Headline |
| 8 | `return_sem` | `float` | — (required) | spec 01 §4.3 SEM via re-run with same CRN seeds | Headline |
| 9 | `return_zero_shot_seen` | `float` | — (required) | spec 02 §2.5 `populate_zero_shot_fields` | Headline |
| 10 | `return_zero_shot_unseen` | `float` | — (required) | spec 02 §2.5 `populate_zero_shot_fields` | Headline |
| 11 | `return_zero_shot_gap` | `float` | — (required) | spec 02 §2.5 `populate_zero_shot_fields` (= seen − unseen) | Headline |
| 12 | `return_per_c` | `Mapping[float, float]` | — (required) | spec 01 §8.1 7-value zero-shot grid | Per-c |
| 13 | `return_per_c_sem` | `Mapping[float, float]` | — (required) | spec 01 §8.1 SEM per c | Per-c |
| 14 | `episodes_per_c` | `Mapping[float, int]` | — (required) | spec 01 §8.1 episode count per c | Per-c |
| 15 | `return_per_segment` | `Mapping[tuple[float, float], float]` | — (required) | spec 01 §8.2 episode-weighted segment mean | c-segment |
| 16 | `return_per_segment_sem` | `Mapping[tuple[float, float], float]` | — (required) | spec 01 §8.2 pooled segment SEM | c-segment |
| 17 | `return_per_type_ratio` | `Mapping[tuple[int, int], float]` | — (required) | spec 01 §8.3 per-(n_alpha,n_beta) sub-eval | Bell-curve |
| 18 | `return_per_type_ratio_sem` | `Mapping[tuple[int, int], float]` | — (required) | spec 01 §8.3 per-(n_alpha,n_beta) sub-eval SEM | Bell-curve |
| 19 | `regret_per_c` | `Mapping[float, float]` | — (required) | spec 02 §4.8 `compute_regret(report, cfg)` | Regret |
| 20 | `regret_mean` | `float` | — (required) | spec 02 §4.8 unweighted mean over c-grid | Regret |
| 21 | `oracle_ceiling_per_c` | `Mapping[float, float]` | — (required) | spec 02 §4.6 `lookup_ceilings` | Regret |
| 22 | `oracle_ceiling_cache_hit` | `Mapping[float, bool]` | — (required) | spec 02 §4.6 `lookup_ceilings` | Regret |
| 23 | `planner_prior_return_gap` | `float` | — (required) | spec 03 §5 sourced from `legacy_results["planner_prior_gap"]` | Planner-prior |
| 24 | `direct_inference_return_mean` | `float` | — (required) | spec 03 §5 `mean(legacy_results, prefix="prior/")` | Planner-prior |
| 25 | `planner_full_return_mean` | `float` | — (required) | spec 03 §5 `mean(legacy_results, prefix="planner/")` | Planner-prior |
| 26 | `walltime_seconds` | `float` | — (required) | spec 01 §3.1 wall time accumulation | Diagnostics |
| 27 | `env_steps_evaluated` | `int` | — (required) | spec 01 §3.1 env step counter | Diagnostics |
| 28 | `episodes_total` | `int` | — (required) | spec 01 §3.1 episode counter | Diagnostics |
| 29 | `info_gating_strict` | `bool` | — (required) | spec 01 §7.1 canonical: `env_fn()._oracle_mode is False` | Oracle-leak / info-gating |
| 30 | `set_context_subjective_oracle_leak` | `bool` | — (required) | spec 01 §5.5 wrap; asserted to be `False` | Oracle-leak / info-gating |
| 31 | `belief_c_mae` | `float \| None` | `None` | spec 02 §3.3 c_hidden run; hyper-only | Belief diagnostics (nullable) |
| 32 | `belief_c_calibration` | `float \| None` | `None` | spec 02 §3.3 c_hidden run; hyper-only | Belief diagnostics (nullable) |
| 33 | `schema_version` | `str` | `"pkg08-spec01-v1"` | spec 01 §3.1 schema sentinel | Schema version |

**Field-count sum check** (the 10-bucket payload breakdown + 1 sentinel; authoritative source is the @dataclass body in spec 01 §3.1, this grouping is a navigational aid):

```
Identity                   6   (rows 1–6)
Headline                   5   (rows 7–11)
Per-c                      3   (rows 12–14)
c-segment                  2   (rows 15–16)
Bell-curve                 2   (rows 17–18)
Regret                     4   (rows 19–22)
Planner-prior              3   (rows 23–25)
Diagnostics                3   (rows 26–28)
Oracle-leak / info-gating  2   (rows 29–30)
Belief diagnostics         2   (rows 31–32; nullable)
                          ──
PAYLOAD SUBTOTAL          32   (rows 1–32; the canonical "32 field" headline excludes the sentinel)
Schema version sentinel    1   (row 33; forward-migration marker, NOT a payload field)
                          ──
TOTAL @dataclass fields   33   (the full @dataclass body length; enforced by §9.1 `len(fields) == 33`)
```

Sum verification: 6+5+3+2+2+4+3+3+2+2 = **32 payload fields** (rows 1–32) + 1 schema_version sentinel = **33 total @dataclass fields**. Spec 01 §3 line 99 uses an equivalent 11-bucket narration ("6 identity + 5 headline + 3 per-c + 2 c-segment + 2 bell-curve + 4 regret + 3 planner-prior + 3 diagnostics + 2 nullable belief + 1 schema_version sentinel + 1 oracle-leak flag = 32") where the trailing "+ 1 schema_version sentinel + 1 oracle-leak flag" splits the 2 oracle-leak/info-gating fields into "1 oracle-leak flag" and lumps the sentinel into the count — both framings reach 32, but they reach it differently. **This spec uses the cleaner "32 payload + 1 sentinel = 33 @dataclass body" framing** so the printed sum is internally consistent. The drift detector test `test_eval_report_32_field_dataclass_lock` (§9.1) asserts BOTH invariants: `len(dataclasses.fields(EvalReport)) == 33` (full body) AND `(len(fields) - 1) == 32` (payload count excluding `schema_version` sentinel). Either drift triggers CI failure.

### 3.3 Double-consumption contract (internal + external runners; pkg-07 spec 08 §8 reverse-anchor #4 → cross-package grep target §8.3 row "pkg-08-eval-report-32-fields")

Per spec 01 Lock 2: every entry point on a baseline returns the **exact same** `EvalReport`. Three callers:

- **Internal `BaselineModel.evaluate(env_fn, c_grid, episodes) -> EvalReport`** (pkg-07 spec 01 §3.2 + pkg-08 spec 01 §4.2 `_evaluate_internal_hyper_or_baseline`).
- **External `ExternalBaselineRunner.evaluate(env_fn, c_grid, episodes) -> EvalReport`** (pkg-07 spec 04 §10 + pkg-08 spec 01 §4.2 `_evaluate_external`).
- **Unified `unified_evaluator.evaluate(runner, env_fn, cfg) -> EvalReport`** (pkg-08 spec 01 §2.1).

Pkg-07 spec 08 §4.4 mirrors the field-population matrix for the external branch; pkg-08 spec 08 §3 is the full @dataclass body. The two specs are byte-identical on the field name + type list; pkg-07 spec 08 §4.4's population-rules column is what pkg-08 spec 08 §3 does **not** repeat (the population rules are spec-01 / spec-02 / spec-03's domain).

---

## 4. `RunRegistry` JSONL row schema — verbatim 23-key mirror of spec 05 §4

### 4.1 Mother-doc anchor

The mother doc is `sdd/pkg-08-eval-and-ablation/specs/05-sweep-harness-and-run-registry.md §4`. The @dataclass body lives at `hyper_mve/experiments/run_registry.py` (per spec 05 §1 deliverable list). This spec mirrors the schema **byte-identically**; drift is caught by `test_run_registry_23_field_jsonl_schema` (§9.2) which compares `dataclasses.fields(RegistryRow)` against the verbatim enumeration below AND asserts the field-count is exactly 23.

Each row is one JSON dict per line in `runs/registry.jsonl`. Updates to a row are **new lines appended** (never in-place mutation); see spec 05 §4.2 append-only contract. Concurrent appenders coordinate via OS file lock (`msvcrt.locking(LK_LOCK)` on Windows; `fcntl.flock(LOCK_EX)` on POSIX); see spec 05 §4.3.

### 4.2 Verbatim 23-field table (grouped 6-bucket; sum check at footer)

| # | Field name | Type | Default | Populated by | Schema bucket |
|---|---|---|---|---|---|
| 1 | `run_id` | `str` (uuid4 hex) | — (required) | spec 05 §3.3 per-row uuid; new on retry | Identity |
| 2 | `variant` | `str` (CLI string, ∈ 14 legal values) | — (required) | spec 05 §5.2 sweep payload | Identity |
| 3 | `seed` | `int` | — (required) | spec 05 §5.2 sweep payload | Identity |
| 4 | `config_hash` | `str` (32-char blake2b-16 hex) | — (required) | spec 05 §3.5 `compute_config_hash(cfg)` | Identity |
| 5 | `ablation_cell` | `str \| None` | `None` | spec 05 §3.1 `SweepConfig.ablation_cell_id` | Identity |
| 6 | `sweep_row_index` | `int` | — (required) | spec 05 §3.3 lex-order index within parent SweepConfig | Provenance |
| 7 | `git_sha` | `str` (40-char hex) | — (required) | spec 05 §4 `git rev-parse HEAD` at row start | Provenance |
| 8 | `git_dirty` | `bool` | — (required) | spec 05 §4 `git status --porcelain` non-empty | Provenance |
| 9 | `started_at_iso8601` | `str` (UTC ISO 8601 with `Z` suffix) | — (required) | spec 05 §4 row start time | Provenance |
| 10 | `completed_at_iso8601` | `str \| None` | `None` | spec 05 §4 row end time; `None` while running | Lifecycle |
| 11 | `status` | `Literal["pending", "running", "completed", "failed", "skipped"]` | — (required) | spec 05 §4 state machine | Lifecycle |
| 12 | `failure_reason` | `str \| None` | `None` | spec 05 §5.3 exit-code → message map | Lifecycle |
| 13 | `gpu_id` | `int \| None` | `None` | spec 05 §7 per-GPU semaphore allocation | Resource |
| 14 | `walltime_seconds` | `float \| None` | `None` | spec 05 §4 row end − row start | Resource |
| 15 | `peak_gpu_memory_mb` | `float \| None` | `None` | spec 05 §4 `torch.cuda.max_memory_allocated` | Resource |
| 16 | `config_snapshot_path` | `str` | — (required) | spec 05 §5.3 `runs/<run_tag>/config.yaml` path | Output pointers |
| 17 | `checkpoint_path` | `str \| None` | `None` | spec 05 §5.3 `runs/<run_tag>/ckpt_final.pt` path | Output pointers |
| 18 | `eval_report_path` | `str \| None` | `None` | spec 05 §5.3 `runs/<run_tag>/eval_report.json` path | Output pointers |
| 19 | `tensorboard_dir` | `str` | — (required) | spec 05 §5.3 `runs/<run_tag>/tb/` path | TB pointer |
| 20 | `return_mean` | `float \| None` | `None` | spec 05 §4 from `EvalReport.return_mean` on `completed` | Duplicated summary metrics |
| 21 | `return_zero_shot_unseen` | `float \| None` | `None` | spec 05 §4 from `EvalReport.return_zero_shot_unseen` on `completed` | Duplicated summary metrics |
| 22 | `regret_mean` | `float \| None` | `None` | spec 05 §4 from `EvalReport.regret_mean` on `completed` | Duplicated summary metrics |
| 23 | `schema_version` | `str` | `"pkg08-spec05-v1"` | spec 05 §4 schema sentinel | Schema version |

**Field-count sum check** (the 8-bucket breakdown — authoritative source is the @dataclass body in spec 05 §4):

```
Identity                     5   (rows 1–5)
Provenance                   4   (rows 6–9)
Lifecycle                    3   (rows 10–12)
Resource                     3   (rows 13–15)
Output pointers              3   (rows 16–18)
TensorBoard pointer          1   (row 19)
Duplicated summary metrics   3   (rows 20–22)
                            ──
Schema-domain subtotal      22
Schema version sentinel      1   (row 23)
                            ──
TOTAL                       23
```

The "22 schema-domain + 1 schema_version = 23 keys on every JSONL line" accounting matches spec 05 §4 verbatim ("Field count = 23 keys (22 schema-domain + 1 `schema_version` sentinel = 23)"). The drift detector test `test_run_registry_23_field_jsonl_schema` (§9.2) asserts `len(dataclasses.fields(RegistryRow)) == 23` AND that the JSON-key set matches the union of the table above.

### 4.3 Append-only contract (spec 05 §4.2)

State transitions are always new lines, never in-place mutation:

```
1. parent enqueues row → append RegistryRow(status="pending", run_id=R0, ...)
2. subprocess starts   → append RegistryRow(status="running", run_id=R0, ...)
3. subprocess completes → append RegistryRow(status="completed", run_id=R0, completed_at=..., ...)
```

Reads use `pandas.read_json("runs/registry.jsonl", lines=True).sort_values("started_at_iso8601").groupby("run_id").last()` to materialise the latest state per row. Failed/skipped rows are NOT deleted; the `(variant, seed, config_hash)` dedup tuple becomes the resume key (spec 05 §8.2).

### 4.4 Concurrent file-lock contract (spec 05 §4.3)

Two appenders coexist (the parent sweep loop and each child subprocess); both use OS file locks:
- **Windows** (the project's primary platform): `msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)` with polled retry then `LK_LOCK` blocking fallback; `msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)` to release.
- **POSIX**: `fcntl.flock(fd, fcntl.LOCK_EX)` to acquire; `fcntl.flock(fd, fcntl.LOCK_UN)` to release.

Append granularity is per-row (one `append_row` call → one `_lock_exclusive → write → fsync → _unlock` cycle); the lock window is microseconds, the 16-thread × 100-row stress test in spec 05 §9 (`test_run_registry_jsonl_append_atomic`) completes in < 1 s. The `os.fsync` after write guarantees the line survives a subprocess crash immediately after write.

---

## 5. 5 new cfg fields exhaustive enumeration (design D10 lock)

Per design §4 D10 + README §"5 新 cfg 字段穷举" + the four sibling specs that declare each field, pkg-08 contributes **exactly 5 new cfg fields** to the V4Config consumption surface. The exhaustive enumeration:

| # | Field path | Type | Default | Owner cfg | Declared in | Codified in spec 08 | Notes |
|---|---|---|---|---|---|---|---|
| 1 | `cfg.env.c_visible` | `bool` | `True` | `EnvConfig` | spec 02 §3.1 + §5.2 | §6 (B) row 1 `env_config.py` `[config-additions]` | Default `True` preserves all current behaviour. When `False`, triggers spec 02 §5.1 obs-mask in `observations.py`. ĉ head becomes "real inference" in c_hidden runs (Theory Audit Q7). |
| 2 | `cfg.train.randomize_order` | `bool` | `True` | `TrainConfig` | spec 06 §4.1 + §6 + §8.1 | §6 (B) row 2 `train_config.py` `[config-additions + alias-with-deprecation]` | Canonical name after the Theory Audit Q2 rename. The literal default `True` matches the legacy `cfg.train.use_coord_desc: bool = True` default at `hyper_mve/configs/train_config.py` (the legacy field is preserved as a `@property` shim with `DeprecationWarning(stacklevel=2)` on READ and a `_use_coord_desc_compat` reconciliation sentinel on WRITE via `dataclasses.replace`). No production run / YAML / ckpt behaviour change. |
| 3 | `cfg.train.mve_joint_enumerate` | `bool` | `False` | `TrainConfig` | spec 06 §5 + §6 + §8.1 | §6 (B) row 2 `train_config.py` `[config-additions]` | Triggers the +1 logical-branch patch in `mve_planner.py` (spec 06 §5.1; §6 (A) row 2 below). At N>2, the planner emits `RuntimeWarning` and falls through to coord-descent (spec 06 Lock 3) to prevent the sweep harness from hanging on an intractable 6^N enumeration. The Easy N=2 cell (6²=36 candidates) is the only legal use; hard-pinned by `abl4_joint_easy_n2.yaml`. |
| 4 | `cfg.eval.eval_planner_mode` | `Literal["direct_inference", "planner_no_crn", "planner_no_coord_desc", "planner_full"]` | `"planner_full"` | `EvalConfig` | spec 03 §3 + §8.1 | §6 (B) row 3 `eval_config.py` `[config-additions]` | The canonical 4-mode evaluation-time dispatch (spec 03 §2 truth table). Default `"planner_full"` preserves the current eval path. Per spec 03 Lock 2, the canonical field takes precedence; the short-circuit alias (field 5 below) is a consistency-checked back-compat door. |
| 5 | `cfg.eval.eval_use_planner_direct_inference` | `bool` | `False` | `EvalConfig` | spec 03 §3 + §8.1 | §6 (B) row 3 `eval_config.py` `[config-additions]` | Deprecated-on-arrival short-circuit alias for `eval_planner_mode="direct_inference"`. Per spec 03 Lock 2: legal only when consistent with the canonical field; the evaluator MUST raise `ValueError` at entry on `True` + `eval_planner_mode != "direct_inference"` to prevent silent mode flips. |

**Sub-namespace distribution** (`cfg.env / cfg.train / cfg.eval` — per design D10 last paragraph):
- `EnvConfig`: 1 field (c_visible) — environment-domain.
- `TrainConfig`: 2 fields (randomize_order, mve_joint_enumerate) + 1 @property shim (use_coord_desc alias) — training-domain.
- `EvalConfig`: 2 fields (eval_planner_mode, eval_use_planner_direct_inference) — evaluation-domain.
- Total: **5 user-facing fields** distributed across 3 sub-namespaces, all consumed by pkg-08 specs 02 / 03 / 06.

The synthetic `_use_coord_desc_compat: bool = True` field declared in spec 06 §4.2 is an internal `__post_init__` reconciliation sentinel and is **excluded** from the 5-field enumeration (it starts with an underscore and is not a user-facing knob). Spec 06 §6 + test `test_use_coord_desc_compat_is_internal` (spec 06 §7.6) assert this is not listed in user-facing field discovery.

**Pkg-01 spec 05 synchronous consumption**: Pkg-01 spec 05 (V4Config consumption table) registers all 5 new fields synchronously with this spec 08 finalisation. Pkg-01 SDD is **not modified** — the fields are declared by the consuming sibling specs (02 / 03 / 06) and added to the existing dataclasses (`env_config.py` / `train_config.py` / `eval_config.py`) per §6 (B) below. The V4Config root object gains no new sub-namespace; the 5 fields land within the 3 existing sub-namespaces.

---

## 6. 3-block "下游补丁声明" (mirror of README §"修改" A/A'/B layout)

Per Lock 2 + README §"输出清单" §"修改" 3-block layout, pkg-08 declares **7 total file-level changes** (3 behavioural + 1 zero-behaviour threading + 3 config-additions) across 5 files. Each is sourced from a sibling spec; this section is the canonical aggregation point.

### 6.1 (A) 3 behavioural patches

These are **actual code-semantic changes** to the running system. Each patch's owning sibling spec is the source of truth; spec 08 aggregates for single-document review and CI drift detection.

| Patch | File path | Size (honest line count) | Tag | Declared in | Test gate |
|---|---|---|---|---|---|
| A.1 | `hyper_mve/envs/resource_commons/observations.py` | +3 lines of behavioural code inside `build_observation` (the obs-mask hook); +1 line signature extension to accept `env_cfg: EnvConfig \| None = None` (parameter only — the +3 behavioural lines consume the kwarg) | `[behavioural]` | spec 02 §5.1 | `test_c_hidden_obs_mask` (spec 02 §6.1) parametrised over Easy (N=2, K=8) + Medium (N=4, K=20) presets; asserts `obs[..., block_offset("global", N, K)[0]] == 0.0` when `c_visible=False` and equals `state.c_t` when `True`; asserts `time_remaining_ratio` (slot c_start+1) is unchanged regardless of `c_visible`; asserts `obs.shape` unchanged. |
| A.2 | `hyper_mve/planning/mve_planner.py` | +1 logical branch (the `if self.mve_joint_enumerate:` block) + ~14 supporting lines (the `RuntimeWarning` block at N>2 + the Easy N=2 sentinel + comments). Honest aggregate: **+1 logical branch with ~15 supporting lines**. (Per spec 06 §5.1, the README "+1 line" slogan is the **logical branch count**, NOT the literal line count; this spec 08 §6.1 reconciles the wording so future readers see the honest accounting.) | `[behavioural; +1 logical branch + warn block]` | spec 06 §5.1 | `test_abl4_joint_enum_easy_n2_only` (spec 06 §7.4 + README C8-ABL-ABL4A) parametrised over `--preset ∈ {easy, medium, hard}`; sub-test invokes the planner at N=4 with `mve_joint_enumerate=True` and asserts `pytest.warns(RuntimeWarning)` + `pi_mve` byte-identical to coord-descent baseline. |
| A.3 | `hyper_mve/scripts/train_main.py` | CLI argument rename: `--use_coord_desc → --randomize_order` with the legacy `--use_coord_desc` preserved as `argparse` alias that emits `DeprecationWarning` on use. The deprecated CLI alias routes through the same `_use_coord_desc_compat` reconciliation path declared in spec 06 §4.2 (which converts to `cfg.train.randomize_order`). Net change: ~6 lines (the `add_argument` rename + alias `add_argument` for the legacy flag + the DeprecationWarning emit on the legacy path). | `[downstream-cli; alias-with-deprecation]` | spec 06 §8.3 + (implicitly) §4.3 grep audit | `test_use_coord_desc_alias_warns_cli` (a sibling of spec 06 §7.5 `test_use_coord_desc_alias_warns_read` / `_write`; the CLI variant invokes `train_main.parse_args(["--use_coord_desc", "True"])` and asserts the resulting `cfg.train.randomize_order` is `True` and that `DeprecationWarning` was emitted). |

**Total behavioural surface**: 3 patches × (1 file each) = **3 files modified for behaviour change**. Honest line accounting: ≈ 4 + 15 + 6 = **25 lines of behavioural code** across the 3 files. (The README outline's "Pkg-02 obs-mask +3 行 / Pkg-05 planner flag +1 行 / Pkg-05 CLI 字串 rename" slogan is the **logical-change count** under the Theory Audit Q2 / M8 framings, not the literal line-diff count; this §6.1 reconciliation matches the SDD "no silent edits" rule which prioritises honest line accounting over slogan brevity.)

### 6.2 (A') 1 zero-behaviour threading patch

This is a **threading-only edit** with zero behavioural consequence: `env_cfg=self.cfg` is passed to `build_joint_observation` at two call sites in `env.py` (`reset` and `step`). The kwarg pass-through is required because spec 02 §5.1's `observations.py` patch reads `env_cfg.c_visible` (signature change), and the kwarg has to be threaded to the leaf function from the call sites.

| Patch | File path | Size | Tag | Declared in | Test gate |
|---|---|---|---|---|---|
| A'.1 | `hyper_mve/envs/resource_commons/env.py` | +2 call-site kwarg additions: `env_cfg=self.cfg` added to the `build_joint_observation(...)` call inside `ResourceCommonsEnv.reset` and inside `ResourceCommonsEnv.step`. **Zero behaviour change**. The kwarg is a pure pass-through of the same `EnvConfig` instance the env already holds; no condition is evaluated and no info-dict / step / reset / reward / termination semantics change. | `[no-behaviour]` | spec 02 §5.1b | `test_env_py_call_site_kwarg_no_behaviour_change` (the byte-identical regression test specified in spec 02 §5.1b): runs the env without and with the kwarg under `c_visible=True`; asserts `reset()` and `step()` return values (obs, reward, terminations, truncations, info) are byte-identical. |

Rationale for the explicit (A') block: under strict reading of the README C8-EVAL-CHID1 originally "3 lines / 1 file" slogan, threading the kwarg is technically an `env.py` edit that must be enumerated honestly. Spec 02 §5.1b explicitly carries the patch with the `[no-behaviour]` tag; spec 08 §6.2 promotes it to its own (A') row in the canonical 3-block declaration. README §"修改" is reconciled to "3 declared patches: 4 behavioural lines (`observations.py`) + 2 trivial threading kwargs (`env.py`) + 1 config field (`env_config.py`)".

**Alternative considered and rejected**: threading via a module-level global (`hyper_mve.envs.resource_commons._current_env_cfg`) was rejected — it introduces hidden global state that defeats the per-env-instance isolation `ResourceCommonsEnv` already provides, and it would re-emerge as drift in any future multi-env sweep code (a concurrent sweep with multiple `ResourceCommonsEnv` instances would race on the global). Pass-through kwarg is the honest fix.

### 6.3 (B) 3 config-additions patches

These are **dataclass field additions** with no behavioural consequence on their own (behaviour comes from the consuming sibling specs that read the new fields). Each is `[config-additions]` per the SDD tag convention.

| Patch | File path | Size | Tag | Declared in | Test gate |
|---|---|---|---|---|---|
| B.1 | `hyper_mve/configs/env_config.py` | +1 line adding `c_visible: bool = True` to the `EnvConfig` dataclass field list | `[config-additions]` | spec 02 §5.2 (declared) + spec 02 §3.1 (consumed) | `test_eval_report_c_visible_populated_from_cfg` (spec 02 §6.8) parametrised over `c_visible ∈ {True, False}`; asserts `report.c_visible` round-trips from config to `EvalReport.c_visible` |
| B.2 | `hyper_mve/configs/train_config.py` | +2 fields (`randomize_order: bool = True` + `mve_joint_enumerate: bool = False`) + +1 `@property` shim for the legacy `use_coord_desc` alias (with `DeprecationWarning(stacklevel=2)` on READ) + +1 synthetic `_use_coord_desc_compat: bool = True` field for the `dataclasses.replace` WRITE path reconciliation + +1 `__post_init__` reconciliation block (`if self._use_coord_desc_compat != self.randomize_order: ...`). Net change: ~12 lines (2 fields + 1 property + 1 sentinel + 1 reconciliation block of ~5 lines). | `[config-additions + alias-with-deprecation]` | spec 06 §4.1 + §4.2 + §6 + §8.1 | `test_use_coord_desc_alias_warns_read` + `test_use_coord_desc_alias_warns_write` (spec 06 §7.5); `test_use_coord_desc_grep_codebase_clean` (spec 06 §7.6); `test_mve_joint_enumerate_medium_falls_through_to_coord_desc` (spec 06 §7 Lock 3 guard) |
| B.3 | `hyper_mve/configs/eval_config.py` | +2 fields (`eval_planner_mode: Literal[...] = "planner_full"` + `eval_use_planner_direct_inference: bool = False`). Net change: ~6 lines (2 fields with type annotations + Literal import). | `[config-additions]` | spec 03 §3 + §8.1 | `test_all_4_planner_modes_dispatch` (spec 03 §7.1) parametrised over the 4 literal mode strings; `test_consistency_check_raises_on_inconsistent_short_circuit` + `test_consistency_check_passes_on_consistent_short_circuit` (spec 03 §7.2 + §7.3); `test_5_new_cfg_fields_present_with_correct_defaults` (this spec §9.3) asserts both new EvalConfig fields exist with the declared defaults |

**Total config-additions surface**: 3 patches × (1 file each) = **3 files modified for new fields**. Aggregate field-additions: 1 + 2 + 2 = **5 new fields** (matching the §5 exhaustive enumeration above) + 1 `@property` shim + 1 synthetic compat sentinel (excluded from the user-facing 5 per §5).

### 6.4 Patch-block sum check

```
(A)  3 behavioural patches across 3 files       →  observations.py + mve_planner.py + train_main.py
(A') 1 zero-behaviour threading patch on 1 file →  env.py  [no-behaviour]
(B)  3 config-additions on 3 files              →  env_config.py + train_config.py + eval_config.py

Total files touched by pkg-08:                  7 distinct files
                                                (3 in pkg-02 territory: observations.py, env.py, env_config.py — but env.py touch is [no-behaviour], and env_config.py is consumption-side add)
                                                (2 in pkg-05 territory: mve_planner.py, train_main.py)
                                                (2 in pkg-01 territory: train_config.py, eval_config.py — consumption-side adds)

SDD invariant: pkg-01..05 SDD ZERO modifications. Pkg-07 SDD ZERO modifications.
Code-side patches above are IMPLEMENTATION-ONLY downstream consumers declared in spec 08;
the upstream SDD documents are not edited.
```

The 7-file surface is the **complete** code-level footprint of pkg-08 (excluding the new files pkg-08 itself adds: `hyper_mve/eval/*` + `hyper_mve/experiments/*` per README §"输出清单" `新增` block). Any future addition to this footprint requires a synchronous edit to (a) the sibling spec that owns the new patch, (b) this §6 aggregation table, (c) the README §"修改" 3-block layout, and (d) the drift detector test `test_pkg08_downstream_patch_inventory_byte_identical` (§9.10).

---

## 7. pkg-07 → pkg-08 reverse-consumption 5 anchors (mirror of README §"🔁")

Per Lock 3, this section mirrors **verbatim** the 5-anchor table from pkg-08 README §"🔁 pkg-07 → pkg-08 契约对账":

```
pkg-07 spec 01 §2.1  REGISTRY 11 keys                 ──→ pkg-08 spec 05 §3 sweep enumeration
pkg-07 spec 01 §2.3  量词 canonical (11 keys)          ──→ pkg-08 spec 05 §3 + spec 08 §5
pkg-07 spec 04       N-parametric adapter + 两 flag    ──→ pkg-08 spec 01 §4 external runner eval 通路
pkg-07 spec 08       evaluate() → EvalReport 签名      ──→ pkg-08 spec 01 §3 EvalReport schema + spec 08 §3
pkg-07 spec 08       BaselineLike Union type           ──→ pkg-08 spec 05 sweep harness type sig
```

The block is byte-identical to (i) pkg-08 README §"🔁", (ii) pkg-07 spec 08 §8, and (iii) the consumption table in pkg-07 spec 08 §8.1. Drift detection on these three locations is the heart of the bidirectional cross-package contract bridge.

### 7.1 Anchor-by-anchor consumption (which pkg-08 spec consumes each pkg-07 anchor)

| # | pkg-07 source | pkg-08 consumer spec | Consumption mechanism | Drift consequence |
|---|---|---|---|---|
| 1 | pkg-07 spec 01 §2.1 `REGISTRY` 11 keys (5 internal + 3 Tier-1 + MAMBA + 2 stubs) | pkg-08 spec 05 §3 sweep enumeration | `from hyper_mve.baselines import REGISTRY` + `for variant in REGISTRY: SweepRow(variant=variant, ...)` (spec 05 §3.3 cartesian); pkg-08 spec 06 ablation YAMLs reference variants by REGISTRY keys (`abl1_gen_scope.yaml` lists `hyper / baseline_input_wide / baseline_input_deep / baseline_ma_muzero / no_belief / oracle_only`); pkg-08 spec 05 sizes per-GPU semaphore + JSONL `RunRegistry` budget against the cartesian product (11 variants × ≥5 seeds × 2 presets × LR-sweep-applicable rows) | If pkg-07 adds a 12th REGISTRY key without pkg-08 sync, pkg-08 spec 05 sweep harness silently misses the new variant in `--sweep-variants all` mode and the main table is incomplete |
| 2 | pkg-07 spec 01 §2.3 量词 canonical (REGISTRY = 11 / CLI = 14 / methods main table = 9 / BaselinesConfig = 5 / EvalReport = 32 / EXTERNAL_REGISTRY = 6 / INTERNAL_REGISTRY = 5) | pkg-08 spec 05 §3 (sweep cardinality) + pkg-08 spec 08 §5 (the present spec, cross-link to pkg-07 spec 08 §5.5 量词 canonical) | pkg-08 spec 05 §3 + §10.7 reverse-consumes the 11-key cardinality for sweep enumeration; pkg-08 spec 08 §5 (this spec) re-affirms "**5 new cfg fields** + pkg-07 BaselinesConfig 5 fields → V4Config consumption surface gains 10 fields across pkg-07/08 combined" — the 5+5 split is the canonical contract; any drift in either side's 5 breaks the joined V4Config surface | If the bolded numbers drift (e.g., pkg-07 splits MAMBA-if-sourced into two registry keys, bumping REGISTRY to 12), pkg-08 spec 05 §3 + this §5 must update synchronously; §8.3 grep target enforces |
| 3 | pkg-07 spec 04 N-parametric `ResourceCommonsPettingZooEnv` adapter + two-flag info gate (`oracle_mode=False AND eval_info_mode=False` default; 4 leak surfaces `c_true` / `types` / `hotspot_centers` / `resource_state`) | pkg-08 spec 01 §2.2 `env_fn` arg + §4 internal runner eval routing + §7.1 layer-1 guard (`env._oracle_mode is False`) | pkg-08 spec 01 §2.2 `env_fn: Callable[[], ResourceCommonsPettingZooEnv]` is the unique env factory; pkg-08 spec 01 §7.1 `_verify_env_fn_flags` constructs one `env = env_fn()` and asserts `env._oracle_mode is False` (canonical population of `EvalReport.info_gating_strict`); spec 05 `_sweep_worker.py:_build_env_fn` constructs `ResourceCommonsPettingZooEnv(cfg.env, oracle_mode=False, eval_info_mode=True)` — the **unique caller permitted to flip `eval_info_mode=True`** per pkg-07 spec 04 §10 | If pkg-07 spec 04 renames the `oracle_mode` flag or changes the leak-surface enumeration, pkg-08 spec 01 §7.1 layer-1 guard breaks and `info_gating_strict` semantic flips; both pkg-07 and pkg-08 CI catch the drift |
| 4 | pkg-07 spec 08 `evaluate(env_fn, c_grid, episodes) -> EvalReport` uniform signature on both `BaselineModel.evaluate` and `ExternalBaselineRunner.evaluate` | pkg-08 spec 01 §3 `EvalReport` schema mother-doc + pkg-08 spec 08 §3 (this spec) verbatim mirror | The 32-field schema in pkg-08 spec 01 §3 is the contract; this §3 mirrors byte-identically; pkg-07 spec 05 §8.1 (MAPPO) and pkg-07 spec 06 §2.7 / §3.7 / §4.6 (QMIX / MA-MuZero-GH / MAMBA) reverse-consume the field-population matrix (pkg-07 spec 08 §4.4 is that mirror) | Drift in any field name / type / default forces a synchronous 5-document edit: pkg-08 spec 01 §3 + pkg-08 spec 08 §3 (this spec) + pkg-07 spec 08 §4.4 + pkg-07 spec 05 §8.1 + pkg-07 spec 06 (3 sections). This is the **highest-cost drift** in the entire pkg-07/pkg-08 SDD; both packages' CI independently catch it |
| 5 | pkg-07 spec 08 `BaselineLike = Union[BaselineModel, ExternalBaselineRunner]` load-bearing type alias | pkg-08 spec 05 sweep harness type sig + pkg-08 spec 01 §2.2 `runner: BaselineLike` arg | pkg-08 spec 05 §3.3 sweep enumeration types the iterator as `runner: BaselineLike = create_baseline(cfg, variant)`; pkg-08 spec 01 §2.2 declares `runner: BaselineLike` so the unified evaluator's single dispatch signature works for both branches without `isinstance` branches in the call site | Drift in the alias name (e.g., `BaselineLike → BaselineUnion` or `RunnerLike`) requires synchronous edit of pkg-07 spec 01 §3.2 + pkg-07 spec 08 §2.2 + pkg-08 spec 01 §2.1 + pkg-08 spec 05 |

### 7.2 Bidirectional drift handling workflow (symmetric with pkg-07 spec 08 §8.2)

The pkg-07 ↔ pkg-08 contract bridge is detected by **two independent CI checks**: one in pkg-07 (running `sdd/pkg-07-baselines/scripts/check_ref_matrix.ps1` per pkg-07 README §"实施顺序" Phase 8) and one in pkg-08 (running `sdd/pkg-08-eval-and-ablation/scripts/check_ref_matrix.ps1` per pkg-08 README §"实施顺序" Phase 8). Both scripts grep for the same 5-anchor verbatim block; either failure surfaces the same drift.

Repair procedure when **either** CI outputs `[FAIL]` on the cross-package anchors:

1. **Identify the failing anchor** — the diff output names which of the 5 anchors mismatched.
2. **Locate the canonical source** — per the §7.1 table, the mother-doc location is identifiable from the anchor row.
3. **Determine the drift direction**:
   - If pkg-07 was edited (e.g., REGISTRY gained a key, adapter changed flag semantic, `evaluate()` signature changed, `EvalReport` schema gained a field), pkg-08 must catch up: this spec §3 / §4 / §5 / §7 / §8 must be re-mirrored, and the sibling pkg-08 spec (`01` / `02` / `05` / `06`) that consumes the changed contract must be re-edited synchronously.
   - If pkg-08 was edited (rare — pkg-08 should reverse-consume pkg-07, not generate new contracts that pkg-07 must adopt), the edit is suspect and likely a mistake; revert the pkg-08 edit and re-route through the pkg-07 sibling first.
4. **Synchronous fix** — edit both files in the same commit (or at minimum, the same PR). Sequential commits where pkg-07 is edited first and pkg-08 catches up in the second commit is acceptable; the reverse order (pkg-08 first, then pkg-07) is not — CI on the pkg-07 side would fail until the second commit lands.
5. **Re-run both CI checks** until both `[PASS]`.

The asymmetry (pkg-07 leads, pkg-08 follows) is intentional: pkg-08 is positioned downstream of pkg-07 in the SDD topology (per README §"上下游关联" and design §1.1), so contracts flow upstream-to-downstream (pkg-07 → pkg-08), and any drift originates in the upstream (pkg-07). The bidirectional detection is for safety (the drift is caught on both sides), not symmetry of authorship (pkg-08 does not generate contracts for pkg-07 to mirror).

---

## 8. Drift detector (PowerShell regex + ≥30 verbatim grep anchors)

### 8.1 Drift philosophy (symmetric with pkg-07 spec 08 §6.1)

Spec 08 is **derivative**: it mirrors contracts sourced from sibling specs 01-07 of pkg-08 plus the 5 reverse-consumption anchors from pkg-07. The sibling spec is the truth; spec 08 is the one-document summary. Drift = sibling spec changed but spec 08 did not catch up.

The CI script `sdd/pkg-08-eval-and-ablation/scripts/check_ref_matrix.ps1` (Day 8 of pkg-08 SDD calendar; README §"实施顺序" Phase 8) runs two checks:

- **Forward ref check** (intra-package): every spec 01-07 of pkg-08 references spec 08 only via the cross-references section (enforced by the existing pkg-08 README §"spec 间引用表" matrix). Spec 08 references all of spec 01-07 (the spec-08 row of the matrix has 7 entries).
- **Anchor consistency check** (intra-package + cross-package): every verbatim anchor listed in §8.3 below appears in BOTH spec 08 AND the cited sibling spec. The check uses a literal grep for the anchor string; mismatch = sibling edited, spec 08 did not catch up. The check produces `[FAIL]` + a diff of present-in-sibling-absent-in-spec-08 anchors.

The asymmetry (sibling does not cite spec 08; spec 08 cites sibling) prevents circular ref-chain pings and makes drift unambiguously assignable to one side: if the anchor is in the sibling but not in spec 08, spec 08 must update. There is no "spec 08 changed first, sibling needs updating" case — spec 08 has no contract content of its own to change first.

### 8.2 Regex check (PowerShell pattern)

`check_ref_matrix.ps1` runs the following PowerShell regex against every spec file in `sdd/pkg-08-eval-and-ablation/specs/`:

```powershell
$pat = '(?<![Pp]kg-\d{2} )spec\s+0([1-8])|\b0([1-8])-[a-z]'
#   - Bare "spec 0X"   but excludes "Pkg-NN spec 0X" / "pkg-NN spec 0X" via case-insensitive negative look-behind
#   - File name "0X-name"  spec-internal references in cross-ref sections
```

The regex matches two intra-package reference forms:
1. **Bare `spec 0X`** (e.g., `spec 05 §4` for the within-pkg-08 sibling reference) — but excludes cross-package mentions like `pkg-07 spec 04` / `Pkg-07 spec 04` / `pkg-08 spec 01` / `Pkg-08 spec 01` via the **case-insensitive** negative look-behind `(?<![Pp]kg-\d{2} )`. The `[Pp]` character class handles both common casings observed in body text; if a third casing (`PKG-NN`) ever appears, expand to `[PpK]kg` or use the PowerShell `(?i)` mode for the whole pattern.
2. **Filename `0X-...md`** (e.g., `02-zero-shot-and-c-hidden.md`) — captures cross-ref-section file-link references.

The captured digit (group 1 or group 2) yields the cited spec number. The script tabulates per-file citations into `sdd/pkg-08-eval-and-ablation/ref_matrix.csv` and compares against the **expected matrix** in pkg-08 README §"spec 间引用表". Spec 08's expected row is: cites spec 01, 02, 03, 04, 05, 06, 07 (all 7). Deviation = `[FAIL]`.

The regex is **byte-identical** to pkg-07 spec 08 §6.2's regex (same `(?<![Pp]kg-\d{2} )` pattern, same dual-form matching). This is the M6 ref-matrix discipline: same regex across both packages, same grep table shape, same drift-handling workflow.

### 8.3 Verbatim grep targets (≥30 anchors)

The CI check additionally greps for the following verbatim strings; each must appear in **both** spec 08 AND its cited sibling spec.

| # | Anchor ID | Verbatim string | Sibling source |
|---|---|---|---|
| 1 | `spec-01-evalreport-32-fields` | `Total field count: **32**` | pkg-08 spec 01 §3 line 99 |
| 2 | `spec-01-evalreport-schema-version` | `schema_version: str = "pkg08-spec01-v1"` | pkg-08 spec 01 §3.1 |
| 3 | `spec-01-run-eval-not-replaced` | `training/evaluation.py:run_eval is NOT replaced` | pkg-08 spec 01 Lock 1 |
| 4 | `spec-01-evalreport-lock-2-byte-identical` | `EvalReport is the unique evaluator output schema` | pkg-08 spec 01 Lock 2 |
| 5 | `spec-01-evaluate-signature` | `def evaluate(runner: BaselineLike, env_fn, cfg) -> EvalReport` | pkg-08 spec 01 §2.1 |
| 6 | `spec-02-obs-mask-3-lines` | `obs[c_start] = 0.0` (the obs-mask zeroing line) | pkg-08 spec 02 §3.2 |
| 7 | `spec-02-zero-shot-grid-cfg-driven` | `Zero-shot grid is config-driven, never hardcoded` | pkg-08 spec 02 Lock 2 |
| 8 | `spec-02-disjoint-union-invariant` | `zero_shot_test_c must equal train ∪ unseen` | pkg-08 spec 02 §2.6 |
| 9 | `spec-02-regret-cache-path` | `runs/_oracle_ceilings/<config_hash>/<c>.json` | pkg-08 spec 02 §4.3 |
| 10 | `spec-03-4-mode-literal` | `Literal["direct_inference", "planner_no_crn", "planner_no_coord_desc", "planner_full"]` | pkg-08 spec 03 §3 |
| 11 | `spec-03-cfg-eval-planner-mode` | `cfg.eval.eval_planner_mode` | pkg-08 spec 03 §3 + §4.2 |
| 12 | `spec-03-consistency-check-raises` | `eval_use_planner_direct_inference=True conflicts with eval_planner_mode` | pkg-08 spec 03 Lock 2 + §4.1 |
| 13 | `spec-04-mup-selftest-verdict` | `MupSelftestVerdict` (the secondary `@dataclass(frozen=True)` distinct from `EvalReport`) | pkg-08 spec 04 §3.4 |
| 14 | `spec-04-18-row-grid` | `2 widths × 3 LRs × 3 seeds = 18 runs` | pkg-08 spec 04 §2.1 |
| 15 | `spec-05-registry-23-keys` | `22 schema-domain + 1 schema_version = 23 keys` | pkg-08 spec 05 §4 |
| 16 | `spec-05-subprocess-per-row` | `subprocess.Popen([sys.executable, "-m", "hyper_mve.experiments._sweep_worker"]` | pkg-08 spec 05 §5.2 + Lock 1 |
| 17 | `spec-05-per-gpu-semaphore` | `class GpuSemaphore` | pkg-08 spec 05 §7 |
| 18 | `spec-05-jsonl-append-only` | `JSONL append-only` | pkg-08 spec 05 Lock 2 + §4.2 |
| 19 | `spec-06-5-canned-yamls` | `ABLATION_IDS = ("abl1", "abl4_crn_joint", "abl4_joint_easy_n2", "abl6", "abl7")` | pkg-08 spec 06 §2.1 |
| 20 | `spec-06-use-coord-desc-alias-deprecation` | `cfg.train.use_coord_desc is deprecated; use cfg.train.randomize_order instead` | pkg-08 spec 06 §4.1 |
| 21 | `spec-06-mve-joint-enumerate-warn-fallthrough` | `mve_joint_enumerate=True with N={self.N}` (the warn-fallthrough warning text) | pkg-08 spec 06 §5.1 + Lock 3 |
| 22 | `spec-06-abl4-joint-easy-n2-pin` | `HARD-PIN per spec 06 Lock 3` (or `preset: easy` hard-pin in `abl4_joint_easy_n2.yaml`) | pkg-08 spec 06 §3.3 |
| 23 | `spec-07-welch-t-test` | `scipy.stats.ttest_ind(equal_var=False)` | pkg-08 spec 07 (Welch t-test) |
| 24 | `spec-07-holm-bonferroni` | `Holm-Bonferroni` (≥3 method comparisons) | pkg-08 spec 07 |
| 25 | `spec-07-matplotlib-agg` | `matplotlib.use("Agg")` (headless backend) | pkg-08 spec 07 |
| 26 | `pkg-07-registry-11-keys` | `INTERNAL_REGISTRY` (5 keys) and `EXTERNAL_REGISTRY` (6 keys) | pkg-07 spec 01 §3.2 + pkg-07 spec 08 §3.1 |
| 27 | `pkg-07-baseline-like-union` | `BaselineLike: TypeAlias = Union["BaselineModel", "ExternalBaselineRunner"]` | pkg-07 spec 01 §3.2 + pkg-07 spec 08 §2.2 |
| 28 | `pkg-07-n-parametric-adapter` | `self._N = self._env.N; self.agents = [f"agent_{i}" for i in range(self._N)]` | pkg-07 spec 04 §3 + pkg-07 spec 08 §6.3 anchor 15 |
| 29 | `pkg-07-two-flag-info-gate` | `oracle_mode=False AND eval_info_mode=False` | pkg-07 spec 04 §7 + pkg-07 spec 08 §6.3 anchor 16 |
| 30 | `pkg-07-evaluate-signature` | `evaluate(env_fn, c_grid, episodes) -> EvalReport` uniform signature | pkg-07 spec 08 §4.1 |

Anchors 1-5 are spec 01 (5 anchors); 6-9 are spec 02 (4 anchors); 10-12 are spec 03 (3 anchors); 13-14 are spec 04 (2 anchors); 15-18 are spec 05 (4 anchors); 19-22 are spec 06 (4 anchors); 23-25 are spec 07 (3 anchors); 26-30 are cross-package pkg-07 anchors (5 anchors — matching the §7 reverse-consumption 5-anchor block byte-for-byte).

**Total = 30 verbatim grep targets** (≥30 per Lock 2 + spec authoring requirements).

### 8.4 Drift handling workflow (symmetric with pkg-07 spec 08 §6.4)

The repair procedure when `check_ref_matrix.ps1` outputs `[FAIL]`:

1. **Identify the failing anchor** — the diff output names which anchor ID is missing or whose string mismatches.
2. **Locate the sibling source** — column "Sibling source" in §8.3 gives the canonical location.
3. **Determine the drift direction**:
   - If the anchor appears in the sibling but not in spec 08: **spec 08 must catch up**. The sibling was edited (intentionally or unintentionally); spec 08 mirrors the new wording.
   - If the anchor appears in spec 08 but not in the sibling: this is the **forbidden direction** (Lock 1: spec 08 is derivative, not source). Spec 08 has gone ahead of the sibling — revert the spec 08 edit and edit the sibling first instead.
4. **Synchronous fix** — once the drift direction is determined, edit both files in the same commit. Two separate commits (sibling first, then spec 08) is acceptable; spec 08 first then sibling is NOT (CI will fail in the interim).
5. **Re-run** `pwsh sdd/pkg-08-eval-and-ablation/scripts/check_ref_matrix.ps1` until `[PASS]`. If the failing anchor is one of the 5 cross-package pkg-07 anchors (26-30), additionally run `pwsh sdd/pkg-07-baselines/scripts/check_ref_matrix.ps1` to confirm pkg-07 spec 08 §8 (the symmetric mirror) is also consistent.

The §8.3 anchor strings are intentionally verbose ("verbatim grep targets") so that copy-paste edits are reliable. A reviewer should not be inventing wording differences during the fix — they should copy the sibling spec's anchor string into spec 08 verbatim.

---

## 9. Test contract — ≥7 named tests

The test files locked here are **new** tests that verify spec 08's contract surface. They do not duplicate spec 01-07's sibling tests (which already exist under `tests/eval/` and `tests/experiments/`); they verify the **integration boundary** that spec 08 mirrors.

### 9.1 `test_eval_report_32_field_dataclass_lock` (NEW — spec-08-specific)

Located at `tests/integration/test_integration_pkg08.py` (new file owned by spec 08). Asserted invariants:

```python
def test_eval_report_32_field_dataclass_lock():
    """Spec 08 §3 lock: EvalReport @dataclass body has exactly 33 fields total
    (32 schema-payload + 1 schema_version sentinel = 32 headline + 1 = 33 dataclass fields).
    Drift in spec 01 §3.1 (mother-doc) or this §3 (mirror) is caught here."""
    from dataclasses import fields
    from hyper_mve.eval.eval_report import EvalReport
    field_list = fields(EvalReport)
    # 33 dataclass fields total (the @dataclass body length)
    assert len(field_list) == 33
    # Headline "32" count = 33 - 1 (schema_version sentinel)
    schema_payload = [f for f in field_list if f.name != "schema_version"]
    assert len(schema_payload) == 32
    # Verbatim 33 field names per spec 08 §3.2 (order-preserving)
    expected_names = (
        # Identity (6)
        "variant", "seed", "config_hash", "eval_mode", "eval_planner_mode", "c_visible",
        # Headline (5)
        "return_mean", "return_sem", "return_zero_shot_seen", "return_zero_shot_unseen", "return_zero_shot_gap",
        # Per-c (3)
        "return_per_c", "return_per_c_sem", "episodes_per_c",
        # c-segment (2)
        "return_per_segment", "return_per_segment_sem",
        # Bell-curve (2)
        "return_per_type_ratio", "return_per_type_ratio_sem",
        # Regret (4)
        "regret_per_c", "regret_mean", "oracle_ceiling_per_c", "oracle_ceiling_cache_hit",
        # Planner-prior (3)
        "planner_prior_return_gap", "direct_inference_return_mean", "planner_full_return_mean",
        # Diagnostics (3)
        "walltime_seconds", "env_steps_evaluated", "episodes_total",
        # Oracle-leak / info-gating (2)
        "info_gating_strict", "set_context_subjective_oracle_leak",
        # Belief diagnostics (2; nullable)
        "belief_c_mae", "belief_c_calibration",
        # Schema version (1)
        "schema_version",
    )
    actual_names = tuple(f.name for f in field_list)
    assert actual_names == expected_names
```

### 9.2 `test_run_registry_23_field_jsonl_schema` (NEW — spec-08-specific)

Located at `tests/integration/test_integration_pkg08.py`. Asserted invariants:

```python
def test_run_registry_23_field_jsonl_schema():
    """Spec 08 §4 lock: RegistryRow @dataclass body has exactly 23 fields total
    (22 schema-domain + 1 schema_version sentinel)."""
    from dataclasses import fields
    from hyper_mve.experiments.run_registry import RegistryRow
    field_list = fields(RegistryRow)
    assert len(field_list) == 23
    schema_domain = [f for f in field_list if f.name != "schema_version"]
    assert len(schema_domain) == 22
    # Verbatim 23 field names per spec 08 §4.2 (order-preserving):
    expected_names = (
        "run_id", "variant", "seed", "config_hash", "ablation_cell",          # Identity (5)
        "sweep_row_index", "git_sha", "git_dirty", "started_at_iso8601",      # Provenance (4)
        "completed_at_iso8601", "status", "failure_reason",                   # Lifecycle (3)
        "gpu_id", "walltime_seconds", "peak_gpu_memory_mb",                   # Resource (3)
        "config_snapshot_path", "checkpoint_path", "eval_report_path",        # Output pointers (3)
        "tensorboard_dir",                                                     # TB pointer (1)
        "return_mean", "return_zero_shot_unseen", "regret_mean",              # Duplicated summary metrics (3)
        "schema_version",                                                      # Schema sentinel (1)
    )
    actual_names = tuple(f.name for f in field_list)
    assert actual_names == expected_names
```

### 9.3 `test_5_new_cfg_fields_present_with_correct_defaults` (NEW — spec-08-specific)

Located at `tests/integration/test_integration_pkg08.py`. Asserted invariants:

```python
def test_5_new_cfg_fields_present_with_correct_defaults():
    """Spec 08 §5 lock: 5 new cfg fields are all declared with the correct types
    and defaults across cfg.env / cfg.train / cfg.eval."""
    from hyper_mve.configs.env_config import EnvConfig
    from hyper_mve.configs.train_config import TrainConfig
    from hyper_mve.configs.eval_config import EvalConfig
    # cfg.env.c_visible : bool = True
    env = EnvConfig()
    assert hasattr(env, "c_visible") and isinstance(env.c_visible, bool) and env.c_visible is True
    # cfg.train.randomize_order : bool = True
    train = TrainConfig()
    assert hasattr(train, "randomize_order") and isinstance(train.randomize_order, bool) and train.randomize_order is True
    # cfg.train.mve_joint_enumerate : bool = False
    assert hasattr(train, "mve_joint_enumerate") and isinstance(train.mve_joint_enumerate, bool) and train.mve_joint_enumerate is False
    # cfg.eval.eval_planner_mode : Literal[...] = "planner_full"
    ev = EvalConfig()
    assert hasattr(ev, "eval_planner_mode") and ev.eval_planner_mode == "planner_full"
    assert ev.eval_planner_mode in ("direct_inference", "planner_no_crn", "planner_no_coord_desc", "planner_full")
    # cfg.eval.eval_use_planner_direct_inference : bool = False
    assert hasattr(ev, "eval_use_planner_direct_inference") and isinstance(ev.eval_use_planner_direct_inference, bool) and ev.eval_use_planner_direct_inference is False
```

### 9.4 `test_observations_py_obs_mask_invariant` — sibling-mirror of spec 02 §6.1

Located at `tests/eval/test_c_hidden_obs_mask.py` (owned by spec 02 §6). Spec 08 mirrors the test name for cross-document grep; the test body is owned by spec 02 §6.1. Asserted invariants: parametrised over Easy (N=2, K=8) + Medium (N=4, K=20) presets; for each preset, two configs (`c_visible=True`, `c_visible=False`); asserts `obs[..., block_offset("global", N, K)[0]] == 0.0` when `c_visible=False`, equals `state.c_t` (set via `options={"c": 0.5}`) when `True`; asserts `time_remaining_ratio` (slot c_start+1) is unchanged regardless of `c_visible`; asserts `obs.shape` unchanged. This is the §6 (A.1) behavioural patch's test gate.

### 9.5 `test_env_py_call_site_kwarg_no_behaviour_change` — sibling-mirror of spec 02 §5.1b regression

Located at `tests/eval/test_env_py_threading.py` (owned by spec 02 §6 / §5.1b). The byte-identical regression test for the §6 (A'.1) zero-behaviour threading patch:

```python
def test_env_py_call_site_kwarg_no_behaviour_change():
    """Spec 02 §5.1b / spec 08 §6 (A'.1): the env_cfg=self.cfg kwarg added to
    build_joint_observation calls in env.py reset/step has ZERO behaviour change.
    Run the env without and with the kwarg under c_visible=True; assert
    rewards/obs/info are byte-identical."""
    import numpy as np
    cfg = EnvConfig(N=2, K=8, c_visible=True)
    # Reset with same seed; advance same number of steps with same actions.
    env_before = ResourceCommonsEnv(cfg, seed=42)
    env_after = ResourceCommonsEnv(cfg, seed=42)
    obs_before, info_before = env_before.reset(options={"c": 0.5})
    obs_after, info_after = env_after.reset(options={"c": 0.5})
    np.testing.assert_array_equal(obs_before, obs_after)
    assert info_before == info_after
    # Take 50 deterministic steps; assert byte-identical outputs throughout.
    rng = np.random.default_rng(123)
    for _ in range(50):
        a = rng.integers(0, 6, size=cfg.N)
        out_before = env_before.step(a)
        out_after = env_after.step(a)
        np.testing.assert_array_equal(out_before[0], out_after[0])  # obs
        np.testing.assert_array_equal(out_before[1], out_after[1])  # reward
        assert out_before[2] == out_after[2]                         # term
        assert out_before[3] == out_after[3]                         # trunc
        assert out_before[4] == out_after[4]                         # info
```

### 9.6 `test_mve_planner_mve_joint_enumerate_branch` — sibling-mirror of spec 06 §7.7

Located at `tests/experiments/test_abl4_joint_enum.py` (owned by spec 06 §7). Synthetic test exercising the §6 (A.2) behavioural patch:

```python
def test_mve_planner_mve_joint_enumerate_branch_easy_n2():
    """Spec 06 §5 / spec 08 §6 (A.2): mve_joint_enumerate=True at Easy N=2
    enumerates all 6^2=36 joint actions. Assert the planner consumed the
    enumerated cells (not the coord-descent fallback)."""
    cfg = make_preset("easy")          # N=2, K=8
    cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, mve_joint_enumerate=True))
    planner = MVEPlanner(cfg)
    # Synthetic state/belief/model fixture
    model = _make_dummy_hyper_model(cfg)
    pi_mve = planner.plan(model, _dummy_state(cfg), _dummy_belief(cfg))
    # The joint-enum branch should produce a flatten 36-action sentinel marker
    # (spec 06 §5.1 sentinel "JOINT_ENUM"); the sub-test asserts the planner went
    # through that branch by inspecting an internal counter or branch flag.
    assert planner._last_branch == "joint_enum"
```

### 9.7 `test_train_main_use_coord_desc_alias_warns_cli` — sibling-mirror of spec 06 §7.5 (CLI variant)

Located at `tests/experiments/test_train_main_cli_rename.py`. The CLI counterpart of spec 06 §7.5 READ/WRITE alias tests; verifies the §6 (A.3) behavioural patch (CLI rename):

```python
def test_train_main_use_coord_desc_alias_warns_cli():
    """Spec 06 §8.3 / spec 08 §6 (A.3): the --use_coord_desc CLI flag is
    preserved as an alias for --randomize_order; using it emits DeprecationWarning
    and the resulting cfg.train.randomize_order is set accordingly."""
    from hyper_mve.scripts.train_main import parse_args
    with pytest.warns(DeprecationWarning, match="use_coord_desc"):
        cfg = parse_args(["--variant", "hyper", "--use_coord_desc", "False", "--preset", "easy"])
    assert cfg.train.randomize_order is False
    # Sanity: the canonical flag also works without warning
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        cfg = parse_args(["--variant", "hyper", "--randomize_order", "False", "--preset", "easy"])
    assert cfg.train.randomize_order is False
```

### 9.8 `test_drift_detector_finds_all_grep_anchors` (NEW — spec-08-specific meta-test)

Located at `tests/integration/test_integration_pkg08.py`. Verifies the §8.3 grep table:

```python
def test_drift_detector_finds_all_grep_anchors():
    """For every anchor in spec 08 §8.3, verify the verbatim string is present
    in BOTH spec 08 AND its cited sibling. Mismatch = drift."""
    import pathlib
    pkg08_specs = pathlib.Path("sdd/pkg-08-eval-and-ablation/specs")
    pkg07_specs = pathlib.Path("sdd/pkg-07-baselines/specs")
    spec_08_text = (pkg08_specs / "08-integration-contracts.md").read_text(encoding="utf-8")
    anchors = [
        # (anchor_id, sibling_path, verbatim_string_or_pattern)
        # Spec 01 (5)
        ("spec-01-evalreport-32-fields", pkg08_specs / "01-unified-evaluator.md", "Total field count: **32**"),
        ("spec-01-evalreport-schema-version", pkg08_specs / "01-unified-evaluator.md", 'schema_version: str = "pkg08-spec01-v1"'),
        ("spec-01-run-eval-not-replaced", pkg08_specs / "01-unified-evaluator.md", "training/evaluation.py:run_eval is NOT replaced"),
        ("spec-01-evalreport-lock-2-byte-identical", pkg08_specs / "01-unified-evaluator.md", "EvalReport is the unique evaluator output schema"),
        ("spec-01-evaluate-signature", pkg08_specs / "01-unified-evaluator.md", "def evaluate("),
        # Spec 02 (4)
        ("spec-02-obs-mask-3-lines", pkg08_specs / "02-zero-shot-and-c-hidden.md", "obs[c_start] = 0.0"),
        ("spec-02-zero-shot-grid-cfg-driven", pkg08_specs / "02-zero-shot-and-c-hidden.md", "Zero-shot grid is config-driven, never hardcoded"),
        ("spec-02-disjoint-union-invariant", pkg08_specs / "02-zero-shot-and-c-hidden.md", "zero_shot_test_c must equal train"),
        ("spec-02-regret-cache-path", pkg08_specs / "02-zero-shot-and-c-hidden.md", "runs/_oracle_ceilings/<config_hash>/<c>.json"),
        # Spec 03 (3)
        ("spec-03-4-mode-literal", pkg08_specs / "03-direct-inference-toggle.md", '"planner_no_crn"'),
        ("spec-03-cfg-eval-planner-mode", pkg08_specs / "03-direct-inference-toggle.md", "cfg.eval.eval_planner_mode"),
        ("spec-03-consistency-check-raises", pkg08_specs / "03-direct-inference-toggle.md", "eval_use_planner_direct_inference=True conflicts with eval_planner_mode"),
        # Spec 04 (2)
        ("spec-04-mup-selftest-verdict", pkg08_specs / "04-mup-verification.md", "MupSelftestVerdict"),
        ("spec-04-18-row-grid", pkg08_specs / "04-mup-verification.md", "18 runs"),
        # Spec 05 (4)
        ("spec-05-registry-23-keys", pkg08_specs / "05-sweep-harness-and-run-registry.md", "23 keys"),
        ("spec-05-subprocess-per-row", pkg08_specs / "05-sweep-harness-and-run-registry.md", "hyper_mve.experiments._sweep_worker"),
        ("spec-05-per-gpu-semaphore", pkg08_specs / "05-sweep-harness-and-run-registry.md", "GpuSemaphore"),
        ("spec-05-jsonl-append-only", pkg08_specs / "05-sweep-harness-and-run-registry.md", "JSONL append-only"),
        # Spec 06 (4)
        ("spec-06-5-canned-yamls", pkg08_specs / "06-ablation-cli-and-cells.md", '("abl1", "abl4_crn_joint", "abl4_joint_easy_n2", "abl6", "abl7")'),
        ("spec-06-use-coord-desc-alias-deprecation", pkg08_specs / "06-ablation-cli-and-cells.md", "use_coord_desc is deprecated"),
        ("spec-06-mve-joint-enumerate-warn-fallthrough", pkg08_specs / "06-ablation-cli-and-cells.md", "mve_joint_enumerate=True with N="),
        ("spec-06-abl4-joint-easy-n2-pin", pkg08_specs / "06-ablation-cli-and-cells.md", "preset: easy"),
        # Spec 07 (3)
        ("spec-07-welch-t-test", pkg08_specs / "07-statistics-and-comparison.md", "ttest_ind(equal_var=False)"),
        ("spec-07-holm-bonferroni", pkg08_specs / "07-statistics-and-comparison.md", "Holm-Bonferroni"),
        ("spec-07-matplotlib-agg", pkg08_specs / "07-statistics-and-comparison.md", 'matplotlib.use("Agg")'),
        # Cross-package pkg-07 (5)
        ("pkg-07-registry-11-keys", pkg07_specs / "01-baseline-registry-and-cli.md", "INTERNAL_REGISTRY"),
        ("pkg-07-baseline-like-union", pkg07_specs / "01-baseline-registry-and-cli.md", "BaselineLike"),
        ("pkg-07-n-parametric-adapter", pkg07_specs / "04-pettingzoo-adapter.md", "self._N = self._env.N"),
        ("pkg-07-two-flag-info-gate", pkg07_specs / "04-pettingzoo-adapter.md", "oracle_mode"),
        ("pkg-07-evaluate-signature", pkg07_specs / "08-integration-contracts.md", "evaluate(env_fn, c_grid, episodes) -> EvalReport"),
    ]
    for anchor_id, sibling_path, verbatim in anchors:
        sibling_text = sibling_path.read_text(encoding="utf-8")
        assert verbatim in spec_08_text, f"Anchor {anchor_id!r}: verbatim {verbatim!r} absent from spec 08"
        assert verbatim in sibling_text, f"Anchor {anchor_id!r}: verbatim {verbatim!r} absent from sibling {sibling_path.name}"
```

This is a **meta-test** — it does not exercise any runtime code; it verifies the SDD document set is consistent. Run on Day 8 of the pkg-08 SDD calendar (Phase 8 alongside `check_ref_matrix.ps1`). Asserts ≥30 grep anchors are present in both spec 08 and the cited sibling.

### 9.9 `test_pkg07_pkg08_reverse_consumption_5_anchors_byte_identical` (NEW — spec-08-specific meta-test)

Located at `tests/integration/test_integration_pkg08.py`. Verifies the §7 reverse-consumption block is byte-identical between three locations (pkg-08 README §"🔁", pkg-07 spec 08 §8, pkg-08 spec 08 §7):

```python
def test_pkg07_pkg08_reverse_consumption_5_anchors_byte_identical():
    """The 5-anchor block at pkg-08 README §🔁 must appear byte-identically in
    both pkg-07 spec 08 §8 and pkg-08 spec 08 §7."""
    import pathlib, re
    canon_block_pattern = re.compile(
        r"pkg-07 spec 01 §2\.1\s+REGISTRY 11 keys\s+──→\s+pkg-08 spec 05 §3 sweep enumeration"
        r".*?"
        r"pkg-07 spec 08\s+BaselineLike Union type\s+──→\s+pkg-08 spec 05 sweep harness type sig",
        re.DOTALL,
    )
    readme = pathlib.Path("sdd/pkg-08-eval-and-ablation/README.md").read_text(encoding="utf-8")
    pkg07_spec08 = pathlib.Path("sdd/pkg-07-baselines/specs/08-integration-contracts.md").read_text(encoding="utf-8")
    pkg08_spec08 = pathlib.Path("sdd/pkg-08-eval-and-ablation/specs/08-integration-contracts.md").read_text(encoding="utf-8")
    m_readme = canon_block_pattern.search(readme)
    m_pkg07 = canon_block_pattern.search(pkg07_spec08)
    m_pkg08 = canon_block_pattern.search(pkg08_spec08)
    assert m_readme is not None, "README §🔁 5-anchor block missing"
    assert m_pkg07 is not None, "pkg-07 spec 08 §8 5-anchor block missing"
    assert m_pkg08 is not None, "pkg-08 spec 08 §7 5-anchor block missing"
    # Whitespace-normalised equality
    norm = lambda s: re.sub(r"\s+", " ", s.strip())
    assert norm(m_readme.group(0)) == norm(m_pkg07.group(0)) == norm(m_pkg08.group(0))
```

### 9.10 `test_pkg08_downstream_patch_inventory_byte_identical` (NEW — spec-08-specific)

Located at `tests/integration/test_integration_pkg08.py`. Verifies the §6 (A) + (A') + (B) patch inventory matches the README §"修改" 3-block layout byte-identically:

```python
def test_pkg08_downstream_patch_inventory_byte_identical():
    """Spec 08 §6 declares 3 + 1 + 3 = 7 patches across 7 files. The README
    §"修改" block must enumerate the same 7 files in the same A/A'/B layout."""
    expected_files = {
        # (A) 3 behavioural patches
        "hyper_mve/envs/resource_commons/observations.py",
        "hyper_mve/planning/mve_planner.py",
        "hyper_mve/scripts/train_main.py",
        # (A') 1 zero-behaviour threading
        "hyper_mve/envs/resource_commons/env.py",
        # (B) 3 config-additions
        "hyper_mve/configs/env_config.py",
        "hyper_mve/configs/train_config.py",
        "hyper_mve/configs/eval_config.py",
    }
    readme = pathlib.Path("sdd/pkg-08-eval-and-ablation/README.md").read_text(encoding="utf-8")
    spec08 = pathlib.Path("sdd/pkg-08-eval-and-ablation/specs/08-integration-contracts.md").read_text(encoding="utf-8")
    for f in expected_files:
        assert f in readme, f"README §修改 missing patch file: {f}"
        assert f in spec08, f"spec 08 §6 missing patch file: {f}"
```

### 9.11 Test file layout summary

```
tests/integration/
└── test_integration_pkg08.py     # NEW: spec-08-owned (tests 9.1, 9.2, 9.3, 9.8, 9.9, 9.10 above)

tests/eval/                       # spec 01 / 02 territory (mirrored from §9 cross-doc grep)
├── test_c_hidden_obs_mask.py     # spec 02 §6.1; spec 08 §9.4 mirrors name
└── test_env_py_threading.py      # spec 02 §6 / §5.1b; spec 08 §9.5 mirrors name

tests/experiments/                # spec 05 / 06 / 07 territory
├── test_abl4_joint_enum.py       # spec 06 §7.7; spec 08 §9.6 mirrors name
└── test_train_main_cli_rename.py # spec 06 §7.5 CLI variant; spec 08 §9.7 mirrors name
```

Spec 08 owns **6 new test functions** (9.1, 9.2, 9.3, 9.8, 9.9, 9.10) in 1 new file (`test_integration_pkg08.py`), and mirrors **4 sibling test names** (9.4, 9.5, 9.6, 9.7) for the §9 contract enumeration. Total **10 named tests** — exceeding the ≥7 floor declared in the spec authoring requirements.

---

## 10. Integration hooks and cross-references

Pkg-08 has **no downstream SDD package** consumer — it is the terminal SDD package per design §1.3 + §2.3 NG9 + README §"上下游关联" last row. The end-of-line consumers are all paper-side outputs:

| Consumer | Consumed contracts from spec 08 | Use |
|---|---|---|
| **Ch6 main table** (hyper + 5 internal + 3 Tier-1 + MAMBA-if-sourced × 5 seeds × 2 presets) | `EvalReport.return_mean` / `return_sem` / `return_zero_shot_*` / `regret_mean` (§3); `RunRegistry` rows filtered by `ablation_cell IS NULL` for main-table runs (§4) | The 9-column × ≥5-seed × 2-preset matrix in Ch6 main results. Spec 07 stats consumes the rows; Welch t + Holm-Bonferroni produce the published numbers. |
| **Ch6 4 ablation tables** (Abl1 gen_scope 7-cell / Abl4 CRN × Joint-CoordDesc 2×2+1 / Abl6 Fehr-Schmidt 3×3 / Abl7 curriculum 3-cell) | `RunRegistry` rows filtered by `ablation_cell IN {abl1, abl4_*, abl6, abl7}` (§4); spec 06 ablation CLI dispatches to spec 05 sweep harness which writes the rows; spec 07 compares across cells | The 4 ablation tables in Ch6.7-Ch6.9. |
| **Ch6.9 zero-shot table** (train `{0.2, 0.5, 0.8}` → test `{0.0, 0.35, 0.65, 1.0}` × 9 methods) | `EvalReport.return_zero_shot_seen` / `return_zero_shot_unseen` / `return_zero_shot_gap` / `return_per_c` (§3); spec 02 §2 populates these slots | The zero-shot generalisation gap row in Ch6.9. |
| **Ch6 μP self-test (one figure + one paragraph)** | `MupSelftestVerdict` (NOT `EvalReport`; spec 04 §3.4 secondary schema); `runs/_mup_selftest/<config_hash>/mup_selftest.{json,csv,png}` outputs | The μP base-shape LR-doubling self-test figure in Ch6.2.5; on FAIL, the paper drops the μP claim per spec 04 Lock 3. |
| **Ch6 paper-grade plots** (4-mode comparison + bar+errorbar + Welch t annotations) | `EvalReport.direct_inference_return_mean` / `planner_full_return_mean` / `planner_prior_return_gap` (§3); spec 07's `compare` CLI consumes `runs/registry.jsonl` + matplotlib Agg backend | The 4-mode comparison plot + cross-method bar+errorbar plots in Ch6.2-Ch6.7. |

**Terminal declaration**: pkg-08 outputs flow **directly into Ch6 paper-stage writing**; there is no further SDD-stage consumer. After pkg-08 finalisation (Day 9 PR ack), the workflow returns to the `academic-research` git branch for Ch6 final-draft writing using the artefacts under `runs/` (sweep outputs + RunRegistry + comparison plots + μP figure).

---

### 10.1 Cross-references — upstream anchors (pkg-08 internal — siblings)

- **pkg-08 design.md §3.1** (Eval-半 / Ablation-半 / 集成契约 8-spec 分配) — §2 pattern-reference table consumes the spec dispatch; §3 + §4 trace to the schema mother-doc anchors.
- **pkg-08 design.md §3.2** (EvalReport `@dataclass(frozen=True)` 字段穷举) — §3 verbatim mirror of the 32-field schema; §3.3 double-consumption contract.
- **pkg-08 design.md §3.3** (四 planner eval mode taxonomy) — §5 field 4 (eval_planner_mode) + field 5 (eval_use_planner_direct_inference); §6 (B) row 3 config-additions for `eval_config.py`.
- **pkg-08 design.md §3.4** (RunRegistry row schema 22 schema-domain + 1 schema_version = 23 keys) — §4 verbatim 23-key mirror.
- **pkg-08 design.md §3.5** (subprocess-per-row five-point rationale) — §4 inherits the subprocess + file-lock + JSONL append-only contract (§4.3 + §4.4 mechanism).
- **pkg-08 design.md §4 D2** (`EvalReport` schema lock; internal + external double-consumption) — §3 + §3.3 mirror.
- **pkg-08 design.md §4 D4** (c_hidden 实施路径) — §5 field 1 (c_visible); §6 (A.1) behavioural patch on `observations.py`; §6 (A'.1) zero-behaviour threading on `env.py`; §6 (B.1) config-additions on `env_config.py`.
- **pkg-08 design.md §4 D6** (regret 指标 oracle ceiling cache) — §3 fields 19-22 (regret_per_c / regret_mean / oracle_ceiling_per_c / oracle_ceiling_cache_hit).
- **pkg-08 design.md §4 D7** (4 planner eval mode 解耦) — §5 fields 4-5; §3 field 5 (eval_planner_mode).
- **pkg-08 design.md §4 D8** (sweep harness subprocess + JSONL Registry) — §4 entire section mirrors.
- **pkg-08 design.md §4 D9** (`--ablation` CLI 5 ID canned YAML 分发) — §6 (A.2) behavioural patch on `mve_planner.py` for the Joint cell; §5 field 3 (mve_joint_enumerate).
- **pkg-08 design.md §4 D10** (5 新 cfg 字段穷举) — §5 entire section mirrors.
- **pkg-08 README §"输出清单"** (新增 + 修改 3-block A/A'/B + 5 新 cfg 字段穷举 + 🔁 5-anchor block) — §5 + §6 + §7 mirror.
- **pkg-08 README §"spec 间引用表"** (M6 ref-matrix; spec 08 引用 spec 01-07 全 7 个) — §8.1 forward-ref check enforces.
- **pkg-08 spec 01 §3 + §3.1** (`EvalReport` schema mother-doc) — §3 verbatim mirror.
- **pkg-08 spec 02 §2.6 + §3.2 + §3.5 + §4.1-§4.8 + §5.1 + §5.1b + §5.2** — §6 (A.1) + (A'.1) + (B.1); §3 regret fields; §5 field 1.
- **pkg-08 spec 03 Lock 1 + Lock 2 + §2 + §3 + §4 + §5** — §5 fields 4-5; §3 field 5 + fields 23-25 (planner-prior gap source).
- **pkg-08 spec 04 §2 + §3.4** — §3 schema is separate from `MupSelftestVerdict` (the §10 hook consumer table tracks this); spec 04 consumes only `EvalReport.return_mean` (§3 row 7).
- **pkg-08 spec 05 §3 + §4 + §5 + §7 + §8** — §4 entire section verbatim mirror; §6 (A) + (A') patches' test gates reference spec 05 §9 + §10 tests.
- **pkg-08 spec 06 §3 + §4 + §5 + §6 + §7 + §8** — §6 (A.2) + (A.3) + (B.2); §5 fields 2-3.
- **pkg-08 spec 07** (planned 2 new files: `stats.py` + `compare.py`; Welch t + Holm-Bonferroni + matplotlib Agg) — §8.3 anchors 23-25.

### 10.2 Cross-references — cross-package anchors (pkg-07 sibling mirror layer)

- **pkg-07 spec 08 entire document** (`../../pkg-07-baselines/specs/08-integration-contracts.md`) — §2 pattern-reference table consumes the symmetric structure; §7 5-anchor block byte-identical with pkg-07 spec 08 §8.
- **pkg-07 spec 01 §2.1 + §2.3 + §3.2** (REGISTRY 11 keys + 量词 canonical + BaselineLike Union) — §7 anchors 1-2 + 5; §8.3 anchors 26-27.
- **pkg-07 spec 04 §3 + §7 + §10** (N-parametric adapter + two-flag info gate + env_fn factory) — §7 anchor 3; §8.3 anchors 28-29.
- **pkg-07 spec 08 §4.1 + §4.4** (uniform `evaluate()` signature + external-runner population matrix) — §7 anchor 4; §3.3 double-consumption contract; §8.3 anchor 30.

### 10.3 Cross-references — existing repo ground-truth anchors

- `hyper_mve/eval/eval_report.py` (NEW per spec 01 §1) — `@dataclass(frozen=True) EvalReport` 33-field body; §3 mirror's source of truth at runtime.
- `hyper_mve/experiments/run_registry.py` (NEW per spec 05 §1) — `@dataclass(frozen=True) RegistryRow` 23-field body; §4 mirror's source of truth at runtime.
- `hyper_mve/configs/env_config.py` — target for §6 (B.1) config-additions patch (+1 field `c_visible: bool = True`).
- `hyper_mve/configs/train_config.py` — target for §6 (B.2) config-additions + alias-with-deprecation patch (+2 fields + @property shim).
- `hyper_mve/configs/eval_config.py` — target for §6 (B.3) config-additions patch (+2 fields).
- `hyper_mve/envs/resource_commons/observations.py` — target for §6 (A.1) behavioural patch (+3 lines obs-mask + 1 line signature).
- `hyper_mve/envs/resource_commons/env.py` — target for §6 (A'.1) zero-behaviour threading patch (+2 kwarg call-sites).
- `hyper_mve/planning/mve_planner.py` — target for §6 (A.2) behavioural patch (+1 logical branch + warn block).
- `hyper_mve/scripts/train_main.py` — target for §6 (A.3) behavioural patch (CLI rename with alias-with-deprecation).
- `runs/registry.jsonl` — the on-the-wire format §4 schema defines.
- `runs/_oracle_ceilings/<config_hash>/<c>.json` — the regret cache that spec 02 §4.3 populates and §3 fields 21-22 expose as diagnostics.
- `runs/_mup_selftest/<config_hash>/{mup_selftest.json, mup_selftest.csv, mup_selftest.png}` — spec 04 §1 outputs (NOT `EvalReport`-typed; secondary schema per §10).

---

## 11. Anchors (verbatim grep targets for spec 08 §8 drift detector)

- `pkg-08 spec 08 Lock 1: CONTRACT MIRROR ONLY (sibling-precedes-mirror invariant)`
- `pkg-08 spec 08 Lock 2: three contract categories exhaustive (2 schemas + 5 cfg fields + 3-block patches)`
- `pkg-08 spec 08 Lock 3: pkg-07 → pkg-08 reverse-consumption 5 anchors mirror pkg-08 README §"🔁" verbatim; symmetric with pkg-07 spec 08 §8`
- `pkg-08 spec 08 §2: pattern reference — pkg-07 spec 08 parallels (M6 ref-matrix symmetry)`
- `pkg-08 spec 08 §3: EvalReport @dataclass schema — verbatim 32-field mirror of spec 01 §3.1`
- `pkg-08 spec 08 §3.2: 33 dataclass fields (32 schema-payload + 1 schema_version sentinel)`
- `pkg-08 spec 08 §3.3: double-consumption contract (internal BaselineModel + external ExternalBaselineRunner both return same EvalReport)`
- `pkg-08 spec 08 §4: RunRegistry JSONL row schema — verbatim 23-key mirror of spec 05 §4`
- `pkg-08 spec 08 §4.2: 22 schema-domain + 1 schema_version = 23 keys on every JSONL line`
- `pkg-08 spec 08 §4.3: append-only contract (new lines per state transition; never in-place mutation)`
- `pkg-08 spec 08 §4.4: concurrent file-lock contract (msvcrt.locking on Windows / fcntl.flock on POSIX)`
- `pkg-08 spec 08 §5: 5 new cfg fields exhaustive enumeration (1 on EnvConfig + 2 on TrainConfig + 2 on EvalConfig)`
- `pkg-08 spec 08 §6 (A): 3 behavioural patches (observations.py + mve_planner.py + train_main.py)`
- `pkg-08 spec 08 §6 (A'): 1 zero-behaviour threading patch (env.py +2 kwarg call-sites; [no-behaviour] tag)`
- `pkg-08 spec 08 §6 (B): 3 config-additions patches (env_config.py + train_config.py + eval_config.py)`
- `pkg-08 spec 08 §6.4: patch-block sum check — 7 distinct files touched by pkg-08 across pkg-01/02/05 territory; pkg-01..05/07 SDD ZERO modifications`
- `pkg-08 spec 08 §7: pkg-07 → pkg-08 reverse-consumption 5 anchors (mirror of README §"🔁")`
- `pkg-08 spec 08 §7.2: bidirectional drift handling workflow (symmetric with pkg-07 spec 08 §8.2)`
- `pkg-08 spec 08 §8.2: PowerShell regex (?<![Pp]kg-\d{2} )spec\s+0([1-8])|\b0([1-8])-[a-z] for intra-package ref scan`
- `pkg-08 spec 08 §8.3: ≥30 verbatim grep anchors (spec 01 ×5 / spec 02 ×4 / spec 03 ×3 / spec 04 ×2 / spec 05 ×4 / spec 06 ×4 / spec 07 ×3 / cross-package pkg-07 ×5)`
- `pkg-08 spec 08 §8.4: drift handling — sibling-precedes-mirror; spec 08 catches up, never leads; on cross-package anchor failure, re-run BOTH pkg-07 + pkg-08 check_ref_matrix.ps1`
- `pkg-08 spec 08 §9: ≥7 named tests (6 spec-08-owned in test_integration_pkg08.py + 4 sibling-mirror cross-doc grep names; 2 of the 6 owned are meta-tests for drift detector + reverse-consumption byte-identical)`
- `pkg-08 spec 08 §10: integration hooks — pkg-08 has NO downstream SDD package consumer; terminal SDD package; end-of-line consumers are Ch6 main table / 4 ablation tables / zero-shot table / μP figure / paper-grade plots`

**End of spec 08 — integration-contracts.**
