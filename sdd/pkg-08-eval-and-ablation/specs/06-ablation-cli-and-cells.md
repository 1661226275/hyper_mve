# Spec 06 — Ablation CLI + Canned Cells (5 YAMLs; `use_coord_desc → randomize_order` Rename; `mve_joint_enumerate` Pkg-05 Patch)

> **Parent docs**: [`../proposal.md`](../proposal.md) · [`../design.md`](../design.md) §3.1 / §3.3 / §4 D9 / §4 D10 · [`../README.md`](../README.md) C8-ABL-CLI1 / C8-ABL-ABL4A / C8-ABL-RENAME1 + §"5 新 cfg 字段穷举" rows 2–3
> **Upstream consumed**: [`./05-sweep-harness-and-run-registry.md`](./05-sweep-harness-and-run-registry.md) §3 (`SweepConfig` dataclass + `from_yaml` + cartesian enumerator) + §4 (`RunRegistry` 23-key row schema) + §10.4 (sweep harness ↔ ablate CLI dispatch contract) · [`./02-zero-shot-and-c-hidden.md`](./02-zero-shot-and-c-hidden.md) Lock 2 (cfg-driven `zero_shot_*` grid) · [`./03-direct-inference-toggle.md`](./03-direct-inference-toggle.md) Lock 1 + §3 (4 planner eval modes; `eval_planner_mode="planner_full"` default) · [`./01-unified-evaluator.md`](./01-unified-evaluator.md) §3 (`EvalReport` 32-field schema — every ablation row emits exactly one report)
> **Cross-refs**: [`./07-statistics-and-comparison.md`](./07-statistics-and-comparison.md) (Welch t consumes ablation rows for 2-method comparison; spec 07 §3 `compare` CLI joins `ablation_cell` groupings) · [`./08-integration-contracts.md`](./08-integration-contracts.md) §5 (canonical "下游补丁声明" — codifies 3 downstream patches declared by THIS spec) · [`../../pkg-07-baselines/specs/01-baseline-registry-and-cli.md`](../../pkg-07-baselines/specs/01-baseline-registry-and-cli.md) §2.2 (`oracle_only` / `mixed` / `infer_only` curriculum-override variants consumed by `abl7_curriculum.yaml`) · [`../../docs/Review_v4_TheoryAudit_2026-06.md`](../../docs/Review_v4_TheoryAudit_2026-06.md) §M8 (Abl4 redefinition) + §Q2 (`use_coord_desc → randomize_order` rename) + §2.2 (Abl1 gen_scope 7-cell) + §2.4 (Joint enum motivation) · pkg-05 `hyper_mve/planning/mve_planner.py` (downstream patch site) · pkg-05 `hyper_mve/scripts/train_main.py` (CLI rename site)
> **Status**: SDD only — describes the contract for `hyper_mve/experiments/ablate.py` + the five YAML files under `hyper_mve/experiments/ablations/`. No production code changes here; this spec **DECLARES** three downstream code patches (codified in spec 08 §5) but does not codify them.

---

## ⚠️ Header — three hard locks

### Lock 1 — Five canned YAMLs ≠ four logical ablation cells; the 5/4 asymmetry is locked

This spec ships **5 canned YAML dispatch IDs** under `hyper_mve/experiments/ablations/*.yaml`, but these IDs cover **4 logical ablation cells**. The asymmetry is locked because Abl4 (CRN × Joint/CoordDesc, Theory Audit §M8) dispatches to two YAMLs: `abl4_crn_joint.yaml` (the 2×2 CRN × CoordDesc main matrix at Medium preset) and `abl4_joint_easy_n2.yaml` (the Easy N=2 exhaustive 6²=36 enumeration child, which is mathematically tractable only at N=2 and therefore lives in its own file). The remaining three logical cells — Abl1 gen_scope (7 cells), Abl6 Fehr-Schmidt (3×3 α/β), Abl7 curriculum (3 cells) — each occupy one YAML.

The canonical wording lives in [`../README.md`](../README.md) §"量词澄清" ("4 逻辑 cells = 5 分发 ID"). This spec mirrors that wording verbatim and pins it as Lock 1. **Any future addition of a 6th YAML must be accompanied by a sibling ablation logical cell — 5/4 cannot drift to 6/4 without explicit decision recorded in design.md §4 D9 and synchronously edited into this Lock 1.** Conversely, removing one of the 5 YAMLs requires a synchronous edit of design.md §4 D9, README §"量词澄清", and this Lock 1; the canned-YAML inventory and the logical-cell inventory are joined at the hip.

A drift detector test `test_canned_yaml_inventory_matches_design` (§7) globs `hyper_mve/experiments/ablations/*.yaml` and asserts the five-file set is exactly `{abl1_gen_scope, abl4_crn_joint, abl4_joint_easy_n2, abl6_fehr_schmidt, abl7_curriculum}`; a separate test `test_ablation_cli_dispatch_all_5` (§7) parametrises over the 5 CLI IDs and asserts each resolves to a `SweepConfig`. The two tests together enforce both halves of the 5/4 contract.

### Lock 2 — `use_coord_desc → randomize_order` rename is alias-WITH-deprecation, not a breaking rename

The Theory Audit Q2 rename `cfg.train.use_coord_desc → cfg.train.randomize_order` is implemented as a **backwards-compatible alias-with-deprecation**, NOT as a breaking field rename. The canonical field is `cfg.train.randomize_order: bool = True`. The legacy name `cfg.train.use_coord_desc` continues to work via a `@property` shim on `TrainConfig` that returns `self.randomize_order` and emits `DeprecationWarning` on access; the corresponding `@use_coord_desc.setter` writes to `self.randomize_order` similarly and warns. Both READ and WRITE paths warn, so any latent reader in the codebase (or in user YAML overrides) surfaces as a deprecation warning rather than a silent `AttributeError`.

The named test `test_use_coord_desc_alias_warns` (per README C8-ABL-RENAME1) is **parametrised over READ and WRITE paths** to enforce both halves: the READ test accesses `cfg.train.use_coord_desc` after constructing a default `V4Config()` and asserts `DeprecationWarning` is emitted; the WRITE test calls `dataclasses.replace(cfg.train, use_coord_desc=False)` (or the equivalent setter path on the live instance) and asserts both `DeprecationWarning` and that the resulting `cfg.train.randomize_order` equals `False`. The implementation note (§4.1 below) describes the property-shim mechanic; spec 08 §5 codifies the patch.

The default value `randomize_order: bool = True` matches the legacy `use_coord_desc: bool = True` default at `hyper_mve/configs/train_config.py` line 88 verbatim, so no production run, YAML config, or checkpoint loaded from disk changes behaviour. The alias is preserved for **at least one release** (no removal timeline pinned in this spec); future removal would require an explicit decision in design.md §4 D10 and a synchronous edit to this Lock 2.

### Lock 3 — `mve_joint_enumerate=True` is mathematically tractable ONLY at Easy N=2 (6²=36); the YAML hard-pins preset=easy and the planner warns-with-fallthrough at N>2

The new field `cfg.train.mve_joint_enumerate: bool = False` (declared in §6 below) controls whether the MVE planner's agent-order-permutation block does **full joint enumeration** over all `A^N` joint actions (`itertools.product(range(A), repeat=N)`) instead of the default coordinate-descent agent-order pass. At `A=6` (the env's action cardinality per `EnvConfig.A`) and N=2 (Easy preset), this is 6²=36 candidates — tractable. At N=4 (Medium preset), it is 6⁴=1,296 — feasible but no longer a meaningful "exhaustive" benchmark vs coord-descent. At N=8 (Hard preset), it is 6⁸=1,679,616 — **intractable** within the per-row time budget and would silently hang the sweep.

The YAML `abl4_joint_easy_n2.yaml` (§3.3 below) **HARD-PINS `preset: easy`** so the dispatch path can only legally enumerate at N=2. To defend against malicious YAML overrides or user-supplied SweepConfig that flip `mve_joint_enumerate=True` at N>2, **the planner emits `RuntimeWarning` and falls through to coordinate-descent**:

```python
# Spec 06 §5.2 (codified by spec 08 §5 in mve_planner.py)
if self.mve_joint_enumerate and self.N > 2:
    warnings.warn(
        f"mve_joint_enumerate=True with N={self.N} would enumerate "
        f"{self.A}^{self.N}={self.A**self.N} joint actions (intractable); "
        f"falling through to coord-descent",
        RuntimeWarning,
    )
    # fall through to standard coord-descent path
```

The warn-with-fallthrough policy is the **locked policy** so YAML overrides cannot accidentally hang the sweep harness (which would corrupt the `RunRegistry` JSONL — a hung subprocess never emits a `status="completed"` row). Raising `RuntimeError` was considered and rejected because it would mark the entire sweep row as `status="failed"` and require manual `--retry-failed` intervention; warning-with-fallthrough preserves Sweep-row throughput while still surfacing the intent-mismatch to stderr.

A named test `test_mve_joint_enumerate_medium_falls_through_to_coord_desc` (§7) constructs a synthetic `V4Config` with `env.N=4` (Medium) and `train.mve_joint_enumerate=True`, calls the planner, and asserts both that `RuntimeWarning` is raised and that the resulting `pi_mve` is identical to the coord-descent baseline.

---

## 1. Purpose

Three reviewer-facing motivations make this spec necessary, each binding a paper claim to an executable ablation cell.

**Motivation 1 — backing the paper's "generalisation scope" claim (Abl1 gen_scope).** Per Theory Audit §2.2 + Ch6.9, the paper claims hyper-MuZero generalises across several dimensions of distribution shift: within-distribution (re-evaluation of training c), zero-shot interpolation (`c=0.35/0.65` inside train hull), zero-shot extrapolation (`c=0.0/1.0` outside hull), c_hidden (no c on observation), agent-type-ratio shift (Bell-curve from majority-α to majority-β), and combined c_hidden × type-shift. The Abl1 gen_scope ablation runs the same trained model (hyper) across 7 generalisation cells, with 5 internal baselines as control, producing the headline grid for §6.9. Without this cell the paper cannot defend the claim that the hyper representation transfers; spec 06 ships `abl1_gen_scope.yaml` to materialise the grid via the sweep harness.

**Motivation 2 — backing the "CRN at step 0 is load-bearing" + "coord-descent agrees with joint optimum at small N" claims (Abl4 CRN × Joint/CoordDesc).** Per Theory Audit §M8 + §Q2 + §2.4 + DESIGN_DOC_FINAL.md §4.1 + §5.8, the v4.6 algorithm relies on two ingredients: (i) Common Random Numbers (CRN) at step 0 of the MVE rollout, and (ii) coordinate descent over a random agent ordering. The paper claims each ingredient is load-bearing: removing CRN should collapse `π_mve` toward uniform (signal-to-noise collapse), and removing coord-descent should produce a measurable but smaller drop (single-ordering vs joint search). Abl4 trains 4 cells in a 2×2 grid (CRN-on/off × CoordDesc-on/off) to surface both effects empirically; the sub-cell `abl4_joint_easy_n2` adds a third orthogonal check at Easy N=2 — exhaustive 6²=36 joint enumeration vs coord-descent — to verify coord-descent agrees with the joint optimum on the only preset where exhaustive search is tractable. Together these YAMLs back both Theory Audit assertions.

**Motivation 3 — backing the Fehr-Schmidt sensitivity + curriculum staging claims (Abl6 + Abl7).** Ch6 reviewers will ask: how sensitive is the trained policy to the Fehr-Schmidt α/β inequality-aversion parameters (does a small α/β change collapse the result)? And how necessary is the three-stage curriculum (does training oracle-only or infer-only produce a comparable result)? Abl6 ships a 3×3 α/β scan (`abl6_fehr_schmidt.yaml`) to surface the sensitivity surface; Abl7 ships a 3-cell curriculum sweep (`oracle_only / mixed / infer_only`) per pkg-07 spec 01 §2.2 to surface the curriculum-staging dependence. Without these cells the paper has no defensible answer to the sensitivity / necessity questions.

The deliverable is **two files + five YAMLs**:

```
hyper_mve/experiments/ablate.py                       — thin --ablation <id> CLI dispatcher
hyper_mve/experiments/ablations/abl1_gen_scope.yaml   — 7-cell generalisation-scope grid
hyper_mve/experiments/ablations/abl4_crn_joint.yaml   — 2×2 CRN × CoordDesc main matrix (Medium)
hyper_mve/experiments/ablations/abl4_joint_easy_n2.yaml — Easy N=2 exhaustive 36-action enum child
hyper_mve/experiments/ablations/abl6_fehr_schmidt.yaml — 3×3 α/β sensitivity scan
hyper_mve/experiments/ablations/abl7_curriculum.yaml   — 3-cell oracle_only/mixed/infer_only sweep
```

The downstream patches (declared in §8, codified in spec 08 §5):
- `hyper_mve/configs/train_config.py` — +2 fields (`randomize_order` + `mve_joint_enumerate`) + `@property` alias for `use_coord_desc` with `DeprecationWarning`. [config-additions + alias-with-deprecation]
- `hyper_mve/planning/mve_planner.py` — +1 if-branch line for `mve_joint_enumerate` consumption + warn-with-fallthrough at N>2. [behavioural; +1 line + warn block]
- `hyper_mve/scripts/train_main.py` — argparse argument rename `--use_coord_desc → --randomize_order` with deprecated alias preserved. [downstream-cli; alias-with-deprecation]

No other files are touched by this spec. The sweep harness (spec 05) is consumed transparently via `SweepConfig.from_yaml(...) → run_sweep(sweep_cfg)`; the unified evaluator (spec 01) is consumed via the worker subprocess's `unified_evaluator.evaluate(runner, env_fn, cfg)` call which emits one `EvalReport` per row.

---

## 2. The `ablate` CLI surface

### 2.1 Argparse signature

`python -m hyper_mve.experiments.ablate --ablation <id>` is the user-facing entry point. The argparser is declared as follows (locked verbatim; sourced from design §4 D9 + pkg-08 spec 05 §10.4 sketch):

```python
# hyper_mve/experiments/ablate.py
import argparse
import pathlib
from hyper_mve.experiments.sweep import SweepConfig, run_sweep

ABLATION_IDS = ("abl1", "abl4_crn_joint", "abl4_joint_easy_n2", "abl6", "abl7")

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Dispatch a canned ablation cell defined by a YAML under "
            "hyper_mve/experiments/ablations/. Five legal IDs (4 logical cells, "
            "with Abl4 split into two dispatch files per spec 06 Lock 1):\n"
            "  abl1                  — gen_scope 7-cell generalisation grid\n"
            "  abl4_crn_joint        — CRN × CoordDesc 2×2 matrix (Medium)\n"
            "  abl4_joint_easy_n2    — Easy N=2 exhaustive 36-action enum\n"
            "  abl6                  — Fehr-Schmidt α/β 3×3 sensitivity scan\n"
            "  abl7                  — curriculum oracle_only/mixed/infer_only"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ablation", required=True, choices=ABLATION_IDS)
    p.add_argument("--preset", default=None, choices=("easy", "medium", "hard"),
                   help="Override the YAML's default preset; respected only when the YAML does not hard-pin a preset (Lock 3).")
    p.add_argument("--seeds", type=int, default=None,
                   help="Override the YAML's default seed count (n seeds → tuple(0..n-1)).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the materialised SweepConfig and exit 0 without dispatching to run_sweep.")
    p.add_argument("--max-parallel", type=int, default=1)
    p.add_argument("--n-gpus", type=int, default=1)
    p.add_argument("--retry-failed", action="store_true")
    return p.parse_args(argv)
```

The five CLI flags `--max-parallel / --n-gpus / --retry-failed / --dry-run / --preset` are forwarded to `run_sweep` per spec 05 §8.1; `--ablation` and `--seeds` are spec-06-specific. `--preset` and `--seeds` are *overrides* of the YAML default — they apply only when the YAML does not hard-pin the field (Lock 3 hard-pins `preset: easy` in `abl4_joint_easy_n2.yaml`, which makes `--preset` a no-op for that ID; a dispatcher-level guard logs `[WARN] --preset=medium ignored: abl4_joint_easy_n2 hard-pins preset=easy`).

### 2.2 Dispatch table

The CLI is a **thin dispatcher**: it loads the canned YAML, materialises a `SweepConfig`, optionally applies `--preset` / `--seeds` overrides, and invokes `run_sweep`. No parallel sweep harness is introduced. The dispatch table:

| `--ablation` ID | YAML path | Logical cell | Hard-pinned fields |
|---|---|---|---|
| `abl1` | `hyper_mve/experiments/ablations/abl1_gen_scope.yaml` | Abl1 gen_scope (7-cell) | None — `--preset` legal |
| `abl4_crn_joint` | `hyper_mve/experiments/ablations/abl4_crn_joint.yaml` | Abl4 (2×2 CRN × CoordDesc, main matrix) | None — `--preset` legal |
| `abl4_joint_easy_n2` | `hyper_mve/experiments/ablations/abl4_joint_easy_n2.yaml` | Abl4 (Easy N=2 exhaustive enum child) | `preset: easy` (Lock 3) + `mve_joint_enumerate: True` |
| `abl6` | `hyper_mve/experiments/ablations/abl6_fehr_schmidt.yaml` | Abl6 (Fehr-Schmidt 3×3 α/β) | None — `--preset` legal |
| `abl7` | `hyper_mve/experiments/ablations/abl7_curriculum.yaml` | Abl7 (oracle_only/mixed/infer_only) | None — `--preset` legal |

### 2.3 Behaviour on unknown ID

`argparse` rejects unknown IDs via the `choices=ABLATION_IDS` constraint: invoking `python -m hyper_mve.experiments.ablate --ablation abl9` exits 2 with `argparse: error: argument --ablation: invalid choice: 'abl9' (choose from 'abl1', 'abl4_crn_joint', 'abl4_joint_easy_n2', 'abl6', 'abl7')`. Test `test_ablation_cli_rejects_unknown_id` (§7) asserts `SystemExit(2)` is raised.

### 2.4 `--dry-run` semantics

`--dry-run` short-circuits sweep dispatch:

```
[dry-run] Loaded SweepConfig from hyper_mve/experiments/ablations/abl4_crn_joint.yaml
[dry-run] variants=('hyper',), seeds=(0,1,2,3,4), preset=medium, overrides=4 (CRN×CoordDesc 2×2)
[dry-run] eval_planner_mode=planner_full (per spec 03 Lock 1)
[dry-run] ablation_cell_id=abl4_crn_joint
[dry-run] Total rows would be: 1 × 5 × 4 = 20
[dry-run] Exiting without calling run_sweep.
```

The dry-run path constructs the `SweepConfig` and computes the cartesian cardinality but **does not** call `run_sweep`. Test `test_ablation_cli_dry_run_prints_sweepconfig` (§7) monkey-patches `run_sweep` to raise and asserts the call succeeds (raise is not triggered).

### 2.5 YAML resolution

YAML paths are resolved relative to the package install root:

```python
ABLATIONS_DIR = pathlib.Path(__file__).parent / "ablations"
yaml_path = ABLATIONS_DIR / f"{args.ablation}.yaml"
sweep_cfg = SweepConfig.from_yaml(yaml_path)
```

This makes the CLI invariant to the user's current working directory; the YAMLs ship with the package and are versioned in-tree.

---

## 3. The five canned YAMLs — schema + per-cell content

Each YAML below specifies the `SweepConfig` fields it materialises (per spec 05 §3.1 dataclass), the expected row count under default `--seeds` / `--preset` overrides, and the test gate. The order below mirrors the canonical Abl1/Abl4/Abl6/Abl7 enumeration in design §4 D9.

### 3.1 `abl1_gen_scope.yaml` — 7-cell generalisation-scope grid

**Purpose**: surface hyper-MuZero's generalisation behaviour across 7 cells of distribution shift, with 5 internal baselines as control. Backs the paper claim in Ch6.9 + Theory Audit §2.2.

**The 7 cells**:

| Cell name | Distribution shift |
|---|---|
| `within_distribution` | Re-evaluate at training c (seen grid) |
| `zs_interp` | Zero-shot c=0.35 / 0.65 (interpolation; inside train hull) |
| `zs_extrap` | Zero-shot c=0.0 / 1.0 (extrapolation; outside train hull) |
| `c_hidden` | `cfg.env.c_visible=False` (BeliefNet ĉ inference probe) |
| `type_ratio_shift_alpha` | Bell-curve type ratio shifted toward majority-α |
| `type_ratio_shift_beta` | Bell-curve type ratio shifted toward majority-β |
| `both_axes_hidden` | `c_visible=False` AND type-ratio shift (combined) |

**Lock 2 of spec 02 (cfg-driven c-grid) compliance**: the YAML **does not hardcode** `c=0.35` / `c=0.65` / `c=0.0` / `c=1.0` literals. Instead it relies on `cfg.eval.zero_shot_*` reserved fields at `hyper_mve/configs/eval_config.py` lines 37–39 (`zero_shot_train_c = (0.2, 0.5, 0.8)`, `zero_shot_test_c = (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)`, `zero_shot_unseen_c = (0.0, 0.35, 0.65, 1.0)`); the `unified_evaluator.evaluate` consumes these fields per spec 02 §2 and emits the seen/unseen breakdowns in `EvalReport.return_per_c` + `return_zero_shot_*`. The Abl1 YAML's cell labels (`zs_interp` / `zs_extrap`) are post-hoc partition labels applied by spec 07's `compare.py` when generating the gen-scope table; the underlying eval rollout is the same 7-c grid.

**SweepConfig fields**:

```yaml
# hyper_mve/experiments/ablations/abl1_gen_scope.yaml
variants:
  - hyper
  - baseline_input_wide
  - baseline_input_deep
  - baseline_ma_muzero
  - no_belief
  - oracle_only
seeds: [0, 1, 2, 3, 4]
preset: medium
overrides:
  # 7 cells encoded as cfg-overrides; cfg.eval.zero_shot_* drives the c-grid.
  - {legacy.ablation_cell_tag: "within_distribution", env.c_visible: true}
  - {legacy.ablation_cell_tag: "zs_interp",          env.c_visible: true}
  - {legacy.ablation_cell_tag: "zs_extrap",          env.c_visible: true}
  - {legacy.ablation_cell_tag: "c_hidden",           env.c_visible: false}
  - {legacy.ablation_cell_tag: "type_ratio_shift_alpha", env.c_visible: true, eval.bell_curve_majority_type: "alpha"}
  - {legacy.ablation_cell_tag: "type_ratio_shift_beta",  env.c_visible: true, eval.bell_curve_majority_type: "beta"}
  - {legacy.ablation_cell_tag: "both_axes_hidden",   env.c_visible: false, eval.bell_curve_majority_type: "alpha"}
max_steps: 200000
eval_planner_mode: planner_full
max_parallel: 2
n_gpus: 2
ablation_cell_id: abl1_gen_scope
```

**Expected row count**: 6 variants × 5 seeds × 7 cells = **210 rows**.

**Test gate**: `test_abl1_gen_scope_yaml_materialises_7_cells` (§7) loads the YAML, computes `len(enumerate_cartesian(sweep_cfg))`, and asserts `== 210` (under default `--seeds=5`); also asserts each of the 7 distinct `legacy.ablation_cell_tag` values appears in the override list.

### 3.2 `abl4_crn_joint.yaml` — 2×2 CRN × CoordDesc main matrix (Medium)

**Purpose**: surface the load-bearing effect of CRN-at-step-0 and coordinate-descent agent ordering during training. Backs Theory Audit §M8 + §Q2 + DESIGN_DOC_FINAL.md §4.1 + §5.8. The 2×2 matrix is **training-time** axes (per spec 03 Lock 1): `cfg.train.use_crn ∈ {True, False}` × `cfg.train.randomize_order ∈ {True, False}` — note **renamed** field, not `use_coord_desc`. The eval-time evaluation is uniformly `eval_planner_mode="planner_full"` (per spec 03 Lock 1 + §3 row 4) so the training-time effect is surfaced under fair eval.

**SweepConfig fields**:

```yaml
# hyper_mve/experiments/ablations/abl4_crn_joint.yaml
variants:
  - hyper
seeds: [0, 1, 2, 3, 4]
preset: medium
overrides:
  # 2×2 training-time CRN × CoordDesc — renamed field randomize_order, not use_coord_desc
  - {train.use_crn: true,  train.randomize_order: true}    # baseline (CRN on, CoordDesc on)
  - {train.use_crn: false, train.randomize_order: true}    # CRN ablated
  - {train.use_crn: true,  train.randomize_order: false}   # CoordDesc ablated
  - {train.use_crn: false, train.randomize_order: false}   # both ablated
max_steps: 200000
eval_planner_mode: planner_full
max_parallel: 2
n_gpus: 2
ablation_cell_id: abl4_crn_joint
```

**Expected row count**: 1 variant × 5 seeds × 4 cells = **20 rows**.

**Eval mode pin** (spec 03 Lock 1 compliance): the YAML hard-pins `eval_planner_mode: planner_full` at the SweepConfig level so every Abl4 cell, including the CRN-off and CoordDesc-off training cells, evaluates under the same fair `planner_full` policy. Without this pin, a careless future edit could entangle training-time and eval-time effects.

**Test gate**: `test_abl4_crn_joint_yaml_materialises_2x2` (§7) loads the YAML, asserts override count = 4, asserts each override is a dict with both `train.use_crn` and `train.randomize_order` keys (catches the rename: legacy `use_coord_desc` in this YAML would cause `train_main.py` to emit a `DeprecationWarning` on override apply, which is acceptable but not desired in the canonical YAML).

### 3.3 `abl4_joint_easy_n2.yaml` — Easy N=2 exhaustive 36-action enumeration

**Purpose**: empirically verify that coordinate-descent agent-ordering agrees with the joint optimum at the smallest tractable preset. Backs Theory Audit §2.4 + DESIGN_DOC_FINAL.md §5.8. **Lock 3 hard-pins `preset: easy`** because N=2 is the only preset where 6²=36 exhaustive enumeration is mathematically tractable within a per-row time budget (and where `pi_mve` over the joint space is small enough to meaningfully compare against coord-descent's marginal output).

**SweepConfig fields**:

```yaml
# hyper_mve/experiments/ablations/abl4_joint_easy_n2.yaml
variants:
  - hyper
seeds: [0, 1, 2, 3, 4]
preset: easy            # HARD-PIN per spec 06 Lock 3; --preset CLI override is logged-and-ignored
overrides:
  # Single override: enable joint enumeration; coord-descent + CRN unchanged at default True.
  - {train.mve_joint_enumerate: true}
max_steps: 100000       # Easy preset converges faster; lower budget
eval_planner_mode: planner_full
max_parallel: 1
n_gpus: 1
ablation_cell_id: abl4_joint_easy_n2
```

**Expected row count**: 1 variant × 5 seeds × 1 cell = **5 rows**.

**Hard-pin enforcement**: spec 05 §3.1 `SweepConfig.preset: Literal["easy", "medium", "hard"]` lives at the SweepConfig level — the YAML setting `preset: easy` becomes the SweepConfig default, and the `ablate.py` dispatcher's `--preset` override only applies when the YAML default is not the only legal value. The dispatcher checks `args.ablation == "abl4_joint_easy_n2"` and logs `[WARN] --preset=<arg> ignored: abl4_joint_easy_n2 hard-pins preset=easy per spec 06 Lock 3` if `--preset` is supplied with a non-easy value, then proceeds with the hard-pinned `easy` value.

**Test gate**: `test_abl4_joint_enum_easy_n2_only` (per README C8-ABL-ABL4A, §7) parametrised over (`--preset=easy`, `--preset=medium`, `--preset=hard`); asserts the materialised `SweepConfig.preset == "easy"` regardless of CLI override, asserts `mve_joint_enumerate=True` in the override dict, asserts the warn-fallthrough is emitted when the planner is invoked at N>2 (a synthetic sub-test).

### 3.4 `abl6_fehr_schmidt.yaml` — 3×3 α/β sensitivity scan

**Purpose**: surface the sensitivity of the trained policy to the Fehr-Schmidt inequality-aversion parameters. Backs Ch6 sensitivity claim. The 3×3 grid is `cfg.train.fehr_schmidt_alpha ∈ {0.0, 0.5, 1.0}` × `cfg.train.fehr_schmidt_beta ∈ {0.0, 0.25, 0.5}` — 9 cells.

> **Cfg-field note**: `fehr_schmidt_alpha` / `fehr_schmidt_beta` are existing `TrainConfig` fields (not new in spec 06). If they are not yet present at `hyper_mve/configs/train_config.py`, the cell falls back to the Fehr-Schmidt parameters carried in the reward computation (per DESIGN_DOC_FINAL.md §5.6) — the override path is the same `--override train.fehr_schmidt_alpha=<json>` regardless. Spec 06 does NOT declare these fields as new cfg fields; they live with pkg-04 / pkg-05's existing reward-model SDD.

**SweepConfig fields**:

```yaml
# hyper_mve/experiments/ablations/abl6_fehr_schmidt.yaml
variants:
  - hyper
seeds: [0, 1, 2, 3, 4]
preset: medium
overrides:
  - {train.fehr_schmidt_alpha: 0.0, train.fehr_schmidt_beta: 0.0}
  - {train.fehr_schmidt_alpha: 0.0, train.fehr_schmidt_beta: 0.25}
  - {train.fehr_schmidt_alpha: 0.0, train.fehr_schmidt_beta: 0.5}
  - {train.fehr_schmidt_alpha: 0.5, train.fehr_schmidt_beta: 0.0}
  - {train.fehr_schmidt_alpha: 0.5, train.fehr_schmidt_beta: 0.25}
  - {train.fehr_schmidt_alpha: 0.5, train.fehr_schmidt_beta: 0.5}
  - {train.fehr_schmidt_alpha: 1.0, train.fehr_schmidt_beta: 0.0}
  - {train.fehr_schmidt_alpha: 1.0, train.fehr_schmidt_beta: 0.25}
  - {train.fehr_schmidt_alpha: 1.0, train.fehr_schmidt_beta: 0.5}
max_steps: 200000
eval_planner_mode: planner_full
max_parallel: 2
n_gpus: 2
ablation_cell_id: abl6_fehr_schmidt
```

**Expected row count**: 1 variant × 5 seeds × 9 cells = **45 rows**.

**Test gate**: `test_abl6_fehr_schmidt_yaml_materialises_3x3` (§7) asserts override count = 9 and that each override carries both α and β keys.

### 3.5 `abl7_curriculum.yaml` — 3-cell curriculum sweep

**Purpose**: surface the necessity of the three-stage curriculum by comparing the default `mixed` curriculum (= hyper) against `oracle_only` (stage-1-end=1.0, oracle context throughout training) and `infer_only` (stage-2-end=0.0, BeliefNet ĉ throughout). Backs the curriculum-staging dependence claim from pkg-07 spec 01 §2.2. The three cells are **variants**, not overrides — they map to the curriculum-override entries in `REGISTRY` per pkg-07 spec 01 §2.2.

**SweepConfig fields**:

```yaml
# hyper_mve/experiments/ablations/abl7_curriculum.yaml
variants:
  - oracle_only      # curriculum_stage_1_end_frac=1.0; oracle c throughout training
  - hyper            # = "mixed"; default 0.3/0.7 curriculum
  - infer_only       # curriculum_stage_2_end_frac=0.0; BeliefNet ĉ throughout
seeds: [0, 1, 2, 3, 4]
preset: medium
overrides:
  - {}               # single empty override; the variant is the cell axis
max_steps: 200000
eval_planner_mode: planner_full
max_parallel: 2
n_gpus: 2
ablation_cell_id: abl7_curriculum
```

**Expected row count**: 3 variants × 5 seeds × 1 cell = **15 rows** (per spec 05 §3.3 empty-overrides edge case: `overrides=({},)` collapses to single empty-dict row).

**Note** (rationale for variant-axis encoding): `oracle_only` and `infer_only` cannot be encoded as simple overrides because `TrainConfig.__post_init__` (lines 94-100 of `train_config.py`) rejects `curriculum_stage_1_end_frac=1.0` and `curriculum_stage_2_end_frac=0.0` (the validation requires `0 < s1 < s2 < 1`). The variant-axis encoding routes the cell to pkg-07's `REGISTRY[<variant>]` factory which bypasses the validation via its own `TrainConfig` construction. The `mixed` cell is exactly the `hyper` variant under its default 0.3/0.7 curriculum.

**Test gate**: `test_abl7_curriculum_yaml_three_variants` (§7) asserts `variants == ("oracle_only", "hyper", "infer_only")` (order-preserving) and that `overrides == ({},)`.

---

## 4. `use_coord_desc → randomize_order` rename mechanic (Lock 2 codified)

This section describes the **read-and-write deprecation mechanic** that implements the rename as alias-with-deprecation. The actual patches live in `hyper_mve/configs/train_config.py`, `hyper_mve/scripts/train_main.py`, and (transitively) `hyper_mve/planning/mve_planner.py`; spec 08 §5 codifies them.

### 4.1 Migration path on `TrainConfig`

`TrainConfig` (located at `hyper_mve/configs/train_config.py`) carries the legacy field `use_coord_desc: bool = True` at line 88. The migration adds the canonical field `randomize_order: bool = True` and reshapes `use_coord_desc` into a `@property` shim:

```python
# hyper_mve/configs/train_config.py — spec 06 §4 + spec 08 §5
@dataclass(frozen=True)
class TrainConfig:
    # ... other fields ...

    # 2x2 ablation switches (Ch6.7)
    use_crn: bool = True
    randomize_order: bool = True              # NEW canonical name (spec 06 §4, Theory Audit Q2 rename)
    mve_joint_enumerate: bool = False          # NEW (spec 06 §5, Abl4 Joint cell)

    # ... rest of the dataclass ...

    @property
    def use_coord_desc(self) -> bool:
        """Deprecated alias for ``randomize_order`` (spec 06 §4.1, Theory Audit Q2)."""
        import warnings
        warnings.warn(
            "cfg.train.use_coord_desc is deprecated; use cfg.train.randomize_order instead "
            "(spec 06 Lock 2 alias-with-deprecation). The alias is preserved for backwards "
            "compatibility but reads/writes warn on every access.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.randomize_order
```

**`@property` on a frozen dataclass note**: `dataclass(frozen=True)` does not block `@property` definitions; it only blocks ordinary attribute assignment via `__setattr__`. The property's getter is a method that reads `self.randomize_order` without assignment, so it is compatible with `frozen=True`. The setter is more delicate (see §4.2 below).

### 4.2 WRITE path (setter) — `dataclasses.replace` route

Because `TrainConfig` is frozen, direct attribute assignment is rejected:

```python
cfg.train.use_coord_desc = False   # ❌ FrozenInstanceError
```

The canonical write path is `dataclasses.replace(cfg.train, use_coord_desc=False)`. The `replace` call accepts only declared fields, not `@property` shims, so `replace(cfg.train, use_coord_desc=False)` raises `TypeError: __init__() got an unexpected keyword argument 'use_coord_desc'`. To preserve the alias on the write path, spec 06 §4.2 adds an `__init__` override (via `__post_init__` + a `_use_coord_desc_compat` sentinel) that catches the legacy kwarg and forwards it to `randomize_order`:

```python
@dataclass(frozen=True)
class TrainConfig:
    # ... fields including randomize_order: bool = True ...

    # Synthetic field to absorb legacy use_coord_desc=... kwargs from dataclasses.replace.
    # Default value matches randomize_order's default; __post_init__ reconciles.
    _use_coord_desc_compat: bool = True

    def __post_init__(self) -> None:
        # ... existing curriculum / lr_schedule validation ...
        # Spec 06 §4.2: if caller passed use_coord_desc=... (via __init__ or replace),
        # the compat sentinel will differ from the canonical randomize_order; reconcile.
        if self._use_coord_desc_compat != self.randomize_order:
            import warnings
            warnings.warn(
                "cfg.train.use_coord_desc kwarg is deprecated; use randomize_order instead "
                "(spec 06 Lock 2). Reconciled to randomize_order=<compat value>.",
                DeprecationWarning,
                stacklevel=3,
            )
            # Use object.__setattr__ to bypass frozen on this one reconciliation.
            object.__setattr__(self, "randomize_order", self._use_coord_desc_compat)
```

> **Alternative considered + rejected**: a custom `__init__` that accepts `use_coord_desc` as a keyword and forwards it to `randomize_order` was rejected because `@dataclass(frozen=True)` auto-generates `__init__` and overriding it bypasses the dataclass machinery (loses `__repr__`, `__eq__`, `__hash__` generation). The `_use_coord_desc_compat` synthetic-field approach preserves all dataclass machinery and only adds one reconciliation in `__post_init__`.

> **Cleaner alternative considered + rejected**: a custom `__init_subclass__` that intercepts kwargs at construction time was rejected because it complicates the class hierarchy (TrainConfig is constructed in many places, including `dataclasses.replace`, and the kwarg interception must be transparent).

A cleaner long-term implementation may emerge once `dataclasses` gains first-class deprecated-field support (PEP-XXXX); spec 06 stays at the synthetic-field reconciliation level to minimise code surface.

### 4.3 grep audit on codebase

The README §"输出清单"+ Theory Audit §Q2 require that no direct reader of `cfg.train.use_coord_desc` survives outside the `@property` shim itself. Per the Grep audit performed at spec 06 drafting time (Grep for `use_coord_desc` across `hyper_mve/`), the current readers are:

- `hyper_mve/configs/train_config.py` line 88 (the field definition itself — replaced by the shim per §4.1).
- `hyper_mve/planning/mve_planner.py` line 84 (`self.use_coord_desc = cfg.train.use_coord_desc`) and line 161 (`if self.use_coord_desc:`) — must be updated to `randomize_order` (the planner-side rename is part of the §8.2 downstream patch).
- `hyper_mve/configs/presets/medium.py` line 105 (`use_coord_desc=True` in the preset's `TrainConfig(...)` call) — the kwarg path is exercised on every preset construction; the `_use_coord_desc_compat` reconciliation in §4.2 ensures this still works with `DeprecationWarning` until the preset is updated to use `randomize_order=True`. Spec 06 declares the preset migration as opt-in (no version bump required); the deprecation warning surfaces the migration need.
- `hyper_mve/scripts/train_main.py` (currently has no direct `use_coord_desc` reference per Grep, but exposes overrides via `--override` which can spell `train.use_coord_desc=<bool>`; the override path routes through `dataclasses.replace(sub, **{field: value})` and hits the §4.2 reconciliation path).

The test `test_use_coord_desc_grep_codebase_clean` (§7) Greps the codebase for `use_coord_desc` outside the `@property` shim and the `_use_coord_desc_compat` reconciliation; the test asserts the migrating files listed above are the **only** remaining references, and fails if any new reader is added without spec 06 amendment.

### 4.4 Default behaviour preserved

The default `randomize_order: bool = True` matches the legacy `use_coord_desc: bool = True` default at `train_config.py` line 88 byte-identically. No production run, YAML config, or checkpoint loaded from disk changes behaviour after the rename. The deprecation warnings are emitted only when the legacy field is accessed *explicitly* (either by READ via the property or WRITE via the compat sentinel reconciliation); default construction (`TrainConfig()`) does not emit a warning.

### 4.5 Removal timeline

The alias is preserved for **at least one release minimum**; spec 06 deliberately **does not lock a removal date**. Future removal requires:
1. Synchronous edit of spec 06 Lock 2 to specify the removal version.
2. Synchronous edit of design.md §4 D10 to remove the alias-with-deprecation clause.
3. Confirmation via `test_use_coord_desc_grep_codebase_clean` that no remaining readers exist.

---

## 5. `mve_joint_enumerate` Pkg-05 downstream patch (Lock 3 codified)

This section describes the **+1 if-branch** patch site in `hyper_mve/planning/mve_planner.py` (the pkg-05 SDD owner of the planner) and the warn-with-fallthrough policy at N>2. The actual patch lives in pkg-05's planner file; spec 08 §5 codifies it.

### 5.1 Patch site (symbol-anchored)

The patch site lives **inside the agent-order-permutation block** of `hyper_mve/planning/mve_planner.py` — search anchor: `if self.use_coord_desc:` (the existing 4-line block that selects `agent_order` between a permuted and a fixed ordering). The current code shape:

```python
# Existing code at hyper_mve/planning/mve_planner.py (symbol-anchored:
# 'if self.use_coord_desc:' branch inside the agent-order-permutation block).
# Coordinate-descent agent ordering (use_coord_desc=False -> fixed order).
if self.use_coord_desc:
    agent_order = self.crn_rng.permutation(N).tolist()
else:
    agent_order = list(range(N))
```

Line numbers are intentionally omitted because pkg-08 spec 03 §10's `mve_planner.py` references (constructor reads `self.use_crn = cfg.train.use_crn`, etc.) and any future edits will shift line counts; the symbol anchor (`if self.use_coord_desc:`) is stable across spec 03 / spec 06 edit windows.

After the spec 06 §4 rename + the spec 06 §5 patch, the symbol-anchored block becomes:

```python
# Spec 06 §5.1 + §5.2 — codified in spec 08 §5 as Pkg-05 downstream patch
if self.mve_joint_enumerate:                                            # +1 (new branch)
    if self.N > 2:                                                       # +2 (warn-fallthrough guard)
        warnings.warn(                                                   # +3
            f"mve_joint_enumerate=True with N={self.N} would enumerate " # +4
            f"{self.A}^{self.N}={self.A**self.N} joint actions "         # +5
            f"(intractable); falling through to coord-descent",          # +6
            RuntimeWarning,                                              # +7
        )                                                                # +8
        # fall through to coord-descent below                            # +9
    else:                                                                # +10
        # Easy N=2: 6²=36 exhaustive enum is tractable.                  # +11
        # NOTE: the actual joint-enum compute path is a separate spec    # +12
        # 06 §5.4 deliverable; the +1-line declaration here is the       # +13
        # SDD anchor — the pkg-05 SDD owns the inner loop.               # +14
        agent_order = "JOINT_ENUM"  # sentinel; consumed in §5.4         # +15
# Coordinate-descent agent ordering (randomize_order=False -> fixed order).
if self.randomize_order:                                                 # renamed from use_coord_desc
    agent_order = self.crn_rng.permutation(N).tolist()
else:
    agent_order = list(range(N))
```

**Line count honesty**: spec 06 declared "+1 line `mve_joint_enumerate` branch" in the README outline. The accurate count under the warn-with-fallthrough policy is **+1 logical branch with ~15 supporting lines** (the warning message text + the sentinel for downstream consumption). Spec 08 §5 codifies this as a multi-line patch tagged `[behavioural; +1 logical branch + warn block]`, NOT as a literal `+1` line. The "1-line declaration" in README §"输出清单" is the **logical** branch count, not the literal line count; spec 06 reconciles the wording so future readers see the honest accounting.

### 5.2 Warn-with-fallthrough at N>2 (Lock 3 codified)

Per Lock 3 at the top of this spec, `mve_joint_enumerate=True` with `N>2` emits `RuntimeWarning` and falls through to coord-descent. The fallthrough is the locked policy because:

1. **Sweep harness throughput**: a hung subprocess (which is what a 6⁸=1.6M-action enumeration becomes on commodity GPU) never emits a `status="completed"` row to `runs/registry.jsonl`, corrupting the spec 05 §4 dedup tuple invariant. Warn-with-fallthrough preserves throughput.
2. **User intent**: a user setting `mve_joint_enumerate=True` at N>2 has almost certainly not intended to wait the necessary compute; the warning surfaces the intent-mismatch via stderr so it is visible in TB scalars / sweep stderr capture (per spec 05 §5.4).
3. **Idempotence**: a sweep that crashes mid-row leaves the registry in a `status="running"` orphan state requiring `--retry-failed` (spec 05 §8.2 line 6). Warn-with-fallthrough avoids the orphan.

The alternative — raising `RuntimeError` — was considered and rejected because it would mark the row as `status="failed"` and require manual `--retry-failed` intervention; warning-with-fallthrough preserves the row's analytic value (coord-descent baseline is still a valid result for the row).

### 5.3 New cfg field declaration

`cfg.train.mve_joint_enumerate: bool = False` is the new field declared in §6 below. Default `False` preserves all existing behaviour (no current run is affected); the field is opt-in via the `abl4_joint_easy_n2.yaml` override or via direct `--override train.mve_joint_enumerate=true` on `train_main.py`.

### 5.4 Why this is a Pkg-05 downstream patch (not a pkg-08 internal patch)

Pkg-05 owns `mve_planner.py` per the original v4 SDD discipline (pkg-05 spec 06 §"deliverable" lists `planning/mve_planner.py` as the canonical owner). Pkg-08 cannot modify pkg-05's owned files directly without violating the "Pkg-01..05 SDD 零修改" invariant declared in pkg-08 README §"输出清单" + design §2.3 NG2. Spec 06 declares the patch site; spec 08 §5 of pkg-08 codifies the canonical "下游补丁声明" list which carries the patch as one of the 3 implementation-only edits.

The downstream-consumer responsibility split is: pkg-05 SDD owns the planner's algorithmic semantics (CRN, coord-descent, MVE rollout depth); pkg-08 declares the +1 branch that consumes `cfg.train.mve_joint_enumerate` and the warn-fallthrough policy. The patch is implementation-only; pkg-05's SDD anchors (DESIGN_DOC_FINAL.md §4.1 + §5.8) remain byte-identical.

---

## 6. New cfg fields contributed by this spec

Per design §4 D10 + README §"5 新 cfg 字段穷举" rows 2–3, spec 06 contributes **two new cfg fields** to `TrainConfig`:

| Field | Type | Default | Owner | Consumer | Notes |
|---|---|---|---|---|---|
| `cfg.train.randomize_order` | `bool` | `True` | `TrainConfig` | `mve_planner.py` line 161 (renamed from `use_coord_desc`) | Spec 06 §4 alias-with-deprecation rename. Default matches legacy `use_coord_desc: bool = True`. |
| `cfg.train.mve_joint_enumerate` | `bool` | `False` | `TrainConfig` | `mve_planner.py` agent-order-permutation block | Spec 06 §5 Abl4 Joint cell trigger. Default `False` preserves coord-descent. Warn-with-fallthrough at N>2 per Lock 3. |

Both fields are declared HERE in spec 06 §6 and codified in spec 08 §5 of pkg-08 (which is the canonical "5 新 cfg 字段穷举" list, joined with `cfg.env.c_visible` from spec 02 + `cfg.eval.eval_planner_mode` + `cfg.eval.eval_use_planner_direct_inference` from spec 03). Pkg-01 spec 05 (V4Config consumption table) is synchronously updated when spec 08 is finalised so the canonical V4Config schema view discovers the new fields.

The synthetic `_use_coord_desc_compat: bool = True` field declared in §4.2 is **not** a user-facing cfg field — it is an internal `__post_init__` reconciliation sentinel and is excluded from the "5 新 cfg 字段穷举" enumeration. Test `test_use_coord_desc_compat_is_internal` (§7) asserts `_use_coord_desc_compat` is not listed in `TrainConfig.__dataclass_fields__` as a user-facing field (verified by inspecting the leading underscore convention).

---

## 7. Test contract — ≥8 named tests

Located under `tests/experiments/` (and one cross-cutting test under `tests/configs/`). Each test name corresponds to a specific anchor in spec 06 / README C8-ABL-*.

### 7.1 `test_ablation_cli_dispatch_all_5` — C8-ABL-CLI1 (§2)

Parametrised over the five CLI IDs `(abl1, abl4_crn_joint, abl4_joint_easy_n2, abl6, abl7)`. For each ID, invokes `parse_args(["--ablation", id, "--dry-run"])`, asserts `args.ablation == id`, then calls the dispatcher up to (but not including) `run_sweep`, asserting the YAML loads and `SweepConfig.from_yaml(...)` returns a valid `SweepConfig` instance with `sweep_cfg.ablation_cell_id == id` for the four non-Abl4-sub-cell IDs and `sweep_cfg.ablation_cell_id == "abl4_joint_easy_n2"` for the sub-cell.

### 7.2 `test_ablation_cli_rejects_unknown_id` — §2.3

```python
def test_ablation_cli_rejects_unknown_id():
    with pytest.raises(SystemExit) as exc_info:
        parse_args(["--ablation", "abl9"])
    assert exc_info.value.code == 2  # argparse error exit
```

Asserts the help text contains the canonical five IDs (regex match for `abl1.*abl4_crn_joint.*abl4_joint_easy_n2.*abl6.*abl7`).

### 7.3 `test_ablation_cli_dry_run_prints_sweepconfig` — §2.4

Monkey-patches `run_sweep` to raise `RuntimeError("should not be called")` and asserts `python -m hyper_mve.experiments.ablate --ablation abl1 --dry-run` exits 0 without raising. Verifies the dry-run stdout contains "variants=" and "Total rows would be:".

### 7.4 `test_abl4_joint_enum_easy_n2_only` — C8-ABL-ABL4A + Lock 3 (§3.3 + §5.2)

Parametrised over `--preset ∈ {easy, medium, hard}`. For each preset CLI arg:
- Asserts the materialised `SweepConfig.preset == "easy"` regardless of the CLI override (Lock 3 hard-pin enforcement).
- Asserts the override list contains exactly one dict with `train.mve_joint_enumerate=True`.

A sub-test invokes the planner directly with a synthetic `cfg` at `env.N=4` and `train.mve_joint_enumerate=True`, asserts `RuntimeWarning` is raised, and asserts the resulting `pi_mve` is byte-identical to the coord-descent baseline.

### 7.5 `test_use_coord_desc_alias_warns` — C8-ABL-RENAME1 (§4.1 + §4.2)

Parametrised over READ + WRITE paths:

**READ path**:
```python
def test_use_coord_desc_alias_warns_read():
    cfg = V4Config()
    with pytest.warns(DeprecationWarning, match="use_coord_desc is deprecated"):
        value = cfg.train.use_coord_desc
    assert value == cfg.train.randomize_order
```

**WRITE path** (via `dataclasses.replace`):
```python
def test_use_coord_desc_alias_warns_write():
    cfg = V4Config()
    with pytest.warns(DeprecationWarning, match="use_coord_desc kwarg is deprecated"):
        new_train = dataclasses.replace(cfg.train, use_coord_desc=False)
    assert new_train.randomize_order is False
```

### 7.6 `test_use_coord_desc_grep_codebase_clean` — §4.3

Greps `hyper_mve/` for `use_coord_desc` and asserts only the migrating files listed in §4.3 contain the string:

```python
def test_use_coord_desc_grep_codebase_clean():
    repo_root = pathlib.Path("hyper_mve")
    expected_files = {
        "hyper_mve/configs/train_config.py",       # @property + _use_coord_desc_compat
        "hyper_mve/configs/presets/medium.py",     # legacy kwarg (warned, opt-in migration)
        # mve_planner.py removed from list after spec 06 §8.2 patch is applied
    }
    found = set()
    for py_file in repo_root.rglob("*.py"):
        if "use_coord_desc" in py_file.read_text():
            found.add(str(py_file).replace("\\", "/"))
    extra = found - expected_files
    assert not extra, f"Unexpected use_coord_desc readers: {extra}"
```

### 7.7 `test_mve_joint_enumerate_easy_n2_36_cells` — §3.3 + §5.1

Synthetic test that exercises the `mve_joint_enumerate=True` path at Easy N=2:

```python
def test_mve_joint_enumerate_easy_n2_36_cells():
    cfg = V4Config.from_preset("easy")  # N=2, A=6
    cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, mve_joint_enumerate=True))
    candidates = list(itertools.product(range(cfg.env.A), repeat=cfg.env.N))
    assert len(candidates) == 36
    # planner-level sub-test: assert joint enum is actually invoked (no fallthrough at N=2)
    planner = MVEPlanner(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        # No warning should fire at N=2
        _ = planner.sample_mve_plan(...)
```

### 7.8 `test_mve_joint_enumerate_medium_falls_through_to_coord_desc` — Lock 3 + §5.2

```python
def test_mve_joint_enumerate_medium_falls_through_to_coord_desc():
    cfg = V4Config.from_preset("medium")  # N=4, A=6
    cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, mve_joint_enumerate=True))
    planner = MVEPlanner(cfg)
    with pytest.warns(RuntimeWarning, match="intractable.*falling through"):
        pi_mve_enum_path = planner.sample_mve_plan(...)
    # Compare against coord-descent baseline (mve_joint_enumerate=False)
    cfg_coord = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, mve_joint_enumerate=False))
    planner_coord = MVEPlanner(cfg_coord)
    pi_mve_coord = planner_coord.sample_mve_plan(...)
    # The fallthrough path must produce coord-descent output (same RNG seed assumed)
    assert torch.allclose(pi_mve_enum_path, pi_mve_coord)
```

### 7.9 `test_all_5_yamls_valid_yaml_syntax` — §3 cross-cutting

Loads each of the five YAMLs and asserts no parse errors:

```python
@pytest.mark.parametrize("ablation_id", ["abl1", "abl4_crn_joint", "abl4_joint_easy_n2", "abl6", "abl7"])
def test_all_5_yamls_valid_yaml_syntax(ablation_id):
    yaml_path = pathlib.Path("hyper_mve/experiments/ablations") / f"{ablation_id}.yaml"
    assert yaml_path.exists(), f"missing YAML: {yaml_path}"
    sweep_cfg = SweepConfig.from_yaml(yaml_path)
    assert isinstance(sweep_cfg, SweepConfig)
    assert sweep_cfg.ablation_cell_id == ablation_id
```

### 7.10 `test_canned_yaml_inventory_matches_design` — Lock 1 drift detector

```python
def test_canned_yaml_inventory_matches_design():
    yaml_dir = pathlib.Path("hyper_mve/experiments/ablations")
    found = {p.stem for p in yaml_dir.glob("*.yaml")}
    expected = {"abl1_gen_scope", "abl4_crn_joint", "abl4_joint_easy_n2",
                "abl6_fehr_schmidt", "abl7_curriculum"}
    assert found == expected, f"Canned YAML inventory drift: extra={found - expected}, missing={expected - found}"
```

**Test count: 10 named tests** (≥ 8 required).

---

## 8. Downstream patches (declared here, codified in spec 08 §5)

This spec declares **three downstream patches** to pkg-01 / pkg-05 implementation files. Spec 08 §5 carries the canonical "下游补丁声明" list and codifies the patches; spec 06 only declares them.

### 8.1 `hyper_mve/configs/train_config.py` — config-additions + alias-with-deprecation

**Size**:
- **+2 dataclass field declarations** (`randomize_order: bool = True`, `mve_joint_enumerate: bool = False`).
- **+1 synthetic field declaration** (`_use_coord_desc_compat: bool = True`) for the WRITE-path reconciliation per §4.2.
- **+1 `@property` shim** (`use_coord_desc`) with `DeprecationWarning` on READ per §4.1.
- **+1 `__post_init__` reconciliation block** (~6 lines) for the WRITE path per §4.2.
- **REMOVE** `use_coord_desc: bool = True` line 88 (replaced by the synthetic + property).

**Behaviour**: zero net behaviour change at default `TrainConfig()` construction (the property + reconciliation reproduce the legacy semantics byte-identically); deprecation warnings fire only on explicit legacy-name access.

**Tag**: `[config-additions + alias-with-deprecation]`.

**Test gate**: `test_use_coord_desc_alias_warns` (§7.5) + `test_use_coord_desc_compat_is_internal` (§6).

### 8.2 `hyper_mve/planning/mve_planner.py` — behavioural +1 branch + warn-with-fallthrough

**Size**:
- **+1 logical if-branch** (`if self.mve_joint_enumerate:`) at the agent-order-permutation block per §5.1, with ~15 supporting lines (warning message text + sentinel for downstream consumption).
- **+1 rename** of the existing line 84 `self.use_coord_desc = cfg.train.use_coord_desc` → `self.randomize_order = cfg.train.randomize_order` (with optional `getattr` fallback for backwards-compat against old checkpoints' cfg snapshots).
- **+1 rename** of the existing line 161 `if self.use_coord_desc:` → `if self.randomize_order:`.

**Behaviour**: when `cfg.train.mve_joint_enumerate is False` (the default), output is byte-identical to the legacy planner. When `True` and `N <= 2`, the agent-order block enters the joint-enumeration path (full implementation owned by pkg-05's planner inner loop; spec 06 only declares the branch). When `True` and `N > 2`, the planner emits `RuntimeWarning` and falls through to coord-descent.

**Tag**: `[behavioural; +1 logical branch + warn block + 2 line renames]`.

**Test gate**: `test_mve_joint_enumerate_easy_n2_36_cells` (§7.7) + `test_mve_joint_enumerate_medium_falls_through_to_coord_desc` (§7.8).

### 8.3 `hyper_mve/scripts/train_main.py` — argparse alias-with-deprecation

**Size**:
- **+1 argparse argument rename** (`--use_coord_desc` → `--randomize_order`).
- **+1 deprecated alias preservation** (`--use_coord_desc` retained as an alternate flag emitting `DeprecationWarning` on parse).

**Behaviour**: at the CLI level, `--use_coord_desc=true` still works and writes to `cfg.train.randomize_order` via the `apply_overrides` → `dataclasses.replace` path (which routes through the §4.2 reconciliation). The argparse-level alias emits its own `DeprecationWarning` so users see the migration nudge at the CLI even before the dataclass-level warning fires.

**Tag**: `[downstream-cli; alias-with-deprecation]`.

**Test gate**: `test_use_coord_desc_alias_warns_write` (§7.5) exercises the dataclass-level path; a separate spec 08 §5 test covers the argparse-level alias.

> **Honest line-count summary**: §8.1 is roughly **+10 lines** of dataclass code (2 fields + 1 synthetic + 1 property + 6 lines `__post_init__` + 1 import); §8.2 is roughly **+15 lines** of planner code (1 branch + warning block + 2 renames); §8.3 is roughly **+5 lines** of argparse code (1 rename + 1 alias). The README §"输出清单" claim of "+1 行 `mve_joint_enumerate` 分支" is the **logical** branch count, not the literal line count — spec 06 §5.1 reconciles the wording.

---

## 9. Integration hooks (cross-spec)

### 9.1 With pkg-08 spec 02 (zero-shot + c_hidden + regret)

`abl1_gen_scope.yaml` respects spec 02 Lock 2 (cfg-driven `zero_shot_*` grid). The YAML does NOT hardcode c-values; it relies on `cfg.eval.zero_shot_test_c` (7-c grid) consumed by the unified evaluator. The `c_hidden` cell in Abl1 sets `env.c_visible: false` via override, triggering the spec 02 §3.2 obs-mask path. Drift in `cfg.eval.zero_shot_*` defaults → drift in Abl1's seen/unseen partition reporting → caught by spec 02 disjoint-union invariant test.

### 9.2 With pkg-08 spec 03 (four planner eval modes)

All five YAMLs hard-pin `eval_planner_mode: planner_full` at the `SweepConfig` level per spec 03 Lock 1 (eval-time vs training-time axes are orthogonal). The Abl4 cells (`abl4_crn_joint.yaml` + `abl4_joint_easy_n2.yaml`) vary training-time `cfg.train.use_crn` / `cfg.train.randomize_order` while keeping eval-time `eval_planner_mode="planner_full"`, ensuring fair eval-time comparison of training-time effects.

### 9.3 With pkg-08 spec 05 (sweep harness + RunRegistry)

`ablate.py` dispatches to `run_sweep(SweepConfig.from_yaml(<id>.yaml))` per spec 05 §10.4 sketch. The sweep harness consumes `SweepConfig.ablation_cell_id` (spec 05 §3.1 field) and writes it to every `RunRegistry` row's `ablation_cell` field (spec 05 §4 schema slot 5). Spec 07's `compare.py` later groups by `ablation_cell` to surface per-cell statistics.

### 9.4 With pkg-08 spec 07 (statistics + comparison)

Spec 07's `compare.py` consumes ablation rows from `runs/registry.jsonl` filtered by `ablation_cell.startswith("abl4_crn_joint")` (and similar prefix filters for the other YAMLs) to produce the 2-method comparison plots. The Welch t-test pipeline in spec 07 §3 reads `eval_report_path` from each row and joins per-c returns for paired comparisons across CRN-on/off cells.

### 9.5 With pkg-08 spec 08 (integration contracts)

Spec 08 §5 codifies the three downstream patches declared in §8 of this spec verbatim. Spec 08 §6 drift detector cross-locks the canonical "下游补丁声明" list against the patch declarations here; any drift triggers reviewer reconciliation.

### 9.6 With pkg-07 spec 01 (REGISTRY + curriculum-override variants)

`abl7_curriculum.yaml` consumes pkg-07 spec 01 §2.2 curriculum-override variants `oracle_only` / `infer_only` (and the default `hyper` for the `mixed` cell). The REGISTRY 11-key set per pkg-07 spec 01 §2.1 (`oracle_only`, `mixed` ≡ `hyper`, `infer_only`) is reverse-consumed by this YAML; drift in REGISTRY → drift in Abl7 cell composition → caught by pkg-07 spec 01 §2.3 量词 canonical test.

### 9.7 With pkg-05 mve_planner.py (downstream patch site)

Per §8.2, the `mve_joint_enumerate` branch + the `use_coord_desc → randomize_order` rename are applied to pkg-05's `planning/mve_planner.py`. Pkg-05 SDD is **not modified**; the patch is implementation-only per the "Pkg-01..05 SDD 零修改" invariant.

### 9.8 With pkg-05 train_main.py (CLI rename site)

Per §8.3, the `--use_coord_desc → --randomize_order` argparse rename + deprecated alias are applied to pkg-05's `scripts/train_main.py`. Pkg-05 SDD is **not modified**; the patch is implementation-only.

---

## 10. Cross-references

### 10.1 Intra-pkg-08

- **spec 01 unified-evaluator** — every ablation row emits one `EvalReport` via the worker subprocess's `unified_evaluator.evaluate(runner, env_fn, cfg)` call; spec 06 does not introduce a parallel evaluator path.
- **spec 02 zero-shot + c_hidden + regret** — `abl1_gen_scope.yaml` cells `zs_interp` / `zs_extrap` / `c_hidden` consume spec 02 Lock 2's cfg-driven c-grid + Lock 1's obs-mask patch.
- **spec 03 direct-inference toggle** — all five YAMLs hard-pin `eval_planner_mode: planner_full` per spec 03 Lock 1.
- **spec 05 sweep harness + RunRegistry** — `ablate.py` dispatches to `run_sweep(SweepConfig.from_yaml(<id>.yaml))`.
- **spec 07 statistics + comparison** — consumes ablation rows for Welch t / Holm-Bonferroni / bar+errorbar plots.
- **spec 08 integration contracts** — §5 codifies the 3 downstream patches declared by spec 06 §8.

### 10.2 Pkg-07 reverse consumption

- **pkg-07 spec 01 §2.1 REGISTRY** — `abl7_curriculum.yaml` consumes `oracle_only` / `mixed` ≡ `hyper` / `infer_only` variants; `abl1_gen_scope.yaml` consumes 5 internal baselines + `hyper` + `oracle_only`.
- **pkg-07 spec 01 §2.2 curriculum-override variants** — `oracle_only` / `infer_only` factory dispatch consumed by `abl7_curriculum.yaml`.
- **pkg-07 spec 01 §2.3 量词 canonical (11 keys)** — drift in REGISTRY → drift in Abl1 / Abl7 variant lists → caught by pkg-07 量词 canonical test.

### 10.3 Existing repo facts consumed

- **`hyper_mve/configs/train_config.py`** lines 87–88 (`use_crn: bool = True`, `use_coord_desc: bool = True`) → §4 rename mechanic + §6 new field declarations.
- **`hyper_mve/configs/train_config.py`** lines 94–110 (`__post_init__`) → §4.2 WRITE-path reconciliation extension site.
- **`hyper_mve/configs/presets/medium.py`** line 105 (`use_coord_desc=True` kwarg) → §4.3 grep audit baseline reader (warned, opt-in migration).
- **`hyper_mve/planning/mve_planner.py`** lines 83–84 (`self.use_crn = cfg.train.use_crn`, `self.use_coord_desc = cfg.train.use_coord_desc`) + lines 160–164 (agent-order block) → §5.1 patch site + §8.2 renames.
- **`hyper_mve/scripts/train_main.py`** lines 55–75 (`parse_args`) + line 86 (`apply_overrides`) → §8.3 argparse rename + alias preservation.
- **`hyper_mve/configs/eval_config.py`** lines 37–39 (`zero_shot_*` reserved fields) → §3.1 Abl1 cfg-driven c-grid compliance.

### 10.4 Theory Audit anchors

- **§M8 Abl4 redefinition** — §1 Motivation 2 + §3.2 abl4_crn_joint cell + §3.3 abl4_joint_easy_n2 sub-cell.
- **§Q2 `use_coord_desc → randomize_order` rename** — Lock 2 + §4 rename mechanic + §8.1 patch.
- **§2.2 Abl1 gen_scope 7-cell** — §3.1 abl1_gen_scope cells.
- **§2.4 Joint enum motivation** — §3.3 abl4_joint_easy_n2 cell + §5 mve_joint_enumerate patch.

### 10.5 Upstream design anchors

- **design.md §3.1** — Eval/Ablation half allocation (spec 06 lives in the Ablation half).
- **design.md §3.3** — four planner eval mode taxonomy + Abl4 decoupling table (spec 06 Lock 1 mirrors).
- **design.md §4 D9** — canned YAML dispatch table (spec 06 §2.2 mirrors verbatim).
- **design.md §4 D10** — 5 new cfg fields (spec 06 §6 contributes 2 of them).

---

## 11. Anchors (verbatim grep targets for spec 08 §6 drift detector)

- `pkg-08 spec 06 Lock 1: Five canned YAMLs equal four logical ablation cells; the 5/4 asymmetry is locked`
- `pkg-08 spec 06 Lock 2: use_coord_desc to randomize_order rename is alias-with-deprecation, not breaking`
- `pkg-08 spec 06 Lock 3: mve_joint_enumerate=True is tractable ONLY at Easy N=2; YAML hard-pins preset=easy; planner warns-with-fallthrough at N>2`
- `pkg-08 spec 06 §2: ablate CLI argparse signature with five choices abl1 / abl4_crn_joint / abl4_joint_easy_n2 / abl6 / abl7`
- `pkg-08 spec 06 §3.1: abl1_gen_scope.yaml — 7 cells × 6 variants × 5 seeds = 210 rows`
- `pkg-08 spec 06 §3.2: abl4_crn_joint.yaml — 2×2 CRN × CoordDesc × 5 seeds = 20 rows; eval_planner_mode=planner_full pinned`
- `pkg-08 spec 06 §3.3: abl4_joint_easy_n2.yaml — preset:easy HARD-PIN; mve_joint_enumerate=True; 5 rows`
- `pkg-08 spec 06 §3.4: abl6_fehr_schmidt.yaml — 3×3 alpha/beta × 5 seeds = 45 rows`
- `pkg-08 spec 06 §3.5: abl7_curriculum.yaml — 3 variants oracle_only/hyper/infer_only × 5 seeds = 15 rows`
- `pkg-08 spec 06 §4: use_coord_desc to randomize_order alias-with-deprecation mechanic via @property + _use_coord_desc_compat __post_init__ reconciliation`
- `pkg-08 spec 06 §5: mve_joint_enumerate Pkg-05 patch — +1 logical branch in mve_planner.py agent-order block; warn-with-fallthrough at N>2`
- `pkg-08 spec 06 §6: 2 new cfg fields — cfg.train.randomize_order (rename) + cfg.train.mve_joint_enumerate (Joint cell)`
- `pkg-08 spec 06 §7: ≥8 named tests (10 declared) covering CLI dispatch, alias READ/WRITE warnings, joint-enum tractability, grep audit, YAML inventory drift`
- `pkg-08 spec 06 §8: 3 downstream patches — train_config.py + mve_planner.py + train_main.py; codified in spec 08 §5`
