# Phase 6 — Remote Linux Runtime Checklist

This file collects the smokes that need a **real torch + GPU env** (your Linux box).
The Windows local box has no torch, so everything below was left unrun.

All commands assume **cwd = `D:\RL\hyper_mve`** (or your Linux equivalent).

---

## 1. Static gates (already PASS locally on Windows)

| Check | Status | Command |
|---|---|---|
| pkg-07 ref-matrix | `[PASS]` | `pushd sdd/pkg-07-baselines; pwsh ./scripts/check_ref_matrix.ps1; popd` |
| pkg-08 ref-matrix | `[PASS]` | `pushd sdd/pkg-08-eval-and-ablation; pwsh ./scripts/check_ref_matrix.ps1; popd` |
| 4 planner-mode literals | 4 unique | (see verification block at bottom of this file) |
| 11 REGISTRY keys | 11 unique | ditto |
| 33-field EvalReport | 33 in spec | ditto |
| `cli_to_factory_arg` totality | OK (local) | `py -3 -c "from hyper_mve.baselines import cli_to_factory_arg, CLI_CHOICES; ..."` |
| AST sweep | 224/224 clean | `py -3 -c "import ast, pathlib; ..."` |

---

## 2. Phase 6 runtime smokes (need torch + GPU; **PLEASE RUN ON LINUX**)

### 2.1 pytest collection sanity (5 min)

Verifies every test file can be **collected** (parses + imports OK) under the
real torch env. No `-x` so we see the full set.

```bash
cd ~/hyper_mve     # adjust to your path
pytest tests/ --collect-only -q 2>&1 | tail -40
```

Expected: no `ERRORS` / `Internal Error` lines; total tests in the hundreds.

### 2.2 Fast unit tests (no torch heavy lifting; **target: all green, ≤ 5 min**)

```bash
pytest tests/baselines/test_factory_internal.py \
       tests/baselines/test_internal_equal_params.py \
       tests/baselines/test_internal_7api.py \
       tests/baselines/external/test_registry.py \
       tests/baselines/external/test_stub_external_baselines.py \
       tests/baselines/external/test_external_adapter_consumption.py \
       tests/baselines/external/test_external_eval_contract.py \
       tests/eval/ \
       tests/envs/ \
       tests/experiments/test_run_registry_row_schema.py \
       tests/experiments/test_run_registry_jsonl_append_atomic.py \
       tests/experiments/test_sweep_cartesian_correct.py \
       tests/experiments/test_config_hash_stable.py \
       tests/experiments/test_gpu_semaphore.py \
       tests/experiments/test_ablation_cli_dispatch.py \
       tests/experiments/test_stats_holm_bonferroni.py \
       tests/experiments/test_compare_plot_renders_headless.py \
       tests/experiments/test_disclosure_table_schema.py \
       tests/experiments/test_use_coord_desc_rename.py \
       tests/integration/test_pkg08_drift_detectors.py \
       -x --timeout=120
```

If `statsmodels` is missing locally, `test_holm_bonferroni_against_statsmodels`
auto-skips (parametrised over k∈{2,3,4,5}). Install with `pip install statsmodels`
to lift the skip.

### 2.3 Per-variant 100-step `hyper` smoke (≤ 5 min)

`train_main.py` only drives `hyper` directly (internal baselines + external
runners are runner-owned per pkg-07 spec 08; they run through the sweep
harness). So the train-side direct smoke is just:

```bash
python hyper_mve/scripts/train_main.py \
    --preset easy --variant hyper --max_steps 100 --seed 0
```

Expected: training runs ~100 iterations, prints scalar progress, no crash.

### 2.4 Sweep-harness end-to-end smoke (one row; ≤ 10 min)

Tiny end-to-end via the new sweep harness. This exercises subprocess-per-row
isolation + JSONL registry + EvalReport pipeline:

```bash
mkdir -p runs/smoke_phase6

# Build a 1-row SweepConfig YAML inline.
cat > /tmp/smoke_sweep.yaml <<'EOF'
variants: [hyper]
seeds: [0]
overrides:
  - {}
preset: easy
max_steps: 100
eval_planner_mode: planner_full
max_parallel: 1
n_gpus: 1
ablation_cell_id: smoke_phase6
EOF

python -m hyper_mve.experiments.sweep \
    --config /tmp/smoke_sweep.yaml \
    --runs-root runs/smoke_phase6 \
    --max-parallel 1 --n-gpus 1
```

Expected outputs:
- `runs/smoke_phase6/registry.jsonl` exists with **≥ 2 lines** (pending + completed).
- Final line has `"status": "completed"` and `"schema_version": "pkg08-spec05-v1"`.
- `runs/smoke_phase6/<run_tag>/eval_report.json` exists; loads back as a 33-key dict.

Verify with:

```bash
tail -1 runs/smoke_phase6/registry.jsonl | python -c "
import json, sys
row = json.loads(sys.stdin.read())
assert row['status'] == 'completed', row
assert row['schema_version'] == 'pkg08-spec05-v1'
print('[OK] sweep end-to-end smoke')
"
```

### 2.5 Ablate dispatcher dry-run (≤ 30 s; **does NOT need torch**)

Pure metadata. Already-passing-style locally, but confirms the package install.

```bash
for id in abl1 abl4_crn_joint abl4_joint_easy_n2 abl6 abl7; do
    python -m hyper_mve.experiments.ablate --ablation "$id" --dry-run
done
```

Expected: 5 successful dry-run blocks, each printing
`Total rows would be: N x M x K = ...`.

### 2.6 Tier-1 external smokes (gated `@pytest.mark.slow`; **≥ 30 min wall-clock**)

Only needed once you can spare GPU time. Authored under Phase 4 (Block 10):

```bash
pytest tests/baselines/external/test_mappo_smoke.py \
       tests/baselines/external/test_qmix_smoke.py \
       tests/baselines/external/test_ma_muzero_gh_smoke.py \
       --runslow --timeout=2400
```

Each is a `20K env-step Easy preset` C7-EXT-SMOKE1 gate: 3 seeds, Welch t vs
random baseline at one-sided α=0.05.

### 2.7 Compare CLI disclosure smoke (≤ 10 s; needs a populated registry)

After 2.4 (and ideally 2.6), exercise the `compare --disclose` mode:

```bash
python -m hyper_mve.experiments.compare \
    --disclose --registry runs/smoke_phase6/registry.jsonl \
    --out runs/smoke_phase6/_disclose
cat runs/smoke_phase6/_disclose/disclosure.md
```

Expected: a markdown file with `## Disclosure table — easy preset` and the
10-column header.

---

## 3. Gate map (where each acceptance criterion is satisfied)

| Gate | Spec | Where verified |
|---|---|---|
| `C7-INT-FACT1/2`, `FAIR1/2`, `API1`, `SELF1`, `REUSE1`, `GRAD1`, `STRUCT1/2` | pkg-07 spec 01-03 | `tests/baselines/test_factory_internal.py` + siblings (Phase 2; needs torch) |
| `C7-EXT-FACT1`, `API1`, `STUB1` | pkg-07 spec 05/06 | `tests/baselines/external/test_registry.py` + `test_stub_external_baselines.py` |
| `C7-EXT-ADPT1/2` | pkg-07 spec 04 | `tests/envs/test_pettingzoo_adapter_smoke.py` |
| `C7-EXT-SMOKE1` (3×) | pkg-07 spec 05/06 §10 | §2.6 above (gated `--runslow`) |
| `C8-EVAL-CHID1`, `MODE1..4`, `EXT1`, `LEAK1`, `REGRET1` | pkg-08 spec 01-03 | `tests/eval/*.py` |
| `C8-ABL-SWEEP1`, `ISO1`, `REG1` | pkg-08 spec 05 | `tests/experiments/test_sweep_cartesian_correct.py` + `_atomic.py` + `_gpu_semaphore.py` + `_config_hash_stable.py` |
| `C8-ABL-CLI1`, `CELL1..5`, `RENAME1` | pkg-08 spec 06 | `tests/experiments/test_ablation_cli_dispatch.py` + `_use_coord_desc_rename.py` |
| `C8-ABL-STAT1`, `PLOT1` | pkg-08 spec 07 | `tests/experiments/test_stats_holm_bonferroni.py` + `_compare_plot_renders_headless.py` |
| `C8-ABL-E2E1` | pkg-08 spec 08 §9 | §2.4 above (sweep end-to-end smoke) |
| Cross-package byte-identity (4 invariants) | pkg-07 + pkg-08 spec 08 | `tests/integration/test_pkg08_drift_detectors.py` |

---

## 4. Cleanup after smokes pass

```bash
rm -rf runs/smoke_phase6
rm /tmp/smoke_sweep.yaml
```

Then mark Task #7 (Phase 6) **completed** and call the SDD loop closed.
