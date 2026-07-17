# Hyper-MuZero (v4) — Experiment Structure Overview

> **Live structural map of the current project.** The workspace `CLAUDE.md` still describes the
> earlier **v4.7 "Non-stationary Tag"** design, now archived under [`_legacy_v4_7/`](_legacy_v4_7/)
> and superseded. This file is the authoritative quick-reference for the **v4 "Resource Commons"
> pipeline** that is actually running.

---

## 1. One-paragraph picture
A single model — **`HyperMuZeroModel`** — is compared against a battery of ablations and external
MARL baselines on a **non-stationary common-pool resource dilemma** (`ResourceCommonsEnv`).
Every variant is the same model + config overrides. All experiments are declared as **cells** under
[`experiments/suite/`](experiments/suite/) and driven by **one launcher** ([`scripts/run_suite.py`](scripts/run_suite.py))
over a subprocess-per-row **sweep harness**, with **selective re-run down to a single `(variant, seed)`**.
Each run writes a frozen **36-field `EvalReport`** (incl. 5 thesis welfare metrics) + an append-only
registry; the **analysis layer** turns those into the thesis tables/figures.

## 2. What is actually running — the suite
The thesis experiment matrix lives in [`experiments/suite/manifest.yaml`](experiments/suite/manifest.yaml)
(one cell per experiment, in decision-gate order). Drive it with **`run_suite.py`**:
```powershell
python hyper_mve/scripts/run_suite.py --list                       # every cell + tier + row count + blocked status
python hyper_mve/scripts/run_suite.py --only abl3_easy_n2 --dry-run # the 生死判官 Easy gate, preview
python hyper_mve/scripts/run_suite.py --tier must_have --gpus 2,3,4,5 --slots-per-gpu 2
python hyper_mve/scripts/run_suite.py --only main_comparison --variants hyper --seeds 0 --force  # re-run ONE row
```
Cells (decision-gate order): **gen_scope gate** (`lora_*` ×6) → **决策点 1/2 Easy gates** (`abl3_easy_n2`,
`abl1_easy`, `zero_shot_easy`, `main_comparison_easy`) → **Medium headlines** (`abl3_type_heterogeneity`,
`abl1_gen_scope`, `zero_shot_generalization`, `main_comparison`) → **other ablations** (`abl4_*`, `abl6`,
`abl7`) → **blocked/supplementary** (`abl2_context_paths` [BLOCKED], `lr_sweep`). Existing ablations stay
in [`experiments/ablations/`](experiments/ablations/) (now carry a `meta:` header). Output: `runs/suite/<runs_subdir>/`
(per-cell `registry.jsonl` + per-row TB/ckpt/`eval_report.json`).

> The old `train_fast_sweep.py` / `run_lora_experiments.py` launchers were **removed** — their matrices
> are now the `main_comparison` and `lora_*` cells.

### Selective re-run
`run_sweep` skips rows whose `(variant, seed, config_hash)` already completed, so re-invoking never
retrains finished work. `--only/--variants/--seeds/--size` narrow *which* rows a cell enumerates;
`--force` appends a `failed` tombstone (append-only safe) for exactly the narrowed rows + re-runs only
those — use it after editing a cell's underlying code (the harness has no code-version invalidation).

### The 14-variant surface (`--variant` / cell `variants:`, in [`baselines/__init__.py`](baselines/__init__.py))
| Group | Variants | How driven |
|---|---|---|
| Curriculum overrides (3) | `hyper`, `oracle_only`, `infer_only` | same `HyperMuZeroModel`, different curriculum |
| Internal baselines (5) | `baseline_input_wide`, `baseline_input_deep`, `baseline_ma_muzero`, `no_belief`, `rewardhead_explicit_type` | shared backbones, ablated conditioning |
| External baselines (6) | `external_mappo`, `external_qmix`, `external_ma_muzero_gh` (real) + `external_mamba`, `external_marie`, `external_ga` (stubs → auto-skip) | own `.train()/.evaluate()` |

## 3. Repo map (`hyper_mve/hyper_mve/`)
```
configs/     V4Config = env+model+train+mup+eval+legacy+baselines; presets/ (11)
schemas/     typed: AgentType, CapabilityVector, Observation, TimeStepRecord, Context
envs/resource_commons/   env + state, dynamics, rewards, context_evolution, spaces, spawn
models/      HyperMuZeroModel + tri_context, belief_net, hyper_network, functional_nets, rep_net
training/    muzero_trainer, worker (welfare-signal hook), episode_buffer, curriculum, evaluation(run_eval)
planning/    mve_planner  (CRN coordinate descent → policy target π_mve)
baselines/   internal/ (5) + external/ (6, vendored) + shared_backbones
eval/        unified_evaluator + eval_report (frozen 36-field EvalReport, incl. 5 welfare metrics)
experiments/
  sweep + _sweep_worker + run_registry      (run harness)
  ablate + ablations/*.yaml                 (canned ablation cells, now meta-tagged)
  suite/  cell.py + manifest.yaml + cells/  (the thesis experiment matrix)
  analysis/ registry_io + tb_scraper + tables + figures   (thesis table/figure renderers)
  stats + compare                           (Welch-t / Holm-Bonferroni / disclosure)
scripts/     run_suite, analyze_results, make_thesis_artifacts, train_main, run_full_tests,
             diagnose_mve, proposition_3_1_validation
_legacy_v4_7/  archived old "Non-stationary Tag" project (what the workspace CLAUDE.md describes)
```

## 4. The method — `HyperMuZeroModel`
`obs → RepresentationNet → latent s →` functional nets whose **weights are generated per-agent by a
hypernetwork** from a **tri-context** `[objective c_ctx, role, belief]`. A GRU **belief net** infers
hidden state (`c_hat`, opponent types `z_hat`), trained by its own loss + oracle-blended via the
**curriculum** (oracle → anneal → inference). The **MVE planner** ([`planning/mve_planner.py`](planning/mve_planner.py))
does per-agent coordinate descent with **CRN** → `π_mve` (load-bearing; no planner ⇒ uniform policy).
Trainer: K-step unroll; losses = policy + value + reward + consistency(BYOL) + belief; EMA target.
Hypernet generation scope (`full/film_head/base_gen/lora_fc2`) is the axis the `lora_*` cells vary.

## 5. The environment — `ResourceCommonsEnv` ([`envs/resource_commons/`](envs/resource_commons/))
N agents harvest K patchy resources on an L×L grid (actions = move/noop/harvest). Two **types**: ALPHA
(selfish) vs BETA (inequity-averse / Fehr-Schmidt). A **non-stationary context** `c_t∈[0,1]` shifts the
resource regime and flips BETA's reward sign. `c_t` and types are **oracle-only** info — the model must
infer them; external baselines assert they never read them. The env `info` exposes per-step physical
harvest (`harvests`) and resource stock (`resource_state`) — the signal the welfare metrics read.

## 6. Configs & presets
`V4Config.from_preset(name)` → frozen config (`env/model/train/mup/eval/legacy/baselines`). 11 presets
along 4 axes: size (`easy`/`medium`/`hard`) × non-stationarity (static vs `duo` random-walk) × hypernet
scope (`full`/`film_head`/`base_gen`) × LoRA. Cells pick a preset + `--override "section.field=value"`.

## 7. Eval & reporting
- **In-training**: `run_eval` ([`training/evaluation.py`](training/evaluation.py)) — dual mode (prior vs
  planner) over a c-grid; now also computes the **welfare family** (physical welfare, sustainability,
  fairness/Gini, tragedy) from per-episode harvest + final resource stock.
- **At sweep time**: [`eval/unified_evaluator.py`](eval/unified_evaluator.py) → **`EvalReport`**
  ([`eval/eval_report.py`](eval/eval_report.py), frozen 36-field, `schema_version="pkg08-spec01-v2"`):
  return/SEM, zero-shot seen/unseen gap, per-c/segment, regret, planner-prior gap, belief MAE, and the
  4 welfare metrics (`welfare_physical_mean`, `sustainability_mean`, `fairness_mean`, `tragedy_index_mean`;
  `return_mean` is social TOTAL welfare ΣR).
- **Bookkeeping**: [`experiments/run_registry.py`](experiments/run_registry.py) (append-only `registry.jsonl`).
- **Analysis**: [`scripts/analyze_results.py`](scripts/analyze_results.py) (generic compare / TB-scrape /
  disclosure) and [`scripts/make_thesis_artifacts.py`](scripts/make_thesis_artifacts.py) (renders Table
  6.1–6.7 + Fig 6.1–6.7 into `results/` via [`experiments/analysis/`](experiments/analysis/), reusing
  `stats.compare_methods` + `compare.render_plot`).

## 8. Entry-point cheat-sheet
```powershell
cd D:\RL\hyper_mve
# single run (hyper, directly via MuZeroTrainer):
python hyper_mve/scripts/train_main.py --preset medium --variant hyper --max_steps 1000
# the suite:
python hyper_mve/scripts/run_suite.py --list
python hyper_mve/scripts/run_suite.py --only abl3_easy_n2 --gpus 2,3 --slots-per-gpu 1
# analysis:
python hyper_mve/scripts/analyze_results.py --compare --registry runs/suite/main_comparison/registry.jsonl --reference hyper
python hyper_mve/scripts/make_thesis_artifacts.py --suite-root runs/suite --out results
# tests / diagnostics:
python hyper_mve/scripts/run_full_tests.py --mode fast
python hyper_mve/scripts/diagnose_mve.py
```
