# v7 stage-1: what ran, what it measured, and what the next session must do (2026-08-10)

Handoff for a fresh session. Branch `worktree-remove-g1`, pushed, tip `7928fca`.

**The results below are NOT to be analysed here.** The next session re-runs the
experiments in full per §4; the user analyses them. What matters in this
document is the *measurement infrastructure* that stage 1 validated, and the
four defects it exposed — those carry forward regardless of the numbers.

---

## 1. What stage 1 actually was

Six algorithms, `relation_coopmix` (family `g2cm`), seed 0, 200k env steps,
128 eval episodes per regime. Method arm
`ref_bc_anneal_scaled_hardval_decoupled`. All six completed and wrote
`eval_report.json` with `schema_version = rel-v3`.

Vendored baselines are reached from the worktree by symlinking
`hyper_mve/comparison/vendor/*` from the main checkout — the main checkout has
~1442 uncommitted lines of unrelated work, so the merge the previous handoff
assumed is still blocked. The symlinks resolve transparently (`.resolve()`
lands on the module file, not the vendor dir).

### Infrastructure that is now proven

* `rel-v3` `return_per_regime_per_agent` on `EvalReport`, populated by all six
  runners. The invariant `sum(per-agent) == scalar per-regime return` holds
  exactly for every algorithm × regime — this exercises all five hand-edited
  evaluate loops plus the native path, which was stage 1's stated purpose.
* Per-agent TB tags emit *during* training, not only at final eval.
* `scripts/v7_readout.py` assembles the headline: per-role table, global reward
  on g0/g3/g4, role-swap consistency, NashConv with a comparability guard.
* Post-hoc NashConv adapters for happo and mbom (written blind in the previous
  session) execute correctly at scale.

---

## 2. Four defects stage 1 exposed — these are the reason it was worth running

### 2.1 mappo is not trained — DECISION (2026-08-10): mappo is dropped

**User decision: mappo is removed from all subsequent training.** The roster
becomes `mazero_mixed` (method) vs `happo`, `mamba`, `mbom`, `m3w_adapted`.

Two consequences that must be honoured in the writeup:

* **Never report that MAPPO performs poorly.** Its stage-1 score (global reward
  22.60 vs happo's 115.70) is a configuration artefact of the update cadence
  below, not a property of the algorithm. Dropping it is legitimate; citing it
  as a weak baseline would not be.
* **Any archived table carrying mappo numbers from this configuration is
  suspect** for the same reason, and should not be reused as a reference point.

The measurement below is retained because it is the evidence for the decision,
and because the same cadence audit must be applied to the remaining four
baselines before they are compared.

Gradient updates over the whole 200k-step run, read from TB
`progress/train_steps`:

| algo | train_steps | updates / 1k env steps | eval points | walltime | min / 100k |
|---|---|---|---|---|---|
| m3w_adapted | 50000 | 250.0 | 251 | 4.49 h | 134.7 |
| mazero_mixed | 12401 | 62.0 | 64 | 2.48 h | 74.5 |
| happo | 2000 | 10.0 | 52 | 0.31 h | 9.2 |
| mamba | 2000 | 10.0 | 52 | 7.52 h | 225.7 |
| mbom | 500 | 2.5 | 52 | 4.71 h | 141.2 |
| **mappo** | **62** | **0.31** | **2** | 0.08 h | 2.5 |

mappo (dropped) gets 62 updates — 32× fewer than happo, 800× fewer than m3w — and
produces 2 eval points, so it has essentially no sample-efficiency curve. Its
low score is a configuration artefact, not an algorithmic finding — which is
why it is dropped rather than reported.

**The cadence disparity does not end with mappo.** Among the four surviving
baselines the spread is still 100× (mbom 2.5 vs m3w 250 updates per 1k env
steps), and "equal env steps" therefore does not mean "equal training". Decide
explicitly what is being held constant across the comparison — env steps,
gradient updates, or wall-clock — and state it, because the three give
different rankings. This table is the seed-0 answer to TODO #2.

### 2.2 NashConv can silently measure a policy the algorithm never plays

`--frozen-mode prior` freezes mazero_mixed's distilled prediction net, which
reproduces only **28 %** of the deployed MCTS planner's return (g0: 33.48 vs
114.02). Its prior-mode NashConv therefore lands next to the weakest baseline
while saying nothing about the method. The model-free baselines are at 97–100 %,
so their numbers mean what they appear to mean.

`v7_readout.py` now computes this ratio per row and prints
`**NOT COMPARABLE**` below 80 %. **Keep that guard.**

**DECISION (2026-08-10): mazero_mixed deploys via the MCTS planner, not the
prior.** Therefore:

* Every mazero NashConv must use `--frozen-mode planner` (or `both`).
  **The prior-mode number must not be reported at all** — it describes a
  policy that is never deployed.
* Budget for the cost. The planner runs a full search at *every* BR env step
  (~1 M searches at br = 100k × 2 agents × 5 regimes); the stage-1 `both` job
  was still running after 7 h. Prior-mode is ~1.9 h, so essentially all of that
  is the planner pass. This is the single most expensive measurement in the
  wave and should be scheduled first, not last.
* The same applies to m3w_adapted, which is also search-based — and which is
  additionally *excluded* from NashConv entirely because its planner is
  regime-CONDITIONED while `FrozenExternalPolicy`'s contract is `(obs, t)` with
  no regime, so an adapter would silently plan as g0 in every regime.
  Supporting it means threading the regime through `FrozenExternalPolicy`.

### 2.3 NashConv is a lower bound, and `0.000` is ambiguous

`NashConv = Σ_i max(0, V_i(BR_i, π_−i) − V_i(π))` (`game_metrics.py:426`), so a
best response that fails to beat the incumbent reports exactly `0.000`. That
reads as "at equilibrium" but can equally mean the BR was under-resourced.
Distinguishing the two requires raising the budget.

Budget curve, measured on the mappo checkpoint (5 regimes × 2 agents):

| br_steps | sum NashConv | Δ per doubling |
|---|---|---|
| 2 000 | — | two regimes reported exactly 0.000 |
| 5 000 | 71.2 | — |
| 20 000 | 301.9 | — |
| 50 000 | 387.3 | +8.2 % (→100k) |
| 100 000 | 419.0 | +3.8 % (→200k) |
| 200 000 | 435.1 | extrapolated asymptote ≈ 451 |

br = 100 000 captures ≈ 93 % of the extrapolated asymptote; 20 000 captures
≈ 72 %. **Always report `br_env_steps` beside the number, and only compare at
equal budget.** Cost is ~12 s per 1000 br_steps for a model-free baseline.

Correction to the previous handoff: it called in-training NashConv infeasible
at 20k BR env-steps. Measured, that budget costs ~4 min per checkpoint, so the
in-training trend tier is affordable after all. The real obstacle is 2.3+2.4,
not cost.

### 2.4 mbom NashConv does not scale

The mbom adapter deliberately runs **without** `no_grad`, because MBOM's
`choose_action` performs imagined opponent-model fine-tuning as part of its
acting path; suppressing it would freeze a weaker policy than the one that
deploys. Consequence: ~17× the per-BR-step cost of mamba, ≈ **11.5 h per
checkpoint** at br = 100k. Three seeds ≈ 35 GPU-hours for mbom alone, before
ablation arms. Hence TODO #1.

Measured per-BR-step cost (br = 1000 probe, ~40 s fixed startup subtracted;
the model reproduces mamba's observed 100k runtime exactly):

| algo | ≈ s / 1k br_steps | predicted 100k |
|---|---|---|
| mamba | 24 | ~40 min |
| mazero_mixed (prior) | 68 | ~1.9 h |
| mbom | 415 | ~11.5 h |

---

## 3. Facts the next session needs and should not re-derive

* **Current method of record** (`ref_bc_anneal_scaled_hardval_decoupled`) =
  `ref_bc` + `--value_hard_select --decoupled_selection` (`arms.py:209`). It
  passes **no** `--policy_target_type`, so it uses the default **`visit`**
  (visit-count target, `config.py:131`), and **no** `--root_cover star`.
  This answers "先确定当前使用的是什么方法" in TODO #4.
* Search-target arms that already exist: `_qtarget` (q_softmax only), `_cover`
  (star root enumeration only), `_mctsfix` (both), `_agentq`
  (agent_q_softmax), `_blend_t025/t05/t1/t2` (visit_q_blend τ sweep).
  Choices in `config.py:132`: `visit`, `q_softmax`, `visit_q_blend`,
  `agent_q_softmax`, `agent_q_blend`.
* **`marginal_visit_count` is a provable no-op** — do not spend a run on it.
* Role-awareness (Module-1) control: `ref_bc_anneal_scaled_no_subjective_decoupled`,
  matched to the method (same anneal, same decoupled selection). Disclosed
  asymmetry: removing the subjective module deletes the per-regime value heads,
  so `--value_hard_select` is necessarily absent (`arms.py:225`).
* **Two regime sets that must stay separate** (they coincided under the old
  `g2` family, which is how they were conflated):
  * `global_reward = (0, 3, 4)` — a *scoring* choice, applies where all of
    g0–g4 are trained and tested.
  * `seen/held = (0,1,2)/(3,4)` — the train/holdout *partition*, used only by
    the generalization experiment.
* `g2` ids 1–3 mean different things than `g2cm` ids 1–3. Key per-regime
  fields on `RegimeFamily.names()`, never the integer.
* g2 `asym_exploited` is g1 `asym_exploit` with the agent indices swapped
  (`relation.py:220`), which is what makes the role-swap diagnostic valid.
* λ = 1.0 (`env_config.py:54`).
* happo cannot be found with `pgrep -f eval_game_metrics` — vendored HARL calls
  `setproctitle` and renames the process `happo-relation-runner`. Track by PID.
* Launch detached with `setsid nohup`, never tmux: the tmux server died once
  mid-wave and took every run with it. `scripts/launch_grid.sh:17` hardcodes
  the main checkout path and is unusable from a worktree.
* `git add -A` mid-run sweeps live TB files into the commit. `/results_*/` is
  now gitignored; `results/` stays tracked.

---

## 4. TODO for the next session

0. **mappo 剔除于之后的训练**（2026-08-10 决定，见 §2.1）。Roster is
   `mazero_mixed` vs `happo`, `mamba`, `mbom`, `m3w_adapted`. Do not describe
   MAPPO as a weak baseline anywhere — it was misconfigured, not outperformed.

1. **将 mbom 从 NashConv 表中剔除，仅报告其按角色划分的指标。**
   Rationale measured in §2.4 (≈11.5 h/checkpoint). Per-role metrics for mbom
   cost nothing extra and stay in.

2. **弄清各算法每 100K env_steps 所需要的训练时间、train_steps 次数（梯度更新次数）、
   eval 次数（指标记录到 tensorboard 的次数）。**
   The §2.1 table is the seed-0 answer. With mappo dropped the spread among the
   four survivors is still ~100× (mbom 2.5 vs m3w 250 updates per 1k env
   steps), so the open question is not "record the numbers" but **decide what
   is held constant** across the comparison — env steps, gradient updates, or
   wall-clock — and say so, since the three orderings differ.

3. **整体实验顺序**（每一步都是所有 g 参与训练）：
   1. seed0：世界模型真实度 (fidelity) → 全局 reward → per-agent reward → NashConv
   2. 消融实验 + seed0 的泛化性对比实验
   3. 全部完成后，依次运行 seed1、seed2

   For the generalization experiment specifically, `rel_coopmix.py:94` still
   has `train_regime_ids=(0,1,4)` and must become `(0,1,2)`. Two tests then
   break: `tests/configs/test_presets_coopmix.py:65-66` (names — simple
   update) and `:75-89` `test_..._covers_every_own_row_value`, which asserts
   the training own-rows are `{+1,−1,0}` and **will fail** because `(0,1,2)`
   trains only on own-rows `{+1,−1}`. Do not delete it — rewrite it to pin the
   new invariant. Be precise about what that invariant is (verified from
   `_w2`: own-row `w01` is g0 `+1`, g1 `−1`, g2 `+1`, g3 `−1`, g4 `0`):
   **only g4 tests an unseen own-row value** (zero, never trained); **g3's
   `−1` is already seen in g1**, so g3 tests an unseen *combination* (partner
   row `0` instead of `+1`). The two held-out regimes therefore probe
   different things, and a rewritten test should say so rather than assert a
   uniform claim. Then re-run
   `scripts/probes/regime_voi_probe.py --preset rel_coopmix`. Expect held-out
   `regime_accuracy` near zero: the head has 5 classes, only 3 occur in
   training. Chance accuracy stays 0.200 under `g2cm` (|G| is still 5).

4. **消融实验仍为两大模块**：
   * 角色感知模块消融 — use `ref_bc_anneal_scaled_no_subjective_decoupled`.
   * 搜索模块消融 — 对比 `marginal_visit_count`、`q_softmax`、`visit-count`、无树搜索。
     Two corrections from §3: the current method already **is** visit-count, so
     that cell needs **no retraining**; and `marginal_visit_count` is a proven
     no-op, so it buys nothing. The tree-search arms were originally selected
     on g1 NashConv, and g1 no longer exists — so they need re-selection
     against the new headline regardless.

5. **将所有算法的 `train_steps` 统一到 ~20 次 / 1K env steps；`eval` 改为每 **2K**
   env 一次 + `episodes_per_regime = 2`**（2026-08-10 决定）。At 200k env steps
   that is **4 000 train_steps** and **~100 eval points** per run.

   Current values and the factor each must move (from §2.1):

   | algo | now (upd/1k) | → 20/1k | now (evals) | → ~100 |
   |---|---|---|---|---|
   | m3w_adapted | 250.0 | ÷12.5 | 251 | ÷2.5 |
   | mazero_mixed | 62.0 | ÷3.1 | 64 | ×1.6 |
   | happo | 10.0 | ×2 | 52 | ×1.9 |
   | mamba | 10.0 | ×2 | 52 | ×1.9 |
   | mbom | 2.5 | ×8 | 52 | ×1.9 |

   **Where the knobs are:**
   * eval cadence: `_PROBE_EVERY_ENV_STEPS = 4000` in `happo.py:45`,
     `mamba.py:57`, `mbom.py:62` → set to `2000`. m3w_adapted instead uses
     `every_train_steps=200` (`m3w_adapted/runner.py:201`) and so is already at
     ~1.25 evals/1k; re-derive it after its update cadence changes, since its
     probe is keyed to train steps, not env steps. mazero_mixed has its own
     probe in `mazero_mixed/core/train.py`.
   * eval episodes: `episodes_per_regime=8` at every `PeriodicEvalProbe`
     construction site → set to `2`.

   **Why the episode count moves with the cadence.** An episode is 100 steps
   (`env_steps_evaluated 64000 / episodes_total 640`), so one eval point costs
   `episodes_per_regime × 5 regimes × 100` env steps:

   | setting | eval points | eval rollout cost / 200k run | vs training budget |
   |---|---|---|---|
   | now: every 4k, 8 eps | 52 | 200 000 steps | 1.0× |
   | every 1k, 8 eps | 200 | 800 000 steps | **4.0×** |
   | every 1k, 2 eps | 200 | 200 000 steps | 1.0× |
   | **chosen: every 2k, 2 eps** | **100** | **100 000 steps** | **0.5×** |

   Evaluation *already* costs as much as training at the current setting, and
   for the two search-based algorithms (mazero_mixed, m3w_adapted) every eval
   action runs an MCTS search — so eval cost is not a rounding error. The
   chosen setting nearly doubles the curve points (52 → 100) while **halving**
   eval cost, at ~2× the per-point noise. Headline numbers are unaffected: the
   final `eval_report` still uses 128 episodes per regime.

   Also note the two directions are not symmetric in cost: raising mbom's
   updates 8× and mamba's 2× makes the two slowest baselines slower still
   (mamba is already 7.52 h / 200k), while m3w ÷12.5 and mazero ÷3.1 get
   cheaper. Re-measure walltime after the change rather than assuming it nets
   out.

### 限制

**实验不是越多越好，我们要沿着理论主线只做必要的实验，否则会导致理论发散。**

Applied to the above: prefer the cells that discriminate the theory. Concretely
— skip `marginal_visit_count`; do not retrain the visit-count cell; drop mbom
from NashConv rather than paying 11.5 h/checkpoint for a number that will be
reported with a caveat anyway; and do not add seeds until §4.3's first two
tiers are complete, since a second seed multiplies every unresolved defect.
