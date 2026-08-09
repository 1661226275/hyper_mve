# Part 0b — main-method arm decision (2026-07-24)

> **Regime ids here are v5 `g2`**: g0 mutual_coop, g1 mutual_comp,
> g2 asym_exploit, g3 asym_exploited, g4 neutral. Under the current `g2cm`
> family the same ids name different regimes and `mutual_comp` does not
> exist — see `results/analysis/g1_removal.md`. Numbers below are
> unedited.


## Result: budget-proportional anneal RECOVERS the 600K→1M regression

`ref_bc_anneal_scaled` @1M env-steps (`results_competence_1M_v2`, old train-steps logging):

| run | budget | return_mean | note |
|---|---|---|---|
| **ref_bc_anneal_scaled** | **1M** | **67.49** | g0=93.4 g1=-0.7 g2=56.1 g3=68.3 g4=120.3; regime_acc=0.554 |
| plain ref_bc | 1M | 49.82 | the regression |
| plain ref_bc | 600K | 66.59 | pre-regression baseline |
| ref_bc_hardval | 600K | 62.56 | hard-value-select |

The hardcoded 28000-step BC anneal is 74.7% of training at 600K but only 44.8%
at 1M; preserving the **fraction** (→ 46667 steps at 1M) recovers return
49.82 → **67.49** (+35%), back above the 600K number. The schedule was the
primary driver of the regression.

## Decision
**Main method = `ref_bc_anneal_scaled`** (+ centralized MCTS default + Bayes leaf).
Module-1 ablation = **`ref_bc_anneal_scaled_no_subjective`** (new arm: plain
MAZero carrying the same scaled anneal, so the ablation changes only the
subjective module). Runner applies the scaled anneal to any arm whose name
contains `anneal_scaled`.

Both verified: subjective True/False, decoupled_selection=False (centralized),
reference_episode_anneal_steps=46667, test_interval=200.
