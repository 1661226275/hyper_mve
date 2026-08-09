# Method of record — MixMAZero (locked 2026-07-27)

## The method

**Ablation arm:** `ref_bc_anneal_scaled_hardval_decoupled`
**Registry variant:** `mazero_mixed_ref_bc_anneal_scaled_hardval_decoupled`

| component | setting | flag |
|---|---|---|
| subjective / role-aware hypernet heads | **on** | `--subjective_model` |
| per-regime value-head training | **hard-select the true regime head** | `--value_hard_select` |
| leaf value at deploy | **Bayes average over the belief posterior** | `set_value_deploy("bayes")` |
| MCTS selection | **per-agent decoupled** | `--decoupled_selection` |
| behaviour-cloning guidance | coeff 1.0, prob 0.5 → 0.1 | `--bc_loss_coeff 1.0` |
| BC anneal | **budget-proportional (74.7% of training)** = 46 667 steps @1M | `--reference_episode_anneal_steps` |
| budget / env | 1 000 000 env-steps, `--env relation` (all 5 regimes seen) | |
| search width | `--num-pmcts 16`, 25 simulations | |
| eval | 16 episodes/regime, periodic probe every 200 train steps | |

Training is CTDE: the true regime id is used at **train time only** (belief supervision and
`value_hard_select`). Deployment is oracle-free — the belief net infers the regime from
observations and the planner Bayes-averages the value heads.

## Why this version (seed 0, 1M, robust statistic)

Selected from the complete `{plain, hardval} × {centralized, decoupled}` screen
(`stageA_selection_2x2.md`). `last-20%` = mean of the run's own periodic `eval/return_mean` over
the final 20% of training; the single final-checkpoint number has sd ≈ 8–12 and is not a usable
selection basis (`late_training_instability.md`).

| cell | last-20% | sd | head_diversity | regime_acc |
|---|---|---|---|---|
| plain, centralized | 64.95 | 9.61 | 1.33 | 0.513 |
| plain, decoupled | 58.36 | 12.03 | 2.07 | 0.200 |
| hardval, centralized | **73.41** | 9.10 | 11.58 | **0.200 (chance)** |
| **hardval, decoupled ← SELECTED** | 71.95 | **7.70** | 9.54 | **0.561** |

Rationale: both hardval cells lead on return and on head diversity, and their returns are
statistically indistinguishable (Δ 1.46 vs sd ≈ 8–9). The centralized cell's belief net sits at
**chance** regime accuracy (0.200 = 1/5, a near-constant posterior — this test SURVIVES the
2026-08-09 scope change, because `g2cm` also has |G| = 5, so chance stays 0.200 and a
collapsed posterior still reads as collapsed), which would undercut a
role-aware claim regardless of return. The decoupled cell has genuine above-chance inference
(0.561) with strongly differentiated heads (9.54) and the lowest variance of any cell.

Reference: mamba (strongest baseline) last-20% = 71.91, sd 10.39 — the method matches it.

## Result — all 3 seeds complete (1M env-steps, `--env relation`)

| seed | endpoint | **robust (last-20%)** | sd | `head_diversity` | `regime_acc` | UB − A1 |
|---|---|---|---|---|---|---|
| 0 | 60.91 | 71.95 | 7.70 | 9.54 | 0.561 | −2.65 |
| 1 | 60.34 | 73.27 | 10.40 | 7.98 | 0.564 | +2.42 |
| 2 | 64.56 | 77.05 | 8.22 | 3.95 | 0.571 | +2.23 |
| **mean** | **61.94 ± 1.87** | **74.09 ± 2.16** | | 7.16 | **0.565 ± 0.004** | +0.67 ± 2.35 |

Reference (seed 0 only so far): mamba at its **upstream 8.4M** capacity — 6.7× the method —
robust **71.91**; m3w_adapted 33.38; mbom 20.30; happo 12.08. Capacity-matched baselines
(`mamba_pm`/`happo_pm`/`mbom_pm`) are in flight.

Notable: `regime_accuracy` is remarkably stable at **0.565 ± 0.004** — well above the 0.200
chance level, and unlike the centralized cell (0.200, a collapsed posterior), which is what the
selection was made on. The seed-to-seed spread of the robust statistic (±2.16) is far smaller
than the within-run late-training oscillation (sd ≈ 8–10), which is why the robust statistic is
the right basis for every comparison.

## Honest caveats to carry into the write-up

- **Oracle deploy ≈ bayes across all 3 seeds** (−2.65 / +2.42 / +2.23, mean +0.67 ± 2.35 against
  a within-run sd of ≈8): perfect regime knowledge yields **no reliable benefit**, so regime
  inference is not the binding limiter. Note this is *not* evidence that the true regime hurts —
  an earlier single-seed reading suggested that and the sign turned out unstable
  (see `oracle_deploy_1M_registry_protocol.md` §CORRECTION).
  Two readings remain consistent with the data and cannot be separated by it: the belief is
  already accurate enough (0.565) that residual error costs nothing, **or** the value heads'
  regime-specificity does not translate into better action selection. With `belief_blind`
  (uniform posterior, head diversity 0.000) still scoring 62.94, the posterior's
  *informativeness* is clearly not the main driver of return.
- Runs oscillate by sd ≈ 8–12 late in training; all headline numbers must use the robust
  statistic, with the endpoint value reported alongside.
- Comparison against baselines is pending the capacity-matched runs; the current baseline
  reference points are seed 0 only.

## Retention policy (user instruction, 2026-07-27)

**All versions' comparative results are retained** in `results/v5_final/registry.jsonl` — the
non-selected 2×2 cells (`…_anneal_scaled`, `…_decoupled`, `…_hardval`) and the earlier arms stay
on disk for manual processing. Nothing is pruned.

## Matched ablation control

`ref_bc_anneal_scaled_no_subjective_decoupled` — plain MAZero, same scaled anneal, same decoupled
selection; differs from the method only by Module 1. (`--value_hard_select` is necessarily absent:
with no subjective module there are no per-regime heads to hard-select.)
