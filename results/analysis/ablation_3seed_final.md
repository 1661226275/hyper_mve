# Ablation at 3 seeds — final (1M env-steps)

> **Regime ids here are v5 `g2`**: g0 mutual_coop, g1 mutual_comp,
> g2 asym_exploit, g3 asym_exploited, g4 neutral. Under the current `g2cm`
> family the same ids name different regimes and `mutual_comp` does not
> exist — see `results/analysis/g1_removal.md`. Numbers below are
> unedited.


Generated 2026-07-28 from `results/v5_final/registry.jsonl` via
`scripts/assemble_ablation.py --main-arm ref_bc_anneal_scaled_hardval_decoupled
--nosubj-arm ref_bc_anneal_scaled_no_subjective_decoupled`.

**Protocol.** Every number is the canonical registry protocol
(`eval_diagnostics.json`, 16 episodes, env seeds `10_000+97g+ep`, search RNG
`RandomState(12345)`) at the final checkpoint — the *same* protocol as the
headline comparison, so ablation and comparison numbers are directly
comparable. The `value_deploy_probe` protocol is NOT mixed in here; it scores
the same checkpoint ~12 points higher and is reported separately as extras.

- **Method of record:** `ref_bc_anneal_scaled_hardval_decoupled`
- **Module-1 control:** `ref_bc_anneal_scaled_no_subjective_decoupled` (same
  selection mode, same anneal; removing the per-regime heads leaves
  `--value_hard_select` nothing to act on, so the control is necessarily
  "plain MAZero + same anneal + same selection mode")

## Levels

| # | setup | s0 | s1 | s2 | mean ± sd |
|---|---|---|---|---|---|
| A1 | Module 2 = Bayes-avg (**ours**) | 60.91 | 60.34 | 64.56 | **61.94 ± 2.29** |
| A2 | Module 2 = argmax (MAP head) | 60.97 | 63.53 | 68.69 | **64.40 ± 3.93** |
| A3 | Module 2 removed (prior, no search) | 15.11 | 13.86 | 13.86 | 14.28 ± 0.72 |
| A4 | Module 1 removed (plain MAZero) | 21.88 | 27.92 | 43.11 | 30.97 ± 10.94 |
| A5 | Modules 1 & 2 removed | 15.11 | 15.11 | 18.04 | 16.09 ± 1.69 |
| UB | oracle — true `g` at deploy (**privileged**) | 58.25 | 62.76 | 66.80 | 62.60 ± 4.28 |
| — | `head_diversity` back-half | 9.54 | 7.98 | 3.94 | 7.15 ± 2.89 |

## Paired deltas (per-seed difference, then averaged)

| contrast | meaning | value |
|---|---|---|
| A1 − A3 | value of search, **with** role-aware heads | **+47.66 ± 2.65** |
| A4 − A5 | value of search, **without** role-aware heads | **+14.88 ± 9.32** |
| A1 − A4 | value of Module 1 | **+30.97 ± 8.88** |
| A1 − A2 | value of Bayes-averaging over hard MAP | **−2.46 ± 2.13** |
| UB − A1 | headroom lost to imperfect regime inference | **+0.67 ± 2.88** |

## What this supports

**Modules 1 and 2 are synergistic.** Search is worth +47.7 when the role-aware
per-regime heads are present but only +14.9 without them — a 3.2× difference,
consistent in sign across all three seeds. Neither module carries the result
alone: removing Module 1 costs 31.0 points, removing Module 2 costs 47.7.

**`head_diversity` stays well above zero** on every seed (9.54 / 7.98 / 3.94),
so the per-regime heads remain genuinely differentiated at 1M — they do not
re-collapse after the prior-collapse fix. Note the downward trend across seeds
tracks nothing in the return (s2 has the *lowest* diversity and the *highest*
return), so diversity is a mechanism-validity check, not a quality metric.

## What this does NOT support

**Bayes-averaging over the belief posterior is not earning its place.**
A1 − A2 = −2.46 ± 2.13: hard MAP selection is at least as good on every seed
and better on two. Three independent measurements agree that the *posterior*
contributes little:

1. A1 − A2 = **−2.46** (averaging loses to argmax)
2. UB − A1 = **+0.67 ± 2.88** (the true regime id buys nothing — belief accuracy
   is not the limiter; the value heads are)
3. `belief_blind`, with the posterior forced uniform, scored **62.94** at
   `head_diversity` exactly 0.000

The defensible claim is about the **role-aware per-regime value heads plus
search**, not about Bayesian belief-averaging as the inference rule. Module 2
should be described as *regime-conditioned planning*, with the averaging-vs-MAP
choice reported honestly as a wash-to-slightly-negative.

## Caveats

- **A4 is the noisiest cell** (sd 10.94; 21.88 → 43.11). The Module-1 and
  synergy magnitudes inherit that spread — treat +30.97 and the 3.2× ratio as
  approximate. The *signs* are stable across seeds; the magnitudes are not.
- **`belief_blind` (62.94) came from the older logging regime**, before the
  env-steps axis and 200-step cadence. It is reported as a qualitative null,
  not as a fourth column in the table above.
- UB is a privileged upper bound using the true regime id at deploy time. It is
  a diagnostic reference only and never touches `return_mean`.

## Corrections to earlier statements in this work

- An earlier note put A4 − A5 at **+6.8** and the synergy ratio near 7×. That
  was **seed 0 only**. At 3 seeds it is **+14.88 ± 9.32**, ratio **3.2×**. The
  synergy conclusion stands; the magnitude was overstated.
- An earlier note reported the method of record as "robust 74.09 ± 2.16". That
  is the *last-20%-of-env-steps* statistic from the training curves, not the
  final-checkpoint registry protocol used here (61.94 ± 2.29). Both are valid;
  they must not be mixed in one table.
