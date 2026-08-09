# Belief settle time on rel_recip (2026-08-09)

Measured **before** the g1-removal scope change, on the untouched `g2` family, because
once the family identity changes there is no comparable 5-regime baseline left. Method
of record arm, `rel_recip`, seeds 0 and 1:
`results_v6_2x2/mazero_mixed_ref_bc_anneal_scaled_hardval_decoupled/relation_recip/seed{0,1}`.

```bash
$PY scripts/probes/belief_confusion_probe.py --gpus 3 --episodes 16 --planner-dist \
  --run-dir .../relation_recip/seed0 --run-dir .../relation_recip/seed1
```

Raw: `results/analysis/belief_ceiling/belief_settle_time_rel_recip_ep16.json`.
**Rollout policy: the distilled-prior rollout** (the one that reproduces `regime_accuracy`
bit-for-bit and is what actually acts). The planner-rollout cross-check is reported
separately at the end, at 4 episodes.

## Headline

**Settle time is 0 exactly where there is nothing to infer, and undefined exactly where
inference is required.** The belief posterior is a deterministic function of the agent's
own observed row; it performs no inference at all.

## The confusion structure is the whole result

Counts are steps (16 episodes x 100 steps x 2 agents = 3200 per regime).

| g_true | seed 0 predictions | acc | seed 1 predictions | acc |
|---|---|---|---|---|
| g0 mutual_coop | 2426 g0, 774 g3 | 0.758 | 3200 g0 | 1.000 |
| g1 mutual_comp | 3200 g1 | 1.000 | 2704 g1, 496 g3 | 0.845 |
| g2 asym_exploit | 1376 g0, 1600 g1, 224 g3 | **0.000** | 1600 g0, 1201 g1, 399 g3 | **0.000** |
| g3 asym_exploited | 1376 g0, 1600 g1, 224 g3 | 0.070 | 1600 g0, 1029 g1, 571 g3 | 0.178 |
| g4 neutral | 3200 g4 | 1.000 | 3200 g4 | 1.000 |

Two facts to read off it:

1. **`asym_exploit` is a dead class.** Column g2 is zero in *every* row, on *both* seeds.
   The head never emits that label — not merely "never correct in g2", but never
   predicted anywhere, 16000 steps per seed.
2. **Every prediction is the symmetric member of the agent's own-row bucket.** Agent 0
   observes `w_01`, agent 1 observes `w_10`; each bucket holds one symmetric and one
   asymmetric regime, and the net picks the symmetric one deterministically. In g2, agent
   0 sees `w_01 = -λ` (candidates {g1, g2}) and answers g1 on all 1600 of its steps; agent
   1 sees `w_10 = +λ` (candidates {g0, g2}) and answers g0 on 1376 of 1600. That is a
   maximum-a-priori tie-break, not a posterior — a genuine posterior under the uniform
   training prior would be ~50/50 within the bucket.

This is why headline accuracy sits at 0.565 / 0.605: the three symmetric regimes are free,
the two asymmetric ones are lost. 3/5 = 0.60.

## Settle time t*

First step at which the posterior is correct **and** stays above the confidence bar for
the whole rest of the episode; `never%` is the fraction of (episode, agent) trajectories
that never reach such a step. `regime_switch_prob = 0` makes the regime static within an
episode, which is what makes t* well-defined. Episode length is 100.

Seed 0:

| regime | thr 0.35 never% / med | thr 0.50 | thr 0.70 | value swing |
|---|---|---|---|---|
| g0 mutual_coop | 25.0% / 32.5 | 40.6% / 76.0 | 93.8% / 89.5 | 5.66 |
| g1 mutual_comp | **0.0% / 0** | 100% / — | 100% / — | 6.92 |
| g2 asym_exploit | **100% / —** | 100% / — | 100% / — | 6.41 |
| g3 asym_exploited | 93.8% / 98.5 | 100% / — | 100% / — | 6.12 |
| g4 neutral | **0.0% / 0** | 0.0% / 0 | 0.0% / 0 | **0.000** |

Seed 1:

| regime | thr 0.35 never% / med | thr 0.50 | thr 0.70 | value swing |
|---|---|---|---|---|
| g0 mutual_coop | **0.0% / 0** | 100% / — | 100% / — | 3.56 |
| g1 mutual_comp | 25.0% / 34.5 | 59.4% / 76.0 | 87.5% / 96.0 | 4.13 |
| g2 asym_exploit | **100% / —** | 100% / — | 100% / — | 4.03 |
| g3 asym_exploited | 65.6% / 75.0 | 100% / — | 100% / — | 4.40 |
| g4 neutral | **0.0% / 0** | 0.0% / 0 | 0.0% / 0 | **0.000** |

`g2` never settles at any threshold on either seed — necessarily, since it is never
predicted. The regimes that settle at t* = 0 are exactly the ones the own-row read
answers for free, and they differ between seeds (g1 on seed 0, g0 on seed 1) according to
which member of each bucket that seed's net collapsed onto.

The threshold sweep matters: at 0.70 almost everything stops settling, so the posterior
is not merely correct-or-wrong but weakly confident throughout. Reporting a single
threshold would have made this look like a much cleaner result than it is.

## Pairing with the value swing — the part that decides what it means

Per the standing QMDP / frozen-belief-root result
(`hyper-mve-bayes-averaging-negative-result`), a fast settle time on its own shows
nothing. Paired here:

- **g4 neutral: settles at t* = 0, value swing exactly 0.000.** Knowing the regime is
  worth literally nothing there, and the belief resolves instantly. This is the cleanest
  possible demonstration that settle time and usefulness are independent axes.
- The regimes with a *large* swing (g2 at 6.41 / 4.03) are precisely the ones that never
  settle. Where the belief would change the value, the net cannot supply it; where it
  supplies it instantly, it changes nothing.
- Calibration is consistent with a tie-break rather than a posterior: mean max-prob 0.668
  when correct vs 0.479-0.481 when wrong, i.e. confidently right on the free regimes and
  hedged-but-still-wrong on the rest.

## Planner-rollout cross-check

Run separately at 4 episodes (`--planner-dist`; the compiled `cytree` extension is
required and is a gitignored build artifact, absent from fresh worktrees):

| seed | prior rollout | planner rollout | per-regime under planner |
|---|---|---|---|
| 0 | 0.5565 | 0.5913 | g0 0.80, g1 1.00, **g2 0.00**, g3 0.16, g4 1.00 |
| 1 | 0.6165 | 0.6170 | g0 1.00, g1 0.78, **g2 0.00**, g3 0.30, g4 1.00 |

`g2 = 0.00` under the de-collapsed planner rollout too, on both seeds. The dead class is
not an artifact of the prior rollout's distribution.

## What this implies for the scope change

It is orthogonal to it, and that is the useful part. The failure is structural — the net
resolves each own-row bucket to a fixed member — and `g2cm` has the same bucket geometry
(`+λ -> {mutual_coop, asym_exploited}`, `-λ -> {asym_exploit, asym_exploit_mild}`), so
removing `mutual_comp` neither causes nor fixes it.

One incidental gain: under `g2` both buckets held exactly one symmetric regime, so
"collapses to the symmetric member" and "collapses to the lower id" are indistinguishable
in this data. Under `g2cm` the `-λ` bucket holds two *asymmetric* regimes, which separates
those two explanations the first time the probe is run on a `rel_coopmix` checkpoint.

## Caveats

- Two seeds. The collapse direction differs between them; the collapse itself does not.
- Prior-rollout numbers are 16 episodes; the planner cross-check is 4.
- t* is defined against a static within-episode regime. If `regime_switch_prob` is ever
  raised, this definition does not carry over.
