# Seed variance dominates every arm difference in the v7 wave

**2026-08-12, `results_v7_500k`, 500k env steps, robust last-20% statistic.**
This supersedes the variance assumptions in `v7_search_module.md` and
`v7_module1_evidence.md`. Both were written when only seed 0 existed and both
used a **±2.16** seed sd imported from the 3-seed method of record — a different
environment at a 1M budget. Measured here, that figure is wrong by more than an
order of magnitude for the `mazero_mixed` family.

## 1. Measured spread per arm

| arm | n | mean | sd | min | max | range |
|---|---|---|---|---|---|---|
| happo | 3 | 99.26 | **0.76** | 98.54 | 100.05 | 1.51 |
| mamba | 2 | 86.91 | **0.18** | 86.78 | 87.04 | 0.26 |
| m3w_adapted | 2 | 97.23 | 2.69 | 95.33 | 99.13 | 3.80 |
| mbom | 2 | 72.95 | 16.72 | 61.13 | 84.78 | 23.65 |
| no_subjective | 3 | 95.47 | 11.23 | 82.51 | 102.04 | 19.53 |
| **method** | 3 | **78.55** | **39.25** | **33.24** | 102.35 | **69.11** |
| margvisit (null control) | 2 | 97.66 | 6.72 | 92.91 | 102.41 | 9.50 |
| mctsfix | 2 | 87.15 | 6.90 | 82.27 | 92.03 | 9.76 |
| qtarget | 2 | 65.92 | 28.53 | 45.74 | 86.09 | 40.35 |
| cover | 2 | 52.68 | 54.90 | **13.86** | 91.50 | 77.64 |

The model-free baselines are stable to within 0.2–2.7. The `mazero_mixed` family
is not: the method's three seeds are 102.35 / 100.05 / **33.24**, and the `cover`
arm's two are 91.50 / **13.86**. These are not crashes — every run logged its full
501,619 env steps and exited cleanly in the usual ~6.5–7 h.

**No between-arm difference in the mazero family currently exceeds its own
within-arm seed spread.** At n=2–3 the family cannot be ranked.

## 2. The null control puts a number on the attribution floor

`..._margvisit` is a *provably identical* estimator to the method's `visit`
target — `test_marginal_visit_equals_visit_loss_exactly` asserts bitwise-equal
losses. So method-vs-margvisit **at the same seed** is a direct measurement of
run-to-run nondeterminism with the arm held constant:

| seed | method | margvisit | \|diff\| |
|---|---|---|---|
| 0 | 102.35 | 102.41 | **0.06** |
| 1 | 100.05 | 92.91 | **7.14** |

The mechanism is not mysterious: `scatter_add_` on CUDA is non-deterministic, so
two mathematically identical runs diverge numerically and training amplifies the
difference. `v7_search_module.md` reported the 0.06 as *the* nondeterminism
floor. That was a single sample of a quantity whose second sample is 119x larger,
and the claim should not have been made from n=1.

**Working figure: differences below ~7 points between `mazero_mixed` arms are not
attributable to the arm.** That threshold swallows the Module-1 gap (0.31) and
most of the search-arm gaps.

## 3. What predicts a collapsed run: head diversity

Across all 13 finished `mazero_mixed` runs that log both, late-training
`train/head_diversity` and robust return move together:

**Pearson r = 0.818, Spearman rho = 0.940 (n = 13).**

| run | head div | robust |
|---|---|---|
| margvisit s0 | 162.98 | 102.41 |
| method s0 | 129.39 | 102.35 |
| method s1 | 136.74 | 100.05 |
| margvisit s1 | 57.03 | 92.91 |
| cover s0 | 72.05 | 91.50 |
| qtarget s0 | 65.46 | 86.09 |
| qtarget s1 | 7.62 | 45.74 |
| **method s2** | **22.97** | **33.24** |
| **cover s1** | **8.28** | **13.86** |

Every low-return run is also a low-diversity run. This is the same variable that
`stageA_selection_2x2.md` used to reject the centralized cell, and it is already
logged by every run, so it is available as a live health check rather than a
post-hoc one. Note the direction is the opposite of `v7_cadence_probe.md`, where
diversity rose while return fell — that was a comparison *across cadences*, this
is *within* a cadence, so the two are not in conflict, but neither is the
relationship causal on this evidence.

## 4. What this voids, and what survives

**Voided:**

* Every "this gap is Nx the seed sd of ±2.16" statement in
  `v7_search_module.md` §"What it shows". The gaps are ~0.2–0.5x the measured sd.
* That doc's claim that the search arms "separate cleanly". At n=2 the ordering
  is not stable: `mctsfix` was the worst arm at seed 0 (82.27) and beats `cover`,
  `qtarget` and `margvisit` at seed 1 (92.03 vs 13.86 / 45.74 / 92.91).
* The 0.06 nondeterminism floor, as above.
* `v7_generalization.md` §2's NashConv table, already flagged: it is computed on
  seed-0 checkpoints, and seed 0 is the high end of a very wide distribution.

**Survives, and is in fact strengthened:**

* `v7_module1_evidence.md`'s conclusion. Its §2 argued the environment is
  underpowered because VoI (2.256) sits at the seed noise floor (then believed to
  be 2.16). With the noise floor measured at 7–39 instead, the environment is
  **far more** underpowered than that section claimed, not less.
* §1's "the ablation costs nothing" — at n=3 the ablation's mean is *higher*
  (95.47 vs 78.55), though that gap is itself inside the spread and should not be
  read as the ablation winning.
* `v7_generalization.md` §1 (the return-gap confound). It is a within-checkpoint
  sign pattern replicated across five arms and two seeds, so seed variance in the
  level does not touch it.

## 5. What would settle it

Not more arms — more seeds on the arms already run, and a cause for the
collapses. The seed-2 runs are in flight (`v7_seed12_queue.json`). Anything that
distinguishes "this configuration is bimodal" from "these runs diverged for a
findable reason" is worth more than another ablation cell, since at present no
mazero comparison in this wave is resolvable.

## Related

`v7_search_module.md`, `v7_module1_evidence.md`, `v7_generalization.md`,
`late_training_instability.md` (within-run oscillation, sd ~8–12 — a different
and smaller effect than the between-seed spread measured here),
`stageA_selection_2x2.md` (head diversity as a rejection criterion).
