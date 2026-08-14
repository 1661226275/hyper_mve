# Run variance dominates every arm difference in the v7 wave

> **CORRECTION, 2026-08-13 — this document was first written as "seed variance"
> and that framing is wrong.** The seed-2 null control settles it: `margvisit` is
> a provably identical estimator to the method's `visit` target, and at seed 2
> the method collapses to 33.24 while `margvisit` reaches 85.11 — **51.88 apart
> at the same seed, from mathematically equivalent code**. The variance is
> therefore *run-level*, not seed-determined.
>
> **CONFIRMED DIRECTLY, 2026-08-14 — see §7.** The method was re-run at seed 2
> three more times under an identical command. Four draws give 33.24 / 33.94 /
> 54.83 / 95.21: **sd 29.05, range 61.97 at one seed**, which is 90% of the
> whole between-seed range. Two things in this document are wrong as a result
> and are corrected in §7: the spread is **not** bimodal, so "collapse
> probability" is the wrong frame; and §5's `value_loss` signature does **not**
> predict which draw ends low.

**2026-08-12, `results_v7_500k`, 500k env steps, robust last-20% statistic.**
This supersedes the variance assumptions in `v7_search_module.md` and
`v7_module1_evidence.md`. Both were written when only seed 0 existed and both
used a **±2.16** seed sd imported from the 3-seed method of record — a different
environment at a 1M budget. Measured here, that figure is wrong by more than an
order of magnitude for the `mazero_mixed` family.

## 1. Measured spread per arm

**Complete at n=3** (all 30 runs of `v7_seed12_queue.json` finished, 0 failed):

| arm | seed 0 | seed 1 | seed 2 | mean | sd | range |
|---|---|---|---|---|---|---|
| happo | 99.19 | 100.05 | 98.54 | 99.26 | **0.76** | 1.51 |
| m3w_adapted | 99.13 | 95.33 | 97.89 | 97.45 | 1.94 | 3.80 |
| mamba | 87.04 | 86.78 | 102.70 | 92.17 | 9.12 | 15.92 |
| mbom | 84.78 | 61.13 | 68.53 | 71.48 | 12.10 | 23.65 |
| — | | | | | | |
| method | 102.35 | 100.05 | **33.24** | 78.55 | **39.25** | **69.11** |
| no_subjective | 102.04 | 101.87 | 82.51 | 95.47 | 11.23 | 19.53 |
| margvisit (null control) | 102.41 | 92.91 | 85.11 | 93.48 | 8.66 | 17.30 |
| cover | 91.50 | **13.86** | 88.94 | 64.77 | **44.11** | **77.64** |
| qtarget | 86.09 | 45.74 | 77.65 | 69.83 | 21.28 | 40.35 |
| mctsfix | 82.27 | 92.03 | 92.24 | 88.85 | 5.70 | 9.97 |

These are not crashes — every run logged its full 501,619 env steps and exited
cleanly in the usual ~6.5–7 h.

**No between-arm difference in the mazero family exceeds the null control's own
spread (17.30, and 51.88 at seed 2).** The family cannot be ranked from this
wave.

Two cautions on reading the sds above, both learned the hard way in this wave:

* **They are not seed effects** — see the correction banner and §2. ~~They are
  collapse rates.~~ **Retracted by §7:** re-running the method at seed 2 gives a
  *graded* 33 / 34 / 55 / 95 rather than a collapsed/healthy split, so there is
  no rate to estimate. The honest reading is that these are 3-sample estimates
  of a wide continuous within-arm distribution. On the same evidence the
  bimodality I read into `cover` (two runs at ~90, one at 13.86) is an artifact
  of n=3, not a property of the arm; `qtarget`'s graded 86 / 78 / 46 is what
  every arm may look like given enough draws.
* **n=3 sds are themselves unstable.** mamba read sd **0.18** at n=2 and **9.12**
  at n=3, because its first two seeds happened to land 0.26 apart. Any claim of
  the form "arm X is stable" from two samples is worth very little; the honest
  distinction here is only between the model-free baselines (range 1.5–3.8 for
  happo/m3w) and the collapsing mazero arms (range up to 77.6).

## 2. The null control puts a number on the attribution floor

`..._margvisit` is a *provably identical* estimator to the method's `visit`
target — `test_marginal_visit_equals_visit_loss_exactly` asserts bitwise-equal
losses. So method-vs-margvisit **at the same seed** is a direct measurement of
run-to-run nondeterminism with the arm held constant:

| seed | method | margvisit | \|diff\| |
|---|---|---|---|
| 0 | 102.35 | 102.41 | **0.06** |
| 1 | 100.05 | 92.91 | **7.14** |
| 2 | **33.24** | **85.11** | **51.88** |

The mechanism is not mysterious: `scatter_add_` on CUDA is non-deterministic, so
two mathematically identical runs diverge numerically and training amplifies the
difference. `v7_search_module.md` reported the 0.06 as *the* nondeterminism
floor. That was a single sample of a quantity whose next two samples are 119x and
865x larger, and the claim should not have been made from n=1.

**Working figure: differences below ~52 points between `mazero_mixed` arms are
not attributable to the arm.** No gap anywhere in this wave's ablation tables
approaches that. The Module-1 gap is 0.31; the widest search-arm gap at a fixed
seed is ~20.

> **Superseded as the primary measurement by §7.** `margvisit` is an identical
> *estimator* but a different code path behind a different flag, so this table
> is evidence *about* run-level variance rather than a measurement of it. §7
> repeats the identical command and measures it directly: **61.97 points across
> 4 draws**, which replaces the ~52 working figure above. The conclusion is
> unchanged in direction and larger in size.

**The seed-2 row is the single most important measurement in this document.** It
is the same seed and mathematically the same estimator, so everything that could
differ between the two runs is nondeterminism — and one collapsed while the other
did not. Whatever causes the collapses, it is not the seed, and it is not the
arm. Two corollaries:

* Re-running a collapsed configuration is a **fresh draw, not a repeat**. It will
  probably not collapse again. That makes re-running-and-keeping a
  selection-bias trap rather than a fix: it redraws until the number is
  acceptable and silently deletes the collapse rate, which is the finding.
* Per-arm sds computed across seeds (§1) are not measuring a seed effect. They
  are a 2–3 sample estimate of how often that configuration collapses, which is
  both what makes them so large and why they are not comparable between arms at
  this sample size.

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

## 5. The collapse has a signature, and it is local

Diagnosed on method seed 2 (the curve is in the session log; `train/value_loss`
is on the env-step axis):

* The run trained **normally and even ahead** of seeds 0/1 to ~265k env steps
  (74.0 at 177k vs seed 0's 55.7).
* `train/value_loss` blew up over env steps 260.8k–265.6k: 85.0 → 57.4 → 96.2 →
  **120.4**. Across the whole run seed 2 has **8 excursions above 50 and one
  above 100**; seeds 0 and 1 never exceed **34.1** and **32.3** respectively.
* Return collapsed immediately after, at 273.6k: 60.4 → 48.0 → 37.6, and did not
  recover over the remaining 224k steps.
* Seed 2 ran hotter throughout — median `value_loss` 13.7 vs 8.0 / 8.5, and
  `head_diversity` ~50 vs seed 0's ~77 over the same window, so **diversity
  degradation preceded the collapse** rather than following it.

> **The signature does not generalize (§7).** It describes this run; it does not
> predict the outcome. Draw 1 has *more* excursions than draw 0 (12 above 50,
> three above 100) and finishes **21 points higher**; draw 2 has almost none
> (one above 50, none above 100) and finishes at the same 33 as draw 0. So the
> `value_loss` blowup is not the mechanism, or not the only one, and the leads
> below are weaker than this section originally implied.

Two configuration leads, neither yet tested:

* `lr` at the collapse is ~0.0092–0.0107, still near the top of the anneal.
  `ef76dcc` found the scaled model needed 0.005.
* `runner.py:226` sets `--max_grad_norm 10`, **loosening** the algorithm's own
  default of 5.0 (`core/config.py:114`). Gradients were clipped and the value
  loss still reached 120.

## 6. What would settle it

Not more arms. Three options, in increasing cost:

1. **More draws** on the arms already run. ~~to turn the collapse rate into an
   estimate rather than a 1-in-3 anecdote.~~ **Partly done — §7**, and it
   changed the question: there is no rate to estimate, because the outcome is
   continuous rather than collapsed/healthy. What §7 leaves open is whether
   there is *also* a seed effect on top of the run variance, which needs
   replicates at a second seed, not more seeds at n=1.
2. **Fix the cause and re-run every arm** under the fixed config — tighten
   `max_grad_norm`, lower `lr`. This is a configuration change, so partial
   re-runs are not comparable and the whole table has to move together.
3. **Report the collapse rate as the result.** It is a real property of the
   method at this budget, and the baselines do not share it (happo sd 0.76,
   mamba 0.18).

What is *not* an option is re-running individual collapsed runs and keeping the
better draw — see §2.

## 7. The direct measurement: four same-command draws at seed 2

**2026-08-14.** Everything above infers run-level variance from `margvisit`,
which is a mathematically identical *estimator* but a different code path behind
a different flag. **No experiment in this wave had ever repeated an identical
command.** The method was therefore re-run at seed 2 three more times, same
command, no config change.

Provenance, so this is checkable rather than asserted: each draw wrote to its
own `--out` root (`results_v7_s2rerun/r{1,2,3}`, queue file
`scripts/grids/v7_s2_rerun_queue.json`), because `train.py:205` mkdirs `run_dir`
with `exist_ok=True` and re-running into `results_v7_500k` would have merged a
second event file into the original's TB directory and destroyed the draw-0
curve. All three carry `config_hash b89b69343a229ca1c5653ba667651e76c46ded22`,
identical to the original `method_s2` — `--out` and `--gpus` are not in the hash
(`train.py:255-260`) — and all three logged the full 501,619 env steps /
31,251 train steps in 6.25–6.47 h.

| draw | robust | peak | vl med | vl max | #>50 | #>100 | head div | final eval (128 ep) |
|---|---|---|---|---|---|---|---|---|
| 0 (original) | 33.24 | 78.69 | 13.69 | 120.44 | 8 | 1 | 22.97 | 32.34 |
| 1 | 54.83 | 94.11 | 9.86 | 115.41 | 12 | 3 | 16.82 | 54.97 |
| 2 | 33.94 | 79.40 | 9.41 | 54.36 | 1 | 0 | 5.93 | 30.06 |
| 3 | **95.21** | 98.24 | 9.69 | 64.56 | 1 | 0 | 101.85 | 97.01 |
| *seed 0* | *102.35* | *104.48* | *8.01* | *34.08* | *0* | *0* | *129.39* | — |
| *seed 1* | *100.05* | *103.33* | *8.51* | *32.29* | *0* | *0* | *136.74* | — |

**mean 54.30, sd 29.05, range 61.97** — at one seed, from one command.

### What it establishes

* **Most of the "between-seed" spread is within-seed.** §1 records the method's
  sd across seeds 0/1/2 as 39.25 and its range as 69.11. The within-seed range
  at seed 2 alone is 61.97 — **90% of it**. The seed axis explains very little
  of what §1 attributed to it.
* **The distribution is graded, not bimodal.** 33 / 34 / 55 / 95 has no
  collapsed/healthy split. This retracts the "collapse probability" framing in
  the correction banner and §1: there is no rate to estimate.
* **The `value_loss` signature does not predict the outcome** — see the note in
  §5. Ranking the four draws by excursion count does not reproduce their return
  ordering, in either direction.
* **The attribution floor is 61.97 points, replacing §2's ~52.** Every gap in
  this wave's ablation tables remains far inside it.

### What it leaves open

**All four seed-2 draws fall below both seed-0 and seed-1** (best draw 95.21 vs
100.05 and 102.35). That is consistent with a seed effect sitting on top of the
run variance — but seeds 0 and 1 have **n=1 each**, so their own within-seed
distributions are unmeasured and the two cannot be separated. Settling it needs
replicates at a second seed, not more seeds at n=1. This is the one open
question in this document that the data cannot currently decide.

Finally, §3's head-diversity relationship does not survive at this resolution.
Across the four draws Pearson r = 0.934 but **Spearman rho = 0.400** — the
Pearson is carried entirely by draw 3's 101.85, and the rank ordering is not
reproduced. §3's rho = 0.940 was measured across 13 runs spanning arms and
seeds; within a single command at n=4 it does not hold, so head diversity is not
usable as the live per-run health check §3 suggested.

## Related

`v7_search_module.md`, `v7_module1_evidence.md`, `v7_generalization.md`,
`late_training_instability.md` (within-run oscillation, sd ~8–12 — a different
and smaller effect than the between-seed spread measured here),
`stageA_selection_2x2.md` (head diversity as a rejection criterion).
