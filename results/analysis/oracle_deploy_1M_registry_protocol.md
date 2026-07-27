# Oracle deploy mode at 1M under the registry protocol (2026-07-26)

First measurement of the oracle upper bound **in the canonical protocol** (previously oracle
existed only in `value_deploy_probe.py`, which scores the same checkpoint ~12 points higher and
is therefore not comparable to the headline table).

Checkpoint: `results/v5_final/mazero_mixed_ref_bc_anneal_scaled/relation/seed0`
(anneal_scaled, **centralized**, 1M env-steps, seed 0). Produced by
`scripts/reeval_checkpoint.py` → `eval_diagnostics_reeval.json`.

## Determinism check (prerequisite for backfilling older checkpoints)

| point | original | reeval | |
|---|---|---|---|
| A1 planner (bayes) | 56.5138 | 56.5138 | identical |
| A2 planner (argmax/MAP) | 55.8301 | 55.8301 | identical |
| A3 prior (no MCTS) | 19.9088 | 19.9088 | identical |

`evaluate()` is deterministic (env seeds `10_000+97g+ep`, search RNG `RandomState(12345)`), so
re-evaluating pre-oracle checkpoints backfills the oracle row without perturbing anything else.
`info_gating_strict=True` and `set_context_subjective_oracle_leak=False` are preserved; the
headline `return_mean` remains the bayes planner.

## Result

| mode | g0 | g1 | g2 | g3 | g4 | MEAN |
|---|---|---|---|---|---|---|
| A1 bayes (ours) | 79.5 | −0.6 | 53.2 | 62.5 | 87.9 | **56.51** |
| **UB oracle** (true g) | 74.8 | −0.7 | 50.6 | 65.2 | 87.9 | **55.59** |
| A2 argmax (MAP) | — | — | — | — | — | 55.83 |
| Δ oracle − bayes | −4.7 | −0.1 | −2.6 | **+2.7** | 0.0 | **−0.92** |

## Interpretation

**oracle ≈ bayes (Δ = −0.92, i.e. oracle is slightly WORSE).** By the locked decision rule this
localizes the limiter: **belief/regime-inference accuracy is NOT what holds the method back — the
value heads are.** Giving the planner perfect regime knowledge buys nothing. This replicates the
600K/hardval finding (oracle 74.24 vs bayes 74.21, Δ+0.03) in the canonical protocol and at the
standardized 1M budget, so it is not a protocol artifact.

**The sharper implication for how Module 2 should be described.** Hard-selecting a *single*
regime's value head is worse than averaging **even when the selected head is the true one**
(oracle 55.59 < bayes 56.51; argmax 55.83 < bayes 56.51). So the individual per-regime heads are
each less reliable than their posterior-weighted ensemble. Module 2's measured benefit here is
**variance reduction by ensembling**, not regime-conditioning — the averaging is doing the work,
and it does not depend on identifying the right regime. Any claim that the gain comes from
correctly inferring the relationship regime is unsupported by this checkpoint.

Per-regime structure is consistent across budgets: oracle helps the *exploitable* asymmetric
regime (g3 +2.7 here; g2 +5.2 / g3 +3.4 at 600K) and hurts cooperation (g0 −4.7 here, −8.4 at
600K), which is why the means cancel. Reporting only the mean hides this and should be avoided.

## CORRECTION (2026-07-27, after seed 2 of the method landed)

The claim above that oracle is **negative wherever measured** was over-read from single seeds.
On the method (`hardval_decoupled`) the oracle gap **flips sign across seeds**:

| seed | A1 bayes (endpoint) | A2 argmax | UB oracle | UB − A1 |
|---|---|---|---|---|
| 0 | 60.91 | 60.97 | 58.25 | **−2.65** |
| 2 | 64.56 | 68.69 | 66.80 | **+2.23** |

Against a per-run sd of ≈8 on these endpoint measures, a ±2.5 gap is **noise**. The correct
statement is therefore the weaker one:

> **oracle ≈ bayes** — perfect regime knowledge yields no reliable benefit (|gap| ≈ 2.5 vs
> sd ≈ 8), so regime-inference accuracy is not the binding limiter.

What is **not** supported: that supplying the true regime actively *hurts*. The direction is not
stable across seeds and must not be reported as a finding. Likewise "argmax ≈ bayes" is the safe
reading (A2 − A1 = +0.06 on seed 0, +4.13 on seed 2 — again sign-unstable and inside noise),
rather than "Bayes-averaging beats hard selection".

The ensembling-vs-regime-conditioning interpretation should be stated only as far as the
evidence goes: the belief posterior's *informativeness* is not what drives the return (oracle
buys nothing, `belief_blind` at zero head diversity still scores 62.94), but claiming the
averaging is *better* than selection overstates it.

## Consequences

- The gate note for the selected method must cite this: a shortfall vs mamba (68.00) is a
  value-head/planning-quality problem, so more belief/inference machinery would not close it.
- Whether `hardval` changes this picture is the open question — it is the one arm that makes the
  per-regime heads genuinely differentiated (`head_diversity` 3.80 vs ~0.5–1.3). If the oracle
  gap stays ≈0 even under hardval, the ensembling interpretation holds for the method of record.
  Both hardval cells (centralized + decoupled) will carry the oracle row natively.
