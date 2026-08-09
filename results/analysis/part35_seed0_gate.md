# Part 3.5 — seed0 gate: PAUSED (investigation result)

> **Regime ids here are v5 `g2`**: g0 mutual_coop, g1 mutual_comp,
> g2 asym_exploit, g3 asym_exploited, g4 neutral. Under the current `g2cm`
> family the same ids name different regimes and `mutual_comp` does not
> exist — see `results/analysis/g1_removal.md`. Numbers below are
> unedited.


## Gate outcome

Rule: main mixmazero ≥ best baseline → proceed to seed1/2; below → pause & investigate.

| variant (seed0, 1M env-steps) | return_mean | fidelity reward_mae |
|---|---|---|
| mamba (baseline) | **68.00** | 0.0346 |
| **mixmazero main (centralized)** | **56.51** | 0.0368 |
| m3w_adapted (baseline) | 38.24 | 0.0236 |
| mbom (baseline) | 22.37 | — (no learned reward head) |
| happo (baseline) | *running* | — (model-free) |
| mixmazero no_subjective (Module-1 ablation) | *running* | — |
| mixmazero decoupled (deployment-cost cell) | *running* | — |

**mazero-main (56.51) < mamba (68.00) → GATE FAILED → seed1/2 NOT launched.**

## Root cause: centralized action broadcasting costs ~11 return points

The same arm (`ref_bc_anneal_scaled`, 1M env-steps, seed 0) run with the two
selection modes:

| | centralized (v5_final) | decoupled (1M_v2) | Δ |
|---|---|---|---|
| return_mean | 56.51 | **67.49** | **−10.97** |
| g0 coop | 79.47 | 93.43 | −13.96 |
| g1 comp | −0.58 | −0.71 | +0.13 |
| g2 exploit | 53.22 | 56.10 | −2.88 |
| g3 exploitable | 62.54 | 68.28 | −5.74 |
| g4 neutral | 87.92 | 120.34 | **−32.42** |
| regime_accuracy | 0.513 | 0.554 | −0.041 |

Budgets matched exactly (env_steps 1000016 vs 1000000; train_steps 62401 vs
62400; ~12.1h vs 12.0h each). The loss concentrates in the **high-return**
regimes (g4, g0) — where coordinating on the best joint action matters most —
and barely touches the asymmetric regimes (g2/g3).

**Confounds ruled out:**
- *Probe belief leakage:* `update_weights` re-sets the belief from each batch's
  own `belief_ctx_b` before every training forward (core/train.py:240), so the
  eval probes cannot leak state into training.
- *Cadence 500→200:* gradient steps are paced off `transitions_collected`, so
  total env steps and total gradient steps are invariant to eval frequency
  (confirmed by the matched step counts above). The extra eval costs wall time
  only.
- *Logging axis change:* pure logging, no learning effect.

So the only learning-relevant difference is the selection mode. With decoupled
selection the method scores **67.49 ≈ mamba's 68.00** (a statistical tie); the
user-locked centralized deployment is exactly what puts it behind.

## Deploy-mode ablation (already valid, centralized checkpoint)

| point | mode | return_mean |
|---|---|---|
| A1 | Bayes-avg (ours) | **56.51** |
| A2 | argmax (MAP head) | 55.83 |
| A3 | no MCTS (prior) | 19.91 |

A1 > A2 > A3 — consistent with the Part 0a oracle probe (bayes ≥ argmax, and
search contributes a large +36.6 over the distilled prior). Search is working;
root_child_count=3.28, coverage=0.384, visit_entropy=0.904 (not collapsed).

## Decision required

Centralized broadcasting was locked "to reduce complexity"; it measurably costs
~11 points and the headline "≥ baselines" claim. Options: keep centralized and
report honestly; make decoupled the main method and report centralized as a
measured complexity/performance trade-off; or carry both as a deployment
comparison. A like-for-like decoupled run under the new logging is in flight.
