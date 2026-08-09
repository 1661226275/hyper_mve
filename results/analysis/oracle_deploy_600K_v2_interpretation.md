# Oracle deploy test — 600K_v2 (`ref_bc_hardval`) — interpretation

> **Regime ids here are v5 `g2`**: g0 mutual_coop, g1 mutual_comp,
> g2 asym_exploit, g3 asym_exploited, g4 neutral. Under the current `g2cm`
> family the same ids name different regimes and `mutual_comp` does not
> exist — see `results/analysis/g1_removal.md`. Numbers below are
> unedited.


Raw table: `results/analysis/oracle_deploy_600K_v2.log`
Checkpoint: `results_competence_600K_v2/mazero_mixed_ref_bc_hardval/relation/seed0`, 16 eps/regime.

| mode | g0 coop | g1 comp | g2 explt | g3 explable | g4 neutrl | MEAN |
|---|---|---|---|---|---|---|
| bayes (ours) | 119.96 | -0.40 | 61.71 | 62.57 | 127.22 | **74.21** |
| oracle (true g) | 111.58 | -0.39 | 66.88 | 65.91 | 127.22 | **74.24** |
| argmax (MAP) | 101.51 | -0.39 | 72.05 | 62.88 | 127.22 | **72.65** |

## Findings
- **oracle ≈ bayes** (Δ = +0.03 overall). Perfect regime knowledge does **not** raise the
  headline return. Oracle helps the aliased asymmetric regimes **g2 (+5.16), g3 (+3.35)** but
  **regresses g0 (−8.38)** under hard-select; the soft belief posterior hedges g0 better.
- **argmax < bayes** (−1.56 overall; g0 −18.45). Hard MAP-selection is worse than belief
  averaging → **Bayes-average (ours) is the best deploy-time value aggregation.**

## Implications
- **Seed0 gate (3.5):** oracle≈bayes ⇒ if mixmazero underperforms baselines, the limiter is
  **NOT belief/inference accuracy** — look upstream at value-head quality / the g0 hard-select
  penalty / an overall ceiling, not at the belief net.
- **Ablation preview (A1 vs A2):** expect **A1 Bayes ≥ A2 argmax**; oracle≈Bayes means the
  belief-averaging already extracts ~all the available per-regime value signal.
- Note: probe MEAN (74.2, seed base 0, 16 eps) differs from the run's `eval_report` mean (62.56,
  different eval seeds) mainly via high-variance g3/g4 — the **within-probe** bayes/oracle/argmax
  comparison is the valid, seed-matched signal, not the absolute level.
