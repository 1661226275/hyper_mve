# Stage A — version selection screen, 2×2 complete (2026-07-26)

All four cells at 1M env-steps, seed 0, `--env relation`, under the new logging regime.
`last-20%` = mean of the run's own periodic `eval/return_mean` over the final 20% of training
(the robust statistic; see `late_training_instability.md` — the single-point final metric has
sd ≈ 8–12 and is not a usable basis for selection).

| cell | final (1 pt) | **last-20%** | sd | `head_diversity` | `regime_acc` | oracle − bayes |
|---|---|---|---|---|---|---|
| plain, centralized | 56.51 | 64.95 | 9.61 | 1.33 | 0.513 | −0.92 |
| plain, decoupled | 36.71 | 58.36 | 12.03 | 2.07 | 0.200 | — |
| **hardval, centralized** | 62.99 | **73.41** | 9.10 | **11.58** | **0.200** | −2.61 |
| **hardval, decoupled** | 60.91 | 71.95 | **7.70** | 9.54 | **0.561** | — |

Reference: **mamba last-20% = 71.91** (sd 10.39). Null control `belief_blind`: `head_diversity`
0.000, scored 62.94.

## Findings

1. **hardval is confirmed as the decisive optimization.** Both hardval cells beat both plain
   cells on the robust return (73.41 / 71.95 vs 64.95 / 58.36) *and* on `head_diversity`
   (11.58 / 9.54 vs 1.33 / 2.07, i.e. 5–9×). It is the only setting in which the per-regime value
   heads are strongly differentiated, and it costs nothing in return — it gains.
2. **The earlier "centralized costs ~11 points" claim is RETRACTED.** That came from comparing
   single final checkpoints (56.51 vs 67.49). On the robust statistic centralized is **equal or
   better** in both rows: 73.41 vs 71.95 (hardval) and 64.95 vs 58.36 (plain). The original
   "centralized to reduce complexity" design lock is not costing performance.
3. **Both hardval cells match the strongest baseline** (73.41 / 71.95 vs mamba 71.91), within noise.

## Unresolved tension for the selection (decision pending multi-seed data)

The two hardval cells differ in a way the locked rule did not anticipate:

- **hardval-CEN** has the highest return (73.41) and the highest head diversity (11.58), but
  `regime_acc = 0.200` — **exactly chance for 5 regimes**, the signature of a collapsed belief net
  emitting a near-constant posterior. Its Bayes average is then a *fixed* mixture of diverse
  heads: an ensemble, not a regime-conditioned selector.
- **hardval-DEC** returns 71.95 (−1.46, far inside sd 7.7–9.1, i.e. indistinguishable) but has
  `regime_acc = 0.561`, genuine above-chance regime inference, with heads still strongly
  differentiated (9.54).

For a claim about **role-aware, regime-adaptive** planning, hardval-DEC is far more defensible:
a headline configuration that identifies the relationship regime at chance would undercut the
narrative regardless of its return. Recommendation: **hardval-DEC**, unless the multi-seed data
shows hardval-CEN's belief collapse is a seed artifact and its return advantage is real.

Corroborating this reading, the oracle rows are **negative wherever measured** (−0.92 plain-CEN,
−2.61 hardval-CEN): supplying the true regime at deploy *hurts*. Together with `belief_blind`
scoring 62.94 at zero head diversity, the evidence says the measured benefit of the value-head
machinery is **ensembling / variance reduction**, not regime identification. That should be
stated plainly rather than framed as successful regime inference.

## Running (resolves the open decision)

hardval-DEC seed 1 (GPU4), seed 2 (GPU5); hardval-CEN seed 1 (GPU3) — tests whether both the
73.41 and the `regime_acc = 0.200` collapse reproduce; mamba seed 1 (GPU6), needed regardless.
