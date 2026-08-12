# Search-module ablations on `rel_coopmix`: return and NashConv agree

**2026-08-12, seed 0, 500k env steps, `results_v7_500k`.** Unlike the Module-1
result (`v7_module1_evidence.md`, where six measurements all land inside the
noise), the search-module arms separate cleanly and the two axes rank them the
same way.

## The table

Return is the robust last-20% of each run's own periodic `eval/return_mean`
(the endpoint is not a usable basis — `late_training_instability.md`). NashConv
is at `br_env_steps = 100000`, 16 episodes/regime, `--frozen-mode planner`;
prior mode is not reportable for these arms at all (it reproduces 28% of the
deployed policy's return). Lower NashConv is better.

| arm | return | sd | NashConv | G(0,3,4) | welfare |
|---|---|---|---|---|---|
| `..._margvisit` (null control) | 102.41 | 0.99 | — | — | — |
| **method `..._hardval_decoupled`** | **102.35** | 1.52 | **15.89** | **14.36** | **142.85** |
| `..._cover` (star root-cover) | 91.50 | 2.41 | 22.08 | 19.08 | 132.68 |
| `..._qtarget` (q_softmax target) | 86.09 | 1.79 | 28.56 | 30.77 | 121.29 |
| `..._mctsfix` | 82.27 | 3.50 | — | — | — |

Per-regime NashConv:

| arm | g0 | g1 | g2 | g3 | g4 |
|---|---|---|---|---|---|
| method | 5.14 | 19.46 | 16.90 | 27.67 | 10.28 |
| cover | 7.96 | 27.24 | 25.91 | 29.39 | 19.89 |
| qtarget | 16.60 | 27.75 | 22.73 | 46.75 | 28.95 |

## What it shows

**The three axes agree, monotonically.** Return, exploitability, and welfare
order the three measured arms identically: method > cover > qtarget. The return
gaps (10.9 and 16.3) are ~5x and ~7.5x the method's seed sd of ±2.16, so unlike
the Module-1 ablation these separate well clear of seed noise at n=1.

This matters mainly as a *contrast*. The same measurement protocol, on the same
environment and budget, does separate arms when there is something to separate —
so Module-1's null result is not an artefact of a metric too blunt to see
anything. That is the strongest thing this table does for the theory line.

**The method's advantage here is on the search side, not the belief side.** Both
measured arms change how the root/target is constructed, and both cost 6–16
points of return and 6–13 points of NashConv. Combined with
`v7_module1_evidence.md` §6 — where removing the subjective module *lowers*
NashConv — what is currently supported by measurement is that the decoupled
search construction carries the method, and the role-aware module does not.

**`margvisit` is doing its job as a null control.** It is a provably identical
estimator to `visit` (`test_marginal_visit_equals_visit_loss_exactly`), and both
rows are seed 0 — so its 102.41 vs the method's 102.35 is **not** a seed-noise
estimate. It reads the *fixed-seed nondeterminism floor*: **0.06**, the amount
two mathematically identical runs drift apart through RNG-consumption order and
non-deterministic GPU reductions alone. That is a floor, not the seed sd (±2.16
from the 3-seed method of record), and only the seed sd is the right yardstick
for comparing arms. Its value is as a sanity check that the pipeline reproduces
itself, and it passes.

## Gaps

* **No NashConv for `mctsfix` or `margvisit`.** Not queued; each costs ~24 h of
  the wave's most expensive measurement. `mctsfix` is the worst arm on return
  (82.27) so its NashConv would very likely follow the trend and add little;
  `margvisit`'s would be a second read of the noise floor on the NashConv axis,
  which is the more informative of the two if a slot ever frees.
* **n=1 seed on every row.** Seed-1 replications of `cover`, `qtarget`,
  `mctsfix` and `margvisit` are queued (`v7_seed12_queue.json`) but the NashConv
  evals for them are not.
* The `..._cover` arm's mechanism (star multiplicity) was confirmed separately —
  see `hyper-mve-action-axis-target`.

## Related

`v7_module1_evidence.md` (the contrasting null result, and §6's NashConv table
including the mamba/happo comparison points), `v7_cadence_probe.md` (why the
cadence is 16), `late_training_instability.md` (why the statistic is robust
last-20% and not the endpoint).
