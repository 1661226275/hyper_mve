# Wave 2 — the action-axis target `agent_q_softmax` (2026-08-09)

8 runs, 2 arms × seeds 0–3, `relation_recip`, 1M env steps, lr 0.02, 128-episode
evals. All 8 completed; no crashes; all reached 314 periodic-eval points.
Launched 2026-08-08, landed 2026-08-09.

**Verdict: `agent_q_softmax` is NOT adopted. `q_softmax` remains the target of
record.** It does not improve return by a detectable margin, and it makes the
zero-sum regime substantially *worse* on the metric that regime is scored on.

## What was pre-registered

`scripts/grids/v6_agent_target_2x2.yaml`, before launch:

> * If `agentq_cover` collapses the star seed spread toward the none cells, the
>   multiplicity account is supported.
> * If it does not, the account is FALSIFIED and the visits-per-child budget
>   explanation survives instead.
> Do not reinterpret after the fact.

And in the plan's Risk 1:

> The visit-weighted mean is the wrong operator in g1. g1 is zero-sum; an
> expectation against the opponent's empirical visits is not a best response, a
> `min` is — and g1 is exactly where `q_softmax` currently wins by 2.4×.

**Risk 1 is what actually fired.**

## Return — last-20% of each run's own periodic eval

| cell | per-seed | mean | sd | sem |
|---|---|---|---|---|
| none/visit (baseline) | 26.77, 29.10 | 27.94 | 1.65 | 1.17 |
| star/visit | 37.55, 13.95 | 25.75 | 16.69 | 11.80 |
| none/q_softmax | 28.95, 41.28 | 35.12 | 8.72 | 6.17 |
| star/q_softmax | 26.80, 42.50 | 34.65 | 11.10 | 7.85 |
| none/agent_q | 56.63, 31.72, 35.09, 38.73 | 40.54 | 11.10 | 5.55 |
| star/agent_q | 25.24, 72.86, 38.33, 26.30 | 40.68 | 22.26 | 11.13 |

`agent_q` is nominally highest in both covers, and every scored regime moves the
same way (g0 69.6/69.7 vs 63.2/62.1; g4 60.6/68.0 vs 58.8/53.6). **This is not a
result.** The contrasts:

| contrast | delta | se | |
|---|---|---|---|
| none, return | **+5.43** | 8.30 | +0.65 se — not significant |
| star, return | **+6.03** | 13.62 | +0.44 se — not significant |

Both sit inside the pre-registered "~4-point gaps are potentially artefactual"
band, and both are dwarfed by their own seed spread. The wave was powered for
the spread question, not for a 5-point return difference; detecting one at this
seed variance would need roughly n≈30 per cell.

## g1 NashConv — the pre-registered instrument

Planner frozen, `br_env_steps=20000`, 16 eps. Lower = closer to equilibrium.
**All 12 values were measured in one session**, including a re-measure of the
wave-1 `qtarget`/`mctsfix` checkpoints.

| cell | per-seed NashConv | mean | sd | range(s0,s1) |
|---|---|---|---|---|
| none/q_softmax | 4.90, 5.96 | **5.43** | 0.75 | 1.06 |
| star/q_softmax | 3.72, 22.71 | 13.21 | 13.43 | 18.98 |
| none/agent_q | 19.19, 13.52, 28.76, 30.68 | **23.04** | 8.09 | 5.67 |
| star/agent_q | 12.44, 24.13, 26.49, 6.92 | 17.49 | 9.35 | 11.69 |

| contrast | delta | se | |
|---|---|---|---|
| none, NashConv | **+17.61** | 4.08 | **+4.31 se — significant, and worse** |
| star, NashConv | +4.28 | 10.58 | +0.40 se — not significant |

**The only statistically supported effect in the entire wave is that
`agent_q_softmax` is 4.2× more exploitable than `q_softmax` in the zero-sum
regime.** This is the risk that was written down in advance: averaging the
opponent's advantages under their empirical visit distribution is an expectation
against a fixed opponent, and in a zero-sum game the quantity that matters is a
`min`. `q_softmax`'s sum-of-exponentials is the softer, more `max`-like
aggregation, and that turns out to be worth more in g1 than the multiplicity
bias costs.

### The instrument has no noise floor

Re-measuring wave-1's four checkpoints reproduced the reported values **exactly**
— 4.90 → 4.90, 5.96 → 5.96, 3.72 → 3.72, 22.71 → 22.71, all deltas ±0.00. The
BR is deterministic given `(checkpoint, --seed)`. So the cross-session
reproducibility floor documented for *training* does not apply to NashConv:
differences on this metric are real, and the wave-1 numbers can be quoted
alongside wave-2's without a session caveat. The `q_softmax` rows remain n=2,
which is a power limit, not a comparability one.

## The pre-registered mechanism test: partially supported, and it does not matter

The star-specific *excess* spread is gone:

| target | spread, none | spread, star | star − none |
|---|---|---|---|
| q_softmax (range, s0–s1) | 1.06 | 18.98 | **+17.93** |
| agent_q (range, s0–s1) | 5.67 | 11.69 | +6.02 |
| agent_q (sd, n=4) | 7.01 | 8.10 | **+1.09** |

Under `q_softmax`, adding `star` multiplies the seed spread ~18×. Under
`agent_q`, adding `star` changes it by about one point. That interaction is the
multiplicity signature, and removing the multiplicity term removed it — as
predicted.

**But the convergence happens in the wrong direction.** The registered wording
was "collapses the star seed spread *toward the none cells*." Star came down
(18.98 → 11.69) while none went *up* (1.06 → 5.67); the two cells met in the
middle rather than at the low baseline. So the multiplicity account survives as
an explanation of *why star and q_softmax interact*, and it is now the standing
explanation for that interaction — but `agent_q_softmax` is not a repair,
because it degrades the cell that was already healthy.

Range is n-dependent, so the ranges above are restricted to seeds 0–1 wherever
they are compared against wave-1's n=2 rows; the n=4 sd is reported separately
and is the honest summary of the new arms.

## Two things worth keeping

**g1 damage is invisible in return.** `agentq` seeds 2 and 3 have g1
`welfare_physical` of 30.94 and 27.19 against 86.56 and 92.41 for seeds 0 and 1
— a collapse in half the seeds — while their last-20% returns (35.09, 38.73) sit
mid-pack and unremarkable. This is the sharpest evidence yet for the protocol's
core rule that return is structurally blind in g1, and it was found only because
the regime was scored on its own instrument.

**`star` protected g1 welfare here.** All four `agentq_cover` seeds hold welfare
73–96; two of four `agentq` seeds collapse. Unexplained, n=4, noted rather than
claimed.

## Confound to state plainly

The agent cells ran at **τ = 0.85**, the `q_softmax` cells at **τ = 1.0**. That
was deliberate — the probe measured a span ratio of 0.891/0.800, and inheriting
τ=1.0 would have compared per-agent marginalization against a flatter target —
but it does mean `agent_q` vs `q_softmax` is not a pure single-factor contrast.
A τ sweep on the agent cells would be needed to attribute the g1 loss to the
aggregation operator rather than to sharpness. Given that the effect is large
(+17.61, 4.3 se) and the τ gap is small and span-corrected, the operator is the
more plausible cause, but this is not settled by this wave.

## What this retires, and what is next

Retired: `agent_q_softmax` and `agent_q_blend` as candidate defaults. They stay
in the codebase (tested, arm-locked, 9 unit tests + 4 wiring tests) because the
helper is the machinery any action-axis target needs, but no arm should adopt
them without a g1 story.

The natural follow-up, already named in the plan as a one-line change now that
`agent_marginal_target` exists: **`min`-aggregation** —
`Q̲_i(a) = min_{c: a_i^c=a} adv_i(c)` in place of the visit-weighted mean. That
is the operator the zero-sum argument actually calls for, and it removes the
multiplicity term just as exactly as the mean does. It is the only variant this
wave's evidence positively motivates.

Still queued and untouched: the frozen-root/QMDP belief question
(`hyper_mve_bayes_averaging_negative_result`, VoI 3.99 on v6), which remains the
other live candidate for the binding constraint.

## Reproduce

```bash
PY=/home/zhengwenbo/.conda/envs/lightzero/bin/python
# returns: last-20% of eval/return_mean from each run's tb/
# g1 NashConv, one checkpoint (>15 min each; do NOT wrap in a short timeout):
CUDA_VISIBLE_DEVICES=0 $PY scripts/eval_game_metrics.py \
  --ckpt results_v6_agentq/mazero_mixed_ref_bc_anneal_scaled_hardval_decoupled_agentq/relation_recip/seed0/ckpt.pt \
  --external mazero_mixed --preset rel_recip --frozen-mode planner \
  --regimes 1 --br-steps 20000 --episodes 16 --out gm_agentq_s0.json
```
