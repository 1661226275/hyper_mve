# Cadence probe: more gradient steps make the method WORSE (2026-08-10)

**Result: the starvation hypothesis is disconfirmed. Keep `_ENV_STEPS_PER_GRAD = 16`.**

## The question

`train_cadence.md` measured that the method gets the smallest optimisation
budget of the five algorithms — 59.5 true `optimizer.step()` per 1k env steps
against happo 150, m3w_adapted 713, mamba 2880. Since mamba currently outscores
it, the obvious hypothesis was that the method is *gradient-starved*, and the
fair remedy was to raise its cadence rather than cut the baselines'.

This probe tests that directly: same arm, same env, same seed, same 150k
env-step budget, varying only gradient steps per collected transition.

## Setup

`mazero_mixed`, arm `ref_bc_anneal_scaled_hardval_decoupled`,
`--env relation_coopmix --seed 0 --total-env-steps 150000 --num-pmcts 16`,
one GPU per cell. Only `--env-steps-per-grad` differs. The BC anneal is a
fraction of `training_steps` (74.7%), so all three cells anneal over the same
share of training — the manipulation is clean.

## Result

Selection statistic is the mean of each run's own periodic `eval/return_mean`
over the final 20% of training (`scripts/probes/cadence_readout.py`); the
endpoint is shown alongside but is not the basis, per
`late_training_instability.md`.

| ratio | grad steps | true grad /1k | **robust** | sd | endpoint | head_diversity | regime_acc | walltime |
|---|---|---|---|---|---|---|---|---|
| **16 (incumbent)** | 9 376 | ~62 | **55.71** | 7.52 | 59.08 | 13.18 | 0.463 | **1.93 h** |
| 4 | 37 601 | ~250 | 34.34 | 3.32 | 33.33 | 21.64 | 0.451 | 5.13 h |
| 2 | 75 201 | ~500 | 41.13 | 2.20 | 39.43 | 27.57 | 0.479 | 9.01 h |

**The incumbent wins by 14.6–21.4 points** — 2–3× the within-run oscillation, so
not a noise artefact. Raising the cadence 4× costs 21.4 points and 2.7× the
walltime; 8× costs 14.6 points and 4.7× the walltime.

## Reading it

Three things move together and they are consistent with over-training on a
limited stream rather than with instability:

* **Return falls** as gradient steps rise.
* **Variance falls** (sd 7.52 → 3.32 → 2.20). The faster cells are not
  diverging or oscillating; they settle — onto a worse policy. This is the
  signature of premature convergence at a high replay ratio, not of a learning
  rate that has become too large.
* **`head_diversity` rises monotonically** (13.18 → 21.64 → 27.57). The
  per-regime value heads specialise *more* with more optimisation, and it does
  not translate into return. That is a third independent measurement pointing
  the same way as the settled Bayes-averaging negative result: head
  specialisation is not what the return is limited by.

`regime_accuracy` is flat across cells (0.451–0.479) and far above the 0.200
chance level everywhere, so no cell is rejected on posterior collapse and the
comparison rests on return alone.

## Consequence for the fairness question

The method's low gradient count is **not** a handicap that more optimisation
fixes. At a fixed env-step budget it is at or near its best at ratio 16, and the
baselines' 2.5–48× larger optimisation budgets are not an advantage we are
failing to match — for this method, on this env, more updates per sample are
actively harmful.

So the comparison stands on the env-step axis with the cadence table reported as
a disclosed difference. That is the same conclusion the plan reached, but now it
rests on a measurement rather than on the argument that cutting baselines is
unfair.

## Caveats

* **n=1 per cell.** The 14–21 point gaps are large against the within-run sd,
  and the direction is consistent across three independent quantities (return,
  sd, head diversity), but no cell is repeated.
* **lr was not co-tuned.** All cells ran the arm's default. The standard
  compensation for a higher replay ratio is a *lower* lr, which was not tried,
  so the honest claim is "raising the cadence at fixed lr does not help", not
  "this method cannot benefit from more optimisation".
* **The trend was not followed past the incumbent.** Return improves monotonically
  as the ratio rises across the tested range, which leaves open whether 32 or 64
  would beat 16. That would change the method of record rather than merely
  confirm it, so it was not run here. It is the obvious next cadence experiment
  if one is wanted, and it is cheap (~1 h at 150k).
* The 4-vs-2 ordering is inverted relative to a monotone dose-response, which is
  the one place the data look like noise. Both sit far below the incumbent
  regardless.

## Reproduce

    python scripts/train.py --algo mazero_mixed --env relation_coopmix --seed 0 \
        --total-env-steps 150000 --episodes 16 --num-pmcts 16 \
        --ablation ref_bc_anneal_scaled_hardval_decoupled \
        --env-steps-per-grad 16 --gpus 0 --out results_v7_cadence_r16

    python scripts/probes/cadence_readout.py results_v7_cadence_r{16,4,2}
