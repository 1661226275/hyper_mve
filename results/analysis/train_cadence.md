# How much training each algorithm actually gets (measured 2026-08-10)

Measured with `scripts/probes/grad_step_probe.py`: every concrete
`torch.optim.*` class is wrapped with a counter, so the number is whatever
actually ran — no vendored file is edited and no config is trusted. One process
per algorithm, `relation_coopmix`, seed 0, 10 000 env steps, method arm
`ref_bc_anneal_scaled_hardval_decoupled`.

## Why this document exists

`progress/train_steps` does **not** mean the same thing across algorithms, and
the v7 stage-1 cadence table (`v7_session2_handoff.md` §2.1) compared those
numbers as if it did. It increments once per *gradient step* for
`mazero_mixed`, but once per multi-epoch update *round* for the others. The
error is not small — for mamba the round expands 288×.

The §4.5 plan to normalise the **logged** number to ~20/1k env steps would
therefore not have equalised training. It would have cut the method a further
3.1× and `m3w_adapted` 12.5× while leaving true optimisation effort orders of
magnitude apart. That plan is superseded by this measurement.

## Measured

True `optimizer.step()` calls per 1 000 env steps:

| algo | logged /1k | **true /1k** | expansion | walltime /10k env steps |
|---|---|---|---|---|
| mbom | 2.5 | **9 087.2** | 3 635× | 457 s |
| mamba | 10.0 | **2 880.0** | 288× | 1 382 s |
| m3w_adapted | 250.0 | **712.8** | 2.9× | 340 s |
| happo | 10.0 | **150.0** | 15× | 57 s |
| **mazero_mixed (ours)** | 53.6 | **59.5** | 1.1× | 661 s (11.7k steps) |

**The method has the smallest optimisation budget of the five** — 2.5× below
happo, 12× below m3w_adapted, 48× below mamba. It is not being flattered by
extra training; it is being compared against algorithms that all train harder
per collected sample.

## Two readings that must not be conflated

**mbom's 9 087 is not 9 087 policy updates.** Instance-level attribution
(`distinct_optimizers: 5103`) shows the count is dominated by short-lived
optimizers over a ~3.7k-parameter network — MBOM fine-tunes an *imagined
opponent model* inside `choose_action`, so the adaptation is part of its acting
path (`vendor/MBOM/policy/MBOM.py:88-99`, `imagine_model_learning_times=5`).
Its actual policy training is the four long-lived optimizers at exactly 250
steps each (2 agents × actor+critic, `a_update_times=10 + v_update_times=10`
over 25 epochs) = **1 000 steps = 100 /1k env steps**. The other ~8 987 /1k are
inner-loop adaptation on throwaway copies.

This is the same mechanism that makes mbom NashConv cost ~11.5 h per checkpoint
(§2.4): its acting path cannot run under `no_grad`.

**mamba's 2 880 are mostly on imagined states**, not real transitions —
`train_agent` loops `PPO_EPOCHS=5` over minibatches of the imagined rollout,
updating actor and critic per minibatch. Comparing imagined-data updates to
real-data updates one-for-one overstates the gap.

So the ranking by *policy training on real data* is much tighter than the raw
column suggests — roughly m3w_adapted 713, happo 150, mbom 100, ours 59.5, with
mamba's world-model updates (~10/1k, one per round) separate from its imagined
actor/critic work. Ours is still last.

## What was decided

Raise the method's cadence; leave every baseline at its published one. Cutting
baselines would detune them (cf. the MAMBA optimal-config lesson) and reads as
handicapping. `hyper_mve/algo/runner.py:53` `_ENV_STEPS_PER_GRAD` is now a
default overridable per run via `scripts/train.py --env-steps-per-grad`, and is
recorded in each run's `meta.json` because two runs at the same env-step budget
are not comparable across different values of it.

Selected by the `{16, 4, 2}` probe in `v7_cadence_probe.md`. Env steps remain
the held-constant axis for the headline comparison; this table is the disclosed
asymmetry that goes with it.

## Reproduce

    python scripts/probes/grad_step_probe.py --algo mamba --gpus 1 \
        --env-steps 10000 --out results/analysis/cadence/mamba.json

Raw JSON per algorithm in `results/analysis/cadence/`.

## Caveat

n=1 per algorithm, 10k env steps, seed 0. Rates that depend on buffer warmup
(`m3w_adapted` warms up ≤500 steps; mamba gates on `MIN_BUFFER_SIZE`) are
slightly understated at this budget relative to a full 500k run. The ordering is
far too large to be affected.
