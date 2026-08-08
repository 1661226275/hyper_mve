# Comparison at 3 seeds — final (1M env-steps, capacity-controlled)

Generated 2026-07-29 from `results/v5_final/registry.jsonl`. Same protocol
throughout: `eval_report.json` at the final checkpoint, 16 episodes per regime,
5 regimes, `return_mean` = mean over regimes. All logging on the shared
env-steps axis.

Capacity is controlled — every baseline sits at **0.45–1.0×** the method's
1,256,453 network parameters (see `parameter_matching.md`).

## Headline

| algo | net params | s0 | s1 | s2 | mean ± sd |
|---|---|---|---|---|---|
| **mamba_pm** | 1.24M (0.98×) | 69.12 | 80.02 | *running* | **74.57 ± 7.70** (n=2) |
| **MixMAZero** (`hardval_decoupled`) | 1.26M (1.00×) | 60.91 | 60.34 | 64.56 | **61.94 ± 2.29** |
| m3w_adapted | 0.56M (0.45×) | 38.24 | 36.99 | 41.13 | 38.79 ± 2.13 |
| happo_pm | 0.88M (0.70×) | 15.72 | 10.26 | 10.46 | 12.15 ± 3.10 |
| happo | 0.07M (0.06×) | 11.13 | 7.85 | 16.37 | 11.78 ± 4.30 |
| mbom_pm | 0.77M (0.61×) | 12.32 | 10.04 | 8.23 | 10.19 ± 2.05 |

**MixMAZero does not lead. `mamba_pm` is ahead by ~12.6 points at matched
capacity** (n=2 pending seed 2). This is reported as measured; the plan
committed in advance to reporting it honestly rather than re-tuning, because the
version was selected on *mechanism validity* (`head_diversity` ≫ 0) rather than
on raw return.

## Per-regime breakdown

| algo | g0 | g1 | g2 | g3 | g4 | mean | spread |
|---|---|---|---|---|---|---|---|
| mamba_pm | 120.28 | −0.20 | 64.58 | 64.47 | 123.73 | 74.57 | 123.93 |
| MixMAZero | 90.90 | −0.48 | 61.44 | 55.49 | 102.33 | 61.94 | 102.82 |
| m3w_adapted | 73.93 | −0.12 | 23.40 | 36.49 | 60.24 | 38.79 | 74.05 |
| happo_pm | 24.81 | −0.11 | 7.60 | 9.45 | 18.98 | 12.15 | 24.92 |
| mbom_pm | 19.19 | −0.57 | 6.83 | 9.74 | 15.77 | 10.19 | 19.76 |

Two things to be straight about:

1. **`mamba_pm` beats MixMAZero on every regime individually**, not just on the
   mean (g0 120 vs 91, g2 64.6 vs 61.4, g3 64.5 vs 55.5, g4 124 vs 102). There
   is no regime where the method wins.
2. **Regime g1 is ≈ 0 for every algorithm**, method and baselines alike. It is
   not a discriminating regime and should not be presented as one — whatever g1
   asks for, nothing in the comparison achieves it.

The method's spread across regimes (102.8) is lower than `mamba_pm`'s (123.9),
but it is also uniformly lower in level, so "more uniform across regimes" is not
a defensible advantage here — a weaker method is trivially more uniform.

## Why the method trails — the oracle localises it

From `ablation_3seed_final.md`: **UB − A1 = +0.67 ± 2.88**. Supplying the *true*
regime id at deploy time buys nothing. So the shortfall is **not** regime
inference and **not** belief accuracy — more or better inference work would not
close the gap. The limiter is the **quality of the per-regime value heads
themselves**. That is where any future effort belongs.

## The high-parameter result was the learning rate, not capacity

This resolves the open attribution question. Three arms, one variable at a time:

| arm | model_scale | lr | net params | mean ± sd |
|---|---|---|---|---|
| `hardval_decoupled` (method of record) | 1.0 | 0.02 | 1.26M | **61.94 ± 2.29** |
| `…_lrctl` | 1.0 | **0.005** | 1.26M | **25.49 ± 7.05** |
| `…_big` | **3.0** | 0.005 | 5.25M | **34.83 ± 1.55** |

Decomposing the big model's 34.83 against the method's 61.94 (−27.1):

- **lr 0.02 → 0.005 at fixed capacity: −36.45** (61.94 → 25.49)
- **capacity 1× → 3× at fixed lr 0.005: +9.34** (25.49 → 34.83)

**The learning-rate reduction fully explains — over-explains — the deficit, and
the added capacity actually *helps*.** My earlier reading, that the
high-parameter variant showed capacity to be harmful, was wrong.

Two consequences:

1. **The capacity-scaled row is confounded and cannot be reported as a capacity
   result.** The wide model was forced to lr 0.005 only because every
   `model_scale > 1.0` crashes with SIGFPE at the inherited lr 0.02. A fair
   high-capacity comparison needs a stable route to a higher lr (warmup,
   gradient clipping, or an intermediate lr ≈ 0.01) — untested so far.
2. **The method is severely learning-rate sensitive**: a 4× lr reduction costs
   36 points, far more than 3× capacity gains. That is worth stating as a
   robustness limitation in its own right.

Note also that `…_big`'s 34.83 ± 1.55 is *more stable across seeds* than the
method of record's own lr-matched control (`lrctl`, ± 7.05), so the wide model is
not unstable at lr 0.005 — it is simply under-trained by that lr.

## Capacity matching changed nothing for the weak baselines

- happo 11.78 ± 4.30 → happo_pm 12.15 ± 3.10 (**+0.37** on a ±3–4 spread)
- mbom upstream (0.02× capacity) scored 22.37 at seed 0 historically;
  `mbom_pm` at 0.61× scores 10.19 ± 2.05

So HAPPO's and MBOM's distance from the method is **not** a capacity artifact,
which forecloses the "you under-parameterised the baselines" objection. It also
means the `_pm` variants are the honest rows to report for those two.

## Status

`mamba_pm` seed 2 still training (ETA Jul 30 ~00:00). Its arrival will move
`mamba_pm` from n=2 to n=3; given s0=69.12 and s1=80.02, the mean will move but
the ordering versus MixMAZero (61.94) is very unlikely to change.
