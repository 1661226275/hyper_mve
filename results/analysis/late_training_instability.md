# Late-training instability: the final-checkpoint metric is unreliable (2026-07-26)

## What triggered this

`ref_bc_anneal_scaled` + decoupled, 1M env-steps, **seed 0**, scored **67.49** in
`results_competence_1M_v2` and **36.71** in `results/v5_final` — same arm, same budget, same
seed. A 31-point swing looked like a bug.

It is not a bug. The periodic curves show **every** MixMAZero run oscillates violently in the
last third of training, and the registry's `return_mean` is a **single sample** of that
oscillation taken at whatever the final checkpoint happens to be.

`eval/return_mean` (periodic, every 200 train steps), late trajectory:

```
plain-CEN   : 600k:86 648k:72 696k:91 744k:69 792k:51 840k:71 888k:74 936k:61 984k:54
plain-DEC   : 600k:48 648k:71 696k:84 744k:56 792k:70 840k:58 888k:45 936k:29 984k:61
hardval-DEC : 600k:62 648k:69 696k:65 744k:64 792k:72 840k:82 888k:87 936k:72 984k:68
no_subj-CEN : 600k:23 648k:32 696k:20 744k:40 792k:33 840k:34 888k:21 936k:30 984k:22
```

plain-DEC reached **75.6 at 904k** and ended at 36.7. The endpoint is a lottery over a
±10-point-sd process, so two runs of one config differing by 30 points is expected, not anomalous.

**This was invisible before.** The pre-2026-07-24 code logged no periodic per-regime reward curve
for the fork (only a pooled, agent-averaged `test/mean_score`), so every earlier conclusion drawn
from a single final `return_mean` — including the "centralized costs ~11 points" reading — rests
on one draw from this distribution. The 200-step per-regime reward probe is what exposed it.

## Robust comparison (last-20%-of-training mean of each run's own periodic curve)

| algo / variant | final (1 pt) | **last-20% mean** | sd | pts |
|---|---|---|---|---|
| **mazero hardval + decoupled (ours)** | 60.91 | **71.95** | **7.70** | 314 |
| mamba | 68.00 | 71.91 | 10.39 | 52 |
| mazero plain + centralized | 56.51 | 64.95 | 9.61 | 314 |
| mazero plain + decoupled | 36.71 | 58.36 | 12.03 | 314 |
| m3w_adapted | 38.24 | 33.38 | 6.41 | 1251 |
| mazero no_subjective (Module-1 removed) | 26.41 | 29.38 | 6.75 | 314 |
| mbom | 22.37 | 20.30 | 4.05 | 14 |
| happo | 11.13 | 12.08 | 3.90 | 52 |

Every algorithm is scored the same way, on its own periodic curve, over the same final 20% of
its 1M-env-step budget. Episode counts per point differ (mazero probe 2/regime, baseline probes
8/regime) but the last-20% window aggregates 400–630 episodes for both, so the *means* are
comparably estimated.

## Consequences

1. **hardval + decoupled ties mamba** (71.95 vs 71.91) and is the more stable of the two
   (sd 7.70 vs 10.39) — where the endpoint metric had said the method trailed by 11 points.
2. **`head_diversity` back-half: hardval-DEC = 9.54**, versus 2.07 (plain-DEC), 1.33 (plain-CEN)
   and 0.00 (belief_blind). hardval is the only configuration whose per-regime value heads are
   strongly differentiated — 4.6× the nearest alternative. It therefore wins on **both** locked
   criteria at once (mechanism-active AND highest return), which removes the trade-off flagged
   earlier.
3. **Reporting rule going forward:** the headline comparison must use the robust last-20%
   statistic (or an explicitly averaged set of late checkpoints), with the single-point registry
   `return_mean` reported alongside as the endpoint sample. Publishing a single final checkpoint
   number for a process with sd ≈ 8–12 would be misleading in either direction.
4. **Open question, not yet acted on:** whether the oscillation should be *fixed* rather than
   averaged over — candidate causes are the BC/reference guidance reaching its floor at ~747k
   env-steps (74.7% of training under the scaled anneal, which lines up with where the decay
   starts) and the late-training LR schedule. A fix would be a new experiment, not a reporting
   change, so it is flagged rather than assumed.

## Status

Multi-seed confirmation launched: hardval-DEC seeds 1 and 2, plus mamba seed 1 (needed for the
3-seed table regardless). hardval-CEN seed 0 is still running and completes the selection 2×2.
