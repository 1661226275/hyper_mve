# Parameter matching across the comparison (2026-07-27)

## Why

Observation: mamba's memory footprint was several times the method's. Measuring the
**network** parameter counts (optimizer/normalizer state excluded, read from each run's
`ckpt.pt`) showed the comparison was not capacity-controlled **in either direction**:

| network | net params | vs method |
|---|---|---|
| mamba (upstream widths) | 8,405,821 | **6.7×** |
| method — hardval_decoupled | 1,256,453 | 1.00× |
| m3w_adapted | 560,792 | 0.45× |
| happo (upstream widths) | 73,413 | **0.06×** |
| mbom (upstream widths) | 24,244 | **0.02×** |

mamba's world model alone (6.75M) is 5× the method's entire network, while happo and mbom are
17× and 52× *smaller*. Any result could be attributed to capacity rather than algorithm.

## How it was fixed

Capacity is a **runner variant**, not a global switch — so no baseline is ever reported only in
a handicapped form, and no existing run directory is overwritten:

| registry key | widths | net params | vs method |
|---|---|---|---|
| `mamba` | upstream (HIDDEN/EMBED/DET 256, 32×32 latents) | 8,405,821 | 6.7× |
| **`mamba_pm`** | HIDDEN/EMBED/DET 128, 16×16 latents, heads 128 | **1,235,645** | **0.98×** |
| `happo` | `hidden_sizes [128,128]` | 73,413 | 0.06× |
| **`happo_pm`** | `hidden_sizes [512,512]` | **882,501** | **0.70×** |
| `mbom` | `[64,32]` | 24,244 | 0.02× |
| **`mbom_pm`** | `[512,256]` | **767,252** | **0.61×** |
| `m3w_adapted` | unchanged (already same order) | 560,792 | 0.45× |

Method side, via the new `--model_scale` flag (multiplies every fork width **and** the
hypernet/context widths in `ModelConfig`):

| arm | scale | net params | vs method |
|---|---|---|---|
| `…_hardval_decoupled` (method of record) | 1.0 | 1,256,453 | 1.00× |
| **`…_hardval_decoupled_big`** | 3.0 | **5,254,341** | **4.18×** |

`--model_scale 1.0` is a no-op by construction, verified to reproduce the historical
architecture exactly — so every existing run stays reproducible and comparable.

All `_pm` variants plus `m3w_adapted` now sit within **0.45–1.0×** of the method: the same order
of magnitude, as required. The high-parameter method variant (4.18×) covers roughly the capacity
mamba originally had, so the comparison is controlled in both directions:

- **capacity-matched row:** method (1.26M) vs `mamba_pm` / `happo_pm` / `mbom_pm` / m3w (0.45–1.0×)
- **capacity-scaled row:** method-big (5.25M) vs `mamba` upstream (8.41M)

## The high-parameter variant needs a lower learning rate (2026-07-27)

Any `--model_scale > 1.0` **crashes with SIGFPE** (exit 136, core dumped, *no Python traceback*)
inside the vendored C++ tree — faulthandler localised it to `mcts_sampled.py:131
trees.batch_selection`, called from selfplay. Evidence:

| config | params | lr | logged losses | outcome |
|---|---|---|---|---|
| scale 1.0 (method of record) | 1.26M | 0.02 | finite | **stable** (many 12 h runs) |
| scale 2.0 | 2.78M | 0.02 | **finite** (20.8, no NaN) | SIGFPE < 10 min |
| scale 3.0 | 5.25M | 0.02 | **NaN by step 1600** | SIGFPE 1–12 min |
| scale 2.0 | 2.78M | **0.005** | finite | **survived** the crash window |

The scale-2.0 row is the informative one: its losses stay finite and it still crashes, so **NaN is
not the trigger** — the wider net at the inherited lr=0.02 drives transient extremes into the
tree between log points, and an integer division by zero there kills the process. Lowering the
learning rate removes it. (A separate, real defect was found and fixed along the way: the guard
at `mcts_sampled.py:111` was `assert ~(np.sum(...) == 0).sum()`, and `~0` is truthy, so it never
fired for the all-zero row it was written to catch. That fix does **not** resolve this crash.)

**Consequence for the comparison.** The high-parameter row cannot be run at the method's lr, so
it carries a disclosed lr change — which would confound capacity with optimisation. To keep the
capacity claim clean the sweep therefore includes an **lr control at the method's own size**:

| row | params | lr | role |
|---|---|---|---|
| method of record | 1.26M | 0.02 | headline |
| lr control | 1.26M | 0.005 | isolates the lr change |
| high-parameter | 2.78M / 5.25M | 0.005 | capacity vs the lr control |

Capacity is then the *only* difference between the lr control and the high-parameter row.

## Method of measurement

Counts are **measured, not estimated** — by constructing the models offline
(`DreamerModel`+`Actor`+`Critic` for mamba, `get_uniform_network()` for the method) and by short
2 000-step calibration runs for happo/mbom, then summing `numel()` over the resulting
checkpoints with optimizer and normalizer state filtered out.

## Consequences for existing results

The upstream-capacity baseline results already on disk (`mamba`, `happo`, `mbom` seed 0, plus
mamba seed 1) are **retained and correctly labelled** — they are the "baseline at its own tuned
size" rows. The parameter-matched rows land under the distinct `_pm` keys, so nothing collides.
`m3w_adapted` needed no change, so its seed-0 result stands.

The headline comparison should be read off the capacity-matched rows; the upstream rows remain
available (and are the fairer view of each baseline's own best configuration).
