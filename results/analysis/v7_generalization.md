# Generalization on `rel_coopmix_holdout`: the return gap is not the metric

**2026-08-12, seed 0, 500k env steps.** Trained on `train_regime_ids=(0, 1, 2)`,
evaluated on all five. **g3 `asym_exploit_mild` and g4 `neutral` are held out.**
The partition was chosen so the two held-out regimes probe different things: g4
is the only regime with an unseen own-row VALUE (zero), while g3's `-1` is
already seen in g1, so g3 tests an unseen COMBINATION (partner row 0 instead of
+1). See `tests/configs/test_presets_coopmix.py`.

## 1. The zero-shot return gap measures regime difficulty, not generalization

| arm | mean | seen (g0–2) | unseen (g3–4) | gap |
|---|---|---|---|---|
| method `..._hardval_decoupled` | 102.97 | 98.07 | **110.32** | **−12.25** |
| happo | 101.93 | 97.98 | **107.86** | **−9.88** |
| m3w_adapted | 91.59 | 92.86 | 89.68 | +3.18 |
| mamba | 73.43 | 75.04 | 71.01 | +4.03 |
| mbom | 52.26 | 49.59 | **56.25** | **−6.66** |

Three of five algorithms — including the method and the strongest model-free
baseline — score **higher** on the regimes they never trained on. A negative gap
is not evidence of generalization; it is evidence that g3/g4 are simply easier
regimes to earn return in than g0/g1/g2. Since every arm is evaluated on the same
held-out set, the difficulty term is common to all of them and does not cancel
inside a single arm's seen-vs-unseen difference.

**Consequence: do not report the seen/unseen return gap as a generalization
result.** It was flagged as confounded when the first of these numbers landed;
with five arms and three sign flips the confound is now demonstrated rather than
suspected. The quantity that survives is a *cross-arm* comparison on a fixed
held-out set — which is what §2 does.

## 2. NashConv degradation, which does compare across arms on fixed regimes

Same checkpoints, `br_env_steps = 100000`, 16 episodes/regime. Lower is better.

| arm | trained (g0–2) | held out (g3–4) | Δ | Δ % | all 5 | welfare |
|---|---|---|---|---|---|---|
| **method** (planner) | **13.41** | **16.85** | **+3.44** | **+25.7 %** | **14.78** | **146.99** |
| happo | 25.01 | 37.85 | +12.84 | +51.3 % | 30.15 | 144.61 |
| mamba | 19.68 | 30.87 | +11.19 | +56.9 % | 24.16 | 89.64 |

The method is the least exploitable on the regimes it trained on, on the ones it
did not, and overall — and it degrades roughly half as much in relative terms
(+25.7 % vs +51.3 % / +56.9 %). The absolute degradation is 3.7x smaller.

g3/g4 may well be intrinsically more exploitable than g0–g2, exactly as they are
intrinsically easier for return. That is why the comparison to report is the
**differential** across arms — all three meet the same held-out regimes — and not
any single arm's trained-vs-held-out increase on its own.

**Caveat 1 — the seed-0 checkpoint is not representative, and this is the worst
problem with the table above.** The method's holdout run scores **102.97 at seed
0 and 76.32 at seed 1**: a 26.6-point swing on an identical configuration.
~~more than 12x the method's ±2.16 seed sd on the full regime set.~~ **The
±2.16 is retracted** — it came from a different environment at 1M steps. Against
the measured floor (61.97 points across four identical-command draws,
`v7_seed_variance.md` §7) a 26.6-point swing is *unremarkable*, which makes this
caveat stronger rather than weaker: single checkpoints are not informative here.
Every NashConv figure
in this section was computed on the seed-0 checkpoint, i.e. on the *lucky* one of
the two runs seen so far. Until the seed-1 checkpoint is evaluated at br=100k,
the +25.7 % degradation should be read as "what the seed-0 checkpoint did", not
as a property of the method. The comparison rows have the same exposure: they are
also single checkpoints, and mbom already showed a 23.7-point seed swing on the
full regime set.

This does **not** touch §1 — the return-gap confound is a within-checkpoint sign
pattern across five arms, and seed 1 reproduces it (method −4.72, m3w flipping
+3.18 → −2.54).

**Caveat 2 — frozen-mode.** The method row is `--frozen-mode planner` (it runs
MCTS at eval); happo and mamba have no planner and are frozen as direct policies.
Some of the level difference between the blocks is that, and this data cannot say
how much. The *within-arm* degradation ratio is much less exposed to it than the
levels are, so the +25.7 % vs +51.3 % / +56.9 % comparison is the more defensible
one; the "13.41 vs 25.01" level comparison is not, on its own. Note that caveats
1 and 2 bite the two different halves of the table — 1 the rows, 2 the columns —
so neither the levels nor the ratios are currently safe at n=1.

## 3. Regime accuracy behaves as predicted

The method's holdout run reports `regime_accuracy = 0.289` against a chance of
0.200 (|G| is still 5, only 3 occur in training). Barely above chance, as the
plan predicted for a 5-class head that never sees two of its classes. Consistent
with `v7_module1_evidence.md` §4–5: the posterior is not what is carrying this.

## Gaps

* n=1 seed on every row. Seed-1 holdout training is queued; the NashConv evals
  for seed 1 are not (~24 h each for the mazero rows).
* No NashConv for `m3w_adapted` (its planner is regime-CONDITIONED while
  `FrozenExternalPolicy`'s contract is `(obs, t)` with no regime, so an adapter
  would silently plan as g0 in every regime) or `mbom` (~11.5 h/checkpoint, its
  adapter must run without `no_grad`). Both exclusions are from §4.1 of the
  session-2 handoff and are unchanged here.
* Aggregation note: NashConv figures here are **means over regimes**. Earlier
  session notes quoted the same runs as **sums** (e.g. happo/coopmix 205.25 =
  5 x 41.05). Same data, and the two must never be put in one table.

## Related

`v7_module1_evidence.md` (§6 has the non-holdout NashConv table),
`v7_search_module.md`, `regime_knowledge_ceiling.md` (VoI = 2.256 on this
family, which bounds what any regime-inference mechanism can be worth here).
