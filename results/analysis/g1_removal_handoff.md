# Removing g1: scope decision + implementation handoff (2026-08-09)

> **SUPERSEDED by `results/analysis/g1_removal.md` — and Trap 4 below is WRONG.**
>
> Trap 4 predicted VoI would survive or rise. It measures **2.00** for the bare
> removal, down from 3.99: the reasoning considered `mutual_comp`'s own
> contribution but missed its role as the *confusion partner* that kept the
> `w_01 = −λ` observation ambiguous. A singleton own-row bucket contributes
> exactly 0 by construction. The implemented family adds a non-adversarial
> `asym_exploit_mild (−λ, 0)` to restore that ambiguity, reaching 2.26.
>
> Trap 1 (id renumbering) and Trap 2 (chance level 0.200 → 0.250) also never
> materialised: because a regime was added rather than merely removed, |G| stays
> 5, ids stay contiguous, and the collapsed-belief diagnostic stays valid.
> Trap 3 was resolved by retargeting the holdout to `(0, 1, 4)`.
>
> Retained for the file:line inventory, which is still accurate.

Written at the close of the agent-target wave for a fresh session. The decision
is the user's; everything below is the surface it touches, measured or read from
the code, with file:line so nothing has to be re-derived.

## The decision

Restrict the regime family to **cooperative (g0, g4)** and **mixed (g2, g3)**,
**dropping g1**. Rationale: a change in role relationship should not change the
*nature of the game* or the training objective. With g1 present the family spans
cooperative through zero-sum, so no single objective is correct everywhere and
tractability suffers. With g1 gone every regime is compatible with individual
optimality.

The reporting payoff is direct: `v6_reporting_protocol.md`'s most complex rule —
"g1 is scored on NashConv, everything else on return, never mix the tables" —
**disappears**. All four retained regimes are scored on return (g0/g4 team,
g2/g3 per-agent). NashConv drops from required instrument to optional diagnostic.

Then: **belief settle-time measurement** (last section).

## What g1 actually is

`hyper_mve/utils/schemas/relation.py:135-152`, `build_g2`, N=2, `_w2(w01, w10)`:

| id | name | w01 | w10 | nature |
|---|---|---|---|---|
| 0 | `mutual_coop` | +λ | +λ | cooperative |
| **1** | **`mutual_comp`** | **−λ** | **−λ** | **adversarial — the one being removed** |
| 2 | `asym_exploit` | −λ | +λ | mixed |
| 3 | `asym_exploited` | +λ | −λ | mixed |
| 4 | `neutral` | 0 | 0 | independent |

g1 is the only regime with *both* weights negative, so the scope claim is exact:
removing it leaves no purely adversarial regime.

## Trap 1 — regime ids MUST be renumbered (the big one)

`RegimeFamily.__post_init__` (`relation.py:89-97`) **enforces contiguous ids
0..n−1** and raises otherwise. You cannot keep `(0, 2, 3, 4)`. Removing g1
forces:

| old | old name | new id |
|---|---|---|
| 0 | mutual_coop | 0 |
| 2 | asym_exploit | **1** |
| 3 | asym_exploited | **2** |
| 4 | neutral | **3** |

**So "g1" in a new run means `asym_exploit`, while "g1" in every archived table
means `mutual_comp`.** Every per-regime number in `results/analysis/` and every
`return_per_regime` key in every archived `eval_report.json` silently changes
meaning. This is the single highest-risk part of the refactor.

Two ways out:

* **(a) Relax the contiguity invariant** to allow sparse ids, keeping labels
  stable forever. Touches a validated invariant and every consumer that indexes
  by id positionally (`W_stack`, `rows_stack`, `F.one_hot(..., n_regimes)` at
  `belief_losses.py:127`). More edits, but no historical break.
* **(b) Renumber and make the break loud** — new `relation_family` name (not
  `"g2"`), new `preset_name` (e.g. `rel_coopmix`), and bump the report
  `schema_version` off `"rel-v1"` so an old and a new report cannot be compared
  without noticing.

**Recommendation: (b), plus key per-regime report fields on the regime *name*
rather than the integer id.** Names (`mutual_coop`, …) are already available via
`RegimeFamily.names()` (`relation.py:123`). That permanently removes this class
of bug instead of deferring it, and it is a small change confined to the report
writers.

## Trap 2 — the belief-collapse diagnostic silently inverts

The regime head is sized from the family (`belief_net.py:59`,
`belief_encoder.py:33`), so it auto-resizes to 4. But:

* `subjective_model.py:171` hardcodes `n_regimes: int = 5` as a default —
  must become family-derived or updated.
* **Chance accuracy moves 0.200 → 0.250.** "`regime_acc = 0.200` — exactly
  chance for 5 regimes, the signature of a collapsed belief net" is the standing
  collapse test and appears in `stageA_selection_2x2.md:35`,
  `method_of_record.md:34-40,59-60`, and the v6 protocol. After the change a
  **collapsed** posterior reads 0.250, which those docs would call "above
  chance". Fix the docs in the same commit as the code, not after.
* `belief_confusion_probe.py` emits a "5x5 confusion matrix" (docstring + code).

## Trap 3 — `rel_recip_holdout` breaks by design

`presets/rel_recip.py:build_rel_recip_holdout_config` trains on regime ids
`(0, 1, 4)` — "the symmetric training regimes" — and holds out the asymmetric
`(2, 3)`. Removing g1 leaves it training on 2 regimes and holding out 2, which
is a much thinner zero-shot design and no longer "the symmetric ones" in any
natural sense. **Needs an explicit decision**: retarget the holdout, or retire
it for this family.

## Trap 4 — VoI is probably safe, but verify it first

The preset exists because VoI on `rel_duo` was 0.00 and `rel_recip` measures
**3.99** (`regime_knowledge_ceiling.md`; preset docstring). Under `reciprocal`
coupling agent `i`'s weight on `u_j` is the hidden `w_ji`, so VoI comes from
regimes where `w_ji ≠ w_ij` — that is **g2/g3, both retained**. g0/g1/g4 are
symmetric, so g1 should contribute ~nothing and removing it may even *raise* VoI
by removing a confusable symmetric regime.

**This is reasoning, not a measurement.** The prior over regimes changes (5 → 4
uniform), which alone can move the number. Run it before anything else — it is
cheap and it gates the whole preset:

```bash
$PY scripts/probes/regime_voi_probe.py --preset rel_recip --episodes 20
```

Note the probe has **no regime-subset flag** (`regime_voi_probe.py:188-193`); it
takes `--preset`, so measure it *after* the new preset exists, or add a
`--regimes` flag.

## What the scope change does to the last wave's conclusion

**It reopens it.** The only significant effect in the agent-target wave was g1
NashConv (+17.61 ± 4.08); return differences were inside noise. Rescored on the
retained family (`agent_target_wave2/rescore_without_g1.py`):

| contrast | delta | se | |
|---|---|---|---|
| none | +6.85 | 10.40 | +0.66 se — not significant |
| star | +7.56 | 17.07 | +0.44 se — not significant |

So in the g1-free scope `agent_q_softmax` is **unresolved and underpowered, not
rejected** — nominally ahead ~7 points with a consistent sign in all four cells.
Resolving it would need n≈30 per cell at the observed seed variance, which is
why it should stay parked. The banner in `agent_target_wave2_results.md` records
this so the word "REJECTED" is not carried forward past its evidence.

Also worth keeping from that wave: **NashConv is deterministic** given
`(ckpt, --seed)` (re-measured wave-1's four checkpoints to ±0.00), so if it is
kept as an optional diagnostic its numbers stay comparable across sessions.

## Suggested order

1. `regime_voi_probe` on the current preset (baseline before touching anything).
2. Decide Trap 1 (a) vs (b) and Trap 3. These change what the code looks like.
3. New family builder + preset; report fields keyed on regime name.
4. Update the chance-level docs **in the same commit**.
5. Re-run VoI on the new preset; gate on it not collapsing.
6. Delete the g1/NashConv split from `v6_reporting_protocol.md` — the payoff.
7. Belief settle time.

## Belief settle time — what already exists

Do not build from scratch. `scripts/probes/belief_confusion_probe.py` already
computes, offline on a checkpoint, **accuracy vs within-episode timestep**
("early-episode aliasing vs flat ceiling"), plus calibration (mean max-prob and
entropy split by correct/wrong) and the value swing |V(belief) − V(oracle)|.
Settle time is a statistic over the trajectory it already walks.

The measurement to add: per (episode, agent), the first timestep `t*` at which
the posterior over regimes is both correct and stays above a confidence
threshold for the rest of the episode — reported as a distribution, per regime,
not a mean. `regime_switch_prob = 0.0` in this preset (`rel_recip.py`), so the
regime is **static within an episode** and settle time is well-defined; if that
is ever turned on, the definition needs rethinking.

Two cautions carried from prior work: the distilled-prior rollout and the
planner rollout are different policies with different belief accuracy (the probe
has `--planner-dist` for exactly this — report which one), and the standing
result that the belief posterior contributes ~nothing to control
(`hyper_mve_bayes_averaging_negative_result`, QMDP + frozen-root) means a fast
settle time would *not* by itself imply belief is doing work. Pair it with the
value-swing number the probe already produces.

## Repo state

Branch `worktree-per-agent-mcts`, pushed. Wave-2 results, raw game-metrics
JSONs, and the aggregation + rescore scripts are under
`results/analysis/agent_target_wave2/`. Test gate green (52 passed) at the time
of writing. **No part of the g1 removal has been started** — it is left clean
because Traps 1 and 3 are genuine design decisions, and a half-applied
renumbering is far worse than none.
