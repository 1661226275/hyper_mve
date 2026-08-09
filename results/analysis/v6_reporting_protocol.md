# How to score a run on `rel_recip` (v6) — 2026-08-05

`return_mean` is **not** the v6 headline. It sums over agents, and on this
reward that sum is structurally blind in three of five regimes. This is the
protocol for anything comparing arms on v6, including the policy-target 2×2.

## Why the summed metric cannot be the headline

Team return collapses per regime. Measured with scripted controllers on
`rel_recip`, 64 eps/regime (`reference_ceiling_rel_recip.json`):

| regime | team return reduces to | span, noop → oracle |
|---|---|---|
| g0 coop | `u_0 + u_1` | 0.00 → 118.21 |
| g1 comp | **`−ε·(moves)`** | 0.00 → −1.34 |
| g2 exploit | `u_1` only | 0.00 → 52.83 |
| g3 exploited | `u_0` only | 0.00 → 52.80 |
| g4 neutral | `u_0 + u_1` | 0.00 → 43.38 |

**g1 is anti-informative.** `R_0 + R_1 = (u_0−u_1)/2 + (u_1−u_0)/2 − ε(m_0+m_1)`
— the harvests cancel exactly and only movement cost survives. The column is
*maximised by standing still*: `noop` and `harvest_only` score exactly 0.00,
`random` −1.34, every competent policy −0.02 to −0.04. Including it in a mean
adds a small penalty for moving and nothing else.

**g2/g3 measure one agent each.** An improvement in the other agent is
invisible, and can read as a regression: when the oracle controller correctly
optimised agent 0's own objective in g3, the team sum *fell* by 12.87.

Agent-slots the summed metric can respond to: 2 + 0 + 1 + 1 + 2 = **6 of 10**.

## The protocol

| regimes | metric | where it comes from |
|---|---|---|
| g0, g4 | team return | `eval_report.json` → `return_per_regime` |
| g2, g3 | **per-agent** return | `eval_diagnostics.json` → `return_per_regime_planner_per_agent`, or game metrics `v_pi[g][i]` |
| g1 | **NashConv only** — report return as `n/a` | `scripts/eval_game_metrics.py` |

`return_mean` stays in the schema unchanged, so archived v5 runs remain
comparable. It is simply not what a v6 comparison is read on.

## Instruments

```
# per-regime, per-agent exploitability + welfare. 4-16 min per checkpoint.
python scripts/eval_game_metrics.py --ckpt <run>/ckpt.pt \
    --external mazero_mixed --preset rel_recip --frozen-mode both \
    --br-steps 20000 --episodes 10 --out results/analysis/gm_<run>.json
```

`--frozen-mode both` runs the distilled prior *and* the MCTS planner and prints
the per-regime gap. Freezing the prior alone would measure the wrong policy:
on an archived v5 checkpoint the prior's physical welfare is 23.24 in g0 —
**exactly the constant-HARVEST collapse constant** — while the planner reaches
90.43. They are not the same policy and their exploitability is not the same
number.

NashConv is a **lower bound**: the best response is an independent double-DQN at
a fixed budget, so a stronger BR would only raise it. Report `br_env_steps`
alongside any NashConv value.

**NashConv is deterministic given `(checkpoint, --seed)`, so it has no session
noise floor.** Re-running the four wave-1 checkpoints on 2026-08-09 reproduced
their 2026-08-07 values to ±0.00 (4.90, 5.96, 3.72, 22.71). The cross-session
reproducibility floor that applies to *training* (archived g4 100.905 re-running
at 96.969) therefore does **not** apply here: NashConv values measured in
different sessions are directly comparable, and a difference on this metric is
real. Seed count is still a power limit — it just is not a comparability one.
Cost is >15 min per checkpoint even for a single regime, since the BR budget
dominates; do not wrap a run in a short `timeout`.

### The BR budget is not a detail — do not cut it

Measured on the same v6 checkpoint, g1, planner frozen:

| `--br-steps` | NashConv |
|---|---|
| 3 000 | **0.000** |
| 20 000 | **1.622** |

`exploitability = max(0, v_br − v_pi)` is clamped at zero, so an underpowered BR
reports a competent policy as unexploitable. At 3 000 nearly every cell came
back 0 — that is the BR failing to find an improvement, **not** a property of
the policy. Use the 20 000 default; a run whose NashConv is all zeros should be
suspected of an underpowered BR before it is believed.

### g1 is the case the per-agent split exists for

Same checkpoint, same regime:

```
v_pi (per-agent)  : [-7.345, +6.775]     team sum ≈ -0.57
v_br (per-agent)  : [-5.723, +2.680]
exploitability    : [ 1.622,  0.000]     nashconv = 1.622
```

The team return reads ≈ 0 — "nothing happening" — while agent 1 is ahead of
agent 0 by **14 points**. Reporting only the sum here would describe a
14-point asymmetry as a null result.

### Do not mix game-metrics returns with `eval_report` returns in one table

`v_pi[g]` and `eval_report`'s `return_per_regime[g]` are the same policy but not
the same number. Measured on the v5 method-of-record checkpoint, 16 eps/regime:

| g | `sum(v_pi)` (act_fn, batch 1) | `eval_report` | diff |
|---|---|---|---|
| 0 | 77.04 | 82.23 | −5.19 |
| 1 | −0.49 | −0.53 | +0.04 |
| 2 | 68.38 | 67.93 | +0.45 |
| 3 | 56.87 | 54.00 | +2.87 |
| 4 | 95.74 | 100.91 | −5.16 |

**Keep this in proportion.** The episode-sampling SEM at 16 episodes is ±6.7 to
±10.7 on the v5 method — *larger* than either cause below. Sampling noise is the
dominant error; the effects here are second-order. See "Averaging" below.

Two separate causes, both measured:

- **Batch size.** `evaluate()` runs a regime's episodes as one 16-episode
  lockstep search batch; the act_fn that BR training drives runs at batch 1. The
  compiled tree *is* batch-size invariant, but the model forward is not
  bit-identical across batch sizes on GPU, and over 100 steps that occasionally
  flips an argmax. At matched batch size the two agree bit-for-bit
  (`tests/algo/test_act_fn_contract.py`). Aligning the episode seeding removed
  the systematic offset; this residual is not seeding.
- **Cross-session drift.** `_rollout_planner` is bit-stable *within* a session
  (3 repeats, spread 0.000) and reproduces the archived g0 = 82.227 exactly —
  but archived g4 = 100.905 re-runs today at 96.969 through the identical code
  path. So archived per-regime values carry a cross-session reproducibility
  floor of a few points, presumably GPU/library. `oracle_deploy_1M_registry_protocol.md`'s
  determinism check was a same-session re-eval and does not cover this.

Practical consequence: read arm-vs-arm differences from numbers produced in the
same way, and treat any ~4-point difference against an archived value as
potentially artefactual.

## Caveats to carry into the write-up

- **The ±2.29 seed sd from `rel_duo` does not transfer.** v6's return scale is
  roughly half. Re-measure it on `rel_recip` before calling any 2×2 difference
  significant.
- **Late-training instability**: the single final checkpoint has sd ≈ 8–12
  (`late_training_instability.md`). Select on the last-20% statistic of the
  run's own periodic eval, as `stageA_selection_2x2.md` does.
- **v6 changes only g2 and g3 relative to v5's reward.** `Ŵ = Wᵀ` equals `W`
  whenever `W` is symmetric, which g0/g1/g4 are. Those three differ from v5 only
  through the geometry (L=3, K=2) and the logistic regrowth law.
- **`return_zero_shot_gap` is a trap on non-holdout presets** — unseen is empty
  and scores 0, so the "gap" just repeats `return_zero_shot_seen`
  (`comparison/base.py:133-141`). Use `rel_recip_holdout` for a real zero-shot
  cell.
- **`welfare_physical_mean` / `sustainability_mean` / `fairness_mean` /
  `tragedy_index_mean` in `EvalReport` have no writer** — they are placeholder
  zeros. Take welfare from game metrics (`welfare_physical`), not from these.

## Averaging: more EPISODES, not more repeats

The dominant error in these numbers is episode sampling, not anything numerical.
At the usual 16 episodes the per-regime SEM is **±6.7 to ±10.7** on the v5
method of record — bigger than most differences an ablation is trying to
resolve, and bigger than the cross-session artifact above.

**Repeating an eval and averaging is a no-op.** `evaluate()` is bit-deterministic
within a session: three repeats of `_rollout_planner` gave spread exactly 0.000.
Averaging N identical numbers returns that number. The variation is *between*
sessions, which you cannot sample by re-running here.

**Raising the episode count is nearly free**, because `_rollout_planner` runs a
whole regime's episodes as ONE lockstep search batch and CUDA wall-time is close
to flat in batch size (the same property documented at `_DEFAULT_NUM_PMCTS`).
Measured on the v5 method-of-record checkpoint, g4:

| episodes | mean | SEM | planner wall-s |
|---|---|---|---|
| 16 | 96.97 | 10.27 | 10.8 |
| 64 | 110.97 | 5.00 | 14.4 |
| 128 | 106.64 | **3.54** | 19.2 |

**2.9× less noise for 1.8× the time.** Note the mean itself moves ~10 points
between n=16 and n=128 — at 16 episodes the estimate is not just noisy, it can
be off by more than the effects under study.

So: **run evals at 128 episodes**, not 16. The v6 2×2 grid does. For runs
already completed at 16, re-evaluate the checkpoint rather than re-deriving from
the stored report — `scripts/reeval_checkpoint.py` runs the full dual-mode
protocol on an existing `ckpt.pt` without retraining, and re-evaluating an
archived checkpoint in the *current* session also removes the cross-session
artifact by construction.

Caveat on cost: `episodes` also drives the *prior* pass, which is sequential
(batch-1) and therefore scales linearly. It is a single forward per step with no
search, so 128 episodes is still ~seconds per regime — but it is not flat the
way the planner pass is.

## g1 NashConv across the 2×2, wave 1 (n=2, 2026-08-07)

The first measurement of the regime the summed return cannot see. Planner
frozen, `br_env_steps=20000`, 16 eps. **Lower NashConv = closer to equilibrium.**

| arm | NashConv (s0, s1) | mean | spread | welfare (s0, s1) |
|---|---|---|---|---|
| baseline none/visit | 12.93, 12.72 | 12.83 | 0.21 | 42.3, 42.6 |
| A star/visit | 9.91, 17.94 | 13.93 | 8.03 | 84.3, 27.3 |
| **B none/q_softmax** | **4.90, 5.96** | **5.43** | **1.06** | 81.5, 99.9 |
| A+B star/q_softmax | 3.72, 22.71 | 13.21 | 18.98 | 81.2, 86.1 |

Main effects: `root_cover=star` **+4.44** (worse), `q_softmax` **−4.06** (better).

**This falsifies the standing prediction about `q_softmax` in g1.** `core/utils.py:465`
records that g1's advantage spread is ~0.29 of the batch-pooled spread on v5,
so ~20% of transitions would carry near-uniform `q_softmax` targets — the
expectation was therefore that `q_softmax` is *weakest* in g1. It is strongest
there, by 2.4× on the metric built for the regime, and it is also the tightest
arm after the baseline. Whatever the flat-advantage concern costs, it is
dominated by what the per-agent target buys.

The qualitative signature agrees: baseline's `v_pi` is lopsided in both seeds
(−4.13/+3.69, +3.72/−4.18 — one agent dominating), while B-alone seed 0 is
(−0.11, −0.61), near-symmetric and near zero, which is what approaching a
zero-sum equilibrium looks like.

**Instability is attributable to `star`, not to `q_softmax`.** Adding `star`
raises the seed spread in both directions (0.21→8.03 with visit, 1.06→18.98
with q_softmax); adding `q_softmax` lowers it when `star` is present
(8.03→18.98 is star-on-qtarget, but 12.83→5.43 baseline→B). Same ordering as
the return data. A range over two seeds is a crude σ; the direction is
consistent across three instruments, the magnitude is not established.

Caveat: several per-agent `exploitability` entries are exactly 0.00, meaning the
BR found no improvement for that agent. Per "the BR budget is not a detail"
above, an all-zero report would indicate an underpowered BR — here every arm has
a non-zero agent, so the numbers are usable, but each arm's NashConv is set by
one agent, as expected in a zero-sum game where one side is already near-optimal.

## `visit_q_blend`: q_softmax with the visit counts restored (2026-08-07)

`q_softmax` won wave 1 on every instrument but at ~5x the baseline's seed
spread. The diagnosis is that it **discards the visit allocation entirely**
(`policy_target_weights` took only advantages and the mask), so a 1-visit Q is
weighted like a 20-visit Q. At `num_simulations=25` that is severe:

| `root_cover` | children | visits/child |
|---|---|---|
| none | 5 | 5.0 |
| star | 1 + N·A = 13 | **1.9** |

and it explains the interaction wave 1 could not: same target, 2.6x fewer
visits per child, ~18x the NashConv seed spread (B-alone 1.06, A+B 18.98).

**The form.** The direct product `visit(c)·adv_i(c)` cannot be used: `adv` is
centered, so ~half the children are negative, and a negative weight in
`−Σ w log p` becomes `+|w| log p` — an *unbounded* incentive to drive `p → 0`,
not a soft down-weight. Normalizing does not help either, since
`Σ_c visit(c)·adv_i(c)` is a visit-weighted mean advantage and is ~0 by
construction. The visit term therefore goes in the **exponent**:

    w_i(c) ∝ visit(c) · exp(adv_i(c) / τ)

Both shipped targets are exact endpoints — `τ → ∞` is the visit target,
uniform visits is `q_softmax` — and the first holds at the *loss* level, not
just the weights, because `sampled_actions_log_prob = per_agent_log_prob.sum(dim=1)`
and `visit(c)` does not depend on the agent. `tests/algo/test_policy_target_blend.py`
pins both endpoints plus monotonicity in τ.

### τ was measured, not guessed

`scripts/probes/blend_tau_probe.py` reports `τ_balanced = span(adv) / span(log visit)`,
the point where the two logit terms contribute equally. Seed 0 of each arm, 8 eps/regime:

| arm | children | log-visit span | adv span (normalized) | τ_balanced |
|---|---|---|---|---|
| baseline none/visit | 3.2 | 1.734 | 0.937 | **0.54** |
| B none/q_softmax | 3.9 | 1.934 | 1.090 | **0.56** |
| A+B star/q_softmax | 11.2 | 2.491 | 0.947 | **0.38** |

Swept bracket: **τ ∈ {0.25, 0.5, 1.0, 2.0}**, straddling the measured value,
4 arms × 2 seeds on GPUs 0–7.

**Measure the advantage in the units the loss uses.** The first version of this
probe reported τ_balanced ≈ 0.24 because it used the *raw* advantage span;
`reanalyze_worker` divides by a batch-pooled `adv_std` of 0.44–0.64, so the
loss-space span — and τ — is ~2x larger. A bracket centred on 0.24 would have
put three of four arms on the advantage-dominated side, i.e. re-measuring
`q_softmax` under another name.

### The blend self-regulates in g1

Per-regime τ_balanced is lowest in g1 on every arm (0.47 / 0.40 / 0.16) because
g1's advantage span is the smallest — the documented flat-advantage property.
At any fixed τ the log-visit term therefore dominates *most* in g1, so the
target automatically reverts toward the visit counts exactly where the
advantages are least reliable. That is what `--policy_target_min_qstd` was
built to do with a hard threshold, obtained here continuously and for free —
another reason not to spend an arm on the guard.

## Advantage scale on v6: the `min_qstd` guard arm is NOT motivated

`adv_scale_probe.py`, seed 0 of each arm, 8 eps/regime. The pre-registered
trigger was: *"if g1 on v6 reproduces v5's ~0.29 advantage spread, that explains
q_softmax weakness there and motivates a `--policy_target_min_qstd` guard arm."*

**It does not reproduce.** g1 distortion (`own_std / pooled_std`) is **0.633**
on the visit baseline and **0.576** on the q_softmax arm — flattened, but ~2×
less severely than v5's 0.29. The condition for spending an arm on the guard is
not met, and the guard would mask precisely the transitions where q_softmax is
currently winning.

| | pooled adv_std | g1 distortion | g1 q_spread | g0 q_spread | g0 return |
|---|---|---|---|---|---|
| baseline none/visit | 0.372, 0.399 | 0.633 | 0.311 | 0.510 | 30.4 |
| B none/q_softmax | **0.670, 0.658** | 0.576 | 0.483 | 0.958 | 46.0 |

**Training on the q_softmax target nearly doubles the absolute advantage scale**
(0.372→0.670). That is the opposite of the "q_softmax amplifies Q-noise"
account: noise would widen the spread while the return stayed flat, but spread
and return rise **together** (g0 q_spread 0.510→0.958 with return 30.4→46.0).
Wider spread with better realized return is the value head discriminating
between actions better, not the value head getting noisier.

### The likely mechanism, tying three measurements together

The probe's own decision rule reads small `adv_std` + small `q_spread` as "the
actions genuinely don't discriminate here". For g1 on the baseline that reading
is contradicted by NashConv: a best response gains **12.93**, so there is plenty
to discriminate. The resolution is that `adv_std` is measured over the root's
**sampled children, drawn from the prior** — so a collapsed prior yields
near-duplicate children whose advantages are genuinely close, and that looks
identical to a flat game. On this reading:

- visit target at `root_cover=none` self-reinforces the prior → narrow sampled
  set → small adv_std (0.372) → *and* 12.93 exploitability;
- q_softmax weights each agent by its **own** advantage column → wider sampled
  set → adv_std 0.670 → exploitability 5.43.

Consistent with the return data, the per-agent g2/g3 split, and NashConv. Stated
as a hypothesis: it is three correlated measurements at n≤2, not a controlled
test, and `adv_scale_probe` cannot separate "collapsed action set" from "flat
game" by construction. A direct test would count *distinct* sampled root actions
per arm.

## The `star` instability has a mechanism: prior multiplicity (2026-08-08)

The seed spread under `root_cover=star` was attributed above to the thin
budget (~1.9 visits/child). There is a second, larger, and *structural* cause,
found by reading the cover construction and then measured.

`expand` under `root_cover=star` (`cnode.cpp:341-365`) enumerates, for each
agent `i`, all of agent `i`'s actions against a CRN anchor in which every
*other* agent `j` is pinned at one draw `z_j` from **its own noised policy
prior**. So the block built for agent `j` pins agent `i` at the single sample
`z_i` across `A` of the root's children.

Because the policy head is factorized, every policy target is really an
action-axis weight `W_i(a) = Σ_{c: a_i^c=a} (per-child term)` — and
`q_softmax` sums `exp(adv_i(c)/τ)`, so an action carried by `m` children
collects `m` exp-terms. Under star, `m = A` for exactly one action per agent:
the prior draw. Measured with `scripts/probes/agent_target_probe.py`, seed 0
of each arm, 8 eps/regime:

| arm | children | vis/child | distinct | max mult | mass `q_softmax` | mass `agent_q` | TV |
|---|---|---|---|---|---|---|---|
| none/q_softmax | 4.0 | 6.25 | 3.14 | 1.90 | 0.466 | 0.364 | 0.108 |
| star/q_softmax | 11.2 | **2.22** | **6.00** | **6.00** | **0.567** | 0.228 | 0.340 |

Max multiplicity is exactly `A = 6` in all five regimes, and `q_softmax` puts
**0.567** of that agent's target mass on that one prior sample — resampled at
every root. So `star`, whose purpose was to break prior self-reinforcement,
re-injects each agent's own prior into its own marginal at A-fold multiplicity,
and `q_softmax` amplifies it (`visit` only reaches ~24%, via the forced cover
visits).

This predicts the measured g1 NashConv seed-spread rank order across all four
wave-1 cells: none/visit 0.21 < none/q_softmax 1.06 < star/visit 8.03 <
star/q_softmax 18.98. Pre-registered in
`scripts/grids/v6_agent_target_2x2.yaml`: `agent_q_softmax`
(`core/train.py:agent_marginal_target`) removes the multiplicity term exactly,
so if the star seed spread does *not* collapse under `..._agentq_cover`, this
account is falsified and the visits-per-child account survives.

**Wave 2 result (2026-08-09, n=4): the interaction is confirmed, the fix is
not.** Adding `star` multiplies the g1 NashConv seed spread ~18× under
`q_softmax` (1.06 → 18.98) but changes it by ~1 point under `agent_q` (sd 7.01 →
8.10). That interaction is the multiplicity signature and removing the
multiplicity term removed it, so **this is now the standing explanation for the
`star` × `q_softmax` interaction.** But the registered wording was "collapses
toward the none cells", and it does not: star fell (18.98 → 11.69) while none
*rose* (1.06 → 5.67). `agent_q_softmax` is therefore not a repair — it is 4.2×
more exploitable than `q_softmax` in g1 overall (23.04 vs 5.43, +17.61 ± 4.08,
the only significant effect in the wave) and its return advantage is inside
noise. Full write-up: `agent_target_wave2_results.md`.

**Two figures elsewhere are corrected by this table.** The star root holds
**~11.2** children, not the `1 + N·A = 13` upper bound quoted in
`train.py:policy_target_weights` and the handoff — the anchor joint
`(z_0, …, z_N)` is enumerated once per agent block and deduped
(`cnode.cpp:362`). The budget is therefore **2.22** visits/child, not 1.9.

**And it answers the open question below** about counting *distinct* sampled
root actions: the `none` baseline carries only **3.14** distinct actions per
agent out of `A = 6`, against star's 6.00. That is direct evidence for the
"collapsed action set" reading of the small `adv_std` — the sampled children
genuinely are near-duplicates — rather than the "flat game" reading, which
`adv_scale_probe` could not separate by construction.

## Selected lr for v6: **0.02** (sweep, 2026-08-06)

Step-1.5 sweep on the baseline cell (`ref_bc_anneal_scaled_hardval_decoupled`,
`relation_recip`, 1M steps, seed 0), one run per lr:

| lr | last-20% | sd | `return_mean` n=16 | n=128 | SEM n=128 |
|---|---|---|---|---|---|
| 0.04 | **SIGFPE** | — | crashed | — | — |
| **0.02** | **41.96** | 18.10 | 46.46 | **46.68** | 1.0–1.8 |
| 0.01 | 19.15 | 7.39 | 24.65 | 23.78 | ~1.1 |
| 0.005 | 18.32 | 7.38 | 22.97 | 23.07 | ~1.1 |

The inherited v5 value survives the move to v6, winning by >2× on both the
robust last-20% statistic and the tightened n=128 evaluation.

**0.04 diverged rather than underperforming.** `rc=-8` is SIGFPE inside the
vendored C++ tree — the documented failure where a too-high lr produces a
degenerate root distribution and the C++ selection divides by a zero visit
count. It logged **zero** eval points, i.e. it died before the first periodic
eval. So the bracket is bounded above by instability somewhere in (0.02, 0.04).

**Do not extend the bracket upward.** Normally a winner at the top usable value
would call for extending, but here (a) the boundary is a hard crash, (b) lr=0.02
already shows elevated late-training variance (sd 18.10 vs ~7.4 for the slower
rates), and (c) the 2×2's `root_cover=star` arms change the search and could push
a marginal lr over the edge. Risking twelve 12-hour runs for unknown upside is a
bad trade.

Caveat: the sweep is **one seed per lr**. It separates 46.68 from 23.78, which no
plausible seed variance closes, but it does not establish the seed sd on v6 —
that comes from the 2×2's three seeds and is what any significance claim there
will depend on.
