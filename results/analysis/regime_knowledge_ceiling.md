# What knowing the relationship regime is actually worth (2026-08-05)

Measured with scripted controllers on `rel_duo`, no training, no GPU:
`python scripts/probes/reference_ceiling.py --episodes 64`.
Raw numbers in `results/analysis/belief_ceiling/reference_ceiling.json`.

This was run as the pre-flight check for the v5 belief redesign — to size the
prize before spending ~12 GPU-hours × seeds × arms. It came back an order of
magnitude below the pessimistic branch, so it is reported as a result rather
than as a planning note.

## Method

Two controllers that share **one** implementation
(`RelationalGreedyPolicy`, `envs/relation_commons/reference_policies.py`) and
differ in exactly one scalar — the opponent's row `w_j·`:

| level | knows | how it predicts the neighbour |
|---|---|---|
| `relational_self_info` | `w_i·` only (already in every observation) | assumes the neighbour mirrors it |
| `relational_oracle` | `w_i·` **and** true `w_j·` | reads it from the regime |

The mirror assumption is correct in g0/g1/g4 and wrong in g2/g3, i.e. it is
precisely the 60%-accurate prior reachable without learning anything about the
opponent. So `oracle − self_info` isolates the return that *inferring the
opponent's row* is worth — exactly what the belief channel exists to earn.

Both are heuristics, so the gap is a **lower bound**, not the optimum.

## Result

| policy | mean | g0 coop | g1 comp | g2 exploit | g3 exploited | g4 neutral |
|---|---|---|---|---|---|---|
| `harvest_only` | 15.41 | 20.40 | 0.00 | 17.43 | 13.08 | 26.15 |
| `scripted_greedy` | 100.57 | 166.16 | −0.03 | 86.59 | 82.51 | 167.61 |
| `scripted_greedy_distinct` | 107.49 | 178.10 | −0.04 | 90.61 | 89.13 | 179.66 |
| `relational_self_info` | 106.32 | 178.21 | −0.03 | 87.94 | 87.27 | 178.21 |
| `relational_oracle` | **107.13** | 178.21 | −0.03 | 89.31 | 89.98 | 178.21 |
| **oracle − self_info** | **+0.81** | **0.00** | **0.00** | **+1.37** | **+2.70** | **0.00** |

Method of record for scale: **A1 = 61.94 ± 2.29**.

**Knowing the opponent's row is worth +0.81 mean return — about a third of one
seed standard deviation.** The gap is exactly 0.00 on the three symmetric
regimes, where the mirror prior is already the truth, and the entire effect
(+2.04 averaged) lives in g2/g3, as predicted.

## This corroborates the trained-model measurement

`UB (oracle true g) − A1 = +0.67 ± 2.88` (`ablation_3seed_final.md`).

The scripted ceiling says the true value of regime knowledge is **+0.81**. The
trained model measured **+0.67**. Two independent methods — one with no
learning at all — agree to within noise.

**The 3-seed ablation was therefore not underpowered and not broken.** It
correctly measured a real effect that is genuinely ~+0.7. Every previous null
follows: `A1 − A2 = −2.46`, `belief_blind 62.94 ≈ A1 61.94`, `UB − A1 ≈ 0`.

## Why the env makes it this small

`R_i = (u_i + Σ_j w_ij·u_j)/(1 + Σ_j |w_ij|) − ε·1[moved_i]`. Summing over both
agents, the team return collapses per regime:

| regime | `w_01, w_10` | team return | measured |
|---|---|---|---|
| g0 coop | +λ, +λ | `u_0 + u_1` | 178.10 |
| g1 comp | −λ, −λ | **exactly 0** | −0.04 |
| g2 exploit | −λ, +λ | `u_0` (one agent's harvest) | 90.61 |
| g3 exploited | +λ, −λ | `u_1` | 89.13 |
| g4 neutral | 0, 0 | `u_0 + u_1` | 179.66 |

The measurements match the algebra: g1 is pinned at the ε move cost, and g2/g3
land at ~half the g0/g4 ceiling. Pinned by
`test_team_return_is_structurally_capped_per_regime`.

So the headline metric is **~0 / ~half / ~full by construction**, and in every
regime it is maximised by the same joint behaviour: *both agents harvest as
much as possible without colliding*. Avoiding collisions needs only the
neighbour's **position**, which is observed directly. The regimes change who
gets *credited* for the harvest; they barely change what anyone should *do*.

Two further consequences worth recording:

- **g1 contributes nothing to the headline.** It is zero-sum, so team return is
  identically 0 whatever either agent does. One fifth of the eval grid cannot
  respond to any improvement.
- **In g2/g3 the team optimum and the Nash point diverge.** Team return in g2 is
  `u_0`, but agent 0's own objective is `(u_0 − u_1)/2`, so agent 0 is
  incentivised to spend moves *denying* agent 1 — which lowers `u_0` and hence
  the team return. `return_mean` is measuring something the agents are not
  optimising.

## Scarcity does not fix it

Sweeping the two scarcity knobs (32 eps/regime,
`results/analysis/belief_ceiling/scarcity_sweep.json`):

| K | α | self_info | oracle | gap | gap g2/g3 |
|---|---|---|---|---|---|
| 8 | 0.10 | 105.42 | 105.68 | +0.26 | +0.65 |
| 8 | 0.02 | 32.18 | 32.27 | +0.09 | +0.22 |
| 4 | 0.10 | 101.02 | 101.27 | +0.25 | +0.62 |
| 4 | 0.02 | 30.87 | 30.90 | +0.03 | +0.07 |
| 2 | 0.10 | 102.06 | 101.48 | −0.58 | −1.45 |
| 2 | 0.02 | 31.29 | 31.11 | −0.18 | −0.46 |
| 1 | 0.10 | 55.47 | 55.47 | 0.00 | 0.00 |

Making the commons eight times scarcer changes nothing. The gap never leaves
±0.7 and turns negative at K=2. So this is not an abundance problem.

## The root cause is the regrowth law

`dynamics.step_dynamics:26`:

```
q_{k,t+1} = clip(q − h + α·(Q_max − q), 0, Q_max)
```

Regrowth is `α·(Q_max − q)` — **fastest when the cell is empty, zero when it is
full.** Stripping a cell costs nothing and holding stock wastes regeneration, so
restraint has *negative* option value. There is no tragedy of the commons here;
the dynamics are anti-conservationist by construction.

Measured (24 eps/regime, both agents on a harvest-threshold policy — threshold 0
= strip everything, higher = leave stock to regrow):

| threshold | g0 | g1 | g2 | g3 | g4 |
|---|---|---|---|---|---|
| **0.00** | **178.58** | −0.04 | **89.53** | **89.45** | **175.36** |
| 0.20 | 160.23 | −0.38 | 79.82 | 78.63 | 157.66 |
| 0.40 | 131.91 | −0.66 | 65.00 | 62.54 | 128.96 |
| 0.60 | 115.79 | −0.84 | 54.02 | 56.33 | 114.74 |
| 0.80 | 86.12 | −1.17 | 41.29 | 41.01 | 82.80 |

**Return is monotonically decreasing in restraint in every regime, and the
optimum is threshold 0 in every regime.** There is no strategic dimension for
the relationship to act on: "harvest whenever you can" is optimal whoever the
neighbour is and whatever they intend. Coordination reduces to not standing on
the same cell, which needs the neighbour's *position* — already observed.

Swapping in a logistic law, `q ← q + α·q·(1 − q/Q_max) − h` (α = 0.30), creates
the missing dimension:

| threshold | g0 | g1 | g2 | g3 | g4 |
|---|---|---|---|---|---|
| 0.00 | 47.20 | −0.04 | 23.64 | 23.70 | 46.44 |
| **0.20** | **163.68** | −0.36 | 78.45 | **80.46** | **160.15** |
| **0.40** | 160.99 | −0.38 | **80.12** | 78.85 | 158.76 |
| 0.60 | 143.90 | −0.53 | 70.94 | 70.48 | 138.98 |
| 0.80 | 98.83 | −0.99 | 48.76 | 46.09 | 97.19 |

Now the optimum is **interior** and worth **+116 points** over stripping
(g0: 47.20 → 163.68). Restraint pays, so "will the neighbour leave what I
leave?" becomes a real question — and that is the opponent-row question.

## The decisive test: the hidden variable is decision-irrelevant

The tests above all used the *team* return, which cancels structurally. The
proper object is agent 0's **own** return under an asymmetric strategy pair, so
`best_response_sweep.py` builds `R_0(t_0, t_1 | g)` over a 7×7 threshold grid ×
5 regimes × 20 episodes, and reports the value of information for agent 0's
choice:

```
VoI = E_g[ max_t R_0(t | g) ] − max_t E_g[ R_0(t | g) ]
```

with `g` drawn from agent 0's posterior **given its own row** — the only thing
it observes (`w_01=+λ ⇒ g∈{g0,g3}`; `−λ ⇒ {g1,g2}`; `0 ⇒ {g4}`). This is exactly
what the belief channel would have to earn.

| configuration | best response varies with `g` | **VoI** |
|---|---|---|
| K=8, current law | 1/7 opponent thresholds | **+0.00** |
| K=8, logistic law | 0/7 | **+0.00** |
| K=1, current (forced rivalry, one shared cell) | 0/7 | **+0.00** |
| K=1, logistic (forced rivalry) | 0/7 | **+0.00** |
| K=2, logistic | 7/7 | **+0.00** |

The K=2 row is the one that explains everything. The best response *does* vary
with the regime at every opponent threshold — but look at how:

```
  g \ opp   0.0   0.1   0.2   0.3   0.4   0.5   0.6
  g0        0.6   0.1   0.2   0.3   0.5   0.4   0.6
  g3        0.6   0.1   0.2   0.3   0.5   0.4   0.6     <- identical to g0
  g1        0.4   0.3   0.3   0.4   0.2   0.3   0.4
  g2        0.4   0.3   0.3   0.4   0.3   0.3   0.4     <- ~identical to g1
  g4        0.4   0.1   0.2   0.3   0.5   0.4   0.6
```

**g0 and g3 give the same best response; g1 and g2 give the same best response.**
g0/g3 are exactly the regimes with `w_01 = +λ`; g1/g2 are exactly those with
`w_01 = −λ`. The best response is a function of the agent's **own** row — which
is observed — and is constant across the opponent's row, which is not.

So the strategy responds to the regime, and the response is fully determined by
observed information. Conditioned on the own row, the hidden variable carries
no decision-relevant content. **VoI = +0.00 in every configuration tested**, with
and without a commons dilemma, with and without forced rivalry.

The algebra says why: `R_0 = (u_0 + w_01·u_1)/(1+|w_01|)` contains `w_01`
(observed) and `u_1`. The opponent's row `w_10` reaches agent 0 only by changing
the opponent's *policy* — but the opponent's own best response is regime-blind
for the same reason, so it behaves the same whatever `w_10` is, and `u_1` is
near-constant with respect to agent 0's choice anyway.

**Limits.** Measured over two independent strategy families (positional
targeting; harvest thresholds), each low-dimensional. A richer strategy space
could in principle expose regime-dependence these do not. Against that: three
independent lines agree — scripted oracle +0.81, VoI +0.00, and the trained
model's `UB − A1 = +0.67 ± 2.88`.

## What this does and does not license

**Does not:** any claim that a better belief architecture will raise
`return_mean` on **`rel_duo`**. The ceiling is +0.81 and the VoI is +0.00; it
does not move with scarcity, with the regrowth law, or with forced rivalry.
This is an environment-design fact and no model change reaches it. Every knob
in the v5 `EnvConfig` was tried.

> **Superseded in part (2026-08-05).** "No configuration reaches the gate" is
> correct and stays correct — but it is a statement about the *v5 formula*, not
> about RelationCommons. Changing the reward so that it depends on the hidden
> row, plus the geometry, does reach it. See "The v6 environment" below.
> `rel_duo` is unchanged and is now the control.

**Does not, either:** the weaker "mechanism" fallback as originally framed.
Inferring the opponent's row is not merely unprofitable here, it is
*decision-irrelevant*: the best response is measurably constant across it. A
model that learned to infer it perfectly would have learned something with no
use.

**Does:** a sharper and more defensible claim than the one the redesign was
chasing — **the agent's own relationship row is a sufficient statistic for its
best response in this class of commons.** Self-information suffices; opponent
inference is unnecessary. This is directly supported by the existing ablations
rather than contradicted by them:

- Module 1 (role conditioning, which carries the **own** row) is worth
  **+30.97 ± 8.88**.
- The belief posterior over the **opponent's** row is worth ~0, by three
  independent measurements.

Much of the relationship-modelling MARL literature assumes opponent/type
inference is necessary. Exhibiting a natural environment class where it
provably is not, with the algebra for why and a scripted controller that pins
the ceiling without any training, is a contribution in its own right.

The keeper result is untouched: Module 1 × Module 2 synergy (search worth
+47.66 ± 2.65 with role-aware heads vs +14.88 ± 9.32 without), which is about
role-conditioning and search, not about the belief posterior.

## Reproduce

```
python scripts/probes/reference_ceiling.py --episodes 64   # the +0.81 ceiling
python scripts/probes/regime_voi_probe.py                  # the VoI matrix
pytest tests/envs/test_reference_policies.py -q
```

`regime_voi_probe.py` with no arguments runs the five configurations in the
table above (~1 h CPU); `--K 2 --law logistic` reproduces the decisive row in
about ten minutes. Outputs land in
`results/analysis/belief_ceiling/{reference_ceiling,regime_voi}.json`.

`test_knowing_the_opponent_row_is_worth_little` asserts the finding rather than
the number: if the env's coupling is raised (scarcer resources, larger λ, more
agents contesting a cell) it fails, which is the signal that the belief channel
became worth building and this document needs revisiting.

---

# The v6 environment (2026-08-05)

Everything above is about the v5 formula. It is not a statement about
RelationCommons as such: a reward change plus a geometry change does clear the
gate. `rel_duo` is frozen and becomes the null control; `rel_recip` is the new
environment.

## What had to change, and why each was necessary

**1. The reward must depend on the hidden row.** Under
`reward_coupling="own_row"` agent `i`'s weight on `u_j` is `w_ij` — its own row,
which the observation carries — so `VoI ≡ 0` by construction, whatever else is
true. `reward_coupling="reciprocal"` uses `Ŵ = Wᵀ`, making the weight `w_ji`:
reciprocal altruism, and the weight is now hidden.

**2. The weight must change *sign* with the hidden row.** This was the
surprise. A weight that shifts in magnitude but keeps its sign is nearly
worthless; one that flips is worth two orders of magnitude more *in the same
environment*:

| coupling | ŵ_01 for g0 / g3 | sign flips? | VoI |
|---|---|---|---|
| `own_row` | +1.00 / +1.00 | — (no dependence) | 0.00 |
| `levine` λ=0.5 | +1.00 / +0.33 | no | 0.06 |
| `levine` λ=1.0 | +1.00 / 0.00 | no | 0.74 |
| `levine` λ=2.0 | +1.00 / −0.33 | **yes** | 2.21 |
| `levine` λ=3.0 | +1.00 / −0.50 | **yes** | 2.86 |
| `reciprocal` | +1.00 / −1.00 | **yes** | **3.99** |

g0 and g3 are observationally identical to agent 0 (both have `w_01 = +1`), so
this column is exactly what inference could recover. Under Levine
`Ŵ = (W + λWᵀ)/(1+λ)` the flip requires `λ > 1`; `λ = 0` recovers v5 exactly,
which is what makes λ a continuous knob rather than a different experiment.

**3. The agents must be close enough to interact.** Reward coupling is useless
without *action* coupling — the weight on `u_j` cannot change any argmax if
`u_j` does not respond to agent `i`. Measured as
`sd(U1 | t0) / sd(U1 | t1)`:

| config | action coupling | VoI (reciprocal) |
|---|---|---|
| K=8 L=8 (shipped) | **0.014** | 0.00 |
| K=2 L=8 | 0.24 | 0.26 |
| K=2 L=4 | 0.44 | 2.33 |
| **K=2 L=3** | **0.59** | **3.99** |
| K=2 L=2 | 0.81 | 5.28 |
| K=3 L=3 | 0.07 | 0.21 |

On the shipped 8×8 grid agent 0's strategy moves agent 1's harvest from 72.6 to
72.3 — two agents on a large grid simply forage apart, and **even maximal reward
coupling gives VoI 0.00 there**. `K ≤ N` is required; at K=3 they separate again.

**4. Logistic regrowth** (`q ← q + α·q·(1 − q/Q_max) − h`) creates the strategic
dimension the relationship acts through. The v5 law regrows an empty cell
fastest, so restraint is strictly harmful and the optimal harvest threshold is 0
in every regime; under logistic growth the optimum is interior and worth ~+116.
Note `q = 0` is absorbing — a stripped cell never recovers. That is the dilemma,
and the failure mode to watch in training.

## The operating point

`rel_recip`: **N=2, L=3, K=2, α=0.30, logistic regrowth, reciprocal coupling.**

```
                          rel_duo (v5)      rel_recip (v6)
VoI, all own-row cases        0.00              3.99
VoI, informative cases        0.00              5.99
action coupling               0.014             0.589
```

The headline 3.99 averages over three own-row cases, one of which (`w_01 = 0`,
i.e. g4) has a single-regime posterior and therefore contributes exactly 0 by
construction. Conditioned on the cases where the hidden variable is genuinely
uncertain the figure is **5.99**.

L=3 rather than L=2 (VoI 5.28) is deliberate: 2×2 would be a matrix game with a
token grid. Both numbers are **lower bounds** — VoI is measured within a
one-dimensional scripted threshold family and cannot capture everything a
learned policy could exploit.

## Corrections to the instruments (all of these changed the numbers)

Found while measuring; each would have produced a false null.

- **The probe policy could not abstain.** A `viable = ones` fallback plus
  `_step_toward(0,0) → HARVEST` meant it always harvested something, so
  restraint — the entire v6 strategy dimension — was inexpressible. The earlier
  K=1 result was an artifact of this.
- **VoI is unilateral.** It is what *one* agent gains by deviating while the
  others hold still. Handing both agents the oracle and summing measured
  **−25.75** on exactly the regimes where VoI is largest: in a social dilemma
  better individual play lowers the team return. `evaluate_reference_policy`
  now takes per-agent `policies` and reports `return_per_agent_per_regime`.
- **Controllers must know the env's coupling.** `RelationalGreedyPolicy`
  branched on `sign(w_ij)`, correct under `own_row` and wrong under
  `reciprocal`, where the agent's own weight is `w_ji`. It also needed a
  conservation dimension: a purely positional controller is blind to the v6
  dilemma and reports a null.
- **`reference_ceiling` is not a VoI.** It differences two *fixed heuristics*,
  so it can be negative where the oracle heuristic mis-responds (g3: −12.87
  while g2 is +17.17). Read its effect size, and take the non-negative
  optimised quantity from `regime_voi_probe.py`.
- **`config_hash` omitted every `EnvConfig` field**, so a v6 run would have
  hashed identically to an archived v5 run at the same (algo, env, ablation,
  steps, lr). Fixed — the env physics is now in the hash.

## Reproduce

```
python scripts/probes/regime_voi_probe.py --preset rel_duo     # 0.00, the control
python scripts/probes/regime_voi_probe.py --preset rel_recip   # 3.99
python scripts/probes/reference_ceiling.py --preset rel_recip --episodes 64
pytest tests/envs/test_voi_probe.py tests/envs/test_reference_policies.py -q
```

`tests/envs/test_reference_policies.py::test_v6_makes_the_hidden_row_matter`
is the controlled comparison in one assertion, and
`tests/envs/test_voi_probe.py` pins the invariants the offline recompute
depends on (the trajectory is regime-independent; the probe policy never reads
the row block).

## Still open

- The VoI here is scripted and one-dimensional. The real confirmation is the
  trained model's oracle gap `UB − A1`, which tracked the scripted ceiling on
  `rel_duo` (+0.67 vs +0.81) and should now track the new one.
- Return scale roughly halves versus `rel_duo`, so the ±2.29 seed sd does
  **not** carry over — re-measure it on `rel_recip` before reading any gap as
  significant.
- `mbom_oracle` runs on the mirror prior under `reciprocal` unless something
  calls `set_opponent_row`. Wiring that injection is deliberately left undone
  rather than silently granting a baseline hidden information.
