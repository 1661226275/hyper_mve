# v7 re-run: scope, metrics, and what has to change first (2026-08-09)

Handoff for a fresh session. This session removed `g1 mutual_comp` and built the
`g2cm` family / `rel_coopmix` preset; the experiments themselves are **not**
started. Everything below is either measured in this session or read from the
code with file:line, so nothing has to be re-derived.

Branch `worktree-remove-g1`, pushed. Nine commits, `a0d6fc7` at the tip.

---

## 1. What this session established

### The scope change

`g1 mutual_comp` `(-λ, -λ)` was the only regime with both weights negative, hence
the only purely adversarial one — with it present no single training objective is
correct across the family. It is removed. New family **`g2cm`**, new preset
**`rel_coopmix`** (env id `relation_coopmix`); `g2` / `rel_duo` / `rel_recip` are
untouched and frozen, so nothing archived changes meaning.

| id | name | (w01, w10) | old `g2` id | character |
|---|---|---|---|---|
| 0 | `mutual_coop` | (+λ, +λ) | 0 — unchanged | cooperative |
| 1 | `asym_exploit` | (−λ, +λ) | 2 | mixed |
| 2 | `asym_exploited` | (+λ, −λ) | 3 | mixed |
| 3 | `asym_exploit_mild` | (−λ, 0) | — **new** | mixed |
| 4 | `neutral` | (0, 0) | 4 — unchanged | independent |

**Ids 1–3 changed meaning.** `g1` is `mutual_comp` in every archived table and
`asym_exploit` in anything new. `mutual_coop` stays 0 and `neutral` stays 4
deliberately — the PoA denominator reads `welfare_physical["0"]` and the VoI probe
pins the zero-coupling regime.

### Why a regime was added, not just removed

The handoff this session worked from (`g1_removal_handoff.md`, Trap 4) predicted
VoI would survive. **It was wrong.** Under `reciprocal` coupling an agent observes
`w_01` and must infer the hidden `w_10`; VoI comes only from own-row values
consistent with more than one regime, and a singleton bucket contributes exactly
zero. Measured with `scripts/probes/regime_voi_probe.py --episodes 20`:

| family | `+λ` branch | `−λ` branch | `0` branch | **VoI** |
|---|---|---|---|---|
| `g2` | 5.99 | 5.99 | 0 | **3.99** |
| `g2` minus `mutual_comp`, nothing added | 5.99 | **0** | 0 | **2.00** |
| `g2cm` | 5.99 | 0.79 | 0 | **2.26** |

`2.26` is a **ceiling**, not a tuning choice: a negative weight in the `−λ` bucket
requires both weights negative, i.e. the adversarial regime being removed. The one
lever left is the `0` bucket, still a singleton — a `(0, ±λ)` regime would open it
(est. ~2.5 for one, ~4.3 for a pair) at |G| = 6–7. Full derivation:
`results/analysis/g1_removal.md`.

### Phase 0 — belief settle time, measured before the change

`results/analysis/belief_settle_time.md`. On `rel_recip` (`g2`), seeds 0–1:

**The belief posterior is a deterministic function of the agent's own row, not an
inference.** `asym_exploit` is a dead class — column g2 is zero in every row of the
confusion matrix, on both seeds, under both the prior and planner rollouts. Settle
time is `t* = 0` exactly in the regimes the own-row read answers for free, and
undefined in the two where inference is actually required. `neutral` settles
instantly with a value swing of exactly **0.000**.

This is orthogonal to the scope change — `g2cm` has the same bucket geometry — so
removing g1 neither caused nor fixes it. It does bear on the re-run: **do not
expect the belief channel to carry the comparison.**

### Answers to the three questions raised this session

1. **Headline metric → NashConv (exploitability) + a per-role table.** The
   observation that forced the question: `return_mean` is `ΣR_i` over the
   *subjective* rewards, so the mean of the per-agent returns is exactly
   `return_mean / N` and ranks algorithms identically. Averaging per-agent returns
   does not escape the global metric — it *is* the global metric.
2. **Plumbing → both.** Add per-agent returns natively to `EvalReport` (rel-v3)
   *and* use `game_metrics` for exploitability.
3. **Scope → staged.** Method + strongest baseline (`mamba`, robust 71.91) on
   `relation_coopmix` seed 0 first, to confirm the metric behaves before
   committing the full matrix.

---

## 2. Why the experiments must be re-run

Two independent reasons, both disqualifying for the archived numbers:

1. **The environment changed.** Archived comparative and ablation runs are on
   `relation` (= `rel_duo`, the **v5** control) or `rel_recip`, both family `g2`.
   The regime set is different now and ids 1–3 mean different things, so no
   per-regime column is transferable.
2. **The metric was wrong.** Everything was scored on `return_mean`, the summed
   subjective reward. That sum cancels wherever the two agents' rewards oppose:
   under `g2` it was *identically zero* in `mutual_comp` regardless of policy, and
   reduces to a single agent's harvest in the asymmetric regimes. It cannot
   express individual optimality, which is the property the method claims.

Under `g2cm` the summed metric is visible in 8 of 10 agent-slots (up from 6 of 10),
but the asymmetric regimes still show one agent each — so the fix is a different
metric, not a different family.

---

## 3. Comparative experiment — metric specification

| metric | status | logged | notes |
|---|---|---|---|
| World-model fidelity | unchanged | during training | `fidelity.py`, schema now `fidelity-v2`, already per-regime |
| Global (summed) reward | **restricted** | during training, **g0 and g4 only** | see the flag below |
| Per-agent reward | **new** | during training, **all regimes** | needs rel-v3, see §5 |
| NashConv / exploitability | **new** | during training, **all regimes** | **cost problem, see §4** |

**All five regimes participate in training** for the main comparison — no regime is
present at test but absent at train.

### Flag: g3 also qualifies for the global metric

"Global reward on g0 and g4 only" is exactly the correct rule for the old `g2`
family, where those were the only regimes whose team return is `u_0 + u_1`. Under
`g2cm` a third regime qualifies: `asym_exploit_mild` (id 3) has `ŵ_01 = 0` and
`ŵ_10 = -λ`, so `R_0 = u_0`, `R_1 = (u_1-u_0)/2`, and the pair sums to
`(u_0+u_1)/2` — it responds to **both** agents' harvests. It is the regime that
replaced `mutual_comp` precisely so that nothing cancels.

Restricting to g0/g4 is defensible as a conservative choice, but if the rule was
carried over from the v5 protocol rather than chosen, g3 can be included.
`results/analysis/v6_reporting_protocol.md` currently documents g0/g3/g4.
**Decide explicitly** — this is exactly the kind of carry-over the id renumbering
makes dangerous.

### Generalization split

Train on **g0, g1, g2**; test on **g3, g4**. Under `g2cm` that is:

* train `{mutual_coop, asym_exploit, asym_exploited}`
* held out `{asym_exploit_mild, neutral}`

**This is a different split from the `rel_coopmix_holdout` preset built this
session**, which trains on `(0, 1, 4)`. The preset must be changed to
`train_regime_ids=(0, 1, 2)` — one line in
`hyper_mve/utils/configs/presets/rel_coopmix.py`, plus its test, which asserts on
resolved regime *names* rather than the id tuple and will therefore fail loudly.

The new split is analysed as follows, and it is **better** on the axis that
worried this session:

* Weight values seen in training: `w_01 ∈ {+λ, −λ}`, `w_10 ∈ {+λ, −λ}`. **Zero is
  never seen**, and both held-out regimes contain a zero. So this is a clean
  "unseen weight value" test rather than merely an unseen combination.
* Within training, own-row `+λ` → `{g0, g2}` with hidden `w_10 ∈ {+λ, −λ}`:
  **ambiguous, so there is a genuine inference task during training.** The
  `(0, 1, 4)` split had none — every training own-row value mapped to a unique
  regime, which would have let the posterior be deterministic at train time and
  then asked it to resolve an ambiguity it had never met. This split fixes that.
* Caveat: the regime head has 5 classes but only 3 ever occur in training, so it
  will almost certainly never predict g3 or g4. Combined with the Phase-0 finding
  (the net collapses each own-row bucket to a fixed member), expect the
  generalization test to be **hard**, and read `regime_accuracy` on the held-out
  regimes as near zero rather than as a bug.

---

## 4. NashConv during training — the cost problem

**At default fidelity this is infeasible, by a wide margin.**

`BRConfig.env_steps = 20_000` (`game_metrics.py:155`) is the budget for **one**
best-response DQN, and NashConv needs one per (agent, regime): with N=2 and 5
regimes that is 10 × 20 000 = **200 000 env steps of best-response training per
NashConv evaluation** — equal to the entire main run's budget (`total_env_steps`
200 000 in the current grids). Logging it even ten times during training would
cost 2 M env steps of BR training on top of a 200 K run: a ~10× blow-up.

The handoff's independently recorded figure agrees: >15 min per checkpoint.

Options, roughly in order of what is likely wanted:

1. **Two-tier.** A cheap in-training NashConv as a *trend* line (e.g.
   `br_steps` 1 000–2 000, a handful of probe points) plus one full-fidelity
   post-hoc NashConv at the final checkpoint via `scripts/eval_game_metrics.py`.
2. **Post-hoc only**, at 2–3 checkpoints per run. This is what the current tooling
   is built for and costs nothing during training.
3. Full fidelity in-training on a **subset of regimes**.

**Methodological point that must be recorded whichever is chosen:** NashConv from
an under-trained best response is a **lower bound** on exploitability. Two
algorithms are comparable only at an *equal* BR budget, and the absolute value is
not "the" NashConv. `br_env_steps` is already a field on the report
(`GameMetricsReport.br_env_steps`) — report it beside every number.

Also useful: NashConv is **deterministic** given `(ckpt, --seed)` — wave-1's four
checkpoints re-measured to ±0.00 — so numbers stay comparable across sessions.

---

## 5. Work that must land before launch

### (a) Per-agent returns in `EvalReport` — rel-v3

Today only `mazero_mixed` produces per-agent returns, in
`eval_diagnostics.json` → `return_per_regime_planner_per_agent`
(`hyper_mve/algo/runner.py:800`). The five baselines accumulate a **scalar** per
episode (e.g. `mappo.py:390` `g_returns: list[float]`; `_run_one_episode` returns a
step count), so their `EvalReport` carries no per-agent breakdown at all.

Needed:

* new field `return_per_regime_per_agent` on `EvalReport`, sentinel `rel-v2` →
  `rel-v3`;
* the five baselines' evaluate loops accumulate an `(N,)` vector instead of a
  scalar — small change each, zero extra runtime cost;
* `hyper_mve/comparison/_probe.py` (the periodic in-training probe) logs only
  `return_regime_{g}` today (`_probe.py:129`) and `_one_episode` returns a float —
  per-agent has to be threaded through there too, since the requirement is
  *during training*;
* `tests/integration/test_pkg08_drift_detectors.py` field count 29 → 30, and
  `tests/comparison/test_external_eval_contract.py` extended.

**Cross-algorithm per-agent numbers already exist** via
`game_metrics.v_pi[g][i]` — the mean per-agent *subjective* return — and
`eval_game_metrics.py --external <variant>` covers every registry key through
`_probe_act`. Episode seeding (`seed + 97*g + ep`) was deliberately matched to
`runner.evaluate` so `v_pi` is comparable against `eval_report.json`. That is the
zero-code path for the post-hoc table; the rel-v3 work is what makes it native and
available *during* training.

### (b) The holdout preset

`train_regime_ids=(0, 1, 4)` → `(0, 1, 2)`. See §3.

### (c) Run from the main checkout

The vendored baselines (`HARL`, `m3w-marl`, `MBOM`, `mamba`, `MAZero`,`DIMA`) live
in `hyper_mve/comparison/vendor/`, which is **gitignored** (own clones, SHAs pinned
in `VENDOR.md`). They exist in `/home/data/zhengwenbo/hyper_mve` but **not in any
worktree** — which is the real cause of the 13 comparison-test failures seen here
(`ModuleNotFoundError: harl / m3w / utils`), identical at the base commit. Merge
`worktree-remove-g1` into the main checkout before launching; `launch_grid.sh:17`
hardcodes `REPO` to it anyway.

### (d) New grids

The existing `compare_gpu*.yaml` / `ablation_gpu*.yaml` all point at
`envs: [relation, relation_holdout]` — v5 `rel_duo`. They are not re-runnable as
written. `scripts/grids/v6_coopmix_smoke.yaml` (added this session) is the
template. Staged first pass: `mazero_mixed` + `mamba` on `relation_coopmix`,
seed 0.

---

## 6. Ablation scope

Two modules, unchanged in intent:

**Role-awareness module.** Existing arms are usable as-is:
`ref_bc_anneal_scaled_no_subjective_decoupled` (the matched control — plain
MAZero, same anneal, same decoupled selection, differs only by this module),
`ref_bc_belief_blind` (uniform posterior, head diversity 0.000),
`point_estimate_leaf`, and the conditioning-mechanism arms (`moe_router`, `film`).
Note the standing result that `belief_blind` scored 62.94 against A1's 61.94 —
the posterior's *informativeness* was not the driver of return under the old
metric. Re-measuring this under per-agent/NashConv is one of the more interesting
things the re-run can settle.

**Tree-search module — the design does need rework.** Why, concretely:

* The 2×2 policy-target screen (`_cover` / `_qtarget` / `_mctsfix`) and the
  action-axis wave (`_agentq`, `_agentq_cover`) were selected largely on **g1
  NashConv** — the regime that no longer exists. `agent_q_softmax` was recorded as
  "REJECTED" on the strength of a g1 result (+17.61 ± 4.08); rescored without g1
  it is +6.85 / +7.56 at 0.66 / 0.44 se — **unresolved and underpowered, not
  rejected**. Settling it needs n ≈ 30/cell at the observed seed variance.
* `min_qstd` (`core/utils.py:465`) — the advantage-spread guard — was tuned on
  measurements taken *in* `mutual_comp`. It is scale-free by construction so it
  should degrade gracefully, but the "~20% of transitions" figure does not carry
  over and the threshold is **unvalidated on `g2cm`**. Annotated in the code this
  session, deliberately not retuned.
* `visit_q_blend` (`arms.py:315`) was selected partly on g1 NashConv. Same status:
  annotated, not re-selected.

So the tree-search ablation needs a new arm set chosen against the *new* headline
(NashConv + per-role return) rather than inherited from g1-scored screens. The
architectural axis (`joint_selection` vs decoupled per-agent selection,
`point_estimate_leaf` vs Bayes-average leaf) is unaffected by the scope change and
can carry over directly.

---

## 7. State of the repository

* Branch `worktree-remove-g1`, pushed to `origin`, tip `a0d6fc7`.
* Tests: 217 passed (schemas/configs/envs/eval/integration), 192 passed + 1 skipped
  (algo), 56 passed (comparison). Comparison also fails 13 — **identical to the
  count at base commit `d0df2d4`**, all missing vendored deps, verified by running
  the suite at the base commit.
* End-to-end verified on `relation_coopmix`: a real 3 000-step training run emitted
  `rel-v2` / `evaldiag-v3` / `fidelity-v2` / `game-metrics-v2` artifacts with
  `regime_names` populated on every one.
* Key documents: `g1_removal.md` (the decision and the VoI derivation),
  `belief_settle_time.md` (Phase 0), `v6_reporting_protocol.md` (rewritten — the
  g1/NashConv split is deleted), `g1_removal_handoff.md` (superseded, Trap 4
  corrected in a banner).
