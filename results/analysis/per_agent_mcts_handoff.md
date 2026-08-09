# Per-agent MCTS — context handoff (2026-08-08)

Written at the close of the policy-target investigation, for a fresh session
starting the per-agent MCTS work. Everything below is measured, not assumed;
sources are named so nothing has to be re-derived.

---

> ## ⚠️ RESOLVED 2026-08-08 — read this before acting on the plan below
>
> **The proposed first step in this document is a provable no-op. Do not build
> it.** Section ["Most of the machinery already exists and is
> unused"](#most-of-the-machinery-already-exists-and-is-unused) proposes
> plumbing `marginal_visit_count` through as a per-agent visit target. That is
> *mathematically identical* to the `visit` target already shipped.
>
> The policy head is factorized — `sampled_actions_log_prob =
> per_agent_log_prob.sum(dim=1)` — so the `visit` loss
> `-Σ_c π(c)·Σ_i log p_i(a_i^c)` reassociates into
> `-Σ_i Σ_a [Σ_{c: a_i^c=a} π(c)]·log p_i(a)`, and that bracket is exactly what
> `get_marginal_visit_count` computes (`cnode.cpp:95-105`). Both normalizers are
> `num_simulations` — every `marginal_visits` row in
> `hyper_mve/algo/mazero_mixed/test/ctree_golden.json` sums to exactly 25 — so
> not even a scale factor differs. (That also means the commented-out invariant
> at `mcts_sampled.py:213` is **true**; it was disabled as collateral with the
> two genuinely broken assertions beside it.) Pinned as an executable proof:
> `tests/algo/test_policy_target_agent_marginal.py::test_visit_prior_at_huge_temperature_is_the_visit_loss`.
>
> **The corollary is the useful part.** The sampled-child axis is *not* an
> independent design space — it is a parameterization of the `(N, A)` marginal,
> and every target differs only in how it aggregates children onto actions:
>
> | target | induced action-axis weight `W_i(a)` |
> |---|---|
> | `visit` | `Σ_{c: a_i^c=a} visit(c)` |
> | `marginal_visit` (proposed here) | **the same expression** |
> | `q_softmax` | `Σ_{c: a_i^c=a} exp(adv_i(c)/τ)` |
> | `visit_q_blend` | `Σ_{c: a_i^c=a} visit(c)·exp(adv_i(c)/τ)` |
> | `agent_q_softmax` (new) | `exp( [Σ visit·adv_i]/[Σ visit] / τ )` |
>
> The last row is what was built instead: it averages *inside* the exponent
> where the others sum *outside* it, which removes a **multiplicity** term —
> `q_softmax` gives an action carried by `m` children `m` exp-terms.
>
> **This also supplies the mechanism for the star instability this document
> could only attribute to "1.9 visits per child."** The star cover pins each
> agent at one draw from its own noised prior across `A` of its children
> (`cnode.cpp:341-365`). Measured (`scripts/probes/agent_target_probe.py`,
> seed 0, 8 eps/regime):
>
> | arm | children | vis/child | distinct | max mult | mass `q_softmax` | mass `agent_q` | TV |
> |---|---|---|---|---|---|---|---|
> | none/q_softmax | 4.0 | 6.25 | 3.14 | 1.90 | 0.466 | 0.364 | 0.108 |
> | star/q_softmax | 11.2 | 2.22 | 6.00 | **6.00** | **0.567** | 0.228 | 0.340 |
>
> Multiplicity under star is exactly `A = 6` in every regime, and `q_softmax`
> puts 0.567 of that agent's target mass on that one prior draw. This predicts
> the measured g1 NashConv seed-spread rank order — none/visit 0.21 <
> none/q_softmax 1.06 < star/visit 8.03 < star/q_softmax 18.98.
>
> Two figures below are also corrected by that table: the star root holds
> **~11.2** children, not `1 + N·A = 13` (the anchor joint dedups), so the
> budget is **2.22** visits/child, not 1.9. And the `none` baseline carries only
> ~3.1 *distinct* actions per agent out of 6 — the direct count this document's
> protocol asks for.
>
> Status: `agent_q_softmax` / `agent_q_blend` implemented
> (`core/train.py:agent_marginal_target`), tested, probed, registered as arms
> `..._agentq` / `..._agentq_cover`, and **run at n=4 (8 runs, 2026-08-09) —
> then REJECTED.** `q_softmax` remains the target of record. Return gains were
> inside noise (+5.43 / +6.03, 0.65 / 0.44 se); g1 NashConv got significantly
> *worse* (23.04 vs 5.43 in the none cell, +17.61 ± 4.08 — the only significant
> effect in the wave). The multiplicity prediction above was **confirmed** — the
> star × q_softmax interaction vanishes once the multiplicity term is removed
> (star−none spread gap +17.93 → +1.09) — but the cells converge in the middle
> rather than at the low baseline, so this is an explanation, not a fix. Full
> write-up: `agent_target_wave2_results.md`. The variant the evidence motivates
> next is **`min`-aggregation** (`Q_i(a) = min_{c: a_i^c=a} adv_i(c)`), which is
> the operator a zero-sum regime actually calls for and is one line now that
> `agent_marginal_target` exists.
>
> The "intent-count" variant (recording each
> agent's pre-projection choice in `select_child_decoupled`) was considered and
> rejected: the exploration bonus is driven by *realized* child visits, so
> intent never self-corrects and drifts toward the actions the search could not
> evaluate.

---

## Why we are here

`rel_recip` (v6) is the current env: N=2, L=3, K=2, logistic regrowth,
**reciprocal coupling** `Ŵ = Wᵀ`. It exists because on v5 (`rel_duo`) the hidden
regime was provably decision-irrelevant — VoI **0.00**, confirmed three
independent ways. On v6 VoI is **3.99**
(`results/analysis/regime_knowledge_ceiling.md`).

Two policy-target attempts, both on the method of record
(`ref_bc_anneal_scaled_hardval_decoupled`), lr 0.02, 1M steps, 128 eval eps:

| target | return (n=2) | g1 NashConv | disadv. agent g2/g3 | seed spread |
|---|---|---|---|---|
| `visit` (default) | 25.08 | 12.83 | +0.39 | 1.50 |
| `q_softmax` | **34.12** | **5.43** | **+2.07** | 10.85 |
| `visit_q_blend` τ=0.25 | 33.06 | 2.48* | +1.01 | 18.04 |
| `visit_q_blend` τ=0.5 | 26.29 | 1.56* | +0.84 | 3.15 |
| `visit_q_blend` τ=1.0 | 26.50 | 5.25* | +1.17 | 0.08 |
| `visit_q_blend` τ=2.0 | 24.29 | 2.35* | +0.64 | 0.07 |

`*` **Do not trust the blend NashConv column.** 3 of 8 cells came back exactly
0.00 while no wave-1 cell did, and τ=1.0's two seeds have near-identical returns
(26.5, 26.5) but NashConv 10.51 vs 0.00 — that is best-response variance, not a
policy property. A 60k-budget re-run was launched and **cancelled before
finishing**; if g1 exploitability ever matters for a claim, re-measure at
matched budget first (`results/analysis/v6_reporting_protocol.md`, "The BR
budget is not a detail").

**Result:** `q_softmax` is the best target found. The blend interpolates
monotonically from `q_softmax` (τ=0.25) to `visit` (τ=2.0) with **no interior
optimum** — every unit of visit information costs return — and it fails its own
purpose: the only τ retaining the gain has *worse* seed spread than `q_softmax`
(18.04 vs 10.85). Not a bug: `tests/algo/test_policy_target_blend.py` proves
both endpoints exactly. The idea is simply wrong for this problem.

**Caveat carried forward:** every number above is **n=2**. The 4-seed extension
of the 2×2 was launched and stopped ~40 min in, so `visit` vs `q_softmax` is
*not* resolved at n=4. `q_softmax` leads on all three instruments (return, the
last-20% trace, g1 NashConv) but its own seed spread is comparable to its lead.

## The structural fact that motivates per-agent MCTS

`hyper_mve/algo/mazero_mixed/core/mcts/ctree/ctree_sampled/lib/cnode.h:19-26`:

```cpp
int visit_count, num_children, hidden_state_index_x, agent_num;  // ONE scalar — JOINT
std::vector<float> reward_vec;                    // (agent_num,) — PER-AGENT
std::vector<float> pred_value_vec;                // (agent_num,) — PER-AGENT
std::vector<tools::SubTreeValueSet> subtree_info; // one per agent — PER-AGENT
std::vector<std::vector<int>> children_action;    // (num_children, num_agents) — JOINT
```

**Values are per-agent; visits are joint.** That one asymmetry explains the whole
investigation: the `visit` target cannot express per-agent credit assignment
(one scalar per child cannot carry two signs), which is why `q_softmax` — the
only per-agent target — is the only one that lifts the disadvantaged agent in
the asymmetric regimes g2/g3.

### Most of the machinery already exists and is unused
<a id="most-of-the-machinery-already-exists-and-is-unused"></a>

> **SUPERSEDED — see the box at the top of this file.** Point 1 below is the
> no-op. Points 2-3 (`select_child_decoupled`, `--decoupled_selection`) remain
> accurate, and note that the method of record already passes
> `--decoupled_selection`, so *selection* is already per-agent; only the target
> was not.

1. **`get_marginal_visit_count`** (`cnode.cpp:95-103`) marginalises joint child
   visits onto per-agent action slots:
   `logits(j, children_action[i][j]) += child->visit_count`. Exported through
   `cytree.pyx:105` → `SearchOutput.marginal_visit_count`, shape
   `(batch, num_agents, action_space_size)`.
2. **`select_child_decoupled`** (`cnode.cpp:519+`) already computes per-agent
   marginal visits *and* visit-weighted Q:
   `vis[a] += child->visit_count; wq[a] += child->visit_count * child->get_qsa(i, discount)`.
   Used for **selection only**.
3. **`--decoupled_selection`** (`core/config.py:208`) already flags it
   (`select_mode` 0 = joint team-UCB, 1 = decoupled per-agent UCB), and
   `arms.py` already has arms that pass it. There is also per-agent q
   normalisation (`minmax_agent`, `cnode.h:90`).

**What is missing is only the consumption side.** The training target reads
*joint* per-child visits — `reanalyze_worker.py:533-534`,
`batch_sampled_policies = batch_sampled_visit_counts / num_simulations` — and
`marginal_visit_count` is used in exactly one place, `runner.py:562`, a
diagnostic labelled *"what the search actually wanted, before argmax."*

So the first per-agent MCTS step is plumbing, not a C++ rewrite.

**Prerequisite before using marginals as a target:** `mcts_sampled.py:213` has
its invariant **commented out** —
`# assert np.all(np.sum(roots_marginal_visit_count, axis=-1) == num_simulations)`.
Root-cover forces the first `num_children` sims to visit each child once
(`cnode.cpp:627-628`), so the marginals may not sum to a clean constant. Check
before normalising.

## On the two points raised

### 1. "Per-agent MCTS requires an opponent model"

Correct in general, but note what `select_child_decoupled` already does: each
agent selects from its **own marginal statistics** in the *shared* tree, and the
joint tuple of per-agent choices is then mapped onto the sampled children
(`cnode.cpp:592`). That is decoupled UCT — the opponent model is *implicit* and
comes from the opponent's own visit/Q marginals accumulated in the same tree. So
a separate opponent network may not be needed for a first version; the question
is whether the implicit model is good enough, which `--decoupled_selection` can
answer empirically as an arm.

Worth knowing: it also changes the budget arithmetic, which is currently brutal.
At `num_simulations=25` the joint space is `A^N = 36` and the sampled root holds
5 children (`root_cover=none`) or 13 (`star`) — i.e. **5.0 or 1.9 visits per
child**. Per-agent statistics pool over `A·N = 12` slots instead, so the same
budget buys far more evidence per decision unit, and it scales linearly rather
than exponentially in N. The 1.9 figure is not academic: it is the best
available explanation for why the `star` arms are the least stable cells
measured (g1 NashConv seed spread 18.98 vs 1.06).

There is existing opponent-modelling code to draw on: `hyper_mve/comparison/mbom.py`
(MBOM / MBOM-oracle baselines).

### 2. "Is freezing the belief at the root still right?"

This is the sharper of the two questions, and the answer is probably **no — but
only on v6**.

The planner is QMDP: `runner.py` calls `model.set_belief(g_hat)` once per real
step and the belief is then **held constant for the entire search**. A QMDP
planner cannot value information — it implicitly assumes the regime becomes known
after one step, so information-gathering actions get no credit. That is the
documented root cause of the settled negative result that the belief posterior
contributes ~nothing (memory: `hyper_mve_bayes_averaging_negative_result`; three
independent measurements, plus `mamba_pm` 74.57 vs ours 61.94).

The reasoning that "the regime is fixed within an episode, so freezing is fine"
is half right and the half that fails is the important one. Because the regime is
constant, the belief changes **only through observation** — and no new
observations arrive inside the tree. So freezing is *self-consistent*. What it
destroys is the value of the agent's **future** observations: a correct planner
would account for the posterior sharpening over the episode, which is exactly
what makes a probing action worth taking.

Why this now matters when it previously did not: on v5 VoI was **0.00**, so no
amount of belief machinery could have helped — the ceiling was zero. v6 was built
specifically to break that, and its VoI is **3.99**. The frozen-root assumption
was therefore untestable before and is testable now. It is a live candidate for
the binding constraint.

Two cheap things before building anything: (a) measure the belief's **settle
time** over an episode — how many steps until the posterior is sharp — since if
it settles in 2-3 steps of a 100-step episode the recoverable VoI is small; and
(b) note that `head_regime`'s gradient path had a latent test bug
(`test_head_regime_gradient_flow` was backwarding a constant), so verify the
belief net actually trains before concluding it is inadequate.

## How to score anything on v6 (do not skip)

`return_mean` is **not** the headline — it is structurally blind in 3 of 5
regimes. Full protocol and evidence in
**`results/analysis/v6_reporting_protocol.md`**. Summary:

| regimes | metric | source |
|---|---|---|
| g0, g4 | team return | `eval_report.json` → `return_per_regime` |
| g2, g3 | **per-agent** return | `eval_diagnostics.json` → `return_per_regime_planner_per_agent` |
| g1 | **NashConv only** (return is pinned at `−ε·moves`) | `scripts/eval_game_metrics.py` |

Also: run evals at **128 episodes** (SEM 3.54 vs 10.27 at 16, for 1.8× the time);
select on the **last-20%** of the run's own periodic eval, not the final
checkpoint (single-point sd ≈ 8-12); lr **0.02** (swept — 0.04 SIGFPEs in the
C++ tree); and never put act_fn-driven numbers in the same table as
`eval_report` numbers.

## Instruments available

- `scripts/probes/blend_tau_probe.py` — per-root log-visit span vs *normalised*
  advantage span (measure in the units the loss uses; the raw span is ~2× off).
- `scripts/probes/adv_scale_probe.py` — per-regime advantage distortion. The
  `--policy_target_min_qstd` guard arm was checked and is **not** motivated:
  v5's 0.29 does not reproduce on v6 (0.576-0.633).
- `scripts/probes/cross_play_probe.py` — cycling/intransitivity (gates NFSP).
- `scripts/probes/regime_voi_probe.py` — VoI, with NOOP abstention.
- `scripts/eval_game_metrics.py --frozen-mode {prior,planner,both}` — NashConv.

## Repo state

Branch `v5-relation-refactor`, uncommitted. Added this session: the
`visit_q_blend` target (`core/train.py`) + its config choice, 4 sweep arms,
`tests/algo/test_policy_target_blend.py` (7 tests), `blend_tau_probe.py`, a fix
to the stale frozen-policy provenance note in `game_metrics.py`, and protocol-doc
sections. Full suite was 211 passed / 1 skipped with the arm-lock test updated.
Results live in `results_v6_2x2/` (wave 1, n=2, complete) and `results_v6_blend/`
(τ sweep, n=2, complete).
