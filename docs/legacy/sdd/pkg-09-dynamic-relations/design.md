# Pkg-09 — Dynamic Relationship Regimes (v5)

**Status:** implemented (branch `v5-relation-refactor`, 2026-07).
**Supersedes:** the v4 "emergent relationship" design (c_t × Fehr–Schmidt types).
**Thesis framing:** Incomplete Information Games based on MARL under Dynamic
Role Relationships — research point 1 = relationships resampled per game
(static within), research point 2 = relationships may switch at any step.

---

## 1. Formalization: Dynamic-Relationship Markov Game (DR-MG)

Tuple ⟨N, S, {A_i}, P, {u_i}, G, ρ, κ, p⟩:

- `u_i(s, a)` — individual task utility (own harvest); identical functional
  form for all agents. Physics `P` is **fixed** and independent of the
  relationship state (v5 removed c_t and every regeneration-adjustment
  parameter; regen rate is the constant `EnvConfig.alpha`).
- Regime `g ∈ G` (finite, named); `W(g) ∈ [−1, 1]^{N×N}`, `w_ii = 1`.
  **Agent i's role ≡ its diagonal-free row `w_i·`** — roles differ only
  through the reward.
- **Reward** (replaces Fehr–Schmidt entirely; `schemas/relation.py` is the
  single citable symbol):

      R_i = (u_i + Σ_{j≠i} w_ij·u_j) / (1 + Σ_{j≠i} |w_ij|) − ε·1[moved_i]

  Row-normalized ⇒ `|R_i + ε·moved_i| ≤ max_j u_j` (convex combination), so
  returns are comparable across regimes.
- **Regime dynamics (one kernel, two research points):** at reset
  `g_0 ~ ρ` (uniform default; restricted to `train_regime_ids` when set);
  per step `g_{t+1} = g_t` w.p. `1 − p`, else `~ κ(·|g_t)` (uniform over
  G∖{g}).
  - `p = 0` → **research point 1**: a Bayesian (Harsanyi) Markov game with
    types `θ_i = w_i·` and common prior ρ.
  - `p > 0` → **research point 2**: a hidden Markov-switching game (POSG
    with latent (s, g_t)).
- **Information structure:** agent i privately observes its own row (the
  obs `row` block; on a within-episode switch the next observation already
  carries the new row — "private switch notification"). Others' rows / g
  are hidden; oracle `g_true` exists only as a train-time supervision label.
- **Value decomposition (thesis hook):** for fixed π and fixed W,
  `Q_i^π(s,a) = Σ_j w_ij Q_j^{u,π}(s,a)` — linear (successor-feature-style)
  recombination across regimes; holds piecewise between switches when p>0.
  Tested empirically via the regime-holdout cell (`rel_zero_shot_duo`).

**Positioning honesty:** linear reward mixing is the social-value-orientation
/ reward-interdependence form (well-precedented; verify citations before
thesis writing). The novelty is the *dynamic + hidden* relationship, its
online inference, and decomposition-based generalization — not the mixing.

### Regime families (`schemas/relation.py`)

| family | N | regimes (id: name — W off-diagonals, intensity λ) |
|---|---|---|
| `g2` | 2 | 0: mutual_coop (+λ,+λ) · 1: mutual_comp (−λ,−λ) · 2: asym_exploit (w01=−λ, w10=+λ; agent 0 exploits) · 3: asym_exploited (mirror) · 4: neutral (0,0) |
| `g4` | 4 | 0: all_coop · 1: all_comp · 2–4: the three 2+2 pairings (+λ within block, −λ across) |
| `g4_ext` | 4 | g4 + 5–8: the four 3+1 coalitions (+λ within trio, −λ trio↔singleton) |

Default λ = 1.0 (`EnvConfig.relation_intensity`).

---

## 2. v5 API amendment (supersedes Pkg-04 spec 02 §"7-API" lock)

The stable model surface is now **6 APIs**:

    update_step / set_context_subjective /
    encode / transition / predict_reward / predict

Amendment record:

1. **`set_context_objective` deleted.** With c_t removed nothing objective
   varies between contexts. The transition function is a plain weight-shared
   SGD module (`models/transition_net.TransitionNet`) — perspective
   invariance (one physical model for all agents) is carried by weight
   sharing itself; the dual-path architecture's objective half is the shared
   transition, its subjective half the hypernet-generated reward/prediction
   heads. Reintroducing objective conditioning (physics parameters, for
   cross-environment transfer) is a documented future hook, not built.
2. **`set_context_subjective(agent_id, row_i: (B, N−1), belief: (B, |G|))`.**
   The v4 capability slot carries the agent's OWN relationship row (a
   per-episode sample cannot travel via cfg, unlike the v4 static
   type_assignment); the belief tuple `(ĉ, ẑ)` becomes the single regime
   posterior `ĝ_i`. Self-Info discipline: the worker feeds agent i
   `rows[i]` only (row-i-only-for-agent-i), the same information its own
   observation carries.
3. **ctx_aug 80 → 64:** `[role (0:32) | belief (32:64)]`, role = id_emb(8) +
   row_emb(24). Belief-grad-gating slice is (32, 64).
4. **Planner θ-cache** is subjective-only (`install_subjective_theta` /
   `current_subjective_thetas`); the objective install/current helpers are
   gone (a plain transition module is batch-agnostic).
5. **p > 0 approximation:** within an imagined K-step rollout the regime is
   frozen at the current belief — the world model does not simulate
   switches. Exact under p = 0; biased over horizons ≳ 1/p under p > 0
   (documented trade-off; real-step belief updates handle switches between
   plans).

Signature locks: `tests/migration/test_pkg05_v47_to_v4_set_context.py`.

---

## 3. Environment (`envs/relation_commons/`, info schema v5.0)

Fixed physics: uniform no-replacement resource spawn, constant-α logistic
regen `q ← clip(q − h + α(Q_max − q), 0, Q_max)`, deterministic moves,
homogeneous agents (η = 1), fair-share harvest, T_max termination only.

Observation (`RelationObservationLayout`, total = 5 + 3K + 10(N−1)):
`self(4) | resource(3K, no FOV, index order) | neighbor(9(N−1), presence≡1) |
global(1)=[time_remaining] | row(N−1)`.

Info groups: public `(harvests, step_idx)`; oracle `(g_true, rows)` —
trainer supervision only, never model forward; eval-only `(resource_state)`.
Step ordering: reward uses the W in effect during the step; the regime chain
advances after reward, before the post-step observation.

`reset(options={"g": int})` pins a regime and **bypasses**
`train_regime_ids` — that is the holdout-evaluation path.

---

## 4. Evaluation metrics

Primary: **social welfare** = Σ_i physical harvest (`phys_returns`;
regime-comparable because it ignores the subjective mixing). Per-regime
returns + `belief/regime_accuracy` (per regime — own-row aliasing across
regimes makes a global chance level misleading) come from
`training/evaluation.run_eval`.

Two game-theoretic metrics (post-hoc, `eval/game_metrics.py` +
`scripts/eval_game_metrics.py`, schema `game-metrics-v1`):

1. **NashConv / exploitability per regime** — independent double-DQN best
   response per (agent, regime) against the frozen distilled prior;
   `NashConv(g) = Σ_i max(0, V_i(BR_i, π_{−i}) − V_i(π))`. Approximate BR ⇒
   reported as a **lower bound**; state the BR budget (`br_env_steps`).
2. **Empirical Price of Anarchy** — `Eff(g) = W_phys(π, g) / Ŵ*`, Ŵ* from a
   cooperative-optimum reference (e.g. a policy trained on the all-coop
   regime, `--welfare-only` run). Eff may exceed 1 if Ŵ* undershoots —
   report provenance.

Adaptation/dynamic regret is the designated point-2 (p>0) metric — future.

---

## 5. Experiment surface (v5)

Presets: `rel_duo` (N=2, L=8, K=8, T_max=100, g2, p=0, 200K, film_head) and
`rel_duo_holdout` (train_regime_ids = (0, 1, 4)); `rel_quad` (N=4, g4) later.
Cells: `rel_gate_duo` (hyper vs baseline_ma_muzero vs no_belief vs
externals) and `rel_zero_shot_duo` (holdout regimes 2, 3). Variant registry
is 10 keys / 13 CLI entries — `rewardhead_explicit_type` was deleted
(discrete-type reward branching is meaningless under continuous rows);
`no_belief` is the headline "no theory-of-mind under hidden g" ablation.

Old `runs/` results (v4 design) are not comparable — accepted.
