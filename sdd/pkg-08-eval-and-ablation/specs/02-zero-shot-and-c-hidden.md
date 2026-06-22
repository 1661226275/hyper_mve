# Spec 02 — Zero-Shot Generalisation + c_hidden Probe + Regret-vs-Oracle-Ceiling

> **Parent docs**: [`../proposal.md`](../proposal.md) · [`../design.md`](../design.md) §3.2 / §4 D4 / §4 D5 / §4 D6 · [`../README.md`](../README.md) C8-EVAL-ZS1 / C8-EVAL-CHID1 / C8-EVAL-REGRET1 / C8-EVAL-SCHEMA1 / C8-EVAL-SEG1
> **Upstream consumed**: [`./01-unified-evaluator.md`](./01-unified-evaluator.md) §3 (`EvalReport` schema — 32 fields) + §11 (downstream-consumer table) · [`../../pkg-07-baselines/specs/04-external-runner-and-adapter.md`](../../pkg-07-baselines/specs/04-external-runner-and-adapter.md) §3 (N-parametric `ResourceCommonsPettingZooEnv`) + §6 (`reset(options={"c": c})` signature) + §7 (two-flag info gate)
> **Downstream patches declared**: spec 08 §5 (codified) — (i) `hyper_mve/envs/resource_commons/observations.py` +3 lines obs-mask; (ii) `hyper_mve/configs/env_config.py` +1 field `c_visible`.
> **Status**: SDD only — describes the contract for `hyper_mve/eval/zero_shot.py` + `hyper_mve/eval/c_hidden.py` + `hyper_mve/eval/regret.py`. No code changes here.

---

## ⚠️ Header — three hard locks

### Lock 1 — Pkg-02 touchpoint is exactly 3 behavioural lines (+ 1 trivial signature kwarg + 2 env.py call-site threading kwargs; all declared)

This spec produces **the unique behavioural touchpoint** between pkg-08 and pkg-02: a 3-line obs-mask hook in `hyper_mve/envs/resource_commons/observations.py` (in `build_observation`) that zeroes the c-channel slice of the per-agent observation when `env_cfg.c_visible` is `False`. Spec 08 §5 carries the canonical "下游补丁声明" (downstream patches list) which this spec drafts.

**Honest patch accounting (3 declared patches; spec 08 §5 enumerates):**
- **§5.1** `observations.py`: +3 behavioural lines + 1 signature kwarg `env_cfg: EnvConfig | None = None` on `build_observation` (the +3 lines consume the kwarg; the kwarg itself is a parameter-list edit, not behaviour).
- **§5.1b** `env.py`: +2 call-site kwarg additions inside `ResourceCommonsEnv.reset` / `ResourceCommonsEnv.step` (single `env_cfg=self.cfg` threaded into the `build_joint_observation` call). **Zero behaviour change** — the kwarg is a pass-through of the env's own already-held cfg; `test_env_py_call_site_kwarg_no_behaviour_change` (§6) asserts reset/step return-values are byte-identical before vs after the kwarg under `c_visible=True`.
- **§5.2** `env_config.py`: +1 field `c_visible: bool = True` on `EnvConfig` dataclass.

**What is byte-identical** (no edits to any of these surfaces):
- `env.py` lines **308–323** (`_build_info` dict construction) — the info dict / oracle `c_true` field / `types` field are unchanged; trainer-side gating per pkg-07 spec 04 §7 (the `oracle_mode` flag on the adapter) still owns the info-dict-side oracle stripping.
- `spaces.py` — obs shape is identity-preserving (masking writes zeros into existing slots, not removing channels).
- All pkg-02 SDD files — zero edits. The contract change is a **value-domain** restriction at the leaf function (`build_observation` / `_global_block`), not a **shape-domain** change.
- `c_t` semantics in `ResourceCommonsState` — unchanged.

The downstream-consumer responsibility split is locked: pkg-02 owns the obs construction; pkg-08 owns the value-domain restriction at the leaf function; spec 08 declares all three patches and the responsibility boundary so any drift triggers reviewer reconciliation.

### Lock 2 — Zero-shot grid is config-driven, never hardcoded

The three reserved fields on `cfg.eval` (`zero_shot_train_c`, `zero_shot_test_c`, `zero_shot_unseen_c`) — already present at `hyper_mve/configs/eval_config.py` lines 37–39 — are the **single source of truth** for the zero-shot c-grid. This spec **MUST NOT** hardcode `(0.2, 0.5, 0.8)` / `(0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)` / `(0.0, 0.35, 0.65, 1.0)` anywhere in `eval/zero_shot.py`; the implementation reads from `cfg.eval.zero_shot_*` and respects whatever override the sweep harness (spec 05) injects. Tests in §6 assert the disjoint-union invariant (`set(zero_shot_test_c) == set(zero_shot_train_c) ∪ set(zero_shot_unseen_c)` and `set(zero_shot_train_c) ∩ set(zero_shot_unseen_c) == ∅`) regardless of which grids the caller selects, so cherry-picked grids that violate the train/unseen partition fail at evaluator entry rather than silently producing meaningless seen-vs-unseen gaps.

### Lock 3 — Regret requires either oracle ceiling cache hit OR explicit sweep harness re-dispatch

Regret is computed only when the oracle ceiling cache at `runs/_oracle_ceilings/<config_hash>/<c>.json` already contains a record for **every** c in `cfg.eval.zero_shot_test_c`. On any cache miss, the unified evaluator (spec 01) **does not block** and **does not synchronously launch** an oracle ceiling computation inside its own process — instead it returns an `EvalReport` with `oracle_ceiling_cache_hit=False`, `regret_per_c={c: None for c in zero_shot_test_c}` semantically (mapped to a sentinel `float("nan")` at JSON-serialisation time because the schema field type is `Mapping[float, float]`, not `Mapping[float, float | None]`), `regret_mean=float("nan")`, and `oracle_ceiling_per_c={}` (empty mapping). The cache-miss path then defers to spec 05 sweep harness, which detects the miss (post-hoc, by inspecting `oracle_ceiling_cache_hit` on the returned report) and **dispatches a new `oracle_only` sweep row** for the missing (config_hash, c) tuple. On the next eval pass after the oracle row completes and writes the ceiling JSON, regret is computed. This **never-block** policy is non-negotiable; tests in §6 assert that cache miss returns rather than raising.

---

## 1. Purpose

This spec exists to deliver three of the four headline numbers Ch6 needs to support the paper's main claims, all of which are unique to pkg-08 (none of them lives in `training/evaluation.py:run_eval`):

1. **Zero-shot generalisation gap (Ch6.9 thesis number).** The single scalar that answers "does the hypernet learn a representation that transfers to c values it never saw at training time?". Implemented as `return_zero_shot_seen - return_zero_shot_unseen`, where seen is the mean over `cfg.eval.zero_shot_train_c` (default 3 train c values: 0.2, 0.5, 0.8) and unseen is the mean over `cfg.eval.zero_shot_unseen_c` (default 4 unseen c values: 0.0, 0.35, 0.65, 1.0). A small or negative gap is the assertion the paper makes; a large positive gap falsifies the claim. The headline scalar lives in `EvalReport.return_zero_shot_gap` (spec 01 §3.1, schema slot 11).
2. **c_hidden BeliefNet quality probe (Theory Audit Q7).** The diagnostic that answers "when c is removed from the observation, does the BeliefNet's ĉ head reconstruct the true c well enough to substitute?". Implemented as two paired runs (one with `cfg.env.c_visible=True`, one with `cfg.env.c_visible=False`) and a comparison of `return_mean`. The c_hidden run additionally populates `belief_c_mae` and `belief_c_calibration` for hyper-variant runs; non-hyper variants leave those fields `None` (spec 01 §3.1 schema slots 30–31). A small return gap between `c_visible` modes corroborates BeliefNet quality; a large gap is the falsifier that drops the BeliefNet-quality claim from the paper.
3. **Regret-vs-oracle-ceiling distance (Ch6.7 Assertion D).** The bounded performance distance from the method to the oracle's perfect-information ceiling, computed per c and aggregated. `regret(c) = oracle_ceiling(c) - method_return(c)`, where `oracle_ceiling(c)` is the cached mean return of the `oracle_only` variant (the `HyperMuZeroModel` constructed with curriculum stage-1-end set to 1.0 so the model receives the true c throughout training) at that c. The per-c regret + the mean across c live in `EvalReport.regret_per_c` and `EvalReport.regret_mean` (spec 01 §3.1 schema slots 21–22); the cached ceiling values and cache-hit flag are exported as diagnostic slots 23–24 so reviewers can audit whether regret was computed from a fresh ceiling or a stale one.

The deliverable is **three new files** plus declarations for two downstream patches consumed elsewhere:

```
hyper_mve/eval/zero_shot.py     — train/test split + aggregation into EvalReport headline scalars
hyper_mve/eval/c_hidden.py      — paired-run protocol gated on cfg.env.c_visible
hyper_mve/eval/regret.py        — oracle ceiling cache + regret(c) computation
```

The downstream patches (declared in §5, codified in spec 08 §5):
- `hyper_mve/envs/resource_commons/observations.py` — 3 lines (behavioural).
- `hyper_mve/configs/env_config.py` — 1 field (config addition, no behaviour change on its own).

---

## 2. Zero-shot protocol

### 2.1 The three reserved cfg fields (already present, this spec consumes them verbatim)

`hyper_mve/configs/eval_config.py` lines 37–39 already define:

```python
zero_shot_train_c: tuple[float, ...] = (0.2, 0.5, 0.8)
zero_shot_test_c: tuple[float, ...] = (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)
zero_shot_unseen_c: tuple[float, ...] = (0.0, 0.35, 0.65, 1.0)
```

This spec **does not add** any new field to `EvalConfig` for zero-shot. The defaults above (3-value train, 7-value test, 4-value unseen) match Theory Audit §10.3 + Ch6.9 and are the locked defaults for the main table. The sweep harness (spec 05) may override these for the appendix sweeps; the only invariants the unified evaluator enforces at entry are the disjoint-union properties listed in §2.6.

### 2.2 Train grid

`cfg.eval.zero_shot_train_c` defaults to `(0.2, 0.5, 0.8)` — the three c values the model was actually trained on under the standard `c_mode="static"` preset cycled across rollouts. The unified evaluator does **not** verify that the trained model was actually trained on these c values (because the trainer's c distribution is `c_mode`-dependent and the evaluator has no introspection into trainer history); this is a documentation invariant locked at spec 06 (curriculum cells) + Ch6.9 prose. Tests at §6 assert only the partition properties, not the trainer-history match.

### 2.3 Test grid (the 7-point evaluation grid)

`cfg.eval.zero_shot_test_c` defaults to `(0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)` — 7 values forming `train ∪ unseen`. This is **the same grid** the unified evaluator (spec 01 §3) uses to populate `EvalReport.return_per_c`, so every eval call materialises exactly 7 per-c means and SEMs. The 7-point grid is small enough to recompute on every `evaluate()` call without budget concerns (7 × 30 episodes = 210 episodes per mode; with 4 planner modes total under the default `planner_full` single-mode dispatch this is 210 episodes per evaluation; spec 03 §3 owns the mode budget table). No caching of return_per_c is done; spec 02 treats the cost as acceptable and recomputes on every call.

### 2.4 Unseen grid (the 4-point held-out grid)

`cfg.eval.zero_shot_unseen_c` defaults to `(0.0, 0.35, 0.65, 1.0)` — 4 values not in the train grid. The unseen-grid mean is the falsification target for zero-shot transfer claims. Specifically:

- c = 0.0 and c = 1.0 are **extrapolation** points (outside the convex hull of train).
- c = 0.35 and c = 0.65 are **interpolation** points (inside train hull).

The unified evaluator does not currently split these two sub-categories (the paper treats unseen as a single average); a future spec may extend `EvalReport` with separate `return_zero_shot_extrap` / `return_zero_shot_interp` slots if Ch6.9 prose demands it. For the present spec the mean over all 4 unseen c values is the locked headline.

### 2.5 EvalReport population (the four zero-shot fields)

Per spec 01 §3.1 schema slots 9–11 + `return_per_c` (slot 12), the unified evaluator populates four headline + 1 mapping field directly from the zero-shot test grid:

```python
# Pseudocode, lives in hyper_mve/eval/zero_shot.py
def populate_zero_shot_fields(
    return_per_c: Mapping[float, float],     # 7 c-vals from spec 01 §8.1
    episodes_per_c: Mapping[float, int],     # 7 c-vals from spec 01 §8.1
    cfg: V4Config,
) -> tuple[float, float, float]:
    """Returns (seen_mean, unseen_mean, gap)."""
    train_grid = set(cfg.eval.zero_shot_train_c)
    unseen_grid = set(cfg.eval.zero_shot_unseen_c)
    seen_returns = [return_per_c[c] for c in return_per_c if c in train_grid]
    unseen_returns = [return_per_c[c] for c in return_per_c if c in unseen_grid]
    seen_mean = float(np.mean(seen_returns))
    unseen_mean = float(np.mean(unseen_returns))
    gap = seen_mean - unseen_mean
    return seen_mean, unseen_mean, gap
```

Episode-count weighting is **not** applied — the headline is the unweighted mean over c, consistent with Ch6.9 prose. If a future preset assigns asymmetric episode budgets per c, the segment-aggregation logic in spec 01 §8.2 (episode-count-weighted) remains the per-segment path and zero-shot remains the unweighted-by-c path. The two paths are intentionally different and documented as such.

### 2.6 Disjoint-union invariant (the test contract)

The unified evaluator asserts at entry (before any eval rollout):

```python
def _verify_zero_shot_partition(cfg: V4Config) -> None:
    train = set(cfg.eval.zero_shot_train_c)
    unseen = set(cfg.eval.zero_shot_unseen_c)
    test = set(cfg.eval.zero_shot_test_c)
    if train & unseen:
        raise RuntimeError(
            f"zero_shot_train_c and zero_shot_unseen_c must be disjoint; "
            f"intersection = {train & unseen}"
        )
    if train | unseen != test:
        raise RuntimeError(
            f"zero_shot_test_c must equal train ∪ unseen; "
            f"got test = {test}, train ∪ unseen = {train | unseen}, "
            f"diff = {test ^ (train | unseen)}"
        )
```

A named test `test_zero_shot_grid_disjoint_union` (§6) parametrises over the default grids and over a corner-case override (e.g. train = `(0.5,)`, unseen = `(0.0, 1.0)`, test = `(0.0, 0.5, 1.0)`) asserting both branches of the invariant. A second test `test_zero_shot_grid_rejects_overlap` constructs a deliberately-overlapping config (train = `(0.5,)`, unseen = `(0.5,)`) and asserts the `RuntimeError` is raised. The invariant is checked at evaluator entry, not at config-load time, because `EvalConfig` is `frozen=True` and `__post_init__` would couple it to pkg-08 semantics — keeping the check in the evaluator preserves config-layer locality.

---

## 3. c_hidden mode (BeliefNet c-inference probe)

### 3.1 The new cfg field — `cfg.env.c_visible: bool = True`

This spec declares a **new field on `EnvConfig`**:

```python
# hyper_mve/configs/env_config.py — DOWNSTREAM PATCH (1 line added)
c_visible: bool = True
```

Default `True` preserves all existing behaviour (no current run is affected). The downstream patch is declared here, codified in spec 08 §5 as part of the "config-additions block" — it is a **config addition**, not a behavioural change to env.py / observations.py on its own. The behavioural change comes from how `observations.py` interprets the new field (§3.2). Pkg-01 spec 05 synchronously registers `c_visible` in the consumption table at the same time spec 08 is finalised so that the field is discoverable from the canonical V4Config schema view.

### 3.2 The 3-line obs-mask hook (the unique pkg-02 behavioural patch)

Target file: `hyper_mve/envs/resource_commons/observations.py`. The current `_global_block` (lines 156–162) emits a 2-dim block `[c_t, time_remaining_ratio]`. Per `ObservationLayout` (`hyper_mve/schemas/observation.py`):

- `ObservationLayout.GLOBAL_DIM = 2` — the global block dim.
- Block order = `("self", "resource", "neighbor", "global", "capability", "type")`.
- `ObservationLayout.block_offset("global", N, K)` returns `(start, end)` for the global block in the flattened observation.
- Within the global block, **slot 0 is `c_t`** (the c-channel) and slot 1 is `time_remaining_ratio`. The c-channel index within the global block is therefore `0`; the absolute index within the flattened observation is `ObservationLayout.block_offset("global", N, K)[0] + 0`.

The 3-line behavioural patch lives at the end of `build_observation` (between the existing `obs = np.concatenate(blocks).astype(...)` line and the existing shape assertion):

```python
# Inside build_observation, after `obs = np.concatenate(blocks)...`:
if not getattr(env_cfg, "c_visible", True):                                # +1 (guard via getattr for backward-compat with older configs)
    c_start, _ = ObservationLayout.block_offset("global", N, K)            # +2
    obs[c_start] = 0.0                                                     # +3 (zero the c-channel slot only)
```

Implementation notes (locked in this spec, codified in spec 08 §5):

- **The patch reads `env_cfg.c_visible`, not a globally-injected flag.** The `build_observation` signature must therefore accept `env_cfg` (or be changed to do so) — but `build_observation` currently takes only `(state, agent_id, L, T_max, K)`. To avoid changing the call signature (which would propagate to every callsite in env.py and tests), the patch hooks `c_visible` via the `EnvConfig` instance stored on `ResourceCommonsEnv.cfg` and threaded as a new optional kwarg `env_cfg: EnvConfig | None = None` to `build_observation`. Spec 08 §5 documents the threading; the patch line count remains 3 in `build_observation` and the call-site update in `ResourceCommonsEnv.reset` / `.step` (one kwarg added at each of the two `build_joint_observation` callsites — counted as 2 trivial call-site updates in spec 08, not as behaviour changes).
- **Channel index uses `ObservationLayout` constant**, never a magic number. The block-offset computation is a pure function of `(N, K)` and is cheap; recomputing on every `build_observation` call is acceptable.
- **The c-channel index is the constant `ObservationLayout.block_offset("global", N, K)[0]`** (start of the global block). Spec 08 §5 cites this constant by name so any future `ObservationLayout` reordering automatically updates the patch.

### 3.3 c_hidden evaluation protocol

The c_hidden probe is a **paired-run protocol**: the same trained model is evaluated twice, once with `cfg.env.c_visible=True` (default) and once with `cfg.env.c_visible=False`. The paired evaluation produces two `EvalReport`s; the quality of the BeliefNet's ĉ inference is measured as:

- `return_gap_c_hidden = report_visible.return_mean - report_hidden.return_mean` (smaller = better BeliefNet).
- `belief_c_mae` on `report_hidden` (the c_hidden run; hyper-only).
- `belief_c_calibration` on `report_hidden` (the c_hidden run; hyper-only).

The unified evaluator (spec 01) does **not** automatically run the paired protocol on every call — it runs one `evaluate()` per `cfg` and the caller decides whether to flip `cfg.env.c_visible`. Sweep harness (spec 05) is the canonical caller that materialises the pair: spec 05 emits two rows in `runs/registry.jsonl` for each c_hidden ablation (one with `c_visible=True`, one with `c_visible=False`) and the stats layer (spec 07) consumes the pair to compute the headline gap.

### 3.4 EvalReport.c_visible population

Per spec 01 §3.1 schema slot 6, `EvalReport.c_visible: bool` is populated **at evaluator entry** from `cfg.env.c_visible`. The unified evaluator reads the flag from the config (not from the env instance) because the config is the source of truth and the env may not yet be constructed at the moment the report identity bucket is being filled. Tests at §6 assert this populates correctly for both `c_visible=True` and `c_visible=False` configs.

### 3.5 Test contract (parametrised over Easy + Medium presets)

A named test `test_c_hidden_obs_mask` (§6) parametrises over:

- The Easy preset (`N=2`, `K=8`).
- The Medium preset (`N=4`, `K=20`).

and asserts, for each preset:

1. `cfg.env.c_visible=True` → the c-channel slice (`obs[block_offset("global", N, K)[0]]`) equals `state.c_t` (the true context).
2. `cfg.env.c_visible=False` → the c-channel slice equals `0.0` exactly.
3. The remaining 1 slot of the global block (`time_remaining_ratio` at index `block_offset[0] + 1`) is **unchanged** between the two configs (mask must not over-reach).
4. The total obs dimension `ObservationLayout.total_dim(N, K)` is the same between the two configs (no shape change).

Parametrising over Easy + Medium verifies that the c-channel index correctly depends on `(N, K)` (via `block_offset`) and that the mask is N-parametric — not hardcoded for `N=4`. This is the key correctness gate for the 3-line patch, because if the channel index were wrong by even 1 slot the model would silently train on zeroed `time_remaining_ratio` instead of zeroed `c_t`.

---

## 4. Regret metric + oracle ceiling cache

### 4.1 The regret formula

```
regret(c) = oracle_ceiling(c) - method_return(c)
```

Both terms are mean episode returns (scalars) at the same c value. `method_return(c)` is `EvalReport.return_per_c[c]` for the variant being evaluated. `oracle_ceiling(c)` is the cached ceiling for the same `c` and the same `config_hash` (§4.3). Units = mean return (the same scalar the rest of `EvalReport.return_per_c` carries). The mean across c is the simple unweighted mean `regret_mean = mean({regret(c) : c in cfg.eval.zero_shot_test_c})` — same convention as zero-shot mean (§2.5), not the episode-count-weighted segment mean (§ spec 01 §8.2).

### 4.2 Oracle ceiling computation (one-time, cached)

Per (`config_hash`, `c`) tuple, the ceiling is computed exactly once via the `oracle_only` variant defined at pkg-07 spec 01 §2 (the `HyperMuZeroModel` constructed with curriculum stage-1-end forced to 1.0 throughout training, so the model receives the true c throughout). The compute protocol:

- **5 seeds** (default; sourced from a hard-coded list `(0, 1, 2, 3, 4)` in the sweep harness, not from cfg, because seeds are run-domain not config-domain).
- **`cfg.eval.evaluate_episodes` episodes per seed** (default 30 from `eval_config.py:19`).
- For each (seed, c) pair: train an `oracle_only` model, then evaluate at this single c value.
- The ceiling is the mean of `5 × 30 = 150` episode returns (or however many `evaluate_episodes` is set to).
- The ceiling SEM is the standard error of the 5 per-seed means (small-N SEM via the sample std div sqrt(5)).

The cost per ceiling is **one full `oracle_only` training run × 5 seeds**. The cache is therefore designed to amortise this cost across every method evaluation that shares the same `config_hash`.

### 4.3 Cache path schema

Cache root: `runs/_oracle_ceilings/`. The underscore prefix distinguishes the cache from regular `runs/` rows. The per-(config_hash, c) cache JSON path is:

```
runs/_oracle_ceilings/<config_hash>/<c>.json
```

where `<config_hash>` is the 32-character hexadecimal blake2b-16 digest of the canonical V4Config JSON (the same `config_hash` field carried in `EvalReport.config_hash` and `runs/registry.jsonl`; spec 05 §3.5 defines the canonical JSON normalisation). The `<c>` filename component is the c value formatted via `f"{c:.4f}"` (e.g. `0.3500.json`) to disambiguate `0.5` and `0.50000001` — the rounding granularity matches the c-grid resolution and avoids float-equality cache misses on round-trips.

### 4.4 Cache JSON schema (verbatim field list)

```json
{
  "c": 0.35,
  "config_hash": "abc123...",
  "ceiling_mean": 12.34,
  "ceiling_sem": 0.45,
  "n_seeds": 5,
  "n_episodes_per_seed": 30,
  "computed_at_iso8601": "2026-06-19T10:23:00Z",
  "git_sha": "0d0f955abcdef1234567890abcdef1234567890ab",   # 40-char full SHA — locked length (NOT short-SHA; spec 02 §4.4 contract)
  "method": "oracle_only"
}
```

Field-by-field semantics:

- `c: float` — the c value the ceiling was computed at. Must be in `[0.0, 1.0]`.
- `config_hash: str` — 32-character hex (blake2b-16). Must match the directory name.
- `ceiling_mean: float` — mean of (n_seeds × n_episodes_per_seed) episode returns.
- `ceiling_sem: float` — SEM of the 5 per-seed means (small-N SEM with denominator `sqrt(5)`).
- `n_seeds: int` — must equal 5 under the default protocol; a future spec may bump this.
- `n_episodes_per_seed: int` — sourced from `cfg.eval.evaluate_episodes`.
- `computed_at_iso8601: str` — ISO 8601 timestamp with UTC `Z` suffix.
- `git_sha: str` — repo HEAD at the time of computation, for reproducibility audit. **Locked length: 40-char full SHA** (NOT short-SHA; matches `git rev-parse HEAD` output verbatim so audit-trail tooling in spec 08 can regex `^[0-9a-f]{40}$`).
- `method: Literal["oracle_only"]` — sentinel; only `"oracle_only"` is currently legal. A future spec may extend with `"oracle_and_belief"` etc.

A test `test_oracle_ceiling_cache_schema` (§6) round-trips a JSON file through `json.dump` + `json.load` and asserts every field is present with the correct type.

### 4.5 Cache invalidation

**Under normal operation: NONE.** The cache is treated as content-addressed (the `config_hash` is the key; the cache contents are immutable). If the user wants to invalidate, the procedure is **manual file deletion** — e.g. `rm runs/_oracle_ceilings/<config_hash>/<c>.json` or `rm -rf runs/_oracle_ceilings/<config_hash>/` to wipe all c values for a config. Spec 02 explicitly does **not** define an "invalidate by age" or "invalidate by git_sha mismatch" policy because (i) the ceiling is the platonic perfect-information return at that c, which depends only on the env preset (which is captured in `config_hash`), and (ii) the `git_sha` field exists for reproducibility audit, not for invalidation.

On manual deletion, the next eval pass that needs a ceiling for that (config_hash, c) sees a cache miss and falls through to §4.7 (the cache-miss path, which dispatches an `oracle_only` re-computation via spec 05). The eval pass itself does not raise.

### 4.6 Cache lookup (the unified evaluator's hot path)

When the unified evaluator (spec 01) reaches the regret-computation step, it does **a cache hit-check before any compute**:

```python
# Pseudocode in hyper_mve/eval/regret.py
def lookup_ceilings(
    config_hash: str,
    c_grid: tuple[float, ...],
    cache_root: Path = Path("runs/_oracle_ceilings"),
) -> tuple[dict[float, float], dict[float, bool]]:
    """Returns (ceiling_per_c, cache_hit_per_c)."""
    ceilings: dict[float, float] = {}
    hits: dict[float, bool] = {}
    for c in c_grid:
        cache_file = cache_root / config_hash / f"{c:.4f}.json"
        if cache_file.exists():
            with cache_file.open() as fh:
                record = json.load(fh)
            ceilings[c] = float(record["ceiling_mean"])
            hits[c] = True
        else:
            ceilings[c] = float("nan")    # placeholder
            hits[c] = False
    return ceilings, hits

def compute_regret(
    report: EvalReport,
    cfg: V4Config,
) -> EvalReport:
    """Returns a new EvalReport with regret_* fields populated.
    
    Precondition: report.return_per_c.keys() is a SUPERSET of cfg.eval.zero_shot_test_c.
    (Per spec 01 §3 slot 12: return_per_c is keyed by the 7-c-val zero_shot_test_c grid;
     this is the canonical aggregation contract — any partial-c evaluator MUST first
     re-fill to the full zero_shot_test_c grid before regret can be computed.)
    """
    missing = set(cfg.eval.zero_shot_test_c) - set(report.return_per_c.keys())
    if missing:
        raise ValueError(
            f"compute_regret: return_per_c is missing c-values {missing}. "
            "Precondition violated — re-run evaluator on the full zero_shot_test_c grid "
            "before computing regret (spec 02 §4.6 + spec 01 §3 slot 12)."
        )
    ceilings, hits = lookup_ceilings(report.config_hash, cfg.eval.zero_shot_test_c)
    all_hit = all(hits.values())
    if not all_hit:
        # Cache miss: populate sentinel NaN regret + False hit flag; sweep harness will re-dispatch.
        return dataclasses.replace(
            report,
            regret_per_c=MappingProxyType({c: float("nan") for c in cfg.eval.zero_shot_test_c}),
            regret_mean=float("nan"),
            oracle_ceiling_per_c=MappingProxyType(ceilings),    # carries NaN for missing c
            oracle_ceiling_cache_hit=MappingProxyType(hits),
        )
    regret_per_c = {c: ceilings[c] - report.return_per_c[c] for c in cfg.eval.zero_shot_test_c}
    regret_mean = float(np.mean(list(regret_per_c.values())))
    return dataclasses.replace(
        report,
        regret_per_c=MappingProxyType(regret_per_c),
        regret_mean=regret_mean,
        oracle_ceiling_per_c=MappingProxyType(ceilings),
        oracle_ceiling_cache_hit=MappingProxyType(hits),
    )
```

The cache-hit flag is a `Mapping[float, bool]` (per spec 01 §3.1 schema slot 24), so reviewers can see per-c which c values had a ceiling and which didn't. The "all hits" condition (`oracle_ceiling_cache_hit` all `True`) is the trigger for `regret_mean` being a real number; a single miss flips `regret_mean` to NaN.

### 4.7 Cache miss: does NOT block evaluate()

Per Lock 3 (top of file): a cache miss does **not** raise, does **not** synchronously launch an `oracle_only` training run inside the unified evaluator's process, and does **not** block the calling sweep harness's iteration. The returned `EvalReport` carries `oracle_ceiling_cache_hit={c: False for c in missing}` and `regret_mean=float("nan")`, and the calling sweep harness (spec 05) inspects these post-hoc and dispatches a new `oracle_only` row for the missing (config_hash, c) tuple. On the **next** eval pass for the same config (which may be in the next sweep invocation, or in the same sweep if the harness re-evaluates after the oracle row completes), the cache is populated and regret is computed.

**Why never-block:** synchronously launching an oracle row from inside the evaluator would deadlock the per-row subprocess isolation that spec 05 §3.5 establishes (each sweep row is its own process; an evaluator launching an `oracle_only` training run from inside that process would require nested process management). Deferring to the harness keeps the process model linear: every training row is a top-level subprocess, every evaluator is a leaf, no nesting.

### 4.8 EvalReport population (regret fields)

Per spec 01 §3.1 schema slots 21–24, the unified evaluator populates four regret-domain fields:

- `regret_per_c: Mapping[float, float]` — keyed by c in `cfg.eval.zero_shot_test_c`; values are `ceiling(c) - return_per_c[c]` on cache hit, `float("nan")` on cache miss. Wrapped in `MappingProxyType`.
- `regret_mean: float` — unweighted mean across c on full cache hit; `float("nan")` on any cache miss.
- `oracle_ceiling_per_c: Mapping[float, float]` — the cached ceiling values (diagnostic; carries the same NaN sentinel on miss). Wrapped in `MappingProxyType`.
- `oracle_ceiling_cache_hit: Mapping[float, bool]` — per-c cache-hit flag. Wrapped in `MappingProxyType`.

The `oracle_ceiling_cache_hit` field exists specifically so the reviewer can audit, for any given main-table cell, whether its regret was computed from a freshly-computed ceiling or a stale one. The audit trail combines this flag with the `computed_at_iso8601` + `git_sha` fields in the cache JSON (§4.4) to reconstruct provenance.

---

## 5. Downstream patches (declared here, codified in spec 08 §5)

This spec produces **three declared patches** (1 behavioural + 1 config-additions + 1 argument-threading; the last has zero behavioural change but must be declared honestly per the SDD "no silent edits" rule).

### 5.1 Behavioural patch — `hyper_mve/envs/resource_commons/observations.py`

- **Size**: +3 lines of behavioural code inside `build_observation` (the obs-mask hook described in §3.2), **plus** +1 line signature extension on `build_observation` to accept `env_cfg: EnvConfig | None = None` (parameter only — the +3 behavioural lines are what consume the kwarg).
- **Behaviour**: when `env_cfg.c_visible is False`, zero the single c-channel slot at `ObservationLayout.block_offset("global", N, K)[0]`. Otherwise leave the observation unmodified.
- **Test gate**: `test_c_hidden_obs_mask` parametrised over Easy + Medium presets (§3.5 + §6).

### 5.1b Argument-threading patch — `hyper_mve/envs/resource_commons/env.py` (NO behaviour change)

- **Size**: **+2 call-site kwarg additions in `env.py`** (`build_joint_observation` calls inside `ResourceCommonsEnv.reset` and `ResourceCommonsEnv.step` — one each — gain a single `env_cfg=self.cfg` kwarg).
- **Behaviour**: **none**. The added kwarg is a pure pass-through of the same `EnvConfig` instance the env already holds; no condition is evaluated and no info dict / step / reset semantics change.
- **Why declared as a separate patch**: under strict reading of the SDD "no silent edits to env.py" rule (README C8-EVAL-CHID1 originally said "3 lines / 1 file"), threading the kwarg is technically an env.py edit and must be enumerated. Spec 08 §5 carries it as patch #3 in the canonical 下游补丁声明 list with the explicit `[no-behaviour]` tag. README is reconciled to "3 declared patches: 4 behavioural lines (observations.py) + 2 trivial threading kwargs (env.py) + 1 config field (env_config.py)".
- **Alternative considered + rejected**: threading via a module-level global (`hyper_mve.envs.resource_commons._current_env_cfg`) was rejected — it introduces hidden global state that defeats the per-env-instance isolation `ResourceCommonsEnv` already provides, and it would re-emerge as drift in any future multi-env sweep code.
- **Test gate**: `test_env_py_call_site_kwarg_no_behaviour_change` asserts the env.py reset/step return-values are byte-identical before vs after the kwarg is threaded (run the env without and with the kwarg under `c_visible=True`; rewards/obs/info must be identical).

### 5.2 Config-additions patch — `hyper_mve/configs/env_config.py`

- **Size**: +1 line adding `c_visible: bool = True` to the `EnvConfig` dataclass field list. This is a config-domain addition with no behavioural consequence on its own (the behaviour comes from §5.1 which reads the new field).
- **Default**: `True` — preserves all existing run behaviour. Existing yaml configs and ckpts that omit `c_visible` resolve to `True` via the dataclass default.
- **Pkg-01 spec 05 sync**: pkg-01 spec 05 synchronously registers `c_visible` in the consumption table at the same time spec 08 §5 is finalised, so the field is discoverable from the canonical V4Config schema view.
- **Test gate**: `test_eval_report_c_visible_populated_from_cfg` (§6) asserts the field round-trips from config to `EvalReport.c_visible`.

### 5.3 Explicit non-changes

The following pkg-02 surfaces are explicitly **NOT** modified by this spec (anything **not** in §5.1 / §5.1b / §5.2 above):

- `hyper_mve/envs/resource_commons/env.py` lines **308–323** (`_build_info` dict construction) — **byte-identical**, no edits to the info dict. Oracle `c_true` and `types` remain in the info dict; trainer-side gating per pkg-07 spec 04 §7 (the `oracle_mode` flag on the adapter) is responsible for stripping them at eval time. Pkg-08 spec 02 does **not** add any `c_visible`-driven filtering to the info dict; the info dict is the trainer's side channel, and `c_visible` governs only the observation side. (The env.py edits in §5.1b are limited to two `build_joint_observation` call-site kwargs in `reset` and `step`; they do not touch `_build_info` or any reward / termination logic.)
- `hyper_mve/envs/resource_commons/spaces.py` — unchanged. Observation shape is identity-preserving (mask zeroes a slot, does not remove channels). The action space is not affected.
- Pkg-02 SDD files — zero edits. The patches are implementation-only, declared in spec 08 §5 of pkg-08, not in pkg-02.

---

## 6. Test contract — C8-EVAL-CHID1 / C8-EVAL-ZS1 / C8-EVAL-REGRET1 enforcement

Located under `tests/eval/`. Each name below corresponds to a specific C8-EVAL constraint (README) and to a section of this spec.

### 6.1 `test_c_hidden_obs_mask` — C8-EVAL-CHID1 (§3.5)

Parametrised over the Easy preset (`N=2`, `K=8`) and the Medium preset (`N=4`, `K=20`). For each preset, two configs are constructed (`c_visible=True` and `c_visible=False`), the env is reset with a known `c_t` value (via `options={"c": 0.5}`), and the assertions are:

```python
@pytest.mark.parametrize("preset", ["easy", "medium"])
def test_c_hidden_obs_mask(preset):
    cfg = make_preset(preset)
    N, K = cfg.env.N, cfg.env.K
    c_start, _ = ObservationLayout.block_offset("global", N, K)
    for c_visible in (True, False):
        env_cfg = dataclasses.replace(cfg.env, c_visible=c_visible)
        env = ResourceCommonsEnv(env_cfg, seed=0)
        obs, info = env.reset(options={"c": 0.5})
        # obs shape unchanged
        assert obs.shape == (N, ObservationLayout.total_dim(N, K))
        # c-channel correctly masked or not
        for i in range(N):
            if c_visible:
                assert obs[i, c_start] == pytest.approx(0.5)
            else:
                assert obs[i, c_start] == 0.0
            # time_remaining_ratio (slot c_start + 1) unchanged regardless of c_visible
            assert 0.0 < obs[i, c_start + 1] <= 1.0
```

### 6.2 `test_zero_shot_grid_disjoint_union` — C8-EVAL-ZS1 (§2.6)

```python
def test_zero_shot_grid_disjoint_union_default():
    cfg = V4Config()
    train = set(cfg.eval.zero_shot_train_c)
    unseen = set(cfg.eval.zero_shot_unseen_c)
    test = set(cfg.eval.zero_shot_test_c)
    assert train & unseen == set()
    assert train | unseen == test

def test_zero_shot_grid_disjoint_union_override():
    cfg = V4Config()
    cfg_eval = dataclasses.replace(
        cfg.eval,
        zero_shot_train_c=(0.5,),
        zero_shot_unseen_c=(0.0, 1.0),
        zero_shot_test_c=(0.0, 0.5, 1.0),
    )
    train = set(cfg_eval.zero_shot_train_c)
    unseen = set(cfg_eval.zero_shot_unseen_c)
    test = set(cfg_eval.zero_shot_test_c)
    assert train & unseen == set()
    assert train | unseen == test
```

### 6.3 `test_zero_shot_grid_rejects_overlap` — C8-EVAL-ZS1 negative path (§2.6)

Constructs a deliberately-overlapping config and asserts the unified evaluator raises `RuntimeError` at entry (before any rollout):

```python
def test_zero_shot_grid_rejects_overlap():
    cfg = V4Config()
    cfg_eval = dataclasses.replace(
        cfg.eval,
        zero_shot_train_c=(0.5,),
        zero_shot_unseen_c=(0.5,),    # overlaps train
        zero_shot_test_c=(0.5,),
    )
    cfg = dataclasses.replace(cfg, eval=cfg_eval)
    runner = make_dummy_baseline(cfg)
    env_fn = lambda: ResourceCommonsPettingZooEnv(cfg.env, oracle_mode=False, eval_info_mode=True)
    with pytest.raises(RuntimeError, match="disjoint"):
        evaluate(runner, env_fn, cfg)
```

### 6.4 `test_regret_oracle_ceiling_formula` — C8-EVAL-REGRET1 (§4.1)

Constructs a mock ceiling cache with known values and asserts the regret formula is applied correctly:

```python
def test_regret_oracle_ceiling_formula(tmp_path):
    cfg = V4Config()
    config_hash = "abc1234567890" * 2 + "abc1234"   # 32 hex
    cache_dir = tmp_path / "_oracle_ceilings" / config_hash
    cache_dir.mkdir(parents=True)
    for c in cfg.eval.zero_shot_test_c:
        cache_file = cache_dir / f"{c:.4f}.json"
        with cache_file.open("w") as fh:
            json.dump({
                "c": c,
                "config_hash": config_hash,
                "ceiling_mean": 20.0,
                "ceiling_sem": 0.5,
                "n_seeds": 5,
                "n_episodes_per_seed": 30,
                "computed_at_iso8601": "2026-06-19T10:00:00Z",
                "git_sha": "0d0f955abcdef1234567890abcdef1234567890ab",  # 40-char full SHA per §4.4 contract
                "method": "oracle_only",
            }, fh)
    report = make_dummy_eval_report(
        config_hash=config_hash,
        return_per_c={c: 15.0 for c in cfg.eval.zero_shot_test_c},
    )
    updated = compute_regret(report, cfg, cache_root=tmp_path / "_oracle_ceilings")
    for c in cfg.eval.zero_shot_test_c:
        assert updated.regret_per_c[c] == pytest.approx(5.0)    # 20 - 15
    assert updated.regret_mean == pytest.approx(5.0)
    assert all(updated.oracle_ceiling_cache_hit.values())
```

### 6.5 `test_regret_cache_hit` — C8-EVAL-REGRET1 (§4.6)

Asserts a second call with the cache pre-populated returns the same regret values without re-reading the cache from a different path (cache lookup is path-deterministic):

```python
def test_regret_cache_hit_returns_cached_values(tmp_path):
    # ... setup cache as in 6.4 ...
    updated1 = compute_regret(report, cfg, cache_root=tmp_path / "_oracle_ceilings")
    updated2 = compute_regret(report, cfg, cache_root=tmp_path / "_oracle_ceilings")
    assert dict(updated1.regret_per_c) == dict(updated2.regret_per_c)
    assert updated1.regret_mean == updated2.regret_mean
```

### 6.6 `test_regret_cache_miss_does_not_block` — C8-EVAL-REGRET1 Lock 3 (§4.7)

```python
def test_regret_cache_miss_returns_evalreport_with_nan(tmp_path):
    cfg = V4Config()
    config_hash = "nonexistent_hash_aaaa" + "0" * 12
    # No cache files written.
    report = make_dummy_eval_report(
        config_hash=config_hash,
        return_per_c={c: 15.0 for c in cfg.eval.zero_shot_test_c},
    )
    updated = compute_regret(report, cfg, cache_root=tmp_path / "_oracle_ceilings")
    # Must NOT raise; must return a valid EvalReport.
    assert isinstance(updated, EvalReport)
    assert math.isnan(updated.regret_mean)
    assert all(not hit for hit in updated.oracle_ceiling_cache_hit.values())
    assert all(math.isnan(v) for v in updated.regret_per_c.values())
```

### 6.7 `test_regret_cache_miss_triggers_oracle_only_dispatch` — Lock 3 + spec 05 hook (§4.7)

Verifies that the sweep harness's post-hoc dispatch path (spec 05) is invoked on cache miss. Implemented as a mock-injection test:

```python
def test_regret_cache_miss_triggers_oracle_only_dispatch(tmp_path, mock_sweep_dispatcher):
    cfg = V4Config()
    config_hash = "miss_hash_" + "0" * 22
    report = make_dummy_eval_report(config_hash=config_hash, return_per_c={...})
    updated = compute_regret(report, cfg, cache_root=tmp_path / "_oracle_ceilings")
    # Spec 05 sweep harness inspects oracle_ceiling_cache_hit and dispatches:
    mock_sweep_dispatcher.observe_eval_report(updated)
    dispatched = mock_sweep_dispatcher.pending_oracle_rows
    assert len(dispatched) == len(cfg.eval.zero_shot_test_c)
    for row in dispatched:
        assert row.variant == "oracle_only"
        assert row.config_hash == config_hash
        assert row.c in cfg.eval.zero_shot_test_c
```

### 6.8 `test_eval_report_c_visible_populated_from_cfg` — §3.4

```python
@pytest.mark.parametrize("c_visible", [True, False])
def test_eval_report_c_visible_populated_from_cfg(c_visible):
    cfg = V4Config()
    cfg = dataclasses.replace(cfg, env=dataclasses.replace(cfg.env, c_visible=c_visible))
    runner = make_dummy_baseline(cfg)
    env_fn = lambda: ResourceCommonsPettingZooEnv(cfg.env, oracle_mode=False, eval_info_mode=True)
    report = evaluate(runner, env_fn, cfg)
    assert report.c_visible is c_visible
```

---

## 7. Integration hooks (cross-spec)

| Consumer | Consumed from this spec | Use |
|---|---|---|
| **spec 01** `unified_evaluator.evaluate` | The three submodules `eval/zero_shot.py`, `eval/c_hidden.py`, `eval/regret.py` — called as inner subroutines of `evaluate()` to populate schema slots 9–11 (zero-shot headline), 6 (c_visible flag) + optional 30–31 (belief diagnostics), 21–24 (regret + ceiling). | Spec 01's `evaluate()` flow at §4.2 calls `populate_zero_shot_fields()` after the per-c aggregation, calls `compute_regret()` after the regret cache lookup, and reads `cfg.env.c_visible` at entry to populate `EvalReport.c_visible`. The unified evaluator does **not** call a c_hidden-specific subroutine — c_hidden is a paired-run protocol orchestrated by sweep harness (spec 05), not by the evaluator. |
| **spec 03** four planner eval mode | Orthogonal — apply per-c independently of zero-shot grid composition. The four modes (`direct_inference` / `planner_no_crn` / `planner_no_coord_desc` / `planner_full`) each populate `return_per_c` over the full 7-value test grid; the zero-shot seen/unseen partition is computed once on the chosen single mode's per-c map. | Spec 03 does not alter the zero-shot partition logic; spec 02 does not alter the mode dispatch logic. |
| **spec 05** sweep harness | (a) Pre-populates the oracle ceiling cache by dispatching `oracle_only` rows for every (config_hash, c) tuple encountered for the first time. The harness is responsible for the post-hoc cache-miss dispatch (Lock 3 / §4.7). (b) Orchestrates the c_hidden paired-run protocol (§3.3): emits two rows per config (one `c_visible=True`, one `c_visible=False`) and the stats layer (spec 07) consumes the pair to compute the headline c_hidden gap. | Spec 05 §3 sweep cartesian includes a "Phase E pre-warm" pass that walks the planned (config_hash, c) tuples and pre-dispatches `oracle_only` rows for any cache miss before the main-table sweep begins, so the bulk of main-table rows hit the cache. |
| **spec 08** integration contracts | (a) Declares the two downstream patches in §5 (one behavioural pkg-02 patch + one config-additions patch). (b) Hosts the canonical "下游补丁声明" list. (c) Hosts the byte-identical copy of `EvalReport` schema slots 6, 9–11, 12, 21–24, 30–31, ensuring this spec's contributions to the schema are locked symmetrically with spec 01. | Spec 08 §5 mirrors §5 of this spec verbatim. Any drift triggers reviewer reconciliation per spec 08 §6 drift detector. |

---

## 8. Cross-references

### Upstream anchors (consumed by this spec)

- **pkg-08 design.md §3.2** — `EvalReport` schema layout (this spec populates slots 6, 9–11, 21–24, 30–31).
- **pkg-08 design.md §4 D4** — c_hidden implementation path: `cfg.env.c_visible: bool = True` default + pkg-02 obs-mask 3-line patch.
- **pkg-08 design.md §4 D5** — zero-shot cfg-driven, never hardcoded; three reserved fields on `cfg.eval`.
- **pkg-08 design.md §4 D6** — regret indicator: `regret(c) = oracle_ceiling(c) - method(c)`; cache miss triggers `oracle_only` re-run; 5 seeds.
- **pkg-08 spec 01 §3.1** — `EvalReport` `@dataclass` schema; this spec populates 8 of the 32 fields.
- **pkg-08 spec 01 §8.1** — c-grid aggregation (the 7-value zero-shot grid is sourced from `cfg.eval.zero_shot_test_c` and is the canonical c-grid for `return_per_c`).
- **pkg-08 README C8-EVAL-CHID1 / C8-EVAL-ZS1 / C8-EVAL-REGRET1** — each surfaced as a named test in §6.
- **pkg-07 spec 04 §3** — N-parametric `ResourceCommonsPettingZooEnv` (the env adapter that pkg-08 spec 02 routes c_visible through; the adapter does not need changes — `c_visible` is consumed at the underlying `ResourceCommonsEnv` level).
- **pkg-07 spec 04 §6** — `reset(options={"c": c})` signature: this spec consumes the `options["c"]` path in the test contract (§6.1) to drive c_t to a known value when verifying the c-channel mask.
- **pkg-07 spec 04 §7** — two-flag info gate: `oracle_mode=False` and `eval_info_mode=True` for eval. Pkg-08 spec 02 does not change either flag — the c_hidden masking is orthogonal to the info gate (info dict carries `c_true` regardless; the obs mask zeroes the c slot in the observation).
- **`hyper_mve/configs/eval_config.py`** — lines 37–39 carry the three reserved `zero_shot_*` fields consumed verbatim.
- **`hyper_mve/configs/env_config.py`** — the target file for the +1 `c_visible` field; declared in §5.2 as a downstream patch.
- **`hyper_mve/schemas/observation.py`** — `ObservationLayout` defines the c-channel index via `block_offset("global", N, K)[0]`; the 3-line patch (§3.2) references this constant by name.
- **`hyper_mve/envs/resource_commons/observations.py`** — the target file for the +3-line behavioural patch; declared in §5.1.
- **`hyper_mve/envs/resource_commons/env.py`** lines 280 and 308–323 — read but **not modified** by this spec. Line 280 is the `step()` return tuple; lines 308–323 are the `_build_info` oracle/eval-only grouping. Both stay byte-identical; pkg-07 spec 04 §7's adapter-level filtering remains the responsibility for info-dict gating.

### Downstream consumption (spec 02 → others)

- **spec 01** consumes `populate_zero_shot_fields()`, `compute_regret()`, and `cfg.env.c_visible` at evaluator entry.
- **spec 05** consumes the cache-miss path and the c_hidden paired-run protocol.
- **spec 08** mirrors §5 of this spec as the canonical "下游补丁声明" list.

### Existing repo ground-truth anchors

- `hyper_mve/configs/eval_config.py:37-39` — reserved `zero_shot_*` fields.
- `hyper_mve/configs/env_config.py` — target for `c_visible` addition.
- `hyper_mve/envs/resource_commons/observations.py:36-79` — `build_observation` body; target for 3-line patch at the function tail.
- `hyper_mve/envs/resource_commons/observations.py:156-162` — `_global_block` showing `[c_t, time_remaining_ratio]` ordering (c at slot 0).
- `hyper_mve/schemas/observation.py:32-100` — `ObservationLayout` class with `BLOCK_ORDER` and `block_offset` static method.
- `hyper_mve/envs/resource_commons/env.py:128-149` — `reset(options={"c": c, "types": ...})` signature consumed in §6.1 tests.
- `hyper_mve/envs/resource_commons/env.py:295-324` — `_build_info` dict construction (untouched by this spec).

---

## Anchors (verbatim grep targets for spec 08 §6 drift detector)

- `pkg-08 spec 02 Lock 1: Pkg-02 touchpoint is exactly 3 lines, declared here, codified in spec 08`
- `pkg-08 spec 02 Lock 2: Zero-shot grid is config-driven, never hardcoded`
- `pkg-08 spec 02 Lock 3: Regret requires either oracle ceiling cache hit OR explicit sweep harness re-dispatch`
- `pkg-08 spec 02 §2: zero-shot protocol — train (0.2, 0.5, 0.8) / test 7-point / unseen 4-point — all CFG-DRIVEN`
- `pkg-08 spec 02 §3.2: 3-line obs-mask hook in observations.py using ObservationLayout.block_offset("global", N, K)[0]`
- `pkg-08 spec 02 §4.3: cache path runs/_oracle_ceilings/<config_hash>/<c>.json`
- `pkg-08 spec 02 §4.4: cache JSON 9-field schema (c, config_hash, ceiling_mean, ceiling_sem, n_seeds, n_episodes_per_seed, computed_at_iso8601, git_sha, method)`
- `pkg-08 spec 02 §4.7: cache miss does NOT block evaluate(); defers to spec 05 sweep harness`
- `pkg-08 spec 02 §5: downstream patches — 1 behavioural (observations.py) + 1 config (env_config.py)`
- `pkg-08 spec 02 §6: test contract — 7 named tests`
