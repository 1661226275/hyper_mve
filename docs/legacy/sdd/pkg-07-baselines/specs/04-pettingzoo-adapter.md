# Spec 04 — PettingZoo Adapter (`ResourceCommonsPettingZooEnv`)

> **Anchors**: design.md §3.5 (Day-1 HARD GATE #6) · design.md §D7 (Tier-1 external consumption) · proposal.md §2.1.5 (adapter pseudocode + `test_adapter_info_gating.py`) · README.md C7-EXT-ADPT1 + C7-EXT-ADPT2.
> **Status**: SDD only — describes the contract for `hyper_mve/envs/adapters/pettingzoo_wrapper.py`, not the implementation.
> **Cross-refs**: spec 05 (MAPPO) · spec 06 (QMIX / MA-MuZero-GH / MAMBA) · spec 08 (integration contracts).

---

## 1. Purpose

This spec locks the **architectural keystone** through which every external baseline in pkg-07 (3 Tier-1 + Tier-2 MAMBA + 2 stubs MARIE/GA) consumes `ResourceCommonsEnv`. It exists because:

1. **DRY single point of env coupling** — without the adapter, each external runner would re-wrap `ResourceCommonsEnv` independently, producing ≥ 3 copies of action-dict marshalling, info-dict filtering, and obs reshaping logic. A single shared adapter caps the surface area at one file.
2. **Oracle / eval-only information firewall** — `ResourceCommonsEnv._build_info` (env.py:295-324) deliberately mixes three tiers (Public / Oracle / Eval-only) inside one dict so that the trainer can route fields by schema marker. Without a central enforcement point, any external runner could inadvertently consume `info["c_true"]` (the ground-truth context) or `info["resource_state"]` (the global resource map), invalidating the assertion-B′ generation-spectrum comparison and the CTDE legitimacy claim.
3. **CTDE-legitimacy boundary anchor** — pkg-07 makes a load-bearing claim that external centralized critics see *only* `concat([obs_i for i ∈ agents])` ± `caps` (public), never privileged ground truth. This adapter is where that boundary is mechanized.
4. **N-parametric uniformity** — `ResourceCommonsEnv` is N-parametric (env.py:84 `self.N = cfg.N`, `N ∈ {2, 4, 8}` per Easy/Medium/Hard presets). The adapter must propagate N from `env_cfg.N` rather than hard-code 4, so the same code path covers all three presets and the Easy↔Medium smoke matrix at C7-EXT-SMOKE1.

The deliverable is a single file `hyper_mve/envs/adapters/pettingzoo_wrapper.py` implementing `pettingzoo.ParallelEnv`. No upstream env code changes (see §11).

---

## 2. Class signature (locked)

```python
# hyper_mve/envs/adapters/pettingzoo_wrapper.py

from pettingzoo.utils.env import ParallelEnv
from hyper_mve.envs.resource_commons.env import ResourceCommonsEnv
from hyper_mve.configs.env_config import EnvConfig

class ResourceCommonsPettingZooEnv(ParallelEnv):
    """N-parametric (N = env_cfg.N ∈ {2, 4, 8}) PettingZoo ParallelEnv wrapper
    around ResourceCommonsEnv. The default construction (oracle_mode=False AND
    eval_info_mode=False) is the *only* configuration external baselines may use
    in training; pkg-08 evaluator is the unique caller permitted to flip
    eval_info_mode=True (read-only metric collection)."""

    def __init__(
        self,
        env_cfg: EnvConfig,
        oracle_mode: bool = False,
        eval_info_mode: bool = False,
    ) -> None: ...
```

- **Constructor args**: exactly three. `env_cfg: EnvConfig` (typed; passing a raw dict raises `TypeError` — inherited from `ResourceCommonsEnv.__init__` env.py:76-80). `oracle_mode` and `eval_info_mode` are independent booleans (see §7 orthogonality).
- **No `seed` arg on `__init__`** — PettingZoo passes seed via `reset(seed=…)`. Forwards verbatim to `ResourceCommonsEnv.reset` (env.py:128-129).
- **No `agent_selector` / sequential AEC API** — this is `ParallelEnv` only. External runners that require AEC must call `pettingzoo.utils.conversions.parallel_to_aec()` themselves; not in scope here.

---

## 3. N-parametric agents contract

```python
# Inside __init__:
self._env = ResourceCommonsEnv(env_cfg)
self._oracle_mode = bool(oracle_mode)
self._eval_info_mode = bool(eval_info_mode)
self._N: int = self._env.N                              # env.py:84
assert self._N in (2, 4, 8), (
    f"env_cfg.N={self._N} outside locked preset domain {{2,4,8}} — "
    "preset family is Easy(2)/Medium(4)/Hard(8)."
)
self.agents: list[str] = [f"agent_{i}" for i in range(self._N)]
self.possible_agents: list[str] = list(self.agents)     # ParallelEnv contract
```

Hard rules (every `range(...)` and dict-comprehension in the adapter uses `self._N`):

- Agent IDs are **always** the literal strings `"agent_0"`, `"agent_1"`, …, `f"agent_{self._N-1}"`. No alternative naming scheme (no `"hunter_0"`, no `"agent0"`).
- `self.agents` is mutable per PettingZoo convention (agents drop from the list on termination). `self.possible_agents` is the immutable roster used by external libraries to size networks.
- `self._N` is captured once at construction; the adapter does not re-read `self._env.N` per step (no agent-count change mid-episode).
- C7-EXT-ADPT2 verifies `env.agents == [f"agent_{i}" for i in range(expected_N)]` for `expected_N ∈ {2, 4, 8}` at Easy/Medium/Hard.

---

## 4. Action space

Per-agent action space is `Discrete(6)` with the **fixed encoding** inherited from `make_action_space` (spaces.py:25-30):

```
0 = NOOP, 1 = UP, 2 = DOWN, 3 = LEFT, 4 = RIGHT, 5 = HARVEST
```

PettingZoo `ParallelEnv` exposes action space *per agent*, not joint:

```python
from gym.spaces import Discrete

def action_space(self, agent: str) -> Discrete:
    return Discrete(6)            # A=6, identical across agents
```

The underlying env's `MultiDiscrete([A] * N)` joint action space (spaces.py:25, also visible as `self._env.action_space`) is reconstituted in `step()` by stacking the per-agent ints. The adapter does **not** expose `MultiDiscrete` directly to external runners — they get the per-agent `Discrete(6)` view, which is what PettingZoo-native libraries (PyMARL, muzero-general, etc.) expect.

Constants are **not** hard-coded inside the adapter — they are imported from `hyper_mve.envs.resource_commons.spaces` so the adapter stays in lockstep if `make_action_space` ever changes (it will not, per Pkg-02 spec 08 §2.4).

---

## 5. Observation space

Per-agent observation space is the row-i slice of the joint observation space built by `make_observation_space` (spaces.py:10-22):

```python
from gym.spaces import Box
from hyper_mve.schemas.obs_layout import ObservationLayout

def observation_space(self, agent: str) -> Box:
    obs_dim = ObservationLayout.total_dim(self._N, self._env.K)
    return Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
```

Key invariants:

- **No oracle types appear in `obs`** — `ObservationLayout` is defined in pkg-01 to include only public per-agent fields (own position, own capability vector `cap_i`, local deltas, sensed neighborhood). Per-agent type labels and ground-truth context live exclusively in `info` and are gated by `oracle_mode` (§7).
- **`cap_i` is public** — an agent observing its own capability vector is part of `ObservationLayout` and is the same surface the internal hyper model consumes via `set_context_subjective(agent_id, cap_i, belief)`. The footnote at design.md §3.5 calls this out: per-agent caps are public; the joint cap matrix is also public.
- **CTDE legitimate concat** — when an external runner (e.g. QMIX mixer, MAPPO centralized critic) needs a "global state", the *only* legal construction is `np.concatenate([obs_dict[a] for a in self.agents])` ± per-agent action one-hots ± `caps`. Anything else (`info["resource_state"]`, `info["hotspot_centers"]`, `info["c_true"]`, `info["types"]`) is privileged and must be gated.
- The adapter does **not** mutate or augment `obs` — the wrapper slices the joint `(N, obs_dim)` array returned by `ResourceCommonsEnv.reset` / `.step` into per-agent rows; no projection, no normalisation, no re-encoding.

---

## 6. `reset` / `step` API (pseudocode, verbatim from proposal §2.1.5)

```python
def reset(
    self,
    seed: int | None = None,
    options: dict | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    obs_arr, info = self._env.reset(seed=seed, options=options)
    # obs_arr shape: (N, obs_dim). Slice into per-agent dict (no copy needed; numpy view).
    obs_dict = {f"agent_{i}": obs_arr[i] for i in range(self._N)}
    info_dict = self._filter_info(info, per_agent=True)
    # PettingZoo ParallelEnv contract: reset returns (obs, info) — both dicts.
    self.agents = list(self.possible_agents)        # restore roster on reset
    return obs_dict, info_dict


def step(
    self,
    action_dict: dict[str, int],
) -> tuple[
    dict[str, np.ndarray],   # obs
    dict[str, float],        # reward
    dict[str, bool],          # terminated
    dict[str, bool],          # truncated
    dict[str, dict],          # info
]:
    # Hard contract: every agent_i in self.agents must be a key.
    missing = [a for a in self.agents if a not in action_dict]
    assert not missing, f"action_dict missing keys: {missing}"
    action_arr = np.array(
        [action_dict[f"agent_{i}"] for i in range(self._N)], dtype=np.int64,
    )
    obs_arr, reward_arr, term, trunc, info = self._env.step(action_arr)
    obs_dict   = {f"agent_{i}": obs_arr[i]    for i in range(self._N)}
    reward_dict = {f"agent_{i}": float(reward_arr[i]) for i in range(self._N)}
    term_dict  = {f"agent_{i}": bool(term)   for i in range(self._N)}
    trunc_dict = {f"agent_{i}": bool(trunc)  for i in range(self._N)}
    info_dict  = self._filter_info(info, per_agent=True)   # SAME filter as reset()
    # Drop terminated agents from self.agents (PettingZoo convention).
    if term or trunc:
        self.agents = []
    return obs_dict, reward_dict, term_dict, trunc_dict, info_dict
```

Hard rules:

- **`reset` and `step` apply the SAME `_filter_info` call** (literally identical kwargs `per_agent=True`). This is C7-EXT-ADPT1's double-path coverage requirement — every leak surface must be tested on both `reset` and `step` to catch any future asymmetric branching.
- **`action_dict` keying is strict** — missing or extra keys are an `AssertionError`, not a silent default. External runners that pass `action_dict` without all `self._N` keys (`"agent_0"` … `f"agent_{N-1}"`) fail loudly at the adapter boundary, not deep inside `ResourceCommonsEnv.step`.
- **`term` / `trunc` are scalar in `ResourceCommonsEnv`** (episode-level) but exposed per-agent in PettingZoo, hence the broadcast.
- **`options` is forwarded verbatim** to `ResourceCommonsEnv.reset` — including `options["c"]` (forces `c_0`) and `options["types"]` (overrides `cfg.type_assignment` for Ablation 3). The adapter does not mediate or validate `options`; that contract belongs to `ResourceCommonsEnv.reset` (env.py:128-149).

---

## 7. Two-flag info-gating taxonomy (verbatim from design.md §3.5)

The adapter consumes `info` from `ResourceCommonsEnv._build_info` (env.py:295-324), which publishes three classes of fields plus schema markers:

| Key | Class | Env schema marker | Gate flag | Default external-visible? |
|----|------|------------------|-----------|--------------------------|
| `caps` | Public (per-agent capability vector) | — | — | YES (agent's own row is also in `obs`) |
| `deltas`, `step_idx`, `harvests` | Public | — | — | YES |
| `c_true` | Oracle (ground-truth context) | `_oracle_fields` (env.py:322) | `oracle_mode` | NO (default `oracle_mode=False`) |
| `types` | Oracle (per-agent α / β type label) | `_oracle_fields` (env.py:322) | `oracle_mode` | NO (default `oracle_mode=False`) |
| `hotspot_centers` | Eval-only (spawn distribution parameters) | `_eval_only_fields` (env.py:323) | `eval_info_mode` | NO (default `eval_info_mode=False`) |
| `resource_state` | Eval-only (per-resource exact position + stock) | `_eval_only_fields` (env.py:323) | `eval_info_mode` | NO (default `eval_info_mode=False`) |
| `_oracle_fields`, `_eval_only_fields`, `_info_schema_version` | Schema metadata | — | Always stripped | NO (prevents introspection-based leak) |

**Flag orthogonality (locked)**:

- `oracle_mode=True` is permitted **only** in the internal hyper evaluation path (when oracle context is being injected for assertion-A regret-upper-bound runs via the `oracle_only` curriculum override). **No external runner ever flips `oracle_mode=True`** — there is no code path in `MAPPOAlgorithm`, `QMIXAlgorithm`, `MAMuZeroGHAlgorithm`, `MAMBAAlgorithm`, or any stub that may construct the adapter with this flag set.
- `eval_info_mode=True` is permitted **only** in the pkg-08 evaluator's metric-collection path (read-only; the gated fields are recorded for plotting / regret accounting and never enter any policy, critic, mixer, or rollout buffer). Training-time `env_fn` factories always construct the adapter with `eval_info_mode=False`.
- The two flags are **independent booleans** (4 combinations); any conjunction is syntactically possible but only three are operationally legal: `(False, False)` for all training, `(True, False)` for internal hyper oracle eval, `(False, True)` for pkg-08 eval-only metric collection. The fourth, `(True, True)`, is not used by any pkg-07 caller; the adapter does not forbid it but no production code path constructs it.

**Schema-marker strip (always-on)**:

The three keys `_info_schema_version`, `_oracle_fields`, `_eval_only_fields` are **always** removed, irrespective of flag values. They publish the *names* of the privileged fields, and any external runner that introspects them could reconstruct what was hidden ("here is the oracle field roster, please use it to debug"). Stripping is unconditional.

**CTDE legitimacy footnote** (canonical wording, design.md §3.5):

- **LEGAL global state** (consumable by centralized critic / value-decomposition mixer): `concat([obs_i for i in agents])` ± action one-hots ± `caps` (public). This is the joint observation already available to every agent under the public obs space.
- **PRIVILEGED state** (must be gated): `c_true` / `types` / `resource_state` / `hotspot_centers`. Even though a centralized critic in CTDE is conventionally fed "extra global information", in this codebase that information must come from concat-of-obs only. The adapter enforces this by default.
- **Sole exception**: pkg-08 evaluator's `eval_info_mode=True` channel — read-only, metric-collection only, never fed back into policy / critic / mixer / buffer.

---

## 8. `_filter_info` implementation (verbatim from proposal §2.1.5)

```python
def _filter_info(self, info: dict, per_agent: bool) -> dict:
    """Two-flag info gate. Reads env's published schema markers — DRY against env evolution.

    Implementation invariants:
      - Always strips _info_schema_version, _oracle_fields, _eval_only_fields
        (they leak structure even if the values they name are gated).
      - Drop set is derived from env-published marker tuples, NOT a hard-coded
        list, so a future env change that adds a field to _oracle_fields is
        automatically gated without an adapter edit.
      - Returned per-agent dict: each agent sees the SAME kept-fields dict
        (no per-agent personalisation of info; per-agent personalisation
        belongs in obs, not info).
    """
    drop: set[str] = set()
    if not self._oracle_mode:
        drop.update(info.get("_oracle_fields", ()))        # env.py:322: ('c_true', 'types')
    if not self._eval_info_mode:
        drop.update(info.get("_eval_only_fields", ()))     # env.py:323: ('hotspot_centers', 'resource_state')
    # Always strip the schema markers themselves — they leak structure even if values gated.
    drop.update({"_info_schema_version", "_oracle_fields", "_eval_only_fields"})
    kept = {k: v for k, v in info.items() if k not in drop}
    if per_agent:
        return {f"agent_{i}": kept for i in range(self._N)}
    return kept
```

Key properties:

- **Marker-driven, not hard-coded** — `drop.update(info.get("_oracle_fields", ()))` reads the env's own declaration. If a future pkg-02 patch promotes a new field into `_oracle_fields` (say `"belief_log_prior"`), the adapter gates it automatically; no edit required. C7-EXT-ADPT1's reference to "DRY抗 env 演化" (DRY against env evolution) is grounded here.
- **Order of operations matters**: marker strip happens *after* the conditional value drops, but unconditionally. The `kept` dict-comprehension runs once at the end. There is no path on which the marker keys survive into external runner view.
- **Per-agent dict is a shallow alias** — every `f"agent_{i}"` key points to the same `kept` dict object. External runners must treat info as read-only (PettingZoo convention); the adapter does not deep-copy on each fan-out.
- **`info.get(..., ())` default** — if a future env strips a marker tuple entirely, gating degrades safely to "drop nothing extra"; the adapter still strips the marker key itself.

---

## 9. Unit test surface (`tests/baselines/external/test_adapter_info_gating.py`)

One file covers C7-EXT-ADPT1 (info gating, 4 leak surfaces × 2 flags) and C7-EXT-ADPT2 (PettingZoo + N-parametric + A=6 + Easy/Medium smoke) jointly.

**Test matrix**:

| Axis | Values | Source |
|------|--------|--------|
| Leak surface | `c_true`, `types`, `hotspot_centers`, `resource_state` | env.py:309-313 |
| Schema marker (always-strip) | `_info_schema_version`, `_oracle_fields`, `_eval_only_fields` | env.py:321-323 |
| Flag combination | `(False, False)` default · `(True, False)` oracle-only · `(False, True)` eval-only | §7 orthogonality |
| API path | `reset`, `step(NOOP)` | env.py — `_build_info` called on both |
| Preset | Easy (N=2), Medium (N=4), Hard (N=8) for the ID test | C7-EXT-SMOKE1 |

**Test cases (sketch — full code in proposal §2.1.5)**:

1. `test_defaults_strip_all_leak_surface_and_markers(preset_name, path)` — parametrized over `preset_name ∈ {"easy", "medium"}` × `path ∈ {"reset", "step"}`. Asserts that under default flags every key in `LEAK_SURFACE ∪ SCHEMA_MARKERS` (7 keys) is absent from `info[agent]` for **every** `agent ∈ env.agents`. Covers C7-EXT-ADPT1 default-leak surface × double-path × N-parametric.
2. `test_oracle_mode_exposes_only_oracle_fields()` — flips `oracle_mode=True` while keeping `eval_info_mode=False`; asserts `c_true` AND `types` present, `hotspot_centers` AND `resource_state` absent. Covers flag orthogonality (oracle does not leak eval).
3. `test_eval_info_mode_exposes_only_eval_fields()` — flips `eval_info_mode=True` while keeping `oracle_mode=False`; asserts `hotspot_centers` AND `resource_state` present, `c_true` AND `types` absent. Covers flag orthogonality (eval does not leak oracle).
4. `test_agent_ids_are_N_parametric()` — for `preset ∈ {"easy", "medium", "hard"}` constructs adapter and asserts `env.agents == [f"agent_{i}" for i in range(expected_N)]` with `expected_N ∈ {2, 4, 8}`. Covers C7-EXT-ADPT2 N-parametric agents.
5. `test_action_space_is_discrete_6_per_agent()` — asserts `env.action_space(a) == Discrete(6)` for every `a ∈ env.agents`. Covers C7-EXT-ADPT2 A=6.
6. `test_action_dict_missing_key_raises()` — passes `step({})` and asserts `AssertionError`. Covers the strict action-dict contract at §6.

**Joint coverage claim**: tests 1+2+3 cover C7-EXT-ADPT1 (4 leak surfaces × 2-flag orthogonality × reset/step double-path); tests 4+5+6 cover C7-EXT-ADPT2 (PettingZoo structural contract + N-parametric over {Easy, Medium} + A=6). Per the README C7-EXT-ADPT2 verification note, `test_adapter_info_gating.py` regresses both constraints simultaneously — there is no separate `test_adapter_structure.py`.

**No CI flakes**: the tests do not depend on RNG order or stochastic step outcomes; the NOOP joint action used in path-`step` is deterministic in observation routing, and only info-key *presence/absence* is asserted (not numeric values).

---

## 10. External runner consumption contract

External baselines (spec 05 MAPPO, spec 06 QMIX / MA-MuZero-GH / MAMBA) consume the adapter via an `env_fn` factory injected into the `ExternalBaselineRunner.train()` entry point:

```python
# In ExternalBaselineRunner.train signature (locked in spec 08 §3):
def train(
    self,
    cfg: V4Config,
    env_fn: Callable[[], ResourceCommonsPettingZooEnv],
    ...
) -> None: ...
```

Hard rules for external runners (enforced by spec 05/06 lint + spec 08 integration test):

- **Per-baseline runners MUST NEVER instantiate `ResourceCommonsEnv` directly.** A grep for `ResourceCommonsEnv(` inside `hyper_mve/baselines/external/*.py` must return zero hits outside type-only imports. The `env_fn` callable is the only legal env construction site.
- The `env_fn` factory is built by spec 01's `create_baseline(cfg, variant)` at the moment it constructs an `ExternalBaselineRunner`:
  ```python
  env_fn = lambda: ResourceCommonsPettingZooEnv(
      cfg.env, oracle_mode=False, eval_info_mode=False,
  )
  ```
  Both flags are pinned `False` at factory time. There is no kwarg on `create_baseline` to override them.
- Pkg-08's unified evaluator constructs a *second* `env_fn` (for `.evaluate(env_fn, c_grid, episodes)`) with `eval_info_mode=True`. This second factory exists *only* inside the evaluator and is never passed to `.train()`. The split-factory pattern guarantees training never sees eval-only info.
- External runners receive `env_fn` as opaque — they may call it any number of times (for vector-env style parallelisation) but must not unwrap to `ResourceCommonsEnv` via `env._env` or any other attribute access. The `_env` attribute on the adapter is private (underscore prefix); accessing it from external runner code is a spec violation flagged by the spec 08 integration test.

---

## 11. No upstream env modification

This spec is **purely additive**. It introduces one new file:

- `hyper_mve/envs/adapters/pettingzoo_wrapper.py` (new directory `envs/adapters/` may also need creating)
- `tests/baselines/external/test_adapter_info_gating.py` (new test file)

The following files are **NOT touched** by pkg-07 spec 04:

- `hyper_mve/envs/resource_commons/env.py` — confirmed unchanged. The schema-marker tuples at env.py:322-323 (`_oracle_fields`, `_eval_only_fields`) already exist in v4 and are leveraged read-only by the adapter. The `self.N = cfg.N` line at env.py:84 is consumed by the adapter as `self._env.N`.
- `hyper_mve/envs/resource_commons/spaces.py` — confirmed unchanged. `make_action_space(N, A=6)` at spaces.py:25 and `make_observation_space(N, K)` at spaces.py:10 are consumed read-only.
- `hyper_mve/envs/resource_commons/state.py`, `context.py`, `reward.py`, etc. — out of scope.
- pkg-01 / pkg-02 / pkg-03 / pkg-04 / pkg-05 SDD specs — confirmed unchanged. The two implementation patches mentioned at design.md Q7 (Pkg-02 obs-mask, Pkg-05 planner flag, Pkg-05 CLI extension) are spec 08's concern, not spec 04's.

This is consistent with pkg-07 Non-Goal NG6 (design.md §2.3): "不重写 ResourceCommonsEnv（仅适配器封装）". The adapter is pure additive wrapper code.

---

## Cross-references

- **design.md §3.5** — Day-1 HARD GATE #6: two-flag info gating + 4 leak surface + CTDE legitimacy footnote + N-parametric agents (verbatim 7-row taxonomy table in §7 above).
- **design.md §D7** — Tier-1 external runners all consume `ResourceCommonsPettingZooEnv` with `oracle_mode=False, eval_info_mode=False` defaults; external training flow diagram (mermaid §7) shows adapter as keystone.
- **README C7-EXT-ADPT1** — 4 leak-surface fields (`c_true`/`types`/`hotspot_centers`/`resource_state`) + 2-flag orthogonality + schema-marker strip; verified by `test_adapter_info_gating.py`.
- **README C7-EXT-ADPT2** — PettingZoo `ParallelEnv` + N-parametric (`agent_0..agent_{N-1}` from `env_cfg.N`) + A=6 + smoke on Easy (N=2) and Medium (N=4); jointly verified by `test_adapter_info_gating.py`.
- **proposal.md §2.1.5** — adapter pseudocode + `_filter_info` implementation + `test_adapter_info_gating.py` test sketch (this spec lifts both verbatim).
- **env.py:84** — `self.N: int = cfg.N` (N-parametric source of truth).
- **env.py:308-323** — `_build_info()` returns the three-tier info dict with `_oracle_fields = ("c_true", "types")` and `_eval_only_fields = ("hotspot_centers", "resource_state")` schema markers.
- **spaces.py:25** — `make_action_space(N, A=6)` with locked encoding `0=NOOP / 1=UP / 2=DOWN / 3=LEFT / 4=RIGHT / 5=HARVEST`.
- **spec 05** (MAPPO) — first consumer of `env_fn` factory.
- **spec 06** (QMIX / MA-MuZero-GH / MAMBA) — second through fourth consumers.
- **spec 08** (integration contracts) — locks `ExternalBaselineRunner.train(cfg, env_fn, ...)` signature and the lint rule against direct `ResourceCommonsEnv` instantiation in external runners.
