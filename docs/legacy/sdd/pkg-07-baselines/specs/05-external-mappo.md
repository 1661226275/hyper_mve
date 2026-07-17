# Spec 05 — External MAPPO (`MAPPOAlgorithm`, Tier-1)

> **Anchors**: design.md §D5 (External 披露式 fairness: params + walltime + LR sweep + 5 seeds) · design.md §D7 (Tier-1 external: MAPPO/QMIX/MA-MuZero-GH consume PettingZoo adapter) · design.md §3.4 (vendoring source = `D:\RL\lzj\MAPPO\`) · design.md §3.5 (N-parametric adapter + two-flag info gating + CTDE legitimacy: concat-of-obs is legal global state for central critic; raw `info["resource_state"]` / `info["hotspot_centers"]` is NOT) · README C7-EXT-FACT1 / C7-EXT-API1 / C7-EXT-SMOKE1 / C7-EXT-FAIR1.
> **Status**: SDD only — describes the contract for `hyper_mve/baselines/external/mappo.py`, not the implementation.
> **Cross-refs**: spec 01 (factory dispatch) · spec 04 (adapter env_fn contract + two-flag info gate) · spec 07 (披露式 fairness) · spec 08 (downstream patches — CLI string + cfg.baselines field) · pkg-08 spec 01 (EvalReport schema) · pkg-08 spec 05 (sweep harness consumes LR grid).

---

## 1. Purpose + Tier-1 status declaration

This spec locks the **Tier-1 external policy-gradient baseline** for pkg-07. MAPPO (Multi-Agent PPO with a centralized critic, decentralized actors) is one of three Tier-1 external baselines in the pkg-07 Methods 主表 9-column (hyper + 5 internal + 3 Tier-1, design §3.1 canonical 量词). Its presence anchors the "on-policy / policy-gradient" 横向坐标 in Ch6.10 — without it, the assertion that hyper is competitive against mainstream MARL is unbacked.

Tier-1 means:

1. **Must ship**: MAPPO is one of the three baselines whose final-return numbers appear in the Ch6 Methods main table. Failure to ship blocks pkg-07 finalize.
2. **Required smoke gate**: `tests/baselines/external/test_mappo_smoke.py` must show return-at-step-20K significantly above the random baseline at step 0 (3 seeds, p<0.05; §10 below). Failure of the smoke gate is a Day-N blocker, not a soft warning.
3. **LR-swept fairness**: 3 learning rates × ≥3 seeds = ≥9 runs per Easy/Medium preset (C7-EXT-FAIR1 + design D5 External 披露式 fairness). The full sweep is consumed by pkg-08 spec 05 sweep harness via `cfg.baselines.external_lr_sweep_grid["external_mappo"]`.
4. **Adapter-only env coupling**: `MAPPOAlgorithm` must NEVER instantiate `ResourceCommonsEnv` directly. The `env_fn: Callable[[], ResourceCommonsPettingZooEnv]` is the only legal env construction site (spec 04 §10 lint).

The deliverable is a single file `hyper_mve/baselines/external/mappo.py` implementing `ExternalBaselineRunner` (the protocol declared in `hyper_mve/baselines/_runner_protocol.py`; pkg-07 spec 01 §3.2 `BaselineLike Union`). One sibling smoke test under `tests/baselines/external/`.

---

## 2. Vendoring source + license + commit hash placeholder

| Aspect | Value |
|--------|-------|
| Source path | `D:\RL\lzj\MAPPO\` (local workspace; not under git submodule) |
| Source files consumed | `MAPPO_main.py` (Runner_MAPPO_MPE; class shape) · `mappo.py` (MAPPO_MPE + Actor_MLP/RNN + Critic_MLP/RNN) · `replay_buffer.py` (ReplayBuffer; episode-rollout layout) · `normalization.py` (Normalization, RewardScaling) |
| License | None declared (local user code; design §3.4 footnote: "本地已有 + 离散动作 friendly + 与 NS-MMG-style env 兼容; 本地用户代码（无外部许可问题）"). No upstream license file to vendor. |
| Commit hash | `<LOCKED-AT-IMPLEMENTATION-TIME>` — pkg-07 implementation day captures `git log -1` of the lzj directory (if it is under git) or `sha256sum MAPPO_main.py mappo.py replay_buffer.py normalization.py` of the four files, recording the value into `hyper_mve/baselines/external/mappo.py` as a module-level constant `_VENDORED_FROM = "lzj-mappo @ <hash>"`. The SDD does not pin a hash; the implementation does. |
| Upstream issue tracker | n/a (local) |

**Implication for the SDD**: the spec assumes the four lzj source files are stable for the duration of the pkg-07 implementation window. Any in-flight modification to `D:\RL\lzj\MAPPO\` during implementation MUST be paused; the four-file snapshot at implementation Day 1 of Phase B (README implementation calendar §"实施期 Phase B") is the source of truth.

**No external license concern**: no GitHub fetch, no PyMARL-style Apache-2.0 propagation, no MIT propagation. The vendored code is moved into `hyper_mve/baselines/external/mappo.py` as a fresh authoring under the `hyper_mve` license (which is the umbrella project license).

---

## 3. Port strategy: class-ify `MAPPO_main.py` into `MAPPOAlgorithm`

The lzj source is a CLI-driven script (`if __name__ == '__main__': ...` at `MAPPO_main.py:138`) that:

1. Takes `argparse.Namespace` with ~25 hyperparameters,
2. Constructs `Runner_MAPPO_MPE(args, env_name, number, seed)`,
3. Hard-codes `make_env(env_name, discrete=True)` for MPE Spread,
4. Drives the train loop with TensorBoard logging at `runs/MAPPO/MAPPO_env_<env>_number_<n>_seed_<s>`.

The port keeps the **algorithm logic** (`MAPPO_MPE` class in `mappo.py`, lines 111-308) intact (PPO ratio + GAE + advantage normalisation + value clip + gradient clip + LR decay + K_epochs + mini_batch_size + entropy bonus — all PPO tricks 1-9) and **rewrites the surrounding plumbing**:

| Concern | lzj source | pkg-07 MAPPOAlgorithm |
|---------|------------|----------------------|
| Hyperparameter ingress | `argparse.Namespace` | `cfg: V4Config` + spec 05 §11 per-impl defaults dict |
| Env construction | `make_env(env_name, discrete=True)` + hard-coded MPE | `env_fn: Callable[[], ResourceCommonsPettingZooEnv]` injected by spec 01 factory; spec 04 §10 contract |
| `args.N` source | `self.env.n` (MPE-specific) | `env_fn()._env.N` read once at `__init__` (N-parametric, env_cfg.N ∈ {2,4,8}) |
| `args.obs_dim_n` source | `[env.observation_space[i].shape[0] for i in range(N)]` | Same logic against `env_fn().observation_space(f"agent_{i}")` (PettingZoo per-agent space) |
| `args.action_dim_n` | `[env.action_space[i].n for i in range(N)]` | Fixed at `[6] * N` per spec 04 §4 (`Discrete(6)` per agent) |
| `args.state_dim` (global) | `np.sum(args.obs_dim_n)` (concat-of-obs) | **Identical**: `sum(obs_dim_i for i in agents)` (spec 04 §3.5 footnote: concat-of-obs is the legal CTDE global state) |
| MPE-specific code | `MAPPO_MPE` class name, `run_episode_mpe`, save-path `simple_spread` | dropped: class renamed `MAPPOAlgorithm`, episode runner inlined into `train()`, save-path uses pkg-08 sweep harness's `run_tag` |
| TensorBoard log dir | `'runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}'` | injected by sweep harness `runs/<exp_id>/<run_tag>/tb/` (pkg-08 spec 05) |
| Hard-coded device | `"cuda:2" if available else cpu` | `cfg.train.device` (V4Config field) |
| Replay buffer | `ReplayBuffer` (episodic, MPE-specific layout) | port to `hyper_mve/baselines/external/_episodic_buffer.py` (sibling module); episode_limit reads from `cfg.env.max_steps` not `args.episode_limit` |
| Normalization | `Normalization`, `RewardScaling` | port to `hyper_mve/baselines/external/_normalization.py` (sibling module); identical math |

**Drop list (MPE-specific code that does NOT cross-port)**:

- `MAPPO_main.py:43` TensorBoard SummaryWriter init — moved to caller (sweep harness owns log dir).
- `MAPPO_main.py:113` `obs_next_n, r_n, done_n, _ = self.env.step(a_n)` — assumes gym 4-tuple. The PettingZoo adapter returns the 5-tuple `(obs, reward, term, trunc, info)` (spec 04 §6); MAPPOAlgorithm consumes the adapter's tuple and synthesizes `done = term or trunc` for the buffer's `done_n` field (which is per-agent broadcast of the episode-level scalar; same as spec 04 §6).
- `MAPPO_main.py:138-167` `__main__` block — removed entirely.
- `mappo.py:297-308` `save_model` / `load_model` — replaced by §4 `save_checkpoint(path)` / `load_checkpoint(path)` that consume an arbitrary path (sweep harness picks the path; not a hard-coded `./models/` directory).
- `make_env` import (`MAPPO_main.py:8`) — dropped; env_fn injection replaces it.

**Keep list (algorithm logic that ports verbatim)**:

- `mappo.py:111-308` `MAPPO_MPE` class body (rename to `MAPPOAlgorithm` and refactor I/O as above). All PPO tricks (advantage normalisation, value clip, gradient clip, LR decay, K_epochs, mini_batch_size, entropy bonus) port unchanged.
- GAE computation `mappo.py:208-214` (deltas + reverse loop) ports unchanged.
- Categorical action sampling (`mappo.py:182-186`) + argmax for eval (`mappo.py:179-181`) ports unchanged; consumed by §7 below.

---

## 4. `MAPPOAlgorithm` signature: implements `ExternalBaselineRunner` protocol

```python
# hyper_mve/baselines/external/mappo.py

from __future__ import annotations

from pathlib import Path
from typing import Callable

from hyper_mve.baselines._runner_protocol import ExternalBaselineRunner   # spec 01 §3.2
from hyper_mve.configs import V4Config
from hyper_mve.envs.adapters.pettingzoo_wrapper import ResourceCommonsPettingZooEnv
from hyper_mve.eval.eval_report import EvalReport                          # pkg-08 spec 01 §3


_VENDORED_FROM = "lzj-mappo @ <commit-hash-locked-at-implementation>"


class MAPPOAlgorithm(ExternalBaselineRunner):
    """Tier-1 external policy-gradient baseline.

    Implements ExternalBaselineRunner via the four-method protocol:
      - train(cfg, env_fn, *, total_env_steps, lr, seed) -> None
      - evaluate(env_fn, c_grid, episodes) -> EvalReport
      - save_checkpoint(path: Path | str) -> None
      - load_checkpoint(path: Path | str) -> None

    Centralized critic consumes concat([obs_i for i in agents]) — the legal CTDE
    global state per pkg-07 design §3.5 footnote. NEVER touches info["c_true"],
    info["types"], info["resource_state"], or info["hotspot_centers"].
    """

    def __init__(self, cfg: V4Config) -> None: ...

    def train(
        self,
        cfg: V4Config,
        env_fn: Callable[[], ResourceCommonsPettingZooEnv],
        *,
        total_env_steps: int,
        lr: float,
        seed: int,
    ) -> None: ...

    def evaluate(
        self,
        env_fn: Callable[[], ResourceCommonsPettingZooEnv],
        c_grid: tuple[float, ...],
        episodes: int,
    ) -> EvalReport: ...

    def save_checkpoint(self, path: Path | str) -> None: ...

    def load_checkpoint(self, path: Path | str) -> None: ...
```

### 4.1 `__init__` contract

Reads only `cfg.baselines.external_lr_sweep_grid["external_mappo"]` (declared but not consumed at construction; consumed by sweep harness for `lr` selection) + the per-impl tuning constants enumerated in §11 below. Does not construct any env, does not allocate any tensor — env-dependent shapes (obs_dim, N) are resolved lazily inside `train()` and `evaluate()` so the constructor stays cheap (spec 01 factory dispatch test `test_factory_dispatches_11_keys` constructs every registry entry once; cheap `__init__` keeps that test fast).

### 4.2 `train(cfg, env_fn, *, total_env_steps, lr, seed)` contract

Three keyword-only args make sweep harness call-sites self-documenting:

- `total_env_steps`: budget cap (matches lzj's `args.max_train_steps`). Sweep harness passes `cfg.train.total_env_steps` (Easy preset default at implementation time; pkg-08 spec 05 owns the budget table).
- `lr`: one value from `cfg.baselines.external_lr_sweep_grid["external_mappo"]`. The sweep harness loops over the grid and passes one `lr` per `.train()` call (no inner LR loop in `MAPPOAlgorithm`).
- `seed`: one seed from `cfg.train.seeds` tuple. Same pattern as `lr`: sweep harness loops; the algorithm runs one seed per call.

Inside `train`:

1. Construct env: `env = env_fn()` (single ParallelEnv instance). The lzj source allocated one env in the runner constructor and reused; this port does the same but at `train()` entry. Vector-env style parallelisation (multiple `env_fn()` calls) is NOT supported in v0 — the lzj source did not support it and adding it now would diverge from the verbatim port.
2. Resolve `N = env._N` (via spec 04 §3 attribute exposure), `obs_dim = env.observation_space("agent_0").shape[0]`, `state_dim = N * obs_dim` (concat-of-obs; see §5 below).
3. Construct actor + critic (Actor_MLP / Critic_MLP from lzj `mappo.py:66-108`, port unchanged); actor consumes per-agent obs (`shape=(N, obs_dim)`), critic consumes the centralized state (`shape=(N, state_dim)` after agent-id one-hot concat if `add_agent_id`; §6).
4. Construct optimizer (`torch.optim.Adam(ac_parameters, lr=lr, eps=1e-5)`).
5. Construct replay buffer (`_EpisodicBuffer(N=N, obs_dim=obs_dim, state_dim=state_dim, episode_limit=cfg.env.max_steps, batch_size=...)`; sibling module port from lzj `replay_buffer.py`).
6. Run the train loop verbatim from lzj `MAPPO_main.py:54-69` (while-loop on `total_steps < total_env_steps`, episode collection, train when buffer is full of `batch_size` episodes).
7. Periodic `evaluate()` is NOT called inside `train` (the lzj source did, but pkg-07 separates eval into the unified evaluator path — pkg-08 spec 01 §4 wraps it). Train loop ends when `total_env_steps` is hit; sweep harness then calls `.evaluate()` once with the final policy.
8. RNG: `torch.manual_seed(seed)` + `np.random.seed(seed)` at `train` entry (lzj `MAPPO_main.py:19-20`). The env's own seed is forwarded via `env.reset(seed=seed)` per spec 04 §6.

### 4.3 `evaluate(env_fn, c_grid, episodes)` contract

Returns a single `EvalReport` (pkg-08 spec 01 §3 schema). See §8 below.

### 4.4 `save_checkpoint(path)` / `load_checkpoint(path)` contract

Writes / reads a single `.pt` file at `path`:

```python
torch.save({
    "actor_state_dict": self.actor.state_dict(),
    "critic_state_dict": self.critic.state_dict(),
    "ac_optimizer_state_dict": self.ac_optimizer.state_dict(),
    "vendored_from": _VENDORED_FROM,
    "lr": self.lr,
    "seed": self.seed,
    "config_hash": self._config_hash,
}, path)
```

`load_checkpoint` is the inverse. Used by pkg-08 sweep harness when a sweep row is resumed (rare; primarily for OOM-recovery). The path is arbitrary — `MAPPOAlgorithm` does not hard-code `./models/<name>.pth` (which lzj `mappo.py:297-308` did).

---

## 5. Centralized critic contract (CTDE legitimacy boundary)

This is the single most important contract MAPPOAlgorithm carries. The lzj source builds the global state from `np.array(obs_n).flatten()` (lzj `MAPPO_main.py:111`, `mappo.py:286`) — exactly concat-of-obs. The port preserves this verbatim and additionally encodes the boundary as a code-level assertion + lint surface.

### 5.1 Legal global state construction

```python
def _build_global_state(self, obs_dict: dict[str, np.ndarray]) -> np.ndarray:
    """Build the centralized critic's global state.

    Per pkg-07 design §3.5 footnote:
      LEGAL global state = concat([obs_i for i in agents]) ± action one-hots
                         ± caps (public).
      PRIVILEGED state = info["c_true"] / info["types"] / info["resource_state"]
                       / info["hotspot_centers"]. NEVER consumed.
    """
    return np.concatenate(
        [obs_dict[f"agent_{i}"] for i in range(self._N)],
        axis=0,
    )   # shape: (N * obs_dim,) — matches lzj args.state_dim semantic.
```

The `caps` field is in `info` (spec 04 §7 Public class). MAPPOAlgorithm chooses NOT to consume `caps` from `info`: every agent's row in `obs` already includes `cap_i` (spec 04 §5 invariant + design §3.5 footnote), so concat-of-obs already carries the joint cap matrix. Adding `caps` separately would double-count. This matches lzj's verbatim port.

### 5.2 Forbidden info keys

```python
_FORBIDDEN_INFO_KEYS: frozenset[str] = frozenset(
    {"c_true", "types", "resource_state", "hotspot_centers"}
)
```

Code-level invariant: at every `step()` call inside the train rollout, MAPPOAlgorithm asserts:

```python
obs_dict, reward_dict, term_dict, trunc_dict, info_dict = env.step(action_dict)
for agent_key, per_agent_info in info_dict.items():
    leaked = _FORBIDDEN_INFO_KEYS & set(per_agent_info)
    assert not leaked, (
        f"MAPPOAlgorithm consumed forbidden info keys {leaked} at agent {agent_key}. "
        "Either env_fn was constructed with oracle_mode=True (which violates spec 04 §7) "
        "or the adapter is leaking. See design §3.5 CTDE boundary."
    )
```

This is a defensive runtime check — the adapter (spec 04 §8 `_filter_info`) is the primary enforcement point, but a code-level assertion inside MAPPOAlgorithm provides a second line of defense AND surfaces violations at the consuming runner (faster blame triangulation than seeing a generic "missing key" error 1000 steps later).

### 5.3 Spec 04 §10 lint surface

Per spec 04 §10:

> Per-baseline runners MUST NEVER instantiate `ResourceCommonsEnv` directly. A grep for `ResourceCommonsEnv(` inside `hyper_mve/baselines/external/*.py` must return zero hits outside type-only imports.

MAPPOAlgorithm satisfies this trivially — there is no `ResourceCommonsEnv(` import in `mappo.py`, only `from hyper_mve.envs.adapters.pettingzoo_wrapper import ResourceCommonsPettingZooEnv` (and that only as a type hint, not as a constructor call). The `env_fn` callable is the unique env construction site.

### 5.4 `info["c_true"]` / `info["types"]` semantic boundary

Even if a future code path inside MAPPOAlgorithm (e.g., a debug logger) consumed `info["c_true"]` for non-policy purposes, it would still constitute a CTDE legitimacy violation under pkg-07's strict definition (spec 04 §7 + design §3.5: "external runner training **绝不** 翻 oracle_mode=True"). The §5.2 assertion catches this regardless of where the consumption happens — actor input, critic input, logger, callback, anywhere in the same Python process.

---

## 6. Share-policy / separate-policy toggle

The lzj source maintains a **single** actor (`Actor_MLP` / `Actor_RNN`) shared across all N agents (mappo.py:148-152: one `self.actor`, fed per-agent obs in a batched call). The default is therefore **share-policy** (one actor net for all agents). For heterogeneous-type envs like ResourceCommonsEnv (α / β capability split), separate-policy (N actors) is sometimes argued in MARL literature.

### 6.1 `cfg.baselines.internal_mappo_share_policy` — NOT on cfg.baselines

Per design §D10 末段 (canonical wording: "Per-impl tuning constants (external_smoke_max_env_steps / external_mappo_share_policy / external_qmix_mixer_hidden_dim / external_ma_muzero_gh_simulations) live in spec 05/06 defaults — they are impl-internal, not cross-spec contract."), the share-policy toggle does NOT appear on `BaselinesConfig`. The 5-field BaselinesConfig穷举 enumerated in spec 01 §6.1 excludes per-impl tuning constants.

The toggle therefore lives in §11 below as a module-level default constant in `mappo.py`:

```python
# hyper_mve/baselines/external/mappo.py
_DEFAULT_SHARE_POLICY: bool = True   # share single actor net across all N agents (lzj default; respects N-parametric env)
```

### 6.2 Why default `True`

1. **Verbatim port**: lzj source shares the actor; flipping the default would diverge from the source's tested behaviour.
2. **N-parametric env friendliness**: N changes between presets (Easy=2 / Medium=4 / Hard=8). Separate-policy means N actor nets, with N varying — the train loop's parameter count would change per preset, complicating LR sweep interpretation.
3. **Capability vector is part of obs**: each agent observes its own `cap_i` (spec 04 §5 invariant). A shared actor net thus implicitly conditions on per-agent capability via the obs input — heterogeneous behaviour emerges without separate-net dedication.
4. **Critic-side homogeneity**: the centralized critic always sees concat-of-obs and produces N independent V(s) values via `s.repeat(N, 1)` (lzj `mappo.py:192`) — share-critic is the only consistent pairing with share-actor.

### 6.3 The `add_agent_id` lzj subtlety

The lzj source has a separate `args.add_agent_id` boolean (lzj `mappo.py:141-144`, default `False`) that concatenates a one-hot agent ID to both actor and critic inputs. This is a DIFFERENT toggle from share-policy: even with share-policy=True, `add_agent_id=True` would inject per-agent positional identity.

The port keeps `add_agent_id` as a separate module-level constant in §11 (`_DEFAULT_ADD_AGENT_ID: bool = False`), preserving the lzj default. The two together (`share_policy=True` AND `add_agent_id=False`) is the canonical "fully shared, identity-blind" MAPPO config that the lzj source ships.

### 6.4 If a future spec needs to flip `share_policy=False`

That would change the actor parameter count and therefore the per-baseline `params` reported under External 披露式 fairness (spec 07 §"披露式" table). Flipping would require a synchronous spec 05 §11 + spec 07 edit AND a re-run of the smoke gate; the SDD does not allow silent flipping.

---

## 7. Action representation: discrete A=6 via Categorical

The PettingZoo adapter exposes `Discrete(6)` per agent (spec 04 §4, fixed encoding `0=NOOP / 1=UP / 2=DOWN / 3=LEFT / 4=RIGHT / 5=HARVEST`). MAPPO is on-policy, so:

### 7.1 Training-time: Categorical sampling

```python
# Verbatim port from lzj mappo.py:182-186
prob = self.actor(actor_inputs)        # shape: (N, 6)
dist = Categorical(probs=prob)
a_n = dist.sample()                    # shape: (N,)
a_logprob_n = dist.log_prob(a_n)
```

`a_logprob_n` is stored in the replay buffer (for PPO ratio computation), `a_n.cpu().numpy()` is fed to the adapter's `step({f"agent_{i}": int(a_n[i]) for i in range(N)})`. No Gumbel-softmax, no straight-through estimator: PPO is on-policy and Categorical sampling is differentiable through the log-prob (not through the action), so the PPO ratio + clipping formulation handles the discrete action gradient correctly.

### 7.2 Evaluation-time: argmax

```python
# Verbatim port from lzj mappo.py:179-181
prob = self.actor(actor_inputs)        # shape: (N, 6)
a_n = prob.argmax(dim=-1)
```

This matches the lzj `evaluate=True` branch. No exploration noise at eval time.

### 7.3 Action-dict marshalling

The adapter's `step()` expects `dict[str, int]` keyed by `f"agent_{i}"` for `i ∈ range(N)` (spec 04 §6 strict contract). MAPPOAlgorithm builds the dict via the obvious comprehension:

```python
action_dict = {f"agent_{i}": int(a_n[i].item()) for i in range(self._N)}
```

The strict-key contract in spec 04 §6 means MAPPOAlgorithm must never call `step` with a partial dict — even if a future code path wants to "skip" an agent's action (e.g., for an absorbing terminal state), it must always pass A=6 actions for all N agents. The `NOOP=0` action exists precisely for this purpose.

### 7.4 Why no Gumbel-softmax

Gumbel-softmax is needed for off-policy methods (DDPG-style) that want to backprop through discrete actions. PPO does NOT — the policy gradient is computed against `log_prob(a)`, not against `a` itself. Adding Gumbel-softmax would only introduce variance with no upside. The lzj source explicitly uses Categorical + log_prob; the port keeps this.

---

## 8. `evaluate()` implementation: per-c grid → `EvalReport`

The pkg-08 spec 01 §6 external delegation contract states: external runners return `EvalReport` via the runner's own `.evaluate(env_fn, c_grid, episodes)`. The unified evaluator wraps the returned report (no schema rewrite). MAPPOAlgorithm's `evaluate()` therefore must produce a schema-conformant `EvalReport`.

### 8.1 Population matrix

Per pkg-08 spec 01 §6.2 (external runner field population):

| `EvalReport` field group | MAPPOAlgorithm populates | Reason |
|--------------------------|--------------------------|--------|
| Identity (6: variant, seed, config_hash, eval_mode, eval_planner_mode, c_visible) | All 6, with `eval_mode="planner"` (legacy back-compat) and `eval_planner_mode="planner_full"` (legacy "uses the policy as-is"). | `eval_planner_mode` is set to `"planner_full"` because the runner has no MVE planner; pkg-08 spec 01 §6 allows external runners to set any of the 4 mode strings — `"planner_full"` is the conventional choice meaning "the runner's strongest mode" (cf. pkg-08 spec 01 §12 edge case "External runner returns `EvalReport` with `eval_planner_mode` set to a value other than `'planner_full'` — Allowed"). |
| Headline (5: return_mean, return_sem, return_zero_shot_seen, return_zero_shot_unseen, return_zero_shot_gap) | All 5, computed from per-c rollouts. | The headline scalars are the central metric. |
| Per-c (3: return_per_c, return_per_c_sem, episodes_per_c) | All 3 over `c_grid`. | Direct output of the per-c loop below. |
| c-segment (2: return_per_segment, return_per_segment_sem) | EMPTY `MappingProxyType({})`. | Per pkg-08 spec 01 §6.3, external runners can leave these empty; the unified evaluator runs `_aggregate_segments` post-hoc on `return_per_c`. |
| Bell-curve (2: return_per_type_ratio, return_per_type_ratio_sem) | EMPTY `MappingProxyType({})`. | Same — pkg-08 spec 01 §6.3 post-hoc aggregation. |
| Regret (4: regret_per_c, regret_mean, oracle_ceiling_per_c, oracle_ceiling_cache_hit) | EMPTY / `0.0`. | Regret computation needs oracle ceiling, which is a hyper-only construct (pkg-08 spec 02 §3). External runners leave these for the unified evaluator's post-hoc regret pass. |
| Planner-prior gap (3: planner_prior_return_gap, direct_inference_return_mean, planner_full_return_mean) | `planner_prior_return_gap=0.0`, `direct_inference_return_mean=return_mean`, `planner_full_return_mean=return_mean`. | Per pkg-08 spec 01 §6.2: external runners have no planner, so the gap is 0 and both modes are identified with the headline mean. |
| Diagnostics (5: walltime_seconds, env_steps_evaluated, episodes_total, info_gating_strict, set_context_subjective_oracle_leak) | All 5. `info_gating_strict` is verified via `env_fn()._oracle_mode is False` (the canonical expression in pkg-08 spec 01 §3.1). `set_context_subjective_oracle_leak=False` (no `set_context_subjective` call path). | Verified by `test_external_eval_contract.py` (pkg-08 spec 01 §10.3). |
| Belief diagnostics (2: belief_c_mae, belief_c_calibration) | `None` for both. | No belief head; pkg-08 spec 01 §6.2 explicitly allows `None` for external runners. |
| Schema version (1: schema_version) | `"pkg08-spec01-v1"`. | Sentinel; bumped only by synchronous spec 01 + spec 08 §3 edit. |

### 8.2 Per-c loop pseudocode

```python
def evaluate(
    self,
    env_fn: Callable[[], ResourceCommonsPettingZooEnv],
    c_grid: tuple[float, ...],
    episodes: int,
) -> EvalReport:
    return_per_c: dict[float, float] = {}
    return_per_c_sem: dict[float, float] = {}
    episodes_per_c: dict[float, int] = {}
    all_returns: list[float] = []
    t_start = time.time()
    env_steps = 0

    for c in c_grid:
        per_episode_returns: list[float] = []
        for ep in range(episodes):
            env = env_fn()
            obs_dict, _ = env.reset(seed=self._eval_seed_base + ep, options={"c": c})
            ep_return = 0.0
            for step in range(self._cfg.env.max_steps):    # canonical V4Config-side path; avoids reaching into adapter's private inner attr
                # Argmax (eval): no exploration.
                with torch.no_grad():
                    obs_arr = np.stack([obs_dict[f"agent_{i}"] for i in range(self._N)])
                    actor_inputs = torch.tensor(obs_arr, dtype=torch.float32, device=self.device)
                    if self._add_agent_id:
                        actor_inputs = torch.cat([actor_inputs, torch.eye(self._N, device=self.device)], dim=-1)
                    prob = self.actor(actor_inputs)
                    a_n = prob.argmax(dim=-1).cpu().numpy()
                action_dict = {f"agent_{i}": int(a_n[i]) for i in range(self._N)}
                obs_dict, reward_dict, term, trunc, info = env.step(action_dict)
                # §5.2 defensive check (cheap; only at eval-time it is even cheaper).
                for k, v in info.items():
                    assert not (_FORBIDDEN_INFO_KEYS & set(v))
                ep_return += sum(reward_dict.values())
                env_steps += 1
                if any(term.values()) or any(trunc.values()):
                    break
            per_episode_returns.append(ep_return)
            all_returns.append(ep_return)
        arr = np.array(per_episode_returns)
        return_per_c[c] = float(arr.mean())
        return_per_c_sem[c] = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        episodes_per_c[c] = len(per_episode_returns)

    all_arr = np.array(all_returns)
    walltime = time.time() - t_start

    # Defensive: c values come from the grid (no arithmetic), but pin to nearest-by-tolerance
    # to surface drift if a caller ever passes c = 0.5 + 1e-9 instead of 0.5 verbatim.
    def _snap(c, candidates, tol=1e-9):
        for cand in candidates:
            if abs(c - cand) < tol:
                return cand
        return None
    seen_c = tuple(c for c in c_grid if _snap(c, self._cfg.eval.zero_shot_train_c) is not None)
    unseen_c = tuple(c for c in c_grid if _snap(c, self._cfg.eval.zero_shot_unseen_c) is not None)
    for c in c_grid:                                                  # belt-and-braces
        assert c in seen_c or c in unseen_c, f"c={c!r} fell out of both buckets — grid drift"
    seen_mean = float(np.mean([return_per_c[c] for c in seen_c])) if seen_c else 0.0
    unseen_mean = float(np.mean([return_per_c[c] for c in unseen_c])) if unseen_c else 0.0
    return_mean = float(all_arr.mean())
    return EvalReport(
        variant="external_mappo",
        seed=int(self._seed),
        config_hash=str(self._config_hash),
        eval_mode="planner",
        eval_planner_mode="planner_full",
        c_visible=bool(self._cfg.env.c_visible),
        return_mean=return_mean,
        return_sem=float(all_arr.std(ddof=1) / np.sqrt(len(all_arr))),
        return_zero_shot_seen=seen_mean,
        return_zero_shot_unseen=unseen_mean,
        return_zero_shot_gap=seen_mean - unseen_mean,
        return_per_c=MappingProxyType(return_per_c),
        return_per_c_sem=MappingProxyType(return_per_c_sem),
        episodes_per_c=MappingProxyType(episodes_per_c),
        return_per_segment=MappingProxyType({}),
        return_per_segment_sem=MappingProxyType({}),
        return_per_type_ratio=MappingProxyType({}),
        return_per_type_ratio_sem=MappingProxyType({}),
        regret_per_c=MappingProxyType({}),
        regret_mean=0.0,
        oracle_ceiling_per_c=MappingProxyType({}),
        oracle_ceiling_cache_hit=MappingProxyType({}),
        planner_prior_return_gap=0.0,
        direct_inference_return_mean=return_mean,
        planner_full_return_mean=return_mean,
        walltime_seconds=float(walltime),
        env_steps_evaluated=int(env_steps),
        episodes_total=int(len(all_returns)),
        info_gating_strict=(env_fn()._oracle_mode is False),
        set_context_subjective_oracle_leak=False,
        belief_c_mae=None,
        belief_c_calibration=None,
        schema_version="pkg08-spec01-v1",
    )
```

### 8.3 `options["c"]` semantic

Per spec 04 §6: "`options` is forwarded verbatim to `ResourceCommonsEnv.reset` — including `options["c"]` (forces `c_0`) and `options["types"]` (overrides cfg.type_assignment for Ablation 3). The adapter does not mediate or validate `options`; that contract belongs to `ResourceCommonsEnv.reset`."

MAPPOAlgorithm's `evaluate()` therefore uses `options={"c": c}` to pin the eval context, identical to how internal baseline eval pins c. The c-grid loop is the explicit per-c iteration that `EvalReport.return_per_c` requires.

### 8.4 Determinism

`self._eval_seed_base` (default `42`; per-impl const in §11) is the env-side seed base. The episode loop uses `seed=self._eval_seed_base + ep`, so seeds 42..42+episodes-1 cover the eval. Argmax has no RNG. Therefore `evaluate()` is fully deterministic — repeated calls with the same trained model return byte-identical `EvalReport.return_per_c`. The smoke test (§10) relies on this determinism for its p<0.05 multi-seed comparison.

---

## 9. LR sweep contract

### 9.1 Grid source

Per design D10 + spec 01 §6.1, `cfg.baselines.external_lr_sweep_grid: Mapping[str, tuple[float, ...]]` is a `MappingProxyType` with the entry:

```python
"external_mappo": (1e-4, 3e-4, 1e-3),
```

(default in `configs/v4_config.py::BaselinesConfig`). 3 LRs is the contract minimum (C7-EXT-FAIR1: "External LR sweep ≥3 LR × ≥3 seeds"). Implementation may extend the tuple to a 4th or 5th value, but the **sweep harness contract** in pkg-08 spec 05 requires ≥3 values; failure to provide ≥3 fails the C7-EXT-FAIR1 gate.

### 9.2 Total run count per preset

3 LRs × 5 seeds = **15 runs per preset** (Easy or Medium; Hard is out of pkg-07 smoke scope, in spec 07's披露式 fairness table). 5 seeds is the README C7-EXT-FAIR1 minimum (≥3 stated; pkg-07 implementation pins 5 to align with the rest of the Methods table). pkg-08 spec 05 sweep harness enumerates 15 × 2 presets = 30 MAPPO runs in the full sweep.

### 9.3 Disclosure (spec 07 §"披露式" feed)

For each LR sweep, MAPPOAlgorithm reports to spec 07:

- Final `return_mean` per (LR, seed) pair — extracted from `EvalReport.return_mean`.
- Wall-clock to converge (per spec 07's convergence definition — locked there) — extracted from `EvalReport.walltime_seconds`.
- Parameter count — computed once at `__init__` via `sum(p.numel() for p in (list(self.actor.parameters()) + list(self.critic.parameters())))`; exposed as a method `param_count() -> int` for spec 07's disclosure table.
- Best LR per seed — argmax over LR of final `return_mean`. spec 07 then averages "best-LR final return" over seeds for the披露式 column.

### 9.4 No LR scheduling inside `train()`

The lzj source has `args.use_lr_decay = True` (lzj `MAPPO_main.py:158`) which linearly decays the optimizer LR over training. The port **keeps** this (it's a verbatim algorithm trick; §11 default). The decay is internal to one `train()` call — it does NOT interact with the sweep harness's outer LR sweep. The sweep harness picks an initial LR; `train()` decays from there.

---

## 10. Smoke convergence test (C7-EXT-SMOKE1)

### 10.1 Test file

`tests/baselines/external/test_mappo_smoke.py`. One test function; runs in `pytest --runslow` mode (CI may exclude unless explicitly enabled; sweep harness CI runs it).

### 10.2 Test body

```python
import numpy as np
import pytest

from hyper_mve.baselines import create_baseline
from hyper_mve.configs import V4Config
from hyper_mve.envs.adapters.pettingzoo_wrapper import ResourceCommonsPettingZooEnv


@pytest.mark.slow
def test_mappo_easy_smoke_return_beats_random_at_20k():
    """C7-EXT-SMOKE1: MAPPO on Easy preset (N=2) trained for 20K env steps must
    achieve mean return strictly greater than the random-policy mean return at
    step 0, with p<0.05 over 3 seeds (one-sided Welch t-test)."""
    cfg = V4Config(preset="easy")
    smoke_budget = 20_000   # env steps; matches design D7 / README C7-EXT-SMOKE1
    seeds = (0, 1, 2)
    lr = 3e-4               # middle of the sweep grid

    # 1. Random baseline at step 0 (one eval per seed; no training).
    random_returns = []
    for s in seeds:
        env = ResourceCommonsPettingZooEnv(cfg.env, oracle_mode=False, eval_info_mode=False)
        obs, _ = env.reset(seed=s)
        ep_return = 0.0
        np.random.seed(s)
        for _ in range(cfg.env.max_steps):
            action_dict = {a: int(np.random.randint(6)) for a in env.agents}
            obs, reward, term, trunc, info = env.step(action_dict)
            ep_return += sum(reward.values())
            if any(term.values()) or any(trunc.values()):
                break
        random_returns.append(ep_return)
    random_returns = np.array(random_returns)

    # 2. MAPPO trained for 20K env steps (one run per seed).
    trained_returns = []
    for s in seeds:
        runner = create_baseline(cfg, "external_mappo")
        env_fn = lambda: ResourceCommonsPettingZooEnv(
            cfg.env, oracle_mode=False, eval_info_mode=False,
        )
        runner.train(cfg, env_fn, total_env_steps=smoke_budget, lr=lr, seed=s)
        report = runner.evaluate(env_fn, c_grid=(0.5,), episodes=10)
        trained_returns.append(report.return_mean)
    trained_returns = np.array(trained_returns)

    # 3. Compare: one-sided Welch t-test, alpha=0.05.
    from scipy import stats
    t_stat, p_two_sided = stats.ttest_ind(
        trained_returns, random_returns, equal_var=False,
    )
    p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2
    assert trained_returns.mean() > random_returns.mean(), (
        f"MAPPO mean return after 20K steps ({trained_returns.mean():.2f}) "
        f"not above random baseline ({random_returns.mean():.2f})."
    )
    assert p_one_sided < 0.05, (
        f"MAPPO smoke gate p-value {p_one_sided:.4f} above 0.05 threshold "
        f"(t={t_stat:.2f}). C7-EXT-SMOKE1 violated."
    )
```

### 10.3 Why 20K env steps

Per design §3.4 + README C7-EXT-SMOKE1: "Easy preset 20K env-steps, return > random". 20K is chosen to fit inside ~5 minutes of GPU time on the workstation MAPPO uses (lzj source's `5e6` default is for full convergence; 20K is enough to clearly separate from random on the Easy preset's small N=2 / shallow horizon).

### 10.4 Why 3 seeds + p<0.05

3 seeds gives 4 degrees of freedom in Welch t — enough for a clean p<0.05 separation if MAPPO is actually learning anything. The C7-EXT-SMOKE1 README bullet states "return > random within 20K env steps"; this spec strengthens it to a multi-seed statistical claim because a single-seed comparison is fragile (any sufficiently lucky / unlucky seed flips the inequality). 3 seeds is the README C7-EXT-FAIR1 minimum, reused here for consistency with the sweep gate.

### 10.5 Easy preset only

The smoke gate runs Easy preset (N=2). Medium (N=4) and Hard (N=8) are not in the smoke gate budget — they belong to the full sweep (pkg-08 spec 05). C7-EXT-SMOKE1 explicitly states "Easy preset"; this spec inherits that scope.

### 10.6 What failure means

A failed smoke gate is a Day-N implementation blocker. Triage tree:

1. **Random baseline not reproducible across seeds**: check env determinism — `ResourceCommonsEnv.reset(seed=s)` should be fully seeded. If not, spec 04 §6 `options` forwarding is buggy.
2. **MAPPO trained returns not separable from random**: 20K env steps may be insufficient on Medium / Hard preset shapes; but on Easy this is a real bug. Inspect `cfg.train.device`, `lr=3e-4` may be wrong, replay buffer batching may be off-by-one.
3. **p-value above 0.05 but means clearly separated**: 3 seeds is the minimum; bumping to 5 may help. Document the bump in the test docstring.

---

## 11. Per-impl tuning constants (NOT in `cfg.baselines`)

Per design §D10 末段, the following constants live in `mappo.py` as module-level defaults — they are impl-internal, not cross-spec contract. The 5-field BaselinesConfig穷举 in spec 01 §6.1 explicitly excludes them.

```python
# hyper_mve/baselines/external/mappo.py — module-level defaults

# Architecture
_DEFAULT_MLP_HIDDEN_DIM: int = 64           # lzj args.mlp_hidden_dim
_DEFAULT_RNN_HIDDEN_DIM: int = 64           # lzj args.rnn_hidden_dim
_DEFAULT_USE_RNN: bool = False              # lzj args.use_rnn (False = MLP variant)
_DEFAULT_USE_RELU: bool = False             # lzj args.use_relu (False = Tanh)
_DEFAULT_USE_ORTHOGONAL_INIT: bool = True   # lzj args.use_orthogonal_init

# PPO algorithmic
_DEFAULT_GAMMA: float = 0.99                # lzj args.gamma
_DEFAULT_LAMDA: float = 0.95                # lzj args.lamda (GAE λ)
_DEFAULT_EPSILON: float = 0.2               # lzj args.epsilon (PPO clip)
_DEFAULT_K_EPOCHS: int = 15                 # lzj args.K_epochs
_DEFAULT_ENTROPY_COEF: float = 0.01         # lzj args.entropy_coef
_DEFAULT_USE_ADV_NORM: bool = True          # lzj args.use_adv_norm (Trick 1)
_DEFAULT_USE_REWARD_NORM: bool = True       # lzj args.use_reward_norm (Trick 3)
_DEFAULT_USE_REWARD_SCALING: bool = False   # lzj args.use_reward_scaling (Trick 4; off when reward_norm is on)
_DEFAULT_USE_VALUE_CLIP: bool = False       # lzj args.use_value_clip
_DEFAULT_USE_LR_DECAY: bool = True          # lzj args.use_lr_decay (Trick 6)
_DEFAULT_USE_GRAD_CLIP: bool = True         # lzj args.use_grad_clip (Trick 7)
_DEFAULT_SET_ADAM_EPS: bool = True          # lzj args.set_adam_eps (Trick 9: eps=1e-5)

# Batching
_DEFAULT_BATCH_SIZE: int = 32               # lzj args.batch_size (episodes per train step)
_DEFAULT_MINI_BATCH_SIZE: int = 8           # lzj args.mini_batch_size

# Sharing / identity (§6)
_DEFAULT_SHARE_POLICY: bool = True          # share single actor across all N agents
_DEFAULT_ADD_AGENT_ID: bool = False         # lzj args.add_agent_id (False = identity-blind)

# Eval-side
_DEFAULT_EVAL_SEED_BASE: int = 42           # base seed for evaluate() episode loop
```

All twenty defaults port verbatim from lzj's `argparse` defaults (cross-check lzj `MAPPO_main.py:138-167`). Any change to a default requires a spec 05 §11 edit AND a re-run of the smoke gate (any change might invalidate the C7-EXT-SMOKE1 guarantee).

`MAPPOAlgorithm.__init__` reads these constants once and stores them on `self`. No `cfg.baselines.external_mappo_*` field exists. Future hyperparameter sweeps over these constants (e.g., entropy coefficient sweep) would be a follow-up spec 05.1, not a `cfg.baselines` extension.

---

## 12. Test contract (5 named tests)

`tests/baselines/external/`:

### 12.1 `test_factory_dispatches_external_mappo` (C7-EXT-FACT1)

Located in the shared `tests/baselines/external/test_registry.py` (per pkg-07 README §"输出清单"). Asserts:

```python
def test_factory_dispatches_external_mappo():
    cfg = V4Config(preset="easy")
    runner = create_baseline(cfg, "external_mappo")
    assert isinstance(runner, MAPPOAlgorithm)
    # ExternalBaselineRunner protocol surface (spec 01 §4.3 dispatch)
    assert hasattr(runner, "train")
    assert hasattr(runner, "evaluate")
    assert hasattr(runner, "save_checkpoint")
    assert hasattr(runner, "load_checkpoint")
    # NOT a BaselineModel (no internal 7-API)
    assert not hasattr(runner, "set_context_subjective")
```

### 12.2 `test_mappo_consumes_adapter_only` (spec 04 §10 lint)

Located in `tests/baselines/external/test_mappo_adapter_consumption.py`. Asserts:

```python
def test_mappo_does_not_construct_resource_commons_env():
    """spec 04 §10 lint: grep for ResourceCommonsEnv( inside mappo.py."""
    src = Path("hyper_mve/baselines/external/mappo.py").read_text()
    # Allow type-only import lines; reject constructor calls.
    bad_patterns = [
        "ResourceCommonsEnv(", "ResourceCommonsEnv (",
    ]
    for pat in bad_patterns:
        assert pat not in src, (
            f"MAPPO source contains {pat!r} — spec 04 §10 lint violation. "
            f"env_fn is the only legal env construction site."
        )

def test_mappo_forbidden_info_keys_assertion_present():
    """§5.2 defensive check must be coded (grep for _FORBIDDEN_INFO_KEYS)."""
    src = Path("hyper_mve/baselines/external/mappo.py").read_text()
    assert "_FORBIDDEN_INFO_KEYS" in src
    for k in ("c_true", "types", "resource_state", "hotspot_centers"):
        assert k in src   # the frozenset literal must enumerate all four
```

### 12.3 `test_mappo_evaluate_contract` (C7-EXT-API1 + pkg-08 spec 01 §6 schema)

Located in `tests/baselines/external/test_external_eval_contract.py` (shared with QMIX / MA-MuZero-GH). Asserts:

```python
def test_mappo_evaluate_returns_evalreport_with_correct_schema():
    cfg = V4Config(preset="easy")
    runner = create_baseline(cfg, "external_mappo")
    env_fn = lambda: ResourceCommonsPettingZooEnv(cfg.env, oracle_mode=False, eval_info_mode=False)
    # Cheap eval: 2 c values × 3 episodes; no train required (random policy is fine for shape test)
    report = runner.evaluate(env_fn, c_grid=(0.0, 1.0), episodes=3)
    assert isinstance(report, EvalReport)
    assert report.variant == "external_mappo"
    assert report.eval_mode == "planner"
    assert report.eval_planner_mode == "planner_full"
    # External-runner field population per §8.1
    assert report.planner_prior_return_gap == 0.0
    assert report.direct_inference_return_mean == report.return_mean
    assert report.planner_full_return_mean == report.return_mean
    assert report.belief_c_mae is None
    assert report.belief_c_calibration is None
    assert report.info_gating_strict is True
    assert report.set_context_subjective_oracle_leak is False
    assert report.return_per_segment == MappingProxyType({})
    assert report.return_per_type_ratio == MappingProxyType({})
    assert report.regret_per_c == MappingProxyType({})
    assert report.schema_version == "pkg08-spec01-v1"
    # Per-c populated for the two c values in the grid
    assert set(report.return_per_c) == {0.0, 1.0}
    assert set(report.return_per_c_sem) == {0.0, 1.0}
    assert set(report.episodes_per_c) == {0.0, 1.0}
```

### 12.4 `test_mappo_smoke_easy_20k` (C7-EXT-SMOKE1)

Per §10.2. The single most expensive test in pkg-07 (≈5 min GPU). Gated by `@pytest.mark.slow`.

### 12.5 `test_mappo_lr_sweep_dispatch` (C7-EXT-FAIR1)

Located in `tests/baselines/external/test_mappo_lr_sweep.py`. Asserts the LR sweep grid contract:

```python
def test_mappo_lr_sweep_grid_is_at_least_3_lrs():
    """C7-EXT-FAIR1: external LR sweep ≥3 LR. cfg.baselines.external_lr_sweep_grid
    must have a key 'external_mappo' with ≥3 entries."""
    cfg = V4Config(preset="easy")
    grid = cfg.baselines.external_lr_sweep_grid
    assert "external_mappo" in grid
    assert len(grid["external_mappo"]) >= 3

def test_mappo_train_consumes_lr_kwarg():
    """train(..., lr=...) must be a keyword-only arg (per §4.2 contract)."""
    import inspect
    sig = inspect.signature(MAPPOAlgorithm.train)
    assert "lr" in sig.parameters
    assert sig.parameters["lr"].kind == inspect.Parameter.KEYWORD_ONLY

def test_mappo_train_lr_changes_persisted_lr(tmp_path):
    """Mutation test: different lr arg → different lr persisted in save_checkpoint round-trip.
    
    Asserts via the formal ExternalBaselineRunner contract (save_checkpoint/load_checkpoint) 
    rather than reaching into runner.ac_optimizer (which is an internal implementation 
    detail of the lzj port, NOT part of the protocol surface in §4).
    """
    cfg = V4Config(preset="easy")
    env_fn = lambda: ResourceCommonsPettingZooEnv(cfg.env)
    runner_a = create_baseline(cfg, "external_mappo")
    runner_a.train(cfg, env_fn, total_env_steps=1, lr=1e-4, seed=0)
    path_a = tmp_path / "a.pt"
    runner_a.save_checkpoint(path_a)
    saved_a = torch.load(path_a, map_location="cpu")
    assert saved_a["lr"] == pytest.approx(1e-4, rel=1e-3)              # §4.4 ckpt schema row "lr"
    
    runner_b = create_baseline(cfg, "external_mappo")
    runner_b.train(cfg, env_fn, total_env_steps=1, lr=1e-3, seed=0)
    path_b = tmp_path / "b.pt"
    runner_b.save_checkpoint(path_b)
    saved_b = torch.load(path_b, map_location="cpu")
    assert saved_b["lr"] == pytest.approx(1e-3, rel=1e-3)
```

(Note: `lr_decay` is enabled by default (§11), so the persisted `lr` after one step has been slightly decayed; the `rel=1e-3` tolerance accommodates one-step decay. The test uses `save_checkpoint`'s `lr` field — declared in the §4.4 ckpt schema — rather than introspecting `runner.ac_optimizer`, which is an internal lzj-port attribute not part of the `ExternalBaselineRunner` protocol surface.)

---

## 13. Integration hooks (downstream specs consume)

| Consumer | Consumed | Use |
|----------|----------|-----|
| **spec 01** (factory entry) | `MAPPOAlgorithm` constructor | `EXTERNAL_REGISTRY["external_mappo"] = MAPPOAlgorithm` (spec 01 §3.2) |
| **spec 04** (adapter env_fn) | `ResourceCommonsPettingZooEnv` consumed via `env_fn` in `train()` / `evaluate()` | §4.2 + §8.2 |
| **spec 07** (披露式 fairness reporting) | `param_count()` method + final `return_mean` + `walltime_seconds` per (LR, seed) | spec 07 disclosure table feed |
| **spec 08** (downstream patches — CLI string) | `external_mappo` CLI string in `train_main.py` `_DEFERRED_VARIANTS` extension | spec 08 §"3 处下游补丁声明" |
| **pkg-08 spec 01** (EvalReport schema) | `EvalReport` constructed in §8.2 must conform to the 32-field schema | pkg-08 spec 01 §6 external delegation contract |
| **pkg-08 spec 05** (sweep harness consumes LR grid) | `cfg.baselines.external_lr_sweep_grid["external_mappo"]` enumerated to drive 15 runs / preset | pkg-08 spec 05 §"external sweep enumeration" |

### 13.1 What this spec does NOT add to cfg.baselines

Per design §D10 末段 and spec 01 §6.4: no `external_mappo_*` field on `BaselinesConfig`. All per-impl tuning constants are in §11 module defaults. Only `external_lr_sweep_grid["external_mappo"]` (the cross-spec contract) is on cfg.

### 13.2 What this spec does NOT touch

- **NOT modified**: `hyper_mve/envs/resource_commons/*.py` (NG6: 不重写 ResourceCommonsEnv).
- **NOT modified**: `hyper_mve/envs/adapters/pettingzoo_wrapper.py` (spec 04 is the source of truth; MAPPO consumes, does not extend).
- **NOT modified**: any pkg-01..05 SDD spec (NG2: 不修改 Pkg-01..05 任何 SDD).
- **NOT modified**: `D:\RL\lzj\MAPPO\*.py` (the vendoring is a one-way port; lzj remains unchanged).

---

## 14. Cross-references

### 14.1 Upstream anchors

- **design.md §3.4** — Vendoring source table (`D:\RL\lzj\MAPPO\` row); license = local user code.
- **design.md §3.5** — N-parametric adapter + two-flag info gating + CTDE legitimacy footnote (§5 anchor).
- **design.md §D5** — External 披露式 fairness (LR sweep + params + walltime + 5 seeds).
- **design.md §D7** — Tier-1 external selection + smoke convergence direction test 20K (§10 anchor).
- **design.md §D10** — `cfg.baselines.external_lr_sweep_grid` mapping (§9 anchor); per-impl tuning constants exclusion (§11 anchor).
- **README C7-EXT-FACT1** — 3 Tier-1 instantiable (§12.1 test).
- **README C7-EXT-API1** — `.evaluate(env_fn, c_grid, episodes) -> EvalReport` (§4.3 + §8 anchor).
- **README C7-EXT-SMOKE1** — Easy preset return > random within 20K env steps (§10 anchor).
- **README C7-EXT-FAIR1** — LR sweep ≥3 LR × ≥3 seeds (§9 + §12.5 anchor).
- **proposal §1.2 + §1.4** — MAPPO necessity argument + ExternalBaselineRunner protocol introduction.

### 14.2 Sibling spec anchors

- **spec 01 §2.1 / §3.2** — REGISTRY dispatch; `EXTERNAL_REGISTRY["external_mappo"] = MAPPOAlgorithm` (§4 anchor).
- **spec 01 §6.1** — `cfg.baselines.external_lr_sweep_grid` 5-field BaselinesConfig穷举 (§9.1 anchor).
- **spec 04 §3** — N-parametric agents `f"agent_{i}" for i in range(self._N)` (§4.2 step 2 anchor).
- **spec 04 §4** — Discrete(6) per-agent action space (§7 anchor).
- **spec 04 §5** — Per-agent obs space + `cap_i` public + CTDE legitimate concat (§5 anchor).
- **spec 04 §6** — `reset` / `step` API + `options["c"]` forwarding (§8.3 anchor).
- **spec 04 §7** — Two-flag info gating + default `(False, False)` (§5.2 anchor).
- **spec 04 §10** — External runner consumption contract: `env_fn` factory, `ResourceCommonsEnv(` lint (§5.3 + §12.2 anchor).
- **spec 07 §"披露式"** — External fairness disclosure table (§9.3 anchor; spec 07 owns the table's schema).
- **spec 08 §"3 处下游补丁声明"** — `train_main.py` CLI extension (`external_mappo` string) + `cfg.baselines.external_lr_sweep_grid` consumption claim (§13 anchor).

### 14.3 Downstream pkg-08 anchors

- **pkg-08 spec 01 §3** — `@dataclass(frozen=True) EvalReport` 32-field schema (§8 anchor).
- **pkg-08 spec 01 §6** — External delegation contract: `runner.evaluate(env_fn, c_grid, episodes) -> EvalReport`; field-population matrix (§8.1 anchor).
- **pkg-08 spec 01 §6.2** — External runners: `planner_prior_return_gap=0.0`, `direct_inference_return_mean=return_mean`, `planner_full_return_mean=return_mean`, `belief_c_mae=None`, `belief_c_calibration=None` (§8.1 anchor).
- **pkg-08 spec 01 §6.3** — Post-hoc segment + bell-curve aggregation by unified evaluator (§8.1 anchor; MAPPOAlgorithm leaves these empty).
- **pkg-08 spec 05** — Sweep harness enumerates `cfg.baselines.external_lr_sweep_grid["external_mappo"]` × seeds (§9.2 anchor).

### 14.4 Vendoring ground-truth anchors

- `D:\RL\lzj\MAPPO\MAPPO_main.py:11-69` — `Runner_MAPPO_MPE` class shape (§3 port table source).
- `D:\RL\lzj\MAPPO\mappo.py:111-308` — `MAPPO_MPE` class body (§3 keep list).
- `D:\RL\lzj\MAPPO\replay_buffer.py` — Episode-rollout buffer layout (§3 port to `_episodic_buffer.py`).
- `D:\RL\lzj\MAPPO\normalization.py` — `Normalization` + `RewardScaling` (§3 port to `_normalization.py`).

---

## Anchors (verbatim grep targets for spec 08 §6 drift detector)

- `pkg-07 spec 05 §1: Tier-1 external policy-gradient baseline (MAPPO)`
- `pkg-07 spec 05 §2: vendored from D:\RL\lzj\MAPPO\ — local, no external license`
- `pkg-07 spec 05 §3: drop MPE-specific code; port MAPPO_MPE → MAPPOAlgorithm`
- `pkg-07 spec 05 §4: MAPPOAlgorithm implements ExternalBaselineRunner (train/evaluate/save/load)`
- `pkg-07 spec 05 §5: centralized critic consumes concat([obs_i for i in agents]) — legal CTDE global state`
- `pkg-07 spec 05 §5.2: _FORBIDDEN_INFO_KEYS = {c_true, types, resource_state, hotspot_centers}`
- `pkg-07 spec 05 §6: share_policy default True (lzj verbatim, NOT on cfg.baselines)`
- `pkg-07 spec 05 §7: discrete A=6 via Categorical (training) + argmax (eval); no Gumbel-softmax`
- `pkg-07 spec 05 §8: evaluate() returns EvalReport per pkg-08 spec 01 §6 external delegation`
- `pkg-07 spec 05 §9: LR sweep grid (1e-4, 3e-4, 1e-3) × 5 seeds = 15 runs per preset`
- `pkg-07 spec 05 §10: C7-EXT-SMOKE1 test — Easy 20K env steps, p<0.05 over 3 seeds vs random`
- `pkg-07 spec 05 §11: per-impl tuning constants (20 defaults) NOT in cfg.baselines`
- `pkg-07 spec 05 §12: 5 named tests (factory dispatch / adapter consumption / evaluate contract / smoke / LR sweep dispatch)`
- `pkg-07 spec 05 §13: integration hooks — spec 01/04/07/08 + pkg-08 spec 01/05`
