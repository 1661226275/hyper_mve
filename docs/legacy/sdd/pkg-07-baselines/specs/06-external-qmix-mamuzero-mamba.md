# Spec 06 — External QMIX + MA-MuZero-GH + MAMBA + MARIE/GA stubs

> **Anchors**: design.md §3.4 (vendoring sources: QMIX = `oxwhirl/pymarl` Apache 2.0; MA-MuZero-GH = `werner-duvaud/muzero-general` MIT + thin MA wrapper; MAMBA = active sourcing) · design.md §D7 (Tier-1 selection + all consume same PettingZoo adapter) · design.md §D8 (MAMBA sourcing protocol: 2-day budget + paper repo → community fork → stub fallback) · design.md §D9 (MARIE/GA stubs: `NotImplementedError` + reserved CLI + not in main table) · design.md §3.5 (N-parametric adapter + two-flag info gate + CTDE legitimacy: concat-of-obs legal, `resource_state`/`hotspot_centers` privileged) · design.md §D5 (External 披露式 fairness) · design.md §D10 末段 (per-impl tuning constants stay in module defaults; `cfg.baselines` is cross-spec contract not per-impl dict) · README C7-EXT-FACT1 / C7-EXT-API1 / C7-EXT-ADPT1 / C7-EXT-ADPT2 / C7-EXT-SMOKE1 / C7-EXT-FAIR1 / C7-EXT-STUB1.
> **Status**: SDD only — describes the contracts for `hyper_mve/baselines/external/qmix.py`, `ma_muzero_gh.py`, `mamba.py`, and `stubs.py`, not the implementation.
> **Cross-refs**: spec 01 (factory dispatch + `EXTERNAL_REGISTRY` 6 keys + `MAMBAAlgorithm` `IS_SOURCED` class-binding mechanism at §3 line 139) · spec 04 (adapter `env_fn` contract + two-flag info gate + N-parametric agents + `Discrete(6)`) · spec 05 (MAPPO structurally analogous; this spec **inherits verbatim** the `_FORBIDDEN_INFO_KEYS` frozenset, `save_checkpoint` schema, `evaluate()` external-delegation field-population matrix, and smoke-test pattern) · spec 07 (披露式 fairness) · spec 08 (downstream patches — 4 CLI strings + cfg.baselines field) · pkg-08 spec 01 (`EvalReport` 32-field schema + §6 external delegation contract) · pkg-08 spec 05 (sweep harness consumes LR grid).

---

## Header — three hard locks

### Lock 1 — MAMBA sourcing protocol is declared HERE, not deferred

The MAMBA sourcing protocol (Day-1 search targets / Day-2 port evaluation / fallback to stub) is fully specified in §4 of THIS spec. It is **not** deferred to implementation. If pkg-07 SDD finalize is reached without §4 sourcing log being usable for an implementer who has never read the design.md, this spec has failed. The 2-day budget begins on the first implementation calendar day of Phase C' (README "实施期 Phase C'") and ends at end-of-day-2 regardless of whether a usable source has been found. The fallback — `IS_SOURCED: Final[bool] = False` + class binding `MAMBAAlgorithm = _MAMBAStub` at import time — is the mechanism inherited verbatim from spec 01 §3 line 139.

### Lock 2 — MARIE/GA stub class binding is declared HERE

`tests/baselines/external/test_stub_external_baselines.py` (named in pkg-07 README §"输出清单") expects that `create_baseline(cfg, "external_marie")` and `create_baseline(cfg, "external_ga")` raise `NotImplementedError` at the `__init__` call inside the factory (the factory itself does not raise; the stub class's `__init__` does). §5 of THIS spec locks the stub class shape so the test can be written without ambiguity: `MARIEStub.__init__` and `GAStub.__init__` each have a one-line `raise NotImplementedError(...)` body. There is no constructor work that completes; there is no instance to be returned. The CLI name is reserved for a future pkg-07.5 follow-up that may promote them (design D9 epilogue).

### Lock 3 — CTDE legitimacy boundary inherited verbatim from spec 05 §5

All three Tier-1 external runners specified here (QMIX, MA-MuZero-GH, and MAMBA-if-sourced) inherit **verbatim** the `_FORBIDDEN_INFO_KEYS = frozenset({"c_true", "types", "resource_state", "hotspot_centers"})` constant and the per-step runtime assertion shape from spec 05 §5.2. The constant is hoisted to a module-level import in `hyper_mve/baselines/external/__init__.py` (§6.1) so that all four runner modules (`mappo.py`, `qmix.py`, `ma_muzero_gh.py`, `mamba.py`) share one source of truth — if a future field is added to the env's `_oracle_fields` / `_eval_only_fields` markers, the synchronous edit is in `external/__init__.py`, not four places. The grep-level lint (no `ResourceCommonsEnv(` in any `external/*.py` outside type-only imports) is inherited from spec 04 §10 and re-tested in §12.2 below for each of the three Tier-1 runners.

---

## 1. Purpose + tier status table

This spec locks **three Tier-1 external baselines** + **MAMBA (Tier-2)** + **two permanent stubs (MARIE/GA)** in a single doc because:

1. **Shared port pattern** — QMIX (from PyMARL) and MA-MuZero-GH (from muzero-general) follow the same vendoring + class-ification + `ExternalBaselineRunner` adaptation pattern that spec 05 locks for MAPPO. Co-locating them keeps the four-runner family `(MAPPO, QMIX, MA-MuZero-GH, MAMBA)` reviewable as one consistent block.
2. **Shared sourcing risk surface** — MAMBA's active-sourcing protocol is operationally adjacent to MA-MuZero-GH (both are model-based MARL; both consume the muzero-general lineage one way or another). Keeping them in one spec lets reviewers compare port budgets directly.
3. **Shared CTDE / info-gating contract** — all three Tier-1 runners pin the same `_FORBIDDEN_INFO_KEYS` and the same adapter consumption pattern (§6). Repeating the contract in three sibling specs would risk drift; one spec with three §-blocks under a shared §6 is honest.
4. **Stub family co-location** — MARIE/GA are short, NotImplementedError-only, and structurally identical. Their entire contract fits in §5 of this spec without needing a separate file.

### 1.1 Tier status (locked)

| Runner | Tier | Vendoring source | License | Must ship? | Smoke gate? | LR sweep? | In Methods main table? |
|--------|------|------------------|---------|-----------|-------------|-----------|------------------------|
| **QMIX** | Tier-1 | `oxwhirl/pymarl` (commit hash placeholder) | Apache 2.0 | YES — blocks pkg-07 finalize | YES (§9) | YES (§8) | YES (9-column / column 8) |
| **MA-MuZero-GH** | Tier-1 | `werner-duvaud/muzero-general` (commit hash placeholder) + thin MA wrapper | MIT | YES — blocks pkg-07 finalize | YES (§9) | YES (§8) | YES (9-column / column 9) |
| **MAMBA** | Tier-2 (active sourcing) | TBD per §4 sourcing log; failure → stub | TBD | NO (2-day budget; fallback to stub) | YES iff `IS_SOURCED is True` | YES iff sourced | YES iff sourced (10th column) |
| **MARIE** | Stub (permanent) | n/a | n/a | n/a — `NotImplementedError` at `__init__` | n/a | n/a | NO (design D9 epilogue) |
| **GA** | Stub (permanent) | n/a | n/a | n/a — `NotImplementedError` at `__init__` | n/a | n/a | NO (design D9 epilogue) |

Tier semantics:

- **Tier-1 must ship** = `QMIXAlgorithm` and `MAMuZeroGHAlgorithm` final-return numbers appear in the Ch6 Methods main table. Failure to converge on the smoke gate is a Day-N blocker, not a soft warning (see §9 + §3.10 weak-fallback for MA-MuZero-GH).
- **Tier-2 active sourcing** = MAMBA's `IS_SOURCED` flag drives a module-level class-binding at import time. Sourcing failure does NOT block pkg-07 finalize; the stub is the legal fallback (design D8). If sourced, MAMBA is added as the 10th column of the Methods main table (design §3.1 canonical 量词).
- **Permanent stubs** = MARIE/GA reserve their CLI names for future promotion. They are never in the main table during pkg-07 (design D9). The `external_marie` / `external_ga` CLI strings are reachable through `train_main.py` (per spec 08 §"下游补丁声明") but raise `NotImplementedError` immediately at the factory call.

---

## 2. QMIX (Tier-1)

### 2.1 Vendoring source + license + commit hash placeholder

| Aspect | Value |
|--------|-------|
| Source | `https://github.com/oxwhirl/pymarl` |
| Source files consumed | `src/modules/agents/rnn_agent.py` (shared GRU agent net) · `src/modules/mixers/qmix.py` (QMixer class with state-dependent hypernet-style mixing weights) · `src/learners/q_learner.py` (target update + TD loss; SC-specific control flow dropped) · `src/components/action_selectors.py` (epsilon-greedy schedule) |
| License | Apache 2.0 (`LICENSE` file at repo root) — must be vendored alongside the code under `hyper_mve/baselines/external/_third_party_licenses/pymarl_LICENSE.txt` per pkg-07 license-propagation policy |
| Commit hash | `<LOCKED-AT-IMPLEMENTATION-TIME>` — Phase C (README "实施期") Day 1 captures `git log -1 oxwhirl/pymarl HEAD` and records the value into `hyper_mve/baselines/external/qmix.py` as `_VENDORED_FROM = "pymarl @ <hash>"`. The SDD does not pin a hash; the implementation does. |
| Upstream issue tracker | github.com/oxwhirl/pymarl/issues — checked once at Phase C Day 1 for any known correctness issues against the captured commit |

### 2.2 Port strategy: drop StarCraft glue + class-ify

PyMARL is a SMAC-coupled framework. The port keeps the **algorithm logic** intact (shared GRU agent net + state-dependent QMixer + double-Q + epsilon-greedy + target update) and **rewrites the surrounding plumbing** (SC2-specific obs / action / reward / done shaping, the YAML config system, the Sacred experiment logger, the multiprocess SC2 env workers):

| Concern | PyMARL source | pkg-07 `QMIXAlgorithm` |
|---------|---------------|------------------------|
| Hyperparameter ingress | Sacred YAML config dicts | `cfg: V4Config` + §7.1 per-impl defaults dict |
| Env construction | `env_REGISTRY[env_args["env"]](...)` (SC2 env class) | `env_fn: Callable[[], ResourceCommonsPettingZooEnv]` injected by spec 01 factory; spec 04 §10 contract |
| Agent count N | SC2 scenario-fixed | `env_fn()._env.N` read once at `train()` entry (N-parametric, env_cfg.N ∈ {2,4,8}) |
| Obs shape | SC2-specific (unit features) | per-agent obs from `env_fn().observation_space("agent_0").shape[0]` |
| Action space | SC2 discrete action set (per-unit; variable) | Fixed at `Discrete(6)` per agent per spec 04 §4 |
| Global state for mixer | SC2 `env.get_state()` (privileged) | **concat-of-obs**: `np.concatenate([obs_i for i in agents])` (spec 04 §5 + design §3.5 footnote — legal CTDE state) |
| Replay buffer | `EpisodeBatch` (SC2-coupled tensor layout) | port to `_episode_batch.py` (sibling module); episode_limit reads from `cfg.env.max_steps` |
| Target update | `target_update_interval` (steps) | preserved verbatim (§7.1) |
| Logger | Sacred + TB | sweep harness owns TB log dir (pkg-08 spec 05); Sacred dropped |
| SC2 multiprocess workers | `parallel_runner.py` | dropped — single-process episode collection (matches MAPPO §3 verbatim port; MAPPO didn't vector-env either) |

**Keep list** (algorithm logic that ports verbatim):
- `rnn_agent.py` — `RNNAgent(input_shape, args)`: Linear → GRUCell → Linear; per-agent obs → Q-values. The hidden-dim is `args.rnn_hidden_dim`; the per-impl default is `64` (§7.1).
- `qmix.py` — `QMixer(args)`: state-dependent hypernet producing positive-weighted monotonic mixing (the IGM principle). Mixer hidden dim default `32` (§7.1).
- `q_learner.py:train(...)` — TD-target loss with EMA target net + double-Q (`args.double_q=True`).
- Epsilon-greedy schedule from `action_selectors.py` — linear from `epsilon_start=1.0` to `epsilon_finish=0.05` over `epsilon_anneal_time` env steps.

**Drop list**:
- All SC2-specific env classes and parsers.
- Sacred experiment logger.
- YAML config loading.
- `parallel_runner.py` (multiprocess vector-env).
- `runners/episode_runner.py:run(test_mode=True)` separate eval branch — replaced by §2.7 `evaluate()`.

### 2.3 `QMIXAlgorithm` signature: implements `ExternalBaselineRunner` protocol

```python
# hyper_mve/baselines/external/qmix.py

from __future__ import annotations

from pathlib import Path
from typing import Callable

from hyper_mve.baselines._runner_protocol import ExternalBaselineRunner   # spec 01 §3.2
from hyper_mve.baselines.external import _FORBIDDEN_INFO_KEYS             # §6.1 hoisted
from hyper_mve.configs import V4Config
from hyper_mve.envs.adapters.pettingzoo_wrapper import ResourceCommonsPettingZooEnv
from hyper_mve.eval.eval_report import EvalReport                          # pkg-08 spec 01 §3

_VENDORED_FROM = "pymarl @ <commit-hash-locked-at-implementation>"


class QMIXAlgorithm(ExternalBaselineRunner):
    """Tier-1 external value-decomposition baseline.

    Implements ExternalBaselineRunner via the four-method protocol:
      - train(cfg, env_fn, *, total_env_steps, lr, seed) -> None
      - evaluate(env_fn, c_grid, episodes) -> EvalReport
      - save_checkpoint(path: Path | str) -> None
      - load_checkpoint(path: Path | str) -> None

    Centralized mixer consumes concat([obs_i for i in agents]) — the legal CTDE
    global state per pkg-07 design §3.5 footnote. NEVER touches info["c_true"],
    info["types"], info["resource_state"], or info["hotspot_centers"].
    """

    def __init__(self, cfg: V4Config) -> None: ...
    def train(self, cfg: V4Config, env_fn: Callable[[], ResourceCommonsPettingZooEnv],
              *, total_env_steps: int, lr: float, seed: int) -> None: ...
    def evaluate(self, env_fn: Callable[[], ResourceCommonsPettingZooEnv],
                 c_grid: tuple[float, ...], episodes: int) -> EvalReport: ...
    def save_checkpoint(self, path: Path | str) -> None: ...
    def load_checkpoint(self, path: Path | str) -> None: ...
```

The four-method protocol is verbatim from spec 05 §4 — keyword-only `total_env_steps` / `lr` / `seed` on `train`, `evaluate` returning a single `EvalReport`, `save_checkpoint` / `load_checkpoint` taking arbitrary paths.

### 2.4 Shared GRU agent net

```python
# Per pymarl rnn_agent.py + state shape contract
agent_inputs: torch.Tensor             # shape (B, N, obs_dim)
hidden: torch.Tensor                   # shape (B, N, rnn_hidden_dim)
q_values, new_hidden = self.agent_net(agent_inputs, hidden)
# q_values shape: (B, N, A=6)
```

The agent net is shared across all N agents (one parameter set; per-agent rows in the batch dimension). This is the canonical PyMARL pattern. For heterogeneous types (α/β capability split), the shared agent conditions on per-agent capability via the obs input (each agent's row already contains its own `cap_i` per spec 04 §5). No `add_agent_id` toggle in v0 — pymarl has one (`args.obs_agent_id`); the port leaves it at default `False` (§7.1) for parity with MAPPO §6.

### 2.5 Centralized mixer (CTDE legitimacy boundary)

```python
def _build_global_state(self, obs_dict: dict[str, np.ndarray]) -> np.ndarray:
    """Build the QMixer's global state.

    Per pkg-07 design §3.5 footnote + spec 05 §5.1 (inherited verbatim):
      LEGAL global state = concat([obs_i for i in agents]) ± action one-hots
                         ± caps (public).
      PRIVILEGED state = info["c_true"] / info["types"] / info["resource_state"]
                       / info["hotspot_centers"]. NEVER consumed.
    """
    return np.concatenate(
        [obs_dict[f"agent_{i}"] for i in range(self._N)], axis=0,
    )   # shape: (N * obs_dim,)
```

The mixer's `state_dim` is `self._N * obs_dim` (concat-of-obs), set once at `train()` entry. Per-step runtime assertion (inherited from spec 05 §5.2):

```python
obs_dict, reward_dict, term_dict, trunc_dict, info_dict = env.step(action_dict)
for agent_key, per_agent_info in info_dict.items():
    leaked = _FORBIDDEN_INFO_KEYS & set(per_agent_info)
    assert not leaked, (
        f"QMIXAlgorithm consumed forbidden info keys {leaked} at agent {agent_key}. "
        "Either env_fn was constructed with oracle_mode=True (which violates spec 04 §7) "
        "or the adapter is leaking. See design §3.5 CTDE boundary + spec 05 §5.2."
    )
```

Defensive runtime check — adapter (spec 04 §8 `_filter_info`) is the primary enforcement; this is the second line of defense + faster blame triangulation.

### 2.6 Action representation: `Discrete(6)` via epsilon-greedy (training) + argmax (eval)

Training-time:
```python
# Per pymarl action_selectors.py:EpsilonGreedyActionSelector
if np.random.rand() < self._epsilon(env_step):
    a_n = np.random.randint(0, 6, size=self._N)
else:
    with torch.no_grad():
        q_values, new_hidden = self.agent_net(agent_inputs, hidden)   # (1, N, 6)
        a_n = q_values.argmax(dim=-1).squeeze(0).cpu().numpy()
action_dict = {f"agent_{i}": int(a_n[i]) for i in range(self._N)}
```

Eval-time: same but with `epsilon=0` (pure argmax). Tie-break is `torch.argmax`'s deterministic lowest-index rule — sufficient for the §9 smoke gate's reproducibility. The adapter's strict-key contract (spec 04 §6) means QMIXAlgorithm must always pass A=6 actions for all N agents; the `NOOP=0` exists for this.

### 2.7 `evaluate()` implementation — external delegation field-population matrix

Per pkg-08 spec 01 §6 external delegation contract (inherited verbatim from spec 05 §8 field-population matrix; QMIX is structurally identical for these fields):

| `EvalReport` field group | QMIXAlgorithm populates | Reason |
|--------------------------|--------------------------|--------|
| Identity (6) | All 6, `eval_mode="planner"`, `eval_planner_mode="planner_full"` (legacy "runner's strongest mode"; cf. pkg-08 spec 01 §12 edge case) | Conventional choice for external runners with no MVE planner. |
| Headline (5: return_mean, return_sem, return_zero_shot_seen, return_zero_shot_unseen, return_zero_shot_gap) | All 5, computed from per-c rollouts. | Central metric. |
| Per-c (3: return_per_c, return_per_c_sem, episodes_per_c) | All 3 over `c_grid`. | Direct output of per-c loop. |
| c-segment (2) | EMPTY `MappingProxyType({})` | Per pkg-08 spec 01 §6.3, unified evaluator runs `_aggregate_segments` post-hoc on `return_per_c`. |
| Bell-curve (2) | EMPTY `MappingProxyType({})` | Same — pkg-08 spec 01 §6.3 post-hoc aggregation. |
| Regret (4: regret_per_c, regret_mean, oracle_ceiling_per_c, oracle_ceiling_cache_hit) | EMPTY `MappingProxyType({})` and `regret_mean=0.0` and `oracle_ceiling_per_c=MappingProxyType({})` and `oracle_ceiling_cache_hit=MappingProxyType({})` — `oracle_ceiling_per_c` field is None for non-hyper variants per spec 01 §6 (kept as empty MappingProxyType in the constructor; documented as nullable for external) | Regret needs oracle ceiling — hyper-only construct (pkg-08 spec 02 §3). External runners leave these for post-hoc regret pass. |
| Planner-prior gap (3) | `planner_prior_return_gap=0.0`, `direct_inference_return_mean=return_mean`, `planner_full_return_mean=return_mean` | Per pkg-08 spec 01 §6.2: no MVE planner → gap is 0; both modes identified with headline mean. |
| Diagnostics (3) | `walltime_seconds` + `env_steps_evaluated` + `episodes_total` real; `info_gating_strict=(env_fn()._oracle_mode is False)` — canonical expression per pkg-08 spec 01 §3.1 | Verified by `test_external_eval_contract.py` (§12.3). |
| Belief (2) | `None` for both | No belief head; pkg-08 spec 01 §6.2 allows. |
| Schema version (1) | `"pkg08-spec01-v1"` | Sentinel. |
| Oracle-leak flag (1: set_context_subjective_oracle_leak) | `False` | No `set_context_subjective` call path. |

Per-c loop pseudocode is structurally identical to spec 05 §8.2 with `MAPPOAlgorithm` → `QMIXAlgorithm` and the actor-call replaced by an epsilon=0 agent-net argmax call. The `_eval_seed_base` constant defaults to `42` (per-impl §7.1).

### 2.8 LR sweep contract

Per design D10 + spec 01 §6.1, `cfg.baselines.external_lr_sweep_grid: Mapping[str, tuple[float, ...]]` has the entry:

```python
"external_qmix": (1e-4, 3e-4, 1e-3),
```

(default in `configs/v4_config.py::BaselinesConfig`). The C7-EXT-FAIR1 minimum is ≥3 LRs × ≥3 seeds; pkg-07 implementation pins **5 seeds × 3 LRs = 15 runs per preset** for parity with MAPPO §9.2 + the Methods main table 5-seed convention. The pkg-08 spec 05 sweep harness enumerates 15 × 2 presets = 30 QMIX runs in the full sweep. Schedule: Easy preset cheap first → narrow to 1 best-LR-per-seed → run that LR on Medium with 5 seeds.

### 2.9 Smoke (C7-EXT-SMOKE1)

`tests/baselines/external/test_qmix_smoke.py`. Structurally identical to spec 05 §10.2 (MAPPO smoke test pattern inherited verbatim):
- Easy preset (N=2), 20K env-steps budget (`smoke_budget = 20_000`).
- 3 seeds `(0, 1, 2)`, single LR `lr = 3e-4` (middle of grid).
- Random baseline at step 0 (no training) vs QMIX trained for 20K env steps.
- One-sided Welch t-test, `alpha=0.05`. Failure is a Day-N blocker (§9 below).

Triage tree for failure inherited from spec 05 §10.6: env-determinism / 20K too short on this preset / 3-seed sample too small.

---

## 3. MA-MuZero-GH (Tier-1)

### 3.1 Vendoring source + license + commit hash placeholder

| Aspect | Value |
|--------|-------|
| Source | `https://github.com/werner-duvaud/muzero-general` |
| Source files consumed | `muzero.py` (top-level training loop class — heavily refactored for MA) · `models.py` (MuZeroNetwork: representation / dynamics / prediction heads) · `self_play.py` (MCTS implementation — `MCTS.run`) · `replay_buffer.py` (PER-style buffer with priority resampling) · `trainer.py` (reanalyze + target value computation) |
| License | MIT (`LICENSE` at repo root) — vendored alongside the code under `hyper_mve/baselines/external/_third_party_licenses/muzero_general_LICENSE.txt` |
| Commit hash | `<LOCKED-AT-IMPLEMENTATION-TIME>` — Phase C Day 1 captures `git log -1 werner-duvaud/muzero-general HEAD`. Constant `_VENDORED_FROM = "muzero-general @ <hash>"` recorded in `ma_muzero_gh.py`. |
| Upstream issue tracker | github.com/werner-duvaud/muzero-general/issues |

### 3.2 Port strategy: vendor single-agent + thin MA wrapper

The honest read of the MARL-MuZero literature (Phase A research note) is that community MARL forks of muzero-general are research-quality and frequently abandoned. **The strategy is therefore: vendor the single-agent muzero-general code intact + write a thin MA wrapper in pkg-07**, rather than depend on a third-party fork. The wrapper sits above the single-agent learner.

Why not a fork: forks fragment along scenarios (SMAC, MPE, MAgent), often hard-code 2-agent assumptions, and rarely preserve upstream's evaluation correctness. The thin wrapper is 200–400 LOC, reviewable, and explicitly under pkg-07 ownership.

### 3.3 MA wrapper architecture

```
ResourceCommonsPettingZooEnv (N agents, Discrete(6) per agent)
            ↓
MAMuZeroGHAlgorithm
    ├── world_model: MuZeroNetwork (shared across N agents)
    │       ├── representation: obs_joint → state
    │       ├── dynamics:       (state, action_joint) → (state', reward)
    │       └── prediction:     state → (value, policy_logits_joint)
    ├── per_agent_policy_heads: list[PolicyHead] (N heads, one per agent)
    ├── per_agent_value_heads:  list[ValueHead]  (N heads, one per agent)
    └── shared_buffer: ReplayBuffer (single buffer, multi-agent transitions)
```

Three design choices in this wrapper (locked):

1. **Shared world model** across N agents. The world model consumes the joint obs (concat-of-obs, spec 04 §5) and produces a joint state. Per-agent heads then read this state.
2. **Per-agent policy/value heads**. Each agent has its own policy head outputting `Discrete(6)` logits and its own value head outputting a scalar V. This allows heterogeneous behavior on heterogeneous α/β types without splitting the world model.
3. **Mean reward estimate across agents**. The dynamics head predicts a single scalar reward (interpreted as the mean reward across N agents). Per-agent rewards from the env are mean-aggregated for the target. This matches the standard MA-MuZero literature convention and keeps the head shape stable as N changes between presets (Easy=2 → Medium=4 → Hard=8).

This architecture is intentionally simpler than research MARL-MuZero variants. The aim is "honest convergent baseline", not "SOTA".

### 3.4 `MAMuZeroGHAlgorithm` signature: implements `ExternalBaselineRunner` protocol

```python
# hyper_mve/baselines/external/ma_muzero_gh.py

from __future__ import annotations

from pathlib import Path
from typing import Callable

from hyper_mve.baselines._runner_protocol import ExternalBaselineRunner
from hyper_mve.baselines.external import _FORBIDDEN_INFO_KEYS
from hyper_mve.configs import V4Config
from hyper_mve.envs.adapters.pettingzoo_wrapper import ResourceCommonsPettingZooEnv
from hyper_mve.eval.eval_report import EvalReport

_VENDORED_FROM = "muzero-general @ <commit-hash-locked-at-implementation>"


class MAMuZeroGHAlgorithm(ExternalBaselineRunner):
    """Tier-1 external model-based MARL baseline.

    Vendors muzero-general single-agent code + thin MA wrapper authored in pkg-07.
    World model shared across N agents; per-agent value/policy heads; mean reward
    estimate across agents (§3.3).

    Implements the four-method ExternalBaselineRunner protocol.
    """

    def __init__(self, cfg: V4Config) -> None: ...
    def train(self, cfg: V4Config, env_fn: Callable[[], ResourceCommonsPettingZooEnv],
              *, total_env_steps: int, lr: float, seed: int) -> None: ...
    def evaluate(self, env_fn: Callable[[], ResourceCommonsPettingZooEnv],
                 c_grid: tuple[float, ...], episodes: int) -> EvalReport: ...
    def save_checkpoint(self, path: Path | str) -> None: ...
    def load_checkpoint(self, path: Path | str) -> None: ...
```

The four-method shape is verbatim inherited from spec 05 §4 (same protocol, same kwargs).

### 3.5 MCTS budget — per-impl tuning constant (NOT in `cfg.baselines`)

Per design §D10 末段 (canonical wording): "Per-impl tuning constants (`external_smoke_max_env_steps` / `external_mappo_share_policy` / `external_qmix_mixer_hidden_dim` / `external_ma_muzero_gh_simulations`) live in spec 05/06 defaults — they are impl-internal, not cross-spec contract."

MCTS-related defaults in `ma_muzero_gh.py` (§7.2 below has the full list):
- `num_simulations: int = 50` — number of MCTS simulations per env step. Lower than muzero-general's `50–800` upstream default to fit the 20K env-step smoke budget (§9.2).
- `root_dirichlet_alpha: float = 0.3` — exploration noise at the MCTS root.
- `root_exploration_fraction: float = 0.25` — interpolation weight between policy prior and Dirichlet noise at root.
- `pb_c_init: float = 1.25`, `pb_c_base: float = 19652` — PUCT exploration constants (verbatim from muzero-general).

None of these go on `cfg.baselines` — the 5-field BaselinesConfig穷举 in spec 01 §6.1 explicitly excludes them. A future hyperparameter sweep over them would be a follow-up spec 06.1, not a `cfg.baselines` extension.

### 3.6 Consume adapter via `env_fn`; same `_FORBIDDEN_INFO_KEYS` assertion

`MAMuZeroGHAlgorithm` consumes `ResourceCommonsPettingZooEnv` via the `env_fn: Callable[[], ResourceCommonsPettingZooEnv]` injected by spec 01 factory (spec 04 §10 contract). NO direct `ResourceCommonsEnv(` construction — spec 04 §10 lint enforces (§12.2 below). Per-step runtime assertion inherited verbatim from spec 05 §5.2 / §2.5 above:

```python
obs_dict, reward_dict, term_dict, trunc_dict, info_dict = env.step(action_dict)
for agent_key, per_agent_info in info_dict.items():
    leaked = _FORBIDDEN_INFO_KEYS & set(per_agent_info)
    assert not leaked, f"MAMuZeroGHAlgorithm consumed forbidden info keys {leaked} ..."
```

The world-model representation head consumes `concat([obs_i for i in agents])` (legal CTDE per spec 04 §5 + design §3.5 footnote). It never reads `info["c_true"]`, `info["types"]`, `info["resource_state"]`, or `info["hotspot_centers"]`.

### 3.7 `evaluate()` — external delegation field-population matrix

Identical to QMIX §2.7 (which inherits from spec 05 §8.1). The field-population matrix is the same for all three Tier-1 external runners — the external-delegation contract in pkg-08 spec 01 §6 is uniform.

Two minor differences specific to MA-MuZero-GH evaluation:
1. Eval-time action selection runs **one MCTS rollout per step** (`num_simulations` as in §3.5; same as training because eval is deterministic). The c-grid loop wraps a planning step per env step rather than a single argmax over learned Q-values; throughput is ~50× slower than QMIX eval at the same `num_simulations`.
2. `walltime_seconds` will be substantially higher than QMIX/MAPPO for the same episode count — accept this as the cost of model-based eval. The §9.2 smoke gate sets `episodes=10` for the cheapest converging check.

### 3.8 LR sweep contract

```python
"external_ma_muzero_gh": (1e-4, 3e-4, 1e-3),
```

Same C7-EXT-FAIR1 minimum (≥3 LRs × ≥3 seeds; pkg-07 pins 5 seeds × 3 LRs = 15 runs per preset). The pkg-08 spec 05 sweep harness enumerates 15 × 2 presets = 30 MA-MuZero-GH runs in the full sweep. Schedule: Easy preset cheap first → narrow to 1 best LR → Medium with 5 seeds.

### 3.9 Smoke (C7-EXT-SMOKE1)

`tests/baselines/external/test_ma_muzero_gh_smoke.py`. Structurally identical to QMIX §2.9 + MAPPO spec 05 §10.2. Easy preset (N=2), 20K env-steps, 3 seeds, one-sided Welch t p<0.05 vs random baseline. The 20K budget is tight for a model-based learner with MCTS; if 20K proves insufficient on Easy in early implementation, the budget may be raised to 40K with a synchronous spec 06 §9 + README C7-EXT-SMOKE1 footnote edit.

### 3.10 Weak-fallback for MA-MuZero-GH (3-day implementation budget)

The thin MA wrapper (§3.3) carries non-trivial implementation risk: the joint world model + per-agent heads must converge in 20K Easy env steps. The implementer's 3-day Phase C budget for the MA wrapper is the explicit risk window.

**Fallback definition (locked here, not deferred)**: if at end-of-day-3 Phase C, the MA wrapper smoke gate has flat-lined return on 3+ consecutive attempts with different LRs (smoke budget extension to 40K already exhausted per §3.9), the implementer SHALL pivot to the **"per-agent vanilla MuZero + averaged reward" weak version**:

- N independent muzero-general single-agent learners, one per agent.
- Each agent's learner sees only its own per-agent observation `obs_i`, not the joint state.
- Reward signal to each learner is `np.mean(reward_dict.values())` (shared cooperative reward).
- No shared world model; no centralized critic; this is essentially N parallel single-agent MuZero runs with a shared training budget.

The weak version is structurally weaker than the §3.3 architecture and is honest about being so — its existence in spec 07 披露式 fairness table makes the comparison fair. The pivot is recorded in `spec 06 §3.10 implementation record` (an inline edit to this spec at implementation time, capturing date + reason + which weak version was shipped). The Methods main table footnote labels the 9th column "MA-MuZero-GH (per-agent weak)" if the weak version ships.

This fallback only triggers on smoke-gate flat-line, not on slow convergence or modest underperformance.

---

## 4. MAMBA (Tier-2 active sourcing)

### 4.1 Day-1 search targets (paper repo → community fork → benchmark)

The implementer SHALL search the following 4 channels on Phase C' Day 1 (README "实施期 Phase C'"; 1-day budget for the search itself; 1-day budget for port evaluation = 2-day total per design D8):

1. **Paper-linked GitHub** — locate the original MAMBA paper on arXiv via Google Scholar query `"MAMBA" "multi-agent reinforcement learning"` (note: distinguish from the unrelated state-space-model "Mamba" architecture which dominates 2024+ results; the MARL-MAMBA reference is Egorov & Shpilman 2022 era). Extract the GitHub link from the arXiv abstract footer or the paper's reproducibility appendix.
2. **Google Scholar citation chain** — examine works that cite the MAMBA paper for follow-up reproduction codes or community ports. Filter to last 3 years.
3. **Community forks under "mamba marl"** — GitHub search query `"mamba" "marl" OR "multi-agent"`; rank by `stars` and `last commit`. Skip any fork older than 2 years with <10 stars.
4. **MARL benchmark repos** — check `oxwhirl/epymarl` and `facebookresearch/BenchMARL` for MAMBA implementations; these benchmark repos often include canonical reference impls.

### 4.2 Day-2 port evaluation (minimal port size budget)

For each candidate source from Day-1, the implementer evaluates port feasibility against this budget:

- **Minimal port size target**: < 1000 LOC (excluding tests + license files). Above 1000 LOC, the integration cost likely exceeds the value of having MAMBA in the main table → mark as `Port-feasible? = No` → fallback to stub.
- **Dependency check**: no large RL frameworks (RLlib, SB3, JAX-Brax, etc.) — pkg-07 NG4 ("不引入 RLlib/SB3 等大型 RL 框架依赖"). Pure PyTorch is acceptable.
- **License check**: must be MIT / Apache-2.0 / BSD-3 / similar permissive. GPL is incompatible with pkg-07's umbrella license.
- **Architectural sanity**: source must clearly implement the MAMBA-paper algorithm (model-based + belief, world model + belief net coupling). A fork that only adapts MAMBA to one specific scenario and discards generality fails this check.

### 4.3 Sourcing log table format (filled by implementer)

The implementer SHALL fill this table inline in spec 06 §4.3 during Phase C' Day 1-2, even if the final outcome is "sourcing failed":

| Source URL | Commit | License | LOC | Port-feasible? | Notes |
|------------|--------|---------|-----|----------------|-------|
| (paper repo or fork URL) | (commit hash visited) | (license file declaration) | (LOC count) | YES / NO | (1-2 sentence reason: why feasible OR why blocked) |
| ... | ... | ... | ... | ... | ... |

Empty table at finalize = sourcing not yet attempted (illegal — Phase C' Day 1 forces the table to have ≥3 rows from §4.1 channels even if all are "Port-feasible? = NO"). The table is **preserved in spec 06 §4.3 even if every row is NO** — this preserves the search record for future re-attempts (e.g., a pkg-07.5 follow-up may revisit MAMBA when the community publishes a cleaner port).

### 4.4 If usable source found → vendor + write `MAMBAAlgorithm`

If at least one row in §4.3 has `Port-feasible? = YES`:

1. Vendor the source under `hyper_mve/baselines/external/mamba.py` + license file at `hyper_mve/baselines/external/_third_party_licenses/mamba_LICENSE.txt`.
2. Write `MAMBAAlgorithm(ExternalBaselineRunner)` matching the four-method protocol verbatim from spec 05 §4 / §2.3 above.
3. Add the LR sweep grid entry at vendor time: `cfg.baselines.external_lr_sweep_grid["external_mamba"] = (1e-4, 3e-4, 1e-3)` in `configs/v4_config.py::BaselinesConfig`.
4. Flip `IS_SOURCED: Final[bool] = True` in `hyper_mve/baselines/external/mamba.py` (the class-binding mechanism § 4.6).
5. Add `external_mamba` smoke test `tests/baselines/external/test_mamba_smoke.py` per §9.1 (gated on `IS_SOURCED is True`).

### 4.5 If sourcing fails (2-day timeout)

If end-of-Day-2 Phase C' the table in §4.3 has zero `Port-feasible? = YES` rows:

1. `IS_SOURCED: Final[bool] = False` in `hyper_mve/baselines/external/mamba.py`.
2. The stub class `_MAMBAStub`'s `__init__` raises:
   ```python
   raise NotImplementedError(
       "MAMBA sourcing failed; see spec 06 §4 sourcing log."
   )
   ```
3. NO LR sweep grid entry for `external_mamba` in `BaselinesConfig`. NO smoke test runs (smoke test file is gated on `IS_SOURCED is True` per §4.7).
4. `external_mamba` remains as a reachable CLI string (per spec 08 §"下游补丁声明" CLI extension). The factory call `create_baseline(cfg, "external_mamba")` raises `NotImplementedError` (the stub's `__init__`).
5. MAMBA does NOT appear as a column in the Methods main table (the 10th column slot is left empty; the table reverts to 9 columns).

### 4.6 `IS_SOURCED` toggle mechanism — verbatim from spec 01 §3 EXTERNAL_REGISTRY `"external_mamba"` row comment

(Referenced by section anchor, not line number, so re-pagination of spec 01 does not silently break this cross-reference.)

The mechanism is **module-level `Final[bool]` constant + class-binding at import-time**:

```python
# hyper_mve/baselines/external/mamba.py

from typing import Final
from hyper_mve.baselines._runner_protocol import ExternalBaselineRunner

IS_SOURCED: Final[bool] = False    # flipped to True at sourcing time per §4.4


class _MAMBAStub(ExternalBaselineRunner):
    """Stub fallback for MAMBAAlgorithm when sourcing fails (§4.5)."""

    def __init__(self, cfg) -> None:
        raise NotImplementedError(
            "MAMBA sourcing failed; see spec 06 §4 sourcing log."
        )

    # Protocol method stubs to satisfy `ExternalBaselineRunner` Protocol shape
    # — never reachable because __init__ raises. Bodies are explicit raise.
    def train(self, cfg, env_fn, *, total_env_steps, lr, seed): raise NotImplementedError
    def evaluate(self, env_fn, c_grid, episodes): raise NotImplementedError
    def save_checkpoint(self, path): raise NotImplementedError
    def load_checkpoint(self, path): raise NotImplementedError


class _RealMAMBA(ExternalBaselineRunner):
    """Vendored MAMBA impl, written at sourcing time per §4.4. Body
    deliberately elided in the SDD."""

    def __init__(self, cfg) -> None: ...
    # ... full four-method protocol body, structurally identical to QMIX §2.3 ...


# Class binding at import-time — this is the single point of MAMBA dispatch.
# spec 01 §3 EXTERNAL_REGISTRY "external_mamba" row comment mirrors this expression byte-identically:
#     "MAMBAAlgorithm = _RealMAMBA if IS_SOURCED else _MAMBAStub"
MAMBAAlgorithm = _RealMAMBA if IS_SOURCED else _MAMBAStub
```

Why class-binding-at-import (not runtime-dispatch in the factory): the factory `create_baseline` is a simple `REGISTRY[variant](cfg)` call (spec 01 §3.2); pushing the IS_SOURCED check into the factory would couple spec 01 to MAMBA's sourcing state. Class-binding-at-import keeps the factory pure and isolates the MAMBA-specific logic to `mamba.py`.

### 4.7 No smoke test if MAMBA is stub-only

`tests/baselines/external/test_mamba_smoke.py` exists at SDD time but its body is gated:

```python
import pytest
from hyper_mve.baselines.external.mamba import IS_SOURCED


@pytest.mark.skipif(not IS_SOURCED, reason="MAMBA not sourced; see spec 06 §4 sourcing log.")
def test_mamba_easy_smoke_return_beats_random_at_20k():
    """C7-EXT-SMOKE1 for MAMBA (gated on IS_SOURCED). Structurally identical
    to spec 05 §10.2 MAPPO smoke + §2.9 QMIX smoke + §3.9 MA-MuZero-GH smoke."""
    # ... verbatim adapted from spec 05 §10.2 with create_baseline(cfg, "external_mamba")
    ...
```

The skipif gate means CI passes regardless of MAMBA sourcing outcome — the test is reachable + structured but inert when stubbed.

### 4.8 Sourcing log preserved even if failed

The table in §4.3 SHALL be preserved in this spec file (in-place edit during Phase C' Day 1-2) regardless of outcome. Rationale: future pkg-07.5 / pkg-08.5 follow-ups may revisit MAMBA when (a) a community produces a cleaner port, (b) state-space-model "Mamba" matures to a MARL impl worth retrying, or (c) MARIE/GA are promoted alongside. The log is the institutional memory.

A spec 06 §4 amendment that "drops" the log table is forbidden by §12 anchors (the table header is a grep target for the spec 08 §6 drift detector).

---

## 5. MARIE + GA stubs

### 5.1 Reserved CLI names

Per pkg-07 spec 01 §2 `EXTERNAL_REGISTRY`:

```python
EXTERNAL_REGISTRY = MappingProxyType({
    "external_mappo":         MAPPOAlgorithm,
    "external_qmix":          QMIXAlgorithm,
    "external_ma_muzero_gh":  MAMuZeroGHAlgorithm,
    "external_mamba":         MAMBAAlgorithm,      # _RealMAMBA or _MAMBAStub per §4.6
    "external_marie":         MARIEStub,            # this spec §5.2
    "external_ga":            GAStub,               # this spec §5.2
})
```

The CLI strings `external_marie` and `external_ga` are reachable through `train_main.py` (per spec 08 §"下游补丁声明" CLI extension), but the factory call raises `NotImplementedError` at `__init__`.

### 5.2 Stub class binding

```python
# hyper_mve/baselines/external/stubs.py

from hyper_mve.baselines._runner_protocol import ExternalBaselineRunner


class MARIEStub(ExternalBaselineRunner):
    """Permanent stub. design D9 epilogue: 'MARIE/GA 不进 main table,
    不阻塞 pkg-07/08 包出;未来若做 pkg-07.5 follow-up 在那里 promote'."""

    def __init__(self, cfg) -> None:
        raise NotImplementedError(
            "MARIE/GA is a stub; not promoted in pkg-07. See pkg-07 design D9."
        )

    def train(self, cfg, env_fn, *, total_env_steps, lr, seed): raise NotImplementedError
    def evaluate(self, env_fn, c_grid, episodes): raise NotImplementedError
    def save_checkpoint(self, path): raise NotImplementedError
    def load_checkpoint(self, path): raise NotImplementedError


class GAStub(ExternalBaselineRunner):
    """Permanent stub. Same shape as MARIEStub; separate class to keep type
    identity for `isinstance` checks in tests + telemetry."""

    def __init__(self, cfg) -> None:
        raise NotImplementedError(
            "MARIE/GA is a stub; not promoted in pkg-07. See pkg-07 design D9."
        )

    def train(self, cfg, env_fn, *, total_env_steps, lr, seed): raise NotImplementedError
    def evaluate(self, env_fn, c_grid, episodes): raise NotImplementedError
    def save_checkpoint(self, path): raise NotImplementedError
    def load_checkpoint(self, path): raise NotImplementedError
```

Two **separate classes** (not a shared base or shared instance) so that `isinstance(runner, MARIEStub) vs isinstance(runner, GAStub)` is a meaningful check in tests and downstream code paths. The shared one-line message string is intentional.

### 5.3 Not in main Methods Comparison table

Per design D9 epilogue: MARIE/GA are NOT columns in the Ch6 Methods main table. The CLI strings exist for future-proofing; the runtime behavior is "skipped: NotImplementedError" in pkg-08 sweep harness (which detects the exception and records it in `runs/registry.jsonl` as a `status="skipped"` row — pkg-08 spec 05 owns this contract).

### 5.4 Parametrized stub test

```python
# tests/baselines/external/test_stub_external_baselines.py

import pytest
from hyper_mve.baselines import create_baseline
from hyper_mve.baselines.external.mamba import IS_SOURCED
from hyper_mve.configs import V4Config

_STUB_VARIANTS_BASE = ("external_marie", "external_ga")
_STUB_VARIANTS = _STUB_VARIANTS_BASE + (("external_mamba",) if not IS_SOURCED else ())


@pytest.mark.parametrize("variant", _STUB_VARIANTS)
def test_stub_baselines_raise_notimplementederror_at_init(variant: str):
    """C7-EXT-STUB1: stubs raise NotImplementedError at __init__.

    Covers:
      - external_marie (always stub; design D9)
      - external_ga (always stub; design D9)
      - external_mamba when IS_SOURCED is False (§4.5 fallback)

    The factory itself does NOT raise; the stub class's __init__ does.
    """
    cfg = V4Config(preset="easy")
    with pytest.raises(NotImplementedError):
        create_baseline(cfg, variant)
```

The conditional inclusion of `external_mamba` reflects §4.6 class-binding-at-import: when sourced, MAMBA is `_RealMAMBA` and this test should not parametrize over it (the smoke gate in §4.7 covers the real impl); when not sourced, MAMBA is `_MAMBAStub` and this test asserts the fallback.

---

## 6. CTDE legitimacy enforcement (verbatim inheritance from spec 05 §5)

### 6.1 `_FORBIDDEN_INFO_KEYS` hoisted to `external/__init__.py`

```python
# hyper_mve/baselines/external/__init__.py

from typing import Final

# Hoisted from spec 05 §5.2 — single source of truth for all four runner modules
# (mappo.py, qmix.py, ma_muzero_gh.py, mamba.py) + stubs (which never reach this
# constant because their __init__ raises). Future field additions to the env's
# _oracle_fields / _eval_only_fields markers require a synchronous edit HERE,
# not in four runner files.
_FORBIDDEN_INFO_KEYS: Final[frozenset[str]] = frozenset({
    "c_true", "types", "resource_state", "hotspot_centers",
})
```

All four runner modules (`mappo.py`, `qmix.py`, `ma_muzero_gh.py`, `mamba.py`) import this constant: `from hyper_mve.baselines.external import _FORBIDDEN_INFO_KEYS`. The literal `frozenset({...})` lives only in `__init__.py`.

### 6.2 Runtime assertion at every `env.step()`

Inherited verbatim from spec 05 §5.2. The assertion shape is identical across all three Tier-1 runners (QMIX, MA-MuZero-GH, MAMBA-if-sourced). The runtime assertion runs at every `env.step()` during training rollout AND at every `env.step()` during `evaluate()`. The cost is `O(N * |info_dict|)` per step — negligible compared to network forward.

### 6.3 Grep-level lint test (spec 04 §10 / spec 05 §12.2 inherited)

```python
# tests/baselines/external/test_external_adapter_consumption.py

from pathlib import Path
import pytest


@pytest.mark.parametrize("runner_module", [
    "qmix.py", "ma_muzero_gh.py", "mamba.py",
])
def test_runner_does_not_construct_resource_commons_env(runner_module: str):
    """spec 04 §10 lint inherited verbatim from spec 05 §12.2: grep for
    ResourceCommonsEnv( inside the runner source file."""
    src = Path("hyper_mve/baselines/external") / runner_module
    text = src.read_text()
    bad_patterns = ["ResourceCommonsEnv(", "ResourceCommonsEnv ("]
    for pat in bad_patterns:
        assert pat not in text, (
            f"{runner_module} contains {pat!r} — spec 04 §10 lint violation. "
            "env_fn is the only legal env construction site."
        )


@pytest.mark.parametrize("runner_module", [
    "qmix.py", "ma_muzero_gh.py", "mamba.py",
])
def test_runner_forbidden_info_keys_assertion_present(runner_module: str):
    """spec 05 §5.2 + §6.2 above: every Tier-1 runner asserts on _FORBIDDEN_INFO_KEYS."""
    src = Path("hyper_mve/baselines/external") / runner_module
    text = src.read_text()
    assert "_FORBIDDEN_INFO_KEYS" in text
```

---

## 7. Per-impl tuning constants (NOT in `cfg.baselines`)

Per design §D10 末段, the following constants live in the relevant runner module as module-level defaults — they are impl-internal, not cross-spec contract. The 5-field BaselinesConfig穷举 in spec 01 §6.1 explicitly excludes them.

### 7.1 QMIX defaults (`hyper_mve/baselines/external/qmix.py`)

```python
# Epsilon-greedy schedule
_DEFAULT_EPSILON_START: float = 1.0
_DEFAULT_EPSILON_FINISH: float = 0.05
_DEFAULT_EPSILON_ANNEAL_TIME: int = 50_000   # env steps for linear anneal

# Target update
_DEFAULT_TARGET_UPDATE_INTERVAL: int = 200   # train steps (EMA-style hard sync)

# Mixer
_DEFAULT_MIXER_HIDDEN_DIM: int = 32          # QMixer hypernet hidden dim (pymarl default)

# Double-Q
_DEFAULT_DOUBLE_Q: bool = True

# Optim / gradient
_DEFAULT_GRADIENT_CLIP_NORM: float = 10.0    # pymarl default
_DEFAULT_GAMMA: float = 0.99
_DEFAULT_BATCH_SIZE: int = 32                # episodes per train step
_DEFAULT_BUFFER_SIZE: int = 5000             # episode capacity

# Agent net
_DEFAULT_RNN_HIDDEN_DIM: int = 64
_DEFAULT_OBS_AGENT_ID: bool = False          # pymarl args.obs_agent_id (identity-blind by default)

# Eval-side
_DEFAULT_EVAL_SEED_BASE: int = 42
```

### 7.2 MA-MuZero-GH defaults (`hyper_mve/baselines/external/ma_muzero_gh.py`)

```python
# MCTS
_DEFAULT_NUM_SIMULATIONS: int = 50           # per env step; lowered from upstream 50-800 for 20K smoke
_DEFAULT_ROOT_DIRICHLET_ALPHA: float = 0.3
_DEFAULT_ROOT_EXPLORATION_FRACTION: float = 0.25
_DEFAULT_PB_C_INIT: float = 1.25
_DEFAULT_PB_C_BASE: float = 19652

# Loss weighting
_DEFAULT_VALUE_LOSS_WEIGHT: float = 1.0
_DEFAULT_REWARD_LOSS_WEIGHT: float = 1.0
_DEFAULT_POLICY_LOSS_WEIGHT: float = 1.0

# Buffer + train
_DEFAULT_BUFFER_SIZE: int = 10_000           # transitions
_DEFAULT_PER_ALPHA: float = 1.0              # PER priority exponent
_DEFAULT_PER_BETA: float = 1.0
_DEFAULT_BATCH_SIZE: int = 128
_DEFAULT_TD_STEPS: int = 5                   # n-step TD bootstrap depth
_DEFAULT_GAMMA: float = 0.997                # muzero-general default

# World model arch
_DEFAULT_REPRESENTATION_HIDDEN_DIM: int = 64
_DEFAULT_DYNAMICS_HIDDEN_DIM: int = 64
_DEFAULT_PREDICTION_HIDDEN_DIM: int = 64

# Eval-side
_DEFAULT_EVAL_SEED_BASE: int = 42
```

### 7.3 MAMBA (if sourced)

Per-impl defaults TBD at vendor time (§4.4). When sourcing succeeds, this §7.3 SHALL be filled in inline with the vendored algorithm's defaults — same format as §7.1 / §7.2. When sourcing fails, §7.3 remains "TBD at vendor time" as a placeholder.

`# §7.3: TBD-MAMBA-SOURCING-PENDING` — explicit grep-able sentinel marker so spec 08 §6 drift detector and any audit tooling can detect the unresolved placeholder; tooling SHALL emit a warning if this marker is still present after MAMBA is reported as sourced (status mismatch indicates spec 06 was not re-finalised after vendor success).

---

## 8. LR sweep contract (披露式 fairness, design §D5)

### 8.1 Grid source

All three Tier-1 runners use `cfg.baselines.external_lr_sweep_grid` keyed by their CLI string:

```python
# configs/v4_config.py::BaselinesConfig (locked at spec 01 §6.1)
external_lr_sweep_grid: Mapping[str, tuple[float, ...]] = field(
    default_factory=lambda: MappingProxyType({
        "external_mappo":         (1e-4, 3e-4, 1e-3),   # spec 05 §9.1
        "external_qmix":          (1e-4, 3e-4, 1e-3),   # §2.8 above
        "external_ma_muzero_gh":  (1e-4, 3e-4, 1e-3),   # §3.8 above
        # external_mamba added at vendor time per §4.4 if sourced;
        # external_marie / external_ga never swept (stubs).
    })
)
```

### 8.2 ≥3 LR × ≥3 seeds (C7-EXT-FAIR1)

The README C7-EXT-FAIR1 minimum is ≥3 LRs × ≥3 seeds. pkg-07 implementation pins **5 seeds × 3 LRs = 15 runs per preset** for parity with the Methods main table 5-seed convention.

### 8.3 Run schedule

The pkg-08 spec 05 sweep harness drives the schedule:
1. **Easy preset first** (cheaper N=2, shorter horizon) — 15 runs per Tier-1 baseline = 45 Tier-1 + 15 if MAMBA sourced = 45 or 60 Easy runs.
2. **Narrow to 1 best-LR-per-seed** by argmax over final `return_mean` per (LR, seed).
3. **Then Medium with best LR × 5 seeds** = 5 Medium runs per Tier-1 = 15 Medium + 5 if MAMBA sourced.

Total: 60–80 runs per spec 06 Tier-1 family (vs 30 for MAPPO alone in spec 05 — spec 06 is the larger budget).

### 8.4 Disclosure feed (spec 07 §"披露式" table)

For each Tier-1 runner, spec 07 receives:
- Final `return_mean` per (LR, seed) — extracted from `EvalReport.return_mean`.
- Wall-clock to converge — extracted from `EvalReport.walltime_seconds`.
- Parameter count — `param_count() -> int` method on each runner (the implementation owes this method; it returns `sum(p.numel() for p in self.<all_modules>.parameters())`).
- Best LR per seed — argmax over LR of final `return_mean`.

---

## 9. Smoke convergence (C7-EXT-SMOKE1)

### 9.1 Three smoke test files

- `tests/baselines/external/test_qmix_smoke.py` (§2.9) — always runs.
- `tests/baselines/external/test_ma_muzero_gh_smoke.py` (§3.9) — always runs.
- `tests/baselines/external/test_mamba_smoke.py` (§4.7) — gated on `IS_SOURCED is True`.

### 9.2 Common test pattern (inherited from spec 05 §10.2)

Easy preset (N=2). 20K env-steps budget (`smoke_budget = 20_000`). 3 seeds `(0, 1, 2)`. Single LR `lr = 3e-4` (middle of grid). One-sided Welch t-test, `alpha=0.05`. Random baseline at step 0 (no training) is the control.

The verbatim test body is in spec 05 §10.2; QMIX and MA-MuZero-GH and MAMBA smoke tests differ only in the `create_baseline(cfg, <variant>)` argument. Failure of any of the three is a Day-N implementation blocker.

### 9.3 Failure record table — "honest best-effort, appendix" downgrade

If at end-of-Phase-C implementation, any of QMIX / MA-MuZero-GH / MAMBA-if-sourced flat-lines the smoke gate after the §3.10 weak-fallback and 40K budget extension exhausted, the runner is **downgraded to "honest best-effort, appendix"** per design.md risk table:

| Runner | Outcome | Main table? | Appendix? |
|--------|---------|-------------|-----------|
| QMIX flat-lined | Document in spec 06 §9.3 record; record final achieved return + hyperparameter exploration log | NO (column dropped or labeled "did not converge") | YES (best-effort row with footnote) |
| MA-MuZero-GH flat-lined post weak-fallback | Document in spec 06 §3.10 record + §9.3 record | NO (or label "weak version, did not converge") | YES |
| MAMBA-if-sourced flat-lined | Document in spec 06 §9.3 record | NO (drop the 10th column) | YES (sourcing succeeded but convergence didn't) |

A flat-lined runner with its smoke test failing is NOT a "passing" test — the test fails red in CI, but the spec 06 §9.3 record makes the failure honest and the paper claim accurate ("we ran QMIX with full LR sweep; it did not converge above random on Easy within 20K env steps; the appendix shows the full trace"). This is the design.md §"risk table" honest path.

---

## 10. Test contract (C7-EXT-* enforcement) — 9 named tests

Each runner contributes 3 named tests (factory dispatch / adapter consumption / evaluate contract); the smoke + LR sweep dispatch tests are owned per runner. Total ≥9 named tests in `tests/baselines/external/` arising from this spec (the actual file count is lower because some tests are parametrized across runners).

### 10.1 QMIX tests

1. `test_factory_dispatches_external_qmix` (C7-EXT-FACT1) — in shared `test_registry.py` (per README §"输出清单"). Asserts `isinstance(runner, QMIXAlgorithm)` + `ExternalBaselineRunner` protocol surface present.
2. `test_qmix_consumes_adapter_only` (spec 04 §10 lint) — covered by `test_external_adapter_consumption.py` §6.3 parametrization.
3. `test_qmix_evaluate_contract` (C7-EXT-API1 + pkg-08 spec 01 §6) — in shared `test_external_eval_contract.py`. Inherits structure from spec 05 §12.3; differs only in `variant="external_qmix"` and the runner class type.

### 10.2 MA-MuZero-GH tests

4. `test_factory_dispatches_external_ma_muzero_gh` (C7-EXT-FACT1) — same shared `test_registry.py`.
5. `test_ma_muzero_gh_consumes_adapter_only` — parametrized in `test_external_adapter_consumption.py`.
6. `test_ma_muzero_gh_evaluate_contract` — shared `test_external_eval_contract.py`. Same structure as spec 05 §12.3.

### 10.3 MAMBA tests (gated on `IS_SOURCED`)

7. `test_factory_dispatches_external_mamba_if_sourced` — `test_registry.py`. If `IS_SOURCED is True`, asserts `isinstance(runner, _RealMAMBA)`; if `False`, asserts `pytest.raises(NotImplementedError)` (covered also by §5.4 parametrized stub test).
8. `test_mamba_consumes_adapter_only` — parametrized in `test_external_adapter_consumption.py` (skipif when not sourced).
9. `test_mamba_evaluate_contract` — `test_external_eval_contract.py` (skipif when not sourced).

### 10.4 Smoke tests + LR sweep dispatch tests

Three smoke files (§9.1) — `test_qmix_smoke.py`, `test_ma_muzero_gh_smoke.py`, `test_mamba_smoke.py`. Gated by `@pytest.mark.slow` (per spec 05 §10.1).

LR sweep dispatch test — single parametrized test in `tests/baselines/external/test_external_lr_sweep.py` that iterates over `("external_qmix", "external_ma_muzero_gh") + (("external_mamba",) if IS_SOURCED else ())` and asserts (a) the grid entry exists in `cfg.baselines.external_lr_sweep_grid`, (b) the grid has ≥3 entries, (c) `train()` has `lr` as a keyword-only arg. Inherits the test body from spec 05 §12.5 (MAPPO LR sweep dispatch).

### 10.5 Stub test

§5.4 — `test_stub_external_baselines.py` parametrized over `{external_marie, external_ga}` + conditionally `external_mamba` (C7-EXT-STUB1).

---

## 11. Integration hooks

| Consumer | Consumed | Use |
|----------|----------|-----|
| **spec 01** (factory entry) | `QMIXAlgorithm`, `MAMuZeroGHAlgorithm`, `MAMBAAlgorithm` (class-bound), `MARIEStub`, `GAStub` | `EXTERNAL_REGISTRY` 6 keys (§1.1) |
| **spec 04** (adapter `env_fn`) | `ResourceCommonsPettingZooEnv` consumed via `env_fn` in `train()` / `evaluate()` | §2.3/§2.5, §3.4/§3.6, §4.4/§4.6 |
| **spec 07** (披露式 fairness reporting) | `param_count()` method + final `return_mean` + `walltime_seconds` per (LR, seed) for each Tier-1 runner | spec 07 disclosure table feed |
| **spec 08** (downstream patches — CLI strings) | `external_qmix`, `external_ma_muzero_gh`, `external_mamba`, `external_marie`, `external_ga` CLI strings extending `train_main.py` `_DEFERRED_VARIANTS` | spec 08 §"下游补丁声明" |
| **pkg-08 spec 01** (`EvalReport` schema) | `EvalReport` constructed in each runner's `evaluate()` must conform to the 32-field schema | pkg-08 spec 01 §6 external delegation contract |
| **pkg-08 spec 05** (sweep harness) | `cfg.baselines.external_lr_sweep_grid["external_qmix"]` / `[..._ma_muzero_gh"]` / `[..._mamba"]` (if sourced) enumerated to drive 15 runs/preset/runner | pkg-08 spec 05 §"external sweep enumeration" |

### 11.1 What this spec does NOT add to `cfg.baselines`

Per design §D10 末段 and spec 01 §6.4: no per-impl fields. All §7.1 / §7.2 / §7.3 constants are module-level defaults in the runner files. Only `external_lr_sweep_grid["external_qmix"]` / `[..._ma_muzero_gh"]` / `[..._mamba"]` (cross-spec contract) are on `cfg.baselines`.

### 11.2 What this spec does NOT touch

- **NOT modified**: `hyper_mve/envs/resource_commons/*.py` (NG6).
- **NOT modified**: `hyper_mve/envs/adapters/pettingzoo_wrapper.py` (spec 04 owns it).
- **NOT modified**: any pkg-01..05 SDD spec (NG2).
- **NOT modified**: `oxwhirl/pymarl` / `werner-duvaud/muzero-general` / MAMBA upstream sources (vendoring is one-way).
- **NOT modified**: spec 05 (MAPPO) — this spec inherits structure but never edits spec 05.

---

## 12. Cross-references

### 12.1 Upstream anchors

- **design.md §3.4** — Vendoring source table (QMIX = pymarl Apache 2.0; MA-MuZero-GH = muzero-general MIT; MAMBA active sourcing).
- **design.md §3.5** — N-parametric adapter + two-flag info gating + CTDE legitimacy footnote (§2.5/§3.6/§4 / §6 anchor).
- **design.md §D5** — External 披露式 fairness (LR sweep + params + walltime + 5 seeds) (§8 anchor).
- **design.md §D7** — Tier-1 external selection (§1.1 anchor) + smoke convergence direction test 20K (§9 anchor).
- **design.md §D8** — MAMBA sourcing protocol 2-day budget + fallback (§4 anchor).
- **design.md §D9** — MARIE/GA stubs not in main table; reserved CLI; `NotImplementedError` (§5 anchor).
- **design.md §D10** — `cfg.baselines.external_lr_sweep_grid` mapping (§8 anchor); per-impl tuning constants exclusion (§7 anchor).
- **README C7-EXT-FACT1** — 3 Tier-1 instantiable (§10.1/§10.2 tests).
- **README C7-EXT-ADPT1 / ADPT2** — info-gating + N-parametric adapter (§6.3 lint).
- **README C7-EXT-API1** — `.evaluate(env_fn, c_grid, episodes) -> EvalReport` (§2.7/§3.7 anchor).
- **README C7-EXT-SMOKE1** — Easy preset return > random within 20K env steps (§9 anchor).
- **README C7-EXT-FAIR1** — LR sweep ≥3 LR × ≥3 seeds (§8 anchor).
- **README C7-EXT-STUB1** — stubs raise `NotImplementedError` (§5 + §4.5 anchor).

### 12.2 Sibling spec anchors

- **spec 01 §2 `EXTERNAL_REGISTRY` 6 keys** — class binding registry (§5.1 anchor; §4.6 MAMBA class binding anchor).
- **spec 01 §3 EXTERNAL_REGISTRY `"external_mamba"` row comment** — MAMBA `IS_SOURCED` class-binding mechanism quoted byte-identically (§4.6 anchor; referenced by section + symbol, not line number, so re-pagination of spec 01 cannot silently break this cross-reference).
- **spec 01 §6.1** — `cfg.baselines.external_lr_sweep_grid` 5-field BaselinesConfig穷举 (§8.1 anchor).
- **spec 04 §3** — N-parametric agents `f"agent_{i}" for i in range(self._N)` (§2.3/§3.4 anchor).
- **spec 04 §4** — `Discrete(6)` per-agent action space (§2.6 anchor).
- **spec 04 §5** — Per-agent obs space + `cap_i` public + CTDE legitimate concat (§2.5/§3.6 anchor).
- **spec 04 §6** — `reset` / `step` API + `options["c"]` forwarding (§2.7/§3.7 anchor).
- **spec 04 §7** — Two-flag info gating + default `(False, False)` (§6.2 anchor).
- **spec 04 §10** — External runner consumption contract: `env_fn` factory, `ResourceCommonsEnv(` lint (§6.3 anchor).
- **spec 05 §4** — `MAPPOAlgorithm` four-method protocol shape inherited verbatim by `QMIXAlgorithm` / `MAMuZeroGHAlgorithm` / `MAMBAAlgorithm` (§2.3/§3.4/§4.6 anchor).
- **spec 05 §5** — Centralized critic CTDE legitimacy boundary + `_FORBIDDEN_INFO_KEYS` (§2.5/§3.6/§6 anchor).
- **spec 05 §8** — `evaluate()` external-delegation field-population matrix inherited (§2.7/§3.7 anchor).
- **spec 05 §10** — Smoke test pattern inherited verbatim (§9.2 anchor).
- **spec 05 §12.5** — LR sweep dispatch test pattern inherited (§10.4 anchor).
- **spec 07 §"披露式"** — External fairness disclosure table (§8.4 anchor).
- **spec 08 §"3 处下游补丁声明"** — `train_main.py` CLI extension (`external_qmix` / `external_ma_muzero_gh` / `external_mamba` / `external_marie` / `external_ga` strings) + `cfg.baselines.external_lr_sweep_grid` consumption claim (§11 anchor).

### 12.3 Downstream pkg-08 anchors

- **pkg-08 spec 01 §3** — `@dataclass(frozen=True) EvalReport` 32-field schema (§2.7/§3.7 anchor).
- **pkg-08 spec 01 §6** — External delegation contract; field-population matrix (§2.7 anchor).
- **pkg-08 spec 01 §6.2** — External runners: `planner_prior_return_gap=0.0`, `direct_inference_return_mean=return_mean`, `planner_full_return_mean=return_mean`, `belief_c_mae=None`, `belief_c_calibration=None` (§2.7 anchor).
- **pkg-08 spec 01 §6.3** — Post-hoc segment + bell-curve aggregation by unified evaluator (§2.7 anchor).
- **pkg-08 spec 05** — Sweep harness enumerates `cfg.baselines.external_lr_sweep_grid["external_qmix"]` / `[..._ma_muzero_gh"]` / `[..._mamba"]` × seeds (§8.3 anchor).

### 12.4 Vendoring ground-truth anchors

- `oxwhirl/pymarl` HEAD as of pkg-07 Phase C Day 1 — `_VENDORED_FROM` hash captured (§2.1 anchor).
- `werner-duvaud/muzero-general` HEAD as of pkg-07 Phase C Day 1 — `_VENDORED_FROM` hash captured (§3.1 anchor).
- MAMBA — `_VENDORED_FROM` recorded at Phase C' Day 1-2 per §4.3 sourcing log table (TBD).

---

## Anchors (verbatim grep targets for spec 08 §6 drift detector)

- `pkg-07 spec 06 §1: Tier-1 external = QMIX (pymarl) + MA-MuZero-GH (muzero-general + MA wrapper); Tier-2 = MAMBA (active sourcing); stubs = MARIE + GA`
- `pkg-07 spec 06 §2: QMIX vendored from oxwhirl/pymarl Apache 2.0`
- `pkg-07 spec 06 §2.2: drop SC2 glue; port RNNAgent + QMixer + epsilon-greedy + double-Q + target update verbatim`
- `pkg-07 spec 06 §2.3: QMIXAlgorithm implements ExternalBaselineRunner (train/evaluate/save/load)`
- `pkg-07 spec 06 §2.5: centralized mixer consumes concat([obs_i for i in agents]) — legal CTDE global state`
- `pkg-07 spec 06 §2.5: _FORBIDDEN_INFO_KEYS = {c_true, types, resource_state, hotspot_centers} inherited from spec 05 §5.2`
- `pkg-07 spec 06 §2.6: discrete A=6 via epsilon-greedy (training) + argmax (eval)`
- `pkg-07 spec 06 §2.7: evaluate() returns EvalReport per pkg-08 spec 01 §6 external delegation`
- `pkg-07 spec 06 §2.8: LR sweep grid (1e-4, 3e-4, 1e-3) × 5 seeds = 15 runs per preset`
- `pkg-07 spec 06 §2.9: C7-EXT-SMOKE1 QMIX test — Easy 20K env steps, p<0.05 over 3 seeds vs random`
- `pkg-07 spec 06 §3: MA-MuZero-GH vendored from werner-duvaud/muzero-general MIT + thin MA wrapper`
- `pkg-07 spec 06 §3.3: shared world model + per-agent value/policy heads + mean reward estimate across agents`
- `pkg-07 spec 06 §3.4: MAMuZeroGHAlgorithm implements ExternalBaselineRunner`
- `pkg-07 spec 06 §3.5: num_simulations=50 default (per-impl tuning constant; NOT on cfg.baselines per design D10 末段)`
- `pkg-07 spec 06 §3.7: evaluate() identical field-population matrix to QMIX (external delegation pattern)`
- `pkg-07 spec 06 §3.8: LR sweep grid (1e-4, 3e-4, 1e-3) × 5 seeds = 15 runs per preset`
- `pkg-07 spec 06 §3.9: C7-EXT-SMOKE1 MA-MuZero-GH test — Easy 20K env steps`
- `pkg-07 spec 06 §3.10: 3-day implementation budget for MA wrapper; weak-fallback to per-agent vanilla MuZero + averaged reward`
- `pkg-07 spec 06 §4: MAMBA active sourcing protocol 2-day budget`
- `pkg-07 spec 06 §4.1: Day-1 search targets — paper repo / Google Scholar / community forks / MARL benchmarks`
- `pkg-07 spec 06 §4.2: Day-2 port evaluation — <1000 LOC budget; no large RL framework deps`
- `pkg-07 spec 06 §4.3: sourcing log table format — Source URL / Commit / License / LOC / Port-feasible / Notes`
- `pkg-07 spec 06 §4.5: sourcing failure fallback — IS_SOURCED=False; _MAMBAStub raises NotImplementedError`
- `pkg-07 spec 06 §4.6: IS_SOURCED Final[bool] mechanism — class binding at import-time MAMBAAlgorithm = _RealMAMBA if IS_SOURCED else _MAMBAStub`
- `pkg-07 spec 06 §4.7: no smoke test if MAMBA is stub-only (gated on IS_SOURCED is True)`
- `pkg-07 spec 06 §4.8: sourcing log preserved even if failed`
- `pkg-07 spec 06 §5: MARIE + GA permanent stubs — NotImplementedError at __init__; not in main Methods table per design D9`
- `pkg-07 spec 06 §5.1: reserved CLI names external_marie + external_ga`
- `pkg-07 spec 06 §5.2: MARIEStub + GAStub separate classes for isinstance check`
- `pkg-07 spec 06 §5.4: test_stub_external_baselines.py parametrized over {external_marie, external_ga, external_mamba-when-IS_SOURCED-False}`
- `pkg-07 spec 06 §6.1: _FORBIDDEN_INFO_KEYS hoisted to hyper_mve.baselines.external.__init__`
- `pkg-07 spec 06 §6.2: runtime assertion at every env.step() in all 3 Tier-1 runners`
- `pkg-07 spec 06 §6.3: grep-level lint test — no ResourceCommonsEnv( in external/*.py`
- `pkg-07 spec 06 §7: per-impl tuning constants in module defaults (NOT cfg.baselines per design D10 末段)`
- `pkg-07 spec 06 §8: cfg.baselines.external_lr_sweep_grid keyed by CLI string for all 3 Tier-1`
- `pkg-07 spec 06 §9: C7-EXT-SMOKE1 three smoke files (QMIX + MA-MuZero-GH + MAMBA-if-sourced)`
- `pkg-07 spec 06 §9.3: failure record table — flat-line downgrade to honest best-effort appendix`
- `pkg-07 spec 06 §10: 9 named tests covering FACT1/ADPT1/ADPT2/API1/SMOKE1/FAIR1/STUB1 for each runner`
- `pkg-07 spec 06 §11: integration hooks — spec 01/04/07/08 + pkg-08 spec 01/05`
