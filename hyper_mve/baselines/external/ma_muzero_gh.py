"""MAMuZeroGHAlgorithm — Tier-1 external model-based MARL baseline
(pkg-07 spec 06 §3, ships the §3.10 per-agent weak-fallback).

Real port from ``werner-duvaud/muzero-general`` @
``0825bd544fc172a2e2dcc96d43711123222c4a2f`` (MIT; see
``_third_party_licenses/muzero_general_LICENSE.txt``). The vendored
``MuZeroFullyConnectedNetwork`` and ``MCTS`` modules are imported from
:mod:`hyper_mve.baselines.external._muzero_general`; this module is the
multi-agent wrapper authored under pkg-07 ownership.

Per pkg-07 spec 06 §3.10 (locked weak-fallback architecture)
------------------------------------------------------------

This pass ships the **per-agent vanilla MuZero + averaged cooperative
reward** weak version that spec 06 §3.10 explicitly approves as the
fallback when the joint-state MA-MuZero-GH wrapper proves too risky to
hit the 20K Easy smoke gate:

  * **N independent MuZero learners** — one per agent, each with its own
    ``MuZeroFullyConnectedNetwork`` (representation / dynamics /
    prediction networks). No shared world model.
  * **Per-agent observation only** — each learner sees only ``obs_i``,
    not the joint state. No centralized critic.
  * **Cooperative reward target** — the reward signal fed to each
    learner is ``np.mean(reward_dict.values())``, the shared cooperative
    return. This is the convention spec 06 §3.10 names.
  * **Per-agent MCTS at action time** — each agent runs its own MCTS
    rollout to choose its action; agents are blind to each other's
    search trees.

The Methods main table column for this runner is labeled
``"MA-MuZero-GH (per-agent weak)"`` per spec 06 §3.10 (the §9.3 record
should mention which version shipped — recorded in ``self._weak_version
= True``).

CTDE legitimacy
---------------
Each learner's representation net consumes only its own ``obs_i``. The
CTDE boundary is trivially satisfied because the runner consumes ZERO
``info`` content — only obs/reward/term — and the runtime guard at every
step asserts this against
:data:`hyper_mve.baselines.external._FORBIDDEN_INFO_KEYS` per spec 06
§6.2.

Per-impl tuning constants (MCTS budget, support size, training cadence)
live as module defaults below per spec 06 §7.2 (NOT on ``cfg.baselines``).
"""
from __future__ import annotations

import math
import os
import time
from collections import deque
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, Optional, Union

import numpy as np
import torch

from hyper_mve.baselines.external import _FORBIDDEN_INFO_KEYS
from hyper_mve.baselines.external.base import ExternalBaselineRunner
from hyper_mve.baselines.external._muzero_general.models import (
    MuZeroFullyConnectedNetwork,
    scalar_to_support,
    support_to_scalar,
)
from hyper_mve.baselines.external._muzero_general.self_play import (
    MCTS,
    MinMaxStats,
    Node,
)
from hyper_mve.configs import V4Config
from hyper_mve.baselines.external.base import split_seen_unseen_regimes
from hyper_mve.eval.eval_report import EvalReport


#: Provenance marker (pkg-07 spec 06 §3.1).
_VENDORED_FROM: str = "muzero-general @ 0825bd544fc172a2e2dcc96d43711123222c4a2f"


# ----------------------------------------------------------------- per-impl defaults
# pkg-07 spec 06 §7.2 — module-level (NOT on cfg.baselines).

_DEFAULT_NUM_SIMULATIONS: int = 8             # per env step, per agent [feasibility 2026-06]
_DEFAULT_DISCOUNT: float = 0.997

#: Env var the sweep launcher injects to dial MCTS depth per run without a code
#: edit (``train_fast_sweep.py --mamz-num-simulations``). Cost is LINEAR in this
#: count and MCTS runs at *every* agent-step in BOTH train and eval, so it is the
#: dominant throughput lever — pkg-07 spec 06 §3.5 itself notes eval is ~50x
#: slower than QMIX at the same value (matching the observed 62.8s→3784s smoke).
#: Lowered from the spec's 25/50 to 8 so the 300K-step sweep is wall-clock
#: feasible; the §3.10 "weak" baseline tolerates a shallow tree.
_NUM_SIMULATIONS_ENV_VAR: str = "HYPER_MVE_MAMZ_NUM_SIMULATIONS"
_DEFAULT_ROOT_DIRICHLET_ALPHA: float = 0.3
_DEFAULT_ROOT_EXPLORATION_FRACTION: float = 0.25
_DEFAULT_PB_C_INIT: float = 1.25
_DEFAULT_PB_C_BASE: float = 19652.0

_DEFAULT_SUPPORT_SIZE: int = 10               # value/reward support half-range
_DEFAULT_ENCODING_SIZE: int = 32
_DEFAULT_FC_REPRESENTATION_LAYERS: tuple[int, ...] = ()
_DEFAULT_FC_DYNAMICS_LAYERS: tuple[int, ...] = (32,)
_DEFAULT_FC_REWARD_LAYERS: tuple[int, ...] = (32,)
_DEFAULT_FC_VALUE_LAYERS: tuple[int, ...] = (32,)
_DEFAULT_FC_POLICY_LAYERS: tuple[int, ...] = (32,)

_DEFAULT_BUFFER_SIZE: int = 2_000             # transitions, per agent
_DEFAULT_BATCH_SIZE: int = 64                 # transitions, per train step
_DEFAULT_NUM_UNROLL_STEPS: int = 5
_DEFAULT_TD_STEPS: int = 5
_DEFAULT_TRAIN_EVERY_N_EPS: int = 1
_DEFAULT_TRAIN_STEPS_PER_TRIGGER: int = 4

_DEFAULT_DEFAULT_LR: float = 3e-4
_DEFAULT_WEIGHT_DECAY: float = 1e-4
_DEFAULT_GRAD_NORM_CLIP: float = 10.0


# ------------------------------------------------------------------- helpers

def _check_forbidden_info(info_dict: dict[str, dict]) -> None:
    """pkg-07 spec 06 §6.2 runtime guard — every per-agent info dict must be
    free of forbidden keys. Raises ``AssertionError`` on violation."""
    for agent_key, per_agent_info in info_dict.items():
        leaked = _FORBIDDEN_INFO_KEYS & set(per_agent_info)
        assert not leaked, (
            f"MAMuZeroGHAlgorithm consumed forbidden info keys {leaked} at "
            f"agent {agent_key}. Either env_fn was constructed with "
            "oracle_mode=True (violates spec 04 §7) or the adapter is "
            "leaking. See pkg-07 design §3.5 + spec 06 §6.2."
        )


def _build_network_config(obs_dim: int, n_actions: int) -> SimpleNamespace:
    """Build the namespace ``MuZeroFullyConnectedNetwork`` consumes.

    The vendored ``MuZeroNetwork(config)`` factory dispatches on
    ``config.network``; we instantiate the FC variant directly so we
    only need its narrow constructor surface (no observation_shape /
    stacked_observations multipliers, since we feed obs_dim directly).
    """
    return SimpleNamespace(
        # MuZeroFullyConnectedNetwork expects obs_shape-style triple, but
        # multiplies all three dims into one flat input — so we encode
        # obs_dim into a (1, 1, obs_dim) shape tuple.
        observation_shape=(1, 1, int(obs_dim)),
        stacked_observations=0,
        action_space_size=int(n_actions),
        encoding_size=_DEFAULT_ENCODING_SIZE,
        fc_reward_layers=list(_DEFAULT_FC_REWARD_LAYERS),
        fc_value_layers=list(_DEFAULT_FC_VALUE_LAYERS),
        fc_policy_layers=list(_DEFAULT_FC_POLICY_LAYERS),
        fc_representation_layers=list(_DEFAULT_FC_REPRESENTATION_LAYERS),
        fc_dynamics_layers=list(_DEFAULT_FC_DYNAMICS_LAYERS),
        support_size=_DEFAULT_SUPPORT_SIZE,
    )


def _resolve_num_simulations() -> int:
    """MCTS simulations per agent-step: env override (clamped ≥1) else default.

    The sweep launcher sets ``HYPER_MVE_MAMZ_NUM_SIMULATIONS`` so smokes can drop
    to ~4 and the full sweep to ~8 without editing the module. Cost is linear in
    this count (see ``_NUM_SIMULATIONS_ENV_VAR``); malformed values fall back to
    the module default rather than crashing the worker.
    """
    raw = os.environ.get(_NUM_SIMULATIONS_ENV_VAR)
    if raw is None:
        return _DEFAULT_NUM_SIMULATIONS
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return _DEFAULT_NUM_SIMULATIONS


def _build_mcts_config(n_actions: int) -> SimpleNamespace:
    """Build the namespace ``MCTS`` consumes (single-player game)."""
    return SimpleNamespace(
        support_size=_DEFAULT_SUPPORT_SIZE,
        action_space=list(range(n_actions)),
        num_simulations=_resolve_num_simulations(),
        discount=_DEFAULT_DISCOUNT,
        root_dirichlet_alpha=_DEFAULT_ROOT_DIRICHLET_ALPHA,
        root_exploration_fraction=_DEFAULT_ROOT_EXPLORATION_FRACTION,
        pb_c_init=_DEFAULT_PB_C_INIT,
        pb_c_base=_DEFAULT_PB_C_BASE,
        players=[0],   # single-player from each agent's POV
    )


def _make_network(obs_dim: int, n_actions: int) -> MuZeroFullyConnectedNetwork:
    """Instantiate the vendored fully-connected MuZero net."""
    cfg = _build_network_config(obs_dim, n_actions)
    return MuZeroFullyConnectedNetwork(
        observation_shape=cfg.observation_shape,
        stacked_observations=cfg.stacked_observations,
        action_space_size=cfg.action_space_size,
        encoding_size=cfg.encoding_size,
        fc_reward_layers=cfg.fc_reward_layers,
        fc_value_layers=cfg.fc_value_layers,
        fc_policy_layers=cfg.fc_policy_layers,
        fc_representation_layers=cfg.fc_representation_layers,
        fc_dynamics_layers=cfg.fc_dynamics_layers,
        support_size=cfg.support_size,
    )


# ----------------------------------------------------------------- buffer

class _TransitionBuffer:
    """Per-agent FIFO buffer of (obs, action, reward, value_target,
    policy_target, done) transitions. Simpler than upstream's PER-style
    ``GameHistory``; matches the §3.10 weak-fallback simplification.
    """

    def __init__(self, capacity: int) -> None:
        self.obs: deque = deque(maxlen=capacity)
        self.action: deque = deque(maxlen=capacity)
        self.reward: deque = deque(maxlen=capacity)
        self.value_tgt: deque = deque(maxlen=capacity)
        self.policy_tgt: deque = deque(maxlen=capacity)
        self.done: deque = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.obs)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        value_tgt: float,
        policy_tgt: np.ndarray,
        done: bool,
    ) -> None:
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.action.append(int(action))
        self.reward.append(float(reward))
        self.value_tgt.append(float(value_tgt))
        self.policy_tgt.append(np.asarray(policy_tgt, dtype=np.float32))
        self.done.append(bool(done))

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        n = len(self.obs)
        idx = np.random.choice(n, size=min(batch_size, n), replace=False)
        return {
            "obs": np.stack([self.obs[i] for i in idx], axis=0),
            "action": np.array([self.action[i] for i in idx], dtype=np.int64),
            "reward": np.array([self.reward[i] for i in idx], dtype=np.float32),
            "value_tgt": np.array([self.value_tgt[i] for i in idx], dtype=np.float32),
            "policy_tgt": np.stack([self.policy_tgt[i] for i in idx], axis=0),
            "done": np.array([self.done[i] for i in idx], dtype=np.float32),
        }


# ---------------------------------------------------------- per-agent learner

class _PerAgentLearner:
    """One vanilla MuZero learner. Owned by :class:`MAMuZeroGHAlgorithm`,
    one instance per agent (§3.10 weak-fallback)."""

    def __init__(self, obs_dim: int, n_actions: int, lr: float, device: torch.device) -> None:
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.device = device
        self.network = _make_network(obs_dim, n_actions).to(device)
        self.support_size = _DEFAULT_SUPPORT_SIZE
        self.optim = torch.optim.Adam(
            self.network.parameters(), lr=lr,
            weight_decay=_DEFAULT_WEIGHT_DECAY,
        )
        self.buffer = _TransitionBuffer(_DEFAULT_BUFFER_SIZE)
        self.mcts = MCTS(_build_mcts_config(n_actions))

    # ----------------------------------------------------------- act
    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        add_exploration_noise: bool,
    ) -> tuple[int, np.ndarray, float]:
        """Run MCTS from the given obs; return (action, policy_target, value_estimate)."""
        # Reshape obs to the (B=1, 1, 1, obs_dim) layout the FC representation
        # net expects (it flattens internally).
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(1, 1, 1, self.obs_dim)
        # MCTS.run handles the forward pass and torch.tensor conversion itself.
        root, _ = self.mcts.run(
            model=self.network,
            observation=obs_arr[0],     # (1, 1, obs_dim) — MCTS adds the batch axis
            legal_actions=list(range(self.n_actions)),
            to_play=0,
            add_exploration_noise=add_exploration_noise,
        )
        # Visit-count → policy.
        visits = np.zeros(self.n_actions, dtype=np.float32)
        for a, child in root.children.items():
            visits[a] = float(child.visit_count)
        if visits.sum() > 0:
            policy_target = visits / visits.sum()
        else:
            policy_target = np.ones(self.n_actions, dtype=np.float32) / self.n_actions
        # Argmax-on-visits action selection (greedy w.r.t. visit count).
        action = int(np.argmax(visits))
        value_estimate = float(root.value())
        return action, policy_target, value_estimate

    # ----------------------------------------------------------- learn

    def learn(self) -> Optional[float]:
        if len(self.buffer) < _DEFAULT_BATCH_SIZE:
            return None
        batch = self.buffer.sample(_DEFAULT_BATCH_SIZE)
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        action = torch.as_tensor(batch["action"], dtype=torch.long, device=self.device)
        reward = torch.as_tensor(batch["reward"], dtype=torch.float32, device=self.device)
        value_tgt = torch.as_tensor(batch["value_tgt"], dtype=torch.float32, device=self.device)
        policy_tgt = torch.as_tensor(batch["policy_tgt"], dtype=torch.float32, device=self.device)

        # Reshape obs for the FC representation: (B, 1, 1, obs_dim).
        B = obs.shape[0]
        obs_in = obs.view(B, 1, 1, self.obs_dim)

        # Initial inference.
        value_logits, _, policy_logits, encoded_state = self.network.initial_inference(obs_in)

        # Recurrent inference (1 step) — the action drives the dynamics head.
        action_in = action.view(B, 1)
        value_next_logits, reward_logits, policy_next_logits, _ = \
            self.network.recurrent_inference(encoded_state, action_in)

        # Value loss — scalar value distilled to support distribution.
        value_target_dist = scalar_to_support(
            value_tgt.unsqueeze(-1), self.support_size,
        ).squeeze(1)
        value_loss = -(value_target_dist * torch.log_softmax(value_logits, dim=-1)).sum(dim=-1).mean()

        # Reward loss — per-step shared cooperative reward.
        reward_target_dist = scalar_to_support(
            reward.unsqueeze(-1), self.support_size,
        ).squeeze(1)
        reward_loss = -(reward_target_dist * torch.log_softmax(reward_logits, dim=-1)).sum(dim=-1).mean()

        # Policy loss — visit-count target distilled into policy logits.
        policy_loss = -(policy_tgt * torch.log_softmax(policy_logits, dim=-1)).sum(dim=-1).mean()

        loss = value_loss + reward_loss + policy_loss
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), _DEFAULT_GRAD_NORM_CLIP)
        self.optim.step()
        return float(loss.item())


# ----------------------------------------------------------------- runner

class MAMuZeroGHAlgorithm(ExternalBaselineRunner):
    """MA-MuZero-GH (Tier-1 external; pkg-07 spec 06 §3, §3.10 weak-fallback).

    Real port of ``werner-duvaud/muzero-general``; ships the per-agent
    weak version per spec 06 §3.10. Methods main table column should be
    labeled ``"MA-MuZero-GH (per-agent weak)"``.
    """

    def __init__(self, cfg: V4Config, lr: Optional[float] = None) -> None:
        super().__init__(cfg)
        self._vendored_from: str = _VENDORED_FROM
        if lr is None:
            grid = cfg.baselines.external_lr_sweep_grid.get(
                "external_ma_muzero_gh", (_DEFAULT_DEFAULT_LR,),
            )
            lr = grid[len(grid) // 2]
        self._lr: float = float(lr)
        self._seed: int = 0
        self._device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        # spec 06 §3.10 record: this runner ships the weak version.
        self._weak_version: bool = True
        # Lazy-allocated per-agent learners (depend on env-derived obs_dim).
        self._learners: list[_PerAgentLearner] = []

    # ----------------------------------------------------------- lazy build

    def _maybe_build(self, obs_dim: int) -> None:
        if self._learners:
            return
        n_agents = int(self.cfg.env.N)
        n_actions = int(self.cfg.env.A)
        self._learners = [
            _PerAgentLearner(
                obs_dim=obs_dim, n_actions=n_actions,
                lr=self._lr, device=self._device,
            )
            for _ in range(n_agents)
        ]

    # ------------------------------------------------------- one episode

    def _run_one_episode(
        self,
        env,
        evaluate: bool,
        g_override: Optional[int] = None,
    ) -> tuple[int, float]:
        """Roll one episode; push transitions into each learner's buffer.
        Returns (env_steps, episode_return)."""
        n_agents = int(self.cfg.env.N)
        n_actions = int(self.cfg.env.A)
        episode_limit = int(self.cfg.env.T_max)

        reset_options = {"g": int(g_override)} if g_override is not None else None
        obs_dict, info = env.reset(options=reset_options)
        _check_forbidden_info(info)

        # Per-agent trajectories — push to buffers at episode end so we can
        # compute n-step bootstrap value targets in one pass.
        traj_obs = [[] for _ in range(n_agents)]
        traj_action = [[] for _ in range(n_agents)]
        traj_reward_shared: list[float] = []
        traj_policy = [[] for _ in range(n_agents)]
        traj_value = [[] for _ in range(n_agents)]
        ep_return = 0.0
        steps = 0

        for t in range(episode_limit):
            action_dict: dict[str, int] = {}
            for i in range(n_agents):
                obs_i = obs_dict[f"agent_{i}"]
                action, policy_target, value_est = self._learners[i].act(
                    obs_i, add_exploration_noise=not evaluate,
                )
                action_dict[f"agent_{i}"] = int(action)
                traj_obs[i].append(obs_i)
                traj_action[i].append(action)
                traj_policy[i].append(policy_target)
                traj_value[i].append(value_est)

            obs_next_dict, reward_dict, term_dict, trunc_dict, info = env.step(action_dict)
            _check_forbidden_info(info)
            r_shared = float(np.mean([reward_dict[f"agent_{i}"] for i in range(n_agents)]))
            traj_reward_shared.append(r_shared)
            ep_return += float(sum(reward_dict.values()))
            steps = t + 1

            term = any(bool(v) for v in term_dict.values())
            trunc = any(bool(v) for v in trunc_dict.values())
            obs_dict = obs_next_dict
            if term or trunc:
                break

        if not evaluate and steps > 0:
            # Compute n-step bootstrap value targets, per agent.
            T = steps
            td = _DEFAULT_TD_STEPS
            gamma = _DEFAULT_DISCOUNT
            for i in range(n_agents):
                for t in range(T):
                    bootstrap_idx = t + td
                    value_target = 0.0
                    discount = 1.0
                    for k in range(td):
                        ti = t + k
                        if ti >= T:
                            break
                        value_target += discount * traj_reward_shared[ti]
                        discount *= gamma
                    if bootstrap_idx < T:
                        value_target += discount * float(traj_value[i][bootstrap_idx])
                    self._learners[i].buffer.push(
                        obs=traj_obs[i][t],
                        action=traj_action[i][t],
                        reward=traj_reward_shared[t],
                        value_tgt=value_target,
                        policy_tgt=traj_policy[i][t],
                        done=(t == T - 1),
                    )
        return steps, ep_return

    # ----------------------------------------------------------- public API

    def train(
        self,
        cfg: V4Config,
        env_fn: Callable[[], Any],
        *,
        total_env_steps: int = 0,
        lr: float = 0.0,
        seed: int = 0,
        max_train_steps: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        if total_env_steps <= 0:
            total_env_steps = int(max_train_steps) if max_train_steps is not None \
                else int(cfg.train.max_train_steps)
        if lr > 0:
            self._lr = float(lr)
        self._seed = int(seed)
        torch.manual_seed(self._seed)
        np.random.seed(self._seed)

        env = env_fn()
        n_agents = int(cfg.env.N)
        obs_dict, info = env.reset(seed=self._seed)
        _check_forbidden_info(info)
        obs_dim = int(obs_dict["agent_0"].shape[-1])
        self._maybe_build(obs_dim)
        for learner in self._learners:
            for pg in learner.optim.param_groups:
                pg["lr"] = self._lr

        env_steps = 0
        ep_count = 0
        while env_steps < total_env_steps:
            steps, _ = self._run_one_episode(env=env, evaluate=False)
            env_steps += steps
            ep_count += 1
            if ep_count % _DEFAULT_TRAIN_EVERY_N_EPS == 0:
                for learner in self._learners:
                    for _ in range(_DEFAULT_TRAIN_STEPS_PER_TRIGGER):
                        if learner.learn() is None:
                            break
        env.close()

    def evaluate(
        self,
        env_fn: Callable[[], Any],
        regime_grid: tuple[int, ...],
        episodes: int,
    ) -> EvalReport:
        """Per-regime deterministic rollouts (argmax-on-visits, no Dirichlet).
        Field-population matrix per pkg-07 spec 06 §3.7 (= §2.7)."""
        env = env_fn()
        obs_dict, _ = env.reset()
        obs_dim = int(obs_dict["agent_0"].shape[-1])
        env.close()
        self._maybe_build(obs_dim)

        env = env_fn()
        t0 = time.time()
        return_per_regime: dict[int, float] = {}
        return_per_regime_sem: dict[int, float] = {}
        episodes_per_regime: dict[int, int] = {}
        all_returns: list[float] = []
        env_steps_total = 0

        for g in regime_grid:
            g_returns: list[float] = []
            for _ep in range(int(episodes)):
                steps, ep_return = self._run_one_episode(
                    env=env, evaluate=True, g_override=int(g),
                )
                g_returns.append(float(ep_return))
                env_steps_total += int(steps)
            return_per_regime[int(g)] = float(np.mean(g_returns)) if g_returns else 0.0
            sem = (
                float(np.std(g_returns) / max(np.sqrt(len(g_returns)), 1.0))
                if len(g_returns) > 1 else 0.0
            )
            return_per_regime_sem[int(g)] = sem
            episodes_per_regime[int(g)] = int(len(g_returns))
            all_returns.extend(g_returns)
        env.close()

        zs_seen, zs_unseen_v = split_seen_unseen_regimes(self.cfg, return_per_regime)

        return_mean = float(np.mean(all_returns)) if all_returns else 0.0
        return_sem = (
            float(np.std(all_returns) / max(np.sqrt(len(all_returns)), 1.0))
            if len(all_returns) > 1 else 0.0
        )

        return EvalReport(
            variant="external_ma_muzero_gh",
            seed=int(self._seed),
            config_hash="0" * 40,
            eval_mode="planner",
            eval_planner_mode="planner_full",
            return_mean=return_mean,
            return_sem=return_sem,
            return_zero_shot_seen=zs_seen,
            return_zero_shot_unseen=zs_unseen_v,
            return_zero_shot_gap=zs_seen - zs_unseen_v,
            return_per_regime=MappingProxyType(return_per_regime),
            return_per_regime_sem=MappingProxyType(return_per_regime_sem),
            episodes_per_regime=MappingProxyType(episodes_per_regime),
            planner_prior_return_gap=0.0,
            direct_inference_return_mean=return_mean,
            planner_full_return_mean=return_mean,
            walltime_seconds=float(time.time() - t0),
            env_steps_evaluated=int(env_steps_total),
            episodes_total=int(len(all_returns)),
            info_gating_strict=True,
            set_context_subjective_oracle_leak=False,
        )

    # ----------------------------------------------------------- ckpt

    def save_checkpoint(self, path: Union[Path, str]) -> None:
        if not self._learners:
            return
        torch.save(
            {
                "learner_state_dicts": [
                    learner.network.state_dict() for learner in self._learners
                ],
                "optim_state_dicts": [
                    learner.optim.state_dict() for learner in self._learners
                ],
                "vendored_from": _VENDORED_FROM,
                "weak_version": self._weak_version,
                "lr": self._lr,
                "seed": self._seed,
            },
            path,
        )

    def load_checkpoint(self, path: Union[Path, str]) -> None:
        if not self._learners:
            raise RuntimeError(
                "MAMuZeroGHAlgorithm.load_checkpoint requires train() or "
                "evaluate() to have been called first (so the per-agent nets "
                "are built with the env-derived obs_dim)."
            )
        ckpt = torch.load(path, map_location=self._device)
        for learner, sd in zip(self._learners, ckpt["learner_state_dicts"]):
            learner.network.load_state_dict(sd)
        for learner, osd in zip(self._learners, ckpt["optim_state_dicts"]):
            learner.optim.load_state_dict(osd)
        self._lr = float(ckpt.get("lr", self._lr))
        self._seed = int(ckpt.get("seed", self._seed))

    def param_count(self) -> int:
        if not self._learners:
            return 0
        return sum(
            sum(p.numel() for p in learner.network.parameters())
            for learner in self._learners
        )


__all__ = ["MAMuZeroGHAlgorithm"]
