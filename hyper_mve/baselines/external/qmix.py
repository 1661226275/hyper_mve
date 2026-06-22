"""QMIXAlgorithm — Tier-1 external value-decomposition baseline (pkg-07 spec 06 §2).

Real port from ``oxwhirl/pymarl`` @ ``c971afdceb34635d31b778021b0ef90d7af51e86``
(Apache-2.0; see ``_third_party_licenses/pymarl_LICENSE.txt``). The vendored
``RNNAgent`` and ``QMixer`` modules are imported unmodified from
:mod:`hyper_mve.baselines.external._pymarl`; this module is the bridging
layer that:

  * builds the ``args`` namespace the upstream modules consume
    (rnn_hidden_dim, n_actions, mixing_embed_dim, etc.);
  * drives episode collection from
    :class:`hyper_mve.envs.adapters.pettingzoo_wrapper.ResourceCommonsPettingZooEnv`
    via the injected ``env_fn`` factory (no direct ``ResourceCommonsEnv``
    construction — pkg-07 spec 04 §10);
  * implements the Q-learning loss + double-Q + EMA target update body
    (port of ``q_learner.py:QLearner.train`` minus pymarl's
    ``EpisodeBatch``/``MAC``/Sacred plumbing);
  * implements ``train(cfg, env_fn, *, total_env_steps, lr, seed)`` and
    ``evaluate(env_fn, c_grid, episodes)`` matching the four-method
    :class:`ExternalBaselineRunner` protocol.

Per-impl tuning constants (epsilon schedule, mixer hidden dim, target update
interval, etc.) live as module defaults below per pkg-07 spec 06 §7.1
(NOT on ``cfg.baselines``).

CTDE legitimacy
---------------
The mixer's centralized state is ``concat([obs_i for i in agents])`` — the
legal CTDE global state per pkg-07 design §3.5 footnote (also reaffirmed
verbatim by spec 06 §2.5). Training-time ``info_dict`` is asserted free of
:data:`hyper_mve.baselines.external._FORBIDDEN_INFO_KEYS` at every
``env.step()`` (spec 06 §6.2 runtime guard).
"""
from __future__ import annotations

import copy
import time
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
from torch.optim import RMSprop

from hyper_mve.baselines.external import _FORBIDDEN_INFO_KEYS
from hyper_mve.baselines.external.base import ExternalBaselineRunner
from hyper_mve.baselines.external._pymarl.qmix_mixer import QMixer
from hyper_mve.baselines.external._pymarl.rnn_agent import RNNAgent
from hyper_mve.configs import V4Config
from hyper_mve.eval.eval_report import EvalReport


#: Provenance marker (pkg-07 spec 06 §2.1).
_VENDORED_FROM: str = "pymarl @ c971afdceb34635d31b778021b0ef90d7af51e86"


# ----------------------------------------------------------------- per-impl defaults
# pkg-07 spec 06 §7.1 — module-level (NOT on cfg.baselines).

_DEFAULT_RNN_HIDDEN_DIM: int = 64
_DEFAULT_MIXING_EMBED_DIM: int = 32
_DEFAULT_HYPERNET_LAYERS: int = 2
_DEFAULT_HYPERNET_EMBED: int = 64

_DEFAULT_EPSILON_START: float = 1.0
_DEFAULT_EPSILON_FINISH: float = 0.05
_DEFAULT_EPSILON_ANNEAL_TIME: int = 50_000   # env steps (linear anneal)

_DEFAULT_TARGET_UPDATE_INTERVAL: int = 200   # episodes between hard sync
_DEFAULT_DOUBLE_Q: bool = True
_DEFAULT_GAMMA: float = 0.99
_DEFAULT_GRAD_NORM_CLIP: float = 10.0
_DEFAULT_OPTIM_ALPHA: float = 0.99           # RMSprop alpha (pymarl default)
_DEFAULT_OPTIM_EPS: float = 1e-5

_DEFAULT_BATCH_SIZE: int = 32                # episodes per train step
_DEFAULT_BUFFER_SIZE: int = 5_000            # max episodes
_DEFAULT_DEFAULT_LR: float = 5e-4
_DEFAULT_EVAL_SEED_BASE: int = 42


# --------------------------------------------------------------------- helpers

def _obs_dict_to_array(obs_dict: dict[str, np.ndarray], n_agents: int) -> np.ndarray:
    """Stack PettingZoo per-agent obs dict into an ``(N, obs_dim)`` array."""
    return np.stack(
        [obs_dict[f"agent_{i}"] for i in range(n_agents)], axis=0,
    ).astype(np.float32)


def _action_array_to_dict(actions: np.ndarray, n_agents: int) -> dict[str, int]:
    return {f"agent_{i}": int(actions[i]) for i in range(n_agents)}


def _reward_dict_to_array(reward_dict: dict[str, float], n_agents: int) -> np.ndarray:
    return np.array(
        [reward_dict[f"agent_{i}"] for i in range(n_agents)], dtype=np.float32,
    )


def _check_forbidden_info(info_dict: dict[str, dict]) -> None:
    """pkg-07 spec 06 §6.2 runtime guard — every per-agent info dict must be
    free of forbidden keys (``c_true`` / ``types`` / ``resource_state`` /
    ``hotspot_centers``). Raises ``AssertionError`` on violation."""
    for agent_key, per_agent_info in info_dict.items():
        leaked = _FORBIDDEN_INFO_KEYS & set(per_agent_info)
        assert not leaked, (
            f"QMIXAlgorithm consumed forbidden info keys {leaked} at agent "
            f"{agent_key}. Either env_fn was constructed with oracle_mode=True "
            "(violates spec 04 §7) or the adapter is leaking. See pkg-07 design "
            "§3.5 CTDE boundary + spec 06 §6.2."
        )


def _build_qmix_args(cfg: V4Config, obs_dim: int) -> SimpleNamespace:
    """Build the ``args`` namespace consumed by RNNAgent + QMixer.

    Field names mirror upstream pymarl's YAML config keys verbatim — this
    keeps the vendored modules byte-faithful while letting the runner
    choose the values from V4Config + module defaults.
    """
    n_agents = int(cfg.env.N)
    n_actions = int(cfg.env.A)
    state_dim = int(obs_dim * n_agents)   # concat-of-obs (CTDE legitimate)
    return SimpleNamespace(
        # Agent net (RNNAgent)
        rnn_hidden_dim=_DEFAULT_RNN_HIDDEN_DIM,
        n_actions=n_actions,
        # Mixer (QMixer)
        n_agents=n_agents,
        state_shape=(state_dim,),
        mixing_embed_dim=_DEFAULT_MIXING_EMBED_DIM,
        hypernet_layers=_DEFAULT_HYPERNET_LAYERS,
        hypernet_embed=_DEFAULT_HYPERNET_EMBED,
        # Algorithm
        gamma=_DEFAULT_GAMMA,
        double_q=_DEFAULT_DOUBLE_Q,
        grad_norm_clip=_DEFAULT_GRAD_NORM_CLIP,
        optim_alpha=_DEFAULT_OPTIM_ALPHA,
        optim_eps=_DEFAULT_OPTIM_EPS,
        target_update_interval=_DEFAULT_TARGET_UPDATE_INTERVAL,
    )


# --------------------------------------------------------- shared agent-net wrapper

class _SharedAgent(torch.nn.Module):
    """Wrap RNNAgent so the shared agent net runs all N agents in batch over
    a single ``(B, N, T, obs_dim)`` rollout — pymarl's "shared parameters
    across agents" pattern (one ``RNNAgent`` instance, fed per-agent rows
    along the batch axis).
    """

    def __init__(self, args: SimpleNamespace) -> None:
        super().__init__()
        self.args = args
        self.agent = RNNAgent(input_shape=int(args.obs_dim), args=args)

    def init_hidden(self, batch_size: int, n_agents: int) -> torch.Tensor:
        return self.agent.init_hidden().expand(batch_size * n_agents, -1).contiguous()

    def forward_step(
        self,
        obs_step: torch.Tensor,                  # (B, N, obs_dim)
        hidden: torch.Tensor,                    # (B*N, rnn_hidden_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, n, d = obs_step.shape
        flat = obs_step.reshape(b * n, d)
        q, h = self.agent(flat, hidden)
        return q.view(b, n, -1), h


# ----------------------------------------------------------------- replay buffer

class _EpisodeBuffer:
    """Per-episode tensor store, lightweight reimplementation of pymarl's
    ``EpisodeBatch`` (without the multi-scheme abstraction it needs for SC2).
    """

    def __init__(
        self,
        buffer_size: int,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        episode_limit: int,
        device: torch.device,
    ) -> None:
        self.buffer_size = buffer_size
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.T = episode_limit
        self.device = device

        self.obs = torch.zeros((buffer_size, self.T + 1, n_agents, obs_dim), dtype=torch.float32)
        self.state = torch.zeros((buffer_size, self.T + 1, state_dim), dtype=torch.float32)
        self.actions = torch.zeros((buffer_size, self.T, n_agents, 1), dtype=torch.long)
        self.reward = torch.zeros((buffer_size, self.T, 1), dtype=torch.float32)
        self.terminated = torch.zeros((buffer_size, self.T, 1), dtype=torch.float32)
        self.filled = torch.zeros((buffer_size, self.T, 1), dtype=torch.float32)
        self._size: int = 0
        self._head: int = 0

    def __len__(self) -> int:
        return self._size

    def store_episode(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        actions: np.ndarray,
        reward: np.ndarray,
        terminated: np.ndarray,
        filled: np.ndarray,
    ) -> None:
        i = self._head
        T_real = obs.shape[0] - 1
        T = min(T_real, self.T)
        self.obs[i].zero_()
        self.state[i].zero_()
        self.actions[i].zero_()
        self.reward[i].zero_()
        self.terminated[i].zero_()
        self.filled[i].zero_()
        self.obs[i, : T + 1] = torch.as_tensor(obs[: T + 1], dtype=torch.float32)
        self.state[i, : T + 1] = torch.as_tensor(state[: T + 1], dtype=torch.float32)
        self.actions[i, :T, :, 0] = torch.as_tensor(actions[:T], dtype=torch.long)
        self.reward[i, :T, 0] = torch.as_tensor(reward[:T], dtype=torch.float32)
        self.terminated[i, :T, 0] = torch.as_tensor(terminated[:T], dtype=torch.float32)
        self.filled[i, :T, 0] = torch.as_tensor(filled[:T], dtype=torch.float32)

        self._head = (self._head + 1) % self.buffer_size
        self._size = min(self._size + 1, self.buffer_size)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = np.random.choice(self._size, size=batch_size, replace=False)
        idx_t = torch.as_tensor(idx, dtype=torch.long)
        return {
            "obs": self.obs[idx_t].to(self.device),
            "state": self.state[idx_t].to(self.device),
            "actions": self.actions[idx_t].to(self.device),
            "reward": self.reward[idx_t].to(self.device),
            "terminated": self.terminated[idx_t].to(self.device),
            "filled": self.filled[idx_t].to(self.device),
        }


# ----------------------------------------------------------------- runner

class QMIXAlgorithm(ExternalBaselineRunner):
    """QMIX (Tier-1 external; pkg-07 spec 06 §2). Real port from ``oxwhirl/pymarl``."""

    def __init__(self, cfg: V4Config, lr: Optional[float] = None) -> None:
        super().__init__(cfg)
        self._vendored_from: str = _VENDORED_FROM
        if lr is None:
            grid = cfg.baselines.external_lr_sweep_grid.get(
                "external_qmix", (_DEFAULT_DEFAULT_LR,),
            )
            lr = grid[len(grid) // 2]
        self._lr: float = float(lr)
        self._seed: int = 0
        self._device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        self._args: Optional[SimpleNamespace] = None
        self._mac: Optional[_SharedAgent] = None
        self._target_mac: Optional[_SharedAgent] = None
        self._mixer: Optional[QMixer] = None
        self._target_mixer: Optional[QMixer] = None
        self._optim: Optional[RMSprop] = None
        self._episode_count: int = 0
        self._last_target_update_episode: int = 0

    # ----------------------------------------------------------- lazy build

    def _maybe_build(self, obs_dim: int) -> None:
        if self._args is not None:
            return
        args = _build_qmix_args(self.cfg, obs_dim=obs_dim)
        args.obs_dim = int(obs_dim)
        self._args = args
        self._mac = _SharedAgent(args).to(self._device)
        self._target_mac = copy.deepcopy(self._mac).to(self._device)
        self._target_mac.load_state_dict(self._mac.state_dict())
        self._mixer = QMixer(args).to(self._device)
        self._target_mixer = copy.deepcopy(self._mixer).to(self._device)
        self._target_mixer.load_state_dict(self._mixer.state_dict())
        params = list(self._mac.parameters()) + list(self._mixer.parameters())
        self._optim = RMSprop(
            params=params, lr=self._lr,
            alpha=args.optim_alpha, eps=args.optim_eps,
        )

    # ----------------------------------------------------------- epsilon

    def _epsilon_at(self, env_step: int) -> float:
        anneal = max(1, _DEFAULT_EPSILON_ANNEAL_TIME)
        frac = min(1.0, env_step / anneal)
        return float(_DEFAULT_EPSILON_START
                     + (_DEFAULT_EPSILON_FINISH - _DEFAULT_EPSILON_START) * frac)

    # ----------------------------------------------------------- act

    def _act(
        self,
        obs_n: np.ndarray,                 # (N, obs_dim)
        hidden: torch.Tensor,              # (1*N, rnn_hidden_dim)
        epsilon: float,
    ) -> tuple[np.ndarray, torch.Tensor]:
        with torch.no_grad():
            obs_step = torch.as_tensor(obs_n, dtype=torch.float32, device=self._device).unsqueeze(0)
            q, h = self._mac.forward_step(obs_step, hidden)   # q: (1, N, A)
            q = q.squeeze(0)
            if np.random.rand() < epsilon:
                a = np.random.randint(0, self.cfg.env.A, size=self.cfg.env.N)
            else:
                a = q.argmax(dim=-1).cpu().numpy()
        return a.astype(np.int64), h

    # ----------------------------------------------------------- one episode

    def _run_one_episode(
        self,
        env,
        episode_limit: int,
        n_agents: int,
        obs_dim: int,
        evaluate: bool,
        env_step_offset: int,
        c_override: Optional[float] = None,
    ) -> tuple[dict[str, np.ndarray], int, float]:
        reset_options = {"c": float(c_override)} if c_override is not None else None
        obs_dict, info = env.reset(options=reset_options)
        _check_forbidden_info(info)

        T = episode_limit
        state_dim = obs_dim * n_agents
        obs_buf = np.zeros((T + 1, n_agents, obs_dim), dtype=np.float32)
        state_buf = np.zeros((T + 1, state_dim), dtype=np.float32)
        actions_buf = np.zeros((T, n_agents), dtype=np.int64)
        reward_buf = np.zeros((T,), dtype=np.float32)
        terminated_buf = np.zeros((T,), dtype=np.float32)
        filled_buf = np.zeros((T,), dtype=np.float32)

        obs_n = _obs_dict_to_array(obs_dict, n_agents)
        obs_buf[0] = obs_n
        state_buf[0] = obs_n.flatten()

        hidden = self._mac.init_hidden(batch_size=1, n_agents=n_agents)
        ep_return = 0.0
        steps = 0
        for t in range(T):
            eps = 0.0 if evaluate else self._epsilon_at(env_step_offset + t)
            a_n, hidden = self._act(obs_n, hidden, eps)
            obs_next_dict, reward_dict, term_dict, trunc_dict, info = env.step(
                _action_array_to_dict(a_n, n_agents)
            )
            _check_forbidden_info(info)
            r_n = _reward_dict_to_array(reward_dict, n_agents)
            r_global = float(r_n.sum())
            term = any(bool(v) for v in term_dict.values())
            trunc = any(bool(v) for v in trunc_dict.values())

            actions_buf[t] = a_n
            reward_buf[t] = r_global
            terminated_buf[t] = float(term)
            filled_buf[t] = 1.0
            ep_return += r_global

            obs_next_n = _obs_dict_to_array(obs_next_dict, n_agents)
            obs_buf[t + 1] = obs_next_n
            state_buf[t + 1] = obs_next_n.flatten()
            obs_n = obs_next_n
            steps = t + 1
            if term or trunc:
                break

        return (
            {
                "obs": obs_buf,
                "state": state_buf,
                "actions": actions_buf,
                "reward": reward_buf,
                "terminated": terminated_buf,
                "filled": filled_buf,
            },
            steps,
            ep_return,
        )

    # ----------------------------------------------------------- learn

    def _learn(self, batch: dict[str, torch.Tensor]) -> float:
        """Q-learning loss + double-Q + EMA hard sync (port of
        ``q_learner.py:QLearner.train``, lines 38-103)."""
        args = self._args
        n_agents = args.n_agents

        rewards = batch["reward"]                          # (B, T, 1)
        terminated_full = batch["terminated"].float()      # (B, T, 1)
        mask_full = batch["filled"].float()                # (B, T, 1)

        # Forward through full sequence (T+1 obs rows -> T+1 q rows).
        bs = batch["obs"].shape[0]
        T_full = batch["obs"].shape[1]
        hidden = self._mac.init_hidden(bs, n_agents)
        target_hidden = self._target_mac.init_hidden(bs, n_agents)

        mac_out_list, target_mac_out_list = [], []
        for t in range(T_full):
            obs_t = batch["obs"][:, t]
            q_t, hidden = self._mac.forward_step(obs_t, hidden)
            target_q_t, target_hidden = self._target_mac.forward_step(obs_t, target_hidden)
            mac_out_list.append(q_t)
            target_mac_out_list.append(target_q_t)
        mac_out = torch.stack(mac_out_list, dim=1)               # (B, T+1, N, A)
        target_mac_out = torch.stack(target_mac_out_list[1:], dim=1)  # (B, T, N, A)

        T_eff = mac_out.shape[1] - 1   # transitions count

        # Q-values for the actions actually taken.
        actions_eff = batch["actions"][:, :T_eff]                      # (B, T_eff, N, 1)
        chosen_qvals = torch.gather(
            mac_out[:, :T_eff], dim=3, index=actions_eff
        ).squeeze(3)                                                   # (B, T_eff, N)

        # Double-Q.
        if args.double_q:
            with torch.no_grad():
                cur_max_actions = mac_out[:, 1: T_eff + 1].argmax(dim=3, keepdim=True)
            target_max_qvals = torch.gather(
                target_mac_out[:, :T_eff], dim=3, index=cur_max_actions
            ).squeeze(3)
        else:
            target_max_qvals = target_mac_out[:, :T_eff].max(dim=3)[0]

        # Mix.
        chosen_q_tot = self._mixer(chosen_qvals, batch["state"][:, :T_eff])
        target_q_tot = self._target_mixer(
            target_max_qvals, batch["state"][:, 1: T_eff + 1]
        )

        rewards_eff = rewards[:, :T_eff]
        terminated_eff = terminated_full[:, :T_eff]
        targets = rewards_eff + args.gamma * (1.0 - terminated_eff) * target_q_tot
        td_error = chosen_q_tot - targets.detach()
        mask_eff = mask_full[:, :T_eff]
        if mask_eff.shape[1] > 1:
            # mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
            mask_eff = mask_eff.clone()
            mask_eff[:, 1:] = mask_eff[:, 1:] * (1.0 - terminated_eff[:, :-1])
        mask_eff = mask_eff.expand_as(td_error)
        masked_td = td_error * mask_eff
        denom = mask_eff.sum().clamp(min=1.0)
        loss = (masked_td ** 2).sum() / denom

        self._optim.zero_grad()
        loss.backward()
        params = list(self._mac.parameters()) + list(self._mixer.parameters())
        torch.nn.utils.clip_grad_norm_(params, args.grad_norm_clip)
        self._optim.step()
        return float(loss.item())

    def _maybe_update_targets(self) -> None:
        args = self._args
        if (self._episode_count - self._last_target_update_episode) \
                / args.target_update_interval >= 1.0:
            self._target_mac.load_state_dict(self._mac.state_dict())
            self._target_mixer.load_state_dict(self._mixer.state_dict())
            self._last_target_update_episode = self._episode_count

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

        # Update optim's lr if it changed at train() entry.
        for pg in self._optim.param_groups:
            pg["lr"] = self._lr

        episode_limit = int(cfg.env.T_max)
        state_dim = obs_dim * n_agents
        buffer = _EpisodeBuffer(
            buffer_size=_DEFAULT_BUFFER_SIZE,
            n_agents=n_agents,
            obs_dim=obs_dim,
            state_dim=state_dim,
            episode_limit=episode_limit,
            device=self._device,
        )

        env_steps = 0
        while env_steps < total_env_steps:
            transitions, steps, _ep_return = self._run_one_episode(
                env=env,
                episode_limit=episode_limit,
                n_agents=n_agents,
                obs_dim=obs_dim,
                evaluate=False,
                env_step_offset=env_steps,
            )
            buffer.store_episode(**transitions)
            env_steps += steps
            self._episode_count += 1

            if len(buffer) >= _DEFAULT_BATCH_SIZE:
                _ = self._learn(buffer.sample(_DEFAULT_BATCH_SIZE))
                self._maybe_update_targets()

        env.close()

    def evaluate(
        self,
        env_fn: Callable[[], Any],
        c_grid: tuple[float, ...],
        episodes: int,
    ) -> EvalReport:
        """Per-c deterministic rollouts; assemble the locked 32-field EvalReport.

        Field-population matrix per pkg-07 spec 06 §2.7 (= spec 05 §8.1).
        """
        env = env_fn()
        obs_dict, _ = env.reset()
        obs_dim = int(obs_dict["agent_0"].shape[-1])
        env.close()
        self._maybe_build(obs_dim)

        env = env_fn()
        n_agents = int(self.cfg.env.N)
        episode_limit = int(self.cfg.env.T_max)

        t0 = time.time()
        return_per_c: dict[float, float] = {}
        return_per_c_sem: dict[float, float] = {}
        episodes_per_c: dict[float, int] = {}
        all_returns: list[float] = []
        env_steps_total = 0

        for c in c_grid:
            c_returns: list[float] = []
            for _ep in range(int(episodes)):
                _, steps, ep_return = self._run_one_episode(
                    env=env,
                    episode_limit=episode_limit,
                    n_agents=n_agents,
                    obs_dim=obs_dim,
                    evaluate=True,
                    env_step_offset=0,
                    c_override=float(c),
                )
                c_returns.append(float(ep_return))
                env_steps_total += int(steps)
            return_per_c[float(c)] = float(np.mean(c_returns)) if c_returns else 0.0
            sem = (
                float(np.std(c_returns) / max(np.sqrt(len(c_returns)), 1.0))
                if len(c_returns) > 1 else 0.0
            )
            return_per_c_sem[float(c)] = sem
            episodes_per_c[float(c)] = int(len(c_returns))
            all_returns.extend(c_returns)
        env.close()

        zs_train = set(float(c) for c in self.cfg.eval.zero_shot_train_c)
        zs_unseen = set(float(c) for c in self.cfg.eval.zero_shot_unseen_c)
        seen_returns = [return_per_c[c] for c in return_per_c if c in zs_train]
        unseen_returns = [return_per_c[c] for c in return_per_c if c in zs_unseen]
        zs_seen = float(np.mean(seen_returns)) if seen_returns else 0.0
        zs_unseen_v = float(np.mean(unseen_returns)) if unseen_returns else 0.0

        segments = tuple(self.cfg.eval.c_segments)
        ratios = tuple(self.cfg.eval.bell_curve_type_ratios)

        return_mean = float(np.mean(all_returns)) if all_returns else 0.0
        return_sem = (
            float(np.std(all_returns) / max(np.sqrt(len(all_returns)), 1.0))
            if len(all_returns) > 1 else 0.0
        )

        return EvalReport(
            variant="external_qmix",
            seed=int(self._seed),
            config_hash="0" * 40,
            eval_mode="planner",
            eval_planner_mode="planner_full",   # spec 06 §2.7
            c_visible=bool(self.cfg.env.c_visible),
            return_mean=return_mean,
            return_sem=return_sem,
            return_zero_shot_seen=zs_seen,
            return_zero_shot_unseen=zs_unseen_v,
            return_zero_shot_gap=zs_seen - zs_unseen_v,
            return_per_c=MappingProxyType(return_per_c),
            return_per_c_sem=MappingProxyType(return_per_c_sem),
            episodes_per_c=MappingProxyType(episodes_per_c),
            return_per_segment=MappingProxyType({seg: 0.0 for seg in segments}),
            return_per_segment_sem=MappingProxyType({seg: 0.0 for seg in segments}),
            return_per_type_ratio=MappingProxyType({r: 0.0 for r in ratios}),
            return_per_type_ratio_sem=MappingProxyType({r: 0.0 for r in ratios}),
            regret_per_c=MappingProxyType({c: 0.0 for c in c_grid}),
            regret_mean=0.0,
            oracle_ceiling_per_c=MappingProxyType({c: 0.0 for c in c_grid}),
            oracle_ceiling_cache_hit=MappingProxyType({c: False for c in c_grid}),
            planner_prior_return_gap=0.0,
            direct_inference_return_mean=return_mean,
            planner_full_return_mean=return_mean,
            walltime_seconds=float(time.time() - t0),
            env_steps_evaluated=int(env_steps_total),
            episodes_total=int(len(all_returns)),
            info_gating_strict=True,
            set_context_subjective_oracle_leak=False,
            belief_c_mae=None,
            belief_c_calibration=None,
        )

    # ----------------------------------------------------------- ckpt

    def save_checkpoint(self, path: Union[Path, str]) -> None:
        if self._mac is None:
            return
        torch.save(
            {
                "mac_state_dict": self._mac.state_dict(),
                "mixer_state_dict": self._mixer.state_dict(),
                "target_mac_state_dict": self._target_mac.state_dict(),
                "target_mixer_state_dict": self._target_mixer.state_dict(),
                "optim_state_dict": self._optim.state_dict(),
                "vendored_from": _VENDORED_FROM,
                "lr": self._lr,
                "seed": self._seed,
            },
            path,
        )

    def load_checkpoint(self, path: Union[Path, str]) -> None:
        if self._mac is None:
            raise RuntimeError(
                "QMIXAlgorithm.load_checkpoint requires train() or evaluate() to "
                "have been called first (so the agent net is built with the "
                "env-derived obs_dim)."
            )
        ckpt = torch.load(path, map_location=self._device)
        self._mac.load_state_dict(ckpt["mac_state_dict"])
        self._mixer.load_state_dict(ckpt["mixer_state_dict"])
        self._target_mac.load_state_dict(ckpt["target_mac_state_dict"])
        self._target_mixer.load_state_dict(ckpt["target_mixer_state_dict"])
        self._optim.load_state_dict(ckpt["optim_state_dict"])
        self._lr = float(ckpt.get("lr", self._lr))
        self._seed = int(ckpt.get("seed", self._seed))

    def param_count(self) -> int:
        if self._mac is None:
            return 0
        return sum(p.numel() for p in self._mac.parameters()) \
             + sum(p.numel() for p in self._mixer.parameters())


__all__ = ["QMIXAlgorithm"]
