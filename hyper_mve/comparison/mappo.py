"""MAPPOAlgorithm — Tier-1 external baseline (pkg-07 spec 05).

Real port from ``D:\\RL\\lzj\\MAPPO\\`` (md5(mappo.py) =
``960d27648d56315a95cba837efd844dd``, mtime 2025-04-01). The vendored
algorithm logic lives unmodified in ``_lzj_mappo/`` (sibling subpackage);
this module is the plumbing layer that:

  * bridges :class:`hyper_mve.utils.configs.V4Config` into the ``args``-shaped
    namespace ``MAPPO_MPE`` consumes;
  * drives episode collection from
    :class:`hyper_mve.envs.adapters.pettingzoo_wrapper.ResourceCommonsPettingZooEnv`
    via the injected ``env_fn`` factory (no direct ``ResourceCommonsEnv``
    construction — pkg-07 spec 04 §10);
  * implements ``train(cfg, env_fn)`` and overrides ``evaluate(env_fn,
    regime_grid, episodes)`` with deterministic per-regime rollouts that populate
    the locked 32-field :class:`hyper_mve.utils.eval.eval_report.EvalReport`.

Per-impl tuning constants (mlp_hidden_dim, K_epochs, batch_size, etc.) live
as module defaults below — they are impl-internal, not cross-spec contract
(pkg-07 spec 01 §6.4).
"""
from __future__ import annotations

import time
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, Optional

import numpy as np
import torch

from hyper_mve.comparison import _FORBIDDEN_INFO_KEYS
from hyper_mve.comparison.base import ExternalBaselineRunner
from hyper_mve.comparison._lzj_mappo.mappo import MAPPO_MPE
from hyper_mve.comparison._lzj_mappo.normalization import Normalization
from hyper_mve.comparison._lzj_mappo.replay_buffer import ReplayBuffer
from hyper_mve.utils.configs import V4Config
from hyper_mve.comparison.base import split_seen_unseen_regimes
from hyper_mve.utils.eval.eval_report import EvalReport


def _check_forbidden_info(info_dict: dict[str, dict]) -> None:
    """pkg-07 spec 05 §5.2 / spec 06 §6.2 runtime guard — every per-agent
    info dict must be free of forbidden keys (``c_true`` / ``types`` /
    ``resource_state`` / ``hotspot_centers``). Raises ``AssertionError``
    on violation."""
    for agent_key, per_agent_info in info_dict.items():
        leaked = _FORBIDDEN_INFO_KEYS & set(per_agent_info)
        assert not leaked, (
            f"MAPPOAlgorithm consumed forbidden info keys {leaked} at agent "
            f"{agent_key}. Either env_fn was constructed with oracle_mode=True "
            "(violates spec 04 §7) or the adapter is leaking. See pkg-07 design "
            "§3.5 CTDE boundary + spec 05 §5.2."
        )


#: Provenance marker for the audit trail (pkg-07 spec 05 §1).
_VENDORED_FROM: str = "lzj-mappo md5=960d27648d56315a95cba837efd844dd mtime=2025-04-01"


# ----------------------------------------------------------------- per-impl defaults

_MLP_HIDDEN_DIM_DEFAULT: int = 64       # MAPPO_main.py:148
_RNN_HIDDEN_DIM_DEFAULT: int = 64
_BATCH_SIZE_DEFAULT: int = 32           # MAPPO_main.py:145 — episodes per training round
_MINI_BATCH_SIZE_DEFAULT: int = 8       # MAPPO_main.py:146
_K_EPOCHS_DEFAULT: int = 15
_LAMDA_DEFAULT: float = 0.95
_EPSILON_DEFAULT: float = 0.2
_ENTROPY_COEF_DEFAULT: float = 0.01
_DEFAULT_LR: float = 3e-4               # mid of the LR sweep grid


def _build_args(cfg: V4Config, obs_dim: int, lr: float, max_train_steps: int) -> SimpleNamespace:
    """Build the namespace the vendored ``MAPPO_MPE`` consumes.

    Every field below maps 1:1 to the ``argparse`` block in MAPPO_main.py
    (lines 139-167 of the source). State dimension follows the source
    convention ``state_dim = N * obs_dim`` (concatenation of all per-agent
    observations — MPE/CTDE legitimate global state per pkg-07 spec 04 §7
    CTDE-legitimacy footnote).
    """
    return SimpleNamespace(
        # Environment dimensions (filled in once env_fn is invoked).
        N=cfg.env.N,
        obs_dim=int(obs_dim),
        state_dim=int(obs_dim * cfg.env.N),
        action_dim=int(cfg.env.A),
        episode_limit=int(cfg.env.T_max),
        # Training schedule.
        max_train_steps=int(max_train_steps),
        batch_size=_BATCH_SIZE_DEFAULT,
        mini_batch_size=_MINI_BATCH_SIZE_DEFAULT,
        # Optimisation.
        lr=float(lr),
        gamma=float(cfg.train.gamma),
        lamda=_LAMDA_DEFAULT,
        epsilon=_EPSILON_DEFAULT,
        K_epochs=_K_EPOCHS_DEFAULT,
        entropy_coef=_ENTROPY_COEF_DEFAULT,
        # Tricks (1, 6, 7, 8, 9 from MAPPO_main.py).
        use_adv_norm=True,
        use_lr_decay=True,
        use_grad_clip=True,
        use_orthogonal_init=True,
        set_adam_eps=True,
        # Architecture.
        rnn_hidden_dim=_RNN_HIDDEN_DIM_DEFAULT,
        mlp_hidden_dim=_MLP_HIDDEN_DIM_DEFAULT,
        use_relu=False,             # tanh activations
        use_rnn=False,               # MLP variant — simpler default
        add_agent_id=False,
        use_value_clip=False,
        # Reward shaping (MAPPO_main.py:155-156 — norm by default).
        use_reward_norm=True,
        use_reward_scaling=False,
    )


def _obs_dict_to_array(obs_dict: dict[str, np.ndarray], n_agents: int) -> np.ndarray:
    """Stack a PettingZoo per-agent obs dict into the ``(N, obs_dim)`` array
    the vendored MAPPO expects."""
    return np.stack(
        [obs_dict[f"agent_{i}"] for i in range(n_agents)], axis=0,
    ).astype(np.float32)


def _action_array_to_dict(actions: np.ndarray, n_agents: int) -> dict[str, int]:
    """Convert MAPPO's ``(N,)`` action vector into the PettingZoo action dict."""
    return {f"agent_{i}": int(actions[i]) for i in range(n_agents)}


def _reward_dict_to_array(reward_dict: dict[str, float], n_agents: int) -> np.ndarray:
    return np.array(
        [reward_dict[f"agent_{i}"] for i in range(n_agents)], dtype=np.float32,
    )


def _term_dict_to_array(term_dict: dict[str, bool], n_agents: int) -> np.ndarray:
    return np.array(
        [bool(term_dict[f"agent_{i}"]) for i in range(n_agents)], dtype=np.float32,
    )


class MAPPOAlgorithm(ExternalBaselineRunner):
    """MAPPO (Tier-1 external; pkg-07 spec 05). Real port from ``D:\\RL\\lzj\\MAPPO``."""

    def __init__(self, cfg: V4Config, lr: Optional[float] = None) -> None:
        super().__init__(cfg)
        self._vendored_from: str = _VENDORED_FROM
        # LR sweep grid is read from cfg.baselines.external_lr_sweep_grid;
        # the runner's actual LR can be overridden at construction time
        # for the spec 05 LR sweep harness.
        if lr is None:
            grid = cfg.baselines.external_lr_sweep_grid.get(
                "mappo", (_DEFAULT_LR,),
            )
            lr = grid[len(grid) // 2]   # middle of the grid (3e-4 by default)
        self._lr: float = float(lr)
        self._device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        self._agent: Optional[MAPPO_MPE] = None
        self._reward_norm: Optional[Normalization] = None

    # -------------------------------------------------------------- training

    def train(
        self,
        cfg: V4Config,
        env_fn: Callable[[], Any],
        *,
        total_env_steps: int = 0,
        lr: float = 0.0,
        seed: int = 0,
        max_train_steps: Optional[int] = None,
        tensorboard_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Train the MAPPO agent against ``ResourceCommonsPettingZooEnv``.

        Args:
            cfg: V4Config; consumed for env / training-budget fields.
            env_fn: factory returning a fresh ``ResourceCommonsPettingZooEnv``
                with ``oracle_mode=False, eval_info_mode=False`` (training-time
                contract, pkg-07 spec 04 §10).
            total_env_steps: pkg-07 spec 05 §4.2 canonical kwarg — sweep
                harness's per-row env-step budget.
            lr: pkg-07 spec 05 §4.2 canonical kwarg — one value from
                ``cfg.baselines.external_lr_sweep_grid["mappo"]``.
            seed: pkg-07 spec 05 §4.2 canonical kwarg — one seed.
            max_train_steps: legacy alias for ``total_env_steps``; honoured
                when ``total_env_steps`` is unset.
            tensorboard_dir: when set (sweep harness passes the row's tb/ dir),
                a :class:`~hyper_mve.comparison._probe.PeriodicEvalProbe`
                writes deterministic per-regime eval returns keyed by cumulative
                env steps (sample-efficiency curves).
            **kwargs: forward-compat slots; currently ignored.
        """
        del kwargs  # forward-compat
        if total_env_steps > 0:
            budget = int(total_env_steps)
        elif max_train_steps is not None:
            budget = int(max_train_steps)
        else:
            budget = int(cfg.train.max_train_steps)
        if lr > 0:
            self._lr = float(lr)
        if seed:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))

        env = env_fn()
        n_agents = cfg.env.N
        obs_dict, info = env.reset(seed=int(seed) if seed else None)
        _check_forbidden_info(info)
        obs_dim = int(obs_dict[f"agent_0"].shape[-1])

        args = _build_args(cfg, obs_dim=obs_dim, lr=self._lr, max_train_steps=budget)
        self._agent = MAPPO_MPE(args, self._device)
        replay_buffer = ReplayBuffer(args, self._device)
        self._reward_norm = Normalization(shape=n_agents) if args.use_reward_norm else None

        probe = None
        tb_writer = None
        if tensorboard_dir:
            from torch.utils.tensorboard import SummaryWriter

            from hyper_mve.comparison._probe import PeriodicEvalProbe

            tb_writer = SummaryWriter(tensorboard_dir)

            def _probe_act(obs: np.ndarray, t: int) -> np.ndarray:
                del t  # MLP policy — no recurrent state to reset
                a_n, _ = self._agent.choose_action(obs, evaluate=True)
                return np.asarray(a_n)

            probe = PeriodicEvalProbe(env_fn, cfg, tb_writer, act_fn=_probe_act)

        total_steps = 0
        while total_steps < budget:
            episode_steps = self._run_one_episode(
                env=env,
                env_fn=env_fn,
                args=args,
                replay_buffer=replay_buffer,
                evaluate=False,
            )
            total_steps += episode_steps

            if replay_buffer.episode_num == args.batch_size:
                self._agent.train(replay_buffer, total_steps)
                replay_buffer.reset_buffer()

            if probe is not None:
                probe.maybe_run(total_steps)

        env.close()
        if probe is not None:
            probe.close()
        if tb_writer is not None:
            tb_writer.flush()

    # ----------------------------------------------------------- one-episode

    def _run_one_episode(
        self,
        env,
        env_fn,
        args: SimpleNamespace,
        replay_buffer: Optional[ReplayBuffer],
        evaluate: bool,
        g_override: Optional[int] = None,
    ) -> int:
        """Run a single episode; returns the number of steps actually taken.

        When ``evaluate=True``, ``replay_buffer`` is unused (``None`` allowed)
        and actions are deterministic argmax samples.
        When ``g_override`` is set, the env is reset with ``options={"g": g}``
        — used by ``evaluate()`` to pin the relationship regime per grid row.
        """
        n_agents = args.N
        reset_options = {"g": int(g_override)} if g_override is not None else None
        obs_dict, info = env.reset(options=reset_options)
        _check_forbidden_info(info)
        obs_n = _obs_dict_to_array(obs_dict, n_agents)
        episode_reward = 0.0
        episode_step = 0
        done_n = np.zeros(n_agents, dtype=np.float32)

        for episode_step in range(args.episode_limit):
            a_n, a_logprob_n = self._agent.choose_action(obs_n, evaluate=evaluate)
            s = obs_n.flatten()                          # CTDE-legitimate global state
            v_n = self._agent.get_value(s)
            action_dict = _action_array_to_dict(a_n, n_agents)
            obs_next_dict, reward_dict, term_dict, trunc_dict, info = env.step(action_dict)
            _check_forbidden_info(info)
            obs_next_n = _obs_dict_to_array(obs_next_dict, n_agents)
            r_n = _reward_dict_to_array(reward_dict, n_agents)
            term_n = _term_dict_to_array(term_dict, n_agents)
            trunc_n = _term_dict_to_array(trunc_dict, n_agents)
            done_n = np.maximum(term_n, trunc_n)
            episode_reward += float(r_n.sum())

            if not evaluate:
                if self._reward_norm is not None:
                    r_n = self._reward_norm(r_n)
                replay_buffer.store_transition(
                    episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n,
                )

            obs_n = obs_next_n
            if bool(done_n.all()):
                break

        if not evaluate:
            # Bootstrap value at the final state.
            s = obs_n.flatten()
            v_n = self._agent.get_value(s)
            replay_buffer.store_last_value(episode_step + 1, v_n)

        # Stash the per-episode return on the runner so ``evaluate`` can
        # reach it without re-plumbing the function signature.
        self._last_episode_return = episode_reward
        return episode_step + 1

    # -------------------------------------------------------------- evaluate

    def evaluate(
        self,
        env_fn: Callable[[], Any],
        regime_grid: tuple[int, ...],
        episodes: int,
    ) -> EvalReport:
        """Per-regime deterministic rollouts; assemble the locked rel-v1 EvalReport.

        Args:
            env_fn: factory returning ``ResourceCommonsPettingZooEnv`` with
                ``oracle_mode=False, eval_info_mode=True`` (eval-time contract,
                pkg-07 spec 04 §10).
            regime_grid: regime ids to sweep (tuple of ints; v5).
            episodes: episode count per regime (pkg-07 spec 08 §4.1).
        """
        if self._agent is None:
            # Train was never called; build a randomly-initialised agent so
            # ``evaluate`` still produces a syntactically-valid EvalReport
            # (used for smoke tests).
            tmp_env = env_fn()
            obs_dict, _ = tmp_env.reset()
            obs_dim = int(obs_dict[f"agent_0"].shape[-1])
            tmp_env.close()
            args = _build_args(
                self.cfg,
                obs_dim=obs_dim,
                lr=self._lr,
                max_train_steps=int(self.cfg.train.max_train_steps),
            )
            self._agent = MAPPO_MPE(args, self._device)
        else:
            args = SimpleNamespace(
                N=self._agent.N,
                obs_dim=self._agent.obs_dim,
                state_dim=self._agent.state_dim,
                action_dim=self._agent.action_dim,
                episode_limit=self._agent.episode_limit,
            )

        t0 = time.time()
        env = env_fn()
        return_per_regime: dict[int, float] = {}
        return_per_regime_sem: dict[int, float] = {}
        episodes_per_regime: dict[int, int] = {}
        all_returns: list[float] = []
        env_steps_total = 0
        # Per-episode returns stashed for distribution views (box plots);
        # same episodes that produce the report means.
        self._eval_episode_returns: dict[int, list[float]] = {}

        for g in regime_grid:
            g_returns: list[float] = []
            for _ in range(int(episodes)):
                episode_steps = self._run_one_episode(
                    env=env,
                    env_fn=env_fn,
                    args=args,
                    replay_buffer=None,
                    evaluate=True,
                    g_override=int(g),
                )
                g_returns.append(float(self._last_episode_return))
                env_steps_total += int(episode_steps)
            self._eval_episode_returns[int(g)] = list(g_returns)
            return_per_regime[int(g)] = float(np.mean(g_returns)) if g_returns else 0.0
            sem = (
                float(np.std(g_returns) / max(np.sqrt(len(g_returns)), 1.0))
                if len(g_returns) > 1 else 0.0
            )
            return_per_regime_sem[int(g)] = sem
            episodes_per_regime[int(g)] = int(len(g_returns))
            all_returns.extend(g_returns)
        env.close()

        # Zero-shot seen / unseen / gap (v5: train_regime_ids split).
        zs_seen, zs_unseen_v = split_seen_unseen_regimes(self.cfg, return_per_regime)

        return_mean = float(np.mean(all_returns)) if all_returns else 0.0
        return_sem = (
            float(np.std(all_returns) / max(np.sqrt(len(all_returns)), 1.0))
            if len(all_returns) > 1 else 0.0
        )

        return EvalReport(
            variant="mappo",
            seed=0,
            config_hash="0" * 40,
            eval_mode="planner",
            eval_planner_mode=self.cfg.eval.eval_planner_mode,
            return_mean=return_mean,
            return_sem=return_sem,
            return_zero_shot_seen=zs_seen,
            return_zero_shot_unseen=zs_unseen_v,
            return_zero_shot_gap=zs_seen - zs_unseen_v,
            return_per_regime=MappingProxyType(return_per_regime),
            return_per_regime_sem=MappingProxyType(return_per_regime_sem),
            episodes_per_regime=MappingProxyType(episodes_per_regime),
            # External runners have no MVE planner; pkg-08 spec 01 §6.2 says
            # set planner-prior gap to 0 and the two mode-means to return_mean.
            planner_prior_return_gap=0.0,
            direct_inference_return_mean=return_mean,
            planner_full_return_mean=return_mean,
            walltime_seconds=float(time.time() - t0),
            env_steps_evaluated=int(env_steps_total),
            episodes_total=int(len(all_returns)),
            info_gating_strict=True,
            set_context_subjective_oracle_leak=False,
        )

    # ----------------------------------------------------------- ckpt + diag

    def save_checkpoint(self, path) -> None:
        """pkg-07 spec 05 §4.4 — actor + critic + optimizer state + provenance."""
        if self._agent is None:
            return
        ckpt = {
            "actor_state_dict": self._agent.actor.state_dict(),
            "critic_state_dict": self._agent.critic.state_dict(),
            "ac_optimizer_state_dict": self._agent.ac_optimizer.state_dict(),
            "vendored_from": _VENDORED_FROM,
            "lr": self._lr,
        }
        torch.save(ckpt, path)

    def load_checkpoint(self, path) -> None:
        if self._agent is None:
            raise RuntimeError(
                "MAPPOAlgorithm.load_checkpoint requires train() or evaluate() "
                "to have been called first (so the agent is built with the "
                "env-derived obs_dim)."
            )
        ckpt = torch.load(path, map_location=self._device)
        self._agent.actor.load_state_dict(ckpt["actor_state_dict"])
        self._agent.critic.load_state_dict(ckpt["critic_state_dict"])
        self._agent.ac_optimizer.load_state_dict(ckpt["ac_optimizer_state_dict"])
        self._lr = float(ckpt.get("lr", self._lr))

    def param_count(self) -> int:
        """pkg-07 spec 07 disclosure feed — actor + critic params."""
        if self._agent is None:
            return 0
        return sum(p.numel() for p in self._agent.actor.parameters()) \
             + sum(p.numel() for p in self._agent.critic.parameters())


__all__ = ["MAPPOAlgorithm"]
