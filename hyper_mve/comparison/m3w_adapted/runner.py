"""M3WAdaptedRunner — the M3W-adapted baseline behind the four-method
``ExternalBaselineRunner`` protocol (phase-6 realignment).

User-locked spec: M3W's core mechanism (MoE world model + task-embedding
conditioning) preserved via the clone's ``CenMoEDynamicsModel`` (SoftMoE
dynamics) and ``CenMoERewardModel`` (SparseMoE, per-agent rewards) imported
UNMODIFIED; task conditioning = GIVEN one-hot regime ID → ``nn.Embedding``
conditioning both MoE modules; acting = greedy-sampling planner; policy
base = HASAC-derived per-agent discrete SAC. General-sum, discrete,
dynamic-regime with the regime ID given.

DISCLOSED oracle-ID protocol (this baseline doubles as the given-ID vs
inferred-belief contrast against our method):
  * ``train`` ignores ``env_fn`` and builds its own collection env with
    ``oracle_mode=True`` (same train-time-only privileged-g channel as our
    method's belief supervision); the given ID is ``info["g_true"]``.
  * ``evaluate`` consumes the injected oracle-free ``env_fn`` and conditions
    on the PINNED eval-grid regime ID (we set g at reset, so no oracle info
    is read; the ``_FORBIDDEN_INFO_KEYS`` runtime guard is enforced).
  * Critics are decentralized (per-agent, own conditioned latent);
    centralization lives in the MoE world model.
"""
from __future__ import annotations

import time
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Optional, Union

import numpy as np

from hyper_mve.utils.configs import V4Config
from hyper_mve.utils.eval.eval_report import EvalReport, regime_names_for

from hyper_mve.utils.schemas.relation import get_regime_family
from hyper_mve.comparison.base import (
    freeze_per_agent,
    reward_vector,
    ExternalBaselineRunner,
    _FORBIDDEN_INFO_KEYS,
    split_seen_unseen_regimes,
)


def _check_forbidden_info(info_dict) -> None:
    for agent_info in info_dict.values():
        leaked = _FORBIDDEN_INFO_KEYS & set(agent_info)
        assert not leaked, f"forbidden info keys leaked to m3w_adapted: {leaked}"


class _ReplayBuffer:
    """Flat ring buffer of joint transitions (numpy storage)."""

    def __init__(self, capacity: int, n_agents: int, obs_dim: int) -> None:
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.g = np.zeros(capacity, dtype=np.int64)
        self.a = np.zeros((capacity, n_agents), dtype=np.int64)
        self.r = np.zeros((capacity, n_agents), dtype=np.float32)
        self.next_obs = np.zeros_like(self.obs)
        self.next_g = np.zeros_like(self.g)
        self.done = np.zeros(capacity, dtype=np.float32)  # terminal only
        self._n = 0
        self._i = 0

    def __len__(self) -> int:
        return self._n

    def add(self, obs, g, a, r, next_obs, next_g, done) -> None:
        i = self._i
        self.obs[i], self.g[i], self.a[i], self.r[i] = obs, g, a, r
        self.next_obs[i], self.next_g[i], self.done[i] = next_obs, next_g, done
        self._i = (i + 1) % self.capacity
        self._n = min(self._n + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, self._n, size=batch_size)
        return (self.obs[idx], self.g[idx], self.a[idx], self.r[idx],
                self.next_obs[idx], self.next_g[idx], self.done[idx])


class M3WAdaptedRunner(ExternalBaselineRunner):
    name = "m3w_adapted"

    def __init__(self, cfg: V4Config) -> None:
        self.cfg = cfg
        self._wm = None
        self._sac = None
        self._planner = None
        self._device = None
        self._gen = None                # torch.Generator on self._device

    # ------------------------------------------------------------ internals
    def _make_oracle_env(self, env_cfg):
        """Own oracle-mode collection env (mirrors scripts/train.py's
        make_env_fn per env_kind, with oracle_mode=True)."""
        if getattr(env_cfg, "env_kind", "relation") == "mpe_tag":
            from hyper_mve.envs.mpe_tag.env import MPETagRegimeEnv
            return MPETagRegimeEnv(
                env_cfg, oracle_mode=True, eval_info_mode=False,
                fixed_regime=env_cfg.fixed_regime,
            )
        from hyper_mve.envs.adapters.pettingzoo_wrapper import (
            RelationCommonsPettingZooEnv,
        )
        return RelationCommonsPettingZooEnv(
            env_cfg, oracle_mode=True, eval_info_mode=False)

    def _build(self, obs_dim: int) -> None:
        import torch
        from .world_model import RegimeCondWorldModel
        from .sac import PerAgentDiscreteSAC
        from .planner import GreedyModelPlanner

        env_cfg = self.cfg.env
        n_regimes = get_regime_family(env_cfg).size
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self._wm = RegimeCondWorldModel(
            obs_dim=obs_dim, n_agents=env_cfg.N, n_actions=env_cfg.A,
            n_regimes=n_regimes,
        ).to(self._device)
        self._sac = PerAgentDiscreteSAC(
            n_agents=env_cfg.N, d_in=self._wm.d_zc, n_actions=env_cfg.A,
        ).to(self._device)
        self._planner = GreedyModelPlanner(self._wm, self._sac)
        self._gen = torch.Generator(device=self._device)
        self._gen.manual_seed(0)

    # ---------------------------------------------------------------- train
    def train(self, cfg, env_fn, *, total_env_steps: int = 0, lr: float = 0.0,
              seed: int = 0, **kwargs) -> None:
        import torch

        self.cfg = cfg
        # NOTE: env_fn is NOT used for training below (own oracle-mode env,
        # given-ID protocol, see docstring) -- kept alive only for the
        # periodic probe, which mirrors evaluate()'s oracle-free env_fn use.
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        rng = np.random.default_rng(int(seed))
        unified_logger = kwargs.get("unified_logger")

        lr = float(lr) if lr and lr > 0 else 3e-4
        total = int(total_env_steps)
        warmup = min(max(total // 8, 64), 500)
        update_every = 4                       # env steps per train step
        batch_size = 128
        log_every = 25

        env = self._make_oracle_env(cfg.env)
        agents = list(env.possible_agents)
        N = len(agents)

        obs_dict, info = env.reset(seed=int(seed))
        obs = np.stack([np.asarray(obs_dict[a], dtype=np.float32)
                        for a in agents])
        g = int(info[agents[0]]["g_true"])
        if self._wm is None:
            self._build(obs.shape[-1])
        self._gen.manual_seed(int(seed))
        buffer = _ReplayBuffer(max(total, 10_000), N, obs.shape[-1])

        wm_opt = torch.optim.Adam(self._wm.parameters(), lr=lr)
        q_opt = torch.optim.Adam(
            [p for m in (self._sac.q1s, self._sac.q2s)
             for p in m.parameters()], lr=lr)
        actor_opt = torch.optim.Adam(self._sac.actors.parameters(), lr=lr)

        if unified_logger is not None:
            unified_logger.declare_ratio(float(update_every))

        probe = None
        tensorboard_dir = kwargs.get("tensorboard_dir")
        if tensorboard_dir or unified_logger is not None:
            from hyper_mve.comparison._probe import PeriodicEvalProbe

            probe_writer = unified_logger
            if probe_writer is None:
                from torch.utils.tensorboard import SummaryWriter

                probe_writer = SummaryWriter(tensorboard_dir)
            probe_gen = torch.Generator(device=self._device)

            def _probe_act(obs: np.ndarray, t: int, g: int) -> np.ndarray:
                del t
                probe_gen.manual_seed(0)
                return self._planner.plan(obs, int(g), explore=False,
                                          generator=probe_gen)

            def _fidelity_fn():
                from hyper_mve.utils.eval.fidelity import compute_fidelity_report
                from hyper_mve.utils.schemas import get_regime_family

                grid = tuple(range(get_regime_family(cfg.env).size))
                return compute_fidelity_report(self, env_fn, grid,
                                               episodes=2, seed=1234)

            probe = PeriodicEvalProbe(
                env_fn, cfg, probe_writer, act_fn=_probe_act,
                every_env_steps=2000, episodes_per_regime=2,
                fidelity_fn=_fidelity_fn,
            )

        self._wm.train()
        self._sac.train()
        train_step, episode = 0, 0
        for t in range(1, total + 1):
            if t <= warmup:
                acts = rng.integers(0, cfg.env.A, size=N)
            else:
                acts = self._planner.plan(obs, g, explore=True,
                                          generator=self._gen)
            action_dict = {a: int(acts[i]) for i, a in enumerate(agents)}
            obs_dict, rew, term, trunc, info = env.step(action_dict)
            next_obs = np.stack([np.asarray(obs_dict[a], dtype=np.float32)
                                 for a in agents])
            next_g = int(info[agents[0]]["g_true"])
            r = np.array([float(rew[a]) for a in agents], dtype=np.float32)
            terminal = bool(any(term.values()))
            buffer.add(obs, g, acts, r, next_obs, next_g, float(terminal))

            if terminal or any(trunc.values()):
                episode += 1
                obs_dict, info = env.reset(seed=int(seed) + 7919 * episode)
                obs = np.stack([np.asarray(obs_dict[a], dtype=np.float32)
                                for a in agents])
                g = int(info[agents[0]]["g_true"])
            else:
                obs, g = next_obs, next_g

            if t >= warmup and t % update_every == 0 and len(buffer) >= batch_size:
                b_obs, b_g, b_a, b_r, b_next_obs, b_next_g, b_done = \
                    buffer.sample(batch_size, rng)
                to = lambda x, dt: torch.as_tensor(x, dtype=dt,
                                                   device=self._device)
                obs_t, next_obs_t = to(b_obs, torch.float32), to(b_next_obs, torch.float32)
                g_t, next_g_t = to(b_g, torch.long), to(b_next_g, torch.long)
                a_t, r_t = to(b_a, torch.long), to(b_r, torch.float32)
                done_t = to(b_done, torch.float32)

                wm_total, wm_parts = self._wm.loss(
                    obs_t, g_t, a_t, r_t, next_obs_t, next_g_t)
                wm_opt.zero_grad()
                wm_total.backward()
                torch.nn.utils.clip_grad_norm_(self._wm.parameters(), 10.0)
                wm_opt.step()

                with torch.no_grad():
                    zc = self._wm.encode(obs_t, g_t)
                    zc_next = self._wm.encode(next_obs_t, next_g_t)
                sac_parts = self._sac.update(
                    zc, a_t, r_t, zc_next, done_t, q_opt, actor_opt)

                train_step += 1
                if unified_logger is not None and train_step % log_every == 0:
                    unified_logger.set_progress(env_steps=t,
                                                train_steps=train_step)
                    for tag, v in {**wm_parts, **sac_parts}.items():
                        unified_logger.log_scalar(
                            f"train/{tag}", v, train_step=train_step)
                if probe is not None:
                    self._wm.eval()
                    self._sac.eval()
                    probe.maybe_run(t, train_steps=train_step)
                    self._wm.train()
                    self._sac.train()
        if unified_logger is not None:
            unified_logger.set_progress(env_steps=total,
                                        train_steps=max(train_step, 1))
        if probe is not None:
            probe.close()
        env.close()

    # ------------------------------------------------------------- evaluate
    def evaluate(self, env_fn: Callable[[], Any], regime_grid, episodes) -> EvalReport:
        if self._wm is None:
            return super().evaluate(env_fn, regime_grid, episodes)
        import torch

        t0 = time.time()
        self._wm.eval()
        self._sac.eval()
        env = env_fn()
        agents = list(env.possible_agents)

        return_per_regime: dict[int, float] = {}
        return_per_regime_sem: dict[int, float] = {}
        return_per_regime_per_agent: dict = {}
        episodes_per_regime: dict[int, int] = {}
        all_returns: list[float] = []
        env_steps_total = 0

        gen = torch.Generator(device=self._device)
        for g in regime_grid:
            g_returns: list[float] = []
            g_agent: list = []
            for ep in range(int(episodes)):
                gen.manual_seed(50_000 + 977 * int(g) + ep)
                obs_dict, info = env.reset(
                    seed=50_000 + 97 * int(g) + ep, options={"g": int(g)})
                _check_forbidden_info(info)
                obs = np.stack([np.asarray(obs_dict[a], dtype=np.float32)
                                for a in agents])
                ep_ret, done = 0.0, False
                ep_agent = np.zeros(len(agents), dtype=np.float64)
                while not done:
                    # pinned eval-grid regime ID (given-ID protocol)
                    acts = self._planner.plan(obs, int(g), explore=False,
                                              generator=gen)
                    action_dict = {a: int(acts[i])
                                   for i, a in enumerate(agents)}
                    obs_dict, rew, term, trunc, info = env.step(action_dict)
                    _check_forbidden_info(info)
                    obs = np.stack([np.asarray(obs_dict[a], dtype=np.float32)
                                    for a in agents])
                    step_vec = reward_vector(rew, agents)
                    ep_agent += step_vec
                    ep_ret += float(step_vec.sum())
                    env_steps_total += 1
                    done = bool(any(term.values()) or any(trunc.values()))
                g_returns.append(ep_ret)
                g_agent.append(ep_agent)
            return_per_regime[int(g)] = float(np.mean(g_returns)) if g_returns else 0.0
            return_per_regime_sem[int(g)] = (
                float(np.std(g_returns) / max(np.sqrt(len(g_returns)), 1.0))
                if len(g_returns) > 1 else 0.0
            )
            episodes_per_regime[int(g)] = len(g_returns)
            return_per_regime_per_agent[int(g)] = (
                np.mean(g_agent, axis=0) if g_agent else np.zeros(len(agents))
            )
            all_returns.extend(g_returns)
        env.close()
        self._wm.train()
        self._sac.train()

        zs_seen, zs_unseen = split_seen_unseen_regimes(self.cfg, return_per_regime)
        return_mean = float(np.mean(all_returns)) if all_returns else 0.0
        return_sem = (
            float(np.std(all_returns) / max(np.sqrt(len(all_returns)), 1.0))
            if len(all_returns) > 1 else 0.0
        )
        return EvalReport(
            variant=self.name,
            regime_names=regime_names_for(self.cfg),
            seed=0,
            config_hash="0" * 40,
            eval_mode="planner",
            eval_planner_mode=self.cfg.eval.eval_planner_mode,
            return_mean=return_mean,
            return_sem=return_sem,
            return_zero_shot_seen=zs_seen,
            return_zero_shot_unseen=zs_unseen,
            return_zero_shot_gap=zs_seen - zs_unseen,
            return_per_regime=MappingProxyType(return_per_regime),
            return_per_regime_sem=MappingProxyType(return_per_regime_sem),
            return_per_regime_per_agent=freeze_per_agent(
                return_per_regime_per_agent
            ),
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

    # ------------------------------------------------------------------ ckpt
    def save_checkpoint(self, path: Union[Path, str]) -> None:
        import torch

        if self._wm is None:
            return
        torch.save({
            "obs_dim": self._wm.obs_dim,
            "wm": self._wm.state_dict(),
            "sac": self._sac.state_dict(),
        }, str(path))

    def load_checkpoint(self, path: Union[Path, str]) -> None:
        import torch

        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        if self._wm is None:
            self._build(int(ckpt["obs_dim"]))
        self._wm.load_state_dict(ckpt["wm"])
        self._sac.load_state_dict(ckpt["sac"])

    def param_count(self) -> int:
        if self._wm is None:
            return 0
        return int(sum(p.numel() for p in self._wm.parameters())
                   + sum(p.numel() for p in self._sac.parameters()))

    # ------------------------------------------------------- fidelity hook
    def predict_rewards(self, episode) -> Optional[np.ndarray]:
        """fidelity-v1 hook: one-step per-agent reward predictions (T, N)
        from the SparseMoE reward model, conditioned on the episode's GIVEN
        regime ID (the disclosed oracle-ID protocol)."""
        if self._wm is None:
            return None
        import torch

        was_training = self._wm.training
        self._wm.eval()
        try:
            with torch.no_grad():
                obs = torch.as_tensor(
                    np.asarray(episode["obs"][:-1], dtype=np.float32),
                    device=self._device)                       # (T, N, D)
                a = torch.as_tensor(
                    np.asarray(episode["actions"], dtype=np.int64),
                    device=self._device)                       # (T, N)
                g = torch.full((obs.shape[0],), int(episode["g"]),
                               dtype=torch.long, device=self._device)
                pred = self._wm.predict_rewards(self._wm.encode(obs, g), a)
            return pred.cpu().numpy().astype(np.float64)
        finally:
            if was_training:
                self._wm.train()
