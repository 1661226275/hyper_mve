"""HAPPO comparison runner — drives the vendored HARL clone (phase-4).

Design (realignment plan): the published algorithm is NOT reimplemented —
``harl.runners.RUNNER_REGISTRY["happo"]`` (OnPolicyHARunner: sequential
per-agent updates with factor bookkeeping and randomized order) runs as-is.
Integration happens at the enumerated adaptive-edit surface inside the clone
(``harl/envs/relation/*``, one ``elif`` per env-maker, the logger registry
entry, ``envs_cfgs/relation.yaml``) plus this adapter.

Per-agent rewards (the mixed-game requirement) come natively from HARL's
``state_type="FP"`` critic path — the relation env emits the true per-agent
relational rewards and HARL computes per-agent returns/advantages.

Deviations disclosed:
  * ``train`` ignores ``env_fn`` — HARL builds envs through its own vec-env
    machinery; the identity is pinned by passing the SAME ``cfg.env``
    (EnvConfig) through ``env_args["env_cfg"]``, oracle-free. Live objects in
    ``env_args`` require ``n_rollout_threads=1`` (in-process DummyVecEnv),
    which this adapter enforces.
  * ``evaluate`` DOES consume ``env_fn``: deterministic per-regime rollouts
    of the trained per-agent actors (decentralized execution), standard
    rel-v1 EvalReport assembly.
"""
from __future__ import annotations

import copy
import os
import sys
import tempfile
import time
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Optional, Union

import numpy as np

from hyper_mve.utils.configs import V4Config
from hyper_mve.utils.eval.eval_report import EvalReport
from hyper_mve.comparison.base import (
    ExternalBaselineRunner,
    _FORBIDDEN_INFO_KEYS,
    split_seen_unseen_regimes,
)

_HARL_DIR = Path(__file__).resolve().parent / "vendor" / "HARL"

# Side channel for live (non-JSON-safe) objects: HARL json-dumps env_args via
# save_config, so the EnvConfig / UnifiedLogger cannot ride in env_args. The
# clone's relation_env.py / relation_logger.py read these slots instead.
# Single-process only (n_rollout_threads=1 enforced below).
_ACTIVE: dict = {"env_cfg": None, "unified_logger": None, "probe": None}


def get_active_env_cfg():
    return _ACTIVE["env_cfg"]


def get_active_unified_logger():
    return _ACTIVE["unified_logger"]


def get_active_probe():
    """PeriodicEvalProbe for the in-flight train() call, or None.

    HAPPO owns no step-loop of its own (``self._runner.run()`` is a single
    blocking call into unmodified vendored HARL) -- relation_logger.py's
    episode_log, which HARL already calls back into periodically, is the
    only available hook for a periodic probe. Constructed once in train()
    (where self._runner.actor is reachable) and read from there."""
    return _ACTIVE["probe"]


def _ensure_harl_on_path() -> None:
    p = str(_HARL_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def _check_forbidden_info(info_dict) -> None:
    for agent_info in info_dict.values():
        leaked = _FORBIDDEN_INFO_KEYS & set(agent_info)
        assert not leaked, f"forbidden info keys leaked to HAPPO: {leaked}"


class HAPPORunner(ExternalBaselineRunner):
    name = "happo"

    def __init__(self, cfg: V4Config):
        self.cfg = cfg
        self._runner = None            # harl OnPolicyHARunner
        self._lr: float = 0.0

    # ------------------------------------------------------------ internals
    def _build_runner(self, *, total_env_steps: int, lr: float, seed: int,
                      log_dir: Optional[str], unified_logger=None):
        _ensure_harl_on_path()
        import yaml
        from harl.runners import RUNNER_REGISTRY

        algo_args = yaml.safe_load(
            (_HARL_DIR / "harl" / "configs" / "algos_cfgs" / "happo.yaml")
            .read_text(encoding="utf-8")
        )
        T = int(self.cfg.env.T_max)
        algo_args["seed"]["seed_specify"] = True
        algo_args["seed"]["seed"] = int(seed)
        algo_args["train"]["n_rollout_threads"] = 1   # live env_args objects
        algo_args["train"]["num_env_steps"] = int(max(total_env_steps, T))
        algo_args["train"]["episode_length"] = T
        algo_args["train"]["log_interval"] = 1
        algo_args["eval"]["use_eval"] = False
        algo_args["render"]["use_render"] = False
        if lr and lr > 0:
            algo_args["model"]["lr"] = float(lr)
            algo_args["model"]["critic_lr"] = float(lr)
        # HARL builds its OWN tensorboardX SummaryWriter under this dir
        # (utils/configs_tools.py) and emits the full actor/critic scalar set
        # against env steps. Pointing it at the run's tb/ therefore buries a
        # second, differently-x-axed event tree inside the UnifiedLogger's
        # directory, which TensorBoard then loads as extra runs. Keep HARL's
        # raw tree OUTSIDE tb/ (as a sibling, so it is still inspectable) —
        # the canonical scalars reach tb/ through the unified funnel via
        # relation_logger.py.
        if log_dir:
            harl_dir = os.path.join(os.path.dirname(os.path.normpath(log_dir)),
                                    "harl_raw")
            os.makedirs(harl_dir, exist_ok=True)
        else:
            harl_dir = tempfile.mkdtemp(prefix="happo_harl_")
        algo_args["logger"]["log_dir"] = harl_dir
        # JSON-safe env_args (HARL's save_config dumps them); live objects go
        # through the module side channel read by the clone's relation files.
        env_args = {
            "state_type": "FP",                    # per-agent rewards (mixed-game)
            "preset": self.cfg.preset_name,
        }
        _ACTIVE["env_cfg"] = self.cfg.env
        _ACTIVE["unified_logger"] = unified_logger
        args = {"algo": "happo", "env": "relation", "exp_name": "runner"}
        # slots stay set for the runner's lifetime (the logger emits during run())
        return RUNNER_REGISTRY["happo"](args, algo_args, env_args)

    # -------------------------------------------------------------- train
    def train(self, cfg, env_fn, *, total_env_steps: int = 0, lr: float = 0.0,
              seed: int = 0, **kwargs) -> None:
        self.cfg = cfg
        self._lr = float(lr)
        unified_logger = kwargs.get("unified_logger")
        tensorboard_dir = kwargs.get("tensorboard_dir")
        self._runner = self._build_runner(
            total_env_steps=int(total_env_steps), lr=lr, seed=int(seed),
            log_dir=tensorboard_dir, unified_logger=unified_logger,
        )

        probe = None
        if tensorboard_dir or unified_logger is not None:
            from hyper_mve.comparison._probe import PeriodicEvalProbe

            probe_writer = unified_logger
            if probe_writer is None:
                from torch.utils.tensorboard import SummaryWriter

                probe_writer = SummaryWriter(tensorboard_dir)

            recurrent_n = int(self._runner.recurrent_n)
            rnn_hidden = int(self._runner.rnn_hidden_size)
            n_agents = int(cfg.env.N)
            probe_rnn = {"state": None}

            def _probe_act(obs: np.ndarray, t: int, g: int) -> np.ndarray:
                del g
                if t == 0:
                    probe_rnn["state"] = [
                        np.zeros((1, recurrent_n, rnn_hidden), dtype=np.float32)
                        for _ in range(n_agents)
                    ]
                masks = np.ones((1, 1), dtype=np.float32)
                acts = np.zeros(n_agents, dtype=np.int64)
                for i in range(n_agents):
                    action, rnn_i = self._runner.actor[i].act(
                        obs[i][None], probe_rnn["state"][i], masks, None,
                        deterministic=True,
                    )
                    probe_rnn["state"][i] = rnn_i.detach().cpu().numpy()
                    acts[i] = int(action.detach().cpu().numpy().reshape(-1)[0])
                return acts

            probe = PeriodicEvalProbe(
                env_fn, cfg, probe_writer, act_fn=_probe_act,
                every_train_steps=500, episodes_per_regime=8,
            )
        _ACTIVE["probe"] = probe

        try:
            self._runner.run()
        finally:
            _ACTIVE["probe"] = None
            if probe is not None:
                probe.close()
        # release HARL's train envs / writer; actors stay in memory for eval
        self._runner.close()

    # ------------------------------------------------------------ evaluate
    def evaluate(self, env_fn: Callable[[], Any], regime_grid, episodes) -> EvalReport:
        if self._runner is None:
            # untrained runner: schema-complete zero-valued report (base default)
            return super().evaluate(env_fn, regime_grid, episodes)
        import torch

        t0 = time.time()
        env = env_fn()
        agents = list(env.possible_agents)
        N = len(agents)
        recurrent_n = int(self._runner.recurrent_n)
        rnn_hidden = int(self._runner.rnn_hidden_size)

        return_per_regime: dict[int, float] = {}
        return_per_regime_sem: dict[int, float] = {}
        episodes_per_regime: dict[int, int] = {}
        all_returns: list[float] = []
        env_steps_total = 0

        with torch.no_grad():
            for g in regime_grid:
                g_returns: list[float] = []
                for ep in range(int(episodes)):
                    obs_dict, info = env.reset(
                        seed=20_000 + 97 * int(g) + ep, options={"g": int(g)}
                    )
                    _check_forbidden_info(info)
                    rnn = [np.zeros((1, recurrent_n, rnn_hidden), dtype=np.float32)
                           for _ in range(N)]
                    masks = np.ones((1, 1), dtype=np.float32)
                    ep_ret, done = 0.0, False
                    while not done:
                        acts = {}
                        for i, a in enumerate(agents):
                            obs_i = np.asarray(obs_dict[a], dtype=np.float32)[None]
                            action, rnn_i = self._runner.actor[i].act(
                                obs_i, rnn[i], masks, None, deterministic=True
                            )
                            rnn[i] = rnn_i.detach().cpu().numpy()
                            acts[a] = int(action.detach().cpu().numpy().reshape(-1)[0])
                        obs_dict, rew, term, trunc, info = env.step(acts)
                        _check_forbidden_info(info)
                        ep_ret += float(sum(rew.values()))
                        env_steps_total += 1
                        done = bool(any(term.values()) or any(trunc.values()))
                    g_returns.append(ep_ret)
                return_per_regime[int(g)] = float(np.mean(g_returns)) if g_returns else 0.0
                return_per_regime_sem[int(g)] = (
                    float(np.std(g_returns) / max(np.sqrt(len(g_returns)), 1.0))
                    if len(g_returns) > 1 else 0.0
                )
                episodes_per_regime[int(g)] = len(g_returns)
                all_returns.extend(g_returns)
        env.close()

        zs_seen, zs_unseen = split_seen_unseen_regimes(self.cfg, return_per_regime)
        return_mean = float(np.mean(all_returns)) if all_returns else 0.0
        return_sem = (
            float(np.std(all_returns) / max(np.sqrt(len(all_returns)), 1.0))
            if len(all_returns) > 1 else 0.0
        )
        return EvalReport(
            variant="happo",
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

    # ---------------------------------------------------------------- ckpt
    def save_checkpoint(self, path: Union[Path, str]) -> None:
        import torch

        if self._runner is None:
            return
        state = {
            "actors": [a.actor.state_dict() for a in self._runner.actor],
            "critic": self._runner.critic.critic.state_dict(),
        }
        if self._runner.value_normalizer is not None:
            state["value_normalizer"] = self._runner.value_normalizer.state_dict()
        torch.save(state, str(path))

    def load_checkpoint(self, path: Union[Path, str]) -> None:
        import torch

        if self._runner is None:
            # build an untrained runner shell of identical shape (0-episode run)
            self._runner = self._build_runner(
                total_env_steps=0, lr=self._lr, seed=0, log_dir=None,
            )
            self._runner.close()
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        for agent, sd in zip(self._runner.actor, ckpt["actors"]):
            agent.actor.load_state_dict(sd)
        self._runner.critic.critic.load_state_dict(ckpt["critic"])
        if "value_normalizer" in ckpt and self._runner.value_normalizer is not None:
            self._runner.value_normalizer.load_state_dict(ckpt["value_normalizer"])

    def param_count(self) -> int:
        if self._runner is None:
            return 0
        n = sum(p.numel() for a in self._runner.actor for p in a.actor.parameters())
        n += sum(p.numel() for p in self._runner.critic.critic.parameters())
        return int(n)
