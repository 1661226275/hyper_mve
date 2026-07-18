"""MBOM / MBOM-oracle comparison runners — drive the vendored MBOM clone (phase-5).

MBOM (Yu et al., model-based opponent modeling) is strictly 2-player and does
NOT learn an environment model: its recursive imagined best-response rollouts
require an externally supplied differentiable batched simulator. Both facts
shape the integration (user-locked 2026-07-17):

* **RelationCommons duo ONLY** (``cfg.env.N == 2`` enforced; simple_tag N/A,
  disclosed).
* Two registry rows differing ONLY in the simulator's information set:
    - ``mbom``        — physical model WITHOUT the hidden ``W(g)``: reward =
      own harvest minus move cost (``W = I``). Information parity with our
      method, which must infer the regime.
    - ``mbom_oracle`` — true relational reward using the agent's own W-row
      (read from its observation tail): an upper-bound row.
  The dynamics part (movement / fair-share harvest / regrowth / time) is the
  TRUE simulator in both rows — exact torch port of
  ``hyper_mve/envs/relation_commons/{env,dynamics}.py``.

Upstream-source integrity: the algorithm classes (``policy/MBOM.py``,
``baselines/PPO.py`` + ``PPO_Buffer``) and the trajectory collection
(``utils/rl_utils.collect_trajectory``) run unmodified (one enumerated
defect patch in ``rl_utils.py``, see VENDOR.md). The upstream ``trainer.py``
multiprocessing scaffold is replaced by its exact single-process equivalent
(the ``worker()`` loop at ``ranks=1`` without IPC): same construction, same
collection call, same ``learn`` cadence.

Protocol notes (disclosed):
* The trained joint policy = PPO (agent 0) + MBOM (agent 1), the upstream
  pairing; EvalReport returns are the social sum of that joint policy.
* ``is_prophetic=False`` / ``prophetic_onehot=False``: MBOM observes only
  the opponent's executed actions (its opponent-model learning path) — no
  privileged access to the REAL opponent's policy distribution.
* ``true_prob=True`` mirrors the SHIPPED clone state: ``base/MLP.py`` has
  the "trub prob" return active (returns the opponent-MODEL's action prob,
  not the hidden layer; upstream couples this flag to that hand-toggled
  line — see main.py's ``--true_prob`` help). The prob comes from MBOM's
  own learned/imagined opponent model, so no privilege is added.
* MBOM consults its supplied simulator at evaluation too (imagined rollouts
  are part of the method's acting path).
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, Optional, Union

import numpy as np

from hyper_mve.utils.configs import V4Config
from hyper_mve.utils.eval.eval_report import EvalReport
from hyper_mve.comparison.base import (
    ExternalBaselineRunner,
    _FORBIDDEN_INFO_KEYS,
    split_seen_unseen_regimes,
)

_MBOM_DIR = Path(__file__).resolve().parent / "vendor" / "MBOM"

# RelationCommons action encoding (envs/relation_commons/spaces.py):
# 0=NOOP, 1=UP(+y), 2=DOWN(-y), 3=LEFT(-x), 4=RIGHT(+x), 5=HARVEST
_DELTAS = ((0, 0), (0, 1), (0, -1), (-1, 0), (1, 0), (0, 0))


def _ensure_mbom_on_path() -> None:
    p = str(_MBOM_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def _check_forbidden_info(info_dict) -> None:
    for agent_info in info_dict.values():
        leaked = _FORBIDDEN_INFO_KEYS & set(agent_info)
        assert not leaked, f"forbidden info keys leaked to MBOM: {leaked}"


# --------------------------------------------------------------------- facade
class _DuoEnvFacade:
    """MBOM env API (list-based, 2 agents) over RelationCommonsPettingZooEnv."""

    def __init__(self, env):
        self._env = env
        self._agents = list(env.possible_agents)
        assert len(self._agents) == 2, "MBOM is strictly 2-player"
        self._episode = 0

    def reset(self):
        self._episode += 1
        obs_dict, info = self._env.reset(seed=31_000 + self._episode)
        _check_forbidden_info(info)
        return [np.asarray(obs_dict[a], dtype=np.float32) for a in self._agents]

    def step(self, actions):
        acts = np.asarray(actions).astype(np.int64).flatten()
        action_dict = {a: int(acts[i]) for i, a in enumerate(self._agents)}
        obs_dict, rew, term, trunc, info = self._env.step(action_dict)
        _check_forbidden_info(info)
        obs = [np.asarray(obs_dict[a], dtype=np.float32) for a in self._agents]
        reward = [float(rew[a]) for a in self._agents]
        done = bool(any(term.values()) or any(trunc.values()))
        return obs, reward, done, dict(info[self._agents[0]])


# ------------------------------------------------------------------ env model
class RelationDuoEnvModel:
    """Differentiable-API batched simulator for MBOM's imagined rollouts.

    Torch port of the RelationCommons step (movement → fair-share harvest on
    pre-update stocks → constant-rate regrowth → reward → time), operating
    directly on agent 1's observation vector (the five-block v5 layout is
    state-sufficient for the duo: own block, all K resource cells, the
    opponent's relative position, time, own W-row).

    ``reward_mode``:
      * ``"own_harvest"`` — R_i = u_i − ε·moved_i  (W = I; no regime access)
      * ``"true_W"``      — R_1 uses the agent's OWN row w (from the obs
        tail): R_1 = (u_1 + w·u_0)/(1+|w|) − ε·moved_1  (oracle row)
    R_0 is returned as the W=I form in both modes — MBOM's rollout only ever
    consumes ``reward[agent_idx=1]``.
    """

    def __init__(self, env_cfg, device, reward_mode: str = "own_harvest"):
        import torch

        assert env_cfg.N == 2, "RelationDuoEnvModel is duo-only"
        assert reward_mode in ("own_harvest", "true_W"), reward_mode
        self.K = int(env_cfg.K)
        self.L = int(env_cfg.L)
        self.T_max = int(env_cfg.T_max)
        self.alpha = float(env_cfg.alpha)
        self.q_max = float(env_cfg.Q_max)
        self.eps_move = float(env_cfg.epsilon_move)
        self.reward_mode = reward_mode
        self.device = device
        self._deltas = torch.tensor(_DELTAS, dtype=torch.float32, device=device)
        # five-block layout offsets for N=2 (self 4 | resource 3K | neighbor 9
        # | global 1 | row 1)
        self._res0 = 4
        self._nb0 = 4 + 3 * self.K
        self._t0 = self._nb0 + 9
        self._row0 = self._t0 + 1
        self.n_state = self._row0 + 1

    def reset(self) -> None:  # rollout-batch lifecycle hook (stateless model)
        pass

    def step(self, state, actions):
        import torch

        s = state.to(self.device)
        B = s.shape[0]
        a_me = torch.as_tensor(actions[1]).reshape(-1).long().to(self.device)
        a_op = torch.as_tensor(actions[0]).reshape(-1).long().to(self.device)
        K, Lm1, T = self.K, float(self.L - 1), float(self.T_max)

        # ------- decode (agent 1's obs; grid quantities are exact under round)
        own = torch.round(s[:, 0:2] * Lm1)                        # (B,2)
        cum = s[:, 2] * T                                          # η = 1
        since = s[:, 3] * T
        res = s[:, self._res0:self._res0 + 3 * K].view(B, K, 3)
        res_pos = torch.round(res[:, :, 0:2] + own.unsqueeze(1))   # (B,K,2)
        q = res[:, :, 2] * self.q_max
        other = torch.round(own + s[:, self._nb0:self._nb0 + 2])   # (B,2)
        t_rem = s[:, self._t0]
        w = s[:, self._row0]

        # ------- 1-2. move intent + deterministic clipped moves
        moved_me = (a_me >= 1) & (a_me <= 4)
        moved_op = (a_op >= 1) & (a_op <= 4)
        own_n = (own + self._deltas[a_me.clamp(0, 5)]).clamp(0.0, Lm1)
        other_n = (other + self._deltas[a_op.clamp(0, 5)]).clamp(0.0, Lm1)

        # ------- 3. fair-share harvest on PRE-update stocks (η = 1)
        harv_me = (a_me == 5)
        harv_op = (a_op == 5)
        on_me = (res_pos == own_n.unsqueeze(1)).all(-1) & harv_me.unsqueeze(1)
        on_op = (res_pos == other_n.unsqueeze(1)).all(-1) & harv_op.unsqueeze(1)
        cnt = on_me.float() + on_op.float()                        # (B,K)
        share = torch.where(cnt > 0, q / cnt.clamp(min=1.0),
                            torch.zeros_like(q))
        u_cell = torch.minimum(share, torch.ones_like(share))
        u_me = (u_cell * on_me.float()).sum(-1)                    # (B,)
        u_op = (u_cell * on_op.float()).sum(-1)
        h_per_res = u_cell * cnt

        # ------- 5. constant-rate regen (post-harvest)
        q_n = (q + self.alpha * (self.q_max - q) - h_per_res).clamp(
            0.0, self.q_max)

        # ------- 6. rewards
        r0 = u_op - self.eps_move * moved_op.float()
        if self.reward_mode == "true_W":
            r1 = (u_me + w * u_op) / (1.0 + torch.abs(w)) \
                - self.eps_move * moved_me.float()
        else:
            r1 = u_me - self.eps_move * moved_me.float()

        # ------- 7. time + counters
        cum_n = cum + u_me
        since_n = torch.where(u_me > 0, torch.zeros_like(since), since + 1.0)
        t_n = t_rem - 1.0 / T
        done = t_n < (0.5 / T)

        # ------- rebuild agent 1's next obs (five-block layout)
        onehot_op = torch.zeros(B, 6, device=self.device)
        onehot_op[torch.arange(B, device=self.device), a_op.clamp(0, 5)] = 1.0
        state_ = torch.cat([
            own_n / Lm1,                                           # self x,y
            (cum_n / T).unsqueeze(1),
            (since_n / T).unsqueeze(1),
            torch.cat([res_pos - own_n.unsqueeze(1),
                       (q_n / self.q_max).unsqueeze(-1)], dim=-1).view(B, 3 * K),
            (other_n - own_n),                                     # neighbor rel
            onehot_op,
            torch.ones(B, 1, device=self.device),                  # presence
            t_n.unsqueeze(1),
            w.unsqueeze(1),                                        # row (static)
        ], dim=1)
        return state_, [r0.view(-1, 1), r1.view(-1, 1)], done.view(-1, 1)


# --------------------------------------------------------------------- confs
def _build_confs(env_cfg, *, lr: float, eps_per_epoch: int) -> tuple[dict, dict]:
    """Two conf dicts modeled on ``config/gfootball_conf.py`` (n_state=39,
    n_action=6). Level-1 recursion (num_om_layers=2) with roll_out_length=2
    (imagined branching 6²=36 per step)."""
    import math

    n_state = 4 + 3 * env_cfg.K + 9 * (env_cfg.N - 1) + 1 + (env_cfg.N - 1)
    buffer_size = max(4 * eps_per_epoch * env_cfg.T_max, 512)
    base = {
        "n_state": n_state,
        "n_action": int(env_cfg.A),
        "n_opponent_action": int(env_cfg.A),
        "action_dim": 1,
        "type_action": "discrete",
        "action_bounding": 0,
        "action_scaling": [1, 1],
        "action_offset": [0, 0],
        "v_hidden_layers": [64, 32],
        "a_hidden_layers": [64, 32],
        "v_learning_rate": lr,
        "a_learning_rate": lr,
        "gamma": 0.99,
        "lambda": 0.95,
        "epsilon": 0.115,
        "entcoeff": 0.0015,
        "a_update_times": 10,
        "v_update_times": 10,
        "buffer_memory_size": buffer_size,
        "num_om_layers": 1,
        "opponent_model_hidden_layers": [64, 32],
        "opponent_model_memory_size": 1000,
        "opponent_model_learning_rate": 0.001,
        "opponent_model_batch_size": 64,
        "opponent_model_learning_times": 1,
    }
    conf_ppo = dict(base, conf_id="relation_ppo_conf")
    conf_mbom = dict(
        base,
        conf_id="relation_mbom_conf",
        num_om_layers=2,
        imagine_model_learning_rate=0.001,
        imagine_model_learning_times=5,
        roll_out_length=2,
        short_term_decay=0.9,
        short_term_horizon=10,
        mix_factor=1.1 / math.e,
    )
    return conf_ppo, conf_mbom


# -------------------------------------------------------------------- runner
class MBOMRunner(ExternalBaselineRunner):
    name = "mbom"
    _reward_mode = "own_harvest"

    def __init__(self, cfg: V4Config):
        self.cfg = cfg
        self._agents = None            # [PPO(agent0), MBOM(agent1)]
        self._env_model = None
        self._device = None

    # ------------------------------------------------------------ internals
    def _make_args(self, *, eps_per_epoch: int, max_epoch: int) -> SimpleNamespace:
        return SimpleNamespace(
            train_mode=0, ranks=1,
            eps_per_epoch=int(eps_per_epoch),
            max_epoch=int(max_epoch),
            save_per_epoch=10 ** 9,
            actor_rnn=False, rnn_mixer=False,
            true_prob=True, prophetic_onehot=False,
            device=str(self._device),
        )

    def _build_agents(self, *, lr: float, eps_per_epoch: int, max_epoch: int,
                      log_root: str):
        _ensure_mbom_on_path()
        import torch
        from baselines.PPO import PPO, PPO_Buffer
        from policy.MBOM import MBOM
        from utils.Logger import Logger

        # The clone's only working device path is device=None (implicit CPU):
        # Base_ActorCritic.change_device raises NotImplementedError for any
        # non-None device, and the opponent-model mixing is CUDA-unsafe.
        # The nets are tiny 39-dim MLPs — CPU is the upstream-faithful device.
        self._device = torch.device("cpu")
        args = self._make_args(eps_per_epoch=eps_per_epoch, max_epoch=max_epoch)
        conf_ppo, conf_mbom = _build_confs(
            self.cfg.env, lr=(lr if lr and lr > 0 else 0.001),
            eps_per_epoch=eps_per_epoch,
        )
        logger = Logger(log_root, "mbom_runner", 0)
        self._env_model = RelationDuoEnvModel(
            self.cfg.env, self._device, self._reward_mode)
        ppo = PPO(args, conf_ppo, name="relation_rank0", logger=logger,
                  actor_rnn=False, device=None)
        mbom = MBOM(args=args, conf=conf_mbom, name="relation", logger=logger,
                    agent_idx=1, actor_rnn=False, env_model=self._env_model,
                    device=None)
        buffers = [
            PPO_Buffer(args=args, conf=agent.conf, name=agent.name,
                       actor_rnn=False, device=None)
            for agent in (ppo, mbom)
        ]
        return args, [ppo, mbom], buffers

    # -------------------------------------------------------------- train
    def train(self, cfg, env_fn, *, total_env_steps: int = 0, lr: float = 0.0,
              seed: int = 0, **kwargs) -> None:
        self.cfg = cfg
        assert cfg.env.N == 2, "MBOM baseline is RelationCommons-duo only"
        _ensure_mbom_on_path()
        import torch
        from utils.rl_utils import collect_trajectory

        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

        T = int(cfg.env.T_max)
        eps_per_epoch = 4
        max_epoch = max(1, int(total_env_steps) // (eps_per_epoch * T))
        log_root = kwargs.get("tensorboard_dir") or tempfile.mkdtemp(
            prefix="mbom_")
        unified_logger = kwargs.get("unified_logger")

        args, agents, buffers = self._build_agents(
            lr=lr, eps_per_epoch=eps_per_epoch, max_epoch=max_epoch,
            log_root=log_root,
        )
        env = _DuoEnvFacade(env_fn())

        # Single-process equivalent of upstream trainer.worker() at ranks=1
        # (same construction, same collection, same learn cadence — the mp
        # queue plumbing is the only thing dropped).
        global_step = 0
        for epoch in range(1, args.max_epoch + 1):
            memory, scores, global_step = collect_trajectory(
                agents, env, args, global_step, is_prophetic=False,
            )
            for i in range(2):
                buffers[i].store_multi_memory(memory[i], last_val=0)
            agents[0].learn(data=buffers[0].get_batch(), iteration=epoch,
                            no_log=True)
            agents[1].learn(data=buffers[1].get_batch(), iteration=epoch,
                            no_log=True)
            if unified_logger is not None:
                unified_logger.set_progress(env_steps=int(global_step))
                unified_logger.advance(train_steps=1)
                unified_logger.log_scalar(
                    "train/score_ppo_agent0", float(scores[0]),
                    env_step=int(global_step))
                unified_logger.log_scalar(
                    "train/score_mbom_agent1", float(scores[1]),
                    env_step=int(global_step))
        self._agents = agents

    # ------------------------------------------------------------ evaluate
    def evaluate(self, env_fn: Callable[[], Any], regime_grid, episodes) -> EvalReport:
        if self._agents is None:
            return super().evaluate(env_fn, regime_grid, episodes)
        import torch

        t0 = time.time()
        env = env_fn()
        agents_names = list(env.possible_agents)
        N = len(agents_names)
        assert N == 2

        return_per_regime: dict[int, float] = {}
        return_per_regime_sem: dict[int, float] = {}
        episodes_per_regime: dict[int, int] = {}
        all_returns: list[float] = []
        env_steps_total = 0

        # NO torch.no_grad() here: MBOM's acting path runs gradient-based
        # imagined opponent-model fine-tuning inside choose_action (that IS
        # the method); upstream collect_trajectory/tester call it bare too.
        for g in regime_grid:
            g_returns: list[float] = []
            for ep in range(int(episodes)):
                obs_dict, info = env.reset(
                    seed=40_000 + 97 * int(g) + ep, options={"g": int(g)})
                _check_forbidden_info(info)
                state = [np.asarray(obs_dict[a], dtype=np.float32)
                         for a in agents_names]
                ep_ret, done = 0.0, False
                while not done:
                    acts = {}
                    for i, a in enumerate(agents_names):
                        action_info = self._agents[i].choose_action(
                            state[i], greedy=True, hidden_state=None,
                            oppo_hidden_prob=None)
                        acts[a] = int(np.asarray(action_info[0]).reshape(-1)[0])
                    obs_dict, rew, term, trunc, info = env.step(acts)
                    _check_forbidden_info(info)
                    state = [np.asarray(obs_dict[a], dtype=np.float32)
                             for a in agents_names]
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
            variant=self.name,
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

        if self._agents is None:
            return
        ppo, mbom = self._agents
        torch.save({
            "reward_mode": self._reward_mode,
            "ppo": {"a_net": ppo.a_net.state_dict(),
                    "v_net": ppo.v_net.state_dict()},
            "mbom": {"a_net": mbom.a_net.state_dict(),
                     "v_net": mbom.v_net.state_dict(),
                     "om_phi0": mbom.om_phis[0]},
        }, str(path))

    def load_checkpoint(self, path: Union[Path, str]) -> None:
        import torch

        if self._agents is None:
            _, self._agents, _ = self._build_agents(
                lr=0.001, eps_per_epoch=4, max_epoch=1,
                log_root=tempfile.mkdtemp(prefix="mbom_load_"),
            )
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        ppo, mbom = self._agents
        ppo.a_net.load_state_dict(ckpt["ppo"]["a_net"])
        ppo.v_net.load_state_dict(ckpt["ppo"]["v_net"])
        mbom.a_net.load_state_dict(ckpt["mbom"]["a_net"])
        mbom.v_net.load_state_dict(ckpt["mbom"]["v_net"])
        mbom.om_phis[0] = ckpt["mbom"]["om_phi0"]

    def param_count(self) -> int:
        if self._agents is None:
            return 0
        n = 0
        for agent in self._agents:
            n += sum(p.numel() for p in agent.a_net.parameters())
            n += sum(p.numel() for p in agent.v_net.parameters())
        return int(n)


class MBOMOracleRunner(MBOMRunner):
    name = "mbom_oracle"
    _reward_mode = "true_W"
