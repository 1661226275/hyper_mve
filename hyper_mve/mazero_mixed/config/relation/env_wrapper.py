"""RelationCommons → MAZero ``Game`` adapter (fork stage 1, pristine cooperative base).

Consumes the repo's PettingZoo wrapper (``oracle_mode=False, eval_info_mode=False``
— the same info-filtering discipline the external baselines use, so oracle-only
keys such as the regime id never reach the model), and converts it to MAZero's
``core.game.Game`` contract.

Stage-1 keeps MAZero's cooperative A1 contract intact: the scalar team reward
is the **mean of the per-agent relational rewards**, exactly mirroring how the
vendored MAMBA port and MAZero's own SMACWrapper aggregate rewards. The raw
per-agent reward vector is preserved in ``info["rewards_per_agent"]`` so the
stage-3 mixed-game migration can surface it without touching the env layer.
"""
from typing import List

import numpy as np

from core.game import Game


class RelationCommonsGame(Game):
    def __init__(self, env, T_max: int):
        # env: hyper_mve.envs.adapters.pettingzoo_wrapper.RelationCommonsPettingZooEnv
        self.env = env
        self.n_agents = len(env.possible_agents)
        self.obs_size = int(env.observation_space(env.possible_agents[0]).shape[0])
        self.action_space_size = int(env.action_space(env.possible_agents[0]).n)
        self._T_max = int(T_max)
        self._seed = None
        self._seed_consumed = False
        self._g_true = -1   # oracle regime id; -1 when env is not in oracle mode

    def _update_g_true(self, info_dict) -> None:
        # per-agent info: every agent sees the same kept dict; g_true present
        # only when the wrapper runs with oracle_mode=True (train-time only).
        kept = info_dict.get("agent_0", {}) if isinstance(info_dict, dict) else {}
        self._g_true = int(kept.get("g_true", -1))

    def oracle_g(self) -> int:
        """Current oracle regime id (train-time supervision), -1 if gated."""
        return self._g_true

    def set_seed(self, seed):
        self._seed = seed
        self._seed_consumed = False

    def get_max_episode_steps(self) -> int:
        return self._T_max

    def _obs_to_array(self, obs_dict) -> np.ndarray:
        arr = np.stack(
            [obs_dict[f"agent_{i}"] for i in range(self.n_agents)]
        ).astype(np.float32)
        # MAZero expects (n_agents, obs_size, 1, 1) image-format vectors.
        return arr[:, :, None, None]

    def reset(self, **kwargs) -> np.ndarray:
        seed = None
        if self._seed is not None and not self._seed_consumed:
            seed = self._seed
            self._seed_consumed = True
        obs_dict, info = self.env.reset(seed=seed)
        self._update_g_true(info)
        return self._obs_to_array(obs_dict)

    def step(self, action: List[int]):
        action_dict = {f"agent_{i}": int(a) for i, a in enumerate(action)}
        obs_dict, reward_dict, term, trunc, info = self.env.step(action_dict)
        rewards = np.array(
            [reward_dict[f"agent_{i}"] for i in range(self.n_agents)],
            dtype=np.float32,
        )
        # A1 mixed-game contract: per-agent relational reward vector (N,).
        # (The all-cooperative special case has identical entries, in which
        # case the vectorized pipeline reduces to the original team-mean.)
        done = bool(any(term.values()) or any(trunc.values()))
        info = dict(info) if isinstance(info, dict) else {}
        self._update_g_true(info)
        return self._obs_to_array(obs_dict), rewards, done, info

    def close(self):
        self.env.close()
