"""RelationCommonsPettingZooEnv — N-parametric PettingZoo ParallelEnv wrapper (v5).

Pkg-07 spec 04 (v5 form) — load-bearing for every external baseline runner.
The two constructor flags (``oracle_mode`` + ``eval_info_mode``) gate two
orthogonal information surfaces (spec 04 §7):

  * ``oracle_mode=True`` exposes ``info["g_true"]`` and ``info["rows"]``
    (ground-truth regime id + all agents' relationship rows). Permitted
    *only* on the internal oracle-supervision path; **no external runner
    ever flips this flag** — externals must infer the hidden regime from
    behavior, exactly like hyper's BeliefNet.

  * ``eval_info_mode=True`` exposes ``info["resource_state"]`` (eval-only
    diagnostic scaffolding). Permitted *only* on the unified evaluator path;
    **never** at training time.

Schema markers (``_info_schema_version``, ``_oracle_fields``,
``_eval_only_fields``) are unconditionally stripped — they leak the *names* of
privileged fields even when the values are gated.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from gym.spaces import Box, Discrete
from pettingzoo.utils.env import ParallelEnv

from hyper_mve.configs.env_config import EnvConfig
from hyper_mve.envs.relation_commons import RelationCommonsEnv
from hyper_mve.schemas import RelationObservationLayout


_VALID_N: tuple[int, ...] = (2, 4, 8)


class RelationCommonsPettingZooEnv(ParallelEnv):
    """N-parametric PettingZoo ``ParallelEnv`` wrapper around ``RelationCommonsEnv``.

    Agent ids are the literal strings ``"agent_0"`` .. ``f"agent_{N-1}"``,
    where N is read from ``env_cfg.N`` (locked preset domain {2, 4, 8}).
    The action space is ``Discrete(6)`` per agent (NOOP/UP/DOWN/LEFT/RIGHT/HARVEST,
    from ``hyper_mve.envs.relation_commons.spaces.make_action_space``).
    """

    metadata = {"render_modes": ["rgb_array"], "name": "relation_commons_v5"}

    def __init__(
        self,
        env_cfg: EnvConfig,
        oracle_mode: bool = False,
        eval_info_mode: bool = False,
    ) -> None:
        # RelationCommonsEnv.__init__ already raises TypeError on a non-EnvConfig
        # input; no extra check needed here.
        self._env = RelationCommonsEnv(env_cfg)
        self._oracle_mode: bool = bool(oracle_mode)
        self._eval_info_mode: bool = bool(eval_info_mode)
        self._N: int = self._env.N
        if self._N not in _VALID_N:
            raise ValueError(
                f"env_cfg.N={self._N} outside locked preset domain "
                f"{set(_VALID_N)} — preset family is Easy(2)/Medium(4)/Hard(8)."
            )
        self.possible_agents: list[str] = [f"agent_{i}" for i in range(self._N)]
        self.agents: list[str] = list(self.possible_agents)

        # Per-agent space sizes (constant; cached for the spec contract).
        obs_dim = RelationObservationLayout.total_dim(self._N, self._env.K)
        self._obs_dim: int = obs_dim
        self._action_space_per_agent: Discrete = Discrete(int(self._env.A))
        self._observation_space_per_agent: Box = Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------ spaces

    def observation_space(self, agent: str) -> Box:
        return self._observation_space_per_agent

    def action_space(self, agent: str) -> Discrete:
        return self._action_space_per_agent

    # ------------------------------------------------------------------ API

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        obs_arr, info = self._env.reset(seed=seed, options=options)
        # obs_arr shape: (N, obs_dim). numpy view (no copy).
        obs_dict = {f"agent_{i}": obs_arr[i] for i in range(self._N)}
        info_dict = self._filter_info(info, per_agent=True)
        self.agents = list(self.possible_agents)
        return obs_dict, info_dict

    def step(
        self,
        action_dict: dict[str, int],
    ) -> tuple[
        dict[str, np.ndarray],   # obs
        dict[str, float],        # reward
        dict[str, bool],         # terminated
        dict[str, bool],         # truncated
        dict[str, dict],         # info
    ]:
        # Strict action-dict contract: every agent in self.agents must be a key.
        missing = [a for a in self.agents if a not in action_dict]
        assert not missing, f"action_dict missing keys: {missing}"
        action_arr = np.array(
            [int(action_dict[f"agent_{i}"]) for i in range(self._N)],
            dtype=np.int64,
        )
        obs_arr, reward_arr, term, trunc, info = self._env.step(action_arr)
        obs_dict = {f"agent_{i}": obs_arr[i] for i in range(self._N)}
        reward_dict = {f"agent_{i}": float(reward_arr[i]) for i in range(self._N)}
        term_dict = {f"agent_{i}": bool(term) for i in range(self._N)}
        trunc_dict = {f"agent_{i}": bool(trunc) for i in range(self._N)}
        info_dict = self._filter_info(info, per_agent=True)
        if term or trunc:
            self.agents = []
        return obs_dict, reward_dict, term_dict, trunc_dict, info_dict

    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        return self._env.render(mode=mode)

    def close(self) -> None:
        self._env.close()

    # ------------------------------------------------------------------ info gate

    def _filter_info(self, info: dict[str, Any], per_agent: bool) -> dict[str, Any]:
        """Two-flag info gate. Reads env's published schema markers — DRY against env evolution.

        Implementation invariants (pkg-07 spec 04 §8):
          - Always strips ``_info_schema_version``, ``_oracle_fields``,
            ``_eval_only_fields`` (they leak structure even if the values they
            name are gated).
          - Drop set is derived from env-published marker tuples, NOT a
            hard-coded list, so a future env change that adds a field to
            ``_oracle_fields`` is automatically gated without an adapter edit.
          - Returned per-agent dict: every agent sees the SAME ``kept`` dict
            (shallow alias — read-only by PettingZoo convention).
        """
        drop: set[str] = set()
        if not self._oracle_mode:
            drop.update(info.get("_oracle_fields", ()))         # ('g_true', 'rows')
        if not self._eval_info_mode:
            drop.update(info.get("_eval_only_fields", ()))      # ('resource_state',)
        drop.update({"_info_schema_version", "_oracle_fields", "_eval_only_fields"})
        kept = {k: v for k, v in info.items() if k not in drop}
        if per_agent:
            return {f"agent_{i}": kept for i in range(self._N)}
        return kept


# Legacy import alias (v4 name) — kept until Stage-6 cleanup; new code should
# import RelationCommonsPettingZooEnv.
ResourceCommonsPettingZooEnv = RelationCommonsPettingZooEnv
