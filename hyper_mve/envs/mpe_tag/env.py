"""MPETagRegimeEnv — regime-ified MPE simple_tag (phase-3 realignment).

Wraps ``mpe2.simple_tag_v3`` (pettingzoo ≥1.26 moved MPE out of the core
package) behind the SAME surface every runner already consumes for
RelationCommons (``envs/adapters/pettingzoo_wrapper.py``):

* PettingZoo-parallel dict API with agent ids ``agent_0..agent_3``.
  Index convention (matches ``schemas.relation.build_tag4``): agents 0–2 map
  to the MPE predators ``adversary_0..2``, agent 3 maps to the MPE prey
  (upstream name ``agent_0``).
* ``reset(options={"g": int})`` pins the regime; otherwise it is resampled
  per episode from ``train_regime_ids`` (hidden-task protocol).
* Per-agent reward dict: ``u_i`` = raw per-agent physical simple_tag reward
  ("harvests" in RelationCommons terms), re-weighted as
  ``R = compute_relational_rewards(harvests=u, moved_mask=(a != 0), W=W_g,
  epsilon_move=cfg.epsilon_move)``. With the ``all_solo`` regime (W = I) and
  ``epsilon_move=0`` this reproduces the raw simple_tag rewards exactly —
  the fixed-role calibration configuration.
  Note: tag rewards can be negative (the prey's is); the convex-combination
  bound of the relational formula holds for signed ``u`` as well.
* Two-flag info gate identical to the RelationCommons adapter: ``g_true`` /
  ``rows`` only under ``oracle_mode=True`` (train-time belief supervision);
  schema markers always stripped.

Observation layout (homogeneous across agents, ``float32``):

    [ raw MPE obs, zero-padded to 16 | own W-row (N-1 = 3 dims) ]  → 19

The zero-padding homogenizes the predator (16) / prey (14) native dims; the
appended own-row mirrors the RelationCommons observation contract (an agent
privately knows its OWN relation weights; the full W / regime id stays
hidden) and keeps the mazero_mixed subjective model's obs-tail row
extraction working unchanged. All methods see the same observation.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from gym.spaces import Box, Discrete
from pettingzoo.utils.env import ParallelEnv

from hyper_mve.utils.configs.env_config import EnvConfig
from hyper_mve.utils.schemas.relation import (
    compute_relational_rewards,
    get_regime_family,
    sample_initial_regime,
    step_regime,
)

from hyper_mve.utils.schemas.observation import MPE_TAG_PAD_DIM

# max native obs dim over simple_tag_v3 default agents. Shared with the models
# via schemas.observation so an encoder can size itself without building an env.
_PAD_DIM = MPE_TAG_PAD_DIM
_TAG_N = 4             # 3 predators + 1 prey (simple_tag_v3 defaults)


def _make_underlying(T_max: int):
    try:
        from mpe2 import simple_tag_v3
    except ImportError:  # pragma: no cover — older pettingzoo layouts
        from pettingzoo.mpe import simple_tag_v3
    return simple_tag_v3.parallel_env(
        max_cycles=int(T_max), continuous_actions=False,
    )


class MPETagRegimeEnv(ParallelEnv):
    """Regime-ified simple_tag behind the RelationCommons adapter surface."""

    metadata = {"render_modes": ["rgb_array"], "name": "mpe_tag_regime_v1"}

    def __init__(
        self,
        env_cfg: EnvConfig,
        oracle_mode: bool = False,
        eval_info_mode: bool = False,
        fixed_regime: Optional[int] = None,
    ) -> None:
        if not isinstance(env_cfg, EnvConfig):
            raise TypeError(f"env_cfg must be EnvConfig, got {type(env_cfg)}")
        if env_cfg.env_kind != "mpe_tag":
            raise ValueError(
                f"MPETagRegimeEnv needs env_kind='mpe_tag', got {env_cfg.env_kind!r}"
            )
        if env_cfg.N != _TAG_N:
            raise ValueError(f"mpe_tag requires N={_TAG_N}, got N={env_cfg.N}")
        self._cfg = env_cfg
        self._family = get_regime_family(env_cfg)
        self._oracle_mode = bool(oracle_mode)
        self._eval_info_mode = bool(eval_info_mode)
        self._fixed_regime: Optional[int] = (
            int(fixed_regime) if fixed_regime is not None
            else (int(env_cfg.fixed_regime) if env_cfg.fixed_regime is not None
                  else None)
        )
        if (self._fixed_regime is not None
                and not (0 <= self._fixed_regime < self._family.size)):
            raise ValueError(
                f"fixed_regime={self._fixed_regime} outside |G|={self._family.size}"
            )

        self._N = _TAG_N
        self._env = _make_underlying(env_cfg.T_max)
        self._rng = np.random.default_rng()
        self._g: int = 0
        self._t: int = 0
        # underlying agent order fixed by construction: predators then prey
        self._underlying: list[str] = []

        self.possible_agents: list[str] = [f"agent_{i}" for i in range(self._N)]
        self.agents: list[str] = list(self.possible_agents)

        self._obs_dim = _PAD_DIM + (self._N - 1)
        self._observation_space_per_agent = Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32,
        )
        self._action_space_per_agent = Discrete(int(env_cfg.A))
        if int(env_cfg.A) != 5:
            raise ValueError(f"simple_tag has 5 discrete actions, cfg.A={env_cfg.A}")

    # ------------------------------------------------------------------ spaces
    def observation_space(self, agent: str) -> Box:
        return self._observation_space_per_agent

    def action_space(self, agent: str) -> Discrete:
        return self._action_space_per_agent

    # ------------------------------------------------------------------ helpers
    def _sample_regime(self) -> int:
        if self._fixed_regime is not None:
            return self._fixed_regime
        return sample_initial_regime(
            self._family, self._rng,
            prior=self._cfg.regime_prior,
            allowed_ids=self._cfg.train_regime_ids,
        )

    def _own_row(self, i: int) -> np.ndarray:
        return self._family.regimes[self._g].row(i, dtype=np.float32)

    def _wrap_obs(self, raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for i, u_name in enumerate(self._underlying):
            o = np.asarray(raw[u_name], dtype=np.float32).ravel()
            padded = np.zeros(self._obs_dim, dtype=np.float32)
            padded[: o.shape[0]] = o
            padded[_PAD_DIM:] = self._own_row(i)
            out[f"agent_{i}"] = padded
        return out

    def _info(self, harvests: np.ndarray) -> dict[str, Any]:
        info: dict[str, Any] = {
            # Public
            "harvests": harvests.astype(np.float32),
            "step_idx": self._t,
            # Oracle (train-time belief supervision only)
            "g_true": self._g,
            "rows": self._family.rows_stack()[self._g],
            # Schema markers (always stripped by the gate)
            "_info_schema_version": "mpe-tag-regime-v1",
            "_oracle_fields": ("g_true", "rows"),
            "_eval_only_fields": (),
        }
        return info

    def _filter_info(self, info: dict[str, Any]) -> dict[str, dict]:
        """Two-flag info gate — identical semantics to the RelationCommons
        adapter (drop set derived from env-published marker tuples)."""
        drop: set[str] = set()
        if not self._oracle_mode:
            drop.update(info.get("_oracle_fields", ()))
        if not self._eval_info_mode:
            drop.update(info.get("_eval_only_fields", ()))
        drop.update({"_info_schema_version", "_oracle_fields", "_eval_only_fields"})
        kept = {k: v for k, v in info.items() if k not in drop}
        return {f"agent_{i}": kept for i in range(self._N)}

    # ------------------------------------------------------------------ API
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        if seed is not None:
            self._rng = np.random.default_rng(int(seed) + 7919)
        raw_obs, _ = self._env.reset(seed=seed)
        self._underlying = list(self._env.agents)
        assert len(self._underlying) == self._N, self._underlying
        # options={"g": int} pins the regime (bypasses train_regime_ids),
        # mirroring RelationCommonsEnv reset semantics.
        if options is not None and "g" in options:
            g = int(options["g"])
            if not (0 <= g < self._family.size):
                raise ValueError(f"options['g']={g} outside |G|={self._family.size}")
            self._g = g
        else:
            self._g = self._sample_regime()
        self._t = 0
        self.agents = list(self.possible_agents)
        obs = self._wrap_obs(raw_obs)
        info = self._filter_info(self._info(np.zeros(self._N, dtype=np.float32)))
        return obs, info

    def step(
        self,
        action_dict: dict[str, int],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        missing = [a for a in self.agents if a not in action_dict]
        assert not missing, f"action_dict missing keys: {missing}"
        actions = np.array(
            [int(action_dict[f"agent_{i}"]) for i in range(self._N)],
            dtype=np.int64,
        )
        raw_actions = {
            u_name: int(actions[i]) for i, u_name in enumerate(self._underlying)
        }
        raw_obs, raw_rew, raw_term, raw_trunc, _ = self._env.step(raw_actions)
        self._t += 1

        u = np.array(
            [float(raw_rew[u_name]) for u_name in self._underlying],
            dtype=np.float32,
        )
        W = self._family.regimes[self._g].w_array(dtype=np.float32)
        rewards = compute_relational_rewards(
            harvests=u,
            moved_mask=(actions != 0),          # MPE action 0 = no_action
            W=W,
            epsilon_move=float(self._cfg.epsilon_move),
            # v6 coupling knobs pass through, but mpe_tag presets leave them at
            # the v5 default so W=I still reproduces raw simple_tag rewards.
            coupling=str(self._cfg.reward_coupling),
            reciprocity_lambda=float(self._cfg.reciprocity_lambda),
        )

        # regime chain (p=0 in the phase-3 protocol: never draws from rng)
        self._g = step_regime(
            self._g, self._family, self._rng,
            switch_prob=float(self._cfg.regime_switch_prob),
            kernel=self._cfg.regime_kernel,
            allowed_ids=self._cfg.train_regime_ids
            if self._fixed_regime is None else (self._fixed_regime,),
        )

        obs = self._wrap_obs(raw_obs)
        reward_dict = {f"agent_{i}": float(rewards[i]) for i in range(self._N)}
        term_dict = {
            f"agent_{i}": bool(raw_term[u_name])
            for i, u_name in enumerate(self._underlying)
        }
        trunc_dict = {
            f"agent_{i}": bool(raw_trunc[u_name])
            for i, u_name in enumerate(self._underlying)
        }
        info = self._filter_info(self._info(u))
        if any(term_dict.values()) or any(trunc_dict.values()):
            self.agents = []
        return obs, reward_dict, term_dict, trunc_dict, info

    def render(self):  # pragma: no cover
        return self._env.render()

    def close(self) -> None:
        self._env.close()

    # ------------------------------------------------------------------ oracle
    def oracle_g(self) -> int:
        """Ground-truth regime id (train-time supervision path only)."""
        return self._g
