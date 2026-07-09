"""RelationCommonsEnv — v5 gym.Env (Pkg-09).

The v5 simplification of ResourceCommons: fixed physics (uniform resource
spawn, constant regen rate α, deterministic moves, homogeneous agents, no
FOV), with the **relationship regime** ``g`` as the only latent identity
state. Cooperation / competition is carried entirely by the relational
reward ``R_i = (u_i + Σ_{j≠i} w_ij·u_j) / (1 + Σ_{j≠i}|w_ij|) − ε·moved``.

Regime dynamics (one kernel, two research points):

* reset: ``g_0 ~ ρ`` restricted to ``cfg.train_regime_ids`` when set;
* per step: ``g_{t+1} = g_t`` w.p. ``1 − p`` else ``κ(·|g_t)``
  (``p = cfg.regime_switch_prob``; ``p = 0`` ⇒ static within episode).

Step ordering contract: the reward for step *t* uses the W in effect
*during* the step; the regime chain advances **after** reward computation,
so the post-step observation already shows each agent its (possibly new)
own row — the "private switch notification".
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import gym
import numpy as np

from hyper_mve.configs.env_config import EnvConfig
from hyper_mve.schemas import get_regime_family, sample_initial_regime, step_regime
from hyper_mve.schemas.relation import compute_relational_rewards

from .dynamics import fair_share_harvest, step_dynamics
from .observations import build_joint_observation
from .render import render_rgb_array
from .spaces import make_action_space, make_observation_space
from .spawn import spawn_uniform_resources
from .state import RelationCommonsState


# Action encoding (same as v4): 0=NOOP, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=HARVEST
_ACTION_DELTAS: dict[int, tuple[int, int]] = {
    1: (0, 1),    # UP
    2: (0, -1),   # DOWN
    3: (-1, 0),   # LEFT
    4: (1, 0),    # RIGHT
}
NOOP = 0
HARVEST = 5


def _action_to_delta(action: int) -> tuple[int, int]:
    """Map a discrete action to a grid displacement; NOOP/HARVEST/unknown → (0, 0)."""
    return _ACTION_DELTAS.get(int(action), (0, 0))


class RelationCommonsEnv(gym.Env):
    """RelationCommons gym.Env (v5, Pkg-09).

    Observation
        ``np.ndarray`` shape ``(N, obs_dim)`` float32, where
        ``obs_dim = RelationObservationLayout.total_dim(N, K)``. Each agent
        sees its **own** relationship row (Self-Info); others' rows / the
        regime id are never observed.

    Action
        ``np.ndarray`` shape ``(N,)`` int, ∈ ``{0, ..., 5}``.

    Reward
        ``np.ndarray`` shape ``(N,)`` float32; relational reward.

    Info dict groups (load-bearing contract, v5):

    - **Public** (model may consume): ``harvests``, ``step_idx``.
    - **Oracle** (trainer supervision only, **must not** enter model
      forward): ``g_true``, ``rows`` — the worker applies the
      row-i-only-for-agent-i discipline when conditioning the model.
    - **Eval only**: ``resource_state``.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, cfg: EnvConfig, seed: Optional[int] = None):
        super().__init__()
        if not isinstance(cfg, EnvConfig):
            raise TypeError(
                f"cfg must be EnvConfig, got {type(cfg).__name__}; "
                "construct via hyper_mve.configs.V4Config.from_preset(...)."
            )
        self.cfg: EnvConfig = cfg

        # Static dimensions
        self.N: int = cfg.N
        self.L: int = cfg.L
        self.K: int = cfg.K
        self.T_max: int = cfg.T_max
        self.A: int = cfg.A

        # Regime family (validates relation_family ↔ N consistency)
        self.family = get_regime_family(cfg)

        # Spaces
        self.observation_space = make_observation_space(self.N, self.K)
        self.action_space = make_action_space(self.N, self.A)

        # RNG
        self._rng: np.random.Generator = np.random.default_rng(seed)

        # State + per-step caches
        self._state: Optional[RelationCommonsState] = None
        self._last_harvests: np.ndarray = np.zeros(self.N, dtype=np.float32)

    # ------------------------------------------------------------------ API

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode.

        Args:
            seed: re-seed the RNG (determines resource layout, agent
                positions, ``g_0`` and the regime chain).
            options: optional dict; supported key (explicit ``in`` check):

                * ``"g"``: int — pins the initial regime, **bypassing**
                  ``cfg.train_regime_ids`` (this is how holdout regimes are
                  evaluated).

        Returns:
            ``(obs, info)``: see class docstring for the ``info`` schema.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if options is not None and "g" in options:
            g_0 = int(options["g"])
            if not (0 <= g_0 < self.family.size):
                raise AssertionError(
                    f"options['g']={g_0} out of [0, {self.family.size})"
                )
        else:
            g_0 = sample_initial_regime(
                self.family,
                self._rng,
                prior=self.cfg.regime_prior,
                allowed_ids=self.cfg.train_regime_ids,
            )

        positions = spawn_uniform_resources(self.K, self.L, self._rng)
        agent_positions = self._rng.integers(
            0, self.L, size=(self.N, 2)
        ).astype(np.int32)
        stocks = np.full(self.K, float(self.cfg.Q_max), dtype=np.float32)

        regime = self.family.regimes[g_0]
        self._state = RelationCommonsState(
            agent_positions=agent_positions,
            cumulative_harvests=np.zeros(self.N, dtype=np.float32),
            steps_since_harvest=np.zeros(self.N, dtype=np.int32),
            last_actions=np.zeros(self.N, dtype=np.int64),
            resource_positions=positions,
            resource_stocks=stocks,
            g=g_0,
            W=regime.w_array(),
            rows=np.stack([regime.row(i) for i in range(self.N)]),
            step_idx=0,
            done=False,
        )
        self._last_harvests = np.zeros(self.N, dtype=np.float32)

        obs = build_joint_observation(self._state, self.L, self.T_max, self.K)
        return obs, self._build_info()

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("step() called before reset()")
        action = np.asarray(action)
        if action.shape != (self.N,):
            raise AssertionError(f"action shape {action.shape} != ({self.N},)")
        if action.dtype not in (np.int32, np.int64):
            action = action.astype(np.int64)

        # 1. Decode action intent (move cost depends on intent, as in v4).
        move_intent = (action >= 1) & (action <= 4)
        harvest_mask = (action == HARVEST)

        # 2. Deterministic moves (v5: no ν reliability roll).
        for i in range(self.N):
            if not move_intent[i]:
                continue
            dx, dy = _action_to_delta(int(action[i]))
            new_pos = self._state.agent_positions[i] + np.array(
                [dx, dy], dtype=np.int32
            )
            np.clip(new_pos, 0, self.L - 1, out=new_pos)
            self._state.agent_positions[i] = new_pos

        # 3. Fair-share harvest (pre-update stocks).
        harvests, harvests_per_resource = fair_share_harvest(
            self._state.resource_positions,
            self._state.resource_stocks,
            self._state.agent_positions,
            harvest_mask,
        )
        self._last_harvests = harvests

        # 4. Harvest counters.
        self._state.steps_since_harvest = np.where(
            harvests > 0.0,
            0,
            self._state.steps_since_harvest + 1,
        ).astype(np.int32)
        self._state.cumulative_harvests = (
            self._state.cumulative_harvests + harvests
        ).astype(np.float32)

        # 5. Constant-rate regen.
        step_dynamics(
            self._state.resource_stocks,
            harvests_per_resource,
            alpha=float(self.cfg.alpha),
            q_max=float(self.cfg.Q_max),
        )

        # 6. Relational reward with the W in effect DURING this step.
        reward = compute_relational_rewards(
            harvests=harvests,
            moved_mask=move_intent,
            W=self._state.W,
            epsilon_move=float(self.cfg.epsilon_move),
        )

        # 7. Advance the regime chain (after reward; p=0 consumes no RNG,
        #    so the point-1 random stream is identical to a chain-free env).
        g_next = step_regime(
            self._state.g,
            self.family,
            self._rng,
            switch_prob=float(self.cfg.regime_switch_prob),
            kernel=self.cfg.regime_kernel,
            allowed_ids=self.cfg.train_regime_ids,
        )
        if g_next != self._state.g:
            regime = self.family.regimes[g_next]
            self._state.g = g_next
            self._state.W = regime.w_array()
            self._state.rows = np.stack(
                [regime.row(i) for i in range(self.N)]
            )

        # 8. Bookkeeping.
        self._state.last_actions = action.astype(np.int64)
        self._state.step_idx += 1
        done = self._state.step_idx >= self.T_max
        self._state.done = bool(done)

        obs = build_joint_observation(self._state, self.L, self.T_max, self.K)
        return obs, reward, bool(done), False, self._build_info()

    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        if mode != "rgb_array":
            raise NotImplementedError(f"Render mode {mode!r} not supported")
        if self._state is None:
            raise RuntimeError("render() called before reset()")
        return render_rgb_array(self._state, self.L)

    def close(self) -> None:
        return None

    # ------------------------------------------------------------------ info

    def _build_info(self) -> dict[str, Any]:
        """Construct the info dict (v5 strict grouping).

        Oracle fields must NOT flow into model forward: the worker consumes
        ``rows`` under the row-i-only-for-agent-i discipline (agent *i* is
        conditioned on ``rows[i]`` only — the same information its own
        observation carries) and ``g_true`` only as a supervision target.
        """
        assert self._state is not None
        return {
            # ====== Public (model may consume) ======
            "harvests": self._last_harvests,
            "step_idx": int(self._state.step_idx),
            # ====== Oracle (trainer supervision only) ======
            "g_true": int(self._state.g),
            "rows": self._state.rows.copy(),
            # ====== Eval only ======
            "resource_state": np.concatenate(
                [
                    self._state.resource_positions.astype(np.float32),
                    self._state.resource_stocks[:, None],
                ],
                axis=1,
            ),
            # ====== Schema markers ======
            "_info_schema_version": "v5.0",
            "_oracle_fields": ("g_true", "rows"),
            "_eval_only_fields": ("resource_state",),
        }


def make_relation_commons(
    cfg: EnvConfig, seed: Optional[int] = None
) -> RelationCommonsEnv:
    """Factory wrapper (stable import surface)."""
    return RelationCommonsEnv(cfg, seed=seed)
