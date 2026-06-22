"""ResourceCommonsEnv — gym.Env implementation (Pkg-02 spec 07 + 08).

This is the Phase A skeleton: ``__init__``, ``close``, and stubs for
``reset`` / ``step`` / ``render``. The real body of reset/step/render lands
in Phase E once dynamics / rewards / observations / context_evolution exist.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import gym
import numpy as np

from hyper_mve.configs.env_config import EnvConfig
from hyper_mve.schemas import (
    AgentType,
    CapabilityVector,
    ObservationLayout,
    sample_n,
)

from .context_evolution import build_context_evolution
from .dynamics import fair_share_harvest, step_dynamics
from .observations import build_joint_observation
from .render import render_rgb_array
from .rewards import compute_rewards
from .spaces import make_action_space, make_observation_space
from .spawn import spawn_patchy_resources
from .state import ResourceCommonsState


# Action encoding (Pkg-02 spec 08 §2.6): 0=NOOP, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=HARVEST
_ACTION_DELTAS: dict[int, tuple[int, int]] = {
    1: (0, 1),    # UP
    2: (0, -1),   # DOWN
    3: (-1, 0),   # LEFT
    4: (1, 0),    # RIGHT
}
NOOP = 0
HARVEST = 5

# [v4-opt 2026-06c] P0.3: offset of c_t within each agent's flattened observation.
# c_t sits at the start of the `global` block; we resolve the (start, end) at env
# construction time and mask `obs[:, c_slot]` post-build when cfg.c_visible=False.
_GLOBAL_BLOCK_NAME = "global"


def _action_to_delta(action: int) -> tuple[int, int]:
    """Map a discrete action to a grid displacement; NOOP/HARVEST/unknown → (0, 0)."""
    return _ACTION_DELTAS.get(int(action), (0, 0))


class ResourceCommonsEnv(gym.Env):
    """ResourceCommons gym.Env (Ch3 full chapter + Ch4 v4 Oracle signals).

    Observation
        ``np.ndarray`` shape ``(N, obs_dim)`` float32, where
        ``obs_dim = ObservationLayout.total_dim(N, K)``.

    Action
        ``np.ndarray`` shape ``(N,)`` int, ∈ ``{0, ..., 5}``.
        0=NOOP, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=HARVEST.

    Reward
        ``np.ndarray`` shape ``(N,)`` float32; per-agent type-aware reward.

    Info dict groups (Pkg-02 spec 08 §2.3 — *load-bearing contract*):

    - **Public** (model may consume): ``caps``, ``deltas``, ``step_idx``,
      ``harvests``.
    - **Oracle** (v4 — trainer supervision only, **must not** enter model
      forward): ``c_true``, ``types``.
    - **Eval only** (evaluator visualisation): ``hotspot_centers``,
      ``resource_state``.
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

        # Static dimensions (pulled out for hot-path readability).
        self.N: int = cfg.N
        self.L: int = cfg.L
        self.K: int = cfg.K
        self.M: int = cfg.M
        self.T_max: int = cfg.T_max
        self.A: int = cfg.A

        # Spaces (Pkg-02 spec 08 §2.4)
        self.observation_space = make_observation_space(self.N, self.K)
        self.action_space = make_action_space(self.N, self.A)

        # RNG + context evolution (Pkg-02 spec 05)
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._context_evo = build_context_evolution(cfg.c_mode, cfg)

        # [v4-opt 2026-06c] P0.3: precompute c_t column index for post-build masking
        # when cfg.c_visible=False. The `global` block is `[c_t, time_remaining]`, so
        # c_t lives at `global` block start.
        self._c_obs_slot: int = ObservationLayout.block_offset(
            _GLOBAL_BLOCK_NAME, self.N, self.K,
        )[0]

        # State + per-step caches (filled by reset / step)
        self._state: Optional[ResourceCommonsState] = None
        self._last_harvests: np.ndarray = np.zeros(self.N, dtype=np.float32)
        self._last_deltas: np.ndarray = np.zeros(self.N, dtype=np.float32)

    # ------------------------------------------------------------------ API

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode.

        Args:
            seed: re-seed the RNG. Determines hotspot positions, capability
                samples, type / position initialisation, and the c_t time
                series.
            options: optional dict; supported keys (must use ``in options``
                checks — do NOT ``options.get(k) or fallback`` because ``0.0``
                and empty tuples are falsy):

                * ``"c"``: float in ``[0, 1]`` — forces ``c_0``.
                * ``"types"``: ``tuple[AgentType, ...]`` length ``N`` —
                  overrides ``cfg.type_assignment`` for Ablation 3.

        Returns:
            ``(obs, info)``: see class docstring for ``info`` schema.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # --- Resolve options (explicit `in` checks; falsy short-circuit is a bug) ---
        if options is not None and "c" in options:
            c_0 = float(options["c"])
            if not (0.0 <= c_0 <= 1.0):
                raise AssertionError(
                    f"options['c']={c_0} out of [0, 1]"
                )
        else:
            c_0 = float(self._context_evo.initial_c(self._rng))

        if options is not None and "types" in options:
            types_tuple: tuple[AgentType, ...] = tuple(options["types"])
        else:
            types_tuple = self.cfg.type_assignment

        if len(types_tuple) != self.N:
            raise ValueError(
                f"types length {len(types_tuple)} != N {self.N}"
            )

        # --- Sample patchy resources ---
        centers, positions = spawn_patchy_resources(
            K=self.K, M=self.M, L=self.L,
            sigma_patch=float(self.cfg.sigma_patch),
            rng=self._rng,
        )

        # --- Initial agent positions: uniform on the grid ---
        agent_positions = self._rng.integers(0, self.L, size=(self.N, 2)).astype(np.int32)

        # --- Capabilities (Pkg-01 sample_n) ---
        caps: tuple[CapabilityVector, ...] = sample_n(self.N, self._rng)

        # --- All resource cells start full ---
        stocks = np.full(self.K, float(self.cfg.Q_max), dtype=np.float32)

        # --- Build state ---
        self._state = ResourceCommonsState(
            agent_positions=agent_positions,
            cumulative_harvests=np.zeros(self.N, dtype=np.float32),
            steps_since_harvest=np.zeros(self.N, dtype=np.int32),
            last_actions=np.zeros(self.N, dtype=np.int64),
            resource_positions=positions,
            resource_stocks=stocks,
            c_t=c_0,
            c_history=np.zeros(self.T_max + 1, dtype=np.float32),
            agent_caps=caps,
            agent_types=np.array([int(t) for t in types_tuple], dtype=np.int8),
            hotspot_centers=centers,
            step_idx=0,
            done=False,
        )
        self._state.c_history[0] = c_0

        # --- Reset per-step caches ---
        self._last_harvests = np.zeros(self.N, dtype=np.float32)
        self._last_deltas = np.zeros(self.N, dtype=np.float32)

        obs = build_joint_observation(self._state, self.L, self.T_max, self.K, env_cfg=self.cfg)
        # [pkg-08 spec 08 §6.2 A'.1] env_cfg threading is the c_visible
        # consumption path; the mask itself lives inside build_joint_observation.
        info = self._build_info()
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("step() called before reset()")
        action = np.asarray(action)
        if action.shape != (self.N,):
            raise AssertionError(
                f"action shape {action.shape} != ({self.N},)"
            )
        if action.dtype not in (np.int32, np.int64):
            action = action.astype(np.int64)

        # 1. Decode action intent (Pkg-02 spec 03: move cost depends on intent,
        #    not execution success — D7 semantics).
        move_intent = (action >= 1) & (action <= 4)
        harvest_mask = (action == HARVEST)

        # 2. Attempt to move each intended mover with probability cap.nu.
        for i in range(self.N):
            if not move_intent[i]:
                continue
            if self._rng.random() < float(self._state.agent_caps[i].nu):
                dx, dy = _action_to_delta(int(action[i]))
                new_pos = self._state.agent_positions[i] + np.array([dx, dy], dtype=np.int32)
                np.clip(new_pos, 0, self.L - 1, out=new_pos)
                self._state.agent_positions[i] = new_pos
            # else: stay in place (D7)

        # 3. Fair-share harvest (formula 3.2). NOTE: uses pre-update stocks.
        harvests, harvests_per_resource = fair_share_harvest(
            self._state, self._state.agent_positions, harvest_mask,
        )
        self._last_harvests = harvests

        # 4. Update harvest counters.
        self._state.steps_since_harvest = np.where(
            harvests > 0.0,
            0,
            self._state.steps_since_harvest + 1,
        ).astype(np.int32)
        self._state.cumulative_harvests = (
            self._state.cumulative_harvests + harvests
        ).astype(np.float32)

        # 5. Resource regen (formula 3.1). Neighbor factor uses pre-update stocks
        #    via state.resource_stocks; step_dynamics handles the swap.
        step_dynamics(
            self._state,
            harvests_per_resource=harvests_per_resource,
            alpha_min=float(self.cfg.alpha_min),
            alpha_max=float(self.cfg.alpha_max),
            q_max=float(self.cfg.Q_max),
            kappa_f=float(self.cfg.kappa_f),
            theta_f=float(self.cfg.theta_f),
            d_nbr=int(self.cfg.d_nbr),
        )

        # 6. Evolve c_t (Pkg-02 spec 05).
        self._state.c_t = float(self._context_evo.step(
            self._state.c_t, self._state.step_idx, self._rng,
        ))

        # 7. Compute reward using the *intent* mask for the move cost
        #    (matches D7: physical-layer cost is independent of execution).
        reward, deltas = compute_rewards(
            harvests=harvests,
            moved_mask=move_intent,
            agent_types=self._state.agent_types,
            c_t=self._state.c_t,
            kappa=float(self.cfg.kappa),
            lambda_disadv=float(self.cfg.lambda_disadv),
            lambda_adv=float(self.cfg.lambda_adv),
            epsilon_move=float(self.cfg.epsilon_move),
        )
        self._last_deltas = deltas

        # 8. Bookkeeping.
        self._state.last_actions = action.astype(np.int64)
        self._state.step_idx += 1
        self._state.c_history[self._state.step_idx] = self._state.c_t
        done = self._state.step_idx >= self.T_max
        self._state.done = bool(done)

        obs = build_joint_observation(self._state, self.L, self.T_max, self.K, env_cfg=self.cfg)
        # [pkg-08 spec 08 §6.2 A'.1] env_cfg threading is the c_visible
        # consumption path; the mask itself lives inside build_joint_observation.
        info = self._build_info()
        return obs, reward, bool(done), False, info

    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        if mode != "rgb_array":
            raise NotImplementedError(f"Render mode {mode!r} not supported")
        if self._state is None:
            raise RuntimeError("render() called before reset()")
        return render_rgb_array(self._state, self.L)

    def close(self) -> None:
        # No external resources to release.
        return None

    # ------------------------------------------------------------------ info

    def _build_info(self) -> dict[str, Any]:
        """Construct the info dict (Pkg-02 spec 08 §2.3 — strict grouping).

        The grouping is a **trainer-side contract**: Oracle fields must NOT
        flow into model forward. Pkg-05 spec will enforce this when it lands.
        """
        assert self._state is not None
        return {
            # ====== Public (model may consume) ======
            "caps": self._state.agent_caps,
            "deltas": self._last_deltas,
            "step_idx": int(self._state.step_idx),
            "harvests": self._last_harvests,
            # ====== Oracle (v4 — trainer supervision only) ======
            "c_true": float(self._state.c_t),
            "types": self._state.agent_types.copy(),
            # ====== Eval only ======
            "hotspot_centers": self._state.hotspot_centers.copy(),
            "resource_state": np.concatenate(
                [
                    self._state.resource_positions.astype(np.float32),
                    self._state.resource_stocks[:, None],
                ],
                axis=1,
            ),
            # ====== Schema markers (help trainer route fields) ======
            "_info_schema_version": "v4.0",
            "_oracle_fields": ("c_true", "types"),
            "_eval_only_fields": ("hotspot_centers", "resource_state"),
        }


def make_resource_commons(cfg: EnvConfig, seed: Optional[int] = None) -> ResourceCommonsEnv:
    """Factory wrapper (matches the stable import surface promised in spec 08 §6.1)."""
    return ResourceCommonsEnv(cfg, seed=seed)
