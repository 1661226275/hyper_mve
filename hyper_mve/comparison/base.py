"""ExternalBaselineRunner — abstract base for the 6 external baselines.

pkg-07 spec 05 §4 / spec 06 §2.3 lock the four-method protocol:

    - train(cfg, env_fn, *, total_env_steps, lr, seed) -> None
    - evaluate(env_fn, regime_grid, episodes) -> EvalReport   (v5: regime ids)
    - save_checkpoint(path: Path | str) -> None
    - load_checkpoint(path: Path | str) -> None

Subclasses bring their own trainer/buffer/loss (no MuZeroTrainer reuse) and
consume :class:`hyper_mve.envs.adapters.pettingzoo_wrapper.RelationCommonsPettingZooEnv`
via the injected ``env_fn`` factory. ``train`` accepts ``**kwargs`` for
forward-compat / adapter-port slack: each concrete runner may consume
extra kwargs (e.g. ``max_train_steps`` aliasing ``total_env_steps`` for the
MAPPO port), but the **three keyword-only params** are the cross-runner
contract that the pkg-08 sweep harness drives.
"""
from __future__ import annotations

import abc
import time

import numpy as np
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Final, Union

from hyper_mve.utils.configs import V4Config
from hyper_mve.utils.eval.eval_report import EvalReport, regime_names_for

# CTDE legitimacy boundary (pkg-07 spec 06 Lock 3 + §6.1) — single source of
# truth for every runner module: oracle / eval-only env-info fields that must
# never reach a runner at train or eval time. Runners import it from here (or
# via the ``hyper_mve.comparison`` package re-export).
_FORBIDDEN_INFO_KEYS: Final[frozenset[str]] = frozenset({
    "g_true", "rows",
    "resource_state",
    "c_true", "types", "hotspot_centers",   # legacy v4 names (defence in depth)
})


class ExternalBaselineRunner(abc.ABC):
    """Abstract base class for external baseline runners (pkg-07 spec 05 §4)."""

    def __init__(self, cfg: V4Config) -> None:
        self.cfg = cfg

    @abc.abstractmethod
    def train(
        self,
        cfg: V4Config,
        env_fn: Callable[[], Any],
        *,
        total_env_steps: int = 0,
        lr: float = 0.0,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        """Train the runner (pkg-07 spec 05 §4.2).

        The three keyword-only params (``total_env_steps`` / ``lr`` /
        ``seed``) are the cross-runner contract the pkg-08 sweep harness
        drives. Concrete runners may consume additional kwargs (forward-
        compat slack), e.g. ``max_train_steps`` as a legacy alias for
        ``total_env_steps``.
        """
        raise NotImplementedError

    def evaluate(
        self,
        env_fn: Callable[[], Any],
        regime_grid: tuple[int, ...],
        episodes: int,
    ) -> EvalReport:
        """Evaluate the trained runner; returns the locked rel-v1 EvalReport.

        Default implementation returns a zero-valued, schema-complete report.
        Concrete real-port subclasses (MAPPO/QMIX/MA-MuZero-GH) override
        this with the actual evaluation logic.
        """
        t0 = time.time()
        grid = tuple(int(g) for g in regime_grid)
        return EvalReport(
            variant=type(self).__name__,
            regime_names=regime_names_for(self.cfg),
            seed=0,
            config_hash="0" * 40,
            eval_mode="planner",
            eval_planner_mode=self.cfg.eval.eval_planner_mode,
            return_mean=0.0,
            return_sem=0.0,
            return_zero_shot_seen=0.0,
            return_zero_shot_unseen=0.0,
            return_zero_shot_gap=0.0,
            return_per_regime=MappingProxyType({g: 0.0 for g in grid}),
            return_per_regime_sem=MappingProxyType({g: 0.0 for g in grid}),
            return_per_regime_per_agent=MappingProxyType(
                {g: (0.0,) * int(self.cfg.env.N) for g in grid}
            ),
            episodes_per_regime=MappingProxyType({g: 0 for g in grid}),
            planner_prior_return_gap=0.0,
            direct_inference_return_mean=0.0,
            planner_full_return_mean=0.0,
            walltime_seconds=float(time.time() - t0),
            env_steps_evaluated=0,
            episodes_total=0,
            info_gating_strict=True,
            set_context_subjective_oracle_leak=False,
        )

    def save_checkpoint(self, path: Union[Path, str]) -> None:
        """Save runner state to ``path`` (pkg-07 spec 05 §4.4 / spec 06 §2.3).

        Default implementation is a no-op so smoke / contract tests don't
        force every runner to declare it. Real ports override.
        """
        del path

    def load_checkpoint(self, path: Union[Path, str]) -> None:
        """Inverse of :meth:`save_checkpoint`. Default is a no-op."""
        del path

    def param_count(self) -> int:
        """Return total trainable parameter count (spec 06 §8.4 disclosure feed).

        Default implementation returns 0 (used by stubs); concrete runners
        override with ``sum(p.numel() for p in self.<modules>.parameters())``.
        """
        return 0


def reward_vector(rew, agents) -> np.ndarray:
    """Per-agent reward for one env step, ordered by ``agents``.

    The external runners each accumulate returns in their own evaluate loop;
    ordering the reward dict through one helper is what keeps "agent i" the same
    agent in every report. Accumulate this vector and derive the scalar episode
    return from ``vec.sum()`` rather than summing the dict separately -- that is
    what makes ``sum(return_per_regime_per_agent[g]) == return_per_regime[g]``
    true by construction rather than by coincidence.
    """
    return np.asarray([float(rew[a]) for a in agents], dtype=np.float64)


def freeze_per_agent(per_regime_per_agent) -> MappingProxyType:
    """Freeze ``{regime: per-agent means}`` for ``EvalReport`` (rel-v3).

    Accepts numpy vectors or plain sequences; emits plain float tuples so the
    report stays JSON-safe.
    """
    return MappingProxyType({
        int(g): tuple(float(x) for x in vec)
        for g, vec in per_regime_per_agent.items()
    })


def split_seen_unseen_regimes(cfg: V4Config, return_per_regime) -> tuple[float, float]:
    """v5 zero-shot split: "seen" = regimes in ``cfg.env.train_regime_ids``
    (all evaluated regimes when None), "unseen" = the rest. Returns
    (seen_mean, unseen_mean); 0.0 for an empty side. Shared by the three
    real external runners + the unified evaluator."""
    train_ids = cfg.env.train_regime_ids
    vals = {int(g): float(v) for g, v in return_per_regime.items()}
    if train_ids is None:
        seen = list(vals.values())
        unseen = []
    else:
        train = {int(i) for i in train_ids}
        seen = [v for g, v in vals.items() if g in train]
        unseen = [v for g, v in vals.items() if g not in train]
    zs_seen = float(sum(seen) / len(seen)) if seen else 0.0
    zs_unseen = float(sum(unseen) / len(unseen)) if unseen else 0.0
    return zs_seen, zs_unseen


__all__ = ["ExternalBaselineRunner", "split_seen_unseen_regimes"]
