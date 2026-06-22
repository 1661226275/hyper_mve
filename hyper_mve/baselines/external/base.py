"""ExternalBaselineRunner — abstract base for the 6 external baselines.

pkg-07 spec 05 §4 / spec 06 §2.3 lock the four-method protocol:

    - train(cfg, env_fn, *, total_env_steps, lr, seed) -> None
    - evaluate(env_fn, c_grid, episodes) -> EvalReport
    - save_checkpoint(path: Path | str) -> None
    - load_checkpoint(path: Path | str) -> None

Subclasses bring their own trainer/buffer/loss (no MuZeroTrainer reuse) and
consume :class:`hyper_mve.envs.adapters.pettingzoo_wrapper.ResourceCommonsPettingZooEnv`
via the injected ``env_fn`` factory. ``train`` accepts ``**kwargs`` for
forward-compat / adapter-port slack: each concrete runner may consume
extra kwargs (e.g. ``max_train_steps`` aliasing ``total_env_steps`` for the
MAPPO port), but the **three keyword-only params** are the cross-runner
contract that the pkg-08 sweep harness drives.
"""
from __future__ import annotations

import abc
import time
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Union

from hyper_mve.configs import V4Config
from hyper_mve.eval.eval_report import EvalReport


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
        c_grid: tuple[float, ...],
        episodes: int,
    ) -> EvalReport:
        """Evaluate the trained runner; returns the locked-schema EvalReport.

        Default implementation returns a zero-valued, schema-complete report.
        Concrete real-port subclasses (MAPPO/QMIX/MA-MuZero-GH) override
        this with the actual evaluation logic.
        """
        t0 = time.time()
        segments = tuple(self.cfg.eval.c_segments)
        ratios = tuple(self.cfg.eval.bell_curve_type_ratios)
        return EvalReport(
            variant=type(self).__name__,
            seed=0,
            config_hash="0" * 40,
            eval_mode="planner",
            eval_planner_mode=self.cfg.eval.eval_planner_mode,
            c_visible=bool(self.cfg.env.c_visible),
            return_mean=0.0,
            return_sem=0.0,
            return_zero_shot_seen=0.0,
            return_zero_shot_unseen=0.0,
            return_zero_shot_gap=0.0,
            return_per_c=MappingProxyType({c: 0.0 for c in c_grid}),
            return_per_c_sem=MappingProxyType({c: 0.0 for c in c_grid}),
            episodes_per_c=MappingProxyType({c: 0 for c in c_grid}),
            return_per_segment=MappingProxyType({seg: 0.0 for seg in segments}),
            return_per_segment_sem=MappingProxyType({seg: 0.0 for seg in segments}),
            return_per_type_ratio=MappingProxyType({r: 0.0 for r in ratios}),
            return_per_type_ratio_sem=MappingProxyType({r: 0.0 for r in ratios}),
            regret_per_c=MappingProxyType({c: 0.0 for c in c_grid}),
            regret_mean=0.0,
            oracle_ceiling_per_c=MappingProxyType({c: 0.0 for c in c_grid}),
            oracle_ceiling_cache_hit=MappingProxyType({c: False for c in c_grid}),
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


__all__ = ["ExternalBaselineRunner"]
