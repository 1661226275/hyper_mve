"""UnifiedLogger — the single TensorBoard funnel for ALL experimental scalars.

Realignment lock (2026-07-24): the canonical x-axis for every scalar of every
algorithm is **env_steps** (cumulative environment transitions). Different
algorithms take a different number of gradient updates per env step, so only
the env-step axis aligns them on the shared 1M-env-step budget. The
train-native fork maps its gradient-step x to env steps via a self-calibrated
env→train ratio (from ``train/transitions_collected``); env-native callers
(the ported baselines + their eval probes) key on env steps directly. Every
emit also records ``progress/train_steps`` (and ``progress/env_steps``) at the
same x, so either axis is recoverable post-hoc (this replaces the two
contradictory converters that used to live in ``plot_eval_result.py`` and
``make_baseline_comparison.py``).

Tag schema (enforced by convention, not assertion):

* ``train/*``    — training losses / lr / internals
* ``eval/*``     — periodic + final evaluation returns, zero-shot split
* ``fidelity/*`` — world-model reward-fidelity scalars (realignment phase 7)
* ``progress/*`` — ``env_steps`` (written automatically), ``wall_s``
* ``meta/*``     — run identity text

The class **duck-types** ``SummaryWriter.add_scalar(tag, value, global_step)``
(+ ``add_text``/``flush``/``close``) so writer-consuming code — in particular
the mazero_mixed fork's ``core/log.py:_log`` — can take a UnifiedLogger in
place of a raw ``SummaryWriter`` without modification. ``native_step_unit``
declares how such verbatim ``global_step`` values are interpreted:
``"train"`` (the fork, already gradient steps) or ``"env"`` (the MAPPO /
MAMBA adapters and their eval probes).

One logger instance per training run.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Union


class UnifiedLogger:
    """Single-funnel TensorBoard logger with a canonical train_steps x-axis."""

    def __init__(
        self,
        tb_dir: Union[str, Path],
        *,
        algo: str,
        env_id: str = "",
        seed: int = 0,
        env_steps_per_train_step: Optional[float] = None,
        native_step_unit: str = "train",
        flush_secs: int = 30,
    ) -> None:
        from torch.utils.tensorboard import SummaryWriter

        if native_step_unit not in ("train", "env"):
            raise ValueError(
                f"native_step_unit must be 'train' or 'env', got {native_step_unit!r}"
            )
        self.algo = str(algo)
        self.env_id = str(env_id)
        self.seed = int(seed)
        self._native = native_step_unit
        self._ratio: Optional[float] = (
            float(env_steps_per_train_step) if env_steps_per_train_step else None
        )
        self._train_steps = 0          # monotonic high-water marks
        self._env_steps = 0
        self._t0 = time.time()
        self._writer = SummaryWriter(str(tb_dir), flush_secs=flush_secs)
        self._writer.add_text(
            "meta/identity",
            f"algo={self.algo} env={self.env_id} seed={self.seed} "
            f"native_step_unit={self._native}",
        )

    # ------------------------------------------------------------- counters
    @property
    def train_steps(self) -> int:
        return self._train_steps

    @property
    def env_steps(self) -> int:
        return self._env_steps

    def declare_ratio(self, env_steps_per_train_step: float) -> None:
        """Declare (or refresh) the env-steps-per-train-step conversion."""
        r = float(env_steps_per_train_step)
        if r <= 0:
            raise ValueError(f"ratio must be > 0, got {r}")
        self._ratio = r

    def advance(self, *, train_steps: int = 0, env_steps: int = 0) -> None:
        """INCREMENT the counters (e.g. +1 per optimizer round)."""
        self._train_steps += int(train_steps)
        self._env_steps += int(env_steps)

    def set_progress(self, *, train_steps: Optional[int] = None,
                     env_steps: Optional[int] = None) -> None:
        """Set cumulative counters (monotonic — lower values are ignored)."""
        if train_steps is not None:
            self._train_steps = max(self._train_steps, int(train_steps))
        if env_steps is not None:
            self._env_steps = max(self._env_steps, int(env_steps))

    def to_train_step(self, env_step: int) -> int:
        """Map an env-step x to the canonical train-step x.

        Uses the declared/calibrated ratio when available; otherwise falls
        back to the current train-step high-water mark (a piecewise-constant
        map that is exact whenever emits happen right after update rounds).
        """
        if self._ratio:
            return int(math.ceil(int(env_step) / self._ratio))
        return self._train_steps

    # -------------------------------------------------------------- scalars
    def log_scalar(self, tag: str, value: float, *,
                   train_step: Optional[int] = None,
                   env_step: Optional[int] = None) -> None:
        """Write one scalar at the canonical x = ENV steps. Exactly ONE step kwarg.

        The canonical x-axis is env_steps (see the module docstring): only that
        axis aligns algorithms with different env-steps-per-gradient-step ratios
        on the shared 1M-env-step budget. ``progress/train_steps`` is emitted as
        the recoverable companion.
        """
        if (train_step is None) == (env_step is None):
            raise ValueError("pass exactly one of train_step= / env_step=")
        if env_step is not None:
            # env-native caller (baselines + PeriodicEvalProbe): x IS env steps.
            self.set_progress(env_steps=int(env_step))
            x = int(env_step)
            if self._ratio:
                self.set_progress(
                    train_steps=int(math.ceil(int(env_step) / self._ratio)))
        else:
            # train-native caller (the fork): map its gradient-step x to env
            # steps via the self-calibrated ratio (train/transitions_collected).
            # round(train_step * ratio) is monotonic in train_step, so emits
            # within one collect-burst don't collapse onto a single x.
            self.set_progress(train_steps=int(train_step))
            if self._ratio:
                x = int(round(int(train_step) * self._ratio))
            else:
                x = int(self._env_steps)  # until the first transitions_collected
            self.set_progress(env_steps=x)
        self._writer.add_scalar(tag, float(value), x)
        self._writer.add_scalar("progress/env_steps", float(self._env_steps), x)
        self._writer.add_scalar("progress/train_steps", float(self._train_steps), x)

    def log_scalars(self, scalars: Mapping[str, float], *,
                    train_step: Optional[int] = None,
                    env_step: Optional[int] = None) -> None:
        for tag, value in scalars.items():
            self.log_scalar(tag, value, train_step=train_step, env_step=env_step)

    def log_eval_report(self, report: Any, *,
                        train_step: Optional[int] = None,
                        env_step: Optional[int] = None) -> None:
        """Emit the eval/* scalar family from a rel-v1 EvalReport."""
        if train_step is None and env_step is None:
            # canonical axis is env_steps: land the final point at cumulative
            # env steps (~budget) rather than the train-step high-water mark.
            env_step = self._env_steps
        scalars = {
            "eval/return_mean": report.return_mean,
            "eval/return_seen": report.return_zero_shot_seen,
            "eval/return_unseen": report.return_zero_shot_unseen,
            "eval/zero_shot_gap": report.return_zero_shot_gap,
        }
        for g, ret in report.return_per_regime.items():
            scalars[f"eval/return_regime_{int(g)}"] = ret
        if report.regime_accuracy is not None:
            scalars["eval/regime_accuracy"] = report.regime_accuracy
        self.log_scalars(scalars, train_step=train_step, env_step=env_step)
        self._writer.add_scalar(
            "progress/wall_s", time.time() - self._t0,
            self._env_steps,
        )

    # ------------------------------------- SummaryWriter duck-type surface
    def add_scalar(self, tag: str, value: Any, global_step: Optional[int] = None,
                   *args: Any, **kwargs: Any) -> None:
        """Verbatim writer calls from ported code (fork `_log`, eval probes).

        ``global_step`` is interpreted per ``native_step_unit``. The fork's
        ``train/transitions_collected`` scalar (env steps at a train-step x)
        continuously self-calibrates the env→train ratio.
        """
        step = int(global_step) if global_step is not None else 0
        if self._native == "env":
            self.log_scalar(tag, float(value), env_step=step)
            return
        if tag == "train/transitions_collected" and step > 0:
            self.set_progress(env_steps=int(value))
            self._ratio = max(1.0, float(value) / float(step))
        self.log_scalar(tag, float(value), train_step=step)

    def add_text(self, tag: str, text: str,
                 global_step: Optional[int] = None) -> None:
        self._writer.add_text(tag, text, global_step)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.flush()
        self._writer.close()
