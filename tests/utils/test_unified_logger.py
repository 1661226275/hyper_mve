"""UnifiedLogger — canonical train_steps x-axis, dual-axis recovery,
SummaryWriter duck-typing (phase-2 realignment, single logging funnel)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from hyper_mve.utils.unified_logger import UnifiedLogger  # noqa: E402


def _scalars(tb_dir, tag):
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    ea = EventAccumulator(str(tb_dir))
    ea.Reload()
    return [(e.step, e.value) for e in ea.Scalars(tag)]


def test_env_step_maps_to_train_step_via_declared_ratio(tmp_path):
    lg = UnifiedLogger(tmp_path, algo="x", env_id="relation", seed=0,
                       env_steps_per_train_step=16.0)
    lg.log_scalar("train/loss", 1.5, env_step=160)
    lg.close()
    assert _scalars(tmp_path, "train/loss") == [(10, 1.5)]
    # dual-axis recovery: progress/env_steps carries the env x at the same step
    assert _scalars(tmp_path, "progress/env_steps") == [(10, 160.0)]


def test_env_step_piecewise_fallback_without_ratio(tmp_path):
    lg = UnifiedLogger(tmp_path, algo="x")
    lg.advance(train_steps=3)
    lg.log_scalar("eval/return_mean", 2.0, env_step=500)
    lg.close()
    assert _scalars(tmp_path, "eval/return_mean") == [(3, 2.0)]
    assert lg.env_steps == 500


def test_exactly_one_step_kwarg_enforced(tmp_path):
    lg = UnifiedLogger(tmp_path, algo="x")
    with pytest.raises(ValueError):
        lg.log_scalar("train/loss", 1.0)
    with pytest.raises(ValueError):
        lg.log_scalar("train/loss", 1.0, train_step=1, env_step=1)
    lg.close()


def test_add_scalar_ducktype_train_native_and_self_calibration(tmp_path):
    """The fork path: verbatim add_scalar at gradient steps;
    train/transitions_collected self-calibrates the env ratio."""
    lg = UnifiedLogger(tmp_path, algo="mazero_mixed", native_step_unit="train")
    lg.add_scalar("train/loss", 0.7, 7)
    lg.add_scalar("train/transitions_collected", 320.0, 20)  # ratio -> 16
    lg.close()
    assert _scalars(tmp_path, "train/loss") == [(7, pytest.approx(0.7))]
    assert lg.env_steps == 320
    assert lg.to_train_step(160) == 10


def test_add_scalar_ducktype_env_native(tmp_path):
    """The mappo/mamba path: verbatim add_scalar carries env steps."""
    lg = UnifiedLogger(tmp_path, algo="mamba", native_step_unit="env",
                       env_steps_per_train_step=10.0)
    lg.add_scalar("mamba/model_loss", 3.0, 100)
    lg.close()
    assert _scalars(tmp_path, "mamba/model_loss") == [(10, 3.0)]


def test_log_eval_report_scalar_family(tmp_path):
    report = SimpleNamespace(
        return_mean=5.0,
        return_zero_shot_seen=6.0,
        return_zero_shot_unseen=4.0,
        return_zero_shot_gap=2.0,
        return_per_regime={0: 5.5, 2: 4.5},
        regime_accuracy=0.8,
    )
    lg = UnifiedLogger(tmp_path, algo="x")
    lg.advance(train_steps=42)
    lg.log_eval_report(report)
    lg.close()
    assert _scalars(tmp_path, "eval/return_mean") == [(42, 5.0)]
    assert _scalars(tmp_path, "eval/return_regime_2") == [(42, 4.5)]
    assert _scalars(tmp_path, "eval/zero_shot_gap") == [(42, 2.0)]
    assert _scalars(tmp_path, "eval/regime_accuracy") == [(42, pytest.approx(0.8))]
