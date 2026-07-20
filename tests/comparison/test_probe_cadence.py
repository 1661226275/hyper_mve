"""PeriodicEvalProbe cadence gating (2026-07-20).

MAPPO/MAMBA's env-steps-per-train-step ratio isn't fixed (it tracks episode
length, which varies), so a fixed env-step cadence drifts away from an even
train-step spacing over a run. every_train_steps fixes the spacing by gating
on a caller-supplied train_steps count instead. Uses a 1-step fake env so
these run near-instantly and exercise only the gating arithmetic, not real
environment stepping (that's covered by test_mappo_smoke.py /
test_mamba_smoke.py's full train() roundtrip).
"""
from __future__ import annotations

import numpy as np
import pytest

from hyper_mve.comparison._probe import PeriodicEvalProbe
from hyper_mve.utils.configs import V4Config


class _FakeEnv:
    """Two-agent env whose episodes always terminate after exactly one step."""

    def __init__(self, n: int = 2):
        self.n = n

    def reset(self, options=None):
        del options
        obs = {f"agent_{i}": np.zeros(4, dtype=np.float32) for i in range(self.n)}
        return obs, {}

    def step(self, actions):
        del actions
        obs = {f"agent_{i}": np.zeros(4, dtype=np.float32) for i in range(self.n)}
        reward = {f"agent_{i}": 1.0 for i in range(self.n)}
        term = {f"agent_{i}": True for i in range(self.n)}
        trunc = {f"agent_{i}": False for i in range(self.n)}
        return obs, reward, term, trunc, {}

    def close(self):
        pass


class _CountingWriter:
    """Records every write to eval/return_mean -- exactly one per _run() fire,
    regardless of the regime grid size or episodes_per_regime."""

    def __init__(self):
        self.return_mean_calls: list[float] = []

    def add_scalar(self, tag, value, step):
        if tag.endswith("/return_mean"):
            self.return_mean_calls.append(step)

    def flush(self):
        pass


def _fake_act(obs: np.ndarray, t: int) -> np.ndarray:
    del obs, t
    return np.zeros(2, dtype=np.int64)


def _make_probe(writer, **kwargs) -> PeriodicEvalProbe:
    return PeriodicEvalProbe(
        lambda: _FakeEnv(), V4Config.from_preset("rel_duo"), writer,
        act_fn=_fake_act, episodes_per_regime=1, **kwargs,
    )


def test_train_step_cadence_fires_on_schedule():
    writer = _CountingWriter()
    probe = _make_probe(writer, every_train_steps=500)

    probe.maybe_run(env_steps=1, train_steps=1)          # first call always fires
    assert len(writer.return_mean_calls) == 1

    probe.maybe_run(env_steps=999_999, train_steps=250)  # short of the next multiple
    assert len(writer.return_mean_calls) == 1

    probe.maybe_run(env_steps=1, train_steps=500)         # crosses 500
    assert len(writer.return_mean_calls) == 2

    probe.maybe_run(env_steps=1, train_steps=750)         # short of 1000
    assert len(writer.return_mean_calls) == 2

    probe.maybe_run(env_steps=1, train_steps=1000)        # crosses 1000
    assert len(writer.return_mean_calls) == 3


def test_train_step_cadence_ignores_env_steps():
    """A huge env_steps jump alone must not trigger a fire once
    every_train_steps is configured -- only train_steps crossing counts."""
    writer = _CountingWriter()
    probe = _make_probe(writer, every_train_steps=500)

    probe.maybe_run(env_steps=1, train_steps=1)
    assert len(writer.return_mean_calls) == 1

    probe.maybe_run(env_steps=10_000_000, train_steps=2)
    assert len(writer.return_mean_calls) == 1


def test_falls_back_to_env_steps_when_train_steps_unset():
    """Backward compat: no every_train_steps configured -> the original
    env_steps-only gating, independent of any train_steps argument."""
    writer = _CountingWriter()
    probe = _make_probe(writer, every_env_steps=1000)

    probe.maybe_run(env_steps=1)
    assert len(writer.return_mean_calls) == 1

    probe.maybe_run(env_steps=500)
    assert len(writer.return_mean_calls) == 1

    probe.maybe_run(env_steps=1000)
    assert len(writer.return_mean_calls) == 2


def test_falls_back_to_env_steps_when_train_steps_configured_but_not_passed():
    """every_train_steps is set, but a caller that forgets to pass train_steps
    falls back to the env_steps counter for that call. The two counters are
    independent, so env_steps fires on ITS OWN first use too, then gates
    normally (against every_env_steps) afterward."""
    writer = _CountingWriter()
    probe = _make_probe(writer, every_train_steps=500, every_env_steps=1000)

    probe.maybe_run(env_steps=1, train_steps=1)  # fires: train_steps path, 1st use
    assert len(writer.return_mean_calls) == 1

    probe.maybe_run(env_steps=500)               # fires: env_steps path, ITS 1st use
    assert len(writer.return_mean_calls) == 2

    probe.maybe_run(env_steps=999)               # short of the next 1000-multiple
    assert len(writer.return_mean_calls) == 2

    probe.maybe_run(env_steps=1000)              # crosses it
    assert len(writer.return_mean_calls) == 3


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
