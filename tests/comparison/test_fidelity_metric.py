"""World-model fidelity metric gates (phase-7, fidelity-v1).

* Exact-MAE: a synthetic runner whose ``predict_rewards`` returns
  ``true + offset`` yields ``reward_mae == |offset|`` exactly.
* Model-free / no-hook runners → ``None`` (N/A, no artifact).
* NaN rows (MAMBA's first-transition convention) are masked and reported as
  reduced coverage, not counted in the MAE.
* Probe determinism: identical probe datasets across two ``env_fn`` builds.
"""
from __future__ import annotations

import numpy as np
import pytest

from hyper_mve.utils.eval.fidelity import (
    collect_probe_set,
    compute_reward_fidelity,
    compute_fidelity_report,
)


class _Env:
    """Tiny deterministic 2-agent, 3-action env for probe collection."""

    possible_agents = ["agent_0", "agent_1"]

    class _Space:
        n = 3

        def sample(self):  # unused
            return 0

    def __init__(self):
        self._t = 0
        self._g = 0

    def action_space(self, _agent):
        return self._Space()

    def reset(self, seed=None, options=None):
        self._t = 0
        self._g = int((options or {}).get("g", 0))
        obs = {a: np.full(4, self._g, dtype=np.float32)
               for a in self.possible_agents}
        return obs, {a: {} for a in self.possible_agents}

    def step(self, action_dict):
        self._t += 1
        obs = {a: np.full(4, self._g + self._t, dtype=np.float32)
               for a in self.possible_agents}
        rew = {a: float(self._g) + 0.5 * int(action_dict[a])
               for a in self.possible_agents}
        done = self._t >= 5
        term = {a: done for a in self.possible_agents}
        trunc = {a: False for a in self.possible_agents}
        return obs, rew, term, trunc, {a: {} for a in self.possible_agents}

    def close(self):
        pass


def _env_fn():
    return _Env()


class _SyntheticRunner:
    name = "synthetic"

    def __init__(self, offset):
        self._offset = float(offset)

    def predict_rewards(self, episode):
        return np.asarray(episode["rewards"], dtype=np.float64) + self._offset


class _NaNFirstRowRunner:
    name = "nanfirst"

    def predict_rewards(self, episode):
        pred = np.asarray(episode["rewards"], dtype=np.float64).copy()
        pred[0] = np.nan
        return pred


class _ModelFreeRunner:
    name = "modelfree"
    # no predict_rewards attribute at all


class _NoneRunner:
    name = "none"

    def predict_rewards(self, episode):
        return None


def test_exact_mae_on_synthetic_offset():
    probe = collect_probe_set(_env_fn, (0, 1, 2), episodes=2, seed=7)
    for offset in (0.0, 0.25, 1.5):
        rep = compute_reward_fidelity(_SyntheticRunner(offset), probe)
        assert rep["schema_version"] == "fidelity-v1"
        assert rep["reward_mae"] == pytest.approx(abs(offset), abs=1e-9)
        assert rep["transitions_scored"] == rep["transitions_total"]
        # per-agent MAE also exact
        for m in rep["reward_mae_per_agent"]:
            assert m == pytest.approx(abs(offset), abs=1e-9)


def test_zero_offset_gives_zero_rmae():
    probe = collect_probe_set(_env_fn, (1, 2), episodes=1, seed=3)
    rep = compute_reward_fidelity(_SyntheticRunner(0.0), probe)
    for g, v in rep["reward_rmae_per_regime"].items():
        assert v == pytest.approx(0.0, abs=1e-9)


def test_nan_first_row_masked_and_coverage_reported():
    probe = collect_probe_set(_env_fn, (0,), episodes=1, seed=1)
    rep = compute_reward_fidelity(_NaNFirstRowRunner(), probe)
    # only row 0 of a 5-step episode is NaN → 4 of 5 scored, MAE exact 0
    assert rep["reward_mae"] == pytest.approx(0.0, abs=1e-9)
    assert rep["transitions_scored"] == rep["transitions_total"] - 1


def test_model_free_and_none_are_na():
    probe = collect_probe_set(_env_fn, (0,), episodes=1, seed=1)
    assert compute_reward_fidelity(_ModelFreeRunner(), probe) is None
    assert compute_reward_fidelity(_NoneRunner(), probe) is None
    # compute_fidelity_report short-circuits before probing for no-hook
    assert compute_fidelity_report(_ModelFreeRunner(), _env_fn, (0,)) is None


def test_probe_set_is_deterministic():
    a = collect_probe_set(_env_fn, (0, 1), episodes=2, seed=99)
    b = collect_probe_set(_env_fn, (0, 1), episodes=2, seed=99)
    assert len(a) == len(b) == 4
    for ea, eb in zip(a, b):
        assert ea["g"] == eb["g"]
        assert np.array_equal(ea["actions"], eb["actions"])
        assert np.array_equal(ea["rewards"], eb["rewards"])
        assert np.array_equal(ea["obs"], eb["obs"])
