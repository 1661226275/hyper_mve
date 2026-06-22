"""external_ma_muzero_gh MCTS-budget knob (Test-1 feasibility fix).

The runner's per-agent MCTS runs at every agent-step in BOTH train and eval and
its cost is linear in ``num_simulations`` — pkg-07 spec 06 §3.5 notes eval is
~50x slower than QMIX at the same value (observed: qmix 62.8s vs ma_muzero_gh
3784s on the same smoke). ``HYPER_MVE_MAMZ_NUM_SIMULATIONS`` lets the launcher
dial it per run; this pins the resolver + that the value reaches the MCTS config.
"""
import importlib

import pytest

mod = importlib.import_module("hyper_mve.baselines.external.ma_muzero_gh")


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv(mod._NUM_SIMULATIONS_ENV_VAR, raising=False)
    yield


def test_default_is_lowered_for_feasibility():
    # Must stay well below the spec's 25/50 so the 300K sweep is wall-clock
    # feasible (the §3.10 "weak" baseline tolerates a shallow tree).
    assert mod._DEFAULT_NUM_SIMULATIONS <= 10
    assert mod._resolve_num_simulations() == mod._DEFAULT_NUM_SIMULATIONS


def test_env_override_applied(monkeypatch):
    monkeypatch.setenv(mod._NUM_SIMULATIONS_ENV_VAR, "4")
    assert mod._resolve_num_simulations() == 4
    # ...and it flows into the MCTS config the learners consume.
    assert mod._build_mcts_config(n_actions=6).num_simulations == 4


def test_env_override_clamped_to_at_least_one(monkeypatch):
    monkeypatch.setenv(mod._NUM_SIMULATIONS_ENV_VAR, "0")
    assert mod._resolve_num_simulations() == 1


@pytest.mark.parametrize("bad", ["", "abc", "3.5", "-"])
def test_malformed_override_falls_back_to_default(monkeypatch, bad):
    monkeypatch.setenv(mod._NUM_SIMULATIONS_ENV_VAR, bad)
    assert mod._resolve_num_simulations() == mod._DEFAULT_NUM_SIMULATIONS
