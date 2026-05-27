"""Unit tests for c_t evolution modes (Pkg-02 spec 05 §5.1)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from hyper_mve.envs.resource_commons.context_evolution import (
    OscillateContext,
    RandomWalkContext,
    StaticContext,
    build_context_evolution,
)


@dataclass
class _FakeCfg:
    c_oscillate_period: int = 50
    c_random_walk_sigma: float = 0.05
    c_shock_prob: float = 0.2
    c_shock_range: float = 0.3


# ---------- StaticContext ----------

def test_static_is_constant():
    ce = StaticContext()
    rng = np.random.default_rng(42)
    c0 = ce.initial_c(rng)
    for t in range(100):
        assert ce.step(c0, t, rng) == c0


def test_static_initial_in_unit_interval():
    ce = StaticContext()
    rng = np.random.default_rng(42)
    for _ in range(1000):
        c = ce.initial_c(rng)
        assert 0.0 <= c <= 1.0


# ---------- OscillateContext ----------

def test_oscillate_step_zero_at_origin_and_period():
    ce = OscillateContext(period=50)
    rng = np.random.default_rng(0)
    assert ce.step(0.0, 0, rng) == pytest.approx(0.5)
    assert ce.step(0.0, 25, rng) == pytest.approx(0.5)   # sin(π) = 0
    assert ce.step(0.0, 50, rng) == pytest.approx(0.5)   # one full period


def test_oscillate_amplitude_in_unit_interval():
    ce = OscillateContext(period=50)
    rng = np.random.default_rng(0)
    for t in range(200):
        c = ce.step(0.0, t, rng)
        assert 0.0 <= c <= 1.0


def test_oscillate_rejects_invalid_period():
    with pytest.raises(ValueError):
        OscillateContext(period=0)


# ---------- RandomWalkContext ----------

def test_random_walk_clipped_to_unit_interval():
    ce = RandomWalkContext(sigma=0.05, shock_prob=0.0)
    rng = np.random.default_rng(42)
    c = 0.5
    for t in range(200):
        c = ce.step(c, t, rng)
        assert 0.0 <= c <= 1.0


def test_random_walk_drifts_over_time():
    ce = RandomWalkContext(sigma=0.05, shock_prob=0.0)
    rng = np.random.default_rng(42)
    c = 0.5
    history = [c]
    for t in range(200):
        c = ce.step(c, t, rng)
        history.append(c)
    # 200-step Brownian-style walk should explore more than 0.1 of the unit interval
    assert (max(history) - min(history)) > 0.1


def test_random_walk_shock_bounded():
    ce = RandomWalkContext(sigma=0.0, shock_prob=1.0, shock_range=0.5)
    rng = np.random.default_rng(42)
    c0 = 0.5
    c1 = ce.step(c0, 0, rng)
    # σ=0 ⇒ delta is purely the shock ⇒ |c1 - c0| ≤ shock_range (then clipped)
    assert abs(c1 - c0) <= 0.5 + 1e-6


def test_random_walk_reproducible_under_same_seed():
    ce = RandomWalkContext()
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    c1 = ce.initial_c(rng1)
    c2 = ce.initial_c(rng2)
    assert c1 == c2
    for t in range(100):
        c1 = ce.step(c1, t, rng1)
        c2 = ce.step(c2, t, rng2)
        assert c1 == c2


def test_random_walk_validates_inputs():
    with pytest.raises(ValueError):
        RandomWalkContext(sigma=-0.01)
    with pytest.raises(ValueError):
        RandomWalkContext(shock_prob=1.5)
    with pytest.raises(ValueError):
        RandomWalkContext(shock_range=-0.1)


# ---------- factory ----------

def test_factory_returns_each_mode():
    cfg = _FakeCfg()
    assert isinstance(build_context_evolution("static", cfg), StaticContext)
    assert isinstance(build_context_evolution("oscillate", cfg), OscillateContext)
    assert isinstance(build_context_evolution("random_walk", cfg), RandomWalkContext)


def test_factory_propagates_oscillate_period():
    cfg = _FakeCfg(c_oscillate_period=20)
    ce = build_context_evolution("oscillate", cfg)
    assert isinstance(ce, OscillateContext)
    assert ce.period == 20


def test_factory_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown c_mode"):
        build_context_evolution("piecewise", _FakeCfg())
