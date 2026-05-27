"""End-to-end env integration (Pkg-02 spec 08 §5.2)."""
from __future__ import annotations

import numpy as np
import pytest

from hyper_mve.configs import V4Config
from hyper_mve.envs.resource_commons import ResourceCommonsEnv


@pytest.fixture
def medium_env():
    cfg = V4Config.from_preset("medium")
    return ResourceCommonsEnv(cfg.env, seed=42), cfg


def test_one_episode_no_nan(medium_env):
    env, cfg = medium_env
    obs, info = env.reset()
    rng = np.random.default_rng(0)
    for _ in range(cfg.env.T_max):
        action = rng.integers(0, 6, size=cfg.env.N, dtype=np.int64)
        obs, reward, done, _, info = env.step(action)
        assert not np.isnan(obs).any()
        assert not np.isnan(reward).any()
        assert not np.isinf(obs).any()
        assert not np.isinf(reward).any()
        if done:
            break


def test_many_episodes_no_crash(medium_env):
    env, cfg = medium_env
    rng = np.random.default_rng(0)
    for ep in range(20):
        env.reset(seed=ep)
        for _ in range(20):
            action = rng.integers(0, 6, size=cfg.env.N, dtype=np.int64)
            obs, reward, done, _, _ = env.step(action)
            assert obs.shape == (cfg.env.N, 99)
            assert reward.shape == (cfg.env.N,)
            if done:
                break


def test_types_constant_across_episode(medium_env):
    env, cfg = medium_env
    _, info = env.reset()
    initial_types = info["types"].copy()
    rng = np.random.default_rng(0)
    for _ in range(50):
        action = rng.integers(0, 6, size=cfg.env.N, dtype=np.int64)
        _, _, done, _, info = env.step(action)
        assert np.array_equal(info["types"], initial_types)
        if done:
            break


def test_static_c_constant_across_episode():
    cfg = V4Config.from_preset("medium")     # static c_mode
    env = ResourceCommonsEnv(cfg.env, seed=42)
    _, info = env.reset()
    c_initial = info["c_true"]
    rng = np.random.default_rng(0)
    for _ in range(50):
        action = rng.integers(0, 6, size=cfg.env.N, dtype=np.int64)
        _, _, done, _, info = env.step(action)
        assert info["c_true"] == pytest.approx(c_initial)
        if done:
            break


def test_resource_stock_bounded():
    cfg = V4Config.from_preset("medium")
    env = ResourceCommonsEnv(cfg.env, seed=42)
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(100):
        action = rng.integers(0, 6, size=cfg.env.N, dtype=np.int64)
        _, _, done, _, info = env.step(action)
        stocks = info["resource_state"][:, 2]
        assert stocks.min() >= 0.0
        assert stocks.max() <= float(cfg.env.Q_max) + 1e-5
        if done:
            break


def test_step_before_reset_raises():
    env = ResourceCommonsEnv(V4Config.from_preset("medium").env, seed=42)
    with pytest.raises(RuntimeError):
        env.step(np.zeros(4, dtype=np.int64))


def test_action_shape_mismatch_raises():
    env = ResourceCommonsEnv(V4Config.from_preset("medium").env, seed=42)
    env.reset()
    with pytest.raises(AssertionError):
        env.step(np.zeros(3, dtype=np.int64))    # wrong N


def test_cfg_must_be_envconfig():
    with pytest.raises(TypeError):
        ResourceCommonsEnv({"N": 4}, seed=42)    # plain dict not allowed
