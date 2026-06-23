"""Env smoke (migrated from scripts/test_resource_commons.py).

Random-policy rollouts on Easy must complete with no NaN/Inf. The original
1000-episode / perf-budget acceptance run lives in the manual long-run protocol;
this CI smoke is a small no-NaN check.
"""
from __future__ import annotations

import pytest


def test_resource_commons_no_nan_smoke():
    pytest.importorskip("torch")  # V4Config → schemas → torch
    import numpy as np
    from hyper_mve.configs import V4Config
    from hyper_mve.envs.resource_commons import ResourceCommonsEnv

    cfg = V4Config.from_preset("easy")
    env = ResourceCommonsEnv(cfg.env, seed=0)
    rng = np.random.default_rng(0)
    steps = 0
    for ep in range(5):
        obs, info = env.reset(seed=ep)
        assert not np.isnan(obs).any()
        for _ in range(cfg.env.T_max):
            action = rng.integers(0, 6, size=cfg.env.N, dtype=np.int64)
            obs, reward, done, _trunc, info = env.step(action)
            assert not np.isnan(obs).any() and not np.isinf(obs).any()
            assert not np.isnan(reward).any() and not np.isinf(reward).any()
            # the welfare-metric hook reads these public/eval info fields:
            assert "harvests" in info and "resource_state" in info
            steps += 1
            if done:
                break
    assert steps > 0
