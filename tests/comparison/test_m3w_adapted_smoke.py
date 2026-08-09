"""M3W-adapted contract smoke (phase-6 gate).

Tiny train (oracle-ID collection env) → evaluate on the pinned regime grid →
checkpoint roundtrip (fresh runner's planner reproduces greedy actions).
"""
from __future__ import annotations

import numpy as np
import pytest


def test_m3w_adapted_tiny_train_eval_ckpt_roundtrip(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("einops")
    import torch

    from hyper_mve.comparison import create_runner
    from hyper_mve.utils.configs import V4Config
    from hyper_mve.envs.adapters.pettingzoo_wrapper import RelationCommonsPettingZooEnv
    from hyper_mve.utils.eval.eval_report import EvalReport

    cfg = V4Config.from_preset("rel_duo")
    runner = create_runner(cfg, "m3w_adapted")
    assert runner.param_count() == 0  # lazy pre-build
    env_fn = lambda: RelationCommonsPettingZooEnv(
        cfg.env, oracle_mode=False, eval_info_mode=False,
    )

    runner.train(cfg, env_fn, total_env_steps=800, lr=1e-3, seed=0)
    assert runner.param_count() > 0

    report = runner.evaluate(env_fn, regime_grid=(0, 1), episodes=1)
    assert isinstance(report, EvalReport)
    assert report.schema_version == "rel-v3"
    assert report.variant == "m3w_adapted"
    assert set(report.return_per_regime) == {0, 1}
    assert report.env_steps_evaluated > 0
    assert np.isfinite(report.return_mean)

    ckpt = tmp_path / "m3w_adapted.pt"
    runner.save_checkpoint(ckpt)
    fresh = create_runner(cfg, "m3w_adapted")
    fresh.load_checkpoint(ckpt)
    assert fresh.param_count() == runner.param_count()

    # loaded planner reproduces the trained one's greedy actions
    obs = np.zeros((cfg.env.N, 39), dtype=np.float32)
    for r in (runner, fresh):
        r._wm.eval()
        r._sac.eval()
    g1 = torch.Generator(device=runner._device)
    g2 = torch.Generator(device=fresh._device)
    g1.manual_seed(123)
    g2.manual_seed(123)
    a_trained = runner._planner.plan(obs, 0, explore=False, generator=g1)
    a_loaded = fresh._planner.plan(obs, 0, explore=False, generator=g2)
    assert np.array_equal(a_trained, a_loaded)
