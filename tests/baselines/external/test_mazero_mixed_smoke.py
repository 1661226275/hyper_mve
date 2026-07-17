"""mazero_mixed runner contract smoke (mirrors test_mamba_smoke.py tiny tier).

Fast gate: a tiny train (enough transitions to trigger real search+update
cycles through the fork's train_sync_serial), per-regime evaluate filling the
rel-v1 EvalReport, and a checkpoint save/load roundtrip incl. standalone load.
"""
from __future__ import annotations

import pytest


def test_mazero_mixed_tiny_train_eval_ckpt_roundtrip(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("pettingzoo")

    from hyper_mve.baselines import create_baseline
    from hyper_mve.configs import V4Config
    from hyper_mve.envs.adapters.pettingzoo_wrapper import RelationCommonsPettingZooEnv
    from hyper_mve.eval.eval_report import EvalReport

    cfg = V4Config.from_preset("rel_duo")
    runner = create_baseline(cfg, "mazero_mixed")
    env_fn = lambda: RelationCommonsPettingZooEnv(
        cfg.env, oracle_mode=False, eval_info_mode=False,
    )

    # ~6 episodes (600 env steps) — crosses start_transition so the serial
    # loop runs real search + reanalyze + update cycles.
    runner.train(cfg, env_fn, total_env_steps=600, lr=0.02, seed=0)
    assert runner.param_count() > 0

    report = runner.evaluate(env_fn, regime_grid=(0, 1), episodes=1)
    assert isinstance(report, EvalReport)
    assert report.variant == "mazero_mixed"
    assert set(report.return_per_regime) == {0, 1}
    assert report.episodes_total == 2

    ckpt = tmp_path / "mazero_mixed_smoke.pt"
    runner.save_checkpoint(ckpt)
    fresh = create_baseline(cfg, "mazero_mixed")
    fresh.load_checkpoint(ckpt)
    assert fresh.param_count() == runner.param_count()
    report2 = fresh.evaluate(env_fn, regime_grid=(0,), episodes=1)
    assert isinstance(report2, EvalReport)
