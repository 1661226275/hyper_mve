"""HAPPO contract smoke (phase-4 gate) — tiny train → evaluate → ckpt roundtrip.

Exercises the vendored-HARL drive path end to end on rel_duo: 2K env steps,
schema-complete rel-v1 EvalReport with per-regime population, checkpoint
save/load into a fresh runner, and forbidden-info discipline (the relation
env inside HARL runs oracle-free; eval rollouts assert no leak per step).
"""
from __future__ import annotations

import numpy as np
import pytest


def test_happo_tiny_train_eval_ckpt_roundtrip(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("yaml")

    from hyper_mve.comparison import create_runner
    from hyper_mve.utils.configs import V4Config
    from hyper_mve.envs.adapters.pettingzoo_wrapper import RelationCommonsPettingZooEnv
    from hyper_mve.utils.eval.eval_report import EvalReport

    cfg = V4Config.from_preset("rel_duo")
    runner = create_runner(cfg, "happo")
    env_fn = lambda: RelationCommonsPettingZooEnv(
        cfg.env, oracle_mode=False, eval_info_mode=False,
    )

    runner.train(
        cfg, env_fn, total_env_steps=2_000, lr=5e-4, seed=0,
        tensorboard_dir=str(tmp_path / "tb"),
    )
    assert runner.param_count() > 0

    regime_grid = (0, 1)
    report = runner.evaluate(env_fn, regime_grid=regime_grid, episodes=2)
    assert isinstance(report, EvalReport)
    assert report.schema_version == "rel-v3"
    assert report.variant == "happo"
    assert set(report.return_per_regime) == set(regime_grid)
    assert set(report.episodes_per_regime) == set(regime_grid)
    for g in regime_grid:
        assert report.episodes_per_regime[g] == 2
    assert report.info_gating_strict is True
    assert report.env_steps_evaluated > 0

    # ckpt roundtrip into a FRESH runner (standalone load path)
    ckpt = tmp_path / "happo_ckpt.pt"
    runner.save_checkpoint(ckpt)
    assert ckpt.exists()

    fresh = create_runner(cfg, "happo")
    fresh.load_checkpoint(ckpt)
    report2 = fresh.evaluate(env_fn, regime_grid=(0,), episodes=1)
    assert isinstance(report2, EvalReport)
    assert report2.episodes_per_regime[0] == 1

    # loaded actors reproduce the trained actors' deterministic actions
    import torch

    obs = np.zeros((1, 39), dtype=np.float32)
    rnn = np.zeros((1, runner._runner.recurrent_n,
                    runner._runner.rnn_hidden_size), dtype=np.float32)
    masks = np.ones((1, 1), dtype=np.float32)
    with torch.no_grad():
        a1, _ = runner._runner.actor[0].act(obs, rnn, masks, None,
                                            deterministic=True)
        a2, _ = fresh._runner.actor[0].act(obs, rnn, masks, None,
                                           deterministic=True)
    assert int(a1.reshape(-1)[0]) == int(a2.reshape(-1)[0])
