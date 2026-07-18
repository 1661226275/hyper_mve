"""MBOM + MBOM-oracle contract smoke (phase-5 gate).

Trains BOTH rows sequentially in ONE process (2 epochs each — proves no
multiprocessing/set_start_method traps remain on the single-process drive
path), then evaluates and roundtrips a checkpoint.
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.parametrize("variant", ["mbom", "mbom_oracle"])
def test_mbom_tiny_train_eval_ckpt_roundtrip(variant, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("scipy")

    from hyper_mve.comparison import create_runner
    from hyper_mve.utils.configs import V4Config
    from hyper_mve.envs.adapters.pettingzoo_wrapper import RelationCommonsPettingZooEnv
    from hyper_mve.utils.eval.eval_report import EvalReport

    cfg = V4Config.from_preset("rel_duo")
    runner = create_runner(cfg, variant)
    env_fn = lambda: RelationCommonsPettingZooEnv(
        cfg.env, oracle_mode=False, eval_info_mode=False,
    )

    # 2 epochs × 4 episodes × T_max=100 = 800 env steps
    runner.train(cfg, env_fn, total_env_steps=800, lr=1e-3, seed=0,
                 tensorboard_dir=str(tmp_path / variant))
    assert runner.param_count() > 0

    report = runner.evaluate(env_fn, regime_grid=(0, 1), episodes=1)
    assert isinstance(report, EvalReport)
    assert report.schema_version == "rel-v1"
    assert report.variant == variant
    assert set(report.return_per_regime) == {0, 1}
    assert report.env_steps_evaluated > 0

    ckpt = tmp_path / f"{variant}.pt"
    runner.save_checkpoint(ckpt)
    fresh = create_runner(cfg, variant)
    fresh.load_checkpoint(ckpt)
    # loaded joint policy reproduces the trained one's greedy actions
    obs = np.zeros(39, dtype=np.float32)
    a_trained = int(np.asarray(
        runner._agents[0].choose_action(obs, greedy=True)[0]).reshape(-1)[0])
    a_loaded = int(np.asarray(
        fresh._agents[0].choose_action(obs, greedy=True)[0]).reshape(-1)[0])
    assert a_trained == a_loaded
