"""exp_path must be unique per run, not just per seed.

Regression test for a checkpoint-collision bug: MAZeroMixedRunner.train()
used to derive the fork's exp_path/model_dir from f"seed={seed}" alone, which
is always 0 across a grid's ablation arms/budgets. Two same-seed runs
launched concurrently (routine for a multi-GPU grid) wrote into the identical
model_dir and clobbered each other's mid-training checkpoints. exp_path is
now anchored off the caller-supplied tensorboard_dir (unique per
algo+arm/env/seed), which this test exercises with two distinct values at
seed=0 to confirm they no longer collide.
"""
from __future__ import annotations

import pytest


def _train_tiny(tmp_path, tensorboard_dir, ablation="ref_bc"):
    from hyper_mve.comparison import create_runner
    from hyper_mve.utils.configs import V4Config
    from hyper_mve.envs.adapters.pettingzoo_wrapper import RelationCommonsPettingZooEnv

    cfg = V4Config.from_preset("rel_duo")
    runner = create_runner(cfg, "mazero_mixed")
    env_fn = lambda: RelationCommonsPettingZooEnv(
        cfg.env, oracle_mode=False, eval_info_mode=False,
    )
    runner.train(
        cfg, env_fn, total_env_steps=600, lr=0.02, seed=0,
        ablation=ablation, tensorboard_dir=str(tensorboard_dir),
    )
    return runner


def test_same_seed_different_run_dirs_do_not_collide(tmp_path):
    """Two seed=0 runs (as every arm in a competence grid is) with distinct
    tensorboard_dirs must land in distinct exp_path/model_dir, mirroring
    scripts/train.py's run_dir/"tb" layout for two different ablation arms."""
    pytest.importorskip("torch")
    pytest.importorskip("pettingzoo")

    run_dir_a = tmp_path / "mazero_mixed_ref_bc" / "relation" / "seed0"
    run_dir_b = tmp_path / "mazero_mixed_ref_bc_rw" / "relation" / "seed0"
    (run_dir_a / "tb").mkdir(parents=True)
    (run_dir_b / "tb").mkdir(parents=True)

    runner_a = _train_tiny(tmp_path, run_dir_a / "tb", ablation="ref_bc")
    runner_b = _train_tiny(tmp_path, run_dir_b / "tb", ablation="ref_bc_rw")

    exp_a = runner_a._game_config.exp_path
    exp_b = runner_b._game_config.exp_path
    assert exp_a != exp_b, (
        f"seed=0 runs with different tensorboard_dirs collided on exp_path: {exp_a}"
    )
    assert runner_a._game_config.model_dir != runner_b._game_config.model_dir
    # both anchored under their own run_dir, not the old hardcoded fork-tree path
    assert str(run_dir_a) in exp_a
    assert str(run_dir_b) in exp_b


def test_missing_tensorboard_dir_falls_back_to_legacy_path():
    """No tensorboard_dir supplied (e.g. ad hoc train() call) keeps the old
    hardcoded seed-keyed path -- only harness-launched grids get isolation."""
    pytest.importorskip("torch")
    pytest.importorskip("pettingzoo")
    from hyper_mve.comparison import create_runner
    from hyper_mve.utils.configs import V4Config
    from hyper_mve.envs.adapters.pettingzoo_wrapper import RelationCommonsPettingZooEnv

    cfg = V4Config.from_preset("rel_duo")
    runner = create_runner(cfg, "mazero_mixed")
    env_fn = lambda: RelationCommonsPettingZooEnv(
        cfg.env, oracle_mode=False, eval_info_mode=False,
    )
    runner.train(cfg, env_fn, total_env_steps=600, lr=0.02, seed=0)
    assert "seed=0" in runner._game_config.exp_path
