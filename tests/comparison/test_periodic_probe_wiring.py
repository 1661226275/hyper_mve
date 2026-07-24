"""2026-07-20: periodic eval/* wiring for happo, mbom, mbom_oracle, m3w_adapted.

These four previously logged NOTHING during training, only a final point.
Unlike test_{happo,mbom,m3w_adapted}_smoke.py (which don't all pass
tensorboard_dir and so don't necessarily exercise the probe construction
path at all), these tests assert the periodic tags actually land in a real
TB event file -- proving the probe genuinely fired, not just that training
didn't crash. happo's wiring in particular goes through a side-channel
(_ACTIVE["probe"], read by the vendored HARL clone's relation_logger.py) with
a real construction-order constraint (the probe is set into _ACTIVE AFTER
the HARL runner + its logger are already built), so an actual firing is the
only thing that proves the timing is right.
"""
from __future__ import annotations

import pytest


_CANDIDATE_TAGS = (
    b"eval/return_mean", b"eval/return_regime_0", b"eval/return_seen",
    b"fidelity/reward_mae", b"fidelity/reward_mae_regime_0",
)


def _tags(tb_dir) -> set[str]:
    files = list(tb_dir.rglob("events.out.tfevents.*"))
    assert files, f"no tfevents file under {tb_dir}"
    text = b"".join(f.read_bytes() for f in files)
    # Tag names are stored as plain UTF-8 strings inside the tfevents
    # protobuf stream -- cheap, dependency-free way to check what was
    # written without spinning up a TensorBoard EventAccumulator.
    return {tag for tag in _CANDIDATE_TAGS if tag in text}


def test_happo_periodic_probe_fires(tmp_path):
    pytest.importorskip("torch")

    from hyper_mve.comparison import create_runner
    from hyper_mve.utils.configs import V4Config
    from hyper_mve.envs.adapters.pettingzoo_wrapper import RelationCommonsPettingZooEnv

    cfg = V4Config.from_preset("rel_duo")
    runner = create_runner(cfg, "happo")
    env_fn = lambda: RelationCommonsPettingZooEnv(
        cfg.env, oracle_mode=False, eval_info_mode=False,
    )
    tb_dir = tmp_path / "tb"
    runner.train(cfg, env_fn, total_env_steps=400, lr=1e-3, seed=0,
                 tensorboard_dir=str(tb_dir))

    found = _tags(tb_dir)
    assert b"eval/return_mean" in found, (
        "periodic probe never fired -- check the _ACTIVE['probe'] "
        "construction-order timing in happo.py vs relation_logger.py")


def test_mbom_periodic_probe_fires(tmp_path):
    pytest.importorskip("torch")

    from hyper_mve.comparison import create_runner
    from hyper_mve.utils.configs import V4Config
    from hyper_mve.envs.adapters.pettingzoo_wrapper import RelationCommonsPettingZooEnv

    cfg = V4Config.from_preset("rel_duo")
    runner = create_runner(cfg, "mbom")
    env_fn = lambda: RelationCommonsPettingZooEnv(
        cfg.env, oracle_mode=False, eval_info_mode=False,
    )
    tb_dir = tmp_path / "tb"
    runner.train(cfg, env_fn, total_env_steps=400, lr=1e-3, seed=0,
                 tensorboard_dir=str(tb_dir))

    assert b"eval/return_mean" in _tags(tb_dir)


def test_m3w_adapted_periodic_probe_fires(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("einops")

    from hyper_mve.comparison import create_runner
    from hyper_mve.utils.configs import V4Config
    from hyper_mve.envs.adapters.pettingzoo_wrapper import RelationCommonsPettingZooEnv

    cfg = V4Config.from_preset("rel_duo")
    runner = create_runner(cfg, "m3w_adapted")
    env_fn = lambda: RelationCommonsPettingZooEnv(
        cfg.env, oracle_mode=False, eval_info_mode=False,
    )
    tb_dir = tmp_path / "tb"
    # Needs to clear warmup + at least one update_every=4 step for the first
    # train_step (and therefore the first probe.maybe_run call) to happen.
    runner.train(cfg, env_fn, total_env_steps=800, lr=1e-3, seed=0,
                 tensorboard_dir=str(tb_dir))

    assert b"eval/return_mean" in _tags(tb_dir)


# ------------------------------------------------------ periodic fidelity


def test_mamba_periodic_fidelity_fires(tmp_path):
    pytest.importorskip("torch")

    from hyper_mve.comparison import create_runner
    from hyper_mve.utils.configs import V4Config
    from hyper_mve.envs.adapters.pettingzoo_wrapper import RelationCommonsPettingZooEnv

    cfg = V4Config.from_preset("rel_duo")
    runner = create_runner(cfg, "mamba")
    env_fn = lambda: RelationCommonsPettingZooEnv(
        cfg.env, oracle_mode=False, eval_info_mode=False,
    )
    tb_dir = tmp_path / "tb"
    runner.train(cfg, env_fn, total_env_steps=400, lr=1e-3, seed=0,
                 tensorboard_dir=str(tb_dir))

    assert b"fidelity/reward_mae" in _tags(tb_dir)


def test_m3w_adapted_periodic_fidelity_fires(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("einops")

    from hyper_mve.comparison import create_runner
    from hyper_mve.utils.configs import V4Config
    from hyper_mve.envs.adapters.pettingzoo_wrapper import RelationCommonsPettingZooEnv

    cfg = V4Config.from_preset("rel_duo")
    runner = create_runner(cfg, "m3w_adapted")
    env_fn = lambda: RelationCommonsPettingZooEnv(
        cfg.env, oracle_mode=False, eval_info_mode=False,
    )
    tb_dir = tmp_path / "tb"
    runner.train(cfg, env_fn, total_env_steps=800, lr=1e-3, seed=0,
                 tensorboard_dir=str(tb_dir))

    assert b"fidelity/reward_mae" in _tags(tb_dir)


def test_mazero_mixed_periodic_fidelity_fires(tmp_path):
    """The hardest wiring: train_sync_serial (core/train.py, vendored fork's
    OWN loop, no env_fn parameter at all) constructs its own oracle-free
    PettingZoo env from config.env_cfg_override, guarded by case=="relation",
    and calls compute_fidelity_report at the same test_interval cadence as
    the fork's own periodic test(). Verifies that guard resolves correctly
    and the whole chain (belief-GRU rollout -> recurrent_inference -> MAE)
    actually runs against the LIVE in-training model, not just that nothing
    crashes."""
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
    # test_interval is a fixed 500 train steps (runner.py); this tiny budget
    # only reaches training_steps ~= 600//16 = 37, so the periodic block
    # fires exactly once, at step_count==0 (0 % 500 == 0) -- untrained
    # model, but that still exercises the full real chain end-to-end.
    tb_dir = tmp_path / "tb"
    runner.train(cfg, env_fn, total_env_steps=600, lr=0.02, seed=0,
                 tensorboard_dir=str(tb_dir))

    assert b"fidelity/reward_mae" in _tags(tb_dir)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
