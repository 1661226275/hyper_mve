"""C7-EXT-SMOKE-MAMBA — MAMBA real-port smoke gates (spec 06 §4.4 amendment,
sourced 2026-07-10; mirrors test_mappo_smoke.py).

Two tiers:
  * ``test_mamba_tiny_train_eval_ckpt_roundtrip`` — FAST: a few-episode train
    (enough to cross MIN_BUFFER_SIZE and trigger real model+agent updates),
    per-regime evaluate, checkpoint save/load roundtrip. Catches port breakage
    (torch-2.x API, shape bugs, buffer schema) without GPU-minutes.
  * ``test_mamba_rel_duo_smoke_return_beats_random_at_20k`` — @slow learning
    gate, regime 0 pinned, identical protocol to the MAPPO smoke.
"""
from __future__ import annotations

from dataclasses import replace

import pytest


def test_mamba_tiny_train_eval_ckpt_roundtrip(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("pettingzoo")

    from hyper_mve.comparison import create_runner
    from hyper_mve.utils.configs import V4Config
    from hyper_mve.envs.adapters.pettingzoo_wrapper import RelationCommonsPettingZooEnv
    from hyper_mve.utils.eval.eval_report import EvalReport

    cfg = V4Config.from_preset("rel_duo")
    runner = create_runner(cfg, "mamba")
    env_fn = lambda: RelationCommonsPettingZooEnv(
        cfg.env, oracle_mode=False, eval_info_mode=False,
    )

    # 3 episodes (300 env steps) — crosses MIN_BUFFER_SIZE=100, so the
    # learner runs real model + agent updates at least twice.
    runner.train(cfg, env_fn, total_env_steps=250, lr=3e-4, seed=0)
    assert runner.param_count() > 0

    report = runner.evaluate(env_fn, regime_grid=(0, 1), episodes=1)
    assert isinstance(report, EvalReport)
    assert report.variant == "mamba"
    assert set(report.return_per_regime) == {0, 1}

    # Checkpoint roundtrip incl. standalone load (fresh runner, no train()).
    ckpt = tmp_path / "mamba_smoke.pt"
    runner.save_checkpoint(ckpt)
    fresh = create_runner(cfg, "mamba")
    fresh.load_checkpoint(ckpt)
    assert fresh.param_count() == runner.param_count()
    report2 = fresh.evaluate(env_fn, regime_grid=(0,), episodes=1)
    assert isinstance(report2, EvalReport)


@pytest.mark.slow
def test_mamba_rel_duo_smoke_return_beats_random_at_20k():
    """MAMBA smoke gate (rel_duo, regime 0 pinned) — protocol identical to
    C7-EXT-SMOKE1 (MAPPO)."""
    pytest.importorskip("torch")
    pytest.importorskip("pettingzoo")
    pytest.importorskip("scipy")

    import numpy as np
    from scipy import stats

    from hyper_mve.comparison import create_runner
    from hyper_mve.utils.configs import V4Config
    from hyper_mve.envs.adapters.pettingzoo_wrapper import RelationCommonsPettingZooEnv

    cfg = V4Config.from_preset("rel_duo")
    cfg = replace(cfg, env=replace(cfg.env, train_regime_ids=(0,)))  # mutual_coop
    smoke_budget = 20_000
    seeds = (0, 1, 2)
    lr = 3e-4

    random_returns = []
    for s in seeds:
        env = RelationCommonsPettingZooEnv(
            cfg.env, oracle_mode=False, eval_info_mode=False,
        )
        obs, _ = env.reset(seed=s)
        ep_return = 0.0
        rng = np.random.default_rng(s)
        for _ in range(cfg.env.T_max):
            action_dict = {a: int(rng.integers(0, cfg.env.A)) for a in env.agents}
            obs, reward, term, trunc, info = env.step(action_dict)
            ep_return += sum(reward.values())
            if any(term.values()) or any(trunc.values()):
                break
        random_returns.append(ep_return)
        env.close()
    random_returns = np.array(random_returns)

    trained_returns = []
    for s in seeds:
        runner = create_runner(cfg, "mamba")
        env_fn = lambda: RelationCommonsPettingZooEnv(
            cfg.env, oracle_mode=False, eval_info_mode=False,
        )
        runner.train(cfg, env_fn, total_env_steps=smoke_budget, lr=lr, seed=s)
        report = runner.evaluate(env_fn, regime_grid=(0,), episodes=10)
        trained_returns.append(report.return_mean)
    trained_returns = np.array(trained_returns)

    t_stat, p_two_sided = stats.ttest_ind(
        trained_returns, random_returns, equal_var=False,
    )
    p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2
    assert trained_returns.mean() > random_returns.mean(), (
        f"MAMBA mean {trained_returns.mean():.2f} not above random "
        f"{random_returns.mean():.2f}."
    )
    assert p_one_sided < 0.05, (
        f"MAMBA smoke p={p_one_sided:.4f} above 0.05 threshold "
        f"(t={t_stat:.2f})."
    )
