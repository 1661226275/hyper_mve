"""C7-EXT-SMOKE1 — MAPPO Easy 20K env-step smoke gate (pkg-07 spec 05 §10.2).

Gated by ``@pytest.mark.slow`` because a 20K env-step training loop on
the workstation takes ~5 minutes of GPU time. The pkg-08 sweep harness
CI runs this; default ``pytest`` does not unless ``--runslow`` is passed.
"""
from __future__ import annotations

import pytest


@pytest.mark.slow
def test_mappo_easy_smoke_return_beats_random_at_20k():
    """C7-EXT-SMOKE1 MAPPO smoke gate. Per spec 05 §10.2 verbatim."""
    pytest.importorskip("torch")
    pytest.importorskip("pettingzoo")
    pytest.importorskip("scipy")

    import numpy as np
    from scipy import stats

    from hyper_mve.baselines import create_baseline
    from hyper_mve.configs import V4Config
    from hyper_mve.envs.adapters.pettingzoo_wrapper import ResourceCommonsPettingZooEnv

    cfg = V4Config.from_preset("easy")
    smoke_budget = 20_000
    seeds = (0, 1, 2)
    lr = 3e-4

    # Random baseline at step 0.
    random_returns = []
    for s in seeds:
        env = ResourceCommonsPettingZooEnv(
            cfg.env, oracle_mode=False, eval_info_mode=False,
        )
        obs, _ = env.reset(seed=s)
        ep_return = 0.0
        rng = np.random.default_rng(s)
        for _ in range(cfg.env.T_max):
            action_dict = {a: int(rng.integers(0, 6)) for a in env.agents}
            obs, reward, term, trunc, info = env.step(action_dict)
            ep_return += sum(reward.values())
            if any(term.values()) or any(trunc.values()):
                break
        random_returns.append(ep_return)
        env.close()
    random_returns = np.array(random_returns)

    # MAPPO trained for 20K env steps per seed.
    trained_returns = []
    for s in seeds:
        runner = create_baseline(cfg, "external_mappo")
        env_fn = lambda: ResourceCommonsPettingZooEnv(
            cfg.env, oracle_mode=False, eval_info_mode=False,
        )
        runner.train(cfg, env_fn, total_env_steps=smoke_budget, lr=lr, seed=s)
        report = runner.evaluate(env_fn, c_grid=(0.5,), episodes=10)
        trained_returns.append(report.return_mean)
    trained_returns = np.array(trained_returns)

    # One-sided Welch t-test, alpha=0.05.
    t_stat, p_two_sided = stats.ttest_ind(
        trained_returns, random_returns, equal_var=False,
    )
    p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2
    assert trained_returns.mean() > random_returns.mean(), (
        f"MAPPO mean {trained_returns.mean():.2f} not above random "
        f"{random_returns.mean():.2f}."
    )
    assert p_one_sided < 0.05, (
        f"MAPPO smoke p={p_one_sided:.4f} above 0.05 threshold "
        f"(t={t_stat:.2f}). C7-EXT-SMOKE1 violated."
    )
