"""C7-EXT-SMOKE1 — QMIX Easy 20K env-step smoke gate (pkg-07 spec 06 §2.9).

Test pattern inherited verbatim from spec 05 §10.2 (MAPPO smoke).
Gated by ``@pytest.mark.slow``.
"""
from __future__ import annotations

import pytest


@pytest.mark.slow
def test_qmix_easy_smoke_return_beats_random_at_20k():
    """C7-EXT-SMOKE1 QMIX smoke gate. Per spec 06 §2.9 + §9.2."""
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

    random_returns = []
    for s in seeds:
        env = ResourceCommonsPettingZooEnv(
            cfg.env, oracle_mode=False, eval_info_mode=False,
        )
        _obs, _ = env.reset(seed=s)
        ep_return = 0.0
        rng = np.random.default_rng(s)
        for _ in range(cfg.env.T_max):
            action_dict = {a: int(rng.integers(0, 6)) for a in env.agents}
            _obs, reward, term, trunc, _info = env.step(action_dict)
            ep_return += sum(reward.values())
            if any(term.values()) or any(trunc.values()):
                break
        random_returns.append(ep_return)
        env.close()
    random_returns = np.array(random_returns)

    trained_returns = []
    for s in seeds:
        runner = create_baseline(cfg, "external_qmix")
        env_fn = lambda: ResourceCommonsPettingZooEnv(
            cfg.env, oracle_mode=False, eval_info_mode=False,
        )
        runner.train(cfg, env_fn, total_env_steps=smoke_budget, lr=lr, seed=s)
        report = runner.evaluate(env_fn, c_grid=(0.5,), episodes=10)
        trained_returns.append(report.return_mean)
    trained_returns = np.array(trained_returns)

    t_stat, p_two_sided = stats.ttest_ind(
        trained_returns, random_returns, equal_var=False,
    )
    p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2
    assert trained_returns.mean() > random_returns.mean()
    assert p_one_sided < 0.05, (
        f"QMIX smoke p={p_one_sided:.4f} above 0.05 threshold "
        f"(t={t_stat:.2f})."
    )
