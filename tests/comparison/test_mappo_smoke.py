"""C7-EXT-SMOKE1 — MAPPO rel_duo 20K env-step smoke gate (pkg-07 spec 05 §10.2,
retargeted to v5 RelationCommons by Pkg-09).

Training and eval are pinned to regime 0 (``mutual_coop``): the gate compares
summed-over-agents episode return against random play, and in ``mutual_comp``
the relational reward is zero-sum by construction (Σ_i R_i ≡ 0 up to move
penalties), which would dilute the signal on a mixed-regime grid.

Gated by ``@pytest.mark.slow`` because a 20K env-step training loop on
the workstation takes minutes of GPU time. The pkg-08 sweep harness
CI runs this; default ``pytest`` does not unless ``--runslow`` is passed.
"""
from __future__ import annotations

from dataclasses import replace

import pytest


@pytest.mark.slow
def test_mappo_rel_duo_smoke_return_beats_random_at_20k():
    """C7-EXT-SMOKE1 MAPPO smoke gate (rel_duo, regime 0 pinned)."""
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

    # Random baseline at step 0.
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

    # MAPPO trained for 20K env steps per seed.
    trained_returns = []
    for s in seeds:
        runner = create_runner(cfg, "mappo")
        env_fn = lambda: RelationCommonsPettingZooEnv(
            cfg.env, oracle_mode=False, eval_info_mode=False,
        )
        runner.train(cfg, env_fn, total_env_steps=smoke_budget, lr=lr, seed=s)
        report = runner.evaluate(env_fn, regime_grid=(0,), episodes=10)
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
