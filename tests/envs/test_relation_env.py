"""Unit tests for ``hyper_mve.envs.relation_commons`` (v5 Pkg-09).

Covers: observation layout/offsets, own-row block correctness, regime
statics under p=0 / switching under p>0, reset pinning + holdout
restriction, info schema grouping, determinism, and the scripted-agent
reward-sign sanity checks (coop ⇒ bystander gains, comp ⇒ bystander loses,
neutral ⇒ bystander unaffected).
"""
from __future__ import annotations

import numpy as np
import pytest

from hyper_mve.configs.env_config import EnvConfig
from hyper_mve.envs.relation_commons import RelationCommonsEnv, make_relation_commons
from hyper_mve.schemas import AgentType, RelationObservationLayout, slice_relation_block

NOOP, HARVEST = 0, 5


def _cfg(N: int = 2, **kw) -> EnvConfig:
    base = dict(
        N=N, L=8, K=8, M=1, T_max=20,
        relation_family="g2" if N == 2 else "g4",
        type_assignment=tuple([AgentType.ALPHA] * N),
    )
    base.update(kw)
    return EnvConfig(**base)


def _place(env: RelationCommonsEnv, agent_id: int, pos) -> None:
    env._state.agent_positions[agent_id] = np.asarray(pos, dtype=np.int32)


# ---------------------------------------------------------------------------
# Layout / observation
# ---------------------------------------------------------------------------

def test_obs_shape_and_layout_dims():
    assert RelationObservationLayout.total_dim(2, 8) == 39
    assert RelationObservationLayout.total_dim(4, 20) == 95
    env = make_relation_commons(_cfg(), seed=0)
    obs, _ = env.reset()
    assert obs.shape == (2, 39) and obs.dtype == np.float32
    assert env.observation_space.shape == (2, 39)


def test_block_offsets_are_contiguous():
    N, K = 2, 8
    start = 0
    for b in RelationObservationLayout.BLOCK_ORDER:
        s, e = RelationObservationLayout.block_offset(b, N, K)
        assert s == start
        start = e
    assert start == RelationObservationLayout.total_dim(N, K)


def test_own_row_block_matches_regime():
    env = make_relation_commons(_cfg(), seed=1)
    obs, info = env.reset(options={"g": 2})  # asym_exploit: W = [[1,-1],[1,1]]
    row0 = slice_relation_block(obs[0], "row", 2, 8)
    row1 = slice_relation_block(obs[1], "row", 2, 8)
    np.testing.assert_array_equal(row0, [-1.0])
    np.testing.assert_array_equal(row1, [1.0])
    np.testing.assert_array_equal(info["rows"], [[-1.0], [1.0]])


def test_global_block_is_time_remaining_only():
    env = make_relation_commons(_cfg(), seed=2)
    obs, _ = env.reset()
    g_block = slice_relation_block(obs[0], "global", 2, 8)
    assert g_block.shape == (1,)
    assert g_block[0] == pytest.approx(1.0)
    obs, *_ = env.step(np.array([NOOP, NOOP]))
    g_block = slice_relation_block(obs[0], "global", 2, 8)
    assert g_block[0] == pytest.approx((20 - 1) / 20)


# ---------------------------------------------------------------------------
# Regime dynamics
# ---------------------------------------------------------------------------

def test_p0_regime_static_within_episode_resampled_across():
    env = make_relation_commons(_cfg(regime_switch_prob=0.0), seed=3)
    _, info = env.reset()
    g0 = info["g_true"]
    for _ in range(20):
        _, _, done, _, info = env.step(np.array([NOOP, NOOP]))
        assert info["g_true"] == g0
    assert done
    seen = {env.reset()[1]["g_true"] for _ in range(60)}
    assert len(seen) >= 4  # 5 regimes, 60 resets: all-but-degenerate coverage


def test_p_half_switch_frequency():
    env = make_relation_commons(_cfg(regime_switch_prob=0.5, T_max=4000), seed=4)
    _, info = env.reset()
    prev, switches = info["g_true"], 0
    for _ in range(4000):
        _, _, _, _, info = env.step(np.array([NOOP, NOOP]))
        switches += info["g_true"] != prev
        prev = info["g_true"]
    assert abs(switches / 4000 - 0.5) < 0.03


def test_switch_updates_own_row_observation():
    env = make_relation_commons(_cfg(regime_switch_prob=1.0, T_max=50), seed=5)
    obs, info = env.reset()
    for _ in range(10):
        obs, _, _, _, info = env.step(np.array([NOOP, NOOP]))
        # post-step obs must already show the (possibly switched) new row
        for i in range(2):
            np.testing.assert_array_equal(
                slice_relation_block(obs[i], "row", 2, 8), info["rows"][i],
            )


def test_reset_pin_and_holdout_restriction():
    cfg = _cfg(train_regime_ids=(0, 1, 4))
    env = make_relation_commons(cfg, seed=6)
    for _ in range(50):
        assert env.reset()[1]["g_true"] in {0, 1, 4}
    # pinning bypasses the restriction (holdout evaluation path)
    assert env.reset(options={"g": 3})[1]["g_true"] == 3
    with pytest.raises(AssertionError, match="out of"):
        env.reset(options={"g": 5})


def test_determinism_same_seed_same_trajectory():
    def rollout():
        env = make_relation_commons(_cfg(regime_switch_prob=0.3), seed=42)
        obs, info = env.reset()
        acc = [obs.copy()]
        rng = np.random.default_rng(0)
        for _ in range(10):
            a = rng.integers(0, 6, size=2)
            obs, r, *_ , info = env.step(a)
            acc.append((obs.copy(), r.copy(), info["g_true"]))
        return acc

    a, b = rollout(), rollout()
    np.testing.assert_array_equal(a[0], b[0])
    for (oa, ra, ga), (ob, rb, gb) in zip(a[1:], b[1:]):
        np.testing.assert_array_equal(oa, ob)
        np.testing.assert_array_equal(ra, rb)
        assert ga == gb


# ---------------------------------------------------------------------------
# Info schema
# ---------------------------------------------------------------------------

def test_info_schema_grouping():
    env = make_relation_commons(_cfg(), seed=7)
    _, info = env.reset()
    assert info["_info_schema_version"] == "v5.0"
    assert info["_oracle_fields"] == ("g_true", "rows")
    assert info["_eval_only_fields"] == ("resource_state",)
    assert info["rows"].shape == (2, 1)
    assert info["resource_state"].shape == (8, 3)
    assert "c_true" not in info and "types" not in info and "caps" not in info


# ---------------------------------------------------------------------------
# Scripted reward-sign sanity (the cheap behavioral contract)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "g, expected_sign",
    [(0, +1), (1, -1), (4, 0)],  # mutual_coop / mutual_comp / neutral
)
def test_bystander_reward_sign(g, expected_sign):
    env = make_relation_commons(_cfg(), seed=8)
    _, _ = env.reset(options={"g": g})
    # agent 0 harvests a full cell; agent 1 idles far away
    _place(env, 0, env._state.resource_positions[0])
    _place(env, 1, [0, 0] if not np.array_equal(env._state.resource_positions[0], [0, 0]) else [7, 7])
    _, reward, *_ = env.step(np.array([HARVEST, NOOP]))
    u0 = 1.0  # η=1 cap, full stock
    assert reward[0] == pytest.approx({0: u0, 1: 0.5, 4: u0}[g] if g != 0 else 0.5)
    if expected_sign > 0:
        assert reward[1] == pytest.approx(0.5)      # (0 + 1·u0) / 2
    elif expected_sign < 0:
        assert reward[1] == pytest.approx(-0.5)     # (0 − 1·u0) / 2
    else:
        assert reward[1] == pytest.approx(0.0)      # (0 + 0·u0) / 1


def test_harvest_depletes_and_regen_restores():
    env = make_relation_commons(_cfg(alpha=0.1), seed=9)
    env.reset(options={"g": 4})
    _place(env, 0, env._state.resource_positions[0])
    _place(env, 1, [7, 7])
    q0 = float(env._state.resource_stocks[0])
    env.step(np.array([HARVEST, NOOP]))
    q1 = float(env._state.resource_stocks[0])
    # depleted by η=1, regrown by α(Q_max − q0) with q0 = Q_max ⇒ regrowth 0
    assert q1 == pytest.approx(q0 - 1.0)
    env.step(np.array([NOOP, NOOP]))
    q2 = float(env._state.resource_stocks[0])
    assert q2 == pytest.approx(q1 + 0.1 * (10.0 - q1))


def test_move_cost_and_bounds():
    env = make_relation_commons(_cfg(), seed=10)
    env.reset(options={"g": 4})
    _place(env, 0, [0, 0])
    _place(env, 1, [7, 7])
    _, reward, *_ = env.step(np.array([3, NOOP]))  # LEFT into the wall
    assert env._state.agent_positions[0].tolist() == [0, 0]  # clipped
    assert reward[0] == pytest.approx(-0.01)  # intent cost despite no motion
    assert reward[1] == pytest.approx(0.0)


def test_episode_terminates_at_t_max():
    env = make_relation_commons(_cfg(T_max=5), seed=11)
    env.reset()
    for t in range(5):
        _, _, done, truncated, _ = env.step(np.array([NOOP, NOOP]))
        assert not truncated
        assert done == (t == 4)


def test_g4_family_env():
    env = make_relation_commons(_cfg(N=4, K=20), seed=12)
    obs, info = env.reset(options={"g": 2})  # pair_01_23
    assert obs.shape == (4, 95)
    np.testing.assert_array_equal(info["rows"][0], [1.0, -1.0, -1.0])
    np.testing.assert_array_equal(info["rows"][3], [-1.0, -1.0, 1.0])
