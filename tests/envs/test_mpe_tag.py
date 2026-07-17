"""MPETagRegimeEnv — regime-ified simple_tag (phase-3 gates).

Covers: homogeneous padded obs (+ own-row tail), W(g) reward equal to the
hand-computed relational formula, per-episode resampling honoring
train_regime_ids, the oracle info gate, and the W=I calibration identity.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mpe2")

from hyper_mve.envs.mpe_tag import MPETagRegimeEnv  # noqa: E402
from hyper_mve.utils.configs import V4Config  # noqa: E402
from hyper_mve.utils.schemas.relation import (  # noqa: E402
    compute_relational_rewards,
    get_regime_family,
)


def _mk(preset="mpe_tag", oracle=False, fixed_regime=None):
    cfg = V4Config.from_preset(preset)
    return cfg, MPETagRegimeEnv(cfg.env, oracle_mode=oracle,
                                fixed_regime=fixed_regime)


def _acts(env, a=1):
    return {name: a for name in env.possible_agents}


def test_homogeneous_padded_obs_with_own_row_tail():
    cfg, env = _mk(oracle=True)
    obs, info = env.reset(seed=0, options={"g": 0})  # pred_full_coalition
    assert env.possible_agents == [f"agent_{i}" for i in range(4)]
    for name, o in obs.items():
        assert o.shape == (19,) and o.dtype == np.float32
    # prey (agent_3) native obs is 14-dim → padded dims 14:16 are zero
    assert obs["agent_3"][14:16].tolist() == [0.0, 0.0]
    # own-row tail: g0 gives predator 0 the row (w01,w02,w03) = (+1,+1,0)
    fam = get_regime_family(cfg.env)
    assert np.allclose(obs["agent_0"][16:], fam.regimes[0].row(0))
    assert np.allclose(obs["agent_3"][16:], fam.regimes[0].row(3))
    assert env.action_space("agent_0").n == 5


@pytest.mark.parametrize("g", [0, 3, 4])
def test_reward_equals_hand_computed_relational_formula(g):
    cfg, env = _mk(oracle=True)
    env.reset(seed=1, options={"g": g})
    fam = get_regime_family(cfg.env)
    W = fam.regimes[g].w_array()
    for _ in range(5):
        acts = {name: int(i % 5) for i, name in enumerate(env.possible_agents)}
        obs, rew, term, trunc, info = env.step(acts)
        u = info["agent_0"]["harvests"]
        expected = compute_relational_rewards(
            harvests=u,
            moved_mask=np.array([i % 5 != 0 for i in range(4)]),
            W=W,
            epsilon_move=cfg.env.epsilon_move,
        )
        got = np.array([rew[f"agent_{i}"] for i in range(4)], dtype=np.float32)
        assert np.allclose(got, expected, atol=1e-6), (g, got, expected)


def test_resampling_honors_train_regime_ids():
    cfg, env = _mk(oracle=True)  # mpe_tag preset: train_regime_ids=(0, 1, 2)
    seen = set()
    for k in range(40):
        env.reset(seed=k)
        _, _, _, _, info = env.step(_acts(env))
        seen.add(int(info["agent_0"]["g_true"]))
    assert seen <= {0, 1, 2}, f"holdout leak: sampled {seen}"
    assert len(seen) >= 2  # actually resamples


def test_options_g_pins_regime_bypassing_holdout():
    cfg, env = _mk(oracle=True)
    env.reset(seed=0, options={"g": 4})  # 4 is OUTSIDE train_regime_ids
    _, _, _, _, info = env.step(_acts(env))
    assert int(info["agent_0"]["g_true"]) == 4


def test_oracle_gate_strips_g_true_and_rows_by_default():
    _, env = _mk(oracle=False)
    obs, info = env.reset(seed=0)
    for key in ("g_true", "rows", "_oracle_fields", "_info_schema_version"):
        assert key not in info["agent_0"], key
    _, _, _, _, info = env.step(_acts(env))
    assert "g_true" not in info["agent_0"]
    assert "harvests" in info["agent_0"]  # public field survives


def test_fixed_regime_calibration_reproduces_raw_rewards():
    """mpe_tag_fixed: W = I + epsilon_move = 0 ⇒ reward ≡ raw physical u."""
    cfg, env = _mk(preset="mpe_tag_fixed", oracle=True)
    obs, _ = env.reset(seed=3)
    saw_nonzero = False
    for t in range(25):
        # prey (agent_3) runs in one constant direction → boundary penalty
        # guarantees a nonzero physical reward within the episode
        acts = {"agent_0": 0, "agent_1": 0, "agent_2": 0, "agent_3": 1}
        obs, rew, term, trunc, info = env.step(acts)
        u = info["agent_0"]["harvests"]
        got = np.array([rew[f"agent_{i}"] for i in range(4)], dtype=np.float32)
        assert np.allclose(got, u, atol=1e-6)
        assert int(info["agent_0"]["g_true"]) == 2  # all_solo, never resampled
        saw_nonzero = saw_nonzero or bool(np.any(u != 0))
        if any(term.values()) or any(trunc.values()):
            break
    # simple_tag's prey draws boundary penalties once it leaves the arena —
    # the harvests plumbing must surface real (nonzero) physical rewards
    assert saw_nonzero


def test_episode_truncates_at_t_max():
    cfg, env = _mk()
    env.reset(seed=0)
    for t in range(cfg.env.T_max):
        _, _, term, trunc, _ = env.step(_acts(env))
    assert any(trunc.values()) or any(term.values())
    assert env.agents == []
