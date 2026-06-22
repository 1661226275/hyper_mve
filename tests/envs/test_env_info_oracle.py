"""info dict contract tests — v4 Oracle signal (Pkg-02 spec 08 §5.1).

These are load-bearing: Pkg-03 BeliefNet's L_c / L_opp supervised training
reads ``info["c_true"]`` and ``info["types"]`` from this env. Two regression
tests pin the falsy-short-circuit bug that bit v3.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from hyper_mve.configs import V4Config
from hyper_mve.envs.resource_commons import ResourceCommonsEnv
from hyper_mve.schemas import AgentType, ObservationLayout


# ---------- presence and types of Oracle fields ----------

def test_info_has_oracle_fields():
    env = ResourceCommonsEnv(V4Config.from_preset("medium").env, seed=42)
    _, info = env.reset()
    assert "c_true" in info
    assert "types" in info
    assert "caps" in info
    assert isinstance(info["c_true"], float)
    assert info["types"].shape == (4,)
    assert info["types"].dtype == np.int8
    assert len(info["caps"]) == 4


def test_info_oracle_matches_state():
    env = ResourceCommonsEnv(V4Config.from_preset("medium").env, seed=42)
    _, info = env.reset()
    assert info["c_true"] == env._state.c_t
    assert np.array_equal(info["types"], env._state.agent_types)


def test_info_eval_fields_present():
    env = ResourceCommonsEnv(V4Config.from_preset("medium").env, seed=42)
    _, info = env.reset()
    assert info["hotspot_centers"].shape == (3, 2)
    assert info["resource_state"].shape == (20, 3)


def test_info_schema_version_marker():
    env = ResourceCommonsEnv(V4Config.from_preset("medium").env, seed=42)
    _, info = env.reset()
    assert info["_info_schema_version"] == "v4.0"
    assert "c_true" in info["_oracle_fields"]
    assert "types" in info["_oracle_fields"]
    assert "hotspot_centers" in info["_eval_only_fields"]


# ---------- gym 5-tuple step contract ----------

def test_step_returns_five_tuple_with_right_shapes():
    env = ResourceCommonsEnv(V4Config.from_preset("medium").env, seed=42)
    env.reset()
    action = np.zeros(4, dtype=np.int64)
    obs, reward, done, truncated, info = env.step(action)
    assert obs.shape == (4, 99)
    assert reward.shape == (4,)
    assert reward.dtype == np.float32
    assert isinstance(done, (bool, np.bool_))
    assert truncated is False
    assert "c_true" in info


def test_observation_space_matches_obs_shape():
    env = ResourceCommonsEnv(V4Config.from_preset("medium").env, seed=42)
    assert env.observation_space.shape == (4, 99)


def test_action_space_is_multidiscrete():
    env = ResourceCommonsEnv(V4Config.from_preset("medium").env, seed=42)
    assert env.action_space.nvec.tolist() == [6, 6, 6, 6]


# ---------- reset reproducibility ----------

def test_reset_same_seed_same_obs():
    cfg = V4Config.from_preset("medium")
    env1 = ResourceCommonsEnv(cfg.env, seed=42)
    env2 = ResourceCommonsEnv(cfg.env, seed=42)
    obs1, info1 = env1.reset()
    obs2, info2 = env2.reset()
    assert np.allclose(obs1, obs2)
    assert info1["c_true"] == info2["c_true"]


# ---------- options ----------

def test_reset_options_force_c():
    env = ResourceCommonsEnv(V4Config.from_preset("medium").env, seed=42)
    _, info = env.reset(options={"c": 0.7})
    assert info["c_true"] == pytest.approx(0.7)


def test_reset_options_force_c_zero_regression():
    """Bug: ``options.get('c') or fallback()`` silently swallows c=0.0.

    Implementation MUST use ``"c" in options`` so the lean-season evaluation
    point (c=0) is not lost.
    """
    env = ResourceCommonsEnv(V4Config.from_preset("medium").env, seed=42)
    _, info = env.reset(options={"c": 0.0})
    assert info["c_true"] == 0.0
    _, info = env.reset(options={"c": 1.0})
    assert info["c_true"] == 1.0


def test_reset_options_c_out_of_range_rejected():
    env = ResourceCommonsEnv(V4Config.from_preset("medium").env, seed=42)
    with pytest.raises(AssertionError, match=r"options\['c'\]"):
        env.reset(options={"c": 1.5})
    with pytest.raises(AssertionError):
        env.reset(options={"c": -0.1})


def test_reset_options_force_types():
    env = ResourceCommonsEnv(V4Config.from_preset("medium").env, seed=42)
    custom_types = (AgentType.ALPHA,) * 3 + (AgentType.BETA,)
    _, info = env.reset(options={"types": custom_types})
    assert info["types"].tolist() == [0, 0, 0, 1]


def test_reset_options_none_falls_back_to_cfg():
    env = ResourceCommonsEnv(V4Config.from_preset("medium").env, seed=42)
    _, info_default = env.reset()
    _, info_none = env.reset(options=None)
    assert info_default["types"].tolist() == info_none["types"].tolist() == [0, 0, 1, 1]


# ---------- c_t hidden mode wiring (Ch3.7.3) ----------

def test_c_hidden_mode_masks_obs_but_keeps_oracle_label():
    """c_visible=False masks the obs c slot; info['c_true'] stays the truth."""
    cfg = dataclasses.replace(V4Config.from_preset("medium").env, c_visible=False)
    env = ResourceCommonsEnv(cfg, seed=42)
    obs, info = env.reset(options={"c": 0.9})
    start, _ = ObservationLayout.block_offset("global", N=cfg.N, K=cfg.K)
    assert np.allclose(obs[:, start], 0.5)
    assert info["c_true"] == pytest.approx(0.9)

    obs, _, _, _, info = env.step(np.zeros(cfg.N, dtype=np.int64))
    assert np.allclose(obs[:, start], 0.5)
    # only the c slot is masked: time_remaining must keep ticking
    assert np.allclose(obs[:, start + 1], (cfg.T_max - 1) / cfg.T_max)
    # static c_mode: oracle label still reports the true (unchanged) c_t
    assert info["c_true"] == pytest.approx(0.9)


def test_c_hidden_mode_affects_observations_only():
    """Same seed, c_visible flipped: dynamics/rewards/info identical, obs
    differ in at most the per-agent global c slot (evolving c via oscillate)."""
    base = V4Config.from_preset("medium").env
    cfg_vis = dataclasses.replace(base, c_mode="oscillate")
    cfg_hid = dataclasses.replace(base, c_mode="oscillate", c_visible=False)
    env_vis = ResourceCommonsEnv(cfg_vis, seed=7)
    env_hid = ResourceCommonsEnv(cfg_hid, seed=7)
    obs_v, info_v = env_vis.reset(seed=7)
    obs_h, info_h = env_hid.reset(seed=7)
    start, _ = ObservationLayout.block_offset("global", N=base.N, K=base.K)
    non_c = np.ones(obs_v.shape[1], dtype=bool)
    non_c[start] = False

    rng = np.random.default_rng(0)
    for _ in range(5):
        action = rng.integers(0, base.A, size=base.N).astype(np.int64)
        obs_v, rew_v, _, _, info_v = env_vis.step(action)
        obs_h, rew_h, _, _, info_h = env_hid.step(action)
        assert np.allclose(rew_v, rew_h)
        assert info_v["c_true"] == pytest.approx(info_h["c_true"])
        assert np.allclose(obs_v[:, non_c], obs_h[:, non_c])
        assert np.allclose(obs_h[:, start], 0.5)
        assert np.allclose(obs_v[:, start], info_v["c_true"])


def test_c_visible_default_keeps_c_in_obs_regression():
    """Default cfg (c_visible=True) still exposes c_t in the global block."""
    cfg = V4Config.from_preset("medium").env
    assert cfg.c_visible is True
    env = ResourceCommonsEnv(cfg, seed=42)
    obs, info = env.reset(options={"c": 0.9})
    start, _ = ObservationLayout.block_offset("global", N=cfg.N, K=cfg.K)
    assert np.allclose(obs[:, start], 0.9)
    assert info["c_true"] == pytest.approx(0.9)


# ---------- episode termination ----------

def test_done_at_t_max():
    cfg = V4Config.from_preset("medium")
    env = ResourceCommonsEnv(cfg.env, seed=42)
    env.reset()
    T = cfg.env.T_max
    for _ in range(T - 1):
        _, _, done, _, _ = env.step(np.zeros(4, dtype=np.int64))
        assert not done
    _, _, done, _, _ = env.step(np.zeros(4, dtype=np.int64))
    assert done


# ---------- render ----------

def test_render_rgb_array():
    env = ResourceCommonsEnv(V4Config.from_preset("medium").env, seed=42)
    env.reset()
    img = env.render(mode="rgb_array")
    assert isinstance(img, np.ndarray)
    assert img.dtype == np.uint8
    assert img.ndim == 3 and img.shape[2] == 3


def test_render_invalid_mode_raises():
    env = ResourceCommonsEnv(V4Config.from_preset("medium").env, seed=42)
    env.reset()
    with pytest.raises(NotImplementedError):
        env.render(mode="human")


# ---------- factory + close ----------

def test_make_resource_commons_factory():
    from hyper_mve.envs.resource_commons import make_resource_commons

    env = make_resource_commons(V4Config.from_preset("easy").env, seed=0)
    assert isinstance(env, ResourceCommonsEnv)
    obs, _ = env.reset()
    assert obs.shape == (2, 45)


def test_close_is_noop():
    env = ResourceCommonsEnv(V4Config.from_preset("medium").env, seed=42)
    env.reset()
    env.close()    # must not raise
