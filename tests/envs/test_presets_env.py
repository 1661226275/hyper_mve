"""End-to-end preset dimension tests (Pkg-02 spec 07 §5.1)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hyper_mve.configs import V4Config
from hyper_mve.envs.resource_commons import ResourceCommonsEnv
from hyper_mve.schemas import AgentType, ObservationLayout


@pytest.fixture(params=["easy", "medium", "hard"])
def preset_env(request):
    cfg = V4Config.from_preset(request.param)
    return request.param, cfg, ResourceCommonsEnv(cfg.env, seed=42)


def test_easy_env_dimensions():
    cfg = V4Config.from_preset("easy")
    env = ResourceCommonsEnv(cfg.env, seed=42)
    assert (env.N, env.L, env.K, env.M, env.T_max) == (2, 8, 8, 1, 100)
    obs, info = env.reset()
    assert obs.shape == (2, 45)            # 4 + 24 + 9 + 2 + 4 + 2


def test_medium_env_dimensions():
    cfg = V4Config.from_preset("medium")
    env = ResourceCommonsEnv(cfg.env, seed=42)
    assert env.N == 4 and env.K == 20
    obs, info = env.reset()
    assert obs.shape == (4, 99)


def test_hard_env_dimensions():
    cfg = V4Config.from_preset("hard")
    env = ResourceCommonsEnv(cfg.env, seed=42)
    assert env.N == 8
    obs, info = env.reset()
    assert obs.shape == (8, 195)


def test_easy_type_assignment_1a_1b():
    cfg = V4Config.from_preset("easy")
    assert cfg.env.type_assignment == (AgentType.ALPHA, AgentType.BETA)


def test_medium_type_assignment_2a_2b():
    cfg = V4Config.from_preset("medium")
    assert cfg.env.type_assignment == (
        AgentType.ALPHA, AgentType.ALPHA,
        AgentType.BETA, AgentType.BETA,
    )


def test_hard_type_assignment_4a_4b():
    cfg = V4Config.from_preset("hard")
    types = cfg.env.type_assignment
    assert len(types) == 8
    assert sum(1 for t in types if t == AgentType.ALPHA) == 4
    assert sum(1 for t in types if t == AgentType.BETA) == 4


def test_hard_c_mode_is_oscillate():
    cfg = V4Config.from_preset("hard")
    assert cfg.env.c_mode == "oscillate"


def test_preset_switch_zero_branching(preset_env):
    """obs dim is always derived via ObservationLayout — no preset string in env code."""
    name, cfg, env = preset_env
    obs, info = env.reset()
    expected = ObservationLayout.total_dim(cfg.env.N, cfg.env.K)
    assert obs.shape == (cfg.env.N, expected)


def test_no_hardcoded_preset_name_in_env_module():
    """env.py must not branch on preset names."""
    import inspect

    from hyper_mve.envs.resource_commons import env as env_module

    src = inspect.getsource(env_module)
    assert "if preset ==" not in src
    assert "if cfg.preset_name ==" not in src


def test_custom_type_assignment_override_via_replace():
    """Ablation 3 — overriding type_assignment via dataclasses.replace works."""
    cfg = V4Config.from_preset("medium")
    cfg_3a1b = replace(
        cfg,
        env=replace(
            cfg.env,
            type_assignment=(AgentType.ALPHA,) * 3 + (AgentType.BETA,),
        ),
    )
    env = ResourceCommonsEnv(cfg_3a1b.env, seed=42)
    _, info = env.reset()
    assert info["types"].tolist() == [0, 0, 0, 1]
