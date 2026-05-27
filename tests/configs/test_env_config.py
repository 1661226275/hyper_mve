"""Unit tests for ``hyper_mve.configs.env_config``."""
from __future__ import annotations

import pytest

from hyper_mve.configs import EnvConfig
from hyper_mve.schemas import AgentType


def test_construct_minimum():
    env = EnvConfig(
        N=4, L=16, K=20, M=3, T_max=200,
        type_assignment=(
            AgentType.ALPHA, AgentType.ALPHA,
            AgentType.BETA, AgentType.BETA,
        ),
    )
    assert env.N == 4
    assert env.K == 20


def test_type_assignment_length_mismatch():
    with pytest.raises(ValueError, match="type_assignment"):
        EnvConfig(
            N=4, L=16, K=20, M=3, T_max=200,
            type_assignment=(AgentType.ALPHA,) * 3,
        )


def test_invalid_c_mode():
    with pytest.raises(ValueError, match="c_mode"):
        EnvConfig(
            N=2, L=8, K=8, M=1, T_max=100,
            type_assignment=(AgentType.ALPHA, AgentType.BETA),
            c_mode="bogus",
        )


def test_invalid_alpha_range():
    with pytest.raises(ValueError, match="alpha"):
        EnvConfig(
            N=2, L=8, K=8, M=1, T_max=100,
            type_assignment=(AgentType.ALPHA, AgentType.BETA),
            alpha_min=0.3, alpha_max=0.2,
        )


def test_frozen():
    env = EnvConfig(
        N=2, L=8, K=8, M=1, T_max=100,
        type_assignment=(AgentType.ALPHA, AgentType.BETA),
    )
    with pytest.raises(Exception):
        env.N = 4  # type: ignore[misc]


def test_default_capability_ranges_from_constants():
    env = EnvConfig(
        N=2, L=8, K=8, M=1, T_max=100,
        type_assignment=(AgentType.ALPHA, AgentType.BETA),
    )
    assert env.eta_range == (0.5, 1.5)
    assert env.phi_fov_range == (2.0, 4.0)
    assert env.nu_range == (0.8, 1.0)
    assert env.zeta_range == (10.0, 30.0)
