"""Unit tests for ``hyper_mve.utils.configs.env_config`` (v5 Pkg-09)."""
from __future__ import annotations

import pytest

from hyper_mve.utils.configs import EnvConfig


def test_construct_minimum():
    env = EnvConfig(N=2, L=8, K=8, T_max=100)
    assert env.N == 2
    assert env.K == 8
    # v5 defaults
    assert env.relation_family == "g2"
    assert env.relation_intensity == 1.0
    assert env.regime_switch_prob == 0.0
    assert env.regime_kernel == "uniform"
    assert env.train_regime_ids is None
    assert env.alpha == 0.10
    assert env.epsilon_move == 0.01


def test_invalid_relation_family():
    with pytest.raises(ValueError, match="relation_family"):
        EnvConfig(N=2, L=8, K=8, T_max=100, relation_family="bogus")


def test_invalid_relation_intensity():
    with pytest.raises(ValueError, match="relation_intensity"):
        EnvConfig(N=2, L=8, K=8, T_max=100, relation_intensity=0.0)


def test_invalid_switch_prob():
    with pytest.raises(ValueError, match="regime_switch_prob"):
        EnvConfig(N=2, L=8, K=8, T_max=100, regime_switch_prob=1.5)


def test_invalid_regime_kernel():
    with pytest.raises(ValueError, match="regime_kernel"):
        EnvConfig(N=2, L=8, K=8, T_max=100, regime_kernel="metropolis")


def test_regime_prior_must_be_distribution():
    with pytest.raises(ValueError, match="sum to 1"):
        EnvConfig(N=2, L=8, K=8, T_max=100,
                  regime_prior=(0.5, 0.2, 0.1, 0.1, 0.05))
    with pytest.raises(ValueError, match="negative"):
        EnvConfig(N=2, L=8, K=8, T_max=100,
                  regime_prior=(1.2, -0.2, 0.0, 0.0, 0.0))


def test_train_regime_ids_must_be_none_or_nonempty():
    with pytest.raises(ValueError, match="train_regime_ids"):
        EnvConfig(N=2, L=8, K=8, T_max=100, train_regime_ids=())


def test_invalid_alpha():
    with pytest.raises(ValueError, match="alpha"):
        EnvConfig(N=2, L=8, K=8, T_max=100, alpha=0.0)


def test_invalid_dims():
    with pytest.raises(ValueError):
        EnvConfig(N=0, L=8, K=8, T_max=100)
    with pytest.raises(ValueError):
        EnvConfig(N=2, L=8, K=8, T_max=0)


def test_frozen():
    env = EnvConfig(N=2, L=8, K=8, T_max=100)
    with pytest.raises(Exception):
        env.N = 4  # type: ignore[misc]


def test_v4_fields_gone():
    """Stage-6 lock: the type/c_t/capability surface must not regrow."""
    from dataclasses import fields
    names = {f.name for f in fields(EnvConfig)}
    for gone in ("type_assignment", "c_mode", "c_visible", "alpha_min",
                 "alpha_max", "kappa", "lambda_disadv", "lambda_adv",
                 "eta_range", "sigma_patch", "kappa_f", "theta_f",
                 "d_nbr", "M"):
        assert gone not in names, f"EnvConfig regrew v4 field {gone!r}"
