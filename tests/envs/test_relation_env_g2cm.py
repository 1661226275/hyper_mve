"""Env-level reward signs for the ``g2cm`` family.

Mirrors ``test_relation_env.py::test_bystander_reward_sign`` for the new family,
under both couplings, and pins ``asym_exploit_mild`` explicitly — it is the
regime that did not exist before, so nothing else covers it.
"""
from __future__ import annotations

import numpy as np
import pytest

from hyper_mve.utils.configs.env_config import EnvConfig
from hyper_mve.envs.relation_commons import RelationCommonsEnv, make_relation_commons

NOOP, HARVEST = 0, 5


def _cfg(**kw) -> EnvConfig:
    base = dict(N=2, L=8, K=8, T_max=20, relation_family="g2cm")
    base.update(kw)
    return EnvConfig(**base)


def _isolate(env: RelationCommonsEnv) -> None:
    """agent 0 on a full cell, agent 1 idling far away."""
    target = env._state.resource_positions[0]
    env._state.agent_positions[0] = np.asarray(target, dtype=np.int32)
    away = [0, 0] if not np.array_equal(target, [0, 0]) else [7, 7]
    env._state.agent_positions[1] = np.asarray(away, dtype=np.int32)


@pytest.mark.parametrize(
    "g, name, bystander",
    [
        (0, "mutual_coop", +0.5),        # (0 + 1*u0)/2
        (1, "asym_exploit", +0.5),       # agent 1's own row is +lam
        (2, "asym_exploited", -0.5),     # agent 1's own row is -lam
        (3, "asym_exploit_mild", 0.0),   # agent 1's own row is 0
        (4, "neutral", 0.0),
    ],
)
def test_bystander_reward_sign_own_row(g, name, bystander):
    """Under ``own_row`` each agent weights by its OWN row, so the idle agent's
    reward is decided by ``w_10``."""
    from hyper_mve.utils.schemas import get_regime_family

    cfg = _cfg(reward_coupling="own_row")
    assert get_regime_family(cfg).names()[g] == name
    env = make_relation_commons(cfg, seed=8)
    env.reset(options={"g": g})
    _isolate(env)
    _, reward, *_ = env.step(np.array([HARVEST, NOOP]))
    assert reward[1] == pytest.approx(bystander)


def test_asym_exploit_mild_under_reciprocal_is_not_zero_sum():
    """The regime that replaced ``mutual_comp``, at env level.

    Under ``reciprocal`` the weights swap: agent 0 weights u1 by ``w_10 = 0`` and
    agent 1 weights u0 by ``w_01 = -lam``. So agent 0 keeps its full harvest,
    agent 1 is hurt by it, and the PAIR still sums to ``(u0+u1)/2`` — unlike
    ``mutual_comp``, where the pair cancelled to zero for any harvests.
    """
    env = make_relation_commons(_cfg(reward_coupling="reciprocal"), seed=8)
    env.reset(options={"g": 3})
    _isolate(env)
    _, reward, *_ = env.step(np.array([HARVEST, NOOP]))
    u0 = 1.0                                   # eta=1 cap, full stock
    assert reward[0] == pytest.approx(u0)      # (u0 + 0*u1)/1
    assert reward[1] == pytest.approx(-0.5)    # (0 - 1*u0)/2
    assert float(reward.sum()) == pytest.approx(u0 / 2.0)


def test_g2cm_holdout_sampling_stays_inside_train_regime_ids():
    env = make_relation_commons(_cfg(train_regime_ids=(0, 1, 4)), seed=3)
    seen = {env.reset(seed=s)[1]["g_true"] for s in range(40)}
    assert seen <= {0, 1, 4}


def test_g2cm_pinning_rejects_out_of_range_regime():
    env = make_relation_commons(_cfg(), seed=0)
    env.reset(options={"g": 4})                        # highest valid id
    with pytest.raises(AssertionError, match="out of"):
        env.reset(options={"g": 5})
