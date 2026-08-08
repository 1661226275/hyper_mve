"""Invariants the VoI probe's offline recompute depends on.

``scripts/probes/regime_voi_probe.py`` rolls out its threshold grid **once** and
then evaluates every candidate reward coupling in closed form. That shortcut is
what made the v6 design surface affordable (minutes instead of hours), and it is
valid only because, for a fixed threshold pair, the trajectory is identical
across all five regimes:

* ``W(g)`` enters the reward and nothing else — never the dynamics; and
* the probe policy never reads the observation's row block, which is the only
  place the regime is visible to an agent.

If either stops holding, the probe silently reports numbers for reward
weightings it never actually rolled out. These tests fail loudly instead.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "probes"))

from hyper_mve.utils.configs import V4Config                             # noqa: E402
from hyper_mve.envs.adapters.pettingzoo_wrapper import (                 # noqa: E402
    RelationCommonsPettingZooEnv,
)
from hyper_mve.utils.schemas import RelationObservationLayout            # noqa: E402

from regime_voi_probe import threshold_action                            # noqa: E402


@pytest.fixture(scope="module")
def rel_recip():
    return V4Config.from_preset("rel_recip").env


def test_probe_policy_ignores_the_row_block(rel_recip):
    """Perturbing only the own-row block must not change the chosen action."""
    env = RelationCommonsPettingZooEnv(rel_recip, oracle_mode=False,
                                       eval_info_mode=False)
    obs, _ = env.reset(seed=7, options={"g": 0})
    N, K = rel_recip.N, rel_recip.K
    start, end = RelationObservationLayout.block_offset("row", N, K)

    rng = np.random.default_rng(0)
    for _ in range(40):
        acts = {}
        for i, a in enumerate(env.possible_agents):
            o = np.asarray(obs[a], dtype=np.float32)
            base = threshold_action(o, i, N, K, 0.3)
            for row_val in (-1.0, 0.0, +1.0, 0.37):
                perturbed = o.copy()
                perturbed[start:end] = row_val
                assert threshold_action(perturbed, i, N, K, 0.3) == base, (
                    "probe policy reacted to the row block; the VoI probe's "
                    "offline recompute is invalid"
                )
            acts[a] = base
        obs, _, term, trunc, _ = env.step(acts)
        if any(term.values()) or any(trunc.values()):
            obs, _ = env.reset(seed=int(rng.integers(1 << 30)), options={"g": 0})
    env.close()


def test_trajectory_is_identical_across_regimes(rel_recip):
    """Same thresholds ⇒ same physical rollout in every regime.

    Only the reward differs, which is what lets one rollout score all five.
    """
    def rollout(g):
        # eval_info_mode exposes resource_state; it is read-only diagnostics and
        # does not reach the policy, which sees only obs.
        env = RelationCommonsPettingZooEnv(rel_recip, oracle_mode=False,
                                           eval_info_mode=True)
        obs, _ = env.reset(seed=123, options={"g": g})
        stocks, positions, done = [], [], False
        while not done:
            acts = {a: int(threshold_action(np.asarray(obs[a]), i,
                                            rel_recip.N, rel_recip.K, (0.2, 0.4)[i]))
                    for i, a in enumerate(env.possible_agents)}
            obs, _, term, trunc, info = env.step(acts)
            stocks.append(np.array(info[env.possible_agents[0]]["resource_state"]))
            positions.append(np.concatenate(
                [np.asarray(obs[a])[:2] for a in env.possible_agents]))
            done = bool(any(term.values()) or any(trunc.values()))
        env.close()
        return np.array(stocks), np.array(positions)

    ref_stocks, ref_pos = rollout(0)
    for g in (1, 2, 3, 4):
        s, p = rollout(g)
        np.testing.assert_allclose(s, ref_stocks, atol=1e-6,
                                   err_msg=f"g{g} stocks diverged from g0")
        np.testing.assert_allclose(p, ref_pos, atol=1e-6,
                                   err_msg=f"g{g} positions diverged from g0")


def test_probe_policy_can_actually_abstain(rel_recip):
    """Restraint must be expressible, or every VoI measurement is understated.

    An earlier version fell back to targeting *some* cell and
    ``_step_toward(0,0) -> HARVEST``, so it always harvested and the commons
    dilemma was invisible to it.
    """
    from hyper_mve.envs.relation_commons.reference_policies import NOOP

    N, K = rel_recip.N, rel_recip.K
    obs = np.zeros(RelationObservationLayout.total_dim(N, K), dtype=np.float32)
    r_start, _ = RelationObservationLayout.block_offset("resource", N, K)
    # Standing on cell 0, which holds a little stock; everything below threshold.
    cells = np.zeros((K, 3), dtype=np.float32)
    cells[:, 2] = 0.05
    obs[r_start:r_start + K * 3] = cells.reshape(-1)

    assert threshold_action(obs, 0, N, K, 0.30) == NOOP
    assert threshold_action(obs, 0, N, K, 0.00) != NOOP     # harvests when free to
