"""RelationDuoEnvModel exactness gate (phase-5).

The RelationCommons step is fully deterministic given the state (p=0 regime
chain draws nothing), so the torch port must reproduce the REAL env's next
observation of agent 1 exactly (float32 tolerance) across random transitions,
and the reward channel must match per reward_mode:
  * true_W      — model r1 == the real env's relational reward for agent_1
  * own_harvest — model r1 == u_1 − ε·moved_1 (harvests are public info)
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from hyper_mve.utils.configs import V4Config  # noqa: E402
from hyper_mve.envs.adapters.pettingzoo_wrapper import (  # noqa: E402
    RelationCommonsPettingZooEnv,
)
from hyper_mve.comparison.mbom import RelationDuoEnvModel  # noqa: E402


def _run_transitions(reward_mode: str, n_target: int = 200,
                     preset: str = "rel_duo", inject_opponent_row: bool = False) -> int:
    cfg = V4Config.from_preset(preset)
    env = RelationCommonsPettingZooEnv(cfg.env, oracle_mode=False,
                                       eval_info_mode=False)
    model = RelationDuoEnvModel(cfg.env, torch.device("cpu"), reward_mode)
    rng = np.random.default_rng(0)
    checked = 0
    ep = 0
    while checked < n_target:
        g = int(rng.integers(0, 5))
        obs, _ = env.reset(seed=1000 + ep, options={"g": g})
        ep += 1
        if inject_opponent_row:
            # Privileged: the true w_01, which agent 1 never observes. Only
            # meaningful under a coupling that puts it in agent 1's reward.
            from hyper_mve.utils.schemas.relation import get_regime_family
            W = get_regime_family(cfg.env).regimes[g].w_array()
            model.set_opponent_row(torch.tensor([float(W[0, 1])]))
        done = False
        while not done and checked < n_target:
            a0, a1 = int(rng.integers(0, 6)), int(rng.integers(0, 6))
            state1 = torch.from_numpy(
                np.asarray(obs["agent_1"], dtype=np.float32)[None]
            )
            model.reset()
            state_, rewards, m_done = model.step(
                state1,
                [torch.tensor([[a0]]), torch.tensor([[a1]])],
            )
            obs, rew, term, trunc, info = env.step({"agent_0": a0, "agent_1": a1})
            done = bool(any(term.values()) or any(trunc.values()))

            real_next1 = np.asarray(obs["agent_1"], dtype=np.float32)
            got_next1 = state_.numpy()[0]
            assert np.allclose(got_next1, real_next1, atol=1e-4), (
                reward_mode, checked,
                np.max(np.abs(got_next1 - real_next1)),
                np.argmax(np.abs(got_next1 - real_next1)),
            )
            u = info["agent_1"]["harvests"]
            moved1 = 1 <= a1 <= 4
            if reward_mode == "true_W":
                expected_r1 = float(rew["agent_1"])
            else:
                expected_r1 = float(u[1]) - cfg.env.epsilon_move * float(moved1)
            got_r1 = float(rewards[1].reshape(-1)[0])
            assert got_r1 == pytest.approx(expected_r1, abs=1e-5), (
                reward_mode, checked, got_r1, expected_r1,
            )
            assert bool(m_done.reshape(-1)[0]) == done
            checked += 1
    env.close()
    return checked


@pytest.mark.parametrize("reward_mode", ["own_harvest", "true_W"])
def test_env_model_matches_real_env_on_random_transitions(reward_mode):
    assert _run_transitions(reward_mode) == 200


@pytest.mark.parametrize("reward_mode", ["own_harvest", "true_W"])
def test_env_model_tracks_the_v6_physics(reward_mode):
    """The same gate on rel_recip — logistic regrowth and reciprocal reward.

    This is the test that stops MBOM silently planning against v5 physics after
    an env change. ``true_W`` needs the true w_01 here, because under reciprocal
    that is agent 1's reward weight and it is not in the observation.
    """
    assert _run_transitions(reward_mode, preset="rel_recip",
                            inject_opponent_row=(reward_mode == "true_W")) == 200


def test_mirror_prior_is_wrong_exactly_on_the_asymmetric_regimes():
    """Without the injection the model uses the mirror prior, which is right in
    g0/g1/g4 and wrong in g2/g3 — and it must report itself as non-privileged.

    A silent mismatch here would score a fair baseline as if it knew the regime.
    """
    from hyper_mve.utils.schemas.relation import get_regime_family

    cfg = V4Config.from_preset("rel_recip")
    model = RelationDuoEnvModel(cfg.env, torch.device("cpu"), "true_W")
    assert not model.uses_privileged_row

    fam = get_regime_family(cfg.env)
    for g in (0, 1, 4):
        W = fam.regimes[g].w_array()
        w_own = torch.tensor([float(W[1, 0])])
        assert float(model._effective_weight(w_own)) == pytest.approx(float(W[0, 1]))
    for g in (2, 3):
        W = fam.regimes[g].w_array()
        w_own = torch.tensor([float(W[1, 0])])
        assert float(model._effective_weight(w_own)) != pytest.approx(float(W[0, 1]))

    model.set_opponent_row(torch.tensor([1.0]))
    assert model.uses_privileged_row


def test_env_model_batched_rollout_shapes():
    """MBOM's _rollout drives batched (B=6^k) states — shape contract."""
    cfg = V4Config.from_preset("rel_duo")
    model = RelationDuoEnvModel(cfg.env, torch.device("cpu"), "own_harvest")
    B = 36
    state = torch.randn(B, model.n_state)
    # keep decode sane: clamp normalized blocks into range
    state[:, 0:4] = torch.rand(B, 4)
    state[:, model._t0] = 0.5
    my = torch.randint(0, 6, (B,))
    op = torch.randint(0, 6, (B, 1))
    state_, rewards, done = model.step(state, [op, my])
    assert state_.shape == (B, model.n_state)
    assert rewards[0].shape == (B, 1) and rewards[1].shape == (B, 1)
    assert done.shape == (B, 1) and done.dtype == torch.bool
