"""Pkg-04 spec 02 acceptance tests: HyperMuZeroModel (v5 6-API, Pkg-09 amendment)."""
from dataclasses import replace

import pytest
import torch

from hyper_mve.configs import V4Config
from hyper_mve.models import HyperMuZeroModel

_G = 5


@pytest.fixture
def cfg_duo():
    return V4Config.from_preset("rel_duo")


@pytest.fixture
def model(cfg_duo):
    return HyperMuZeroModel(cfg_duo)


def _row(B, N):
    return torch.rand(B, N - 1) * 2 - 1


def _g_hat(B):
    return torch.softmax(torch.randn(B, _G), dim=-1)


# ====== v5 API surface ======

def test_model_has_6_public_apis(model):
    for api in ("update_step", "set_context_subjective",
                "encode", "transition", "predict_reward", "predict"):
        assert hasattr(model, api), f"Missing API: {api}"
        assert callable(getattr(model, api))


def test_set_context_objective_deleted(model):
    """v5 amendment: the objective context API no longer exists."""
    assert not hasattr(model, "set_context_objective")
    assert not hasattr(model, "install_objective_theta")
    assert not hasattr(model, "current_objective_theta")


def test_ctx_aug_dim_assertion(cfg_duo):
    assert cfg_duo.model.d_ctx_aug == 64
    bad = replace(cfg_duo, model=replace(cfg_duo.model, d_belief=48))
    with pytest.raises(AssertionError, match="d_ctx_aug"):
        HyperMuZeroModel(bad)


# ====== call-order contract ======

def test_transition_needs_no_context(model, cfg_duo):
    """v5: transition is a plain shared module — callable immediately."""
    s = torch.randn(1, cfg_duo.model.latent_dim)
    action = torch.zeros(1, cfg_duo.env.N * cfg_duo.env.A)
    s_next = model.transition(s, action)
    assert s_next.shape == s.shape


def test_predict_reward_before_subjective_raises(model, cfg_duo):
    s = torch.randn(1, cfg_duo.model.latent_dim)
    action = torch.zeros(1, cfg_duo.env.N * cfg_duo.env.A)
    with pytest.raises(AssertionError, match="predict_reward.*before set_context_subjective"):
        model.predict_reward(s, action)


def test_predict_before_subjective_raises(model, cfg_duo):
    s = torch.randn(1, cfg_duo.model.latent_dim)
    with pytest.raises(AssertionError, match="predict.*before set_context_subjective"):
        model.predict(s)


# ====== conditioning semantics ======

def test_subjective_replaces_per_agent(model, cfg_duo):
    N = cfg_duo.env.N
    row, g = _row(1, N), _g_hat(1)
    model.set_context_subjective(0, row, g)
    theta_rew_0 = model._theta_rew.clone()
    model.set_context_subjective(1, row, g)
    theta_rew_1 = model._theta_rew.clone()
    assert not torch.allclose(theta_rew_0, theta_rew_1)


def test_row_differentiates_theta(model, cfg_duo):
    """The core v5 mechanism: different own rows ⇒ different generated θ."""
    N = cfg_duo.env.N
    g = _g_hat(1)
    model.set_context_subjective(0, torch.tensor([[1.0]]), g)
    theta_ally = model._theta_rew.clone()
    model.set_context_subjective(0, torch.tensor([[-1.0]]), g)
    theta_rival = model._theta_rew.clone()
    assert not torch.allclose(theta_ally, theta_rival)


def test_belief_differentiates_theta(model, cfg_duo):
    N = cfg_duo.env.N
    row = _row(1, N)
    model.set_context_subjective(0, row, torch.eye(_G)[0].unsqueeze(0))
    theta_a = model._theta_rew.clone()
    model.set_context_subjective(0, row, torch.eye(_G)[1].unsqueeze(0))
    theta_b = model._theta_rew.clone()
    assert not torch.allclose(theta_a, theta_b)


# ====== update_step (Q4) ======

def test_update_step(model):
    model.update_step(1000)
    assert model._step == 1000


def test_default_step_is_zero(model):
    """update_step not called ⇒ _step=0 ⇒ belief grad gating always active."""
    assert model._step == 0


# ====== stateful API contract ======

def test_predict_uses_latest_subjective_agent(model, cfg_duo):
    N = cfg_duo.env.N
    row, g = _row(1, N), _g_hat(1)
    s = torch.randn(1, cfg_duo.model.latent_dim)
    action = torch.zeros(1, N * cfg_duo.env.A)
    action[0, 0] = 1.0

    model.set_context_subjective(0, row, g)
    r_A_first = model.predict_reward(s, action).clone()
    pi_A_first = model.predict(s)[0].clone()

    model.set_context_subjective(1, row, g)
    r_B = model.predict_reward(s, action).clone()
    pi_B = model.predict(s)[0]

    model.set_context_subjective(0, row, g)
    r_A_second = model.predict_reward(s, action).clone()

    assert not torch.allclose(r_A_first, r_B, atol=1e-6)
    assert not torch.allclose(pi_A_first, pi_B, atol=1e-6)
    assert torch.allclose(r_A_first, r_A_second, atol=1e-6)


# ====== Self-Info strictness (C11, v5 form) ======

def test_row_shape_assertion(model):
    """A full-W (or otherwise wrong-width) row is rejected — info-leak guard."""
    with pytest.raises(AssertionError, match="row_i"):
        model.set_context_subjective(0, torch.rand(1, 2), _g_hat(1))


def test_belief_shape_assertion(model, cfg_duo):
    with pytest.raises(AssertionError, match="belief"):
        model.set_context_subjective(0, _row(1, cfg_duo.env.N), torch.rand(1, 3))


# ====== θ-cache fast path ======

def test_install_subjective_theta_equivalence(model, cfg_duo):
    """install_subjective_theta(tiled θ) == set_context_subjective on tiled ctx."""
    model.eval()
    N = cfg_duo.env.N
    B, reps = 2, 3
    row, g = _row(B, N), _g_hat(B)
    s = torch.randn(B * reps, cfg_duo.model.latent_dim)
    action = torch.zeros(B * reps, N * cfg_duo.env.A)
    action[:, 0] = 1.0

    with torch.no_grad():
        # slow path: set context at the expanded batch
        model.set_context_subjective(
            0, row.repeat_interleave(reps, 0), g.repeat_interleave(reps, 0))
        r_slow = model.predict_reward(s, action).clone()
        pi_slow, v_slow = model.predict(s)

        # fast path: generate θ at base batch, tile, install
        model.set_context_subjective(0, row, g)
        tr, tp = model.current_subjective_thetas()
        model.install_subjective_theta(
            0, tr.repeat_interleave(reps, 0), tp.repeat_interleave(reps, 0))
        r_fast = model.predict_reward(s, action)
        pi_fast, v_fast = model.predict(s)

    assert torch.allclose(r_slow, r_fast, atol=1e-6)
    assert torch.allclose(pi_slow, pi_fast, atol=1e-6)
    assert torch.allclose(v_slow, v_fast, atol=1e-6)


# ====== forward end-to-end ======

def _e2e(cfg):
    m = HyperMuZeroModel(cfg)
    B, N = 2, cfg.env.N
    obs = torch.randn(B, N, m.rep_net.obs_dim)
    s = m.encode(obs)
    m.update_step(0)
    action = torch.zeros(B, N * cfg.env.A)
    action[:, 0] = 1.0
    s_next = m.transition(s, action)
    m.set_context_subjective(0, _row(B, N), _g_hat(B))
    r = m.predict_reward(s, action)
    pi, v = m.predict(s)
    for t in (s_next, r, pi, v):
        assert not torch.isnan(t).any()
    return m


def test_end_to_end_forward_no_nan(cfg_duo):
    _e2e(cfg_duo)


# ====== gen_scope variants (film_head default in rel_duo; full / base_gen / lora) ======

def test_rel_duo_default_is_film_head(cfg_duo):
    m = HyperMuZeroModel(cfg_duo)
    # film_head generated counts are N-independent for rew/pred (the trunk is SGD).
    assert m.reward_head.generated_param_count == 641
    assert m.prediction_net.generated_param_count == 1415
    assert m.hyper_net.rew_param_count == 641
    assert m.hyper_net.pred_param_count == 1415
    assert m.hyper_net.hyper_rew.output_groups == [512, 129]
    assert m.hyper_net.hyper_pred.output_groups == [512, 903]
    # v5: the transition net is plain SGD — no generated θ, no hyper_trans.
    assert not hasattr(m.hyper_net, "hyper_trans")
    from hyper_mve.models import TransitionNet
    assert isinstance(m.state_trans_net, TransitionNet)


@pytest.mark.parametrize("scope", ["full", "base_gen"])
def test_other_gen_scopes_end_to_end(cfg_duo, scope):
    cfg = replace(cfg_duo, model=replace(cfg_duo.model, hyper_gen_scope=scope))
    _e2e(cfg)


def test_output_layer_lora_end_to_end(cfg_duo):
    cfg = replace(cfg_duo, model=replace(cfg_duo.model, hyper_output_rank=32))
    m = _e2e(cfg)
    for sub in (m.hyper_net.hyper_rew, m.hyper_net.hyper_pred):
        assert sub.output_rank == 32
        assert hasattr(sub, "output_A") and hasattr(sub, "output_B")


def test_lora_fc2_end_to_end(cfg_duo):
    cfg = replace(cfg_duo, model=replace(
        cfg_duo.model, hyper_gen_scope="lora_fc2", lora_fc2_rank=8))
    m = _e2e(cfg)
    assert m.reward_head.generated_param_count == 2689
    assert m.prediction_net.generated_param_count == 3463
    assert hasattr(m.reward_head, "fc2")   # shared SGD base for the rank-r delta


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
