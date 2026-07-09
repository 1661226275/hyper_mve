"""Pkg-04 spec 04 acceptance tests: belief gradient gating (defence 5, v5 form)."""
import pytest
import torch

from hyper_mve.configs import V4Config
from hyper_mve.models import HyperMuZeroModel
from hyper_mve.models.grad_gating import BeliefGradGating

_G = 5


# ====== BeliefGradGating unit ======

def test_grad_gating_detach_before_warmup():
    gating = BeliefGradGating(num_warmup_steps=5000)
    g_hat = torch.softmax(torch.randn(1, _G, requires_grad=True), dim=-1)
    g_out = gating.apply(g_hat, step=1000)
    assert g_out.requires_grad is False


def test_grad_gating_passthrough_after_warmup():
    gating = BeliefGradGating(num_warmup_steps=5000)
    g_hat = torch.softmax(torch.randn(1, _G), dim=-1).requires_grad_(True)
    g_out = gating.apply(g_hat, step=10000)
    assert g_out.requires_grad is True


def test_grad_gating_boundary_step_eq_warmup():
    gating = BeliefGradGating(num_warmup_steps=5000)
    g_hat = torch.softmax(torch.randn(1, _G), dim=-1).requires_grad_(True)
    g_out = gating.apply(g_hat, step=5000)
    assert g_out.requires_grad is True


def test_apply_ctx_detaches_only_belief_slice():
    """v5 ctx_aug = [role (0:32) | belief (32:64)]."""
    gating = BeliefGradGating(num_warmup_steps=5000)
    ctx = torch.randn(2, 64, requires_grad=True)
    out = gating.apply_ctx(ctx, step=1000, belief_slice=(32, 64))
    (out[..., :32].sum() + out[..., 32:].sum()).backward()
    assert ctx.grad is not None
    assert ctx.grad[..., :32].abs().sum().item() > 0
    assert ctx.grad[..., 32:64].abs().sum().item() == 0.0


def test_apply_ctx_passthrough_after_warmup():
    gating = BeliefGradGating(num_warmup_steps=5000)
    ctx = torch.randn(2, 64, requires_grad=True)
    out = gating.apply_ctx(ctx, step=10000, belief_slice=(32, 64))
    out[..., 32:].sum().backward()
    assert ctx.grad[..., 32:64].abs().sum().item() > 0


# ====== HyperMuZeroModel integration (v5 6-API) ======

@pytest.fixture
def cfg_duo():
    return V4Config.from_preset("rel_duo")


@pytest.fixture
def model(cfg_duo):
    return HyperMuZeroModel(cfg_duo)


def _subjective_inputs(B, N, requires_grad=True):
    row = (torch.rand(B, N - 1) * 2 - 1).requires_grad_(requires_grad)
    g_hat = torch.softmax(torch.randn(B, _G), dim=-1)
    if requires_grad:
        g_hat = g_hat.clone().requires_grad_(True)
    return row, g_hat


def test_model_has_grad_gating(model):
    assert isinstance(model.grad_gating, BeliefGradGating)


def test_grad_gating_num_warmup_from_cfg(cfg_duo, model):
    assert model.grad_gating.num_warmup_steps == cfg_duo.train.belief_grad_gating_steps
    assert model.grad_gating.num_warmup_steps == 5000


def test_update_step_changes_internal_step(model):
    for step in (0, 1000, 10000):
        model.update_step(step)
        assert model._step == step


def _reward_backward(model, cfg, row, g_hat):
    B = row.shape[0]
    N = cfg.env.N
    obs = torch.randn(B, N, model.rep_net.obs_dim)
    s = model.encode(obs)
    model.set_context_subjective(0, row, g_hat)
    action = torch.zeros(B, N * cfg.env.A)
    action[:, 0] = 1.0
    r = model.predict_reward(s, action)
    (r ** 2).sum().backward()


def test_pre_5k_belief_detached(model, cfg_duo):
    """Main-loss backprop never reaches BeliefNet during warmup."""
    model.update_step(1000)
    B, N = 2, cfg_duo.env.N
    obs = torch.randn(B, N, model.rep_net.obs_dim)
    prev_hidden = model.belief_net.init_hidden(B, N)
    _, g_hat = model.belief_net.step(obs, prev_hidden)

    for p in model.belief_net.parameters():
        p.grad = None
    _reward_backward(model, cfg_duo, torch.rand(B, N - 1), g_hat[:, 0])

    for name, p in model.belief_net.named_parameters():
        if p.grad is not None:
            assert p.grad.abs().sum().item() == 0.0, (
                f"BeliefNet param {name} grad nonzero but step=1000 should be gated."
            )


def test_post_5k_belief_grad_flow(model, cfg_duo):
    model.update_step(10000)
    B, N = 2, cfg_duo.env.N
    obs = torch.randn(B, N, model.rep_net.obs_dim)
    prev_hidden = model.belief_net.init_hidden(B, N)
    _, g_hat = model.belief_net.step(obs, prev_hidden)

    for p in model.belief_net.parameters():
        p.grad = None
    _reward_backward(model, cfg_duo, torch.rand(B, N - 1), g_hat[:, 0])

    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.belief_net.parameters()
    )
    assert has_grad, "step=10000: BeliefNet should receive main-loss gradient"


def test_pre_5k_belief_encoder_also_detached(model, cfg_duo):
    model.update_step(1000)
    B, N = 2, cfg_duo.env.N
    row, g_hat = _subjective_inputs(B, N)
    for p in model.tri_context_encoder.belief_encoder.parameters():
        p.grad = None
    _reward_backward(model, cfg_duo, row, g_hat)
    for name, p in model.tri_context_encoder.belief_encoder.named_parameters():
        if p.grad is not None:
            assert p.grad.abs().sum().item() == 0.0, (
                f"BeliefEncoder param {name} grad nonzero but step=1000 should be gated."
            )


def test_post_5k_belief_encoder_grad_flow(model, cfg_duo):
    model.update_step(10000)
    B, N = 2, cfg_duo.env.N
    row, g_hat = _subjective_inputs(B, N)
    for p in model.tri_context_encoder.belief_encoder.parameters():
        p.grad = None
    _reward_backward(model, cfg_duo, row, g_hat)
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.tri_context_encoder.belief_encoder.parameters()
    )
    assert has_grad


def test_l_belief_path_not_detached(model, cfg_duo):
    """The independent L_belief path always reaches BeliefNet."""
    model.update_step(1000)
    B, N = 2, cfg_duo.env.N
    obs = torch.randn(B, N, model.rep_net.obs_dim)
    for p in model.belief_net.parameters():
        p.grad = None
    prev_hidden = model.belief_net.init_hidden(B, N)
    _, g_hat = model.belief_net.step(obs, prev_hidden)
    (g_hat ** 2).sum().backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.belief_net.parameters()
    )
    assert has_grad


def test_role_path_grad_flows_during_gating(model, cfg_duo):
    """Gating detaches only the belief slice — the role path keeps training."""
    model.update_step(1000)
    B, N = 2, cfg_duo.env.N
    row, g_hat = _subjective_inputs(B, N)
    for p in model.tri_context_encoder.role_encoder.parameters():
        p.grad = None
    _reward_backward(model, cfg_duo, row, g_hat)
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.tri_context_encoder.role_encoder.parameters()
    )
    assert has_grad, "RoleEncoder must keep gradient while gating is active"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
