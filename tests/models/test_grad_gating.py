"""Pkg-04 spec 04 acceptance tests: belief gradient gating (防线 5)."""
import pytest
import torch

from hyper_mve.configs import V4Config
from hyper_mve.models import HyperMuZeroModel
from hyper_mve.models.grad_gating import BeliefGradGating


# ====== BeliefGradGating 单元 ======

def test_grad_gating_detach_before_warmup():
    gating = BeliefGradGating(num_warmup_steps=5000)
    c_hat = torch.tensor([0.5], requires_grad=True)
    z_hat = torch.softmax(torch.randn(1, 3, 2, requires_grad=True), dim=-1)
    c_out, z_out = gating.apply(c_hat, z_hat, step=1000)
    assert c_out.requires_grad is False
    assert z_out.requires_grad is False


def test_grad_gating_passthrough_after_warmup():
    gating = BeliefGradGating(num_warmup_steps=5000)
    c_hat = torch.tensor([0.5], requires_grad=True)
    z_hat = torch.softmax(torch.randn(1, 3, 2), dim=-1).requires_grad_(True)
    c_out, z_out = gating.apply(c_hat, z_hat, step=10000)
    assert c_out.requires_grad is True
    assert z_out.requires_grad is True


def test_grad_gating_boundary_step_eq_warmup():
    gating = BeliefGradGating(num_warmup_steps=5000)
    c_hat = torch.tensor([0.5], requires_grad=True)
    z_hat = torch.softmax(torch.randn(1, 3, 2), dim=-1).requires_grad_(True)
    c_out, z_out = gating.apply(c_hat, z_hat, step=5000)
    assert c_out.requires_grad is True


def test_apply_ctx_detaches_only_belief_slice():
    gating = BeliefGradGating(num_warmup_steps=5000)
    ctx = torch.randn(2, 80, requires_grad=True)
    out = gating.apply_ctx(ctx, step=1000, belief_slice=(48, 80))
    (out[..., :48].sum() + out[..., 48:].sum()).backward()
    # c_ctx + role 段有梯度, belief 段被 detach (grad 仅来自前 48 维)
    assert ctx.grad is not None
    assert ctx.grad[..., :48].abs().sum().item() > 0
    assert ctx.grad[..., 48:80].abs().sum().item() == 0.0


def test_apply_ctx_passthrough_after_warmup():
    gating = BeliefGradGating(num_warmup_steps=5000)
    ctx = torch.randn(2, 80, requires_grad=True)
    out = gating.apply_ctx(ctx, step=10000, belief_slice=(48, 80))
    out[..., 48:].sum().backward()
    assert ctx.grad[..., 48:80].abs().sum().item() > 0


# ====== HyperMuZeroModel 集成 ======

@pytest.fixture
def cfg_medium():
    return V4Config.from_preset("medium")


@pytest.fixture
def model(cfg_medium):
    return HyperMuZeroModel(cfg_medium)


def test_model_has_grad_gating(model):
    assert hasattr(model, "grad_gating")
    assert isinstance(model.grad_gating, BeliefGradGating)


def test_grad_gating_num_warmup_from_cfg(cfg_medium, model):
    assert model.grad_gating.num_warmup_steps == cfg_medium.train.belief_grad_gating_steps
    assert model.grad_gating.num_warmup_steps == 5000


def test_update_step_changes_internal_step(model):
    model.update_step(0)
    assert model._step == 0
    model.update_step(1000)
    assert model._step == 1000
    model.update_step(10000)
    assert model._step == 10000


def test_pre_5k_belief_detached(model, cfg_medium):
    model.update_step(1000)
    B = 2
    N = cfg_medium.env.N
    obs_dim = model.rep_net.obs_dim

    obs = torch.randn(B, N, obs_dim)
    s = model.encode(obs)
    model.set_context_objective(torch.full((B,), 0.5))

    c_hat = torch.rand(B, requires_grad=True)
    z_hat = torch.softmax(torch.randn(B, N - 1, 2), dim=-1).requires_grad_(True)

    for p in model.belief_net.parameters():
        p.grad = None

    model.set_context_subjective(0, torch.rand(B, 4), (c_hat, z_hat))

    action = torch.zeros(B, N * cfg_medium.env.A)
    action[:, 0] = 1.0
    r = model.predict_reward(s, action)
    (r ** 2).sum().backward()

    for name, p in model.belief_net.named_parameters():
        if p.grad is not None:
            assert p.grad.abs().sum().item() == 0.0, (
                f"BeliefNet param {name} grad nonzero but step=1000 should be gated."
            )


def test_post_5k_belief_grad_flow(model, cfg_medium):
    model.update_step(10000)
    B = 2
    N = cfg_medium.env.N
    obs_dim = model.rep_net.obs_dim

    obs = torch.randn(B, N, obs_dim)
    prev_hidden = model.belief_net.init_hidden(B, N)
    _, c_hat, z_hat = model.belief_net.step(obs, prev_hidden)
    c_hat_0 = c_hat[:, 0]
    z_hat_0 = z_hat[:, 0]

    for p in model.belief_net.parameters():
        p.grad = None

    model.set_context_objective(torch.full((B,), 0.5))
    model.set_context_subjective(0, torch.rand(B, 4), (c_hat_0, z_hat_0))

    s = model.encode(obs)
    action = torch.zeros(B, N * cfg_medium.env.A)
    action[:, 0] = 1.0
    r = model.predict_reward(s, action)
    (r ** 2).sum().backward()

    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.belief_net.parameters()
    )
    assert has_grad, "step=10000 时 BeliefNet 应有梯度 (grad gating 已关闭)"


def test_pre_5k_belief_encoder_also_detached(model, cfg_medium):
    model.update_step(1000)
    B = 2
    N = cfg_medium.env.N
    obs_dim = model.rep_net.obs_dim
    obs = torch.randn(B, N, obs_dim)

    c_hat = torch.rand(B, requires_grad=True)
    z_hat = torch.softmax(torch.randn(B, N - 1, 2), dim=-1).requires_grad_(True)

    for p in model.tri_context_encoder.belief_encoder.parameters():
        p.grad = None

    model.set_context_objective(torch.full((B,), 0.5))
    model.set_context_subjective(0, torch.rand(B, 4), (c_hat, z_hat))

    s = model.encode(obs)
    action = torch.zeros(B, N * cfg_medium.env.A)
    action[:, 0] = 1.0
    r = model.predict_reward(s, action)
    (r ** 2).sum().backward()

    for name, p in model.tri_context_encoder.belief_encoder.named_parameters():
        if p.grad is not None:
            assert p.grad.abs().sum().item() == 0.0, (
                f"BeliefEncoder param {name} grad nonzero but step=1000 should be gated."
            )


def test_post_5k_belief_encoder_grad_flow(model, cfg_medium):
    model.update_step(10000)
    B = 2
    N = cfg_medium.env.N
    obs_dim = model.rep_net.obs_dim
    obs = torch.randn(B, N, obs_dim)

    c_hat = torch.rand(B, requires_grad=True)
    z_hat = torch.softmax(torch.randn(B, N - 1, 2), dim=-1).requires_grad_(True)

    for p in model.tri_context_encoder.belief_encoder.parameters():
        p.grad = None

    model.set_context_objective(torch.full((B,), 0.5))
    model.set_context_subjective(0, torch.rand(B, 4), (c_hat, z_hat))

    s = model.encode(obs)
    action = torch.zeros(B, N * cfg_medium.env.A)
    action[:, 0] = 1.0
    r = model.predict_reward(s, action)
    (r ** 2).sum().backward()

    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.tri_context_encoder.belief_encoder.parameters()
    )
    assert has_grad, "step=10000 时 BeliefEncoder 应有梯度"


def test_l_belief_path_not_detached(model, cfg_medium):
    model.update_step(1000)
    B = 2
    N = cfg_medium.env.N
    obs_dim = model.rep_net.obs_dim
    obs = torch.randn(B, N, obs_dim)

    for p in model.belief_net.parameters():
        p.grad = None

    prev_hidden = model.belief_net.init_hidden(B, N)
    _, c_hat, z_hat = model.belief_net.step(obs, prev_hidden)

    l_belief = (c_hat ** 2).sum() + (z_hat ** 2).sum()
    l_belief.backward()

    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.belief_net.parameters()
    )
    assert has_grad, "L_belief 路径应始终反向到 BeliefNet, 不受 grad gating 影响"


def test_c_ctx_role_path_grad_flows_during_gating(model, cfg_medium):
    model.update_step(1000)
    B = 2
    N = cfg_medium.env.N
    obs_dim = model.rep_net.obs_dim
    obs = torch.randn(B, N, obs_dim)

    c_hat = torch.rand(B, requires_grad=True)
    z_hat = torch.softmax(torch.randn(B, N - 1, 2), dim=-1).requires_grad_(True)

    for p in model.tri_context_encoder.c_encoder.parameters():
        p.grad = None
    for p in model.tri_context_encoder.role_encoder.parameters():
        p.grad = None

    model.set_context_objective(torch.full((B,), 0.5))
    model.set_context_subjective(0, torch.rand(B, 4), (c_hat, z_hat))

    s = model.encode(obs)
    action = torch.zeros(B, N * cfg_medium.env.A)
    action[:, 0] = 1.0
    r = model.predict_reward(s, action)
    (r ** 2).sum().backward()

    for sub_name, sub in [("CEncoder", model.tri_context_encoder.c_encoder),
                          ("RoleEncoder", model.tri_context_encoder.role_encoder)]:
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in sub.parameters()
        )
        assert has_grad, f"{sub_name} 参数应有梯度 (gating active 时 c_ctx/role 路径未 detach)"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
