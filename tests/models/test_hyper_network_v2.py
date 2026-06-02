"""Pkg-04 spec 01 acceptance tests: DualHyperNetwork v2 API."""
import pytest
import torch

from hyper_mve.configs import V4Config
from hyper_mve.models import DualHyperNetwork
from hyper_mve.models.hyper_network import HyperNetMLP


@pytest.fixture
def cfg_medium():
    return V4Config.from_preset("medium")


@pytest.fixture
def hyper_net(cfg_medium):
    """构造 DualHyperNetwork (medium config)."""
    return DualHyperNetwork(
        c_ctx_dim=cfg_medium.model.d_c,
        ctx_aug_dim=cfg_medium.model.d_ctx_aug,
        trans_param_count=37_000,
        rew_param_count=4_000,
        pred_param_count=42_000,
        hidden_dims=cfg_medium.model.hyper_hidden_dims,
        rew_hidden_dims=cfg_medium.model.hyper_rew_hidden_dims,
        trans_output_scale_init=cfg_medium.model.trans_output_scale_init,
        rew_output_scale_init=cfg_medium.model.rew_output_scale_init,
        pred_output_scale_init=cfg_medium.model.pred_output_scale_init,
    )


# ====== C1: hyper_trans 仅 c_ctx ======

def test_hyper_trans_only_c_ctx(hyper_net):
    c_ctx = torch.randn(2, 16)
    theta_state = hyper_net.forward_trans(c_ctx)
    assert theta_state.shape == (2, 37_000)


def test_hyper_trans_invariant_under_role_change(hyper_net):
    c_ctx_a = torch.randn(2, 16)
    c_ctx_b = c_ctx_a.clone()
    theta_a = hyper_net.forward_trans(c_ctx_a)
    theta_b = hyper_net.forward_trans(c_ctx_b)
    assert torch.allclose(theta_a, theta_b)


def test_hyper_trans_c_ctx_dim_assertion(hyper_net):
    bad_c_ctx = torch.randn(2, 8)
    with pytest.raises(AssertionError, match="c_ctx last dim"):
        hyper_net.forward_trans(bad_c_ctx)


# ====== C2: hyper_rew/pred 接 80 维 ======

def test_subjective_input_dim_80(hyper_net):
    ctx_aug = torch.randn(2, 80)
    theta_rew, theta_pred = hyper_net.forward_subjective(ctx_aug)
    assert theta_rew.shape == (2, 4_000)
    assert theta_pred.shape == (2, 42_000)


def test_subjective_ctx_aug_dim_assertion(hyper_net):
    bad_ctx = torch.randn(2, 64)
    with pytest.raises(AssertionError, match="ctx_aug last dim"):
        hyper_net.forward_subjective(bad_ctx)


# ====== C5: ctx_aug_dim == 80 ======

def test_ctx_aug_dim_eq_80(cfg_medium):
    assert cfg_medium.model.d_ctx_aug == 80
    assert cfg_medium.model.d_c == 16
    assert cfg_medium.model.d_role == 32
    assert cfg_medium.model.d_belief == 32


# ====== C8: output_scale 三初值 ======

def test_output_scale_inits_trans_rew_pred(hyper_net):
    assert torch.isclose(hyper_net.hyper_trans.output_scale, torch.tensor(0.01))
    assert torch.isclose(hyper_net.hyper_rew.output_scale, torch.tensor(0.1))
    assert torch.isclose(hyper_net.hyper_pred.output_scale, torch.tensor(0.01))


# ====== HyperNetMLP 单元 ======

def test_hypernet_mlp_l2_norm():
    mlp = HyperNetMLP(input_dim=16, output_dim=100, norm_output=True, output_scale_init=0.1)
    x = torch.randn(2, 16)
    out = mlp(x)
    norms = torch.linalg.norm(out, dim=-1)
    expected = torch.full_like(norms, 0.1)
    assert torch.allclose(norms, expected, atol=1e-5)


def test_hypernet_mlp_no_l2_norm():
    mlp = HyperNetMLP(input_dim=16, output_dim=100, norm_output=False, output_scale_init=0.01)
    x = torch.randn(2, 16)
    out = mlp(x)
    norms = torch.linalg.norm(out, dim=-1)
    assert not torch.allclose(norms, torch.full_like(norms, 0.01), atol=1e-3)


def test_hypernet_mlp_small_init_output_layer():
    mlp = HyperNetMLP(input_dim=16, output_dim=100, output_scale_init=0.1)
    out_w = mlp.output_layer.weight
    assert out_w.std().item() < 0.05


# ====== D5: detach_pred_context ======

def test_detach_pred_context_default_true(hyper_net):
    assert hyper_net.detach_pred_context is True


def test_detach_pred_context_blocks_grad_to_ctx_aug(hyper_net):
    ctx_aug = torch.randn(2, 80, requires_grad=True)
    theta_rew, theta_pred = hyper_net.forward_subjective(ctx_aug)
    (theta_pred ** 2).sum().backward()
    assert ctx_aug.grad is None or ctx_aug.grad.abs().sum().item() == 0.0


def test_detach_pred_context_false_allows_grad():
    net = DualHyperNetwork(
        c_ctx_dim=16, ctx_aug_dim=80,
        trans_param_count=37_000, rew_param_count=4_000, pred_param_count=42_000,
        detach_pred_context=False,
    )
    ctx_aug = torch.randn(2, 80, requires_grad=True)
    _, theta_pred = net.forward_subjective(ctx_aug)
    (theta_pred ** 2).sum().backward()
    assert ctx_aug.grad is not None
    assert ctx_aug.grad.abs().sum().item() > 0


def test_rew_path_grad_to_ctx_aug(hyper_net):
    ctx_aug = torch.randn(2, 80, requires_grad=True)
    theta_rew, _ = hyper_net.forward_subjective(ctx_aug)
    (theta_rew ** 2).sum().backward()
    assert ctx_aug.grad is not None
    assert ctx_aug.grad.abs().sum().item() > 0


# ====== 梯度流完整性 ======

def test_gradient_flow_trans(hyper_net):
    c_ctx = torch.randn(2, 16)
    theta = hyper_net.forward_trans(c_ctx)
    (theta ** 2).sum().backward()
    for name, p in hyper_net.hyper_trans.named_parameters():
        assert p.grad is not None, f"No grad for hyper_trans.{name}"
        assert p.grad.abs().sum() > 0, f"Zero grad for hyper_trans.{name}"


def test_gradient_flow_subjective(hyper_net):
    ctx_aug = torch.randn(2, 80)
    theta_rew, theta_pred = hyper_net.forward_subjective(ctx_aug)
    ((theta_rew ** 2).sum() + (theta_pred ** 2).sum()).backward()
    for name, p in hyper_net.hyper_rew.named_parameters():
        assert p.grad is not None, f"No grad for hyper_rew.{name}"
        assert p.grad.abs().sum() > 0, f"Zero grad for hyper_rew.{name}"
    for name, p in hyper_net.hyper_pred.named_parameters():
        assert p.grad is not None, f"No grad for hyper_pred.{name}"
        assert p.grad.abs().sum() > 0, f"Zero grad for hyper_pred.{name}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
