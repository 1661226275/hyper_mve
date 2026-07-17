"""Pkg-04 spec 01 acceptance tests: DualHyperNetwork (v5 subjective-only form)."""
import pytest
import torch

from hyper_mve.utils.configs import V4Config
from hyper_mve.algo.modules import DualHyperNetwork
from hyper_mve.algo.modules.hyper_network import HyperNetMLP


@pytest.fixture
def cfg_duo():
    return V4Config.from_preset("rel_duo")


@pytest.fixture
def hyper_net(cfg_duo):
    """Construct a v5 DualHyperNetwork (subjective rew + pred only)."""
    return DualHyperNetwork(
        ctx_aug_dim=cfg_duo.model.d_ctx_aug,
        rew_param_count=4_000,
        pred_param_count=42_000,
        hidden_dims=cfg_duo.model.hyper_hidden_dims,
        rew_hidden_dims=cfg_duo.model.hyper_rew_hidden_dims,
        rew_output_scale_init=0.1,
        pred_output_scale_init=0.01,
    )


# ====== v5: objective path deleted ======

def test_no_objective_path(hyper_net):
    assert not hasattr(hyper_net, "hyper_trans")
    assert not hasattr(hyper_net, "forward_trans")


# ====== C2: hyper_rew/pred take the 64-dim ctx_aug ======

def test_subjective_input_dim_64(hyper_net):
    ctx_aug = torch.randn(2, 64)
    theta_rew, theta_pred = hyper_net.forward_subjective(ctx_aug)
    assert theta_rew.shape == (2, 4_000)
    assert theta_pred.shape == (2, 42_000)


def test_subjective_ctx_aug_dim_assertion(hyper_net):
    bad_ctx = torch.randn(2, 80)   # the old v4 width must now be rejected
    with pytest.raises(AssertionError, match="ctx_aug last dim"):
        hyper_net.forward_subjective(bad_ctx)


# ====== C5 (v5): ctx_aug_dim == 64 ======

def test_ctx_aug_dim_eq_64(cfg_duo):
    assert cfg_duo.model.d_ctx_aug == 64
    assert cfg_duo.model.d_role == 32
    assert cfg_duo.model.d_belief == 32


# ====== C8: output_scale inits ======

def test_output_scale_inits_rew_pred(hyper_net):
    assert torch.isclose(hyper_net.hyper_rew.output_scale, torch.tensor(0.1))
    assert torch.isclose(hyper_net.hyper_pred.output_scale, torch.tensor(0.01))


# ====== HyperNetMLP unit ======

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
    ctx_aug = torch.randn(2, 64, requires_grad=True)
    theta_rew, theta_pred = hyper_net.forward_subjective(ctx_aug)
    (theta_pred ** 2).sum().backward()
    assert ctx_aug.grad is None or ctx_aug.grad.abs().sum().item() == 0.0


def test_detach_pred_context_false_allows_grad():
    net = DualHyperNetwork(
        ctx_aug_dim=64,
        rew_param_count=4_000, pred_param_count=42_000,
        detach_pred_context=False,
    )
    ctx_aug = torch.randn(2, 64, requires_grad=True)
    _, theta_pred = net.forward_subjective(ctx_aug)
    (theta_pred ** 2).sum().backward()
    assert ctx_aug.grad is not None
    assert ctx_aug.grad.abs().sum().item() > 0


def test_rew_path_grad_to_ctx_aug(hyper_net):
    ctx_aug = torch.randn(2, 64, requires_grad=True)
    theta_rew, _ = hyper_net.forward_subjective(ctx_aug)
    (theta_rew ** 2).sum().backward()
    assert ctx_aug.grad is not None
    assert ctx_aug.grad.abs().sum().item() > 0


# ====== gradient completeness ======

def test_gradient_flow_subjective(hyper_net):
    ctx_aug = torch.randn(2, 64)
    theta_rew, theta_pred = hyper_net.forward_subjective(ctx_aug)
    ((theta_rew ** 2).sum() + (theta_pred ** 2).sum()).backward()
    for name, p in hyper_net.hyper_rew.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, f"hyper_rew.{name}"
    for name, p in hyper_net.hyper_pred.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, f"hyper_pred.{name}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
