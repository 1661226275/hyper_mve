"""Pkg-04 spec 03 acceptance tests: 4 道稳定性防线保留 + 防线 5 引用.

NOTE (plan 决议 4): FunctionalStateTransNet 保留 v4.7 位置参数签名
(latent_dim, joint_action_dim, hidden_dim), 不改为 (cfg). 故此处用位置参数构造,
与 spec 03 §5.1 伪代码的 Functional*(cfg) 不同 (以本计划为准).
"""
import pytest
import torch

from hyper_mve.configs import V4Config
from hyper_mve.models import HyperMuZeroModel
from hyper_mve.models.functional_nets import adaln_forward, FunctionalStateTransNet
from hyper_mve.models.hyper_network import HyperNetMLP
from hyper_mve.models.grad_gating import BeliefGradGating


def _make_state_trans(cfg):
    return FunctionalStateTransNet(
        cfg.model.latent_dim,
        cfg.env.N * cfg.env.A,
        cfg.model.hidden_dim,
    )


# ====== 防线 1: small_init + L2 norm + output_scale ======

def test_small_init_output_layer_std():
    mlp = HyperNetMLP(input_dim=16, output_dim=1000)
    std = mlp.output_layer.weight.std().item()
    assert std < 0.05


def test_l2_norm_output_magnitude():
    mlp = HyperNetMLP(input_dim=16, output_dim=100, norm_output=True, output_scale_init=0.1)
    x = torch.randn(4, 16)
    out = mlp(x)
    norms = torch.linalg.norm(out, dim=-1)
    assert torch.allclose(norms, torch.full_like(norms, 0.1), atol=1e-5)


def test_output_scale_inits_trans_rew_pred():
    cfg = V4Config.from_preset("medium")
    model = HyperMuZeroModel(cfg)
    assert torch.isclose(model.hyper_net.hyper_trans.output_scale, torch.tensor(0.01))
    assert torch.isclose(model.hyper_net.hyper_rew.output_scale, torch.tensor(0.1))
    assert torch.isclose(model.hyper_net.hyper_pred.output_scale, torch.tensor(0.01))


# ====== 防线 2: AdaLN (1+γ) factor ======

def test_adaln_one_plus_gamma_factor():
    B, in_dim, out_dim = 2, 16, 32
    x = torch.randn(B, in_dim)
    weight = torch.randn(B, out_dim, in_dim) * 0.01
    bias = torch.zeros(B, out_dim)
    gamma = torch.zeros(B, out_dim)
    beta = torch.randn(B, out_dim) * 0.5

    out = adaln_forward(x, weight, bias, gamma, beta)
    weight_zero = torch.zeros(B, out_dim, in_dim)
    out_no_weight = adaln_forward(x, weight_zero, bias, gamma, beta)
    assert not torch.allclose(out, out_no_weight, atol=1e-3)


def test_adaln_gamma_nonzero():
    B, in_dim, out_dim = 1, 8, 16
    x = torch.randn(B, in_dim)
    weight = torch.eye(out_dim, in_dim).unsqueeze(0).expand(B, -1, -1).contiguous()
    weight = weight[:, :, :in_dim]
    bias = torch.zeros(B, out_dim)
    gamma = torch.ones(B, out_dim) * 1.0
    beta = torch.zeros(B, out_dim)

    out = adaln_forward(x, weight, bias, gamma, beta)
    assert not torch.isnan(out).any()
    assert out.std().item() > 0.01


# ====== 防线 3: Δs 残差 ======

def test_state_trans_delta_s_residual():
    cfg = V4Config.from_preset("medium")
    stn = _make_state_trans(cfg)

    B = 2
    state = torch.randn(B, cfg.model.latent_dim)
    action = torch.zeros(B, cfg.env.N * cfg.env.A)
    action[:, 0] = 1.0
    flat_params = torch.zeros(B, stn.total_params)

    s_next = stn(state, action, flat_params)
    diff = (s_next - state).abs().mean().item()
    assert diff < 1.0


def test_state_trans_residual_not_identity_with_nonzero_params():
    cfg = V4Config.from_preset("medium")
    stn = _make_state_trans(cfg)

    B = 2
    state = torch.randn(B, cfg.model.latent_dim)
    action = torch.zeros(B, cfg.env.N * cfg.env.A)
    action[:, 0] = 1.0
    flat_params = torch.randn(B, stn.total_params) * 0.5

    s_next = stn(state, action, flat_params)
    assert not torch.allclose(s_next, state, atol=1e-3)


def test_state_trans_uses_fixed_layernorm():
    cfg = V4Config.from_preset("medium")
    stn = _make_state_trans(cfg)
    assert isinstance(stn.ln, torch.nn.LayerNorm)
    has_ln_param = any(p is stn.ln.weight or p is stn.ln.bias for p in stn.parameters())
    assert has_ln_param


# ====== 数值回归对比 v4.7 (可选, 需 v4.7 archive 启用) ======

@pytest.mark.skip(reason="Requires v4.7 archive snapshot; enable in regression CI")
def test_adaln_matches_v47_numerical():
    pass


# ====== 防线 5 (v4 新增): belief 梯度门控引用 ======

def test_grad_gating_referenced():
    cfg = V4Config.from_preset("medium")
    model = HyperMuZeroModel(cfg)
    assert isinstance(model.grad_gating, BeliefGradGating)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
