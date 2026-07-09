"""Pkg-04 spec 03 acceptance tests: stability safeguards (v5 form).

v5 (Pkg-09): the transition is a plain shared ``TransitionNet`` (its residual +
fixed output LayerNorm are covered here); the hypernet safeguards (small_init,
L2 norm + output_scale, AdaLN (1+γ)) apply to the two subjective hypernets.
"""
import pytest
import torch

from hyper_mve.configs import V4Config
from hyper_mve.models import HyperMuZeroModel, TransitionNet
from hyper_mve.models.functional_nets import adaln_forward
from hyper_mve.models.hyper_network import HyperNetMLP
from hyper_mve.models.grad_gating import BeliefGradGating


def _make_transition(cfg):
    return TransitionNet(
        cfg.model.latent_dim,
        cfg.env.N * cfg.env.A,
        cfg.model.hidden_dim,
    )


# ====== defence 1: small_init + L2 norm + output_scale ======

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


def test_output_scale_inits_rew_pred():
    cfg = V4Config.from_preset("rel_duo")
    model = HyperMuZeroModel(cfg)
    # rel_duo sets both subjective scales to 0.1 (film_head grouped RMS).
    assert torch.isclose(model.hyper_net.hyper_rew.output_scale, torch.tensor(0.1))
    assert torch.isclose(model.hyper_net.hyper_pred.output_scale, torch.tensor(0.1))
    # v5: no objective hypernet.
    assert not hasattr(model.hyper_net, "hyper_trans")


# ====== defence 2: AdaLN (1+γ) factor ======

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


# ====== defence 3: Δs residual (v5 plain TransitionNet) ======

def test_transition_delta_s_residual_bounded():
    cfg = V4Config.from_preset("rel_duo")
    stn = _make_transition(cfg)

    B = 2
    state = torch.randn(B, cfg.model.latent_dim)
    action = torch.zeros(B, cfg.env.N * cfg.env.A)
    action[:, 0] = 1.0

    s_next = stn(state, action)
    # residual form: Δs passes a LayerNorm, so |Δs| stays O(1) at init.
    diff = (s_next - state).abs().mean().item()
    assert diff < 2.0


def test_transition_not_identity():
    cfg = V4Config.from_preset("rel_duo")
    stn = _make_transition(cfg)
    B = 2
    state = torch.randn(B, cfg.model.latent_dim)
    action = torch.zeros(B, cfg.env.N * cfg.env.A)
    action[:, 0] = 1.0
    assert not torch.allclose(stn(state, action), state, atol=1e-3)


def test_transition_uses_fixed_output_layernorm():
    cfg = V4Config.from_preset("rel_duo")
    stn = _make_transition(cfg)
    assert isinstance(stn.ln_out, torch.nn.LayerNorm)
    has_ln_param = any(p is stn.ln_out.weight or p is stn.ln_out.bias for p in stn.parameters())
    assert has_ln_param


def test_transition_action_sensitivity():
    """The shared transition must react to the joint action (perspective-invariant physics)."""
    cfg = V4Config.from_preset("rel_duo")
    stn = _make_transition(cfg)
    B = 2
    state = torch.randn(B, cfg.model.latent_dim)
    a0 = torch.zeros(B, cfg.env.N * cfg.env.A); a0[:, 0] = 1.0
    a1 = torch.zeros(B, cfg.env.N * cfg.env.A); a1[:, 5] = 1.0
    assert not torch.allclose(stn(state, a0), stn(state, a1), atol=1e-5)


# ====== defence 5 (v4 addition): belief gradient gating reference ======

def test_grad_gating_referenced():
    cfg = V4Config.from_preset("rel_duo")
    model = HyperMuZeroModel(cfg)
    assert isinstance(model.grad_gating, BeliefGradGating)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
