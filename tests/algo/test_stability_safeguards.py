"""Pkg-04 spec 03 acceptance tests: stability safeguards (phase-2 trim).

Kept: hypernet safeguards (small_init, L2 norm + output_scale) and the AdaLN
(1+γ) factor — these modules survive in ``hyper_mve.algo.modules`` and are
consumed by the mazero_mixed subjective model. The retired
``HyperMuZeroModel`` / ``TransitionNet`` defence cases were deleted with the
v5 stack (the fork's own smoke/regression gates cover the integrated model).
"""
import pytest
import torch

from hyper_mve.algo.modules.functional_nets import adaln_forward
from hyper_mve.algo.modules.hyper_network import HyperNetMLP


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


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
