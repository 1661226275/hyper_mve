"""Unit tests for ``hyper_mve.models.c_encoder`` (Pkg-03 spec 02 §5.1)."""
from __future__ import annotations

import pytest
import torch

from hyper_mve.models.c_encoder import CEncoder


def test_output_shape_default():
    """默认 d_c=16."""
    enc = CEncoder()
    c_t = torch.zeros(8, 1)
    out = enc(c_t)
    assert out.shape == (8, 16)


def test_output_shape_custom_d_c():
    """d_c 可配置."""
    enc = CEncoder(d_c=24)
    c_t = torch.zeros(8, 1)
    out = enc(c_t)
    assert out.shape == (8, 24)


def test_input_shape_assertion():
    """c_t 必须 (B, 1), 不接 (B,)."""
    enc = CEncoder()
    c_t = torch.zeros(8)
    with pytest.raises(AssertionError, match="c_t shape"):
        enc(c_t)


def test_gradient_flow():
    """反向传播覆盖所有可学参数."""
    enc = CEncoder()
    c_t = torch.rand(4, 1, requires_grad=True)
    out = enc(c_t)
    loss = (out ** 2).sum()
    loss.backward()
    for name, p in enc.named_parameters():
        assert p.grad is not None, f"No grad for {name}"
        assert p.grad.norm() > 0, f"Zero grad for {name}"


def test_distinct_c_distinct_output():
    """不同 c_t 应产生不同 c_ctx (确保 MLP 非常数函数)."""
    enc = CEncoder()
    c_t_a = torch.tensor([[0.2]])
    c_t_b = torch.tensor([[0.8]])
    out_a = enc(c_t_a)
    out_b = enc(c_t_b)
    assert not torch.allclose(out_a, out_b)


def test_param_count():
    """MLP 默认参数量 ≈ 1.6K."""
    enc = CEncoder()
    total = sum(p.numel() for p in enc.parameters())
    assert 1500 < total < 1800


def test_boundary_values():
    """c=0.0 和 c=1.0 边界正常."""
    enc = CEncoder()
    c_t = torch.tensor([[0.0], [1.0]])
    out = enc(c_t)
    assert out.shape == (2, 16)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_no_internal_layernorm():
    """本模块输出不带 LN, 由 TriContextEncoder 外部加."""
    enc = CEncoder()
    # 模块中无 LayerNorm 子模块
    has_ln = any(isinstance(m, torch.nn.LayerNorm) for m in enc.modules())
    assert not has_ln, "CEncoder should NOT contain internal LayerNorm (D7: external)"
