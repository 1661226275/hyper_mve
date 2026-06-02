"""Unit tests for ``hyper_mve.models.permutation_invariant_pool`` (Pkg-03 spec 07 §5.1)."""
from __future__ import annotations

import pytest
import torch

from hyper_mve.models.permutation_invariant_pool import (
    MeanPool, MaxPool, AttentionPool, make_pool,
)


def test_mean_pool_output_shape():
    pool = MeanPool(feat_dim=2)
    x = torch.randn(2, 4, 3, 2)             # (B, N, N-1, F)
    out = pool(x)
    assert out.shape == (2, 4, 2)


def test_max_pool_output_shape():
    pool = MaxPool(feat_dim=2)
    x = torch.randn(2, 4, 3, 2)
    out = pool(x)
    assert out.shape == (2, 4, 2)


def test_attention_pool_output_shape():
    pool = AttentionPool(feat_dim=2, head_dim=8)
    x = torch.randn(2, 4, 3, 2)
    out = pool(x)
    assert out.shape == (2, 4, 2)


def test_mean_pool_set_invariant():
    """mean pool 严格 set-invariant: 打乱顺序输出不变."""
    pool = MeanPool(feat_dim=2)
    x = torch.randn(2, 4, 5, 2)
    out_orig = pool(x)

    perm = torch.randperm(5)
    x_perm = x[:, :, perm, :]
    out_perm = pool(x_perm)

    assert torch.allclose(out_orig, out_perm, atol=1e-6)


def test_max_pool_set_invariant():
    pool = MaxPool(feat_dim=2)
    x = torch.randn(2, 4, 5, 2)
    out_orig = pool(x)

    perm = torch.randperm(5)
    x_perm = x[:, :, perm, :]
    out_perm = pool(x_perm)

    assert torch.allclose(out_orig, out_perm, atol=1e-6)


def test_attention_pool_set_invariant():
    """attention pool (K, V 共享 projection) 也应 set-invariant."""
    torch.manual_seed(0)
    pool = AttentionPool(feat_dim=2, head_dim=8)
    pool.eval()

    x = torch.randn(2, 4, 5, 2)
    out_orig = pool(x)

    perm = torch.randperm(5)
    x_perm = x[:, :, perm, :]
    out_perm = pool(x_perm)

    assert torch.allclose(out_orig, out_perm, atol=1e-5)


def test_mean_pool_computes_average():
    """验证 mean = sum / N."""
    pool = MeanPool(feat_dim=2)
    x = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],   # B=0, N=0
    ]).unsqueeze(0)                              # (1, 1, 3, 2)
    out = pool(x)
    expected = torch.tensor([[[3.0, 4.0]]])     # (1+3+5)/3=3, (2+4+6)/3=4
    assert torch.allclose(out, expected)


def test_max_pool_computes_max():
    """验证 max = elementwise max."""
    pool = MaxPool(feat_dim=2)
    x = torch.tensor([
        [[1.0, 6.0], [3.0, 4.0], [5.0, 2.0]],
    ]).unsqueeze(0)                              # (1, 1, 3, 2)
    out = pool(x)
    expected = torch.tensor([[[5.0, 6.0]]])     # max([1,3,5])=5, max([6,4,2])=6
    assert torch.allclose(out, expected)


def test_mean_pool_no_params():
    """MeanPool 无可学参数."""
    pool = MeanPool(feat_dim=2)
    total = sum(p.numel() for p in pool.parameters())
    assert total == 0


def test_max_pool_no_params():
    pool = MaxPool(feat_dim=2)
    total = sum(p.numel() for p in pool.parameters())
    assert total == 0


def test_attention_pool_has_params():
    """AttentionPool 含 query + K/V projection 参数."""
    pool = AttentionPool(feat_dim=2, head_dim=16)
    total = sum(p.numel() for p in pool.parameters())
    # query: 1*16=16; k_proj: 2*16=32; v_proj: 2*2=4; total=52
    assert 40 < total < 80


def test_make_pool_factory():
    """工厂构造正确类型."""
    mp = make_pool(kind="mean", feat_dim=2)
    assert isinstance(mp, MeanPool)

    xp = make_pool(kind="max", feat_dim=2)
    assert isinstance(xp, MaxPool)

    ap = make_pool(kind="attention", feat_dim=2, head_dim=8)
    assert isinstance(ap, AttentionPool)
    assert ap.head_dim == 8


def test_make_pool_unknown_kind():
    with pytest.raises(ValueError, match="Unknown pool kind"):
        make_pool(kind="weird", feat_dim=2)


def test_pool_n_minus_one_equals_1():
    """N=2 时 N-1=1, mean/max/attention 应正常处理单元素."""
    x = torch.randn(2, 2, 1, 2)                 # (B=2, N=2, N-1=1, F=2)

    mp = MeanPool(2)
    out_m = mp(x)
    assert out_m.shape == (2, 2, 2)
    # 单元素 mean 应直接返回该值
    assert torch.allclose(out_m, x.squeeze(-2))

    xp = MaxPool(2)
    out_x = xp(x)
    assert torch.allclose(out_x, x.squeeze(-2))

    ap = AttentionPool(2, head_dim=4)
    ap.eval()
    out_a = ap(x)
    assert out_a.shape == (2, 2, 2)


def test_gradient_flow_attention():
    """AttentionPool 反向传播覆盖 query / k_proj / v_proj."""
    pool = AttentionPool(feat_dim=2, head_dim=8)
    x = torch.randn(2, 4, 3, 2, requires_grad=True)
    out = pool(x)
    loss = (out ** 2).sum()
    loss.backward()

    for name, p in pool.named_parameters():
        assert p.grad is not None, f"No grad for {name}"
        assert p.grad.norm() > 0, f"Zero grad for {name}"
