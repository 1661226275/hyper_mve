"""PermutationInvariantPool — Pool({ẑ_{i,j}}) 三策略 (Pkg-03 spec 07, Ch4.2.3).

把 agent i 对 N-1 个对手的类型预测 ẑ_{i,j} ∈ Δ^2 聚合为单个 set-invariant 表示,
喂入 TriContextEncoder 的 belief 子通路.

三种策略 (D2), 由 ModelConfig.belief_pool 切换 (Pkg-01 默认 "mean"):
    - mean (默认): pool = (1/(N-1)) * sum_j z_j
    - max:        pool = elementwise_max_j z_j
    - attention:  pool = sum_j softmax(query · k_j) * v_j  (learnable query)

pool 统一在倒数第二维 (dim=-2, N-1 对手维) 聚合.

注意: BiQueryAttentionPool (P10, 以 b_i 为 query 的 cross-attention) 属 future work,
不在 v4 主线实现范围内, 此处不实现.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


class MeanPool(nn.Module):
    """Mean pooling over the second-to-last dim (默认: dim=-2).

    输入: (..., N-1, F)
    输出: (..., F)

    数学: pool = (1/(N-1)) * sum_j x_j

    无可学参数. 严格 set-invariant.
    """

    def __init__(self, feat_dim: int):
        super().__init__()
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=-2)


class MaxPool(nn.Module):
    """Elementwise max pooling over the second-to-last dim.

    输入: (..., N-1, F)
    输出: (..., F)

    数学: pool[f] = max_j x[j, f]

    无可学参数. 严格 set-invariant.
    """

    def __init__(self, feat_dim: int):
        super().__init__()
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.max(dim=-2).values


class AttentionPool(nn.Module):
    """Attention pooling with learnable query.

    输入: (..., N-1, F)
    输出: (..., F)

    数学:
        weights[j] = softmax_j(query · k_j)
        pool = sum_j weights[j] * v_j

    K, V 通过 linear projection 从 x 得到. Q 是 learnable parameter.
    """

    def __init__(self, feat_dim: int, head_dim: int = 16):
        super().__init__()
        self.feat_dim = feat_dim
        self.head_dim = head_dim

        # learnable query (1, head_dim)
        self.query = nn.Parameter(torch.randn(1, head_dim) * 0.02)

        # K, V projections from input
        self.k_proj = nn.Linear(feat_dim, head_dim, bias=False)
        self.v_proj = nn.Linear(feat_dim, feat_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., N-1, F)
        K = self.k_proj(x)                              # (..., N-1, head_dim)
        V = self.v_proj(x)                              # (..., N-1, F)

        # attention weights: (..., N-1, 1)
        # query shape (1, head_dim) -> broadcast 到 (..., N-1, head_dim) 内积
        scores = (K * self.query).sum(dim=-1, keepdim=True)  # (..., N-1, 1)
        scores = scores / (self.head_dim ** 0.5)             # scale
        weights = torch.softmax(scores, dim=-2)              # softmax over N-1

        # weighted sum
        pool = (weights * V).sum(dim=-2)                # (..., F)
        return pool


def make_pool(
    kind: Literal["mean", "max", "attention"],
    feat_dim: int,
    **kwargs,
) -> nn.Module:
    """工厂: 按 kind 构造对应 pool 模块.

    Args:
        kind:     "mean" / "max" / "attention"
        feat_dim: 特征维度 F
        **kwargs: 传递给特定 pool 类 (如 attention 的 head_dim)
    """
    if kind == "mean":
        return MeanPool(feat_dim)
    elif kind == "max":
        return MaxPool(feat_dim)
    elif kind == "attention":
        head_dim = kwargs.get("head_dim", 16)
        return AttentionPool(feat_dim, head_dim=head_dim)
    raise ValueError(f"Unknown pool kind: {kind}. Valid: mean/max/attention")
