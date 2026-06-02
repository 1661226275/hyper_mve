"""CEncoder — c_ctx 客观通路编码器 (Pkg-03 spec 02, Ch4.2.1).

将共享 context 标量 c_t ∈ [0, 1] 编码为 c_ctx ∈ ℝ^{d_c=16}, 作为 TriContextEncoder
的第一路子模块, 同时是 hyper_trans 客观通路的唯一输入 (保证物理状态预测在所有
agent 间共享).

结构 (Ch4.2.1 默认): Linear(1, 32) -> ReLU -> Linear(32, 32) -> ReLU -> Linear(32, 16)

输出不带 LayerNorm (LN 由 TriContextEncoder 调用层做, D7: 每路独立 LN).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CEncoder(nn.Module):
    """c_t 客观通路编码器 (Ch4.2.1).

    将共享 context 标量 c_t ∈ [0, 1] 映射到 c_ctx ∈ ℝ^{d_c=16}.

    结构 (Ch4.2.1 默认):
        Linear(1, 32) -> ReLU -> Linear(32, 32) -> ReLU -> Linear(32, 16)

    输出不带 LayerNorm (LN 在 TriContextEncoder 调用层做, D7).
    """

    def __init__(self, d_c: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.d_c = d_c
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_c),
        )

    def forward(
        self,
        c_t: torch.Tensor,                  # (B, 1) float32
    ) -> torch.Tensor:                      # (B, d_c) float32
        """
        Args:
            c_t: shape (B, 1) 共享 context 标量
        Returns:
            c_ctx: shape (B, d_c=16)
        """
        assert c_t.dim() == 2 and c_t.shape[-1] == 1, (
            f"c_t shape {c_t.shape}, expected (B, 1)"
        )
        return self.mlp(c_t)
