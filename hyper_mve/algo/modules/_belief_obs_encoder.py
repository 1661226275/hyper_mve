"""BeliefObsEncoder — BeliefNet 内部观测编码器 (Pkg-03 spec 04, Ch4.2.3).

将 (..., obs_dim) 编码为 (..., gru_input_dim), 独立于 Pkg-04 主 RepNet 以避免
Pkg-03 ↔ Pkg-04 循环依赖.

结构钉死 (P6 修订):
    Linear(obs_dim, 128) -> ReLU -> Linear(128, gru_input_dim=64) -> LayerNorm(64)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BeliefObsEncoder(nn.Module):
    """BeliefNet 内部的观测编码器 (独立于 Pkg-04 主 RepNet, 避免循环依赖).

    将 (B, N, obs_dim) 编码为 (B, N, gru_input_dim).

    与主 RepNet 的差异:
        主 RepNet (Pkg-04): obs -> s ∈ ℝ^latent_dim (供 StateTransNet 用)
        本编码器:           obs -> gru_input ∈ ℝ^gru_input_dim (供 GRU 用)

    保持独立有 3 个理由:
        1. 避免 Pkg-03 ↔ Pkg-04 循环依赖
        2. BeliefNet 输入维度可独立调整 (gru_input_dim) 而不影响主 latent
        3. BeliefNet 训练梯度独立流动 (符合 Ch4.6.5 梯度门控设计)
    """

    def __init__(
        self,
        obs_dim: int,
        gru_input_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.gru_input_dim = gru_input_dim

        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, gru_input_dim),
            nn.LayerNorm(gru_input_dim),
        )

    def forward(
        self,
        obs: torch.Tensor,                  # (..., obs_dim) float32
    ) -> torch.Tensor:                      # (..., gru_input_dim) float32
        """支持任意前缀维度 (B,), (B, N), (B, T, N) 等."""
        return self.mlp(obs)
