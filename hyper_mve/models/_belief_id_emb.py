"""BeliefIdEmbedding — head_opp 用的 agent_id embedding (Pkg-03 spec 05).

BeliefNet 的 head_opp 需要为不同对手 j 生成不同 ẑ_{i,j}, 通过把 agent_j 的 id
embedding 作为条件输入实现.

本模块与 role_encoder.id_emb 是独立的两个 embedding 表 (避免训练时 head_opp 梯度
污染 role 通路), 且维度不同 (本模块 d_emb=16, RoleEncoder.id_emb=8).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BeliefIdEmbedding(nn.Module):
    """head_opp 用的 agent_id embedding (条件化对手类型预测).

    BeliefNet 的 head_opp 需要为不同对手 j 生成不同 ẑ_{i,j},
    通过把 agent_j 的 id embedding 作为条件输入实现.

    本模块与 role_encoder.id_emb 是**独立**的两个 embedding 表
    (避免训练时 head_opp 梯度污染 role 通路).
    """

    def __init__(self, N: int, d_emb: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(N, d_emb)

    def forward(self, agent_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            agent_ids: (...) int64 agent 索引
        Returns:
            (..., d_emb) float32 embedding
        """
        return self.embedding(agent_ids)
