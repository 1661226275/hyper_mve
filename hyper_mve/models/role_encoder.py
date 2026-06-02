"""RoleEncoder — role 角色通路 (Pkg-03 spec 03, Ch4.2.2, v4 含 type_emb).

把 agent i 的 (id, type τ_i, capability cap_i) 编码为 role_i ∈ ℝ^{d_role=32}:

    role_i = Concat[id_emb_i (8), type_emb_i (8), cap_emb_i (16)]  (精确填满, 无 pad)

v4 关键改动 (相对 v3 role = id_emb + cap_emb): 加入 type_emb —— Ch4.1.3 瓶颈 1
(类型梯度撕裂) 的架构落地. type_emb_α / type_emb_β 是两个独立可学向量, 经
hyper_rew/hyper_pred 生成两套差异巨大的 θ, 从根本消除共享 RewardHead 的类型撕裂.

Self-Info 严格性 (Ch3.7 + Ch4.2.2): role_i 仅含 agent i 自己的 type 与 cap; 他人
type 由 BeliefNet head_opp 推断 (spec 05).

C 修订: forward 接收 RAW cap (B, N, 4), 内部按 CAP_NORM_LO/HI 自动归一到 [0,1]^4.
P7 修订: cap_mlp 内部不含 LayerNorm (LN 责任集中在 TriContextEncoder.ln_role).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from hyper_mve.schemas.capability import CAP_NORM_LO, CAP_NORM_HI


class RoleEncoder(nn.Module):
    """role 角色通路 (Ch4.2.2, v4 含 type_emb).

    将 (id, type, cap) 三元组编码为 role_i ∈ ℝ^{d_role=32}.

    v4 关键 (相对 v3):
        v3: role = Concat[id_emb (8), cap_emb (16)] = 24 (+ 8 pad)
        v4: role = Concat[id_emb (8), type_emb (8), cap_emb (16)] = 32 精确

    Self-Info 严格性 (Ch3.7 + Ch4.2.2):
        本模块仅接收 agent 自己的 (type, cap), 不含他人信息.
        他人 type 由 BeliefNet head_opp 推断 (spec 05).

    C 修订 — cap 输入自动归一化:
        forward() 接收 RAW CapabilityVector (B, N, 4), 内部按 CAP_NORM_LO/HI
        自动归一到 [0, 1]^4 再喂 cap_mlp. 解决 ζ (10-30) vs η (0.5-1.5) ~30 倍
        差距导致 cap_mlp 第一层梯度被 ζ 主导的问题.
        env/buffer/info 仍存 RAW cap; 仅本模块内部消费时归一化.

    P7 修订 — cap_mlp 内部不含 LayerNorm:
        LN 责任集中在 TriContextEncoder.ln_role; cap_mlp 内 LN 与 ln_role 是
        double LN (cap 子段被 LN 两次), 故移除.
    """

    def __init__(
        self,
        N: int,
        d_id_emb: int = 8,
        d_type_emb: int = 8,
        d_cap_emb: int = 16,
        num_types: int = 2,         # AgentType.ALPHA / BETA
        cap_input_dim: int = 4,     # CapabilityVector: η, φ_fov, ν, ζ
        cap_hidden_dim: int = 16,   # cap_mlp 隐藏维 (D8)
    ):
        super().__init__()
        self.N = N
        self.d_id_emb = d_id_emb
        self.d_type_emb = d_type_emb
        self.d_cap_emb = d_cap_emb
        self.d_role = d_id_emb + d_type_emb + d_cap_emb

        # 精确填满约束 (与 Pkg-01 ModelConfig __post_init__ 一致)
        assert self.d_role == 32, (
            f"d_role={self.d_role}, expected 32 (8+8+16). v4 要求精确填满, 无 pad."
        )

        # id_emb: agent_id ∈ {0, ..., N-1} -> d_id_emb 维向量
        self.id_emb = nn.Embedding(N, d_id_emb)

        # type_emb (v4 关键): AgentType ∈ {ALPHA=0, BETA=1} -> d_type_emb 维向量
        self.type_emb = nn.Embedding(num_types, d_type_emb)

        # cap_emb: 4 维 CapabilityVector (已归一) -> d_cap_emb 维 (D8: 2 层 MLP)
        # P7 修订: 无内部 LayerNorm (LN 在 TriContextEncoder.ln_role)
        self.cap_mlp = nn.Sequential(
            nn.Linear(cap_input_dim, cap_hidden_dim),
            nn.ReLU(),
            nn.Linear(cap_hidden_dim, d_cap_emb),
        )

        # cap 归一化常量 (C 修订) - 注册为 buffer 以正确处理 device 迁移
        # CAP_NORM_LO = (0.5, 2.0, 0.8, 10.0), CAP_NORM_HI = (1.5, 4.0, 1.0, 30.0)
        self.register_buffer(
            "_cap_lo", torch.tensor(CAP_NORM_LO, dtype=torch.float32),
        )
        self.register_buffer(
            "_cap_hi", torch.tensor(CAP_NORM_HI, dtype=torch.float32),
        )

    def _normalize_caps(self, caps_raw: torch.Tensor) -> torch.Tensor:
        """归一 (B, N, 4) raw cap 到 [0, 1]^4 (C 修订).

        与 Pkg-01 spec 02 ``CapabilityVector.normalize()`` 数学等价 (向量化版本).
        范围常量 CAP_NORM_LO/HI 来自 Pkg-01 (Ch3.6 硬约束).
        """
        # caps_raw: (B, N, 4); _cap_lo / _cap_hi: (4,) broadcast 自动适配
        return (caps_raw - self._cap_lo) / (self._cap_hi - self._cap_lo)

    def forward(
        self,
        agent_ids: torch.Tensor,            # (B, N) int64
        types: torch.Tensor,                # (B, N) int64 (AgentType.value)
        caps: torch.Tensor,                 # (B, N, 4) float32 RAW CapabilityVector
    ) -> torch.Tensor:                      # (B, N, 32) float32
        """
        Args:
            agent_ids: (B, N) agent 索引 ∈ {0, ..., N-1}
            types:     (B, N) AgentType.value ∈ {0, 1}
            caps:      (B, N, 4) RAW CapabilityVector flatten (η, φ_fov, ν, ζ)
                       内部自动归一到 [0, 1]^4 (C 修订)
        Returns:
            role_i: (B, N, 32) = Concat[id_emb, type_emb, cap_emb]
        """
        B, N = agent_ids.shape
        assert types.shape == (B, N)
        assert caps.shape == (B, N, 4)

        id_vec = self.id_emb(agent_ids)                # (B, N, 8)
        type_vec = self.type_emb(types)                # (B, N, 8)

        # C 修订: cap 归一化后再喂 MLP
        caps_normed = self._normalize_caps(caps)       # (B, N, 4) ∈ [0, 1]^4
        cap_vec = self.cap_mlp(caps_normed)            # (B, N, 16)

        role = torch.cat([id_vec, type_vec, cap_vec], dim=-1)  # (B, N, 32)
        return role
