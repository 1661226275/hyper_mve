"""TriContextEncoder — v4 三路条件向量组合 (Pkg-03 spec 01, Ch4.2.4).

本包总入口. 把 c_ctx (16) + role_i (32) + belief_i (32) 三路 concat 为 80 维
ctx_i ∈ ℝ^80, 作为 DualHyperNetwork v2 hyper_rew / hyper_pred 的统一输入. 同时提供
c_ctx 单独取出接口 forward_c_ctx_only (供 hyper_trans 客观通路使用).

职责 (P5 修订): 仅做 "调三 encoder + 每路 LN + concat" 协调; 具体投影/池化逻辑在
各 sub-encoder (CEncoder / RoleEncoder / BeliefEncoder).

每路 LN (D7 + P7): 每路输出 LayerNorm (ln_c/ln_role/ln_belief) 防三路量级悬殊;
sub-encoder 内部不含 LN (P7 责任集中).

三路 concat 顺序固定 (Ch4.2.4): ctx_i = Concat[c_ctx, role, belief]
    切片: [0:16] c_ctx, [16:48] role, [48:80] belief.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from hyper_mve.configs import EnvConfig, ModelConfig
from hyper_mve.models.c_encoder import CEncoder
from hyper_mve.models.role_encoder import RoleEncoder
from hyper_mve.models.belief_encoder import BeliefEncoder


class TriContextEncoder(nn.Module):
    """v4 三路条件向量组合 (Ch4.2.4).

    输出 ctx_i ∈ ℝ^80 = Concat[c_ctx (16), role (32), belief (32)].

    职责 (P5 修订):
        本模块仅做 "调三 encoder + 每路 LN + concat" 协调.
        具体投影 / 池化逻辑在各 encoder (BeliefEncoder/RoleEncoder/CEncoder).

    每路 LN 设计 (D7 + P7 修订):
        - 每路输出 LayerNorm (ln_c / ln_role / ln_belief), 防止三路量级悬殊
        - 各 sub-encoder 内部**不**含 LN (P7 责任集中)
    """

    def __init__(self, env_cfg: EnvConfig, model_cfg: ModelConfig):
        super().__init__()
        self.env_cfg = env_cfg
        self.model_cfg = model_cfg

        self.N = env_cfg.N
        self.d_c = model_cfg.d_c                        # 16
        self.d_role = model_cfg.d_role                  # 32
        self.d_belief = model_cfg.d_belief              # 32

        # 三路 sub-encoders (各自独立模块, P5)
        self.c_encoder = CEncoder(d_c=self.d_c)
        self.role_encoder = RoleEncoder(
            N=self.N,
            d_id_emb=model_cfg.d_id_emb,        # 8
            d_type_emb=model_cfg.d_type_emb,    # 8
            d_cap_emb=model_cfg.d_cap_emb,      # 16
        )
        self.belief_encoder = BeliefEncoder(env_cfg, model_cfg)

        # 三路 LayerNorm (D7: 每路独立 LN, 防止 concat 后量级偏斜)
        # P7: sub-encoder 内部不含 LN, LN 责任集中在此
        self.ln_c = nn.LayerNorm(self.d_c)
        self.ln_role = nn.LayerNorm(self.d_role)
        self.ln_belief = nn.LayerNorm(self.d_belief)

    @property
    def d_ctx_aug(self) -> int:
        """ctx_i 总维度 = 16 + 32 + 32 = 80 (Ch4.2.4)."""
        return self.d_c + self.d_role + self.d_belief

    def forward(
        self,
        c_t: torch.Tensor,                  # (B,) or (B, 1) float32
        agent_ids: torch.Tensor,            # (B, N) int64
        types: torch.Tensor,                # (B, N) int64 (AgentType.value)
        caps: torch.Tensor,                 # (B, N, 4) float32 RAW CapabilityVector
        belief: tuple[torch.Tensor, torch.Tensor],
                                            # (c_hat: (B, N), z_hat: (B, N, N-1, 2))
    ) -> torch.Tensor:                      # (B, N, 80) float32
        """组合三路 → ctx_i.

        三路语义:
            c_ctx (16): batch-level 共享 (所有 N agent 看到同一 c_t, Harsanyi 共同知识)
            role (32):  per-agent (Self-Info: 仅 own type + cap + id)
            belief (32): per-agent (BeliefNet 推断的 ĉ_i 与 ẑ_{i,j})
        """
        B = c_t.shape[0]
        N = agent_ids.shape[1]
        assert agent_ids.shape == (B, N)
        assert types.shape == (B, N)
        assert caps.shape == (B, N, 4)
        c_hat, z_hat = belief
        assert c_hat.shape == (B, N)
        assert z_hat.shape == (B, N, N - 1, 2)

        # ====== 1. c_ctx 客观通路 ======
        # c_t shape (B,) or (B, 1) -> (B, 1)
        if c_t.dim() == 1:
            c_t = c_t.unsqueeze(-1)
        c_ctx = self.c_encoder(c_t)                     # (B, 16)
        c_ctx = self.ln_c(c_ctx)                        # 每路 LN
        # broadcast 到 (B, N, 16)
        c_ctx = c_ctx.unsqueeze(1).expand(B, N, self.d_c)

        # ====== 2. role 角色通路 ======
        # role_encoder 内部自动 normalize cap (C 修订, 见 spec 03)
        role = self.role_encoder(agent_ids, types, caps)  # (B, N, 32)
        role = self.ln_role(role)

        # ====== 3. belief 信念通路 ======
        belief_vec = self.belief_encoder(c_hat, z_hat)    # (B, N, 32)
        belief_vec = self.ln_belief(belief_vec)

        # ====== 4. concat 三路 ======
        ctx_i = torch.cat([c_ctx, role, belief_vec], dim=-1)  # (B, N, 80)

        return ctx_i

    def forward_c_ctx_only(
        self,
        c_t: torch.Tensor,                  # (B,) or (B, 1)
    ) -> torch.Tensor:                      # (B, 16)
        """仅返回 c_ctx (供 hyper_trans 客观通路使用).

        hyper_trans 不依赖 N (所有 agent 共享 θ_state),
        故此接口不接 agent_ids / types / caps.
        """
        if c_t.dim() == 1:
            c_t = c_t.unsqueeze(-1)
        c_ctx = self.c_encoder(c_t)
        c_ctx = self.ln_c(c_ctx)
        return c_ctx  # (B, 16)
