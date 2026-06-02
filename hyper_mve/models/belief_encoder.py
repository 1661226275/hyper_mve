"""BeliefEncoder — belief 通路投影 MLP (Pkg-03 spec 01, P5 拆出, Ch4.2.4).

把 BeliefNet 的 raw heads (c_hat, z_hat) 经 Pool + 投影变为 belief_vec ∈ ℝ^{d_belief=32},
与 CEncoder / RoleEncoder 对称, 是 belief 通路的独立 encoder.

架构 (Ch4.2.4):
    c_hat (B, N) → unsqueeze → Linear(1, 16) → ReLU → c_hat_proj (B, N, 16)
    z_hat (B, N, N-1, 2) → Pool (dim=-2) → (B, N, 2) → Linear(2, 16) → ReLU → (B, N, 16)
    belief_vec = Concat[c_hat_proj, z_pooled_proj] = (B, N, 32)

不含内部 LayerNorm (P7: LN 责任集中在 TriContextEncoder.ln_belief).

梯度门控 (Ch4.6.5): Pkg-04 model 在前 5K step 对 (c_hat, z_hat) 做 .detach() 切断
梯度; 本模块投影 MLP 仍由 main loss 反向传播训练. 切断点在 BeliefNet 输出处 (raw
heads), 不在本模块内.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from hyper_mve.configs import EnvConfig, ModelConfig
from hyper_mve.models.permutation_invariant_pool import make_pool


class BeliefEncoder(nn.Module):
    """belief 通路投影 MLP (P5 拆出, 与 CEncoder/RoleEncoder 对称).

    把 BeliefNet 的 raw heads (c_hat, z_hat) 经 Pool + 投影变为
    belief_vec ∈ ℝ^{d_belief=32}.

    **不含内部 LayerNorm** (P7 修订: LN 责任集中在 TriContextEncoder.ln_belief).
    """

    def __init__(self, env_cfg: EnvConfig, model_cfg: ModelConfig):
        super().__init__()
        self.env_cfg = env_cfg
        self.model_cfg = model_cfg

        self.d_belief = model_cfg.d_belief                      # 32
        self.d_belief_proj = model_cfg.d_belief_proj            # 16

        # 维度一致 (与 Pkg-01 ModelConfig __post_init__ 一致)
        assert self.d_belief == 2 * self.d_belief_proj, (
            f"d_belief({self.d_belief}) != 2 * d_belief_proj({self.d_belief_proj})"
        )

        # c_hat 投影: (B, N, 1) → (B, N, 16); 无内部 LN (P7)
        self.proj_c_hat = nn.Sequential(
            nn.Linear(1, self.d_belief_proj),
            nn.ReLU(),
        )

        # z_hat pooling (D2): 默认 mean, 可切换 max/attention
        self.z_pool = make_pool(
            kind=model_cfg.belief_pool,
            feat_dim=2,                                          # softmax 概率 2 类
        )

        # Pool 后投影: (B, N, 2) → (B, N, 16)
        self.proj_z_pooled = nn.Sequential(
            nn.Linear(2, self.d_belief_proj),
            nn.ReLU(),
        )

    def forward(
        self,
        c_hat: torch.Tensor,                # (B, N) sigmoid ∈ [0, 1]
        z_hat: torch.Tensor,                # (B, N, N-1, 2) softmax
    ) -> torch.Tensor:                      # (B, N, 32) belief_vec
        """组合 c_hat + Pool(z_hat) → belief_vec.

        Args:
            c_hat: BeliefNet head_c 输出 (Ch4.2.3 Head 1, sigmoid scalar)
            z_hat: BeliefNet head_opp 输出 (Ch4.2.3 Head 2, softmax 概率)
                   顺序约定: agent_id 升序跳过 self (Pkg-01 z_hat 字段一致)
        Returns:
            belief_vec: (B, N, 32) = Concat[Proj(c_hat), Proj(Pool(z_hat))]
        """
        B, N = c_hat.shape
        assert z_hat.shape[:3] == (B, N, N - 1)
        assert z_hat.shape[3] == 2

        # c_hat 子分量 (1 → 16)
        c_hat_proj = self.proj_c_hat(c_hat.unsqueeze(-1))   # (B, N, 16)

        # z_hat pooling 在 dim=-2 (N-1 个对手维度) → (B, N, 2)
        z_pooled = self.z_pool(z_hat)
        z_pooled_proj = self.proj_z_pooled(z_pooled)        # (B, N, 16)

        belief_vec = torch.cat([c_hat_proj, z_pooled_proj], dim=-1)  # (B, N, 32)
        return belief_vec
