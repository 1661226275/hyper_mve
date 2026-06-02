"""BeliefNet — v4 GRU 主干 + 两 head (Pkg-03 spec 04 + 05, Ch4.2.3 + 4.5).

把 agent i 的观测历史编码为隐状态 b_i^t ∈ ℝ^128, 再经两 head 推断:
    head_c   -> ĉ_i^t (B, N) sigmoid scalar ∈ [0, 1]   (资源丰度推断, MSE 监督)
    head_opp -> ẑ_{i,j}^t (B, N, N-1, 2) softmax         (对手类型推断, Oracle CE 监督)

关键性质:
    - 共享权重, 独立隐状态: N 个 agent 用同一 GRU 参数, 各自维护独立 b_i^t.
    - uni-directional (D6): causal, RL 在线推断必需.
    - 两套 API: step() 单步 (worker 在线) + forward() 序列 (trainer K-step unroll).
    - GRU 后接 LayerNorm (Ch4.6.2 防线 2): 稳定 b_i^t 演化.

z_hat 顺序约定 (硬约束, 与 Pkg-01 TimeStepRecord 一致):
    z_hat[b, i, k] 对应 agent_id = (k if k < i else k + 1)  (agent_id 升序, 跳过 self).

v4 关键改动 (head_opp): 从 v3 自监督动作预测改为 Oracle 监督类型 2 分类.
belief 梯度门控 (Ch4.6.5) 在 Pkg-04 model 实现; 本模块 forward 不参与门控逻辑.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from hyper_mve.configs import EnvConfig, ModelConfig
from hyper_mve.schemas import ObservationLayout
from hyper_mve.models._belief_obs_encoder import BeliefObsEncoder
from hyper_mve.models._belief_id_emb import BeliefIdEmbedding


class BeliefNet(nn.Module):
    """v4 BeliefNet (Ch4.2.3 + 4.5).

    完整架构 (GRU 主干见 spec 04, 两 head 见 spec 05):

        obs (B, N, obs_dim)
          -> BeliefObsEncoder -> gru_input (B, N, 64)
          -> GRUCell(64, 128) -> b_i^t (B, N, 128) -> LN -> hidden
          -> head_c (MLP) -> sigmoid -> c_hat (B, N) scalar ∈ [0, 1]
          -> head_opp:
               for each opponent j:
                   trunk_in = Concat[b_i, id_emb(j)] (B, N, 128+16=144)
                   logits = MLP_opp(trunk_in) (B, N, 2)
                   z_{i,j} = softmax(logits)
               stack over j != i -> z_hat (B, N, N-1, 2)

    顺序约定 (硬约束, Pkg-01 TimeStepRecord z_hat 一致):
        z_hat[b, i, k] 对应 agent_id = (k if k < i else k + 1)
        即按 agent_id 升序, 跳过 self.
    """

    def __init__(self, env_cfg: EnvConfig, model_cfg: ModelConfig):
        super().__init__()
        self.env_cfg = env_cfg
        self.model_cfg = model_cfg

        self.N = env_cfg.N
        self.obs_dim = ObservationLayout.total_dim(env_cfg.N, env_cfg.K)
        self.gru_input_dim = 64                     # 内部, 不暴露到 ModelConfig
        self.gru_hidden = model_cfg.belief_gru_hidden       # 128
        self.d_opp_id_emb = 16                       # head_opp 内部用 (独立于 role.id_emb)

        # ====== GRU 主干 (spec 04) ======
        self.obs_encoder = BeliefObsEncoder(
            obs_dim=self.obs_dim,
            gru_input_dim=self.gru_input_dim,
            hidden_dim=128,
        )
        # GRU 主干 (D6: uni-directional, 单 cell 复用)
        self.gru = nn.GRUCell(self.gru_input_dim, self.gru_hidden)
        # GRU 输出 LayerNorm (Ch4.6.2 防线 2)
        self.ln_belief = nn.LayerNorm(self.gru_hidden)

        # ====== Head 1: head_c (sigmoid scalar) ======
        head_c_hidden = 64
        self.head_c = nn.Sequential(
            nn.Linear(self.gru_hidden, head_c_hidden),
            nn.ReLU(),
            nn.Linear(head_c_hidden, 1),
            nn.Sigmoid(),
        )

        # ====== Head 2: head_opp (Oracle 监督类型 2 分类, v4 关键) ======
        self.opp_id_emb = BeliefIdEmbedding(N=self.N, d_emb=self.d_opp_id_emb)
        head_opp_hidden = 64
        self.head_opp_mlp = nn.Sequential(
            nn.Linear(self.gru_hidden + self.d_opp_id_emb, head_opp_hidden),
            nn.ReLU(),
            nn.Linear(head_opp_hidden, 2),  # 2 类: α / β
        )
        # softmax 在 forward 内部应用 (而非 layer), 便于训练时与 CE 配合;
        # 但 v4 ẑ 进 belief 通路是概率 (D3), 故 forward 默认返回 softmax 概率

    def init_hidden(
        self,
        batch_size: int,
        num_agents: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:                      # (B, N, 128) zeros
        """初始化 GRU hidden (每个 (batch_idx, agent_idx) 独立 state)."""
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(batch_size, num_agents, self.gru_hidden, device=device)

    def _gru_step(
        self,
        obs_t: torch.Tensor,                # (B, N, obs_dim)
        prev_hidden: torch.Tensor,          # (B, N, 128)
    ) -> torch.Tensor:                      # new_hidden (B, N, 128) LN 后
        """单步 GRU update (内部方法, step 与 forward 共享).

        将 (B, N, ...) reshape 为 (B*N, ...) 以适配 nn.GRUCell 期望的 (batch, dim).
        """
        B, N, _ = obs_t.shape
        gru_input = self.obs_encoder(obs_t)             # (B, N, 64)
        gi_flat = gru_input.reshape(B * N, -1)
        h_flat = prev_hidden.reshape(B * N, -1)
        new_h_flat = self.gru(gi_flat, h_flat)
        new_h = new_h_flat.reshape(B, N, -1)
        return self.ln_belief(new_h)                    # Ch4.6.2 防线 2

    def _compute_c_hat(self, hidden: torch.Tensor) -> torch.Tensor:
        """head_c forward.

        Args:
            hidden: (..., 128)
        Returns:
            c_hat: (...) sigmoid ∈ [0, 1] (last dim squeezed)
        """
        c_hat = self.head_c(hidden)                # (..., 1)
        return c_hat.squeeze(-1)

    def _compute_z_hat(self, hidden: torch.Tensor) -> torch.Tensor:
        """head_opp forward (v4 关键: Oracle 监督类型 2 分类).

        Args:
            hidden: (B, N, 128) per-agent GRU hidden
        Returns:
            z_hat: (B, N, N-1, 2) softmax 概率
                   z_hat[b, i, k] 对应 agent_id = (k if k < i else k + 1)
        """
        B, N, _ = hidden.shape
        device = hidden.device

        # 1. 构造对手 id 张量: opp_ids[b, i, k] = (k if k < i else k + 1)
        #    其中 k ∈ {0, ..., N-2}
        opp_ids = torch.zeros(B, N, N - 1, dtype=torch.long, device=device)
        for i in range(N):
            opp_id_list = [j for j in range(N) if j != i]   # [agent_id 升序, 跳过 self]
            opp_ids[:, i, :] = torch.tensor(opp_id_list, dtype=torch.long, device=device)

        # 2. 取对手 id embedding: (B, N, N-1, d_opp_id_emb=16)
        opp_id_vec = self.opp_id_emb(opp_ids)                # (B, N, N-1, 16)

        # 3. 重复 hidden 到匹配 opp 维度: (B, N, 128) -> (B, N, N-1, 128)
        hidden_exp = hidden.unsqueeze(2).expand(B, N, N - 1, self.gru_hidden)

        # 4. concat: (B, N, N-1, 128+16=144)
        trunk_in = torch.cat([hidden_exp, opp_id_vec], dim=-1)

        # 5. MLP -> logits (B, N, N-1, 2)
        logits = self.head_opp_mlp(trunk_in)

        # 6. softmax (D3: 概率形式进 belief 通路)
        z_hat = torch.softmax(logits, dim=-1)

        return z_hat

    def step(
        self,
        obs_t: torch.Tensor,                # (B, N, obs_dim)
        prev_hidden: torch.Tensor,          # (B, N, 128)
        oracle_z: Optional[torch.Tensor] = None,
                                            # (B, N, N-1, 2), Stage 1/2 oracle 用
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """单步推断 (worker 在线调用).

        Args:
            obs_t:        当前 obs
            prev_hidden:  上一步 hidden
            oracle_z:     可选, 课程 Stage 1 用 oracle τ one-hot 替代 head_opp.
                          若给出, 仍计算 head_opp (用于 L_opp 训练), 但返回 oracle_z.

        Returns:
            new_hidden: (B, N, 128)
            c_hat:      (B, N) sigmoid ∈ [0, 1]
            z_hat:      (B, N, N-1, 2) softmax (或 oracle_z 直接返回)
        """
        new_hidden = self._gru_step(obs_t, prev_hidden)
        c_hat = self._compute_c_hat(new_hidden)
        z_hat_predicted = self._compute_z_hat(new_hidden)

        if oracle_z is not None:
            # Stage 1 oracle 模式: 用 oracle 替代 head_opp 输出进 belief 通路
            # 但 head_opp 仍训练 (head_opp logits 在 z_hat_predicted 内, 由 belief_loss 计算 L_opp)
            assert oracle_z.shape == z_hat_predicted.shape, (
                f"oracle_z shape {oracle_z.shape} != z_hat shape {z_hat_predicted.shape}"
            )
            z_hat_out = oracle_z
        else:
            z_hat_out = z_hat_predicted

        return new_hidden, c_hat, z_hat_out

    def forward(
        self,
        obs_seq: torch.Tensor,              # (B, T, N, obs_dim)
        init_hidden: Optional[torch.Tensor] = None,
                                            # (B, N, 128), 默认 zeros
        oracle_z_seq: Optional[torch.Tensor] = None,
                                            # (B, T, N, N-1, 2), Stage 1/2 用
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """序列推断 (trainer K-step unroll).

        内部循环 T 步 _gru_step, 每步收集 c_hat / z_hat.

        Returns:
            hidden_seq: (B, T, N, 128)
            c_hat_seq:  (B, T, N)
            z_hat_seq:  (B, T, N, N-1, 2)
                        (若 oracle_z_seq 给出, 返回 oracle_z_seq; 但 head_opp 仍 forward 计算)
        """
        B, T, N, _ = obs_seq.shape
        device = obs_seq.device

        if init_hidden is None:
            init_hidden = self.init_hidden(B, N, device=device)

        hidden = init_hidden
        hidden_list = []
        c_hat_list = []
        z_hat_predicted_list = []

        for t in range(T):
            hidden = self._gru_step(obs_seq[:, t], hidden)
            c_hat_t = self._compute_c_hat(hidden)
            z_hat_pred_t = self._compute_z_hat(hidden)

            hidden_list.append(hidden)
            c_hat_list.append(c_hat_t)
            z_hat_predicted_list.append(z_hat_pred_t)

        hidden_seq = torch.stack(hidden_list, dim=1)            # (B, T, N, 128)
        c_hat_seq = torch.stack(c_hat_list, dim=1)              # (B, T, N)
        z_hat_predicted_seq = torch.stack(z_hat_predicted_list, dim=1)
                                                                # (B, T, N, N-1, 2)

        if oracle_z_seq is not None:
            assert oracle_z_seq.shape == z_hat_predicted_seq.shape
            z_hat_out_seq = oracle_z_seq
        else:
            z_hat_out_seq = z_hat_predicted_seq

        return hidden_seq, c_hat_seq, z_hat_out_seq

    def get_head_opp_predictions(
        self,
        hidden_seq: torch.Tensor,           # (B, T, N, 128)
    ) -> torch.Tensor:                      # (B, T, N, N-1, 2)
        """单独取 head_opp 预测 (Stage 1 oracle 模式下, 仍需 head_opp 训练).

        Pkg-05 trainer 在 Stage 1 时, forward 返回 oracle_z 作为下游输入,
        但 L_opp 训练 head_opp 需要 head_opp 的真实预测 — 用本方法取.
        """
        B, T, N, _ = hidden_seq.shape
        out = []
        for t in range(T):
            out.append(self._compute_z_hat(hidden_seq[:, t]))
        return torch.stack(out, dim=1)
