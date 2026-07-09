"""HyperMuZero Model v5 (Pkg-09 amendment; base: Pkg-04 spec 02).

======================================================================
v5 API amendment (Pkg-09) — 6 个对外稳定 API (原 spec 08 §1 的 7-API 锁定
经本修正案变更; 见 sdd/pkg-09-dynamic-relations/design.md):

    update_step / set_context_subjective /
    encode / transition / predict_reward / predict

变更记录:
    - ``set_context_objective`` 删除: c_t 随物理再生参数一并移除, 客观通路
      不再有随 context 变化的输入. transition 改为普通共享 SGD 模块
      (models/transition_net.TransitionNet), 视角不变性由权重共享保证.
    - ``set_context_subjective(agent_id, row_i, belief)``: cap 槽位改为
      agent 自己的关系行 row_i (B, N-1) — per-episode 采样的 row 无法经 cfg
      传递, 必须走 API 实参 (v4 的 own type 从 cfg.env.type_assignment 解析,
      v5 无静态类型); belief 由 (c_hat, z_hat) 二元组改为单张量 regime 后验
      g_hat_i (B, |G|).
    - ctx_aug 80 -> 64 (c_ctx 路删除): [role (0:32) | belief (32:64)].
======================================================================

不对外暴露 forward(...) / compute_losses(...) (spec 02 §5.4): trainer 自管 loss 组装.

实施要点:
    - set_context_subjective 不调用 TriContextEncoder.forward (其形状契约是
      N-agent batch); 改为直接复用其 sub-encoders (role_encoder/belief_encoder
      + ln_role/ln_belief).
    - 双层 belief 梯度门控: apply_raw (raw head) + apply_ctx (ctx_aug belief 子段).
    - Self-Info 严格性 (C11/D6): row_i shape 严格断言 (B, N-1) — agent i 只被
      喂它自己的行 (worker 端 row-i-only-for-agent-i 纪律); 他人行/regime 仅由
      BeliefNet 推断.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from hyper_mve.models.tri_context_encoder import TriContextEncoder
from hyper_mve.models.belief_net import BeliefNet
from hyper_mve.models.hyper_network import DualHyperNetwork
from hyper_mve.models.functional_nets import (
    FunctionalRewardHead,
    FunctionalPredictionNet,
)
from hyper_mve.models.transition_net import TransitionNet
from hyper_mve.models.representation_net import RepresentationNet
from hyper_mve.models.grad_gating import BeliefGradGating
from hyper_mve.schemas import get_regime_family


# cfg 字段完备性自检 (spec 08 §8, M4; v5 修订)
REQUIRED_MODEL_FIELDS = [
    "d_role", "d_belief", "d_ctx_aug",
    "d_id_emb", "d_row_emb",
    "latent_dim", "hidden_dim",
    "hyper_hidden_dims", "hyper_rew_hidden_dims",
    "rew_output_scale_init", "pred_output_scale_init",
    "hyper_gen_scope", "share_subjective_trunk",
    "hyper_output_rank", "lora_fc2_rank",
    "use_adaln", "adaln_residual_one_plus",
    "belief_gru_hidden", "proj_dim",
]
REQUIRED_TRAIN_FIELDS = [
    "belief_grad_gating_steps", "detach_pred_context", "unroll_K",
]


class HyperMuZeroModel(nn.Module):
    """v5 统一 HyperMuZero Model (Pkg-09)."""

    #: pkg-07 spec 02 §3 SB5 — backbone attribute names. Filtered out by
    #: :func:`hyper_mve.baselines.shared_backbones.count_conditioning_params`
    #: so the conditioning-subsystem parameter count is comparable across
    #: hyper + internal baseline variants.
    SHARED_BACKBONE_PREFIXES: tuple[str, ...] = (
        "rep_net", "belief_net", "tri_context_encoder",
    )

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # ====== cfg 字段完备性自检 (spec 08 §8) ======
        for field in REQUIRED_MODEL_FIELDS:
            assert hasattr(cfg.model, field), f"cfg.model missing field: {field}"
        for field in REQUIRED_TRAIN_FIELDS:
            assert hasattr(cfg.train, field), f"cfg.train missing field: {field}"

        # ====== C5 (v5 修订): ctx_aug_dim 硬约束 ======
        assert cfg.model.d_ctx_aug == 64, (
            f"d_ctx_aug={cfg.model.d_ctx_aug} != 64. "
            "v5 双路 ctx 总维度精确硬约束 (Pkg-09: role 32 + belief 32)."
        )

        self.N = cfg.env.N
        self.n_regimes = get_regime_family(cfg.env).size

        # ====== 模块组装 ======
        # RepNet (objective, 与 BeliefNet 独立 obs_encoder 并列)
        self.rep_net = RepresentationNet(cfg)

        # BeliefNet (Pkg-09: regime 后验)
        self.belief_net = BeliefNet(cfg.env, cfg.model)

        # TriContextEncoder (v5: role + belief 双路, 内部含各路 LN)
        self.tri_context_encoder = TriContextEncoder(cfg.env, cfg.model)

        # Functional nets (主观通路; v5: transition 不再 functional)
        latent_dim = cfg.model.latent_dim
        joint_action_dim = cfg.env.N * cfg.env.A
        hidden_dim = cfg.model.hidden_dim
        gen_scope = cfg.model.hyper_gen_scope
        # [lora_fc2] rank only takes effect for the lora_fc2 scope (film_head-derived).
        lora_rank = cfg.model.lora_fc2_rank if gen_scope == "lora_fc2" else None
        # v5: 客观转移 = 普通共享 SGD 模块 (视角不变性 = 权重共享).
        self.state_trans_net = TransitionNet(latent_dim, joint_action_dim, hidden_dim)
        self.reward_head = FunctionalRewardHead(latent_dim, joint_action_dim, hidden_dim, gen_scope, lora_rank)
        self.prediction_net = FunctionalPredictionNet(latent_dim, cfg.env.A, hidden_dim, gen_scope, lora_rank)

        # DualHyperNetwork (v5: 主观 rew/pred 两路生成)
        self.hyper_net = DualHyperNetwork(
            ctx_aug_dim=cfg.model.d_ctx_aug,
            rew_param_count=self.reward_head.generated_param_count,
            pred_param_count=self.prediction_net.generated_param_count,
            hidden_dims=cfg.model.hyper_hidden_dims,
            rew_hidden_dims=cfg.model.hyper_rew_hidden_dims,
            rew_output_scale_init=cfg.model.rew_output_scale_init,
            pred_output_scale_init=cfg.model.pred_output_scale_init,
            detach_pred_context=cfg.train.detach_pred_context,
            rew_output_groups=self.reward_head.gen_groups,
            pred_output_groups=self.prediction_net.gen_groups,
            share_subjective_trunk=cfg.model.share_subjective_trunk,
            hyper_output_rank=cfg.model.hyper_output_rank,
        )

        # Belief gradient gating helper (spec 04)
        self.grad_gating = BeliefGradGating(
            num_warmup_steps=cfg.train.belief_grad_gating_steps,
        )

        # belief_vec 段在 ctx_aug 中的切片 [d_role : d_ctx_aug] = [32:64]
        self._belief_slice = (cfg.model.d_role, cfg.model.d_ctx_aug)

        # ====== 内部状态 (per-call 缓存) ======
        self._step: int = 0
        self._theta_rew: Optional[torch.Tensor] = None     # 主观, 当前 agent
        self._theta_pred: Optional[torch.Tensor] = None    # 主观, 当前 agent
        self._current_agent_id: Optional[int] = None       # 调试用

    # ====================================================================
    # API 1: update_step (Q4)
    # ====================================================================

    def update_step(self, global_step: int) -> None:
        """更新内部 step 计数器, 用于 belief grad gating 阈值判断.

        必须在每个 train_step 起点调用一次. worker 端不必调用 (永远 inference,
        梯度门控对它无意义). 默认 _step=0 -> 门控永远启用.
        """
        self._step = global_step

    # ====================================================================
    # API 2: set_context_subjective (v5 amendment)
    # ====================================================================

    def set_context_subjective(
        self,
        agent_id: int,
        row_i: torch.Tensor,                # (B, N-1) own relationship row w_i·
        belief: torch.Tensor,               # (B, |G|) regime posterior g_hat_i
    ) -> None:
        """计算并缓存当前 agent 的 theta_rew^i / theta_pred^i.

        Args:
            agent_id: 当前 agent 索引 in {0, ..., N-1}
            row_i:    (B, N-1) agent 自己的对角线外关系行 (升序 j 跳过 self,
                      与 obs row 块同序). Self-Info: 只能传 agent_id 自己的行.
            belief:   (B, |G|) BeliefNet head_regime 输出 (softmax 后验)
        """
        # row_i shape 硬约束 (C11 第一层防御, v5 形式)
        assert row_i.dim() == 2 and row_i.shape[-1] == self.N - 1, (
            f"row_i.shape must be (B, {self.N - 1}), got {tuple(row_i.shape)}. "
            "v5 row 是对角线外 (N-1,) 关系行; 传入其他宽度疑似信息泄漏 (C11)."
        )
        assert belief.dim() == 2 and belief.shape[-1] == self.n_regimes, (
            f"belief.shape must be (B, {self.n_regimes}), got {tuple(belief.shape)}."
        )

        B = row_i.shape[0]
        device = row_i.device
        tce = self.tri_context_encoder

        # ====== 切断 1: raw head (阻反向到 BeliefNet GRU/head) ======
        g_hat = self.grad_gating.apply_raw(belief, self._step)

        # ====== 单-agent ctx_aug 拼装 (不走 N-agent forward) ======
        agent_ids_one = torch.full((B, 1), int(agent_id), dtype=torch.long, device=device)
        role = tce.role_encoder(agent_ids_one, row_i.unsqueeze(1))   # (B, 1, 32)
        role = tce.ln_role(role).squeeze(1)                          # (B, 32)

        belief_vec = tce.belief_encoder(g_hat.unsqueeze(1))          # (B, 1, 32)
        belief_vec = tce.ln_belief(belief_vec).squeeze(1)            # (B, 32)

        # concat 双路 -> ctx_aug (B, 64)
        ctx_aug = torch.cat([role, belief_vec], dim=-1)

        # ====== 切断 2: ctx_aug belief 子段 (阻反向到 BeliefEncoder 投影) ======
        ctx_aug = self.grad_gating.apply_ctx(ctx_aug, self._step, belief_slice=self._belief_slice)

        # 生成 theta_rew^i / theta_pred^i
        self._theta_rew, self._theta_pred = self.hyper_net.forward_subjective(ctx_aug)
        self._current_agent_id = agent_id

    # ====================================================================
    # API 3-6: encode / transition / predict_reward / predict
    # ====================================================================

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """obs (B, N, obs_dim) -> s (B, latent_dim) via RepNet (objective)."""
        return self.rep_net(obs)

    def transition(self, s: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """(s, joint_action_onehot) -> s_next via TransitionNet (v5 普通共享模块).

        无需任何 set_context_* 前置调用; 权重共享保证视角不变性.

        Args:
            s:      (B, latent_dim)
            action: (B, joint_action_dim) one-hot N*A flat
        Returns:
            s_next: (B, latent_dim)  (内部 return s + delta_s)
        """
        return self.state_trans_net(s, action)

    def predict_reward(self, s: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """(s, action) -> r via RewardHead (用当前 agent 的 theta_rew^i).

        Stateful: 返回 r 对应最后一次 set_context_subjective 的 agent_id.
        """
        assert self._theta_rew is not None, (
            "predict_reward() called before set_context_subjective()."
        )
        return self.reward_head(s, action, self._theta_rew)

    def predict(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """s -> (policy_logits, value) via PredictionNet (用当前 agent 的 theta_pred^i).

        Stateful: 返回 (pi, v) 对应最后一次 set_context_subjective 的 agent_id.
        """
        assert self._theta_pred is not None, (
            "predict() called before set_context_subjective()."
        )
        return self.prediction_net(s, self._theta_pred)

    # ====================================================================
    # 诊断辅助 (非 6-API; 只读已缓存的 per-agent 生成参数, 用于角色 cos 相似性)
    # ====================================================================

    def current_subjective_thetas(self) -> tuple[torch.Tensor, torch.Tensor]:
        """返回最近一次 set_context_subjective 缓存的 (theta_rew, theta_pred).

        供 loss_composition 的超网络角色诊断读取 (避免外部直接碰 _private 字段).
        每个张量 shape (B, param_count); 对应最后一次 set_context_subjective 的 agent.
        """
        assert self._theta_rew is not None and self._theta_pred is not None, (
            "current_subjective_thetas() called before set_context_subjective()."
        )
        return self._theta_rew, self._theta_pred

    # ====================================================================
    # Planner θ-cache fast path (非 6-API; 仅供 MVEPlanner 复用同一 context 的
    # theta, 避免 K-step rollout 内每步重新跑 hypernet).
    # ====================================================================
    # 数值等价保证 (与 set_context_subjective 逐位一致):
    #   1. functional_nets 全程 per-row (bmm + per-sample LayerNorm/L2norm), 行间
    #      互不耦合, 所以"先在 batch B 生成 theta 再 repeat_interleave 到 B*k"与
    #      "先 repeat_interleave 上下文再生成 theta"输出逐位相同.
    #   2. hypernet.forward_subjective 在 eval 下为确定性映射 (无 dropout /
    #      无 RNG); 同一输入恒得同一 theta.
    #   3. MVEPlanner 的 batch 扩张是纯 repeat_interleave (B -> B*spa -> B*M),
    #      所以 export 出的 base-batch theta 经 repeat_interleave 即为
    #      set_context_subjective 在扩张 batch 上会生成的 theta.
    # v5 注: transition 是普通共享模块, 天然 batch 无关 — 客观 θ-cache 不复存在.

    def install_subjective_theta(
        self, agent_id: int, theta_rew: torch.Tensor, theta_pred: torch.Tensor
    ) -> None:
        """直接装载预生成的 theta_rew^i / theta_pred^i, 不重跑 hyper_rew/pred.

        与 set_context_subjective(agent_id, row_i, belief) 语义等价 (当传入的
        theta 正是它对相同 (tiled) 上下文的输出时, 逐位一致).
        """
        self._theta_rew = theta_rew
        self._theta_pred = theta_pred
        self._current_agent_id = agent_id
