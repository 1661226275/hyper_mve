"""HyperMuZero Model v2 (Pkg-04 spec 02, Ch4.3 + 4.4).

v4 统一 HyperMuZeroModel, 废弃 v4.7 Oracle/Infer 双类. BeliefNet (Pkg-03) 提供
belief; set_context 两步分离 (objective + subjective); update_step 独立方法管理
belief grad gating 状态.

7 个对外稳定 API (spec 08 §1 锁定):
    update_step / set_context_objective / set_context_subjective /
    encode / transition / predict_reward / predict

不对外暴露 forward(...) / compute_losses(...) (spec 02 §5.4): trainer 自管 loss 组装.

实施要点 (review 修订 2/3):
    - set_context_subjective 不调用 TriContextEncoder.forward (其形状契约是 N-agent
      batch: z_hat == (B, N, N-1, 2), 与 single-agent (B, N-1, 2) 不兼容 -- N=1 时
      N-1=0 断言失败). 改为直接复用其 sub-encoders (role_encoder/belief_encoder +
      ln_role/ln_belief), c_ctx 复用 set_context_objective 缓存值.
    - 双层 belief 梯度门控: apply_raw (raw heads) + apply_ctx (ctx_aug belief 子段).
    - Self-Info 严格性 (C11/D6): own type 取自 cfg.env.type_assignment[agent_id],
      不从 env.info; cap_i shape 严格断言 (B, 4) 拒绝 type leak.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from hyper_mve.models.tri_context_encoder import TriContextEncoder
from hyper_mve.models.belief_net import BeliefNet
from hyper_mve.models.hyper_network import DualHyperNetwork
from hyper_mve.models.functional_nets import (
    FunctionalStateTransNet,
    FunctionalRewardHead,
    FunctionalPredictionNet,
)
from hyper_mve.models.representation_net import RepresentationNet
from hyper_mve.models.grad_gating import BeliefGradGating


# cfg 字段完备性自检 (spec 08 §8, M4)
REQUIRED_MODEL_FIELDS = [
    "d_c", "d_role", "d_belief", "d_ctx_aug",
    "d_id_emb", "d_type_emb", "d_cap_emb", "d_belief_proj",
    "latent_dim", "hidden_dim",
    "hyper_hidden_dims", "hyper_rew_hidden_dims",
    "trans_output_scale_init", "rew_output_scale_init", "pred_output_scale_init",
    "hyper_gen_scope",
    "use_adaln", "adaln_residual_one_plus", "state_trans_residual",
    "belief_gru_hidden", "belief_pool", "proj_dim",
]
REQUIRED_TRAIN_FIELDS = [
    "belief_grad_gating_steps", "detach_pred_context", "unroll_K",
]


class HyperMuZeroModel(nn.Module):
    """v4 统一 HyperMuZero Model (Ch4.3 + 4.4)."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # ====== cfg 字段完备性自检 (spec 08 §8) ======
        for field in REQUIRED_MODEL_FIELDS:
            assert hasattr(cfg.model, field), f"cfg.model missing field: {field}"
        for field in REQUIRED_TRAIN_FIELDS:
            assert hasattr(cfg.train, field), f"cfg.train missing field: {field}"

        # ====== C5: ctx_aug_dim 硬约束 ======
        assert cfg.model.d_ctx_aug == 80, (
            f"d_ctx_aug={cfg.model.d_ctx_aug} != 80. "
            "v4 三路 ctx 总维度精确硬约束 (Ch4.2.4)."
        )

        # ====== 模块组装 ======
        # RepNet (objective, 与 BeliefNet 独立 obs_encoder 并列)
        self.rep_net = RepresentationNet(cfg)

        # BeliefNet (Pkg-03)
        self.belief_net = BeliefNet(cfg.env, cfg.model)

        # TriContextEncoder (Pkg-03, 内部含 CEncoder/RoleEncoder/BeliefEncoder + 三路 LN)
        self.tri_context_encoder = TriContextEncoder(cfg.env, cfg.model)

        # Functional nets (positional cfg-derived args, 见 plan 决议 4: 保留位置签名)
        latent_dim = cfg.model.latent_dim
        joint_action_dim = cfg.env.N * cfg.env.A
        hidden_dim = cfg.model.hidden_dim
        gen_scope = cfg.model.hyper_gen_scope
        self.state_trans_net = FunctionalStateTransNet(latent_dim, joint_action_dim, hidden_dim, gen_scope)
        self.reward_head = FunctionalRewardHead(latent_dim, joint_action_dim, hidden_dim, gen_scope)
        self.prediction_net = FunctionalPredictionNet(latent_dim, cfg.env.A, hidden_dim, gen_scope)

        # DualHyperNetwork v2 (spec 01). 按 generated_param_count 定尺寸 (FULL 时 ==
        # total_params; film_head 时只生成 FiLM γ/β + 头), 并把分组边界传给各 HyperNetMLP
        # (FULL 时 gen_groups=None -> 维持整段 L2 norm, 行为不变).
        self.hyper_net = DualHyperNetwork(
            c_ctx_dim=cfg.model.d_c,
            ctx_aug_dim=cfg.model.d_ctx_aug,
            trans_param_count=self.state_trans_net.generated_param_count,
            rew_param_count=self.reward_head.generated_param_count,
            pred_param_count=self.prediction_net.generated_param_count,
            hidden_dims=cfg.model.hyper_hidden_dims,
            rew_hidden_dims=cfg.model.hyper_rew_hidden_dims,
            trans_output_scale_init=cfg.model.trans_output_scale_init,
            rew_output_scale_init=cfg.model.rew_output_scale_init,
            pred_output_scale_init=cfg.model.pred_output_scale_init,
            detach_pred_context=cfg.train.detach_pred_context,
            trans_output_groups=self.state_trans_net.gen_groups,
            rew_output_groups=self.reward_head.gen_groups,
            pred_output_groups=self.prediction_net.gen_groups,
        )

        # Belief gradient gating helper (spec 04)
        self.grad_gating = BeliefGradGating(
            num_warmup_steps=cfg.train.belief_grad_gating_steps,
        )

        # belief_vec 段在 ctx_aug 中的切片 [d_c + d_role : d_ctx_aug] = [48:80]
        self._belief_slice = (cfg.model.d_c + cfg.model.d_role, cfg.model.d_ctx_aug)

        # ====== 内部状态 (per-call 缓存) ======
        self._step: int = 0
        self._theta_state: Optional[torch.Tensor] = None   # 客观, 所有 agent 共享
        self._theta_rew: Optional[torch.Tensor] = None     # 主观, 当前 agent
        self._theta_pred: Optional[torch.Tensor] = None    # 主观, 当前 agent
        self._cached_c_ctx: Optional[torch.Tensor] = None  # objective 缓存, 供 subjective 复用
        self._current_agent_id: Optional[int] = None       # 调试用
        self._has_objective: bool = False                  # 顺序断言用

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
    # API 2: set_context_objective (Q3 + D1 + D4)
    # ====================================================================

    def set_context_objective(self, c_t: torch.Tensor) -> None:
        """C1: 仅接 c_t, 计算并缓存 theta_state (所有 agent 共享).

        必须在每次 K-step unroll 起点调用一次, 在 set_context_subjective 之前.

        Args:
            c_t: (B,) or (B, 1) float32 共享 context scalar
        """
        c_ctx = self.tri_context_encoder.forward_c_ctx_only(c_t)  # (B, 16) 已 LN
        self._theta_state = self.hyper_net.forward_trans(c_ctx)   # (B, trans_param_count)
        self._cached_c_ctx = c_ctx                                # 复用于 subjective
        self._has_objective = True
        # 清空 subjective 缓存 (强制重设)
        self._theta_rew = None
        self._theta_pred = None
        self._current_agent_id = None

    # ====================================================================
    # API 3: set_context_subjective (Q3 + D4 + D6)
    # ====================================================================

    def set_context_subjective(
        self,
        agent_id: int,
        cap_i: torch.Tensor,                       # (B, 4) RAW CapabilityVector
        belief: tuple[torch.Tensor, torch.Tensor], # (c_hat (B,), z_hat (B, N-1, 2))
    ) -> None:
        """C2 + D6: 计算并缓存当前 agent 的 theta_rew^i / theta_pred^i.

        必须在 set_context_objective 之后调用 (顺序断言 D4).

        Args:
            agent_id: 当前 agent 索引 in {0, ..., N-1}
            cap_i:    (B, 4) RAW CapabilityVector (RoleEncoder 内部自动 normalize)
            belief:   (c_hat (B,), z_hat (B, N-1, 2)) BeliefNet 输出 (Pkg-03)
        """
        # 顺序断言 (D4)
        assert self._has_objective, (
            "set_context_subjective() called before set_context_objective(). "
            "Pkg-04 spec 02 §3.4: 必须先 set_context_objective 后 set_context_subjective."
        )

        # cap_i shape 硬约束 (C11 第一层防御)
        assert cap_i.dim() == 2 and cap_i.shape[-1] == 4, (
            f"cap_i.shape must be (B, 4), got {tuple(cap_i.shape)}. "
            f"v4 CapabilityVector 是严格 4 元组 (eta, phi_fov, nu, zeta); "
            f"若传入 5 维 (B, 5) 等扩展形, 疑似含 type leak -- Self-Info 违规 (C11)."
        )

        B = cap_i.shape[0]
        device = cap_i.device
        tce = self.tri_context_encoder

        # ====== 切断 1: raw heads (阻反向到 BeliefNet GRU/heads) ======
        c_hat, z_hat = belief
        c_hat, z_hat = self.grad_gating.apply_raw(c_hat, z_hat, self._step)

        # ====== 单-agent ctx_aug 拼装 (不走 N-agent forward, review 修订 2) ======
        # c_ctx: 复用 objective 缓存 (B, 16), 已 LN
        c_ctx = self._cached_c_ctx

        # role: 直接调 role_encoder (M=1 agent) + ln_role
        agent_ids_one = torch.full((B, 1), int(agent_id), dtype=torch.long, device=device)
        own_type = torch.full(
            (B, 1),
            int(self.cfg.env.type_assignment[agent_id].value),
            dtype=torch.long, device=device,
        )
        role = tce.role_encoder(agent_ids_one, own_type, cap_i.unsqueeze(1))  # (B, 1, 32)
        role = tce.ln_role(role).squeeze(1)                                   # (B, 32)

        # belief: 直接调 belief_encoder 的 sub-layers (绕开 N-1 形状断言) + ln_belief
        be = tce.belief_encoder
        c_hat_proj = be.proj_c_hat(c_hat.reshape(B, 1, 1))      # (B, 1, 16)
        z_pooled = be.z_pool(z_hat.unsqueeze(1))                # (B, 1, N-1, 2) -> (B, 1, 2)
        z_pooled_proj = be.proj_z_pooled(z_pooled)              # (B, 1, 16)
        belief_vec = torch.cat([c_hat_proj, z_pooled_proj], dim=-1)  # (B, 1, 32)
        belief_vec = tce.ln_belief(belief_vec).squeeze(1)            # (B, 32)

        # concat 三路 -> ctx_aug (B, 80)
        ctx_aug = torch.cat([c_ctx, role, belief_vec], dim=-1)

        # ====== 切断 2: ctx_aug belief 子段 (阻反向到 BeliefEncoder 投影 MLP) ======
        ctx_aug = self.grad_gating.apply_ctx(ctx_aug, self._step, belief_slice=self._belief_slice)

        # 生成 theta_rew^i / theta_pred^i
        self._theta_rew, self._theta_pred = self.hyper_net.forward_subjective(ctx_aug)
        self._current_agent_id = agent_id

    # ====================================================================
    # API 4-7: encode / transition / predict_reward / predict
    # ====================================================================

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """obs (B, N, obs_dim) -> s (B, latent_dim) via RepNet (objective)."""
        return self.rep_net(obs)

    def transition(self, s: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """(s, joint_action_onehot) -> s_next via StateTransNet (用缓存 theta_state).

        Args:
            s:      (B, latent_dim)
            action: (B, joint_action_dim) one-hot N*A flat
        Returns:
            s_next: (B, latent_dim)  (C6: 内部 return s + delta_s)
        """
        assert self._theta_state is not None, (
            "transition() called before set_context_objective()."
        )
        return self.state_trans_net(s, action, self._theta_state)

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
    # 诊断辅助 (非 7-API; 只读已缓存的 per-agent 生成参数, 用于角色 cos 相似性)
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
