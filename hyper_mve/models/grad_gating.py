"""Belief gradient gating helper (Pkg-04 spec 04, v4 新增防线 5, Ch4.6.5).

前 ``num_warmup_steps`` 步 (默认 5000) 切断主任务 loss 反向到 BeliefNet 与
BeliefEncoder 的梯度路径, 让二者仅由独立的 L_belief loss (Pkg-03 belief_loss)
驱动训练, 避免 "chicken-and-egg" 噪声反馈循环.

双层 detach (review 修订 3):
    切断 1 (apply_raw): raw heads (c_hat, z_hat) -> 阻 main loss 反向到 BeliefNet GRU/heads
    切断 2 (apply_ctx): ctx_aug 内 belief 子段 [48:80] -> 阻 main loss 反向到 BeliefEncoder

L_belief loss 独立路径不受 gating 影响 (trainer 直接对 belief_net 参数 backward).

设计要点 (D3): 用显式 .detach() 而非 nn.Hook (前向可读可调试, 与计算图语义直接对应).
"""
from __future__ import annotations

import torch


class BeliefGradGating:
    """Belief gradient gating helper (无可学参数, model.forward 内部 helper)."""

    def __init__(self, num_warmup_steps: int = 5000):
        """
        Args:
            num_warmup_steps: 前多少步切断 belief 梯度.
                从 cfg.train.belief_grad_gating_steps 读取 (默认 5000).
        """
        self.num_warmup_steps = num_warmup_steps

    def apply_raw(
        self,
        c_hat: torch.Tensor,
        z_hat: torch.Tensor,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """切断 1: raw heads (c_hat, z_hat) -> 阻 main loss 反向到 BeliefNet GRU/heads.

        Args:
            c_hat: (B,) or (B, N) sigmoid 输出
            z_hat: (B, N-1, 2) or (B, N, N-1, 2) softmax 输出
            step:  当前 global_step (由 HyperMuZeroModel.update_step 提供)
        Returns:
            (c_hat_out, z_hat_out): step < num_warmup_steps 时 .detach(), 否则透传.
        """
        if step < self.num_warmup_steps:
            return c_hat.detach(), z_hat.detach()
        return c_hat, z_hat

    def apply_ctx(
        self,
        ctx_aug: torch.Tensor,
        step: int,
        belief_slice: tuple[int, int] = (48, 80),
    ) -> torch.Tensor:
        """切断 2: ctx_aug 内 belief 子段 -> 阻 main loss 反向到 BeliefEncoder.

        ctx_aug 内部结构: [c_ctx (0:16), role (16:48), belief_vec (48:80)].
        仅对 [48:80] 子段 .detach(), 保留 c_ctx + role 路径梯度.

        Args:
            ctx_aug:      (..., 80) ctx_aug 张量
            step:         当前 global_step
            belief_slice: (start, end) 默认 (48, 80) 对应 belief_vec 段
        Returns:
            ctx_aug_gated: 同 shape; step < num_warmup_steps 时 belief 子段已 .detach()
        """
        if step >= self.num_warmup_steps:
            return ctx_aug

        start, end = belief_slice
        return torch.cat([
            ctx_aug[..., :start],
            ctx_aug[..., start:end].detach(),
            ctx_aug[..., end:],
        ], dim=-1)

    # 兼容旧调用名 (spec 02 §2.2 内调用 .apply 仍可用, 等价于 apply_raw)
    apply = apply_raw

    @property
    def is_gating_active(self) -> bool:
        """便于 debug; 是否生效由 apply(step) 决定 (本类不持有 step 状态)."""
        return True
