"""BeliefNet 三损失 + Oracle z 构造 (Pkg-03 spec 06 + spec 08 §4.3).

三损失 (Ch4.5.1-4.5.4):
    L_c   (MSE):  per-agent ĉ 与 oracle c_t 的均方误差.
    L_opp (CE):   per-agent 对手类型推断的 Oracle 交叉熵 (v4 关键改动).
    L_div (hinge):防 b_i^t 在 N 个 agent 间坍缩为常数.

总损失: L = λ_c L_c + λ_opp L_opp + λ_div L_div, 默认 (1.0, 0.5, 0.01).

L_div 与论文 Ch4.5.3 `-Var` 形式有意偏差为 hinge variant (σ_target=0.1), 备案见
spec 06 §0: hinge 防 collapse 工程更稳, 与 GRU LayerNorm 不冲突.

build_oracle_z_seq (spec 08 §4.3, P9): 归属 Pkg-03, 与 l_opp / head_opp 共用同一
opp_indices 映射 (i, k) -> agent_id = (k if k < i else k + 1), 保证课程 Stage 1
oracle 注入与 L_opp Oracle 监督顺序一致.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F  # noqa: F401  (保留以备数值稳定 CE 替代实现)


def l_c(
    c_hat_seq: torch.Tensor,            # (B, T, N) sigmoid ∈ [0, 1]
    c_true_seq: torch.Tensor,           # (B, T) Oracle from env.info["c_true"]
    mask: Optional[torch.Tensor] = None,
                                        # (B, T) bool, 可选 (如 padding mask)
) -> torch.Tensor:                      # scalar (含 grad)
    """L_c MSE: per-agent ĉ 与 oracle c_t 的均方误差.

    c_true_seq 来自 env.info["c_true"], Pkg-02 暴露的 Oracle 信号.
    所有 N 个 agent 看到同一 c_t (Harsanyi 共同知识), 故 c_true_seq shape (B, T).

    **Broadcast 约定 (P8 修订)**:
        c_true_seq shape (B, T) — 所有 N agent 看到同一 c_t
        c_hat_seq shape  (B, T, N) — per-agent BeliefNet 推断
        内部 broadcast c_true.unsqueeze(-1) → (B, T, 1) → expand 到 (B, T, N)

        标量等价公式:
            L_c = (1 / (B*T*N)) * Σ_{b,t,i} (c_hat[b,t,i] - c_true[b,t])²

    Args:
        c_hat_seq: BeliefNet head_c 输出 (sigmoid, per-agent)
        c_true_seq: Oracle 标签 (per-batch, per-step), shape (B, T) **不含 N 维**
        mask: 可选 (B, T) bool, 仅对 mask==True 的 (b, t) 计算损失

    Returns:
        scalar tensor
    """
    B, T, N = c_hat_seq.shape
    assert c_true_seq.shape == (B, T), (
        f"c_true_seq shape {c_true_seq.shape}, expected (B={B}, T={T}). "
        f"c_true 是共同知识 scalar, **不含** N 维; 由本函数内部 broadcast 到 (B, T, N)."
    )
    # broadcast c_true 到 (B, T, N)
    c_true_expanded = c_true_seq.unsqueeze(-1).expand(B, T, N)

    sq_err = (c_hat_seq - c_true_expanded) ** 2  # (B, T, N)

    if mask is not None:
        mask_expanded = mask.unsqueeze(-1).expand(B, T, N).float()
        sq_err = sq_err * mask_expanded
        denom = mask_expanded.sum().clamp(min=1.0)
    else:
        denom = float(B * T * N)

    return sq_err.sum() / denom


def l_opp(
    z_hat_seq: torch.Tensor,            # (B, T, N, N-1, 2) softmax 概率
    types_true: torch.Tensor,           # (B, T, N) int64 (AgentType.value)
                                        # 注: 每 step 都传, 但 episode 内不变
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """L_opp v4 Oracle CE: per-agent 对手类型推断的交叉熵损失.

    v4 关键改动 (vs v3):
        v3: head_opp 输出 d_z 维动作预测, 自监督训练
        v4: head_opp 输出 2 维类型 softmax, **Oracle 监督**

    ⚠️ 关键 (P3): j → agent_id 映射是 silent bug 入口!
    ===========================================================================
    z_hat 顺序约定 (与 Pkg-01 TimeStepRecord z_hat 一致, 硬约束):
        对 agent i, 第 k 个对手 (k ∈ {0, ..., N-2}) 对应:
            opp_agent_id(i, k) = k if k < i else k + 1
        即按 agent_id 升序跳过 self.

    向量化实现:
        opp_indices[i, k] = (k if k < i else k + 1)   # (N, N-1) 静态索引
        opp_labels = gather(types_true, dim=-1, index=opp_indices)
        log_probs = log(z_hat_seq + eps)
        nll = -gather(log_probs, dim=-1, index=opp_labels.unsqueeze(-1)).squeeze(-1)
        L_opp = mean(nll)

    ⚠️ 错位后果: 若 opp_indices 构造错误, BeliefNet 会在错位标签上 CE 反向,
    学到"agent i 看到 agent i 自己的 type", Pkg-05 主观通路收到错乱 ẑ → 训练崩溃.

    Args:
        z_hat_seq: BeliefNet head_opp 输出 (softmax 概率)
        types_true: Oracle 标签 (per-batch, per-step, per-agent 真实 type)
        mask: 可选 (B, T) bool

    Returns:
        scalar tensor
    """
    B, T, N, N_minus_1, num_classes = z_hat_seq.shape
    assert N_minus_1 == N - 1
    assert num_classes == 2
    assert types_true.shape == (B, T, N)

    device = z_hat_seq.device

    # opp_indices: (N, N-1), opp_indices[i, k] = agent_id of opp k for agent i
    opp_indices = torch.zeros(N, N - 1, dtype=torch.long, device=device)
    for i in range(N):
        opp_list = [j for j in range(N) if j != i]
        opp_indices[i] = torch.tensor(opp_list, dtype=torch.long, device=device)

    # gather: opp_labels[b, t, i, k] = types_true[b, t, opp_indices[i, k]]
    types_expand = types_true.unsqueeze(2).expand(B, T, N, N)  # (B, T, N, N)
    opp_indices_expand = opp_indices.unsqueeze(0).unsqueeze(0).expand(B, T, N, N - 1)
    opp_labels = torch.gather(types_expand, dim=-1, index=opp_indices_expand)
    # opp_labels shape: (B, T, N, N-1) int64

    # CE on softmax probs: -log(z_hat[label])
    eps = 1e-8
    log_probs = torch.log(z_hat_seq + eps)         # (B, T, N, N-1, 2)

    opp_labels_expand = opp_labels.unsqueeze(-1)   # (B, T, N, N-1, 1)
    nll = -torch.gather(log_probs, dim=-1, index=opp_labels_expand).squeeze(-1)
    # nll shape: (B, T, N, N-1)

    if mask is not None:
        mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).expand(B, T, N, N - 1).float()
        nll = nll * mask_expanded
        denom = mask_expanded.sum().clamp(min=1.0)
    else:
        denom = float(B * T * N * (N - 1))

    return nll.sum() / denom


def l_div(
    hidden_seq: torch.Tensor,           # (B, T, N, h_dim) GRU hidden state
    target_std: float = 0.1,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """L_div variance hinge: 防 BeliefNet b_i^t 在 N 个 agent 间坍缩为常数.

    形式:
        loss = max(0, target_std² - Var_i(b_i^t))
        Var_i 在 batch 内 N 个 agent 上对每维度求方差, 再 mean over dims & steps.

    Args:
        hidden_seq: BeliefNet GRU hidden 序列
        target_std: σ_target (默认 0.1)
        mask: 可选 (B, T) bool

    Returns:
        scalar tensor
    """
    B, T, N, h_dim = hidden_seq.shape
    target_var = target_std ** 2

    # 在 dim=2 (N 维度) 上计算 batch-wise variance: (B, T, h_dim)
    var_per_dim = hidden_seq.var(dim=2, unbiased=False)

    # mean over h_dim: (B, T)
    var_avg = var_per_dim.mean(dim=-1)

    # hinge: max(0, target_var - var_avg)
    hinge = (target_var - var_avg).clamp(min=0)    # (B, T)

    if mask is not None:
        hinge = hinge * mask.float()
        denom = mask.float().sum().clamp(min=1.0)
    else:
        denom = float(B * T)

    return hinge.sum() / denom


def belief_loss(
    c_hat_seq: torch.Tensor,            # (B, T, N) sigmoid
    z_hat_seq: torch.Tensor,            # (B, T, N, N-1, 2) softmax
    hidden_seq: torch.Tensor,           # (B, T, N, 128)
    c_true_seq: torch.Tensor,           # (B, T)
    types_true: torch.Tensor,           # (B, T, N) int64
    weights: tuple[float, float, float] = (1.0, 0.5, 0.01),
                                        # (λ_c, λ_opp, λ_div)
    div_target_std: float = 0.1,
    mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """BeliefNet 总损失 (Ch4.5.4 + 4.5.5).

    Returns:
        total: scalar (含 grad), 已加权
        breakdown: dict 含 "l_c", "l_opp", "l_div", "total" (detached, 仅 logging)
    """
    lam_c, lam_opp, lam_div = weights

    loss_c = l_c(c_hat_seq, c_true_seq, mask=mask)
    loss_opp = l_opp(z_hat_seq, types_true, mask=mask)
    loss_div = l_div(hidden_seq, target_std=div_target_std, mask=mask)

    total = lam_c * loss_c + lam_opp * loss_opp + lam_div * loss_div

    breakdown = {
        "l_c": loss_c.detach(),
        "l_opp": loss_opp.detach(),
        "l_div": loss_div.detach(),
        "total": total.detach(),
    }

    return total, breakdown


def build_oracle_z_seq(
    types_true: torch.Tensor,           # (B, T, N) int64 (AgentType.value ∈ {0, 1})
) -> torch.Tensor:                      # (B, T, N, N-1, 2) one-hot Δ^2
    """构造 Stage 1 / Stage 2 anneal 用的 Oracle z 序列 (spec 08 §4.3, P9).

    顺序约定 (硬约束):
        z[b, t, i, k] 对应 agent_id = (k if k < i else k + 1)
        z[b, t, i, k, types_true[b, t, agent_id]] = 1.0; 其他位置 = 0.0

    与 spec 05 head_opp 输出 / spec 06 l_opp 标签 gather / Pkg-01 TimeStepRecord
    z_hat 字段三处使用同一映射, 保证课程 Stage 1 oracle 注入与 L_opp Oracle 监督一致.

    Args:
        types_true: Oracle types from env.info["types"] (per-batch, per-step,
                    per-agent). episode 内不变, 但每 step 仍传 (与 z_hat_seq shape 对齐).

    Returns:
        oracle_z: (B, T, N, N-1, 2) one-hot 概率分布 ∈ Δ^2
    """
    B, T, N = types_true.shape
    device = types_true.device

    # 1. 构造 opp_indices: (N, N-1) — 与 spec 06 l_opp 完全一致
    opp_indices = torch.zeros(N, N - 1, dtype=torch.long, device=device)
    for i in range(N):
        opp_list = [j for j in range(N) if j != i]
        opp_indices[i] = torch.tensor(opp_list, dtype=torch.long, device=device)

    # 2. gather: opp_labels[b, t, i, k] = types_true[b, t, opp_indices[i, k]]
    types_expand = types_true.unsqueeze(2).expand(B, T, N, N)
    opp_indices_expand = opp_indices.unsqueeze(0).unsqueeze(0).expand(B, T, N, N - 1)
    opp_labels = torch.gather(types_expand, dim=-1, index=opp_indices_expand)
    # opp_labels shape: (B, T, N, N-1) int64 ∈ {0, 1}

    # 3. one-hot 编码: (B, T, N, N-1) → (B, T, N, N-1, 2)
    oracle_z = torch.nn.functional.one_hot(opp_labels, num_classes=2).float()

    return oracle_z
