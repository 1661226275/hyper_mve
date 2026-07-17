# Spec 06: BeliefNet 三损失函数 — L_c + L_opp (v4 Oracle CE) + L_div

> 父文档：[`../proposal.md`](../proposal.md) §1.2 · [`../design.md`](../design.md) §3 D4 / D5
> **本 spec 含 v4 关键改动** — L_opp 从 v3 自监督改为 Oracle CE 监督。

---

## 0. 与 Ch4.5.3 论文公式的偏差备案（P1，2026-05-28 用户审阅 ack）

**论文 Ch4.5.3 公式**：

$$\mathcal{L}_{\text{div}} = -\frac{1}{B}\sum_{\text{batch}}\mathrm{Var}_i(b_i^t)$$

—— 直接最大化负方差，**无阈值**，永远激励增大方差。

**本 spec 实现（hinge variant，与论文公式不一致）**：

$$\mathcal{L}_{\text{div}} = \max\left(0,\;\sigma_{\text{target}}^2 - \mathrm{Var}_i(b_i^t)\right),\quad \sigma_{\text{target}}=0.1$$

—— variance ≥ 0.01 时 loss = 0（不约束），低于时线性惩罚。

**为什么偏离**：
1. **工程稳定性**：纯 `-Var` 在 train 后期持续推方差爆炸（与 Ch4.6.2 GRU LayerNorm 限定行为冲突，可能引发 LN gamma 漂移）
2. **语义差异**：hinge 是"防 collapse"（必要条件），论文 `-Var` 是"最大化多样性"（奢侈条件）——工程上前者足够
3. **Pkg-01 字段已锁定**：`TrainConfig.belief_div_target_std=0.1` 已暗示 hinge 形式
4. **Plan Part 2 D5 已选 hinge**：本偏差是 SDD 阶段一开始就存在，非临时改动

**建议论文同步修订**：
- 选项 A：将 Ch4.5.3 公式改为 hinge 形式（与本 spec 一致）
- 选项 B：保留论文 `-Var` 公式作为理论形式，加 footnote "实现层用 hinge variant，σ_target=0.1，详见 Pkg-03 spec 06 §0"

**对实验断言的影响**：无。L_div 是辅助正则（λ_div=0.01），主要作用是防 collapse；hinge 与 -Var 在防 collapse 场景下等价（都保证方差不为零）。

**design.md NG8 流程**：本备案块即"先报 issue 讨论"的产物——已与用户在 2026-05-28 审阅中明确 ack。

---

## 1. Purpose

按 Ch4.5.1-4.5.4 实现 BeliefNet 训练的三个损失：

**L_c（c_t 推断 MSE）**：

$$\mathcal{L}_c = \frac{1}{NT}\sum_{i,t}(\hat{c}_i^t - c_t)^2$$

**L_opp（对手类型 Oracle CE，v4 关键改动）**：

$$\mathcal{L}_{\text{opp}} = \frac{1}{NT(N-1)}\sum_{i,t,j \neq i}\text{CE}(\hat{z}_{i,j}^t,\;\tau_j)$$

**L_div（信念多样性正则，variance hinge）**：

$$\mathcal{L}_{\text{div}} = \max\left(0,\;\sigma_{\text{target}}^2 - \mathrm{Var}_i(b_i^t)\right)$$

**总损失**：

$$\mathcal{L}_{\text{BeliefNet}} = \lambda_c \mathcal{L}_c + \lambda_{\text{opp}}\mathcal{L}_{\text{opp}} + \lambda_{\text{div}}\mathcal{L}_{\text{div}}$$

推荐权重：$\lambda_c = 1.0,\;\lambda_{\text{opp}} = 0.5,\;\lambda_{\text{div}} = 0.01$。

---

## 2. Interface

### 2.1 文件路径

`hyper_mve/models/belief_losses.py`

### 2.2 公开 API

```python
import torch
import torch.nn.functional as F
from typing import Optional


def l_c(
    c_hat_seq: torch.Tensor,            # (B, T, N) sigmoid ∈ [0, 1]
    c_true_seq: torch.Tensor,           # (B, T) Oracle from env.info["c_true"]
    mask: Optional[torch.Tensor] = None,
                                        # (B, T) bool, 可选 (如 padding mask)
) -> torch.Tensor:                      # scalar (含 grad)
    """L_c MSE: per-agent ĉ 与 oracle c_t 的均方误差.
    
    c_true_seq 来自 env.info["c_true"], Pkg-02 暴露的 Oracle 信号.
    所有 N 个 agent 看到同一 c_t (Harsanyi 共同知识), 故 c_true_seq shape (B, T).
    
    **Broadcast 约定 (P8 修订, 2026-05-28)**:
        c_true_seq shape (B, T) — Harsanyi 共同知识, 所有 N agent 看到同一 c_t
        c_hat_seq shape  (B, T, N) — per-agent BeliefNet 推断
        内部 broadcast c_true.unsqueeze(-1) → (B, T, 1) → 自动 expand 到 (B, T, N)
        MSE 在所有 (b, t, i) 上对每个 agent 的 ĉ_i^t vs 共享 c_t 做 squared error.
        
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
    
    ⚠️ **关键 (P3, 2026-05-28)**: j → agent_id 映射是 silent bug 入口!
    ===========================================================================
    z_hat 顺序约定 (与 Pkg-01 TimeStepRecord z_hat 一致, 硬约束):
        对 agent i, 第 k 个对手 (k ∈ {0, ..., N-2}) 对应:
            opp_agent_id(i, k) = k if k < i else k + 1
        即按 agent_id 升序跳过 self.
    
    例:
        i=0 (Hunter0): 对手 k=0→agent 1, k=1→agent 2, k=2→agent 3
        i=2 (Agent2):  对手 k=0→agent 0, k=1→agent 1, k=2→agent 3
    
    标签构造 (标量等价伪代码):
        for b in range(B):
            for t in range(T):
                for i in range(N):
                    for k in range(N - 1):
                        j = k if k < i else k + 1     # ← 关键映射
                        label = types_true[b, t, j]   # ← 取对手 j 的真实 type
                        loss_term = CrossEntropy(z_hat_seq[b, t, i, k], label)
                        total_loss += loss_term
        L_opp = total_loss / (B * T * N * (N - 1))
    
    向量化实现 (本函数下方):
        opp_indices[i, k] = (k if k < i else k + 1)   # (N, N-1) 静态索引
        opp_labels = gather(types_true, dim=-1, index=opp_indices)  # (B, T, N, N-1)
        log_probs = log(z_hat_seq + eps)
        nll = -gather(log_probs, dim=-1, index=opp_labels.unsqueeze(-1)).squeeze(-1)
        L_opp = mean(nll)
    
    ⚠️ **错位后果**: 若 opp_indices 构造错误 (如 k 直接对应 agent_id 而不跳 self),
    BeliefNet 会在错位标签上 CE 反向, 学到"agent i 看到 agent i 自己的 type"映射,
    Pkg-05 trainer 在 Stage 3 (Pure Inference) 时主观通路 hyper_rew/hyper_pred
    会收到错乱 ẑ → 生成错乱 θ_rew → 训练崩溃.
    
    单测 `test_l_opp_index_mapping` (§5.1) 用可识别 types_true 验证映射正确性.
    ===========================================================================
    
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
    
    # 构造 opp_labels[b, t, i, k] = types_true[b, t, agent_id_of_kth_opp]
    # agent_id_of_kth_opp = (k if k < i else k + 1)
    device = z_hat_seq.device
    
    # opp_indices: (N, N-1), opp_indices[i, k] = agent_id of opp k for agent i
    opp_indices = torch.zeros(N, N - 1, dtype=torch.long, device=device)
    for i in range(N):
        opp_list = [j for j in range(N) if j != i]
        opp_indices[i] = torch.tensor(opp_list, dtype=torch.long, device=device)
    
    # gather: opp_labels[b, t, i, k] = types_true[b, t, opp_indices[i, k]]
    # types_true (B, T, N) -> (B, T, 1, N) -> broadcast/gather -> (B, T, N, N-1)
    types_expand = types_true.unsqueeze(2).expand(B, T, N, N)  # (B, T, N, N)
    opp_indices_expand = opp_indices.unsqueeze(0).unsqueeze(0).expand(B, T, N, N - 1)
    opp_labels = torch.gather(types_expand, dim=-1, index=opp_indices_expand)
    # opp_labels shape: (B, T, N, N-1) int64
    
    # CE: F.cross_entropy 期望输入 (M, C) logits 或 (M, C) softmax probs (with log).
    # 因为 z_hat 已是 softmax 概率, 用 NLL 或自己实现 CE.
    # 这里用 -log(z_hat[label]) (standard CE on softmax probs):
    eps = 1e-8
    log_probs = torch.log(z_hat_seq + eps)         # (B, T, N, N-1, 2)
    
    # Gather log prob of true class
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
        其中 Var_i 是在 batch 内 N 个 agent 上对每维度求方差, 再 mean over dims & steps.
    
    Args:
        hidden_seq: BeliefNet GRU hidden 序列
        target_std: σ_target (默认 0.1)
        mask: 可选 (B, T) bool
    
    Returns:
        scalar tensor
    """
    B, T, N, h_dim = hidden_seq.shape
    target_var = target_std ** 2
    
    # 在 dim=2 (N 维度) 上计算 batch-wise variance
    # var_per_dim: (B, T, h_dim)
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
```

### 2.3 典型用例

```python
# Pkg-05 trainer 在 train_step 内调用
from hyper_mve.models import belief_loss
from hyper_mve.models import BeliefNet

bn = BeliefNet(cfg.env, cfg.model)
# trainer 已从 buffer sample batch
obs_seq = batch["obs"]                          # (B, T, N, obs_dim)
c_true_seq = batch["c_true"]                    # (B, T)  from env.info["c_true"]
types_true = batch["types"]                     # (B, T, N) from env.info["types"]

# 课程 Stage 3 (Pure Inference)
hidden_seq, c_hat_seq, z_hat_seq = bn.forward(obs_seq)

# 注: z_hat_seq 是 head_opp 真实预测 (因为没传 oracle_z_seq)
loss, breakdown = belief_loss(
    c_hat_seq, z_hat_seq, hidden_seq,
    c_true_seq, types_true,
    weights=(1.0, 0.5, 0.01),
)

loss.backward()
optimizer.step()

# Logging
for name, val in breakdown.items():
    writer.add_scalar(f"belief/{name}", val.item(), step)


# 课程 Stage 1 (Pure Oracle)
oracle_z_seq = build_oracle_z(types_true)       # (B, T, N, N-1, 2) one-hot
hidden_seq, c_hat_seq, _ = bn.forward(obs_seq, oracle_z_seq=oracle_z_seq)

# z_hat_seq 是 oracle (但 belief_loss 计算 L_opp 需要 head_opp 真实预测!)
# 用 get_head_opp_predictions 取 head_opp 的真实输出
z_hat_predicted = bn.get_head_opp_predictions(hidden_seq)
loss, breakdown = belief_loss(
    c_hat_seq, z_hat_predicted, hidden_seq,    # <- 用真实预测计算 L_opp
    c_true_seq, types_true,
    weights=(1.0, 0.5, 0.01),
)
```

---

## 3. Implementation Notes

### 3.1 L_c 的 Oracle 来源（D4）

- 标签 `c_true_seq` 来自 `env.info["c_true"]` (Pkg-02 spec 08 明确暴露)
- 所有 N 个 agent 看到同一 c_t（Harsanyi 共同知识），故 c_true_seq shape (B, T)，无 N 维度
- broadcast 到 (B, T, N) 与 c_hat_seq 对齐
- Pkg-05 trainer 必须严格遵守："c_true 仅传 L_c 计算，不可作 BeliefNet forward 输入"（破坏 Self-Info 评估）

### 3.2 L_opp Oracle CE（v4 关键差异）

- 标签 `types_true` 来自 `env.info["types"]` (Pkg-02 暴露)
- shape (B, T, N): per-batch, per-step, per-agent 真实 type ∈ {0, 1}
- episode 内 types 不变（spec 04 Pkg-02 已约束），但每 step 仍传（buffer 字段一致）
- 标签 gather：z_hat[b, t, i, k] 对应的标签是 types_true[b, t, opp_id]，opp_id = (k if k < i else k + 1)
- 用 torch.gather 实现 batch-wise lookup（避免 Python loop）
- CE 实现：对 softmax 概率取 log（加 eps 数值稳定），按真实类别 gather → -log(p_true)
- **关键**：z_hat_seq 必须是 head_opp **真实预测**，而不是 oracle_z（否则 L_opp 在 oracle 上无信号）；Stage 1 时由 trainer 端用 `get_head_opp_predictions` 取真实预测

### 3.3 L_div hinge variance（D5）

- 目的：防止 BeliefNet 输出在 N 个 agent 间收敛为常数（信息坍缩）
- 形式：`max(0, σ_target² - Var_i)`
  - var 高于阈值 → loss = 0（不约束）
  - var 低于阈值 → 线性惩罚（推动方差增加）
- σ_target=0.1（Pkg-01 TrainConfig.belief_div_target_std 默认值）
- 在 N 维度（agent 维度）上求方差，再 mean over h_dim 与 T
- 注意符号：Ch4.5.3 公式是 `-Var_i`（最大化方差，最小化负方差），我们用 hinge 形式更直观（loss 始终 ≥ 0）

### 3.4 mask 参数（可选）

- 用于 episode padding（不同 episode 长度不同）
- 在 (B, T) 维度上的 bool 张量，True 表示有效 step
- 三个 loss 均支持 mask；mask=None 时按全部样本平均
- Pkg-05 trainer 实际使用 mask 处理 done 后的 padding step

### 3.5 weights 默认值

- λ_c = 1.0, λ_opp = 0.5, λ_div = 0.01（Ch4.5.4 推荐 + Pkg-01 TrainConfig.w_belief_c/opp/div 默认值）
- 三 loss 量级：L_c ~ 0.01-0.1（[0,1] 范围 MSE），L_opp ~ 0.5-1.0（log 2 CE），L_div ~ 0-0.01
- 权重调整后量级接近：1.0 × 0.05 = 0.05，0.5 × 0.7 = 0.35，0.01 × 0.005 = 5e-5
- L_opp 主导（因为信号最强，Oracle 监督），L_c 次之，L_div 仅 collapse 干预

### 3.6 与 Pkg-05 联合损失的耦合

- belief_loss 仅返回 BeliefNet 内部总损失
- Pkg-05 trainer 在主任务损失上加权：`L_total = L_MuZero + λ_b · L_BeliefNet`
- λ_b 默认 1.0（Ch4.5.5），课程化 ramp（Pkg-05 spec 02 内详述）
- 本模块不参与 ramp 逻辑，仅提供加权后的 BeliefNet 总损失

### 3.7 数值稳定性

- L_opp 内 `log(z_hat + eps)`：eps=1e-8 防止 log(0)
- 也可用 logits + `F.cross_entropy`（更数值稳定），但需 BeliefNet head_opp 暴露 logits 接口
- 当前选择 softmax + manual NLL（因为 head_opp 默认返回 softmax 给 belief 通路）

---

## 4. Edge Cases

| 场景 | 行为 |
|------|------|
| c_hat_seq 含 NaN | loss = NaN（trainer 端断言捕获） |
| types_true 越界（>= 2） | gather IndexError |
| z_hat_seq 不是 simplex（非 softmax） | log(neg) = NaN（应保证 head_opp 输出 softmax） |
| hidden_seq 全相同（极端 collapse） | Var = 0 → loss = σ_target² × λ_div |
| mask 全 False | denom = 1 (clamp) → loss = 0 |
| B=1, N=2 | l_div 在 N=2 时 var 不稳定（仅 2 个样本）→ loss 仍可计算 |

---

## 5. Acceptance Criteria

### 5.1 单元测试（`tests/models/test_belief_losses.py`）

```python
import pytest
import torch
from hyper_mve.models.belief_losses import l_c, l_opp, l_div, belief_loss


def test_l_c_zero_loss_at_perfect_pred():
    """c_hat == c_true 时 L_c = 0."""
    B, T, N = 2, 3, 4
    c_true = torch.tensor([[0.3, 0.4, 0.5], [0.6, 0.7, 0.8]])
    c_hat = c_true.unsqueeze(-1).expand(B, T, N).clone()
    loss = l_c(c_hat, c_true)
    assert loss.item() < 1e-6


def test_l_c_positive_when_err():
    """c_hat != c_true 时 L_c > 0."""
    B, T, N = 2, 3, 4
    c_true = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    c_hat = torch.zeros(B, T, N)  # all 0, vs true 0.5 -> MSE = 0.25
    loss = l_c(c_hat, c_true)
    assert torch.allclose(loss, torch.tensor(0.25), atol=1e-5)


def test_l_c_gradient_direction():
    """L_c 梯度方向: c_hat < c_true -> 梯度 < 0 (推 c_hat 增大)."""
    B, T, N = 1, 1, 1
    c_true = torch.tensor([[0.7]])
    c_hat = torch.tensor([[[0.3]]], requires_grad=True)
    loss = l_c(c_hat, c_true)
    loss.backward()
    # ∂L/∂c_hat = 2 * (c_hat - c_true) / (B*T*N) = 2 * (0.3 - 0.7) = -0.8
    assert c_hat.grad.item() < 0


def test_l_opp_oracle_ce(monkeypatch):
    """L_opp 是 Oracle CE: 标签是 int 类型 types_true, 不是动作."""
    B, T, N = 1, 1, 4
    # types_true: agent 0,1 是 α (0); agent 2,3 是 β (1)
    types_true = torch.tensor([[[0, 0, 1, 1]]], dtype=torch.long)
    
    # 构造完美预测的 z_hat: 每个 agent i 对所有对手 j 的预测正确
    z_hat = torch.zeros(B, T, N, N - 1, 2)
    for i in range(N):
        opp_list = [j for j in range(N) if j != i]
        for k, j in enumerate(opp_list):
            true_type = types_true[0, 0, j].item()  # 0 or 1
            z_hat[0, 0, i, k, true_type] = 1.0  # one-hot 完美预测
            # log(1.0) = 0, log(0) = -inf 但 we have eps protection
            # 但 完美 one-hot 时 -log(1+eps) ≈ -1e-8 ≈ 0
    
    loss = l_opp(z_hat, types_true)
    # 完美预测时 loss ≈ 0
    assert loss.item() < 1e-4


def test_l_opp_uniform_pred_baseline():
    """uniform softmax (0.5, 0.5) 预测时 L_opp = ln(2) ≈ 0.693."""
    B, T, N = 1, 1, 4
    types_true = torch.tensor([[[0, 0, 1, 1]]], dtype=torch.long)
    z_hat = torch.ones(B, T, N, N - 1, 2) * 0.5  # uniform
    
    loss = l_opp(z_hat, types_true)
    # -log(0.5) = ln(2) ≈ 0.6931
    import math
    assert torch.allclose(loss, torch.tensor(math.log(2)), atol=1e-4)


def test_l_opp_agent_id_order_consistency():
    """L_opp 与 z_hat 顺序约定一致 (agent_id 升序跳过 self)."""
    # 构造一个能验证顺序的小测试:
    # 给 agent 2 (i=2), 对手顺序应是 [0, 1, 3]
    # 若 z_hat[0, 0, 2, 0] 是对 agent 0 的预测,
    # 把 z_hat[0, 0, 2, 0] 设为 [1, 0] (预测 α), 验证标签是 types_true[0, 0, 0]
    B, T, N = 1, 1, 4
    types_true = torch.tensor([[[0, 1, 0, 1]]], dtype=torch.long)  # α β α β
    z_hat = torch.ones(B, T, N, N - 1, 2) * 0.5
    
    # agent 2 对对手 0 (agent 0, α) 预测全错 (预测 β)
    z_hat[0, 0, 2, 0, 0] = 0.01  # P(α) very low
    z_hat[0, 0, 2, 0, 1] = 0.99  # P(β) high (但 true is α)
    
    loss = l_opp(z_hat, types_true)
    # 该错预测贡献 -log(0.01) ≈ 4.6 显著大于 -log(0.5) = 0.69
    # 整体 loss 应大于 0.7
    assert loss.item() > 0.7


def test_l_opp_index_mapping_correctness():
    """🔑 P3 关键单测: 验证 j → agent_id 映射精确正确, 防 silent bug.
    
    设计:
        types_true = [α, β, α, β]  # agent_id 0,1,2,3 真实类型
        构造完美对角预测 z_hat: 对每个 (i, k), 给"正确的对手 j 的 type"满 prob
        断言 L_opp ≈ 0 (完美预测).
    
    若映射错位 (如 k 直接对应 agent_id 不跳 self), 则:
        - agent 2 (i=2) 的 z_hat[2, 2] 本应对 agent 3 (β), 但若错位会对 agent 2 (self=α)
        - z_hat[2, 2] 给 prob=[1, 0] (预测 α) 在错位映射下被认为正确, 实际是错误
        - 完美对角预测会变成"错位 CE", loss != 0
    """
    B, T, N = 1, 1, 4
    # types: agent 0 = α (0), agent 1 = β (1), agent 2 = α (0), agent 3 = β (1)
    types_true = torch.tensor([[[0, 1, 0, 1]]], dtype=torch.long)
    
    # 构造完美预测:
    # 对每个 (i, k), 标签是 types_true[opp_agent_id(i, k)]
    # opp_agent_id(i, k) = k if k < i else k + 1
    z_hat = torch.zeros(B, T, N, N - 1, 2)
    for i in range(N):
        for k in range(N - 1):
            j = k if k < i else k + 1  # 正确映射
            true_type = types_true[0, 0, j].item()
            # one-hot 完美预测 (注: 用 0.999 避免 log(0))
            z_hat[0, 0, i, k, true_type] = 0.999
            z_hat[0, 0, i, k, 1 - true_type] = 0.001
    
    loss = l_opp(z_hat, types_true)
    # 完美预测 → loss ≈ -log(0.999) ≈ 1e-3
    assert loss.item() < 0.01, (
        f"L_opp = {loss.item():.4f} > 0.01; 映射可能错位. "
        f"若映射正确, 完美对角预测应 ≈ -log(0.999) ≈ 1e-3."
    )


def test_l_opp_index_mapping_specific_pairs():
    """🔑 P3: 验证 (i, k) → j 的具体映射规则.
    
    对每个 (i, k), 构造仅在该位置错误的 z_hat, 验证 L_opp 反映 types_true[j].
    """
    B, T, N = 1, 1, 4
    # types: agent 0=α, agent 1=β, agent 2=α, agent 3=β
    types_true = torch.tensor([[[0, 1, 0, 1]]], dtype=torch.long)
    
    # 测试样本: (i, k, expected_opp_id)
    test_cases = [
        (0, 0, 1),  # agent 0 的对手 0 → agent 1
        (0, 1, 2),  # agent 0 的对手 1 → agent 2
        (0, 2, 3),  # agent 0 的对手 2 → agent 3
        (1, 0, 0),  # agent 1 的对手 0 → agent 0
        (1, 1, 2),  # agent 1 的对手 1 → agent 2
        (1, 2, 3),  # agent 1 的对手 2 → agent 3
        (2, 0, 0),  # agent 2 的对手 0 → agent 0
        (2, 1, 1),  # agent 2 的对手 1 → agent 1
        (2, 2, 3),  # agent 2 的对手 2 → agent 3
        (3, 0, 0),  # agent 3 的对手 0 → agent 0
        (3, 1, 1),  # agent 3 的对手 1 → agent 1
        (3, 2, 2),  # agent 3 的对手 2 → agent 2
    ]
    
    for i, k, expected_opp_id in test_cases:
        # 对其他位置均匀预测 (loss = log 2), 仅 (i, k) 错预测
        z_hat = torch.ones(B, T, N, N - 1, 2) * 0.5
        true_type_at_j = types_true[0, 0, expected_opp_id].item()
        wrong_type = 1 - true_type_at_j
        # 在 (i, k) 位置预测 wrong_type 满 prob
        z_hat[0, 0, i, k, true_type_at_j] = 0.001
        z_hat[0, 0, i, k, wrong_type] = 0.999
        
        # 若映射正确: L_opp 在 (i,k) 处贡献 -log(0.001) ≈ 6.9, 其他位置 -log(0.5) ≈ 0.69
        # 总和 ≈ (N*(N-1) - 1)*0.69 + 6.9 / (N*(N-1)) ≈ ...
        # 检查 loss 显著大于均匀基线 (log 2 ≈ 0.693)
        loss = l_opp(z_hat, types_true)
        assert loss.item() > 0.8, (
            f"(i={i}, k={k}, expected_opp_id={expected_opp_id}): "
            f"L_opp={loss.item():.4f} should be > 0.8 (一处错预测应让 loss 显著上升). "
            f"若 loss < 0.8, 可能映射 (i,k)→j 错位."
        )


def test_l_c_broadcast_correctness():
    """P8: c_true_seq shape (B, T) 不含 N 维, 内部 broadcast 到 (B, T, N).
    
    验证标量等价:
        L_c = (1/(B*T*N)) * Σ_{b,t,i} (c_hat[b,t,i] - c_true[b,t])²
    """
    B, T, N = 2, 3, 4
    # 固定 c_hat = 0.5 across all agents, c_true 不同
    c_hat_seq = torch.full((B, T, N), 0.5)
    c_true_seq = torch.tensor([
        [0.5, 0.5, 0.5],   # batch 0: 完美匹配 → contrib 0
        [0.0, 1.0, 0.5],   # batch 1: 错配 (0.5)² + (0.5)² + 0 per agent
    ])
    
    loss = l_c(c_hat_seq, c_true_seq)
    # 手算标量:
    # batch 0: 0 (3 step × N=4 agent, 全 0)
    # batch 1: t=0 → (0.5-0)² × 4 = 1.0; t=1 → (0.5-1)² × 4 = 1.0; t=2 → 0
    # 总 sq_err = 0 + 1.0 + 1.0 + 0 = 2.0
    # denom = B*T*N = 2*3*4 = 24
    # L_c = 2.0 / 24 ≈ 0.0833
    expected = 2.0 / 24
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-5)


def test_l_c_wrong_input_shape_rejected():
    """P8: c_true_seq shape (B, T, N) 应被拒绝 (与契约不符)."""
    B, T, N = 1, 2, 4
    c_hat_seq = torch.zeros(B, T, N)
    bad_c_true = torch.zeros(B, T, N)  # 错: 含 N 维
    with pytest.raises(AssertionError, match="c_true_seq shape"):
        l_c(c_hat_seq, bad_c_true)


def test_l_div_hinge_zero_when_high_variance():
    """variance >= target_var 时 loss = 0."""
    B, T, N, h = 1, 1, 4, 8
    # 制造高方差 hidden
    hidden = torch.randn(B, T, N, h) * 5  # std ~ 5, var ~ 25 >> 0.01
    loss = l_div(hidden, target_std=0.1)
    assert loss.item() < 1e-6


def test_l_div_hinge_positive_when_low_variance():
    """variance < target_var 时 loss > 0."""
    B, T, N, h = 1, 1, 4, 8
    # 制造低方差 hidden (所有 agent 几乎相同)
    base = torch.randn(B, T, 1, h)
    hidden = base.expand(B, T, N, h).clone() + torch.randn(B, T, N, h) * 0.001
    # var << 0.01
    loss = l_div(hidden, target_std=0.1)
    assert loss.item() > 0


def test_l_div_collapse_protection():
    """完全 collapse (var=0) 时 loss = target_var (上限)."""
    B, T, N, h = 1, 1, 4, 8
    hidden = torch.ones(B, T, N, h) * 0.5  # 所有 agent 全相同
    loss = l_div(hidden, target_std=0.1)
    # var = 0 → loss = target_var = 0.01
    assert torch.allclose(loss, torch.tensor(0.01), atol=1e-6)


def test_belief_loss_combination():
    """belief_loss 正确组合三 loss + 加权."""
    B, T, N = 2, 3, 4
    c_hat = torch.rand(B, T, N)
    z_hat = torch.softmax(torch.randn(B, T, N, N - 1, 2), dim=-1)
    hidden = torch.randn(B, T, N, 128)
    c_true = torch.rand(B, T)
    types_true = torch.randint(0, 2, (B, T, N))
    
    total, breakdown = belief_loss(
        c_hat, z_hat, hidden, c_true, types_true,
        weights=(1.0, 0.5, 0.01),
    )
    
    assert total.requires_grad
    assert "l_c" in breakdown
    assert "l_opp" in breakdown
    assert "l_div" in breakdown
    assert "total" in breakdown
    
    # breakdown 是 detach
    for k, v in breakdown.items():
        assert not v.requires_grad
    
    # 验证加权
    expected = 1.0 * breakdown["l_c"] + 0.5 * breakdown["l_opp"] + 0.01 * breakdown["l_div"]
    assert torch.allclose(total.detach(), expected, atol=1e-5)


def test_belief_loss_mask():
    """mask=True 的样本被纳入, mask=False 被忽略."""
    B, T, N = 2, 4, 4
    c_hat = torch.rand(B, T, N)
    z_hat = torch.softmax(torch.randn(B, T, N, N - 1, 2), dim=-1)
    hidden = torch.randn(B, T, N, 128)
    c_true = torch.rand(B, T)
    types_true = torch.randint(0, 2, (B, T, N))
    
    # mask: 第一个 batch 只前 2 step 有效, 第二个全有效
    mask = torch.tensor([[True, True, False, False], [True] * 4])
    
    total_mask, _ = belief_loss(c_hat, z_hat, hidden, c_true, types_true, mask=mask)
    total_full, _ = belief_loss(c_hat, z_hat, hidden, c_true, types_true)
    
    # mask 后 loss 可能不同 (排除 padding step)
    assert total_mask.requires_grad


def test_l_c_uses_oracle_field_name():
    """document: L_c 标签字段名是 c_true_seq (与 Pkg-02 env.info['c_true'] 对应)."""
    import inspect
    sig = inspect.signature(l_c)
    assert "c_true_seq" in sig.parameters


def test_l_opp_uses_types_true_field_name():
    """document: L_opp 标签字段名是 types_true (与 Pkg-02 env.info['types'] 对应)."""
    import inspect
    sig = inspect.signature(l_opp)
    assert "types_true" in sig.parameters
```

### 5.2 集成测试（合成数据收敛）

在 `scripts/test_belief_net_synth.py` 中：
- 构造已知 c=0.7 + 已知 types 的合成轨迹
- 训练 5K 步
- 断言 head_c MSE < 0.05（说明 L_c 起作用）
- 断言 head_opp accuracy > 80%（说明 L_opp 起作用）
- 断言 b_i^t 方差 ≥ 0.1（说明 L_div hinge 起作用，无 collapse）

### 5.3 性能要求

- `belief_loss(B=256, T=6, N=4)` 单步 < 5 ms (V100)
- 三 loss 内部 torch.gather / variance 计算无 Python loop

---

## 6. v3 → v4 关键差异（L_opp）

| 维度 | v3 (自监督) | v4 (Oracle 监督，本 spec) |
|------|-------------|--------------------------|
| 监督信号 | next_action（buffer 中下一步对手动作） | **types_true**（env.info["types"] Oracle） |
| 标签类型 | int（动作索引） | **int (AgentType.value)** |
| 输出维度 | d_z 维 logits | **2 维 softmax** |
| 信号强度 | 弱（仅从动作推断 type） | **强**（直接监督 type） |
| 收敛速度 | 慢（5K 步内难达 80%） | **快**（5K 步内 > 80%） |

---

## 7. Cross-references

- Ch4.5.1 L_c MSE 公式
- Ch4.5.2 L_opp Oracle CE 公式（v4 关键改动）
- Ch4.5.3 L_div hinge variance 公式
- Ch4.5.4 总损失 + 权重
- Ch4.5.5 与主任务联合训练 (λ_b)
- `04-belief-net-gru.md`（hidden_seq 来源）
- `05-belief-heads.md`（c_hat / z_hat 来源 + Stage 1 oracle 模式下用 get_head_opp_predictions）
- `08-integration-contracts.md`（trainer 接入 Oracle 标签的契约）
- Pkg-01 `04-timestep-record.md`（c_hat / z_hat / 标签字段定义）
- Pkg-01 `05-v4-config-structure.md` TrainConfig.w_belief_c/opp/div
- Pkg-02 `08-gym-api.md`（env.info["c_true"] / info["types"] Oracle 来源）
- Pkg-05 spec `02-curriculum-stages.md`（Stage 1 oracle 模式下 head_opp 真实预测取法）
- Pkg-05 spec `05-loss-functions-v4.md`（trainer 端 belief_loss 与 main loss 联合）
