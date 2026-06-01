# Spec 08: 集成契约 — 与 Pkg-04 (Model) / Pkg-05 (Trainer) 接口约定

> 父文档：[`../proposal.md`](../proposal.md) §3.2 / §4.2 · [`../design.md`](../design.md) §6.5
> **本 spec 是 Pkg-03 对外的"硬契约"** — 定义本包对 Pkg-04 / Pkg-05 提供的接口稳定性 + Self-Info 严格性 + 课程接入约定 + 梯度门控协作。

---

## 1. Purpose

把本包内部模块（TriContextEncoder + BeliefNet + belief_loss + PermutationInvariantPool）暴露给下游 Pkg-04 / Pkg-05 / Pkg-06a/b 时，**定义必须严格遵守的契约**，防止 v4 关键设计约束（Self-Info 严格性、Oracle 监督 vs Self-Info 评估、belief 梯度门控、课程 stage 切换）被下游误用破坏。

本 spec 是 Pkg-03 的"对外说明书"，下游包 spec（Pkg-04 spec 02/04，Pkg-05 spec 02/04/05/06）将引用本 spec 作为接入参考。

---

## 2. 契约 1：Self-Info 严格性（Ch3.7 + Ch4.2.2）

### 2.1 规则

**`role_i` 仅含 agent i 自己的 (id, type, cap)；他人的 type 不可经任何通路进入 model forward**。

### 2.2 允许的数据流

```python
# ✅ 正确：trainer 端构造 RoleEncoder 输入
# types: (B, N) per-agent 自报告 type（即 batch 内每个位置的 agent 看到自己的 type）

# Worker 收集 episode 时:
agent_self_type = env._state.agent_types[agent_idx]  # ← 自己的真实类型 (自报告 OK)
# 或 trainer 从 buffer 取:
types_in_batch = batch["tau"]  # (B, N) - 每个位置存的是该 agent 自报告 type
role = role_encoder(agent_ids, types_in_batch, caps)  # ← 仅 own type 进 role
```

### 2.3 禁止的数据流

```python
# ❌ 错误：把 env.info["types"] 全 N 真实 type 矩阵传给 RoleEncoder
# (相当于把"他人 type"也通过 role 通路泄漏给 hypernet)

oracle_types = env.info["types"]   # (N,) - **所有** agent 的真实 type
# 这是 Oracle 监督专用, 只能传 L_opp CE!

# 误用:
role = role_encoder(agent_ids, oracle_types_broadcast, caps)
# 现在 agent 0 通过 role[0] 看到了 agent 1/2/3 的 type → 违反 Self-Info
```

### 2.4 工程实施

| 谁负责 | 内容 |
|--------|------|
| **Pkg-02 env** | spec 08 已明确：info 分组为 "Public" (caps, deltas) / "Oracle" (c_true, types) / "Eval only"；`_oracle_fields = ("c_true", "types")` 显式标识 |
| **Pkg-05 trainer** | spec 04 (trainer-loop-v2) 必须严格分组消费 info：`types_true = info["types"]` **仅**传 belief_loss(L_opp)；不可作 BeliefNet forward 或 RoleEncoder 输入 |
| **Pkg-03 本包** | RoleEncoder 接收 types 参数命名为 `types: (B, N)`（不是 `oracle_types`）；docstring 明确"仅 own type，每个 batch 位置一个" |
| **Pkg-04 model** | spec 02 set_context API 接收 `type_i`（单个 agent 视角下的自己 type）→ 内部转换为 (B, N) tensor 传 role_encoder |

### 2.5 审计单测

- Pkg-03 `tests/models/test_role_encoder.py::test_self_info_severity_only_own_type` —— RoleEncoder 拒绝 (B, N, N) 全 N×N type 矩阵
- Pkg-05 spec 04 将定义 `test_trainer_no_oracle_types_in_model_forward` —— 静态分析 trainer 代码，确保 `info["types"]` 仅用于 belief_loss 调用

---

## 3. 契约 2：Oracle 监督与 Self-Info 评估的分离

### 3.1 训练时（Oracle）

```python
# Pkg-05 trainer (train_step)
batch = buffer.sample(B=256)
obs_seq = batch["obs"]
c_true_seq = batch["c_true"]            # Oracle (env.info["c_true"])
types_true = batch["tau"]               # Oracle (env.info["types"] / TimeStepRecord.tau)

# BeliefNet 训练 (Stage 3 Pure Inference)
hidden_seq, c_hat_seq, z_hat_seq = belief_net.forward(obs_seq)

# L_opp 用 Oracle types_true 作 CE 标签
loss, breakdown = belief_loss(
    c_hat_seq, z_hat_seq, hidden_seq,
    c_true_seq, types_true,             # Oracle 标签
)
```

### 3.2 评估时（Self-Info, Pkg-07）

```python
# Pkg-07 evaluator (Self-Info 评估协议)
obs_seq = env_obs_seq                   # 真实 env 收集

# ⚠️ 关键: BeliefNet forward 不传 oracle_z_seq, 必须用 head_opp 推断
hidden_seq, c_hat_seq, z_hat_seq = belief_net.forward(obs_seq)

# z_hat_seq 是 head_opp 真实预测 (没有 Oracle 泄漏)

# Model forward (主任务推断)
ctx_i = tri_context_encoder.forward(
    c_t, agent_ids, types_self_only,    # ← 仅 own type (Self-Info)
    caps, (c_hat_seq[..., t], z_hat_seq[..., t]),
)
# ctx_i 中的 belief 子段完全来自 BeliefNet 推断, 不含 Oracle 泄漏
```

### 3.3 工程实施

| 评估协议 | BeliefNet forward 调用 | role types 来源 |
|----------|------------------------|-----------------|
| 训练（含 Stage 1/2/3） | trainer 控制 oracle_z_seq（Stage 切换）| trainer 传 batch["tau"] (自报告) |
| Self-Info 评估 | **不传 oracle_z_seq**（全 head_opp 推断） | trainer 传 own type (env._state.agent_types[i]) |
| Oracle 评估（对照实验） | 传 oracle_z_seq（用 Oracle 替代推断） | 仍仅 own type |

### 3.4 审计单测

- Pkg-07 spec 04 (`self-info-eval.md`) 将定义 `test_self_info_no_oracle_leak`：Self-Info eval 时 `belief_net.forward()` 不接受 oracle_z_seq 参数（用断言或代码静态检查）

---

## 4. 契约 3：课程学习 Stage 接入（Ch5.7）

### 4.1 三阶段 belief 输入构造（与 Pkg-05 spec 02 协同）

```python
# Pkg-05 trainer 在每个 train_step 内
step = trainer.global_step
max_steps = cfg.train.max_train_steps
s1_end = cfg.train.curriculum_stage_1_end_frac * max_steps   # 0.3 * max
s2_end = cfg.train.curriculum_stage_2_end_frac * max_steps   # 0.7 * max

if step < s1_end:
    # ====== Stage 1: Pure Oracle ======
    # 用 Oracle τ_j one-hot 替代 head_opp 输出
    oracle_z_seq = build_oracle_z_seq(types_true)
    # shape: (B, T, N, N-1, 2) one-hot, 顺序与 z_hat 约定一致
    
    hidden_seq, c_hat_seq, z_for_main = belief_net.forward(
        obs_seq, oracle_z_seq=oracle_z_seq,
    )
    # z_for_main == oracle_z_seq (传给下游 hyper_rew 用)
    # 但 head_opp 仍在内部 forward (供 L_opp 训练)
    
    # 取 head_opp 真实预测计算 L_opp
    z_hat_predicted = belief_net.get_head_opp_predictions(hidden_seq)
    
elif step < s2_end:
    # ====== Stage 2: Anneal ======
    lambda_t = (step - s1_end) / (s2_end - s1_end)  # ∈ [0, 1]
    
    # 先得 head_opp 真实预测
    hidden_seq, c_hat_seq, z_hat_predicted = belief_net.forward(obs_seq)
    
    # 构造 Oracle one-hot
    oracle_z_seq = build_oracle_z_seq(types_true)
    
    # 加权混合: (1-λ) * oracle + λ * predicted
    z_for_main = (1 - lambda_t) * oracle_z_seq + lambda_t * z_hat_predicted
    
else:
    # ====== Stage 3: Pure Inference ======
    hidden_seq, c_hat_seq, z_hat_predicted = belief_net.forward(obs_seq)
    z_for_main = z_hat_predicted

# Main task forward (Pkg-04 model)
ctx_i = encoder.forward(c_t, agent_ids, types_self, caps, (c_hat_seq, z_for_main))
# hyper_rew, hyper_pred 接收 ctx_i

# BeliefNet loss (始终用 head_opp 真实预测)
loss_belief, _ = belief_loss(
    c_hat_seq, z_hat_predicted, hidden_seq,  # ← 真实预测，不是 oracle
    c_true_seq, types_true,
)
```

### 4.2 关键约定

- **Stage 1 / Stage 2 时 z_for_main 用 Oracle / 加权混合**：下游主任务（hyper_rew/hyper_pred）训练时看到稳定信号，避免 head_opp 早期不准导致主任务陷入次优
- **L_opp 始终用 head_opp 真实预测**：BeliefNet head_opp 必须从 Stage 1 就开始训练（用 Oracle 监督），否则 Stage 3 切换时 head_opp 还是随机的
- **L_c 始终用 head_c 真实预测**：与 L_opp 同理
- **BeliefNet.forward 的 oracle_z_seq 参数仅影响返回的 z_seq，不影响内部 head_opp 计算**：实现见 spec 05 `_compute_z_hat` 始终运行

### 4.3 `build_oracle_z_seq` 实现签名（P9，2026-05-28）

**归属约定**：本函数定义**在 Pkg-03 提供**（确保与 Pkg-03 head_opp 顺序约定严格一致），Pkg-05 trainer 调用。文件路径：`hyper_mve/models/belief_losses.py`（与 belief_loss 同模块，便于 import）。

```python
import torch


def build_oracle_z_seq(
    types_true: torch.Tensor,           # (B, T, N) int64 (AgentType.value ∈ {0, 1})
) -> torch.Tensor:                      # (B, T, N, N-1, 2) one-hot Δ^2
    """构造 Stage 1 / Stage 2 anneal 用的 Oracle z 序列 (P9).
    
    **顺序约定 (硬约束)**:
        z[b, t, i, k] 对应 agent_id = (k if k < i else k + 1)
        z[b, t, i, k, types_true[b, t, agent_id]] = 1.0
        其他位置 = 0.0
    
    与 spec 05 head_opp 输出 / spec 06 l_opp 标签 gather / Pkg-01 TimeStepRecord
    z_hat 字段三处使用同一映射, 保证课程 Stage 1 oracle 注入与 L_opp Oracle
    监督一致 (否则会引入错位干扰).
    
    数学等价 (标量伪代码):
        for b in range(B):
            for t in range(T):
                for i in range(N):
                    for k in range(N - 1):
                        j = k if k < i else k + 1
                        z[b, t, i, k, types_true[b, t, j]] = 1.0
        return z
    
    向量化实现 (本函数下方): 复用 spec 06 l_opp 的 opp_indices 构造逻辑.
    
    Args:
        types_true: Oracle types from env.info["types"] (per-batch, per-step,
                    per-agent). episode 内不变, 但每 step 仍传 (与 z_hat_seq shape 对齐).
    
    Returns:
        oracle_z: (B, T, N, N-1, 2) one-hot 概率分布 ∈ Δ^2
    """
    B, T, N = types_true.shape
    device = types_true.device
    
    # 1. 构造 opp_indices: (N, N-1), opp_indices[i, k] = (k if k < i else k + 1)
    #    与 spec 06 l_opp 完全一致
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


# 单测 (与 spec 06 配对)
def test_build_oracle_z_seq_one_hot():
    """oracle_z 输出 one-hot."""
    types = torch.tensor([[[0, 1, 0, 1]]], dtype=torch.long)
    z = build_oracle_z_seq(types)
    assert z.shape == (1, 1, 4, 3, 2)
    # 每个 (b, t, i, k) 概率分布是 one-hot
    sums = z.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums))


def test_build_oracle_z_seq_matches_head_opp_order():
    """oracle_z 顺序与 l_opp gather 顺序严格一致 (端到端验证)."""
    from hyper_mve.models.belief_losses import l_opp
    types = torch.tensor([[[0, 1, 0, 1]]], dtype=torch.long)
    oracle_z = build_oracle_z_seq(types)
    # 若用 oracle_z 作 z_hat 喂 l_opp, 损失应 ≈ 0 (完美匹配)
    loss = l_opp(oracle_z * 0.999 + 0.001 / 2, types)  # 加 smoothing 避免 log(0)
    assert loss.item() < 0.01
```

**关键约束**：
- 本函数与 `l_opp`、`BeliefNet._compute_z_hat`（spec 05）**共用同一 opp_indices 构造逻辑**
- 三处的顺序映射 `(i, k) → agent_id = (k if k < i else k + 1)` 必须严格一致
- 单测 `test_build_oracle_z_seq_matches_head_opp_order` 用端到端方式（oracle_z → l_opp）验证

### 4.3 工程实施

| 谁负责 | 内容 |
|--------|------|
| **Pkg-03 本包** | BeliefNet.forward 实现 `oracle_z_seq` 参数；提供 `get_head_opp_predictions(hidden_seq)` 接口 |
| **Pkg-05 trainer** | spec 02 (curriculum-stages) 实现 stage 切换逻辑 + oracle_z 构造 + 加权混合 + 调用本包 API |
| **Pkg-05 trainer** | spec 04 (trainer-loop) 在 train_step 内调用 belief_loss(z_hat_predicted)，不是 z_for_main |

### 4.4 审计单测

- Pkg-03 `tests/models/test_belief_net.py::test_head_opp_still_trained_in_oracle_mode` —— Stage 1 oracle 时 head_opp 仍 forward + 有梯度
- Pkg-05 spec 02 将定义 `test_stage_1_oracle_mode_l_opp_computed_from_predicted`

---

## 5. 契约 4：belief 梯度门控（Ch4.6.5，与 Pkg-04 协作）

### 5.1 规则

**训练前 5K step（`cfg.train.belief_grad_gating_steps=5000`）belief 通路梯度被切断**：主任务 loss 反向传播时不通过 BeliefNet 参数更新，只更新主任务参数（hypernet, RepNet 等）。

### 5.2 切断的实现

belief gradient gating 在 **Pkg-04 model.forward** 内实现（不在本包），通过 `.detach()` 切断 belief tensor：

```python
# Pkg-04 hyper_muzero_model.set_context (示意)
def set_context(self, c_t, agent_id, cap_i, belief_tuple):
    c_hat, z_hat = belief_tuple
    
    if self.belief_grad_gating_enabled():  # global_step < 5000
        c_hat = c_hat.detach()
        z_hat = z_hat.detach()
    
    # 走 TriContextEncoder 时, belief 子段无梯度
    ctx_i = self.tri_context_encoder.forward(
        c_t, agent_id, type_i, cap_i, (c_hat, z_hat),
    )
    # ctx_i 经 hypernet 生成 θ, θ 反向传梯度时:
    # - 经 c_ctx + role 路径 → 正常梯度 (c_encoder, role_encoder 更新)
    # - 经 belief 子段 → 被 .detach() 切断 (BeliefNet 参数不更新)
    # 
    # 但 BeliefNet 自己的 L_belief loss 仍正常反向 (独立优化)
```

### 5.3 BeliefNet 端的影响

- **本包 BeliefNet.forward 不参与门控逻辑**——是否 .detach() 由 Pkg-04 model 决定
- BeliefNet 自身的 `belief_loss(...)` 损失反向时**始终更新** BeliefNet 参数（与主任务梯度门控独立）
- 这意味着 BeliefNet 在前 5K step 中**仍**正常学习（L_c, L_opp, L_div 起作用），只是主任务暂时不通过 belief 通路获益

### 5.4 5K 步后的开启

- step ≥ 5000 后 Pkg-04 不再调用 .detach()，belief 通路梯度恢复
- 此时 main task loss 也开始通过 BeliefNet 参数反向传播
- 联合优化的稳定性靠 Pkg-05 课程学习与 EMA target net 保证

### 5.5 工程实施

| 谁负责 | 内容 |
|--------|------|
| **Pkg-03 本包** | 提供 BeliefNet.forward 接口（无 detach 内部逻辑） |
| **Pkg-04 model** | spec 04 (belief-gradient-gating) 实现 step-conditional .detach() |
| **Pkg-05 trainer** | 传递 `global_step` 给 model（让 model 知道当前是否应门控） |

### 5.6 审计单测

- Pkg-04 spec 04 将定义 `test_pre_5k_belief_detached` / `test_post_5k_belief_grad`

---

## 6. 契约 5：BeliefNet 在 Baseline 间共享（Pkg-06a/b 公平性）

### 6.1 规则

**所有 7 baseline 共享同一 BeliefNet/TriContextEncoder 实例**（断言 B 等参等条件对比要求）。

### 6.2 工厂接口（Pkg-06a `shared_backbones.py` 调用本包）

```python
# Pkg-06a/b shared_backbones.py
from hyper_mve.models import BeliefNet, TriContextEncoder

def create_belief_net(env_cfg, model_cfg):
    """所有 baseline 调用此工厂得到 BeliefNet 实例.
    
    返回未训练的 BeliefNet (每个 baseline 各自训练自己的 BeliefNet 实例,
    但架构完全一致 → 参数量对齐).
    """
    return BeliefNet(env_cfg, model_cfg)


def create_tri_context_encoder(env_cfg, model_cfg):
    return TriContextEncoder(env_cfg, model_cfg)
```

### 6.3 工程实施

| 谁负责 | 内容 |
|--------|------|
| **Pkg-03 本包** | 暴露 `from hyper_mve.models import BeliefNet, TriContextEncoder` 一行 import |
| **Pkg-06a baselines** | spec 01 (shared-backbones) 实现工厂 + 验证 7 baseline 参数量 ≤ 5% 差异 |
| **Pkg-06b baselines** | 同上 |

### 6.4 注意：共享类，不共享实例

- 7 baseline 各自构造 `BeliefNet(cfg)` 实例（参数独立训练）
- 但 BeliefNet 的**结构** + 参数量完全相同（断言 B 公平性核心）
- 训练时各 baseline 的 BeliefNet 独立优化（不共享梯度），与 main task model 联合训练

---

## 7. 契约 6：buffer 字段一致性（与 Pkg-01 / Pkg-05 协作）

### 7.1 BeliefNet 输出与 TimeStepRecord 字段对应

| BeliefNet 输出 | TimeStepRecord 字段 | shape |
|----------------|---------------------|-------|
| `c_hat` (B, N) | `TimeStepRecord.c_hat` (N,) | sigmoid scalar |
| `z_hat` (B, N, N-1, 2) | `TimeStepRecord.z_hat` (N, N-1, 2) | softmax 概率 |

### 7.2 Worker 写入 buffer

```python
# Pkg-05 worker collect_episode
for t in range(T):
    # ... env.step ...
    
    # BeliefNet 单步推断
    new_hidden, c_hat, z_hat = belief_net.step(obs_t, prev_hidden, oracle_z=...)
    
    # 写入 TimeStepRecord
    record = TimeStepRecord(
        o=obs_t.cpu().numpy(),
        # ...
        c_hat=c_hat[0].cpu().numpy(),        # (N,) - batch_size=1 worker
        z_hat=z_hat[0].cpu().numpy(),        # (N, N-1, 2)
        # ...
    )
    buffer.append(record)
    prev_hidden = new_hidden
```

### 7.3 顺序约定的端到端一致性

- BeliefNet head_opp 输出顺序：spec 05 约定 "agent_id 升序跳过 self"
- TimeStepRecord z_hat 字段顺序：Pkg-01 spec 04 约定一致
- L_opp 标签 gather：spec 06 实现一致
- **三处必须严格一致**，否则 L_opp 在错位标签 CE → 训练失败

### 7.4 审计单测

- Pkg-03 `test_z_hat_agent_id_order_convention`（spec 05）
- Pkg-01 `test_z_hat_order_convention`（spec 04 TimeStepRecord）
- Pkg-05 spec 06 worker spec 将增加 `test_worker_z_hat_buffer_order_match_belief_net`

---

## 8. 集成测试 checklist（Pkg-03 → Pkg-04/05 联调）

在 Pkg-04/05 PR merge 前，本包应通过以下集成测试：

| # | 测试 | 责任包 |
|---|------|--------|
| 1 | TriContextEncoder + DualHyperNetwork v2 forward 维度对齐（80 维 → θ） | Pkg-04 spec 01 |
| 2 | BeliefNet.step 在 worker 内调用，TimeStepRecord 字段填充正确 | Pkg-05 spec 06 |
| 3 | belief_loss 在 trainer train_step 内调用，三 loss 数值范围合理 | Pkg-05 spec 04 |
| 4 | 课程 Stage 1/2/3 切换时 oracle_z 构造与 z_for_main 加权混合正确 | Pkg-05 spec 02 |
| 5 | belief gradient gating 前 5K step 切断 BeliefNet 参数梯度 | Pkg-04 spec 04 |
| 6 | Self-Info 评估时 BeliefNet 不接受 oracle_z（无泄漏） | Pkg-07 spec 04 |
| 7 | 7 baseline 共享 BeliefNet 结构，参数量 ≤ 5% 差异 | Pkg-06a/b spec 01 |

---

## 9. 接口稳定性承诺

| API | 稳定性 | 备注 |
|-----|--------|------|
| `TriContextEncoder.__init__(env_cfg, model_cfg)` | 稳定 | cfg 字段由 Pkg-01 ModelConfig 控制 |
| `TriContextEncoder.forward(c_t, agent_ids, types, caps, belief)` | 稳定 | 参数顺序与类型不变 |
| `TriContextEncoder.forward_c_ctx_only(c_t)` | 稳定 | hyper_trans 用 |
| `BeliefNet.init_hidden(B, N, device)` | 稳定 | – |
| `BeliefNet.step(obs_t, prev_hidden, oracle_z=None)` | 稳定 | oracle_z 是 Optional 参数 |
| `BeliefNet.forward(obs_seq, init_hidden=None, oracle_z_seq=None)` | 稳定 | – |
| `BeliefNet.get_head_opp_predictions(hidden_seq)` | 稳定 | Stage 1 oracle 模式下取 head_opp 真实预测 |
| `belief_loss(c_hat, z_hat, hidden, c_true, types_true, weights, ...)` | 稳定 | 参数顺序固定 |
| `l_c`, `l_opp`, `l_div` 独立 API | 稳定 | – |
| `build_oracle_z_seq(types_true)` | 稳定 | P9 新增，归属 Pkg-03，Pkg-05 调用 |
| `make_pool(kind, feat_dim, **kwargs)` | 稳定 | – |
| `BeliefEncoder.forward(c_hat, z_hat)` | 稳定 | P5 新增，归属 Pkg-03 belief_encoder.py |
| `CapabilityVector.normalize()` | 稳定 | C 修订，归属 Pkg-01 capability.py 增补 |

**注**：稳定性指 Pkg-03 → Pkg-08 全程不变。如需修改，必须发起 issue + 跨包讨论。

---

## 10. Cross-references

- 全部 7 个 Pkg-03 specs
- Pkg-01 spec 04 (TimeStepRecord), spec 05 (V4Config)
- Pkg-02 spec 08 (env.info Oracle 信号)
- Pkg-04 spec 02 (set_context), spec 04 (belief 梯度门控)
- Pkg-05 spec 02 (curriculum-stages), spec 04 (trainer-loop), spec 06 (worker)
- Pkg-06a spec 01 (shared-backbones)
- Pkg-07 spec 04 (self-info-eval)
- Ch3.7 + Ch4.2.2 Self-Info 严格性
- Ch4.5 BeliefNet 训练
- Ch4.6.5 belief 梯度门控
- Ch5.7 课程学习
