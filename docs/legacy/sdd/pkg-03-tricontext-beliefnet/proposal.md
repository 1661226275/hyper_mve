# Pkg-03: TriContextEncoder & BeliefNet — Proposal

| 元信息 | 值 |
|--------|----|
| **包 ID** | `pkg-03-tricontext-beliefnet` |
| **状态** | Draft, awaiting review |
| **工期估计** | 1 周（5-7 天） |
| **GPU 算力** | 10 小时（BeliefNet 合成数据收敛 + collapse 测试） |
| **依赖** | Pkg-01 (Foundation Schema), Pkg-02 (env.info Oracle 信号) |
| **被依赖** | Pkg-04 (Model), Pkg-05 (Trainer), Pkg-06a/b (Baselines) |
| **论文章节** | **Ch4.2 全部填实**（三路输入维度表）+ **Ch4.5 全部填实**（BeliefNet 三损失 + 收敛曲线） |
| **PR 体量** | 9 新文件 + 4 测试文件 + 1 验证脚本 |

---

## 1. Why（为什么需要这个包）

### 1.1 v4.7 缺失 belief 通道与 type_emb

v4.7 的上下文编码体系是 `context_encoder.py`（rule + id 双路）+ `gru_context_encoder.py`（rule 推断）——**架构上完全缺失 belief 通道**，也没有 type_emb 进入 role。这两个缺失直接对应论文断言的物理不可能：

| 论文断言 | v4.7 缺失 → 后果 |
|----------|------------------|
| **断言 A** 类型梯度撕裂 | role 无 type_emb → hyper_rew 无法生成类型差异化的 θ_rew^i → 共享 RewardHead 撕裂无法解决 |
| **断言 B** 信念专用容量 | 无 BeliefNet → 无 belief 通路 → Hyper vs Input-Wide 在 belief 维度上的容量分配几何差异无法构造 |
| **断言 C** 三路必要性 | 仅 rule + id 两路 → Ablation 2 移除"任一路"中本应消融的 belief 路压根不存在 |
| **断言 D** 规划器双技术 | 与本包无直接关系（属于 Pkg-05） |

**没有本包，前三个断言的实验载体不存在**——后续 Pkg-04 即使把 hypernet 接口拉好，没有 ctx_i = (c_ctx, role, belief) 80 维输入也无东西可接。

### 1.2 Ch4 v4 对本包的硬约束

v4 相对 v3 在本包对应章节做了 5 项关键演进，必须全部实现：

| Ch4 v4 演进 | 对本包影响 | v3 行为 → v4 行为 |
|-------------|------------|-------------------|
| **Ch4.2.2 role_i 加入 type_emb** | `role_encoder.py` 必须含 `nn.Embedding(2, 8)` type 表 | v3: role = id + cap → v4: role = id + **type** + cap |
| **Ch4.2.3 ẑ 改为类型 2 分类** | `belief_net.py` head_opp 输出 (B, N-1, 2) softmax | v3: d_z 维动作预测（自监督） → v4: 2 维类型概率（Oracle 监督） |
| **Ch4.5.2 L_opp Oracle 监督** | `belief_losses.py::l_opp` 接收 `env.info["types"]` 作 CE 标签 | v3: 自监督下一步动作 → v4: Oracle τ_j 真值 CE |
| **Ch4.2.3 ĉ 是 raw scalar** | `belief_net.py` head_c 输出 sigmoid scalar | v3 / 早期 v4 草稿: (N, 16) 向量 → v4: (N,) scalar，投影在 TriContextEncoder 内 |
| **Ch4.2.4 belief 总维度 32** | `tri_context_encoder.py` 内 belief 子分量各投影到 16 后 concat | v3: belief 维度笼统 → v4: d_belief = 2 × d_b^proj = 2 × 16 = 32 精确 |

**关键 v4 差异本质**：head_opp 改为 Oracle 类型 2 分类，使 BeliefNet 训练从"弱自监督"变为"强 Oracle 监督"——这是 Pkg-02 暴露 `env.info["types"]` 的下游消费者，也是 Harsanyi 不完美信息博弈（Ch4.1.4）"他人 type 的分布信念"在深度世界模型中的**首次显式实现**。

### 1.3 断言 B 公平性要求所有 baseline 共享 BeliefNet

Ch6.4 Q4 强化 A/B/C 要求：**所有 7 baseline 共享 RepNet + BeliefNet + TriContextEncoder**——只在如何处理 (c_ctx, role, belief) → 模型参数这一环节差异：
- Hyper-MuZero：三路 → hypernet → 生成 per-context θ
- Input-Wide/Deep：三路 → concat → 加宽/加深的共享网络
- MA-MuZero / MAMBA / MARIE / MAPPO / QMIX / Conflict-Aware GA：三路 → 各自的方法适配

**若本包未提供可复用的 BeliefNet/TriContextEncoder 工厂接口（具体在 Pkg-06a `shared_backbones.py`，但本包须先满足"独立模块、易组合"的工程要求），断言 B 的等参等条件对比直接破产**。

### 1.4 对应路线图条款

| Roadmap / Chapter 条款 | 本包如何响应 |
|------------------------|----------------|
| Roadmap §4 Stage 1 Week 2: 实现 belief_net.py | 本包 9 新文件实现 |
| Ch4.2.1 c_ctx 客观通路 | `c_encoder.py` MLP[1→32→32→16] |
| Ch4.2.2 role 通路（含 type_emb v4 关键） | `role_encoder.py` id+type+cap concat → 32 |
| Ch4.2.3 belief 通路（含 v4 关键改动） | `belief_net.py` GRU + 两 head |
| Ch4.2.4 三通路汇总 | `tri_context_encoder.py` 80 维 ctx_i |
| Ch4.5.1-4.5.3 三损失定义 | `belief_losses.py` L_c + L_opp + L_div |
| Ch4.5.4 联合权重 | `belief_losses.py` 1.0 / 0.5 / 0.01 |
| Ch4.6.5 belief 梯度门控（v4 新增） | 本包仅提供 `forward(detach_belief: bool)` 接口；具体 step-conditional 切换在 Pkg-04 model + Pkg-05 trainer |

---

## 2. What Changes（具体改动清单）

### 2.1 新增文件（10 个核心 + 4 测试 + 1 脚本）

#### 核心实现（10 个）

| 路径 | 行数估计 | 内容 |
|------|----------|------|
| `hyper_mve/models/__init__.py` | ~30 | 暴露 `TriContextEncoder`, `BeliefNet`, `belief_loss`, `build_oracle_z_seq`（与 Pkg-04/05 共享 module 入口） |
| `hyper_mve/models/c_encoder.py` | ~60 | `CEncoder` 模块：MLP[1→32→32→16] + ReLU（无内部 LN，**P7 修订**） |
| `hyper_mve/models/role_encoder.py` | ~130 | `RoleEncoder` 模块：id_emb + **type_emb (v4)** + cap_emb concat → 32；cap_mlp 输入归一化（**C 修订**，调用 `CapabilityVector.normalize()`） |
| `hyper_mve/models/belief_encoder.py` | ~110 | `BeliefEncoder` 模块：raw (c_hat, z_hat) → Pool + 投影 → 32 维 belief_vec（**P5 拆出**，与 c/role encoder 对称） |
| `hyper_mve/models/belief_net.py` | ~280 | `BeliefNet` 主类：obs_encoder + GRU + head_c + head_opp + step/forward 双 API |
| `hyper_mve/models/belief_losses.py` | ~140 | `l_c` / `l_opp` (**v4 Oracle CE**) / `l_div` (**hinge, P1 偏差备案**) / `belief_loss` 组合器 + `build_oracle_z_seq` |
| `hyper_mve/models/permutation_invariant_pool.py` | ~150 | `MeanPool` / `MaxPool` / `AttentionPool`（含 learnable query 默认 + b_i-query 变体）+ 工厂 |
| `hyper_mve/models/tri_context_encoder.py` | ~120 | `TriContextEncoder` 主类：调三 encoder + LN 每路 + concat → 80 维 ctx_i（仅 concat 协调，**LN 责任集中** P7） |
| `hyper_mve/models/_belief_obs_encoder.py` | ~70 | BeliefNet 内部观测编码器 `[obs_dim→128→64]`（独立于 Pkg-04 主 RepNet） |
| `hyper_mve/models/_belief_id_emb.py` | ~30 | head_opp 用的 agent_id embedding（独立于 RoleEncoder.id_emb，语义分离） |

#### 测试（4 个）

| 路径 | 覆盖 |
|------|------|
| `tests/models/__init__.py` | – |
| `tests/models/test_tri_context.py` | TriContextEncoder 三路 dim、梯度流、permutation invariance |
| `tests/models/test_belief_net.py` | GRU step/seq 等价、heads 输出形状、Stage 1 oracle 注入 |
| `tests/models/test_belief_losses.py` | L_c MSE、L_opp Oracle CE、L_div hinge variance |
| `tests/models/test_permutation_pool.py` | mean/max/attention 一致性 |

#### 验证脚本（1 个）

| 路径 | 用途 |
|------|------|
| `hyper_mve/scripts/test_belief_net_synth.py` | 10K 步合成数据训练；输出 head_c MSE / head_opp acc / b_i 方差 TensorBoard 曲线 |

### 2.2 修改文件（1 个，additive patch）

| 文件 | 改动 | 性质 |
|------|------|------|
| `hyper_mve/schemas/capability.py` (Pkg-01) | 新增 `CapabilityVector.normalize()` read-only 方法 + 模块级常量 `CAP_NORM_LO/HI` | **Additive bug-fix patch**（既有字段不动；env/buffer/info 仍存 raw cap；仅 Pkg-03 cap_mlp 消费时调用） |

**Pkg-01 状态影响**：Q1 已关闭，Q2/Q3/Q4 仍开着；本改动是 additive 不破坏 finalized 状态。详见 Pkg-01 [spec 02-capability-vector.md §3.5](../pkg-01-foundation-schema/specs/02-capability-vector.md)。

其他 Pkg-01/02 文件**不修改**：`ModelConfig.belief_gru_hidden=128`、`belief_pool="mean"`、`d_belief_proj=16` 字段已就位；Pkg-02 `env.info["c_true"]` / `info["types"]` Oracle 信号已暴露。

### 2.3 删除文件（0 个）

v4.7 的 `context_encoder.py` / `gru_context_encoder.py` 已在 Pkg-01 归档至 `_legacy_v4_7/models/`，本包无需删除。

---

## 3. Capabilities（完成后系统获得的新能力）

### 3.1 用户视角

```python
# 1. 一行创建 BeliefNet
from hyper_mve.configs import V4Config
from hyper_mve.models import BeliefNet, TriContextEncoder

cfg = V4Config.from_preset("medium")
belief_net = BeliefNet(cfg.env, cfg.model)
encoder = TriContextEncoder(cfg.env, cfg.model)

# 2. BeliefNet 单步推断（在线 inference, worker 内调用）
import torch
B, N, obs_dim = 2, 4, 99
obs_t = torch.zeros(B, N, obs_dim)
prev_hidden = belief_net.init_hidden(batch_size=B, num_agents=N)  # (B, N, 128)

new_hidden, c_hat, z_hat = belief_net.step(obs_t, prev_hidden)
assert c_hat.shape == (B, N)              # sigmoid scalar per agent (v4)
assert z_hat.shape == (B, N, N-1, 2)      # softmax per (i, opponent)

# 3. BeliefNet 序列推断（trainer 内 K-step unroll, 离线训练）
obs_seq = torch.zeros(B, T, N, obs_dim)   # T = unroll_K + 1
hidden_seq, c_hat_seq, z_hat_seq = belief_net.forward(obs_seq)
assert c_hat_seq.shape == (B, T, N)
assert z_hat_seq.shape == (B, T, N, N-1, 2)

# 4. TriContextEncoder：组合三路得到 ctx_i ∈ ℝ^80
c_t = torch.zeros(B, 1)                   # (B, 1) shared context scalar
agent_ids = torch.tensor([[0, 1, 2, 3]] * B)  # (B, N)
cap_i = torch.zeros(B, N, 4)              # (B, N, 4) CapabilityVector
types = torch.zeros(B, N, dtype=torch.long)  # (B, N) AgentType.value
belief_i_raw = (c_hat, z_hat)             # 见 spec 01 详细 signature

ctx_i = encoder.forward(c_t, agent_ids, types, cap_i, belief_i_raw)
assert ctx_i.shape == (B, N, 80)          # 16 c_ctx + 32 role + 32 belief

# 5. 三 loss 联合计算（Pkg-05 trainer 调用）
from hyper_mve.models import belief_loss
c_true = torch.zeros(B, T)                # (B, T) Oracle from env.info["c_true"]
types_true = torch.zeros(B, T, N, dtype=torch.long)  # (B, T, N) Oracle from env.info["types"]
loss, breakdown = belief_loss(
    c_hat_seq, z_hat_seq, hidden_seq,
    c_true, types_true,
    weights=(1.0, 0.5, 0.01),             # λ_c, λ_opp, λ_div
)
# breakdown == {"l_c": ..., "l_opp": ..., "l_div": ..., "total": ...}
```

### 3.2 工程团队视角

- **零跨包数据契约偏移**：BeliefNet 输出严格按 Pkg-01 TimeStepRecord 的 c_hat / z_hat 字段契约（含 z_hat 顺序约定 "agent_id 升序跳过 self"）
- **可独立测试**：每个 encoder / loss / pool 都有独立单测；合成数据收敛实验可在 1-2 GPU 小时内完成
- **模块可组合**：Pkg-06a `shared_backbones.py` 直接 `from hyper_mve.models import BeliefNet, TriContextEncoder` 实例化复用（无需重复实现）
- **Stage 1 oracle 注入接口预留**：BeliefNet.forward 接受 `oracle_z: Optional[Tensor]` 参数；课程 Stage 1 用 oracle τ one-hot 替代 head_opp 输出（实现细节见 spec 08）

### 3.3 论文视角

完成本包后，**Ch4.2 + Ch4.5 全部章节可填实**：

- **Ch4.2.1** c_ctx 客观通路：c_encoder 维度对照表 + MLP 结构图
- **Ch4.2.2** role 通路（含 type_emb v4 关键）：role_encoder 8+8+16=32 拆分表
- **Ch4.2.3** belief 通路（含 v4 关键改动）：head_c / head_opp 输出维度对照
- **Ch4.2.4** 三通路汇总：ctx_i 80 维总维度表
- **Ch4.5.1** L_c MSE：公式 + 合成数据 5K 步收敛曲线
- **Ch4.5.2** L_opp Oracle CE（v4 关键）：公式 + 与 v3 自监督的对比表
- **Ch4.5.3** L_div hinge variance：公式 + collapse 长训练实测
- **Ch4.5.4** 三 loss 总权重 + 联合训练损失
- **Ch4.5.5** 与主任务耦合：λ_b 默认值（具体课程 ramp 在 Pkg-05）

---

## 4. Impact（影响分析）

### 4.1 对 v4.7 现有功能的影响

**保护**：完全脱钩 v4.7。v4.7 `context_encoder.py` / `gru_context_encoder.py` 已归档至 `_legacy_v4_7/models/`，可独立 import 用于回归对比测试。

**破坏**：无（本包是新增）。

### 4.2 对后续包的影响

| 后续包 | 本包提供 | 影响 |
|--------|----------|------|
| **Pkg-04 DualHyperNetwork v2** | `TriContextEncoder.forward(...) → ctx_i ∈ ℝ^80` | hyper_rew/hyper_pred 直接消费 80 维 ctx_i；hyper_trans 仅消费 c_ctx 子维度 |
| **Pkg-04 HyperMuZeroModel** | `BeliefNet` 实例 | model.set_context(c_t, τ_i, cap_i, b_i) 内的 b_i 由 BeliefNet 提供 |
| **Pkg-05 Trainer** | `belief_loss(...)` | trainer 主循环联合优化 L_main + λ_b · L_BeliefNet；课程 Stage 切换通过 `belief_net.forward(oracle_z=...)` 控制 |
| **Pkg-05 Worker** | `belief_net.step(...)` 在线推断 | worker 每步调用 step 取得 c_hat / z_hat，写入 TimeStepRecord（即 belief 状态进 buffer） |
| **Pkg-06a/b Baselines** | `BeliefNet` + `TriContextEncoder` 工厂复用 | 7 baseline 共享同一 BeliefNet 实例（断言 B 公平性强制要求） |
| **Pkg-07 Eval Protocols** | head_c MSE + head_opp accuracy 评估指标 | Self-Info 评估时 ẑ 必须来自 BeliefNet 推断（不允许 oracle leak） |

### 4.3 对论文贡献的影响

**直接贡献**：
- **断言 B 实验前提**：BeliefNet 提供 belief 专用容量的功能模块
- **断言 C 实验前提**：三路通路完整实现，可消融任一路
- **Ch4.2 完整数据**：80 维 ctx_i 拆分表 + 三路维度对照
- **Ch4.5 完整数据**：三 loss 公式 + 5K 步合成数据收敛曲线

**间接贡献**：
- **Harsanyi 对应的首次架构实现**：head_opp 类型 2 分类 + Oracle 监督是 Ch4.1.4 表格"他人 type 信念"在深度世界模型中的首次落地
- **type_emb 进 role 的架构论证**：从根本解决类型梯度撕裂的硬件基础（v3 共享 RewardHead 撕裂的根本原因是缺失这个通路）
- **课程学习接口预留**：Stage 1 oracle 注入接口允许 Pkg-05 trainer 实现"先 oracle 后推断"的渐进训练，避免 chicken-and-egg 困境

### 4.4 风险与对策

| 风险 | 触发条件 | 对策 |
|------|----------|------|
| head_opp Oracle 监督在长训练中过拟合 | trainer 后期 BeliefNet 在 buffer 中重复看到同一 (obs_seq, types) 对 | buffer 5000 episode 自然提供多样性；监控 train/val accuracy gap |
| BeliefNet collapse (b_i 输出方差骤降) | 长训练 BeliefNet 输出在所有 agent 间收敛为常数 | D5 L_div hinge variance 阻断；监控 b_i^t batch 方差，< 0.1 触发 alarm |
| head_c MSE 不下降（合成数据 5K 步内 > 0.05） | c_encoder 投影维度过小 / MSE 梯度被 L_opp 主导 | 默认权重 (1.0, 0.5, 0.01) 已平衡；如需调整在 Pkg-05 trainer 配置层 |
| z_hat 顺序错位导致 L_opp 标签污染 | head_opp 输出顺序与 Pkg-01 z_hat 字段约定不一致 | spec 05 显式约定 "agent_id 升序跳过 self"；单测 `test_z_hat_order_convention` 验证 |
| Pkg-05 trainer 误把 `env.info["types"]` 传入 BeliefNet forward（破坏 Self-Info） | trainer 端代码错误用 oracle 作 model 输入 | spec 08 明确："Oracle types 仅作 L_opp CE 标签，不可作 BeliefNet forward 输入"；Pkg-05 spec 04 trainer loop 强制接口分离 |
| GRU uni-directional 在长 episode 早期推断不足 | T=300 (Hard) 前 50 步信息量低 | 可接受（agent 也只能用过去信息）；head_c 早期 MSE 高是预期行为 |
| 三路 LayerNorm 后量级仍不平衡 | concat 后 hypernet 输入分布偏斜 | D7 已 LayerNorm 每路；如有问题在 Pkg-04 hypernet 入口再加一层 input LayerNorm |

---

## 5. References

| 来源 | 引用条款 |
|------|----------|
| `D:\RL\hyper_mve\docs\Chapter4_Architecture_v4.md` | §4.2.1-4.2.4 三联通路、§4.5.1-4.5.5 BeliefNet 损失、§4.6.5 梯度门控 |
| `D:\RL\hyper_mve\docs\Chapter4_1_Motivation_v4.md` | §4.1.3 三层 motivation（主层 type_emb）、§4.1.4 Harsanyi 对应 |
| `D:\RL\hyper_mve\docs\Chapter5_Planner_Training_v4.md` | §5.7 课程学习三阶段（Stage 1 oracle 接口要求） |
| `D:\RL\hyper_mve\docs\Hyper_MuZero_v4_Roadmap.md` | §4 Stage 1 Week 2 任务清单 |
| Pkg-01 SDD | TimeStepRecord c_hat / z_hat 字段契约 + ModelConfig BeliefNet 字段 |
| Pkg-02 SDD | env.info Oracle 信号契约（c_true / types / caps）|
| 项目 Plan File | Part 2 Pkg-03 + Part 9 v4 修订 |

---

## 6. Acceptance Criteria Summary

> 详见 [`design.md`](./design.md) 与各 `specs/*.md`。本节仅速览。

### 结构硬约束（must pass）

- [ ] `test_tri_context.py::test_output_dim`: `encoder(c, aid, cap, belief).shape == (B, N, 80)`
- [ ] `test_tri_context.py::test_role_dim_exact`: role concat 后维度 == 32（8+8+16，精确无 pad）
- [ ] `test_tri_context.py::test_gradient_flow`: 反向传播覆盖三路 + 三 head 所有参数有梯度
- [ ] `test_belief_net.py::test_head_opp_output_shape`: head_opp 输出 (B, N, N-1, 2) softmax
- [ ] `test_belief_net.py::test_head_c_output_shape`: head_c 输出 (B, N) sigmoid ∈ [0, 1]
- [ ] `test_belief_net.py::test_z_hat_agent_id_order`: head_opp 输出顺序符合 z_hat 字段约定（agent_id 升序跳过 self）
- [ ] `test_belief_net.py::test_gru_step_seq_equiv`: `belief_net.step()` 逐步与 `forward(seq)` 输出差 ≤ 1e-5
- [ ] `test_belief_losses.py::test_l_opp_oracle_ce`: L_opp 是 CE，标签是 types_true (int)，不是动作
- [ ] `test_belief_losses.py::test_l_div_hinge`: variance < 0.1 时 loss > 0；variance >= 0.1 时 loss = 0

### 收敛验证（hard gate）

- [ ] `scripts/test_belief_net_synth.py`: 合成数据 5K 步内 head_c MSE < 0.05 + head_opp accuracy > 80%

### 论文章节增量

- [ ] Ch4.2.1 c_ctx 维度表（d_c=16，MLP 结构）
- [ ] Ch4.2.2 role 通路拆分表（d_id=8 + d_type=8 + d_cap=16 = 32）
- [ ] Ch4.2.3 belief 通路（v4 关键改动 head_opp 类型 2 分类）
- [ ] Ch4.2.4 三通路汇总（80 维 ctx_i）
- [ ] Ch4.5 三损失公式 + 5K 步合成数据收敛曲线

---

## 7. Out of Scope（明确不做）

- **不实现** Pkg-04 DualHyperNetwork v2（仅提供 TriContextEncoder 输出供其消费）
- **不实现** Pkg-05 课程学习 stage 切换逻辑（仅在 BeliefNet.forward 提供 `oracle_z: Optional[Tensor]` 参数接口）
- **不实现** belief gradient gating 的 step-conditional 切换（具体在 Pkg-04 model 内）
- **不实现** type-stratified sampling（属于 Pkg-05 buffer）
- **不实现** μP base_shape 集成（属于 Pkg-07）
- **不实现** 端到端训练脚本（仅合成数据收敛验证 scripts/test_belief_net_synth.py）
- **不实现** BeliefNet 的 EMA target net（用普通梯度训练即可，与 main task 的 EMA τ=0.99 是 Pkg-05 trainer 层的设计）
- **不引入新依赖**（仅 PyTorch + numpy）
- **不修改** `D:\RL\hyper_mve\docs\` 论文章节文件（如发现 spec 与 Ch4 不一致，先报 issue 再讨论）
