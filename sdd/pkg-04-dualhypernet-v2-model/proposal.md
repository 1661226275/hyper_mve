# Pkg-04: DualHyperNetwork v2 & HyperMuZero Model v2 — Proposal

| 元信息 | 值 |
|--------|----|
| **包 ID** | `pkg-04-dualhypernet-v2-model` |
| **状态** | Draft, awaiting review |
| **工期估计** | 1 周（5-7 天） |
| **GPU 算力** | 15 小时（Forward/Backward 单测 + 数值稳定测试） |
| **依赖** | Pkg-01 (Foundation Schema), Pkg-02 (env), Pkg-03 (TriContextEncoder + BeliefNet) |
| **被依赖** | Pkg-05 (Trainer), Pkg-06a/b (Baselines), Pkg-07 (Eval) |
| **论文章节** | **Ch4.3** 数据流 + **Ch4.4** 六网络表 + **Ch4.6** 四道防线（v4 扩展为 5 道） |
| **PR 体量** | 4 重写 + 1 新增 + 4 测试 + 1 验证脚本 + 3 处调用点迁移 |

---

## 1. Why（为什么需要这个包）

### 1.1 v4.7 DualHyperNetwork 在架构层无法支撑 v4 三大断言

v4.7 `DualHyperNetwork` 的 forward 签名是 `(rule_emb, id_emb)`（双路 24-32 维），与 Ch4 v4 的三路 80 维 ctx_aug 之间**结构性鸿沟**：

| 论文断言 | v4.7 缺失 → 后果 |
|----------|------------------|
| **断言 A** 类型梯度撕裂 | role 无 type_emb（Pkg-03 已加，但 v4.7 hypernet 不接 role 通路）→ hyper_rew 仍生成"类型平均" θ_rew → 无法验证 per-type 容量分配 |
| **断言 B** 信念专用容量 | hypernet 不接 belief 通路 → 与 Input-Wide/Deep 的容量分配几何对比无对照组 → μP 等参实验无意义 |
| **断言 C** 三路必要性 | 仅 rule + id 两路 → Ablation 2 移除"任一路"的"三路"压根不存在 |

**没有本包重写 DualHyperNetwork，前三大断言的架构基础不存在**。Pkg-03 提供了 TriContextEncoder + BeliefNet 把三路组装成 80 维 ctx_aug，但**消费这 80 维并生成 per-agent θ 的能力**必须由本包提供。

### 1.2 v4.7 Oracle / Infer 两模型类无法统一消费 Pkg-03 BeliefNet

v4.7 把 HyperMuZero 拆为：
- `OracleHyperMuZeroModel` (Exp2)：rule 由训练脚本显式给真值
- `InferHyperMuZeroModel` (Exp3)：rule 由模型内部 GRU 从历史推断

v4 用 Pkg-03 BeliefNet 统一替代 v4.7 的 `gru_context_encoder.py`（已归档），**Oracle/Infer 的二分应折叠为单一 `HyperMuZeroModel`**，由 BeliefNet 提供 belief，课程学习（Pkg-05）控制 Stage 1 oracle 注入 vs Stage 3 纯推断。这要求 model 的 `set_context` API 重新设计——v4.7 `set_context(rule, agent_id)` 两参签名废弃。

### 1.3 Ch4.6.5 belief 梯度门控（v4 新增防线 5）是 model 内部责任

Ch4.6.5 明确：前 5K step（`cfg.train.belief_grad_gating_steps=5000`）belief 通路梯度被 `.detach()` 切断，主任务 loss 不通过 BeliefNet 参数。**这个切断必须在 model.forward 内部实现**——BeliefNet 自身（Pkg-03）的 forward 不能做（否则 BeliefNet 的独立 L_belief loss 也被切断）。

具体设计（与 Pkg-03 spec 08 §5 协作）：
- BeliefNet 输出 raw heads (c_hat, z_hat) 由 Pkg-03 提供
- Pkg-04 model.forward 内 `if self._step < 5000: c_hat, z_hat = c_hat.detach(), z_hat.detach()`
- BeliefEncoder（Pkg-03 范围）在 detach 之后接收，**main loss 经 BeliefEncoder 投影 MLP 反向**（投影 MLP 仍训练）→ 仅 BeliefNet GRU/heads 不被 main loss 更新
- BeliefNet 自身的 L_belief loss（Pkg-03 belief_loss）仍**始终更新** BeliefNet（独立优化路径）

### 1.4 v4.7 mve_planner.py 4 处 set_context 调用点的迁移责任

`planning/mve_planner.py` 在 L71/213/219/258 调用 `model.set_context(rule_exp, id_i)`（duck-typed 双参签名）。Pkg-04 改 set_context 为两步分离 API（用户决议 Q3）后，**这 4 处必须同步迁移到 `set_context_objective` + `set_context_subjective`**。

按用户决议 Q2：mve_planner.py 改造**归 Pkg-04 范围**（理由：planner 是 model.set_context 的直接调用者，签名兼容性由 model 包负责）。这是 Pkg-04 的 PR 范围一部分（spec 08 §3 列出 4 处迁移行号 + 迁移模板）。

### 1.5 对应路线图条款

| Roadmap / Chapter / Plan 条款 | 本包如何响应 |
|------------------------------|----------------|
| Roadmap §4 Stage 1 Week 3：DualHyperNetwork v2 重写 | 本包 4 重写 + 1 新增 |
| Ch4.3.1 数据流总图 | spec 06 mermaid 图 + 张量 shape 对照 |
| Ch4.3.2 客观通路 hyper_trans 仅 c_ctx | spec 01 forward_trans API + 单测 |
| Ch4.3.3 主观通路 hyper_rew/hyper_pred 三联输入 | spec 01 forward_subjective API |
| Ch4.4 六大网络客观-主观分工 | spec 06 数据流图 + RepNet+StateTransNet 客观流 / BeliefNet+RewardHead+PredictionNet 主观流 |
| Ch4.6 四道稳定性防线（v4 扩展 5 道） | spec 03 防线 1-4 + spec 04 防线 5 |
| Ch4.6.5 belief 梯度门控 | spec 04 全 spec 内容 |
| Plan Part 2 Pkg-04 8 Decisions | design.md §3 D1-D8 完整 |
| Plan 修订 Q1-Q4 决议（spec 数=8 / planner 归 Pkg-04 / set_context 两步 / update_step 独立） | README + design §8 + spec 02 + spec 04 + spec 08 |

---

## 2. What Changes（具体改动清单）

### 2.1 重写文件（4 个）

| 路径 | 行数估计 | v4.7 现状 → v4 改动 |
|------|----------|---------------------|
| `hyper_mve/models/hyper_network.py` | ~250 | v4.7 `__init__(rule_emb_dim, id_emb_dim, ...)` + 单一 forward(rule_emb, id_emb)；v4 `__init__(c_ctx_dim=16, ctx_aug_dim=80, ...)` + 双 forward：`forward_trans(c_ctx) → θ_state` + `forward_subjective(ctx_aug) → (θ_rew, θ_pred)`；HyperNetMLP 保留（L2 norm + output_scale 不变）|
| `hyper_mve/models/hyper_muzero_model.py` | ~350 | v4.7 OracleHyperMuZeroModel / InferHyperMuZeroModel 两类废弃；v4 统一 `HyperMuZeroModel`，5 个新 API：`update_step / set_context_objective / set_context_subjective / encode / transition / predict_reward / predict` |
| `hyper_mve/models/functional_nets.py` | ~50 行 diff | 保留 v4.7 主体（AdaLN L43-77 + Δs 残差 L245-271 完整不动）；仅修改 obs_dim 输入维度通过 cfg 注入；spec 03 详述 |
| `hyper_mve/models/representation_net.py` | ~20 行 diff | obs_dim 从硬编码改为 `cfg.env.observation_space.shape` 派生（与 Pkg-02 ObservationLayout 一致） |

### 2.2 新增文件（1 核心 + 4 测试 + 1 验证脚本）

#### 核心实现（1）

| 路径 | 行数估计 | 内容 |
|------|----------|------|
| `hyper_mve/models/grad_gating.py` | ~80 | `BeliefGradGating` helper（不是 nn.Module 子类）：`apply(c_hat, z_hat, step)` 方法，step < threshold 时 .detach() 两 tensor，否则透传 |

#### 测试（4）

| 路径 | 覆盖 |
|------|------|
| `tests/models/test_hyper_network_v2.py` | hyper_trans 仅 c_ctx 不变性、forward_subjective 维度断言、output_scale 三初值、L2 norm 后 magnitude |
| `tests/models/test_hyper_muzero_model.py` | set_context_objective/subjective 缓存策略、type-aware reward 分化（cos-sim < 0.95）、update_step 状态管理 |
| `tests/models/test_stability_safeguards.py` | AdaLN (1+γ) 数值不变性、Δs 残差、small_init gain=0.01、4 道防线与 v4.7 数值回归对比 |
| `tests/models/test_grad_gating.py` | step < 5000 时 BeliefNet 参数梯度 norm = 0；step >= 5000 时 > 0；BeliefEncoder 参数始终有梯度（投影 MLP 不被 detach 影响） |

#### 验证脚本（1）

| 路径 | 用途 |
|------|------|
| `hyper_mve/scripts/test_hyper_model_forward.py` | medium config 100 步端到端 forward；性能 profile（输出单步时间 + 显存 + 参数量）；硬断言 `elapsed_ms < 15.0` |

### 2.3 修改文件（3 处调用点迁移，Q2 用户确认 Pkg-04 范围）

| 路径 | 改动 |
|------|------|
| `hyper_mve/planning/mve_planner.py` | L71/213/219/258 四处 `model.set_context(rule_exp, id_i)` → 改为先调一次 `model.set_context_objective(c_t)`（planner 入口），再循环 `model.set_context_subjective(id_i, cap[id_i], belief[id_i])` |
| `hyper_mve/training/muzero_trainer.py` | grep 找所有 set_context 调用点同步迁移；新增 `model.update_step(step)` 在 train_step 起点调用 |
| `hyper_mve/training/worker.py` | worker 内 set_context 调用点同步迁移；worker step API 适配新签名 |

### 2.4 删除文件（依赖 Pkg-01 归档已完成）

无需本包删除。以下 v4.7 文件**已在 Pkg-01 归档至 `_legacy_v4_7/`**：

- `models/baseline_model.py`（Pkg-06a MA-MuZero 取代）
- `models/oracle_muzero_model.py` / `infer_muzero_model.py`（本包统一 model 取代）
- `models/context_encoder.py` / `gru_context_encoder.py`（Pkg-03 TriContextEncoder + BeliefNet 取代）

---

## 3. Capabilities（完成后系统获得的新能力）

### 3.1 用户视角

```python
# 1. 一行创建 v4 统一模型（替代 v4.7 Oracle/Infer 二分）
from hyper_mve.configs import V4Config
from hyper_mve.models import HyperMuZeroModel

cfg = V4Config.from_preset("medium")
model = HyperMuZeroModel(cfg)

# 2. 训练循环（trainer 调用模式，Pkg-05 spec 04 详述）
for step in range(max_train_steps):
    model.update_step(step)                          # ← Q4: 独立方法管理 belief grad gating 阈值
    
    # K-step unroll 起点
    model.set_context_objective(c_t=batch["c_t"])    # ← Q3: 计算 θ_state 一次（所有 agent 共享）
    
    for k in range(num_agents):
        model.set_context_subjective(                # ← 切 agent 时换 θ_rew^k / θ_pred^k
            agent_id=k,
            cap_i=batch["cap"][k],
            belief=(c_hat[k], z_hat[k]),
        )
        # forward 调用
        s_next = model.transition(s, action)
        r_k = model.predict_reward(s, action)
        pi_k, v_k = model.predict(s)

# 3. mve_planner 调用模式（spec 08 §3 迁移指引）
def sample_action(model, root_s, ...):
    model.set_context_objective(c_t=rule_exp)
    for j in agent_order:
        model.set_context_subjective(j, cap[j], belief[j])
        # ... rollout ...

# 4. 评估时（Self-Info 严格，spec 08 §6）
# 与 trainer 完全相同 API，但 belief 必须来自 BeliefNet 推断（不传 oracle_z）
model.update_step(eval_step)
model.set_context_objective(c_t)
for k in agents:
    model.set_context_subjective(k, cap[k], belief=(c_hat_predicted[k], z_hat_predicted[k]))
```

### 3.2 工程团队视角

- **零跨包数据契约偏移**：model 输入严格按 Pkg-01 schema（ModelConfig 字段）+ Pkg-03 TriContextEncoder / BeliefNet 接口
- **API 稳定性承诺**：7 个对外接口在 spec 08 §1 锁定，Pkg-05/06/07 可放心消费
- **梯度门控自动化**：trainer 只需调 `update_step(step)`，model 内部自动判断是否 .detach()（无需手动管 detach）
- **可测试性**：4 道防线 + 1 道梯度门控每道都有具名单测，11 项硬约束全部覆盖

### 3.3 论文视角

完成本包后，**Ch4.3 + Ch4.4 + Ch4.6 全部章节可填实**：

- **Ch4.3.1** 数据流总图：spec 06 mermaid 图
- **Ch4.3.2** hyper_trans 仅 c_ctx：spec 01 + 单测 `test_hyper_trans_only_c_ctx` 数据
- **Ch4.3.3** hyper_rew/pred 三联输入：spec 01 forward_subjective + 参数量分解（spec 07）
- **Ch4.3.4** 参数量表：spec 07 §3 完整对照 Ch4.3.4 ~3.2M
- **Ch4.4** 六大网络客观-主观分工：spec 06 数据流图 + 张量 shape 表
- **Ch4.6.1-4.6.4** 四道防线 + (v4 新增) 4.6.5 belief 梯度门控：spec 03 + spec 04
- **断言 A 物理基础**：spec 05 type-aware reward 分化实测（cos-sim < 0.95）

---

## 4. Impact（影响分析）

### 4.1 对 v4.7 现有功能的影响

**保护**：v4.7 已通过 Pkg-01 归档至 `_legacy_v4_7/`，可独立运行（`git checkout v4.7-final`）。

**破坏（必要的）**：
- v4.7 `OracleHyperMuZeroModel` / `InferHyperMuZeroModel` 类废弃（在 `_legacy_v4_7/` 内仍可读）
- v4.7 `DualHyperNetwork.__init__` / `forward` 签名 incompatible 变更（rule_emb/id_emb 双参 → c_ctx/ctx_aug 双 forward API）
- v4.7 `model.set_context(rule, agent_id)` 两参签名废弃（统一为两步分离）
- mve_planner.py / muzero_trainer.py / worker.py 三处调用点必须同步迁移（spec 08 §3）

### 4.2 对后续包的影响

| 后续包 | 本包提供 | 影响 |
|--------|----------|------|
| **Pkg-05 Trainer & Worker** | 7 个 model API + grad gating 自动化 + Stage 接入预留 | 直接消费，spec 08 §3-§4 是 Pkg-05 SDD 启动条件 |
| **Pkg-06a MA-MuZero / MAMBA / MARIE / Input variants** | shared_backbones 工厂复用 RepNet + BeliefNet 实例（共享 backbone 公平性） | 断言 B 等参对比的架构基础 |
| **Pkg-06b MAPPO / QMIX / Conflict-Aware GA** | RepNet + BeliefNet 复用（虽然这些 baseline 用 policy gradient / Q-learning，但 BeliefNet 共享是公平性硬约束） | 见 spec 08 §5 |
| **Pkg-07 μP + Eval Protocols** | model.set_context_subjective 接口稳定 → Self-Info eval 可放心调用 | spec 08 §6 |
| **Pkg-08 Experiments** | 实验脚本调用 model API（不直接 import 任何内部模块） | 通过 spec 08 §1 API 表稳定使用 |

### 4.3 对论文贡献的影响

**直接贡献**：
- **断言 A 物理基础**：spec 05 type-aware reward 分化实测（cos-sim < 0.95）→ Ch4.1.1 偏导表 + Pkg-02 reward 计算 + 本包 hyper_rew 链路完整闭环
- **断言 B 实验前提**：DualHyperNetwork v2 与 Input-Wide/Deep 等参对比的"Hyper 路径"
- **断言 C 实验前提**：三路通路完整实现，可消融任一路
- **Ch4.3-4.6 完整数据**：所有数据流图 / 网络表 / 防线表 / 参数量表的具体实现

**间接贡献**：
- **梯度门控的首次架构实现**：Ch4.6.5 是 v4 新增防线，本包是第一个完整实现 + 单测验证的载体
- **set_context 两步分离 API**：N agents 共享 θ_state 节省 ~30% forward 开销（hyper_trans 在 N agents 间只算一次），为大规模实验铺路

### 4.4 风险与对策

| 风险 | 触发条件 | 对策 |
|------|----------|------|
| **R1** 三路输入维度不匹配 | TriContextEncoder.d_ctx_aug != 80 | spec 01 __init__ 内 `assert cfg.model.d_ctx_aug == 80`；Pkg-01 spec 05 已有 ModelConfig __post_init__ 验证（双重保险） |
| **R2** z_hat 顺序约定被破坏 | Pkg-04 调用 BeliefNet 时顺序错位 | 完全复用 Pkg-03 spec 08 契约 6 验证（端到端 build_oracle_z_seq → l_opp 闭环） |
| **R3** Self-Info 违反 | set_context_subjective 误接 oracle types | spec 02 + spec 08 §6 + 单测 `test_set_context_subjective_no_oracle_types_leak` |
| **R4** belief grad gating 5K 阈值未同步 | model 硬编码 vs cfg 不一致 | 严格从 `cfg.train.belief_grad_gating_steps` 读取（spec 04） |
| **R5** hyper_trans 误接 role/belief | forward_trans 签名 drift | 单测 `test_hyper_trans_only_c_ctx`：改 role 输入时 θ_state 输出不变 |
| **R6** output_scale 初值错误 | v4.7 → v4 迁移遗忘 rew=0.1 | spec 03 三初值表 + 单测 `test_output_scale_inits_trans_rew_pred` |
| **R7** RewardHead type-aware 未实现 | type-aware 路径理解偏差 | spec 05 链路图 + 单测 `test_type_aware_reward_differentiation`（cos-sim < 0.95）|
| **R8** Forward 性能超 15ms 预算 | hyper 三模块 + functional 三网络串联 | spec 07 性能预算自带单测护栏（M7：`assert elapsed_ms < 15.0`） |
| **R9** Pkg-03 BeliefNet 接口变化 | Pkg-03 spec 08 未完全锁定 | Pkg-03 spec 08 §9 API 稳定性表已 finalized；spec 08 §7 引用确认 |
| **R10** Δs 残差与 v4.7 不一致 | functional_nets 迁移疏漏 | spec 03 防线 4 + 单测 `test_state_trans_delta_s_residual` 与 v4.7 数值回归对比 |

---

## 5. References

| 来源 | 引用条款 |
|------|----------|
| `D:\RL\hyper_mve\docs\Chapter4_Architecture_v4.md` | §4.3 数据流 + §4.4 六网络表 + §4.6 防线 + §4.6.5 梯度门控 |
| `D:\RL\hyper_mve\docs\Hyper_MuZero_v4_Roadmap.md` | §4 Stage 1 Week 3 |
| 项目 Plan File | Part 2 Pkg-04 完整规格 + 8 Decisions |
| `C:\Users\zhengwenbo01\.claude\plans\review-pkg-01-additive-patch-pkg-04-starry-quiche.md` | Q1-Q4 用户决议 + 7 项避免事后修改机制 |
| `D:\RL\hyper_mve\hyper_mve\models\hyper_network.py` | v4.7 L63-82 (L2 norm + output_scale) 保留 |
| `D:\RL\hyper_mve\hyper_mve\models\functional_nets.py` | v4.7 L43-77 (AdaLN) + L245-271 (Δs 残差) 保留 |
| `D:\RL\hyper_mve\hyper_mve\planning\mve_planner.py` | L71/213/219/258 四处 set_context 调用点（Pkg-04 范围迁移） |
| Pkg-01 SDD | spec 02 (CapabilityVector.normalize), spec 04 (TimeStepRecord), spec 05 (ModelConfig 字段) |
| Pkg-02 SDD | spec 08 (env.info 三段分组) |
| Pkg-03 SDD | spec 01 (TriContextEncoder), spec 04/05 (BeliefNet), spec 06 (belief_loss), spec 08 (集成契约) |

---

## 6. Acceptance Criteria Summary

> 详见 [`design.md`](./design.md) §2 与各 `specs/*.md` §5。本节仅速览。

### 结构硬约束（must pass）

- [ ] `test_hyper_network_v2.py::test_hyper_trans_only_c_ctx`: 改 role 输入时 θ_state 不变（C1）
- [ ] `test_hyper_network_v2.py::test_subjective_input_dim_80`: forward_subjective 输入 shape == (B, 80)（C2）
- [ ] `test_hyper_network_v2.py::test_ctx_aug_dim_eq_80`: cfg.model.d_ctx_aug == 80 断言（C5）
- [ ] `test_hyper_network_v2.py::test_output_scale_inits_trans_rew_pred`: 三初值 0.01/0.1/0.01（C8）
- [ ] `test_stability_safeguards.py::test_adaln_one_plus_gamma_factor`: AdaLN (1+γ) 数值不变性（C7）
- [ ] `test_stability_safeguards.py::test_state_trans_delta_s_residual`: Δs 残差（C6）
- [ ] `test_grad_gating.py::test_pre_5k_belief_detached`: step < 5000 时 BeliefNet 参数梯度 norm = 0（C4）
- [ ] `test_grad_gating.py::test_post_5k_belief_grad_flow`: step >= 5000 时 BeliefNet 参数梯度 > 0
- [ ] `test_hyper_muzero_model.py::test_type_aware_reward_differentiation`: 同一 (s, a) 下 α/β agent reward cos-sim < 0.95（C9, 断言 A 物理基础）
- [ ] `test_hyper_muzero_model.py::test_set_context_subjective_no_oracle_types_leak`: Self-Info 严格性（C11）

### 性能/集成（must pass）

- [ ] `scripts/test_hyper_model_forward.py`: medium config 100 步无 NaN + reward ∈ [-3, 3] + 单步 < 15ms（R8 硬阈值）
- [ ] 参数量 ∈ [2.8M, 3.6M]（与 Ch4.3.4 ~3.2M 一致）
- [ ] mve_planner.py 4 处调用点全部迁移完成（grep 验证无 `set_context\(.*,.*\)` 双参签名）

### 论文章节增量

- [ ] Ch4.3 全章数据流 + 张量 shape 表
- [ ] Ch4.4 六网络客观-主观分工表
- [ ] Ch4.6.1-4.6.4 四道防线 + 4.6.5 梯度门控（v4 扩展为 5 道）
- [ ] Ch4.3.4 参数量分解表

---

## 7. Out of Scope（明确不做）

- **不实现** Pkg-05 trainer 主循环（仅提供 model API 供其消费）
- **不实现** 课程学习 Stage 切换逻辑（仅在 set_context_subjective 接受 belief tuple，Stage 切换由 Pkg-05 控制）
- **不实现** μP base_shape 集成（属于 Pkg-07）
- **不实现** baseline 适配（属于 Pkg-06a/b）
- **不实现** type-aware RewardHead 内部分支（按 D2 + spec 05，type-aware 完全通过 type_emb → hyper_rew 路径隐含）
- **不引入新依赖**（仅 PyTorch + Pkg-01/02/03 已暴露接口）
- **不修改** Pkg-01/02/03 任何 spec 或代码（仅消费已发布的接口）
- **不修改** `D:\RL\hyper_mve\docs\` 论文章节文件
