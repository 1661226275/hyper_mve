# Pkg-04: DualHyperNetwork v2 & HyperMuZero Model v2

> **状态**：Draft — awaiting user review
> **包 ID**：`pkg-04-dualhypernet-v2-model`
> **工期**：1 周（5-7 天） · **GPU 算力**：15 小时（Forward/Backward 单测 + 数值稳定测试） · **PR 体量**：10 新增/重写文件 + 4 测试 + 1 forward smoke script

---

## 📚 文档导览

| 文档 | 目的 | 字数 |
|------|------|------|
| [`proposal.md`](./proposal.md) | **先读这个**：Why / What Changes / Capabilities / Impact | ~4000 |
| [`design.md`](./design.md) | 设计抉择：Context / Goals / 8 项 Decisions（含 Q1-Q4 决议） | ~5000 |
| [`specs/01-dualhypernet-v2-api.md`](./specs/01-dualhypernet-v2-api.md) | DualHyperNetwork v2 接口（双 forward API：trans/subjective） | ~2000 |
| [`specs/02-hyper-muzero-model-v2.md`](./specs/02-hyper-muzero-model-v2.md) | HyperMuZeroModel v2（5 个核心 API + ModelConfig 字段穷举） | ~2500 |
| [`specs/03-stability-safeguards-preservation.md`](./specs/03-stability-safeguards-preservation.md) | 4 道稳定性防线保留 + v4.7→v4 行号映射 | ~2000 |
| [`specs/04-belief-gradient-gating.md`](./specs/04-belief-gradient-gating.md) | grad_gating.py + 5K step `.detach()`（v4 新增防线 5） | ~1500 |
| [`specs/05-type-aware-reward.md`](./specs/05-type-aware-reward.md) | type_emb → hyper_rew → θ_rew^i 路径（断言 A 物理基础） | ~1500 |
| [`specs/06-data-flow-diagram.md`](./specs/06-data-flow-diagram.md) | mermaid 数据流图 + 张量 shape 对照表 + N agents 调用顺序 | ~1500 |
| [`specs/07-forward-performance-budget.md`](./specs/07-forward-performance-budget.md) | 单步 < 15ms 硬阈值 + 参数量 ~3.2M + 显存 < 8GB | ~1500 |
| [`specs/08-integration-contracts.md`](./specs/08-integration-contracts.md) | **对外硬契约**：API 稳定性 + v4.7→v4 三处调用点迁移 | ~2500 |

---

## 🎯 一句话目标

**重写 v4.7 DualHyperNetwork 为 v4 三路输入版**（hyper_trans 仅 c_ctx 16 维 / hyper_rew + hyper_pred 接 80 维 ctx_aug），**统一 Oracle/Infer 两模型为单一 HyperMuZeroModel**（`set_context_objective` + `set_context_subjective` 两步分离 API），**保留 4 道稳定性防线 + 新增 belief gradient gating 防线 5**，**同步迁移 mve_planner.py / muzero_trainer.py / worker.py 三处 set_context 调用点**，为 Pkg-05 trainer 提供完整可消费 model API。

---

## ✅ 关键 Acceptance Criteria（速览）

### 结构硬约束（must pass）

- [ ] **DualHyperNetwork v2 forward_trans 仅接 c_ctx (B, 16)**（C1：改 role 时 θ_state 不变）
- [ ] **forward_subjective 接 ctx_aug (B, 80)**（C2：维度精确）
- [ ] **AdaLN `h × (1 + γ) + β` factor (1+) 完整保留**（C7：与 v4.7 functional_nets.py:43-77 数值一致）
- [ ] **output_scale 三初值**：trans=0.01, rew=0.1, pred=0.01（C8）
- [ ] **StateTransNet 预测 Δs（残差）**（C6：v4.7 functional_nets.py:245-271 保留）
- [ ] **belief gradient gating**：step < 5000 时 BeliefNet 参数梯度 norm = 0（C4 + Q4）
- [ ] **mve_planner.py 4 处 set_context 调用点全部迁移到两步分离 API**（spec 08 §3）

### 收敛/性能验证

- [ ] **单步 forward < 15 ms** (V100, Medium config B=256, N=4)（**硬 fail 阈值**）
- [ ] **参数量 ∈ [2.8M, 3.6M]**（与 Ch4.3.4 ~3.2M 一致）
- [ ] **type-aware reward 分化**：同一 (s, a) 输入下 α/β agent reward cos-sim < 0.95（断言 A 物理基础）

### 期望达到

- [ ] `pytest tests/models/test_hyper_network_v2.py tests/models/test_hyper_muzero_model.py tests/models/test_stability_safeguards.py tests/models/test_grad_gating.py` 全过
- [ ] `python scripts/test_hyper_model_forward.py --preset medium --steps 100` 端到端无 NaN
- [ ] `mypy --strict hyper_mve/models/{hyper_network,hyper_muzero_model,functional_nets,grad_gating}.py` 零错误

---

## 📅 实施顺序（7 天）

| 阶段 | 天数 | 任务 | 输出 / 审阅点 |
|------|------|------|---------------|
| **Phase 1** | Day 1 | README + proposal + design.md（8 Decisions 锁定） | **design.md 审阅**（最关键，决定后续 7 specs 走向） |
| **Phase 2** | Day 2 | spec 01 + spec 03（维度断言 + 稳定性护栏） | hyper_trans / hyper_subjective forward 接口锁定 |
| **Phase 3** | Day 3 | spec 02 + spec 04（API 签名 + 梯度门控） | set_context 双 API + grad_gating 实现路径锁定 |
| **Phase 4** | Day 4 | spec 05 + spec 06（type-aware 路径 + 数据流图） | 断言 A 实现机制 + mermaid 图 |
| **Phase 5** | Day 5 | spec 07 + spec 08（性能预算 + 对外契约） | **Pkg-05 SDD 启动条件**（spec 08 §3-§4 完整） |
| **Phase 6** | Day 6 | 全 spec **机器可校验**交叉引用核对（澄清 3）+ 11 项硬约束单测命名表对账 | 整包审阅，产出 ref_matrix.csv |
| **Phase 7** | Day 7 | PR description + 用户最终 ack | Pkg-04 SDD finalized |

---

## 🔑 8 项关键 Decisions（速览，含 Q1-Q4 用户决议）

| # | Decision | 推荐 | 详见 |
|---|----------|------|------|
| D1 | hyper_trans 输入 | **仅 c_ctx (16 维)**，不接 role / belief（C1 物理转移上下文不变） | [design §3 D1](./design.md) |
| D2 | RewardHead type-aware 实现 | **通过 type_emb → hyper_rew → θ_rew^i 隐含**（RewardHead 结构不变） | [design §3 D2](./design.md) |
| D3 | belief gradient gating 实现 | **显式 step-conditional `.detach()`**（不用 nn.Hook） | [design §3 D3](./design.md) |
| D4 | set_context 缓存策略 | 缓存 θ_state（objective 一次性算）+ 当前 agent θ_rew/θ_pred | [design §3 D4](./design.md) |
| D5 | hyper_pred detach context 配置 | **配置开关 `detach_pred_context=True`**（v4.8 继承） | [design §3 D5](./design.md) |
| D6 | role_i 自我类型 | **仅自报告**（ẑ 仅针对他人） | [design §3 D6](./design.md) |
| D7 | StateTransNet 输入 | **joint action onehot (N*A 平铺)**（与 v4.7 一致） | [design §3 D7](./design.md) |
| D8 | v4.7 → v4 替换策略 | **inplace 修改** + 旧版本在 git history 与 `_legacy_v4_7/` | [design §3 D8](./design.md) |

**用户已 ack 的 4 项关键决策（2026-05-28，在 design.md §8 锁定）**：

| Q | 决议 | 影响位置 |
|---|------|---------|
| **Q1** spec 数量 | **8 specs**（加 integration-contracts） | 本文档导览 + spec 08 |
| **Q2** mve_planner.py 改造归属 | **Pkg-04 范围** | spec 08 §3 |
| **Q3** set_context API 拆分 | **两步分离**：`set_context_objective(c_t)` + `set_context_subjective(agent_id, cap_i, belief)` | spec 02 |
| **Q4** global_step 传递 | **`model.update_step(step)` 独立方法** | spec 02 + spec 04 |

---

## 🚨 v4 关键约束（Ch4 v4 + Pkg-01/02/03 contract → 本包）

| # | 约束 | 来源 | 对本包的硬约束 |
|---|------|------|----------------|
| **C1** | hyper_trans 仅接 c_ctx | Ch4.3.2 + D1 | 改 role/belief 时 θ_state 不变（spec 01 单测） |
| **C2** | hyper_rew / hyper_pred 接 80 维 ctx_aug | Ch4.3.3 + D1 | DualHyperNetwork v2 接口签名 |
| **C3** | set_context 新签名 | Ch4.3 + D4 + **Q3** | 两步分离 API (spec 02) |
| **C4** | belief gradient gating 前 5K step | Ch4.6.5 + D3 + **Q4** | model.forward 内 step-conditional `.detach()` (spec 04) |
| **C5** | d_c=16, d_role=32, d_belief=32, d_ctx_aug=80 | Ch4.2.4 + Pkg-01 spec 05 | spec 01 init 时断言 ctx_aug==80 |
| **C6** | StateTransNet Δs 残差 | Ch4.6 + v4.7 | spec 03 防线 4 + v4.7 行号映射 |
| **C7** | AdaLN `h × (1 + γ) + β` | Ch4.6 + v4.7 | spec 03 防线 2 + (1+) factor 不能丢 |
| **C8** | output_scale (trans=0.01, rew=0.1, pred=0.01) | Ch4.6 + v4.7 | spec 03 防线 1 + 三初值表 |
| **C9** | type-aware via type_emb → hyper_rew | Ch4.3.3 + D2 | spec 05 全 spec 内容 |
| **C10** | BeliefNet + main loss 联合训练 | Ch4.5.5 + Pkg-05 | spec 02 接口预留 + spec 08 §4 |
| **C11** | role_i Self-Info 严格 | Ch3.7 + Ch4.2.2 + Pkg-03 spec 08 契约 1 | spec 02 set_context_subjective 不接 oracle types |
| **C12** | z_hat 顺序约定（agent_id 升序跳过 self） | Pkg-01 spec 04 + Pkg-03 spec 05 / 06 / 08 | spec 02 调用 BeliefNet API 时遵守 |
| **C13** | hyper_pred detach_pred_context 配置 | Ch4.6 + D5 | spec 01 接受 detach_pred_context bool 配置 |
| **C14** | StateTransNet joint action onehot N*A | Ch4.4 + D7 | spec 01 + functional_nets 接口不变 |

---

## 📦 输出清单（PR 时检查）

### 重写（4 文件）

```
hyper_mve/models/
├── hyper_network.py                # DualHyperNetwork v2（重写：双 forward API + 三路输入）
├── hyper_muzero_model.py           # 重写（v4 统一模型 + set_context 两步分离）
├── functional_nets.py              # 修改（保留 4 道防线 + obs_dim 配置化）
└── representation_net.py           # 修改（obs_dim 从配置取）
```

### 新增（1 核心 + 4 测试 + 1 脚本）

```
hyper_mve/models/
└── grad_gating.py                  # BeliefGradGating helper

tests/models/
├── test_hyper_network_v2.py        # 三路输入维度、forward_trans 单独性、output_scale
├── test_hyper_muzero_model.py      # set_context 双 API、缓存策略、type-aware
├── test_stability_safeguards.py    # 4 道防线数值不变性（AdaLN / L2 norm / Δs / small_init）
└── test_grad_gating.py             # 前 5K step belief 参数梯度 = 0；之后 > 0

hyper_mve/scripts/
└── test_hyper_model_forward.py     # 端到端 100 步 forward + 性能 profile
```

### 修改（3 个调用点同步迁移，Q2 用户确认 Pkg-04 范围）

```
hyper_mve/planning/mve_planner.py   # L71/213/219/258 四处 set_context → set_context_objective+subjective
hyper_mve/training/muzero_trainer.py  # set_context 调用点同步迁移
hyper_mve/training/worker.py        # set_context 调用点同步迁移
```

---

## 🔗 上下游关联

| 关系 | 包 | 接口 |
|------|-----|------|
| **本包 ← Pkg-01** | Foundation Schema | `ModelConfig.d_ctx_aug=80`, `TimeStepRecord`, `CapabilityVector.normalize()`, `AgentType`, `V4Config` |
| **本包 ← Pkg-02** | ResourceCommons env | `env.observation_space` (RepNet input dim) + `env.info` 三段分组（仅 trainer 消费 Oracle 字段） |
| **本包 ← Pkg-03** | TriContextEncoder + BeliefNet | `TriContextEncoder.forward / forward_c_ctx_only`, `BeliefNet.step / forward / get_head_opp_predictions`, `BeliefEncoder`, `belief_loss`, `build_oracle_z_seq` |
| **本包 → Pkg-05** | Trainer & Worker | model 7 个对外 API（spec 08 §1 表）+ 课程 Stage 接入 + global_step 传递（update_step） |
| **本包 → Pkg-06a/b** | Baselines | `shared_backbones` 工厂复用 RepNet + BeliefNet 实例（断言 B 公平性强制） |
| **本包 → Pkg-07** | Eval Protocols | Self-Info 评估时 set_context_subjective 不接 oracle types（spec 08 §6） |

---

## 📖 引用源

- 项目 Plan File: `Part 2 Pkg-04`（8 Decisions + 7 specs 原版规格）
- Plan 修订：`review-pkg-01-additive-patch-pkg-04-starry-quiche.md`（Q1-Q4 用户决议、8 specs 升级、避免事后修改 7 机制）
- Hyper-MuZero v4 Roadmap §4 Stage 1 Week 3
- Chapter 4 v4: [`docs/Chapter4_Architecture_v4.md`](../../docs/Chapter4_Architecture_v4.md) §4.3 + §4.4 + §4.6
- v4.7 现有代码（保留行号映射至 v4）：
  - `models/hyper_network.py` L63-82 (L2 norm + output_scale)
  - `models/functional_nets.py` L43-77 (AdaLN) + L245-271 (Δs 残差)
  - `planning/mve_planner.py` L71/213/219/258 (4 处 set_context 调用点)
- Pkg-01 SDD: [`sdd/pkg-01-foundation-schema/specs/05-v4-config-structure.md`](../pkg-01-foundation-schema/specs/05-v4-config-structure.md) ModelConfig 字段
- Pkg-02 SDD: [`sdd/pkg-02-resource-commons-env/specs/08-gym-api.md`](../pkg-02-resource-commons-env/specs/08-gym-api.md) env.info 三段分组
- Pkg-03 SDD: [`sdd/pkg-03-tricontext-beliefnet/specs/08-integration-contracts.md`](../pkg-03-tricontext-beliefnet/specs/08-integration-contracts.md) 与 Pkg-04 协作契约

---

## 📑 spec 间引用表（M6：避免事后修改）

### 期望引用矩阵（人工对账表）

| Spec | 引用的其他 spec |
|------|----------------|
| spec 01 | spec 03 (output_scale init), spec 06 (forward 张量 shape) |
| spec 02 | spec 01 (DualHyperNetwork 接口), spec 04 (grad gating 调用位置), spec 08 (API 稳定性表) |
| spec 03 | spec 01 (output_scale 三初值), spec 04 (防线 5 belief gating), v4.7 行号映射 |
| spec 04 | spec 02 (model.forward 内调用点), Pkg-03 spec 04/05/08 (BeliefNet 输出与 BeliefEncoder 分层) |
| spec 05 | spec 01 (hyper_rew 输入), spec 02 (set_context_subjective), Pkg-03 spec 03 (role_encoder type_emb) |
| spec 06 | spec 01/02 (数据流节点), spec 04 (.detach() 切断点) |
| spec 07 | spec 01 (forward 性能), Pkg-03 spec 04/05 (BeliefNet step / forward 性能) |
| spec 08 | spec 01/02/03/04/05/06/07 全引用 + v4.7 三处调用点行号 + Pkg-05/06/07 预留接口 |

### Day 6 机器可校验引用核对（澄清 3）

Day 6 任务必须产出**机器可校验**的引用矩阵，避免人工对账漏洞：

```powershell
# Step 1: 每个 spec 文件 grep 引用计数
cd D:\RL\hyper_mve\sdd\pkg-04-dualhypernet-v2-model

# 对每个 spec 0X 文件统计它引用了哪些其他 spec (0Y, Y != X)
foreach ($spec in (Get-ChildItem specs/0*.md)) {
    $name = $spec.BaseName
    $content = Get-Content $spec.FullName -Raw
    
    # 匹配 "spec 01" / "spec 02" / ... 引用
    $refs = [regex]::Matches($content, 'spec\s+0([1-8])') |
            ForEach-Object { $_.Groups[1].Value } |
            Sort-Object -Unique
    Write-Output "$name -> $($refs -join ',')"
}

# Step 2: 输出对照 README §6 期望表的差集（缺/多）
# 用人工或脚本对账, 缺的引用需补回相应 spec, 多的引用需 README §6 同步补
```

**产出**：`ref_matrix.csv`（spec → refs_actual / refs_expected / delta），Day 6 检查 delta 列全空。

**验收**：
1. 每个 spec 至少引用 README §6 表所列的其他 spec
2. 未在 README §6 表中列出的引用需要解释（要么补到表里，要么删除引用）
3. spec 08 引用 spec 01-07 全 7 个 — 这是契约层文件的特性

**实施建议**：把上述 PowerShell 脚本固化为 `sdd/pkg-04-*/scripts/check_ref_matrix.ps1`，作为 PR merge 前置检查。

---

## 🚦 Review Checklist（用户审阅时勾选）

- [ ] proposal.md 读完，**Why** 充分（v4.7 双路无法表达三路 80 维 ctx_aug + 无 type_emb → 无法支撑断言 A/B/C）
- [ ] design.md 读完，**8 项 Decisions** 中无反对意见（特别 D2 type-aware 仅通过 type_emb 路径、D5 detach_pred_context 默认 True）
- [ ] specs/01-08 抽查至少 3 个（重点 specs/02 + specs/08，对应 set_context API + 对外契约）
- [ ] 7 天实施顺序合理（Day 1 design.md 审阅是 hard gate）
- [ ] Q1-Q4 决议正确反映在 design.md §8 + 各 spec
- [ ] v4 关键差异认可：set_context 两步分离 API + global_step 通过 update_step 传递
- [ ] mve_planner.py 4 处调用点迁移到 Pkg-04 范围接受（用户已 ack Q2）
- [ ] 出包后可启动 Pkg-05 (Trainer)
