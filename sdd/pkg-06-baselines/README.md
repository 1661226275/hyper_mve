# Pkg-06: Baselines

> ⛔ **SUPERSEDED BY [`pkg-07-baselines`](../pkg-07-baselines/README.md)** (2026-06-19) — pkg-07 absorbs the 5 internal variants spec'd here AND adds external paradigm baselines (MAPPO/QMIX/MA-MuZero + MAMBA-if-sourced). The factory `create_baseline_model` is renamed to `create_baseline` in pkg-07 (alias preserved with `DeprecationWarning`). This directory is kept for history; do not implement against pkg-06 specs directly.

> **状态**：Draft — Day 1 三件套待用户审阅（design.md = HARD GATE）
> **包 ID**：`pkg-06-baselines`
> **工期**：1 周（5-7 天） · **GPU 算力**：~5 小时（5 variant 实例化 smoke + 单步 forward micro-benchmark） · **PR 体量**：1 工厂 + 1 共享后端 + 5 模型类 + 11 测试（纯新增，不改上游）

---

## 📚 文档导览

| 文档 | 目的 | 字数 |
|------|------|------|
| [`proposal.md`](./proposal.md) | **先读这个**：Why / What Changes / Capabilities / Impact | ~4000 |
| [`design.md`](./design.md) | 设计抉择：Context / Goals / 10 项 Decisions（含 Q1-Q4 + 修复 1-7 + 调整 1-5）| ~5500 |
| [`specs/01-baselines-factory-and-registry.md`](./specs/01-baselines-factory-and-registry.md) | `create_baseline_model(cfg, variant)` + 5 variant registry + supersede 声明 + CLI 映射表 | ~3000 |
| [`specs/02-shared-backbones.md`](./specs/02-shared-backbones.md) | `create_rep_net/belief_net/tri_context_encoder` + 等参共享契约 | ~2500 |
| [`specs/03-input-conditioned-baselines.md`](./specs/03-input-conditioned-baselines.md) | `input_wide` + `input_deep`（断言 B 等参对照）| ~2500 |
| [`specs/04-ma-muzero-baseline.md`](./specs/04-ma-muzero-baseline.md) | `ma_muzero` 共享 RewardHead + type 输入（断言 A）| ~2000 |
| [`specs/05-belief-type-ablation-baselines.md`](./specs/05-belief-type-ablation-baselines.md) | `no_belief`（断言 C/Abl7）+ `rewardhead_explicit_type`（断言 A/Abl6.x）| ~2500 |
| [`specs/06-model-7api-conformance.md`](./specs/06-model-7api-conformance.md) | 7-API 一致性 + set_context 拆分 + Self-Info 严格 + stateful + mermaid 门控图 | ~3000 |
| [`specs/07-param-fairness-and-lr-sweep.md`](./specs/07-param-fairness-and-lr-sweep.md) | 等参双阈值 + LR sweep 协议 + 单步 forward 性能护栏 | ~2500 |
| [`specs/08-integration-contracts.md`](./specs/08-integration-contracts.md) | **对外硬契约**：与 Pkg-05 工厂 + Pkg-07/08 契约 + 五条复用约束 | ~3000 |

---

## 🎯 一句话目标

**为论文 4 条断言提供对照基线模型**：在 `hyper_mve/baselines/` 下新增**一个统一工厂 `create_baseline_model(cfg, variant)`** + **一个共享后端 `shared_backbones.py`** + **5 个 baseline 模型类**（`input_wide` / `input_deep` 打断言 B，`ma_muzero` 打断言 A，`no_belief` 打断言 C/Abl7，`rewardhead_explicit_type` 打断言 A/Abl6.x），所有 baseline **复用 Pkg-05 同一 `MuZeroTrainer` + `Worker` + `EpisodeReplayBuffer` + `compose_total_loss`，仅 model 类不同**，并在 spec 层**量化保证等参公平性**（条件化消费子系统双阈值 5%/10%）与 belief 梯度门控一致性，为 Pkg-07 eval + Pkg-08 experiments 提供可消费的对照模型族。

---

## ✅ 关键 Acceptance Criteria（速览）

### 结构硬约束（must pass）

- [ ] **5 variant 均可经工厂实例化**（C6-FACT1）
- [ ] **工厂收 "hyper" 抛 ValueError**（C6-FACT2，hyper 走 HyperMuZeroModel 直接构造）
- [ ] **RepNet/BeliefNet/TriCtx 跨 variant 参数量逐位相同**（C6-FAIR2，公平性核心）
- [ ] **5 variant 实现完整 7-API（签名与 Pkg-04 spec02 一致）**（C6-API1）
- [ ] **stateful：predict 用最后一次 subjective agent_id（每 variant）**（C6-API2）
- [ ] **不泄漏 oracle types：assert cap_i.shape[-1]==4（每 variant）**（C6-SELF1）
- [ ] **同一 MuZeroTrainer/Worker/Buffer/compose_total_loss（仅 model 类不同）**（C6-REUSE1）
- [ ] **5 variant update_step 共用同一 BeliefGradGating 类 + 5K 阈值（每 variant 各自实例化），pre-5K BeliefNet 梯度=0**（C6-GRAD1）
- [ ] **input-conditioned baseline 不含 DualHyperNetwork**（C6-STRUCT1）
- [ ] **ma_muzero 共享单一 RewardHead（无 per-agent θ_rew）**（C6-STRUCT2）
- [ ] **工厂消费 4 个 baseline cfg 字段，无 hard-coded 默认值**（C6-CFG1）

### 等参/性能验证

- [ ] **条件化消费子系统参数 ≤5% warn / ≤10% fail**（C6-FAIR1，仅 input_wide/input_deep/no_belief vs hyper）
- [ ] **ma_muzero / rewardhead_explicit_type 结构性豁免对齐，但报告 LR sweep wall-clock 步数等价**
- [ ] **5 variant 单步 forward 在 spec 07 分档预算内**（性能护栏）

### 期望达到

- [ ] `pytest tests/baselines/` 全过（C6-* 11 测试，含每 variant 跑的 self-info / stateful / grad-gating）
- [ ] `pwsh sdd/pkg-06-baselines/scripts/check_ref_matrix.ps1` 输出 `[PASS]`（delta 全空）
- [ ] `git diff` 确认 Pkg-01..05 SDD + `hyper_mve/**` 代码零改动

---

## 📅 实施顺序（7 天）

| 阶段 | 天数 | 任务 | 输出 / 审阅点 |
|------|------|------|---------------|
| **Phase 1** | Day 1 | README + proposal + design.md（10 Decisions 锁定）| **design.md 审阅 = HARD GATE**（7 项验收清单，全过才放行）|
| **Phase 2** | Day 2 | spec 01（工厂&registry）+ spec 02（shared_backbones）| 工厂签名对账 Pkg-05 §3.1 + supersede 声明 |
| **Phase 3** | Day 3 | spec 03（input wide/deep）+ spec 04（ma_muzero）| 断言 B 等参对照 + 断言 A 共享头 |
| **Phase 4** | Day 4 | spec 05（belief/type ablation）+ spec 06（7-API 一致性）| Self-Info 严格 + stateful + mermaid 门控图 |
| **Phase 5** | Day 5 | spec 07（等参预算&LR sweep）+ spec 08（集成契约）| 双阈值单测 + 五条复用约束 |
| **Phase 6** | Day 6 | `scripts/check_ref_matrix.ps1`（纯 ASCII）+ `ref_matrix.csv` | 真跑出 `[PASS]` + M3 映射表自查 |
| **Phase 7** | Day 7 | PR_DESCRIPTION.md + 用户最终 ack | Pkg-06 SDD finalized |

### Day 1 HARD GATE 验收清单（7 项，全过才放行）

| # | 验收项 | design.md 落点 |
|---|--------|---------------|
| 1 | 5 variant 各自核心区别表（含 ma_muzero vs rewardhead_explicit_type 撕分）| §4 D7 + proposal §2.1.3 |
| 2 | 等参对照单元定义（条件化消费子系统 + 逐 variant 账目 + 豁免规则）| §4 D5 |
| 3 | mermaid belief grad-gating 双路径图框架（pre-/post-5K，覆盖 5 variant）| §4 D6 |
| 4 | "7 训练入口 vs 5 工厂 variant" framing 澄清 + 7×3 三角矩阵 + CLI 映射表 | §3.1/§3.2/§3.3 |
| 5 | cfg 新字段穷举（4 个待声明字段，挂"消费态、不改上游"标签）| §4 D10 |
| 6 | ref_matrix 预期表（8×expected）| §8.2 |
| 7 | 7 天日历 + 响应式 SLA | §6 |

### 响应式 SLA（调整 5）

日历是**名义节奏，非硬截止**。**Day 1 hard gate 不通过 → 全流程顺延 1 天**（不带病进 Day 2）。每个 spec 当天末交审，等用户 ack 再进下一个，审阅往返时间不计入"天"。

---

## 🔑 10 项关键 Decisions（速览，含 Q1-Q4 用户决议）

| # | Decision | 推荐 | 详见 |
|---|----------|------|------|
| D1 | spec 数量与文档结构 | **8 specs 对称**（与 Pkg-03/04/05 对称）| [design §4 D1](./design.md) |
| D2 | 工厂归属与签名 | **`baselines/__init__.py` 统一 `create_baseline_model(cfg, variant)`**（上游已锁）| [design §4 D2](./design.md) |
| D3 | 工厂收 "hyper" 行为 | **raise ValueError**（hyper 走 HyperMuZeroModel 直接构造）| [design §4 D3](./design.md) |
| D4 | 共享后端粒度 | **RepNet/BeliefNet/TriCtx 同类同构、独立实例、独立梯度**| [design §4 D4](./design.md) |
| D5 | 等参对照单元定义 | **条件化消费子系统 + 双阈值 5%/10% + 逐 variant 账目/豁免**| [design §4 D5](./design.md) |
| D6 | belief 门控一致性 | **5 variant 共用同一 BeliefGradGating**（断言 B 公平核心）| [design §4 D6](./design.md) |
| D7 | ma_muzero vs explicit_type 边界 | **撕分不同失败模式**（平均梯度 vs 离散分支无法吸收连续 cap）| [design §4 D7](./design.md) |
| D8 | stateful 契约 | **缓存 (agent_id,cap,belief) + predict 用最后一次 subjective**| [design §4 D8](./design.md) |
| D9 | Self-Info 严格 | **每 variant assert cap_i.shape[-1]==4 + belief 来自 BeliefNet**| [design §4 D9](./design.md) |
| D10 | cfg 新字段归属 | **cfg.model.* + Pkg-01 spec05 同步（消费态、不改上游）**| [design §4 D10](./design.md) |

**用户已 ack 的 4 项关键决策（在 design.md §9 锁定）**：

| Q | 决议 | 影响位置 |
|---|------|---------|
| **Q1** spec 数量 | **8 specs**（与 Pkg-03/04/05 对称）| 本文档导览 + 8 specs |
| **Q2** variant 清单 | **5 个进工厂**；hyper/oracle_only/infer_only 不进工厂；MAPPO/QMix 出范围 | spec 01 registry |
| **Q3** 工厂归属 & 共享粒度 | **工厂在 baselines/__init__.py；RepNet/BeliefNet/TriCtx 同类同构、独立实例、独立梯度** | spec 01 / 02 |
| **Q4** 等参验证口径 | **仅条件化消费子系统 + 双阈值(5%/10%) + 具名单测** | spec 07 |

---

## 🚨 v4 关键约束（Ch4/Ch5 + Pkg-03/04/05 contract → 本包）

> 统一前缀：`FACT`(工厂) / `FAIR`(等参) / `SELF`(Self-Info) / `API` / `REUSE`(复用) / `GRAD`(梯度门控) / `STRUCT`(结构) / `CFG`(配置消费) / `PERF`(性能护栏)。

| # | 约束 | 来源 | 对本包的硬约束 |
|---|------|------|----------------|
| **C6-FACT1** | 5 variant 均可实例化 | Pkg-05 spec08 §3.1 | spec 01 单测 `test_create_baseline_model_all_5_variants` |
| **C6-FACT2** | 工厂收 "hyper" 抛 ValueError | design D3 | spec 01 单测 `test_factory_rejects_hyper_variant` |
| **C6-CFG1** | 工厂消费 4 个 baseline cfg 字段，无 hard-coded 默认值 | design D10 | spec 01 单测 `test_factory_reads_baseline_cfg_fields` |
| **C6-FAIR1** | 条件化消费子系统 ≤5% warn / ≤10% fail | Ch5 §5.6 step1 | spec 07 单测 `test_baseline_param_count_within_5pct` |
| **C6-FAIR2** | RepNet/BeliefNet/TriCtx 跨 variant 参数量逐位相同 | Pkg-03 spec08 §6 | spec 02 单测 `test_shared_backbone_identical_param_count` |
| **C6-SELF1** | 不泄漏 oracle types（每 variant）| C11/D6 Self-Info | spec 06 单测 `test_baseline_set_context_subjective_no_oracle_types_leak` |
| **C6-API1** | 7 方法签名一致 | Pkg-04 spec02 | spec 06 单测 `test_baseline_implements_full_7api` |
| **C6-API2** | stateful 用最后一次 subjective agent_id（每 variant）| Pkg-04 spec02 L308-310 | spec 06 单测 `test_baseline_predict_uses_last_subjective_agent_id` |
| **C6-REUSE1** | 同一 MuZeroTrainer/Worker/Buffer/compose_total_loss | Pkg-05 spec08 §3.2 | spec 08 单测 `test_baseline_reuses_same_trainer_class` |
| **C6-GRAD1** | pre-5K BeliefNet 梯度=0（每 variant）| Pkg-04 spec02 §3.4 + spec04 | spec 06 单测 `test_baseline_update_step_gates_belief_grad` |
| **C6-STRUCT1** | input-conditioned 不含 DualHyperNetwork（结构约束）| design D5/D6 | spec 03 单测 `test_input_baseline_no_hypernet` |
| **C6-STRUCT2** | ma_muzero 无 per-agent θ_rew（共享单一头，结构约束）| design D7 | spec 04 单测 `test_ma_muzero_shared_rewardhead` |
| **C6-PERF1** | input baseline 单步 forward ≤ 2.0× hyper（性能护栏）| design G9 | spec 07 单测 `test_baseline_forward_budget` |

---

## 📦 输出清单（PR 时检查）

### 新增（1 工厂 + 1 共享后端 + 5 模型类，全部在 `hyper_mve/baselines/`）

```
hyper_mve/baselines/
├── __init__.py                       # create_baseline_model(cfg, variant) + registry + reject hyper
├── shared_backbones.py               # create_rep_net / create_belief_net / create_tri_context_encoder
├── input_conditioned.py              # InputWideBaselineModel + InputDeepBaselineModel（断言 B）
├── ma_muzero.py                      # MAMuZeroBaselineModel（断言 A）
├── no_belief.py                      # NoBeliefBaselineModel（断言 C / Abl7）
└── explicit_type_reward.py           # ExplicitTypeRewardBaselineModel（断言 A / Abl6.x）

tests/baselines/
├── test_factory.py                   # C6-FACT1/FACT2/CFG1
├── test_shared_backbones.py          # C6-FAIR2
├── test_param_fairness.py            # C6-FAIR1
├── test_7api_conformance.py          # C6-API1/API2
├── test_self_info.py                 # C6-SELF1
├── test_reuse.py                     # C6-REUSE1
└── test_grad_gating.py               # C6-GRAD1 + C6-STRUCT1/STRUCT2
```

### 修改（无 — 本包纯新增）

```
（无）— 本包不修改 hyper_mve/models/* 或 hyper_mve/training/* 或 hyper_mve/schemas/* 任何文件
待声明 cfg 字段（4 个）由 Pkg-01 spec05 同步声明（消费态、不改上游）
```

---

## 🔗 上下游关联

| 关系 | 包 | 接口 |
|------|-----|------|
| **本包 ← Pkg-03** | TriContextEncoder + BeliefNet | RepNet/BeliefNet/TriCtx 共享契约（spec08 §6）|
| **本包 ← Pkg-04** | DualHyperNetwork v2 & Model | `HyperMuZeroModel` 7-API + stateful（spec02 L308-310）+ `BeliefGradGating`（spec02 §3.4 + spec04）|
| **本包 ← Pkg-05** | Trainer & Worker | `create_baseline_model` 签名（spec08 §3.1 line 136）+ 五条复用约束（§3.2）+ --variant CLI（§6.2）|
| **本包 → Pkg-07** | Eval Protocols | baseline 与 hyper 走同一 evaluator（7-API 一致 + 同 trainer）|
| **本包 → Pkg-08** | Experiments | `create_baseline_model(cfg, variant)` 可调 → train_main.py `--variant baseline_*`；Abl 6.x/7 复用 `rewardhead_explicit_type` / `no_belief` |

### 📌 下游启动条件速查

| 下游包 | 启动条件 | 验收 spec |
|--------|---------|-----------|
| **Pkg-07 (Eval)** | (a) 5 variant 7-API finalized（spec 06）<br>(b) 工厂签名稳定（spec 01）| spec 01 + spec 06 |
| **Pkg-08 (Experiments)** | (a) spec 08 五条复用约束 finalized<br>(b) CLI↔工厂映射表 finalized（spec 01）<br>(c) `rewardhead_explicit_type` / `no_belief` 模型类可调（spec 05）| spec 01 + spec 05 + spec 08 |

---

## 📖 引用源

- 项目 Plan File: `pkg-06-baselines-sdd-groovy-cupcake.md`（Pkg-06 SDD 撰写计划 + Q1-Q4 决议 + 修复 1-7 + 调整 1-5）
- Chapter 4_1 Motivation: 4 条断言（A 类型梯度撕裂 / B belief 专用容量 / C 三路必要性 / D planner 双技术）
- Chapter 5 v4: [`docs/Chapter5_Methodology_v4.md`](../../docs/Chapter5_Methodology_v4.md) §5.6（等参公平协议 step1/2/3）
- Pkg-03 SDD: spec 08 §6（RepNet/BeliefNet/TriContextEncoder 共享契约）
- Pkg-04 SDD: spec 02（7-API + stateful L308-310）+ spec 02 §3.4 / spec 04（BeliefGradGating 双层 detach）+ spec 08 §5.1（旧草案，被本包 supersede）
- Pkg-05 SDD: spec 08 §3.1（工厂签名 line 136）+ §3.2（五条复用约束）+ §6.2（--variant CLI 语义）

---

## 📑 spec 间引用表（M6：避免事后修改）

### 期望引用矩阵（人工对账表）

| Spec | 引用的其他 spec |
|------|----------------|
| spec 01 | spec 02 (shared_backbones 工厂), spec 03 (input variant 归属), spec 04 (ma_muzero 归属), spec 05 (no_belief/explicit_type 归属), spec 06 (7-API), spec 08 (集成契约) |
| spec 02 | spec 03 (input baseline 消费 backbone), spec 04 (ma_muzero), spec 05 (no_belief/explicit_type), spec 06 (服务 7-API contract), spec 08 |
| spec 03 | spec 02 (backbone), spec 06 (7-API), spec 07 (等参), spec 08 |
| spec 04 | spec 02 (backbone), spec 06 (7-API), spec 07 (LR sweep 豁免), spec 08 |
| spec 05 | spec 02 (backbone), spec 06 (7-API), spec 08 |
| spec 06 | spec 02 (backbone 接入), spec 03/04/05 (每 variant stateful/Self-Info/update_step 落地), spec 08 (契约) |
| spec 07 | spec 03 (input baseline 等参), spec 04 (ma_muzero 豁免), spec 05 (no_belief/explicit_type 等参账目), spec 08 |
| spec 08 | spec 01-07 全引用 + Pkg-05 spec08 契约模式 + Pkg-07/08 下游接入伪签名 |

### Day 6 机器可校验引用核对

Day 6 必须产出 `ref_matrix.csv`：

```powershell
cd D:\RL\hyper_mve\sdd\pkg-06-baselines
pwsh scripts\check_ref_matrix.ps1        # ASCII-only, 也可在 Windows PowerShell 5.1 下运行
# → 生成 ref_matrix.csv + 终端 [PASS]/[FAIL]
```

引用扫描正则匹配两种 **包内** 引用形式，并排除 **跨包** 引用：

```powershell
$pat = '(?<!Pkg-\d{2} )spec\s+0([1-8])|\b0([1-8])-[a-z]'
#   - 裸 "spec 0X"     但排除 "Pkg-NN spec 0X"（负向后顾 (?<!Pkg-\d{2} )）
#   - 文件名 "0X-name"（spec 间在上下游表里以文件名互引）
```

**验收**：每个 spec 至少引用 README §spec 间引用表所列的其他 spec（delta 列全空）；spec 08 引用 spec 01-07 全 7 个（契约层文件特性）。Day 6 真跑校验，不写"理想态"csv。

---

## 🚦 Review Checklist（用户审阅时勾选）

- [ ] proposal.md 读完，**Why** 充分（4 断言需对照 / 等参是断言 B 生死线 / 复用训练设施是公平另一半 / Pkg-04 §5.1 旧草案需 supersede）
- [ ] design.md 读完，**Day 1 HARD GATE 7 项验收清单**全过（特别 §3 framing 澄清 + §4 D5 等参单元 + §4 D6 门控一致性 + §4 D7 撕分）
- [ ] **5 variant 各自核心区别**清晰（input_wide/deep 加宽加深 vs ma_muzero 共享头 vs no_belief 置零 vs explicit_type 离散分支）
- [ ] **等参对照单元**口径认可（仅条件化消费子系统 + 双阈值 + 豁免 ma_muzero/explicit_type）
- [ ] **belief 门控一致性**认可（5 variant 共用同一 gating，否则断言 B 不公平）
- [ ] 7 天实施顺序合理（Day 1 design.md 审阅是 hard gate，未过顺延 1 天）
- [ ] Q1-Q4 + 修复 1-7 + 调整 1-5 正确反映在 design.md §9
- [ ] 出包后可启动 Pkg-07 (Eval) + Pkg-08 (Experiments)
