# Pkg-05: Trainer & Worker

> **状态**：Draft — awaiting user review
> **包 ID**：`pkg-05-trainer-and-worker`
> **工期**：1 周（5-7 天） · **GPU 算力**：20 小时（单 train_step + collect_episode 多场景 smoke） · **PR 体量**：4 重写 + 2 新增 + 11 处调用点迁移 + 6 测试

---

## 📚 文档导览

| 文档 | 目的 | 字数 |
|------|------|------|
| [`proposal.md`](./proposal.md) | **先读这个**：Why / What Changes / Capabilities / Impact | ~4500 |
| [`design.md`](./design.md) | 设计抉择：Context / Goals / 10 项 Decisions（含 Q1-Q4 决议） | ~5500 |
| [`specs/01-trainer-loop-v2.md`](./specs/01-trainer-loop-v2.md) | MuZeroTrainer + 单一 train_step + TrainConfig 27 字段穷举 | ~3000 |
| [`specs/02-worker-collection.md`](./specs/02-worker-collection.md) | Worker + 5 处 set_context 迁移 + BeliefNet.step 接入 | ~2500 |
| [`specs/03-episode-buffer-v2.md`](./specs/03-episode-buffer-v2.md) | EpisodeReplayBuffer 重写 + stratified sampling | ~2500 |
| [`specs/04-curriculum-scheduler.md`](./specs/04-curriculum-scheduler.md) | 3 Stage 课程 + λ_b 曲线 + oracle_z_mixing_weight | ~2000 |
| [`specs/05-loss-composition.md`](./specs/05-loss-composition.md) | main + L_belief 加权 + 双路径 backward 图 | ~2500 |
| [`specs/06-mve-planner-v4.md`](./specs/06-mve-planner-v4.md) | CRN 4 phase 保留 + 4 处 set_context 迁移 | ~2000 |
| [`specs/07-ema-and-scheduler.md`](./specs/07-ema-and-scheduler.md) | EMA tau=0.99 + warmup_cosine LR | ~1500 |
| [`specs/08-integration-contracts.md`](./specs/08-integration-contracts.md) | **对外硬契约**：4 类 API 稳定性 + Pkg-06/07/08 启动条件 + train_main.py 入口 | ~3000 |

---

## 🎯 一句话目标

**重写 v4.7 muzero_trainer.py 为 v4 单一 train_step**（废弃 train_step_infer，BeliefNet 取代 GRU 推理），**重写 worker.py 接 BeliefNet.step 在线推断 + 写 v4 TimeStepRecord**，**重写 episode_buffer.py 为 TimeStepRecord 容器**（保留 v4.7 stratified sampling），**新增 curriculum.py + loss_composition.py 模块化课程与 loss 拼装**，**最小化迁移 mve_planner.py 4 处 set_context 调用点**（保留 CRN 4 phase），**统一入口 `scripts/train_main.py`** + cfg 选项控制变体，为 Pkg-06 baselines + Pkg-07 eval + Pkg-08 experiments 提供完整可消费的训练系统。

---

## ✅ 关键 Acceptance Criteria（速览）

### 结构硬约束（must pass）

- [ ] **trainer 每 train_step 起点调 model.update_step(step) 1 次**（C5-T1）
- [ ] **K-step unroll 内 set_context_objective 仅调 1 次**（C5-T2，复用 θ_state）
- [ ] **N agents 循环 set_context_subjective**（C5-T3）
- [ ] **worker 不调 update_step**（C5-W1，worker no_grad）
- [ ] **worker 用 BeliefNet.step 在线推断**（C5-W2，取代 v4.7 set_context_from_history）
- [ ] **TimeStepRecord z_hat 顺序按 Pkg-01 spec 04**（agent_id 升序跳过 self，C5-W3 / C5-B1）
- [ ] **课程 3 stage 切换边界 0.3 / 0.7**（C5-S1，可配置）
- [ ] **Stage 1 100% oracle 注入**（C5-S2）
- [ ] **Stage 2 oracle mixing 单调下降**（C5-S3）
- [ ] **λ_b(step) 曲线与 cfg.train.curriculum_stage_*_end_frac 一致**（C5-L1）
- [ ] **L_belief 路径与 main loss 分离反向**（C5-L2，main loss 受 Pkg-04 grad_gating 控制，L_belief 始终不受影响）
- [ ] **mve_planner.py 4 处 set_context 调用点全部迁移**（C5-P2）
- [ ] **EMA tau=0.99 衰减率正确**（C5-E1）
- [ ] **warmup_cosine LR 曲线**（C5-E2）
- [ ] **v4.7 → v4 调用点 grep 验证 0 残留**（C5-I1，沿用 Pkg-04 spec 08 §3.4 模式）

### 收敛/性能验证

- [ ] **单 train_step < 400 ms** (V100, Medium config B=256)（R5-1，硬 fail 阈值；review 修订 3 放宽，分摊表见下）
- [ ] **sample_batch < 50 ms** (B=256, K=5)（R5-2）
- [ ] **collect_episode < 5 s** (T=200, N=4)（R5-3）
- [ ] **Medium config 1K train_steps 无 NaN，loss 单调下降**

### 期望达到

- [ ] `pytest tests/training/test_trainer_loop.py tests/training/test_worker.py tests/training/test_episode_buffer.py tests/training/test_curriculum.py tests/training/test_loss_composition.py tests/training/test_ema_scheduler.py tests/planning/test_mve_planner.py` 全过
- [ ] `python scripts/train_main.py --preset medium --variant hyper --max_steps 1000` 端到端无 NaN
- [ ] `mypy --strict hyper_mve/training/{muzero_trainer,worker,episode_buffer,curriculum,loss_composition}.py` 零错误

---

## 📅 实施顺序（7 天）

| 阶段 | 天数 | 任务 | 输出 / 审阅点 |
|------|------|------|---------------|
| **Phase 1** | Day 1 | README + proposal + design.md（10 Decisions 锁定）| **design.md 审阅**（最关键，决定后续 7 specs 走向）|
| **Phase 2** | Day 2 | spec 01 + spec 02（trainer loop + worker collection）| 与 Pkg-04 spec 08 调用契约对账 |
| **Phase 3** | Day 3 | spec 03 + spec 04（episode buffer + curriculum scheduler）| TimeStepRecord 一致性 + 课程 3 stage 边界 |
| **Phase 4** | Day 4 | spec 05 + spec 07（loss composition + EMA/scheduler）| λ_b 课程曲线 + 双路径 backward 图 |
| **Phase 5** | Day 5 | spec 06 + spec 08（MVE planner + integration contracts）| **Pkg-06/07/08 SDD 启动条件**（spec 08 §3-§5 完整）|
| **Phase 6** | Day 6 | 全 spec 机器可校验交叉引用核对 + 19 项硬约束单测命名表对账 | 整包审阅，产出 ref_matrix.csv |
| **Phase 7** | Day 7 | PR description + 用户最终 ack | Pkg-05 SDD finalized |

---

## 🔑 10 项关键 Decisions（速览，含 Q1-Q4 用户决议）

| # | Decision | 推荐 | 详见 |
|---|----------|------|------|
| D1 | train_step 双套合并 | **合并为单一 train_step**（v4 BeliefNet 取代 Infer GRU 推理）| [design §3 D1](./design.md) |
| D2 | EpisodeReplayBuffer 归属 | **Pkg-05 范围**（Pkg-01 仅保留 TimeStepRecord schema）| [design §3 D2](./design.md) |
| D3 | 训练脚本架构 | **单一 train_main.py + cfg 选项**（变体由 curriculum / shared_backbones 控制）| [design §3 D3](./design.md) |
| D4 | curriculum 模块独立 | **独立 `CurriculumScheduler` 类**（lightweight，仅 step → stage 映射）| [design §3 D4](./design.md) |
| D5 | loss 组合归属 | **独立 `loss_composition.py`**（trainer 调，不在 trainer 内联）| [design §3 D5](./design.md) |
| D6 | EMA + scheduler 内联 vs 抽出 | **内联 MuZeroTrainer**（与 v4.7 一致，无必要抽出）| [design §3 D6](./design.md) |
| D7 | MVE planner cap/belief 传参 | **planner 接受 `cap` / `belief` dict 显式参数**（trainer/worker 准备）| [design §3 D7](./design.md) |
| D8 | checkpoint 兼容性 | **v4 字段，向 v4.7 不兼容**（明确 breaking + `_legacy_v4_7/` shim）| [design §3 D8](./design.md) |
| D9 | stratified sampling 实现 | **Pkg-05 spec 03 内实现**（不抽出独立 sampler 类）| [design §3 D9](./design.md) |
| D10 | v4.7 → v4 替换策略 | **inplace 修改**（D8 Pkg-04 同模式）+ git tag v4.7-final + _legacy_v4_7 归档 | [design §3 D10](./design.md) |

**用户已 ack 的 4 项关键决策（2026-05-29，在 design.md §8 锁定）**：

| Q | 决议 | 影响位置 |
|---|------|---------|
| **Q1** spec 数量 | **8 specs**（与 Pkg-03/04 对称） | 本文档导览 + 8 specs |
| **Q2** train_step 合并 | **合并为单一 train_step**（删除 train_step_infer） | spec 01 + D1 |
| **Q3** EpisodeReplayBuffer 归属 | **Pkg-05 范围**（Pkg-01 仅 TimeStepRecord schema） | spec 03 + D2 |
| **Q4** 训练脚本架构 | **单一 `train_main.py` + cfg 选项控制变体** | spec 08 §6 + D3 |

---

## 🚨 v4 关键约束（Ch5 v4 + Pkg-01/02/03/04 contract → 本包）

| # | 约束 | 来源 | 对本包的硬约束 |
|---|------|------|----------------|
| **C5-T1** | trainer 每 train_step 调 model.update_step | Pkg-04 spec 02 + Q4 | spec 01 单测 `test_trainer_calls_update_step_per_step` |
| **C5-T2** | K-step unroll 内 set_context_objective 一次 | Pkg-04 spec 02 §2.3 | spec 01 单测 `test_objective_called_once_per_unroll` |
| **C5-T3** | N agents 循环 set_context_subjective | Pkg-04 spec 02 §2.3 | spec 01 单测 `test_subjective_called_per_agent` |
| **C5-W1** | worker 不调 update_step | Pkg-04 spec 04 §3.4 | spec 02 单测 `test_worker_no_update_step` |
| **C5-W2** | worker 用 BeliefNet.step | Pkg-04 spec 08 §3.3 + Pkg-03 spec 04 | spec 02 单测 `test_worker_uses_belief_net_step` |
| **C5-W3** | TimeStepRecord 字段填充顺序 | Pkg-01 spec 04 z_hat 约定 | spec 02 单测 `test_z_hat_order_matches_pkg01_spec04` |
| **C5-B1** | buffer z_hat 顺序一致性（端到端）| Pkg-01 spec 04 + Pkg-03 spec 08 §7 | spec 03 单测 `test_buffer_z_hat_order_e2e` |
| **C5-B2** | stratified sampling 比例 | Pkg-01 spec 05 TrainConfig | spec 03 单测 `test_stratified_min_per_type_frac` |
| **C5-S1** | 课程 3 stage 切换边界 | Pkg-01 spec 05 + Ch5.7 | spec 04 单测 `test_curriculum_stage_boundaries` |
| **C5-S2** | Stage 1 100% oracle 注入 | Pkg-03 spec 08 §4.1 | spec 04 单测 `test_stage_1_full_oracle` |
| **C5-S3** | Stage 2 anneal 单调下降 | Ch5.7 | spec 04 单测 `test_oracle_mixing_anneal_monotonic` |
| **C5-L1** | λ_b(step) 课程加权曲线 | Pkg-01 spec 05 + Pkg-03 spec 08 §4.2 | spec 05 单测 `test_lambda_b_curve_matches_cfg` |
| **C5-L2** | L_belief 与 main loss 分离 | Pkg-04 spec 04 §3.1 双路径 | spec 05 单测 `test_belief_gradient_isolation_pre_5k` |
| **C5-P1** | CRN seed 相同 step 0 输出一致 | v4.7 DESIGN_DOC §4.1 + §5.8 | spec 06 单测 `test_crn_step0_deterministic_same_seed` |
| **C5-P2** | 4 处 set_context 调用点全部迁移 | Pkg-04 spec 08 §3.1 | spec 06 单测 `test_planner_4_set_context_migrated` |
| **C5-E1** | EMA tau=0.99 衰减率正确 | v4.4 + Pkg-01 spec 05 | spec 07 单测 `test_ema_decay_correctness_tau_099` |
| **C5-E2** | warmup_cosine LR 曲线 | v4.7 + Pkg-01 spec 05 | spec 07 单测 `test_lr_warmup_5k_then_cosine_anneal` |
| **C5-I1** | v4.7 → v4 调用点 grep 全消失 | Pkg-04 spec 08 §3.4 | spec 08 沿用 `test_no_legacy_set_context_calls` |
| **R5-1** | train_step < 400 ms（含分摊表）| Pkg-04 spec 07 档位 3 (100 ms) + target forward (50) + belief forward (30) + backward (180) + optim (20) + buffer (20) | spec 01 `test_train_step_under_400ms` + 3 个 micro-benchmark 单测 |
| **R5-2** | sample_batch < 50 ms | M7 性能护栏 | spec 03 `test_sample_batch_under_50ms` |

---

## 📦 输出清单（PR 时检查）

### 重写（4 文件）

```
hyper_mve/training/
├── muzero_trainer.py       # 重写（单一 train_step + EMA + scheduler 内联）
├── worker.py               # 重写（BeliefNet.step 接入 + TimeStepRecord 输出）
└── episode_buffer.py       # 重写（TimeStepRecord 容器 + stratified sampling）

hyper_mve/planning/
└── mve_planner.py          # 修改（4 处 set_context 调用点迁移 + cap/belief 传参）
```

### 新增（2 核心 + 6 测试 + 1 入口）

```
hyper_mve/training/
├── curriculum.py           # CurriculumScheduler（lightweight，step → stage）
└── loss_composition.py     # compose_total_loss + 双路径 backward 拼装

tests/training/
├── test_trainer_loop.py    # C5-T1/T2/T3 + R5-1
├── test_worker.py          # C5-W1/W2/W3
├── test_episode_buffer.py  # C5-B1/B2 + R5-2
├── test_curriculum.py      # C5-S1/S2/S3
├── test_loss_composition.py # C5-L1/L2
└── test_ema_scheduler.py   # C5-E1/E2

tests/planning/
└── test_mve_planner.py     # C5-P1/P2

hyper_mve/scripts/
└── train_main.py           # 单一统一入口（Q4）+ --preset/--variant 选项
```

### 修改（无 — Pkg-04 已迁移 model 部分）

```
（无）— 本包不修改 hyper_mve/models/* 或 hyper_mve/schemas/* 任何文件
```

---

## 🔗 上下游关联

| 关系 | 包 | 接口 |
|------|-----|------|
| **本包 ← Pkg-01** | Foundation Schema | `TimeStepRecord`（12 字段 + z_hat 顺序）, `V4Config`, `TrainConfig`（27 字段）|
| **本包 ← Pkg-02** | ResourceCommons env | `env.reset / step / info`（c_true / types / caps 写入 TimeStepRecord）|
| **本包 ← Pkg-03** | TriContextEncoder + BeliefNet | `belief_loss`, `build_oracle_z_seq`, `BeliefNet.step / forward` |
| **本包 ← Pkg-04** | DualHyperNetwork v2 & Model | `HyperMuZeroModel` 7 API（update_step / set_context_* / encode / transition / predict_reward / predict）|
| **本包 → Pkg-06a/b** | Baselines | `MuZeroTrainer` + `Worker` 复用 + `shared_backbones` 工厂签名 |
| **本包 → Pkg-07** | Eval Protocols | trainer 暴露 model / buffer / scheduler + `CurriculumScheduler` 评估期复用 |
| **本包 → Pkg-08** | Experiments | `scripts/train_main.py --ablation` flag 列表 + checkpoint 格式 |

### 📌 下游启动条件速查（review 澄清 3）

| 下游包 | 启动条件 | 验收 spec |
|--------|---------|-----------|
| **Pkg-06 (Baselines)** | (a) spec 08 §3 `shared_backbones` 工厂签名 finalized<br>(b) `Worker` / `EpisodeReplayBuffer` 接口稳定（spec 02 / spec 03）<br>(c) baseline 共享 `compose_total_loss`（spec 05）签名稳定 | spec 02 + spec 03 + spec 05 + spec 08 §3 |
| **Pkg-07 (Eval)** | (a) spec 08 §4 evaluator 接入接口 finalized<br>(b) `trainer.model` / `trainer.buffer` / `trainer.scheduler` 暴露（spec 01 公开属性）<br>(c) `CurriculumScheduler.stage(step)` 评估期可重入查询（spec 04） | spec 01 + spec 04 + spec 08 §4 |
| **Pkg-08 (Experiments)** | (a) spec 08 §5 `--ablation` flag 列表 finalized<br>(b) checkpoint 格式 finalized（spec 01 save/load + spec 08 §7）<br>(c) Pkg-06 + Pkg-07 都已 ack（实验需要 baseline + eval 一起）| spec 01 + spec 08 §5 §7 + Pkg-06/07 ack |

**速查表用途**：
- 撰写 spec 时优先满足这些条件（spec 08 §3-§5 必须列出 finalized 伪签名）
- Pkg-06/07/08 SDD 起手前对照本表，缺一不启动
- 避免 Pkg-05 完成后下游包"不知道能不能开始"

---

## 📖 引用源

- 项目 Plan File: `review-pkg-01-additive-patch-pkg-04-starry-quiche.md` Part D（Pkg-05 SDD 撰写计划 + Q1-Q4 决议 + 19 硬约束）
- Hyper-MuZero v4 Roadmap §5 Stage 1 Week 4
- Chapter 5 v4: [`docs/Chapter5_Methodology_v4.md`](../../docs/Chapter5_Methodology_v4.md) §5.6 + §5.7 + §5.8
- v4.7 现有代码（重写参考行号）：
  - `training/muzero_trainer.py` L56-83/138-181/203-430/432-598/85-130
  - `training/worker.py` L84-180 + L125/128/143/145/149
  - `training/episode_buffer.py` L72-215（EpisodeData + buffer）
  - `planning/mve_planner.py` L80-98/128-290 + L71/213/219/258
- Pkg-01 SDD: spec 04 (TimeStepRecord) + spec 05 (TrainConfig 27 字段)
- Pkg-03 SDD: spec 06 (belief_loss) + spec 08 (课程 3 stage 接入 + λ_b 系数)
- Pkg-04 SDD: spec 02 (model 7 API + §2.3 调用模板) + spec 04 (grad gating 双路径) + spec 08 (trainer 6 + worker 5 + planner 4 迁移指引)

---

## 📑 spec 间引用表（M6：避免事后修改）

### 期望引用矩阵（人工对账表）

| Spec | 引用的其他 spec |
|------|----------------|
| spec 01 | spec 04 (CurriculumScheduler.lambda_b 用于 loss 加权), spec 05 (compose_total_loss), spec 06 (mve_planner 调用), spec 07 (EMA + scheduler), spec 08 (API 稳定性), Pkg-04 spec 02 (7 API) |
| spec 02 | spec 03 (TimeStepRecord 写入 buffer), spec 06 (mve_planner 调用), Pkg-04 spec 02 §2.3 (worker 调用模板), Pkg-03 spec 04 (BeliefNet.step) |
| spec 03 | Pkg-01 spec 04 (TimeStepRecord 字段定义), spec 02 (worker 输出), spec 08 (与 Pkg-06 baselines 复用) |
| spec 04 | spec 01 (trainer 调用), spec 05 (loss 加权), Pkg-03 spec 08 §4 (oracle 注入路径) |
| spec 05 | spec 04 (CurriculumScheduler), Pkg-03 spec 06 (belief_loss), Pkg-04 spec 04 (双路径 backward) |
| spec 06 | Pkg-04 spec 02 (7 API + 调用顺序), Pkg-04 spec 08 §3.1 (4 处迁移行号) |
| spec 07 | spec 01 (trainer 内部状态), Pkg-01 spec 05 (cfg.train.ema_tau / lr_schedule) |
| spec 08 | spec 01-07 全引用 + Pkg-04 spec 08 (沿用契约模式) + Pkg-06/07/08 下游接入伪签名 |

### Day 6 机器可校验引用核对（沿用 Pkg-04 澄清 3）

Day 6 必须产出 `ref_matrix.csv`：

```powershell
cd D:\RL\hyper_mve\sdd\pkg-05-trainer-and-worker
pwsh scripts\check_ref_matrix.ps1        # ASCII-only, 也可在 Windows PowerShell 5.1 下运行
# → 生成 ref_matrix.csv + 终端 [PASS]/[FAIL]
```

引用扫描正则匹配两种 **包内** 引用形式，并排除 **跨包** 引用：

```powershell
# 见 scripts/check_ref_matrix.ps1
#   - 裸 "spec 0X"     但排除 "Pkg-NN spec 0X"（负向后顾 (?<!Pkg-\d{2} )）
#   - 文件名 "0X-name"（spec 间在上下游表里以文件名互引）
$pat = '(?<!Pkg-\d{2} )spec\s+0([1-8])|\b0([1-8])-[a-z]'
```

**验收**：每个 spec 至少引用 README §6 表所列的其他 spec（delta 列全空）；spec 08 引用 spec 01-07 全 7 个（契约层文件特性）。当前 `ref_matrix.csv`：**8/8 spec delta 全空 [PASS]**（机器校验，2026-06-01 重跑）。

---

## 🚦 Review Checklist（用户审阅时勾选）

- [ ] proposal.md 读完，**Why** 充分（v4.7 train_step_infer 已被 BeliefNet 取代 / buffer 字段不匹配 v4 TimeStepRecord / 11 处调用点迁移压力 → v4 必须重构 trainer/worker/buffer）
- [ ] design.md 读完，**10 项 Decisions** 中无反对意见（特别 D1 合并双套、D3 单一 train_main.py、D5 loss 抽出独立模块）
- [ ] specs/01-08 抽查至少 3 个（重点 specs/01 + specs/05 + specs/08，对应 trainer loop + loss 拼装 + 对外契约）
- [ ] 7 天实施顺序合理（Day 1 design.md 审阅是 hard gate）
- [ ] Q1-Q4 决议正确反映在 design.md §8 + 各 spec
- [ ] v4 关键差异认可：单一 train_step + 单一 train_main.py + EpisodeReplayBuffer 留 Pkg-05
- [ ] mve_planner.py 4 处调用点迁移与 Pkg-04 spec 08 §3.1 一致
- [ ] 出包后可启动 Pkg-06 (Baselines) + Pkg-07 (Eval) + Pkg-08 (Experiments)
