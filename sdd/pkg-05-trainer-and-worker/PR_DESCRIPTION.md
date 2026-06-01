# PR: Pkg-05 (Trainer & Worker) SDD

> **状态**：Awaiting user final ack · **类型**：SDD only（不含代码实施）
> **范围**：`sdd/pkg-05-trainer-and-worker/` 11 文档 + ref_matrix.csv + check_ref_matrix.ps1

---

## Summary

- 完整 SDD 11 文档（README + proposal + design + specs 01-08）
- 落地 Day 1 review 5 项修订 + 3 项澄清（design.md §6.2/6.3/6.4/6.6 签名修订 + README 下游启动条件速查表 + spec 08 §2/§6 澄清 1/2）
- 19 项硬约束（C5-T1...R5-2）→ 具名单测全部映射到 spec（机器校验 PASS）
- 引用矩阵机器可校验（ref_matrix.csv + check_ref_matrix.ps1）

## Test plan

- [ ] 用户审阅 design.md §3 D1-D10 + §8 Q1-Q10（10 项决策 + 10 项决议无反对）
- [ ] 用户审阅 spec 01 §2.2 MuZeroTrainer 7 API + spec 05 §6.6 双路径 backward 强制顺序
- [ ] 用户审阅 spec 08 §1 4 类 API 稳定表 + §6 train_main.py 入口（澄清 1：infer_only 语义）
- [ ] 执行 ref_matrix 验证：`cd sdd/pkg-05-trainer-and-worker && pwsh scripts/check_ref_matrix.ps1` → 期望 [PASS]
- [ ] 与 Pkg-04 spec 07 性能预算联动确认（R5-1 400 ms 分摊与 Pkg-04 档位 3 100 ms 对账）

## 文档清单

```
sdd/pkg-05-trainer-and-worker/
├── README.md                            # 包索引 + 下游启动条件速查表 + 19 硬约束 + Q1-Q4 决议
├── proposal.md                          # Why / What Changes / R5-1...R5-10 风险（R5-1 含分摊表）
├── design.md                            # 10 Decisions D1-D10 + 6.2-6.8 API contract + R5-1 修订 3 分摊
├── ref_matrix.csv                       # Day 6 机器校验产出
├── PR_DESCRIPTION.md                    # 本文档
├── scripts/
│   └── check_ref_matrix.ps1             # 引用矩阵自动校验脚本
└── specs/
    ├── 01-trainer-loop-v2.md            # MuZeroTrainer + 单一 train_step + 400 ms 分摊
    ├── 02-worker-collection.md          # Worker + BeliefNet.step + planner 持有
    ├── 03-episode-buffer-v2.md          # EpisodeReplayBuffer + c_t_seq 显式 + stratified
    ├── 04-curriculum-scheduler.md       # CurriculumScheduler + 3 stage + lambda_b
    ├── 05-loss-composition.md           # compose_total_loss + 双路径 backward 强制顺序
    ├── 06-mve-planner-v4.md             # MVE planner CRN 保留 + 4 处 set_context 迁移
    ├── 07-ema-and-scheduler.md          # EMA tau=0.99 + warmup_cosine LR
    └── 08-integration-contracts.md      # 4 类 API 稳定 + 11 处迁移 + train_main.py 入口
```

## Verification 结果

### 引用矩阵（Day 6）

| spec | actual | expected | delta |
|------|--------|----------|-------|
| 01-trainer-loop-v2 | 02,03,04,05,06,07,08 | 04,05,06,07,08 | ✅ |
| 02-worker-collection | 01,03,04,06,08 | 03,06,08 | ✅ |
| 03-episode-buffer-v2 | 01,02,04,08 | 01,02,08 | ✅ |
| 04-curriculum-scheduler | 01,05,08 | 01,05,08 | ✅ |
| 05-loss-composition | 01,04,06,07,08 | 01,04,07,08 | ✅ |
| 06-mve-planner-v4 | 02,08 | 02,08 | ✅ |
| 07-ema-and-scheduler | 01,05,08 | 01,05,08 | ✅ |
| 08-integration-contracts | 01,02,03,04,05,06,07 | 01,02,03,04,05,06,07 | ✅ |

**8/8 spec delta 全空** ✅

### 19 项硬约束单测命名对账

| # | 约束 | 单测 | spec | 状态 |
|---|------|------|------|------|
| C5-T1 | trainer 每 train_step 调 update_step 1 次 | `test_trainer_calls_update_step_per_step` | spec 01 | ✅ |
| C5-T2 | K-step unroll 内 set_context_objective 一次 | `test_objective_called_once_per_unroll` | spec 01 | ✅ |
| C5-T3 | N agents 循环 set_context_subjective | `test_subjective_called_per_agent` | spec 01 | ✅ |
| C5-W1 | worker 不调 update_step | `test_worker_no_update_step` | spec 02 | ✅ |
| C5-W2 | worker 用 BeliefNet.step | `test_worker_uses_belief_net_step` | spec 02 | ✅ |
| C5-W3 | TimeStepRecord 字段顺序 | `test_z_hat_order_matches_pkg01_spec04` | spec 02 | ✅ |
| C5-B1 | buffer z_hat 顺序 | `test_buffer_z_hat_order_e2e` | spec 03 | ✅ |
| C5-B2 | stratified sampling 比例 | `test_stratified_min_per_type_frac` | spec 03 | ✅ |
| C5-S1 | 课程 3 stage 边界 | `test_curriculum_stage_boundaries` | spec 04 | ✅ |
| C5-S2 | Stage 1 100% oracle | `test_stage_1_full_oracle` | spec 04 | ✅ |
| C5-S3 | Stage 2 anneal 单调下降 | `test_oracle_mixing_anneal_monotonic` | spec 04 | ✅ |
| C5-L1 | λ_b 课程加权曲线 | `test_lambda_b_curve_matches_cfg` | spec 05 | ✅ |
| C5-L2 | L_belief 与 main 分离反向 | `test_belief_gradient_isolation_pre_5k` | spec 05 | ✅ |
| C5-P1 | CRN seed 相同 step 0 一致 | `test_crn_step0_deterministic_same_seed` | spec 06 | ✅ |
| C5-P2 | 4 处 set_context 迁移 | `test_planner_4_set_context_migrated` | spec 06 | ✅ |
| C5-E1 | EMA tau=0.99 衰减率 | `test_ema_decay_correctness_tau_099` | spec 07 | ✅ |
| C5-E2 | warmup_cosine LR 曲线 | `test_lr_warmup_5k_then_cosine_anneal` | spec 07 | ✅ |
| C5-I1 | v4.7 → v4 调用点 grep 0 残留 | `test_no_legacy_set_context_calls` | spec 08 | ✅ |
| R5-1 | train_step < 400 ms（含分摊表）| `test_train_step_under_400ms` | spec 01 | ✅ |
| R5-2 | sample_batch < 50 ms | `test_sample_batch_under_50ms` | spec 03 | ✅ |

**19/19 单测命名映射 PASS** ✅

## Pkg-04 联动确认（Day 2 前置 TODO）

- ✅ R5-1 性能预算与 Pkg-04 spec 07 档位 3 对账：v4 forward 100ms + target forward 50ms + belief forward 30ms + backward 180ms + optim 20ms + buffer 20ms = **400ms**（含 20ms 缓冲）
- ✅ C5-L2 双路径 backward 单测与 Pkg-04 spec 04 grad_gating 双层 detach 联动验证（`test_belief_gradient_isolation_pre_5k` + `test_belief_gradient_both_sources_post_5k`）
- ✅ spec 08 §2.3 grep 精确正则（澄清 2）排除 _objective/_subjective 误命中
- ✅ spec 08 §6.2 train_main.py --variant 表显式说明 infer_only ≠ v4.7 InferHyperMuZeroModel（澄清 1）

## 下游 SDD 启动条件（README §🔗 速查表）

| 包 | 启动条件 | 验收 spec |
|----|---------|----------|
| **Pkg-06 (Baselines)** | spec 08 §3 shared_backbones 工厂签名 + spec 02/03/05 接口稳定 | 已 finalize ✅ |
| **Pkg-07 (Eval)** | spec 08 §4 evaluator 接入 + spec 01/04 公开属性 | 已 finalize ✅ |
| **Pkg-08 (Experiments)** | spec 08 §5 ablation flag 列表 + §6 train_main.py 入口 + §7 checkpoint 格式 + Pkg-06/07 ack | 待 Pkg-06/07 ack |

---

🤖 Generated as part of Hyper-MuZero v4 SDD writing workflow
