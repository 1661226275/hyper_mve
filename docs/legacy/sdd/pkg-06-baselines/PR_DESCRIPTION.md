# PR: Pkg-06 (Baselines) SDD

> **状态**：Awaiting user final ack · **类型**：SDD only（不含代码实施）
> **范围**：`sdd/pkg-06-baselines/` 11 文档 + ref_matrix.csv + check_ref_matrix.ps1

---

## Summary

- 完整 SDD 11 文档（README + proposal + design + specs 01-08）
- 5 个 baseline 模型类（`input_wide` / `input_deep` / `ma_muzero` / `no_belief` / `rewardhead_explicit_type`）进 `create_baseline_model(cfg, variant)` 工厂；`hyper`/`oracle_only`/`infer_only` 不进工厂（共用 `HyperMuZeroModel` + curriculum override）
- 落地 Day 1 hard-gate 7 项验收清单 + 代码 review 6 项 BLOCKING（B1-B6）+ 5 项 SUGGESTED（M1-M5）修订
- 13 项硬约束（C6-FACT/FAIR/SELF/API/REUSE/GRAD/STRUCT/CFG/PERF）→ 具名单测全部映射到 spec
- 引用矩阵机器可校验（ref_matrix.csv + check_ref_matrix.ps1，实跑 [PASS]）

## Test plan

- [ ] 用户审阅 design.md §3 D1-D9（决策）+ §8.1 C6-* 13 约束映射表
- [ ] 用户审阅 spec 01 工厂签名 + 4 个待声明 cfg 字段（C6-CFG1，须 Pkg-01 spec05 同步）
- [ ] 用户审阅 spec 06 §2 统一 7-API + Self-Info 严格（`cap_i.shape[-1]==4`）+ belief grad-gating mermaid 双路径图
- [ ] 用户审阅 spec 07 §2 等参对照单元定义（双阈值 5%/10%）+ §4 OQ-1 wall-clock 步数等价性（W=100k / R*=0.80×oracle_only / ε=0.05×R*/100k）
- [ ] 执行 ref_matrix 验证：`powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\check_ref_matrix.ps1` → 期望 [PASS]

## 文档清单

```
sdd/pkg-06-baselines/
├── README.md                            # 包索引 + 决策速查表 + 13 硬约束 + 7×3 三角矩阵 + 下游启动条件
├── proposal.md                          # Why / What Changes / Capabilities / Impact
├── design.md                            # 9 Decisions D1-D9 + 等参对照单元 + 7 入口 vs 5 variant framing + §8 验证
├── ref_matrix.csv                       # Day 6 机器校验产出（delta 全空）
├── PR_DESCRIPTION.md                    # 本文档
├── scripts/
│   └── check_ref_matrix.ps1             # 引用矩阵自动校验脚本（纯 ASCII，实跑 [PASS]）
└── specs/
    ├── 01-baselines-factory-and-registry.md   # create_baseline_model + registry + supersede 声明 + 4 cfg 字段
    ├── 02-shared-backbones.md                  # create_rep/belief/tri_context + count_conditioning_params + 等参共享契约
    ├── 03-input-conditioned-baselines.md       # input_wide / input_deep（断言 B 等参对照，concat ctx_aug）
    ├── 04-ma-muzero-baseline.md                # ma_muzero 共享 RewardHead + own_type one-hot（断言 A 失败模式①）
    ├── 05-belief-type-ablation-baselines.md    # no_belief（断言 C/Abl7）+ rewardhead_explicit_type（断言 A/Abl6.x）
    ├── 06-model-7api-conformance.md            # 7-API 一致 + set_context 拆分 + Self-Info 严格 + gating 一致
    ├── 07-param-fairness-and-lr-sweep.md       # 双阈值 5%/10% + LR sweep + OQ-1 wall-clock + 性能护栏
    └── 08-integration-contracts.md             # 与 Pkg-05 工厂 + Pkg-07/08 硬契约 + 五条复用约束
```

## Verification 结果

### 引用矩阵（Day 6，实跑 [PASS]）

| spec | actual | expected | delta |
|------|--------|----------|-------|
| 01-baselines-factory-and-registry | 02,03,04,05,06,07,08 | 02,03,04,05,06,08 | ✅ |
| 02-shared-backbones | 01,03,04,05,06,07,08 | 03,04,05,06,08 | ✅ |
| 03-input-conditioned-baselines | 01,02,06,07,08 | 02,06,07,08 | ✅ |
| 04-ma-muzero-baseline | 02,05,06,07,08 | 02,06,07,08 | ✅ |
| 05-belief-type-ablation-baselines | 02,04,06,07,08 | 02,06,08 | ✅ |
| 06-model-7api-conformance | 02,03,04,05,08 | 02,03,04,05,08 | ✅ |
| 07-param-fairness-and-lr-sweep | 02,03,04,05,08 | 03,04,05,08 | ✅ |
| 08-integration-contracts | 01,02,03,04,05,06,07 | 01,02,03,04,05,06,07 | ✅ |

**8/8 spec delta 全空** ✅（脚本判据：expected ⊆ actual）

### 13 项硬约束单测命名对账

| # | 约束 | 单测 | spec | 状态 |
|---|------|------|------|------|
| C6-FACT1 | 5 variant 均可实例化 | `test_create_baseline_model_all_5_variants` | spec 01 | ✅ |
| C6-FACT2 | 工厂收 "hyper" 抛 ValueError | `test_factory_rejects_hyper_variant` | spec 01 | ✅ |
| C6-CFG1 | 工厂消费 4 个 baseline cfg 字段，无 hard-coded 默认 | `test_factory_reads_baseline_cfg_fields` | spec 01 | ✅ |
| C6-FAIR1 | 条件化子系统 ≤5% warn / ≤10% fail（input_wide/deep/no_belief）| `test_baseline_param_count_within_5pct` | spec 07 | ✅ |
| C6-FAIR2 | RepNet/BeliefNet/TriCtx 跨 variant 参数量逐位相同 | `test_shared_backbone_identical_param_count` | spec 02 | ✅ |
| C6-SELF1 | 不泄漏 oracle types（每 variant）| `test_baseline_set_context_subjective_no_oracle_types_leak` | spec 06 | ✅ |
| C6-API1 | 7 方法签名一致 | `test_baseline_implements_full_7api` | spec 06 | ✅ |
| C6-API2 | stateful 用最后一次 subjective agent_id（每 variant）| `test_baseline_predict_uses_last_subjective_agent_id` | spec 06 | ✅ |
| C6-REUSE1 | 同一 MuZeroTrainer/Worker/Buffer/compose_total_loss | `test_baseline_reuses_same_trainer_class` | spec 08 | ✅ |
| C6-GRAD1 | pre-5K BeliefNet 梯度=0（每 variant）| `test_baseline_update_step_gates_belief_grad` | spec 06 | ✅ |
| C6-STRUCT1 | input-conditioned 不含 DualHyperNetwork | `test_input_baseline_no_hypernet` | spec 03 | ✅ |
| C6-STRUCT2 | ma_muzero 无 per-agent θ_rew（共享单一头）| `test_ma_muzero_shared_rewardhead` | spec 04 | ✅ |
| C6-PERF1 | input baseline 单步 forward ≤ 2.0× hyper（性能护栏）| `test_baseline_forward_budget` | spec 07 | ✅ |

**13/13 单测命名映射 PASS** ✅

## Day 1 hard-gate 7 项验收（全过）

- ✅ 5 variant 核心区别表（含 ma_muzero vs rewardhead_explicit_type 撕分，D7）
- ✅ 等参对照单元定义（条件化消费子系统 + 逐 variant 账目 + 豁免规则，D5）
- ✅ belief grad-gating mermaid 双路径图（pre-/post-5K，覆盖 5 variant，D6）
- ✅ "7 训练入口 vs 5 工厂 variant" framing 澄清 + 7×3 三角矩阵
- ✅ cfg 新字段穷举（4 个待声明字段，挂"消费态、不改上游"标签）
- ✅ ref_matrix 预期表（8×expected）
- ✅ 7 天日历 + 响应式 SLA

## 代码 review 修订落地（B1-B6 BLOCKING + M1-M5 SUGGESTED）

- ✅ B1：spec 04 ma_muzero `set_context_subjective` 补 3-arg `grad_gating.apply(c_hat, z_hat, self._step)`
- ✅ B2：spec 04 `pred_net` 改无条件构建（防 `predict()` AttributeError）
- ✅ B3/B4：spec 03/05 `grad_gating.apply` 统一为 3-arg（逐字对齐 Pkg-04 spec 02 line 258）
- ✅ B5：spec 03/04 §2.2 头部注明"完整 7-API 见 spec 06 §2"
- ✅ B6：spec 08 头部加 Supersede 声明（联合 spec 01 supersede Pkg-04 spec 08 §5.1 旧草案）
- ✅ M1：spec 02 import 改 `from hyper_mve.models import BeliefNet, TriContextEncoder`（对齐 Pkg-03 spec 08 §6.2 line 350）
- ✅ M2：spec 05 explicit_type 补 `set_context_subjective` 骨架含 `assert cap_i.shape[-1]==4`
- ✅ M3：spec 07 `pytest.warns(UserWarning)` 改 `warnings.warn(..., UserWarning)`（真触发）
- ✅ M4：spec 07 OQ-1 给具体数值（W=100k / R*=0.80×oracle_only_plateau / ε=0.05×R*/100k）
- ✅ M5：spec 08 "line 136" → "lines 136-137"（136 import / 137 call）

## 下游 SDD 启动条件

| 包 | 启动条件 | 状态 |
|----|---------|------|
| **Pkg-07 (Eval)** | baseline 与 hyper 走同一 evaluator（spec 06 7-API 一致）| ✅ 本包已提供 |
| **Pkg-08 (Experiments)** | `train_main.py --variant baseline_*` + Abl 6.x/7 复用本包模型（spec 08 §3）| ✅ 本包已提供（待 Pkg-06 ack）|

---

🤖 Generated as part of Hyper-MuZero v4 SDD writing workflow
