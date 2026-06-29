# 训练路线 — run_suite → analyze_results（工作流重建）

> **重建说明**：这份文档由代码（`scripts/run_suite.py`、`scripts/analyze_results.py`、
> `experiments/suite/manifest.yaml` 及各 cell）反推重建，用以替代丢失的对话记录
> （`D:\RL\data-restore` 中的记录经核对全部是无关主题，与本项目无关）。
> 工程来源：plan-mode 设计稿 `~/.claude/plans/training-has-now-commenced-rippling-hollerith.md`
> （suite + 分析层 + 4 指标扩展，已建好入 `main`）。
> 权威结构参考 [`OVERVIEW.md`](OVERVIEW.md)；**决策门 / 生死判官口径以
> [`docs/文章框架/Hyper_MuZero_v4_Roadmap.md`](../docs/文章框架/Hyper_MuZero_v4_Roadmap.md) Part 4 +
> [`docs/文章框架/Chapter6_Experiments_v4.md`](../docs/文章框架/Chapter6_Experiments_v4.md) §6.6 为准**
> （二者与本文冲突时以它们为权威）。

> ⚠️ **Phase 1 跑完不是"接着跑 Phase 2"，而是先撞「生死判官」GO/NO-GO 闸门。**
> 见下方 [§0.5 生死判官](#05-生死判官--phase-1-之后的强制暂停go-no-go) —— 这是你记得的那个
> "Phase 1 之后的重要暂停"。**没做出 GO 判定之前，不要启动 ~210 GPU-hr 的 Medium 主对比。**

---

## 0. 一句话流程

```
run_suite.py  (按 cell 跑训练，子进程一行一个 (variant,seed))
      │   每行写出 runs/suite/<runs_subdir>/registry.jsonl + 每行 TB/ckpt/eval_report.json
      ▼
analyze_results.py  (从某个 cell 的 registry 出表/图：Welch-t + Holm-Bonferroni)
      ▼
make_thesis_artifacts.py  (遍历整个 manifest，一次性渲染 Table/Fig 6.x → results/)
```

训练**没有真正的代码级 "phase" 概念**——所谓 phase 来自 `manifest.yaml` 里按
**决策门顺序**分组的 cell。本项目把它们排成「先便宜后昂贵」：

| Phase | 别名（决策门） | cells | 规模 | 目的 |
|---|---|---|---|---|
| **Phase 0** | 决策门 0 · gen_scope 选型 | `lora_duo_film` `lora_duo_film_fc2` `lora_duo_base` `lora_medium_film` `lora_medium_film_fc2` `lora_medium_base` | Medium / 300K | 先定 Hyper 的生成范围形态（断言 B′） |
| **Phase 1** | 决策点 1/2 · Easy 门（**便宜早期信号**） | `abl3_easy_n2` `abl1_easy` `zero_shot_easy` `main_comparison_easy` | Easy / 200K · 3 seeds | go/no-go 闸门，跑通且 Hyper 显著占优才进 Phase 2 |
| **Phase 2** | Medium headlines（**昂贵主结果**） | `abl3_type_heterogeneity` `abl1_gen_scope` `zero_shot_generalization` `main_comparison` | Medium / 0.2–1M · 5 seeds | 论文正文主表/主图 |
| **Phase 3** | 其余消融 | `abl4_crn_joint` `abl4_joint_easy_n2` `abl6_fehr_schmidt` `abl7_curriculum` | Easy/Medium | 补充消融 |
| **Phase 4** | BLOCKED / 补充 | `abl2_context_paths` `lr_sweep` | — | 依赖未就绪，最后或不做 |

> 你上次「全量训练只走完了 phase1」= **Easy 门四个 cell 已完成**，接下来要做的是：
> ① **撞生死判官闸门（§0.5）做 GO/NO-GO 判定** → ②（GO 才）跑 Phase 2 Medium 主结果。

**关键：`run_sweep` 会按 `(variant, seed, config_hash)` 跳过已完成的行**，所以重跑 suite
不会重训 Phase 1，会直接续到 Phase 2+。这就是「续训」机制——无需手动指定从哪开始。

> **前置约束**：决策门 0（gen_scope 选型，6 个 `lora_*`）**必须在生死判官之前收口**——
> 消融 3 要先固定 Hyper 自身形态。门 0 的硬门槛是 **TB 标量**（不是 registry 指标）：
> `diag/pi_mve_entropy` 要离开 ln(6)≈1.79 并持续下降、`diag/cos_pred_cross` < 0.95 且不上行。
> 用 `analyze_results.py --tb`（见 §2c）看，**不是** `--compare`。门 0 选型口径见
> `Review_v4_TheoryAudit_2026-06.md` §3.5 预注册矩阵。

---

## 0.5 生死判官 — Phase 1 之后的强制暂停（GO / NO-GO）

> 这是**本项目的核心闸门**。文档原话：「**这是本项目的核心生死判官实验**…决定是否继续
> Medium 主对比」（`Chapter6_Experiments_v4.md:356`）；「决策点 1：Easy 消融 3 -
> **项目生死判官**」（`Hyper_MuZero_v4_Roadmap.md:281`）。算力轻（~18–27 GPU-hr）但**决定项目命运**。

**生死判官 = 决策点 1 = `abl3_easy_n2` cell**：Easy / N=2，`hyper` vs `baseline_ma_muzero`，
跨 3 个类型组成点 ×3 seeds。指标 = `return_mean`（社会总福利 W_total）。

**判定矩阵（预注册，口径不可临时改）**：

| 在 `abl3_easy_n2` 的结果 | 含义 | 行动 |
|---|---|---|
| **`1α1β`（异质）处 Hyper vs MA-MuZero 福利差 > 8% 且 p<0.05** | 断言 A 在 Easy 立住 | ✅ **GO → 跑 Phase 2 Medium** |
| **`2α0β` / `0α2β`（同质）两者无显著差距** | 同质场景 Hyper 不额外占优（符合预期） | ✅ 继续 |
| **`1α1β` 处 Hyper 优势 < 5%** | 断言 A 可能失败 | 🛑 **NO-GO → 立即触发 P0 重新定位，不要跑 Medium** |
| **同质场景下 Hyper 也显著占优** | 优势来源**不是**"类型异质" | ⚠️ GO 但 Medium 须加 No-type/`no_belief` 消融排查通路 |

> 触发 NO-GO 的硬阈值：**钟形曲线最大差距 < 8% 且 p > 0.05**；Easy 峰值差 **< 5% 立即** P0，不等 Medium。

**怎么算这张表**（cell 内有 3 个 override 点，要按类型组成分组比，不能整体平均）：

```powershell
# 方式 A（推荐）：直接渲染 Table 6.4 钟形曲线表——按 (n_alpha,n_beta) 分组的 W_total
python hyper_mve/scripts/make_thesis_artifacts.py --suite-root runs/suite --out results
#   → results/tables/Table_6_4_Easy_gate/Table_6_4.md：每个 αβ 比例一列，hyper / ma_muzero 两行
#     肉眼比 1α1β 列两行的差距，并与 2α0β / 0α2β 列对照。
#   （依赖 config_snapshot 里的 env.type_assignment 才能分组；缺了会报 partial。）

# 方式 B：整体 Welch-t（不分组，作为辅助参考，不能单独支撑判定）
python hyper_mve/scripts/analyze_results.py --compare `
    --registry runs/suite/abl3_easy_n2/registry.jsonl `
    --metric return_mean --reference hyper --out runs/_analysis/abl3_easy_n2
```

**P0 重新定位预案（NO-GO 时按序排查，`Roadmap.md:296`）**：① BeliefNet 课程阶段 1 末主任务
loss 是否仍未降；② `set_context(τ_i)` 是否真的改变 `hyper_rew` 输出（type_emb 进 role 通路）；
③ 共享 RewardHead 是否真被两类梯度撕裂（打印偏导）；④ 三项都过仍无优势 = **真 P0**，与导师讨论
把贡献 2 从"核心创新"降为"统一框架"，或把重心移到 MVE+CRN 规划器（贡献 3）。

> Phase 1 另外三个 Easy 门（`abl1_easy`/`zero_shot_easy`/`main_comparison_easy`）是**决策点 2 的辅助
> 信号**，判定见 §2 的 go/no-go 段；它们不如生死判官"一票否决"，但 `abl1_easy` 若 Hyper 不优于
> Input-Wide/Deep，断言 B′(i) 走措辞降级预案。

---

## 0.7 执行 runbook — 从头到尾按这个顺序跑

> 来源：plan-mode 设计稿 `~/.claude/plans/training-has-now-commenced-rippling-hollerith.md`
> （*Modular Thesis Experiment Suite*）。**注意：那份 plan 是"把 suite/分析层/4 指标扩展
> 建出来"的工程规划，代码已建好并入 `main`。所以下面不再是写代码，而是按决策门顺序
> *运行* 已建好的 cell。** 每一步都能安全重跑（已完成行按 `(variant,seed,config_hash)` 跳过）。

| 步 | 做什么 | 命令（详见对应小节） | 闸门 / 产出 |
|---|---|---|---|
| **0** | 看盘，确认已完成到哪一格 | `run_suite.py --list`（§1） | — |
| **A** | 决策门 0：gen_scope 选型（6 `lora_*`） | `--only lora_duo_film,…,lora_medium_base`（§3 同款命令） | TB 硬门槛（§2c）→ 选 thesis-default gen_scope |
| **B** | Phase 1 Easy 门（4 cell） | `--only abl3_easy_n2,abl1_easy,zero_shot_easy,main_comparison_easy` | 产出 Easy 门 registry |
| **C** | ★ **生死判官 GO/NO-GO 暂停** | `make_thesis_artifacts.py`（出 Table 6.4）+ 判定（§0.5） | **GO 才进 D；NO-GO → P0** |
| **D** | Phase 2 Medium 主结果（4 headline） | `--only abl3_type_heterogeneity,abl1_gen_scope,zero_shot_generalization,main_comparison`（§3） | 决策点 3 判定 |
| **E** | Phase 3 其余消融 | `--only abl4_crn_joint,abl4_joint_easy_n2,abl6_fehr_schmidt,abl7_curriculum`（§4） | 断言 C/D + 鲁棒性 |
| **F** | 渲染全部论文表/图 | `make_thesis_artifacts.py`（§5） | `results/` |

**GPU 池**：headline cell 默认 `--gpus 2,3,4,5 --slots-per-gpu 2`；Phase 2/3 可加到
`2,3,4,5,6,7`。多 cell 同时跑会共享一个 GPU semaphore 跨 cell 填满（§3）。

**故意不跑的（plan 的 BLOCKED/deferred 清单，论文里如实披露，不进上面任何一步）**：
- `abl2_context_paths`（Table 6.3）— 缺 `no_type/no_cap/only_c` 变体，未实现 → 别跑。
- 外部 stub `external_mamba/marie/ga` — 跑到自动 skip（exit 2），方法表留空列。
- Fig 6.7 ∂R/∂u 梯度探针 — 未实现，`make_thesis_artifacts` 出占位符。
- `lr_sweep`（cuttable，μP LR 选型）— 要做单独 `--only lr_sweep`，不在主线。

> 路线一句话：**`--list` → (门0 gen_scope) → (Easy 门) → ★生死判官 GO/NO-GO → GO 才跑 Medium → 其余消融 → 出图表**。

---

## 1. 先看清盘面（不训练）

```powershell
cd D:\RL\hyper_mve
python hyper_mve/scripts/run_suite.py --list
```

打印每个 cell 的 tier / size / 行数 / deliverables / 是否 BLOCKED。
单 cell 预览（只枚举不跑）：

```powershell
python hyper_mve/scripts/run_suite.py --only main_comparison --dry-run
```

---

## 2. 分析 Phase 1（Easy 门）结果 ← 你卡住的这一步

每个 cell 各有自己的 registry：`runs/suite/<runs_subdir>/registry.jsonl`。
Phase 1 四个 cell 的 registry 路径与对应 deliverable：

| cell | registry | 主指标 | 交付物 |
|---|---|---|---|
| `abl3_easy_n2` (生死判官) | `runs/suite/abl3_easy_n2/registry.jsonl` | `return_mean` | Table 6.4 (Easy) |
| `abl1_easy` | `runs/suite/abl1_easy/registry.jsonl` | `return_mean` | Table 6.2 (Easy) |
| `zero_shot_easy` | `runs/suite/zero_shot_easy/registry.jsonl` | `return_zero_shot_unseen` | Table 6.7 (Easy) |
| `main_comparison_easy` | `runs/suite/main_comparison_easy/registry.jsonl` | `return_mean` | Table 6.1 (Easy) |

### 2a. 方法对比表（`--compare`，最常用）

以 `hyper` 为参照做 Welch-t + Holm-Bonferroni，输出 `compare_<metric>.md` + 柱状图 + 每变体 CSV：

```powershell
# 生死判官 Easy 门（断言 A 的核心闸门）—— 注意：判定要按类型组成分组比，见 §0.5；
# 下面这条整体 Welch-t 仅作辅助参考，不能单独支撑 GO/NO-GO。
python hyper_mve/scripts/analyze_results.py --compare `
    --registry runs/suite/abl3_easy_n2/registry.jsonl `
    --metric return_mean --reference hyper `
    --out runs/_analysis/abl3_easy_n2

# 条件化谱 Easy 门（断言 B′）
python hyper_mve/scripts/analyze_results.py --compare `
    --registry runs/suite/abl1_easy/registry.jsonl `
    --metric return_mean --reference hyper --out runs/_analysis/abl1_easy

# 主对比 Easy 门
python hyper_mve/scripts/analyze_results.py --compare `
    --registry runs/suite/main_comparison_easy/registry.jsonl `
    --metric return_mean --reference hyper --out runs/_analysis/main_comparison_easy

# 零样本 Easy 门 —— 用 unseen 指标更切题
python hyper_mve/scripts/analyze_results.py --compare `
    --registry runs/suite/zero_shot_easy/registry.jsonl `
    --metric return_zero_shot_unseen --reference hyper --out runs/_analysis/zero_shot_easy
```

> seed 不足 2（或变体 < 2）时 Welch-t 无定义，脚本自动退回到 `mean ± sem` 表。
> 可用 `--metric` 换任意 EvalReport 字段（如 `sustainability_mean`、`fairness_mean`、
> `tragedy_index_mean`、`regret_mean`、`return_zero_shot_unseen`）。

### 2b. 外部 runner 合规披露表（`--disclose`，仅 main_comparison*）

```powershell
python hyper_mve/scripts/analyze_results.py --disclose `
    --registry runs/suite/main_comparison_easy/registry.jsonl `
    --preset easy --out runs/_analysis/main_comparison_easy
```

### 2c. 纯 TB 标量（`--tb`，没有 registry 的目录用）

```powershell
python hyper_mve/scripts/analyze_results.py --tb `
    --tb-root runs/suite/abl3_easy_n2 --tag eval/planner/return_total `
    --out runs/_analysis/abl3_easy_n2
```

### Go / No-Go 判据

- **生死判官 `abl3_easy_n2`**：判定矩阵与阈值见 **[§0.5](#05-生死判官--phase-1-之后的强制暂停go-no-go)**
  （`1α1β` 差 > 8% & p<0.05 → GO；< 5% → NO-GO/P0）。这是**一票否决**门。
- **`abl1_easy` / `zero_shot_easy`**（决策点 2 辅助）：`hyper` 须优于 `baseline_input_wide/deep`
  → 断言 B′ 早期信号；不达标走 B′(i) 措辞降级，不一定砍项目。
- **只有生死判官 GO** 才启动 Phase 2 Medium（~210 GPU-hr）；NO-GO 先按 §0.5 的 P0 排查。

---

## 3. 跑 Phase 2（Medium 主结果）

> **前提：生死判官（§0.5）已判 GO。** 否则先做 P0 排查，别开 Medium。

因为 Phase 1 行已 completed，`run_sweep` 会自动跳过，所以**直接点名 Phase 2 的 cell** 即可。
跨 cell 共享一个 GPU 池（`run_suite` 会把多 cell 的行汇入同一 semaphore 填满 GPU）：

```powershell
# 一次跑完 4 个 Medium headline（按需调整 --gpus / --slots-per-gpu）
python hyper_mve/scripts/run_suite.py `
    --only abl3_type_heterogeneity,abl1_gen_scope,zero_shot_generalization,main_comparison `
    --gpus 2,3,4,5 --slots-per-gpu 2
```

或按 tier 一把梭（会自动跳过已完成的 Phase 1 must_have 行）：

```powershell
python hyper_mve/scripts/run_suite.py --tier must_have --gpus 2,3,4,5 --slots-per-gpu 2
```

> - `main_comparison` 的 3 个外部 stub（`external_mamba/marie/ga`）未 source 前 worker 退出码 2
>   → 自动 skip，不阻塞其余行。
> - 单 cell（如只想先跑主对比）：`--only main_comparison --gpus 2,3 --slots-per-gpu 1`。
> - `--max-steps N` 可临时压低步数做冒烟（不写进 cell）。

Phase 2 的分析命令同 §2，只是换 registry 路径（去掉 `_easy` 后缀）：

```powershell
python hyper_mve/scripts/analyze_results.py --compare `
    --registry runs/suite/main_comparison/registry.jsonl --reference hyper `
    --out runs/_analysis/main_comparison
python hyper_mve/scripts/analyze_results.py --compare `
    --registry runs/suite/zero_shot/registry.jsonl `
    --metric return_zero_shot_unseen --reference hyper --out runs/_analysis/zero_shot
python hyper_mve/scripts/analyze_results.py --compare `
    --registry runs/suite/abl3_type_het/registry.jsonl --reference hyper `
    --out runs/_analysis/abl3_type_het
```

（注意：`zero_shot_generalization` cell 的 `runs_subdir` 是 `zero_shot`，
`abl3_type_heterogeneity` 的是 `abl3_type_het`。）

---

## 4. Phase 3 / 其余消融

```powershell
python hyper_mve/scripts/run_suite.py `
    --only abl4_crn_joint,abl4_joint_easy_n2,abl6_fehr_schmidt,abl7_curriculum `
    --gpus 2,3,4,5 --slots-per-gpu 2
```

`abl2_context_paths` 处于 BLOCKED（`--list` 会标注），暂不跑。

---

## 5. 一次性渲染所有论文表/图

所有 cell 跑完（或部分跑完）后，遍历 manifest 把每个 deliverable 渲染到 `results/`：

```powershell
python hyper_mve/scripts/make_thesis_artifacts.py --suite-root runs/suite --out results
```

- 输出 `results/tables/`、`results/figures/`、`results/results_index.md`（逐 deliverable 状态：
  rendered / partial / blocked / no_data / error）。
- 未跑的 cell → `no_data`；BLOCKED 的（Table 6.3 / Fig 6.7）→ 占位 + TODO；
  图渲染需 matplotlib，缺了报 `error` 但表照常出。

---

## 6. 选择性重跑 / 续训（改了代码后）

harness **没有代码版本失效检测**，所以改了某 cell 底层代码后要显式强制重跑：

```powershell
# 只重跑 main_comparison 里 hyper / seed 0 这一行（其余已完成行不动）
python hyper_mve/scripts/run_suite.py --only main_comparison `
    --variants hyper --seeds 0 --force --gpus 2 --slots-per-gpu 1
```

`--force` 给被 narrow 的行追加一个 `failed` 墓碑（append-only 安全），再以 `retry_failed`
只重跑这些行。`--variants` / `--seeds` / `--size` 用来缩小「跑哪些行」。

---

## 7. 速查

```powershell
cd D:\RL\hyper_mve
python hyper_mve/scripts/run_suite.py --list                                  # 看盘
python hyper_mve/scripts/run_suite.py --only <cell> --dry-run                  # 预览
python hyper_mve/scripts/run_suite.py --tier must_have --gpus 2,3,4,5 --slots-per-gpu 2   # 跑必保（自动跳已完成）
python hyper_mve/scripts/analyze_results.py --compare --registry runs/suite/<subdir>/registry.jsonl --reference hyper --out runs/_analysis/<name>
python hyper_mve/scripts/make_thesis_artifacts.py --suite-root runs/suite --out results
```

变体面（cell 的 `variants:`）共 14 个：曲线 `hyper/oracle_only/infer_only`、内部
baseline 5 个、外部 6 个（`external_mamba/marie/ga` 为 stub 自动 skip）。
参照（`--reference`）一律用 `hyper`。
