# Pkg-08: Eval + Ablation Framework

> **状态**：Draft — Day 1 三件套待用户审阅（design.md = HARD GATE）
> **包 ID**：`pkg-08-eval-and-ablation`
> **上游契约**：[`pkg-07-baselines`](../pkg-07-baselines/README.md)（消费 `REGISTRY` + `ResourceCommonsPettingZooEnv` adapter + 统一 `.evaluate(env_fn, c_grid, episodes) -> EvalReport` 签名）
> **工期**：SDD 9 天（pkg-07 Day 3 起 overlap） · 实施 2–3 周 · **GPU 算力**（实施期）：≈ **850 GPU-hr**（下限 ≈ 700 砍 Tier-2 MAMBA + Abl6 部分单元；详见 proposal §4.2 细表）；2 GPU 并行 → wall-clock ≈ 350 hr ≈ 2 周。**不含** pkg-07 Tier-1 LR sweep；包括 oracle ceiling 缓存预算 + zero-shot grid + μP 自检 18 runs + Joint-N=2 enum + dual-mode planner eval。Plan File 早期估 620 GPU-hr 偏紧（仅 310 runs × 2 GPU-hr/run，未含 oracle ceiling 与多 seed × 多 preset）；本 README 与 proposal §4.2 合一为 850 GPU-hr。

---

## 📚 文档导览

| 文档 | 目的 | 字数 |
|------|------|------|
| [`proposal.md`](./proposal.md) | **先读这个**：Why（4 条断言需要可执行的消融与统计闭环 + Theory Audit §10.3 必备清单 + pkg-07 已提供的契约必须被消费而非重发明）/ What Changes / Capabilities / Impact / R8-1..R8-12 | ~5000 |
| [`design.md`](./design.md) | 设计抉择：Context / Goals / 框架澄清 / 10 项 Decisions（D1-D10）/ Decision Acceptance | ~7500 |
| [`specs/01-unified-evaluator.md`](./specs/01-unified-evaluator.md) | `hyper_mve/eval/unified_evaluator.py` 包装 `training/evaluation.py:run_eval`（**扩展不替换**）；`@dataclass EvalReport` 稳定 schema；c-grid / c-segment / bell-curve 三轴；pkg-07 spec 01 / spec 08 同步锁 | ~3500 |
| [`specs/02-zero-shot-and-c-hidden.md`](./specs/02-zero-shot-and-c-hidden.md) | 零样本协议（train `{0.2,0.5,0.8}` → test `{0.0,0.35,0.65,1.0}`，headline gap）；`cfg.env.c_visible` + Pkg-02 obs-mask 3 行下游补丁；`regret(c) = oracle_ceiling(c) - method(c)` 用 `runs/_oracle_ceilings/<config_hash>/<c>.json` 缓存 | ~3000 |
| [`specs/03-direct-inference-toggle.md`](./specs/03-direct-inference-toggle.md) | 四种 planner eval mode：`direct_inference` (alias 当前 `prior`) / `planner_no_crn` / `planner_no_coord_desc` / `planner_full`；新 `cfg.eval.eval_planner_mode: Literal[...]`；将 Ablation 4 cell 与 eval-mode 轴解耦 | ~2500 |
| [`specs/04-mup-verification.md`](./specs/04-mup-verification.md) | μP base-shape LR-doubling 自检（2 widths × 3 LRs × 3 seeds = 18 Easy runs）；一图一段落进论文；**失败 fallback**：drop μP 主张 + appendix 披露 | ~2500 |
| [`specs/05-sweep-harness-and-run-registry.md`](./specs/05-sweep-harness-and-run-registry.md) | `hyper_mve/experiments/sweep.py` — `SweepConfig` cartesian（variant × seed × override）；subprocess-per-row（process isolation）；`RunRegistry` = JSONL append-only at `runs/registry.jsonl`；per-GPU semaphore | ~3500 |
| [`specs/06-ablation-cli-and-cells.md`](./specs/06-ablation-cli-and-cells.md) | `python -m hyper_mve.experiments.ablate --ablation <id>` 分发到 `hyper_mve/experiments/ablations/*.yaml`；4 cells（Abl1 gen_scope 7-cell / Abl4 CRN × Joint-CoordDesc 重定义 / Abl6 Fehr-Schmidt 3×3 / Abl7 curriculum oracle_only/mixed/infer_only）；`use_coord_desc → randomize_order` rename（alias-with-deprecation, Theory Audit Q2） | ~3500 |
| [`specs/07-statistics-and-comparison.md`](./specs/07-statistics-and-comparison.md) | Welch t-test pipeline at `hyper_mve/experiments/stats.py`；>2 方法时 Holm-Bonferroni；CLI `compare --a runs/<a>.csv --b runs/<b>.csv` → markdown 表 + matplotlib bar+errorbar；**5 seeds 最低**（< 5 警告但不阻塞） | ~2500 |
| [`specs/08-integration-contracts.md`](./specs/08-integration-contracts.md) | **对外硬契约**：`EvalReport` + `RunRegistry` row schema 稳定 + 5 新 cfg 字段穷举（`c_visible` / `randomize_order` / `mve_joint_enumerate` / `eval_planner_mode` / `eval_use_planner_direct_inference`）+ 3 下游代码补丁（Pkg-02 obs-mask / Pkg-05 planner flag / Pkg-05 CLI 字串）+ 对 pkg-07 spec 01 / spec 08 的反向消费声明 | ~3500 |

---

## 🎯 一句话目标

**为论文 4 条断言闭合"消融 → 统计 → 对比"实验框架**：在 `hyper_mve/eval/` 下扩展 `training/evaluation.py:run_eval` 为 `unified_evaluator`（**扩展不替换**）并注册 `@dataclass EvalReport` 稳定 schema；在 `hyper_mve/experiments/` 下新增 `sweep.py`（subprocess-per-row + JSONL `RunRegistry`）+ `ablate` CLI（4 ablation cells + canned YAML 分发）+ `stats.py`（Welch t / Holm-Bonferroni）+ `compare` CLI；消费 pkg-07 提供的 `REGISTRY` 与 `ResourceCommonsPettingZooEnv` 适配器与统一 `evaluate()` 契约；对 Pkg-02/05 各打 1 处微补丁补齐 c_hidden 与 Joint 枚举（共 3 处下游代码补丁，**SDD 零修改**）；最终输出主表（hyper + 5 internal + 3 Tier-1 + MAMBA-if-sourced × 5 seeds × 2 presets）+ 4 ablation 表 + zero-shot 表 + μP 自检 + 论文-grade 比较图。

---

## ✅ 关键 Acceptance Criteria（速览）

> Eval 半（C8-EVAL-*）与 Ablation 半（C8-ABL-*）拆开列。两半在 spec 08 集成契约处合并。

### Eval 半（C8-EVAL-*） — must pass

- [ ] **`run_eval` 不被替换**，仅由 `unified_evaluator` 包装；现有 TB tag 全保留（C8-EVAL-REUSE1）
- [ ] **`@dataclass EvalReport` schema 冻结**（spec 08 §3 字段穷举）；internal `BaselineModel.evaluate()` 与 external `ExternalBaselineRunner.evaluate()` 都返回同一 schema（C8-EVAL-SCHEMA1）
- [ ] **零样本协议固化**：train `{0.2,0.5,0.8}` → test `{0.0,0.35,0.65,1.0}`，强制 4 个 unseen c 全部出现在 report（C8-EVAL-ZS1）
- [ ] **c_hidden 模式可运行**：`cfg.env.c_visible=False` 触发 Pkg-02 obs-mask 3 行下游补丁；信念叙事 ĉ 头在 c_hidden 档为「真推断」（Theory Audit Q7）（C8-EVAL-CHID1）
- [ ] **regret 指标**：每个 (config_hash, c) 用 oracle_only 跑出 ceiling 写 `runs/_oracle_ceilings/<config_hash>/<c>.json` 缓存；`regret(c) = ceiling(c) - method(c)`，cache miss 触发 oracle ceiling 重跑（C8-EVAL-REGRET1）
- [ ] **四种 planner eval mode** 全可枚举（`direct_inference` / `planner_no_crn` / `planner_no_coord_desc` / `planner_full`）；新 `cfg.eval.eval_planner_mode` literal，默认 `"planner_full"`（C8-EVAL-MODE1）
- [ ] **μP 自检 18 runs 跑出**（2 widths × 3 LRs × 3 seeds, Easy preset）；若 LR-doubling 不成立 → 论文 drop μP 主张并 appendix 披露（C8-EVAL-MUP1）
- [ ] **eval Self-Info 严格**：评估期 `set_context_subjective` **不接** oracle types（继承 pkg-04 spec02 + pkg-07 C7-INT-SELF1）（C8-EVAL-SELF1）
- [ ] **c-segment 三段聚合** 出现在 report（`[0,0.3]`/`[0.3,0.7]`/`[0.7,1.0]`，沿用现有 `cfg.eval.c_segments`）（C8-EVAL-SEG1）
- [ ] **bell-curve type-ratio sweep** 出现在 report（沿用现有 `cfg.eval.bell_curve_type_ratios`）（C8-EVAL-BELL1）

### Ablation 半（C8-ABL-*） — must pass

- [ ] **`SweepConfig` cartesian** 正确枚举 variant × seed × override；空 override 等价于裸 baseline 跑（C8-ABL-SWEEP1）
- [ ] **subprocess-per-row 进程隔离**：每 row 独立 Python 进程，GPU 故障不传染（C8-ABL-ISO1）
- [ ] **`RunRegistry` JSONL append-only**：row schema 冻结（spec 08 §4），并发安全（fcntl/msvcrt 文件锁）（C8-ABL-REG1）
- [ ] **`--ablation` CLI 5 个 canned ID 全可分发**：`abl1` / `abl4_crn_joint` / `abl4_joint_easy_n2` / `abl6` / `abl7`（C8-ABL-CLI1）
  - **量词澄清**：**4 逻辑 cells = 5 分发 ID**（Abl4 是 1 个逻辑 ablation，但 dispatch 拆成 2 个 YAML：`abl4_crn_joint`（2×2 CRN × Joint/CoordDesc 主表）+ `abl4_joint_easy_n2`（Easy N=2 的 6²=36 exhaustive enum 子单元）。其余 Abl1/Abl6/Abl7 各 1 YAML。)
- [ ] **Abl4 重定义**（Theory Audit M8 + Q2）：CRN × Joint/CoordDesc 真正撕分；Joint cell **仅** Easy N=2（6²=36 exhaustive enum，触发 Pkg-05 `mve_joint_enumerate=True` 单行下游补丁）（C8-ABL-ABL4A）
- [ ] **`use_coord_desc → randomize_order` rename**（Theory Audit Q2）：alias-with-deprecation；grep codebase 确认无直接 `use_coord_desc` 读者残留（C8-ABL-RENAME1）
- [ ] **Welch t-test 正确性**：使用 `scipy.stats.ttest_ind(equal_var=False)`；>2 方法 Holm-Bonferroni 校正；< 5 seeds 输出 `[WARN]` 但不阻塞（C8-ABL-STAT1）
- [ ] **comparison plot 论文-grade**：bar + errorbar（SEM）+ Welch t 标注；matplotlib backend = Agg，无 GUI 依赖（C8-ABL-PLOT1）

### 期望达到

- [ ] `pytest tests/eval/ tests/experiments/` 全过（C8-EVAL-* 与 C8-ABL-* 全条具名单测）
- [ ] `pwsh sdd/pkg-08-eval-and-ablation/scripts/check_ref_matrix.ps1` 输出 `[PASS]`
- [ ] `git diff` 确认 Pkg-01..05 SDD 零改动（代码上 Pkg-02 obs-mask + Pkg-05 planner flag + Pkg-05 CLI 三处微补丁由 spec 08 显式声明）；Pkg-07 SDD 零改动（仅消费契约）

---

## 📅 实施顺序（9 天 SDD + 后续实施）

### SDD 阶段（9 天；从 pkg-07 Day 3 起 overlap）

| 阶段 | 天数 | 任务 | 输出 / 审阅点 |
|------|------|------|---------------|
| **Phase 1** | Day 1 | README + proposal + design.md（10 Decisions 锁定）| **design.md 审阅 = HARD GATE**（**10 项**验收清单，全过才放行；详见 §"Day 1 HARD GATE 验收清单"）|
| **Phase 2** | Day 2 | spec 01（unified-evaluator + `EvalReport` schema）| **与 pkg-07 spec 01 / spec 08 同步**（`evaluate(env_fn, c_grid, episodes) -> EvalReport` 签名锁死 + `EvalReport` 字段冻结）|
| **Phase 3** | Day 3 | spec 05（sweep harness + RunRegistry）| sweep cartesian + JSONL row schema + subprocess 隔离 |
| **Phase 4** | Day 4 | spec 02（zero-shot + c_hidden + regret 缓存）| Pkg-02 obs-mask 3 行下游补丁声明 + ceiling cache schema |
| **Phase 5** | Day 5 | spec 03（direct-inference toggle）+ spec 04（μP 自检）| `eval_planner_mode` literal 锁 + μP 18-run 计划 |
| **Phase 6** | Day 6 | spec 06（ablation CLI + 4 cells）| `randomize_order` rename + Pkg-05 `mve_joint_enumerate` 1 行下游补丁声明 |
| **Phase 7** | Day 7 | spec 07（Welch t + Holm-Bonf + compare CLI）+ spec 08（集成契约 + 反向消费 pkg-07）| 5-seed 协议 + 3 下游补丁清单 + pkg-07 → pkg-08 契约对账 |
| **Phase 8** | Day 8 | `scripts/check_ref_matrix.ps1`（纯 ASCII）+ `ref_matrix.csv` 真跑 `[PASS]` | M6 引用核对 |
| **Phase 9** | Day 9 | PR_DESCRIPTION.md + 用户最终 ack | Pkg-08 SDD finalized |

### Day 1 HARD GATE 验收清单（10 项，全过才放行）

| # | 验收项 | design.md 落点 |
|---|--------|---------------|
| 1 | 8-spec Eval/Ablation 半分配清晰（specs 01-04 = Eval 半 / specs 05-08 = Ablation 半 / spec 08 = 集成契约）| §3.1 + §4 D1/D2 |
| 2 | **`@dataclass EvalReport` 字段穷举**（per-c return / planner-prior gap / zero-shot unseen-c block / c-segment 三段 / bell-curve type-ratio / regret-vs-ceiling / 4-mode planner 标签）| §3.2 + §4 D3 |
| 3 | **c_hidden 模式实施路径**（`cfg.env.c_visible: bool=True` 默认；Pkg-02 `envs/resource_commons/observations.py` +3 行 obs-mask；与 Theory Audit Q7 对齐）| §3.3 + §4 D4 |
| 4 | **regret 指标定义** + oracle ceiling cache schema（`runs/_oracle_ceilings/<config_hash>/<c>.json` 字段穷举 + cache miss 触发 oracle 重跑）| §3.4 + §4 D6 |
| 5 | **四 planner eval mode 解耦表**（mode × `use_crn` × `use_coord_desc/randomize_order` × `direct_inference` 真值表）| §3.5 + §4 D7 |
| 6 | **sweep harness 进程隔离** + `RunRegistry` row schema（subprocess-per-row 理由 + JSONL append-only + 并发文件锁）| §3.6 + §4 D8 |
| 7 | **`--ablation` CLI 5 ID 分发表**（abl1 / abl4_crn_joint / abl4_joint_easy_n2 / abl6 / abl7 → canned YAML 路径表）| §3.7 + §4 D9 |
| 8 | **5 新 cfg 字段穷举**（`c_visible` / `randomize_order` / `mve_joint_enumerate` / `eval_planner_mode` / `eval_use_planner_direct_inference` — 字段名、默认值、归属、消费 spec）| §3.8 + §4 D10 |
| 9 | **8-spec ref_matrix 预期表**（per-spec 引用矩阵）| §8 |
| 10 | 9 天日历 + 响应式 SLA + Phase 8 真跑 | §6 |

### 响应式 SLA

日历是**名义节奏，非硬截止**。**Day 1 hard gate 不通过 → 全流程顺延 1 天**（不带病进 Day 2）。每个 spec 当天末交审，等用户 ack 再进下一个，审阅往返时间不计入"天"。

### 实施期（SDD 落地后 2–3 周，详见 Plan File 绝对路径 `C:\Users\zhengwenbo01\.claude\plans\after-some-thought-i-iridescent-rain.md` §"Code-implementation" Phase D-F；注：plan file 跨 C:/D: 盘符，相对路径不可解析，请用绝对路径打开）

- Phase D（wk 4–5）：unified evaluator + run registry + sweep harness（spec 01/05 落码）
- Phase E（wk 5–6）：zero-shot + c_hidden + regret + 4 ablation cells（spec 02/03/06 落码 + 3 下游补丁）
- Phase F（wk 6–7）：stats + compare CLI + μP 自检 + 主表 + 论文图（spec 04/07/08 落码 + 最终图签字）

---

## 🔑 10 项关键 Decisions（速览）

> **量词澄清**：Decisions 数 = **10**（D1-D10）。**Decision Acceptance / Day-1 HARD GATE 验收清单 = 10 项**（与 Decisions 一一对应；不像 pkg-07 多 1 行 CLI 表落 §3.3）。

| # | Decision | 推荐 | 详见 |
|---|----------|------|------|
| D1 | spec 数量与文档结构 | **8 specs 对称**（继承 pkg-04/05/06/07 模式），分 Eval 半（01-04）+ Ablation 半（05-08）| [design §4 D1](./design.md) |
| D2 | Eval / Ablation 半分配 | **specs 01-04 = Eval 半**（unified evaluator / zero-shot+c_hidden / direct-inference / μP）；**specs 05-08 = Ablation 半**（sweep+registry / ablation CLI+cells / stats / 集成契约）| [design §4 D2](./design.md) |
| D3 | `EvalReport` dataclass schema | **`@dataclass(frozen=True) EvalReport`**：per-c return / planner-prior gap / zero-shot unseen-c block / c-segment 三段 / bell-curve 比 / regret-vs-ceiling / 4-mode planner 标签；同时供 `internal BaselineModel.evaluate()` 与 `external ExternalBaselineRunner.evaluate()` 返回（pkg-07 spec 08 反向消费）| [design §4 D3](./design.md) |
| D4 | c_hidden 实施路径 | **`cfg.env.c_visible: bool=True` 默认 + Pkg-02 `envs/resource_commons/observations.py` +3 行 obs-mask**（Theory Audit Q7 处置；下游补丁，不改 Pkg-02 SDD）| [design §4 D4](./design.md) |
| D5 | zero-shot 网格归属 | **cfg-驱动而非硬编码**：沿用 `cfg.eval.zero_shot_train_c` + `zero_shot_test_c` + `zero_shot_unseen_c` 三字段（已在 `configs/eval_config.py` line 37-39 reserved）；spec 02 锁默认值与 Pkg-07 Ch6.9 一致 | [design §4 D5](./design.md) |
| D6 | regret 指标计算 | **使用 oracle ceiling cache**（`runs/_oracle_ceilings/<config_hash>/<c>.json`）；cache miss → 自动触发 `oracle_only` 跑 ceiling 写入；`regret(c) = ceiling(c) - method(c)`；ceiling 5 seeds 平均 | [design §4 D6](./design.md) |
| D7 | planner eval mode 解耦 | **4 mode literal**：`direct_inference`（= 当前 `prior`，π̂ 直推，alias）/ `planner_no_crn`（planner + `use_crn=False`）/ `planner_no_coord_desc`（planner + `randomize_order=False`，原 `use_coord_desc=False`）/ `planner_full`（planner + 全启用）；与 Abl4 cell 解耦（Abl4 跑的是 training-time 选项；eval mode 跑的是 evaluation-time 选项）| [design §4 D7](./design.md) |
| D8 | sweep harness 进程隔离 | **subprocess-per-row**（`subprocess.Popen` + JSON over stdin）+ **JSONL append-only `RunRegistry`** + per-GPU semaphore；理由：CUDA OOM/segfault 不传染 sibling rows | [design §4 D8](./design.md) |
| D9 | `--ablation` CLI 分发策略 | **canned YAMLs** under `hyper_mve/experiments/ablations/*.yaml`；5 ID：`abl1` / `abl4_crn_joint` / `abl4_joint_easy_n2` / `abl6` / `abl7`；YAML 描述 sweep cartesian + override，CLI 仅做 dispatch | [design §4 D9](./design.md) |
| D10 | 5 新 cfg 字段归属 | **`cfg.env.c_visible: bool`**（spec 02 消费）/ **`cfg.train.randomize_order: bool`**（spec 06 消费，rename from `use_coord_desc`, alias-with-deprecation）/ **`cfg.train.mve_joint_enumerate: bool`**（spec 06 消费，Pkg-05 1 行下游补丁）/ **`cfg.eval.eval_planner_mode: Literal[...]`**（spec 03 消费）/ **`cfg.eval.eval_use_planner_direct_inference: bool`**（spec 03 消费，留作 mode 短路开关）| [design §4 D10](./design.md) |

---

## 🚨 v4 关键约束（继承 + 新增）

> 统一前缀：`REUSE`(复用) / `SCHEMA`(数据契约) / `ZS`(zero-shot) / `CHID`(c_hidden) / `REGRET`(regret 指标) / `MODE`(planner eval mode) / `MUP`(μP 自检) / `SELF`(Self-Info) / `SEG`(c-segment) / `BELL`(bell-curve) / `SWEEP`(sweep harness) / `ISO`(进程隔离) / `REG`(RunRegistry) / `CLI` / `ABL4`(Abl4 重定义) / `RENAME` / `STAT`(统计) / `PLOT`(论文图)。
> 二级前缀：`EVAL`(Eval 半) / `ABL`(Ablation 半)。

### Eval 半（C8-EVAL-*）

| # | 约束 | 来源 | 对本包的硬约束 |
|---|------|------|----------------|
| **C8-EVAL-REUSE1** | `training/evaluation.py:run_eval` 仅被包装，**不被替换**；现有 TB tag 全保留 | 复用代码现状 + Plan §pkg-08 spec 01 | spec 01 单测 `test_run_eval_not_modified` |
| **C8-EVAL-SCHEMA1** | `@dataclass(frozen=True) EvalReport` 字段冻结（spec 08 §3 穷举）；internal+external runner 都返回同一 schema | design D3 + pkg-07 spec 08 | spec 01 + spec 08 单测 `test_eval_report_schema_frozen` |
| **C8-EVAL-ZS1** | 零样本协议固化（`train={0.2,0.5,0.8}` → `test={0.0,0.35,0.65,1.0}`），4 个 unseen c 全部出现 | Theory Audit §10.3 + Ch6.9 | spec 02 单测 `test_zero_shot_unseen_c_all_present` |
| **C8-EVAL-CHID1** | `cfg.env.c_visible=False` 触发 Pkg-02 obs-mask；ĉ 头在 c_hidden 档为真推断 | Theory Audit Q7 + M12 | spec 02 单测 `test_c_hidden_obs_masked` + Pkg-02 obs-mask 3 行下游补丁 |
| **C8-EVAL-REGRET1** | `regret(c) = ceiling(c) - method(c)`；cache miss 触发 oracle ceiling 重跑；ceiling 5 seeds 平均 | design D6 | spec 02 单测 `test_regret_cache_miss_triggers_oracle_run` |
| **C8-EVAL-MODE1** | 四种 planner eval mode 全可枚举（`direct_inference` / `planner_no_crn` / `planner_no_coord_desc` / `planner_full`）；默认 `"planner_full"` | design D7 + Theory Audit M8 | spec 03 单测 `test_all_4_planner_modes_dispatch` |
| **C8-EVAL-MUP1** | μP 自检 18 runs（2 widths × 3 LRs × 3 seeds, Easy preset）；LR-doubling 失败 → 论文 drop μP 主张 | Theory Audit §10.3 pkg-07 备忘 | spec 04 + 一图一段落进 Ch6 |
| **C8-EVAL-SELF1** | eval 期 `set_context_subjective` **不接** oracle types | 继承 pkg-04 spec02 + pkg-07 C7-INT-SELF1 | spec 01 单测 `test_eval_set_context_no_oracle_types_leak` |
| **C8-EVAL-SEG1** | c-segment 三段聚合出现在 report（沿用 `cfg.eval.c_segments`）| Ch6.2.4 | spec 01 单测 `test_c_segment_aggregation_in_report` |
| **C8-EVAL-BELL1** | bell-curve type-ratio sweep 出现在 report（沿用 `cfg.eval.bell_curve_type_ratios`）| Ch6.6 | spec 01 单测 `test_bell_curve_in_report` |

### Ablation 半（C8-ABL-*）

| # | 约束 | 来源 | 对本包的硬约束 |
|---|------|------|----------------|
| **C8-ABL-SWEEP1** | `SweepConfig` cartesian 正确枚举 variant × seed × override；空 override 等价于裸 baseline 跑 | design D8 | spec 05 单测 `test_sweep_cartesian_correct` |
| **C8-ABL-ISO1** | subprocess-per-row 进程隔离；CUDA OOM/segfault 不传染 sibling rows | design D8 | spec 05 单测 `test_sweep_subprocess_isolation` |
| **C8-ABL-REG1** | `RunRegistry` JSONL append-only；row schema 冻结（spec 08 §4）；并发安全（fcntl/msvcrt 文件锁）| design D8 + spec 08 | spec 05 单测 `test_run_registry_concurrent_safe` |
| **C8-ABL-CLI1** | `--ablation` CLI 5 ID 全分发（`abl1` / `abl4_crn_joint` / `abl4_joint_easy_n2` / `abl6` / `abl7`）| design D9 | spec 06 单测 `test_ablation_cli_dispatch_all_5` |
| **C8-ABL-ABL4A** | Abl4 重定义：CRN × Joint/CoordDesc 真正撕分；Joint cell 仅 Easy N=2（6²=36 enum）；触发 Pkg-05 `mve_joint_enumerate=True` 单行下游补丁 | Theory Audit M8 + Q2 | spec 06 单测 `test_abl4_joint_enum_easy_n2_only` |
| **C8-ABL-RENAME1** | `use_coord_desc → randomize_order` rename（alias-with-deprecation）；grep codebase 确认无直接 `use_coord_desc` 读者残留 | Theory Audit Q2 | spec 06 单测 `test_use_coord_desc_alias_warns` + grep 自检 |
| **C8-ABL-STAT1** | Welch t-test 用 `scipy.stats.ttest_ind(equal_var=False)`；>2 方法 Holm-Bonferroni 校正；< 5 seeds 输出 `[WARN]` 但不阻塞 | Ch6.2.3 + Theory Audit §10.3 | spec 07 单测 `test_welch_t_correctness` |
| **C8-ABL-PLOT1** | comparison plot 论文-grade：bar + errorbar (SEM) + Welch t 标注；matplotlib backend = Agg | design D9 + Ch6 论文图风格 | spec 07 单测 `test_compare_plot_renders_headless` |

---

## 📦 输出清单（PR 时检查）

### 新增（unified evaluator + sweep harness + ablation CLI + stats + 5 canned YAMLs，全部在 `hyper_mve/eval/` & `hyper_mve/experiments/`）

```
hyper_mve/eval/
├── __init__.py                          # re-export EvalReport + UnifiedEvaluator
├── unified_evaluator.py                 # wraps training/evaluation.py:run_eval（扩展不替换）
├── eval_report.py                       # @dataclass(frozen=True) EvalReport（spec 08 schema）
├── zero_shot.py                         # train {0.2,0.5,0.8} → test {0.0,0.35,0.65,1.0}
├── c_hidden.py                          # cfg.env.c_visible=False 评估通路（消费 Pkg-02 obs-mask）
├── regret.py                            # oracle ceiling cache + regret(c) 计算
├── planner_modes.py                     # direct_inference / planner_no_crn / planner_no_coord_desc / planner_full
└── mup_verification.py                  # μP base-shape LR-doubling 18-run 自检

hyper_mve/experiments/
├── __init__.py
├── sweep.py                             # SweepConfig + subprocess-per-row + per-GPU semaphore
├── run_registry.py                      # JSONL append-only + 并发文件锁
├── ablate.py                            # __main__ entry: --ablation <id> 分发
├── stats.py                             # Welch t + Holm-Bonferroni
├── compare.py                           # __main__ entry: --a / --b → md 表 + plot
└── ablations/
    ├── abl1_gen_scope.yaml              # 7-cell gen_scope（Theory Audit §2.2）
    ├── abl4_crn_joint.yaml              # CRN × Joint/CoordDesc 2×2（重定义后）
    ├── abl4_joint_easy_n2.yaml          # Easy N=2 only，6²=36 exhaustive enum cell
    ├── abl6_fehr_schmidt.yaml           # Fehr-Schmidt 3×3 扫描
    └── abl7_curriculum.yaml             # oracle_only / mixed / infer_only

tests/eval/
├── test_run_eval_not_modified.py        # C8-EVAL-REUSE1
├── test_eval_report_schema_frozen.py    # C8-EVAL-SCHEMA1
├── test_zero_shot_unseen_c_all_present.py # C8-EVAL-ZS1
├── test_c_hidden_obs_masked.py          # C8-EVAL-CHID1
├── test_regret_cache_miss_triggers_oracle_run.py # C8-EVAL-REGRET1
├── test_all_4_planner_modes_dispatch.py # C8-EVAL-MODE1
├── test_eval_set_context_no_oracle_types_leak.py # C8-EVAL-SELF1
├── test_c_segment_aggregation_in_report.py # C8-EVAL-SEG1
└── test_bell_curve_in_report.py         # C8-EVAL-BELL1

tests/experiments/
├── test_sweep_cartesian_correct.py      # C8-ABL-SWEEP1
├── test_sweep_subprocess_isolation.py   # C8-ABL-ISO1
├── test_run_registry_concurrent_safe.py # C8-ABL-REG1
├── test_ablation_cli_dispatch_all_5.py  # C8-ABL-CLI1
├── test_abl4_joint_enum_easy_n2_only.py # C8-ABL-ABL4A
├── test_use_coord_desc_alias_warns.py   # C8-ABL-RENAME1
├── test_welch_t_correctness.py          # C8-ABL-STAT1
└── test_compare_plot_renders_headless.py # C8-ABL-PLOT1
```

### 修改（**3 行为补丁 + 1 零行为 threading 补丁 + 3 配置字段声明，分三块**；纯下游契约，不动 pkg-01..05 SDD，不动 pkg-07 SDD）

**(A) 3 行为补丁（实际代码语义变化；由 spec 08 「下游补丁声明」 锁定）:**

```
hyper_mve/envs/resource_commons/observations.py    # +3 行 c_visible 掩码 + 1 行 env_cfg kwarg 签名（spec 02 §5.1 + spec 08 声明）
hyper_mve/planning/mve_planner.py                  # +1 行 mve_joint_enumerate 分支（spec 06 + spec 08 声明）
hyper_mve/scripts/train_main.py                    # 重命名 --use_coord_desc → --randomize_order CLI 字串（alias 保留旧名 + DeprecationWarning，spec 06 + spec 08 声明）
```

**(A') 1 零行为 threading 补丁（由 spec 02 §5.1b 显式声明；带 `[no-behaviour]` 标签 + byte-identical 回归单测）:**

```
hyper_mve/envs/resource_commons/env.py             # +2 行 `env_cfg=self.cfg` kwarg 透传到 build_joint_observation 的两处 call-site（reset / step）；零行为变化；test_env_py_call_site_kwarg_no_behaviour_change 断言（spec 02 §5.1b + spec 08 声明）
```

**(B) 3 配置字段声明（消费态 +5 字段加到现有 3 个 config dataclass；不改语义；由 Pkg-01 spec05 同步消费）:**

```
hyper_mve/configs/env_config.py                    # +1 字段 c_visible: bool=True
hyper_mve/configs/train_config.py                  # +2 字段 randomize_order (alias of use_coord_desc) + mve_joint_enumerate
hyper_mve/configs/eval_config.py                   # +2 字段 eval_planner_mode: Literal[...] + eval_use_planner_direct_inference: bool
```

（合计 5 字段，匹配下表「5 新 cfg 字段穷举」；A 块 = 3 行为补丁；A' 块 = 1 零行为 threading 补丁（带回归单测）；B 块 = 3 dataclass 字段增量。诚实分块,不混计。）

### 5 新 cfg 字段穷举（消费态、不改上游 SDD；由 Pkg-01 spec05 同步声明）

| 字段 | 默认 | 归属 | 消费 spec |
|------|------|------|----------|
| `cfg.env.c_visible: bool` | `True` | `EnvConfig` | spec 02（c_hidden 评估） |
| `cfg.train.randomize_order: bool` | mirrors `use_coord_desc` (alias) | `TrainConfig` | spec 06（Abl4 重定义 + rename） |
| `cfg.train.mve_joint_enumerate: bool` | `False` | `TrainConfig` | spec 06（Abl4 Joint cell, Easy N=2 only） |
| `cfg.eval.eval_planner_mode: Literal["direct_inference","planner_no_crn","planner_no_coord_desc","planner_full"]` | `"planner_full"`（**精化自 Plan File loose default `"planner"`**；pkg-08 design D7 锁定四模式精确命名，`"planner_full"` ≡ Plan File 中 `"planner"` 的语义,即 CRN+CoordDesc 全开的默认规划路径）| `EvalConfig` | spec 03（four-mode eval dispatch） |
| `cfg.eval.eval_use_planner_direct_inference: bool` | `False` | `EvalConfig` | spec 03（mode 短路开关，与 `eval_planner_mode="direct_inference"` 等价） |

---

## 🔗 上下游关联

| 关系 | 包 | 接口 |
|------|-----|------|
| **本包 ← Pkg-07**（核心契约） | Baselines | `from hyper_mve.baselines import REGISTRY`（11-key 只读 `MappingProxyType`）+ `from hyper_mve.envs.adapters.pettingzoo_wrapper import ResourceCommonsPettingZooEnv` + `BaselineModel.evaluate() / ExternalBaselineRunner.evaluate() -> EvalReport` 统一签名（pkg-07 spec 01 + spec 08 锁死，pkg-08 spec 01 + spec 08 反向消费）|
| **本包 ← Pkg-02** | ResourceCommonsEnv | `envs/resource_commons/observations.py` +3 行 obs-mask 下游补丁（spec 02 + spec 08 声明）|
| **本包 ← Pkg-05** | Trainer + Planner | `planning/mve_planner.py` +1 行 `mve_joint_enumerate` 分支（spec 06 + spec 08）；`scripts/train_main.py` `--randomize_order` CLI rename（spec 06 + spec 08）|
| **本包 ← Pkg-01** | Foundation Schema | 5 新 cfg 字段在 Pkg-01 spec05 同步消费态声明（不改 Pkg-01 SDD）|
| **本包 ← Pkg-04** | DualHyperNetwork v2 | `set_context_subjective` Self-Info 严格（eval 期不接 oracle types，C8-EVAL-SELF1 继承）|
| **本包 → 终端** | （无下游 SDD 包） | 产出主表 + 4 ablation 表 + zero-shot 表 + μP 自检图，直接进 Ch6；论文成稿后回到 `academic-research` 分支续写 |

### 📌 上游启动条件（pkg-07 提供给 pkg-08 的"已就绪"信号）

| 来自 pkg-07 | 启动条件 | 验收 spec |
|------------|---------|-----------|
| `REGISTRY` 11 keys finalized（5 internal + 3 Tier-1 + MAMBA + 2 stubs；stubs 在 sweep 中产生 "skipped: NotImplementedError" rows）| pkg-07 spec 01 §2 锁 | pkg-08 spec 05 消费 |
| `evaluate(env_fn, c_grid, episodes) -> EvalReport` 签名稳定 | pkg-07 spec 08 锁 | pkg-08 spec 01 反向消费 |
| `ResourceCommonsPettingZooEnv` N-parametric 契约 + 两 flag 信息门控（`oracle_mode=False` AND `eval_info_mode=False`）| pkg-07 spec 04 锁 | pkg-08 spec 01（external runner eval 通路）/ spec 02（c_hidden 通路，与 pkg-07 `oracle_mode` 正交）|
| `cfg.baselines.*` namespace + `cfg.baselines.external_lr_sweep_grid: Mapping[str, tuple]` 锁 | pkg-07 D10 | pkg-08 spec 05（sweep 默认枚举 LR sweep grid） |

### 🔁 pkg-07 → pkg-08 契约对账（spec 08 内逐字回引）

```
pkg-07 spec 01 §2.1  REGISTRY 11 keys                 ──→ pkg-08 spec 05 §3 sweep enumeration
pkg-07 spec 01 §2.3  量词 canonical (11 keys)         ──→ pkg-08 spec 05 §3 + spec 08 §5
pkg-07 spec 04       N-parametric adapter + 两 flag   ──→ pkg-08 spec 01 §4 external runner eval 通路
pkg-07 spec 08       evaluate() → EvalReport 签名     ──→ pkg-08 spec 01 §3 EvalReport schema + spec 08 §3
pkg-07 spec 08       BaselineLike Union type          ──→ pkg-08 spec 05 sweep harness type sig
```

任何 pkg-07 上述 5 锚点漂移 → 触发 pkg-08 spec 01 / spec 05 / spec 08 同步修订（spec 08 §6 §"漂移检测"章节强制 reviewer 对账）。

---

## 📖 引用源

- Plan File 绝对路径：`C:\Users\zhengwenbo01\.claude\plans\after-some-thought-i-iridescent-rain.md`（pkg-07/08 整体规划与决策记录，pkg-08 specs 01-08 spec table；跨 C:/D: 盘符，无法用相对路径链接）
- Pkg-07 SDD: [`pkg-07-baselines/README.md`](../pkg-07-baselines/README.md) + [`pkg-07-baselines/design.md`](../pkg-07-baselines/design.md) + [`pkg-07-baselines/specs/01-baseline-registry-and-cli.md`](../pkg-07-baselines/specs/01-baseline-registry-and-cli.md)（REGISTRY + adapter + evaluate 签名上游）
- Theory Audit: [`docs/Review_v4_TheoryAudit_2026-06.md`](../../docs/Review_v4_TheoryAudit_2026-06.md) §10.3（pkg-08 必备：消融 1 gen_scope / Abl4 重定义 / Welch t / 5 seeds / curriculum 钩子）+ §2.3 Q7（c_hidden）+ §2.4 Q2（`use_coord_desc → randomize_order` rename + Joint enum cell）+ M8 + M12
- Roadmap: [`docs/Hyper_MuZero_v4_Roadmap.md`](../../docs/Hyper_MuZero_v4_Roadmap.md)（Stage 5 + 决策点 1-4）
- Chapter 6: §6.2.3（统计协议）+ §6.2.4（c-segment）+ §6.6（bell-curve）+ §6.7（断言 D 重定义）+ §6.9（zero-shot）
- Pkg-02 SDD: 不修改；下游补丁 `envs/resource_commons/observations.py` +3 行（spec 02 / 08 声明）
- Pkg-05 SDD: 不修改；下游补丁 `planning/mve_planner.py` +1 行 + `scripts/train_main.py` CLI 字串（spec 06 / 08 声明）
- Pkg-04 SDD: spec 02（7-API + Self-Info 严格）— eval 期 `set_context_subjective` 不接 oracle types
- Existing repo facts:
  - [`hyper_mve/training/evaluation.py`](../../hyper_mve/training/evaluation.py)（`run_eval(model, cfg)` 现状；本包扩展不替换）
  - [`hyper_mve/configs/eval_config.py`](../../hyper_mve/configs/eval_config.py)（`eval_c_grid`/`c_segments`/`zero_shot_*`/`bell_curve_type_ratios` 已 reserved，本包消费）
  - [`hyper_mve/configs/train_config.py`](../../hyper_mve/configs/train_config.py)（`use_crn`、`use_coord_desc` 现状；spec 06 rename）

---

## 📑 spec 间引用表（M6：避免事后修改）

### 期望引用矩阵（人工对账表）

| Spec | 引用的其他 spec |
|------|----------------|
| spec 01 | spec 02（zero-shot/c_hidden 评估通路）, spec 03（planner mode dispatch）, spec 04（μP 自检 EvalReport 字段）, spec 08（EvalReport schema 锁）|
| spec 02 | spec 01（EvalReport 输出位）, spec 06（regret 用 oracle_only curriculum cell）, spec 08（cfg.env.c_visible 字段声明 + 下游补丁）|
| spec 03 | spec 01（EvalReport 4-mode 标签）, spec 06（与 Abl4 cell 解耦表）, spec 08（`eval_planner_mode` 字段声明）|
| spec 04 | spec 01（μP 自检走 EvalReport）, spec 08（μP fallback 披露规则）|
| spec 05 | spec 06（ablate CLI 消费 sweep harness）, spec 07（stats 消费 RunRegistry rows）, spec 08（RunRegistry schema + 反向消费 pkg-07 REGISTRY）|
| spec 06 | spec 05（sweep cartesian 消费）, spec 03（与 4 planner mode 解耦表）, spec 08（`randomize_order`/`mve_joint_enumerate` 字段 + Pkg-05 下游补丁声明）|
| spec 07 | spec 05（消费 RunRegistry）, spec 06（ablation cells 比较）, spec 08（compare CLI 输出契约）|
| spec 08 | spec 01-07 全引用 + Pkg-07 spec 01/spec 04/spec 08 反向消费对账 + 3 下游代码补丁清单 |

### Day 8 机器可校验引用核对

```powershell
cd D:\RL\hyper_mve\sdd\pkg-08-eval-and-ablation
pwsh scripts\check_ref_matrix.ps1
# → 生成 ref_matrix.csv + 终端 [PASS]/[FAIL]
```

引用扫描正则匹配两种**包内**引用形式，并排除**跨包**引用（与 pkg-06/07 同模式）：

```powershell
$pat = '(?<!Pkg-\d{2} )spec\s+0([1-8])|\b0([1-8])-[a-z]'
#   - 裸 "spec 0X"     但排除 "Pkg-NN spec 0X"（负向后顾 (?<!Pkg-\d{2} )）
#   - 文件名 "0X-name"（spec 间在上下游表里以文件名互引）
```

**验收**：每个 spec 至少引用 README §spec 间引用表所列的其他 spec（delta 列全空）；spec 08 引用 spec 01-07 全 7 个（契约层文件特性）。Day 8 真跑校验，不写"理想态"csv。

---

## 🚦 Review Checklist（用户审阅时勾选）

- [ ] proposal.md 读完，**Why** 充分（4 条断言需可执行消融闭环 + Theory Audit §10.3 必备清单 + pkg-07 契约消费而非重发明 + c_hidden 缺口必补 + Abl4 重定义必落实 + Welch t / 5 seeds 是 Ch6.2.3 既定协议）
- [ ] design.md 读完，**Day 1 HARD GATE 10 项验收清单**全过（10 Decision-anchored，每条对应 §3.x 框架澄清 + §4 D*）
- [ ] **Eval 半 / Ablation 半分配**清晰（specs 01-04 = Eval；specs 05-08 = Ablation；spec 08 = 集成契约层）
- [ ] **`@dataclass EvalReport` 字段穷举**可对账 pkg-07 spec 08 evaluate 签名（同 schema 双消费）
- [ ] **c_hidden 处置**与 Theory Audit Q7 对齐（`cfg.env.c_visible=True` 默认 + Pkg-02 obs-mask 3 行下游补丁 + ĉ 头在 c_hidden 档为真推断）
- [ ] **regret 指标**口径认可（oracle ceiling cache + 5 seeds 平均 + cache miss 自动重跑）
- [ ] **四 planner eval mode 真值表** 可对账（mode × `use_crn` × `randomize_order` × `direct_inference`）
- [ ] **sweep harness 进程隔离** 理由认可（CUDA OOM/segfault 不传染 sibling rows）+ JSONL `RunRegistry` 并发文件锁
- [ ] **`--ablation` CLI 5 ID 分发表** 可对账 canned YAMLs 路径（`abl1` / `abl4_crn_joint` / `abl4_joint_easy_n2` / `abl6` / `abl7`）
- [ ] **`use_coord_desc → randomize_order` rename** 认可（alias-with-deprecation；Theory Audit Q2）
- [ ] **5 新 cfg 字段穷举** + 归属（`c_visible` / `randomize_order` / `mve_joint_enumerate` / `eval_planner_mode` / `eval_use_planner_direct_inference`）
- [ ] **3 下游代码补丁** 范围最小（Pkg-02 obs-mask +3 行 / Pkg-05 planner flag +1 行 / Pkg-05 CLI 字串 rename）；**Pkg-01..05 SDD 零修改**；**Pkg-07 SDD 零修改**（仅消费契约）
- [ ] **pkg-07 → pkg-08 契约对账** 5 锚点齐备（REGISTRY / 量词 canonical / N-parametric adapter / evaluate 签名 / BaselineLike Union），spec 08 §6 漂移检测章节强制 reviewer 对账
- [ ] 9 天 SDD 顺序合理（Day 1 HARD GATE 未过顺延 1 天；Day 2 与 pkg-07 spec 01/spec 08 同步）
- [ ] 出包后即可进入论文成稿阶段（thesis-resume gate：sweep harness + ablation cells live → 回到 `academic-research` 分支续写 Ch6）
