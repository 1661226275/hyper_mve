# PR: Pkg-08 (Eval + Ablation) SDD

> **状态**：Awaiting user final ack · **类型**：SDD only（不含代码实施）
> **范围**：`sdd/pkg-08-eval-and-ablation/` 11 文档 + `ref_matrix.csv` + `scripts/check_ref_matrix.ps1`
> **上游依赖**：[`pkg-07-baselines`](../pkg-07-baselines/README.md)（消费 `REGISTRY` 11 keys + `ResourceCommonsPettingZooEnv` 适配器 + 统一 `evaluate(env_fn, c_grid, episodes) -> EvalReport` 签名 + `BaselineLike Union`）

---

## Summary

- 完整 SDD 11 文档（README + proposal + design + specs 01-08 + PR_DESCRIPTION + ref_matrix.csv + scripts/check_ref_matrix.ps1）
- `unified_evaluator` **包装**（NOT 替换）`training/evaluation.py:run_eval`；`@dataclass(frozen=True) EvalReport` 持久 schema = **32 payload + 1 `schema_version` sentinel = 33 @dataclass body length**（10 payload buckets: identity 6 / headline 5 / per-c 3 / c-segment 2 / bell-curve 2 / regret 4 / planner-prior 3 / diagnostics 3 / oracle-leak 2 / belief nullable 2，spec 08 §3.2 footer 锁；canonical headline = 32 = payload subtotal，与 RunRegistry 23-key（22 schema-domain + 1 sentinel）平行表述）；internal `BaselineModel.evaluate()` 与 external `ExternalBaselineRunner.evaluate()` 双消费同 schema（pkg-07 spec 08 反向消费）
- **4 planner eval mode**（`direct_inference` / `planner_no_crn` / `planner_no_coord_desc` / `planner_full`），新 `cfg.eval.eval_planner_mode: Literal[...]="planner_full"`；与 Abl4 (training-time) cell 通过 §3.3 解耦表撕分
- **Sweep harness subprocess-per-row**（每 row 独立 `subprocess.Popen` + per-GPU `multiprocessing.Semaphore` + 优雅 SIGTERM）+ JSONL `RunRegistry` 23-key 行 schema（22 schema-domain + 1 `schema_version` sentinel；fcntl/msvcrt 文件锁 + atomic newline append）
- **5 canned ablation YAMLs**（`abl1_gen_scope` 7-cell / `abl4_crn_joint` 2×2 / `abl4_joint_easy_n2` 6²=36 enum / `abl6_fehr_schmidt` 3×3 / `abl7_curriculum` 3-cell）；CLI `--ablation <id>` thin-dispatch 到统一 sweep harness
- **Welch t-test**（`scipy.stats.ttest_ind(equal_var=False)`，via `monkeypatch` 校验 kwarg 不漂移）+ Holm-Bonferroni 校正（k≥2 才校正，单对 short-circuit）+ matplotlib `Agg` headless 论文-grade 图（bar + errorbar + sig 标注）+ `compare` CLI 三模式（`--a/--b` 对、`--methods` 多元、`--disclose` 完整 markdown 表）
- **Zero-shot 协议**（train `{0.2,0.5,0.8}` → test `{0.0,0.35,0.65,1.0}`，disjoint-union 自检）+ **c_hidden BeliefNet 探针**（`cfg.env.c_visible=False` 触发 Pkg-02 obs-mask 下游补丁 + obs-mask invariance 单测）+ **regret-vs-oracle-ceiling 缓存**（`runs/_oracle_ceilings/<config_hash>/<c>.json`；cache miss 不阻塞）+ **μP 自检 18 runs**（2 widths × 3 LRs × 3 seeds Easy；3-bucket PASS/WARN/FAIL + fallback 披露）
- **3 行为补丁 + 1 zero-behaviour threading 补丁 + 3 config-additions**（spec 08 §6 A/A'/B 3-block layout；Pkg-01..05 / Pkg-07 SDD 零改动）
- **5 新 cfg 字段穷举**（`c_visible` / `randomize_order` / `mve_joint_enumerate` / `eval_planner_mode` / `eval_use_planner_direct_inference`）+ 归属 + 默认值 + 消费 spec
- **pkg-07 → pkg-08 反向消费 5 anchors**（REGISTRY 11 keys / 量词 canonical / N-parametric adapter + 两 flag / evaluate 签名 / BaselineLike Union）+ 双向 drift detector（spec 08 §6 + meta-test `test_drift_detector_finds_all_grep_anchors`）
- 8-spec 引用矩阵机器实跑（`ref_matrix.csv` delta 全空，`scripts/check_ref_matrix.ps1` 实跑 `[PASS]`）

## Test plan

- [ ] 用户审阅 design.md §3 框架澄清五件套 + §4 D1-D10 + §5 Decision Acceptance 10 项验收清单
- [ ] 用户审阅 spec 01 §3.1 `@dataclass(frozen=True) EvalReport` 33-row body 逐字段穷举（spec 08 §3.2 canonical 10-bucket payload breakdown: identity 6 + headline 5 + per-c 3 + c-segment 2 + bell-curve 2 + regret 4 + planner-prior 3 + diagnostics 3 + oracle-leak 2 + belief nullable 2 = **32 payload** + 1 `schema_version` sentinel = **33 total @dataclass fields**；canonical headline 32 = payload subtotal）+ §7 Lock 3 `run_eval` `inspect.getsource` 比 git baseline 单测 `test_run_eval_not_modified.py`
- [ ] 用户审阅 spec 02 §2.6 zero-shot disjoint-union（train ∩ unseen = ∅）+ §4 regret 缓存 schema + §3.5 c_hidden obs-mask 3 行补丁 + §6.1 `test_c_hidden_obs_mask` + §6.6 `test_regret_cache_miss_does_not_block`（Lock 3 never-block）
- [ ] 用户审阅 spec 03 §2 四 planner eval mode 真值表 + Lock 3 一致性检查 + §7.1 `test_all_4_planner_modes_dispatch`（4 mode 全产出非 None `EvalReport`）+ 与 Abl4 (training-time) cell 解耦表
- [ ] 用户审阅 spec 04 μP 18-run plan（2 widths × 3 LRs × 3 seeds Easy）+ 3-bucket PASS/WARN/FAIL（LR-doubling 严格 / ±50% 窗口 / fallback drop 主张）+ §7.5 `test_mup_selftest_plot_renders_headless`
- [ ] 用户审阅 spec 05 §3.3 `SweepConfig` cartesian 语义（lexicographic over (variant, seed, override-idx)）+ subprocess-per-row 五点理由（§3.5）+ §4 `RegistryRow` 23-key（22 schema-domain + 1 `schema_version`）@dataclass body
- [ ] 用户审阅 spec 06 §2 5 canned YAMLs（`abl1` / `abl4_crn_joint` / `abl4_joint_easy_n2` / `abl6` / `abl7`）+ §3.3 Abl4 重定义（Joint cell 仅 Easy N=2）+ §4 `use_coord_desc → randomize_order` rename（alias-with-deprecation）+ §5.2 `mve_joint_enumerate` 单行补丁 + §7.1/7.4/7.5 CLI dispatch / Joint enum / alias warn 三项单测
- [ ] 用户审阅 spec 07 §3.1 Welch t（`equal_var=False` literal via `monkeypatch`）+ §4 Holm-Bonferroni（k≥2 校正 + 单对 short-circuit）+ §8 论文-grade plot（matplotlib `Agg`、bar + errorbar (SEM) + Welch t 标注、`matplotlib.use("Agg")` 为模块第一非注释行）+ §2 compare CLI 三模式
- [ ] 用户审阅 spec 08 §3 EvalReport 32-field byte-identical mirror + §4 RegistryRow 23-field mirror + §6 A/A'/B 3-block 下游补丁 layout + §6 5 anchors 反向消费对账 + §9.1 / §9.2 / §9.8 三 drift detector
- [ ] 执行 `ref_matrix` 校验：`powershell.exe -NoProfile -ExecutionPolicy Bypass -File sdd\pkg-08-eval-and-ablation\scripts\check_ref_matrix.ps1` → 期望 `[PASS]`
- [ ] 用户 ack 实施期 timeline（Phase D wk4-5 / Phase E wk5-6 / Phase F wk6-7；GPU 算力 ≈ 850 GPU-hr，2 GPU 并行 wall-clock ≈ 2 周）

## 文档清单

```
sdd/pkg-08-eval-and-ablation/
├── README.md                                  # 包索引 + Decision 速查表 + 18 硬约束 + 8-spec 三角矩阵 + 上游启动条件 + Review Checklist
├── proposal.md                                # Why / What Changes / Capabilities / Impact / R8-1..R8-12
├── design.md                                  # 10 Decisions D1-D10 + 框架澄清五件套 + Decision Acceptance 10 项 + mermaid
├── ref_matrix.csv                             # Day 8 机器校验产出（8-row delta 全空）
├── PR_DESCRIPTION.md                          # 本文档（Day 9 finalize artifact）
├── scripts/
│   └── check_ref_matrix.ps1                   # 引用矩阵自动校验脚本（纯 ASCII，实跑 [PASS]）
└── specs/
    ├── 01-unified-evaluator.md                # @dataclass EvalReport 32-field + unified_evaluator 包装 + 4-mode dispatcher + Self-Info strict
    ├── 02-zero-shot-and-c-hidden.md           # zero-shot disjoint-union + c_hidden obs-mask 3 行 + regret 缓存 + never-block
    ├── 03-direct-inference-toggle.md          # 4 planner eval mode + cfg.eval.eval_planner_mode literal + Abl4 解耦
    ├── 04-mup-verification.md                 # μP base-shape 18-run + 3-bucket PASS/WARN/FAIL + appendix 披露 fallback
    ├── 05-sweep-harness-and-run-registry.md   # SweepConfig cartesian + subprocess-per-row + JSONL append-only + RegistryRow 23-key
    ├── 06-ablation-cli-and-cells.md           # --ablation 5 ID canned YAML 分发 + Abl4 重定义 + use_coord_desc→randomize_order rename + mve_joint_enumerate 补丁
    ├── 07-statistics-and-comparison.md        # Welch t + Holm-Bonferroni + compare CLI 三模式 + Agg headless 论文图
    └── 08-integration-contracts.md            # EvalReport 32-field + RegistryRow 23-field 双 byte-identical mirror + 3-block 补丁 + 5-anchor drift detector
```

## Verification 结果

### 引用矩阵（Day 8 实跑 [PASS]）

| spec | actual | expected | delta |
|------|--------|----------|-------|
| 01-unified-evaluator | 02,03,04,05,06,07,08 | 02,03,04,08 | ✅ |
| 02-zero-shot-and-c-hidden | 01,03,04,05,06,07,08 | 01,06,08 | ✅ |
| 03-direct-inference-toggle | 01,02,04,05,06,07,08 | 01,06,08 | ✅ |
| 04-mup-verification | 01,02,05,07,08 | 01,08 | ✅ |
| 05-sweep-harness-and-run-registry | 01,02,03,04,06,07,08 | 06,07,08 | ✅ |
| 06-ablation-cli-and-cells | 01,02,03,05,07,08 | 03,05,08 | ✅ |
| 07-statistics-and-comparison | 01,02,03,05,06,08 | 05,06,08 | ✅ |
| 08-integration-contracts | 01,02,03,04,05,06,07 | 01,02,03,04,05,06,07 | ✅ |

**8/8 spec delta 全空** ✅（脚本判据：`expected ⊆ actual`；spec 08 引用 01-07 全 7 个契约层文件特性）

> **注**：上表 `delta` 列 markdown 渲染为 ✅ 以提高可读性；`ref_matrix.csv` 文件中该列为空字符串（CSV 字段 = `""`），两者语义等价（empty = PASS）。`check_ref_matrix.ps1` 退出码 0 即 [PASS]。

### C8-EVAL-* + C8-ABL-* 硬约束单测命名对账

#### Eval 半 — C8-EVAL-* (10 约束)

| # | 约束 | 单测 | spec § | 状态 |
|---|------|------|--------|------|
| C8-EVAL-REUSE1 | `training/evaluation.py:run_eval` 仅被包装不被替换 | `test_run_eval_not_modified.py` | spec 01 §10.1 | ✅ |
| C8-EVAL-SCHEMA1 | `@dataclass(frozen=True) EvalReport` schema 冻结；internal+external 同 schema | `test_unified_evaluator_schema.py` + `test_external_eval_contract.py` + `test_eval_report_32_field_dataclass_lock` | spec 01 §10.2/§10.3 + spec 08 §9.1 | ✅ |
| C8-EVAL-ZS1 | zero-shot train `{0.2,0.5,0.8}` → test `{0.0,0.35,0.65,1.0}` 4 unseen c 全部出现 | `test_zero_shot_grid_disjoint_union` + `test_zero_shot_grid_rejects_overlap`（负路径）| spec 02 §6.2 / §6.3 | ✅ |
| C8-EVAL-CHID1 | `cfg.env.c_visible=False` 触发 Pkg-02 obs-mask + ĉ 头真推断 | `test_c_hidden_obs_mask` | spec 02 §6.1 | ✅ |
| C8-EVAL-REGRET1 | `regret(c) = ceiling(c) - method(c)` + cache miss 不阻塞 | `test_regret_oracle_ceiling_formula` + `test_regret_cache_hit` + `test_regret_cache_miss_does_not_block`（Lock 3）| spec 02 §6.4 / §6.5 / §6.6 | ✅ |
| C8-EVAL-MODE1 | 4 planner eval mode 全可枚举（`direct_inference` / `planner_no_crn` / `planner_no_coord_desc` / `planner_full`） | `test_all_4_planner_modes_dispatch` | spec 03 §7.1 | ✅ |
| C8-EVAL-MUP1 | μP base-shape 18 runs (2 widths × 3 LRs × 3 seeds) Easy；fallback 披露 | `test_mup_selftest_plot_renders_headless` + 3-bucket assertion suite | spec 04 §7.5 + §7 | ✅ |
| C8-EVAL-SELF1 | eval 期 `set_context_subjective` 不接 oracle types | `test_self_info_strict_at_eval.py` | spec 01 §10.4 | ✅ |
| C8-EVAL-SEG1 | c-segment 三段聚合（`[0,0.3]`/`[0.3,0.7]`/`[0.7,1.0]`） | `test_c_segment_aggregation.py` | spec 01 §10.5 | ✅ |
| C8-EVAL-BELL1 | bell-curve type-ratio sweep 出现在 report | `test_bell_curve_in_report.py` | spec 01 §10.6 | ✅ |

#### Ablation 半 — C8-ABL-* (8 约束)

| # | 约束 | 单测 | spec § | 状态 |
|---|------|------|--------|------|
| C8-ABL-SWEEP1 | `SweepConfig` cartesian 正确枚举 variant × seed × override；空 override 等价裸 baseline | `test_sweep_cartesian_correct.py` + `test_config_hash_stable.py` + `test_resume_skips_completed.py` + `test_dry_run_no_subprocess.py` | spec 05 §3.3 + §9 | ✅ |
| C8-ABL-ISO1 | subprocess-per-row 进程隔离；CUDA OOM/segfault 不传染 sibling | `test_sweep_subprocess_isolation.py` + `test_gpu_semaphore.py` | spec 05 §5.1 + §5.2 + §6 + §9 | ✅ |
| C8-ABL-REG1 | `RunRegistry` JSONL append-only；row schema 冻结；并发安全（fcntl/msvcrt 文件锁）| `test_run_registry_jsonl_append_atomic.py` + `test_run_registry_row_schema_matches_spec_08.py` + `test_run_registry_23_field_jsonl_schema` | spec 05 §4 + §9 + spec 08 §9.2 | ✅ |
| C8-ABL-CLI1 | `--ablation` CLI 5 ID 全分发（`abl1` / `abl4_crn_joint` / `abl4_joint_easy_n2` / `abl6` / `abl7`）| `test_ablation_cli_dispatch_all_5` | spec 06 §7.1 | ✅ |
| C8-ABL-ABL4A | Abl4 重定义：CRN × Joint/CoordDesc 撕分；Joint cell 仅 Easy N=2 (6²=36 enum) | `test_abl4_joint_enum_easy_n2_only` | spec 06 §7.4 | ✅ |
| C8-ABL-RENAME1 | `use_coord_desc → randomize_order` rename（alias-with-deprecation）；codebase 无残留直接读者 | `test_use_coord_desc_alias_warns` + grep 自检 | spec 06 §7.5 | ✅ |
| C8-ABL-STAT1 | Welch t 用 `scipy.stats.ttest_ind(equal_var=False)`；>2 方法 Holm-Bonferroni；<5 seeds `[WARN]` 不阻塞 | `test_welch_t_against_scipy_reference` + `test_holm_bonferroni_against_statsmodels` + `test_holm_bonferroni_skipped_for_single_comparison` + `test_compare_methods_warns_on_lt_5_seeds` + `test_compare_methods_raises_on_lt_2_seeds` + `test_compare_methods_pairwise_against_reference` + `test_welch_t_raises_on_nan` | spec 07 §11.1-§11.6 | ✅ |
| C8-ABL-PLOT1 | comparison plot 论文-grade：bar + errorbar (SEM) + Welch t 标注；matplotlib backend = Agg | `test_compare_plot_renders_headless` | spec 07 §11.7 | ✅ |

**18/18 单测命名映射 PASS** ✅（10 EVAL + 8 ABL；spec 08 §9.1/§9.2/§9.8 三 spec-08-specific drift detector 额外补强）

## Downstream patches (3 行为补丁 + 1 zero-behaviour threading + 3 config-additions)

> **layout 依据**：spec 08 §6 A/A'/B 3-block；诚实分块、不混计。Pkg-01..05 SDD 零修改；Pkg-07 SDD 零修改（仅消费契约）。

### Block A — 3 行为补丁（实际代码语义变化）

| # | 文件（属上游包）| 补丁 | 由谁声明 / 消费 |
|---|----------------|------|----------------|
| A1 | `hyper_mve/envs/resource_commons/observations.py`（Pkg-02 spec02 范围）| +3 行 `if not env_cfg.c_visible: obs[..., C_OBS_INDICES] = 0.0`（c-channel mask；不改 obs shape）+ 1 行 `env_cfg` kwarg 签名 | spec 02 §5.1 + spec 08 §6 Block A |
| A2 | `hyper_mve/planning/mve_planner.py`（Pkg-05 spec03 范围）| +1 行 `if cfg.train.mve_joint_enumerate: candidates = list(itertools.product(...))` 分支（Easy N=2 only，6²=36 exhaustive enum）| spec 06 §5.2 + spec 08 §6 Block A |
| A3 | `hyper_mve/scripts/train_main.py`（Pkg-05 spec03 范围）| CLI 字串 rename `--use_coord_desc → --randomize_order`（alias 保留旧名 + `DeprecationWarning`）| spec 06 §8.3 + spec 08 §6 Block A |

### Block A' — 1 零行为 threading 补丁（带 `[no-behaviour]` 标签 + byte-identical 回归单测）

| # | 文件 | 补丁 | 由谁声明 |
|---|------|------|--------|
| A'1 | `hyper_mve/envs/resource_commons/env.py`（Pkg-02 spec02 范围）| +2 行 `env_cfg=self.cfg` kwarg 透传到 `build_joint_observation` 的两处 call-site（`reset` / `step`）；零行为变化；`test_env_py_call_site_kwarg_no_behaviour_change` 断言 byte-identical | spec 02 §5.1b `[no-behaviour]` + spec 08 §6 Block A' |

### Block B — 3 config-additions（消费态 +5 字段加到 3 个现有 config dataclass；不改语义）

| # | 文件 | 补丁 | 字段（共 5）|
|---|------|------|------------|
| B1 | `hyper_mve/configs/env_config.py` | +1 字段 | `c_visible: bool = True` |
| B2 | `hyper_mve/configs/train_config.py` | +2 字段 + `@property` shim | `randomize_order: bool`（alias of `use_coord_desc`）+ `mve_joint_enumerate: bool = False` |
| B3 | `hyper_mve/configs/eval_config.py` | +2 字段 | `eval_planner_mode: Literal[...] = "planner_full"` + `eval_use_planner_direct_inference: bool = False` |

**合计**：3 行为补丁 + 1 零行为 threading + 3 dataclass 字段增量；总修改 5 个文件 7 处微改动。诚实分块，不混计 Block A (语义变化) 与 Block A' (字节级零变化)。

## 5 cfg 字段穷举

> 依据 design D10；本包**仅声明字段**，Pkg-01 spec05 同步消费态登记。Pkg-01 SDD 零修改。

| 字段 | 类型 | 默认值 | 归属 | 消费 spec | 用途 |
|------|------|--------|------|----------|------|
| `cfg.env.c_visible` | `bool` | `True` | `EnvConfig` | spec 02 | c_hidden 评估通路（D4）|
| `cfg.train.randomize_order` | `bool` | mirrors `use_coord_desc` (alias-with-deprecation) | `TrainConfig` | spec 06 | Abl4 重定义 + Theory Audit Q2 rename |
| `cfg.train.mve_joint_enumerate` | `bool` | `False` | `TrainConfig` | spec 06 | Abl4 Joint cell, Easy N=2 only（Pkg-05 1 行补丁 A2）|
| `cfg.eval.eval_planner_mode` | `Literal["direct_inference","planner_no_crn","planner_no_coord_desc","planner_full"]` | `"planner_full"` | `EvalConfig` | spec 03 | 四 planner eval mode dispatch（D7）|
| `cfg.eval.eval_use_planner_direct_inference` | `bool` | `False` | `EvalConfig` | spec 03 | mode 短路开关（与 `eval_planner_mode="direct_inference"` 等价）|

`alias-with-deprecation` 协议（针对 `use_coord_desc → randomize_order`）：`randomize_order` 是 canonical 名；`use_coord_desc` 保留为 `@property` shim，读时 `warnings.warn(DeprecationWarning)` 并返回 `randomize_order`；codebase grep 自检无残留直接读者（`test_use_coord_desc_alias_warns`，spec 06 §7.5）；旧 cfg 反序列化 round-trip 兼容（R8-6 缓解）。

## Implementation timeline

> 实施期 2–3 周，从 pkg-07 落码完成（Phase C wk3-4 收尾）起 overlap 启动。详见 Plan File 绝对路径 `C:\Users\zhengwenbo01\.claude\plans\after-some-thought-i-iridescent-rain.md` §"Code-implementation" Phase D-F。

| Phase | 周次 | 范围 | 产出 |
|-------|------|------|------|
| **Phase D** | wk 4-5 | spec 01 / spec 05 落码 | `hyper_mve/eval/unified_evaluator.py` + `eval_report.py` + `hyper_mve/experiments/sweep.py` + `run_registry.py` + per-GPU semaphore + JSONL append-only RunRegistry |
| **Phase E** | wk 5-6 | spec 02 / spec 03 / spec 06 落码 + 3 下游补丁 + 1 zero-behaviour threading 补丁 | zero-shot 通路 + c_hidden + Pkg-02 obs-mask 3 行 + regret 缓存 + 4 planner mode dispatch + `mve_joint_enumerate` 1 行 + `randomize_order` rename + 5 canned YAML（abl1/abl4_crn_joint/abl4_joint_easy_n2/abl6/abl7）|
| **Phase F** | wk 6-7 | spec 04 / spec 07 / spec 08 落码 + 主表 + 论文图 | μP 18-run 自检 + Welch t + Holm-Bonferroni + `compare` CLI 三模式 + matplotlib Agg 论文图 + 主表（hyper + 5 internal + 3 Tier-1 + MAMBA-if-sourced × 5 seeds × 2 presets）+ 4 ablation 表 + zero-shot 表 + 7 drift detector（spec 08 §9）|

**GPU 算力**：≈ **850 GPU-hr**（下限 ≈ 700 砍 Tier-2 MAMBA + Abl6 部分单元；详见 proposal §4.2）；2 GPU 并行（per-GPU semaphore，spec 05 §5.2）→ wall-clock ≈ 350 hr ≈ **2 周**。**不含** pkg-07 Tier-1 LR sweep；**包括** oracle ceiling cache 预算 + zero-shot grid + μP 自检 18 runs + Joint-N=2 enum + dual-mode planner eval。

## Risks (R8-1..R8-12)

| # | 风险 | 缓解 |
|---|------|------|
| R8-1 | μP base-shape LR-doubling 自检失败（width × 2 时最优 LR 不严格 ÷ 2）| spec 04 fallback：3-bucket PASS/WARN/FAIL；FAIL → 论文 drop μP 主张 + appendix 披露；`test_mup_selftest_plot_renders_headless` + 3-bucket assertion suite |
| R8-2 | subprocess sweep CUDA 资源未释放（driver-level leak）→ 第 N row OOM | spec 05 §5.1 per-row subprocess 强制 GC + `torch.cuda.empty_cache` + `os._exit(0)` 退出；100-row 内存稳定性单测 |
| R8-3 | regret cache 因 config 哈希算法不稳定（dict ordering）造成 cache miss 风暴 | spec 02 §4 `json.dumps(sort_keys=True)` + `blake2b(16)`；cache key 在 cfg round-trip 后稳定 |
| R8-4 | Joint-enum 36-cell 只在 Easy N=2 跑 → Medium/Hard 上断言 D 无独立证据 | spec 06 §3 narrow-scope 段落：Easy N=2 是 Joint 可枚举的唯一档；Medium/Hard CoordDesc 上界由 Easy enum 间接覆盖 |
| R8-5 | n=5 seeds 时 Welch t power 不足（中等效应 d ≈ 0.5 检不出）| Ch6.2.3 协议：5 seeds 最低；关键 contrast 加跑 10 seeds；spec 07 §2 power 估算表 |
| R8-6 | `use_coord_desc → randomize_order` rename 碰撞历史 ckpt cfg 序列化字段名 | spec 06 §4 alias-with-deprecation：旧 cfg 反序列化 `use_coord_desc` 映射到 `randomize_order`；`test_use_coord_desc_alias_warns` round-trip 覆盖 |
| R8-7 | c_hidden obs-mask 下游补丁破坏 Pkg-02 现有单测 | spec 08 §6 Block A 强制 `pytest tests/envs/resource_commons/` 全过；CI `--strict-markers`；A' Block 1 零行为 threading 补丁带 byte-identical 回归单测 `test_env_py_call_site_kwarg_no_behaviour_change` |
| R8-8 | RunRegistry JSONL 并发写竞争 → row 损坏（半行）| spec 05 §4.3 fcntl/msvcrt 跨平台文件锁 + `_lock_exclusive → write → fsync → _unlock` 原子周期；`test_run_registry_jsonl_append_atomic`（16-thread × 100-row stress）< 1s |
| R8-9 | `EvalReport` schema 漂移（实施期某 spec 加字段未同步 spec 08）| spec 08 §9.1 `test_eval_report_32_field_dataclass_lock` 用 `dataclasses.fields(EvalReport)` 对账 spec 08 §3 byte-identical mirror；spec 01 §3.1 mother doc + spec 08 §3 镜像；frozen=True 强 read-only |
| R8-10 | matplotlib 在无 X server CI 环境下 crash | spec 07 §8 `matplotlib.use("Agg")` 强制为 `compare.py` 首非注释行；spec 04 spec 07 共用 Agg；`test_compare_plot_renders_headless` 校验 `matplotlib.get_backend().lower() == "agg"` |
| R8-11 | 4 planner eval mode `Literal[...]` 未来需扩展（如加 `planner_no_qstd_floor`）| spec 03 §3 扩展协议：Literal 只增不删；新 mode 同步修 spec 03 + spec 08 + 默认值不变 |
| R8-12 | Oracle ceiling cache miss 触发自动 `oracle_only` 重跑 → 隐式 GPU 预算膨胀 | spec 02 §4.7 Lock 3 cache miss never-block：print `[INFO] cache miss, will spawn oracle_only run (~5 GPU-hr)`；`cfg.eval.regret_cache_strict=True` 可拒绝自动重跑；MAMBA-not-sourced row 同模式产出 `"skipped: NotImplementedError"` 不阻塞主表稳定性 |

## pkg-07 ↔ pkg-08 5-anchor reverse-consumption

> 镜像 README §"🔁 pkg-07 → pkg-08 契约对账"；任一锚点漂移 → 触发 pkg-08 spec 01 / spec 05 / spec 08 同步修订；Day 8 `check_ref_matrix.ps1` 真跑校验，spec 08 §9.8 `test_drift_detector_finds_all_grep_anchors` meta-test 强制 reviewer 对账。

| # | pkg-07 source（上游契约）| pkg-08 consumer spec § | drift handling |
|---|--------------------------|------------------------|----------------|
| 1 | pkg-07 spec 01 §2.1 `REGISTRY` 11 keys（只读 `MappingProxyType`；5 internal + 3 Tier-1 + MAMBA + 2 stubs）| pkg-08 spec 05 §3（sweep enumeration 起点）| `check_ref_matrix.ps1` grep `REGISTRY` 双向出现；spec 08 §6 reviewer 对账栏强制勾确认；stubs 在 sweep 中产生 `"skipped: NotImplementedError"` row |
| 2 | pkg-07 spec 01 §2.3 量词 canonical（11 keys）| pkg-08 spec 05 §3 + spec 08 §5 | 同 (1)；数量漂移直接触发 ref_matrix delta |
| 3 | pkg-07 spec 04 N-parametric adapter + 两 flag 信息门控（`oracle_mode=False` AND `eval_info_mode=False` 默认；4 leak-surface 全 gated）| pkg-08 spec 01 §4（external runner eval 通路）+ pkg-08 spec 02 §3（c_hidden 与 `oracle_mode` 正交）| spec 02 单测 `test_c_hidden_orthogonal_to_oracle_mode`；spec 08 §6 漂移检测 |
| 4 | pkg-07 spec 08 `evaluate(env_fn, c_grid, episodes) -> EvalReport` 签名锁 | pkg-08 spec 01 §3 EvalReport schema + spec 08 §3 字段穷举 | spec 08 §9.1 `test_eval_report_32_field_dataclass_lock` 与 pkg-07 spec 08 `test_external_runner_evaluate_returns_eval_report` 共用同一 `EvalReport` import |
| 5 | pkg-07 spec 08 `BaselineLike = Union[BaselineModel, ExternalBaselineRunner]` Union | pkg-08 spec 05 §4（sweep harness `RowResult.baseline_kind` 字段区分 internal/external）| spec 05 单测 `test_row_result_records_baseline_kind`；spec 08 §6 reviewer 对账栏 |

**双向 drift detector**：spec 08 §9.8 `test_drift_detector_finds_all_grep_anchors` meta-test grep 仓库内所有"pkg-07 spec 0X §Y"引用并对账 pkg-08 spec 01 / spec 05 / spec 08 内的反向消费段，任一遗漏即 fail。

## Review Checklist（用户最终 ack 时勾选）

- [ ] proposal.md 读完，**Why** 充分（4 条断言需可执行消融闭环 + Theory Audit §10.3 6 项必备清单 + pkg-07 契约消费而非重发明 + c_hidden 缺口必补 + Abl4 重定义必落实 + Welch t / 5 seeds 是 Ch6.2.3 既定协议 + R8-1..R8-12 逐项可接受）
- [ ] design.md 读完，**Day 1 HARD GATE 10 项 Decision Acceptance** 全过（每条对应 §3 框架澄清 + §4 D1-D10）
- [ ] **Eval 半 / Ablation 半分配**清晰（specs 01-04 = Eval；specs 05-08 = Ablation；spec 08 = 集成契约层）
- [ ] **`@dataclass(frozen=True) EvalReport` 32-field 穷举**（spec 01 §3.1 mother doc + spec 08 §3 byte-identical mirror）；hashable + JSON-serializable；internal + external 双消费同 schema（pkg-07 spec 08 反向消费）
- [ ] **`RegistryRow` 23-field 穷举**（spec 05 §4 mother doc + spec 08 §4 byte-identical mirror；22 schema-domain + 1 `schema_version` sentinel）；JSONL append-only + fcntl/msvcrt 跨平台文件锁
- [ ] **四 planner eval mode 真值表** 可对账（spec 03 §2 + design §3.3；与 Abl4 (training-time) cell 解耦表清晰）
- [ ] **sweep harness 进程隔离** 五点理由认可（spec 05 §5 + design §3.5：CUDA OOM 不传染 / GPU 真清理 / 配置独立 / per-GPU semaphore / 优雅中断）
- [ ] **`--ablation` CLI 5 ID 分发表** 可对账 canned YAMLs 路径（`abl1` / `abl4_crn_joint` / `abl4_joint_easy_n2` / `abl6` / `abl7`）
- [ ] **`use_coord_desc → randomize_order` rename** 认可（alias-with-deprecation；Theory Audit Q2；spec 06 §4）
- [ ] **5 新 cfg 字段穷举** + 归属（`c_visible` / `randomize_order` / `mve_joint_enumerate` / `eval_planner_mode` / `eval_use_planner_direct_inference`）；alias-with-deprecation 协议；Pkg-01 spec05 同步消费态
- [ ] **3 行为补丁 + 1 零行为 threading + 3 dataclass 字段增量**（spec 08 §6 A/A'/B 3-block layout 诚实分块；Pkg-01..05 / Pkg-07 SDD 零修改）
- [ ] **pkg-07 → pkg-08 5-anchor 反向消费对账** 齐备（REGISTRY 11 keys / 量词 canonical / N-parametric adapter + 两 flag / evaluate 签名 / BaselineLike Union）；spec 08 §6 漂移检测 + §9.8 meta-test
- [ ] **`ref_matrix.csv` 8 行 delta 全空** + Day 8 `check_ref_matrix.ps1` 实跑 `[PASS]`
- [ ] **18/18 单测命名映射 PASS**（10 C8-EVAL-* + 8 C8-ABL-*）+ spec 08 §9.1/§9.2/§9.8 三 spec-08-specific drift detector
- [ ] 9 天 SDD 顺序合理（Day 1 HARD GATE 未过顺延 1 天；Day 2 与 pkg-07 spec 01/spec 08 同步签名锁）；实施期 timeline（Phase D-F wk4-7）+ GPU 算力 ≈ 850 GPU-hr 可接受
- [ ] 出包后即可进入论文成稿阶段（thesis-resume gate：sweep harness + ablation cells live → 回到 `academic-research` 分支续写 Ch6 主表叙事）

---

Generated as part of Hyper-MuZero v4 SDD writing workflow · pkg-08 Day 9 finalize artifact
