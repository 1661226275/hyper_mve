# Pkg-08 Proposal: Eval + Ablation Framework — 消融闭环 + 统计 + 论文表生产线

> 配套阅读：[`README.md`](./README.md) · [`design.md`](./design.md)（**先读 proposal 再读 design**）
> **上游契约**：[`pkg-07-baselines/proposal.md`](../pkg-07-baselines/proposal.md)（消费 `REGISTRY` + `ResourceCommonsPettingZooEnv` 适配器 + 统一 `evaluate(env_fn, c_grid, episodes) -> EvalReport` 签名）
> **Theory Audit 责任**：[`docs/Review_v4_TheoryAudit_2026-06.md`](../../docs/Review_v4_TheoryAudit_2026-06.md) §10.3 pkg-08 备忘（μP 自检 / 零样本 / c_hidden / planner direct-inference / Abl4 Joint cell / Fehr-Schmidt 3×3 / sweep harness / Welch t 5 seeds）

---

## 1. Why（为什么需要本包）

### 1.1 闭合实验环：pkg-07 内部+外部基线给齐了主体，但论文还不能生成 Methods 表

Pkg-07 SDD 在 4-5 周实施期内将给齐：5 internal variant（input_wide/deep / ma_muzero / no_belief / rewardhead_explicit_type）+ 3 Tier-1 external（MAPPO / QMIX / MA-MuZero-GH）+ Tier-2 MAMBA（若 sourced）+ 2 stub（MARIE / GA）。这 11 个 model class 各自能 `train()` 出 checkpoint，但**论文要的是 Methods 表**，需要：

1. 一个 evaluation framework 把 11 个 checkpoint 都跑过同样的 c-grid、同样的 episode 数、同样的 seed 集合，输出**同 schema** 的报告；
2. 一个 sweep harness 把"variant × seed × override"乘起来，逐 row 落盘并支持复跑；
3. 一个 ablation CLI 把 Theory Audit §10.3 列出的 4 个消融 cell（gen_scope 7-cell / CRN-Joint / Fehr-Schmidt 3×3 / curriculum）一键分发；
4. 一个 statistical layer 把 5 seed 结果做 Welch t-test + Holm-Bonferroni 校正，输出 markdown 表 + bar+errorbar 图。

没有这四件套，pkg-07 落地的 11 个 model **只是 11 个 checkpoint 文件夹**，不是论文。Pkg-08 的全部价值是把这 11 个 checkpoint 翻译成"主表 + 4 ablation 表 + zero-shot 表 + μP 自检图"，**直接进 Ch6**。

### 1.2 Theory Audit §10.3 强制清单：6 项必备实验，缺一不可

Review_v4_TheoryAudit_2026-06.md §10.3 给出了 pkg-08 必须输出的清单（与 §10.1 / §10.2 一致）：

| # | Theory Audit §10.3 必备 | 失败模式（如缺）| 落点 spec |
|---|------------------------|--------------|---------|
| 1 | **μP base-shape LR-doubling 验证**（2 widths × 3 LRs × 3 seeds Easy）| 论文 §4.7.x μP 主张无证据，审稿人质疑 LR 选择是 cherry-pick | spec 04 |
| 2 | **零样本协议**（train `{0.2,0.5,0.8}` → test `{0.0,0.35,0.65,1.0}` 4 个 unseen c）| 断言 B′ 的"per-context 容量真在最优生成范围内"无法证实，泛化 headline 数缺失 | spec 02 |
| 3 | **c_hidden 模式**（`cfg.env.c_visible=False` + Pkg-02 obs-mask 3 行下游补丁）| Q7 处置悬空：BeliefNet ĉ 头是否真在做"信念推断"无法验证，可能只是把 obs 里的 c 通道复制了一份 | spec 02 |
| 4 | **planner direct-inference toggle**（π_mve 直推 vs planner full）| Abl4 Joint cell 与 eval-mode 撕不开，断言 D 的"planner 双技术联合解 SNR 崩塌"无法在 eval 期独立检验 | spec 03 |
| 5 | **Abl4 重定义**（CRN × Joint/CoordDesc 真撕分；Joint cell 仅 Easy N=2，6²=36 enum）| Q2 + M8 处置悬空：原 Abl4 把 CRN 与 coord-descent 混在一个开关里，撕不出独立贡献；rename `use_coord_desc → randomize_order` 否则 codebase 命名与新解释自相矛盾 | spec 06 |
| 6 | **Welch t-test 5 seeds + Holm-Bonferroni**（Ch6.2.3 协议）| Methods 表数字"差不多"无显著性论证，审稿人直接拒 | spec 07 |

附加 2 项 §10.3 收敛性 / 公平性钩子：

- **Fehr-Schmidt 3×3 扫描**（α/β 公平度 × 不平等敏感度，Theory Audit §6.2 锚点）→ spec 06 abl6 cell；
- **Curriculum 3-cell**（oracle_only / mixed / infer_only，Theory Audit §10.3 收尾建议）→ spec 06 abl7 cell。

每一项 §10.3 备忘**逐字落到** 8 个 spec 之一，无一遗漏。详见 spec 08 §6 的 §10.3 备忘 → spec 反向引用对账。

### 1.3 两半分割是包结构的自然分解：Eval 半 vs Ablation 半

继承 pkg-04/05/06/07 的 8-spec 对称模式，但 pkg-08 内部有一条天然分割线：

- **Eval 半（spec 01-04）**：**method-agnostic**。一个 evaluator 必须能跑 hyper 也能跑 MAPPO，不能假设有 `set_context_subjective` 这样的 internal API。所以 Eval 半的接口（`EvalReport` schema、planner_mode literal、zero-shot c-grid、μP 自检参数）必须只依赖 pkg-07 提供的统一 `evaluate(env_fn, c_grid, episodes) -> EvalReport` 签名，**不**触碰 model 内部。
- **Ablation 半（spec 05-08）**：**method-specific**。Abl4 改的是 `cfg.train.use_crn` / `cfg.train.randomize_order` / `cfg.train.mve_joint_enumerate`（hyper-only 训练时开关）；Abl6 Fehr-Schmidt 改的是 reward 函数参数；Abl7 curriculum 改的是 trainer stage 比例。这些都是 hyper（或 hyper 的 ablation variant）专属。

两半在 spec 08（集成契约）合流：spec 08 既冻结 Eval 半的 `EvalReport` schema，也冻结 Ablation 半的 `RunRegistry` row schema，并声明 5 新 cfg 字段 + 3 下游代码补丁。这种结构让 reviewer 审 Eval 半时可以**完全不思考** ablation 细节，反之亦然。

### 1.4 `EvalReport` 统一 schema 是 pkg-07 与 pkg-08 之间唯一的载荷契约

Pkg-07 spec 08 已锁定：所有 baseline（internal 7-API 模型 + external 自带 trainer runner）的 `evaluate(env_fn, c_grid, episodes)` 必须返回**同一个** `EvalReport` 实例。Pkg-08 的全部对接表面就是这一个 dataclass：

```python
@dataclass(frozen=True)
class EvalReport:
    method: str                                          # variant name, e.g. "hyper", "external_mappo"
    seed: int                                            # which seed produced this report
    per_c_return: Mapping[float, ReturnStats]            # c → (mean, sem, n_episodes)
    planner_prior_gap: Mapping[float, float]             # c → (return_planner - return_prior); hyper-only nonzero
    zero_shot_unseen: Mapping[float, ReturnStats]        # c ∈ unseen_c → return
    c_segment_returns: Mapping[tuple[float,float], ReturnStats]  # (lo, hi) → return on c uniform in [lo,hi]
    bell_curve_returns: Mapping[tuple[int,int], ReturnStats]     # (n_alpha, n_beta) → return
    regret_vs_ceiling: Mapping[float, float]             # c → ceiling(c) - method(c); reads runs/_oracle_ceilings cache
    planner_mode: Literal["direct_inference","planner_no_crn","planner_no_coord_desc","planner_full"]
    wall_clock_seconds: float
    n_params: int
```

字段冻结（spec 08 §3 穷举）。Internal `BaselineModel.evaluate()` 与 external `ExternalBaselineRunner.evaluate()` 都返回同一实例 → sweep harness 可以无差别拼接 → stats.py 可以无差别 t-test → compare CLI 可以无差别画图。这一条 schema 是整个论文 Methods 表的"单点失败"，因此必须在 design.md D3 + spec 08 §3 双重锁定。

### 1.5 三处下游代码补丁——诚实地标出，不偷偷合并

Pkg-08 要落 c_hidden 模式、Abl4 Joint enum cell、`use_coord_desc → randomize_order` rename，这三件事都不能在 pkg-08 自己的 `eval/` 或 `experiments/` 目录内完成——它们必须改：

| 文件（属上游包） | 改动 | 必要性 |
|----------------|------|--------|
| `hyper_mve/envs/resource_commons/observations.py`（Pkg-02 spec02 范围）| `if not cfg.env.c_visible: obs[:, c_channel] = 0`（3 行）| c_hidden 在 evaluator 侧 mask 走不通（obs 已 build 完），必须在 obs build 时切断 |
| `hyper_mve/planning/mve_planner.py`（Pkg-05 spec03 范围）| `if cfg.train.mve_joint_enumerate: candidates = list(itertools.product(...))`（1 行 + 上下文）| Easy N=2 时 6²=36 联合动作可枚举，CoordDesc 在这一档可被严格上界覆盖（Abl4 Joint cell 的可信性） |
| `hyper_mve/scripts/train_main.py`（Pkg-05 spec03 范围）| `--use_coord_desc` CLI 字串保留 alias + `DeprecationWarning`，新 `--randomize_order` 字串生效 | rename 必须落到 CLI 层否则用户脚本不会被迁移 |

加上 `hyper_mve/configs/env_config.py`（Pkg-01 spec05 范围）+`hyper_mve/configs/train_config.py` + `hyper_mve/configs/eval_config.py` 的 cfg 字段加入（5 字段），合共**5 个文件 6 处微改动**。所有改动都属于"消费态变更"，不动 Pkg-01/02/05 SDD，但 pkg-08 spec 08 必须把它们**逐一列出**作为对外硬声明——否则后人 grep `c_visible` 找不到归属。**Pkg-01..05 SDD 零修改**，**Pkg-07 SDD 零修改**（仅消费契约），这一点 spec 08 §5 强制 reviewer 确认。

---

## 2. What Changes（具体改动）

> **本包只写 SDD 文档**，不写实现代码。以下"改动"描述的是 SDD 锁定的**待实施接口**，供 pkg-08 实施期 2-3 周（Phase D-F）落码与未来 Ch6 主表生产消费。

### 2.1 新增（`hyper_mve/eval/` 目录 — Eval 半）

```
hyper_mve/eval/
├── __init__.py                          # re-export EvalReport + UnifiedEvaluator
├── unified_evaluator.py                 # 包装 training/evaluation.py:run_eval（扩展不替换；C8-EVAL-REUSE1）
├── eval_report.py                       # @dataclass(frozen=True) EvalReport（spec 08 §3 schema）
├── zero_shot.py                         # train {0.2,0.5,0.8} → test {0.0,0.35,0.65,1.0}，强制 4 个 unseen c
├── c_hidden.py                          # cfg.env.c_visible=False 评估通路（消费 Pkg-02 obs-mask 补丁）
├── regret.py                            # oracle ceiling cache + regret(c) 计算
├── planner_modes.py                     # direct_inference / planner_no_crn / planner_no_coord_desc / planner_full
└── mup_verification.py                  # μP base-shape LR-doubling 18-run 自检（spec 04）
```

**关键设计**：`unified_evaluator.py` **包装**现有 `training/evaluation.py:run_eval`（扩展不替换；C8-EVAL-REUSE1）。理由：

1. `run_eval` 现有的 dual-mode（prior / planner）+ 6 TB tag 是 v4-opt 2026-06 的成果，TB 历史轨迹连续性必须保护；
2. 替换会让"In-training periodic eval"与"Pkg-08 full-suite eval"分叉，未来 bug 修两遍；
3. UnifiedEvaluator 只在 run_eval 输出基础上补：c-segment 聚合、bell-curve type-ratio 循环、zero-shot unseen-c 块、regret 计算、planner_mode dispatch。本质是"装饰器"而非"替代品"。

### 2.2 新增（`hyper_mve/experiments/` 目录 — Ablation 半）

```
hyper_mve/experiments/
├── __init__.py
├── sweep.py                             # SweepConfig + subprocess-per-row + per-GPU semaphore
├── run_registry.py                      # JSONL append-only + 并发文件锁（fcntl/msvcrt）
├── ablate.py                            # __main__ entry: `--ablation <id>` 分发到 ablations/*.yaml
├── stats.py                             # Welch t-test + Holm-Bonferroni 校正
├── compare.py                           # __main__ entry: `--a runs/<a>.csv --b runs/<b>.csv` → md + plot
└── ablations/
    ├── abl1_gen_scope.yaml              # 7-cell gen_scope（Theory Audit §2.2）
    ├── abl4_crn_joint.yaml              # CRN × Joint/CoordDesc 2×2（重定义后）
    ├── abl4_joint_easy_n2.yaml          # Easy N=2 only，6²=36 exhaustive enum cell
    ├── abl6_fehr_schmidt.yaml           # Fehr-Schmidt 3×3 扫描
    └── abl7_curriculum.yaml             # oracle_only / mixed / infer_only
```

**关键设计**：

- **subprocess-per-row 进程隔离**（C8-ABL-ISO1）：每个 sweep row 是一个独立 `subprocess.Popen` 调用 `python -m hyper_mve.scripts.train_main --json-config <override>`。理由：(a) CUDA OOM / segfault 在 subprocess 内不传染 sibling row；(b) 训练态的 `torch._dynamo` / `torch.cuda` 全局状态被 OS 回收，避免 memory leak 累积；(c) per-GPU semaphore（`asyncio.Semaphore(num_gpus)`）实现简单的并发上限。
- **`RunRegistry` JSONL append-only**（C8-ABL-REG1）：`runs/registry.jsonl` 每行一个 row（row schema 在 spec 08 §4 冻结）。append-only + 文件锁（POSIX `fcntl.flock` / Windows `msvcrt.locking`）→ 并发 sweep 不会丢 row。读取时整文件 line-by-line parse，没有索引（n=数百 row 足够快）。
- **`--ablation <id>` canned YAML 分发**（C8-ABL-CLI1）：CLI 只做 dispatch（5 ID → 5 YAML 路径），YAML 描述 sweep cartesian（variant list × seed list × override list）；新增 ablation cell 只改 YAML 不改 CLI。

### 2.3 `EvalReport` schema（spec 08 §3 锁定的稳定字段列表）

```python
@dataclass(frozen=True)
class ReturnStats:
    mean: float
    sem: float
    n_episodes: int

@dataclass(frozen=True)
class EvalReport:
    method: str
    seed: int
    per_c_return: Mapping[float, ReturnStats]
    planner_prior_gap: Mapping[float, float]
    zero_shot_unseen: Mapping[float, ReturnStats]
    c_segment_returns: Mapping[tuple[float, float], ReturnStats]
    bell_curve_returns: Mapping[tuple[int, int], ReturnStats]
    regret_vs_ceiling: Mapping[float, float]
    planner_mode: Literal[
        "direct_inference",
        "planner_no_crn",
        "planner_no_coord_desc",
        "planner_full",
    ]
    wall_clock_seconds: float
    n_params: int
```

**frozen=True** 防止下游 in-place 修改造成跨 row 污染。所有字段**必填**，缺失字段（如 external runner 没有 planner_prior_gap）应填 `{}` 而非 `None`，类型严格度由 mypy 在 CI 强制。

### 2.4 5 新 cfg 字段穷举（Pkg-01 spec05 同步消费态声明，本包不改上游）

```
cfg.env.c_visible: bool = True
    # spec 02 消费；False 时 Pkg-02 obs-mask 下游补丁触发，c 通道清零
cfg.train.randomize_order: bool = (mirrors cfg.train.use_coord_desc)
    # spec 06 消费；rename from use_coord_desc（Theory Audit Q2），alias-with-deprecation
cfg.train.mve_joint_enumerate: bool = False
    # spec 06 消费；True 时 Pkg-05 mve_planner.py 走联合枚举（Abl4 Joint cell, Easy N=2 only）
cfg.eval.eval_planner_mode: Literal[
    "direct_inference","planner_no_crn","planner_no_coord_desc","planner_full"
] = "planner_full"
    # spec 03 消费；统一 4-mode dispatch
cfg.eval.eval_use_planner_direct_inference: bool = False
    # spec 03 消费；mode 短路开关，与 eval_planner_mode="direct_inference" 等价（兼容用）
```

字段归属、默认值、消费 spec 完整列在 spec 08 §2。**Pkg-01 spec05 同步更新消费态声明**——本包不动 Pkg-01 SDD，但 Pkg-01 spec05 必须知道这 5 个字段存在（design.md D10 锁）。

### 2.5 下游代码补丁（3 处微补丁，spec 08 §5 显式声明）

| 文件 | 补丁 | 由谁消费 | spec 08 §5 行号 |
|------|------|---------|-----------------|
| `hyper_mve/envs/resource_commons/observations.py` | +3 行 `if not cfg.env.c_visible: obs[:, c_channel] = 0` | Pkg-08 spec 02（c_hidden 模式）| §5.1 |
| `hyper_mve/planning/mve_planner.py` | +1 行 `if cfg.train.mve_joint_enumerate: candidates = list(itertools.product(...))` 分支 | Pkg-08 spec 06（Abl4 Joint cell）| §5.2 |
| `hyper_mve/scripts/train_main.py` | CLI 字串 `--use_coord_desc` 保留 alias + `DeprecationWarning`，新 `--randomize_order` 生效 | Pkg-08 spec 06 + 通用 rename | §5.3 |

**Pkg-02 / Pkg-05 SDD 零修改**——pkg-08 spec 08 把这 3 处补丁作为**对外硬契约**列出，实施期由 pkg-08 编码者打补丁，但 SDD 文档层面 pkg-02 / pkg-05 不变。这与 pkg-07 处理 baseline-internal 下游补丁同模式。

---

## 3. Capabilities（本包带来的能力）

### 3.1 Methods 主表生成：9 列 × 主表指标 × 5 seeds × 2 presets

`python -m hyper_mve.experiments.ablate --ablation main_table` 一键产出：

| 列 | 行（method）| 数据来源 |
|----|------------|---------|
| return-Easy / return-Medium | hyper | `EvalReport.per_c_return` 在 c-grid 上的平均 |
| 同上 | 5 internal（pkg-07 §2.1.3）| `BaselineModel.evaluate() -> EvalReport` |
| 同上 | 3 Tier-1 external（pkg-07 §2.1.4）| `ExternalBaselineRunner.evaluate() -> EvalReport` |
| 同上 | MAMBA-if-sourced（pkg-07 §2.1.4 Tier-2）| 同上 |
| params / walltime | 全 9 行 | `EvalReport.n_params` / `wall_clock_seconds` |
| regret(c) | 全 9 行 | `EvalReport.regret_vs_ceiling`（cache miss 触发 oracle_only 重跑）|

5 seeds × 2 presets（Easy N=2 / Medium N=4）= 90 sweep rows for hyper + 90 × 8 = 720 rows for others = ~810 rows 主表（不含 MAMBA / stubs）。Welch t 在 spec 07 算 hyper vs 每个 baseline 的显著性。

### 3.2 零样本泛化 gap：论文 headline 数

`spec 02` 锁定 train `{0.2, 0.5, 0.8}` → test `{0.0, 0.35, 0.65, 1.0}`。`EvalReport.zero_shot_unseen` 强制 4 个 unseen c 全部出现（C8-EVAL-ZS1）。论文 §6.9 输出：

> "在 train 集 c ∈ {0.2, 0.5, 0.8} 上训练后，hyper 在 4 个 unseen c ∈ {0, 0.35, 0.65, 1.0} 上的平均 return = X ± Y，与 train 集均值差 ΔR = Z（gap = Z / X = W%）"

这一个数字是断言 B′"per-context θ 给 belief 组合专用容量"的最直接证据。

### 3.3 c_hidden BeliefNet 质量探针

`cfg.env.c_visible=False`（spec 02 + Pkg-02 obs-mask 3 行补丁）→ obs 里 c 通道被 mask → BeliefNet 的 ĉ 头**只能从 obs+history 推断 c**，无法照抄 obs 通道。同 c-grid 下 c_visible=True 与 False 的 ĉ 预测 MSE 对比 → "ĉ 头真在做推断"的 Theory Audit Q7 处置。

### 3.4 Joint-enum vs CoordDesc 保真度检验

Easy N=2 时联合动作空间 6² = 36（A=6 含 NOOP）。`cfg.train.mve_joint_enumerate=True`（spec 06 Pkg-05 1 行补丁）→ MVE planner 真做 36-candidate exhaustive enum。CoordDesc（默认 50 个 sample × N 个 agent）在同一 setting 下的 return → 撕分"CoordDesc 是否在 SNR 崩塌前就把 Joint 最优捡了"。这是 Theory Audit M8 的精确处置：Joint cell **仅** Easy N=2 跑（Medium N=4 联合 6⁴=1296 不可枚举），但 Easy 的对比足够支持断言 D。

### 3.5 Welch t-test + Holm-Bonferroni 显著性 + 论文图

`hyper_mve/experiments/stats.py` 实现 `welch_t_test(a: Sequence[float], b: Sequence[float], alpha: float = 0.05) -> TTestResult`（用 `scipy.stats.ttest_ind(a, b, equal_var=False)`）。`>2` 方法对比时 Holm-Bonferroni 校正 family-wise error。`compare` CLI 输入两个 `RunRegistry` JSONL（hyper vs baseline）→ 输出 markdown 表（mean ± sem, p-value, sig flag）+ matplotlib bar+errorbar 图（backend = Agg，无 GUI）。

5 seeds 是 Ch6.2.3 的最低门槛；<5 seeds 触发 `[WARN]` log 但不阻塞 stats 计算（C8-ABL-STAT1）。

### 3.6 预计算 oracle ceiling 缓存：regret 分母

`runs/_oracle_ceilings/<config_hash>/<c>.json` 缓存每个 (config, c) 下 oracle_only 跑出的 ceiling return（5 seeds 平均）。`config_hash` = `hashlib.blake2b(json.dumps(cfg.env + cfg.model, sort_keys=True))[:16]`，env+model 改变 → cache 失效。`regret(c) = ceiling(c) - method(c)`（C8-EVAL-REGRET1）。cache miss → unified_evaluator 自动触发一次 `python -m hyper_mve.scripts.train_main --variant hyper --curriculum_mode oracle_only --seeds 5 --eval_only` 重跑写入。

---

## 4. Impact（影响范围）

### 4.1 代码影响估计（实施期 2-3 周, Phase D-F）

| 范畴 | 文件 | 行数估计 |
|------|------|---------|
| `hyper_mve/eval/` 8 文件 | unified_evaluator / eval_report / zero_shot / c_hidden / regret / planner_modes / mup_verification / __init__ | +1200 |
| `hyper_mve/experiments/` 6 文件 + 5 YAML | sweep / run_registry / ablate / stats / compare / __init__ + 5 ablation YAML | +1200 |
| 测试（C8-EVAL-* + C8-ABL-*）| `tests/eval/` 9 文件 + `tests/experiments/` 8 文件 | +600 |
| 下游补丁（3 处）| observations.py +3 / mve_planner.py +1 / train_main.py CLI rename +约 5 | +10 |
| cfg 字段加入（5 字段）| env_config.py / train_config.py / eval_config.py | +20 |
| **合计** | | **≈ +3000 LOC** |

加上 pkg-07 的 +5000 LOC，整个论文实验侧总 LOC ≈ +8000，属于"中型实验包"，与 v4 主体（~6000 LOC）相当。

### 4.2 GPU 预算估算（实施期 + 论文跑表期）

| 项目 | 配置 | 估算 |
|------|------|------|
| 主表 hyper 5 seeds × 2 presets | hyper full-train + eval | 80 GPU-hr |
| 主表 5 internal × 5 seeds × 2 presets | internal full-train + eval | 400 GPU-hr |
| 主表 3 Tier-1 external × 3 LR × 5 seeds × 2 presets | external full-train + eval | 100 GPU-hr（pkg-07 自带 LR sweep）|
| Tier-2 MAMBA（if sourced）× 5 seeds × 2 presets | external full-train + eval | 30 GPU-hr |
| Zero-shot 协议（train 子 c → test 4 unseen）| 5 seeds × 4 unseen c × eval-only | 5 GPU-hr |
| μP 自检（2 widths × 3 LRs × 3 seeds, Easy）| 18 full-train Easy | 60 GPU-hr |
| Abl4 CRN × Joint/CoordDesc 2×2 × 5 seeds × 2 presets | 20 full-train | 80 GPU-hr |
| Abl4 Joint Easy N=2 only × 5 seeds | 5 full-train Easy | 10 GPU-hr |
| Abl6 Fehr-Schmidt 3×3 × 3 seeds × Medium | 27 full-train | 50 GPU-hr |
| Abl7 curriculum 3-cell × 5 seeds × Medium | 15 full-train | 30 GPU-hr |
| Oracle ceiling cache 预算（hyper oracle_only × 5 seeds × 7 c-grid）| 35 eval-only | 5 GPU-hr |
| **小计** | | **≈ 850 GPU-hr** |

Plan File §"GPU budget" 给的 620 GPU-hr 估计偏紧（基于 310 runs × 2 GPU-hr/run）；本表更细。论文实际执行可砍 Tier-2 MAMBA + Abl6 部分单元，下限 ≈ 700 GPU-hr。两 GPU 并行（per-GPU semaphore）→ 实际 wall-clock ≈ 350 hr ≈ 2 周。可控。

### 4.3 与 pkg-07 的契约消费表面

pkg-07 → pkg-08 单向消费，pkg-07 SDD 零修改。消费 5 锚点（spec 08 §6 漂移检测对账）：

| 来自 pkg-07 | 启动条件 | 被 pkg-08 哪个 spec 消费 |
|------------|---------|-------------------------|
| `REGISTRY` 11 keys finalized（5 internal + 3 Tier-1 + MAMBA + 2 stubs）| pkg-07 spec 01 §2 锁 | pkg-08 spec 05（sweep enumeration） |
| `evaluate(env_fn, c_grid, episodes) -> EvalReport` 签名稳定 | pkg-07 spec 08 锁 | pkg-08 spec 01（unified evaluator 反向消费） |
| `ResourceCommonsPettingZooEnv` N-parametric + 两 flag 信息门控 | pkg-07 spec 04 锁 | pkg-08 spec 01（external runner 通路）+ spec 02（c_hidden 通路与 oracle_mode 正交）|
| `cfg.baselines.*` namespace + `external_lr_sweep_grid` | pkg-07 D10 | pkg-08 spec 05（sweep 默认枚举 LR sweep grid） |
| pkg-06 → pkg-07 supersede 链 + `create_baseline` 工厂 | pkg-07 spec 01 头部 | pkg-08 spec 05 工厂调用一致 |

任一锚点漂移 → spec 08 §6 强制 reviewer 在 pkg-08 spec 01 / spec 05 / spec 08 同步修订对账栏勾确认。

---

## 5. R8-1 ~ R8-12 风险列表

| # | 风险 | 缓解 | 检测 |
|---|------|------|------|
| **R8-1** | μP base-shape LR-doubling 自检失败（width × 2 时最优 LR 不严格 ÷ 2）| spec 04 fallback：论文 drop μP 主张，appendix 披露"我们采用 μP 框架但 base-shape 校验未通过，最优 LR 由独立 sweep 决定" | spec 04 单测 `test_mup_lr_doubling_or_disclose` |
| **R8-2** | subprocess sweep CUDA 资源未释放（驱动级 leak）→ 第 N 个 row OOM | per-row subprocess 强制 GC + `torch.cuda.empty_cache` + 最后 `os._exit(0)` 退出；spec 05 GPU 内存上界监控；每 100 row dump 状态 | spec 05 单测 `test_sweep_memory_stable_over_100_rows` |
| **R8-3** | regret cache 因 config 哈希算法不稳定（dict ordering）导致 cache miss 风暴 | `json.dumps(sort_keys=True)` + `blake2b(16)` 锁定；spec 02 单测验证 cache key 在 cfg round-trip 后稳定 | spec 02 单测 `test_oracle_ceiling_cache_key_stable` |
| **R8-4** | Joint-enum 36-cell 只在 Easy N=2 跑 → Medium / Hard 上断言 D 的"双技术联合解 SNR 崩塌"无独立证据 | 论文明示"Easy N=2 是 Joint 可枚举的唯一档，Medium/Hard 走 CoordDesc，是 CoordDesc 在 SNR 崩塌时仍 ≥ 均匀的间接证据"——spec 06 abl4_joint_easy_n2 README 声明 | spec 06 §3 narrow scope 段落 |
| **R8-5** | 统计 power 在 n=5 seeds 时不足，Welch t 检测不到中等效应（d ≈ 0.5）| Ch6.2.3 声明"5 seeds 是最低门槛，关键 contrast 跑到 10 seeds"；hyper vs 主要 baseline 加跑 10 seeds | spec 07 §2 power 估算表 |
| **R8-6** | `use_coord_desc → randomize_order` rename 碰撞历史 checkpoint（cfg 序列化字段名）| alias-with-deprecation：旧 cfg 反序列化时把 `use_coord_desc` 映射到 `randomize_order`；spec 06 单测覆盖 round-trip | spec 06 单测 `test_use_coord_desc_alias_roundtrip` |
| **R8-7** | c_hidden 下游补丁（Pkg-02 obs-mask 3 行）破坏 Pkg-02 现有单测 | spec 02 + spec 08 §5 强制 `pytest tests/envs/resource_commons/` 全过；CI 加 `--strict-markers` | spec 08 §5 单测 `test_obs_mask_backwards_compatible` |
| **R8-8** | RunRegistry JSONL 并发写竞争 → row 损坏（半行）| 文件锁（POSIX `fcntl.flock` LOCK_EX / Windows `msvcrt.locking`）+ row 写入前 `json.dumps` 确保单行原子；row 校验 `json.loads` 失败时 quarantine | spec 05 单测 `test_run_registry_concurrent_safe`（fork 8 worker） |
| **R8-9** | `EvalReport` schema 漂移（实施期某 spec 加字段未同步 spec 08）| spec 08 §3 字段穷举 + `@dataclass(frozen=True)` 严格；CI grep `EvalReport(` 调用确保参数集合匹配 | spec 08 单测 `test_eval_report_schema_frozen` |
| **R8-10** | matplotlib 在无 X server CI 环境下 crash | `matplotlib.use("Agg")` 在 `compare.py` 顶部强制 backend；spec 07 单测 headless 跑 | spec 07 单测 `test_compare_plot_renders_headless` |
| **R8-11** | 4 planner eval mode literal 未来需要扩展（如加 "planner_no_qstd_floor"）| `Literal[...]` 在 mode 列表只增不删；新 mode 加入需修 spec 03 + spec 08，但默认值不变 | spec 03 §3 扩展协议 |
| **R8-12** | Oracle ceiling cache miss 触发自动 oracle_only 重跑 → 隐式 GPU 预算膨胀 | cache miss 时 print `[INFO] cache miss, will spawn oracle_only run (~5 GPU-hr)` 让用户决定；可设 `cfg.eval.regret_cache_strict=True` 拒绝自动重跑 | spec 02 §4 cache 协议 |

---

## 6. Non-Goals（明确不做）

- **NG1**: ❌ 不修改 Ch4 / Ch5 论文文档（pkg-08 落地 + 主表跑完后由 Ch6 改写阶段统一处理；pkg-08 期间只产数）
- **NG2**: ❌ 不引入新 RL framework（不 vendor stable-baselines3 / RLlib；stats 只用 scipy，画图只用 matplotlib）
- **NG3**: ❌ 不实现任何 baseline model 类（由 pkg-07 实施期承担；pkg-08 只消费 `create_baseline()` 与 `evaluate()` 返回）
- **NG4**: ❌ 不修改 Pkg-01..05 任何 SDD（5 cfg 字段 + 3 下游补丁是消费态声明，spec 08 §5 显式列出）；不修改 Pkg-07 任何 SDD（仅消费契约）
- **NG5**: ❌ 不在 Hard preset（N=8）上跑主表（GPU 预算受限；Hard 留作 follow-up，spec 07 注释"Hard 评估 = 论文 future work"）
- **NG6**: ❌ 不为 Tier-2 / stub baseline（MAMBA / MARIE / GA）做"backfill 补救实施"——若 pkg-07 实施期某 Tier-2 sourcing 失败转 stub，pkg-08 sweep 输出 `"skipped: NotImplementedError"` row，不阻塞主表
- **NG7**: ❌ 不做论文图最终 LaTeX 排版 / Tikz 重画（matplotlib 输出 png 与 svg，由论文成稿期人工 polish）
- **NG8**: ❌ 不做 thesis Chapter 7（discussion / future work）writing——pkg-08 出包后 thesis-resume gate 触发，回 `academic-research` 分支续写 Ch6 主表叙事

---

## 7. pkg-07 契约消费引用（显式锚点对账）

> 本节是 spec 08 §6 漂移检测的预演。pkg-08 全部对外契约消费**只**走以下 5 锚点，任何漂移都触发 pkg-08 spec 01 / spec 05 / spec 08 同步修订。

| 锚点 | pkg-07 出处 | pkg-08 消费位置 | 漂移检测方法 |
|------|-------------|----------------|------------|
| **REGISTRY 11 keys**（`MappingProxyType` 只读）| `pkg-07/specs/01-baseline-registry-and-cli.md` §2.1 + §2.3（量词 canonical）| `pkg-08/specs/05-sweep-harness-and-run-registry.md` §3（sweep enumeration 起点）| Day 8 `check_ref_matrix.ps1` grep `REGISTRY` 在 pkg-07 spec 01 / pkg-08 spec 05 双向出现 |
| **`evaluate(env_fn, c_grid, episodes) -> EvalReport` 签名**（internal + external 同 schema）| `pkg-07/specs/08-integration-contracts.md` §3 evaluate 签名锁 | `pkg-08/specs/01-unified-evaluator.md` §3 EvalReport schema + `pkg-08/specs/08-integration-contracts.md` §3 字段穷举 | 单测 `test_eval_report_schema_frozen` 与 pkg-07 spec 08 `test_external_runner_evaluate_returns_eval_report` 共用同一 `EvalReport` import |
| **`ResourceCommonsPettingZooEnv` N-parametric + 两 flag 信息门控**（`oracle_mode=False` AND `eval_info_mode=False`）| `pkg-07/specs/04-pettingzoo-adapter.md` §2 适配器协议 + LEAK_SURFACE 4 字段 + SCHEMA_MARKERS 3 字段 | `pkg-08/specs/01-unified-evaluator.md` §4（external runner eval 通路）+ `pkg-08/specs/02-zero-shot-and-c-hidden.md` §3（c_hidden 与 oracle_mode 正交）| spec 02 单测 `test_c_hidden_orthogonal_to_oracle_mode` |
| **`cfg.baselines.*` namespace** + `external_lr_sweep_grid: Mapping[str, tuple]` | `pkg-07/design.md` §4 D10 + `pkg-07/specs/01-baseline-registry-and-cli.md` §2.4 | `pkg-08/specs/05-sweep-harness-and-run-registry.md` §3.2（sweep 默认枚举 LR sweep grid）| spec 05 单测 `test_sweep_default_enumerates_external_lr_grid` |
| **`BaselineLike = Union[BaselineModel, ExternalBaselineRunner]` Union type**（pkg-07 用于 sweep harness type sig）| `pkg-07/specs/08-integration-contracts.md` §4 BaselineLike 定义 | `pkg-08/specs/05-sweep-harness-and-run-registry.md` §4 sweep `RowResult.baseline_kind` 字段（区分 internal/external row）| spec 05 单测 `test_row_result_records_baseline_kind` |

**漂移检测启动条件**：pkg-07 SDD finalize 后任一上述 5 锚点修改 → pkg-08 spec 08 §6 reviewer 对账栏强制勾确认；不勾不出包。Day 8 `check_ref_matrix.ps1` 真跑校验，不写"理想态"csv。

---

## 8. 出包后的 thesis-resume gate

pkg-08 SDD 9 天 + 实施期 2-3 周（Phase D-F）落地后，主表跑完即触发 thesis-resume gate：

1. 切回 `academic-research` 分支（git branch `academic-research`）；
2. 把 `runs/registry.jsonl` + `runs/_oracle_ceilings/` + `runs/_figures/` 软链接进 `docs/thesis_plan/data/`；
3. 续写 Ch6（Methods 主表 + 4 ablation 表 + zero-shot 表 + μP 自检图）；
4. Ch6 完稿后回 `main` 分支 merge pkg-07/08 SDD，开 Pkg-09（如有 follow-up 实验）。

---

**END of pkg-08 proposal.md**
