# Pkg-07: Baselines (Internal + External)

> **状态**：Draft — Day 1 三件套待用户审阅（design.md = HARD GATE）
> **包 ID**：`pkg-07-baselines`
> **Supersedes**：[`pkg-06-baselines`](../pkg-06-baselines/README.md)（2026-06-19） — 吸收其 5 个内部 variant，并扩展外部范式 baseline
> **工期**：SDD 10 天 · 实施 4–5 周（含 MAMBA sourcing & vendoring）· **GPU 算力**（实施期）：≈ 250 GPU-hr（含 Tier-1 LR sweep + smoke 收敛性检查；不含 Pkg-08 全量主表）

---

## 📚 文档导览

| 文档 | 目的 | 字数 |
|------|------|------|
| [`proposal.md`](./proposal.md) | **先读这个**：Why（断言 + 外部 baseline 必要性）/ What Changes / Capabilities / Impact / R7-1..R7-12 | ~5000 |
| [`design.md`](./design.md) | 设计抉择：Context / Goals / 框架澄清 / 10 项 Decisions（D1-D10）/ Decision Acceptance | ~7000 |
| [`specs/01-baseline-registry-and-cli.md`](./specs/01-baseline-registry-and-cli.md) | `create_baseline(cfg, variant)` 统一工厂 + **11-key REGISTRY**（5 internal + 3 Tier-1 + MAMBA + 2 stubs）+ supersede 声明 + CLI ↔ factory-arg ↔ model-class 三列映射表（继承 pkg-06 §3.3） | ~3500 |
| [`specs/02-shared-backbones-internal.md`](./specs/02-shared-backbones-internal.md) | `create_rep_net/belief_net/tri_context_encoder` + 等参共享契约（继承自 pkg-06 spec 02）| ~2500 |
| [`specs/03-internal-variants.md`](./specs/03-internal-variants.md) | `input_wide/deep`（断言 B）+ `ma_muzero`（断言 A）+ `no_belief` / `rewardhead_explicit_type`（断言 A/C/Abl6.x/7）合并自 pkg-06 specs 03+04+05 | ~3500 |
| [`specs/04-pettingzoo-adapter.md`](./specs/04-pettingzoo-adapter.md) | `ResourceCommonsPettingZooEnv` 适配器 + Oracle 信息门控（架构 keystone） | ~3000 |
| [`specs/05-external-mappo.md`](./specs/05-external-mappo.md) | MAPPO Tier-1（lzj port）+ CTDE 集中 critic + LR sweep 表 | ~3000 |
| [`specs/06-external-qmix-mamuzero-mamba.md`](./specs/06-external-qmix-mamuzero-mamba.md) | QMIX + MA-MuZero-GH 两个 Tier-1 vendoring 计划 + MAMBA Tier-2 sourcing 协议 + MARIE/GA 占位 stub | ~3500 |
| [`specs/07-fairness-protocol.md`](./specs/07-fairness-protocol.md) | **双层公平协议**：Internal 严格等参（5%/10% 双阈值，继承自 pkg-06）+ External 披露式（params/wall-clock/LR-swept best）| ~2500 |
| [`specs/08-integration-contracts.md`](./specs/08-integration-contracts.md) | **对外硬契约**：`create_baseline` 签名 + `evaluate()` 统一签名 + `REGISTRY` 只读映射 + Pkg-02/05 下游补丁清单 + Pkg-08 接入伪签名 | ~3500 |

---

## 🎯 一句话目标

**为论文 4 条断言提供 5 个内部对照 + 4 个外部范式对照（Tier-1: MAPPO/QMIX/MA-MuZero;Tier-2: MAMBA-if-sourced）**：在 `hyper_mve/baselines/` 下新增统一工厂 `create_baseline(cfg, variant)` + 共享后端 + 5 内部 baseline + 3 Tier-1 外部 baseline + 1 适配器（`hyper_mve/envs/adapters/pettingzoo_wrapper.py`），所有 internal baseline 复用 Pkg-05 同一 `MuZeroTrainer/Worker/Buffer/compose_total_loss`（仅 model 类不同），external baseline 通过统一 PettingZoo 适配器消费 `ResourceCommonsEnv` 并自带训练器，所有 baseline 通过相同的 `evaluate(env_fn, c_grid, episodes) -> EvalReport` 契约喂给 Pkg-08 unified evaluator。

---

## ✅ 关键 Acceptance Criteria（速览）

### 结构硬约束（must pass）

**Internal 变体（C7-INT-*）** — 13 项继承自 pkg-06 C6-*（详见 spec 03/07）：
- [ ] 5 internal variant 均可经工厂实例化（C7-INT-FACT1）
- [ ] 工厂收 "hyper" 抛 ValueError（C7-INT-FACT2）
- [ ] RepNet/BeliefNet/TriCtx 跨 internal variant 参数量逐位相同（C7-INT-FAIR2）
- [ ] 5 internal variant 实现完整 7-API（C7-INT-API1）
- [ ] 不泄漏 oracle types（C7-INT-SELF1）
- [ ] 同一 MuZeroTrainer/Worker/Buffer/compose_total_loss（C7-INT-REUSE1）
- [ ] 5 internal variant 共用同一 `BeliefGradGating`（C7-INT-GRAD1）
- [ ] input-conditioned 不含 DualHyperNetwork（C7-INT-STRUCT1）
- [ ] ma_muzero 共享单一 RewardHead（C7-INT-STRUCT2）
- [ ] 条件化消费子系统参数 ≤5% warn / ≤10% fail（C7-INT-FAIR1）

**External 变体（C7-EXT-*）** — 新增约束（详见 spec 04/05/06/07）：
- [ ] 3 Tier-1 external variant 均可经工厂实例化（C7-EXT-FACT1）
- [ ] PettingZoo 适配器默认 `oracle_mode=False` **AND** `eval_info_mode=False`,不泄漏 `info["c_true"]` / `info["types"]` / `info["hotspot_centers"]` / `info["resource_state"]` 任一字段（C7-EXT-ADPT1）
- [ ] 适配器实现 `pettingzoo.ParallelEnv`，**N agents**（从 `env_cfg.N` 读，`agent_0..agent_{N-1}`），离散 A=6 actions；smoke 跑 Easy (N=2) 与 Medium (N=4)（C7-EXT-ADPT2）<br/>验证机制：`test_adapter_info_gating.py` 同时回归 ADPT1 (info gating) + ADPT2 (结构契约) 两条约束
- [ ] 3 Tier-1 external runner 实现统一 `.evaluate(env_fn, c_grid, episodes) -> EvalReport`（C7-EXT-API1）
- [ ] 3 Tier-1 external runner 在 Easy preset 上 smoke pass（return 优于 random within 20K env steps）（C7-EXT-SMOKE1）
- [ ] External LR sweep 表声明并执行（每 baseline ≥3 LR × ≥3 seeds）（C7-EXT-FAIR1）
- [ ] MARIE / GA / MAMBA-未 sourced 走 `NotImplementedError` 显式占位（C7-EXT-STUB1）

### 等参/性能验证

- [ ] Internal: pkg-06 等参 5%/10% 双阈值规则（C6-FAIR1 继承）
- [ ] External: 披露式 fairness（参数量 + wall-clock-to-converge + LR-swept best + final return ≥5 seeds）
- [ ] 11-key REGISTRY 单步 forward（internal）/ smoke step（external）在 spec 07 分档预算内（stubs 不参与 perf 预算）

### 期望达到

- [ ] `pytest tests/baselines/` 全过（C7-INT-* 11 测试 + C7-EXT-* ≥7 测试，对应 FACT1/ADPT1/ADPT2/API1/SMOKE1/FAIR1/STUB1）
- [ ] `pwsh sdd/pkg-07-baselines/scripts/check_ref_matrix.ps1` 输出 `[PASS]`
- [ ] `git diff` 确认 Pkg-01..05 SDD 零改动（代码上 Pkg-02 obs-mask + Pkg-05 planner flag + Pkg-05 CLI 三处微补丁由 spec 08 显式声明）

---

## 📅 实施顺序（10 天 SDD + 后续实施）

### SDD 阶段（10 天）

| 阶段 | 天数 | 任务 | 输出 / 审阅点 |
|------|------|------|---------------|
| **Phase 1** | Day 1 | README + proposal + design.md（10 Decisions 锁定 + §3.3 CLI 表）| **design.md 审阅 = HARD GATE**（**11 项**验收清单，全过才放行；详见 §"Day 1 HARD GATE 验收清单"）|
| **Phase 2** | Day 2 | spec 01（工厂 & 11-key REGISTRY）+ spec 04（N-parametric PettingZoo adapter） | 工厂签名 + adapter contract 同时锁定（pkg-08 spec 01 在 Day 2 同步） |
| **Phase 3** | Day 3 | spec 02（shared_backbones, 继承 pkg-06 02）+ spec 03（internal variants 合并）| pkg-06 内容承接 + 等参契约保留 |
| **Phase 4** | Day 4 | spec 05（MAPPO）| MAPPO source-port 计划 + CTDE 公约审阅 |
| **Phase 5** | Day 5 | spec 06（QMIX + MA-MuZero-GH vendor + MAMBA sourcing log）| vendoring 计划 + 许可审阅 + MAMBA 搜索目标声明 |
| **Phase 6** | Day 6 | spec 07（双层 fairness 协议）| Internal 严格 + External 披露式 双段审阅 |
| **Phase 7** | Day 7 | spec 08（集成契约）| 与 Pkg-02/05 下游补丁声明 + Pkg-08 接入伪签名 |
| **Phase 8** | Day 8 | `scripts/check_ref_matrix.ps1`（纯 ASCII）+ `ref_matrix.csv` 真跑 `[PASS]` | M6 引用核对 |
| **Phase 9** | Day 9 | PR_DESCRIPTION.md | 提交准备 |
| **Phase 10** | Day 10 | 用户最终 ack | Pkg-07 SDD finalized |

### Day 1 HARD GATE 验收清单（11 项，全过才放行）

| # | 验收项 | design.md 落点 |
|---|--------|---------------|
| 1 | Internal 5 variant 核心区别表（继承 pkg-06 D7；`ma_muzero` vs `rewardhead_explicit_type` 失败模式撕分：共享头平均梯度 vs 离散分支无法吸收 capability 异质）| §3.2 11-key 工厂矩阵 internal 行 + §4 D7 |
| 2 | External 3 Tier-1 baseline 选型 + 选源记录（MAPPO/QMIX/MA-MuZero）| §3.4 vendoring 表 + §4 D7 |
| 3 | Tier-2 (MAMBA) sourcing 协议（2 日预算 + 失败 fallback）| §4 D8 |
| 4 | MARIE/GA stub 策略 + 不进 main table 的声明 | §4 D9 |
| 5 | 双层 fairness 单元定义（Internal 严格 vs External 披露式）| §4 D5 |
| 6 | PettingZoo 适配器**两 flag** 信息门控（`oracle_mode` + `eval_info_mode`，覆盖 c_true/types/hotspot_centers/resource_state 全 4 字段；含 CTDE 合法 vs 特权状态边界）| §3.5 + §4 D7 + spec 04 接入 |
| 7 | mermaid: external 训练流程 vs internal MuZeroTrainer 流程对比图 | §4 D7 "External 训练流程" 块 + §7 mermaid |
| 8 | cfg 新字段**5 字段穷举**（`cfg.baselines.*` namespace，消费态、不改上游）| §4 D10 |
| 9 | 8-spec ref_matrix 预期表（per-spec 引用矩阵，**不是** variant 矩阵） | §8 |
| 10 | 10 天日历 + 响应式 SLA + Phase 8 真跑 | §6 |
| **11** | **CLI ↔ factory-arg ↔ model-class 三列映射表（11 in-registry + 3 curriculum-override = 14 行）+ 前缀约定脚注（继承 pkg-06 §3.3）** | §3.3 |

### 响应式 SLA

日历是**名义节奏，非硬截止**。**Day 1 hard gate 不通过 → 全流程顺延 1 天**（不带病进 Day 2）。每个 spec 当天末交审，等用户 ack 再进下一个，审阅往返时间不计入"天"。

### 实施期（SDD 落地后 4–5 周，详见 [Plan File](../../../../../Users/zhengwenbo01/.claude/plans/after-some-thought-i-iridescent-rain.md) §"Code-implementation"）

- Phase A（wk 1–2）：internal 5 model classes + registry + 11 tests
- Phase B（wk 2–3）：PettingZoo adapter + MAPPO + smoke
- Phase C（wk 3–4）：QMIX + MA-MuZero-GH vendoring + smoke
- Phase C'（wk 4，并行）：MAMBA sourcing（2 天搜索 → 3 天 port 或转 stub）

---

## 🔑 10 项关键 Decisions（速览）

> **量词澄清**：Decisions 数 = **10**（D1-D10）。**Decision Acceptance / Day-1 HARD GATE 验收清单 = 11 项**（含 #11 = §3.3 CLI 三列映射表，落点在 §3.3 而非 §4 D*）。两个 10/11 不冲突，分别指 Decision 集合与 Acceptance 集合。

| # | Decision | 推荐 | 详见 |
|---|----------|------|------|
| D1 | spec 数量与文档结构 | **8 specs 对称**（继承 pkg-04/05/06 模式）| [design §4 D1](./design.md) |
| D2 | 工厂归属与签名 | **`baselines/__init__.py` 统一 `create_baseline(cfg, variant)`**（取代 pkg-06 `create_baseline_model`，alias 加 `DeprecationWarning`）| [design §4 D2](./design.md) |
| D3 | Internal vs External 分派 | **两 namespace**：`internal/*` → `BaselineModel`（7-API），`external/*` → `ExternalBaselineRunner`（自带 trainer）| [design §4 D3](./design.md) |
| D4 | 共享后端粒度 | **RepNet/BeliefNet/TriCtx 同类同构、独立实例、独立梯度**（继承 pkg-06 D4，仅 internal variant 用）| [design §4 D4](./design.md) |
| D5 | 双层 fairness | **Internal 严格 5%/10% 双阈值（继承 pkg-06 D5）+ External 披露式（params/walltime/LR-swept）**| [design §4 D5](./design.md) |
| D6 | belief 门控一致性 | **5 internal variant 共用同一 BeliefGradGating**（继承 pkg-06 D6）| [design §4 D6](./design.md) |
| D7 | Tier-1 external 选型 | **MAPPO（lzj-port）+ QMIX（PyMARL-vendor）+ MA-MuZero-GH（muzero-general + thin MA wrapper）**| [design §4 D7](./design.md) |
| D8 | Tier-2 MAMBA sourcing 协议 | **原论文 repo → 社区 fork → 2 日预算; 失败转 stub 保留搜索日志**| [design §4 D8](./design.md) |
| D9 | MARIE/GA stub 策略 | **`NotImplementedError` 占位 + 保留 CLI 名 + 不进 main table 声明**| [design §4 D9](./design.md) |
| D10 | cfg 字段归属 | **`cfg.baselines.*` 新 namespace + Pkg-01 spec05 同步声明（消费态、不改上游）**| [design §4 D10](./design.md) |

---

## 🚨 v4 关键约束（继承 + 新增）

> 统一前缀：`FACT`(工厂) / `FAIR`(等参/披露) / `SELF`(Self-Info) / `API` / `REUSE`(复用) / `GRAD`(梯度门控) / `STRUCT`(结构) / `CFG`(配置消费) / `PERF`(性能护栏) / `ADPT`(适配器) / `SMOKE`(收敛性检查) / `STUB`(占位)。
> 二级前缀：`INT`(internal) / `EXT`(external)。

### Internal（继承自 pkg-06 C6-*；重号为 C7-INT-*）

| # | 约束 | 来源 | 对本包的硬约束 |
|---|------|------|----------------|
| **C7-INT-FACT1** | 5 internal variant 均可实例化 | pkg-06 C6-FACT1 | spec 01 单测 |
| **C7-INT-FACT2** | 工厂收 "hyper" 抛 ValueError | pkg-06 C6-FACT2 / D3 | spec 01 单测 |
| **C7-INT-FAIR1** | 条件化消费子系统 ≤5% warn / ≤10% fail | pkg-06 C6-FAIR1 | spec 07 单测 |
| **C7-INT-FAIR2** | RepNet/BeliefNet/TriCtx 跨 internal variant 参数量逐位相同 | pkg-06 C6-FAIR2 | spec 02 单测 |
| **C7-INT-SELF1** | 不泄漏 oracle types（每 internal variant）| pkg-06 C6-SELF1 | spec 03 单测 `test_internal_self_info.py`（注：pkg-06 spec 06 = 7-API/Self-Info,在 pkg-07 已合并入 spec 03 internal-variants；pkg-07 spec 06 是 QMIX/MA-MuZero/MAMBA external） |
| **C7-INT-API1** | 7 方法签名一致 | pkg-06 C6-API1 | spec 03 单测 |
| **C7-INT-API2** | stateful 用最后一次 subjective agent_id | pkg-06 C6-API2 | spec 03 单测 |
| **C7-INT-REUSE1** | 同一 MuZeroTrainer/Worker/Buffer/compose_total_loss | pkg-06 C6-REUSE1 | spec 08 单测 |
| **C7-INT-GRAD1** | pre-5K BeliefNet 梯度=0（每 internal variant）| pkg-06 C6-GRAD1 | spec 03 单测 |
| **C7-INT-STRUCT1** | input-conditioned 不含 DualHyperNetwork | pkg-06 C6-STRUCT1 | spec 03 单测 |
| **C7-INT-STRUCT2** | ma_muzero 共享单一 RewardHead | pkg-06 C6-STRUCT2 | spec 03 单测 |
| **C7-INT-CFG1** | 工厂消费 4 个 internal baseline cfg 字段无 hardcoded 默认值 | pkg-06 C6-CFG1 | spec 01 单测 |
| **C7-INT-PERF1** | input baseline 单步 forward ≤ 2.0× hyper | pkg-06 C6-PERF1 | spec 07 单测 |

### External（新增 C7-EXT-*）

| # | 约束 | 来源 | 对本包的硬约束 |
|---|------|------|----------------|
| **C7-EXT-FACT1** | 3 Tier-1 external variant 均可实例化 | design D3 | spec 01 单测 |
| **C7-EXT-ADPT1** | 适配器默认 `oracle_mode=False` AND `eval_info_mode=False`,不泄漏 `info["c_true"]/info["types"]/info["hotspot_centers"]/info["resource_state"]` 任一字段；两 flag 正交,strip 任何 `_*_fields` schema-marker | design D7 + §3.5 + spec 04 | spec 04 单测 `test_adapter_info_gating.py`（覆盖 reset + step 双路径） |
| **C7-EXT-ADPT2** | 适配器实现 `pettingzoo.ParallelEnv`，**N agents (从 `env_cfg.N` 读，agent_0..agent_{N-1})**，离散 A=6 actions；smoke 跑 Easy (N=2) 与 Medium (N=4) | spec 04 | spec 04 单测 N-parametric over {Easy, Medium} |
| **C7-EXT-API1** | 3 Tier-1 external runner 实现 `.evaluate(env_fn, c_grid, episodes) -> EvalReport` | design D3 + spec 08 | spec 08 单测 |
| **C7-EXT-SMOKE1** | 3 Tier-1 external runner 在 Easy preset 上 return > random within 20K env steps | design D7 | spec 05/06 单测 |
| **C7-EXT-FAIR1** | External LR sweep ≥3 LR × ≥3 seeds，结果披露 | design D5 | spec 07 单测/协议 |
| **C7-EXT-STUB1** | MARIE/GA/MAMBA-未 sourced 走 `NotImplementedError` | design D9 | spec 06 单测 |

---

## 📦 输出清单（PR 时检查）

### 新增（1 工厂 + 1 共享后端 + 1 适配器 + 5 internal + 3-4 external，全部在 `hyper_mve/baselines/` & `hyper_mve/envs/adapters/`）

```
hyper_mve/baselines/
├── __init__.py                          # create_baseline + REGISTRY + supersede shim
├── _runner_protocol.py                  # ExternalBaselineRunner protocol (eval contract)
├── shared_backbones.py                  # 继承 pkg-06 02
├── internal/
│   ├── __init__.py
│   ├── input_conditioned.py             # wide + deep
│   ├── ma_muzero.py
│   ├── no_belief.py
│   └── explicit_type_reward.py
└── external/
    ├── __init__.py
    ├── mappo.py                         # Tier-1
    ├── qmix.py                          # Tier-1（vendored from PyMARL）
    ├── ma_muzero_gh.py                  # Tier-1（vendored from muzero-general + MA wrapper）
    ├── mamba.py                         # Tier-2（active sourcing）或 stub
    ├── marie.py                         # 占位 stub
    └── ga.py                            # 占位 stub

hyper_mve/envs/adapters/
├── __init__.py
└── pettingzoo_wrapper.py                # ResourceCommonsPettingZooEnv

tests/baselines/
├── internal/                            # 承接 pkg-06 11 单测
│   ├── test_factory.py
│   ├── test_shared_backbones.py
│   ├── test_param_fairness.py
│   ├── test_7api_conformance.py
│   ├── test_self_info.py
│   ├── test_reuse.py
│   └── test_grad_gating.py
└── external/
    ├── test_adapter_info_gating.py      # 4-字段 × 2-flag × {reset,step} N-parametric over {Easy,Medium}
    ├── test_mappo_smoke.py
    ├── test_qmix_smoke.py
    ├── test_ma_muzero_gh_smoke.py
    ├── test_external_eval_contract.py
    ├── test_registry.py                 # 11 REGISTRY keys enumerable + dispatch + reject "hyper" + CLI↔factory-arg round-trip
    └── test_stub_external_baselines.py  # MARIE/GA + MAMBA-未 sourced 抛 NotImplementedError
```

### 修改（2 处下游补丁；纯下游契约,不动 pkg-01..05 SDD）

```
hyper_mve/scripts/train_main.py                    # +6 个 --variant 字串（external_mappo/qmix/ma_muzero_gh/mamba/marie/ga；后 3 个调用工厂时抛 NotImplementedError）（spec 08 声明）
hyper_mve/configs/v4_config.py                     # 新增 cfg.baselines 子 dataclass（消费态，5 字段）（spec 08 声明）
```

> **量词澄清**：pkg-07 spec 08 §7.1 列出 4 个 patches 时包括上述 2 处修改 + §新增 块的 2 个新模块（`envs/adapters/`、`baselines/`）。本 §修改 块仅列对**既有文件**的修改；新模块在 §新增 块。两个列法不冲突，spec 08 是 4-patch 一总览，README 拆为「新增 = 新模块」+「修改 = 既有文件 patch」。
>
> **明确不在 pkg-07 范围的下游补丁**（避免混淆）：
> - `hyper_mve/envs/resource_commons/observations.py` +3 行 c_visible 掩码 — 归 pkg-08 spec 02 §5.1 + pkg-08 spec 08 §5 声明（zero-shot c_hidden 评估通路）。
> - `hyper_mve/planning/mve_planner.py` +1 行 mve_joint_enumerate 分支 — 归 pkg-08 spec 06 §5.1 + pkg-08 spec 08 §5 声明（Ablation 4 Joint cell）。

待声明 cfg 字段（5 个）由 Pkg-01 spec05 同步声明（消费态、不改上游）：
- `cfg.baselines.internal_wide_hidden_dim`
- `cfg.baselines.internal_deep_layers`
- `cfg.baselines.internal_ma_muzero_share_pred_head`
- `cfg.baselines.internal_explicit_type_branches`
- `cfg.baselines.external_lr_sweep_grid`

---

## 🔗 上下游关联

| 关系 | 包 | 接口 |
|------|-----|------|
| **本包 supersedes** | Pkg-06 baselines | 吸收其 5 internal variant + 工厂签名（重命名 + alias）|
| **本包 ← Pkg-03** | TriContextEncoder + BeliefNet | RepNet/BeliefNet/TriCtx 共享契约（继承 pkg-06 spec 02）|
| **本包 ← Pkg-04** | DualHyperNetwork v2 & Model | 7-API + stateful + `BeliefGradGating`（仅 internal variant 用）|
| **本包 ← Pkg-05** | Trainer & Worker | 5 条复用约束（仅 internal）+ `--variant` CLI 扩展 6 个 `external_*` 名（含 stubs）|
| **本包 ← Pkg-02** | ResourceCommonsEnv | PettingZoo 适配器封装；spec 08 声明 obs-mask 单行补丁 |
| **本包 → Pkg-08** | Eval + Ablation | `REGISTRY` + 适配器 + `evaluate() -> EvalReport` 统一契约 |

### 📌 下游启动条件速查

| 下游包 | 启动条件 | 验收 spec |
|--------|---------|-----------|
| **Pkg-08（Eval + Ablation）** | (a) `REGISTRY` 11 keys finalized（5 internal + 3 Tier-1 + MAMBA + 2 stubs；stubs 在 sweep 中 produce "skipped: NotImplementedError" rows）（spec 01）<br>(b) `evaluate() -> EvalReport` 签名稳定（spec 08）<br>(c) N-parametric adapter contract + 两 flag 信息门控 finalized（spec 04）| spec 01 + 04 + 08 |

---

## 📖 引用源

- Plan File: [`plans/after-some-thought-i-iridescent-rain.md`](../../../../Users/zhengwenbo01/.claude/plans/after-some-thought-i-iridescent-rain.md)（pkg-07/08 整体规划与决策记录）
- Supersedes: [`pkg-06-baselines/README.md`](../pkg-06-baselines/README.md)（5 internal variant 来源；C6-* 约束继承）
- Theory Audit: [`docs/Review_v4_TheoryAudit_2026-06.md`](../../docs/Review_v4_TheoryAudit_2026-06.md) §10.3（pkg-07/08 必备内容备忘）
- Roadmap: [`docs/Hyper_MuZero_v4_Roadmap.md`](../../docs/Hyper_MuZero_v4_Roadmap.md)（Stage 4-5 + 决策点 1-4）
- Pkg-03 SDD: spec 08 §6（RepNet/BeliefNet/TriContextEncoder 共享契约）
- Pkg-04 SDD: spec 02（7-API）+ spec02 §3.4 / spec04（BeliefGradGating）
- Pkg-05 SDD: spec 08 §3.1/§3.2/§6.2（工厂签名 + 5 条复用约束 + --variant CLI 语义）
- 外部源（vendoring 计划）：
  - MAPPO: [`D:\RL\lzj\MAPPO\MAPPO_main.py`](file:///D:/RL/lzj/MAPPO/MAPPO_main.py)
  - QMIX: PyMARL `oxwhirl/pymarl`（github）
  - MA-MuZero-GH: `werner-duvaud/muzero-general`（github）+ thin MA wrapper
  - MAMBA: 待 spec 06 sourcing 协议（原论文 repo → 社区 fork）

---

## 📑 spec 间引用表（M6：避免事后修改）

### 期望引用矩阵（人工对账表）

| Spec | 引用的其他 spec |
|------|----------------|
| spec 01 | spec 02（shared_backbones）, spec 03（internal variants）, spec 04（adapter）, spec 05/06（external variants）, spec 08 |
| spec 02 | spec 03（internal 消费 backbone）, spec 08 |
| spec 03 | spec 02（backbone）, spec 07（等参）, spec 08 |
| spec 04 | spec 05/06（external 消费 adapter）, spec 08 |
| spec 05 | spec 04（adapter）, spec 07（LR sweep）, spec 08 |
| spec 06 | spec 04（adapter）, spec 07（LR sweep + sourcing log）, spec 08 |
| spec 07 | spec 03（internal 等参）, spec 05/06（external 披露式）, spec 08 |
| spec 08 | spec 01-07 全引用 + Pkg-05 spec08 契约模式 + Pkg-08 下游接入伪签名 |

### Day 8 机器可校验引用核对

```powershell
cd D:\RL\hyper_mve\sdd\pkg-07-baselines
pwsh scripts\check_ref_matrix.ps1
# → 生成 ref_matrix.csv + 终端 [PASS]/[FAIL]
```

引用扫描正则匹配两种**包内**引用形式，并排除**跨包**引用（与 pkg-06 同模式）。

---

## 🚦 Review Checklist（用户审阅时勾选）

- [ ] proposal.md 读完，**Why** 充分（继承 pkg-06 4 条断言论证 + 新增"外部 baseline 必要性：论文需对比外部范式" + "PettingZoo 适配器单点封装" + "Tier 化以管理 GPU 与日历预算"）
- [ ] design.md 读完，**Day 1 HARD GATE 11 项验收清单**全过（10 Decision-anchored + #11 CLI 表 in §3.3）
- [ ] 5 internal variant 区别清晰（继承 pkg-06，加入新工厂签名 + ma_muzero vs rewardhead_explicit_type 失败模式区分：共享头平均梯度 vs 离散分支无法吸收 capability 异质 —— 注：术语「撕分」沿用 pkg-06，意为"按失败模式拆分"，与断言 A 名称「梯度撕裂」不同）
- [ ] 3 Tier-1 external baseline 选型合理（MAPPO/QMIX/MA-MuZero）+ 选源记录可追溯
- [ ] MAMBA sourcing 协议有失败 fallback（不阻塞 SDD finalize）
- [ ] MARIE/GA stub 策略合理（不进 main table）
- [ ] **双层 fairness 协议**口径认可（Internal 严格 vs External 披露式）
- [ ] PettingZoo 适配器**两 flag** 信息门控认可（`oracle_mode=False` AND `eval_info_mode=False` 默认；4 leak-surface 全覆盖；schema-marker strip；CTDE 合法/特权边界；N-parametric agent IDs）
- [ ] **CLI ↔ factory-arg ↔ model-class 三列映射表**（§3.3 14 行）+ 前缀约定脚注可对账 train_main.py:45-52
- [ ] 10 天 SDD 顺序合理（Day 1 HARD GATE 未过顺延 1 天）
- [ ] 出包后可启动 Pkg-08（Eval + Ablation）
