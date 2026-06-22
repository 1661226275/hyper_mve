# Pkg-07: Baselines (Internal + External) — Design

> 配套阅读：[`proposal.md`](./proposal.md)（**先读 proposal 再读本文**）· [`README.md`](./README.md)
> **Day 1 HARD GATE**：§3.3 CLI 表 + §3.5 两 flag 信息门控 + §4 D1-D10 全部通过用户审阅（§5 Decision Acceptance 11 项；§10 用户审阅清单 11 项），方可放行进 spec 01-08 起草

---

## 1. Context

### 1.1 项目阶段

本包是 v4 重构的**第七个 SDD 包**,承接 Pkg-01..05 + **supersedes Pkg-06**：

```
Pkg-01..05                  ✅ 完成
Pkg-06 (Internal Baselines) 📄 SDD only, SUPERSEDED BY pkg-07（5 internal variant 内容继承）
    ↓
Pkg-07 (本包, Internal + External Baselines)
    ↓
Pkg-08 (Eval + Ablation Framework)
```

启动条件:
- Pkg-05 spec08 §3.1 `create_baseline_model` 签名 → pkg-07 spec 01 supersede 并 rename 为 `create_baseline`,alias 加 `DeprecationWarning`
- Pkg-02 `ResourceCommonsEnv` 已稳定,适配器封装的对象明确
- Theory Audit §10.3 锁定 pkg-07/08 必备内容;Plan File 锁定 Tier-1/Tier-2 split

### 1.2 前置依赖（本包消费什么）

| 来源 | 内容 |
|------|------|
| **pkg-06**（superseded）| 5 internal variant 设计 + C6-* 约束 + 等参公平协议 |
| **Pkg-05** spec08 §3.1 | `create_baseline_model(cfg, variant)` 签名（→ rename `create_baseline`）|
| Pkg-05 spec08 §3.2 | 5 条复用约束（仅 internal）|
| Pkg-05 spec08 §6.2 | `--variant` CLI 语义（扩展 3 个 `external_*` 名）|
| **Pkg-04** spec02 | `HyperMuZeroModel` 7-API（仅 internal 实现）|
| Pkg-04 spec02 §3.4 + spec04 | `BeliefGradGating`（仅 internal 用）|
| **Pkg-03** spec08 §6 | RepNet/BeliefNet/TriContextEncoder 共享契约（仅 internal）|
| **Pkg-02** ResourceCommonsEnv | gym 5-tuple API + per-agent obs + heterogeneous types |
| **Ch4_1_Motivation / Ch5** | 4 条断言 + §5.6 等参协议;Theory Audit §10.3 pkg-07/08 备忘 |
| **外部源**（vendoring 计划）| MAPPO（lzj）/ QMIX（PyMARL）/ MA-MuZero-GH（muzero-general）/ MAMBA（active sourcing）|

### 1.3 本包提供（Pkg-08 消费什么）

| 输出 | 消费者 | 用途 |
|------|--------|------|
| `create_baseline(cfg, variant)` **11-key REGISTRY** 工厂（5 internal + 3 Tier-1 + MAMBA + 2 stubs）| Pkg-08 sweep harness | enumerate baseline 跑 main table（stubs 产生 "skipped: NotImplementedError"）|
| `REGISTRY` 只读 mapping | Pkg-08 sweep harness | `--sweep-variants all` 自动枚举 |
| `ResourceCommonsPettingZooEnv` 适配器 | Pkg-08 unified evaluator | external runner 共用同一 env |
| `ExternalBaselineRunner.evaluate() -> EvalReport` | Pkg-08 unified evaluator | internal/external 统一报告格式 |
| `tests/baselines/external/test_*` | Pkg-08 CI | smoke + adapter + registry |

---

## 2. Goals

### 2.1 主目标（必须完成）

1. **G1**: 统一工厂 `create_baseline(cfg, variant)` 覆盖 **11 keys**（5 internal + 3 Tier-1 external + MAMBA Tier-2 + 2 stubs MARIE/GA）+ supersede pkg-06 alias
2. **G2**: `shared_backbones.py` 共享后端契约（继承 pkg-06 spec 02,仅 internal 用）
3. **G3**: 5 internal variant 的 7-API 一致性 + pkg-06 13 C6-* 全部继承为 C7-INT-*
4. **G4**: **N-parametric** PettingZoo 适配器（`ResourceCommonsPettingZooEnv`，agent IDs = `agent_0..agent_{N-1}` where N = `env_cfg.N` ∈ {2,4,8}）+ **两 flag 信息门控**（`oracle_mode` + `eval_info_mode`，4 leak-surface 全覆盖）+ CTDE 合法性边界 + 强制 `test_adapter_info_gating.py`
5. **G5**: 3 Tier-1 external baseline（MAPPO/QMIX/MA-MuZero-GH）的 vendoring + ExternalBaselineRunner 协议 + smoke 收敛性约束
6. **G6**: Tier-2 MAMBA 主动 sourcing 协议（2 日 budget + 失败 fallback 转 stub）
7. **G7**: MARIE / GA `NotImplementedError` 占位策略
8. **G8**: 双层 fairness 协议（Internal 严格 5%/10% + External 披露式）
9. **G9**: 3 处下游补丁的显式声明（Pkg-02 obs-mask + Pkg-05 planner flag + Pkg-05 CLI）—— **不修改 pkg-01..05 SDD**
10. **G10**: `cfg.baselines.*` 新 namespace + Pkg-01 spec05 同步消费态声明
11. **G11**: 8 specs 对称（同 pkg-04/05/06）+ `check_ref_matrix.ps1` 真跑 [PASS]

### 2.2 衍生目标

- **G12**: External LR sweep 协议（≥3 LR × ≥3 seeds）
- **G13**: Smoke 收敛性检查（Easy preset 20K env-steps,return > random）
- **G14**: 每条 C7-INT-*/C7-EXT-* 有具名单测,无悬空（M3 完备）

### 2.3 Non-Goals

- **NG1**: 不实现 baseline 代码（仅 SDD）
- **NG2**: 不修改 Pkg-01..05 任何 SDD（仅消费;3 处实施期补丁由 spec 08 声明）
- **NG3**: 不提供断言 D 的对照 model（Pkg-08 cfg flag 承载）
- **NG4**: 不引入 RLlib/SB3 等大型 RL 框架依赖（minimal-port 政策）
- **NG5**: 不强求 external 与 hyper 等参（披露式 fairness 替代）
- **NG6**: 不重写 ResourceCommonsEnv（仅适配器封装）
- **NG7**: 不修改论文 Ch4/Ch5（实施完成后再统一改 Ch6.x）

---

## 3. 核心框架澄清（Day 1 hard-gate 前置 5 项）

### 3.1 框架澄清①：Internal vs External 两 namespace 分派

> **Internal** = 5 个继承自 pkg-06 的 model 变体,共用 `HyperMuZeroModel` 子集结构 + 实现 7-API + 用 `MuZeroTrainer`。
> **External** = 6 个不同 MARL 范式（policy gradient / Q decomposition / model-based MARL / model-based + belief / 占位 stubs）的"完整算法",自带 trainer + buffer + loss,通过 PettingZoo 适配器消费 env。

两者通过同一工厂 `create_baseline(cfg, variant)` 入口分派,但返回类型不同 → spec 01 锁定 union type `BaselineModel | ExternalBaselineRunner`。

> **Canonical 量词**（避免 doc 漂移）：**REGISTRY = 11 keys**（5 internal + 3 Tier-1 + MAMBA + 2 stubs）。**Methods 主表 = 9 列**（hyper + 5 internal + 3 Tier-1）；MAMBA-if-sourced 加为第 10 列；MARIE/GA stubs 不入主表（仅 CLI-reachable 占位）。**`§8 ref_matrix` 是 8-spec 矩阵**，不是 variant 矩阵。

### 3.2 框架澄清②：create_baseline 工厂矩阵 (11-key REGISTRY)

| 训练入口 | 创建路径 | 返回类型 | 断言 / 用途 |
|----------|----------|---------|-----------|
| `hyper` | `HyperMuZeroModel(cfg)`（**不走工厂**）| HyperMuZeroModel | A/B′/C 主线 |
| `oracle_only` | `HyperMuZeroModel(cfg)` + curriculum override | HyperMuZeroModel | regret 上界（Pkg-08 用）|
| `infer_only` | `HyperMuZeroModel(cfg)` + curriculum override | HyperMuZeroModel | 零样本 belief 推断 |
| `input_wide` | `create_baseline(cfg, "input_wide")` | InputWideBaselineModel | B′ 等参 |
| `input_deep` | 同上 | InputDeepBaselineModel | B′ 等参 |
| `ma_muzero` | 同上 | MAMuZeroBaselineModel | A 共享头 |
| `no_belief` | 同上 | NoBeliefBaselineModel | C / Abl7 |
| `rewardhead_explicit_type` | 同上 | ExplicitTypeRewardBaselineModel | A / Abl6.x |
| **`external_mappo`** | 同上 | MAPPOAlgorithm | external policy gradient |
| **`external_qmix`** | 同上 | QMIXAlgorithm | external value decomposition |
| **`external_ma_muzero_gh`** | 同上 | MAMuZeroGHAlgorithm | external model-based MARL |
| `external_mamba` | 同上（若 sourced）| MAMBAAlgorithm | model-based + belief（Tier-2）|
| `external_marie` | 同上 | NotImplementedError | stub（保留 CLI）|
| `external_ga` | 同上 | NotImplementedError | stub（保留 CLI）|

### 3.3 框架澄清③：CLI ↔ factory-arg ↔ model-class 三列映射表（继承 pkg-06 §3.3 + 扩展 external namespace）

> pkg-06 design.md §3.3 锁定的 CLI 三列表是 Day-1 acceptance #4；pkg-07 继承该契约并扩展到 11-key REGISTRY + 3 curriculum-override (= 14 行)。pkg-06 的前缀约定保留：`baseline_*` 前缀标记 internal-with-shared-backbone；bare 名标记 ablation-only internal；新增 `external_*` 前缀标记 external namespace。`train_main.py` `_DEFERRED_VARIANTS` (line 45-52) 是 ground truth，CLI 字串与之逐字对账。

| `--variant` CLI 字串 | `create_baseline(cfg, ?)` 第 2 参 | 模型类 / 运行体 | 断言 / 用途 |
|---|---|---|---|
| `hyper` | — (不进工厂；`HyperMuZeroModel(cfg)` 直接构造) | `HyperMuZeroModel` | A / B′ / C 主线 |
| `oracle_only` | — (不进工厂；`HyperMuZeroModel(cfg)` + `curriculum_stage_1_end_frac=1.0`) | `HyperMuZeroModel` (curriculum override) | regret 上界 |
| `infer_only` | — (不进工厂；`HyperMuZeroModel(cfg)` + `curriculum_stage_1_end_frac=0.0`) | `HyperMuZeroModel` (curriculum override) | 零样本 belief 推断 |
| `baseline_input_wide` | `"input_wide"` | `InputWideBaselineModel` | B′ 等参 (internal, 有 `baseline_` 前缀) |
| `baseline_input_deep` | `"input_deep"` | `InputDeepBaselineModel` | B′ 等参 (internal, 有 `baseline_` 前缀) |
| `baseline_ma_muzero` | `"ma_muzero"` | `MAMuZeroBaselineModel` | A 共享头 (internal, 有 `baseline_` 前缀) |
| `no_belief` | `"no_belief"` | `NoBeliefBaselineModel` | C / Abl7 (internal, **bare 无前缀**) |
| `rewardhead_explicit_type` | `"rewardhead_explicit_type"` | `ExplicitTypeRewardBaselineModel` | A / Abl6.x (internal, **bare 无前缀**) |
| `external_mappo` | `"external_mappo"` | `MAPPOAlgorithm` (`ExternalBaselineRunner`) | external PG (Tier-1) |
| `external_qmix` | `"external_qmix"` | `QMIXAlgorithm` (`ExternalBaselineRunner`) | external Q-decomp (Tier-1) |
| `external_ma_muzero_gh` | `"external_ma_muzero_gh"` | `MAMuZeroGHAlgorithm` (`ExternalBaselineRunner`) | external model-based MARL (Tier-1) |
| `external_mamba` | `"external_mamba"` | `MAMBAAlgorithm` if sourced; else 工厂抛 `NotImplementedError` | model-based + belief (Tier-2) |
| `external_marie` | `"external_marie"` | 工厂抛 `NotImplementedError` (stub) | 保留 CLI / registry 占位 |
| `external_ga` | `"external_ga"` | 工厂抛 `NotImplementedError` (stub) | 保留 CLI / registry 占位 |

**前缀约定脚注**：pkg-06 5-variant 表使用「`baseline_` 前缀 → 共享 backbone 变体；bare 名 → ablation-only 变体」的不对称约定（asymmetry intentional）。pkg-07 逐字继承，**并新增 `external_` 前缀** 标记 external namespace。工厂第 2 参（factory arg）**保留前缀**于 external（`"external_mappo"` 而非 `"mappo"`），**剥前缀**于 internal-with-shared-backbone（`"input_wide"` 而非 `"baseline_input_wide"`），以保证 union return type 在工厂内可消歧。spec 01 单测 `test_registry_keys_equal_cli_choices_and_factory_args` 强制 CLI ↔ factory-arg 派生函数 round-trip 正确。

### 3.4 框架澄清④：External Vendoring 来源选定

| External | 来源 | 选源理由 | 许可 |
|----------|------|---------|------|
| MAPPO | `D:\RL\lzj\MAPPO\` | 本地已有 + 离散动作 friendly + 与 NS-MMG-style env 兼容 | 本地用户代码（无外部许可问题）|
| QMIX | PyMARL `oxwhirl/pymarl` | 官方 SOTA reference impl + 与 PettingZoo 兼容 | Apache 2.0 |
| MA-MuZero-GH | `werner-duvaud/muzero-general` + thin MA wrapper | 单 agent 干净 + 移植成本可控（thin wrapper) | MIT |
| MAMBA | active sourcing（spec 06 §3 协议）| 原论文 repo + 社区 fork | 待 sourcing |
| MARIE / GA | stub | 不进 main table | N/A |

**选源 spec 04/05/06 锁定**：每个 vendoring 记录 commit hash + 许可文件路径,实施期不漂移。

### 3.5 框架澄清⑤：N-parametric Adapter + 两 flag 信息门控（PettingZoo 适配器架构 keystone）

**N-parametric 契约**：ResourceCommonsEnv 是 N-parametric (`env.py:84 self.N = cfg.N`，Easy=2 / Medium=4 / Hard=8 per `configs/presets/`)。适配器 **不得** 硬编码 4 agents，必须从 `env_cfg.N` 读：

```python
self._N = self._env.N
self.agents = [f"agent_{i}" for i in range(self._N)]
```

所有 `range(...)` 与 obs/action dict 索引同样以 `self._N` 参数化。C7-EXT-SMOKE1 在 Easy（N=2）与 Medium（N=4）双 preset 跑 smoke。

**两 flag 信息门控**：`ResourceCommonsEnv` 在 info dict 中暴露三套字段（env.py:308-323），且自带 schema-marker tuples `_oracle_fields` / `_eval_only_fields`。适配器读这两个 marker（DRY 抗 env 演化）+ 两 flag 独立分别 gate：

| 键 | 类别 | env schema marker | gate flag | 默认 external 可见？ |
|----|------|-------------------|-----------|---------------------|
| `caps` | Public（per-agent capability vector）| — | — | ✅ 是（agent 看自己合法）|
| `deltas`, `step_idx`, `harvests` | Public | — | — | ✅ 是 |
| `c_true` | Oracle（真实 c）| `_oracle_fields` | `oracle_mode` | ❌ 否 (`oracle_mode=False` 默认) |
| `types` | Oracle（每 agent 的 α/β type）| `_oracle_fields` | `oracle_mode` | ❌ 否 (`oracle_mode=False` 默认) |
| `hotspot_centers` | Eval-only（spawn distribution 参数）| `_eval_only_fields` | `eval_info_mode` | ❌ 否 (`eval_info_mode=False` 默认) |
| `resource_state` | Eval-only（每 resource 精确位置+stock）| `_eval_only_fields` | `eval_info_mode` | ❌ 否 (`eval_info_mode=False` 默认) |
| `_oracle_fields`, `_eval_only_fields`, `_info_schema_version` | schema 元数据 | — | 始终 strip | ❌ 否（防 introspection 泄漏结构）|

**Flag 语义**：
- `oracle_mode=True` 仅限**内部 hyper 评估**（注入 oracle context 时）。external runner 训练**绝不**翻 True。
- `eval_info_mode=True` 仅限 **Pkg-08 evaluator 的 metric-collection 路径**（只读，绝不喂回 policy/critic）。两 flag 独立正交。

**CTDE 合法性边界（footnote）**：
- **LEGAL global state**（中心 critic / mixer 可消费）= `concat([obs_i for i in agents])` ± action one-hots ± `caps`（public）。
- **PRIVILEGED 状态**（必须 gated）= `c_true` / `types` / `resource_state` / `hotspot_centers`。
- **规则**：external baseline 需要 centralized critic 时，**只能** 从 `concat(obs)` 派生 global state；**不可** 触碰 `info["resource_state"]` 或 `info["hotspot_centers"]`。适配器默认 enforce；唯一例外 = Pkg-08 evaluator 的 `eval_info_mode=True` 通道（只读 metric collection）。

强制单测 `test_adapter_info_gating.py`：4 字段 × 2 flag 正交 × `{reset, step}` 双路径 × `{Easy, Medium}` N-parametric。

---

## 4. 10 项 Decisions

### D1: spec 数量与文档结构

**Decision**: 8 specs 对称（与 Pkg-03/04/05/06 对称）

**Specs**:
1. `01-baseline-registry-and-cli.md` — 工厂 + REGISTRY + CLI 映射
2. `02-shared-backbones-internal.md` — 继承 pkg-06 spec 02
3. `03-internal-variants.md` — 合并 pkg-06 specs 03+04+05
4. `04-pettingzoo-adapter.md` — 架构 keystone
5. `05-external-mappo.md` — MAPPO Tier-1
6. `06-external-qmix-mamuzero-mamba.md` — QMIX/MA-MuZero/MAMBA + MARIE/GA stub
7. `07-fairness-protocol.md` — 双层 fairness
8. `08-integration-contracts.md` — 对外硬契约 + 下游补丁声明

**Rationale**: 与 pkg-06 8 specs 一致便于交叉对比;internal/external 互不污染（spec 02/03 内 / spec 04/05/06 外 / spec 01/07/08 共）;spec 07 公平协议双层是合并写不是分开,避免 internal 与 external 公平契约割裂理解。

### D2: 工厂归属与签名（supersede pkg-06）

**Decision**: 
- 工厂位置：`hyper_mve/baselines/__init__.py::create_baseline(cfg, variant)`
- pkg-06 `create_baseline_model(cfg, variant)` 保留为 alias + `DeprecationWarning`
- Pkg-05 spec08 §3.1 line 136 引用的旧名通过 alias 兼容（不改 Pkg-05 SDD）

**Rationale**: rename 必要因为 internal-only → internal+external 语义扩展;alias 防止破坏 Pkg-05 spec08 §3.1 引用稳定性;DeprecationWarning 给未来一次 release 的过渡窗口。

### D3: Internal vs External 分派（架构核心）

**Decision**: 两 namespace 分派,返回 union type
- `internal/*` → 返回 `BaselineModel`（继承自 HyperMuZeroModel 或 share parts）, 实现 7-API,被 `MuZeroTrainer` 消费
- `external/*` → 返回 `ExternalBaselineRunner` Protocol 实例,自带 `train()/evaluate()/save_checkpoint()/load_checkpoint()`
- 工厂收 `"hyper"` 抛 `ValueError`（继承 pkg-06 D3)

**Rationale**: 强行让 external 实现 7-API 是范式错位,损失公平性；让 internal 用 external Protocol 又丢失共享 trainer 的复用价值。两 namespace 是诚实的折中。

### D4: 共享后端粒度（继承 pkg-06 D4,仅 internal 用）

**Decision**: 5 internal variant 各自实例化 RepNet/BeliefNet/TriContextEncoder（**同类同构、独立实例、独立梯度、参数量逐位对齐**）。External 不消费 shared_backbones。

**Rationale**: 继承 pkg-06 D4。External 与 hyper 没有共同的"条件化子系统"，强行共享 backbone 反而引入耦合。

### D5: 双层 Fairness 协议（**新增双层**）

**Decision**: 
- **Internal 严格 fairness**（继承 pkg-06 D5）：仅条件化消费子系统统计 + 双阈值（warn ≤5% / fail ≤10%）+ ma_muzero/explicit_type 结构性豁免
- **External 披露式 fairness**：报告 (参数量,wall-clock-to-converge,LR-swept best LR,final return ≥5 seeds);LR sweep ≥3 LR × ≥3 seeds;Methods 表三列披露（params/walltime/return-X）

**Rationale**: External 没有可对齐的"条件化子系统",强行对齐毫无意义。披露式让审稿人看到所有维度,自行判断公平度（标准做法）。Internal 保持严格不放松（断言 B′ 生死线）。

### D6: Belief 门控一致性（继承 pkg-06 D6,仅 internal）

**Decision**: 5 internal variant `update_step` 共用同一 `BeliefGradGating` 类（pre-5K 梯度=0,post-5K 解锁）

**Rationale**: 继承 pkg-06 D6。external 不消费 BeliefNet,不在此约束内。

### D7: Tier-1 External 选型 + 适配器契约

**Decision**: 3 Tier-1 = MAPPO + QMIX + MA-MuZero-GH（vendoring 来源见 §3.4）。所有 external 通过 **N-parametric** `ResourceCommonsPettingZooEnv` 消费 env（agent IDs = `agent_0..agent_{N-1}` where N = `env_cfg.N`；详见 §3.5）。**两 flag 信息门控**：`oracle_mode=False` AND `eval_info_mode=False` 默认,覆盖 4 leak-surface 字段（`c_true`/`types`/`hotspot_centers`/`resource_state`）+ schema-marker strip。强制单测 `test_adapter_info_gating.py` 覆盖 `{reset,step}` × 4 字段 × 2 flag 正交 × `{Easy,Medium}` N-parametric。

**External 训练流程**（vs internal）:

```
Internal (5 variant):
  cfg → create_baseline(cfg, internal_*) → BaselineModel
       → MuZeroTrainer(cfg, BaselineModel) → 同一 Worker/Buffer/compose_total_loss

External (3 Tier-1):
  cfg → create_baseline(cfg, external_*) → ExternalBaselineRunner
       → runner.train(cfg, env_fn=ResourceCommonsPettingZooEnv(env_cfg)) → 自带 trainer/buffer/loss
  
  Eval 统一:
  internal & external runner 都暴露 .evaluate(env_fn, c_grid, episodes) -> EvalReport
  → Pkg-08 unified evaluator 调度无差异
```

**Rationale**: 3 Tier-1 覆盖 MARL 三大主流范式（PG / Q-decomp / model-based）;PettingZoo 适配器单点封装,降复杂度。

### D8: Tier-2 MAMBA Sourcing 协议

**Decision**: 
1. Day-1: 在 spec 06 §3 列出搜索目标（原论文 repo,Google scholar 引用回链,社区 fork 关键词）
2. Day-2: 实际搜索 GitHub + arxiv-sanity,记录到 `spec 06 §3 sourcing log`
3. 若 2 日内找到可移植 source（minimal port < 1000 LOC）→ vendor 进 pkg-07
4. 若失败 → 转 stub（`NotImplementedError`）保留 `external_mamba` CLI 名 + 搜索日志写进 spec 06 §3 供未来重启
5. **不阻塞 pkg-07 SDD finalize**

**Rationale**: 主动 sourcing 比假设找不到诚实;失败 fallback 不阻塞包出。

### D9: MARIE / GA stub 策略

**Decision**: 
- spec 01 registry 列 `external_marie` / `external_ga` 两键,工厂返回 `NotImplementedError` 抛出
- spec 06 epilogue 声明：MARIE/GA 不进 main table,不阻塞 pkg-07/08 包出;未来若做 pkg-07.5 follow-up 在那里 promote
- 强制单测 `test_stub_external_baselines.py` 验证抛出符合预期

**Rationale**: 保留 CLI 名 + 占位是诚实的"知道但暂不做"标记;`NotImplementedError` 是 Python 习惯。

### D10: cfg 新字段归属

**Decision**: 引入 `cfg.baselines: BaselinesConfig` 新 dataclass（替代 pkg-06 提议的 `cfg.model.baseline_*`）。字段:

```python
from types import MappingProxyType
from typing import Mapping
from dataclasses import dataclass, field

@dataclass(frozen=True)
class BaselinesConfig:
    """5-field exhaustive cfg namespace (4 internal from pkg-06 D10 + 1 external new).
    
    Per-impl tuning constants (external_smoke_max_env_steps / external_mappo_share_policy /
    external_qmix_mixer_hidden_dim / external_ma_muzero_gh_simulations) live in spec 05/06
    defaults — they are impl-internal, not cross-spec contract.
    """
    # Internal (4, inherited from pkg-06 D10)
    internal_wide_hidden_dim: int = 512
    internal_deep_layers: int = 8
    internal_ma_muzero_share_pred_head: bool = True
    internal_explicit_type_branches: int = 2          # α / β
    
    # External (1, new) — per-baseline LR sweep mapping (NOT a single shared tuple).
    external_lr_sweep_grid: Mapping[str, tuple[float, ...]] = field(
        default_factory=lambda: MappingProxyType({
            "external_mappo":         (1e-4, 3e-4, 1e-3),
            "external_qmix":          (1e-4, 3e-4, 1e-3),
            "external_ma_muzero_gh":  (1e-4, 3e-4, 1e-3),
            # external_mamba added at sourcing time; external_marie/ga not swept (stubs)
        })
    )
```

挂在 **`cfg.baselines`** 作为 V4Config 顶层新增 sub-config（与 `cfg.env/model/train/mup/eval/legacy` 平级；V4Config 无 `.v4` 中间层 — verified `configs/v4_config.py:23-37`）。**消费态**：本包仅声明字段,实施期由 Pkg-01 spec05 同步消费。

**Rationale**: 新 namespace 防止 `cfg.model.*` 污染；**5 字段穷举**（pkg-06 的 4 + 1 external sweep grid mapping），无 hardcoded 默认。Per-impl tuning constants 留在 spec 05/06 内部 defaults — cfg.baselines 字段是「跨 spec 共享契约」，不是「所有可调参数的字典」。

---

## 5. Decision Acceptance（Day 1 HARD GATE **11 项**）

| # | 验收项 | Decision 落点 | 检测 |
|---|--------|--------------|------|
| 1 | Internal 5 variant 核心区别表（继承 pkg-06 D7）| §3.2 11-key 工厂矩阵的 internal 行 | spec 03 等参账目 |
| 2 | External 3 Tier-1 baseline 选型 + 选源记录 | §3.4 vendoring 表 + D7 | spec 04/05/06 vendoring 计划 |
| 3 | Tier-2 (MAMBA) sourcing 协议 + 失败 fallback | D8 | spec 06 §3 sourcing log |
| 4 | MARIE/GA stub 策略 + 不进 main table 声明 | D9 | spec 06 epilogue + 单测 |
| 5 | 双层 fairness 单元定义（Internal 严格 vs External 披露式）| D5 | spec 07 § 双段 |
| 6 | PettingZoo 适配器**两 flag** 信息门控 + 4 leak-surface + CTDE 合法/特权边界 + N-parametric | §3.5 + D7 + adapter | spec 04 + 单测 `test_adapter_info_gating.py` |
| 7 | mermaid:external 训练流程 vs internal MuZeroTrainer 流程对比图 | D7 § "External 训练流程" 块 | design.md §7 mermaid |
| 8 | cfg 新字段**5 字段穷举**（`cfg.baselines.*`，Mapping[str, tuple] LR sweep）| D10 dataclass | spec 01 + Pkg-01 spec05 同步 5 keys |
| 9 | **8-spec** ref_matrix 预期表（per-spec 引用矩阵，不是 variant 矩阵）| §8 | scripts/check_ref_matrix.ps1 真跑 |
| 10 | 10 天日历 + 响应式 SLA + Phase 8 真跑 | README §"实施顺序" | Day 8 ref_matrix [PASS] |
| **11** | **CLI ↔ factory-arg ↔ model-class 三列映射表（11 in-registry + 3 curriculum-override = 14 行）+ 前缀约定脚注（继承 pkg-06 §3.3）** | §3.3 | spec 01 单测 `test_registry_keys_equal_cli_choices_and_factory_args` |

---

## 6. 10 天日历（详见 [README §"实施顺序"](./README.md)）

| Day | 文档 | 审阅点 |
|-----|------|--------|
| 1 | README + proposal + design.md（**HARD GATE**）| 11 项 §5 验收清单 |
| 2 | spec 01（registry/CLI）+ spec 04（PettingZoo adapter）| 同步 pkg-08 spec 01 |
| 3 | spec 02（shared backbones）+ spec 03（internal variants 合并）| 继承 pkg-06 内容 |
| 4 | spec 05（MAPPO）| port 计划 + CTDE 审阅 |
| 5 | spec 06（QMIX + MA-MuZero-GH + MAMBA sourcing log）| vendoring + 许可 |
| 6 | spec 07（fairness）| 双层公平协议双段 |
| 7 | spec 08（integration contracts）| 下游补丁声明 |
| 8 | `check_ref_matrix.ps1` + `ref_matrix.csv` 真跑 | M6 引用对账 |
| 9 | PR_DESCRIPTION.md | 提交准备 |
| 10 | 用户最终 ack | finalize |

**响应式 SLA**: Day 1 hard gate 不过 → 全流程顺延 1 天;每 spec 当天末交审,审阅往返不计入"天"。

---

## 7. Mermaid: Internal vs External 训练流程对比（Day 1 HARD GATE #7）

```mermaid
flowchart TB
    subgraph Internal["Internal Baseline (5 variant + hyper)"]
        cfg1[V4Config] --> factory1[create_baseline / HyperMuZeroModel直接构造]
        factory1 --> model[BaselineModel / HyperMuZeroModel]
        model --> trainer[MuZeroTrainer<br/>shared, pkg-05]
        trainer --> worker[Worker<br/>shared, pkg-05]
        worker --> buffer[EpisodeReplayBuffer<br/>shared, pkg-05]
        buffer --> loss[compose_total_loss<br/>shared, pkg-05]
        loss -.update.-> model
        worker --> envI[ResourceCommonsEnv<br/>raw, pkg-02]
    end
    
    subgraph External["External Baseline (3 Tier-1 + Tier-2 MAMBA + 2 stubs MARIE/GA)"]
        cfg2[V4Config] --> factory2[create_baseline]
        factory2 --> runner[ExternalBaselineRunner<br/>自带 trainer/buffer/loss<br/>stubs raise NotImplementedError]
        runner --> adapter[ResourceCommonsPettingZooEnv<br/>N-parametric (env_cfg.N)<br/>oracle_mode=False, eval_info_mode=False<br/>spec 04]
        adapter --> envE[ResourceCommonsEnv<br/>raw, pkg-02]
    end
    
    subgraph UnifiedEval["统一 Eval (pkg-08)"]
        model --> eval[evaluate<br/>env_fn, c_grid, episodes<br/>→ EvalReport]
        runner --> eval
        eval --> report[EvalReport<br/>统一 schema]
    end
    
    style adapter fill:#ffe4b5,stroke:#ff8c00
    style eval fill:#90ee90,stroke:#006400
```

---

## 8. ref_matrix 预期表（Day 1 HARD GATE #9）

> **澄清**：本节是 **8-spec 引用矩阵**（per-spec ↔ per-spec），**不是** variant 矩阵。Variant 清单见 §3.2 11-key 工厂矩阵。

| Spec | 期望引用（spec 0X） |
|------|--------------------|
| spec 01 | 02, 03, 04, 05, 06, 08 |
| spec 02 | 03, 08 |
| spec 03 | 02, 07, 08 |
| spec 04 | 05, 06, 08 |
| spec 05 | 04, 07, 08 |
| spec 06 | 04, 07, 08 |
| spec 07 | 03, 05, 06, 08 |
| spec 08 | 01, 02, 03, 04, 05, 06, 07（全 7 引）|

Day 8 真跑 `pwsh scripts/check_ref_matrix.ps1` 验证 delta 全空。

---

## 9. 关键决议（用户 ack 锁定）

| Q | 决议 | 影响位置 |
|---|------|---------|
| **Q1** SDD 与 pkg-06 关系 | pkg-07 supersedes pkg-06,继承 5 internal variant + C6-* 重号为 C7-INT-*;pkg-06 README 加 banner | spec 01 头部 |
| **Q2** Tier-1 选型 | MAPPO（lzj-port）+ QMIX（PyMARL vendor）+ MA-MuZero-GH（muzero-general + MA wrapper）| §3.4 + spec 05/06 |
| **Q3** Tier-2 / Stub 策略 | MAMBA 主动 sourcing（2 日 + fallback stub）;MARIE/GA `NotImplementedError` | §D8/D9 + spec 06 |
| **Q4** Fairness 协议 | 双层:Internal 严格继承 pkg-06 / External 披露式 | §D5 + spec 07 |
| **Q5** Adapter 架构 | **N-parametric** `ResourceCommonsPettingZooEnv`（agent IDs from `env_cfg.N`）+ **两 flag** 信息门控（`oracle_mode` + `eval_info_mode`）+ CTDE 合法/特权状态边界 + 强制 `test_adapter_info_gating.py` | §3.5 + spec 04 |
| **Q6** cfg namespace | `cfg.baselines.*` 新 dataclass（**5 字段穷举**,LR sweep 用 `Mapping[str, tuple]`,消费态）| §D10 + spec 01 |
| **Q7** 下游补丁 | 3 处实施期代码微补丁（Pkg-02 obs-mask + Pkg-05 planner flag + Pkg-05 CLI 扩 +6 个 `external_*` 名）,**不动 Pkg-01..05 SDD** | spec 08 显式声明 |
| **Q8** CLI 映射 | **CLI ↔ factory-arg ↔ model-class 三列表** 14 行（11 in-registry + 3 curriculum-override）+ 前缀约定脚注（继承 pkg-06 §3.3）| §3.3 + spec 01 |
| **Q9** 量词 canonical | REGISTRY = 11 keys；Methods 主表 = 9 列（hyper + 5 internal + 3 Tier-1，MAMBA-if-sourced 为第 10）；§8 是 8-spec ref_matrix（非 variant） | §3.1 + §3.2 + §8 |

---

## 10. 用户 Day 1 HARD GATE 审阅清单（**11 项**）

请用户在 design.md 审阅时勾选:

- [ ] §3.2 11-key REGISTRY 分层区别清晰（5 internal / 3 Tier-1 / MAMBA / 2 stubs；hyper/oracle_only/infer_only 不进工厂）
- [ ] §3.3 CLI ↔ factory-arg ↔ model-class 三列映射表 14 行（11 in-registry + 3 curriculum-override）+ 前缀约定脚注可对账 `train_main.py:45-52 _DEFERRED_VARIANTS`
- [ ] §3.4 External vendoring 选源记录可追溯
- [ ] §D8 MAMBA sourcing 协议有失败 fallback
- [ ] §D9 MARIE/GA stub 策略合理（不进 main table）
- [ ] §D5 双层 fairness 协议口径认可
- [ ] §3.5 **两 flag** 信息门控架构（`oracle_mode=False` AND `eval_info_mode=False` 默认，4 leak-surface 全覆盖，schema-marker strip，CTDE 合法/特权状态边界，N-parametric agents）认可
- [ ] §7 mermaid 流程图正确反映 Internal vs External 区别（含 stubs）
- [ ] §D10 cfg.baselines.* **5 字段穷举**（`Mapping[str, tuple]` 类型，挂 `cfg.baselines` 无 .v4 层）合理
- [ ] §8 **8-spec** ref_matrix 预期表合理（per-spec ↔ per-spec，非 variant 矩阵；Day 8 真跑）
- [ ] §6 10 天 SDD 日历 + 响应式 SLA 合理

**Day 1 HARD GATE 全过 → 放行进 Day 2 spec 01 + spec 04 起草。**
**任一项未过 → 顺延 1 天修订 design.md,不带病进 Day 2。**
