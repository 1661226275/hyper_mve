# Pkg-02: ResourceCommons Environment

> **状态**：Draft — awaiting user review
> **包 ID**：`pkg-02-resource-commons-env`
> **工期**：1.5 周（10 天） · **GPU 算力**：5 小时（单测 + 1000 episode smoke） · **PR 体量**：12 新文件 + 3 集成测试

---

## 📚 文档导览

| 文档 | 目的 | 字数 |
|------|------|------|
| [`proposal.md`](./proposal.md) | **先读这个**：Why / What Changes / Capabilities / Impact | ~3500 |
| [`design.md`](./design.md) | 设计抉择：Context / Goals / 8 项 Decisions | ~4500 |
| [`specs/01-env-formal-tuple.md`](./specs/01-env-formal-tuple.md) | Ch3.2 14 分量形式化 + state 内部表示 | ~1500 |
| [`specs/02-resource-dynamics.md`](./specs/02-resource-dynamics.md) | 公式 3.1-3.4：logistic 再生 + neighbor factor + α(c) | ~2000 |
| [`specs/03-fehr-schmidt-reward.md`](./specs/03-fehr-schmidt-reward.md) | 公式 3.5-3.10：type α/β reward + φ(c) 调制 | ~2200 |
| [`specs/04-six-block-observation.md`](./specs/04-six-block-observation.md) | Ch3.7 六块观测 + Self-Info + FOV 过滤 | ~2000 |
| [`specs/05-context-evolution.md`](./specs/05-context-evolution.md) | 三模式 c_t (static/oscillate/random_walk + shock) | ~1500 |
| [`specs/06-patchy-resource-spawn.md`](./specs/06-patchy-resource-spawn.md) | Ch3.3.2 hotspot 资源生成 | ~1200 |
| [`specs/07-difficulty-presets-env.md`](./specs/07-difficulty-presets-env.md) | Easy/Medium/Hard env 配置实现 | ~1500 |
| [`specs/08-gym-api.md`](./specs/08-gym-api.md) | gym.Env 接口 + **info dict v4 Oracle 信号**（关键） | ~2000 |

---

## 🎯 一句话目标

**实现首个同时支持"关系动态 + 偏好异质 + φ(c) 调制 Fehr-Schmidt"的混合动机 MARL 环境**——ResourceCommons，匹配 Ch3 完整规范，为 Pkg-03/04/05 提供 env 接口，**暴露 v4 Oracle 监督信号**（c_true / types / caps）以支撑 BeliefNet 训练。

---

## ✅ 关键 Acceptance Criteria（速览）

### 必通过（数值硬约束）

- [ ] **Table 3.5.4 四场景 100% 数值匹配**（type β reward 在 (c, Δ) 四象限）
- [ ] **Ch4.1.1 偏导表四象限 100% 数值匹配**（自动微分提取 ∂R^β/∂u_i ∈ {0.7, 2.0, 1.3, 0.0}）
- [ ] **Ch3.3-3.4 资源动力学公式**实现与解析解误差 ≤ 1e-4
- [ ] **三 preset 字段与 Ch3.9 Table 100% 匹配**（继承 Pkg-01 V4Config）
- [ ] **env.info 必含 `c_true`, `types`, `caps` 三 Oracle 字段**（v4 BeliefNet 监督训练前提）
- [ ] `python scripts/test_resource_commons.py` 1000 random episode 无 NaN/Inf

### 性能要求

- [ ] 单线程 CPU ≥ 1000 step/s
- [ ] 1000 episode smoke test < 5 分钟
- [ ] GPU memory 0 占用（env 纯 CPU）

### 期望达到

- [ ] `pytest tests/envs/` 覆盖率 ≥ 85%
- [ ] `mypy --strict hyper_mve/envs/resource_commons/` 零错误
- [ ] Proposition 3.1（Pareto-Nash gap 单调）实证图

---

## 📅 实施顺序（10 天）

| 阶段 | 天数 | 任务 | 输出 |
|------|------|------|------|
| **Phase 1** | Day 1-2 | `spaces.py` + reset/step 骨架；写 specs/01 + specs/08 | env 可实例化，返回零观测 |
| **Phase 2** | Day 3-4 | `dynamics.py` + `spawn.py`（公式 3.1-3.4）；写 specs/02 + specs/06 | 单点资源再生解析解通过 |
| **Phase 3** | Day 5-6 | `rewards.py`（公式 3.5-3.10 + Table 3.5.4 单测）；写 specs/03 | 四象限偏导测试通过 |
| **Phase 4** | Day 7 | `observations.py` + Self Info；写 specs/04 | 六块布局与 Pkg-01 自洽 |
| **Phase 5** | Day 8 | `context_evolution.py` 三模式；写 specs/05 | 三 c_mode 切换测试 |
| **Phase 6** | Day 9 | preset 集成 + info dict 完整化；写 specs/07 | 三 preset env 跑通 |
| **Phase 7** | Day 10 | smoke test + Proposition 3.1 实证 + PR | 1000 episode pass |

---

## 🔑 8 项关键 Decisions（速览）

| # | Decision | 推荐 | 详见 |
|---|----------|------|------|
| D1 | 资源点坐标存储 | `np.array shape=(K, 3) = (x, y, q)` | [design §3 D1](./design.md) |
| D2 | FOV 过滤实现 | 每步 Chebyshev 全扫描 | [design §3 D2](./design.md) |
| D3 | Fehr-Schmidt Δ 计算粒度 | 全 agent 向量化 + type mask | [design §3 D3](./design.md) |
| D4 | 浮点稳定性 | float32 + clip(0, Q_max) 每步 | [design §3 D4](./design.md) |
| D5 | Context 模式切换 | 构造参数固定 + reset 参数指定初值 | [design §3 D5](./design.md) |
| D6 | Patchy 资源是否每 reset 重生成 | 每 reset 重生成（with seed） | [design §3 D6](./design.md) |
| D7 | Move 失败处理 | 留在原地 | [design §3 D7](./design.md) |
| D8 | HARVEST 目标资源 | 仅当前格子 | [design §3 D8](./design.md) |

---

## 🚨 v4 关键约束（Ch4 反向要求 env）

| Ch4 章节 | 对 env 的硬约束 |
|----------|----------------|
| **Ch4.5.1 L_c oracle 监督** | `env.step()` 返回的 info **必须**含 `c_true: float`（真实 c_t） |
| **Ch4.5.2 L_opp oracle 监督**（v4 关键差异） | info **必须**含 `types: tuple[AgentType, ...]`（**全部 N 个 agent 真实类型**，不只自己） |
| **Ch4.2.2 type_emb 来源** | info 必须含 `caps: tuple[CapabilityVector, ...]`；observation 中 type 块**仅自己 2 维**（Self Info） |
| **Ch4.6.5 课程稳定性** | info schema 必须**稳定不变**（无 stage 切换；stage 切换在 Pkg-05 trainer 层） |
| **Ch4.4 客观-主观分工** | env reward 已按公式 3.10 **分类型计算**（type α / β 各自 R），存进 buffer 后 Pkg-04 RewardHead 直接预测 |

**最关键**：v4 是 **Oracle 监督 BeliefNet**——`env.info["types"]` 暴露**全部 N 个真实类型**，用于 Pkg-05 trainer 计算 L_opp 监督信号。这是与 v3 自监督的本质差异。

---

## ❓ Open Questions（待用户确认）

| # | Question | 默认 |
|---|----------|------|
| Q1 | env step 返回的 reward 是 `np.array shape=(N,)` 还是 `dict[int, float]`？ | **np.array (N,) float32**（与 Pkg-01 TimeStepRecord.r 一致） |
| Q2 | info["types"] 类型用 `tuple[AgentType, ...]` 还是 `np.ndarray int8`？ | **np.ndarray int8 shape=(N,)**（serializable + Pkg-05 trainer 直接消费） |
| Q3 | 是否在 info 暴露 `hotspot_centers` / 资源完整状态用于评估可视化？ | **是**，加入 info（仅 evaluator 用，trainer 不读） |
| Q4 | env 是否提供 `set_seed()` 独立 API 还是仅通过 `reset(seed=...)`？ | **仅通过 reset(seed=...)**（gym 标准） |
| Q5 | neighbor 块的 9 维（含 presence_flag）vs Ch3.7 文档的 8 维差异是否需要在 spec 中正式说明？ | **是**，spec 04 中显式标注为"v4 实施 augment"，论文 Ch3.7 末段加 footnote |

---

## 📦 输出清单（PR 时检查）

### 新增（12 文件）

```
hyper_mve/envs/resource_commons/
├── __init__.py
├── env.py                    # ResourceCommonsEnv (gym.Env)
├── dynamics.py               # 公式 3.1-3.4
├── rewards.py                # 公式 3.5-3.10 (Fehr-Schmidt)
├── observations.py           # 六块构造 + FOV 过滤
├── context_evolution.py      # 三 c_mode
├── spawn.py                  # patchy hotspot 生成
└── spaces.py                 # gym.spaces 构造
```

### 测试（≥4 文件）

```
tests/envs/
├── __init__.py
├── test_dynamics.py          # 公式 3.1-3.4 单测
├── test_rewards.py           # Table 3.5.4 + Ch4.1.1 偏导
├── test_observations.py      # 六块 + FOV + Self-Info
├── test_context_evolution.py # 三 c_mode
├── test_spawn.py             # patchy 生成
├── test_env_info_oracle.py   # v4 关键: info 含 c_true/types/caps
└── test_env_integration.py   # 端到端
```

### 脚本（2 文件）

```
hyper_mve/scripts/
├── test_resource_commons.py  # 1000 episode smoke
└── proposition_3_1_validation.py  # Pareto-Nash gap 实证
```

---

## 🔗 上下游关联

| 关系 | 包 | 接口 |
|------|-----|------|
| **本包 ← Pkg-01** | Foundation Schema | `EnvConfig`, `AgentType`, `CapabilityVector`, `ObservationLayout`, `V4Config.from_preset()` |
| **本包 → Pkg-03** | BeliefNet | env.info Oracle 信号（c_true / types） |
| **本包 → Pkg-04** | Model | env.observation_space.shape → RepNet 输入维度 |
| **本包 → Pkg-05** | Trainer | env step API + info dict 全字段 |
| **本包 → Pkg-06a/b** | Baselines | 全部 baseline 接同一 env |
| **本包 → Pkg-08** | Experiments | preset 选择驱动实验配置 |

---

## 📖 引用源

- 项目 Plan File: `Part 2 Pkg-02` + `Part 9.B Pkg-02 SDD 计划`
- Hyper-MuZero v4 Roadmap §4 Stage 1
- Chapter 3 Environment v4: `D:\RL\docs\Chapter3_Environment_v4.md` 全章
- Chapter 4 Architecture v4: `D:\RL\docs\Chapter4_Architecture_v4.md` §4.5.1 / 4.5.2（Oracle 监督要求）
- Pkg-01 SDD: `sdd/pkg-01-foundation-schema/` 全部 schema

---

## 🚦 Review Checklist（用户审阅时勾选）

- [ ] proposal.md 读完，**Why** 充分（ResourceCommons 是 v4 唯一支撑 4 个断言的环境）
- [ ] design.md 读完，**8 项 Decisions** 中无反对意见
- [ ] specs/01-08 抽查至少 3 个（重点 specs/03 + specs/08，对应断言 A 物理基础 + v4 Oracle 信号）
- [ ] 10 天实施顺序合理（Phase 3 数值验证是 hard gate）
- [ ] Open Questions Q1-Q5 默认决定 OK
- [ ] v4 Oracle 信号清单（c_true / types / caps）认可
- [ ] neighbor 块 9 维（含 presence_flag）vs Ch3.7 文档 8 维的差异接受
- [ ] 出包后可启动 Pkg-03 (BeliefNet + TriContextEncoder)
