# Pkg-02: ResourceCommons Environment — Proposal

| 元信息 | 值 |
|--------|----|
| **包 ID** | `pkg-02-resource-commons-env` |
| **状态** | Draft, awaiting review |
| **工期估计** | 1.5 周（10 天） |
| **GPU 算力** | 5 小时（单测 + 1000 random episode smoke） |
| **依赖** | Pkg-01 (Foundation Schema) |
| **被依赖** | Pkg-03, 04, 05, 06a/b, 08 |
| **论文章节** | **Ch3 全部填实**（Table 3.5.4、Table 3.9、Proposition 3.1）+ **Ch4.1.1 偏导表** |
| **PR 体量** | 12 新文件 + 7 测试文件 + 2 验证脚本 |

---

## 1. Why（为什么需要这个包）

### 1.1 v4 论文实验平台缺一个环境

Pkg-01 已归档 v4.7 的 NonStationaryTag。**当前主路径 `hyper_mve/envs/` 完全空白**——没有 env，Pkg-03 (BeliefNet) 无法生成训练数据，Pkg-04 (Model) 无法验证 forward 端到端，Pkg-05 (Trainer) 无法跑 smoke test。**整个 v4 工作流被本包阻塞**。

但单纯"补一个 env"不够。Ch3 与 Ch4 v4 对该 env 有 7 项**论文级硬约束**，必须同时满足：

| # | 约束 | 来源 | v4.7 NonStationaryTag 是否满足 |
|---|------|------|------------------------------|
| 1 | 关系动态（c_t 调制博弈结构） | Ch3.4 公式 3.4 + Roadmap §1 | ❌ rule 是离散派系切换 |
| 2 | 偏好异质（α/β 两类型共存） | Ch3.5 + 断言 A | ❌ 全员同型 |
| 3 | 物理转移上下文不变（C1 约束） | Ch3 Principle III | ⚠️ rule 影响动作语义 |
| 4 | φ(c) 调制 Fehr-Schmidt | Ch3.5.3 公式 3.7-3.9 | ❌ 无 Fehr-Schmidt |
| 5 | Self-Info 类型可观测性（own only） | Ch3.7 + Ch4.2.2 | ❌ 无类型概念 |
| 6 | env 暴露 oracle types 给 BeliefNet L_opp | **Ch4.5.2 v4 关键** | ❌ |
| 7 | 物理层完全自利同质，偏好层显式异质 | Ch3 Principle I | ❌ hunt/guard 是物理层异质 |

**唯一选项**：从头实现 ResourceCommons。这是 Roadmap §4 Stage 1 (Week 1) 的核心任务，也是论文贡献 4 的核心载体。

### 1.2 ResourceCommons 的独特性（Ch3.10 已论证）

| 环境 | 关系动态？ | 偏好异质 | φ(c) 调制？ |
|------|----------|---------|------------|
| Harvest (Leibo 2017) | Static | Homogeneous selfish | No |
| Cleanup (Hughes 2018) | Static | Homogeneous + gifting | No |
| IPD + Inequity (Hughes 2018) | Static | Heterogeneous F-S | No |
| Melting Pot (Leibo 2021) | Static scenario switch | Scenario-dependent | No |
| Conflict-Aware GA (Kim 2025) | Static | Homogeneous | No |
| **ResourceCommons (v4 本包)** | **Dynamic (c_t)** | **Heterogeneous F-S + φ(c)** | **Yes (公式 3.7)** |

ResourceCommons **唯一具备**三者兼备的混合动机环境——这是贡献 4 的学术新颖性基础。本包是论文 Ch3 的具体实现。

### 1.3 v4 Oracle 监督的反向约束（Ch4 → env）

v4 与 v3 最重要的差异之一：**BeliefNet head_opp 从自监督动作预测改为 Oracle 类型 2 分类**（Ch4.5.2）。这要求 env 在 `step()` 时**暴露全部 N 个 agent 的真实类型 τ_j** 到 info dict——v3 自监督不需要这个。Pkg-05 trainer 计算 L_opp 时直接读 `info["types"]` 作为 CE 标签。

**若 env 不暴露 oracle types → BeliefNet head_opp 无法训练 → Assertion B 实验前提崩塌**。这是本包必须实现的 v4 硬约束。

### 1.4 对应路线图条款

| Roadmap / Chapter 条款 | 本包如何响应 |
|------------------------|----------------|
| Roadmap §4 Stage 1: 创建 resource_commons.py | 本包 12 新文件实现 |
| Ch3.3.1 设计原则 I-III | env.step() 保证物理转移 c-invariance；reward 严格按类型分离 |
| Ch3.3 资源动力学（公式 3.1-3.4） | `dynamics.py` 严格实现，单测对照解析解 |
| Ch3.5 类型机制 + Fehr-Schmidt | `rewards.py` 实现公式 3.5-3.10；Table 3.5.4 数值精确通过 |
| Ch3.7 六块观测 + Self-Info | `observations.py` 严格遵循 Pkg-01 ObservationLayout |
| Ch3.9 三难度配置 | env 接受 EnvConfig，preset 切换零代码改动 |
| Ch4.5.1/4.5.2 Oracle 监督 | info dict 含 `c_true` / `types` / `caps` v4 Oracle 信号 |
| 断言 A 实验前提 | 单测验证 Ch4.1.1 偏导表四象限 |
| Proposition 3.1 实证 | `proposition_3_1_validation.py` 输出 Pareto-Nash gap 曲线 |

---

## 2. What Changes（具体改动清单）

### 2.1 新增文件（12 个核心 + 5 测试 + 2 脚本）

#### 核心实现（12 个）

| 路径 | 行数估计 | 内容 |
|------|----------|------|
| `hyper_mve/envs/resource_commons/__init__.py` | ~20 | 暴露 `ResourceCommonsEnv` + `make_resource_commons(cfg)` |
| `hyper_mve/envs/resource_commons/env.py` | ~400 | 主类 `ResourceCommonsEnv(gym.Env)`：reset / step / render |
| `hyper_mve/envs/resource_commons/dynamics.py` | ~180 | 公式 3.1-3.4：logistic 再生 + neighbor factor + α(c) |
| `hyper_mve/envs/resource_commons/rewards.py` | ~150 | 公式 3.5-3.10：type α/β reward + φ(c) + ψ(Δ) |
| `hyper_mve/envs/resource_commons/observations.py` | ~200 | 六块观测构造 + FOV 过滤 + padding |
| `hyper_mve/envs/resource_commons/context_evolution.py` | ~100 | 三模式：static / oscillate / random_walk (+ shock) |
| `hyper_mve/envs/resource_commons/spawn.py` | ~80 | Patchy 资源生成（3 hotspot + σ_patch=2.0） |
| `hyper_mve/envs/resource_commons/spaces.py` | ~50 | `gym.spaces.Box` (obs) + `gym.spaces.Discrete` (action) |
| `hyper_mve/envs/resource_commons/state.py` | ~80 | `ResourceCommonsState` dataclass（env 内部状态封装） |
| `hyper_mve/envs/resource_commons/render.py` | ~120 | matplotlib rgb_array 渲染（论文 Fig 3.1 用） |

#### 测试（7 个）

| 路径 | 覆盖 |
|------|------|
| `tests/envs/__init__.py` | – |
| `tests/envs/test_dynamics.py` | 公式 3.1-3.4 解析解对照 |
| `tests/envs/test_rewards.py` | **Table 3.5.4 四场景 + Ch4.1.1 偏导四象限** |
| `tests/envs/test_observations.py` | 六块布局、FOV 过滤、Self-Info |
| `tests/envs/test_context_evolution.py` | 三 c_mode + shock 概率分布 |
| `tests/envs/test_spawn.py` | hotspot 重叠、K/M 整除性 |
| `tests/envs/test_env_info_oracle.py` | **v4 关键**：info 含 c_true / types / caps |
| `tests/envs/test_env_integration.py` | 端到端 reset → step → step → done |

#### 验证脚本（2 个）

| 路径 | 用途 |
|------|------|
| `hyper_mve/scripts/test_resource_commons.py` | 1000 random episode smoke + 性能测试 |
| `hyper_mve/scripts/proposition_3_1_validation.py` | Pareto-Nash gap vs c 实证图（论文 Fig 3.X） |

### 2.2 修改文件（0 个）

本包**不修改**任何现有文件。Pkg-01 已完成所有 schema/config 准备工作。

### 2.3 删除文件（0 个）

v4.7 env 已在 Pkg-01 归档到 `_legacy_v4_7/`，本包无需删除任何文件。

---

## 3. Capabilities（完成后系统获得的新能力）

### 3.1 用户视角

```python
# 1. 一行创建 Medium env (主对比配置)
from hyper_mve.configs import V4Config
from hyper_mve.envs.resource_commons import ResourceCommonsEnv

cfg = V4Config.from_preset("medium")
env = ResourceCommonsEnv(cfg.env, seed=42)

obs, info = env.reset()
assert obs.shape == (4, 99)              # N=4, 99 = ObservationLayout.total_dim(4, 20)
assert "c_true" in info                  # v4 Oracle 信号
assert "types" in info                   # v4 Oracle 信号
assert "caps" in info                    # v4 Oracle 信号

# 2. 强制 c 值（评估 c-segment 用）
obs, info = env.reset(options={"c": 0.5})
assert info["c_true"] == 0.5

# 3. 显式覆盖 type 分配（Ablation 3 类型扫描）
from dataclasses import replace
from hyper_mve.schemas import AgentType
cfg_3a1b = replace(cfg, env=replace(cfg.env, 
    type_assignment=(AgentType.ALPHA,)*3 + (AgentType.BETA,)))
env = ResourceCommonsEnv(cfg_3a1b.env)

# 4. 渲染（论文 Fig 3.1）
img = env.render(mode="rgb_array")  # (256, 256, 3) uint8

# 5. step + info 完整
joint_action = np.array([5, 5, 0, 1])  # 2 HARVEST + 1 NOOP + 1 UP
obs, reward, done, truncated, info = env.step(joint_action)
assert reward.shape == (4,)
assert info["c_true"] is float            # 真实 c_t (本步)
assert info["types"].shape == (4,)        # int8 array, AgentType.value
assert len(info["caps"]) == 4             # tuple of CapabilityVector
assert "deltas" in info                   # 瞬时 Δ_i (用于 buffer)
```

### 3.2 工程团队视角

- **零跨包数据契约偏移**：env 输出严格按 Pkg-01 schema（ObservationLayout, AgentType, CapabilityVector）
- **配置驱动**：preset 切换是单文件修改（不动 env 代码）
- **可复现**：`reset(seed=42)` 完全确定（hotspot 位置 + cap 采样 + c_t 初值）
- **可测试**：每个公式（3.1-3.10）对应一个 pytest 单测；Table 3.5.4 精确数值匹配是 hard gate

### 3.3 论文视角

完成本包后，**Ch3 全部章节可填实数据**：

- **Ch3.3** 资源动力学：公式 3.1-3.4 全部对应到 `dynamics.py` 行号
- **Ch3.4** Context 演化：三模式时序图（实际跑出来的）
- **Ch3.5** 类型机制：Table 3.5.4 完整数据（单测自动生成）
- **Ch3.6** 异质能力：cap 4 维采样分布直方图
- **Ch3.7** 观测函数：Table 3.7.1 六块维度对照（已与 Pkg-01 自洽）
- **Ch3.8** 评估指标：W_soc^phys 等指标在 env 内即可计算
- **Ch3.9** 难度配置：三 preset 实测时序
- **Proposition 3.1**：实证图（c 高时 Pareto-Nash 趋同）

**Ch4.1.1 偏导表**：单测 autograd 提取 ∂R^β/∂u_i 在 (c, Δ) 四象限的数值，与论文表对照——这是断言 A 的物理基础证明。

---

## 4. Impact（影响分析）

### 4.1 对 v4.7 现有功能的影响

**保护**：完全脱钩 v4.7。v4.7 NonStationaryTag 已归档至 `_legacy_v4_7/envs/`，可独立运行（`python hyper_mve/_legacy_v4_7/scripts/test_env.py`）。

**破坏**：无（本包是新增）。

### 4.2 对后续包的影响

| 后续包 | 本包提供 | 影响 |
|--------|----------|------|
| **Pkg-03 BeliefNet** | env.info["c_true"], info["types"] | L_c / L_opp **Oracle 监督**的真值标签来源（v4 关键） |
| **Pkg-04 Model** | env.observation_space, obs.shape | RepNet 输入维度自动对齐（无需 hardcode） |
| **Pkg-05 Trainer** | env.step → reward / delta / info | episode_buffer 直接消费 env 输出，无需中间转换 |
| **Pkg-06a/b Baselines** | env 标准 gym 接口 | 全部 7 baseline 接同一 env，断言 B 公平性 |
| **Pkg-07 Eval Protocols** | env.reset(options={"c": x}) | c-segment / zero-shot 评估的 c 控制接口 |
| **Pkg-08 Experiments** | preset 切换 | Easy/Medium/Hard 三 preset 实验自动化 |

### 4.3 对论文贡献的影响

**直接贡献**：
- **贡献 4 核心载体**：ResourceCommons 是首个同时具备 "关系动态 + 偏好异质 + φ(c) 调制" 的混合动机环境
- **Ch3 全章数据**：所有图表数值直接从本包跑出
- **Ch4.1.1 偏导表**：自动微分提取的实测值与理论值对比（断言 A 物理基础）
- **Proposition 3.1**：实证 Pareto-Nash gap 单调性（v4 vs v3 改进的实证证据）

**间接贡献**：
- 断言 A（类型梯度撕裂）实验前提：env 必须支持 1α+1β / 2α+2β / 3α+1β 等灵活类型配置
- 断言 B（信念专用容量）实验前提：env 必须支持 zero-shot c 评估（强制 c=未训练值）
- 断言 C（三路必要性）实验前提：env 必须支持 c 隐藏模式（hide c_t in obs，仅 BeliefNet 推断）

### 4.4 风险与对策

| 风险 | 触发条件 | 对策 |
|------|----------|------|
| 公式实现误差（如 logistic 数值漂移） | 长 episode (T=300) 累积浮点误差 | float32 + clip 每步；单测对照 5 步内解析解 |
| Table 3.5.4 数值不匹配 | reward 公式实现差异 | 单测 hard gate：四场景 100% 匹配才能 merge |
| Ch4.1.1 偏导不匹配 | autograd 提取与理论不一致 | 单测 autograd 对照；diff > 5% 阻塞 PR |
| 性能 < 1000 step/s | FOV 过滤 / patchy spawn / reward 公式开销 | profile 后定位瓶颈；FOV 用 numpy vectorize |
| Oracle types 字段被 Pkg-05 误用为 model 输入 | trainer 端代码误读 info 字段 | spec 08 明确 info 字段分为 "Oracle (仅 trainer 监督用)" 和 "Public (model 可用)" 两组 |
| neighbor 块 9 维 vs Ch3.7 文档 8 维不一致 | 论文审稿质疑 | spec 04 标注为 "v4 实施 augment with presence_flag"，论文 Ch3.7 末段加 footnote |
| Patchy hotspot 重叠导致资源稀疏 | 3 hotspot 中心距离过近 | spawn.py 加 min-distance 约束（≥ 2 σ_patch） |

---

## 5. References

| 来源 | 引用条款 |
|------|----------|
| `D:\RL\docs\Chapter3_Environment_v4.md` | §3.2 形式化、§3.3 动力学、§3.5 类型、§3.7 观测、§3.9 preset |
| `D:\RL\docs\Chapter4_Architecture_v4.md` | **§4.5.1 L_c oracle**、**§4.5.2 L_opp oracle**（v4 关键约束） |
| `D:\RL\docs\Chapter4_1_Motivation_v4.md` | §4.1.1 偏导表（实证目标） |
| `D:\RL\docs\Hyper_MuZero_v4_Roadmap.md` | §4 Stage 1 任务清单 |
| Pkg-01 SDD | 全部 schema 接口 |
| 项目 Plan File | Part 2 Pkg-02 + Part 9.B 修订 |

---

## 6. Acceptance Criteria Summary

> 详见 [`design.md`](./design.md) 与各 `specs/*.md`。本节仅速览。

### 数值硬约束（must pass）

- [ ] `test_rewards.py::test_table_3_5_4`: 四场景 φψ 项数值 100% 匹配（误差 < 1e-6）
- [ ] `test_rewards.py::test_alpha_constant_grad`: 100 个随机 (c, Δ) 上 ∂R^α/∂u_i = 1.0
- [ ] `test_rewards.py::test_beta_grad_four_quadrants`: Ch4.1.1 四象限 ∂R^β/∂u_i ∈ {0.7, 2.0, 1.3, 0.0} 100% 匹配
- [ ] `test_dynamics.py::test_logistic_regen`: 单点资源 5 步内与解析解误差 < 1e-4
- [ ] `test_dynamics.py::test_no_overharvest`: fair-share 上限严格执行
- [ ] `test_env_info_oracle.py::test_info_has_oracle_fields`: env.info 必含 c_true, types, caps

### 性能 & 集成

- [ ] `scripts/test_resource_commons.py`: 1000 episode pass + avg step ≤ 1ms
- [ ] `scripts/proposition_3_1_validation.py`: Pareto-Nash gap 曲线 c→1 时 → 0

### 论文章节增量

- [ ] Ch3.3 公式 3.1-3.4 实现行号对照
- [ ] Ch3.5 Table 3.5.4 完整数据
- [ ] Ch3.7 Table 3.7.1 六块维度（自洽 Pkg-01）
- [ ] Ch3.9 三 preset 实测
- [ ] Ch4.1.1 偏导表实测值
- [ ] Proposition 3.1 实证图

---

## 7. Out of Scope（明确不做）

- **不实现** Pkg-03/04/05 任何 model / trainer / planner 逻辑
- **不实现** distributed env（多进程并行）—— 单进程足够 1000 step/s
- **不实现** PettingZoo / RLLib MultiAgentEnv 适配——保持轻量 gym.Env
- **不实现** 完整 GUI（仅 matplotlib rgb_array 静态渲染）
- **不引入新依赖**（仅 numpy + gym + 现有 matplotlib）
- **不修改** Pkg-01 schema 或 config
- **不修改** `D:\RL\docs\` 论文章节文件（如发现 spec 与 Ch3 不一致，先报 issue 再讨论）
