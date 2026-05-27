# Pkg-02: ResourceCommons Environment — Design

> 配套阅读：[`proposal.md`](./proposal.md)（**先读 proposal 再读本文**）

---

## 1. Context

### 1.1 项目阶段

本包是 v4 重构的**第二个 SDD 包**，紧随 Pkg-01（Foundation Schema）。完成本包后，Pkg-03 (BeliefNet + TriContextEncoder) 可立即启动——BeliefNet 的训练数据完全来自本环境的 `env.info` Oracle 信号。

```
Pkg-01 (Schema)  ✅ 完成
    ↓
Pkg-02 (本包)    ⏳ 当前
    ↓
Pkg-03 (BeliefNet) ← 立即依赖 env.info["c_true"] / info["types"]
Pkg-04 (Model)     ← 依赖 env.observation_space
Pkg-05 (Trainer)   ← 依赖 env.step → buffer
```

### 1.2 前置依赖（本包消费什么）

| 来源 | 内容 |
|------|------|
| Pkg-01 `schemas/` | `AgentType`, `CapabilityVector`, `ObservationLayout` |
| Pkg-01 `configs/env_config.py` | `EnvConfig` 所有动力学参数 |
| Pkg-01 `configs/presets/{easy,medium,hard}.py` | 三 preset env 字段 |
| Ch3 全章 | 14 分量元组 + 公式 3.1-3.10 + 三 preset 表 |
| Ch4.5.1, 4.5.2 (v4) | env.info Oracle 信号要求 |
| Ch4.1.1 | 偏导表（实测对照目标） |

### 1.3 本包提供（后续包消费什么）

| 输出 | 消费者 | 用途 |
|------|--------|------|
| `ResourceCommonsEnv` 类 | 全部 03-08 | gym.Env 标准接口 |
| `env.info["c_true"]` | Pkg-03 BeliefNet, Pkg-05 trainer | L_c 监督 MSE 标签 |
| `env.info["types"]` | Pkg-03 BeliefNet head_opp, Pkg-05 trainer | L_opp Oracle CE 标签（**v4 关键**） |
| `env.info["caps"]` | Pkg-04 model role_encoder | cap_emb 输入（per agent） |
| `env.info["deltas"]` | Pkg-05 trainer | TimeStepRecord.delta 字段 |
| `env.observation_space.shape` | Pkg-04 RepNet init | 观测维度（自动） |

---

## 2. Goals

### 2.1 主目标（必须完成）

1. **G1：公式 3.1-3.10 严格实现**——所有动力学与 reward 公式与 Ch3 数学定义一致，浮点误差 ≤ 1e-6
2. **G2：Table 3.5.4 四场景 100% 数值匹配**——type β reward 在 (c, Δ) 四象限的 φψ 项精确等于论文表
3. **G3：Ch4.1.1 偏导表 100% 数值匹配**——autograd 提取 ∂R^β/∂u_i 在四象限 ∈ {0.7, 2.0, 1.3, 0.0}
4. **G4：v4 Oracle 监督信号暴露**——env.info 必含 `c_true`, `types`, `caps` 三字段（无遗漏）
5. **G5：三 preset 端到端运行**——Easy/Medium/Hard 三 preset env 1000 random episode 无 NaN/Inf
6. **G6：性能 ≥ 1000 step/s**——单线程 CPU，Medium 配置

### 2.2 衍生目标（应尽量达到）

- **G7**：`mypy --strict hyper_mve/envs/resource_commons/` 零错误
- **G8**：单测覆盖率 ≥ 85%
- **G9**：Proposition 3.1 实证图（Pareto-Nash gap 单调）

### 2.3 Non-Goals（明确不解决）

- **NG1**：**不实现** Pkg-03/04/05 任何 model / trainer / planner 逻辑
- **NG2**：**不实现** distributed env（SubprocVecEnv）——单进程足够
- **NG3**：**不实现** PettingZoo / RLLib MultiAgentEnv 适配
- **NG4**：**不实现**完整 GUI——仅 matplotlib rgb_array
- **NG5**：**不引入新依赖**——仅 numpy + gym + matplotlib
- **NG6**：**不修改** Pkg-01 schema 或 config
- **NG7**：**不修改**论文 Ch3 文档（如发现 spec 与 Ch3 不一致，先报 issue 讨论）

---

## 3. Decisions

> 8 项关键设计抉择。每项格式：**Decision** → 候选 → 推荐 → 理由 → 风险与回滚。

### D1: 资源点坐标存储

**Decision**：env 内部如何存储 K 个资源点的位置与库存？

**候选**：
- **A1**：`dict[(x, y), float]`，key 为坐标 tuple，value 为 q
- **A2**：`np.ndarray shape=(K, 3) = (x, y, q)`，每行一个资源点
- **A3**：稠密 grid `np.ndarray shape=(L, L)`，每格存 q

**推荐**：**A2**

**理由**：
- Ch3.3.2 patchy 生成时 K << L²（如 Medium K=20 vs L²=256）——稠密 grid 浪费 92% 内存
- A1 dict 无法向量化 logistic update（公式 3.1）
- A2 数组可用 numpy broadcast：`q[k] - sum(harvests_at_k) + alpha * f * (Q_max - q[k])` 单行向量化
- agent 查询"我在某格上有资源吗"用 KDTree 一次性预计算（O(log K) per query）
- 数组在 buffer 序列化时高效（zero-copy 转 numpy）

**风险**：
- agent 移动后查询 FOV 内资源需重新匹配坐标 → 用 KDTree（scipy.spatial）一次性查询所有 agent，比 Chebyshev O(K) 扫描快但增加依赖

**回滚**：A3 稠密 grid（仅 Easy L=8 K=8 时合理；Medium 浪费严重）

---

### D2: FOV 过滤实现

**Decision**：agent i 的观测中只包含 φ_fov 半径内的资源/邻居，过滤如何实现？

**候选**：
- **B1**：每步 O(N × K) Chebyshev 全扫描
- **B2**：scipy KDTree 一次性查询
- **B3**：cached neighbor index（按格子缓存，O(1) 查询）

**推荐**：**B1**（每步全扫描）

**理由**：
- Medium N=4, K=20：N × K = 80 次比较，单步 < 0.1 ms
- 即使 Hard N=8, K=40：320 次比较，单步 < 0.5 ms
- B2 KDTree 构造每 step O(K log K) + 查询 O(N log K)——在 K=40 时实际慢于全扫描（因为构造开销）
- B3 缓存机制需要 agent 移动 invalidate 逻辑，引入 bug 风险
- 全扫描代码最简单（5 行 numpy）

**风险**：
- 性能瓶颈出现在 Hard 配置 N=8 → 1000 episode × 300 step × 8 agent × 40 resource = 96M 操作 ≈ 1 秒（仍可接受）
- 若性能不达标，profile 后再切 B2

**回滚**：B2 KDTree（仅需改 `observations.py` 中 _visible_in_fov 函数）

---

### D3: Fehr-Schmidt Δ 计算粒度

**Decision**：公式 3.9 中 Δ_i = u_i − mean_{j≠i} u_j，对全 N agent 计算还是仅 type β agent？

**候选**：
- **C1**：仅 type β agent 计算（按 mask）
- **C2**：全 N agent 计算 Δ_i，按 type mask 选用 φψ 项
- **C3**：全 N agent 计算 Δ_i，type α 的 Δ 也存进 buffer（供 trainer 端类型 ablation）

**推荐**：**C3**

**理由**：
- 向量化效率：numpy 一次性计算 `delta = u - (u.sum() - u) / (N - 1)` 比按 type mask 分支快
- buffer schema 统一：所有 N 个 agent 都有 delta 字段（即使 type α 不用）
- Ablation 灵活性：Pkg-05 trainer 可做"虚拟 type 切换"实验（如把某个 α agent 临时按 β reward 评估）
- 多余开销可忽略：N × float32 加减运算 < 1 μs
- 与 TimeStepRecord 字段 delta shape (N,) 一致（Pkg-01 已定义）

**风险**：
- type α agent 的 delta 字段被误用作 reward 计算 → 单测 `test_alpha_reward_no_psi` 验证 R^α 中无 φψ 项

**回滚**：C1 仅 β 计算（节省 ~10% reward 计算时间，但破坏 buffer schema 一致性）

---

### D4: 浮点稳定性

**Decision**：资源动力学公式 3.1 长 episode 累积浮点误差如何控制？

**候选**：
- **D1**：float32 + 每步 `clip(q, 0, Q_max)`
- **D2**：float64（精度翻倍但内存翻倍）
- **D3**：整数量化（如 q × 1000 存 int32）

**推荐**：**D1**

**理由**：
- T_max=300 (Hard) × 公式 3.1 单步浮点运算 ~5 次乘法 → 累积相对误差 ~1e-5（float32 epsilon × 1500）
- 公式 3.1 截断误差（logistic 离散化）本身 ~1e-3 量级（α=0.02 step 累积）→ 浮点误差远小于模型截断误差
- D2 float64 内存翻倍但精度收益 < 模型截断
- D3 整数量化需要重新校准所有公式（不值得）
- clip(0, Q_max) 防止数值溢出导致负资源或超容

**风险**：
- 公式 3.3 sigmoid 输入接近 0 时数值不稳定 → 用 `torch.sigmoid` 或 `np.tanh` 等数值稳定实现

**回滚**：D2 float64（仅 `dynamics.py` 内部计算，buffer 仍存 float32）

---

### D5: Context 模式切换接口

**Decision**：env 如何在三种 c_t 演化模式（static / oscillate / random_walk）之间切换？

**候选**：
- **E1**：env 构造参数指定模式，运行时不可变
- **E2**：reset 参数动态切换
- **E3**：env 实例化后 setter 方法

**推荐**：**E1**（构造参数固定模式）+ reset 参数指定**初值**

**理由**：
- 同一 env 实例切换模式会污染 episode 之间的 c_t 历史 → 调试困难
- E2 reset 切换看似灵活，但 trainer 期望每个 worker 跑同一模式（统计可比性）
- 显式构造参数避免歧义：`env = ResourceCommonsEnv(cfg, c_mode="static")`，所有 episode 都是 static
- reset 仅控制**初值**：`env.reset(options={"c": 0.5})` 强制 c_0=0.5，后续按 c_mode 演化
- 评估时切换模式 = 创建新 env 实例（开销 < 10 ms）

**风险**：
- 用户期望"边训练边切模式"做 curriculum——明确不支持，由 Pkg-08 实验脚本控制多 env 实例

**回滚**：E3 setter（仅需 add `env.set_c_mode(mode)` 方法，约 10 行）

---

### D6: Patchy 资源是否每 reset 重生成

**Decision**：Ch3.3.2 中 3 hotspot 资源布局，每 reset 是否重新采样位置？

**候选**：
- **F1**：每 reset 重生成（with seed）
- **F2**：episode 间复用同一布局
- **F3**：env 实例化时固定，不可变

**推荐**：**F1**

**理由**：
- 评估需要不同布局测试泛化——固定布局相当于"训练集 = 测试集"
- seed 保证可复现：`env.reset(seed=42)` 的 hotspot 位置确定
- F2 复用会引入 episode 间 correlated noise（违反 i.i.d. 假设）
- 1000 episode 重生成开销 ~1 ms × 1000 = 1 秒（可忽略）
- 实验报告对照（论文 Fig 3.1）可固定 seed 显示同一布局

**风险**：
- hotspot 重叠导致资源稀疏 → spawn.py 加 min_distance ≥ 2σ_patch 约束（spec 06）
- 三 hotspot 中心都在 grid 角落 → 资源集中，agent 难以协调；min_distance 约束可缓解

**回滚**：F3（仅评估时固定布局，训练时仍 F1）

---

### D7: Move 失败处理

**Decision**：cap.ν < 1.0 时移动有概率失败（采样 Bernoulli(ν)），失败后 agent 状态如何？

**候选**：
- **G1**：留在原地（动作变 NOOP）
- **G2**：沿目标方向走部分距离（如 0.5 格）
- **G3**：重新采样动作

**推荐**：**G1**（留在原地）

**理由**：
- Ch3 未明确部分移动语义；保持简洁
- ν 影响通过 binary 体现（成功 vs 失败）足以做"高 ν 倾向稳定 / 低 ν 倾向冒险"的策略学习
- G2 部分移动引入连续动作语义，破坏离散动作空间
- G3 重新采样改变 episode 长度，破坏 buffer 时序

**风险**：
- low ν agent 在 episode 中频繁原地不动，可能困在 dead zone → 通过 cap 采样范围 [0.8, 1.0] 限制（最差 20% 失败率）

**回滚**：G2 部分移动（需重新设计 action space 为连续）

---

### D8: HARVEST 目标资源

**Decision**：HARVEST 动作的目标资源是当前格子，还是当前 + 邻居？

**候选**：
- **H1**：仅当前格子
- **H2**：当前 + 8 个邻居
- **H3**：最近资源点（在 FOV 内）

**推荐**：**H1**

**理由**：
- Ch3.3 公式 3.2 明确 H_{k,t} 是"在 k 点的 agent 集合"——只能在格子上才能采
- H2 邻居采集会破坏"agent 必须靠近资源"的空间博弈
- H3 最近资源点会让 HARVEST 变成"自动寻路"，agent 无需学习移动策略
- 简洁性：1 行实现 vs H2/H3 需 ~10 行

**风险**：
- agent 学习"移动 + 采集"两步策略需要更多 sample → 增加 T_max（已 200/300）

**回滚**：无需回滚（最自然语义）

---

## 4. 设计决策对照表

| Decision | 推荐 | 影响范围 | 后续修改成本 |
|----------|------|----------|--------------|
| D1 资源存储 | np.array (K, 3) | dynamics, spawn, observations | 中（3 处） |
| D2 FOV 过滤 | Chebyshev 全扫描 | observations.py | 低（1 函数） |
| D3 Δ 计算粒度 | 全 agent vectorized | rewards.py | 低（保持 schema 一致） |
| D4 浮点精度 | float32 + clip | dynamics.py | 低（数据类型） |
| D5 Context 切换 | 构造固定 + reset 初值 | env.py | 低（API 约束） |
| D6 Patchy 重生成 | 每 reset (seeded) | spawn.py | 低（seed 控制） |
| D7 Move 失败 | 留原地 | env.py step | 低（1 行） |
| D8 HARVEST 目标 | 当前格子 | env.py step | 低（语义） |

---

## 5. 实现顺序建议

```
Day 1（半天）:
  - spaces.py + state.py + env.py 骨架 (reset/step 返回零观测)
  - tests/envs/test_env_integration.py (端到端跑通)
  - 写 specs/01-env-formal-tuple.md
  
Day 1（半天）+ Day 2:
  - dynamics.py + spawn.py (公式 3.1-3.4 + patchy hotspot)
  - tests/envs/test_dynamics.py (解析解对照)
  - tests/envs/test_spawn.py (hotspot 重叠 + K/M)
  - 写 specs/02-resource-dynamics.md + specs/06-patchy-resource-spawn.md
  
Day 3:
  - rewards.py (公式 3.5-3.10)
  - tests/envs/test_rewards.py: **Table 3.5.4 + Ch4.1.1 偏导四象限**
  - 写 specs/03-fehr-schmidt-reward.md
  - **Hard gate**: 四场景 100% 通过才能进下一步

Day 4-5:
  - observations.py + Self-Info (六块 + FOV)
  - tests/envs/test_observations.py
  - 写 specs/04-six-block-observation.md
  
Day 6:
  - context_evolution.py 三模式
  - tests/envs/test_context_evolution.py
  - 写 specs/05-context-evolution.md
  
Day 7:
  - difficulty preset 集成 + 全 env reset → 三 preset 跑通
  - 写 specs/07-difficulty-presets-env.md
  
Day 8:
  - **info dict v4 Oracle 信号** (c_true/types/caps) 完整化
  - tests/envs/test_env_info_oracle.py
  - 写 specs/08-gym-api.md
  - render.py + matplotlib 渲染

Day 9:
  - scripts/test_resource_commons.py 1000 episode smoke + 性能 profile
  - 优化（如需）

Day 10:
  - scripts/proposition_3_1_validation.py
  - 文档补完 + PR
```

---

## 6. 跨包接口约定（API contract）

### 6.1 import 路径（稳定）

```python
# 后续 6 个包应使用这些 import 路径，本包承诺不变更
from hyper_mve.envs.resource_commons import (
    ResourceCommonsEnv,
    make_resource_commons,         # 工厂函数 (cfg → env)
)

# 标准用法
from hyper_mve.configs import V4Config
cfg = V4Config.from_preset("medium")
env = ResourceCommonsEnv(cfg.env, seed=42)
obs, info = env.reset()
```

### 6.2 env.step 返回签名（稳定）

```python
def step(self, action: np.ndarray) -> tuple[
    np.ndarray,    # obs: (N, obs_dim) float32
    np.ndarray,    # reward: (N,) float32  (per-agent reward, type-aware)
    bool,          # done: 是否 episode 结束 (T_max 到达)
    bool,          # truncated: 是否被截断 (本环境不用，恒 False)
    dict[str, Any] # info: 见 6.3
]:
    ...
```

### 6.3 info dict 完整字段（稳定，v4 关键）

| 字段 | 类型 | 用途 | 谁可读 |
|------|------|------|--------|
| **`c_true`** | `float` (∈ [0, 1]) | **L_c Oracle 监督标签** (Ch4.5.1) | **仅 trainer**，不传给 model |
| **`types`** | `np.ndarray int8 shape=(N,)` | **L_opp Oracle 监督标签** (Ch4.5.2) | **仅 trainer**，不传给 model |
| `caps` | `tuple[CapabilityVector, ...]` 长度 N | role_encoder cap_emb 输入 | model + trainer |
| `deltas` | `np.ndarray float32 shape=(N,)` | TimeStepRecord.delta 字段 | trainer (buffer 写入) |
| `harvests` | `np.ndarray float32 shape=(N,)` | u_i 单步采集量 | 评估指标 W_soc^phys |
| `hotspot_centers` | `np.ndarray float32 shape=(M, 2)` | 评估可视化 | **仅 evaluator** |
| `resource_state` | `np.ndarray float32 shape=(K, 3)` | 评估可视化（资源完整状态） | **仅 evaluator** |
| `step_idx` | `int` | episode 内步数 | trainer (TimeStepRecord.t) |

**严格分组**：
- **Public** (可用作 model 输入): `caps`, `deltas`, `step_idx`
- **Oracle (仅 trainer 监督用)**: `c_true`, `types`
- **Eval only**: `hotspot_centers`, `resource_state`

Pkg-05 trainer 必须严格遵守："Oracle 字段不传入 model forward"——否则违反 v4 Self-Info 原则（Ch3.7 + Ch4.2.2）。

### 6.4 env.reset 返回签名

```python
def reset(self, seed: int | None = None, 
          options: dict | None = None) -> tuple[np.ndarray, dict]:
    """
    Args:
        seed: RNG seed; 影响 hotspot 位置、cap 采样、c_t 初值
        options: 可选覆盖, 支持的 keys:
            - "c": float, 强制 c_0 = 此值 (评估 c-segment 用)
            - "type_assignment": tuple[AgentType, ...] 长度 N (Ablation 3 用)
    
    Returns:
        obs: (N, obs_dim) float32
        info: 含 c_true / types / caps 等
    """
```

### 6.5 observation_space / action_space（稳定）

```python
env.observation_space = gym.spaces.Box(
    low=-np.inf, high=np.inf,
    shape=(N, obs_dim), dtype=np.float32,
)  # obs_dim = ObservationLayout.total_dim(N, K)

env.action_space = gym.spaces.MultiDiscrete([6] * N)
# 6 = (NOOP, UP, DOWN, LEFT, RIGHT, HARVEST)
```

---

## 7. 验证策略概览

> 详细 acceptance criteria 见 `specs/*.md`。本节列 10 个关键测试。

1. `test_dynamics.py::test_logistic_regen_5steps`: 单点解析解 5 步内误差 < 1e-4
2. `test_dynamics.py::test_neighbor_factor_sigmoid`: f(neighbors) 与公式 3.3 一致（κ_f=6, θ_f=0.3）
3. `test_dynamics.py::test_no_overharvest`: fair-share 上限严格 (公式 3.2)
4. `test_rewards.py::test_table_3_5_4`: **四场景 φψ 100% 匹配** (Ch3.5.4)
5. `test_rewards.py::test_alpha_grad_one`: ∂R^α/∂u_i = 1.0 (100 random samples)
6. `test_rewards.py::test_beta_grad_quadrants`: **Ch4.1.1 偏导表四象限 100% 匹配**
7. `test_observations.py::test_self_info_type`: type 块仅 own 2 维
8. `test_env_info_oracle.py::test_info_has_c_true_types_caps`: **v4 关键**
9. `test_env_integration.py::test_1000_random_episodes`: 无 NaN/Inf
10. `scripts/test_resource_commons.py`: ≥ 1000 step/s

---

## 8. Open Questions（待与用户讨论）

| # | Question | 默认决定 | 影响 |
|---|----------|----------|------|
| Q1 | env step 返回的 reward 是 `np.array (N,)` 还是 `dict[int, float]`？ | **`np.array (N,) float32`** | Pkg-05 buffer 字段格式 |
| Q2 | info["types"] 类型用 `tuple[AgentType, ...]` 还是 `np.ndarray int8`？ | **`np.ndarray int8 shape=(N,)`** | 序列化 + Pkg-05 直接消费 |
| Q3 | 是否在 info 暴露 `hotspot_centers` / `resource_state` 完整可视化字段？ | **是**，仅 evaluator 读 | render + 论文 Fig 3.1 |
| Q4 | env 是否提供 `set_seed()` 独立 API 还是仅通过 `reset(seed=...)`？ | **仅通过 reset(seed=...)** | gym 标准 |
| Q5 | neighbor 块 9 维（含 presence_flag）vs Ch3.7 文档 8 维差异如何标注？ | **spec 04 显式标注 "v4 实施 augment"** + 论文 Ch3.7 加 footnote | 论文一致性 |
| Q6 | env 是否在 `reset(options={"types": ...})` 接受运行时类型覆盖？ | **是**，便于 Ablation 3 类型扫描 | Pkg-08 实验脚本 |
| Q7 | render 输出图像分辨率？ | **256×256 RGB uint8** | 论文 Fig 3.1 |

> 若用户对默认决定无异议，本 design.md 视为 finalized。

---

## 9. References

- `proposal.md`（本包）
- Pkg-01 design.md（schema/config 跨包契约）
- `D:\RL\docs\Chapter3_Environment_v4.md` 全章
- `D:\RL\docs\Chapter4_Architecture_v4.md` §4.5.1, §4.5.2
- `D:\RL\docs\Chapter4_1_Motivation_v4.md` §4.1.1
- 项目 Plan File: Part 2 Pkg-02 + Part 9.B
