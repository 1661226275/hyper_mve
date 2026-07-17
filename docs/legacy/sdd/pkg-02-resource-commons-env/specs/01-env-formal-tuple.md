# Spec 01: 环境形式化元组

> 父文档：[`../proposal.md`](../proposal.md) §2.1 · [`../design.md`](../design.md) §1

---

## 1. Purpose

为 Ch3.2 定义的 ResourceCommons 形式化元组提供 Python 实现。env 内部状态封装为 `ResourceCommonsState` dataclass，14 个元组分量明确对应到 Python 字段（避免后续模块各自推断）。

---

## 2. Interface

### 2.1 文件路径

`hyper_mve/envs/resource_commons/state.py`

### 2.2 ResourceCommonsState

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class ResourceCommonsState:
    """ResourceCommons env 内部完整状态 (Ch3.2 形式化元组)。
    
    14 分量元组对应:
        1.  N                    → 由 cfg.env.N 静态决定 (此处不存)
        2.  G = {0,...,L-1}²     → 由 cfg.env.L 静态决定
        3.  R = {r_1,...,r_K}    → resource_positions / resource_stocks
        4.  T = {α, β}           → 由 cfg.env.type_assignment 静态决定
        5.  S 全状态              → 本 dataclass
        6.  {A_i}                → 由 spaces.py 静态决定
        7.  P 转移函数             → 由 dynamics.py 实现
        8.  {R_i} reward 函数      → 由 rewards.py 实现
        9.  {O_i} observation     → 由 observations.py 实现
        10. C = [0, 1]            → 由 c_t 字段表示
        11. P_c 上下文演化         → 由 context_evolution.py 实现
        12. ρ_c U([0,1])          → reset 时采样
        13. ρ_τ type 分配         → 由 cfg.env.type_assignment 静态决定
        14. T_max                 → 由 cfg.env.T_max 静态决定
    
    State 字段 (随 episode 时间变化):
    """
    # Agent 位置 (Ch3.2 分量 5 的子集)
    agent_positions: np.ndarray         # (N, 2) int32, (x, y) in [0, L)
    
    # 资源场 (Ch3.2 分量 5 的子集)
    resource_positions: np.ndarray      # (K, 2) int32, hotspot 内 (x, y)
    resource_stocks: np.ndarray         # (K,) float32, 当前 q_k ∈ [0, Q_max]
    
    # 累积统计 (per agent)
    cumulative_harvests: np.ndarray     # (N,) float32, sum_{t} u_i^t
    steps_since_harvest: np.ndarray     # (N,) int32, 自上次成功 HARVEST 步数
    
    # 共享上下文 c_t (Ch3.2 分量 10)
    c_t: float                          # ∈ [0, 1]
    c_history: np.ndarray               # (T_max,) float32, episode 内 c_t 全历史 (评估用)
    
    # 时序
    step_idx: int                       # 当前 episode 内步数
    done: bool                          # 是否 T_max 到达
    
    # Episode-fixed (从 cfg 初始化, 但每 reset 重新采样)
    agent_caps: tuple                   # tuple[CapabilityVector, ...] 长度 N (Pkg-01 schema)
    agent_types: np.ndarray             # (N,) int8 AgentType.value
    hotspot_centers: np.ndarray         # (M, 2) int32
    last_actions: np.ndarray            # (N,) int64, 上步动作 (用于 neighbor obs block)
    
    def copy(self) -> "ResourceCommonsState":
        """深拷贝, 用于 reset 时 snapshot."""
        return ResourceCommonsState(
            agent_positions=self.agent_positions.copy(),
            resource_positions=self.resource_positions.copy(),
            resource_stocks=self.resource_stocks.copy(),
            cumulative_harvests=self.cumulative_harvests.copy(),
            steps_since_harvest=self.steps_since_harvest.copy(),
            c_t=self.c_t,
            c_history=self.c_history.copy(),
            step_idx=self.step_idx,
            done=self.done,
            agent_caps=self.agent_caps,  # 不可变, 共享 OK
            agent_types=self.agent_types.copy(),
            hotspot_centers=self.hotspot_centers.copy(),
            last_actions=self.last_actions.copy(),
        )
```

### 2.3 reset / step 不变量

`ResourceCommonsEnv` 必须维护以下 invariants:

| Invariant | 校验位置 | 失败行为 |
|-----------|---------|---------|
| `state.agent_positions ∈ [0, L)²` | 每 step 后 | `assert` (开发模式) |
| `state.resource_stocks ∈ [0, Q_max]` | 每 step 后 | `clip` 自动修复 |
| `len(state.agent_caps) == N` | reset 后 | ValueError |
| `state.agent_types.shape == (N,)` | reset 后 | ValueError |
| `state.step_idx ∈ [0, T_max]` | step 后 | done=True 触发 |
| `state.last_actions.dtype == int64` | step 后 | dtype 强制转换 |
| `state.c_t ∈ [0, 1]` | reset 后 + context_evolution 后 | `clip(0, 1)` |
| `0 ≤ sum(harvests_per_step) ≤ Q_max × K` | step 后 | 单测验证 |

---

## 3. Implementation Notes

### 3.1 为何用 mutable dataclass

- env 内部状态频繁变更（agent 移动、资源消耗）→ frozen 不可行
- 但 reset 时需要保留初始 snapshot（用于 debug）→ 提供 `copy()` 方法
- buffer 不存储 state 整体，只存 obs / action / reward → 不需要 frozen 保证

### 3.2 c_history 字段的取舍

- 存储全 T_max 历史增加内存（每 episode +1.2 KB for T=300）
- 但评估时 Proposition 3.1 需要 c_t vs welfare 时序图
- 折中：仅在 evaluation 模式下记录（cfg.eval.record_c_history=True）；训练模式 None 节省内存

### 3.3 hotspot_centers 字段

- 每 reset 由 spawn.py 重新采样
- 进入 info dict 仅供 evaluator 可视化（spec 08 §6.3 严格分组）
- agent / model 不可访问

### 3.4 last_actions 字段

- 用于 observations.py 中 neighbor 块的 `last_action_onehot` 字段（Ch3.7）
- 初始化为 0 (NOOP) 第 0 步
- step 末尾更新为当前 step 的 action

---

## 4. Edge Cases

| 场景 | 行为 |
|------|------|
| `state.copy()` 后修改副本 | 不影响原 state |
| Agent 移动到边界外 | env.step 内部 clip 到 [0, L-1] |
| HARVEST 在无资源格子 | u_i = 0, steps_since_harvest++ |
| 所有资源 q=0 | episode 不 early-terminate, 继续到 T_max |
| step_idx > T_max | done=True 触发 (env 内部 invariant) |
| 多 agent 同格 HARVEST | 公式 3.2 fair-share (rewards.py 处理) |

---

## 5. Acceptance Criteria

### 5.1 单元测试（`tests/envs/test_state.py`）

```python
import numpy as np
import pytest
from hyper_mve.envs.resource_commons.state import ResourceCommonsState

def test_state_construct_minimal():
    s = ResourceCommonsState(
        agent_positions=np.zeros((4, 2), dtype=np.int32),
        resource_positions=np.zeros((20, 2), dtype=np.int32),
        resource_stocks=np.ones(20, dtype=np.float32),
        cumulative_harvests=np.zeros(4, dtype=np.float32),
        steps_since_harvest=np.zeros(4, dtype=np.int32),
        c_t=0.5,
        c_history=np.zeros(200, dtype=np.float32),
        step_idx=0,
        done=False,
        agent_caps=tuple(),
        agent_types=np.zeros(4, dtype=np.int8),
        hotspot_centers=np.zeros((3, 2), dtype=np.int32),
        last_actions=np.zeros(4, dtype=np.int64),
    )
    assert s.c_t == 0.5

def test_state_copy_independent():
    s1 = _make_state()
    s2 = s1.copy()
    s2.agent_positions[0, 0] = 99
    assert s1.agent_positions[0, 0] != 99  # 原 state 不变

def test_state_dtype_strict():
    """所有 numpy 字段必须严格 dtype."""
    s = _make_state()
    assert s.agent_positions.dtype == np.int32
    assert s.resource_stocks.dtype == np.float32
    assert s.agent_types.dtype == np.int8
    assert s.last_actions.dtype == np.int64
```

### 5.2 集成测试

- env 实例化后 `env._state` 字段类型正确
- 1000 random episode 后 invariants 全部保持

### 5.3 性能

- `ResourceCommonsState(...)` 构造 < 20 μs
- `state.copy()` < 50 μs (含 numpy 数组拷贝)

---

## 6. Cross-references

- Ch3.2 形式化元组定义
- `02-resource-dynamics.md`（state 字段如何被 dynamics.py 更新）
- `04-six-block-observation.md`（state → observation 映射）
- `08-gym-api.md`（state 字段如何暴露到 info）
- Pkg-01 `02-capability-vector.md`（CapabilityVector 在 agent_caps 中）
- Pkg-01 `01-agent-type-schema.md`（AgentType 在 agent_types 中存 int8）
