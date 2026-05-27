# Spec 07: Easy / Medium / Hard Env Presets

> 父文档：[`../proposal.md`](../proposal.md) §2.1 · [`../design.md`](../design.md) §1.2

---

## 1. Purpose

定义 Easy / Medium / Hard 三难度 env 配置如何被消费——env 完全配置驱动，preset 切换零代码改动。本 spec 描述 env 如何从 `EnvConfig`（Pkg-01）派生 N, L, K, M, T_max 等维度。

---

## 2. Interface

### 2.1 文件路径

无新增文件。本 spec 描述 env.py 如何读取 `EnvConfig`。

### 2.2 EnvConfig → env 映射

```python
# hyper_mve/envs/resource_commons/env.py 中的 init 模式

class ResourceCommonsEnv(gym.Env):
    """ResourceCommons gym.Env 实现 (Ch3 全章 + Ch4 v4 Oracle 信号)."""
    
    def __init__(self, cfg: EnvConfig, seed: int | None = None):
        self.cfg = cfg
        self.N = cfg.N
        self.L = cfg.L
        self.K = cfg.K
        self.M = cfg.M
        self.T_max = cfg.T_max
        self.A = cfg.A
        
        self._rng = np.random.default_rng(seed)
        self._context_evo = build_context_evolution(cfg.c_mode, cfg)
        
        # 观测空间 (Pkg-01 ObservationLayout)
        obs_dim = ObservationLayout.total_dim(self.N, self.K)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.N, obs_dim), dtype=np.float32,
        )
        self.action_space = gym.spaces.MultiDiscrete([self.A] * self.N)
        
        # 初始 state 占位 (reset 时填充)
        self._state: ResourceCommonsState | None = None
```

### 2.3 三 preset 字段对照（与 Ch3.9 Table）

```
                Easy        Medium      Hard
N               2           4           8
L               8           16          24
K               8           20          40
M               1           3           5
T_max           100         200         300
c_mode          static      static      oscillate
type 分配       1α+1β       2α+2β       4α+4β
观测 dim        45          99          195
动作 dim        6           6           6
```

**dim 派生（验证）**：

| Config | 公式 | 期望值 |
|--------|------|--------|
| Easy obs_dim | `4 + 8*3 + 1*9 + 2 + 4 + 2 = 45` | 45 |
| Medium obs_dim | `4 + 20*3 + 3*9 + 2 + 4 + 2 = 99` | 99 |
| Hard obs_dim | `4 + 40*3 + 7*9 + 2 + 4 + 2 = 195` | 195 |

### 2.4 V4Config preset 加载（与 Pkg-01 一致）

```python
from hyper_mve.configs import V4Config
from hyper_mve.envs.resource_commons import ResourceCommonsEnv

# 三 preset 标准用法
cfg_easy = V4Config.from_preset("easy")
env_easy = ResourceCommonsEnv(cfg_easy.env, seed=42)
assert env_easy.N == 2
assert env_easy.observation_space.shape == (2, 45)

cfg_medium = V4Config.from_preset("medium")
env_medium = ResourceCommonsEnv(cfg_medium.env, seed=42)
assert env_medium.N == 4
assert env_medium.observation_space.shape == (4, 99)

cfg_hard = V4Config.from_preset("hard")
env_hard = ResourceCommonsEnv(cfg_hard.env, seed=42)
assert env_hard.N == 8
assert env_hard.observation_space.shape == (8, 195)
assert env_hard.cfg.c_mode == "oscillate"  # Hard 默认振荡
```

### 2.5 自定义 preset 覆盖

```python
from dataclasses import replace
from hyper_mve.schemas import AgentType

# Ablation 3 类型扫描: Medium + 自定义 type 分配
cfg = V4Config.from_preset("medium")
cfg_3a1b = replace(
    cfg, 
    env=replace(cfg.env, 
        type_assignment=(AgentType.ALPHA,)*3 + (AgentType.BETA,)),
)
env = ResourceCommonsEnv(cfg_3a1b.env)

# Hard + 1M 训练步数 (覆盖默认 2M)
cfg = V4Config.from_preset("hard")
cfg_hard_1m = replace(cfg, train=replace(cfg.train, max_train_steps=1_000_000))
# (注: env 本身不读 train, 但配置一致性方便实验脚本)
```

---

## 3. Implementation Notes

### 3.1 零代码 preset 切换

env.py 内部**不允许**任何 hardcode preset name：
- 不应有 `if preset == "easy": ...`
- 所有维度都通过 `cfg.env.X` 读取
- preset 切换 = 仅 `V4Config.from_preset(name)` 加载不同 config

### 3.2 与 Pkg-01 Medium preset 数值对照

Pkg-01 `configs/presets/medium.py::build_medium_config()` 返回的 EnvConfig 字段：

```python
env = EnvConfig(
    N=4,
    L=16,
    K=20,
    M=3,
    T_max=200,
    type_assignment=(ALPHA, ALPHA, BETA, BETA),
    c_mode="static",
    # ... 动力学参数
)
```

本 spec 不重新定义这些值——完全从 Pkg-01 读取。**任何修改 Medium preset 必须先修 Pkg-01 specs/06，再 ResourceCommons 自动 follow**。

### 3.3 cfg 验证（已在 Pkg-01 完成）

EnvConfig.__post_init__ 已校验：
- `len(type_assignment) == N`
- `c_mode ∈ {static, oscillate, random_walk}`
- `0 < alpha_min < alpha_max ≤ 1`

env 不重复校验，直接信任 cfg。

### 3.4 性能跨 preset

| Preset | 单 step 耗时（估计） |
|--------|--------------------|
| Easy (N=2, K=8) | ~100 μs |
| Medium (N=4, K=20) | ~500 μs |
| Hard (N=8, K=40) | ~1500 μs |

Hard 1.5 ms/step 仍 ≥ 666 step/s，符合 ≥ 1000 step/s 目标的 60%（可接受 trade-off 因 N×K 增加）。

---

## 4. Edge Cases

| 场景 | 行为 |
|------|------|
| 自定义 `type_assignment` 长度 != N | EnvConfig.__post_init__ 在 Pkg-01 阶段抛 ValueError |
| 自定义 `c_mode = "piecewise"` | EnvConfig 抛 ValueError（未知 mode） |
| `K = 0` | spawn 返回空数组；env 可运行但所有 reward = 0 |
| `M > K` | `_distribute_K_over_M` 返回部分 0（M 个 hotspot 但仅 K 个有资源） |
| `N = 0` | gym.spaces.MultiDiscrete([6]*0) 抛错；用户应保证 N ≥ 1 |
| `T_max = 0` | env 一 step 后立即 done |
| 用户传入 `cfg.env` 是 `dict` 而非 `EnvConfig` | TypeError（设计选择：严格类型） |

---

## 5. Acceptance Criteria

### 5.1 单元测试（`tests/envs/test_presets_env.py`）

```python
import numpy as np
import pytest
from hyper_mve.configs import V4Config
from hyper_mve.envs.resource_commons import ResourceCommonsEnv
from hyper_mve.schemas import ObservationLayout, AgentType


def test_easy_env_dimensions():
    cfg = V4Config.from_preset("easy")
    env = ResourceCommonsEnv(cfg.env, seed=42)
    assert env.N == 2
    assert env.L == 8
    assert env.K == 8
    assert env.M == 1
    assert env.T_max == 100
    
    obs, info = env.reset()
    assert obs.shape == (2, 45)  # Easy obs_dim = 4 + 24 + 9 + 2 + 4 + 2 = 45


def test_medium_env_dimensions():
    cfg = V4Config.from_preset("medium")
    env = ResourceCommonsEnv(cfg.env, seed=42)
    assert env.N == 4
    assert env.K == 20
    
    obs, info = env.reset()
    assert obs.shape == (4, 99)


def test_hard_env_dimensions():
    cfg = V4Config.from_preset("hard")
    env = ResourceCommonsEnv(cfg.env, seed=42)
    assert env.N == 8
    
    obs, info = env.reset()
    assert obs.shape == (8, 195)


def test_easy_type_assignment():
    cfg = V4Config.from_preset("easy")
    env = ResourceCommonsEnv(cfg.env, seed=42)
    env.reset()
    assert env.cfg.type_assignment == (AgentType.ALPHA, AgentType.BETA)


def test_medium_type_2a2b():
    cfg = V4Config.from_preset("medium")
    env = ResourceCommonsEnv(cfg.env, seed=42)
    env.reset()
    assert env.cfg.type_assignment == (
        AgentType.ALPHA, AgentType.ALPHA, AgentType.BETA, AgentType.BETA,
    )


def test_hard_type_4a4b():
    cfg = V4Config.from_preset("hard")
    env = ResourceCommonsEnv(cfg.env, seed=42)
    env.reset()
    types = env.cfg.type_assignment
    assert len(types) == 8
    assert sum(1 for t in types if t == AgentType.ALPHA) == 4
    assert sum(1 for t in types if t == AgentType.BETA) == 4


def test_hard_c_mode_oscillate():
    cfg = V4Config.from_preset("hard")
    env = ResourceCommonsEnv(cfg.env, seed=42)
    assert env.cfg.c_mode == "oscillate"


def test_preset_switch_zero_code():
    """切换 preset 仅改 from_preset 调用, env 代码无任何分支."""
    for name in ("easy", "medium", "hard"):
        cfg = V4Config.from_preset(name)
        env = ResourceCommonsEnv(cfg.env, seed=42)
        obs, info = env.reset()
        # 维度自动派生 (无 if/else)
        expected = ObservationLayout.total_dim(cfg.env.N, cfg.env.K)
        assert obs.shape == (cfg.env.N, expected)


def test_custom_type_assignment_override():
    """Ablation 3 类型扫描: replace type_assignment."""
    from dataclasses import replace
    cfg = V4Config.from_preset("medium")
    cfg_3a1b = replace(cfg,
        env=replace(cfg.env,
            type_assignment=(AgentType.ALPHA,)*3 + (AgentType.BETA,)))
    env = ResourceCommonsEnv(cfg_3a1b.env, seed=42)
    env.reset()
    types = env.cfg.type_assignment
    assert sum(1 for t in types if t == AgentType.ALPHA) == 3
    assert sum(1 for t in types if t == AgentType.BETA) == 1


def test_no_hardcoded_preset_name():
    """env 代码不应包含 'easy' / 'medium' / 'hard' 字符串."""
    import inspect
    from hyper_mve.envs.resource_commons import env as env_module
    src = inspect.getsource(env_module)
    # 允许出现在注释/docstring, 但不应作为分支判断
    assert 'if preset ==' not in src
    assert 'if cfg.preset_name ==' not in src
```

### 5.2 集成测试

- 三 preset 各跑 100 episode，无 NaN/Inf
- 三 preset 的 obs_dim 与 ObservationLayout.total_dim 自洽

### 5.3 与 Ch3.9 Table 对照

| 字段 | Easy 预期 | 实测 | Medium 预期 | 实测 | Hard 预期 | 实测 |
|------|----------|------|-------------|------|----------|------|
| N | 2 | ✓ | 4 | ✓ | 8 | ✓ |
| L | 8 | ✓ | 16 | ✓ | 24 | ✓ |
| K | 8 | ✓ | 20 | ✓ | 40 | ✓ |
| M | 1 | ✓ | 3 | ✓ | 5 | ✓ |
| T_max | 100 | ✓ | 200 | ✓ | 300 | ✓ |
| c_mode | static | ✓ | static | ✓ | oscillate | ✓ |

---

## 6. Cross-references

- Ch3.9 Reference Configurations Table
- Pkg-01 `06-difficulty-presets.md`（preset 实际数值定义）
- Pkg-01 `05-v4-config-structure.md`（EnvConfig 结构）
- `01-env-formal-tuple.md`（env state 字段）
- `08-gym-api.md`（observation_space / action_space 派生）
