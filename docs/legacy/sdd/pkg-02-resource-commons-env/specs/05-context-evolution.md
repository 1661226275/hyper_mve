# Spec 05: Context Evolution — 三模式 c_t

> 父文档：[`../proposal.md`](../proposal.md) §2.1 · [`../design.md`](../design.md) §3 D5

---

## 1. Purpose

实现 Ch3.4.3 定义的三种 c_t 演化模式：
- **Mode A (static)**：c_t ≡ c_0，episode 内固定（**主训练 & 评估默认**）
- **Mode B (oscillate)**：c_t = 0.5 + 0.5·sin(2πt/50)，季节性周期
- **Mode C (random_walk + shock)**：高斯随机游走 + 20% 概率脉冲

每种模式必须独立可选 + 可复现（seed 控制）。

---

## 2. Interface

### 2.1 文件路径

`hyper_mve/envs/resource_commons/context_evolution.py`

### 2.2 公开 API

```python
import numpy as np
from abc import ABC, abstractmethod


class ContextEvolution(ABC):
    """Context c_t 演化的抽象基类."""
    
    @abstractmethod
    def initial_c(self, rng: np.random.Generator) -> float:
        """采样初始 c_0."""
        pass
    
    @abstractmethod
    def step(self, c_prev: float, step_idx: int, rng: np.random.Generator) -> float:
        """演化一步: c_{t+1} = f(c_t, t, rng)."""
        pass


class StaticContext(ContextEvolution):
    """Mode A: c_t ≡ c_0, 一次采样后固定 (主训练默认)."""
    
    def initial_c(self, rng):
        return float(rng.uniform(0.0, 1.0))
    
    def step(self, c_prev, step_idx, rng):
        return c_prev


class OscillateContext(ContextEvolution):
    """Mode B: c_t = 0.5 + 0.5 sin(2πt/T), T=50 默认.
    
    用于验证季节性节律适应.
    """
    
    def __init__(self, period: int = 50):
        self.period = period
    
    def initial_c(self, rng):
        # 振荡模式初值 = sin(0) 中心点 = 0.5
        return 0.5
    
    def step(self, c_prev, step_idx, rng):
        return float(0.5 + 0.5 * np.sin(2.0 * np.pi * step_idx / self.period))


class RandomWalkContext(ContextEvolution):
    """Mode C: c_{t+1} = clip(c_t + N(0, σ²), 0, 1) + 20% prob shock.
    
    Shock: Δc ~ U(-0.3, 0.3) 叠加.
    用于 BeliefNet 核心考验 (Ch6.9 zero-shot 场景).
    """
    
    def __init__(self, 
                 sigma: float = 0.05,
                 shock_prob: float = 0.2,
                 shock_range: float = 0.3):
        self.sigma = sigma
        self.shock_prob = shock_prob
        self.shock_range = shock_range
    
    def initial_c(self, rng):
        return float(rng.uniform(0.0, 1.0))
    
    def step(self, c_prev, step_idx, rng):
        # 1. 高斯游走
        noise = rng.normal(0.0, self.sigma)
        c_new = c_prev + noise
        
        # 2. 20% 概率脉冲
        if rng.random() < self.shock_prob:
            delta = rng.uniform(-self.shock_range, self.shock_range)
            c_new += delta
        
        return float(np.clip(c_new, 0.0, 1.0))


def build_context_evolution(c_mode: str, cfg) -> ContextEvolution:
    """工厂函数: 按 cfg 构造对应 mode 实例."""
    if c_mode == "static":
        return StaticContext()
    elif c_mode == "oscillate":
        return OscillateContext(period=cfg.c_oscillate_period)
    elif c_mode == "random_walk":
        return RandomWalkContext(
            sigma=cfg.c_random_walk_sigma,
            shock_prob=cfg.c_shock_prob,
            shock_range=cfg.c_shock_range,
        )
    raise ValueError(f"Unknown c_mode: {c_mode}")
```

### 2.3 与 env 的集成

```python
# env.py 中的使用
class ResourceCommonsEnv:
    def __init__(self, cfg: EnvConfig, seed: int | None = None):
        self.cfg = cfg
        self._context_evo = build_context_evolution(cfg.c_mode, cfg)
        self._rng = np.random.default_rng(seed)
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        # 初始 c_0
        c_0 = (options.get("c") if options else None) \
              or self._context_evo.initial_c(self._rng)
        
        # ... 其他 reset 逻辑
    
    def step(self, action):
        # ... 主要 step 逻辑
        
        # 演化 c_t
        self._state.c_t = self._context_evo.step(
            self._state.c_t, self._state.step_idx, self._rng
        )
        self._state.c_history[self._state.step_idx] = self._state.c_t
```

---

## 3. Implementation Notes

### 3.1 为何抽象基类

- 三模式行为差异大（static 完全不变，random_walk 每步采样）
- 抽象基类让 `env.step` 不关心 mode 细节
- 未来扩展（如 piecewise 切换、周期跳变）只需新增一个 subclass

### 3.2 RNG 隔离

- env._rng 是 episode 级 RNG（reset 时重置）
- 三模式都从 env._rng 取随机性，保证 seed 完全决定 c_t 时序
- shock 概率判定也使用同一 RNG（确保可复现）

### 3.3 与约束 C3（物理不可控）的契合

Ch3 Principle: c_t 物理不可控——所有 mode 的 step 都**不接受 agent action 作为输入**，只依赖 step_idx + RNG。这是设计上的硬约束，违反将破坏 RNS-MMG 形式化。

### 3.4 评估时的特殊用法

- **c-segment 评估**（Ch6.2.4）：用 StaticContext + `reset(options={"c": x})` 强制 c_0
- **zero-shot 评估**（Ch6.9）：训练 c ∈ {0.2, 0.5, 0.8}，测试 c ∈ {0.0, 0.35, 0.65, 1.0}
- **online 适应评估**：训练 static，测试 oscillate 或 random_walk（v4 BeliefNet 关键场景）

### 3.5 默认参数来源

| 参数 | 值 | 来源 |
|------|----|------|
| `OscillateContext.period` | 50 | Ch3.4.3 Mode B "2πt/50" |
| `RandomWalkContext.sigma` | 0.05 | Ch3.4.3 Mode C N(0, 0.05²) |
| `RandomWalkContext.shock_prob` | 0.2 | Ch3.4.3 Mode C "20% shock prob" |
| `RandomWalkContext.shock_range` | 0.3 | Ch3.4.3 Mode C Δc ~ U(-0.3, 0.3) |

---

## 4. Edge Cases

| 场景 | 行为 |
|------|------|
| static 模式 + `reset(options={"c": 0.5})` | c_0=0.5 强制，后续不变 |
| oscillate 模式 + `options.c` 指定 | initial_c 被 options.c 覆盖，但 step 仍按 sin 演化 → 不符直觉 |
| 不同 seed 同 mode | c 时序不同 |
| 相同 seed 同 mode | c 时序完全相同（可复现） |
| random_walk 漂移到边界 | clip(0, 1) 保证不越界 |
| oscillate t=0 | c = 0.5 + 0.5·sin(0) = 0.5（中性） |
| oscillate t=12 (period=50, 1/4 周期) | c ≈ 1.0 |
| random_walk 偶发触发多次 shock | 概率独立，可能连续 shock（罕见） |

**对 oscillate + options.c 的处理决策**：若用户传 `options.c`，OscillateContext 应在 step 内将 c 偏移到该值（保持 sin 周期）；或简单警告并忽略。**当前默认**：忽略（initial_c 总是返回 0.5），用户用 static + options.c 做评估。

---

## 5. Acceptance Criteria

### 5.1 单元测试（`tests/envs/test_context_evolution.py`）

```python
import numpy as np
import pytest
from hyper_mve.envs.resource_commons.context_evolution import (
    StaticContext, OscillateContext, RandomWalkContext, build_context_evolution,
)


def test_static_unchanged():
    ce = StaticContext()
    rng = np.random.default_rng(42)
    c_0 = ce.initial_c(rng)
    for t in range(100):
        c_t = ce.step(c_0, t, rng)
        assert c_t == c_0  # 严格不变


def test_static_initial_in_range():
    ce = StaticContext()
    rng = np.random.default_rng(42)
    for _ in range(1000):
        c = ce.initial_c(rng)
        assert 0.0 <= c <= 1.0


def test_oscillate_sin_formula():
    ce = OscillateContext(period=50)
    rng = np.random.default_rng(42)
    # t=0: sin(0)=0, c=0.5
    assert abs(ce.step(0.0, 0, rng) - 0.5) < 1e-6
    # t=12.5: sin(π/2)=1, c=1.0
    # (但 step_idx 是 int, 用 t=12 ≈ 0.957, t=13 ≈ 0.999)
    # t=25: sin(π)=0, c=0.5
    assert abs(ce.step(0.0, 25, rng) - 0.5) < 1e-6
    # t=37.5: sin(3π/2)=-1, c=0
    # t=50: 周期回到 0
    assert abs(ce.step(0.0, 50, rng) - 0.5) < 1e-6


def test_oscillate_period_50():
    ce = OscillateContext(period=50)
    rng = np.random.default_rng(0)
    c_at_0 = ce.step(0.0, 0, rng)
    c_at_50 = ce.step(0.0, 50, rng)
    assert abs(c_at_0 - c_at_50) < 1e-6  # 周期对齐


def test_random_walk_drift():
    ce = RandomWalkContext(sigma=0.05, shock_prob=0.0)  # 禁用 shock 测试纯游走
    rng = np.random.default_rng(42)
    c = 0.5
    history = [c]
    for t in range(100):
        c = ce.step(c, t, rng)
        history.append(c)
        assert 0.0 <= c <= 1.0  # clip 保证
    # 100 步累积漂移应有显著范围
    assert max(history) - min(history) > 0.1


def test_random_walk_shock_triggers():
    """shock_prob=1.0 强制每步 shock."""
    ce = RandomWalkContext(sigma=0.0, shock_prob=1.0, shock_range=0.5)
    rng = np.random.default_rng(42)
    c_prev = 0.5
    c_new = ce.step(c_prev, 0, rng)
    # σ=0 → 仅 shock 贡献, |Δc| ≤ 0.5
    assert abs(c_new - c_prev) <= 0.5 + 1e-6


def test_random_walk_reproducible():
    """相同 seed → 相同 c 时序."""
    ce = RandomWalkContext()
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    
    c1 = ce.initial_c(rng1)
    c2 = ce.initial_c(rng2)
    assert c1 == c2
    
    for t in range(100):
        c1 = ce.step(c1, t, rng1)
        c2 = ce.step(c2, t, rng2)
        assert c1 == c2


def test_factory_static():
    from dataclasses import dataclass
    @dataclass
    class FakeCfg:
        c_oscillate_period: int = 50
        c_random_walk_sigma: float = 0.05
        c_shock_prob: float = 0.2
        c_shock_range: float = 0.3
    
    ce = build_context_evolution("static", FakeCfg())
    assert isinstance(ce, StaticContext)


def test_factory_unknown_mode():
    from dataclasses import dataclass
    @dataclass
    class FakeCfg:
        c_oscillate_period: int = 50
        c_random_walk_sigma: float = 0.05
        c_shock_prob: float = 0.2
        c_shock_range: float = 0.3
    
    with pytest.raises(ValueError, match="Unknown c_mode"):
        build_context_evolution("piecewise", FakeCfg())
```

### 5.2 集成测试

- env(c_mode="static") 100 episode 后 c_history 全相同
- env(c_mode="oscillate") 50 step 后 c 回到初值
- env(c_mode="random_walk") 不同 seed 产生不同时序

### 5.3 性能

- `StaticContext.step` < 0.1 μs
- `OscillateContext.step` < 1 μs（sin 计算）
- `RandomWalkContext.step` < 5 μs（含 2 次 RNG 采样）

---

## 6. Cross-references

- Ch3.4 Context c_t 演化
- Ch3.4.3 三模式定义
- Ch6.9 zero-shot c 评估（消费此 spec）
- `01-env-formal-tuple.md`（state.c_t / c_history 字段）
- `07-difficulty-presets-env.md`（Hard 默认 oscillate）
- `08-gym-api.md`（reset(options={"c": x}) 接口）
- `../design.md` D5 (构造固定模式 + reset 初值)
