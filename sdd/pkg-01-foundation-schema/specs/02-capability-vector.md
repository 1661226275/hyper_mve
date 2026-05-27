# Spec 02: CapabilityVector dataclass

> 父文档：[`../proposal.md`](../proposal.md) §2.1 · [`../design.md`](../design.md) §3 D2 / D7

---

## 1. Purpose

为 Ch3.6 定义的 agent 异质能力提供**不可变、范围验证、可采样**的 dataclass 表示。4 维 (η, $\phi^{fov}$, ν, ζ) 在 episode 内固定，跨 episode 重新采样。

字段命名严格遵循 Ch3.6 符号约定，特别 **$\phi^{fov}$ 命名为 `phi_fov`**（避免与 φ(c) 调制函数符号冲突，Ch3.6 文档已注明此 notation adjustment）。

---

## 2. Interface

### 2.1 文件路径

`hyper_mve/schemas/capability.py`

### 2.2 公开 API

```python
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import torch

@dataclass(frozen=True)
class CapabilityVector:
    """Agent 异质能力 4 维向量 (Ch3.6)。
    
    Fields (顺序与 Ch3.6 文档一致, 不可重排):
        eta:     harvest speed,    采样范围 U(0.5, 1.5)
        phi_fov: field of view,   采样范围 U(2.0, 4.0); env 内部 round 为 int
        nu:      move reliability, 采样范围 U(0.8, 1.0)
        zeta:    carry capacity,   采样范围 U(10.0, 30.0)
    
    Frozen: 一旦创建禁止修改，确保 episode 内 cap 真正不变。
    """
    eta: float
    phi_fov: float
    nu: float
    zeta: float

    def __post_init__(self) -> None:
        if not (0.5 <= self.eta <= 1.5):
            raise ValueError(f"CapabilityVector.eta={self.eta} ∉ [0.5, 1.5] (Ch3.6)")
        if not (2.0 <= self.phi_fov <= 4.0):
            raise ValueError(f"CapabilityVector.phi_fov={self.phi_fov} ∉ [2.0, 4.0] (Ch3.6)")
        if not (0.8 <= self.nu <= 1.0):
            raise ValueError(f"CapabilityVector.nu={self.nu} ∉ [0.8, 1.0] (Ch3.6)")
        if not (10.0 <= self.zeta <= 30.0):
            raise ValueError(f"CapabilityVector.zeta={self.zeta} ∉ [10.0, 30.0] (Ch3.6)")

    def to_array(self, dtype: np.dtype = np.float32) -> np.ndarray:
        """转 (4,) numpy 数组，字段顺序固定 (eta, phi_fov, nu, zeta)。"""
        return np.array([self.eta, self.phi_fov, self.nu, self.zeta], dtype=dtype)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "CapabilityVector":
        """(4,) 数组 → CapabilityVector，自动校验范围。"""
        assert arr.shape == (4,), f"Expected shape (4,), got {arr.shape}"
        return cls(eta=float(arr[0]), phi_fov=float(arr[1]), 
                   nu=float(arr[2]), zeta=float(arr[3]))

    @property
    def fov_int(self) -> int:
        """env FOV 过滤用整数视野半径 (Chebyshev distance)。"""
        return int(round(self.phi_fov))

# 模块级工具

def sample_default(rng: np.random.Generator) -> CapabilityVector:
    """按 Ch3.6 标准范围采样一个 capability。
    
    使用 RNG 接口（不用全局 random）以保证可复现性。
    """
    return CapabilityVector(
        eta=float(rng.uniform(0.5, 1.5)),
        phi_fov=float(rng.uniform(2.0, 4.0)),
        nu=float(rng.uniform(0.8, 1.0)),
        zeta=float(rng.uniform(10.0, 30.0)),
    )

def sample_n(n: int, rng: np.random.Generator) -> tuple[CapabilityVector, ...]:
    """采样 N 个独立 capability。"""
    return tuple(sample_default(rng) for _ in range(n))

def to_batch_tensor(caps: Sequence[CapabilityVector], 
                    device: torch.device | None = None) -> torch.Tensor:
    """Sequence[CapabilityVector] → (N, 4) float tensor，供 cap_emb MLP 输入。"""
    arr = np.stack([c.to_array() for c in caps])  # (N, 4)
    t = torch.from_numpy(arr)
    return t.to(device) if device else t
```

### 2.3 典型用例

```python
# 1. env reset 时采样 N 个 capability (Pkg-02)
import numpy as np
from hyper_mve.schemas import sample_n
rng = np.random.default_rng(seed=42)
caps = sample_n(n=4, rng=rng)

# 2. 观测块中包含自身 cap (Pkg-02 ObservationLayout)
own_cap_obs = caps[i].to_array()  # shape (4,)

# 3. model 中 cap_emb 计算 (Pkg-04 role_encoder)
cap_tensor = to_batch_tensor(caps, device='cuda')  # (N, 4)
cap_embs = cap_mlp(cap_tensor)  # (N, d_cap_emb=16)

# 4. env FOV 过滤 (Pkg-02)
visible_radius = caps[i].fov_int  # int ∈ {2, 3, 4}
```

---

## 3. Implementation Notes

### 3.1 frozen=True 的意义

防止 episode 内意外修改 cap（如某个回调把 cap.eta = 0.0 调试遗留）。任何 cap 修改必须显式 `dataclasses.replace(cap, eta=new_val)` 创建新实例。

### 3.2 浮点范围比较

`__post_init__` 用闭区间 `0.5 <= x <= 1.5`。边界采样 `rng.uniform(0.5, 1.5)` 实际是 `[0.5, 1.5)`（numpy 默认），但极少触发右边界。如果发现采样溢出（如 1.5000001），可加 `np.clip(x, 0.5, 1.5)`。

### 3.3 `fov_int` 缓存

frozen dataclass 不能直接缓存 fov_int（property 重计算每次开销 ~微秒）。如果性能瓶颈（FOV 过滤 hot path），可改 `@functools.cached_property`，但需要 `dataclass(frozen=True, eq=False)` + 自定义 `__hash__`。**当前默认不缓存**——4 维 round 操作极廉价。

### 3.4 默认范围的来源

四维范围全部来自 Ch3.6 表格。**禁止**在本 spec 之外重新定义这些范围。如果需要新范围（如 Ablation 5 sensitivity scan），通过 Pkg-02 env 的 `sample_default` override 参数提供 custom rng，而不是修改 schema。

---

## 4. Edge Cases

| 场景 | 行为 |
|------|------|
| `CapabilityVector(eta=0.5, ...)` | 通过（闭区间下界） |
| `CapabilityVector(eta=2.0, ...)` | ValueError（越上界） |
| `CapabilityVector(eta=float('nan'), ...)` | ValueError（`nan <= x` 为 False，自动拒绝） |
| `cap.eta = 999` | FrozenInstanceError |
| `sample_default(rng)` 在 1e6 次调用 | 全部满足范围（rng 闭区间） |
| `to_batch_tensor([])` | 抛 numpy ValueError（无法 stack 空序列）；Pkg-02 应保证 N ≥ 1 |
| `from_array(np.array([0.6, 3.0, 0.9, 20.0, 0.0]))` | AssertionError（shape != (4,)） |
| pickle 序列化 | dataclass 原生支持 |
| `__hash__` | frozen dataclass 默认按字段 tuple 哈希 |

---

## 5. Acceptance Criteria

### 5.1 单元测试（`tests/schemas/test_capability_vector.py`）

```python
import pytest
import numpy as np
import torch
from hyper_mve.schemas import (
    CapabilityVector, sample_default, sample_n, to_batch_tensor
)

def test_construct_valid():
    cap = CapabilityVector(eta=1.0, phi_fov=3.0, nu=0.9, zeta=20.0)
    assert cap.eta == 1.0
    assert cap.fov_int == 3

def test_boundary_lower():
    cap = CapabilityVector(eta=0.5, phi_fov=2.0, nu=0.8, zeta=10.0)
    assert cap.fov_int == 2

def test_boundary_upper():
    cap = CapabilityVector(eta=1.5, phi_fov=4.0, nu=1.0, zeta=30.0)
    assert cap.fov_int == 4

def test_eta_out_of_range():
    with pytest.raises(ValueError, match="eta.*∉"):
        CapabilityVector(eta=2.0, phi_fov=3.0, nu=0.9, zeta=20.0)
    with pytest.raises(ValueError):
        CapabilityVector(eta=0.4, phi_fov=3.0, nu=0.9, zeta=20.0)

def test_phi_fov_out_of_range():
    with pytest.raises(ValueError, match="phi_fov.*∉"):
        CapabilityVector(eta=1.0, phi_fov=5.0, nu=0.9, zeta=20.0)

def test_nu_out_of_range():
    with pytest.raises(ValueError):
        CapabilityVector(eta=1.0, phi_fov=3.0, nu=0.5, zeta=20.0)

def test_zeta_out_of_range():
    with pytest.raises(ValueError):
        CapabilityVector(eta=1.0, phi_fov=3.0, nu=0.9, zeta=50.0)

def test_nan_rejected():
    with pytest.raises(ValueError):
        CapabilityVector(eta=float('nan'), phi_fov=3.0, nu=0.9, zeta=20.0)

def test_frozen():
    cap = CapabilityVector(eta=1.0, phi_fov=3.0, nu=0.9, zeta=20.0)
    with pytest.raises(Exception):  # FrozenInstanceError
        cap.eta = 0.5  # type: ignore

def test_fov_int_round():
    assert CapabilityVector(eta=1.0, phi_fov=2.4, nu=0.9, zeta=20.0).fov_int == 2
    assert CapabilityVector(eta=1.0, phi_fov=2.5, nu=0.9, zeta=20.0).fov_int == 2  # banker's rounding
    assert CapabilityVector(eta=1.0, phi_fov=2.6, nu=0.9, zeta=20.0).fov_int == 3
    assert CapabilityVector(eta=1.0, phi_fov=3.5, nu=0.9, zeta=20.0).fov_int == 4

def test_to_array_dtype():
    cap = CapabilityVector(eta=1.0, phi_fov=3.0, nu=0.9, zeta=20.0)
    arr = cap.to_array()
    assert arr.shape == (4,)
    assert arr.dtype == np.float32
    assert np.allclose(arr, [1.0, 3.0, 0.9, 20.0])

def test_from_array_roundtrip():
    cap = CapabilityVector(eta=1.2, phi_fov=2.5, nu=0.85, zeta=15.0)
    arr = cap.to_array(dtype=np.float64)
    cap2 = CapabilityVector.from_array(arr)
    assert cap == cap2

def test_sample_default_in_range():
    rng = np.random.default_rng(seed=42)
    for _ in range(1000):
        cap = sample_default(rng)
        # __post_init__ 已保证范围，但显式 assert 防止默认参数错配
        assert 0.5 <= cap.eta <= 1.5
        assert 2.0 <= cap.phi_fov <= 4.0
        assert 0.8 <= cap.nu <= 1.0
        assert 10.0 <= cap.zeta <= 30.0

def test_sample_n_reproducible():
    caps_a = sample_n(4, np.random.default_rng(seed=42))
    caps_b = sample_n(4, np.random.default_rng(seed=42))
    assert caps_a == caps_b  # 相同 seed → 相同输出

def test_to_batch_tensor_shape():
    caps = sample_n(4, np.random.default_rng(seed=0))
    t = to_batch_tensor(caps)
    assert t.shape == (4, 4)
    assert t.dtype == torch.float32

def test_pickle_roundtrip():
    import pickle
    cap = CapabilityVector(eta=1.0, phi_fov=3.0, nu=0.9, zeta=20.0)
    assert pickle.loads(pickle.dumps(cap)) == cap
```

### 5.2 集成测试

- 与 Pkg-02 env 协同测试：env.reset() 返回的 info dict 含 `caps: tuple[CapabilityVector, ...]`，长度 == cfg.env.N
- 与 Pkg-04 model 协同测试：`role_encoder(id, cap, type)` 接受 CapabilityVector 实例

### 5.3 性能要求

- 单 cap 构造 + `__post_init__` 验证 < 5 μs
- `sample_n(8, rng)` < 50 μs（Hard config）
- `to_batch_tensor([cap]*1000)` < 1 ms

---

## 6. Cross-references

- Ch3.6 异质能力定义
- Ch4.2.2 cap_emb MLP 接口
- `../design.md` D2（cap 字段类型）、D7（范围验证策略）
- `01-agent-type-schema.md`（同为 schema 模块）
- `03-observation-layout.md`（cap 进入观测块）
- Pkg-02 spec 02-resource-dynamics.md（cap 在 env 中的使用）
