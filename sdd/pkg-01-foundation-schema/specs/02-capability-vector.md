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

    def normalize(self, dtype: np.dtype = np.float32) -> np.ndarray:
        """返回 (η, φ_fov, ν, ζ) 各维标准化到 [0, 1] 的 (4,) 数组.

        **用途 (v4 修订, Pkg-03 集成需要)**:
        供 Pkg-03 role_encoder cap_mlp 输入归一化用. cap 4 维原始量级悬殊
        (η ∈ [0.5, 1.5], ν ∈ [0.8, 1.0], ζ ∈ [10, 30]; ζ 与 η 量级差 ~30 倍),
        直接喂 Linear 会让第一层权重梯度被 ζ 列主导, 早期 loss landscape 偏斜,
        η/ν 信号在 warmup 期被淹没.

        本方法是**只读 utility**: env / buffer / info 仍存原始 CapabilityVector
        (保持物理可解释性 — render / log / debug 看到的是 η=1.2 而非 0.7).
        仅 cap_emb MLP 消费时调用 normalize() 归一.

        归一化范围使用模块级常量 CAP_NORM_LO / CAP_NORM_HI (Ch3.6 硬约束):
            CAP_NORM_LO = (0.5, 2.0, 0.8, 10.0)
            CAP_NORM_HI = (1.5, 4.0, 1.0, 30.0)

        Returns:
            (4,) array ∈ [0, 1]^4, 字段顺序固定 (eta, phi_fov, nu, zeta).
        """
        raw = np.array([self.eta, self.phi_fov, self.nu, self.zeta], dtype=dtype)
        lo = np.array(CAP_NORM_LO, dtype=dtype)
        hi = np.array(CAP_NORM_HI, dtype=dtype)
        return (raw - lo) / (hi - lo)


# ====== 模块级常量 (v4 修订, Pkg-03 normalize 用) ======

CAP_NORM_LO: tuple[float, float, float, float] = (0.5, 2.0, 0.8, 10.0)
"""cap 归一化下界 (η, φ_fov, ν, ζ), 与 Ch3.6 采样范围一致."""

CAP_NORM_HI: tuple[float, float, float, float] = (1.5, 4.0, 1.0, 30.0)
"""cap 归一化上界 (η, φ_fov, ν, ζ), 与 Ch3.6 采样范围一致."""


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

# 3. model 中 cap_emb 计算 (Pkg-03 role_encoder, v4 修订)
#    cap_mlp 接收 normalize 后的 [0,1] 范围输入, 避免 ζ (10-30) 量级主导
cap_normed = np.stack([cap.normalize() for cap in caps])  # (N, 4) ∈ [0,1]
cap_tensor = torch.from_numpy(cap_normed).to('cuda')      # (N, 4)
cap_embs = cap_mlp(cap_tensor)                            # (N, d_cap_emb=16)

# 4. env FOV 过滤 (Pkg-02) - 用 raw cap (整数视野半径)
visible_radius = caps[i].fov_int  # int ∈ {2, 3, 4}, 不归一

# 5. buffer 与 info 中仍存 raw CapabilityVector (保持物理可解释)
info["caps"]  # tuple[CapabilityVector, ...] - 不归一
record.cap   # (N, 4) raw values - 不归一
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

### 3.5 `normalize()` 的不变量（v4 修订）

**职责分离**：
- env / buffer / info / TimeStepRecord.cap 仍存 **raw** CapabilityVector（保持 Ch3.6 物理可解释性）
- 仅 cap_emb MLP 输入路径调用 `normalize()` 做归一化
- `CAP_NORM_LO/HI` 与 Ch3.6 采样范围（U(0.5,1.5) 等）一一对应，未来若 Ch3.6 范围调整，本常量必须同步

**为什么不在采样时归一化**：
- render / log / debug 时仍需看到 η=1.2 / ζ=20.0 等物理值
- buffer 序列化的字段语义稳定，不依赖归一化范围
- 归一化仅是 cap_emb 训练稳定性的工程优化，不应污染 env/buffer 语义

**为什么不靠 MLP 自学量级**：
- cap_mlp 第一层 `Linear(4, 16)` 仅 64 个权重，ζ 列梯度量级是 η 列 ~30 倍
- Adam 自适应缩放能部分缓解，但早期 warmup 期 η/ν 信号被淹没（cap_emb 直接喂 hyper_rew/hyper_pred 生成 θ）
- 零成本归一化（Ch3.6 范围确定性）换早期收敛速度，性价比高

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
| `cap.normalize()` 边界值 (η=0.5) | 输出第一维 = 0.0 |
| `cap.normalize()` 边界值 (η=1.5) | 输出第一维 = 1.0 |
| `cap.normalize()` 中点 (η=1.0) | 输出第一维 = 0.5 |
| 1000 次随机采样 cap → normalize | 每维 ∈ [0, 1]，均值 ~0.5 |

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


# ====== v4 修订: normalize() utility 单测 ======

def test_capability_normalize_shape_and_dtype():
    """normalize 输出 (4,) float32."""
    cap = CapabilityVector(eta=1.0, phi_fov=3.0, nu=0.9, zeta=20.0)
    out = cap.normalize()
    assert out.shape == (4,)
    assert out.dtype == np.float32


def test_capability_normalize_lower_boundary():
    """每维下界 → 输出 = 0.0."""
    cap = CapabilityVector(eta=0.5, phi_fov=2.0, nu=0.8, zeta=10.0)
    out = cap.normalize()
    assert np.allclose(out, [0.0, 0.0, 0.0, 0.0])


def test_capability_normalize_upper_boundary():
    """每维上界 → 输出 = 1.0."""
    cap = CapabilityVector(eta=1.5, phi_fov=4.0, nu=1.0, zeta=30.0)
    out = cap.normalize()
    assert np.allclose(out, [1.0, 1.0, 1.0, 1.0])


def test_capability_normalize_midpoint():
    """每维中点 → 输出 = 0.5."""
    cap = CapabilityVector(eta=1.0, phi_fov=3.0, nu=0.9, zeta=20.0)
    out = cap.normalize()
    assert np.allclose(out, [0.5, 0.5, 0.5, 0.5])


def test_capability_normalize_eta_only():
    """单维度归一化正确性: eta 在范围内任意值."""
    # eta = 0.5 + 0.3*(1.5-0.5) = 0.8 → normalized = 0.3
    cap = CapabilityVector(eta=0.8, phi_fov=3.0, nu=0.9, zeta=20.0)
    out = cap.normalize()
    assert np.isclose(out[0], 0.3, atol=1e-5)


def test_capability_normalize_zeta_scale_handling():
    """zeta (10-30) 归一化解决量级问题."""
    cap_small = CapabilityVector(eta=1.0, phi_fov=3.0, nu=0.9, zeta=10.0)
    cap_large = CapabilityVector(eta=1.0, phi_fov=3.0, nu=0.9, zeta=30.0)
    # raw 量级差 3x; normalize 后差 1.0 (与其他维度同量级)
    assert np.isclose(cap_small.normalize()[3], 0.0)
    assert np.isclose(cap_large.normalize()[3], 1.0)


def test_capability_normalize_1000_samples_in_unit_range():
    """1000 次采样 cap, normalize 后每维 ∈ [0, 1]; 均值约 0.5."""
    rng = np.random.default_rng(seed=42)
    samples = np.stack([sample_default(rng).normalize() for _ in range(1000)])
    
    assert samples.shape == (1000, 4)
    assert (samples >= 0.0).all()
    assert (samples <= 1.0).all()
    
    means = samples.mean(axis=0)
    # uniform 分布均值 = 0.5, 1000 samples 容差 ~ 0.05
    assert np.allclose(means, [0.5, 0.5, 0.5, 0.5], atol=0.05)


def test_capability_normalize_does_not_modify_raw_fields():
    """normalize 是只读 utility, 不改 frozen 字段."""
    cap = CapabilityVector(eta=1.2, phi_fov=2.5, nu=0.85, zeta=15.0)
    _ = cap.normalize()
    # 原字段仍是 raw values
    assert cap.eta == 1.2
    assert cap.phi_fov == 2.5
    assert cap.nu == 0.85
    assert cap.zeta == 15.0


def test_cap_norm_constants_match_ch36():
    """CAP_NORM_LO/HI 必须与 Ch3.6 采样范围一致 (env/sample_default 同源)."""
    from hyper_mve.schemas.capability import CAP_NORM_LO, CAP_NORM_HI
    assert CAP_NORM_LO == (0.5, 2.0, 0.8, 10.0)
    assert CAP_NORM_HI == (1.5, 4.0, 1.0, 30.0)


def test_capability_normalize_custom_dtype():
    """normalize 接受 dtype 参数."""
    cap = CapabilityVector(eta=1.0, phi_fov=3.0, nu=0.9, zeta=20.0)
    out_f64 = cap.normalize(dtype=np.float64)
    assert out_f64.dtype == np.float64
    assert np.allclose(out_f64, [0.5, 0.5, 0.5, 0.5])
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
- Pkg-03 spec 03-role-encoder.md（cap_mlp 调用 normalize() 输入）
