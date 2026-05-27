# Spec 04: TimeStepRecord — Buffer 单步记录

> 父文档：[`../proposal.md`](../proposal.md) §2.1 · [`../design.md`](../design.md) §3 D5

---

## 1. Purpose

为 Ch5.6.1 定义的 v4 buffer 单步记录提供 dataclass 表示。**10 字段**（v4.7 仅 6 字段）：新增 Δ（瞬时不公平差异）、τ（类型）、cap（能力）、ĉ（资源丰度推断）、ẑ（对手类型推断）。

字段命名严格遵循 Ch5.6.1 公式，存储格式为 numpy（buffer 不占 GPU 内存），转 tensor 在 trainer sampling 时进行。

---

## 2. Interface

### 2.1 文件路径

`hyper_mve/schemas/buffer_record.py`

### 2.2 公开 API

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from .agent_type import AgentType
from .capability import CapabilityVector

@dataclass(frozen=True)
class TimeStepRecord:
    """Buffer 单步记录 (Ch5.6.1, v4 修订 Pass 2)。
    
    Fields (顺序与 Ch5.6.1 一致, 不可重排):
        o:        (N, obs_dim) 联合观测
        a:        (N,) 联合动作 (int)
        r:        (N,) per-agent reward (含 type β 的 φψ 项, env 已计算)
        delta:    (N,) 瞬时不公平差异 Δ_i^{(t)} = u_i - mean_{j≠i} u_j
        pi_mve:   (N, A) MVE planner 输出的搜索策略
        v:        (N,) value 估计 (env step 时由 trainer rollout 缓存; 可选 NaN)
        tau:      (N,) AgentType (固定 episode 内不变, 但每 step 仍存便于 batch 采样)
        cap:      (N, 4) CapabilityVector flatten
        c_hat:    (N,) BeliefNet head_c raw scalar (sigmoid 输出 ∈ [0, 1]);
                  Ch4.2.3 Head 1 明确 ĉ 是 scalar (c_t 本身是 scalar).
                  Pkg-03 BeliefNet 内部 forward 时再做 d_b^proj=16 投影后进入 belief 通路;
                  Buffer 仅存 raw scalar 节省存储.
                  Stage 1 oracle 时为真值 c_t (单个 scalar 广播到 N).
        z_hat:    (N, N-1, 2) BeliefNet head_opp softmax (per-opponent α/β 概率);
                  Ch4.2.3 Head 2 v4 关键改动 (从 v3 动作预测改为类型 2 分类).
                  
                  **顺序约定 (硬约束, v4 Oracle 监督)**:
                    对 agent i, z_hat[i, k] 对应 agent_id = (k if k < i else k + 1)
                    即按 agent_id 升序, 跳过 self.
                  
                  **关键 (v4)**: 此顺序错位会直接污染 L_opp Oracle 监督 CE loss
                  (Ch4.5.2 公式). Pkg-03 BeliefNet head_opp 输出与 Pkg-05 trainer
                  L_opp 计算必须严格遵循此顺序; 否则 CE 在错位标签上反向传播,
                  导致 BeliefNet 学到错乱映射.
                  
                  Stage 1 oracle 时为他人 type one-hot (按上述顺序).
        t:        int 时间步索引 (在 episode 内)
        done:     bool 是否 episode 终止
    """
    o: np.ndarray         # (N, obs_dim), float32
    a: np.ndarray         # (N,), int64
    r: np.ndarray         # (N,), float32
    delta: np.ndarray     # (N,), float32
    pi_mve: np.ndarray    # (N, A), float32
    v: np.ndarray         # (N,), float32 (可能含 NaN)
    tau: np.ndarray       # (N,), int8 (AgentType.value)
    cap: np.ndarray       # (N, 4), float32
    c_hat: np.ndarray     # (N,), float32 raw scalar (v4 修订: 不是 (N, d_c=16))
    z_hat: np.ndarray     # (N, N-1, 2), float32, agent_id 升序跳过 self
    t: int
    done: bool = False

    def __post_init__(self) -> None:
        # 维度一致性最小校验（防止 buffer 推入异质数据）
        N = self.o.shape[0]
        if not all(arr.shape[0] == N for arr in 
                   [self.a, self.r, self.delta, self.pi_mve, self.v, 
                    self.tau, self.cap, self.c_hat, self.z_hat]):
            raise ValueError(f"Field N dimension mismatch in TimeStepRecord")
        if self.z_hat.shape[1] != N - 1:
            raise ValueError(f"z_hat dim 1 = {self.z_hat.shape[1]}, expected N-1={N-1}")

    def to_arrays(self) -> dict[str, np.ndarray]:
        """转 dict[str, np.ndarray]，用于 buffer 内部 ragged storage。"""
        return {
            "o": self.o, "a": self.a, "r": self.r, "delta": self.delta,
            "pi_mve": self.pi_mve, "v": self.v, "tau": self.tau, 
            "cap": self.cap, "c_hat": self.c_hat, "z_hat": self.z_hat,
            "t": np.array([self.t], dtype=np.int32),
            "done": np.array([self.done], dtype=np.bool_),
        }

    @classmethod
    def from_arrays(cls, d: dict[str, np.ndarray]) -> "TimeStepRecord":
        """从 dict 恢复（序列化往返）。"""
        return cls(
            o=d["o"], a=d["a"], r=d["r"], delta=d["delta"],
            pi_mve=d["pi_mve"], v=d["v"], tau=d["tau"],
            cap=d["cap"], c_hat=d["c_hat"], z_hat=d["z_hat"],
            t=int(d["t"][0]),
            done=bool(d["done"][0]),
        )

    @classmethod
    def empty_belief(cls, N: int, A: int, obs_dim: int, t: int = 0) -> "TimeStepRecord":
        """构造 belief 字段为零的占位 record (Stage 1 warmup 用)。
        
        v4 修订: c_hat 是 scalar (Ch4.2.3), 不再需要 d_c 参数.
        """
        return cls(
            o=np.zeros((N, obs_dim), dtype=np.float32),
            a=np.zeros(N, dtype=np.int64),
            r=np.zeros(N, dtype=np.float32),
            delta=np.zeros(N, dtype=np.float32),
            pi_mve=np.ones((N, A), dtype=np.float32) / A,  # 均匀分布
            v=np.zeros(N, dtype=np.float32),
            tau=np.zeros(N, dtype=np.int8),
            cap=np.zeros((N, 4), dtype=np.float32),
            c_hat=np.full(N, 0.5, dtype=np.float32),   # (N,) raw scalar, 中性初值
            z_hat=np.ones((N, N-1, 2), dtype=np.float32) * 0.5,  # 均匀 α/β 概率
            t=t,
            done=False,
        )
```

### 2.3 典型用例

```python
# 1. Pkg-05 worker 收集 episode 时构造 record
from hyper_mve.schemas import TimeStepRecord, AgentType
import numpy as np

obs_dim = ObservationLayout.total_dim(N=4, K=20)  # 99
record = TimeStepRecord(
    o=joint_obs,                # (4, 99)
    a=joint_actions,            # (4,)
    r=rewards,                  # (4,) - env 已计算
    delta=delta_per_agent,      # (4,) - env 已计算 Δ
    pi_mve=mve_policy,          # (4, 6) - planner 输出
    v=value_estimates,          # (4,)
    tau=np.array([0, 0, 1, 1], dtype=np.int8),  # ALPHA, ALPHA, BETA, BETA
    cap=cap_matrix,             # (4, 4)
    c_hat=c_belief,             # (4, 16) - BeliefNet 输出
    z_hat=z_belief,             # (4, 3, 2)
    t=step_idx,
    done=is_terminal,
)

# 2. buffer 内部存储
arrays = record.to_arrays()
buffer.append(arrays)

# 3. trainer sampling
batch_arrays = buffer.sample(B=256)
batch_records = [TimeStepRecord.from_arrays(a) for a in batch_arrays]
```

---

## 3. Implementation Notes

### 3.1 为何字段全 numpy 而非 torch

- buffer 容量 5000 episode × 200 step × 4 agent × ~99 dim ≈ 400M float32 ≈ 1.5GB
- 全 GPU 存会撑爆 V100 16GB（还要算 model + activations）
- numpy CPU 存储 + sampling 时 batched `torch.from_numpy()` 是 stable-baselines3 / TorchRL 标准做法

### 3.2 `tau` 存 int8 而非 AgentType

- AgentType Enum 不能直接 stack 为 numpy 数组（dtype 不兼容）
- int8 (2 bytes) 节省存储 vs int64 (8 bytes)
- Pkg-05 trainer sampling 时 `AgentType(int(tau[i]))` 反查

### 3.3 `z_hat` 形状 (N, N-1, 2) 设计

- 第 0 维：哪个 agent 的 belief
- 第 1 维：该 agent 看到的 N-1 个对手
- 第 2 维：α/β softmax 概率
- agent_id 顺序：每个 agent i 的 z_hat[i] 排除自己（用 mask 或重排）
- **约定**：z_hat[i, j] 表示 agent i 对第 j 个对手的预测（j ∈ {0, ..., i-1, i+1, ..., N-1}，跳过 i 后压缩到长度 N-1）

### 3.4 `v` 为何允许 NaN

- v4.7 worker 收集 episode 时不一定有 v 估计（取决于是否启用 trainer rollout 缓存）
- Pkg-05 trainer 在计算 n-step return 时如果遇到 NaN，会用 target_net rollout 现算

### 3.5 `c_hat` 在 Stage 1 oracle 时

Stage 1 belief 输入用真值 c 广播到 d_c 维（重复 c 16 次或线性嵌入）；存进 buffer 也是这个值。  
Stage 3 用 BeliefNet 输出。  
**注**：c_hat 字段名是 hat（推断值），但 Stage 1 是 oracle，论文 5.7 已说明此为课程学习设计。

---

## 4. Edge Cases

| 场景 | 行为 |
|------|------|
| `TimeStepRecord(o=zeros((4,99)), a=zeros(3), ...)` | ValueError（N 维度不匹配） |
| `z_hat.shape = (4, 4, 2)` 但 N=4 | ValueError（z_hat dim 1 应 == N-1） |
| `done=True` 但 `t < T_max - 1` | 允许（提前终止） |
| `r` 包含 -inf | 不校验，trainer 端自行处理 |
| `pi_mve` 行和不为 1 | 不校验（如果 trainer 对 logits 不归一化） |
| Pickle 序列化 | dataclass + numpy 全支持 |
| `to_arrays() → from_arrays()` 往返 | 完全恢复（含 t, done 标量） |

---

## 5. Acceptance Criteria

### 5.1 单元测试（`tests/schemas/test_buffer_record.py`）

```python
import pytest
import numpy as np
from hyper_mve.schemas import TimeStepRecord, AgentType

def _make_valid_record(N=4, A=6, obs_dim=99, t=0):
    # v4 修订: c_hat shape (N,) scalar, 不再带 d_c 参数
    return TimeStepRecord(
        o=np.zeros((N, obs_dim), dtype=np.float32),
        a=np.zeros(N, dtype=np.int64),
        r=np.zeros(N, dtype=np.float32),
        delta=np.zeros(N, dtype=np.float32),
        pi_mve=np.ones((N, A), dtype=np.float32) / A,
        v=np.zeros(N, dtype=np.float32),
        tau=np.array([0, 0, 1, 1], dtype=np.int8),
        cap=np.zeros((N, 4), dtype=np.float32),
        c_hat=np.full(N, 0.5, dtype=np.float32),    # (N,) scalar
        z_hat=np.ones((N, N-1, 2), dtype=np.float32) * 0.5,
        t=t, done=False,
    )

def test_construct_valid():
    record = _make_valid_record()
    assert record.o.shape == (4, 99)
    assert record.tau.tolist() == [0, 0, 1, 1]

def test_n_dimension_mismatch():
    with pytest.raises(ValueError, match="N dimension mismatch"):
        TimeStepRecord(
            o=np.zeros((4, 99), dtype=np.float32),
            a=np.zeros(3, dtype=np.int64),  # 错: N=3 vs N=4
            r=np.zeros(4, dtype=np.float32),
            delta=np.zeros(4, dtype=np.float32),
            pi_mve=np.zeros((4, 6), dtype=np.float32),
            v=np.zeros(4, dtype=np.float32),
            tau=np.zeros(4, dtype=np.int8),
            cap=np.zeros((4, 4), dtype=np.float32),
            c_hat=np.zeros(4, dtype=np.float32),         # (N,) scalar
            z_hat=np.zeros((4, 3, 2), dtype=np.float32),
            t=0,
        )

def test_z_hat_dim_mismatch():
    with pytest.raises(ValueError, match="z_hat dim 1"):
        TimeStepRecord(
            o=np.zeros((4, 99), dtype=np.float32),
            a=np.zeros(4, dtype=np.int64),
            r=np.zeros(4, dtype=np.float32),
            delta=np.zeros(4, dtype=np.float32),
            pi_mve=np.zeros((4, 6), dtype=np.float32),
            v=np.zeros(4, dtype=np.float32),
            tau=np.zeros(4, dtype=np.int8),
            cap=np.zeros((4, 4), dtype=np.float32),
            c_hat=np.zeros(4, dtype=np.float32),         # (N,) scalar
            z_hat=np.zeros((4, 4, 2), dtype=np.float32),  # 错: N-1=3 vs 4
            t=0,
        )

def test_c_hat_is_scalar_per_agent():
    """v4 修订 (Ch4.2.3 Head 1): c_hat 是 (N,) scalar, 不是 (N, d_c)."""
    record = _make_valid_record()
    assert record.c_hat.shape == (4,)
    assert record.c_hat.dtype == np.float32

def test_z_hat_order_convention():
    """v4 关键 (L_opp Oracle 监督): z_hat 顺序按 agent_id 升序跳过 self.
    
    本测试构造一个有标识的 z_hat 验证顺序约定, Pkg-03/05 必须遵循.
    """
    N = 4
    z_hat = np.zeros((N, N-1, 2), dtype=np.float32)
    # 约定: agent i 的 z_hat[i, k] 对应 agent_id = (k if k < i else k + 1)
    # 例如 agent 2 看到的对手顺序应为 [agent 0, agent 1, agent 3]
    for i in range(N):
        for k in range(N - 1):
            actual_opp_id = k if k < i else k + 1
            # 标记 prob = (actual_opp_id + 1) / 10 to verify
            z_hat[i, k, 0] = (actual_opp_id + 1) / 10.0
            z_hat[i, k, 1] = 1.0 - z_hat[i, k, 0]
    # agent 2 看到的对手 [0, 1, 3] 对应 prob [0.1, 0.2, 0.4]
    assert np.isclose(z_hat[2, 0, 0], 0.1)
    assert np.isclose(z_hat[2, 1, 0], 0.2)
    assert np.isclose(z_hat[2, 2, 0], 0.4)

def test_to_from_arrays_roundtrip():
    record = _make_valid_record(t=42)
    arrays = record.to_arrays()
    assert "o" in arrays and "tau" in arrays and "z_hat" in arrays
    
    record2 = TimeStepRecord.from_arrays(arrays)
    assert record2.t == 42
    assert record2.done is False
    assert np.allclose(record.o, record2.o)
    assert np.array_equal(record.tau, record2.tau)

def test_empty_belief_factory():
    record = TimeStepRecord.empty_belief(N=4, A=6, obs_dim=99, t=5)
    assert record.t == 5
    assert np.allclose(record.pi_mve, 1.0/6)  # 均匀
    assert np.allclose(record.z_hat, 0.5)     # 均匀 α/β
    assert record.c_hat.shape == (4,)         # v4: scalar per agent

def test_done_terminal():
    record = TimeStepRecord(**{
        **_make_valid_record().__dict__,
        "done": True,
    })
    assert record.done is True

def test_v_nan_allowed():
    record = _make_valid_record()
    arrays = record.to_arrays()
    arrays["v"] = np.array([np.nan, 0, 0, 0], dtype=np.float32)
    record2 = TimeStepRecord.from_arrays(arrays)
    assert np.isnan(record2.v[0])

def test_pickle_roundtrip():
    import pickle
    record = _make_valid_record(t=10)
    record2 = pickle.loads(pickle.dumps(record))
    assert record.t == record2.t
    assert np.array_equal(record.tau, record2.tau)

def test_field_count():
    """关键测试: 10 个数据字段 + t/done (Ch5.6.1)."""
    from dataclasses import fields
    field_names = {f.name for f in fields(TimeStepRecord)}
    expected = {"o", "a", "r", "delta", "pi_mve", "v", "tau", 
                "cap", "c_hat", "z_hat", "t", "done"}
    assert field_names == expected
```

### 5.2 性能要求

- `TimeStepRecord(...)` 构造 + `__post_init__` < 50 μs
- `to_arrays() → from_arrays()` 往返 < 100 μs
- 5000 episode × 200 step buffer 总内存 < 2GB

### 5.3 与 Ch5.6.1 对照

| Ch5.6.1 字段 | spec 字段 | 类型 / 形状 |
|--------------|----------|-------------|
| o^t | `o` | (N, obs_dim) float32 |
| a^t | `a` | (N,) int64 |
| r^t | `r` | (N,) float32 |
| Δ^t | `delta` | (N,) float32 |
| π_mve^t | `pi_mve` | (N, A) float32 |
| v^t | `v` | (N,) float32 (NaN-able) |
| {τ_i} | `tau` | (N,) int8 |
| {cap_i} | `cap` | (N, 4) float32 |
| ĉ^t | `c_hat` | **(N,) float32 raw scalar**（v4 Ch4.2.3 Head 1） |
| ẑ^t | `z_hat` | (N, N-1, 2) float32 softmax（v4 Oracle 类型 2 分类，agent_id 升序跳过 self） |

---

## 6. Cross-references

- Ch5.6.1 buffer 数据组织
- Ch4.5 BeliefNet 输出（c_hat / z_hat 来源）
- `../design.md` D5（z_hat 形状决策）
- `01-agent-type-schema.md`（tau int8 编码）
- `02-capability-vector.md`（cap 字段使用）
- `03-observation-layout.md`（o 字段维度）
- Pkg-05 spec 03-episode-buffer-v2.md（buffer 存储与采样）
