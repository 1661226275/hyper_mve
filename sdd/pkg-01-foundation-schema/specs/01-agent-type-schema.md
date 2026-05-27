# Spec 01: AgentType Enum 与嵌入

> 父文档：[`../proposal.md`](../proposal.md) §2.1 · [`../design.md`](../design.md) §3 D1

---

## 1. Purpose

为 Ch3.5.1 定义的类型机制 τ_i ∈ {α, β} 提供**类型安全的 Python 表示**，统一全工作区的类型编码（避免 0/1 int vs one-hot 浮点 vs 字符串 'alpha'/'beta' 散落实现）。

---

## 2. Interface

### 2.1 文件路径

`hyper_mve/schemas/agent_type.py`

### 2.2 公开 API

```python
import enum
import numpy as np
import torch
from typing import Sequence

class AgentType(enum.IntEnum):
    """Agent 偏好类型 (Ch3.5.1)。
    
    - ALPHA = 0: 纯自利 (R_i = u_i - ε·𝟙[moved])
    - BETA  = 1: Fehr-Schmidt 不公平厌恶 (R_i = u_i - ε·𝟙[moved] + φ(c)·ψ(Δ))
    
    选择 IntEnum 让 enum value 可直接用作 nn.Embedding 索引和 numpy int 化 (Ch4.2.2)。
    """
    ALPHA = 0
    BETA = 1

# 工具函数 (module level)

def one_hot(t: AgentType, dtype: np.dtype = np.float32) -> np.ndarray:
    """返回 (2,) one-hot 数组，仅用于六块观测中 o_i^{type} 字段。"""
    arr = np.zeros(2, dtype=dtype)
    arr[t.value] = 1.0
    return arr

def from_index(idx: int) -> AgentType:
    """int → AgentType 反查。idx ∉ {0, 1} 抛 ValueError。"""
    if idx not in (0, 1):
        raise ValueError(f"AgentType index {idx} ∉ {{0, 1}}")
    return AgentType(idx)

def from_str(name: str) -> AgentType:
    """'alpha' / 'ALPHA' / 'α' / 'beta' / 'BETA' / 'β' → AgentType。"""
    norm = name.strip().lower()
    if norm in ("alpha", "α", "a"):
        return AgentType.ALPHA
    elif norm in ("beta", "β", "b"):
        return AgentType.BETA
    raise ValueError(f"Cannot parse AgentType from {name!r}")

def count_in_assignment(types: Sequence[AgentType]) -> dict[AgentType, int]:
    """统计 type 分配中各类型数量（Ablation 3 类型扫描用）。
    
    Example:
        >>> count_in_assignment([ALPHA, ALPHA, BETA, BETA])
        {AgentType.ALPHA: 2, AgentType.BETA: 2}
    """
    return {AgentType.ALPHA: sum(1 for t in types if t == AgentType.ALPHA),
            AgentType.BETA:  sum(1 for t in types if t == AgentType.BETA)}

def to_long_tensor(types: Sequence[AgentType], device: torch.device | None = None) -> torch.Tensor:
    """Sequence[AgentType] → (N,) long tensor，供 nn.Embedding 输入。"""
    arr = np.array([t.value for t in types], dtype=np.int64)
    t = torch.from_numpy(arr)
    return t.to(device) if device else t
```

### 2.3 典型用例

```python
# 1. 在 env 中初始化 type 分配 (Pkg-02)
from hyper_mve.schemas import AgentType
agent_types = [AgentType.ALPHA, AgentType.ALPHA, AgentType.BETA, AgentType.BETA]

# 2. 观测 type 块构造 (Ch3.7 Self-Info)
own_type_obs = one_hot(agent_types[i])  # shape (2,)

# 3. model 中 type_emb 查表 (Pkg-04)
type_ids = to_long_tensor(agent_types, device='cuda')  # (N,)
type_embs = type_embedding(type_ids)  # (N, d_type=4)

# 4. Ablation 3 扫描配置（Pkg-08 实验 driver）
for n_alpha, n_beta in [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)]:
    types = [AgentType.ALPHA] * n_alpha + [AgentType.BETA] * n_beta
    # ... run experiment
```

---

## 3. Implementation Notes

### 3.1 为何选 `IntEnum` 而非 `Enum`

- `IntEnum` 可直接转 int：`int(AgentType.ALPHA) == 0`
- numpy / torch 索引时无需 `.value`：`embedding(AgentType.ALPHA)` 自动工作
- JSON 序列化时 `json.dumps(AgentType.ALPHA)` 直接输出 0（IntEnum 子类是 int）

### 3.2 自定义 JSON encoder（可选）

如果实验日志需要可读：

```python
class AgentTypeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, AgentType):
            return {"_type": "AgentType", "name": obj.name, "value": obj.value}
        return super().default(obj)
```

但默认行为（输出 int）已足够 Pkg-08 的 csv 日志使用。

### 3.3 反索引性能

`AgentType(0)` / `AgentType(1)` 是 O(1) 字典查表（Enum 内部缓存）；`from_index` 函数主要是为了显式 ValueError，性能可忽略。

---

## 4. Edge Cases

| 场景 | 行为 |
|------|------|
| `AgentType(2)` | 抛 `ValueError`（Enum 内置） |
| `AgentType("ALPHA")` | 抛 `ValueError`；用 `from_str` |
| `one_hot(AgentType.ALPHA, dtype=np.int8)` | 返回 int8 数组 `[1, 0]` |
| `to_long_tensor([])` | 返回 shape (0,) tensor，**不报错**（允许 N=0 边界） |
| pickle 序列化 | IntEnum 原生支持 |
| `np.array([AgentType.ALPHA, AgentType.BETA])` | 自动转 `array([0, 1])`，dtype=int64 |

---

## 5. Acceptance Criteria

### 5.1 单元测试（`tests/schemas/test_agent_type.py`）

```python
def test_enum_value_stable():
    assert AgentType.ALPHA.value == 0
    assert AgentType.BETA.value == 1

def test_one_hot_shape_and_dtype():
    arr = one_hot(AgentType.ALPHA)
    assert arr.shape == (2,)
    assert arr.dtype == np.float32
    assert np.allclose(arr, [1.0, 0.0])

def test_one_hot_beta():
    arr = one_hot(AgentType.BETA)
    assert np.allclose(arr, [0.0, 1.0])

def test_from_index_valid():
    assert from_index(0) == AgentType.ALPHA
    assert from_index(1) == AgentType.BETA

def test_from_index_invalid():
    with pytest.raises(ValueError, match="∉.*0.*1"):
        from_index(2)
    with pytest.raises(ValueError):
        from_index(-1)

def test_from_str_variants():
    assert from_str("alpha") == AgentType.ALPHA
    assert from_str("ALPHA") == AgentType.ALPHA
    assert from_str("α") == AgentType.ALPHA
    assert from_str("a") == AgentType.ALPHA
    assert from_str("beta") == AgentType.BETA
    with pytest.raises(ValueError):
        from_str("gamma")

def test_count_in_assignment():
    types = [AgentType.ALPHA, AgentType.ALPHA, AgentType.BETA, AgentType.BETA]
    counts = count_in_assignment(types)
    assert counts[AgentType.ALPHA] == 2
    assert counts[AgentType.BETA] == 2

def test_to_long_tensor():
    types = [AgentType.ALPHA, AgentType.BETA, AgentType.BETA]
    t = to_long_tensor(types)
    assert t.dtype == torch.int64
    assert t.tolist() == [0, 1, 1]

def test_pickle_roundtrip():
    import pickle
    assert pickle.loads(pickle.dumps(AgentType.BETA)) == AgentType.BETA

def test_intenum_arithmetic():
    """IntEnum 可直接作为 int 使用 (Pkg-04 nn.Embedding 索引)."""
    assert int(AgentType.ALPHA) + 0 == 0
    assert AgentType.BETA == 1   # IntEnum 与 int 相等比较
```

### 5.2 集成测试

- `from hyper_mve.schemas import AgentType` 顶层 import 成功（不需要任何 torch / numpy 导入）
- `mypy --strict hyper_mve/schemas/agent_type.py` 零错误

### 5.3 跨包验证

Pkg-02 env 实例化时 type_assignment 字段类型应为 `tuple[AgentType, ...]`（不可变）；Pkg-04 model 的 `set_context(c_t, agent_id, ..., type=AgentType.ALPHA)` 参数注解严格用 AgentType。

---

## 6. Cross-references

- Ch3.5.1 类型机制定义
- Ch4.2.2 type_emb 查表机制
- `../design.md` D1 编码方式决策
- `02-capability-vector.md`（同为 schema 模块）
- `04-timestep-record.md`（`tau: AgentType` 字段使用）
