# Spec 03: 六块观测张量布局

> 父文档：[`../proposal.md`](../proposal.md) §2.1 · [`../design.md`](../design.md) §3 D5

---

## 1. Purpose

为 Ch3.7 定义的六块观测结构提供**静态维度常量**与**切片/padding 工具**。所有维度可在不实例化任何 env 的前提下计算（用于 model init 时申请 MLP 第一层权重）。

**关键原则（Self-Info, Ch3.7）**：`o_i^{type}` 块**仅含 agent i 自身**的 type one-hot（2 维），不含其他人——他人类型由 BeliefNet 推断 ẑ_{i,j}。

---

## 2. Interface

### 2.1 文件路径

`hyper_mve/schemas/observation.py`

### 2.2 公开 API

```python
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ObservationBlockSpec:
    """单个观测块的规格 (Ch3.7)."""
    name: str
    per_item_dim: int       # 单个 entity 的特征维度
    item_count_formula: str # 形如 "1" / "K" / "N-1" - 用于文档与维度推导

class ObservationLayout:
    """六块观测张量布局 (Ch3.7)。
    
    Self-Info Principle (Ch3.7):
        - o_i^{type} 仅含自身 type one-hot (维度固定 2)
        - 他人类型由 BeliefNet 推断 (Pkg-03)
    
    布局顺序 (固定, 不可重排):
        1. self        - 自身状态        (4 维)
        2. resource    - FOV 内资源点    (K × 3, padded)
        3. neighbor    - FOV 内其他 agent (N-1 × 9, padded)
        4. global      - c_t + episode 剩余比例 (2 维)
        5. capability  - 自身 cap 向量    (4 维)
        6. type        - 自身 type one-hot (2 维)
    """
    
    # 各块单 entity 维度 (Ch3.7 表格定型)
    SELF_DIM = 4              # (x_norm, y_norm, cum_harvest_norm, steps_since_harvest_norm)
    RESOURCE_PER_ITEM = 3     # (rel_dx, rel_dy, q_norm) per resource point
    NEIGHBOR_PER_ITEM = 9     # (rel_dx, rel_dy, last_action_onehot[6], presence_flag) per other agent
    GLOBAL_DIM = 2            # (c_t, episode_time_remaining_ratio)
    CAPABILITY_DIM = 4        # (eta, phi_fov, nu, zeta) 与 CapabilityVector 一致
    TYPE_DIM = 2              # one-hot α/β (Self-Info: only own type)
    
    # 块顺序（与上表对应）
    BLOCK_ORDER: tuple[str, ...] = (
        "self", "resource", "neighbor", "global", "capability", "type"
    )
    
    @staticmethod
    def block_dim(block_name: str, N: int, K: int) -> int:
        """单个块的总维度（已 flatten）。
        
        Args:
            N: 总 agent 数（含 self）
            K: 总资源点数（不论 FOV 内外，因为要 padding 到 K）
        
        Returns:
            该块 flatten 后的维度。
        """
        if block_name == "self":
            return ObservationLayout.SELF_DIM
        elif block_name == "resource":
            return K * ObservationLayout.RESOURCE_PER_ITEM
        elif block_name == "neighbor":
            return (N - 1) * ObservationLayout.NEIGHBOR_PER_ITEM
        elif block_name == "global":
            return ObservationLayout.GLOBAL_DIM
        elif block_name == "capability":
            return ObservationLayout.CAPABILITY_DIM
        elif block_name == "type":
            return ObservationLayout.TYPE_DIM
        raise ValueError(f"Unknown block name: {block_name}")
    
    @staticmethod
    def total_dim(N: int, K: int) -> int:
        """单 agent 观测总维度（全部 6 块 flatten 拼接）。
        
        formula: 4 + 3K + 9(N-1) + 2 + 4 + 2 = 12 + 3K + 9(N-1)
        
        Example:
            Medium (N=4, K=20): 12 + 60 + 27 = 99
            Easy   (N=2, K=8):  12 + 24 + 9  = 45
            Hard   (N=8, K=40): 12 + 120 + 63 = 195
        """
        return sum(ObservationLayout.block_dim(b, N, K) 
                   for b in ObservationLayout.BLOCK_ORDER)
    
    @staticmethod
    def block_offset(block_name: str, N: int, K: int) -> tuple[int, int]:
        """指定块在 flatten 观测中的 (start, end) 索引。
        
        Example:
            >>> ObservationLayout.block_offset("type", N=4, K=20)
            (97, 99)   # 最后两维
        """
        if block_name not in ObservationLayout.BLOCK_ORDER:
            raise ValueError(f"Unknown block: {block_name}")
        start = 0
        for b in ObservationLayout.BLOCK_ORDER:
            d = ObservationLayout.block_dim(b, N, K)
            if b == block_name:
                return (start, start + d)
            start += d
        raise RuntimeError("unreachable")

# Padding 工具

def pad_resource_block(visible: list[tuple[float, float, float]], 
                       K: int) -> np.ndarray:
    """将 FOV 内可见资源点 padding 到固定 K 个 entity。
    
    Args:
        visible: list of (rel_dx, rel_dy, q_norm)，长度 ≤ K
        K: 总资源点数
    
    Returns:
        (K * 3,) flatten 数组；不可见位填 0。
    """
    out = np.zeros((K, 3), dtype=np.float32)
    for i, (dx, dy, q) in enumerate(visible[:K]):
        out[i] = (dx, dy, q)
    return out.reshape(-1)

def pad_neighbor_block(visible: list[tuple[float, float, np.ndarray, bool]], 
                       N: int) -> np.ndarray:
    """将 FOV 内可见 neighbor padding 到 N-1 个 slot。
    
    Args:
        visible: list of (rel_dx, rel_dy, last_action_onehot[6], present_flag)
        N: 总 agent 数
    
    Returns:
        ((N-1) * 9,) flatten 数组；不可见位填 0。
    """
    out = np.zeros((N - 1, 9), dtype=np.float32)
    for i, (dx, dy, act_oh, presence) in enumerate(visible[:N-1]):
        out[i, 0] = dx
        out[i, 1] = dy
        out[i, 2:8] = act_oh  # one-hot of 6 actions (NOOP, UP, DOWN, LEFT, RIGHT, HARVEST)
        out[i, 8] = float(presence)
    return out.reshape(-1)

def slice_block(obs: np.ndarray, block_name: str, N: int, K: int) -> np.ndarray:
    """从 flatten 观测中切出某个块（调试与可视化用）。"""
    start, end = ObservationLayout.block_offset(block_name, N, K)
    return obs[start:end]
```

### 2.3 典型用例

```python
# 1. 计算观测总维度 (Pkg-04 model init 时申请 RepNet 输入维度)
from hyper_mve.schemas import ObservationLayout
obs_dim = ObservationLayout.total_dim(N=4, K=20)
self.rep_net = nn.Linear(obs_dim, latent_dim)  # 99 → 64

# 2. env 构造观测 (Pkg-02)
def observe(self, i: int) -> np.ndarray:
    blocks = []
    blocks.append(self_block(i))  # (4,)
    blocks.append(pad_resource_block(visible_resources(i), K=self.K))  # (60,)
    blocks.append(pad_neighbor_block(visible_neighbors(i), N=self.N))  # (27,)
    blocks.append(global_block())  # (2,)
    blocks.append(self.caps[i].to_array())  # (4,)
    blocks.append(one_hot(self.types[i]))  # (2,)
    return np.concatenate(blocks)  # (99,)

# 3. 调试切片
type_block = slice_block(obs, "type", N=4, K=20)  # (2,)
print(f"Agent type one-hot: {type_block}")  # e.g. [1.0, 0.0] for ALPHA
```

---

## 3. Implementation Notes

### 3.1 为何 neighbor per_item_dim = 9（不是 7）

Ch3.7 表格：`(rel_dx, rel_dy, last_action_onehot)` = 2 + 6 = 8 维。  
本 spec **加 1 维 `presence_flag`**（0/1 标记该 slot 是否真有 agent vs 是 padding）。  
理由：FOV 外的 agent 与 FOV 内但还未行动的 agent，仅靠 `last_action_onehot == [1,0,0,0,0,0]`（NOOP）无法区分。presence_flag 显式标识。  
**注**：这是对 Ch3.7 表的轻微 augment，需要在论文实施小节注明。如果用户审阅时反对，可去掉（neighbor_per_item = 8，total_dim 公式相应调整）。

### 3.2 padding 策略

- 资源块：visible[:K] 截断；不可见位填 (0, 0, 0)
- neighbor 块：visible[:N-1] 截断；不可见位填 (0, 0, [0]*6, 0)，presence_flag=0
- Pkg-04 model 可选择用 attention mask 进一步利用 presence 信息

### 3.3 dtype 统一 float32

所有块 flatten 后统一 `np.float32`，对应 PyTorch 默认浮点类型。如果未来需要 mixed precision（fp16），仅在 model 层 cast，不修改 schema。

### 3.4 块顺序为何"type"放最后

- type 是 2 维，放最后便于调试时 `obs[-2:]` 一眼看到
- 与 capability（也是自身静态信息）相邻，语义聚类
- model 端通过 `block_offset` 切片，不依赖具体位置

---

## 4. Edge Cases

| 场景 | 行为 |
|------|------|
| `total_dim(N=1, K=0)` | 4 + 0 + 0 + 2 + 4 + 2 = 12（边界，无 neighbor 无 resource） |
| `block_dim("foo", ...)` | ValueError |
| `pad_resource_block([], K=20)` | (60,) 全零数组 |
| `pad_resource_block([...K+1 items], K=20)` | 截断到前 K 个 |
| `slice_block(obs, "type", N=4, K=20)` 但 obs 维度错 | numpy 切片自动报错或返回空（依赖 obs 形状） |
| FOV 半径变化 | layout 不变（visible 列表长度变，padding 后维度固定） |

---

## 5. Acceptance Criteria

### 5.1 单元测试（`tests/schemas/test_observation_layout.py`）

```python
import pytest
import numpy as np
from hyper_mve.schemas import (
    ObservationLayout, pad_resource_block, pad_neighbor_block, slice_block
)

def test_total_dim_medium():
    """Medium config (N=4, K=20): 4 + 60 + 27 + 2 + 4 + 2 = 99."""
    assert ObservationLayout.total_dim(N=4, K=20) == 99

def test_total_dim_easy():
    """Easy config (N=2, K=8): 4 + 24 + 9 + 2 + 4 + 2 = 45."""
    assert ObservationLayout.total_dim(N=2, K=8) == 45

def test_total_dim_hard():
    """Hard config (N=8, K=40): 4 + 120 + 63 + 2 + 4 + 2 = 195."""
    assert ObservationLayout.total_dim(N=8, K=40) == 195

def test_block_dim_each():
    assert ObservationLayout.block_dim("self", 4, 20) == 4
    assert ObservationLayout.block_dim("resource", 4, 20) == 60
    assert ObservationLayout.block_dim("neighbor", 4, 20) == 27
    assert ObservationLayout.block_dim("global", 4, 20) == 2
    assert ObservationLayout.block_dim("capability", 4, 20) == 4
    assert ObservationLayout.block_dim("type", 4, 20) == 2

def test_block_dim_invalid():
    with pytest.raises(ValueError, match="Unknown block"):
        ObservationLayout.block_dim("foo", 4, 20)

def test_block_offset_type_is_last_two():
    start, end = ObservationLayout.block_offset("type", N=4, K=20)
    assert end == 99
    assert end - start == 2

def test_block_offset_self_is_first():
    start, end = ObservationLayout.block_offset("self", N=4, K=20)
    assert start == 0
    assert end == 4

def test_block_offsets_sum_to_total():
    N, K = 4, 20
    for block in ObservationLayout.BLOCK_ORDER:
        s, e = ObservationLayout.block_offset(block, N, K)
        assert e - s == ObservationLayout.block_dim(block, N, K)

def test_pad_resource_block_empty():
    out = pad_resource_block([], K=20)
    assert out.shape == (60,)
    assert np.allclose(out, 0)

def test_pad_resource_block_partial():
    visible = [(0.1, 0.2, 0.5), (0.3, 0.4, 0.8)]
    out = pad_resource_block(visible, K=20)
    assert out.shape == (60,)
    assert np.allclose(out[:6], [0.1, 0.2, 0.5, 0.3, 0.4, 0.8])
    assert np.allclose(out[6:], 0)

def test_pad_resource_block_overflow():
    visible = [(0.1, 0.2, 0.5)] * 25  # > K=20
    out = pad_resource_block(visible, K=20)
    assert out.shape == (60,)
    # 仅保留前 20 个

def test_pad_neighbor_block_with_presence():
    visible = [
        (0.0, 0.1, np.array([1,0,0,0,0,0], dtype=np.float32), True),
        (0.0, 0.0, np.array([0,1,0,0,0,0], dtype=np.float32), False),
    ]
    out = pad_neighbor_block(visible, N=4)  # N-1 = 3 slots
    assert out.shape == (27,)
    # slot 0: dx=0, dy=0.1, act=[1,0,0,0,0,0], presence=1
    assert out[0] == 0.0 and out[1] == 0.1
    assert np.allclose(out[2:8], [1,0,0,0,0,0])
    assert out[8] == 1.0
    # slot 1: presence=0
    assert out[17] == 0.0  # presence flag

def test_slice_block_type():
    """构造一个 99 维观测，切 type 块。"""
    obs = np.arange(99, dtype=np.float32)
    type_block = slice_block(obs, "type", N=4, K=20)
    assert type_block.shape == (2,)
    assert np.allclose(type_block, [97.0, 98.0])

def test_self_info_principle():
    """关键测试：type 块只有 2 维 (own type one-hot)，不是 N*2 维。"""
    assert ObservationLayout.TYPE_DIM == 2
    # 即使 N=8，type 块仍是 2
    assert ObservationLayout.block_dim("type", N=8, K=40) == 2
```

### 5.2 集成测试

- Pkg-02 env.reset() 后 `obs.shape == (cfg.env.N, ObservationLayout.total_dim(N, K))`
- Pkg-04 model RepNet 输入维度 == `ObservationLayout.total_dim(cfg.env.N, cfg.env.K)`
- 跨包一致性：env 输出维度 == model 期望维度（在 V4Config preset 加载后自动对齐）

### 5.3 与 Ch3.7 文档对照

| Block | Ch3.7 描述 | Spec 维度 |
|-------|-----------|-----------|
| `o_i^{self}` | position, cumulative harvest, steps since last harvest | 4 |
| `o_i^{resource}` | rel (Δx, Δy), norm stock | K × 3 |
| `o_i^{neighbor}` | rel (Δx, Δy), last action one-hot | (N-1) × 9 (含 presence flag) |
| `o_i^{global}` | c_t, time remaining ratio | 2 |
| `o_i^{capability}` | $\mathbf{cap}_i$ 4 维 | 4 |
| `o_i^{type}` | 自身 type one-hot | 2 |

---

## 6. Cross-references

- Ch3.7 观测函数定义
- `../design.md` D5（自己 type 仅 2 维）
- `01-agent-type-schema.md`（`one_hot` 用于 type 块）
- `02-capability-vector.md`（`to_array` 用于 cap 块）
- `04-timestep-record.md`（`o` 字段使用此 layout）
- Pkg-02 spec 04-six-block-observation.md（env 构造观测的实现）
