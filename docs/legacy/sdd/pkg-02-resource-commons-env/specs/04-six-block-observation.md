# Spec 04: 六块观测构造 — Ch3.7 + Self-Info

> 父文档：[`../proposal.md`](../proposal.md) §2.1 · [`../design.md`](../design.md) §3 D2

---

## 1. Purpose

实现 Ch3.7 六块观测的构造，严格遵循 Pkg-01 `ObservationLayout` 维度常量。**Self-Info 原则**（Ch3.7.4 + Ch4.2.2）：type 块仅含自己 2 维 one-hot，他人类型由 BeliefNet 推断；不在观测中暴露其他人的 type / cap。

---

## 2. Interface

### 2.1 文件路径

`hyper_mve/envs/resource_commons/observations.py`

### 2.2 公开 API

```python
import numpy as np
from hyper_mve.schemas import ObservationLayout, AgentType, one_hot as type_one_hot
from hyper_mve.envs.resource_commons.state import ResourceCommonsState


def build_observation(
    state: ResourceCommonsState,
    agent_id: int,
    L: int,
    T_max: int,
    K: int,
) -> np.ndarray:
    """为单个 agent 构造六块观测 (Ch3.7).
    
    Returns:
        obs: shape (ObservationLayout.total_dim(N, K),) float32 flattened
    """
    N = state.agent_positions.shape[0]
    blocks = []
    
    # Block 1: self (4 维)
    blocks.append(_self_block(state, agent_id, L, T_max))
    
    # Block 2: resource - FOV 内 (K × 3 维, padded)
    blocks.append(_resource_block(state, agent_id, K))
    
    # Block 3: neighbor - FOV 内 (N-1 × 9 维, padded with presence_flag)
    blocks.append(_neighbor_block(state, agent_id, N))
    
    # Block 4: global (2 维)
    blocks.append(_global_block(state, T_max))
    
    # Block 5: capability (4 维, self only)
    blocks.append(state.agent_caps[agent_id].to_array(dtype=np.float32))
    
    # Block 6: type (2 维, self only, Self-Info)
    own_type = AgentType.from_index(int(state.agent_types[agent_id]))
    blocks.append(type_one_hot(own_type, dtype=np.float32))
    
    obs = np.concatenate(blocks, dtype=np.float32)
    expected_dim = ObservationLayout.total_dim(N, K)
    assert obs.shape == (expected_dim,), f"obs dim {obs.shape} != {expected_dim}"
    return obs


def build_joint_observation(
    state: ResourceCommonsState,
    L: int,
    T_max: int,
    K: int,
) -> np.ndarray:
    """为全部 N agent 构造联合观测.
    
    Returns:
        obs: (N, obs_dim) float32
    """
    N = state.agent_positions.shape[0]
    obs_dim = ObservationLayout.total_dim(N, K)
    joint_obs = np.zeros((N, obs_dim), dtype=np.float32)
    for i in range(N):
        joint_obs[i] = build_observation(state, i, L, T_max, K)
    return joint_obs


# --- 私有辅助 ---

def _self_block(state, agent_id, L, T_max):
    """(4 维): x_norm, y_norm, cum_harvest_norm, steps_since_harvest_norm."""
    pos = state.agent_positions[agent_id]
    cap_zeta = state.agent_caps[agent_id].zeta
    return np.array([
        pos[0] / max(L - 1, 1),
        pos[1] / max(L - 1, 1),
        state.cumulative_harvests[agent_id] / max(cap_zeta, 1.0),
        state.steps_since_harvest[agent_id] / max(T_max, 1),
    ], dtype=np.float32)


def _resource_block(state, agent_id, K):
    """(K * 3 维): FOV 内资源, padded to K."""
    pos = state.agent_positions[agent_id]
    fov = state.agent_caps[agent_id].fov_int
    
    visible = []
    for k in range(K):
        rel = state.resource_positions[k] - pos
        chebyshev = np.max(np.abs(rel))
        if chebyshev <= fov:
            visible.append((
                float(rel[0]),
                float(rel[1]),
                float(state.resource_stocks[k] / 10.0),  # Q_max 归一化
            ))
    
    return _pad_resource(visible, K)


def _pad_resource(visible, K):
    """Padding to K entities, 不可见位填 0."""
    out = np.zeros((K, 3), dtype=np.float32)
    for i, (dx, dy, q) in enumerate(visible[:K]):
        out[i] = (dx, dy, q)
    return out.reshape(-1)


def _neighbor_block(state, agent_id, N):
    """((N-1) * 9 维): FOV 内其他 agent.
    
    每 entity: (rel_dx, rel_dy, last_action_onehot[6], presence_flag)
    """
    pos = state.agent_positions[agent_id]
    fov = state.agent_caps[agent_id].fov_int
    
    visible = []
    for j in range(N):
        if j == agent_id:
            continue
        rel = state.agent_positions[j] - pos
        chebyshev = np.max(np.abs(rel))
        if chebyshev <= fov:
            last_act_onehot = np.zeros(6, dtype=np.float32)
            last_act_onehot[state.last_actions[j]] = 1.0
            visible.append((
                float(rel[0]),
                float(rel[1]),
                last_act_onehot,
                True,
            ))
    
    return _pad_neighbor(visible, N)


def _pad_neighbor(visible, N):
    """Padding to N-1 slots."""
    out = np.zeros((N - 1, 9), dtype=np.float32)
    for i, (dx, dy, act_oh, presence) in enumerate(visible[:N-1]):
        out[i, 0] = dx
        out[i, 1] = dy
        out[i, 2:8] = act_oh
        out[i, 8] = float(presence)
    return out.reshape(-1)


def _global_block(state, T_max):
    """(2 维): c_t, episode 剩余比例."""
    time_remaining = (T_max - state.step_idx) / max(T_max, 1)
    return np.array([state.c_t, time_remaining], dtype=np.float32)
```

---

## 3. Implementation Notes

### 3.1 与 Pkg-01 ObservationLayout 的契约

| Pkg-01 常量 | 本 spec 使用 |
|------------|-------------|
| `ObservationLayout.SELF_DIM = 4` | `_self_block` 返回 (4,) |
| `ObservationLayout.RESOURCE_PER_ITEM = 3` | resource entity dim |
| `ObservationLayout.NEIGHBOR_PER_ITEM = 9` | neighbor entity dim |
| `ObservationLayout.GLOBAL_DIM = 2` | `_global_block` 返回 (2,) |
| `ObservationLayout.CAPABILITY_DIM = 4` | `cap.to_array()` 返回 (4,) |
| `ObservationLayout.TYPE_DIM = 2` | `type_one_hot()` 返回 (2,) |

实现完毕后 `obs.shape == (ObservationLayout.total_dim(N, K),)` 是 invariant，单测强制。

### 3.2 Self-Info 严格性（Ch3.7.4 + Ch4.2.2）

- **type 块仅 own 2 维**：不含他人 type one-hot
- **capability 块仅 own 4 维**：不含他人 cap
- **neighbor 块的 last_action_onehot 只有动作语义**：不暴露他人 type 或 cap
- 他人 type 由 BeliefNet head_opp 推断（v4 Oracle 监督训练）

违反 Self-Info 会破坏论文断言基础：
- 断言 B（信念专用容量）：如果他人 type 已在 obs 暴露 → BeliefNet 无需推断 → 实验无意义
- Harsanyi 对应（Ch4.1.4）：私人信念路径失效

### 3.3 9 维 neighbor entity（v4 实施 augment）

Ch3.7 文档表格写 `(rel_dx, rel_dy, last_action_onehot)` = 2 + 6 = 8 维。  
本 spec **加 1 维 `presence_flag`**（共 9 维）：
- FOV 外的 agent → 全 0 填充，presence_flag=0
- FOV 内的 agent → 真实数据，presence_flag=1
- 区分"邻居采 NOOP 动作"vs"邻居不可见"（last_action_onehot 都是 [1,0,0,0,0,0]）

**论文标注**：在论文 Ch3.7 末段加 footnote "实现中 neighbor entity 加入 1 维 presence_flag 以区分 padding 与真实 NOOP 动作，总维度 9。"

### 3.4 性能优化方向

如果 FOV 过滤成为瓶颈（Hard config N=8 × K=40 × 300 step）：
1. 每 step 预计算 agent-resource Chebyshev 距离矩阵（O(N*K) 一次）
2. 用 numpy boolean mask 替代 Python loop
3. 不引入 scipy KDTree（与 D2 决策一致）

当前实现已经 vectorized，预期性能符合目标。

---

## 4. Edge Cases

| 场景 | 行为 |
|------|------|
| Agent 在边界（pos.x = 0） | self_block 的 x_norm = 0（正常） |
| FOV 内 0 个资源 | resource_block 全 0 (K × 3) |
| FOV 内 > K 个资源（理论不可能，但保险） | 截断到前 K |
| FOV 内 0 个其他 agent | neighbor_block 全 0 (N-1) × 9 |
| 全部 agent 在同一格 | neighbor_block 含 N-1 个 entity，presence=1 |
| 多次 reset 后 obs 维度不变 | invariant (preset 决定 N, K) |
| Stage 1 oracle 模式（c hidden） | global_block 的 c_t 值替换为常量 0.5（评估时由 evaluator 控制） |

---

## 5. Acceptance Criteria

### 5.1 单元测试（`tests/envs/test_observations.py`）

```python
import numpy as np
import pytest
from hyper_mve.envs.resource_commons.observations import (
    build_observation, build_joint_observation,
)
from hyper_mve.schemas import ObservationLayout, AgentType


def test_obs_total_dim_matches_layout():
    """obs.shape 与 Pkg-01 ObservationLayout 自洽."""
    state = _make_medium_state()  # N=4, K=20
    obs = build_observation(state, agent_id=0, L=16, T_max=200, K=20)
    expected = ObservationLayout.total_dim(N=4, K=20)
    assert obs.shape == (expected,)
    assert expected == 99  # Medium config


def test_obs_dtype_float32():
    state = _make_medium_state()
    obs = build_observation(state, agent_id=0, L=16, T_max=200, K=20)
    assert obs.dtype == np.float32


def test_self_info_type_block_own_only():
    """v4 关键: type 块仅 own 2 维, 不含他人."""
    state = _make_state(
        N=4, K=20,
        agent_types=np.array([0, 0, 1, 1], dtype=np.int8),
    )
    # Agent 0 是 ALPHA
    obs = build_observation(state, agent_id=0, L=16, T_max=200, K=20)
    start, end = ObservationLayout.block_offset("type", N=4, K=20)
    type_block = obs[start:end]
    assert type_block.shape == (2,)
    assert np.allclose(type_block, [1.0, 0.0])  # ALPHA one-hot
    
    # Agent 2 是 BETA
    obs2 = build_observation(state, agent_id=2, L=16, T_max=200, K=20)
    type_block2 = obs2[start:end]
    assert np.allclose(type_block2, [0.0, 1.0])  # BETA one-hot


def test_self_info_no_others_type_leak():
    """v4 关键: 改变他人 type 不应改变本 agent 观测."""
    state1 = _make_state(N=4, K=20, agent_types=np.array([0,0,1,1], dtype=np.int8))
    state2 = _make_state(N=4, K=20, agent_types=np.array([0,1,0,1], dtype=np.int8))
    
    # Agent 0 在两 state 中都是 ALPHA, 其观测应相同 (除非他人位置/动作变化)
    state2.agent_positions = state1.agent_positions.copy()
    state2.last_actions = state1.last_actions.copy()
    
    obs1 = build_observation(state1, agent_id=0, L=16, T_max=200, K=20)
    obs2 = build_observation(state2, agent_id=0, L=16, T_max=200, K=20)
    # 唯一差异应在 BeliefNet 推断, 不在 obs 中
    assert np.allclose(obs1, obs2)


def test_capability_block_own_only():
    """capability 块仅 own 4 维."""
    state = _make_medium_state()
    obs = build_observation(state, agent_id=0, L=16, T_max=200, K=20)
    start, end = ObservationLayout.block_offset("capability", N=4, K=20)
    cap_block = obs[start:end]
    assert cap_block.shape == (4,)
    # 与 state.agent_caps[0] 一致
    assert np.allclose(cap_block, state.agent_caps[0].to_array())


def test_resource_block_fov_filter():
    """FOV 外资源被过滤为 0."""
    state = _make_state(
        N=1, K=2,
        agent_positions=np.array([[0,0]], dtype=np.int32),
        resource_positions=np.array([[0,1], [10,10]], dtype=np.int32),  # 一近一远
        resource_stocks=np.array([5.0, 5.0], dtype=np.float32),
        agent_caps=(CapabilityVector(eta=1.0, phi_fov=2.0, nu=1.0, zeta=20.0),),
    )
    obs = build_observation(state, agent_id=0, L=16, T_max=200, K=2)
    start, end = ObservationLayout.block_offset("resource", N=1, K=2)
    res_block = obs[start:end].reshape(2, 3)
    # 第一个资源 (1, 1) 内, 应有数据
    assert res_block[0, 0] != 0 or res_block[0, 1] != 0 or res_block[0, 2] != 0
    # 第二个资源 (10, 10) 外, 应全 0
    assert np.allclose(res_block[1], 0.0)


def test_neighbor_block_presence_flag():
    """Presence flag: 可见邻居 = 1, 不可见 = 0."""
    state = _make_state(
        N=3, K=20,
        agent_positions=np.array([[0,0], [1,1], [10,10]], dtype=np.int32),
        agent_caps=tuple(CapabilityVector(eta=1.0, phi_fov=2.0, nu=1.0, zeta=20.0) for _ in range(3)),
    )
    obs = build_observation(state, agent_id=0, L=16, T_max=200, K=20)
    start, end = ObservationLayout.block_offset("neighbor", N=3, K=20)
    nb_block = obs[start:end].reshape(2, 9)  # N-1 = 2
    # Neighbor 1 (1,1) 在 FOV 内 → presence = 1
    assert nb_block[0, 8] == 1.0
    # Neighbor 2 (10,10) FOV 外 → presence = 0
    assert nb_block[1, 8] == 0.0


def test_joint_observation_shape():
    state = _make_medium_state()
    joint_obs = build_joint_observation(state, L=16, T_max=200, K=20)
    assert joint_obs.shape == (4, 99)


def test_global_block_c_t():
    state = _make_state(N=4, K=20, c_t=0.7)
    obs = build_observation(state, agent_id=0, L=16, T_max=200, K=20)
    start, end = ObservationLayout.block_offset("global", N=4, K=20)
    global_block = obs[start:end]
    assert global_block[0] == pytest.approx(0.7)
```

### 5.2 集成测试

- env.reset 后 `obs.shape == env.observation_space.shape`（gym 标准）
- 1000 random step 后无 NaN/Inf

### 5.3 与 Ch3.7 文档对照

| Block | Ch3.7 维度 | 本 spec 实现 | ✓/✗ |
|-------|------------|--------------|------|
| `self` | 4 | 4 | ✓ |
| `resource` | K×3 | K×3 | ✓ |
| `neighbor` | (N-1)×8 文档 | **(N-1)×9 实现** (+presence_flag) | ⚠️ augment |
| `global` | 2 | 2 | ✓ |
| `capability` | 4 | 4 | ✓ |
| `type` | 2 (own only) | 2 (own only) | ✓ |

**neighbor 9 维 augment 在论文 Ch3.7 末段加 footnote 说明**。

---

## 6. Cross-references

- Ch3.7 观测函数
- Ch3.7.4 Self-Info 原则
- Ch4.2.2 type_emb 自身可见（与 Self-Info 一致）
- Ch4.1.4 Harsanyi 对应（私人信念路径前提）
- Pkg-01 `03-observation-layout.md`（ObservationLayout 维度常量）
- Pkg-01 `02-capability-vector.md`（cap.to_array 接口）
- Pkg-01 `01-agent-type-schema.md`（one_hot 工具）
- `01-env-formal-tuple.md`（state 字段读取）
- `08-gym-api.md`（observation_space 定义）
- `../design.md` D2 (FOV 过滤)
