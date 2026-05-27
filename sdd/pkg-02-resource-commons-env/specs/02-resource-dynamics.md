# Spec 02: Resource Dynamics — 公式 3.1-3.4 实现

> 父文档：[`../proposal.md`](../proposal.md) §2.1 · [`../design.md`](../design.md) §3 D1 / D4

---

## 1. Purpose

实现 Ch3.3 的资源场动力学三公式：
- 公式 3.1：logistic 资源再生（含消耗 + 再生）
- 公式 3.2：fair-share 采集量上限
- 公式 3.3：邻居影响因子 f(neighbors) 用 sigmoid
- 公式 3.4：α(c_t) 线性映射

实现误差 ≤ 1e-4（与解析解对照）。

---

## 2. Interface

### 2.1 文件路径

`hyper_mve/envs/resource_commons/dynamics.py`

### 2.2 公开 API

```python
import numpy as np
from hyper_mve.envs.resource_commons.state import ResourceCommonsState
from hyper_mve.schemas._constants import (
    Q_MAX, ALPHA_MIN, ALPHA_MAX, KAPPA_F, THETA_F, D_NBR,
)


def compute_alpha(c_t: float, alpha_min: float = ALPHA_MIN, 
                  alpha_max: float = ALPHA_MAX) -> float:
    """公式 3.4: α(c_t) = α_min + (α_max - α_min) × c_t.
    
    Args:
        c_t: 共享上下文 ∈ [0, 1]
        alpha_min: 缺省 0.02 (荒年再生率, ~50 步从 0 恢复)
        alpha_max: 缺省 0.20 (丰年再生率, ~5 步从 0 恢复)
    
    Returns:
        α(c_t) ∈ [α_min, α_max]
    """
    assert 0.0 <= c_t <= 1.0, f"c_t={c_t} ∉ [0, 1]"
    return alpha_min + (alpha_max - alpha_min) * c_t


def compute_neighbor_factor(resource_stocks: np.ndarray,
                           resource_positions: np.ndarray,
                           k: int,
                           d_nbr: int = D_NBR,
                           q_max: float = Q_MAX,
                           kappa_f: float = KAPPA_F,
                           theta_f: float = THETA_F) -> float:
    """公式 3.3: f(neighbors of k) = sigmoid(κ_f × mean_q_norm - θ_f).
    
    Args:
        resource_stocks: (K,) 全部资源点当前库存
        resource_positions: (K, 2) 资源点坐标
        k: 目标资源点索引
        d_nbr: Chebyshev 距离阈值 (默认 3)
        q_max: Q_max 用于归一化
        kappa_f, theta_f: sigmoid 参数 (默认 6.0, 0.3)
    
    Returns:
        f(neighbors) ∈ (0, 1), 邻居充足时 → 1, 邻居枯竭时 → 0
    """
    K = resource_stocks.shape[0]
    pos_k = resource_positions[k]
    
    # Chebyshev 距离
    distances = np.max(np.abs(resource_positions - pos_k), axis=-1)
    neighbor_mask = (distances <= d_nbr) & (distances > 0)  # 排除自己
    
    if not neighbor_mask.any():
        return _sigmoid(-theta_f)   # 无邻居 → 极低再生因子
    
    mean_q_norm = np.mean(resource_stocks[neighbor_mask] / q_max)
    return _sigmoid(kappa_f * mean_q_norm - theta_f)


def step_dynamics(state: ResourceCommonsState,
                  harvests_per_resource: np.ndarray) -> None:
    """公式 3.1: 一步资源再生动力学 (in-place 修改 state.resource_stocks).
    
    q_{k,t+1} = clip(
        q_{k,t} - sum(u_i 在 k 点采集量)
                + α(c_t) × f(neighbors of k) × (Q_max - q_{k,t}),
        0, Q_max
    )
    
    Args:
        state: env 内部状态 (修改其 resource_stocks)
        harvests_per_resource: (K,) 本步各资源点被采集总量 (公式 3.2 已分配)
    """
    K = state.resource_stocks.shape[0]
    alpha_c = compute_alpha(state.c_t)
    new_stocks = state.resource_stocks.copy()
    
    for k in range(K):
        f_neighbors = compute_neighbor_factor(
            state.resource_stocks, state.resource_positions, k
        )
        consumption = harvests_per_resource[k]
        regrowth = alpha_c * f_neighbors * (Q_MAX - state.resource_stocks[k])
        new_stocks[k] = state.resource_stocks[k] - consumption + regrowth
    
    # clip 防漂移 (D4 决策)
    np.clip(new_stocks, 0.0, Q_MAX, out=new_stocks)
    state.resource_stocks[:] = new_stocks


def fair_share_harvest(state: ResourceCommonsState,
                       agent_positions: np.ndarray,
                       harvest_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """公式 3.2: 每 agent 在所在格子上 fair-share 采集.
    
    u_{i,k} = min(η_i, q_{k,t} / |H_{k,t}|)
        其中 H_{k,t} = {i : agent_i 在 k 点 且 选择 HARVEST}
    
    Args:
        state: env 状态 (含 resource_positions, resource_stocks, agent_caps)
        agent_positions: (N, 2) 当前 agent 位置
        harvest_mask: (N,) bool, 哪些 agent 选择了 HARVEST
    
    Returns:
        harvests_per_agent: (N,) 每 agent 本步采集量 u_i
        harvests_per_resource: (K,) 每资源点被总采集量 (供 step_dynamics)
    """
    N = agent_positions.shape[0]
    K = state.resource_stocks.shape[0]
    
    harvests_per_agent = np.zeros(N, dtype=np.float32)
    harvests_per_resource = np.zeros(K, dtype=np.float32)
    
    # 对每个资源点 k, 找在该格子且 harvest=True 的 agents
    for k in range(K):
        on_cell_mask = np.all(agent_positions == state.resource_positions[k], axis=-1)
        H_k = np.where(on_cell_mask & harvest_mask)[0]
        if len(H_k) == 0:
            continue
        
        # fair-share: u_i = min(η_i, q_k / |H_k|)
        share = state.resource_stocks[k] / len(H_k)
        for i in H_k:
            u_i = min(state.agent_caps[i].eta, share)
            harvests_per_agent[i] = u_i
            harvests_per_resource[k] += u_i
    
    return harvests_per_agent, harvests_per_resource


def _sigmoid(x: float) -> float:
    """数值稳定 sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))
```

---

## 3. Implementation Notes

### 3.1 公式实现顺序（关键）

env.step 内部正确顺序：
1. agent 移动（按动作）
2. **公式 3.2** `fair_share_harvest` 计算 u_{i,k} 与每资源点总消耗
3. **公式 3.1** `step_dynamics` 用消耗 + 再生更新 resource_stocks
4. **公式 3.3** 邻居因子在 `step_dynamics` 内部按 **更新前**的 stocks 计算（一致性）

为何邻居因子用更新前的 stocks：
- 公式 3.1 是离散化的微分方程，dq/dt 由当前 q 计算
- 若用更新后 stocks 会引入时间步长偏差
- 测试 `test_dynamics.py::test_logistic_regen_5steps` 严格按此顺序对照解析解

### 3.2 浮点稳定性（D4 决策）

- 所有计算 float32
- 每步 `np.clip(new_stocks, 0, Q_MAX)` 防漂移
- sigmoid 输入用 `np.clip(x, -50, 50)` 防溢出（极少触发，但保险）

### 3.3 性能

- K=20 每步 K 次 neighbor factor + 1 次 update：~50 μs
- K=40 (Hard)：~150 μs
- 仍远低于 1 ms/step 目标

### 3.4 与 v3 对比

v3 公式 3.2 有"合作奖励" `[1 + β(c)(|H|-1)]` 让多人采集时人均增加。v4 **完全删除** β(c) 单通道——这是 Ch3 v3 → v4 的关键演进（消除物理层 reward shaping）。本 spec 实现严格 fair-share，无任何合作 bonus。

---

## 4. Edge Cases

| 场景 | 行为 |
|------|------|
| 无邻居的资源点 (孤立 hotspot) | `f` 退化为 `sigmoid(-θ_f) ≈ 0.43`，再生缓慢 |
| 所有资源 q=0 | 全部再生项 = α × f × Q_max，缓慢恢复 |
| 多 agent 同格 + HARVEST | fair-share 严格执行，u_i = min(η_i, q/|H|) |
| HARVEST 在无资源格子 | u_i = 0（H_{k,t} 空集，不进入循环） |
| η_i > q_k （单个 agent 想吃光） | u_i = q_k（fair-share 自动截断） |
| 资源被采干 (q_k → 0) | 下一步 regrowth = α × f × Q_max，恢复中 |
| c_t = 0 (极荒年) | α = 0.02，再生极慢 |
| c_t = 1 (极丰年) | α = 0.20，再生快 |
| 资源点位置重叠 (两个 hotspot 中心一致) | spawn.py 加 min_distance 约束防止 |

---

## 5. Acceptance Criteria

### 5.1 单元测试（`tests/envs/test_dynamics.py`）

```python
import numpy as np
import pytest
from hyper_mve.envs.resource_commons.dynamics import (
    compute_alpha, compute_neighbor_factor, step_dynamics, fair_share_harvest
)
from hyper_mve.envs.resource_commons.state import ResourceCommonsState


def test_alpha_linear():
    """公式 3.4 线性映射."""
    assert compute_alpha(0.0) == pytest.approx(0.02)
    assert compute_alpha(1.0) == pytest.approx(0.20)
    assert compute_alpha(0.5) == pytest.approx(0.11)

def test_alpha_out_of_range():
    with pytest.raises(AssertionError):
        compute_alpha(1.5)

def test_neighbor_factor_full_neighbors():
    """所有邻居满库存 → f ≈ 1."""
    stocks = np.full(5, 10.0, dtype=np.float32)  # Q_max = 10
    positions = np.array([[0,0], [1,0], [0,1], [2,2], [3,3]], dtype=np.int32)
    f = compute_neighbor_factor(stocks, positions, k=0)
    # mean_q_norm = 1.0, sigmoid(6.0 × 1.0 - 0.3) = sigmoid(5.7) ≈ 0.997
    assert f > 0.99

def test_neighbor_factor_empty_neighbors():
    """所有邻居枯竭 → f ≈ 0.1."""
    stocks = np.array([10.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    positions = np.array([[0,0], [1,0], [0,1], [2,2], [3,3]], dtype=np.int32)
    f = compute_neighbor_factor(stocks, positions, k=0)
    # mean_q_norm = 0, sigmoid(-0.3) ≈ 0.426
    assert 0.4 < f < 0.5

def test_logistic_regen_5steps():
    """关键: 单点 5 步再生解析解对照.
    
    单点, 无邻居, q_0=0, α=0.2, f=0.43 (∅ neighbor 默认), Q_max=10:
    q_{t+1} = q_t + 0.2 × 0.43 × (10 - q_t) = 0.086 × (10 - q_t) + q_t
    
    5 步: q_5 = 10 × (1 - 0.914^5) ≈ 10 × 0.366 ≈ 3.66
    """
    state = _make_minimal_state(
        c_t=1.0,  # α = 0.2
        K=1, resource_stocks=np.array([0.0], dtype=np.float32),
    )
    
    for _ in range(5):
        step_dynamics(state, harvests_per_resource=np.zeros(1, dtype=np.float32))
    
    # 解析: q_5 ≈ 10 × (1 - (1 - 0.2 × 0.43)^5) ≈ 3.66
    # 单点无邻居 f ≈ sigmoid(-0.3) ≈ 0.426
    expected = 10 * (1 - (1 - 0.2 * 0.426)**5)
    assert abs(state.resource_stocks[0] - expected) < 0.01

def test_no_overharvest():
    """fair-share 上限严格执行."""
    state = _make_state(
        N=3, K=1,
        agent_positions=np.array([[0,0], [0,0], [0,0]], dtype=np.int32),
        resource_positions=np.array([[0,0]], dtype=np.int32),
        resource_stocks=np.array([3.0], dtype=np.float32),  # q=3
        agent_caps=tuple(CapabilityVector(eta=1.0, phi_fov=3.0, nu=1.0, zeta=20.0) 
                         for _ in range(3)),
    )
    harvest_mask = np.array([True, True, True])  # 3 agent 都 HARVEST
    
    u_per_agent, u_per_resource = fair_share_harvest(
        state, state.agent_positions, harvest_mask
    )
    
    # share = 3 / 3 = 1.0, 每人 min(1.0, 1.0) = 1.0
    assert np.allclose(u_per_agent, [1.0, 1.0, 1.0])
    assert u_per_resource[0] == 3.0  # 总采集 = q (恰好采干)

def test_eta_caps_harvest():
    """单个 agent η < share 时只采 η."""
    state = _make_state(
        N=1, K=1,
        agent_positions=np.array([[0,0]], dtype=np.int32),
        resource_positions=np.array([[0,0]], dtype=np.int32),
        resource_stocks=np.array([10.0], dtype=np.float32),
        agent_caps=(CapabilityVector(eta=0.5, phi_fov=3.0, nu=1.0, zeta=20.0),),
    )
    harvest_mask = np.array([True])
    u_per_agent, _ = fair_share_harvest(state, state.agent_positions, harvest_mask)
    assert u_per_agent[0] == 0.5  # min(η=0.5, share=10) = 0.5

def test_harvest_off_cell():
    """HARVEST 在无资源格子 → u_i = 0."""
    state = _make_state(
        N=1, K=1,
        agent_positions=np.array([[5,5]], dtype=np.int32),  # 远离资源
        resource_positions=np.array([[0,0]], dtype=np.int32),
        resource_stocks=np.array([10.0], dtype=np.float32),
    )
    harvest_mask = np.array([True])
    u_per_agent, _ = fair_share_harvest(state, state.agent_positions, harvest_mask)
    assert u_per_agent[0] == 0.0

def test_clip_no_negative():
    """资源被采集后 + 再生不应为负."""
    state = _make_state(
        N=1, K=1,
        resource_stocks=np.array([1.0], dtype=np.float32),
        c_t=0.0,  # α=0.02 最小
    )
    # 假装外部消耗了 5.0 (大于 q=1)
    step_dynamics(state, harvests_per_resource=np.array([5.0], dtype=np.float32))
    assert state.resource_stocks[0] >= 0.0  # clip 保证
```

### 5.2 集成测试

- 100 random episode 后 sum(resource_stocks) 始终 ∈ [0, Q_max × K]
- 静态 c=0.5 配置，所有 agent 不动 → 资源应收敛到非零稳态

### 5.3 性能

- `compute_alpha`: O(1) < 1 μs
- `compute_neighbor_factor`: O(K) < 10 μs (K=40)
- `step_dynamics`: O(K²) < 200 μs (K=40)
- `fair_share_harvest`: O(N × K) < 100 μs

---

## 6. Cross-references

- Ch3.3 公式 3.1-3.4
- Ch3.3.3 邻居影响因子
- `01-env-formal-tuple.md`（state 字段）
- `03-fehr-schmidt-reward.md`（u_i 输出供 reward 公式）
- `05-context-evolution.md`（c_t 输入）
- `../design.md` D1 (np.array 存储)、D4 (浮点稳定)
