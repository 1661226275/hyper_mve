# Spec 03: Fehr-Schmidt Reward — 公式 3.5-3.10 + Ch4.1.1 偏导表

> 父文档：[`../proposal.md`](../proposal.md) §2.1 · [`../design.md`](../design.md) §3 D3

---

## 1. Purpose

实现 Ch3.5 类型机制与公式 3.5-3.10 的 reward 计算。**Table 3.5.4 四场景 + Ch4.1.1 偏导四象限 100% 数值匹配是断言 A 的物理前提**——本 spec 的单测是断言 A 实验启动的 hard gate。

---

## 2. Interface

### 2.1 文件路径

`hyper_mve/envs/resource_commons/rewards.py`

### 2.2 公开 API

```python
import numpy as np
from hyper_mve.schemas import AgentType
from hyper_mve.schemas._constants import (
    KAPPA, LAMBDA_DISADV, LAMBDA_ADV, EPSILON_MOVE,
)


def compute_phi(c_t: float, kappa: float = KAPPA) -> float:
    """公式 3.7: φ(c) = κ(1 - 2c).
    
    Args:
        c_t: ∈ [0, 1]
        kappa: 默认 0.5
    
    Returns:
        φ(c_t) ∈ [-κ, +κ] = [-0.5, +0.5]
            c=0 (荒年): φ=+0.5 (放大 Fehr-Schmidt)
            c=0.5 (中性): φ=0 (β 退化为自利)
            c=1 (丰年): φ=-0.5 (反转 Fehr-Schmidt)
    """
    return kappa * (1.0 - 2.0 * c_t)


def compute_psi(delta: float, 
                lambda_disadv: float = LAMBDA_DISADV,
                lambda_adv: float = LAMBDA_ADV) -> float:
    """公式 3.8: ψ(Δ) = -[λ_disadv × max(0, -Δ) + λ_adv × max(0, Δ)].
    
    Args:
        delta: 瞬时不公平差异 Δ_i
        lambda_disadv: 自己劣势厌恶 (默认 2.0, 强)
        lambda_adv: 自己优势厌恶 (默认 0.6, 弱)
    
    Returns:
        ψ(Δ) ≤ 0, 不公平时为负
            Δ > 0 (自己优势): ψ = -λ_adv × Δ
            Δ < 0 (自己劣势): ψ = -λ_disadv × |Δ|  (更负)
            Δ = 0: ψ = 0
    """
    return -(lambda_disadv * max(0.0, -delta) + lambda_adv * max(0.0, delta))


def compute_delta(harvests: np.ndarray) -> np.ndarray:
    """公式 3.9: Δ_i = u_i - mean_{j≠i} u_j (向量化, 全 agent 计算).
    
    Args:
        harvests: (N,) 本步各 agent 采集量 u_i
    
    Returns:
        deltas: (N,) 瞬时不公平差异
    """
    N = harvests.shape[0]
    total = harvests.sum()
    # mean_{j≠i} u_j = (total - u_i) / (N - 1)
    mean_others = (total - harvests) / max(N - 1, 1)
    return harvests - mean_others


def compute_rewards(
    harvests: np.ndarray,     # (N,) u_i
    moved_mask: np.ndarray,    # (N,) bool, 本步是否移动
    agent_types: np.ndarray,   # (N,) int8 AgentType.value
    c_t: float,
) -> tuple[np.ndarray, np.ndarray]:
    """公式 3.5 / 3.6 / 3.10 (统一形式): per-agent type-aware reward.
    
    R_i = u_i - ε × 𝟙[moved]  +  𝟙[τ_i=β] × φ(c) × ψ(Δ_i)
    
    Args:
        harvests: (N,) u_i
        moved_mask: (N,) bool
        agent_types: (N,) int8
        c_t: ∈ [0, 1]
    
    Returns:
        rewards: (N,) per-agent reward (type-aware)
        deltas: (N,) 瞬时不公平差异 (写入 buffer)
    """
    N = harvests.shape[0]
    deltas = compute_delta(harvests)
    phi_c = compute_phi(c_t)
    
    # 物理层: u_i - ε × 𝟙[moved]
    physical = harvests - EPSILON_MOVE * moved_mask.astype(np.float32)
    
    # 偏好层 (仅 type β agent 有 φψ 项):
    is_beta = (agent_types == AgentType.BETA.value)   # (N,)
    psi_per_agent = np.array([compute_psi(d) for d in deltas], dtype=np.float32)
    preference = phi_c * psi_per_agent * is_beta.astype(np.float32)
    
    rewards = physical + preference
    return rewards, deltas
```

### 2.3 关键公式 - 完整 Table 3.5.4

```
type β 在 4 个 (c, Δ) 场景的 φψ 项数值表 (Ch3.5.4):

| 场景 | c   | φ(c) | Δ  | ψ(Δ) | φ×ψ |
|------|-----|------|-----|------|-----|
| 荒年+优势 | 0.0 | +0.5 | +1 | -0.6 | -0.3 |
| 荒年+劣势 | 0.0 | +0.5 | -1 | -2.0 | -1.0 |
| 丰年+优势 | 1.0 | -0.5 | +1 | -0.6 | +0.3 |
| 丰年+劣势 | 1.0 | -0.5 | -1 | -2.0 | +1.0 |
```

### 2.4 Ch4.1.1 偏导表（断言 A 物理基础）

```
∂R^α/∂u_i = 1 (恒定)

∂R^β/∂u_i = 1 + φ(c) × ψ'(Δ_i)

ψ' 推导（来自 ψ(Δ) = -λ_disadv·max(0,-Δ) - λ_adv·max(0,Δ)）：
  - Δ > 0:  ψ'(Δ) = d/dΔ[-λ_adv · Δ]          = **-λ_adv = -0.6**
  - Δ < 0:  ψ'(Δ) = d/dΔ[-λ_disadv · (-Δ)]   = **+λ_disadv = +2.0**
  - Δ = 0:  ψ' 不可微（次梯度，左右极限 [-λ_adv, +λ_disadv]）

四象限（修正）:

| c | Δ | ψ'(Δ) | φ(c) | φ × ψ' | 1 + φ × ψ' = ∂R^β/∂u_i |
|---|---|-------|------|--------|------------------------|
| 0.0 | + | -0.6 | +0.5 | -0.3 | **0.7** |
| 0.0 | - | +2.0 | +0.5 | +1.0 | **2.0** |
| 1.0 | + | -0.6 | -0.5 | +0.3 | **1.3** |
| 1.0 | - | +2.0 | -0.5 | -1.0 | **0.0** |
```

**Δ 是 N-1 个其他 agent 的相对量，autograd 反向时需要小心**：
∂(u_i - mean_{j≠i} u_j) / ∂u_i = 1 - 0 = 1（自己对自己的 Δ 偏导是 1，对他人的 Δ 偏导是 -1/(N-1)）

详细推导：
- u_i 单位增加 1 → Δ_i 单位增加 1
- 若 Δ_i > 0 (自己优势): ψ 减少 λ_adv (因为 ψ 中 max(0, Δ) 项斜率为 +1，前面有负号 → ψ' = -λ_adv)
- 若 Δ_i < 0 (自己劣势): ψ 增加 λ_disadv (因为 ψ 中 max(0, -Δ) 项斜率为 -1，前面有负号，整体相对 Δ 斜率为 +λ_disadv)

---

## 3. Implementation Notes

### 3.1 vectorized vs loop for ψ

`compute_psi` 是 element-wise，可以向量化：

```python
def compute_psi_vec(deltas: np.ndarray) -> np.ndarray:
    """向量化 ψ."""
    disadv_term = LAMBDA_DISADV * np.maximum(0.0, -deltas)
    adv_term = LAMBDA_ADV * np.maximum(0.0, deltas)
    return -(disadv_term + adv_term)
```

但当前 list comprehension 实现也只需 ~5 μs（N=4），不是瓶颈。如未来 N 变大可切向量化。

### 3.2 为何 type β-only gate 用 mask 而非 if

- `is_beta` mask 与 preference 计算可全 N 向量化
- 与"按 type 分支 if/else"等价但更快
- 与 D3 决策一致（全 agent 计算 Δ + mask 选用）

### 3.3 ε_move 计算粒度

- ε_move=0.01 × bool(moved) 极小
- 用 moved_mask 数组形式，不分类型——所有 agent 都有这个移动成本
- 这是物理层（自利同质），与 type 无关

### 3.4 autograd 验证

为支持 Ch4.1.1 偏导表测试，需要 reward 函数支持 PyTorch autograd：

```python
import torch

def compute_rewards_torch(
    harvests: torch.Tensor,   # (N,) requires_grad
    moved_mask: torch.Tensor,
    agent_types: torch.Tensor,
    c_t: float,
) -> torch.Tensor:
    """PyTorch 版本, 用于 autograd 偏导验证 (测试用, 非 env 主路径)."""
    N = harvests.shape[0]
    total = harvests.sum()
    mean_others = (total - harvests) / max(N - 1, 1)
    deltas = harvests - mean_others
    
    phi_c = KAPPA * (1.0 - 2.0 * c_t)
    psi = -(LAMBDA_DISADV * torch.clamp(-deltas, min=0.0) 
            + LAMBDA_ADV * torch.clamp(deltas, min=0.0))
    
    is_beta = (agent_types == AgentType.BETA.value).float()
    physical = harvests - EPSILON_MOVE * moved_mask.float()
    preference = phi_c * psi * is_beta
    
    return physical + preference
```

测试用 `torch.autograd.grad(rewards[i], harvests, retain_graph=True)` 提取偏导。

---

## 4. Edge Cases

| 场景 | 行为 |
|------|------|
| N=1（单 agent） | mean_{j≠i} 退化为 0，Δ_i = u_i，ψ 计算正常 |
| 所有 agent 同 u_i | Δ_i = 0，ψ = 0（公平时无 Fehr-Schmidt 惩罚） |
| Δ = 0 严格相等 | ψ = 0（两边 max(0, ·) 都为 0） |
| c = 0.5 | φ = 0，β 退化为完全自利（reward = R^α） |
| 全 α 配置 | preference 项全 mask 掉，reward = u_i - ε×moved |
| 全 β 配置 | 所有 agent 计算 φψ，preference 全启用 |
| 极端 Δ（|Δ| >> 1） | ψ 线性增长，无截断（设计如此，与 Hughes 2018 一致） |

---

## 5. Acceptance Criteria

### 5.1 单元测试（`tests/envs/test_rewards.py`）

```python
import numpy as np
import torch
import pytest
from hyper_mve.envs.resource_commons.rewards import (
    compute_phi, compute_psi, compute_delta, compute_rewards,
    compute_rewards_torch,
)
from hyper_mve.schemas import AgentType


def test_phi_endpoints():
    """公式 3.7 端点."""
    assert compute_phi(0.0) == pytest.approx(0.5)
    assert compute_phi(1.0) == pytest.approx(-0.5)
    assert compute_phi(0.5) == pytest.approx(0.0)


def test_psi_quadrants():
    """公式 3.8 四象限."""
    assert compute_psi(0.0) == 0.0
    assert compute_psi(1.0) == pytest.approx(-0.6)   # 优势 (Δ > 0): -λ_adv
    assert compute_psi(-1.0) == pytest.approx(-2.0)  # 劣势 (Δ < 0): -λ_disadv
    assert compute_psi(2.0) == pytest.approx(-1.2)
    assert compute_psi(-0.5) == pytest.approx(-1.0)


def test_table_3_5_4_lean_advantage():
    """Ch3.5.4 场景 1: 荒年 + 自己优势."""
    phi = compute_phi(0.0)
    psi = compute_psi(1.0)
    assert phi * psi == pytest.approx(-0.3, abs=1e-6)


def test_table_3_5_4_lean_disadvantage():
    """Ch3.5.4 场景 2 (关键): 荒年 + 自己劣势 → 极端追赶."""
    phi = compute_phi(0.0)
    psi = compute_psi(-1.0)
    assert phi * psi == pytest.approx(-1.0, abs=1e-6)


def test_table_3_5_4_abundance_advantage():
    """Ch3.5.4 场景 3: 丰年 + 自己优势."""
    phi = compute_phi(1.0)
    psi = compute_psi(1.0)
    assert phi * psi == pytest.approx(+0.3, abs=1e-6)


def test_table_3_5_4_abundance_disadvantage():
    """Ch3.5.4 场景 4 (关键): 丰年 + 自己劣势 → 偏好他人采集."""
    phi = compute_phi(1.0)
    psi = compute_psi(-1.0)
    assert phi * psi == pytest.approx(+1.0, abs=1e-6)


def test_delta_zero_when_equal():
    """所有 agent 平等 → Δ_i = 0."""
    harvests = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    deltas = compute_delta(harvests)
    assert np.allclose(deltas, 0.0)


def test_delta_unit_advantage():
    """单 agent 多采 1 → Δ_i = +1."""
    harvests = np.array([2.0, 1.0, 1.0, 1.0], dtype=np.float32)
    deltas = compute_delta(harvests)
    # u_0 = 2, mean_{j≠0} = (1+1+1)/3 = 1 → Δ_0 = 1
    assert deltas[0] == pytest.approx(1.0)
    # u_1 = 1, mean_{j≠1} = (2+1+1)/3 ≈ 1.333 → Δ_1 = -0.333
    assert deltas[1] == pytest.approx(-1/3)


def test_alpha_no_psi():
    """type α reward 不含 φψ 项."""
    harvests = np.array([2.0, 1.0], dtype=np.float32)
    moved = np.array([False, False])
    types = np.array([AgentType.ALPHA.value, AgentType.ALPHA.value], dtype=np.int8)
    rewards, deltas = compute_rewards(harvests, moved, types, c_t=0.0)
    # R^α = u_i, 无偏好项
    assert np.allclose(rewards, [2.0, 1.0])


def test_beta_includes_psi():
    """type β reward 含 φψ 项."""
    harvests = np.array([2.0, 1.0], dtype=np.float32)
    moved = np.array([False, False])
    types = np.array([AgentType.BETA.value, AgentType.BETA.value], dtype=np.int8)
    rewards, _ = compute_rewards(harvests, moved, types, c_t=0.0)
    # u_0=2, Δ_0=+1, φ(0)=+0.5, ψ(+1)=-0.6, 项=−0.3
    # R^β_0 = 2 + (-0.3) = 1.7
    assert rewards[0] == pytest.approx(1.7, abs=1e-5)
    # u_1=1, Δ_1=-1, φ(0)=+0.5, ψ(-1)=-2, 项=−1.0
    # R^β_1 = 1 + (-1.0) = 0.0
    assert rewards[1] == pytest.approx(0.0, abs=1e-5)


def test_mixed_types():
    """1α + 1β: α 无 φψ, β 有."""
    harvests = np.array([2.0, 1.0], dtype=np.float32)
    moved = np.array([False, False])
    types = np.array([AgentType.ALPHA.value, AgentType.BETA.value], dtype=np.int8)
    rewards, _ = compute_rewards(harvests, moved, types, c_t=0.0)
    assert rewards[0] == pytest.approx(2.0)  # α: 仅 u_0
    assert rewards[1] == pytest.approx(0.0, abs=1e-5)  # β: 0 + (φ·ψ)


def test_alpha_grad_constant_one():
    """断言 A 物理基础: ∂R^α/∂u_i = 1.0 (100 random)."""
    for _ in range(100):
        harvests = torch.rand(4, requires_grad=True)
        moved = torch.zeros(4, dtype=torch.bool)
        types = torch.full((4,), AgentType.ALPHA.value, dtype=torch.int8)
        c_t = np.random.uniform(0, 1)
        rewards = compute_rewards_torch(harvests, moved, types, c_t)
        for i in range(4):
            (grad_i,) = torch.autograd.grad(rewards[i], harvests, retain_graph=True)
            assert abs(grad_i[i].item() - 1.0) < 1e-5


def test_beta_grad_four_quadrants():
    """断言 A 关键: Ch4.1.1 偏导表四象限 100% 匹配."""
    # 场景 1: c=0, Δ>0 (荒年+优势)
    harvests = torch.tensor([2.0, 1.0, 1.0, 1.0], requires_grad=True)
    moved = torch.zeros(4, dtype=torch.bool)
    types = torch.full((4,), AgentType.BETA.value, dtype=torch.int8)
    rewards = compute_rewards_torch(harvests, moved, types, c_t=0.0)
    (grad,) = torch.autograd.grad(rewards[0], harvests)
    # 期望 ∂R^β/∂u_0 = 0.7
    # 推导: φ(0)=+0.5, ψ'(Δ>0)=-λ_adv=-0.6
    #       1 + 0.5 × (-0.6) = 1 - 0.3 = 0.7
    assert abs(grad[0].item() - 0.7) < 1e-3
    
    # 场景 2: c=0, Δ<0 (荒年+劣势)
    harvests = torch.tensor([1.0, 2.0, 2.0, 2.0], requires_grad=True)
    rewards = compute_rewards_torch(harvests, moved, types, c_t=0.0)
    (grad,) = torch.autograd.grad(rewards[0], harvests)
    # 期望 ∂R^β/∂u_0 = 2.0
    # 推导: φ(0)=+0.5, ψ'(Δ<0)=+λ_disadv=+2.0
    #       1 + 0.5 × (+2.0) = 1 + 1.0 = 2.0
    assert abs(grad[0].item() - 2.0) < 1e-3
    
    # 场景 3: c=1, Δ>0 (丰年+优势)
    harvests = torch.tensor([2.0, 1.0, 1.0, 1.0], requires_grad=True)
    rewards = compute_rewards_torch(harvests, moved, types, c_t=1.0)
    (grad,) = torch.autograd.grad(rewards[0], harvests)
    # 期望 ∂R^β/∂u_0 = 1.3
    # 推导: φ(1)=-0.5, ψ'(Δ>0)=-0.6
    #       1 + (-0.5) × (-0.6) = 1 + 0.3 = 1.3
    assert abs(grad[0].item() - 1.3) < 1e-3
    
    # 场景 4: c=1, Δ<0 (丰年+劣势)
    harvests = torch.tensor([1.0, 2.0, 2.0, 2.0], requires_grad=True)
    rewards = compute_rewards_torch(harvests, moved, types, c_t=1.0)
    (grad,) = torch.autograd.grad(rewards[0], harvests)
    # 期望 ∂R^β/∂u_0 = 0.0
    # 推导: φ(1)=-0.5, ψ'(Δ<0)=+2.0
    #       1 + (-0.5) × (+2.0) = 1 - 1.0 = 0.0
    assert abs(grad[0].item() - 0.0) < 1e-3


def test_move_penalty():
    """ε × 𝟙[moved] 移动成本."""
    harvests = np.array([1.0], dtype=np.float32)
    moved = np.array([True])
    types = np.array([AgentType.ALPHA.value], dtype=np.int8)
    rewards, _ = compute_rewards(harvests, moved, types, c_t=0.5)
    assert rewards[0] == pytest.approx(0.99)  # 1.0 - 0.01
```

### 5.2 集成测试

- 100 random episode: 检查 reward 范围 ∈ [-3, 3]
- 全 α 配置 vs 全 β 配置：β agent 在 c=0 时社会福利分布显著差异（Proposition 3.1 实证）

### 5.3 数值验证（论文级别）

| Ch3.5.4 场景 | 期望 φψ | 单测验证 |
|--------------|---------|---------|
| 荒年优势 | -0.3 | `test_table_3_5_4_lean_advantage` ✓ |
| 荒年劣势 | -1.0 | `test_table_3_5_4_lean_disadvantage` ✓ |
| 丰年优势 | +0.3 | `test_table_3_5_4_abundance_advantage` ✓ |
| 丰年劣势 | +1.0 | `test_table_3_5_4_abundance_disadvantage` ✓ |

| Ch4.1.1 偏导 | 期望 ∂R^β/∂u_i | 单测验证 |
|--------------|----------------|---------|
| 荒年优势 | 0.7 | `test_beta_grad_four_quadrants` ✓ |
| 荒年劣势 | 2.0 | 同上 |
| 丰年优势 | 1.3 | 同上 |
| 丰年劣势 | 0.0 | 同上 |

**两表 100% 匹配是 PR merge 的 hard gate**。

---

## 6. Cross-references

- Ch3.5.1 类型机制
- Ch3.5.2 type α reward (公式 3.5)
- Ch3.5.3 type β reward (公式 3.6-3.9)
- Ch3.5.4 Table 4 场景表（**精确数值匹配**）
- Ch4.1.1 偏导表（**autograd 精确匹配**）
- `02-resource-dynamics.md`（u_i 输入来源）
- `04-six-block-observation.md`（不读 reward, 但 Δ 写入 buffer 经由 info）
- `08-gym-api.md`（reward 与 info["deltas"] 字段定义）
- `../design.md` D3 (Δ 计算粒度)
