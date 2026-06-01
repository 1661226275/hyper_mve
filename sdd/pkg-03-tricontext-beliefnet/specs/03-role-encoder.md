# Spec 03: RoleEncoder — role 角色通路（含 type_emb，v4 关键改动）

> 父文档：[`../proposal.md`](../proposal.md) §1.2 · [`../design.md`](../design.md) §3 D8
> **本 spec 含 v4 关键改动** — type_emb 加入 role 通路是 Ch4 v4 第 1 项关键演进。

---

## 1. Purpose

按 Ch4.2.2 把 agent $i$ 的 (id, **type τ_i**, capability cap_i) 编码为 role 向量 $\in \mathbb{R}^{32}$，作为 TriContextEncoder 的第二路子模块。

$$\text{role}_i = \text{Concat}[\text{id\_emb}_i,\;\text{type\_emb}_i,\;\text{cap\_emb}_i] \in \mathbb{R}^{32}$$

默认子维度：$d_{\text{id}} = 8,\;d_{\text{type}} = 8,\;d_{\text{cap}} = 16$，精确填满 32（无 pad）。

**v4 关键改动**：相对 v3 (`role = id_emb + cap_emb`)，v4 加入 **type_emb**——这是 Chapter 4.1.3 节主层（瓶颈 1：类型梯度撕裂）的架构落地。type_emb_α 与 type_emb_β 是两个完全独立的可学习向量，通过 hyper_rew/hyper_pred 分别生成两套差异巨大的 θ_rew^i / θ_pred^i，从根本消除共享 RewardHead 的类型撕裂。

**Self-Info 严格性**（Ch3.7 + Ch4.2.2）：role_i **仅包含 agent i 自己**的 type 与 capability。他人的 type 不进入 role_i，而由 belief_i 通路（spec 04+05）推断。

---

## 2. Interface

### 2.1 文件路径

`hyper_mve/models/role_encoder.py`

### 2.2 类签名

```python
import torch
import torch.nn as nn
from hyper_mve.schemas.capability import CAP_NORM_LO, CAP_NORM_HI


class RoleEncoder(nn.Module):
    """role 角色通路 (Ch4.2.2, v4 含 type_emb).
    
    将 (id, type, cap) 三元组编码为 role_i ∈ ℝ^{d_role=32}.
    
    v4 关键 (相对 v3):
        v3: role = Concat[id_emb (8), cap_emb (16)] = 24 (+ 8 pad)
        v4: role = Concat[id_emb (8), type_emb (8), cap_emb (16)] = 32 精确
    
    Self-Info 严格性 (Ch3.7 + Ch4.2.2):
        本模块仅接收 agent 自己的 (type, cap), 不含他人信息.
        他人 type 由 BeliefNet head_opp 推断 (spec 05).
    
    C 修订（2026-05-28）— cap 输入自动归一化:
        forward() 接收 RAW CapabilityVector (B, N, 4), 内部按
        CAP_NORM_LO/HI (Pkg-01 spec 02 模块常量) 自动归一到 [0, 1]^4
        再喂 cap_mlp. 解决 ζ 量级 (10-30) vs η (0.5-1.5) ~30 倍差距导致
        cap_mlp 第一层梯度被 ζ 主导的问题.
        
        env/buffer/info 仍存 RAW cap; 仅本模块内部消费时归一化.
    
    P7 修订（2026-05-28）— cap_mlp 内部不含 LayerNorm:
        LN 责任集中在 TriContextEncoder.ln_role; cap_mlp 内 LN 与 ln_role
        是 double LN (cap 子段被 LN 两次), 故移除.
    """
    
    def __init__(
        self,
        N: int,
        d_id_emb: int = 8,
        d_type_emb: int = 8,
        d_cap_emb: int = 16,
        num_types: int = 2,         # AgentType.ALPHA / BETA
        cap_input_dim: int = 4,     # CapabilityVector: η, φ_fov, ν, ζ
        cap_hidden_dim: int = 16,   # cap_mlp 隐藏维 (D8)
    ):
        super().__init__()
        self.N = N
        self.d_id_emb = d_id_emb
        self.d_type_emb = d_type_emb
        self.d_cap_emb = d_cap_emb
        self.d_role = d_id_emb + d_type_emb + d_cap_emb
        
        # 精确填满约束 (与 Pkg-01 ModelConfig __post_init__ 一致)
        assert self.d_role == 32, (
            f"d_role={self.d_role}, expected 32 (8+8+16). v4 要求精确填满, 无 pad."
        )
        
        # id_emb: agent_id ∈ {0, ..., N-1} -> d_id_emb 维向量
        self.id_emb = nn.Embedding(N, d_id_emb)
        
        # type_emb (v4 关键): AgentType ∈ {ALPHA=0, BETA=1} -> d_type_emb 维向量
        self.type_emb = nn.Embedding(num_types, d_type_emb)
        
        # cap_emb: 4 维 CapabilityVector (已归一) -> d_cap_emb 维 (D8: 2 层 MLP)
        # P7 修订: 无内部 LayerNorm (LN 在 TriContextEncoder.ln_role)
        self.cap_mlp = nn.Sequential(
            nn.Linear(cap_input_dim, cap_hidden_dim),
            nn.ReLU(),
            nn.Linear(cap_hidden_dim, d_cap_emb),
        )
        
        # cap 归一化常量 (C 修订) - 注册为 buffer 以正确处理 device 迁移
        # CAP_NORM_LO = (0.5, 2.0, 0.8, 10.0), CAP_NORM_HI = (1.5, 4.0, 1.0, 30.0)
        self.register_buffer(
            "_cap_lo", torch.tensor(CAP_NORM_LO, dtype=torch.float32),
        )
        self.register_buffer(
            "_cap_hi", torch.tensor(CAP_NORM_HI, dtype=torch.float32),
        )
    
    def _normalize_caps(self, caps_raw: torch.Tensor) -> torch.Tensor:
        """归一 (B, N, 4) raw cap 到 [0, 1]^4 (C 修订).
        
        与 Pkg-01 spec 02 `CapabilityVector.normalize()` 数学等价 (向量化版本).
        范围常量 CAP_NORM_LO/HI 来自 Pkg-01 (Ch3.6 硬约束).
        """
        # caps_raw: (B, N, 4); _cap_lo / _cap_hi: (4,) broadcast 自动适配
        return (caps_raw - self._cap_lo) / (self._cap_hi - self._cap_lo)
    
    def forward(
        self,
        agent_ids: torch.Tensor,            # (B, N) int64
        types: torch.Tensor,                # (B, N) int64 (AgentType.value)
        caps: torch.Tensor,                 # (B, N, 4) float32 RAW CapabilityVector
    ) -> torch.Tensor:                      # (B, N, 32) float32
        """
        Args:
            agent_ids: (B, N) agent 索引 ∈ {0, ..., N-1}
            types:     (B, N) AgentType.value ∈ {0, 1}
            caps:      (B, N, 4) RAW CapabilityVector flatten (η, φ_fov, ν, ζ)
                       内部自动归一到 [0, 1]^4 (C 修订)
        Returns:
            role_i: (B, N, 32) = Concat[id_emb, type_emb, cap_emb]
        """
        B, N = agent_ids.shape
        assert types.shape == (B, N)
        assert caps.shape == (B, N, 4)
        
        id_vec = self.id_emb(agent_ids)                # (B, N, 8)
        type_vec = self.type_emb(types)                # (B, N, 8)
        
        # C 修订: cap 归一化后再喂 MLP
        caps_normed = self._normalize_caps(caps)       # (B, N, 4) ∈ [0, 1]^4
        cap_vec = self.cap_mlp(caps_normed)            # (B, N, 16)
        
        role = torch.cat([id_vec, type_vec, cap_vec], dim=-1)  # (B, N, 32)
        return role
```

### 2.3 典型用例

```python
from hyper_mve.models.role_encoder import RoleEncoder

# Medium preset N=4
enc = RoleEncoder(N=4, d_id_emb=8, d_type_emb=8, d_cap_emb=16)

B = 2
agent_ids = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])  # (B, N)
types = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]])      # 2 batches, 2α+2β each
caps = torch.rand(B, 4, 4)                              # (B, N=4, 4) cap features

role = enc(agent_ids, types, caps)                      # (B=2, N=4, 32)

# 验证内部子段
id_segment = role[..., :8]                              # id_emb 子段
type_segment = role[..., 8:16]                          # type_emb 子段 (v4)
cap_segment = role[..., 16:32]                          # cap_emb 子段
```

---

## 3. Implementation Notes

### 3.1 为何 type_emb 而非 one-hot concat（v4 关键设计）

- Ch4.2.2 明确：type_emb 通过 nn.Embedding 查表，**不**用 one-hot 直接 concat
- 学习独立可学向量比固定 one-hot 表达力更强（embedding 在反向传播中可调整）
- type_emb_α 与 type_emb_β 完全独立的 8 维向量，hyper_rew 看到它们时生成的 θ_rew^i 显著不同
- one-hot 直接 concat 是"硬编码"，缺乏可学性
- 与 v4.7 `BaselineModel` 的 id_embedding 设计模式一致（Pkg-01 D1 决策）

### 3.2 子维度精确填满约束（v4 关键）

- d_id (8) + d_type (8) + d_cap (16) = 32 精确等于 d_role
- 不接受 pad（如 8+4+16=28 + 4 pad）
- Pkg-01 ModelConfig.__post_init__ 已验证此约束，本模块 init 时再次断言
- v3 (8+16=24) 用 pad 凑 32 已废弃；v4 type_emb 8 维填上 pad 空间

### 3.3 Self-Info 严格性的代码体现

- 本模块输入 `types: (B, N)` 表示**每个 batch 中第 i 个 agent 的 type**
- Pkg-04 model.set_context 调用本模块时，必须**仅传入当前 agent 视角下的 own type**
- 如果 trainer 在 train loop 中误把 `info["types"]` （含所有 N 个 agent 真实 type）作为本模块的 types 输入，相当于把他人 type 也通过 role 通路泄漏给 hypernet——**违反 Ch3.7 Self-Info 原则**
- spec 08 集成契约明确：trainer 端构造本模块 types 输入时，只能基于 `agent_self_types`（自报告），不能用 `env.info["types"]`（Oracle 监督专用）

### 3.4 cap_mlp 结构（D8 + C + P7 修订）

**最终结构**：
```
input (4) ← 内部自动 normalize 到 [0, 1]^4
  → Linear(4, 16) → ReLU → Linear(16, 16)
  (无内部 LayerNorm; LN 责任在 TriContextEncoder.ln_role 集中)
```

**C 修订（cap 输入归一化）**：
- 早期 warmup 期 cap 4 维量级悬殊（ζ vs η ~30 倍）会让 cap_mlp 第一层梯度被 ζ 列主导，η/ν 信号被淹没
- 解决：forward 内部用 `_normalize_caps()` 自动归一（数学等价 Pkg-01 `CapabilityVector.normalize()`）
- CAP_NORM_LO/HI 常量从 `hyper_mve.schemas.capability` 导入（Pkg-01 spec 02），与 Ch3.6 采样范围一一对应
- 归一化常量作为 `register_buffer` 注册，正确处理 device 迁移

**P7 修订（无内部 LN）**：
- 移除 v4 草稿中的 `nn.LayerNorm(d_cap_emb)`
- 理由：与 TriContextEncoder.ln_role 是 double LN（cap 子段被 LN 两次）；LN 责任集中在 TriContextEncoder

**为什么"内部自动归一"而非"trainer 端归一后传入"**：
- 避免调用方忘记调用 `normalize()` 的 silent bug
- 接口签名 `caps: (B, N, 4)` 保持简单（raw cap，与 TimeStepRecord.cap / env.info["caps"] 一致）
- 性能开销：单步 (B, N, 4) 减法 + 除法 < 10 μs，可忽略

### 3.5 参数量（P7 修订后）

| 子模块 | 参数 |
|--------|------|
| id_emb | N × 8 = 32 (Medium N=4) |
| type_emb (v4) | 2 × 8 = 16 |
| cap_mlp Linear(4→16) | 4×16 + 16 = 80 |
| cap_mlp Linear(16→16) | 16×16 + 16 = 272 |
| ~~cap_mlp LayerNorm(16)~~ | ~~32~~ (P7 移除) |
| _cap_lo / _cap_hi buffer | 4 + 4 = 8 (非可学参数) |
| **总计可学参数** | ~400 (Medium) |

参数量随 N 线性增长（id_emb），其他不变。Hard preset (N=8) 总可学参数 ~432。

### 3.6 episode 内 role_i 缓存优化（实施建议，非本包必须）

- role_i = f(id_i, τ_i, cap_i)，三者在 episode 内不变 → role_i 也不变
- Pkg-04 model.set_context 可以在 episode 开始时调用一次本模块并缓存，避免每 step 重算
- 本模块不强制缓存（保持纯函数 forward），缓存由调用方实现

---

## 4. Edge Cases

| 场景 | 行为 |
|------|------|
| agent_ids >= N | nn.Embedding 越界 IndexError |
| types >= 2 | type_emb nn.Embedding 越界 IndexError |
| caps shape != (B, N, 4) | _normalize_caps broadcast 失败 RuntimeError |
| N=2 (Easy) / N=8 (Hard) | 正常运行（N 仅影响 id_emb 表大小） |
| caps 超出 [CAP_NORM_LO, CAP_NORM_HI] 范围 | 归一化后 < 0 或 > 1，MLP 仍正常输出（无校验） |
| caps 边界值 (η=0.5) | 归一后第一维 = 0.0，正常 |
| caps 中点 (η=1.0) | 归一后第一维 = 0.5，正常 |
| dtype mismatch（agent_ids float） | nn.Embedding 报错（要求 long） |
| caps 在 GPU 但 cap_lo buffer 在 CPU | register_buffer 自动随 module 迁移 device，不会发生 |

---

## 5. Acceptance Criteria

### 5.1 单元测试（`tests/models/test_role_encoder.py`）

```python
import pytest
import torch
from hyper_mve.models.role_encoder import RoleEncoder


def test_output_shape():
    """role shape (B, N, 32)."""
    enc = RoleEncoder(N=4)
    B = 2
    agent_ids = torch.arange(4).unsqueeze(0).expand(B, 4)
    types = torch.tensor([[0, 0, 1, 1]] * B)
    caps = torch.rand(B, 4, 4)
    
    role = enc(agent_ids, types, caps)
    assert role.shape == (B, 4, 32)


def test_role_dim_exact_fill_v4():
    """v4 关键: d_role = d_id + d_type + d_cap = 8 + 8 + 16 = 32 精确, 无 pad."""
    enc = RoleEncoder(N=4, d_id_emb=8, d_type_emb=8, d_cap_emb=16)
    assert enc.d_role == 32
    assert enc.d_id_emb + enc.d_type_emb + enc.d_cap_emb == enc.d_role


def test_role_dim_assertion_on_mismatch():
    """d_id + d_type + d_cap != 32 应抛."""
    with pytest.raises(AssertionError, match="d_role"):
        RoleEncoder(N=4, d_id_emb=8, d_type_emb=4, d_cap_emb=16)  # 28 != 32


def test_type_emb_distinct_for_alpha_beta():
    """v4 关键: type α 与 type β 的 embedding 应不同 (训练后)."""
    enc = RoleEncoder(N=2)
    # 初始随机, 仅检查表大小
    assert enc.type_emb.num_embeddings == 2
    assert enc.type_emb.embedding_dim == 8
    
    # 同一 agent_id 不同 type 输出 role 不同
    agent_ids = torch.tensor([[0]])
    cap = torch.rand(1, 1, 4)
    role_alpha = enc(agent_ids, torch.tensor([[0]]), cap)
    role_beta = enc(agent_ids, torch.tensor([[1]]), cap)
    # type 子段 (8:16) 必然不同
    assert not torch.allclose(role_alpha[..., 8:16], role_beta[..., 8:16])


def test_id_emb_distinct():
    """不同 agent_id 的 id_emb 子段不同."""
    enc = RoleEncoder(N=4)
    types = torch.zeros(1, 4, dtype=torch.long)
    caps = torch.rand(1, 4, 4)
    
    role = enc(torch.tensor([[0, 1, 2, 3]]), types, caps)
    # id_emb 子段 (0:8) 在 4 个 agent 间不同
    for i in range(4):
        for j in range(i + 1, 4):
            assert not torch.allclose(role[0, i, :8], role[0, j, :8])


def test_self_info_severity_only_own_type():
    """Self-Info 严格性: 本模块仅看 own type, 不接受 (N, N) 全 agent type 矩阵."""
    enc = RoleEncoder(N=4)
    # 正确: types shape (B, N) - 每个 agent 一个 type (自己的)
    agent_ids = torch.tensor([[0, 1, 2, 3]])
    types = torch.tensor([[0, 0, 1, 1]])      # 每个 agent 自己的 type
    caps = torch.rand(1, 4, 4)
    
    role = enc(agent_ids, types, caps)
    assert role.shape == (1, 4, 32)
    
    # 错误用法 (会被 shape check 捕获):
    # 假设 trainer 误传 types shape (B, N, N) (全 N×N 类型矩阵)
    bad_types = torch.zeros(1, 4, 4, dtype=torch.long)
    with pytest.raises((AssertionError, RuntimeError)):
        enc(agent_ids, bad_types, caps)


def test_cap_normalization_handles_scale_diff():
    """C 修订: cap 4 维量级悬殊在内部 normalize 后被消除.
    
    验证: 输入 raw cap (含 ζ=30 vs η=0.5 差 60x), normalize 后所有维 ∈ [0, 1].
    """
    enc = RoleEncoder(N=4)
    agent_ids = torch.tensor([[0, 1, 2, 3]])
    types = torch.tensor([[0, 0, 1, 1]])
    # 极端 cap: η=0.5, φ_fov=4, ν=0.8, ζ=30 (规模差 60x)
    caps = torch.tensor([[
        [0.5, 4.0, 0.8, 30.0],
        [1.5, 2.0, 1.0, 10.0],
        [1.0, 3.0, 0.9, 20.0],
        [0.8, 4.0, 0.85, 25.0],
    ]], dtype=torch.float32)
    
    # 直接验证 _normalize_caps 输出 ∈ [0, 1]^4
    caps_normed = enc._normalize_caps(caps)
    assert (caps_normed >= 0.0).all()
    assert (caps_normed <= 1.0).all()
    # 边界: agent 0 (η=0.5) → 第 0 维 = 0.0; agent 1 (η=1.5) → 第 0 维 = 1.0
    assert torch.isclose(caps_normed[0, 0, 0], torch.tensor(0.0))
    assert torch.isclose(caps_normed[0, 1, 0], torch.tensor(1.0))
    
    # forward 输出无 NaN/Inf
    role = enc(agent_ids, types, caps)
    assert not torch.isnan(role).any()
    assert not torch.isinf(role).any()


def test_cap_normalization_matches_pkg01_normalize():
    """C 修订: _normalize_caps 与 Pkg-01 CapabilityVector.normalize() 数学等价."""
    from hyper_mve.schemas import CapabilityVector
    
    enc = RoleEncoder(N=4)
    cap_dataclass = CapabilityVector(eta=1.2, phi_fov=2.5, nu=0.85, zeta=15.0)
    
    # Pkg-01 path
    normed_pkg01 = cap_dataclass.normalize()  # (4,) numpy
    
    # Pkg-03 path (vectorized)
    cap_tensor = torch.tensor([[[1.2, 2.5, 0.85, 15.0]]], dtype=torch.float32)
    normed_pkg03 = enc._normalize_caps(cap_tensor)  # (1, 1, 4)
    
    # 数学等价 (容差 1e-6)
    import numpy as np
    assert np.allclose(normed_pkg03[0, 0].numpy(), normed_pkg01, atol=1e-6)


def test_cap_mlp_no_internal_layernorm():
    """P7 修订: cap_mlp 内部不含 LayerNorm (LN 责任在 TriContextEncoder)."""
    enc = RoleEncoder(N=4)
    has_ln_in_cap_mlp = any(
        isinstance(m, torch.nn.LayerNorm) for m in enc.cap_mlp.modules()
    )
    assert not has_ln_in_cap_mlp, (
        "cap_mlp should NOT contain LayerNorm (P7: LN 责任集中在 TriContextEncoder.ln_role)"
    )


def test_cap_norm_buffer_registered():
    """C 修订: _cap_lo / _cap_hi 注册为 buffer (随 device 迁移)."""
    enc = RoleEncoder(N=4)
    # buffer 名字在 named_buffers 中
    buffer_names = [name for name, _ in enc.named_buffers()]
    assert "_cap_lo" in buffer_names
    assert "_cap_hi" in buffer_names
    # 数值与 Pkg-01 常量一致
    from hyper_mve.schemas.capability import CAP_NORM_LO, CAP_NORM_HI
    assert torch.allclose(enc._cap_lo, torch.tensor(CAP_NORM_LO))
    assert torch.allclose(enc._cap_hi, torch.tensor(CAP_NORM_HI))


def test_cap_norm_buffer_device_migration():
    """C 修订: _cap_lo / _cap_hi 随 module .cuda() 迁移."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    enc = RoleEncoder(N=4).cuda()
    assert enc._cap_lo.device.type == "cuda"
    assert enc._cap_hi.device.type == "cuda"


def test_gradient_flow():
    """反向传播覆盖 id_emb + type_emb + cap_mlp 所有参数."""
    enc = RoleEncoder(N=4)
    agent_ids = torch.tensor([[0, 1, 2, 3]])
    types = torch.tensor([[0, 0, 1, 1]])
    caps = torch.rand(1, 4, 4)
    
    role = enc(agent_ids, types, caps)
    loss = (role ** 2).sum()
    loss.backward()
    
    for name, p in enc.named_parameters():
        assert p.grad is not None, f"No grad for {name}"
        assert p.grad.abs().sum() > 0, f"Zero grad for {name}"


def test_episode_invariance():
    """同一 (id, type, cap) 多次调用输出一致 (无随机性, episode 内可缓存)."""
    enc = RoleEncoder(N=4)
    enc.eval()  # 关闭 dropout (本模块无 dropout 但仍 eval)
    
    agent_ids = torch.tensor([[0, 1, 2, 3]])
    types = torch.tensor([[0, 0, 1, 1]])
    caps = torch.rand(1, 4, 4)
    
    role_1 = enc(agent_ids, types, caps)
    role_2 = enc(agent_ids, types, caps)
    assert torch.allclose(role_1, role_2)


def test_param_count_medium():
    """Medium preset (N=4) 参数量 ~432."""
    enc = RoleEncoder(N=4, d_id_emb=8, d_type_emb=8, d_cap_emb=16)
    total = sum(p.numel() for p in enc.parameters())
    assert 400 < total < 500


def test_n_scaling():
    """id_emb 表大小随 N 缩放."""
    enc2 = RoleEncoder(N=2)
    enc8 = RoleEncoder(N=8)
    assert enc2.id_emb.num_embeddings == 2
    assert enc8.id_emb.num_embeddings == 8
```

### 5.2 性能要求

- 单步 forward (B=256, N=4) < 1 ms (V100)
- 参数量 ≤ 500 (Medium N=4)

---

## 6. v3 → v4 演进对比

| 维度 | v3/v4.6 (`context_encoder.py`, 已归档) | v4 (本 spec) |
|------|---------------------------------------|---------------|
| role 组成 | id_emb + cap_emb | **id_emb + type_emb + cap_emb** |
| d_role | 24 (+ 8 pad to align) | 32 (精确填满) |
| type 信息 | 不进入 role（仅在 RewardHead 末端用） | **进入 role**（hyper_rew 接收）|
| 类型梯度撕裂应对 | 共享 RewardHead 学平均偏导 | **per-type hyper_rew 生成专属 θ_rew^i** |
| 与 Self-Info 关系 | 隐含 | **显式**（own type only） |
| cap 输入处理 | raw 喂 MLP (量级悬殊) | **内部归一到 [0, 1]^4**（C 修订） |
| cap_mlp LN | 内部 LayerNorm | **无内部 LN**（P7 修订，责任集中 TriContextEncoder） |

---

## 7. Cross-references

- Ch4.2.2 角色通路（公式 + v4 关键改动 type_emb）
- Ch4.1.3 三层 motivation（主层：type_emb 解决类型梯度撕裂）
- Ch4.3.3 hyper_rew 接收完整三联输入（含 type_emb 经 role）
- `01-tri-context-encoder.md`（TriContextEncoder 调用本模块 + ln_role 集中 LN）
- `08-integration-contracts.md`（Self-Info 严格性集成约定）
- Pkg-01 `01-agent-type-schema.md`（AgentType Enum + type_emb 设计决策 D1）
- Pkg-01 `02-capability-vector.md`（**CapabilityVector + normalize() + CAP_NORM_LO/HI 常量**，C 修订集成来源）
- Pkg-01 `05-v4-config-structure.md` ModelConfig.d_id_emb/d_type_emb/d_cap_emb
- Pkg-02 `08-gym-api.md`（env.info["caps"] 来源；env 仍存 raw cap，本模块内部归一）
- Pkg-04 spec `02-hyper-muzero-model-v2.md`（set_context 内调用本模块，传 raw cap）
