# Spec 02: CEncoder — c_ctx 客观通路编码器

> 父文档：[`../proposal.md`](../proposal.md) §1.2 · [`../design.md`](../design.md) §3 D7

---

## 1. Purpose

将共享上下文标量 $c_t \in [0, 1]$（Harsanyi 共同知识）编码为 $c_{\text{ctx}} \in \mathbb{R}^{d_c=16}$，作为 TriContextEncoder 的第一路子模块。同时是 hyper_trans 客观通路的唯一输入（保证物理状态预测在所有 agent 间共享）。

按 Ch4.2.1 公式：

$$\text{c\_ctx} = \text{MLP}_{\text{c}}(c_t) \in \mathbb{R}^{16}$$

MLP 结构：`[1 → 32 → 32 → 16]` + ReLU + LayerNorm。

---

## 2. Interface

### 2.1 文件路径

`hyper_mve/models/c_encoder.py`

### 2.2 类签名

```python
import torch
import torch.nn as nn


class CEncoder(nn.Module):
    """c_t 客观通路编码器 (Ch4.2.1).
    
    将共享 context 标量 c_t ∈ [0, 1] 映射到 c_ctx ∈ ℝ^{d_c=16}.
    
    结构 (Ch4.2.1 默认):
        Linear(1, 32) -> ReLU -> Linear(32, 32) -> ReLU -> Linear(32, 16)
    
    输出不带 LayerNorm (LN 在 TriContextEncoder 调用层做, D7).
    """
    
    def __init__(self, d_c: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.d_c = d_c
        self.hidden_dim = hidden_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_c),
        )
    
    def forward(
        self,
        c_t: torch.Tensor,                  # (B, 1) float32
    ) -> torch.Tensor:                      # (B, d_c) float32
        """
        Args:
            c_t: shape (B, 1) 共享 context 标量
        Returns:
            c_ctx: shape (B, d_c=16)
        """
        assert c_t.dim() == 2 and c_t.shape[-1] == 1, (
            f"c_t shape {c_t.shape}, expected (B, 1)"
        )
        return self.mlp(c_t)
```

### 2.3 典型用例

```python
# 单独调用 (Pkg-04 hyper_trans 内)
from hyper_mve.models.c_encoder import CEncoder

c_encoder = CEncoder(d_c=16)
c_t = torch.tensor([[0.3], [0.7], [0.5]])  # (B=3, 1)
c_ctx = c_encoder(c_t)                     # (3, 16)

# 在 TriContextEncoder 内被调用 (含外部 LN)
# 见 spec 01
```

---

## 3. Implementation Notes

### 3.1 为何 MLP 而非线性

- Ch4.2.1 默认结构是 MLP（含两层 ReLU 非线性）
- c_t ∈ [0, 1] 是单维度连续值，单线性 (1→16) 只能学到 affine 变换 (16 个固定方向)
- 论文场景中 c_t 与社会福利的关系非线性（φ(c) = κ(1-2c) 是线性 + ψ(Δ) 折线，组合非线性），MLP 提供必要的表达能力
- 32 hidden 是 Ch4.2.1 默认，参数量 1×32 + 32×32 + 32×16 = 1568 + bias ≈ 1.6K

### 3.2 输出 LayerNorm 在外部

- 本模块输出 raw MLP 结果，不含 LN
- LN 由 TriContextEncoder 调用层添加（D7：每路独立 LN）
- 单独使用本模块（如 hyper_trans 仅消费 c_ctx）时，调用方需自行决定是否 LN

### 3.3 c_t 输入 shape 约束

- 严格要求 (B, 1) 而非 (B,)
- 调用方需 unsqueeze；TriContextEncoder 已处理（spec 01 forward 内部 unsqueeze 后传入）
- 这样避免 nn.Linear(1, 32) 收到 1D tensor 触发 shape 推断 warning

### 3.4 与 v4.6 c_encoder 的对比

| 维度 | v4.6 rule_encoder (已归档) | v4 CEncoder (本 spec) |
|------|---------------------------|----------------------|
| 输入 | rule_idx (int) 离散派系 | c_t (float) 连续标量 |
| 编码 | nn.Embedding(num_rules, d_rule) | MLP(1 → 32 → 32 → 16) |
| 维度 | d_rule = 8/16/24 (配置) | d_c = 16 (Ch4.2.1 默认) |
| LN | 内部含 LN | 外部 LN (D7) |

v4 改用 MLP 是因为 c_t 是连续值不是离散索引；nn.Embedding 仅适用离散输入。

### 3.5 参数量

- 总参数：1568 (weights) + 32 + 32 + 16 (biases) = 1648 ≈ 1.6K
- 占模型总参数量 (~3.2M) 0.05%，可忽略

---

## 4. Edge Cases

| 场景 | 行为 |
|------|------|
| c_t shape (B,) 而非 (B, 1) | AssertionError |
| c_t = 0.0 / 1.0 边界值 | 正常输出 (MLP 不强制 [0,1] 输入约束) |
| c_t < 0 或 > 1 | 不校验，输出可能偏离训练分布 |
| c_t 含 NaN | 输出含 NaN（不做特殊处理） |
| B = 1 | 正常运行 |

---

## 5. Acceptance Criteria

### 5.1 单元测试（`tests/models/test_c_encoder.py`）

```python
import pytest
import torch
from hyper_mve.models.c_encoder import CEncoder


def test_output_shape_default():
    """默认 d_c=16."""
    enc = CEncoder()
    c_t = torch.zeros(8, 1)
    out = enc(c_t)
    assert out.shape == (8, 16)


def test_output_shape_custom_d_c():
    """d_c 可配置."""
    enc = CEncoder(d_c=24)
    c_t = torch.zeros(8, 1)
    out = enc(c_t)
    assert out.shape == (8, 24)


def test_input_shape_assertion():
    """c_t 必须 (B, 1), 不接 (B,)."""
    enc = CEncoder()
    c_t = torch.zeros(8)
    with pytest.raises(AssertionError, match="c_t shape"):
        enc(c_t)


def test_gradient_flow():
    """反向传播覆盖所有可学参数."""
    enc = CEncoder()
    c_t = torch.rand(4, 1, requires_grad=True)
    out = enc(c_t)
    loss = (out ** 2).sum()
    loss.backward()
    for name, p in enc.named_parameters():
        assert p.grad is not None, f"No grad for {name}"
        assert p.grad.norm() > 0, f"Zero grad for {name}"


def test_distinct_c_distinct_output():
    """不同 c_t 应产生不同 c_ctx (确保 MLP 非常数函数)."""
    enc = CEncoder()
    c_t_a = torch.tensor([[0.2]])
    c_t_b = torch.tensor([[0.8]])
    out_a = enc(c_t_a)
    out_b = enc(c_t_b)
    assert not torch.allclose(out_a, out_b)


def test_param_count():
    """MLP 默认参数量 ≈ 1.6K."""
    enc = CEncoder()
    total = sum(p.numel() for p in enc.parameters())
    assert 1500 < total < 1800


def test_boundary_values():
    """c=0.0 和 c=1.0 边界正常."""
    enc = CEncoder()
    c_t = torch.tensor([[0.0], [1.0]])
    out = enc(c_t)
    assert out.shape == (2, 16)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_no_internal_layernorm():
    """本模块输出不带 LN, 由 TriContextEncoder 外部加."""
    enc = CEncoder()
    # 模块中无 LayerNorm 子模块
    has_ln = any(isinstance(m, torch.nn.LayerNorm) for m in enc.modules())
    assert not has_ln, "CEncoder should NOT contain internal LayerNorm (D7: external)"
```

### 5.2 性能要求

- 单步 forward (B=256, c_t shape (256, 1)) < 0.5 ms (V100)
- 参数量 < 2K
- 内存占用 < 50KB

---

## 6. Cross-references

- Ch4.2.1 客观通路（公式 + 默认 MLP 结构）
- `01-tri-context-encoder.md`（TriContextEncoder 调用本模块）
- `08-integration-contracts.md`（hyper_trans 调用 forward_c_ctx_only）
- Pkg-04 spec `01-dualhypernet-v2-api.md`（hyper_trans 仅消费 c_ctx）
- Pkg-01 `05-v4-config-structure.md` ModelConfig.d_c
