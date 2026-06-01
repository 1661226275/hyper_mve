# Spec 07: PermutationInvariantPool — Pool({ẑ_{i,j}}) 三策略

> 父文档：[`../proposal.md`](../proposal.md) §1.2 · [`../design.md`](../design.md) §3 D2

---

## 1. Purpose

按 Ch4.2.3 实现 `Pool({ẑ_{i,j}}_{j \neq i})` 模块：把 agent $i$ 对 $N - 1$ 个对手的类型预测 $\hat{z}_{i,j} \in \Delta^2$ 聚合为单个 set-invariant 表示，喂入 TriContextEncoder 的 belief 子通路。

提供三种策略（D2）：
- **mean**（默认）：`pool = (1/(N-1)) * sum_j z_j`
- **max**：`pool = elementwise_max_j z_j`
- **attention**：`pool = sum_j softmax(query · k_j) * v_j`（learnable query）

策略由 `ModelConfig.belief_pool` 切换（Pkg-01 默认 `"mean"`）。

---

## 2. Interface

### 2.1 文件路径

`hyper_mve/models/permutation_invariant_pool.py`

### 2.2 公开 API

```python
import torch
import torch.nn as nn
from typing import Literal


class MeanPool(nn.Module):
    """Mean pooling over the second-to-last dim (默认: dim=-2).
    
    输入: (..., N-1, F)
    输出: (..., F)
    
    数学: pool = (1/(N-1)) * sum_j x_j
    
    无可学参数. 严格 set-invariant.
    """
    
    def __init__(self, feat_dim: int):
        super().__init__()
        self.feat_dim = feat_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=-2)


class MaxPool(nn.Module):
    """Elementwise max pooling over the second-to-last dim.
    
    输入: (..., N-1, F)
    输出: (..., F)
    
    数学: pool[f] = max_j x[j, f]
    
    无可学参数. 严格 set-invariant.
    """
    
    def __init__(self, feat_dim: int):
        super().__init__()
        self.feat_dim = feat_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.max(dim=-2).values


class AttentionPool(nn.Module):
    """Attention pooling with learnable query.
    
    输入: (..., N-1, F)
    输出: (..., F)
    
    数学:
        weights[j] = softmax_j(query · k_j)
        pool = sum_j weights[j] * v_j
    
    K, V 通过 linear projection 从 x 得到. Q 是 learnable parameter.
    """
    
    def __init__(self, feat_dim: int, head_dim: int = 16):
        super().__init__()
        self.feat_dim = feat_dim
        self.head_dim = head_dim
        
        # learnable query (1, head_dim)
        self.query = nn.Parameter(torch.randn(1, head_dim) * 0.02)
        
        # K, V projections from input
        self.k_proj = nn.Linear(feat_dim, head_dim, bias=False)
        self.v_proj = nn.Linear(feat_dim, feat_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., N-1, F)
        K = self.k_proj(x)                              # (..., N-1, head_dim)
        V = self.v_proj(x)                              # (..., N-1, F)
        
        # attention weights: (..., N-1, 1)
        # query shape (1, head_dim) -> broadcast 到 (..., N-1, head_dim) 内积
        scores = (K * self.query).sum(dim=-1, keepdim=True)  # (..., N-1, 1)
        scores = scores / (self.head_dim ** 0.5)             # scale
        weights = torch.softmax(scores, dim=-2)              # softmax over N-1
        
        # weighted sum
        pool = (weights * V).sum(dim=-2)                # (..., F)
        return pool


def make_pool(
    kind: Literal["mean", "max", "attention"],
    feat_dim: int,
    **kwargs,
) -> nn.Module:
    """工厂: 按 kind 构造对应 pool 模块.
    
    Args:
        kind:     "mean" / "max" / "attention"
        feat_dim: 特征维度 F
        **kwargs: 传递给特定 pool 类 (如 attention 的 head_dim)
    """
    if kind == "mean":
        return MeanPool(feat_dim)
    elif kind == "max":
        return MaxPool(feat_dim)
    elif kind == "attention":
        head_dim = kwargs.get("head_dim", 16)
        return AttentionPool(feat_dim, head_dim=head_dim)
    raise ValueError(f"Unknown pool kind: {kind}. Valid: mean/max/attention")
```

### 2.3 典型用例

```python
# TriContextEncoder 内部调用 (spec 01)
from hyper_mve.models.permutation_invariant_pool import make_pool

# 默认配置
pool = make_pool(kind="mean", feat_dim=2)
z_hat = torch.softmax(torch.randn(2, 4, 3, 2), dim=-1)  # (B=2, N=4, N-1=3, 2)
pooled = pool(z_hat)                                    # (2, 4, 2)

# 切换为 max
pool_max = make_pool(kind="max", feat_dim=2)
pooled_max = pool_max(z_hat)

# 切换为 attention (含可学参数)
pool_att = make_pool(kind="attention", feat_dim=2, head_dim=8)
pooled_att = pool_att(z_hat)
```

---

## 3. Implementation Notes

### 3.1 dim=-2 而非 dim=1（统一约定）

- pool 在倒数第二维（N-1 个对手维度）
- 不写死 dim=1，因为输入张量可能有不同前缀（(B, N-1, F)、(B, T, N, N-1, F) 等）
- TriContextEncoder spec 01 输入 z_hat shape (B, N, N-1, 2)，pool 在 dim=-2 输出 (B, N, 2)

### 3.2 mean / max set-invariant 严格性

- mean: `(1/(N-1)) * sum_j` 与 j 顺序无关
- max: `elementwise max` 与 j 顺序无关
- 单测 `test_set_invariance` 验证：打乱 N-1 维度，输出不变（mean/max）

### 3.3 attention 的近似 set-invariant

- attention 在 K, V 共享 projection 时是 set-invariant（数学上）
- 但训练初期 query 随机，可能引入 j 顺序偏好；训练收敛后趋近 set-invariant
- 实际中可接受（softmax 对置换不敏感的本质）
- 单测：attention 验证 set-invariance（误差 ≤ 1e-5）

### 3.4 配置切换

- `ModelConfig.belief_pool: str = "mean"` 是默认（Pkg-01 spec 05 字段）
- 切换无需改架构代码，只改 cfg
- 注意：attention 引入可学参数（~50 params for head_dim=16, feat_dim=2），mean/max 无

### 3.5 mean pool 在 simplex 上的语义

- z_hat 是 softmax 概率 ∈ Δ^2
- mean pool 在 simplex 上仍是有效概率分布（凸组合）
- 物理意义：「N-1 个对手类型的边际后验」（如 50% α 50% β agent 群里，pool 接近 (0.5, 0.5)）

### 3.6 与 Ch4.2.3 文档差异

- Ch4.2.3 公式：`belief_i = Concat[ĉ_i, Pool({ẑ_{i,j}})]`，pool 输出维度未硬约束
- 本 spec 输出维度 = feat_dim = 2（对应 z_hat 类别数）
- 后续 TriContextEncoder 内的 `proj_z_pooled: Linear(2, 16)` 把它投影到 d_belief_proj=16

---

## 4. Edge Cases

| 场景 | 行为 |
|------|------|
| N=2 → N-1=1 | mean/max 对单元素直接返回；attention 单元素 softmax = 1.0 |
| feat_dim 与输入不符 | attention 内 Linear shape mismatch RuntimeError |
| 输入全相同 | mean 输出该值；max 输出该值；attention 加权和 = 该值 |
| 输入含 NaN | 输出含 NaN（不特殊处理） |
| `kind` 未识别 | ValueError |

---

## 5. Acceptance Criteria

### 5.1 单元测试（`tests/models/test_permutation_pool.py`）

```python
import pytest
import torch
from hyper_mve.models.permutation_invariant_pool import (
    MeanPool, MaxPool, AttentionPool, make_pool,
)


def test_mean_pool_output_shape():
    pool = MeanPool(feat_dim=2)
    x = torch.randn(2, 4, 3, 2)             # (B, N, N-1, F)
    out = pool(x)
    assert out.shape == (2, 4, 2)


def test_max_pool_output_shape():
    pool = MaxPool(feat_dim=2)
    x = torch.randn(2, 4, 3, 2)
    out = pool(x)
    assert out.shape == (2, 4, 2)


def test_attention_pool_output_shape():
    pool = AttentionPool(feat_dim=2, head_dim=8)
    x = torch.randn(2, 4, 3, 2)
    out = pool(x)
    assert out.shape == (2, 4, 2)


def test_mean_pool_set_invariant():
    """mean pool 严格 set-invariant: 打乱顺序输出不变."""
    pool = MeanPool(feat_dim=2)
    x = torch.randn(2, 4, 5, 2)
    out_orig = pool(x)
    
    perm = torch.randperm(5)
    x_perm = x[:, :, perm, :]
    out_perm = pool(x_perm)
    
    assert torch.allclose(out_orig, out_perm, atol=1e-6)


def test_max_pool_set_invariant():
    pool = MaxPool(feat_dim=2)
    x = torch.randn(2, 4, 5, 2)
    out_orig = pool(x)
    
    perm = torch.randperm(5)
    x_perm = x[:, :, perm, :]
    out_perm = pool(x_perm)
    
    assert torch.allclose(out_orig, out_perm, atol=1e-6)


def test_attention_pool_set_invariant():
    """attention pool (K, V 共享 projection) 也应 set-invariant."""
    torch.manual_seed(0)
    pool = AttentionPool(feat_dim=2, head_dim=8)
    pool.eval()
    
    x = torch.randn(2, 4, 5, 2)
    out_orig = pool(x)
    
    perm = torch.randperm(5)
    x_perm = x[:, :, perm, :]
    out_perm = pool(x_perm)
    
    assert torch.allclose(out_orig, out_perm, atol=1e-5)


def test_mean_pool_computes_average():
    """验证 mean = sum / N."""
    pool = MeanPool(feat_dim=2)
    x = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],   # B=0, N=0
    ]).unsqueeze(0)                              # (1, 1, 3, 2)
    out = pool(x)
    expected = torch.tensor([[[3.0, 4.0]]])     # (1+3+5)/3=3, (2+4+6)/3=4
    assert torch.allclose(out, expected)


def test_max_pool_computes_max():
    """验证 max = elementwise max."""
    pool = MaxPool(feat_dim=2)
    x = torch.tensor([
        [[1.0, 6.0], [3.0, 4.0], [5.0, 2.0]],
    ]).unsqueeze(0)                              # (1, 1, 3, 2)
    out = pool(x)
    expected = torch.tensor([[[5.0, 6.0]]])     # max([1,3,5])=5, max([6,4,2])=6
    assert torch.allclose(out, expected)


def test_mean_pool_no_params():
    """MeanPool 无可学参数."""
    pool = MeanPool(feat_dim=2)
    total = sum(p.numel() for p in pool.parameters())
    assert total == 0


def test_max_pool_no_params():
    pool = MaxPool(feat_dim=2)
    total = sum(p.numel() for p in pool.parameters())
    assert total == 0


def test_attention_pool_has_params():
    """AttentionPool 含 query + K/V projection 参数."""
    pool = AttentionPool(feat_dim=2, head_dim=16)
    total = sum(p.numel() for p in pool.parameters())
    # query: 1*16=16; k_proj: 2*16=32; v_proj: 2*2=4; total=52
    assert 40 < total < 80


def test_make_pool_factory():
    """工厂构造正确类型."""
    mp = make_pool(kind="mean", feat_dim=2)
    assert isinstance(mp, MeanPool)
    
    xp = make_pool(kind="max", feat_dim=2)
    assert isinstance(xp, MaxPool)
    
    ap = make_pool(kind="attention", feat_dim=2, head_dim=8)
    assert isinstance(ap, AttentionPool)
    assert ap.head_dim == 8


def test_make_pool_unknown_kind():
    with pytest.raises(ValueError, match="Unknown pool kind"):
        make_pool(kind="weird", feat_dim=2)


def test_pool_n_minus_one_equals_1():
    """N=2 时 N-1=1, mean/max/attention 应正常处理单元素."""
    x = torch.randn(2, 2, 1, 2)                 # (B=2, N=2, N-1=1, F=2)
    
    mp = MeanPool(2)
    out_m = mp(x)
    assert out_m.shape == (2, 2, 2)
    # 单元素 mean 应直接返回该值
    assert torch.allclose(out_m, x.squeeze(-2))
    
    xp = MaxPool(2)
    out_x = xp(x)
    assert torch.allclose(out_x, x.squeeze(-2))
    
    ap = AttentionPool(2, head_dim=4)
    ap.eval()
    out_a = ap(x)
    assert out_a.shape == (2, 2, 2)


def test_gradient_flow_attention():
    """AttentionPool 反向传播覆盖 query / k_proj / v_proj."""
    pool = AttentionPool(feat_dim=2, head_dim=8)
    x = torch.randn(2, 4, 3, 2, requires_grad=True)
    out = pool(x)
    loss = (out ** 2).sum()
    loss.backward()
    
    for name, p in pool.named_parameters():
        assert p.grad is not None, f"No grad for {name}"
        assert p.grad.norm() > 0, f"Zero grad for {name}"
```

### 5.2 集成测试

集成在 TriContextEncoder spec 01 `test_permutation_invariance_belief_mean_pool` 中，验证 z_hat 顺序打乱后 belief 子段不变。

### 5.3 性能要求

- mean/max forward 单步 < 0.5 ms
- attention forward 单步 < 1 ms (V100)

---

## 6. AttentionPool 变体（P10，future ablation）

当前 `AttentionPool` 实现使用 **learnable query**（一个可学 `nn.Parameter`），是 set-attention 的简单形式。用户审阅 P10 提出另一种实现：**用 `b_i` 作为 query 的 cross-attention**——更符合"对手类型推断"的语义（query 是"agent i 的 belief"，attend to "N-1 个对手特征"）。

### 6.1 两种实现的对比

| 维度 | 默认 (learnable query, 当前) | b_i-query 变体 (P10 future) |
|------|------------------------------|----------------------------|
| Query 来源 | `nn.Parameter(1, head_dim)` 静态可学 | 调用方传入 `b_i: (B, N, h)` 动态条件 |
| 语义 | "学一个通用 attention 模式" | "agent i 的 belief 引导 attention" |
| 接口 | `pool(x)` | `pool(x, query)` |
| set-invariant | 严格 | 严格 (K/V 共享 projection) |
| 参数量 | ~52 (默认 head_dim=16) | ~120 (含 query projection from b_i) |
| 适用场景 | belief 通路 z_pool 默认 | 未来 ablation：BeliefNet head_opp 内部 attention |

### 6.2 b_i-query 变体的实现草案（future work，不在 v4 主线）

```python
class BiQueryAttentionPool(nn.Module):
    """以 b_i 作为 query 的 cross-attention pool (P10 future ablation).
    
    与 AttentionPool 差异:
        - query 不再是 learnable parameter, 而是调用方传入的 b_i
        - 需要 q_proj 把 b_i 投影到 head_dim
    
    用法 (未来 BeliefNet 内部 head_opp 改造):
        b_i (B, N, h_belief=128)
        z_per_opp (B, N, N-1, 2)
        pool_out = pool(z_per_opp, query=b_i)  # (B, N, 2)
    """
    
    def __init__(self, feat_dim: int, query_dim: int, head_dim: int = 16):
        super().__init__()
        self.feat_dim = feat_dim
        self.head_dim = head_dim
        
        self.q_proj = nn.Linear(query_dim, head_dim, bias=False)
        self.k_proj = nn.Linear(feat_dim, head_dim, bias=False)
        self.v_proj = nn.Linear(feat_dim, feat_dim, bias=False)
    
    def forward(
        self,
        x: torch.Tensor,                # (..., N-1, feat_dim)
        query: torch.Tensor,            # (..., query_dim)
    ) -> torch.Tensor:                  # (..., feat_dim)
        Q = self.q_proj(query).unsqueeze(-2)        # (..., 1, head_dim)
        K = self.k_proj(x)                          # (..., N-1, head_dim)
        V = self.v_proj(x)                          # (..., N-1, feat_dim)
        
        scores = (K * Q).sum(dim=-1, keepdim=True) / (self.head_dim ** 0.5)
        weights = torch.softmax(scores, dim=-2)
        return (weights * V).sum(dim=-2)
```

### 6.3 v4 主线不采用的理由

- v4 ablation 实验只用 `mean` pool（cfg.belief_pool 默认）；attention 仅作为 `Pkg-08 ablation 6.x` 候选
- 默认 learnable-query AttentionPool 已能覆盖 set-attention 语义，足以做 mean vs max vs attention 的对比
- b_i-query 变体引入接口变化（forward 多一个 query 参数），破坏 `make_pool(kind, feat_dim)` 工厂的统一签名
- 如未来 BeliefNet head_opp 内部需要 cross-attention（对每个对手 j 用 b_i 加权），可单独实现，**不**通过本工厂

**标注**：本节属于 future work，**不**在 v4 主线 PR 实现范围内。

---

## 7. Cross-references

- Ch4.2.3 Pool({ẑ_{i,j}}) 描述（默认 mean）
- `01-tri-context-encoder.md`（调用 make_pool）
- `05-belief-heads.md`（z_hat 来源）
- Pkg-01 `05-v4-config-structure.md` ModelConfig.belief_pool 字段
- Pkg-04 spec `01-dualhypernet-v2-api.md`（如 hypernet 内部需要 Pool 复用本工厂）
