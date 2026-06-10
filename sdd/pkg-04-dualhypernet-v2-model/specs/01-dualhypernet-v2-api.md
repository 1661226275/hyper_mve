# Spec 01: DualHyperNetwork v2 API

> 父文档：[`../proposal.md`](../proposal.md) §2.1 · [`../design.md`](../design.md) §3 D1 / D5 / §6.3
> **v4 关键改动**：v4.7 双路 `(rule_emb, id_emb)` → v4 双 forward API `forward_trans(c_ctx) + forward_subjective(ctx_aug)`。
>
> **[v4-opt 2026-06] 本 spec 已被优化阶段修订**：HyperNetMLP 新增 `output_groups`（分组 RMS 归一）与 `output_rank`（输出层 LoRA）；新增 `SubjectiveHyperNet`（`share_subjective_trunk`）；§3.3 的 scale 语义随分组 RMS 改变；§3.4 D5 改为 gen_scope 依赖。详见文末**修订 A1-A5 与修订记录**；事实底稿 `docs/Review_v4_TheoryAudit_2026-06.md` §3 与 DESIGN_DOC §5.12。

---

## 1. Purpose

按 Ch4.3.2 + Ch4.3.3 重写 `DualHyperNetwork`，从 v4.7 双路输入（`rule_emb + id_emb`）扩展为 v4 三路输入（`c_ctx + role + belief = 80 维`）。同时拆分 forward 为两个独立 API：

- **`forward_trans(c_ctx)`** → `θ_state`（客观通路，仅 16 维 c_ctx，**C1 物理转移上下文不变**）
- **`forward_subjective(ctx_aug)`** → `(θ_rew, θ_pred)`（主观通路，完整 80 维 ctx_aug）

保留 `HyperNetMLP` 类（v4.7 实现已含 L2 norm + output_scale 稳定性技巧，**结构不动**），仅修改 `DualHyperNetwork` 顶层组合。

---

## 2. Interface

### 2.1 文件路径

`hyper_mve/models/hyper_network.py`（v4.7 文件 inplace 重写，D8）

### 2.2 HyperNetMLP（v4.7 保留，结构不变）

```python
import torch
import torch.nn as nn
from utils.utils import orthogonal_init, small_init


class HyperNetMLP(nn.Module):
    """v4.7 保留: 从 context embedding 生成 flat parameter vector.

    Architecture: context -> FC1 -> LN -> ReLU -> FC2 -> LN -> ReLU -> FC_out -> flat_params

    稳定性技巧（spec 03 详述）:
        - small_init(std=0.01) 在 output_layer 防初期权重爆炸
        - 可选 L2 normalize + learnable output_scale (norm_output=True)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        norm_output: bool = True,
        output_scale_init: float = 0.01,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.LayerNorm(h), nn.ReLU()]
            prev_dim = h
        self.trunk = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)

        for m in self.trunk:
            if isinstance(m, nn.Linear):
                orthogonal_init(m)
        small_init(self.output_layer, std=0.01)

        self.norm_output = norm_output
        self.output_scale = nn.Parameter(torch.tensor(float(output_scale_init)))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        h = self.trunk(context)
        raw = self.output_layer(h)
        if self.norm_output:
            norms = torch.linalg.norm(raw, dim=-1, keepdim=True)
            raw = raw / (norms + 1e-8)
        return raw * self.output_scale
```

### 2.3 DualHyperNetwork v2（重写）

```python
class DualHyperNetwork(nn.Module):
    """v4 三路输入版 DualHyperNetwork (Ch4.3).
    
    v4 关键改动 (相对 v4.7):
        v4.7: __init__(rule_emb_dim, id_emb_dim, ...) + forward(rule_emb, id_emb)
        v4:   __init__(c_ctx_dim=16, ctx_aug_dim=80, ...) +
              forward_trans(c_ctx) → θ_state +
              forward_subjective(ctx_aug) → (θ_rew, θ_pred)
    
    三个 HyperNetMLP 内部独立 (不共用 trunk):
        - hyper_trans: c_ctx (16) → θ_state          ← 客观通路 (C1)
        - hyper_rew:   ctx_aug (80) → θ_rew^i        ← 主观通路 per-agent
        - hyper_pred:  ctx_aug (80) → θ_pred^i       ← 主观通路 per-agent (可选 detach context, D5)
    
    output_scale 三初值 (v4.7 经验保留, C8):
        - trans_output_scale_init = 0.01
        - rew_output_scale_init   = 0.1   ← v4.7 关键 (避免 init trap)
        - pred_output_scale_init  = 0.01
    """
    
    def __init__(
        self,
        c_ctx_dim: int,
        ctx_aug_dim: int,
        trans_param_count: int,
        rew_param_count: int,
        pred_param_count: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        rew_hidden_dims: tuple[int, ...] | None = (256, 256, 256),
        norm_output: bool = True,
        trans_output_scale_init: float = 0.01,
        rew_output_scale_init: float = 0.1,        # v4.7 关键
        pred_output_scale_init: float = 0.01,
        detach_pred_context: bool = True,          # D5: hyper_pred 输入 detach
    ):
        super().__init__()
        
        # C5: ctx_aug_dim 必须等于 D_CTX_AUG=80（与 Pkg-01 ModelConfig 一致）
        # 实际数值断言放 HyperMuZeroModel.__init__ (spec 02)，本类仅接受参数
        
        self.c_ctx_dim = c_ctx_dim
        self.ctx_aug_dim = ctx_aug_dim
        self.detach_pred_context = detach_pred_context
        
        rew_hdims = rew_hidden_dims if rew_hidden_dims is not None else hidden_dims
        
        # C1: hyper_trans 仅接 c_ctx_dim (16)
        self.hyper_trans = HyperNetMLP(
            input_dim=c_ctx_dim,
            output_dim=trans_param_count,
            hidden_dims=hidden_dims,
            norm_output=norm_output,
            output_scale_init=trans_output_scale_init,
        )
        
        # C2: hyper_rew 接完整 ctx_aug_dim (80)
        self.hyper_rew = HyperNetMLP(
            input_dim=ctx_aug_dim,
            output_dim=rew_param_count,
            hidden_dims=rew_hdims,                     # 更深 (3 层) for type 分化
            norm_output=norm_output,
            output_scale_init=rew_output_scale_init,   # 0.1 关键
        )
        
        # C2: hyper_pred 同接 ctx_aug_dim (80)
        self.hyper_pred = HyperNetMLP(
            input_dim=ctx_aug_dim,
            output_dim=pred_param_count,
            hidden_dims=hidden_dims,
            norm_output=norm_output,
            output_scale_init=pred_output_scale_init,
        )
        
        self.trans_param_count = trans_param_count
        self.rew_param_count = rew_param_count
        self.pred_param_count = pred_param_count
    
    def forward_trans(self, c_ctx: torch.Tensor) -> torch.Tensor:
        """C1: 仅接 c_ctx (B, 16) 生成 θ_state.
        
        改变 role / belief 输入时, 本方法输出不变 (物理转移上下文不变性).
        
        Args:
            c_ctx: (B, c_ctx_dim=16) float32
        Returns:
            θ_state: (B, trans_param_count) float32
        """
        assert c_ctx.shape[-1] == self.c_ctx_dim, (
            f"c_ctx last dim {c_ctx.shape[-1]} != c_ctx_dim {self.c_ctx_dim}"
        )
        return self.hyper_trans(c_ctx)
    
    def forward_subjective(
        self,
        ctx_aug: torch.Tensor,                    # (B, 80) c_ctx + role + belief
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """C2: 接完整 80 维 ctx_aug 生成 (θ_rew, θ_pred).
        
        若 detach_pred_context=True (D5), hyper_pred 接收的 ctx_aug 被 .detach():
            - 防止 value loss 反向时扭曲 context_encoder 学习
            - reward loss 反向仍能驱动 context_encoder（经 hyper_rew 路径）
        
        Args:
            ctx_aug: (B, ctx_aug_dim=80) float32
        Returns:
            θ_rew: (B, rew_param_count) float32
            θ_pred: (B, pred_param_count) float32
        """
        assert ctx_aug.shape[-1] == self.ctx_aug_dim, (
            f"ctx_aug last dim {ctx_aug.shape[-1]} != ctx_aug_dim {self.ctx_aug_dim}"
        )
        
        theta_rew = self.hyper_rew(ctx_aug)
        
        # D5: hyper_pred 输入是否 detach
        if self.detach_pred_context:
            theta_pred = self.hyper_pred(ctx_aug.detach())
        else:
            theta_pred = self.hyper_pred(ctx_aug)
        
        return theta_rew, theta_pred
```

### 2.4 reward_diversity_loss（v4.7 保留作辅助正则，可选）

```python
def reward_diversity_loss(
    hyper_rew: HyperNetMLP,
    ctx_aug_per_agent: torch.Tensor,    # (N, B, ctx_aug_dim) per-agent context
    target_cos: float = 0.3,
    skip_pairs: list[tuple[int, int]] | None = None,
) -> torch.Tensor:
    """[v4.7 保留] θ_reward 多样性正则.
    
    按 D2: v4 type-aware 已通过 type_emb → hyper_rew → θ_rew^i 隐含实现.
    本 loss 在 v4 作为**辅助正则**（cfg.train.w_rew_diversity 控制权重）,
    默认权重 0.0（禁用），断言 A 实验需要时启用做对照.
    
    Args:
        hyper_rew: HyperNetMLP 实例
        ctx_aug_per_agent: (N, B, ctx_aug_dim) 每个 agent 的 ctx_aug 张量
                           (v4.7 是 rule_emb + id_emb concat, v4 改为完整 ctx_aug)
        target_cos: hinge 阈值
        skip_pairs: 跳过的 (i, j) 对（如同 type agent 不需要分化）
    """
    import torch.nn.functional as F
    
    N, B, _ = ctx_aug_per_agent.shape
    thetas = [hyper_rew(ctx_aug_per_agent[i]) for i in range(N)]
    
    loss = torch.tensor(0.0, device=ctx_aug_per_agent.device)
    count = 0
    skip_set = set(skip_pairs) if skip_pairs else set()
    for i in range(N):
        for j in range(i + 1, N):
            if (i, j) in skip_set or (j, i) in skip_set:
                continue
            cos_sim = F.cosine_similarity(thetas[i], thetas[j], dim=-1)
            loss = loss + F.relu(cos_sim - target_cos).mean()
            count += 1
    
    return loss / max(count, 1)
```

### 2.5 典型用例

```python
from hyper_mve.models import DualHyperNetwork

# 1. 初始化（HyperMuZeroModel.__init__ 内部调用，spec 02）
hyper_net = DualHyperNetwork(
    c_ctx_dim=16,
    ctx_aug_dim=80,
    trans_param_count=37_000,     # FunctionalStateTransNet 参数总数
    rew_param_count=4_000,        # FunctionalRewardHead
    pred_param_count=42_000,      # FunctionalPredictionNet
    hidden_dims=(256, 256),
    rew_hidden_dims=(256, 256, 256),
    rew_output_scale_init=0.1,    # 关键
    detach_pred_context=True,     # D5 默认
)

# 2. 客观通路（model.set_context_objective 内）
c_t = torch.rand(B=256, 1)
c_ctx = tri_context_encoder.forward_c_ctx_only(c_t)  # (B, 16)
theta_state = hyper_net.forward_trans(c_ctx)         # (B, 37000)

# 3. 主观通路（model.set_context_subjective 内, 对每个 agent k）
ctx_aug_k = tri_context_encoder.forward(
    c_t, agent_ids[k:k+1], types_k, caps_k, belief_k
)  # (B, 80)
theta_rew_k, theta_pred_k = hyper_net.forward_subjective(ctx_aug_k.squeeze(1))
```

---

## 3. Implementation Notes

### 3.1 v4.7 → v4 API 变化

| 维度 | v4.7 | v4 |
|------|------|-----|
| `__init__` 输入维度 | `rule_emb_dim, id_emb_dim` (双路) | `c_ctx_dim=16, ctx_aug_dim=80` (双 dim) |
| forward 数量 | 单 `forward(rule_emb, id_emb)` 返回 3 个 θ | 双 forward：`forward_trans` + `forward_subjective` |
| `generate_trans_params` / `generate_subjective_params` | 显式独立方法 | **废弃**（由新双 forward 替代） |
| reward_diversity_loss 输入 | `(rule_emb, id_embedding, num_agents)` | `(ctx_aug_per_agent)` 直接传 |
| detach_pred_context | 不存在（v4.8 引入） | 作为 init 参数（D5，默认 True） |

### 3.2 三 HyperNetMLP 独立 trunk（与 v4.7 一致）

- 不共用 trunk：hyper_trans / hyper_rew / hyper_pred 各自独立的 Sequential
- 参数量代价：3 × (256² + 256²) = ~400K（在总 ~3.2M 中可接受）
- 收益：三路职责清晰，hyper_rew 可独立配 rew_hidden_dims=(256, 256, 256) 加深以促进 type 分化（v4.7 经验）

### 3.3 output_scale 三初值（C8）

| HyperNet | output_scale_init | 理由 |
|----------|-------------------|------|
| hyper_trans | 0.01 | 标准 small_init（θ_state 由 RepNet+StateTransNet 用，初期接近 0 不影响 transition） |
| hyper_rew | **0.1** | v4.7 关键：避免 init trap（AdaLN modulation 太小时 reward 无法分化）|
| hyper_pred | 0.01 | 标准 small_init |

v4.7 经验：rew=0.01 时 v4.5 训练 100K step 后 hyper_rew 仍未分化（断言 A failed），改 0.1 后立即收敛。Pkg-01 ModelConfig.rew_output_scale_init=0.1 已锁定此值。

### 3.4 detach_pred_context（D5）的实施位置

- 在 `forward_subjective` 内对 ctx_aug 调 `.detach()` 后传给 hyper_pred
- 不影响 hyper_rew（reward loss 仍能反向到 context_encoder）
- 不影响 theta_pred 本身的梯度（θ_pred 仍可反向到 hyper_pred 内部参数）
- 仅切断 value loss → ctx_aug → context_encoder 的路径
- 配置开关在 init 时设定，运行时不变；如需 ablation 创建新 model 实例

### 3.5 与 v4.7 hypnettorch ChunkedHMLP 的关系

v4.7 `models_advanced/chunked_hyper_network.py` 是 ChunkedHMLP 替代实现（vanilla MLP 替代 ChunkedHMLP）。**v4 不实现 ChunkedHMLP**——`hypnettorch` 依赖未在 Pkg-01 引入，本包仅用 PyTorch 原生 nn.Linear。如未来需要可作独立 Pkg-04b。

### 3.6 reward_diversity_loss 的去留

按 D2：v4 type-aware 已通过 hyper_rew 路径隐含实现。reward_diversity_loss 作为**辅助正则**保留：

- `cfg.legacy.w_rew_diversity` 默认 0.0（禁用）
- 仅 Pkg-08 Ablation 5 启用做对照（"加 diversity loss vs 不加"）
- Pkg-01 LegacyConfig 已含该字段

### 3.7 参数量

DualHyperNetwork 内部三个 HyperNetMLP 的精确参数量取决于 `trans_param_count` / `rew_param_count` / `pred_param_count`（由 FunctionalStateTransNet / RewardHead / PredictionNet 的 architecture 决定，spec 03 定义）。详细参数量分解 + 与 Ch4.3.4 ~3.2M 总参数表对照见 [spec 07 §3](./07-forward-performance-budget.md)。

本 spec 不重复数值，避免双源不一致。

---

## 4. Edge Cases

| 场景 | 行为 |
|------|------|
| `c_ctx` shape (B, 8) 但 c_ctx_dim=16 | AssertionError（spec 01 内 assert 触发） |
| `ctx_aug` shape (B, 32) 但 ctx_aug_dim=80 | AssertionError |
| `c_ctx` 含 NaN | hyper_trans 输出含 NaN（不做特殊处理，由调用方保证） |
| L2 norm `norm_output=False` | hyper output 不归一化（仅 small_init * output_scale） |
| `detach_pred_context=False` | hyper_pred 接收 raw ctx_aug，value loss 可反向 context_encoder |
| `rew_hidden_dims=None` | 退化为 `hidden_dims`（与 hyper_trans/pred 同结构） |
| `output_scale_init=0.0` | scale 是 nn.Parameter 可学，0 也合法（但训练初期输出全 0） |
| B=1 单样本 | 正常 forward（trunk MLP 支持任意 batch） |

---

## 5. Acceptance Criteria

### 5.1 单元测试（`tests/models/test_hyper_network_v2.py`）

```python
import pytest
import torch
from hyper_mve.configs import V4Config
from hyper_mve.models import DualHyperNetwork
from hyper_mve.models.hyper_network import HyperNetMLP


@pytest.fixture
def cfg_medium():
    return V4Config.from_preset("medium")


@pytest.fixture
def hyper_net(cfg_medium):
    """构造 DualHyperNetwork (medium config)."""
    return DualHyperNetwork(
        c_ctx_dim=cfg_medium.model.d_c,
        ctx_aug_dim=cfg_medium.model.d_ctx_aug,
        trans_param_count=37_000,
        rew_param_count=4_000,
        pred_param_count=42_000,
        hidden_dims=cfg_medium.model.hyper_hidden_dims,
        rew_hidden_dims=cfg_medium.model.hyper_rew_hidden_dims,
        trans_output_scale_init=cfg_medium.model.trans_output_scale_init,
        rew_output_scale_init=cfg_medium.model.rew_output_scale_init,
        pred_output_scale_init=cfg_medium.model.pred_output_scale_init,
    )


# ====== C1: hyper_trans 仅 c_ctx ======

def test_hyper_trans_only_c_ctx(hyper_net):
    """C1: forward_trans 接 c_ctx (B, 16) → θ_state (B, 37000)."""
    c_ctx = torch.randn(2, 16)
    theta_state = hyper_net.forward_trans(c_ctx)
    assert theta_state.shape == (2, 37_000)


def test_hyper_trans_invariant_under_role_change(hyper_net):
    """C1 关键: 改 role/belief 输入时 forward_trans 输出不变.
    
    构造方法: hyper_trans 接 c_ctx 子段; 改 ctx_aug 的其他部分时
    forward_trans(c_ctx_slice) 输出不变.
    """
    c_ctx_a = torch.randn(2, 16)
    c_ctx_b = c_ctx_a.clone()  # 相同 c_ctx
    
    # 即使 role / belief 不同（在外部由 ctx_aug 拼装），
    # forward_trans 仅看 c_ctx, 输出相同
    theta_a = hyper_net.forward_trans(c_ctx_a)
    theta_b = hyper_net.forward_trans(c_ctx_b)
    assert torch.allclose(theta_a, theta_b)


def test_hyper_trans_c_ctx_dim_assertion(hyper_net):
    """c_ctx shape mismatch 抛 AssertionError."""
    bad_c_ctx = torch.randn(2, 8)  # 错: 8 != 16
    with pytest.raises(AssertionError, match="c_ctx last dim"):
        hyper_net.forward_trans(bad_c_ctx)


# ====== C2: hyper_rew/pred 接 80 维 ======

def test_subjective_input_dim_80(hyper_net):
    """C2: forward_subjective 接 ctx_aug (B, 80) → (θ_rew, θ_pred)."""
    ctx_aug = torch.randn(2, 80)
    theta_rew, theta_pred = hyper_net.forward_subjective(ctx_aug)
    assert theta_rew.shape == (2, 4_000)
    assert theta_pred.shape == (2, 42_000)


def test_subjective_ctx_aug_dim_assertion(hyper_net):
    bad_ctx = torch.randn(2, 64)  # 错: 64 != 80
    with pytest.raises(AssertionError, match="ctx_aug last dim"):
        hyper_net.forward_subjective(bad_ctx)


# ====== C5: ctx_aug_dim == 80 ======

def test_ctx_aug_dim_eq_80(cfg_medium):
    """C5: cfg.model.d_ctx_aug == 80 硬约束 (与 Pkg-01 ModelConfig 一致)."""
    assert cfg_medium.model.d_ctx_aug == 80
    assert cfg_medium.model.d_c == 16
    assert cfg_medium.model.d_role == 32
    assert cfg_medium.model.d_belief == 32


# ====== C8: output_scale 三初值 ======

def test_output_scale_inits_trans_rew_pred(hyper_net):
    """C8: trans=0.01, rew=0.1, pred=0.01."""
    assert torch.isclose(hyper_net.hyper_trans.output_scale, torch.tensor(0.01))
    assert torch.isclose(hyper_net.hyper_rew.output_scale, torch.tensor(0.1))   # v4.7 关键
    assert torch.isclose(hyper_net.hyper_pred.output_scale, torch.tensor(0.01))


# ====== HyperNetMLP 单元 ======

def test_hypernet_mlp_l2_norm():
    """L2 norm: 输出 L2 norm == output_scale (在 norm_output=True 时)."""
    mlp = HyperNetMLP(input_dim=16, output_dim=100, norm_output=True, output_scale_init=0.1)
    x = torch.randn(2, 16)
    out = mlp(x)
    norms = torch.linalg.norm(out, dim=-1)
    expected = torch.full_like(norms, 0.1)  # output_scale
    assert torch.allclose(norms, expected, atol=1e-5)


def test_hypernet_mlp_no_l2_norm():
    """norm_output=False 时不归一化."""
    mlp = HyperNetMLP(input_dim=16, output_dim=100, norm_output=False, output_scale_init=0.01)
    x = torch.randn(2, 16)
    out = mlp(x)
    norms = torch.linalg.norm(out, dim=-1)
    # 不应严格 == output_scale (因为不归一化)
    # 仅断言不全为 output_scale
    assert not torch.allclose(norms, torch.full_like(norms, 0.01), atol=1e-3)


def test_hypernet_mlp_small_init_output_layer():
    """small_init: output_layer weights 标准差 ≈ 0.01."""
    mlp = HyperNetMLP(input_dim=16, output_dim=100, output_scale_init=0.1)
    out_w = mlp.output_layer.weight
    # small_init 标准差 0.01
    assert out_w.std().item() < 0.05  # 容差


# ====== D5: detach_pred_context ======

def test_detach_pred_context_default_true(hyper_net):
    """D5 默认 True."""
    assert hyper_net.detach_pred_context is True


def test_detach_pred_context_blocks_grad_to_ctx_aug(hyper_net):
    """D5: detach_pred_context=True 时, theta_pred 的梯度不回 ctx_aug."""
    ctx_aug = torch.randn(2, 80, requires_grad=True)
    theta_rew, theta_pred = hyper_net.forward_subjective(ctx_aug)
    
    # 仅 theta_pred 的 loss 反向
    (theta_pred ** 2).sum().backward()
    
    # ctx_aug 不应有梯度（被 detach 切断）
    assert ctx_aug.grad is None or ctx_aug.grad.abs().sum().item() == 0.0


def test_detach_pred_context_false_allows_grad():
    """D5: detach_pred_context=False 时, theta_pred 梯度可反向到 ctx_aug."""
    net = DualHyperNetwork(
        c_ctx_dim=16, ctx_aug_dim=80,
        trans_param_count=37_000, rew_param_count=4_000, pred_param_count=42_000,
        detach_pred_context=False,
    )
    ctx_aug = torch.randn(2, 80, requires_grad=True)
    _, theta_pred = net.forward_subjective(ctx_aug)
    (theta_pred ** 2).sum().backward()
    assert ctx_aug.grad is not None
    assert ctx_aug.grad.abs().sum().item() > 0


def test_rew_path_grad_to_ctx_aug(hyper_net):
    """hyper_rew 始终可反向到 ctx_aug (无论 detach_pred_context 如何)."""
    ctx_aug = torch.randn(2, 80, requires_grad=True)
    theta_rew, _ = hyper_net.forward_subjective(ctx_aug)
    (theta_rew ** 2).sum().backward()
    assert ctx_aug.grad is not None
    assert ctx_aug.grad.abs().sum().item() > 0


# ====== 梯度流完整性 ======

def test_gradient_flow_trans(hyper_net):
    """所有 hyper_trans 参数有梯度."""
    c_ctx = torch.randn(2, 16)
    theta = hyper_net.forward_trans(c_ctx)
    (theta ** 2).sum().backward()
    for name, p in hyper_net.hyper_trans.named_parameters():
        assert p.grad is not None, f"No grad for hyper_trans.{name}"
        assert p.grad.abs().sum() > 0, f"Zero grad for hyper_trans.{name}"


def test_gradient_flow_subjective(hyper_net):
    """所有 hyper_rew + hyper_pred 参数有梯度."""
    ctx_aug = torch.randn(2, 80)
    theta_rew, theta_pred = hyper_net.forward_subjective(ctx_aug)
    ((theta_rew ** 2).sum() + (theta_pred ** 2).sum()).backward()
    for name, p in hyper_net.hyper_rew.named_parameters():
        assert p.grad is not None, f"No grad for hyper_rew.{name}"
        assert p.grad.abs().sum() > 0, f"Zero grad for hyper_rew.{name}"
    for name, p in hyper_net.hyper_pred.named_parameters():
        assert p.grad is not None, f"No grad for hyper_pred.{name}"
        assert p.grad.abs().sum() > 0, f"Zero grad for hyper_pred.{name}"
```

### 5.2 性能要求

- `forward_trans(B=256)` 单步 < 5 ms (V100)
- `forward_subjective(B=256)` 单步 < 10 ms (V100, hyper_rew 更深三层)
- HyperNetMLP 参数量取决于 input/output_dim（详 spec 07）

---

## 6. Cross-references

- Ch4.3.2 客观通路 hyper_trans（公式 + C1 物理转移上下文不变）
- Ch4.3.3 主观通路 hyper_rew / hyper_pred 三联输入版
- Ch4.6.1 防线 1 权重初始化（small_init + output_scale）
- Ch4.6.5 belief 梯度门控（与 spec 04 协作；本 spec 不处理 detach 逻辑）
- `02-hyper-muzero-model-v2.md`（HyperMuZeroModel 内调用本类的两个 forward）
- `03-stability-safeguards-preservation.md`（防线 1 详述 + 与 v4.7 数值回归）
- `06-data-flow-diagram.md`（在数据流图中位置）
- `07-forward-performance-budget.md`（参数量分解 + 性能预算）
- `08-integration-contracts.md` §1（API 稳定性表）
- Pkg-01 spec 05 ModelConfig（c_ctx_dim / ctx_aug_dim 字段来源）
- v4.7 `hyper_network.py` L21-82（HyperNetMLP 保留）+ L85-161（DualHyperNetwork v1 重写）

---

## A. [v4-opt 2026-06] 优化阶段修订(就地生效,与正文冲突处以本节为准)

> 动机与实证:FULL 全量生成在 duo 运行中角色坍缩(cos_pred_cross 0.61→0.998),触发部分生成机制族。实现提交:`079fcdf`(film_head + 分组 RMS)、`29e03f9`(base_gen + SubjectiveHyperNet)、`dc5bbcd`(输出层 LoRA + lora_fc2)。代码锚点:`hyper_mve/models/hyper_network.py`。

### A1. HyperNetMLP 接口扩展(修订 §2.2)

```python
HyperNetMLP(input_dim, output_dim, hidden_dims=None, norm_output=True,
            output_scale_init=0.01,
            output_groups=None,   # [新] list[int] 连续段大小,sum==output_dim
            output_rank=None)     # [新] 输出层 LoRA 秩 r
```

- `output_groups=None`(FULL/legacy):整向量 L2 归一,行为与原 spec 逐字节一致;
- `output_groups=[film_total, weight_total]`(部分生成):**分组 RMS 归一**——每段独立 RMS 归一后 × output_scale,每生成元幅值 ≈ scale(维度无关)。理由:整向量 L2 会把单个 FiLM γ 稀释到 scale/√dim(0.1/√512≈4e-3 ⇒ (1+γ)≈1,调制失效);
- `output_rank=r`:输出投影 `Linear(prev, pc)` → `Linear(prev, r, bias=False) → Linear(r, pc)`。初始化纪律:A 正交,**B small_init(std=0.01)且禁止置 0**(B=0 ⇒ raw=0 ⇒ 分组 RMS 的 1e-8 下限在 step-0 产生 ~1e4 梯度尖峰)。

### A2. SubjectiveHyperNet(新增类;修订 §2.3 的"三 MLP 独立"约定)

`share_subjective_trunk=True` 时,hyper_rew/hyper_pred 合并为单 trunk(深度=rew_hidden_dims)+ 双头;每头保留各自 output_scale 与 output_groups。**D5 语义变化**:detach 在 **trunk 输出**处(`h.detach()`),value loss 只训练 pred_head——比非共享的输入 detach 更强。与 `output_rank` 互斥(同开抛 NotImplementedError)。`hyper_trans` 不受影响。

### A3. output_scale 语义修订(修订 §3.3)

分组 RMS 下 scale 含义从"生成向量整体范数"变为"**每生成元幅值**"。部分生成预设(duo 族 / *_lora 族)三路 scale 统一 **0.1**(原 0.01/0.1/0.01 表仅适用于 FULL);`lora_fc2` 下由 ModelConfig 守门断言强制 scale ≥ 0.05(ΔW ≈ scale²·√r 尺度律,spec 03 修订 B3)。

### A4. detach_pred_context 改为 gen_scope 依赖(修订 §3.4 D5)

| gen_scope | 推荐取值 | 理由 |
|---|---|---|
| full | True(原 D5) | 防 value loss 扭曲上下文编码器 |
| film_head / lora_fc2 / base_gen | **False**(duo 族预设) | 功能网主干已是稳定共享 SGD 网络;放开 value 梯度为上下文编码器解饿 |

"运行时不变、ablation 需新建实例"的约定保留。

### A5. 参数量指针更新(修订 §3.7)

spec 07 §3.3 的参数预算以 FULL 估算;实际(medium,+LoRA r=32):film_head ~694k / lora_fc2 ~896k / base_gen ~2.30M / full(无 LoRA) 3.09M。数值由 `tests/models/test_hyper_network_lora.py::test_lora_param_count_formula` 锚定;权威表 = Ch4.3.4 修订表 + DESIGN_DOC §5.12。

## 修订记录 (Changelog)

| 日期 | 修订 | 依据 |
|---|---|---|
| 2026-06-10 | A1-A5:output_groups / output_rank / SubjectiveHyperNet / scale 语义 / D5 gen_scope 化 / 参数量指针 | 优化阶段提交 079fcdf/29e03f9/dc5bbcd;复审 `docs/Review_v4_TheoryAudit_2026-06.md` §3、§5-M10、Q8 决议 |
