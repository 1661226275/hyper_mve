# Spec 01: TriContextEncoder & BeliefEncoder — 三路条件向量组合

> 父文档：[`../proposal.md`](../proposal.md) §1.2 / §3.1 · [`../design.md`](../design.md) §3 D3 / D7 / §6.2
> **本 spec 是包总入口** — 定义 Pkg-04 hyper_rew/hyper_pred 消费的 80 维条件向量。
> **本 spec 含两个模块**（P5 修订）：`BeliefEncoder`（belief 投影 MLP）+ `TriContextEncoder`（三路 concat 协调）。

---

## 1. Purpose

按 Ch4.2.4 把 c_ctx (16) + role_i (32) + belief_i (32) 三路 concat 为 80 维 ctx_i ∈ ℝ^80，作为 DualHyperNetwork v2 hyper_rew / hyper_pred 的统一输入。同时提供 c_ctx 单独取出接口（供 hyper_trans 客观通路使用）。

**两模块分工**（P5 修订）：

| 模块 | 职责 | 与其他 encoder 的关系 |
|------|------|------------------------|
| **`BeliefEncoder`**（`belief_encoder.py`） | 把 raw (c_hat, z_hat) 经 Pool + 投影变为 belief_vec (B, N, 32) | 与 `CEncoder` / `RoleEncoder` 对称，是 belief 通路独立 encoder |
| **`TriContextEncoder`**（`tri_context_encoder.py`） | 调三个 encoder，每路 LN，concat 为 80 维 ctx_i | 协调层，不含投影/池化具体逻辑 |

**为什么拆 BeliefEncoder**（P5 motivation）：
- 三 encoder 对称（c / role / belief 各自有 module 文件）
- 单测粒度更细（belief 投影逻辑独立可测）
- TriContextEncoder 职责单一（仅 concat + LN）

**为什么 BeliefEncoder 不在 BeliefNet 内**（P4 motivation，见 design.md D3 附加）：
- belief gradient gating 需在 raw heads 处切断 .detach()
- 投影 MLP 在 BeliefEncoder（外部模块），main task loss 经投影 MLP 反向传播 → 即使 BeliefNet GRU 暂时不更新，投影 MLP 仍由 main loss 驱动训练

---

## 2. Interface

### 2.1 BeliefEncoder（belief 投影 MLP，P5 拆出）

#### 2.1.1 文件路径

`hyper_mve/models/belief_encoder.py`

#### 2.1.2 类签名

```python
import torch
import torch.nn as nn
from hyper_mve.configs import EnvConfig, ModelConfig
from hyper_mve.models.permutation_invariant_pool import make_pool


class BeliefEncoder(nn.Module):
    """belief 通路投影 MLP (P5 拆出, 与 CEncoder/RoleEncoder 对称).
    
    把 BeliefNet 的 raw heads (c_hat, z_hat) 经 Pool + 投影变为
    belief_vec ∈ ℝ^{d_belief=32}.
    
    架构 (Ch4.2.4):
        c_hat (B, N) → unsqueeze → Linear(1, 16) → ReLU → c_hat_proj (B, N, 16)
        z_hat (B, N, N-1, 2) → Pool (dim=-2) → (B, N, 2)
                                            → Linear(2, 16) → ReLU → z_pooled_proj (B, N, 16)
        belief_vec = Concat[c_hat_proj, z_pooled_proj] = (B, N, 32)
    
    **不含内部 LayerNorm**（P7 修订: LN 责任集中在 TriContextEncoder.ln_belief）.
    
    梯度门控（Ch4.6.5）说明:
        Pkg-04 model 在前 5K step 对 (c_hat, z_hat) 做 .detach() 切断梯度,
        本模块 (BeliefEncoder) 的投影 MLP 仍由 main loss 反向传播训练.
        梯度切断点是 BeliefNet 输出处 (raw heads), 不在本模块内.
    """
    
    def __init__(self, env_cfg: EnvConfig, model_cfg: ModelConfig):
        super().__init__()
        self.env_cfg = env_cfg
        self.model_cfg = model_cfg
        
        self.d_belief = model_cfg.d_belief                      # 32
        self.d_belief_proj = model_cfg.d_belief_proj            # 16
        
        # 维度一致 (与 Pkg-01 ModelConfig __post_init__ 一致)
        assert self.d_belief == 2 * self.d_belief_proj, (
            f"d_belief({self.d_belief}) != 2 * d_belief_proj({self.d_belief_proj})"
        )
        
        # c_hat 投影: (B, N, 1) → (B, N, 16)
        # 无内部 LN (P7: TriContextEncoder.ln_belief 集中归一)
        self.proj_c_hat = nn.Sequential(
            nn.Linear(1, self.d_belief_proj),
            nn.ReLU(),
        )
        
        # z_hat pooling (D2): 默认 mean, 可切换 max/attention
        self.z_pool = make_pool(
            kind=model_cfg.belief_pool,
            feat_dim=2,                                          # softmax 概率 2 类
        )
        
        # Pool 后投影: (B, N, 2) → (B, N, 16)
        self.proj_z_pooled = nn.Sequential(
            nn.Linear(2, self.d_belief_proj),
            nn.ReLU(),
        )
    
    def forward(
        self,
        c_hat: torch.Tensor,                # (B, N) sigmoid ∈ [0, 1]
        z_hat: torch.Tensor,                # (B, N, N-1, 2) softmax
    ) -> torch.Tensor:                      # (B, N, 32) belief_vec
        """组合 c_hat + Pool(z_hat) → belief_vec.
        
        Args:
            c_hat: BeliefNet head_c 输出 (Ch4.2.3 Head 1, sigmoid scalar)
            z_hat: BeliefNet head_opp 输出 (Ch4.2.3 Head 2, softmax 概率)
                   顺序约定: agent_id 升序跳过 self (Pkg-01 z_hat 字段一致)
        Returns:
            belief_vec: (B, N, 32) = Concat[Proj(c_hat), Proj(Pool(z_hat))]
        """
        B, N = c_hat.shape
        assert z_hat.shape[:3] == (B, N, N - 1)
        assert z_hat.shape[3] == 2
        
        # c_hat 子分量 (1 → 16)
        c_hat_proj = self.proj_c_hat(c_hat.unsqueeze(-1))   # (B, N, 16)
        
        # z_hat pooling 在 dim=-2 (N-1 个对手维度) → (B, N, 2)
        z_pooled = self.z_pool(z_hat)
        z_pooled_proj = self.proj_z_pooled(z_pooled)        # (B, N, 16)
        
        belief_vec = torch.cat([c_hat_proj, z_pooled_proj], dim=-1)  # (B, N, 32)
        return belief_vec
```

### 2.2 TriContextEncoder（三路 concat 协调层）

#### 2.2.1 文件路径

`hyper_mve/models/tri_context_encoder.py`

#### 2.2.2 类签名

```python
import torch
import torch.nn as nn
from hyper_mve.configs import EnvConfig, ModelConfig
from hyper_mve.models.c_encoder import CEncoder
from hyper_mve.models.role_encoder import RoleEncoder
from hyper_mve.models.belief_encoder import BeliefEncoder


class TriContextEncoder(nn.Module):
    """v4 三路条件向量组合 (Ch4.2.4).
    
    输出 ctx_i ∈ ℝ^80 = Concat[c_ctx (16), role (32), belief (32)].
    
    职责（P5 修订）:
        本模块仅做 "调三 encoder + 每路 LN + concat" 协调.
        具体投影 / 池化逻辑在各 encoder (BeliefEncoder/RoleEncoder/CEncoder).
    
    每路 LN 设计（D7 + P7 修订）:
        - 每路输出 LayerNorm (ln_c / ln_role / ln_belief), 防止三路量级悬殊
        - 各 sub-encoder 内部**不**含 LN (P7 责任集中)
    """
    
    def __init__(self, env_cfg: EnvConfig, model_cfg: ModelConfig):
        super().__init__()
        self.env_cfg = env_cfg
        self.model_cfg = model_cfg
        
        self.N = env_cfg.N
        self.d_c = model_cfg.d_c                        # 16
        self.d_role = model_cfg.d_role                  # 32
        self.d_belief = model_cfg.d_belief              # 32
        
        # 三路 sub-encoders (各自独立模块, P5)
        self.c_encoder = CEncoder(d_c=self.d_c)
        self.role_encoder = RoleEncoder(
            N=self.N,
            d_id_emb=model_cfg.d_id_emb,        # 8
            d_type_emb=model_cfg.d_type_emb,    # 8
            d_cap_emb=model_cfg.d_cap_emb,      # 16
        )
        self.belief_encoder = BeliefEncoder(env_cfg, model_cfg)
        
        # 三路 LayerNorm (D7: 每路独立 LN, 防止 concat 后量级偏斜)
        # P7: sub-encoder 内部不含 LN, LN 责任集中在此
        self.ln_c = nn.LayerNorm(self.d_c)
        self.ln_role = nn.LayerNorm(self.d_role)
        self.ln_belief = nn.LayerNorm(self.d_belief)
    
    @property
    def d_ctx_aug(self) -> int:
        """ctx_i 总维度 = 16 + 32 + 32 = 80 (Ch4.2.4)."""
        return self.d_c + self.d_role + self.d_belief
    
    def forward(
        self,
        c_t: torch.Tensor,                  # (B,) or (B, 1) float32
        agent_ids: torch.Tensor,            # (B, N) int64
        types: torch.Tensor,                # (B, N) int64 (AgentType.value)
        caps: torch.Tensor,                 # (B, N, 4) float32 RAW CapabilityVector
                                            # (注: role_encoder 内部自动调用 normalize, C 修订)
        belief: tuple[torch.Tensor, torch.Tensor],
                                            # (c_hat: (B, N), z_hat: (B, N, N-1, 2))
    ) -> torch.Tensor:                      # (B, N, 80) float32
        """组合三路 → ctx_i.
        
        三路语义:
            c_ctx (16): batch-level 共享 (所有 N 个 agent 看到同一 c_t, Harsanyi 共同知识)
            role (32):  per-agent (Self-Info: 仅 own type + cap + id)
            belief (32): per-agent (BeliefNet 推断的 ĉ_i 与 ẑ_{i,j})
        """
        B = c_t.shape[0]
        N = agent_ids.shape[1]
        assert agent_ids.shape == (B, N)
        assert types.shape == (B, N)
        assert caps.shape == (B, N, 4)
        c_hat, z_hat = belief
        assert c_hat.shape == (B, N)
        assert z_hat.shape == (B, N, N - 1, 2)
        
        # ====== 1. c_ctx 客观通路 ======
        # c_t shape (B,) or (B, 1) -> (B, 1)
        if c_t.dim() == 1:
            c_t = c_t.unsqueeze(-1)
        c_ctx = self.c_encoder(c_t)                     # (B, 16)
        c_ctx = self.ln_c(c_ctx)                        # 每路 LN
        # broadcast 到 (B, N, 16)
        c_ctx = c_ctx.unsqueeze(1).expand(B, N, self.d_c)
        
        # ====== 2. role 角色通路 ======
        # role_encoder 内部自动 normalize cap (C 修订, 见 spec 03)
        role = self.role_encoder(agent_ids, types, caps)  # (B, N, 32)
        role = self.ln_role(role)
        
        # ====== 3. belief 信念通路 ======
        belief_vec = self.belief_encoder(c_hat, z_hat)    # (B, N, 32)
        belief_vec = self.ln_belief(belief_vec)
        
        # ====== 4. concat 三路 ======
        ctx_i = torch.cat([c_ctx, role, belief_vec], dim=-1)  # (B, N, 80)
        
        return ctx_i
    
    def forward_c_ctx_only(
        self,
        c_t: torch.Tensor,                  # (B,) or (B, 1)
    ) -> torch.Tensor:                      # (B, 16)
        """仅返回 c_ctx (供 hyper_trans 客观通路使用).
        
        hyper_trans 不依赖 N (所有 agent 共享 θ_state),
        故此接口不接 agent_ids / types / caps.
        """
        if c_t.dim() == 1:
            c_t = c_t.unsqueeze(-1)
        c_ctx = self.c_encoder(c_t)
        c_ctx = self.ln_c(c_ctx)
        return c_ctx  # (B, 16)
```

### 2.3 典型用例

```python
# Pkg-04 model.set_context 内的调用
from hyper_mve.models import TriContextEncoder, BeliefEncoder

encoder = TriContextEncoder(cfg.env, cfg.model)

# trainer 在 batch 内调用
B, N = 256, 4
c_t = torch.zeros(B, dtype=torch.float32)
agent_ids = torch.arange(N).unsqueeze(0).expand(B, N)  # (B, N)
types = torch.tensor([[0, 0, 1, 1]] * B, dtype=torch.long)  # 2α+2β
caps = torch.rand(B, N, 4)                                  # RAW cap (η, φ_fov, ν, ζ)

# BeliefNet 输出 (Pkg-05 trainer 已 unroll 完得到当前 step 的)
c_hat = torch.rand(B, N)             # sigmoid scalar
z_hat = torch.softmax(torch.randn(B, N, N-1, 2), dim=-1)

# 主接口
ctx_i = encoder.forward(c_t, agent_ids, types, caps, (c_hat, z_hat))
assert ctx_i.shape == (B, N, 80)

# hyper_trans 单独取
c_ctx = encoder.forward_c_ctx_only(c_t)
assert c_ctx.shape == (B, 16)

# 单独使用 BeliefEncoder (P5 拆出后可独立测试 / 复用)
belief_encoder = BeliefEncoder(cfg.env, cfg.model)
belief_vec = belief_encoder(c_hat, z_hat)
assert belief_vec.shape == (B, N, 32)
```

---

## 3. Implementation Notes

### 3.1 三路 concat 顺序固定 (c_ctx → role → belief)

- 顺序对应 Ch4.2.4 公式 `ctx_i = Concat[c_ctx, role, belief]`
- Pkg-04 hyper_rew/hyper_pred 内部如做切片操作 (如 attention 仅看 belief 子段) 依赖此顺序
- 单测 `test_concat_order` 验证：ctx_i[..., :16] == c_ctx (LN 后)

### 3.2 c_ctx 的 batch-level 共享广播

- c_t 是 batch-level 共享 (所有 N 个 agent 看到同一 c_t, Harsanyi 共同知识)
- c_encoder 输出 (B, 16), 后 broadcast 到 (B, N, 16) (不增加参数, 仅 expand)
- 注意 expand 不复制数据 (PyTorch view 操作), 内存开销 0

### 3.3 BeliefEncoder 内部维度计算

- `c_hat` shape (B, N) → unsqueeze(-1) → (B, N, 1) → proj_c_hat → (B, N, 16)
- `z_hat` shape (B, N, N-1, 2) → z_pool 在 dim=-2 → (B, N, 2) → proj_z_pooled → (B, N, 16)
- concat → (B, N, 32) = d_belief

### 3.4 LayerNorm 参数量（P7 修订后）

- ln_c: 16 × 2 = 32 (γ, β)
- ln_role: 32 × 2 = 64
- ln_belief: 32 × 2 = 64
- BeliefEncoder.proj_c_hat / proj_z_pooled: **不含**内部 LN（P7）
- RoleEncoder.cap_mlp: **不含**内部 LN（P7）
- **总计**：160 params LN，相对模块总 ~5000 params 可忽略

### 3.5 gradient flow 完整性

- 三路所有子模块 + 投影 + LN 均启用梯度
- z_pool (mean) 无可学参数；max 无；attention 含可学 query
- 反向传播路径：loss → ctx_i → [c_ctx, role, belief] 三路独立 → 各 sub-encoder 参数
- 单测 `test_gradient_flow` 验证：随机 (B, N, 80) loss = (ctx_i ** 2).sum(), 反向后所有可学参数有非零梯度

### 3.6 与 v4.6 ContextEncoder 的差异（v4 关键演进）

| 维度 | v4.6 (`context_encoder.py`, 已归档) | v4 (本 spec) |
|------|------------------------------------|---------------|
| 路数 | 2 路 (rule + id) | **3 路 (c_ctx + role + belief)** |
| role 内容 | id_emb only | id + **type_emb** + cap_emb |
| belief | 不存在 | head_c (scalar) + Pool(head_opp (N-1, 2)) |
| 总维度 | ~24-48 (rule_emb_dim 配置) | **80 精确** (16+32+32) |
| LayerNorm | concat 后整体 LN | **每路独立 LN**（P7：sub-encoder 不再含 LN，责任集中） |
| 投影模块组织 | 全部内嵌 ContextEncoder | **BeliefEncoder 独立模块**（P5: 与 c/role 对称） |

---

## 4. Edge Cases

| 场景 | 行为 |
|------|------|
| B=1 单样本 | 正常运行（broadcast 自动处理） |
| N=2 (Easy preset) | z_hat shape (B, 2, 1, 2), z_pool 输出 (B, 2, 2) 正常 |
| N=8 (Hard preset) | z_hat shape (B, 8, 7, 2), 同上 |
| c_t = NaN | 输出含 NaN (不校验，由 trainer 端保证 c_t 有效) |
| z_hat 非 simplex（非 softmax） | 不校验（trainer 端确保 BeliefNet head_opp softmax 输出） |
| types 越界（>= 2） | role_encoder 内 nn.Embedding 越界 IndexError |
| caps shape != (B, N, 4) | role_encoder cap_mlp 形状不匹配 RuntimeError |

---

## 5. Acceptance Criteria

### 5.1 单元测试（`tests/models/test_tri_context.py`）

```python
import pytest
import torch
from hyper_mve.configs import V4Config
from hyper_mve.models import TriContextEncoder, BeliefEncoder


@pytest.fixture
def cfg_medium():
    return V4Config.from_preset("medium")


def test_output_dim(cfg_medium):
    """ctx_i shape (B, N, 80) 精确."""
    encoder = TriContextEncoder(cfg_medium.env, cfg_medium.model)
    B, N = 2, 4
    c_t = torch.rand(B)
    agent_ids = torch.arange(N).unsqueeze(0).expand(B, N).contiguous()
    types = torch.zeros(B, N, dtype=torch.long)
    caps = torch.rand(B, N, 4)
    c_hat = torch.rand(B, N)
    z_hat = torch.softmax(torch.randn(B, N, N-1, 2), dim=-1)
    
    ctx_i = encoder.forward(c_t, agent_ids, types, caps, (c_hat, z_hat))
    assert ctx_i.shape == (B, N, 80)


def test_d_ctx_aug_property(cfg_medium):
    """d_ctx_aug = 16 + 32 + 32 = 80."""
    encoder = TriContextEncoder(cfg_medium.env, cfg_medium.model)
    assert encoder.d_ctx_aug == 80
    assert encoder.d_c == 16
    assert encoder.d_role == 32
    assert encoder.d_belief == 32


def test_role_dim_exact(cfg_medium):
    """role 内部 8+8+16 = 32 精确填满, 无 pad."""
    encoder = TriContextEncoder(cfg_medium.env, cfg_medium.model)
    assert cfg_medium.model.d_id_emb + cfg_medium.model.d_type_emb + cfg_medium.model.d_cap_emb == 32
    assert encoder.role_encoder.d_role == 32


def test_gradient_flow(cfg_medium):
    """反向传播覆盖三路 + 三 head 所有可学参数."""
    encoder = TriContextEncoder(cfg_medium.env, cfg_medium.model)
    B, N = 2, 4
    c_t = torch.rand(B, requires_grad=False)
    agent_ids = torch.arange(N).unsqueeze(0).expand(B, N).contiguous()
    types = torch.zeros(B, N, dtype=torch.long)
    caps = torch.rand(B, N, 4)
    c_hat = torch.rand(B, N, requires_grad=True)
    z_hat = torch.softmax(torch.randn(B, N, N-1, 2, requires_grad=True), dim=-1)
    
    ctx_i = encoder.forward(c_t, agent_ids, types, caps, (c_hat, z_hat))
    loss = (ctx_i ** 2).sum()
    loss.backward()
    
    # 三路 sub-encoder + LN + projections 所有可学参数有梯度
    for name, p in encoder.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No grad for {name}"
            assert p.grad.norm() > 0, f"Zero grad for {name}"


def test_concat_order(cfg_medium):
    """ctx_i 切片顺序: [0:16] c_ctx, [16:48] role, [48:80] belief."""
    encoder = TriContextEncoder(cfg_medium.env, cfg_medium.model)
    B, N = 2, 4
    c_t = torch.rand(B)
    agent_ids = torch.arange(N).unsqueeze(0).expand(B, N).contiguous()
    types = torch.zeros(B, N, dtype=torch.long)
    caps = torch.rand(B, N, 4)
    c_hat = torch.rand(B, N)
    z_hat = torch.softmax(torch.randn(B, N, N-1, 2), dim=-1)
    
    ctx_i = encoder.forward(c_t, agent_ids, types, caps, (c_hat, z_hat))
    
    # c_ctx 子段应在 [0:16]
    c_ctx_only = encoder.forward_c_ctx_only(c_t)        # (B, 16)
    c_ctx_broadcast = c_ctx_only.unsqueeze(1).expand(B, N, 16)
    assert torch.allclose(ctx_i[..., :16], c_ctx_broadcast)


def test_permutation_invariance_belief_mean_pool(cfg_medium):
    """mean pool 下, 改变 z_hat 对手维度顺序, belief 子段不变."""
    cfg = cfg_medium
    encoder = TriContextEncoder(cfg.env, cfg.model)
    B, N = 2, 4
    
    c_t = torch.rand(B)
    agent_ids = torch.arange(N).unsqueeze(0).expand(B, N).contiguous()
    types = torch.zeros(B, N, dtype=torch.long)
    caps = torch.rand(B, N, 4)
    c_hat = torch.rand(B, N)
    z_hat = torch.softmax(torch.randn(B, N, N-1, 2), dim=-1)
    
    ctx_i_orig = encoder.forward(c_t, agent_ids, types, caps, (c_hat, z_hat))
    
    # 打乱 z_hat 在 N-1 维度的顺序 (每个 agent 内独立打乱)
    z_hat_shuffled = z_hat.clone()
    perm = torch.randperm(N - 1)
    z_hat_shuffled = z_hat_shuffled[:, :, perm, :]
    
    ctx_i_shuf = encoder.forward(c_t, agent_ids, types, caps, (c_hat, z_hat_shuffled))
    
    # belief 子段 (ctx_i[..., 48:80]) 应不变 (mean pool 是 set-invariant)
    assert torch.allclose(
        ctx_i_orig[..., 48:80], ctx_i_shuf[..., 48:80], atol=1e-5,
    )


def test_c_ctx_only_dim(cfg_medium):
    encoder = TriContextEncoder(cfg_medium.env, cfg_medium.model)
    c_t = torch.rand(8)
    c_ctx = encoder.forward_c_ctx_only(c_t)
    assert c_ctx.shape == (8, 16)


def test_c_t_input_shapes(cfg_medium):
    """c_t 接受 (B,) 或 (B, 1) 两种形式."""
    encoder = TriContextEncoder(cfg_medium.env, cfg_medium.model)
    B, N = 2, 4
    agent_ids = torch.arange(N).unsqueeze(0).expand(B, N).contiguous()
    types = torch.zeros(B, N, dtype=torch.long)
    caps = torch.rand(B, N, 4)
    c_hat = torch.rand(B, N)
    z_hat = torch.softmax(torch.randn(B, N, N-1, 2), dim=-1)
    
    # (B,)
    c_t_1d = torch.rand(B)
    ctx_1d = encoder.forward(c_t_1d, agent_ids, types, caps, (c_hat, z_hat))
    
    # (B, 1)
    c_t_2d = c_t_1d.unsqueeze(-1)
    ctx_2d = encoder.forward(c_t_2d, agent_ids, types, caps, (c_hat, z_hat))
    
    assert torch.allclose(ctx_1d, ctx_2d)


def test_n_variation(cfg_medium):
    """N=2 (Easy) 与 N=8 (Hard) 均应可运行."""
    from dataclasses import replace
    from hyper_mve.schemas import AgentType
    
    # N=2
    cfg2 = replace(cfg_medium, env=replace(cfg_medium.env, N=2, 
        type_assignment=(AgentType.ALPHA, AgentType.BETA)))
    enc2 = TriContextEncoder(cfg2.env, cfg2.model)
    c_t = torch.rand(1)
    aid = torch.tensor([[0, 1]])
    typ = torch.tensor([[0, 1]])
    cap = torch.rand(1, 2, 4)
    c_h = torch.rand(1, 2)
    z_h = torch.softmax(torch.randn(1, 2, 1, 2), dim=-1)
    ctx2 = enc2.forward(c_t, aid, typ, cap, (c_h, z_h))
    assert ctx2.shape == (1, 2, 80)
    
    # N=8
    cfg8 = replace(cfg_medium, env=replace(cfg_medium.env, N=8,
        type_assignment=(AgentType.ALPHA,)*4 + (AgentType.BETA,)*4))
    enc8 = TriContextEncoder(cfg8.env, cfg8.model)
    aid8 = torch.arange(8).unsqueeze(0)
    typ8 = torch.zeros(1, 8, dtype=torch.long)
    cap8 = torch.rand(1, 8, 4)
    c_h8 = torch.rand(1, 8)
    z_h8 = torch.softmax(torch.randn(1, 8, 7, 2), dim=-1)
    ctx8 = enc8.forward(c_t, aid8, typ8, cap8, (c_h8, z_h8))
    assert ctx8.shape == (1, 8, 80)


# ====== BeliefEncoder 独立单测 (P5 拆出后可独立测试) ======

def test_belief_encoder_output_shape(cfg_medium):
    """BeliefEncoder 输出 (B, N, 32)."""
    be = BeliefEncoder(cfg_medium.env, cfg_medium.model)
    B, N = 2, 4
    c_hat = torch.rand(B, N)
    z_hat = torch.softmax(torch.randn(B, N, N-1, 2), dim=-1)
    
    out = be(c_hat, z_hat)
    assert out.shape == (B, N, 32)


def test_belief_encoder_no_internal_ln(cfg_medium):
    """P7: BeliefEncoder 不含内部 LayerNorm."""
    be = BeliefEncoder(cfg_medium.env, cfg_medium.model)
    has_ln = any(isinstance(m, torch.nn.LayerNorm) for m in be.modules())
    assert not has_ln, "BeliefEncoder should NOT contain internal LayerNorm (P7)"


def test_belief_encoder_gradient_flow(cfg_medium):
    be = BeliefEncoder(cfg_medium.env, cfg_medium.model)
    B, N = 2, 4
    c_hat = torch.rand(B, N, requires_grad=True)
    z_hat = torch.softmax(torch.randn(B, N, N-1, 2, requires_grad=True), dim=-1)
    
    out = be(c_hat, z_hat)
    loss = (out ** 2).sum()
    loss.backward()
    
    for name, p in be.named_parameters():
        assert p.grad is not None, f"No grad for {name}"
        assert p.grad.norm() > 0, f"Zero grad for {name}"


def test_belief_encoder_concat_order(cfg_medium):
    """belief_vec[..., :16] 来自 proj_c_hat, [..., 16:32] 来自 proj_z_pooled."""
    be = BeliefEncoder(cfg_medium.env, cfg_medium.model)
    B, N = 1, 4
    
    # 给一个固定输入
    c_hat = torch.rand(B, N)
    z_hat = torch.softmax(torch.randn(B, N, N-1, 2), dim=-1)
    
    out = be(c_hat, z_hat)
    
    # 单独走 proj_c_hat 路径验证
    c_proj_alone = be.proj_c_hat(c_hat.unsqueeze(-1))
    assert torch.allclose(out[..., :16], c_proj_alone)
```

### 5.2 性能要求

- `encoder.forward(B=256, N=4)` 单步 < 5 ms (V100 GPU)
- `encoder.forward_c_ctx_only(B=256)` 单步 < 1 ms
- 参数量 ≤ 10K (sub-encoders 共 ~5K + LN/projections ~2K)

### 5.3 集成测试

集成在 `scripts/test_belief_net_synth.py` 中（合成数据 5K 步训练 BeliefNet + TriContextEncoder + 简化 hyper_rew 模拟下游），验证：
- ctx_i 在训练过程中无 NaN / Inf
- ctx_i 各路子段 mean / std 稳定（不发散）
- belief 子段（[48:80]）随 BeliefNet 训练逐渐变化（早期接近 0 噪声，后期反映学到的 ĉ / ẑ）

---

## 6. Cross-references

- Ch4.2.4 三通路汇总（公式 + 默认维度）
- Ch4.3.3 hyper_rew/hyper_pred 输入 80 维 ctx_i
- Ch4.6.5 belief 梯度门控（与 BeliefEncoder/BeliefNet 分层设计的关系）
- `02-c-encoder.md`（c_ctx 内部实现）
- `03-role-encoder.md`（role_i 内部实现，含 v4 type_emb + C 修订 normalize 集成）
- `04-belief-net-gru.md`（c_hat / z_hat 来源）
- `05-belief-heads.md`（head_opp 输出 (B, N, N-1, 2) softmax）
- `07-permutation-invariant-pool.md`（z_hat 池化）
- `08-integration-contracts.md`（Pkg-04 消费契约 + 梯度门控分层协作）
- Pkg-01 `05-v4-config-structure.md` ModelConfig.d_ctx_aug property
- Pkg-04 spec `01-dualhypernet-v2-api.md`（forward 输入接口）
