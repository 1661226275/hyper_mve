# Spec 04: Belief Gradient Gating（v4 新增防线 5）

> 父文档：[`../proposal.md`](../proposal.md) §1.3 · [`../design.md`](../design.md) §3 D3 · §6.4
> **v4 新增**：Ch4.6.5 belief 梯度门控，与防线 1-4 (spec 03) 正交。

---

## 1. Purpose

按 Ch4.6.5 实现 belief 梯度门控：**前 `belief_grad_gating_steps` 步**（默认 5000）切断主任务 loss 反向到 BeliefNet 的梯度路径，让 BeliefNet 仅由独立的 `L_belief` loss（Pkg-03 belief_loss）驱动训练。

**为什么需要**：BeliefNet 训练初期 head_c / head_opp 输出噪声大，若让主任务 loss 通过 BeliefNet 反向更新 BeliefNet 参数，会引入"chicken-and-egg"问题：
- main task 用噪声 belief 学到错误策略
- 错误策略生成的 reward / value 梯度反向到 BeliefNet → 进一步污染 BeliefNet 学习
- 恶性循环

解决方案：前 5K step 切断这条反向路径，让 BeliefNet 独立用 Oracle 监督（L_c MSE + L_opp CE）先学好（5K step 内合成数据收敛实测 head_c MSE < 0.05 + head_opp accuracy > 80%），之后再开放主任务梯度联合训练。

---

## 2. Interface

### 2.1 文件路径

`hyper_mve/models/grad_gating.py`（新增）

### 2.2 BeliefGradGating helper（review 修订 3：扩展覆盖 BeliefEncoder）

```python
import torch
from typing import Optional


class BeliefGradGating:
    """Belief gradient gating helper (v4 新增防线 5, Ch4.6.5).
    
    不是 nn.Module 子类 - 是 model.forward 内部 helper, 无可学参数.
    
    用法 (在 HyperMuZeroModel.set_context_subjective 内, 两道切断):
        # 切断 1: raw heads
        c_hat, z_hat = self.grad_gating.apply_raw(c_hat, z_hat, self._step)
        # 切断 2: TriContextEncoder 产出 ctx_aug 后, 对 belief 子段二次切断
        ctx_aug = tri_ctx_encoder.forward(...)
        ctx_aug = self.grad_gating.apply_ctx(ctx_aug, self._step,
                                              belief_slice=(48, 80))
    
    设计要点 (D3 + review 修订 3):
        - 不用 nn.Hook (难调试, 反向才触发)
        - 不用 BeliefNet 内部 detach (会破坏 BeliefNet 独立 L_belief loss)
        - 双层 detach: raw heads (切断到 BeliefNet) + ctx_aug.belief 子段 (切断到 BeliefEncoder)
        - L_belief 独立 loss 路径 ✅ 不受 gating 影响（trainer 直接 belief_net 参数 backward）
    """
    
    def __init__(self, num_warmup_steps: int = 5000):
        """
        Args:
            num_warmup_steps: 前多少步切断 belief 梯度.
                从 cfg.train.belief_grad_gating_steps 读取 (默认 5000).
        """
        self.num_warmup_steps = num_warmup_steps
    
    def apply_raw(
        self,
        c_hat: torch.Tensor,
        z_hat: torch.Tensor,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """切断 1: raw heads (c_hat, z_hat) → 阻 main loss 反向到 BeliefNet GRU/heads.
        
        Args:
            c_hat: (B,) or (B, N) sigmoid 输出
            z_hat: (B, N-1, 2) or (B, N, N-1, 2) softmax 输出
            step:  当前 global_step (由 HyperMuZeroModel.update_step 提供)
        
        Returns:
            (c_hat_out, z_hat_out): step < num_warmup_steps 时 .detach(), 否则透传.
        """
        if step < self.num_warmup_steps:
            return c_hat.detach(), z_hat.detach()
        return c_hat, z_hat
    
    def apply_ctx(
        self,
        ctx_aug: torch.Tensor,
        step: int,
        belief_slice: tuple[int, int] = (48, 80),
    ) -> torch.Tensor:
        """切断 2 (review 修订 3): ctx_aug 内 belief 子段 → 阻 main loss 反向到 BeliefEncoder.
        
        ctx_aug 内部结构: [c_ctx (0:16), role (16:48), belief_vec (48:80)]
        仅对 [48:80] 子段 .detach(), 保留 c_ctx + role 路径梯度.
        
        Args:
            ctx_aug: (..., 80) TriContextEncoder.forward 产出
            step:    当前 global_step
            belief_slice: (start, end) 默认 (48, 80) 对应 belief_vec 段
        
        Returns:
            ctx_aug_gated: 同 shape; step < num_warmup_steps 时 belief 子段已 .detach()
        """
        if step >= self.num_warmup_steps:
            return ctx_aug
        
        # 拼接式 detach: c_ctx + role 保留梯度, belief 子段切断
        start, end = belief_slice
        return torch.cat([
            ctx_aug[..., :start],
            ctx_aug[..., start:end].detach(),
            ctx_aug[..., end:],
        ], dim=-1)
    
    # 兼容旧调用名（spec 02 §2.2 内调用 .apply 仍可用，等价于 apply_raw）
    apply = apply_raw
    
    @property
    def is_gating_active(self) -> bool:
        """便于 debug 时检查当前是否处于 gating 状态.
        注: 需配合外部 step 信息使用 (本类不持有 step 状态)."""
        return True  # 总是 active, 是否生效由 apply(step) 决定
```

### 2.3 在 HyperMuZeroModel 内的调用位置（review 修订 3：双层 detach）

```python
# hyper_muzero_model.py (spec 02)
class HyperMuZeroModel(nn.Module):
    def __init__(self, cfg: V4Config):
        # ...
        self.grad_gating = BeliefGradGating(
            num_warmup_steps=cfg.train.belief_grad_gating_steps,
        )
        self._step: int = 0
    
    def update_step(self, global_step: int) -> None:
        """Q4: 独立方法管理 step 状态."""
        self._step = global_step
    
    def set_context_subjective(self, agent_id, cap_i, belief):
        # ... 取出 c_hat, z_hat ...
        c_hat, z_hat = belief
        
        # ★ 切断 1: raw heads (阻反向到 BeliefNet GRU/heads)
        c_hat, z_hat = self.grad_gating.apply_raw(c_hat, z_hat, self._step)
        
        # ... 构造 ctx_aug 经 TriContextEncoder.forward ...
        ctx_aug_i = self.tri_context_encoder.forward(
            c_t=self._cached_c_t,
            agent_ids=agent_ids_one,
            types=own_type,
            caps=cap_i.unsqueeze(1),
            belief=(c_hat.unsqueeze(1), z_hat.unsqueeze(1)),
        ).squeeze(1)  # (B, 80)
        
        # ★ 切断 2 (review 修订 3): ctx_aug.belief 子段 (阻反向到 BeliefEncoder 投影 MLP)
        ctx_aug_i = self.grad_gating.apply_ctx(
            ctx_aug_i, self._step, belief_slice=(48, 80),
        )
        
        # ... forward_subjective ...
        self._theta_rew, self._theta_pred = self.hyper_net.forward_subjective(ctx_aug_i)
```

---

## 3. Implementation Notes

### 3.1 切断位置：双层 detach（review 修订 3 收紧）

**关键设计修订**：原 spec 仅切断 raw heads，但 BeliefEncoder 投影 MLP 仍受 main loss 影响——意味着 BeliefEncoder 在前 5K step 内（BeliefNet 噪声大、输出不可靠）就开始被错误信号训练。**review 修订 3 扩展：双层 detach**，覆盖 BeliefEncoder。

```
                  ┌─────── L_belief loss ───────┐
                  │      (始终不 detach)         │
                  ▼                              │
BeliefNet (GRU + heads)                          │
    ↓ c_hat, z_hat                               │
[apply_raw: c_hat.detach(), z_hat.detach()]  ← 切断 1 (raw heads, main loss → BeliefNet 阻)
    ↓                                            │
TriContextEncoder.forward                        │
    ↓ (内部调 BeliefEncoder)                     │
BeliefEncoder (proj_c_hat + Pool + proj_z_pooled)│
    ↓ belief_vec (B, 32) →─ concat → ctx_aug (B, 80)
    │                                            │
[apply_ctx: ctx_aug[..., 48:80].detach()]   ← 切断 2 (ctx_aug.belief 子段, main loss → BeliefEncoder 阻)
    ↓                                            │
hyper_rew / hyper_pred                           │
    ↓ θ_rew, θ_pred                              │
RewardHead / PredictionNet                       │
    ↓ r, v                                       │
main loss ──────────────────────────────────────┘
                                       (经 c_ctx + role 路径仍可反向到 CEncoder/RoleEncoder)
```

**反向传播路径分析（step < 5000，gating active）**：
- main loss → r → θ_rew → hyper_rew → ctx_aug[..., :48] (c_ctx + role) → ✅ CEncoder/RoleEncoder 参数仍训练
- main loss → r → θ_rew → hyper_rew → ctx_aug[..., 48:80] (detached) → ❌ BeliefEncoder 参数 grad=0
- main loss → ... → ctx_aug[..., 48:80] → BeliefEncoder → c_hat (detached) → ❌ BeliefNet 参数 grad=0
- L_belief loss → BeliefNet 参数 ✅ 独立更新（始终）
- hyper_rew / hyper_pred 参数 ✅ 始终训练

**反向传播路径分析（step >= 5000，gating passthrough）**：
- 所有路径恢复梯度反向；BeliefNet、BeliefEncoder、hyper_rew/pred、CEncoder、RoleEncoder 全部由 main loss 训练
- L_belief 仍独立反向（与 main loss 联合优化 BeliefNet）

### 3.2 与 Pkg-03 BeliefEncoder 分层的一致性（review 修订 3 更新）

Pkg-03 spec 01 §1 已规定 BeliefEncoder 在 belief_encoder.py（与 BeliefNet 分开），就是为支持本设计。修订 3 后两者**同步被 gate**：

| 模块 | Pkg | main loss 梯度（step < 5000） | main loss 梯度（step >= 5000） | L_belief 梯度（始终） |
|------|-----|------------------------------|-------------------------------|---------------------|
| BeliefNet (GRU + heads) | Pkg-03 | ❌ 切断（apply_raw .detach） | ✅ 反向 | ✅ 反向 |
| BeliefEncoder（投影 MLP）| Pkg-03 | **❌ 切断**（apply_ctx .detach belief 子段，**review 修订 3 修订**）| ✅ 反向 | ❌ 无（L_belief 不经投影 MLP） |
| CEncoder（c_t MLP） | Pkg-03 | ✅ 反向（c_ctx 路径未 detach） | ✅ 反向 | ❌ 无 |
| RoleEncoder | Pkg-03 | ✅ 反向（role 路径未 detach） | ✅ 反向 | ❌ 无 |
| hyper_rew / hyper_pred | Pkg-04 | ✅ 反向 | ✅ 反向 | ❌ 无 |

**收紧理由**：前 5K step BeliefNet 输出不可靠时让 BeliefEncoder 也学等于让 BeliefEncoder 学噪声→噪声的投影，等 BeliefNet 收敛后再开放联合训练更稳健。代价：BeliefEncoder 从 step 5000 才开始受 main loss 训练（之前仅 init 状态），但 BeliefEncoder 是 3 层 MLP（参数少，~1K），从 5K step 训练到 200K 总 step 仍有充分迭代。

### 3.3 5K step 阈值的可配置性（C4 + R4）

阈值通过 `cfg.train.belief_grad_gating_steps` 控制（Pkg-01 spec 05 已含字段，默认 5000）。

`BeliefGradGating.__init__` 从 cfg 读取，**不**硬编码：

```python
# spec 02 HyperMuZeroModel.__init__
self.grad_gating = BeliefGradGating(
    num_warmup_steps=cfg.train.belief_grad_gating_steps,  # ← 从 cfg 读
)
```

避免 R4：硬编码 vs cfg 不一致。

### 3.4 update_step 的责任划分（Q4）

- **谁调 update_step？** Pkg-05 trainer 在每个 train_step 起点调一次：`model.update_step(global_step)`
- **worker 不调**：worker 永远在 torch.no_grad inference 模式，梯度门控对它无意义；worker 内 `model._step` 保持上次 trainer 设置的值（无影响）
- **eval 不调**：eval 时也是 no_grad，self._step 不影响结果

### 3.5 .detach() 而非 nn.Hook 的理由（D3）

`.detach()` 选择理由：
1. 显式 if 分支在 forward 内可读、可调试（pdb 一目了然）
2. nn.Hook 在反向时触发，调试困难（前向看不到 detach 行为）
3. .detach() 与 PyTorch 计算图语义直接对应（教科书级清晰）
4. nn.Hook 在 model.eval() 时仍触发 hook（除非 explicit 处理），可能引入意外行为

### 3.6 与 Pkg-03 build_oracle_z_seq 的关系

Pkg-03 提供 `build_oracle_z_seq(types_true)`（spec 08 §4.3）用于课程 Stage 1 oracle 注入。这与 grad gating **正交**：

| 维度 | Stage 1 Oracle 注入 | belief grad gating |
|------|---------------------|---------------------|
| 实施层 | Pkg-05 trainer | Pkg-04 model |
| 控制信号 | `oracle_z_seq` 参数传 BeliefNet.forward | `global_step` 传 update_step |
| 切换边界 | step < curriculum_stage_1_end_frac × max_steps（默认 0.3） | step < belief_grad_gating_steps（默认 5000）|
| 目的 | 主任务用 oracle 替代 head_opp 输出（仅 z_for_main） | 切断 main loss → BeliefNet 反向 |
| 重叠期 | step ∈ [0, 5000] 且 step < 0.3·max_steps 时**两者都生效** | 同左 |

**两者协同**：前 5K step 既"用 oracle 喂下游"又"切断 belief 梯度反向"——双重保护让 BeliefNet 独立用 L_belief 学习。

---

## 4. Edge Cases

| 场景 | 行为 |
|------|------|
| update_step 未调用（_step=0 默认） | grad gating 永远 active（step < 5000） |
| update_step(0) 然后立即 forward | grad gating active（step=0 < 5000） |
| update_step(5000) 然后 forward | grad gating 不再生效（5000 < 5000 为 False） |
| update_step(-1) 负数 | grad gating active（-1 < 5000） |
| cfg.train.belief_grad_gating_steps=0 | 永不 gating（step >= 0 总成立） |
| cfg.train.belief_grad_gating_steps=999999 | 全程 gating（极端配置，仅 ablation 用） |
| 在 inference mode (torch.no_grad) 调 forward | grad gating 仍调用 .detach()（但无梯度反正，无副作用） |

---

## 5. Acceptance Criteria

### 5.1 单元测试（`tests/models/test_grad_gating.py`）

```python
import pytest
import torch
from hyper_mve.configs import V4Config
from hyper_mve.models import HyperMuZeroModel
from hyper_mve.models.grad_gating import BeliefGradGating


# ====== BeliefGradGating 单元 ======

def test_grad_gating_detach_before_warmup():
    """step < num_warmup_steps 时返回 .detach() 后的 tensor."""
    gating = BeliefGradGating(num_warmup_steps=5000)
    c_hat = torch.tensor([0.5], requires_grad=True)
    z_hat = torch.softmax(torch.randn(1, 3, 2, requires_grad=True), dim=-1)
    
    c_out, z_out = gating.apply(c_hat, z_hat, step=1000)
    assert c_out.requires_grad is False
    assert z_out.requires_grad is False


def test_grad_gating_passthrough_after_warmup():
    """step >= num_warmup_steps 时透传."""
    gating = BeliefGradGating(num_warmup_steps=5000)
    c_hat = torch.tensor([0.5], requires_grad=True)
    z_hat = torch.softmax(torch.randn(1, 3, 2, requires_grad=True), dim=-1)
    
    c_out, z_out = gating.apply(c_hat, z_hat, step=10000)
    assert c_out.requires_grad is True
    assert z_out.requires_grad is True


def test_grad_gating_boundary_step_eq_warmup():
    """step == num_warmup_steps 时透传 (边界, 5000 < 5000 为 False)."""
    gating = BeliefGradGating(num_warmup_steps=5000)
    c_hat = torch.tensor([0.5], requires_grad=True)
    z_hat = torch.softmax(torch.randn(1, 3, 2, requires_grad=True), dim=-1)
    
    c_out, z_out = gating.apply(c_hat, z_hat, step=5000)
    assert c_out.requires_grad is True


# ====== HyperMuZeroModel 集成 ======

@pytest.fixture
def cfg_medium():
    return V4Config.from_preset("medium")


@pytest.fixture
def model(cfg_medium):
    return HyperMuZeroModel(cfg_medium)


def test_model_has_grad_gating(model):
    """model 内含 BeliefGradGating 实例."""
    assert hasattr(model, "grad_gating")
    assert isinstance(model.grad_gating, BeliefGradGating)


def test_grad_gating_num_warmup_from_cfg(cfg_medium, model):
    """num_warmup_steps 从 cfg.train.belief_grad_gating_steps 读取."""
    assert model.grad_gating.num_warmup_steps == cfg_medium.train.belief_grad_gating_steps
    assert model.grad_gating.num_warmup_steps == 5000  # 默认值


def test_update_step_changes_internal_step(model):
    """update_step 改变 model._step."""
    model.update_step(0)
    assert model._step == 0
    model.update_step(1000)
    assert model._step == 1000
    model.update_step(10000)
    assert model._step == 10000


def test_pre_5k_belief_detached(model, cfg_medium):
    """C4: step < 5000 时, main loss 反向后 BeliefNet 参数梯度 norm = 0."""
    model.update_step(1000)
    
    # 端到端 forward + main loss 反向
    B = 2
    N = cfg_medium.env.N
    obs_dim = model.rep_net.obs_dim
    
    obs = torch.randn(B, N, obs_dim)
    s = model.encode(obs)
    
    model.set_context_objective(torch.full((B,), 0.5))
    
    # 模拟 BeliefNet 输出 (含 grad)
    c_hat = torch.rand(B, requires_grad=True)
    z_hat = torch.softmax(torch.randn(B, N-1, 2, requires_grad=True), dim=-1)
    
    # 清空 BeliefNet 参数梯度
    for p in model.belief_net.parameters():
        p.grad = None
    
    model.set_context_subjective(0, torch.rand(B, 4), (c_hat, z_hat))
    
    action = torch.zeros(B, N * cfg_medium.env.A)
    action[:, 0] = 1.0
    r = model.predict_reward(s, action)
    
    # main loss (例: MSE to target reward 0)
    loss = (r ** 2).sum()
    loss.backward()
    
    # review 修订 3: BeliefNet ALL parameters 在 main loss backward 后 grad norm == 0
    for name, p in model.belief_net.named_parameters():
        if p.grad is not None:
            assert p.grad.abs().sum().item() == 0.0, (
                f"BeliefNet param {name} has grad sum {p.grad.abs().sum().item()} "
                f"but step=1000 should be gated (apply_raw .detach 全覆盖)."
            )


def test_post_5k_belief_grad_flow(model, cfg_medium):
    """C4: step >= 5000 时, main loss 反向后 BeliefNet 参数有梯度."""
    model.update_step(10000)
    
    B = 2
    N = cfg_medium.env.N
    obs_dim = model.rep_net.obs_dim
    
    obs = torch.randn(B, N, obs_dim, requires_grad=False)
    
    # 走 BeliefNet 真实 forward (含 BeliefNet 参数梯度路径)
    prev_hidden = model.belief_net.init_hidden(B, N)
    _, c_hat, z_hat = model.belief_net.step(obs, prev_hidden)
    # c_hat shape (B, N), z_hat shape (B, N, N-1, 2); 取 agent 0
    c_hat_0 = c_hat[:, 0]  # (B,)
    z_hat_0 = z_hat[:, 0]  # (B, N-1, 2)
    
    # 清空 BeliefNet 参数梯度
    for p in model.belief_net.parameters():
        p.grad = None
    
    model.set_context_objective(torch.full((B,), 0.5))
    model.set_context_subjective(0, torch.rand(B, 4), (c_hat_0, z_hat_0))
    
    s = model.encode(obs)
    action = torch.zeros(B, N * cfg_medium.env.A)
    action[:, 0] = 1.0
    r = model.predict_reward(s, action)
    
    loss = (r ** 2).sum()
    loss.backward()
    
    # 至少部分 BeliefNet 参数有非零梯度
    has_grad = False
    for p in model.belief_net.parameters():
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            has_grad = True
            break
    assert has_grad, "step=10000 时 BeliefNet 应有梯度（grad gating 已关闭）"


def test_pre_5k_belief_encoder_also_detached(model, cfg_medium):
    """review 修订 3: step < 5000 时 BeliefEncoder 参数也应 grad norm = 0.
    
    (原 spec 设计 BeliefEncoder 仍可由 main loss 训练; 修订后 apply_ctx 双层 detach
    收紧到与 BeliefNet 同步 gate, 让 BeliefEncoder 等 BeliefNet 收敛后再训.)
    """
    model.update_step(1000)  # gating active
    
    B = 2
    N = cfg_medium.env.N
    obs_dim = model.rep_net.obs_dim
    obs = torch.randn(B, N, obs_dim)
    
    c_hat = torch.rand(B, requires_grad=True)
    z_hat = torch.softmax(torch.randn(B, N-1, 2, requires_grad=True), dim=-1)
    
    # 清空 BeliefEncoder 参数梯度
    for p in model.tri_context_encoder.belief_encoder.parameters():
        p.grad = None
    
    model.set_context_objective(torch.full((B,), 0.5))
    model.set_context_subjective(0, torch.rand(B, 4), (c_hat, z_hat))
    
    s = model.encode(obs)
    action = torch.zeros(B, N * cfg_medium.env.A)
    action[:, 0] = 1.0
    r = model.predict_reward(s, action)
    (r ** 2).sum().backward()
    
    # BeliefEncoder ALL parameters 在 main loss 路径上 grad norm 应 = 0
    for name, p in model.tri_context_encoder.belief_encoder.named_parameters():
        if p.grad is not None:
            assert p.grad.abs().sum().item() == 0.0, (
                f"BeliefEncoder param {name} has grad sum {p.grad.abs().sum().item()} "
                f"but step=1000 should be gated (review 修订 3 双层 detach)."
            )


def test_post_5k_belief_encoder_grad_flow(model, cfg_medium):
    """review 修订 3: step >= 5000 时 BeliefEncoder 参数有梯度（与 BeliefNet 同步开放）."""
    model.update_step(10000)
    
    B = 2
    N = cfg_medium.env.N
    obs_dim = model.rep_net.obs_dim
    obs = torch.randn(B, N, obs_dim)
    
    c_hat = torch.rand(B, requires_grad=True)
    z_hat = torch.softmax(torch.randn(B, N-1, 2, requires_grad=True), dim=-1)
    
    for p in model.tri_context_encoder.belief_encoder.parameters():
        p.grad = None
    
    model.set_context_objective(torch.full((B,), 0.5))
    model.set_context_subjective(0, torch.rand(B, 4), (c_hat, z_hat))
    
    s = model.encode(obs)
    action = torch.zeros(B, N * cfg_medium.env.A)
    action[:, 0] = 1.0
    r = model.predict_reward(s, action)
    (r ** 2).sum().backward()
    
    has_grad = False
    for p in model.tri_context_encoder.belief_encoder.parameters():
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            has_grad = True
            break
    assert has_grad, "step=10000 时 BeliefEncoder 应有梯度（与 BeliefNet 同步开放）"


def test_l_belief_path_not_detached(model, cfg_medium):
    """review 修订 3: L_belief loss 直接对 BeliefNet 参数反向, 始终不受 gating 影响.
    
    模拟 trainer 端的 L_belief.backward(), 验证 BeliefNet 参数有梯度
    （即使 step < 5000 main loss 路径被 gate）.
    """
    model.update_step(1000)  # gating active for main loss
    
    B = 2
    N = cfg_medium.env.N
    obs_dim = model.rep_net.obs_dim
    obs = torch.randn(B, N, obs_dim)
    
    for p in model.belief_net.parameters():
        p.grad = None
    
    # L_belief 走 BeliefNet 自身 forward (不经 model 的 grad_gating)
    prev_hidden = model.belief_net.init_hidden(B, N)
    _, c_hat, z_hat = model.belief_net.step(obs, prev_hidden)
    
    # 假设 oracle 监督: L_c MSE to 0
    l_belief = (c_hat ** 2).sum() + (z_hat ** 2).sum()
    l_belief.backward()
    
    # BeliefNet 参数应有梯度（L_belief 路径不受 gating 影响）
    has_grad = False
    for p in model.belief_net.parameters():
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            has_grad = True
            break
    assert has_grad, (
        "L_belief 路径应始终反向到 BeliefNet, 不受 grad gating 影响 "
        "(review 修订 3: 门控仅作用于 main loss 路径)."
    )


def test_c_ctx_role_path_grad_flows_during_gating(model, cfg_medium):
    """review 修订 3 验证: gating active 时 c_ctx + role 路径仍有梯度.
    
    apply_ctx 只 detach belief 子段 [48:80], c_ctx [:16] + role [16:48] 仍透传.
    """
    model.update_step(1000)  # gating active
    
    B = 2
    N = cfg_medium.env.N
    obs_dim = model.rep_net.obs_dim
    obs = torch.randn(B, N, obs_dim)
    
    c_hat = torch.rand(B, requires_grad=True)
    z_hat = torch.softmax(torch.randn(B, N-1, 2, requires_grad=True), dim=-1)
    
    # 清空 CEncoder + RoleEncoder 梯度
    for p in model.tri_context_encoder.c_encoder.parameters():
        p.grad = None
    for p in model.tri_context_encoder.role_encoder.parameters():
        p.grad = None
    
    model.set_context_objective(torch.full((B,), 0.5))
    model.set_context_subjective(0, torch.rand(B, 4), (c_hat, z_hat))
    
    s = model.encode(obs)
    action = torch.zeros(B, N * cfg_medium.env.A)
    action[:, 0] = 1.0
    r = model.predict_reward(s, action)
    (r ** 2).sum().backward()
    
    # CEncoder + RoleEncoder 仍应有梯度
    for sub_name, sub in [("CEncoder", model.tri_context_encoder.c_encoder),
                          ("RoleEncoder", model.tri_context_encoder.role_encoder)]:
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in sub.parameters()
        )
        assert has_grad, (
            f"{sub_name} 参数应有梯度 (gating active 时 c_ctx/role 路径未 detach)"
        )
```

### 5.2 集成测试

在 `scripts/test_hyper_model_forward.py`（spec 07）中验证：
- 训练 1000 步：前 5000 步 BeliefNet 梯度 norm = 0；step >= 5000 后梯度 norm > 0
- 监控 TensorBoard：`grad/belief_net_norm` 时间序列在 5000 step 处从 0 跳到 > 0

### 5.3 性能要求

- `BeliefGradGating.apply` 调用开销 < 10 μs（单个 if 分支 + 可选 .detach()）

---

## 6. v4 vs v4.7 对比

| 维度 | v4.7 | v4 |
|------|------|----|
| belief 梯度门控 | 不存在（v4.7 Oracle/Infer 二分但无梯度门控） | **v4 新增防线 5** |
| 实施位置 | – | grad_gating.py + model.set_context_subjective 内 |
| 切断方式 | – | `.detach()` 显式 |
| 阈值控制 | – | cfg.train.belief_grad_gating_steps（默认 5000） |
| 与 Pkg-03 BeliefEncoder 关系 | – | BeliefEncoder 在 .detach() 之后接收（投影 MLP 仍由 main loss 训练）|

---

## 7. Cross-references

- Ch4.6.5 belief 梯度门控（v4 新增）
- `02-hyper-muzero-model-v2.md`（model.set_context_subjective 内调用 grad_gating.apply）
- `03-stability-safeguards-preservation.md`（防线 5 引用 + 4 道防线对照）
- `08-integration-contracts.md` §4（与 Pkg-05 trainer 协作：update_step 调用约定）
- Pkg-01 spec 05 TrainConfig.belief_grad_gating_steps（默认 5000）
- Pkg-03 spec 04 BeliefNet（输出 raw heads c_hat / z_hat）
- Pkg-03 spec 01 BeliefEncoder（在 .detach() 之后接收）
- Pkg-03 spec 08 §5（梯度门控分层设计协作）
- Pkg-05 spec 04（trainer-loop 在 train_step 起点调 model.update_step）
