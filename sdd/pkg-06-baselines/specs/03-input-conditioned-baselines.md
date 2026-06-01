# Spec 03: Input-Conditioned Baselines — `input_wide` / `input_deep`（断言 B 等参对照）

> 父文档：[`../proposal.md`](../proposal.md) §2.1.3 · [`../design.md`](../design.md) §4 D5/D6/D8 · §7.4
> **断言覆盖**：**断言 B**（belief 专用容量）。这是等参公平性的**核心验证场**——证明"加宽/加深的 input-conditioning 即使吃掉同等参数预算，也学不出 hypernet 的 per-context 专用容量"。
> **Pkg-08 用途**：断言 B 主对比（`baseline_input_wide` / `baseline_input_deep`）。
> **上游对齐**：7-API 逐字对齐 Pkg-04 spec 02；功能网结构复用 Pkg-04 `functional_nets.py`（仅改 input 维度/层数，不重写）。

---

## 1. Purpose

提供 `hyper_mve/baselines/input_conditioned.py`，实现两个 input-conditioning baseline 模型类。两者与 hyper 的唯一差异：**不经 DualHyperNetwork 生成 per-context 权重，而是把 `ctx_aug` 直接 concat 进固定权重功能网的 input**，再用"加宽 / 加深"消耗等量参数预算：

| 模型类 | variant | 条件化机制 | 参数预算消耗方式 |
|--------|---------|-----------|-----------------|
| `InputWideBaselineModel` | `input_wide` | ctx_aug concat 进 input | **加宽**功能网隐层（`baseline_wide_hidden_dim`）|
| `InputDeepBaselineModel` | `input_deep` | ctx_aug concat 进 input | **加深**功能网层数（`baseline_deep_layers`）|

**断言 B 的对照逻辑**：input baseline 拿到与 hyper **同等参数预算**（spec 07 双阈值），共享同一 RepNet/BeliefNet/TriContextEncoder（spec 02），唯一变量是"per-context 专用权重 vs 全局共享权重 + ctx 拼接"。若 hyper 显著优于二者，则证明专用容量不可由"更宽/更深 + 输入条件化"替代。

---

## 2. Interface

### 2.1 文件路径

`hyper_mve/baselines/input_conditioned.py`（纯新增）

### 2.2 类签名（input-conditioning 特化部分；**完整 7-API 见 spec 06 §2**）

> 本节仅给 `set_context_*` / `transition` 等 input-conditioning 特化逻辑；`update_step` / `encode` / `predict_reward` / `predict` 的统一签名与调用序契约见 **spec 06 §2**（5 variant 一致，逐字对齐 Pkg-04 spec 02）。

```python
import torch
import torch.nn as nn
from hyper_mve.configs import V4Config
from hyper_mve.baselines.shared_backbones import (
    create_rep_net, create_belief_net, create_tri_context_encoder,
)
from hyper_mve.models.belief_grad_gating import BeliefGradGating   # Pkg-04 spec04


class _InputConditionedBase(nn.Module):
    """input_wide / input_deep 共享基类.

    与 HyperMuZeroModel 的差异: 无 DualHyperNetwork; 功能网为固定权重,
    在 forward 内 concat ctx_aug 进 input. 其余 (后端 / gating / stateful /
    Self-Info) 与 hyper 逐字一致 (spec 06).
    """

    # SB5: count_conditioning_params 过滤共享后端用 (spec 02)
    SHARED_BACKBONE_PREFIXES = ("rep_net", "belief_net", "tri_context_encoder")

    def __init__(self, cfg: V4Config):
        super().__init__()
        self.cfg = cfg
        # ---- 共享后端 (spec 02, 与 hyper 同类同构) ----
        self.rep_net = create_rep_net(cfg.env, cfg.model)
        self.belief_net = create_belief_net(cfg.env, cfg.model)
        self.tri_context_encoder = create_tri_context_encoder(cfg.env, cfg.model)
        # ---- belief 梯度门控 (D6, 与 hyper 共用同一类 + 同一阈值) ----
        self.grad_gating = BeliefGradGating(cfg.train.belief_grad_gating_steps)
        # ---- 条件化消费子系统: 固定权重功能网 (无 hypernet) ----
        ctx_dim = self.tri_context_encoder.out_dim          # ctx_aug 维度
        self._build_functional_nets(cfg, ctx_dim)           # 子类实现 (宽 / 深)
        # ---- stateful 缓存 (D8, 7-API 契约) ----
        self._ctx_obj = None        # set_context_objective 缓存 c_t
        self._agent_id = None       # 最后一次 set_context_subjective 的 agent_id
        self._ctx_aug = None        # 最后一次组装的 ctx_aug (gating 已施加)
        self._step = 0              # 由 update_step(global_step) 维护 (spec 06), 喂 grad_gating.apply

    def _build_functional_nets(self, cfg, ctx_dim):
        raise NotImplementedError   # InputWide / InputDeep 各自实现


class InputWideBaselineModel(_InputConditionedBase):
    """断言 B 对照: ctx_aug concat 进 input + 加宽隐层."""

    def _build_functional_nets(self, cfg, ctx_dim):
        h = cfg.model.baseline_wide_hidden_dim          # 加宽旋钮 (spec 01 §1.1)
        in_trans = cfg.model.latent_dim + cfg.env.N * cfg.env.A + ctx_dim
        self.trans_net = _MLP(in_trans, h, cfg.model.latent_dim, n_layers=2)
        self.reward_head = _MLP(in_trans, h, 1, n_layers=2)
        self.pred_net = _PredMLP(cfg.model.latent_dim + ctx_dim, h, cfg.env.A)


class InputDeepBaselineModel(_InputConditionedBase):
    """断言 B 对照: ctx_aug concat 进 input + 加深层数."""

    def _build_functional_nets(self, cfg, ctx_dim):
        L = cfg.model.baseline_deep_layers              # 加深旋钮 (spec 01 §1.1)
        h = cfg.model.hidden_dim                        # 宽度同 hyper, 仅加深
        in_trans = cfg.model.latent_dim + cfg.env.N * cfg.env.A + ctx_dim
        self.trans_net = _MLP(in_trans, h, cfg.model.latent_dim, n_layers=L)
        self.reward_head = _MLP(in_trans, h, 1, n_layers=L)
        self.pred_net = _PredMLP(cfg.model.latent_dim + ctx_dim, h, cfg.env.A, n_layers=L)
```

> `_MLP` / `_PredMLP` 是本文件内的固定权重小网络（无 AdaLN、无 hypernet 注入）；它们是条件化消费子系统的全部参数（spec 02 `count_conditioning_params` 计入）。

### 2.3 ctx_aug concat + gating 路径（断言 B 公平性关键，D6）

```python
def set_context_objective(self, c_t):
    self._ctx_obj = c_t                                  # (B,) | (B,1) f32

def set_context_subjective(self, agent_id, cap_i, belief):
    assert cap_i.shape[-1] == 4, "Self-Info 严格 (spec 06 C6-SELF1)"   # D9
    c_hat, z_hat = belief                                # tuple (Pkg-04 spec 02 L206)
    # ★ D6: 与 hyper 共用同一 gating 类 + 5K 阈值 —— pre-5K 时 belief grad 被 detach
    # apply 三参 (c_hat, z_hat, step), 逐字对齐 Pkg-04 spec 02 line 258
    c_hat_g, z_hat_g = self.grad_gating.apply(c_hat, z_hat, self._step)
    ctx_aug = self.tri_context_encoder(self._ctx_obj, agent_id, cap_i, (c_hat_g, z_hat_g))
    self._agent_id = agent_id                            # D8: 缓存最后一次 subjective
    self._ctx_aug = ctx_aug                              # forward 内 concat

def transition(self, s, action):
    x = torch.cat([s, action, self._ctx_aug], dim=-1)    # concat 而非 hypernet
    return s + self.trans_net(x)                         # 残差 (与 hyper StateTransNet 一致)
```

> 关键：input baseline **没有 hypernet**，但 `ctx_aug` 仍经 `BeliefGradGating.apply` —— 因此 pre-5K 时加宽/加深功能网拿到的 belief 梯度同样为 0，与 hyper 公平（D6 mermaid 图，spec 06）。

---

## 3. Implementation Notes

### 3.1 为什么 concat 进 input 而非生成权重（断言 B 的本质对照）

hyper 的核心主张是"per-context 专用权重容量"。input-conditioning 是文献最常见的平替：把 context 拼进输入，让一套**全局共享**权重去拟合所有 context。两者参数预算可对齐（spec 07），但表达力结构不同——input baseline 必须用同一组权重同时服务所有 (rule, agent) 组合，无法像 hypernet 那样为每个 context 切换"整套权重"。断言 B 即检验这一差异是否实证显著。

### 3.2 加宽 vs 加深：两个独立失败模式

| variant | 旋钮 | 检验的反驳 |
|---------|------|-----------|
| `input_wide` | `baseline_wide_hidden_dim` | "把网络加宽到同等参数量就够了" |
| `input_deep` | `baseline_deep_layers` | "把网络加深到同等参数量就够了" |

两者分开是为堵住"宽度不够 / 深度不够"两种审稿质疑。等参逼近迭代见 spec 07。

### 3.3 不含 DualHyperNetwork（C6-STRUCT1）

input baseline **不得**实例化 `DualHyperNetwork` / `ChunkedHyperNetwork`。具名单测 `test_input_baseline_no_hypernet` 遍历 `model.modules()` 断言无 hypernet 类型。这是断言 B 对照的结构前提——若混入 hypernet，对照即失效。

### 3.4 stateful 契约（D8，与 hyper 完全一致）

`transition/predict_reward/predict` 用**最后一次** `set_context_subjective` 缓存的 `_ctx_aug`（含 `_agent_id`）。即便内部是 concat 而非 hypernet，调用面语义与 hyper 逐字一致（Pkg-04 spec 02 L308-310），保证 MuZeroTrainer 的 N-agent 循环对它透明。详见 spec 06。

### 3.5 复用 functional_nets 结构（NG2/NG5）

`_MLP` / `_PredMLP` 优先复用 Pkg-04 `functional_nets.py` 的层定义（仅调 in_dim/hidden/层数），不重写网络语义。AdaLN 调制对 input baseline 无意义（无外部权重注入），可关闭——但残差连接（预测 Δs）保留以对齐 hyper 的 StateTransNet。

---

## 4. Edge Cases

| 场景 | 行为 |
|------|------|
| `transition` 在 `set_context_subjective` 前被调 | `_ctx_aug is None` → AssertionError（调用序约束，spec 06）|
| `cap_i.shape[-1] != 4`（泄漏 oracle types）| AssertionError（C6-SELF1，spec 06）|
| `baseline_wide_hidden_dim` 未在 cfg 声明 | `__init__` AttributeError（M4 待 Pkg-01 同步，spec 01 §1.1）|
| pre-5K 调 backward | BeliefNet 梯度=0（gating，C6-GRAD1）|
| 加宽/加深后参数量 > hyper 110% | spec 07 `test_baseline_param_count_within_5pct` fail（调小旋钮）|

---

## 5. Acceptance Criteria

### 5.1 单元测试（`tests/baselines/test_input_conditioned.py`）

```python
import pytest
import torch
from hyper_mve.configs import V4Config
from hyper_mve.baselines import create_baseline_model
from hyper_mve.models.hyper_network import DualHyperNetwork


@pytest.fixture
def cfg():
    return V4Config.from_preset("medium")


# ====== C6-STRUCT1: input-conditioned 不含 DualHyperNetwork ======

def test_input_baseline_no_hypernet(cfg):
    """input_wide / input_deep 内部无任何 hypernet 模块 (断言 B 结构前提)."""
    for variant in ("input_wide", "input_deep"):
        model = create_baseline_model(cfg, variant)
        for m in model.modules():
            assert not isinstance(m, DualHyperNetwork), (
                f"{variant} 含 hypernet, 断言 B 对照失效"
            )


# ====== ctx_aug concat 生效 (条件化机制) ======

def test_input_baseline_ctx_affects_output(cfg):
    """不同 subjective context → transition 输出不同 (证明 ctx_aug 真接入)."""
    model = create_baseline_model(cfg, "input_wide")
    B = 4
    s = torch.randn(B, cfg.model.latent_dim)
    a = torch.zeros(B, cfg.env.N * cfg.env.A)
    cap0 = torch.zeros(B, 4); cap1 = torch.ones(B, 4)
    belief = (torch.zeros(B), torch.zeros(B, cfg.env.N - 1, 2))
    model.set_context_objective(torch.zeros(B))
    model.set_context_subjective(0, cap0, belief); out0 = model.transition(s, a)
    model.set_context_subjective(1, cap1, belief); out1 = model.transition(s, a)
    assert not torch.allclose(out0, out1)


# ====== pre-5K belief 梯度门控 (C6-GRAD1, 与 hyper 公平) ======

def test_input_baseline_pre5k_belief_grad_zero(cfg):
    """pre-5K: input baseline 的 BeliefNet 梯度为 0 (与 hyper 一致)."""
    model = create_baseline_model(cfg, "input_deep")
    model.update_step(global_step=100)        # < 5000
    B = 4
    obs = torch.randn(B, cfg.env.N, cfg.env.obs_dim)
    model.set_context_objective(torch.zeros(B))
    cap = torch.zeros(B, 4)
    belief = model.belief_net(obs)            # 带 grad
    model.set_context_subjective(0, cap, belief)
    s = model.encode(obs)
    loss = model.transition(s, torch.zeros(B, cfg.env.N * cfg.env.A)).sum()
    loss.backward()
    for p in model.belief_net.parameters():
        assert p.grad is None or torch.all(p.grad == 0)
```

> C6-API*/C6-SELF1/C6-API2（stateful）对 input_wide/input_deep 的覆盖在 spec 06 统一单测（每 variant 跑）；等参 5%/10% 在 spec 07。

---

## 6. Cross-references

- [`02-shared-backbones.md`](./02-shared-backbones.md)（`create_rep_net` / `create_belief_net` / `create_tri_context_encoder` + `count_conditioning_params`）
- [`06-model-7api-conformance.md`](./06-model-7api-conformance.md)（7-API 一致性 + stateful + Self-Info + gating mermaid）
- [`07-param-fairness-and-lr-sweep.md`](./07-param-fairness-and-lr-sweep.md)（input baseline 是 5%/10% 双阈值的主对象 + 加宽/加深逼近流程）
- [`08-integration-contracts.md`](./08-integration-contracts.md)（五条复用约束）
- Pkg-04 spec 02（7-API + stateful L308-310 + functional_nets 残差）
- Pkg-04 spec 04（`BeliefGradGating`）
- Ch4_1_Motivation（断言 B）+ Ch5 §5.6（等参协议）
