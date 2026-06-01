# Spec 04: MA-MuZero Baseline — `ma_muzero`（断言 A：共享 RewardHead 平均梯度）

> 父文档：[`../proposal.md`](../proposal.md) §2.1.4 · [`../design.md`](../design.md) §4 D5/D7/D9 · §7.4
> **断言覆盖**：**断言 A**（类型梯度撕裂）失败模式①。
> **Pkg-08 用途**：断言 A 主对比（`baseline_ma_muzero`）。
> **边界**：与 `rewardhead_explicit_type`（spec 05，失败模式②）撕分——见 §1.2 边界表（design D7）。
> **上游对齐**：7-API 逐字对齐 Pkg-04 spec 02；vanilla MARL world-model（无 hypernet）。

---

## 1. Purpose

提供 `hyper_mve/baselines/ma_muzero.py`，实现 vanilla 多智能体 MuZero 世界模型。它是断言 A 的"最朴素对照"：**完全不区分 type 的专用容量**，所有 agent 共享单一 RewardHead，仅把 `agent_id one-hot + own_type one-hot` 拼进 input。

| 模型类 | variant | RewardHead | type 处理 | hypernet |
|--------|---------|-----------|-----------|----------|
| `MAMuZeroBaselineModel` | `ma_muzero` | **共享单一头** | agent_id one-hot + own_type one-hot 进 input | **无** |

### 1.1 断言 A 的对照逻辑（共享头学"平均梯度"）

不同 type 的 agent 有不同的奖励结构（α/β 异质）。共享单一 RewardHead 被迫用**同一组权重**拟合所有 type 的奖励 → 反向传播时不同 type 的梯度在共享权重上**相互抵消/平均**，学到的是"所有 type 的平均奖励函数"，无法精确刻画任一 type。hyper 的 `hyper_rew(rule_emb ⊕ id_emb)` 为每个 (rule, agent) 生成专用 θ_reward，消除这一撕裂。`ma_muzero` 即此撕裂的实证对照点。

### 1.2 与 rewardhead_explicit_type 的边界（D7，避免撞车）

两者都打断言 A，但**失败模式不同**，必须分清：

| variant | hypernet | RewardHead | type 输入 | 失败模式（断言 A 对照点）| 归属 |
|---------|----------|-----------|-----------|--------------------------|------|
| `ma_muzero`（本 spec）| 无 | 共享单一头 | agent_id + own_type one-hot | 共享头学 α/β **平均梯度** | 本 spec |
| `rewardhead_explicit_type` | 有 `hyper_{trans,pred}`，无 `hyper_rew` | 共享头但**显式按 type 分支** | 同上 | 显式 type 分支**无法吸收 capability 连续异质性** | spec 05 |

> `ma_muzero` 证伪"完全不区分 type 也行"；`rewardhead_explicit_type` 是更强的对照（证伪"离散 type 分支够用"）。两者构成断言 A 的递进证据链。

---

## 2. Interface

### 2.1 文件路径

`hyper_mve/baselines/ma_muzero.py`（纯新增）

### 2.2 类签名（ma_muzero 特化部分；**完整 7-API 见 spec 06 §2**）

> 本节仅给 `set_context_*` 等 ma_muzero 特化逻辑；`update_step` / `encode` / `transition` / `predict_reward` / `predict` 的统一签名与调用序契约见 **spec 06 §2**（5 variant 一致，逐字对齐 Pkg-04 spec 02）。

```python
import torch
import torch.nn as nn
from hyper_mve.configs import V4Config
from hyper_mve.baselines.shared_backbones import (
    create_rep_net, create_belief_net, create_tri_context_encoder,
)
from hyper_mve.models.belief_grad_gating import BeliefGradGating


class MAMuZeroBaselineModel(nn.Module):
    """vanilla MA-MuZero: 共享 RewardHead, 无 hypernet, 无 ctx_aug 拼接.

    断言 A 失败模式①: 共享头学 α/β 平均梯度.
    type 信息仅以 own_type one-hot 进 input (self-info 合法, D9).
    """

    SHARED_BACKBONE_PREFIXES = ("rep_net", "belief_net", "tri_context_encoder")

    def __init__(self, cfg: V4Config):
        super().__init__()
        self.cfg = cfg
        # ---- 共享后端 (spec 02, 等参公平: 仍构造完整三后端) ----
        self.rep_net = create_rep_net(cfg.env, cfg.model)
        self.belief_net = create_belief_net(cfg.env, cfg.model)
        self.tri_context_encoder = create_tri_context_encoder(cfg.env, cfg.model)
        self.grad_gating = BeliefGradGating(cfg.train.belief_grad_gating_steps)  # D6
        # ---- 条件化消费子系统: 全部 vanilla, 无 hypernet ----
        id_dim = cfg.env.N + cfg.env.num_types          # agent_id + own_type one-hot
        in_dim = cfg.model.latent_dim + cfg.env.N * cfg.env.A + id_dim
        h = cfg.model.hidden_dim
        self.trans_net = _MLP(in_dim, h, cfg.model.latent_dim)
        self.reward_head = _MLP(in_dim, h, 1)           # ★ 共享单一头 (无 per-agent θ_rew)
        # pred_net 无条件构建 (否则 predict() AttributeError); cfg flag 仅控制权重共享语义
        self.pred_net = _PredMLP(cfg.model.latent_dim + id_dim, h, cfg.env.A)
        self._share_pred_head = cfg.model.baseline_ma_muzero_share_pred_head
        # ---- stateful 缓存 (D8, 7-API 契约) ----
        self._id_onehot = None      # agent_id + own_type one-hot (最后一次 subjective)
        self._agent_id = None
        self._step = 0              # 由 update_step(global_step) 维护 (spec 06), 喂 grad_gating.apply

    def set_context_objective(self, c_t):
        self._ctx_obj = c_t

    def set_context_subjective(self, agent_id, cap_i, belief):
        assert cap_i.shape[-1] == 4, "Self-Info 严格 (spec 06 C6-SELF1)"   # D9
        # ma_muzero 不消费 belief 做条件化, 但 gating 仍 apply —— D6 要求 5 variant
        # 的 update_step / gating 行为完全一致 (C6-GRAD1 对每个 variant 跑, 必挂否则).
        c_hat, z_hat = belief
        _c, _z = self.grad_gating.apply(c_hat, z_hat, self._step)   # 三参, Pkg-04 spec 02 line 258
        self._agent_id = agent_id
        self._id_onehot = self._build_id_onehot(agent_id, cap_i)   # agent_id + own_type
```

> **关键①**：`reward_head` 是**单一 nn.Module**，所有 agent_id 共用同一组权重——这是"平均梯度"撕裂的来源（C6-STRUCT2）。
> **关键②**：`belief` tuple 进入 `set_context_subjective` 不参与 ma_muzero 条件化，但**仍须经 `grad_gating.apply`**——否则 C6-GRAD1（对每个 variant 跑）会发现 ma_muzero 的 gating 行为偏离其余 variant。`baseline_ma_muzero_share_pred_head` 仅控制 pred 头权重共享语义，`pred_net` **无条件构建**（cfg=False 不得让 `predict()` 触发 AttributeError；若该旋钮语义未实现，构造时 `raise NotImplementedError` 显式拒绝，见 §4）。

### 2.3 own_type vs oracle types 的 Self-Info 边界（D9）

`ma_muzero` 用 agent 自己的 type（`own_type`）作 input —— 这是 **self-info 合法**（agent 知道自己是什么 type）。但**不得**把 opponents 的 oracle types 拼进 input：

```python
def _build_id_onehot(self, agent_id, cap_i):
    # own_type: 从 cap_i (B,4) 派生或 env self-info, 合法
    # ✗ 禁止: env.info["types"][其他 agent]  —— 那是 oracle 泄漏 (C6-SELF1)
    ...
```

具名单测 `test_baseline_set_context_subjective_no_oracle_types_leak`（spec 06，对 ma_muzero 跑）验证 `cap_i.shape[-1]==4`（拒绝增广的 oracle 输入）。

---

## 3. Implementation Notes

### 3.1 仍构造完整三后端（等参公平，对齐 spec 02 SB2）

`ma_muzero` 虽是 vanilla MARL，仍经 `create_rep_net/belief_net/tri_context_encoder` 构造完整三后端（参数量与 hyper 逐位相同，C6-FAIR2）。原因同 `no_belief`（spec 02 §3.1）：移除的是"专用容量机制"，不是"后端参数"——否则对照混入"参数更少"干扰。条件化消费子系统则结构性偏小（无 hypernet、无 ctx_aug 拼接）。

### 3.2 等参豁免（D5，spec 07）

`ma_muzero` 的条件化子系统**结构性偏小**（无生成器、无 ctx 拼接，仅 id one-hot），无法、也不应强行对齐 hyper 参数量。它**豁免** 5%/10% 双阈值（spec 07 只对 input_wide/input_deep/no_belief 断言）。豁免代价：必须在 LR sweep 协议下报告 **wall-clock 步数等价性**（spec 07 OQ-1 详化），证明对照不是"参数太少导致欠拟合"。

### 3.3 共享单一 RewardHead（C6-STRUCT2）

`reward_head` 必须是**单一** `nn.Module`，不得按 agent_id/type 分裂为多个头（那是 `rewardhead_explicit_type` 的领域，spec 05）。具名单测 `test_ma_muzero_shared_rewardhead` 验证：不同 agent_id 的 `predict_reward` 走同一组 `reward_head` 参数（无 per-agent θ_rew）。

### 3.4 belief 门控仍一致（D6）

虽然 ma_muzero 不消费 belief 做条件化，其 `update_step(global_step)` 仍维护与其余 variant **完全一致**的 gating 计数逻辑（共用 `BeliefGradGating` 类 + 5K 阈值）。这保证 5 variant 的 `update_step` 行为可由同一具名单测 `test_baseline_update_step_gates_belief_grad`（spec 06）覆盖，无 variant 特例。

---

## 4. Edge Cases

| 场景 | 行为 |
|------|------|
| 试图拼 opponent oracle types 进 input | `set_context_subjective` 仅接 `cap_i (B,4)`；增广输入触发 assert（C6-SELF1）|
| `baseline_ma_muzero_share_pred_head=False` | `pred_net` 仍**无条件已构建**（predict() 不会 AttributeError）；若"非共享 pred 头"语义未实现则构造时 `raise NotImplementedError`（默认 True 全 vanilla）|
| `predict_reward` 在 `set_context_subjective` 前调 | `_id_onehot is None` → AssertionError（调用序，spec 06）|
| 与 hyper 比参数量 | 条件化子系统偏小 → spec 07 **豁免**双阈值，改报 wall-clock |
| pre-5K backward | BeliefNet 梯度=0（C6-GRAD1，与其余 variant 一致）|

---

## 5. Acceptance Criteria

### 5.1 单元测试（`tests/baselines/test_ma_muzero.py`）

```python
import pytest
import torch
from hyper_mve.configs import V4Config
from hyper_mve.baselines import create_baseline_model


@pytest.fixture
def cfg():
    return V4Config.from_preset("medium")


# ====== C6-STRUCT2: 共享单一 RewardHead, 无 per-agent θ_rew ======

def test_ma_muzero_shared_rewardhead(cfg):
    """不同 agent_id 的 predict_reward 走同一组 reward_head 参数."""
    model = create_baseline_model(cfg, "ma_muzero")
    ids = list(model.reward_head.parameters())
    # 只有一个 reward_head, 无 reward_head_0 / reward_head_1 ... 多头
    head_modules = [n for n, _ in model.named_modules() if "reward_head" in n]
    # 允许 reward_head 本身 + 其子层, 但不得有按 agent index 命名的并列多头
    assert not any(n.rstrip("0123456789").endswith("reward_head_") for n in head_modules), (
        "ma_muzero 出现 per-agent reward 头, 与共享头语义冲突"
    )
    assert len(ids) > 0


def test_ma_muzero_no_hypernet(cfg):
    """ma_muzero 是 vanilla MARL, 无 hypernet."""
    from hyper_mve.models.hyper_network import DualHyperNetwork
    model = create_baseline_model(cfg, "ma_muzero")
    for m in model.modules():
        assert not isinstance(m, DualHyperNetwork)


# ====== own_type 合法 / opponent oracle types 非法 (C6-SELF1, spec 06 统一跑) ======

def test_ma_muzero_rejects_augmented_cap(cfg):
    """cap_i 维度 != 4 (疑似拼了 oracle types) → assert."""
    model = create_baseline_model(cfg, "ma_muzero")
    B = 4
    bad_cap = torch.zeros(B, 8)        # 4 (cap) + 4 (泄漏的 oracle type) = 8
    belief = (torch.zeros(B), torch.zeros(B, cfg.env.N - 1, 2))
    model.set_context_objective(torch.zeros(B))
    with pytest.raises(AssertionError):
        model.set_context_subjective(0, bad_cap, belief)
```

> 等参豁免下的 wall-clock 步数等价性报告协议见 spec 07（OQ-1）。

---

## 6. Cross-references

- [`02-shared-backbones.md`](./02-shared-backbones.md)（仍构造完整三后端，等参公平 SB2）
- [`06-model-7api-conformance.md`](./06-model-7api-conformance.md)（7-API + Self-Info own_type 边界 + gating 一致）
- [`07-param-fairness-and-lr-sweep.md`](./07-param-fairness-and-lr-sweep.md)（ma_muzero 豁免双阈值 → wall-clock 步数等价性）
- [`08-integration-contracts.md`](./08-integration-contracts.md)（五条复用约束）
- Pkg-04 spec 02（7-API + stateful）
- Pkg-04 spec 04（`BeliefGradGating`）
- Ch4_1_Motivation（断言 A：类型梯度撕裂）
