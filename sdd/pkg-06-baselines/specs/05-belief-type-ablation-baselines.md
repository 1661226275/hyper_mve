# Spec 05: Belief & Type Ablation Baselines — `no_belief`（断言 C/Abl7）+ `rewardhead_explicit_type`（断言 A/Abl6.x）

> 父文档：[`../proposal.md`](../proposal.md) §2.1.5 · [`../design.md`](../design.md) §4 D5/D7/D9 · §7.4
> **断言覆盖**：`no_belief` → **断言 C**（三路必要性，Ablation 7）；`rewardhead_explicit_type` → **断言 A** 失败模式②（Ablation 6.x）。
> **Pkg-08 用途**：Ablation 7（belief 路移除）+ Ablation 6.x（显式 type 分支）。
> **上游对齐**：7-API 逐字对齐 Pkg-04 spec 02；二者均为 `HyperMuZeroModel` 的受控变体（保留 hypernet 骨架，单点改动）。

---

## 1. Purpose

提供两个 ablation baseline 模型类，各自精确移除/替换一个机制，构成断言 C / A 的对照：

| 模型类 | variant | 改动点 | 断言 | Pkg-08 |
|--------|---------|--------|------|--------|
| `NoBeliefBaselineModel` | `no_belief` | TriContextEncoder belief 路**置零** | C | Abl7 |
| `ExplicitTypeRewardBaselineModel` | `rewardhead_explicit_type` | 去掉 `hyper_rew`，改**显式 type 分支** RewardHead | A | Abl6.x |

两者都是"单点受控变体"——除指定改动外，其余（后端、hypernet 骨架、gating、7-API）与 hyper 逐字一致，使对照干净。

---

## 2. Interface

### 2.1 文件路径

- `hyper_mve/baselines/no_belief.py` → `NoBeliefBaselineModel`
- `hyper_mve/baselines/explicit_type_reward.py` → `ExplicitTypeRewardBaselineModel`

### 2.2 `NoBeliefBaselineModel`（断言 C：belief 路置零）

```python
class NoBeliefBaselineModel(nn.Module):
    """断言 C 对照 (Abl7): 三路 (c_ctx / role_i / belief_i) 移除 belief_i.

    实现方式: 仍构造完整 BeliefNet (等参, spec 02 SB2), 仅在
    set_context_subjective 把 belief tuple 送 TriContextEncoder 前置零
    —— 移除的是信息通路, 非参数 (spec 02 §3.1).
    保留 hypernet 骨架 (hyper_trans / hyper_rew / hyper_pred), 与 hyper 仅差 belief 路.
    """

    SHARED_BACKBONE_PREFIXES = ("rep_net", "belief_net", "tri_context_encoder")

    def __init__(self, cfg):
        super().__init__()
        # 后端 + hypernet 骨架与 hyper 完全一致 (复用 HyperMuZeroModel 构造路径)
        ...
        self.grad_gating = BeliefGradGating(cfg.train.belief_grad_gating_steps)  # D6
        self._step = 0     # 由 update_step(global_step) 维护 (spec 06), 喂 grad_gating.apply

    def set_context_subjective(self, agent_id, cap_i, belief):
        assert cap_i.shape[-1] == 4                       # D9 Self-Info
        c_hat, z_hat = belief
        # 三参 apply (c_hat, z_hat, step), 逐字对齐 Pkg-04 spec 02 line 258
        c_hat_g, z_hat_g = self.grad_gating.apply(c_hat, z_hat, self._step)   # gating 仍走 (D6)
        # ★ 断言 C 改动: belief 路置零 (c_ctx / role_i 保留)
        zero_belief = (torch.zeros_like(c_hat_g), torch.zeros_like(z_hat_g))
        ctx_aug = self.tri_context_encoder(self._ctx_obj, agent_id, cap_i, zero_belief)
        # 经 hypernet 生成 θ (与 hyper 一致), 仅 ctx_aug 缺 belief 信息
        ...
```

> **为什么仍构造 BeliefNet + 仍走 gating**：等参公平（spec 02 SB2）要求 `no_belief` 与 hyper 在后端参数量逐位相同；gating 一致（D6）要求 `update_step` 行为与其余 variant 无差。移除仅发生在"belief tuple → TriContextEncoder belief 路"这一信息接口处（置零），保证对照只隔离"belief 信息是否进入条件化"这一个变量。

### 2.3 `ExplicitTypeRewardBaselineModel`（断言 A 失败模式②：显式 type 分支）

```python
class ExplicitTypeRewardBaselineModel(nn.Module):
    """断言 A 对照② (Abl6.x): 保留 hyper_trans / hyper_pred, 去掉 hyper_rew,
    RewardHead 改为显式按 type 分支 (type-conditioned, 离散).

    对照点: 离散 type 分支无法吸收 capability 连续异质性 (cap 连续, type 离散).
    比 ma_muzero 更强 —— 它确实区分 type, 但用离散分支而非连续 capability 专用 θ.
    """

    SHARED_BACKBONE_PREFIXES = ("rep_net", "belief_net", "tri_context_encoder")

    def __init__(self, cfg):
        super().__init__()
        ...
        # 保留 trans / pred 的 hypernet
        self.hyper_trans = DualHyperNetwork(...)     # 同 hyper
        self.hyper_pred = DualHyperNetwork(...)      # 同 hyper
        # ★ 无 hyper_rew; 改显式 type 分支 RewardHead
        n_branch = cfg.model.baseline_explicit_type_branches   # = cfg.env.num_types
        self.reward_head = _TypeBranchedRewardHead(n_branch, ...)
        self.grad_gating = BeliefGradGating(cfg.train.belief_grad_gating_steps)  # D6
        self._step = 0     # 由 update_step(global_step) 维护 (spec 06), 喂 grad_gating.apply

    def set_context_subjective(self, agent_id, cap_i, belief):
        assert cap_i.shape[-1] == 4                       # D9 Self-Info (与 ma_muzero 对称)
        c_hat, z_hat = belief
        # 三参 apply (c_hat, z_hat, step), 逐字对齐 Pkg-04 spec 02 line 258 (gating 仍走, D6)
        c_hat_g, z_hat_g = self.grad_gating.apply(c_hat, z_hat, self._step)
        ctx_aug = self.tri_context_encoder(self._ctx_obj, agent_id, cap_i, (c_hat_g, z_hat_g))
        self._own_type = self._derive_own_type(cap_i)     # self-info 选分支 (非 opponent oracle)
        ...

    def predict_reward(self, s, action):
        # 用 own_type 选分支 (而非 hyper_rew 生成专用 θ)
        return self.reward_head(s, action, branch=self._own_type)
```

> `_TypeBranchedRewardHead` 含 `n_branch` 组并列分支，按离散 own_type 选一组——与 `ma_muzero`（spec 04，单一共享头）的关键差异：它**区分 type**，但分支是离散的，无法表达连续 capability 的细粒度异质性。

### 2.4 边界对照表（与 spec 04 ma_muzero，design D7）

| variant | hypernet | RewardHead | 断言 A 对照点 | spec |
|---------|----------|-----------|---------------|------|
| `ma_muzero` | 无 | 共享单一头 | 完全不分 type → 平均梯度 | spec 04 |
| `rewardhead_explicit_type` | `hyper_{trans,pred}`，无 `hyper_rew` | 显式 type 分支 | 离散分支无法吸收连续 capability | 本 spec |

---

## 3. Implementation Notes

### 3.1 no_belief 是"信息消融"而非"参数消融"（断言 C 干净对照）

若 `no_belief` 直接不构造 BeliefNet，它与 hyper 在后端层就少了 BeliefNet 参数 → 断言 C 对照混入"参数更少"干扰。正确做法（spec 02 §3.1）：构造完整 BeliefNet（参数对齐），仅置零 belief tuple 进 TriContextEncoder 的通路。这样隔离的唯一变量是"belief_i 路是否携带信息"。`no_belief` 因此**进** 5%/10% 双阈值（spec 07）——它的条件化子系统与 hyper 同构（hypernet 3 头 + 3 功能网）。

### 3.2 explicit_type 的等参豁免（D5，spec 07）

`rewardhead_explicit_type` 用离散 type 分支替换 `hyper_rew`，条件化子系统结构与 hyper 不同（多了 type 分支、少了一个 hypernet 头）→ **结构性差异，豁免** 5%/10% 双阈值。豁免代价同 ma_muzero：LR sweep 下报 wall-clock 步数等价性（spec 07）。

### 3.3 两者都保留 gating 一致性（D6）

`no_belief` 与 `rewardhead_explicit_type` 的 `update_step` 均共用 `BeliefGradGating` 类 + 5K 阈值。`no_belief` 尤其重要：即便它把 belief 路置零，pre-5K gating 仍需触发同一逻辑（否则 `test_baseline_update_step_gates_belief_grad` 对 no_belief 行为偏离）。置零发生在 gating **之后**，gating 计数逻辑不受影响。

### 3.4 Self-Info 严格（D9，两者均 assert）

两个模型 `set_context_subjective` 均 `assert cap_i.shape[-1] == 4`。`rewardhead_explicit_type` 的 own_type 选分支用 self-info（agent 自己的 type），不得用 opponent oracle types。

---

## 4. Edge Cases

| 场景 | 行为 |
|------|------|
| `no_belief` 未构造 BeliefNet | SB2 失败（spec 02 `test_no_belief_still_builds_full_belief_net`）|
| `no_belief` belief 路未真正置零（仍泄漏 belief 信息）| 断言 C 对照失效 → `test_no_belief_zeros_belief_path` 拦截 |
| `explicit_type` 仍保留 `hyper_rew` | 与设计冲突 → `test_explicit_type_no_hyper_rew` 拦截 |
| `baseline_explicit_type_branches != num_types` | 分支数错配（M4 默认 = num_types，spec 01 §1.1）|
| 两者 pre-5K backward | BeliefNet 梯度=0（C6-GRAD1）|

---

## 5. Acceptance Criteria

### 5.1 单元测试（`tests/baselines/test_belief_type_ablation.py`）

```python
import pytest
import torch
from hyper_mve.configs import V4Config
from hyper_mve.baselines import create_baseline_model


@pytest.fixture
def cfg():
    return V4Config.from_preset("medium")


# ====== no_belief: belief 路置零 (断言 C) ======

def test_no_belief_zeros_belief_path(cfg):
    """喂不同 belief, no_belief 的 ctx_aug / 输出不变 (belief 信息被置零)."""
    model = create_baseline_model(cfg, "no_belief")
    B = 4
    obs = torch.randn(B, cfg.env.N, cfg.env.obs_dim)
    model.set_context_objective(torch.zeros(B))
    cap = torch.zeros(B, 4)
    s = model.encode(obs)
    a = torch.zeros(B, cfg.env.N * cfg.env.A)
    b1 = (torch.zeros(B), torch.zeros(B, cfg.env.N - 1, 2))
    b2 = (torch.ones(B), torch.ones(B, cfg.env.N - 1, 2))
    model.set_context_subjective(0, cap, b1); out1 = model.transition(s, a)
    model.set_context_subjective(0, cap, b2); out2 = model.transition(s, a)
    assert torch.allclose(out1, out2), "belief 路未置零 (断言 C 对照失效)"


# ====== explicit_type: 无 hyper_rew, 有 type 分支 RewardHead (断言 A②) ======

def test_explicit_type_no_hyper_rew(cfg):
    """rewardhead_explicit_type 无 hyper_rew, 但保留 hyper_trans / hyper_pred."""
    model = create_baseline_model(cfg, "rewardhead_explicit_type")
    names = [n for n, _ in model.named_modules()]
    assert not any("hyper_rew" in n for n in names), "不应保留 hyper_rew"
    assert any("hyper_trans" in n for n in names)
    assert any("hyper_pred" in n for n in names)


def test_explicit_type_reward_branches_by_type(cfg):
    """不同 own_type → reward_head 走不同分支."""
    model = create_baseline_model(cfg, "rewardhead_explicit_type")
    assert model.reward_head.n_branch == cfg.model.baseline_explicit_type_branches
```

> C6-API*/C6-SELF1/C6-GRAD1 对二者的覆盖在 spec 06 统一单测（每 variant 跑）。

---

## 6. Cross-references

- [`02-shared-backbones.md`](./02-shared-backbones.md)（no_belief 仍构造完整 BeliefNet，SB2）
- [`06-model-7api-conformance.md`](./06-model-7api-conformance.md)（7-API + Self-Info + gating 一致）
- [`07-param-fairness-and-lr-sweep.md`](./07-param-fairness-and-lr-sweep.md)（no_belief 进双阈值；explicit_type 豁免 → wall-clock）
- [`08-integration-contracts.md`](./08-integration-contracts.md)（五条复用约束）
- Pkg-04 spec 02（7-API + `HyperMuZeroModel` 骨架）
- Pkg-04 spec 04（`BeliefGradGating`）
- Pkg-03 spec 08 §6（TriContextEncoder 三路 c_ctx / role_i / belief_i）
- Ch4_1_Motivation（断言 C 三路必要性 / 断言 A 失败模式②）
