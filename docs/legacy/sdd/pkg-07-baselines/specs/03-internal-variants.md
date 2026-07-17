# Spec 03: Internal Variants — 5 Model Classes (合并 pkg-06 specs 03+04+05)

> 父文档：[`../proposal.md`](../proposal.md) §2.1.3–§2.1.5 · [`../design.md`](../design.md) §3.2 (11-key 工厂矩阵 internal 行) · §D3 (Internal vs External 分派) · §D4 (共享后端粒度) · §D6 (Belief 门控一致性)
> **断言覆盖**：断言 A（`ma_muzero` + `rewardhead_explicit_type`，类型梯度撕裂）· 断言 B′（`input_wide` + `input_deep`，belief 专用容量）· 断言 C / Ablation 7（`no_belief`，三路必要性）
> **上游对齐**：7-API 逐字对齐 Pkg-04 spec 02；共享后端经 [`02-shared-backbones-internal.md`](./02-shared-backbones-internal.md) `create_rep_net` / `create_belief_net` / `create_tri_context_encoder` 工厂注入。

---

## ⚠️ 头部强制声明：supersede pkg-06 specs 03 + 04 + 05

> 本 spec **吸收并取代** pkg-06 三份 internal-variant spec：
> - [`pkg-06-baselines/specs/03-input-conditioned-baselines.md`](../../pkg-06-baselines/specs/03-input-conditioned-baselines.md)（`input_wide` / `input_deep`）
> - [`pkg-06-baselines/specs/04-ma-muzero-baseline.md`](../../pkg-06-baselines/specs/04-ma-muzero-baseline.md)（`ma_muzero`）
> - [`pkg-06-baselines/specs/05-belief-type-ablation-baselines.md`](../../pkg-06-baselines/specs/05-belief-type-ablation-baselines.md)（`no_belief` + `rewardhead_explicit_type`）
>
> pkg-06 这三份 spec 自此为**只读历史档案**；任何 internal-variant 设计变更须改本 spec。本 spec 将三份内容压缩为**单一连贯文档**：共享 7-API 表面与 stateful/Self-Info/grad-gating 三组横向契约只出现一次（§3–§6），而非每 variant 重复一遍；纵向 5 variant 差异仅在 §7 五子节内描述。
>
> 约束重号：pkg-06 C6-STRUCT1 / C6-STRUCT2 / C6-SELF1 / C6-GRAD1 / C6-API1 / C6-API2 ⇒ **C7-INT-STRUCT1 / C7-INT-STRUCT2 / C7-INT-SELF1 / C7-INT-GRAD1 / C7-INT-API1 / C7-INT-API2**（详见 pkg-07 README §"v4 关键约束" Internal 表）。

---

## 1. Purpose

提供 `hyper_mve/baselines/internal/` 下 5 个 internal baseline 模型类的统一规格：每个 variant 是 `MuZeroTrainer` 的可消费 model（实现 Pkg-04 spec 02 完整 7-API），共享同一 backbone 注入路径（`create_rep_net` / `create_belief_net` / `create_tri_context_encoder`，spec 02）与同一 `BeliefGradGating`（design D6）。5 variant 的差异**全部**集中在「条件化消费子系统」（即 functional nets + 是否含 hypernet + 是否消费 belief 路 + RewardHead 结构），其余（后端、stateful 缓存、Self-Info 校验、grad-gating 行为、7-API 调用面）逐字一致。

| variant | factory arg | 模型类 | 断言 / 用途 | failure mode | 结构 marker |
|---|---|---|---|---|---|
| `baseline_input_wide` | `"input_wide"` | `InputWideBaselineModel` | 断言 B′ 等参 | 加宽全局共享权重无法替代 per-context 专用容量 | 无 hypernet · concat `ctx_aug` 进 input · 加宽 hidden |
| `baseline_input_deep` | `"input_deep"` | `InputDeepBaselineModel` | 断言 B′ 等参 | 加深全局共享权重无法替代 per-context 专用容量 | 无 hypernet · concat `ctx_aug` 进 input · 加深 layers |
| `baseline_ma_muzero` | `"ma_muzero"` | `MAMuZeroBaselineModel` | 断言 A 失败模式① | 共享单一 RewardHead 学 α/β **平均梯度** | 无 hypernet · 共享单一 RewardHead · id+own_type one-hot 进 input |
| `no_belief` | `"no_belief"` | `NoBeliefBaselineModel` | 断言 C / Abl7 | 移除 belief 路后三路条件化失稳 | 含 hypernet 骨架 · belief 路在 TriCtx **置零** |
| `rewardhead_explicit_type` | `"rewardhead_explicit_type"` | `ExplicitTypeRewardBaselineModel` | 断言 A 失败模式② / Abl6.x | 离散 type 分支无法吸收连续 capability 异质 | 含 `hyper_trans` / `hyper_pred` · **无 `hyper_rew`** · RewardHead 显式按 own_type 分支 |

> "结构 marker" 列对应 pkg-07 README C7-INT-STRUCT1 / C7-INT-STRUCT2 + 本 spec §9 单测的断言点。

---

## 2. 文件布局

```
hyper_mve/baselines/internal/
├── __init__.py                     # re-export 5 model classes
├── input_conditioned.py            # InputWideBaselineModel + InputDeepBaselineModel (共享 _InputConditionedBase)
├── ma_muzero.py                    # MAMuZeroBaselineModel
├── no_belief.py                    # NoBeliefBaselineModel
└── explicit_type_reward.py         # ExplicitTypeRewardBaselineModel
```

5 文件均位于 `hyper_mve/baselines/internal/`（spec 01 §3.2 import 路径锚）；`__init__.py` 仅 re-export 类名供 `hyper_mve/baselines/__init__.py::INTERNAL_REGISTRY` 取用。

---

## 3. Common 7-API Surface（逐字对齐 Pkg-04 spec 02）

5 variant 全部实现以下 7 个方法，签名与调用序约束逐字对齐 Pkg-04 spec 02 §2.2（line 169–322）。本节是 5 variant 的**共同表面**，§7 各 variant 子节不重复签名，只描述「该 variant 在该方法内做什么差异化的事」。

### 3.1 7 方法签名

```python
class _InternalBaselineBase(nn.Module):
    """5 internal variant 共同的 7-API + stateful + Self-Info + grad-gating 表面。

    子类只重写 _build_conditioning_subsystem(cfg) (functional nets / hypernet 骨架)
    与 _apply_conditioning(s, action) (forward 内如何用 ctx) 两个 hook。
    """

    # ====== 1) update_step (Pkg-04 spec 02 line 169) ======
    def update_step(self, global_step: int) -> None:
        """每个 train_step 起点调用一次, 用于 belief grad gating 阈值判断.

        worker 端 inference 路径不调 (无梯度需求, 与 hyper 一致 Pkg-04 spec 02 line 173-175).
        """
        self._step = global_step

    # ====== 2) set_context_objective (Pkg-04 spec 02 line 181) ======
    def set_context_objective(self, c_t: torch.Tensor) -> None:
        """C1: 仅接 c_t (Harsanyi 共同知识); K-step unroll 起点调一次, 5 variant 共享 θ_state.

        必须在 set_context_subjective 之前调用 (调用序断言, Pkg-04 spec 02 line 184-185).
        """
        self._ctx_obj = c_t                          # (B,) | (B,1) f32

    # ====== 3) set_context_subjective (Pkg-04 spec 02 line 202) ======
    def set_context_subjective(
        self,
        agent_id: int,
        cap_i: torch.Tensor,                          # (B, 4) RAW CapabilityVector — Self-Info 严格
        belief: tuple[torch.Tensor, torch.Tensor],    # (c_hat (B,), z_hat (B, N-1, 2))
    ) -> None:
        """C2/C3: 缓存最后一次 (agent_id, cap_i, belief) 供 transition/predict_reward/predict 消费.

        - Self-Info 严格 (§5, C7-INT-SELF1): assert cap_i.shape[-1] == 4.
        - grad-gating (§6, C7-INT-GRAD1): belief tuple 必走 self.grad_gating.apply(c_hat, z_hat, self._step).
        - 子类 hook (§7): 在缓存完 ctx 后调 self._build_conditioning_state(agent_id, cap_i_g, belief_gated).
        """
        assert cap_i.shape[-1] == 4, (
            "Self-Info strict: cap_i 必须是 (B, 4) RAW CapabilityVector; "
            "禁止拼接 opponent oracle types 等增广字段 (C7-INT-SELF1)."
        )
        c_hat, z_hat = belief
        # D6 三参 apply, 逐字对齐 Pkg-04 spec 02 line 258
        c_hat_g, z_hat_g = self.grad_gating.apply(c_hat, z_hat, self._step)
        self._agent_id = agent_id
        self._cap_i = cap_i
        self._belief_gated = (c_hat_g, z_hat_g)
        self._build_conditioning_state(agent_id, cap_i, (c_hat_g, z_hat_g))   # 子类 hook (§7)

    # ====== 4) encode (Pkg-04 spec 02 line 277) ======
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """obs (B, N, obs_dim) → s (B, latent_dim) via RepNet.

        RepNet 客观 (5 variant + hyper 共享, 不依赖任何 set_context).
        """
        return self.rep_net(obs)

    # ====== 5) transition (Pkg-04 spec 02 line 284) ======
    def transition(
        self,
        s: torch.Tensor,                               # (B, latent_dim)
        action: torch.Tensor,                          # (B, N*A) joint-action one-hot flat
    ) -> torch.Tensor:                                 # (B, latent_dim) s_next
        """s + Δs (残差, 与 hyper StateTransNet 一致); Δs 由子类 _apply_trans 计算.

        stateful: 使用最后一次 set_context_subjective 缓存的 (agent_id, cap, belief) (§4).
        """
        self._assert_subjective_set("transition")
        return s + self._apply_trans(s, action)        # 子类 hook (§7)

    # ====== 6) predict_reward (Pkg-04 spec 02 line 298) ======
    def predict_reward(
        self,
        s: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:                                 # (B, 1) r_i (subjective)
        """主观 reward; 使用最后一次 subjective agent_id 缓存 (§4)."""
        self._assert_subjective_set("predict_reward")
        return self._apply_reward(s, action)           # 子类 hook (§7)

    # ====== 7) predict (Pkg-04 spec 02 line 318) ======
    def predict(
        self,
        s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """s → (policy_logits (B, A), value (B, 1)); 使用最后一次 subjective agent_id 缓存."""
        self._assert_subjective_set("predict")
        return self._apply_pred(s)                     # 子类 hook (§7)
```

### 3.2 调用序硬约束（Pkg-04 spec 02 line 184-185 / line 308-310 继承）

5 variant 必须在 `transition` / `predict_reward` / `predict` 被调用前，至少一次完整的 `set_context_objective(c_t) → set_context_subjective(agent_id, cap_i, belief)` 序列。`_assert_subjective_set(method_name)` 断言 `self._agent_id is not None`；否则抛 `AssertionError`（与 hyper 行为对齐）。

### 3.3 子类 hook 表（§7 每 variant 实现）

| hook | 职责 | hyper 等价物 |
|---|---|---|
| `_build_conditioning_subsystem(cfg)` (in `__init__`) | 构造 functional nets ± hypernet 骨架 | hyper 的 `hyper_{trans,rew,pred} + functional_nets` |
| `_build_conditioning_state(agent_id, cap_i, belief_gated)` | 用 (cap, belief) 计算或缓存 per-call 条件化状态（ctx_aug / θ / id_onehot） | hyper 的 `θ_{rew,pred}^i = hyper_*(rule_emb ⊕ id_emb)` |
| `_apply_trans(s, action)` | 返回 Δs（残差） | hyper 的 `StateTransNet(s, action, θ_state)` |
| `_apply_reward(s, action)` | 返回 (B,1) r_i | hyper 的 `RewardHead(s, action, θ_rew^i)` |
| `_apply_pred(s)` | 返回 (policy_logits, value) | hyper 的 `PredictionNet(s, θ_pred^i)` |

> 三个 `_apply_*` hook 全部从 `self._agent_id` / `self._cap_i` / `self._belief_gated` 读 stateful 状态，不接 explicit (agent_id, cap, belief) 参数 — 这是 stateful 契约（§4）的实现基础。

---

## 4. Stateful Contract (C7-INT-API2)

### 4.1 状态字段

`__init__` 末尾统一声明：

```python
self._step: int = 0                       # update_step 维护
self._ctx_obj: torch.Tensor | None = None # 最后一次 set_context_objective 的 c_t
self._agent_id: int | None = None         # 最后一次 set_context_subjective 的 agent_id
self._cap_i: torch.Tensor | None = None   # 最后一次 set_context_subjective 的 cap_i (B, 4)
self._belief_gated: tuple | None = None   # gating 已施加的 (c_hat, z_hat)
```

子类可**追加**自己的 stateful 缓存（如 `_ctx_aug` for input_wide / `_id_onehot` for ma_muzero / `_theta_rew` / `_theta_pred` for no_belief & explicit_type），但**不得移除**上述五个公共字段（`_assert_subjective_set` 依赖 `_agent_id` 非空作为"已 set"信号）。

### 4.2 "最后一次 subjective" 语义

`transition` / `predict_reward` / `predict` 三方法**全部**从 `self._agent_id` 读最近一次 `set_context_subjective` 的 agent_id；不接 explicit agent_id 参数（与 Pkg-04 spec 02 line 308-310 hyper 行为逐字一致）。MuZeroTrainer 的 N-agent 循环对此透明 — 每次切换 agent 都先调 `set_context_subjective`，5 variant 自动消费缓存。

### 4.3 缓存重置不耦合 episode 边界

stateful 缓存**不**随 episode 起止重置。Worker / Trainer 在每次新的 `set_context_subjective` 调用时**显式覆盖**前一次状态；本 spec **不**实现 `reset_state()` 接口（与 hyper 对齐，Pkg-04 spec 02）。Edge case 由 §8 表覆盖。

---

## 5. Self-Info Strict (C7-INT-SELF1)

### 5.1 通用断言

```python
assert cap_i.shape[-1] == 4, "..."   # 见 §3.1 set_context_subjective
```

5 variant **全部** 在 `set_context_subjective` 入口断言 `cap_i.shape[-1] == 4`。`cap_i` 是 agent 自己的 RAW CapabilityVector（4 维，per Pkg-02），属于 self-info 合法。任何增广（如把 opponent oracle type one-hot 拼到 cap_i 尾部使维度 > 4）触发 AssertionError，被单测 `test_internal_self_info_strict_cap_dim` 覆盖。

### 5.2 belief 来源限定

5 variant 全部消费的 `belief` tuple **必须** 由 `self.belief_net(obs)` 在 trainer/worker 侧生成（Pkg-03 spec 08 §6 BeliefNet 契约），**不得** 从 `env.info["types"]` / `env.info["c_true"]` 等 oracle 字段直接构造。本约束在 spec 04 PettingZoo adapter `oracle_mode=False` 的 info gating 之外形成第二道防线：即便 env info 被错误暴露，model 类内部 set_context_subjective 接口也**不**留 type/c_true 入口（参数表 `(agent_id, cap_i, belief)` 没有任何 oracle 字段位）。

### 5.3 own_type 例外（ma_muzero / explicit_type）

`ma_muzero` 与 `rewardhead_explicit_type` 需要 **own_type**（agent 自己的离散 α/β 类型 one-hot）进 input / 选分支。约束：own_type 必须从 `cap_i (B, 4)` 派生（Pkg-02 已锁定 RAW capability 含 own type bits）或从 env self-info dict 取得（**不**从 `env.info["types"]` 整张表读），由各 variant 子节描述派生公式。不得读 opponent 的 type — 这是断言 A 对照点要求的「agent 不知道其他 agent 的 type」前提。

---

## 6. BeliefGradGating Consistency (C7-INT-GRAD1, design D6)

### 6.1 共用同一 `BeliefGradGating` 类

5 variant 在 `__init__` 内 **统一**：

```python
from hyper_mve.models.belief_grad_gating import BeliefGradGating   # Pkg-04 spec 04

self.grad_gating = BeliefGradGating(cfg.train.belief_grad_gating_steps)
```

`cfg.train.belief_grad_gating_steps`（默认 5000，Pkg-04 spec 04）跨 5 variant + hyper 共享同一来源，禁止 variant 内 hardcoded 阈值。

### 6.2 gating 行为强制对齐 hyper

pre-5K 步：`grad_gating.apply(c_hat, z_hat, step)` 返回 detach 版本（c_hat / z_hat 的梯度路径被切断），BeliefNet 在反向传播中收到的梯度为 0。post-5K 步：apply 透传原 tensor，BeliefNet 解锁参与端到端训练。

5 variant 行为差异**不在** gating，而在 gating-after 处理：
- `input_wide` / `input_deep`：把 `(c_hat_g, z_hat_g)` 送 TriContextEncoder 拼成 ctx_aug。
- `ma_muzero`：apply **必走**，但 ma_muzero 不消费 belief 做条件化（`_c, _z = self.grad_gating.apply(...)` 弃用返回值）— 这是为了让 §6.3 的 `test_pre5k_belief_grad_zero` 单测对 ma_muzero 也通过（gating 计数逻辑统一）。
- `no_belief`：apply 走完后**置零**：`zero_belief = (torch.zeros_like(c_hat_g), torch.zeros_like(z_hat_g))` 再送 TriContextEncoder（断言 C 的信息消融点）。
- `rewardhead_explicit_type`：apply 走完后 (c_hat_g, z_hat_g) 送 TriContextEncoder（与 input/no_belief 同路径），但 RewardHead 改走显式 type 分支。

### 6.3 单测形态

`tests/baselines/internal/test_grad_gating.py::test_pre5k_belief_grad_zero_all_5_variants`：
```python
@pytest.mark.parametrize("variant", [
    "input_wide", "input_deep", "ma_muzero", "no_belief", "rewardhead_explicit_type"
])
def test_pre5k_belief_grad_zero(cfg, variant):
    model = create_baseline(cfg, variant)
    model.update_step(global_step=100)            # < 5000
    # ... forward + backward ...
    for p in model.belief_net.parameters():
        assert p.grad is None or torch.all(p.grad == 0)
```

Post-5K 解锁单测对称（global_step=10000，p.grad 非零）。

---

## 7. Per-Variant Subsections

每 variant 子节只描述「与 §3–§6 共同表面相比，本 variant 多做 / 少做 / 改做了什么」。基类 / hook 实现样例不重复。

### 7.1 `InputWideBaselineModel`（factory arg `"input_wide"`）

**文件**：`hyper_mve/baselines/internal/input_conditioned.py`（与 `InputDeepBaselineModel` 共用 `_InputConditionedBase` 抽象类）

**结构**：无 hypernet（C7-INT-STRUCT1），固定权重功能网。

**条件化机制**：`set_context_subjective` 把 (c_hat_g, z_hat_g) 与 (c_t, agent_id, cap_i) 一起送 `tri_context_encoder` 得到 `ctx_aug (B, ctx_dim)`，在 `transition` / `predict_reward` / `predict` 内 `torch.cat([s, action, ctx_aug], dim=-1)` 接入功能网 input（concat-conditioning）。

**参数预算消耗**：加宽 functional net hidden width。

**`_build_conditioning_subsystem(cfg)`**：

```python
def _build_conditioning_subsystem(self, cfg):
    h = cfg.baselines.internal_wide_hidden_dim            # 加宽旋钮 (spec 01 §6.1)
    ctx_dim = self.tri_context_encoder.out_dim
    in_trans  = cfg.model.latent_dim + cfg.env.N * cfg.env.A + ctx_dim
    in_reward = cfg.model.latent_dim + cfg.env.N * cfg.env.A + ctx_dim
    in_pred   = cfg.model.latent_dim + ctx_dim
    self.trans_net   = _MLP(in_trans,  h, cfg.model.latent_dim, n_layers=2)
    self.reward_head = _MLP(in_reward, h, 1,                    n_layers=2)
    self.pred_net    = _PredMLP(in_pred, h, cfg.env.A,           n_layers=2)
```

**cfg 字段消费**：`cfg.baselines.internal_wide_hidden_dim`（int，spec 01 §6.1）— 等参逼近的可调旋钮（spec 07 双阈值迭代），禁止 hardcoded（C7-INT-CFG1）。

**结构 marker 断言**：`test_input_wide_no_hypernet` 遍历 `model.modules()` 断言无 `DualHyperNetwork` / `ChunkedHyperNetwork`（C7-INT-STRUCT1）。

**等参 fairness 归属**：进 spec 07 5%/10% 双阈值（前缀 `baseline_*` → 等参强制对齐）。

### 7.2 `InputDeepBaselineModel`（factory arg `"input_deep"`）

**文件**：同 `InputWideBaselineModel`（共用 `_InputConditionedBase`）。

**结构**：无 hypernet（C7-INT-STRUCT1）；条件化机制与 `input_wide` 一致（concat ctx_aug 进 input）。

**参数预算消耗**：加深 functional net 层数（宽度同 hyper）。

**`_build_conditioning_subsystem(cfg)`**：

```python
def _build_conditioning_subsystem(self, cfg):
    L = cfg.baselines.internal_deep_layers                # 加深旋钮 (spec 01 §6.1)
    h = cfg.model.hidden_dim                              # 宽度同 hyper
    ctx_dim = self.tri_context_encoder.out_dim
    in_trans  = cfg.model.latent_dim + cfg.env.N * cfg.env.A + ctx_dim
    in_pred   = cfg.model.latent_dim + ctx_dim
    self.trans_net   = _MLP(in_trans, h, cfg.model.latent_dim, n_layers=L)
    self.reward_head = _MLP(in_trans, h, 1,                    n_layers=L)
    self.pred_net    = _PredMLP(in_pred, h, cfg.env.A,         n_layers=L)
```

**cfg 字段消费**：`cfg.baselines.internal_deep_layers`（int，spec 01 §6.1）。

**结构 marker 断言**：同 §7.1（无 hypernet）。

**等参 fairness 归属**：进 spec 07 5%/10% 双阈值。

### 7.3 `MAMuZeroBaselineModel`（factory arg `"ma_muzero"`）

**文件**：`hyper_mve/baselines/internal/ma_muzero.py`

**结构**：无 hypernet（C7-INT-STRUCT1） · **共享单一 RewardHead**（C7-INT-STRUCT2）。

**条件化机制**：把 `agent_id one-hot ⊕ own_type one-hot` 拼成 `id_onehot (B, N + num_types)`，在 `transition` / `predict_reward` 内 `torch.cat([s, action, id_onehot], dim=-1)`。`set_context_subjective` 接收 belief 但**不**做条件化（只走 gating 让 §6.3 单测过），保留 `tri_context_encoder` 已构造但不消费。

**RewardHead 结构**：

```python
self.reward_head = _MLP(in_dim, h, 1)                      # ★ 单一 nn.Module, 不按 agent 分裂多头 (C7-INT-STRUCT2)
```

不得出现 `self.reward_head_0` / `self.reward_head_1` / `ModuleList([...])` 等并列多头形式（被 §9.3 `test_ma_muzero_shared_rewardhead` 拦截）。

**pred head**：

```python
self._share_pred_head = cfg.baselines.internal_ma_muzero_share_pred_head
self.pred_net = _PredMLP(cfg.model.latent_dim + id_dim, h, cfg.env.A)   # 无条件构建
```

`_share_pred_head=True`（默认）→ 走单一 pred head 共享（与 reward head 对称的 vanilla MA-MuZero 行为）；`_share_pred_head=False` 当前**仅占位**：若该旋钮"非共享 pred 头"语义未在实施期落实，`__init__` 内 `if not self._share_pred_head: raise NotImplementedError("...")`（pkg-06 spec 04 §4 Edge Cases 继承）。但 `pred_net` **总被构造**（否则 `predict()` 触发 AttributeError），这是 Edge Cases 表的固定行（§8）。

**own_type 来源**：从 `cap_i (B, 4)` 派生（Pkg-02 RAW capability bits 含 own type）；**禁止** 从 `env.info["types"]` 整张表读（§5.3）。

**cfg 字段消费**：`cfg.baselines.internal_ma_muzero_share_pred_head`（bool，spec 01 §6.1）。

**等参 fairness 归属**：豁免 spec 07 5%/10% 双阈值（结构性偏小：无 hypernet generator + 无 ctx_aug 拼接，条件化子系统 vs hyper 不可对齐），改报 wall-clock 步数等价性（spec 07 OQ-1）。

### 7.4 `NoBeliefBaselineModel`（factory arg `"no_belief"`）

**文件**：`hyper_mve/baselines/internal/no_belief.py`

**结构**：**保留** hypernet 骨架（`hyper_trans` / `hyper_rew` / `hyper_pred` + functional_nets，与 hyper 逐字一致），单点改动 = belief 路置零。

**条件化机制**：`set_context_subjective` 走完 `grad_gating.apply` 后，把 belief 替换为零张量再送 TriContextEncoder：

```python
def _build_conditioning_state(self, agent_id, cap_i, belief_gated):
    c_hat_g, z_hat_g = belief_gated
    zero_belief = (torch.zeros_like(c_hat_g), torch.zeros_like(z_hat_g))   # ★ 断言 C 改动点
    ctx_aug = self.tri_context_encoder(self._ctx_obj, agent_id, cap_i, zero_belief)
    # 经 hypernet 生成 θ (与 hyper 一致), 仅 ctx_aug 缺 belief 信息
    self._theta_state = self.hyper_trans(self._ctx_obj_emb)      # 客观, 不依赖 belief
    self._theta_rew_i = self.hyper_rew(ctx_aug)                  # 主观, ctx_aug 缺 belief 路
    self._theta_pred_i = self.hyper_pred(ctx_aug)
```

**信息消融而非参数消融**：`belief_net` 仍构造（spec 02 SB2 等参公平），仅在通路上置零。`grad_gating` 仍走（§6.2，pre-5K BeliefNet 梯度=0 与其他 variant 一致）。这样断言 C 隔离的唯一变量是 "belief_i 路是否携带信息" — 不是 "BeliefNet 参数是否存在"。

**cfg 字段消费**：**不读** `cfg.baselines.internal_*` 中任何 variant-specific 旋钮（结构由 hyper 骨架决定）。

**等参 fairness 归属**：进 spec 07 5%/10% 双阈值（条件化子系统结构与 hyper 同构）。

### 7.5 `ExplicitTypeRewardBaselineModel`（factory arg `"rewardhead_explicit_type"`）

**文件**：`hyper_mve/baselines/internal/explicit_type_reward.py`

**结构**：保留 `hyper_trans` + `hyper_pred`（与 hyper 一致），**无 `hyper_rew`**（C7-INT-STRUCT3，等价 pkg-06 spec 05 §5 `test_explicit_type_no_hyper_rew` 断言），RewardHead 改为显式 type 分支。

**条件化机制**：

```python
def _build_conditioning_subsystem(self, cfg):
    self.hyper_trans = DualHyperNetwork(...)               # 同 hyper
    self.hyper_pred  = DualHyperNetwork(...)               # 同 hyper
    n_branch = cfg.baselines.internal_explicit_type_branches   # = cfg.env.num_types
    self.reward_head = _TypeBranchedRewardHead(n_branch, cfg.model.latent_dim, cfg.env.N * cfg.env.A, h=cfg.model.hidden_dim)

def _build_conditioning_state(self, agent_id, cap_i, belief_gated):
    c_hat_g, z_hat_g = belief_gated
    ctx_aug = self.tri_context_encoder(self._ctx_obj, agent_id, cap_i, (c_hat_g, z_hat_g))
    self._theta_state = self.hyper_trans(self._ctx_obj_emb)
    self._theta_pred_i = self.hyper_pred(ctx_aug)
    self._own_type = self._derive_own_type(cap_i)          # self-info 选分支 (§5.3)

def _apply_reward(self, s, action):
    return self.reward_head(s, action, branch=self._own_type)   # 离散 type 分支
```

`_TypeBranchedRewardHead`：内部含 `n_branch` 组并列 `_MLP`（`nn.ModuleList`），按 `branch` 索引选一组前向。对照点：离散 type 分支无法吸收 capability 连续异质性（cap 连续 4 维，type 离散 2 维）— 比 ma_muzero 的"完全不分 type"更强的断言 A 反驳。

**cfg 字段消费**：`cfg.baselines.internal_explicit_type_branches`（int，**默认在 `BaselinesConfig` 顶层定义为 `cfg.env.num_types`** —— spec 01 §6.1 锁定；C7-INT-CFG1 禁止 model class 内部 hardcoded fallback，model class 只读 `cfg.baselines.internal_explicit_type_branches`，对 `cfg.env.num_types` 的同步由 Pkg-01 spec 05 实施期 dataclass `default_factory` 完成）。

**等参 fairness 归属**：豁免 spec 07 5%/10% 双阈值（条件化子系统结构与 hyper 不同：多了 type 分支、少了 hyper_rew），改报 wall-clock 步数等价性。

---

## 8. Edge Cases（5 variant 共通 + variant-specific 合并表）

| 场景 | variant | 行为 |
|---|---|---|
| `transition` / `predict_reward` / `predict` 在 `set_context_subjective` 前被调 | 全 5 | `AssertionError`（`_assert_subjective_set` 拦截，§3.2 调用序）|
| `cap_i.shape[-1] != 4`（疑似拼了 oracle types） | 全 5 | `AssertionError`（C7-INT-SELF1，§5.1）|
| `belief` 不是 (c_hat, z_hat) tuple | 全 5 | `ValueError` / unpack TypeError（与 hyper 行为对齐，Pkg-04 spec 02）|
| 5K-window crossing（pre→post）| 全 5 | gating 自动从 detach 切到透传（`BeliefGradGating` 行为，Pkg-04 spec 04） |
| `cfg.baselines.internal_wide_hidden_dim` 缺失 | `input_wide` | `__init__` `AttributeError`（待 Pkg-01 spec 05 同步消费；C7-INT-CFG1） |
| `cfg.baselines.internal_deep_layers` 缺失 | `input_deep` | 同上 |
| `cfg.baselines.internal_ma_muzero_share_pred_head=False` | `ma_muzero` | `__init__` `NotImplementedError`（"非共享 pred 头"语义未实现；pred_net 仍**无条件构造**避免 `predict()` AttributeError） |
| `cfg.baselines.internal_explicit_type_branches != cfg.env.num_types` | `rewardhead_explicit_type` | warn（M4 默认 = num_types；用户显式设置异常分支数不阻塞，由 spec 07 LR sweep 报告失败模式） |
| `no_belief` belief 路意外未置零（实施漏洞） | `no_belief` | `test_no_belief_zeros_belief_path` 拦截：喂不同 belief，输出不变（断言 C 对照失效守门）|
| `rewardhead_explicit_type` 仍含 `hyper_rew` 模块（实施漏洞） | `rewardhead_explicit_type` | `test_explicit_type_no_hyper_rew` 拦截（C7-INT-STRUCT3 守门）|
| 加宽/加深后参数量超 hyper 110%（spec 07 fail 阈值）| `input_wide` / `input_deep` | spec 07 `test_internal_param_count_within_5pct` fail → 调小旋钮重跑 |
| pre-5K backward | 全 5 | BeliefNet 梯度=0（C7-INT-GRAD1）|

---

## 9. Test Contract — C7-INT-* 强制单测清单

测试根目录：`tests/baselines/internal/`（pkg-07 README §"输出清单"已锁路径）。

### 9.1 7-API conformance（每 variant 跑）

`test_7api_conformance.py`：

```python
@pytest.mark.parametrize("variant", [
    "input_wide", "input_deep", "ma_muzero", "no_belief", "rewardhead_explicit_type"
])
def test_7api_signatures(cfg, variant):
    """C7-INT-API1: 5 variant 实现 Pkg-04 spec 02 完整 7 方法 + 签名匹配."""
    model = create_baseline(cfg, variant)
    for name in ("update_step", "set_context_objective", "set_context_subjective",
                 "encode", "transition", "predict_reward", "predict"):
        assert callable(getattr(model, name))
    # 调用面 smoke
    model.update_step(100)
    model.set_context_objective(torch.zeros(B))
    model.set_context_subjective(0, torch.zeros(B, 4),
                                 (torch.zeros(B), torch.zeros(B, cfg.env.N - 1, 2)))
    s = model.encode(torch.randn(B, cfg.env.N, cfg.env.obs_dim))
    a = torch.zeros(B, cfg.env.N * cfg.env.A)
    _ = model.transition(s, a)
    _ = model.predict_reward(s, a)
    _ = model.predict(s)
```

### 9.2 Stateful contract（每 variant 跑，C7-INT-API2）

`test_stateful.py`：

```python
@pytest.mark.parametrize("variant", [...])
def test_predict_uses_last_subjective_agent_id(cfg, variant):
    """连续调两次 set_context_subjective (agent 0 → agent 1), predict 用最后一次."""
    model = create_baseline(cfg, variant)
    model.set_context_objective(torch.zeros(B))
    cap0 = torch.zeros(B, 4); cap1 = torch.ones(B, 4)
    belief = (torch.zeros(B), torch.zeros(B, cfg.env.N - 1, 2))
    s = torch.randn(B, cfg.model.latent_dim)
    model.set_context_subjective(0, cap0, belief); out_a0 = model.predict(s)
    model.set_context_subjective(1, cap1, belief); out_a1 = model.predict(s)
    assert not torch.allclose(out_a0[0], out_a1[0])   # policy logits 必须不同

def test_transition_before_subjective_raises(cfg, variant):
    """transition 在 set_context_subjective 前调 → AssertionError."""
    model = create_baseline(cfg, variant)
    with pytest.raises(AssertionError, match="transition"):
        model.transition(torch.randn(B, cfg.model.latent_dim),
                         torch.zeros(B, cfg.env.N * cfg.env.A))
```

### 9.3 Self-Info strict（每 variant 跑，C7-INT-SELF1）

`test_self_info.py`：

```python
@pytest.mark.parametrize("variant", [...])
def test_internal_self_info_strict_cap_dim(cfg, variant):
    """C7-INT-SELF1: cap_i.shape[-1] != 4 → AssertionError."""
    model = create_baseline(cfg, variant)
    bad_cap = torch.zeros(B, 8)        # 4 (cap) + 4 (泄漏的 oracle type) = 8
    belief = (torch.zeros(B), torch.zeros(B, cfg.env.N - 1, 2))
    model.set_context_objective(torch.zeros(B))
    with pytest.raises(AssertionError, match="cap_i"):
        model.set_context_subjective(0, bad_cap, belief)
```

### 9.4 BeliefGradGating consistency（每 variant 跑，C7-INT-GRAD1）

`test_grad_gating.py`：见 §6.3。pre-5K + post-5K 双向断言。

### 9.5 Structural markers（per-variant）

`test_structural_markers.py`：

```python
def test_input_wide_no_hypernet(cfg):
    """C7-INT-STRUCT1: input_wide 不含 DualHyperNetwork."""
    model = create_baseline(cfg, "input_wide")
    for m in model.modules():
        assert not isinstance(m, (DualHyperNetwork, ChunkedHyperNetwork))

def test_input_deep_no_hypernet(cfg): ...   # 同 input_wide
def test_ma_muzero_no_hypernet(cfg): ...    # 同上

def test_ma_muzero_shared_rewardhead(cfg):
    """C7-INT-STRUCT2: ma_muzero 共享单一 RewardHead, 不按 agent 分裂多头."""
    model = create_baseline(cfg, "ma_muzero")
    head_modules = [n for n, _ in model.named_modules() if "reward_head" in n]
    assert not any(n.rstrip("0123456789").endswith("reward_head_") for n in head_modules)
    # 单一 reward_head, 非 ModuleList
    assert isinstance(model.reward_head, nn.Module) and not isinstance(model.reward_head, nn.ModuleList)

def test_no_belief_zeros_belief_path(cfg):
    """no_belief: 喂不同 belief, transition 输出不变 (断言 C 对照)."""
    model = create_baseline(cfg, "no_belief")
    # ... 见 pkg-06 spec 05 §5 同名单测继承 ...

def test_explicit_type_no_hyper_rew(cfg):
    """C7-INT-STRUCT3: rewardhead_explicit_type 无 hyper_rew, 但保留 hyper_trans / hyper_pred."""
    model = create_baseline(cfg, "rewardhead_explicit_type")
    names = [n for n, _ in model.named_modules()]
    assert not any("hyper_rew" in n for n in names)
    assert any("hyper_trans" in n for n in names)
    assert any("hyper_pred" in n for n in names)

def test_explicit_type_reward_branches_by_type(cfg):
    """rewardhead_explicit_type: RewardHead 分支数 = cfg.baselines.internal_explicit_type_branches."""
    model = create_baseline(cfg, "rewardhead_explicit_type")
    assert model.reward_head.n_branch == cfg.baselines.internal_explicit_type_branches
```

### 9.6 cfg 字段消费验证（C7-INT-CFG1，与 spec 01 §7.4 协作）

spec 01 §7.4 `test_cfg_baselines_5_fields_consumed_no_defaults` 已覆盖 4 internal 字段 mutation。本 spec 仅在 §7 子节描述各 variant 的字段消费契约，不重复单测。

### 9.7 测试矩阵概览

| 测试文件 | 单测数 | 约束覆盖 |
|---|---|---|
| `test_7api_conformance.py` | 5（每 variant）| C7-INT-API1 |
| `test_stateful.py` | 5 × 2 = 10 | C7-INT-API2 |
| `test_self_info.py` | 5 | C7-INT-SELF1 |
| `test_grad_gating.py` | 5 × 2 (pre/post) = 10 | C7-INT-GRAD1 |
| `test_structural_markers.py` | 8（按 variant 拆）| C7-INT-STRUCT1/2/3 |
| **5 model class smoke**（embed 在 `test_7api_conformance`）| 计入 §9.1 | 实例化 + 前向不抛 |

合计：≥ 38 个具名单测点，承接 pkg-06 spec 03+04+05 共 11 单测 + 新增对每 variant 横向扫描覆盖。

---

## 10. Integration Hooks

### 10.1 BaselineModel 返回类型

5 variant 均**实现** `BaselineModel` 协议（Pkg-04 spec 02 7-API + Pkg-05 spec 08 `MuZeroTrainer` 消费约束）。`create_baseline(cfg, internal_variant)` 返回值是 `BaselineModel` instance（spec 01 §3.2 union return type 的 internal 分支）。

### 10.2 MuZeroTrainer 消费契约（pkg-05 spec 08 §3.2 5 条复用约束继承）

5 variant 共用：
- 同一 `MuZeroTrainer`（K-step unroll + n-step return + EMA target + CosineAnnealingLR）
- 同一 `Worker`（episode collection + epsilon-greedy + MVE planner）
- 同一 `EpisodeReplayBuffer`
- 同一 `compose_total_loss`（policy/value/reward/projection 四损失合成）
- 同一 `BeliefGradGating`（§6）

差异仅在 model 类（5 variant 各一）。`MuZeroTrainer` 不需任何 variant-specific 分支 — 7-API + stateful 契约保证对它透明（pkg-07 README C7-INT-REUSE1）。

### 10.3 与 spec 02 backbone factories 的接合

5 variant `__init__` 均调用：

```python
self.rep_net             = create_rep_net(cfg.env, cfg.model)             # spec 02
self.belief_net          = create_belief_net(cfg.env, cfg.model)          # spec 02
self.tri_context_encoder = create_tri_context_encoder(cfg.env, cfg.model) # spec 02
```

无论 variant 是否消费 belief 路（`no_belief` 置零 / `ma_muzero` 不送进 conditioning），**5 variant 都**调三个工厂构造完整三后端 — 这是 spec 02 SB2（等参公平）+ spec 07（5%/10% 双阈值豁免分割）共同的前提。

### 10.4 与 spec 07 fairness 协议的接合

| variant | 等参账目归属 | 报告口径 |
|---|---|---|
| `input_wide` / `input_deep` | 进 5%/10% 双阈值 | 条件化子系统参数量 |
| `ma_muzero` | 豁免 | wall-clock 步数等价性 |
| `no_belief` | 进 5%/10% 双阈值 | 条件化子系统参数量（与 hyper 同构）|
| `rewardhead_explicit_type` | 豁免 | wall-clock 步数等价性 |

详见 spec 07 §双层 fairness 协议 Internal 段。

### 10.5 与 spec 01 工厂 / cfg 的接合

spec 01 §3.2 INTERNAL_REGISTRY 5 keys 全部映射到本 spec §7 五 model class；spec 01 §6.1 `cfg.baselines.internal_*` 4 字段全部由本 spec §7 子节具体消费（无 hardcoded 默认）。

---

## 11. Cross-references

### 上游锁定
- **pkg-07 design.md §3.2**（11-key 工厂矩阵 internal 行 — 5 variant CLI ↔ factory arg ↔ model class）
- **pkg-07 design.md §D3**（Internal vs External 分派 — BaselineModel 返回 internal 分支）
- **pkg-07 design.md §D4**（共享后端粒度 — 5 variant 各自实例化 RepNet/BeliefNet/TriCtx）
- **pkg-07 design.md §D6**（Belief 门控一致性 — 5 variant 共用同一 BeliefGradGating）
- **pkg-07 spec 01 §6.1**（cfg.baselines.internal_* 4 字段，本 spec §7 子节消费）
- **pkg-07 spec 02**（`02-shared-backbones-internal.md` — `create_rep_net` / `create_belief_net` / `create_tri_context_encoder` 工厂签名，本 spec §10.3 调用）
- **Pkg-04 spec 02**（7-API 签名 + stateful + 调用序，本 spec §3 逐字继承）
- **Pkg-04 spec 04**（`BeliefGradGating` 实现，本 spec §6 共用）
- **Pkg-03 spec 08 §6**（TriContextEncoder 三路 c_ctx / role_i / belief_i，本 spec §7.4 `no_belief` 改动点）

### 下游消费
- **pkg-07 spec 07**（`07-fairness-protocol.md` — Internal 段消费本 spec §10.4 fairness 归属表）
- **pkg-07 spec 08**（`08-integration-contracts.md` — 对外硬契约引用本 spec 7-API + Stateful 形态）
- **pkg-08 sweep harness**（消费 `create_baseline(cfg, internal_variant)` → `BaselineModel` → `.evaluate(...)` 统一接口）

### Supersede 锚点
- **pkg-06 spec 03** `03-input-conditioned-baselines.md`（本 spec §7.1 + §7.2 合并继承 `input_wide` / `input_deep`）
- **pkg-06 spec 04** `04-ma-muzero-baseline.md`（本 spec §7.3 合并继承 `ma_muzero` + C7-INT-STRUCT2 共享单 RewardHead 守门）
- **pkg-06 spec 05** `05-belief-type-ablation-baselines.md`（本 spec §7.4 + §7.5 合并继承 `no_belief` + `rewardhead_explicit_type` 与 C7-INT-STRUCT3 无 hyper_rew 守门）

### 论文 / 文献锚点
- **Ch4_1_Motivation**（断言 A：类型梯度撕裂；断言 B′：belief 专用容量；断言 C：三路必要性）
- **Ch5 §5.6**（等参协议，对接 spec 07）
- **Review_v4_TheoryAudit §10.3**（pkg-07/08 必备内容备忘）

---

## 12. Spec Anchors（grep 锚点）

为方便 grep / cross-doc 引用，本 spec 关键约束的锚定 ID 列表：

| Anchor ID | 出现位置 | 用途 |
|---|---|---|
| `C7-INT-API1` | §3.1 / §9.1 / pkg-07 README | 7-API 完整实现 |
| `C7-INT-API2` | §4 / §9.2 | stateful 最后一次 subjective |
| `C7-INT-SELF1` | §5.1 / §9.3 | cap_i 4 维严格 |
| `C7-INT-GRAD1` | §6 / §9.4 | 5 variant 共用同一 BeliefGradGating |
| `C7-INT-STRUCT1` | §7.1 / §7.2 / §7.3 / §9.5 | input/ma_muzero 无 hypernet |
| `C7-INT-STRUCT2` | §7.3 / §9.5 | ma_muzero 共享单 RewardHead |
| `C7-INT-STRUCT3` | §7.5 / §9.5 | rewardhead_explicit_type 无 hyper_rew |
| `C7-INT-CFG1` | §7 / §8 / spec 01 §7.4 | cfg.baselines.internal_* 无 hardcoded 默认 |
| `C7-INT-REUSE1` | §10.2 | 5 variant 共 MuZeroTrainer/Worker/Buffer/compose_total_loss |
| `C7-INT-FAIR1` | §10.4 / spec 07 | 5%/10% 双阈值（input_wide/deep/no_belief 进，ma_muzero/explicit_type 豁免）|

---

**End of spec 03 — internal-variants.**
