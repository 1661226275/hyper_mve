# Spec 02: Shared Backbones — 等参公平性的共享后端

> 父文档：[`../proposal.md`](../proposal.md) §2.1.2 · [`../design.md`](../design.md) §4 D4/D5 · §7.3
> **断言覆盖**：等参公平性是断言 B 的生死线（design §1.2）。本 spec 是公平性的**结构基石**。
> **上游对齐**：RepNet/BeliefNet/TriContextEncoder 共享契约逐字对齐 Pkg-03 spec 08 §6 + Pkg-04 spec 08 §5。

---

## 1. Purpose

提供 `hyper_mve/baselines/shared_backbones.py`，让 5 个 baseline + hyper **各自实例化结构与参数量完全相同的 RepNet / BeliefNet / TriContextEncoder**，从而把模型间差异收敛到唯一变量——**条件化消费子系统**（功能网 + 生成器，见 spec 07 D5 定义）：

| API | 用途 | 调用者 |
|-----|------|--------|
| `create_rep_net(env_cfg, model_cfg)` | 构造与 hyper 同构的 RepresentationNet | 5 模型类 `__init__` |
| `create_belief_net(env_cfg, model_cfg)` | 构造与 hyper 同构的 BeliefNet | 5 模型类（`no_belief` 仍构造，只在 ctx 路置零，见 spec 05）|
| `create_tri_context_encoder(env_cfg, model_cfg)` | 构造与 hyper 同构的 TriContextEncoder | 5 模型类 |
| `count_conditioning_params(model)` | 统计条件化消费子系统参数（排除共享后端）| spec 01 C6-CFG1 + spec 07 C6-FAIR1 |

**核心不变量**：跨 variant（含 hyper）三个后端的**参数量逐位相同**（C6-FAIR2）；训练时**梯度不共享**（各自独立 nn.Module 实例）。

---

## 2. Interface

### 2.1 文件路径

`hyper_mve/baselines/shared_backbones.py`（纯新增）

### 2.2 工厂签名（对齐 Pkg-03 spec 08 §6 构造参数）

```python
import torch.nn as nn
from hyper_mve.configs import EnvConfig, ModelConfig
# 复用上游已发布的网络类（本包不重定义网络结构, 仅工厂封装）
# 用 hyper_mve.models 包级 re-export, 不直引子模块（对齐 Pkg-03 spec 08 §6.2 line 350）
from hyper_mve.models import RepresentationNet                     # Pkg-04 re-export
from hyper_mve.models import BeliefNet, TriContextEncoder          # Pkg-03 spec 08 §6.2 line 350


def create_rep_net(env_cfg: EnvConfig, model_cfg: ModelConfig) -> RepresentationNet:
    """构造与 hyper 同类同构的 RepresentationNet.

    结构由 (env_cfg, model_cfg) 完全决定 → 同 cfg ⇒ 参数量逐位相同.
    返回独立实例 (各 baseline / hyper 各持有自己的, 梯度不共享).
    """
    return RepresentationNet(env_cfg, model_cfg)


def create_belief_net(env_cfg: EnvConfig, model_cfg: ModelConfig) -> BeliefNet:
    """构造与 hyper 同类同构的 BeliefNet (Pkg-03).

    注意: no_belief variant 仍构造完整 BeliefNet (参数量对齐), 只在
    set_context_subjective 内把 belief tuple 在 TriContextEncoder 路置零 (spec 05),
    保证条件化子系统外的等价性.
    """
    return BeliefNet(env_cfg, model_cfg)


def create_tri_context_encoder(env_cfg: EnvConfig, model_cfg: ModelConfig) -> TriContextEncoder:
    """构造与 hyper 同类同构的 TriContextEncoder (Pkg-03 三路: c_ctx / role_i / belief_i)."""
    return TriContextEncoder(env_cfg, model_cfg)


def count_conditioning_params(model) -> int:
    """统计条件化消费子系统参数 (Ch5 §5.6 step1 / design D5).

    包含 (按 variant 不同):
        - θ 生成器 (hypernet 头, 若有: hyper_trans / hyper_rew / hyper_pred)
        - 接收 ctx_aug 的功能网 (StateTransNet / RewardHead / PredictionNet,
          input baseline 为加宽/加深版本)

    排除 (所有 variant 共享, 不计入):
        - RepresentationNet
        - BeliefNet
        - TriContextEncoder

    实现: 遍历 model.named_parameters(), 按 model 暴露的
        model.SHARED_BACKBONE_PREFIXES (= ('rep_net', 'belief_net', 'tri_context_encoder'))
    前缀过滤掉共享后端, 其余即条件化子系统.
    """
    total = 0
    for name, p in model.named_parameters():
        if any(name.startswith(prefix) for prefix in model.SHARED_BACKBONE_PREFIXES):
            continue
        total += p.numel()
    return total
```

### 2.3 共享契约（5 条不变量，对齐 Pkg-03 spec 08 §6）

| # | 不变量 | 验收 |
|---|--------|------|
| SB1 | 同类：5 baseline + hyper 用同一 `RepresentationNet` / `BeliefNet` / `TriContextEncoder` 类 | grep 实例化点 |
| SB2 | 同构：同 (env_cfg, model_cfg) 构造 → 三后端参数量逐位相同 | `test_shared_backbone_identical_param_count` |
| SB3 | 独立实例：各 variant `__init__` 各自调 `create_*`，**非**共享同一 nn.Module | `test_backbones_are_distinct_instances` |
| SB4 | 独立梯度：训练时各 baseline 的后端梯度互不影响（独立 run）| 由 SB3 保证（不同 Parameter 对象）|
| SB5 | 统一命名前缀：`model.rep_net` / `model.belief_net` / `model.tri_context_encoder`（供 `count_conditioning_params` 过滤）| `SHARED_BACKBONE_PREFIXES` 常量 |

---

## 3. Implementation Notes

### 3.1 为什么 no_belief 仍构造完整 BeliefNet（SB2 关键）

`no_belief` 打断言 C（移除 belief 路）。若它**不构造** BeliefNet，则它与 hyper 在"共享后端"层就不等价（少了 BeliefNet 参数），断言 C 的对照就混入了"参数更少"的干扰。因此 `no_belief` **构造完整 BeliefNet（参数量对齐 hyper）**，仅在 `set_context_subjective` 内把 belief tuple 送进 TriContextEncoder 的 belief 路时置零（spec 05）。这样移除的是**信息通路**而非**参数**，对照才干净。

### 3.2 count_conditioning_params 的边界（对齐 D5）

`count_conditioning_params` 是等参验证（spec 07）与 C6-CFG1（spec 01）的共同度量入口，统一放本模块避免两处实现漂移。它依赖每个模型类暴露 `SHARED_BACKBONE_PREFIXES` 常量——5 模型类 + HyperMuZeroModel 都必须定义该常量（spec 06 §统一 API 附加约定）。

逐 variant 落在条件化子系统内的部件（design D5 账目）：

| variant | 计入 conditioning_params 的部件 |
|---------|-------------------------------|
| `hyper`（基准）| hyper_trans / hyper_rew / hyper_pred（3 头）+ 3 vanilla 功能网 |
| `input_wide` | 3 **加宽**功能网（无生成器）|
| `input_deep` | 3 **加深**功能网（无生成器）|
| `no_belief` | hyper_trans / hyper_rew / hyper_pred + 3 vanilla 功能网（与 hyper 同，差异在 belief 置零）|
| `ma_muzero` | 3 vanilla 功能网（共享单一 RewardHead，无生成器）|
| `rewardhead_explicit_type` | hyper_trans / hyper_pred（**无** hyper_rew）+ 带 type 分支的 RewardHead + 2 功能网 |

### 3.3 不重定义网络结构（NG2/NG5 遵守）

本 spec **不实现** RepresentationNet / BeliefNet / TriContextEncoder 的网络结构——它们是 Pkg-03/04 已发布的类，本模块仅做**工厂封装**（保证 5 baseline 用同一构造路径）。封装的价值：单点保证"同 cfg 构造"，避免某个 baseline 误传不同 cfg 字段导致后端结构漂移。

---

## 4. Edge Cases

| 场景 | 行为 |
|------|------|
| 两个 baseline 用不同 model_cfg 构造后端 | SB2 失败（参数量不同）→ `test_shared_backbone_identical_param_count` 挂红 |
| 模型类未定义 `SHARED_BACKBONE_PREFIXES` | `count_conditioning_params` AttributeError（spec 06 强制 5 模型类定义此常量）|
| 后端类名/前缀不一致（如 `self.repr_net` vs `self.rep_net`）| count 过滤失效 → conditioning_params 把 RepNet 误计入 → SB5 grep 拦截 |
| env_cfg / model_cfg 缺字段 | 上游网络类构造时报错（本模块不额外校验）|

---

## 5. Acceptance Criteria

### 5.1 单元测试（`tests/baselines/test_shared_backbones.py`）

```python
import pytest
from hyper_mve.configs import V4Config
from hyper_mve.baselines import create_baseline_model, BASELINE_REGISTRY
from hyper_mve.baselines.shared_backbones import (
    create_rep_net, create_belief_net, create_tri_context_encoder,
)
from hyper_mve.models import HyperMuZeroModel


@pytest.fixture
def cfg():
    return V4Config.from_preset("medium")


def _backbone_param_counts(model):
    return {
        "rep_net": sum(p.numel() for p in model.rep_net.parameters()),
        "belief_net": sum(p.numel() for p in model.belief_net.parameters()),
        "tri_context_encoder": sum(p.numel() for p in model.tri_context_encoder.parameters()),
    }


# ====== C6-FAIR2: 三后端跨 variant 参数量逐位相同 (SB2) ======

def test_shared_backbone_identical_param_count(cfg):
    """5 baseline + hyper 的 RepNet/BeliefNet/TriCtx 参数量逐位相同."""
    hyper = HyperMuZeroModel(cfg)
    ref = _backbone_param_counts(hyper)

    for variant in BASELINE_REGISTRY:
        model = create_baseline_model(cfg, variant)
        counts = _backbone_param_counts(model)
        assert counts == ref, (
            f"{variant} backbone params {counts} != hyper {ref} "
            f"(SB2 公平性核心崩, 断言 B 失效)"
        )


# ====== SB3: 独立实例 (非共享同一 nn.Module) ======

def test_backbones_are_distinct_instances(cfg):
    """两个 baseline 的后端是独立实例 (梯度不共享)."""
    m1 = create_baseline_model(cfg, "input_wide")
    m2 = create_baseline_model(cfg, "input_deep")
    assert m1.rep_net is not m2.rep_net
    assert m1.belief_net is not m2.belief_net
    assert m1.tri_context_encoder is not m2.tri_context_encoder
    # 参数对象也不共享
    p1 = next(m1.rep_net.parameters())
    p2 = next(m2.rep_net.parameters())
    assert p1.data_ptr() != p2.data_ptr()


# ====== SB5: 统一命名前缀 ======

def test_shared_backbone_prefixes_defined(cfg):
    """5 模型类 + hyper 均定义 SHARED_BACKBONE_PREFIXES, 且属性存在."""
    models = [HyperMuZeroModel(cfg)] + [
        create_baseline_model(cfg, v) for v in BASELINE_REGISTRY
    ]
    for m in models:
        assert hasattr(m, "SHARED_BACKBONE_PREFIXES")
        for prefix in m.SHARED_BACKBONE_PREFIXES:
            assert any(n.startswith(prefix) for n, _ in m.named_parameters()), (
                f"{type(m).__name__} 缺前缀 {prefix} 对应参数"
            )


# ====== no_belief 仍构造完整 BeliefNet (3.1) ======

def test_no_belief_still_builds_full_belief_net(cfg):
    """no_belief 的 BeliefNet 参数量与 hyper 相同 (移除信息通路, 非参数)."""
    hyper = HyperMuZeroModel(cfg)
    no_belief = create_baseline_model(cfg, "no_belief")
    assert (
        sum(p.numel() for p in no_belief.belief_net.parameters())
        == sum(p.numel() for p in hyper.belief_net.parameters())
    )
```

### 5.2 契约一致性 grep（Day 6）

- 5 模型类 `__init__` 均经 `create_rep_net` / `create_belief_net` / `create_tri_context_encoder` 构造后端（不直接 `RepresentationNet(...)`，保证单点构造）。
- 三后端实例化命名统一 `self.rep_net` / `self.belief_net` / `self.tri_context_encoder`（SB5）。

---

## 6. Cross-references

- [`03-input-conditioned-baselines.md`](./03-input-conditioned-baselines.md)（消费 backbone + 加宽/加深功能网）
- [`04-ma-muzero-baseline.md`](./04-ma-muzero-baseline.md)（消费 backbone + 共享 RewardHead）
- [`05-belief-type-ablation-baselines.md`](./05-belief-type-ablation-baselines.md)（`no_belief` 置零 belief 路 + `rewardhead_explicit_type`）
- [`06-model-7api-conformance.md`](./06-model-7api-conformance.md)（`SHARED_BACKBONE_PREFIXES` 常量 + 7-API 接入 backbone）
- [`08-integration-contracts.md`](./08-integration-contracts.md)（等参公平约束 + Pkg-05 复用契约）
- Pkg-03 spec 08 §6（RepNet/BeliefNet/TriContextEncoder 共享契约）
- Pkg-04 spec 08 §5（baseline 共享后端草案）
- Ch5 §5.6 step1（等参统计协议：只算条件化子系统）
