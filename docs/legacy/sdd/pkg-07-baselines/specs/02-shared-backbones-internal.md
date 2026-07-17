# Spec 02: Shared Backbones (Internal) — 5 internal variant 的等参公平结构基石

> 父文档：[`../proposal.md`](../proposal.md) · [`../design.md`](../design.md) §D4 + §3.2 · [`../README.md`](../README.md) C7-INT-FAIR2
> **断言覆盖**：等参公平性是断言 B′ 的生死线（design §1.2 + spec 07 §双层 fairness）。本 spec 是 **Internal namespace** 公平性的**结构基石**。
> **上游对齐**：RepNet/BeliefNet/TriContextEncoder 共享契约逐字对齐 Pkg-03 spec 08 §6 + Pkg-04 spec 02 §3.4。
> **Supersede 锚**：本 spec 内容继承 pkg-06 spec 02 `02-shared-backbones.md` **逐字**，仅作两项扩展：(1) 适用范围从 "5 baseline + hyper" 缩窄为 "**5 internal variant** + hyper"（external namespace 不消费 shared_backbones，design D4 + README L76）；(2) C6-FAIR2 重号为 C7-INT-FAIR2。pkg-06 spec 02 在 supersede 后由本 spec 替代，pkg-06 README banner 已标记。

---

## 1. Purpose 与 supersede 边界

### 1.1 Purpose

提供 `hyper_mve/baselines/shared_backbones.py`，让 **5 个 internal baseline variant** + hyper **各自实例化结构与参数量完全相同的 RepNet / BeliefNet / TriContextEncoder**，从而把 internal namespace 内的模型间差异收敛到唯一变量——**条件化消费子系统**（功能网 + 生成器；spec 07 D5 定义）。

5 internal variant 全 keys 共享后端契约（design D4 锚 + 本 spec §10 cross-ref pkg-07 spec 01 §10）：

| variant key | 工厂 arg | 模型类 | shared backbones 消费 |
|-------------|---------|--------|----------------------|
| `input_wide` | `"input_wide"` | `InputWideBaselineModel` | ✅（spec 03）|
| `input_deep` | `"input_deep"` | `InputDeepBaselineModel` | ✅（spec 03）|
| `ma_muzero` | `"ma_muzero"` | `MAMuZeroBaselineModel` | ✅（spec 03）|
| `no_belief` | `"no_belief"` | `NoBeliefBaselineModel` | ✅（spec 03，BeliefNet 仍构造，仅 ctx 路置零）|
| `rewardhead_explicit_type` | `"rewardhead_explicit_type"` | `ExplicitTypeRewardBaselineModel` | ✅（spec 03）|

| API | 用途 | 调用者 |
|-----|------|--------|
| `create_rep_net(env_cfg, model_cfg)` | 构造与 hyper 同构的 RepresentationNet | 5 internal model class `__init__` |
| `create_belief_net(env_cfg, model_cfg)` | 构造与 hyper 同构的 BeliefNet | 5 internal model class（`no_belief` 仍构造）|
| `create_tri_context_encoder(env_cfg, model_cfg)` | 构造与 hyper 同构的 TriContextEncoder | 5 internal model class |
| `count_conditioning_params(model)` | 统计条件化消费子系统参数（排除共享后端）| spec 07 C7-INT-FAIR1 + spec 01 C7-INT-CFG1 |

**核心不变量**：跨 5 internal variant + hyper 三个后端的**参数量逐位相同**（C7-INT-FAIR2，对应 pkg-06 C6-FAIR2）；训练时**梯度不共享**（各自独立 `nn.Module` 实例）。

### 1.2 Supersede 边界（与 pkg-06 spec 02 的差异）

| 维度 | pkg-06 spec 02（被 supersede）| pkg-07 spec 02（本 spec）|
|------|------------------------------|--------------------------|
| 适用 variant 集合 | 5 baseline + hyper | **5 internal variant** + hyper（external namespace 不消费；design D4 + design §3.5 LL 76）|
| 测试 ID | C6-FAIR2 | **C7-INT-FAIR2**（INT 后缀显式标记 internal-only）|
| 工厂签名 | 同（3 factories + count）| 同（**逐字继承**）|
| 共享契约 5 条不变量 (SB1-SB5) | 同 | 同（**逐字继承**）|
| 等参 5% / 10% 双阈值 | 在本 spec 范围内 | **不在本 spec 范围**——由 pkg-07 spec 07 拆分（"2 vs 3 豁免"分割，README L74-75）|

### 1.3 Cross-ref 锚：pkg-07 spec 01 §10 摘录

pkg-07 spec 01 §10 lines 482-489（cross-ref 段）明确：

> "spec 02 `02-shared-backbones-internal.md`（**5 internal 全 keys 共享后端契约**：`input_wide` / `input_deep` / `ma_muzero` / `no_belief` / `rewardhead_explicit_type`，全部各自实例化 RepNet/BeliefNet/TriContextEncoder；design D4 锚；equal-param check 的「2 vs 3 豁免」分割由 spec 07 拆分，**不在 spec 02 范围**）"

本 spec §9 显式列出 "What spec 02 does NOT cover"，与 spec 01 §10 + spec 07 §双层 fairness 的边界一致。

---

## 2. Interface

### 2.1 文件路径

`hyper_mve/baselines/shared_backbones.py`（纯新增；pkg-06 spec 02 同路径草稿被本 spec 替代）

### 2.2 工厂签名（对齐 Pkg-03 spec 08 §6.2 + Pkg-04 spec 02 §3.4 构造路径）

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
    返回独立实例（5 internal variant + hyper 各持有自己的, 梯度不共享）.

    注意: 上游 Pkg-04 spec 02 §3.4 line 124 用 RepresentationNet(cfg) 签名;
    本工厂在实现期不要求 RepresentationNet 新增 (env_cfg, model_cfg) overload —
    工厂内部先组装一个临时 cfg 然后 forwards 到现有 RepresentationNet(cfg) 构造器
    (见 §8.2 实施期 hint), 对外仍只暴露 (env_cfg, model_cfg) 契约.
    """
    cfg = _assemble_cfg_for_rep_net(env_cfg, model_cfg)   # 实施期 helper, 见 §8.2
    return RepresentationNet(cfg)


def create_belief_net(env_cfg: EnvConfig, model_cfg: ModelConfig) -> BeliefNet:
    """构造与 hyper 同类同构的 BeliefNet (Pkg-03 spec 04).

    注意: no_belief variant 仍构造完整 BeliefNet (参数量对齐 hyper, C7-INT-FAIR2),
    只在 set_context_subjective 内把 belief tuple 在 TriContextEncoder 路置零 (spec 03 + 05),
    保证条件化子系统外的等价性.
    """
    return BeliefNet(env_cfg, model_cfg)


def create_tri_context_encoder(env_cfg: EnvConfig, model_cfg: ModelConfig) -> TriContextEncoder:
    """构造与 hyper 同类同构的 TriContextEncoder (Pkg-03 spec 01: 三路 c_ctx / role_i / belief_i)."""
    return TriContextEncoder(env_cfg, model_cfg)


def count_conditioning_params(model) -> int:
    """统计条件化消费子系统参数 (design §D5 + Ch5 §5.6 step1).

    包含 (按 variant 不同, design D5 账目):
        - θ 生成器 (hypernet 头, 若有: hyper_trans / hyper_rew / hyper_pred)
        - 接收 ctx_aug 的功能网 (StateTransNet / RewardHead / PredictionNet,
          input baseline 为加宽/加深版本)

    排除 (5 internal variant + hyper 共享, 不计入):
        - RepresentationNet
        - BeliefNet
        - TriContextEncoder

    实现: 遍历 model.named_parameters(), 按 model 暴露的
        model.SHARED_BACKBONE_PREFIXES = ('rep_net', 'belief_net', 'tri_context_encoder')
    前缀过滤掉共享后端, 其余即条件化子系统.
    """
    total = 0
    for name, p in model.named_parameters():
        if any(name.startswith(prefix) for prefix in model.SHARED_BACKBONE_PREFIXES):
            continue
        total += p.numel()
    return total
```

### 2.3 共享契约（5 条不变量 SB1-SB5，对齐 Pkg-03 spec 08 §6）

| # | 不变量 | 验收 |
|---|--------|------|
| SB1 | 同类：5 internal variant + hyper 用同一 `RepresentationNet` / `BeliefNet` / `TriContextEncoder` 类 | grep 实例化点 |
| SB2 | 同构：同 (env_cfg, model_cfg) 构造 → 三后端参数量逐位相同 | `test_shared_backbone_identical_param_count` |
| SB3 | 独立实例：各 variant `__init__` 各自调 `create_*`，**非**共享同一 `nn.Module` | `test_backbones_are_distinct_instances` |
| SB4 | 独立梯度：训练时各 internal variant 的后端梯度互不影响（独立 run）| 由 SB3 保证（不同 Parameter 对象）|
| SB5 | 统一命名前缀：`model.rep_net` / `model.belief_net` / `model.tri_context_encoder`（供 `count_conditioning_params` 过滤）| `SHARED_BACKBONE_PREFIXES` 常量 |

---

## 3. Implementation Notes

### 3.1 同类同构 + 独立实例 + 独立梯度（design D4 三联约束）

design.md §D4（line 228）锁定：

> "5 internal variant 各自实例化 RepNet/BeliefNet/TriContextEncoder（**同类同构、独立实例、独立梯度、参数量逐位对齐**）。External 不消费 shared_backbones。"

三联约束的工程意义：

1. **同类同构**：5 variant + hyper 持有相同 PyTorch class 的实例，且同 (env_cfg, model_cfg) 构造下 `sum(p.numel() for p in module.parameters())` **字节级相同**（C7-INT-FAIR2）。
2. **独立实例**：每个 variant 在 `__init__` 内**独立调** `create_rep_net / create_belief_net / create_tri_context_encoder`，5 个 model 各持 5 套 backbone 实例（5 × 3 = 15 个独立 `nn.Module` 对象）。
3. **独立梯度**：因独立实例 ⇒ 各 variant 的 `Parameter` 对象 ID 不同 ⇒ 反向传播时梯度不串扰；这是断言 B′ "5 internal variant 同条件下比拼"的**必要**前提（任何共享会让等参检验失效）。

### 3.2 为什么 no_belief 仍构造完整 BeliefNet（SB2 关键）

`no_belief` 打断言 C（移除 belief 路）。若它**不构造** BeliefNet，则它与 hyper 在"共享后端"层就不等价（少了 BeliefNet 参数），断言 C 的对照就混入了"参数更少"的干扰。因此 `no_belief` **构造完整 BeliefNet（参数量对齐 hyper，C7-INT-FAIR2 不放弃）**，仅在 `set_context_subjective` 内把 belief tuple 送进 TriContextEncoder 的 belief 路时置零（spec 03 + spec 05）。这样移除的是**信息通路**而非**参数**，对照才干净。

### 3.3 count_conditioning_params 的边界（对齐 design D5 + spec 07）

`count_conditioning_params` 是等参验证（spec 07 C7-INT-FAIR1 5%/10% 双阈值）与 C7-INT-CFG1（spec 01 工厂 cfg 消费）的共同度量入口，统一放本模块避免两处实现漂移。它依赖每个 internal model 类暴露 `SHARED_BACKBONE_PREFIXES = ('rep_net', 'belief_net', 'tri_context_encoder')` 常量——5 internal model 类 + `HyperMuZeroModel` 都必须定义该常量（spec 03 + Pkg-04 spec 02 同步约束）。

逐 variant 落在条件化子系统内的部件（design D5 账目）：

| variant | 计入 conditioning_params 的部件 |
|---------|-------------------------------|
| `hyper`（基准）| hyper_trans / hyper_rew / hyper_pred（3 头）+ 3 vanilla 功能网 |
| `input_wide` | 3 **加宽**功能网（无生成器）|
| `input_deep` | 3 **加深**功能网（无生成器）|
| `no_belief` | hyper_trans / hyper_rew / hyper_pred + 3 vanilla 功能网（与 hyper 同；差异在 belief 置零，非参数）|
| `ma_muzero` | 3 vanilla 功能网（共享单一 RewardHead，无生成器）|
| `rewardhead_explicit_type` | hyper_trans / hyper_pred（**无** hyper_rew）+ 带 type 分支的 RewardHead + 2 功能网 |

> **「2 vs 3 豁免」的拆分位置**：上表中 `ma_muzero` / `rewardhead_explicit_type` 的条件化子系统结构与 hyper 显著不同（少了一个 hypernet 头 / 加了 type 分支），属 spec 07 「结构性豁免」清单（README L74-75）。豁免规则**不在 spec 02 范围**：spec 02 只保证 backbone 同构 + count 函数正确，spec 07 决定哪些 variant 走严格 5%/10% 双阈值、哪些走结构性豁免。

### 3.4 不重定义网络结构（NG2 遵守）

本 spec **不实现** RepresentationNet / BeliefNet / TriContextEncoder 的网络结构——它们是 Pkg-03/04 已发布的类，本模块仅做**工厂封装**（保证 5 internal variant 用同一构造路径）。封装的价值：单点保证"同 cfg 构造"，避免某个 variant 误传不同 cfg 字段导致后端结构漂移。

---

## 4. Edge Cases

| 场景 | 行为 |
|------|------|
| 两个 internal variant 用不同 model_cfg 构造后端 | SB2 失败（参数量不同）→ `test_shared_backbone_identical_param_count` 挂红 |
| 模型类未定义 `SHARED_BACKBONE_PREFIXES` | `count_conditioning_params` AttributeError（spec 03 强制 5 internal model 类定义此常量）|
| 后端类名/前缀不一致（如 `self.repr_net` vs `self.rep_net`）| count 过滤失效 → conditioning_params 把 RepNet 误计入 → SB5 grep 拦截 |
| External variant 误调 `create_rep_net` 等工厂 | 不报错（工厂本身允许）；但 design D4 禁止 external 消费 shared_backbones（spec 04 适配器只过 env，不过 backbone）；约束由 spec 03/04/05/06 grep 守护，不在 spec 02 单测内 |
| env_cfg / model_cfg 缺字段 | 上游网络类构造时报错（本模块不额外校验，由 Pkg-01 schema 守护）|

---

## 5. Acceptance Criteria

### 5.1 单元测试（`tests/baselines/test_shared_backbones.py`）

```python
import pytest
from hyper_mve.configs import V4Config
from hyper_mve.baselines import create_baseline, INTERNAL_REGISTRY
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


# ====== C7-INT-FAIR2: 三后端跨 5 internal variant + hyper 参数量逐位相同 (SB2) ======
# parametrize: 5 internal variants × 3 sub-backbones = 15 cells (题示要求)

@pytest.mark.parametrize("variant", INTERNAL_REGISTRY)  # 5 keys
@pytest.mark.parametrize("backbone", ["rep_net", "belief_net", "tri_context_encoder"])
def test_shared_backbone_identical_param_count(cfg, variant, backbone):
    """C7-INT-FAIR2: 5 internal variant 的 RepNet/BeliefNet/TriCtx 参数量逐位与 hyper 相同.

    Parametrized 5 × 3 = 15 cells; 任一 cell 失败 → 等参公平 (断言 B′) 崩.
    """
    hyper = HyperMuZeroModel(cfg)
    ref = _backbone_param_counts(hyper)[backbone]

    model = create_baseline(cfg, variant)
    got = _backbone_param_counts(model)[backbone]
    assert got == ref, (
        f"variant={variant!r} backbone={backbone!r} params {got} != hyper {ref} "
        f"(C7-INT-FAIR2 / SB2 崩, 断言 B′ 失效)"
    )


# ====== SB3: 独立实例 (非共享同一 nn.Module) ======

def test_backbones_are_distinct_instances(cfg):
    """两个 internal variant 的后端是独立实例 (梯度不共享)."""
    m1 = create_baseline(cfg, "input_wide")
    m2 = create_baseline(cfg, "input_deep")
    assert m1.rep_net is not m2.rep_net
    assert m1.belief_net is not m2.belief_net
    assert m1.tri_context_encoder is not m2.tri_context_encoder
    # 参数对象也不共享 (Parameter ID 级独立)
    p1 = next(m1.rep_net.parameters())
    p2 = next(m2.rep_net.parameters())
    assert p1.data_ptr() != p2.data_ptr()


# ====== SB5: 统一命名前缀 ======

def test_shared_backbone_prefixes_defined(cfg):
    """5 internal model 类 + hyper 均定义 SHARED_BACKBONE_PREFIXES, 且属性存在."""
    models = [HyperMuZeroModel(cfg)] + [create_baseline(cfg, v) for v in INTERNAL_REGISTRY]
    for m in models:
        assert hasattr(m, "SHARED_BACKBONE_PREFIXES")
        assert m.SHARED_BACKBONE_PREFIXES == ("rep_net", "belief_net", "tri_context_encoder")
        for prefix in m.SHARED_BACKBONE_PREFIXES:
            assert any(n.startswith(prefix) for n, _ in m.named_parameters()), (
                f"{type(m).__name__} 缺前缀 {prefix} 对应参数"
            )


# ====== no_belief 仍构造完整 BeliefNet (3.2) ======

def test_no_belief_still_builds_full_belief_net(cfg):
    """no_belief 的 BeliefNet 参数量与 hyper 相同 (移除信息通路, 非参数)."""
    hyper = HyperMuZeroModel(cfg)
    no_belief = create_baseline(cfg, "no_belief")
    assert (
        sum(p.numel() for p in no_belief.belief_net.parameters())
        == sum(p.numel() for p in hyper.belief_net.parameters())
    )
```

### 5.2 契约一致性 grep（Day 6）

- 5 internal model 类 `__init__` 均经 `create_rep_net` / `create_belief_net` / `create_tri_context_encoder` 构造后端（不直接 `RepresentationNet(...)`，保证单点构造）。
- 三后端实例化命名统一 `self.rep_net` / `self.belief_net` / `self.tri_context_encoder`（SB5）。
- External runner（`external_mappo` / `external_qmix` / `external_ma_muzero_gh` / `external_mamba` / stubs）**不得** import `hyper_mve.baselines.shared_backbones`（design D4 禁令）；grep 由 spec 04/05/06 守护。

---

## 6. C7-INT-FAIR2 测试矩阵（参数化 5 × 3 = 15 cells）

题示要求 `test_shared_backbone_identical_param_count` parametrized over **5 internal variants × 3 sub-backbones = 15 cells**。落到 pytest 表如下：

| variant \ backbone | rep_net | belief_net | tri_context_encoder |
|--------------------|---------|------------|---------------------|
| `input_wide` | 1 cell | 1 cell | 1 cell |
| `input_deep` | 1 cell | 1 cell | 1 cell |
| `ma_muzero` | 1 cell | 1 cell | 1 cell |
| `no_belief` | 1 cell | 1 cell | 1 cell |
| `rewardhead_explicit_type` | 1 cell | 1 cell | 1 cell |

任一 cell 失败 → C7-INT-FAIR2 红 → 断言 B′ 等参公平崩。15 cells 全绿 = 5 internal variant 的 backbone 与 hyper 等参，是 spec 07 5%/10% 双阈值 fairness 检查的**结构前置**——只有 backbone 等参，才能合法地把 internal namespace 差异收敛到「条件化消费子系统」单一变量。

---

## 7. 跨 variant fairness rationale（与 spec 07 等参检查的因果关系）

### 7.1 spec 02 是结构前置，spec 07 是 fairness 判决

```
spec 02 (本 spec)            spec 07 (双层 fairness)
┌──────────────────┐         ┌────────────────────────┐
│ backbone 等参    │ ───►    │ 条件化子系统 5%/10%    │
│ (15 cells 通过)  │  保证   │ 双阈值检查             │
│ C7-INT-FAIR2     │         │ C7-INT-FAIR1           │
└──────────────────┘         └────────────────────────┘
   结构前置                      fairness 判决
```

- **没有 spec 02 的 backbone 等参** → spec 07 的 5%/10% 检查毫无意义（差异来自 backbone，不来自条件化子系统）。
- **有 spec 02 的 backbone 等参** → spec 07 可以合法宣称「差异收敛到条件化消费子系统单一变量」。

### 7.2 「2 vs 3 豁免」分割的位置

README L74-75 明确：
- `baseline_*` 前缀（`input_wide`, `input_deep`） → **2** 个 strict-equal-param variant，走 spec 07 5%/10% 双阈值；
- bare 名（`ma_muzero`, `no_belief`, `rewardhead_explicit_type`） → **3** 个 structural-exemption variant，spec 07 豁免严格等参（条件化结构刻意不同）。

**「2 vs 3 豁免」分割**完全由 spec 07 拥有；spec 02 一视同仁地对 5 个 internal variant 强制 backbone 等参（C7-INT-FAIR2 不豁免任何 variant），分割只发生在条件化子系统层面（spec 07）。

---

## 8. 与上游 Pkg-03 / Pkg-04 契约的对账

### 8.1 Pkg-03 spec 08 §6 锚（RepNet/BeliefNet/TriContextEncoder 共享契约）

Pkg-03 spec 08 §6.2 line 350 的工厂草案：

```python
from hyper_mve.models import BeliefNet, TriContextEncoder

def create_belief_net(env_cfg, model_cfg):
    return BeliefNet(env_cfg, model_cfg)

def create_tri_context_encoder(env_cfg, model_cfg):
    return TriContextEncoder(env_cfg, model_cfg)
```

本 spec §2.2 工厂签名与 Pkg-03 spec 08 §6.2 **逐字对齐**（含 `from hyper_mve.models import ...` 包级 re-export 而非直引子模块，避免内部重构破坏下游）。

### 8.2 Pkg-04 spec 02 §3.4 锚（HyperMuZeroModel RepNet 构造）

Pkg-04 spec 02 line 124 `self.rep_net = RepresentationNet(cfg)`。本 spec §2.2 `create_rep_net(env_cfg, model_cfg)` 是 thin wrap，下游消费方（5 internal model 类 + `HyperMuZeroModel`）只调本工厂，避免 (env_cfg, model_cfg) vs cfg 两种签名漂移。**实施期 hint**：工厂内部用一个 `_assemble_cfg_for_rep_net(env_cfg, model_cfg)` helper 构造临时 `cfg` 对象（仅包含 RepresentationNet 实际读到的 sub-fields），然后 forwards 到现有 `RepresentationNet(cfg)` 构造器。这样 RepresentationNet 类**不需要新增** (env_cfg, model_cfg) overload —— Pkg-04 spec 02 签名保持不动。

### 8.3 Pkg-04 spec 04 BeliefGradGating 与本 spec 的边界

Pkg-04 spec 04 + 本 pkg spec 03 共拥有 `BeliefGradGating`（pre-5K BeliefNet 梯度=0）；BeliefGradGating 操作 BeliefNet 输出 tensor 的 `.detach()`，**不**修改 BeliefNet 内部参数。本 spec 只保证 BeliefNet **结构**等同 + 实例独立；梯度门控由 spec 03 + Pkg-04 spec 04 拥有，不在本 spec 范围。

---

## 9. What spec 02 does NOT cover

为防止 spec 漂移，本节显式列出**不**由 spec 02 拥有的契约：

| 议题 | 拥有 spec | 备注 |
|------|----------|------|
| 等参 5% warn / 10% fail 双阈值判决 | **spec 07** | spec 02 只提供 `count_conditioning_params`，判决在 spec 07 |
| 「2 vs 3 豁免」分割 (`baseline_*` 严格 vs bare-name 豁免) | **spec 07** + README L74-75 | spec 02 一视同仁强制 backbone 等参 |
| BeliefGradGating (pre-5K 梯度=0) | **spec 03** + Pkg-04 spec 04 | 操作 BeliefNet 输出 tensor，不动 backbone 等参结构 |
| 5 internal model 类的具体实现（forward / set_context / 7-API）| **spec 03** | spec 02 只暴露 3 工厂，5 model 类如何消费在 spec 03 |
| External namespace fairness（披露式：params + walltime + LR-best return）| **spec 07** | external 不消费 shared_backbones，不在 spec 02 范围 |
| `cfg.baselines.*` 字段（`internal_wide_hidden_dim` 等）| **spec 01** + design D10 | spec 02 工厂不读 `cfg.baselines.*`，等参逼近旋钮由 spec 03 model 类消费 |
| PettingZoo 适配器（`ResourceCommonsPettingZooEnv`）| **spec 04** | external-only，与 backbone 无关 |

---

## 10. Integration hooks（5 internal model class `__init__` 接入）

5 internal model class（spec 03 实现）的 `__init__` 严格按下面模板消费本 spec 3 工厂：

```python
# spec 03 任一 internal model class __init__ 内部 (template)
from hyper_mve.baselines.shared_backbones import (
    create_rep_net, create_belief_net, create_tri_context_encoder,
)

class InputWideBaselineModel(nn.Module):
    SHARED_BACKBONE_PREFIXES = ("rep_net", "belief_net", "tri_context_encoder")  # SB5 强制

    def __init__(self, cfg: V4Config):
        super().__init__()
        self.cfg = cfg
        # 三后端经工厂构造 (SB1 同类, SB2 同构, SB3 独立实例, SB4 独立梯度)
        self.rep_net = create_rep_net(cfg.env, cfg.model)
        self.belief_net = create_belief_net(cfg.env, cfg.model)
        self.tri_context_encoder = create_tri_context_encoder(cfg.env, cfg.model)
        # 条件化消费子系统 (spec 03 各 variant 自行实现, 此处仅占位注释)
        # self.functional_nets = ...  ← 不进 shared_backbones, 各 variant 不同
```

`HyperMuZeroModel`（Pkg-04 spec 02）同步接入 `SHARED_BACKBONE_PREFIXES` 常量 + 经本工厂构造 backbone（pkg-06 spec 02 已锁定，pkg-07 继承不变）。

---

## 11. Cross-references

### 上游锁定（pkg-07 内）
- **pkg-07 design.md §D4** (line 228) — 5 internal variant 各自实例化 backbone + 同类同构 + 独立实例 + 独立梯度 + 参数量逐位对齐
- **pkg-07 design.md §3.2** (line 100-117) — 11-key 工厂矩阵中的 5 internal 行
- **pkg-07 spec 01 §10** (lines 482-489) — cross-ref 段明确「5 internal 全 keys 共享后端契约 + 2 vs 3 豁免不在 spec 02 范围」
- **pkg-07 spec 01 §2** — `INTERNAL_REGISTRY` 5 keys（被本 spec §5.1 参数化测试消费）
- **pkg-07 README** §C7-INT-FAIR2 (line 40) — "RepNet/BeliefNet/TriCtx 跨 internal variant 参数量逐位相同" 顶层验收点

### 下游消费（pkg-07 内）
- **spec 03** `03-internal-variants.md` — 5 internal model class `__init__` 消费 3 工厂（本 spec §10 模板）+ 实现 7-API + 等参账目（design D5）+ `SHARED_BACKBONE_PREFIXES` 常量定义
- **spec 07** `07-fairness-protocol.md` — 双层 fairness：(a) Internal 严格 5%/10% 双阈值（消费 `count_conditioning_params`）+ (b) 2 vs 3 豁免分割（baseline_* 严格 / bare 名豁免）+ (c) External 披露式
- **spec 08** `08-integration-contracts.md` — 对外硬契约：`shared_backbones.py` API 稳定 + `SHARED_BACKBONE_PREFIXES` 常量纳入 7-API 外的附加约定

### 上游锁定（跨包）
- **Pkg-03 spec 08 §6** (`08-integration-contracts.md` lines 340-377) — RepNet/BeliefNet/TriContextEncoder 共享契约 + `create_belief_net` / `create_tri_context_encoder` 工厂草案（本 spec §2.2 逐字对齐）
- **Pkg-04 spec 02 §3.4** (`02-hyper-muzero-model-v2.md` lines 123-130, 434-439) — `HyperMuZeroModel` `__init__` 内 `self.rep_net = RepresentationNet(cfg)` + `self.belief_net = BeliefNet(cfg.env, cfg.model)` + `self.tri_context_encoder = TriContextEncoder(cfg.env, cfg.model)` 构造路径
- **Pkg-04 spec 04** `04-belief-gradient-gating.md` — BeliefGradGating（与本 spec 互补，不冲突）

### Supersede 锚
- **pkg-06 spec 02** `02-shared-backbones.md` — 本 spec 内容**逐字继承** + 扩展（5 internal 缩窄 + C6-FAIR2 重号为 C7-INT-FAIR2 + 「2 vs 3 豁免」明确归 spec 07）；pkg-06 README banner 已标记 supersede

### Ch5 论文锚
- **Ch5 §5.6 step1** — 等参统计协议：只算条件化子系统（本 spec `count_conditioning_params` 是该协议的代码代理）
