# Spec 01: Baselines Factory & Registry — `create_baseline_model(cfg, variant)`

> 父文档：[`../proposal.md`](../proposal.md) §2.1.1 · [`../design.md`](../design.md) §3.3 · §4 D2/D3/D10 · §7.2
> **断言覆盖**：本 spec 是 5 个 baseline（断言 A/B/C）的统一入口。
> **上游锁定**：import 路径 + 签名由 Pkg-05 spec 08 §3.1（line 136）锁定，本 spec 不可变更。

---

## ⚠️ 头部强制声明（两处 drift 锁死）

### 声明 1：supersede Pkg-04 spec 08 §5.1 旧草案（修复 2）

> 本 spec 的 `create_baseline_model(cfg, variant)` 是 baseline 工厂的**最终契约**（依据 Pkg-05 spec 08 §3.1 line 136）。
> Pkg-04 spec 08 §5.1（lines 270-285）出现的 `create_input_wide_baseline(cfg, total_params_target)`（每 variant 一函数）是 **illustrative 草案**，被本 spec **supersede**。
> Day 6 grep 须确认实施只采用单一 `create_baseline_model` 签名（否则两套签名都会命中）。

### 声明 2：CLI ↔ 工厂 ↔ 模型类映射表（修复 1：前缀分裂）

Pkg-05 spec 08 §6.2 CLI 用 `baseline_` 前缀；§3.1 工厂传参**无**前缀。两者非冲突，是**同一对象的两层命名**。本表锁死，避免 Pkg-08 调 train_main.py 再踩 drift：

| CLI `--variant` | 工厂传参（`create_baseline_model` 第 2 参）| 模型类 | 断言 | 归属 spec |
|-----------------|------|--------|------|-----------|
| `baseline_input_wide` | `"input_wide"` | `InputWideBaselineModel` | B | spec 03 |
| `baseline_input_deep` | `"input_deep"` | `InputDeepBaselineModel` | B | spec 03 |
| `baseline_ma_muzero` | `"ma_muzero"` | `MAMuZeroBaselineModel` | A | spec 04 |
| `no_belief` | `"no_belief"` | `NoBeliefBaselineModel` | C / Abl7 | spec 05 |
| `rewardhead_explicit_type` | `"rewardhead_explicit_type"` | `ExplicitTypeRewardBaselineModel` | A / Abl6.x | spec 05 |
| `hyper` / `oracle_only` / `infer_only` | —（**不进工厂**）| `HyperMuZeroModel` + curriculum override | A(主线)/B/C | — |

> CLI 前缀剥离由 Pkg-08 train_main.py 负责（`variant.removeprefix("baseline_")` 后传工厂）；`no_belief` / `rewardhead_explicit_type` CLI 无前缀，直接传。本 spec 工厂只认**无前缀**的 5 个 variant 字符串。

---

## 1. Purpose

按 Pkg-05 spec 08 §3.1 提供 v4 统一 baseline 模型工厂，将 Pkg-06 的全部"变化"压缩到一个工厂函数 + 一份 registry：

| API | 用途 | 调用频率 |
|-----|------|---------|
| `create_baseline_model(cfg, variant)` | 5 variant 统一工厂；reject "hyper" | 每个 baseline run 一次 |
| `BASELINE_REGISTRY`（模块级 dict）| variant 字符串 → 模型类映射 | 工厂内部查表 |

工厂被两处消费：
- **Pkg-05** `create_trainer_for_baseline` 内部（spec 08 §6.2）：`if variant=="hyper": HyperMuZeroModel(cfg) else: create_baseline_model(cfg, variant)`，随后 `MuZeroTrainer(cfg, model)`（同一 trainer 实例）。
- **Pkg-08** train_main.py：`--variant baseline_*` 端到端。

---

## 1.1 baseline cfg 字段穷举表（4 项，M4：避免事后修改）

工厂 + 5 模型类完整依赖以下 cfg 字段；**任何字段变更须 Pkg-01 spec 05 同步声明**（消费态声明，本包不改上游）：

### cfg.model.* （4 项新增 — 待 Pkg-01 spec 05 同步）

| 字段 | 默认 | 消费者 variant | 用途 |
|------|------|---------------|------|
| `baseline_wide_hidden_dim` | 待定（逼近 hyper 参数量后锁定）| `input_wide` | 加宽功能网隐层宽度（concat ctx_aug 后）|
| `baseline_deep_layers` | 待定 | `input_deep` | 加深功能网层数 |
| `baseline_ma_muzero_share_pred_head` | `True` | `ma_muzero` | 是否共享 pred 头（与 reward 头一致 vanilla）|
| `baseline_explicit_type_branches` | `= cfg.env.num_types` | `rewardhead_explicit_type` | RewardHead 显式 type 分支数 |

> **无 hard-coded 默认值约束（C6-CFG1）**：工厂构造 5 模型类时，上述字段必须**从 cfg 读取**，不得在工厂/模型类内写死。`input_wide`/`input_deep` 的宽度/层数是等参逼近的可调旋钮（spec 07 双阈值迭代），写死会让等参验证失效。具名单测 `test_factory_reads_baseline_cfg_fields` 验证：篡改 cfg 字段后，对应模型的条件化子系统参数量随之变化。

### 上游已有 cfg 字段（工厂/模型类只读，不新增）

| 字段 | 来源 | 用途 |
|------|------|------|
| `cfg.env.N` / `cfg.env.A` | EnvConfig | 功能网 input/output shape |
| `cfg.env.num_types` | EnvConfig | explicit_type 分支数默认值 |
| `cfg.model.latent_dim` | ModelConfig | encode 输出维度（与 hyper 一致）|
| `cfg.train.belief_grad_gating_steps` | TrainConfig | 5 variant 共用的 gating 阈值（=5000，spec 06）|

---

## 2. Interface

### 2.1 文件路径

`hyper_mve/baselines/__init__.py`（纯新增，import 路径由 Pkg-05 spec 08 line 136 锁定）

### 2.2 工厂签名（逐字对齐 Pkg-05 spec 08 §3.1）

```python
from hyper_mve.configs import V4Config
from hyper_mve.baselines.input_conditioned import (
    InputWideBaselineModel, InputDeepBaselineModel,
)
from hyper_mve.baselines.ma_muzero import MAMuZeroBaselineModel
from hyper_mve.baselines.no_belief import NoBeliefBaselineModel
from hyper_mve.baselines.explicit_type_reward import ExplicitTypeRewardBaselineModel


# variant 字符串 → 模型类（registry, 模块级）
BASELINE_REGISTRY: dict[str, type] = {
    "input_wide":                InputWideBaselineModel,        # 断言 B (spec 03)
    "input_deep":                InputDeepBaselineModel,        # 断言 B (spec 03)
    "ma_muzero":                 MAMuZeroBaselineModel,         # 断言 A (spec 04)
    "no_belief":                 NoBeliefBaselineModel,         # 断言 C / Abl7 (spec 05)
    "rewardhead_explicit_type":  ExplicitTypeRewardBaselineModel,  # 断言 A / Abl6.x (spec 05)
}


def create_baseline_model(cfg: V4Config, variant: str):
    """统一 baseline 模型工厂.

    Args:
        cfg:     V4Config (从 preset 加载).
        variant: baseline 标识 (无 baseline_ 前缀), 取值见 BASELINE_REGISTRY.

    Returns:
        baseline 模型实例 (实现 Pkg-04 spec 02 完整 7-API, 见 spec 06).

    Raises:
        ValueError: 当 variant == "hyper"
            ('hyper' 走 HyperMuZeroModel 直接构造, 不走本工厂; 见 design D3).
        ValueError: 当 variant 不在 BASELINE_REGISTRY 内 (列出合法 variant).
    """
    if variant == "hyper":
        raise ValueError(
            "'hyper' uses HyperMuZeroModel directly, not this factory. "
            "Call HyperMuZeroModel(cfg) instead (see Pkg-05 spec 08 §6.2)."
        )
    if variant not in BASELINE_REGISTRY:
        raise ValueError(
            f"Unknown baseline variant '{variant}'. "
            f"Valid variants: {sorted(BASELINE_REGISTRY)}. "
            f"(For 'hyper'/'oracle_only'/'infer_only' use HyperMuZeroModel + curriculum override.)"
        )
    model_cls = BASELINE_REGISTRY[variant]
    return model_cls(cfg)
```

### 2.3 上游调用契约（Pkg-05 spec 08 §6.2，本 spec 仅引用不重定义）

```python
# Pkg-05 create_trainer_for_baseline 内部（本包不实现, 仅对账）:
def create_trainer_for_baseline(cfg, variant):
    if variant == "hyper":
        model = HyperMuZeroModel(cfg)
    else:
        model = create_baseline_model(cfg, variant)   # ← 本 spec 提供
    return MuZeroTrainer(cfg, model)                   # ★ 同一 trainer 实例（五条复用约束, spec 08）
```

> oracle_only / infer_only 也走 `HyperMuZeroModel(cfg)` 分支 + curriculum cfg override（`curriculum_stage_1_end_frac` = 1.0 / 0.0），同样不进工厂。

---

## 3. Implementation Notes

### 3.1 registry 模式（vs if-elif 链）

用模块级 dict `BASELINE_REGISTRY` 而非 if-elif 链：
- 新增 variant 只改 registry 一行 + 对应模型类文件，工厂主体不动
- `sorted(BASELINE_REGISTRY)` 直接给出合法 variant 列表（错误信息友好）
- `test_create_baseline_model_all_5_variants` 可直接 `for v in BASELINE_REGISTRY` 遍历

### 3.2 工厂不做参数对齐（职责分离）

工厂只负责"按 variant 构造模型类"。等参逼近（调 `baseline_wide_hidden_dim` / `baseline_deep_layers` 让 input baseline 参数量贴近 hyper）是 **spec 07 的协议** + cfg 配置职责，**不在工厂内**。工厂忠实读 cfg 字段构造，参数对齐结果由 spec 07 单测 `test_baseline_param_count_within_5pct` 验收。

### 3.3 reject "hyper" 的语义（D3）

`hyper` 不是 baseline 而是主方法。工厂显式 `raise ValueError` 而非兜底返回 `HyperMuZeroModel`：
- 兜底会让"5 variant"语义膨胀成 6，破坏 design §3.1 framing
- 调用方误传 "hyper" 应快速失败（fail-fast），而非静默走主方法路径

### 3.4 5 模型类 import 来源（文件组织）

| 模型类 | 文件 | spec |
|--------|------|------|
| `InputWideBaselineModel` / `InputDeepBaselineModel` | `baselines/input_conditioned.py` | spec 03 |
| `MAMuZeroBaselineModel` | `baselines/ma_muzero.py` | spec 04 |
| `NoBeliefBaselineModel` | `baselines/no_belief.py` | spec 05 |
| `ExplicitTypeRewardBaselineModel` | `baselines/explicit_type_reward.py` | spec 05 |

5 模型类的 7-API 一致性由 spec 06 统一规范；各自内部差异由 spec 03/04/05 详化。

---

## 4. Edge Cases

| 场景 | 行为 |
|------|------|
| `variant == "hyper"` | `raise ValueError`（指引用 HyperMuZeroModel）|
| `variant == "oracle_only"` / `"infer_only"` | 不在 registry → `raise ValueError`（指引用 HyperMuZeroModel + curriculum override）|
| `variant == "baseline_input_wide"`（含 CLI 前缀）| 不在 registry → `raise ValueError`（前缀剥离是 Pkg-08 train_main.py 职责，工厂只认无前缀）|
| `variant` 大小写错误（如 `"Input_Wide"`）| 不在 registry → `raise ValueError`（registry key 严格小写下划线）|
| cfg 缺 `baseline_wide_hidden_dim` 等新字段 | 模型类构造时 AttributeError（M4 待 Pkg-01 同步；实施前补 cfg）|
| `variant == "mappo"` / `"qmix"` | 不在 registry → `raise ValueError`（异范式出范围，design NG4）|

---

## 5. Acceptance Criteria

### 5.1 单元测试（`tests/baselines/test_factory.py`）

```python
import pytest
from hyper_mve.configs import V4Config
from hyper_mve.baselines import create_baseline_model, BASELINE_REGISTRY


@pytest.fixture
def cfg():
    return V4Config.from_preset("medium")


# ====== C6-FACT1: 5 variant 均可实例化 ======

def test_create_baseline_model_all_5_variants(cfg):
    """registry 内 5 个 variant 均可经工厂构造."""
    assert set(BASELINE_REGISTRY) == {
        "input_wide", "input_deep", "ma_muzero",
        "no_belief", "rewardhead_explicit_type",
    }
    for variant in BASELINE_REGISTRY:
        model = create_baseline_model(cfg, variant)
        assert model is not None
        # 7-API 一致性由 spec 06 test_baseline_implements_full_7api 验收


# ====== C6-FACT2: 工厂收 "hyper" 抛 ValueError ======

def test_factory_rejects_hyper_variant(cfg):
    """'hyper' 不走工厂 (design D3)."""
    with pytest.raises(ValueError, match="HyperMuZeroModel directly"):
        create_baseline_model(cfg, "hyper")


def test_factory_rejects_oracle_infer_only(cfg):
    """oracle_only / infer_only 也不走工厂 (走 curriculum override)."""
    for variant in ("oracle_only", "infer_only"):
        with pytest.raises(ValueError, match="HyperMuZeroModel"):
            create_baseline_model(cfg, variant)


def test_factory_rejects_cli_prefixed_variant(cfg):
    """工厂只认无前缀 variant; CLI 前缀剥离是 Pkg-08 职责."""
    with pytest.raises(ValueError, match="Unknown baseline variant"):
        create_baseline_model(cfg, "baseline_input_wide")


def test_factory_rejects_unknown_variant(cfg):
    """异范式 / 拼写错误 → ValueError + 列合法 variant."""
    for variant in ("mappo", "qmix", "Input_Wide"):
        with pytest.raises(ValueError, match="Unknown baseline variant"):
            create_baseline_model(cfg, variant)


# ====== C6-CFG1: 工厂消费 4 个 baseline cfg 字段, 无 hard-coded 默认值 ======

def test_factory_reads_baseline_cfg_fields(cfg):
    """篡改 cfg 字段后, 对应模型条件化子系统参数量随之变化 (证明无写死)."""
    from hyper_mve.baselines.shared_backbones import count_conditioning_params  # spec 02 / 07

    # input_wide: 加宽宽度翻倍 → 条件化子系统参数量应增大
    base = create_baseline_model(cfg, "input_wide")
    n_base = count_conditioning_params(base)

    cfg_wide = V4Config.from_preset("medium")
    cfg_wide.model.baseline_wide_hidden_dim = cfg.model.baseline_wide_hidden_dim * 2
    wider = create_baseline_model(cfg_wide, "input_wide")
    n_wider = count_conditioning_params(wider)
    assert n_wider > n_base, "baseline_wide_hidden_dim 未被消费 (疑似写死)"

    # input_deep: 加深层数 +2 → 参数量应增大
    cfg_deep = V4Config.from_preset("medium")
    cfg_deep.model.baseline_deep_layers = cfg.model.baseline_deep_layers + 2
    deeper = create_baseline_model(cfg_deep, "input_deep")
    assert count_conditioning_params(deeper) > count_conditioning_params(
        create_baseline_model(cfg, "input_deep")
    ), "baseline_deep_layers 未被消费 (疑似写死)"
```

### 5.2 契约一致性 grep（Day 6）

- 工厂签名 `create_baseline_model(cfg, variant)` 与 Pkg-05 spec 08 §3.1 line 136 逐字一致。
- grep 确认全仓**无** `create_input_wide_baseline` / `create_*_baseline(` 每-variant 函数残留（声明 1 supersede）。

---

## 6. Cross-references

- [`02-shared-backbones.md`](./02-shared-backbones.md)（5 模型类共享后端工厂）
- [`03-input-conditioned-baselines.md`](./03-input-conditioned-baselines.md)（`input_wide` / `input_deep` 模型类）
- [`04-ma-muzero-baseline.md`](./04-ma-muzero-baseline.md)（`ma_muzero` 模型类）
- [`05-belief-type-ablation-baselines.md`](./05-belief-type-ablation-baselines.md)（`no_belief` / `rewardhead_explicit_type` 模型类）
- [`06-model-7api-conformance.md`](./06-model-7api-conformance.md)（5 模型类 7-API 一致性）
- [`08-integration-contracts.md`](./08-integration-contracts.md)（五条复用约束 + Pkg-05/08 契约）
- Pkg-05 spec 08 §3.1（工厂签名 line 136）+ §6.2（--variant CLI 语义 + create_trainer_for_baseline）
- Pkg-04 spec 08 §5.1（旧草案 `create_input_wide_baseline`，被本 spec supersede）
- Pkg-01 spec 05 TrainConfig（4 个 baseline cfg 字段待同步声明）
