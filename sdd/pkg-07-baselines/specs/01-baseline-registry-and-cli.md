# Spec 01: Baseline Registry & CLI — `create_baseline(cfg, variant)` (pkg-07)

> 父文档：[`../proposal.md`](../proposal.md) §2.1.1 · [`../design.md`](../design.md) §3.2 / §3.3 / §D2 / §D3 / §D10
> **断言覆盖**：本 spec 是 14 行 CLI 字串（11 in-registry + 3 curriculum-override）的统一入口；同时承载 internal (断言 A/B′/C) 与 external (横向坐标) 两 namespace 的分派。
> **上游锁定**：import 路径 `hyper_mve/baselines/__init__.py` + factory 名 `create_baseline` 由 design.md §D2 锁定；本 spec 不可变更。

---

## ⚠️ 头部强制声明（hard-gate 两处 supersede 锁死）

### 声明 1：supersede pkg-06 spec 01（rename + alias-with-DeprecationWarning）

> 本 spec 的 `create_baseline(cfg, variant)` 是 baseline 工厂的**最终契约**。
> pkg-06 spec 01 `create_baseline_model(cfg, variant)` 是 internal-only 旧契约，被本 spec **supersede**。
> 旧名 `create_baseline_model` 保留为 alias，加 `DeprecationWarning(stacklevel=2)`，**一个 release 的过渡窗口**后移除（design D2）。
> Day 6 grep 须确认实施只采用单一 `create_baseline` 签名（alias 仅在 `baselines/__init__.py` 同文件内残留一行）。

### 声明 2：supersede pkg-04 spec 08 §5.1 旧草案（per-variant 函数清扫）

> pkg-04 spec 08 §5.1（lines 270-285）列的 `create_input_wide_baseline(cfg, total_params_target)` 等**每 variant 一函数**草案，被本 spec **supersede**：实施期只接受**单一** `create_baseline(cfg, variant)` 签名。
> Day 6 grep 须确认全仓**无** `create_input_wide_baseline` / `create_*_baseline(` 每 variant 函数残留。

### 声明 3：14 行 CLI 表对账锚点（design §3.3 + train_main.py:45-52）

> design.md §3.3 锁定 CLI ↔ factory-arg ↔ model-class 三列映射表 14 行（11 in-registry + 3 curriculum-override）。
> `hyper_mve/scripts/train_main.py:45-52` `_DEFERRED_VARIANTS` 是**部分 ground truth**（其 6 条 CLI 字串：`oracle_only` / `infer_only` / `baseline_input_wide` / `baseline_input_deep` / `baseline_ma_muzero` / `no_belief` 与本 spec §3 表的对应行**逐字一致**；前缀 asymmetry 由该常量验证）。剩余 8 条（`hyper`/`rewardhead_explicit_type`/6 个 `external_*`）由 spec 08 「3 处下游补丁声明」中的 train_main.py CLI 扩展引入；扩展后整张 14 行表才完整对齐 `_DEFERRED_VARIANTS ∪ 新增 --variant choices`。
> 本 spec §3 表与 design §3.3 14-行表**逐字一致**；任何漂移由 `test_registry_keys_equal_cli_choices_and_factory_args` 一次性 catch。

---

## 1. Purpose

按 design.md §D2 / §D3 提供 v4 统一 baseline 工厂，将 pkg-06 5 internal variant + pkg-07 新增 3 Tier-1 external + 1 Tier-2 (MAMBA, if sourced) + 2 stubs (MARIE/GA) 压缩到**一个工厂函数 + 两份 namespace registry**（合并视图供 pkg-08 消费）：

| API | 用途 | 调用频率 |
|-----|------|---------|
| `create_baseline(cfg, variant)` | 11-key REGISTRY 统一工厂；reject "hyper"；stubs 抛 `NotImplementedError` | 每个 baseline run 一次 |
| `INTERNAL_REGISTRY` (模块级 `MappingProxyType`) | 5 internal variant 字串 → 模型类映射 | 工厂内部查表 |
| `EXTERNAL_REGISTRY` (模块级 `MappingProxyType`) | 6 external variant 字串 → runner 类映射（含 MAMBA + 2 stubs） | 工厂内部查表 |
| `REGISTRY` (合并 `MappingProxyType`) | `INTERNAL_REGISTRY ∪ EXTERNAL_REGISTRY` 只读视图 | pkg-08 sweep harness 枚举 |
| `cli_to_factory_arg(cli: str) -> str` | CLI 字串 → factory arg 的派生函数（前缀 asymmetry 单点处理） | train_main.py 调用前转换 |

工厂被两处消费：
- **pkg-08** unified evaluator + sweep harness：`for v in REGISTRY: create_baseline(cfg, v).evaluate(...)`。
- **train_main.py**：`--variant <CLI>` → `cli_to_factory_arg(CLI)` → `create_baseline(cfg, factory_arg)`（hyper / oracle_only / infer_only 三个 curriculum-override 走 `HyperMuZeroModel(cfg)` 直接构造分支，不进工厂）。

---

## 2. REGISTRY 契约 — 11 keys 全枚举

> 本节锁死 design.md §3.2 11-key 工厂矩阵 + §3.3 14-行 CLI 表的「in-registry 11 行」。3 行 curriculum-override（`hyper` / `oracle_only` / `infer_only`）**不在 registry 内**，由 train_main.py 直接 dispatch 到 `HyperMuZeroModel(cfg) + curriculum cfg override`。

### 2.1 14-行三列映射表（in-registry 11 + curriculum-override 3）

| `--variant` CLI 字串 | `create_baseline(cfg, ?)` 第 2 参 | 模型类 / 运行体 | 断言 / 用途 | namespace |
|---|---|---|---|---|
| `hyper` | — (**不进工厂**；`HyperMuZeroModel(cfg)` 直接构造) | `HyperMuZeroModel` | A / B′ / C 主线 | curriculum-override |
| `oracle_only` | — (**不进工厂**；`HyperMuZeroModel(cfg)` + `curriculum_stage_1_end_frac=1.0`) | `HyperMuZeroModel` (curriculum override) | regret 上界 | curriculum-override |
| `infer_only` | — (**不进工厂**；`HyperMuZeroModel(cfg)` + `curriculum_stage_1_end_frac=0.0`) | `HyperMuZeroModel` (curriculum override) | 零样本 belief 推断 | curriculum-override |
| `baseline_input_wide` | `"input_wide"` | `InputWideBaselineModel` | B′ 等参 (internal, **有 `baseline_` 前缀**) | INTERNAL |
| `baseline_input_deep` | `"input_deep"` | `InputDeepBaselineModel` | B′ 等参 (internal, **有 `baseline_` 前缀**) | INTERNAL |
| `baseline_ma_muzero` | `"ma_muzero"` | `MAMuZeroBaselineModel` | A 共享头 (internal, **有 `baseline_` 前缀**) | INTERNAL |
| `no_belief` | `"no_belief"` | `NoBeliefBaselineModel` | C / Abl7 (internal, **bare 无前缀**) | INTERNAL |
| `rewardhead_explicit_type` | `"rewardhead_explicit_type"` | `ExplicitTypeRewardBaselineModel` | A / Abl6.x (internal, **bare 无前缀**) | INTERNAL |
| `external_mappo` | `"external_mappo"` | `MAPPOAlgorithm` (`ExternalBaselineRunner`) | external PG (Tier-1) | EXTERNAL |
| `external_qmix` | `"external_qmix"` | `QMIXAlgorithm` (`ExternalBaselineRunner`) | external Q-decomp (Tier-1) | EXTERNAL |
| `external_ma_muzero_gh` | `"external_ma_muzero_gh"` | `MAMuZeroGHAlgorithm` (`ExternalBaselineRunner`) | external model-based MARL (Tier-1) | EXTERNAL |
| `external_mamba` | `"external_mamba"` | `MAMBAAlgorithm` if sourced; else 工厂抛 `NotImplementedError` | model-based + belief (Tier-2) | EXTERNAL |
| `external_marie` | `"external_marie"` | 工厂抛 `NotImplementedError` (stub) | 保留 CLI / registry 占位 | EXTERNAL |
| `external_ga` | `"external_ga"` | 工厂抛 `NotImplementedError` (stub) | 保留 CLI / registry 占位 | EXTERNAL |

### 2.2 前缀约定脚注（design §3.3 footnote 逐字继承）

`baseline_*` 前缀 → internal-with-equal-param-check 变体（与 hyper 走 §07 严格等参账目，**条件化消费子系统受 5%/10% 双阈值约束**）；
bare 名（无前缀）→ ablation-only internal 变体（**条件化子系统结构性豁免** 等参检查 —— ma_muzero/explicit_type/no_belief 的失败模式刻意改变了条件化结构，无可比对的同位参数集；详见 design D5 + spec 07）；
`external_*` 前缀 → external namespace（**不消费** `shared_backbones.py`，自带 trainer/buffer/loss）。

**重要澄清（design D4 锚）**：上述 5 个 internal variant **全部** 各自实例化 `shared_backbones.py` 提供的 RepNet/BeliefNet/TriContextEncoder（同类同构、独立实例、独立梯度、参数量逐位对齐）。前缀差异**不影响** shared_backbones 消费 —— 它只影响 §07 等参 fairness 是否豁免该 variant。spec 02 的 `create_*` 工厂被 5 个 internal model class 全部消费。

**factory arg 命名规则**（spec 01 单测 `test_registry_keys_equal_cli_choices_and_factory_args` 强制）：
- internal-with-shared-backbone：**剥** `baseline_` 前缀（CLI `baseline_input_wide` → factory arg `"input_wide"`）。
- internal-ablation-only：**保留** bare 名（CLI `no_belief` → factory arg `"no_belief"`）。
- external：**保留** `external_` 前缀（CLI `external_mappo` → factory arg `"external_mappo"`），以保证 union return type 在工厂内可消歧。

### 2.3 量词 canonical（与 design §3.1 对齐）

- **REGISTRY = 11 keys**（5 internal + 3 Tier-1 + MAMBA + 2 stubs）。
- **CLI choices = 14 strings**（REGISTRY ∪ {`hyper`, `oracle_only`, `infer_only`}）。
- **Methods 主表 = 9 列**（hyper + 5 internal + 3 Tier-1）；MAMBA-if-sourced 加为第 10 列；MARIE/GA stubs 不入主表（仅 CLI-reachable 占位）。

---

## 3. Factory Signature

### 3.1 文件路径

`hyper_mve/baselines/__init__.py`（纯新增；alias `create_baseline_model` 同文件残留一行）。

### 3.2 签名（逐字对齐 design.md §D2 + proposal §2.1.1）

```python
from __future__ import annotations

import warnings
from types import MappingProxyType
from typing import Mapping, Union, Callable, TypeAlias

from hyper_mve.configs import V4Config
from hyper_mve.baselines.internal.input_conditioned import (
    InputWideBaselineModel, InputDeepBaselineModel,
)
from hyper_mve.baselines.internal.ma_muzero import MAMuZeroBaselineModel
from hyper_mve.baselines.internal.no_belief import NoBeliefBaselineModel
from hyper_mve.baselines.internal.explicit_type_reward import ExplicitTypeRewardBaselineModel
from hyper_mve.baselines.external.mappo import MAPPOAlgorithm
from hyper_mve.baselines.external.qmix import QMIXAlgorithm
from hyper_mve.baselines.external.ma_muzero_gh import MAMuZeroGHAlgorithm
from hyper_mve.baselines.external.mamba import MAMBAAlgorithm  # may be a stub-class if not sourced
from hyper_mve.baselines.external.stubs import MARIEStub, GAStub  # always stubs

# Type alias for the union return (spec 06 + spec 08 共消费)
BaselineLike: TypeAlias = Union["BaselineModel", "ExternalBaselineRunner"]


# Internal namespace: 5 keys (剥前缀 / bare 混合; 见 §2.2)
INTERNAL_REGISTRY: Mapping[str, Callable[[V4Config], "BaselineModel"]] = MappingProxyType({
    "input_wide":                InputWideBaselineModel,
    "input_deep":                InputDeepBaselineModel,
    "ma_muzero":                 MAMuZeroBaselineModel,
    "no_belief":                 NoBeliefBaselineModel,
    "rewardhead_explicit_type":  ExplicitTypeRewardBaselineModel,
})

# External namespace: 6 keys (保留前缀; 含 1 Tier-2 + 2 stubs)
EXTERNAL_REGISTRY: Mapping[str, Callable[[V4Config], "ExternalBaselineRunner"]] = MappingProxyType({
    "external_mappo":         MAPPOAlgorithm,
    "external_qmix":          QMIXAlgorithm,
    "external_ma_muzero_gh":  MAMuZeroGHAlgorithm,
    "external_mamba":         MAMBAAlgorithm,    # spec 06 §3 选定的 sourcing toggle mechanism: module-level `IS_SOURCED: Final[bool]` flag in hyper_mve.baselines.external.mamba; class binding switches between real impl and stub at import-time (`MAMBAAlgorithm = _RealMAMBA if IS_SOURCED else _MAMBAStub`). __init__ of stub raises NotImplementedError("MAMBA sourcing failed; see spec 06 §3 sourcing log"). Spec 06 owns the toggle; this spec just exports the dispatched class.
    "external_marie":         MARIEStub,         # always raises NotImplementedError
    "external_ga":            GAStub,            # always raises NotImplementedError
})

# 合并视图，pkg-08 sweep harness 消费（key 集合不可变；两 namespace 不重叠由 §4 单测保证）
REGISTRY: Mapping[str, Callable[[V4Config], BaselineLike]] = MappingProxyType(
    {**INTERNAL_REGISTRY, **EXTERNAL_REGISTRY}
)


def create_baseline(cfg: V4Config, variant: str) -> BaselineLike:
    """统一 baseline 工厂（pkg-07，supersedes pkg-06 create_baseline_model）。

    Args:
        cfg:     V4Config (从 preset 加载)。仅消费 cfg.baselines.* 5 字段（§5）。
        variant: factory arg，取值见 INTERNAL_REGISTRY ∪ EXTERNAL_REGISTRY (11 keys)。
                 注意：CLI 前缀 asymmetry 由 cli_to_factory_arg 单点处理；本工厂只
                 接受 §2.1 第 2 列的 11 个 factory arg 字串。

    Returns:
        BaselineLike = BaselineModel | ExternalBaselineRunner.
        - INTERNAL_REGISTRY hit → BaselineModel (实现 pkg-04 spec 02 完整 7-API)
        - EXTERNAL_REGISTRY hit → ExternalBaselineRunner (Protocol; 自带 train/evaluate)

    Raises:
        ValueError: 当 variant == "hyper" (走 HyperMuZeroModel 直接构造路径, design D3)。
        ValueError: 当 variant ∈ {"oracle_only", "infer_only"} (走 curriculum override 路径)。
        ValueError: 当 variant 不在两 registry 内 (列出合法 variant)。
        NotImplementedError: 当 variant ∈ {"external_marie", "external_ga"} (永久 stub),
            或 variant == "external_mamba" 且 sourcing 失败 (spec 06 fallback)。
    """
    if variant == "hyper":
        raise ValueError(
            "'hyper' uses HyperMuZeroModel directly, not this factory. "
            "Call HyperMuZeroModel(cfg) instead (see design.md §3.3 curriculum-override row)."
        )
    if variant in ("oracle_only", "infer_only"):
        raise ValueError(
            f"'{variant}' uses HyperMuZeroModel(cfg) + curriculum_stage_1_end_frac override, "
            f"not this factory (see design.md §3.3 curriculum-override row)."
        )
    if variant in INTERNAL_REGISTRY:
        return INTERNAL_REGISTRY[variant](cfg)
    if variant in EXTERNAL_REGISTRY:
        # MAMBA/MARIE/GA 的 NotImplementedError 由其 __init__ 抛出，自然 propagate。
        return EXTERNAL_REGISTRY[variant](cfg)
    raise ValueError(
        f"Unknown baseline variant '{variant}'. "
        f"Valid factory args: {sorted(REGISTRY)}. "
        f"(For 'hyper'/'oracle_only'/'infer_only' use HyperMuZeroModel + curriculum override.)"
    )


# pkg-06 alias (one-release deprecation window, design D2)
def create_baseline_model(cfg: V4Config, variant: str) -> BaselineLike:
    """DEPRECATED: pkg-06 alias. Use create_baseline (pkg-07)."""
    warnings.warn(
        "create_baseline_model is renamed to create_baseline (pkg-07). "
        "This alias will be removed in the next release.",
        DeprecationWarning, stacklevel=2,
    )
    return create_baseline(cfg, variant)
```

### 3.3 reject 语义（D3 + curriculum-override 三行）

- `variant == "hyper"`：明确 `ValueError`，不兜底；调用方误传应快速失败（fail-fast）。
- `variant ∈ {"oracle_only", "infer_only"}`：同样明确 `ValueError`，提示走 curriculum override 路径（train_main.py 在 dispatch 前已分流，因此到工厂只可能是误传）。
- `variant ∈ {"external_marie", "external_ga"}`：永久 stub，`NotImplementedError` 由 stub class 的 `__init__` 抛出（spec 06 实现）。
- `variant == "external_mamba"` 且 spec 06 §3 sourcing 失败：`MAMBAAlgorithm` 退化为 stub class，同样 `__init__` 抛 `NotImplementedError`。

---

## 4. 两 namespace 分派契约

### 4.1 两 namespace 不重叠

`INTERNAL_REGISTRY.keys() & EXTERNAL_REGISTRY.keys() == set()`。Spec 01 单测 `test_two_namespaces_disjoint` 强制。
理由：合并 `REGISTRY = {**INTERNAL, **EXTERNAL}` 若键重叠会静默后覆盖前 — 检测时机推迟到 sweep。

### 4.2 `MappingProxyType` 只读

三个 registry 全部用 `MappingProxyType` 包裹，pkg-08 sweep harness `for v in REGISTRY` 不可意外注入新 key。任何"扩 registry"的 PR 必须改 spec 01 本节 + 同步 14-行表 + 同步 train_main.py CLI choices 三处。

### 4.3 合并视图 `REGISTRY` 供 pkg-08 消费

```python
REGISTRY: Mapping[str, Callable[[V4Config], BaselineLike]] = MappingProxyType(
    {**INTERNAL_REGISTRY, **EXTERNAL_REGISTRY}
)
```

pkg-08 sweep harness 消费 `REGISTRY` 而非分开消费 INTERNAL/EXTERNAL — sweep 视角下 internal/external 同等地位（都跑 `.evaluate(env_fn, c_grid, episodes)`，统一 `EvalReport` schema）。
然而 trainer-side（pkg-05 `create_trainer_for_baseline`）需消歧：`variant in INTERNAL_REGISTRY` → 喂 `MuZeroTrainer`；`variant in EXTERNAL_REGISTRY` → 调 runner 自带 `.train(cfg, env_fn=PettingZooAdapter(env_cfg))`。
两 namespace 分开导出 + 合并视图三件套是诚实的"sweep 统一 / train 消歧"折中。

---

## 5. CLI Mapping & cli_to_factory_arg

### 5.1 train_main.py CLI choices = REGISTRY keys ∪ {hyper, oracle_only, infer_only}

`hyper_mve/scripts/train_main.py:45-52` `_DEFERRED_VARIANTS` 是**部分 ground truth**（包含 6 条 CLI 字串，与本 spec §3 表对应行逐字一致；剩余 8 条由 spec 08 §"下游补丁声明" 中的 train_main.py CLI 扩展引入）。实施期 train_main.py CLI choices list 由派生函数生成（合并 `_DEFERRED_VARIANTS` 与本 spec REGISTRY 派生的 `external_*` 名）：

```python
# train_main.py (spec 08 §"下游补丁声明" 锁定的实施期改动；本 spec 仅声明形状)
from hyper_mve.baselines import REGISTRY

_CURRICULUM_OVERRIDE_VARIANTS = ("hyper", "oracle_only", "infer_only")

def _build_cli_variant_choices() -> tuple[str, ...]:
    """单点生成 14 行 CLI choices（pkg-08 sweep harness 与 train_main.py 共消费）。"""
    internal_with_prefix = (
        "baseline_input_wide", "baseline_input_deep", "baseline_ma_muzero",
    )
    internal_bare = ("no_belief", "rewardhead_explicit_type")
    external_with_prefix = tuple(k for k in REGISTRY if k.startswith("external_"))
    return _CURRICULUM_OVERRIDE_VARIANTS + internal_with_prefix + internal_bare + external_with_prefix
```

### 5.2 前缀 asymmetry 单点：`cli_to_factory_arg(cli) -> str`

派生函数把 CLI 字串转 factory arg（剥/保留前缀的单点处理）：

```python
# hyper_mve/baselines/__init__.py 末尾

_CLI_TO_FACTORY_INTERNAL_PREFIX = ("baseline_input_wide", "baseline_input_deep", "baseline_ma_muzero")
_CLI_BARE_INTERNAL = ("no_belief", "rewardhead_explicit_type")
_CURRICULUM_OVERRIDE = ("hyper", "oracle_only", "infer_only")


def cli_to_factory_arg(cli: str) -> str:
    """Single source of truth for CLI ↔ factory-arg conversion.

    Rules (per §2.2 + design §3.3 footnote):
      - curriculum-override (hyper/oracle_only/infer_only): no factory; raise ValueError.
      - internal-with-shared-backbone (`baseline_*`): strip 'baseline_' prefix.
      - internal-ablation-only (no_belief/rewardhead_explicit_type): pass-through.
      - external (`external_*`): pass-through (preserve prefix for disambiguation).
    """
    if cli in _CURRICULUM_OVERRIDE:
        raise ValueError(
            f"'{cli}' is a curriculum-override variant, not a factory variant "
            f"(see design.md §3.3 curriculum-override rows)."
        )
    if cli in _CLI_TO_FACTORY_INTERNAL_PREFIX:
        return cli.removeprefix("baseline_")
    if cli in _CLI_BARE_INTERNAL:
        return cli
    if cli.startswith("external_") and cli in EXTERNAL_REGISTRY:
        return cli
    raise ValueError(
        f"Unknown CLI variant '{cli}'. Valid CLI choices: "
        f"{_CURRICULUM_OVERRIDE + tuple(sorted(REGISTRY))}."
    )
```

### 5.3 Round-trip 单测强制

`test_registry_keys_equal_cli_choices_and_factory_args`（§7 第 5 条）枚举 14 个 CLI 字串，对每个 `cli`：
1. 若 `cli` 在 `_CURRICULUM_OVERRIDE` 中，`cli_to_factory_arg(cli)` 抛 `ValueError`；
2. 否则 `cli_to_factory_arg(cli)` 返回值 ∈ `REGISTRY`；
3. 且 `create_baseline(cfg, cli_to_factory_arg(cli))` 不抛 `ValueError`（仅 stubs 抛 `NotImplementedError`，单测显式 `pytest.raises` 捕获）；
4. 三集合（CLI choices, factory args, REGISTRY keys + curriculum-override）的逻辑关系满足 §2.3 量词。

---

## 6. `cfg.baselines.*` 消费契约 — 5 字段穷举

### 6.1 工厂只读 `cfg.baselines` 5 字段（design §D10 dataclass）

design.md §D10 锁定 `BaselinesConfig` **5 字段穷举**。工厂构造 5 internal + 3 Tier-1 external + MAMBA 时，**仅消费这 5 字段**：

| 字段 | 类型 | 消费者 variant | 用途 |
|------|------|---------------|------|
| `internal_wide_hidden_dim` | `int` | `input_wide` | 加宽功能网隐层宽度（spec 03） |
| `internal_deep_layers` | `int` | `input_deep` | 加深功能网层数（spec 03） |
| `internal_ma_muzero_share_pred_head` | `bool` | `ma_muzero` | 共享 pred 头开关（spec 03） |
| `internal_explicit_type_branches` | `int` | `rewardhead_explicit_type` | RewardHead 显式 type 分支数（spec 03） |
| `external_lr_sweep_grid` | `Mapping[str, tuple[float, ...]]` | 3 Tier-1 + MAMBA（spec 04/05/06 LR sweep） | per-baseline LR sweep grid |

### 6.2 无 hardcoded 默认值约束（C7-INT-CFG1 + C7-EXT-CFG1）

工厂 + 8 模型类（5 internal + 3 Tier-1）**必须从 `cfg.baselines.<field>` 读取**，**不得**在工厂/模型类内写死。`input_wide` / `input_deep` 的宽度/层数是等参逼近的可调旋钮（spec 07 双阈值迭代），写死会让等参验证失效。

单测 `test_cfg_baselines_5_fields_consumed_no_defaults`（§7 第 4 条）：
- 篡改 `cfg.baselines.internal_wide_hidden_dim` 翻倍 → `input_wide` 模型条件化子系统参数量随之变化；
- 篡改 `cfg.baselines.internal_deep_layers + 2` → `input_deep` 参数量增大；
- 篡改 `cfg.baselines.internal_ma_muzero_share_pred_head` → `ma_muzero` 模型 pred head 结构改变；
- 篡改 `cfg.baselines.internal_explicit_type_branches` → `rewardhead_explicit_type` 分支数改变；
- 篡改 `cfg.baselines.external_lr_sweep_grid["external_mappo"]` → `MAPPOAlgorithm.lr_grid` 随之变化。

### 6.3 拒绝额外 keys（dataclass frozen）

`BaselinesConfig` 是 `@dataclass(frozen=True)`（design §D10）。运行期向 `cfg.baselines` 注入额外属性会抛 `FrozenInstanceError`。本约束防止"加 baseline 跑通就在 cfg 里塞个新字段"的 ad-hoc 实施习惯 — 任何新字段必须经 design §D10 + Pkg-01 spec 05 同步消费态声明。

### 6.4 Per-impl tuning constants 不进 `cfg.baselines`

per-impl 旋钮（`external_mappo_share_policy` / `external_qmix_mixer_hidden_dim` / `external_ma_muzero_gh_simulations` / `external_smoke_max_env_steps`）留在 spec 05/06 内部 defaults — `cfg.baselines` 字段是「跨 spec 共享契约」，不是「所有可调参数的字典」（design §D10 末段）。

---

## 7. Self-Test 契约（spec 01 5 个具名单测）

`tests/baselines/test_factory.py`（pkg-07 实施期建立；本 spec 锁定名字 + 断言形状）：

### 7.1 `test_factory_dispatches_11_keys`

枚举 `REGISTRY` 11 keys（5 internal + 3 Tier-1 + MAMBA + 2 stubs），对每 key：
- internal (5)：`create_baseline(cfg, k)` 返回 `BaselineModel` 实例。
- Tier-1 (3)：`create_baseline(cfg, k)` 返回 `ExternalBaselineRunner` 协议实例。
- MAMBA：若 spec 06 §3 sourcing 成功 → 返回 `MAMBAAlgorithm` 实例；否则 `pytest.raises(NotImplementedError)`。
- 2 stubs：`pytest.raises(NotImplementedError, match="MARIE|GA")`。

### 7.2 `test_factory_rejects_hyper_and_curriculum_overrides`

```python
for variant in ("hyper", "oracle_only", "infer_only"):
    with pytest.raises(ValueError, match="HyperMuZeroModel|curriculum"):
        create_baseline(cfg, variant)
```

并对 CLI 前缀化误传（`"baseline_input_wide"` 直接传工厂）：
```python
with pytest.raises(ValueError, match="Unknown baseline variant"):
    create_baseline(cfg, "baseline_input_wide")  # CLI 串误传；要 cli_to_factory_arg 先转
```

### 7.3 `test_factory_stubs_raise_NotImplementedError`

MARIE/GA：永远抛 `NotImplementedError`（无论 spec 06 是否 sourced MAMBA）：
```python
for k in ("external_marie", "external_ga"):
    with pytest.raises(NotImplementedError):
        create_baseline(cfg, k)
```

MAMBA：分支断言（用 fixture 切换 sourcing flag）：
```python
@pytest.mark.parametrize("mamba_sourced", [True, False])
def test_factory_mamba_branch(cfg, mamba_sourced, monkeypatch):
    monkeypatch.setattr("hyper_mve.baselines.external.mamba.IS_SOURCED", mamba_sourced)
    if mamba_sourced:
        assert create_baseline(cfg, "external_mamba") is not None
    else:
        with pytest.raises(NotImplementedError, match="MAMBA.*sourcing"):
            create_baseline(cfg, "external_mamba")
```

### 7.4 `test_cfg_baselines_5_fields_consumed_no_defaults`

5 字段每个 ≥1 个 mutation 断言（详见 §6.2 5 条枚举）。任何字段写死会被此单测 catch。

### 7.5 `test_registry_keys_equal_cli_choices_and_factory_args`

CLI ↔ factory-arg ↔ REGISTRY round-trip：

```python
def test_registry_keys_equal_cli_choices_and_factory_args(cfg):
    from hyper_mve.baselines import REGISTRY, cli_to_factory_arg
    from hyper_mve.scripts.train_main import _build_cli_variant_choices  # spec 08 patch

    cli_choices = _build_cli_variant_choices()
    assert len(cli_choices) == 14, "CLI choices must equal 14 (design §3.3 hard gate #11)"

    # 11 in-registry rows: cli_to_factory_arg → factory arg ∈ REGISTRY
    curriculum_override = {"hyper", "oracle_only", "infer_only"}
    for cli in cli_choices:
        if cli in curriculum_override:
            with pytest.raises(ValueError, match="curriculum-override"):
                cli_to_factory_arg(cli)
        else:
            arg = cli_to_factory_arg(cli)
            assert arg in REGISTRY, f"CLI {cli!r} → factory arg {arg!r} not in REGISTRY"

    # Conversely: every REGISTRY key is reachable from exactly one CLI string
    reverse = set()
    for cli in cli_choices:
        if cli not in curriculum_override:
            reverse.add(cli_to_factory_arg(cli))
    assert reverse == set(REGISTRY), "REGISTRY ↔ CLI must be bijective on the 11 in-registry rows"
```

### 7.6 (附) `test_two_namespaces_disjoint` + `test_create_baseline_model_alias_warns`

```python
def test_two_namespaces_disjoint():
    from hyper_mve.baselines import INTERNAL_REGISTRY, EXTERNAL_REGISTRY
    assert set(INTERNAL_REGISTRY) & set(EXTERNAL_REGISTRY) == set()

def test_create_baseline_model_alias_warns(cfg):
    from hyper_mve.baselines import create_baseline_model
    with pytest.warns(DeprecationWarning, match="renamed to create_baseline"):
        create_baseline_model(cfg, "input_wide")
```

---

## 8. Integration Hooks（下游 spec 消费什么）

| 消费方 | 消费内容 | 用途 |
|--------|----------|------|
| **spec 02** `02-shared-backbones-internal.md` | `INTERNAL_REGISTRY` **全 5 keys** (`input_wide` / `input_deep` / `ma_muzero` / `no_belief` / `rewardhead_explicit_type`) 共享 `shared_backbones.py` 契约（design D4 锚：5 variant 各自实例化 RepNet/BeliefNet/TriContextEncoder，同类同构、独立实例、独立梯度、参数量逐位对齐）| spec 02 共享后端注入 5 internal model classes 的 `__init__` |
| **spec 03** `03-internal-variants.md` | 5 internal model class `__init__` 形状（吃 `cfg.baselines.internal_*` 4 字段）| spec 03 详化每 variant 的 7-API 实现 + 等参账目 |
| **spec 04** `04-pettingzoo-adapter.md` | EXTERNAL_REGISTRY 3 Tier-1 + MAMBA 4 keys 的 `runner.train(cfg, env_fn=...)` 入口 | spec 04 适配器 `ResourceCommonsPettingZooEnv` 喂给 4 external runner 作为 `env_fn` |
| **spec 05** `05-external-mappo.md` | `EXTERNAL_REGISTRY["external_mappo"]` 入口 + `cfg.baselines.external_lr_sweep_grid["external_mappo"]` | spec 05 MAPPO vendoring + LR sweep 协议 |
| **spec 06** `06-external-qmix-mamuzero-mamba.md` | `EXTERNAL_REGISTRY["external_qmix"]` / `["external_ma_muzero_gh"]` / `["external_mamba"]` + 2 stubs | spec 06 vendoring + MAMBA sourcing 协议 + 2 stubs NotImplementedError 形状 |
| **spec 07** `07-fairness-protocol.md` | 5 internal 等参账目（spec 03 实现的副产物）+ 3 Tier-1 LR sweep grid（`cfg.baselines.external_lr_sweep_grid`）| spec 07 双层 fairness 协议（Internal 严格 / External 披露式）|
| **spec 08** `08-integration-contracts.md` | 合并 `REGISTRY` + `cli_to_factory_arg` + `create_baseline_model` alias DeprecationWarning | spec 08 锁定对 pkg-08 的对外硬契约 + 3 处下游补丁声明（Pkg-02 obs-mask / Pkg-05 planner flag / Pkg-05 CLI 扩 +6 external 名）|
| **pkg-08 sweep harness** | `REGISTRY` 11 keys + `EvalReport` 统一 schema | pkg-08 `--sweep-variants all` 自动枚举（stubs 产生 "skipped: NotImplementedError"）|

---

## 9. Edge Cases

| 场景 | 行为 |
|------|------|
| `variant == "hyper"` | `ValueError`（指引用 HyperMuZeroModel 直接构造）|
| `variant ∈ {"oracle_only", "infer_only"}` | `ValueError`（指引用 HyperMuZeroModel + curriculum override）|
| `variant == "baseline_input_wide"` （CLI 串误传到工厂）| `ValueError`（前缀剥离是 `cli_to_factory_arg` 职责；工厂只认 factory arg） |
| `variant` 大小写错误（如 `"Input_Wide"`） | `ValueError`（REGISTRY key 严格小写下划线）|
| `variant == "external_marie"` / `"external_ga"` | `NotImplementedError`（永久 stub）|
| `variant == "external_mamba"` 且 spec 06 sourcing 失败 | `NotImplementedError`（spec 06 fallback）|
| `cfg.baselines` 缺字段（实施期 Pkg-01 spec 05 未同步） | model class 构造时 `AttributeError`（M4 待 Pkg-01 同步；实施前补 cfg）|
| `cfg.baselines` 构造时传未知 kwarg（如 `BaselinesConfig(foo=1)`）| `TypeError: unexpected keyword argument 'foo'`（dataclass 构造期 Python 默认行为）|
| `cfg.baselines` 构造后 setattr（如 `cfg.baselines.internal_wide_hidden_dim = 1024`）| `FrozenInstanceError`（`@dataclass(frozen=True)` 拦截 assignment-after-construction；design §D10）|
| pkg-06 alias 调用 | `DeprecationWarning(stacklevel=2)` + 转发到 `create_baseline`（一 release 过渡窗口） |
| `variant ∈ {"mappo", "qmix"}` (无 `external_` 前缀的误传) | `ValueError`（factory arg 必须保留 `external_` 前缀，§2.2 footnote）|

---

## 10. Cross-references

### 上游锁定
- **design.md §3.2** — 11-key 工厂矩阵（本 spec §2.1 表逐字继承）
- **design.md §3.3** — CLI ↔ factory-arg ↔ model-class 14-行三列映射表 + 前缀约定脚注（本 spec §2 全节继承 + §5 派生函数）
- **design.md §D2** — 工厂归属 `hyper_mve/baselines/__init__.py::create_baseline` + supersede pkg-06 + alias-with-DeprecationWarning（本 spec §1 头部声明 + §3.2 签名）
- **design.md §D3** — Internal vs External 两 namespace 分派 + 返回 union type（本 spec §3.2 + §4 全节）
- **design.md §D10** — `cfg.baselines.*` 5 字段穷举 dataclass（本 spec §6 全节）
- **proposal.md §2.1.1** — `create_baseline` 工厂伪代码 + alias-with-DeprecationWarning（本 spec §3.2 严格对齐）

### 下游消费
- **spec 02** `02-shared-backbones-internal.md`（**5 internal 全 keys 共享后端契约**：`input_wide` / `input_deep` / `ma_muzero` / `no_belief` / `rewardhead_explicit_type`，全部各自实例化 RepNet/BeliefNet/TriContextEncoder；design D4 锚；equal-param check 的「2 vs 3 豁免」分割由 spec 07 拆分，不在 spec 02 范围）
- **spec 03** `03-internal-variants.md`（5 internal model class 实现 + 等参账目）
- **spec 04** `04-pettingzoo-adapter.md`（`ResourceCommonsPettingZooEnv` 喂 4 external runner）
- **spec 05** `05-external-mappo.md`（MAPPO Tier-1 vendoring + LR sweep）
- **spec 06** `06-external-qmix-mamuzero-mamba.md`（QMIX/MA-MuZero-GH/MAMBA + 2 stubs）
- **spec 07** `07-fairness-protocol.md`（双层 fairness：Internal 等参 / External 披露式）
- **spec 08** `08-integration-contracts.md`（对外硬契约 + 3 处下游补丁声明）

### 上游 ground truth 对账
- `hyper_mve/scripts/train_main.py:45-52` `_DEFERRED_VARIANTS` — CLI 字串 ground truth（design Day 1 hard-gate #11 + 本 spec §5.1）
- `hyper_mve/configs/v4_config.py:23-37` — V4Config 顶层 sub-config 列表（design §D10 末段：`cfg.baselines` 平级挂 V4Config，无 .v4 中间层）

### Supersede 锚点
- **pkg-06 spec 01** `01-baselines-factory-and-registry.md` — 本 spec 头部声明 1 supersede（rename `create_baseline_model` → `create_baseline` + alias-with-DeprecationWarning）
- **pkg-04 spec 08 §5.1** lines 270-285 `create_input_wide_baseline` 旧草案 — 本 spec 头部声明 2 supersede（单一签名）
