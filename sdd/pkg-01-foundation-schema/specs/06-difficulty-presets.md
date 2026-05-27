# Spec 06: Easy / Medium / Hard Difficulty Presets

> 父文档：[`../proposal.md`](../proposal.md) §2.2 · [`../design.md`](../design.md) §3 D8

---

## 1. Purpose

实现 Ch3.9 Table 中定义的三套难度配置（Easy / Medium / Hard），各为一个独立的 Python 文件，返回完整的 `V4Config` 实例。preset 是**论文实验入口**——所有 Pkg-08 实验脚本通过 `V4Config.from_preset(name)` 获取标准配置。

**关键约束**：每个 preset 文件**字段值与 Ch3.9 Table 100% 一致**（单元测试强制）。如果未来需要修改某 preset 字段，必须先修改 Ch3.9 文档，再修改本文件，避免代码-论文漂移。

---

## 2. Interface

### 2.1 文件路径

```
hyper_mve/configs/presets/
├── __init__.py
├── easy.py
├── medium.py
├── hard.py
└── legacy.py     # LegacyConfig 默认实例
```

### 2.2 `medium.py`（主对比配置，最重要）

```python
"""Medium preset (Ch3.9 主对比配置).

N=4, L=16, K=20, T_max=200, 2α+2β.
对应论文 Chapter 6 所有主对比表/图。
"""
from hyper_mve.configs.env_config import EnvConfig
from hyper_mve.configs.model_config import ModelConfig
from hyper_mve.configs.train_config import TrainConfig
from hyper_mve.configs.mup_config import MupConfig
from hyper_mve.configs.eval_config import EvalConfig
from hyper_mve.configs.legacy_config import LegacyConfig
from hyper_mve.configs.v4_config import V4Config
from hyper_mve.schemas import AgentType


def build_medium_config() -> V4Config:
    """构造 Medium 难度完整配置 (Ch3.9 主对比)."""
    env = EnvConfig(
        N=4,
        L=16,
        K=20,
        M=3,
        T_max=200,
        # 主对比固定 2α+2β
        type_assignment=(AgentType.ALPHA, AgentType.ALPHA,
                         AgentType.BETA, AgentType.BETA),
        c_mode="static",
        # 动力学参数 (Ch3.3 / 3.4)
        Q_max=10.0,
        alpha_min=0.02,
        alpha_max=0.20,
        kappa_f=6.0,
        theta_f=0.3,
        d_nbr=3,
        sigma_patch=2.0,
        # 类型机制 (Ch3.5)
        kappa=0.5,
        lambda_disadv=2.0,
        lambda_adv=0.6,
        epsilon_move=0.01,
    )
    
    model = ModelConfig(
        latent_dim=64,
        hidden_dim=128,
        d_c=16,
        d_role=32,
        d_belief=32,            # v4 修订: 32 (= 2 × d_belief_proj)
        d_belief_proj=16,       # v4 新增: ĉ proj + pooled ẑ proj 各 16
        d_id_emb=8,
        d_type_emb=8,           # v4 修订: 8 (不是 4)
        d_cap_emb=16,
        hyper_hidden_dims=(256, 256),
        hyper_rew_hidden_dims=(256, 256, 256),
        trans_output_scale_init=0.01,
        rew_output_scale_init=0.1,
        pred_output_scale_init=0.01,
        belief_gru_hidden=128,
        belief_pool="mean",
        proj_dim=64,
    )
    
    train = TrainConfig(
        max_train_steps=1_000_000,    # Medium 1M (Roadmap §6.3)
        batch_size=256,
        buffer_size=5000,
        min_buffer_size=1000,
        episodes_per_iter=8,
        train_steps_per_iter=8,
        unroll_K=5,
        n_step=5,
        gamma=0.95,
        lr=1e-4,
        lr_min=5e-6,
        lr_warmup_steps=5000,
        # Loss 权重
        w_policy=1.0,
        w_value=0.25,
        w_reward=3.0,
        w_consist=0.5,
        w_belief=1.0,
        w_belief_c=1.0,
        w_belief_opp=0.5,
        w_belief_div=0.01,
        # 课程学习
        curriculum_stage_1_end_frac=0.3,
        curriculum_stage_2_end_frac=0.7,
        belief_grad_gating_steps=5000,
        # EMA
        ema_tau=0.99,
        # 探索
        epsilon_init=1.0,
        epsilon_min=0.05,
        epsilon_decay_steps=28_000,
        # MVE
        mve_samples=50,
        mve_depth=5,
        mve_temperature=1.0,
        # 默认全开 (DPower)
        use_crn=True,
        use_coord_desc=True,
        stratified_sampling=True,
        stratified_min_per_type_frac=0.3,
    )
    
    return V4Config(
        env=env,
        model=model,
        train=train,
        mup=MupConfig(),
        eval=EvalConfig(),
        legacy=LegacyConfig(),
        preset_name="medium",
    )
```

### 2.3 `easy.py`（快速 ablation + LR sweep）

```python
"""Easy preset (Ch3.9).

N=2, L=8, K=8, T_max=100, 1α+1β.
用于 LR sweep, Decision Point 1 (类型扫描 Ablation 3), Decision Point 2 (Ablation 1).
单 run 训练步数 200K (vs Medium 1M) - 显著快.
"""
from hyper_mve.configs.env_config import EnvConfig
from hyper_mve.configs.train_config import TrainConfig
from hyper_mve.configs.v4_config import V4Config
from hyper_mve.schemas import AgentType
# ... 其他 import
from .medium import build_medium_config


def build_easy_config() -> V4Config:
    """构造 Easy 难度完整配置."""
    base = build_medium_config()
    
    from dataclasses import replace
    env = replace(
        base.env,
        N=2,
        L=8,
        K=8,
        M=1,
        T_max=100,
        type_assignment=(AgentType.ALPHA, AgentType.BETA),  # 1α+1β
    )
    
    train = replace(
        base.train,
        max_train_steps=200_000,    # Easy 200K
        # 其他 Easy 不变, Medium 已 sane
    )
    
    return replace(base, env=env, train=train, preset_name="easy")
```

### 2.4 `hard.py`（在线适应 + scale 测试）

```python
"""Hard preset (Ch3.9).

N=8, L=24, K=40, T_max=300, 4α+4β.
oscillate / random_walk c_mode (Ch3.4).
用于在线适应实验（Roadmap §6.10）.
"""
from dataclasses import replace
from hyper_mve.configs.v4_config import V4Config
from hyper_mve.schemas import AgentType
from .medium import build_medium_config


def build_hard_config() -> V4Config:
    base = build_medium_config()
    
    env = replace(
        base.env,
        N=8,
        L=24,
        K=40,
        M=5,
        T_max=300,
        type_assignment=(AgentType.ALPHA,) * 4 + (AgentType.BETA,) * 4,
        c_mode="oscillate",  # Hard 默认振荡
    )
    
    train = replace(
        base.train,
        max_train_steps=2_000_000,    # Hard 2M (更多 sample)
    )
    
    return replace(base, env=env, train=train, preset_name="hard")
```

### 2.5 `legacy.py`（v4.7 遗留默认）

```python
"""LegacyConfig 默认实例."""
from dataclasses import dataclass


@dataclass(frozen=True)
class LegacyConfig:
    """v4.7 遗留参数 (默认全禁用)。
    
    v4 修订 Pass 2: 删除 w_rew_diversity / detach_pred_context 等 v4 架构已不需要的字段。
    Ch4_v4 §4.6 四道防线无 detach_pred_context;
    type_emb 进 role 通路 (Ch4.2.2 + 4.3.3) 已根治 RewardHead 撕裂, 多样性正则失去存在理由.
    """
    # Adversarial Freezing (v4.2) - 与 v4 正交, 保留作为回退实验开关
    freeze_enabled: bool = False
    freeze_warmup_steps: int = 4000
    freeze_phase_steps: int = 4000
    freeze_hunter_agents: tuple[int, ...] = (0, 1, 2)
    freeze_prey_agents: tuple[int, ...] = (3,)
    
    # ChunkedHMLP (v4.5, deprecated) - 与 v4 正交, 保留兜底
    chunk_alpha: int = 10
    chunk_emb_size: int = 8
    chunk_budget_factor: float = 1.0
    chunk_hyperfan_init: bool = True
    
    # v4 删除字段 (架构已根治, 不再需要):
    # - w_rew_diversity / rew_diversity_target_cos / rew_diversity_skip_pairs
    #   理由: type_emb 进 role 通路从根本上消除 RewardHead 撕裂 (Ch4.3.3 末段)
    # - detach_pred_context
    #   理由: BeliefNet 独立 belief 通路 + gradient gating (Ch4.6.5) 取代该机制
```

---

## 3. Implementation Notes

### 3.1 preset 继承策略

Easy / Hard 从 Medium 衍生（用 `replace`），而不是各写一份。理由：
- Medium 是"完整 baseline"，参数齐全已审阅
- Easy / Hard 仅在 env 维度 + max_train_steps 上 override，其余继承
- 维护成本低：修改 Medium 的 model 配置自动传播到 Easy / Hard
- 单元测试可验证 Easy / Hard 与 Medium 的字段 diff 数量 ≤ 预期

### 3.2 不通过 dict 覆盖

避免：
```python
# 不推荐：dict 风格
preset = {**medium_dict, "env": {**medium_dict["env"], "N": 2}}
```

理由：dict 风格丢失类型信息，IDE 无法跳转，typo 不报错。`replace` 强制类型安全。

### 3.3 preset_name 字段

`V4Config.preset_name` 用于实验日志记录（如 wandb / TensorBoard）：

```python
writer.add_text("preset", cfg.preset_name)
```

`replace` 后 preset_name 保持原值（用户期望"basis preset"为标签）；如果完全重构，用户显式传 `preset_name="custom"`。

### 3.4 数值常量来源

所有数值字段值的来源：

| 字段 | 数值 | 来源 |
|------|------|------|
| Easy/Medium/Hard 维度 | 见 §2.2-2.4 | Ch3.9 Reference Configuration Table |
| `kappa`, `lambda_disadv`, `lambda_adv` | 0.5, 2.0, 0.6 | Ch3.5 Fehr-Schmidt |
| `alpha_min`, `alpha_max` | 0.02, 0.20 | Ch3.4 公式 3.4 |
| `kappa_f`, `theta_f` | 6.0, 0.3 | Ch3.3.3 neighbor factor |
| `gamma`, `unroll_K`, `n_step` | 0.95, 5, 5 | Ch5.6 MuZero 标准 |
| `ema_tau` | 0.99 | v4.4 实证 |
| `lr`, `lr_warmup_steps` | 1e-4, 5000 | v4.7 实证 |
| Loss 权重 | 见 §2.2 | Ch5.6.3 |
| 课程边界 0.3 / 0.7 | 见 §2.2 | Ch5.7 |

**禁止**：在其他文件 hard-code 这些值。所有引用必须通过 `cfg.env.kappa` / `cfg.train.gamma` 等。

---

## 4. Edge Cases

| 场景 | 行为 |
|------|------|
| `replace` 后 type_assignment 长度与 N 不匹配 | EnvConfig `__post_init__` ValueError |
| Easy 想用 c_mode="random_walk" | `replace(easy.env, c_mode="random_walk")` 显式 |
| 用户希望保存 preset 到 yaml | 用 `V4Config.to_dict()` + `yaml.dump`（自实现） |
| Hard preset 2M 步 GPU 时间预算超标 | 用户自行 `replace(hard.train, max_train_steps=1_000_000)` |
| ~~`legacy.detach_pred_context = True` 与 Pkg-04 BeliefNet 干扰~~ | **v4 修订: detach_pred_context 已从 LegacyConfig 删除（架构上由 belief gradient gating 取代）** |

---

## 5. Acceptance Criteria

### 5.1 单元测试（`tests/configs/test_presets.py`）

```python
import pytest
from hyper_mve.configs import V4Config
from hyper_mve.schemas import AgentType

def test_medium_preset_ch_3_9_table():
    """Medium 字段必须 100% 符合 Ch3.9 Table."""
    cfg = V4Config.from_preset("medium")
    
    # 环境维度
    assert cfg.env.N == 4
    assert cfg.env.L == 16
    assert cfg.env.K == 20
    assert cfg.env.M == 3
    assert cfg.env.T_max == 200
    assert cfg.env.c_mode == "static"
    
    # 类型分配 2α+2β
    assert cfg.env.type_assignment == (
        AgentType.ALPHA, AgentType.ALPHA, AgentType.BETA, AgentType.BETA
    )
    
    # 动力学
    assert cfg.env.Q_max == 10.0
    assert cfg.env.alpha_min == 0.02
    assert cfg.env.alpha_max == 0.20
    assert cfg.env.kappa_f == 6.0
    assert cfg.env.theta_f == 0.3
    
    # Fehr-Schmidt
    assert cfg.env.kappa == 0.5
    assert cfg.env.lambda_disadv == 2.0
    assert cfg.env.lambda_adv == 0.6


def test_easy_preset_ch_3_9_table():
    cfg = V4Config.from_preset("easy")
    assert cfg.env.N == 2
    assert cfg.env.L == 8
    assert cfg.env.K == 8
    assert cfg.env.M == 1
    assert cfg.env.T_max == 100
    assert cfg.env.type_assignment == (AgentType.ALPHA, AgentType.BETA)
    assert cfg.train.max_train_steps == 200_000


def test_hard_preset_ch_3_9_table():
    cfg = V4Config.from_preset("hard")
    assert cfg.env.N == 8
    assert cfg.env.L == 24
    assert cfg.env.K == 40
    assert cfg.env.M == 5
    assert cfg.env.T_max == 300
    assert len(cfg.env.type_assignment) == 8
    assert sum(1 for t in cfg.env.type_assignment if t == AgentType.ALPHA) == 4
    assert sum(1 for t in cfg.env.type_assignment if t == AgentType.BETA) == 4
    assert cfg.env.c_mode == "oscillate"
    assert cfg.train.max_train_steps == 2_000_000


def test_preset_name_field():
    assert V4Config.from_preset("easy").preset_name == "easy"
    assert V4Config.from_preset("medium").preset_name == "medium"
    assert V4Config.from_preset("hard").preset_name == "hard"


def test_easy_inherits_from_medium():
    """Easy 应继承 Medium 的 model + train (除 max_train_steps) 字段."""
    medium = V4Config.from_preset("medium")
    easy = V4Config.from_preset("easy")
    
    # model 完全相同
    assert easy.model == medium.model
    
    # train 除 max_train_steps 应相同
    assert easy.train.lr == medium.train.lr
    assert easy.train.batch_size == medium.train.batch_size
    assert easy.train.curriculum_stage_1_end_frac == 0.3


def test_legacy_default_disabled():
    cfg = V4Config.from_preset("medium")
    assert cfg.legacy.freeze_enabled is False
    # v4 修订: 以下字段已从 LegacyConfig 删除 (架构上不再需要)
    assert not hasattr(cfg.legacy, "w_rew_diversity")
    assert not hasattr(cfg.legacy, "rew_diversity_target_cos")
    assert not hasattr(cfg.legacy, "rew_diversity_skip_pairs")
    assert not hasattr(cfg.legacy, "detach_pred_context")


def test_all_presets_no_post_init_errors():
    """三 preset 全部 __post_init__ 通过."""
    for name in ("easy", "medium", "hard"):
        V4Config.from_preset(name)  # 不抛异常


def test_preset_to_dict_serializable():
    import json
    for name in ("easy", "medium", "hard"):
        cfg = V4Config.from_preset(name)
        s = json.dumps(cfg.to_dict())
        assert len(s) > 500   # 完整配置不为空


def test_replace_does_not_mutate_original():
    """frozen + replace 不应修改原 preset."""
    from dataclasses import replace
    
    cfg = V4Config.from_preset("medium")
    original_N = cfg.env.N
    
    _ = replace(cfg, env=replace(cfg.env, N=8))
    
    assert cfg.env.N == original_N  # 原 cfg 不变
```

### 5.2 与 Ch3.9 Table 字段对照

**Easy**：

| 字段 | Ch3.9 值 | spec 值 | ✓/✗ |
|------|----------|---------|------|
| N | 2 | 2 | ✓ |
| L | 8 | 8 | ✓ |
| K | 8 | 8 | ✓ |
| M | 1 | 1 | ✓ |
| T_max | 100 | 100 | ✓ |
| c_mode | Static | "static" | ✓ |
| type | 1α+1β | (ALPHA, BETA) | ✓ |

**Medium**：

| 字段 | Ch3.9 值 | spec 值 | ✓/✗ |
|------|----------|---------|------|
| N | 4 | 4 | ✓ |
| L | 16 | 16 | ✓ |
| K | 20 | 20 | ✓ |
| M | 3 | 3 | ✓ |
| T_max | 200 | 200 | ✓ |
| c_mode | Static | "static" | ✓ |
| type | 2α+2β | (A, A, B, B) | ✓ |

**Hard**：

| 字段 | Ch3.9 值 | spec 值 | ✓/✗ |
|------|----------|---------|------|
| N | 8 | 8 | ✓ |
| L | 24 | 24 | ✓ |
| K | 40 | 40 | ✓ |
| M | 5 | 5 | ✓ |
| T_max | 300 | 300 | ✓ |
| c_mode | Osc/RW | "oscillate" | ✓ |
| type | 4α+4β | 4 A + 4 B | ✓ |

---

## 6. Cross-references

- `05-v4-config-structure.md`（V4Config 结构）
- `01-agent-type-schema.md`（AgentType 在 type_assignment 中使用）
- Ch3.9 Reference Configurations Table
- Ch5.7 课程边界 0.3 / 0.7
- Ch6 整章实验设计（preset 选择依据）
- `../design.md` D8（preset 覆盖语义）
