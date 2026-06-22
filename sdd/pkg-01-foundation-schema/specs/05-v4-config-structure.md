# Spec 05: V4Config 5 层分解

> 父文档：[`../proposal.md`](../proposal.md) §2.2 · [`../design.md`](../design.md) §3 D3 / D6 / D8

---

## 1. Purpose

将 v4.7 `BaseConfig` 单一 70 参数类拆分为 **5 个独立 sub-config dataclass** + 1 个**组合器 `V4Config`**。每个 sub-config 职责单一、字段精简、可独立测试；组合器提供 preset 加载与字段覆盖入口。

---

## 2. Interface

### 2.1 文件路径与模块结构

```
hyper_mve/configs/
├── __init__.py              # re-export V4Config, presets
├── env_config.py            # EnvConfig (环境维度 + 动力学参数)
├── model_config.py          # ModelConfig (网络维度 + hypernet 配置)
├── train_config.py          # TrainConfig (训练循环 + loss + 课程)
├── mup_config.py            # MupConfig (μP base_shape 占位)
├── eval_config.py           # EvalConfig (评估频率 + 协议参数)
├── legacy_config.py         # LegacyConfig (v4.7 遗留, 默认禁用)
├── v4_config.py             # V4Config 组合器
└── presets/
    ├── easy.py
    ├── medium.py
    └── hard.py
```

### 2.2 EnvConfig

```python
@dataclass(frozen=True)
class EnvConfig:
    """环境维度与动力学参数 (Ch3.3-3.6, 3.9)."""
    # 维度
    N: int                  # agent 数, ∈ {2, 4, 8}
    L: int                  # 网格边长 L×L
    K: int                  # 资源点数
    M: int                  # hotspot 数
    T_max: int              # episode 最大步数
    A: int = 6              # 动作空间 (NOOP+4方向+HARVEST)
    
    # 类型分配
    type_assignment: tuple[AgentType, ...] = (AgentType.ALPHA, AgentType.ALPHA, 
                                                 AgentType.BETA, AgentType.BETA)
    
    # Context 演化
    c_mode: str = "static"  # 'static' / 'oscillate' / 'random_walk'
    c_oscillate_period: int = 50
    c_random_walk_sigma: float = 0.05
    c_shock_prob: float = 0.2
    c_shock_range: float = 0.3
    
    # Context 可观测性 (Ch3.7.3) [v4-opt]
    # False = c_t 隐藏模式: obs global 块 c 槽位替换为常量 0.5。仅影响观测;
    # dynamics / rewards / info["c_true"] 始终用真实 c_t。见 Pkg-02 spec 04 §3.4。
    c_visible: bool = True
    
    # 资源动力学 (Ch3.3 公式)
    Q_max: float = 10.0
    alpha_min: float = 0.02       # α(c_t) 下界
    alpha_max: float = 0.20       # α(c_t) 上界
    kappa_f: float = 6.0          # neighbor factor sigmoid 斜率
    theta_f: float = 0.3          # neighbor factor sigmoid 阈值
    d_nbr: int = 3                # 邻居距离阈值 (Chebyshev)
    
    # Patchy 资源生成 (Ch3.3.2)
    sigma_patch: float = 2.0
    
    # 类型机制 (Ch3.5)
    kappa: float = 0.5            # φ(c) = κ(1-2c)
    lambda_disadv: float = 2.0    # Fehr-Schmidt λ
    lambda_adv: float = 0.6
    epsilon_move: float = 0.01    # 移动小成本
    
    # Capability 采样范围 (Ch3.6) - 默认 = schemas 默认
    eta_range: tuple[float, float] = (0.5, 1.5)
    phi_fov_range: tuple[float, float] = (2.0, 4.0)
    nu_range: tuple[float, float] = (0.8, 1.0)
    zeta_range: tuple[float, float] = (10.0, 30.0)
    
    def __post_init__(self) -> None:
        if len(self.type_assignment) != self.N:
            raise ValueError(f"type_assignment length {len(self.type_assignment)} != N {self.N}")
        if self.c_mode not in ("static", "oscillate", "random_walk"):
            raise ValueError(f"Unknown c_mode: {self.c_mode}")
        if not (0.0 < self.alpha_min < self.alpha_max <= 1.0):
            raise ValueError(f"alpha range invalid: ({self.alpha_min}, {self.alpha_max})")
```

### 2.3 ModelConfig

```python
@dataclass(frozen=True)
class ModelConfig:
    """模型维度与 hypernet 配置 (Ch4.2-4.6, v4 修订 Pass 2)."""
    # 表示层
    latent_dim: int = 64
    hidden_dim: int = 128
    
    # 上下文路径维度 (Ch4.2.4 硬约束)
    d_c: int = 16              # c_ctx 维度 (Ch4.2.1)
    d_role: int = 32           # role 维度 = d_id + d_type + d_cap 精确填满
    d_belief: int = 32         # belief 维度 = 2 × d_belief_proj (Ch4.2.4 d_b^{proj·2})
                               # v4 修订: 32 (不是 v3 推测的 48)
    
    # role 内部分解 (Ch4.2.2 默认值, 8+8+16=32 精确填满, 无 pad)
    d_id_emb: int = 8
    d_type_emb: int = 8        # v4 关键: 8 (不是 4)
    d_cap_emb: int = 16
    
    # belief 内部分解 (Ch4.2.3)
    d_belief_proj: int = 16    # 每个 belief 子分量 (ĉ / pooled ẑ) 投影维度
                               # belief = Concat[Proj(c_hat), Proj(Pool(z_hat))] = 16+16 = 32
    
    # HyperNet 配置 (Ch4.6 stability)
    hyper_hidden_dims: tuple[int, ...] = (256, 256)
    hyper_rew_hidden_dims: tuple[int, ...] = (256, 256, 256)  # 更深 (v4.7 经验)
    
    # output_scale 初值 (Ch4.6 防线 1)
    trans_output_scale_init: float = 0.01
    rew_output_scale_init: float = 0.1    # v4.7 0.1 (不是 0.01) - 关键
    pred_output_scale_init: float = 0.01
    
    # AdaLN 配置 (Ch4.6 防线 2)
    use_adaln: bool = True
    adaln_residual_one_plus: bool = True   # h × (1 + gamma) + beta
    
    # Δs 残差 (Ch4.6)
    state_trans_residual: bool = True
    
    # BeliefNet (Ch4.2.3 / 4.5)
    belief_gru_hidden: int = 128
    belief_pool: str = "mean"   # 'mean' / 'max' / 'attention'
    
    # Projector (Ch5.8.2 BYOL consistency)
    proj_dim: int = 64
    
    def __post_init__(self) -> None:
        # v4 修订: d_role = d_id + d_type + d_cap 精确等于 (无 pad)
        expected_d_role = self.d_id_emb + self.d_type_emb + self.d_cap_emb
        if expected_d_role != self.d_role:
            raise ValueError(
                f"d_role mismatch: d_id({self.d_id_emb}) + d_type({self.d_type_emb}) "
                f"+ d_cap({self.d_cap_emb}) = {expected_d_role}, 但 d_role={self.d_role}. "
                f"Ch4.2.2 要求精确填满, 无 pad."
            )
        # v4 修订: d_belief = 2 × d_belief_proj
        if self.d_belief != 2 * self.d_belief_proj:
            raise ValueError(
                f"d_belief mismatch: 2 × d_belief_proj({self.d_belief_proj}) "
                f"= {2 * self.d_belief_proj}, 但 d_belief={self.d_belief}. "
                f"Ch4.2.4 要求 d_belief = 2 × d_b^proj."
            )
        if self.belief_pool not in ("mean", "max", "attention"):
            raise ValueError(f"Unknown belief_pool: {self.belief_pool}")
    
    @property
    def d_ctx_aug(self) -> int:
        """总条件向量维度 (Ch4.2.4): c_ctx + role + belief = 16 + 32 + 32 = 80."""
        return self.d_c + self.d_role + self.d_belief
```

### 2.4 TrainConfig

```python
@dataclass(frozen=True)
class TrainConfig:
    """训练循环 + loss 权重 + 课程学习 (Ch5)."""
    # 总训练步数
    max_train_steps: int = 200_000  # Easy/Medium default; Hard override
    
    # Batch + buffer
    batch_size: int = 256           # v4.4 经验值
    buffer_size: int = 5000         # episode 单位
    min_buffer_size: int = 1000     # warmup 阈值
    
    # 采集与训练比例
    episodes_per_iter: int = 8
    train_steps_per_iter: int = 8
    
    # MuZero unroll
    unroll_K: int = 5
    n_step: int = 5
    gamma: float = 0.95
    
    # Optimizer
    lr: float = 1e-4
    lr_min: float = 5e-6
    adam_eps: float = 1e-5
    grad_clip: float = 10.0
    
    # LR schedule (v4.7)
    lr_schedule: str = "warmup_cosine"    # 'warmup_cosine' / 'cosine' / 'multistep'
    lr_warmup_steps: int = 5000
    
    # Loss 权重 (Ch5.6.3)
    w_policy: float = 1.0
    w_value: float = 0.25
    w_reward: float = 3.0       # v4.8 经验
    w_consist: float = 0.5
    w_belief: float = 1.0       # 课程化 ramp; 这里是 max 值
    
    # BeliefNet sub-loss 权重 (Ch4.5)
    w_belief_c: float = 1.0       # L_c
    w_belief_opp: float = 0.5     # L_opp
    w_belief_div: float = 0.01    # L_div hinge variance
    belief_div_target_std: float = 0.1
    
    # 课程学习 (Ch5.7) - 三阶段边界
    curriculum_stage_1_end_frac: float = 0.3   # Stage 1 (Oracle) ends at 30% of max_train_steps
    curriculum_stage_2_end_frac: float = 0.7   # Stage 2 (Anneal) ends at 70%
    
    # Belief gradient gating (Ch4.6 新增防线)
    belief_grad_gating_steps: int = 5000   # 前 5K step 切断 BeliefNet 梯度
    
    # EMA target net (v4.4)
    ema_tau: float = 0.99
    
    # Exploration
    epsilon_init: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay_steps: int = 28_000
    
    # MVE planner (Pkg-05)
    mve_samples: int = 50
    mve_depth: int = 5
    mve_temperature: float = 1.0
    
    # 2x2 ablation 开关 (Ch6.7)
    use_crn: bool = True
    use_coord_desc: bool = True
    
    # Type-stratified sampling (Ch5.6.5)
    stratified_sampling: bool = True
    stratified_min_per_type_frac: float = 0.3
    
    def __post_init__(self) -> None:
        if not (0 < self.curriculum_stage_1_end_frac < self.curriculum_stage_2_end_frac < 1):
            raise ValueError("Curriculum stage boundaries must be 0 < s1 < s2 < 1")
        if self.lr_schedule not in ("warmup_cosine", "cosine", "multistep"):
            raise ValueError(f"Unknown lr_schedule: {self.lr_schedule}")
```

### 2.5 MupConfig

```python
@dataclass(frozen=True)
class MupConfig:
    """μP base_shape 配置占位 (Pkg-07 填充具体值)."""
    enabled: bool = False
    # 实际 base_shape dict 由 Pkg-07 填充
    base_shape_path: Optional[str] = None  # 指向 yaml/json 文件路径
    
    # LR scaling per module (Pkg-07 实现)
    lr_scaling_enabled: bool = False
```

### 2.6 EvalConfig

```python
@dataclass(frozen=True)
class EvalConfig:
    """评估协议 (Pkg-07 实现, 这里仅参数占位)."""
    evaluate_freq: int = 500          # 训练步数间隔
    evaluate_episodes: int = 30
    
    # c-segment eval (Ch6.2.4)
    c_segments: tuple[tuple[float, float], ...] = (
        (0.0, 0.3),
        (0.3, 0.7),
        (0.7, 1.0),
    )
    
    # Zero-shot c (Ch6.9)
    zero_shot_train_c: tuple[float, ...] = (0.2, 0.5, 0.8)
    zero_shot_test_c: tuple[float, ...] = (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)
    zero_shot_unseen_c: tuple[float, ...] = (0.0, 0.35, 0.65, 1.0)
    
    # Bell curve (Ch6.6)
    bell_curve_type_ratios: tuple[tuple[int, int], ...] = (
        (0, 4), (1, 3), (2, 2), (3, 1), (4, 0), (1, 3),  # 最后一个 1α+3β 诊断
    )
```

### 2.7 V4Config 组合器

```python
@dataclass(frozen=True)
class V4Config:
    """v4 完整配置 (Ch3.9 三 preset)."""
    env: EnvConfig
    model: ModelConfig
    train: TrainConfig
    mup: MupConfig = field(default_factory=MupConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    legacy: LegacyConfig = field(default_factory=LegacyConfig)
    
    # 元数据
    preset_name: str = "custom"
    
    @classmethod
    def from_preset(cls, name: str) -> "V4Config":
        """从 presets/{name}.py 加载完整配置.
        
        Args:
            name: 'easy' / 'medium' / 'hard'
        """
        if name == "easy":
            from .presets.easy import build_easy_config
            return build_easy_config()
        elif name == "medium":
            from .presets.medium import build_medium_config
            return build_medium_config()
        elif name == "hard":
            from .presets.hard import build_hard_config
            return build_hard_config()
        raise ValueError(f"Unknown preset: {name}. Valid: easy, medium, hard")
    
    def to_dict(self) -> dict:
        """JSON-serializable dict 输出 (实验日志)."""
        from dataclasses import asdict
        d = asdict(self)
        # AgentType Enum 转 int
        d["env"]["type_assignment"] = [t.value for t in self.env.type_assignment]
        return d
```

---

## 3. Implementation Notes

### 3.1 sub-config 独立性

- 每个 sub-config 独立可构造（无跨 config 依赖）
- 跨 config 的派生量在 V4Config 层计算（如 obs_dim = ObservationLayout.total_dim(env.N, env.K)）

### 3.2 frozen + replace 用法

```python
# 修改单字段（保持其他不变）
from dataclasses import replace
cfg = V4Config.from_preset("medium")
cfg_lr = replace(cfg, train=replace(cfg.train, lr=3e-4))

# 修改类型分配
new_types = (AgentType.ALPHA,) * 3 + (AgentType.BETA,)
cfg_3a1b = replace(cfg, env=replace(cfg.env, N=4, type_assignment=new_types))
```

### 3.3 JSON 序列化

- `to_dict()` 处理 AgentType Enum → int
- tuple 自动转 list
- 不序列化 LegacyConfig（如果全部默认值）以减小日志体积

### 3.4 import 路径

```python
# 用户视角的统一入口
from hyper_mve.configs import V4Config, EnvConfig, ModelConfig, TrainConfig, MupConfig, EvalConfig, LegacyConfig

# 不建议直接 import sub-module（保持封装）
# from hyper_mve.configs.env_config import EnvConfig  # 不推荐
```

### 3.5 backward-compat shim

`hyper_mve/config.py`（保留原文件名以兼容 v4.7 调用）：

```python
"""[v4 -> v4.7 backward-compat shim]
v4.7 单类 BaseConfig 已拆分为 V4Config 5 层结构。
此 shim 仅为支持 v4.7 训练脚本调用 (`_legacy_v4_7/scripts/`)。
v4 代码应直接 import V4Config。
"""
import warnings
from hyper_mve.configs import V4Config

warnings.warn(
    "hyper_mve.config.BaseConfig is deprecated. "
    "Use 'from hyper_mve.configs import V4Config' instead.",
    DeprecationWarning, stacklevel=2,
)

def _make_base_config_shim():
    """构造 v4.7 风格的 BaseConfig 类 (flat 70 attributes)."""
    cfg = V4Config.from_preset("medium")
    # 构造一个临时类，把所有 sub-config 字段 flatten 到 attribute
    class BaseConfig:
        pass
    instance = BaseConfig()
    for sub in (cfg.env, cfg.model, cfg.train, cfg.eval, cfg.legacy):
        for k, v in sub.__dict__.items() if hasattr(sub, '__dict__') else vars(sub).items():
            setattr(instance, k, v)
    return instance

BaseConfig = _make_base_config_shim
```

> ⚠️ 此 shim 是**最小实现**。v4.7 训练脚本如果调用了 v4 不存在的 attribute（如 freeze_enabled），需要回退到 LegacyConfig.

---

## 4. Edge Cases

| 场景 | 行为 |
|------|------|
| `V4Config.from_preset("invalid")` | ValueError |
| `V4Config.from_preset("medium").env.N = 8` | FrozenInstanceError |
| Easy / Medium / Hard 三 preset 字段不一致 | Pkg-08 实验脚本应明确指定 preset，禁止跨 preset 拷贝字段 |
| sub-config `__post_init__` 不一致（如 EnvConfig N=4 但 type_assignment 长度 3） | ValueError |
| `dataclasses.asdict(V4Config)` 输出嵌套 dict | OK，但 AgentType Enum 需手动转 int（`to_dict()` 已处理） |

---

## 5. Acceptance Criteria

### 5.1 单元测试

```python
# tests/configs/test_v4_config.py

def test_from_preset_medium():
    cfg = V4Config.from_preset("medium")
    assert cfg.env.N == 4
    assert cfg.env.K == 20
    assert cfg.env.T_max == 200
    assert cfg.preset_name == "medium"

def test_from_preset_invalid():
    with pytest.raises(ValueError, match="Unknown preset"):
        V4Config.from_preset("ultra")

def test_replace_propagation():
    cfg = V4Config.from_preset("medium")
    cfg_new = replace(cfg, train=replace(cfg.train, lr=3e-4))
    assert cfg.train.lr == 1e-4  # 原 cfg 不变
    assert cfg_new.train.lr == 3e-4
    assert cfg_new.env.N == 4    # 其他字段保留

def test_to_dict_json_serializable():
    import json
    cfg = V4Config.from_preset("medium")
    d = cfg.to_dict()
    s = json.dumps(d)
    assert len(s) > 100

def test_env_config_type_assignment_length():
    with pytest.raises(ValueError, match="type_assignment"):
        EnvConfig(N=4, L=16, K=20, M=3, T_max=200,
                  type_assignment=(AgentType.ALPHA,) * 3)  # 长度 3 != N=4

def test_train_config_curriculum_boundaries():
    with pytest.raises(ValueError, match="Curriculum"):
        TrainConfig(curriculum_stage_1_end_frac=0.5,
                    curriculum_stage_2_end_frac=0.4)  # s2 < s1

def test_model_config_role_dim_exact_fill():
    """v4 修订: d_role 精确等于 d_id+d_type+d_cap, 无 pad 余地."""
    # 默认值 8+8+16 = 32 通过
    cfg = ModelConfig()
    assert cfg.d_role == 32
    assert cfg.d_id_emb + cfg.d_type_emb + cfg.d_cap_emb == cfg.d_role
    
    # 不匹配应抛
    with pytest.raises(ValueError, match="d_role mismatch"):
        ModelConfig(d_role=32, d_id_emb=8, d_type_emb=4, d_cap_emb=16)
        # 8+4+16 = 28 != 32

def test_model_config_belief_dim_consistency():
    """v4 修订: d_belief = 2 × d_belief_proj (Ch4.2.4)."""
    cfg = ModelConfig()
    assert cfg.d_belief == 32
    assert cfg.d_belief_proj == 16
    assert cfg.d_belief == 2 * cfg.d_belief_proj
    
    with pytest.raises(ValueError, match="d_belief mismatch"):
        ModelConfig(d_belief=48, d_belief_proj=16)  # 48 != 2*16

def test_model_config_total_ctx_aug():
    """v4 修订: 总 ctx_aug = 16+32+32 = 80 (Ch4.2.4)."""
    cfg = ModelConfig()
    assert cfg.d_ctx_aug == 80
```

### 5.2 与 Ch3.9 Reference Config Table 对照

详见 [`06-difficulty-presets.md`](./06-difficulty-presets.md)。

### 5.3 性能

- `V4Config.from_preset("medium")` < 10 ms（一次性加载）
- `to_dict()` < 5 ms

---

## 6. Cross-references

- `../design.md` D3 (dataclass)、D6 (LegacyConfig)、D8 (preset 覆盖)
- `06-difficulty-presets.md`（三 preset 数值表）
- `07-legacy-archive-protocol.md`（backward-compat shim）
- Ch3.9 Reference Configurations
- Ch5.7 课程边界（0.3 / 0.7）
- Ch4.6 stability 参数 (output_scale, AdaLN)
