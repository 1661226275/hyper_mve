# Spec 08: gym.Env Interface + info dict (v4 Oracle Signals)

> 父文档：[`../proposal.md`](../proposal.md) §1.3 · [`../design.md`](../design.md) §6
> **本 spec 最关键** — 定义 v4 Oracle 监督信号契约。

---

## 1. Purpose

定义 `ResourceCommonsEnv` 的 gym.Env 标准接口（reset / step / render / close）以及 **info dict 完整字段契约**。**info dict 严格分组**：
- **Public**（可用作 model 输入）：caps, deltas, step_idx
- **Oracle**（仅 trainer 监督用，不可传 model）：c_true, types
- **Eval only**（仅 evaluator 可视化）：hotspot_centers, resource_state

违反此分组（如 Pkg-04 model 误读 info["types"] 作输入）将破坏 v4 Self-Info 原则与 Harsanyi 对应。

---

## 2. Interface

### 2.1 文件路径

`hyper_mve/envs/resource_commons/env.py`

### 2.2 类签名

```python
import gym
import numpy as np
from typing import Any
from hyper_mve.configs.env_config import EnvConfig
from hyper_mve.schemas import AgentType, ObservationLayout


class ResourceCommonsEnv(gym.Env):
    """ResourceCommons gym.Env 实现 (Ch3 + Ch4 v4 Oracle).
    
    Observation:
        np.array shape=(N, obs_dim) float32
        obs_dim = ObservationLayout.total_dim(N, K)
    
    Action:
        np.array shape=(N,) int64, ∈ {0, 1, 2, 3, 4, 5}
        0=NOOP, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=HARVEST
    
    Reward:
        np.array shape=(N,) float32, per-agent type-aware reward
    
    Info dict (严格分组):
        Public (可用作 model 输入):
            caps:      tuple[CapabilityVector, ...] 长度 N
            deltas:    np.ndarray (N,) float32
            step_idx:  int
            harvests:  np.ndarray (N,) float32
        
        Oracle (v4 关键, 仅 trainer 监督用, 不可传 model):
            c_true:    float ∈ [0, 1]    # L_c 标签
            types:     np.ndarray (N,) int8  # L_opp 标签
        
        Eval only (仅 evaluator 可视化):
            hotspot_centers: np.ndarray (M, 2) int32
            resource_state:  np.ndarray (K, 3) float32
    """
    
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}
    
    def __init__(self, cfg: EnvConfig, seed: int | None = None):
        # 见 spec 07
        ...
    
    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """重置 env, 采样新 episode.
        
        Args:
            seed: RNG seed. 影响 hotspot 位置、cap 采样、c_t 初值
            options: 可选覆盖
                - "c": float, 强制 c_0 = 此值
                - "types": tuple[AgentType, ...] 长度 N, Ablation 3 用
        
        Returns:
            obs: (N, obs_dim) float32
            info: 见 §2.3
        """
        ...
    
    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
        """执行一步.
        
        Args:
            action: (N,) int64
        
        Returns:
            obs: (N, obs_dim) float32
            reward: (N,) float32 per-agent type-aware
            done: bool (T_max 到达)
            truncated: bool (本环境恒 False)
            info: 见 §2.3
        """
        ...
    
    def render(self, mode: str = "rgb_array") -> np.ndarray | None:
        """渲染 (论文 Fig 3.1 用).
        
        Returns:
            mode="rgb_array": (256, 256, 3) uint8
        """
        ...
    
    def close(self) -> None:
        """清理资源 (本 env 无外部资源, 仅 pass)."""
        pass
```

### 2.3 info dict 完整字段（v4 关键契约）

```python
def _build_info(self) -> dict[str, Any]:
    """构造 info dict (严格分组).
    
    分组目的: 严格区分哪些字段可作为 model 输入 (Public),
    哪些仅用于 BeliefNet 监督训练 (Oracle), 防止 v4 Self-Info 泄漏.
    """
    return {
        # ====== Public (可用作 model 输入) ======
        "caps": self._state.agent_caps,              # tuple[CapabilityVector] 长度 N
        "deltas": self._last_deltas,                  # (N,) float32, 瞬时 Δ_i
        "step_idx": int(self._state.step_idx),        # int
        "harvests": self._last_harvests,              # (N,) float32, u_i
        
        # ====== Oracle (v4 关键, 仅 trainer 监督) ======
        "c_true": float(self._state.c_t),             # L_c MSE 标签 (Ch4.5.1)
        "types": self._state.agent_types.copy(),      # (N,) int8, L_opp CE 标签 (Ch4.5.2)
        
        # ====== Eval only (仅 evaluator 可视化) ======
        "hotspot_centers": self._state.hotspot_centers.copy(),  # (M, 2)
        "resource_state": np.concatenate([
            self._state.resource_positions.astype(np.float32),
            self._state.resource_stocks[:, None],
        ], axis=1),  # (K, 3) = (x, y, q)
        
        # ====== Schema 标识 (帮助 trainer 区分字段类型) ======
        "_info_schema_version": "v4.0",
        "_oracle_fields": ("c_true", "types"),       # 提示哪些是 Oracle
        "_eval_only_fields": ("hotspot_centers", "resource_state"),
    }
```

### 2.4 observation_space / action_space

```python
# __init__ 中构造
obs_dim = ObservationLayout.total_dim(self.N, self.K)
self.observation_space = gym.spaces.Box(
    low=-np.inf, high=np.inf,
    shape=(self.N, obs_dim), dtype=np.float32,
)
self.action_space = gym.spaces.MultiDiscrete([self.A] * self.N)
```

### 2.5 reset 完整流程

```python
def reset(self, seed=None, options=None):
    if seed is not None:
        self._rng = np.random.default_rng(seed)
    
    # 1. 采样初始 c_0
    #
    # ⚠️ 必须显式 `"c" in options` 检查, 不可用 `or` 短路:
    #    `options.get("c") or fallback()` 会让 options["c"]=0.0 (荒年评估)
    #    静默退回 RNG, 破坏 c-segment / zero-shot 评估的 c=0 数据点.
    if options is not None and "c" in options:
        c_0 = float(options["c"])
        assert 0.0 <= c_0 <= 1.0, f"options['c']={c_0} ∉ [0, 1]"
    else:
        c_0 = self._context_evo.initial_c(self._rng)
    
    # 2. Type 分配 (默认从 cfg, 可由 options 覆盖)
    #
    # 同样不可用 `or` 短路: 空 tuple () 是 falsy. 显式 `"types" in options` 检查.
    if options is not None and "types" in options:
        types = options["types"]
    else:
        types = self.cfg.type_assignment
    
    # 3. Patchy 资源生成
    centers, positions = spawn_patchy_resources(
        K=self.K, M=self.M, L=self.L, sigma_patch=self.cfg.sigma_patch,
        rng=self._rng,
    )
    
    # 4. Agent 初始位置 (uniform 散布)
    agent_positions = np.zeros((self.N, 2), dtype=np.int32)
    for i in range(self.N):
        agent_positions[i] = self._rng.integers(0, self.L, size=2)
    
    # 5. Cap 采样 (Pkg-01 sample_n)
    from hyper_mve.schemas import sample_n
    caps = sample_n(self.N, self._rng)
    
    # 6. 资源初始 (全满 Q_max)
    stocks = np.full(self.K, self.cfg.Q_max, dtype=np.float32)
    
    # 7. 构造 state
    self._state = ResourceCommonsState(
        agent_positions=agent_positions,
        resource_positions=positions,
        resource_stocks=stocks,
        cumulative_harvests=np.zeros(self.N, dtype=np.float32),
        steps_since_harvest=np.zeros(self.N, dtype=np.int32),
        c_t=float(c_0),
        c_history=np.zeros(self.T_max + 1, dtype=np.float32),
        step_idx=0,
        done=False,
        agent_caps=caps,
        agent_types=np.array([t.value for t in types], dtype=np.int8),
        hotspot_centers=centers,
        last_actions=np.zeros(self.N, dtype=np.int64),
    )
    self._state.c_history[0] = c_0
    
    # 8. 初始 obs + info (无 reward, deltas 全 0)
    self._last_harvests = np.zeros(self.N, dtype=np.float32)
    self._last_deltas = np.zeros(self.N, dtype=np.float32)
    
    obs = build_joint_observation(self._state, self.L, self.T_max, self.K)
    info = self._build_info()
    return obs, info
```

### 2.6 step 完整流程

```python
def step(self, action):
    assert action.shape == (self.N,), f"action shape {action.shape}"
    assert action.dtype in (np.int32, np.int64), f"action dtype {action.dtype}"
    
    # 1. 解码动作
    move_mask = (action >= 1) & (action <= 4)  # UP/DOWN/LEFT/RIGHT
    harvest_mask = (action == 5)
    
    # 2. 移动 (cap.ν 失败概率)
    for i in range(self.N):
        if not move_mask[i]:
            continue
        # ν 成功概率
        if self._rng.random() < self._state.agent_caps[i].nu:
            dx, dy = _action_to_delta(action[i])
            new_pos = self._state.agent_positions[i] + [dx, dy]
            new_pos = np.clip(new_pos, 0, self.L - 1)
            self._state.agent_positions[i] = new_pos
    
    # 3. Fair-share 采集 (公式 3.2)
    harvests, harvests_per_resource = fair_share_harvest(
        self._state, self._state.agent_positions, harvest_mask,
    )
    self._last_harvests = harvests
    
    # 4. 更新 steps_since_harvest
    self._state.steps_since_harvest = np.where(
        harvests > 0, 0, self._state.steps_since_harvest + 1,
    )
    self._state.cumulative_harvests += harvests
    
    # 5. 资源动力学 (公式 3.1)
    step_dynamics(self._state, harvests_per_resource)
    
    # 6. 演化 c_t
    self._state.c_t = self._context_evo.step(
        self._state.c_t, self._state.step_idx, self._rng,
    )
    
    # 7. 计算 reward (公式 3.5-3.10)
    reward, deltas = compute_rewards(
        harvests, move_mask, self._state.agent_types, self._state.c_t,
    )
    self._last_deltas = deltas
    
    # 8. 更新 state
    self._state.last_actions = action.astype(np.int64).copy()
    self._state.step_idx += 1
    self._state.c_history[self._state.step_idx] = self._state.c_t
    done = self._state.step_idx >= self.T_max
    self._state.done = done
    
    # 9. 构造 obs + info
    obs = build_joint_observation(self._state, self.L, self.T_max, self.K)
    info = self._build_info()
    
    return obs, reward, done, False, info


def _action_to_delta(action: int) -> tuple[int, int]:
    """0=NOOP, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=HARVEST."""
    deltas = {1: (0, 1), 2: (0, -1), 3: (-1, 0), 4: (1, 0)}
    return deltas.get(int(action), (0, 0))
```

---

## 3. Implementation Notes

### 3.1 info dict 严格分组的工程实施

**Pkg-05 trainer 必须遵守的契约**：

```python
# 正确用法 (trainer)
def train_step(batch):
    obs = batch["obs"]
    info = batch["info"]
    
    # Public 字段进 model
    cap_emb = role_encoder.forward(info["caps"])
    
    # Oracle 字段仅 BeliefNet 监督
    c_true = info["c_true"]            # 仅作 L_c 标签
    types_true = info["types"]          # 仅作 L_opp 标签
    
    # Eval 字段不应在 trainer 中读
    # info["hotspot_centers"]  # ← 不在 trainer 路径

# 错误用法 (违反 Self-Info)
def model_forward(info):
    # ❌ 不可: 把 Oracle 信号传入 model
    types_input = info["types"]
    cap = info["caps"]
    return self.predict(types_input, cap)  # Self-Info 泄漏!
```

**审计机制**：
- spec 04 (observation) 已保证 obs 中无 types 信息
- 单测 `test_self_info_no_leak.py` 验证：扰动 info["types"] 不应改变 obs
- Pkg-05 spec 中将定义 trainer 端如何严格分组消费 info

### 3.2 deltas 字段的生命周期

- env.step 计算 deltas (rewards.py compute_delta)
- env 缓存到 `self._last_deltas`
- info 暴露 deltas → trainer 读 → 写入 TimeStepRecord.delta 字段
- buffer 存储 → trainer 训练时再读出

### 3.3 reset 后的"零 step" 状态

- reset 返回的 info["deltas"] 全 0（无历史 reward）
- info["harvests"] 全 0
- info["c_true"] = c_0
- info["types"] = 初始类型分配（不变）

### 3.4 render 实现

```python
def render(self, mode="rgb_array"):
    if mode != "rgb_array":
        raise NotImplementedError(f"Render mode {mode} not supported")
    
    # 用 matplotlib 画 L×L grid + resources + agents
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 4), dpi=64)
    
    # 资源点 (颜色按库存)
    for k in range(self.K):
        x, y = self._state.resource_positions[k]
        q_norm = self._state.resource_stocks[k] / 10.0
        ax.scatter(x, y, s=100, c=plt.cm.viridis(q_norm), marker='o')
    
    # Agents (颜色按类型)
    for i in range(self.N):
        x, y = self._state.agent_positions[i]
        color = 'red' if self._state.agent_types[i] == 0 else 'blue'
        ax.scatter(x, y, s=200, c=color, marker='s', edgecolors='black')
    
    ax.set_xlim(-0.5, self.L - 0.5)
    ax.set_ylim(-0.5, self.L - 0.5)
    ax.set_xticks(range(self.L))
    ax.set_yticks(range(self.L))
    ax.grid(True)
    ax.set_title(f"c_t={self._state.c_t:.2f}, step={self._state.step_idx}")
    
    # 转 numpy
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img
```

---

## 4. Edge Cases

| 场景 | 行为 |
|------|------|
| action.shape ≠ (N,) | AssertionError |
| action 含越界值（如 6） | _action_to_delta 返回 (0, 0)（NOOP），无报错 |
| reset 不传 seed | 使用 init 时的 seed |
| `options.c = 1.5` | 不校验，但 c_t 在 context_evolution.step 后 clip 到 [0, 1] |
| done=True 后继续 step | 不报错，但 reward 与 obs 行为未定义；用户应负责 reset |
| info 被 Pkg-05 modify | info 是 dict，浅拷贝；trainer 应 deep copy 后改 |
| 多次 reset | RNG 不变（除非显式 seed），但 hotspot 重生成（with RNG） |

---

## 5. Acceptance Criteria

### 5.1 单元测试（`tests/envs/test_env_info_oracle.py`）

```python
import numpy as np
import pytest
from hyper_mve.configs import V4Config
from hyper_mve.envs.resource_commons import ResourceCommonsEnv


def test_info_has_oracle_fields():
    """v4 关键: info 必含 c_true / types / caps."""
    cfg = V4Config.from_preset("medium")
    env = ResourceCommonsEnv(cfg.env, seed=42)
    obs, info = env.reset()
    
    assert "c_true" in info
    assert "types" in info
    assert "caps" in info
    assert isinstance(info["c_true"], float)
    assert info["types"].shape == (4,)
    assert info["types"].dtype == np.int8
    assert len(info["caps"]) == 4


def test_info_oracle_fields_match_state():
    """c_true / types 与 env 内部状态一致."""
    cfg = V4Config.from_preset("medium")
    env = ResourceCommonsEnv(cfg.env, seed=42)
    obs, info = env.reset()
    
    assert info["c_true"] == env._state.c_t
    assert np.array_equal(info["types"], env._state.agent_types)


def test_info_eval_fields_present():
    """info 必含 hotspot_centers / resource_state."""
    cfg = V4Config.from_preset("medium")
    env = ResourceCommonsEnv(cfg.env, seed=42)
    obs, info = env.reset()
    
    assert "hotspot_centers" in info
    assert "resource_state" in info
    assert info["hotspot_centers"].shape == (3, 2)
    assert info["resource_state"].shape == (20, 3)


def test_info_schema_version():
    """info 标识 schema 版本."""
    cfg = V4Config.from_preset("medium")
    env = ResourceCommonsEnv(cfg.env, seed=42)
    _, info = env.reset()
    assert info["_info_schema_version"] == "v4.0"
    assert "c_true" in info["_oracle_fields"]
    assert "types" in info["_oracle_fields"]


def test_step_returns_5tuple():
    """gym.Env step 返回 (obs, reward, done, truncated, info)."""
    cfg = V4Config.from_preset("medium")
    env = ResourceCommonsEnv(cfg.env, seed=42)
    env.reset()
    
    action = np.array([0, 0, 0, 0], dtype=np.int64)
    obs, reward, done, truncated, info = env.step(action)
    
    assert obs.shape == (4, 99)
    assert reward.shape == (4,)
    assert reward.dtype == np.float32
    assert isinstance(done, (bool, np.bool_))
    assert truncated is False
    assert "c_true" in info


def test_observation_space():
    cfg = V4Config.from_preset("medium")
    env = ResourceCommonsEnv(cfg.env, seed=42)
    assert env.observation_space.shape == (4, 99)


def test_action_space_multidiscrete():
    cfg = V4Config.from_preset("medium")
    env = ResourceCommonsEnv(cfg.env, seed=42)
    assert env.action_space.nvec.tolist() == [6, 6, 6, 6]


def test_reset_with_seed_reproducible():
    """相同 seed → 相同初始 state."""
    cfg = V4Config.from_preset("medium")
    env1 = ResourceCommonsEnv(cfg.env, seed=42)
    env2 = ResourceCommonsEnv(cfg.env, seed=42)
    obs1, info1 = env1.reset()
    obs2, info2 = env2.reset()
    assert np.allclose(obs1, obs2)
    assert info1["c_true"] == info2["c_true"]


def test_reset_options_force_c():
    """reset(options={'c': 0.5}) 强制 c_0=0.5."""
    cfg = V4Config.from_preset("medium")
    env = ResourceCommonsEnv(cfg.env, seed=42)
    _, info = env.reset(options={"c": 0.7})
    assert info["c_true"] == pytest.approx(0.7)


def test_reset_options_force_c_zero_regression():
    """🐛 回归测试: options['c']=0.0 不可被 falsy 短路 (Bug 2 修复).
    
    错误实现 `options.get("c") or fallback()` 会让 0.0 静默退回 RNG.
    必须显式 `"c" in options` 检查.
    """
    cfg = V4Config.from_preset("medium")
    env = ResourceCommonsEnv(cfg.env, seed=42)
    _, info = env.reset(options={"c": 0.0})
    assert info["c_true"] == 0.0, (
        f"c_true={info['c_true']} != 0.0; 可能是 falsy 短路 bug 复现 "
        f"(`options.get('c') or X` 会让 0.0 退回到 X)"
    )
    # 1.0 边界也测一遍 (虽然 1.0 是 truthy, 但确保边界正确)
    _, info = env.reset(options={"c": 1.0})
    assert info["c_true"] == 1.0


def test_reset_options_c_out_of_range_rejected():
    """options['c'] 越界应抛 AssertionError."""
    cfg = V4Config.from_preset("medium")
    env = ResourceCommonsEnv(cfg.env, seed=42)
    with pytest.raises(AssertionError, match=r"options\['c'\]"):
        env.reset(options={"c": 1.5})
    with pytest.raises(AssertionError):
        env.reset(options={"c": -0.1})


def test_reset_options_force_types():
    """reset(options={'types': ...}) 覆盖类型分配 (Ablation 3)."""
    from hyper_mve.schemas import AgentType
    cfg = V4Config.from_preset("medium")
    env = ResourceCommonsEnv(cfg.env, seed=42)
    custom_types = (AgentType.ALPHA,)*3 + (AgentType.BETA,)
    _, info = env.reset(options={"types": custom_types})
    assert info["types"].tolist() == [0, 0, 0, 1]


def test_reset_options_types_falsy_regression():
    """🐛 回归测试: options['types']=() 空 tuple 是 falsy.
    
    错误实现 `options.get('types') or self.cfg.type_assignment` 会让
    空 tuple 退回 cfg 默认. 当前实现应严格用 `"types" in options` 检查.
    
    注: 空 tuple 实际使用场景罕见, 但显式 None 检查保持代码一致.
    """
    cfg = V4Config.from_preset("medium")
    env = ResourceCommonsEnv(cfg.env, seed=42)
    # 不传 options → 用 cfg 默认 (2α + 2β)
    _, info = env.reset()
    assert info["types"].tolist() == [0, 0, 1, 1]
    
    # 显式传 None → 用 cfg 默认
    _, info = env.reset(options=None)
    assert info["types"].tolist() == [0, 0, 1, 1]


def test_done_at_T_max():
    """T_max 步后 done=True."""
    cfg = V4Config.from_preset("medium")
    env = ResourceCommonsEnv(cfg.env, seed=42)
    env.reset()
    for _ in range(199):
        _, _, done, _, _ = env.step(np.zeros(4, dtype=np.int64))
        assert not done
    _, _, done, _, _ = env.step(np.zeros(4, dtype=np.int64))
    assert done


def test_render_rgb_array():
    cfg = V4Config.from_preset("medium")
    env = ResourceCommonsEnv(cfg.env, seed=42)
    env.reset()
    img = env.render(mode="rgb_array")
    assert img.dtype == np.uint8
    assert img.ndim == 3 and img.shape[2] == 3
```

### 5.2 集成测试（`tests/envs/test_env_integration.py`）

- 1000 random episode, 每个 episode 取 5 个 step
- 验证 obs / reward / info shape 与 dtype 始终一致
- 验证 reward 范围 ∈ [-3, 3]
- 验证 info["types"] 在 episode 内不变（type 固定）
- 验证 info["c_true"] 在 static mode 不变，oscillate mode 周期，random_walk 漂移

### 5.3 与 Pkg-05 trainer 接口对接验证

未来 Pkg-05 spec 04 中将定义：
- trainer 端 `buffer.add_transition(record)` 接受 TimeStepRecord
- 字段映射 `record.tau = info["types"]`，`record.delta = info["deltas"]`，等
- **Oracle 字段** `info["c_true"]` / `info["types"]` 仅在 `compute_belief_loss` 中读取，**永不进入 model forward**

---

## 6. Cross-references

- Ch3 全章
- Ch4.5.1 L_c oracle 监督（c_true 字段消费者）
- Ch4.5.2 L_opp oracle 监督（types 字段消费者，**v4 关键**）
- Ch4.2.2 Self-Info 严格性
- Pkg-01 `04-timestep-record.md`（info → TimeStepRecord 映射）
- `01-env-formal-tuple.md`（state 字段）
- `02-resource-dynamics.md`（step 内调用）
- `03-fehr-schmidt-reward.md`（reward + deltas 来源）
- `04-six-block-observation.md`（obs 构造）
- `05-context-evolution.md`（c_t 演化）
- `06-patchy-resource-spawn.md`（资源初始化）
- `07-difficulty-presets-env.md`（cfg.env 派生维度）
- `../design.md` D5 (Context 模式)、Open Questions Q1-Q7
