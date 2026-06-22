# Pkg-07 Proposal: Baselines (Internal + External) — 断言对照基线 + 外部范式对比

> 配套阅读：[`README.md`](./README.md) · [`design.md`](./design.md)（**先读 proposal 再读 design**）
> **Supersedes**：[`pkg-06-baselines/proposal.md`](../pkg-06-baselines/proposal.md) — 继承其 4 条断言论证 + 5 internal variant 框架，扩展为外部范式对照族

---

## 1. Why（为什么需要本包）

### 1.1 论文 4 条断言需要内部对照基线才能成立（继承 pkg-06 §1.1）

Hyper-MuZero v4 的 4 条核心可证伪断言每条都不是"我们好"，而是"**去掉某个机制就出现某个失败模式**"。没有对照基线，断言就是空话。

| 断言 | 主张 | 失败模式（去掉机制后）| 需要的 internal 对照 |
|------|------|---------------------|---------------------|
| **A 类型梯度撕裂** | hyper_rew 按 type 生成 θ_rew^i,消除梯度撕裂 | 共享 RewardHead 学到 α/β 的"平均梯度" | `ma_muzero` + `rewardhead_explicit_type` |
| **B 生成范围谱**（B′）| per-context θ 在最优生成范围内给 belief 组合专用容量 | 谱左端容量被摊薄 / 谱右端坍缩 | `input_wide` + `input_deep` |
| **C 三路必要性** | c_ctx / role_i / belief_i 三路缺一不可 | 移除 belief 路 → 性能掉 | `no_belief` |
| **D planner 双技术** | MVE+CRN+coord-descent 联合解 SNR 崩塌 | 单技术 → planner 退化为均匀 | （Pkg-08 planner ablation,**非 baseline model**）|

→ 断言 A/B/C 的证伪压力**直接**落在 5 个 internal 对照模型上（继承 pkg-06）。

### 1.2 论文还需要**外部范式 baseline** 才能说"hyper 在 MARL benchmark 上有竞争力"

仅有内部消融不足以让审稿人信服 hyper 在 MARL 领域有 SOTA-比较意义。pkg-06 的 internal variant 都共用 `MuZeroTrainer` —— 这证明"机制是必要的",但不证明"hyper 总体上 vs MARL 经典方法（policy gradient / Q decomposition / model-based MARL）有竞争力"。审稿人会问：

> "你的 hyper 跟 MAPPO/QMIX 比谁强？跟 model-based MARL（MA-MuZero）比？"

无外部 baseline → 论文的 contribution 缺少**横向坐标**。Pkg-07 扩展 pkg-06 引入：

| 范式 | Tier | 来源 | 解决什么对比问题 |
|------|------|------|-----------------|
| **MAPPO**（CTDE policy gradient）| Tier-1 | port from `D:\RL\lzj\MAPPO` | on-policy MARL 主流；与 hyper 的 model-based 走线对比 |
| **QMIX**（值分解）| Tier-1 | vendor from PyMARL `oxwhirl/pymarl` | 单调因子分解的 SOTA；与 hyper 的 per-agent reward 对比 |
| **MA-MuZero (GH-port)**（model-based MARL）| Tier-1 | vendor from `werner-duvaud/muzero-general` + MA wrapper | **最直接的近邻范式**：同为 MuZero 系；与 hyper 的 per-context θ 对比 |
| **MAMBA**（model-based with belief）| Tier-2 | active sourcing（原论文 → 社区 fork）| 与 hyper 的 BeliefNet 对比；可能 sourcing 失败转 stub |
| **MARIE / GA** | Stub | reserved CLI names + `NotImplementedError` | 占位以便未来 follow-up；不进 main table |

### 1.3 等参公平性是断言 B 的生死线（内部继承 pkg-06；外部松绑）

**Internal**：pkg-06 §1.2 锁定的严格等参规则（仅条件化消费子系统 + 双阈值 5%/10%）原封继承。input_wide/deep 必须与 hyper 等参,否则断言 B 直接失效。

**External**：外部范式与 hyper 没有"共同条件化子系统"可对齐 —— MAPPO 没有 hypernet,QMIX 没有 RewardHead,MA-MuZero-GH 没有 BeliefNet。强行参数对齐毫无意义。pkg-07 改用**披露式公平**：

> 对每个外部 baseline 报告 (参数量, wall-clock-to-converge, LR-swept best, final return ≥5 seeds)。Methods 表披露所有维度,审稿人自行判断公平度。

这是诚实的折中:不能装作两者等参,但可以保证两者**各自最强**版本对比。

### 1.4 复用同一训练设施 vs. 接入自带训练器

- **Internal variant** 继承 pkg-06 五条复用约束（同一 MuZeroTrainer/Worker/Buffer/compose_total_loss,仅 model 类不同）—— 这是 pkg-06 已锁定且经过审阅的契约。
- **External variant** 必然自带训练器（MAPPO 是 PPO,QMIX 是 DQN-with-mixer,MA-MuZero 是 MCTS+学习）——硬要让它们用 MuZeroTrainer 是范式错位。pkg-07 引入新协议:`ExternalBaselineRunner` 自负其 trainer/buffer/loss,但通过**单点 PettingZoo 适配器**消费同一 `ResourceCommonsEnv` + 通过**统一 `.evaluate(env_fn, c_grid, episodes) -> EvalReport`** 喂给 Pkg-08。

### 1.5 PettingZoo 适配器:架构 keystone

外部 baseline 期待 MPE / PettingZoo / gym MultiAgentEnv 风格的环境接口。直接让每个 external runner 自己 wrap `ResourceCommonsEnv` 会:
1. 产生大量重复代码（3+ 处 env-coupling 重复）
2. 各 runner 可能对 oracle 信息（`info["types"]`,`info["c_true"]`）做不同处理 → 公平性事故
3. heterogeneous 类型信息可能被 RLlib-like 库误判为同质 → 性能 unfair

**单一适配器 `ResourceCommonsPettingZooEnv` at `hyper_mve/envs/adapters/pettingzoo_wrapper.py`**:
- 实现 `pettingzoo.ParallelEnv`(`reset` 返回 obs dict, `step` 收 action dict 返回 obs/reward/done/info dicts)
- 通过 `oracle_mode: bool = False` 旗标**默认不暴露** `info["types"]` / `info["c_true"]`；只有 internal hyper 评估（带 oracle context 注入）才允许翻为 True
- 每 agent obs 保留自己的 `cap_i`（合法：agent 看自己的能力是合法 obs；与 hyper 的 set_context_subjective 同语义）
- spec 04 强制 `test_adapter_info_gating.py`（two-flag 正交性 + 4 leak-surface + reset/step 双路径 + N-parametric over {Easy, Medium}）

→ 4 个 external runner（含 Tier-2 MAMBA）全部消费同一适配器。零代码重复,oracle 公平性单点维护。

### 1.6 Tier 化以管理 GPU 与日历预算

外部范式 baseline 实施成本高：vendoring + 自带 trainer 移植 + 离散动作适配 + 多 LR 多 seed 跑 sweep。3 Tier-1 + 1 Tier-2 + 2 stub 的分层把"必须 vs 可选"分清:
- **Tier-1（MAPPO/QMIX/MA-MuZero）必须 ship**:论文 Methods 表的横向坐标
- **Tier-2（MAMBA）尽力 ship**:与 hyper 最相似（model-based + belief）,但若 sourcing 失败可降级 stub,不阻塞包出
- **Stub（MARIE / GA）保留**:占 CLI 名 + `NotImplementedError`,未来可扩展

GPU 预算（详见 [Plan File §"GPU budget"]）：Tier-1 = 3 baseline × 3 LR × 5 seed × Easy+Medium ≈ 90 LR-sweep runs（≈ 200 GPU-hr）+ MAMBA 若 sourced 增 ≈ 50 GPU-hr。pkg-07 实施期总预算控制在 250 GPU-hr 内（不含 Pkg-08 全表）。

### 1.7 pkg-06 SDD 必须显式 supersede

pkg-06 SDD（status: Draft）锁定了 `create_baseline_model(cfg, variant)` + 11 单测 + 13 C6-* 约束。若 pkg-07 不 supersede,实施期会:
- grep 同时命中 pkg-06 `create_baseline_model` 与 pkg-07 `create_baseline`
- 工厂签名两套并存,CLI dispatch drift
- pkg-08 不知道该消费哪份契约

pkg-07 spec 01 头部强制 supersede 声明（pkg-06 README 已加 banner）。Alias 仍可用,加 `DeprecationWarning`。

---

## 2. What Changes（具体改动）

> **本包只写 SDD 文档,不写实现代码**。以下"改动"描述的是 SDD 锁定的**待实施接口**,供 pkg-07 实施期与 pkg-08 消费。

### 2.1 新增（1 工厂 + 1 共享后端 + 1 适配器 + 5 internal + 3 Tier-1 external + Tier-2/stub）

#### 2.1.1 `hyper_mve/baselines/__init__.py`(统一工厂 + REGISTRY + supersede shim)

```python
def create_baseline(cfg, variant: str) -> Union[BaselineModel, ExternalBaselineRunner]:
    """统一 baseline 工厂。
    
    Internal variant ∈ {input_wide, input_deep, ma_muzero, no_belief, rewardhead_explicit_type}:
      → 返回 BaselineModel（7-API,MuZeroTrainer 消费）
    
    External variant ∈ {external_mappo, external_qmix, external_ma_muzero_gh,
                        external_mamba, external_marie, external_ga}:
      → 返回 ExternalBaselineRunner（自带 trainer,PettingZoo 适配器消费）
      → MARIE/GA/MAMBA-未 sourced 抛 NotImplementedError
    
    "hyper" 不走本工厂（直接 HyperMuZeroModel(cfg)）。
    """
    if variant == "hyper":
        raise ValueError("'hyper' uses HyperMuZeroModel directly, not this factory")
    if variant in INTERNAL_REGISTRY:
        return INTERNAL_REGISTRY[variant](cfg)
    if variant in EXTERNAL_REGISTRY:
        return EXTERNAL_REGISTRY[variant](cfg)
    raise ValueError(f"Unknown baseline variant: {variant!r}")

# pkg-06 alias（DeprecationWarning）
def create_baseline_model(cfg, variant):
    import warnings
    warnings.warn(
        "create_baseline_model is renamed to create_baseline (pkg-07). "
        "This alias will be removed in a future release.",
        DeprecationWarning, stacklevel=2,
    )
    return create_baseline(cfg, variant)

REGISTRY: Mapping[str, Callable] = MappingProxyType({**INTERNAL_REGISTRY, **EXTERNAL_REGISTRY})
```

**最终契约声明**（spec 01 头部强制）:本签名是 baseline 工厂的最终契约,**supersedes** pkg-06 `create_baseline_model`。

#### 2.1.2 `hyper_mve/baselines/shared_backbones.py`（继承 pkg-06,仅 internal 用）

继承 pkg-06 spec 02 内容,逐字保留 3 个 `create_*` 工厂 + 等参共享契约。external runner 不消费本模块。

#### 2.1.3 5 个 internal baseline 模型类（继承 pkg-06,迁入 `internal/` 子包）

| 工厂 variant | 模型类 | 断言 | 一句话（继承 pkg-06）|
|--------------|--------|------|---------|
| `input_wide` | `InputWideBaselineModel` | B' | 无 hypernet,3 功能网**加宽**,concat ctx_aug 进 input |
| `input_deep` | `InputDeepBaselineModel` | B' | 无 hypernet,3 功能网**加深** |
| `ma_muzero` | `MAMuZeroBaselineModel` | A | vanilla MARL,**共享单一 RewardHead**,agent_id+type 进 input |
| `no_belief` | `NoBeliefBaselineModel` | C / Abl7 | HyperMuZero 变体,belief 路在 TriContextEncoder 内置零 |
| `rewardhead_explicit_type` | `ExplicitTypeRewardBaselineModel` | A / Abl6.x | 有 hyper_{trans,pred},**无 hyper_rew**,RewardHead 显式按 type 分支 |

5 internal variant 实现 7-API（继承 pkg-04 spec02）+ pkg-06 13 C6-* 约束。

#### 2.1.4 3 个 Tier-1 external baseline（**新增**）

| 工厂 variant | 模型类 | 范式 | 来源 | 一句话 |
|--------------|--------|------|------|--------|
| `external_mappo` | `MAPPOAlgorithm` | on-policy CTDE PG | port from `D:\RL\lzj\MAPPO\MAPPO_main.py` | 集中 critic 走 concat-obs（合法 CTDE 不算泄漏）, share-policy by default |
| `external_qmix` | `QMIXAlgorithm` | 值分解 + DQN-with-mixer | vendor from `oxwhirl/pymarl` | 单调因子分解,共享 GRU agent net,mixer over global state |
| `external_ma_muzero_gh` | `MAMuZeroGHAlgorithm` | model-based MCTS | vendor `werner-duvaud/muzero-general` + thin MA wrapper | 与 hyper 最近邻范式（同 MuZero 系）；MA wrapper:per-agent learner,共享世界模型,平均 reward 估计 |

每个 external runner 实现 `ExternalBaselineRunner` 协议（spec 08 spec 01 + spec 08 锁定）:
```python
class ExternalBaselineRunner(Protocol):
    def train(self, cfg, env_fn) -> None: ...                                       # 自带训练循环
    def evaluate(self, env_fn, c_grid, episodes) -> EvalReport: ...                  # 与 internal 统一
    def save_checkpoint(self, path: Path) -> None: ...
    def load_checkpoint(self, path: Path) -> None: ...
```

#### 2.1.5 PettingZoo 适配器（**新增,架构 keystone**）

`hyper_mve/envs/adapters/pettingzoo_wrapper.py::ResourceCommonsPettingZooEnv`,实现 `pettingzoo.ParallelEnv`：

```python
class ResourceCommonsPettingZooEnv(ParallelEnv):
    """N-parametric (cfg.N ∈ {2,4,8}) PettingZoo ParallelEnv wrapping ResourceCommonsEnv.

    Two-flag info gating taxonomy:
      - oracle_mode:    gates env's _oracle_fields = ('c_true', 'types').
                        Only HyperMuZero internal eval flips True (oracle context injection).
      - eval_info_mode: gates env's _eval_only_fields = ('hotspot_centers', 'resource_state').
                        Only the Pkg-08 evaluator's metric-collection path flips True.
                        External baseline TRAINING runs keep both False.

    Schema-marker tuples ('_oracle_fields', '_eval_only_fields', '_info_schema_version')
    are themselves stripped from external-facing info, so an external runner cannot
    introspect them to reconstruct what fields existed.
    """
    def __init__(self, env_cfg, oracle_mode: bool = False, eval_info_mode: bool = False):
        self._env = ResourceCommonsEnv(env_cfg)
        self._oracle_mode = oracle_mode
        self._eval_info_mode = eval_info_mode
        self._N = self._env.N                                    # env.py:84 self.N = cfg.N
        self.agents = [f"agent_{i}" for i in range(self._N)]
        ...
    
    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        obs_arr, info = self._env.reset(seed=seed, options=options)
        obs_dict = {f"agent_{i}": obs_arr[i] for i in range(self._N)}
        info_dict = self._filter_info(info, per_agent=True)
        return obs_dict, info_dict
    
    def step(self, action_dict: Dict[str, int]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        action_arr = np.array([action_dict[f"agent_{i}"] for i in range(self._N)])
        obs_arr, reward_arr, term, trunc, info = self._env.step(action_arr)
        ...
        info_dict = self._filter_info(info, per_agent=True)      # SAME filter on step() as reset()
        return obs_dict, reward_dict, term_dict, trunc_dict, info_dict
    
    def _filter_info(self, info: dict, per_agent: bool) -> dict:
        """Two-flag info gate. Reads env's published schema markers — DRY against env evolution."""
        drop: set[str] = set()
        if not self._oracle_mode:
            drop.update(info.get("_oracle_fields", ()))           # ('c_true', 'types')
        if not self._eval_info_mode:
            drop.update(info.get("_eval_only_fields", ()))        # ('hotspot_centers', 'resource_state')
        # Always strip the schema markers themselves — they leak structure even if values gated.
        drop.update({"_info_schema_version", "_oracle_fields", "_eval_only_fields"})
        kept = {k: v for k, v in info.items() if k not in drop}
        return {f"agent_{i}": kept for i in range(self._N)} if per_agent else kept
```

**CTDE-legitimacy 边界（footnote）**:
- **LEGAL global state**（中心 critic / mixer 可消费）= `concat([obs_i for i in agents])` ± action one-hots ± `caps`（public）。是 agents 联合已观测信息,不带特权。
- **PRIVILEGED 状态**（必须 gated）= `c_true`（ground-truth context）/ `types`（per-agent type label）/ `resource_state`（每 resource 的精确位置+stock，包括 agents 本地 sensor radius 外）/ `hotspot_centers`（spawn distribution 参数）。
- **规则**：external baseline 需要 centralized critic 时,**只能** consume `concat(obs)` 派生量；**不可** 触碰 `info["resource_state"]` 或 `info["hotspot_centers"]`。适配器默认 enforce；只有 Pkg-08 evaluator 的 metric-collection 路径才 `eval_info_mode=True`（只读，绝不喂回 policy/critic）。

强制单测 `tests/baselines/external/test_adapter_info_gating.py`（重命名自 `test_adapter_no_oracle_leak.py`，覆盖面扩大):
```python
import pytest
from hyper_mve.envs.adapters.pettingzoo_wrapper import ResourceCommonsPettingZooEnv

LEAK_SURFACE = ("c_true", "types", "hotspot_centers", "resource_state")
SCHEMA_MARKERS = ("_oracle_fields", "_eval_only_fields", "_info_schema_version")

@pytest.mark.parametrize("preset_name", ["easy", "medium"])   # N=2 and N=4; smoke must pass for both
@pytest.mark.parametrize("path", ["reset", "step"])           # _build_info called on both (env.py:280)
def test_defaults_strip_all_leak_surface_and_markers(preset_name, path):
    env_cfg = load_preset(preset_name).env
    env = ResourceCommonsPettingZooEnv(env_cfg, oracle_mode=False, eval_info_mode=False)
    if path == "reset":
        _, info = env.reset()
    else:
        env.reset()
        _, _, _, _, info = env.step({a: 0 for a in env.agents})    # NOOP joint action
    for a in env.agents:
        for k in LEAK_SURFACE + SCHEMA_MARKERS:
            assert k not in info[a], f"leak: {k!r} present on {path} under defaults"

def test_oracle_mode_exposes_only_oracle_fields():
    env = ResourceCommonsPettingZooEnv(env_cfg, oracle_mode=True, eval_info_mode=False)
    _, info = env.reset()
    for a in env.agents:
        assert "c_true" in info[a] and "types" in info[a]
        assert "hotspot_centers" not in info[a] and "resource_state" not in info[a]   # orthogonal

def test_eval_info_mode_exposes_only_eval_fields():
    env = ResourceCommonsPettingZooEnv(env_cfg, oracle_mode=False, eval_info_mode=True)
    _, info = env.reset()
    for a in env.agents:
        assert "hotspot_centers" in info[a] and "resource_state" in info[a]
        assert "c_true" not in info[a] and "types" not in info[a]                     # orthogonal

def test_agent_ids_are_N_parametric():
    for preset_name, expected_N in [("easy", 2), ("medium", 4), ("hard", 8)]:
        env_cfg = load_preset(preset_name).env
        env = ResourceCommonsPettingZooEnv(env_cfg)
        assert env.agents == [f"agent_{i}" for i in range(expected_N)]
```

### 2.2 7-API 一致性（继承自 pkg-06,仅 internal）

5 internal variant 实现 7 方法（与 Pkg-04 spec02 逐字一致):
```python
update_step(global_step: int) -> None
set_context_objective(c_t: Tensor) -> None
set_context_subjective(agent_id: int, cap_i: Tensor, belief: tuple) -> None
encode(obs: Tensor) -> Tensor
transition(s: Tensor, action: Tensor) -> Tensor
predict_reward(s: Tensor, action: Tensor) -> Tensor
predict(s: Tensor) -> tuple[Tensor, Tensor]
```

External runner 不实现 7-API（范式异）。

### 2.3 stateful + Self-Info 严格（继承 pkg-06,仅 internal）

逐字继承 pkg-06 §2.3。

### 2.4 待声明 cfg 字段（消费态声明，须 Pkg-01 spec05 同步,本包不改上游）

**5 字段穷举**（4 internal 继承 pkg-06 D10 + 1 external 新增）:

```
cfg.baselines.internal_wide_hidden_dim                # int  — input_wide 加宽宽度
cfg.baselines.internal_deep_layers                    # int  — input_deep 加深层数
cfg.baselines.internal_ma_muzero_share_pred_head      # bool — ma_muzero 是否共享 pred 头
cfg.baselines.internal_explicit_type_branches         # int  — rewardhead_explicit_type 分支数
cfg.baselines.external_lr_sweep_grid                  # Mapping[str, tuple[float, ...]]
                                                      #   per-baseline LR sweep (例:
                                                      #   {"external_mappo": (1e-4, 3e-4, 1e-3),
                                                      #    "external_qmix":  (1e-4, 3e-4, 1e-3),
                                                      #    "external_ma_muzero_gh": (1e-4, 3e-4, 1e-3)})
```

新 namespace `cfg.baselines.*`（V4Config 顶层新增 sub-config，与 `cfg.env/model/train/mup/eval/legacy` 平级；V4Config 无 `.v4` 中间层）；纯消费态,替代 pkg-06 `cfg.model.baseline_*` 提议。

**外部 baseline 的 per-impl 调参常量**（如 `external_mappo_share_policy` / `external_qmix_mixer_hidden_dim` / `external_ma_muzero_gh_simulations` / `external_smoke_max_env_steps`）**不**入 cfg.baselines —— 它们属于实施期 spec 05/06 内部 defaults,不是 Pkg-01 spec05 同步消费的"5 cfg 字段"。这是诚实的最小化:cfg.baselines 字段是「跨 spec 共享的契约」,不是「所有可调参数的字典」。

### 2.5 下游补丁声明（spec 08 显式列出,纯实施期补丁,不改上游 SDD）

| 文件 | 补丁 | 由谁消费 |
|------|------|---------|
| `hyper_mve/envs/resource_commons/observations.py` | +3 行 `if not cfg.env.c_visible: obs[:, c_channel] = 0` | Pkg-08 spec 02（c_hidden 模式）|
| `hyper_mve/planning/mve_planner.py` | +1 行 `if cfg.train.mve_joint_enumerate: candidates = list(itertools.product(...))` | Pkg-08 spec 06（Abl 4 Joint cell）|
| `hyper_mve/scripts/train_main.py` | +6 个 `--variant external_*` 字串路由（`external_mappo`/`external_qmix`/`external_ma_muzero_gh`/`external_mamba`/`external_marie`/`external_ga`；后 3 个调用工厂时抛 `NotImplementedError`,但 CLI 名保留以便 enumerate）；详见 design.md §3.3 CLI ↔ factory-arg ↔ model-class 三列映射表 | Pkg-07 spec 01 |

---

## 3. Capabilities（本包带来的能力）

### 3.1 断言可证伪化（继承 pkg-06）

- ✅ 断言 A: `ma_muzero` + `rewardhead_explicit_type` 提供两个失败模式不同的对照
- ✅ 断言 B′: `input_wide` + `input_deep` 等参对照
- ✅ 断言 C: `no_belief` 对照
- 断言 D: 由 Pkg-08 cfg flag 承载（`use_crn` / `randomize_order` / `mve_joint_enumerate`）

### 3.2 外部范式横向坐标（**新增**）

- ✅ vs on-policy MARL: MAPPO
- ✅ vs 值分解 MARL: QMIX
- ✅ vs model-based MARL: MA-MuZero-GH（最直接近邻）
- ⚖️ vs model-based + belief: MAMBA（若 sourcing 成功）

### 3.3 对下游的解锁

| 下游 | 解锁 |
|------|------|
| **Pkg-08 Eval + Ablation** | `REGISTRY` 11 keys（5 internal + 3 Tier-1 + MAMBA + 2 stubs;stubs 在 sweep 中产生 "skipped: NotImplementedError" 行）+ N-parametric 适配器 + `evaluate() -> EvalReport` 统一签名 → sweep harness 直接 enumerate;internal/external 共享 unified evaluator |
| **Pkg-08 Ablation cells** | Abl 6.x/7 复用 `rewardhead_explicit_type` / `no_belief` |
| **Pkg-08 Methods Comparison Table** | 横向 **9 列**（hyper + 5 internal + 3 Tier-1 external）× 主表指标；MAMBA-if-sourced 加为第 10 列；MARIE/GA stubs 不入主表 |

### 3.4 论文实验配置覆盖

| 表 | 列 | 行 |
|----|-----|-----|
| **Ch6 Methods Comparison (主表)** | (params, walltime, return-Easy, return-Medium, regret) | hyper + 5 internal + 3 Tier-1 external (+ MAMBA-if-sourced) |
| **Ch6 Fairness Disclosure (附表)** | (params, hidden_dim, LR-sweep range, best LR, seeds) | 同上 |
| **Ch6 Internal Ablation** | 同 pkg-06 等参 + LR sweep 协议 | 5 internal |
| **Ch6 External Smoke** | (init return, 20K-step return, convergence direction OK?) | 3 Tier-1 external |

---

## 4. Impact（影响范围）

### 4.1 代码影响（待实施估计,本包不写代码）

| 范畴 | 行数估计 | 备注 |
|------|---------|------|
| 工厂 + REGISTRY + protocol | +150 | `__init__.py` + `_runner_protocol.py` |
| shared_backbones（继承 pkg-06） | +120 | 与 pkg-06 估计一致 |
| 5 internal 模型类 | +900 | 继承 pkg-06 |
| 3 Tier-1 external + Tier-2 + stub | +2500 | MAPPO ~600（ported）, QMIX ~800（vendored）, MA-MuZero-GH ~1000（vendored + MA wrapper）, MAMBA ~100 stub-or-active |
| PettingZoo 适配器 | +200 | wrapper + oracle filter |
| 内部测试（C7-INT-*）| +700 | 继承 pkg-06 |
| 外部测试（C7-EXT-*）| +500 | adapter no-leak + 3 smoke + eval contract + registry + stub |
| **合计** | **~+5000+** | 远超 pkg-06 的 1800,因新增 external 范式 |

### 4.2 性能影响

| variant | 单步 forward / smoke step | 备注 |
|---------|--------------------------|------|
| 5 internal | 继承 pkg-06 | spec 07 分档预算 |
| MAPPO | 小（无 MCTS）| ~0.1× hyper |
| QMIX | 小（DQN 风格）| ~0.1× hyper |
| MA-MuZero-GH | ~等同 hyper（同 MuZero 系）| MCTS 主导 |
| MAMBA | 中（model-based + belief）| 若 sourced |

External LR sweep + smoke 总 GPU 预算 ≈ 250 GPU-hr（详见 [Plan File]）。

### 4.3 与上游契约对账

| 上游契约 | 本包对账点 |
|---------|-----------|
| Pkg-05 spec08 §3.1 `create_baseline_model(cfg, variant)` | spec 01 supersede（rename → `create_baseline`）+ alias 保留 |
| Pkg-05 spec08 §3.2 五条复用约束（仅 internal）| spec 03 + 单测 |
| Pkg-04 spec02 7-API（仅 internal）| spec 03 |
| Pkg-04 spec08 §5.1 草案 | spec 01 supersede（继承 pkg-06 supersede 链）|
| Pkg-03 spec08 §6 BeliefNet 共享 | spec 02（继承 pkg-06）|
| Pkg-02 ResourceCommonsEnv | spec 04 适配器封装 + spec 08 obs-mask 补丁声明 |

---

## 5. R7-1 ~ R7-12 风险列表

| # | 风险 | 缓解 | 检测 |
|---|------|------|------|
| **R7-1** | input_wide/deep 与 hyper 参数对不齐（断言 B' 失效）| spec 07 双阈值 warn ≤5%/fail ≤10% | `test_baseline_param_count_within_5pct`（继承 pkg-06）|
| **R7-2** | shared backbone 跨 variant 参数量不一致 | spec 02 强制同 cfg 同类构造 | `test_shared_backbone_identical_param_count`（继承 pkg-06）|
| **R7-3** | external baseline 不收敛 / 在 ResourceCommons 上"平地起飞"（Fehr-Schmidt 非常态)| Easy preset smoke 收敛性检查（C7-EXT-SMOKE1）；若 flat-line 降级为 "appendix best-effort" | spec 05/06 smoke 单测 |
| **R7-4** | PettingZoo 适配器泄漏 oracle 或 eval-only 信息（c_true / types / hotspot_centers / resource_state）→ external 偷看 | spec 04 两 flag `oracle_mode=False` AND `eval_info_mode=False` 默认 + 适配器读 env schema-marker tuples（DRY 抗 env 演化）+ 强制单测 | `test_adapter_info_gating.py`（C7-EXT-ADPT1; 4-字段 × 2-flag 正交 × {reset,step} × {Easy,Medium}） |
| **R7-5** | external runner 自带 trainer 引入新依赖（torch RL 库,如 RLlib） | spec 05/06 vendoring 政策 ：尽量纯 PyTorch + minimal port,不引 RLlib;若必要在 spec 头部明示 | spec 头部 dependency 表 |
| **R7-6** | MA-MuZero-GH vendoring 失败（fork 不可移植 / 单 agent 仅）| spec 06 budget 3 天,失败 fallback "每 agent vanilla MuZero + 平均 reward"（弱版本）| spec 06 三日检查点 |
| **R7-7** | MAMBA 找不到可用 source | spec 06 sourcing 协议 2 日 budget + 失败转 stub 保留搜索日志（不阻塞 SDD finalize）| spec 06 SDD 内置搜索日志 |
| **R7-8** | external LR sweep 太贵超 GPU 预算 | spec 07 限制 ≥3 LR × ≥3 seeds（不强求 5）;Easy 先 sweep 缩到 1 LR 再 Medium 跑 5 seed | spec 07 GPU 预算表 |
| **R7-9** | `randomize_order` 重命名碰撞历史代码读取 `use_coord_desc` | alias-with-deprecation;实施前 grep | Pkg-08 spec 06 实施期检查 |
| **R7-10** | pkg-06 supersede 后旧 import path 仍被消费 → 静默走旧契约 | spec 01 alias + `DeprecationWarning`;Day 8 grep 校验 | Day 8 `check_ref_matrix.ps1` |
| **R7-11** | external runner 公平地不公平（隐式假设同质 agent 共享 policy）| spec 05/06 显式声明 share-policy=True/False;adapter 暴露 per-agent obs 以允许 separate-policy | spec 04 obs dict 约定 |
| **R7-12** | Tier-1 任一 baseline smoke 失败 → 降级 appendix → 论文主表少一列 | smoke 早跑（Phase B/C 第 1 周）;若失败立即换 alternative impl 或弱化为 "best-effort" 披露 | G6 (Code week 4) hard gate |

---

## 6. Non-Goals（明确不做）

- ❌ **不实现** baseline 实际代码（本包仅 SDD 文档;实施在 pkg-07 实施期 4-5 周）
- ❌ **不修改** Pkg-01/02/03/04/05 任何 SDD（仅消费;3 处实施期代码微补丁由 spec 08 声明,不动 SDD）
- ❌ **不提供** 断言 D（planner 双技术）的对照 model —— 由 Pkg-08 cfg flag 承载
- ❌ **不引入** RLlib / Stable-Baselines3 等大型 RL 框架依赖（external baseline 走 minimal port 政策,保持依赖面小）
- ❌ **不强求** external baseline 与 hyper 等参（披露式 fairness 替代）
- ❌ **不重写** ResourceCommonsEnv（只加适配器封装）
- ❌ **不修改** 论文 Ch4/Ch5 文档（pkg-07 实施完成 + Pkg-08 主表跑完后再统一改 Ch6.x）
