# Pkg-06 Proposal: Baselines — 断言对照基线模型族

> 配套阅读：[`README.md`](./README.md) · [`design.md`](./design.md)（**先读 proposal 再读 design**）

---

## 1. Why（为什么需要本包）

### 1.1 论文 4 条断言需要对照基线才能成立

Hyper-MuZero v4 的核心贡献是 4 条可证伪断言。每条断言都不是"我们的方法好"，而是"**去掉某个特定机制就会出现某个特定失败模式**"。没有对照基线，断言就是空话：

| 断言 | 主张 | 失败模式（去掉机制后）| 需要的对照基线 |
|------|------|---------------------|----------------|
| **A 类型梯度撕裂** | hyper_rew 按 type 生成 θ_rew^i，消除梯度撕裂 | 共享 RewardHead 学到 α/β 的"平均梯度" | `ma_muzero`（共享头）+ `rewardhead_explicit_type`（显式 type 分支）|
| **B belief 专用容量** | per-context θ（hypernet）给每个 belief 组合专用容量 | input-conditioning 在等参下容量被摊薄 | `input_wide`（加宽）+ `input_deep`（加深）|
| **C 三路必要性** | c_ctx / role_i / belief_i 三路缺一不可 | 移除 belief 路 → 性能掉 | `no_belief`（移除 belief 路）|
| **D planner 双技术** | MVE+CRN+coord-descent 联合解 SNR 崩塌 | 单技术 → planner 退化为均匀 | （Pkg-08 planner ablation，**非 baseline model**）|

**结论**：断言 A/B/C 的证伪压力**直接**落在 Pkg-06 必须提供的 5 个对照模型上。断言 D 由 planner cfg flag 承载（`use_crn` / `use_coord_desc`），不需要独立 model，归 Pkg-08。

### 1.2 等参公平性是断言 B 的生死线

断言 B 说"hypernet 的 per-context θ 比 input-conditioning 强"。这个比较**只有在参数量对齐时才有意义**——否则审稿人会说"你只是参数更多"。Ch5 §5.6 协议 step1 明确：

> baseline 与 Hyper-MuZero 对比时，**只统计条件化消费子系统的参数**（hyper_{trans,rew,pred} + 3 功能网；input baseline 的加宽/加深功能网），**排除**共享的 RepNet / BeliefNet / TriContextEncoder。

如果 `input_wide` / `input_deep` 不与 `hyper` 等参，断言 B 直接失效。**这是 Pkg-06 最硬的约束**，必须在 spec 层量化验证（双阈值 + 具名单测），不能等实验时才发现参数对不齐。

### 1.3 "复用同一套训练设施"是公平性的另一半

等参之外，公平性还要求 baseline 与主方法**走完全相同的训练/采集/采样/loss 流程**——否则差异可能来自 trainer 而非 model。Pkg-05 spec08 §3.2 已锁定 5 条复用约束：

```
1. 同一 MuZeroTrainer 类（无 baseline-specific train_step）
2. 同一 Worker 类（采集行为一致）
3. 同一 EpisodeReplayBuffer 类
4. 同一 compose_total_loss 函数
5. 仅 model 类不同
```

Pkg-06 的全部"变化"被压缩到**一个工厂函数 `create_baseline_model(cfg, variant)`** + **一个共享后端模块 `shared_backbones.py`**。这是本包能够轻量的根本原因：它不重写训练系统，只提供 model 工厂。

### 1.4 Pkg-04 spec08 §5.1 的过时草案必须被 supersede

Pkg-04 spec08 §5.1（lines 270-285）曾给出 illustrative 草案 `create_input_wide_baseline(cfg, total_params_target)`——**每 variant 一个函数**。但 Pkg-05 spec08 §3.1（line 136）已用**统一工厂** `create_baseline_model(cfg, variant)` 覆盖。若 Pkg-06 不显式声明"后者 supersede 前者"，Day 6 grep 会同时命中两套签名，Pkg-08 调 train_main.py 时也会踩 drift。本包 spec 01 头部强制 supersede 声明。

---

## 2. What Changes（具体改动）

> **本包只写 SDD 文档，不写实现代码**。以下"改动"描述的是 SDD 锁定的**待实施接口**，供 Pkg-06 实施期与 Pkg-08 消费。

### 2.1 新增（1 工厂 + 1 共享后端 + 5 模型类，全部在 `hyper_mve/baselines/`）

#### 2.1.1 `hyper_mve/baselines/__init__.py`（工厂，import 路径已被 Pkg-05 spec08 §3.1 line 136 锁定）

```python
def create_baseline_model(cfg, variant: str):
    """统一 baseline 模型工厂.

    variant ∈ {input_wide, input_deep, ma_muzero, no_belief, rewardhead_explicit_type}.
    'hyper' 不走本工厂（直接 HyperMuZeroModel(cfg)）。
    """
    if variant == "hyper":
        raise ValueError("'hyper' uses HyperMuZeroModel directly, not this factory")
    ...
```

**最终契约声明**（spec 01 头部强制）：本签名是 baseline 工厂的最终契约（依据 Pkg-05 spec08 §3.1）；Pkg-04 spec08 §5.1 的 `create_input_wide_baseline(cfg, total_params_target)` 是 illustrative 草案，**被本 spec supersede**。

#### 2.1.2 `hyper_mve/baselines/shared_backbones.py`（等参公平性核心）

```python
def create_rep_net(env_cfg, model_cfg): ...           # 同类同构（复用 Pkg-04 RepresentationNet）
def create_belief_net(env_cfg, model_cfg): ...        # 独立实例、独立梯度、参数量逐位对齐
def create_tri_context_encoder(env_cfg, model_cfg): ...
```

约束（对齐 Pkg-03 spec08 §6 + Pkg-04 spec08 §5）：5 个 baseline + hyper 各自实例化自己的 RepNet/BeliefNet/TriContextEncoder，**结构 + 参数量完全相同**，训练时**梯度不共享**。

#### 2.1.3 5 个 baseline 模型类

| 工厂 variant | 模型类 | 断言 | 一句话 |
|--------------|--------|------|--------|
| `input_wide` | `InputWideBaselineModel` | B | 无 hypernet，3 功能网**加宽**，concat ctx_aug 进 input |
| `input_deep` | `InputDeepBaselineModel` | B | 无 hypernet，3 功能网**加深** |
| `ma_muzero` | `MAMuZeroBaselineModel` | A | vanilla MARL，**共享单一 RewardHead**，agent_id+type 进 input |
| `no_belief` | `NoBeliefBaselineModel` | C / Abl7 | HyperMuZero 变体，belief 路在 TriContextEncoder 内置零 |
| `rewardhead_explicit_type` | `ExplicitTypeRewardBaselineModel` | A / Abl6.x | 有 hyper_{trans,pred}，**无 hyper_rew**，RewardHead 显式按 type 分支 |

每个模型类**必须实现/桩同样的 7-API**（与 Pkg-04 spec02 逐字一致），否则 MuZeroTrainer / Worker / MVEPlanner 无法统一调用。

### 2.2 7-API 一致性（每个 baseline 必须实现）

```python
update_step(global_step: int) -> None
set_context_objective(c_t: Tensor) -> None
set_context_subjective(agent_id: int, cap_i: Tensor, belief: tuple) -> None
encode(obs: Tensor) -> Tensor
transition(s: Tensor, action: Tensor) -> Tensor
predict_reward(s: Tensor, action: Tensor) -> Tensor
predict(s: Tensor) -> tuple[Tensor, Tensor]
```

**内部差异**（spec 06 详述）：
- input-conditioned baseline 在 `set_context_*` 内**缓存** `(c_t, agent_id, cap_i, belief)`，在 `transition/predict_reward/predict` 内 **concat 进加宽/加深功能网**（不走 hypernet）。
- `ma_muzero` 用共享 RewardHead + agent_id/type one-hot 作 input（无 ctx_aug）。
- `no_belief` 把 belief tuple 在 TriContextEncoder 内置零。
- `update_step` 仍维护 belief grad-gating 计数（**5 variant 行为完全一致**，否则断言 B 不公平 —— 见 §1.2 + spec 06）。

### 2.3 stateful 契约 + Self-Info 严格（每个 baseline 继承）

- **stateful "用最后一次 subjective agent_id"**（逐字 Pkg-04 spec02 L308-310）：`predict_reward/predict` 返回值对应**最后一次** `set_context_subjective` 的 agent_id。input baseline 即便不走 hypernet 也必须缓存并遵守此语义。
- **Self-Info 严格**（C11/D6）：baseline 同样**不得**把 oracle types 泄漏进 `set_context_subjective`：`assert cap_i.shape[-1] == 4`；belief 来自 BeliefNet 推断，**非** `env.info["types"]`。

### 2.4 待声明 cfg 字段（消费态声明，须 Pkg-01 spec05 同步，本包不改上游）

```
cfg.model.baseline_wide_hidden_dim          # input_wide 加宽宽度
cfg.model.baseline_deep_layers              # input_deep 加深层数
cfg.model.baseline_ma_muzero_share_pred_head # ma_muzero 是否共享 pred 头
cfg.model.baseline_explicit_type_branches   # rewardhead_explicit_type 分支数
```

---

## 3. Capabilities（本包带来的能力）

### 3.1 断言可证伪化

- ✅ 断言 A：`ma_muzero` / `rewardhead_explicit_type` 提供两个失败模式不同的对照（共享头平均梯度 vs 显式分支无法吸收 capability 异质性）。
- ✅ 断言 B：`input_wide` / `input_deep` 在**等参**下对照 `hyper`，证伪"只是参数多"。
- ✅ 断言 C：`no_belief` 提供 belief 路移除对照。

### 3.2 对下游的解锁

| 下游 | 解锁 |
|------|------|
| **Pkg-08 Experiments** | `create_baseline_model(cfg, variant)` 可调 → train_main.py `--variant baseline_*` 端到端跑；Ablation 6.x/7 直接复用 `rewardhead_explicit_type` / `no_belief` |
| **Pkg-07 Eval** | baseline 与 hyper 走同一 evaluator（7-API 一致 + 同 trainer）|

### 3.3 论文实验配置覆盖

- 主对比：Medium config（N=4, 2α+2β）5 seeds × {hyper, input_wide, input_deep, ma_muzero} = 20 runs（Ch5 §5.6 step3）。
- 等参表：报告每个 baseline 的条件化子系统参数量 vs hyper（Ch5 §5.6 step1）。
- LR sweep 表：每 baseline 独立调 LR（Ch5 §5.6 step2），证明公平调参。

---

## 4. Impact（影响范围）

### 4.1 代码影响（待实施估计，本包不写代码）

| 范畴 | 行数估计 | 备注 |
|------|---------|------|
| 新增 `baselines/__init__.py`（工厂）| +80 | create_baseline_model + registry + reject hyper |
| 新增 `baselines/shared_backbones.py` | +120 | 3 个 create_* 工厂 + 参数量对齐校验 |
| 新增 5 个 baseline 模型类 | +900 | input_wide/deep + ma_muzero + no_belief + explicit_type，各 ~180 |
| 新增 `tests/baselines/test_*.py` | +700 | C6-* 具名单测（含每 variant 跑的 self-info / stateful / grad-gating）|
| **合计** | **~+1800** | 纯新增，不改上游 |

### 4.2 性能影响

| variant | 单步 forward 相对 hyper | 备注 |
|---------|------------------------|------|
| input_wide | ≈ 等参 → 时间相近 | 加宽功能网，无 hypernet 调用 |
| input_deep | ≈ 等参 → 时间相近 | 加深功能网 |
| ma_muzero | **偏小**（无 hypernet + 无 ctx_aug）| 结构性更快，spec 07 豁免等参 |
| no_belief | 略快 | belief 路置零省 BeliefNet→ctx 计算 |
| rewardhead_explicit_type | 略小 | 少一个 hyper_rew |

spec 07 给单步 forward 分档预算 + 单测护栏，避免 baseline 实现意外超预算。

### 4.3 与上游契约对账

| 上游契约 | 本包对账点 |
|---------|-----------|
| Pkg-05 spec08 §3.1 `create_baseline_model(cfg, variant)` | spec 01 工厂签名逐字一致 |
| Pkg-05 spec08 §3.2 五条复用约束 | spec 08 §3 + 单测 `test_baseline_reuses_same_trainer_class` |
| Pkg-04 spec02 7-API | spec 06 逐字一致 + 单测 `test_baseline_implements_full_7api` |
| Pkg-04 spec08 §5.1 草案 | spec 01 supersede 声明 |
| Pkg-03 spec08 §6 BeliefNet 共享 | spec 02 共享契约 + 单测 `test_shared_backbone_identical_param_count` |

---

## 5. R6-1 ~ R6-10 风险列表

| # | 风险 | 缓解 | 检测 |
|---|------|------|------|
| R6-1 | input_wide/deep 与 hyper 参数对不齐（断言 B 失效）| spec 07 双阈值 warn ≤5%/fail ≤10% + 调宽度/层数迭代逼近 | `test_baseline_param_count_within_5pct` |
| R6-2 | shared backbone 跨 variant 参数量不一致（公平性核心崩）| spec 02 强制同 cfg 同类构造 + 逐位校验 | `test_shared_backbone_identical_param_count` |
| R6-3 | baseline `update_step` 不调 BeliefGradGating → pre-5K belief 梯度路径不公平 | spec 06 强制 5 variant 共用同一 gating，画 mermaid 双路径图 | `test_baseline_update_step_gates_belief_grad`（每 variant）|
| R6-4 | baseline 泄漏 oracle types 进 set_context_subjective | spec 06 `assert cap_i.shape[-1]==4` + belief 来自 BeliefNet | `test_baseline_set_context_subjective_no_oracle_types_leak`（每 variant）|
| R6-5 | input baseline 不缓存 agent_id → stateful 契约违反 | spec 06 docstring 化 + 缓存 (agent_id,cap,belief) | `test_baseline_predict_uses_last_subjective_agent_id`（每 variant）|
| R6-6 | 工厂收到 "hyper" 静默构造错误模型 | spec 01 显式 `raise ValueError` | `test_factory_rejects_hyper_variant` |
| R6-7 | Pkg-04 §5.1 旧签名与新工厂双签名共存（grep 双命中）| spec 01 supersede 声明 + Day 6 grep 验证单签名 | Day 6 grep 校验 |
| R6-8 | ma_muzero / rewardhead_explicit_type 边界模糊（同打断言 A 撞车）| spec 04/05 撕分失败模式表 | spec 04/05 设计审阅 |
| R6-9 | CLI `baseline_` 前缀 vs 工厂无前缀 drift | spec 01 CLI↔工厂↔模型类映射表 | spec 01 设计审阅 + Pkg-08 对账 |
| R6-10 | baseline 实现引入 baseline-specific train_step（破坏复用约束）| spec 08 §3 五条复用约束 + 单测 | `test_baseline_reuses_same_trainer_class` |

---

## 6. Non-Goals（明确不做）

- ❌ **不实现** baseline 实际代码（本包仅 SDD 文档；实施在 Pkg-06 实施期）。
- ❌ **不修改** Pkg-01/02/03/04/05 任何 SDD 或代码（仅消费上游已发布接口）。
- ❌ **不提供** 断言 D（planner 双技术）的对照 model —— 由 Pkg-08 cfg flag（`use_crn` / `use_coord_desc`）承载。
- ❌ **不纳入** MAPPO / QMix / Mamba 等异范式 baseline —— 它们无法复用 7-API / MuZeroTrainer，会破坏"仅 model 类不同"的公平契约（属另一条对比线，不在本包）。
- ❌ **不重写** trainer / worker / buffer / loss —— Pkg-06 全部变化压缩到 model 工厂 + shared_backbones。
- ❌ **不引入** 新依赖（仅 PyTorch + numpy + 上游已发布接口）；μP 宽度对齐等新机制不采用（超出依赖约束）。
- ❌ **不修改** 论文 Ch4/Ch5 文档。
