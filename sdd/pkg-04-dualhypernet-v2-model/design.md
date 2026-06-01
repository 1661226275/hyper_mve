# Pkg-04: DualHyperNetwork v2 & HyperMuZero Model v2 — Design

> 配套阅读：[`proposal.md`](./proposal.md)（**先读 proposal 再读本文**）

---

## 1. Context

### 1.1 项目阶段

本包是 v4 重构的**第四个 SDD 包**，承接 Pkg-01 (Schema) / Pkg-02 (Env) / Pkg-03 (TriContextEncoder + BeliefNet)。完成本包后，Pkg-05 (Trainer & Worker) 可立即启动——trainer 直接消费 model 7 个对外 API（spec 08 §1 表）。

```
Pkg-01 (Schema)              ✅ 完成（含 normalize patch）
Pkg-02 (Env)                 ✅ 完成（env.info Oracle 信号已暴露）
Pkg-03 (TriContextEncoder + BeliefNet)  ✅ 完成（API 稳定性已锁定）
    ↓
Pkg-04 (本包)                ⏳ 当前
    ↓
Pkg-05 (Trainer)             ← 立即依赖 model 7 API
Pkg-06a/b (Baselines)        ← 共享 RepNet + BeliefNet（shared_backbones）
Pkg-07 (Eval)                ← Self-Info eval 调用 set_context_subjective
```

### 1.2 前置依赖（本包消费什么）

| 来源 | 内容 |
|------|------|
| **Pkg-01** `schemas/_constants.py` | D_CTX_AUG=80, D_C_CTX=16, D_ROLE=32, D_BELIEF=32 |
| Pkg-01 `schemas/agent_type.py` | AgentType Enum |
| Pkg-01 `schemas/capability.py` | CapabilityVector + `normalize()` 方法（C 修订已 patched） |
| Pkg-01 `configs/model_config.py` | ModelConfig（17 项字段，spec 02 §1 穷举） |
| Pkg-01 `configs/train_config.py` | TrainConfig.belief_grad_gating_steps=5000, detach_pred_context |
| **Pkg-02** `envs/resource_commons/env.py` | env.observation_space, env.info（三段分组：Public/Oracle/EvalOnly） |
| **Pkg-03** `models/tri_context_encoder.py` | TriContextEncoder.forward, forward_c_ctx_only |
| Pkg-03 `models/belief_net.py` | BeliefNet.step / forward / get_head_opp_predictions / init_hidden |
| Pkg-03 `models/belief_encoder.py` | BeliefEncoder（在 .detach() 之后接收，由 main loss 训练） |
| Pkg-03 `models/belief_losses.py` | belief_loss, l_c, l_opp, l_div, build_oracle_z_seq |
| **Ch4 v4** | §4.3 数据流, §4.4 六网络表, §4.6 防线, §4.6.5 梯度门控 |

### 1.3 本包提供（后续包消费什么）

| 输出 | 消费者 | 用途 |
|------|--------|------|
| `HyperMuZeroModel` 类 | Pkg-05, Pkg-06a/b (via shared_backbones), Pkg-07, Pkg-08 | 全部 model forward / 训练 / 评估 |
| `model.update_step(step)` | Pkg-05 trainer 每 train_step 起点 | belief grad gating 阈值判断 |
| `model.set_context_objective(c_t)` | Pkg-05 trainer + worker + mve_planner | 计算 θ_state（所有 agent 共享） |
| `model.set_context_subjective(agent_id, cap_i, belief)` | 同上 | 计算 θ_rew^i / θ_pred^i（per-agent） |
| `model.encode / transition / predict_reward / predict` | 同上 | 标准 MuZero 4 个 forward 接口 |
| `DualHyperNetwork` 类（可选导入） | Pkg-06a Input-Wide/Deep baseline（等参对照） | 等参 baseline 参考实现 |

---

## 2. Goals

### 2.1 主目标（必须完成）

1. **G1: DualHyperNetwork v2 三路输入正确**——forward_trans 仅 c_ctx（16 维）+ forward_subjective 接 ctx_aug（80 维），与 Ch4.3.2-4.3.3 一致
2. **G2: HyperMuZeroModel 两步分离 API**——set_context_objective + set_context_subjective + update_step（用户决议 Q3 + Q4）
3. **G3: 5 道稳定性防线全部保留**——AdaLN (1+γ) + L2 norm + output_scale + Δs 残差 + belief grad gating
4. **G4: type-aware reward 路径实现**——type_emb → role → ctx_aug → hyper_rew → θ_rew^i，单测验证 cos-sim < 0.95
5. **G5: 性能预算达标**——单步 forward < 15 ms (V100, B=256, N=4)，参数量 ∈ [2.8M, 3.6M]
6. **G6: mve_planner.py 4 处调用点迁移完成**——Pkg-04 PR 范围内同步改造（用户决议 Q2）

### 2.2 衍生目标（应尽量达到）

- **G7**: `mypy --strict hyper_mve/models/{hyper_network,hyper_muzero_model,functional_nets,grad_gating}.py` 零错误
- **G8**: 单测覆盖率 ≥ 85%
- **G9**: 与 v4.7 数值回归对比（AdaLN / L2 norm / Δs）误差 ≤ 1e-6
- **G10**: spec 08 §1 API 稳定性表 7 个接口全部 finalized

### 2.3 Non-Goals（明确不解决）

- **NG1**: **不实现** Pkg-05 trainer 主循环
- **NG2**: **不实现** 课程学习 Stage 切换逻辑（仅在 set_context_subjective 接受 belief tuple）
- **NG3**: **不实现** μP base_shape（属 Pkg-07）
- **NG4**: **不实现** baseline 适配（属 Pkg-06a/b）
- **NG5**: **不实现** RewardHead 内 type 分支（按 D2，type-aware 通过 type_emb 路径隐含；spec 05 §6 仅预留扩展点）
- **NG6**: **不引入**新依赖（仅 PyTorch + 已发布 Pkg-01/02/03 接口）
- **NG7**: **不修改** Pkg-01/02/03 任何 spec 或代码（仅消费）
- **NG8**: **不修改**论文 Ch4 文档

---

## 3. Decisions

> 8 项关键设计抉择。每项格式：**Decision** → 候选 → 推荐 → 理由 → 风险与回滚。

### D1: hyper_trans 输入维度

**Decision**：hyper_trans 接收什么输入？

**候选**：
- **A1**：仅 c_ctx（16 维，与 Ch4.3.2 一致）
- **A2**：c_ctx + global_role（所有 agent 共享 role）
- **A3**：完整 ctx_aug（80 维）

**推荐**：**A1**（仅 c_ctx）

**理由**：
- Ch4.3.2 明确："hyper_trans 不接收 role_i 或 belief_i ——这保证了所有 agent 共享同一个 StateTransNet 权重，对应资源场动力学的'共同物理'（约束 C1）"
- 物理转移上下文不变性是 v4 的核心 motivation（断言 D 的物理基础）
- A2 引入"global_role"概念是 hack，破坏 Self-Info 严格性
- A3 让 θ_state 变 per-agent，违反约束 C1，断言 D 实验无意义

**风险**：误实现为 A3 → spec 01 单测 `test_hyper_trans_only_c_ctx`（改 role 输入时 θ_state 输出不变）严格验证

**回滚**：误改后回 git history

---

### D2: type-aware RewardHead 实现路径

**Decision**：v4 type α/β reward 差异化如何实现？

**候选**：
- **B1**：env 已计算 type-aware reward（按 Ch3.10 公式 3.5-3.10），model RewardHead 直接预测，type 信息通过 type_emb → role → ctx_aug → hyper_rew → θ_rew^i 隐含学习
- **B2**：model RewardHead 内显式 type 分支：`if type == α: ... else: ...`
- **B3**：B1 + B2 双重（冗余对照）

**推荐**：**B1**（仅通过 hyper_rew 路径隐含）

**理由**：
- Ch4.3.3 末段明确："**当 τ_i = α 时，生成的 RewardHead 学到'∂R/∂u_i = 1'的简单结构；当 τ_i = β 时，生成的 RewardHead 学到包含 Fehr-Schmidt 项的复杂结构**"——这就是 hyper_rew 路径的设计意图
- env reward 已按 type 计算（Pkg-02 spec 03），存进 buffer，model 直接监督预测
- B2 在 functional_nets 内加 if 分支破坏 functional API 简洁性（functional_nets 接收 flat_params + forward，无 type 概念）
- B3 冗余增加 ~5K 参数，无明显收益（已可用 B1 单测验证）

**风险**：B1 可能因 hyper_rew 训练不充分而 type 分化不明显 → spec 05 单测 `test_type_aware_reward_differentiation`（cos-sim < 0.95）验证；如不达标，spec 05 §6 预留 RewardHead 显式分支扩展点（默认禁用）

**回滚**：spec 05 §6 启用扩展点

---

### D3: belief gradient gating 实现方式

**Decision**：前 5K step 切断 belief 梯度，如何实现？

**候选**：
- **C1**：显式 step-conditional `.detach()`（在 model.forward 内 if step < 5000 then detach）
- **C2**：PyTorch nn.Hook（register_backward_hook 拦截梯度）
- **C3**：BeliefNet 内部加 detach 参数（移出 Pkg-04）

**推荐**：**C1**（显式 step-conditional `.detach()`）

**理由**：
- 显式 if 分支在 forward 内可调试（pdb 可看到 detach 触发）
- C2 Hook 在反向时触发，调试困难（前向看不到 detach 行为，仅梯度异常时才暴露）
- C3 把 detach 放 BeliefNet 内会切断 BeliefNet 自身的 L_belief loss（破坏独立优化路径）
- 与用户决议 Q4 (update_step 独立方法管理 self._step 状态) 完美配合

**风险**：实现遗忘（5K step 后仍 detach）→ spec 04 单测 `test_post_5k_belief_grad_flow`（step >= 5000 时 BeliefNet 参数梯度 norm > 0）严格验证

**回滚**：spec 04 改为 C2 Hook

---

### D4: set_context 缓存策略

**Decision**：调用 set_context 时是否缓存 θ？

**候选**：
- **D1**：完全无缓存（每次 forward 重新生成 θ）
- **D2**：缓存所有 N 个 agent 的 θ_rew^i / θ_pred^i（episode 起点全算）
- **D3**：缓存当前 agent 的 θ_rew / θ_pred（与 set_context_subjective 同步切换）；θ_state 一次性算（set_context_objective）

**推荐**：**D3**（与用户决议 Q3 配合）

**理由**：
- D1 重复算 θ 浪费 ~30% forward 时间（hyper_rew/pred 是 MLP forward，非廉价）
- D2 缓存 N 个 θ 显存开销 N× （Hard preset N=8 时 ~8MB extra，可接受但无必要）
- D3 与两步分离 API 一致：set_context_objective 一次性算 θ_state，set_context_subjective 切 agent 时换 θ_rew/pred
- mve_planner.py per-agent coordinate descent 模式天然适配 D3（按 agent_id 顺序逐个 set_context_subjective）

**风险**：trainer 误用顺序（先 set_subjective 后 set_objective）→ spec 02 docstring + 单测 `test_set_context_order_assertion`（先 subjective 抛 RuntimeError）

**回滚**：D2 全缓存（仅改缓存策略，API 不变）

---

### D5: hyper_pred 是否 detach context

**Decision**：v4.8 引入 `detach_pred_context` 配置开关（hyper_pred 接收的 ctx_aug 是否 detach）

**候选**：
- **E1**：保留 detach（默认 True，与 v4.8 一致）
- **E2**：取消 detach（让 context_encoder 也被 reward + value loss 共同驱动）
- **E3**：配置开关（默认 True，允许 ablation 验证）

**推荐**：**E3**（配置开关，默认 True）

**理由**：
- v4.8 已验证 detach_pred_context=True 时训练更稳定（context_encoder 仅由 reward/consist loss 驱动，避免 value loss 扭曲 belief 学习）
- E3 允许 Pkg-08 实验做消融对比（detach vs no-detach）
- Pkg-01 TrainConfig 已含 `detach_pred_context: bool = True` 字段

**风险**：默认值争议 → spec 01 + spec 02 文档化默认 True，加单测 `test_hyper_pred_detach_default_true`

**回滚**：默认改 False（仅 cfg 改动）

---

### D6: role_i 自我类型与他人类型

**Decision**：role_i 是否含自我类型 + 是否含他人类型？

**候选**：
- **F1**：role_i 仅含 own type（Self-Info 严格）；他人 type 由 belief 通路推断（ẑ）
- **F2**：role_i 含全部 N 个 agent type（Oracle 模式）
- **F3**：role_i 含 own type + 他人 type ground truth（违反 Self-Info）

**推荐**：**F1**（与 Pkg-03 spec 03 一致）

**理由**：
- Ch3.7 + Ch4.2.2 Self-Info 严格性硬约束
- F2/F3 让评估时 model 能看到 Oracle types，"作弊"破坏 zero-shot 评估
- 与 Pkg-03 spec 03 RoleEncoder 接口一致（types: (B, N) 表示每个 batch 中第 i 个 agent 的 own type）
- 与 Pkg-03 spec 08 契约 1 完全一致

**风险**：trainer 误传 oracle types → spec 02 + spec 08 §6 严格规范 + 单测 `test_set_context_subjective_no_oracle_types_leak`

**回滚**：无（Self-Info 是论文核心原则，不可回滚）

---

### D7: StateTransNet 输入格式

**Decision**：StateTransNet 接收 N agents 联合动作的格式？

**候选**：
- **G1**：joint action onehot (N*A 平铺)（与 v4.7 一致）
- **G2**：per-agent (N, A) 矩阵
- **G3**：dict[agent_id, action_onehot]

**推荐**：**G1**（与 v4.7 一致）

**理由**：
- v4.7 已稳定运行；切换会破坏 functional_nets 现有接口（FunctionalStateTransNet.forward 第 2 参数已是 (B, joint_action_dim) shape）
- G2 矩阵在 hyper_trans 生成的 θ_state 与 StateTransNet 内部 reshape 时增加复杂度
- G3 dict 不利于 batched 矩阵运算

**风险**：v4 obs_dim 配置化时 joint_action_dim 也需配置化 → spec 01 + spec 03 同步处理

**回滚**：G2（需改 functional_nets 接口）

---

### D8: v4.7 → v4 替换策略

**Decision**：v4.7 DualHyperNetwork / HyperMuZeroModel 如何处理？

**候选**：
- **H1**：inplace 修改文件（保留文件名）+ 旧版本在 git history + Pkg-01 已归档至 `_legacy_v4_7/`
- **H2**：新增 V2 类（DualHyperNetworkV2、HyperMuZeroModelV2）并保留 V1
- **H3**：重写并删除 V1（完全清空 git history 中的 V1）

**推荐**：**H1**（inplace 修改）

**理由**：
- 用户已确认 v4.7 归档策略（Pkg-01 已完成 `_legacy_v4_7/` 归档 + `git tag v4.7-final`）
- git history 足够回滚（如发现 v4 重写有严重 bug，可 git revert + 复活 _legacy_v4_7）
- H2 保留 V1 类会造成 code base 污染（mve_planner / trainer 调用方需判断 V1/V2，duck-typing 复杂）
- H3 彻底删除 V1 破坏审计跟踪

**风险**：inplace 修改后 git diff 大 → PR 拆分（hyper_network.py 单独一个 commit，hyper_muzero_model.py 单独一个 commit）

**回滚**：git revert + 复活 `_legacy_v4_7/`

---

## 4. 设计决策对照表

| Decision | 推荐 | 影响范围 | 后续修改成本 |
|----------|------|----------|--------------|
| D1 hyper_trans 输入 | 仅 c_ctx (16) | hyper_network.py forward_trans | 中（API 签名） |
| D2 type-aware 实现 | hyper_rew 路径隐含 | functional_nets.py + spec 05 | 低（默认无 if 分支） |
| D3 grad gating 实现 | 显式 .detach() | grad_gating.py + model.forward | 低（if step < threshold） |
| D4 set_context 缓存 | θ_state 一次性 + θ_rew/pred 切 agent | model.set_context_obj/subj | 中（顺序约束） |
| D5 detach_pred_context | 配置开关默认 True | DualHyperNetwork v2 | 低（cfg 切换） |
| D6 role_i Self-Info | 仅 own type | model.set_context_subjective + RoleEncoder 接口 | 低（与 Pkg-03 一致） |
| D7 StateTransNet 输入 | joint action onehot (N*A) | functional_nets.py | 低（与 v4.7 一致） |
| D8 v4.7 替换策略 | inplace 修改 | hyper_network.py + hyper_muzero_model.py | 中（PR diff 大） |

---

## 5. 实现顺序建议

```
Day 1（半天）:
  - README.md + proposal.md + design.md 三件套
  - 用户审阅 8 Decisions
  
Day 1（半天）+ Day 2:
  - spec 01-dualhypernet-v2-api (HyperNetMLP 保留 + 双 forward API)
  - tests/models/test_hyper_network_v2.py 单测代码 sketch
  - 写 spec 03 (4 道防线行号映射)
  
Day 3:
  - spec 02-hyper-muzero-model-v2 (5 个 API + ModelConfig 字段穷举)
  - spec 04-belief-gradient-gating (grad_gating.py + 单测)
  - **Hard gate**: spec 02 set_context 双 API 签名锁定才能进 spec 05/06

Day 4:
  - spec 05-type-aware-reward (链路图 + 断言 A 物理基础)
  - spec 06-data-flow-diagram (mermaid 图 + 张量 shape 表)
  
Day 5:
  - spec 07-forward-performance-budget (性能预算 + 单测护栏)
  - spec 08-integration-contracts (对外硬契约，最大量内容)
  - **Pkg-05 启动条件**: spec 08 §3-§4 完整
  
Day 6:
  - 全 spec 交叉引用核对（README §6 spec 间引用表 vs 实际引用）
  - 11 项硬约束单测命名表对账
  
Day 7:
  - PR description + 用户最终 ack
```

---

## 6. 跨包接口约定（API contract，spec 08 §1 完整版）

### 6.1 import 路径（稳定）

```python
# Pkg-05/06/07/08 应使用这些 import 路径，本包承诺不变更
from hyper_mve.models import HyperMuZeroModel, DualHyperNetwork
from hyper_mve.models.grad_gating import BeliefGradGating

# 标准用法
from hyper_mve.configs import V4Config
cfg = V4Config.from_preset("medium")
model = HyperMuZeroModel(cfg)
```

### 6.2 model 7 个对外 API（稳定）

```python
class HyperMuZeroModel(nn.Module):
    def update_step(self, global_step: int) -> None:
        """更新内部 step 计数器，触发 belief grad gating 阈值判断.
        必须在每个 train_step 起点（K-step unroll 之前）调用一次."""
    
    def set_context_objective(self, c_t: torch.Tensor) -> None:
        """计算 θ_state（所有 agent 共享）并缓存.
        Args:
            c_t: (B,) or (B, 1) float32 共享 context scalar
        必须在每次 K-step unroll 起点调用一次（所有 agent 共享 θ_state）."""
    
    def set_context_subjective(
        self,
        agent_id: int,
        cap_i: torch.Tensor,             # (B, 4) RAW CapabilityVector
        belief: tuple[torch.Tensor, torch.Tensor],  # (c_hat (B,), z_hat (B, N-1, 2))
    ) -> None:
        """计算 θ_rew^i / θ_pred^i 并缓存.
        切 agent 时换缓存."""
    
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """obs (B, N, obs_dim) → s (B, latent_dim) via RepNet."""
    
    def transition(self, s: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """(s, joint_action_onehot) → s_next via StateTransNet (θ_state 缓存)."""
    
    def predict_reward(self, s: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """(s, joint_action_onehot) → r via RewardHead (θ_rew^i 缓存)."""
    
    def predict(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """s → (logits, value) via PredictionNet (θ_pred^i 缓存)."""
```

### 6.3 DualHyperNetwork v2 双 forward API（稳定）

```python
class DualHyperNetwork(nn.Module):
    def __init__(
        self,
        c_ctx_dim: int = 16,         # = D_C_CTX
        ctx_aug_dim: int = 80,       # = D_CTX_AUG (16 + 32 + 32)
        trans_param_count: int = ...,
        rew_param_count: int = ...,
        pred_param_count: int = ...,
        hidden_dims: tuple[int, ...] = (256, 256),
        rew_hidden_dims: Optional[tuple[int, ...]] = (256, 256, 256),  # 更深
        norm_output: bool = True,
        trans_output_scale_init: float = 0.01,
        rew_output_scale_init: float = 0.1,    # ← v4.7 关键
        pred_output_scale_init: float = 0.01,
        detach_pred_context: bool = True,      # ← D5
    ): ...
    
    def forward_trans(
        self,
        c_ctx: torch.Tensor,         # (B, 16) 仅客观通路
    ) -> torch.Tensor:               # θ_state: (B, trans_param_count)
        """C1: 仅接 c_ctx，改 role/belief 时输出不变."""
    
    def forward_subjective(
        self,
        ctx_aug: torch.Tensor,       # (B, 80) c_ctx + role + belief
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """C2: 接完整 80 维 ctx_aug.
        Returns:
            θ_rew: (B, rew_param_count)
            θ_pred: (B, pred_param_count)
        若 detach_pred_context=True: hyper_pred 内部对 ctx_aug 调 .detach()
        """
```

### 6.4 grad_gating.py 接口（稳定；review 修订 3 双 API）

```python
class BeliefGradGating:
    """Helper（不是 nn.Module）：双层 detach 策略 — raw heads + ctx_aug 子段."""
    
    def __init__(self, num_warmup_steps: int = 5000):
        self.num_warmup_steps = num_warmup_steps
    
    def apply_raw(
        self,
        c_hat: torch.Tensor,
        z_hat: torch.Tensor,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """切断 1: raw heads → 阻 main loss 反向到 BeliefNet (GRU + heads)."""
        if step < self.num_warmup_steps:
            return c_hat.detach(), z_hat.detach()
        return c_hat, z_hat
    
    def apply_ctx(
        self,
        ctx_aug: torch.Tensor,
        step: int,
        belief_slice: tuple[int, int] = (48, 80),
    ) -> torch.Tensor:
        """切断 2 (review 修订 3): ctx_aug.belief 子段 → 阻 main loss 反向到 BeliefEncoder.
        
        ctx_aug = [c_ctx (0:16), role (16:48), belief_vec (48:80)]
        仅 [48:80] 子段 detach; c_ctx + role 路径梯度保留.
        """
        if step >= self.num_warmup_steps:
            return ctx_aug
        s, e = belief_slice
        return torch.cat([ctx_aug[..., :s], ctx_aug[..., s:e].detach(), ctx_aug[..., e:]], dim=-1)
    
    apply = apply_raw  # 兼容旧调用名
```

**修订理由**（D3 → review 修订 3）：
- 原设计：BeliefEncoder 仍由 main loss 训练（投影 MLP 早收敛）
- 修订：BeliefEncoder 与 BeliefNet 同步 gate，让两者都等 BeliefNet 收敛（5K step）后再开放
- L_belief 路径（trainer 直接 belief_net.backward()）**始终不受影响**——这是 BeliefNet 独立收敛的保障

### 6.5 ModelConfig 字段依赖（17 项，spec 02 §1 穷举）

Pkg-04 model 完整依赖以下 cfg 字段（变更需 Pkg-01 spec 05 同步）：

**cfg.model.* (13 项)**:
- d_c, d_role, d_belief, d_ctx_aug
- d_id_emb, d_type_emb, d_cap_emb, d_belief_proj
- latent_dim, hidden_dim
- hyper_hidden_dims, hyper_rew_hidden_dims
- trans_output_scale_init, rew_output_scale_init, pred_output_scale_init
- use_adaln, adaln_residual_one_plus
- state_trans_residual
- belief_gru_hidden, belief_pool
- proj_dim

**cfg.train.* (3 项)**:
- belief_grad_gating_steps (=5000, 用于 update_step 阈值)
- detach_pred_context (=True, 用于 hyper_pred D5)

**cfg.env.* (1 项)**:
- observation_space.shape（间接通过 ObservationLayout.total_dim 派生）

### 6.6 v4.7 → v4 调用点迁移指引（spec 08 §3 完整版）

mve_planner.py L71 现状：
```python
model.set_context(rule_exp, id_i)
logits_i, _ = model.predict(curr_s)
```

迁移到 v4：
```python
# planner 入口处先调一次 set_context_objective（c_t 由 rule_exp 派生）
model.set_context_objective(c_t=rule_exp)

# 循环内只换 subjective
model.set_context_subjective(agent_id=id_i, cap_i=cap[id_i], belief=belief[id_i])
logits_i, _ = model.predict(curr_s)
```

同理 L213/219/258 三处迁移；trainer / worker 内 grep 找所有 set_context 调用同步改造。

---

## 7. 验证策略概览

> 详细 acceptance criteria 见 `specs/*.md`。本节列 11 项硬约束 → 具名单测映射（M3：风险预映射）。

| # | 硬约束 | 具名单测 | spec 归属 |
|---|--------|----------|-----------|
| C1 | hyper_trans 仅 c_ctx | `test_hyper_trans_only_c_ctx` | spec 01 |
| C2 | hyper_rew/pred 接 80 维 | `test_subjective_input_dim_80` | spec 01 |
| C4 | belief grad gating 5K | `test_pre_5k_belief_detached` + `test_post_5k_belief_grad_flow` | spec 04 |
| C5 | d_ctx_aug == 80 | `test_ctx_aug_dim_eq_80` | spec 01 |
| C6 | Δs 残差 | `test_state_trans_delta_s_residual` | spec 03 |
| C7 | AdaLN (1+γ) | `test_adaln_one_plus_gamma_factor` | spec 03 |
| C8 | output_scale 初值 | `test_output_scale_inits_trans_rew_pred` | spec 03 |
| C9 | type-aware 路径 | `test_type_aware_reward_differentiation` | spec 05 |
| C11 | Self-Info 严格 | `test_set_context_subjective_no_oracle_types_leak` | spec 08 |
| R8 | forward 性能 | `test_forward_smoke_under_15ms` | spec 07 |
| R9 | Pkg-03 接口兼容 | `test_belief_net_integration` | spec 08 |

---

## 8. Open Questions（含 2026-05-28 用户审阅决议）

| # | Question | 决议 | 影响 |
|---|----------|------|------|
| **Q1** | spec 数量 7 vs 8（加 integration-contracts） | ✅ **8 个**（与 Pkg-03 对称） | README + spec 08 新增 |
| **Q2** | mve_planner.py 改造归属 Pkg-04 vs Pkg-05 | ✅ **Pkg-04 范围** | spec 08 §3 + PR 包含 3 处调用点迁移 |
| **Q3** | set_context API 拆分（合并 vs 两步分离） | ✅ **两步分离**：set_context_objective + set_context_subjective | spec 02 + spec 08 §1 |
| **Q4** | global_step 传递机制 | ✅ **`model.update_step(step)` 独立方法** | spec 02 + spec 04 + spec 08 §1 |
| Q5 | hyper_trans θ_state 在 N agents 间复用 | ✅ **是**（与 D4 + Q3 配合） | spec 02 缓存策略 |
| Q6 | BeliefEncoder 归属（与 Pkg-03 协同复核） | ✅ **Pkg-03 范围**（BeliefEncoder 在 belief_encoder.py） | 与 Pkg-03 spec 01 P5 一致 |
| Q7 | cap_i 是否在 model 内部缓存 | ✅ **否**（每次 set_context_subjective 传入 raw cap_i） | spec 02 |
| Q8 | detach_pred_context 配置开关默认值 | ✅ **True**（D5） | spec 01 init 参数 |
| Q9 | D1-D8 锁定 | ✅ **完全锁定**（design.md §3） | – |
| Q10 | 性能预算硬阈值 | ✅ **15ms 硬 fail**（spec 07 单测 assert） | spec 07 |
| Q11 | 单测层级 | ✅ **4 tests + 1 forward smoke + 1 v4.7→v4 migration**（spec 08 §3 行号验证） | tests/models/* |
| Q12 | λ_b 课程系数归属 + model 训练入口 | ✅ **Pkg-05** trainer 直接调 7 个对外 API（spec 08 §1）拼装 loss，**model 不暴露 `forward()` 或 `compute_losses()` 作为对外稳定 API**（澄清 1）；λ_b 课程系数由 trainer 在加权 loss 时控制 | spec 02 §6 + spec 08 §1 注释 |

> 全部 Q1-Q12 + D1-D8 已 ack。本 design.md 视为 **finalized**。

---

## 9. References

- `proposal.md`（本包）
- Pkg-01 design.md + spec 05 (ModelConfig 17 项字段)
- Pkg-02 spec 08（env.info 三段分组）
- Pkg-03 design.md + spec 01/04/05/06/08（TriContextEncoder / BeliefNet / belief_loss / 集成契约）
- `D:\RL\hyper_mve\docs\Chapter4_Architecture_v4.md` §4.3 + §4.4 + §4.6
- 项目 Plan File: Part 2 Pkg-04 + Plan 修订 (Q1-Q4 决议)
- v4.7 行号映射：
  - `models/hyper_network.py` L63-82 (L2 norm + output_scale)
  - `models/functional_nets.py` L43-77 (AdaLN) + L245-271 (Δs 残差)
  - `planning/mve_planner.py` L71/213/219/258 (4 处 set_context 调用点)
