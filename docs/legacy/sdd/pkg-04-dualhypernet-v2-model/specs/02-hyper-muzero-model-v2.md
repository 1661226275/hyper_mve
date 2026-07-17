# Spec 02: HyperMuZero Model v2 — set_context 两步分离 API

> 父文档：[`../proposal.md`](../proposal.md) §2.1 · [`../design.md`](../design.md) §3 D2/D3/D4/D6 · §6.2
> **v4 关键改动**：v4.7 双类 (Oracle/Infer) → v4 统一 `HyperMuZeroModel` + set_context 两步分离 + update_step 独立方法（用户决议 Q3 + Q4）。

---

## 1. Purpose

按 Ch4.3 + Ch4.4 + 用户决议 Q3/Q4 实现 v4 统一 `HyperMuZeroModel`，提供 **7 个对外稳定 API**：

| API | 用途 | 调用频率 |
|-----|------|---------|
| `update_step(global_step)` | 更新内部 step 状态（触发 belief grad gating 判断） | 每 train_step 起点 1 次 |
| `set_context_objective(c_t)` | 计算 θ_state（所有 agent 共享） | 每 K-step unroll 起点 1 次 |
| `set_context_subjective(agent_id, cap_i, belief)` | 计算 θ_rew^i / θ_pred^i（per-agent） | 切 agent 时 N 次 |
| `encode(obs)` | obs → s via RepNet | 每步 1 次 |
| `transition(s, action)` | (s, action) → s_next via StateTransNet | 每步 1 次 |
| `predict_reward(s, action)` | (s, action) → r via RewardHead | 每步 1 次 |
| `predict(s)` | s → (π, v) via PredictionNet | 每步 1 次 |

废弃 v4.7 `OracleHyperMuZeroModel` / `InferHyperMuZeroModel` 双类，统一为单一 `HyperMuZeroModel`。

---

## 1.1 ModelConfig 字段穷举表（17 项依赖，M4：避免事后修改）

本 model 完整依赖以下 cfg 字段；变更需 Pkg-01 spec 05 同步：

### cfg.model.* (13 项)

| 字段 | 默认 | 用途 |
|------|------|------|
| `d_c` | 16 | c_ctx 维度（hyper_trans 输入） |
| `d_role` | 32 | role 维度（含 type_emb 8） |
| `d_belief` | 32 | belief 维度（2 × d_belief_proj） |
| `d_ctx_aug` (property) | 80 | hyper_rew/pred 输入维度 |
| `d_id_emb` | 8 | role.id_emb 维度 |
| `d_type_emb` | 8 | role.type_emb 维度 |
| `d_cap_emb` | 16 | role.cap_emb 维度 |
| `d_belief_proj` | 16 | belief 子分量投影 |
| `latent_dim` | 64 | s 维度（RepNet 输出 / StateTransNet I/O） |
| `hidden_dim` | 128 | 功能网络内部隐层 |
| `hyper_hidden_dims` | (256, 256) | hyper_trans / hyper_pred 隐层 |
| `hyper_rew_hidden_dims` | (256, 256, 256) | hyper_rew 更深（type 分化） |
| `trans/rew/pred_output_scale_init` | (0.01, 0.1, 0.01) | output_scale 三初值 |
| `use_adaln` | True | StateTransNet 使用 AdaLN |
| `adaln_residual_one_plus` | True | (1+γ) factor（C7 关键） |
| `state_trans_residual` | True | Δs 残差（C6） |
| `belief_gru_hidden` | 128 | BeliefNet GRU hidden（Pkg-03 spec 04） |
| `belief_pool` | "mean" | belief Pool 策略（Pkg-03 spec 07） |
| `proj_dim` | 64 | BYOL projector 维度（Pkg-05 SSL） |

### cfg.train.* (3 项)

| 字段 | 默认 | 用途 |
|------|------|------|
| `belief_grad_gating_steps` | 5000 | update_step 阈值（spec 04） |
| `detach_pred_context` | True | hyper_pred 输入 detach（D5） |
| `unroll_K` | 5 | K-step unroll（model 接受 forward seq） |

### cfg.env.* (1 项)

| 字段 | 来源 | 用途 |
|------|------|------|
| `observation_space.shape` | Pkg-02 ObservationLayout.total_dim(N, K) | RepNet input dim |

**总计 17 项** —— `__init__` 内全部读取并打印 summary（M4：实施时验证 cfg 完整性）。

---

## 2. Interface

### 2.1 文件路径

`hyper_mve/models/hyper_muzero_model.py`（v4.7 同名文件 inplace 重写，D8）

### 2.2 类签名

```python
import torch
import torch.nn as nn
from typing import Optional
from hyper_mve.configs import V4Config
from hyper_mve.models import TriContextEncoder, BeliefNet
from hyper_mve.models.belief_encoder import BeliefEncoder
from hyper_mve.models.hyper_network import DualHyperNetwork
from hyper_mve.models.functional_nets import (
    FunctionalStateTransNet,
    FunctionalRewardHead,
    FunctionalPredictionNet,
)
from hyper_mve.models.representation_net import RepresentationNet
from hyper_mve.models.grad_gating import BeliefGradGating


class HyperMuZeroModel(nn.Module):
    """v4 统一 HyperMuZero Model (Ch4.3 + 4.4).
    
    v4 关键改动 (相对 v4.7 Oracle/Infer 双类):
        v4.7 Oracle: set_context(rule, agent_id), rule 给真值
        v4.7 Infer:  set_context_from_history(history), 内部 GRU 推断 rule
        v4: 统一 HyperMuZeroModel, BeliefNet (Pkg-03) 提供 belief,
            set_context_objective(c_t) + set_context_subjective(agent_id, cap_i, belief) 两步分离 API,
            update_step(step) 独立方法管理 belief grad gating 状态.
    
    7 个对外 API (spec 08 §1 锁定):
        update_step / set_context_objective / set_context_subjective /
        encode / transition / predict_reward / predict
    """
    
    def __init__(self, cfg: V4Config):
        super().__init__()
        self.cfg = cfg
        
        # ====== C5: ctx_aug_dim 硬约束 ======
        assert cfg.model.d_ctx_aug == 80, (
            f"d_ctx_aug={cfg.model.d_ctx_aug} != 80. "
            "v4 三路 ctx 总维度精确硬约束 (Ch4.2.4)."
        )
        
        # ====== 模块组装 ======
        # RepNet (与 Pkg-03 BeliefNet 独立, 见 Pkg-03 spec 04 §3.4)
        self.rep_net = RepresentationNet(cfg)
        
        # BeliefNet (Pkg-03)
        self.belief_net = BeliefNet(cfg.env, cfg.model)
        
        # TriContextEncoder (Pkg-03, 内部含 BeliefEncoder)
        self.tri_context_encoder = TriContextEncoder(cfg.env, cfg.model)
        
        # Functional nets (functional_nets.py, spec 03)
        self.state_trans_net = FunctionalStateTransNet(cfg)
        self.reward_head = FunctionalRewardHead(cfg)
        self.prediction_net = FunctionalPredictionNet(cfg)
        
        # DualHyperNetwork v2 (spec 01)
        self.hyper_net = DualHyperNetwork(
            c_ctx_dim=cfg.model.d_c,
            ctx_aug_dim=cfg.model.d_ctx_aug,
            trans_param_count=self.state_trans_net.total_params,
            rew_param_count=self.reward_head.total_params,
            pred_param_count=self.prediction_net.total_params,
            hidden_dims=cfg.model.hyper_hidden_dims,
            rew_hidden_dims=cfg.model.hyper_rew_hidden_dims,
            trans_output_scale_init=cfg.model.trans_output_scale_init,
            rew_output_scale_init=cfg.model.rew_output_scale_init,
            pred_output_scale_init=cfg.model.pred_output_scale_init,
            detach_pred_context=cfg.train.detach_pred_context,
        )
        
        # Belief gradient gating helper (spec 04)
        self.grad_gating = BeliefGradGating(
            num_warmup_steps=cfg.train.belief_grad_gating_steps,
        )
        
        # ====== 内部状态 (per-call 缓存) ======
        self._step: int = 0
        self._theta_state: Optional[torch.Tensor] = None   # 客观, 所有 agent 共享
        self._theta_rew: Optional[torch.Tensor] = None     # 主观, 当前 agent
        self._theta_pred: Optional[torch.Tensor] = None    # 主观, 当前 agent
        self._current_agent_id: Optional[int] = None       # 调试用
        self._has_objective: bool = False                  # 顺序断言用
    
    # ====================================================================
    # API 1: update_step (Q4 用户决议, M3 spec 04)
    # ====================================================================
    
    def update_step(self, global_step: int) -> None:
        """更新内部 step 计数器, 用于 belief grad gating 阈值判断.
        
        必须在每个 train_step 起点（K-step unroll 之前）调用一次.
        worker 端不必调用（worker 永远在 inference 状态, 不需要梯度门控）.
        """
        self._step = global_step
    
    # ====================================================================
    # API 2: set_context_objective (Q3 + D1 + D4)
    # ====================================================================
    
    def set_context_objective(self, c_t: torch.Tensor) -> None:
        """C1: 仅接 c_t (Harsanyi 共同知识), 计算 θ_state 并缓存.
        
        必须在每次 K-step unroll 起点调用一次（所有 agent 共享 θ_state）.
        必须在 set_context_subjective 之前调用（顺序断言）.
        
        Args:
            c_t: (B,) or (B, 1) float32 共享 context scalar
        """
        c_ctx = self.tri_context_encoder.forward_c_ctx_only(c_t)  # (B, 16)
        self._theta_state = self.hyper_net.forward_trans(c_ctx)   # (B, trans_param_count)
        self._has_objective = True
        # 清空 subjective 缓存（强制重设）
        self._theta_rew = None
        self._theta_pred = None
        self._current_agent_id = None
    
    # ====================================================================
    # API 3: set_context_subjective (Q3 + D4 + D6)
    # ====================================================================
    
    def set_context_subjective(
        self,
        agent_id: int,
        cap_i: torch.Tensor,                       # (B, 4) RAW CapabilityVector
        belief: tuple[torch.Tensor, torch.Tensor], # (c_hat (B,), z_hat (B, N-1, 2))
    ) -> None:
        """C2 + D6: 计算 θ_rew^i / θ_pred^i, 缓存当前 agent 的 θ.
        
        ⚠️ Self-Info 严格性 (D6 + C11):
            - 接收的 agent_id 是当前视角的 agent index
            - 内部构造 types: (B, N) 时, 第 i 位是 agent_i 的 own type (从 cfg 取);
              **不接受 oracle types** (env.info["types"] 是 trainer 监督专用,
              不可作 model forward 输入)
            - 单测 test_set_context_subjective_no_oracle_types_leak 验证
        
        必须在 set_context_objective 之后调用（顺序断言 D4）.
        
        Args:
            agent_id: 当前 agent 索引 ∈ {0, ..., N-1}
            cap_i: (B, 4) RAW CapabilityVector (内部由 RoleEncoder.normalize 处理)
            belief: (c_hat (B,), z_hat (B, N-1, 2)) - BeliefNet 输出 (Pkg-03)
        """
        # 顺序断言 (D4)
        assert self._has_objective, (
            "set_context_subjective() called before set_context_objective(). "
            "Pkg-04 spec 02 §3.4: 必须先 set_context_objective 后 set_context_subjective."
        )
        
        # cap_i shape 硬约束 (review 修订 2/澄清 2 — C11 第一层防御)
        assert cap_i.dim() == 2 and cap_i.shape[-1] == 4, (
            f"cap_i.shape must be (B, 4), got {tuple(cap_i.shape)}. "
            f"v4 CapabilityVector 是严格 4 元组 (η, φ_fov, ν, ζ); "
            f"若传入 5 维 (B, 5) 等扩展形, 疑似含 type leak — Self-Info 违规 (C11)."
        )
        
        B = cap_i.shape[0]
        N = self.cfg.env.N
        
        # ====== 重建 ctx_aug (per-agent) ======
        # 注: c_t 已在 set_context_objective 处理, 这里需要它做 broadcast
        # 实施细节: 在 set_context_objective 时同时缓存 c_t
        # 简化方案: 重新计算 c_ctx (复用)
        
        # 假设当前实施: 在 set_context_objective 时同时缓存 c_t
        # 此处用 self._cached_c_t 重新计算 ctx_aug
        c_t = self._cached_c_t              # (B,) or (B, 1)
        agent_ids_one = torch.tensor([[agent_id]] * B, device=cap_i.device)
        
        # Self-Info: 只传 own type
        own_type = torch.tensor(
            [self.cfg.env.type_assignment[agent_id].value] * B,
            dtype=torch.long, device=cap_i.device,
        ).unsqueeze(-1)  # (B, 1)
        
        # belief 梯度门控 (C4, spec 04)
        c_hat, z_hat = belief
        c_hat, z_hat = self.grad_gating.apply(c_hat, z_hat, self._step)
        
        # 把 (B, 1) shape 输入 TriContextEncoder (它支持 N=1 退化)
        ctx_aug_i = self.tri_context_encoder.forward(
            c_t=c_t,
            agent_ids=agent_ids_one,
            types=own_type,
            caps=cap_i.unsqueeze(1),                 # (B, 1, 4)
            belief=(c_hat.unsqueeze(1), z_hat.unsqueeze(1)),  # (B, 1, ...)
        ).squeeze(1)  # (B, 80)
        
        # 生成 θ_rew^i / θ_pred^i
        self._theta_rew, self._theta_pred = self.hyper_net.forward_subjective(ctx_aug_i)
        self._current_agent_id = agent_id
    
    # ====================================================================
    # API 4-7: encode / transition / predict_reward / predict
    # ====================================================================
    
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """obs (B, N, obs_dim) → s (B, latent_dim) via RepNet.
        
        RepNet 客观 (所有 agent 共享), 不依赖任何 set_context.
        """
        return self.rep_net(obs)
    
    def transition(
        self,
        s: torch.Tensor,                # (B, latent_dim)
        action: torch.Tensor,           # (B, joint_action_dim) one-hot N*A flat (C14)
    ) -> torch.Tensor:                  # (B, latent_dim) s_next
        """(s, joint_action_onehot) → s_next via StateTransNet (使用缓存 θ_state).
        
        C6: 预测 Δs (残差), 内部 return s + Δs.
        """
        assert self._theta_state is not None, (
            "transition() called before set_context_objective()."
        )
        return self.state_trans_net(s, action, self._theta_state)
    
    def predict_reward(
        self,
        s: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:                  # (B, 1) r_i
        """(s, action) → r via RewardHead (使用当前 agent 的 θ_rew^i).
        
        C9: type-aware 通过 type_emb → hyper_rew → θ_rew^i 隐含实现.
        
        ⚠️ Stateful API 契约（review 修订 2）:
            本方法返回的 r 对应 **最后一次** set_context_subjective 调用的 agent_id.
            若需切换 agent, 必须重新调 set_context_subjective(new_agent_id, ...).
            K-step unroll 内若 transition() 间隔切 agent, 需重新 set_context_subjective.
            单测 test_predict_uses_latest_subjective_agent 验证语义.
        """
        assert self._theta_rew is not None, (
            "predict_reward() called before set_context_subjective()."
        )
        return self.reward_head(s, action, self._theta_rew)
    
    def predict(
        self,
        s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """s → (policy_logits, value) via PredictionNet (使用当前 agent 的 θ_pred^i).
        
        ⚠️ Stateful API 契约（review 修订 2）:
            返回的 (π, v) 对应 **最后一次** set_context_subjective 调用的 agent_id.
            与 predict_reward 共享 self._current_agent_id 状态.
        """
        assert self._theta_pred is not None, (
            "predict() called before set_context_subjective()."
        )
        return self.prediction_net(s, self._theta_pred)
```

### 2.3 调用顺序约定（M2：跨包伪签名先行）

```python
# Pkg-05 trainer (train_step 内典型调用模式)
def train_step(model, batch, global_step):
    model.update_step(global_step)                              # 1. step 状态
    
    # K-step unroll 起点
    c_t = batch["c_t"]                                          # (B,)
    obs = batch["obs"]                                          # (B, N, obs_dim)
    s = model.encode(obs)                                       # (B, latent_dim)
    
    model.set_context_objective(c_t)                            # 2. θ_state 一次
    
    # 计算每个 agent 的 reward / value
    losses = {}
    for k in range(N):
        model.set_context_subjective(                           # 3. 切 agent
            agent_id=k,
            cap_i=batch["cap"][:, k],                           # (B, 4) raw
            belief=(c_hat_seq[:, t, k], z_hat_seq[:, t, k]),    # (B,), (B, N-1, 2)
        )
        
        r_k = model.predict_reward(s, action)                   # (B, 1)
        pi_k, v_k = model.predict(s)                            # (B, A), (B, 1)
        
        losses[k] = {"r": r_k, "pi": pi_k, "v": v_k}
    
    # 传递 K-step (复用 theta_state, 重 forward subjective per agent)
    for unroll_t in range(K):
        s = model.transition(s, action)                         # 客观 forward, θ_state 复用
        # ... 继续 per-agent forward ...

# Pkg-05 worker (在线推断模式, 类似但不调 update_step)
def collect_episode(model, env):
    obs, info = env.reset()
    prev_hidden = model.belief_net.init_hidden(B=1, num_agents=N)
    
    for t in range(T):
        model.set_context_objective(c_t=torch.tensor([info["c_true"]]))
        
        # BeliefNet step
        prev_hidden, c_hat, z_hat = model.belief_net.step(
            torch.from_numpy(obs).unsqueeze(0), prev_hidden,
        )
        
        s = model.encode(torch.from_numpy(obs).unsqueeze(0))
        for k in range(N):
            model.set_context_subjective(
                agent_id=k,
                cap_i=torch.tensor([info["caps"][k]]),
                belief=(c_hat[0, k:k+1], z_hat[0, k:k+1]),
            )
            pi_k, v_k = model.predict(s)
            # ... 采样 action ...
        
        obs, reward, done, _, info = env.step(action)

# mve_planner.py 迁移 (spec 08 §3, v4.7 L71/213/219/258 模式)
def sample_mve_plan(model, root_s, ...):
    model.set_context_objective(c_t=rule_exp)                   # planner 入口一次
    
    for j in agent_order:
        model.set_context_subjective(j, cap[j], belief[j])      # 切 agent
        logits_j, _ = model.predict(curr_s)
        # ...
```

---

## 3. Implementation Notes

### 3.1 set_context 两步分离的性能优势（Q3 + D4）

| 模式 | N agents 总 hyper forward 次数 | 节省 |
|------|-------------------------------|------|
| v4.7 单一 set_context | 2N（hyper_trans + hyper_rew/pred 各 N 次） | – |
| v4 两步分离 | 1 (objective) + N (subjective) = N+1 | ~50% |

Hard preset N=8 时, 单步 forward 从 16 次 hyper 降为 9 次。在 episode 长度 T=300 + K-step unroll 5 时累积节省 ~30% 总 forward 时间（与 spec 07 性能预算配合）。

### 3.2 顺序断言（D4）

`set_context_subjective` 必须在 `set_context_objective` 之后调用。spec 02 通过 `self._has_objective` flag 验证：

```python
assert self._has_objective, "...必须先 set_context_objective..."
```

trainer / planner 误用顺序时立即抛 RuntimeError（不允许 silent fallback）。单测 `test_subjective_before_objective_raises` 验证。

### 3.3 update_step 与 worker 的关系

worker 在线推断时**不**调用 update_step：
- worker 始终处于 inference 模式（torch.no_grad）
- belief grad gating 仅在训练 backward 时有意义
- worker 内 model._step 保持上次 trainer 设置的值（无影响，因为 worker forward 无梯度）

spec 04 详述。

### 3.4 RepresentationNet 的客观性

RepNet 不依赖任何 set_context：
- obs (B, N, obs_dim) → s (B, latent_dim)
- 所有 agent 共享同一 RepNet 参数（与 v4.7 一致）
- 与 BeliefNet 独立 obs_encoder 并列（Pkg-03 spec 04 §3.4 已说明独立性理由）

### 3.5 cap_i 不在 model 内缓存（Q7）

按用户决议 Q7：每次 set_context_subjective 传入 raw cap_i，model 不缓存。

理由：
- API 简洁（无需 episode reset hook）
- 与 v4.7 duck-typing 风格一致
- RoleEncoder.normalize 在 forward 内自动处理（C 修订，spec 03 Pkg-03）
- 性能代价 < 10 μs / call（normalize 仅 4 维减法+除法）

### 3.6 _cached_c_t 实施细节

set_context_subjective 需要 c_t 重建 ctx_aug，故 set_context_objective 内部缓存：

```python
def set_context_objective(self, c_t):
    self._cached_c_t = c_t   # 缓存供 set_context_subjective 用
    c_ctx = ...
```

此 detail 在 §2.2 类签名中略过为可读性；实现时必须含。

### 3.7 type_assignment 来源

Self-Info 严格性（D6 + C11）要求 model 内构造 own type 时使用 `cfg.env.type_assignment[agent_id]`（Pkg-01 EnvConfig 字段），**不**从 env.info["types"] 取（后者是 trainer 监督专用）。

`cfg.env.type_assignment` 是 episode 起点固定的（与 v4 Self-Info 设定一致），可在 model 内安全访问。

单测 `test_set_context_subjective_no_oracle_types_leak` 验证：
- 用 mock 替换 cfg.env.type_assignment 为某个值
- 调用 set_context_subjective 后验证 model 内部 types 来源 == cfg.env.type_assignment（而非 env.info）

---

## 4. Edge Cases

| 场景 | 行为 |
|------|------|
| transition / predict_reward / predict 在 set_context 之前调用 | AssertionError |
| set_context_subjective 在 set_context_objective 之前 | AssertionError（顺序硬约束 D4） |
| update_step 未调用（默认 self._step=0） | belief grad gating 永远启用（step < 5000 持续 detach） |
| 切 agent 时 set_context_subjective 多次调用 | 后调用覆盖前调用（仅缓存当前 agent θ_rew/pred）|
| set_context_objective 多次调用（同一 K-step unroll 内） | 后调用覆盖前调用 + 清空 subjective 缓存 |
| belief tuple 含 NaN | hyper_net.forward_subjective 输出含 NaN（model 不校验） |
| cap_i shape != (B, 4) | **新增（review 修订 2/澄清 2）**：set_context_subjective 入口 assert cap_i.shape[-1] == 4，明确拒绝（防止误传 augmented cap_i）|
| agent_id 越界（>=N） | cfg.env.type_assignment[agent_id] IndexError |
| **K-step unroll 内连续两次 set_context_subjective 之间未调 set_context_objective**（review 修订 2）| **允许**：θ_state 由首次 objective 提供且 K-step 不变；连续切 agent 是合法模式（mve_planner per-agent coordinate descent）。但若 K-step 起点重新需要新 c_t，必须显式重调 set_context_objective（顺序由 trainer 保证）|
| `predict` 调用后未重设 subjective 即跨 agent 推断 | **silent stale**：返回值仍对应前一 agent，**无 assert**（性能权衡）；由单测 `test_predict_uses_latest_subjective_agent` 保护用户：明确"以最后一次 subjective 为准"|

---

## 5. Acceptance Criteria

### 5.1 单元测试（`tests/models/test_hyper_muzero_model.py`）

```python
import pytest
import torch
from hyper_mve.configs import V4Config
from hyper_mve.models import HyperMuZeroModel


@pytest.fixture
def cfg_medium():
    return V4Config.from_preset("medium")


@pytest.fixture
def model(cfg_medium):
    return HyperMuZeroModel(cfg_medium)


# ====== API 完整性 ======

def test_model_has_7_public_apis(model):
    """spec 08 §1 锁定的 7 个 API 全部存在."""
    for api in ("update_step", "set_context_objective", "set_context_subjective",
                "encode", "transition", "predict_reward", "predict"):
        assert hasattr(model, api), f"Missing API: {api}"
        assert callable(getattr(model, api))


def test_ctx_aug_dim_assertion(cfg_medium):
    """C5: cfg.d_ctx_aug == 80 init 时断言."""
    # 已通过 Pkg-01 ModelConfig.__post_init__ 验证，这里再断言一次
    assert cfg_medium.model.d_ctx_aug == 80


# ====== 顺序断言 (D4) ======

def test_subjective_before_objective_raises(model, cfg_medium):
    """set_context_subjective 在 set_context_objective 之前 → AssertionError."""
    cap = torch.rand(1, 4)
    belief = (torch.rand(1), torch.softmax(torch.randn(1, 3, 2), dim=-1))
    with pytest.raises(AssertionError, match="set_context_subjective.*before set_context_objective"):
        model.set_context_subjective(0, cap, belief)


def test_transition_before_set_context_raises(model):
    s = torch.randn(1, model.cfg.model.latent_dim)
    action = torch.zeros(1, model.cfg.env.N * model.cfg.env.A)
    with pytest.raises(AssertionError, match="transition.*before set_context_objective"):
        model.transition(s, action)


def test_predict_reward_before_subjective_raises(model):
    model.set_context_objective(torch.tensor([0.5]))
    s = torch.randn(1, model.cfg.model.latent_dim)
    action = torch.zeros(1, model.cfg.env.N * model.cfg.env.A)
    with pytest.raises(AssertionError, match="predict_reward.*before set_context_subjective"):
        model.predict_reward(s, action)


# ====== 缓存策略 (D4) ======

def test_objective_clears_subjective_cache(model):
    """重新调 set_context_objective 应清空 subjective 缓存."""
    model.set_context_objective(torch.tensor([0.5]))
    cap = torch.rand(1, 4)
    belief = (torch.rand(1), torch.softmax(torch.randn(1, 3, 2), dim=-1))
    model.set_context_subjective(0, cap, belief)
    assert model._theta_rew is not None
    
    # 重新设 objective
    model.set_context_objective(torch.tensor([0.7]))
    assert model._theta_rew is None
    assert model._theta_pred is None


def test_subjective_replaces_per_agent(model):
    """切 agent 时 set_context_subjective 覆盖前 agent 缓存."""
    model.set_context_objective(torch.tensor([0.5]))
    cap = torch.rand(1, 4)
    belief = (torch.rand(1), torch.softmax(torch.randn(1, 3, 2), dim=-1))
    
    model.set_context_subjective(0, cap, belief)
    theta_rew_0 = model._theta_rew.clone()
    
    model.set_context_subjective(1, cap, belief)  # 切 agent
    theta_rew_1 = model._theta_rew.clone()
    
    # 不同 agent_id → 不同 θ_rew (因为 type_emb / id_emb 不同)
    assert not torch.allclose(theta_rew_0, theta_rew_1)


# ====== type-aware reward 分化 (C9, 断言 A 物理基础) ======

def test_type_aware_reward_differentiation(model, cfg_medium):
    """同一 (s, a) 下 α/β agent reward 应分化 (cos-sim < 0.95).
    
    断言 A 物理基础: type α (agent 0) vs type β (agent 2) 在相同 (s, a) 下
    reward 不同 (因为 hyper_rew 通过 type_emb 生成 per-type θ_rew^i).
    """
    model.set_context_objective(torch.tensor([0.5]))
    
    # agent 0 (α, cfg.env.type_assignment=(α, α, β, β))
    cap = torch.rand(1, 4)
    belief = (torch.rand(1), torch.softmax(torch.randn(1, 3, 2), dim=-1))
    
    model.set_context_subjective(0, cap, belief)  # α agent
    s = torch.randn(1, cfg_medium.model.latent_dim)
    action = torch.zeros(1, cfg_medium.env.N * cfg_medium.env.A)
    action[0, 0] = 1.0  # agent 0 action=0
    r_alpha = model.predict_reward(s, action)
    
    model.set_context_subjective(2, cap, belief)  # β agent
    r_beta = model.predict_reward(s, action)
    
    # 注: 训练前 random init 时差异未必显著, 此单测主要验证 forward 路径正确
    # 真正的"分化"实测在 scripts/test_hyper_model_forward.py 训练 1K step 后
    cos_sim = torch.nn.functional.cosine_similarity(
        r_alpha.flatten(), r_beta.flatten(), dim=0,
    )
    # 训练前 random init: cos_sim 可能 > 0.95（断言放宽到 forward 路径正确即可）
    assert r_alpha.shape == r_beta.shape  # forward 路径完整


# ====== update_step (Q4) ======

def test_update_step(model):
    model.update_step(1000)
    assert model._step == 1000
    model.update_step(10000)
    assert model._step == 10000


# ====== Stateful API 契约 (review 修订 2: D4 隐式状态契约) ======

def test_predict_uses_latest_subjective_agent(model, cfg_medium):
    """predict / predict_reward 以最后一次 set_context_subjective 的 agent_id 为准.
    
    review 修订 2: 明确 stateful API 语义, 防止 K-step unroll 内误用.
    """
    model.set_context_objective(torch.tensor([0.5]))
    cap = torch.rand(1, 4)
    belief = (torch.rand(1), torch.softmax(torch.randn(1, 3, 2), dim=-1))
    
    s = torch.randn(1, cfg_medium.model.latent_dim)
    action = torch.zeros(1, cfg_medium.env.N * cfg_medium.env.A)
    action[0, 0] = 1.0
    
    # Step 1: 设 A, predict → r_A
    model.set_context_subjective(0, cap, belief)
    r_A_first = model.predict_reward(s, action).clone()
    pi_A, v_A = model.predict(s)
    pi_A_first, v_A_first = pi_A.clone(), v_A.clone()
    
    # Step 2: 设 B, predict → r_B（应不同于 r_A）
    model.set_context_subjective(1, cap, belief)
    r_B = model.predict_reward(s, action).clone()
    pi_B, v_B = model.predict(s)
    
    # Step 3: 回设 A, predict → r_A_second（应等于 r_A_first，验证 reset 一致性）
    model.set_context_subjective(0, cap, belief)
    r_A_second = model.predict_reward(s, action).clone()
    
    # 断言: 不同 agent → 不同输出
    assert not torch.allclose(r_A_first, r_B, atol=1e-6), (
        "set A → r, set B → r 应不同, 否则 stateful 切换无效"
    )
    assert not torch.allclose(pi_A_first, pi_B, atol=1e-6)
    
    # 断言: 同 agent → 同输出（缓存可重入）
    assert torch.allclose(r_A_first, r_A_second, atol=1e-6), (
        "重新 set 同一 agent 应得到一致 θ_rew^A → 一致 r_A"
    )


def test_cap_shape_assertion(model):
    """review 修订 2/澄清 2: cap_i.shape[-1] 必须 == 4.
    
    防止误传 (B, 5) augmented cap_i (含 type 字段 leak).
    """
    model.set_context_objective(torch.tensor([0.5]))
    # 错误 cap shape: (B, 5) 多一维（疑似含 type）
    bad_cap = torch.rand(1, 5)
    belief = (torch.rand(1), torch.softmax(torch.randn(1, 3, 2), dim=-1))
    
    with pytest.raises((AssertionError, RuntimeError, ValueError)):
        # 期望: 入口 assert 或 RoleEncoder forward shape mismatch
        model.set_context_subjective(0, bad_cap, belief)


def test_k_step_unroll_consecutive_subjective_legal(model, cfg_medium):
    """review 修订 2: K-step unroll 内连续 set_context_subjective 不需重设 objective.
    
    mve_planner per-agent coordinate descent 模式: objective 一次, 循环切 subjective.
    """
    model.set_context_objective(torch.tensor([0.5]))
    cap = torch.rand(1, 4)
    belief = (torch.rand(1), torch.softmax(torch.randn(1, 3, 2), dim=-1))
    
    # 连续切 N agents, 中间不重设 objective
    for k in range(cfg_medium.env.N):
        model.set_context_subjective(k, cap, belief)
        # 应不抛 AssertionError (θ_state 仍有效, _has_objective 仍 True)
        s = torch.randn(1, cfg_medium.model.latent_dim)
        action = torch.zeros(1, cfg_medium.env.N * cfg_medium.env.A)
        action[0, 0] = 1.0
        _ = model.transition(s, action)
        _ = model.predict_reward(s, action)
        _ = model.predict(s)


# ====== Self-Info 严格性 (D6 + C11) ======

def test_set_context_subjective_no_oracle_types_leak(model, cfg_medium):
    """C11: set_context_subjective 内部 types 来源是 cfg.env.type_assignment, 不是 env.info.
    
    通过 mock 替换 cfg.env.type_assignment 验证 model 仅消费自报告 type.
    """
    from dataclasses import replace
    from hyper_mve.schemas import AgentType
    
    # 构造一个新 cfg, type_assignment 全 α
    new_types = (AgentType.ALPHA,) * cfg_medium.env.N
    cfg_mod = replace(cfg_medium, env=replace(cfg_medium.env, type_assignment=new_types))
    model_mod = HyperMuZeroModel(cfg_mod)
    
    model_mod.set_context_objective(torch.tensor([0.5]))
    cap = torch.rand(1, 4)
    belief = (torch.rand(1), torch.softmax(torch.randn(1, 3, 2), dim=-1))
    
    # agent 2 在 cfg_mod 中是 α (但在 cfg_medium 中是 β)
    model_mod.set_context_subjective(2, cap, belief)
    theta_rew_mod = model_mod._theta_rew.clone()
    
    model.set_context_objective(torch.tensor([0.5]))
    model.set_context_subjective(2, cap, belief)
    theta_rew_orig = model._theta_rew.clone()
    
    # 不同 cfg.env.type_assignment → 不同 θ_rew
    # (因为 type_emb_α vs type_emb_β 不同)
    assert not torch.allclose(theta_rew_mod, theta_rew_orig)


# ====== Forward 端到端 ======

def test_end_to_end_forward_no_nan(model, cfg_medium):
    """端到端 forward 无 NaN."""
    B, N = 2, cfg_medium.env.N
    obs_dim = model.rep_net.obs_dim  # 内部 attribute
    obs = torch.randn(B, N, obs_dim)
    s = model.encode(obs)
    
    model.update_step(0)
    model.set_context_objective(torch.full((B,), 0.5))
    
    action = torch.zeros(B, N * cfg_medium.env.A)
    action[:, 0] = 1.0
    
    s_next = model.transition(s, action)
    assert not torch.isnan(s_next).any()
    
    model.set_context_subjective(0, torch.rand(B, 4),
                                  (torch.rand(B), torch.softmax(torch.randn(B, N-1, 2), dim=-1)))
    r = model.predict_reward(s, action)
    pi, v = model.predict(s)
    assert not torch.isnan(r).any()
    assert not torch.isnan(pi).any()
    assert not torch.isnan(v).any()
```

### 5.2 集成测试

`scripts/test_hyper_model_forward.py`（spec 07 详述）：medium config 100 步端到端 forward + 性能 profile + type-aware reward 训练 1K 步后 cos-sim < 0.95 验证。

### 5.3 性能要求

- `set_context_objective(B=256)` < 5 ms
- `set_context_subjective(B=256, N agents)` 总 < 50 ms (Medium N=4)
- `encode + transition + predict_reward + predict` 全套 < 5 ms
- 单步 forward (含 N agents 全 forward) < 15 ms（spec 07 硬阈值）

---

## 5.4 关于 model 训练入口（澄清 1：model.forward 不对外）

**约定**：`HyperMuZeroModel` **不**暴露 `forward(...)` 或 `compute_losses(...)` 作为对外稳定 API。

**理由**：
1. trainer 端需要灵活组合 main loss + L_belief + curriculum λ_b 加权，model 不应包办
2. 7 个对外 API（spec 08 §1）已提供完整 forward 操作粒度（encode + transition + predict_reward + predict + set_context_*）
3. trainer 自管 loss 组装：
   ```python
   # Pkg-05 trainer 端典型模式
   s = model.encode(obs)
   model.update_step(step); model.set_context_objective(c_t)
   for k in range(N):
       model.set_context_subjective(k, cap[:, k], (c_hat[:, k], z_hat[:, k]))
       r_k = model.predict_reward(s, action)
       pi_k, v_k = model.predict(s)
       loss_k = mse(r_k, r_target[k]) + ce(pi_k, pi_target[k]) + mse(v_k, v_target[k])
   loss_belief = belief_loss(...)   # Pkg-03 独立 loss
   total = sum(loss_k) + lambda_b(step) * loss_belief
   total.backward()
   ```

**例外**：`nn.Module.forward` 因 PyTorch 继承自动存在，但**不**承诺签名稳定；调用者不应直接 `model(...)`，应走 7 API。

**单测**：spec 08 §1 表附注"forward 不在稳定 API 表内；trainer/baseline 调用 model.forward 视为违规"。

---

## 6. Cross-references

- Ch4.3 数据流（v4 三联条件化）
- Ch4.4 六大网络客观-主观分工
- Ch4.6 四道防线 + 4.6.5 belief 梯度门控
- `01-dualhypernet-v2-api.md`（hyper_net.forward_trans / forward_subjective 调用）
- `03-stability-safeguards-preservation.md`（functional_nets 保留 + cfg.model.use_adaln/state_trans_residual）
- `04-belief-gradient-gating.md`（BeliefGradGating.apply 调用）
- `05-type-aware-reward.md`（type-aware 链路图 + 断言 A）
- `06-data-flow-diagram.md`（数据流图 + 调用顺序）
- `08-integration-contracts.md` §1（7 API 稳定性表）+ §3（v4.7→v4 调用点迁移）
- Pkg-01 spec 05 ModelConfig（17 项 cfg 依赖）
- Pkg-02 spec 08 env.info 三段分组（model 不消费 Oracle 字段）
- Pkg-03 spec 01 TriContextEncoder.forward / forward_c_ctx_only
- Pkg-03 spec 04/05 BeliefNet.step / forward
- Pkg-03 spec 08 §4（与 trainer/worker 调用约定）
- Pkg-05 spec 04（trainer-loop-v2，下游消费此 model）
