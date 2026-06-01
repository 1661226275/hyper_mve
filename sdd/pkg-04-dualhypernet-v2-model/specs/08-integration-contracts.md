# Spec 08: 集成契约 — API 稳定性 + v4.7→v4 迁移 + 与 Pkg-05/06/07 协作

> 父文档：[`../proposal.md`](../proposal.md) §3.2 · [`../design.md`](../design.md) §6
> **本 spec 是 Pkg-04 对外的"硬契约"** — 与 Pkg-03 spec 08 对称。

---

## 1. Purpose

把 Pkg-04 内部模块（HyperMuZeroModel + DualHyperNetwork v2 + grad_gating + functional_nets）暴露给下游 Pkg-05 / Pkg-06 / Pkg-07 / Pkg-08 时，**定义必须严格遵守的契约**，防止：

- API 签名 drift（trainer/planner 端调用 model 时签名不匹配）
- v4.7 → v4 迁移漏点（mve_planner / muzero_trainer / worker 三处调用点未同步）
- Self-Info 违反（trainer 误把 Oracle types 传入 model.forward）
- 梯度门控阈值不同步（model 硬编码 vs cfg.train.belief_grad_gating_steps）

本 spec 是 Pkg-04 的"对外说明书"。

---

## 2. 契约 1：HyperMuZeroModel 7 API 稳定性承诺

### 2.1 API 稳定性表（M2：跨包伪签名先行）

| API | 稳定性 | 签名 | 调用者 |
|-----|--------|------|--------|
| `update_step(global_step: int)` | 🔒 稳定 | int → None | Pkg-05 trainer 每 train_step 起点 |
| `set_context_objective(c_t)` | 🔒 稳定 | (B,) or (B, 1) float32 → None | Pkg-05 trainer + worker + mve_planner |
| `set_context_subjective(agent_id, cap_i, belief)` | 🔒 稳定 | (int, (B, 4) float32, tuple of 2 tensors) → None | 同上 |
| `encode(obs)` | 🔒 稳定 | (B, N, obs_dim) → (B, latent_dim) | 同上 |
| `transition(s, action)` | 🔒 稳定 | ((B, latent), (B, N*A)) → (B, latent) | 同上 |
| `predict_reward(s, action)` | 🔒 稳定 | ((B, latent), (B, N*A)) → (B, 1) | 同上 |
| `predict(s)` | 🔒 稳定 | (B, latent) → ((B, A), (B, 1)) | 同上 |

**额外暴露**（与 Pkg-06a baselines 协作）：

| 模块 | 稳定性 | 暴露途径 | 用途 |
|------|--------|----------|------|
| `DualHyperNetwork` 类 | 🔒 稳定 | `from hyper_mve.models import DualHyperNetwork` | Pkg-06a Input-Wide/Deep 等参对照 baseline |
| `BeliefGradGating` 类 | 🔒 稳定 | `from hyper_mve.models.grad_gating import BeliefGradGating` | Pkg-06a/b shared_backbones 复用 |

**承诺**：Pkg-04 → Pkg-08 全程不变更上述 API。如需修改，必须发起 issue + 跨包讨论。

### 2.1.1 ⛔ 不在稳定 API 表内的入口（澄清 1）

| 入口 | 状态 | 说明 |
|------|------|------|
| `HyperMuZeroModel.forward(...)` | ⚠️ **不对外稳定**（PyTorch nn.Module 继承自动存在，签名不承诺）| trainer/baseline **不应**直接 `model(...)`；应调用上表 7 API 自行组装 forward 与 loss |
| `HyperMuZeroModel.compute_losses(...)` | ⛔ **不提供** | model 不包办 loss 组装；λ_b 课程加权 + L_belief 独立 backward 全在 trainer 端（spec 02 §5.4 + Pkg-05 spec 04） |

**违规检测**：Pkg-05 实施时 PR 审查 + grep 验证：

```powershell
# trainer/baseline 代码内不应直接调 model(...) 或 model.forward(...) / model.compute_losses(...)
grep -rn "model\.forward\|model\.compute_losses\|model(" hyper_mve/training hyper_mve/baselines
# 期望: 仅 model.update_step / model.set_context_* / model.encode / model.transition / model.predict_reward / model.predict
```

### 2.2 调用顺序契约（D4）

```
1. (trainer-only) model.update_step(global_step)        # 每 train_step 起点 1 次
2. model.set_context_objective(c_t)                     # 每 K-step unroll 起点 1 次
3. for k in agents:
       model.set_context_subjective(k, cap, belief)     # 切 agent
       # 以下方法可任意顺序调用
       model.encode(obs)
       model.transition(s, action)
       model.predict_reward(s, action)
       model.predict(s)
4. K-step unroll: 重复 step 3
```

**断言**：
- `set_context_subjective` 在 `set_context_objective` 之前 → AssertionError
- `transition` / `predict_reward` / `predict` 在 `set_context_*` 之前 → AssertionError

---

## 3. 契约 2：v4.7 → v4 三处调用点迁移指引（Q2 用户决议 Pkg-04 范围）

按用户决议 Q2，**mve_planner.py / muzero_trainer.py / worker.py 三处的 set_context 调用点迁移属 Pkg-04 PR 范围**。

### 3.1 mve_planner.py（4 处调用点）

| 行号 | v4.7 代码 | v4 迁移代码 | 备注 |
|------|----------|-------------|------|
| L71 | `model.set_context(rule_exp, id_i)` | `model.set_context_subjective(int(id_i[0]), cap[id_i[0]], belief[id_i[0]])` | planner 入口处需先调 `model.set_context_objective(c_t)`（一次） |
| L213 | `model.set_context(rule_exp, id_0)` | `model.set_context_subjective(int(id_0[0]), cap[id_0[0]], belief[id_0[0]])` | 同 L71 |
| L219 | `model.set_context(rule_exp, id_j)` | `model.set_context_subjective(int(id_j[0]), cap[id_j[0]], belief[id_j[0]])` | 同 |
| L258 | `model.set_context(rule_exp, id_j)` | `model.set_context_subjective(int(id_j[0]), cap[id_j[0]], belief[id_j[0]])` | 同 |

**新增 planner 入口逻辑**（约在 mve_planner.py L80 `sample_mve_plan` 函数入口）：

```python
# v4 新增: planner 入口处一次性 set_context_objective
def sample_mve_plan(model, root_s, cfg, rule=None):
    # ... 原始代码 ...
    
    is_hyper = _is_hyper_model(model)
    if is_hyper:
        c_t = rule_exp  # 假设 rule 即为 c_t（v4 改名后）
        model.set_context_objective(c_t)              # ← 新增, 一次性
    
    # 后续循环内只调 set_context_subjective (已修改 L71/213/219/258)
```

**注意点**：
- v4.7 `rule_exp` 在 v4 改名为 `c_t`（仅语义重命名，shape (B,) 不变）
- v4 需在外部传入 `cap`、`belief` 字典（agent_id → tensor），由 trainer / worker 准备好后传给 planner
- 如 planner 接口签名要扩展（加 `cap` / `belief` 参数），spec 02 + spec 08 §3.1 同步说明

### 3.2 muzero_trainer.py（6 处调用点）

| 行号 | v4.7 代码 | v4 迁移代码 | 备注 |
|------|----------|-------------|------|
| L239 | `model.set_context(rules, agent_ids)` | 拆为 `model.set_context_objective(c_t)` + 循环 `model.set_context_subjective(k, cap, belief)` | trainer K-step unroll 起点 |
| L242 | `self.target_model.set_context(rules, agent_ids)` | 同上（target model 同步迁移）| EMA target 也需同 set_context |
| L474 | `model.set_context_from_history(...)` | **删除**（v4.7 Infer 模式废弃，BeliefNet 取代）| Infer 双类合并 |
| L484 | `self.target_model.set_context(inferred_rule_emb.detach(), agent_ids)` | 拆为两步 + 用 BeliefNet 输出替代 inferred_rule_emb | – |
| L678 | `model.set_context(rule_t, id_t)` | 拆为两步 + cap/belief 准备 | – |
| L680 | `model.set_context_default(id_t, batch_size=1)` | **删除**（default 模式废弃，v4 用 BeliefNet 输出 belief 替代）| – |

**新增 trainer 主循环**（参考 spec 02 §2.3 伪代码）：

```python
# v4 trainer train_step (示意)
def train_step(self, batch, global_step):
    self.model.update_step(global_step)             # ← 新增, Q4
    
    # K-step unroll 起点
    c_t = batch["c_t"]
    self.model.set_context_objective(c_t)           # ← 拆出 objective
    
    if not self.target_model_disabled:
        self.target_model.update_step(global_step)
        self.target_model.set_context_objective(c_t)
    
    for k in range(N):
        self.model.set_context_subjective(k, batch["cap"][:, k],
                                           (batch["c_hat"][:, k], batch["z_hat"][:, k]))
        if not self.target_model_disabled:
            self.target_model.set_context_subjective(k, batch["cap"][:, k],
                                                      (batch["c_hat"][:, k], batch["z_hat"][:, k]))
        
        # K-step unroll 内 per agent forward
        # ... predict_reward / predict / 计算 loss ...
    
    # ... loss.backward(), optimizer.step() ...
```

### 3.3 worker.py（5 处调用点）

| 行号 | v4.7 代码 | v4 迁移代码 |
|------|----------|-------------|
| L125 | `model.set_context_from_history(h_obs, h_act, h_rew, id_0)` | **删除** + 用 `model.belief_net.step(obs_t, prev_hidden)` 替代 |
| L128 | `model.set_context_default(id_0, batch_size=1)` | **删除** + `model.set_context_subjective(0, cap, belief)` |
| L143 | `model.set_context_from_history(h_obs, h_act, h_rew, id_i)` | **删除** + 同上 |
| L145 | `model.set_context_default(id_i, batch_size=1)` | **删除** + `model.set_context_subjective(i, cap, belief)` |
| L149 | `model.set_context(rule_t, id_i)` | 拆为 `model.set_context_objective` + `model.set_context_subjective` |

**新增 worker 主循环**：

```python
# v4 worker collect_episode (示意)
def collect_episode(self, env, model, ...):
    obs, info = env.reset()
    
    N = model.cfg.env.N
    prev_hidden = model.belief_net.init_hidden(B=1, num_agents=N)
    
    for t in range(T):
        # BeliefNet step (取代 v4.7 set_context_from_history)
        obs_tensor = torch.from_numpy(obs).unsqueeze(0)
        prev_hidden, c_hat, z_hat = model.belief_net.step(obs_tensor, prev_hidden)
        # c_hat (1, N), z_hat (1, N, N-1, 2)
        
        # Model forward
        s = model.encode(obs_tensor)
        c_t = torch.tensor([info["c_true"]])
        model.set_context_objective(c_t)            # ← 一次
        
        for k in range(N):
            cap_k = torch.from_numpy(info["caps"][k].to_array()).unsqueeze(0)
            model.set_context_subjective(k, cap_k, (c_hat[0, k:k+1], z_hat[0, k:k+1]))
            
            pi_k, v_k = model.predict(s)
            action[k] = sample_from(pi_k)
        
        obs, _, done, _, info = env.step(action)
```

### 3.4 迁移验证

```powershell
# 迁移完成后 grep 验证: v4.7 旧 API 应全部消失
grep -rn "set_context(" hyper_mve/planning hyper_mve/training
# 期望: 仅出现 set_context_objective( 或 set_context_subjective( 

grep -rn "set_context_from_history\|set_context_default" hyper_mve/training
# 期望: 无任何匹配 (v4.7 Infer/Default API 已删除)
```

**单测**：`tests/migration/test_v47_to_v4_set_context.py` 验证：

```python
def test_no_legacy_set_context_calls():
    """v4.7 旧 set_context(rule, id) 双参签名应全部消失."""
    import subprocess
    result = subprocess.run(
        ['grep', '-rn', r'set_context([^_]', 'hyper_mve/planning', 'hyper_mve/training'],
        capture_output=True, text=True,
    )
    # 应仅匹配 set_context_objective 或 set_context_subjective
    # 不应匹配 set_context(rule, id) 旧签名
    for line in result.stdout.split('\n'):
        if line and 'set_context(' in line:
            assert 'set_context_objective(' in line or 'set_context_subjective(' in line, (
                f"Found legacy set_context call: {line}"
            )
```

---

## 4. 契约 3：与 Pkg-05 (Trainer & Worker) 协作

### 4.1 trainer 端调用约定

| trainer 责任 | 调用 API | 频率 |
|--------------|---------|------|
| 传递 global_step | `model.update_step(step)` | 每 train_step 1 次 |
| 准备 belief tuple (来自 BeliefNet forward) | `model.set_context_subjective(k, cap, (c_hat, z_hat))` | per agent per step |
| 课程 Stage 1 oracle 注入 | 通过 BeliefNet.forward(oracle_z_seq=...) 控制（**不**经 model API） | – |
| 计算 main loss | trainer 自己 backward；model 仅提供 forward | – |
| 计算 L_belief loss | trainer 调 `belief_loss(...)`（Pkg-03 spec 06） | – |

### 4.2 worker 端调用约定

| worker 责任 | 调用 API |
|-------------|----------|
| **不**调 update_step | worker 永远 no_grad，grad gating 无意义 |
| 在线 BeliefNet 推断 | `model.belief_net.step(obs, prev_hidden)` |
| 在线 model forward | 标准 5 API 调用 |
| 写入 TimeStepRecord | c_hat / z_hat 字段按 Pkg-01 spec 04 顺序约定 |

### 4.3 trainer ↔ worker 接口约束

worker collect 的 TimeStepRecord 中 `c_hat` / `z_hat` 字段顺序必须与 trainer 端 `model.set_context_subjective` 调用顺序一致：

| 顺序约定 | 单测 |
|----------|------|
| `record.c_hat[k]` 对应 agent_k 的 ĉ | Pkg-05 spec 06 worker spec |
| `record.z_hat[k]` 对应 agent_k 视角的 (N-1) 对手 | Pkg-01 spec 04 + Pkg-03 spec 05 + spec 06 三处一致 |

---

## 5. 契约 4：与 Pkg-06 (Baselines) 协作 — shared_backbones 工厂

按 Pkg-03 spec 08 §6：所有 7 baseline 共享同一 BeliefNet/TriContextEncoder 结构（断言 B 等参公平性强制）。Pkg-04 提供：

### 5.1 暴露给 shared_backbones 的接口

```python
# Pkg-06a/b shared_backbones.py 调用
from hyper_mve.models import HyperMuZeroModel, DualHyperNetwork
from hyper_mve.models.functional_nets import (
    FunctionalStateTransNet, FunctionalRewardHead, FunctionalPredictionNet,
)


def create_hyper_muzero_model(cfg):
    """Hyper-MuZero (主方法) 工厂."""
    return HyperMuZeroModel(cfg)


def create_input_wide_baseline(cfg, total_params_target):
    """Input-Wide baseline (Ablation 1 等参对照).
    
    复用 RepNet + BeliefNet (Pkg-03 spec 08 §6),
    但**不用 hypernet** — concat ctx_aug 进加宽的 functional nets.
    """
    rep_net = HyperMuZeroModel(cfg).rep_net   # 复用 RepNet 结构
    belief_net = HyperMuZeroModel(cfg).belief_net   # 复用 BeliefNet 结构
    # ... 不用 DualHyperNetwork, 改用 concat + 加宽 functional ...
```

### 5.2 关键约束

- baseline 必须复用同一 RepNet 类（不能各自实现 RepNet）→ 参数量对齐
- baseline 必须复用同一 BeliefNet 类（断言 B 公平性核心）→ 参数量 + 训练状态对齐
- 仅"如何处理 ctx_aug → 功能网络"的部分各 baseline 不同（Hyper vs Input-Wide/Deep vs MA-MuZero）

---

## 6. 契约 5：与 Pkg-07 (Eval Protocols) 协作 — Self-Info 严格性

### 6.1 评估时的 model 调用约束

| 评估协议 | model.set_context_subjective 调用约束 |
|----------|--------------------------------------|
| Self-Info eval（Pkg-07 spec 04） | belief 参数必须来自 `BeliefNet.forward()` 推断输出（**不**传 oracle_z）|
| Oracle eval（对照实验） | belief 参数可传 oracle_z 替代 head_opp 输出 |
| c-segment eval（Pkg-07 spec 05） | c_t 来自 env.reset(options={"c": value}) |
| Zero-shot c eval（Pkg-07 spec 06） | 训练用 c ∈ {0.2, 0.5, 0.8}; 评估 c 为 unseen 值 |

### 6.2 Self-Info 严格性单测（澄清 2：双层防御 — Pkg-04 runtime + Pkg-07 静态审计）

C11 (Self-Info 严格性) 通过 **两层验证** 防止 oracle types 进入 model：

**第一层（Pkg-04 runtime assert，spec 02 §2.2）**：

`set_context_subjective` 入口对 `cap_i` 做严格 shape assert：

```python
# spec 02 §2.2 已加
assert cap_i.dim() == 2 and cap_i.shape[-1] == 4, (
    f"cap_i.shape must be (B, 4), got {tuple(cap_i.shape)}. "
    f"v4 CapabilityVector 严格 4 元组; 5 维等扩展形疑似含 type leak — C11 违规."
)
```

意图：拒绝 caller 误传 `cap_i = (B, 5)`（拼接了 type 维度）或 `cap_i = (B, 4+...)`（augmented）。
单测：`test_cap_shape_assertion`（spec 02 §5.1）。

**第二层（Pkg-04 mock test，spec 02 §5.1）**：

`test_set_context_subjective_no_oracle_types_leak`（已在 spec 02 §5.1）通过 mock 替换 `cfg.env.type_assignment` 验证 model 内部 types **仅** 从 `cfg.env.type_assignment` 取，**不**从 `env.info["types"]` 取：

| 步骤 | 操作 | 期望 |
|------|------|------|
| 1 | 构造 cfg_mod，type_assignment 全设为 ALPHA | – |
| 2 | model_mod.set_context_subjective(agent_id=2, ...) | model 取 own_type = ALPHA（cfg_mod 设的）|
| 3 | model.set_context_subjective(agent_id=2, ...)（原 cfg type_assignment[2]=BETA） | model 取 own_type = BETA |
| 4 | 比较两个 model 的 θ_rew | 不同（type_emb 不同）|

**第三层（Pkg-07 静态审计，spec 04 self-info-eval）**：

Pkg-07 `test_self_info_no_oracle_leak`（待定）：
- 用 AST 扫描 eval 脚本，检测是否出现 `belief=(env.info[..."types"...], ...)` 模式
- 检测 `set_context_subjective(...)` 调用点 belief 参数来源是否为 `BeliefNet.forward(...)` 输出
- grep 反例: `grep -rn "info\[['\"]types['\"]\].*set_context" hyper_mve/`

**Pkg-04 端的责任界限**：
- runtime shape assert（拒绝 cap shape 误传）✅ 在 Pkg-04 范围
- mock test 验证 types 取自 cfg ✅ 在 Pkg-04 范围
- eval 脚本静态审计 ⏳ 在 Pkg-07 范围（model 不主动校验 belief 来源，evaluator 自律）

**实施层关键**：spec 02 documentation 明确"set_context_subjective 接受的 belief 来源 model 不验证（性能权衡），但 evaluator 必须自律 + Pkg-07 静态审计兜底"。

---

## 7. 契约 6：与 Pkg-03 (TriContextEncoder + BeliefNet) 协作回顾

按 Pkg-03 spec 08 §5（belief 梯度门控分层）+ §6（baseline 共享）+ §7（buffer 字段顺序一致性），Pkg-04 严格遵守：

### 7.1 BeliefNet raw heads 接收点

Pkg-04 `set_context_subjective` 内的 `self.grad_gating.apply(c_hat, z_hat, self._step)` 是**唯一**的梯度切断点（与 Pkg-03 spec 08 §5 一致）。

### 7.2 BeliefEncoder 责任分层

Pkg-04 **不**调用 BeliefEncoder（在 Pkg-03 TriContextEncoder 内部）。Pkg-04 仅传 raw (c_hat, z_hat) 给 TriContextEncoder，由其内部 BeliefEncoder 处理。

### 7.3 z_hat 顺序约定（端到端一致性）

三处必须严格一致：

| 位置 | 顺序约定 |
|------|---------|
| Pkg-01 `TimeStepRecord.z_hat` 字段 docstring | agent_id 升序跳过 self |
| Pkg-03 `BeliefNet.head_opp` 输出 | 同 |
| Pkg-03 `l_opp` 标签 gather | 同 |
| Pkg-03 `build_oracle_z_seq` 输出 | 同 |
| Pkg-04 `model.set_context_subjective` 接收的 belief tuple | 同 |

**单测**：Pkg-03 spec 08 §7.4 已含 `test_z_hat_order_convention`（端到端验证）。

---

## 8. 契约 7：cfg 字段依赖完备性（M4）

按 spec 02 §1.1 列出的 17 项 cfg 字段，Pkg-04 model.__init__ 必须**全部读取**。任何字段缺失或重命名需 Pkg-01 spec 05 同步。

**实施时验证**：

```python
# Pkg-04 model.__init__ 内追加
def __init__(self, cfg: V4Config):
    super().__init__()
    
    # 字段完备性自检（实施时启用，PR merge 前删除）
    REQUIRED_MODEL_FIELDS = [
        "d_c", "d_role", "d_belief", "d_ctx_aug",
        "d_id_emb", "d_type_emb", "d_cap_emb", "d_belief_proj",
        "latent_dim", "hidden_dim",
        "hyper_hidden_dims", "hyper_rew_hidden_dims",
        "trans_output_scale_init", "rew_output_scale_init", "pred_output_scale_init",
        "use_adaln", "adaln_residual_one_plus", "state_trans_residual",
        "belief_gru_hidden", "belief_pool", "proj_dim",
    ]
    REQUIRED_TRAIN_FIELDS = [
        "belief_grad_gating_steps", "detach_pred_context", "unroll_K",
    ]
    
    for field in REQUIRED_MODEL_FIELDS:
        assert hasattr(cfg.model, field), f"cfg.model missing field: {field}"
    for field in REQUIRED_TRAIN_FIELDS:
        assert hasattr(cfg.train, field), f"cfg.train missing field: {field}"
```

---

## 9. 集成测试 checklist（Pkg-04 → Pkg-05/06/07 联调）

Pkg-04 PR merge 前应通过以下集成测试：

| # | 测试 | 责任包 |
|---|------|--------|
| 1 | model 7 API 端到端 forward 维度对齐 | Pkg-04 spec 02 §5.1 |
| 2 | BeliefNet.step → TimeStepRecord.c_hat/z_hat 字段填充正确 | Pkg-04 spec 08 §4.3 + Pkg-05 spec 06 |
| 3 | 梯度门控前 5K step BeliefNet 参数梯度 = 0 | Pkg-04 spec 04 |
| 4 | mve_planner.py 4 处调用点迁移完成（grep 验证） | spec 08 §3.4 |
| 5 | muzero_trainer.py 6 处调用点迁移完成 | spec 08 §3.4 |
| 6 | worker.py 5 处调用点迁移完成 | spec 08 §3.4 |
| 7 | 单步 forward < 15 ms (Medium config) | Pkg-04 spec 07 |
| 8 | 参数量 ∈ [2.8M, 3.6M] | Pkg-04 spec 07 |
| 9 | type-aware reward 训练 1K 步后 cos-sim < 0.95 | Pkg-04 spec 05 §5.2 |
| 10 | Self-Info eval 时 belief 来自 BeliefNet 推断 | Pkg-07 spec 04 |
| 11 | Shared backbone 在 7 baseline 间结构一致 | Pkg-06a spec 01 |

---

## 10. Cross-references

- 全部 7 个 Pkg-04 specs
- Pkg-01 spec 04 (TimeStepRecord) + spec 05 (V4Config 17 项字段)
- Pkg-02 spec 08 (env.info 三段分组：Public / Oracle / EvalOnly)
- Pkg-03 spec 01 (TriContextEncoder) + spec 04/05 (BeliefNet) + spec 06 (belief_loss) + spec 08 (集成契约)
- Pkg-05 spec 04 (trainer-loop-v2)、spec 06 (worker-data-collection)
- Pkg-06a spec 01 (shared-backbones)
- Pkg-07 spec 04 (self-info-eval)
- v4.7 行号映射:
  - `models/hyper_network.py:63-82` (L2 norm + output_scale)
  - `models/functional_nets.py:43-77` (AdaLN) + L245-271 (Δs 残差)
  - `planning/mve_planner.py:71/213/219/258` (4 处 set_context 调用点)
  - `training/muzero_trainer.py:239/242/474/484/678/680` (6 处)
  - `training/worker.py:125/128/143/145/149` (5 处)
- Ch3.7 + Ch4.2.2 Self-Info 严格性
- Ch4.6.5 belief 梯度门控
