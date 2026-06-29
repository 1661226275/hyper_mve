# Hyper-MuZero v4 代码主视图（Pkg-01 ~ Pkg-05 + 优化阶段）

> 面向「开始实验」的导读：先看 §1 核心思想 → §2 分层架构 → §5 怎么训练 → **§11 优化阶段（gen_scope / LoRA / sweep）**。
> 权威规格见 `sdd/pkg-0X-*/`，本文件是代码侧的速查地图。理论复审与文档↔代码审计见 `docs/Review_v4_TheoryAudit_2026-06.md`。

---

## 1. 一句话 + 核心思想（view = perspective）

把**环境规则**与**智能体身份/信念**统一成一个 **80 维增广上下文** `C_aug`，
喂给一个 **双超网络（DualHyperNetwork）**，由它**动态生成**三个小功能网的权重。
换 agent 视角 = 重新生成权重，而不是换网络结构。

```
C_aug = [ c_ctx(16) | role(32) | belief(32) ]   = 80 维
          规则丰度    身份(id+type+cap)  信念(ĉ + ẑ)
              │            │                │
              ▼            ▼                ▼
       hyper_trans     hyper_rew        hyper_pred      (DualHyperNetwork, Pkg-04)
              │            │                │
            θ_state      θ_rew^i          θ_pred^i        (动态权重)
              │            │                │
      StateTransNet   RewardHead      PredictionNet      (功能网, Pkg-04)
       s,a → s'        s,a → r          s → π, v
        客观           主观(每个agent)   主观(每个agent)
```

- **客观流**：`hyper_trans` 只吃 `c_ctx`（规则），生成 `θ_state`，所有 agent 共享 → 世界状态转移与「谁在看」无关。
- **主观流**：`hyper_rew / hyper_pred` 吃完整 80 维，per-agent 生成 `θ_rew^i / θ_pred^i` → 奖励、策略、价值随视角变化。

---

## 2. 分层架构与依赖（Pkg-01 → Pkg-05）

```
Pkg-01 基础 Schema + 配置        hyper_mve/schemas/ , hyper_mve/configs/
   │  AgentType / CapabilityVector / TimeStepRecord / ObservationLayout / V4Config
   ▼
Pkg-02 ResourceCommons 环境      hyper_mve/envs/resource_commons/
   │  gym.Env：reset/step/info(三段：Public/Oracle/EvalOnly)
   ▼
Pkg-03 三路上下文编码 + 信念网    hyper_mve/models/  (belief_*, *_encoder, tri_context_encoder)
   │  BeliefNet(GRU→ĉ,ẑ) + TriContextEncoder(→ctx_aug 80) + belief_losses
   ▼
Pkg-04 双超网络 + HyperMuZero     hyper_mve/models/  (hyper_*, functional_nets, representation_net, grad_gating)
   │  HyperMuZeroModel(7 API) + DualHyperNetwork + 功能三网 + RepNet + 梯度门控
   ▼
Pkg-05 训练器 & Worker           hyper_mve/training/ , hyper_mve/planning/ , hyper_mve/scripts/
      MuZeroTrainer / Worker / EpisodeReplayBuffer / CurriculumScheduler /
      compose_total_loss / MVEPlanner / train_main.py
```

---

## 3. 各包模块职责速查

### Pkg-01 — 基础 Schema + 配置（`schemas/` + `configs/`）
| 文件 | 作用 |
|------|------|
| `schemas/agent_type.py` | `AgentType`：`ALPHA=0`(自利) / `BETA=1`(Fehr-Schmidt 不公平厌恶) |
| `schemas/capability.py` | `CapabilityVector(eta,phi_fov,nu,zeta)` 采集速度/视野/移动可靠性/容量 + `normalize()` |
| `schemas/buffer_record.py` | `TimeStepRecord`（12 字段：o,a,r,delta,pi_mve,v,tau,cap,c_hat,z_hat,t,done）|
| `schemas/observation.py` | `ObservationLayout.total_dim(N,K)` 六块观测布局 |
| `configs/v4_config.py` | `V4Config`（env/model/train/mup/eval/legacy）+ `from_preset` / `to_dict` |
| `configs/presets/{easy,medium,hard}.py` | 三档难度参考配置（medium 是主对照）|
| `configs/presets/duo*.py, medium_*lora*.py` | **[v4-opt]** duo 诊断族(N=2, 1α+1β, random_walk)与 gen_scope/LoRA 预设矩阵(§11)|

### Pkg-02 — ResourceCommons 环境（`envs/resource_commons/`）
| 文件 | 作用 |
|------|------|
| `env.py` | `ResourceCommonsEnv`（gym.Env）：`reset()→(obs,info)`、`step(a)→(obs,r,done,trunc,info)` |
| `dynamics.py` | 资源 logistic 再生 + fair-share 采集 |
| `rewards.py` | 类型相关奖励（ALPHA 自利；BETA 不公平厌恶 Δ）|
| `observations.py` | 六块联合观测 `(N, obs_dim)` |
| `context_evolution.py` | 规则丰度 `c_t` 演化（static / drift）|
| `state.py` / `spawn.py` | 状态容器 / patchy 资源生成 |

> **info 三段分组（载荷契约）**：`Public`(caps/deltas/step_idx/harvests) 模型可用；
> `Oracle`(c_true/types) **只给训练监督，绝不进 model.forward**；`EvalOnly`(hotspot/resource_state) 仅评估可视化。

### Pkg-03 — 三路上下文编码 + 信念网（`models/`）
| 文件 | 作用 |
|------|------|
| `belief_net.py` | `BeliefNet`：共享 GRU + `head_c`(sigmoid 标量 ĉ) + `head_opp`(对手类型 2 分类 softmax ẑ)。`step()` 在线单步 / `forward()` 序列 |
| `belief_losses.py` | `l_c`(MSE) / `l_opp`(Oracle CE) / `l_div`(hinge 方差防坍缩) / `belief_loss` / `build_oracle_z_seq` |
| `c_encoder.py` | `CEncoder`：`c_t` → `c_ctx`(16) |
| `role_encoder.py` | `RoleEncoder`：id_emb + 自身 type_emb + cap → `role`(32)（Self-Info：只用自己的 type）|
| `belief_encoder.py` | `BeliefEncoder`：ĉ + 池化(ẑ) → `belief_vec`(32) |
| `tri_context_encoder.py` | `TriContextEncoder`：拼三路 → `ctx_aug`(80)，含三路 LayerNorm |

### Pkg-04 — 双超网络 + HyperMuZero 模型（`models/`）
| 文件 | 作用 |
|------|------|
| `hyper_muzero_model.py` | `HyperMuZeroModel`：**7 个对外 API**（见 §4）|
| `hyper_network.py` | `DualHyperNetwork`：`forward_trans(c_ctx)→θ_state`、`forward_subjective(ctx_aug)→(θ_rew,θ_pred)`；`detach_pred_context=True` 时只让 reward 梯度回流 ctx_aug |
| `functional_nets.py` | `FunctionalStateTransNet`(Δs 残差+AdaLN)、`FunctionalRewardHead`、`FunctionalPredictionNet`（权重外部注入）|
| `representation_net.py` | `RepresentationNet`(obs→s 客观潜状态) + `Projector`(BYOL 一致性) |
| `grad_gating.py` | `BeliefGradGating`：前 `belief_grad_gating_steps`(=5000) 步把 belief 梯度 detach（双层），让 BeliefNet 先由 L_belief 专项训练 |

### Pkg-05 — 训练器 & Worker（`training/` + `planning/` + `scripts/`）
| 文件 | 作用 |
|------|------|
| `training/muzero_trainer.py` | `MuZeroTrainer`：单一 `train_step` + EMA target(τ=0.99) + warmup_cosine LR + n-step return + v4 checkpoint |
| `training/worker.py` | `Worker.collect_episode`：`BeliefNet.step` 在线推断 + 7-API 选动作 + MVE planner → `TimeStepRecord` |
| `training/episode_buffer.py` | `EpisodeReplayBuffer`：TimeStepRecord 容器 + 分层采样(stratified) |
| `training/curriculum.py` | `CurriculumScheduler`：3 stage + `oracle_z_mixing_weight` + `lambda_b` |
| `training/loss_composition.py` | `compose_total_loss`：main loss + L_belief **双路径** backward |
| `planning/mve_planner.py` | `MVEPlanner`：CRN(共同随机数) + 坐标下降，产出 `π_mve` 搜索策略 |
| `scripts/train_main.py` | **统一训练入口**（`--preset/--variant/--override/...`）|

---

## 4. HyperMuZeroModel 的 7 个 API（训练/采集都靠它们拼装）

```python
model.update_step(global_step)                      # 1) 更新内部 step（驱动梯度门控阈值）
model.set_context_objective(c_t)                    # 2) 由规则 c_t 算 θ_state（每次 unroll 起点 1 次）
model.set_context_subjective(agent_id, cap_i, belief)  # 3) 由 (cap, ĉ, ẑ) 算 θ_rew^i / θ_pred^i（每个 agent 切一次）
s   = model.encode(obs)                             # 4) RepNet：obs → 潜状态 s
s2  = model.transition(s, action_onehot)            # 5) 客观转移：s,a → s'（用 θ_state）
r   = model.predict_reward(s, action_onehot)        # 6) 主观奖励（用当前 agent 的 θ_rew）
pi,v= model.predict(s)                              # 7) 主观策略+价值（用当前 agent 的 θ_pred）
```

**调用顺序（硬约束）**：`update_step → set_context_objective(一次) → for k in N: set_context_subjective(k) + 前向`。
注意 reward/value 是**标量变换空间**（MuZero `scalar_transform`），算 loss 时目标也要变换。

---

## 5. 训练数据流（一个 iteration）

```
[采集] Worker.collect_episode(env)            # 不反传，BeliefNet.step 在线推断
   每步：obs → belief_net.step → ĉ,ẑ
         encode(obs)=s；set_context_objective(c_true)
         for k in N: set_context_subjective(k,cap_k,(ĉ_k,ẑ_k)) → predict
         (可选) MVEPlanner.sample_mve_plan → π_mve
         ε-greedy 选 a → env.step → 写 TimeStepRecord(+ c_t)
         ↓
[存储] buffer.store_episode(records, c_t_seq)      # FIFO + 分层

[训练] buffer.sample_batch(B=256,K=5) → batch       # (B, K+1, N, ...)
   trainer.train_step(batch, step):
     model/target.update_step(step)
     compose_total_loss:
       ① belief_net.forward(obs) → (hidden, ĉ_pred, ẑ_pred)        [带梯度]
       ② L_belief = belief_loss(预测值, c_true, types)              [始终训练 BeliefNet]
       ③ 课程混合：w=oracle_z_mixing_weight(step)；z_main = w·oracle + (1-w)·ẑ_pred
       ④ set_context_objective(c_t)；K 步展开，对 N 个 agent：
              set_context_subjective → predict/predict_reward
              累积 L_policy(CE) + L_value(n-step,MSE) + L_reward(MSE) + L_consist(BYOL)
       ⑤ L_total = L_main + λ_b·L_belief
     L_total.backward() → clip → optimizer.step → lr_scheduler.step → EMA 更新 target
```

**双路径梯度（关键）**：`L_belief` 始终直接训练 BeliefNet；main loss 经 `set_context_subjective`
进入模型，前 5000 步被 `grad_gating` detach（BeliefNet 先专项预热），5000 步后才让 main loss
（实际上只有 reward 路，因 `detach_pred_context=True`）也回流 BeliefNet。

**课程 3 阶段**（medium，max=1,000,000）：
| 阶段 | 区间 | oracle 注入权重 w | 含义 |
|------|------|------------------|------|
| Stage 1 Pure Oracle | 0 – 300k | 1.0 | 主路 belief 用真值 type（ẑ 注入 oracle）|
| Stage 2 Anneal | 300k – 700k | 1.0 → 0.0 线性 | 逐步切到 BeliefNet 推断 |
| Stage 3 Pure Inference | 700k – 1M | 0.0 | 完全靠 BeliefNet 自己推 |

---

## 6. 怎么开始训练

> 在 Linux 训练机、`lightzero` 环境、仓库根目录（`hyper_mve/__init__.py` 的上一层）运行。

```bash
cd /home/data/zhengwenbo/hyper_mve
conda activate lightzero

# 0) 先跑一个最小冒烟，确认全链路通（几分钟）
python hyper_mve/scripts/train_main.py --preset medium --variant hyper \
    --max_steps 200 \
    --override "train.min_buffer_size=4" --override "train.batch_size=16" \
    --override "env.T_max=60" --ckpt_dir checkpoints/smoke --seed 0

# 1) 正式训练（medium 主配置，1M 步；先 collect 1000 集再训，耗时较长）
python hyper_mve/scripts/train_main.py --preset medium --variant hyper \
    --max_steps 1000000 --seed 0 \
    --log_dir runs/exp1_hyper --ckpt_dir checkpoints/exp1_hyper

# 2) 续训
python hyper_mve/scripts/train_main.py --preset medium --variant hyper \
    --resume_from checkpoints/exp1_hyper/step_50000.pt --seed 0
```

**命令行参数**：
- `--preset {easy,medium,hard, duo,duo_basegen, duo_film_lora,duo_film_lora_fc2,duo_base_lora, medium_film_lora,medium_film_lora_fc2,medium_base_lora}`：难度/诊断/gen_scope 档（medium 是主对照；duo 族与 *_lora 族见 §11）
- `--no_collect_planner`：**仅调试**——采集关规划器会触发 ln(A) 自蒸馏退化不动点（Ch5.9.1b），训练采集默认 planner-on
- `--variant`：目前 Pkg-05 仅 `hyper` 可用；`oracle_only/infer_only` 因 TrainConfig 约束（0<s1<s2<1）留待 Pkg-08；`baseline_*` 留待 Pkg-06（会 `NotImplementedError`）
- `--max_steps INT`：覆盖 `train.max_train_steps`
- `--override "section.field=value"`（可重复）：任意 cfg 覆盖，例：
  - `--override "train.lr=3e-4"` `--override "train.batch_size=128"`
  - `--override "train.use_crn=False"`（消融 CRN）/ `--override "train.use_coord_desc=False"`（消融坐标下降）
  - `--override "train.stratified_sampling=False"`（消融分层采样）
  - `--override "env.type_assignment=[0,0,0,0]"`（全 α，类型扫描）
- `--resume_from PATH` / `--seed INT` / `--log_dir PATH`(TensorBoard) / `--ckpt_dir PATH`

**关键默认（medium）**：N=4(2α+2β), A=6, T_max=200, latent=64, ctx_aug=80,
batch=256, unroll_K=5, n_step=5, γ=0.95, lr=1e-4(warmup 5k→cosine), ema_τ=0.99,
buffer=5000(min 1000), 损失权重 π=1.0/v=0.25/r=3.0/consist=0.5/belief=1.0。

**产物**：TensorBoard 日志 → `--log_dir`；checkpoint 每 10000 步 → `--ckpt_dir/step_*.pt`。
日志标量含 `total/main/belief/policy/value/reward/consist/belief_c/belief_opp/belief_div/lambda_b/lr`。

---

## 7. 实验/消融建议（与 cfg 开关对应）
- **主线 vs 课程边界**：`--override "train.curriculum_stage_1_end_frac=0.1"` 等（注意需满足 0<s1<s2<1）。
- **2×2 规划消融**：`use_crn` × `use_coord_desc`（CLAUDE.md 强调 CRN 是载荷算法，关掉会让 π_mve 退化）。
- **类型扫描**：`env.type_assignment`（全 α / 全 β / 混合）。
- **采样消融**：`train.stratified_sampling`。
- 验证脚本：`python scripts/run_test_checklist_pkg4_5.py`（Pkg-04/05）、`python scripts/run_test_checklist.py`（Pkg-01~03）。

---

## 8. MVE Planner 细粒度数据流（`planning/mve_planner.py`）

作用：在世界模型里做**前瞻规划**，为每个 agent 产出搜索策略 `π_mve`（作为 worker 选动作的依据 +
训练时 policy 的监督目标）。核心是 **CRN（共同随机数）+ 坐标下降**。
（维度示例 medium：N=4, A=6, S=mve_samples=50, `spa=S//A=8`, `M=A*spa=48`, K=mve_depth=5）

```
输入  root_s (B,64)，cap{i:(B,4)}，belief{i:((B,),(B,N-1,2))}，c_t (B,)
  │  dict→packed: cap(B,N,4), c_hat(B,N), z_hat(B,N,N-1,2)
  │  gen ← crn_rng（可复现采样）；agent_order ← crn_rng.permutation(N)（use_coord_desc=False 则 0..N-1）
  ▼
外层：坐标下降  for j in agent_order:      （逐个 agent 优化；已优化 agent 从其 π_mve 采样=协调）
  │
  ├ Phase 1  CRN 预采样（场景粒度 B*spa）
  │    s_scenarios = root_s ⊗ spa                      # (B*spa, 64)
  │    set_context_objective(c_t ⊗ spa)                # θ_state for B*spa
  │    for i≠j:  step0[i] ← 采样一次 (B*spa,)           # 已优化→π_mve[i]，否则→policy 网络
  │
  ├ Phase 2  展开候选（BM=B*spa*A；布局＝场景外·候选内 → 关键!）
  │    s_exp = s_scenarios ⊗ A                          # (B*M, 64)
  │    set_context_objective(c_t ⊗ spa*A)               # θ_state for B*M
  │    first_action_j = [0..A-1] 重复 B*spa  (B*M,)      # agent j 枚举的候选首动作
  │    step0_expanded[i] = step0[i] ⊗ A      (B*M,)      # ★其他 agent 跨 A 候选【共享】=CRN
  │
  ├ Phase 3  K 步 rollout（折扣累积 agent j 的回报）
  │    for step in K:
  │       for i in N:
  │          step0 且 i==j        → first_action_j        （枚举候选）
  │          step0 且 i≠j 且 CRN  → step0_expanded[i]      （★共享，噪声相消）
  │          其余                 → policy 网络独立采样     （状态已发散）
  │       joint(B*M,N) → onehot(B*M,N*A)
  │       s' = model.transition(s, onehot)                （客观 θ_state，复用）
  │       set_context_subjective(j) → r_j = inv_scalar(predict_reward)   （主观，只为 j）
  │       cum_return_j += γ^step · r_j ;  s = s'
  │
  ├ Phase 3b  终值   set_context_subjective(j) → v_j = inv_scalar(predict);  cum_return_j += γ^K · v_j
  │
  └ Phase 4  CRN 聚合
       cum_return_j (B*M,) → reshape (B, spa, A) → mean over spa → Q (B,A)   # 场景维平均=蒙特卡洛去噪
       Q 标准化(去均值/标准差) → softmax(Q / temperature) → π_mve[:, j]  (B,A)

输出  π_mve (B, N, A)
```

**为什么 CRN 是载荷算法（勿删）**：评估 agent j 的 A 个候选时，候选之间**唯一应有的差异**是「j 的首动作不同」。
若其他 agent 的动作每个候选各自随机采，巨大噪声淹没信号（SNR≈0.02 → softmax 退化为均匀分布）；
CRN 让其他 agent 的 step-0 动作**每场景采一次、在 A 个候选间共享** → 候选间噪声相消，信号显现
（布局 `(B, spa, A)`，最后对 `spa` 维平均做蒙特卡洛去噪）。

- **坐标下降**：随机 agent 顺序逐个优化；已优化 agent 从其 `π_mve` 采样（协调），未优化从先验策略采样。
- **确定性（C5-P1）**：`agent_order` 与动作采样都由 `self.crn_rng` 派生 → 重置 seed 输出复现；
  `crn_rng` 跨 episode 持续（worker 持有同一 planner 实例）。
- **开关（消融）**：`use_crn=False` → step-0 不共享（独立采样，忠实的 −CRN 格子）；`use_coord_desc=False` → **仅**固定顺序 `0..N-1`（坐标下降本体仍运行——已优化 agent 仍按 π_mve 行动；**不是** Joint 联合枚举,见 §10 缺口与 Ch6.7 三轴重定义）。
- reward/value 经 `inverse_scalar_transform` 从标量变换空间还原后再累积折扣回报。

---

## 9. 双路径梯度细粒度数据流（`loss_composition.py` + `grad_gating.py`）

一次 `BeliefNet.forward` 的输出**同时**喂两条 autograd 路径，最后 **一次 backward** 反传：

```
batch.obs (B,K+1,N,obs_dim)
   │  一次前向（带梯度，两路共用，数值一致）
   ▼
BeliefNet.forward → hidden(B,K+1,N,128), ĉ_pred(B,K+1,N), ẑ_pred(B,K+1,N,N-1,2)
   │                                              │
   │ 【路径A：L_belief，始终训练，不受门控】          │ 【路径B：main，经 set_context_subjective】
   ▼                                              ▼
belief_loss(ĉ_pred, ẑ_pred, hidden, c_true, types)   课程混合: z_main = w·oracle + (1-w)·ẑ_pred
 = w_c·L_c + w_opp·L_opp + w_div·L_div                         c_main = ĉ_pred  (c 不注入 oracle)
   │  直接反传                                         │
   ▼                                              set_context_subjective(agent, cap, (c_main, z_main))
GRU + head_c + head_opp  ✅梯度                       │   ┌ grad_gating.apply_raw(ĉ,ẑ, step)      （切断→GRU/heads）
                                                   │   └ grad_gating.apply_ctx(ctx_aug, step, [48:80]) （切断→BeliefEncoder 投影）
                                                   │       step<5000：两层都 detach → main 不回流 BeliefNet
                                                   │       step≥5000：透传
                                                   ▼
                                          ctx_aug(80)=[c_ctx16 | role32 | belief32]
                                                   │  ┌ hyper_rew(ctx_aug)          → θ_rew   （reward 路，回流 ctx_aug→belief）
                                                   │  └ hyper_pred(ctx_aug.detach()) → θ_pred  （policy/value 路，★detach_pred_context 不回流）
                                                   ▼
                                          RewardHead / PredictionNet → L_reward + L_policy + L_value (+ BYOL L_consist)
                                                   │  仅 reward 路、且 step≥5000 时
                                                   ▼  reward → hyper_rew → ctx_aug[48:80] → ẑ_pred/ĉ_pred
                                          小量回流 BeliefNet

L_total = L_main + λ_b · L_belief   →   L_total.backward()   （单次 backward 同时反传 A、B 两路）
```

**梯度门控真值表**（测试时 monkeypatch `mixing=0`，隔离「门控阈值」与「课程阶段」两套 step 机制）：

| step | L_belief → BeliefNet | main → BeliefNet | 谁在训练 BeliefNet |
|------|----------------------|------------------|--------------------|
| `< 5000`（门控预热）| ✅ L_c/L_opp/L_div | ❌ 双层 detach | 仅 L_belief（专项预热）|
| `≥ 5000` | ✅ | ✅ 但**仅 reward 路**（policy/value 因 `detach_pred_context=True` 被切）| L_belief + reward 小量 |

> **[v4-opt] 真值表脚注**：上表第二行的「仅 reward 路」以默认 `detach_pred_context=True` 为前提。**duo 族与 *_lora 族预设取 False**（理由：film_head 系的功能网主干已是稳定共享 SGD 网络，放开 policy/value 梯度通达上下文编码器以解饿，见 `presets/duo.py` docstring 与 Ch4.3.5）——此时 step≥5000 后 policy/value 路也回流 BeliefNet/编码器。该开关已成为 **gen_scope 依赖**的选择。

**关键设计点**：
- 一次 `BeliefNet.forward` 供两路共用，不重复前向（数值一致、省算）。
- `L_belief` 用**预测** `ẑ_pred`（而非 oracle）计算 → 即便 Stage 1 注入 oracle，`head_opp` 仍被训练。
- **禁止反模式**：先 `detach` 再算 `L_belief`；或先 `set_context_subjective` 再算 `L_belief`（那时张量已被 model 内部 detach）。
- 单测验证（compose 级，与 Pkg-04 `grad_gating` 对齐）：`pre-5k` main 路对 BeliefNet 梯度 **== 0**；`post-5k` **> 0**。

---

## 10. 已知缺口（开始实验前知悉）[v4-opt 2026-06 更新]
- ~~**超网络参数量超预算**~~：**已由输出层 LoRA 解决**——film_head+LoRA(r=32) 把 HyperNet 参数从 3.09M 压到 ~694k（§11），无需 chunked hypernet。
- **变体未接全**：`oracle_only/infer_only`(Pkg-08；curriculum.py 已留退化边界钩子) 与 `baseline_*`(Pkg-06) 尚未实现——**Pkg-06 是决策点 1 的阻塞项**。
- **c_t 隐藏模式缺失**：Ch3.7 模式 B（观测中 c_t 替换为常数，BeliefNet ĉ 头的核心检验场景）无环境开关；当前可见-c 下 ĉ 是恒等读出（复审 M12，Pkg-02 待补 `c_visible`）。
- **Joint 联合枚举缺失**：消融 4 的"−协调下降"格子无代码路径；现 `use_coord_desc=False` 仅取消顺序随机化（复审 M8,Ch6.7 三轴重定义,Pkg-08 待实现 Easy N=2 Joint 模式）。
- 这些都不影响 `--variant hyper` 主线训练。

---

## 11. 优化阶段总览（`e5a9e17..dc5bbcd`,2026-06）[v4-opt 新增]

> 时间线与理论影响见 Roadmap Part 3.5;事实底稿见 `docs/Review_v4_TheoryAudit_2026-06.md`。本节是代码侧速查。

### 11.1 两个实测失败模式与修复

| 失败 | 指纹 | 修复 |
|---|---|---|
| 采集无规划信号 → 策略熵钉死 ln(A)=1.79 | `diag/pi_mve_entropy` 不动 | planner-on 采集默认(`worker.collect_episode(use_planner=True)`);`--no_collect_planner` 仅调试 |
| FULL 全量生成 → 角色坍缩 | `diag/cos_pred_cross` 0.61→0.998 | `hyper_gen_scope` 部分生成 + 分组 RMS 归一(见下) |

### 11.2 gen_scope 四档(`ModelConfig.hyper_gen_scope`)

| 档 | fc1 | fc2 | 头 | HyperNet 参数(medium,+LoRA r=32) |
|---|---|---|---|---|
| `full`(legacy 默认) | 全生成 | 全生成 | 生成 | ~3.09M(无 LoRA) |
| `film_head` | SGD 权重+生成 FiLM | 同左 | 生成 | **~694k** |
| `lora_fc2` | SGD+FiLM | SGD+FiLM+**ΔW=B_f A_f**(r=8) | 生成 | **~896k** |
| `base_gen` | 纯 SGD 基座(无 FiLM) | 全生成 | 生成 | ~2.30M |

配套机制(`models/hyper_network.py` + `functional_nets.py`):
- **分组 RMS 归一**(`output_groups=[film 段, weight 段]`):整向量 L2 会把 FiLM γ 稀释到 scale/√dim(≈4e-3,调制失效);分组 RMS 使每生成元 ≈ output_scale → 部分生成预设三路 scale 统一 0.1;
- **输出层 LoRA**(`hyper_output_rank=32`,三路统一):A 正交、B small_init(std=0.01,**不可为 0**——分组 RMS 的 1e-8 下限会在 step-0 产生 ~1e4 梯度尖峰);
- **ΔW 尺度律**(lora_fc2 守门):ΔW ≈ output_scale²·√r;`ModelConfig.__post_init__` 强制 scale ≥ 0.05、base_gen 禁 lora_fc2、LoRA×share_subjective_trunk 抛 NotImplementedError;
- **share_subjective_trunk**:hyper_rew/pred 合一 trunk 双头(detach 在 trunk 输出处,语义更强);LoRA 线暂弃该轴。

### 11.3 预设矩阵与 sweep(决策门 0,在飞)

`scripts/run_lora_experiments.py`:3 建模情形 × 2 环境 = 6 runs;GPU 池默认 {2,3,4}(**0/1 政策禁用**),每 run 单卡 `CUDA_VISIBLE_DEVICES` 钉卡;落盘 `<env>/<model>[/<gen_scope>]/tb|ckpt|train.log`。判定准则(预注册):熵离开 ln A、cos_pred_cross<0.95 为硬门槛,过门槛 cell 按 medium 福利选 thesis-default gen_scope。duo 族(random_walk)结论只作机制存活性证据,断言 B′ 正式判定落在 medium static(复审 Q6 决议)。

### 11.4 诊断设施

- TB 命名空间:`diag_*`→`diag/`,`*_raw`→`loss_raw/`,其余→`loss/`(train_main.py 路由);
- 关键诊断:`diag/pi_mve_entropy`(对照 ln A)、`diag/pi_pred_entropy`、`diag/cos_{pred,rew}_{cross,same}`(角色分化;断言 A 在线证据);
- 离线探针:`scripts/diagnose_mve.py`(checkpoint → 逐 agent returns_per_action / q_normalized)。

### 11.5 2agent 三连跑诊断与修复(2026-06-11)[v4-opt 2026-06b]

> 数据源:`2agent/{basegen, film_head/{on,off}}` 三 run(seed 0,~11k 步)。诊断结论与修复一并落码。

**实测发现:**

| 发现 | 证据 | 修复 |
|---|---|---|
| 吞吐瓶颈:0.08 train-steps/s(1M 步 ≈ 135 天) | event 时间戳;采集 B=1 → 每 env step ~70 次小 GPU 调用 | **向量化采集**:`Worker.collect_episodes` 以 n_envs=episodes_per_iter(8)lockstep 推进,planner 批量 B=8 |
| 零可观测:无回报、无评估(22 个 TB 标签全为损失/诊断) | TB 标签清单 | **`eval/*` + `collect/*` + `perf/*` 三族**(见下) |
| policy loss U 形(~6000 步谷底后回升),H_pi_mve 同步回升 | 三 run 一致 | 三个混杂因素分别处理(下三行) |
| warmup 1000 episodes 存自蒸馏 π 目标(planner-off ⇒ pi_mve=自身先验),~4000 步才被 FIFO 逐出 | worker.py 旧 95-96 行 | **planner_on 掩蔽**:buffer per-episode 元数据 + 策略 CE masked mean |
| belief 闸门(5000)与 LR warmup 终点(5000)重合 ⇒ 双 regime change 混杂 | grad_gating.py / train_config | duo 族预设 `belief_grad_gating_steps=1e9`(N=2 belief 损失本就平凡;L_belief 仍训 BeliefNet) |
| z-score 把近等候选回报的采样噪声放大为单位尺度目标 | mve_planner Phase 4 仅 1e-8 下限 | **`mve_qstd_floor`(默认 0.01)**:q_std 低于阈值的行回退均匀目标 |
| `cos_*_same` N=2 恒 NaN(无同类型对) | loss_composition `_role_cosine` | TB writer 跳过 NaN 标签(语义本就正确,纯日志卫生) |
| `cos_rew_cross`≈0.98(奖励头几乎不随上下文分化;FS 项 ~5% 奖励尺度被淹没) | basegen/film_on TB | 暂记录;靠新评估族判断是否实际损害回报 |

**新 TB 标签族:**
- `eval/{prior,planner}/return_{total,alpha,beta}[_c{0.2,0.5,0.8}]`、`eval/planner_prior_gap`、`eval/planner/{pi_mve_entropy,q_std,uniform_frac}` —— 每 `eval.evaluate_freq`(默认 1000)步一次,双模式(prior=蒸馏策略 argmax π̂ / planner=真实智能体 argmax π_mve)确定性评估,static-c 网格 + 固定种子 + 固定 planner CRN(跨评估可比);best ckpt 按 `planner/return_total` 存 `best.pt`;
- `collect/{return_total,return_alpha,return_beta,epsilon,H_pi_mve_fresh,q_std,q_gap,uniform_frac}` —— 采集侧滚动均值(近 32 episodes);
- `perf/{collect_sec_per_iter,train_sec_per_iter,env_steps_per_sec}`;`diag/target_age_steps`(采样目标陈旧度)。

**验证脚本:**`scripts/test_vectorized_worker.py`(B=1/B=8、确定性重复、掩蔽路径)、`scripts/test_eval_runner.py`(标签集、有限性、确定性)。

**采集后修订 1**:确定性评估 argmax 在均匀概率行(噪声护栏回退或自然平局)上会恒选 action 0(ResourceCommons 中 = NOOP) ⇒ planner-mode eval 系统性低估。修复:`Worker.collect_episodes` 新增 `tiebreak_rng` 参数,均匀行用随机化 argmax(`np.random.choice(flatnonzero(p == p.max()))`);`run_eval` 以固定种子 1234567 注入(CRN 跨评估)。`planner_prior_gap` 改为按 c 网格逐点取均值再做差,消除两模式 episode 数不对称带来的偏置。
