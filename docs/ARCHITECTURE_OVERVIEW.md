# Hyper-MuZero v5 代码主视图（Pkg-01 ~ Pkg-09：动态角色关系版）

> 面向「开始实验」的导读：先看 §1 核心思想 → §2 分层架构 → §5 怎么训练 → §8 规划器。
> 权威规格见 `sdd/pkg-0X-*/`（v5 变更集中在 `sdd/pkg-09-dynamic-relations/design.md`）。
> 本文件是代码侧的速查地图。v4（resource_commons / c_t / 类型系统）已在 2026-07 的
> Pkg-09 重构中整体退役，历史记录见 git history 与 `docs/中期报告_old.md`。

---

## 1. 一句话 + 核心思想（关系 = 角色）

研究命题：**动态角色关系下的不完全信息博弈**。角色不再是离散类型（α/β），而是
**关系矩阵 W(g) 的一行**：regime g 从有限族 G 中抽取，`w_ij` 表示 agent i 对 j 的
福利权重。agent 只知道**自己的行** w_i·（私有信息），g 与他人的行需要**推断**。

```
关系型奖励（Pkg-09 唯一公式）:
    R_i = (u_i + Σ_{j≠i} w_ij·u_j) / (1 + Σ_{j≠i}|w_ij|) − ε·1[moved_i]

C_aug = [ role(32) | belief(32) ]   = 64 维
          身份(id8+row24)  信念(regime 后验 ĝ)
              │                │
              ▼                ▼
          hyper_rew        hyper_pred          (DualHyperNetwork：仅主观路)
              │                │
            θ_rew^i          θ_pred^i            (动态生成权重)
              │                │
         RewardHead      PredictionNet          (功能网)
          s,a → r           s → π, v
        主观(每个agent)    主观(每个agent)

         TransitionNet  s,a_joint → s'          (普通共享 nn.Module，SGD 训练)
              客观：所有 agent 权重共享 = 视角不变性
```

- **客观流**：物理规则固定（常数再生率 α），转移网是**普通共享网络**——权重共享本身
  就保证「世界演化与谁在看无关」。（v4 的 hyper_trans/c_ctx 已删除；未来做环境迁移时
  可重新给客观路挂条件输入，是预留钩子而非现役代码。）
- **主观流**：`hyper_rew / hyper_pred` 吃 64 维 C_aug，per-agent 生成 θ → 奖励、
  策略、价值随「我是谁（row）+ 我认为局势是什么（belief over g）」变化。

**Regime 动力学**：reset 时 `g_0 ~ ρ`（均匀）；每步以概率 p 切换（`κ` 均匀重采样）。
`p=0` = 研究点 1（贝叶斯/Harsanyi 博弈，局内静态）；`p>0` = 研究点 2（隐 Markov
切换博弈）。一套 kernel 两个研究点，由 `EnvConfig.regime_switch_prob` 控制。

---

## 2. 分层架构与依赖

```
Pkg-01/09 Schema + 配置          hyper_mve/schemas/ , hyper_mve/configs/
   │  relation.py(Regime/RegimeFamily/关系奖励) / TimeStepRecord / V4Config
   ▼
Pkg-09 RelationCommons 环境      hyper_mve/envs/relation_commons/
   │  gym.Env：reset(options={"g":k})/step/info(Public/Oracle/EvalOnly 三段)
   ▼
Pkg-03/09 上下文编码 + 信念网     hyper_mve/models/  (belief_*, *_encoder, tri_context_encoder)
   │  BeliefNet(GRU→ĝ) + TriContextEncoder(→ctx_aug 64) + belief_losses(L_regime/L_div)
   ▼
Pkg-04/09 超网络 + HyperMuZero    hyper_mve/models/  (hyper_*, transition_net, functional_nets, ...)
   │  HyperMuZeroModel(6 API) + DualHyperNetwork(仅主观) + TransitionNet + RepNet + 梯度门控
   ▼
Pkg-05/08 训练器 & Worker & 套件  hyper_mve/training/ , planning/ , scripts/ , experiments/
      MuZeroTrainer / Worker / EpisodeReplayBuffer / CurriculumScheduler /
      compose_total_loss / MVEPlanner / train_main.py / run_suite / sweep
```

---

## 3. 各包模块职责速查

### Schema + 配置（`schemas/` + `configs/`）
| 文件 | 作用 |
|------|------|
| `schemas/relation.py` | `Regime(id,name,W)` / `RegimeFamily` / `build_g2·g4·g4_ext(λ)` / `sample_initial_regime` / `step_regime` / **`compute_relational_rewards`**（论文公式的唯一实现）|
| `schemas/buffer_record.py` | `TimeStepRecord` v5（10 字段：o,a,r,pi_mve,v,row,g_hat,g,t,done）|
| `schemas/observation.py` | `RelationObservationLayout.total_dim(N,K) = 5+3K+10(N−1)`（rel_duo N=2,K=8 → 39）|
| `configs/env_config.py` | `relation_family/relation_intensity/regime_prior/regime_switch_prob/train_regime_ids/alpha` |
| `configs/presets/rel_duo.py` | `rel_duo`（N=2, g2, p=0, 200K 主对照）+ `rel_duo_holdout`（train_regime_ids=(0,1,4)，留出非对称 regime 做零样本）|

**g2 族（N=2, |G|=5）**：0 mutual_coop(+λ,+λ) / 1 mutual_comp(−λ,−λ) /
2 asym_exploit / 3 asym_exploited / 4 neutral(0,0)。g4 与 g4_ext（N=4，5/9 个 regime）同文件。

### RelationCommons 环境（`envs/relation_commons/`）
| 文件 | 作用 |
|------|------|
| `env.py` | `RelationCommonsEnv`：reset 时 `g_0~ρ`（受 `train_regime_ids` 限制；`options={"g":k}` 可钉死）；step = 确定性移动 + fair-share 采集 + 常数 α 再生 + `step_regime` + 关系奖励 |
| `observations.py` | 五块观测：self(4) \| resource(3K) \| neighbor(9(N−1)) \| global(1) \| **row(N−1)=自己的 w_i·** |
| `dynamics.py` / `spawn.py` / `rewards.py` / `state.py` | fair-share 采集（复用）/ 均匀重生成 / 关系奖励薄包装 / `W(N,N)+g` 状态容器 |

> **info 三段分组（载荷契约，`_info_schema_version="v5.0"`）**：`Public`(harvests/step_idx)
> 模型可用；`Oracle`(g_true/rows) **只给训练监督，绝不进 model.forward**（worker 遵守
> row-i-only-for-agent-i 纪律）；`EvalOnly`(resource_state) 仅评估可视化。
> 外部基线经 `envs/adapters/pettingzoo_wrapper.py` 消费，`_FORBIDDEN_INFO_KEYS` 运行时护栏。

### 上下文编码 + 信念网（`models/`）
| 文件 | 作用 |
|------|------|
| `belief_net.py` | `BeliefNet`：共享 GRU + `head_regime`（|G| 路 softmax）→ 每 agent 对 g 的后验 ĝ。`step()` 在线单步 / `forward()` 序列 |
| `belief_losses.py` | `l_regime`(oracle-g CE) / `l_div`(hinge 方差防坍缩) / `belief_loss` / `build_oracle_g_seq`(one-hot) |
| `role_encoder.py` | `RoleEncoder`：id_emb(8) + row_mlp(N−1→16→24) → `role`(32)（Self-Info：只用自己的 row）|
| `belief_encoder.py` | `BeliefEncoder`：`Linear(|G|,32)+ReLU` → `belief_vec`(32) |
| `tri_context_encoder.py` | 拼两路 → `ctx_aug`(64)，含分路 LayerNorm |

### 超网络 + HyperMuZero 模型（`models/`）
| 文件 | 作用 |
|------|------|
| `hyper_muzero_model.py` | `HyperMuZeroModel`：**6 个对外 API**（见 §4）+ API lock 注释块 |
| `hyper_network.py` | `DualHyperNetwork`：`forward_subjective(ctx_aug)→(θ_rew,θ_pred)`；`detach_pred_context=True` 时只让 reward 梯度回流 ctx_aug（v4 的 `forward_trans` 已删）|
| `transition_net.py` | **`TransitionNet`：普通共享残差网** `(s, a_joint_onehot)→s'`，SGD 训练 |
| `functional_nets.py` | `FunctionalRewardHead` / `FunctionalPredictionNet`（权重外部注入）|
| `representation_net.py` | `RepresentationNet`(obs→s) + `Projector`(BYOL 一致性) |
| `grad_gating.py` | 前 `belief_grad_gating_steps`(=5000) 步把 belief 梯度 detach，BeliefNet 先由 L_belief 专项预热 |

### 训练器 & Worker & 评估（`training/` + `planning/` + `eval/` + `scripts/`）
| 文件 | 作用 |
|------|------|
| `training/muzero_trainer.py` | 单一 `train_step` + EMA target + warmup_cosine LR + n-step return + checkpoint |
| `training/worker.py` | 向量化采集：`BeliefNet.step` 在线推断 + 6-API 选动作 + MVE planner → `TimeStepRecord(g,row,g_hat)` |
| `training/episode_buffer.py` | TimeStepRecord 容器 + **按 regime 分层采样** |
| `training/curriculum.py` | 3 stage oracle-g 混合权重 + `lambda_b` |
| `training/loss_composition.py` | main loss + L_belief 双路径 backward；`g_main = w·onehot(g)+(1−w)·ĝ` 课程混合 |
| `training/evaluation.py` | 训练内周期评估：**逐 regime**（`reset(options={"g":gid})`）双模式（prior/planner）+ `belief/regime_accuracy` |
| `planning/mve_planner.py` | CRN + 坐标下降 + 主观 θ-cache，产出 `π_mve` |
| `eval/eval_report.py` | 冻结 `EvalReport`（28 字段，`schema_version="rel-v1"`：per-regime returns/sem/episodes + regime_accuracy/nll + 4 福利指标）|
| `eval/game_metrics.py` | **博弈论指标（离线）**：NashConv（DQN 最优反应，下界）+ 经验 PoA（`Eff(g)=W_phys/Ŵ*`）；CLI `scripts/eval_game_metrics.py` |
| `scripts/train_main.py` | 统一训练入口（`--preset/--variant/--override/...`）|
| `scripts/run_suite.py` | 套件驱动（`experiments/suite/manifest.yaml`：rel_gate_duo / rel_zero_shot_duo）|

---

## 4. HyperMuZeroModel 的 6 个 API（v5：`set_context_objective` 已删除）

```python
model.update_step(global_step)                        # 1) 驱动梯度门控阈值
model.set_context_subjective(agent_id, row_i, belief) # 2) row_i (B,N−1)、belief (B,|G|) → θ_rew^i/θ_pred^i
s    = model.encode(obs)                              # 3) RepNet：obs → 潜状态 s
s2   = model.transition(s, action_onehot)             # 4) 客观转移（普通共享网，无需上下文）
r    = model.predict_reward(s, action_onehot)         # 5) 主观奖励（当前 agent 的 θ_rew）
pi,v = model.predict(s)                               # 6) 主观策略+价值（当前 agent 的 θ_pred）
```

**调用顺序（硬约束）**：`update_step → for k in N: set_context_subjective(k) + 前向`。
v4 的 `set_context_objective(c_t)` 及其一切调用点已删除（迁移测试
`test_no_v4_objective_context_calls` 把关）。reward/value 仍在 MuZero
`scalar_transform` 空间，算 loss 时目标同变换。

---

## 5. 训练数据流（一个 iteration）

```
[采集] Worker.collect_episodes(envs)           # 向量化 lockstep，不反传
   每步：obs → belief_net.step → ĝ (N, |G|)
         encode(obs)=s
         for k in N: set_context_subjective(k, row_k, ĝ_k) → predict
         (默认) MVEPlanner.sample_mve_plan → π_mve
         ε-greedy 选 a → env.step → TimeStepRecord(o,a,r,π_mve,v,row,ĝ,g,t,done)
         ↓  row 来自 oracle info["rows"]，但 agent k 只拿第 k 行（纪律同 v4 的 cap）
[存储] buffer.store_episode(records)                # FIFO + 按 regime 分层

[训练] buffer.sample_batch(B,K) → batch             # (B, K+1, N, ...)
   trainer.train_step(batch, step):
     compose_total_loss:
       ① belief_net.forward(obs) → (hidden, ĝ_pred)            [带梯度]
       ② L_belief = λ_regime·L_regime(ĝ_pred, g_true) + λ_div·L_div   [始终训练 BeliefNet]
       ③ 课程混合：w=oracle_mixing(step)；g_main = w·onehot(g_true) + (1−w)·ĝ_pred
       ④ K 步展开，对 N 个 agent：set_context_subjective(k, row_k, g_main_k)
              → 累积 L_policy(CE vs π_mve) + L_value(n-step) + L_reward + L_consist(BYOL)
       ⑤ L_total = L_main + λ_b·L_belief
     backward → clip → optimizer.step → EMA target
```

**双路径梯度（不变的关键设计）**：`L_belief` 始终直接训练 BeliefNet；main loss 经
`set_context_subjective` 进入模型，前 5000 步被 `grad_gating` 双层 detach，之后仅
reward 路回流（`detach_pred_context=True` 时）。禁止反模式：先 detach 再算 L_belief。

**课程 3 阶段（rel_duo，max=200K）**：Stage 1 Pure Oracle（0–30%，w=1：主路用真值 g）
→ Stage 2 Anneal（30–70%，线性 1→0）→ Stage 3 Pure Inference（70–100%，w=0）。
belief 是承载研究点的路（隐 regime 推断），rel_duo **不再**沿用 duo 的
`belief_grad_gating_steps=1e9`，恢复默认 5000。

---

## 6. 怎么开始训练

> Linux 训练机、`lightzero` 环境、仓库根目录运行。GPU 0/1 政策禁用，测试用 2,3。

```bash
cd /home/data/zhengwenbo/hyper_mve
conda activate lightzero

# 0) 最小冒烟（几分钟）
python hyper_mve/scripts/train_main.py --preset rel_duo --variant hyper \
    --max_steps 500 \
    --override "train.min_buffer_size=12" --override "train.batch_size=32" \
    --ckpt_dir checkpoints/smoke --seed 0

# 1) 正式训练（rel_duo 主配置，200K 步）
python hyper_mve/scripts/train_main.py --preset rel_duo --variant hyper \
    --max_steps 200000 --seed 0 \
    --log_dir runs/rel_duo_hyper --ckpt_dir checkpoints/rel_duo_hyper

# 2) 零样本 regime 留出（训练只见 {coop, comp, neutral}）
python hyper_mve/scripts/train_main.py --preset rel_duo_holdout --variant hyper ...

# 3) 套件（决策点 1 + 2）
python -m hyper_mve.scripts.run_suite --list
python -m hyper_mve.scripts.run_suite --only rel_gate_duo

# 4) 博弈论指标（离线，冻结 checkpoint 上）
python -m hyper_mve.scripts.eval_game_metrics --ckpt checkpoints/.../best.pt \
    --preset rel_duo --out runs/rel_duo_hyper/game_metrics.json
```

**常用 override**：
- `--override "env.regime_switch_prob=0.05"`：切到研究点 2（局内非平稳）
- `--override "env.train_regime_ids=[0,1,4]"`：regime 留出
- `--override "train.use_crn=False"` / `"train.use_coord_desc=False"`：规划器消融
- `--override "train.stratified_sampling=False"`：分层采样消融

**关键默认（rel_duo）**：N=2, A=6, T_max=100, K=8 资源, latent=64, **ctx_aug=64**,
`hyper_gen_scope="film_head"`, rew/pred `output_scale_init=0.1`, `mve_temperature=0.5`,
损失权重 π/v/r/consist/belief = 1.0/0.25/3.0/0.5/1.0。

---

## 7. 评估体系（三层）

1. **训练内周期评估**（`training/evaluation.py`，每 `eval.evaluate_freq` 步）：
   逐 regime 钉死 `{"g": gid}` 的确定性双模式（prior=蒸馏 π̂ argmax / planner=π_mve
   argmax）episodes；TB 标签 `eval/{prior,planner}/...` 按 regime 细分 +
   `belief/regime_accuracy`（|G2|=5 时机会水平 0.2）+ 4 个福利指标
   （welfare_physical / sustainability / fairness / tragedy_index）。
2. **冻结 EvalReport**（`eval/eval_report.py`，`rel-v1` 28 字段）：套件/sweep 的每 run
   产物；per-regime returns + zero-shot seen/unseen 切分（由 `train_regime_ids` 定义）。
   外部基线经统一 `evaluate(env_fn, regime_grid, episodes)` 合同产出同 schema。
3. **离线博弈论指标**（`eval/game_metrics.py`，不进 EvalReport——BR 训练太贵）：
   - **NashConv(g)** = Σ_i max(0, V_i(BR_i, π_{−i}) − V_i(π))：冻结策略走蒸馏先验，
     每 (agent, regime) 训一个 double-DQN 最优反应（同信息条件：见 row 不见 g）。
     BR 是近似 ⇒ 报告值是真实可利用度的**下界**，预算随值一并报告。
   - **经验 PoA**：`Eff(g) = W_phys(π,g) / Ŵ*`，Ŵ* 为合作最优参考福利（all_coop
     regime 训出的策略；物理与 regime 无关 ⇒ 一个 Ŵ* 服务所有 g）。Ŵ* 是估计，
     Eff 可 >1，报告其来历。

---

## 8. MVE Planner 细粒度数据流（`planning/mve_planner.py`）

作用不变：在世界模型里做前瞻规划，为每个 agent 产出 `π_mve`（worker 选动作依据 +
policy 蒸馏目标）。核心仍是 **CRN（共同随机数）+ 坐标下降**。
（维度示例 rel_duo：N=2, A=6, S=mve_samples, `spa=S//A`, `M=A*spa`, K=mve_depth）

v5 变化（仅接口/缓存，算法不动）：
- 输入 `cap{i}` → `row{i:(B,N−1)}`；`belief{i}` = regime 后验张量 `(B,|G|)`；`c_t` 删除。
- **θ-cache 只剩主观路**：per-(agent, 上下文) 缓存 θ_rew/θ_pred，rollout/坐标下降内
  复用（客观转移是普通网络，天然无需缓存）。
- 客观转移在 K 步 rollout 内直接 `model.transition`；主观 reward/value 仅为当前
  优化的 agent j 生成。
- `p>0` 时：想象 rollout 内 **W 冻结在当前信念**（文档化近似，研究点 2 落地时再议）。

CRN 仍是载荷算法（勿删）：评估 agent j 的 A 个候选时，其他 agent 的 step-0 动作
每场景采一次、跨 A 个候选共享 → 噪声相消。`(B, spa, A)` 布局对 spa 维平均去噪，
Q 标准化后 softmax(/temperature) 得 `π_mve[:, j]`；`mve_qstd_floor`(=0.01) 护栏：
q_std 过低的行回退均匀目标。确定性：`agent_order` 与采样均由 `crn_rng` 派生。

---

## 9. 双路径梯度细粒度数据流（`loss_composition.py` + `grad_gating.py`）

```
batch.obs (B,K+1,N,obs_dim)
   │  一次前向（带梯度，两路共用）
   ▼
BeliefNet.forward → hidden(B,K+1,N,128), ĝ_pred(B,K+1,N,|G|)
   │                                        │
   │ 【路径A：L_belief，始终训练】             │ 【路径B：main，经 set_context_subjective】
   ▼                                        ▼
belief_loss(ĝ_pred, hidden, g_true)          课程混合: g_main = w·onehot(g_true) + (1−w)·ĝ_pred
 = λ_regime·L_regime + λ_div·L_div            │
   │  直接反传                                 set_context_subjective(k, row_k, g_main_k)
   ▼                                          │   grad_gating: step<5000 双层 detach；≥5000 透传
GRU + head_regime  ✅梯度                      ▼
                                       ctx_aug(64)=[role32 | belief32]
                                              │  ┌ hyper_rew(ctx_aug)           → θ_rew  （回流）
                                              │  └ hyper_pred(ctx_aug.detach()) → θ_pred （detach_pred_context 不回流）
                                              ▼
                                       RewardHead / PredictionNet → L_reward + L_policy + L_value (+ L_consist)

L_total = L_main + λ_b·L_belief  →  一次 backward 同时反传 A、B 两路
```

诊断：类型基诊断（cos_pred_cross 按 α/β 分组）已被 **pairwise θ 诊断**取代
（regime/row 条件下 θ 的跨 agent 余弦）。

---

## 10. 已知缺口 / 注意事项（v5）

- **QMIX 训练在求和奖励上**：mutual_comp 里 Σ_i R_i ≡ 0（零和，至多差移动罚），
  混合 regime 训练对 QMIX 是信号饥饿的——外部基线对比时注明（冒烟测试钉 regime 0）。
- **NashConv 依赖 BR 质量**：预算固定并随值报告；是下界不是点估计。
- **研究点 2（p>0）**：env/kernel/config 从第一天就支持，但 planner 的 W-冻结近似、
  适应性/动态 regret 指标、切换检测评估都未建——留待研究点 1 结果落地后。
- **v5 消融矩阵未定义**：v4 的 ablate.py 调度器（abl1/abl4/abl6/abl7）随
  resource_commons 一并删除；v5 的消融（CRN×坐标下降、belief 门控、oracle 课程角点）
  设计好后作为普通 suite cell 走 run_suite。
- 旧 `runs/` 结果属于 v4 设计，与 v5 不可比（已接受）。

---

## 11. v4 优化阶段遗产（2026-06，仍现役的部分）

> v4 完整记录见 git history（`docs/Review_v4_TheoryAudit_2026-06.md`）。下面只列
> **在 v5 里仍然载荷**的机制：

- **gen_scope 部分生成**（`ModelConfig.hyper_gen_scope`）：`film_head`（SGD 权重 +
  生成 FiLM + 生成头，rel_duo 默认）/ `lora_fc2` / `base_gen` / `full`。配套
  分组 RMS 归一 + 输出层 LoRA（A 正交、B small_init≠0）+ ΔW 尺度律守门,全部保留——
  只是现在只作用于主观路（客观路已是普通网络，不再有 hyper_trans 档位）。
- **向量化采集**（`Worker.collect_episodes`，n_envs lockstep）+ planner-on 采集默认
  （planner-off 触发 ln(A) 自蒸馏退化不动点）+ warmup 自蒸馏目标的 planner_on 掩蔽。
- **评估三标签族** `eval/* collect/* perf/*` + 均匀行随机化 argmax tiebreak
  （固定种子 1234567）+ `mve_qstd_floor` 噪声护栏。
- **诊断离线探针** `scripts/diagnose_mve.py`（checkpoint → 逐 agent
  returns_per_action / q_normalized）。
