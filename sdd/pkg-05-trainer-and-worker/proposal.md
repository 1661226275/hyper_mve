# Pkg-05 Proposal: Trainer & Worker — v4 训练系统重写

> 配套阅读：[`README.md`](./README.md) · [`design.md`](./design.md)（**先读 proposal 再读 design**）

---

## 1. Why（为什么需要本包）

### 1.1 v4.7 训练系统与 v4 架构的根本不匹配

v4.7 训练系统是为"Oracle / Infer 双类 Hyper Model + GRU 上下文推理"设计的：

| v4.7 训练系统组件 | v4 期望 | 不匹配点 |
|-------------------|--------|----------|
| `train_step` (L203-430) | 适配 BeliefNet + 7 API | v4.7 调 `model.set_context(rule, agent_id)` 单参签名，v4 拆为 `update_step` + `set_context_objective` + `set_context_subjective` |
| `train_step_infer` (L432-598) | **整个废弃** | v4 BeliefNet 取代 GRU 推理，Infer 不再是模型类区分 |
| `worker.collect_episode` (L84-180) | 写 TimeStepRecord v4 字段 | v4.7 EpisodeData 仅 6 字段，缺 c_hat/z_hat/types/caps/delta/tau |
| `episode_buffer.EpisodeData` | 重写为 TimeStepRecord 容器 | v4.7 rule 是标量浮点，v4 c_t 同样标量但需 c_hat (N,) + z_hat (N, N-1, 2) Belief 字段 |
| `mve_planner.set_context × 4` (L71/213/219/258) | 迁移到 v4 双步分离 API | 与 trainer/worker 同步改造 |
| LR scheduler + EMA + grad clip | 保留 | v4.7 已稳定，仅需迁移到 v4 trainer |

**结论**：v4.7 训练系统**整体重构**，仅 mve_planner.py CRN 4 phase 实现层可最小化迁移。

### 1.2 v4.7 训练系统的具体 Pain Points

**Pain 1：双 train_step 的认知负担**

v4.7 Oracle / Infer 是模型类区分（OracleHyperMuZeroModel vs InferHyperMuZeroModel），分别走 train_step / train_step_infer。v4 BeliefNet 取代 Infer GRU 推理后，"Infer" 不再是模型类概念，但 v4.7 双套 train_step 会保留下来污染代码（trainer 内 if-else 判 model 类型）。

**Pain 2：Buffer 字段从根本上不兼容**

v4.7 EpisodeData：
```python
EpisodeData(obs, actions, rewards, search_policies, rule, dones, length)
# rule 是标量浮点（环境规则参数）
```

v4 TimeStepRecord（Pkg-01 spec 04 已锁定 12 字段）：
```python
TimeStepRecord(o, a, r, delta, pi_mve, v, tau, cap, c_hat, z_hat, t, done)
# 新增 delta / tau / cap / c_hat / z_hat 5 字段
# 删除 rule（拆为 c_t + types + caps + Belief 推断）
```

v4.7 buffer 完全不能存 v4 数据，重写不可避免。

**Pain 3：11 处 set_context 调用点的迁移压力**

Pkg-04 spec 08 §3 已列：
- mve_planner.py: L71 / L213 / L219 / L258（4 处）
- muzero_trainer.py: L239 / L242 / L474 / L484 / L678 / L680（6 处）
- worker.py: L125 / L128 / L143 / L145 / L149（5 处）

合计 **15 处调用点**，且每处都涉及"如何准备 cap / belief 参数"的逻辑改造（不是简单 rename）。这是 Pkg-05 实施时最大的 grep 验证压力点。

**Pain 4：课程学习的 ad-hoc 实现**

v4.7 训练脚本里 epsilon decay 和 train_step_infer 切换是写死的 if 分支。v4 三 Stage 课程（Pure Oracle / Anneal / Pure Inference）+ λ_b 加权 + oracle_z_mixing_weight 需要**结构化**模块，而非 trainer 内联。

**Pain 5：训练脚本碎片化**

v4.7 有 6 个训练脚本（train_baseline / train_oracle / train_oracle_v2 / train_infer / train_infer_v2 / test_env），每个固定一组 cfg overrides。v4 模型层统一为 HyperMuZeroModel 后，Oracle/Infer/Baseline 都应该通过同一入口 + 配置切换，避免脚本污染。

### 1.3 触发本包的论文断言压力

| 论文断言 | 对训练系统的诉求 |
|---------|------------------|
| **断言 A**（type 梯度撕裂）| trainer 必须把 α/β agent 的 type 信号通过 set_context_subjective 进入 model，hyper_rew 才能学到 per-type θ_rew^i |
| **断言 B**（belief 专项容量）| trainer 必须分离 L_belief 反向路径（不被 main loss 污染，与 Pkg-04 grad_gating 联动）|
| **断言 C**（三路必要性）| trainer 在 K-step unroll 内严格走 7 API 调用顺序（objective → subjective）才能让三路 ctx_aug 拼装正确 |
| **断言 D**（planner 双技术）| mve_planner CRN + coord descent 保留 + 加 cap/belief 输入参数 |

四个断言都依赖训练系统层做对，**Pkg-05 是断言验证的最后基础设施**。

---

## 2. What Changes（具体改动）

### 2.1 重写（4 文件）

#### 2.1.1 `hyper_mve/training/muzero_trainer.py`（v4 单一 train_step）

**关键改动**：
- 删除 `train_step_infer`（L432-598，~167 行）
- `train_step` 内部按 Pkg-04 spec 02 §2.3 调用顺序：
  ```python
  model.update_step(global_step)           # Q4
  model.set_context_objective(c_t)         # 一次
  for k in range(N):
      model.set_context_subjective(k, cap[:, k], (c_hat[:, k], z_hat[:, k]))
      # K-step unroll 内 encode / transition / predict_reward / predict
  ```
- target_model 同步迁移（EMA target 也需调 7 API）
- compute_n_step_return 保留（v4.4 EMA bootstrap）
- LR scheduler + EMA 内联（D6，与 v4.7 一致）
- loss 组装抽出到 `loss_composition.py`（D5）

**输入接口**：
```python
class MuZeroTrainer:
    def __init__(self, cfg: V4Config, model: HyperMuZeroModel, projector=None, device=None): ...
    def train_step(self, batch: dict[str, torch.Tensor], global_step: int) -> dict[str, float]: ...
    def save_checkpoint(self, path: str) -> None: ...
    def load_checkpoint(self, path: str) -> int: ...  # return global_step
```

#### 2.1.2 `hyper_mve/training/worker.py`（BeliefNet.step 接入）

**关键改动**：
- 删除 v4.7 `set_context_from_history / set_context_default`（5 处迁移）
- collect_episode 内：
  ```python
  prev_hidden = model.belief_net.init_hidden(B=1, num_agents=N)
  for t in range(T):
      obs_tensor = torch.from_numpy(obs).unsqueeze(0)
      prev_hidden, c_hat, z_hat = model.belief_net.step(obs_tensor, prev_hidden)
      
      s = model.encode(obs_tensor)
      model.set_context_objective(torch.tensor([info["c_true"]]))
      for k in range(N):
          cap_k = torch.from_numpy(info["caps"][k].to_array()).unsqueeze(0)
          model.set_context_subjective(k, cap_k, (c_hat[0, k:k+1], z_hat[0, k:k+1]))
          # predict π / v / sample action
  ```
- 写 TimeStepRecord 字段（c_hat / z_hat / types / caps / delta 完整填充）
- 单测验证 z_hat 顺序与 Pkg-01 spec 04 一致（agent_id 升序跳过 self）

#### 2.1.3 `hyper_mve/training/episode_buffer.py`（TimeStepRecord 容器）

**关键改动**：
- 废弃 `EpisodeData`（v4.7 6 字段）
- 新增 `EpisodeReplayBuffer.store_episode(records: list[TimeStepRecord])`
- 新增 `EpisodeReplayBuffer.sample_batch(B, K) -> dict[str, torch.Tensor]`
  - 返回 dict 字段对齐 TimeStepRecord（含 c_hat / z_hat）
  - shape 模式 (B, K+1, N, ...) 与 v4.7 兼容
- stratified_sampling 支持（cfg.train.stratified_min_per_type_frac=0.3，D9 Pkg-05 内实现）
- 内存预算 < 2 GB（Pkg-01 spec 04 §5.2）

#### 2.1.4 `hyper_mve/planning/mve_planner.py`（最小化迁移）

**关键改动**：
- CRN 4 phase 完整保留（L128-290，与 v4.7 一致）
- 4 处 set_context 迁移（Pkg-04 spec 08 §3.1 已锁定行号）
- sample_mve_plan 签名扩展（**实例方法**；cfg 移入 `__init__`，P1-3 删除顶层函数）：
  ```python
  def sample_mve_plan(
      self, model, root_s,
      cap: dict[int, torch.Tensor],                          # ← 新增 agent_id → (B, 4)
      belief: dict[int, tuple[torch.Tensor, torch.Tensor]],  # ← 新增 agent_id → (c_hat, z_hat)
      c_t: torch.Tensor = None,    # ← 新增 (B,)
  ) -> torch.Tensor: ...           # (B, N, A)
  ```
- planner 入口 set_context_objective 一次性调用（复用 θ_state）

### 2.2 新增（2 核心模块）

#### 2.2.1 `hyper_mve/training/curriculum.py`（新增）

```python
class CurriculumScheduler:
    """3 Stage 课程：Pure Oracle → Anneal → Pure Inference."""
    
    def __init__(self, cfg: V4Config): ...
    
    def stage(self, global_step: int) -> str:
        """'stage_1' / 'stage_2' / 'stage_3'"""
    
    def oracle_z_mixing_weight(self, global_step: int) -> float:
        """Stage 1: 1.0 / Stage 2: 1.0 → 0.0 linear anneal / Stage 3: 0.0"""
    
    def lambda_b(self, global_step: int) -> float:
        """L_belief 课程加权系数（与 cfg.train.w_belief 配合）"""
```

#### 2.2.2 `hyper_mve/training/loss_composition.py`（新增）

```python
def compose_total_loss(
    model: HyperMuZeroModel,
    batch: dict[str, torch.Tensor],
    scheduler: CurriculumScheduler,
    global_step: int,
    cfg: V4Config,
) -> dict[str, torch.Tensor]:
    """主 loss + L_belief 拼装.
    
    返回 dict[str, Tensor]:
        "total":        total loss (含 grad)
        "main":         w_policy·L_π + w_value·L_v + w_reward·L_r + w_consist·L_BYOL
        "belief":       L_c + L_opp + L_div
        "lambda_b":     课程加权系数（用于日志）
        各 sub-loss 标量（用于 TensorBoard）
    """
```

### 2.3 修改（11 处调用点迁移）

| 文件 | 行号 | v4.7 → v4 |
|------|------|-----------|
| mve_planner.py | L71 | set_context(rule_exp, id_i) → set_context_objective + set_context_subjective |
| mve_planner.py | L213/219/258 | 同上 |
| muzero_trainer.py | L239 | set_context(rules, agent_ids) → 拆为 objective + 循环 subjective |
| muzero_trainer.py | L242 | target_model 同步 |
| muzero_trainer.py | L474 | set_context_from_history → 删除（BeliefNet.step 替代）|
| muzero_trainer.py | L484 | set_context with inferred_rule → 拆为两步 + BeliefNet 输出 |
| muzero_trainer.py | L678/680 | 同 L239/242 模式 |
| worker.py | L125/128/143/145/149 | set_context_from_history / set_context_default / set_context → 全 BeliefNet.step 路径 + 拆两步 |

### 2.4 新增训练入口（1 脚本）

#### `hyper_mve/scripts/train_main.py`（**Q4 单一入口**）

```bash
python scripts/train_main.py \
    --preset medium \
    --variant hyper \              # hyper / baseline_input_wide / baseline_input_deep / baseline_ma_muzero / oracle_only / infer_only
    --max_steps 200000 \
    --override "train.lr=3e-4" \   # 任意 cfg 覆盖
    --override "env.N=8" \
    --resume_from checkpoints/medium_hyper_42/step_50000.pt
```

变体由 cfg overrides 控制：
- `--variant oracle_only` → `curriculum_stage_1_end_frac=1.0`（永远 Stage 1）
- `--variant infer_only` → `curriculum_stage_1_end_frac=0.0`（永远 Stage 3）
- `--variant baseline_*` → Pkg-06a/b shared_backbones 工厂切换（在 Pkg-06 实施）

---

## 3. Capabilities（本包带来的能力）

### 3.1 v4 主线训练能力

- ✅ 单一 train_step 支持 Oracle/Anneal/Inference 三 Stage 自动切换
- ✅ BeliefNet 在线推断 + 训练时 oracle 注入（课程学习）
- ✅ EMA target model + warmup_cosine LR + grad clip（v4.7 稳定性继承）
- ✅ stratified sampling（type α/β 比例硬约束）
- ✅ MVE planner CRN + coord descent 保留（断言 D 物理基础）

### 3.2 对下游包的解锁

| 下游 | 解锁 |
|------|------|
| **Pkg-06 Baselines** | shared_backbones 工厂 + 共享 trainer/worker（断言 B 公平性强制）|
| **Pkg-07 Eval** | trainer 暴露 model / buffer / scheduler 给 eval；Self-Info / Oracle eval 协议 |
| **Pkg-08 Experiments** | scripts/train_main.py --ablation flag 列表 + checkpoint 格式 |

### 3.3 论文实验配置覆盖

- Medium config 200K train_steps × 5 seeds × 4 variants（hyper / baseline × 3 + oracle）
- 课程边界可配置：默认 0.3 / 0.7，Pkg-08 Ablation 4 测试不同边界
- Pkg-08 2x2 ablation（CRN × coord_desc）通过 cfg.train.use_crn / use_coord_desc 切换

---

## 4. Impact（影响范围）

### 4.1 代码影响

| 范畴 | 行数估计 | 备注 |
|------|---------|------|
| 重写 muzero_trainer.py | -300 / +400 | 删除 train_step_infer 167 行，新增单一 train_step + loss 调用 |
| 重写 worker.py | -100 / +180 | BeliefNet.step 接入 + TimeStepRecord 输出 |
| 重写 episode_buffer.py | -150 / +220 | TimeStepRecord 容器 + stratified sampling |
| 修改 mve_planner.py | -20 / +40 | 4 处 set_context 迁移 + cap/belief 传参 |
| 新增 curriculum.py | +120 | lightweight 类 |
| 新增 loss_composition.py | +180 | 函数 + 双路径 backward |
| 新增 train_main.py | +150 | argparse + cfg overrides + 训练循环驱动 |
| 6 个 test_*.py | +800 | 19 硬约束单测 |
| **合计** | **~+2000 / -570** | net ~+1400 |

### 4.2 性能影响

| 维度 | v4.7 | v4 (Pkg-05) | 差异 |
|------|------|-------------|------|
| 单 train_step (B=256, N=4, Medium) | ~290 ms | ~340 ms | +17%（forward 升 27% 由 Pkg-04 spec 07 档位 3 给出，backward + optimizer 不变）|
| collect_episode (T=200) | ~3 s | ~4.5 s | +50%（worker BeliefNet.step 加 + record 字段填充加）|
| sample_batch (B=256, K=5) | ~30 ms | ~40 ms | +33%（v4 字段更多 + stratified sampling 排序）|
| Memory（buffer 5000 episode）| ~1.2 GB | ~1.5 GB | +25%（c_hat + z_hat + delta + tau + cap 5 字段）|
| 200K train_steps wall-clock | ~17 小时 | ~22 小时 | +29%（含 worker collect + train 综合）|

### 4.3 与 Pkg-04 协同验证

| Pkg-04 单测 | Pkg-05 联动验证 |
|------------|----------------|
| `test_pre_5k_belief_detached` | Pkg-05 spec 05 `test_belief_gradient_isolation_pre_5k` 端到端验证 |
| `test_set_context_subjective_no_oracle_types_leak` | Pkg-05 spec 02 worker 不传 oracle types |
| `test_predict_uses_latest_subjective_agent` | Pkg-05 spec 01 trainer K-step unroll 内 stateful 切换 |
| `test_full_step_under_100ms` | Pkg-05 spec 01 `test_train_step_under_350ms`（含 backward 350 - 100 = 250 ms backward + optimizer）|

---

## 5. R1-R10 风险列表

| # | 风险 | 缓解 | 检测 |
|---|------|------|------|
| R5-1 | 单 train_step 性能 > 400 ms（review 修订 3 重算：含 target forward + belief forward 两项遗漏）| 预算分摊表（Day 1 review 修订 3 锁定）：<br>• forward (model) < 100 ms（Pkg-04 spec 07 档位 3）<br>• forward (target, no_grad) < 50 ms（target 不需 set_context_subjective × N 次，仅 encode+predict）<br>• forward (BeliefNet.forward, train 时 oracle_z_seq 注入) < 30 ms（B=256, K=5, N=4 经验）<br>• backward (main + L_belief) < 180 ms<br>• optimizer step + EMA update < 20 ms<br>• buffer.sample_batch + tensor 转移 < 20 ms<br>**合计 < 400 ms（含 20 ms 缓冲）**。spec 01 `test_train_step_under_400ms` + 3 个 micro-benchmark 单测分别计时（model forward / target forward / belief forward）避免 fail 时无法定位瓶颈 | spec 01 + spec 07 |
| R5-2 | sample_batch 因 stratified sampling 排序代价超 50 ms | stratified 仅按 type 分桶不全排序（O(B)）| spec 03 `test_sample_batch_under_50ms` |
| R5-3 | worker BeliefNet.step 在线推断引入 NaN（GRU hidden 初值）| init_hidden 用 zeros + Pkg-03 spec 04 已含 NaN guard | spec 02 `test_collect_episode_no_nan` |
| R5-4 | 课程 Stage 切换跳变（Stage 2 anneal 边界 oracle 突然为 0）| 实现 linear anneal（不是阶跃）| spec 04 `test_oracle_mixing_anneal_continuous` |
| R5-5 | L_belief 反向被 main loss 路径污染（与 Pkg-04 grad_gating 配合错误）| spec 05 §6.6 强制顺序：(1) BeliefNet.forward 拿带 grad c_hat/z_hat → (2) belief_loss 用原图算 L_belief → (3) model.set_context_subjective 把同一张量传入（model 内部按 step 决定 detach）→ (4) L_total = L_main + λ_b·L_belief 一次 backward。**禁止**先 detach 再算 L_belief。双单测：`test_belief_gradient_isolation_pre_5k`（step<5K 时 BeliefNet grad 仅来自 L_belief）+ `test_belief_gradient_both_sources_post_5k`（step≥5K 时 BeliefNet grad_norm 大于纯 belief 路径）| spec 05 + Pkg-04 spec 04 联动 |
| R5-6 | 11 处 set_context 调用点遗漏（grep 残留）| spec 08 沿用 Pkg-04 spec 08 §3.4 grep 验证 | spec 08 `test_no_legacy_set_context_calls` |
| R5-7 | EpisodeReplayBuffer 序列化 / 反序列化失败（v4 字段多）| 沿用 Pkg-01 spec 04 `TimeStepRecord.to_arrays` / `from_arrays` 已含单测 | spec 03 `test_buffer_pickle_roundtrip` |
| R5-8 | train_main.py argparse 与 cfg overrides 不一致 | OmegaConf 风格 `--override "section.field=value"` + cfg.replace 验证 | spec 08 `test_train_main_cfg_override` |
| R5-9 | checkpoint v4.7 → v4 不兼容造成 reproducibility 损失 | 明确 breaking + `_legacy_v4_7/` shim 仅支持 read-only load 老 checkpoint | spec 08 §7 |
| R5-10 | mve_planner.py cap/belief 参数传递时 shape 不匹配 | spec 06 入口 assert + 单测 `test_planner_cap_belief_shape` | spec 06 |

---

## 6. Non-Goals（明确不做）

- ❌ **不实现** Pkg-06 baselines（仅 spec 08 §3 给 shared_backbones 工厂签名预留）
- ❌ **不实现** Pkg-07 eval 协议（仅 spec 08 §4 给 evaluator 接入接口）
- ❌ **不实现** Pkg-08 experiments / ablation 实际执行（仅 spec 08 §5 给 --ablation flag 列表）
- ❌ **不实现** 多进程 worker / async data collection（v4 主线沿用 v4.7 同步单 worker，性能瓶颈未到 multi-process 必要的程度）
- ❌ **不引入** DataLoader / prefetch 异步预加载（同上）
- ❌ **不修改** Pkg-01/02/03/04 任何 SDD 或代码（仅消费）
- ❌ **不修改** 论文 Ch5 文档（实施完成后再回填性能表）
- ❌ **不引入**新依赖（仅 PyTorch + numpy + Pkg-01/02/03/04 已发布接口）
