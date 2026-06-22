# Pkg-05: Trainer & Worker — Design

> 配套阅读：[`proposal.md`](./proposal.md)（**先读 proposal 再读本文**）

---

## 1. Context

### 1.1 项目阶段

本包是 v4 重构的**第五个 SDD 包**，承接 Pkg-01 (Schema) / Pkg-02 (Env) / Pkg-03 (TriContextEncoder + BeliefNet) / Pkg-04 (DualHyperNetwork + Model)。完成本包后，Pkg-06 (Baselines) / Pkg-07 (Eval) / Pkg-08 (Experiments) 三个下游包可并行启动——三者均消费 Pkg-05 的 `MuZeroTrainer / Worker / EpisodeReplayBuffer / CurriculumScheduler / train_main.py`。

```
Pkg-01 (Schema)              ✅ 完成（含 normalize patch + TimeStepRecord 12 字段）
Pkg-02 (Env)                 ✅ 完成（env.info Oracle 信号已暴露）
Pkg-03 (TriContextEncoder + BeliefNet)  ✅ 完成（API 稳定性已锁定）
Pkg-04 (DualHyperNetwork v2 & Model)  ✅ 完成（7 API + grad_gating 双层 detach）
    ↓
Pkg-05 (本包)                ⏳ 当前
    ↓
Pkg-06a/b (Baselines)        ← shared_backbones 复用 trainer/worker
Pkg-07 (Eval)                ← 消费 trainer 暴露的 model / buffer / scheduler
Pkg-08 (Experiments)         ← scripts/train_main.py --ablation 入口
```

### 1.2 前置依赖（本包消费什么）

| 来源 | 内容 |
|------|------|
| **Pkg-01** `schemas/buffer_record.py` | `TimeStepRecord`（12 字段 + z_hat agent_id 升序跳过 self 顺序约定）|
| Pkg-01 `configs/train_config.py` | `TrainConfig`（27 字段：课程边界 / λ_b / EMA tau / MVE 参数 / stratified sampling）|
| Pkg-01 `configs/v4_config.py` | `V4Config.from_preset("easy/medium/hard")` |
| **Pkg-02** `envs/resource_commons/env.py` | `env.reset(options={'c': ...})` / `env.step(action)` / `env.info["c_true" / "types" / "caps"]` |
| **Pkg-03** `models/belief_losses.py` | `belief_loss(c_hat, z_hat, hidden, c_true, types_true, ...)`（trainer L_belief 计算）|
| Pkg-03 `models/belief_losses.py` | `build_oracle_z_seq(types_true)`（Stage 1/2 oracle 注入）|
| Pkg-03 `models/belief_net.py` | `BeliefNet.step(obs, prev_hidden) -> (hidden, c_hat, z_hat)`（worker 在线推断）|
| Pkg-03 `models/belief_net.py` | `BeliefNet.forward(obs_seq, oracle_z_seq=...)`（trainer 训练时）|
| **Pkg-04** `models/hyper_muzero_model.py` | `HyperMuZeroModel` 7 API（update_step / set_context_objective / set_context_subjective / encode / transition / predict_reward / predict）|
| Pkg-04 `models/grad_gating.py` | `BeliefGradGating` 已内置 model 内部，trainer 不直接调（仅 update_step 触发）|
| **Ch5 v4** | §5.6 buffer + sampling, §5.7 课程 3 stage, §5.8 BYOL consistency |

### 1.3 本包提供（后续包消费什么）

| 输出 | 消费者 | 用途 |
|------|--------|------|
| `MuZeroTrainer` 类 | Pkg-06, Pkg-07, Pkg-08 | 主训练循环 |
| `Worker` 类 | 同上 | 数据收集 |
| `EpisodeReplayBuffer` 类 | 同上 | TimeStepRecord 存储 + 采样 |
| `CurriculumScheduler` 类 | Pkg-07 (eval 期复用) + Pkg-08 (ablation curriculum 边界扫描) | step → stage 映射 |
| `compose_total_loss` 函数 | Pkg-06 (baseline 共享 loss 拼装) | 主 loss + L_belief 拼装 |
| `scripts/train_main.py` | Pkg-08 | 单一训练入口 |
| checkpoint 格式 | Pkg-07/08 | resume / eval / ablation |

---

## 2. Goals

### 2.1 主目标（必须完成）

1. **G1**: v4 单一 train_step（合并 v4.7 Oracle/Infer 双套，与 Pkg-04 7 API 严格对齐）
2. **G2**: worker 接 BeliefNet.step 在线推断 + 写 v4 TimeStepRecord 完整 12 字段
3. **G3**: EpisodeReplayBuffer 重写为 TimeStepRecord 容器 + stratified sampling
4. **G4**: CurriculumScheduler 模块化（lightweight，仅 step → stage 映射）
5. **G5**: loss_composition.py 抽出（main loss + L_belief 双路径 backward 拼装）
6. **G6**: 11 处 set_context 调用点全部迁移（grep 验证 0 残留）
7. **G7**: 性能预算达标（**train_step < 400 ms**（review 修订 3 重算，含分摊表见 §7 / proposal §5 R5-1）/ sample_batch < 50 ms / collect_episode < 5 s）

### 2.2 衍生目标（应尽量达到）

- **G8**: `mypy --strict hyper_mve/training/*.py` 零错误
- **G9**: 单测覆盖率 ≥ 80%
- **G10**: scripts/train_main.py 端到端 1K steps 无 NaN（Medium config）
- **G11**: spec 08 §3-§5 三个下游包接入伪签名 finalized（Pkg-06/07/08 启动条件）

### 2.3 Non-Goals（明确不解决）

- **NG1**: 不实现 Pkg-06 baselines 实际代码
- **NG2**: 不实现 Pkg-07 eval 协议实际代码
- **NG3**: 不实现 Pkg-08 ablation 实际实验
- **NG4**: 不引入多进程 worker / async data collection
- **NG5**: 不引入 DataLoader prefetch
- **NG6**: 不修改 Pkg-01/02/03/04 任何 spec 或代码（仅消费）
- **NG7**: 不修改论文 Ch5 文档
- **NG8**: 不引入新依赖

---

## 3. Decisions

> 10 项关键设计抉择。每项格式：**Decision** → 候选 → 推荐 → 理由 → 风险与回滚。

### D1: train_step 双套合并 vs 保留

**Decision**：v4.7 `train_step` (L203-430) + `train_step_infer` (L432-598) 在 v4 是否合并？

**候选**：
- **A1**：合并为单一 `train_step`（**Q2 用户决议**）
- **A2**：保留双套但抽共享 helpers
- **A3**：完全重写为 `trainer.fit(num_steps)` 统一接口

**推荐**：**A1**（合并）

**理由**：
- v4 BeliefNet 取代 v4.7 GRU 推理，"Infer" 不再是模型类区分概念
- v4 HyperMuZeroModel 统一了 Oracle/Infer 双类，trainer 端也应该对应统一
- v4.7 train_step_infer 的 GRU history 上下文推断逻辑被 BeliefNet.step / forward 完全取代
- A2 保留双套会让 trainer 内部判 model 类型，污染代码
- A3 完全重写破坏 v4.7 调用习惯，PR diff 极大

**风险**：删除 train_step_infer 后无法回滚到 v4.7 Infer 模式 → `_legacy_v4_7/` 已归档（Pkg-01 spec 07），需 v4.7 实验时切到 git tag v4.7-final

**回滚**：git revert + 复活 `_legacy_v4_7/scripts/train_infer.py`

---

### D2: EpisodeReplayBuffer 归属

**Decision**：`EpisodeReplayBuffer` 类的代码归属哪个包？

**候选**：
- **B1**：Pkg-05 范围（Pkg-01 仅保留 TimeStepRecord schema）（**Q3 用户决议**）
- **B2**：Pkg-01 范围（buffer 作为 schema 容器一并出）
- **B3**：独立 Pkg-05b 子包

**推荐**：**B1**（Pkg-05 范围）

**理由**：
- 现有目录结构 `hyper_mve/schemas/`（Pkg-01）vs `hyper_mve/training/`（Pkg-05）天然分离 schema 与 storage
- TimeStepRecord 是 dataclass schema（无 IO 副作用），EpisodeReplayBuffer 是带 sample/store 副作用的 storage class，职责不同
- Pkg-01 已 finalize，不允许事后修改（用户强烈反对基础 SDD 事后改）
- B3 颗粒过细，与 Pkg-04 8 specs 体量不匹配

**风险**：buffer 字段与 TimeStepRecord 字段同步压力 → spec 03 单测 `test_v47_episodedata_v4_record_field_mapping` 验证字段映射

**回滚**：将 EpisodeReplayBuffer 上推到 Pkg-01（需 Pkg-01 增量 PR，违反用户原则）

---

### D3: 训练脚本架构

**Decision**：v4 训练脚本（scripts/）如何组织？

**候选**：
- **C1**：单一 `train_main.py` + cfg 选项控制变体（**Q4 用户决议**）
- **C2**：保留 3-4 个 train_*.py（baseline / oracle / infer / hyper）
- **C3**：train_main.py + 多个 ablation_*.py 兄弟脚本

**推荐**：**C1**（单一 train_main.py）

**理由**：
- v4 模型层统一为 HyperMuZeroModel，Oracle/Infer/Baseline 不是模型类区分
- 变体通过 cfg overrides 表达：
  - oracle_only → `curriculum_stage_1_end_frac=1.0`（永远 Stage 1）
  - infer_only → `curriculum_stage_1_end_frac=0.0`（永远 Stage 3）
  - baseline_* → Pkg-06 shared_backbones 工厂切换
- v4.7 多脚本风格的"固定 cfg overrides"已被 cfg 配置系统取代
- C3 ablation 应在 Pkg-08 SDD 规划，Pkg-05 不预测

**风险**：单一脚本入口可能复杂 → spec 08 §6 严格规范 argparse 接口 + cfg override 语法

**回滚**：保留 train_main.py + 拆 train_baseline.py / train_hyper.py 兼容脚本（仅 wrapper 调 train_main）

---

### D4: curriculum 模块独立

**Decision**：课程学习逻辑是 trainer 内联还是抽出独立模块？

**候选**：
- **D1**：独立 `CurriculumScheduler` 类（lightweight）
- **D2**：trainer 内联 if step < stage_1_end 分支
- **D3**：使用第三方课程库（如 ravelm）

**推荐**：**D1**（独立 CurriculumScheduler）

**理由**：
- 课程逻辑有清晰的 step → stage 映射 + λ_b(step) + oracle_z_mixing_weight(step) 三个 query 方法，适合类封装
- Pkg-07 eval 期可能复用 curriculum 状态（评估时看当前 stage）
- Pkg-08 ablation 4 测试不同课程边界，需要替换 scheduler 实例
- D2 trainer 内联会让 trainer 函数过长，loss 调用前后散落 stage 判断
- D3 引入外部依赖违反 NG8

**风险**：模块化抽出有边界划分压力 → spec 04 严格规范 3 个 query 方法 + 单测覆盖

**回滚**：内联到 trainer（删除 curriculum.py）

---

### D5: loss 组合归属

**Decision**：main loss + L_belief 拼装在哪里？

**候选**：
- **E1**：独立 `loss_composition.py` 模块
- **E2**：trainer 内联（与 v4.7 一致）
- **E3**：HyperMuZeroModel.compute_losses（与 Pkg-04 澄清 1 冲突）

**推荐**：**E1**（独立模块）

**理由**：
- Pkg-04 澄清 1 已锁定 model 不暴露 compute_losses（E3 排除）
- v4 loss 拼装含双路径（main 受 grad_gating / L_belief 不受影响），逻辑复杂度足以抽出
- Pkg-06 baselines 复用相同 loss 拼装函数（仅 model 工厂不同），抽出模块化更友好
- E2 trainer 内联会让 train_step 函数过长（含 K-step unroll + N agent 循环 + 双路径 backward + EMA + scheduler）

**风险**：loss_composition 函数参数过多 → spec 05 严格规范签名 + 返回 dict 含分量

**回滚**：内联到 trainer.train_step

---

### D6: EMA + scheduler 内联 vs 抽出

**Decision**：EMA target model 更新 + warmup_cosine scheduler 是 trainer 内部辅助还是独立 spec？

**候选**：
- **F1**：内联 MuZeroTrainer 内部 helpers（v4.7 模式）
- **F2**：抽出独立 ema.py / scheduler.py 模块

**推荐**：**F1**（内联）

**理由**：
- EMA 和 scheduler 是 trainer 一次性 setup + per train_step 调用的辅助逻辑，无独立性
- v4.7 已稳定（_update_ema_target / _build_scheduler），保留模式
- F2 抽出后 trainer 内还是要 import 调用，无收益
- spec 07 单独 spec 描述 EMA + scheduler 行为（独立 spec 用于文档化，不等于独立模块）

**风险**：内联导致 trainer.py 文件大 → 用 method 拆分（_update_ema / _build_scheduler / _step_scheduler 等）

**回滚**：抽出 ema.py / scheduler.py（仅文件位置改动）

---

### D7: MVE planner cap/belief 传参

**Decision**：mve_planner 如何接收 cap / belief 参数？

**候选**：
- **G1**：sample_mve_plan 入口显式接受 `cap` / `belief` dict 参数
- **G2**：planner 内部从 env / model 拉取（v4.7 模式）
- **G3**：global state（不推荐）

**推荐**：**G1**（显式参数）

**理由**：
- v4 planner 不再依赖 env 拿 rule（已废弃），需 cap / belief 从 trainer/worker 准备
- 显式参数让调用清晰，便于单测
- G2 v4.7 模式让 planner 与 env 耦合，难做 baseline 复用
- G3 反模式

**风险**：planner 接口扩展 → spec 06 严格规范签名 + 单测 `test_planner_cap_belief_shape`

**回滚**：planner 内部从 cfg.env.type_assignment / cfg.env.caps 拉取（仅 init 时拉一次）

---

### D8: checkpoint 兼容性

**Decision**：v4 checkpoint 是否兼容 v4.7？

**候选**：
- **H1**：v4 字段，向 v4.7 不兼容（明确 breaking）+ `_legacy_v4_7/` shim 仅 read-only load
- **H2**：双向兼容（自动检测版本 + 字段映射）
- **H3**：v4 不读 v4.7 checkpoint（强制 from scratch）

**推荐**：**H1**（向 v4.7 不兼容 + read-only shim）

**理由**：
- v4 字段（c_hat / z_hat / types / caps / delta）与 v4.7 完全不同，自动映射复杂且易错
- v4.7 checkpoint 通过 `_legacy_v4_7/` 兼容 shim 仅做 read-only resume（不允许写入 v4 checkpoint）
- H2 双向兼容代码量大 + 隐式 bug 风险
- H3 强制 from scratch 浪费已训练 epoch（v4.7 200K step ckpt 完全废弃）

**风险**：v4.7 实验若需 reproduce 必须切到 git tag v4.7-final → spec 08 §7 明确说明

**回滚**：H2 双向兼容（不推荐，仅极端情况）

---

### D9: stratified sampling 实现

**Decision**：stratified sampling 在 Pkg-05 spec 03 内实现还是抽出独立 sampler 类？

**候选**：
- **I1**：Pkg-05 spec 03 内实现（buffer.sample_batch 内 if stratified 分支）
- **I2**：独立 `StratifiedSampler` 类（buffer 持有 sampler 实例）

**推荐**：**I1**（内联）

**理由**：
- stratified sampling 逻辑简单（按 type 分桶 + 比例抽样），不需要独立类
- buffer.sample_batch 已是采样入口，分支处理一致性高
- I2 抽出后只多一个文件，单测维护成本增加

**风险**：sample_batch 内分支多 → spec 03 用 helper method `_stratified_sample` 拆分

**回滚**：抽出 StratifiedSampler（仅文件位置改动）

---

### D10: v4.7 → v4 替换策略

**Decision**：v4.7 muzero_trainer.py / worker.py / episode_buffer.py 如何处理？

**候选**：
- **J1**：inplace 修改文件 + 旧版本在 git history + Pkg-01 已归档至 `_legacy_v4_7/`
- **J2**：新增 V2 类（MuZeroTrainerV2、WorkerV2）并保留 V1
- **J3**：重写并删除 V1

**推荐**：**J1**（inplace 修改，与 Pkg-04 D8 一致）

**理由**：
- 与 Pkg-04 D8 策略一致（inplace + _legacy_v4_7 归档 + git tag v4.7-final）
- git history 足够回滚
- J2 保留 V1 会让 scripts/train_main.py 判断 V1/V2 复杂
- J3 删除 V1 破坏审计跟踪

**风险**：PR diff 大 → 拆分 commit：muzero_trainer.py / worker.py / episode_buffer.py 各自一个 commit + curriculum.py / loss_composition.py 一个 commit

**回滚**：git revert + 复活 `_legacy_v4_7/`

---

## 4. 设计决策对照表

| Decision | 推荐 | 影响范围 | 后续修改成本 |
|----------|------|----------|--------------|
| D1 train_step 合并 | 单一 train_step | muzero_trainer.py | 高（PR diff 大）|
| D2 EpisodeReplayBuffer 归属 | Pkg-05 范围 | episode_buffer.py | 低（仅位置）|
| D3 训练脚本架构 | 单一 train_main.py | scripts/ | 中（cfg override 复杂度）|
| D4 curriculum 模块独立 | 独立类 | training/curriculum.py | 低（lightweight）|
| D5 loss 组合归属 | 独立模块 | training/loss_composition.py | 低（函数式）|
| D6 EMA + scheduler | 内联 trainer | muzero_trainer.py 内部 | 低 |
| D7 planner cap/belief 传参 | 显式参数 | mve_planner.py 入口 | 低（signature）|
| D8 checkpoint 兼容性 | v4 不兼容 v4.7 | trainer save/load | 中（明确 breaking）|
| D9 stratified sampling | spec 03 内联 | episode_buffer.py | 低 |
| D10 v4.7 替换策略 | inplace 修改 | training/* + planning/* | 高（PR diff 大）|

---

## 5. 实现顺序建议

```
Day 1（半天）:
  - README.md + proposal.md + design.md 三件套
  - 用户审阅 10 Decisions
  
Day 1（半天）+ Day 2:
  - spec 01-trainer-loop-v2 (MuZeroTrainer 类签名 + 单一 train_step + TrainConfig 27 字段)
  - spec 02-worker-collection (Worker + BeliefNet.step 接入 + TimeStepRecord 输出)
  
Day 3:
  - spec 03-episode-buffer-v2 (EpisodeReplayBuffer + stratified sampling)
  - spec 04-curriculum-scheduler (CurriculumScheduler lightweight + 3 stage 切换)
  - **Hard gate**: spec 01/02 trainer/worker 调用顺序与 Pkg-04 spec 08 一致
  
Day 4:
  - spec 05-loss-composition (compose_total_loss 函数 + 双路径 backward 图)
  - spec 07-ema-and-scheduler (EMA tau=0.99 + warmup_cosine 实现细节)
  
Day 5:
  - spec 06-mve-planner-v4 (CRN 4 phase 保留 + 4 处 set_context 迁移)
  - spec 08-integration-contracts (4 类 API 稳定性 + Pkg-06/07/08 启动条件 + train_main.py 入口)
  - **Pkg-06/07/08 启动条件**: spec 08 §3-§5 完整
  
Day 6:
  - 全 spec 交叉引用核对（README §6 spec 间引用表 vs 实际引用）
  - 19 项硬约束单测命名表对账
  - 产出 ref_matrix.csv (机器校验)
  
Day 7:
  - PR description + 用户最终 ack
```

---

## 6. 跨包接口约定（API contract，spec 08 §1 完整版）

### 6.1 import 路径（稳定）

```python
# Pkg-06/07/08 应使用这些 import 路径，本包承诺不变更
from hyper_mve.training import MuZeroTrainer, Worker, EpisodeReplayBuffer
from hyper_mve.training.curriculum import CurriculumScheduler
from hyper_mve.training.loss_composition import compose_total_loss

# 标准用法
from hyper_mve.configs import V4Config
from hyper_mve.models import HyperMuZeroModel

cfg = V4Config.from_preset("medium")
model = HyperMuZeroModel(cfg)
trainer = MuZeroTrainer(cfg, model)
worker = Worker(cfg, model, env)
buffer = EpisodeReplayBuffer(cfg)
```

### 6.2 MuZeroTrainer 对外 API（稳定；review 修订 1：scheduler 注入）

```python
class MuZeroTrainer:
    def __init__(
        self,
        cfg: V4Config,
        model: HyperMuZeroModel,
        scheduler: Optional[CurriculumScheduler] = None,  # ← review 修订 1: D4 实例可替换
        projector: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        """
        scheduler:
            review 修订 1: D4 "Pkg-08 ablation 4 替换 scheduler 实例" 的实现保障.
            None 时 trainer 内部 `CurriculumScheduler(cfg)` 默认构造;
            Pkg-08 可显式传入自定义 scheduler 跑课程边界扫描 ablation.
            trainer 持有此实例 (self.scheduler), compose_total_loss 内部从
            trainer.scheduler 取（不再在 §6.6 签名独立传参 — 简化）.
        """
    
    def train_step(
        self,
        batch: dict[str, torch.Tensor],     # 来自 buffer.sample_batch
        global_step: int,
    ) -> dict[str, float]:                  # 返回 loss 分量字典（用于日志）
        """单一 train_step（Q2 合并 v4.7 双套）.
        
        内部按 Pkg-04 spec 02 §2.3 调用顺序:
            model.update_step(global_step)
            model.set_context_objective(c_t)
            for k in range(N):
                model.set_context_subjective(k, cap, belief)
                # K-step unroll
        
        loss 拼装内部调 compose_total_loss(model, batch, self.scheduler, global_step, cfg).
        """
    
    def save_checkpoint(self, path: str) -> None: ...
    def load_checkpoint(self, path: str) -> int: ...  # 返回 global_step
```

### 6.3 Worker 对外 API（稳定；review 修订 5：planner 持有 + CRN seed 跨 episode 持续）

```python
from hyper_mve.planning.mve_planner import MVEPlanner

class Worker:
    def __init__(
        self,
        cfg: V4Config,
        model: HyperMuZeroModel,
        env: ResourceCommonsEnv,
        planner: Optional[MVEPlanner] = None,  # ← review 修订 5: worker 持有 planner
    ):
        """
        planner:
            review 修订 5: worker 持有 planner 实例（不在 collect_episode 内每集创建）,
            保证 CRN seed 跨 episode 持续 (R5-3 NaN 稳定性).
            None 时 worker 内部 `MVEPlanner(cfg)` 默认构造.
            self.planner = planner or MVEPlanner(cfg)
        """
    
    def collect_episode(
        self,
        epsilon: float = 0.1,
        use_planner: bool = True,
    ) -> list[TimeStepRecord]:
        """收集一个完整 episode，输出 TimeStepRecord 列表（Pkg-01 spec 04 字段约定）.
        
        worker **不**调 model.update_step (Pkg-04 spec 04 §3.4).
        worker 用 BeliefNet.step 在线推断 (替代 v4.7 set_context_from_history).
        use_planner=True 时复用 self.planner.sample_mve_plan(model, root_s, cap, belief, c_t).
        
        集间状态：
            - self.planner.crn_rng_state 跨 episode 持续（保证 R5-3 稳定性）
            - BeliefNet hidden state 在 episode 起点 reset (model.belief_net.init_hidden)
            - 单测 test_worker_planner_persistent: 复用同一 planner 实例验证 id(self.planner) 不变
        """
```

### 6.4 EpisodeReplayBuffer 对外 API（稳定；review 修订 4：c_t_seq 显式参数）

```python
class EpisodeReplayBuffer:
    def __init__(self, cfg: V4Config): ...
    
    def store_episode(
        self,
        records: list[TimeStepRecord],
        c_t_seq: torch.Tensor,           # ← review 修订 4: 显式增加, shape (T,) float32
    ) -> None:
        """存储一个 episode.
        
        参数（review 修订 4 澄清）:
            records: T 个 TimeStepRecord（Pkg-01 spec 04 12 字段）
            c_t_seq: (T,) float32，与 records 平行的 c_t 标量序列；
                     来源：worker 从 env.info["c_true"] 逐步采集（与 record 时间步对齐）
        
        实施：
            - TimeStepRecord 不含 c_t（保持 Pkg-01 spec 04 12 字段不变，NG7 遵守）
            - buffer 内部维护与 records 平行的 c_t_seq 数组
            - 这不修改 Pkg-01 schema，仅扩展 Pkg-05 buffer 接口
        """
    
    def sample_batch(
        self,
        batch_size: int,
        unroll_K: int,
    ) -> dict[str, torch.Tensor]:
        """采样一个 batch，返回字典.
        
        Returns dict 字段对齐 TimeStepRecord + c_t:
            obs: (B, K+1, N, obs_dim)
            actions: (B, K+1, N)
            rewards: (B, K+1, N)
            delta: (B, K+1, N)
            pi_mve: (B, K+1, N, A)
            v: (B, K+1, N)
            tau: (B, K+1, N)
            cap: (B, K+1, N, 4)
            c_hat: (B, K+1, N)
            z_hat: (B, K+1, N, N-1, 2)
            t: (B, K+1)
            done: (B, K+1)
            c_t: (B, K+1)  ← 来自 store_episode 入参 c_t_seq（review 修订 4 锁定来源）
        """
    
    def __len__(self) -> int: ...  # 当前 episode 数
```

### 6.5 CurriculumScheduler 对外 API（稳定）

```python
class CurriculumScheduler:
    def __init__(self, cfg: V4Config): ...
    
    def stage(self, global_step: int) -> str:
        """返回 'stage_1' / 'stage_2' / 'stage_3'."""
    
    def oracle_z_mixing_weight(self, global_step: int) -> float:
        """Stage 1: 1.0 / Stage 2: 1.0 → 0.0 anneal / Stage 3: 0.0."""
    
    def lambda_b(self, global_step: int) -> float:
        """L_belief 课程加权系数."""
```

### 6.6 compose_total_loss 函数（稳定；review 修订 1+2：删 scheduler 参数 + 双路径独立张量）

```python
def compose_total_loss(
    model: HyperMuZeroModel,
    batch: dict[str, torch.Tensor],
    trainer: "MuZeroTrainer",       # ← review 修订 1: 从 trainer 取 scheduler (不独立传)
    global_step: int,
    cfg: V4Config,
) -> dict[str, torch.Tensor]:
    """主 loss + L_belief 拼装.
    
    返回 dict[str, Tensor]:
        "total":      total loss (含 grad, 一次 backward 同时反传两路径)
        "main":       w_policy·L_π + w_value·L_v + w_reward·L_r + w_consist·L_BYOL
        "belief":     L_c + L_opp + L_div
        "lambda_b":   课程加权系数（标量, = trainer.scheduler.lambda_b(global_step)）
        "policy":     w_policy·L_π
        "value":      w_value·L_v
        "reward":     w_reward·L_r
        "consist":    w_consist·L_BYOL
        "belief_c":   w_belief_c·L_c
        "belief_opp": w_belief_opp·L_opp
        "belief_div": w_belief_div·L_div
    
    双路径 backward 实现关键（review 修订 2: 必须两份独立 autograd 路径）:
    
    1. BeliefNet 一次 forward 拿带 grad 的 c_hat / z_hat (训练时, 可能含 oracle_z_seq 注入):
           c_hat_raw, z_hat_raw = model.belief_net.forward(
               batch["obs"], oracle_z_seq=trainer.scheduler.build_oracle_z_seq(...)
           )
    
    2. L_belief 路径（不经 model.set_context_*, 不受 grad_gating 影响）:
           L_belief = belief_loss(
               c_hat_raw, z_hat_raw,             # ← 带 grad 原图
               c_true=batch["c_t"], types_true=batch["tau"], ...
           )
           # 此路径 backward 时直接反传到 BeliefNet GRU + heads
    
    3. main 路径（经 model.set_context_subjective, 内部 grad_gating 按 step detach）:
           # c_hat_raw / z_hat_raw 是 alias 不是 detach — 同一 autograd 图节点
           # model.set_context_subjective 内部对 belief tuple 调 grad_gating.apply_raw + apply_ctx
           # → 若 step < 5000: BeliefNet/BeliefEncoder grad=0 (Pkg-04 spec 04 双层 detach)
           # → 若 step >= 5000: 透传, main loss 也参与 BeliefNet 训练
           for k in range(N):
               model.set_context_subjective(k, batch["cap"][:, k],
                   belief=(c_hat_raw[:, k], z_hat_raw[:, k]))
               # ... K-step unroll forward + L_π / L_v / L_r 累加 ...
           L_main = w_policy·L_π + w_value·L_v + w_reward·L_r + w_consist·L_BYOL
    
    4. 一次 backward 同时反传两路径:
           L_total = L_main + lambda_b · L_belief
           L_total.backward()
    
    关键单测（spec 05 §5）:
    - test_belief_gradient_isolation_pre_5k:
        step < 5000 时 BeliefNet 参数 grad 仅来自 L_belief（不含 main 来源）
    - test_belief_gradient_both_sources_post_5k:
        step >= 5000 时 BeliefNet 参数 grad_norm 大于纯 L_belief 路径的 grad_norm
        （证明 main 路径也参与训练 BeliefNet）
    """
```

**禁止反模式（review 修订 2 强调）**：

❌ 不要在第 2 步前对 c_hat_raw / z_hat_raw 调 .detach()——会切断 L_belief 反传到 BeliefNet
❌ 不要先调 model.set_context_subjective 再用 belief tuple 算 L_belief——model 内部已 detach
✅ 必须先用原图（步骤 2）算 L_belief，再让 model（步骤 3）按 step 决定是否 detach

### 6.7 TrainConfig 字段依赖（27 项，spec 01 §1 穷举）

Pkg-05 trainer / worker / buffer / curriculum / loss_composition 完整依赖以下 cfg 字段（变更需 Pkg-01 spec 05 同步）：

**cfg.train.* (24 项)**:
- max_train_steps, batch_size, buffer_size, min_buffer_size
- episodes_per_iter, train_steps_per_iter
- unroll_K, n_step, gamma
- lr, lr_min, adam_eps, grad_clip
- lr_schedule, lr_warmup_steps
- w_policy, w_value, w_reward, w_consist, w_belief
- w_belief_c, w_belief_opp, w_belief_div, belief_div_target_std
- curriculum_stage_1_end_frac, curriculum_stage_2_end_frac
- belief_grad_gating_steps（Pkg-04 model 内消费，trainer 不直接读，仅 update_step 传递）
- ema_tau
- epsilon_init, epsilon_min, epsilon_decay_steps
- mve_samples, mve_depth, mve_temperature
- use_crn, randomize_order
- stratified_sampling, stratified_min_per_type_frac

**cfg.env.* (2 项)**:
- N, type_assignment（worker 写 TimeStepRecord.tau 用）

**cfg.model.* (1 项)**:
- latent_dim（trainer K-step unroll 内 shape 检查）

### 6.8 v4.7 → v4 调用点迁移指引（spec 08 §2 完整版）

| 文件 | v4.7 行号 | v4 迁移 | 备注 |
|------|----------|---------|------|
| muzero_trainer.py | L239 | `set_context(rules, agent_ids)` → 拆为 `set_context_objective(c_t)` + 循环 `set_context_subjective(k, cap, belief)` | trainer K-step unroll 起点 |
| muzero_trainer.py | L242 | target_model 同上 | EMA target 同步 |
| muzero_trainer.py | L474 | `set_context_from_history` → **删除**（BeliefNet.step 取代）| Infer 双类合并 |
| muzero_trainer.py | L484 | 同 L242 模式（含 belief 替代 inferred_rule_emb）| – |
| muzero_trainer.py | L678 | 同 L239 | – |
| muzero_trainer.py | L680 | `set_context_default` → **删除**（v4 BeliefNet 输出 belief 替代）| – |
| worker.py | L125 | `set_context_from_history` → BeliefNet.step + set_context_subjective | – |
| worker.py | L128 | `set_context_default` → set_context_subjective | – |
| worker.py | L143 | 同 L125 | – |
| worker.py | L145 | 同 L128 | – |
| worker.py | L149 | `set_context(rule_t, id_i)` → set_context_objective + set_context_subjective | – |
| mve_planner.py | L71 | `set_context(rule_exp, id_i)` → set_context_subjective | planner 入口处先调 set_context_objective |
| mve_planner.py | L213 | 同 L71 | 目标态转移 |
| mve_planner.py | L219 | 同 L71 | 奖励预测 |
| mve_planner.py | L258 | 同 L71 | 终值估计 |

---

## 7. 验证策略概览

> 详细 acceptance criteria 见 `specs/*.md`。本节列 19 项硬约束 → 具名单测映射（M3：风险预映射）。

| # | 硬约束 | 具名单测 | spec 归属 |
|---|--------|----------|-----------|
| C5-T1 | trainer 每 train_step 调 update_step 1 次 | `test_trainer_calls_update_step_per_step` | spec 01 |
| C5-T2 | K-step unroll 内 set_context_objective 一次 | `test_objective_called_once_per_unroll` | spec 01 |
| C5-T3 | N agents 循环 set_context_subjective | `test_subjective_called_per_agent` | spec 01 |
| C5-W1 | worker 不调 update_step | `test_worker_no_update_step` | spec 02 |
| C5-W2 | worker 用 BeliefNet.step 在线推断 | `test_worker_uses_belief_net_step` | spec 02 |
| C5-W3 | TimeStepRecord 字段填充顺序合规 | `test_z_hat_order_matches_pkg01_spec04` | spec 02 |
| C5-B1 | buffer 内 z_hat 顺序一致性（端到端）| `test_buffer_z_hat_order_e2e` | spec 03 |
| C5-B2 | stratified sampling 比例 | `test_stratified_min_per_type_frac` | spec 03 |
| C5-S1 | 课程 3 stage 切换边界（0.3 / 0.7）| `test_curriculum_stage_boundaries` | spec 04 |
| C5-S2 | Stage 1 100% oracle 注入 | `test_stage_1_full_oracle` | spec 04 |
| C5-S3 | Stage 2 anneal 单调下降 | `test_oracle_mixing_anneal_monotonic` | spec 04 |
| C5-L1 | λ_b(step) 课程加权曲线 | `test_lambda_b_curve_matches_cfg` | spec 05 |
| C5-L2 | L_belief 路径与 main loss 分离 | `test_belief_gradient_isolation_pre_5k` | spec 05 |
| C5-P1 | CRN seed 相同 step 0 输出一致 | `test_crn_step0_deterministic_same_seed` | spec 06 |
| C5-P2 | 4 处 set_context 调用点全部迁移 | `test_planner_4_set_context_migrated` | spec 06 |
| C5-E1 | EMA tau=0.99 衰减率正确 | `test_ema_decay_correctness_tau_099` | spec 07 |
| C5-E2 | warmup_cosine LR 曲线 | `test_lr_warmup_5k_then_cosine_anneal` | spec 07 |
| C5-I1 | v4.7 → v4 grep 验证调用点全消失 | 沿用 Pkg-04 spec 08 §3.4 `test_no_legacy_set_context_calls` | spec 08 |
| R5-1 | train_step < 400 ms（review 修订 3 含分摊表）| `test_train_step_under_400ms` + 3 micro-benchmark | spec 01 + spec 07 联动 |
| R5-2 | sample_batch < 50 ms | `test_sample_batch_under_50ms` | spec 03 |

---

## 8. Open Questions（含 2026-05-29 用户审阅决议）

| # | Question | 决议 | 影响 |
|---|----------|------|------|
| **Q1** | spec 数量 7 vs 8 vs 10 | ✅ **8 个**（与 Pkg-03/04 对称） | README + spec 08 新增 |
| **Q2** | train_step 双套合并 vs 保留 | ✅ **合并为单一 train_step** | spec 01 + D1 |
| **Q3** | EpisodeReplayBuffer 归属 Pkg-01 vs Pkg-05 | ✅ **Pkg-05 范围**（Pkg-01 仅 TimeStepRecord schema）| spec 03 + D2 |
| **Q4** | 训练脚本架构（单一 vs 多个）| ✅ **单一 train_main.py + cfg 选项控制变体** | spec 08 §6 + D3 |
| Q5 | curriculum 模块独立 vs trainer 内联 | ✅ **独立 CurriculumScheduler**（D4）| spec 04 |
| Q6 | loss 组合归属 model.compute_losses vs 独立函数 | ✅ **独立 compose_total_loss 函数**（与 Pkg-04 澄清 1 一致）| spec 05 |
| Q7 | mve_planner cap/belief 传参方式 | ✅ **显式参数 MVEPlanner.sample_mve_plan(model, root_s, cap, belief, c_t)**（D7；cfg 在 __init__，P1-3 删顶层函数）| spec 06 |
| Q8 | checkpoint 兼容性策略 | ✅ **v4 字段，向 v4.7 不兼容 + read-only shim**（D8）| spec 01 save/load + spec 08 §7 |
| Q9 | D1-D10 锁定 | ✅ **完全锁定**（design.md §3）| – |
| Q10 | 性能预算硬阈值粒度 | ✅ **train_step < 400 ms / sample_batch < 50 ms / collect_episode < 5 s**（三档；review 修订 3 重算 R5-1 从 350 → 400 ms 含分摊表）| spec 01 / spec 03 / spec 07 |

> 全部 Q1-Q10 + D1-D10 已 ack。本 design.md 视为 **finalized**。

---

## 9. References

- `proposal.md`（本包）
- `README.md`（本包）
- Pkg-01 design.md + spec 04 (TimeStepRecord 12 字段) + spec 05 (TrainConfig 27 字段)
- Pkg-02 spec 08（env.info 三段分组）
- Pkg-03 design.md + spec 04/06/08（BeliefNet / belief_loss / 课程接入）
- Pkg-04 design.md + spec 02/04/08（model 7 API + grad_gating + trainer/worker/planner 迁移指引）
- `D:\RL\hyper_mve\docs\Chapter5_Methodology_v4.md` §5.6 + §5.7 + §5.8
- 项目 Plan File: Part D Pkg-05 + Q1-Q4 决议
- v4.7 行号映射：
  - `training/muzero_trainer.py` L56-83/138-181/203-430/432-598/85-130
  - `training/worker.py` L84-180 + L125/128/143/145/149
  - `training/episode_buffer.py` L72-215
  - `planning/mve_planner.py` L80-98/128-290 + L71/213/219/258
