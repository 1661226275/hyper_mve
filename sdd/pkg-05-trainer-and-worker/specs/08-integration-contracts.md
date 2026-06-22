# Spec 08: 集成契约 — 4 类 API 稳定性 + v4.7 → v4 迁移 + train_main.py 入口

> 父文档：[`../proposal.md`](../proposal.md) §3 · [`../design.md`](../design.md) §6
> **本 spec 是 Pkg-05 对外的"硬契约"** — 与 Pkg-03/04 spec 08 对称。
> **落地 Day 1 review 澄清 1 + 澄清 2**（spec 08 §6 / §2）。

---

## 1. Purpose

把 Pkg-05 内部模块（MuZeroTrainer + Worker + EpisodeReplayBuffer + CurriculumScheduler + loss_composition + train_main.py）暴露给下游 Pkg-06 / Pkg-07 / Pkg-08 时，**定义必须严格遵守的契约**，防止：

- 4 类对外 API 签名 drift
- v4.7 → v4 11 处调用点迁移漏点（mve_planner / muzero_trainer / worker）
- train_main.py --variant 语义混淆（澄清 1：infer_only ≠ v4.7 InferHyperMuZeroModel）
- grep 验证模式误命中（澄清 2：精确正则）
- checkpoint v4/v4.7 不兼容混用（D8）

---

## 2. 契约 1：4 类对外 API 稳定性 + v4.7 → v4 迁移精确 grep（澄清 2）

### 2.1 API 稳定性承诺表（M2：跨包伪签名先行）

| 类 | API | 稳定性 | 签名 | 调用者 |
|----|-----|--------|------|--------|
| `MuZeroTrainer` | `__init__(cfg, model, scheduler=None, projector=None, device=None)` | 🔒 稳定 | 见 spec 01 §2.2 | Pkg-06 baselines / Pkg-08 experiments |
| | `train_step(batch, global_step)` | 🔒 稳定 | dict → dict[str, float] | 同上 |
| | `save_checkpoint(path)` / `load_checkpoint(path)` | 🔒 稳定 | str → None / str → int | Pkg-07 eval（resume）/ Pkg-08（experiments）|
| `Worker` | `__init__(cfg, model, env, planner=None)` | 🔒 稳定 | 见 spec 02 §2.2（修订 5）| Pkg-06 baselines / Pkg-08 |
| | `collect_episode(epsilon, use_planner)` | 🔒 稳定 | (float, bool) → (list[TimeStepRecord], (T,) Tensor) | 同上 |
| `EpisodeReplayBuffer` | `__init__(cfg)` | 🔒 稳定 | 见 spec 03 §2.2 | Pkg-06 baselines |
| | `store_episode(records, c_t_seq)` | 🔒 稳定 | (list, Tensor) → None（**修订 4 显式 c_t_seq**）| Pkg-06 |
| | `sample_batch(B, K)` | 🔒 稳定 | (int, int) → dict[str, Tensor] | Pkg-06 |
| `CurriculumScheduler` | `__init__(cfg)` | 🔒 稳定 | 见 spec 04 §2.2 | Pkg-08 ablation 4 |
| | `stage(step)` / `oracle_z_mixing_weight(step)` / `lambda_b(step)` | 🔒 稳定 | int → str / float / float | Pkg-07 eval（查 stage）/ Pkg-08 |
| `compose_total_loss` | 函数（不是类）| 🔒 稳定 | 见 spec 05 §2.2 | Pkg-06 baselines（复用相同 loss 拼装）|
| `scripts/train_main.py` | 命令行入口 | 🔒 稳定 | 见 §6 | Pkg-08 experiments |

**承诺**：Pkg-05 → Pkg-08 全程不变更上述 API。如需修改，必须发起 issue + 跨包讨论。

### 2.1.1 ⛔ 不在稳定 API 表内的入口

| 入口 | 状态 | 说明 |
|------|------|------|
| `MuZeroTrainer._update_ema_target` / `_build_scheduler` | 内部私有（D6） | 仅 spec 07 文档化行为，trainer 调用方不应直接访问 |
| `MuZeroTrainer.compute_n_step_return` | spec 01 §2.2 公开但不在跨包契约 | 仅 compose_total_loss 内部用 |
| ~~`Worker._call_planner`~~（已删除） | — | P1-1+P1-2 后 planner 调用内联进 `collect_episode` Step B.5（一次性算 N agents policy + 复用 `self.planner`），无独立方法 |
| `EpisodeReplayBuffer._uniform_sample` / `_stratified_sample` / `_slice_batch` | 内部私有（D9） | spec 03 §2.2 暴露但仅 sample_batch 内部用 |

### 2.2 v4.7 → v4 11 处调用点迁移精确 grep（澄清 2）

按 Pkg-04 spec 08 §3 已锁定的 15 处迁移指引（mve_planner 4 + trainer 6 + worker 5），Pkg-05 PR merge 前必须 grep 验证：

```powershell
# === 必须仅匹配 v4.7 单参旧签名 (不命中 _objective / _subjective) ===
# 澄清 2 精确正则: 用 negative lookahead 排除 _objective/_subjective
cd D:\RL\hyper_mve

Select-String -Path "hyper_mve\planning\*.py","hyper_mve\training\*.py" -Pattern '\bset_context\(' |
    Where-Object { $_.Line -notmatch 'set_context_(objective|subjective)\(' }
# 期望: 0 行输出 (v4.7 旧签名应全部消失)


# === 同样验证 v4.7 已删除的 API ===
Select-String -Path "hyper_mve\planning\*.py","hyper_mve\training\*.py" -Pattern 'set_context_(from_history|default)\('
# 期望: 0 行输出
```

### 2.3 单测验证

`tests/migration/test_pkg05_v47_to_v4_set_context.py`（v4.7 → v4 迁移单测）：

```python
import subprocess
import re

def test_no_legacy_set_context_calls_in_pkg05():
    """v4.7 旧 set_context(rule, id) 双参签名应全部消失 (沿用 Pkg-04 spec 08 §3.4 模式 + 澄清 2 精确正则)."""
    result = subprocess.run(
        ['grep', '-rn', r'\bset_context(', 
         'hyper_mve/planning/', 'hyper_mve/training/'],
        capture_output=True, text=True,
    )
    
    # 仅保留命中 set_context( 但不是 _objective / _subjective 的行
    legacy_calls = []
    for line in result.stdout.split('\n'):
        if line and re.search(r'\bset_context\(', line):
            if not re.search(r'set_context_(objective|subjective)\(', line):
                legacy_calls.append(line)
    
    assert len(legacy_calls) == 0, (
        f"Found {len(legacy_calls)} legacy set_context calls:\n" +
        "\n".join(legacy_calls[:10])  # 显示前 10 个
    )


def test_no_legacy_belief_apis():
    """v4.7 set_context_from_history / set_context_default 应全部消失."""
    result = subprocess.run(
        ['grep', '-rn', r'set_context_(from_history|default)',
         'hyper_mve/planning/', 'hyper_mve/training/'],
        capture_output=True, text=True,
    )
    assert result.stdout.strip() == "", (
        f"Found legacy belief APIs:\n{result.stdout}"
    )
```

---

## 3. 契约 2：与 Pkg-06 (Baselines) 协作 — shared_backbones 工厂

按 Pkg-04 spec 08 §5 + Pkg-03 spec 08 §6：所有 7 baseline 共享同一 RepNet / BeliefNet 结构（断言 B 等参公平性强制）。Pkg-05 提供：

### 3.1 暴露给 shared_backbones 的接口

```python
# Pkg-06a/b shared_backbones.py 调用模式 (Pkg-06 SDD 启动条件)
from hyper_mve.training import MuZeroTrainer, Worker, EpisodeReplayBuffer
from hyper_mve.training.curriculum import CurriculumScheduler
from hyper_mve.training.loss_composition import compose_total_loss
from hyper_mve.models import HyperMuZeroModel, DualHyperNetwork  # Pkg-04
# baseline_models 在 Pkg-06 实施


def create_trainer_for_baseline(cfg, baseline_variant: str):
    """统一 trainer 工厂.
    
    baseline_variant ∈ {"hyper", "input_wide", "input_deep", "ma_muzero", "no_belief", ...}
    """
    if baseline_variant == "hyper":
        model = HyperMuZeroModel(cfg)
    else:
        from hyper_mve.baselines import create_baseline_model
        model = create_baseline_model(cfg, baseline_variant)
    
    # ★ 所有 baseline 共享同一 trainer 实例
    return MuZeroTrainer(cfg, model)


def create_worker_for_baseline(cfg, model, env):
    """统一 worker 工厂."""
    return Worker(cfg, model, env)


def create_buffer_for_baseline(cfg):
    """统一 buffer 工厂."""
    return EpisodeReplayBuffer(cfg)
```

### 3.2 关键约束（断言 B 公平性强制）

- 所有 baseline 必须复用同一 `MuZeroTrainer` 类（无 baseline-specific train_step）
- 所有 baseline 必须复用同一 `Worker` 类（采集行为一致）
- 所有 baseline 必须复用同一 `EpisodeReplayBuffer` 类（buffer 行为一致）
- 所有 baseline 必须复用同一 `compose_total_loss` 函数（loss 拼装一致）
- 仅 model 类不同（HyperMuZeroModel vs BaselineModel*）

**Pkg-06 SDD 启动条件**：本 §3 完整 → Pkg-06 可直接消费上述工厂签名启动。

---

## 4. 契约 3：与 Pkg-07 (Eval Protocols) 协作

### 4.1 evaluator 接入接口

Pkg-07 evaluator 需要从 trainer 拉取以下对象：

```python
# Pkg-07 evaluator 调用模式 (Pkg-07 SDD 启动条件)
from hyper_mve.training import MuZeroTrainer
from hyper_mve.training.curriculum import CurriculumScheduler

class Evaluator:
    def __init__(self, trainer: MuZeroTrainer, ...):
        self.model = trainer.model            # ★ trainer 暴露 model
        self.buffer = trainer.replay_buffer   # ★ trainer 暴露 buffer (若 trainer 持有)
        self.scheduler = trainer.scheduler    # ★ trainer 暴露 scheduler (查当前 stage)
        # 注: trainer 不需要持有 buffer (上层 train_main 持有),
        #      此处 trainer.replay_buffer 是 spec 01 实施时可加的可选公开属性
    
    def eval_at_stage(self, global_step: int) -> dict:
        stage = self.scheduler.stage(global_step)
        # 根据 stage 决定 eval 协议
```

### 4.2 关键约束

- `trainer.model` / `trainer.scheduler` / `trainer.target_model` 是 spec 01 公开属性
- evaluator 不应**修改** trainer 内部状态（仅读取）
- Self-Info eval 时调 model.set_context_subjective 必须用 BeliefNet 推断输出（C11，Pkg-04 spec 08 §6 已锁定）

**Pkg-07 SDD 启动条件**：本 §4 完整 → Pkg-07 可消费 trainer 公开属性启动。

---

## 5. 契约 4：与 Pkg-08 (Experiments) 协作 — ablation flag 列表

### 5.1 train_main.py --ablation flag 列表（finalized）

Pkg-08 ablation 实验通过 cfg overrides 控制（spec 08 §6 train_main.py 入口）：

| Ablation # | flag / cfg override | 用途 |
|------------|---------------------|------|
| Ablation 1 | `--override "train.use_crn=False"` | 禁用 CRN（spec 06 § 3.2）|
| Ablation 2 | `--override "train.randomize_order=False"` | 禁用 coord descent（spec 06 § 3.3）|
| Ablation 3 | `--variant alpha_only` 或 `--override "env.type_assignment=[0,0,0,0]"` | 全 α agent（type 扫描）|
| Ablation 4 | `--override "train.curriculum_stage_1_end_frac=0.1"` 等 | 课程边界扫描（spec 04 D4 实例可替换）|
| Ablation 5 | `--override "legacy.w_rew_diversity=0.01"` | reward_diversity_loss 开关（Pkg-04 spec 01 § 2.4）|
| Ablation 6.x | `--variant rewardhead_explicit_type` | spec 05 § 6 显式 type 分支（Pkg-04 spec 05 修订 1）|
| Ablation 7 | `--variant no_belief` | 移除 BeliefNet（baseline 工厂切换，Pkg-06）|
| Ablation 8 | `--override "train.stratified_sampling=False"` | 禁用 stratified sampling（spec 03 § 3.2）|

### 5.2 Pkg-08 实验启动条件

**Pkg-08 SDD 启动条件**：
1. 本 §5 ablation flag 列表 finalized
2. § 6 train_main.py 入口 API 稳定
3. § 7 checkpoint 格式 finalized
4. Pkg-06 SDD ack（baseline 模型工厂可调）
5. Pkg-07 SDD ack（eval 协议可消费）

---

## 6. 契约 5：train_main.py 入口（Q4 + 澄清 1：variant 语义）

### 6.1 命令行 API 稳定

```bash
python scripts/train_main.py \
    --preset {easy,medium,hard} \
    --variant {hyper, baseline_input_wide, baseline_input_deep, baseline_ma_muzero,
              oracle_only, infer_only, no_belief, ...} \
    --max_steps INT \
    --override "section.field=value" \      # 任意 cfg 覆盖（可重复）
    --resume_from PATH                       # checkpoint 路径
    --seed INT                               # 随机种子
    --log_dir PATH                           # TensorBoard 日志
    --ckpt_dir PATH                          # checkpoint 保存
```

### 6.2 --variant 语义表（澄清 1：infer_only ≠ v4.7 InferHyperMuZeroModel）

| --variant | cfg 等效 | 模型架构 | 与 v4.7 关系 |
|-----------|---------|----------|--------------|
| `hyper`（默认）| 全默认 | HyperMuZeroModel（v4） + BeliefNet + 3 stage 课程 | v4 主线 |
| `oracle_only` | `curriculum_stage_1_end_frac=1.0` + `curriculum_stage_2_end_frac=1.0` | HyperMuZeroModel + BeliefNet + **永远 Stage 1**（oracle 注入全程）| ≈ v4.7 Oracle 模式但用 v4 BeliefNet 接收 oracle z_seq（仍非 v4.7 OracleHyperMuZeroModel）|
| `infer_only` | `curriculum_stage_1_end_frac=0.0` + `curriculum_stage_2_end_frac=0.0` | HyperMuZeroModel + BeliefNet + **永远 Stage 3**（纯 BeliefNet 推断）| ⚠️ **不是** v4.7 InferHyperMuZeroModel（后者用 GRU 推 rule，v4 用 BeliefNet 推 c + types）|
| `baseline_input_wide` | model 工厂切换 | Pkg-06a Input-Wide baseline | v4 新增（v4.7 无对应）|
| `baseline_input_deep` | 同 | Pkg-06a Input-Deep baseline | 同上 |
| `baseline_ma_muzero` | 同 | Pkg-06b MA-MuZero baseline | 同上 |
| `no_belief` | model 工厂切换 + curriculum_stage_2_end_frac=0.0 | 移除 BeliefNet 的 HyperMuZeroModel 变体 | Pkg-08 Ablation 7 |

**澄清 1 关键说明**：

> ⚠️ **`--variant infer_only` ≠ v4.7 InferHyperMuZeroModel reproduce**
> 
> v4.7 InferHyperMuZeroModel 用 GRU 推断 rule（隐式连续向量），v4 BeliefNet 推断 c (sigmoid scalar) + types (2-class softmax) — 两者数学定义不同，输出空间不同。
> 
> 若需 **reproduce v4.7 Infer 实验**：
> ```bash
> git checkout v4.7-final
> python scripts/train_infer.py  # _legacy_v4_7/scripts/train_infer.py
> ```
> 与 D8 checkpoint 不兼容声明对齐。

### 6.3 cfg overrides 语法（M5）

```bash
# 简单字段
--override "train.lr=3e-4"

# 嵌套字段
--override "env.N=8"
--override "env.type_assignment=[0,0,1,1,1,1,1,1]"  # JSON 字面量

# 多个 override（重复 --override）
--override "train.lr=3e-4" --override "train.batch_size=128"
```

实施层用 OmegaConf 或自定义 parser，详见 spec 08 §6 实施伪代码（不在此 spec 详述）。

### 6.4 train_main.py 主循环伪代码

```python
def main():
    args = parse_args()
    cfg = V4Config.from_preset(args.preset)
    cfg = apply_overrides(cfg, args.override)
    cfg = apply_variant(cfg, args.variant)
    
    # 工厂 (Pkg-06 shared_backbones)
    model = create_model(cfg, args.variant)
    env = create_env(cfg.env)
    
    trainer = MuZeroTrainer(cfg, model)
    worker = Worker(cfg, model, env)
    buffer = EpisodeReplayBuffer(cfg)
    
    # Resume
    start_step = 0
    if args.resume_from:
        start_step = trainer.load_checkpoint(args.resume_from)
    
    # 主循环
    for global_step in range(start_step, cfg.train.max_train_steps):
        # 1. 采集
        if buffer.len() < cfg.train.min_buffer_size:
            records, c_t_seq = worker.collect_episode()
            buffer.store_episode(records, c_t_seq)
            continue
        
        if global_step % cfg.train.episodes_per_iter == 0:
            for _ in range(cfg.train.episodes_per_iter):
                records, c_t_seq = worker.collect_episode()
                buffer.store_episode(records, c_t_seq)
        
        # 2. 训练
        for _ in range(cfg.train.train_steps_per_iter):
            batch = buffer.sample_batch(cfg.train.batch_size, cfg.train.unroll_K)
            losses = trainer.train_step(batch, global_step)
            log_to_tensorboard(losses, global_step)
        
        # 3. 评估 (Pkg-07 集成)
        if global_step % cfg.eval.evaluate_freq == 0:
            from hyper_mve.eval import Evaluator
            evaluator = Evaluator(trainer)
            eval_results = evaluator.eval_at_stage(global_step)
            log_to_tensorboard(eval_results, global_step)
        
        # 4. checkpoint
        if global_step % 10_000 == 0:
            trainer.save_checkpoint(f"{args.ckpt_dir}/step_{global_step}.pt")
```

---

## 7. 契约 6：checkpoint 格式（D8 + R5-9）

### 7.1 v4 checkpoint 字段（D8 breaking）

| 字段 | 类型 | 用途 |
|------|------|------|
| `version` | str | 必须为 "v4"，区分 v4.7 |
| `global_step` | int | resume 起点 |
| `model_state` | dict | model.state_dict() |
| `target_state` | dict | target_model.state_dict() |
| `optimizer_state` | dict | optimizer.state_dict() |
| `lr_scheduler_state` | dict | lr_scheduler.state_dict() |
| `cfg` | dict | V4Config.to_dict() 完整序列化（resume 时校验）|

### 7.2 v4 → v4.7 不兼容声明（R5-9）

- v4 trainer 加载 v4.7 checkpoint → RuntimeError（spec 01 §2.2 已规范）
- v4.7 trainer（在 `_legacy_v4_7/` 内）**不**能加载 v4 checkpoint
- 若需切换：`git checkout v4.7-final` 切到对应版本

### 7.3 兼容 shim（read-only load 老 checkpoint，仅供审计）

Pkg-01 spec 07 已规范 `_legacy_v4_7/` 归档目录。Pkg-05 trainer 提供 read-only shim：

```python
# 仅审计用（如 v4.7 实验复现），不修改 trainer 主流程
from hyper_mve._legacy_v4_7.training import load_v47_checkpoint_readonly

state = load_v47_checkpoint_readonly("path/to/v47_ckpt.pt")
# state 仅供查看，不能 load 到 v4 trainer
```

实施层：v4 trainer.load_checkpoint 检测 version 字段，非 "v4" 时抛 RuntimeError + 引导用户切到 v4.7 shim。

---

## 8. 契约 7：cfg 字段完备性（M4）

按 spec 01 §1.1 已列 27 项 cfg 字段，Pkg-05 trainer.__init__ 内 `_validate_cfg_fields` 全部读取并校验。任何字段缺失或重命名需 Pkg-01 spec 05 同步。

**集成测试**：`tests/training/test_trainer_loop.py::test_validate_cfg_fields` 启动时校验。

---

## 9. 集成测试 checklist（Pkg-05 → Pkg-06/07/08 联调）

Pkg-05 PR merge 前应通过以下集成测试：

| # | 测试 | 责任 spec |
|---|------|----------|
| 1 | trainer 4 API + worker 2 API + buffer 3 API + scheduler 4 API 全部稳定 | spec 01-04 + §1 |
| 2 | v4.7 → v4 11 处调用点迁移完成（grep 验证 0 残留，澄清 2 精确正则）| §2.3 |
| 3 | trainer.train_step 100 步无 NaN（Medium config）| spec 01 §5.2 |
| 4 | worker.collect_episode TimeStepRecord 字段合规 + z_hat 顺序一致 | spec 02 §5.1 |
| 5 | buffer.store_episode + sample_batch roundtrip 正确 | spec 03 §5.1 |
| 6 | CurriculumScheduler 3 stage 切换边界 | spec 04 §5.1 |
| 7 | compose_total_loss 双路径 backward（pre-5K BeliefNet grad 仅 belief + post-5K 两源）| spec 05 §5.1 |
| 8 | MVE planner CRN seed 相同 step 0 输出一致 | spec 06 §5.1 |
| 9 | EMA target tau=0.99 衰减 + warmup_cosine LR 曲线 | spec 07 §5.1 |
| 10 | scripts/train_main.py --variant hyper --max_steps 100 端到端 | §6 |
| 11 | checkpoint v4 save/load roundtrip + v4.7 加载抛 RuntimeError | §7 |
| 12 | Pkg-04 spec 04 联动: pre-5K BeliefNet 参数梯度 = 0 | spec 05 §5.1 + Pkg-04 spec 04 §5.1 |
| 13 | Self-Info 严格性：worker 不传 oracle types 进 model.set_context_subjective | spec 02 §5.1 + Pkg-04 spec 02 §5.1 |

---

## 10. Cross-references

- 全部 7 个 Pkg-05 specs
- Pkg-01 spec 04 (TimeStepRecord 12 字段 + z_hat 顺序) + spec 05 (TrainConfig 27 字段)
- Pkg-02 spec 08 (env.info 三段分组：Public / Oracle / EvalOnly)
- Pkg-03 spec 04 (BeliefNet.step / forward) + spec 06 (belief_loss) + spec 08 §4 (课程接入)
- Pkg-04 spec 02 (model 7 API + §2.3 trainer/worker 调用模板) + spec 04 (grad gating 双层 detach) + spec 08 (trainer 6 + worker 5 + planner 4 迁移指引 + 17 cfg 字段穷举)
- Pkg-06 spec 01 (shared_backbones — 待 Pkg-06 SDD)
- Pkg-07 spec 04 (self-info-eval — 待 Pkg-07 SDD)
- Pkg-08 spec 01 (ablation 实验设计 — 待 Pkg-08 SDD)
- v4.7 行号映射:
  - `training/muzero_trainer.py`: L56-83/138-181/203-430/432-598/85-130（v4.7 → v4 整体重写）
  - `training/worker.py`: L84-180 + L125/128/143/145/149（5 处迁移）
  - `training/episode_buffer.py`: L72-215（重写为 TimeStepRecord 容器）
  - `planning/mve_planner.py`: L80-98/128-290 + L71/213/219/258（4 处迁移）
- v4.7 → v4 替换策略（D10）: inplace 修改 + `_legacy_v4_7/` 归档 + `git tag v4.7-final`
