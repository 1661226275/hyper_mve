# Spec 04: Curriculum Scheduler — 3 Stage + λ_b + oracle_z_mixing_weight

> 父文档：[`../proposal.md`](../proposal.md) §2.2.1 · [`../design.md`](../design.md) §3 D4 · §6.5
> **D4 实例可替换**：Pkg-08 ablation 4 课程边界扫描 → trainer.__init__ 接受自定义 scheduler 实例（review 修订 1）。

---

## 1. Purpose

按 Ch5.7 + Pkg-03 spec 08 §4 课程接入路径实现 `CurriculumScheduler`，lightweight 类（仅 step → stage 映射），提供：

| 方法 | 用途 | 调用方 |
|------|------|--------|
| `stage(global_step)` | 'stage_1' / 'stage_2' / 'stage_3' | trainer + Pkg-07 eval（评估期查询当前 stage）|
| `oracle_z_mixing_weight(global_step)` | Stage 1: 1.0 / Stage 2 anneal / Stage 3: 0.0 | compose_total_loss 内部 |
| `lambda_b(global_step)` | L_belief 课程加权系数 | compose_total_loss 内部 |
| `build_oracle_z_seq(types_true)` | 包装 Pkg-03 同名函数（Stage 1/2 注入路径） | compose_total_loss 内部 |

---

## 2. Interface

### 2.1 文件路径

`hyper_mve/training/curriculum.py`（新增）

### 2.2 类签名

```python
import torch
from typing import Optional
from hyper_mve.configs import V4Config
from hyper_mve.models.belief_losses import build_oracle_z_seq as _build_oracle_z_seq


class CurriculumScheduler:
    """3 Stage 课程: Pure Oracle → Anneal → Pure Inference (Ch5.7).
    
    Lightweight class - 仅 step → stage 映射 + 三个 query 方法.
    无内部 stateful 字段 (除 cfg 读取); D4 实例可替换设计.
    
    Stage 边界 (默认 200K max_train_steps):
        Stage 1 (Pure Oracle):    step < 0.3 × max_train_steps = 60K
        Stage 2 (Anneal):         60K ≤ step < 0.7 × max_train_steps = 140K
        Stage 3 (Pure Inference): step ≥ 140K
    """
    
    def __init__(self, cfg: V4Config):
        self.cfg = cfg
        self.max_steps = cfg.train.max_train_steps
        self.stage_1_end = int(cfg.train.curriculum_stage_1_end_frac * self.max_steps)
        self.stage_2_end = int(cfg.train.curriculum_stage_2_end_frac * self.max_steps)
        
        # 放宽断言 (P0-1 修订): 允许等号边界以支持 spec 08 §6.2 oracle_only / infer_only variant
        #   - oracle_only: stage_1_end = stage_2_end = max_steps (永远 Stage 1)
        #   - infer_only:  stage_1_end = stage_2_end = 0          (永远 Stage 3)
        #   - 默认 medium: 0 < stage_1_end (60K) < stage_2_end (140K) < max_steps (200K)
        # 顺序约束仍存在: 0 <= stage_1_end <= stage_2_end <= max_steps
        assert 0 <= self.stage_1_end <= self.stage_2_end <= self.max_steps, (
            f"Stage 边界顺序错: 0 <= {self.stage_1_end} <= {self.stage_2_end} <= {self.max_steps}. "
            f"oracle_only/infer_only 等极端 variant 允许等号边界, 但必须保持单调."
        )
    
    # ====================================================================
    # API 1: stage (C5-S1 边界判断)
    # ====================================================================
    
    def stage(self, global_step: int) -> str:
        """返回当前 stage 字符串 'stage_1' / 'stage_2' / 'stage_3'.
        
        P0-1 修订: 边界等号情况:
            - oracle_only (stage_1_end == max_steps): 所有 step 都返回 'stage_1'
            - infer_only  (stage_2_end == 0):         所有 step 都返回 'stage_3'
            - 退化 stage_1_end == stage_2_end:        无 'stage_2', 直接 stage_1 → stage_3
        """
        # P0-1 修订: 边界等号优先处理
        if self.stage_1_end == self.max_steps:
            return "stage_1"  # oracle_only
        if self.stage_2_end == 0:
            return "stage_3"  # infer_only
        
        # 正常分支
        if global_step < self.stage_1_end:
            return "stage_1"
        elif global_step < self.stage_2_end:
            return "stage_2"
        else:
            return "stage_3"
    
    # ====================================================================
    # API 2: oracle_z_mixing_weight (C5-S2 + C5-S3)
    # ====================================================================
    
    def oracle_z_mixing_weight(self, global_step: int) -> float:
        """oracle 注入权重 ∈ [0.0, 1.0].
        
        Stage 1: 1.0  (100% oracle, C5-S2)
        Stage 2: 1.0 → 0.0 linear anneal (C5-S3 单调下降)
        Stage 3: 0.0  (纯推断)
        
        混合公式 (compose_total_loss 调用):
            z_for_main = w * oracle_z + (1 - w) * predicted_z
        
        P0-1 修订: 边界 stage_1_end == stage_2_end 时按 stage 分流, 避免除零:
            - oracle_only (frac=1.0/1.0): stage(step) == 'stage_2' 但 anneal 区间为 0
              → 不进入 anneal 分支; 由 'stage_3' 兜底返回 0.0?  ❌
              正确语义: 永远 Stage 1, weight = 1.0. 故需先判 stage_1_end == stage_2_end == max_steps.
            - infer_only (frac=0.0/0.0): 永远 Stage 3, weight = 0.0.
        """
        # P0-1 修订: 边界等号情况优先处理 (防止 anneal 区间为 0 时除零 + stage 语义混乱)
        if self.stage_1_end == self.max_steps:
            # oracle_only variant: 永远 Stage 1
            return 1.0
        if self.stage_2_end == 0:
            # infer_only variant: 永远 Stage 3
            return 0.0
        if self.stage_1_end == self.stage_2_end:
            # 退化: 无 Stage 2 anneal 区间 (如 frac=0.3/0.3) — 直接阶跃 1.0 → 0.0
            return 1.0 if global_step < self.stage_1_end else 0.0
        
        # 正常分支
        if global_step < self.stage_1_end:
            return 1.0
        elif global_step < self.stage_2_end:
            # Linear anneal: stage_1_end → stage_2_end 对应 1.0 → 0.0
            progress = (global_step - self.stage_1_end) / (self.stage_2_end - self.stage_1_end)
            return float(1.0 - progress)
        else:
            return 0.0
    
    # ====================================================================
    # API 3: lambda_b (C5-L1 L_belief 课程加权)
    # ====================================================================
    
    def lambda_b(self, global_step: int) -> float:
        """L_belief 课程加权系数 ∈ [0.0, cfg.train.w_belief].
        
        策略 (与 Ch5.7 + Pkg-03 spec 08 §4.2 一致):
            Stage 1 (Pure Oracle): w_belief 全权重 (BeliefNet 主要由 L_belief 训练)
            Stage 2 (Anneal):      w_belief 保持 (main loss + L_belief 联合训练)
            Stage 3 (Pure Inference): w_belief 保持 (BeliefNet 持续优化)
        
        默认: 全程返回 cfg.train.w_belief (=1.0).
        Pkg-08 ablation 可自定义曲线 (子类化 CurriculumScheduler).
        """
        return self.cfg.train.w_belief
    
    # ====================================================================
    # API 4: build_oracle_z_seq (Pkg-03 接入)
    # ====================================================================
    
    def build_oracle_z_seq(
        self,
        types_true: torch.Tensor,         # (B, T, N) int8 from batch["tau"]
    ) -> torch.Tensor:                    # (B, T, N, N-1, 2) one-hot oracle
        """构造 oracle z_seq 注入 BeliefNet.forward.
        
        包装 Pkg-03 `belief_losses.build_oracle_z_seq(types_true)`.
        scheduler 提供此 API 是为了让 compose_total_loss 仅与 scheduler 交互
        (而不直接 import Pkg-03 函数), 便于 Pkg-08 ablation 替换.
        """
        return _build_oracle_z_seq(types_true)
    
    # ====================================================================
    # 状态序列化 (D4 + spec 01 §3.5)
    # ====================================================================
    
    def state_dict(self) -> dict:
        """序列化 scheduler 状态.
        
        当前实现无 stateful 字段 (lightweight), 返回空 dict.
        Pkg-08 ablation 子类如有状态需 override.
        """
        return {}
    
    def load_state_dict(self, state: dict) -> None:
        """反序列化 (lightweight 实现无内部状态, no-op)."""
        pass
```

---

## 3. Implementation Notes

### 3.1 lightweight 设计（D4）

CurriculumScheduler 是**纯 query 类**，不持有 stateful 字段（除 cfg 读取）：
- 所有方法都是 step → 结果 的纯函数
- 不持有 global_step 字段（由 trainer 持有，每次调用传入）
- 无 internal counters / accumulators

好处：
- Pkg-08 ablation 替换实例时不需要状态迁移
- 单测覆盖容易（所有方法都是纯函数）
- Pkg-07 eval 期复用同一 scheduler 实例（查当前 stage）

### 3.2 Stage 边界配置（C5-S1）

| 字段 | 默认 | 用途 |
|------|------|------|
| `cfg.train.max_train_steps` | 200_000 | 边界分母 |
| `cfg.train.curriculum_stage_1_end_frac` | 0.3 | Stage 1 上界 (60K) |
| `cfg.train.curriculum_stage_2_end_frac` | 0.7 | Stage 2 上界 (140K) |

边界检查在 __init__ 时一次性完成（assert）。

**Pkg-08 ablation** 可改 frac：
```python
from dataclasses import replace
cfg_ablation = replace(cfg, train=replace(cfg.train,
    curriculum_stage_1_end_frac=0.1,      # 更短 Stage 1
    curriculum_stage_2_end_frac=0.5,      # 更早进 Stage 3
))
scheduler_ablation = CurriculumScheduler(cfg_ablation)
trainer = MuZeroTrainer(cfg, model, scheduler=scheduler_ablation)
```

### 3.3 oracle_z_mixing_weight Linear Anneal（C5-S3）

Stage 2 anneal 严格单调下降，linear 曲线：

```
weight(step) = 1.0 - (step - stage_1_end) / (stage_2_end - stage_1_end)
            = 1.0 at step = stage_1_end
            = 0.0 at step = stage_2_end
```

边界点：
- step = stage_1_end（如 60K）: weight = 1.0（仍 Stage 1 末尾全 oracle）
- step = stage_2_end - 1（如 139999）: weight ≈ 0.0（Stage 2 末尾）
- step = stage_2_end（如 140K）: weight = 0.0（已 Stage 3）

**实施细节**: stage(step) 在 step = stage_1_end 时返回 "stage_2"（边界归 Stage 2），与 oracle_z_mixing_weight 一致（边界 weight = 1.0 但已属 Stage 2）。

### 3.4 与 Pkg-03 build_oracle_z_seq 协作

Pkg-03 spec 06 / spec 08 §4.3 已定义：
```python
def build_oracle_z_seq(types_true: torch.Tensor) -> torch.Tensor:
    """types_true (B, T, N) int → z_oracle (B, T, N, N-1, 2) one-hot."""
```

scheduler 仅是 pass-through 包装。compose_total_loss 调用模式：
```python
# compose_total_loss 内
oracle_z_seq = scheduler.build_oracle_z_seq(batch["tau"])
c_hat, z_hat = model.belief_net.forward(
    batch["obs"],
    oracle_z_seq=oracle_z_seq,
    oracle_mixing_weight=scheduler.oracle_z_mixing_weight(global_step),
)
```

具体 BeliefNet.forward 如何使用 oracle_z_seq + mixing_weight 见 Pkg-03 spec 04 §4。

### 3.5 lambda_b 默认全程恒定（与 Ch5.7 一致）

当前默认 `lambda_b(step) = cfg.train.w_belief = 1.0` 全程恒定。原因：
- Pkg-03 spec 08 §4.2 已规定 BeliefNet 全程独立训练（L_belief 始终启用）
- 课程主要通过 oracle_z_mixing_weight 切换（不通过 lambda_b 调权）
- 简化设计，避免双重曲线交互（mixing + lambda）

**Pkg-08 ablation** 若需 lambda_b 曲线变化（如 Stage 1 0.5 / Stage 2 1.0 / Stage 3 1.5）：子类化覆盖即可：

```python
class CustomCurriculumScheduler(CurriculumScheduler):
    def lambda_b(self, step):
        stage = self.stage(step)
        return {"stage_1": 0.5, "stage_2": 1.0, "stage_3": 1.5}[stage]
```

### 3.6 与 trainer.scheduler 接入（修订 1）

trainer.__init__ 接受 `scheduler` 参数：
```python
def __init__(self, cfg, model, scheduler=None, ...):
    self.scheduler = scheduler or CurriculumScheduler(cfg)
```

compose_total_loss 通过 `trainer.scheduler` 访问：
```python
def compose_total_loss(model, batch, trainer, global_step, cfg):
    sched = trainer.scheduler
    mixing_w = sched.oracle_z_mixing_weight(global_step)
    lambda_b = sched.lambda_b(global_step)
    ...
```

---

## 4. Edge Cases（P0-1 修订）

| 场景 | 行为 |
|------|------|
| global_step < 0 | stage() 走正常分支返回 "stage_1"（< stage_1_end 总成立），weight = 1.0 |
| global_step >= max_steps | stage() 返回 "stage_3"，weight = 0.0 |
| **oracle_only**（stage_1_end == max_steps）| stage() 恒返回 "stage_1"；weight 恒 1.0（P0-1 修订：边界等号优先分支处理）|
| **infer_only**（stage_2_end == 0）| stage() 恒返回 "stage_3"；weight 恒 0.0（同上）|
| stage_1_end == stage_2_end（中间值，如 frac=0.3/0.3）| 无 Stage 2 anneal 区间；stage() 跳过 "stage_2"；weight 阶跃 1.0 → 0.0（无 anneal）|
| stage_1_end > stage_2_end | __init__ assert 抛（必须单调）|
| max_train_steps = 0 | stage_1_end = 0 但 max_steps = 0 → 满足 `0 <= 0 <= 0 <= 0` → 退化为 infer_only（恒 stage_3）|
| stage_1_end / stage_2_end / max_steps 全 0 | 退化为 infer_only |

---

## 5. Acceptance Criteria

### 5.1 单元测试（`tests/training/test_curriculum.py`）

```python
import pytest
import torch
from dataclasses import replace
from hyper_mve.configs import V4Config
from hyper_mve.training.curriculum import CurriculumScheduler


@pytest.fixture
def cfg_medium():
    return V4Config.from_preset("medium")


@pytest.fixture
def scheduler(cfg_medium):
    return CurriculumScheduler(cfg_medium)


# ====== C5-S1: 边界判断 ======

def test_curriculum_stage_boundaries(scheduler, cfg_medium):
    """3 stage 切换边界正确."""
    max_s = cfg_medium.train.max_train_steps  # 200_000
    s1_end = int(cfg_medium.train.curriculum_stage_1_end_frac * max_s)  # 60_000
    s2_end = int(cfg_medium.train.curriculum_stage_2_end_frac * max_s)  # 140_000
    
    # Stage 1 范围
    assert scheduler.stage(0) == "stage_1"
    assert scheduler.stage(s1_end - 1) == "stage_1"
    
    # Stage 2 范围
    assert scheduler.stage(s1_end) == "stage_2"
    assert scheduler.stage(s2_end - 1) == "stage_2"
    
    # Stage 3 范围
    assert scheduler.stage(s2_end) == "stage_3"
    assert scheduler.stage(max_s) == "stage_3"
    assert scheduler.stage(max_s + 10000) == "stage_3"


# ====== C5-S2: Stage 1 100% oracle ======

def test_stage_1_full_oracle(scheduler, cfg_medium):
    """Stage 1 oracle weight 全 = 1.0."""
    s1_end = int(cfg_medium.train.curriculum_stage_1_end_frac * cfg_medium.train.max_train_steps)
    
    for step in [0, 1000, s1_end // 2, s1_end - 1]:
        assert scheduler.oracle_z_mixing_weight(step) == 1.0


# ====== C5-S3: Stage 2 anneal 单调下降 ======

def test_oracle_mixing_anneal_monotonic(scheduler, cfg_medium):
    """Stage 2 内 oracle weight 严格单调下降."""
    max_s = cfg_medium.train.max_train_steps
    s1_end = int(cfg_medium.train.curriculum_stage_1_end_frac * max_s)
    s2_end = int(cfg_medium.train.curriculum_stage_2_end_frac * max_s)
    
    steps = list(range(s1_end, s2_end, (s2_end - s1_end) // 10))
    weights = [scheduler.oracle_z_mixing_weight(s) for s in steps]
    
    # 严格单调下降
    for w0, w1 in zip(weights[:-1], weights[1:]):
        assert w0 > w1, f"weight 非单调: {w0} -> {w1}"
    
    # 起点 1.0, 终点 ~0.0
    assert weights[0] == 1.0
    assert weights[-1] < 0.15  # 接近 0


def test_oracle_mixing_anneal_continuous(scheduler, cfg_medium):
    """Stage 2 anneal 是 linear 连续 (非阶跃, R5-4 缓解)."""
    max_s = cfg_medium.train.max_train_steps
    s1_end = int(cfg_medium.train.curriculum_stage_1_end_frac * max_s)
    s2_end = int(cfg_medium.train.curriculum_stage_2_end_frac * max_s)
    
    # 步长 1 验证连续性
    for step in range(s1_end, min(s1_end + 100, s2_end)):
        w_prev = scheduler.oracle_z_mixing_weight(step - 1)
        w_curr = scheduler.oracle_z_mixing_weight(step)
        # 单步变化应小（linear, 总变化 1.0 / (s2_end - s1_end) ≈ 1/80000）
        assert abs(w_curr - w_prev) < 0.01


def test_stage_3_zero_oracle(scheduler, cfg_medium):
    """Stage 3 oracle weight 全 = 0.0."""
    s2_end = int(cfg_medium.train.curriculum_stage_2_end_frac * cfg_medium.train.max_train_steps)
    
    for step in [s2_end, s2_end + 10000, cfg_medium.train.max_train_steps]:
        assert scheduler.oracle_z_mixing_weight(step) == 0.0


# ====== C5-L1: lambda_b 曲线 ======

def test_lambda_b_curve_matches_cfg(scheduler, cfg_medium):
    """lambda_b 与 cfg.train.w_belief 一致 (默认全程恒定)."""
    expected = cfg_medium.train.w_belief
    for step in [0, 50_000, 100_000, 150_000, 200_000]:
        assert scheduler.lambda_b(step) == expected


# ====== build_oracle_z_seq 包装 ======

def test_build_oracle_z_seq_passes_through(scheduler):
    """scheduler.build_oracle_z_seq 是 Pkg-03 函数的纯包装."""
    types_true = torch.tensor([[[0, 0, 1, 1]] * 5] * 2, dtype=torch.int8)  # (B=2, T=5, N=4)
    
    oracle_z_seq = scheduler.build_oracle_z_seq(types_true)
    
    # shape (B, T, N, N-1, 2)
    assert oracle_z_seq.shape == (2, 5, 4, 3, 2)
    # one-hot 检查（每行和为 1）
    assert torch.allclose(oracle_z_seq.sum(dim=-1), torch.ones_like(oracle_z_seq[..., 0]))


# ====== D4 实例可替换 ======

def test_scheduler_subclass_override():
    """子类可 override lambda_b 实现自定义课程曲线."""
    class StageBasedLambdaScheduler(CurriculumScheduler):
        def lambda_b(self, step):
            stage = self.stage(step)
            return {"stage_1": 0.5, "stage_2": 1.0, "stage_3": 1.5}[stage]
    
    cfg = V4Config.from_preset("medium")
    sched = StageBasedLambdaScheduler(cfg)
    
    assert sched.lambda_b(0) == 0.5  # Stage 1
    s1_end = int(cfg.train.curriculum_stage_1_end_frac * cfg.train.max_train_steps)
    assert sched.lambda_b(s1_end) == 1.0  # Stage 2
    s2_end = int(cfg.train.curriculum_stage_2_end_frac * cfg.train.max_train_steps)
    assert sched.lambda_b(s2_end) == 1.5  # Stage 3


# ====== 配置错误 ======

def test_invalid_stage_boundaries():
    """Stage 边界顺序错应抛 AssertionError."""
    cfg = V4Config.from_preset("medium")
    cfg_bad = replace(cfg, train=replace(cfg.train,
        curriculum_stage_1_end_frac=0.7,    # > stage_2_end_frac
        curriculum_stage_2_end_frac=0.3,
    ))
    with pytest.raises((AssertionError, ValueError)):
        CurriculumScheduler(cfg_bad)


# ====== P0-1 修订: oracle_only / infer_only 极端 variant 边界 ======

def test_oracle_only_variant():
    """oracle_only (frac=1.0/1.0): 永远 Stage 1, weight 恒 1.0.
    
    P0-1 修订: 验证 spec 08 §6.2 oracle_only variant 与 spec 04 assert 兼容.
    """
    cfg = V4Config.from_preset("medium")
    cfg_oracle = replace(cfg, train=replace(cfg.train,
        curriculum_stage_1_end_frac=1.0,
        curriculum_stage_2_end_frac=1.0,
    ))
    # P0-1: assert 不应抛 (放宽为 <=)
    sched = CurriculumScheduler(cfg_oracle)
    
    # 所有 step 都 stage_1 + weight 1.0
    for step in [0, 100, 50_000, cfg.train.max_train_steps, cfg.train.max_train_steps + 10000]:
        assert sched.stage(step) == "stage_1", f"oracle_only at step {step} should be stage_1"
        assert sched.oracle_z_mixing_weight(step) == 1.0, f"oracle_only at step {step} should be 1.0"


def test_infer_only_variant():
    """infer_only (frac=0.0/0.0): 永远 Stage 3, weight 恒 0.0.
    
    P0-1 修订: 验证 spec 08 §6.2 infer_only variant 与 spec 04 assert 兼容.
    """
    cfg = V4Config.from_preset("medium")
    cfg_infer = replace(cfg, train=replace(cfg.train,
        curriculum_stage_1_end_frac=0.0,
        curriculum_stage_2_end_frac=0.0,
    ))
    # P0-1: assert 不应抛 (放宽为 <=)
    sched = CurriculumScheduler(cfg_infer)
    
    # 所有 step 都 stage_3 + weight 0.0
    for step in [0, 100, 50_000, cfg.train.max_train_steps]:
        assert sched.stage(step) == "stage_3", f"infer_only at step {step} should be stage_3"
        assert sched.oracle_z_mixing_weight(step) == 0.0, f"infer_only at step {step} should be 0.0"


def test_degenerate_stage_1_eq_stage_2():
    """stage_1_end == stage_2_end (中间值, 如 frac=0.3/0.3): 无 Stage 2 anneal, 阶跃跳变.
    
    P0-1 修订: 防除零, anneal 区间为 0 时按 step < stage_1_end 阶跃返回 1.0 / 0.0.
    """
    cfg = V4Config.from_preset("medium")
    cfg_step = replace(cfg, train=replace(cfg.train,
        curriculum_stage_1_end_frac=0.3,
        curriculum_stage_2_end_frac=0.3,    # == stage_1_end
    ))
    sched = CurriculumScheduler(cfg_step)
    s1_end = int(0.3 * cfg.train.max_train_steps)  # 60_000
    
    # 阶跃: step < s1_end → 1.0, step >= s1_end → 0.0 (跳过 stage_2)
    assert sched.oracle_z_mixing_weight(s1_end - 1) == 1.0
    assert sched.oracle_z_mixing_weight(s1_end) == 0.0
    
    # stage() 应不返回 stage_2 (退化无中间段)
    assert sched.stage(s1_end) == "stage_3"


def test_no_div_by_zero_anneal():
    """oracle_z_mixing_weight 在所有等号边界配置下不应抛 ZeroDivisionError.
    
    P0-1 修订关键: 防御 anneal 区间分母为 0.
    """
    cfg = V4Config.from_preset("medium")
    
    # 7 种边界配置
    boundary_configs = [
        (0.0, 0.0),    # infer_only
        (0.0, 0.5),    # 仅 stage_1_end = 0
        (1.0, 1.0),    # oracle_only
        (0.5, 1.0),    # 仅 stage_2_end = max
        (0.3, 0.3),    # 退化中间
        (0.0, 1.0),    # 极端: Stage 1 = 0%, Stage 2 = 100%
        (0.5, 0.5),    # 退化中间
    ]
    for f1, f2 in boundary_configs:
        cfg_b = replace(cfg, train=replace(cfg.train,
            curriculum_stage_1_end_frac=f1,
            curriculum_stage_2_end_frac=f2,
        ))
        sched = CurriculumScheduler(cfg_b)
        # 不应抛任何异常 (特别是 ZeroDivisionError)
        for step in [0, 50_000, 100_000, 200_000]:
            w = sched.oracle_z_mixing_weight(step)
            assert 0.0 <= w <= 1.0, f"weight {w} 越界 at frac=({f1},{f2}) step={step}"


# ====== state_dict roundtrip ======

def test_state_dict_roundtrip(scheduler):
    state = scheduler.state_dict()
    assert isinstance(state, dict)
    # lightweight 实现 state 为空
    assert state == {}
    
    scheduler.load_state_dict(state)  # no-op 不应抛
```

### 5.2 集成测试

`scripts/train_main.py --preset medium --max_steps 1000` 端到端：
- 训练循环开始时 stage = "stage_1"
- 训练循环结束时 stage 仍 = "stage_1"（因 max_steps=1000 << curriculum_stage_1_end_frac × 200K = 60K）
- 长训练（max_steps=200K）应观察到 stage 切换

### 5.3 性能要求

- 所有 query 方法（stage / oracle_z_mixing_weight / lambda_b）单调用 < 1 μs（纯算术）
- build_oracle_z_seq 性能由 Pkg-03 决定（spec 06 验收）

---

## 6. v4.7 → v4 对比

| 维度 | v4.7 | v4 | 差异 |
|------|------|----|------|
| 课程实现 | trainer 内联 if-else | CurriculumScheduler 独立类（D4）| 抽出 |
| stage 概念 | 隐式（depth_warmup_steps 等单一阈值）| 显式 3 stage（Pure Oracle / Anneal / Pure Inference）| 明确化 |
| oracle 注入 | n/a（v4.7 Oracle 模型给真值，Infer 模型不注入）| oracle_z_mixing_weight 显式控制 | 新增 |
| ablation 替换 | 不支持 | 实例可替换（D4） | 新增 |

---

## 7. Cross-references

- Ch5.7 课程 3 stage
- `01-trainer-loop-v2.md`（trainer 持有 self.scheduler）
- `05-loss-composition.md`（compose_total_loss 调 scheduler 三方法）
- `08-integration-contracts.md` §3（与 Pkg-07 eval 期复用 + Pkg-08 ablation 替换）
- Pkg-01 spec 05 TrainConfig（curriculum_stage_1_end_frac / stage_2_end_frac / w_belief）
- Pkg-03 spec 06 build_oracle_z_seq
- Pkg-03 spec 08 §4 课程接入路径
