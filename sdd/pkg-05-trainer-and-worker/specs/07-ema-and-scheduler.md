# Spec 07: EMA Target Model + LR Scheduler 实施细节

> 父文档：[`../design.md`](../design.md) §3 D6 · §6.2
> **D6 内联**：EMA + LR scheduler 是 MuZeroTrainer 内部辅助方法，不抽出独立模块；本 spec 仅文档化行为与单测，不新增文件。

---

## 1. Purpose

按 v4.7 muzero_trainer.py L85-130 (_build_scheduler) + L406 (_update_ema_target) 保留 + v4 cfg.train.ema_tau / lr_schedule 配置化实施：

| 内部方法 | 用途 | 调用频率 |
|---------|------|---------|
| `_build_scheduler()` | 在 trainer.__init__ 末尾构造 self.lr_scheduler | 一次性 |
| `_update_ema_target()` | 每 train_step 末尾调（optimizer.step 之后） | 每 train_step |

均为 MuZeroTrainer 私有方法（无对外稳定性承诺）。

---

## 2. Interface（trainer 内部实施细节）

### 2.1 EMA target model 更新

```python
# muzero_trainer.py 内部
def _update_ema_target(self) -> None:
    """EMA target model 更新 (v4.4 保留, Pkg-05 spec 07).
    
    公式: param_target ← τ · param_target + (1 - τ) · param_online
        默认 τ = cfg.train.ema_tau = 0.99
    
    调用时机: optimizer.step() 之后, scheduler.step() 之后.
    """
    with torch.no_grad():
        tau = self.cfg.train.ema_tau
        for p_online, p_target in zip(
            self.model.parameters(),
            self.target_model.parameters(),
        ):
            p_target.data.mul_(tau).add_(p_online.data, alpha=1.0 - tau)
```

### 2.2 LR Scheduler 构造（v4.7 L85-130 保留 + configurable）

```python
def _build_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
    """LR scheduler 构造 (v4.7 _build_scheduler 保留, 3 种策略).
    
    cfg.train.lr_schedule ∈ {"warmup_cosine", "cosine", "multistep"}:
        warmup_cosine (默认): 0 → lr (linear warmup over warmup_steps) → lr_min (cosine anneal over (max_steps - warmup_steps))
        cosine: lr → lr_min (cosine anneal over max_steps)
        multistep: lr * gamma^k at milestones (v4.7 默认 milestones [60000, 120000], gamma 0.5)
    
    Returns: torch.optim.lr_scheduler._LRScheduler
    """
    cfg_t = self.cfg.train
    
    if cfg_t.lr_schedule == "warmup_cosine":
        # 自定义 LambdaLR 实现 warmup + cosine anneal
        from torch.optim.lr_scheduler import LambdaLR
        import math
        
        def lr_lambda(step):
            if step < cfg_t.lr_warmup_steps:
                # Linear warmup: 0 → 1.0
                return step / cfg_t.lr_warmup_steps
            else:
                # Cosine anneal: 1.0 → lr_min/lr
                progress = (step - cfg_t.lr_warmup_steps) / (cfg_t.max_train_steps - cfg_t.lr_warmup_steps)
                progress = min(progress, 1.0)
                min_ratio = cfg_t.lr_min / cfg_t.lr
                return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    elif cfg_t.lr_schedule == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(
            self.optimizer,
            T_max=cfg_t.max_train_steps,
            eta_min=cfg_t.lr_min,
        )
    
    elif cfg_t.lr_schedule == "multistep":
        from torch.optim.lr_scheduler import MultiStepLR
        # v4.7 默认 milestones (cfg 字段可扩展)
        milestones = getattr(cfg_t, "lr_milestones", [60000, 120000])
        gamma = getattr(cfg_t, "lr_gamma", 0.5)
        return MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)
    
    else:
        raise ValueError(f"Unknown lr_schedule: {cfg_t.lr_schedule}")
```

---

## 3. Implementation Notes

### 3.1 调度顺序（spec 01 §2.3 train_step 内）

```python
# muzero_trainer.train_step 内
self.optimizer.zero_grad(set_to_none=True)
losses["total"].backward()
torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
self.optimizer.step()        # 1. optimizer 更新 online model
self.lr_scheduler.step()     # 2. LR scheduler 推进
self._update_ema_target()    # 3. EMA target 跟随 online (用更新后的 online)
```

**顺序关键**：
- EMA 必须在 optimizer.step() 之后（确保 online 已被 Adam 更新）
- LR scheduler.step() 顺序无关紧要（用上一步的 LR 算了 backward，下一步会用新 LR）

### 3.2 EMA tau=0.99 的衰减率含义

`p_target ← 0.99 · p_target + 0.01 · p_online`

| step | online 变化 | target 滞后 |
|------|------------|-------------|
| 1 | 全新 | 99% 旧值 + 1% 新值 |
| 10 | 持续变 | (0.99)^10 ≈ 90.4% 旧值权重 |
| 100 | – | (0.99)^100 ≈ 36.6% 旧值权重 → 半衰期 ~69 步 |
| 500 | – | (0.99)^500 ≈ 0.6% 旧值权重 → 完全跟随 |

target 滞后 ~70 步收敛，适合 5-step unroll bootstrap（target 比 online 慢但不远，避免 V over-estimation）。

### 3.3 warmup_cosine LR 曲线（v4.7 默认 + Pkg-01 spec 05）

cfg 默认：
- `lr = 1e-4`
- `lr_min = 5e-6`
- `lr_warmup_steps = 5000`
- `max_train_steps = 200_000`

曲线：
| step | lr |
|------|------|
| 0 | 0 |
| 2500 | 0.5 × 1e-4 = 5e-5（warmup 中点）|
| 5000 | 1e-4（warmup 结束）|
| 50_000 | ~1e-4 · 0.91 ≈ 9.1e-5（cosine 中期）|
| 100_000 | ~5.05e-5（cosine 50%）|
| 150_000 | ~1.5e-5 |
| 200_000 | 5e-6 = lr_min |

### 3.4 与 model.belief_grad_gating_steps 的协调

Pkg-04 spec 04 belief gradient gating 阈值默认 5000，与 lr_warmup_steps=5000 巧合相同：
- 前 5000 step warmup + BeliefNet 梯度切断 → main loss 仅训练 model 部分
- 5000 step 后 LR 达 peak + BeliefNet 联合训练
- 与 Ch5.7 课程 Stage 1 (前 60K = 0.3 × 200K) 内部嵌套（前 5K 是 Stage 1 + warmup 重叠期）

### 3.5 EMA target model 与 set_context_objective / set_context_subjective 同步

spec 01 §3.2 已强调：target_model 是 model 的 deep copy，在 train_step 内**也**需要调 set_context_*（与 online 平行）：

```python
# spec 05 compose_total_loss 内
model.set_context_objective(c_t)
target_model.set_context_objective(c_t)    # ← 同步

for k in range(N):
    model.set_context_subjective(k, cap, (c_hat, z_hat))
    target_model.set_context_subjective(k, cap, (c_hat.detach(), z_hat.detach()))  # ← detach (target 不反传)
```

target 内部 grad_gating 仍按 step 执行（即使 detach 入参也不影响），但因 target_model 在 torch.no_grad 内 forward → 不影响梯度。

### 3.6 target_model 不调 update_step（与 worker 类似）？

不对 — **target_model 也需调 update_step**（spec 01 §2.2）：
- target_model.update_step(global_step) 同步触发 grad_gating threshold（虽然 target 不 backward 无影响）
- 保持 model / target 状态一致（便于 inference-time 对比 debug）

实施：
```python
def train_step(self, batch, global_step):
    self.model.update_step(global_step)
    self.target_model.update_step(global_step)   # 同步
    ...
```

---

## 4. Edge Cases

| 场景 | 行为 |
|------|------|
| cfg.train.lr_schedule 非已知值 | _build_scheduler 抛 ValueError |
| cfg.train.ema_tau < 0 或 > 1 | _update_ema_target 行为未定义（数值 NaN 风险）；建议 cfg.__post_init__ 校验（属 Pkg-01 spec 05 范围）|
| target_model 参数与 online 不同 shape | _update_ema_target zip 会跳过 ↔ 不匹配项（torch.nn.Module.parameters 顺序假设）；上层确保 deepcopy 完整 |
| step > max_train_steps | LR scheduler 仍按公式算（cosine 已 ≈ lr_min）；不抛 |
| optimizer.step() 抛 RuntimeError（nan grad）| backward 已校验 grad_clip；上抛 |

---

## 5. Acceptance Criteria

### 5.1 单元测试（`tests/training/test_ema_scheduler.py`）

```python
import pytest
import torch
import math
from hyper_mve.configs import V4Config
from hyper_mve.models import HyperMuZeroModel
from hyper_mve.training import MuZeroTrainer


@pytest.fixture
def cfg_medium():
    return V4Config.from_preset("medium")


@pytest.fixture
def trainer(cfg_medium):
    model = HyperMuZeroModel(cfg_medium)
    return MuZeroTrainer(cfg_medium, model)


# ====== C5-E1: EMA tau=0.99 衰减率 ======

def test_ema_decay_correctness_tau_099(trainer, cfg_medium):
    """EMA target 更新公式: target = tau * target + (1 - tau) * online."""
    tau = cfg_medium.train.ema_tau  # 0.99
    
    # 手动改 online 让差异显著
    for p in trainer.model.parameters():
        p.data.fill_(1.0)
    for p in trainer.target_model.parameters():
        p.data.fill_(0.0)
    
    # 单次 EMA update
    trainer._update_ema_target()
    
    # target ≈ 0.99 * 0 + 0.01 * 1 = 0.01
    for p in trainer.target_model.parameters():
        assert torch.allclose(p, torch.full_like(p, 1.0 - tau), atol=1e-6), (
            f"EMA after 1 step expected {1-tau}, got mean {p.mean().item()}"
        )


def test_ema_converges_to_online(trainer, cfg_medium):
    """多次 EMA 后 target 应收敛到 online (停止改变 online 时)."""
    # 改 online
    for p in trainer.model.parameters():
        p.data.fill_(1.0)
    for p in trainer.target_model.parameters():
        p.data.fill_(0.0)
    
    # 500 次 EMA update (tau=0.99, 应收敛)
    for _ in range(500):
        trainer._update_ema_target()
    
    # target 应非常接近 online (1.0)
    for p in trainer.target_model.parameters():
        assert p.mean().item() > 0.99, f"EMA after 500 steps not converged: {p.mean().item()}"


# ====== C5-E2: warmup_cosine LR 曲线 ======

def test_lr_warmup_5k_then_cosine_anneal(trainer, cfg_medium):
    """warmup_cosine LR 曲线: 0 → lr (5K warmup) → lr_min (cosine)."""
    cfg_t = cfg_medium.train
    sched = trainer.lr_scheduler
    
    # 模拟 scheduler.step 推进
    # Step 0: 应是 0 (warmup 起点)
    actual_lr = trainer.optimizer.param_groups[0]["lr"]
    # LambdaLR step=0 时 lr_lambda(0) = 0/5000 = 0 → actual = base_lr * 0 = 0
    # 但 PyTorch LambdaLR 在 __init__ 调一次 step → step=1; 改用直接 query lr_lambda
    # 此处略, 实施时用 sched.get_last_lr() 验证
    
    # Step lr_warmup_steps (5000): 应是 cfg.train.lr
    for _ in range(cfg_t.lr_warmup_steps):
        sched.step()
    
    lr_after_warmup = trainer.optimizer.param_groups[0]["lr"]
    assert abs(lr_after_warmup - cfg_t.lr) / cfg_t.lr < 0.05, (
        f"LR after warmup expected ~{cfg_t.lr}, got {lr_after_warmup}"
    )
    
    # Step max_train_steps: 应接近 lr_min
    for _ in range(cfg_t.max_train_steps - cfg_t.lr_warmup_steps):
        sched.step()
    
    lr_final = trainer.optimizer.param_groups[0]["lr"]
    assert abs(lr_final - cfg_t.lr_min) / cfg_t.lr_min < 0.05, (
        f"LR final expected ~{cfg_t.lr_min}, got {lr_final}"
    )


def test_lr_schedule_cosine_monotonic_after_warmup(trainer):
    """warmup 后 LR 单调下降 (cosine)."""
    cfg_t = trainer.cfg.train
    sched = trainer.lr_scheduler
    
    # 跑过 warmup
    for _ in range(cfg_t.lr_warmup_steps):
        sched.step()
    
    lrs = []
    for _ in range(100):
        sched.step()
        lrs.append(trainer.optimizer.param_groups[0]["lr"])
    
    # 严格单调下降
    for lr0, lr1 in zip(lrs[:-1], lrs[1:]):
        assert lr0 >= lr1, f"LR 非单调: {lr0} -> {lr1}"


# ====== EMA 不污染 online ======

def test_ema_does_not_modify_online(trainer):
    """EMA 仅修改 target, online 应不变."""
    online_p0 = next(trainer.model.parameters()).clone()
    
    trainer._update_ema_target()
    
    online_p1 = next(trainer.model.parameters())
    assert torch.allclose(online_p0, online_p1)


# ====== target_model 与 model 同步 set_context (spec 05 §3.3) ======

def test_target_model_set_context_synced(trainer, cfg_medium):
    """trainer 调用 set_context 时 target_model 同步.
    
    集成测试: 通过 train_step 跑一遍, 验证 target_model._theta_state 与 model._theta_state shape 一致.
    """
    batch = _make_batch(cfg_medium)
    
    trainer.train_step(batch, global_step=100)
    
    # train_step 内会调 set_context_objective → model._theta_state 已设
    # target_model 也应同步
    assert trainer.target_model._theta_state is not None
    assert trainer.target_model._theta_state.shape == trainer.model._theta_state.shape


def _make_batch(cfg, device="cpu"):
    B, K, N, A = cfg.train.batch_size, cfg.train.unroll_K, cfg.env.N, cfg.env.A
    return {
        "obs": torch.randn(B, K+1, N, 99, device=device),
        "actions": torch.zeros(B, K+1, N, dtype=torch.long, device=device),
        "rewards": torch.randn(B, K+1, N, device=device),
        "c_t": torch.full((B, K+1), 0.5, device=device),
        "cap": torch.rand(B, K+1, N, 4, device=device),
        "c_hat": torch.rand(B, K+1, N, device=device),
        "z_hat": torch.softmax(torch.randn(B, K+1, N, N-1, 2, device=device), dim=-1),
        "tau": torch.zeros(B, K+1, N, dtype=torch.long, device=device),
        "pi_mve": torch.softmax(torch.randn(B, K+1, N, A, device=device), dim=-1),
        "v": torch.randn(B, K+1, N, device=device),
        "delta": torch.randn(B, K+1, N, device=device),
        "dones": torch.zeros(B, K+1, dtype=torch.bool, device=device),
    }
```

### 5.2 集成测试

`scripts/train_main.py --preset medium --max_steps 10000` 端到端：
- 前 5000 step LR 线性增长（5e-5 → 1e-4 应在 step 5000 达到）
- 5000 step 后 LR 开始 cosine 下降
- EMA target 与 online 参数差异维持稳定（不发散）

### 5.3 性能要求

- _update_ema_target: < 5 ms（B=256 时所有参数 3.2M float32 multiply + add）
- _build_scheduler: < 1 ms（一次性 __init__ 内调用）

---

## 6. v4.7 → v4 对比

| 维度 | v4.7 | v4 | 差异 |
|------|------|----|------|
| EMA tau | 硬编码 0.99 | cfg.train.ema_tau = 0.99 | configurable |
| LR scheduler | 3 种策略（hardcoded if-else）| 同 3 种 + cfg.train.lr_schedule 切换 | configurable |
| _build_scheduler 函数 | v4.7 L85-130 | 同 + configurable + warmup_cosine LambdaLR 实现 | 改动最小 |
| target_model set_context 同步 | v4.7 L242 已做 | 同 + 增加 update_step 同步 + belief detach | 修改 |

---

## 7. Cross-references

- v4.7 `training/muzero_trainer.py` L85-130 (_build_scheduler) + L406 (_update_ema_target)
- `01-trainer-loop-v2.md` §2.3 调度顺序
- `05-loss-composition.md` §3.3 target_model 同步细节
- `08-integration-contracts.md` §1（trainer 公开 self.target_model / self.lr_scheduler 给 eval 期读取）
- Pkg-01 spec 05 TrainConfig (ema_tau / lr_schedule / lr_warmup_steps / lr_min / max_train_steps)
- Pkg-04 spec 04 grad_gating（与 lr_warmup_steps 重叠期协调）
