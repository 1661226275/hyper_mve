# Spec 01: Trainer Loop v2 — 单一 train_step + 7 API 调用顺序

> 父文档：[`../proposal.md`](../proposal.md) §2.1.1 · [`../design.md`](../design.md) §3 D1/D5/D6/D8/D10 · §6.2 · §6.6
> **v4 关键改动**：v4.7 `train_step` + `train_step_infer` 双套 → v4 **单一 `train_step`**（Q2 + D1），BeliefNet 取代 GRU 推理；含 scheduler 注入（修订 1）+ 400 ms 性能预算分摊（修订 3）。

---

## 1. Purpose

按 Pkg-04 spec 02 §2.3 trainer 调用模板 + Pkg-03 spec 08 §4 课程接入实现 v4 统一 `MuZeroTrainer`，废弃 v4.7 `train_step_infer` (L432-598)，提供：

| API | 用途 | 调用频率 |
|-----|------|---------|
| `__init__(cfg, model, scheduler=None, projector=None, device=None)` | 构造 + 含 scheduler 注入（D4 实例可替换） | 一次性 |
| `train_step(batch, global_step)` | 单一 train_step（合并 Oracle/Infer 双套） | 每 train_step |
| `save_checkpoint(path)` | v4 字段格式（D8，与 v4.7 不兼容） | 周期性 |
| `load_checkpoint(path)` | 返回 global_step；含 `_legacy_v4_7/` shim 提示 | resume 时 |

---

## 1.1 TrainConfig 字段穷举表（27 项依赖，M4：避免事后修改）

本 trainer 完整依赖以下 cfg 字段；变更需 Pkg-01 spec 05 同步：

### cfg.train.* (24 项)

| 字段 | 默认 | 用途 |
|------|------|------|
| `max_train_steps` | 200_000 | trainer 主循环上限 + scheduler.lambda_b 归一化分母 |
| `batch_size` | 256 | buffer.sample_batch(B) |
| `buffer_size` | 5000 | episode 容量上限 |
| `min_buffer_size` | 1000 | warmup 阈值（trainer 等 buffer 达到再训）|
| `episodes_per_iter` | 8 | worker 每 iter collect 几集 |
| `train_steps_per_iter` | 8 | trainer 每 iter 训几步 |
| `unroll_K` | 5 | K-step unroll 深度 |
| `n_step` | 5 | n-step return 计算 |
| `gamma` | 0.95 | n-step return 折扣 |
| `lr` | 1e-4 | Adam 学习率 |
| `lr_min` | 5e-6 | scheduler 最低 lr |
| `adam_eps` | 1e-5 | Adam optimizer eps |
| `grad_clip` | 10.0 | gradient clipping 阈值 |
| `lr_schedule` | "warmup_cosine" | scheduler 类型（spec 07）|
| `lr_warmup_steps` | 5000 | warmup 步数 |
| `w_policy` | 1.0 | L_π 权重 |
| `w_value` | 0.25 | L_v 权重 |
| `w_reward` | 3.0 | L_r 权重（v4.8 经验）|
| `w_consist` | 0.5 | L_BYOL 权重 |
| `w_belief` | 1.0 | L_belief 全局权重（再乘以 scheduler.lambda_b(step) 课程系数）|
| `curriculum_stage_1_end_frac` | 0.3 | Stage 1 (Pure Oracle) 边界 |
| `curriculum_stage_2_end_frac` | 0.7 | Stage 2 (Anneal) 边界 |
| `ema_tau` | 0.99 | target model EMA 衰减 |
| `stratified_sampling` | True | buffer.sample_batch 启用分层（spec 03）|

### cfg.env.* (2 项)

| 字段 | 来源 | 用途 |
|------|------|------|
| `N` | EnvConfig | K-step unroll 内 N agents 循环范围 |
| `A` | EnvConfig | joint_action_onehot shape 计算（N×A）|

### cfg.model.* (1 项)

| 字段 | 来源 | 用途 |
|------|------|------|
| `latent_dim` | ModelConfig | K-step unroll 内 s tensor shape 检查 |

**总计 27 项** —— `__init__` 内全部读取并打印 summary（实施时验证 cfg 完整性）。

---

## 2. Interface

### 2.1 文件路径

`hyper_mve/training/muzero_trainer.py`（v4.7 同名文件 inplace 重写，D10）

### 2.2 类签名

```python
import copy
import torch
import torch.nn as nn
from typing import Optional
from hyper_mve.configs import V4Config
from hyper_mve.models import HyperMuZeroModel
from hyper_mve.training.curriculum import CurriculumScheduler
from hyper_mve.training.loss_composition import compose_total_loss


class MuZeroTrainer:
    """v4 统一 MuZero Trainer (Pkg-05 spec 01).
    
    v4 关键改动 (相对 v4.7 train_step + train_step_infer 双套):
        v4.7 Oracle: train_step (L203-430) - rule 给真值
        v4.7 Infer:  train_step_infer (L432-598) - GRU 推断 rule
        v4: 单一 train_step, BeliefNet 取代 GRU 推理, 课程 3 stage 控制 oracle 注入.
        删除 train_step_infer.
    
    7 个对外 API (spec 08 §1 锁定):
        __init__ (scheduler 注入, 修订 1)
        train_step (单一, Q2)
        save_checkpoint / load_checkpoint (v4 字段, D8)
        + 内部辅助: _update_ema_target / _build_scheduler (spec 07 详述)
    """
    
    def __init__(
        self,
        cfg: V4Config,
        model: HyperMuZeroModel,
        scheduler: Optional[CurriculumScheduler] = None,   # ← review 修订 1
        projector: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            cfg: V4Config (从 preset 加载)
            model: HyperMuZeroModel (Pkg-04 实例)
            scheduler: 课程 scheduler;
                review 修订 1 (D4 实例可替换):
                    None 时 trainer 默认 CurriculumScheduler(cfg).
                    Pkg-08 ablation 4 课程边界扫描时传自定义实例.
                trainer 持有 (self.scheduler); compose_total_loss 内部从 self.scheduler 取.
            projector: BYOL projector (可选, L_consist 用); None 时跳过 BYOL
            device: torch.device, None 时 cuda if available else cpu
        """
        self.cfg = cfg
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # cfg 字段完备性自检 (M4)
        self._validate_cfg_fields()
        
        # scheduler 注入 (review 修订 1)
        self.scheduler = scheduler or CurriculumScheduler(cfg)
        
        # projector
        self.projector = projector.to(self.device) if projector else None
        
        # EMA target model (v4.4 保留, spec 07)
        self.target_model = copy.deepcopy(self.model)
        self.target_model.eval()
        for p in self.target_model.parameters():
            p.requires_grad = False
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.train.lr,
            eps=cfg.train.adam_eps,
        )
        
        # LR scheduler (v4.7 _build_scheduler 保留, spec 07)
        self.lr_scheduler = self._build_scheduler()
        
        # 内部状态 (stateful API 契约, 与 Pkg-04 spec 02 §2.2 风格一致)
        self.global_step: int = 0
    
    def _validate_cfg_fields(self) -> None:
        """27 项 cfg 字段完备性自检 (M4)."""
        TRAIN_REQUIRED = [
            "max_train_steps", "batch_size", "buffer_size", "min_buffer_size",
            "episodes_per_iter", "train_steps_per_iter",
            "unroll_K", "n_step", "gamma",
            "lr", "lr_min", "adam_eps", "grad_clip",
            "lr_schedule", "lr_warmup_steps",
            "w_policy", "w_value", "w_reward", "w_consist", "w_belief",
            "curriculum_stage_1_end_frac", "curriculum_stage_2_end_frac",
            "ema_tau", "stratified_sampling",
        ]
        ENV_REQUIRED = ["N", "A"]
        MODEL_REQUIRED = ["latent_dim"]
        
        for f in TRAIN_REQUIRED:
            assert hasattr(self.cfg.train, f), f"cfg.train missing {f}"
        for f in ENV_REQUIRED:
            assert hasattr(self.cfg.env, f), f"cfg.env missing {f}"
        for f in MODEL_REQUIRED:
            assert hasattr(self.cfg.model, f), f"cfg.model missing {f}"
    
    # ====================================================================
    # API 1: train_step (单一, Q2 合并 v4.7 双套)
    # ====================================================================
    
    def train_step(
        self,
        batch: dict[str, torch.Tensor],
        global_step: int,
    ) -> dict[str, float]:
        """单一 train_step.
        
        内部按 Pkg-04 spec 02 §2.3 调用顺序 (硬约束 C5-T1/T2/T3):
            1. model.update_step(global_step)                      # C5-T1
            2. model.set_context_objective(c_t)                    # C5-T2 一次
            3. for k in range(N):                                  # C5-T3
                   model.set_context_subjective(k, cap[:, k], (c_hat[:, k], z_hat[:, k]))
                   # K-step unroll: encode / transition / predict_reward / predict
            4. target_model 同步 (EMA, 与 model 平行调 set_context_*)
            5. compose_total_loss(model, batch, self, global_step, cfg) → losses dict
            6. losses["total"].backward()
            7. self.optimizer.step(); self.lr_scheduler.step(); self._update_ema_target()
        
        Args:
            batch: 来自 buffer.sample_batch, 字段含 obs/actions/rewards/c_t/cap/c_hat/z_hat/tau/...
            global_step: 当前训练步全局编号 (用于 update_step + scheduler)
        
        Returns:
            dict[str, float]: loss 分量字典 (用于 TensorBoard 日志)
                "total" / "main" / "belief" / "lambda_b" / sub-losses
        """
        self.global_step = global_step
        self.model.train()
        
        # ====== Step 1: update_step (C5-T1) ======
        self.model.update_step(global_step)
        self.target_model.update_step(global_step)  # target 同步
        
        # ====== Step 2-3: 委托给 compose_total_loss (spec 05) ======
        # 内部完成：
        #   - model.set_context_objective(batch["c_t"])         (C5-T2)
        #   - for k: model.set_context_subjective(k, cap, belief)  (C5-T3)
        #   - K-step unroll forward
        #   - target model 同步 set_context + bootstrap n-step return
        #   - main_loss + L_belief 双路径拼装 (修订 2 强制顺序)
        losses = compose_total_loss(
            model=self.model,
            batch=batch,
            trainer=self,                          # 取 self.scheduler / self.target_model
            global_step=global_step,
            cfg=self.cfg,
        )
        
        # ====== Step 6: 一次 backward (双路径同时反传) ======
        self.optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.train.grad_clip,
        )
        
        # ====== Step 7: optimizer / scheduler / EMA ======
        self.optimizer.step()
        self.lr_scheduler.step()
        self._update_ema_target()
        
        # 返回 float 字典用于日志
        return {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
    
    # ====================================================================
    # API 2: compute_n_step_return (v4.4 EMA bootstrap, 保留)
    # ====================================================================
    
    def compute_n_step_return(
        self,
        rewards: torch.Tensor,        # (B, K+1, N) 来自 batch
        v_target: torch.Tensor,       # (B, K+1, N) 来自 target_model.predict
        dones: torch.Tensor,          # (B, K+1) bool
        n: int,
    ) -> torch.Tensor:                # (B, K+1, N) n-step return
        """n-step return 计算 (v4.7 muzero_trainer.py L138-181 保留).
        
        z_t = r_t + γ·r_{t+1} + ... + γ^{n-1}·r_{t+n-1} + γ^n·v_target(s_{t+n})
        
        done 处理: 一旦遇到 done=True, 后续步骤的 r 和 bootstrap 都置零.
        """
        # 实现细节复用 v4.7 (与 Pkg-05 spec 07 EMA 一致)
        # ... 与 v4.7 muzero_trainer.py L138-181 一致 ...
        pass
    
    # ====================================================================
    # API 3-4: save_checkpoint / load_checkpoint (D8, v4 字段不兼容 v4.7)
    # ====================================================================
    
    def save_checkpoint(self, path: str) -> None:
        """保存 v4 checkpoint.
        
        v4 字段 (D8 明确 breaking, 与 v4.7 不兼容):
            "version":         "v4"  ← 关键标识
            "global_step":     int
            "model_state":     model.state_dict()
            "target_state":    target_model.state_dict()
            "optimizer_state": optimizer.state_dict()
            "lr_scheduler_state": lr_scheduler.state_dict()
            "scheduler_state": scheduler.state_dict() (若 CurriculumScheduler 有状态)
            "cfg":             cfg.to_dict()  ← 完整 V4Config 序列化
        """
        torch.save({
            "version": "v4",
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "target_state": self.target_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "lr_scheduler_state": self.lr_scheduler.state_dict(),
            "cfg": self.cfg.to_dict(),
        }, path)
    
    def load_checkpoint(self, path: str) -> int:
        """加载 v4 checkpoint, 返回 global_step.
        
        若加载 v4.7 checkpoint (version 缺失或为 "v4.7"):
            raise RuntimeError(
                "v4.7 checkpoint detected. v4 与 v4.7 不兼容 (D8)."
                "若需 reproduce v4.7 实验: git checkout v4.7-final + _legacy_v4_7/scripts/."
            )
        """
        ckpt = torch.load(path, map_location=self.device)
        version = ckpt.get("version", "v4.7")  # 缺失字段默认 v4.7
        if version != "v4":
            raise RuntimeError(
                f"Checkpoint version mismatch: expected 'v4', got '{version}'. "
                f"v4 与 v4.7 不兼容 (Pkg-05 D8). "
                f"若需 reproduce v4.7 实验: git checkout v4.7-final + _legacy_v4_7/scripts/."
            )
        
        self.global_step = ckpt["global_step"]
        self.model.load_state_dict(ckpt["model_state"])
        self.target_model.load_state_dict(ckpt["target_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.lr_scheduler.load_state_dict(ckpt["lr_scheduler_state"])
        return self.global_step
    
    # ====================================================================
    # 内部辅助 (spec 07 详述)
    # ====================================================================
    
    def _build_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """warmup_cosine / cosine / multistep 三选一 (v4.7 L85-130 保留)."""
        pass  # spec 07 详述
    
    def _update_ema_target(self) -> None:
        """EMA target model 更新 (tau=0.99, v4.4 保留, spec 07 详述).
        
        param_target.mul_(tau).add_(param_online, alpha=1 - tau)
        """
        with torch.no_grad():
            tau = self.cfg.train.ema_tau
            for p_o, p_t in zip(self.model.parameters(), self.target_model.parameters()):
                p_t.data.mul_(tau).add_(p_o.data, alpha=1.0 - tau)
```

### 2.3 调用顺序契约（D4 + Pkg-04 spec 02 §2.3）

trainer 在 train_step 内严格遵循以下顺序（每一步对应一个硬约束）：

```python
# Pkg-04 spec 02 §2.3 调用模板 + Pkg-05 trainer 实施
def train_step(self, batch, global_step):
    # === 1. update_step (C5-T1, 每 train_step 一次) ===
    self.model.update_step(global_step)
    self.target_model.update_step(global_step)
    
    # === 2-3. compose_total_loss 内部完成 set_context + forward + loss ===
    losses = compose_total_loss(self.model, batch, self, global_step, self.cfg)
    # ↑ 内部:
    #   self.model.set_context_objective(batch["c_t"])           # C5-T2
    #   self.target_model.set_context_objective(batch["c_t"])
    #   for k in range(N):                                       # C5-T3
    #       self.model.set_context_subjective(k, ...)
    #       self.target_model.set_context_subjective(k, ...)
    #       # K-step unroll forward
    
    # === 4. backward + optimizer + scheduler + EMA ===
    self.optimizer.zero_grad(set_to_none=True)
    losses["total"].backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
    self.optimizer.step()
    self.lr_scheduler.step()
    self._update_ema_target()
    
    return {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
```

---

## 3. Implementation Notes

### 3.1 单一 train_step 的语义（D1 + Q2）

v4.7 双套 train_step 的区分是"rule 来源"：
- `train_step`: rule 来自 batch (Oracle，给真值)
- `train_step_infer`: rule 来自 model 内部 GRU 推断（Infer）

v4 BeliefNet 取代 GRU 推理 → "rule 来源"不再是 model 类区分，而是**课程 stage 控制 oracle 注入**：
- Stage 1: BeliefNet.forward 接受 oracle_z_seq=build_oracle_z_seq(batch["tau"])（替代输出）
- Stage 2: oracle / predicted 加权混合（按 scheduler.oracle_z_mixing_weight）
- Stage 3: 纯 BeliefNet 推断

trainer 统一 train_step，stage 切换由 compose_total_loss 内部 query scheduler 完成。

### 3.2 target_model 同步 set_context（与 Pkg-04 spec 02 §2.3 一致）

EMA target model 同样需要调 update_step + set_context_objective + set_context_subjective（与 online model 平行）：
- target model 用于 compute_n_step_return 的 bootstrap V(s_{t+n})
- target model 不 backward，仅 forward
- target 的 set_context_* 与 online 相同参数（同一 c_t / cap / belief）

### 3.3 性能预算分摊（R5-1 review 修订 3 锁定）

| 子项 | 预算 | 实现说明 |
|------|------|---------|
| forward (model, N=4 × K=5) | < 100 ms | Pkg-04 spec 07 档位 3（已 finalize）|
| forward (target, no_grad) | < 50 ms | target 也调 set_context_* 但用 torch.no_grad（节省 ~50%）|
| forward (BeliefNet.forward, oracle_z_seq 注入) | < 30 ms | trainer 训练时（worker 用 BeliefNet.step 不同 API）|
| backward (main + L_belief 一次) | < 180 ms | 经验 ≤ 2× model forward |
| optimizer step + EMA update | < 20 ms | grad_clip + Adam.step + EMA tau decay |
| buffer.sample_batch + tensor 转移 | < 20 ms | spec 03 验收 |
| **合计** | **< 400 ms** | **含 20 ms 缓冲** |

### 3.4 v4.7 train_step_infer 删除策略（D1）

v4.7 train_step_infer (L432-598, ~167 行) 整体删除。**禁止保留任何 if "infer" 分支**：

```python
# ❌ 禁止 (污染代码)
def train_step(self, batch, global_step):
    if self.model_type == "infer":
        return self._train_step_infer(batch, global_step)
    else:
        ...

# ✅ 正确 (统一)
def train_step(self, batch, global_step):
    # 无 if-else 判断, 统一逻辑
    ...
```

若需 reproduce v4.7 Infer 实验：`git checkout v4.7-final + _legacy_v4_7/scripts/train_infer.py`（与 D8 一致）。

### 3.5 stateful API 契约（沿用 Pkg-04 修订 2）

self.global_step / self.scheduler / self.target_model 是 stateful 字段：
- save_checkpoint 必须含 global_step（resume 时 scheduler 状态由 global_step 推导）
- target_model 是 model 的 deep copy，独立持有参数
- scheduler 默认无内部状态（lightweight，仅 query 方法），但若 Pkg-08 ablation 传自定义 scheduler 含状态，需在 save_checkpoint 额外保存

### 3.6 与 Pkg-04 spec 02 §3.6 cached_c_t 配合（P0-2 修订：纠正 set_context_objective 频率）

Pkg-04 spec 02 §3.6 提到 model 内部缓存 c_t 供 set_context_subjective 使用。trainer 端的 batch["c_t"] 是 (B, K+1) 形状：
- **K-step unroll 起点用 `batch["c_t"][:, 0]` 一次性 `set_context_objective`**（与 C5-T2 + spec 01 §5.1 单测 `test_objective_called_once_per_unroll` + spec 05 §2.2 实施一致）
- 同步 target_model 也是 unroll 起点一次性调用

**P0-2 修订说明（原描述错）**：

旧版本 spec 01 §3.6 写"在 K-step unroll 每步起点：`set_context_objective(batch["c_t"][:, t])`"（K+1 次），与三处冲突：
1. § 2.3 调用顺序"set_context_objective 仅调 1 次（C5-T2）"
2. § 5.1 单测 `test_objective_called_once_per_unroll`: `assert spy.call_count == 1`
3. spec 05 § 2.2 实施 `c_t_root = batch["c_t"][:, 0]; model.set_context_objective(c_t_root)` — 单次

正确语义：K-step 内 c_t 演化是 RNS-MMG 物理事实（Ch3.4 c 漂移），但 v4 主线使用 **model-based 客观转移近似**（hyper_trans 生成 θ_state 复用 across K 步，对应 Ch4.3.2 设计）。该近似的代价是 K-step unroll 内 c_t 漂移信号不进 model；好处是 hyper_trans 单次 forward 节省 ~K×2 ms（Pkg-04 spec 07 § 5.2）。

若未来 ablation 需精细 c_t 演化（K-step 内每步重设 θ_state），需在 spec 01 / spec 05 同步修改（且违反 C5-T2 硬约束需重新评估）。

---

## 4. Edge Cases

| 场景 | 行为 |
|------|------|
| train_step 在 buffer 未达 min_buffer_size 时调用 | 上层 train_main.py 控制；trainer 不校验（buffer.sample_batch 抛 ValueError）|
| batch 缺少必需字段（如 c_t 缺失）| compose_total_loss 内 KeyError + 明确错误信息 |
| target_model 与 online model 维度不一致 | _update_ema_target 抛 RuntimeError（dim mismatch）|
| save_checkpoint 路径不存在父目录 | torch.save 抛 FileNotFoundError |
| load_checkpoint 读到 v4.7 格式 | RuntimeError + 明确指引（D8）|
| global_step 回滚（resume 后传比 ckpt 小的值）| trainer 不校验；scheduler / EMA 可能进入未定义行为；上层保证 |
| scheduler 传入但 cfg.train.curriculum_*_end_frac 不一致 | trainer 不校验；以传入 scheduler 为准（Pkg-08 ablation 场景）|

---

## 5. Acceptance Criteria

### 5.1 单元测试（`tests/training/test_trainer_loop.py`）

```python
import pytest
import torch
from hyper_mve.configs import V4Config
from hyper_mve.models import HyperMuZeroModel
from hyper_mve.training import MuZeroTrainer
from hyper_mve.training.curriculum import CurriculumScheduler


@pytest.fixture
def cfg_medium():
    return V4Config.from_preset("medium")


@pytest.fixture
def model(cfg_medium):
    return HyperMuZeroModel(cfg_medium)


@pytest.fixture
def trainer(cfg_medium, model):
    return MuZeroTrainer(cfg_medium, model)


# ====== C5-T1: update_step ======

def test_trainer_calls_update_step_per_step(trainer, cfg_medium, mocker):
    """每 train_step 起点调 model.update_step 1 次."""
    spy = mocker.spy(trainer.model, "update_step")
    batch = _make_dummy_batch(cfg_medium)
    
    trainer.train_step(batch, global_step=100)
    
    assert spy.call_count == 1
    assert spy.call_args[0][0] == 100


# ====== C5-T2: set_context_objective 一次 ======

def test_objective_called_once_per_unroll(trainer, cfg_medium, mocker):
    spy = mocker.spy(trainer.model, "set_context_objective")
    batch = _make_dummy_batch(cfg_medium)
    
    trainer.train_step(batch, global_step=100)
    
    # 一次 train_step 内 set_context_objective 仅调一次（复用 θ_state across N agents）
    assert spy.call_count == 1


# ====== C5-T3: N agents 循环 set_context_subjective ======

def test_subjective_called_per_agent(trainer, cfg_medium, mocker):
    spy = mocker.spy(trainer.model, "set_context_subjective")
    batch = _make_dummy_batch(cfg_medium)
    
    trainer.train_step(batch, global_step=100)
    
    # N agents × K+1 unroll 步 = N × (K+1) 次（每步切 agent）
    # 但取决于实施：若 K-step unroll 内复用同一 batch 仅切 agent 维度，则 N 次
    # 此处宽松断言：至少 N 次
    assert spy.call_count >= cfg_medium.env.N


# ====== scheduler 注入（修订 1）======

def test_scheduler_injection_default(cfg_medium, model):
    """scheduler=None 时 trainer 默认构造."""
    trainer = MuZeroTrainer(cfg_medium, model)
    assert isinstance(trainer.scheduler, CurriculumScheduler)


def test_scheduler_injection_custom(cfg_medium, model):
    """scheduler 显式传入时 trainer 使用该实例（D4 实例可替换）."""
    custom_scheduler = CurriculumScheduler(cfg_medium)
    custom_scheduler._test_marker = "custom"  # 标识
    
    trainer = MuZeroTrainer(cfg_medium, model, scheduler=custom_scheduler)
    assert trainer.scheduler is custom_scheduler
    assert trainer.scheduler._test_marker == "custom"


# ====== EMA target update ======

def test_target_model_ema_update(trainer, cfg_medium):
    """train_step 后 target_model 参数应向 online model 滑动."""
    batch = _make_dummy_batch(cfg_medium)
    
    # 改动 online model 让差异显著
    for p in trainer.model.parameters():
        p.data.add_(torch.randn_like(p) * 0.1)
    
    target_p0 = next(trainer.target_model.parameters()).clone()
    online_p0 = next(trainer.model.parameters()).clone()
    
    trainer.train_step(batch, global_step=100)
    
    target_p1 = next(trainer.target_model.parameters()).clone()
    
    # target 应向 online 滑动 (tau=0.99 → 小步)
    diff_pre = (target_p0 - online_p0).abs().mean()
    diff_post = (target_p1 - online_p0).abs().mean()
    assert diff_post < diff_pre, "EMA target 应向 online 滑动"


# ====== n-step return ======

def test_n_step_return_correctness(trainer, cfg_medium):
    """n-step return 公式验证."""
    B, K, N = 2, cfg_medium.train.unroll_K, cfg_medium.env.N
    rewards = torch.full((B, K+1, N), 1.0)
    v_target = torch.full((B, K+1, N), 10.0)
    dones = torch.zeros(B, K+1, dtype=torch.bool)
    
    z = trainer.compute_n_step_return(rewards, v_target, dones, n=cfg_medium.train.n_step)
    
    # 公式: z_0 = r_0 + γ·r_1 + γ²·r_2 + γ³·r_3 + γ⁴·r_4 + γ⁵·v_target[5]
    gamma = cfg_medium.train.gamma
    expected_0 = sum(gamma**i * 1.0 for i in range(5)) + gamma**5 * 10.0
    assert torch.allclose(z[0, 0, 0], torch.tensor(expected_0), atol=1e-5)


# ====== checkpoint v4/v4.7 兼容 ======

def test_save_load_checkpoint_roundtrip(trainer, tmp_path):
    path = str(tmp_path / "ckpt.pt")
    trainer.global_step = 12345
    trainer.save_checkpoint(path)
    
    # 新 trainer load
    trainer2 = MuZeroTrainer(trainer.cfg, HyperMuZeroModel(trainer.cfg))
    step = trainer2.load_checkpoint(path)
    
    assert step == 12345
    assert trainer2.global_step == 12345


def test_load_v47_checkpoint_raises(trainer, tmp_path):
    """读 v4.7 checkpoint 应抛 RuntimeError (D8)."""
    path = str(tmp_path / "v47_ckpt.pt")
    torch.save({"version": "v4.7", "model_state": {}}, path)
    
    with pytest.raises(RuntimeError, match="v4 与 v4.7 不兼容"):
        trainer.load_checkpoint(path)


# ====== R5-1: train_step < 400 ms (review 修订 3 含分摊) ======

@pytest.mark.gpu
def test_train_step_under_400ms(trainer, cfg_medium):
    """单 train_step < 400 ms (V100, Medium B=256)."""
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    
    import time
    batch = _make_dummy_batch(cfg_medium, device="cuda")
    
    # Warmup
    for _ in range(5):
        trainer.train_step(batch, global_step=0)
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for step in range(20):
        t0 = time.perf_counter()
        trainer.train_step(batch, global_step=step)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    
    mean_ms = sum(times) / len(times)
    assert mean_ms < 400.0, f"train_step {mean_ms:.2f}ms > 400ms (R5-1 分摊预算)"


# ====== R5-1 micro-benchmark: 3 段 forward 分别计时（避免定位困难）======

@pytest.mark.gpu
def test_micro_benchmark_three_forwards(trainer, cfg_medium):
    """分别测 model forward / target forward / belief forward, 避免 R5-1 fail 时无法定位瓶颈."""
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    
    import time
    batch = _make_dummy_batch(cfg_medium, device="cuda")
    
    # 1. model forward (含 N agents × K-step)
    # ... 调 model.set_context_* + K-step unroll ...
    # assert model_forward_ms < 100
    
    # 2. target_model forward (no_grad)
    # ... 同 1 但 with torch.no_grad ...
    # assert target_forward_ms < 50
    
    # 3. BeliefNet.forward (trainer 训练时, oracle_z_seq 注入)
    # ... model.belief_net.forward(batch["obs"], oracle_z_seq=...) ...
    # assert belief_forward_ms < 30
    
    pass  # 实施时补全


def _make_dummy_batch(cfg, device="cpu"):
    """构造 train_step 测试用 batch."""
    B, K, N, A = cfg.train.batch_size, cfg.train.unroll_K, cfg.env.N, cfg.env.A
    obs_dim = 99  # Medium config
    return {
        "obs": torch.randn(B, K+1, N, obs_dim, device=device),
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

`scripts/train_main.py --preset medium --max_steps 100` 端到端：
- 100 train_steps 无 NaN
- loss 单调下降趋势（前 100 步至少 main loss 降 5%）
- BeliefNet 参数前 5000 step grad = 0（与 Pkg-04 spec 04 联动）

### 5.3 性能要求（review 修订 3 锁定）

| 子项 | 阈值 | 单测 |
|------|------|------|
| 单 train_step | < 400 ms | `test_train_step_under_400ms` |
| model forward | < 100 ms | `test_micro_benchmark_three_forwards` 子断言 |
| target forward | < 50 ms | 同上 |
| BeliefNet forward | < 30 ms | 同上 |
| backward + optim + EMA | < 200 ms | 由 R5-1 总预算反推 |

---

## 6. v4.7 → v4 对比

| 维度 | v4.7 | v4 | 差异 |
|------|------|----|------|
| train_step 数量 | 2（Oracle + Infer）| 1（合并）| 删 train_step_infer (167 行) |
| set_context 调用 | 单参（rule, agent_id）| 两步分离（update_step + set_context_objective + set_context_subjective）| Pkg-04 spec 08 §3.2 6 处迁移 |
| target_model 同步 | 已有（EMA, v4.4） | 同 + 加 set_context 同步 | 修改 set_context 调用 |
| 课程切换 | trainer 内联 if 分支 | 委托 CurriculumScheduler（D4 实例可替换）| 抽出 scheduler 类 |
| loss 拼装 | trainer 内联 | 委托 compose_total_loss（D5 + 修订 2 双路径）| 抽出 loss_composition.py |
| checkpoint 格式 | v4.7 字段 | v4 字段（D8 breaking）+ shim 提示 | 不兼容 v4.7 |
| 单 train_step 性能 | ~290 ms | ~340-380 ms 预算 < 400 ms | +17-30%（hyper_rew 80 维 + BeliefNet）|

---

## 7. Cross-references

- Ch5.6 数据组织 + 5.6.3 loss 权重 + 5.6.5 stratified sampling
- Ch5.7 课程 3 stage
- `02-worker-collection.md`（worker 与 trainer 协同：worker 产数据，trainer 消费）
- `03-episode-buffer-v2.md`（buffer.sample_batch 供 trainer.train_step 消费）
- `04-curriculum-scheduler.md`（CurriculumScheduler.stage / lambda_b / oracle_z_mixing_weight）
- `06-mve-planner-v4.md`（trainer 训练时不直接调 planner — worker 端调；本 spec 引用以确认 4 类 API 完整）
- `05-loss-composition.md`（compose_total_loss + 双路径 backward）
- `07-ema-and-scheduler.md`（_update_ema_target + _build_scheduler 实施细节）
- `08-integration-contracts.md` §1（4 类 API 稳定性）+ §2（v4.7 → v4 grep 验证）
- Pkg-01 spec 04 TimeStepRecord（batch 字段约定）
- Pkg-01 spec 05 TrainConfig（27 字段穷举）
- Pkg-03 spec 06 belief_loss（trainer L_belief 调用）
- Pkg-03 spec 08 §4 课程接入路径
- Pkg-04 spec 02 §2.3（trainer 调用模板）+ spec 04 §3.4（worker 不调 update_step）
- Pkg-04 spec 07 档位 3 < 100 ms（R5-1 分摊基准）
- Pkg-04 spec 08 §3.2（muzero_trainer.py 6 处迁移行号）
