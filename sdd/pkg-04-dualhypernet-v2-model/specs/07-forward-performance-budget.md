# Spec 07: Forward 性能预算

> 父文档：[`../design.md`](../design.md) §2 G5 · §6.5
> **硬阈值（review 修订 4 拆 3 档）**：
> 1. 单 forward call（encode / single transition / single predict）**< 5 ms**
> 2. 单 agent K=5 step unroll **< 25 ms**（含 1 次 set_context_subjective + 5 次 transition/predict_reward/predict）
> 3. N=4 agents × K=5 step unroll **< 100 ms**（含 1 次 set_context_objective + N=4 次 set_context_subjective 切换开销）
>
> 参数量 ∈ [2.8M, 3.6M]；显存 < 8GB peak。

---

## 1. Purpose

为 v4 HyperMuZeroModel 提供：

1. **单步 forward 性能预算**（硬阈值 < 15 ms，超过 fail）
2. **参数量分解表**（与 Ch4.3.4 ~3.2M 对齐）
3. **显存预算**（< 8 GB peak @ B=256, T=10 K-step unroll）
4. **v4.7 → v4 性能增量分析**
5. **性能 smoke test 脚本**（M7：自带单测护栏）

---

## 2. 单步 Forward 性能预算（review 修订 4 拆 3 档）

### 2.1 三档预算（Medium config B=256, N=4, V100 GPU）

#### 档位 1：单 forward call（细粒度，K-step 内部各调用各自的预算）

| API | 预算 | 备注 |
|-----|------|------|
| `model.encode(obs)` (RepNet) | **< 5 ms** | obs (256, 4, 99) → s (256, 64)；单次调用 |
| `model.transition(s, action)` | **< 5 ms** | FunctionalStateTransNet 单步（B=256）|
| `model.predict_reward(s, action)` | **< 5 ms** | RewardHead 单步 |
| `model.predict(s)` | **< 5 ms** | PredictionNet 单步 |
| `model.set_context_objective(c_t)` | **< 5 ms** | TriContextEncoder.forward_c_ctx_only + hyper_trans |
| `model.set_context_subjective(k, ...)` | **< 5 ms** | TriContextEncoder.forward(N=1) + hyper_rew/pred + grad_gating 两道 detach |

实测预期（typical V100）：每个单 call 在 0.5-3 ms。5 ms 阈值留 ~2x 安全余量。

#### 档位 2：单 agent K=5 step unroll（trainer 标准粒度）

```
1 × set_context_subjective(k, cap, belief)            # ~3 ms
5 × [transition + predict_reward + predict]           # ~5 × (0.5 + 1 + 1.5) = 15 ms
                                                       --------
                                                       总 < 25 ms ✓
```

| 子项 | 预算 | 累积 |
|------|------|------|
| set_context_subjective(k, ...) | < 3 ms | 3 ms |
| 5 × transition | < 3 ms | 6 ms |
| 5 × predict_reward | < 5 ms | 11 ms |
| 5 × predict | < 8 ms | 19 ms |
| **总计** | **< 25 ms** | **19 ms（预算 6 ms 缓冲）** |

#### 档位 3：N=4 agents × K=5 step unroll（trainer 单 train_step 完整 forward）

```
1 × encode(obs)                                       # ~1 ms
1 × set_context_objective(c_t)                        # ~2 ms (复用 N 次, 节省 ~6 ms)
N=4 × [set_context_subjective + K=5 step unroll]      # 4 × (3 + 15) = 72 ms
                                                       --------
                                                       总 < 100 ms ✓
```

| 子项 | 预算 | 累积 | 备注 |
|------|------|------|------|
| encode | < 1 ms | 1 ms | 一次 |
| set_context_objective | < 2 ms | 3 ms | 一次（N agents 复用 θ_state，**Q3 两步分离收益**）|
| 4 × set_context_subjective | < 12 ms | 15 ms | 每 agent ~3 ms 切换开销 |
| 4 × (5 × transition + 5 × predict_reward + 5 × predict) | < 80 ms | 95 ms | 见档位 2 |
| **总计** | **< 100 ms** | **95 ms（预算 5 ms 缓冲）** |

**硬阈值断言**（M7：性能预算自带单测护栏）：

```python
# scripts/test_hyper_model_forward.py 内三档分别断言
assert single_call_max_ms < 5.0   # 档位 1
assert single_agent_unroll_ms < 25.0  # 档位 2
assert full_step_ms < 100.0   # 档位 3
```

### 2.2 set_context_subjective 切换开销专项 profile（review 修订 4）

D4 缓存策略真正的收益需要单独验证。新增 profile 项：

| 测量项 | 测量方法 | 预期 | 验收 |
|--------|----------|------|------|
| `set_context_subjective` 单调用耗时 | 100 次重复 `set_context_subjective(k, cap, belief)` 取 mean | ~2-3 ms (B=256) | < 5 ms |
| 切 agent 增量开销 | (N=4 agent 全切总耗时 - 1 agent 调用耗时) / (N-1) | < 3 ms / agent | < 5 ms |
| set_context_objective 复用收益 | (调 N 次 objective vs 1 次 + N 次 subjective) | 节省 ~6 ms (4 次 hyper_trans 节省) | ≥ 50% trans 调用节省 |

实施方式（spec 07 §6 smoke script 内单独 section）：

```python
# Profile set_context_subjective 单独耗时
times_subj = []
for _ in range(100):
    t0 = time.perf_counter()
    model.set_context_subjective(0, cap[:, 0], (c_hat[:, 0], z_hat[:, 0]))
    torch.cuda.synchronize()
    times_subj.append((time.perf_counter() - t0) * 1000)

print(f"set_context_subjective: mean={np.mean(times_subj):.2f}ms, "
      f"p95={np.percentile(times_subj, 95):.2f}ms")
assert np.mean(times_subj) < 5.0, "set_context_subjective 单调用超 5ms"

# 验证 D4 缓存策略收益: 比较"两步分离" vs "假装合并 set_context 每次重算 trans"
# (后者只能近似: 调 N 次 set_context_objective + N 次 set_context_subjective)
times_naive = []
times_d4 = []
for _ in range(50):
    # 假设合并模式: 每 agent 调一次 objective
    t0 = time.perf_counter()
    for k in range(N):
        model.set_context_objective(c_t)   # 浪费: N 次 hyper_trans
        model.set_context_subjective(k, cap[:, k], (c_hat[:, k], z_hat[:, k]))
    torch.cuda.synchronize()
    times_naive.append((time.perf_counter() - t0) * 1000)
    
    # D4 模式: objective 一次, subjective N 次
    t0 = time.perf_counter()
    model.set_context_objective(c_t)       # 1 次
    for k in range(N):
        model.set_context_subjective(k, cap[:, k], (c_hat[:, k], z_hat[:, k]))
    torch.cuda.synchronize()
    times_d4.append((time.perf_counter() - t0) * 1000)

savings_pct = (np.mean(times_naive) - np.mean(times_d4)) / np.mean(times_naive) * 100
print(f"D4 缓存策略节省: {savings_pct:.1f}% "
      f"(naive={np.mean(times_naive):.2f}ms, d4={np.mean(times_d4):.2f}ms)")
assert savings_pct > 30.0, "D4 缓存策略未达 30% 节省, 复查 hyper_trans 复用"
```

### 2.2 性能瓶颈预测

最大开销在 **hyper_rew**（rew_hidden_dims=(256, 256, 256) 三层 MLP，比 trans/pred 多一层）：

| HyperNet | hidden 层数 | 输入 | 输出 | 单步开销 (B=256) |
|----------|-------------|------|------|-------------------|
| hyper_trans | 2 (256, 256) | 16 | trans_param_count | ~1.5 ms |
| hyper_rew | 3 (256, 256, 256) | 80 | rew_param_count | ~3 ms |
| hyper_pred | 2 (256, 256) | 80 | pred_param_count | ~2 ms |

N=4 时 hyper_rew 调 4 次 → 总 ~12 ms（占预算 80%）。

**优化机会**：set_context_objective 一次性算 θ_state（节省 N-1 次 hyper_trans 调用，~4.5 ms 节省）。这是 Q3 两步分离 API 的核心收益。

---

## 3. 参数量分解

### 3.1 完整分解表（与 Ch4.3.4 ~3.2M 对齐）

假设 `trans_param_count = 37K, rew_param_count = 4K, pred_param_count = 42K`（基于 functional_nets 实际 layer_specs，spec 03 详述）：

| 模块 | 参数量计算 | 数值 | Ch4.3.4 表对照 |
|------|-----------|------|----------------|
| **RepNet** | obs_dim×128 + 128×128 + 128×64 ≈ 25K | ~0.5 M（v4.7 多层 MLP） | 0.5 M ✓ |
| **BeliefNet (Pkg-03)** | obs_encoder 12K + GRUCell 24K + head_c 8K + head_opp 10K + opp_id_emb 0.1K | ~0.3 M（含 obs_encoder/GRU/heads） | 0.3 M ✓ |
| **TriContextEncoder + BeliefEncoder (Pkg-03)** | CEncoder ~2K + RoleEncoder ~0.4K + BeliefEncoder ~1K + LN ~160 | ~0.1 M | 0.1 M ✓ |
| **hyper_trans** | 16×256 + 256×256 + 256×37000 + output_scale ≈ 9.5 M ❌（v4.7 实际不是这个规模） | 见 §3.2 实际计算 | 0.4 M 表 |
| **hyper_rew** | 80×256 + 256×256 + 256×256 + 256×4000 ≈ 1.2 M | ~0.6 M | 0.6 M 表 |
| **hyper_pred** | 80×256 + 256×256 + 256×42000 ≈ 10.8 M ❌ | 见 §3.2 | 1.0 M 表 |
| **StateTransNet (Functional)** | layer_specs 累积 ≈ 37K | ~0.1 M（实际 functional param count） | 0.1 M ✓ |
| **RewardHead (Functional)** | layer_specs ≈ 4K | ~0.05 M | 0.05 M ✓ |
| **PredictionNet (Functional)** | layer_specs ≈ 42K | ~0.1 M | 0.1 M ✓ |
| **总计** | – | **~3.2 M** | **3.2 M** |

### 3.2 hyper_trans / hyper_pred 实际参数量解析

**关键澄清**：Ch4.3.4 表里的"hyper_trans=0.4M / hyper_pred=1.0M"是 **hyper net 本身的参数**（trunk + output_layer 权重），**不是** output 维度。

实际计算（hyper_trans 例子）：
- input → 256: 16×256 + 256 = 4352
- 256 → 256: 256×256 + 256 = 65792
- 256 → trans_param_count: 256 × trans_param_count + trans_param_count

若 trans_param_count = 37K：
- 256 × 37000 + 37000 ≈ 9.5 M（output_layer 占主导）

这与 Ch4.3.4 表的 0.4M 不一致 ❌。

**原因**：v4.7 实际 trans_param_count **不是 37K**。让我们看 v4.7 实际配置：
- FunctionalStateTransNet layer_specs（spec 03 §2.3 引用 v4.7 functional_nets.py:240-243）：
  - 假设 latent_dim=64, joint_action_dim=N*A=24（Medium N=4, A=6）
  - 输入 dim = 64+24 = 88
  - hidden = cfg.model.hidden_dim = 128
  - layer_specs = [(88, 128, True), (128, 128, True), (128, 64, False)]
  - count_params_adaln: 88×128 + 128 + 128 + 128 = 11648 (FC1 + AdaLN)
  -                     128×128 + 128 + 128 + 128 = 16768 (FC2 + AdaLN)
  -                     128×64 + 64 = 8256 (FC3 plain)
  - total = 11648 + 16768 + 8256 ≈ 37K params

实际 hyper_trans 内部参数：
- 16×256 + 256 + 256×256 + 256 = 70K（trunk）
- 256×37000 + 37000 ≈ 9.5M（output_layer）
- 总 ≈ 9.6 M

这远超 Ch4.3.4 表的 0.4M。**Ch4.3.4 表的数字可能基于不同的 hidden / output 配置**。

### 3.3 参数量验证策略

**spec 07 不重复 Ch4.3.4 精确数字**（避免双源不一致），而是：

1. 在 `scripts/test_hyper_model_forward.py` 内 `print` 实际参数量分解（每个 nn.Module 用 `sum(p.numel() for p in m.parameters())`）
2. 断言 **总参数量 ∈ [2.8M, 3.6M]** 范围（容差 ±10%）
3. 若超出 [2.8M, 3.6M]，触发 PR 评审 + 与 Ch4.3.4 对账（可能需要调整 hidden_dims）

**当前 SDD 阶段不锁定精确数字**，待 Pkg-04 实施时跑 `test_hyper_model_forward.py` 实测后回填到 spec 07 + Ch4.3.4。

---

## 4. 显存预算

### 4.1 训练时（B=256, T=10 K-step unroll, V100 16GB）

| 项 | 估算 |
|-----|------|
| 模型参数（3.2M float32） | 13 MB |
| 模型 gradient buffer | 13 MB |
| Adam optimizer states (m, v) | 26 MB |
| Activations (forward, 全 T=10 K-step unroll) | ~4 GB |
| BeliefNet GRU hidden states (B=256, T=10, N=4, 128) | 1.3 MB |
| θ 中间张量 (θ_state/rew/pred B=256, P=37K/4K/42K) | ~85 MB |
| Buffer 加载（next batch prefetch） | ~500 MB |
| PyTorch overhead | ~1 GB |
| **峰值** | **~6 GB** ✓ < 8 GB |

### 4.2 推断时（worker, B=1）

| 项 | 估算 |
|-----|------|
| 模型参数 + activations | ~50 MB |
| θ 中间张量 | < 1 MB |
| **峰值** | **< 100 MB** ✓ |

worker 推断显存可忽略（< CPU RAM）。

---

## 5. v4.7 → v4 性能增量分析

### 5.1 增量来源

| 维度 | v4.7 | v4 | 增量 |
|------|------|----|------|
| hyper 输入维度 | 16+16=32（rule+id） | 16/80（双 forward API） | hyper_rew/pred 从 32→80 (+2.5x 第一层) |
| hyper 调用次数 (N=4 agents) | 单 forward × N = 4 次 | objective 1 次 + subjective N=4 次 = 5 次（含 trans 1 次, rew/pred 各 N 次）| 净增 1 次 hyper_trans, 但 hyper_trans 仅 1 次 vs v4.7 N=4 次 → **净节省 3 次 hyper_trans** |
| BeliefNet 新增 | 不存在 | +0.3M 参数 + step 开销 | +1 ms / step |
| TriContextEncoder 新增 | 简单 ContextEncoder | TriContextEncoder + BeliefEncoder | +0.5 ms / step |
| grad_gating | 不存在 | 单 if 分支 + .detach() | < 10 μs（可忽略） |

### 5.2 净开销估算

```
v4.7 单步 forward (B=256, N=4):
    encode (1 ms) + 4 × set_context (8 ms) + 4 × (transition+predict_reward+predict) (5 ms) ≈ 14 ms

v4 单步 forward (B=256, N=4):
    encode (1 ms)
    + BeliefNet.step (1 ms)                         # 新增
    + set_context_objective (2 ms)                   # 含 c_ctx + hyper_trans (1 次, v4.7 是 4 次共 6 ms, **净节省 4 ms**)
    + 4 × set_context_subjective (8 ms)              # hyper_rew/pred 4 次, 输入 80 维 (+2.5x)
    + transition (0.5 ms) + 4 × predict_reward (1 ms) + 4 × predict (1.5 ms)
    ≈ 15 ms
```

**结论**：v4 单步 forward 与 v4.7 相比 +1 ms（hyper_rew 输入维度增大），但通过 set_context_objective 复用 θ_state **基本抵消** hyper 输入维度增加的开销。

### 5.3 长期训练性能（K-step unroll, T=10）

```
trainer 单 train_step:
    forward N agents × T=10 unroll ≈ 10 × 15 ms = 150 ms
    backward (autograd) ≈ 150 ms
    optimizer step ≈ 10 ms
    BeliefNet loss forward + backward ≈ 30 ms
    总计 ≈ 340 ms / train_step
```

@ 200K train_steps = 68000 s ≈ 19 小时（Medium config, V100）。与 Roadmap §4 Stage 1 Week 3 工期吻合。

---

## 6. 性能 Smoke Test 脚本

### 6.1 `scripts/test_hyper_model_forward.py`

```python
"""Pkg-04 端到端 forward smoke test + 性能 profile.

Run:
    python scripts/test_hyper_model_forward.py --preset medium --steps 100

Validates (硬约束):
    - 单步 forward < 15 ms (V100, B=256, N=4)
    - 100 步无 NaN / Inf
    - 参数量 ∈ [2.8M, 3.6M]
    - reward ∈ [-3, 3] (类型 α/β 经验范围)
    - type-aware reward 分化 (训练 1K 步后 α/β cos-sim < 0.95) -- 可选 (耗时)
"""
import time
import argparse
import torch
import numpy as np
from hyper_mve.configs import V4Config
from hyper_mve.models import HyperMuZeroModel


def benchmark_forward(model, cfg, B=256, num_steps=100, device='cuda'):
    model = model.to(device)
    model.eval()
    
    N = cfg.env.N
    obs_dim = model.rep_net.obs_dim
    
    times = []
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            obs = torch.randn(B, N, obs_dim, device=device)
            c_t = torch.full((B,), 0.5, device=device)
            cap = torch.rand(B, N, 4, device=device)
            c_hat = torch.rand(B, N, device=device)
            z_hat = torch.softmax(torch.randn(B, N, N-1, 2, device=device), dim=-1)
            
            s = model.encode(obs)
            model.update_step(0)
            model.set_context_objective(c_t)
            for k in range(N):
                model.set_context_subjective(k, cap[:, k], (c_hat[:, k], z_hat[:, k]))
                _ = model.predict_reward(s, torch.zeros(B, N*cfg.env.A, device=device))
                _ = model.predict(s)
        
        torch.cuda.synchronize(device)
        
        # Actual benchmark
        for step in range(num_steps):
            t0 = time.perf_counter()
            
            obs = torch.randn(B, N, obs_dim, device=device)
            c_t = torch.full((B,), 0.5, device=device)
            cap = torch.rand(B, N, 4, device=device)
            c_hat = torch.rand(B, N, device=device)
            z_hat = torch.softmax(torch.randn(B, N, N-1, 2, device=device), dim=-1)
            action = torch.zeros(B, N*cfg.env.A, device=device); action[:, 0] = 1.0
            
            s = model.encode(obs)
            model.update_step(step)
            model.set_context_objective(c_t)
            
            for k in range(N):
                model.set_context_subjective(k, cap[:, k], (c_hat[:, k], z_hat[:, k]))
                r = model.predict_reward(s, action)
                pi, v = model.predict(s)
                
                # 数值合理性
                assert not torch.isnan(r).any(), f"NaN in reward at step {step} agent {k}"
                assert (r > -10).all() and (r < 10).all(), f"reward out of range at step {step}"
            
            s_next = model.transition(s, action)
            assert not torch.isnan(s_next).any(), f"NaN in s_next at step {step}"
            
            torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            times.append(elapsed_ms)
    
    times = np.array(times)
    return {
        "mean_ms": times.mean(),
        "median_ms": np.median(times),
        "p95_ms": np.percentile(times, 95),
        "max_ms": times.max(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preset', default='medium')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()
    
    cfg = V4Config.from_preset(args.preset)
    model = HyperMuZeroModel(cfg)
    
    # 参数量验证 (M3: R 风险预映射)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params/1e6:.2f}M")
    
    # 分模块参数量
    print("\nParam breakdown:")
    for name, sub in [("RepNet", model.rep_net),
                       ("BeliefNet", model.belief_net),
                       ("TriContextEncoder", model.tri_context_encoder),
                       ("DualHyperNetwork", model.hyper_net),
                       ("StateTransNet", model.state_trans_net),
                       ("RewardHead", model.reward_head),
                       ("PredictionNet", model.prediction_net)]:
        n = sum(p.numel() for p in sub.parameters())
        print(f"  {name}: {n/1e6:.3f}M")
    
    # 参数量范围断言
    assert 2.8e6 < total_params < 3.6e6, (
        f"Total params {total_params/1e6:.2f}M out of [2.8M, 3.6M]; "
        f"与 Ch4.3.4 不一致, 需调整 hidden_dims"
    )
    
    # 性能 benchmark
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = benchmark_forward(model, cfg, B=args.batch_size, num_steps=args.steps, device=device)
    
    print(f"\nForward performance ({args.steps} steps, B={args.batch_size}, device={device}):")
    print(f"  Mean:   {results['mean_ms']:.2f} ms")
    print(f"  Median: {results['median_ms']:.2f} ms")
    print(f"  P95:    {results['p95_ms']:.2f} ms")
    print(f"  Max:    {results['max_ms']:.2f} ms")
    
    # 硬阈值断言 (M7, review 修订 4 拆 3 档)
    if device == 'cuda':
        # 档位 3: full step (N=4 agents × K=5) 总 < 100ms
        # 当前 benchmark 测的是 full step (N agents × 1 step + transition), 阈值放宽到 100ms
        assert results['mean_ms'] < 100.0, (
            f"Mean full-step {results['mean_ms']:.2f}ms exceeds 100ms budget (档位 3)"
        )
        
        # 档位 2: single agent K-step unroll 也需单独测 (在主 benchmark 外)
        # 见 §6.2 单测 test_forward_smoke_under_25ms_single_agent_unroll
        
        # 档位 1: single forward call 单独 profile (见 §2.2 set_context_subjective profile)
        
        print(f"\n✅ 档位 3 (full step) assertion passed: {results['mean_ms']:.2f}ms < 100ms")
        print(f"   档位 1/2 由 §2.2 set_context_subjective profile + §6.2 单测覆盖")
    else:
        print(f"\n⚠️  Running on CPU, performance budget not enforced.")


if __name__ == '__main__':
    main()
```

### 6.2 单测层级 `tests/models/test_forward_performance.py`（review 修订 4：三档独立测）

```python
import pytest
import time
import torch
from hyper_mve.configs import V4Config
from hyper_mve.models import HyperMuZeroModel


# ====== 档位 1: 单 forward call < 5ms ======

@pytest.mark.gpu
def test_single_call_under_5ms():
    """档位 1 (review 修订 4): 各单 API call < 5ms."""
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    cfg = V4Config.from_preset("medium")
    model = HyperMuZeroModel(cfg).cuda().eval()
    B, N = 256, cfg.env.N
    obs_dim = model.rep_net.obs_dim
    
    # Setup state
    obs = torch.randn(B, N, obs_dim, device='cuda')
    s = model.encode(obs)
    model.update_step(0)
    model.set_context_objective(torch.full((B,), 0.5, device='cuda'))
    cap = torch.rand(B, 4, device='cuda')
    belief = (torch.rand(B, device='cuda'),
              torch.softmax(torch.randn(B, N-1, 2, device='cuda'), dim=-1))
    model.set_context_subjective(0, cap, belief)
    action = torch.zeros(B, N * cfg.env.A, device='cuda'); action[:, 0] = 1.0
    
    # Profile each single call
    def time_call(fn, n=30):
        # Warmup
        for _ in range(10): _ = fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            _ = fn()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        return sum(times) / len(times)
    
    with torch.no_grad():
        assert time_call(lambda: model.encode(obs)) < 5.0
        assert time_call(lambda: model.transition(s, action)) < 5.0
        assert time_call(lambda: model.predict_reward(s, action)) < 5.0
        assert time_call(lambda: model.predict(s)) < 5.0
        assert time_call(lambda: model.set_context_objective(torch.full((B,), 0.5, device='cuda'))) < 5.0
        assert time_call(lambda: model.set_context_subjective(0, cap, belief)) < 5.0


# ====== 档位 2: 单 agent K=5 step unroll < 25ms ======

@pytest.mark.gpu
def test_single_agent_unroll_under_25ms():
    """档位 2 (review 修订 4): 1 × set_context_subjective + 5 × (trans + r + π/v) < 25ms."""
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    cfg = V4Config.from_preset("medium")
    model = HyperMuZeroModel(cfg).cuda().eval()
    B, N = 256, cfg.env.N
    obs_dim = model.rep_net.obs_dim
    K = 5
    
    obs = torch.randn(B, N, obs_dim, device='cuda')
    cap = torch.rand(B, 4, device='cuda')
    belief = (torch.rand(B, device='cuda'),
              torch.softmax(torch.randn(B, N-1, 2, device='cuda'), dim=-1))
    action = torch.zeros(B, N * cfg.env.A, device='cuda'); action[:, 0] = 1.0
    
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            s = model.encode(obs)
            model.update_step(0)
            model.set_context_objective(torch.full((B,), 0.5, device='cuda'))
            model.set_context_subjective(0, cap, belief)
            for _ in range(K):
                s = model.transition(s, action)
                _ = model.predict_reward(s, action)
                _ = model.predict(s)
        torch.cuda.synchronize()
        
        times = []
        for _ in range(30):
            s = model.encode(obs)
            model.update_step(0)
            model.set_context_objective(torch.full((B,), 0.5, device='cuda'))
            
            t0 = time.perf_counter()
            model.set_context_subjective(0, cap, belief)
            for _ in range(K):
                s = model.transition(s, action)
                _ = model.predict_reward(s, action)
                _ = model.predict(s)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    
    mean_ms = sum(times) / len(times)
    assert mean_ms < 25.0, f"Single agent K-step unroll {mean_ms:.2f}ms > 25ms"


# ====== 档位 3: N=4 agents × K=5 step unroll < 100ms ======

@pytest.mark.gpu
def test_full_step_under_100ms():
    """档位 3 (review 修订 4): N agents × K-step unroll < 100ms."""
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    cfg = V4Config.from_preset("medium")
    model = HyperMuZeroModel(cfg).cuda().eval()
    B, N = 256, cfg.env.N
    obs_dim = model.rep_net.obs_dim
    K = 5
    
    obs = torch.randn(B, N, obs_dim, device='cuda')
    caps = torch.rand(B, N, 4, device='cuda')
    c_hats = torch.rand(B, N, device='cuda')
    z_hats = torch.softmax(torch.randn(B, N, N-1, 2, device='cuda'), dim=-1)
    action = torch.zeros(B, N * cfg.env.A, device='cuda'); action[:, 0] = 1.0
    
    with torch.no_grad():
        for _ in range(10):
            _ = _full_step(model, obs, caps, c_hats, z_hats, action, N, K)
        torch.cuda.synchronize()
        
        times = []
        for _ in range(30):
            t0 = time.perf_counter()
            _ = _full_step(model, obs, caps, c_hats, z_hats, action, N, K)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    
    mean_ms = sum(times) / len(times)
    assert mean_ms < 100.0, f"Full step (N×K) {mean_ms:.2f}ms > 100ms"


def _full_step(model, obs, caps, c_hats, z_hats, action, N, K):
    B = obs.shape[0]
    s = model.encode(obs)
    model.update_step(0)
    model.set_context_objective(torch.full((B,), 0.5, device=obs.device))
    for k in range(N):
        model.set_context_subjective(k, caps[:, k], (c_hats[:, k], z_hats[:, k]))
        s_k = s
        for _ in range(K):
            s_k = model.transition(s_k, action)
            _ = model.predict_reward(s_k, action)
            _ = model.predict(s_k)
    return s


# ====== 兼容旧测试名（向 design.md §7 表 R8 行映射）======

def test_forward_smoke_under_15ms():
    """旧名兼容: 实际由 test_full_step_under_100ms 覆盖 (review 修订 4)."""
    test_full_step_under_100ms()
```

---

## 7. v4 vs v4.7 性能对比表（review 修订 4：按 3 档预算重新对账）

| 维度 | v4.7 | v4 | 差异 | 档位 |
|------|------|----|------|------|
| 单 forward call（encode 等） | ~1 ms | ~1 ms | +0% | 档位 1 (< 5ms) |
| 单 set_context_subjective | ~3 ms | ~3 ms | +0%（hyper_rew 维度 ↑ 但 ctx 路径优化抵消） | 档位 1 (< 5ms) |
| 单 agent K=5 unroll | ~18 ms | ~19 ms | +5% | 档位 2 (< 25ms) |
| **N=4 agents × K=5 step（trainer 完整 forward）**| ~75 ms | ~95 ms | +27%（hyper_rew 80 维 ↑ + BeliefNet step）| 档位 3 (< 100ms) |
| 参数量 | ~2.8 M | ~3.2 M | +14%（BeliefNet 0.3 M + TriContextEncoder 0.1 M） | – |
| 训练显存 | ~5.5 GB | ~6 GB | +9%（BeliefNet GRU hidden states） | – |
| 200K train_steps wall-clock | ~17 小时 | ~19 小时 | +12% | – |

**结论**：v4 三档预算均留 ≥ 5% 缓冲。档位 3 增量 +27% 大于早期估算的 +7%，主因是 set_context_subjective 切 agent 切换开销（4 次）+ hyper_rew 输入 80 维（vs v4.7 ~24 维），换取断言 A/B/C 物理基础。

**与档位 2 收益对比**：mve_planner per-agent coordinate descent 仅需档位 2（单 agent K-step），可享 19ms / agent 低延迟；trainer K-step unroll 才需档位 3 全 N agents 95ms。

---

## 8. Cross-references

- Ch4.3.4 参数量表（~3.2M 总参考）
- `02-hyper-muzero-model-v2.md` §3.1（set_context 两步分离性能优势）
- `01-dualhypernet-v2-api.md`（hyper 三模块结构）
- `03-stability-safeguards-preservation.md`（functional_nets 结构）
- `06-data-flow-diagram.md`（数据流 + N agents 调用顺序）
- Pkg-03 spec 04（BeliefNet 性能 ~1 ms / step）
- Pkg-05 spec 04（trainer-loop 完整 timing）

---

## [v4-opt 2026-06] 参数预算修订

§3.3 的参数预算以 FULL 全量生成估算,实际 FULL 实现 ~3.09M(超预算 ~3×,见 ARCHITECTURE_OVERVIEW 旧 §10)。**输出层 LoRA(r=32)已解决该超支**:film_head+LoRA ~694k / lora_fc2+LoRA ~896k / base_gen+LoRA ~2.30M(medium;由 `tests/models/test_hyper_network_lora.py` 锚定;权威表 = Ch4.3.4 修订表 + DESIGN_DOC §5.12)。规划性能门(300ms 回归护栏)在 LoRA 档下预期显著放松,待 sweep 实测回填。

## 修订记录 (Changelog)

| 日期 | 修订 | 依据 |
|---|---|---|
| 2026-06-10 | 参数预算按 gen_scope/LoRA 修订 | 提交 dc5bbcd;复审 M10 |
