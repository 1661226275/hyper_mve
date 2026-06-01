# Spec 07: Param Fairness & LR Sweep — 等参双阈值 + LR sweep 协议 + 性能护栏

> 父文档：[`../proposal.md`](../proposal.md) §2.3 · [`../design.md`](../design.md) §4 D5 · §9.1 OQ-1
> **断言覆盖**：**断言 B 的生死线**——等参公平性若不成立，"hyper 赢是因为参数多"会推翻整个断言 B。
> **上游对齐**：等参统计口径逐字对齐 Ch5 §5.6 step1（只算条件化子系统）；LR sweep 对齐 step2；主对比对齐 step3。

---

## 1. Purpose

规定 baseline 与 hyper 的"公平比较"量化协议，三层落地（对齐 Ch5 §5.6 三步）：

| 层 | 内容 | 单测 / 产物 |
|----|------|-------------|
| step1 参数预算 | 条件化消费子系统双阈值（warn≤5% / fail≤10%）| `test_baseline_param_count_within_5pct`（C6-FAIR1）|
| step2 LR sweep | 每个 variant 在最佳 LR 下比较（排除"调参不公平"）| LR sweep 协议（§3）|
| step3 wall-clock 等价性 | 豁免 variant 的步数等价性度量（OQ-1 落地）| §4 报告协议 |
| 性能护栏 | 单步 forward 预算分档 + 单测护栏（G9）| `test_baseline_forward_budget`（C6-PERF1）|

---

## 2. 等参预算（step1，双阈值）

### 2.1 统计口径（条件化消费子系统，D5）

复用 spec 02 `count_conditioning_params(model)`——单一度量入口，排除共享 RepNet/BeliefNet/TriContextEncoder，仅算：

- θ 生成器（hypernet 头，若有）
- 接收 ctx_aug 的功能网（input baseline 为加宽/加深版本）

### 2.2 双阈值（warn 先报 / fail 才挂）

设 `P_hyper = count_conditioning_params(hyper)`，`P_v = count_conditioning_params(variant)`，相对偏差 `δ = |P_v − P_hyper| / P_hyper`：

| δ | 判定 | 动作 |
|---|------|------|
| δ ≤ 5% | **PASS** | 通过 |
| 5% < δ ≤ 10% | **WARN** | 报警但不挂红（记入 PR 说明，需调旋钮逼近）|
| δ > 10% | **FAIL** | 单测挂红（调 `baseline_wide_hidden_dim` / `baseline_deep_layers` 重逼近）|

### 2.3 双阈值适用范围（D5 豁免）

| variant | 双阈值 | 理由 |
|---------|--------|------|
| `input_wide` | **强制**（warn 5% / fail 10%）| 同构条件化子系统（加宽功能网）|
| `input_deep` | **强制** | 同构条件化子系统（加深功能网）|
| `no_belief` | **强制** | 与 hyper 同构（hypernet 3 头 + 3 功能网，仅 belief 路置零）|
| `ma_muzero` | **豁免** | 结构性偏小（无生成器、无 ctx 拼接）→ 改报 wall-clock（§4）|
| `rewardhead_explicit_type` | **豁免** | 结构性差异（去 hyper_rew + type 分支）→ 改报 wall-clock（§4）|

### 2.4 加宽/加深逼近流程（input baseline）

```
1. 算 P_hyper = count_conditioning_params(hyper)
2. input_wide: 二分 baseline_wide_hidden_dim, 使 P_v 落入 [0.95, 1.05] * P_hyper
   input_deep: 二分 baseline_deep_layers (整数, 落入 [0.90, 1.10] 即接受, 层数离散粒度粗)
3. 锁定旋钮值写入 cfg.model.* (Pkg-01 spec 05 同步), 标注逼近结果 δ
4. test_baseline_param_count_within_5pct 守门 (回归保护)
```

> 层数离散 → input_deep 可能无法精确落入 5%；接受落入 10%（WARN），并在 PR 说明记录最接近的层数及 δ。

---

## 3. LR sweep 协议（step2，公平调参）

排除"hyper 赢只是因为 LR 调得更好"的质疑：

```
- 每个 variant (含 hyper) 各自在同一 LR 网格上 sweep:
    lr ∈ {1e-4, 3e-4, 1e-3, 3e-3}  (与主实验 cfg 对齐, 不为 baseline 特调)
- 每个 (variant, lr) 跑 K_seed 个种子 (K_seed ≥ 3), 取验证回报均值
- 各 variant 取自己的最佳 LR, 再做 step3 主对比
- 报告: 每个 variant 的最佳 LR + 该 LR 下的回报曲线 (附录表)
```

> 关键公平点：**不为某个 baseline 单独扩大 LR 网格或加 trick**——网格对所有 variant 一致（含 hyper），各取自己最优。这样"hyper 更优"不能归因于调参不公平。

---

## 4. 豁免 variant 的 wall-clock 步数等价性（OQ-1 落地）

design §9.1 OQ-1 在此详化。`ma_muzero` / `rewardhead_explicit_type` 豁免参数对齐，改用**达标步数等价性**证明对照不是"参数太少导致欠拟合"：

### 4.1 度量定义

```
- 在各自最佳 LR (step3) 下, 记每个 variant 的验证回报曲线 R_v(step).
- 达标阈值 R* := 0.80 * R_oracle_only_plateau
    (oracle_only 收敛平台回报的 80%; oracle_only 是已知上界对照, 见 Pkg-05 spec 08 §6.2).
- 稳定窗口 W := 最近 100k 训练步 (该窗内每 5k 步一次评估 = 20 个评估点).
- step_to_R*(v) := R_v 首次满足 "连续 W=100k 步窗内所有评估点 ≥ R*" 的窗起始步数.
- 报告: 豁免 variant 的 step_to_R* vs hyper 的 step_to_R*.
```

### 4.2 判据（证明欠拟合不成立）

豁免 variant 即便参数偏小，只要其 `R_v` 在 step budget 内 **saturate（饱和后不再随步数显著上升）**，即证明其劣势**不是欠训练/欠容量**，而是结构性（共享头平均梯度 / 离散 type 分支）。报告须含：

- `step_to_R*` 对比表（豁免 variant vs hyper）
- 饱和判据：最后 30% 训练步的回报斜率 < ε，其中 **ε := 0.05 * R* / 100k 步**
    （即每 100k 步回报上升不足 R* 的 5% 视为已平; 斜率用最后 30% 步的线性回归估计）
- 若豁免 variant 未 saturate（斜率 ≥ ε）→ 不能下断言 A 结论（需延长 budget 重跑）

> 这把"豁免参数对齐"的代价显式化：用收敛行为而非参数量来论证公平。

---

## 5. 性能护栏（G9，单步 forward 预算）

### 5.1 预算分档

| 部件 | 预算（medium preset, CPU 参考）| 说明 |
|------|-------------------------------|------|
| `encode` 单次 | ≤ 1.5× hyper.encode | 共享 RepNet，应近似相等 |
| `transition` 单次 | input baseline ≤ 2.0× hyper.transition | 加宽/加深允许更慢，设上界防失控 |
| 完整 7-API 一轮 | ≤ 2.5× hyper 一轮 | 防某 variant 引入意外开销 |

> 预算是**护栏（防回归失控）**而非精度指标；倍率宽松，只拦截数量级异常。

---

## 6. Acceptance Criteria

### 6.1 单元测试（`tests/baselines/test_param_fairness.py`）

```python
import pytest
import time
import warnings
import torch
from hyper_mve.configs import V4Config
from hyper_mve.baselines import create_baseline_model
from hyper_mve.baselines.shared_backbones import count_conditioning_params
from hyper_mve.models import HyperMuZeroModel

FAIR_VARIANTS = ("input_wide", "input_deep", "no_belief")   # 强制双阈值
EXEMPT_VARIANTS = ("ma_muzero", "rewardhead_explicit_type") # 豁免


@pytest.fixture
def cfg():
    return V4Config.from_preset("medium")


# ====== C6-FAIR1: 条件化消费子系统 ≤5% warn / ≤10% fail ======

@pytest.mark.parametrize("variant", FAIR_VARIANTS)
def test_baseline_param_count_within_5pct(cfg, variant):
    p_hyper = count_conditioning_params(HyperMuZeroModel(cfg))
    p_v = count_conditioning_params(create_baseline_model(cfg, variant))
    delta = abs(p_v - p_hyper) / p_hyper
    if delta > 0.10:
        pytest.fail(f"{variant} 条件化参数偏差 {delta:.1%} > 10% (断言 B 不公平)")
    elif delta > 0.05:
        # 5%~10%: 显式发 UserWarning 软提示 (不挂红). 用 warnings.warn 真正触发,
        # 而非裸 pytest.warns(UserWarning) —— 后者作为语句是 no-op (需作 context manager 用).
        warnings.warn(
            f"{variant} 条件化参数偏差 {delta:.1%} 落在 5%~10% WARN 带 (调小旋钮逼近)",
            UserWarning,
        )
    assert delta <= 0.10


@pytest.mark.parametrize("variant", EXEMPT_VARIANTS)
def test_exempt_variants_not_param_asserted(cfg, variant):
    """豁免 variant 不强制对齐, 仅确认其条件化子系统结构性不同于 hyper."""
    p_hyper = count_conditioning_params(HyperMuZeroModel(cfg))
    p_v = count_conditioning_params(create_baseline_model(cfg, variant))
    assert p_v != p_hyper      # 结构性差异 (若相等说明实现退化, 需查)


# ====== C6-PERF1: 单步 forward 预算护栏 ======

@pytest.mark.parametrize("variant", ("input_wide", "input_deep"))
def test_baseline_forward_budget(cfg, variant):
    B = 8
    obs = torch.randn(B, cfg.env.N, cfg.env.obs_dim)
    hyper = HyperMuZeroModel(cfg)
    model = create_baseline_model(cfg, variant)

    def _time_transition(m):
        m.set_context_objective(torch.zeros(B))
        m.set_context_subjective(0, torch.zeros(B, 4),
                                 (torch.zeros(B), torch.zeros(B, cfg.env.N - 1, 2)))
        s = m.encode(obs); a = torch.zeros(B, cfg.env.N * cfg.env.A)
        t0 = time.perf_counter()
        for _ in range(20):
            m.transition(s, a)
        return time.perf_counter() - t0

    assert _time_transition(model) <= 2.0 * _time_transition(hyper)
```

> step2 LR sweep / step3 主对比 / wall-clock 报告是**实验协议**（Pkg-08 执行），本 spec 规定其判据，不在单测内跑全量训练。

---

## 7. Cross-references

- [`02-shared-backbones.md`](./02-shared-backbones.md)（`count_conditioning_params` 单一度量入口）
- [`03-input-conditioned-baselines.md`](./03-input-conditioned-baselines.md)（input_wide/input_deep 是双阈值主对象 + 加宽/加深旋钮）
- [`04-ma-muzero-baseline.md`](./04-ma-muzero-baseline.md)（ma_muzero 豁免 → wall-clock 步数等价性）
- [`05-belief-type-ablation-baselines.md`](./05-belief-type-ablation-baselines.md)（no_belief 进双阈值；explicit_type 豁免）
- [`08-integration-contracts.md`](./08-integration-contracts.md)（公平协议是断言 B 的对外契约）
- Ch5 §5.6（等参公平协议 step1 参数统计 / step2 LR sweep / step3 主对比）
