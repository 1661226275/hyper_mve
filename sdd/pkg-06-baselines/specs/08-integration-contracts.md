# Spec 08: Integration Contracts — 与 Pkg-05 工厂 + Pkg-07/08 的硬契约

> 父文档：[`../proposal.md`](../proposal.md) §4 · [`../design.md`](../design.md) §7（M2 伪签名）· §8（验证策略）
> **断言覆盖**：把全部 5 variant 收束到"仅 model 类不同"的复用契约——是断言 A/B/C 公平性的**系统级保证**。
> **上游对齐**：五条复用约束逐字对齐 Pkg-05 spec 08 §3.2；工厂签名逐字对齐 Pkg-05 spec 08 §3.1（lines 136-137：136 import / 137 call）。
> **Supersede 声明**：Pkg-04 spec 08 §5.1 illustrative 草案 `create_input_wide_baseline(cfg, total_params_target)`（每 variant 一函数）被本 spec 联合 spec 01 共同 supersede——本契约层文档定义最终对外接口（单一 `create_baseline_model(cfg, variant)`）。Day 6 grep 须确认实施只命中单一签名。

---

## 1. Purpose

本 spec 是 Pkg-06 的**对外硬契约层**，规定本包向 Pkg-05/07/08 承诺的稳定接口 + 五条复用约束，并把前 7 个 spec 的契约汇总为"跨包稳定性承诺表"（M2）。它不新增模型逻辑，只锁边界。

| 契约族 | 内容 | 落点 |
|--------|------|------|
| 复用约束 | 同一 trainer/worker/buffer/loss，仅 model 类不同 | §2（C6-REUSE1）|
| API 稳定性承诺 | import 路径 + 工厂/backbone/7-API 签名不变更 | §3（M2）|
| 上游消费契约 | Pkg-05 `create_trainer_for_baseline` 调用模式 | §4 |
| 下游契约 | Pkg-07 evaluator / Pkg-08 train_main.py | §5 |

---

## 2. 五条等参复用约束（逐字对齐 Pkg-05 spec 08 §3.2）

所有 5 个 baseline variant 与 hyper **共享同一**：

```
1. 同一 MuZeroTrainer 类      —— 无 baseline-specific train_step
2. 同一 Worker 类             —— 采集行为一致 (epsilon-greedy + MVE planner)
3. 同一 EpisodeReplayBuffer 类
4. 同一 compose_total_loss 函数
5. 仅 model 类不同            —— 唯一变量
```

> **公平性论证**：若某 baseline 用了不同的 trainer/loss，则"hyper 更优"可能归因于训练流程差异而非模型容量。五条约束把差异压缩到 model 类——这是断言 A/B/C 的系统级前提。`create_baseline_model` 返回的模型直接喂给同一 `MuZeroTrainer`（spec 01 §2.3 / 工厂归属 spec 01）。

### 2.1 复用验证（C6-REUSE1）

```python
def test_baseline_reuses_same_trainer_class(cfg):
    """5 variant + hyper 经各自 model 构造的 trainer 是同一 MuZeroTrainer 类."""
    from hyper_mve.training.muzero_trainer import MuZeroTrainer
    from hyper_mve.baselines import create_baseline_model, BASELINE_REGISTRY
    from hyper_mve.models import HyperMuZeroModel
    trainers = [MuZeroTrainer(cfg, HyperMuZeroModel(cfg))]
    for v in BASELINE_REGISTRY:
        trainers.append(MuZeroTrainer(cfg, create_baseline_model(cfg, v)))
    assert all(type(t) is MuZeroTrainer for t in trainers)
```

---

## 3. API 稳定性承诺表（M2 伪签名先行）

本包向下游承诺以下接口稳定（Pkg-07/08 可依赖，本包不变更）：

### 3.1 import 路径（稳定）

```python
from hyper_mve.baselines import create_baseline_model, BASELINE_REGISTRY   # Pkg-05 spec 08 line 136
from hyper_mve.baselines.shared_backbones import (
    create_rep_net, create_belief_net, create_tri_context_encoder,
    count_conditioning_params,
)
```

### 3.2 稳定签名汇总（逐字引前序 spec）

| API | 签名 | 归属 spec |
|-----|------|-----------|
| 工厂 | `create_baseline_model(cfg, variant: str)` | spec 01 §2.2 |
| registry | `BASELINE_REGISTRY: dict[str, type]`（5 key）| spec 01 §2.2 |
| backbone | `create_rep_net / create_belief_net / create_tri_context_encoder(env_cfg, model_cfg)` | spec 02 §2.2 |
| 度量 | `count_conditioning_params(model) -> int` | spec 02 §2.2 |
| 7-API | `update_step / set_context_objective / set_context_subjective / encode / transition / predict_reward / predict` | spec 06 §2.1 |

### 3.3 5 模型类（稳定，归属 spec）

| 模型类 | variant | 归属 spec |
|--------|---------|-----------|
| `InputWideBaselineModel` / `InputDeepBaselineModel` | input_wide / input_deep | spec 03 |
| `MAMuZeroBaselineModel` | ma_muzero | spec 04 |
| `NoBeliefBaselineModel` / `ExplicitTypeRewardBaselineModel` | no_belief / rewardhead_explicit_type | spec 05 |

---

## 4. 上游消费契约（Pkg-05，本包仅对账不实现）

```python
# Pkg-05 create_trainer_for_baseline 内部 (spec 08 §6.2):
def create_trainer_for_baseline(cfg, variant):
    if variant == "hyper":
        model = HyperMuZeroModel(cfg)                  # 不走本包工厂 (spec 01 D3)
    elif variant in ("oracle_only", "infer_only"):
        model = HyperMuZeroModel(cfg)                  # + curriculum override
    else:
        model = create_baseline_model(cfg, variant)    # ← 本包提供 (spec 01)
    return MuZeroTrainer(cfg, model)                    # ★ 同一 trainer 实例
```

> 本包**不修改** Pkg-05 任何代码；上述模式是 Pkg-05 spec 08 §6.2 已锁的调用契约，本节仅复刻对账。`hyper`/`oracle_only`/`infer_only` 走 `HyperMuZeroModel` + curriculum（不进工厂，spec 01 头部映射表）。

---

## 5. 下游契约（Pkg-07 / Pkg-08）

### 5.1 Pkg-07 Eval

baseline 与 hyper 走**同一 evaluator**——因为 7-API 一致（spec 06），evaluator 对 model 类型透明。本包不提供专用 evaluator。

### 5.2 Pkg-08 Experiments

```
- train_main.py --variant baseline_input_wide   # CLI 前缀剥离 → "input_wide" → 工厂 (spec 01 头部映射表)
- 断言 B 主对比: hyper vs input_wide vs input_deep  (等参 spec 07)
- 断言 A 对比:   hyper vs ma_muzero vs rewardhead_explicit_type  (spec 04 / spec 05)
- 断言 C 对比:   hyper vs no_belief  (Ablation 7, spec 05)
- Ablation 6.x:  rewardhead_explicit_type  (spec 05)
```

> CLI `--variant` 前缀剥离（`baseline_*` → 无前缀）是 Pkg-08 train_main.py 职责；本包工厂只认无前缀 5 variant（spec 01 头部映射表）。

---

## 6. 边界与不改上游（验证）

| 约束 | 验证 |
|------|------|
| 不修改 Pkg-01..05 任何 spec/代码 | `git diff` 确认 `sdd/pkg-01..05/` + `hyper_mve/**` 代码零改动 |
| 不引入新依赖 | 仅 PyTorch + numpy + 已发布上游接口 |
| 工厂签名逐字一致 | 与 Pkg-05 spec 08 §3.1 line 136 比对 |
| 五条复用约束 | C6-REUSE1 `test_baseline_reuses_same_trainer_class` |
| cfg 新字段消费态声明 | 4 字段待 Pkg-01 spec 05 同步（spec 01 §1.1），本包不改上游 |

---

## 7. Acceptance Criteria

### 7.1 单元测试（`tests/baselines/test_integration.py`）

```python
import pytest
from hyper_mve.configs import V4Config
from hyper_mve.training.muzero_trainer import MuZeroTrainer
from hyper_mve.baselines import create_baseline_model, BASELINE_REGISTRY
from hyper_mve.models import HyperMuZeroModel


@pytest.fixture
def cfg():
    return V4Config.from_preset("medium")


# ====== C6-REUSE1: 同一 MuZeroTrainer/Worker/Buffer/loss ======

def test_baseline_reuses_same_trainer_class(cfg):
    ref = MuZeroTrainer(cfg, HyperMuZeroModel(cfg))
    for v in BASELINE_REGISTRY:
        t = MuZeroTrainer(cfg, create_baseline_model(cfg, v))
        assert type(t) is type(ref)
        assert type(t.worker) is type(ref.worker)
        assert type(t.buffer) is type(ref.buffer)


def test_import_paths_stable(cfg):
    """承诺的 import 路径可用 (下游依赖)."""
    from hyper_mve.baselines import create_baseline_model, BASELINE_REGISTRY
    from hyper_mve.baselines.shared_backbones import (
        create_rep_net, create_belief_net, create_tri_context_encoder,
        count_conditioning_params,
    )
    assert len(BASELINE_REGISTRY) == 5
```

### 7.2 契约一致性 grep（Day 6）

- 五条复用约束文字与 Pkg-05 spec 08 §3.2 逐字比对。
- 全仓无 baseline-specific trainer/loss（grep `class.*Baseline.*Trainer` 应空）。

---

## 8. Cross-references

- [`01-baselines-factory-and-registry.md`](./01-baselines-factory-and-registry.md)（工厂 + registry + reject hyper）
- [`02-shared-backbones.md`](./02-shared-backbones.md)（共享后端 + `count_conditioning_params`）
- [`03-input-conditioned-baselines.md`](./03-input-conditioned-baselines.md)（input_wide / input_deep）
- [`04-ma-muzero-baseline.md`](./04-ma-muzero-baseline.md)（ma_muzero）
- [`05-belief-type-ablation-baselines.md`](./05-belief-type-ablation-baselines.md)（no_belief / rewardhead_explicit_type）
- [`06-model-7api-conformance.md`](./06-model-7api-conformance.md)（7-API 一致性是复用前提）
- [`07-param-fairness-and-lr-sweep.md`](./07-param-fairness-and-lr-sweep.md)（等参公平协议）
- Pkg-05 spec 08 §3.1（工厂签名）+ §3.2（五条复用约束）+ §6.2（CLI + create_trainer_for_baseline）
- Pkg-04 spec 02（7-API）
