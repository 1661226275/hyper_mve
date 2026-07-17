# Pkg-01: Foundation Schema & Config Restructure — Proposal

| 元信息 | 值 |
|--------|----|
| **包 ID** | `pkg-01-foundation-schema` |
| **状态** | Draft, awaiting review |
| **工期估计** | 0.5 周（3-4 天） |
| **GPU 算力** | 0（纯结构 + 单测） |
| **依赖** | 无（项目入口） |
| **被依赖** | Pkg-02 至 Pkg-08（所有后续包） |
| **论文章节** | 为 Ch3 / Ch4 形式化奠定命名约定（不直接填数据） |
| **PR 体量** | ~8 新文件 + 1 重写 + ~30 文件移动归档 |

---

## 1. Why（为什么需要这个包）

### 1.1 v4.7 现状的三个结构性痛点

**痛点 A：配置单类化爆炸**
当前 `hyper_mve/config.py:9-123` 是单一 `BaseConfig` 类，**70 个参数平铺在一起**，包含环境维度 (`obs_dim_n=[16,16,16,14]`)、模型维度、训练超参、HyperNet 超参、MVE planner 超参、Adversarial Freeze（已禁用的 v4.2 遗留）、Phase 5 ChunkedHMLP（已弃用）等 11 个类别。该类对 NonStationaryTag 环境**硬编码绑定**——切到 ResourceCommons 后所有维度字段（`num_agents=4`, `num_actions=5`, `obs_dim_n=[16,16,16,14]`）都不再有意义，但因为是同一类的字段，**无法增量替换**。

**痛点 B：数据 schema 隐式化**
v4 论文需要**显式**的 type α/β 类型机制（Ch3.5.1）、capability 向量 (η, φ_fov, ν, ζ)（Ch3.6）、六块观测结构（Ch3.7）、新增的 Δ / τ / cap / ẑ buffer 字段（Ch5.6.1）。当前 v4.7 没有任何 schema 定义模块——这些信息**散落在 env 实现、worker 收集逻辑、buffer 存储格式三处**，互相不验证不约束。如果让 Pkg-02（env）、Pkg-04（model）、Pkg-05（trainer）三个包**各自定义类型**，将不可避免地产生不兼容的数据契约（如 type one-hot 维度 2 vs 4、cap 字段顺序错位、observation 块边界不一致），最终需要大量胶水代码与运行时断言来弥补。

**痛点 C：v4.7 与 v4 并存的可复现性需求**
论文 Ch6.12（Failure Case Analysis）与补充材料需要 v4.7 → v4 的对照实验数据；同时如果 Decision Point 1（Roadmap §6.3, Week 7）触发断言 A 失败，**需要回退路径**——这要求 v4.7 代码完整可运行而非散落片段。当前若直接在 v4.7 文件上重写，将永久丢失 v4.7 的工作版本，违反 Roadmap §2.4 "代码基线锁定 `git tag v4.7-final`" 的硬性要求。

### 1.2 对应路线图条款

| Roadmap / Chapter 条款 | 本包如何响应 |
|------------------------|----------------|
| Roadmap §2.4：`git tag v4.7-final` 锁定基线，启用回滚 | 本包执行 git tag + `_legacy_v4_7/` 归档 |
| Roadmap §4 Stage 1 Pre-flight：环境层重构启动 | 本包提供 schema 公共契约，为 Stage 1 解锁 |
| Ch3.5.1 类型机制 τ_i ∈ {α, β}，固定 episode 内不变 | `AgentType` Enum + 不可变 dataclass |
| Ch3.6 capability 向量 4 维（含符号 $\phi^{fov}$ 防与 φ(c) 冲突） | `CapabilityVector` dataclass，字段命名遵循 v4 符号 |
| Ch3.7 六块观测 + Self-Info 原则 | `ObservationLayout` 静态常量，type 块仅含自身 one-hot |
| Ch5.6.1 buffer 新字段 (o, a, r, Δ, π_mve, v, τ, cap, ĉ, ẑ) | `TimeStepRecord` dataclass 显式声明 10 字段 |
| Ch3.9 三难度配置 Easy/Medium/Hard | `configs/{easy,medium,hard}.py` 各自完整 preset |

### 1.3 不做此包的反例代价

若跳过本包直接动 Pkg-02 env：
- Pkg-02 会自定义一份 type / capability schema → Pkg-04 model 复用时**必然偏移**（如 type 表示从 Enum 偏到 int）
- Pkg-05 trainer 若用 dict 自定义 buffer record，与 Pkg-02 env step 返回的 dict 字段名错位
- v4.7 与 v4 共存时如果不归档将产生**导入歧义**：`from hyper_mve.models.hyper_muzero_model import OracleHyperMuZeroModel` 到底引到 v4.7 还是 v4？

**结论**：本包是"廉价但不可省略"的公共契约层。0.5 周投入避免后续 7 包的胶水代码累积。

---

## 2. What Changes（具体改动清单）

### 2.1 新增文件（8 个）

| 路径 | 行数估计 | 内容 |
|------|----------|------|
| `hyper_mve/schemas/__init__.py` | ~30 | 暴露 5 个 schema 类型 + 2 个枚举 |
| `hyper_mve/schemas/agent_type.py` | ~60 | `AgentType` Enum (ALPHA=0, BETA=1) + `one_hot()`, `from_index()`, `count_in_assignment()` 工具 |
| `hyper_mve/schemas/capability.py` | ~80 | `CapabilityVector` frozen dataclass (η, $\phi^{fov}$, ν, ζ) + `sample_default()` + Ch3.6 范围验证 |
| `hyper_mve/schemas/observation.py` | ~100 | `ObservationLayout` 静态类（6 块维度常量 + `total_dim(N, K)`）+ `pad_resource_block()` 等切片工具 |
| `hyper_mve/schemas/buffer_record.py` | ~90 | `TimeStepRecord` frozen dataclass (10 字段) + `to_arrays()`, `from_arrays()` 序列化 |
| `hyper_mve/schemas/context.py` | ~50 | `ContextSchema` 常量类 (d_c=16, d_role=32, d_belief=48) + 三 path 接口签名 |
| `hyper_mve/schemas/_constants.py` | ~40 | 全局常量（α/β 数值、Q_max、各类默认范围），跨 schema 引用 |
| `tests/schemas/__init__.py` | 0 | – |

### 2.2 重写文件（1 个）

| 路径 | v4.7 行数 | v4 行数估计 | 改动 |
|------|-----------|-------------|------|
| `hyper_mve/config.py` | 123 | ~250（分拆后总和） | **删除**原 `BaseConfig`；改为 import + re-export 5 个 sub-config dataclass |

实际拆分：

| 新文件 | 行数估计 | 内容 |
|--------|----------|------|
| `hyper_mve/configs/__init__.py` | ~20 | 暴露 `V4Config`, 3 preset |
| `hyper_mve/configs/env_config.py` | ~60 | `EnvConfig`：N, L, K, M, T_max, c_mode, ρ_τ, α_min/max, κ_f, θ_f, d_nbr, η/ζ/ν/fov 采样范围 |
| `hyper_mve/configs/model_config.py` | ~80 | `ModelConfig`：latent_dim, hidden_dim, d_c/d_role/d_belief, hyper_hidden_dims, rew_hidden_dims, output_scale_init, AdaLN 参数 |
| `hyper_mve/configs/train_config.py` | ~100 | `TrainConfig`：max_train_steps, batch_size, buffer_size, lr, lr_schedule, warmup, EMA τ, K-unroll, n-step, loss weights, curriculum boundaries (0.3/0.7), grad gating step (5000) |
| `hyper_mve/configs/mup_config.py` | ~40 | `MupConfig`：base_shape 占位（实际值由 Pkg-07 填充） |
| `hyper_mve/configs/eval_config.py` | ~60 | `EvalConfig`：evaluate_freq, episodes, c-segment 边界, zero-shot c 集合, bell curve 类型扫描点 |
| `hyper_mve/configs/v4_config.py` | ~80 | `V4Config` 组合器 + `from_preset(name)` |
| `hyper_mve/configs/presets/easy.py` | ~50 | Easy 完整字段（N=2, L=8, K=8, T=100, 1α+1β） |
| `hyper_mve/configs/presets/medium.py` | ~50 | Medium 完整字段（N=4, L=16, K=20, T=200, 2α+2β）— **主对比配置** |
| `hyper_mve/configs/presets/hard.py` | ~50 | Hard 完整字段（N=8, L=24, K=40, T=300, 4α+4β） |
| `hyper_mve/configs/presets/legacy.py` | ~40 | `LegacyConfig`：保留 freeze_enabled, chunk_*, w_rew_diversity 等 v4.7 调试字段（默认禁用） |

`hyper_mve/config.py` 保留为 **backward-compat shim**：`from hyper_mve.configs import V4Config; BaseConfig = V4Config.from_preset("medium")` 一行，加 DeprecationWarning。

### 2.3 单元测试（9 个）

| 路径 | 覆盖 |
|------|------|
| `tests/schemas/test_agent_type.py` | Enum 不变性、one-hot 维度、from_index 边界 |
| `tests/schemas/test_capability_vector.py` | frozen 检查、采样范围、Ch3.6 字段顺序 |
| `tests/schemas/test_observation_layout.py` | 6 块总维度公式、pad/slice 工具 |
| `tests/schemas/test_buffer_record.py` | 10 字段完整、序列化往返、numpy 类型一致 |
| `tests/schemas/test_context_schema.py` | d_c+d_role+d_belief = C_aug 总维度 |
| `tests/configs/test_env_config.py` | 三 preset 字段对照 Ch3.9 Table |
| `tests/configs/test_v4_config.py` | compose 顺序、preset 覆盖、JSON 序列化往返 |
| `tests/configs/test_legacy_config.py` | 默认禁用、显式启用时不污染主 config |
| `tests/test_legacy_import_warning.py` | `import hyper_mve._legacy_v4_7.*` 触发 DeprecationWarning |

### 2.4 归档操作（git mv）

将 v4.7 完整代码移动到 `hyper_mve/_legacy_v4_7/`（保持子目录结构）：

| 源路径 | 目标路径 |
|--------|----------|
| `hyper_mve/envs/non_stationary_tag.py` | `hyper_mve/_legacy_v4_7/envs/non_stationary_tag.py` |
| `hyper_mve/envs/ns_environment.py` | `hyper_mve/_legacy_v4_7/envs/ns_environment.py` |
| `hyper_mve/envs/make_env.py` | `hyper_mve/_legacy_v4_7/envs/make_env.py` |
| `hyper_mve/multiagent/` (整目录 7 文件 925 行) | `hyper_mve/_legacy_v4_7/multiagent/` |
| `hyper_mve/models/baseline_model.py` | `hyper_mve/_legacy_v4_7/models/baseline_model.py` |
| `hyper_mve/models/hyper_muzero_model.py` | `hyper_mve/_legacy_v4_7/models/hyper_muzero_model.py` |
| `hyper_mve/models/infer_muzero_model.py` | `hyper_mve/_legacy_v4_7/models/infer_muzero_model.py` |
| `hyper_mve/models/context_encoder.py` | `hyper_mve/_legacy_v4_7/models/context_encoder.py` |
| `hyper_mve/models/gru_context_encoder.py` | `hyper_mve/_legacy_v4_7/models/gru_context_encoder.py` |
| `hyper_mve/models_advanced/` (整目录) | `hyper_mve/_legacy_v4_7/models_advanced/` |
| `hyper_mve/scripts/train_baseline.py` | `hyper_mve/_legacy_v4_7/scripts/train_baseline.py` |
| `hyper_mve/scripts/train_oracle.py` | `hyper_mve/_legacy_v4_7/scripts/train_oracle.py` |
| `hyper_mve/scripts/train_oracle_v2.py` | `hyper_mve/_legacy_v4_7/scripts/train_oracle_v2.py` |
| `hyper_mve/scripts/train_infer.py` | `hyper_mve/_legacy_v4_7/scripts/train_infer.py` |
| `hyper_mve/scripts/train_infer_v2.py` | `hyper_mve/_legacy_v4_7/scripts/train_infer_v2.py` |
| `hyper_mve/scripts/test_env.py` | `hyper_mve/_legacy_v4_7/scripts/test_env.py` |
| `hyper_mve/scripts/test_discrete_env.py` | `hyper_mve/_legacy_v4_7/scripts/test_discrete_env.py` |
| `hyper_mve/training/buffer.py` (v3 遗留) | `hyper_mve/_legacy_v4_7/training/buffer.py` |
| `hyper_mve/training/mve.py` (stub) | `hyper_mve/_legacy_v4_7/training/mve.py` |
| `DESIGN_DOC_FINAL.html` | `hyper_mve/_legacy_v4_7/DESIGN_DOC_FINAL.html` |
| `DESIGN_DOC_FINAL.md` | **保留原位**（v4 路线图仍引用作为权威 v4.7 spec） |

**保留主路径** 的 v4.7 文件（这些 Pkg-04/05 会重写）：
- `hyper_mve/models/hyper_network.py` — Pkg-04 重写为 v2
- `hyper_mve/models/functional_nets.py` — Pkg-04 扩展接口
- `hyper_mve/models/representation_net.py` — Pkg-04 微调
- `hyper_mve/planning/mve_planner.py` — Pkg-05 接口适配
- `hyper_mve/training/muzero_trainer.py` — Pkg-05 重写
- `hyper_mve/training/episode_buffer.py` — Pkg-05 扩展
- `hyper_mve/training/worker.py` — Pkg-05 适配
- `hyper_mve/utils/utils.py` — 保留
- `hyper_mve/utils/evaluator.py` — Pkg-07 重构
- `hyper_mve/models/interfaces.py` — 评估后保留或删除

### 2.5 新增归档说明文件

| 路径 | 内容 |
|------|------|
| `hyper_mve/_legacy_v4_7/__init__.py` | 触发 DeprecationWarning |
| `hyper_mve/_legacy_v4_7/README.md` | 归档原因 + git tag 恢复命令 + v4.7 → v4 主要差异表 |

### 2.6 Git 操作

```bash
# 锁定 v4.7 工作版本
git add -A && git commit -m "Pkg-01 prep: stage v4.7 working tree before refactor"
git tag v4.7-final
git push origin v4.7-final

# Pkg-01 主分支
git checkout -b pkg-01/foundation-schema
# 后续: 归档 + 新 schema + 拆分 config + 单测，单 PR 合入
```

---

## 3. Capabilities（完成后系统获得的新能力）

### 3.1 用户视角

```python
# 1. 一行获取标准三难度配置
from hyper_mve.configs import V4Config
cfg = V4Config.from_preset("medium")   # 等同 Ch3.9 Medium 行
print(cfg.env.N, cfg.env.K, cfg.env.T_max)   # 4 20 200

# 2. 类型安全的 schema
from hyper_mve.schemas import AgentType, CapabilityVector
cap = CapabilityVector.sample_default(rng=np.random.default_rng(42))
assert 0.5 <= cap.eta <= 1.5    # Ch3.6 字段范围自动校验
cap.eta = 99   # FrozenInstanceError！不可变

# 3. 显式观测维度计算
from hyper_mve.schemas import ObservationLayout
dim = ObservationLayout.total_dim(N=4, K=20)
print(dim)   # 自动算出六块总维度

# 4. 显式 buffer record
from hyper_mve.schemas import TimeStepRecord
record = TimeStepRecord(o=..., a=..., r=..., delta=..., pi_mve=..., v=..., 
                        tau=AgentType.ALPHA, cap=cap, c_hat=..., z_hat=...)
arrays = record.to_arrays()   # 给 buffer 存储

# 5. v4.7 完整可复现
# git checkout v4.7-final  → 整个 v4.7 工作版本

# 6. 跨包零冲突
from hyper_mve.schemas import AgentType
# Pkg-02 env、Pkg-04 model、Pkg-05 trainer 三处 import 同一 Enum
```

### 3.2 工程团队视角

- **零跨包类型偏移**：所有模块强类型 import → IDE 跳转、mypy 检查、单元测试全部受益
- **配置变更可追溯**：preset 切换是单文件 diff（如 `presets/medium.py`），而不是散布在主 config 类
- **v4 / v4.7 平行可运行**：Pkg-08 实验 driver 可同时调用 `_legacy_v4_7` 跑 v4.7 baseline 对照
- **新增 type / cap 字段成本低**：在 `schemas/capability.py` 加字段，所有下游模块通过 IDE 提示发现

### 3.3 论文视角

- **Ch3.2 形式化定义表** 14 个分量的命名定型——后续章节引用一致
- **Ch3.5.1 类型机制** τ_i 表示方式固定——后续 5 个 Ablation 不必重新解释
- **Ch3.6 capability 4 维**符号 $\phi^{fov}$（避免与 φ(c) 冲突）锁定
- **Ch3.7 六块观测** Table 3.7.1 的维度列直接由 `ObservationLayout.total_dim()` 自动验证

---

## 4. Impact（影响分析）

### 4.1 对 v4.7 现有功能的影响

**保护**：
- v4.7 代码完整迁移至 `_legacy_v4_7/`，子目录结构保留
- `git tag v4.7-final` 锁定 commit hash，任何时刻 `git checkout v4.7-final` 可恢复完整工作版本
- `hyper_mve/config.py` 保留 backward-compat shim：`BaseConfig = V4Config.from_preset("medium")`（带 DeprecationWarning）

**破坏**：
- 现有 `from hyper_mve.envs.non_stationary_tag import ...` 类的 import 全部需要改为 `from hyper_mve._legacy_v4_7.envs.non_stationary_tag import ...`
- v4.7 训练脚本 `python hyper_mve/scripts/train_oracle.py` 路径变为 `python hyper_mve/_legacy_v4_7/scripts/train_oracle.py`（README 提供命令对照表）

**v4.7 测试**：
- Pkg-01 完成时**至少**通过：`python hyper_mve/_legacy_v4_7/scripts/test_env.py` 跑通（验证归档无破坏）
- 可选：`python hyper_mve/_legacy_v4_7/scripts/train_baseline.py` 跑 100 步无崩溃

### 4.2 对后续包的影响

| 后续包 | 本包提供 | 影响 |
|--------|----------|------|
| Pkg-02 (Env) | `EnvConfig`, `AgentType`, `CapabilityVector`, `ObservationLayout` | env step 返回的 obs 张量布局完全按 `ObservationLayout`；info dict 包含 `AgentType` 与 `CapabilityVector` 实例 |
| Pkg-03 (BeliefNet) | `ContextSchema` (d_c/d_role/d_belief) | TriContextEncoder 输入维度从常量取，无 magic number |
| Pkg-04 (Model) | `ModelConfig` + `ContextSchema` | DualHyperNetwork v2 三路输入维度从 schema 取 |
| Pkg-05 (Trainer) | `TrainConfig` + `TimeStepRecord` | episode_buffer 存储格式遵循 `TimeStepRecord`；curriculum stage 边界从 `TrainConfig.curriculum_stage_2_start=0.3` 等取 |
| Pkg-06a/b (Baselines) | 全部 schema | 7 baseline 用同一 schema，无 type / cap 表示偏移 |
| Pkg-07 (μP) | `MupConfig` 占位 | μP base_shape 填进 MupConfig |
| Pkg-08 (Experiments) | `V4Config.from_preset()` | 实验 driver 一行切配置 |

### 4.3 对论文贡献的影响

**直接贡献**：
- 无（本包不直接填实任何 Chapter 数据）

**间接贡献（基础设施）**：
- Ch3 全部章节的**变量名固定**：v4 论文重写过程中的术语漂移由 schema 约束阻断
- Ch5.6.1 buffer 字段表 与 `TimeStepRecord` 一对一对照
- Ch6 全部对比实验的 baseline 共享 schema → Q4 强化 A 的"等参对齐范围"清晰定义
- 补充材料 v4.7 vs v4 对照可直接引 `_legacy_v4_7/` 的代码片段

### 4.4 风险与对策

| 风险 | 触发条件 | 对策 |
|------|----------|------|
| 归档时遗漏文件 | git mv 操作大批量，可能漏 1-2 个 v4.7 文件 | Pkg-01 PR 必须包含 `python scripts/audit_legacy.py` 验证脚本 |
| schema 字段顺序后续被迫修改 | 如 Ch4.2.2 cap_emb MLP 期望特定顺序而 schema 已固定不同顺序 | dataclass 字段顺序按 Ch3.6 文档原序；后续如需变更走标准 spec 修订流程 |
| `BaseConfig` shim 引发循环 import | `config.py` import `configs/v4_config.py` 若有反向依赖会循环 | shim 仅做 lazy import；`configs/` 子模块禁止反向 import `config.py` |
| v4.7 训练脚本走 `_legacy_v4_7` 路径时 PYTHONPATH 问题 | `sys.path.insert(0, ...)` 可能找不到归档后的相对路径 | `_legacy_v4_7/__init__.py` 中预置 sys.path 修正逻辑（接受 import 慢） |

---

## 5. References

| 来源 | 引用条款 |
|------|----------|
| `D:\RL\docs\Hyper_MuZero_v4_Roadmap.md` | §2.4 代码基线锁定、§4 Stage 1 启动 |
| `D:\RL\docs\Chapter3_Environment_v4.md` | §3.5.1 类型机制、§3.6 capability 4 维、§3.7 六块观测、§3.9 三难度配置 |
| `D:\RL\docs\Chapter4_1_Motivation_v4.md` | §4.2.4 三路输入维度表 |
| `D:\RL\docs\Chapter5_Planner_Training_v4.md` | §5.6.1 buffer 新字段、§5.7 课程阶段边界 0.3/0.7 |
| `D:\RL\hyper_mve\DESIGN_DOC_FINAL.md` | v4.7 现状权威说明（归档后保留原位作为对照） |
| 本项目 Plan File | `C:\Users\zhengwenbo01\.claude\plans\d-rl-docs-batch-d-rl-hyper-mve-sdd-reflective-dove.md` Part 2 Pkg-01 |

---

## 6. Acceptance Criteria Summary

> 详见 [`design.md`](./design.md) 与各 `specs/*.md`。本节仅速览。

- [ ] `tests/schemas/` 全部通过（≥ 5 test files，≥ 20 test cases）
- [ ] `tests/configs/` 全部通过（包含三 preset 与 Ch3.9 Table 数值对照）
- [ ] `git tag v4.7-final` 在远程仓库可见
- [ ] `hyper_mve/_legacy_v4_7/` 归档完整：`python scripts/audit_legacy.py` 输出 "ALL FILES ACCOUNTED FOR"
- [ ] `python hyper_mve/_legacy_v4_7/scripts/test_env.py` 在归档后仍能跑通
- [ ] `from hyper_mve.configs import V4Config; V4Config.from_preset("medium")` 在 Python REPL 一行可执行
- [ ] PR description 包含 v4.7 → v4 import 路径变更表（README 风格）

---

## 7. Out of Scope（明确不做）

- **不实现**任何环境逻辑（Pkg-02 负责）
- **不实现** μP base_shape 具体数值（Pkg-07 负责）
- **不重写**任何训练循环（Pkg-05 负责）
- **不实现** schema 的运行时验证（如 pydantic 风格的字段约束）——仅静态类型注解 + dataclass `__post_init__` 范围检查
- **不引入新依赖**（仅用 Python 标准库 dataclass / enum / typing）
- **不修改** `D:\RL\docs\` 下任何论文章节文件
- **不接入** OpenSpec 框架（`openspec/config.yaml` 当前为空模板，本包不为其填充内容；后续若决定全工作区接入再讨论）
