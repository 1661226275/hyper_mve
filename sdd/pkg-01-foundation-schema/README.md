# Pkg-01: Foundation Schema & Config Restructure

> **状态**：Draft — awaiting user review
> **包 ID**：`pkg-01-foundation-schema`
> **工期**：0.5 周（3-4 天） · **GPU 算力**：0 · **PR 体量**：8 新文件 + 11 拆分 config + ~30 归档移动

---

## 📚 文档导览

| 文档 | 目的 | 字数 |
|------|------|------|
| [`proposal.md`](./proposal.md) | **先读这个**：Why / What Changes / Capabilities / Impact | ~3500 |
| [`design.md`](./design.md) | 设计抉择：Context / Goals / 8 项 Decisions | ~5000 |
| [`specs/01-agent-type-schema.md`](./specs/01-agent-type-schema.md) | AgentType Enum (α/β) + 工具函数 | ~1500 |
| [`specs/02-capability-vector.md`](./specs/02-capability-vector.md) | CapabilityVector 4 维 frozen dataclass | ~2000 |
| [`specs/03-observation-layout.md`](./specs/03-observation-layout.md) | 六块观测张量布局 + Self-Info | ~2200 |
| [`specs/04-timestep-record.md`](./specs/04-timestep-record.md) | Buffer 单步记录 (10 字段) | ~2000 |
| [`specs/05-v4-config-structure.md`](./specs/05-v4-config-structure.md) | V4Config 5 层分解 | ~3000 |
| [`specs/06-difficulty-presets.md`](./specs/06-difficulty-presets.md) | Easy / Medium / Hard preset | ~2500 |
| [`specs/07-legacy-archive-protocol.md`](./specs/07-legacy-archive-protocol.md) | v4.7 归档 + git tag 协议 | ~3000 |

---

## 🎯 一句话目标

**为 v4 重构奠定公共契约层**：定义 5 个 schema 类型 + 拆 70 参数 BaseConfig 为 5 层 + 归档 v4.7 工作版本（可复现）。

---

## ✅ 关键 Acceptance Criteria（速览）

### 必通过

- [ ] `git tag v4.7-final` 推送至远程
- [ ] `python scripts/audit_legacy.py` 输出 "ALL FILES ACCOUNTED FOR"
- [ ] `python hyper_mve/_legacy_v4_7/scripts/test_env.py` 在归档后跑通
- [ ] `from hyper_mve.configs import V4Config; V4Config.from_preset('medium')` REPL 可执行
- [ ] `pytest tests/schemas/ tests/configs/` 全部通过（≥ 30 test cases）
- [ ] Medium / Easy / Hard 三 preset **字段与 Ch3.9 Table 100% 匹配**（强制单测）

### 期望达到

- [ ] `mypy --strict hyper_mve/schemas/ hyper_mve/configs/` 零错误
- [ ] 单测覆盖率 ≥ 85%
- [ ] PR description 包含 v4.7 → v4 import 路径变更表

---

## 📅 实施顺序（4 天）

| 天数 | 任务 | 验证 |
|------|------|------|
| **Day 1 上午** | `git tag v4.7-final` + `perform_archive.ps1` 归档 | `audit_legacy.py` PASS |
| **Day 1 下午** | 写 `_legacy_v4_7/README.md` + `__init__.py` | v4.7 test_env.py 跑通 |
| **Day 2 上午** | `schemas/_constants.py` + `agent_type.py` + `capability.py` | 对应单测 PASS |
| **Day 2 下午** | `schemas/observation.py` + `buffer_record.py` + `context.py` | 对应单测 PASS |
| **Day 3 上午** | `configs/{env,model,train}_config.py` 三大主 config | 对应单测 PASS |
| **Day 3 下午** | `configs/{mup,eval,legacy,v4}_config.py` + presets | preset 数值对照 PASS |
| **Day 4 上午** | `config.py` backward-compat shim + 端到端 smoke | REPL `V4Config.from_preset('medium')` PASS |
| **Day 4 下午** | PR 描述 + 文档补完 + 自查 | merge ready |

---

## 🔑 8 项关键 Decisions（速览）

| # | Decision | 推荐 | 详见 |
|---|----------|------|------|
| D1 | AgentType 编码 | **Enum + nn.Embedding** | [design §3 D1](./design.md) |
| D2 | cap 字段类型 | **全 float32** | [design §3 D2](./design.md) |
| D3 | 配置实现技术 | **frozen dataclass** | [design §3 D3](./design.md) |
| D4 | 归档目录名 | **`_legacy_v4_7/`** | [design §3 D4](./design.md) |
| D5 | ẑ 字段类型 | **`np.ndarray (N-1, 2)`** | [design §3 D5](./design.md) |
| D6 | 遗留参数处理 | **`LegacyConfig` 命名空间** | [design §3 D6](./design.md) |
| D7 | schema 范围验证 | **`__post_init__` + raise** | [design §3 D7](./design.md) |
| D8 | preset 覆盖语义 | **`replace` 显式** | [design §3 D8](./design.md) |

---

## ❓ Open Questions（v4 Pass 2 修订后）

| # | Question | 状态 |
|---|----------|------|
| ~~Q1~~ | ~~`d_role=32` 由维度推导？~~ | **✅ 已关闭**：v4 Ch4.2.2 硬约束 d_id=8 + d_type=**8** + d_cap=16 = 32 精确填满（无 pad）。常量写入 `schemas/_constants.py`。 |
| Q2 | `TimeStepRecord` 包含 `t: int` 字段？ | **包含** |
| Q3 | `_legacy_v4_7/__init__.py` DeprecationWarning stacklevel？ | **2** |
| Q4 | `hyper_mve/utils/utils.py` 是否纳入本包重构？ | **否** |
| ~~Q5~~ | ~~`LegacyConfig.detach_pred_context=True` 默认值合理？~~ | **✅ 已关闭**：字段从 LegacyConfig 删除（Ch4.6 未提及，belief gradient gating 取代）。 |

**v4 关键数值确认**（Ch4 v2 → v4 升级修订）：
- `d_belief = 32`（**不是 48**）= 2 × d_belief_proj(16) (Ch4.2.4)
- `d_type_emb = 8`（**不是 4**）与 d_id=8 + d_cap=16 精确填满 d_role=32 (Ch4.2.2)
- `c_hat` 是 (N,) **scalar**（不是 (N, d_c) 向量）：Ch4.2.3 Head 1 sigmoid 输出
- `ẑ` **Oracle 监督**（不是自监督）：env.info 需暴露 `types` 真值（Pkg-02 v4 硬约束）

> 见 [`design.md` §8](./design.md)。Q2/Q3/Q4 维持默认，Q1/Q5 已关闭。本包视为 finalized。

---

## 📦 输出清单（PR 时检查）

### 新增（11 文件）
- `hyper_mve/schemas/{__init__,_constants,agent_type,capability,observation,buffer_record,context}.py`（7 个）
- `hyper_mve/_legacy_v4_7/{__init__.py,README.md}`（2 个）
- `scripts/audit_legacy.py`（1 个）
- `scripts/perform_archive.ps1`（1 个）

### 重写（1 文件）
- `hyper_mve/config.py` → backward-compat shim

### 拆分（10 文件，由原 config.py 拆出）
- `hyper_mve/configs/{__init__,env_config,model_config,train_config,mup_config,eval_config,legacy_config,v4_config}.py`（8 个）
- `hyper_mve/configs/presets/{__init__,easy,medium,hard}.py`（4 个）

### 测试（≥9 文件）
- `tests/schemas/{__init__,test_agent_type,test_capability_vector,test_observation_layout,test_buffer_record,test_context_schema}.py`（6 个）
- `tests/configs/{__init__,test_env_config,test_v4_config,test_presets,test_legacy_config}.py`（5 个）
- `tests/test_legacy_import_warning.py`（1 个）

### Git
- `git tag v4.7-final` 已推送
- 分支 `pkg-01/foundation-schema` → PR to `main`

### 归档移动（~30 文件）
- 详见 [`proposal.md` §2.4](./proposal.md)

---

## 🔗 上下游关联

| 关系 | 包 | 接口 |
|------|-----|------|
| **本包 → Pkg-02** | Env | `EnvConfig`, `AgentType`, `CapabilityVector`, `ObservationLayout` |
| **本包 → Pkg-03** | BeliefNet | `ContextSchema`（维度常量） |
| **本包 → Pkg-04** | Model | `ModelConfig`（hyper 维度 + AdaLN/output_scale） |
| **本包 → Pkg-05** | Trainer | `TrainConfig`（curriculum 边界、EMA、loss 权重）+ `TimeStepRecord` |
| **本包 → Pkg-06a/b** | Baselines | 全部 schema |
| **本包 → Pkg-07** | μP/Eval | `MupConfig` 占位 + `EvalConfig` |
| **本包 → Pkg-08** | Experiments | `V4Config.from_preset()` |

---

## 📖 引用源

- 项目 Plan File: [`d-rl-docs-batch-d-rl-hyper-mve-sdd-reflective-dove.md`](file:///C:/Users/zhengwenbo01/.claude/plans/d-rl-docs-batch-d-rl-hyper-mve-sdd-reflective-dove.md) Part 2 Pkg-01
- Hyper-MuZero v4 Roadmap: `D:\RL\docs\Hyper_MuZero_v4_Roadmap.md`
- Chapter 3 Environment v4: `D:\RL\docs\Chapter3_Environment_v4.md` (§3.5, 3.6, 3.7, 3.9)
- Chapter 4.1 Motivation v4: `D:\RL\docs\Chapter4_1_Motivation_v4.md` (§4.2.4)
- Chapter 5 Planner & Training v4: `D:\RL\docs\Chapter5_Planner_Training_v4.md` (§5.6.1, 5.7)
- v4.7 现状权威说明: `D:\RL\hyper_mve\DESIGN_DOC_FINAL.md`

---

## 🚦 Review Checklist（用户审阅时勾选）

- [ ] proposal.md 读完，**Why** 充分（v4.7 三痛点说明清晰）
- [ ] design.md 读完，**8 项 Decisions** 中无反对意见
- [ ] specs/01-07 抽查至少 3 个，接口设计可接受
- [ ] 4 天实施顺序合理
- [ ] Open Questions Q2/Q3/Q4 默认决定 OK（Q1/Q5 已关闭）
- [ ] 归档目录命名 `_legacy_v4_7/` OK
- [ ] 单测覆盖范围足够
- [ ] 出包后可启动 Pkg-02
