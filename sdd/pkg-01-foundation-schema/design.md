# Pkg-01: Foundation Schema & Config Restructure — Design

> 配套阅读：[`proposal.md`](./proposal.md)（**先读 proposal 再读本文**）

---

## 1. Context

### 1.1 项目阶段

本包是 v4 重构的 **入口包**。整个 v4 论文实验平台改造分 9 个 SDD 包（详见 [项目 Plan File](file:///C:/Users/zhengwenbo01/.claude/plans/d-rl-docs-batch-d-rl-hyper-mve-sdd-reflective-dove.md) Part 1）：

```
Pkg-01 (本包) → Pkg-02 (Env) → Pkg-03 (BeliefNet) ↘
                                                    Pkg-04 (Model) → Pkg-05 (Trainer) → Pkg-06a/b (Baselines)
                                                                                            ↓
                                                                                          Pkg-07 (μP/Eval) → Pkg-08 (Experiments)
```

本包必须**最早完成**——其输出（schema + config 结构 + 归档目录）是后续 8 个包的公共契约。

### 1.2 前置依赖（本包消费什么）

| 来源 | 内容 |
|------|------|
| v4.7 现状审计 | 70 个 config 参数清单、7665 LoC 文件树、稳定性护栏位置（详见 plan file Part 0） |
| Ch3.5.1, 3.6, 3.7 | 类型机制 / capability 4 维 / 六块观测的规范文本 |
| Ch4.2.4 (v4) | 三路输入维度（d_c=16, d_role=32, d_belief=32），总 ctx_i=80。**v4 修订**：d_belief 由 v3 推测 48 修正为 32（ĉ 投影 16 + pooled ẑ 投影 16）。role 内部 d_id=8 + d_type=8 + d_cap=16 精确填满，无 pad。 |
| Ch5.6.1 | buffer 字段清单（10 项） |
| Ch3.9 | 三难度 preset 参数表 |

### 1.3 本包提供（后续包消费什么）

| 输出 | 消费者 | 用途 |
|------|--------|------|
| `schemas/` 5 个 dataclass + 2 个 Enum | 02-08 全部 | 类型安全的数据流 |
| `configs/V4Config.from_preset()` | 02-08 全部 | 一行获取标准配置 |
| `configs/{easy,medium,hard}.py` | 06a/b/07/08 实验脚本 | 三难度运行 |
| `_legacy_v4_7/` 归档 | 论文 Ch6.12 + 补充材料 + 回滚预案 | v4.7 对照实验 / 回滚 |
| `git tag v4.7-final` | Roadmap §2.4 合规、回滚锚点 | 任何时刻可恢复 v4.7 |

---

## 2. Goals

### 2.1 主目标（必须完成）

1. **G1：定义 5 个数据 schema 类型**
   `AgentType` Enum、`CapabilityVector` dataclass、`ObservationLayout` 常量类、`TimeStepRecord` dataclass、`ContextSchema` 常量类。全部 frozen / 不可变 / 类型注解完整 / 单元测试覆盖。

2. **G2：拆 `BaseConfig` 为 5 层**
   分为 `EnvConfig` / `ModelConfig` / `TrainConfig` / `MupConfig` / `EvalConfig` 5 个独立 dataclass + 1 个组合器 `V4Config`，三 difficulty preset (Easy/Medium/Hard) 各自完整。

3. **G3：v4.7 完整归档 + 锁定**
   `git mv` 全部 v4.7 业务文件到 `hyper_mve/_legacy_v4_7/`，保持子目录结构；`git tag v4.7-final` 锁定；`_legacy_v4_7/README.md` 提供恢复指南与 v4.7 → v4 路径变更表。

4. **G4：三 preset 数值对照 Ch3.9 Table**
   `V4Config.from_preset("easy")` / `("medium")` / `("hard")` 三组数值与 Ch3.9 Reference Configuration Table **100% 字段匹配**——这是后续 env 实现验证的基准。

5. **G5：零运行时回归**
   归档后 `python hyper_mve/_legacy_v4_7/scripts/test_env.py` 仍能跑通；v4.7 训练脚本 100 步无崩溃。

### 2.2 衍生目标（应尽量达到）

- **G6**：`mypy --strict hyper_mve/schemas/ hyper_mve/configs/` 零错误（外部依赖 stub 缺失可忽略）
- **G7**：单元测试覆盖率（`schemas/` + `configs/`）≥ 85%
- **G8**：所有 schema dataclass 支持 JSON 往返（`to_dict` / `from_dict`）

### 2.3 Non-Goals（明确不解决）

- **NG1**：**不实现** ResourceCommons 环境逻辑——所有动力学、reward、observation 构造在 Pkg-02
- **NG2**：**不定义** μP base_shape 具体数值——仅 `MupConfig` 占位字段；具体数值在 Pkg-07
- **NG3**：**不重写**任何 model / trainer 模块——`hyper_network.py`, `functional_nets.py`, `muzero_trainer.py` 等保留原位待 Pkg-04/05
- **NG4**：**不实现** pydantic 风格运行时验证——仅 dataclass `__post_init__` 中的范围检查（最小开销）
- **NG5**：**不引入新第三方依赖**——只用 Python 3.10+ 标准库（dataclass / enum / typing / json）+ 现有 numpy
- **NG6**：**不修改**论文文档（`D:\RL\docs\*.md`）——schema 命名遵循文档既定符号
- **NG7**：**不实现** schema 的 dynamic registration / plugin 机制——后续如需扩展类型（如 type γ）走标准修订流程

---

## 3. Decisions

> 8 项关键设计抉择。每项格式：**Decision** → 候选 → 推荐 → 理由 → 风险与回滚。

### D1: AgentType 编码方式

**Decision**：α/β 类型如何在代码中表示？

**候选**：
- **A1**：one-hot 2 维 numpy 数组 `[1.0, 0.0]` / `[0.0, 1.0]`
- **A2**：标量 int 0/1
- **A3**：Python `Enum` + 学习时通过 `nn.Embedding(2, d_type=4)` 嵌入

**推荐**：**A3**

**理由**：
- **类型安全**：Enum 在 IDE / mypy 下能阻止 "type=2" 之类错误参数；A1/A2 完全字符化
- **未来扩展**：Ch3.5.1 已留出 "未来扩展 γ/δ" 的空间，Enum 可一行加成员
- **运行时性能**：Enum 比较是 O(1) 指针比较；one-hot 比较是 4 字节扫描——差异可忽略
- **存储格式**：buffer 中 `TimeStepRecord.tau: AgentType` 字段在 `to_arrays()` 时统一转 int8 numpy（节省 87.5% 空间 vs float one-hot）
- **嵌入查表**：Ch4.2.2 明确 type_emb 由 `nn.Embedding(2, d_type)` 生成；输入是 long tensor，Enum.value 完美对接

**风险**：
- 序列化兼容性：JSON dump Enum 需要自定义 encoder
- TensorBoard logging 时 Enum.name 需要 str 转换

**回滚**：切回 A1（one-hot）只需改 4 处：`schemas/agent_type.py` + buffer 序列化 + model 输入处 + 单测。每处 < 10 行。

---

### D2: CapabilityVector 是 float 还是 int 化

**Decision**：cap 4 维（η, $\phi^{fov}$, ν, ζ）的字段类型？特别是 $\phi^{fov}$（视野半径）是离散整数（Chebyshev 距离）。

**候选**：
- **B1**：所有字段 float32，schema 内部存原值，env 内部 round($\phi^{fov}$)
- **B2**：$\phi^{fov}$ int，其他 float
- **B3**：所有字段 float64

**推荐**：**B1**

**理由**：
- **训练信号一致性**：Pkg-03 cap_emb MLP 输入为 float 向量；$\phi^{fov}$ 作为 float（如 2.7）携带比 int 2 更多信息——MLP 可学习"略大于 2.5 的视野倾向于行为 X"
- **env 简单**：env 内部使用 cap 时仅在 FOV 过滤一处 round 整数（其他地方 float 自然计算）
- **存储一致**：4 维统一 float32 numpy array，无类型分支
- **范围验证**：`__post_init__` 中 `assert 2.0 <= phi_fov <= 4.0`，round 后必然 ∈ {2, 3, 4}

**风险**：
- 不同 seed 采样的 $\phi^{fov}$ 边界处理（如 2.499 vs 2.501 round 不同）→ episode 间观测维度差异 → 影响泛化测试可复现性
- 缓解：env 在 episode 开始时一次性 round + cache，记录到 episode metadata

**回滚**：改 B2（$\phi^{fov}$ int）只需 `CapabilityVector.phi_fov: int` + 采样函数 round；其他模块通过类型注解发现并适配（约 5-7 处）。

---

### D3: V4Config 用 dataclass / pydantic / dict

**Decision**：配置类的实现技术选型？

**候选**：
- **C1**：`@dataclass(frozen=True)` + `field(default_factory=...)`
- **C2**：pydantic v2 `BaseModel`
- **C3**：嵌套 dict + helper 函数
- **C4**：`attrs` 库

**推荐**：**C1**

**理由**：
- **零依赖**：Python 3.10+ 内置 dataclass；pydantic v2 引入 ~3MB + C 扩展；项目 NG5 明确不加依赖
- **IDE 体验**：dataclass 在 VSCode/PyCharm 下类型提示完美；pydantic 需要插件
- **`__post_init__` 验证**：足以覆盖范围检查（cap.eta ∈ [0.5, 1.5] 等），无需完整 schema 框架
- **frozen 不可变**：避免训练中误改配置导致的隐藏 bug（如 `cfg.train.lr *= 0.1` 在某个回调里）
- **序列化**：`dataclasses.asdict()` + `json.dumps()` 两行完成；反序列化用 `cattrs` 可选（不引入依赖，自实现 30 行）

**风险**：
- 嵌套 dataclass 的 `field(default_factory=lambda: SubConfig())` 写法初学者不熟悉
- frozen 后无法用 `cfg.env.N = 8` 临时改值——需要 `dataclasses.replace(cfg.env, N=8)`，初期可能繁琐

**回滚**：切 C2 pydantic 需重写所有 dataclass，~2 天工作量。但 frozen + 类型注解的 dataclass 与 pydantic 接口高度相似，自动化迁移可行。

---

### D4: 归档目录命名

**Decision**：v4.7 归档放在哪个目录？

**候选**：
- **D1**：`hyper_mve/_legacy/`
- **D2**：`hyper_mve/_legacy_v4_7/`（含版本号）
- **D3**：`hyper_mve/archived/`
- **D4**：`hyper_mve/v4_7_snapshot/`

**推荐**：**D2**

**理由**：
- **下划线前缀**：IDE / linter / pytest collector 默认忽略 `_xxx` 目录，避免 v4.7 单测污染 CI
- **版本号显式**：未来若进入 v5 时归档 v4 → `_legacy_v4/` 与本目录共存不冲突
- **`legacy` 词义**：比 `archived` 更准确（archived 暗示完全停用；legacy 暗示历史版本但仍可访问）

**风险**：
- 下划线前缀 import 路径 `from hyper_mve._legacy_v4_7.envs.non_stationary_tag import ...` 较长——但归档目录本来就不期望频繁 import

**回滚**：`git mv hyper_mve/_legacy_v4_7 hyper_mve/<new_name>`，更新 README + 1 个 audit 脚本。

---

### D5: TimeStepRecord 的 ẑ 字段类型

**Decision**：BeliefNet 输出的对手类型推断 ẑ 在 buffer 中如何存储？

**候选**：
- **E1**：`Dict[int, np.ndarray]`：key=对手 agent_id，value=2 维 softmax
- **E2**：`np.ndarray shape=(N-1, 2)` 固定形状，按 agent_id 排序
- **E3**：`torch.Tensor`
- **E4**：`np.ndarray shape=(N, 2)`：包含自己（自己位置填 0 或 one-hot 真值）

**推荐**：**E2**

**理由**：
- **定长数组优势**：buffer 批量 sampling 时 stack 操作 O(B) vs dict 需要 O(B·N) 索引
- **2-class 显式**：Ch4.5 明确 ẑ_{i,j} ∈ [0,1]^2 是 α/β 概率分布
- **CPU/GPU 解耦**：numpy 在 buffer 中存储（节省 GPU 内存）；trainer sampling 时 `torch.from_numpy()`
- **排除自己**：N-1 维度排除 self（与 Ch4.5 公式一致）；自己类型已在 obs[type_block] one-hot 显式
- **可变 N 适配**：若未来 Ch3.9 Hard 改 N=8，仅 shape 改变，无字段重命名

**风险**：
- 不同 episode 的 N 必须固定（与 env preset 一致）——已是当前设计
- agent_id 顺序约定：按 0, 1, 2, ..., i-1, i+1, ..., N-1（跳过 i）

**回滚**：切 E1 dict 仅影响 buffer 存储与 BeliefNet 输出适配（~3 处）。

---

### D6: 是否保留 v4.7 的 `freeze_enabled` 等遗留参数

**Decision**：v4.7 引入但 v4 不再用的参数如何处理？

**v4 修订（2026-05-27 Pass 2）**：基于 Ch4_v4 §4.6 四道防线**未提** detach_pred_context，且 type_emb 进 role 通路（Ch4.2.2 + 4.3.3）从架构层根治 RewardHead 撕裂——`w_rew_diversity` 多样性正则失去存在理由；BeliefNet 引入了独立 belief 通路 gradient gating（Ch4.6.5），与 detach_pred_context 机制不同。**这两类字段从 LegacyConfig 删除**。

**保留集合**（与 v4 正交，仍可作为回退实验开关）：
- `freeze_*`（Adversarial Freezing，v4.2 引入）
- `chunk_*`（ChunkedHMLP，v4.5 Phase 5 已弃用但保留兜底）

**删除集合**（v4 架构上已不需要）：
- ~~`w_rew_diversity`, `rew_diversity_target_cos`, `rew_diversity_skip_pairs`~~（type_emb 已根治撕裂）
- ~~`detach_pred_context`~~（belief gradient gating 取代）

**候选**：
- **F1**：从主 config 删除
- **F2**：保留在 `TrainConfig` 但标 `@deprecated`
- **F3**：移到独立 `LegacyConfig` 类（默认禁用，需显式启用）

**推荐**：**F3**（仅保留 freeze_* + chunk_*）

**理由**：
- **干净主 config**：`TrainConfig` 字段保持精简（v4 真正使用的 ~25 个参数）
- **可调试性保留**：若 Pkg-05 课程学习出现意外，可能需要回退到 v4.7 freeze 机制对比——`LegacyConfig(freeze_enabled=True)` 一行恢复
- **架构干净**：删除 v4 不需要的 4 个字段，避免下游包误用

**实现**：
```python
@dataclass(frozen=True)
class LegacyConfig:
    """v4.7 遗留参数，默认禁用。仅用于回退实验。
    
    v4 修订: 删除 w_rew_diversity / detach_pred_context 等
    (Ch4_v4 架构上已不需要，详见 Ch4.3.3 + 4.6.5)。
    """
    # Adversarial Freezing (v4.2) - 与 v4 正交，保留
    freeze_enabled: bool = False
    freeze_warmup_steps: int = 4000
    freeze_phase_steps: int = 4000
    freeze_hunter_agents: tuple[int, ...] = (0, 1, 2)
    freeze_prey_agents: tuple[int, ...] = (3,)
    
    # ChunkedHMLP (v4.5 Phase 5, deprecated) - 保留兜底
    chunk_alpha: int = 10
    chunk_emb_size: int = 8
    chunk_budget_factor: float = 1.0
    chunk_hyperfan_init: bool = True
    
    # v4 删除字段 (架构已根治):
    # - w_rew_diversity / rew_diversity_*  → type_emb 进 role 已解决撕裂
    # - detach_pred_context                → belief gradient gating 取代
```

`V4Config.legacy: LegacyConfig = field(default_factory=LegacyConfig)`，**默认全禁用**（freeze_enabled=False, chunk_* 不使用）。

**风险**：
- v4.7 训练脚本（在 `_legacy_v4_7/` 中）期望直接读 `cfg.freeze_enabled`，归档后字段移到 `cfg.legacy.freeze_enabled` 会触发 AttributeError
- **缓解**：v4.7 脚本继续读自己的 v4.7 BaseConfig（不引用 v4 V4Config）；通过 `_legacy_v4_7/` import 自己的 config 副本

**回滚**：F1 完全删除——若实验过程中发现真的不再需要任何遗留参数。

---

### D7: schema 范围验证策略

**Decision**：dataclass 字段值范围（如 `cap.eta ∈ [0.5, 1.5]`）何时校验？

**候选**：
- **G1**：不校验（信任调用者）
- **G2**：`__post_init__` 中 assert
- **G3**：`__post_init__` 中 raise ValueError + 详细错误信息
- **G4**：pydantic 风格 field validators

**推荐**：**G3**

**理由**：
- **失败显式**：assert 在 `python -O` 优化模式下被剥离 → 隐性 bug；raise 始终执行
- **错误信息可调试**：`raise ValueError(f"CapabilityVector.eta={eta} out of range [0.5, 1.5]")` 比 `AssertionError` 更便于追溯
- **覆盖核心字段**：仅对 Ch3.5/3.6 明确范围的字段校验（不做完整 schema 验证）；性能开销可忽略（dataclass 构造 ~微秒级，env 一个 episode 调用 N 次即可）

**示例**：
```python
@dataclass(frozen=True)
class CapabilityVector:
    eta: float
    phi_fov: float
    nu: float
    zeta: float
    
    def __post_init__(self) -> None:
        if not (0.5 <= self.eta <= 1.5):
            raise ValueError(f"eta={self.eta} ∉ [0.5, 1.5] (Ch3.6)")
        if not (2.0 <= self.phi_fov <= 4.0):
            raise ValueError(f"phi_fov={self.phi_fov} ∉ [2.0, 4.0]")
        if not (0.8 <= self.nu <= 1.0):
            raise ValueError(f"nu={self.nu} ∉ [0.8, 1.0]")
        if not (10.0 <= self.zeta <= 30.0):
            raise ValueError(f"zeta={self.zeta} ∉ [10.0, 30.0]")
```

**风险**：
- 边界采样（eta=0.5 严格）vs (eta < 0.5 + ε)：用 `0.5 <= x <= 1.5` 闭区间
- 训练性能：dataclass 构造在 hot path 上（每 step 创建一次）会增加开销 → 测试若 > 5% 影响则改采样函数返回原始 tuple，仅 schema-level 接口接受 dataclass

**回滚**：G2（assert）一处替换；G1 不校验删除 5 个 `__post_init__` 方法。

---

### D8: difficulty preset 的字段覆盖语义

**Decision**：`V4Config.from_preset("medium")` 加载 preset 时如何合并/覆盖？是否允许用户在 preset 之上局部覆盖（如 `from_preset("medium", overrides={"env.N": 8})`）？

**候选**：
- **H1**：preset 是不可变蓝图，任何修改 `dataclasses.replace(cfg, ...)` 显式
- **H2**：`from_preset(name, **overrides)` 接受关键字覆盖
- **H3**：CLI 风格 `from_preset(name, overrides={"env.N": 8, "train.lr": 1e-3})` 接受 dotted path

**推荐**：**H1**

**理由**：
- **简化语义**：preset 加载一行得到完整不可变 V4Config 实例；用户改字段时通过 `replace` 显式
- **可追溯**：实验日志记录 preset 名（如 "medium"）+ replace 字段 diff，可重建任何配置
- **避免歧义**：dotted path 解析（H3）需要自定义 parser，引入 bug 风险
- **scripts 层**：实验 driver（Pkg-08）在 preset 加载后用 argparse 收集覆盖，逐字段 replace

**示例**：
```python
# 标准用法
cfg = V4Config.from_preset("medium")

# 实验需要 N=8（Ablation 3 类型扫描扩展）
from dataclasses import replace
cfg_n8 = replace(cfg, env=replace(cfg.env, N=8))

# 训练脚本接受 CLI
parser.add_argument("--lr", type=float, default=None)
args = parser.parse_args()
if args.lr is not None:
    cfg = replace(cfg, train=replace(cfg.train, lr=args.lr))
```

**风险**：
- 嵌套 replace 写法较繁琐——可加 helper：`cfg.with_overrides(env_N=8, train_lr=1e-3)`（约 30 行）
- 实验脚本超参覆盖时若遗漏 replace 会得到默认 preset 值——单元测试 `test_v4_config.py::test_replace_propagation` 覆盖

**回滚**：H2（kwargs 覆盖）实现 ~20 行，向后兼容。

---

## 4. 设计决策对照表

| Decision | 推荐 | 影响范围 | 后续修改成本 |
|----------|------|----------|--------------|
| D1 AgentType 编码 | Enum + nn.Embedding | 全部包 | 中（4 处） |
| D2 cap 字段类型 | 全 float32 | Pkg-02, 03, 04 | 低（1 处） |
| D3 配置实现技术 | frozen dataclass | 全部包 | 高（2 天迁移） |
| D4 归档目录名 | `_legacy_v4_7/` | 论文 / 回滚 | 低（1 mv） |
| D5 ẑ 字段类型 | `np.ndarray (N-1, 2)` | Pkg-03, 05 | 中（3 处） |
| D6 遗留参数处理 | `LegacyConfig` 命名空间 | Pkg-05 | 低（删字段） |
| D7 schema 范围验证 | `__post_init__` + raise | Pkg-02 | 低（删 5 处 if） |
| D8 preset 覆盖语义 | `replace` 显式 | Pkg-08 | 低（加 helper） |

---

## 5. 实现顺序建议

```
Day 1（半天）:
  - git tag v4.7-final + 归档 (git mv 全部 v4.7 文件)
  - 验证 _legacy_v4_7/scripts/test_env.py 仍可运行
  - 写 _legacy_v4_7/README.md
  
Day 1（半天）+ Day 2:
  - schemas/_constants.py + agent_type.py + capability.py
  - 对应单测
  - schemas/observation.py + buffer_record.py + context.py
  - 对应单测
  
Day 3:
  - configs/{env, model, train, mup, eval, v4}_config.py
  - configs/presets/{easy, medium, hard, legacy}.py
  - 对应单测（重点：与 Ch3.9 Table 数值对照）
  - config.py backward-compat shim
  
Day 4:
  - scripts/audit_legacy.py (验证归档完整性)
  - 端到端 smoke test: 
    * python hyper_mve/_legacy_v4_7/scripts/test_env.py
    * python -c "from hyper_mve.configs import V4Config; V4Config.from_preset('medium')"
  - PR 描述: import 路径变更表 + v4.7 → v4 迁移指南
```

---

## 6. 跨包接口约定（API contract）

本包对**所有下游包**的接口承诺：

### 6.1 import 路径（稳定）

```python
# 后续 8 个包应使用这些 import 路径，本包承诺不变更
from hyper_mve.schemas import (
    AgentType,
    CapabilityVector,
    ObservationLayout,
    TimeStepRecord,
    ContextSchema,
)
from hyper_mve.configs import V4Config

# 加载 preset
cfg = V4Config.from_preset("medium")

# 访问字段
cfg.env.N            # 4
cfg.env.K            # 20
cfg.model.hidden_dim # 128
cfg.train.lr         # 1e-4
```

### 6.2 schema 字段顺序（稳定）

| Schema | 字段顺序（不可变） |
|--------|-------------------|
| `CapabilityVector` | `eta, phi_fov, nu, zeta`（与 Ch3.6 文档一致） |
| `TimeStepRecord` | `o, a, r, delta, pi_mve, v, tau, cap, c_hat, z_hat`（与 Ch5.6.1 一致） |
| `ObservationLayout` blocks | `self, resource, neighbor, global, capability, type`（与 Ch3.7 一致） |

字段重命名 / 重排序 / 删除 = **breaking change**，需要全工作区 review + 标 major version bump。

### 6.3 数值常量来源（单一）

**Fehr-Schmidt & 资源动力学**：

| 常量 | 定义位置 | 引用 |
|------|----------|------|
| Fehr-Schmidt λ_disadv=2.0, λ_adv=0.6, κ=0.5 | `schemas/_constants.py` | Pkg-02 env, Ch3.5.3 |
| Q_max=10.0 | `schemas/_constants.py` | Pkg-02, Ch3.3 |
| α_min=0.02, α_max=0.20 | `schemas/_constants.py` | Pkg-02, Ch3.4 |
| Curriculum stage boundaries 0.3, 0.7 | `configs/train_config.py::TrainConfig` | Pkg-05, Ch5.7 |
| BeliefNet grad gating step = 5000 | `configs/train_config.py::TrainConfig` | Pkg-04, Ch4.6 |

**v4 架构维度（Ch4.2.2 + 4.2.3 + 4.2.4 硬约束，写入 `schemas/_constants.py`）**：

| 常量 | 值 | 来源 |
|------|----|------|
| `D_ID_EMB` | `8` | Ch4.2.2 默认 |
| `D_TYPE_EMB` | **`8`**（v4 关键，不是 4） | Ch4.2.2 默认 |
| `D_CAP_EMB` | `16` | Ch4.2.2 默认 |
| `D_ROLE` | `32` = 8+8+16 **精确填满，无 pad** | Ch4.2.2 派生 |
| `D_C_CTX` | `16` | Ch4.2.1 默认 |
| `D_BELIEF_PROJ` | `16` | Ch4.2.3 每个子分量 (ĉ / pooled ẑ) 投影 |
| `D_BELIEF` | `32` = 2 × 16 | Ch4.2.4 `d_b^{proj·2}` |
| `D_CTX_AUG` | `80` = 16+32+32 | Ch4.2.4 总 ctx_i |

后续包**禁止**重新定义这些常量；如需修改值，**修改本包 schema 后**走单元测试 → 影响传播分析 → 全局更新流程。v4 文档已通过"默认值正好填满"硬约束限定 D_ROLE=32 无 pad 余地；任何修改需先修 Ch4.2.2。

---

## 7. 验证策略概览

> 详细 acceptance criteria 见 `specs/*.md`。本节仅列 8 个关键测试。

1. `test_agent_type.py::test_enum_value_stable`: `AgentType.ALPHA.value == 0`, `BETA.value == 1`（绝不变）
2. `test_capability_vector.py::test_post_init_range`: 越界值（eta=2.0）触发 ValueError
3. `test_observation_layout.py::test_total_dim_formula`: `ObservationLayout.total_dim(N=4, K=20)` 输出与手算一致
4. `test_buffer_record.py::test_serialize_roundtrip`: `TimeStepRecord(...).to_arrays() → from_arrays()` 全字段恢复
5. `test_v4_config.py::test_medium_preset_matches_ch_3_9`: medium preset 字段与 Ch3.9 Table 100% 一致
6. `test_legacy_config.py::test_default_disabled`: `LegacyConfig().freeze_enabled is False`；`hasattr(LegacyConfig, "w_rew_diversity") is False`；`hasattr(LegacyConfig, "detach_pred_context") is False`（v4 已删除）
7. `test_legacy_import_warning.py::test_deprecation_warning`: `import hyper_mve._legacy_v4_7.envs.non_stationary_tag` 触发 DeprecationWarning
8. `scripts/audit_legacy.py`: 输出 "ALL FILES ACCOUNTED FOR (XX archived, YY in main path)"

---

## 8. Open Questions（待与用户讨论）

以下几点在 SDD review 时建议确认：

| # | Question | 状态 / 默认决定 | 影响 |
|---|----------|----------|------|
| ~~Q1~~ | ~~`ContextSchema` 中 d_role=32 是固定值还是从维度推导？~~ | **✅ 已关闭（v4 Pass 2）**：Ch4.2.2 硬约束 d_id=8 + d_type=8 + d_cap=16 = 32 精确填满，无 pad。常量写入 `schemas/_constants.py`。 | – |
| Q2 | `TimeStepRecord` 是否包含 `step_index`？ | **包含** `t: int` 字段，便于课程学习 stage 判定 | Pkg-05 trainer |
| Q3 | `_legacy_v4_7/__init__.py` 触发 DeprecationWarning 的级别？ | **`stacklevel=2` + `category=DeprecationWarning`**（标准） | 用户体验 |
| Q4 | 是否将 `hyper_mve/utils/utils.py` 中的工具函数（`actions_to_one_hot`, `scalar_transform`, `orthogonal_init` 等）也纳入本包重构？ | **否**——仅 schema/config 拆分，utils 保留原位待 Pkg-04 决定 | 包大小 |
| ~~Q5~~ | ~~`LegacyConfig.detach_pred_context=True` 默认值合理？~~ | **✅ 已关闭（v4 Pass 2）**：字段从 LegacyConfig 删除（Ch4.6 不需要）。无候选项。 | – |

> 若用户对默认决定无异议，本 design.md 视为 finalized。

---

## 9. References

- `proposal.md`（本包）
- `D:\RL\docs\Chapter3_Environment_v4.md` §3.5, §3.6, §3.7, §3.9
- `D:\RL\docs\Chapter4_1_Motivation_v4.md` §4.2.4
- `D:\RL\docs\Chapter5_Planner_Training_v4.md` §5.6.1, §5.7
- `D:\RL\hyper_mve\DESIGN_DOC_FINAL.md`（v4.7 spec，对照用）
- 项目 Plan File §Part 2 Pkg-01 Sections 2.A-2.E
