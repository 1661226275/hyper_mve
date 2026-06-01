# Pkg-03: TriContextEncoder & BeliefNet

> **状态**：Draft — awaiting user review
> **包 ID**：`pkg-03-tricontext-beliefnet`
> **工期**：1 周（5-7 天） · **GPU 算力**：10 小时（BeliefNet 合成数据收敛 + collapse 测试） · **PR 体量**：9 新文件 + 4 测试 + 1 合成数据脚本

---

## 📚 文档导览

| 文档 | 目的 | 字数 |
|------|------|------|
| [`proposal.md`](./proposal.md) | **先读这个**：Why / What Changes / Capabilities / Impact | ~3500 |
| [`design.md`](./design.md) | 设计抉择：Context / Goals / 8 项 Decisions | ~4500 |
| [`specs/01-tri-context-encoder.md`](./specs/01-tri-context-encoder.md) | TriContextEncoder 主接口（三路 concat → 80 维 ctx_i） | ~1500 |
| [`specs/02-c-encoder.md`](./specs/02-c-encoder.md) | c_ctx 客观通路（共同知识 c_t → ℝ^16） | ~900 |
| [`specs/03-role-encoder.md`](./specs/03-role-encoder.md) | role 角色通路（id+**type**+cap → ℝ^32，**v4 关键**） | ~1500 |
| [`specs/04-belief-net-gru.md`](./specs/04-belief-net-gru.md) | BeliefNet GRU 主干（uni-directional, h=128） | ~1500 |
| [`specs/05-belief-heads.md`](./specs/05-belief-heads.md) | head_c（sigmoid scalar）+ head_opp（**类型 2 分类**，v4 关键） | ~1800 |
| [`specs/06-belief-losses.md`](./specs/06-belief-losses.md) | L_c (MSE) + L_opp (**Oracle CE**) + L_div (variance hinge) | ~1800 |
| [`specs/07-permutation-invariant-pool.md`](./specs/07-permutation-invariant-pool.md) | Pool({ẑ_{i,j}}) 三策略（默认 mean） | ~1200 |
| [`specs/08-integration-contracts.md`](./specs/08-integration-contracts.md) | 与 Pkg-04/05 接口契约（Self-Info 严格性 + 课程接入） | ~1800 |

---

## 🎯 一句话目标

**实现 v4 架构两个核心新模块**——TriContextEncoder（拼接 c_ctx + role + belief 三路输入）和 BeliefNet（GRU + head_c 监督 ĉ + head_opp **Oracle 监督 ẑ 类型 2 分类**），为 Pkg-04 DualHyperNetwork v2 提供完整 80 维条件向量 ctx_i，并为 Pkg-05 课程学习 trainer 提供 BeliefNet 联合训练接口。

---

## ✅ 关键 Acceptance Criteria（速览）

### 必通过（结构硬约束）

- [ ] **TriContextEncoder 输出维度精确 = 80**（16 c_ctx + 32 role + 32 belief，对照 [Ch4.2.4](../../docs/Chapter4_Architecture_v4.md)）
- [ ] **role 内部 8+8+16=32 精确填满**（d_id + d_type + d_cap，对照 Pkg-01 ModelConfig）
- [ ] **head_opp 输出 (B, N-1, 2) softmax**（类型 α/β 概率，**不是** v3 动作预测）
- [ ] **z_hat 顺序契约**：agent i 的 head_opp 输出对应 agent_id 升序跳过 self（对照 Pkg-01 TimeStepRecord z_hat docstring）
- [ ] **梯度流测试通过**：反向传播覆盖三路 + 三 head 所有参数有梯度

### 收敛验证（合成数据）

- [ ] **head_c MSE < 0.05 在 5K step 内**（合成 c=0.7 轨迹）
- [ ] **head_opp accuracy > 80% 在 5K step 内**（合成 α/β 混合轨迹）
- [ ] **L_div hinge variance 阻止 collapse**：长训练（10K step）后 b_i^t 方差 ≥ 0.1

### 期望达到

- [ ] `pytest tests/models/test_tri_context.py tests/models/test_belief_net.py tests/models/test_belief_losses.py` 全通过
- [ ] `mypy --strict hyper_mve/models/{tri_context_encoder,c_encoder,role_encoder,belief_net,belief_losses,permutation_invariant_pool}.py` 零错误

---

## 📅 实施顺序（7 天）

| 阶段 | 天数 | 任务 | 输出 |
|------|------|------|------|
| **Phase 1** | Day 1 | `c_encoder.py` + `role_encoder.py`（无外部依赖）；写 specs/02 + specs/03 | 两 encoder forward 通过 |
| **Phase 2** | Day 2 | `permutation_invariant_pool.py` + `belief_net.py` GRU 主干；写 specs/04 + specs/07 | GRU step/seq 等价性测试通过 |
| **Phase 3** | Day 3 | `belief_net.py` 两 head（head_c + head_opp v4 关键改动）；写 specs/05 | head 输出形状契约通过 |
| **Phase 4** | Day 4 | `belief_losses.py`（L_c + L_opp + L_div）；写 specs/06 | 三 loss 单测通过 |
| **Phase 5** | Day 5 | `tri_context_encoder.py`（三路组合）；写 specs/01 | 80 维输出 + 梯度流通过 |
| **Phase 6** | Day 6 | 合成数据收敛实验：`scripts/test_belief_net_synth.py` | head_c MSE < 0.05；head_opp acc > 80% |
| **Phase 7** | Day 7 | 集成契约 + PR description；写 specs/08 | 与 Pkg-04/05 接口锁定 |

---

## 🔑 8 项关键 Decisions（速览）

| # | Decision | 推荐 | 详见 |
|---|----------|------|------|
| D1 | head_opp 输出格式 | 单输出 (B, N-1, 2) reshape，按 agent_id 升序跳过 self | [design §3 D1](./design.md) |
| D2 | belief 池化策略 | **mean**（默认），可配置 max/attention | [design §3 D2](./design.md) |
| D3 | ẑ 进入 belief 通路的形式 | **softmax 概率**（不是 logits / argmax）保留不确定性 | [design §3 D3](./design.md) |
| D4 | head_c 监督信号来源 | **Oracle `env.info["c_true"]`**（v4 Pkg-02 暴露） | [design §3 D4](./design.md) |
| D5 | L_div 实现 | **Variance hinge**: `max(0, σ_target² - Var(b_i^t))` | [design §3 D5](./design.md) |
| D6 | GRU 方向 | **uni-directional**（causal，在线推断必需） | [design §3 D6](./design.md) |
| D7 | TriContextEncoder 归一化策略 | **每路输出 LayerNorm 再 concat**（量级悬殊） | [design §3 D7](./design.md) |
| D8 | cap_emb 实现 | **2 层 MLP** (4 → 16 → d_cap_emb=16) | [design §3 D8](./design.md) |

---

## 🚨 v4 关键约束（Ch4 → 本包）

| Ch4 章节 | 对本包硬约束 |
|----------|----------------|
| **Ch4.2.2 type_emb 进 role**（**v4 新增**） | role_encoder 必须含 `type_emb_table = nn.Embedding(2, d_type_emb=8)`；type 通过 role 通路而非 belief 通路（Self Info）|
| **Ch4.2.3 ẑ 类型 2 分类**（**v4 关键改动**） | head_opp 输出 (B, N-1, 2) softmax；**不是** v3 的 d_z 维动作预测 |
| **Ch4.5.2 L_opp Oracle 监督** | L_opp 用 `env.info["types"]`（Pkg-02 暴露）作 CE 标签；**不是** v3 自监督 |
| **Ch4.2.3 ĉ raw scalar** | head_c 输出 sigmoid scalar ∈ [0, 1]；**不是** d_c 维向量 |
| **Ch4.2.4 belief 总维度 32** | belief = Concat[Proj(ĉ, d_b^proj=16), Proj(Pool(ẑ), d_b^proj=16)] = 32 |
| **Ch4.6.5 belief 梯度门控** | belief 通路前 5K step detach（具体 detach 实现在 Pkg-04，本包仅提供接口） |
| **Ch3.7 / Ch4.2.2 Self-Info 严格性** | role_i 仅含 own type，**不含**他人 type；他人 type 由 belief 通路推断 |

**最关键 v4 差异**：head_opp 从 v3 的"自监督动作预测"（d_z 维）改为 **Oracle 监督类型 2 分类**（2 维 softmax），由 `env.info["types"]` 提供 CE 标签。这是断言 B（信念专用容量）实验前提的架构基础——若 head_opp 不能学到准确的对手类型推断，主观通路 hyper_rew/hyper_pred 无法生成类型敏感 θ。

---

## ❓ Open Questions（待用户确认）

| # | Question | 默认 |
|---|----------|------|
| Q1 | TriContextEncoder 是否在内部 concat 前对每路应用 LayerNorm？ | **是**（D7 三路量级悬殊；切换需消融 ablation） |
| Q2 | BeliefNet 的 RepresentationNet 是否共享 Pkg-04 主 RepNet 实例？ | **否**（独立编码器；BeliefNet RepNet 仅 obs→hidden，无 hypernet 生成） |
| Q3 | head_opp MLP 是否共享 GRU 输出 trunk 还是各自 trunk？ | **共享** trunk（参数节省 + 与 head_c 对称） |
| Q4 | L_div hinge 的 σ_target 是否随训练步数 anneal？ | **否**（固定 σ_target=0.1，简化课程；如有 collapse 再调权重） |
| Q5 | PermutationInvariantPool 的默认策略是否硬编码 mean 还是配置可切换？ | **配置可切换**（ModelConfig.belief_pool ∈ {"mean", "max", "attention"}，已在 Pkg-01 ModelConfig 字段中预留） |

---

## 📦 输出清单（PR 时检查）

### 新增（9 文件核心 + 4 测试 + 1 脚本）

```
hyper_mve/models/
├── tri_context_encoder.py          # TriContextEncoder（三路 concat）
├── c_encoder.py                    # c_ctx 客观通路（MLP_c）
├── role_encoder.py                 # role 通路（id + type + cap, v4 含 type_emb）
├── belief_net.py                   # BeliefNet（GRU + 两 head）
├── belief_losses.py                # L_c / L_opp / L_div 三损失函数
└── permutation_invariant_pool.py   # Pool({ẑ_{i,j}}) mean/max/attention
```

### 测试（≥4 文件）

```
tests/models/
├── __init__.py
├── test_tri_context.py             # 维度断言、梯度流、permutation invariance
├── test_belief_net.py              # GRU step/seq 等价、heads 输出形状
├── test_belief_losses.py           # 三 loss 数值正确性 + 梯度方向
└── test_permutation_pool.py        # mean/max/attention 一致性
```

### 脚本（1 文件）

```
hyper_mve/scripts/
└── test_belief_net_synth.py        # 合成数据 10K 步收敛实验 (TensorBoard)
```

---

## 🔗 上下游关联

| 关系 | 包 | 接口 |
|------|-----|------|
| **本包 ← Pkg-01** | Foundation Schema | `ModelConfig` (d_c / d_role / d_belief / d_id_emb / d_type_emb / d_cap_emb / d_belief_proj / belief_gru_hidden / belief_pool); `AgentType`; `CapabilityVector`; `TimeStepRecord` (c_hat / z_hat 字段契约) |
| **本包 ← Pkg-02** | ResourceCommons env | `env.info["c_true"]` (L_c 标签) + `env.info["types"]` (L_opp Oracle 标签) + `env.info["caps"]` (cap_emb 输入) |
| **本包 → Pkg-04** | DualHyperNetwork v2 | `TriContextEncoder.forward(c_t, agent_id, cap_i, belief_i)` → ctx_i ∈ ℝ^80 直接喂 hyper_rew/hyper_pred；`BeliefNet.forward(obs_seq, action_seq)` → (b, ĉ, ẑ) |
| **本包 → Pkg-05** | Trainer | `belief_loss(...)` 接口供 trainer 联合优化；课程学习 Stage 接入点（Stage 1 oracle z 替代 head_opp 输出）|
| **本包 → Pkg-06a/b** | Baselines | shared_backbones 工厂复用 BeliefNet/TriContextEncoder（断言 B 公平性强制要求） |

---

## 📖 引用源

- 项目 Plan File: `Part 2 Pkg-03` + `Part 9 v4 修订记录`
- Hyper-MuZero v4 Roadmap §4 Stage 1 Week 2
- Chapter 4 Architecture v4: [`docs/Chapter4_Architecture_v4.md`](../../docs/Chapter4_Architecture_v4.md) §4.2 (三联通路) + §4.5 (BeliefNet 损失) + §4.6.5 (belief 梯度门控)
- Pkg-01 SDD: [`sdd/pkg-01-foundation-schema/specs/04-timestep-record.md`](../pkg-01-foundation-schema/specs/04-timestep-record.md) (c_hat / z_hat 字段契约) + [`05-v4-config-structure.md`](../pkg-01-foundation-schema/specs/05-v4-config-structure.md) (ModelConfig BeliefNet 字段)
- Pkg-02 SDD: [`sdd/pkg-02-resource-commons-env/specs/08-gym-api.md`](../pkg-02-resource-commons-env/specs/08-gym-api.md) (env.info Oracle 信号契约)

---

## 🚦 Review Checklist（用户审阅时勾选）

- [ ] proposal.md 读完，**Why** 充分（BeliefNet 是断言 B 实验前提；type_emb 进 role 是 v4 解决类型撕裂的架构落地）
- [ ] design.md 读完，**8 项 Decisions** 中无反对意见
- [ ] specs/01-08 抽查至少 3 个（重点 specs/03 含 type_emb v4 关键 + specs/05 head_opp v4 关键 + specs/08 集成契约）
- [ ] 7 天实施顺序合理（Phase 6 合成数据收敛是 hard gate）
- [ ] Open Questions Q1-Q5 默认决定 OK
- [ ] **v4 关键差异认可**：head_opp 类型 2 分类 + Oracle CE 监督，**不是** v3 自监督动作预测
- [ ] **Self-Info 严格性认可**：role 仅含 own type，他人 type 由 belief 通路推断
- [ ] 出包后可启动 Pkg-04 (DualHyperNetwork v2)
