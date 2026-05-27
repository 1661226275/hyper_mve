# Hyper-MuZero v4 项目实施路线图

> **文档目的**:本文档是从"论文设计阶段"到"代码与实验执行阶段"的交接说明。包含项目背景、设计决策、代码改动清单、实验执行顺序、决策点 checkpoint、风险预案。新工作台开始工作时,优先阅读本文档。
>
> **文档读者**:你本人 + 新工作台的 AI 助手 + (可选)毕设导师。

---

## Part 1:项目背景速读(给新助手的上下文)

### 1.1 项目一句话定义

> 在**关系非平稳混合博弈(RNS-MMG)**——一个由共享上下文 $c_t$ 与 agent 类型异质偏好联合调控合作-竞争关系的新 MARL 问题——上,构建 **Hyper-MuZero**:一个基于双路超网络 + 类型/信念/能力三联条件化 + Per-Agent MVE+CRN 规划器的 model-based MARL 算法。

### 1.2 四项核心贡献

1. **问题层面**:形式化 RNS-MMG,提出包含类型异质性的异构偏好公地博弈;
2. **算法层面**:DualHyperNetwork 通过 per-agent (type, belief, capability) 条件化的参数生成解决类型梯度撕裂;
3. **规划层面**:Per-Agent Coordinate Descent MVE + CRN 解决联合动作组合爆炸与多智能体 SNR 塌陷;
4. **基准与实验**:ResourceCommons 环境 + 6 baseline 对比 + 8 个实验验证 4 个可证伪断言。

### 1.3 四个可证伪断言

| 断言 | 内容 | 验证实验 |
|---|---|---|
| **A** | 类型梯度撕裂(类型异质下共享 RewardHead 显著劣于 per-agent θ_rew) | 消融 3(类型异质性扫描钟形曲线) |
| **B** | 信念专属容量(Hyper > Input-conditioning 在零样本泛化上) | 消融 1(架构骨架) + 零样本泛化 |
| **C** | 三联通路必要性(c_ctx / role / belief 缺一不可) | 消融 2(Context 通路拆分) |
| **D** | 规划器双重技术不可分割性 | 消融 4(Coord × CRN 的 2×2) |

### 1.4 关键设计决策清单(不可妥协)

- **类型机制**:类型 α(纯自利)+ 类型 β(Fehr-Schmidt + φ 调制),固定 2α+2β 主对比;
- **φ(c) = κ(1-2c)**,κ = 0.5(线性反对称);
- **ψ(Δ) Fehr-Schmidt 分段线性**,$\lambda_{\text{disadv}}=2.0, \lambda_{\text{adv}}=0.6$(经典值);
- **Instantaneous Δ**(不是 cumulative);
- **砍除 v3 的 β(c) 协同加成**,物理层严格自利同构;
- **Self Info 类型可观测性**(自己类型可见,他人需推断);
- **课程学习**:Oracle(前 30%) → 退火(中 40%) → 纯推断(后 30%);
- **μP 启用**:所有 baseline 等参数量对齐 + 独立 LR sweep;
- **Input baseline 双跑**:Input-Wide + Input-Deep。

### 1.5 五份 v4 设计文件交付状态

| 文件 | 状态 | 用途 |
|---|---|---|
| `Chapter1_5_Contributions_v4.md` | ✓ 完成 | 4 项贡献与 4 个可证伪断言锚定 |
| `Chapter3_Environment_v4.md` | ✓ 完成 | ResourceCommons 环境完整定义 |
| `Chapter4_1_Motivation_v4.md` | ✓ 完成 | 双路超网络的容量分配几何论证 |
| `Chapter5_Planner_Training_v4.md` | ✓ 完成 | MVE 规划器 + 训练算法 + 课程学习 |
| `Chapter6_Experiments_v4.md` | ✓ 完成 | 实验协议骨架 + 失败预案 + 决策点 |

**待补**:Chapter 4 的 4.2-4.9 节小修(局部冲突修订)、Chapter 2 相关工作、Chapter 7 总结。这些不影响代码与实验启动。

---

## Part 2:Pre-flight Check(开始代码前必须确认的 4 件事)

> 这 4 件事如果任何一件不满足,**不要开始代码改动**,先处理这 4 件事。

### 2.1 ✅ 算力可用度确认

**需求**:约 1800 GPU 小时(296 runs × 平均 6 GPU 小时/run on A100)。

**确认问题**:
- 你的实验室有多少张 A100 / V100 / 3090 / 4090?
- 持续 8-10 周内的可用度?
- 是否需要排队 / 共享资源?

**若算力 < 1500 GPU 小时**:启动 Chapter 6 节 6.2.4 的降级方案(砍消融 2/4/5 到 3 seeds,或将消融 5 移到 supplementary)。

**若算力 < 800 GPU 小时**:**严重资源紧张**,需要重新评估项目范围。优先级:必保主对比 + 消融 1 + 消融 3 + 零样本泛化。

### 2.2 ✅ Baseline 源码可获取性确认

**6 个 baseline 必须的源码状态**:

| Baseline | 状态确认 | 备注 |
|---|---|---|
| MA-MuZero (AAAI 2024) | 待确认 | 若官方未开源,寻找复现版本或自实现 |
| MAMBA (AAMAS 2022) | 待确认 | -- |
| MARIE (ICML 2024) | 待确认 | -- |
| Conflict-Aware GA (NeurIPS 2025) | **高风险** | 2025 论文,可能未开源,**最优先确认** |
| MAPPO | ✓ 通常可得 | -- |
| QMIX | ✓ 通常可得 | -- |

**行动**:在写代码前,**逐个尝试 git clone + 跑通官方 demo**。任何 baseline 无法获取,立即在 Chapter 6 协议中替换或砍除,不要等到实验阶段才发现。

### 2.3 ✅ μP 库可用性确认

**需求**:Microsoft mup 库 (github.com/microsoft/mup),用于贡献 4 加固协议 C。

**行动**:
```bash
pip install mup
python -c "import mup; print(mup.__version__)"
```

**若 mup 不可用或与你的 PyTorch 版本冲突**:启动降级方案,改用"per-baseline 独立 LR sweep"作为替代,论文中诚实说明 μP 不可用并解释替代方案。

### 2.4 ✅ 现有代码库的基线版本固定

**行动**:在开始改动前,先把现有 v1 代码库的所有文件做一次 git tag 或备份,作为"v3 baseline"。

```bash
cd <your_repo>
git tag v3-baseline-final  # 锁定 v3 状态
git checkout -b v4-implementation  # 新分支开始 v4 改动
```

**理由**:v4 改动巨大(环境、架构、训练全部涉及),如果中间出问题需要回到 v3 baseline 对比验证,必须有锁定的版本。

---

## Part 3:代码改动 Checklist(按依赖顺序)

> 改动顺序经过仔细设计,**严格按此顺序实施**,避免"改了 A 之后 B 跑不通"的依赖问题。

### Stage 1:环境层改动(Week 1,优先级 P0)

依赖关系:无前置依赖,可立即开始。

- [ ] **创建 `resource_commons.py`** (基于 `non_stationary_tag.py` 框架)
  - [ ] 资源场动力学(公式 3.1):logistic 增长 + 邻居因子
  - [ ] 斑块化资源点生成(3.3.2):多 hotspot 采样
  - [ ] 物理采集量(公式 3.2):`u_{i,k} = min(η_i, q_k/|H|)`,**严格无 β(c) 协同加成**
  - [ ] 邻居影响因子(公式 3.3):sigmoid 形式
  - [ ] α(c) 再生率(公式 3.4):线性调制
  - [ ] $c_t$ 三种演化模式(静态 / 振荡 / 随机游走)
- [ ] **类型机制**(3.5 节)
  - [ ] 类型分配:episode reset 时按 ρ_τ 分配 τ_i,episode 内固定
  - [ ] 类型 α 奖励(公式 3.5):纯物理项 + 移动代价
  - [ ] 类型 β 奖励(公式 3.6-3.9):物理项 + Fehr-Schmidt φ·ψ
  - [ ] **Instantaneous Δ_i** 计算:`Δ_i = u_i - mean(u_{j≠i})`
- [ ] **观测函数**(3.7)
  - [ ] 六块结构:self / resource / neighbor / global / capability / **type**(v4 新增)
  - [ ] type 块只含自己 type one-hot(Self Info)
  - [ ] 视野机制:切比雪夫距离 ≤ $\phi^{\text{fov}}_i$
- [ ] **环境元接口**
  - [ ] `env.observe()` 返回 `(o, {τ_i}, {cap_i})`
  - [ ] `env.step(action)` 按公式 (3.10) 计算奖励
  - [ ] `env.reset()` 重新采样类型分配 + 能力 + $c_0$
- [ ] **单元测试**
  - [ ] 资源动力学:固定 seed 下 episode 末资源存量可复现
  - [ ] 类型奖励:类型 α 奖励 = u_i;类型 β 在 c=0/Δ>0 场景下奖励减少
  - [ ] 公平共享:多人共采时单点采集量正确均分
  - [ ] **断言性单元测试**:Chapter 4.1.1 的 4 种 c × Δ 场景下,类型 β 的瞬时奖励数值与 Chapter 3.5.4 表格一致

**Stage 1 完成标志**:能在 Easy 配置(N=2)跑 10 episode 的随机策略,资源不爆炸、奖励数值与理论计算一致。

### Stage 2:架构层改动(Week 2,优先级 P0)

依赖关系:Stage 1 完成后开始。

- [ ] **`context_encoder.py` → `TriContextEncoder`**(v4 三联通路版)
  - [ ] `c_encoder`:$c_t$ → c_ctx(d_c 维)
  - [ ] `role_encoder`:[id_emb + cap_emb + **type_emb**(v4 新增)] → role_i
  - [ ] `belief_encoder`:[$\hat{c}_i$ + Pool($\hat{z}_{i,j}$)] → belief_i
- [ ] **`belief_net.py` → 新增 BeliefNet 模块**
  - [ ] GRU 主体,接收 history → b_i^t
  - [ ] `head_c`:$\hat{c}_i^t$ scalar in [0,1],MSE 损失
  - [ ] `head_opp`:$\hat{z}_{i,j}^t$ 2 维 softmax(类型 α / β 概率),**v4 关键改动**
  - [ ] **不再有动作预测 head**
- [ ] **`hyper_network.py` → DualHyperNetwork**(v4 三联输入版)
  - [ ] `hyper_trans(c_ctx)` → θ_state(客观通路,所有 agent 共享)
  - [ ] `hyper_rew(c_ctx, role_i, belief_i)` → θ_rew^i(主观通路,per-agent)
  - [ ] `hyper_pred(c_ctx, role_i, belief_i)` → θ_pred^i(主观通路,per-agent)
- [ ] **`hyper_muzero_model.py` → HyperMuZeroModel**
  - [ ] `set_context(c_t, τ_i, cap_i, b_i)` 接口(v4 接收 τ_i)
  - [ ] 内部调用 TriContextEncoder + DualHyperNetwork 生成 per-agent 权重
  - [ ] `forward_dynamics(s, a)` 使用 θ_state
  - [ ] `forward_reward(s, a)` 使用 θ_rew^i
  - [ ] `forward_prediction(s)` 使用 θ_pred^i

**Stage 2 完成标志**:能用 HyperMuZeroModel 做一次完整的 (s, a) → (r, π, v) 前向传播,各通路 shape 正确。

### Stage 3:训练层改动(Week 3,优先级 P0)

依赖关系:Stage 2 完成后开始。

- [ ] **`episode_buffer.py` / `buffer.py`**
  - [ ] 字段扩展:`(o, a, r, Δ, π_mve, v, {τ}, {cap}, ĉ, ẑ)`
  - [ ] `Δ_i^t` 在 env.step 后立即计算并存入
- [ ] **`mve_planner.py`**(Chapter 5 算法 5.1)
  - [ ] `set_context(c_t, τ_i, cap_i, b_i)` 调用扩展
  - [ ] Per-agent coordinate descent 循环(随机排列顺序)
  - [ ] CRN 实现:CRN 测试(同 seed 下相同 spa_idx 的 a_{-i} 一致)
  - [ ] **数据布局** $(B, \text{spa}, A)$,外层 spa,内层 A
- [ ] **`muzero_trainer.py`** (Chapter 5 算法 5.2)
  - [ ] `curriculum_stage(global_step / T_max)` → 'oracle' / 'anneal' / 'pure'
  - [ ] `construct_belief(stage, ẑ, {τ_j})`:按阶段构造 belief_used
  - [ ] K 步展开:RewardHead 预测目标按 buffer 中真实 r 计算
  - [ ] **类型分层视角采样**:每 batch 内类型 α / β 视角各占 ~50%
  - [ ] EMA target network 同步
  - [ ] BeliefNet loss:`L_c + 0.5·L_opp + 0.01·L_div`
- [ ] **`mup_config.py`** (新增,μP 支持)
  - [ ] 为 hyper_trans / hyper_rew / hyper_pred / StateTransNet / RewardHead / PredictionNet 配置 base_shape
  - [ ] LR sweep helper
- [ ] **`eval_protocols.py`** (新增,统一评估)
  - [ ] Self Info eval(自己类型可见,他人不可见)
  - [ ] c 分段评估([0,0.3], [0.3,0.7], [0.7,1.0])
  - [ ] 零样本泛化 eval(训练 $c \in \{0.2, 0.5, 0.8\}$, 测试未见 $c$)
  - [ ] 主指标:社会物理福利 + 可持续性 + 公平性

**Stage 3 完成标志**:能在 Easy 配置(N=2)训练 50K 步,所有 loss 单调下降,$\hat{z}$ 准确率从 50% 上升到 60%+。

### Stage 4:Baseline 适配(Week 4,优先级 P0)

依赖关系:Stage 3 完成后开始。

- [ ] **`baselines/` 目录创建**
- [ ] **`baselines/ma_muzero.py`** - 单一 PredictionNet,(s, a, type_i) 输入,共享 RewardHead
- [ ] **`baselines/mamba.py`** - Transformer world model + type token(若官方源码可用,基于源码改造)
- [ ] **`baselines/marie.py`** - 多智能体世界模型 + type embedding
- [ ] **`baselines/conflict_aware_ga.py`** - **最高风险**,若 NeurIPS 2025 源码不可得需要自实现
- [ ] **`baselines/mappo.py`** - 共享 critic + per-agent actor + type input
- [ ] **`baselines/qmix.py`** - Q-value 分解 + type conditioning
- [ ] **`baselines/input_conditioning.py`**(v4 关键,贡献 2 断言 B 的对照)
  - [ ] Input-Wide 变体:加宽 hidden_dim 匹配 Hyper 参数量
  - [ ] Input-Deep 变体:加深 depth 匹配 Hyper 参数量
- [ ] **公平性 audit**:
  - [ ] 所有 baseline 都接收 type_i 作为输入(Self Info)
  - [ ] 所有 baseline 共享 BeliefNet/RepNet/CtxEncoder(对齐范围)
  - [ ] 所有 baseline 都启用 μP

**Stage 4 完成标志**:每个 baseline 能在 Easy 配置跑 50K 步,曲线合理(不发散、不快速收敛到 0)。

### Stage 5:LR Sweep + Easy 初步实验(Week 5)

依赖关系:Stage 4 完成后开始。**这一阶段是决策点 1 之前的准备**。

- [ ] **LR Sweep on Easy**
  - [ ] 7 个 method × 5 个 LR × 3 seeds = 105 runs
  - [ ] 每 run 200K 步(Easy 配置)
  - [ ] 算力估算:105 × 1.5 GPU 小时 ≈ 160 GPU 小时
- [ ] **Easy 消融 3(决策点 1 核心实验)**
  - [ ] N=2,扫描 (1α+1β, 2α+0β, 0α+2β) 3 个数据点
  - [ ] Hyper-MuZero vs MA-MuZero
  - [ ] 3 seeds per point × 2 method × 3 point = 18 runs
  - [ ] 每 run 200K 步,算力 ≈ 27 GPU 小时

**Stage 5 完成标志**:**决策点 1 触发**(详见 Part 4)。

---

## Part 4:决策点 Checkpoint 时间表

> 这些 checkpoint 是项目的"风险管控杠杆",每个 checkpoint 都有明确的"继续 / pivot / 降级" 触发条件。

### 决策点 1(Week 5 末):Easy 消融 3 - 项目生死判官

**实验**:Easy 配置 N=2 类型异质性扫描,6 个 runs(2 method × 3 type config × 3 seeds = 18 runs,**或精简为 9 runs**)。

**算力**:约 27 GPU 小时。

**决策标准**:

| 结果 | 含义 | 行动 |
|---|---|---|
| **1α+1β 处 Hyper 比 MA-MuZero 显著优势 (p<0.05, 福利差 > 8%)** | 断言 A 在 Easy 上立住 | **继续 Medium 完整实验** |
| **2α+0β 与 0α+2β 处 Hyper 与 MA-MuZero 无显著差距** | 同质场景下 Hyper 不额外占优(符合预期) | **继续** |
| **1α+1β 处 Hyper 优势 < 5%** | 断言 A 在 Easy 上可能失败 | **触发 P0 重新定位讨论**,不继续 Medium |
| **同质场景下 Hyper 优势 > 同质对比** | 优势来源非"类型异质" | **触发**:在 Medium 阶段加 No type 消融,验证是否其他通路在贡献 |

**P0 重新定位预案**(若决策点 1 失败):

1. 检查 BeliefNet 课程学习:是否阶段 1 末期主任务 loss 仍未下降?
2. 检查 type_emb 是否正确进入 role_i 通路:set_context(τ_i) 是否实际改变 hyper_rew 输出?
3. 检查共享 RewardHead 是否真的被两类梯度撕裂:打印 6.11 节实验的偏导数值;
4. 如果三项检查都通过但 Hyper 仍无优势:**真正的 P0**——需要重新论证为什么 Chapter 4.1.2 的容量分配几何在 ResourceCommons 上不成立。可能的根本原因:
   - 类型异质度不够大(Fehr-Schmidt 参数太弱);
   - 环境物理动力学本身已经太占主导,偏好层信号被淹没;
   - 共享 RewardHead 通过 type_i input 已经能基本解耦,所谓"撕裂"在实际网络容量下不存在。

**触发 P0 的应对**:与导师讨论是否调整贡献 2 的成色(从"核心创新"降为"统一框架"),或者把贡献重心转移到 Chapter 5 的 MVE+CRN 规划器(贡献 3)。

### 决策点 2(Week 7 末):Easy 消融 1 - 断言 B 初步验证

**实验**:Easy 配置 N=2,4 个 variant(Shared / Input-Wide / Input-Deep / Hyper)× 3 seeds = 12 runs。

**算力**:约 18 GPU 小时。

**决策标准**:

| 结果 | 行动 |
|---|---|
| Hyper > max(Input-Wide, Input-Deep),福利差 > 5% | **继续 Medium 完整对照** |
| Hyper 与 Input 无显著差距,但零样本泛化有差距 | **继续**,但断言 B 措辞调整为"分布外泛化优势" |
| Hyper 与 Input 在所有指标无显著差距 | **触发断言 B 降级预案**:Chapter 4.1.2 容量分配几何论证降为"理论可能性",贡献 2 降级 |

### 决策点 3(Week 9 末):Medium 主对比 - 整体性能验证

**实验**:Medium 配置,7 method × 5 seeds = 35 runs,1M 步,约 210 GPU 小时。

**决策标准**:

| 结果 | 行动 |
|---|---|
| Hyper > 所有 baseline,平均福利差 > 15% | **完美**,继续所有消融 |
| Hyper > 大部分 baseline,但某些 baseline 接近 | **正常**,正常推进 |
| Hyper 与 MA-MuZero 无显著差距 | **严重**,触发全面降级方案,论文措辞大调 |
| Hyper 反输某些 baseline | **极严重**,可能存在工程 bug,先排查 |

### 决策点 4(Week 11 末):全实验完成 - 论文写作启动

**所有消融、零样本泛化、可视化完成**。这时候应该有 250+ runs 的数据。

**决策**:
- 哪些断言通过、哪些失败?
- 论文措辞如何调整?
- 哪些 figures / tables 进正文、哪些进 supplementary?

---

## Part 5:风险登记册(Risk Register)

| 风险 | 概率 | 影响 | 缓解措施 |
|---|---|---|---|
| **算力不足** | 中 | 高 | Pre-flight 2.1 早期确认;若不足启动降级方案 |
| **Baseline 源码不可得** | 中(尤其 Conflict-Aware GA) | 中 | Pre-flight 2.2 早期确认;若不可得替换或砍除 |
| **决策点 1 失败(断言 A 不成立)** | 中 | **极高** | Easy 消融 3 提前在 Week 5 末完成,留 7 周时间 pivot |
| **决策点 2 失败(断言 B 不成立)** | 中 | 高 | 降级预案:断言 B 措辞改为"分布外优势" |
| **课程学习不收敛** | 中 | 中 | 延长阶段 2 退火期;若仍无效,简化 BeliefNet 任务 |
| **训练不稳定 / 表示坍缩** | 低 | 中 | 已有 BYOL consistency loss + EMA target,通常能解决 |
| **类型分配的 ID 偏置** | 低 | 低 | seed 间随机化 (类型, agent_id) 绑定 |
| **Reward scale 失控** | 低 | 中 | Instantaneous Δ 设计下尺度自然对齐,但仍需监控 |
| **导师 / 答辩委员会对设计有反对意见** | 低-中 | 高 | 现在阶段就把 5 份 v4 设计文件给导师过一遍,避免后期翻盘 |

---

## Part 6:新工作台启动建议

### 6.1 第一步:把 5 份 v4 文件 + 本文档上传到新工作台

将以下文件上传到新工作台的项目中:

```
项目知识库/
├── Chapter1_5_Contributions_v4.md       (4 项贡献 + 4 断言锚定)
├── Chapter3_Environment_v4.md           (ResourceCommons 完整定义)
├── Chapter4_1_Motivation_v4.md          (双路超网络几何论证)
├── Chapter5_Planner_Training_v4.md      (MVE + 训练算法 + 课程)
├── Chapter6_Experiments_v4.md           (实验协议 + 失败预案)
├── Hyper_MuZero_v4_Roadmap.md           (本文档)
└── 现有代码库/                            (v1 代码,作为改动基线)
    ├── non_stationary_tag.py
    ├── context_encoder.py
    ├── hyper_network.py
    ├── hyper_muzero_model.py
    ├── episode_buffer.py
    ├── buffer.py
    ├── muzero_trainer.py
    ├── mve.py
    └── mve_planner.py
```

### 6.2 第二步:新工作台的第一个 prompt 建议

```
我从一个论文设计阶段的对话中过渡过来,现在进入代码实施与实验执行阶段。

请先阅读项目知识库中的 Hyper_MuZero_v4_Roadmap.md 文档,
这是我们之前讨论达成的项目总体路线图。然后阅读其他 5 份 Chapter 文档了解设计细节。

阅读完成后,请协助我:
1. 完成 Pre-flight Check 的 4 项确认(算力 / baseline 源码 / μP / git tag);
2. 按 Stage 1 顺序开始环境层改动(创建 resource_commons.py)。

我的当前状态:
- 算力情况:[填入你的算力情况]
- 已确认的 baseline 源码:[填入]
- μP 库是否可用:[填入]
- 现有代码库已 git tag 为 v3-baseline:[是/否]
```

### 6.3 第三步:工作节奏建议

- **每周一次进度同步**:与导师对齐当前 stage、blocker、决策点状态;
- **每 stage 完成后 git commit + tag**:`v4-stage1-env-done`, `v4-stage2-arch-done` 等;
- **每个决策点准备 1 页 summary**:实验结果 + 决策建议,与导师对齐;
- **保留每周的工作日志**:Markdown 文件记录"本周完成、本周阻塞、下周计划",便于后续答辩时回顾。

### 6.4 第四步:不要陷入的常见陷阱

1. **不要先调超参再跑主对比**:LR sweep 必须**所有 baseline 公平进行**,不要"我方调好、对方未调";
2. **不要看到 Hyper 输了就改设计**:负结果也是结果,失败预案已经预先定义,按预案执行;
3. **不要为了让钟形曲线漂亮而调 Fehr-Schmidt 参数**:消融 5 就是为了证明结论不依赖参数,提前调反而是数据造假;
4. **不要在论文写完前修改 v4 设计文档**:5 份文件是承诺,实验阶段只填数据不动锚;若必须修改,在 changelog 中明确记录;
5. **不要省略失败案例的报告**:第 6.12 节"失败案例分析"是诚实学术写作的标志,审稿人很看重。

---

## Part 7:本路线图的修订记录

| 版本 | 日期 | 修订内容 |
|---|---|---|
| v1.0 | 2025-XX-XX(填入实际日期) | 初版,从论文设计阶段过渡到实施阶段 |

修订时请在此处追加新行。

---

## 最后的话

到这一步,Hyper-MuZero v4 项目已经从"想法"变成了"工程"。

5 份设计文档定义了**做什么**与**为什么做**,本路线图定义了**怎么做**与**何时检查**。剩下的工作是工程执行+实验验证,这是相对程序化的——只要严格按照 Stage 顺序、严格在 Checkpoint 时做决策、严格按失败预案应对负结果,项目就能在 12 周内完成。

**最重要的是 Week 5 末的决策点 1**。不要拖延、不要美化、不要为了"看起来更好"调参——按协议跑,按结果决策。如果断言 A 在 Easy 上失败,你有 7 周时间 pivot;如果你拖到 Medium 完成才发现,就只剩 5 周,选择空间小得多。

祝顺利。
