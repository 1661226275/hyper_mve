# 硕士论文主规划

**T1 答辩题**：关系非平稳混合动机博弈下的双路超网络 Model-based 多智能体强化学习
**英文题**：Dual-Pathway Hypernetwork Model-Based MARL for Relation-Nonstationary Mixed-Motive Games

总字数目标：**~68 000 字** · 实验算力预算：**~300 GPU hours**

---

## 0. 全文叙事主线（~300 字）

真实社会中的多智能体系统——渔场、电网、带宽——很少处在纯合作或零和的两极，而是处在"合作-竞争"持续张力中的混合动机博弈区段；更关键的是，这种博弈关系本身并不静态，而是随可观测的物理状态（尤其是资源丰度 c_t）动态切换。本论文以这一在现有 mixed-motive MARL 工作中尚未被系统讨论的子类——**关系非平稳混合动机博弈（RNS-MMG）**——为研究对象。前期 v3 工作（NonStationaryTag，毕设汇报已给出 Capture Rate 75% vs 10% 与超网权重显著分化的实证）首次验证了"以超网络条件化博弈视角"这一思路在非平稳对抗任务上的可行性，但也暴露了三重瓶颈：**类型梯度撕裂、信念稀释、角色平均**。

v4 由此把问题推进到更具理论一致性的公地资源博弈（**ResourceCommons**）：物理层严格自利同构、偏好层显式异质（类型 α 纯自利，类型 β 以 φ(c)ψ(Δ) 的 Fehr-Schmidt 结构调制），让合作-竞争从动力学与偏好的交互中真正涌现，而非由 reward shaping 强行注入。方法上，本文提出 **DualHyperNetwork v2**——以三联上下文（共同知识 c_ctx、角色 role_i、信念 belief_i）严格对应 Harsanyi 不完全信息博弈分解，客观路径（hyper_trans）共享、主观路径（hyper_rew/pred）按 agent 分支；并在**条件化谱**（Input ↔ FiLM ↔ LoRA ↔ base_gen ↔ full）上把"超网络是否优于 Input 条件化"升级为"per-context 容量在谱的何处开始得不偿失"。规划层以 **Per-Agent Coordinate Descent + CRN** 将 O(A^N) 压至 O(N·A) 并把第 0 步 SNR 从 0.02 拉回 1.0。

**四项可证伪断言**（A:类型梯度撕裂的钟形曲线；B′:谱内部最优；C:三联通路缺一不可；D:规划器双重技术不可分割）在 ResourceCommons 与 NonStationaryTag 上均给出预登记验证方案，确保问题→方法→基准→结果形成可被否定也可被复现的闭环。

---

## 1. 六章结构（章节字数预算）

| 章 | 标题 | 字数 | 关键内容 |
|---|---|---|---|
| 1 | 绪论 | 8 000 | RNS-MMG 问题陈述 · 四项贡献 · 四项断言 |
| 2 | 相关工作 | 7 500 | mixed-motive MARL · MA 世界模型 · 条件化谱 · 社会偏好（B 三层式） |
| 3 | 问题与环境 | 9 600 | RNS-MMG 形式化 · v3 Genesis · ResourceCommons · 命题 3.1 |
| 4 | 方法 | 18 500 | 三瓶颈 · 条件化谱 · Harsanyi · DualHyperNet v2 (D3) · Coord+CRN · 训练算法 |
| 5 | 实验 | 18 000 | 协议 · 决策门 0 · v3 迁移 · 主对比 · 五项消融 · 鲁棒性 · 信念 · 失败案例 |
| 6 | 总结与展望 | 4 000 | 贡献复述 · 局限 · W2 两方向 · 结语 |
| 附录 A–E | 命题证明 / 符号表 / gen_scope 纪律 / 超参表 / 复现指南 | ~14 页 | （见 ch6.md） |
| **合计** | | **~65 600 字** | + 14 页附录 |

每章节级 beats 详见 `drafts/ch1.md` 到 `drafts/ch6.md`。

---

## 2. 四项可证伪断言 × 实验映射

| 断言 | 一句话陈述 | 实验位置 | 章节 | 失败回退 |
|---|---|---|---|---|
| **A** | 类型异质条件下，单组共享 RewardHead 被 α(梯度=1) 与 β(梯度∈[0,2]) 反向梯度撕裂；per-agent θ_rew 在 ρ_β≈50/50 处优势达峰，形成钟形曲线 | 消融 3 + §6.11 类型梯度可视化 | §5.7 + §4.1 + §3.6 | Easy N=2 峰值差 <5% → 贡献 2 降级为"特定区段优势" |
| **B′** | 条件化谱内部最优：左端容量平均化、右端 FULL 方向坍缩(0.61→0.998)双面失败；峰值在 film_head+LoRA 至 lora_fc2 之间 | 消融 1 + 零样本 c 泛化 + 决策门 0 gen_scope sweep | §5.5 + §5.2 + §4.2 | 三分级分别降级，最差降为相关工作分析视角 |
| **C** | 三联通路（c_ctx / role_i / belief_i）退化效应正交可加，缺一不可；在 hidden-c 下 belief 通路效应可分离 | 消融 2 + 信念质量协议 | §5.6 + §5.10 + §4.3 | 非正交 → 弱版本"三路径互补"；若 No-type ≡ No-belief 补 No-self-type 消融 |
| **D** | CoordDesc × CRN 不可分割：CRN 把第 0 步 SNR 从 0.02 拉回 1.0；CoordDesc 把 O(A^N) 压到 O(N·A) | 消融 4 (2×2 矩阵 Easy N=2) + §5.1.2 SNR 实测 | §5.8 + §4.5 + §5.9 (planner-off) | Medium 不可行只 Easy 拿四象限 → 强调 ε-Nash 理论保证 |

---

## 3. v3 → v4 桥接策略（方案 C 混合）

> **决策**：Q2 = 方案 C 混合 — §3.2 轻量 Genesis（1.5–2 页）+ §5.3 重量级 apples-to-apples 迁移

v3 NonStationaryTag 在毕设汇报已交付 **Capture Rate 75% vs 10%** + 超网权重在阵营翻转处的发散图（cos_rew_0v3 0.37→0.28；cos_pred_0v3 0.65→0.50）。论文采用 "Genesis → Structural Upgrade" 双层叙事：

1. **§3.2 初步研究**（~2 页，G2 中等版）：v3 显式作为"思路可行性的最初实证"，包含核心数字 + Blocking Point 奖励正交化半页 + 示意公式。指出 v3 任务限制（零和、无类型梯度撕裂、偏好层未异质）反向暴露三大瓶颈 → 这三个瓶颈直接成为 §4.1 v4 三联通路的 motivation
2. **§5.3 迁移实验**（~1 200 字）：把 v4 完整方法（DualHyperNetwork v2 + Coord+CRN + 决策门 0 选定 gen_scope）部署回 NonStationaryTag，做 apples-to-apples 对比，证明 "v4 在 v3 任务上不劣于 v3 原版（Capture Rate ≥ 75%）、在 RNS-MMG 上严格优于" —— 同时回应"是否回退"和"是否只在新任务上 work"两类质疑
3. **§1.6 / §6.4**：把研究路径表述为"从 v3 NonStationaryTag 到 v4 ResourceCommons"，让 v3 成为论文身份的一部分

---

## 4. GPU 算力账本（~300 hours 上限）

| 实验 | 配置 | runs | 单 run | 小计 |
|---|---|---|---|---|
| 决策门 0 gen_scope sweep | duo N=2, random walk c | 6 cells × 3 seeds × 11k steps | ~1.7h | ~30h |
| ResourceCommons 主对比 | Medium N=4 2α+2β | 6 baseline × 5 seeds = 30 | ~6h | ~180h |
| μP LR sweep | Medium 主配置 | 5 LR × 3 seeds × 6 baseline | ~1h | ~90h |
| v3 NonStationaryTag 迁移 | REL ∈ {0, 0.5, 1.0}, 3 seeds | 9 | ~3h | ~27h |
| 消融 1 (gen_scope 7 变体) | Medium | 7 × 3 seeds | ~6h | ~126h |
| 消融 2 (Context 路径拆分) | Medium 可见+hidden 双档 | 5 conds × 2 档 × 3 seeds | ~6h | ~180h |
| 消融 3 (类型扫描) | ρ_β 五点 + 对照 | 6 × 5 seeds | ~6h | ~180h |
| 消融 4 (规划器 2×2) | Easy N=2 | 4 × 3 seeds + Medium 二轴 | ~3h | ~30h |
| 消融 5 (FS 参数 3×3) | Medium | 9 × 3 seeds | ~6h | ~162h |
| Hard scale-up | N=8 | 3 seeds | ~12h | ~36h |
| 信念质量 + 课程 | 双档 + no-curriculum | 6 × 3 seeds | ~6h | ~108h |
| **理论合计** | | | | **~1149h** |
| **实际优化后预算** | | | | **~300h** |

**优化策略**：(i) 消融 1/2/3 在主对比 runs 之上做 reuse（共享 baseline 训练曲线）；(ii) Hard N=8 与 FS 参数 3×3 在 P2 优先级，可视进度延后；(iii) μP LR sweep 单独 3 seeds 即可（不重复 5 seeds）；(iv) Easy 配置 wall-clock 较短，可串行而非占用独立 GPU。

实际 300 GPU hours 实施时按 P0 → P1 → P2 三优先级降级（详见 §5.1）：
- **P0**（必跑）：决策门 0、ResourceCommons 主对比、v3 迁移、消融 1/3/4、信念质量 - 约 230h
- **P1**（高优）：消融 2 hidden-c、消融 5 FS sensitivity、no-curriculum baseline - 约 50h
- **P2**（可选）：Hard scale-up、Joint enumeration on Medium、cap_emb ablation - 约 20h

---

## 5. 风险登记（12 项）

| # | 风险 | 论文层缓解 |
|---|---|---|
| R1 | v3 NonStationaryTag 结果被评委视为"与 v4 任务不一致，不应放进同一论文" | §3.2 显式定位 + §5.3 迁移实验 apples-to-apples |
| R2 | FULL 方向坍缩(0.61→0.998)被解读为工程实现 bug 而非可证伪断言 B′ 的失败模式 | §4.4 详述 v4-opt 2026-06 根因分析 + §5.5 预登记 B′(ii) 判据 + group RMS / LoRA 两条独立工程对照 |
| R3 | 断言 A 钟形曲线非对称(峰值偏 α 或偏 β)，被质疑为"拟合现象而非理论预测" | §5.7 预声明非对称钟形为 informative + 1α+3β vs 3α+1β 对照 + §6.11 梯度可视化补强 |
| R4 | 零样本泛化实验中所有方法都 >90%，导致 "ResourceCommons c-modulation 太平滑" 的质疑 | §5.5 预登记"若全部 >90% 扩大 train-test c gap"应急 + 报告 hidden-c 档作为更难泛化测试 |
| R5 | 主对比社会福利提升 <10%，贡献 2 在工业意义上不充分 | §5.4 预登记"若 Medium <10% 或 Easy+Medium 均 <5%，贡献 2 降级为特定区段优势"话术 + §5.5/§5.6 内部峰值与通路必要性提供方法学层面独立贡献 |
| R6 | BeliefNet ẑ 准确率在 Stage 3 纯推断下退化，质疑 Harsanyi 分解的可学性 | §5.10 预登记"GRU hidden_dim 从 128 升 256 或加大 FS 参数提高类型可分性"应急 + 课程 vs 无课程对照证明课程必要性 |
| R7 | Joint 枚举在 Medium 不可行，断言 D 只能在 Easy N=2 拿满四象限，评委可能质疑普适性 | §5.8 坦诚披露 + Medium 上 Coord×CRN 两轴作为必要性证据 + Easy 四象限作为充分性证据 + §4.5 强调 CoordDesc 的 ε-Nash 对应作为可扩展性理论保证 |
| R8 | 评委质疑"用 Fehr-Schmidt 单一偏好家族就声称揭示了 mixed-motive 通用机制" | §5.9 FS 参数敏感性扫描(3×3 网格) + §2.4 文献定位 + §6.2 局限节明确指出 Charness-Rabin 等家族留作未来 |
| R9 | 超网络生成的参数对小数据集过拟合，baseline 在样本效率上反超 | §5.1 独立 LR sweep + μP 对齐 + §4.4 四层稳定性栈防御 + §5.11 失败案例节披露所有观察问题 |
| R10 | Pkg-02 c_visible 与 Joint 枚举工程进度滞后，影响实验完整性 | §5.1 实验协议节明确"已完成/in-flight/计划中"三档 + §5.10 hidden-c 档与 §5.8 Joint 档独立设为可降级单元 + §5.11 透明披露未完工件 |
| R11 | 评委指出"preference invariance principle"与 RLHF / 奖励重塑文献存在表面冲突 | §2.4 + §4.3 明确区分"偏好结构在 RL 训练中保持不变 ≠ 偏好不可学" + v4 砍除 β(c) 的演化作为可操作化实例 |
| R12 | 决策门 0(gen_scope sweep)选择的最优 cell 与 §4.2 理论预测点不一致 | §4.2 明确"最优点在 film_head ↔ lora_fc2 之间"作为 soft prediction + §5.2 记录决策门 0 结果作为 empirical refinement + §5.5 消融 1 独立报告 7 变体 |

---

## 6. 未完工件（必须在 §5 实验前完成）

下游工程任务，详见 Task #7、Task #8：

1. **Pkg-02 `EnvConfig.c_visible` 开关 + global block 常数掩蔽**
   - 影响：§5.10 hidden-c 信念质量实验 + 断言 B′(ii)
   - 设计：env config 加 `c_visible: bool = True`；当 False 时 observation 的 c_t 字段被 `0.5` 常数替换；ground-truth c_t 仍保留在 env state 中作为 BeliefNet 的 oracle 监督

2. **Pkg-05 MVE planner `planner_mode=joint_enum` 路径**
   - 影响：§5.8 断言 D 的完整 2×2 Easy N=2 验证
   - 设计：Easy N=2, A=6 → 36 joint actions；joint_enum 模式下枚举所有 36 组合并完整 K-step rollout；与 CoordDesc 模式共用同一 K-step 评估代码

---

## 7. 写作模式建议

Plan Mode 已完成，后续推荐切换：

- **`/ars-full`** — 用本规划作为种子，让 academic-paper skill 走完整 Phase 0-7 流水线
- **手写优先**：`drafts/ch1.md` 至 `drafts/ch6.md` 已是可直接编辑的开场段，每节约 400-500 字。展开时按 beats 顺序补完即可
- **章节循序**：建议按 1 → 3 → 4 → 2 → 5 → 6 顺序写。理由：第 1 章定调最难，先写；第 3 章 RNS-MMG 形式化是 §4 方法节的依赖，需先于 §4；第 2 章相关工作在 §3/§4 完成后写更精准；第 5 章实验等 P0 实验完成后写；第 6 章最后

---

## 引用与术语

- **完整引用注册表**：`CITATION_REGISTRY.md`
- **15 条 INSIGHT 三角化**（12 保留 + 3 降级）：`INSIGHTS.md`
- **决策日志**（Q1–Q15 含理由）：`README.md`
