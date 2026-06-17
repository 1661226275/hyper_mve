# 第 2 章 相关工作 — 开场段草稿

> 目标字数：~7 500 字 (~15 页) · 锁定决策：Q4 = B 三层式（问题→方法→评测）· 4 节结构 · 不强调"补全文献空白"

---

## §2.1 混合动机多智能体强化学习问题谱系（~2 200 字）

> 多智能体强化学习的早期工作多聚焦于纯合作（团队共享单一奖励）与零和（一方收益等于另一方损失）两类极端。Lowe 等人 [Lowe 2017] 的 MADDPG、Rashid 等人 [Rashid 2018] 的 QMIX、Yu 等人 [Yu 2022] 的 MAPPO 分别在 CTDE（centralized training, decentralized execution）框架下推进了纯合作 MARL 的样本效率与可扩展性。然而真实世界的多智能体系统大多不落在这两个极端。Leibo 等人 [15] 将经典两步囚徒博弈扩展为序贯社会困境（Sequential Social Dilemma, SSD），通过 Harvest（资源采集）与 Cleanup（污染清理）两个环境刻画"个体最优与集体最优不一致"的情形。后续工作沿 SSD 范式发展出多个分支：Hughes 等人 [11] 在奖励函数中引入 Fehr-Schmidt 不平等厌恶项作为内在合作激励；Jaques 等人 [Jaques 2019] 提出 Social Influence 作为内在动机；Yang 等人 [Yang 2020] 的 LIO 让 agent 学习对其他 agent 给予奖励；Kim 等人 [13] 的 Conflict-Aware Gradient Adjustment 在静态混合博弈下调整梯度方向以缓解策略冲突。这些工作的共同假设是博弈关系本身在整个 episode 与训练过程中保持静态：一旦环境被设定为"合作主导"或"竞争主导"，相应的支付结构在所有 episode 中不再发生结构性改变。本文研究的 RNS-MMG 正是放松这一假设的情形——博弈关系由可观测物理状态 c_t 动态调控。

**本节 beats**：

1. 合作-竞争-混合动机三段式分类
2. 序贯社会困境 (SSD) 与代表环境 (Harvest, Cleanup, Melting Pot)
3. reward shaping 与 inequity aversion 家族 (Hughes 2018, Jaques 2019, Yang 2020)
4. 与 RNS-MMG 的关系：从静态博弈结构到关系非平稳
5. 文献小结

**核心引用**：[15][11][14] + Jaques 2019 + Yang 2020 + [13] CAGA + Lowe 2017 MADDPG

---

## §2.2 多智能体世界模型方法（~2 600 字，含中等版 Harsanyi）

> 基于学习世界模型的规划方法在单智能体场景下已取得突破性表现。Schrittwieser 等人 [24] 的 MuZero 用学到的潜在动力学模型替代真实模拟器进行 MCTS 搜索，在 Atari、围棋、国际象棋上同时达到 SOTA；Hafner 等人的 Dreamer 系（DreamerV3，2023）则用学到的递归状态空间模型支持策略学习与长期规划。将这一范式扩展到多智能体场景的代表性工作包括：[30] MA-MuZero 用 Gumbel 采样部分缓解联合动作空间爆炸问题；Egorov 与 Shpilman [Egorov 2022] 的 MAMBA 使用 Transformer 架构融合多 agent 观测；Liu 等人 [Liu 2024] 的 MARIE 进一步引入类型 token 支持异质 agent。这一谱系上的关键设计选择是世界模型中各功能网（状态转移、奖励、策略-价值）的参数化方式：现有方法均采用所有 agent 共享同一组权重的方案，仅通过输入侧拼接 agent_id 或 type token 来区分。这一方案在偏好结构异质的 RNS-MMG 下面临 §4.1 将形式化的"类型梯度撕裂"问题——类型 α 与类型 β 的奖励梯度在 c_t 的部分区段方向相反，反向传播会把共享 RewardHead 拉向类型平均策略。
>
> **条件化机制谱**为此提供了一个统一的分析视角：Input Conditioning（CAVIA [43]）在共享权重之上拼接上下文；FiLM（Perez 等人 [44]）通过对角调制 (γ, β) 实现轻量条件化；LoRA（Hu 等人，2022）通过秩-r 增量 ΔW = BA 实现中等容量；Ha 等人 [42] 的 HyperNetworks 则在谱的另一端——完整生成所有参数。本文第 4 章将沿这一谱系做更细的实证比较。
>
> **社会偏好建模**为本文偏好层设计提供了理论参照。Fehr 与 Schmidt [40] 的不平等厌恶模型用 (λ_disadv, λ_adv) 两参数刻画对劣势与优势的非对称反感，本文沿用其经典值（λ_disadv=2.0, λ_adv=0.6）。Harsanyi [10] 在 1967 年提出不完全信息博弈框架，把每个 player 拆解为"共同知识"（所有人都同意的物理规律与他人理性程度）与"私人信念"（对他人类型的后验估计）两层。本文将这一二分作为 DualHyperNetwork 双路分解的类比依据，并不主张这是 Harsanyi 框架在深度世界模型中的首次实现。

**本节 beats**：

1. 单智能体世界模型：MuZero 系（含 Sampled / Gumbel MuZero）
2. 多智能体世界模型：MA-MuZero / MAMBA / MARIE 的设计与共享 RewardHead 问题
3. 条件化机制谱：Input Conditioning ↔ FiLM ↔ LoRA ↔ HyperNet
4. 社会偏好建模与 Harsanyi 类型博弈（中等版本，~600 字）

**核心引用**：[24] Schrittwieser 2020 Nature + [30] MA-MuZero + MAMBA + MARIE + Dreamer + [42][43][44] + LoRA + ToMnet + [40] + [10] + [45]

---

## §2.3 基准与评测（~1 800 字）

> 现有 mixed-motive MARL 基准可按"博弈关系是否动态切换"与"偏好层是否显式异质"两维度分类。Harvest 与 Cleanup [15][11] 提供静态、同质的 SSD；Melting Pot [14] 在多个静态场景上评测泛化；Conflict-Aware GA [13] 引入静态混合博弈下的类型差异。本文构建的 ResourceCommons 同时引入两维度：c_t 调控的博弈关系动态切换 + 类型 α/β 的偏好结构异质。评测指标家族沿用 SSD 工作的社会福利 V_total 与可持续性指标（回合末资源存量），并补充三项 RNS-MMG 特有指标：(i) 零样本泛化：训练时见 c ∈ {0.2, 0.5, 0.8}，测试 c ∈ {0.0, 0.35, 0.65, 1.0}；(ii) 隐藏-c 信念质量：BeliefNet 推断的 ĉ_i 与真实 c_t 的 MSE 与 Pearson 相关系数；(iii) 类型梯度对齐：学习到的 ∂ř/∂u_i 与理论值 ({1} for α, {0,0.7,1.3,2.0} for β) 的距离。公平对比方面，本文采用 Yang 与 Hu 的 μP 协议进行学习率自动缩放，并对每条 baseline 独立执行 5×3 LR sweep。

**本节 beats**：

1. 现有混合动机基准对比表（Harvest / Cleanup / Melting Pot / ResourceCommons）
2. 评测指标家族：社会福利、可持续性、Gini 公平、零样本泛化
3. 公平对比协议（μP、独立 LR sweep）的需求

**核心引用**：[14] Melting Pot + [9][16] CPR 经典 + Yang & Hu 2021 μP

---

## §2.4 小结：本文位置（~900 字）

> 本文沿用 SSD 的资源公地骨架（[15][11]），但将博弈关系作为 c_t 的函数；沿用 MuZero 的世界模型 + 学习权重生成范式（[24][30]），但加入主-客通路分解与按 type 条件化的权重生成；沿用 Fehr-Schmidt 偏好结构（[40][11]），但只放在 §3.5 偏好层、不进物理层；沿用 Harsanyi 的共同知识/私人信念二分（[10]），仅作为方法选择的理论参照系。本文不主张开拓新研究方向，仅主张：(a) 把 RNS-MMG 这一具体问题写清楚；(b) 给出一套与该问题约束相对应的 DualHyperNetwork v2 架构；(c) 用四项可证伪断言完成实证验证。
