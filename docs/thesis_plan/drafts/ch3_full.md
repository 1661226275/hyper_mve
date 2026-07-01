# 第 3 章 问题与环境

> 本章共分七节。§3.1 在第 1.2 节非形式描述基础上给出关系非平稳混合动机博弈（RNS-MMG）的形式化元组与三条结构性约束（C1/C2/C3）。§3.2 简要复述 v3 NonStationaryTag 阶段的环境设定、DualHyperNetwork 原型与毕设汇报阶段的实证结果，作为本文方法的实证起点。§3.3 给出 ResourceCommons 基准的核心设计原则——物理层与偏好层的二层分离。§3.4 进一步给出资源场动力学与上下文 c_t 通过单通道再生强度 α(c) 调控的具体形式。§3.5 给出偏好层的类型 α / 类型 β 与 Fehr-Schmidt 结构，以及 Self / Full 两档观测设定。§3.6 给出 ResourceCommons 唯一形式化命题——命题 3.1（Gap(c) 单调性），以"正文陈述 + 附录证明"方式呈现。§3.7 给出评测指标体系并标注与各断言的对应关系。

---

## §3.1 RNS-MMG 形式化

第 1.2 节给出 RNS-MMG 的非形式描述：博弈关系由可观测物理状态 c_t 动态调控。本节将其形式化为 N 个 agent 的元组 $G = \langle \mathcal{N}, \mathcal{S}, \{\mathcal{A}_i\}, P, \{R_i\}, \{O_i\}, \mathcal{C}, P_c, \rho_c, \gamma \rangle$。其中 $\mathcal{N} = \{1, \dots, N\}$ 是 agent 集合；$\mathcal{S}$ 是全局状态空间；$\mathcal{A}_i$ 是 agent $i$ 的动作空间，联合动作记 $a = (a_1, \dots, a_N) \in \prod_i \mathcal{A}_i$；$P: \mathcal{S} \times \prod_i \mathcal{A}_i \to \Delta(\mathcal{S})$ 是物理转移核；$R_i: \mathcal{S} \times \prod_i \mathcal{A}_i \to \mathbb{R}$ 是 agent $i$ 的瞬时奖励函数；$O_i$ 是 agent $i$ 的部分观测函数；$\mathcal{C} \subseteq \mathbb{R}$ 是共享上下文空间，$P_c$ 与 $\rho_c$ 分别是 $c_t$ 的内禀转移核与初始分布；$\gamma \in (0, 1)$ 是折扣因子。在该元组之上施加三条结构性约束。

**约束 C1（物理转移的 agent-同构性与上下文内禀演化性）**：转移函数 $P(s' \mid s, a)$ 在所有 agent 间共享（agent-invariant），其函数式由环境本身固定，不随 $c_t$ 切换为不同的函数形式；$c_t$ 作为状态的一部分按内禀动力学 $P_c$ 演化，并通过资源场动力学 $q_{k,t+1} = f(q_{k,t}; \alpha(c_t))$ 间接调控博弈结构。此处需精确区分两件事——$c_t$ 是状态的分量（作为状态进入 $P$ 与 $\{R_i\}$），但物理转移 $P$ 本身的函数式不因 $c_t$ 值而切换；这一区分使 C1 与"上下文索引转移规则切换"（Contextual MDP / HiP-MDP）划清边界。

**约束 C2（博弈关系的类型-上下文联合调控）**：奖励函数 $R_i(s, a; c_t, \tau_i)$ 由 $c_t$ 与 agent 类型 $\tau_i \in \{\alpha, \beta\}$ 联合调制。本文在 §3.5 给出一个具体的偏好实例化（基于 Fehr-Schmidt 不平等厌恶 [40] 并由 $c_t$ 进行强度调制），其中类型异质 agent 在某些 $c_t$ 区段呈现结构性相反的瞬时奖励梯度方向——例如类型 $\alpha$ 的 $\partial R^\alpha / \partial u_i \equiv 1$，类型 $\beta$ 在 $c$ 低区段对自身相对优势状态表现出反向梯度（$\partial R^\beta / \partial u_i < 1$ 甚至 $\le 0$）。"梯度相反"是 §3.5 建模选择下的可检验性质，C2 本身只要求类型与上下文联合非平凡地调制偏好。

**约束 C3（上下文物理不可控性）**：$c_t$ 由环境内禀机制（资源场再生 + 外部驱动）演化，agent 动作对其只有间接或缓慢影响。该约束将 $c_t$ 与"可由 agent 直接操控的环境变量"区分开，避免偏好层因果归因被 agent 动作-上下文耦合干扰；同时使 RNS-MMG 与 Harsanyi [10] 不完全信息博弈框架在结构上相容——类型 $\tau_i$ 属于 agent 私人信息，$c_t$ 属于 episode 内可观测或可推断的共同知识层。

RNS-MMG 与几类相邻问题的边界已在 §1.2 给出，此处仅作要点复述：与一般非平稳 MDP / Markov Game [20][21] 相比，非平稳源自一个可被观测或推断的物理变量而非显式时间索引；与静态 mixed-motive（Cleanup, Harvest）[15][11] 相比，博弈关系允许随 $c_t$ 动态切换；与领域随机化相比，$c_t$ 在 episode 内有明确的内禀演化规律。RNS-MMG 概念上可记作"Contextual MARL + C2 类型异质 + C3 内禀不可控 + intra-episode 演化"四要素的合取——任一要素单独存在均落入已有家族。

**部分可观测的处理**：RNS-MMG 严格来说是部分可观测多智能体马尔可夫博弈（POSG）的子类——每个 agent 通过观测函数 $O_i(s, c_t; \tau_i)$ 得到状态的一个函数式化投影而非全局状态 $s$ 本身。在 Self Info 观测档下，$O_i$ 保留自身位置、速度、视野内资源分布、自身类型 $\tau_i$ 与共享上下文 $c_t$，屏蔽他人类型 $\tau_{-i}$；在 Full Info 档下 $O_i$ 额外保留他人类型（§3.5 展开）。$c_t$ 的可观测性受 $c_{\text{visible}}$ 开关控制（§3.4 展开）。这一部分可观测结构使 RNS-MMG 的策略必须在**信念空间**上定义——形式化地，令 $b_i^t = (o_i^{1:t}, a_i^{1:t-1})$ 为 agent $i$ 在时刻 $t$ 的信息集，则最优策略 $\pi_i^*: b_i^t \to \Delta(\mathcal{A}_i)$ 是历史泛函而非瞬时状态函数。若 $b_i^t$ 的马尔可夫压缩由 GRU 递归状态承载，则策略网络的输入实为 GRU 隐状态，这一实现选择在 §4.4 BeliefNet 结构中给出。

**存在性与均衡假设**：本文形式化的 RNS-MMG 假设满足两条最低正规性条件——(i) 动作空间 $\mathcal{A}_i$ 有限（本文取离散化 $|\mathcal{A}_i| = 5$，见 §3.4）；(ii) 转移与奖励函数关于 $(s, a, c_t, \tau_i)$ 局部利普希茨。这两条条件保证有限步 $\varepsilon$-Nash 均衡的存在性（Fudenberg & Tirole 1991 博弈论标准结果）。本文并不主张对任意 RNS-MMG 实例都能高效求解 Nash 均衡——多智能体强化学习在该设定下仅能追求 $\varepsilon$-Nash 或近似最优反应策略；§4.5 Per-Agent Coordinate Descent 的不动点对应即为这一 $\varepsilon$-Nash 意义下的近似解。

**问题实例化的自由度**：RNS-MMG 元组中的三条约束确定了问题类，但不唯一确定问题实例。具体而言，$c_t$ 的转移核 $P_c$ 可以是静态（$c_t \equiv c_0$，为 static-mix 家族边界情形）、周期（$c_{t+1} = f_{\text{period}}(t)$，如正弦或方波演化）、随机漫步（$c_{t+1} = \text{clip}(c_t + \mathcal{N}(0, \sigma^2), 0, 1)$）三类主要形态，本文 §3.4 会针对这三类分别定义。类型分布 $\{\rho_\tau\}$ 也是自由参数——本文的主对比配置采用 $\rho_\alpha = \rho_\beta = 0.5$（对称），消融 3 会在 $\rho_\beta \in \{0, 0.25, 0.5, 0.75, 1.0\}$ 五点扫描以验证断言 A 的钟形曲线（§5.7）。RNS-MMG 因此不是一个单一环境，而是一个由 $(P_c, \{\rho_\tau\}, N, |\mathcal{A}_i|, \{O_i\})$ 参数化的问题族——本论文的实验以 ResourceCommons 作为该族的一个具体实例展开验证，但方法层（§4）设计不预设特定实例的细节。

---

## §3.2 初步研究：v3 NonStationaryTag

本论文方法的实证起点是作者在毕设阶段构建的 NonStationaryTag 环境与 DualHyperNetwork 原型。该工作已在毕设中期汇报中给出初步结果，本节简要复述以承接 §4 的方法设计；本节所有数字均为**毕设汇报阶段结果**，并非本文新实验的证据。

**任务设定**：4 个 agent 在 2D 连续空间中追捕；Agent 0 / Agent 1 为固定猎人，Agent 3 为猎物，Agent 2 为可变角色；规则变量 $\text{REL} \in \{0.0, 0.5, 1.0\}$ 控制 Agent 2 的阵营归属——$\text{REL} = 1$ 时 Agent 2 与猎人合作围剿猎物，$\text{REL} = 0$ 时 Agent 2 反过来作为护卫保护猎物。同一物理位置下，Agent 2 的奖励方向因 REL 切换而反向，这一零和阵营翻转构成 v3 阶段的非平稳源。

**DualHyperNetwork v3 原型**：rule_emb（由 RELEncoder 生成）与 id_emb（agent id 嵌入）联合编码为增强上下文 $C_{\text{aug}} = [\text{rule\_emb}, \text{id\_emb}]$，经 hyper_trans / hyper_rew / hyper_pred 三路超网分别生成 $(\theta_{\text{state}}, \theta_{\text{rew}}^i, \theta_{\text{pred}}^i)$。**Blocking Point 奖励正交化**（v4.3 引入）：Agent 2 在护卫模式下的奖励信号通过阻截点 $b_i = (1 - \alpha) \cdot p_{\text{hunter}_i} + \alpha \cdot p_{\text{prey}}$ 计算，距离阻截点近则得分；这一构造使"猎人模式奖励"与"护卫模式奖励"指向不同物理位置——两者不在同一物理事件下产生相反符号的奖励冲突。

**实证结果**（毕设汇报阶段）：$\text{REL} = 1.0$ 下 Infer 模型捕获率约 79.0%，对应 Baseline 模型仅约 10.1%；权重余弦相似度显示猎人-猎物对立角色的 cos_rew_0v3 从 0.37 持续下降至 0.28，cos_pred_0v3 从 0.65 下降至 0.50——这表明 DualHyperNetwork 在零和阵营翻转任务上对不同角色的奖励与策略表征实现了显著分化。

**v3 的局限与 v4 的起点**：v3 通过 Blocking Point 处理了基本对称性问题（同一物理事件不在不同角色奖励中出现相反符号），但其方法层仍存在三方面局限。**其一**，所有 agent 共享同一组 RewardHead 训练目标的"事件类型"集合；当问题进一步推广到偏好结构异质的混合动机博弈时（如 ResourceCommons），共享 RewardHead 会被类型 $\alpha$ 与类型 $\beta$ 的反向梯度直接撕裂（详见 §4.1）。**其二**，v3 的 rule_emb 由显式规则索引 REL 生成，依赖任务外部供给的离散标签；当环境上下文是连续变化的物理量（资源丰度 $c_t$）时，rule_emb 必须由 agent 在线推断而非外部读取——这一缺口正是 v4 引入 BeliefNet 的方法层动机（§4.3 / §5.10）。**其三**，v3 的 id_emb 直接编码 agent id，与"角色"和"类型"两类信息混合不清；v4 把这两层信号显式拆解为 role_i（自身私人结构）与 belief_i（关于他人类型的私人推断），形成 c_ctx + role_i + belief_i 三联通路（§4.3 / §4.4）。三个局限共同构成 v4 三个容量瓶颈的实证铺垫（§4.1）。

---

## §3.3 ResourceCommons 基准：物理层 + 偏好层二层分离

为研究 RNS-MMG 这一问题类，本文构建 **ResourceCommons** 基准环境。该环境的核心设计原则是物理层与偏好层的严格二层分离——这一分离决定了命题 3.1（§3.6）的合作-竞争切换是涌现性质而非定义内蕴。

**物理层自利同构**：每个 agent 在物理层只有一项可获得量——自身的物理采集量 $u_{i,k,t} = \min(\eta_i, q_{k,t} / |H_{k,t}|)$，其中 $\eta_i$ 是采集能力上限，$q_{k,t}$ 是 patch $k$ 在时刻 $t$ 的资源量，$|H_{k,t}|$ 是当时正在采集 patch $k$ 的 agent 数量。物理层不引入任何 reward shaping、协同加成或 gifting 机制；同 patch 多 agent 采集按头平均分配，物理上严格对称。从物理层单独看，所有 agent 都是完全自利同构的，对应 C1 在 ResourceCommons 上的具体实现。

**偏好层显式异质**：在物理采集量之上引入 agent 类型 $\tau_i \in \{\alpha, \beta\}$。类型 $\alpha$ 的总奖励 $R^\alpha_i = u_i$ 严格自利；类型 $\beta$ 的总奖励 $R^\beta_i = u_i + \phi(c_t) \cdot \psi(\Delta_i)$ 在自利项之上加入由上下文 $c_t$ 调制的 Fehr-Schmidt 不平等厌恶项，具体形式见 §3.5。类型分配在 episode 开始时确定，整个 episode 内保持不变。该设计对应 C2 在 ResourceCommons 上的具体实现：奖励函数同时由 $c_t$ 与 $\tau_i$ 联合调制。

**博弈关系涌现于交互**：合作-竞争激励通过两条正交通道涌现。**通道一**：资源时序动力学（高 $c_t$ 下 $\alpha(c)$ 大，资源再生快，留种采集的长期收益相对竭泽采集的优势缩小；低 $c_t$ 下资源稀缺，留种行为的物理回报放大）。**通道二**：类型 $\beta$ 偏好结构（$\phi(c)$ 在 $c$ 低区段为正、放大不平等厌恶，$c$ 高区段反向、鼓励向中间靠拢）。两条通道在物理上无人为耦合，命题 3.1 的 Gap(c) 单调性是真实的涌现性质。这一设计与 v3 NonStationaryTag 通过 REL 显式声明博弈关系不同——v4 让博弈关系从结构中涌现，与物理层不引入合作 shaping 这一选择直接对应。

**与现有 mixed-motive 基准的对比**：Harvest [15] 与 Cleanup [11] 通过 reward shaping 显式注入合作信号（Cleanup 中清污获得集体奖励、Harvest 中污染惩罚），物理层与博弈层在这一设计下耦合不清；Melting Pot 套件 [14] 提供多场景但博弈关系仍是 episode-static 的；Hardin [9] / Ostrom [16] 给出的公地博弈叙事在 MARL 工程实现中尚未被同时引入"博弈关系动态切换"与"偏好层类型异质"两个维度。ResourceCommons 与 Hughes 等 [11] 的不平等厌恶基准在结构上类似——均把 Fehr-Schmidt 偏好作为偏好层组件，差别在于本文将偏好强度由 $c_t$ 通过 $\phi(c_t)$ 调制，且物理层严格剥离合作 shaping。

**与 v3 NonStationaryTag 的差异**：v3 是零和（猎人 vs 猎物），通过阵营标签 REL 显式声明博弈关系；v4 是混合动机（公地资源博弈），通过类型偏好与 $c_t$ 联合调制让博弈关系从结构中涌现。v3 的 Blocking Point 处理了对称性反向问题，v4 通过两层分离从根本上避免了这一对称性问题——物理层完全对称、偏好层显式异质。

**二层分离的三条设计公理**。ResourceCommons 的物理层-偏好层分离并非任意工程选择，而是由三条底层公理导出，每条公理都排除一类可能混淆命题 3.1 涌现性的干扰路径。**公理 P1（物理对称性）**：任意 $(i, j)$ agent 对与任意物理状态 $s$，若在 $s$ 下将 agent $i$ 与 agent $j$ 的物理属性（位置、速度、能力）互换得到状态 $s'$，则对任意联合动作 $a$，物理转移分布满足 $P(\cdot | s, a) = \pi_{ij} \cdot P(\cdot | s', \pi_{ij}(a))$，其中 $\pi_{ij}$ 是 agent $i$ 与 $j$ 的置换算子。这条公理排除了"某些 agent 在物理层比其他 agent 更有优势"的可能，使得物理层不构成博弈异质性的来源。**公理 P2（偏好独立**）**：奖励函数 $R_i(s, a; c_t, \tau_i)$ 只通过 $\tau_i$ 与其他 agent 的奖励发生耦合——即 $R_i$ 的函数式不显式地包含 $R_j$（$j \ne i$）作为参数，也不包含"其他 agent 的类型 $\tau_{-i}$"作为参数。这条公理排除了"合作激励通过外部 shaping 直接注入 $R_i$"的可能，使得类型 $\beta$ 的不平等厌恶项 $\phi(c) \psi(\Delta_i)$ 是 agent 自身偏好的一部分而非环境的合作补贴（详见 §3.5）。**公理 P3（上下文正交性）**：$c_t$ 只通过 $P_c$（内禀）与 $R_i$（偏好调制）两条通道进入元组——不通过 $P$（物理转移）的函数式切换、也不通过 $O_i$（观测函数）的类型披露开关进入。这条公理排除了 $c_t$ 直接改变物理规则或观测结构的可能，将 $c_t$ 的作用范围严格限定在"资源再生强度调制 + 偏好强度调制"两个通道内。

三条公理共同保证：**任何合作行为的涌现必然可以追溯到偏好层的类型异质性 + 资源动力学在 $c$ 维度上的调制，而非环境的外部合作 shaping**。这一可追溯性是命题 3.1（Gap(c) 单调性）作为"涌现性质"而非"定义推论"的必要前提；若破坏任一公理（如 v3 阶段的 $\beta(c)$ 协同加成违反 P2），Gap(c) 单调性将退化为定义可推的恒等关系，失去可证伪价值。

**社会福利指标的意义**：本文 §3.7 主指标 $V_{\text{total}} = \sum_i R_i$ 之所以能够作为整体效能的单调标量，正是因为公理 P2 保证 $R_i$ 中的 Fehr-Schmidt 项对总和 $\sum_i \psi(\Delta_i)$ 的贡献具有零和性——$\Delta_i$ 关于 $i$ 求和时 $\sum_i \Delta_i \equiv 0$，故 $\sum_i \phi(c_t) \psi(\Delta_i) \le 0$（$\psi$ 为非正函数），$V_{\text{total}} = \sum_i u_i - |\sum_i \phi(c_t) \psi(\Delta_i)|$。这一分解使 $V_{\text{total}}$ 天然对应"物理产出减去不平等惩罚"的经济学直觉，与 Hughes 等 [11] 的"内在合作激励是负外部性的修正"论述在数学结构上一致——但本文将该修正内化为 agent 偏好而非环境 shaping，与 [11] 在方法层分离。

---

## §3.4 资源场与上下文动力学：α(c) 单通道控制

ResourceCommons 的物理状态由 $K$ 个空间分布的资源 patch（$q_{k,t}, k = 1, \dots, K$）与 $N$ 个 agent 的位置构成。**资源再生动力学**采用离散时间 logistic 增长加邻域影响项：

$$q_{k,t+1} = \text{clip}\Big( q_{k,t} - \sum_{i \in H_{k,t}} u_{i,k,t} + \alpha(c_t) \cdot f(q_{N(k),t}) \cdot (Q_{\max} - q_{k,t}), \; 0, \; Q_{\max} \Big)$$

其中 $\alpha(c_t)$ 是再生强度系数（$c_t$ 高时 $\alpha$ 大，资源恢复快；$c_t$ 低时 $\alpha$ 小，资源恢复慢），$f(q_{N(k),t})$ 是邻域影响因子（邻居 patch 富裕时该 patch 再生加快，鼓励"留种"行为），$Q_{\max}$ 是 patch 容量上限。**公平分配采集**：$u_{i,k,t} = \min(\eta_i, q_{k,t} / |H_{k,t}|)$ 保证多 agent 同时采集同一 patch 时按头平均分配，物理上严格对称。

**v3 → v4 关键演化**：v3 设计中除 $\alpha(c)$ 外还有 $\beta(c) \cdot (|H| - 1)$ 协同采集加成（$\beta(c)$ 在 $c$ 高时为正，鼓励合作采集）——这本质上是物理层的 reward shaping，把合作激励直接注入到瞬时支付层。v4 砍除 $\beta(c)$，让 $c_t$ 对物理动力学的唯一作用通道仅剩 $\alpha(c)$；合作-竞争激励的责任完全交给类型 $\beta$ 的 $\phi(c) \psi(\Delta)$ 偏好项。这一演化使 ResourceCommons 的物理层达到真正的自利同构，命题 3.1 的"两条正交通道"才能成立——若保留 $\beta(c)$，物理层会与偏好层产生耦合，Gap(c) 单调性将变成定义后的直接推论而非涌现性质。

**$c_t$ 时间演化模式**：本环境支持三种 $c_t$ 时间演化模式以对应 C3 不同表现形式。**(A) 静态模式**：episode 内 $c_t$ 恒定，多 episode 间从 $\rho_c$ 采样——用于断言 B′ 零样本泛化协议（训练 $c \in \{0.2, 0.5, 0.8\}$，测试 $c \in \{0.0, 0.35, 0.65, 1.0\}$）。**(B) 周期模式**：$c_{t+1} = 0.5 + 0.5 \cdot \sin(2\pi t / T_{\text{period}})$——用于 intra-episode 演化下的策略追踪能力测试。**(C) 随机走模式**：$c_{t+1} = \text{clip}(c_t + \mathcal{N}(0, \sigma^2), 0, 1)$——用于 C3 不可控性的极端测试，$c_t$ 演化无明显模式。三种模式共享相同的资源场动力学函数式，只在 $c_t$ 的内禀转移核 $P_c$ 上不同。

**c_visible 开关**：本环境提供 $c_{\text{visible}} \in \{\text{True}, \text{False}\}$ 配置。**可见-$c$**（默认）档下 $c_t$ 直接写入观测 $o_i^t$ 的全局块；**隐藏-$c$** 档下 $c_t$ 在全局块中被常数替换，agent 必须从资源场动力学的可观测分量（$q_{k,t}$ 与 $u_{i,k,t-1}$ 的时序）中通过 BeliefNet 推断 $\hat{c}_i$。隐藏-$c$ 档用于 §5.10 信念质量验证（断言 B′(ii) 硬指标），评估 BeliefNet 是否能从资源场动力学中提取上下文信号。

**$\alpha(c)$ 函数形式选择**：本文取 $\alpha(c) = \alpha_{\min} + (\alpha_{\max} - \alpha_{\min}) \cdot c$，即线性单调形式，$\alpha_{\min} = 0.02$ 与 $\alpha_{\max} = 0.20$（默认配置）。选择线性形式而非 sigmoid 或 quadratic 的理由有三：**(i) 单调性充分**——命题 3.1 需要 $\alpha(c)$ 关于 $c$ 单调递增，线性形式是满足该性质的最简形式，Occam's razor 要求方法层不引入额外参数；**(ii) 与 $\phi(c) = \kappa(1 - 2c)$ 的对称结构一致**——两条调制通道（$\alpha$ 通道与 $\phi$ 通道）都在 $c$ 上线性，使 Gap(c) 的两项贡献都是关于 $c$ 的低阶多项式，闭式推导（附录 A）可解；**(iii) 与 sigmoid 相比避免 $c \to 0, 1$ 边界的梯度消失**——sigmoid 形式在 $c \approx 0.5$ 附近敏感、$c \approx 0, 1$ 附近饱和，会使 §5.5 零样本泛化实验在训练点 $\{0.2, 0.5, 0.8\}$ 之外的测试点 $\{0.0, 0.35, 0.65, 1.0\}$ 上 $\alpha$ 变化太小，无法有效检验 c 外推能力。线性形式在这一点上信号更均衡。$\alpha_{\min}$ 与 $\alpha_{\max}$ 的具体取值遵循"$c = 0$ 下资源基本不再生（$\alpha_{\min}$ 小），$c = 1$ 下资源快速恢复但仍不无限（$\alpha_{\max}$ 有限）"的物理意义——若 $\alpha_{\min} = 0$ 则荒年下资源单调耗尽，破坏 episode 内部博弈的可持续性；若 $\alpha_{\max}$ 过大则丰年下博弈退化为无稀缺竞争，Gap(c) 趋近于零，命题 3.1 的可证伪性受损。

**邻域影响因子 $f$ 的构造**：$f(q_{N(k),t}) = \bar{q}_{N(k),t} / Q_{\max}$，其中 $\bar{q}_{N(k),t}$ 是 patch $k$ 的邻居 patch 存量均值（邻居定义为地图上相邻的 4 或 8 个 patch，取决于配置）。该因子在 $[0, 1]$ 区间内单调递增，$\bar{q}_{N(k),t} = 0$ 时 $f = 0$（邻居也已耗尽则本 patch 无援助再生），$\bar{q}_{N(k),t} = Q_{\max}$ 时 $f = 1$（邻居完全饱和，本 patch 获得最强的邻域再生助力）。这一空间耦合结构使采集策略在"就近连续采集"与"分散巡游"之间产生非平凡权衡——集中采集短期物理回报高但长期损耗邻域再生，分散采集短期回报低但保存了资源场的空间自恢复能力。该权衡是 §3.6 命题 3.1 中"留种行为"的空间实现，也是 §3.7 sustainability 指标的物理基础。

**$c_t$ 的初始分布**：$\rho_c$ 采取均匀分布 $\rho_c = \mathcal{U}(0, 1)$。若 $\rho_c$ 集中在 $c = 0.5$ 附近，则训练时 agent 很少见到 $c$ 极端值，$c$ 外推能力（断言 B′）无法充分训练；若 $\rho_c$ 只在 $\{0.2, 0.5, 0.8\}$ 三点上取值（如 §5.5 零样本协议所设），则训练点与测试点严格划分。默认配置的均匀分布覆盖 $c$ 全域，作为方法层设计的一般训练分布。

---

## §3.5 偏好层：类型 α / 类型 β 与 Fehr-Schmidt 结构

偏好层是 ResourceCommons 引入异质性的唯一通道。每个 agent 在 episode 开始时被指派类型 $\tau_i \in \{\alpha, \beta\}$，类型分配在 episode 内保持不变。

**类型 $\alpha$（纯自利）**：$R^\alpha_i = u_i$，即奖励完全等于物理采集量。瞬时奖励梯度 $\partial R^\alpha / \partial u_i \equiv 1$。

**类型 $\beta$（Fehr-Schmidt + $\phi$ 调制）**：$R^\beta_i = u_i + \phi(c_t) \cdot \psi(\Delta_i)$。其中 $\psi$ 是标准 Fehr-Schmidt 非对称分段函数：

$$\psi(\Delta) = -\big[ \lambda_{\text{disadv}} \cdot \max(0, -\Delta) + \lambda_{\text{adv}} \cdot \max(0, \Delta) \big]$$

参数 $\lambda_{\text{disadv}} = 2.0$ 与 $\lambda_{\text{adv}} = 0.6$ 沿用 Fehr & Schmidt [40] 经典实验估计值（不平等厌恶呈非对称形式：对劣势状态的厌恶强于对优势状态的厌恶）；$\Delta_i$ 是 instantaneous 不平等差 $\Delta_i^{(t)} = u_i^{(t)} - (1/(N-1)) \cdot \sum_{j \ne i} u_j^{(t)}$（与 Hughes 等 [11] 的 per-step 形式对应）。

**$\phi(c_t)$ 调制函数**：本文采用线性形式 $\phi(c) = \kappa \cdot (1 - 2c)$，$\kappa = 0.5$。该形式使 $c = 0$ 时 $\phi = \kappa > 0$（放大 Fehr-Schmidt：$\beta$ 极度厌恶被甩开），$c = 1$ 时 $\phi = -\kappa < 0$（反向 Fehr-Schmidt：$\beta$ 反而厌恶超过他人，鼓励向中间靠拢），$c = 0.5$ 时 $\phi = 0$（$\beta$ 与 $\alpha$ 行为相近）。这一形式使 $\phi$ 在 $c$ 轴上单调递减并以 $c = 0.5$ 为中性零点。

**心理偏好不变 / 行为模式涌现**：类型 $\beta$ 的心理偏好结构（Fehr-Schmidt + 固定 $\lambda$）在所有 $c_t$ 区段保持不变，变化的只有偏好的强度与方向（由 $\phi(c)$ 调制）。行为现象——荒年（$c$ 低）下表现为激进追赶 / 剥削、丰年（$c$ 高）下表现为协同采集 / 合作——是 RL 在固定偏好结构下找到的最优策略响应，而非对行为模式的显式编码。这一设计与 v1 阶段的"直接对合作 / 竞争做 reward shaping"路线根本不同——v1 把行为模式直接写入物理层支付，v4 把行为模式留作策略响应的涌现。

**类型梯度量化**：类型 $\alpha$ 的瞬时奖励梯度恒为 $\partial R^\alpha / \partial u_i \equiv 1$；类型 $\beta$ 的瞬时奖励梯度 $\partial R^\beta / \partial u_i = 1 + \phi(c_t) \cdot \partial \psi(\Delta_i) / \partial u_i$，由于 $\Delta_i$ 对 $u_i$ 的偏导贡献来自"自身收益减去他人平均"项，结合 $\psi$ 的非对称结构与 $\phi(c)$ 的符号，该梯度在 $(c, \Delta)$ 不同区段取值于 $\{0, 0.7, 1.3, 2.0\}$ 集合（详细推导见 §4.1.1 类型梯度量化表）。这一非对称取值集与 $\partial R^\alpha / \partial u_i \equiv 1$ 在某些区段方向相反——这正是 §4.1 共享 RewardHead 类型梯度撕裂瓶颈的代数根据。

**观测设定**：ResourceCommons 提供两档可观测性。**Self Info（默认）**：agent 自己的类型对自己可见（编码到 role_i 通路），他人的类型对自己不可见（由 BeliefNet 在线推断 $\hat{z}_{i,j}$）；这与 Harsanyi [10] 不完全信息博弈的"own type + belief over others"结构对应，是 RNS-MMG 在 ResourceCommons 上的标准设定。**Full Info（Oracle 上界对照）**：所有 agent 的类型对所有 agent 可见，作为方法性能上界，用于隔离 BeliefNet 通路的贡献。本文 §5.6 在两档下分别评测以分离 belief 通路的边际效应（断言 C 三联通路缺一不可的实证检验）。需要明确的是，本文不设"No Info"档（agent 自身类型也不可见）——这一设定下私人偏好不可识别，不构成 RNS-MMG 的合理子情形。

**类型二分的唯一性分析**：本文将类型空间限定为 $\mathcal{T} = \{\alpha, \beta\}$ 的二元集合，而非连续偏好参数空间或多元离散类型。这一选择的方法学理由如下。**(i) 与断言 A 的钟形曲线预测对应**——断言 A 预测钟形峰值发生在类型混合比例 $\rho_\beta \approx 0.5$ 的对称点或非对称点 $\rho_\beta \in \{0.25, 0.75\}$；二元类型空间使 $\rho_\beta$ 成为单一扫描维度，五点扫描（消融 3）即覆盖全曲线。若类型是多元或连续，则峰值曲面的扫描代价随类型维度指数膨胀，实证不可行。**(ii) 与 Fehr-Schmidt 二参数模型的自然对应**——类型 $\alpha$ 对应 $(\lambda_{\text{disadv}}, \lambda_{\text{adv}}) = (0, 0)$ 的退化点，类型 $\beta$ 对应经典 Fehr-Schmidt $(2.0, 0.6)$；连续插值 $\tau \in [0, 1]$ 对应 $(\lambda_{\text{disadv}}, \lambda_{\text{adv}}) = \tau \cdot (2.0, 0.6)$，但方法层将此连续变量离散为二元以突出"存在或不存在偏好异质性"这一定性区分。**(iii) 与消融 5 Fehr-Schmidt 参数扫描的解耦**——消融 5 在 $\kappa \times \lambda_{\text{disadv}}$ 网格上扫描（§5.9），保持类型二分不变，将"偏好强度扫描"与"类型比例扫描"作为独立的两条实验轴。这一解耦使实验结果的归因更清晰。

**Full Info 档的信息经济学解读**：Full Info 档下所有 agent 相互知晓类型，等价于 Harsanyi 转换后的完全信息博弈。此时最优策略不再是"关于他人类型的贝叶斯响应"，而是"给定确定类型分布下的 Nash 均衡策略"。Self Info 档与 Full Info 档的性能差可以解读为**类型不确定性的信息经济学成本**——若 BeliefNet 的类型推断精度接近 100%，则两档差距应趋近于零；若推断精度低于随机水平（50% for 2-way），则两档差距应达到方法学上界，且策略必然退化为"忽略类型信息的均衡策略"。断言 C 的 belief 通路必要性正是在这一意义下可证伪：Self Info 档下移除 belief 通路的性能应显著劣于 Full Info 档（belief 通路发挥了信息补偿作用），而 Full Info 档下移除 belief 通路的性能应基本无变化（belief 通路在 Full Info 下退化为冗余通路）。这一"双档差分"实验设计是 §5.6 消融 2 在方法论上的核心杠杆——它不仅验证 belief 通路的必要性，还量化了信息不完备下 belief 推断补偿的效率上限。

---

## §3.6 命题 3.1：合作-竞争切换的涌现性

本节给出 ResourceCommons 唯一形式化命题。完整闭式证明见附录 A；正文给出陈述与直觉证明。

**命题 3.1（Gap(c) 单调性）**：考虑二人单资源的 ResourceCommons 简化博弈（$N = 2$, $K = 1$, 离散时间, 类型分配 1$\alpha$1$\beta$），设 $V^{\text{Pareto}}(c)$ 为社会福利最大化策略对的累积奖励、$V^{\text{Nash}}(c)$ 为非合作 Nash 均衡策略对的累积奖励。定义合作优势

$$\text{Gap}(c) := V^{\text{Pareto}}(c) - V^{\text{Nash}}(c).$$

则 $\text{Gap}(c)$ 关于 $c \in [0, 1]$ 单调递减。$\square$

**直觉证明**（详见附录 A 闭式推导）：分两步说明 Gap(c) 在 $c$ 轴上的单调性来自两条独立通道的同向作用。

**(i) 资源时序动力学项**：$c$ 高时 $\alpha(c)$ 大，资源再生快，留种行为的长期收益与竭泽行为的差距缩小；极端情形 $\alpha(c) \to \infty$ 下任意采集策略都能维持资源充足，Pareto 与 Nash 策略的物理回报差距趋近于零。$c$ 低时 $\alpha(c)$ 小，资源稀缺，留种行为的长期收益显著优于竭泽，Pareto 与 Nash 的差距放大。

**(ii) Fehr-Schmidt 项**：$c$ 高时 $\phi(c) < 0$，类型 $\beta$ 的偏好结构反而厌恶超过对方，与类型 $\alpha$ 的自利行为在策略层趋于一致——非合作 Nash 均衡向 Pareto 解靠拢，差距缩小。$c$ 低时 $\phi(c) > 0$，类型 $\beta$ 极度厌恶被甩开，在非合作均衡下倾向激进追赶行为；这一行为使资源采集偏离社会福利最优分配，Pareto 与 Nash 的差距放大。

两项叠加后 $\text{Gap}(c)$ 关于 $c$ 单调递减。该命题在 $c$ 低区段（荒年）给出"合作优势大"的结论，在 $c$ 高区段（丰年）给出"合作优势趋零"的结论。

**涌现性的关键**：Gap(c) 单调性不是定义内蕴的恒等关系，而是两条正交通道（资源动力学 $\times$ 偏好结构）交互的结果。如果环境像 v3 那样在物理层引入 $\beta(c)$ 协同加成，Gap(c) 的单调形式会变成定义后的直接推论而非涌现性质——这是 §3.4 v4 砍除 $\beta(c)$ 的设计依据。**多步博弈下的推广**：完整多步博弈下命题 3.1 的闭式难以求出，本文将其作为 §5.4 ResourceCommons 主对比与 §5.7 类型扫描的实验观察项（理论预测合作优势在 $c$ 低区段最大，§5.4 / §5.7 实证待补）[证据待补 EVIDENCE PENDING — Gap(c) sweep over c ∈ {0.0, 0.35, 0.65, 1.0}，待 §5.4 补完]。

**与断言 A 钟形曲线的联系**：命题 3.1 给出 Gap(c) 在 $c$ 维度上单调递减；断言 A（§1.5）则给出钟形曲线在 $\rho_\beta$ 维度上的预测。两条预测正交但相互支撑——钟形曲线峰值位置因此预期落在"两类 agent 共存且 $c$ 平均偏低"的区段附近，但精确位置依赖学习动态，不由命题直接给出。

**多步博弈下的 $\varepsilon$-最优性**：命题 3.1 的闭式证明针对二人单资源单步博弈；完整多步博弈下 Pareto 与 Nash 均需在整个策略函数空间 $\pi: b_i^t \to \Delta(\mathcal{A}_i)$ 上比较，闭式解通常不存在。本文的立场是：多步博弈下 Gap(c) 的**定性单调性**（$c$ 低区段合作优势大、$c$ 高区段合作优势趋零）由两条通道的机制保留——资源动力学在多步下延续其"荒年惩罚竭泽"的性质，Fehr-Schmidt 偏好在多步下延续其"$c$ 高时 $\beta$ 与 $\alpha$ 行为趋同"的性质，只有**定量峰值位置与斜率**依赖多步博弈的均衡精确求解。§5.4 主对比在 Medium 配置下的经验 Gap(c) 曲线是命题 3.1 定性预测的多步实证锚定；若观察到 $c$ 低区段合作优势显著大于 $c$ 高区段（$V_{\text{total}}(c_{\text{low}}) > V_{\text{total}}(c_{\text{high}})$ 在 Pareto 策略下超过 Nash 策略的差距），则命题 3.1 的多步推广得到经验支持。

**与 CTMDP / Latent-MDP 家族的形式化比较**：Contextual MDP（Hallak 2015）设置中一个潜在任务索引 $z$ 调制 $(P, R)$，但假设单 agent 且 $z$ 在 episode 内静态；Hidden-Parameter MDP（Doshi-Velez & Konidaris 2016）与 Latent-MDP 一脉相承，同样假设潜变量按 episode 静态。RNS-MMG 与这两类的核心差别是**intra-episode 演化**——$c_t$ 按 $P_c(c_{t+1} | c_t)$ 转移，agent 必须在线感知其变化并据此调整策略。这一 intra-episode 演化使得 RNS-MMG 的最优策略必须是**上下文条件化的历史泛函** $\pi_i^*: (b_i^t, \hat{c}_t) \to \Delta(\mathcal{A}_i)$，而非 CTMDP 意义下的"给定 $z$ 后的静态最优策略族"。这一形式化差别在方法层直接对应：§4.4 中 hyper_trans 消费 $c_{\text{ctx}}$ 生成 $\theta_{\text{state}}$ 的机制，即是"按当前 $\hat{c}_t$ 动态生成功能网权重"的实现——若 $c_t$ 是 episode-static 的，则这一动态权重生成退化为单次初始化。

---

## §3.7 评测指标

ResourceCommons 评测指标分四大类，每类与一项可证伪断言或诊断目的对应。

**(1) 社会福利与个体公平**：$V_{\text{total}} = \sum_i R_i$ 衡量整体效用（断言 A 钟形曲线的主指标）；个体福利的 Gini 系数与累积分布函数（CDF）评估分配公平性。Gini 系数在 $[0, 1]$ 区间，0 对应完全均等、1 对应完全集中——类型 $\beta$ 的 Fehr-Schmidt 偏好在 $c$ 低区段会主动拉低 Gini，这一性质在 §5.7 类型扫描中作为偏好层有效性的间接证据。

**(2) 可持续性**：episode 末资源总存量 $\sum_k q_{k,T}$（与 Hardin [9] 公地悲剧叙事直接对照——存量趋零即对应悲剧情形）；采集量的时序方差（衡量"竭泽"程度，方差大表示集中爆发式采集）。**关于 tragedy 指标的说明**：在 Easy 难度（$N = 2$, $K = 1$ 与 $K = 2$ 简化配置）下，资源场容量与 agent 采集能力的比例使资源耗竭情形几乎不出现——18 个 suite 运行的初步观察显示 tragedy 指标接近恒等于 0，不构成 informative 的诊断量。Easy 难度下的主要诊断指标因此为社会福利、fairness（Gini）与 sustainability（末态存量）三项；tragedy 指标的诊断价值在 Medium / Hard 难度（更高 $N$ 与更紧的资源 / 容量比）下才会显现，相关协议在 §5.1.4 给出。

**(3) RNS-MMG 特有指标**：本节给出三项面向 RNS-MMG 设定的专属指标。**(i) 零样本泛化**——训练 $c \in \{0.2, 0.5, 0.8\}$，测试 $c \in \{0.0, 0.35, 0.65, 1.0\}$ 上的福利保留率，对应断言 B′(i) 的硬指标；$c_{0.2}$ / $c_{0.5}$ / $c_{0.8}$ 三个训练点是 Gap(c) sweep 在条件化谱实验中的数据采样点（§5.5）。**(ii) 隐藏-$c$ 信念质量**——BeliefNet 推断的 $\hat{c}_i$ 与真实 $c_t$ 的 MSE 与 Pearson 相关系数，对应断言 B′(ii) 的硬指标；此指标只在 $c_{\text{visible}} = \text{False}$ 档下可评估。**(iii) 对手类型推断准确率**——$\hat{z}_{i,j}$ 与真实 $\tau_j$ 的 2-way 分类准确率，作为 §5.6 课程有效性的中间指标（断言 C belief 通路边际效应的内部诊断）。

**(4) 类型梯度对齐**：从训练完的模型中数值估计学习到的 $\partial \hat{r} / \partial u_i$，与理论值（类型 $\alpha$ 恒为 1；类型 $\beta$ 在 $(c, \Delta)$ 区段表中取 $\{0, 0.7, 1.3, 2.0\}$）的 $L_2$ 距离。该指标用于 §5.7 类型扫描与 §6.11 可视化：Hyper-MuZero 期望每条 type 通路贴近对应理论值；共享 RewardHead baseline（MA-MuZero）期望落入两类型之间的"折中区"——这正是类型梯度撕裂在数值层的可视化证据。

**指标分级与统计协议**：主对比（§5.4）报告社会福利 + 可持续性 + Gini 三主指标；零样本与信念质量作为 §5.5 / §5.10 的章节专属指标；类型梯度对齐作为 §5.7 的可视化补强指标。所有指标的统计协议：5 seeds $\times$ 完整训练曲线，均值与标准差以蛇形曲线报告；关键对比（断言 A / B′ / C / D 的预登记阈值）使用 Welch's t-test 做显著性检验，显著性水平 $p < 0.05$，效应量阈值在 §5.1.4 与 §1.5 表中明示。所有指标的具体计算实现与日志键见附录 E 复现指南。

**指标与断言的一一对应**：本节所列四大类指标与四项可证伪断言在实验设计上呈精确匹配——每项断言由 2–3 个主指标 + 若干辅助诊断量支撑，且预登记阈值明示。**(a) 断言 A（类型梯度撕裂钟形曲线）**：主指标为 $V_{\text{total}}$ 在 $\rho_\beta$ 五点扫描下的曲线形态，判据为峰值 $\rho_\beta^* \in \{0.25, 0.5, 0.75\}$ 且峰值差 $\ge 5\%$；辅助诊断为类型梯度对齐指标（§5.7 / §6.11 的 $\partial \hat{r} / \partial u_i$ 数值梯度热图）。**(b) 断言 B′（条件化谱内部最优）**：分三子项——B′(i) 内部峰值存在（主指标：$V_{\text{total}}$ 在 gen_scope 谱上的非单调曲线，中间峰值高于两端）；B′(ii) 隐藏-$c$ 下 belief 通路必要性（主指标：$\hat{c}$ 的 Pearson 相关系数 $\ge 0.7$）；B′(iii) 零样本 c 外推能力（主指标：未见 $c$ 上福利保留率 $\ge 80\%$）。**(c) 断言 C（三联通路缺一不可）**：主指标为 5 组零置条件与 all-on 条件的 $V_{\text{total}}$ 差异，判据为任一零置退化 $\ge 10\%$ 且多路联合零置退化近似可加；辅助为 Full Info vs Self Info 双档差分（§3.5 已展开）。**(d) 断言 D（CoordDesc × CRN 不可分割）**：主指标为 Easy N=2 上 2×2 矩阵四象限的 $V_{\text{total}}$ 差异，判据为对角线（both on / both off）与非对角线（单一开启）的性能序 $\text{both on} > \text{single on} > \text{both off}$；辅助为 SNR 实测（0.02 → 1.0）。

**指标的鲁棒性保护**：所有指标在 5 seeds 上聚合前先在同一 seed 内做**训练末段均值**（末 10% 的评估窗口，与 scalar tail-mean 一致）以抑制单一评估点的高频噪声；跨 seed 聚合采用**中位数 + 四分位距（IQR）**辅助均值 ± 标准差报告，用于识别单一 seed 显著偏离的情形。若某一指标在 5 seeds 上的 IQR / median 比例超过 30%，则视为该指标在当前配置下不稳定，需扩展至 8 seeds 或改进训练协议（§5.11 已有初步观察：`lora/duo_film` seed 1 与 seed 0 的 q_gap 分歧即触发这一诊断阈值）。这一双报告协议是本文对"5 seeds 是否足够"这一评委常见质疑的方法学回应。

---

**本章小结**：第 3 章形式化了本论文研究对象 RNS-MMG——一个由物理转移 agent-同构性（C1）、博弈关系类型-上下文联合调制（C2）、上下文物理不可控性（C3）三条约束刻画的多智能体博弈子类——并构建了 ResourceCommons 基准以承载其实证研究。ResourceCommons 的物理层-偏好层严格分离设计（§3.3 三条底层公理）保证了合作-竞争切换是涌现性质而非定义推论（§3.6 命题 3.1）；资源场 $\alpha(c)$ 单通道调制与偏好层 $\phi(c) \psi(\Delta)$ 双通道结构（§3.4–§3.5）共同支撑命题 3.1 的两项独立贡献；四类评测指标（§3.7）与四项可证伪断言一一对应。§3.2 复述的 v3 NonStationaryTag 阶段结果作为本文方法的实证起点，其三项方法层局限（共享 RewardHead 撕裂、rule_emb 依赖外部标签、id_emb 混杂角色与类型）在结构上直接指向 v4 方法的三个容量瓶颈——即第 4 章将展开的 DualHyperNetwork v2 三联通路设计动机。

[证据待补 EVIDENCE PENDING — ResourceCommons 18 suite 运行的完整指标日志（含 Gini、末态资源存量、类型梯度对齐数值），待 §5.4 主对比 + §5.7 类型扫描补完]

---
