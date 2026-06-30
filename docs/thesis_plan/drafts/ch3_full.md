# 第 3 章 问题与环境

> 本章共分七节。§3.1 在第 1.2 节非形式描述基础上给出关系非平稳混合动机博弈（RNS-MMG）的形式化元组与三条结构性约束（C1/C2/C3）。§3.2 简要复述 v3 NonStationaryTag 阶段的环境设定、DualHyperNetwork 原型与毕设汇报阶段的实证结果，作为本文方法的实证起点。§3.3 给出 ResourceCommons 基准的核心设计原则——物理层与偏好层的二层分离。§3.4 进一步给出资源场动力学与上下文 c_t 通过单通道再生强度 α(c) 调控的具体形式。§3.5 给出偏好层的类型 α / 类型 β 与 Fehr-Schmidt 结构，以及 Self / Full 两档观测设定。§3.6 给出 ResourceCommons 唯一形式化命题——命题 3.1（Gap(c) 单调性），以"正文陈述 + 附录证明"方式呈现。§3.7 给出评测指标体系并标注与各断言的对应关系。

---

## §3.1 RNS-MMG 形式化

第 1.2 节给出 RNS-MMG 的非形式描述：博弈关系由可观测物理状态 c_t 动态调控。本节将其形式化为 N 个 agent 的元组 $G = \langle \mathcal{N}, \mathcal{S}, \{\mathcal{A}_i\}, P, \{R_i\}, \{O_i\}, \mathcal{C}, P_c, \rho_c, \gamma \rangle$。其中 $\mathcal{N} = \{1, \dots, N\}$ 是 agent 集合；$\mathcal{S}$ 是全局状态空间；$\mathcal{A}_i$ 是 agent $i$ 的动作空间，联合动作记 $a = (a_1, \dots, a_N) \in \prod_i \mathcal{A}_i$；$P: \mathcal{S} \times \prod_i \mathcal{A}_i \to \Delta(\mathcal{S})$ 是物理转移核；$R_i: \mathcal{S} \times \prod_i \mathcal{A}_i \to \mathbb{R}$ 是 agent $i$ 的瞬时奖励函数；$O_i$ 是 agent $i$ 的部分观测函数；$\mathcal{C} \subseteq \mathbb{R}$ 是共享上下文空间，$P_c$ 与 $\rho_c$ 分别是 $c_t$ 的内禀转移核与初始分布；$\gamma \in (0, 1)$ 是折扣因子。在该元组之上施加三条结构性约束。

**约束 C1（物理转移的 agent-同构性与上下文内禀演化性）**：转移函数 $P(s' \mid s, a)$ 在所有 agent 间共享（agent-invariant），其函数式由环境本身固定，不随 $c_t$ 切换为不同的函数形式；$c_t$ 作为状态的一部分按内禀动力学 $P_c$ 演化，并通过资源场动力学 $q_{k,t+1} = f(q_{k,t}; \alpha(c_t))$ 间接调控博弈结构。此处需精确区分两件事——$c_t$ 是状态的分量（作为状态进入 $P$ 与 $\{R_i\}$），但物理转移 $P$ 本身的函数式不因 $c_t$ 值而切换；这一区分使 C1 与"上下文索引转移规则切换"（Contextual MDP / HiP-MDP）划清边界。

**约束 C2（博弈关系的类型-上下文联合调控）**：奖励函数 $R_i(s, a; c_t, \tau_i)$ 由 $c_t$ 与 agent 类型 $\tau_i \in \{\alpha, \beta\}$ 联合调制。本文在 §3.5 给出一个具体的偏好实例化（基于 Fehr-Schmidt 不平等厌恶 [40] 并由 $c_t$ 进行强度调制），其中类型异质 agent 在某些 $c_t$ 区段呈现结构性相反的瞬时奖励梯度方向——例如类型 $\alpha$ 的 $\partial R^\alpha / \partial u_i \equiv 1$，类型 $\beta$ 在 $c$ 低区段对自身相对优势状态表现出反向梯度（$\partial R^\beta / \partial u_i < 1$ 甚至 $\le 0$）。"梯度相反"是 §3.5 建模选择下的可检验性质，C2 本身只要求类型与上下文联合非平凡地调制偏好。

**约束 C3（上下文物理不可控性）**：$c_t$ 由环境内禀机制（资源场再生 + 外部驱动）演化，agent 动作对其只有间接或缓慢影响。该约束将 $c_t$ 与"可由 agent 直接操控的环境变量"区分开，避免偏好层因果归因被 agent 动作-上下文耦合干扰；同时使 RNS-MMG 与 Harsanyi [10] 不完全信息博弈框架在结构上相容——类型 $\tau_i$ 属于 agent 私人信息，$c_t$ 属于 episode 内可观测或可推断的共同知识层。

RNS-MMG 与几类相邻问题的边界已在 §1.2 给出，此处仅作要点复述：与一般非平稳 MDP / Markov Game [20][21] 相比，非平稳源自一个可被观测或推断的物理变量而非显式时间索引；与静态 mixed-motive（Cleanup, Harvest）[15][11] 相比，博弈关系允许随 $c_t$ 动态切换；与领域随机化相比，$c_t$ 在 episode 内有明确的内禀演化规律。RNS-MMG 概念上可记作"Contextual MARL + C2 类型异质 + C3 内禀不可控 + intra-episode 演化"四要素的合取——任一要素单独存在均落入已有家族。

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

---

## §3.4 资源场与上下文动力学：α(c) 单通道控制

ResourceCommons 的物理状态由 $K$ 个空间分布的资源 patch（$q_{k,t}, k = 1, \dots, K$）与 $N$ 个 agent 的位置构成。**资源再生动力学**采用离散时间 logistic 增长加邻域影响项：

$$q_{k,t+1} = \text{clip}\Big( q_{k,t} - \sum_{i \in H_{k,t}} u_{i,k,t} + \alpha(c_t) \cdot f(q_{N(k),t}) \cdot (Q_{\max} - q_{k,t}), \; 0, \; Q_{\max} \Big)$$

其中 $\alpha(c_t)$ 是再生强度系数（$c_t$ 高时 $\alpha$ 大，资源恢复快；$c_t$ 低时 $\alpha$ 小，资源恢复慢），$f(q_{N(k),t})$ 是邻域影响因子（邻居 patch 富裕时该 patch 再生加快，鼓励"留种"行为），$Q_{\max}$ 是 patch 容量上限。**公平分配采集**：$u_{i,k,t} = \min(\eta_i, q_{k,t} / |H_{k,t}|)$ 保证多 agent 同时采集同一 patch 时按头平均分配，物理上严格对称。

**v3 → v4 关键演化**：v3 设计中除 $\alpha(c)$ 外还有 $\beta(c) \cdot (|H| - 1)$ 协同采集加成（$\beta(c)$ 在 $c$ 高时为正，鼓励合作采集）——这本质上是物理层的 reward shaping，把合作激励直接注入到瞬时支付层。v4 砍除 $\beta(c)$，让 $c_t$ 对物理动力学的唯一作用通道仅剩 $\alpha(c)$；合作-竞争激励的责任完全交给类型 $\beta$ 的 $\phi(c) \psi(\Delta)$ 偏好项。这一演化使 ResourceCommons 的物理层达到真正的自利同构，命题 3.1 的"两条正交通道"才能成立——若保留 $\beta(c)$，物理层会与偏好层产生耦合，Gap(c) 单调性将变成定义后的直接推论而非涌现性质。

**$c_t$ 时间演化模式**：本环境支持三种 $c_t$ 时间演化模式以对应 C3 不同表现形式。**(A) 静态模式**：episode 内 $c_t$ 恒定，多 episode 间从 $\rho_c$ 采样——用于断言 B′ 零样本泛化协议（训练 $c \in \{0.2, 0.5, 0.8\}$，测试 $c \in \{0.0, 0.35, 0.65, 1.0\}$）。**(B) 周期模式**：$c_{t+1} = 0.5 + 0.5 \cdot \sin(2\pi t / T_{\text{period}})$——用于 intra-episode 演化下的策略追踪能力测试。**(C) 随机走模式**：$c_{t+1} = \text{clip}(c_t + \mathcal{N}(0, \sigma^2), 0, 1)$——用于 C3 不可控性的极端测试，$c_t$ 演化无明显模式。三种模式共享相同的资源场动力学函数式，只在 $c_t$ 的内禀转移核 $P_c$ 上不同。

**c_visible 开关**：本环境提供 $c_{\text{visible}} \in \{\text{True}, \text{False}\}$ 配置。**可见-$c$**（默认）档下 $c_t$ 直接写入观测 $o_i^t$ 的全局块；**隐藏-$c$** 档下 $c_t$ 在全局块中被常数替换，agent 必须从资源场动力学的可观测分量（$q_{k,t}$ 与 $u_{i,k,t-1}$ 的时序）中通过 BeliefNet 推断 $\hat{c}_i$。隐藏-$c$ 档用于 §5.10 信念质量验证（断言 B′(ii) 硬指标），评估 BeliefNet 是否能从资源场动力学中提取上下文信号。

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

---

## §3.7 评测指标

ResourceCommons 评测指标分四大类，每类与一项可证伪断言或诊断目的对应。

**(1) 社会福利与个体公平**：$V_{\text{total}} = \sum_i R_i$ 衡量整体效用（断言 A 钟形曲线的主指标）；个体福利的 Gini 系数与累积分布函数（CDF）评估分配公平性。Gini 系数在 $[0, 1]$ 区间，0 对应完全均等、1 对应完全集中——类型 $\beta$ 的 Fehr-Schmidt 偏好在 $c$ 低区段会主动拉低 Gini，这一性质在 §5.7 类型扫描中作为偏好层有效性的间接证据。

**(2) 可持续性**：episode 末资源总存量 $\sum_k q_{k,T}$（与 Hardin [9] 公地悲剧叙事直接对照——存量趋零即对应悲剧情形）；采集量的时序方差（衡量"竭泽"程度，方差大表示集中爆发式采集）。**关于 tragedy 指标的说明**：在 Easy 难度（$N = 2$, $K = 1$ 与 $K = 2$ 简化配置）下，资源场容量与 agent 采集能力的比例使资源耗竭情形几乎不出现——18 个 suite 运行的初步观察显示 tragedy 指标接近恒等于 0，不构成 informative 的诊断量。Easy 难度下的主要诊断指标因此为社会福利、fairness（Gini）与 sustainability（末态存量）三项；tragedy 指标的诊断价值在 Medium / Hard 难度（更高 $N$ 与更紧的资源 / 容量比）下才会显现，相关协议在 §5.1.4 给出。

**(3) RNS-MMG 特有指标**：本节给出三项面向 RNS-MMG 设定的专属指标。**(i) 零样本泛化**——训练 $c \in \{0.2, 0.5, 0.8\}$，测试 $c \in \{0.0, 0.35, 0.65, 1.0\}$ 上的福利保留率，对应断言 B′(i) 的硬指标；$c_{0.2}$ / $c_{0.5}$ / $c_{0.8}$ 三个训练点是 Gap(c) sweep 在条件化谱实验中的数据采样点（§5.5）。**(ii) 隐藏-$c$ 信念质量**——BeliefNet 推断的 $\hat{c}_i$ 与真实 $c_t$ 的 MSE 与 Pearson 相关系数，对应断言 B′(ii) 的硬指标；此指标只在 $c_{\text{visible}} = \text{False}$ 档下可评估。**(iii) 对手类型推断准确率**——$\hat{z}_{i,j}$ 与真实 $\tau_j$ 的 2-way 分类准确率，作为 §5.6 课程有效性的中间指标（断言 C belief 通路边际效应的内部诊断）。

**(4) 类型梯度对齐**：从训练完的模型中数值估计学习到的 $\partial \hat{r} / \partial u_i$，与理论值（类型 $\alpha$ 恒为 1；类型 $\beta$ 在 $(c, \Delta)$ 区段表中取 $\{0, 0.7, 1.3, 2.0\}$）的 $L_2$ 距离。该指标用于 §5.7 类型扫描与 §6.11 可视化：Hyper-MuZero 期望每条 type 通路贴近对应理论值；共享 RewardHead baseline（MA-MuZero）期望落入两类型之间的"折中区"——这正是类型梯度撕裂在数值层的可视化证据。

**指标分级与统计协议**：主对比（§5.4）报告社会福利 + 可持续性 + Gini 三主指标；零样本与信念质量作为 §5.5 / §5.10 的章节专属指标；类型梯度对齐作为 §5.7 的可视化补强指标。所有指标的统计协议：5 seeds $\times$ 完整训练曲线，均值与标准差以蛇形曲线报告；关键对比（断言 A / B′ / C / D 的预登记阈值）使用 Welch's t-test 做显著性检验，显著性水平 $p < 0.05$，效应量阈值在 §5.1.4 与 §1.5 表中明示。所有指标的具体计算实现与日志键见附录 E 复现指南。

[证据待补 EVIDENCE PENDING — ResourceCommons 18 suite 运行的完整指标日志（含 Gini、末态资源存量、类型梯度对齐数值），待 §5.4 主对比 + §5.7 类型扫描补完]

---
