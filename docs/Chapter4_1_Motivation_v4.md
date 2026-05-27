# 4.1 设计原则:从类型梯度撕裂到容量分配的几何论证(v4 最终版)

> **本节定位**:本节是第四章的"motivation 锚",对应 Chapter 1.5 节贡献 2 的三个可证伪断言(断言 A:类型梯度撕裂、断言 B:信念专属容量、断言 C:三联通路必要性)。本节的论证结构严格服务于这三个断言的提出与解释,并为 4.2 节起的具体架构展开提供理论根基。

---

## 4.1.1 RNS-MMG 下的核心架构挑战

在第一章中,我们将 RNS-MMG 的核心结构特性总结为三条约束:物理转移上下文不变 (C1)、博弈关系上下文-类型联合调控 (C2)、上下文物理不可控 (C3)。在第三章 ResourceCommons 环境的具体实例化下,(C2) 进一步细化为:类型 α(纯自利)与类型 β(Fehr-Schmidt 不平等厌恶 + φ 调制)共存,类型 β 的奖励函数包含由 $c$ 调制的不平等厌恶项 $\phi(c)\psi(\Delta_i)$,使得两种类型在不同 $c$ 区段呈现结构性相反的梯度方向。

### 类型 α 与类型 β 的奖励梯度对照

两种类型的奖励函数显式写出:

$$R_i^\alpha(s, a) = u_i \quad \text{(纯自利)}$$

$$R_i^\beta(s, a) = u_i + \phi(c_t) \cdot \psi(\Delta_i),\quad \phi(c) = \kappa(1-2c),\quad \psi(\Delta) = -[\lambda_{\text{disadv}} \max(0,-\Delta) + \lambda_{\text{adv}} \max(0,\Delta)]$$

其中 $\Delta_i = u_i - \frac{1}{N-1}\sum_{j \neq i} u_j$,$\lambda_{\text{disadv}} = 2.0$,$\lambda_{\text{adv}} = 0.6$(Fehr-Schmidt 1999 经典值)。

计算对 $u_i$ 的偏导(暂时忽略 $\Delta_i$ 对 $u_i$ 的间接依赖以观察 leading-order 行为):

$$\frac{\partial R^\alpha}{\partial u_i} = 1 \quad \text{(常数,所有 } c \text{ 区段均为正)}$$

$$\frac{\partial R^\beta}{\partial u_i} \approx 1 + \phi(c) \cdot \psi'(\Delta_i) \cdot 1 = 1 + \phi(c) \cdot \begin{cases} -\lambda_{\text{adv}} & \Delta_i > 0 \\ +\lambda_{\text{disadv}} & \Delta_i < 0 \end{cases}$$

代入具体值,考察四种典型场景:

| 场景 | $\phi(c)$ | $\Delta_i$ 符号 | $\partial R^\beta/\partial u_i$ | $\partial R^\alpha/\partial u_i$ | 梯度方向 |
|---|---|---|---|---|---|
| 荒年 + 优势 | $+\kappa = +0.5$ | $+$ | $1 + 0.5 \times (-0.6) = 0.7$ | $1$ | 同号但 β 较弱 |
| 荒年 + 劣势 | $+\kappa = +0.5$ | $-$ | $1 + 0.5 \times (+2.0) = 2.0$ | $1$ | 同号但 β 极强 |
| 丰年 + 优势 | $-\kappa = -0.5$ | $+$ | $1 + (-0.5) \times (-0.6) = 1.3$ | $1$ | 同号但 β 较强 |
| 丰年 + 劣势 | $-\kappa = -0.5$ | $-$ | $1 + (-0.5) \times (+2.0) = 0$ | $1$ | **β 完全无激励** |

关键观察是**第四个场景**:**丰年 + 自己处于劣势时,类型 α 的梯度 $\partial R^\alpha/\partial u_i = 1$ 仍然激励 agent 增加自身采集,但类型 β 的梯度 $\partial R^\beta/\partial u_i = 0$ —— 完全无激励**。直观解释:类型 β 在丰年对自己落后是开心的(反 inequity aversion,$\phi < 0$ 让"被甩开"变成正效用),这种偏好驱动 β 优先去帮助他人采集而非自己采集,行为上表现为协同。

而在**荒年 + 自己处于劣势**的场景中,类型 β 的梯度 $\partial R^\beta/\partial u_i = 2.0$ 是类型 α 梯度的 2 倍 —— **β 在荒年极度厌恶被甩开,产生异常强的"追赶"激励**,行为上表现为剥削/激进追赶。

### 共享 RewardHead 的容量瓶颈

考虑一个标准的共享 RewardHead(以 (s, a) 为输入,可能附加 agent ID 为额外特征)。在 ResourceCommons 中训练时,该共享架构必然面临以下三种**容量瓶颈**:

**瓶颈 1(类型梯度撕裂)**。 同一组 RewardHead 权重 $W$ 需要同时拟合 $R^\alpha$ 和 $R^\beta$ 两个函数。在荒年 + 劣势场景下,类型 α 的 ground-truth 梯度推动 $W$ 学习"$\partial R/\partial u_i = 1$",类型 β 的 ground-truth 梯度推动 $W$ 学习"$\partial R/\partial u_i = 2$";在丰年 + 劣势场景下,类型 α 仍是 1,类型 β 变成 0。**单组 $W$ 无法同时满足这四种 (c 区段, Δ 符号, type) 组合的预测要求**,在反向传播中被两类梯度拉扯,最终收敛到一组对两种类型都"次优"的折中权重,产出**类型平均策略**——既不能完美刻画类型 α 的纯自利倾向,也不能完美刻画类型 β 在荒年的激进追赶与丰年的协同合作。

PredictionNet 面临相同问题:同一物理状态 $s$ 下,类型 α 的最优策略 $\pi^*_\alpha(s)$ 在 c 低时是"无差别贪婪采集",类型 β 的最优策略 $\pi^*_\beta(s)$ 在 c 低时是"针对劣势状态的激进追赶"——两种策略在动作维度上的概率分布形状不同,共享 PredictionNet 同样面临撕裂。

**瓶颈 2(信念稀释)**。 即使两个 agent 同为类型 β,如果它们的私人信念不同——比如 agent $i$ 通过历史观测推断当前是丰年($\hat{c}_i \approx 0.8$),而 agent $j$ 推断当前是荒年($\hat{c}_j \approx 0.2$)——那么它们的 $\phi(\hat{c})\psi(\Delta)$ 计算结果截然不同,最优策略也截然不同(i 偏好协同,j 偏好激进追赶)。

如果将 belief_i 作为输入特征拼接到共享 PredictionNet 的输入端(即"context as input"方案),belief 信号要穿过若干个共享非线性层才能影响最终输出。在反向传播中,**所有 agent 的所有 belief 取值都在使用同一组权重**,网络容量被强制在 belief 空间上做平均分配——一个 belief 维度的细微变化能在多大程度上改变输出,取决于共享权重的局部敏感性,而这个敏感性是"对所有 belief 都尽可能好"的折中产物。

**瓶颈 3(角色平均化)**。 与瓶颈 2 类似,异质能力向量 $\mathbf{cap}_i$(采集速度、视野、移动速度)使最优策略在 agent 间天然分化——快采集者激进、宽视野者勘探。共享 PredictionNet 同样无法为每种 $\mathbf{cap}$ 组合分配专属容量,只能在 cap 空间上做平均。

---

## 4.1.2 hypernetwork:从输入条件化到容量条件化

针对上述三类容量瓶颈,本文采取的核心架构选择是**用 hypernetwork 替代 input conditioning**。这一选择背后的几何论证如下。

考虑两种条件化方案:

**方案 I(Input Conditioning)**:context = (c_ctx, role_i, belief_i) 作为额外输入特征拼接到 RewardHead/PredictionNet 的输入端,网络权重 $W$ **共享**于所有 agent 所有 context:
$$\hat{r}_i = f_W([s, a, c_{\text{ctx}}, \text{role}_i, \text{belief}_i])$$

**方案 II(Hypernetwork Conditioning)**:context 通过 hypernetwork 生成专属权重 $\theta_i^{\text{rew}}$,功能网络以 $(s, a)$ 为输入、$\theta_i^{\text{rew}}$ 为权重:
$$\theta_i^{\text{rew}} = h_{\Theta}(c_{\text{ctx}}, \text{role}_i, \text{belief}_i),\qquad \hat{r}_i = f_{\theta_i^{\text{rew}}}(s, a)$$

### 理论表达能力对比

两者在足够大的网络宽度下等价——任何 $f_W([s,a,c])$ 都可以被 $f_{h(c)}(s,a)$ 模拟,反之亦然。这是 universal approximation 的标准结论。

### 实际学习行为的差异:容量分配的几何

两者的差异不在表达能力,而在**容量分配的几何**。

- **方案 I 下**:对每个特定的 context 值,$f_W$ 使用的是同一组 $W$。如果两个不同 context $c^{(1)}, c^{(2)}$ 对应的最优函数 $f^*(\cdot; c^{(1)})$ 与 $f^*(\cdot; c^{(2)})$ 差异很大,$W$ 必须同时编码这两种行为模式——网络容量在 context 空间上**平均分配**。当 context 空间维度高(本文为 $d_c + d_r + d_b \approx 80$ 维)、或不同 context 对应的最优函数差异大(本文中类型 α 与类型 β 的 RewardHead 在 4.1.1 节四种场景下偏导值从 0 到 2 跨度极大)时,共享 $W$ 的容量瓶颈成为主导误差源。

- **方案 II 下**:每个 context 值生成专属的 $\theta(c)$,功能网络在 $\theta(c)$ 下编码的就是"在该 context 下的最优函数",**容量分配是 per-context 的**。hypernetwork $h_\Theta$ 的任务从"学习一个能同时表达所有 context 行为的大网络"降级为"学习一个 context → 参数 的映射"——后者通常是低维 context 上的光滑函数,可以用相对小的 $\Theta$ 高效学习。

这一区别在 HyperNetworks (Ha et al., 2017) [42]、CAVIA (Zintgraf et al., 2019) [43]、FiLM (Perez et al., 2018) [44] 等元学习与条件计算文献中已被反复观察:**当不同任务/context 对应的最优函数结构性差异大时,hypernetwork 的样本效率与泛化能力显著优于 input conditioning,即使两者参数总量相同**。

**本文的 (类型, 信念, 能力) 联合 context 正是这种"结构性差异大"的典型场景**——4.1.1 节的偏导表格直接展示了类型 α 与类型 β 的 RewardHead 在不同场景下偏导值从 0 到 2 的剧烈变化,belief 不同导致同类型 agent 的最优响应完全不同,cap 不同决定策略空间的分化。三个维度的差异叠加,使方案 I 的容量瓶颈在 ResourceCommons 中暴露为可观测的性能差距。

---

## 4.1.3 三层 Motivation 的结构性归并

基于 4.1.2 的容量分配论证,本文 DualHyperNetwork 的设计 motivation 形成清晰的三层结构,**每一层对应一种容量瓶颈、对应一类 context 通路、对应一个可证伪断言**:

| 层级 | 容量瓶颈 | Context 通路 | 对应断言(贡献 2) |
|---|---|---|---|
| **主层** | 瓶颈 1:类型梯度撕裂 | role_i 中的 type_emb | 断言 A(类型异质性扫描钟形曲线) |
| **辅层 1** | 瓶颈 2:信念稀释 | belief_i 中的 $\hat{c}$ 与 $\hat{z}$ | 断言 B(未见 c 值零样本泛化优势) |
| **辅层 2** | 瓶颈 3:角色平均化 | role_i 中的 cap_emb | 断言 C 的一个分量(去除 cap 通路) |

**主层**:类型异质导致的奖励梯度撕裂是 RNS-MMG 在架构层面最尖锐的挑战,也是本文 DualHyperNetwork 的核心动机。这一动机的承诺是:当类型 α 与类型 β 共存时(类型混合比例接近 50/50),per-type θ_rew 显著优于共享 RewardHead;当类型同质时(0% 或 100% β),两者趋于一致。第六章消融 3 通过扫描类型 β 比例 $\in \{0, 1/4, 2/4, 3/4, 4/4\}$ 加上 1α+3β 对照点来检验这一钟形曲线断言。**注**:若实际曲线峰值偏离 50/50(例如类型 β 自身学习难度较大、曲线偏向 α-多侧),这本身是 informative 结果,论文中将作为正向发现报告。

**辅层 1**:信念稀释是 RNS-MMG 在 Self Info 设定下的内在挑战——agent 必须从历史推断他人类型与当前 $c_t$,推断结果不同导致最优策略不同。hypernetwork 通过 belief_i 输入为每种"信念组合"分配专属容量,使同类型不同信念的 agent 也能产生差异化策略。该层的承诺由零样本泛化实验(训练 $c \in \{0.2, 0.5, 0.8\}$,测试 $c \in \{0.0, 0.35, 0.65, 1.0\}$)直接检验。

**辅层 2**:角色平均化是异质能力 $\mathbf{cap}_i$ 引入的挑战。即便同类型同信念,$\eta_i, \phi_i, \nu_i, \zeta_i$ 不同的 agent 最优策略也不同。该层的承诺由消融 2 中"去除 cap_emb"的变体检验。

---

## 4.1.4 Harsanyi 不完全信息博弈的几何对应

上述三层 motivation 与 Harsanyi (1967) [10] 不完全信息博弈的概念框架存在一一对应:

| Harsanyi 框架 | 本文实现 | 架构组件 |
|---|---|---|
| Common Knowledge(物理规律) | 客观通路 hyper_trans | 输入 = c_ctx,输出 = θ_state(所有 agent 共享) |
| Own Type(自己的类型/能力) | role_i 通路 | 输入 = type_emb + cap_emb + id_emb,先验固定(Self Info) |
| Beliefs over Others' Types | belief_i 通路 | 输入 = $\hat{c}_i^t$ + $\text{Pool}(\hat{z}_{i,j}^t)$,后验推断 |
| Type-Conditional Best Response | 主观通路 hyper_rew / hyper_pred | 接收完整三联输入,生成 per-agent θ |

Harsanyi 框架的核心洞察是:任何关于环境与他人的不确定性,都可以归约为对**类型**的不确定性。在标准博弈论中这一框架是描述性的——它告诉你"博弈应当被这样建模";本文 DualHyperNetwork 的贡献是将其转化为构造性的——它告诉你**深度世界模型应当被这样设计**。

具体地,共同知识对应"所有 agent 同意的物理事实",在架构上表现为 hyper_trans 仅以 c_ctx 为输入、生成所有 agent 共享的 θ_state;自己的类型对应"先验固定的我",在架构上表现为 role_i 由可学习的 type_emb、cap_emb、id_emb 组合而成;对他人类型的信念对应"后验推断的他人",在架构上表现为 belief_i 由 BeliefNet 在历史上 unroll 得到。这一对应使本文的架构选择不再是"工程经验",而是博弈论上的几何必然——这一对应**在 Harsanyi 1967 之后的 60 年里首次在深度世界模型中被显式实现**。

---

## 4.1.5 方法论原则:偏好不变,行为涌现

类型 β 的设计严格遵循以下方法论原则,该原则也是本文与 v1 "可变捕猎者" 路线的根本区别:

> **agent 的心理偏好结构在 RL 训练过程中保持不变,行为模式作为最优策略涌现于环境动力学与偏好结构的交互**。

具体到类型 β:

- **不变项**:Fehr-Schmidt 不平等厌恶结构 $\psi(\Delta) = -[\lambda_{\text{disadv}} \max(0,-\Delta) + \lambda_{\text{adv}} \max(0,\Delta)]$ 在所有 $c$ 区段保持不变;参数 $\lambda_{\text{disadv}}, \lambda_{\text{adv}}$ 在所有 episode 保持不变。
- **变化项**:不平等厌恶的强度与方向由 $\phi(c)$ 调制,但 $\phi$ 本身是环境 $c$ 的确定性函数,不依赖 agent 的行为决策。
- **涌现项**:荒年表现为剥削/激进追赶、丰年表现为协同合作——这两种行为模式不是被显式编码到奖励函数中,而是 RL 在偏好空间中找到的最优响应。

这一原则在博弈论与社会困境文献中有明确对应:Fehr-Schmidt 偏好属于"distributional preferences"类别,行为不被指定,只被偏好驱动;Hughes (2018) [11] 的不平等厌恶 agent 同样遵循此原则。本文的关键演进是引入 $\phi(c)$ 上下文调制,使不平等厌恶的强度成为博弈关系动态切换的纽带。

**与 v3 版本中 β(c) 协同加成的对比**:v3 的 β(c) 直接修改物理采集量公式 $u_{i,k} \mapsto \eta_i[1 + \beta(c)(|H|-1)]$,这相当于将"合作行为"显式编码到瞬时支付中,违反"偏好不变,行为涌现"原则。v4 砍除 β(c),将合作-竞争激励完全交给类型 β 的偏好结构,环境物理层回归严格自利同构,合作 ↔ 剥削的切换完全从偏好与动力学交互中涌现。

---

## 4.1.6 本节论断与后续章节的承诺映射

本节提出的论断在后续章节如何被兑现:

- **4.2 节**:展开三联通路(c_ctx、role_i、belief_i)的具体编码方式,role_i 加入 type_emb(由 Self Info 设定保证自己类型可见),并给出 BeliefNet 的 GRU 结构与两个 belief head($\hat{c}$ 与 $\hat{z}$)的设计。
- **4.3 节**:展开 hyper_trans、hyper_rew、hyper_pred 的具体输入-输出维度与参数生成方式,显式对应 4.1.4 节的 Harsanyi 表格。
- **4.4 节**:列出包含 BeliefNet 的六大核心网络数据流,显式标注每个网络的"客观/主观"归属。
- **4.5 节**:给出 BeliefNet 的多任务训练目标($\hat{c}$ 监督 + $\hat{z}$ 对手类型离散分类 + 多样性正则),以及与 4.1.3 节辅层 1 的对应关系。
- **4.6 节**:复用 v1 验证有效的四层稳定性防线,并指出三联输入引入的新挑战及其应对。
- **第五章**:给出训练阶段的课程学习协议(对手类型推断头从 Oracle → 退火 → 纯推断三阶段),以及主任务损失与 BeliefNet 损失的联合优化方案。
- **第六章**:通过消融 1(架构骨架,启用 μP + LR sweep + Input-Wide/Deep 双跑)、消融 2(Context 通路)、消融 3(类型异质性扫描 + 1α+3β 对照)、消融 5(Fehr-Schmidt 参数敏感性)四组实验,系统检验断言 A、B、C 的成立性。

**特别承诺**:第六章消融 1 的"Hyper-MuZero vs Input-Wide vs Input-Deep(全部启用 μP 学习率对齐,各自 LR sweep)"对照实验是断言 B 的硬验证,也是本章 4.1.2 节"容量分配几何"论证的最终落地。如该对照实验未能展示出显著差距,本章 4.1.2 节的论证需要降级为"理论可能性而非实际优势",贡献 2 的成色相应调整。这一承诺在 Chapter 1.5 节贡献 2 中已明确提出,本节再次确认。

---

## 4.1.7 本节小结

本节将 v3 版本中"主客解耦防止梯度撕裂"的笼统 motivation,重新组织为基于**容量分配几何**的严格论证,并将 DualHyperNetwork 的设计 motivation 分解为三层:类型梯度撕裂(主)、信念稀释(辅 1)、角色平均化(辅 2)。每一层对应一个 context 通路、一个可证伪断言、一组消融实验。

类型梯度撕裂的具体形式由 Fehr-Schmidt 不平等厌恶 + $\phi(c)$ 调制给出:类型 α 偏导恒为 1,类型 β 偏导在 (c 区段, $\Delta$ 符号) 联合下取值从 0 到 2 不等,共享 RewardHead 必然在反向传播中被两类梯度撕裂。该撕裂的可视化与定量证据在第六章消融 1 中给出。

该重组的核心方法论意义在于:DualHyperNetwork 不再是一个"看起来有道理但没有硬验证"的设计选择,而是一个**与 Harsanyi 不完全信息博弈严格对应、与 hypernetwork vs input conditioning 容量分配理论严格对应、与 Fehr-Schmidt 不平等厌恶具体梯度结构严格对应、与三组消融实验严格对应**的多重锚定架构。
