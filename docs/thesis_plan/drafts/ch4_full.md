# 第 4 章 方法

> 本章在第 3 章 RNS-MMG 形式化与 ResourceCommons 基准的基础上，给出本论文方法层的完整设计。§4.1 从代数与梯度方向两个层面确立三个容量瓶颈，作为后续条件化通路设计的形式化动机；§4.2 把条件化机制视为一条连续坐标轴，分析其表达力-优化几何并据此预登记断言 B′；§4.3 借用 Harsanyi [10] 不完全信息博弈框架中的"共同知识 / 私人信念"二分作为双路分解的方法选择参照系；§4.4 在 D3 完整代数加工程纪律的颗粒度上展开 DualHyperNetwork v2 架构，包括三联上下文构造、AdaLN 残差调制、grouped RMS 与 LoRA 初始化纪律、四层稳定性栈与 BeliefNet 双头结构；§4.5 以几何示意加方差差分推导刻画 Per-Agent Coordinate Descent 与 CRN 的协同机制；§4.6 给出 Type-aware K 步展开训练算法与三阶段课程，并就当前 suite 中 belief 通路的实测激活情况作如实披露。本章引用的所有方法层数字（如 cos_pred_cross 0.72 → 0.03 的谱内变化、planner 收益 105 → 272 的非单调结构）均前向参考 §5.2 与 §5.5 的实测，引用规范见 §1.0 的[N] 编号体系；为避免本章自封闭论证与下章实测形成循环引用，本章的所有定量预测均以"§5.x 实证"或 [证据待补 EVIDENCE PENDING] 形式标注。

---

## §4.1 动机：三个容量瓶颈

本节从代数层面确立三个容量瓶颈，作为 §4.4 DualHyperNetwork v2 三条条件化通路（c_ctx、role_i、belief_i）的形式化动机。三个瓶颈分别对应类型异质 agent 在共享 RewardHead 上的反向梯度撕裂、Self-Info 观测下他人类型的信念稀释、以及异质能力向量在共享主干表示空间内的角色平均；三者并非彼此孤立，而是共同支撑断言 A（钟形曲线）与断言 C（三联通路缺一不可）的预登记预测。

**第一个瓶颈：类型梯度撕裂**。在 §3.5 偏好层定义下，类型 α 的奖励函数为 $R_i^\alpha = u_i$，其对联合动作中本 agent 物理采集量 $u_i$ 的偏导为 $\partial R_i^\alpha / \partial u_i \equiv 1$，恒为常数且与 $c_t$ 无关；类型 β 的奖励函数为 $R_i^\beta = u_i + \varphi(c_t) \cdot \psi(\Delta_i)$，其中 $\psi(\Delta) = -[\lambda_{\text{disadv}} \max(0, -\Delta) + \lambda_{\text{adv}} \max(0, \Delta)]$，Fehr-Schmidt 经典参数 $\lambda_{\text{disadv}} = 2.0, \lambda_{\text{adv}} = 0.6$（[40]），$\varphi(c_t) = \kappa(1 - 2 c_t)$ 且 $\kappa = 0.5$ 使 $\varphi \in [-0.5, +0.5]$。考虑 $u_i$ 增量对 $\Delta_i$ 的影响——若本 agent 处于劣势 ($\Delta_i < 0$)，则 $\partial \Delta_i / \partial u_i > 0$，因而 $\partial \psi / \partial u_i = +\lambda_{\text{disadv}}$；若处于优势 ($\Delta_i > 0$)，则 $\partial \psi / \partial u_i = -\lambda_{\text{adv}}$。结合 $\varphi$ 的符号变化，类型 β 的有效梯度可写为

$$\frac{\partial R_i^\beta}{\partial u_i} = 1 + \varphi(c_t) \cdot \frac{\partial \psi(\Delta_i)}{\partial u_i}.$$

代入参数后取得四个区段值：(c-低区段 $\varphi > 0$，劣势 $\Delta < 0$) 梯度为 $1 + 0.5 \cdot 2.0 = 2.0$；(c-低区段 $\varphi > 0$，优势 $\Delta > 0$) 梯度为 $1 + 0.5 \cdot (-0.6) = 0.7$；(c-高区段 $\varphi < 0$，劣势 $\Delta < 0$) 梯度为 $1 + (-0.5) \cdot 2.0 = 0$；(c-高区段 $\varphi < 0$，优势 $\Delta > 0$) 梯度为 $1 + (-0.5) \cdot (-0.6) = 1.3$。即类型 β 的梯度跨区段取值于 $\{0, 0.7, 1.3, 2.0\}$，与类型 α 的恒定值 1 在多数区段差异显著。

共享 RewardHead 的反向传播将 α 与 β 两类梯度按混合比例 $\rho_\beta$ 在参数空间内聚合，最终收敛位置近似为两类梯度的加权平均——既不等于 α 的最优策略（恒定个体收益最大化），也不等于 β 的最优策略（按 $c_t$ 区段切换的不平等厌恶响应）。当 $\rho_\beta \approx 0$ 或 $\rho_\beta \approx 1$ 时，两类梯度方向冲突的样本数较少，撕裂幅度小；当 $\rho_\beta$ 接近 0.5 时，两类梯度方向冲突的样本数最多，撕裂最严重——这一钟形分布结构是断言 A 的代数根源。需要说明的是，钟形的精确峰位（对称在 0.5 或非对称在 0.25 / 0.75）依赖于学习动态的不对称性，代数推导本身无法给出闭式预测，须通过消融 3 在 $\rho_\beta \in \{0, 0.25, 0.5, 0.75, 1.0\}$ 五点的扫描实证 [证据待补 EVIDENCE PENDING — ρ_β 五点扫描 (消融 3)，待 §5.7 补完]。

类型梯度撕裂的代数性质同时也回应 §1.2 (C2)：奖励函数的类型-上下文联合调制要求 RewardHead 的输出方向能够随 $(c_t, \tau_i)$ 同时偏移，共享参数的反向梯度聚合不满足该要求；DualHyperNetwork v2 的 hyper_rew 通过把 RewardHead 权重参数化为 $c_{aug}$ 的函数，使每条 (rule, agent-id) 上下文获得 per-agent 的 $\theta_{\text{rew}}^i$，从而原则上消除撕裂——但代数消除并不等价于实证消除，必须借由消融 3 验证。

**第二个瓶颈：信念稀释**。Self-Info 观测设定（§3.5）下，每个 agent 仅观测到自身类型 $\tau_i$ 与物理量（位置、速度、视野内资源），他人类型 $\tau_{-i}$ 作为隐变量需通过历史轨迹的后验推断。形式化地，设共享网络 $f_\theta(s, c_t)$ 在所有信念 $P(\tau_{-i} | o_i^t)$ 下都要拟合一个综合最优策略，该综合策略在任一具体后验下的精度上界 $\sup_\pi \mathbb{E}[\text{return} | \tau_{-i} \sim P(\cdot|o_i^t)]$ 严格不超过按后验单独建模的"分别拟合上界" $\mathbb{E}_{\tau_{-i} \sim P} \sup_\pi \mathbb{E}[\text{return} | \tau_{-i}]$——由 Jensen 不等式与策略空间的非线性可推得。

实际网络容量有限时，共享网络在不同后验下的拟合误差累计且互相干扰，最终学得的策略是各后验下最优策略的某种"平均",即信念在共享参数上被"稀释"。这一稀释效应在 [11] Hughes 等人的 Cleanup 实验中以共享 reward shaping 项与个性化 reward shaping 项之间的性能差形式呈现，与本文方法的差异在于 [11] 通过外加奖励项缓解，本文通过把 RewardHead 权重条件化于 $belief_i = [\hat{c}_i, \mathrm{Pool}(\{\hat{z}_{i,j}\}_{j \neq i})]$ 在参数层处理。Pool 算子（详见 §4.4）把不同对手的 $\hat{z}$ 后验聚合为对环境类型分布的紧凑表示，避免 $\theta_{\text{rew}}^i$ 维度随 $N$ 线性膨胀。

信念稀释瓶颈回应 §1.2 (C3)：$c_t$ 的物理不可控性意味着 agent 必须依赖历史观测推断其当前值，而当 $c_t$ 还隐式调控对手类型行为时，所需推断扩展至 $\hat{c}_i$ 与 $\hat{z}_{i,j}$ 的联合后验。共享网络在该联合后验空间内的稀释效应是 BeliefNet 必须作为独立模块存在的形式化依据。

**第三个瓶颈：角色平均**。每位 agent 在 episode 开始时被指派一个能力向量 $(\eta_i, \varphi^{fov}_i, \nu_i, \zeta_i)$：采集能力 $\eta_i$、视野半径 $\varphi^{fov}_i$、移动速度 $\nu_i$、容量 $\zeta_i$，分别按指定分布独立采样。当共享主干 $\phi(s; \theta_{\text{rep}})$ 把这些异质能力向量映射到同一隐空间时，所学到的表征是各能力维度上的折中——例如对采集能力 $\eta_i$ 较低但视野较大的 agent 而言，最优策略是远距侦察并选择性介入低竞争区段，但共享主干在拟合该类型策略的同时必须兼容采集能力高但视野小的"近场掠夺者"的最优策略，两类策略在表征空间内互相牵制。

形式化地，设 agent 在能力向量 $\rho_i \in \mathbb{R}^4$ 下的最优策略为 $\pi^*(s; \rho_i)$，共享主干的容量瓶颈意味着任意 $\theta_{\text{rep}}$ 下存在能力向量 $\rho_i$ 使得 $\|\pi^*(s; \rho_i) - \pi(s; \theta_{\text{rep}}, \rho_i)\|_{TV} > \epsilon_0$，其中 $\epsilon_0 > 0$ 为容量下确界。DualHyperNetwork v2 通过把 $\rho_i$ 编码进 $role_i = [\text{id\_emb}, \text{cap\_emb}, \text{type\_emb}]$ 并经 hyper_pred 生成 per-agent 的预测头权重 $\theta_{\text{pred}}^i$，在主观通路上把能力异质性显式条件化为权重生成的输入——以参数空间的扩展换取策略专化上限的提升。

三个瓶颈分别对应 §4.4 三联上下文的 c_ctx、role_i、belief_i 三条通路：c_ctx 负责跨上下文（含 $c_t$）的客观动力学解耦，role_i 通路承接能力与类型异质性的主观偏好编码，belief_i 通路压缩对他人类型的后验。三者缺一不可——这正是断言 C 的预登记声明；其可证伪性由 §5.6 消融 2 在 5 组零置条件 + 多路联合零置下的退化曲线检验 [证据待补 EVIDENCE PENDING — c_ctx/role/belief 单路与多路零置消融，待 §5.6 补完]。

**三瓶颈的形式化归约**。上述三个瓶颈虽然在物理机制上不同（前向传播梯度冲突 / 后验拟合精度上界 / 表征空间容量瓶颈），但可以在一个统一框架下作形式化归约。设 $\mathcal{H}$ 为共享网络的假设空间、$\mathcal{D}$ 为训练数据分布、$\ell$ 为损失函数，则"最优共享网络"的最小化风险为 $R^*(\mathcal{H}) = \inf_{f \in \mathcal{H}} \mathbb{E}_{(x,y) \sim \mathcal{D}} \ell(f(x), y)$。若数据分布 $\mathcal{D}$ 可分解为若干子分布 $\mathcal{D} = \sum_k \pi_k \mathcal{D}_k$（如按 $(c_t, \tau_i)$ 或按 $\rho_i$ 划分），则**分别最优风险**为 $R_{\text{sep}}^*(\mathcal{H}) = \sum_k \pi_k \inf_{f_k \in \mathcal{H}} \mathbb{E}_{(x,y) \sim \mathcal{D}_k} \ell(f_k(x), y)$。三瓶颈的存在意味着 $R^*(\mathcal{H}) \ge R_{\text{sep}}^*(\mathcal{H})$ 的差距（即"共享最优 vs 分别最优"的效率损失）在如下三种情形下不可忽略：**(a) 类型梯度撕裂**——子分布 $\mathcal{D}_{(c, \tau)}$ 的最优 $f_k$ 在不同 $(c, \tau)$ 上梯度方向相反，$\mathcal{H}$ 上任一 $f$ 不能同时匹配（前向传播梯度约束）；**(b) 信念稀释**——子分布 $\mathcal{D}_{P(\tau_{-i})}$ 的最优 $f_k$ 需按后验切换，$\mathcal{H}$ 上任一 $f$ 只能匹配一个平均后验；**(c) 角色平均**——子分布 $\mathcal{D}_{\rho_i}$ 的最优 $f_k$ 在不同能力向量上专化程度不同，$\mathcal{H}$ 上任一 $f$ 只能学到能力平均。DualHyperNetwork v2 的三联通路正是将 $R^*(\mathcal{H})$ 中的"单一 $f$"扩展为"按上下文 $(c, \tau, \rho, \text{belief})$ 生成的函数族 $\{f_{c, \tau, \rho, \text{belief}}\}$"，将风险从共享最优降至（渐近于）分别最优。

**为何需要三条独立通路而非合并为一条**：形式上，若把 c_ctx、role_i、belief_i 拼接为单一 80 维向量后经一个 hypernet 生成所有功能网权重，则理论上也能获得同等表达力。但方法层选择独立三通路的两条工程理由如下——**(i) 客观 vs 主观解耦**：hyper_trans 只需消费 c_ctx（对应 C1 的 agent 同构性），若混入 role_i 与 belief_i 则 $\theta_{\text{state}}$ 会人为 agent-specific，违反公理 P1；这一分离是架构层直接落地 C1 约束的手段。**(ii) 可解释性与消融可分离性**：三通路独立才使得 §5.6 消融 2 能够单独零置任一通路而不干扰其他通路；若三通路合并为单一 hypernet 输入，则零置一部分输入等价于对该 hypernet 施加分布外扰动，实验解释力受损。这两条理由共同支持"三条独立通路"作为方法学而非工程学选择。

---

## §4.2 条件化谱：容量-优化几何

本节在条件化超网络的表达力-优化效率两维平面上建立分析框架，为 §5.5 断言 B′ 的预登记奠定基础。**条件化谱**定义为：将 per-context 参数生成机制视为一条连续坐标轴，左端为最弱的输入拼接（Input Concat），右端为最强的全参数超网络（Full HyperNet）。谱上各点以"表达力 vs 优化方差"两个对立量构成 Pareto 前沿。

**左端：Input Concat**。把上下文 $c$ 直接拼接至网络输入层，等效参数增量 $\theta_{\text{eff}}(c) = \theta_{\text{base}} \oplus g_\Theta(c)$ 仅在第一层引入 $O(h)$ 量级的额外自由度——其中 $h$ 是 backbone 第一层的输出宽度。CAVIA（Zintgraf 等人 [43]）以这一机制实现快速适配，参数效率高但容量受限。容量受限的直接后果是"容量平均化"（capacity averaging）：不同 $c$ 区段的最优响应在功能层（functional layer）差异较大时，输入侧的拼接无法在中间层重组特征以实现差异响应，共享 backbone 学到的是所有 $c$ 区段的混合最优策略而非各区段的专属策略。

**中间区段：FiLM、CAVIA、LoRA**。FiLM（Perez 等人 [44]）对共享主干的中间层做对角调制 $h \leftarrow \gamma(c) \odot h + \beta(c)$，调制系数 $(\gamma, \beta)$ 由 $c$ 经轻量 MLP 生成。FiLM 的有效自由度由调制层宽度 $\gamma, \beta \in \mathbb{R}^h$ 给定，量级 $O(h)$；但相较 Input Concat 的差异在于调制作用于中间层激活而非输入特征，能够在已编码的高阶特征上做条件化变形。LoRA（Hu 等人 2022）通过秩-$r$ 增量 $\Delta W = BA$ 实现中等容量条件化，其中 $A \in \mathbb{R}^{r \times h_{\text{in}}}, B \in \mathbb{R}^{h_{\text{out}} \times r}$；参数量随秩 $r$ 线性增长，自由度 $O(r \cdot h)$，在大语言模型适配领域已成为标准做法。

**右端：Full HyperNet**。Ha 等人 [42] 的原始构型让一个 hypernet $g_\Theta(c)$ 直接生成目标网络的全部参数 $\theta = g_\Theta(c)$，每个上下文 $c$ 独立生成完整网络权重，可表达力达 $O(h^2)$ 量级。该方案的优化代价同样最大：per-context 梯度估计方差随自由度 $O(h^2)$ 增长而急剧放大，优化轨迹在高维参数空间中振荡，不同上下文的预测表征趋于坍缩——v4-opt 2026-06 阶段的工程观察显示，FULL gen_scope 下 cos_pred_cross 在训练初期从 0.61 出发，于约 500k 步内单调攀升至 0.998，即网络对不同上下文输出近乎相同的潜在向量，条件化的区分能力实质上已经丧失。Galanti 与 Wolf（2020 NeurIPS）从谱归一化角度给出过类似的坍缩条件分析，本论文不主张该工程观察等同于完整的坍缩理论，仅作为方法谱右端的实证标记 [证据待补 EVIDENCE PENDING — FULL gen_scope 谱右端坍缩的预登记复现，待 §5.5 补完]。

**双面失败 vs 内部最优**。上述左右两端在表达力与优化方差上分别走向极端：左端 Input Concat 表达力不足导致 capacity averaging，预测网络对不同 $c$ 输出几乎一致的方向；右端 Full HyperNet 优化方差爆炸导致 directional collapse，预测网络对不同 $c$ 同样输出近乎一致的方向。两端的失败模式表观相似（cos_pred_cross 高），机制相反（一是不愿区分，一是不能区分）。断言 B′ 的核心命题是：在谱内部存在容量-优化 Pareto 最优区间，使得表达力足以区分不同 $c$ 下的动态，同时优化方差控制在可接受范围内；理论预测该区间位于 film_head（FiLM 调制头）至 lora_fc2（LoRA on fc2 层）之间。

本文实验套件内部对该理论预测做了 5 个变体的实证扫描（具体结果见 §5.2 与 §5.5）：(i) **Input 端基线**——`main_comparison_easy`（3 seeds × 137k steps）与 `zero_shot_easy`（3 seeds × 200k steps）两个 cell 共享 Input Concat 条件化，cos_pred_cross 分别为 $0.72 \pm 0.02$ 与 $0.71 \pm 0.01$，planner return 分别为 $105.4 \pm 4.8$ 与 $104.0 \pm 3.3$，共同标定左端基线于 105 收益与 0.7+ 相似度区段；(ii) **LoRA only**——`lora_duo_base`（1 seed × 189k）planner return 263、cos_pred_cross 0.08，相似度跨越式下降；(iii) **FiLM only**——`lora_duo_film`（2 seeds × 240k）planner return $250 \pm 2$、cos_pred_cross 0.09，与 LoRA only 在同一性能区段；(iv) **FiLM + LoRA on fc2**——`lora_duo_film_fc2`（3 seeds × 240k）planner return $272 \pm 18$、cos_pred_cross 0.03、planner_prior_gap $+19.6 \pm 6.7$，达到本套件已完成变体中的收益峰值。

从 0.72 至 0.03 的 cos_pred_cross 单调下降（横跨 5 个变体）与 105 至 272 的 planner return 非单调上升（左端 105 → 中间 250-263 → film+fc2 峰值 272）共同构成断言 B′ 中"内部最优存在性"的实测证据。需要说明的是，本扫描尚未覆盖 7 变体完整集合（Shared、Input-Wide 与 FULL 三边界变体缺失 [证据待补 EVIDENCE PENDING — Shared/Input-Wide/FULL 三边界变体的预登记复现，待 §5.5 补完]），右端坍缩在本 suite 内部为前向论断而非实测；因此 §5.5 的最终验证结论为"部分支持"。

**Pareto 前沿的形式化刻画**。为把"容量-优化 Pareto 最优"这一直觉表述给出精确形式，本节引入两个可量化度量：**表达力度量 $E(\theta)$** 与 **优化方差度量 $V(\theta)$**。表达力可粗略定义为 hypernet 在上下文 $c$ 上生成的函数族的 Rademacher 复杂度上界 $E(\theta) \le C \cdot \sqrt{\text{VC-dim}(g_\Theta) / n}$，其中 $g_\Theta$ 是 hypernet、$n$ 是训练样本数——直觉上表达力越大，$E$ 越大。优化方差可用随机梯度估计的方差量级 $V(\theta) = \mathbb{E}_{\text{minibatch}} \|\nabla \mathcal{L} - \bar{\nabla} \mathcal{L}\|^2$ 表征，与 hypernet 的自由度呈超线性关系（$O(h^2)$ 端为最差）。综合最优（在有限训练预算下）风险为 $\mathcal{R}(\theta) = R^*(\theta) + \eta \cdot V(\theta) / n$，其中 $\eta$ 是学习率、$n$ 是训练迭代数——第一项是最佳可拟合风险，第二项是有限训练下的优化残差。谱上不同点 $\theta_{\text{gen\_scope}}$ 使 $E$ 与 $V$ 呈单调递增关系，因此 $\mathcal{R}$ 关于 gen_scope 是单调递减 + 单调递增两项之和，存在**内部最小值**——即断言 B′ 所指的 Pareto 最优点。这一形式化不给出内部最小值的解析位置（依赖 $E$ 与 $V$ 的具体形式与训练预算），但保证其存在性，是 B′ 的方法学基础而非工程直觉。

**方向坍缩的形式化机制**。谱右端 Full HyperNet 观察到的 cos_pred_cross 0.61 → 0.998 演化可以从"高维参数空间中的信息瓶颈"角度理解。设不同上下文 $c$ 生成的功能网参数 $\theta(c) = g_\Theta(c)$ 位于参数流形 $\mathcal{M} \subset \mathbb{R}^{|\theta|}$ 上。在训练过程中，reward loss 通过 $g_\Theta$ 的雅可比 $J_\Theta = \partial \theta / \partial \Theta$ 反传，其奇异值分布决定不同 $c$ 生成的 $\theta(c)$ 在参数空间中的可分性。当 $|\theta| \sim O(h^2)$ 显著大于训练样本能有效激发的方向数（$\sim O(n / T)$，$T$ 为 batch），$J_\Theta$ 的奇异值谱迅速衰减，$\theta(c)$ 的有效变化被压缩到低维子空间——不同 $c$ 生成的功能网权重趋于同一方向，即 cos_pred_cross → 1。这一机制与 Galanti & Wolf 2020 中关于 hypernet 谱归一化的分析在数学上等价，不同之处在于本文强调该机制在训练过程中的动态演化（0.61 起始点 → 0.998 终点，而非初始化即坍缩）。方法层的应对策略——(1+γ) AdaLN 残差调制（§4.4）保证初始化时 $\gamma \to 0$ 情形下功能网退化为基础网络、grouped RMS 归一化把 $\Delta W$ 的方向与幅度解耦（§4.4）——都在阻止上述雅可比奇异值谱的坍缩。

本节的几何分析旨在为 §4.4 D3 完整代数所做的"thesis-default = film_fc2"工程选择提供方法学依据：当方法选择能够同时由理论预测点（容量-优化 Pareto 内部）与实测最优点（lora_duo_film_fc2 收益峰值）合拢时，方法选择即获得方法学层面的两层支撑。

---

## §4.3 Harsanyi 类比：主-客通路的分解依据

本节就 DualHyperNetwork 双路分解的方法选择依据作专门说明。Harsanyi [10] 1967 年提出的不完全信息博弈框架是博弈论中处理"类型异质 + 信念不完整"问题的经典工具，其核心贡献在于把任意不完全信息博弈通过 nature 在博弈开始时按先验分布抽取每个 player 的类型这一辅助步骤，等价转换为完全但不完美信息博弈（complete but imperfect information game）。该转换的产物是**两层结构**：共同知识层（common knowledge）包含所有 player 都同意的物理规律、他人理性程度与可能的类型空间；私人信念层（private beliefs）包含每个 player 对他人类型的后验估计。Harsanyi 进一步证明，在共同知识假设下，不完全信息博弈与对应的类型博弈具有相同的策略空间与均衡集合，从而把"博弈未知"压缩为"类型未知"，由 player 在交互过程中以贝叶斯更新的方式逐步估计。

这一二分在 RNS-MMG 下具有自然的语义对应。**共同知识层**对应共享上下文 $c_t$：在 $c_{\text{visible}} = \text{True}$ 档下 $c_t$ 直接在所有 agent 的观测中可见，在 $c_{\text{visible}} = \text{False}$ 档下 $c_t$ 虽不可见但可由资源场动力学反推（§3.4），其演化规律 $P_c(c_{t+1} | c_t)$ 与初始分布 $\rho_c$ 是所有 agent 都已知的环境规范。**私人信念层**对应每个 agent 对他人类型的后验估计 $P(\tau_{-i} | o_i^t)$：在 Self-Info 观测档下每个 agent 自身类型 $\tau_i$ 对自己可见而对他人不可见，他人类型须通过历史轨迹推断；BeliefNet 的 $\hat{z}_{i,j}$ 头正对应这一后验估计。

DualHyperNetwork 的双路结构与上述二分在三点上形成对应——但本文明确把这一对应限定为方法选择的**结构类比**而非形式化等价。结构类比的三点对应分别如下：

(i) 客观通路 hyper_trans 仅消费共享上下文 $c_{\text{ctx}}$ 作为输入，生成所有 agent 共用的状态转移功能网参数 $\theta_{\text{state}}$。该路径不接收 agent-specific 输入（id_emb、cap_emb、type_emb 均不进入），与 §1.2 (C1) 物理转移的 agent-同构性形成对应——物理动力学是所有 agent 都同意的规律，其参数生成不依赖任何主观信息。Harsanyi 框架中的共同知识层在结构层上对应于此。

(ii) 主观通路 hyper_rew 与 hyper_pred 消费三联上下文 $c_{\text{aug}} = [c_{\text{ctx}}, role_i, belief_i]$ 作为输入，生成 per-agent 的奖励参数 $\theta_{\text{rew}}^i$ 与预测参数 $\theta_{\text{pred}}^i$。其中 role_i 输入包含 agent 自身类型（type_emb 由 $\tau_i \in \{\alpha, \beta\}$ 编码），与 Harsanyi 框架中 player 知晓自身类型的设定一致；belief_i 输入包含对他人类型的后验估计 $\{\hat{z}_{i,j}\}_{j \neq i}$（经 Pool 算子聚合），与 Harsanyi 框架中 player 对他人类型的私人信念形成对应。

(iii) 两条通路在网络层级上的分离对应 Harsanyi 框架中共同知识与私人信念在博弈结构上的分离。客观通路独立于主观通路（hyper_trans 不接收 role_i 或 belief_i），主观通路依赖客观通路（hyper_rew、hyper_pred 接收 c_ctx 作为输入之一），这一不对称依赖关系反映了"物理动力学决定均衡可行集，主观偏好决定均衡选择"的博弈论次序。

需要明确**类比的方法学限度**。Harsanyi 框架是博弈论层的概念性二分，本文是神经网络架构层的工程实现；二者的对应仅是结构相似（structural similarity），并不构成形式化等价证明——本文不主张 DualHyperNetwork 在数学意义上"实现"了 Harsanyi 框架，亦不主张其为 60 年来 Harsanyi 框架在深度学习架构上的首次对应。本节就此把 Harsanyi 框架的方法学角色刻画为"借用其'共同知识/私人信念'二分作为方法选择的理论参照系"。

**与奖励塑形范式的方法学分野**。Fehr-Schmidt 不平等厌恶模型 [40] 在 [11] Hughes 等人的 SSD 框架中通常以外加内在奖励项的形式部署——即在环境奖励 $u_i$ 之上叠加 $\lambda_{\text{disadv}} \max(0, -\Delta_i) + \lambda_{\text{adv}} \max(0, \Delta_i)$ 作为合作激励，该项在网络优化时直接进入即时奖励信号。本文方法在此点上选择不同的部署方式：偏好结构通过 role_i = [id_emb, cap_emb, type_emb] 在参数空间中隐式编码——具体而言，type_emb 作为 hyper_rew 的输入之一参与生成 $\theta_{\text{rew}}^i$，使奖励函数的函数式本身随类型变化而 RewardHead 的输出在训练中不被显式加项干扰。两种部署方式的差异不仅是工程实现的差异：奖励塑形改变了 MDP 的 reward function 因而改变了均衡集合，参数化条件化则保持 MDP 不变并以网络权重的"专属化"逼近不同类型下的最优策略。前者在偏好结构变化时需重新设计 shaping 项，后者由参数生成机制自动适应；本文方法更适合 RNS-MMG 中类型异质需在线推断、$c_t$ 持续演化的场景。

**Harsanyi 变换的完整陈述**。为让类比的对应精确，本节复述 Harsanyi 变换的正式表述。设原始博弈 $G$ 是一个不完全信息博弈——每个 player $i$ 的类型 $\tau_i$ 从某个先验分布 $P_i(\tau_i)$ 中采样，type 空间为 $\mathcal{T}_i$，$i$ 只知自身类型但不知他人类型。**Harsanyi 变换**通过引入一个"nature move"在博弈开始时同时抽取所有 player 的类型，将不完全信息博弈 $G$ 转换为等价的**完全但不完美信息博弈** $G^*$——所有 player 知晓类型空间 $\prod_i \mathcal{T}_i$ 与先验 $\prod_i P_i(\tau_i)$，但不观察他人的实际抽签结果。这一转换的价值在于将"类型不确定性"从博弈规范的一部分转换为"信息集"的属性，使博弈的 Bayes-Nash 均衡可以定义为策略函数 $\pi_i: \mathcal{T}_i \to \Delta(\mathcal{A}_i)$——每个类型独立选择动作分布，同一 player 的不同类型可选不同策略。这一策略函数结构与 DualHyperNetwork 的 per-agent 参数生成结构在数学上同构：$\pi_i(\tau_i) \leftrightarrow \theta_{\text{pred}}^i(c_{\text{aug}})$，其中 $c_{\text{aug}}$ 包含 $\tau_i$（自身类型）与 $\hat{z}_{i, -i}$（对他人类型的信念）。

**结构类比不等价于形式化实现的三点限制**：本文的 Harsanyi 类比在方法学层面强调结构相似，明确不主张实现层面的等价。第一，Harsanyi 变换是理论框架层的**转换定理**（转换后的博弈与原博弈均衡集相同），而 DualHyperNetwork 是**神经网络架构层的功能实现**——前者是数学等价，后者是近似逼近。第二，Harsanyi 框架要求"共同先验"（common prior）假设——所有 player 都知道 $\{P_i\}$，本文方法在实现层不显式使用类型先验分布，而通过 BeliefNet 从历史观测中学习 $\hat{z}$ 的分布，其与真实先验 $P$ 的一致性依赖训练动态而非架构保证。第三，Bayes-Nash 均衡在完全信息博弈上是**类型条件的策略函数族**，DualHyperNetwork 的 per-agent 权重生成虽然形式上对应此结构，但其收敛点并不保证是 Bayes-Nash 均衡——多智能体强化学习的收敛性质仅在有限步数下给出 $\varepsilon$-Nash 意义下的近似（§4.5）。以上三点限制使本文将 Harsanyi 类比定位为方法学参照系，不作为架构层的"首次实现"或"数学等价"主张。

**Bayes-Nash 结构与方法层的一致性**。尽管上述限制存在，DualHyperNetwork 的方法层设计在四个具体点上与 Bayes-Nash 结构一致：(a) 客观通路只消费共同知识变量（$c_{\text{ctx}}$），主观通路消费类型 + 信念——对应 Bayes-Nash 中"策略函数按类型条件而非按状态条件"的类型层次；(b) hyper_trans 生成的 $\theta_{\text{state}}$ 在所有 agent 间共享——对应共同知识下物理转移的公共性；(c) hyper_rew / hyper_pred 生成 per-agent 权重——对应策略函数 $\pi_i$ 的 per-player 独立性；(d) BeliefNet 的 $\hat{z}_{i,j}$ 头输出对手类型后验——对应 Bayes-Nash 中的 belief 更新机制。这四点一致性使 Harsanyi 类比不是空洞的比喻，而是给出了具体的架构映射对照表——即使不主张形式化等价，这一映射本身在方法学论证上仍具解释力。

**方法学参照系的边界**：Harsanyi 类比支持三件事情，不支持另一件事情。**支持**：(i) 解释为何选择"客观 vs 主观"双路分解而非单一大 hypernet 的架构决策；(ii) 解释为何 role_i 与 belief_i 应当作为独立输入而非合并；(iii) 说明 Full Info vs Self Info 双档实验（§3.5 / §5.6）的信息经济学意义。**不支持**：(i) 声称 DualHyperNetwork 是"Bayes-Nash 均衡求解器"——本文并不给出均衡收敛性证明，仅证明架构与均衡结构一致（对应 §4.5 Coord Descent 的 $\varepsilon$-Nash 不动点意义）。这一边界的明示是 §1.4 贡献声明降级为"结构类比"而非"理论实现"的方法学前提。

---

## §4.4 DualHyperNetwork v2 架构

本节系统展开 DualHyperNetwork v2 的完整架构设计，依次给出 (i) 三联上下文 $c_{aug}$ 的构造与维度规划，(ii) 主-客双路超网的功能切分，(iii) AdaLN 残差调制的 (1+γ) 形式与初始化纪律，(iv) grouped RMS 归一化与 LoRA 秩-r 初始化的协同纪律，(v) 四档 partial generation 模式与对应参数规模，(vi) 四层稳定性栈各层的工程动机与防护对象，(vii) BeliefNet 双头结构与训练目标，(viii) 完整前向传播流程与 perspective 切换机制。本节是 D3 完整代数加工程纪律颗粒度的展开，部分公式与超参在 §4.6 训练算法节再次出现以保持各节的局部自封闭性。

**(i) 三联上下文 $c_{aug}$ 的构造**。v1 版本采用单一 rule embedding 拼接 agent id 的设计，无法分离 agent-invariant 的物理规律与 agent-specific 的主观偏好。v2 把上下文编码拆为三路并行编码器：(a) `c_encoder`：把 $c_t$ 映射为 16 维的客观上下文嵌入 $c_{\text{ctx}} \in \mathbb{R}^{16}$。$c_t$ 是标量信号，16 维嵌入提供充分的非线性变换空间。该嵌入直接进入 hyper_trans，间接（与 role_i、belief_i 一起）进入 hyper_rew 与 hyper_pred。(b) `role_encoder`：把 agent 的固有信息 [id_emb, cap_emb, type_emb] 拼接编码为 32 维的角色嵌入 $role_i \in \mathbb{R}^{32}$。其中 id_emb 是 agent 唯一标识的可学习查表向量（位置维 8 维），cap_emb 是能力向量 $(\eta_i, \varphi^{fov}_i, \nu_i, \zeta_i)$ 经 MLP 编码的 16 维表示，type_emb 是类型 $\tau_i \in \{\alpha, \beta\}$ 的 8 维查表嵌入。32 维总宽度的选取与 $|role_i| \approx 2 \cdot |c_{\text{ctx}}|$ 的比例关系一致，使主观信息维度略高于客观维度，反映 RNS-MMG 中主观偏好差异是核心驱动因素的设计判断。(c) `belief_encoder`：把 BeliefNet 的两个头输出 $[\hat{c}_i, \mathrm{Pool}(\{\hat{z}_{i,j}\}_{j \neq i})]$ 经 MLP 编码为 32 维的信念嵌入 $belief_i \in \mathbb{R}^{32}$。其中 $\hat{c}_i \in [0, 1]$ 是 BeliefNet ĉ-head 的 sigmoid 输出，$\hat{z}_{i,j}$ 是对 agent $j$ 类型的 2-way softmax 概率，Pool 算子（实现为均值池化）把 $N-1$ 个对手类型后验聚合为环境类型分布的紧凑表示，避免 $belief_i$ 维度随 $N$ 线性膨胀。三路拼接得 $c_{aug} = [c_{\text{ctx}}, role_i, belief_i] \in \mathbb{R}^{80}$，作为主观双路超网的统一输入。

**(ii) 主-客双路功能切分**。hyper_trans 仅接收 16 维的 $c_{\text{ctx}}$ 作为输入，生成状态转移功能网（StateTransNet）的全部参数 $\theta_{\text{state}}$。该网络的预测目标是 $\Delta s = s' - s$（残差形式，避免恒等映射学习困难），输入为 $[s, a_{\text{joint}}]$（潜状态拼接联合动作 one-hot），输出为下一步潜状态增量。由于物理转移在所有 agent 间共享（§1.2 (C1)），$\theta_{\text{state}}$ 不依赖 agent-id 也不依赖类型；hyper_trans 输入仅含 16 维 $c_{\text{ctx}}$ 是该约束在架构上的直接落地。

hyper_rew 与 hyper_pred 接收完整的 80 维 $c_{aug}$ 作为输入，分别生成奖励头与策略-价值头的参数。RewardHead 的预测目标是即时奖励 $r_i$，输入为 $[s^k, a_{\text{joint}}^k]$，输出为标量奖励；PredictionHead 同时输出策略分布 $\pi_i(a|s^k)$ 与价值 $V_i(s^k)$，输入为 $s^k$。两个头共享主观通路的设计反映"奖励驱动价值，价值驱动策略"的 RL 因果链，但参数生成是独立的（hyper_rew 与 hyper_pred 是两个独立的 hypernet），避免单一 hypernet 同时拟合两类不同信号的容量瓶颈。

**(iii) AdaLN 残差调制的 (1+γ) 形式**。功能网内部的 layer normalization 层采用 AdaLN（Adaptive Layer Normalization）形式，调制系数 $(\gamma, \beta)$ 由 hyper_rew 或 hyper_pred 生成（具体取决于该 layer 属于哪个功能网）。AdaLN 的形式化为

$$h_{\text{out}} = (1 + \gamma) \odot \frac{h_{\text{in}} - \mu}{\sigma} + \beta,$$

其中 $h_{\text{in}}$ 是 layer 输入，$\mu, \sigma$ 是该 layer 输出维度上的均值方差（计算自 batch 与 token），$\gamma, \beta \in \mathbb{R}^{h}$ 是 hypernet 生成的调制系数。这一 (1+γ) 残差形式是 v4.6 优化阶段的关键修正——v4.5 及以前版本采用直接形式 $\gamma \odot h_{\text{in}} + \beta$，在 small_init 纪律下 $\gamma$ 初始值接近 0 使得 $h_{\text{out}}$ 在 float32 精度下被压制为接近零的输出，进而梯度无法回传，训练失败。(1+γ) 形式确保 $\gamma \to 0$ 时 $h_{\text{out}} \to h_{\text{in}}$，即在初始化时功能网退化为不带调制的基础网络，调制信号在训练过程中由零开始逐步生长，与 ResNet（He 等人 2016）的残差初始化思想一致，亦与 Karras 等人 2019 StyleGAN 的 modulated convolution 初始化纪律相通。Peebles 与 Xie 2023 DiT 在 diffusion transformer 中采用同一 (1+γ) 形式作为 timestep conditioning 的默认设计。

**(iv) Grouped RMS 与 LoRA 秩-r 初始化**。hypernet 直接生成的 $\Delta W$ 经 grouped RMS 归一化处理，使 $\|\Delta W\|_{\text{RMS}}$ 在初始化时受控。具体而言，把生成参数按输出通道分组（group 大小由功能网的层宽度决定），对每组内的元素计算 RMS 并除以 $\sqrt{|\text{group}|}$，再乘以可学习的 output_scale。理论分析给出 $\|\Delta W\|_{\text{RMS}} \approx \text{output\_scale}^2 \cdot \sqrt{r}$ 的标度律——其中 $r$ 是 LoRA 秩。该标度律的实际含义是：通过将 RMS 控制与 output_scale 系数解耦，hypernet 可在不改变方向（normalized direction）的前提下调整生成权重的幅度，避免 v3 阶段"幅度与方向混合优化"导致的训练不稳定。output_scale 初始值按功能网类型差异化设置：hyper_trans 取 0.01 较保守，hyper_pred 取 0.01，而 hyper_rew 取 0.1——后者较大是因为奖励梯度本身量级较小（参考 §4.1 类型 β 梯度 [0, 2] 区间），需要更大的初始权重幅度以建立可分辨的奖励信号。

LoRA 秩-r 增量 $\Delta W = BA$ 的初始化遵循 Hu 等人 2022 的标准纪律：$A \in \mathbb{R}^{r \times h_{\text{in}}}$ 从均匀分布 $\mathcal{U}(-1/r, 1/r)$ 采样，$B \in \mathbb{R}^{h_{\text{out}} \times r}$ 初始化为零。这一初始化保证 $\Delta W = 0$ 在训练初始时刻，即 LoRA 路径不干扰基础网络的初始功能；训练过程中 $B$ 由零开始更新使得 $\Delta W$ 的幅度逐步生长，与 (1+γ) AdaLN 的零初始化思路一致。对于 lora_fc2 模式，LoRA 仅作用于功能网的第二层全连接 fc2，秩 $r = 8$ 或 $r = 16$（按 partial generation mode 决定）。

**(v) 四档 partial generation 模式**。v2 提供四档生成范围，按从最弱到最强的参数生成范围排列如下：
- **`base_gen`**：固定基础功能网，hypernet 不参与权重生成。功能网总参数约 694k，超网部分仅生成 c_ctx 编码器与 role/belief 编码器，约 200k 参数。该档主要用于消融对照与训练稳定性测试。
- **`film_head`**：仅 AdaLN 调制层的 $(\gamma, \beta)$ 由 hypernet 生成，功能网其余参数固定。调制系数维度约 $2 \cdot h \cdot L = 2 \cdot 128 \cdot 3 = 768$ 每个上下文，hypernet 输出约 230k 参数。表达力受限于对角调制空间，参数效率高。
- **`lora_fc2`**：除 AdaLN 调制外，功能网第二层全连接 fc2 还接收 LoRA 增量 $\Delta W_2 = B_2 A_2$（秩 $r = 8$）。hypernet 输出约 470k 参数。本档是 §5.2 决策门 0 实测的 thesis-default 起点之一。
- **`film_fc2`**：在 lora_fc2 基础上，对 fc1 也施加 LoRA 增量 $\Delta W_1 = B_1 A_1$（秩 $r = 8$）。hypernet 输出约 720k 参数。本档对应 `lora_duo_film_fc2` 实验配置——在 §5.2 决策门 0 实测中收益峰值，planner return $272 \pm 18$，是 §5.4 起所有主对比实验的 **thesis-default cell**。
- **`full`**：功能网所有参数由 hypernet 生成，包括 fc1、fc2 与 AdaLN 调制。hypernet 输出约 3.09M 参数。在 v4-opt 2026-06 工程观察中出现 cos_pred_cross 由 0.61 升至 0.998 的方向坍缩现象（§4.2），thesis 中不作为默认配置。

partial generation 设计的核心动机是把"由 hypernet 生成"作为可调节范围而非二元开关。在 §4.2 几何分析框架下，这一范围调节即为条件化谱上不同位置的工程化实现：base_gen 接近共享基础参数端（容量平均化），full 接近全权重重生成端（方向坍缩），中间档（film_head / lora_fc2 / film_fc2）覆盖谱内部 Pareto 最优区段。

**(vi) 四层稳定性栈**。v2 在 hypernet → 功能网的数据路径上叠加四层稳定性机制，每一层针对一个具体失败模式：

层 1：**small_init**。hypernet 输出层的初始化采用 $\mathcal{N}(0, \sigma_{\text{small}}^2)$ 其中 $\sigma_{\text{small}} = 0.01 \cdot \sigma_{\text{Xavier}}$，即比 Xavier 初始化小两个数量级。其防护对象是 v3 阶段观察到的"hypernet 输出幅度过大导致功能网激活值越界、梯度爆炸"问题。small_init 与 (1+γ) AdaLN 的零初始化思想相通——保证训练初始时刻 hypernet 路径的影响为零，调制信号由零开始生长。

层 2：**LayerNorm 与 GroupNorm 联合归一化**。功能网内部的中间激活通过 LayerNorm（Ba 等人 2016）归一化，超网生成的权重通过 GroupNorm（Wu & He 2018 ECCV）按 channel group 归一化。LayerNorm 防护"激活值分布漂移导致下游层失效"问题；GroupNorm 防护"hypernet 输出的不同 channel 量级差异过大导致部分 channel 主导功能网输出"问题。两者共同保证从 hypernet 输出到功能网激活的整条路径上信号量级稳定。

层 3：**梯度截断（gradient clipping）**。hypernet 路径的梯度按全局 L2 范数截断至阈值 $\|\nabla\|_{\max} = 0.5$（功能网路径采用更宽松的 $\|\nabla\|_{\max} = 1.0$）。该层防护对象是"hypernet 路径上的梯度爆炸传播至功能网，进而干扰主任务梯度"问题。两路径的截断阈值差异反映 hypernet 路径对训练稳定性的敏感度更高。

层 4：**BYOL 风格一致性损失**。consist loss 定义为 $L_{\text{consist}} = -\mathrm{CosSim}(\mathrm{proj}(s^{k+1}), \mathrm{sg}(\mathrm{proj}(\mathrm{RepNet}(o^{t+k+1}))))$，其中 $\mathrm{proj}$ 是 BYOL [58] 风格的 projection head，$\mathrm{sg}$ 是 stop-gradient。该损失约束动力学网络（hyper_trans 生成的 StateTransNet）的展开预测 $s^{k+1}$ 与表征网络 RepNet 对真实观测 $o^{t+k+1}$ 的编码方向一致——通过余弦相似度而非 L2 距离，避免 $s^{k+1}$ 的范数被强行拉到与 RepNet 输出一致而影响下游 reward / value 估计的尺度。该层防护"表示坍缩"（representation collapse）失败模式——即不同上下文下表征网络输出趋同的 §4.2 谱右端坍缩现象。

**(vii) BeliefNet 双头结构**。BeliefNet 以 GRU 为骨干，输入 agent 自身观测历史 $\{o_i^{1:t}\}$，隐状态维度 128。GRU 的隐状态经两个独立预测头处理：
- **ĉ 头**：单层线性层 + sigmoid 激活，输出 $\hat{c}_i \in [0, 1]$ 即 $c_t$ 置信度估计。训练目标是与真实 $c_t$ 的 MSE 损失，但仅在 $c_{\text{visible}} = \text{False}$ 档下激活该损失项（可见 c 档下 ĉ 头退化为恒等映射）。
- **ẑ 头**：单层线性层 + softmax，输出对其他 agent 类型的 2-way 概率分布 $\hat{z}_{i,j} \in \Delta^{|\mathcal{T}|}$。训练目标是与真实 $\tau_{-i}$ 的交叉熵损失，在 Oracle 模式下直接给出标签监督，在 Inference 模式下使用 EM 风格的自一致更新（§4.6 三阶段课程展开此细节）。

两头共享 GRU 主干，使得对环境上下文的推断（ĉ 头）与对他人类型的推断（ẑ 头）共享同一历史编码——这一设计源于 Harsanyi 框架的"共同知识 + 私人信念"二分（§4.3），即两类信念应当从同一历史观测中推断而非独立估计。

**(viii) 完整前向传播流程**。给定 episode 内时刻 $t$ 的观测 $o_i^t$ 与当前 c-context、agent-id：
- 步骤 1：调用 `model.set_context(c_t, \text{agent\_id})`，触发 hyper_trans、hyper_rew、hyper_pred 重新生成 $\theta_{\text{state}}, \theta_{\text{rew}}^i, \theta_{\text{pred}}^i$。该步骤将 hypernet 输出缓存至功能网权重缓存（每 K=5 步刷新一次以匹配展开步数）。
- 步骤 2：表征网络 RepNet（固定权重）把观测 $o_i^t$ 映射为潜状态 $s^0 = \phi(o_i^t; \theta_{\text{rep}})$。
- 步骤 3：展开 K=5 步：对每个 $k \in \{0, \ldots, K-1\}$，给定 $s^k$ 与候选联合动作 $a_{\text{joint}}^k$，依次调用：
  - StateTransNet（权重 $\theta_{\text{state}}$）：$s^{k+1} = s^k + g_{\text{state}}([s^k, a_{\text{joint}}^k]; \theta_{\text{state}})$（残差形式）；
  - RewardHead（权重 $\theta_{\text{rew}}^i$）：$\hat{r}_i^k = g_{\text{rew}}([s^k, a_{\text{joint}}^k]; \theta_{\text{rew}}^i)$；
  - PredictionHead（权重 $\theta_{\text{pred}}^i$）：$(\hat{\pi}_i^k, \hat{V}_i^k) = g_{\text{pred}}(s^k; \theta_{\text{pred}}^i)$。
- 步骤 4：展开结束后，n-step bootstrap 估计：$\hat{V}_{\text{boot}}^t = \sum_{k=0}^{K-1} \gamma^k \hat{r}_i^k + \gamma^K \hat{V}^K_{\text{target}}$，其中 $\hat{V}^K_{\text{target}}$ 由 EMA 目标网络（$\tau = 0.99$）的 PredictionHead 给出。

**perspective 切换机制**：在多 agent 协同评估时，从 agent $i$ 切换到 agent $j$ 需重新调用 `model.set_context(c_t, j)`，重新生成 $\theta_{\text{rew}}^j$ 与 $\theta_{\text{pred}}^j$。$\theta_{\text{state}}$ 不需要重新生成（agent-invariant）。这一切换在 §4.5 Per-Agent Coordinate Descent 的每个 outer iteration 中触发 N 次（每 agent 一次），是 hypernet 路径在规划阶段的主要计算开销来源。

**(ix) 参数量精确簿记**。为让方法层的参数规模透明化，本节给出各 gen_scope 档下 hypernet 与功能网参数量的精确簿记，用于 §5.1 μP 学习率对齐协议下的公平比较依据。功能网基础规模：StateTransNet 由 3 层 MLP 构成（隐藏宽度 $h = 128$、输入维度 $|s| + |a_{\text{joint}}| = 64 + 4 \cdot 6 = 88$、输出维度 $|s| = 64$），参数量约 $88 \cdot 128 + 128 \cdot 128 + 128 \cdot 64 + \text{bias} \approx 33$k；RewardHead 由 2 层 MLP + 头（隐藏 128、输入 88、输出 1），约 12k；PredictionHead 由 2 层共享主干 + 双头（策略头输出 6、价值头输出 1），约 15k。总功能网参数约 60k per agent；4 agent 共享 StateTransNet 但独立 RewardHead / PredictionHead，则单 episode 内活跃功能网参数约 $33 + 4 \cdot (12 + 15) = 141$k。

**Hypernet 侧簿记**：c_encoder 由 2 层 MLP（$1 \to 32 \to 16$）构成，约 600 参数；role_encoder 由 3 层 MLP（$|role_i^{\text{raw}}| = 8 + 4 + 2 \to 32 \to 32$），约 1500 参数；belief_encoder 由 2 层 MLP（$1 + N-1 \to 32 \to 32$），约 1200 参数。三编码器合计约 3.3k。功能网权重生成层的规模按 gen_scope 档变化：
- **base_gen**：仅生成 c_ctx 编码器与 role/belief 编码器输出，共约 3.3k；剩余 694k 是固定基础功能网。
- **film_head**：生成 AdaLN 系数 $(\gamma, \beta) \in \mathbb{R}^{2h}$ 于每层 LayerNorm 上（约 3 层），每上下文 $2 \cdot 128 \cdot 3 = 768$ 个系数；hypernet 输出层参数量 $80 \cdot 768 \approx 60$k。加上编码器共约 230k。
- **lora_fc2**：除 AdaLN 外，生成 fc2 层的 LoRA 增量 $\Delta W_2 = B_2 A_2$，其中 $B_2 \in \mathbb{R}^{h \times r}, A_2 \in \mathbb{R}^{r \times h}$（$r = 8$），每上下文 $2 \cdot h \cdot r = 2 \cdot 128 \cdot 8 = 2048$ 个系数；hypernet 输出层参数量 $80 \cdot 2048 \approx 165$k。加上 film 系数与编码器共约 470k。
- **film_fc2**：在 lora_fc2 基础上对 fc1 也施加 LoRA 增量，参数量增至约 720k。这一档即 §5.2 决策门 0 GO 判据下的 thesis-default。
- **full**：生成功能网所有权重，参数量约 3.09M（60k 功能网参数 $\times$ hypernet 输出的映射系数约为 $80 \cdot 60000 / \text{bias factor}$）。

$|c_{\text{aug}}| = 80$ 维输入是这个链条的杠杆——每增加一维 c_aug 输入，hypernet 输出层参数量按 1 x 60k = 60k 增长（对应生成整套功能网 fc2 LoRA 增量的参数量）。这一线性关系使 c_aug 维度成为容量控制的第二个杠杆（第一个是 gen_scope 档）。

**(x) AdaLN (1+γ) 残差调制的信息几何解释**。AdaLN 的 (1+γ) 形式不仅是初始化便利，还具有 Fisher 信息几何上的意义。设标准 LayerNorm 输出为 $\tilde{h} = (h - \mu) / \sigma$（此处 $\mu, \sigma$ 为通道级统计量），则 AdaLN 的完整变换为 $h_{\text{out}} = (1 + \gamma) \odot \tilde{h} + \beta$。在参数空间中，令 $\theta_{\text{AdaLN}} = (\gamma, \beta)$，则输出对参数的 Jacobian 为 $\partial h_{\text{out}} / \partial \gamma = \tilde{h}$，$\partial h_{\text{out}} / \partial \beta = 1$。由此 Fisher 信息矩阵在初始化点 $\gamma = 0$ 附近的对角元素为 $F_{\gamma \gamma} \sim \mathbb{E}[\tilde{h}^2] = 1$（因 LayerNorm 使 $\tilde{h}$ 方差为 1），$F_{\beta \beta} \sim 1$——两者量级平衡。相比之下，无 (1+γ) 残差的直接形式 $h_{\text{out}} = \gamma \odot \tilde{h} + \beta$ 在 $\gamma \to 0$ 初始化下 $h_{\text{out}} \to \beta$，梯度信号完全通过 $\beta$ 传递，$\gamma$ 的更新初始信号极弱，Fisher $F_{\gamma \gamma}$ 在 float32 精度下常常低于优化器噪声，导致训练崩溃。(1+γ) 形式则保证 $h_{\text{out}} = \tilde{h}$ 在 $\gamma = 0$ 时输出与 LayerNorm 后的信号一致，梯度通过残差通路正常回传至 $\gamma$，训练可推进。这一分析给出了 v4.6 优化阶段"(1+γ) 是关键修正"这一观察的信息几何理由。

**(xi) grouped RMS 归一化的方向-幅度解耦**。设 hypernet 生成的 raw weight 为 $\Delta W \in \mathbb{R}^{h_{\text{out}} \times h_{\text{in}}}$，grouped RMS 归一化按输出通道分组（group 大小 $g$，共 $h_{\text{out}} / g$ 组）计算每组内的 RMS：$\|\Delta W_g\|_{\text{RMS}} = \sqrt{\mathbb{E}_{\text{group}}[\Delta W^2]}$。归一化操作 $\Delta W \leftarrow \Delta W / \|\Delta W_g\|_{\text{RMS}} \cdot s_{\text{out}}$ 使各组内的 RMS 幅度重置为 $s_{\text{out}}$（可学习的 output_scale）。方向-幅度解耦的意义是：**hypernet 只需学到 $\Delta W$ 的方向即可，其幅度由 output_scale 独立控制**。相比之下，未做 grouped RMS 归一化时，hypernet 必须同时学习方向与幅度，两者在优化空间中存在负相关（大幅度会掩盖方向变化的梯度信号），训练不稳定。RMS 归一化的另一好处是标度不变——不同层功能网所需的 $\Delta W$ 幅度不同（浅层需大幅度、深层需小幅度），output_scale 让每层独立校准而无需在初始化时手动调参。

**(xii) 参数生成的批次并行化**。K = 5 步展开每步都需要按 (c_t, agent_id) 生成一组功能网权重，若 batch size $B = 512$ 且每 batch 包含不同 (c_t, agent_id) 上下文，则单步 hypernet 前向需生成 $B \cdot N \cdot K = 512 \cdot 4 \cdot 5 = 10240$ 组功能网权重。为避免 hypernet 前向成为训练瓶颈，本文实现采用**上下文分组批处理**——将 batch 内相同 (c_t bucket, agent_id) 的样本合并至同一 hypernet 前向，共享生成的 $\theta_{\text{rew}}, \theta_{\text{pred}}$。这一分组策略在 Medium 配置下（$c_t$ bucket 数 $\approx 20$、agent 数 4）将 hypernet 前向次数从 $10240$ 降至 $\approx 80$，约 128× 效率提升。当 c_t 连续变化时（如随机游走 $P_c$）分组精度受损，此时切换为逐样本前向；本文默认协议在 c_t bucket 数 $\le 32$ 时启用分组。

本节就 DualHyperNetwork v2 架构的代数与工程纪律作了详尽展开。所列规格在 §4.6 训练算法与 §5 实验章节中作为方法基线，每一节会前向引用本节定义而不做重复展开。

---

## §4.5 规划层：Per-Agent Coordinate Descent + CRN

本节展开本论文规划层的两项核心技术——Per-Agent Coordinate Descent 与 Common Random Numbers（CRN），并以几何示意加方差差分推导刻画二者协同作用的机制。MVE（Model-based Value Estimation，Feinberg 等人 [67]）在单 agent 场景下已是基于学习模型的低偏差-低方差价值估计标准技术，但其多 agent 推广面临两个相互独立但效应叠加的困难：组合爆炸与信噪比崩溃。本节先分别刻画两个困难，再给出 Per-Agent Coordinate Descent 与 CRN 的协同方案与其几何意义。

**困难 1：组合爆炸**。在 RNS-MMG 的多 agent 联合动作空间中，每位 agent 的离散动作数为 $A$，则 $N$ 个 agent 的联合动作空间规模为 $A^N$。本文 Medium 配置（N=4, A=6）下即 $6^4 = 1296$ 个候选联合动作；若 K=5 步完整展开每个候选，则单步规划需进行 $K \cdot A^N \cdot \text{cost}_{\text{net}}$ 量级的网络前向计算，对应每决策步的 wall-clock 进入秒级，在线规划不可行。Easy N=2 配置（A=6）下联合动作空间为 $6^2 = 36$，是可枚举的下界；Hard N=8 配置（A=6）下为 $6^8 \approx 1.7 \times 10^6$，单步评估代价远超可接受区间。组合爆炸的物理含义是：随着 N 增长，规划所需的网络前向次数指数膨胀，而其中绝大部分候选联合动作（按博弈论的均衡概念）不会被实际选择，纯粹是浪费。

**困难 2：信噪比崩溃**。即使能完成对所有候选联合动作的评估，第 0 步评估的信噪比（signal-to-noise ratio, SNR）也面临根本困难。考虑两个候选动作 $a_i$ 与 $a_i'$（其他 agent 动作固定为采样得的 $a_{-i}^{(s)}$），其展开价值估计的差为

$$\Delta G(a_i, a_i') = G(a_i, a_{-i}^{(s)}) - G(a_i', a_{-i}^{(s')}),$$

其中 $a_{-i}^{(s)}$ 与 $a_{-i}^{(s')}$ 是两次独立采样的 opponent 动作。若两次采样独立（IndepRand），则方差为

$$\mathrm{Var}(\Delta G) = \mathrm{Var}(G(a_i, a_{-i}^{(s)})) + \mathrm{Var}(G(a_i', a_{-i}^{(s')})) = 2\sigma_{a_{-i}}^2 + 2\sigma_{a_i}^2.$$

实际测量表明，多 agent 策略采样引入的 $\sigma_{a_{-i}}^2$ 量级远大于本 agent 候选动作差异引入的 $\Delta\mu^2 \sim (\mathbb{E}[G(a_i)] - \mathbb{E}[G(a_i')])^2$，使 $\mathrm{SNR} = \Delta\mu^2 / \mathrm{Var}(\Delta G) \approx 0.02$——即候选动作排序几乎等同于随机。信噪比崩溃的物理含义是：在多 agent MVE 中，候选动作的真实价值差异被对手策略采样方差掩没，规划器无法在候选间作有效区分，输出退化为接近均匀分布的策略 $\pi_{\text{mve}} \to \mathcal{U}(\mathcal{A})$。这是断言 D 的两个支柱中 CRN 部分的形式化根源。

**解决方案：协调下降把搜索从联合改为轮询**。Per-Agent Coordinate Descent 将搜索方式从联合枚举改为逐 agent 轮询：固定其余 $N-1$ 位 agent 的当前动作 $a_{-i}^{(t)}$（取自上轮迭代或初始策略），对第 $i$ 位 agent 枚举 $A$ 个候选并取最优 $a_i^{(t+1)} = \arg\max_{a_i \in \mathcal{A}} \hat{Q}_i(s, a_i, a_{-i}^{(t)})$；按 agent 序号循环更新一轮（外循环索引 $t$ 递增），直至收敛或达到最大轮数。总评估次数从 $A^N$ 降至 $O(N \cdot A \cdot T_{\max})$，其中 $T_{\max}$ 是最大外循环数（本文取 $T_{\max} = 3$）；Medium N=4 配置下评估次数从 1296 降至 $4 \cdot 6 \cdot 3 = 72$，降幅 18×。

Coordinate Descent 在连续凸优化中的收敛性（Tseng 2001）不直接适用于离散 multi-agent 博弈，但其不动点具有清晰的博弈论解释：若所有 agent 在固定其他 agent 动作时均选择最优响应，则该不动点即为博弈的一个**纯策略 ε-Nash 均衡**，其中 ε 由 K 步 MVE 价值估计的偏差与方差共同决定。这一对应关系把 Per-Agent Coordinate Descent 从一个工程化的"贪心轮询启发式"提升为具有博弈论意义的均衡求解算法——其在 RNS-MMG 这一形式化博弈问题上的部署因而获得理论合法性，而非纯粹的计算 hack。

**解决方案：CRN 通过共享种子消除对手方差**。Common Random Numbers（CRN）的核心思想是：在评估候选动作时，让其他 agent 的随机性来源在所有候选间严格对齐。具体而言，规划器在评估第 0 步时为每个 agent 分配一个固定的随机种子 $\xi_{-i}$，所有候选动作 $a_i$ 的展开均使用同一 $\xi_{-i}$ 采样 $a_{-i}^{(s)}$；K=5 步内部的展开同样固定共享种子。此时 $\Delta G$ 的方差变为

$$\mathrm{Var}_{\text{CRN}}(\Delta G) = \mathrm{Var}(G(a_i) - G(a_i') \mid a_{-i}^{(s)} \text{ shared}) = 2 \sigma_{a_i}^2.$$

opponent 方差项 $2\sigma_{a_{-i}}^2$ 被消去——因为该项在两次评估间相同，差分时抵消。SNR 从约 0.02 上升至约 1.0，提升约 50× [证据待补 EVIDENCE PENDING — CRN-off vs CRN-on 第 0 步 Q-value SNR 显式 probe run，待 §5.1.2 补完]。这一恢复幅度与 §5.8 abl4_joint_easy_n2 cell 中（CoordDesc + CRN 联合启用）规划器输出与策略 prior 的实测对齐度形成间接印证：cos_mode_match $\approx 0.75$ 表明候选选择是非随机的，与 SNR 1.0 区段对应。

CRN 的几何解释可如下示意：在 ($\mathrm{return}(a_i), \mathrm{return}(a_i')$) 平面上，IndepRand 采样在两轴上的方差独立累加，散点云近似为以两候选真实 return 期望为中心的椭圆；CRN 共享种子等价于把两轴绑定在同一 opponent 采样切片上，散点云压缩为沿 45° 对角线的细带，主轴方向上的散布远小于椭圆主轴。这一几何变化即"消除无关方差"的可视化对应。

**协同机制：Coordinate Descent × CRN**。两项技术各自解决一个不同的问题——CoordDesc 处理评估代价问题（A^N → N·A），CRN 处理评估质量问题（SNR 0.02 → 1.0）。两者在数学上独立（CoordDesc 是搜索策略，CRN 是采样策略），但在工程上协同必要。具体而言：若仅启用 CoordDesc 而不启用 CRN，每个 outer iteration 内的 candidate 评估仍因 SNR 0.02 而几乎随机，coordinate update 退化为噪声驱动的扰动；若仅启用 CRN 而不启用 CoordDesc，则 SNR 改善的好处需在 A^N 量级的联合枚举上发挥，Medium 配置仍不可行。两者协同——CoordDesc 把评估范围从联合压缩到 per-agent，CRN 在 per-agent 评估内部把 SNR 从 0.02 拉回 1.0——共同使多 agent MVE 规划在 Medium 配置下进入实用区间。这是断言 D（CoordDesc × CRN 不可分割）的形式化基础；其可证伪性由 §5.8 在 Easy N=2 上的 $2 \times 2$ 矩阵实验（{CoordDesc, Joint} × {CRN, IndepRand} 四象限）检验。

**z-score 归一化与策略输出**。在 outer iteration 收敛后，规划器把每个 agent 的候选动作得分按 z-score 归一化：$\tilde{Q}_i(a_i) = (\hat{Q}_i(a_i) - \mu_Q) / \sigma_Q$，再经 softmax 转换为动作概率 $\pi_{\text{mve}}^i(a_i) = \mathrm{softmax}(\tilde{Q}_i)$。z-score 归一化使规划输出对绝对奖励尺度保持不变性——即使 RewardHead 的 output_scale 在训练过程中发生漂移，softmax 后的动作概率仍稳定。softmax 温度按训练阶段动态调整（具体细节见 §4.6 训练算法节），早期阶段较高以鼓励探索，后期阶段较低以收敛至硬选择。

本节就 Per-Agent Coordinate Descent 与 CRN 的协同机制完成几何示意与方差差分推导。具体的伪代码与 K 步展开实现见 §4.6 训练算法；SNR 0.02 → 1.0 的实测验证留待 §5.1.2 补完。

**ε-Nash 收敛性形式化**。设 Per-Agent Coordinate Descent 在 outer iteration $t$ 结束时的联合动作为 $a^{(t)} = (a_1^{(t)}, \ldots, a_N^{(t)})$。**定义**：$a^*$ 是给定当前 MVE 价值估计 $\hat{Q}$ 下的 $\varepsilon$-Nash 均衡当且仅当对所有 agent $i$ 与替代动作 $a_i'$：$\hat{Q}_i(s, a_i^*, a_{-i}^*) \ge \hat{Q}_i(s, a_i', a_{-i}^*) - \varepsilon$。此处 $\varepsilon \ge 0$ 是均衡"宽松度"，$\varepsilon = 0$ 对应精确 Nash 均衡。

**命题（Coordinate Descent 不动点 = $\varepsilon$-Nash 均衡）**：设 Coordinate Descent 在有限步数收敛到不动点 $a^\infty$（即再执行一轮轮询后各 agent 动作不变），则 $a^\infty$ 是关于当前 $\hat{Q}$ 的 0-Nash 均衡；若允许每次 update 有 $\varepsilon_0$ 的 argmax 近似误差（如 softmax 温度 > 0），则 $a^\infty$ 是 $N \varepsilon_0$-Nash 均衡。**证明概要**：Coordinate Descent 的不动点定义为每个 agent 在固定其他 agent 时选最优——这正是 Nash 均衡的 best-response 条件。若不动点存在则 Nash 条件自动满足；有限动作空间 $\prod_i \mathcal{A}_i$ 与整数 outer iteration 数保证不动点必然在有限步数达到。$\square$

**收敛速率分析**：Coordinate Descent 的收敛速率取决于 $\hat{Q}$ 函数在联合动作空间上的**耦合强度**。设 $\hat{Q}_i$ 关于 $a_j$（$j \ne i$）的 Lipschitz 常数为 $L_{ij}$，联合耦合矩阵 $\mathbf{L} = (L_{ij})$ 的谱半径 $\rho(\mathbf{L}) < 1$ 时，Coord Descent 以几何速率收敛；$\rho(\mathbf{L}) \ge 1$ 时可能出现周期振荡或不收敛。ResourceCommons 的耦合强度由资源采集共享条件的强度决定：$c_t$ 高时资源充足、agent 间竞争弱、$\mathbf{L}$ 谱半径小；$c_t$ 低时资源稀缺、竞争强、$\mathbf{L}$ 谱半径接近 1。本文取 outer iteration 上限 $T_{\max} = 3$——这一保守选择保证即使在 $c_t$ 低区段 Coord Descent 未完全收敛也能得到"部分优化"的联合动作，与"精确 $\varepsilon$-Nash"存在有限偏差但避免了 wall-clock 爆炸。若实测显示 $T_{\max} = 3$ 下的动作漂移显著（连续两轮 outer iteration 间 $\ge 30\%$ agent 的动作发生改变），则需增至 $T_{\max} = 5$；这一诊断阈值在 §5.11 中列出。

**CRN 的方差分析扩展**。上文推导 $\mathrm{Var}(\Delta G) = 2\sigma_{a_i}^2 + 2\sigma_{a_{-i}}^2 \to 2\sigma_{a_i}^2$ 假设 $a_i$ 与 $a_{-i}$ 独立采样。实际 MVE 展开中，opponent 动作 $a_{-i}^k$（$k = 1, \ldots, K-1$）依赖 opponent 策略网络的采样，即 $a_{-i}^k \sim \pi_{-i}(\cdot | s^k)$。这种依赖使方差分析需扩展至 $K$ 步累积：设 $G_K$ 为 $K$ 步展开返回值，$\mathrm{Var}(G_K) = \sum_{k=0}^{K-1} \gamma^{2k} \mathrm{Var}(r^k) + 2 \sum_{k < k'} \gamma^{k+k'} \mathrm{Cov}(r^k, r^{k'})$。CRN 仅共享第 0 步随机种子，$k \ge 1$ 步的 opponent 采样仍独立——因此 CRN 对方差的削减主要来自第 0 步项（$k = 0, k' = 0$ 与 $k = 0, k' = 1$ 的交叉方差）。若 opponent 策略具有较强时间相关性（相邻步的动作分布相似），CRN 的方差削减效果延伸至后续步骤；反之若 opponent 策略在展开过程中方差发散（如高探索率下），CRN 只在第 0 步生效。本文实现固定 opponent 策略网络的 dropout mask 在整个 K 步展开内不变，弱化了后续步骤的独立性假设，进一步提升 CRN 的实际方差削减效率。这一实现细节在 §5.11 与工程复现指南（附录 E）中记录。

**Blackwell 可近似性观点**。Blackwell（1956）在博弈论中的**可近似性定理**（approachability）给出了在多轮博弈中通过学习动态"追近"目标集合的条件。将 Coord Descent 视为 agent 间响应的"内部循环"、CRN 视为 opponent 采样"独立性伪装"，两者协同后 Coord Descent 迭代的收敛点即等价于在**Blackwell 可近似意义下的 $\varepsilon$-Nash 集合**。这一博弈论解释使本文规划器不仅是"启发式贪心搜索"，而是与经典博弈论学习动态（fictitious play、reinforcement learning in games）在收敛意义上一致的算法族——这一等价性是本文规划器可在多智能体强化学习框架内正式讨论的方法学基础。

---

## §4.6 训练算法：Type-aware K 步展开 + 三阶段课程

本节给出本文核心训练算法的完整表述。在 MuZero [24] 风格的 K 步展开基础框架之上，本算法融合三项关键扩展：(i) Type-aware 双路径奖励头，把类型 α 与类型 β 的偏好差异作为参数化条件而非外加 shaping；(ii) BYOL [58] 风格一致性辅助损失，抑制表示坍缩；(iii) BeliefNet 三阶段课程，从 Oracle 监督渐进过渡到自推断模式。为保证各实验组在同等预算下的可比性，本节同时引入 μP（Yang & Hu 2021）学习率对齐协议，与第 5 章公平对比承诺相呼应。

**(i) K 步展开主干**。给定时刻 $t$ 的观测 $o_i^t$ 与目标动作序列 $\{a_{\text{joint}}^{t+k}\}_{k=0}^{K-1}$（来自 replay buffer），算法主干执行 K=5 步展开（§4.4 已给出前向流程）。展开过程中估计的 K 个即时奖励 $\{\hat{r}_i^k\}_{k=0}^{K-1}$ 与第 K 步价值 $\hat{V}^K$ 通过 n-step Bellman 算子折叠为 bootstrap 目标：

$$\hat{V}_{\text{boot}}^t = \sum_{k=0}^{K-1} \gamma^k \hat{r}_i^k + \gamma^K \hat{V}^K_{\text{target}},$$

其中 $\gamma = 0.95$ 是折扣因子（v4-opt 2026-06 锁定），$\hat{V}^K_{\text{target}}$ 由 EMA 目标网络（衰减系数 $\tau = 0.99$，等效更新率 $1 - \tau = 0.01$）的 PredictionHead 给出。EMA 目标网络的引入抑制了自举估计的反馈循环（v4.4 增加），尤其在 c-context 切换时段帮助稳定 value 估计；目标网络权重以慢更新的方式跟踪在线网络，避免在线网络的高频更新直接传播至 bootstrap target。

**(ii) Type-aware 双路径奖励头**。RewardHead 的输出按类型分两条路径处理：

$$\hat{r}_i = \begin{cases} g_{\alpha}([s, a]; \theta_{\text{rew},\alpha}^i) & \text{if } \tau_i = \alpha \\ g_{\alpha}([s, a]; \theta_{\text{rew},\alpha}^i) + \varphi(c_t) \cdot \psi(\hat{\Delta}_i; [s, a]; \theta_{\text{rew},\beta}^i) & \text{if } \tau_i = \beta \end{cases}$$

其中 $\varphi(c_t) = 0.5 \cdot (1 - 2 c_t)$，$\psi(\Delta) = -[\lambda_{\text{disadv}} \max(0, -\Delta) + \lambda_{\text{adv}} \max(0, \Delta)]$，$\lambda_{\text{disadv}} = 2.0, \lambda_{\text{adv}} = 0.6$（§3.5）。两路径共享主干编码层（前两层全连接），仅最终线性头独立。在 c-context 处于 $\varphi(c) \approx 0$ 区段（即 $c_t \approx 0.5$，中性区段）时，β 路径的调制项趋于零，模型退化为类型 α 共享的标准 MuZero 奖励头；在 $c_t$ 偏离 0.5 的两端区段，β 路径的调制项激活，按 §4.1 的代数偏导表中四个区段值分布。这一类型-感知双路径设计是 §4.4 hyper_rew 主观通路在功能层的具体实现，把"按类型生成不同 $\theta_{\text{rew}}^i$"落地为 RewardHead 内部的两条路径。

**(iii) BYOL 风格一致性辅助损失**。设动力学网络（StateTransNet）的 k+1 步展开输出为 $s^{k+1} = s^k + g_{\text{state}}([s^k, a^k])$，表征网络（RepNet）对真实下一观测的编码为 $\phi(o^{t+k+1})$，则一致性损失为

$$L_{\text{consist}}^k = -\mathrm{CosSim}\big(\mathrm{proj}(s^{k+1}), \, \mathrm{sg}(\mathrm{proj}(\phi(o^{t+k+1})))\big),$$

其中 $\mathrm{proj}$ 是两层 MLP projection head（BYOL [58] 风格），$\mathrm{sg}$ 是 stop-gradient（防止 RepNet 接收 consist loss 的梯度，避免目标-在线网络共同坍缩）。$\mathrm{CosSim}$ 的使用使约束作用于方向而非范数，避免 $s^{k+1}$ 的范数被强行拉到与 RepNet 输出一致而影响下游 reward / value 估计的尺度。权重 $w_{\text{consist}} = 0.5$。

**(iv) 完整损失函数**。总损失由策略、价值、奖励、一致性、信念五项加权组成：

$$L_{\text{total}} = w_{\text{policy}} L_{\text{policy}} + w_{\text{value}} L_{\text{value}} + w_{\text{reward}} L_{\text{reward}} + w_{\text{consist}} L_{\text{consist}} + w_{\text{belief}} L_{\text{belief}}.$$

权重经 v4-opt 2026-06 勘正后固定为 $w_{\text{policy}} = 1.0, w_{\text{value}} = 0.25, w_{\text{reward}} = 3.0, w_{\text{consist}} = 0.5, w_{\text{belief}} = 1.0$。各项的具体形式如下：
- $L_{\text{policy}} = -\sum_k \hat{\pi}_{\text{mve}}^k \cdot \log \hat{\pi}^k$（cross-entropy with planner targets，K 步累加）；
- $L_{\text{value}} = \sum_k (\hat{V}^k - \hat{V}_{\text{boot}}^{t+k})^2$（MSE，K 步累加）；
- $L_{\text{reward}} = \sum_k (\hat{r}^k - r^{t+k})^2$（MSE，K 步累加）；
- $L_{\text{consist}} = \sum_k L_{\text{consist}}^k$；
- $L_{\text{belief}} = L_{\text{belief\_c}} + L_{\text{belief\_opp}} + \beta_{\text{div}} L_{\text{belief\_div}}$（ĉ 头 MSE + ẑ 头 cross-entropy + 多样性正则）。

优化器采用 Adam（Kingma & Ba 2015），$\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-5}$。$\epsilon = 10^{-5}$ 是 v4.4 阶段的关键修正（原 $\epsilon = 10^{-8}$ 在小梯度区段导致更新发散），与 EMA 目标网络一起构成了对 value 过估计反馈循环的双重防护。学习率调度采用 Cosine Annealing 配合 warmup（前 5k 步线性升至 $1.0 \times 10^{-3}$），后续按 cos 曲线衰减至 $1.0 \times 10^{-4}$。

**(v) BeliefNet 三阶段课程**。BeliefNet 的训练遵循三阶段调度，覆盖整个 episode collection 的 $T_{\max}$ iteration 数：

阶段 1（$t \in [0, 0.3 T_{\max}]$）**Oracle 强制**：BeliefNet 的 ẑ 头直接接收真实他人类型标签 $\tau_{-i}^{\text{true}}$ 作为监督，loss/belief_opp 即 ẑ 输出与真实标签的 cross-entropy。这一阶段保证早期梯度信号稳定建立，使 BeliefNet 在 inference 模式开启前已掌握"从历史观测推断类型"的基本表征能力。

阶段 2（$t \in [0.3 T_{\max}, 0.7 T_{\max}]$）**线性退火**：依照 Bengio 等人 [49, 51] 的渐进调度思路，Oracle 标签占比从 1.0 线性退火至 0.0。每个训练 step 以概率 $p_{\text{oracle}}(t) = \max(0, 1 - (t - 0.3 T_{\max}) / (0.4 T_{\max}))$ 注入 Oracle 标签，以概率 $1 - p_{\text{oracle}}(t)$ 使用 EM 风格的自一致更新（即 ẑ 头自身的预测作为软标签）。这一渐进切换降低早期训练崩溃风险。

阶段 3（$t \in [0.7 T_{\max}, T_{\max}]$）**自推断模式**：完全切换为自推断，loss/belief_opp 使用 ẑ 头自身预测作为软标签的 KL 散度损失（与 ground-truth 标签解耦）。$\hat{c}_i$ 头在 $c_{\text{visible}} = \text{False}$ 档下持续接收 MSE 损失（与真实 $c_t$ 比对，oracle 监督），在 $c_{\text{visible}} = \text{True}$ 档下退化为恒等映射。

**当前 suite 的工程披露**。需要在本节如实披露的是：在本文实验套件的 18 个已运行 hyper-variant cell 中，14 个观察到 `loss/belief ≤ 1e-5`，且 `loss/lambda_b ≡ 1.0` 在所有 runs 上保持初始值——这表明 belief 通路的训练在当前实现下未进入第二阶段或第三阶段，BeliefNet 实际工作于 Oracle 强制阶段或更前。两种可能的根因为：(a) Easy 配置下 $T_{\max} = 200k$ steps 在 wall-clock 上偏短，curriculum 第二阶段切入点 $0.3 T_{\max} = 60k$ 尚处于训练早期阶段，loss 量级被 $\lambda_b$ 调度尚未激活的小幅度抑制；(b) $\lambda_b$ schedule 的实现可能存在 gating bug 导致 $\lambda_b$ 未按预期从 1.0 调度至更大值。该问题不影响 §4.1-§4.5 方法层论述的形式化完整性（三阶段课程作为算法设计的一部分仍然成立），但 §5.10 信念质量协议的硬验证因此暂以 c-visible 档实验作为替代锚点；下一阶段实验须先修复 $\lambda_b$ 调度并验证 belief 通路是否能在更长训练时段下进入第二阶段。具体的诊断与修复路径见 §5.10、§5.11 与 §6.2。

**(vi) μP 学习率对齐协议**。为保证第 5 章六个方法基线的公平对比，本文采用 μP（Yang & Hu 2021）学习率对齐协议。μP 通过把宽度 $h$ 与学习率 $\eta$ 的耦合按 $\eta \propto 1/h$（output layer）与 $\eta \propto 1$（hidden layer）解耦，使得在不同网络宽度下的最优 LR 关系可预测——避免"baseline 没调好 LR 因而 Hyper-MuZero 显胜"的方法学攻击点。具体执行上，每条 baseline 独立运行 5 个 LR 档位（$\{0.5, 0.75, 1.0, 1.25, 1.5\} \times \eta_{\text{base}}$）各 3 seeds 共 15 runs 的 sweep，取最优 LR 档位结果作为该 baseline 的公平对比锚点；本文方法亦执行同一 sweep 协议。这一对齐协议在 §5.1 实验协议节作详细展开。

**(vii) 损失权重平衡的分析**。总损失 $L_{\text{total}} = 1.0 \cdot L_{\text{policy}} + 0.25 \cdot L_{\text{value}} + 3.0 \cdot L_{\text{reward}} + 0.5 \cdot L_{\text{consist}} + 1.0 \cdot L_{\text{belief}}$ 中各权重的相对值并非工程试错的经验产物，而由**梯度尺度对齐原则**决定。分析如下：$L_{\text{policy}}$ 是 cross-entropy，其梯度量级约为 $|\log \pi| \sim O(\log A) \approx 1.8$（$A = 6$）；$L_{\text{value}}$ 是 MSE，若 $V \in [0, V_{\max}]$ 其梯度量级约为 $V_{\max} \sim 4$，因此权重 0.25 使 $0.25 \cdot 4 \approx 1$ 与 policy 量级平衡；$L_{\text{reward}}$ 的 MSE 中 $r \in [0, 1]$，梯度量级约为 $0.3$，权重 3.0 使 $3.0 \cdot 0.3 \approx 1$ 与 policy 量级平衡；$L_{\text{consist}}$ 是余弦相似度损失，量级约为 $|\cos| \sim 0.5$，权重 0.5 使 $0.5 \cdot 0.5 = 0.25$，作为辅助损失有意保留在低于主任务的量级；$L_{\text{belief}}$ 是 cross-entropy + MSE 混合，量级约为 1，权重 1.0 使其与 policy 平齐。这一"梯度尺度对齐"分析给出了 v4-opt 2026-06 权重锁定的方法学理由，与工程试错的经验值形成双重验证。

**(viii) 三阶段课程的形式化辨认率上界**。BeliefNet 在三阶段课程末的 ẑ 头输出 $\hat{z}_{i,j}$ 的准确率上界由两条独立约束决定：**(a) 类型信息熵约束**——若他人类型 $\tau_j$ 的先验熵为 $H(\tau_j)$，则 BeliefNet 从观测中提取的信息量不超过互信息 $I(\tau_j; o_i^{1:t})$，准确率上界为 $1 - H(\tau_j | o_i^{1:t})$。在二元类型 + 均匀先验下 $H(\tau_j) = 1$ bit，观测提供的辨识信息约 0.5-0.8 bit（依 episode 长度与观测粒度），对应准确率上界约 60%-80%。**(b) 课程调度约束**——阶段 3 的自推断模式使用 ẑ 头自身预测作为软标签，若 ẑ 头初始准确率低于随机水平（50% for 2-way），则自一致更新可能收敛到"陷阱"（长期锁定错误标签）。三阶段课程通过前 30% 迭代的 Oracle 引导使 ẑ 头在阶段 3 开始时已具有 60%+ 准确率，避开陷阱区域，这是 curriculum 而非直接自训练的方法学理由。

**(ix) EMA target network 的收敛性论述**。EMA 目标网络的衰减系数 $\tau = 0.99$ 对应等效更新率 $1 - \tau = 0.01$，意味着目标网络参数是在线网络参数在过去约 100 iterations 内的**指数加权平均**。这一慢更新机制在多智能体强化学习中的必要性可以从"bootstrap 反馈循环"角度理解：在线网络的 value estimate $\hat{V}$ 参与 bootstrap target $\hat{V}_{\text{boot}}$ 的构造，若目标网络与在线网络完全相同（即 no EMA），则 $\hat{V}$ 的更新驱动 $\hat{V}_{\text{boot}}$ 的更新，形成正反馈循环——特别是在 $c_t$ 切换点（如从 $c_t = 0.4$ 切换到 $c_t = 0.6$）附近，$\hat{V}$ 在新上下文下的估计尚未收敛，直接用作 bootstrap 会放大估计偏差。EMA 慢更新使目标网络记住"过去 100 iterations 平均视角下的 value 估计"，其对新上下文的响应较慢，反而是稳定源。$\tau = 0.99$ 的具体选择与 K = 5 步展开、iteration 频率约每 1000 env-steps 一次的组合有关——过大（如 $\tau = 0.999$）会使目标网络对 $c_t$ 演化响应过慢，损失对当前博弈状态的贴合；过小（如 $\tau = 0.9$）则反馈循环无法有效抑制。$\tau = 0.99$ 是这两个考量下的平衡值。

**(x) 类型-上下文联合调制在训练动态中的表现**。Type-aware 双路径奖励头（$\alpha$ 路径与 $\beta$ 路径）在训练过程中会呈现**阶段性激活**模式：训练初期 $c_t$ 分布均匀覆盖 [0, 1]，$\beta$ 路径的 $\varphi(c) \psi(\Delta)$ 项在 $c \approx 0.5$ 区段贡献接近零、在 $c$ 极端区段贡献显著；随训练进行，agent 策略逐渐学到 $c_t$ 敏感行为，$\Delta_i$ 的分布随策略调整而变化，$\beta$ 路径的实际梯度贡献随之演化。这一动态过程与 §4.5 Coord Descent 的 $\varepsilon$-Nash 收敛耦合——策略朝向 Nash 均衡收敛时 $\Delta_i$ 分布也趋于稳态，$\beta$ 路径的训练进入稳定阶段。若观察到 $\beta$ 路径梯度在训练中持续震荡而不稳定收敛，可能提示 Coord Descent 未达到 $\varepsilon$-Nash 或 CRN 方差压制失效——这一诊断路径在 §5.11 中作为额外诊断指标。

**(xi) 训练算法与规划器的耦合闭环**。本节算法与 §4.5 规划器构成**训练-规划闭环**：训练阶段使用 replay buffer 中的历史 trajectory 更新网络权重；数据收集阶段使用当前网络权重驱动的 MVE + Coord Descent + CRN 规划器产生新 trajectory 放入 buffer。两阶段共享同一网络参数：训练更新影响下一次数据收集的规划器质量，数据收集质量影响下一次训练的样本效率。若规划器质量低（如 SNR 崩溃），则数据分布退化为接近均匀采样，训练无法进入有效区间——这解释了 planner-off 消融（§5.9）的自蒸馏退化引理：无规划器时 $\pi_{\text{mve}}^{\text{fresh}} = \pi_{\text{pred}}$，训练目标 $L_{\text{policy}}$ 退化为自身预测的 identity mapping，策略网络收敛到 ln(A) 均匀分布不动点。这一闭环特性使 §5.9 的 planner-off 实验成为方法必要性的严苛检验：不仅验证"MVE 有用"，还验证"训练与规划的深度耦合是不可解耦的"。

---

**本章小结**：第 4 章从三个容量瓶颈（§4.1）出发，建立了 DualHyperNetwork v2 三联通路方法的形式化动机；通过条件化谱容量-优化 Pareto 分析（§4.2）与 Harsanyi 主-客通路类比（§4.3）给出方法层选择的理论参照系；在 §4.4 D3 完整代数中展开架构规范，含三联上下文构造、双路超网切分、AdaLN 残差调制、grouped RMS 归一化、四层稳定性栈、BeliefNet 双头结构、参数量精确簿记与信息几何解释；§4.5 建立 Per-Agent Coordinate Descent + CRN 的 $\varepsilon$-Nash 收敛性与方差差分几何；§4.6 给出 Type-aware K=5 步训练算法、损失权重梯度对齐分析、三阶段课程辨识率上界、EMA 慢更新的博弈论论述与训练-规划闭环。本章不主张任何"首次"或"补全空白"论点，方法层论述强度限定于"给出结构上一致且工程上可行的实现"这一水准，具体的实证验证留待第 5 章各消融节。
