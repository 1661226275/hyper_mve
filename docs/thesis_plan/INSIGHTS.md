# INSIGHT 集合（硕士论文化筛选）

> 来源：Plan Mode 会话综合 16 份 doc digest + 用户 Q4/Q5 校准
>
> 原 15 条 INSIGHT 按"是否服务于全文逻辑自洽"重新分级：12 保留 + 3 降级。

---

## 保留：12 条（强化内部逻辑）

每条 INSIGHT 都直接服务于论文某节的论证链路，且不构成"开拓新方向"的宣称。

### I2 心理偏好不变 / 行为模式涌现

**陈述**：与 reward shaping 范式的根本分野。Hughes 2018 inequity aversion 家族把不平等厌恶作为可调奖励项，本文把它作为不可训练的偏好结构，只让 c_t 通过 φ(c) 动态调制其"强度"，让荒年剥削/丰年合作真正从环境动力学中涌现。

**部署位置**：§3.4 v3→v4 演化（解释为何砍除 β(c)）· §3.5 偏好层 · §4.3 方法论承诺 · §6.4 结语

**支撑证据**：Chapter4_1_Motivation_v4.md §4.1.5；Chapter3_Environment_v4.md v3→v4 砍除 β(c) 的演化

---

### I4 v3 → v4 演化叙事

**陈述**：v3 NonStationaryTag 的成功不是论文的累赘，而是其方法论身份的起点。v3 用零和阵营翻转证明了"超网络可以条件化博弈视角"，v4 把同一思路推进到混合动机偏好异质 —— "起源验证 → 结构升级"的双层叙事比"v4 独立成篇"更有说服力。

**部署位置**：§3.2 初步研究 · §5.3 迁移实验 · §6.4 结语 · §1.6 组织（显式提及 v3-v4 桥接）

**支撑证据**：毕设汇报_展示文档.md Capture Rate 75% vs 10%；v3 DESIGN_DOC_FINAL.md DualHyperNetwork 设计原型

---

### I5 CRN 训练可行性必要条件 + Self-Distillation 退化引理

**陈述**：CRN（Common Random Numbers）被定位为"多智能体 MVE 的可行性必要条件"，而非"锦上添花的方差减少技巧"。第 0 步 SNR 从 0.02 拉回 1.0 意味着没有 CRN 则 π_mve 退化为均匀；planner-off 自蒸馏退化引理证明这是整个闭环的不动点（∇L_policy = π̂ - π_tgt ≡ 0 时策略熵钉死在 ln(A)）。

**部署位置**：§4.5 规划层 · §5.8 消融 4 · §5.9 planner-off 演示 · §1.4 贡献 3

**支撑证据**：Chapter5_Planner_Training_v4.md §5.1.2 SNR≈0.02 实测；§5.9.1b 自蒸馏退化引理；Review_v4_TheoryAudit_2026-06.md §4.1

---

### I6 CoordDesc ↔ ε-Nash 均衡对应

**陈述**：Per-Agent Coordinate Descent 的不动点恰好对应 ε-纳什均衡 —— 把"工程上为了规模化压缩搜索空间"的 trick 重新升格为"与博弈论均衡概念在数学上对齐"的算法选择。

**部署位置**：§4.5 · §2.2 related work 把方法定位为博弈论一致的算法 · §6.1 贡献总结

**支撑证据**：Chapter5_Planner_Training_v4.md §5.3.2：「协调下降不动点恰恰对应于 ε-纳什均衡」

---

### I7 DualHyperNet ≡ C1/C2 结构同构

**陈述**：DualHyperNetwork 的双路设计（客观/主观）与 RNS-MMG 的 C1/C2 约束（物理同构/偏好调制）严格同构。这不是 engineers' decoupling，而是问题结构强加给方法的必然分解 —— 因此方法贡献的"必要性"有了来自问题设定本身的形式化辩护。

**部署位置**：§1.4 贡献 2 · §3.1 RNS-MMG 形式化的呼应 · §4.4 方法细节

**支撑证据**：Chapter1_Introduction_v3.md C1-C3 约束；Chapter4_Architecture_v4.md 客观/主观分路设计

---

### I9 Instantaneous Δ 设计的 Reward Scale 自然对齐

**陈述**：v3 的 cumulative Δ 可达 O(T·η_max)，迫使 Fehr-Schmidt 项需要复杂 normalization；v4 换成 instantaneous 后，物理项与 Fehr-Schmidt 项都落在 [0, 1.5] 内，任何 reward normalization 的工程麻烦都消失了。

**部署位置**：§3.5 环境偏好层 · §4.6 训练算法稳定性 · §5.11 失败诊断

**支撑证据**：Chapter3_Environment_v4.md instantaneous Δ；Chapter5_Planner_Training_v4.md §5.6.4 自然对齐

---

### I10 v4 砍除 β(c) 的纯净化

**陈述**：v4 砍除 v3 的 β(c) cooperative bonus，把合作激励完全交给类型 β 的偏好结构 —— 把 v3 混杂的"物理 reward shaping + 偏好层异质"净化为"物理严格自利同构 + 偏好显式异质 + 博弈涌现于交互"的三段式纯净设计。是命题 3.1 能成立的前提（否则 Gap(c) 单调就是定义内蕴而非涌现）。

**部署位置**：§3.4 v3→v4 演化 · §1.2 问题陈述 · §3.6 命题 3.1

**支撑证据**：Chapter3_Environment_v4.md v4 砍除 β(c) 的章节；Chapter4_1_Motivation_v4.md §4.1.5

---

### I11 Self Info 观测设定 ↔ Harsanyi own type + belief over others

**陈述**：Self Info 观测设定（自己类型可见、他人类型隐藏）不是任意选择，而是对应 Harsanyi 的"own type + belief over others"结构。把"部分可观测"从一般 POMDP 的 technicality，升级为博弈论结构性约束 —— BeliefNet 的存在因此不是工程方便，而是与 Harsanyi 框架的自然对应。

**部署位置**：§3.5 偏好层观测设定 · §4.3 Harsanyi 类比 · §4.4 BeliefNet

**支撑证据**：Chapter3_Environment_v4.md Self Info；Chapter4_Architecture_v4.md 三联上下文

---

### I12 预登记失败回退方案 = 论文规范性体现

**陈述**：四个可证伪断言的预登记失败回退方案，是这篇论文比一般方法论论文更接近实证科学规范的关键。每个断言都标注了"若 XYZ 则贡献降级为 ABC"的预案 —— 把 thesis 从"我们的方法很好"转换为"我们的方法可被否定也可被复现"，大幅提升评委对方法可信度的判断。

**部署位置**：§1.5 可证伪断言总览 · §5.1 实验协议 · 每个消融节的"失败回退"子节 · §6.4 结语

**支撑证据**：Chapter6_Experiments_v4.md 各节预登记；Hyper_MuZero_v4_Roadmap.md 失败回退表

---

### I13 Gap(c) 单调性是真涌现，非分析性

**陈述**：Gap(c) 单调性是从两条正交通道（资源时序动力学 × 类型 β 偏好结构）交互涌现的性质，而非定义内蕴 —— 这意味着"合作-竞争切换"在 ResourceCommons 中是真正的 emergent property，而非环境定义直接推得的 tautology。命题 3.1 的证明因此具有"实质性"而非"分析性"内容。

**部署位置**：§3.6 命题 3.1 · §1.1 引言定调 · §6.1 贡献总结

**支撑证据**：Chapter3_Environment_v4.md v4 证明依赖两通道无人为耦合的论述

---

### I14 类型梯度撕裂的代数硬化

**陈述**：DualHyperNetwork 的 Reward Gradient Quantification Table（类型 α 常数 1 vs 类型 β ∈ {0, 0.7, 1.3, 2.0}）把"类型梯度撕裂"这个抽象概念硬化为四个具体 (c, Δ) 区段下的数值表 —— 把消融 3 的钟形曲线从"我们希望看到"转换为"反向传播的代数必然"，大幅强化了断言 A 的理论根基。

**部署位置**：§4.1.1 类型梯度撕裂 · §5.7 消融 3 的预测依据 · §6.11 可视化

**支撑证据**：Chapter4_1_Motivation_v4.md §4.1.1 梯度表；Chapter4_Architecture_v4.md ∂R^β/∂u_i ∈ {0,0.7,1.3,2.0}

---

### I15 μP 学习率对齐协议消除"baseline 没调好"攻击

**陈述**：μP 学习率对齐协议（Yang & Hu 2021）的引入，把"baseline 没调好"这一最常见的方法学攻击点系统性消除。Hyper-MuZero 与所有 baseline 走同一个 LR sweep 协议（5 LR × 3 seeds → 选最优），是把工程公平性写进论文协议的少见做法，对评委的复现性问题是直接防御。

**部署位置**：§5.1 实验协议 · §1.4 贡献 4 实证 · §5.4 主对比 · 引言中关于公平性的承诺

**支撑证据**：Chapter6_Experiments_v4.md μP 对齐协议；Chapter5_Planner_Training_v4.md §5.10.2

---

## 降级：3 条（避免过度宣称）

按 Q4/Q5 校准——本论文为硕士毕业设计，不主张开拓新领域——以下 3 条 INSIGHT 改写或降级处理。

### ~~I1 Harsanyi 60 年首次架构对应~~ → 改写为"借用 Harsanyi 框架解释架构选择"

**原陈述**：把 Harsanyi 1967 的"共同知识/私人信念"二分直接对应到神经网络层级——这是 Harsanyi 定理在深度世界模型中 60 年内的首次架构性实现。

**降级理由**：硕士论文不主张"60 年内首次"这样的开拓性宣称。

**改写后部署**：仅在 §4.3 Harsanyi 类比节出现，措辞改为"本节将 DualHyperNet 双路结构与 Harsanyi 1967 共同知识/私人信念二分做类比，作为方法选择的理论参照系"。**不放 §1.4 贡献声明**。

---

### ~~I3 条件化谱"升级"研究问题~~ → 改写为"沿条件化谱进行系统性比较"

**原陈述**：把"是否用超网络"的二元争论升级为"per-context 容量在谱上何处开始得不偿失"的几何问题。

**降级理由**：硕士论文不需要"升级研究问题"这种范式宣言，描述本文的具体做法即可。

**改写后部署**：作为分析框架而非范式升级出现。§2.3 / §4.2 / §5.5 中都改用"本文沿条件化谱（Input → FiLM → LoRA → base_gen → full）对超网络生成范围做系统性比较"的措辞。FULL 方向坍缩 (0.61→0.998) 仍作为关键观察保留，但定性为"工程现象"而非"范式失败的证据"。

---

### ~~I8 BeliefNet Oracle 监督改进~~ → 合并到 I12 工程改进项

**原陈述**：v4 将 BeliefNet 对手类型推断从 v3 的动作预测改为 Oracle 2-way classification (α/β)，并配以三阶段课程 —— 收敛速度从 200k 步降到 50k 步。

**降级理由**：这是工程改进，不是 INSIGHT 级别的论点。

**改写后部署**：在 §4.4 BeliefNet 细节节作为常规工程改动陈述；不单列为 INSIGHT。

---

## 整体调性原则（来自 Q4/Q5 校准）

所有 INSIGHT 部署时遵循以下原则：

1. **禁止措辞**：首次/开拓/补全空白/60 年内首次/被长期忽视的关键子方向/方法论身份的核心宣言
2. **保留措辞**：本文给出/本文将……形式化/本文借用 X 框架解释/与 X 在结构上类似/本文承诺
3. **以解释替代宣称**：每个 INSIGHT 落地为"为什么这样做"的解释，而非"我们做了什么前无古人的事"
4. **可证伪绑定**：方法层 INSIGHT 都要绑定到 §5 的具体实验（特别是 INSIGHT 5/6/7/12/14）
