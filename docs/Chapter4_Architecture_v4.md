# 第四章 双路超网络架构与三联条件化(v4 完整版)

> **本章定位**:第三章定义了 ResourceCommons 这一异构偏好公地博弈环境,本章定义在该环境上工作的网络架构——**DualHyperNetwork v4**,通过 (type, belief, capability) 三联条件化的双路超网络解决类型梯度撕裂、信念稀释、角色平均化三类容量瓶颈。本章的核心论证(4.1 节)已经将 DualHyperNetwork 与 Harsanyi 不完全信息博弈、hypernetwork vs input conditioning 容量分配理论、Fehr-Schmidt 偏好梯度结构严格对应。
>
> **v4 关键演进总览**(相对 v3):
> 1. **4.1 motivation 重构**:从"主客解耦防止梯度撕裂"升级为基于"容量分配几何"的严格论证,锚定三个可证伪断言(A:类型梯度撕裂、B:信念专属容量、C:三联通路必要性);
> 2. **4.2 三联通路**:role_i 加入 **type_emb**(自己类型可见,Self Info 设定);
> 3. **4.2.3 BeliefNet**:对手类型推断头 $\hat{z}_{i,j}$ 从动作预测改为**类型 2 分类**;
> 4. **4.5 训练目标**:$\mathcal{L}_{\text{opp}}$ 从自监督动作预测改为**Oracle 类型监督交叉熵**,详细课程学习协议放在 Chapter 5.7;
> 5. **4.7 代码 Diff**:更新到 v4 实际改动清单。

---

## 4.1 设计原则:从类型梯度撕裂到容量分配的几何论证

> **本节是 Chapter 4 的 motivation 锚**,对应 Chapter 1.5 节贡献 2 的三个可证伪断言。完整的论证文本见独立文件 `Chapter4_1_Motivation_v4.md`,本节给出简明版概要。

### 4.1.1 RNS-MMG 下的核心架构挑战(概要)

在第三章 ResourceCommons 环境中,类型 α(纯自利,$R^\alpha_i = u_i$)与类型 β(Fehr-Schmidt + φ 调制,$R^\beta_i = u_i + \phi(c)\psi(\Delta_i)$)共存。计算两种类型对 $u_i$ 的偏导,得到下表(假设 $|\Delta| = 1$):

| 场景 | $\phi(c)$ | $\Delta$ 符号 | $\partial R^\beta/\partial u_i$ | $\partial R^\alpha/\partial u_i$ |
|---|---|---|---|---|
| 荒年 + 自己优势 | $+0.5$ | $+$ | $1 - 0.3 = 0.7$ | $1$ |
| 荒年 + 自己劣势 | $+0.5$ | $-$ | $1 + 1.0 = 2.0$ | $1$ |
| 丰年 + 自己优势 | $-0.5$ | $+$ | $1 + 0.3 = 1.3$ | $1$ |
| 丰年 + 自己劣势 | $-0.5$ | $-$ | $1 - 1.0 = 0$ | $1$ |

**关键观察**:类型 α 偏导恒为 +1,类型 β 偏导在 (c 区段, Δ 符号) 联合下取值从 0 到 2 不等。在**丰年 + 自己劣势**场景下,β 完全无激励增加自己采集量——单组共享 RewardHead 权重不可能同时满足 "β=0 ∧ α=1" 这一矛盾要求,反向传播被两类梯度撕裂为类型平均策略。

这一现象产生三类容量瓶颈,共享架构在 ResourceCommons 下不可避免:

- **瓶颈 1(类型梯度撕裂)**:共享 RewardHead 学到"平均偏导",不能精确刻画任一类型;
- **瓶颈 2(信念稀释)**:同类型不同信念的 agent 最优响应不同,共享网络容量在 belief 空间被平均分配;
- **瓶颈 3(角色平均化)**:异质 cap_i 使最优策略在 agent 间天然分化,共享 PredictionNet 无法 per-cap 定制。

### 4.1.2 解决路径:hypernetwork 替代 input conditioning(概要)

两种条件化方案在表达能力上等价(universal approximation),但**容量分配的几何不同**:

- **Input conditioning** $f_W([s, a, c_{\text{ctx}}, \text{role}_i, \text{belief}_i])$:共享 $W$ 必须同时编码所有 context 的行为,容量被平均分配;
- **Hypernetwork** $f_{h_\Theta(c, r, b)}(s, a)$:每个 context 生成专属 $\theta$,容量是 per-context 的。

当不同 context 对应的最优函数结构性差异大(如本文类型 α 与类型 β 在 c 低区段偏导差从 +1 到 +2 跨度大),hypernetwork 的样本效率与泛化能力显著优于 input conditioning。这是 Ha 2017、CAVIA 2019、FiLM 2018 等元学习文献反复观察到的现象。

### 4.1.3 三层 Motivation 结构

| 层级 | 容量瓶颈 | Context 通路 | 可证伪断言 |
|---|---|---|---|
| **主层** | 瓶颈 1:类型梯度撕裂 | role_i 中的 **type_emb** | 断言 A(类型异质性钟形曲线) |
| **辅层 1** | 瓶颈 2:信念稀释 | belief_i 中的 $\hat{c}, \hat{z}$ | 断言 B(零样本泛化) |
| **辅层 2** | 瓶颈 3:角色平均化 | role_i 中的 **cap_emb** | 断言 C 分量(去除 cap 通路) |

### 4.1.4 Harsanyi 不完全信息博弈的几何对应

| Harsanyi 框架 | 本文实现 | 架构组件 |
|---|---|---|
| Common Knowledge(物理规律) | 客观通路 hyper_trans | 输入 = c_ctx,输出 = θ_state |
| Own Type(自己的类型/能力) | role_i 通路 | type_emb + cap_emb + id_emb,Self Info |
| Beliefs over Others' Types | belief_i 通路 | $\hat{c}_i^t$ + Pool($\hat{z}_{i,j}^t$),后验推断 |
| Type-Conditional Best Response | 主观通路 hyper_rew / hyper_pred | 接收完整三联输入 |

这一对应在 Harsanyi 1967 之后的 60 年里**首次在深度世界模型中被显式实现**。

### 4.1.5 方法论原则:偏好不变,行为涌现

agent 心理偏好结构在训练过程中保持不变(Fehr-Schmidt 结构 + 固定参数),行为模式(荒年剥削 / 丰年合作)作为最优策略**涌现**于环境动力学与偏好结构的交互。这一原则严格区分本文方法与 v1 reward shaping、v3 β(c) 协同加成两条路线。

---

## 4.2 三联通路:c_ctx、role_i、belief_i

> **本节的工程目标**:把 Chapter 4.1.3 节的"三层 motivation + 三 context 通路"转换为具体的张量编码方案。

### 4.2.1 客观通路:c_ctx(共同知识)

**输入**:共享上下文 $c_t \in [0, 1]$(由环境提供,或在"$c_t$ 隐藏模式"下由 BeliefNet 推断的 $\hat{c}_i^t$ 替代)。

**编码**:简单线性投影 + 非线性。

$$\text{c\_ctx} = \text{MLP}_{\text{c}}\bigl(c_t\bigr) \in \mathbb{R}^{d_c}$$

默认 $d_c = 16$。MLP 结构:`[1 → 32 → 32 → 16] + ReLU + LayerNorm`。

**关键性质**:c_ctx 对所有 agent 相同——这是 Harsanyi 框架中"共同知识"的实现。仅 c_ctx 进入客观通路 hyper_trans,确保资源场动力学的预测在所有 agent 间共享(满足约束 C1 物理转移上下文不变)。

### 4.2.2 角色通路:role_i(自己的类型 + 能力 + 身份,Self Info)

**v4 关键改动**:role_i 加入 **type_emb**(自己类型),严格对应 Chapter 3.7 节 Self Info 设定。

**输入**:
- agent ID:$i \in \{0, 1, \ldots, N-1\}$
- 类型:$\tau_i \in \{\alpha, \beta\}$,one-hot 编码 $\in \{0,1\}^2$
- 能力向量:$\mathbf{cap}_i = (\eta_i, \phi^{\text{fov}}_i, \nu_i, \zeta_i) \in \mathbb{R}^4$

**编码**:

$$\text{id\_emb}_i = \text{Embedding}_{\text{id}}(i) \in \mathbb{R}^{d_{\text{id}}}$$

$$\text{type\_emb}_i = \text{Embedding}_{\text{type}}(\tau_i) \in \mathbb{R}^{d_{\text{type}}} \quad \text{(v4 新增)}$$

$$\text{cap\_emb}_i = \text{MLP}_{\text{cap}}(\mathbf{cap}_i) \in \mathbb{R}^{d_{\text{cap}}}$$

$$\text{role}_i = \text{Concat}\bigl[\text{id\_emb}_i,\;\text{type\_emb}_i,\;\text{cap\_emb}_i\bigr] \in \mathbb{R}^{d_r}$$

其中 $d_r = d_{\text{id}} + d_{\text{type}} + d_{\text{cap}}$。默认 $d_{\text{id}} = 8, d_{\text{type}} = 8, d_{\text{cap}} = 16$,$d_r = 32$。

**关键性质**:

- **type_emb 的核心作用**:这是 4.1.3 节主层(瓶颈 1 类型梯度撕裂)的架构实现。type_emb_α 与 type_emb_β 是两个完全独立的可学习向量,通过 hyper_rew/hyper_pred 时分别生成两套差异巨大的 θ_rew / θ_pred,从根本上消除共享 RewardHead 的类型撕裂;
- **Self Info 严格性**:role_i 只包含 agent $i$ **自己**的类型与能力——这是 Chapter 3.7.4 节 Self Info 设定的硬约束。他人的类型不进入 role_i,而由 belief_i 通路(下面 4.2.3)推断;
- **episode 内固定**:类型与能力在 episode 内不变,故 role_i 也在 episode 内固定。这意味着每个 episode 开始时计算一次 role_i 并缓存即可,无需每步重算。

### 4.2.3 信念通路:belief_i(对环境与他人的推断)

**输入**:agent $i$ 的观测历史 $h_i^{0:t} = (o_i^0, a^0, o_i^1, a^1, \ldots, o_i^t)$。

**BeliefNet 主干**(GRU-based):

$$b_i^t = \text{GRU}_{\text{belief}}\bigl(\text{Encoder}(o_i^t),\;b_i^{t-1}\bigr) \in \mathbb{R}^{d_b^{\text{hidden}}}$$

默认 $d_b^{\text{hidden}} = 128$。

**两个 belief head**:

**Head 1:$c_t$ 推断**($\hat{c}$ head)

$$\hat{c}_i^t = \sigma\bigl(\text{MLP}_{\hat{c}}(b_i^t)\bigr) \in [0, 1]$$

输出为 scalar(因为 $c_t$ 本身是 scalar)。MSE 监督训练。

**Head 2:对手类型推断**($\hat{z}$ head,v4 关键改动)

$$\hat{z}_{i,j}^t = \text{softmax}\bigl(\text{MLP}_{\hat{z}}(b_i^t,\;j)\bigr) \in \Delta^2 \quad (\text{类型 α / 类型 β 概率})$$

**v4 与 v3 的关键差异**:
- **v3 中** $\hat{z}_{i,j}$ 是对手未来动作的预测(自监督,$d_z$ 维向量);
- **v4 中** $\hat{z}_{i,j}$ 是对手 $j$ 的类型 2 分类概率(Oracle 监督,2 维 softmax)。

这一改动的理由(详见 Chapter 4.1.3 节辅层 1 论证 + Chapter 5.6 训练讨论):

1. 类型异质成立时(v4 新增设定),"对手是 α 还是 β"是有意义的离散推断目标;
2. 监督训练比自监督训练信号更强、收敛更快;
3. 类型 2 分类输出与 type_emb 维度可对齐,便于后续 belief_i 的 pooling 操作。

**MLP_$\hat{z}$ 的输入处理**:$\text{MLP}_{\hat{z}}(b_i^t, j)$ 实际实现中,以 (b_i^t concat agent_j 的 id_emb) 为输入,这样同一 $b_i^t$ 可以对不同的 $j$ 输出不同的 $\hat{z}$。

**信念向量的最终形式**:

$$\text{belief}_i^t = \text{Concat}\bigl[\hat{c}_i^t,\;\text{Pool}\bigl(\{\hat{z}_{i,j}^t\}_{j \neq i}\bigr)\bigr] \in \mathbb{R}^{d_b}$$

其中 Pool 操作对 $N - 1$ 个对手类型概率做 set-invariant 聚合(默认 mean pooling)。

**v4 belief 维度估算**:$d_b = 1 + 2 = 3$(1 维 $\hat{c}$ + 2 维 pooled $\hat{z}$)。这看起来很小,但实际工程中我们会先把 $\hat{c}$ 与 pooled $\hat{z}$ 各自经过一个小 MLP 投影到 $d_b^{\text{proj}} = 16$,使 belief_i 总维度更适合作为 hypernetwork 输入。

**关键性质**:

- **BeliefNet 在所有 agent 间共享权重**:即同一个 GRU + 同一组 head 参数,但 N 个 agent 各自维护独立的 $b_i^t$ 隐状态。这种"网络共享、状态独立"的设计与 RNN 多智能体模型(如 R2D2-MA)一致;
- **课程学习阶段控制**:训练时 belief_i 的具体构造方式按课程阶段变化——阶段 1 用 Oracle type 替代 $\hat{z}$,阶段 3 用纯推断。详见 Chapter 5.7。

### 4.2.4 三通路汇总:广义条件向量

主观通路 hyper_rew / hyper_pred 接收的完整条件向量为:

$$\text{ctx}_i = \text{Concat}\bigl[\text{c\_ctx},\;\text{role}_i,\;\text{belief}_i\bigr] \in \mathbb{R}^{d_c + d_r + d_b^{\text{proj} \cdot 2}}$$

默认 $d_c = 16, d_r = 32, d_b^{\text{proj} \cdot 2} = 32$,总维度 80 维。

**客观通路 hyper_trans 仅接收 c_ctx**(共同知识),严格对应 Harsanyi 框架。

---

## 4.3 DualHyperNetwork v4

### 4.3.1 数据流总图

```mermaid
flowchart TB
    OBS["观测 o_i^t<br/>(含自己 τ_i, cap_i)"]
    HIST["观测历史 h_i^{0:t}"]
    CT["共享上下文 c_t<br/>(或 隐藏模式下不可见)"]
    
    REP["RepresentationNet"]
    BN["BeliefNet (GRU)"]
    HC["Head ĉ"]
    HZ["Head ẑ_{i,j}<br/>(v4: 2分类)"]
    
    CENC["c_encoder"]
    RENC["role_encoder<br/>(含 type_emb v4)"]
    BENC["belief_encoder"]
    
    CCTX["c_ctx"]
    ROLE["role_i"]
    BELIEF["belief_i"]
    
    HTRANS["hyper_trans<br/>(客观通路)"]
    HREW["hyper_rew<br/>(主观通路)"]
    HPRED["hyper_pred<br/>(主观通路)"]
    
    TSTATE["θ_state<br/>(共享于所有 agent)"]
    TREW["θ_rew^i<br/>(per-agent)"]
    TPRED["θ_pred^i<br/>(per-agent)"]
    
    STN["StateTransNet<br/>(s, a) → s'"]
    RH["RewardHead<br/>(s, a) → r_i"]
    PN["PredictionNet<br/>s → (π_i, v_i)"]
    
    OBS --> REP
    REP -->|"s^0"| STN
    HIST --> BN
    BN --> HC
    BN --> HZ
    HC -->|"ĉ_i"| BENC
    HZ -->|"Pool(ẑ_{i,j})"| BENC
    
    CT --> CENC
    OBS -->|"τ_i, cap_i, id_i"| RENC
    
    CENC -->|""| CCTX
    RENC -->|""| ROLE
    BENC -->|""| BELIEF
    
    CCTX --> HTRANS
    CCTX --> HREW
    ROLE --> HREW
    BELIEF --> HREW
    CCTX --> HPRED
    ROLE --> HPRED
    BELIEF --> HPRED
    
    HTRANS -->|"生成"| TSTATE
    HREW -->|"生成"| TREW
    HPRED -->|"生成"| TPRED
    
    TSTATE -.->|"权重"| STN
    TREW -.->|"权重"| RH
    TPRED -.->|"权重"| PN
    
    style CCTX fill:#e3f2fd,stroke:#1976D2
    style ROLE fill:#fff3e0,stroke:#FF9800
    style BELIEF fill:#f3e5f5,stroke:#9C27B0
    style TSTATE fill:#e8f5e9,stroke:#4CAF50
    style TREW fill:#fce4ec,stroke:#C2185B
    style TPRED fill:#fce4ec,stroke:#C2185B
    style HZ fill:#fff9c4,stroke:#F9A825
```

图例说明:
- **蓝色**:客观通路输入(共同知识);
- **橙色**:角色通路输入(Self Info);
- **紫色**:信念通路输入(私人推断);
- **绿色**:客观输出权重(共享);
- **粉色**:主观输出权重(per-agent);
- **黄色**:v4 新增 / 改动的关键模块。

### 4.3.2 客观通路:hyper_trans

**输入**:c_ctx ∈ ℝ^{16}。

**输出**:$\theta_{\text{state}}$,StateTransNet 的完整权重集合。

**结构**:多头 MLP,每头生成 StateTransNet 中某一层的权重张量。

```python
class HyperTrans(nn.Module):
    def __init__(self, d_c, state_net_arch):
        super().__init__()
        self.heads = nn.ModuleDict({
            name: MLPHead(d_c, shape)
            for name, shape in state_net_arch.items()
        })

    def forward(self, c_ctx):
        return {name: head(c_ctx) for name, head in self.heads.items()}
```

**关键性质**:hyper_trans **不接收 role_i 或 belief_i**——这保证了所有 agent 共享同一个 StateTransNet 权重,对应资源场动力学的"共同物理"(约束 C1)。

### 4.3.3 主观通路:hyper_rew 与 hyper_pred(v4 三联输入版)

**输入**:$\text{ctx}_i = \text{Concat}[\text{c\_ctx}, \text{role}_i, \text{belief}_i] \in \mathbb{R}^{80}$。

**输出**:
- **hyper_rew** → $\theta_{\text{rew}}^i$:RewardHead 的 per-agent 权重;
- **hyper_pred** → $\theta_{\text{pred}}^i$:PredictionNet 的 per-agent 权重(含 policy head 与 value head)。

**结构**:与 hyper_trans 类似的多头 MLP。

**v4 关键改动**:输入维度从 v3 的 `c_ctx + role_i` 扩展为 `c_ctx + role_i + belief_i`,且 role_i 内部加入 type_emb。这一扩展使 hyper_rew / hyper_pred 能够针对 (类型, 信念, 能力) 的每种组合生成专属权重,实现 Chapter 4.1.3 节三层 motivation 的架构落地。

**生成的 θ 的实际作用**:

考虑 RewardHead 是一个 `[d_s + d_a → 64 → 32 → 1]` 的 MLP,共有约 4000 个参数。hyper_rew 为每个 agent 生成这 4000 个权重,使得 agent $i$ 的 RewardHead 完全针对其 (τ_i, b_i, cap_i) 定制。**当 τ_i = α 时,生成的 RewardHead 学到"∂R/∂u_i = 1"的简单结构;当 τ_i = β 时,生成的 RewardHead 学到包含 Fehr-Schmidt 项的复杂结构**——这就是 Chapter 4.1.1 节"类型梯度撕裂"在架构层的根本解决。

### 4.3.4 参数量分析

| 模块 | 参数量(M) | 说明 |
|---|---|---|
| RepresentationNet | 0.5 | 观测 → s_0 |
| BeliefNet (GRU + heads) | 0.3 | $b_i^t$ + $\hat{c}$ + $\hat{z}$ |
| ContextEncoder (c/role/belief) | 0.1 | 三通路 encoding |
| **hyper_trans** | 0.4 | c_ctx (16) → θ_state |
| **hyper_rew** | 0.6 | ctx (80) → θ_rew |
| **hyper_pred** | 1.0 | ctx (80) → θ_pred |
| StateTransNet | 0.1(由 hyper_trans 生成) | 实际运行时 |
| RewardHead | 0.05(由 hyper_rew 生成) | -- |
| PredictionNet | 0.1(由 hyper_pred 生成) | -- |
| **总计** | **~3.2 M** | 用于 Chapter 6 等参数量对照 |

**对照协议**(详见 Chapter 1.5 节贡献 4 加固协议 A):

- **对齐范围**:hyper_trans + hyper_rew + hyper_pred + StateTransNet + RewardHead + PredictionNet 计入对齐(~2.25 M);
- **不对齐**:RepresentationNet + BeliefNet + ContextEncoder(~0.9 M)在三条 baseline 间共享,完全相同。

---

## 4.4 六大核心网络与数据流(v4 完整版)

> **v3 → v4 改动**:从五大网络扩展为六大网络,新增 **BeliefNet** 作为独立模块(v3 中信念是 ContextEncoder 内部组件)。

### 4.4.1 六大核心网络一览

| 网络 | 输入 | 输出 | 客观/主观 | v4 关键性质 |
|---|---|---|---|---|
| **RepresentationNet** | 观测 $o_i^t$ | 隐状态 $s_i^t$ | 客观(共享) | 与 v3 相同 |
| **BeliefNet**(v4 独立) | 观测历史 $h_i^{0:t}$ | $b_i^t, \hat{c}_i^t, \hat{z}_{i,j}^t$ | 主观(per-agent) | $\hat{z}$ 改为类型 2 分类 |
| **DualHyperNetwork** | $(c_{\text{ctx}}, \text{role}_i, \text{belief}_i)$ | $\theta_{\text{state}}, \theta_{\text{rew}}^i, \theta_{\text{pred}}^i$ | 桥接 | v4 三联输入,role 含 type_emb |
| **StateTransNet** | $(s, a)$, 权重 $\theta_{\text{state}}$ | $s'$ | 客观(共享 θ) | v3 与 v4 相同 |
| **RewardHead** | $(s, a)$, 权重 $\theta_{\text{rew}}^i$ | $r_i$ | 主观(per-agent θ) | 类型分支预测 |
| **PredictionNet** | $s$, 权重 $\theta_{\text{pred}}^i$ | $(\pi_i, v_i)$ | 主观(per-agent θ) | 类型分支策略与价值 |

### 4.4.2 数据流的客观-主观分工

**客观流**(所有 agent 共享):
- 观测 → RepresentationNet → $s$;
- $c_t$ → c_encoder → c_ctx → hyper_trans → $\theta_{\text{state}}$;
- $(s, a)$ + $\theta_{\text{state}}$ → StateTransNet → $s'$。

**主观流**(per-agent):
- 观测历史 → BeliefNet → $(b_i^t, \hat{c}_i^t, \hat{z}_{i,j}^t)$;
- belief_i + role_i + c_ctx → hyper_rew → $\theta_{\text{rew}}^i$;
- $(s, a)$ + $\theta_{\text{rew}}^i$ → RewardHead → $r_i$;
- belief_i + role_i + c_ctx → hyper_pred → $\theta_{\text{pred}}^i$;
- $s$ + $\theta_{\text{pred}}^i$ → PredictionNet → $(\pi_i, v_i)$。

**核心洞察**:**所有 agent 走同一条物理状态预测路径($s' = \text{StateTransNet}(s, a; \theta_{\text{state}})$),但走 N 条不同的奖励/策略路径**。这一架构性分工严格对应 Chapter 4.1.4 节 Harsanyi 表格中的"共同知识 / 私人信念"二分。

---

## 4.5 BeliefNet 训练目标(v4 课程学习版)

> **本节聚焦 BeliefNet 的损失函数定义**。完整的课程学习协议(Oracle → 退火 → 纯推断三阶段)放在 Chapter 5.7 节,本节给出三个损失的形式定义。

### 4.5.1 $c_t$ 推断损失 $\mathcal{L}_c$(监督 MSE)

$$\mathcal{L}_c \;=\; \frac{1}{NT} \sum_{i,t} \bigl(\hat{c}_i^t \;-\; c_t\bigr)^2$$

训练时 $c_t$ 作为 Oracle 信号给出。

**模式 A(默认)**:$c_t$ 在观测中可见 → $\hat{c}_i^t$ 的训练是"对显式信号的复现",收敛快;

**模式 B($c_t$ 隐藏)**:$c_t$ 不在观测中 → $\hat{c}_i^t$ 必须从资源场演化模式中推断,这是 BeliefNet 的核心考验场景,也是 Chapter 6.9 节零样本泛化实验的关键设定。

### 4.5.2 对手类型预测损失 $\mathcal{L}_{\text{opp}}$(v4 监督 2 分类)

**v4 关键改动**:从 v3 的自监督动作预测改为 Oracle 类型 2 分类。

$$\mathcal{L}_{\text{opp}} \;=\; \frac{1}{NT(N-1)} \sum_{i,t,j \neq i} \text{CE}\bigl(\hat{z}_{i,j}^t,\;\tau_j\bigr)$$

其中:
- $\hat{z}_{i,j}^t \in \Delta^2$ 是 agent $i$ 对 agent $j$ 类型的概率分布;
- $\tau_j \in \{\alpha, \beta\}$ 是 agent $j$ 的真实类型,**训练时作为 Oracle 监督信号给出,测试时不给**;
- CE 是 2 分类交叉熵。

**与 v3 的对比**:

| 维度 | v3(动作预测) | v4(类型 2 分类) |
|---|---|---|
| 监督信号 | 自监督(对手未来动作) | Oracle(对手真实类型) |
| 输出维度 | $d_z$ 维向量 | 2 维 softmax |
| 训练难度 | 自监督信号弱、易退化 | 监督信号强、目标明确 |
| 与博弈论对应 | 无显式对应 | Harsanyi "他人 type 的分布信念" |

**对应 Chapter 5.7 课程学习**:训练初期(阶段 1)主观通路接收 Oracle type one-hot 替代 $\hat{z}$ 的输出,中期(阶段 2)线性退火混合,后期(阶段 3)纯使用 $\hat{z}$ 推断。这避免了 $\hat{z}$ 训练不充分时拖累主任务的 chicken-and-egg 困境。

### 4.5.3 信念多样性正则 $\mathcal{L}_{\text{div}}$

为防止 BeliefNet 输出在所有 agent 间坍缩为常数(失去"信念"的语义),引入多样性正则:

$$\mathcal{L}_{\text{div}} \;=\; -\frac{1}{B} \sum_{\text{batch}} \mathrm{Var}_{i}\bigl(b_i^t\bigr)$$

其中 $\mathrm{Var}_i$ 是在 batch 内 N 个 agent 的 belief 向量上的方差(各维度方差求平均)。**注**:负号是因为我们要**最大化**方差(防止坍缩),loss 最小化框架下取负号。

### 4.5.4 BeliefNet 总损失

$$\mathcal{L}_{\text{BeliefNet}} \;=\; \lambda_c \cdot \mathcal{L}_c \;+\; \lambda_{\text{opp}} \cdot \mathcal{L}_{\text{opp}} \;+\; \lambda_{\text{div}} \cdot \mathcal{L}_{\text{div}}$$

推荐权重:$\lambda_c = 1.0$,$\lambda_{\text{opp}} = 0.5$,$\lambda_{\text{div}} = 0.01$。**这些权重在课程学习三阶段保持不变**,仅 belief_i 进入主观通路的方式按阶段变化。

### 4.5.5 与主任务的耦合

BeliefNet 的训练**与主任务损失联合优化**:

$$\mathcal{L}_{\text{total}} \;=\; \mathcal{L}_{\text{MuZero-main}} \;+\; \lambda_b \cdot \mathcal{L}_{\text{BeliefNet}}$$

默认 $\lambda_b = 0.5$。具体的联合训练协议见 Chapter 5.6-5.7 节。

---

## 4.6 功能网络的四层稳定性防线

> 双路超网络架构的训练复杂度比标准 MuZero 高(动态权重生成 + 三联条件化 + BeliefNet 共训),需要专门的稳定化技术。本节给出四层防线,与 v3 大体一致,仅根据 v4 的新增模块做小幅扩展。

### 4.6.1 防线 1:权重生成的初始化

hypernetwork 输出层(生成 θ 的最后一层)的初始化必须谨慎,否则生成的功能网络权重在训练初期可能产生剧烈梯度,导致训练发散。

**具体做法**:
- hypernetwork 的输出层用 **Xavier uniform 初始化,gain = 0.01**(标准 Xavier 的 1/100);
- 这使得初期生成的功能网络权重接近 0(但非完全 0),功能网络在初期表现类似一个"无信息"网络,训练慢启动但稳定。

### 4.6.2 防线 2:LayerNorm 与 GroupNorm

- 所有 ContextEncoder(c/role/belief encoders)输出后接 LayerNorm,稳定 ctx_i 的分布;
- StateTransNet / RewardHead / PredictionNet 在每个隐藏层后用 GroupNorm(因为权重是 hypernetwork 生成的,BatchNorm 不适用);
- BeliefNet 的 GRU 后接 LayerNorm,稳定 $b_i^t$ 的演化。

### 4.6.3 防线 3:梯度裁剪

- hypernetwork 的梯度范数裁剪到 $\leq 1.0$(因为动态权重生成对梯度敏感);
- 主任务的梯度范数裁剪到 $\leq 5.0$;
- BeliefNet 的梯度范数裁剪到 $\leq 1.0$。

### 4.6.4 防线 4:一致性损失(BYOL-style)

详见 Chapter 5.8.2 节。这是 v3 验证有效的稳定化技术,v4 保留。

### 4.6.5 v4 特殊考虑:课程学习阶段切换的稳定性

阶段 1 → 阶段 2(开始退火)与 阶段 2 → 阶段 3(完全使用 $\hat{z}$)的两次切换可能引入主任务 loss 的突跃。**应对**:

- 阶段 2 线性退火窗口足够长(40% 步数),避免硬切换;
- 监控指标:阶段 2 开始时主任务 loss 的 50K 步内涨幅 < 5%,否则延长退火窗口至 50% 步数;
- 学习率在阶段切换时不重置(避免破坏 momentum)。

---

## 4.7 关键代码 Diff 指引(v4 版)

> **本节用途**:为代码实施提供具体的文件级 diff 指引。这一指引与 `Hyper_MuZero_v4_Roadmap.md` 的 Part 3(代码改动 Checklist)严格对应。

### 4.7.1 新增文件

| 文件 | 用途 | 大致行数 |
|---|---|---|
| `resource_commons.py`(替代 `non_stationary_tag.py`) | ResourceCommons 环境,含类型机制 | ~1200 |
| `belief_net.py` | BeliefNet 独立模块,GRU + 两个 head | ~200 |
| `baselines/input_conditioning.py` | Input-Wide / Input-Deep 变体 | ~300 |
| `baselines/ma_muzero.py` | MA-MuZero baseline | ~400 |
| `baselines/mappo.py`, `qmix.py`, `conflict_aware_ga.py`, `mamba.py`, `marie.py` | 其他 baseline | ~400 each |
| `mup_config.py` | μP 学习率对齐 | ~150 |
| `eval_protocols.py` | 统一评估协议(Self Info / c 分段 / 零样本) | ~300 |
| `curriculum_scheduler.py` | 课程学习阶段控制 | ~100 |

### 4.7.2 重大修改

| 文件 | 关键修改 |
|---|---|
| `context_encoder.py` | role_encoder 加入 type_emb;belief_encoder 接收新的 $\hat{z}$ 2 维输出 |
| `hyper_network.py` | hyper_rew / hyper_pred 输入维度扩展为 80 维(c_ctx + role + belief);hyper_trans 保持 16 维(仅 c_ctx) |
| `hyper_muzero_model.py` | `set_context(c_t, τ_i, cap_i, b_i)` 接口扩展;内部调用 ContextEncoder 与 DualHyperNetwork |
| `muzero_trainer.py` | (1) 引入 curriculum_scheduler;(2) 类型分层视角采样;(3) RewardHead 预测目标按 buffer 中真实 r 计算(类型 α/β 真实奖励已经按 Chapter 3 公式 3.10 在环境侧分别计算并存入 buffer);(4) BeliefNet 联合优化 |
| `mve_planner.py` | `set_context` 调用扩展;BeliefNet 推断接入 |
| `episode_buffer.py`, `buffer.py` | 字段扩展:`Δ`, `{τ_i}`, `{cap_i}`, `{ẑ}` 等 |

### 4.7.3 轻微修改

| 文件 | 关键修改 |
|---|---|
| `mve.py` | type-specific θ_rew/θ_pred 的传入 |
| (训练入口脚本) | LR sweep 协议;Easy/Medium/Hard 配置切换 |

### 4.7.4 不需要修改

- StateTransNet / RewardHead / PredictionNet 的内部结构(它们的权重由 hypernetwork 生成,接口不变);
- RepresentationNet 的内部结构。

---

## 4.8 与 v3 架构的对比小结(v4 演进表)

| 维度 | v3 (Chapter 4 v2) | v4 (本章) |
|---|---|---|
| **Motivation 论证** | 主客解耦防止梯度撕裂(笼统) | 容量分配几何 + Fehr-Schmidt 偏导对照表(可计算,可消融) |
| **三联通路** | c_ctx + role(id + cap) + belief | c_ctx + role(id + cap + **type_emb**) + belief(**$\hat{c}$ + 2分类 $\hat{z}$**) |
| **role 通路** | id_emb + cap_emb | id_emb + cap_emb + **type_emb**(v4 新增) |
| **belief 通路 $\hat{z}$** | 自监督动作预测,$d_z$ 维 | Oracle 类型 2 分类,2 维 softmax |
| **BeliefNet 训练** | 自监督 + 主任务联合 | 课程学习三阶段(Oracle → 退火 → 纯推断) |
| **可证伪断言** | 隐含(无明确锚) | **显式 4 个断言**,与 Chapter 6 消融实验 1:1 绑定 |
| **博弈论对应** | 提及 Harsanyi | **显式 Harsanyi 表格**(共同知识 / 自己 Type / 对他人 Type 的信念) |
| **容量分配理论支撑** | 无 | Ha 2017 / CAVIA 2019 / FiLM 2018 等元学习文献支撑 |
| **等参数量对照** | 无 | 对齐范围 + Input-Wide/Deep 双跑 + μP |

---

## 4.9 本章小结

本章在第三章 ResourceCommons 异构偏好公地博弈环境的基础上,完成了 DualHyperNetwork v4 架构的完整设计。核心要点:

1. **设计原则的严格化(4.1)**:从 v3"主客解耦防止梯度撕裂"的笼统 motivation 升级为基于"容量分配几何"的严格论证,Fehr-Schmidt 偏导对照表(4.1.1)给出了类型梯度撕裂的可计算证据,hypernetwork vs input conditioning 的容量论证(4.1.2)有 Ha 2017、CAVIA 2019、FiLM 2018 等文献支撑;

2. **三联通路的工程实现(4.2)**:c_ctx(共同知识)+ role_i(Self Info,**含 type_emb**)+ belief_i(BeliefNet 推断的 $\hat{c}$ + 类型 2 分类 $\hat{z}$),严格对应 Chapter 4.1.4 节 Harsanyi 表格;

3. **DualHyperNetwork 的双路设计(4.3)**:客观通路 hyper_trans 仅接收 c_ctx 生成共享 θ_state(物理),主观通路 hyper_rew / hyper_pred 接收完整三联输入生成 per-agent θ_rew^i / θ_pred^i(偏好);

4. **六大核心网络的客观-主观分工(4.4)**:RepresentationNet 与 StateTransNet 走客观流(所有 agent 共享),BeliefNet、RewardHead、PredictionNet 走主观流(per-agent);

5. **BeliefNet 训练目标的 v4 升级(4.5)**:$\hat{z}$ 改为 Oracle 类型 2 分类(从自监督动作预测改为监督类型分类),与 Chapter 5.7 节课程学习协议配套;

6. **四层稳定性防线(4.6)**:权重初始化、Norm 层、梯度裁剪、一致性损失,v3 验证有效的方案保留并按 v4 课程学习需求小幅扩展;

7. **代码 Diff 指引(4.7)**:为 Chapter 5 训练算法与 `Hyper_MuZero_v4_Roadmap.md` Part 3 提供文件级实施清单;

8. **v3 → v4 演进表(4.8)**:9 个维度的对比,明确每一处改动的"为什么"。

整个 Chapter 4 v4 严格服务于 Chapter 1.5 节贡献 2 的三个可证伪断言:断言 A 的架构来源是 type_emb 进入 role_i 通路 + per-type θ_rew(4.2.2 + 4.3.3);断言 B 的架构来源是 hypernetwork vs input conditioning 的容量分配几何(4.1.2);断言 C 的架构来源是 c_ctx + role + belief 三联通路的不可替代性(4.2 + 4.3.3)。

下一章(Chapter 5)将在本章架构基础上,定义"如何做决策与如何训练"——MVE+CRN 规划器、K 步展开训练、课程学习协议,完成 Hyper-MuZero v4 方法的最后一公里。
