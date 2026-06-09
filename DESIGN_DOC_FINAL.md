# Hyper-MuZero: 基于超网络与视角自适应规划的非平稳多智能体强化学习框架
## 最终设计文档 v4.7

> **本文档为最终实施版本**，整合了 v1.0(环境/实验)、v1.1(接口/依赖)、v3.0(MuZero训练/视角建模)、v3.2(稳定性防线) 的所有设计决策，并纳入 v4.0 四项关键改进（L2 Norm、Hinge Variance、Projection+CosineSim、Loss 平均化）、v4.1 统一评估框架、v4.2 阶段性冻结（对抗训练节奏控制）、v4.3 奖励正交化重构（Blocking Point + 信号消除修复）、v4.4 训练稳定性改进（Target Network EMA + CosineAnnealingLR + Adam eps）、v4.6 Planner CRN 方差消减 + 回退 PG 辅助 Loss，以及 **v4.7 Reward HyperNet 分化改进（output_scale 初始化陷阱修复 + Diversity Loss + 加深 hyper_rew + Warmup Cosine LR）**。
> 所有先前文档(DESIGN_DOC.md, DESIGN_DOC_v1.1_APPENDIX.md, design3.0.md, HYPER_MUZERO_IMPROVEMENTS.md)归档为参考，不再更新。

---

## 一、问题定义

### 1.1 核心问题
在 **非平稳多智能体环境** 中，环境规则(Rule)随 Episode 动态变化：
- 转移分布：$s_{t+1} \sim P(\cdot | s_t, a_t; \text{Rule})$
- 奖励函数：$r_{t,i} = R_i(s_t, a_t; \text{Rule})$ （per-agent，取决于角色）

### 1.2 核心创新
1. **视角即上下文 (Perspective as Context)**：将环境规则 Rule 和智能体身份 Agent_ID 统一为广义上下文 $C_{aug}$
2. **双超网络架构 (DualHyperNetwork)**：客观状态转移 vs 主观奖励/价值分离
3. **Per-Agent Coordinate Descent MVE Planner + CRN**：逐 Agent 坐标下降 + 相关采样方差消减 (v4.6)
4. **MuZero 风格展开训练**：K 步 Unroll + Consistency Loss

---

## 二、非平稳环境设计（NonStationaryTag — Faction Re-alignment v2）

> 基于 v1.0 §三，v2 引入 **阵营重组 (Faction Re-alignment)** 机制：Rule 改变的不仅是 Agent 2 的行为，而是**全局敌友关系矩阵**，所有 Agent 的奖励函数都随 Rule 变化。

### 2.1 智能体角色

| 智能体 | 角色 | Rule=1 行为 | Rule=0 行为 | 颜色 |
|--------|------|------------|------------|------|
| Agent 0 | 固定猎人 | 追猎物，Agent 2 是队友 | 追猎物，Agent 2 是障碍 | 红色 |
| Agent 1 | 固定猎人 | 追猎物，Agent 2 是队友 | 追猎物，Agent 2 是障碍 | 红色 |
| Agent 2 | **可变角色** | 猎人（完全复用 Hunter reward） | 保镖（拦截猎人，保护猎物） | 黄色 |
| Agent 3 | 猎物 | 恐惧所有人（含 Agent 2） | 恐惧 {0,1}，**靠近 Agent 2 寻求庇护** | 绿色 |

### 2.2 Rule 机制（阵营重组）
- `Rule ∈ [0, 1]`，Episode 间采样，Episode 内不变
- **所有 Agent** 的奖励 = `Rule × r_hunt + (1-Rule) × r_guard`
- **Rule 不进入观测**，仅记录在 info 中

**阵营配置**：

| Rule 值 | 红方 (Hunters) | 黄绿联盟 (Defenders) | 博弈模式 |
|---------|---------------|---------------------|----------|
| Rule = 1 | {0, 1, 2} | — | 全面围剿：3 vs {0,1,2} |
| Rule = 0 | {0, 1} | {2, 3} | 保卫战：{0,1} vs {2,3} |
| Rule ∈ (0,1) | — | — | 连续插值，行为渐变 |

### 2.3 正交化奖励设计（v4.3 重构）

> **v4.3 变更**：完全重构奖励函数，解决 v2 设计中的四层问题（P0 方向性缺失、P1 幅度不对称、P2 信号消除、P3 GRU 推断恶性循环）。核心思想：**让 r_hunt 和 r_guard 关注不同的物理事件/几何特征，消除所有符号对冲**。

#### 2.3.1 设计原则

1. **无符号对冲**：同一物理事件不在 r_hunt / r_guard 中出现相反符号——每行要么同号叠加，要么一侧为零
2. **独立密集引导**：Hunt 用 dist_to_prey 引导；Guard 用 dist_to_block_point 引导。两个引导指向**不同物理位置**
3. **RewardHead 可预测性约束**：奖励函数的每一项等价于"某 agent 到某个由其他 agent 位置线性决定的参考点的距离的单调函数"——这是 RewardHead（小 MLP）已验证能学会的函数类
4. **渐进验证**：先二值 rule {0,1} 验证极端模式，再扩展到连续 rule

#### 2.3.2 阻截点（Blocking Point）机制

**核心思想**：Guard 模式下，Agent 2 应被奖励"站在猎人到猎物的路径上"。定义阻截目标点，用点-点距离提供密集方向引导。

对每个固定猎人 $h_i$（$i \in \{0, 1\}$）：

$$\mathbf{b}_i = \mathbf{p}_{h_i} + \alpha \cdot (\mathbf{p}_{prey} - \mathbf{p}_{h_i}) = (1-\alpha) \cdot \mathbf{p}_{h_i} + \alpha \cdot \mathbf{p}_{prey}$$

单猎人 Shielding Score（线性衰减）：

$$\text{shield\_score}_i = \max\left(0, \; R_{max} \cdot \left(1 - \frac{\|\mathbf{p}_{a2} - \mathbf{b}_i\|}{d_{max}}\right)\right)$$

多猎人聚合（取最大值——优先阻截最有效目标）：

$$\text{shield\_reward} = \max(\text{shield\_score}_0, \; \text{shield\_score}_1)$$

```
几何示意:

场景 1: 单猎人                    场景 2: 双猎人

H0 ------●------ Prey           H0 ---●--- Prey ---●--- H1
          ↑                            ↑              ↑
     block_point_0                 block_0        block_1
     (H0→Prey 60%处)             Agent2 用 max 选较高 score 一侧
```

| 参数 | 值 | 依据 |
|------|-----|------|
| α | 0.6 | 阻截点偏近猎物侧，符合护卫直觉；两极端场景均合理 |
| R_max | 0.5 / step | 完美护卫 episode 累计 ≈ +4.5，与 catch +10 同数量级 |
| d_max | 1.5 | ≈ 半对角线，保证大部分地图有非零梯度 |
| 聚合 | max | 语义清晰，离散动作下不可导问题不严重 |

**RewardHead 可预测性论证**：阻截点奖励 $f(\|\mathbf{p}_{a2} - (c_1 \mathbf{p}_h + c_2 \mathbf{p}_{prey})\|)$ 与 hunt shaped $g(\|\mathbf{p}_{a2} - \mathbf{p}_{prey}\|)$ **同构**——都是 agent 到其他 agent 位置线性组合的距离的单调函数。RewardHead 能预测后者即能预测前者。

#### 2.3.3 Agent 2（Variable Agent）完整奖励

$$r_{agent2} = \text{rule} \times r_{hunt} + (1 - \text{rule}) \times r_{guard} - \text{bound\_penalty}$$

| 事件 | r_hunt | r_guard | rule=0.5 合计 | 消除？ |
|------|--------|---------|--------------|--------|
| 接近猎物 shaped | -0.1·d_prey | **0** | -0.05·d | ✅ 一侧为零 |
| 阻截位 shaped | **0** | +shield_reward ≤ 0.5 | +0.5·shield | ✅ 一侧为零 |
| 碰到猎物 | +10 | **0** | +5.0 | ✅ 一侧为零 |
| 碰到固定猎人 | -1.0 | **0** | -0.5 | ✅ 一侧为零 |
| 猎物被猎人抓 | 0 | -10 | -5.0 | ✅ 一侧为零 |

**所有行对侧为零，无任何信号消除。**

#### 2.3.4 Prey（Agent 3）完整奖励

$$r_{prey} = \text{rule} \times r_{hunt} + (1 - \text{rule}) \times r_{guard} - \text{bound\_penalty}$$

| 事件 | r_hunt | r_guard | rule=0.5 合计 | 消除？ |
|------|--------|---------|--------------|--------|
| 远离固定猎人 shaped | +0.1·d per h | +0.1·d per h | +0.1·d per h | ✅ 同号 |
| 对 Agent 2 距离 | **0** | +0.1·max(0, 2.0-d) | +proximity/2 | ✅ 一侧为零 |
| 碰固定猎人 | -10 | -10 | -10 | ✅ 同号 |
| 碰 Agent 2 | -10 | **0** | -5.0 | ✅ 一侧为零 |

**已知退化**：rule=1 时 Prey 对 Agent 2 无距离 shaped → 通过碰撞 -10 间接学会闪避。如不足，回退方案：加回距离项系数 0.02。

#### 2.3.5 Fixed Hunters（Agent 0, 1）——无变更

| 事件 | r_hunt | r_guard | rule=0.5 合计 | 消除？ |
|------|--------|---------|--------------|--------|
| 接近猎物 shaped | -0.1·d | -0.1·d | -0.1·d | ✅ 同号 |
| 碰猎物 | +10 | +10 | +10 | ✅ 同号 |
| 碰 Agent 2 | -1.0 | -0.5 | -0.75 | ✅ 同号 |

#### 2.3.6 行为光谱（连续 Rule 语义验证）

| rule | Agent 2 密集引导 | 稀疏激励 | 行为解读 |
|------|-----------------|---------|---------|
| 1.0 | 100% 向猎物 | 碰猎物 +10 | 纯猎人 |
| 0.7 | 70% 向猎物 + 30% 向阻截点 | 碰猎物 +7, 保护失败 -3 | 偏攻——追猎物为主，路过阻截位有 bonus |
| 0.5 | 50% + 50% | 碰猎物 +5, 保护失败 -5 | 机会主义者——两个吸引子间权衡 |
| 0.3 | 30% + 70% | 碰猎物 +3, 保护失败 -7 | 偏守——占位为主，被猎物微弱吸引 |
| 0.0 | 100% 向阻截点 | 保护失败 -10 | 纯护卫 |

**数值标定**：

| 模式 | 理想 episode 总奖励 | 计算 |
|------|-------------------|------|
| Guard (rule=0) | ≈ +3.5 | shield: 25×0.6×0.3 ≈ +4.5, boundary ≈ -1.0 |
| Hunt (rule=1) | ≈ +6.0 | catch +10, shaped ≈ -2.5, collision ≈ -1.0, boundary ≈ -0.5 |

#### 2.3.7 与 v2 设计差异对照

| 模块 | v2 原设计 | v4.3 新设计 | 修改原因 |
|------|----------|-----------|---------|
| Agent 2 guard shaped | -0.1·dist_to_prey | +shield_reward(阻截点) | P0: 引导方向与 hunt 一致，无法涌现拦截行为 |
| Agent 2 guard 碰猎人 | +1.0 | 0 | P0+P2: 稀疏 + 与 hunt -1.0 符号对冲 |
| Agent 2 guard 碰猎物 | 隐含在 -0.1d | 0 | 正交化：guard 不关注猎物接触 |
| Prey hunt 远离 Agent 2 | +0.1·dist | 0 | P2: 消除与 guard proximity 的对冲 |
| Prey guard 贴身 Agent 2 | +1.0 | 0 | 消除与 hunt -10 的对立 |
| rule 采样 | U[0,1] | 阶段性：{0,1} → U[0,0.3]∪U[0.7,1.0] → U[0,1] | 渐进验证 |
| 新增组件 | — | `_compute_shield_reward()` | 阻截点计算 |

### 2.4 离散动作空间（v3.1 新增）
每个 Agent 有 **5 个离散动作**：

| 动作ID | 语义 | 力向量 |
|--------|------|--------|
| 0 | 停 | [0.0, 0.0] |
| 1 | 右 | [1.0, 0.0] |
| 2 | 左 | [-1.0, 0.0] |
| 3 | 上 | [0.0, 1.0] |
| 4 | 下 | [0.0, -1.0] |

**实现方式**：DiscreteActionWrapper 将网络输出的 0-4 映射为环境需要的 [x, y] 力向量。

```python
class DiscreteActionWrapper(gym.ActionWrapper):
    FORCE_MAP = {
        0: [0.0, 0.0],  # None
        1: [1.0, 0.0],  # Right
        2: [-1.0, 0.0], # Left
        3: [0.0, 1.0],  # Up
        4: [0.0, -1.0], # Down
    }
    def action(self, act_idx):
        return self.FORCE_MAP[act_idx]
```

### 2.5 环境接口

| 方法 | 输入 | 输出 |
|------|------|------|
| `reset(options=None)` | `options`: 可选 dict, 支持 `{'rule': float}` 强制指定 Rule | `obs_n`, `rule` |
| `step(action_n)` | `List[int]` (每agent一个离散动作) | `obs_n, reward_n, done_n, info_n` |
| `get_rule()` | — | `float` |

**强制 Rule 机制 (v4.1 新增)**：
- `reset(options={'rule': 0.5})` → 该 episode 使用 Rule=0.5
- `reset()` 或 `reset(options=None)` → 随机采样 Rule
- 实现路径：`ns_environment.py` 将 `options['rule']` 存入 `world.force_rule`，`non_stationary_tag.py` 的 `reset_world` 优先使用 `force_rule`
- 颜色/奖励等均在 `reset_world` 内根据 Rule 正确设置，无时序漏洞

---

## 三、系统架构设计

### 3.1 核心理念：视角即上下文

$$C_{aug} = [\text{RuleEncoder}(rule), \text{Embedding}(agent\_id)]$$

- **Rule**：决定客观物理反馈机制（如碰撞是否得分）
- **Agent ID**：决定主观价值判断（如我想抓人还是想逃跑）
- **超网络**：$H(C_{aug}) \to \theta$，生成"在该规则下，身为该角色"的世界观

### 3.2 网络架构总览

```
┌──────────────── Context Path ──────────────────────┐
│  Rule ──→ [RuleEncoder] ──→ rule_emb               │
│  Agent_ID ──→ [Embedding] ──→ id_emb               │
│                                                      │
│  hyper_trans(rule_emb)           → θ_state           │
│  hyper_rew(rule_emb ⊕ id_emb)   → θ_reward          │
│  hyper_pred(rule_emb ⊕ id_emb)  → θ_pred            │
└──────────────────────────────────────────────────────┘

┌──────────────── Perception Path ─────────────────────┐
│  o_joint ──→ [RepresentationNet(固定权重)] ──→ s      │
└──────────────────────────────────────────────────────┘

┌──────────────── Dynamics Path ───────────────────────┐
│  s, A_joint ──→ [StateTransNet(θ_state)] ──→ s'     │  ← 客观
│  s, A_joint ──→ [RewardHead(θ_reward)] ──→ r_i      │  ← 主观 (因果性: r=R(s,a))
└──────────────────────────────────────────────────────┘

┌──────────────── Prediction Path ─────────────────────┐
│  s ──→ [PredictionNet(θ_pred)] ──→ p_i [5], v_i [1] │  ← 主观
└──────────────────────────────────────────────────────┘
```

### 3.3 五大核心网络

| 网络 | 输入 | 输出 | 权重来源 | 客观/主观 |
|------|------|------|---------|----------|
| **RepresentationNet** | $o_{joint}$ | $s$ | 固定训练参数 | 客观 |
| **StateTransNet** | $s, A_{joint}$ | $s'$ | hyper_trans(rule) | 客观 |
| **RewardHead** | $s, A_{joint}$ | $r_i$ (标量) | hyper_rew(rule, id) | 主观 |
| **PredictionNet** | $s$ | $p_i$ [5], $v_i$ [1] | hyper_pred(rule, id) | 主观 |
| **ContextEncoder** | rule, agent_id | $c_{aug}$ | 固定训练参数 | — |

> **RewardHead 输入说明**：使用当前状态 $s$（而非 $s'$），符合 MDP 因果性 $r = R(s, a)$，即"在状态 s 执行动作 a 得到奖励 r"。避免 reward 预测对 transition 输出质量的耦合依赖。

### 3.4 DualHyperNetwork

```python
class DualHyperNetwork(nn.Module):
    """三路超网络：客观转移 + 主观奖励 + 主观预测"""
    def __init__(self, rule_dim, id_dim, trans_params, rew_params, pred_params):
        self.hyper_trans = SimpleMLP(rule_dim, trans_params)           # Rule only
        self.hyper_rew   = SimpleMLP(rule_dim + id_dim, rew_params)   # Rule + ID
        self.hyper_pred  = SimpleMLP(rule_dim + id_dim, pred_params)  # Rule + ID

    def forward(self, rule_emb, id_emb):
        θ_state  = self.hyper_trans(rule_emb)
        θ_reward = self.hyper_rew(cat([rule_emb, id_emb]))
        θ_pred   = self.hyper_pred(cat([rule_emb, id_emb]))
        return θ_state, θ_reward, θ_pred
```

### 3.5 Functional Net 稳定性设计 (v3.2 新增)

超网络动态生成权重存在"权重爆炸 → 激活爆炸 → Loss 爆炸"的风险链。v3.2 引入三层稳定性防线：

#### 3.5.1 Adaptive LayerNorm (AdaLN) — 切断激活爆炸链

所有 Functional Net 的**隐藏层**内嵌 AdaLN：超网络除了生成每层的 `weight` 和 `bias`，还额外生成 `gamma` 和 `beta`。

```
每层参数: weight (out, in) + bias (out) + gamma (out) + beta (out)

Forward: x = W·x + b → LayerNorm → x * (1 + gamma) + beta → ReLU
```

**原理**：无论权重 $W$ 多大，LayerNorm 都会将激活值归一化到 $\mathcal{N}(0, 1)$，然后由超网络生成的 $\gamma, \beta$ 做可控的仿射变换。

**v4.6 关键修复 — 残差调制 `(1 + gamma)`**：

原始公式 `h * gamma + beta` 存在严重的信号衰减问题。当 HyperNet 的 output_scale 较小时（初始化为 0.01，L2 归一化后每个参数 ≈ 10^-5），gamma 接近 0，导致 LayerNorm 归一化后的单位尺度信号被乘以 ~0 → 后续层只能看到 bias 常数，输入信息（特别是 action 信息）完全丢失。

改为 `h * (1 + gamma) + beta` 后：当 gamma ≈ 0 时，输出 ≈ h_norm（恒等传递），归一化信号以单位尺度传播到后续层。这是 DiT / StyleGAN / FiLM 等条件归一化架构的标准做法。

**适用范围**：
- StateTransNet: FC1, FC2 加 AdaLN，输出层 FC3 不加（接固定 LayerNorm）
- RewardHead: FC1, FC2 加 AdaLN，输出层 FC3 不加（输出标量 reward）
- PredictionNet: shared trunk 的 FC1, FC2 加 AdaLN，PolicyHead/ValueHead 不加（单层线性映射）

```python
# AdaLN Forward (per hidden layer) — v4.6 residual modulation
def adaln_forward(x, weight, bias, gamma, beta):
    h = functional_linear(x, weight, bias)
    # Instance-wise LayerNorm (per-sample normalization)
    mean = h.mean(dim=-1, keepdim=True)
    var = h.var(dim=-1, keepdim=True, unbiased=False)
    h = (h - mean) / torch.sqrt(var + 1e-5)
    # Adaptive affine transform with residual modulation (v4.6)
    # (1 + gamma) ensures identity pass-through when gamma ≈ 0
    h = h * (1 + gamma) + beta
    return F.relu(h)
```

#### 3.5.2 残差连接 (Residual Connection) — StateTransNet 安全网

StateTransNet 使用残差结构，学习状态变化量 $\Delta s$ 而非绝对状态：

$$s' = s + \text{TransNet}(s, A_{joint}; \theta_{state})$$

**初始行为**：当超网络输出接近零时，$\text{TransNet}$ 输出 $\approx 0$，因此 $s' \approx s$（静止不动）。这比随机输出一个巨大的 $s'$ 合理得多。

> 注意：仅 StateTransNet 使用残差。RewardHead（输出标量）和 PredictionNet（输出 policy+value）的维度不匹配，不适合残差。

#### 3.5.3 超网络输出 L2 归一化 + 可学习缩放 (v4.0 升级) — 方向/模长解耦

> **v4.0 变更**：原 v3.2 仅使用标量 `output_scale=0.1` 缩放，实验发现不足以约束权重模长的剧烈波动。v4.0 升级为 **L2 归一化 + 小初始模长**，将"方向"和"模长"解耦。

ChunkedHyperNetWrapper 在输出端使用：

```python
self.norm_output = True  # 默认开启 L2 归一化
self.output_scale = nn.Parameter(torch.tensor(0.01))  # 可学习模长 (初始=0.01)

def forward(self, context):
    raw = self.hnet.forward(cond_input=context, ret_format="flattened")
    # [v4.0] L2 归一化：固定方向分布，模长恒为 1.0
    if self.norm_output:
        norms = torch.linalg.norm(raw, dim=-1, keepdim=True)
        raw = raw / (norms + 1e-8)
    return raw * self.output_scale  # 初始模长 = 0.01
```

**初始行为链**：
1. L2 Norm → 参数向量模长 = 1.0（方向随机，但模长固定）
2. × output_scale(0.01) → 实际模长 = 0.01
3. 其中 gamma/beta ≈ 0.01 → AdaLN 输出 ≈ LayerNorm 输出 → **静默启动 (Quiet Start)**
4. 训练过程中 output_scale 逐渐增长，超网络从"微调方向"起步

**为何 0.01 而非 0.1**：L2 Norm 后模长恒为 1.0（vs 原始随机初始化模长 ≈ 0.1×std），所以需要更小的 scale 来补偿。

**四层防线协作** (v4.0)：

| 层 | 机制 | 保护对象 |
|----|------|---------|
| 第0层 | L2 归一化 | 固定参数模长，防止方向/模长耦合波动 |
| 第1层 | output_scale=0.01 | 全局缩小至安全模长 |
| 第2层 | AdaLN (每层) | 切断层间激活爆炸链 |
| 第3层 | 残差连接 (StateTransNet) | 保证初始状态转移合理 |

### 3.6 Baseline (Exp1) 的等价实现

Exp1 无超网络。保持**完全相同的网络结构**，但用固定权重 + agent_id 拼接输入：

```python
# Baseline: 无 HyperNet，agent_id 作为输入特征
class BaselineDynamicNet(nn.Module):
    def state_transition(self, s, A_joint):
        return self.state_mlp(cat([s, A_joint]))  # 无 agent_id

    def reward(self, s_next, A_joint, id_emb):
        return self.reward_mlp(cat([s_next, A_joint, id_emb]))  # 拼接 id

class BaselinePredNet(nn.Module):
    def forward(self, s, id_emb):
        x = cat([s, id_emb])
        return self.policy_head(x), self.value_head(x)
```

---

## 四、规划器设计 (Planner)

### 4.1 Per-Agent Coordinate Descent MVE Planner + CRN (v4.6)

> **v4.5 重写**：原 Joint-Space 采样在多 Agent 下完全失效，改为逐 Agent 坐标下降搜索。
> **v4.6 修复**：加入 Common Random Numbers (CRN) 消除其他 Agent 随机动作的方差，使 Planner 能检测到候选动作间的真实 return 差异。

#### 问题层级

Planner 产出均匀分布 (`l_pol ≡ ln5 = 1.6094`) 存在两层独立的根因：

| 层级 | 问题 | 解决方案 | 版本 |
|------|------|---------|------|
| 第1层 | 联合空间 $5^4=625$/step → 50 samples 覆盖率 ≈ 0.08 | Per-Agent Coordinate Descent → 5/step | v4.5 |
| 第2层 | 其他 Agent 独立采样噪声淹没候选动作信号 (SNR ≈ 0.02) | CRN 相关采样 → 噪声完全抵消 | v4.6 |

#### 第1层问题：联合动作空间爆炸 (v4.5 已解决)

| 因素 | 单 Agent (Atari) | 4 Agent 设置 |
|------|-----------------|-------------|
| 每步动作空间 | 18 | $5^4 = 625$ |
| 50 samples 覆盖率 | $50/18 \approx 2.8$ per step | $50/625 \approx 0.08$ per step |

解决方案：Coordinate Descent 将搜索空间从 625/step 降到 5/step。

#### 第2层问题：方差淹没信号 (v4.6 修复)

即使搜索空间只有 5 个候选，评估每个候选时，其他 3 个 Agent 的动作是**独立采样**的。其他 Agent 动作造成的 return 波动远大于 Agent j 动作选择带来的差异：

```
评估 "右" (10 个采样):
    sample 1: Q = r(右, 其他agent随机动作_1) + ... = 3.2
    sample 2: Q = r(右, 其他agent随机动作_2) + ... = -1.5
    平均 Q_右 = 0.73

评估 "左" (10 个采样):
    sample 1: Q = r(左, 其他agent随机动作_A) + ... = -0.9
    sample 2: Q = r(左, 其他agent随机动作_B) + ... = 2.1
    平均 Q_左 = 0.69

信号（agent j 的动作差异）:    ~0.02-0.1 / step
噪声（其他 agent 随机动作）:   ~0.5-2.0 / step
SNR ≈ 0.05 / 1.0 = 0.05

10 个采样的标准误差 ≈ 1.0 / √10 ≈ 0.32
信号 0.05 << 标准误差 0.32 → 完全淹没

softmax([0.73, 0.69, 0.71, 0.70, 0.72] / 1.0) ≈ 均匀
```

#### 解决方案：Common Random Numbers (CRN)

核心思想：评估不同候选动作时，让其他 Agent 的 **step-0 动作完全相同**。这样 return 差异只来自 Agent j 的动作选择，其他 Agent 的随机性完全抵消。

```
修复前（独立采样）:
    评估 "右" sample_3: 其他agent动作 = [猎人0=上, 猎人1=右, 猎物=下]
    评估 "左" sample_3: 其他agent动作 = [猎人0=左, 猎人1=停, 猎物=右]  ← 不同！

    Q_右 - Q_左 = (signal差异) + (noise差异) ≈ noise差异 (信号被淹没)

修复后（CRN 相关采样）:
    评估 "右" sample_3: 其他agent动作 = [猎人0=上, 猎人1=右, 猎物=下]
    评估 "左" sample_3: 其他agent动作 = [猎人0=上, 猎人1=右, 猎物=下]  ← 相同！

    Q_右 - Q_左 = signal差异 (noise 完全抵消!)
```

**数据布局变更**：

```
v4.5 布局: candidate outer, sample inner
    [a0_s0, a0_s1, ..., a0_s9, a1_s0, ..., a1_s9, ..., a4_s9]
    reshape: (B, A, spa) → mean over spa → (B, A)

v4.6 布局: scenario outer, candidate inner
    [s0_a0, s0_a1, ..., s0_a4, s1_a0, ..., s1_a4, ..., s9_a4]
    reshape: (B, spa, A) → mean over spa → (B, A)
```

**Step > 0 的处理**：只在 step 0 使用 CRN。Step > 0 状态已因不同候选动作而分叉，其他 Agent 面对的状态不同，不能严格共享动作。Step > 0 的独立噪声被 $\gamma^k$ 折扣，影响递减。

```python
def sample_mve_plan(model, root_s, cfg, rule=None):
    """Per-agent coordinate descent MVE planning with CRN."""
    B, N, A, K, S = ...
    spa = S // A          # scenarios (e.g. 50//5=10)
    M = A * spa           # total trajectories per batch element

    agent_order = random_permutation(N)
    pi_mve = zeros(B, N, A)
    optimised = set()

    for j in agent_order:
        # ── Phase 1: Pre-sample other agents' step-0 actions (CRN) ──
        s_scenarios = root_s.repeat_interleave(spa, dim=0)  # (B*spa, latent)
        step0_actions = {}
        for i in range(N):
            if i == j: continue
            if i in optimised:
                step0_actions[i] = sample(pi_mve[:, i], B*spa)  # once per scenario
            else:
                step0_actions[i] = sample(policy(s_scenarios, i))

        # ── Phase 2: Expand to (B*M,) — replicate across A candidates ──
        s_exp = s_scenarios.repeat_interleave(A, dim=0)  # (B*M, latent)
        first_action_j = arange(A).repeat(B * spa)       # [0,1,2,3,4, 0,1,2,3,4, ...]
        for i in step0_actions:
            step0_actions[i] = step0_actions[i].repeat_interleave(A, dim=0)

        # ── Phase 3: Rollout K steps ──
        for step in range(K):
            for i in range(N):
                if step == 0 and i == j:
                    a_i = first_action_j              # enumerated
                elif step == 0 and i in step0_actions:
                    a_i = step0_actions[i]            # CRN: same within scenario
                else:
                    a_i = sample(policy(curr_s, i))   # step>0: independent
            ...

        # ── Phase 4: Aggregate with CRN layout ──
        returns_per_action = cum_return_j.view(B, spa, A).mean(dim=1)  # (B, A)
        pi_mve[:, j] = softmax(returns_per_action / temperature)

        optimised.add(j)
    return pi_mve
```

#### 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| Agent 间搜索方式 | 顺序 Coordinate Descent | 后优化的 Agent 可利用前面的改进 |
| 优化顺序 | 每次随机打乱 | 消除位置偏差，长期公平 |
| Step 0 方差消减 | CRN (Common Random Numbers) | 消除其他 Agent 随机动作的噪声 |
| Step > 0 | 独立采样 | 状态已分叉，CRN 不严格适用；γ^k 折扣降低影响 |
| 已优化 Agent 的 step 0 动作 | CRN 采样：每 scenario 一次，复制给 A 个候选 | 保留随机性 + 噪声抵消 |
| 当前 Agent 的 step 0 动作 | 枚举所有 A 个候选 | 确定性，消除首步采样方差 |
| 每候选采样数 (scenarios) | $S / A$ (默认 50/5=10) | CRN 下甚至 spa=1 也有信号，10 绰绰有余 |

#### id_emb 多 Agent 切换（实现要点）

超网络架构下，`set_context(rule, agent_id)` 会生成并缓存 θ_state/θ_reward/θ_pred。每次切换 Agent 视角必须重新调用：

- **采样 Agent i 的动作**：`set_context(rule, id_i)` → `predict(s)` 获取 logits
- **状态转移**：θ_state 仅依赖 rule（客观），任意 `set_context` 后可调用
- **Agent j 的奖励/价值**：`set_context(rule, id_j)` → `predict_reward(s, a)` 或 `predict(s)`

Baseline 模型无 `set_context`，通过 `get_id_emb(id)` 显式传入 id_emb。

#### 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `mve_samples` | 50 | 每个 Agent 的采样总数（= A × scenarios = 5 × 10） |
| `mve_depth` | 5 | 展开步数 K |
| `mve_temperature` | 1.0 | π_mve softmax 温度 |

### 4.2 MCTS Planner (Pro Mode, Phase 5+ 预留)
- 标准 MuZero MCTS，使用超网络生成的 DynNet 模拟
- 暂不实现，预留接口

---

## 五、训练流程 (MuZero Unroll)

### 5.1 Episode Replay Buffer

**存储项（per episode）**:

| 字段 | Shape | 说明 |
|------|-------|------|
| `obs` | $(T, O_{joint})$ | 联合观测序列 |
| `actions` | $(T, N_{agents})$ | 离散动作序列 (int) |
| `rewards` | $(T, N_{agents})$ | per-agent 真实奖励 |
| `search_policies` | $(T, N_{agents}, 5)$ | Planner 输出的 π_mve |
| `rule` | scalar | 该episode的Rule值 |
| `dones` | $(T,)$ | 终止标志 |

**Buffer 配置 (v4.4)**：
- `buffer_size = 5000`：FIFO 循环缓冲区最大 episode 数。从 20000 缩减至 5000 以提升数据新鲜度——无 reanalyze 架构下，旧 π_mve 目标会过时，过大 buffer 导致训练信号被陈旧数据稀释。
- `min_buffer_size = 1000`：Warmup guard，训练开始前需积累 ≥1000 个有效 episode，确保初始采样多样性。
- `episodes_per_iter = 32, train_steps_per_iter = 8`：Replay ratio ≈ 2.56（原 ≈ 82），大幅降低数据复用次数。Buffer 完整刷新周期从 20000 步缩短至 1250 步。

**采样方式**：
1. 均匀随机采样 episode（v4.4: 移除 PER，原 reward-magnitude 优先级采样与小 buffer 数据新鲜度策略冲突）
2. 在该 episode 内随机选起始位置 t（确保 t+K ≤ T）
3. 取 [t, t+K] 的连续序列

### 5.2 N-step Return (训练时实时计算, v4.4 使用 Target Network)

> **v4.4 变更**：Bootstrap value 改用 Target Network 计算，阻断 V 过估的正反馈循环。详见 §5.7。

```python
def compute_n_step_return(rewards_i, target_model, obs_seq, agent_id, params, n, gamma):
    """
    训练时用 Target Network 的 V 计算 per-agent n-step return
    rewards_i: [B, T] - agent_i 的真实奖励序列
    target_model: EMA 慢更新的模型副本 (v4.4)
    """
    z = torch.zeros(B)
    for k in range(n):
        z += gamma**k * rewards_i[:, k]
    # [v4.4] Bootstrap with Target V (EMA-smoothed, resistant to oscillation)
    with torch.no_grad():
        s_n = target_model.repr_net(obs_seq[:, n])
        _, v_n = target_model.pred_net(s_n, params['pred'][agent_id])
    z += gamma**n * v_n.squeeze()
    return z
```

### 5.3 训练步骤 (The Unroll Loop) — v4.6 更新

> **v4.0 变更**：
> 1. Consistency Loss 从 MSE(s, s_target) 升级为 **Projection + Negative Cosine Similarity**
> 2. Total Loss 循环结束后除以 K（平均化），防止梯度累积过大
> 3. Exp3 额外加入 Hinge Variance Context Loss
>
> **v4.4 变更**：
> 4. N-step return 的 bootstrap value 改用 **Target Network (EMA)** 计算（见 §5.7）
> 5. 学习率调度器从 MultiStepLR 改为 **CosineAnnealingLR**
> 6. Adam 优化器 eps 改为 1e-5
>
> **v4.6 变更**：
> 7. Policy loss 恢复为纯 CE（移除 PG 辅助 loss + 熵门控，见 §5.8 + §5.9）

```python
def train_step(self):
    batch = buffer.sample_sequence(batch_size, unroll_K)
    # batch.obs:     [B, K+1, O]
    # batch.actions: [B, K, N]
    # batch.rewards: [B, K, N]
    # batch.pi_mve:  [B, K, N, 5]
    # batch.rule:    [B]

    # 1. 视角采样 (可选: 阶段性冻结控制, 见 §5.4)
    agent_ids = sample_agent_ids(batch_size, active_agents)  # None=全员, [0,1,2]=猎人, [3]=猎物

    # 2. 生成参数 (per-sample perspective) — 在线模型
    rule_emb = rule_encoder(batch.rule)
    id_emb = id_embedding(agent_ids)
    θ_state = hyper_trans(rule_emb)
    θ_reward = hyper_rew(cat([rule_emb, id_emb]))
    θ_pred = hyper_pred(cat([rule_emb, id_emb]))

    # 2.1 [v4.4] 设置 Target Model 上下文 (用于 bootstrap)
    #   Oracle: target_model.set_context(rule, agent_ids) — target 超网络生成 target θ
    #   Infer:  target_model.set_context(online_rule_emb, agent_ids) — 在线 GRU 推断 + target 超网络
    target_model.set_context(...)

    # 3. 初始编码
    s = repr_net(batch.obs[:, 0])  # [B, S]

    total_loss = 0
    for k in range(K):
        # 3.1 Prediction (该视角)
        p_k, v_k = pred_net(s, θ_pred)  # [B, 5], [B, 1]

        # 3.2 Dynamics
        s_next = state_trans_net(s, batch.actions[:, k], θ_state)  # [B, S]
        r_k = reward_head(s, batch.actions[:, k], θ_reward)        # [B, 1]  ← r=R(s,a)

        # 3.3 Targets
        target_pi = batch.pi_mve[:, k, agent_ids]    # [B, 5]
        target_r  = batch.rewards[:, k, agent_ids]    # [B]
        target_z  = compute_n_step_return(...)         # [B] 使用 target_model 计算 (v4.4)

        # 3.4 Consistency target — [v4.0] Projection + Cosine Similarity
        proj_pred = projector(s_next)                  # [B, proj_dim]
        with torch.no_grad():
            s_target = repr_net(batch.obs[:, k+1])     # 注意: consistency 仍用在线 RepNet
            proj_target = projector(s_target)           # [B, proj_dim]

        # 3.5 Losses
        loss_policy  = F.cross_entropy(p_k, target_pi)
        loss_value   = F.mse_loss(v_k.squeeze(), target_z)
        loss_reward  = F.mse_loss(r_k.squeeze(), target_r)
        loss_consist = negative_cosine_similarity(proj_pred, proj_target)  # [v4.0]

        total_loss += (w_policy * loss_policy
                     + w_value  * loss_value
                     + w_reward * loss_reward
                     + w_consist * loss_consist)

        # 3.6 梯度半衰 (保留)
        s = 0.5 * s_next + 0.5 * s_next.detach()

    # 4. [v4.0] Loss 平均化 (除以展开步数 K，在循环结束后)
    total_loss = total_loss / K

    # 5. [v4.0][Exp3 only] Context 正则化 (Hinge Variance)
    if is_infer_mode:
        ctx_loss = context_variance_loss(rule_emb, target_std=0.1)
        total_loss += w_context * ctx_loss

    # 6. 统一 backward
    optimizer.zero_grad()
    total_loss.backward()
    clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    # 7. [v4.4] EMA 更新 Target Network
    _soft_update_target(target_model, model, tau=ema_tau)

    # 8. [v4.4] CosineAnnealingLR 调度
    scheduler.step()
```

### 5.4 阶段性冻结 — 对抗训练节奏控制 (v4.2 新增)

#### 动机

在猎人-猎物对抗环境中，双方策略同时更新会导致**策略循环震荡**：猎人学会了追击策略 → 猎物适应逃跑 → 猎人策略失效 → 重新学习 → 循环往复。通过控制视角采样节奏，让一方暂时"静止"，使另一方有时间收敛到稳定策略。

#### 核心机制：Phased Perspective Sampling

**不冻结网络参数**，而是通过控制 `train_step` 中 `agent_ids` 的采样范围来限制哪些 agent 视角的主观 Loss（policy/value/reward）参与训练。

- 世界模型（RepNet/StateTransNet）持续收到 consistency loss 梯度 → **客观学习不中断**
- 只有主观网络（PredNet/RewardHead）的特定视角暂停进化

#### Phase 调度时间线

```
步数:  0 ─────── 5000 ──────── 15000 ──────── 25000 ──────── 35000 ──── ...
Phase: [ Warmup (全员) ] [ Hunters ] [  Prey  ] [ Hunters ] [  Prey  ] ...
```

| 阶段 | active_agents | 训练的主观网络 | 静止的主观网络 |
|------|--------------|--------------|--------------|
| Warmup (< warmup_steps) | [0,1,2,3] | 全部 | 无 |
| Hunter Phase (偶数) | [0,1,2] | Hunter/Agent2 视角的 PredNet+RewardHead | Prey 视角 |
| Prey Phase (奇数) | [3] | Prey 视角的 PredNet+RewardHead | Hunter/Agent2 视角 |

#### 实现

```python
def get_active_agents(train_steps, cfg):
    if not cfg.freeze_enabled:
        return None, 'All'
    if train_steps < cfg.freeze_warmup_steps:
        return None, 'Warmup'
    elapsed = train_steps - cfg.freeze_warmup_steps
    phase_idx = elapsed // cfg.freeze_phase_steps
    if phase_idx % 2 == 0:
        return cfg.freeze_hunter_agents, 'Hunters'  # [0, 1, 2]
    else:
        return cfg.freeze_prey_agents, 'Prey'        # [3]

# In MuZeroTrainer.train_step():
def _sample_agent_ids(self, B, active_agents=None):
    if active_agents is None:
        return torch.randint(0, self.cfg.num_agents, (B,))
    else:
        source = torch.tensor(active_agents, device=self.device)
        return source[torch.randint(0, len(active_agents), (B,))]
```

#### 对不同实验的影响

- **HyperMuZero (Exp2/3)**：HyperNet 为不同 agent 生成不同 θ_pred/θ_reward。不采样某 agent 的视角 → 对应 HyperNet 映射区域无梯度 → 该 agent 策略真正冻结。✅ 效果最佳
- **Baseline (Exp1)**：共享权重但输入不同 (`cat([s, id_emb])`)。不采样 agent_i → 网络对 id_i 输入区域的响应不被主动修正。虽有共享层干扰，但作为 Baseline 已足够。✅ 可接受

#### 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `freeze_enabled` | True | 总开关，设 False 可完全关闭 |
| `freeze_warmup_steps` | 4000 | 全员训练预热期 (v4.4: 5000→4000) |
| `freeze_phase_steps` | 2400 | 每个阶段持续训练步数（对称）(v4.4: 10000→2400，与 lr 衰减对齐) |
| `freeze_hunter_agents` | [0, 1, 2] | Hunter 阶段活跃 agent |
| `freeze_prey_agents` | [3] | Prey 阶段活跃 agent |

> **禁用对照**：设 `freeze_enabled = False` 即可回退到原始全员训练，方便 A/B 对比。

### 5.5 Projection Head 与 Cosine Similarity (v4.0 新增)

```python
class Projector(nn.Module):
    """非线性映射头，用于 Consistency Loss 的投影空间"""
    def __init__(self, latent_dim, proj_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
    def forward(self, x):
        return self.net(x)

def negative_cosine_similarity(p, z):
    """p: predicted projection, z: target projection (detached)"""
    p = F.normalize(p, p=2, dim=-1)
    z = F.normalize(z, p=2, dim=-1)
    return -(p * z).sum(dim=-1).mean()
```

> **Projector 共享**：RepNet 输出和 StateTransNet 输出共用同一个 Projector。三组实验 (Exp1/2/3) 全部使用 Projection + Cosine Sim，确保对比公平。

### 5.6 Hinge Variance Context Loss (v4.0 新增, Exp3 专用)

```python
def context_variance_loss(rule_emb, target_std=0.1):
    """
    鼓励 batch 内 rule_emb 的标准差不低于 target_std
    防止 embedding 坍缩，替代不稳定的 1/var 正则
    """
    batch_std = torch.std(rule_emb, dim=0).mean()  # per-dim std 的均值
    return torch.relu(target_std - batch_std)
```

### 5.7 Target Network (EMA) + 优化器改进 (v4.4 新增)

#### 问题诊断

训练中后期出现 loss 波动上升、reward 停滞的现象，根因定位为两个独立但相互放大的因素：

1. **无 Target Network**：N-step return 的 bootstrap value $V(s_{t+n})$ 使用在线模型最新参数计算。当 $V$ 偶然过估时，target 被抬高 → $V$ 进一步过估 → 正反馈不受控。
2. **Phase 切换冲击**：每 2400 步活跃 agent 从 $\{0,1,2\} \leftrightarrow \{3\}$ 突变，数据分布剧变。在线 $V$ 立即反映新分布，导致 n-step target 不稳定。

#### 因果机制

```
Phase 切换 (每 2400 步)
  │
  ├─→ 活跃 agent 突变 → reward/policy target 分布突变
  │
  └─→ 在线 V 立即反映新分布
        │
        └─→ n-step target = Σr + γ^n · V_online
              │
              ├─→ V_online 不稳定 → target 不稳定 → l_val 波动
              │     └─→ pred_net 权重震荡 → l_pol 波动
              │
              └─→ V 过估 → target 被抬高 → V 进一步过估
                    (无 target network 阻尼 → 正反馈不受控)
```

#### 改动 1: Target Network (EMA) — 核心

维护在线模型的 EMA 慢更新副本，用于计算 n-step return 的 bootstrap value：

```python
# 初始化
target_model = deepcopy(model)
target_model.requires_grad_(False)  # 不参与梯度计算

# 每步 optimizer.step() 之后
def _soft_update_target(target_model, online_model, tau=0.99):
    for tp, op in zip(target_model.parameters(), online_model.parameters()):
        tp.data.mul_(tau).add_(op.data, alpha=1 - tau)

# compute_n_step_return 中
with torch.no_grad():
    s_boot = target_model.encode(obs_seq[:, bootstrap_idx])  # target repr_net
    id_emb = target_model.get_id_emb(agent_ids)
    _, v_boot = target_model.predict(s_boot, id_emb)          # target pred_net (权重由 target hyper 生成)
```

**超网络特殊处理**：EMA 作用在超网络参数本身（repr_net + hyper_net + context_encoder），而非超网络生成的权重 θ。deepcopy 保留模型类型，API 兼容。

**Exp3 (Infer) 的 context 设置**：

```python
# 在线 GRU 推断 rule_emb（推断准确性需要快速更新）
model.set_context_from_history(hist_obs, hist_actions, hist_rewards, agent_ids, hist_mask)
online_rule_emb = model.get_current_rule_emb()

# Target 超网络生成权重（V 预测需要慢速稳定）
target_model.set_context(online_rule_emb, agent_ids)
```

**设计决策**：GRU 推断用在线版本（快速适应），V 预测用 target 版本（稳定 bootstrap）。

#### 改动 2: CosineAnnealingLR — 辅助

替换 MultiStepLR 的阶梯式衰减为平滑的 Cosine 退火：

```python
# Before (v4.3): MultiStepLR — 在 milestone 处 lr 突降 70%，可能引发训练不稳定
scheduler = MultiStepLR(optimizer, milestones=[28000, 66400], gamma=0.3)

# After (v4.4): CosineAnnealingLR — 平滑衰减，末期收敛到 lr_min 而非 0
scheduler = CosineAnnealingLR(optimizer, T_max=max_train_steps, eta_min=1e-5)
```

| 对比 | MultiStepLR | CosineAnnealingLR |
|------|-------------|-------------------|
| 衰减方式 | 阶梯跳崖 | 平滑曲线 |
| 末期 lr | 9e-6 (第 66.4k 步后固定) | 1e-5 (持续可学习) |
| 与冻结交替兼容性 | lr 跳变叠加 phase 切换 → 双重冲击 | 平滑过渡，无叠加风险 |
| 500k 步利用率 | 87% 步数在极低 lr 下浪费 | 全程有效学习 |

#### 改动 3: Adam eps = 1e-5 — 辅助

```python
optimizer = torch.optim.Adam(params, lr=cfg.lr, eps=1e-5)
```

当梯度二阶矩 $v$ 很小时，默认 $\epsilon=10^{-8}$ 导致更新量 $\frac{m}{\sqrt{v}+\epsilon}$ 爆炸。$\epsilon=10^{-5}$ 提供隐式更新量上限，与 grad_clip 互补。

#### 配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `ema_tau` | 0.99 | Target Network EMA 系数（每步更新 1%） |
| `lr_min` | 1e-5 | CosineAnnealingLR 最小学习率 |
| `adam_eps` | 1e-5 | Adam 优化器 epsilon |

#### 不改动的项

| 项目 | 理由 |
|------|------|
| loss_consist / projector | 工作正常，0.0004-0.0005 已收敛 |
| w_consist = 0.5 | 不需要调整 |
| 网络结构 | 无任何架构变化 |
| 奖励函数 | v4.3 正交化已完成 |

### 5.8 Planner 均匀分布修复历程 (v4.5 → v4.6)

#### 原始问题

训练中 `l_pol` 恒为 $\ln 5 = 1.6094$，即 π_mve 退化为均匀分布。

#### 第一层根因：联合动作空间爆炸 (v4.5 已解决)

```
4 agents × 5 actions → 联合空间 625/step
  └─→ 50 samples 覆盖率 ≈ 0.08 → softmax → 均匀分布
```

**v4.5 解决方案**：Per-Agent Coordinate Descent，搜索空间从 625/step 降到 5/step（详见 §4.1）。

#### 第二层根因：方差淹没信号 (v4.6 修复)

Coordinate Descent 解决了搜索空间问题，但 π_mve 仍接近均匀。原因不是"预测噪声"，而是**其他 Agent 独立采样的方差**淹没了候选动作间的真实信号差异：

```
评估 agent j 的 5 个候选动作时，其他 3 个 Agent 的 step-0 动作独立采样:

信号（agent j 动作差异）:    ~0.05 / step
噪声（其他 agent 随机动作）:  ~1.0 / step
SNR ≈ 0.05

10 samples 标准误差 ≈ 1.0 / √10 ≈ 0.32
信号 0.05 << 标准误差 0.32 → 完全淹没 → softmax → 均匀
```

**v4.6 解决方案**：Common Random Numbers (CRN)。评估不同候选动作时，让其他 Agent 的 step-0 动作**每 scenario 只采样一次**，复制给所有 A 个候选。噪声完全抵消，SNR → ∞（详见 §4.1）。

#### 改动文件

| 文件 | 改动 |
|------|------|
| `planning/mve_planner.py` | 数据布局从 (B,A,spa) 改为 (B,spa,A)；新增 CRN pre-sampling 逻辑 |
| `config.py` | 无变更（mve_samples/depth/temperature 不变） |
| Worker / Trainer | **无改动**（接口 `sample_mve_plan(model, root_s, cfg, rule)` 保持不变） |

### 5.9 回退 Policy Gradient 辅助 Loss (v4.6)

#### v4.5 曾引入的 PG + 熵门控 CE

v4.5 为解决"π_mve 均匀 → CE 拉 policy 回均匀"的死锁，引入了 PG 辅助 loss + 熵门控 CE：

```python
# v4.5 (已移除):
loss_policy = w_ce * ce_gate * loss_ce + w_pg * loss_pg - w_ent * entropy
```

#### l_pg 失控的根因：Off-Policy PG 的根本缺陷

实验观测：

```
Iter  250: l_pg = -0.003   (正常)
Iter 2050: l_pg = -1.66    (异常)
Iter 2450: l_pg = -2.46    (失控)
```

`loss_pg = -(log_prob_taken × adv).mean()` 假设 `a_taken ~ π_current`。但 Buffer 中的动作来自旧策略（高 ε 时代）。当前 policy 远离旧策略时：

```
Buffer 中存储着 ε=0.5 时代的动作（大量随机动作）
  └─→ 当前 policy 给这些旧动作赋予很低概率
        └─→ log_prob_taken 变得非常负 (如 -4, -5...)
              └─→ 对 adv < 0 的样本: -(很负 × 负) = -(正) → 很负的 loss
                    └─→ 梯度推动 policy 进一步降低这些动作的概率
                          └─→ log_prob 更负 → loss 更负 → 正反馈循环
```

**本质问题**：PG 梯度 $\nabla \log \pi(a|s) \times \text{adv}$ 假设 $a$ 是从当前 $\pi$ 采样的。但 $a$ 是从旧策略采样的。这个 off-policy bias 导致 policy 优化的是"不做随机策略做过的事"，而非"做能获得高奖励的事"。

#### v4.6 决策：移除 PG，恢复纯 CE

CRN 修复后 Planner 能产出非均匀 π_mve，标准 CE 即可提供有效学习信号，无需 PG 辅助：

```python
# v4.6 (当前):
loss_policy = CE(p_k, target_pi)  # 标准交叉熵，无门控
```

#### 配套参数恢复

| 参数 | v4.5 值 | v4.6 值 | 原因 |
|------|---------|---------|------|
| `epsilon_init` | 0.5 | 1.0 | 无 PG，恢复标准 ε-greedy |
| `epsilon_decay_steps` | 20000 | 28000 | 恢复标准衰减节奏 |
| `w_ce`, `w_pg`, `w_entropy` | 1.0, 1.0, 0.01 | **移除** | 不再需要 |

#### 改动文件

| 文件 | 改动 |
|------|------|
| `training/muzero_trainer.py` | `train_step` + `train_step_infer` 移除 PG/entropy/gate |
| `config.py` | 移除 `w_ce`, `w_pg`, `w_entropy`；恢复 `epsilon_init/decay` |
| `scripts/train_*.py` | 控制台 print 移除 `l_pg`, `ce_g`；适配新 loss key names |

### 5.10 AdaLN 信号衰减修复 (v4.6)

#### 问题诊断

CRN (§5.8) 修复后，调试输出显示 CRN 确实消除了其他 Agent 的随机噪声，但 π_mve 仍为均匀分布。进一步检查发现**所有 5 个候选动作产生完全相同的 reward 预测**（差异为 0）：

```
[Step-0 Reward] 5 candidates: [-0.10098863 -0.10098863 -0.10098863 -0.10098863 -0.10098863]
  range=0.000000, std=0.000000
```

#### 第三层根因：AdaLN `h * gamma` 信号衰减

HyperNet 生成的 flat_params 经过 L2 归一化 + output_scale (0.01) 缩放后，每个参数值 ≈ 0.01/√28000 ≈ 6×10⁻⁵。AdaLN 公式 `h * gamma + beta` 中 gamma 同样来自 flat_params，因此 gamma ≈ 6×10⁻⁵。

**信号衰减链**：

```
FC1: x = [state, action_onehot] → W₁x + b₁ → LayerNorm → h_norm (单位尺度)
AdaLN: h_norm * gamma + beta ≈ O(1) * 6e-5 + 6e-5 ≈ 1.2e-4 ← 信号被压缩到 10⁻⁴

FC2: W₂ @ h₁ ≈ 128 * 6e-5 * 1.2e-4 ≈ 10⁻⁶  (远小于 b₂ ≈ 6e-5)
     → bias 主导，输入信号丢失 → 所有候选动作计算出相同的 reward

结果: action 信息在 float32 精度下完全消失
```

#### 修复方案：残差调制 `(1 + gamma)`

将 AdaLN 仿射变换从 `h * gamma` 改为 `h * (1 + gamma)`：

```python
# Before (v3.2): gamma ≈ 0 → output ≈ 0 → 信号死亡
h = h * gamma + beta

# After (v4.6): gamma ≈ 0 → output ≈ h_norm → 恒等传递
h = h * (1 + gamma) + beta
```

**效果**：当 gamma ≈ 0（训练初期 / output_scale 小），层表现为恒等传递，归一化后的信号以单位尺度传播。action 信息在所有 Functional Net 层中保持可检测：

```
FC1: h_norm * (1 + 6e-5) + 6e-5 ≈ h_norm (单位尺度)
FC2: W₂ @ h₁ ≈ 128 * 6e-5 * 1.0 ≈ 8e-3 (比修复前大 10⁵ 倍!)
→ action 变化 (10% of h_norm) 可传播到最终输出
```

此模式是 DiT (Scalable Diffusion Models with Transformers)、StyleGAN、FiLM 等条件归一化架构的标准做法。

#### 改动文件

| 文件 | 改动 |
|------|------|
| `models/functional_nets.py` | `adaln_forward`: `h * gamma` → `h * (1 + gamma)` |
| `planning/mve_planner.py` | 新增 Level 5 诊断：action_onehot 差异、s_next 差异、output_scale 值 |

### 5.11 Reward HyperNet 分化改进 (v4.7)

#### 问题诊断

训练监控发现 `hyper_rew` 的 θ_reward 在 agent 2 vs agent 3 配对上余弦相似度持续坍缩（cos_sim ≈ 0.88，无下降趋势），而其他 agent 对（如 0v2, 0v3, 1v2）均在正常分化（cos_sim 从 0.85 降至 0.58-0.69）。hyper_pred 的分化效果相对正常。

**坍缩不是全局性的，而是特异性地发生在 (2,3) 配对上。**

#### 根因分析：output_scale 初始化陷阱

**核心根因：`output_scale=0.01` 导致 AdaLN 调制在训练初期完全无效**

```
θ_rew 有 28033 个参数
L2 归一化后每个参数 ≈ 1/√28033 ≈ 0.006
× output_scale(0.01) → 每个参数 ≈ 0.00006

AdaLN: h * (1 + gamma) + beta
gamma ≈ 0.00006 → (1 + 0.00006) ≈ 1.0
→ 所有 agent 的 reward prediction 几乎完全相同
→ reward MSE 对不同 agent 产生几乎相同的梯度
→ 无分化信号
```

output_scale 是可学习的，但增长速度受 l_rew 梯度控制：

```
l_rew ≈ 0.05（弱梯度源），l_pol ≈ 1.5（强梯度源，30x）

hyper_rew 的 output_scale 增长: ~2.2e-5/step → 从 0.01 到 0.1 需 ~4000 步
hyper_pred 的 output_scale 增长: ~6.6e-4/step → 从 0.01 到 0.1 需 ~130 步
```

**在 output_scale 增长到有意义的值之前（~4000 步），hyper_rew 的 MLP 权重已在"零区分"环境中坍缩。** 这解释了为什么 hyper_pred（30x 更快的 output_scale 增长）能正常分化而 hyper_rew 不能——不是架构问题，而是初始化 + 梯度幅度的组合导致 hyper_rew 错过了分化窗口。

**(2,3) 特异性坍缩的附加因素**：Agent 2（可变角色）的 reward 在不同 rule 下方向冲突（rule=0 是猎人，rule=1 偏向 prey），使 hyper_rew 更难学习 (rule, id=2) 的复杂交互映射。

#### 解决方案：四项改动

**改动 1: hyper_rew output_scale 初始化提升 — 打破初始化陷阱（核心）**

将 hyper_rew 的 `output_scale_init` 从 0.01 提升到 0.1：

```python
class HyperNetMLP(nn.Module):
    def __init__(self, ..., output_scale_init=0.01):
        self.output_scale = nn.Parameter(torch.tensor(float(output_scale_init)))

class DualHyperNetwork(nn.Module):
    def __init__(self, ..., rew_output_scale_init=0.01):
        self.hyper_rew = HyperNetMLP(..., output_scale_init=rew_output_scale_init)
        # hyper_trans 和 hyper_pred 保持默认 0.01
```

**效果**：初始 θ_rew 参数放大 10x → AdaLN 调制从 (1+6e-5)≈1.0 到 (1+6e-4)≈1.0006 → 虽仍小，但不同 agent 的 random MLP 输出方向差异被 10x 放大 → reward prediction 产生可测量的 agent 间差异 → reward MSE 立即产生分化梯度。

**仅影响 hyper_rew**：hyper_trans 和 hyper_pred 保持 0.01，避免影响已正常工作的部分。

**改动 2: Reward Diversity Loss — 直接对抗方向坍缩**

显式惩罚不同 agent 的 θ_reward 方向过于相似。使用 Hinge 机制，cos_sim 降到阈值以下后自动关闭：

```python
def reward_diversity_loss(hyper_rew, rule_emb, id_embedding, num_agents,
                          target_cos=0.5, skip_pairs=None):
    thetas = {}
    for aid in range(num_agents):
        id_emb = id_embedding(full(B, aid))
        aug_ctx = cat([rule_emb, id_emb], dim=-1)
        thetas[aid] = hyper_rew(aug_ctx)
    loss = 0
    for i, j in agent_pairs:  # 排除 skip_pairs
        cos_sim = cosine_similarity(thetas[i], thetas[j], dim=-1)
        loss += relu(cos_sim - target_cos).mean()
    return loss / count
```

**设计决策**：
- `target_cos = 0.5`：只管真正坍缩的 pair（如 2v3 ~0.88 → penalty 0.38），不干扰自然分化到 0.6-0.7 的 pair（如 0v3 ~0.58 → penalty 0.08）
- `skip_pairs = [(0, 1)]`：Agent 0 和 1 是同角色固定猎人，θ_reward 相似是正确的
- **cos_sim 是 scale-invariant 的**：output_scale 改变不影响 diversity loss → 两个机制互补而非重叠

**改动 3: 加深 hyper_rew MLP — 增强 (rule, id) 交互建模**

hyper_rew 从 [256, 256] 加深为 [256, 256, 256]。更深的 MLP 有更强的 rule_emb × id_emb 交互建模能力，这正是 (2,3) 坍缩的关键——需要学习 agent 2 在不同 rule 下的复杂条件映射。

**改动 4: w_reward 温和放大 — 搭配 output_scale 综合提升**

`w_reward` 从 1.0 提升到 3.0（非 5.0，因真正的杠杆在 output_scale）。

```
梯度到达 context_encoder 的贡献比:
  w=1, scale=0.01: ∝ 0.01  → ~1%（完全被淹没）
  w=3, scale=0.1:  ∝ 0.3   → ~30%（有意义的贡献）
  w=5, scale=0.01: ∝ 0.05  → ~5%（仍不够）
```

3.0 + output_scale=0.1 的综合贡献比 ~30%，给了 reward pathway 有意义的影响力，且避免 5.0 对已正常分化 pair 的过度放大。

#### 四项改动的协作

```
训练初期 (ε=1.0):
  output_scale=0.1 → θ_rew 产生可测量的 agent 间差异 → 打破初始化陷阱
  Diversity Loss → 在方向空间推开 θ_reward 向量 → 打破方向坍缩
  w_reward=3.0 → 放大 reward MSE 的自然分化梯度

训练中期 (策略改善):
  Diversity Loss → cos_sim < 0.5 后对正常 pair 自动关闭
  output_scale 继续增长 → reward prediction 差异越来越显著
  深层 hyper_rew → 创建更精细的 agent-specific 参数区域

训练后期 (收敛):
  Diversity Loss → 仅在偶尔回升时激活（安全网）
  所有梯度来自自然的 reward/policy/value loss
```

#### LR Schedule: Warmup Cosine (v4.7)

训练数据显示约 5k 步时出现不稳定震荡（GRU 表征相变 + replay buffer 数据分布变化）。引入 warmup_cosine LR 调度，在 5000 步内从 lr_min 线性升至 lr，然后余弦衰减：

```python
# config.py
lr_schedule = 'warmup_cosine'   # 'cosine' | 'multistep' | 'warmup_cosine'
lr_warmup_steps = 5000          # 覆盖 GRU 相变期 + buffer 分布变化期
```

#### 阶段性冻结改为全员训练 (v4.7)

v4.2 引入的阶段性冻结将多任务学习问题转化为 continual learning，导致灾难性遗忘。v4.7 恢复全员训练（`freeze_enabled = False`），依赖超网络通过 context 输入区分不同 agent 视角来避免负迁移。

#### 监控指标

| 指标 | TensorBoard key | 预期行为 |
|------|----------------|---------|
| θ_rew 余弦相似度 | `hyper/*/cos_rew_{i}v{j}` | (2,3) 从 ~0.88 降至 < 0.5；其他 pair 自然分化 |
| output_scale 增长 | `hyper/output_scale_rew` | 从 0.1 逐步增长（对比 trans/pred 从 0.01 增长） |
| 梯度范数 | `train/grad_norm` | 监控 5k 步附近是否有 spike |
| diversity loss | `train/l_div` | 初始正值，随分化推进逐步下降 |

#### 配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `rew_output_scale_init` | 0.1 | hyper_rew 的 output_scale 初始值（打破初始化陷阱） |
| `hyper_rew_hidden_dims` | [256, 256, 256] | hyper_rew 独立隐藏层（更深） |
| `w_reward` | 3.0 | Reward Loss 权重（原 1.0，搭配 output_scale=0.1 综合贡献比 ~30%） |
| `w_rew_diversity` | 0.1 | Diversity Loss 权重 |
| `rew_diversity_target_cos` | 0.5 | Hinge 阈值（只管真正坍缩的 pair） |
| `rew_diversity_skip_pairs` | [(0, 1)] | 跳过同角色 agent 对 |
| `lr_schedule` | 'warmup_cosine' | LR 调度策略 |
| `lr_warmup_steps` | 5000 | Warmup 步数（覆盖 5k 不稳定期） |
| `freeze_enabled` | False | 全员训练（移除阶段性冻结） |

#### 改动文件

| 文件 | 改动 |
|------|------|
| `config.py` | 新增 rew_output_scale_init, lr_schedule, lr_warmup_steps; w_reward 1.0→3.0; target_cos 0.3→0.5 |
| `models/hyper_network.py` | HyperNetMLP 支持 output_scale_init; DualHyperNetwork 支持 rew_output_scale_init + rew_hidden_dims; 新增 reward_diversity_loss() |
| `models/hyper_muzero_model.py` | 传递 rew_hidden_dims + rew_output_scale_init |
| `models/infer_muzero_model.py` | 传递 rew_hidden_dims + rew_output_scale_init |
| `training/muzero_trainer.py` | 新增 _build_scheduler (warmup_cosine); diversity loss 计算; 梯度范数 logging; output_scale 监控 |
| `scripts/train_oracle.py` | 控制台新增 l_div 输出 |
| `scripts/train_infer.py` | 控制台新增 l_div 输出 |

---

## 六、三组对比实验

| 实验 | 名称 | Context 来源 | HyperNet | 特点 |
|------|------|-------------|----------|------|
| **Exp1** | Baseline | agent_id 拼接输入 | ❌ 固定权重 | 无规则适应能力 |
| **Exp2** | Oracle-HyperMuZero | Rule(显式) + agent_id | ✅ DualHyperNet | 训练时已知Rule |
| **Exp3** | Infer-HyperMuZero | GRU推断Rule + agent_id | ✅ DualHyperNet | 训练测试都不知Rule |

### Exp1 Baseline 细节
- 网络结构与 Exp2/3 完全相同
- DynNet state head: 固定权重，输入 (s, A)
- DynNet reward head: 固定权重，输入 (s, A, id_emb)  — r=R(s,a) 因果性
- PredNet: 固定权重，输入 (s, id_emb)
- 同样使用 MVE Planner + MuZero 训练

### Exp2 Oracle 细节
- rule 直接输入 ContextEncoder
- HyperNet(rule) → θ_state
- HyperNet(rule, id) → θ_reward, θ_pred
- 测试时不提供 rule → 用训练集平均 context 作为 fallback

### Exp3 Infer 细节
- GRU 从历史 (obs, action, reward) 推断 rule_embedding
- 替代显式 rule 输入
- 额外 loss: context 正则化 $L_{ctx}$
- episode 初期推断不准，随步数增加趋于准确

---

### 5.12 HyperNet 参数效率 + 表达力 (LoRA / lora_fc2)

主轴是 `gen_scope`（`film_head` vs `base_gen`）。在其上叠加两条正交的轴，均为 opt-in：

**(1) Output-layer LoRA（参数效率轴，三路统一）。** 在 `film_head` 下，HyperNet 约 92% 的参数集中在 output_layer（尤以 `hyper_trans` 的 `Linear(256→8768)` 生成 `StateTransNet.fc3` 的 128×64 权重，占全部 HyperNet 参数 ~76%）。将每个 `HyperNetMLP.output_layer` 的 `Linear(prev, θ_pc)` 分解为 `Linear(prev, r, bias=False) → Linear(r, θ_pc)`（`hyper_output_rank=r`，默认关闭）。统一施加于 `hyper_trans / hyper_rew / hyper_pred` 三路（单一 cfg 字段，单一 rank）。`r=32` 时 HyperNet 参数：`film_head` 3.09M→~694k，`base_gen` 15.6M→~2.30M。初始化：A 用 `orthogonal_init`，B 用 `small_init(std=0.01)`（**不可为 0**——`B=0` 时 `raw=0`，分组 RMS 除以 `1e-8` 下限会在 step-0 产生 ~1e4 梯度尖峰）。`share_subjective_trunk=True`（SubjectiveHyperNet）暂未接线，二者同开会抛 `NotImplementedError`。

**(2) `lora_fc2`（表达力轴，仅 film_head）。** `film_head` 因 fc1/fc2 冻结 + FiLM 为对角调制，缺乏 per-context 的特征**混合**能力。新增 gen_scope `lora_fc2` = `film_head` + 对共享 fc2 的 per-context 低秩权重增量：`W2_eff = W2_base + Bf·Af`（秩 r，默认 8）。生成量 = `film_head_count + 256·r`，是 `film_head` 与 `base_gen`（整权重生成）之间的"中间档"。`base_gen` 下被禁止（已整权重生成 fc2，ΔW 冗余，由 `ModelConfig.__post_init__` 断言拦截）。**尺度纪律（断言强制）**：`lora_fc2` 要求三路 `*_output_scale_init ≥ 0.05`（推荐 0.1）——分组 RMS 下每个生成元 ≈ `output_scale`，故 `ΔW ≈ output_scale²·√r`；在 0.1/r=8 时 ≈ 0.028（约 kaiming fc2 基权 0.088 的 32%，有效），在默认 0.01 时坍缩到 ~3e-4（失效）。`r=0` 与 `film_head` 逐字节等价（回归门）。

| 配置 (medium, A=6, N=4) | HyperNet 参数 |
|---|---|
| film_head + LoRA(r=32), lora_fc2 OFF | ~694k |
| film_head + LoRA(r=32) + lora_fc2 r=8 | ~896k |
| base_gen + LoRA(r=32) | ~2.30M |

**`hyper_rew`(3 层) vs `hyper_pred`(2 层) 深度不对称是有意保留的**：reward 需要更深的 type 分化（§5.11），value 不需要——不做"对称化修正"。预设：`{duo,medium}_film_lora` / `{duo,medium}_film_lora_fc2` / `{duo,medium}_base_lora`（2agent=duo 的 N=2 random_walk，4agent=medium 的 N=4 static；均 `share_subjective_trunk=False`）。全量实验（3 建模情形 × 2 环境 = 6 runs）由 `hyper_mve/scripts/run_lora_experiments.py` 编排，GPU 池默认 {2,3,4}（每进程 `CUDA_VISIBLE_DEVICES=<单卡>` 钉一张卡），结果按 `<env>/<model>[/<gen_scope>]` 落盘。

---

## 6.1 统一评估框架 (v4.1 新增)

### 设计原则

三组实验的 Model 接口差异较大（Baseline 传 `id_emb`、Oracle 传 `rule+id`、Infer 从历史推断），因此采用 **"各自 evaluate + 公共指标收集"** 的架构，而非一个通用 Evaluator 硬塞 `hasattr` 判断。

### 架构分层

```
┌─────────────────────────────────────────────────┐
│  scripts/train_xxx.py                            │
│    └── evaluate_xxx(model, env, ...)             │  ← 各实验独立的 evaluate 函数
│          ├── 模型调用逻辑（接口各异）              │     (保持各自的 set_context / predict 方式)
│          └── 返回统一格式的 EvalResult dict        │
│                                                   │
│  utils/evaluator.py                              │
│    └── Evaluator                                 │  ← 公共指标收集 + TensorBoard 日志
│          ├── run_eval_suite(evaluate_fn, ...)     │     对 test_rules=[0.0, 0.5, 1.0] 循环调用
│          ├── log_results(results, step)           │     统一写 TensorBoard
│          └── print_summary(results)              │     格式化打印对比表
└─────────────────────────────────────────────────┘
```

### EvalResult 统一格式

每次 evaluate 函数返回：
```python
{
    'mean_rewards': np.array([r0, r1, r2, r3]),  # per-agent 平均奖励
    'mean_total_reward': float,                    # 全体奖励之和
    'mean_ep_length': float,                       # 平均 episode 长度
}
```

### Evaluator 核心接口

```python
class Evaluator:
    def __init__(self, cfg, device, writer=None):
        self.test_rules = [0.0, 0.5, 1.0]  # 固定测试 Rule 集合
    
    def run_eval_suite(self, evaluate_fn, model, env, step, **kwargs):
        """
        对 test_rules 中的每个 Rule 调用 evaluate_fn，收集结果。
        
        Args:
            evaluate_fn: 各实验自己的 evaluate 函数 (model, cfg, device, ..., fixed_rule) -> EvalResult
            model: 当前模型
            env: 环境实例
            step: 当前训练步数
        Returns:
            dict[rule_value -> EvalResult]
        """
        
    def log_results(self, results, step, prefix='eval'):
        """写入 TensorBoard: per-rule 总奖励 + per-agent 分项"""
        # eval/{prefix}/rule_{rule}/total_reward
        # eval/{prefix}/rule_{rule}/agent_{i}_reward
        # eval/{prefix}/rule_{rule}/ep_length
        
    def print_summary(self, results, step):
        """格式化打印 Agent-wise 对比表"""
        # [Eval step=5000]
        # Rule | Agent0 | Agent1 | Agent2 | Agent3 | Total | EpLen
        # 0.0  |  -2.30 |  -1.80 |  +3.50 |  -5.20 |  -5.8 | 22.3
        # 0.5  |  +1.20 |  +1.50 |  +0.80 |  -8.00 |  -4.5 | 18.7
        # 1.0  |  +3.40 |  +3.10 |  +2.90 | -12.00 |  -2.6 | 15.2
```

### 环境 force_rule 机制

评估时通过 `env.reset(options={'rule': rule_val})` 强制指定 Rule，确保：
1. Rule 在 `reset_world` 内部生效（颜色、奖励函数均正确）
2. 无需在 reset 后手动覆盖 `env.world.rule`（消除时序漏洞）
3. 向后兼容：不传 options 时行为不变（随机采样）

### Agent-wise 关键指标

| 指标 | 说明 | 预期行为差异 |
|------|------|------------|
| Agent 0,1 (Hunter) reward | 追捕效率 | Rule↑ → reward↑ (更多帮手) |
| Agent 2 (Variable) reward | 角色适应 | Rule=1 猎人高分, Rule=0 保镖高分 |
| Agent 3 (Prey) reward | 生存能力 | Rule=0 → 有庇护更易存活 |
| Episode length | 博弈持续时间 | Rule=1 → 快速围剿(短), Rule=0 → 保卫战(长) |

---

## 七、代码文件结构（v3.1 重构）

```
D:/RL/hyper_mve/                      # 目录暂保留原名，避免破坏环境
├── config.py                          # [重写] 新架构配置
│
├── envs/                              # [修改] 支持离散动作
│   ├── __init__.py
│   ├── non_stationary_tag.py          # [修改] 确保离散兼容
│   ├── ns_environment.py              # [修改] DiscreteActionWrapper
│   └── make_env.py                    # [修改] discrete=True
│
├── models/                            # [大量重写]
│   ├── __init__.py
│   ├── interfaces.py                  # [重写] 新接口定义
│   ├── representation_net.py          # [保留] 微调
│   ├── functional_nets.py             # [新建] FunctionalMLP (支持外部权重)
│   ├── dynamic_net.py                 # [新建] StateTransNet + RewardHead (functional)
│   ├── prediction_net.py              # [新建] PredictionNet: policy+value (functional)
│   ├── context_encoder.py             # [新建] AugmentedContextEncoder (Rule+ID)
│   ├── hyper_network.py               # [新建] DualHyperNetwork (3路)
│   ├── hyper_muzero_model.py          # [新建] 总模型封装 (Exp2/3)
│   └── baseline_model.py             # [重写] Exp1 (固定权重+id_emb输入)
│
├── planning/                          # [新建] 规划器模块
│   ├── __init__.py
│   ├── mve_planner.py                 # [v4.5] Per-Agent Coordinate Descent MVE
│   └── mcts.py                        # [预留] MCTS Pro Mode
│
├── training/                          # [大量重写]
│   ├── __init__.py
│   ├── episode_buffer.py              # [重写] Episode序列Buffer
│   ├── muzero_trainer.py              # [新建] MuZero展开训练
│   ├── n_step_return.py               # [新建] 实时N-step return计算
│   └── worker.py                      # [新建] 数据收集worker
│
├── utils/
│   ├── __init__.py
│   ├── utils.py                       # 通用工具
│   └── evaluator.py                   # [v4.1新建] 公共评估指标收集+日志
│
├── scripts/
│   ├── train_baseline.py              # [重写] Exp1
│   ├── train_oracle.py                # [新建] Exp2
│   ├── train_oracle_v2.py             # [新建] Exp2 + ChunkedHMLP
│   ├── train_infer.py                 # [新建] Exp3
│   ├── train_infer_v2.py              # [新建] Exp3 + ChunkedHMLP
│   └── evaluate_all.py                # [新建] 对比评估
│
└── multiagent/                        # [保留] MPE核心
    ├── core.py
    ├── environment.py
    ├── multi_discrete.py
    ├── rendering.py
    └── scenario.py
```

---

## 八、实施阶段（v3.1 重构）

### Phase 1: 环境离散化
- [ ] 实现 DiscreteActionWrapper
- [ ] 修改 make_env.py 支持 discrete=True
- [ ] 验证环境可正常运行（随机离散策略测试）
- **验证标准**：4个agent各输出0-4，env.step正常返回

### Phase 2: 核心网络 + Baseline(Exp1)
- [ ] RepresentationNet (保留/微调)
- [ ] FunctionalMLP (支持外部权重的MLP骨架)
- [ ] BaselineDynamicNet (StateTransNet + RewardHead, 固定权重+id_emb)
- [ ] BaselinePredNet (PolicyHead + ValueHead, 固定权重+id_emb)
- [ ] EpisodeReplayBuffer (完整episode存储+序列采样)
- [x] Per-Agent Coordinate Descent MVE Planner (v4.5)
- [ ] MuZero展开训练循环 + 实时N-step return
- [ ] **端到端训练Baseline, 固定Rule验证收敛**
- **验证标准**：固定Rule=0.5，loss下降，eval reward上升

### Phase 3: Oracle-HyperMuZero(Exp2)
- [ ] AugmentedContextEncoder (Rule + Agent ID)
- [ ] DualHyperNetwork (hyper_trans + hyper_rew + hyper_pred)
- [ ] FunctionalDynamicNet (外部权重版 StateTransNet + RewardHead)
- [ ] FunctionalPredNet (外部权重版 PredictionNet)
- [ ] HyperMuZeroModel 总封装
- [ ] 训练 + 验证已知Rule时优于Baseline
- **验证标准**：多Rule训练，已知Rule评估优于Exp1

### Phase 4: Infer-HyperMuZero(Exp3, 核心贡献)
- [ ] GRUContextEncoder (历史轨迹推断rule_embedding)
- [ ] Context正则化Loss
- [ ] 修改Buffer支持历史窗口
- [ ] 训练 + 三组实验对比
- **验证标准**：Exp3 > Exp1，接近 Exp2

### Phase 5: 高级扩展(可选)
- [ ] MCTS Pro Mode
- [ ] hypnettorch ChunkedHMLP 替换简单超网络
- [ ] 可视化/分析

---

## 九、关键超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_agents` | 4 | 2捕猎+1可变+1猎物 |
| `num_actions` | 5 | 离散动作数 |
| `latent_dim` | 64 | 隐状态维度 |
| `hidden_dim` | 128 | 网络隐藏层 |
| `rule_emb_dim` | 16 | Rule编码维度 |
| `id_emb_dim` | 16 | Agent ID编码维度 |
| `context_dim` | 32 | rule_emb + id_emb |
| `unroll_K` | 5 | 展开步数 |
| `mve_samples` | 50 | MVE采样轨迹数 |
| `mve_depth` | 5 | MVE展开深度 |
| `n_step` | 5 | N-step return 的 n |
| `gamma` | 0.95 | 折扣因子 |
| `lr` | 1e-4 | 学习率 |
| `lr_min` | 1e-5 | CosineAnnealingLR 最小学习率 (v4.4) |
| `adam_eps` | 1e-5 | Adam 优化器 epsilon (v4.4) |
| `batch_size` | 256 | 批大小 (v4.4: 512→256 降低 replay ratio) |
| `buffer_size` | 5000 | Buffer episode数 (v4.4: 20000→5000 提升数据新鲜度) |
| `min_buffer_size` | 1000 | Warmup guard: 训练前最少 episode 数 (v4.4) |
| `episodes_per_iter` | 32 | 每轮收集 episode 数 (v4.4: 4→32) |
| `train_steps_per_iter` | 8 | 每轮训练步数 (v4.4: 16→8) |
| `grad_clip` | 10.0 | 梯度裁剪 |
| `episode_limit` | 25 | Episode步数 |
| `proj_dim` | 64 | Projector 投影维度 (v4.0) |
| `output_scale_init` | 0.01 | HyperNet 可学习模长初始值 (v4.0) |
| `w_policy` | 1.0 | Policy Loss 权重 |
| `w_value` | 0.25 | Value Loss 权重 |
| `w_reward` | 5.0 | Reward Loss 权重 (v4.7: 1.0→5.0 放大梯度信号) |
| `w_consist` | 0.5 | Consistency Loss 权重 (v4.1: 2.0→0.5 防止表征坍缩) |
| `w_context` | 0.01 | Context Hinge Loss 权重 (Exp3) |
| `target_context_std` | 0.1 | Hinge Variance 阈值 (Exp3) |
| `ema_tau` | 0.99 | Target Network EMA 系数 (v4.4) |
| `freeze_enabled` | False | 阶段性冻结总开关 (v4.7: True→False 全员训练) |
| `freeze_warmup_steps` | 4000 | 冻结预热期 (v4.4: 5000→4000) |
| `freeze_phase_steps` | 4000 | 每阶段步数 (v4.4: 10000→2400, v4.7: 2400→4000) |
| `epsilon_decay_steps` | 28000 | Epsilon 衰减步数 (v4.4: 与 lr 衰减对齐) |
| `hyper_rew_hidden_dims` | [256, 256, 256] | hyper_rew 独立隐藏层 (v4.7 新增) |
| `w_rew_diversity` | 0.1 | Reward Diversity Loss 权重 (v4.7 新增) |
| `rew_diversity_target_cos` | 0.3 | Diversity Hinge 阈值 (v4.7 新增) |
| `shield_alpha` | 0.6 | 阻截点插值系数 (v4.3) |
| `shield_r_max` | 0.5 | 阻截位奖励每步上限 (v4.3) |
| `shield_d_max` | 1.5 | 阻截位奖励衰减距离 (v4.3) |

---

## 十、梯度传播设计

### 10.1 统一Optimizer
一个 Adam (eps=1e-5, v4.4) 管理所有在线模型参数（RepNet + HyperNet + FunctionalNets 由HyperNet间接优化）。Target Model 不参与优化器，仅通过 EMA 更新（见 §5.7）。

### 10.2 Unroll 梯度流

```
Loss_total = Σ_{k=0}^{K} (L_policy^k + L_value^k + L_reward^k + L_consistency^k)

每步 k 的梯度路径:
  L_policy^k  → PredNet(θ_pred) → HyperNet_pred → ContextEncoder
  L_value^k   → PredNet(θ_pred) → HyperNet_pred → ContextEncoder
  L_reward^k  → RewardHead(θ_rew) → HyperNet_rew → ContextEncoder
  L_consist^k → Projector → s_pred → StateTransNet(θ_state) → HyperNet_trans
               (target 侧 Projector+RepNet 均 detach，不接受 consist 梯度)

所有 Loss 都回传到 RepNet（因为 s_0 = RepNet(obs)，且展开链未断）
梯度半衰 s = 0.5*s + 0.5*sg(s) 防止长展开爆炸
```

### 10.3 Consistency Loss 的梯度 (v4.0 更新)
- `L_consist = -CosineSim(Projector(s_pred), sg(Projector(RepNet(o_next))))`
- sg = stop_gradient (detach)
- 梯度路径: L_consist → Projector(参数) → s_pred → StateTransNet(θ_state) → HyperNet_trans
- target 侧 Projector + RepNet 均 detach，不接受 consistency 梯度
- 但 RepNet 通过初始编码 s_0 = RepNet(o_0) 仍然收到其他 Loss 的梯度
- Projector 的参数由 consistency loss 的 pred 侧训练（target 侧 detach）

---

## 十一、现有代码影响评估

| 文件 | 状态 | 说明 |
|------|------|------|
| `multiagent/*` | ✅ 保留 | 无需修改 |
| `envs/non_stationary_tag.py` | 🔧 微调 | 确保离散兼容 |
| `envs/ns_environment.py` | 🔧 修改 | 加 DiscreteActionWrapper |
| `envs/make_env.py` | 🔧 修改 | discrete=True |
| `models/representation_net.py` | ✅ 保留 | 可能微调维度 |
| `models/baseline_model.py` | ❌ 重写 | Actor+ValueNet → PredNet+DynNet |
| `models/interfaces.py` | ❌ 重写 | 新接口定义 |
| `training/buffer.py` | ❌ 重写 | transition → episode sequence |
| `training/mve.py` | ❌ 删除 | 被 planning/mve_planner.py 替代 |
| `scripts/train_baseline.py` | ❌ 重写 | MuZero训练循环 |
| `config.py` | ❌ 重写 | 新参数体系 |

---

## 十二、风险与应对

| 风险 | 应对选项 |
|------|---------|
| 超网络输出爆炸 | A:小权重init(std=0.01) / B:LayerNorm / C:hyperfan_init |
| AdaLN 信号衰减 (输入不敏感) | ✅ **v4.6 已解决**: `h*(1+gamma)+beta` 残差调制 (§3.5.1, §5.10) |
| MVE Planner 采样效率低 | ✅ **v4.6 已解决**: Coordinate Descent (v4.5) + CRN 方差消减 (v4.6) (§4.1, §5.8) |
| MuZero展开梯度爆炸 | A:半衰trick / B:减小K / C:梯度裁剪 |
| N-step return 方差大 | A:增大n / B:减小n / C:TD(λ) |
| Baseline 不收敛 | A:固定Rule调试 / B:简化为1-step / C:检查obs/reward尺度 |
| 策略循环震荡 | A:阶段性冻结(freeze_enabled) / B:Target Network EMA(v4.4) / C:降低Replay Ratio |
| V 值过估正反馈 | A:Target Network EMA(v4.4) / B:Clipped Double-Q / C:减小 n_step |
| GRU推断不准(Exp3) | A:增加历史窗口 / B:调正则系数 / C:Gumbel-Softmax |
| **hyper_rew 参数坍缩** (cos_sim→1) | ✅ **v4.7 已解决**: Diversity Loss + 加深 hyper_rew [256,256,256] + w_reward 5× 放大 (§5.11) |

---

*文档版本: v4.7 | 最后更新: 2026-02-22*
*v4.7 新增: Reward HyperNet 分化改进 — Reward Diversity Loss (Hinge cos_sim<0.3) 打破初始坍缩 + hyper_rew 加深至 [256,256,256] 增强分化容量 + w_reward 1.0→5.0 放大梯度信号 + 全员训练 freeze_enabled=False 避免 continual learning 灾难性遗忘 (§5.11)*
*v4.6 新增: Planner CRN 方差消减 — Common Random Numbers 消除其他 Agent 随机动作噪声 (SNR 0.02→∞) + 数据布局 (B,A,spa)→(B,spa,A) + 回退 PG 辅助 loss (off-policy bias 导致 l_pg 正反馈失控) + 恢复纯 CE policy loss + epsilon_init 1.0 恢复 + AdaLN 残差调制 h\*(1+gamma)+beta 修复 Functional Net 信号衰减 (§5.10)*
*v4.4 新增: 训练稳定性改进 — Target Network (EMA τ=0.99) 阻断 V 过估正反馈 + CosineAnnealingLR 平滑衰减 + Adam eps=1e-5 + freeze/epsilon 参数重新对齐 + buffer_size 20000→5000 提升数据新鲜度 + min_buffer_size=1000 warmup guard + 移除 PER 改用均匀采样 + Replay Ratio 82→2.56 (batch_size 512→256, episodes_per_iter 4→32, train_steps_per_iter 16→8)*
*v4.3 新增: 奖励正交化重构 — Blocking Point 阻截位引导 + r_hunt/r_guard 信号消除修复 + Prey/Agent2 奖励正交化*
*v4.2 新增: 阶段性冻结 (Phased Perspective Sampling) — 对抗训练节奏控制, 防止策略循环震荡*
*v4.1 新增: 统一评估框架(Evaluator + force_rule) + Consistency Loss 改为 (1-cos) + w_consist 2.0→0.5*
*v4.0 新增: HyperNet L2 归一化(output_scale=0.01) + Projection+CosineSim Consistency + Hinge Variance Context Loss + Loss/K 平均化*
*v3.2 新增: Functional Net 稳定性三层防线 (AdaLN + 残差连接 + 输出缩放)*
*整合来源: DESIGN_DOC.md(v1.0), DESIGN_DOC_v1.1_APPENDIX.md, design3.0.md, HYPER_MUZERO_IMPROVEMENTS.md*
*所有先前文档归档为参考，本文档为唯一实施依据*
