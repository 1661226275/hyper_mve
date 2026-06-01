# Pkg-03: TriContextEncoder & BeliefNet — Design

> 配套阅读：[`proposal.md`](./proposal.md)（**先读 proposal 再读本文**）

---

## 1. Context

### 1.1 项目阶段

本包是 v4 重构的**第三个 SDD 包**，与 Pkg-02 后半段可并行（Pkg-02 W2 后半 + Pkg-03 全程 W2）。完成本包后，Pkg-04 (DualHyperNetwork v2) 可立即启动——DualHyperNetwork v2 的 80 维 ctx 输入完全来自 `TriContextEncoder.forward(...)`，per-agent θ 生成的下游接口完全依赖 BeliefNet 提供 b_i^t / ĉ_i^t / ẑ_{i,j}^t 三个张量。

```
Pkg-01 (Schema)  ✅ 完成
    ↓
Pkg-02 (Env)     ⏳ 进行中（暴露 env.info Oracle 信号）
    ↓
Pkg-03 (本包)    ⏳ 当前（消费 env.info + 提供 BeliefNet/TriContextEncoder）
    ↓
Pkg-04 (Model)   ← 立即依赖 TriContextEncoder + BeliefNet
Pkg-05 (Trainer) ← 依赖 belief_loss + 课程接口
Pkg-06a/b        ← 共享 BeliefNet/TriContextEncoder 实例
```

### 1.2 前置依赖（本包消费什么）

| 来源 | 内容 |
|------|------|
| Pkg-01 `schemas/_constants.py` | `D_C_CTX=16`, `D_ROLE=32`, `D_BELIEF=32`, `D_BELIEF_PROJ=16`, `D_CTX_AUG=80` |
| Pkg-01 `schemas/agent_type.py` | `AgentType.ALPHA/BETA` Enum |
| Pkg-01 `schemas/capability.py` | `CapabilityVector` dataclass（4 维：η/φ_fov/ν/ζ） |
| Pkg-01 `configs/model_config.py` | `ModelConfig.belief_gru_hidden=128`, `belief_pool="mean"`, `d_id_emb=8`, `d_type_emb=8`, `d_cap_emb=16` |
| Pkg-01 `schemas/buffer_record.py` | `TimeStepRecord.c_hat: (N,)`、`z_hat: (N, N-1, 2)` 字段约定（含顺序契约） |
| Pkg-02 `envs/resource_commons/env.py` | `env.info["c_true"]: float`, `info["types"]: np.int8 (N,)`, `info["caps"]: tuple[CapabilityVector, ...]` |
| Ch4.2 / Ch4.5 / Ch4.6.5 | 三联通路 + BeliefNet 损失 + 梯度门控规格 |

### 1.3 本包提供（后续包消费什么）

| 输出 | 消费者 | 用途 |
|------|--------|------|
| `TriContextEncoder` 类 | Pkg-04 DualHyperNetwork v2 | `forward(...)` → ctx_i ∈ ℝ^80 喂 hyper_rew/hyper_pred；c_ctx 子分量喂 hyper_trans |
| `BeliefNet` 类 | Pkg-04 HyperMuZeroModel + Pkg-05 worker | `step()` 在线推断（worker）+ `forward(seq)` 离线训练（trainer） |
| `belief_loss(...)` 函数 | Pkg-05 trainer | 联合 L_c / L_opp / L_div 计算 |
| `PermutationInvariantPool` 工厂 | Pkg-04 / Pkg-06a baselines | belief 内 ẑ 池化（保持 set-invariant） |
| `oracle_z: Optional[Tensor]` 接口 | Pkg-05 课程 Stage 1/2 | Stage 1 用 oracle τ one-hot 替代 head_opp 输出 |

---

## 2. Goals

### 2.1 主目标（必须完成）

1. **G1：维度精确**——TriContextEncoder 输出 80 维（16+32+32），role 内 8+8+16=32 精确填满（无 pad），与 Pkg-01 ModelConfig 一致
2. **G2：head 输出形状契约**——head_c 输出 (B, N) sigmoid scalar；head_opp 输出 (B, N, N-1, 2) softmax；z_hat 顺序按 agent_id 升序跳过 self（与 Pkg-01 TimeStepRecord 一致）
3. **G3：v4 Oracle 监督正确接入**——L_opp 接收 `env.info["types"]` 作 CE 标签；L_c 接收 `env.info["c_true"]` 作 MSE 标签
4. **G4：梯度流完整**——反向传播覆盖三路 + 三 head 所有参数有梯度（梯度 norm > 0）
5. **G5：合成数据收敛**——5K 步内 head_c MSE < 0.05 + head_opp accuracy > 80%（在合成 c=0.7 + 2α2β 轨迹上）
6. **G6：可复用工厂接口**——Pkg-06a `shared_backbones.py` 可一行 `BeliefNet(env_cfg, model_cfg)` 复用

### 2.2 衍生目标（应尽量达到）

- **G7**：`mypy --strict hyper_mve/models/{tri_context_encoder,c_encoder,role_encoder,belief_net,belief_losses,permutation_invariant_pool}.py` 零错误
- **G8**：单测覆盖率 ≥ 85%
- **G9**：BeliefNet collapse 长训练（10K 步）实测 b_i^t 方差 ≥ 0.1（L_div 起作用）
- **G10**：TriContextEncoder 单步 forward < 5 ms (V100, B=256, N=4)

### 2.3 Non-Goals（明确不解决）

- **NG1**：**不实现** Pkg-04 DualHyperNetwork v2（仅提供 ctx_i 输出供其消费）
- **NG2**：**不实现** Pkg-05 课程学习 stage 切换逻辑（仅暴露 oracle_z 接口）
- **NG3**：**不实现** belief gradient gating 的 step-conditional 切换（具体在 Pkg-04 model 内）
- **NG4**：**不实现** type-stratified sampling（属于 Pkg-05 buffer）
- **NG5**：**不实现** μP base_shape 集成（属于 Pkg-07）
- **NG6**：**不实现**端到端训练脚本（仅合成数据收敛 scripts/test_belief_net_synth.py）
- **NG7**：**不引入**新依赖（仅 PyTorch + numpy）
- **NG8**：**不修改**论文 Ch4 文档（如发现 spec 与 Ch4 不一致，先报 issue 讨论）

---

## 3. Decisions

> 8 项关键设计抉择。每项格式：**Decision** → 候选 → 推荐 → 理由 → 风险与回滚。

### D1: head_opp 输出格式

**Decision**：BeliefNet head_opp 输出 N-1 个对手类型概率，如何组织张量？

**候选**：
- **A1**：单输出 `(B, N, N-1, 2)` reshape（按 agent_id 升序跳过 self）
- **A2**：多 head per opponent（N-1 个独立 head，参数量 (N-1)× 大）
- **A3**：图神经网络 attention（每个对手用 attention 计算）

**推荐**：**A1**（单输出 reshape，agent_id 升序跳过 self）

**理由**：
- Ch4.2.3 指明 Pool 之前每个 ẑ_{i,j} 是 j 索引的张量；reshape 让 batched 矩阵运算高效
- A2 多 head 在 N 变化时（Easy N=2 vs Hard N=8）参数量不一致；A1 单 head 加 id_emb(j) 条件化可适配任意 N
- A3 attention 增加 ~10K 参数；当前 N≤8 收益不显著
- **顺序约定**：Pkg-01 TimeStepRecord z_hat docstring 已硬约束 "agent_id 升序跳过 self"（agent i 的 z_hat[i, k] 对应 agent_id = (k if k < i else k + 1)）
- A1 的 head_opp MLP 输入是 `(b_i, opp_id_emb(j))`，输出 2 维 softmax；遍历 j 排除 i 后 reshape 为 (N-1, 2)
- **id_emb 独立性（与 role.id_emb 解耦）**：head_opp 的 opp_id_emb 是 `BeliefIdEmbedding(N, d=16)`，与 RoleEncoder.id_emb (d=8) 是**两个独立 nn.Embedding 实例**。语义分离：role.id_emb 表示 "agent 自己身份"（own identity），opp_id_emb 表示 "对手身份"（opponent identity）。本就应解耦：(1) 训练时 head_opp 梯度不应通过 id_emb 污染 role 通路；(2) 维度可独立调整。代价是 N 个 id 嵌入存两份（共 N×24 参数，N=8 时 192 参数，可忽略）

**风险**：
- 顺序错位 → L_opp 在错位标签上 CE → BeliefNet 学到错乱对手 type → 主观通路 hyper_rew 生成错乱 θ_rew → 训练崩溃
- **强制单测** `test_z_hat_agent_id_order` 验证顺序约定

**回滚**：A2 多 head（参数翻倍，需重写 head_opp 模块）

---

### D2: belief 池化策略

**Decision**：将 N-1 个 ẑ_{i,j} 聚合为单个 belief 子分量，用什么 set-invariant pool？

**候选**：
- **B1**：mean pooling（默认）
- **B2**：max pooling
- **B3**：attention pooling（learnable query）
- **B4**：concat（不 set-invariant，但保留所有信息）

**推荐**：**B1**（mean），配置可切换至 B2/B3

**理由**：
- Ch4.2.3 推荐 mean 或 max 保持排列不变；mean 是 deep set 文献标准默认
- B4 concat 在 N 变化时维度不一致（Easy N-1=1 vs Hard N-1=7），破坏 ModelConfig 静态维度
- B3 attention 增加 ~5K 参数；当前合成数据收敛实验未观测到 mean 收敛瓶颈
- Pkg-01 ModelConfig 已预留 `belief_pool: str = "mean"` 字段；切换不改架构代码
- mean 池化数学：`pool({z_j}) = (1/(N-1)) * sum_j z_j ∈ Δ^2`（仍是概率分布）；max 池化按元素逐维取最大；attention 用 b_i 作 query 对 {id_emb(j) ⊕ z_j} 做 weighted sum

**风险**：
- mean 在 α/β 50/50 mix 时输出近似 (0.5, 0.5)，信息丢失 → 但 belief 通路仅是辅层 2，主层 type_emb（自己 type 真值）仍提供强信号
- 若收敛实验观察到 mean 信息不足，切 max 或 attention（仅配置）

**回滚**：切 max / attention（仅改 cfg.belief_pool）

---

### D3: ẑ 进入 belief 通路的形式

**Decision**：head_opp 输出 softmax 概率，进入 TriContextEncoder 的 belief 子通路前如何处理？

**候选**：
- **C1**：直接 softmax 概率 (B, N, N-1, 2)
- **C2**：logits（softmax 前）
- **C3**：hard one-hot（argmax）

**推荐**：**C1**（softmax 概率）

**理由**：
- 软标签保留不确定性信息（Ch4.5 论证 + Bayesian 视角）
- C3 argmax 在课程学习 Stage 2 anneal 时不可微（混合 oracle one-hot + ẑ 时 ẑ 是不可微输出）
- C2 logits 量级不稳定（pre-softmax），LayerNorm 后再喂 hypernet 增加复杂度
- C1 softmax 自带概率范围 [0, 1]，与 Pool 后续操作（mean pool 仍是概率分布）兼容
- C1 与 Stage 1 oracle 形式一致（oracle 是 one-hot ∈ Δ^2，C1 是 soft ∈ Δ^2，渐变退火可微）

**风险**：
- softmax 输出可能过自信（接近 (1, 0) 或 (0, 1)），mean pool 后仍偏自信 → 用 temperature softmax 缓解（默认 T=1，可配置）
- 当前默认 T=1 不暴露；如观察到过自信，加 `head_opp_temperature: float = 1.0` 到 ModelConfig

**回滚**：C2 logits（需 belief 通路加 LayerNorm）/ C3 one-hot（破坏 Stage 2 可微性）

---

### D3 附加：为什么 raw heads 进 TriContextEncoder 而非 BeliefNet 内部投影（P4 motivation）

`TriContextEncoder.forward(...)` 接收 `belief: (c_hat, z_hat)` raw heads（spec 01），而**不**让 BeliefNet 直接输出 `belief_proj: (B, N, 32)`。这是为 **Ch4.6.5 belief 梯度门控**预留的设计：

**梯度门控分层**：
- BeliefNet 内部 (GRU + heads) 产生 raw heads (c_hat, z_hat)
- 投影 MLP（`BeliefEncoder`：proj_c_hat / Pool / proj_z_pooled，**P5 拆出**至独立模块）由 TriContextEncoder 协调调用
- Pkg-04 model 在前 5K step 做 `.detach()` 切断梯度：仅切 raw heads（c_hat / z_hat tensor），不动投影 MLP

**后果**：
- 前 5K step：main task loss 经投影 MLP 反向 → **投影 MLP 仍由 main loss 训练**；但梯度在 raw heads 处停止 → BeliefNet GRU 不被 main task 更新（仅由独立 L_belief 更新）
- 5K step 后：移除 .detach()，main task 梯度也回到 BeliefNet GRU

**反方案的问题**：
- 若 BeliefNet 直接输出 belief_proj (B, N, 32)，投影 MLP 也在 BeliefNet 内 → 前 5K step .detach() 会一刀切断**投影 MLP 训练** → main task 无法通过 belief 路径学到表示
- 等于"前 5K step belief 通路完全失效"，违背门控初衷（门控目的是让 main task 与 belief 异步训练，而非完全屏蔽 belief）

**结论**：raw heads 进 Tri 是有意为之。投影 MLP 在 BeliefEncoder（独立模块），切断点在 BeliefNet 输出处。

---

### D4: head_c 监督信号来源

**Decision**：L_c MSE 的 c_true 标签来自哪里？

**候选**：
- **D1**：仅 oracle `env.info["c_true"]`（Pkg-02 暴露）
- **D2**：仅 reconstruction loss（从 obs 重建 c_t）
- **D3**：两者结合（Oracle 主、reconstruction 辅）

**推荐**：**D1**（仅 Oracle）

**理由**：
- Ch3.4 c_t 物理可观（默认观测中含 c_t）；Pkg-02 已暴露 `env.info["c_true"]: float`
- D2 reconstruction 引入 decoder，增加参数量 + 训练不稳定
- D3 双信号增加权重调参复杂度（reconstruction loss 权重难定）
- v4 Ch4.5.1 公式明确：$\mathcal{L}_c = \frac{1}{NT}\sum_{i,t}(\hat{c}_i^t - c_t)^2$，c_t 即 Oracle
- 模式 B（c_t 隐藏）评估时不影响训练：训练始终用 Oracle，推断时 BeliefNet 从资源场推断 ĉ（Self-Info）
- Pkg-02 spec 08 已明确 `info["c_true"]` 是"仅 trainer 监督用，不可传 model"（严格分组）

**风险**：
- 若 Pkg-02 暴露的 `c_true` 字段名变更 → 单测 `test_belief_losses.py::test_l_c_uses_oracle_field` 验证字段名
- Pkg-05 trainer 误把 `c_true` 传入 BeliefNet forward（破坏 Self-Info 评估）→ spec 08 集成契约明确禁止

**回滚**：D3 加入 reconstruction（需新增 decoder + 重训）

---

### D5: L_div 实现

**Decision**：防 BeliefNet 输出在 N 个 agent 间坍缩为常数，用什么正则？

**候选**：
- **E1**：Variance hinge `max(0, σ_target² - Var_i(b_i^t))`
- **E2**：KL to uniform prior
- **E3**：Barlow Twins 风格 cross-correlation 损失

**推荐**：**E1**（Variance hinge with σ_target=0.1）

**理由**：
- Ch4.5.3 已指定 hinge variance 形式；σ_target=0.1 是合理默认（在 8-32 维 belief 上有信号区分度）
- E2 KL 要求 prior 设定（uniform？某种 mixture？），增加先验歧义
- E3 Barlow Twins 是表示学习常用，但要求 batch 内 N 较大才有信号；当前 N=4 不足
- hinge 形式：`max(0, σ_target² - Var_i)` ——variance 高于阈值时损失 = 0（不约束），低于阈值时线性惩罚；不会过度抑制 belief 多样性
- 与 Ch4.5.4 总权重 λ_div=0.01 配套，影响小但 collapse 触发时及时干预

**风险**：
- σ_target=0.1 可能过高（导致 L_div 长期非零）或过低（无 collapse 保护）
- 单测 + 合成数据 10K 步实测确定阈值；如有问题加配置项 `belief_div_target_std` 已在 Pkg-01 TrainConfig 中存在（值 0.1）

**回滚**：E3 Barlow Twins（需 batch 内多 view，引入 augment 复杂度）

---

### D5 偏差备案：与 Ch4.5.3 论文公式不一致（P1）

**Ch4.5.3 论文公式**：

$$\mathcal{L}_{\text{div}} = -\frac{1}{B}\sum_{\text{batch}} \mathrm{Var}_i(b_i^t)$$

—— 直接负方差，**无阈值**，永远激励增大方差。

**本 spec 实现（hinge）**：

$$\mathcal{L}_{\text{div}} = \max\left(0,\;\sigma_{\text{target}}^2 - \mathrm{Var}_i(b_i^t)\right),\quad \sigma_{\text{target}}=0.1$$

—— variance 高于 0.01 时 loss = 0（不约束），低于时线性惩罚。

**偏差理由**：
1. **工程稳定性**：纯 -Var 在 train 后期持续推方差爆炸（与 GRU LayerNorm 限定行为冲突，可能引发 LN gamma 漂移）
2. **语义差异**：hinge 是"防 collapse"，论文 -Var 是"最大化多样性"——前者是必要条件（不坍缩），后者是奢侈条件（无上限多样化）；工程上前者足够
3. **Pkg-01 字段已锁定**：`TrainConfig.belief_div_target_std=0.1` 已暗示 hinge 形式

**建议论文同步修订**（已在 Pkg-03 spec 06 §0 记录）：将 Ch4.5.3 公式改为 hinge 形式，或在论文加 footnote 说明实现层用 hinge variant。

---

### D6 附加：obs_encoder 架构 + BeliefNet LayerNorm（P6）

**BeliefObsEncoder 架构**（spec 04 §2.2 已实现）：
```
Linear(obs_dim, 128) → ReLU → Linear(128, 64) → LayerNorm(64)
```
- 选 `→ 64`（而非 `→ 128`）：GRUCell 矩阵乘 3×input×hidden，input=64 时参数 24576，input=128 时 49152（翻倍）；64 足够编码 ResourceCommons obs
- 末端 LayerNorm 稳定 GRU 输入分布

**BeliefNet GRU 输出 LayerNorm**（Ch4.6.2 防线 2，spec 04 §2.3 已实现）：
- `self.ln_belief = nn.LayerNorm(self.gru_hidden)` 在 `_gru_step` 末尾应用
- 输出的 hidden 是 LN 后的值；下次 step 时 prev_hidden 是 LN 后的 → feed 回 GRUCell 也是 LN 后的（设计：防止 GRU 内部分布漂移）

---

### D6: GRU 方向

**Decision**：BeliefNet GRU 是单向还是双向？

**候选**：
- **F1**：uni-directional（causal）
- **F2**：bi-directional（biLSTM/biGRU）
- **F3**：Transformer with causal mask

**推荐**：**F1**（uni-directional）

**理由**：
- RL 在线推断必须只用过去信息（worker 在 t 时刻不知道 t+1 的 obs）
- F2 biGRU 训练时用全序列，但在 worker step API 中不可用 → 训练/推断不一致
- F3 Transformer causal 等价于 uni-directional 但 O(T²) 复杂度；T=300 (Hard) 单 episode 90000 attention ops，超出 GRU O(T·h²) ~38400 ops
- uni-directional GRU step API 简单：`hidden = gru(obs_emb, prev_hidden)`，与 PyTorch nn.GRU 默认对齐
- 与 v4.7 `gru_context_encoder.py` 经验一致（已验证稳定）

**风险**：
- 长 episode 早期信息不足（如 T=300 前 50 步），head_c 早期 MSE 高 → 可接受（agent 也只能用过去信息）
- 监控：head_c MSE 按 step_idx 分组的曲线（早期高、中后期下降）

**回滚**：F3 Transformer causal（需重写 model）/ F2 biGRU（破坏 worker step API）

---

### D7: TriContextEncoder 归一化策略

**Decision**：三路输出量级悬殊（c_ctx 单维度 [-1, 1]、role 多维 embeddings、belief 概率值），如何归一化？

**候选**：
- **G1**：每路输出 LayerNorm 再 concat
- **G2**：仅 concat 后整体 LayerNorm
- **G3**：不归一（让 hypernet 入口处理）

**推荐**：**G1**（每路 LayerNorm 再 concat）

**理由**：
- c_ctx 是 [0, 1] 经 MLP 后输出 [-某, 某]，量级与 role embedding 不同
- role concat 后含 id (small embedding) + type (one-hot like) + cap (4D real)，内部量级也不齐
- belief 是 sigmoid scalar [0,1] 与 softmax 概率 [0,1] 与池化后的 mean，量级相近但与 c_ctx/role 差异大
- G2 整体 LayerNorm 会让某一路（量级最大的）主导归一化统计量，其他路被压低
- G1 每路独立 LayerNorm 保证三路在 concat 时数值范围一致（mean=0, var=1）
- 与 v4.6 context_encoder 已验证 stability 一致

**风险**：
- LayerNorm 参数（每路 16/32/32 维 affine）增加 ~80 参数，可忽略
- 如 hypernet 入口仍量级失衡，Pkg-04 在 hyper_rew/hyper_pred 入口加 input LayerNorm（设计上预留）

**回滚**：G2 整体 LayerNorm（只需删除每路 LN，加 concat 后 LN）

---

### D8: cap_emb 实现（含 P7 P/C 修订）

**Decision**：CapabilityVector (4 维：η, φ_fov, ν, ζ) 编码为 d_cap_emb=16 维，用什么模块？

**候选**：
- **H1**：2 层 MLP (4 → 16 → 16) + ReLU + LayerNorm
- **H2**：单线性 (4 → 16)
- **H3**：拼接原值 + LayerNorm（4 → padded to 16）

**推荐**：**H1 修订版**（2 层 MLP，**输入归一化 + 去内部 LN**）

**最终方案**：
```
input (4) ← CapabilityVector.normalize() 归一到 [0, 1]^4 (Pkg-01 utility)
  → Linear(4, 16) → ReLU → Linear(16, 16)
  (无内部 LayerNorm; LN 责任在 TriContextEncoder.ln_role 集中, P7)
```

**理由**：
- Ch4.2.2 指明 "Self Info prior"，cap 4 维异质值需要非线性变换提取有用特征
- η ∈ [0.5, 1.5]、φ_fov ∈ {2,3,4}、ν ∈ [0.8, 1.0]、ζ ∈ [10, 30] 量级差异大（ζ vs η ~30 倍），单线性难以充分编码
- **修订 C（输入归一化）**：原方案"靠 MLP 自学量级"在早期 warmup 期会让 ζ 列梯度主导（Adam 仅部分自适应），淹没 η/ν 信号；Pkg-01 加 `CapabilityVector.normalize()` 是最小侵入修复（详见 D8 附加：cap 输入归一化）
- **修订 P7（去内部 LN）**：LN 责任集中在 TriContextEncoder.ln_role；cap_mlp 内 LN 与 ln_role 是 double LN（cap 子段被 LN 两次）
- 2 层 MLP 是 deep RL 标准 encoder pattern（足够表达力、不过参数化）

**风险**：
- 2 层 MLP 参数 ~336（4×16 + 16×16 + bias），相对总参数量可忽略
- normalize 接口稳定性依赖 Pkg-01 spec 02 `CapabilityVector.normalize()` 持久存在
- 输入归一化后 MLP 第一层不再处理量级问题，单测 `test_cap_mlp_uniform_input_scale` 验证输入范围

**回滚**：H2 单线性（损失非线性能力）/ H3 拼接（需手工归一化各维，本质同当前方案但更繁琐）

---

### D8 附加：cap 输入归一化（P7 + C 修订，2026-05-28）

**问题（用户审阅 P7+C）**：原 design.md 方案是"cap_mlp 内含 LayerNorm 处理 cap 4 维量级悬殊"，但有两个问题：
1. 与 TriContextEncoder.ln_role（D7 每路 LN）是 double LN
2. cap_mlp 第一层 `Linear(4, 16)` 仅 64 个权重，ζ 列梯度量级是 η 列 ~30 倍；Adam 自适应缩放能部分缓解，但早期 warmup 期 η/ν 信号被淹没（cap_emb 直接喂 hyper_rew/hyper_pred 生成 θ → 影响 v4 主目标）

**最终方案**（用户决定）：在 **Pkg-01 加 `CapabilityVector.normalize()` read-only utility**：
```python
# Pkg-01 schemas/capability.py
class CapabilityVector:
    def normalize(self) -> np.ndarray:
        """各维归一到 [0, 1]^4. raw cap 不变 (env/buffer/info 保持物理可解释)."""
```

**集成方式**（Pkg-03 spec 03 forward 内）：
```python
def forward(self, agent_ids, types, caps_normed):
    # caps_normed: (B, N, 4) 已归一 (Pkg-05 trainer 调用 normalize 后传入)
    cap_emb = self.cap_mlp(caps_normed)  # 第一层不再处理量级问题
```

或更安全的方式（cap_mlp 内自动归一）：
```python
def forward(self, agent_ids, types, caps_raw):
    # caps_raw: (B, N, 4) raw values
    caps_normed = (caps_raw - CAP_NORM_LO_T) / (CAP_NORM_HI_T - CAP_NORM_LO_T)
    cap_emb = self.cap_mlp(caps_normed)
```

**选择**：cap_mlp 内部自动归一（避免 trainer 端忘记调用）。详见 spec 03。

**关键不变量**：
- env / buffer / `info["caps"]` / `TimeStepRecord.cap` 仍存 raw CapabilityVector
- 仅 cap_emb MLP 消费时归一化
- 物理可解释性保留（render / log / debug 看到 η=1.2 而非 0.7）

---

## 4. 设计决策对照表

| Decision | 推荐 | 影响范围 | 后续修改成本 |
|----------|------|----------|--------------|
| D1 head_opp 输出格式 | (B, N, N-1, 2) reshape; opp_id_emb 独立于 role.id_emb | belief_net.py | 中（顺序契约严格） |
| D2 belief 池化 | mean（可切换） | permutation_invariant_pool.py | 低（cfg 切换） |
| D3 ẑ 进 belief 通路 | softmax 概率; raw heads 进 Tri 而非 BeliefNet 内投影 (P4) | belief_encoder.py + tri_context_encoder.py | 低（接口签名） |
| D4 head_c 监督来源 | Oracle c_true | belief_losses.py | 低（字段名稳定） |
| D5 L_div 实现 | **Variance hinge**（偏差 Ch4.5.3，已备案 P1） | belief_losses.py | 低（hinge 公式） |
| D6 GRU 方向 | uni-directional + ln_belief after GRU + obs_encoder [obs_dim→128→64] | belief_net.py | 高（双向需重写 worker step） |
| D7 三路归一化 | 每路 LayerNorm；sub-encoder 内**不**含 LN (P7 责任集中) | tri_context_encoder.py | 低（LN 增删） |
| D8 cap_emb 实现 | 2 层 MLP **+ Pkg-01 normalize() 输入归一**（C 修订）；无内部 LN | role_encoder.py + Pkg-01 spec 02 | 低（MLP 替换） |

---

## 5. 实现顺序建议

```
Day 1（半天）:
  - c_encoder.py (MLP [1→32→32→16] + LN)
  - tests/models/test_c_encoder.py
  - 写 specs/02-c-encoder.md

Day 1（半天）:
  - role_encoder.py (id + type + cap_mlp concat → 32)
  - tests/models/test_role_encoder.py
  - 写 specs/03-role-encoder.md (v4 关键: type_emb)

Day 2:
  - permutation_invariant_pool.py (mean / max / attention)
  - tests/models/test_permutation_pool.py
  - 写 specs/07-permutation-invariant-pool.md
  - belief_net.py GRU 主干 (obs_encoder + GRU + init_hidden / step API)
  - tests/models/test_belief_net.py::test_gru_step_seq_equiv
  - 写 specs/04-belief-net-gru.md

Day 3:
  - belief_net.py 两 head (head_c + head_opp v4 关键)
  - tests/models/test_belief_net.py::test_head_*_output_shape + test_z_hat_agent_id_order
  - 写 specs/05-belief-heads.md
  - **Hard gate**: head_opp 顺序契约单测通过才能进下一步

Day 4:
  - belief_losses.py (L_c + L_opp + L_div + 组合器 belief_loss)
  - tests/models/test_belief_losses.py
  - 写 specs/06-belief-losses.md

Day 5:
  - tri_context_encoder.py (调三 encoder + LN 每路 + concat → 80 维 ctx_i)
  - tests/models/test_tri_context.py (维度 + 梯度流 + permutation invariance)
  - 写 specs/01-tri-context-encoder.md

Day 6:
  - scripts/test_belief_net_synth.py (合成数据 10K 步 + TensorBoard)
  - **Hard gate**: 5K 步内 head_c MSE < 0.05 + head_opp acc > 80% 才能 merge

Day 7:
  - 写 specs/08-integration-contracts.md (与 Pkg-04/05 接口契约)
  - PR description + e2e validation recipe
```

---

## 6. 跨包接口约定（API contract）

### 6.1 import 路径（稳定）

```python
# Pkg-04 / Pkg-05 / Pkg-06a/b 应使用这些 import 路径
from hyper_mve.models import (
    TriContextEncoder,
    BeliefNet,
    belief_loss,
)
from hyper_mve.models.belief_losses import l_c, l_opp, l_div  # 单独 loss（可选）
from hyper_mve.models.permutation_invariant_pool import make_pool  # 工厂

# 标准用法
from hyper_mve.configs import V4Config
cfg = V4Config.from_preset("medium")
belief_net = BeliefNet(cfg.env, cfg.model)
encoder = TriContextEncoder(cfg.env, cfg.model)
```

### 6.2 TriContextEncoder.forward 签名（稳定）

```python
def forward(
    self,
    c_t: torch.Tensor,                # (B,) or (B, 1) shared context scalar
    agent_ids: torch.Tensor,          # (B, N) int64
    types: torch.Tensor,              # (B, N) int64 (AgentType.value)
    caps: torch.Tensor,               # (B, N, 4) float32 CapabilityVector flatten
    belief: tuple[torch.Tensor, torch.Tensor],
                                      # (c_hat: (B, N) sigmoid, z_hat: (B, N, N-1, 2) softmax)
) -> torch.Tensor:                    # ctx_i: (B, N, 80) float32
    """
    返回 ctx_i ∈ ℝ^80 = Concat[c_ctx (16), role (32), belief (32)].
    
    其中:
        c_ctx 共享于同一 B (所有 agent 看到的 c_t 相同)
        role 含 own type (Self-Info, Ch4.2.2)
        belief 含 c_hat 与 Pool(z_hat) 的投影 (各 16 维)
    """
```

**c_ctx 单独取出（hyper_trans 用）**：

```python
def forward_c_ctx_only(
    self,
    c_t: torch.Tensor,                # (B,) or (B, 1)
) -> torch.Tensor:                    # (B, 16)
    """仅 c_ctx 子分量（hyper_trans 客观通路用，不依赖 N）."""
```

### 6.3 BeliefNet 两套 API（稳定）

```python
class BeliefNet(nn.Module):
    def init_hidden(
        self,
        batch_size: int,
        num_agents: int,
    ) -> torch.Tensor:                # (B, N, 128) zeros
        """初始化 GRU hidden（每个 (batch_idx, agent_idx) 独立 state）."""
    
    def step(
        self,
        obs_t: torch.Tensor,          # (B, N, obs_dim) float32
        prev_hidden: torch.Tensor,    # (B, N, 128) float32
        oracle_z: Optional[torch.Tensor] = None,
                                      # (B, N, N-1, 2) softmax-like, Stage 1/2 用
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """单步推断 (worker 在线调用).
        
        Returns:
            new_hidden: (B, N, 128)
            c_hat:      (B, N) sigmoid
            z_hat:      (B, N, N-1, 2) softmax
                        (若 oracle_z 给出, 直接返回 oracle_z)
        """
    
    def forward(
        self,
        obs_seq: torch.Tensor,        # (B, T, N, obs_dim) float32
        init_hidden: Optional[torch.Tensor] = None,
                                      # (B, N, 128) float32, 默认 zeros
        oracle_z_seq: Optional[torch.Tensor] = None,
                                      # (B, T, N, N-1, 2), Stage 1/2 用
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """序列推断 (trainer 在 K-step unroll 时调用).
        
        Returns:
            hidden_seq: (B, T, N, 128)
            c_hat_seq:  (B, T, N) sigmoid
            z_hat_seq:  (B, T, N, N-1, 2) softmax
        """
```

### 6.4 belief_loss 接口（稳定）

```python
def belief_loss(
    c_hat_seq: torch.Tensor,           # (B, T, N) sigmoid 输出
    z_hat_seq: torch.Tensor,           # (B, T, N, N-1, 2) softmax 输出
    hidden_seq: torch.Tensor,          # (B, T, N, 128) GRU hidden
    c_true_seq: torch.Tensor,          # (B, T) Oracle from env.info["c_true"]
    types_true: torch.Tensor,          # (B, T, N) int64 Oracle from env.info["types"]
                                       #   注: 每 step 都传, 但 episode 内 types 不变
    weights: tuple[float, float, float] = (1.0, 0.5, 0.01),
                                       # (λ_c, λ_opp, λ_div)
    div_target_std: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """计算 BeliefNet 联合损失.
    
    Returns:
        total: scalar tensor (含 grad)
        breakdown: dict 含 "l_c", "l_opp", "l_div", "total" (不含 grad, detached)
    """
```

### 6.5 课程 Stage 接口约定（与 Pkg-05 契约）

```python
# Pkg-05 trainer 的调用约定（在 Pkg-05 spec 04 中详细规定）
#
# Stage 1 (0% - 30% steps): Pure Oracle
#     oracle_z_seq = build_oracle_z(types_true)  # (B, T, N, N-1, 2) one-hot
#     _, c_hat, z_hat = belief_net.forward(obs_seq, oracle_z_seq=oracle_z_seq)
#     # head_opp 输出仍计算但不进入主任务通路 (用于 L_opp 训练 head_opp)
#
# Stage 2 (30% - 70%): Anneal
#     lambda_t = compute_anneal_lambda(step, stage_1_end=0.3, stage_2_end=0.7)
#     z_hat_predicted = belief_net.forward(obs_seq)[2]
#     z_for_main_task = (1 - lambda_t) * oracle_z + lambda_t * z_hat_predicted
#
# Stage 3 (70% - 100%): Pure Inference
#     _, c_hat, z_hat = belief_net.forward(obs_seq)
#     z_for_main_task = z_hat
#
# Belief Gradient Gating (前 5K step):
#     在 Pkg-04 model.forward 内, 对 belief tensor 调用 .detach() 切断梯度
#     本包仅暴露 BeliefNet 模块, 不参与切换逻辑
```

---

## 7. 验证策略概览

> 详细 acceptance criteria 见 `specs/*.md`。本节列 10 个关键测试。

1. `test_tri_context.py::test_output_dim`: `(B, N, 80)` 维度精确
2. `test_tri_context.py::test_role_dim_exact`: 8+8+16=32 精确无 pad
3. `test_tri_context.py::test_gradient_flow`: 三路 + 三 head 所有参数有梯度
4. `test_tri_context.py::test_permutation_invariance_belief`: mean pool 下改变对手顺序，belief 子分量不变
5. `test_belief_net.py::test_head_c_output_shape`: `(B, N)` sigmoid ∈ [0, 1]
6. `test_belief_net.py::test_head_opp_output_shape`: `(B, N, N-1, 2)` softmax
7. `test_belief_net.py::test_z_hat_agent_id_order`: head_opp[i, k] 对应 agent_id = (k if k < i else k + 1)
8. `test_belief_net.py::test_gru_step_seq_equiv`: step ≈ forward 差 ≤ 1e-5
9. `test_belief_losses.py::test_l_opp_oracle_ce`: L_opp 是 CE，标签 int 类型
10. `scripts/test_belief_net_synth.py`: 5K 步内 head_c MSE < 0.05 + head_opp acc > 80%

---

## 8. Open Questions（含 2026-05-28 用户审阅决议）

| # | Question | 决议 | 影响 |
|---|----------|------|------|
| Q1 | TriContextEncoder 是否在内部 concat 前对每路应用 LayerNorm？ | ✅ **是**（D7）；P7 进一步集中：sub-encoder 内**不**含 LN | 三路量级均衡 |
| Q2 | BeliefNet 的 RepresentationNet 是否共享 Pkg-04 主 RepNet 实例？ | ✅ **否**（独立 `_belief_obs_encoder.py`） | 避免循环依赖 |
| Q3 | head_opp MLP 是否共享 GRU 输出 trunk 还是各自 trunk？ | ✅ **共享** trunk | 参数节省 |
| Q4 | L_div hinge 的 σ_target 是否随训练步数 anneal？ | ✅ **否**（固定 σ_target=0.1） | 课程简化 |
| Q5 | PermutationInvariantPool 默认策略是否硬编码 mean 还是配置可切换？ | ✅ **配置可切换**（cfg.belief_pool） | 灵活性 |
| Q6 | BeliefNet.step / forward 是否暴露 `return_attention_weights` 用于可视化？ | ✅ **否**（Pkg-07 评估时如需可加） | 仅可视化 |
| Q7 | belief_loss 返回 breakdown 是 detach 后的 float 还是含 grad 的 tensor？ | ✅ **detach 后**（仅用于 logging） | TensorBoard 日志 |
| **A** | spec 数量是 8 个还是 9 个（BeliefEncoder 内嵌 vs 独立 spec）？ | ✅ **8 个**（BeliefEncoder 嵌入 spec 01） | 文档组织 |
| **B** | obs_encoder 内部结构 `[obs_dim→128→64]` 还是 `[→128→128]`？ | ✅ **→64**（参数节省 50%） | GRUCell 输入 |
| **C** | P7 cap 量级处理：靠 MLP 自学 vs Pkg-01 加 `CapabilityVector.normalize()`？ | ✅ **改 Pkg-01 加 normalize()**（read-only utility，env/buffer 不变） | 训练稳定性 |
| **D** | P11 是否预留 `ModelConfig.max_N=8` 字段？ | ✅ **不预留**（标注 future work in spec 05 §6） | 跨 preset transfer |

> 全部 Q1-Q7 + A-D 已 ack。本 design.md 在 P1-P11 + C 修订后视为 **finalized**。

---

## 9. References

- `proposal.md`（本包）
- Pkg-01 design.md（schema/config 跨包契约）
- Pkg-02 design.md + spec 08（env.info Oracle 信号契约）
- `D:\RL\hyper_mve\docs\Chapter4_Architecture_v4.md` §4.2 + §4.5 + §4.6.5
- `D:\RL\hyper_mve\docs\Chapter4_1_Motivation_v4.md` §4.1.3 / §4.1.4
- `D:\RL\hyper_mve\docs\Chapter5_Planner_Training_v4.md` §5.7
- 项目 Plan File: Part 2 Pkg-03 + Part 9 v4 修订
