# Hyper-MuZero v4 理论根基与逻辑完备性复审(优化阶段视角)

> **文档定位**:本文档是在 Pkg-01~05 实现完成、并经历优化阶段(提交 `e5a9e17..dc5bbcd`,2026-06)之后,对整个研究的理论根基、四个可证伪断言的适定性、以及文档↔代码一致性的一次全局复审。它是后续论文章节修订(Ch1.5 / Ch4 / Ch5 / Ch6)、Roadmap 更新与 SDD 修订的**措辞母本**——各文档引用本文档已批准的表述,不另造措辞。
>
> **复审触发原因**:优化阶段的两个实证发现动摇了原设计的两个隐含假设——(1) FULL 全量权重生成在 duo 运行中坍缩(`cos_pred_cross` 0.61→0.998),证伪了"生成式条件化的优化是免费的"这一 Ch4.1.2 隐含前提;(2) 采集不带规划器时策略熵钉死在 ln(A),证明规划信号是训练可行性的**必要条件**而非工程选项。
>
> **作者决议(2026-06-10,约束本文及全部后续修订)**:
> 1. 断言 B 重构为**生成范围谱**(见 §2.2 / §3),FULL 坍缩作为正式发现报告;
> 2. 6-cell LoRA sweep 在飞,所有 gen_scope 结论按预注册式写法(预测 + 判定准则)标注;
> 3. SDD pkg-04/05 就地修订,带 `[v4-opt 2026-06]` 标记 + 修订记录(符合 Roadmap Part 6.4 陷阱 4 的 changelog 要求);
> 4. pkg-07/08 暂不撰写,其必备内容记录于 §10。

---

## 0. 复审方法与证据状态标签

**复审范围**:`docs/Chapter1_5 / 3 / 4_1 / 4 / 5 / 6` + `Hyper_MuZero_v4_Roadmap.md` + `ARCHITECTURE_OVERVIEW.md` + `DESIGN_DOC_FINAL.md §5.11-5.12` ↔ `hyper_mve/` 全部实现(HEAD = `dc5bbcd`)+ `sdd/pkg-01..06`。

**证据状态标签体系**(全文统一使用):

| 标签 | 含义 |
|---|---|
| `[已验证-代码级]` | 本次复审逐行读码确认,附 file:line 证据(附录 B) |
| `[已验证-测试级]` | 由仓库单元测试锚定(本次复审读了测试断言,未本地执行——本机无 torch) |
| `[已观察-单次运行]` | 来自单次训练运行的观察(如 duo FULL 坍缩),未重复、未跨 seed |
| `[初步-在飞]` | 6-cell sweep 正在运行,结论待填 |
| `[理论推断]` | 有形式论证但无实验证据 |
| `[未检验]` | 既无论证也无实验 |

**预注册式写法约定**:凡 `[初步-在飞]` / `[未检验]` 的断言,必须给出(i)现象/动机记录、(ii)机理假设、(iii)可判定的预测、(iv)判定准则与失败预案,缺一不可。

---

## 1. 理论→代码链条重述

研究主线的每一环,从问题定义到实现落点,逐环列出理论锚点、代码落点、验证状态:

| # | 环节 | 理论锚点 | 代码落点 | 验证状态 |
|---|---|---|---|---|
| 1 | RNS-MMG 问题定义 (C1-C3) | Ch1.3 定义 1 | — (问题层,无直接代码) | — |
| 2 | ResourceCommons 物理层(logistic + 邻居因子 + 公平共享) | Ch3.1-3.4 | `envs/resource_commons/dynamics.py` | `[已验证-测试级]` H1-H3 |
| 3 | 类型机制(α 纯自利 / β Fehr-Schmidt×φ(c)) | Ch3.5,Ch4.1.1 偏导表 {0.7, 2.0, 1.3, 0.0} | `envs/resource_commons/rewards.py`(含 torch 镜像供 autograd 测试) | `[已验证-代码级]` 四象限逐值一致 |
| 4 | 六块观测布局 + Self-Info | Ch3.7(45/99/195 维) | `schemas/observation.py`(SELF=4, RES=3K, NBR=9(N-1), GLB=2, CAP=4, TYPE=2) | `[已验证-代码级]` 但见 §5-M12(c_t 可观测性) |
| 5 | 三联上下文 80 维(c_ctx 16 + role 32 + belief 32) | Ch4.2.2-4.2.4 | `configs/model_config.py` 硬约束 + `models/tri_context_encoder.py` | `[已验证-代码级]` |
| 6 | BeliefNet(GRU + ĉ 头 + ẑ 头) | Ch4.2.3 / 4.5 | `models/belief_net.py`(共享 GRU、独立隐状态、z_hat 升序跳自身) | `[已验证-代码级]` 但 ĉ 头在可见-c 下退化,见 §2.3 |
| 7 | DualHyperNetwork 双路(客观 c_ctx→θ_state / 主观 80 维→θ_rew,θ_pred) | Ch4.3 | `models/hyper_network.py` `forward_trans`/`forward_subjective` | `[已验证-代码级]` |
| 8 | 生成范围谱(full/film_head/base_gen/lora_fc2 + 输出层 LoRA) | **原章节无**;DESIGN_DOC §5.12 | `models/functional_nets.py` `plan_generated_layers`/`count_generated`/`split_generated`;`hyper_network.py` 分组 RMS + output_rank | `[已验证-代码级]`(机制)+ `[已验证-测试级]`(参数计数)——**Ch4 待补,本文 §3 为措辞母本** |
| 9 | 五道稳定性防线(output_scale / AdaLN(1+γ) / Δs 残差 / EMA / 梯度门控) | Ch4.6 | `model_config.py` + `functional_nets.py` + `grad_gating.py`(双层 detach @5000) | `[已验证-代码级]` 防线 1 语义已被分组 RMS 改变,见 §3.4 |
| 10 | MVE 规划器(协调下降 + CRN,(B,spa,A) 布局) | Ch5.3-5.5 算法 5.1 | `planning/mve_planner.py` 四阶段 | `[已验证-代码级]` 布局一致;但 π_mve 实际含 z-score 归一(M7)、协调下降开关语义偏离(M8) |
| 11 | 训练循环(K 步展开 + n-step + EMA + 课程) | Ch5.6-5.9 算法 5.2 | `training/muzero_trainer.py` + `loss_composition.py` + `curriculum.py` | `[已验证-代码级]` 多处数值漂移,见 §5 |
| 12 | 三阶段课程(Oracle→退火→纯推断,30/40/30) | Ch5.7 | `training/curriculum.py`(含 oracle_only/infer_only 退化边界,已为 pkg-08 预留) | `[已验证-代码级]` |
| 13 | 规划信号必要性(planner-on 采集) | 算法 5.2 第 a10 行**已隐含**,但未论证其必要性 | `training/worker.py`(默认 use_planner=True)+ `scripts/train_main.py`(--no_collect_planner 仅作调试) | `[已观察-单次运行]` + `[理论推断]`(§4.1 不动点论证) |

**总评**:理论→代码链条在结构层面完整且严格(三联维度、双路分离、CRN 布局、课程边界、梯度门控均逐字落实);漂移集中在**数值超参层**(§5)与**优化阶段新增机制层**(环节 8、13,文档尚未吸收)。

---

## 2. 四个可证伪断言的适定性复审

### 2.1 断言 A(类型梯度撕裂)——仍适定,附两点边界警示

**原表述**(Ch1.5):类型异质下共享 RewardHead 显著劣于 per-agent θ_rew;差距在 50/50 混合时最大,同质时趋零。

**复审结论:适定性不受优化阶段影响**。类型梯度撕裂的物理基础(Ch4.1.1 偏导表)已 `[已验证-代码级]`——env 的 reward 实现与理论偏导逐值一致,且仓库存有 autograd 偏导测试。type_emb→hyper_rew→θ_rew^i 的架构通路在所有 gen_scope 下保留(gen_scope 只改变"生成多少",不改变"以什么为条件")。

**边界警示 1(N=2 零和退化)**:duo/easy 配置下 N=2,瞬时不平等差 $\Delta_\alpha = u_\alpha - u_\beta = -\Delta_\beta$ 严格反对称。这意味着 (c, Δ) 四象限在 N=2 时只剩两个自由象限(α 优势 ⇔ β 劣势同时发生),"撕裂"的几何结构比 N=4 更尖锐但更低维。消融 3 的 N=2 扫描(决策点 1)预期需按此修正:钟形曲线在 N=2 的三个数据点 (2α, 1α1β, 2β) 上,中点的撕裂强度可能**高于** N=4 的对应点(逐对完全对抗),不应直接外推到 Medium。`[理论推断]` → 挑战问题 Q11。

**边界警示 2(诊断指标已先行)**:`diag_cos_rew_cross / diag_cos_rew_same`(`loss_composition.py`)给出断言 A 的**训练时在线证据**(跨类型 θ_rew 余弦应显著低于同类型)。这是消融 3 之外的一条新证据通道,优化阶段已实装,Ch6.11 的离线偏导可视化可与之互证。

### 2.2 断言 B → 断言 B′(生成范围谱)——重构【本次复审核心决议】

**原表述的失效方式**:原断言 B 建立在"hypernetwork(per-context 权重生成)vs input conditioning(共享权重)"二分上,且 Ch4.1.2 把 FiLM (Perez et al., 2018) 引为 hypernetwork 一侧的支持文献。优化阶段两个事实使二分失效:

1. **FULL 全量生成坍缩** `[已观察-单次运行]`:duo 运行中预测超网络方向坍缩(`cos_pred_cross` 0.61→0.998),value/reward 拉锯,每步从"饥饿且被 detach 的上下文"重新生成全部功能网络权重不可训练(提交 `079fcdf` 动机记录);
2. **可工作配置是部分生成**:`film_head`(超网络只生成 FiLM γ/β + 输出头)与 `lora_fc2`(再加共享 fc2 的秩-r 增量)——而 FiLM 式调制在文献分类上通常归入 conditioning 而非 weight generation。原二分的两极之间出现了一个连续谱,且现有证据指向**最优点在谱的内部**。

**新表述(断言 B′,措辞母本——各章引用此处)**:

> **断言 B′(生成范围谱上的容量分配)**:把条件化机制按"上下文相关参数子空间的维度与秩结构"排成谱:input conditioning(仅上下文相关偏置)↔ FiLM/film_head(对角调制)↔ lora_fc2(秩-r 混合)↔ base_gen(顶层满秩)↔ full(全函数重实例化)。则:
> **(i)** 谱左端因**容量平均化**(表达力轴失败)在未见 c 值的零样本泛化上显著劣于部分生成——原断言 B 的方向保留;
> **(ii)** 谱右端因**优化失败**(权重估计方差与多损失拉锯导致的方向坍缩)显著劣于部分生成——FULL 坍缩作为发现报告;
> **(iii)** 性能峰值位于谱的**内部**,其位置揭示"上下文间结构性差异住在哪一层"——预注册预测:峰值在 film_head+LoRA 或 lora_fc2(差异主要住在 reward/policy 头与特征重混合,而非整个函数)。

**可证伪性增强**:B′ 比 B 有更多失败面——左失败(input conditioning 在零样本上追平 film_head ⇒ (i) 倒)、右失败(适当条件化下 full 不坍缩 ⇒ (ii) 倒,坍缩只是工程缺陷)、内点失败(单调递增或递减 ⇒ (iii) 倒,谱退化回二分)。每个失败面都有对应的判定 cell(§3.5)。

**对消融 1 的影响**:Ch6.4 变体表从 4 行扩展为 7 行 {Shared, Input-Wide, Input-Deep, Hyper-film_head, Hyper-lora_fc2, Hyper-base_gen, Hyper-full},gen_scope 成为消融 1 的新轴。等参数量协议在该轴上需要重新定义(挑战问题 Q5):各 cell 的 HyperNet 参数量天然不等(694k~3.09M),"等参数"应改为对齐**功能网络总参数 + 报告生成子空间维度**双指标。

### 2.3 断言 C(三联通路必要性)——主体适定,ĉ 分量的语义在当前实现下退化

**复审发现(本次新增,高优先级)**:实现中 `c_t` **直接写入每个 agent 的观测全局块** [`observations.py:160-162`],BeliefNet 不掩蔽该块 [`_belief_obs_encoder.py` 全量 MLP]。Ch3.7(391-393 行)其实**预先声明并辩护**了这一默认设定("环境信号显式标志",如气象预报),并规定了配套的"**c_t 隐藏模式**"(全局块中 c_t 替换为常数)作为 BeliefNet ĉ 推断头的"核心检验场景"。但 **c_t 隐藏模式在 v4 env 中未实现**(`EnvConfig` 无开关,观测构建无掩蔽路径)`[已验证-代码级]`。

**后果链**:
1. 可见-c 下,ĉ 头是输入中已有标量的恒等读出,MSE 监督**平凡可解**——ĉ 的"推断"语义为空;
2. Ch4.1.3 辅层 1(信念稀释)的动机叙事("agent i 推断丰年 ĉ≈0.8、agent j 推断荒年 ĉ≈0.2")在可见-c 下**不可能发生**——所有 agent 读出同一 c_t;信念稀释只能经由 ẑ(对手类型,确实不可观测)起作用;
3. 断言 C 的 No-belief 消融在当前 env 下测的是"ẑ 的贡献 + 一份冗余的 c 复制",而非完整 belief 通路;
4. worker 把 `info["c_true"]` 传入 `set_context_objective` [`worker.py:76-78`] 在可见-c 读法下自洽,但**违反 pkg-02 info 契约的字面规定**("Oracle 字段绝不进 model forward")——契约措辞需要修订(c_true 应降为 Public,或 env 直接经 obs 提供)。

**处置建议**(待作者决议,挑战问题 Q7):实现 c_hidden 开关(改动极小:`EnvConfig.c_visible: bool` + 全局块常数替换),并把信念实验拆为两档——可见-c 档(ẑ 是唯一推断对象,ĉ 头退化为正则化读出)与隐藏-c 档(ĉ 头成为真推断,Ch3.7 预定的核心场景)。零样本泛化实验(断言 B′ (i))在可见-c 下依然适定:模型在测试时**被给予**未见 c 值,检验的是 c→θ 映射的外推,与 ĉ 推断无关。

**detach_pred_context 的扰动**:duo 系预设改 `detach_pred_context=False`(理由记录于 `presets/duo.py` docstring:trunk 已是稳定共享 SGD 网络,放开 policy/value 梯度通达上下文编码器以解饿)。这改变了断言 C 的"通路贡献正交性"前提——value 损失现在也塑形 role/belief 编码器。消融 2 跑批前需固定该开关的取值并写明(挑战问题 Q8)。

### 2.4 断言 D(协调下降 × CRN 不可分割)——可测性危机,需重定义

**复审发现** `[已验证-代码级]`:

1. **−CRN cell 忠实**:`use_crn=False` 时其他 agent 第 0 步动作在 B·M 粒度独立采样 [`mve_planner.py:218-225`],严格对应 Ch5.4.1 方案 A。✓
2. **−CoordDesc cell 不忠实**:`use_coord_desc=False` **仅**把 agent 顺序从随机排列改为固定 0..N-1 [`mve_planner.py:153-157`];`optimised` 集合(已解 agent 以其 π_mve 行动)**无条件生效** [`162, 180-182, 270`]。即"关闭协调下降"后,协调下降的本体(逐 agent 顺序求解 + 已解者用解后策略)依然完整运行,只是去掉了顺序随机化。该 cell 实测的是"随机顺序 vs 固定顺序",**不是** Ch6.7 的"Joint 联合枚举 vs 协调下降"。
3. **Joint 枚举 cell 无任何代码路径**(仅 Easy N=2 时 6²=36 可行,Medium 1296 不可行——Ch6.7 自己已注明)。
4. **planner-off 基线空洞**:§4.1 的 ln(A) 不动点论证表明"无规划器"不是一个可用的 2×2 角点——它不是"弱一点的规划",而是训练信号整体归零。

**重定义建议(待作者决议,挑战问题 Q2)**:

| 轴 | 忠实实现 | 现状 |
|---|---|---|
| 轴 1:CRN on/off | `use_crn` | ✅ 已忠实 |
| 轴 2:协调下降 vs Joint 枚举 | 需在 pkg-08 新增 Joint 枚举模式(仅 Easy N=2) | ❌ 未实现 |
| 附加对照:顺序随机化 on/off | 现 `use_coord_desc` 改名为 `randomize_order` 更诚实 | ⚠️ 开关存在但名实不符 |

Ch6.7 在 Joint cell 实现前应加注说明,避免"已设计未实现"的断言在论文中被误读为已验证。

---

## 3. 生成范围谱:容量分配几何的重推导(Ch4.1.2 改写母本)

### 3.1 谱的形式化

把功能网络的有效参数写作 $\theta_{\text{eff}}(c) = \theta_{\text{base}} \oplus g_\Theta(c)$,其中 $\theta_{\text{base}}$ 由 SGD 直接训练(跨上下文共享),$g_\Theta(c)$ 由超网络 $h_\Theta$ 从条件向量生成(per-context)。条件化机制按 $g_\Theta(c)$ 的**维度与秩结构**排序(单隐层宽 $h$,以 fc2 为例):

| 谱位置 | 生成对象 | 等效结构 | per-context 自由度 | 实现 |
|---|---|---|---|---|
| Input conditioning | 无($c$ 进输入) | 首层 $W_c c$ ⇒ 仅上下文相关**偏置**(预激活平移) | $\mathcal{O}(h)$ | 消融 1 Input-Wide/Deep |
| FiLM / film_head | 各层 γ/β + 输出头 | $W_{\text{eff}} = \mathrm{diag}(1+\gamma) W_{\text{base}}$:**对角**乘性调制,可重加权特征、不能混合特征 | $\mathcal{O}(2h)$/层 + 头 | `gen_scope="film_head"` |
| lora_fc2 | + fc2 的 $\Delta W = B_f A_f$ | **秩-r** 非对角混合,$\Delta W$ 元素 RMS ≈ scale²·√r | $(h_{\text{in}}{+}h_{\text{out}})r$ | `gen_scope="lora_fc2"` |
| base_gen | fc2 + 头全生成(fc1 为 SGD 基座) | 顶层**满秩**上下文依赖 | $h^2$ + 头 | `gen_scope="base_gen"` |
| full | 全部权重生成 | 整函数 per-context 重实例化;SGD 只训练 $h_\Theta$ | 全部 | `gen_scope="full"`(legacy) |

**引文修复**:FiLM (Perez 2018) 从"支持 hypernetwork 的文献"重新归位为**谱上的对角点**;LoRA (Hu et al., 2021) 引为秩-r 点的出处;Ha (2017) 是 full 端;CAVIA (Zintgraf 2019) 实为靠近 input-conditioning 端的上下文参数适应,原 Ch4.1.2 将其与 Ha 并列属于误归组。

### 3.2 双轴权衡:表达力 × 优化

**表达力轴(原论证,保留)**:per-context 容量沿谱单调上升。Ch4.1.1 偏导表证明上下文间最优函数的结构性差异真实且巨大(β 偏导 0→2),input conditioning 的容量平均化(瓶颈 1-3)是谱左端的真实失败模式——这部分论证不需要撤回。

**优化轴(新增,优化阶段的教训)**:每个生成参数都必须经 $h_\Theta$ 从 80 维条件向量中"**估计**"出来,而该条件向量本身训练缓慢(belief 段前 5000 步被门控、value 路径可选 detach)。共享 SGD 基座是一个**跨上下文梯度累加器**——所有上下文的数据都在改进同一组 $\theta_{\text{base}}$;full 生成移除了这个累加器,于是:(a) 权重估计方差主导早期训练;(b) reward/value 两路损失在同一根生成向量上拉锯;(c) per-context 输出坍缩到单一方向以求自保——即观察到的 `cos_pred_cross` 0.61→0.998 `[已观察-单次运行]`。

**结论(替换原单边论证)**:谱的两端因**不同的机理**失败——左端败于表达力(容量平均化),右端败于优化(估计方差/方向坍缩)。原 Ch4.1.2 隐含假设"生成式条件化的优化是免费的",改进日志证伪了该假设。研究问题从"hyper 是否优于 input"升级为"**per-context 容量在谱的哪个位置开始得不偿失**"——预注册预测:结构性差异住在 reward/policy 头与特征重混合层,故峰值在 film_head+LoRA 至 lora_fc2 之间。

**Harsanyi 对应的强化(而非削弱)**:部分生成下,共享 SGD 基座 = **共同知识先验**(所有类型 agent 同意的函数骨架),生成的 FiLM/头/ΔW = **类型条件最优响应**(私有信念对骨架的调制)。这比 full 生成("每个 agent 的世界模型整体重实例化"——Harsanyi 框架并不要求如此)是**更贴切**的 Ch4.1.4 几何对应。`[理论推断]` → 挑战问题 Q4。

### 3.3 FULL 坍缩纪实(发现报告的事实底稿)

- **现象**(提交 `079fcdf` 动机记录,duo 单次运行):`cos_pred_cross` 从 0.61 升至 0.998(跨类型 θ_pred 方向几乎重合 = 角色坍缩);value 与 reward 损失拉锯;策略熵无改善。
- **机理假设**:见 §3.2 优化轴;另有一个独立的工程诱因——**整向量 L2 归一的 FiLM 稀释**:对整根生成向量做 L2 归一时,单个 γ 元素幅值 ≈ scale/√dim(0.1/√512 ≈ 4e-3)⇒ (1+γ)≈1 ⇒ FiLM 调制名存实亡 [`hyper_network.py:37-41` docstring 量化]。修复 = 分组 RMS 归一(film 段与 weight 段分开,每元素幅值 ≈ scale)。
- **证据等级**:`[已观察-单次运行]`。**正式结论需 sweep 的 FULL 对照 cell 复现**(若 sweep 不含 FULL cell,需补 1-2 seed 的 FULL 对照,否则论文中只能以"单次观察 + 机理论证"的弱形式报告)。

### 3.4 ΔW ≈ output_scale²·√r 尺度律与分组 RMS 纪律

分组 RMS 归一下每个生成元起始幅值 ≈ output_scale,故 lora_fc2 的 $\Delta W = B_f A_f$ 元素 RMS ≈ **output_scale²·√r**(r 项内积的方差合成)。数值:scale=0.1, r=8 ⇒ ΔW ≈ 0.028 ≈ kaiming fc2 基权(0.088)的 32%(有效);scale=0.01 ⇒ ΔW ≈ 3e-4(死)。该尺度律已固化为 `ModelConfig.__post_init__` 守门断言(scale ≥ 0.05)`[已验证-代码级]`,并有配套初始化纪律:LoRA 的 A 正交初始化、**B 用 small_init(std=0.01) 而非 0**(B=0 ⇒ raw=0 ⇒ 分组 RMS 除以 1e-8 下限 ⇒ step-0 约 1e4 的梯度尖峰)[`hyper_network.py:116-122`]。

**未决风险** `[理论推断]`:output_scale 是**可学习**参数,ΔW 随其**二次**增长——训练后期若 scale 增长(v4.7 经验:rew scale 会从 0.1 持续增长),ΔW 可能越过基权幅值。是否需要 scale clamp 或对 lora_fc2 改用 scale 的 sqrt 参数化 → 挑战问题 Q3。

### 3.5 6-cell sweep 预注册矩�阵

**实验设置** `[已验证-代码级]`:`scripts/run_lora_experiments.py`,3 建模情形 × 2 环境 = 6 runs,GPU 池 {2,3,4}(0/1 禁用),每 run 单卡(CUDA_VISIBLE_DEVICES 钉卡),落盘 `<env>/<model>[/<gen_scope>]/tb|ckpt|train.log`。

| cell | gen_scope | HyperNet 参数(§5.12,`[已验证-测试级]`) | 预测(预注册) |
|---|---|---|---|
| {duo,medium}_film_lora | film_head + LoRA(r=32) | ~694k | 稳定;`diag_cos_pred_cross` 维持 < 0.9;若失败模式出现,应为表达力不足(reward 拟合差) |
| {duo,medium}_film_lora_fc2 | lora_fc2(r=8) + LoRA(r=32) | ~896k | **预注册首选**:在 film_lora 基础上 reward 拟合显著改善且不引入坍缩 |
| {duo,medium}_base_lora | base_gen + LoRA(r=32) | ~2.30M | 表达力最高;风险 = 向 FULL 端的优化失败回归(监控 cos_pred_cross 上行) |

**判定准则**(thesis-default gen_scope 选型,= 新增"决策门 0"):
1. **硬门槛**:训练全程 `diag_pi_mve_entropy` 离开 ln(A)=ln 6≈1.79 并持续下降;`diag_cos_pred_cross` < 0.95 且不单调上行;
2. **选型**:在过硬门槛的 cell 中,以 medium 社会物理福利为主、参数量为辅(同福利取小);
3. **失败预案**:若仅 duo 过门槛而 medium 全不过 ⇒ gen_scope 结论限定于 N=2,Medium 需独立排查(容量或课程);若 film_lora_fc2 不优于 film_lora ⇒ ΔW 混合能力假设(§3.1 lora_fc2 行)被削弱,断言 B′(iii) 的峰值预测左移。

**证据力警示**(挑战问题 Q6):duo 系用 `c_mode="random_walk"`,而断言 B′(i) 的零样本协议是 **static 未见 c**。duo 结果对 B′ 只有间接证据力(机制存活性),正式判定必须落在 medium static cells + 后续消融 1。

---

## 4. 训练算法的修订事实(Ch5 修订母本)

### 4.1 ln(A) 不动点:规划信号是训练可行性的必要条件

**事实**:Ch5 算法 5.2 第 a10 行**本就规定**采集时运行规划器;实现初版偏离了文档(采集默认 use_planner=False),导致策略熵钉死在 ln(A) `[已观察-单次运行]`;提交 `0ba2eac` 恢复 planner-on 并加 `--no_collect_planner` 调试旗。**故这不是文档缺陷的修复,而是文档未论证的约束被违反后的实证**。文档真正缺的是"为什么 a10 行是 load-bearing"的论证:

> **自蒸馏退化引理(非正式)**:策略损失为 $\mathcal{L}_{\text{policy}} = -\sum_a \pi_{\text{tgt}}(a)\log\hat\pi(a)$。若采集时不运行规划器,则 $\pi_{\text{tgt}} = \hat\pi$(worker 在 use_planner=False 时直接存模型自身先验 [`worker.py:95-96`]),交叉熵对 logits 的梯度 $\nabla \mathcal{L} = \hat\pi - \pi_{\text{tgt}} \equiv 0$。策略网络没有任何改进信号,熵停留在初始化的 ln(A);且因 $\pi_{\text{mve}}$ 同时是 MVE 展开中其他 agent 的行为先验,规划器的后续调用也在退化分布上自洽——**均匀策略是整个"采集-规划-训练"闭环的不动点**。ε-greedy 不解此锁:它改变行为策略,但不改变 $\pi_{\text{tgt}}=\hat\pi$ 的恒等。

**写作含义**:该引理把贡献 3(MVE+CRN)从"提升样本效率的规划技巧"升格为"多智能体 MuZero 式训练在大 A^N 下**可行性**的必要组件"——是否值得正文一节 → 挑战问题 Q1。

### 4.2 实现相对算法 5.2 的其余修订(全部 `[已验证-代码级]`)

| 项 | 算法 5.2 / Ch5 文本 | 实现 | 处置 |
|---|---|---|---|
| 动作选择 | 从 π_mve 采样(a12-a14) | **ε-greedy**:ε 概率均匀随机,否则按 π_mve 采样;ε 1.0→0.05 线性 28k 步 [`worker.py:104-112`, `train_config.py:70-72`] | 文档补 ε-greedy + warmup(见下) |
| Warmup | 无 | 缓冲区达 min_buffer_size 前:ε=1.0 且 use_planner=False(纯随机填充)[`train_main.py:164`] | 文档补 |
| π_mve 归一 | softmax(Ḡ/τ) | softmax(**z-score**(Ḡ)/τ) [`mve_planner.py:262-266`] — τ 因此具有"每标准差"语义,对 reward 量纲不变 | 文档化(算法 5.1 修订)→ 挑战问题 Q12 |
| 视角采样 | 每样本随机/分层抽 1 个视角(5.6.5) | **每步训练全部 N 个 agent 视角** [`loss_composition.py:26-27, 193`];buffer 的"分层"是 episode 级 α-heavy/β-heavy 桶,且在固定 type_assignment 配置下**恒为单桶 = no-op** [`episode_buffer.py:117-130`] | 5.6.5 重写;stratified 开关的去留 → 挑战问题 Q9 |
| θ_state 在展开内 | 未规定 | 由窗口根部 c_t[:,0] 一次生成,K 步内复用 [`loss_composition.py:148-149`] — static c 下精确;**random_walk(duo)下是陈旧近似** | 文档注明近似及适用域 |
| 梯度半衰 | 无 | `s_pred = 0.5·s_next + 0.5·s_next.detach()`(v4.6 技巧)[`loss_composition.py:221`] | 文档补(5.8 稳定化) |
| n-step 截断 | z = Σγʳr + γⁿV | n_eff = min(n, K−t) 窗口内截断 + done 掩蔽 [`muzero_trainer.py:137-149`] | 文档补(细节级) |

### 4.3 诊断指标族(优化阶段新增,标准训练健康清单)

TB 命名空间路由 [`train_main.py:202-208`]:`diag_*` → `diag/`,`*_raw` → `loss_raw/`,其余 → `loss/`。

| 指标 | 语义 | 健康阈值 |
|---|---|---|
| `diag/pi_mve_entropy` | 规划器判别力;≈ ln A(=1.79, A=6)⇒ 未判别 | 训练中应离开 ln A 持续下降 |
| `diag/pi_pred_entropy` | 策略网络锐度(被 π_mve 蒸馏的下游) | 滞后于 pi_mve_entropy 下降 |
| `diag/cos_pred_cross` vs `cos_pred_same` | 跨/同类型 θ_pred 余弦;cross→1 = 角色坍缩(FULL 失败的直接指纹) | cross 显著低于 same;cross < 0.95 |
| `diag/cos_rew_cross` vs `cos_rew_same` | 同上,θ_rew;断言 A 的在线证据 | 同上 |
| `loss_raw/*` | 未加权损失量纲(w_* 预乘前) | 用于诊断权重配比 |

另有离线探针 `scripts/diagnose_mve.py`(加载 checkpoint,打印逐 agent `returns_per_action` / `q_normalized`),与 `MVEPlanner.sample_mve_plan(return_diagnostics=True)` 对应 `[已验证-代码级]`。

---

## 5. 理论↔代码一致性审计表

**一致项**(全部 `[已验证-代码级]`,证据见附录 B):∂R^β/∂u 四象限 {0.7, 2.0, 1.3, 0.0};CRN (B,spa,A) 场景外/候选内布局与 spa 均值;课程 30/40/30 与退化边界;ctx 维度 16+32+32=80 硬约束;梯度门控双层 detach(raw + ctx[48:80])@5000;观测总维 45/99/195 与 Self-Info type 2 维;lora_fc2 守门断言(≥0.05 / base_gen 禁用 / LoRA×共享 trunk NotImplementedError);LoRA 初始化纪律;分组 RMS 数学;−CRN cell 忠实性;§5.12 参数表(测试级)。

**漂移/缺口项**(编号 M1-M14,严重度:🔴 高 / 🟡 中 / 🟢 低):

| # | 项 | 文档值(锚点) | 代码值(锚点) | 严重度 | 处置 |
|---|---|---|---|---|---|
| M1 | EMA 形式与速率 | θ_tgt ← τ·θ + (1−τ)·θ_tgt,τ=0.005(Ch5.8.1) | θ_tgt ← τ·θ_tgt + (1−τ)·θ,τ=0.99 ⇒ 等效更新率 0.01(2 倍)[`muzero_trainer.py:217-222`] | 🟡 | Ch5 改写为代码约定 + 标记;速率差异注明"待回归验证" |
| M2 | 折扣 γ | 0.99,"γ⁵≈0.95"(Ch5.4.3) | 0.95 [`train_config.py:29`] | 🟡 | Ch5 改 0.95 并修连带算例 |
| M3 | 梯度裁剪 | 三路 1.0/5.0/1.0(Ch4.6.3)**且** 5.0(Ch6.12.2 表) | 全局单值 10.0 [`train_config.py:35`] | 🟡 | 两章统一为单值 10.0 + 标记 |
| M4 | 损失权重 | 主损失全 1.0,λ_b=0.5(Ch5.6.3) | w_policy=1.0, **w_value=0.25, w_reward=3.0**(v4.7 §5.11 教训), w_consist=0.5, **λ_b=w_belief=1.0 恒定** [`train_config.py:42-46`, `curriculum.py:87-94`] | 🟡 | Ch5.6.3 改写 + 引 §5.11 理由 |
| M5 | 视角采样 | 单视角分层采样(Ch5.6.5) | 全 N 视角逐步训练;buffer 分层在固定 type_assignment 下 no-op | 🔴(叙事级) | §4.2;5.6.5 重写 → Q9 |
| M6 | spa | 4(Ch5.5.1 + 复杂度算例) | spa = mve_samples//A = 50//6 = **8** [`mve_planner.py:146`] | 🟢 | Ch5 修数 + 算例 |
| M7 | π_mve 归一 | softmax(Ḡ/τ) | softmax(z-score(Ḡ)/τ) [`mve_planner.py:262-266`] | 🟡 | 算法 5.1 修订 → Q12 |
| M8 | 断言 D 的 2×2 | Joint/CoordDesc × CRN(Ch6.7) | −CoordDesc cell 仅固定顺序;Joint 未实现;planner-off 空洞 | 🔴 | §2.4 重定义 → Q2 |
| M9 | 预设漂移 | Easy/Medium static、scale 0.01/0.1/0.01、detach=True(pkg-01 spec06 / Ch3.9) | duo 系:random_walk + 0.1³ + detach=False(动机已记录于 preset docstring) | 🟡 | §6 登记;分类=有意优化,待回归验证 |
| M10 | Ch4.3.4 参数表 | ~3.2M(0.4/0.6/1.0M 三路) | FULL 3.09M;film_head+LoRA ~694k;+lora_fc2 ~896k;base_gen+LoRA ~2.30M(§5.12) | 🟡 | Ch4.3.4 增 per-gen_scope 列 |
| M11 | Ch4.1.2 FiLM 引文 | FiLM 列为 hypernet 侧证据 | film_head 即 FiLM 式调制 | 🔴(理论级) | §3 重推导(本次决议核心) |
| M12 | c_t 可观测性 | Ch1.3 "部分可观测";Ch3.7 默认可见 + **规定 c_hidden 模式**;pkg-02 契约 "c_true 绝不进 forward" | obs 全局块含真 c_t;**c_hidden 未实现**;worker 把 c_true 传 set_context_objective | 🔴 | §2.3;补 c_hidden(pkg-02 修订)→ Q7 |
| M13 | 展开内 θ_state | 未规定 | 窗口根部 c 一次生成,random_walk 下陈旧 | 🟢 | Ch5 注明近似 |
| M14 | 运行预算口径 | Ch1.5 "150-180 runs" vs Ch6.2.4 "~296 runs / 1800 GPU·h";Roadmap 决策点周序 (5/7/9/11) vs Ch6 附录 A (3/5/8/10/12) | — | 🟢 | 统一口径(以 Ch6.2.4 + Roadmap Part 4 为准) |

**纪律**:以上漂移仅做**文档侧**修订(代码值即意图值的,文档跟码并标记;代码缺口的,登记为 pkg 工作项)——本轮不改代码。

---

## 6. 配置漂移登记册(duo 系预设 vs 规范默认)

| 字段 | 规范默认 | duo 系取值 | 分类 | 动机记录 |
|---|---|---|---|---|
| `c_mode` | static(Easy/Medium) | random_walk | **有意设计**(非漂移):给 BeliefNet 非常数目标,防 static 下 ĉ 退化为常数读出 | `presets/duo.py` docstring |
| `trans/pred_output_scale_init` | 0.01 | 0.1(三路对称) | 有意优化,**待回归验证**(分组 RMS 改变了 scale 语义:per-element ≈ scale,旧 0.01 在 film 段无调制) | 同上 + `hyper_network.py:37-41` |
| `detach_pred_context` | True(D5) | False | 有意优化,**待回归验证**(FULL 下防 value 扭曲编码器;film_head 下 trunk 稳定、反而需要放开解饿)→ gen_scope 依赖开关,Q8 | `presets/duo.py` docstring |
| `hyper_output_rank` / `lora_fc2_rank` | None / None | 32 / 8(lora 系) | 新机制 opt-in | §3.4 守门断言 |
| `share_subjective_trunk` | False | False(LoRA 系放弃该轴) | 与 LoRA 互斥(NotImplementedError),暂搁置 | `hyper_network.py:278-282` |
| `easy` 预设 | N=2, 1α+1β, L=8 | **未漂移**,与 spec06 一致 | — | `presets/easy.py` |

**注**:duo 是规范从未定义的**新诊断预设族**(Medium 尺度 + N=2 + random_walk),与 easy(决策点 1 用)并存而不冲突——pkg-01 spec 06 修订时登记为新增,不是对 easy 的改动。

---

## 7. Roadmap 与决策门状态

**包进度**(HEAD = `dc5bbcd`):

| 包 | 状态 |
|---|---|
| Pkg-01 基础 schema/config | ✅ 实现 + 测试 |
| Pkg-02 ResourceCommons | ✅ 实现 + 测试;**缺 c_hidden 模式**(M12) |
| Pkg-03 TriContext + BeliefNet | ✅ 实现 + 测试 |
| Pkg-04 DualHyperNetwork v2 + 模型 | ✅ 实现 + 测试;优化阶段大幅扩展(gen_scope 族),SDD 待修订 |
| Pkg-05 Trainer + Worker + Planner | ✅ 实现 + 测试(Linux GPU 修复 `2916037`/`fd2204d`);SDD 待修订 |
| Pkg-06 基线族 | 📄 仅 SDD,**未实现**(断言验证的阻塞项) |
| Pkg-07 μP + 评估协议 | ❌ SDD 未撰写 |
| Pkg-08 实验/消融驱动 | ❌ SDD 未撰写(curriculum.py 已留 oracle_only/infer_only 钩子) |

**计划外阶段**:Stage 3(训练层)与 Stage 4(基线)之间插入了 Roadmap 从未定义的"**优化阶段**"(`e5a9e17..dc5bbcd`:planner-on 修复 → 诊断设施 → film_head → base_gen/共享 trunk → LoRA/lora_fc2/sweep)。该阶段事实上创建了一个新决策门:

> **决策门 0(gen_scope 选型,在飞)**:6-cell sweep 按 §3.5 判定准则选出 thesis-default gen_scope。**该门在决策点 1 之前**——因为消融 3(Hyper vs MA-MuZero)必须先固定 Hyper 自身的形态。

**原决策门状态**(Roadmap Part 4 口径,Week 5/7/9/11):

| 门 | 实验 | 状态 |
|---|---|---|
| 决策点 1(生死判官) | Easy 消融 3(N=2 类型扫描,Hyper vs MA-MuZero) | ❌ **未跑**——前置依赖:决策门 0 + Pkg-06 的 ma_muzero 基线 |
| 决策点 2 | Easy 消融 1(Hyper vs Input-Wide/Deep) | ❌ 未跑;且按 §2.2 需要重设计(+gen_scope 轴) |
| 决策点 3 | Medium 主对比 | ❌ 未跑 |
| 决策点 4 | 全实验汇总 | ❌ |

**口径冲突**(M14):Ch6 附录 A 的周序(3/5/8/10/12)与 Roadmap Part 4(5/7/9/11)不一致,且两者的"日历周"在优化阶段插入后均已失效——修订时改为**事件驱动序**(门 0 → 门 1 → …),不再绑定周数。

---

## 8. 风险与缺口清单(按严重度)

1. 🔴 **断言 D 实现缺口**(M8):Joint 枚举 cell 不存在 + planner-off 空洞 → 消融 4 按现状不可执行,需 pkg-08 实现或断言重定义。
2. 🔴 **c_hidden 模式缺失**(M12):BeliefNet ĉ 头的"核心检验场景"(Ch3.7 原文)不可运行;信念叙事(辅层 1)在可见-c 下部分空转。
3. 🔴 **Pkg-06 未实现**:断言 A/B′/C 的全部对照基线缺位,决策点 1 被阻塞。
4. 🟡 **FULL 坍缩证据等级不足**:单次运行;sweep 若无 FULL 对照 cell,需补跑,否则 B′(ii) 只能以弱形式写。
5. 🟡 **gen_scope 结论的环境外推**:在飞证据多来自 duo(N=2, random_walk);medium static 的结论独立性待 sweep 检验(§3.5 失败预案)。
6. 🟡 **等参数量协议失配**(Q5):谱上各 cell 参数量天然不等,消融 1 协议需重新定义后才可执行。
7. 🟡 **数值超参文档漂移**(M1-M4, M6):单独看皆小,但累计会让"按论文复现"失败——本轮文档对齐统一清偿。
8. 🟢 **口径/周序不一致**(M14)、θ_state 陈旧近似(M13)、stratified no-op(M5 附属)。

---

## 9. 致作者的挑战问题(讨论议程)

> 以下 12 题是本次复审认为**必须由作者决断**的理论/写作/排期问题。每题给出复审者的倾向供讨论,但不预设答案。

**Q1(ln(A) 引理的写作地位)**:§4.1 的自蒸馏退化引理是否进 Ch5 正文(作为贡献 3 的强化:规划信号 = 训练可行性必要条件)?还是仅作脚注?倾向:正文短节 + 一张"planner-off 熵曲线钉死 ln A"的实证图——它是审稿人能立刻复现的干净论断。**风险**:若写成正文断言,需补 1-2 seed 的 planner-off 对照曲线作图(目前是单次观察)。

**Q2(断言 D 重定义)**:接受 §2.4 的三轴重定义吗?Joint 枚举 cell(仅 Easy N=2,6²=36)是否值得 pkg-08 实现?`use_coord_desc` 是否改名 `randomize_order`?倾向:实现 Joint cell(代价低、是 2×2 的语义基石);开关改名。

**Q3(尺度律入正文 + scale clamp)**:ΔW ≈ scale²·√r 是否作为 boxed derivation 进 Ch4(随 §4.3.5 新节)?可学习 scale 的二次增长是否需要 clamp(如 scale ≤ 0.3)或 sqrt 参数化?倾向:进正文(它是 lora_fc2 可用性的判定性约束);clamp 待 sweep 的 scale 轨迹数据再定。

**Q4(Harsanyi 对应的再表述)**:接受 §3.2 的"部分生成 = 更贴切的 Harsanyi 对应"(共享 SGD 基座 = 共同知识先验,生成段 = 类型条件最优响应)吗?这会把"首次显式实现"的表述从 full 生成迁移到部分生成——新颖性是强化(对应更准)还是被质疑(FiLM 调制早已存在)?需要作者权衡答辩风险。

**Q5(等参数量协议在谱上的含义)**:消融 1 的对齐目标改为什么?选项:(a) 对齐功能网络总参数,HyperNet 参数另行报告;(b) 对齐"上下文相关子空间维度";(c) 双指标都报,放弃单一"等参数"声明。倾向:(c)——谱的本质就是参数量与结构的联动,硬对齐反而制造新的不公平。Input-Wide/Deep 对齐到哪个 cell(预注册首选 lora_fc2?)需同时定。

**Q6(duo 证据的适用边界)**:sweep 的 duo cells(random_walk)对断言 B′ 的 static 零样本协议只有机制存活性证据。正式结论是否严格限定在 medium static cells?倾向:是——并在 §3.5 判定准则中写死。

**Q7(c_t 可观测性与 c_hidden)**:接受 §2.3 的处置吗——(i) pkg-02 补 `c_visible` 开关;(ii) pkg-02 info 契约把 c_true 从"绝不进 forward"改为"可见-c 模式下作为公共上下文进入 set_context_objective";(iii) 信念实验拆可见/隐藏两档,ĉ 的推断叙事仅在隐藏档主张?这是本次复审发现的最深的一处理论-实现错位,建议优先讨论。

**Q8(detach_pred_context 的重述)**:是否将 D5 重述为 **gen_scope 依赖**的开关(FULL:True 防 value 扭曲编码器;film_head/lora_fc2:False 解饿,因 trunk 已是稳定 SGD 网络)?这改变 pkg-04 spec 01 §3.4 的"默认 True"语义,也影响消融 2 的固定值选择。

**Q9(Ch5.6.5 与 stratified 的去留)**:5.6.5 重写为"全 N 视角训练"(实现现状)即可,还是单视角采样在 N=8 Hard 下仍有算力价值、应保留为可选项?episode 级 stratified 在固定 type_assignment 下是 no-op——删除开关,还是保留给消融 3(混合 type_assignment 时才生效)?倾向:文档跟随实现 + stratified 保留并注明生效条件。

**Q10(决策点 1 排期)**:sweep 结束后的第一个 GPU 任务是什么?复审建议顺序:决策门 0 选型 → 补 FULL 对照(若缺)→ **实现 pkg-06 的 ma_muzero(最小可用)→ 决策点 1(Easy 消融 3)**。duo 诊断能否部分替代决策点 1?倾向:不能——决策点 1 的本体是 Hyper vs 共享 RewardHead 的**对照**,duo 只有 Hyper 自身。

**Q11(N=2 钟形曲线退化)**:§2.1 边界警示 1——决策点 1 的 N=2 扫描预期是否按 Δ_α=−Δ_β 零和结构修正(中点撕裂强度可能高于 N=4 外推)?这影响"Easy 结果外推 Medium"的降级预案触发条件。

**Q12(π_mve z-score 的地位)**:z-score 归一改变了温度的语义(τ 变为"每标准差"),且与 CRN 共同构成信号放大链。是否(a)文档化为算法 5.1 正式组件,(b)在消融 4 中加 z-score on/off 对照,还是(c)仅作实现注记?倾向:(a)+(c);(b)仅在审稿人质疑时补。

---

## 10. 下一步行动

### 10.1 本轮(文档对齐,本仓库,无 GPU)

按本文档措辞修订:Ch4_1(§3 母本)→ Ch4(新 4.3.5 + 4.3.4 参数表 + 4.6 防线修订)→ Ch1.5(断言 B′)→ Ch5(§4 母本)→ Ch6(消融 1 重设计 + 消融 4 加注)→ Roadmap(状态 + 决策门 0 + 事件驱动序)→ ARCHITECTURE_OVERVIEW(§11 优化阶段)→ SDD pkg-04/05 就地修订。

### 10.2 GPU 环境交接清单(按序)

1. **跑完 6-cell sweep** → 回填 §3.5 预注册矩阵 → 决策门 0 选型;
2. **补 FULL 对照 cell**(若 sweep 未含;1-2 seed 即可)→ 把 §3.3 从单次观察升级为可报告发现;
3. 文档对齐合入后,在 GPU 环境重跑 **pkg-4/5 测试清单**(`tests/` 全量 + checklist runner)确认无回归;
4. **实现 c_hidden 开关**(pkg-02 小修)+ ma_muzero 最小基线(pkg-06 首项)→ **决策点 1(Easy 消融 3)**;
5. 决策点 1 过门后再展开 pkg-06 其余基线与消融 1(按 Q5 决议的新协议)。

### 10.3 pkg-07 / pkg-08 SDD 必备内容备忘(本轮不撰写)

**pkg-07(μP + 评估协议)**:MupConfig base_shape 具体值与宽度缩放验证实验(LR 翻倍检验);统一 evaluator(Self-Info 严格:eval 期 set_context_subjective 不接 oracle types;c 分段 [0,0.3]/[0.3,0.7]/[0.7,1.0];零样本协议 train {0.2,0.5,0.8} / test {0.0,0.35,0.65,1.0});**新增**:可见-c / 隐藏-c 双档评估(Q7);π_mve 直推 vs 完整规划两种推断模式的评估开关(Ch5.9.3)。

**pkg-08(实验/消融驱动)**:消融 1 的 gen_scope 轴 cell 定义(§2.2);消融 4 重定义后的 cell(CRN × Joint/CoordDesc + randomize_order 对照,Joint 仅 Easy);oracle_only/infer_only(curriculum.py 钩子已在);Fehr-Schmidt 3×3 扫描;`--ablation` CLI 与 run 注册表;统计协议(Welch t、5 seeds、p<0.05)沿 Ch6.2.3。

---

## 附录 A:断言-实验-代码开关映射

| 断言 | 验证实验 | 关键代码开关/落点 | 当前可执行性 |
|---|---|---|---|
| A 类型梯度撕裂 | 消融 3(钟形曲线)+ Ch6.11 偏导可视化 + 在线 `diag/cos_rew_*` | `type_assignment`(EnvConfig);`rewards.py` torch 镜像(偏导);ma_muzero 基线(pkg-06,缺) | ⏳ 阻塞于 pkg-06 |
| B′ 生成范围谱 | 消融 1(7 变体)+ 零样本泛化 + sweep(决策门 0) | `hyper_gen_scope` / `hyper_output_rank` / `lora_fc2_rank`(ModelConfig);Input-Wide/Deep(pkg-06,缺) | sweep ✅ 在飞;其余 ⏳ |
| C 三联通路 | 消融 2(5 变体置零) | TriContextEncoder 通路置零(pkg-08 待接);`detach_pred_context`(Q8);c_hidden(缺,Q7) | ⏳ |
| D 规划器双重技术 | 消融 4(重定义后) | `use_crn` ✅;`use_coord_desc`(仅顺序,Q2);Joint 枚举(缺) | ⏳ 阻塞于 pkg-08 |

## 附录 B:审计证据索引(file:line)

- ∂R^β/∂u 四象限:`hyper_mve/envs/resource_commons/rewards.py`(compute_phi/psi + torch 镜像)
- CRN 布局:`hyper_mve/planning/mve_planner.py:146`(spa=S//A), `:166`(scenario 展开), `:190,205,210`(candidate 内层), `:262`(view(B,spa,A).mean)
- −CRN 忠实/−CoordDesc 仅顺序:`mve_planner.py:218-225` / `:153-157,162,180-182,270`
- π_mve z-score:`mve_planner.py:262-266`
- ε-greedy / warmup / planner-on:`hyper_mve/training/worker.py:49-50,95-96,104-112`;`hyper_mve/scripts/train_main.py:66-67,164,169-173`
- TB 路由:`train_main.py:202-208`
- EMA:`hyper_mve/training/muzero_trainer.py:217-222`;n-step 截断:`:137-149`
- 全 N 视角 / oracle 混合 / θ_state 根部 c / 梯度半衰:`hyper_mve/training/loss_composition.py:26-27,137-143,148-149,193-210,221`
- 诊断指标:`loss_composition.py:46-69,229-237,252-277`
- 梯度门控:`hyper_mve/models/grad_gating.py:31-79`
- 分组 RMS / LoRA 初始化 / 共享 trunk detach / NotImplementedError:`hyper_mve/models/hyper_network.py:28-57,111-122,149-216,278-282`
- gen_scope 计划/计数/拆分:`hyper_mve/models/functional_nets.py:192-336`
- 守门断言与尺度律:`hyper_mve/configs/model_config.py:123-143`
- 数值超参:`hyper_mve/configs/train_config.py:29,35,42-46,55-56,59,64,67,70-77,80-85`
- λ_b 恒定:`hyper_mve/training/curriculum.py:87-94`
- c_t 入观测 / 无掩蔽:`hyper_mve/envs/resource_commons/observations.py:156-162`;`hyper_mve/schemas/observation.py:44-49`;`hyper_mve/models/_belief_obs_encoder.py`
- c_hidden 模式缺失:`hyper_mve/envs/`、`hyper_mve/configs/env_config.py` 全文无 hidden/mask 开关(grep 验证)
- 预设:`hyper_mve/configs/presets/duo.py`(docstring 动机)、`easy.py`、`duo_film_lora*.py`、`run_lora_experiments.py`(GPU 池策略)
- 参数表锚定:`tests/models/test_hyper_network_lora.py`(test_lora_param_count_formula)、`tests/models/test_functional_nets_gen_scope.py`、`test_hyper_network_grouped_norm.py`
- 坍缩纪实 / 尺度律出处:`DESIGN_DOC_FINAL.md §5.11-5.12`;提交 `079fcdf` / `0ba2eac` / `dc5bbcd` 信息

---

*复审执行:2026-06-10,基于 HEAD `dc5bbcd`。本文档为后续修订的措辞母本;修订完成后,各文档与本文档冲突处以本文档为准,直至下一次复审。*
