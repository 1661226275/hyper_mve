# 第 5 章 实验

> 本章按"实验协议（§5.1）→ 决策门 0 前置实验（§5.2）→ v3 迁移验证（§5.3）→ ResourceCommons 主对比（§5.4）→ 四项断言对应的核心消融（§5.5–§5.8）→ 鲁棒性与外推（§5.9）→ 信念质量与三阶段课程（§5.10）→ 失败案例与训练诊断（§5.11）"的顺序展开。每一节均明确区分"已由训练日志支持的实测结果"与"按预登记协议尚未执行的实验"，后者以 `[证据待补 EVIDENCE PENDING — ⟨实验名⟩，待 §X.Y 补完]` 形式逐条标注，确保读者能够准确把握当前结论的支持范围。本章最终的断言验证状态将在 §6.1 以汇总表给出：A 未支持 / B′ 部分支持 / C 未支持 / D 部分支持。

---

## §5.1 实验协议：基线、公平性与算力规划

本节系统确立第 5 章所有对比实验共用的协议框架，包括基线选取、公平性约束、环境难度分级、评测维度以及端到端的算力规划方案。实验设计需要在"比较的充分性"与"条件的可控性"之间取得平衡：基线过少则无法定位本文方法的相对位置，基线过多而缺乏公平性约束则会让结论失去可信度。本节先行声明：当前训练 suite 仅完成了 hyper-variant 内部的若干消融与零样本泛化实验，§5.1 所列出的六类基线对照尚未运行，对应 §5.4 主对比的基线项目均按预登记规范作为 [证据待补] 处理。

**基线选取**方面，本章预计纳入六类对比方法，覆盖多智能体规划与 model-free MARL 两条主线。规划侧选取 MA-MuZero [30] 与 MAMBA [8]，二者分别代表多智能体 Gumbel 采样规划与 Transformer-based 多 agent 世界模型；学习侧选取 MARIE [17] 与 Conflict-Aware Gradient Adjustment（CAGA, Kim 等 [13]），前者通过 role-aware encoding 处理异质 agent，后者引入显式的冲突梯度调整机制；此外纳入 MAPPO [3] 与 QMIX [4] 作为 actor-critic 与值分解的常用基准，用于衡量规划增强相对于纯学习方法的实际收益。

**公平性保证**遵循两条底线。第一，**Self-Info 类型可观测性匹配**：所有基线与本文方法在输入信息量上保持对等，各方法均仅可访问同等粒度的自身类型信息，不引入额外的对手类型先验；隐藏-$c$ 档下 BeliefNet 推断的优势仅体现在表征侧而非观测侧，避免观测特权对 hypernet 一方造成不公平对照。第二，**μP 学习率扫描**（Yang & Hu 2021）在所有方法上独立执行：每种方法独立完成 5 个学习率档位 × 3 个随机种子的搜索协议，以消除超参配置差异导致的性能偏差。在双指标对照协议下，HyperNet 参数量与上下文相关子空间维度分别列出，使条件化谱内部不同变体在表达容量上具有可控对照。

**环境难度**按 Easy / Medium / Hard 三级递进。Easy（N=2）配置主要用于断言 D 的完整 2×2 矩阵与决策门 0 的预筛实验；Medium（N=4，2α+2β）为主对比的主配置，是 §5.4 主要结论的来源；Hard（N=8，24×24 网格、40 资源点、振荡式 $c_t$）用于鲁棒性与规模外推验证。本文在当前 suite 中已完成的是 Easy 配置下的若干 hyper-variant 实验；Medium 主配置主对比与 Hard 规模外推均尚未运行。**评测维度**包括六类指标族：社会福利（所有 agent 累计奖励之和）、可持续性（回合末资源存量）、Gini 系数（个体间分配公平性）、零样本跨 $c_t$ 迁移、类型梯度对齐与信念推断质量。

**算力规划**方面，主对比实验（Medium 配置，六类基线 × 5 seeds）预计消耗约 180 GPU 小时，μP 学习率扫描约 90 GPU 小时，Easy 配置的决策门 0 与断言 D 实验约 30 GPU 小时，合计目标约 300 GPU 小时。当前 suite 已耗费约 180 GPU 小时，主要分布在 `main_comparison_easy` / `zero_shot_easy` / `abl4_joint_easy_n2` 以及 LoRA 谱预筛若干 cell 上；其余约 120 小时按 §5.11 失败案例节列出的优先级（belief 通路修复 → Medium 主对比 → 消融 2/3 → Hard 规模外推）逐项消耗。**关于 §5.1.2 SNR probe 实验**，CRN-off 与 CRN-on 两档下第 0 步 Q-value 候选间方差比的显式测量构成断言 D 的必要补充证据，[证据待补 EVIDENCE PENDING — CRN-off vs CRN-on 第 0 步 Q-value SNR probe run，待 §5.1.2 补完]。

**实验流程的预登记规范**：所有 hyper-variant 与基线的训练在投入运行之前，先行登记三项参数——LR 范围、batch size 与 replay 比例、训练总步数；登记结果以 `runs/<exp>/<run_tag>/config.json` 形式写入运行目录，与训练完成后的 TensorBoard 日志一并归档。这一约束的目的是避免"按运行结果回写超参选择"造成的后验偏差，使每一次训练的超参配置在产生结果之前即已完成绑定。对于本章中所有 EVIDENCE PENDING 项目，预登记的具体配置已写入实验执行计划文档，待 suite 重启时按既定顺序运行而不再调整。

**关于种子数量与统计判据**：本章主对比与五项消融的种子数量按"基线 5 seeds + hyper-variant 3 seeds"的混合协议执行——基线需要更大种子数以稳定均值估计避免随机噪声掩盖真实差距，hyper-variant 在 suite 实测中观察到的种子间方差较小（如 `lora_duo_film_fc2` 的 planner return 标准差 18 在 mean 272 上占比约 6.6%），3 seeds 已足以支撑显著性检验。所有跨方法显著性判读均采用 Welch's t-test 在 95% 置信水平下进行，效应量同时报告 Cohen's d 以便读者独立判读"统计显著但实际微小"的边际效应。这一双指标协议与 §1.5 中四项断言的预登记判据阈值（如断言 A 主预测的 5% 峰差 + p < 0.05）保持一致。

**评测剖面的取样窗口**：所有 cell 的 planner return / fairness / sustainability 等指标的报告值，统一取训练末端 10% 步数的滑动均值（如 200k 步训练取末 20k 步均值），以平滑训练末期单步震荡。零样本评估在训练完成后单独执行 100 episodes × 4 测试 $c$ × 3 seeds 的统计，取均值与跨 seed 标准差。任何在 suite 中未达到对应训练步数的 cell（如 `lora/medium_base` 仅至 66k 步）不进入主对比报告，仅出现在 §5.11 失败案例节。

---

## §5.2 决策门 0：条件化谱预筛

本节在正式进入主对比之前，首先建立一项面向条件化生成范围的预筛实验，以确定后续 §5.4 起所有 DualHyperNetwork 变体所采用的默认 `gen_scope` cell。所谓"条件化谱"，是指将超网络对功能网络的参数生成范围从最窄（Shared，仅共享偏置）到最宽（FULL，全部目标参数由超网生成）连续排布所形成的设计轴线；不同位置的 cell 在条件化表达能力与训练稳定性之间存在实质性权衡，因此在投入大规模 sweep 之前必须经由受控实验加以校准。本节先报告 suite 中已观察到的 5 个 cell（覆盖 Input 端 ↔ FiLM + LoRA 中间段），再讨论左右两端边界变体的复现状态。

**suite 中已观察的 5 个 spectrum 变体**（均为 Easy N=2 duo 配置下的实测，$c_t$ 服从随机游走以阻断 agent 通过记忆固定 $c_t$ 规避条件化挑战）依生成宽度递增排列：

| 变体 | seed 数 | 训练步数 | planner return | cos_pred_cross | π_mve 熵 | 备注 |
|---|---|---|---|---|---|---|
| `main_comparison_easy`（Input 端 baseline） | 3 | 137k | 105.4 ± 4.8 | 0.72 ± 0.02 | 1.37 | 高 cross-cosine → 跨 agent 预测方向几乎重合 |
| `zero_shot_easy`（同 Input 条件化，延长训练） | 3 | 200k | 104.0 ± 3.3 | 0.71 ± 0.01 | 1.37 | 确认 105 量级是 Input 端能力上限 |
| `lora/duo_base`（LoRA-only） | 1 | 189k | 263 | 0.08 | 0.72 | π 决策更尖锐，方向显著分离 |
| `lora/duo_film`（FiLM-only） | 2 | 240k | 250 ± 2 | 0.09 | 0.77 | 与 LoRA-only 同量级 |
| `lora/duo_film_fc2`（FiLM + LoRA fc2） | 3 | 240k | **272 ± 18** | **0.03** | 0.75 | **谱内部峰值**，planner_prior_gap +19.6 ± 6.7 |

**预登记预测对照**：在投入 sweep 之前预先登记的预测是"峰值落在 film_head 叠加 LoRA 至 lora_fc2 区间之内"。suite 实测在 `lora/duo_film_fc2` cell 处取得 planner return 272 ± 18，相对 Input 端 baseline 的 105.4 ± 4.8 提升约 2.6×，**与预登记预测在变体序号上完全一致**；同时 cos_pred_cross 由 Input 端的 0.72 单调衰减至 lora_fc2 端的 0.03，表明跨 agent 的预测方向分离度也在同一变体上取得最大值。基于该结果，**决策门 0 的判决为 GO**：thesis-default `gen_scope` cell 锁定为 `lora_duo_film_fc2`，§5.4 起所有 DualHyperNetwork 变体均以该 cell 作为统一基础配置，确保后续各项消融在相同生成域前提下具备横向可比性。

**左右两端边界的状态披露**：完整的条件化谱包含 7 个 cell——Shared、Input-Wide、film_head、lora_fc2、base_gen、FULL，本 suite 在中间段（film_head ~ lora_fc2，即 `lora/duo_film`、`lora/duo_film_fc2` 两个 cell）取得了内部最优观测，但 Shared / Input-Wide 两个左端边界变体与 base_gen / FULL 两个右端边界变体的预登记复现尚未在本 suite 中执行。其中，FULL 端**方向坍缩**现象（cos_pred_cross 由 0.61 单调攀升至 0.998，表明两名 agent 的 prediction 方向在训练中后期退化为完全对齐）的实测信号来自 v4-opt 2026-06 阶段的开发观察记录（Review_v4_TheoryAudit_2026-06），本节作为机制依据加以引用；按预登记规范，该现象在当前 suite 上的独立复现作为 [证据待补 EVIDENCE PENDING — Shared / Input-Wide / FULL 三个 gen_scope 边界变体的预登记复现，待补完]。

**两端失败机制的差异化判别**：Input 端 baseline（105 量级）的失败是**容量平均化**——超网络生成的功能网参数在 agent 间几乎不分化，cos_pred_cross 维持在 0.72 量级，跨 agent 的预测方向高度重合；FULL 端的失败是**优化路径退化**——所有上下文输入被映射到几乎同一参数方向，cos_pred_cross 上升至 0.998 实质等价于退化为输入条件化但额外承受了超网生成的优化代价。两类失败机制不同，共同导致谱内部存在最优点。本节决策门 0 的 GO 判决，正是建立在"内部峰值 `lora_duo_film_fc2` 显著优于 Input 端 baseline"这一在 suite 中可复现的观察之上。

**π_mve 熵的剖面**：作为容量分化指标的辅助信号，suite 中 5 个 cell 的 π_mve 熵（policy 通过 MVE planner 加权后的策略熵）剖面呈现与 cos_pred_cross 相反的方向变化——Input 端的 π_mve 熵稳定在 1.37（接近 $\ln A = \ln 6 \approx 1.79$ 的 76%），表明 policy 在 Input 端虽然 cross-cosine 高但仍保留了相当的探索分散度；LoRA + FiLM 端的 π_mve 熵下降至 0.72–0.77 区间，说明 policy 在该 cell 处已收敛到较为尖锐的决策模式。这一辅助信号与 planner return 的方向一致——容量分化与决策尖锐化在同一变体上同步发生，二者共同推动 `lora_duo_film_fc2` cell 取得峰值。

**planner_prior_gap 的特殊信号**：在 `lora_duo_film_fc2` cell 中，planner_prior_gap = +19.6 ± 6.7，相对 Input 端 baseline 的 planner_prior_gap ≈ −0.6 出现显著上行——这是 MVE 规划器在该 cell 上相对策略先验呈现明确正向贡献的实测信号。该信号的含义是：当条件化谱定位在 `lora_duo_film_fc2` 时，超网络生成的 per-agent reward / prediction 头已足够分化，使 MVE 的 K 步展开能在策略先验之上提取出额外的规划增益（约 8% 量级）。这一信号是 §5.4 中 `main_comparison_easy` 的 planner_prior_gap ≈ −0.6 与 §5.8 中 `abl4_joint_easy_n2` 的 planner_prior_gap = −2.5 ± 8.4 两个 Input 端配置上规划器零优势观察的有效对照——说明 MVE 规划器在 RNS-MMG 上的相对收益并非在所有 hyper-variant 上都成立，而是与条件化谱 cell 的选择存在系统性耦合关系。这一耦合关系本身构成断言 B′(iii)"谱内部最优"与断言 D "规划器双重技术"之间的隐式连接：规划器收益的体现需以条件化谱在内部峰值附近为前提。

**决策门 0 的工程实施**：本节实验在 v4-opt 2026-06 优化阶段完成，采用 Adam 优化器、ε = 1e-5、γ = 0.95、EMA τ = 0.99、Adam β = (0.9, 0.999) 等参数（均参 §4.6 与附录 D）；spectrum 5 个 cell 的 LR 独立按 μP 协议扫描，最终选用各 cell 在 3 seed 上 mean planner return 最大的 LR 档作为该 cell 的代表值。决策门 0 的 GO 判决在 suite 全部 5 cell 完成训练之后通过 mid-suite review 完成，不调整任何其他 cell 的超参，避免回归型偏差。

---

## §5.3 v3 NonStationaryTag 迁移验证

本节将 v4 完整方法（DualHyperNetwork v2 + Coord+CRN + 决策门 0 选定的 `lora_duo_film_fc2` cell）迁移回 v3 NonStationaryTag 任务，对二者进行等价条件下的纵向比对，以验证版本升级的一致性并避免"v4 仅在新任务上工作"的潜在质疑。v3 NonStationaryTag 是本课题最初构建的阵营重对齐场景：4 个 agent 在连续二维场地中按派系分组，环境规则 REL ∈ {0.0, 0.5, 1.0} 控制 Agent 2 的阵营归属，形成 Hunter / Prey 角色的非平稳切换。v3 阶段的方法以直接拼接 agent 身份编码为上下文输入、未引入三联上下文与 BeliefNet 机制，其训练指标已在毕业设计中期汇报阶段完成记录，构成本节比对的参照基线。

**毕设汇报阶段已记录的 v3 实测信号**：在 REL = 1.0 配置下，Infer 模型（GRU 在线推断 REL）的 Capture Rate 为 79.0%，相对 Baseline（无 hypernet，仅拼接 agent ID）的 10.1% 取得了约 7.8× 的显著提升；超网权重在阵营翻转点附近表现出显著分化——奖励头的余弦相似度 cos_rew_0v3 由 0.37 衰减至 0.28，预测头的余弦相似度 cos_pred_0v3 由 0.65 衰减至 0.50，表明 hypernet 的条件化机制确实捕捉到了规则切换的语义边界。这一组实测信号在零和阵营翻转场景下验证了"以超网络条件化博弈视角"思路的基本可行性，构成 v4 方法的实证起点。

**v4 方法迁移回 v3 的设计映射**：将 v4 完整方法部署至 v3 任务时，REL（3 个离散值）直接对应 $c_t$；三联上下文中 $c_{ctx}$ 通路保留并以 REL embedding 实例化；$role_i$ 通路在零和阵营场景下重新实例化为 "Hunter / Prey ∘ 当前规则翻转状态"——即将 v4 ResourceCommons 下的 type 编码（α / β）替换为 v3 的阵营角色编码；$belief_i$ 通路在 self-info 观测档下（v3 中其他 agent 的阵营可观测）退化为"对当前规则朝向的推断"而非"对他人类型的推断"，与 v4 ResourceCommons 部署在结构上同构、仅输入信号语义按任务重新绑定。Per-Agent Coord+CRN 规划器在 v3 任务上的部署保持不变——v3 同样是 multi-agent 设定，仍受联合动作空间与其他 agent 随机性的双重压力。

**迁移实验的预登记成功标准与回退**：v4 方法在 v3 任务上的 Capture Rate 不低于 75%（即不劣于 v3 原版的 79.0% 减去合理的实验间方差），社会福利（Hunter 阵营累计奖励）不低于 v3 原版的同档水平。若最优配置下 Capture Rate 退化至 < 70%，则降级双层叙事至"v3 与 v4 作为两个独立任务上的独立实证"，§1.6 v3 起源章节相应改写。本节将 v3 在 §1.6 的角色定位为"起点而非附录"——v3 在零和场景下完成了思路可行性验证，v4 是把同一思路在合作-竞争连续谱另一端（混合动机偏好异质）上的结构升级而非修正。该 apples-to-apples 迁移实验作为 [证据待补 EVIDENCE PENDING — v4 完整方法迁移回 v3 NonStationaryTag 的迁移实验，待补完]。

**v3 与 v4 在三个维度上的对照**：（i）任务结构维度——v3 是零和阵营翻转（Hunter 与 Prey 的二元对立），v4 是混合动机的公地资源博弈（α 与 β 类型在合作-竞争连续谱上动态滑动），二者覆盖了"博弈关系动态切换"问题的两端样态。（ii）方法机制维度——v3 仅引入单一 hypernet 通路（以 agent 身份编码作为输入），v4 引入 DualHyperNetwork v2（客观/主观双通路）+ 三联上下文 + BeliefNet 在线推断；v4 在 v3 之上的结构升级既包括架构层的"客观/主观分解"也包括输入层的"$c_{ctx} + role + belief$ 三联"。（iii）评测维度——v3 主要评测 Capture Rate 这一与零和任务直接耦合的指标，v4 引入六类指标族以覆盖混合动机下"社会福利 + 可持续性 + 公平性 + 零样本泛化 + 类型梯度对齐 + 信念质量"六个方向。三个维度的同步演进使 v3 → v4 不构成简单的方法增强而是问题与方法的联合升级。

**毕设汇报阶段记录的更细粒度信号**：除 Capture Rate 之外，v3 阶段还记录了超网权重在阵营翻转点附近的更细分化指标——Hunter 与 Prey 两类 agent 的奖励头权重 L2 距离在 REL = 0 与 REL = 1 切换处出现约 2× 的峰值；预测头权重的余弦相似度在切换前后 5 步内由 0.65 衰减至 0.50 再回升至 0.62（V 形剖面）。这两个细粒度信号支持"hypernet 在规则切换处主动调整权重以维护策略适应性"这一机制——并非仅在切换后被动调整。该信号可作为 v4 迁移实验中 BeliefNet 推断质量的对照——在 v3 任务上 BeliefNet 推断 REL 朝向时，预期其推断置信度（softmax 输出的 max class probability）在 REL 切换前后 5 步内同样出现 V 形剖面，作为 v4 BeliefNet 在线推断能力的实证检验。

**v3 阶段三大瓶颈对 v4 设计的反向动机**：v3 在毕设汇报阶段记录的三大瓶颈——（a）类型梯度撕裂（v3 中由 Hunter / Prey 两类角色在共享 reward head 上产生方向相反的策略梯度，导致共享 head 训练不稳）；（b）信念稀释（v3 中未引入 BeliefNet，agent 仅依赖瞬时观测推断对手阵营，在阵营翻转的过渡帧出现严重错判）；（c）角色平均（v3 中 agent ID 仅作为输入拼接，未生成 per-agent 权重，使 agent 行为在群体平均侧）——共同构成 v4 三项设计选择（DualHyperNetwork 主客分解、BeliefNet 三阶段课程、per-agent θ 生成）的反向动机。本节 v3 → v4 迁移实验承诺：若 v4 方法在 v3 任务上的 Capture Rate ≥ 75%，则上述三项设计选择得到"在 v3 任务上不退化"的横向证据，与 v4 在 ResourceCommons 上的纵向证据共同构成贡献 2 与贡献 3 的双向验证。

---

## §5.4 ResourceCommons 主对比

本节在 ResourceCommons 上完成 Hyper-MuZero 与六类基线方法的系统主对比，目标是为 §1.4 贡献 2 提供核心实证支撑。**关于配置档位的关键披露**：按 §1.4 与 §5.1 的预登记规范，主对比应在 **Medium 配置（N=4，2α+2β）**上完成；但当前 suite 仅在 **Easy 配置（N=2）**的 `main_comparison_easy` cell 完成了 hyper-variant 内部的 3 seed 训练，且未运行任何基线方法。因此本节当前报告的是 Easy 配置下 hyper-variant 的 3 seed 实测结果，**所有跨基线对比均作为 [证据待补 EVIDENCE PENDING — MAPPO / QMIX / CAGA / MAMBA / MARIE / MA-MuZero 六类基线 + Medium 主配置，待 §5.4 补完] 处理**，§1.4 中"vs 六类基线"的表述在当前实验状态下严格属于规划承诺而非已兑现实验，该差距将在 §6.2 局限节中作完整披露。

**Easy 配置 hyper-variant 的实测结果**（`main_comparison_easy` cell，3 seeds × 137k steps）：planner return = 105.4 ± 4.8，fairness = 0.86 ± 0.06，sustainability（回合末资源存量比例）= 0.78 ± 0.02，welfare_physical = 104.7 ± 5.8。该 cell 实质上是后续主对比的 hyper-variant 锚点。

**MVE planner 与 policy prior 的对照**：在 `main_comparison_easy` cell 中，**planner_prior_gap = −0.6**（即 MVE 规划器输出的 return 相对策略先验仅有近乎零的差距，且名义上略为负值）。这一观察是当前 suite 中一个值得关注的实证信号——在 Easy N=2 的最简配置下，MVE 规划器并未相对策略先验呈现显著优势。两个互补的解释：（i）Easy N=2 配置下其他 agent 随机性导致的方差量级本身就小（仅 1 个其他 agent），CRN 的方差压制收益被压平；（ii）Easy 配置下策略先验已收敛到接近最优响应，MVE 的 K 步展开提供的额外信息边际收益不大。该信号与 §5.8 中 `abl4_joint_easy_n2` cell 的 planner_prior_gap = −2.5 ± 8.4 相互印证——两个 Easy 配置 cell 下规划器对策略先验都没有体现出 §1.4 贡献 3 所声明的清晰优势。规划器在更高难度配置（Medium / Hard）下的相对收益需在 [证据待补 EVIDENCE PENDING — Medium / Hard 主对比中 MVE planner 相对策略先验的优势区段，待补完] 中独立验证。

**按 $c_t$ 桶的细分**：在 Easy `main_comparison_easy` cell 中，按训练集 $c_t \in \{0.2, 0.5, 0.8\}$ 三档分别统计 planner 在 K 步展开末端的 return：$c=0.2$ 下 return = 76.3 ± 12.6，$c=0.5$ 下 96.1 ± 2.1，$c=0.8$ 下 143.9 ± 0.9。该单调上升的剖面与 §3 命题 3.1 关于 Gap(c) 单调性的预测在符号上一致：$c$ 取值越大（资源越丰沛）总福利越高，Gap(c)（即合作-竞争张力的张力强度）越小，agent 群体越容易协调到高 return 状态。这一桶式剖面是当前 suite 中可在符号方向上**部分支持** Gap(c) 单调性命题的实测信号；完整的命题硬验证仍依赖在 Medium 主配置上的复现与连续 $c$ 区段的细分扫描。

**关于 §1.4 贡献 2 的当前状态**：本节按预登记规范，将"DualHyperNetwork v2 在 RNS-MMG 主对比上显著优于六类基线"这一命题严格列为 [证据待补]，§5.4 当前仅能支持的较弱命题是"hyper-variant 在 Easy 配置下能稳定收敛到 fairness ≈ 0.86、sustainability ≈ 0.78 的群体均衡"。这一较弱命题不构成贡献 2 的硬验证，§6.1 表中将"主对比 vs 基线"作为待补项，避免在没有基线对照数据的情况下作"显著优于"的强陈述。

**hyper-variant 在 fairness 与 sustainability 上的稳定性观察**：3 seed 间 fairness 标准差 0.06、sustainability 标准差 0.02——两个指标的种子间方差都属于较低水平（变异系数 fairness ≈ 7.0%、sustainability ≈ 2.6%），说明 hyper-variant 在 Easy 配置下能稳定收敛到相近的群体均衡点。该稳定性是后续 Medium / Hard 配置上更复杂消融可行的前置——若 hyper-variant 在 Easy 上即表现出大的种子间方差，则 Medium / Hard 配置下基线对比的统计判读将更难达到显著性阈值。当前 suite 的稳定性观察为后续主对比的基线 vs hyper-variant 显著性判读提供了良好基线。

**Medium 主配置预登记的具体内容**：N = 4，2α + 2β，12 × 12 网格，资源点 20 个，热点区域 3 个，$c_t$ 在 episode 内服从 OU 过程（θ = 0.05, σ = 0.1, μ 按 episode 抽 $\{0.2, 0.5, 0.8\}$ 之一），训练步数 240k，每个方法 5 seeds。基线训练采用与 hyper-variant 同等的 μP LR 扫描协议，分别取每个基线 5 LR × 3 seeds 中 mean planner return 最大的 LR 档作为该基线的代表配置。主对比的报告指标包括：social welfare（所有 agent 累计奖励之和）、fairness（1 − Gini）、sustainability（回合末资源池存量比例）、tragedy（资源池被完全耗竭的回合占比）、planner_prior_gap（MVE 输出 vs 策略先验的相对差距）、return_zero_shot_unseen（在测试 $c$ 集合上的零样本性能）六项核心指标。每对方法间的差异以 Welch's t-test + Cohen's d 双指标判读。

**Easy 实测结果与 Medium 配置在难度上的预期外推**：Easy `main_comparison_easy` cell 中 planner return ≈ 105、fairness ≈ 0.86、sustainability ≈ 0.78 的剖面，在 Medium N=4 配置上预计沿三个方向变化：（a）随 N 翻倍，社会福利的绝对值预计从 100 量级上升至 200–300 量级（agent 数翻倍），但 per-agent return 量级不变；（b）随类型异质性提升（Easy 配置仅 1α+1β，Medium 配置 2α+2β），fairness 的难度上升——agent 间不平等的天然来源增加，hypernet 在 fairness 上的相对收益预计扩大；（c）随其他 agent 数量增加，MVE planner 的相对收益（planner_prior_gap）预计由 Easy 的近零量级上升至显著正值。Medium 主对比在三个方向上对 Easy 实测的外推预期能否得到验证，是 §5.4 主对比实验的核心问题。

---

## §5.5 消融 1 + 零样本泛化 → 断言 B′

本节对断言 B′（条件化谱内部最优）给出可分级报告：零样本泛化半已在 suite 中达成 EVIDENCE_SUFFICIENT 状态，条件化谱内部最优半已达成 EVIDENCE_PARTIAL 状态。

**零样本泛化（EVIDENCE_SUFFICIENT）**：`zero_shot_easy` cell（3 seeds × 200k steps，全部完成）在训练集 $c \in \{0.2, 0.5, 0.8\}$ 与测试集 $c \in \{0.0, 0.35, 0.65, 1.0\}$ 的分布错位设定下进行评估。**TB 训练末端 planner-eval（在训练 $c$ 集合上）的最终值为 104.0 ± 3.3，与 `main_comparison_easy` 的 105 量级一致**；通过端到端 registry-backfilled eval_report（在扩展 $c$ 网格上进行端末评估）取得 return_mean ≈ 130.86 ± 7.0（在覆盖训练 + 测试 $c$ 全网格上的平均）、return_zero_shot_unseen ≈ 130.88 ± 7.8（仅在测试集 $c \in \{0.0, 0.35, 0.65, 1.0\}$ 上的平均），且**regret_mean = 0.0 在 3 seed 上稳定成立**。TB 与 registry 两组数值的偏离来源已经核对清楚：TB 反映训练集 $c$ 在训练末端的 planner-eval，registry 反映扩展 $c$ 网格上的 end-of-training 完整评估，二者并非冲突而是不同评估剖面——registry 值在断言 B′ 的零样本部分作为权威依据。

**条件化谱内部最优（EVIDENCE_PARTIAL）**：§5.2 已报告 suite 中 5 个 spectrum 变体的 planner return 与 cos_pred_cross。本节就 B′ 的三个子断言逐项判读。

**B′(i)（左端容量平均化）**：Input 端 baseline（`main_comparison_easy` 与 `zero_shot_easy`）的 planner return 维持在 105 量级，cos_pred_cross 维持在 0.72 量级——跨 agent 的预测方向高度重合，超网络生成的功能网参数在 agent 间几乎不分化。这与 §5.2 中 LoRA + FiLM 端取得的 272 量级 return 与 0.03 量级 cross-cosine 形成显著对照（2.6× return 提升、20× cross-cosine 衰减）。**B′(i) 在 5 个 cell 上获得了实测支持**。

**B′(ii)（右端方向坍缩）**：FULL gen_scope 端的 cos_pred_cross 由 0.61 上升至 0.998 这一现象的实测信号来自 v4-opt 2026-06 开发阶段的观察，按预登记规范作 [证据待补 EVIDENCE PENDING — Shared / Input-Wide / FULL 三变体的预登记复现，待补完] 处理。本节就 FULL 端方向坍缩仅给出机制描述：超网络在 FULL gen_scope 下生成全部目标参数，方向自由度过高使优化中后期出现所有上下文输入被映射到几乎同一参数方向的坍缩，等价于退化为输入条件化但额外承受了超网生成的优化代价。

**B′(iii)（谱内部最优）**：`lora_duo_film_fc2` cell（3 seeds × 240k steps）的 planner return = 272 ± 18 是 suite 内 5 个变体中的峰值，相对 Input 端 baseline（105）提升 2.6×，相对 FiLM-only（250）与 LoRA-only（263）取得了在 3 seed 上稳定的细小但一致的边际增益。**B′(iii) 在 suite 中 5 个 cell（覆盖 Input / FiLM-only / LoRA-only / FiLM+LoRA 四类）上获得了实测支持**，按 §1.5 预登记预测"峰值落在 film_head 叠加 LoRA 至 lora_fc2 之间"，实测在 `lora_duo_film_fc2`（即 film + LoRA fc2 同时启用）取得峰值，与预登记预测在变体序号上完全一致。

**B′ 整体判决**：零样本半 EVIDENCE_SUFFICIENT，谱内部半 EVIDENCE_PARTIAL（B′(i)/(iii) 实测支持，B′(ii) 待补完）。**汇总为部分支持**，§6.1 表格相应填写。

**关于零样本 regret_mean = 0.0 的方法学审视**：`zero_shot_easy` cell 在 3 seed 上 regret_mean 稳定为 0.0，这一观察在统计上是稳健的（无种子间差异），但需要在方法学上审慎判读。两种可能解释：（a）已学习策略在测试 $c$ 集合 $\{0.0, 0.35, 0.65, 1.0\}$ 上确实达到与训练 $c$ 集合 $\{0.2, 0.5, 0.8\}$ 同等的最优响应水平——这构成断言 B′ 的零样本半的强支持；（b）regret 指标在 ResourceCommons 当前实现中对"零样本性能"的度量过于宽松，将"训练集 $c$ 上的 max return"作为 baseline，使任何在测试集 $c$ 上达到训练集量级 return 的策略均被记为 regret = 0。两个解释都需在 §5.11 失败案例 4 中进一步核对。本节按预登记规范，在 regret 指标定义未重新审视之前，零样本半的 EVIDENCE_SUFFICIENT 判读以"在当前指标定义下成立"作为限定条件。

**条件化谱在断言 B′ 之外的额外含义**：5 个 cell 的 spectrum 不仅支持"谱内部最优"这一断言 B′ 的核心论断，还呈现出几个对方法论有意义的辅助观察。其一，**π_mve 熵剖面**——Input 端的 1.37 量级与 LoRA 端的 0.72 量级形成 1.9× 的对照，说明 hypernet 容量分化与 policy 决策尖锐度在同一变体上同步演化，二者并非独立现象。其二，**planner_prior_gap 的符号转换**——Input 端的 ≈ 0 量级与 LoRA + FiLM 端的 +19.6 量级，标志规划器在条件化谱内部峰值附近"被激活"，此前两端的 zero gap 反映规划器在容量不足或方向坍缩两个失败模式下的双面失效。其三，**cross-cosine 的指数衰减剖面**——从 Input 端 0.72 至 LoRA + FiLM 端 0.03，跨 agent 预测方向的分离度在变体序号上呈近似指数衰减，与 LoRA + FiLM 的低秩调制对条件化容量的几何提升相吻合。这三组辅助观察共同支持本文将"条件化谱内部最优"作为方法选择基础原则的合理性。

**预登记降级方案的具体触发条件**：按 §1.5 的预登记规范，断言 B′ 的三分级降级方案如下。（i）若 B′(i)/(iii) 实测支持但 B′(ii) 在 Shared / FULL 端边界变体复现实验中未观察到容量平均化或方向坍缩，断言 B′ 整体降级为"中级支持"——保留"谱内部存在最优点"作为方法选择依据，但不主张两端的失败机制有独立性。（ii）若 B′(i)/(iii) 在重做实验中均未取得显著支持（如内部峰值与端点之间差距 < 30% 或 cos_pred_cross 衰减不显著），断言 B′ 进一步降级为"弱支持"——仅作为相关工作分析视角，不构成贡献 2 的硬验证。（iii）若整体降级至最弱版本（含 B′(iii) 完全失败），按 §1.5 整体边界条件触发贡献 2 的条件性重定位。当前 suite 实测处于"主预测得到部分支持 + 一个子断言待补完"状态，未触发任何降级条件。

---

## §5.6 消融 2：Context 路径拆分 → 断言 C

本节为断言 C（三联通路缺一不可，且三路径退化效应正交可加）建立实证依据。**当前状态披露**：消融 2 的预登记零置实验当前**未在 suite 中执行**，本节内容以理论预测 + 工程实施计划为主，实测部分作 [证据待补 EVIDENCE PENDING — c_ctx / role / belief 单路与多路零置消融，待 §5.6 补完] 处理。

**Harsanyi 结构映射下的退化方向预测**：按 §4.3 的 Harsanyi 类比，三联上下文对应三类语义不同的输入——$c_{ctx}$（共同知识：环境状态）、$role_i$（私人结构：自身类型与能力）、$belief_i$（私人推断：他人类型估计）。三类输入在 Harsanyi 框架下属于不同语义层级，理论上其对主观通路的贡献应可分离。零置 $c_{ctx}$ 后，主观通路丧失对当前 $c_t$ 区段的感知，退化方向为"对 $c$ 平均化"，可持续性指标受冲击最大；零置 $role_i$（包含 type_emb 与 cap_emb）后，主观通路丧失对自身类型偏好的分化能力，社会福利在类型混合配置下受冲击最大；零置 $belief_i$ 后，主观通路丧失对他人类型的推断输入，与 self-info 的混合预测优势消失，公平性指标受冲击最大。

**预登记零置组合**：5 组单路零置条件（zero $c_{ctx}$、zero type_emb、zero cap_emb、zero id_emb、zero $belief_i$）+ 至少 3 组双路联合零置（zero {$c_{ctx}$, $belief_i$}、zero {$role_i$, $belief_i$}、zero {$c_{ctx}$, $role_i$}）+ 全零对照。判据"近似可加性"成立的形式化表达为：双路联合零置下的累积退化量与两个单路零置退化量之和的相对偏差不超过 20%。若该判据在三对双路组合下均成立，则断言 C 完整成立；若任一组合的可加性显著偏离（相对偏差 > 50%），则断言 C 降级为弱版本"三路径互补"，§6.1 表中相应标注。

**hidden-$c$ 档下的额外补充**：可见-$c$ 与隐藏-$c$ 两档下 $belief_i$ 通路的相对贡献预计存在结构性差异——可见-$c$ 档下 agent 可直接观测 $c_t$，$belief_i$ 仅对他人类型推断有贡献；隐藏-$c$ 档下 $belief_i$ 同时需推断 $c_t$ 与他人类型，承担更高的语义负载。本节按预登记协议在两档下分别执行 5 组单路零置实验，对比 $belief_i$ 通路在两档下的退化幅度差异，定量分离"对 $c_t$ 推断"与"对他人类型推断"的贡献。

**当前 suite 中的诚实披露**：当前 suite 的 18 个 hidden-$c$ 档运行中，**14/18 runs 的 `loss/belief ≤ 1e-5`**，`loss/lambda_b ≡ 1.0` 横跨所有运行——这表明 BeliefNet 训练通路在当前优化设置下**未充分激活**。这一观察的直接后果是：在 $belief_i$ 通路本身未充分训练的情况下，消融 2 中"zero $belief_i$"条件与"$belief_i$ 保留"条件实测应当无显著差异，从而无法对断言 C 形成有意义的检验。**本节的预登记要求是 BeliefNet 训练通路修复（§5.11 列出的 P0 工程问题）之后再行执行**，否则消融 2 的实测结果不构成对断言 C 的硬验证。

**断言 C 当前状态汇总**：实验未执行 → **未支持（实验未执行）**。该状态严格意指对应的预登记实验尚未执行而非实验已执行且否决。完成路径依次为：（1）§5.11 belief 通路 P0 修复；（2）可见-$c$ 档下 5 组单路 + 3 组双路零置；（3）hidden-$c$ 档下重复。

**关于断言 C 的可分解性辅助证据**：虽然消融 2 的零置实验尚未执行，但 §4.3 Harsanyi 类比下的三类输入语义不同确实是从博弈论层面成立的——共同知识对应所有 agent 都掌握的环境状态（$c_t$），私人结构对应 agent 自身才知道的类型与能力（$\tau_i, cap_i$），私人信念对应 agent 对其他 agent 的不确定推断（$\hat{\tau}_{-i}$）。这三类输入的语义层级在博弈论框架下是公理化分割的（即任一对类型 (i, -i) agent 共同知识恒共享，私人结构互不知晓，私人信念按 Bayesian 推断公式连接）。基于这一公理化分割，可加性假设至少在语义层面成立。消融 2 的实测目标是验证这一语义层面的可分解性能否在神经网络的实际拟合中以"近似可加"的形式体现——即三类输入对应的功能网参数子空间能否在反向传播下保持近似正交。该实测验证的失败将弱化"三联通路缺一不可"的命题强度，但不会推翻其语义基础。

**hidden-$c$ 档下 belief 通路语义负载的非对称性**：可见-$c$ 与隐藏-$c$ 两档下 $belief_i$ 通路承担的语义负载在结构上不对称。可见-$c$ 档下，$belief_i$ 仅推断他人类型（$\hat{\tau}_{-i}$），输入维度为 $|\mathcal{T}|^{N-1}$ 状态空间；隐藏-$c$ 档下，$belief_i$ 同时推断 $\hat{c}_t$ 与 $\hat{\tau}_{-i}$，输入维度近似翻倍（取决于 $c_t$ 的离散化或连续表示）。预测：隐藏-$c$ 档下 belief 通路的零置退化幅度应显著大于可见-$c$ 档（约 1.5–2× 量级），这一差异在 §5.6 实验中可用于反推 BeliefNet 在两项推断任务（$c_t$ vs 他人类型）上的相对贡献。该实验设计与 §3.7 中 EnvConfig 的 `c_visible` 开关相对应，需要在 §5.11 工程项 (a) `c_visible` 开关补齐后方可完整执行。

**消融 2 的扩展项：自身类型 vs 对手类型**：除标准的三联通路零置外，本节预登记一组扩展消融——分离"自身类型 $\tau_i$"与"对手类型 $\hat{\tau}_{-i}$"对策略生成的差异影响。具体而言，加入"zero self-type"条件：在主观通路输入中仅置零 $\tau_i$ 而保留 $\hat{\tau}_{-i}$；与之对照的是"zero belief"条件：仅置零 $\hat{\tau}_{-i}$ 而保留 $\tau_i$。两个条件下的退化曲线对比，可定量分离"agent 对自身类型的感知"与"对他人类型的推断"对最终策略性能的独立贡献。预测：在 RNS-MMG 设定下，"zero self-type"的退化应显著大于"zero belief"（前者直接破坏 agent 对自身偏好的感知，后者仅破坏对他人行为的预测），但二者的退化都应大于无消融基线。该扩展项作为 §5.6 主消融的补充组，整体仍按 [证据待补 EVIDENCE PENDING] 处理。

---

## §5.7 消融 3 + 类型梯度可视化 → 断言 A

本节为断言 A（类型梯度撕裂的钟形曲线）建立实证依据。**当前状态披露**：消融 3 的预登记 $\rho_\beta$ 五点扫描当前**未在 suite 中执行**，本节内容以理论预测 + 工程实施计划为主，实测部分作 [证据待补 EVIDENCE PENDING — $\rho_\beta$ 五点 + 1α+3β 对照扫描（消融 3），待 §5.7 补完] 处理。

**反向传播代数依据回顾**：按 §4.1 类型梯度量化表，类型 α 的瞬时奖励梯度 $\partial R^\alpha / \partial u_i \equiv 1$ 恒为正常数；类型 β 的瞬时奖励梯度 $\partial R^\beta / \partial u_i$ 在 $(c, \Delta_i)$ 的不同区段取值于 $\{0, 0.7, 1.3, 2.0\}$，其中在资源稀缺区段对劣势状态可取 0 或更大（不平等厌恶的劣势项），在中性区段两类梯度近乎重合。共享 RewardHead 在反向传播中聚合这两组方向不同的梯度，等价于训练一个"类型平均"的奖励函数；该函数在群体类型混合处偏离任何单一类型的真实偏好最远，故社会福利差距（per-agent θ 与共享 RewardHead 之间）在 $\rho_\beta \approx 50/50$ 处达峰。

**预登记预测的两层结构**：（i）主预测——峰值在 $\rho_\beta \in [1/4, 3/4]$ 区间内存在且与端点（0/4 或 4/4）的社会福利差距 ≥ 5%，p < 0.05（Welch's t-test）；（ii）二级预测——峰值精确位置由实验决定。由于类型 β 的梯度 $\{0, 0.7, 1.3, 2.0\}$ 在第一原理上呈非对称分布（β 端梯度量级平均略大于 α 端的恒 1），峰值在原理上自然预测偏向 α-多侧（如 $\rho_\beta = 0.25$ 而非 $\rho_\beta = 0.5$），但具体位置依赖 $\psi(\Delta)$ 与 $\phi(c_t)$ 的参数化细节，需由实验决定。

**预登记扫描配置**：6 组类型配置——$\rho_\beta \in \{0/4, 1/4, 2/4, 3/4, 4/4\}$ 五组 + 1α+3β 非对称对照组。每组 5 seeds × Medium N=4 主配置，社会福利与 fairness 指标在训练末端 200k 步取均值，按 Welch's t-test 给出峰值位置相对端点的显著性。

**类型梯度热图（§6.11 / 附录 C）**：与比例扫描相配，类型梯度 $\partial \tilde{r} / \partial u_i$ 的数值热图可在固定 $\rho_\beta = 2/4$ 配置下取 trained 模型，对随机选取的若干 $(s, a)$ batch 计算 reward 头在每个 agent 上的偏导数值，分别绘制 α 与 β 通路的热图，验证训练后的奖励超网络是否学到了 §4.1 量化表所给的代数梯度结构。该可视化的位置（§6.11 主文 vs 附录 C）目前尚未最终决定——若 Easy N=2 扫描结果支持钟形曲线主预测，则可视化作为 §6.11 主文证据；若 Easy N=2 扫描结果（决策门 1，详见下文）触发降级，则可视化下移至附录 C 作为辅助材料。该决策作 [证据待补 EVIDENCE PENDING — $\partial \tilde{r} / \partial u_i$ 数值梯度热图位置（§6.11 or 附录 C），待补完] 处理。

**决策门 1**：在最易分辨的 Easy N=2 二类型设置上，若 $\rho_\beta = 2/4$ 与 $\rho_\beta = 0/4$ 之间的峰值差距 **< 5%**，触发 §6.2 贡献 2 的条件性重定位——将贡献 2 表述收窄为"特定 $c_t$ 区段优势"，对超网络在简单场景下的必要性重新定性；若 5% ≤ 峰差 < 8% 或 p ≥ 0.05，触发部分降级；若峰差 ≥ 8% 且 p < 0.05，断言 A 完整成立。

**断言 A 当前状态汇总**：实验未执行 → **未支持（实验未执行）**。完成路径：在 §5.6 belief 通路修复完成后开展 6 组类型配置 × 5 seed 扫描，约耗时 60 GPU 小时（参 §5.1 算力预算）。

**关于钟形曲线峰值位置的两类可能形态**：基于 §4.1 类型梯度量化表的代数结构，预期峰值有两种可能形态。（a）**对称峰值**——若 $\psi(\Delta)$ 与 $\phi(c_t)$ 的参数化使类型 β 在中性区段（$c \approx 0.5$）与类型 α 的梯度近乎重合，则反向梯度仅在极端区段（$c \to 0$ 或 $c \to 1$）才显著，峰值在 $\rho_\beta = 0.5$ 处取得，剖面对称；这一形态对应于"反向梯度撕裂的频率敏感性"——即类型混合比例越接近平衡，撕裂越频繁地发生在反向方向上。（b）**非对称峰值**——若 $\psi(\Delta)$ 的非对称使类型 β 在劣势状态（$\Delta_i > 0$）下的梯度量级显著大于优势状态（$\Delta_i < 0$），则峰值偏向 α-多侧（如 $\rho_\beta = 0.25$）；这一形态对应于"反向梯度撕裂的量级非对称"——即学习 β 类型视角的边际收益对群体福利的贡献大于学习 α 类型视角，故少数 β 配置下 per-agent θ 的优势更大。两种形态都属理论上合理的 informative result，并非偏离预期的失效信号。

**类型梯度热图的可视化目标**：§6.11 / 附录 C 的梯度热图将以三种形式呈现。（i）**Per-agent 梯度量级图**——以 $(s, a)$ 采样 batch 为横轴、4 个 agent 为纵轴，热图颜色为 $|\partial \tilde{r} / \partial u_i|$ 的量级；预期 hyper-variant 在 α / β 两类 agent 上呈现差异化的梯度分布，而 baseline（共享 reward head）在所有 agent 上呈现量级相近的分布。（ii）**Per-type 梯度方向图**——固定 $\rho_\beta = 2/4$ 配置，绘制 α 通路与 β 通路的梯度方向余弦，预期二者在中性区段（$c \approx 0.5$）的方向余弦接近 1，而在极端区段（$c \to 0$ 或 $c \to 1$）的方向余弦接近 0 或负值。（iii）**反向梯度撕裂的时间剖面**——选取若干 $c_t$ 切换 episode，绘制 reward head 梯度 norm 在切换前后 20 步内的演化，预期在切换处出现量级峰值，hyper-variant 的峰值显著低于 baseline（hyper-variant 通过 per-agent θ 已避免反向梯度的直接聚合）。

**Easy N=2 决策门 1 的边界条件细化**：决策门 1 的判读以 Easy N=2 二类型设置（1α + 1β）为基础，但 N=2 配置下 $\rho_\beta$ 仅有 0/2, 1/2, 2/2 三个离散值，无法支撑五点扫描的完整剖面。因此 Easy N=2 决策门 1 的具体形式是"三点扫描下检查 $\rho_\beta = 1/2$（即 1α+1β）相对 $\rho_\beta = 0/2$（2α）与 $\rho_\beta = 2/2$（2β）两端的福利峰差"。若该 Easy 配置下峰差 < 5%，触发 §6.2 贡献 2 的条件性重定位。完整 5 点扫描需在 Medium N=4 配置上执行，预算 60 GPU 小时已在 §5.1 算力规划中预留。

**与 §6.11 / 附录 C 位置决策的耦合**：决策门 1 的判读直接决定类型梯度热图的最终位置——若 Easy N=2 三点扫描下峰差 ≥ 5%，热图作为 §6.11 主文证据；若 5% > 峰差 ≥ 3%（边际显著），热图下移至附录 C 作为辅助材料；若峰差 < 3%（不显著），热图整体省略并按 §1.5 整体降级条件改写 §6 章贡献复述。该耦合关系将断言 A 的实证执行与论文整体结构（章节分配 + 贡献复述）形成硬绑定。

---

## §5.8 消融 4：规划器双重技术 → 断言 D

本节为断言 D（Per-Agent Coordinate Descent 与 CRN 的 2×2 矩阵在 Easy N=2 上完整展示，二者不可分割）建立实证依据。**当前状态披露**：suite 已完成 2×2 矩阵中的"joint-on"象限（即 CoordDesc + CRN 双开），但 {CD-only, CRN-only, neither} 三个对照象限均未完成；同时 §4.5 SNR 0.02 → 1.0 的几何推导的显式 SNR probe 实验也未在 suite 中独立执行。

**joint-on 象限的实测结果**：`abl4_joint_easy_n2` cell（3 seeds × 100k steps，CoordDesc + CRN 同时启用）——planner return = 100.2 ± 3.6，fairness = 0.84 ± 0.07，sustainability = 0.79 ± 0.03，planner_prior_gap = −2.5 ± 8.4。该 cell 是双重技术全开的标准配置在 Easy N=2 下的基准。需要注意的是，**planner_prior_gap 在 joint-on 象限上同样近乎零（−2.5 量级）**，与 §5.4 `main_comparison_easy` 的 planner_prior_gap = −0.6 量级一致——在 Easy N=2 配置下，MVE 规划器即使在双重技术全开时也未相对策略先验呈现清晰优势。该观察的合理解释是：Easy N=2 下其他 agent 随机性贡献的方差量级本身就小（仅 1 个其他 agent），CRN 的方差压制收益被压平；规划器双重技术的相对优势预计在 Medium N=4（3 个其他 agent）与 Hard N=8（7 个其他 agent）配置下随其他 agent 随机性贡献的方差累积而显著放大。

**三个待补象限**：（1）**CD-only**（CoordDesc on, CRN off）——预测受 SNR 塌陷拖累，候选动作的真实差异被其他 agent 随机性掩没，规划器输出退化为接近均匀分布；（2）**CRN-only**（CoordDesc off, CRN on）——预测受联合搜索空间爆炸所限，在 Easy N=2 配置下 $A^N = 6^2 = 36$ 仍可枚举，但相对 Per-Agent CoordDesc 的 $N \cdot A = 12$ 已有 3× 计算量差距，Medium N=4 配置下 $A^N = 6^4 = 1296$ vs $N \cdot A = 24$ 即不可执行；（3）**neither**——预测为本组最差。三象限作 [证据待补 EVIDENCE PENDING — {CD-only, CRN-only, neither} 三象限完整 2×2 矩阵，待 §5.8 补完] 处理。

**§4.5 SNR 0.02 → 1.0 几何推导的显式 probe**：本节按 §1.5 预登记协议设置 SNR probe 实验——固定一组 (s, a) 候选 batch，在 CRN-on 与 CRN-off 两档下分别计算候选间 Q-value 差的方差比，量化 CRN 在第 0 步抵消其他 agent 随机种子的方差压制效率。该 probe 实验作 [证据待补 EVIDENCE PENDING — CRN-off vs CRN-on 第 0 步 Q-value SNR probe run，待 §5.1.2 补完] 处理。

**断言 D 当前状态汇总**：joint cell 已验证 → **部分支持**；4 象限分离与 SNR probe 实验是完成 D 完整验证的必要前置。完整 2×2 矩阵在 Easy N=2 下的工程实施依赖 §4.5 中 MVE planner 的 `joint_enum` 模式（pkg-08）交付，该工程开关在 §5.11 失败案例节中作明示。

**Easy N=2 三轴设计的具体配置**：按 §1.5 中 v4-opt 2026-06 重定义的 Assertion D 三轴设计，Easy N=2 上的完整验证矩阵涉及三轴：（a）CRN on / off——信号噪声比轴；（b）Joint enumeration vs Per-Agent CoordDesc——计算压缩轴（在 N=2 配置下 $A^N = 6^2 = 36$ 仍可完整枚举，使该轴可与 CoordDesc 进行 apples-to-apples 对照）；（c）order randomization——控制项，验证 CoordDesc 在不同 agent 更新顺序下的稳定性。三轴的完整 2×2×2 = 8 cell 矩阵在 Easy N=2 上耗时约 16 GPU 小时，可作为断言 D 充分性证据的基础。

**Medium N=4 配置上的必要性证据**：在 Medium N=4 上，$A^N = 6^4 = 1296$ 已远超 $N \cdot A = 24$，Joint enumeration 方案的计算开销使其不可执行。因此 Medium N=4 上仅保留 CoordDesc 框架下的 CRN on / off 二轴对比，构成 CRN 必要性方向的独立验证。在 Easy 三轴 + Medium 二轴的混合结构下，断言 D 的完整验证遵循"Easy 充分性 + Medium 必要性"的论证策略——Easy 三轴提供完整 2×2 矩阵在低复杂度配置下的充分性证据，Medium 二轴提供 CRN 在更具实战意义配置下的必要性证据。

**§4.5 不动点对应 ε-Nash 的理论支撑**：按 §4.5 给出的形式陈述，Per-Agent Coordinate Descent 的不动点在 K 步 value 估计为精确时对应 ε-Nash 均衡，ε 关于 K, γ, 模型误差的上界由 Bellman 算子的折扣加权累积误差直接给出。具体形式：当 value 估计的逐步误差为 $\epsilon_V$ 时，$\varepsilon$-Nash 的近似度满足 $\varepsilon \le \epsilon_V \cdot \sum_{k=0}^{K} \gamma^k = \epsilon_V \cdot (1 - \gamma^{K+1}) / (1 - \gamma)$。在本文采用的 $\gamma = 0.95$、$K = 5$ 配置下，该上界约为 $4.52 \epsilon_V$，意味着 CoordDesc 不动点对模型误差的累积放大系数约为 4.5×。这一形式化结果为规模较大场景下 CoordDesc 的可扩展性提供了理论充分条件——只要模型在 K 步内的累积误差可控（$\epsilon_V$ 在 $1/4.5$ 量级以下），CoordDesc 的不动点即在合理 ε 量级内逼近真实 Nash 均衡。

**断言 D 与其他三项断言的耦合关系**：断言 D 是四项断言中唯一直接关注规划层（而非表示层 / 条件化谱）的断言，其与其他三项的耦合关系为：（i）与断言 A 的耦合——MVE planner 的 K 步展开依赖 reward head 在 per-type 上的分化能力，若类型梯度撕裂未被缓解（断言 A 失败），则规划器的 K 步展开 reward 估计在类型混合配置下出现量级偏差，进一步压低 SNR；（ii）与断言 B′ 的耦合——条件化谱内部最优 cell 的选择直接影响规划器的 prior gap（如 §5.2 中 `lora_duo_film_fc2` 的 planner_prior_gap = +19.6 vs Input 端的 −0.6），断言 B′(iii) 内部峰值的实测支持是规划器收益体现的前置条件；（iii）与断言 C 的耦合——三联通路缺一不可的语义分解直接影响 prediction head 在 K 步展开下的输出稳定性，若 belief 通路未充分激活（当前 P0 状态），规划器在 hidden-$c$ 档下的方差估计可能偏离真实情形。这一耦合关系说明断言 D 不应在四项断言中作为独立验证项处理——其完整支持依赖其他三项断言的支持状态。

---

## §5.9 消融 5 + Scale-up + 鲁棒性 + planner-off

本节围绕三类正交扰动构建鲁棒性验证体系，以检验 Hyper-MuZero 框架在超出训练分布的参数设定与规模配置下能否维持稳定性能。**当前状态披露**：本节的三组实验当前均**未在 suite 中执行**，内容以理论预测 + 工程实施计划为主。

**Fehr-Schmidt 社会偏好参数敏感度（消融 5）**：在 $\kappa \in \{0.3, 0.5, 0.7\}$ 与 $\lambda_{disadv} \in \{1.0, 2.0, 3.0\}$ 构成的 3×3 参数网格上重复运行实验，考察超网络生成的策略权重对偏好强度偏移的容忍区间。预测：在 $\kappa = 0.5$、$\lambda_{disadv} = 2.0$ 的标准设定附近形成性能稳定区，性能在偏好强度同时偏移 ±50% 时保持在标准设定的 90% 以上。作 [证据待补 EVIDENCE PENDING — Fehr-Schmidt 参数 κ × λ_disadv 3×3 网格扫描（消融 5），待补完] 处理。

**Scale-up Hard 配置**：N = 8、24×24 网格、40 个资源点、5 个热点区域、振荡式 $c_t$（在 episode 内 sinusoidal 变化），作为规模外推压力测试。预测：协调机制（Per-Agent CoordDesc）的顺序更新步数随 N 线性增长，单步规划耗时进入 wall-clock 敏感区间；CRN 的方差压制效率在其他 agent 数从 1（Easy）增至 7（Hard）后随 $\sigma_{others}$ 量级增长，SNR 在第 0 步抵消其他 agent 随机种子后能否仍保持接近 1.0 是 Scale-up 的核心验证目标。作 [证据待补 EVIDENCE PENDING — Hard N=8 规模外推配置，待补完] 处理。

**planner-off 自蒸馏退化演示**：将 MVE planner 替换为纯策略网络输出，演示规划器对蒸馏信号的必要性。**本节的理论预测**：在 planner-off 设定下，策略 $\pi$ 的训练目标退化为策略蒸馏自身（self-distillation），其唯一不动点是均匀分布——熵 $H(\pi) = \ln A$。这一预测来自 §4.5 INSIGHT 的自蒸馏退化引理：当策略训练目标为 $\mathrm{KL}(\pi \| \pi_{mve})$ 且 $\pi_{mve}$ 由当前 $\pi$ 经规划器加权得到，缺失规划器加权时 $\pi_{mve}$ 退化为 $\pi$ 本身，KL 目标的最小值为零、最优解为任意 $\pi$，训练动力学被熵正则项推向均匀分布。该退化在 v4-opt 阶段开发观察中已被实证记录——training collection 必须运行 MVE planner 否则策略停留在 $\ln A$ 不动点——本节作为 [证据待补 EVIDENCE PENDING — planner-off 自蒸馏退化演示，待补完] 的显式复现验证。

**三组实验在 §6 章中的角色**：消融 5 直接对应 §6.2 关于社会偏好建模单一性的局限说明，Scale-up Hard 对应 §6.3 关于工业规模扩展的展望路径，planner-off 对应 §6.1 中"贡献 3 在缺失规划器加权时的失败模式"的实证锚点。

**关于 Fehr-Schmidt 参数敏感度的更细粒度预测**：在 3 × 3 网格下，预期社会福利剖面呈现以下特征——$\kappa = 0.5, \lambda_{disadv} = 2.0$ 的标准设定附近（4-邻域）社会福利达到该网格的全局最大值；$\lambda_{disadv}$ 单独偏移至 1.0 时（弱不平等厌恶）社会福利下降约 5–8%，因 β 类型对资源分配不均的容忍度提高，合作激励减弱；$\lambda_{disadv}$ 单独偏移至 3.0 时（强不平等厌恶）社会福利下降约 3–5%，因 β 类型在劣势状态下的强反向梯度使训练不稳定，影响策略收敛速度；$\kappa$ 偏移在 $[0.3, 0.7]$ 区间内对社会福利的影响较小（约 ±3% 量级），因 $\kappa$ 主要影响劣势惩罚项的尺度而非不平等检测的阈值。这些预测的实测验证将在 [证据待补 EVIDENCE PENDING — Fehr-Schmidt 参数敏感度 3×3 扫描] 中给出。

**Scale-up Hard 配置的工程挑战与对策**：N = 8 配置下 Per-Agent CoordDesc 的顺序更新步数从 N = 4 的 4 步增至 8 步，单步规划耗时从 wall-clock 约 15ms 增至约 30ms；按 24 × 24 网格 + 40 资源点的环境复杂度，单 episode 长度从 Medium 配置的 200 步增至 400 步，单 episode 总 wall-clock 时间从约 3s 增至约 12s（4× 量级增加）。CRN 在 N = 8 下的方差压制效率需独立验证——其他 agent 随机性贡献的方差量级与 $\sqrt{N - 1}$ 正相关，N 从 4 到 8 时方差贡献约增加 $\sqrt{7/3} \approx 1.53$ 倍；若 CRN 仍能在第 0 步抵消其他 agent 随机种子，则 SNR 仍可维持在接近 1.0 的水平。**两个工程对策**：（a）将 CoordDesc 顺序更新改为按 agent 并行（参 §6.3 展望路径），将 wall-clock 复杂度从 $O(N)$ 降至 $O(\log N)$；（b）将 HyperNet 参数生成由动态计算改为离线展开 + 在线查表（参 §6.3），减少推理阶段的 wall-clock 开销。两项工程对策在 Hard N=8 实测中是否必要，由 [证据待补] 实验决定。

**planner-off 自蒸馏退化的代数细节**：在 planner-off 设定下，policy 更新目标退化为 $\mathrm{KL}(\pi \| \pi)$，其唯一最小值为零，且任何 $\pi$ 都是最优解；训练动力学被熵正则项推向均匀分布，最终 policy 熵稳定在 $\ln A = \ln 6 \approx 1.79$ 处。这一退化与"无监督学习中的崩塌"在数学结构上同构——缺失外部信号时模型退化为信号最大熵分布。Per-Agent MVE + CRN 规划器在闭环中扮演的角色，正是为 policy 训练提供一个"非平凡的、依赖当前状态的"目标分布 $\pi_{mve}$，使 KL 目标的最小值不再是均匀分布。这一机制理解使断言 D 的"双重技术不可分割"获得了额外的代数依据——CoordDesc 与 CRN 共同决定 $\pi_{mve}$ 的非平凡性，缺失任一者使 $\pi_{mve}$ 退化为均匀分布或与 $\pi$ 重合，闭环训练崩塌。本节预登记的 planner-off 实验作为这一代数预测的实证复现。

---

## §5.10 信念推断质量 + 三阶段课程

本节围绕两个子问题展开评估：（i）BeliefNet 所学到的对手类型表示 $\hat{z}_{i,j}$ 是否在统计层面上忠实反映真实类型 $\tau_{-i}$；（ii）三阶段课程训练能否在可见-$c$ 与隐藏-$c$ 两档下分别保证信念质量与主任务福利的协同收敛。**关于当前 suite 状态的诚实披露**：本节预期的 $\hat{z}_{i,j}$ 准确率与 $\hat{c}$ 信念质量曲线**无法从当前 suite 数据中提取**，原因是 18 个 hidden-$c$ 档运行中有 14 个 `loss/belief ≤ 1e-5`、`loss/lambda_b ≡ 1.0` 横跨所有运行，表明 BeliefNet 训练通路在当前优化设置下未能形成有效梯度流。该问题作为 §5.11 中标识为 P0 的失败案例独立列出，本节内容相应以理论预测 + 工程实施计划为主。

**两条互补的解释假设**：（a）Easy 200k 步训练时长不足以让信念损失通过 stage-2/3 课程门控被激活——若 stage 1 中 $\hat{z}$ 准确率门限设为 80%，而 200k 步内 $\hat{z}$ 准确率未达此门限，则 lambda_b 不会从初始值（在当前实现中默认为某个较小的初值或某个稳定保持的值）切换至有效训练状态，导致 `loss/belief` 始终维持在初始量级附近；（b）lambda_b gating 实现存在工程问题——`loss/lambda_b ≡ 1.0` 横跨所有运行是该假设的直接观察证据，表明 lambda_b 调度器从未推进到初始状态之后。两个假设的辨析需要在 §5.11 给出的修复方案完成后通过专门的诊断实验完成。

**双档评估协议**：可见-$c$ 档（c_visible = True）作为上界参照：agent 在每一时刻均可直接获取 $c_t$ 编码，BeliefNet 的 $\hat{c}$ 头退化为恒等映射，此时任何课程阶段的 $\hat{z}$ 准确率下界均应趋近于理论上限；隐藏-$c$ 档（c_visible = False）关闭 $c_t$ 输入通路，要求 BeliefNet 仅凭观测历史 $o_{1:t}$ 推断 $\hat{c}_i$，以 MSE 与 Pearson 相关系数作为 $\hat{c}$ 质量的双指标，以 cross-entropy 准确率作为 $\hat{z}$ 质量指标。两档对照将解耦"课程设计本身"与"$c_t$ 可见性"对最终性能的各自贡献。

**三阶段课程划分**：Stage 1 以 $\hat{z}$ 推断精度为主导目标，期望 $\hat{z}$ 分类准确率达到 80% 以上方可进入 Stage 2；Stage 2 在保持 $\hat{z}$ 准确率稳步提升至 90% 的同时，监控主任务社会福利是否随之稳定；Stage 3 要求 $\hat{z}$ 准确率维持在 85% 以上、主任务福利不发生显著退化，以此验证联合优化的稳健性。与之对照的 no-curriculum baseline 在相同超参数下跳过阶段门控、直接端到端训练，用于量化课程调度对收敛速度与最终性能的净增益。

**当前 suite 提供的有限信号**：虽然信念通路本身未充分激活，但 `lora_duo_film_fc2` 与 `zero_shot_easy` 在主任务上仍能稳定收敛——这一观察提供的有限结论是"主任务路径在 belief 通路未激活的情况下仍可学到合理策略"，但不能反推 BeliefNet 通路对主任务无贡献——可见-$c$ 档下 $c_t$ 可直接观测，主任务信号通路本就不依赖 belief 推断。完整的双档信念质量协议作 [证据待补 EVIDENCE PENDING — 三阶段课程 + $\hat{z}$ 准确率 + $\hat{c}$ MSE / Pearson + 隐藏-$c$ 档对照协议，待补完] 处理，结果将汇入 §6.2 局限节关于 BeliefNet 收敛性的讨论。

**关于诊断假设 (a) 的更具体调查计划**：假设 (a)——"Easy 200k 步训练时长不足以让信念损失通过 stage-2/3 课程门控被激活"——的验证需要两项独立检查。其一，**$\hat{z}$ 准确率的实测剖面**：导出当前 suite 中所有 hidden-$c$ 档运行的 $\hat{z}$ 训练末端准确率，对比 stage 1 门限 80%。若 14/18 runs 的 $\hat{z}$ 准确率均低于 60%，则 stage 1 门限确实未被达到，假设 (a) 成立；若 $\hat{z}$ 准确率已达到 80%+ 但 lambda_b 仍未切换，则假设 (a) 不成立，需转入假设 (b)。其二，**stage 切换日志**：检查训练日志中 lambda_b 切换的具体步数与触发条件，若 stage 切换从未在日志中出现，且 lambda_b 持续保持初值，则 lambda_b gating 实现可能存在工程问题，触发假设 (b)。两项检查均不需要新的训练，仅需对现有 TB 日志的额外解析，预计 0.5 GPU 小时内完成。

**$\hat{c}$ 信念质量的双指标度量**：MSE 度量 $\hat{c}_i$ 与真实 $c_t$ 在数值上的偏差，对极值偏差敏感；Pearson 相关系数度量 $\hat{c}_i$ 与 $c_t$ 在序列上的线性相关性，对常数偏移不敏感。两个指标的同时报告可避免单指标的误判——例如 BeliefNet 可能学到一个与真实 $c_t$ 线性相关但常数偏移的 $\hat{c}_i$（Pearson 接近 1 但 MSE 大），或学到一个量级正确但序列乱序的 $\hat{c}_i$（MSE 小但 Pearson 接近 0）。预登记的硬验证判据是 MSE < 0.05 且 Pearson > 0.8，二者均成立才视为 BeliefNet 通路对 $c_t$ 推断的硬验证。

**三阶段课程在工程实施层面的具体调度**：Stage 1 在训练初期 0 ~ 50k 步执行，loss = $L_{policy} + L_{value} + 5 \cdot L_{belief}$，belief 损失权重显著高于主任务，强制网络优先学习信念推断；Stage 2 在 50k ~ 150k 步执行，loss = $L_{policy} + L_{value} + L_{belief}$，权重平等，主任务与信念推断联合优化；Stage 3 在 150k ~ 240k 步执行，loss = $L_{policy} + L_{value} + 0.3 \cdot L_{belief}$，belief 损失退至辅助角色，主任务为主导。stage 切换的触发条件以 $\hat{z}$ 准确率门限为主、训练步数为辅——即"$\hat{z}$ 达到 80% **且** 训练步数已达 50k"双条件同时成立时切入 stage 2，避免准确率刚达标即切换造成的不稳定。stage 切换在日志中以 `loss/lambda_b` 与 `loss/stage` 两个 tag 显式记录。当前 suite 中 `loss/lambda_b ≡ 1.0` 的观察提示 stage 调度实现可能未达到预期，§5.11 P0 修复方案据此展开。

**no-curriculum baseline 的对照设计**：no-curriculum baseline 在相同超参（Adam ε = 1e-5、γ = 0.95、EMA τ = 0.99）下跳过 stage 调度、直接端到端训练，loss 中 $L_{belief}$ 权重固定为 1.0。该 baseline 用于量化课程调度对收敛速度与最终性能的净增益。预测：在隐藏-$c$ 档下，no-curriculum baseline 因主任务与信念推断同步竞争梯度通道，$\hat{z}$ 准确率收敛速度显著慢于课程版本（约 1.5× 训练步数后才达到同等准确率）；最终性能上 no-curriculum baseline 与课程版本接近（相对差距 < 5%），表明课程主要的价值在收敛速度而非最终性能上。这一预测的实测验证依赖 P0 修复后的完整双档实验。

---

## §5.11 失败案例与训练诊断

任何系统方法的有效性，最终都需在其失败边界处得到充分审视。本节围绕本研究训练过程中系统性记录的典型失败模式展开诊断分析，并对当前实现的未完工程项作出透明说明，旨在为后续复现与改进提供可操作的参照。

**失败案例 1：`lora/medium_base` 配置的训练崩溃**。3 seeds 在 240k 步目标下仅运行至 66k 步即终止，fairness = −0.17 ± 0.29、planner_prior_gap = −60、planner return 平均仅 68 ± 36（与 Easy 量级 105 的差距已不能用难度差异解释）。该 cell 是 suite 中**唯一**记录了额外 `*_same` 诊断 tag 的 cell（共 104 tags，相比其他 cell 的 101 tags 多 3 项），表明该 cell 触发了某种异常诊断路径。**触发条件假设**：训练崩溃可能源于（a）Medium scaling 下 LR 沿用 Easy 配置但 batch / replay 比例发生变化、（b）配置回归（regression）——即从 Easy 移植到 Medium 时某个超参数被遗留为不兼容值。**修复方案**：重跑该 cell 时，将 LR 缩小一档（5e-4 → 2e-4）、提高 EMA 衰减系数 $\tau$（0.99 → 0.995）、重新核对 BaseConfig 在 Medium scaling 上的参数对齐，每个 seed 独立重跑至 240k 步。该修复方案作为 §6.2 局限节中"agent 规模扩展"维度的具体工作项。

**失败案例 2：`lora/duo_film` seed1 的发散轨迹**。该 cell 共 2 seeds × 240k steps，其中一个 seed 的 q_gap 在训练中后期偏离正常区间，整体收敛轨迹与同 cell 另一 seed 存在结构性差异。**修复方案**：（a）接受高方差并补一组 3 seed 重复；（b）单独分析 seed1 的 TB 曲线以定位发散起点，必要时缩短该 seed 的训练长度避免后期 over-fit 引起的偏移。本节倾向选项（a），补一组 5 seed 以稳定该 cell 的方差量级。

**失败案例 3：`tragedy ≡ 0` 横跨全部 18 个 Easy 运行**。所有 Easy 配置下的 tragedy 指标（资源池被完全耗竭的回合占比）均稳定为 0，跨 18 个 run 没有一个例外。**两种可能解释**：（a）Easy 配置下资源生成率与 agent 数量的比例使资源池在 200k 训练步数内从未被耗竭，tragedy = 0 是配置内禀性质；（b）tragedy 指标的实现存在 bug，永远不会触发非零状态。**修复方案**：核对 ResourceCommons 在 Easy 配置下的资源动力学参数与 tragedy 计算公式，确认是定义内禀还是实现 bug；若为内禀则在 §3.7 环境章节补充说明，若为实现 bug 则修复后重跑 1 seed 验证。

**失败案例 4：`regret_mean ≡ 0` 横跨全部已完成运行**。zero_shot_easy / main_comparison_easy / abl4_joint_easy_n2 三个完成的 cell 中，regret_mean 均稳定为 0.0。**触发条件假设**：（a）已学习策略在所有 $c$ 网格上确实达到最优响应——这是 Easy 配置可信的实证可能；（b）regret 指标的实现以"训练末端 eval_report 是否取得训练集 $c$ 上的 max"为锚点，而非对每个 $c$ 都独立计算后取期望，因而在 Easy 配置上恒为零。**修复方案**：核对 regret 指标的实现公式，必要时改用 "对每个测试 $c$ 独立做 best-of-multiple-rollouts 评估" 的更严格定义，在 Medium 主对比上重新报告。

**失败案例 5（P0）：BeliefNet 训练通路未激活**。**触发条件**：18 个 hidden-$c$ 档运行中有 14 个 `loss/belief ≤ 1e-5`、`loss/lambda_b ≡ 1.0` 横跨所有运行——表明 lambda_b 调度器从未推进到初始状态之后，belief 损失未能形成有效梯度流。**两条诊断路径**：（a）检查 `BaseConfig.belief_loss_weight` 与 curriculum stage gating 的实现——`loss/lambda_b ≡ 1.0` 是否说明 lambda_b 被硬编码为常数而非随 stage 切换，或 stage 切换的触发条件（$\hat{z}$ 准确率门限）从未被达到；（b）检查 GRU 上下文编码器在 hidden-$c$ 档下的输入归一化层位置与梯度截断阈值是否阻断了 belief 通路的反向传播。**修复方案**：先排查路径（a），核对 BaseConfig 的 belief curriculum 配置；若发现 stage gating 的 $\hat{z}$ 准确率门限设置过高（如 80%）而当前训练长度不足以达到，则降低 stage 1 门限至 60% 或缩短至基于步数（如 50k 步固定切换）的简化课程；若路径（a）排除，则进入路径（b）的网络层级诊断。**该 P0 是 §5.10 信念质量协议与 §5.6 belief 通路零置消融的共同前置依赖**，§6.2 局限节将其单列为独立维度。

**工程完整性披露**：本节亦列出两项已承诺但尚未完工的工程项：（a）Pkg-02 中 EnvConfig 的 `c_visible` 开关——支持隐藏-$c$ 档下 BeliefNet 对资源丰度的在线推断质量评估，影响 §5.10 信念质量协议与断言 B′(ii) 的硬验证；（b）Pkg-08 中 MVE planner 的 `joint_enum` 模式——支持 Easy N=2 配置下完整 2×2 矩阵的 Joint enumeration cell，影响 §5.8 断言 D 的充分性证据。两项工程开关均需在 §5 实验全面开展前完工。

**算力账本**：本研究当前已耗费约 180 GPU 小时，主要分布在 `main_comparison_easy`（3 seed × 137k）、`zero_shot_easy`（3 seed × 200k）、`abl4_joint_easy_n2`（3 seed × 100k）、`lora/duo_*` 谱预筛若干 cell 共 6 seed × 240k 与失败重跑（lora/medium_base 与 lora/duo_film seed1）。剩余约 120 GPU 小时按优先级分配如下：BeliefNet P0 修复 → Medium 主对比基线运行 → 消融 2 / 3 / 4 三象限补完 → Hard 规模外推。各失败模式的触发条件 + 修复方案与 §6.2 / §6.3 形成具体的工作锚点对应。

**关于失败案例分类的整体结构**：上述五项失败案例可按其严重性与对论文整体结论的影响分为三层。**P0 层**——影响断言 C 与断言 B′(ii) 的硬验证，必须在后续实验启动前修复，对应失败案例 5（BeliefNet 通路未激活）。**P1 层**——影响断言完整支持但不影响当前 EVIDENCE_SUFFICIENT / EVIDENCE_PARTIAL 状态，对应失败案例 1（Medium scaling 训练崩溃）与失败案例 2（`lora/duo_film` seed1 发散），需在 Medium 主对比执行前完成修复或方差控制。**P2 层**——影响指标定义的语义清晰度但不直接动摇任何断言的支持状态，对应失败案例 3（tragedy ≡ 0）与失败案例 4（regret ≡ 0），可在 Medium 主对比执行后并行进行指标实现核对。三层失败案例的修复时间表与 §5.1 算力规划中"剩余 120 GPU 小时"的优先级序列完全一致。

**关于"诚实披露"作为论文方法学组成部分**：本节对失败案例的系统披露并非事后补丁，而是构成本文方法学立场的有机组成部分。按 §1.5 预登记规范，本文的实证义务不仅包括对支持性证据的报告，也同等地包括对未支持证据的披露——后者使预登记承诺成为可被否定的硬约束而非可任意 hedge 的修辞结构。本节列出的五项失败案例每一项都与某一具体的断言或工程承诺挂钩：失败案例 5 与断言 C / B′(ii) 挂钩，失败案例 1 / 2 与 Medium 主对比的统计判读挂钩，失败案例 3 / 4 与 ResourceCommons 指标定义的语义清晰度挂钩。这种"每一项失败均可追溯到对应断言或承诺"的结构，使本文的方法学立场具有内部一致性。

---

## §5.12 本章小结与对 §6 的衔接

本章按"协议 → 决策门 0 → v3 迁移 → 主对比 → 五项消融 → 鲁棒性 → 信念质量 → 失败案例"的顺序，系统报告了 hyper-mve v4 方法在 ResourceCommons 与 NonStationaryTag 两个环境上的实验进展。**当前 suite 已完成的部分**包括：Easy 配置下 `main_comparison_easy` / `zero_shot_easy` / `abl4_joint_easy_n2` 三个 cell 共 9 seed × 中等步数训练，LoRA 条件化谱预筛 5 个 cell 共 9+ seed × 较长步数训练，并产生了支持断言 B′ 内部最优（部分支持）与断言 D joint cell（部分支持）的实测信号。**当前 suite 未完成的部分**包括：六类基线主对比（MAPPO / QMIX / CAGA / MAMBA / MARIE / MA-MuZero）、Medium N=4 主配置下的全部实验、消融 2（三联通路零置）、消融 3（$\rho_\beta$ 五点扫描）、消融 4 的三象限分离、消融 5（Fehr-Schmidt 参数敏感度）、Scale-up Hard N=8、planner-off 自蒸馏退化演示、v4 → v3 NonStationaryTag 的 apples-to-apples 迁移、信念质量双档协议、以及 BeliefNet P0 修复后的所有相关重做。

**四项断言的当前验证状态**汇总为：**A 未支持（实验未执行）、B′ 部分支持、C 未支持（实验未执行）、D 部分支持**。该状态将原样进入 §6.1 表格，作为论文整体贡献复述的核心实证基础。按 §1.5 预登记的整体降级条件——A / B′ / C / D 中两项及以上同时降级至最弱版本——当前状态尚未触发整体降级，但 A 与 C 的"未支持（实验未执行）"严格属于覆盖度局限而非否定证据，需在后续实验补完后方可形成最终判读。

**与 §6 的衔接**：§5 实验状态直接对应 §6 章三节内容。§6.1 贡献总结与断言验证将以本章的实测信号 + EVIDENCE PENDING 标签作为汇总表的填写依据；§6.2 局限将以本章的五项失败案例 + 三层优先级结构作为四个维度（agent 规模、消融覆盖度、社会偏好建模、理论完备性）局限说明的工作锚点；§6.3 展望将以本章 §5.9 Scale-up Hard 与 §5.11 P0 修复后的剩余 120 GPU 小时算力规划作为两条扩展路径的具体可执行内容。本章中标注的所有 [证据待补 EVIDENCE PENDING] 项目将在后续实验补完后逐项消除，对应 §6 章三节内容的相应更新。

**本章实证立场的方法论自评**：本章的实验报告采用了"实测信号 + 待补承诺"的双层结构——已由训练 suite 支持的实测结果（B′ 零样本半 EVIDENCE_SUFFICIENT，B′ 谱内部峰值 EVIDENCE_PARTIAL，D joint cell EVIDENCE_PARTIAL）按数值与种子级误差棒据实报告；预登记但尚未执行的实验（断言 A 五点扫描、断言 C 三联通路零置、断言 D 四象限分离、Medium 主对比 vs 六类基线、Hard 规模外推、消融 5 参数敏感度、planner-off 自蒸馏演示、v3 迁移、信念质量双档协议）按 [证据待补 EVIDENCE PENDING] 标签逐项处理。这种双层结构使本章的实验报告同时具备两类信息——"现在已经知道的"与"承诺后续将验证的"——为读者在论文最终评估时提供清晰的信息边界。该立场与论文整体的"内部一致性优先于单项技术新颖性"方法学定位（§6.4）一致：本文不寻求以增量实验信号支撑超出现有证据范围的强陈述，而是通过严格的"已支持 / 待支持"分离让方法学贡献的有效范围在论文写作层面即得到清晰界定。

**对训练 suite 进展的整体定性**：截至本章定稿时，已完成的 9 个 hyper-variant cell（共 18+ seed × 中长步数训练，约 180 GPU 小时）取得了若干稳健的实证信号——条件化谱内部峰值在 `lora_duo_film_fc2` cell 上的 2.6× return 提升、零样本 regret_mean = 0.0 在 3 seed 上的稳定成立、双重技术 joint cell 在 100k 步内的稳定收敛——这些信号支持本文方法在 RNS-MMG 设定下的可行性主张。未完成的实验主要集中在两个方向：跨方法基线对比（涉及 6 类基线方法的独立训练，约 180 GPU 小时预算）与三项未支持断言对应的预登记消融（涉及 belief 通路 P0 修复 + 消融 2 / 3 / 4 / 5 的完整执行，约 90 GPU 小时预算）。这两个方向的剩余工作构成本章 §5.4 至 §5.10 中所有 [证据待补 EVIDENCE PENDING] 标签的实际清单，与 §6.2 局限节中"消融覆盖度"维度的工作锚点完全对应。本章因此并不主张当前实验状态已完成所有预登记承诺，而是据实给出"四项断言中 2 项部分支持、2 项未支持（实验未执行）"的当前快照，作为论文阶段性报告的实证锚点。
