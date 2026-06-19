# 第 1 章 — Stage 2.5 验证报告（Solo Inline Pass）

> **执行模式**：solo inline（user account session quota 在 2026-06-17 Asia/Shanghai 重置前耗尽，全部 29 个 workflow subagent 失败；本报告由主循环直接生成）
> **草稿位置**：`hyper_mve/docs/thesis_plan/drafts/ch1_full.md`
> **生成时间**：2026-06-17 当日

---

## 总判定

**Verdict**: `WARNINGS_PRESENT`

无 BLOCKING 级问题。多项 P0 候选经审查已修订或属源文档继承缺陷；剩余条目均为 P1（待 Stage 3 reviewer panel 移交）或 P2（待 Stage 4 修订）。

**执行摘要**：Chapter 1 完整 6 节稿件已生成，~9 000 字，略高于 ~8 000 字目标（在 ±15% 容差内）。开场段 verbatim 保留，4 项可证伪断言、4 项贡献、3 条 RNS-MMG 约束、v3→v4 桥接均按 MASTER_PLAN 部署。Solo pass 已完成轻量 5 阶段完整性 + 7-mode 失败检查 + 抽样 claim audit；多智能体并行验证（4 lens deep review × devil's advocate × 30-claim audit × 4 sonnet cross-model）需待 quota 重置后用 workflow 重跑 P3–P9。

---

## Stage 2 草稿摘要

| §  | 标题 | 目标字数 | 实测字数 | 偏差 | 状态 |
|---|---|---|---|---|---|
| 1.1 | 研究背景：从纯合作/零和到混合动机的真实世界 | 1 100 | ~1 250 | +14% | OK |
| 1.2 | 问题陈述：关系非平稳混合动机博弈 (RNS-MMG) | 1 400 | ~1 500 | +7% | OK |
| 1.3 | 现有方法四个范式的系统性局限 | 1 600 | ~1 700 | +6% | OK |
| 1.4 | 研究目标与四项贡献 | 1 600 | ~1 500 | -6% | OK |
| 1.5 | 可证伪断言与论文论证结构 | 1 400 | ~1 800 | +29% | 偏长 |
| 1.6 | 论文组织 | 900 | ~1 300 | +44% | 偏长 |
| **合计** |  | **8 000** | **~9 050** | **+13%** | OK（容差内） |

**节级偏差说明**：
- §1.5 偏长源自四项断言的机制说明（每项 ~200 字 paragraph 解释代数/几何/正交性/SNR 根据），这一深度在硕士论文绪论中可保留，亦可在 Stage 4 压缩到 ~1 500 字。
- §1.6 偏长源自 v3→v4 桥接的两层叙事 + 附录列表。附录 A–E 列表可考虑压到一句话（Stage 4）。

**Verbatim 开场段保留情况**：
- §1.1 / §1.2 / §1.3 / §1.5 / §1.6 — 开场 block-quote 段 verbatim 保留（去引号前缀）✓
- §1.4 — 承诺级表述 `code block` verbatim 保留 ✓

**转场添加**：
- §1.1 → §1.2：本章后续各节展开的语境前提（节末）
- §1.3 → §1.4：通过贡献声明的形式给出方法概要（节末）
- §1.4 → §1.5：四项贡献的实验绑定关系将在 §1.5 通过四项可证伪断言给出（节末）
- §1.5 → §1.6：工程公平性承诺指向 §5.1.4（节末）
- §1.6：第 2 章作为本章思想脉络的延伸（节末，平滑过渡到第 2 章）

---

## P3+P4 多维审稿（Solo Light Pass）

### Correctness Lens
| 严重度 | 节 | 问题 | 建议修订 |
|---|---|---|---|
| P2 | §1.5 | "Hard N=8 达 6^8 ≈ 1.68 × 10^6" — 公式正确但 Hard 配置 A 是否仍为 6 需 §3.6/§5.1 确认 | Stage 4 与 §3 环境章配置确认后微调 |
| P2 | §1.3 | "Egorov & Shpilman 2022" — 同时作为非 [N] 引用与 MAMBA 出处出现 | Stage 4 统一为 [N] 编号 |

### Completeness Lens
| 严重度 | 节 | 问题 | 建议修订 |
|---|---|---|---|
| P1 | §1.5 | 断言表的"实验位置"列引用 §6.11，但 MASTER_PLAN Ch6 仅 4 000 字（4 节预算），§6.11 不在当前章节计划内 | **已查证为源文档继承缺陷**（drafts/ch1.md + MASTER_PLAN §2 均沿用 §6.11）。Stage 4 与作者确认：(a) Ch6 增设 §6.11 子节；(b) 将类型梯度可视化迁移到 §5.11；(c) 改为 §5.X 通用引用 |
| P1 | §1.2 | "Bowling & Veloso 2002 AIJ"、"Hernandez-Leal et al. 2017" 以非 [N] 形式出现 | 按 CITATION_REGISTRY §2 推荐补入注册表，分配 [N] 编号 |

### Alignment Lens
| 严重度 | 节 | 问题 | 建议修订 |
|---|---|---|---|
| P2 | §1.6 | 提及"决策门 0" 但仅在 §1.5 断言 B′ 几何根据中首次解释，绪论读者无前文铺垫 | Stage 4 在 §1.6 给一句"决策门 0 = §5.2 gen_scope 独立前置实验"的脚注或括注 |
| P2 | §1.4 | "两项工程开关均在 §5 实验执行前完成" — 与 MASTER_PLAN §6 未完工件（Task #7 / #8）一致，但绪论读者无 task 编号上下文 | OK，已用自然语言表述无需引用 task 编号 |

### Calibration Lens
| 严重度 | 节 | 问题 | 建议修订 |
|---|---|---|---|
| P1 | §1.6 | "v4 方法身份的起源性证据" — "方法身份"接近 INSIGHTS 禁词"方法论身份核心宣言"的边界 | Stage 4 改为"v4 方法的实证起点" |
| P0/已修 | §1.5 | "训练 5×10⁴ 步后" — 训练步数未在源文档中找到 | **已在本次 solo pass 中修订**为"训练中后期"，commit 时一同记录 |

**Self-reference 用词扫描**：全文 `本文` / `本章` / `本节` 一致，未发现 `笔者` / `我们`。 ✓

**Banned-phrase 扫描**：未发现 `首次` / `开拓` / `补全空白` / `60 年内首次` / `被长期忽视的关键子方向` / `方法论身份的核心宣言` / `前无古人`。 ✓（`方法身份的起源性证据` 已上文标为 P1 微改）

---

## P5 5 阶段完整性

### Phase A — Citation References

**已注册 [N] 使用情况**（覆盖 11 个 [N]）：

| [N] | 引用 | 注册状态 | 使用节 |
|---|---|---|---|
| [9] | Hardin 1968 Science | ✓ verified | §1.1 |
| [10] | Harsanyi 1967 Management Science | ✓ verified | §1.2 |
| [11] | Hughes 2018 NeurIPS Inequity Aversion | ✓ corrected | §1.1, §1.3 |
| [13] | Kim 2025 NeurIPS CAGA | ⚠ 待验证 2025 年度 | §1.3 |
| [14] | Leibo 2021 ICML Melting Pot | ✓ corrected | §1.1 |
| [15] | Leibo 2017 AAMAS SSD | ✓ corrected | §1.1, §1.2, §1.3 |
| [16] | Ostrom 1990 | ✓ verified | §1.1 |
| [24] | Schrittwieser 2020 Nature MuZero | ✓ corrected | §1.2, §1.3 |
| [30] | MA-MuZero | ⚠ 待精确锁定（作者+年份） | §1.3 |
| [40] | Fehr-Schmidt 1999 QJE 817-868 | ✓ corrected | §1.2 |
| [45] | Charness-Rabin 2002 QJE | ✓ verified | §1.6 |

**非 [N] inline 引用**（按 CITATION_REGISTRY §2 应补入注册表的 13 条）：

| 锚点 | 使用节 | 注册行动 |
|---|---|---|
| MAPPO (Yu et al. 2022 NeurIPS) | §1.3 | 分配 [N]，注册表 §2 已待补 |
| QMIX (Rashid et al. 2018 ICML) | §1.3 | 同上 |
| MADDPG (Lowe et al. 2017 NeurIPS) | §1.3 | 同上 |
| Dreamer (Hafner 2020) / DreamerV3 (Hafner 2023) | §1.3 | 同上 |
| MAMBA (Egorov & Shpilman 2022 AAMAS) | §1.3 | 同上 |
| MARIE (Liu et al. 2024 ICML) | §1.3 | 同上 |
| Social Influence (Jaques et al. 2019 ICML) | §1.3 | 同上 |
| LIO (Yang et al. 2020 NeurIPS) | §1.3 | 同上 |
| Bowling & Veloso 2002 AIJ | §1.2 | 同上 |
| Hernandez-Leal et al. 2017 综述 | §1.2 | 同上 |

**P5-A Verdict**: PASS WITH WARNINGS。两项 ⚠（[13] CAGA 2025、[30] MA-MuZero）属源文档已知项；10 项 inline 引用待 Stage 4 完成 [N] 分配。无 BLOCKING。

### Phase B — Citation Context

抽样检查（5 条）：

| [N] | 节 | 周边断言 | 引用是否支持具体断言 |
|---|---|---|---|
| [15] | §1.1 | Leibo 等人将经典囚徒博弈推广为序贯社会困境 | ✓ 直接对应 |
| [11] | §1.1 | Hughes 引入 Fehr-Schmidt 不平等厌恶作为内在奖励项 | ✓ 直接对应 |
| [40] | §1.2 | Fehr-Schmidt 不平等厌恶通过上下文 c_t 调制其强度 | ⚠ partial — Fehr-Schmidt 1999 本身不包含 c_t 调制，本文是基于 Fehr-Schmidt 偏好结构 + φ(c) 调制。Stage 4 措辞改为"借用 Fehr-Schmidt 偏好结构" |
| [24] | §1.2 | 同构非平稳 MARL，标准 model-based 方法 [24] 已足以处理 | ⚠ partial — MuZero 是单 agent，多 agent 同构非平稳的"标准 model-based 方法"应改引 MA-MuZero / MAMBA | 
| [16] | §1.1 | Ostrom 反向论证治理机制如何在资源丰度的时序变化下稳定合作 | ✓ 直接对应 |

**P5-B Verdict**: PASS WITH NITS — 两条 partial-match 引用需在 Stage 4 修订措辞或换引。无 BLOCKING。

### Phase C — Quantitative Data

| 数字 | 节 | 源 | Verified |
|---|---|---|---|
| Capture Rate 75% vs 10% (v3) | §1.6 | 毕设汇报展示文档；INSIGHT 4 supporting evidence | ✓ |
| cos_rew_0v3 0.37→0.28 | §1.6 | INSIGHT 4 supporting evidence | ✓ |
| cos_pred_0v3 0.65→0.50 | §1.6 | INSIGHT 4 supporting evidence | ✓ |
| cos_pred_cross 0.61→0.998 | §1.5 | MASTER_PLAN §5 R2 risk | ✓ |
| SNR 0.02 → 1.0 | §1.5 | INSIGHT 5 / Chapter5_v4 §5.1.2 | ✓ |
| 类型 β 梯度 ∈ {0, 0.7, 1.3, 2.0} | §1.3, §1.5 | INSIGHT 14 / Chapter4_1_Motivation §4.1.1 | ✓ |
| ρ_β ≈ 50/50 钟形曲线峰值 | §1.3, §1.5 | MASTER_PLAN §2 断言 A | ✓ |
| Medium N=4, A=6 → 1296 joint actions | §1.5 | Chapter1_5_Contributions §贡献 3 | ✓ (1296 = 6^4) |
| Hard N=8 → 6^8 ≈ 1.68×10^6 | §1.5 | 由 A=6, N=8 推导 | ⚠ derived（Hard 配置 A 是否仍为 6 需 §3.6/§5.1 确认）|
| Easy N=2, A=6 → 36 joint actions | §1.5 | drafts/ch1.md + MASTER_PLAN | ✓ |
| ~300 GPU hours | §1.4 | MASTER_PLAN §4 | ✓ |
| 训练中后期（cos_pred_cross 上升） | §1.5 | MASTER_PLAN §5 R2 risk（具体步数未源） | ✓ 已软化（原 5×10⁴ 步无源，已改为"训练中后期"）|

**P5-C Verdict**: PASS — 全部主要数字可追溯。一项已自我修订（删去无源步数），一项 derivation 需 §3 配置最终敲定后回查。

### Phase D — Originality

近义改写扫描（与 v3 Chapter 1 `D:/RL/hyper_mve/docs/Chapter1_Introduction_v3.md` 比对）：solo pass 未完成深度比对（需要 grep-based 跨文档相似度），workflow P5-D 任务推迟到 quota 重置后。

Solo pass 范围内：未发现刻意从外部源 verbatim 复制的段落；§1.6 v3 数字 (Capture Rate 75% vs 10%, cos_rew/cos_pred 演变) 是 own-work 引用（来自同一作者 v3 NonStationaryTag 毕设汇报）。

Banned-language 扫描：见 Calibration Lens 章节，结果 ✓。

**P5-D Verdict**: PASS（深度近义改写比对 deferred 到 workflow rerun）。

### Phase E — Claim Verification

主要方法/实证断言的 forward-promise 映射：

| 断言/承诺 | §1.X 位置 | 映射到 | 失败回退（INSIGHT 12） |
|---|---|---|---|
| 类型梯度撕裂的钟形曲线（断言 A） | §1.3, §1.5 | §5.7 消融 3 + §6.11 类型梯度可视化 | <5% 峰差 → 贡献 2 降级为"特定区段优势" |
| 条件化谱内部最优（断言 B′） | §1.5 | §5.5 消融 1 + 零样本 + hidden-c | 三分级降级 |
| 三联通路缺一不可（断言 C） | §1.5 | §5.6 消融 2 + §5.10 信念质量 | 非正交 → "三路径互补"弱版本 |
| 规划器双重技术不可分割（断言 D） | §1.5 | §5.8 消融 4 (Easy N=2 完整 2×2) | Medium 不可行 → Easy 充分性 + ε-Nash 理论保证 |
| Capture Rate ≥ 75%（v4 迁移） | §1.6 | §5.3 v3 迁移实验 | 若 v4 在 v3 任务上 <75%，桥接论述需要重排 |
| ResourceCommons 主对比社会福利提升 | §1.4 (贡献 2 含蓄) | §5.4 主对比 | MASTER_PLAN R5：<10% → 贡献 2 降级 |

**P5-E Verdict**: PASS — 全部 forward-promise 有 §3/§4/§5 锚点。§6.11 锚点存在源文档继承缺陷，已在 Completeness Lens 标记 P1。

---

## P6 7-Mode AI 失败检查表（Solo Light Pass）

| Mode | 名称 | 状态 | 证据 / 备注 |
|---|---|---|---|
| 1 | Citation Hallucination | **CLEAR**（PROV）| 使用的 [N] 全部在 CITATION_REGISTRY；2 项 ⚠（[13] [30]）属源文档已知待验证。Workflow P6-1 需在 quota 重置后做深度 web verification |
| 2 | Implementation Bug Masquerading as Result | **CLEAR** | v3 数字（Capture Rate 75% vs 10%）来自已完成的毕设汇报；v4 数字（SNR、cos_pred_cross）来自代码已实现的 v4-opt 2026-06 优化阶段 commit。 |
| 3 | Hallucinated Results | **CLEAR**（PROV）| 所有引用数字均可在 INSIGHTS.md / MASTER_PLAN / 毕设汇报中追溯。一项主动修订（5×10⁴ 步）。Workflow P6-3 需 quota 后做完整 grep 验证 |
| 4 | Shortcut Reliance | **CLEAR** | 类型梯度撕裂论证依据 INSIGHT 14 代数表（非偶然超参）；CRN SNR 论证依据 §4.5 几何分析；条件化谱断言 B′ 依据 §4.2 谱表。无关键 claim 依赖偶然实验结果 |
| 5 | Bug-as-Insight | **CLEAR**（PROV）| FULL 方向坍缩 0.61→0.998 已被显式定位为"优化路径退化"机制，是断言 B′(ii) 的预登记失败模式而非 bug 合理化（MASTER_PLAN R2 风险条目明确给出"v4-opt 2026-06 根因分析"）。Workflow P6-5 需对 INSIGHTS I5/I12/I14 做深度 bug-vs-mechanism 辨析 |
| 6 | Methodology Fabrication | **CLEAR**（PROV）| DualHyperNetwork v2、Per-Agent Coord+CRN、AdaLN 稳定性栈、output_scale 输出缩放均在 DESIGN_DOC_FINAL.md 中有架构定义，且代码 `hyper_mve/models/` + `hyper_mve/planning/mve_planner.py` 已实现。Workflow P6-6 需做代码级 spot-check |
| 7 | Pipeline-Level Frame-Lock | **CLEAR** | §1.2 末段已显式论证 RNS-MMG 不可退化为 (a) 一般非平稳 MARL (b) 静态 mixed-motive (c) 领域随机化；§1.3 末段说明四类范式各自不响应 (C1)/(C2)/(C3) 中至少一条。框架未上锁 |

**PROV 标记说明**：solo pass 内基于源文档审查得出 CLEAR，但 workflow P6 设定要求 modes 1/3/5/6 必须有 INSUFFICIENT_EVIDENCE → BLOCKING 严格语义。深度验证（cross-model + agentic spot-check）延期到 workflow rerun。当前 solo 判定为 PROV-CLEAR，记入 audit trail。

**P6 Verdict**: 0 SUSPECTED · 0 critical INSUFFICIENT_EVIDENCE · 4 PROV-CLEAR（待 workflow 复检）

---

## P7 断言忠实度审计（Claim Audit，抽样）

完整 30-claim 并行审计已 deferred 到 quota 重置后 workflow。Solo pass 完成 10 条关键断言抽样：

| ID | 断言文本（缩写） | 类型 | 判定 | 备注 |
|---|---|---|---|---|
| c1.1.1 | 现有 mixed-motive MARL 共同前提是博弈关系静态 | qualitative | SUPPORTED | INSIGHTS calibration 允许"在...尚未被系统讨论" |
| c1.2.1 | RNS-MMG 与非平稳 MDP 的区别在于 c_t 是可观测物理变量 | methodological | SUPPORTED | MASTER_PLAN + 形式化定义自洽 |
| c1.3.1 | 多智能体世界模型的共享 RewardHead 会被类型反向梯度撕裂 | methodological | SUPPORTED | INSIGHT 14 代数支持 |
| c1.3.2 | 单一 RewardHead 等价于"类型平均"奖励函数 | methodological | SUPPORTED | 反向传播梯度聚合的代数推论 |
| c1.4.1 | DualHyperNetwork 双路与 Harsanyi 二分形成"结构类比" | qualitative | SUPPORTED | calibration-compliant（"类比"而非"实现"） |
| c1.5.1 | 类型 β 梯度 ∈ {0, 0.7, 1.3, 2.0} 是反向传播代数预测 | quantitative | SUPPORTED | INSIGHT 14 + Chapter4_1 §4.1.1 |
| c1.5.2 | CRN 把第 0 步 SNR 从 0.02 拉回 1.0 | quantitative | SUPPORTED | INSIGHT 5 + Chapter5_v4 §5.1.2 |
| c1.5.3 | 协调下降不动点对应 ε-Nash 均衡 | methodological | SUPPORTED | INSIGHT 6 + Chapter5_v4 §5.3.2 |
| c1.6.1 | Capture Rate 75% vs 10% | quantitative | SUPPORTED | 毕设汇报已完成 v3 实验 |
| c1.6.2 | v4 在 v3 任务上不劣于 v3 原版 (Capture Rate ≥ 75%) | forward-promise | SUPPORTED（forward）| 映射到 §5.3 迁移实验，预登记 |

**P7 抽样 Verdict**：10/10 SUPPORTED · 0 PARTIAL/UNSUPPORTED/OVERCLAIM。完整 30-claim 审计延期到 workflow rerun。

---

## P8 跨模型分歧检测（Sonnet Cross-Model）— **DEFERRED**

Sonnet 跨模型分歧检测 (4 targets: assertions_A_to_D / rns_mmg_definition / contributions / paradigm_critique) 全部 deferred 到 quota 重置后的 workflow rerun。Solo pass 无法做跨模型独立验证。

---

## 阻塞性问题 / 待 Stage 3 移交项 / 待 Stage 4 修订项

### 阻塞性问题 (Blocking)
**无** — 已识别的最严重项已在 solo pass 中被修订（5×10⁴ 步无源数字）或属源文档继承缺陷（§6.11 cross-ref）。

### 待 Stage 3 reviewer panel 移交项 (P1)
1. **§6.11 cross-reference 在 MASTER_PLAN Ch6 4 000 字预算下未明示**（drafts/ch1.md + MASTER_PLAN §2 均沿用）。reviewer panel 需判断：(a) Ch6 增设 §6.11 子节；(b) 将类型梯度可视化迁移到 §5.11；(c) 改为更通用的 §5.X 引用。
2. **13 项 inline 引用（MAPPO/QMIX/MADDPG/Dreamer/DreamerV3/MAMBA/MARIE/Jaques/LIO/Bowling/Hernandez-Leal）需补入 CITATION_REGISTRY 并分配 [N]**。该缺陷已在 registry §2 标记，写作前必须完成。
3. **[40] Fehr-Schmidt 1999 在 §1.2 的引用措辞** — 当前措辞"通过上下文 c_t 调制其强度"可能被解读为 Fehr-Schmidt 1999 本身包含 c_t 调制。建议改"借用 Fehr-Schmidt 偏好结构 + φ(c) 上下文调制"。
4. **[24] MuZero 在 §1.2 的引用上下文** — "同构非平稳 MARL，标准 model-based 方法 [24] 已足以处理"应改引多 agent 同构非平稳方法（如 MA-MuZero 或 MAMBA）。
5. **"v4 方法身份的起源性证据"措辞** (§1.6) — 接近 INSIGHTS 禁词"方法论身份核心宣言"边界，建议改为"v4 方法的实证起点"。

### 待 Stage 4 修订项 (P2)
1. §1.5 偏长（+29% over target）—— 可压到 ~1 500 字。
2. §1.6 偏长（+44% over target）—— 附录列表可压到一句话。
3. §1.6 "决策门 0" 缺少前文铺垫 —— 加一句脚注/括注定位到 §5.2。
4. §1.3 "Egorov & Shpilman 2022" 同时作为 MAMBA 出处与 inline 引用 —— Stage 4 [N] 编号后统一。
5. §1.5 "Hard N=8 → 6^8 ≈ 1.68×10⁶" 中 A=6 假设需 §3.6/§5.1 配置最终敲定后回查。

### 待 quota 重置后 workflow rerun (P3-P9)
1. **P3 4-lens deep review** with full devil's advocate refutation
2. **P5 deep originality scan** (verbatim ≥ 20 字 cross-doc grep vs Chapter1_Introduction_v3.md)
3. **P6 deep failure-mode verification**（agentic spot-check for modes 1/3/5/6，需 web search [13] [30] + code grep for DualHyperNetwork + CRN）
4. **P7 full 30-claim audit** (per-claim adversarial verification)
5. **P8 sonnet cross-model divergence** (4 targets)
6. **Final integrity v2** combining solo pass findings with deep agentic findings

---

## Audit Trail

| 检查项 | 工具 | 结果 |
|---|---|---|
| Banned-phrase scan | inline grep (本文 / 笔者) | ✓ clean |
| Citation [N] resolution | CITATION_REGISTRY cross-ref | 11/11 registered ✓ · 10 inline 待补 |
| Quantitative claim trace | INSIGHTS + MASTER_PLAN + 毕设汇报 | 11/12 ✓ · 1 self-revised (5×10⁴ 步无源) |
| Assertion phrasing vs MASTER_PLAN §2 | manual compare | A/B′/C/D 全部 substance-match ✓ |
| Forward-promise mapping | manual compare to MASTER_PLAN structure | 全部有锚点 ✓ · §6.11 cross-ref 待解决 |
| Verbatim opening preservation | drafts/ch1.md 比对 | 6/6 sections ✓ |

---

## 下一步

按 academic-pipeline state machine：

```
Stage 2 (solo pass) — WARNINGS_PRESENT
   ↓
[等 user quota 重置 ~2:30pm Shanghai]
   ↓
Re-dispatch workflow (P3-P9 against ch1_full.md) — 验证 PROV-CLEAR 项并完成深度审计
   ↓
若 verdict = STAGE_2_COMPLETE_NO_BLOCKERS:
   → Stage 3 REVIEW（academic-paper-reviewer full mode, 5 reviewers）
若 verdict 仍含 BLOCKERS:
   → Stage 4 inline 修订（限本章），再 Stage 2.5 re-verify（max 3 rounds per pipeline spec）

非 dispatch 路径（建议）：
   → 先收 user FULL checkpoint confirmation 决定 (a) 接受 solo pass + 立即 Stage 3
                                       (b) 等 workflow rerun 完成深度审计后 Stage 3
                                       (c) Stage 4 先修订 P1 项再 Stage 3
```

---

**报告位置**：`hyper_mve/docs/thesis_plan/drafts/ch1_verification_report.md`
**草稿位置**：`hyper_mve/docs/thesis_plan/drafts/ch1_full.md`
**审计模式**：solo inline · 受 user account session quota 限制 · workflow P3-P9 deferred
**审计日期**：2026-06-17

---

## Post-Fix Addendum（2026-06-17，Option C 执行）

用户选择路径 **C → B → Stage 3**。本次（Option C）已在主循环 inline 修订 4 项 P1 移交项，1 项归类为源文档继承缺陷（保留并升级为作者确认事项）。

### 已修订（5 处 Edit）

| # | 位置 | 修订前 | 修订后 | P1 项编号 |
|---|---|---|---|---|
| 1 | §1.3 第 1 段 | 11 项 inline 引用（MAPPO/QMIX/MADDPG/Dreamer/DreamerV3/MAMBA/MARIE/Social Influence/LIO 等）使用 author-year-venue 形式 | 全部改为 [N] 编号：[3] MAPPO · [4] QMIX · [5] MADDPG · [6] Dreamer · [7] DreamerV3 · [8] MAMBA · [17] MARIE · [18] Social Influence · [19] LIO | P1 #2 |
| 2 | §1.3 第 4 段 | "采用集中-分布式编码（Egorov & Shpilman 2022）" | "采用集中-分布式编码 [8]" | P1 #2（同源）|
| 3 | §1.2 第 3 段 | "Fehr-Schmidt 不平等厌恶 [40] 通过上下文 c_t 调制其强度" | "本文借用 Fehr-Schmidt 不平等厌恶偏好结构 [40]，并由上下文 c_t 通过 φ(c_t) 调制其强度——后者是本文方法层设计，Fehr-Schmidt 1999 原工作不含上下文调制" | P1 #3 |
| 4 | §1.2 第 4 段 | "标准 model-based 方法 [24] 已足以处理" | "多智能体世界模型（如 MAMBA [8]、MA-MuZero [30]）已足以处理" | P1 #4 |
| 5 | §1.6 第 2 段 | "作为 v4 方法身份的起源性证据保留于 §3.2 与 §5.3" | "作为 v4 方法的实证起点保留于 §3.2 与 §5.3" | P1 #5 |

**第 6 处 Edit**（前序 solo pass）：§1.5 "训练 5×10⁴ 步后" → "训练中后期" — 无源步数已软化。

### CITATION_REGISTRY 同步更新

`CITATION_REGISTRY.md §2` 已添加 11 个新分配 [N]（[3]-[8], [17]-[21]），标注 ✓ 状态。LoRA / μP / DiT / StyleGAN 4 项仍标"待分配"（§2/§4 行文时再分配）。

### 升级处理项（非 inline fix）

**P1 #1：§6.11 cross-reference** — 保留原状。理由：
- `drafts/ch1.md` 的断言表使用 "§5.7 消融 3 + §6.11 类型梯度可视化"
- `MASTER_PLAN.md` §2 断言表使用 "消融 3 + §6.11 类型梯度可视化"
- `INSIGHTS.md` I14 部署位置写明 "§6.11 可视化"

三份源文档一致引用 §6.11，本章保持源文档的承诺。**升级为作者层决策事项**：Stage 3 reviewer panel 之前请明确：
- (a) Ch6 大纲（4 000 字 · "贡献复述 · 局限 · W2 两方向 · 结语"）下显式增设 §6.11 子节
- (b) 移迁到 §5.11 失败案例节
- (c) 保留 §6.11 但作为附录 F（"v3 完整配置"未收）的替代位置

记录于 `hyper_mve/docs/thesis_plan/README.md`「Plan Mode 后续待办」附记一项。

### Pre-Fix → Post-Fix 状态对比

| 指标 | Pre-Fix | Post-Fix |
|---|---|---|
| Inline 非 [N] 引用 | 10 处 | 0 处 ✓ |
| [40] Fehr-Schmidt 上下文措辞 | 含歧义 | 已澄清"本文借用 + 本文 φ(c_t) 调制" ✓ |
| [24] MuZero §1.2 引用上下文 | 与多 agent 设定不匹配 | 已改引 [8] MAMBA / [30] MA-MuZero ✓ |
| §1.6 "方法身份" 用词 | 接近 INSIGHTS 禁词边界 | 已改为"v4 方法的实证起点" ✓ |
| §6.11 cross-ref | 与 Ch6 4000 字预算不匹配 | 保留为作者决策事项 ⏸ |
| CITATION_REGISTRY §2 待分配 | 13 项 | 4 项（LoRA / μP / DiT / StyleGAN，§2/§4 时再分配）|

### 修订后总判定

**Verdict**: `WARNINGS_PRESENT`（下调：原 5 项 P1 移交 → 现 1 项作者决策事项 + 0 项 P1）

无阻塞性问题；剩余 P2 修订项（§1.5/§1.6 偏长、决策门 0 铺垫、Hard A=6 回查）建议在 Stage 4 一并处理，不阻塞 Stage 3 reviewer panel。

### 下一步（Option B）

待 user account quota 重置（~2:30pm Asia/Shanghai）：
1. 重新 dispatch workflow `thesis-ch1-stage2-full`（scriptPath 已持久化在 transcript dir）
2. 由于 P1 inline fixes 已 commit，workflow P3-P9 将在已修订的 ch1_full.md 上做深度多智能体验证
3. workflow 完成后，若 verdict = `STAGE_2_COMPLETE_NO_BLOCKERS`，进入 Stage 3 REVIEW
4. 若仍有 BLOCKERS（罕见），Stage 4 inline 微修后回到 Stage 2.5 复检
