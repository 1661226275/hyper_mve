I have everything. Now I'll return the inventory as a single Markdown report.

---

# Thesis Drafts Inventory — `D:/RL/hyper_mve/docs/thesis_plan/drafts/`

Path prefix omitted in headers below; all files live under `D:\RL\hyper_mve\docs\thesis_plan\drafts\`.

---

## `ch1.md` — Plan-Mode skeleton (Chapter 1)

- **Full path**: `D:\RL\hyper_mve\docs\thesis_plan\drafts\ch1.md`
- **Line count**: 117
- **Size estimate**: ~11.4 KB ≈ **~3 200 Chinese chars** (matches README claim)
- **Status**: **skeleton-only** (Plan-Mode "opening paragraph + beats + core citations" template, all six sections present as opening block-quote draft + bullet-list "本节 beats" + "核心引用" footer)
- **Section structure** (`§` headers only):
  - `# 第 1 章 绪论 — 开场段草稿`
  - `## §1.1 研究背景：从纯合作/零和到混合动机的真实世界（~1 100 字）`
  - `## §1.2 问题陈述：关系非平稳混合动机博弈 (RNS-MMG)（~1 400 字）`
  - `## §1.3 现有方法四个范式的系统性局限（~1 600 字）`
  - `## §1.4 研究目标与四项贡献（~1 600 字，承诺级表述）`
  - `## §1.5 可证伪断言与论文论证结构（~1 400 字）`
  - `## §1.6 论文组织（~900 字）`
- **Per-subsection draft state** (opening paragraph drafted vs TODO):
  - §1.1 — opening paragraph drafted (3 block-quote paras); beats listed; cites listed
  - §1.2 — opening paragraph drafted (2 paras); beats + cites
  - §1.3 — opening paragraph drafted (2 paras); beats + cites
  - §1.4 — **承诺级 code-block drafted verbatim** (contributions 1–4); no beats list
  - §1.5 — opening paragraph drafted + **assertion table A/B′/C/D drafted**; no beats
  - §1.6 — opening paragraph drafted (2 paras); no beats list

---

## `ch1_full.md` — Stage-2 drafted Chapter 1

- **Full path**: `D:\RL\hyper_mve\docs\thesis_plan\drafts\ch1_full.md`
- **Line count**: 161
- **Size estimate**: ~35.3 KB ≈ **~6 228 Chinese chars** (per v2 verification report § 1 — confirmed -22% under the 8 000-char target)
- **Status**: **drafted** (full prose, 17 inline fixes applied across Option-C and salvage passes; Stage-2.5 verdict `WARNINGS_PRESENT`)
- **Section structure**:
  - `# 第 1 章 绪论`
  - `## §1.1 研究背景：从纯合作/零和到混合动机的真实世界`
  - `## §1.2 问题陈述：关系非平稳混合动机博弈 (RNS-MMG)`
  - `## §1.3 现有方法五个范式的系统性局限` (note: header says **五个范式** — was originally **四个**; §1.3 was expanded to add the 5th paradigm "非平稳 MARL 通用应对策略")
    - sub-headers: `### Model-free 合作型 MARL 与 CTDE 框架` · `### 单智能体与多智能体世界模型` · `### 社会困境专用方法` · `### 非平稳 MARL 通用应对策略` · `### 五范式批评的综合`
  - `## §1.4 研究目标与四项贡献`
  - `## §1.5 可证伪断言与论文论证结构`
  - `## §1.6 论文组织`
- **[TODO] / [PENDING] / [EVIDENCE PENDING] markers**: **none found in the body text** of `ch1_full.md`. The chapter is prose-complete; pending issues are tracked externally in the v2 verification report and Stage-3 reviews (e.g., -22% length gap, §6.11 cross-ref decision, Stage-4 P2 prose-tightening list). No inline `[TODO]`, `[PENDING]`, `[EVIDENCE PENDING]`, `[ ]`, or `<待填写>` placeholders.

---

## `ch1_stage3_reviews.md` — Stage-3 5-reviewer panel notes

- **Full path**: `D:\RL\hyper_mve\docs\thesis_plan\drafts\ch1_stage3_reviews.md`
- **Line count**: 887
- **Size estimate**: ~81 KB (mixed English/Chinese; predominantly English review prose)
- **Status**: **review-notes** (5-reviewer panel × full reports + roadmap + traceability matrix; verdict **MINOR_REVISIONS**, dated 2026-06-18)
- **Section structure**:
  - `# 第 1 章 — Stage 3 REVIEW: 5-Reviewer Panel`
  - `## Editorial Decision`
  - `## 5 Reviewer Verdicts at a Glance`
  - `## Common Concerns (≥2 reviewers raised)` (12 concerns enumerated)
  - `## EIC Report`
  - `## R1 Report — Methodology`
  - `## R2 Report — Empirical`
  - `## R3 Report — Related Work`
  - `## DA Report — Devil's Advocate`
  - `## Revision Roadmap`
  - `## R&R Traceability Matrix (Schema 11)`
  - `## Next Stage Recommendation`

---

## `ch1_verification_report.md` — Stage-2.5 verification v1 (solo inline pass)

- **Full path**: `D:\RL\hyper_mve\docs\thesis_plan\drafts\ch1_verification_report.md`
- **Line count**: 352
- **Size estimate**: ~24.4 KB
- **Status**: **verification-report** (Verdict: `WARNINGS_PRESENT`; solo pass + Option-C post-fix addendum; dated 2026-06-17)
- **Section structure**:
  - `# 第 1 章 — Stage 2.5 验证报告（Solo Inline Pass）`
  - `## 总判定` — verdict `WARNINGS_PRESENT`
  - `## Stage 2 草稿摘要` (per-section word-count table; first-pass numbers later corrected in v2)
  - `## P3+P4 多维审稿（Solo Light Pass）` — Correctness / Completeness / Alignment / Calibration lenses
  - `## P5 5 阶段完整性` — Phase A (Citation Refs) / B (Context) / C (Quantitative Data) / D (Originality) / E (Claim Verification)
  - `## P6 7-Mode AI 失败检查表（Solo Light Pass）`
  - `## P7 断言忠实度审计（Claim Audit，抽样）` — 10/10 SUPPORTED sample
  - `## P8 跨模型分歧检测（Sonnet Cross-Model）— DEFERRED`
  - `## 阻塞性问题 / 待 Stage 3 移交项 / 待 Stage 4 修订项`
  - `## Audit Trail`
  - `## 下一步`
  - `## Post-Fix Addendum（2026-06-17，Option C 执行）` — 5 inline Edits applied, registry sync, residual §6.11 cross-ref decision

---

## `ch1_verification_report_v2.md` — Stage-2.5 verification v2 (workflow salvage + inline fixes)

- **Full path**: `D:\RL\hyper_mve\docs\thesis_plan\drafts\ch1_verification_report_v2.md`
- **Line count**: 242
- **Size estimate**: ~17.6 KB
- **Status**: **verification-report** (Verdict: **`WARNINGS_PRESENT`**; salvage from workflow runs #1 & #2, 11 additional inline fixes; dated 2026-06-17)
- **Section structure**:
  - `# 第 1 章 — Stage 2.5 验证报告 v2（Workflow Salvage + Solo Pass + Inline Fixes）`
  - `## 总判定 v2`
  - `## 与 Solo Pass 的差异（Workflow Salvage 新增发现）` — 7 new P1 findings table; 5 inline-fixed, 2 escalated
  - `## P3+P4 4-Lens Deep Review（Workflow 实际结果）` — Lens 1 Calibration (PASS_WITH_NITS) / Lens 2 Alignment (NEEDS_REVISION) / Lens 3 Correctness (PASS_WITH_NITS) / Lens 4 Completeness (NEEDS_REVISION)
  - `## 本轮 11 处 Inline 修订汇总`
  - `## P5 5-Phase Integrity (Workflow Missing — Solo Pass v1 Coverage)`
  - `## P6 7-Mode AI Failure Checklist (Workflow Missing — Solo Pass v1 PROV-CLEAR)`
  - `## P7 30-Claim Faithfulness Audit (Workflow Missing — Solo Pass 10 抽样 SUPPORTED)`
  - `## P8 Sonnet Cross-Model (Workflow Missing — 跨模型独立验证未执行)`
  - `## 阻塞性问题 / 待 Stage 3 移交项 / 待 Stage 4 修订项`
  - `## Stage 3 Dispatch Readiness` — verdict ✓ READY
  - `## 工作流执行档案`
- **Key signal**: `completeness#0` — actual Chinese-char count = **6 228 vs 8 000 target = −22%**, migrated to Stage 4 as the main expansion driver. Solo P6/P7/P8 deferred.

---

## `ch2.md` — Plan-Mode skeleton (Chapter 2: Related Work)

- **Full path**: `D:\RL\hyper_mve\docs\thesis_plan\drafts\ch2.md`
- **Line count**: 58
- **Size estimate**: ~7.2 KB ≈ ~2 000 chars
- **Status**: **skeleton-only** (Q4 = "B 三层式 问题→方法→评测"; 4 sections; target ~7 500 chars)
- **Section structure**:
  - `# 第 2 章 相关工作 — 开场段草稿`
  - `## §2.1 混合动机多智能体强化学习问题谱系（~2 200 字）`
  - `## §2.2 多智能体世界模型方法（~2 600 字，含中等版 Harsanyi）`
  - `## §2.3 基准与评测（~1 800 字）`
  - `## §2.4 小结：本文位置（~900 字）`
- **Per-subsection state**:
  - §2.1 — **opening paragraph drafted** (1 dense block-quote para); beats + core cites listed
  - §2.2 — **opening paragraph drafted** (3 block-quote paras covering MA world models, 条件化谱, Harsanyi); beats + cites
  - §2.3 — **opening paragraph drafted** (1 para); beats + cites
  - §2.4 — **opening paragraph drafted** (1 closing-statement para); no beats list

---

## `ch3.md` — Plan-Mode skeleton (Chapter 3: Problem & Environment)

- **Full path**: `D:\RL\hyper_mve\docs\thesis_plan\drafts\ch3.md`
- **Line count**: 45
- **Size estimate**: ~12.9 KB ≈ ~4 100 chars
- **Status**: **skeleton-only** (Q7 = G2 中等版 v3; Q8 = 命题证明入附录 A; Q9 = Self+Full 双档; 7 sections; target ~9 600 chars)
- **Section structure**:
  - `# 第 3 章 问题与环境 — 开场段草稿`
  - `## §3.1 RNS-MMG 形式化（~1 500 字）`
  - `## §3.2 初步研究：v3 NonStationaryTag（~1 100 字 / 约 2 页，G2 中等版）`
  - `## §3.3 ResourceCommons 基准：物理层 + 偏好层二层分离（~1 700 字）`
  - `## §3.4 资源场与上下文动力学：α(c) 单通道控制（~1 400 字）`
  - `## §3.5 偏好层：类型 α / 类型 β 与 Fehr-Schmidt 结构（~1 600 字）`
  - `## §3.6 命题 3.1：合作-竞争切换的涌现性（~1 200 字，P3：正文陈述 + 附录证明）`
  - `## §3.7 评测指标（~1 100 字）`
- **Per-subsection state**: all 7 sections have an **opening paragraph drafted** as a single dense block-quote (no "beats list" appendix). §3.1 includes the formal tuple `G = ⟨N, S, {A_i}, P, {R_i}, {O_i}, C, P_c, ρ_c, γ⟩` and C1/C2/C3 spelled out. §3.6 contains the Proposition 3.1 statement with proof-sketch (full proof deferred to Appendix A).

---

## `ch4.md` — Plan-Mode skeleton (Chapter 4: Method)

- **Full path**: `D:\RL\hyper_mve\docs\thesis_plan\drafts\ch4.md`
- **Line count**: 87
- **Size estimate**: ~15.4 KB ≈ ~5 000 chars
- **Status**: **skeleton-only** (Q10 = D3 完整代数; Q11 = B 几何+实测; Q6 = 谱图放 §4.2; 6 sections; target ~18 500 chars). Header carries **⚠ 编辑提醒**: §4.1 / §4.6 still contain residual v3 (Faction Re-alignment / 阵营 / 猎人猎物) wording flagged for v4 replacement.
- **Section structure**:
  - `# 第 4 章 方法 — 开场段草稿`
  - `## §4.1 动机：三个容量瓶颈（~2 500 字）`
  - `## §4.2 条件化谱：容量-优化几何（~2 500 字）`
  - `## §4.3 Harsanyi 类比：主-客通路的分解依据（~1 500 字）`
  - `## §4.4 DualHyperNetwork v2 架构（D3 完整代数+工程纪律，~7 500 字）`
  - `## §4.5 规划层：Per-Agent Coordinate Descent + CRN（B 几何+实测，~2 500 字）`
  - `## §4.6 训练算法：Type-aware K 步展开 + 三阶段课程（~2 000 字）`
- **Per-subsection state**:
  - §4.1 — **opening paragraph drafted** (1 substantive block-quote covering type-gradient tearing / belief dilution / role averaging); cites + 编辑提醒 note (v3 → v4 wording cleanup needed)
  - §4.2 — **opening paragraph drafted** (3 paras: spectrum geometry, cos_pred_cross 0.61→0.998 evidence, B′ pre-registration); cites
  - §4.3 — **opening paragraph drafted** (3 paras: Harsanyi common-knowledge / private-belief mapping); cites
  - §4.4 — **opening paragraph drafted** (4 paras: 80-dim c_aug, dual-path generation, 4-mode partial generation, 4-layer stability stack); cites — note this is the longest target (7 500 chars) and currently only opens
  - §4.5 — **opening paragraph drafted** (2 paras: A^N→N·A reduction, SNR 0.02→1.0 mechanism); cites
  - §4.6 — **opening paragraph drafted** (4 paras: K=5 unroll, type-aware reward head, BYOL consistency loss, 3-stage curriculum); cites + 编辑提醒 (v3 wording)

---

## `ch5.md` — Plan-Mode skeleton (Chapter 5: Experiments)

- **Full path**: `D:\RL\hyper_mve\docs\thesis_plan\drafts\ch5.md`
- **Line count**: 121
- **Size estimate**: ~23.2 KB ≈ ~7 600 chars
- **Status**: **skeleton-only** (Q12 = F1 全跑 6 baseline; Q13 = L1 决策门 0 独立前置实验; 11 sections; target ~18 000 chars; ~300 GPU hours)
- **Section structure**:
  - `# 第 5 章 实验 — 开场段草稿`
  - `## §5.1 实验协议：基线、公平性与算力规划（~1 500 字）`
  - `## §5.2 决策门 0：条件化谱预筛（~1 800 字，独立前置实验）`
  - `## §5.3 v3 NonStationaryTag 迁移验证（Genesis 验证，~1 200 字）`
  - `## §5.4 ResourceCommons 主对比（Medium, N=4, 2α+2β，~2 200 字）`
  - `## §5.5 消融 1 + 零样本泛化 → 断言 B′（~2 300 字）`
  - `## §5.6 消融 2：Context 路径拆分 → 断言 C（~1 600 字）`
  - `## §5.7 消融 3 + 类型梯度可视化 → 断言 A（~2 000 字）`
  - `## §5.8 消融 4：规划器双重技术 → 断言 D（~1 800 字）`
  - `## §5.9 消融 5 + Scale-up + 鲁棒性（~1 500 字）`
  - `## §5.10 信念推断质量 + 三阶段课程有效性（~1 500 字）`
  - `## §5.11 失败案例与训练诊断（~1 600 字）`
- **Per-subsection state**: all 11 sections have **opening paragraphs drafted** as dense block-quotes (each section: 1–4 paras outlining protocol, key numbers like SNR 0.02→1.0, cos_pred_cross 0.61→0.998, pre-registered判据 and decision-gate triggers). §5.2 mentions a pre-registered "**决策门 1**" inside §5.7 also. No "beats list" appendix on these; cites are inline. **Note**: §5.7's hooked reference uses "§6.11 节所得的 ∂ř/∂u_i 梯度热图" — this is the unresolved §6.11 cross-ref escalated for author decision.

---

## `ch6.md` — Plan-Mode skeleton (Chapter 6: Conclusion)

- **Full path**: `D:\RL\hyper_mve\docs\thesis_plan\drafts\ch6.md`
- **Line count**: 63
- **Size estimate**: ~9.2 KB ≈ ~2 900 chars
- **Status**: **skeleton-only** (Q14 = W2 两方向重点; Q15 = 附录 A+B+C+D+E; 4 sections + 5 appendices; target ~4 000 chars). Header carries **⚠ 编辑提醒**: §6.1 still has residual v3 (阵营 / r_t / view_emb) wording.
- **Section structure**:
  - `# 第 6 章 总结与展望 — 开场段草稿`
  - `## §6.1 贡献总结（~1 200 字）`
  - `## §6.2 局限（~900 字）`
  - `## §6.3 展望（~900 字，W2 两方向重点）`
  - `## §6.4 结语（~1 000 字）`
  - `## 附录结构概览` — Appendix A (命题 3.1 闭式证明, ~4 pages) · Appendix B (符号/缩写/术语, ~3 pp) · Appendix C (gen_scope 工程纪律, ~3 pp) · Appendix D (超参数表, ~2 pp) · Appendix E (复现指南, ~2 pp)
- **Per-subsection state**:
  - §6.1 — **opening paragraph drafted** (2 paras); §5 assertion-validation status is **explicitly placeholder**: lines contain "得到验证/部分支持/未支持 **[写作时填入实测结果]**" — this is the closest thing to a `[TODO]` marker in the drafts. 编辑提醒 (v3 → v4 wording cleanup needed)
  - §6.2 — **opening paragraph drafted** (4 paras on agent scale, ablation coverage, social-preference family, theoretical completeness)
  - §6.3 — **opening paragraph drafted** (3 paras: industrial scale-up W1, offline-RL on historical commons data W2, Ostrom institutional-coupling long-term vision)
  - §6.4 — **opening paragraph drafted** (2 paras on v3→v4 trajectory and methodological positioning)

---

## Assertion ↔ Experiment Map (quoted verbatim from MASTER_PLAN.md § 2)

The following is reproduced verbatim from `D:\RL\hyper_mve\docs\thesis_plan\MASTER_PLAN.md`, section "## 2. 四项可证伪断言 × 实验映射":

```
## 2. 四项可证伪断言 × 实验映射

| 断言 | 一句话陈述 | 实验位置 | 章节 | 失败回退 |
|---|---|---|---|---|
| **A** | 类型异质条件下，单组共享 RewardHead 被 α(梯度=1) 与 β(梯度∈[0,2]) 反向梯度撕裂；per-agent θ_rew 在 ρ_β≈50/50 处优势达峰，形成钟形曲线 | 消融 3 + §6.11 类型梯度可视化 | §5.7 + §4.1 + §3.6 | Easy N=2 峰值差 <5% → 贡献 2 降级为"特定区段优势" |
| **B′** | 条件化谱内部最优：左端容量平均化、右端 FULL 方向坍缩(0.61→0.998)双面失败；峰值在 film_head+LoRA 至 lora_fc2 之间 | 消融 1 + 零样本 c 泛化 + 决策门 0 gen_scope sweep | §5.5 + §5.2 + §4.2 | 三分级分别降级，最差降为相关工作分析视角 |
| **C** | 三联通路（c_ctx / role_i / belief_i）退化效应正交可加，缺一不可；在 hidden-c 下 belief 通路效应可分离 | 消融 2 + 信念质量协议 | §5.6 + §5.10 + §4.3 | 非正交 → 弱版本"三路径互补"；若 No-type ≡ No-belief 补 No-self-type 消融 |
| **D** | CoordDesc × CRN 不可分割：CRN 把第 0 步 SNR 从 0.02 拉回 1.0；CoordDesc 把 O(A^N) 压到 O(N·A) | 消融 4 (2×2 矩阵 Easy N=2) + §5.1.2 SNR 实测 | §5.8 + §4.5 + §5.9 (planner-off) | Medium 不可行只 Easy 拿四象限 → 强调 ε-Nash 理论保证 |
```

For TB-scalar matching, the **per-assertion section anchors** are:

| Assertion | Primary experiment section | Supporting sections | Failure-rollback trigger |
|---|---|---|---|
| **A** (type-gradient tearing, bell curve) | `§5.7` (Ablation 3 + ρ_β scan) | `§4.1` (3 capacity bottlenecks) + `§3.6` (Prop. 3.1) + `§6.11` cross-ref (gradient heatmap — unresolved location, escalated for author decision in `ch1_verification_report.md` Post-Fix Addendum P1#1) | Easy N=2 peak-gap < 5% (note: `ch1_full.md` §1.5 main prediction tightens this to peak-gap ≥ 5% with p < 0.05; rollback at < 8% or p ≥ 0.05) |
| **B′** (conditioning-spectrum internal optimum) | `§5.5` (Ablation 1, 7 gen_scope variants) | `§5.2` (Decision-Gate 0 sweep) + `§4.2` (spectrum geometry) + zero-shot c-gen + hidden-c | 3-tier degradation (see §1.5 table); worst case → demoted to "related-work analytical lens" |
| **C** (triplet pathway necessity, orthogonal) | `§5.6` (Ablation 2, context-path zeroing) | `§5.10` (belief-quality protocol, dual c_visible regimes) + `§4.3` (Harsanyi) | non-orthogonal → weak version "three-path complementarity"; supplement No-self-type if No-type ≡ No-belief |
| **D** (CoordDesc × CRN inseparable) | `§5.8` (Ablation 4, Easy N=2 2×2 matrix) | `§4.5` (planner geometry + SNR proof) + `§5.9` (planner-off self-distillation collapse) + `§5.1.2` (SNR measurement) | Joint enumeration cell missing on Medium → Easy 2-axis + ε-Nash theoretical guarantee; CRN gain < 30% on Medium → assertion D downgraded to "SNR necessity evidence" |

---

**Aggregate readiness summary** (for the script's downstream gap matrix):

| File | Drafted | Open issues |
|---|---|---|
| `ch1_full.md` | ✓ prose-complete | -22% length gap (Stage 4 expansion in §1.1-§1.4); §6.11 cross-ref pending author decision; 12 reviewer common-concerns from Stage 3 (Minor Revisions verdict, no structural rework) |
| `ch2.md` – `ch6.md` | skeleton only | All openers drafted; "beats" lists only on `ch1.md` and `ch2.md`; `ch4.md` / `ch6.md` carry **编辑提醒** flags for residual v3 wording in §4.1, §4.6, §6.1; `ch6.md` §6.1 has the only inline placeholder text: `[写作时填入实测结果]` (assertion-validation status awaiting §5 experiment results) |