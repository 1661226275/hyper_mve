# Gap Matrix: MASTER_PLAN §2 Assertions × Chapter Outlines × TB Evidence

## Master Gap Matrix

| § | Section title | Claim/Beat | Required experiment | TB cell(s) found | Seeds completed | Classification | Notes |
|---|---|---|---|---|---|---|---|
| **§1.1** | 研究背景：从纯合作/零和到混合动机 | Contextual motivation; no empirical claim | None | — | — | **NOT_APPLICABLE** | Pure narrative; cites I13 Gap(c) emergence |
| **§1.2** | 问题陈述：RNS-MMG | Formal problem statement; conceptual | None (formal definition) | — | — | **NOT_APPLICABLE** | Theory-only by design |
| **§1.3** | 五范式系统性局限 | Five-paradigm critique (related-work style) | None directly; references §5 results | — | — | **NOT_APPLICABLE** | Related work analytical; ch1_full §1.3 already expanded |
| **§1.4** | 研究目标与四项贡献 | Four contributions (承诺级) | Forward-references §5.5/§5.6/§5.7/§5.8 | All 5 lora cells + main + zero_shot + abl4 | Mixed (see below) | **EVIDENCE_PARTIAL** | Contribution #2 (B′) needs gen_scope sweep — only 3 of 7 variants present (duo_base, duo_film, duo_film_fc2, medium_base); contributions #1/#3 (A/C) have NO ablation data |
| **§1.5** | 可证伪断言与论证结构 | A/B′/C/D + main predictions + rollbacks | All 5 ablations + zero-shot | Partial; see per-assertion rows | — | **EVIDENCE_PARTIAL** | Table already drafted in ch1_full; rollback triggers can be written but only B′/D have any data |
| **§1.6** | 论文组织 | Chapter map | None | — | — | **NOT_APPLICABLE** | Pure structure |
| **§2.1** | 混合动机 MARL 问题谱系 | Literature taxonomy | None | — | — | **NOT_APPLICABLE** | Related work |
| **§2.2** | 多智能体世界模型 (incl. Harsanyi) | Conditioning-spectrum positioning | None empirically; §4.2 evidence referenced | duo_base / duo_film / duo_film_fc2 / medium_base | 0/1, 0/2, 0/3, 0/3 | **THEORY_ONLY** | Will cite §4.2 / §5.5 results; can hold figure pending |
| **§2.3** | 基准与评测 | Benchmark survey | None | — | — | **NOT_APPLICABLE** | Conceptual |
| **§2.4** | 小结：本文位置 | Positioning paragraph | None | — | — | **NOT_APPLICABLE** | Synthesis |
| **§3.1** | RNS-MMG 形式化 | Formal tuple G = ⟨N,S,A,P,R,O,C,P_c,ρ_c,γ⟩, C1/C2/C3 | None (definition) | — | — | **NOT_APPLICABLE** | Theory; opener already drafted |
| **§3.2** | v3 NonStationaryTag 初步研究 (G2 中等版 ~2 pp) | v3 hypernet 范式验证 | Legacy v3 runs (NOT in current suite) | None in `runs/suite/` | 0 in suite | **THEORY_ONLY** | v3 results live in `_legacy_v4_7/`; must be re-sourced from legacy or written as historical recap. **No suite TB evidence.** |
| **§3.3** | ResourceCommons 物理+偏好二层分离 | Environment description | None (env spec) | — | — | **NOT_APPLICABLE** | Definitional |
| **§3.4** | 资源场 + α(c) 单通道控制 | env mechanics | None empirical; cites I2/I10 | — | — | **NOT_APPLICABLE** | Definitional |
| **§3.5** | 偏好层 α / β + Fehr-Schmidt (I9/I11) | Preference structure | None (definition) | — | — | **NOT_APPLICABLE** | Definitional |
| **§3.6** | **Prop 3.1: 合作-竞争切换涌现性** | Gap(c) monotonicity + emergent threshold (I13) | Closed-form proof (Appendix A) + Gap(c) measurement | Implicitly visible via `eval/planner/return_total` vs `eval/prior/return_total` & fairness at varying c values: cells use c0.2/c0.5/c0.8 metric branches | **eval/*_c0.{2,5,8}** logged in all 7 cells, all seeds | **EVIDENCE_PARTIAL** | The `_c0.2/c0.5/c0.8` scalar variants are *exactly* the Gap(c) sweep data. zero_shot_easy + abl4 (6 seeds) give clean Gap(c) curves; need to aggregate. Proposition proof is theory-only (Appendix A). |
| **§3.7** | 评测指标 | fairness / sustainability / tragedy / welfare definitions | Verify metric semantics from TB | All cells log all 4 + c0.{2,5,8} variants | All | **EVIDENCE_SUFFICIENT** | All metrics present; **note tragedy ≡ 0 across all runs — flag in §3.7 or §5.11** |
| **§4.1** | 三个容量瓶颈 (I14 reward gradient table) | type-α=1 vs type-β∈[0,2] tearing motivation | Forward to §5.7; gradient algebra | None directly for §4.1 | — | **THEORY_ONLY** | Editor flag: residual v3 wording. Use I14 algebraic table; no §5.7 data to back it. **[EVIDENCE PENDING]** for empirical illustration |
| **§4.2** | 条件化谱：容量-优化几何 | cos_pred_cross 0.61→0.998 spectrum collapse | gen_scope sweep | duo_base / duo_film / duo_film_fc2 / medium_base; `diag/cos_pred_cross` present in all | duo_base 0/1, duo_film 0/2, duo_film_fc2 0/3, medium_base 0/3 (TB only, no registry completion) | **EVIDENCE_PARTIAL** | Spectrum evidence exists for **4 variants** in TB; **medium_base ran only 66k steps, fairness went negative** — cannot publish without rerun; **3 spectrum variants missing entirely** (per MASTER_PLAN expected 7 variants for §5.5) |
| **§4.3** | Harsanyi 主-客通路分解 | Conceptual mapping (I7/I11) | None | — | — | **NOT_APPLICABLE** | Theoretical analogy only |
| **§4.4** | DualHyperNetwork v2 架构 (D3 完整代数, 7500字) | Architecture spec | None (definition) | — | — | **NOT_APPLICABLE** | Definitional. Cites I7 structural isomorphism |
| **§4.5** | 规划层：CoordDesc + CRN (I5/I6) | A^N→N·A; SNR 0.02→1.0 | SNR measurement protocol (§5.1.2) | `collect/q_gap`, `collect/q_std`, `collect/H_pi_mve_fresh`, `collect/uniform_frac` in all cells | All | **EVIDENCE_PARTIAL** | TB has the *symptoms* (q_gap, q_std, entropy) across all runs but no **explicit SNR=0.02→1.0 measurement run**. Need a dedicated CRN-on/off SNR probe to get the headline number. Geometry/algebra is theory. |
| **§4.6** | Type-aware K=5 + 三阶段课程 | Training algorithm spec | None (definition) | — | — | **NOT_APPLICABLE** | Editor flag: residual v3 wording in opener |
| **§5.1** | 实验协议 + μP 公平性 (I15) | Protocol description | None | All cells share schema | — | **NOT_APPLICABLE** | Protocol + I15 narrative |
| **§5.2** | **决策门 0：条件化谱预筛 (gen_scope sweep, 独立前置实验)** | Pre-registered gate for B′ spectrum | gen_scope sweep across 7 variants on Easy | **lora/duo_base, lora/duo_film, lora/duo_film_fc2, lora/medium_base** | 0/1, 0/2, 0/3, 0/3 (TB-only, all registry pending) | **EVIDENCE_PARTIAL** | **CRITICAL GAP**: only 4 of expected 7 gen_scope variants present; **no variant has completed registry entry**; medium_base failed (66k/240k steps, fairness negative). Gate-0 cannot be cleanly declared GO/NO-GO. |
| **§5.3** | v3 NonStationaryTag 迁移验证 | Genesis verification | v3 runs | **None in current suite** | 0 | **THEORY_ONLY** | Must source from legacy v4_7 runs or write as historical recap |
| **§5.4** | ResourceCommons 主对比 Medium 2α+2β | Main comparison vs 6 baselines | `main_comparison_easy` is **Easy**, not Medium; **no Medium cell in suite** | `main_comparison_easy` only | 0/3 (all registry pending despite ~135k TB steps) | **EVIDENCE_PARTIAL** | Wrong cell name (`*_easy`) for §5.4 which targets Medium. Either rename §5.4 to Easy or run Medium. **All 3 seeds have TB but registry not finalized — backfill required.** Also: this is a **hyper-only** suite; "vs 6 baselines" claim needs baseline runs that don't exist. |
| **§5.5** | **消融 1 + 零样本泛化 → 断言 B′** | gen_scope sweep + zero-shot c-generalization | zero_shot_easy + lora/* | `zero_shot_easy` (3/3 ✓), `lora/duo_base` 0/1, `lora/duo_film` 0/2, `lora/duo_film_fc2` 0/3, `lora/medium_base` 0/3 | **zero_shot_easy: 3/3 COMPLETED** | **EVIDENCE_PARTIAL** | Zero-shot half: **EVIDENCE_SUFFICIENT** (3 seeds, return_zero_shot_unseen ≈ 130.88, regret_mean = 0.0). gen_scope half: incomplete (see §5.2). **Critical caveat: registry `return_mean` (≈130.86) > TB final (≈103.81) — disambiguate which is authoritative before quoting.** |
| **§5.6** | **消融 2：Context 路径拆分 → 断言 C** | c_ctx / role_i / belief_i triple ablation | Triple-ablation cell | **None in suite** | 0 | **THEORY_ONLY** | **No data**. Must write theoretically with [EVIDENCE PENDING] or run new cells |
| **§5.7** | **消融 3 + 类型梯度可视化 → 断言 A** | per-agent θ_rew bell curve; ρ_β scan | ρ_β-scan cell (50/50, 25/75, 75/25, 0/100, 100/0) | **None in suite** | 0 | **THEORY_ONLY** | **No data**. I14 reward-gradient algebraic table is best available scaffold. Also: §6.11 gradient-heatmap cross-ref **unresolved** |
| **§5.8** | **消融 4：规划器双重技术 → 断言 D** | 2×2 CoordDesc × CRN matrix on Easy N=2 | 2×2 ablation cell | **`abl4_joint_easy_n2` = 3 seeds completed** | **3/3 ✓** | **EVIDENCE_PARTIAL** | Cell name "abl4_joint" matches a *joint* ablation, but a **2×2 matrix needs 4 conditions** — only 1 of 4 cells appears here. Need to verify what `abl4_joint_easy_n2` actually toggles (joint vs CoordDesc+CRN factorial). Likely missing 3 of 4 quadrants. **regret_mean=0.0 across seeds** — verify whether metric is wired up. |
| **§5.9** | **消融 5 + Scale-up + 鲁棒性 (planner-off self-distillation collapse)** | planner-off → π → uniform (ln A) | planner-off cell | **None in suite** | 0 | **THEORY_ONLY** | Existing TB shows `collect/H_pi_mve_fresh` healthy across all runs; no planner-off run for the collapse demo (I5) |
| **§5.10** | 信念推断质量 + 三阶段课程有效性 | belief loss curve + curriculum stages | `loss/belief`, `loss/belief_div`, `loss/belief_opp`, `loss/belief_c`, `loss/lambda_b` | All cells log all 5 belief tags | All | **EVIDENCE_PARTIAL** | **Major issue**: `loss/belief ≤ 1e-5` in 14 of 18 runs — belief learning appears inactive. Only `lora/medium_base` (3 seeds, the failed cell) shows `loss/belief ≈ 1e-3`. Curriculum-stage transitions are not separately tagged. **Belief-quality protocol cannot be claimed positively from this data.** |
| **§5.11** | 失败案例与训练诊断 | Diagnostic narrative | TB diagnostics across all runs | All cells | — | **EVIDENCE_SUFFICIENT** | Plenty to diagnose: `lora/duo_film` seed1 diverged (gap≈101), `lora/medium_base` 3 seeds failed (fairness negative, 66k steps), `tragedy ≡ 0` everywhere, `regret_mean ≡ 0` everywhere. Rich material for honest failure-case writing. |
| **§6.1** | 贡献总结 + §5 断言验证状态 | A/B′/C/D verification table | §5.5/5.6/5.7/5.8 results | Mixed; B′ partial, D partial, A/C empty | — | **EVIDENCE_PARTIAL** | Contains the only inline `[写作时填入实测结果]` placeholder. **Write A=未支持(数据不足) / B′=部分支持 / C=未支持(数据不足) / D=部分支持** until missing cells run |
| **§6.2** | 局限 | Limitations | None directly; recap | — | — | **NOT_APPLICABLE** | Honest limitations: agent scale, ablation coverage, social-preference family |
| **§6.3** | 展望 W1+W2 (offline-RL on commons, Ostrom) | Future directions | None | — | — | **NOT_APPLICABLE** | Forward-looking |
| **§6.4** | 结语 | Closing | None; cites I2/I4/I12 | — | — | **NOT_APPLICABLE** | Synthesis |
| **§6.11** (cross-ref from §5.7 / §4.1) | **类型梯度热图** ∂ř/∂u_i | gradient visualization | Gradient extraction from trained θ_rew | None | 0 | **THEORY_ONLY** | **Unresolved location** (escalated in `ch1_verification_report.md` Post-Fix Addendum P1#1). Either move to Appendix C or §5.7. Algebraic table I14 is the only current scaffold. |
| **Appx A** | Prop 3.1 闭式证明 | Closed-form proof | Theory only | — | — | **NOT_APPLICABLE** | Pure proof; Q8 fixes appendix placement |
| **Appx B** | 符号/缩写/术语 | Glossary | None | — | — | **NOT_APPLICABLE** | |
| **Appx C** | gen_scope 工程纪律 | Implementation detail | None | — | — | **NOT_APPLICABLE** | |
| **Appx D** | 超参数表 | HP table | Extract from registry / config.py | All cells share `pkg08-spec05-v1` | — | **EVIDENCE_SUFFICIENT** | Pull from `BaseConfig` + per-cell metadata |
| **Appx E** | 复现指南 | Repro guide | None; references git SHA `8ea7832` | All registry rows pin same SHA | — | **EVIDENCE_SUFFICIENT** | Clean tree, single SHA, deterministic; ideal for repro guide |

---

## Per-Assertion Roll-Up (MASTER_PLAN §2)

| Assertion | Status | Cells available | Cells missing | Verdict |
|---|---|---|---|---|
| **A** (type-gradient tearing, bell curve, peak ≥5%/p<0.05) | **THEORY_ONLY** | None (ρ_β scan not in suite) | ρ_β ∈ {0/100, 25/75, 50/50, 75/25, 100/0} × per-agent θ_rew | **Cannot evaluate.** Write algebra (I14) + [EVIDENCE PENDING] |
| **B′** (spectrum internal optimum, 0.61→0.998 collapse) | **EVIDENCE_PARTIAL** | 4 spectrum variants in TB (no registry completion); zero_shot_easy (3 completed seeds) | 3 of 7 gen_scope variants; medium_base needs rerun (only 66k steps, fairness negative); registry finalization for 9 runs | **Partial support.** Can write zero-shot half with full evidence; spectrum half is preliminary |
| **C** (triplet pathway orthogonal necessity) | **THEORY_ONLY** | None | c_ctx / role_i / belief_i triple ablation; hidden-c condition | **Cannot evaluate.** Write Harsanyi mapping (I11) + [EVIDENCE PENDING]. **Belief data in TB is degenerate (≤1e-5).** |
| **D** (CoordDesc × CRN inseparable, SNR 0.02→1.0) | **EVIDENCE_PARTIAL** | `abl4_joint_easy_n2` 3 seeds completed | 3 of 4 cells in 2×2 matrix; explicit SNR measurement run | **Partial.** Have the joint cell; need {CD-only, CRN-only, neither} on Easy N=2. Also need SNR-probe to substantiate the 0.02→1.0 claim |

---

## Writing strategy

### 1. Chapters/sections writable now with full empirical support

- **§3.7 评测指标**: all metrics logged across all 18 runs. Write directly; flag `tragedy ≡ 0` as definitional in Easy.
- **§5.5 (zero-shot half only)**: `zero_shot_easy` has 3/3 completed seeds with backfilled metrics. Headline: return_mean ≈ 130.86, return_zero_shot_unseen ≈ 130.88, regret_mean = 0.0. **Disambiguation required**: registry value (~130.86) vs TB final (~103.81) — pick the authoritative number (likely registry, as it comes from end-of-training `eval_report.json` over a fuller ctx grid). State the eval protocol explicitly.
- **§5.8 (joint cell only)**: `abl4_joint_easy_n2` is the **only** ablation cell with 3 completed seeds. Use to ground "joint MVE coordinate descent + CRN" but be honest that this is the all-on cell, not a 2×2 matrix. The 2×2 D-assertion claim must be downgraded or partially deferred.
- **§5.11 失败案例**: rich material — duo_film seed1 divergence, medium_base failure, tragedy/regret degeneracy. Honest failure-case writing strengthens the thesis.
- **Appendix D / E**: hyperparameters and reproducibility — all rows pin one git SHA; clean.
- **Ch6 §6.2 局限**: empirical evidence (medium_base failure, belief loss ≈ 0, only 1 of 4 D-quadrants) makes the limitations section concrete and credible.

### 2. Sections needing [EVIDENCE PENDING] + theoretical scaffolding

| Section | Scaffold to write | [EVIDENCE PENDING] anchor |
|---|---|---|
| **§4.1** (3 capacity bottlenecks) | I14 reward-gradient algebraic table (type α=1 const vs β ∈ {0, 0.7, 1.3, 2.0}); derive bell-curve necessity algebraically | "Empirical bell-curve validation deferred to §5.7 [EVIDENCE PENDING — Ablation 3 cell not yet run]" |
| **§4.5** (CoordDesc + CRN geometry) | A^N→N·A reduction (I6); CRN variance-reduction proof for step-0 SNR | "Explicit SNR 0.02→1.0 measurement deferred to §5.1.2 [EVIDENCE PENDING — SNR probe run not yet executed]" |
| **§5.2** (Decision-Gate 0) | Pre-registration narrative; gen_scope variant taxonomy from §4.2; **report the 4 variants we have as preliminary GO/NO-GO** | "Full 7-variant gate decision pending [EVIDENCE PENDING — 3 spectrum variants + medium_base rerun required]" |
| **§5.3** (v3 NonStationaryTag) | Cite `_legacy_v4_7/` results as historical recap; "Genesis validation" framing per I4 | "v3 ChunkedHMLP runs reproduced from legacy v4.7 codebase [EVIDENCE PENDING — re-extract from legacy run dir]" |
| **§5.4** (Main Medium) | Either rename to "Easy main comparison" using `main_comparison_easy` (3 TB seeds, registry pending → **finalize registry first**), OR write skeleton with [EVIDENCE PENDING — Medium runs not yet executed]. **Also**: the "vs 6 baselines" framing has no baseline data — soften to "hyper-only conditioning-spectrum scan vs hyper-baseline" |
| **§5.6** (Triple-pathway ablation) | I11 Harsanyi mapping; argue orthogonality from architectural decomposition | "Triple-ablation cell pending [EVIDENCE PENDING — c_ctx/role/belief zeroing runs not yet executed]" |
| **§5.7** (ρ_β scan + heatmap) | I14 algebraic gradient table; bell-curve prediction from algebra | "ρ_β-scan ablation cell pending [EVIDENCE PENDING]; §6.11 gradient heatmap location to be decided" |
| **§5.9** (planner-off collapse) | I5 ln(A) fixed-point argument; cite local memory note "v4 collection needs planner" | "Planner-off run pending [EVIDENCE PENDING — explicit self-distillation collapse demonstration not yet executed]" |
| **§5.10** (belief quality) | Stage curriculum description; cite I8 demoted insight | **Special caveat**: TB shows `loss/belief ≤ 1e-5` in 14/18 runs — must either explain why belief is dormant (curriculum stage not reached?) or **acknowledge a P0 issue with the belief pipeline**. Cannot claim "belief learning works" from current data. |
| **§6.1** (verification table) | Replace `[写作时填入实测结果]` with: A=未支持(实验未执行) / B′=部分支持(零样本路径已验证, gen_scope sweep 部分) / C=未支持(实验未执行) / D=部分支持(联合单元已验证, 2×2 矩阵未完成) | These are **honest preliminary verdicts** — re-rate after missing cells run |
| **§6.11** (gradient heatmap) | Decide location: **recommend Appendix C** (engineering visualization) rather than §5.7 main text, to keep §5.7 focused on ρ_β-scan results | Cross-ref location resolved; content [EVIDENCE PENDING] |

### 3. Experimental cells that need attention (re-run / finalize / new)

**Highest priority (P0 — blocks thesis-grade claims):**

1. **`lora/medium_base` — RE-RUN all 3 seeds.** Currently 66k/240k steps; `fairness` went **negative** (-0.27 to -0.40); planner_prior_gap deeply negative (-67 to -71). Either training crashed or config is broken. This is the **only** cell that logs the extra `_same` diagnostics — confirm it's not a regression.
2. **Belief-pipeline investigation.** `loss/belief ≤ 1e-5` in 14/18 runs is a P0 signal: either §5.10 cannot make positive claims, or belief was unintentionally disabled. **Inspect `BaseConfig` belief loss weight / curriculum stage gating before running anything else.**
3. **Finalize registry for 12 pending runs** (`main_comparison_easy` ×3, `lora/duo_base` ×1, `lora/duo_film` ×2, `lora/duo_film_fc2` ×3, `lora/medium_base` ×3). TB data exists; the post-training eval + registry-completion step appears not to have run. **Engineering fix, not a re-train.**

**P1 — needed for headline assertions:**

4. **gen_scope sweep completion.** Currently 4 spectrum variants present (duo_base, duo_film, duo_film_fc2, medium_base). MASTER_PLAN expects 7. Add the missing 3 variants per `gen_scope` taxonomy in §4.2.
5. **Ablation 3 (ρ_β scan) — Assertion A.** Run ρ_β ∈ {0/100, 25/75, 50/50, 75/25, 100/0} on Easy N=2 (or N=4), 3 seeds each. **15 runs total.** Without this, Contribution #2 cannot be validated.
6. **Ablation 2 (triple-pathway) — Assertion C.** Run {c_ctx-off, role-off, belief-off, all-on} × 3 seeds on Easy. **12 runs.** Without this, Contribution #3 cannot be validated.
7. **Ablation 4 (2×2 CoordDesc × CRN) — Assertion D completion.** Have joint cell; need {CD-only, CRN-only, neither} on Easy N=2 × 3 seeds. **9 additional runs.**
8. **SNR-probe run.** Single-purpose: log step-0 Q-value variance for CRN-on vs CRN-off identical policies. Needed for the headline 0.02→1.0 number.

**P2 — nice-to-have:**

9. **v3 NonStationaryTag re-extraction** from `_legacy_v4_7/` for §5.3 historical recap, OR a fresh v3 reproduction run.
10. **Medium-scale main comparison** for §5.4 (currently only Easy). If GPU budget allows; else rename §5.4 to "Easy main comparison" and move Medium to future work.
11. **Baseline runs** (Exp1/Exp2/Exp3 from legacy or fresh) to substantiate the "vs 6 baselines" claim in §1.4 contribution-set narrative.
12. **`lora/duo_film` seed1 inspection.** Gap≈101 vs seed0 gap≈1 — either accept high variance or rerun with different RNG.

**Drafting sequence recommendation:**

- **Round 1 (now):** Write all NOT_APPLICABLE sections + §3.7/§5.11/§6.2/Appx D/E (EVIDENCE_SUFFICIENT). Write §5.5 zero-shot half and §5.8 joint cell with full numbers.
- **Round 2 (after registry finalization P0 #3):** Re-quote registry-backfilled metrics in §5.4 / §5.5 / lora cells; update §6.1 verdict table.
- **Round 3 (after P0 #1 + #2):** Decide whether §5.10 makes positive or negative claim; whether medium_base remains in spectrum sweep.
- **Round 4 (after P1 experiments):** Replace [EVIDENCE PENDING] tags in §4.1/§4.5/§5.2/§5.6/§5.7/§5.9 with real results; finalize §6.1 verification table.
- **Round 5:** Ch1 §1.4/§1.5 contribution-strength language tightens to match final verdicts.

**Headline risk:** As of current TB state, **only Assertion B′ (zero-shot half) and Assertion D (joint cell)** can be claimed with empirical backing. Assertions A and C are entirely theory-only. The thesis can be drafted in full but **§6.1's verification table will report 2 partial / 2 未支持** unless P1 experiments run. Recommend pausing chapter-drafting on §5.5/§5.6/§5.7/§5.8 until either (a) the missing ablation cells run or (b) the author confirms a downgrade path per MASTER_PLAN's "失败回退" column.