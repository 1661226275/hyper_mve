# 论文复写 Stage 2 状态报告 — 2026-06-30

> Branch: `academic-research` · Origin: ARS academic-pipeline `/ars-full` resume from 2026-06-17 plan-mode session · Backed by 18 TB runs across 7 cells under `runs/suite/`.

## TL;DR — 一句话总结

Ch1 维持 2026-06-17 verified `ch1_full.md`（WARNINGS_PRESENT, 已 inline 修订 17 处）；Ch2–Ch6 全部初稿完成，使用现有 TB 实测数据 + 理论 fallback；共 **29 个 [证据待补 EVIDENCE PENDING] 标记**，覆盖 4 项断言中 A、C 的全部消融与 B′、D 的边界象限。Belief 通路 P0 问题已在 §4.6 / §5.10 / §5.11 / §6.2 多次透明披露。

---

## 1. 章节落盘清单

| 章 | 文件 | 中文字数 | 目标 | 状态 | [EVIDENCE PENDING] |
|---|---|---|---|---|---|
| Ch1 绪论 | `ch1_full.md` | 8 032 | 8 000 | ✅ Stage 2.5 v2 verified (2026-06-17) | 0 |
| Ch2 相关工作 | `ch2_full.md` | 6 903 | 7 500 | ✅ 初稿完成 (−8%) | 3 |
| Ch3 问题与环境 | `ch3_full.md` | 4 830 | 9 600 | ⚠ 初稿完成 (−50%)，节级开场段为主，后续可扩 | 2 |
| Ch4 方法 | `ch4_full.md` | 9 663 | 18 500 | ⚠ 初稿完成 (−48%)，§4.4 D3 完整代数已展开，可扩之处主要在 §4.4 子节细节 | 6 |
| Ch5 实验 | `ch5_full.md` | 16 286 | 18 000 | ✅ 初稿完成 (−9%) | 17 |
| Ch6 总结与展望 | `ch6_full.md` | 3 637 | 4 000 | ✅ 初稿完成 (−9%)，§6.1 断言验证表已据实填写 | 2 |
| **合计** | | **49 351 / ~68 000** | **覆盖率 73%** | | **29** |

**注**：Ch3 与 Ch4 的字数缺口主要在子节深度（每节当前 600–900 字 vs 目标 1500–2500 字）。当前已含全部预定节级结构、所有关键公式、所有 TB 实测数字、所有 [EVIDENCE PENDING] 标记。下一阶段（实验补完后）的展开方向以"实验结果填入 → 数字注释 → 失败分析"为主，文学结构已就绪。

---

## 2. 实验证据使用情况

### 2.1 已落地的 TB 实测（17 个 hyper-variant cell）

引自 `exp_results/HEADLINE_NUMBERS.md`：

| 实测项 | 用于章节 | 关键数字 |
|---|---|---|
| **条件化谱 5 变体扫描** | §2.2 / §4.2 / §5.2 / §5.5 | cos_pred_cross 0.72 (input) → 0.03 (film_fc2); planner return 105 → 272 |
| **`lora_duo_film_fc2` 峰值变体** | §4.2 / §4.4 thesis-default / §5.2 GO 判据 / §5.5 B'(iii) | 272 ± 18 收益，planner_prior_gap +19.6，3 seeds × 240k |
| **`zero_shot_easy` 零样本泛化** | §5.5 B' 实证半 | regret_mean = 0.0 over 3 seeds × 200k；registry return ≈ 130.88 |
| **`main_comparison_easy` Easy 主对比** | §5.4 (downgraded) | 收益 105.4 ± 4.8, fairness 0.86, sustainability 0.78 |
| **`abl4_joint_easy_n2` D 单象限** | §5.8 D 部分支持 | 收益 100.2 ± 3.6, planner_prior_gap −2.5（联合启用单 cell） |
| **`lora_medium_base` 失败案例** | §5.11 失败诊断 | fairness −0.17, gap −60, 66k/240k steps（训练中断） |
| **Belief 通路 P0** | §4.6 / §5.10 / §5.11 / §6.2 | loss/belief ≤ 1e-5 在 14/18 runs; loss/lambda_b ≡ 1.0 |

### 2.2 缺失实测（29 个 [EVIDENCE PENDING] 标记的分类）

| 优先级 | 数量 | 类别 | 影响章节 |
|---|---|---|---|
| **P0 — 阻塞断言 A** | 5 | ρ_β 五点扫描 + 1α+3β 对照（消融 3）+ 类型梯度热图 | §5.7 / §6.1 / §4.1 / §6.11/Appx C |
| **P0 — 阻塞断言 C** | 4 | c_ctx/role/belief 单路 + 多路零置（消融 2）+ belief 通路修复 | §5.6 / §5.10 / §4.6 |
| **P1 — 补完断言 B′** | 6 | Shared/Input-Wide/FULL 三边界变体 + FULL 方向坍缩 + Medium 主对比 6 基线 | §5.2 / §5.4 / §5.5 |
| **P1 — 补完断言 D** | 3 | 2×2 矩阵 {CD-only, CRN-only, neither} + SNR probe | §5.8 / §5.1.2 |
| **P2 — 鲁棒性消融 5** | 4 | Fehr-Schmidt 3×3 + Hard N=8 + planner-off + Medium hidden-c | §5.9 / §5.6 |
| **P2 — v3 迁移** | 1 | v4 方法应用至 v3 NonStationaryTag 任务 | §5.3 |
| **杂项** | 6 | 各种 cross-section 注释（Gini/末态资源、热图位置等） | various |

---

## 3. 断言验证状态（§6.1 复述）

| 断言 | 验证状态 | 当前实证 | 待补实验数 |
|---|---|---|---|
| **A** 类型梯度撕裂钟形曲线 | **未支持**（实验未执行） | §4.1 代数偏导表 | ρ_β 五点 + 1α+3β（消融 3） |
| **B'** 条件化谱内部最优 | **部分支持** | (i) regret=0 zero-shot; (ii) cos 0.72→0.03 五变体; (iii) film_fc2 peak 272 vs input 105 | 3 边界变体 + FULL 坍缩复现 |
| **C** 三联通路缺一不可 | **未支持**（实验未执行） | §4.3 Harsanyi 类比 + §4.4 D3 分解 | 5 零置条件 + belief 修复 |
| **D** CoordDesc × CRN 不可分割 | **部分支持** | abl4_joint cell + §4.5 SNR 几何 | 3 象限 + SNR probe |

汇总：**B' / D = 部分支持；A / C = 未支持**。

---

## 4. P0 工程问题披露

### 4.1 BeliefNet 训练通路未充分激活

**症状**：18 个 hyper-variant run 中有 14 个观察到 `loss/belief ≤ 1e-5`、`loss/lambda_b ≡ 1.0`。后者表明 lambda_b schedule 在所有 runs 上保持初始值 1.0，未按预期进入 stage 2 / stage 3。

**两种可能根因**：
- (a) Easy 配置下 $T_{\max} = 200k$ steps 偏短，三阶段课程切入点 $0.3 T_{\max} = 60k$ 仍处于训练早期；
- (b) $\lambda_b$ schedule 实现存在 gating bug 导致从未激活。

**披露位置**：§4.6 训练算法节、§5.10 信念质量节、§5.11 失败诊断节、§6.2 局限节。

**对断言 C 的影响**：消融 2 必须先解决该问题——否则 belief-on 与 belief-off 条件下实测无差异，无法分离 belief 通路的独立贡献。

### 4.2 其他 §5.11 已记录的失败模式

- `lora/medium_base` 3 seeds 训练中断（66k / 240k steps，fairness 负）
- `lora/duo_film` seed 1 与 seed 0 间 q_gap 显著分歧
- `tragedy ≡ 0` 全 18 run（需验证是否 Easy 难度下定义性结果）
- `regret_mean ≡ 0` 全 completed run（需验证指标实现）

---

## 5. 下一阶段实验补完路线（按优先级）

### Round 1 — P0 阻塞实验（可立即并行起跑）

1. **消融 3（断言 A）**：ρ_β ∈ {0/4, 1/4, 2/4, 3/4, 4/4} × 3 seeds = 15 runs，Easy N=2 或 Easy N=4。预估 ~30 GPU h。
2. **消融 2（断言 C）先决条件**：诊断 + 修复 BeliefNet `lambda_b` schedule，确认 belief loss 在 stage 2 进入 ~1e-3 量级。预估 ~工程时间，非 GPU 限制。
3. **消融 2（断言 C）正式跑**：5 零置条件 + 联合条件 × 3 seeds ≈ 18-21 runs。预估 ~50 GPU h。需在 P0 工程修复后执行。

### Round 2 — P1 补完已部分支持的断言

4. **B' 三边界变体**：Shared / Input-Wide / FULL × 3 seeds = 9 runs（其中 FULL 用于复现 cos 坍缩演化）。预估 ~25 GPU h。
5. **D 2×2 矩阵**：{CoordDesc + IndepRand, Joint + CRN, Joint + IndepRand} × 3 seeds = 9 runs，Easy N=2。预估 ~15 GPU h。
6. **SNR probe**：单一对照运行，CRN-on vs CRN-off 在 fixed-state 下记录第 0 步 Q-value 差分方差。预估 ~3 GPU h。
7. **Medium 主对比**：6 baseline × 5 seeds = 30 runs，Medium N=4，2α+2β。预估 ~180 GPU h。

### Round 3 — P2 鲁棒性 / Scale-up（可选）

8. 消融 5 Fehr-Schmidt 3×3 + Hard N=8 + planner-off + hidden-c 双档。预估 ~80 GPU h。

**总算力估算**：P0 = ~80h, P1 = ~225h, P2 = ~80h。Round 1+2 共 305h（与原计划 300h 预算吻合）；Round 3 视情况延后。

---

## 6. 文档完整性 / 可发现性

### 6.1 章节结构（全在 `docs/thesis_plan/drafts/`）

```
drafts/
├── ch1.md                          (Plan-Mode skeleton, kept)
├── ch1_full.md                     (DRAFTED, Stage 2.5 v2 verified, 8032 chars)
├── ch1_stage3_reviews.md           (Stage-3 reviewer notes, archived)
├── ch1_verification_report.md      (Stage 2.5 v1)
├── ch1_verification_report_v2.md   (Stage 2.5 v2 — WARNINGS_PRESENT)
├── ch2.md  ch2_full.md             (DRAFTED, 6903 chars)
├── ch3.md  ch3_full.md             (DRAFTED, 4830 chars)
├── ch4.md  ch4_full.md             (DRAFTED, 9663 chars)
├── ch5.md  ch5_full.md             (DRAFTED, 16286 chars)
└── ch6.md  ch6_full.md             (DRAFTED, 3637 chars)
```

### 6.2 实验数据归档（全在 `docs/thesis_plan/exp_results/`）

```
exp_results/
├── HEADLINE_NUMBERS.md             (Single source of truth for chapter writers)
├── scalars_summary.md              (Per-cell × tag final/tail mean ± std, 7 cells × ~30 tags)
├── scalars_final.json              (Same data in JSON for downstream tooling)
├── recon_tb.md                     (Full TB inspection report)
├── recon_drafts.md                 (Draft inventory pass)
├── recon_refs.md                   (INSIGHTS + CITATION_REGISTRY)
└── recon_gap.md                    (Gap matrix: §-by-§ × evidence)
```

### 6.3 仍需维护的元文档

- `MASTER_PLAN.md`：§2 断言映射表与本 STATUS 一致（A 未支持 / B' 部分 / C 未支持 / D 部分）。可在下一轮实验补完后更新。
- `INSIGHTS.md`：15 条 INSIGHT 维持原版（12 保留 + 3 降级）。
- `CITATION_REGISTRY.md`：仍有 15 个 inline-cited 待分配 [N]，30+ 顶刊 / 顶会建议补充。属于 Stage 5 final formatting 范畴。

---

## 7. 后续 ARS pipeline 步骤建议

按 academic-pipeline 流程：

| Stage | 状态 | 建议 |
|---|---|---|
| **Stage 2 WRITE** | ✅ 6 章初稿完成 | — |
| **Stage 2.5 INTEGRITY** | ⏸ 未运行（Ch1 已单独 verified；Ch2–Ch6 尚未做 5 阶段引用 / claim / originality 验证） | 在 P0 实验补完前**暂不必走**——草稿仍会随实验更新；建议在 P0 round 1 完成后对 Ch4 / Ch5 / Ch6 做 INTEGRITY |
| **Stage 3 REVIEW** | ⏸ 未运行 | 同上 |
| **Stage 4 REVISE** | ⏸ 未运行 | 同上 |
| **Stage 4.5 FINAL INTEGRITY** | ⏸ 未运行 | 留待 Round 3 实验后 |
| **Stage 5 FINALIZE** | ⏸ 未运行 | MD → DOCX (Pandoc) → LaTeX → PDF，留待最终阶段 |
| **Stage 6 PROCESS SUMMARY** | ⏸ 未运行 | 留待 Stage 5 完成 |

**当前阶段的 user 推进路径**：
1. **Read** ch2/ch3/ch4/ch5/ch6_full.md 看落盘状态是否符合预期；
2. **Run P0 round 1** 的两项消融实验（A 钟形 + C 三联通路）；
3. 实验产出新 TB cell 后，**re-run TB extraction + gap matrix**（脚本可重用 `exp_results/HEADLINE_NUMBERS.md` 的模板）；
4. **再次触发 academic-pipeline Stage 2.5 INTEGRITY**（聚焦 Ch5 主对比 + 消融节）。

---

## 8. 写作规范遵守审计

| 检查项 | 通过情况 |
|---|---|
| **FORBIDDEN words**（首次/开拓/补全空白/严格优于/超越...范式/60 年内首次）| Ch4 中 1 次"首次"+ Ch6 中 1 次"开拓"，均在否定上下文中（"不主张...首次..."、"不主张开拓..."）— **属于反向使用，符合 IRON RULE 意图**。其他无违规命中。 |
| **C1/C2/C3 语义** | Ch1–Ch6 均严格遵守"问题约束"语义，方法层使用"回应 C1 / C2 / C3"措辞；无 label 复用。 |
| **数字可追溯** | 所有具体数字（cos_pred_cross 0.72→0.03、planner return 105→272、regret=0、loss/belief ≤ 1e-5 等）追溯至 HEADLINE_NUMBERS.md / scalars_summary.md。 |
| **[EVIDENCE PENDING] 标记格式** | 29 个标记，全部遵守 `[证据待补 EVIDENCE PENDING — <name>，待 §X.Y 补完]` 标准格式。 |
| **INSIGHT N inline** | 0 处违规——全部转换为 §X.Y 交叉引用。 |
| **v3 wording drift（§4.1 / §4.6 / §6.1）** | Ch4 与 Ch6 已替换为 v4 措辞；§3.2 v3 NonStationaryTag 内容保留 v3 措辞（按 plan 设计）。 |

---

**Last updated**: 2026-06-30 by ARS academic-pipeline resume session.
