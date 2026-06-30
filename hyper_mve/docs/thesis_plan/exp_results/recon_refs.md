I have everything needed. Compiling the inventory.

---

# INSIGHTS.md inventory (15 total: 12 retained + 3 demoted)

## Retained (12)

| ID | One-line statement | Deployment (§) |
|---|---|---|
| **I2** | Psychological preferences are immutable; cooperation/exploitation emerges from dynamics, not reward shaping. | §3.4, §3.5, §4.3, §6.4 |
| **I4** | v3→v4 evolution narrative: v3 validated "hypernet conditions game perspective"; v4 extends to mixed-motive preference heterogeneity. | §3.2, §5.3, §6.4, §1.6 |
| **I5** | CRN is a feasibility *necessity* for multi-agent MVE (lifts step-0 SNR 0.02→1.0); planner-off self-distillation collapses π to uniform (ln A fixed point). | §4.5, §5.8, §5.9, §1.4 |
| **I6** | Per-Agent Coordinate Descent's fixed point ≡ ε-Nash equilibrium — algorithm choice aligns with game-theoretic equilibrium concept. | §4.5, §2.2, §6.1 |
| **I7** | DualHyperNetwork ≡ structural isomorphism with RNS-MMG C1/C2 constraints (objective physics / subjective preference). | §1.4, §3.1, §4.4 |
| **I9** | Instantaneous Δ design makes physical and Fehr-Schmidt terms both fall in [0, 1.5] → no reward normalization needed. | §3.5, §4.6, §5.11 |
| **I10** | v4 removes β(c) cooperative bonus → "strictly self-interested physics + explicit preference heterogeneity + emergent game"; precondition for Prop 3.1. | §3.4, §1.2, §3.6 |
| **I11** | Self Info observation setting ≡ Harsanyi "own type + belief over others"; BeliefNet corresponds to Harsanyi framework, not engineering convenience. | §3.5, §4.3, §4.4 |
| **I12** | Pre-registered falsification fallback plans elevate thesis from "our method is good" to "our method is falsifiable and reproducible". | §1.5, §5.1, per-ablation, §6.4 |
| **I13** | Gap(c) monotonicity is true emergence from two orthogonal channels (resource dynamics × type-β preferences), not tautology — gives Prop 3.1 substantive content. | §3.6, §1.1, §6.1 |
| **I14** | Reward Gradient Quantification Table (type α=1 const vs type β ∈ {0,0.7,1.3,2.0}) hardens "type gradient tearing" into algebraic necessity for Ablation 3's bell curve. | §4.1.1, §5.7, §6.11 |
| **I15** | μP learning-rate alignment protocol (Yang & Hu 2021) systematically defeats "baseline wasn't tuned" attack — fairness written into protocol. | §5.1, §1.4, §5.4, intro |

## Demoted (3)

| ID | Original → Rewrite | Deployment |
|---|---|---|
| **~~I1~~** | "Harsanyi 60-year first architectural correspondence" → "borrow Harsanyi framework to explain architecture choice". NOT in §1.4 contributions. | §4.3 only |
| **~~I3~~** | "Conditioning-spectrum upgrades the research question" → "systematic comparison along conditioning spectrum". FULL collapse 0.61→0.998 kept as engineering observation. | §2.3, §4.2, §5.5 |
| **~~I8~~** | "BeliefNet Oracle supervision improvement" → folded into routine engineering note (200k→50k steps). Not standalone INSIGHT. | §4.4 |

**Tone rules (Q4/Q5):** Forbidden — "首次/开拓/60 年内首次/补全空白". Permitted — "本文给出/本文借用 X 框架解释/与 X 在结构上类似".

---

# CITATION_REGISTRY.md inventory

## Total [N] entries

- **Registered numbered slots:** 19 entries with ID span up to **[67]** (46 unused IDs in gaps).
- **Inline-cited but un-numbered, to be added:** **15 items** total — 11 assigned in this pass (✓ in Ch1), 4 still "待分配" (LoRA / μP / DiT / StyleGAN).
- **Recommended supplementary refs:** 30+ across §1–§6.
- **Final projected registry size:** **30–40 entries** after renumbering 1..M per GB/T 7714-2015.

## Inline [N] citations actually appearing in ch1_full.md

Unique [N] numbers found (lines 11–159):
**[3], [4], [5], [6], [7], [8], [9], [10], [11], [13], [14], [15], [16], [17], [18], [19], [20], [21], [24], [30], [40], [45]**

(22 distinct citation keys; [11] and [15] are the most frequent.)

## "44 missing top-venue references" gap

The registry does **not** use the phrase "44 missing top-venue references" verbatim. Closest equivalents:

- **46 unused ID slots** in the [1]–[67] range (gaps: [1]–[8], [12], [17]–[23], [25]–[29], [31]–[39], [41], [46]–[48], [52]–[57], [59]–[66]) — these are renumbering placeholders, not missing refs.
- **15 inline-cited items** flagged as needing registry entries before submission (§2 of registry).
- **30+ supplementary refs** listed in §3 by chapter (§1: 8 items; §2: 10; §4: 8; §5: 1; §6: 2).

The "gap" is the union of these two lists (≈45 entries) — the **15 must-add inline cites + 30+ recommended top-venue supplements** that, taken together, bring the registry from 19 → ~40 final entries. No single section labels this as "44 missing references"; it is reconstructed from §2 + §3.

## Key venue corrections (must fix pre-submission)

[11] Hughes → **Inequity Aversion, NeurIPS 2018**; [14] Leibo 2021 → **ICML 2021**; [15] Leibo 2017 → **AAMAS 2017 (Sequential Social Dilemmas)**; [24] MuZero → **Schrittwieser et al., Nature 2020**; [40] Fehr-Schmidt → **QJE 114(3), 1999** (not JPE); [42] Ha → **ICLR 2017**; [67] MVE → **Feinberg et al., arXiv:1803.00101**.

Two TODOs: **[30] MA-MuZero** author/venue unconfirmed; **[13] CAGA 2025 (Kim et al.)** existence/venue unverified.

---

**Source files (absolute paths):**
- `D:\RL\hyper_mve\docs\thesis_plan\INSIGHTS.md`
- `D:\RL\hyper_mve\docs\thesis_plan\CITATION_REGISTRY.md`
- `D:\RL\hyper_mve\docs\thesis_plan\drafts\ch1_full.md` (scanned for inline [N])