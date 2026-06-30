# Headline Numbers — Single Source of Truth for Chapter Drafts

> Authoritative final-step / tail-mean values for thesis writing.
> Source: `scalars_summary.md` + `scalars_final.json` (TB EventAccumulator).
> Registry-backfilled metrics in zero_shot_easy + abl4_joint cells.

## Cell ↔ § map for writers
| Cell | Seeds | Steps | Primary § | Headline result |
|---|---|---|---|---|
| `main_comparison_easy` | 3 | 137k | §5.4 (Easy main) | planner return 105.4 ± 4.8, fairness 0.86, sustainability 0.78 |
| `zero_shot_easy` | 3 (completed) | 200k | §5.5 zero-shot half | registry return_zero_shot_unseen ≈ 130.88; TB planner final 104 ± 3.3, regret_mean = 0.0 |
| `abl4_joint_easy_n2` | 3 (completed) | 100k | §5.8 (joint cell of 2×2) | planner return 100.2 ± 3.6, planner_prior_gap = −2.5 |
| `lora/duo_base` | 1 | 189k | §5.2 / §5.5 spectrum | planner return 263, cos_pred_cross 0.08 |
| `lora/duo_film` | 2 | 240k | §5.2 / §5.5 spectrum (FiLM head only) | planner return 250 ± 2.3, cos_pred_cross 0.09 |
| `lora/duo_film_fc2` | 3 | 240k | §5.2 / §5.5 spectrum (FiLM + LoRA fc2) | **planner return 272 ± 18**, **cos_pred_cross 0.03**, planner_prior_gap +19.6 ± 6.7 (MVE adds ~8%) |
| `lora/medium_base` | 3 | 66k FAILED | — | fairness −0.17, gap −60 → § 5.11 failure case |

## Conditioning-spectrum table (§4.2 / §5.2 / §5.5) — VARIANTS WE HAVE
| Variant (gen_scope) | n_seeds | planner return | cos_pred_cross | π_mve entropy | Notes |
|---|---|---|---|---|---|
| `main_comparison_easy` (input-conditioning baseline) | 3 | 105 ± 5 | 0.72 ± 0.02 | 1.37 | High pred cross → little perspective separation; baseline of spectrum |
| `zero_shot_easy` (same conditioning, longer training) | 3 | 104 ± 3 | 0.71 ± 0.01 | 1.37 | Confirms 105-tier ceiling without LoRA |
| `lora_duo_base` (LoRA only, no FiLM) | 1 | 263 | 0.08 | 0.72 | π more decisive |
| `lora_duo_film` (FiLM head only) | 2 | 250 ± 2 | 0.09 | 0.77 | Comparable to LoRA base |
| `lora_duo_film_fc2` (FiLM + LoRA on fc2) | 3 | **272 ± 18** | **0.03** | 0.75 | **Peak variant — supports B'(iii) interior optimum** |
| `lora_medium_base` (Medium scaling attempt) | 3 (66k) | 68 ± 36 (FAILED) | 0.21 | 1.40 | §5.11 failure case |

## Assertion verification status (§6.1)
| Assertion | Verdict | Evidence available | Evidence missing |
|---|---|---|---|
| A (type-gradient tearing, bell curve) | **未支持(实验未执行)** | — | ρ_β scan ablation (0/4 .. 4/4) — not in suite |
| B′ (spectrum internal optimum) | **部分支持** | cos_pred_cross 0.72→0.03 across spectrum; lora_duo_film_fc2 = peak (272 return) > input variants (105); zero-shot regret = 0.0 over 3 seeds | 3 of 7 gen_scope variants missing; right-end (FULL) collapse not directly demonstrated in current suite |
| C (triple-pathway necessity) | **未支持(实验未执行)** | — | c_ctx/role/belief zero-out ablation; **belief loss ≡ 0 in 14/18 runs ⇒ belief pathway dormant** |
| D (CoordDesc × CRN inseparability, SNR 0.02→1.0) | **部分支持** | abl4_joint cell (3 seeds) running with both technologies on | {CD-only, CRN-only, neither} cells; explicit SNR probe |

## Belief pipeline P0 issue
- `loss/belief ≤ 1e-5` in 14 of 18 runs (only lora_medium_base shows ≈ 1e-3, and that cell failed).
- `loss/lambda_b ≡ 1.0` everywhere → belief lambda was NOT activated.
- Implication: §5.10 belief-quality protocol **cannot** be claimed positively from current data; must either disclose belief pipeline was inactive (curriculum stage not reached?) or fold into §5.11 failure case as honest limitation.

## Recommended language for [EVIDENCE PENDING] in chapter drafts
Use this exact marker format for any forward-looking claim without TB support:
> [证据待补 EVIDENCE PENDING — <experiment name>，待 §X.Y 补完]

Examples:
- `[证据待补 EVIDENCE PENDING — ρ_β 五点扫描 (消融 3)，待 §5.7 补完]`
- `[证据待补 EVIDENCE PENDING — 三联通路零置消融 (消融 2)，待 §5.6 补完]`
- `[证据待补 EVIDENCE PENDING — planner-off self-distillation collapse，待 §5.9 补完]`
- `[证据待补 EVIDENCE PENDING — CRN-off SNR 显式测量，待 §5.1.2 补完]`
- `[证据待补 EVIDENCE PENDING — 类型梯度 ∂ř/∂u_i 热图，待 §6.11/附录 C 决定位置后补完]`
