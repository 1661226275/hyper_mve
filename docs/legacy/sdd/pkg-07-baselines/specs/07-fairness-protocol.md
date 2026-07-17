# Spec 07 — Two-Tier Fairness Protocol (Internal Strict 5%/10% Equal-Param + External Disclosure-Style)

> **Anchors**: design.md §4 D5 (双层 fairness — Internal 严格继承 pkg-06 D5 + External 披露式) · design.md §4 D7 (Tier-1 external selection + adapter contract + LR-sweep grid table source) · design.md §4 D10 (`cfg.baselines.external_lr_sweep_grid: Mapping[str, tuple[float, ...]]` + `internal_wide_hidden_dim` / `internal_deep_layers` / `internal_ma_muzero_share_pred_head` / `internal_explicit_type_branches` — 5 perms locked) · README §"v4 关键约束" C7-INT-FAIR1 (≤5% warn / ≤10% fail on conditioned subsystem) · README C7-INT-FAIR2 (RepNet/BeliefNet/TriCtx 跨 internal variant 参数量逐位相同) · README C7-INT-PERF1 (input baseline 单步 forward ≤ 2.0× hyper) · README C7-EXT-FAIR1 (external LR sweep ≥3 LR × ≥3 seeds, results 披露) · pkg-06 spec 07 (`07-param-fairness-and-lr-sweep.md` — supersede source of the 5%/10% double-threshold rule).
> **Status**: SDD only — describes the contract for the internal fairness gate set (consumed at `tests/baselines/internal/test_param_fairness.py` + `tests/baselines/internal/test_perf.py`) and the external disclosure-table contract (consumed by pkg-08 spec 07 stats reduction + pkg-08 spec 05 sweep harness enumeration). No production code is added by this spec; it locks the contract surface.
> **Cross-refs**: spec 02 (`02-shared-backbones-internal.md` — bit-exact `count_conditioning_params` enforcement site for C7-INT-FAIR2) · spec 03 (`03-internal-variants.md` — 5 internal variant conditioned-subsystem account) · spec 05 (`05-external-mappo.md` §9 LR-sweep contract + §11 per-impl tuning constants — MAPPO disclosure feed) · spec 06 (`06-external-qmix-mamuzero-mamba.md` — QMIX / MA-MuZero-GH / MAMBA Tier-2 disclosure feed + MARIE/GA stub footnote) · spec 08 (`08-integration-contracts.md` — `cfg.baselines.external_lr_sweep_grid` declaration site + BaselinesConfig 5-field 穷举) · pkg-08 spec 05 (`05-sweep-harness-and-run-registry.md` — sweep enumeration over (variant, lr, seed)) · pkg-08 spec 07 (stats reduction → disclosure markdown) · pkg-08 spec 08 (RunRegistry row schema must carry `param_count` / `walltime_to_converge_seconds` / `lr`).

---

## ⚠️ Header — three hard locks

### Lock 1 — Internal fairness is dual-axis; spec 07 owns the comparison test, not the bit-exact lock

The internal-vs-hyper fairness protocol has **two independent axes** and each axis is enforced at a different site:

- **Axis A — Bit-exact shared-backbone equal-param (C7-INT-FAIR2).** RepNet / BeliefNet / TriContextEncoder participation counts must be **byte-identical** across the 5 internal variants AND vs hyper. This is the structural prerequisite for any conditioned-subsystem fairness claim to be meaningful (per spec 02 §7.1 causality diagram: "no backbone equal-param → 5%/10% check is meaningless"). The bit-exact lock is enforced at construction by `tests/baselines/internal/test_shared_backbones.py::test_shared_backbone_param_count_bitexact` (spec 02 §5.1). **Spec 07 does not redundantly re-test this**; it only declares the cross-variant comparison invariant for completeness and points to spec 02 for the actual test.
- **Axis B — Conditioned-subsystem soft envelope (C7-INT-FAIR1).** Predictor heads + reward heads + (BeliefNet output projection IF the variant has hypernet integration) — the parameter buckets that are intentionally varied across the 5 internal variants — must fall inside the 5% / 10% double-threshold envelope around hyper. This is the soft fairness gate; it allows structural exemption for variants whose conditioned subsystem is structurally different from hyper's (per §2.4 capacity-tuning knobs + pkg-06 §2.3 D5 exemption inheritance). **Spec 07 owns this axis** at `tests/baselines/internal/test_param_fairness.py::test_internal_param_fairness_5pct_warn_10pct_fail`.

The two-axis separation is mandatory: Axis A is binary (bit-exact or fail), Axis B is graded (PASS / WARN / FAIL). Conflating them at one test site loses information (a WARN on Axis B is meaningful diagnostic; a WARN on Axis A is undefined).

### Lock 2 — External fairness is DISCLOSURE-only; no equal-param attempt

External Tier-1 runners (MAPPO, QMIX, MA-MuZero-GH) + MAMBA Tier-2 have their own architectures (Actor+Critic / RNN-agent+Mixer / Dynamics+Reward+PredictionNet), their own optimizers, their own training loops. **Strict equal-param is impossible** (you cannot make a centralized PPO critic have the same parameter count as a state-dependent QMixer without distorting one or both) **and conceptually inappropriate** (MARL paper norms — MAPPO Yu et al. 2022 / QMIX Rashid et al. 2018 / muzero-general — consistently report params + walltime + LR-best, not equal-param comparisons). The disclosure table (§4.2 schema, locked here) is the contract; the table reports 5 axes per external variant: (i) `param_count`, (ii) `walltime_to_converge_seconds`, (iii) `lr_swept_best`, (iv) `final_return_per_seed` (≥5 seeds), (v) `seeds_run`. **Any drift from the §4.2 schema is a synchronous spec 07 + spec 05 + spec 06 edit** — reviewers can rely on the disclosure table's column set being stable.

### Lock 3 — LR-sweep grid is sourced exclusively from `cfg.baselines.external_lr_sweep_grid`

The LR sweep grid is declared in `cfg.baselines.external_lr_sweep_grid: MappingProxyType[str, tuple[float, ...]]` (declared by spec 08 §3, consumed by spec 05 §9.1 + spec 06 §2.8 / §3.8 / §4.7 / §5.4) with ≥3 LR values per Tier-1 variant + MAMBA-if-sourced. The grid is enumerated by pkg-08 spec 05 sweep harness; 3 LRs × 5 seeds = 15 runs per (variant, preset). **5-seed minimum is reported** as `seeds_run` in the disclosure table; rows with `seeds_run < 5` emit `[WARN seeds_run=<n><5]` in the rendered markdown but **do not block** pkg-07 finalize (README C7-EXT-FAIR1 baseline is ≥3 seeds; spec 07 strengthens to 5 as a soft target). Hard-coding the LR grid anywhere in `qmix.py` / `mappo.py` / `ma_muzero_gh.py` / `mamba.py` / `compare.py` / `sweep_harness.py` is a contract violation; the grid must always flow through `cfg.baselines.external_lr_sweep_grid["external_<variant>"]`.

---

## 1. Purpose

This spec is the **canonical home** for both fairness protocols pkg-07 needs to ship and the supersede target for pkg-06 spec 07. Why two tiers:

1. **Internal variants share the v4.7 backbone.** All 5 internal variants (input_wide, input_deep, ma_muzero, no_belief, rewardhead_explicit_type) consume the same RepNet / BeliefNet / TriContextEncoder factories from spec 02. The cross-variant difference is concentrated in the conditioned subsystem (predictor heads, reward heads, BeliefNet projection routing). Because the structural locus of difference is narrow and well-defined, **strict equal-param under the 5%/10% double-threshold rule is meaningful** — it isolates the "is the conditioned subsystem actually doing the work, or is it just parameter count?" question to the same single variable.
2. **External Tier-1 runners have their own architectures.** MAPPO is centralized critic + decentralized actors with PPO loss; QMIX is shared GRU + state-dependent QMixer with TD loss; MA-MuZero-GH is muzero-general's dynamics + reward + prediction networks + MCTS planner with model-based loss. There is no shared "conditioned subsystem" between hyper and MAPPO — there's no factor of the architecture that can be aligned without distorting the algorithm's identity. The standard MARL paper practice (paraphrased: report params + walltime + LR-best + 5+ seeds, let reviewers judge) is **disclosure-style** fairness. Spec 07 codifies this as a public 10-column table whose schema cannot drift.
3. **One spec, two protocols.** Splitting internal and external into two specs would (a) duplicate the "what's the convergence threshold" + "how is seeds_run computed" boilerplate and (b) lose the single review surface that lets a reviewer see "internal is strict, external is disclosure" side by side. Pkg-06 spec 07 chose the single-spec convention (its §2-3 internal + §3-4 LR sweep) and pkg-07 spec 07 inherits that convention, augmented for the supersede.

Spec 07 supersedes pkg-06 spec 07 by **absorbing the 5%/10% double-threshold rule verbatim** (§2 below) and **extending the LR-sweep protocol** (§4 below) from "LR sweep over the same grid for hyper + 5 internal" to "external-only LR sweep over per-variant grids declared in cfg.baselines + disclosure-style fairness reporting at the §4.2 table".

---

## 2. Internal strict equal-param protocol (5%/10% double threshold)

### 2.1 What counts as the "conditioned subsystem"

The conditioned subsystem (the parameter buckets that are intentionally varied across the 5 internal variants and against hyper) is defined per pkg-06 C6-FAIR1 + spec 02 §3.3 design D5 account. It includes:

- **θ generators (hypernet heads, if any)**: `hyper_trans` / `hyper_rew` / `hyper_pred` for variants that retain them.
- **Functional nets that consume `ctx_aug`**: `StateTransNet` / `RewardHead` / `PredictionNet`. For `input_wide` / `input_deep`, these are the widened / deepened MLP versions (no hypernet head, concat-rule-embed-to-input).
- **BeliefNet output projection (IF the variant has hypernet integration)**: for `no_belief`, the projection is structurally present in the model but the value path is zeroed at `set_context_subjective` (spec 02 §3.2 SB2 keep-full-BeliefNet rule); the projection parameters still count under the conditioned-subsystem total because they are downstream of BeliefNet output.

The buckets are computed by `hyper_mve.baselines.shared_backbones.count_conditioning_params(model: nn.Module) -> int` (spec 02 §2.2). The function is the **single source of truth** — spec 07 does not duplicate the computation; it only references it. Per pkg-06 §2.1 (inherited verbatim): the count **excludes** the shared backbone (RepNet + BeliefNet body + TriContextEncoder), which are separately bit-exact-checked by spec 02.

The per-variant parameter buckets (design D5 account, locked here for grep stability):

| variant | counted into `count_conditioning_params` |
|---------|------------------------------------------|
| `hyper` (reference) | `hyper_trans` + `hyper_rew` + `hyper_pred` + 3 vanilla functional nets |
| `input_wide` | 3 **widened** functional nets (no hypernet heads) |
| `input_deep` | 3 **deepened** functional nets (no hypernet heads) |
| `ma_muzero` | 3 vanilla functional nets (**single shared** RewardHead across all N agents; no hypernet heads) |
| `no_belief` | `hyper_trans` + `hyper_rew` + `hyper_pred` + 3 vanilla functional nets (same as hyper; difference is BeliefNet-info zeroing, not params) |
| `rewardhead_explicit_type` | `hyper_trans` + `hyper_pred` (**no** `hyper_rew`) + type-branched RewardHead (one head per agent type, see §2.4 `cfg.baselines.internal_explicit_type_branches`) + 2 functional nets |

### 2.2 Reference variant = hyper

The reference variant for the 5%/10% comparison is `hyper` (Exp2 Oracle-HyperMuZero; pkg-07 §3.2 11-key 工厂矩阵 row 1). `P_hyper = count_conditioning_params(HyperMuZeroModel(cfg))` is computed once per (preset, config_hash) pair and used as the denominator for the relative-delta computation.

The 5 internal variants compared against hyper:

1. **`input_wide`** — input-conditioned wide MLP (concat-rule-embed-to-input). Capacity knob: `cfg.baselines.internal_wide_hidden_dim` (§2.4).
2. **`input_deep`** — input-conditioned deep MLP (concat-rule-embed-to-input). Capacity knob: `cfg.baselines.internal_deep_layers` (§2.4).
3. **`ma_muzero`** — single shared RewardHead across all agents (no per-agent RewardHead). Capacity knob: `cfg.baselines.internal_ma_muzero_share_pred_head` (§2.4).
4. **`no_belief`** — no BeliefNet (no prefix; design D7 exemption: BeliefNet body retained for bit-exact backbone, but belief value path zeroed at `set_context_subjective` per spec 02 §3.2).
5. **`rewardhead_explicit_type`** — discrete-type-conditioned RewardHead (one head per agent type α/β). Capacity knob: `cfg.baselines.internal_explicit_type_branches` (§2.4).

### 2.3 Threshold rule (verbatim from pkg-06 §2.2 C6-FAIR1)

For each cross-variant pair (`hyper`, `variant`) of conditioned subsystems, compute the relative parameter-count delta:

```
P_hyper = count_conditioning_params(HyperMuZeroModel(cfg))
P_v     = count_conditioning_params(create_baseline(cfg, variant))
delta   = abs(P_v - P_hyper) / P_hyper
```

Then:

| `delta` | Judgement | Action |
|---------|-----------|--------|
| `delta ≤ 0.05` | **PASS** | No action; the test passes silently. |
| `0.05 < delta ≤ 0.10` | **WARN** | The test emits `warnings.warn(..., UserWarning)` with the message `f"{variant} 条件化参数偏差 {delta:.1%} 落在 5%~10% WARN 带 (调小旋钮逼近)"`. The PR description must mention the WARN; the test continues to PASS (no assertion failure). |
| `delta > 0.10` | **FAIL** | The test calls `pytest.fail(f"{variant} 条件化参数偏差 {delta:.1%} > 10% (断言 B 不公平)")`. The capacity-tuning knob from §2.4 must be adjusted and the test re-run before merge. |

The rule is **inherited verbatim from pkg-06 §2.2** (the supersede source). pkg-06 spec 07 owned the same warn/fail semantics; pkg-07 spec 07 absorbs the rule, refreshes the prose, and extends the parametrize matrix from "3 strict variants" to "5 internal variants" — the 2 structurally-exempt variants (`ma_muzero`, `rewardhead_explicit_type`) inherit pkg-06 §2.3's exemption table and emit a `[WARN structural-exemption]` annotation in the test log instead of running the 5%/10% check (see §2.6 exemption table below).

### 2.4 Capacity-tuning knobs (declared in `BaselinesConfig`, codified at spec 08 §3)

The four internal capacity-tuning knobs are declared in `BaselinesConfig` (the spec-08-owned dataclass at `hyper_mve/configs/v4_config.py::BaselinesConfig`, design D10):

```python
cfg.baselines.internal_wide_hidden_dim: int = 512
cfg.baselines.internal_deep_layers: int = 8
cfg.baselines.internal_ma_muzero_share_pred_head: bool = True
cfg.baselines.internal_explicit_type_branches: int = 2     # α / β
```

Each knob's role in the 5%/10% reconciliation:

- **`cfg.baselines.internal_wide_hidden_dim`** — bisected at implementation time to land `delta(input_wide)` inside `[-0.05, +0.05]`. pkg-06 §2.4 documents the bisection procedure (binary-search the hidden_dim integer until the relative delta falls inside the PASS band); pkg-07 inherits it.
- **`cfg.baselines.internal_deep_layers`** — bisected at implementation time. Per pkg-06 §2.4 last paragraph: "层数离散 → input_deep 可能无法精确落入 5%；接受落入 10%（WARN），并在 PR 说明记录最接近的层数及 δ". pkg-07 inherits this caveat: integer-discrete layer count may force a [+5%, +10%] WARN rather than a clean PASS; the WARN is acceptable and documented.
- **`cfg.baselines.internal_ma_muzero_share_pred_head`** — boolean; toggles between "shared single head" (the structurally-exempt variant, default `True`) and "per-agent heads with a shared backbone" (an equal-param-attempted sub-variant, `False`). When `True`, ma_muzero is structurally exempt (§2.6); when `False`, ma_muzero is treated as an equal-param variant and runs the 5%/10% check. The default is `True` per design D10 (the structurally-exempt sub-variant is the one whose failure mode the paper analyzes for Assertion A).
- **`cfg.baselines.internal_explicit_type_branches`** — integer; the count of discrete agent-type branches in the RewardHead. Default `2` (α / β). When this equals the actual number of agent types in the env, the variant is structurally-exempt (§2.6); when it differs, the variant is an equal-param-attempted sub-variant. The default `2` matches the env's α / β capability split.

All four knobs are read-only consumed by the internal model classes (spec 03 §3.x) and by `count_conditioning_params` indirectly through the model's instantiated parameter set. **No hardcoded fallback** is allowed inside `hyper_mve/baselines/internal/*.py` — `cfg.baselines.internal_*` is the single source (C7-INT-CFG1 per pkg-06 inheritance; spec 01 §6 of pkg-07 enforces this via the `test_factory_consumes_cfg_baselines_fields_no_hardcoded_fallback` test).

### 2.5 Bit-exact shared-backbone (C7-INT-FAIR2)

Per Lock 1: spec 02 owns this axis. The contract is:

```
sum(p.numel() for p in HyperMuZeroModel(cfg).rep_net.parameters())
  == sum(p.numel() for p in create_baseline(cfg, variant).rep_net.parameters())
```

for `variant ∈ {input_wide, input_deep, ma_muzero, no_belief, rewardhead_explicit_type}`, and analogously for `belief_net` and `tri_context_encoder`. The test that asserts this is `tests/baselines/internal/test_shared_backbones.py::test_shared_backbone_param_count_bitexact` (spec 02 §5.1), parametrized over 5 variants × 3 backbones = 15 cells. Spec 07 §6 cross-references this test as the enforcement site for C7-INT-FAIR2; it does **not** duplicate the assertion.

### 2.6 Exemption table for structurally-different conditioned subsystems

Per pkg-06 §2.3 D5 (inherited verbatim, refreshed for pkg-07 IDs):

| variant | 5%/10% double-threshold | Reason |
|---------|--------------------------|--------|
| `input_wide` | **STRICT** (warn 5% / fail 10%) | Same-class conditioned subsystem (widened functional nets) |
| `input_deep` | **STRICT** | Same-class conditioned subsystem (deepened functional nets) |
| `no_belief` | **STRICT** | Same-class conditioned subsystem (hypernet 3 heads + 3 functional nets, only belief value path is zeroed — params identical) |
| `ma_muzero` | **EXEMPT** | Structurally smaller (single shared RewardHead, no hypernet heads) → §3 wall-clock saturation evidence instead |
| `rewardhead_explicit_type` | **EXEMPT** | Structurally different (no `hyper_rew`, type-branched RewardHead) → §3 wall-clock saturation evidence instead |

The 2 exempt variants get a `[WARN structural-exemption: <variant>]` annotation in the test log and **do not** run the 5%/10% assertion. Their fairness justification is the wall-clock saturation argument (per pkg-06 §4: "report `step_to_R*` for the exempt variant; if its return saturates inside the step budget, its disadvantage is structural rather than under-training/under-capacity"). Pkg-07 inherits this protocol; it is reported in pkg-08's appendix table, not in the main disclosure table (which is for external variants only, §4.2).

### 2.7 The cross-variant parametrize grid: 5 × 3 = 15 cells for backbone bit-exact + 5 × 1 for conditioned subsystem

Two separate parametrize matrices live in `tests/baselines/internal/`:

- **`test_shared_backbones.py::test_shared_backbone_param_count_bitexact`** — owned by spec 02. Parametrized over 5 variants × 3 backbones = 15 cells. Asserts bit-exact equal-param. (Lock 1 Axis A.)
- **`test_param_fairness.py::test_internal_param_fairness_5pct_warn_10pct_fail`** — owned by spec 07 (this spec). Parametrized over 5 variants × 1 conditioned-subsystem-bucket = 5 cells (3 strict + 2 exempt-annotated). Asserts the 5%/10% double-threshold rule for strict variants; emits `[WARN structural-exemption]` for exempt variants. (Lock 1 Axis B.)

The two matrices are **non-overlapping** — each cell asserts a distinct invariant, and a failure in one does not subsume the other. Combined coverage = 15 + 5 = 20 cells across the two test files for internal fairness.

---

## 3. Internal single-step forward perf (C7-INT-PERF1)

### 3.1 Budget

Per pkg-06 §5.1 + pkg-07 README C7-INT-PERF1: "input baseline 单步 forward ≤ 2.0× hyper". Verified on the Easy preset (`V4Config.from_preset("easy")`, N=2) under `torch.no_grad()` with the following timing protocol:

```
- warmup: 10 transition() calls (each = encode + transition) discarded
- timed: 100 transition() calls; median walltime across the 100 trials
- per-variant timed run is repeated 3 times; the median of the 3 medians is the report
```

The budget asserts `median(variant) <= 2.0 * median(hyper)` where the median is over the 3 outer trials. The protocol is verbatim from pkg-06 §5.1's `test_baseline_forward_budget` style (the pkg-06 test used 20 inner iterations; spec 07 strengthens to 100 inner + 3 outer for variance reduction — the pkg-06 test was noisy under CI load).

The protocol intentionally:

- Runs on **CPU** to remove GPU launch-time variance (CPU walltimes are deterministic enough to compare 2.0× ratios; GPU walltimes are noisy under shared-runner load).
- Uses **Easy preset N=2** — the smallest preset — to keep total wall time under 60 seconds.
- Uses `transition` rather than the full 7-API loop — `transition` is the inner-loop hot path; if `transition` is within budget, the full 7-API loop is too (no 7-API call is more expensive than `transition`).
- Uses `torch.no_grad()` — the budget is the forward-only path, not the training step.

### 3.2 Test name and contract

`tests/baselines/internal/test_perf.py::test_internal_input_variant_forward_under_2x_hyper`. Parametrized over `variant ∈ ("input_wide", "input_deep")`. Fails if the ratio exceeds 2.0×; the other 3 internal variants are not in scope (per pkg-06 §5 — only input-conditioned variants need the perf gate because they're the only ones that can balloon the per-step cost via wider/deeper functional nets; ma_muzero/no_belief/rewardhead_explicit_type have hyper-comparable forward cost by construction).

### 3.3 What this perf gate does NOT cover

- **Training-step wall-clock** — outside scope. Training-step cost is dominated by backprop, optimizer update, and replay-buffer ops, which are variant-independent.
- **Memory budget** — outside scope. Hyper's hypernet heads consume more activation memory than `input_wide`'s widened functional nets; comparing memory is uninformative.
- **Eval-time MCTS budget** — outside scope. The MVE planner cost is dominated by MCTS simulations and is variant-independent (the planner consumes the model's `predict()` output, which is constant-time per call).

---

## 4. External disclosure-style protocol

### 4.1 Conceptual: why disclosure, not equal-param

External Tier-1 runners have own architecture, own loop, own optimizer config. Equal-param is meaningless across MAPPO (PPO + GAE + Adam) vs QMIX (Q-learning + state-dependent QMixer + RMSprop, per pymarl default) vs MA-MuZero-GH (MCTS planner + dynamics + reward + prediction + Adam). The standard MARL paper practice is to **disclose 5 axes per external variant** and let reviewers judge:

- **`param_count: int`** — `sum(p.numel() for p in algorithm.parameters())` at the end of training (after the optimizer state has settled). Exposed by `MAPPOAlgorithm.param_count()` (spec 05 §9.3), `QMIXAlgorithm.param_count()` (spec 06 §2.x), `MAMuZeroGHAlgorithm.param_count()` (spec 06 §3.x), and `MAMBAAlgorithm.param_count()` (spec 06 §4.x if sourced).
- **`walltime_to_converge_seconds: float`** — wall-clock to the first env-step where the convergence threshold (§4.3 below) is crossed. Captured by the sweep harness (pkg-08 spec 05) via wall-clock timestamping around the train loop.
- **`lr_swept_best: float`** — best LR found in the sweep, defined as `argmax_lr mean_over_seeds(final_return)`. Computed by pkg-08 spec 07 stats reduction over the (variant, lr, seed) RunRegistry rows.
- **`final_return_per_seed: tuple[float, ...]`** — N≥5 seeds, raw final-return scalars at training end. Captured as `EvalReport.return_mean` per (variant, lr, seed) row.
- **`seeds_run: int`** — must be ≥5 for no WARN. Computed as `len(final_return_per_seed)` per variant per preset.

The disclosure-style protocol is the explicit standard from the MAPPO paper (Yu et al. 2022, paraphrased: "we report params + walltime + 5-seed final return + best LR over a swept grid") and the QMIX paper (Rashid et al. 2018, similar). Spec 07 codifies this practice as a table schema (§4.2) so reviewers can audit each external variant on the same 10 columns without reading 3 separate vendoring records.

### 4.2 Disclosure table schema (LOCKED; any drift = synchronous edit)

The disclosure table has exactly **10 columns** in this order:

| Column # | Column name | Type | Source |
|---|-------------|------|--------|
| 1 | `variant` | `str` | RunRegistry row `.variant` (e.g. `"external_mappo"`) |
| 2 | `preset` | `str` | RunRegistry row `.preset` (e.g. `"easy"`, `"medium"`) |
| 3 | `param_count` | `int` | `algorithm.param_count()` at training end |
| 4 | `walltime_to_converge_seconds` | `float` | Sweep harness wall-clock; `inf` if never crossed (§4.3) |
| 5 | `lr_swept_best` | `float` | argmax over LR of `mean(final_return_per_seed)` |
| 6 | `final_return_mean` | `float` | `mean(final_return_per_seed)` at `lr=lr_swept_best` |
| 7 | `final_return_sem` | `float` | `std(final_return_per_seed, ddof=1) / sqrt(len)` at `lr=lr_swept_best` |
| 8 | `seeds_run` | `int` | `len(final_return_per_seed)` at `lr=lr_swept_best` |
| 9 | `smoke_pass` | `bool` | `True` iff the variant's smoke gate (spec 05 §10 / spec 06 §2.9 / §3.9) passes |
| 10 | `sourced` | `bool` | Always `True` for MAPPO / QMIX / MA-MuZero-GH; `True` for MAMBA iff `IS_SOURCED` (spec 06 §4); `False` otherwise |

Plus a **footnotes block** below the table noting:

- MAMBA sourcing status: if `sourced=False`, footnote = `"see spec 06 §7.3 sourcing log; row exists for schema stability but all numeric columns are null"`.
- MARIE / GA stubs: the canonical footnote string is defined verbatim in §4.7 below; spec 08 §6 drift detector greps the §4.7 wording as the single source of truth. (Do not paraphrase here — see §4.7 for the locked text.)

The 10-column schema is **stable** — any addition / removal / reordering of columns requires synchronous edits to (i) this spec §4.2, (ii) `hyper_mve/experiments/compare.py::_render_disclosure_table` (the markdown emitter, downstream consumer), (iii) pkg-08 spec 07 stats reduction, and (iv) pkg-08 spec 08 RunRegistry row schema. The synchronous-edit invariant is asserted by spec 08 §6 drift detector (which greps for the verbatim column count "10 columns" across the four sites).

### 4.3 Convergence threshold definition

A run is considered "converged" at env-step `t*` iff:

```
mean(eval_returns[-5:]) >= 0.80 * row_plateau_mean
```

where:

- `eval_returns` is the time series of `EvalReport.return_mean` recorded at every 1k env-steps during training (the sweep harness's eval cadence; pkg-08 spec 05 §3).
- `eval_returns[-5:]` is the last 5 eval points = the most recent 5k env-steps window.
- `row_plateau_mean` is `mean(eval_returns)` over the **last** 10% of training env-steps within THIS row (the per-row, time-series "settled plateau" reference). For a `total_env_steps = 1_000_000` run, this is the mean over `eval_returns` recorded in steps `[900_000, 1_000_000]`. **`row_plateau_mean` is a per-row quantity; it is NOT the §4.2 column 6 `final_return_mean`** (which is the cross-seed aggregate `mean(final_return_per_seed)` at `lr=lr_swept_best`). The two quantities live at different aggregation levels: `row_plateau_mean` is computed inside one (variant, lr, seed) row's training time series and is consumed only here to compute that row's `t*`; the §4.2 `final_return_mean` column is computed downstream of `t*` via the seed-level aggregation step in §5 (step 2 stats reduction) and does NOT participate in `t*` computation.
- `t*` is the **first** env-step at which the inequality is satisfied; `walltime_to_converge_seconds` is the wall-clock elapsed from training start to env-step `t*` (linearly interpolated from the per-1k-step wall-clock checkpoints).

If the inequality is **never satisfied** within `total_env_steps`, `walltime_to_converge_seconds = float("inf")` and the disclosure table cell shows `"inf"`. The disclosure footnote auto-emits `[WARN never-converged]` for that row, but the row remains in the table (rather than being dropped) — the absence of convergence is itself a fairness-relevant disclosure.

The 80% threshold matches pkg-06 §4.1 `R* = 0.80 * R_oracle_only_plateau` (the "saturation evidence" threshold for exempt internal variants). Spec 07 reuses the same coefficient for external convergence detection — same conceptual idea ("close enough to its own settled plateau"), distinct denominator (internal: oracle_only plateau; external: own per-row plateau via `row_plateau_mean`). The reuse of the 0.80 coefficient is intentional and grep-stable.

### 4.4 LR-sweep grid contract

Each Tier-1 external variant declares its grid in `cfg.baselines.external_lr_sweep_grid["external_<variant>"]: tuple[float, ...]` with ≥3 LR values:

```python
# cfg.baselines.external_lr_sweep_grid (default, MappingProxyType, design D10)
{
    "external_mappo":         (1e-4, 3e-4, 1e-3),
    "external_qmix":          (5e-4, 1e-3, 5e-3),
    "external_ma_muzero_gh":  (3e-4, 1e-3, 3e-3),
    # "external_mamba" added at sourcing time (spec 06 §4.7); stubs not swept
}
```

The per-variant grids are calibrated to each algorithm's standard operating range:

- **MAPPO `(1e-4, 3e-4, 1e-3)`** — verbatim from spec 05 §9.1; centered on `3e-4` (lzj source's tested default). Range matches Adam-PPO typical sweep range from the MAPPO benchmark suite.
- **QMIX `(5e-4, 1e-3, 5e-3)`** — Q-learning typical range; pymarl default is `5e-4` with RMSprop, but the port uses Adam (spec 06 §2.x) so the grid skews higher. Confirmed via spec 06 §2.8 LR-sweep section.
- **MA-MuZero-GH `(3e-4, 1e-3, 3e-3)`** — muzero-general default is `1e-3`; the grid brackets it ±half-decade. Confirmed via spec 06 §3.8 LR-sweep section.

The grid is enumerated by pkg-08 spec 05 sweep harness; 3 LRs × 5 seeds = 15 runs per (variant, preset). For Easy + Medium presets (Hard is appendix-only), total = 15 × 2 = 30 runs per Tier-1 variant.

### 4.5 5-seed minimum (WARN-only)

README C7-EXT-FAIR1 baseline is `≥3 seeds`; spec 07 strengthens to `≥5 seeds` to align with the rest of the Methods main table (pkg-08 spec 01 stats reduction protocol uses 5 seeds for the headline numbers). The strengthening is **soft**:

- Rows with `seeds_run >= 5` → no warning.
- Rows with `3 <= seeds_run < 5` → disclosure footnote auto-emits `[WARN seeds_run=<n><5]` in the rendered markdown but the row remains in the table.
- Rows with `seeds_run < 3` → disclosure footnote auto-emits `[FAIL seeds_run=<n><3]` and the row is marked red in the rendered HTML version (markdown shows the FAIL tag); the row remains in the table but should not be cited in the paper.

The WARN-only policy means pkg-07 finalize is not blocked by a 4-seed row (which can happen if one seed OOMs and is dropped from the sweep). The FAIL policy is the README C7-EXT-FAIR1 minimum — anything below 3 seeds violates the baseline contract and should be re-run.

### 4.6 MAMBA Tier-2 row

If `IS_SOURCED is True` (spec 06 §4.1 sets this at the `mamba.py` module level when sourcing succeeds):

- MAMBA runs the full LR sweep just like the 3 Tier-1 variants.
- Disclosure row has all 10 columns populated.
- `sourced=True`.

If `IS_SOURCED is False`:

- The disclosure row **exists** in the table for schema stability (so the table's row count doesn't depend on sourcing status; downstream consumers can rely on a fixed schema).
- Every numeric column (`param_count`, `walltime_to_converge_seconds`, `lr_swept_best`, `final_return_mean`, `final_return_sem`, `seeds_run`) is `null` (rendered as `"-"` in markdown).
- `sourced=False`, `smoke_pass=False`.
- The disclosure footnote auto-emits `"MAMBA not sourced; see spec 06 §7.3 sourcing log"` for that row.

This dual behaviour (row-always-present, contents-null-when-not-sourced) lets the disclosure table be byte-stable across "MAMBA sourced" vs "MAMBA not sourced" implementation outcomes. Reviewers can audit the sourcing decision without the table's structure shifting.

### 4.7 MARIE / GA stubs (NOT in the disclosure table)

Per design D9 + spec 06 §5: MARIE and GA are reserved CLI names that raise `NotImplementedError` at `create_baseline()`. They do **not** appear as rows in the disclosure table — adding them would conflate "not yet implemented" with "implemented but not converged" and make the `sourced` column ambiguous. Their reservation is documented in a single disclosure-table **footnote** (below the 10-column table, above the MAMBA footnote):

> "Reserved CLI names not in disclosure table: `external_marie` (MARIE), `external_ga` (GA). Both raise `NotImplementedError` at construction time per design D9. Future pkg-07.5 follow-up may promote them; this table will gain rows at that time, not before."

The footnote is grep-stable; spec 08 §6 drift detector confirms it appears verbatim in `hyper_mve/experiments/compare.py::_render_disclosure_table` output.

---

## 5. Disclosure table data flow

End-to-end pipeline (per (variant, lr, seed) row → table cell):

1. **Sweep harness emits one RunRegistry row per (variant, lr, seed) tuple** (pkg-08 spec 05). Each row carries `param_count` (snapshot at training end), `walltime_to_converge_seconds` (computed via §4.3 against the row's own eval-return time series), `final_return` (`EvalReport.return_mean`), and metadata (variant, preset, lr, seed).
2. **Stats layer reduces seeds → mean+SEM** (pkg-08 spec 07). For each (variant, preset, lr) group, the stats layer computes `mean(final_return_per_seed)`, `std(final_return_per_seed, ddof=1) / sqrt(len)`, and `seeds_run = len(final_return_per_seed)`. For each (variant, preset) group, it then computes `lr_swept_best = argmax_lr mean(final_return)`, and reports the 6 numeric columns (`param_count`, `walltime_to_converge_seconds`, `lr_swept_best`, `final_return_mean`, `final_return_sem`, `seeds_run`) at `lr = lr_swept_best`.
3. **Compare CLI emits disclosure table markdown** via `python -m hyper_mve.experiments.compare --disclose`. The CLI renders the markdown table per preset (one table block per preset, columns as in §4.2), preceded by a preset name header (`## Disclosure table — easy preset`, `## Disclosure table — medium preset`), and emits the footnotes block at the end of each preset block. The output is appended to `runs/<exp_id>/disclosure.md`.
4. **Paper consumes the disclosure markdown** verbatim — pkg-07 ships the disclosure markdown as a copy-paste-ready table for Ch6 appendix. No further reformatting; the column order in §4.2 is the column order in the paper.

The single-direction flow (sweep → stats → markdown → paper) means any number in the disclosure table is traceable back to a single RunRegistry row, which in turn is traceable back to a (variant, lr, seed) sweep cell. Reviewers can audit any cell by `grep`-ing the RunRegistry for the cell's identity.

---

## 6. Where each fairness gate fires

| Gate ID | Description | Enforced at | Spec / file | Test name |
|---------|-------------|-------------|-------------|-----------|
| **C7-INT-FAIR1** | Conditioned subsystem ≤5% warn / ≤10% fail | spec 07 §2 | `tests/baselines/internal/test_param_fairness.py` | `test_internal_param_fairness_5pct_warn_10pct_fail` |
| **C7-INT-FAIR2** | Shared backbone (RepNet/BeliefNet/TriCtx) bit-exact across variants + hyper | spec 02 §5.1 | `tests/baselines/internal/test_shared_backbones.py` | `test_shared_backbone_param_count_bitexact` |
| **C7-INT-PERF1** | Input baseline single-step forward ≤ 2.0× hyper | spec 07 §3 | `tests/baselines/internal/test_perf.py` | `test_internal_input_variant_forward_under_2x_hyper` |
| **C7-EXT-FAIR1** | External LR sweep ≥3 LR × ≥3 seeds (5 seeds aspirational) | spec 07 §4 + spec 05/06 §LR-sweep | `tests/baselines/external/test_<var>_lr_sweep.py` | `test_<var>_lr_sweep_grid_is_at_least_3_lrs` for each of MAPPO/QMIX/MA-MuZero-GH |

The 4 gates collectively form pkg-07's fairness contract. C7-INT-FAIR2 is in spec 02 (Axis A, bit-exact lock); C7-INT-FAIR1 + C7-INT-PERF1 + C7-EXT-FAIR1 are in spec 07 (this spec). The disclosure table itself is not a "gate" in the test-suite sense — it does not fail; it is the output deliverable. Drift detection on the disclosure-table schema (§4.2 10-column count) is owned by spec 08 §6 (the schema drift detector).

---

## 7. Test contract (8 named tests)

The fairness protocol surface lives across two test directories:

- `tests/baselines/internal/` — Axis A (spec 02) + Axis B (this spec) + perf gate (this spec).
- `tests/baselines/external/` — LR-sweep grid (this spec) + disclosure-table rendering (this spec).

### 7.1 `test_internal_param_fairness_5pct_warn_10pct_fail`

`tests/baselines/internal/test_param_fairness.py`. Parametrized over 5 variants × 1 conditioned-subsystem-bucket = 5 cells (3 strict + 2 exempt). Asserts the 5%/10% double-threshold rule for strict variants; emits `[WARN structural-exemption]` annotation for exempt variants. Pseudocode:

```python
import pytest
import warnings
from hyper_mve.configs import V4Config
from hyper_mve.baselines import create_baseline
from hyper_mve.baselines.shared_backbones import count_conditioning_params
from hyper_mve.models import HyperMuZeroModel

STRICT_VARIANTS = ("input_wide", "input_deep", "no_belief")
EXEMPT_VARIANTS = ("ma_muzero", "rewardhead_explicit_type")

# Capacity knob for each STRICT variant. `no_belief` has NO knob — it is
# structurally identical to hyper at the conditioned subsystem (only the
# belief value path is zeroed at runtime); a no_belief failure indicates a
# count_conditioning_params accounting bug, not a knob misconfiguration.
KNOB_BY_VARIANT = {
    "input_wide": "cfg.baselines.internal_wide_hidden_dim",
    "input_deep": "cfg.baselines.internal_deep_layers",
    "no_belief": "(no knob — investigate count_conditioning_params accounting; spec 02 §2.1)",
}


@pytest.fixture
def cfg():
    return V4Config.from_preset("medium")


@pytest.mark.parametrize("variant", STRICT_VARIANTS)
def test_internal_param_fairness_5pct_warn_10pct_fail(cfg, variant):
    """C7-INT-FAIR1: strict variants must fall inside the 5%/10% envelope."""
    p_hyper = count_conditioning_params(HyperMuZeroModel(cfg))
    p_v = count_conditioning_params(create_baseline(cfg, variant))
    delta = abs(p_v - p_hyper) / p_hyper
    if delta > 0.10:
        pytest.fail(
            f"{variant} 条件化参数偏差 {delta:.1%} > 10% (断言 B 不公平); "
            f"adjust {KNOB_BY_VARIANT[variant]}"
        )
    elif delta > 0.05:
        warnings.warn(
            f"{variant} 条件化参数偏差 {delta:.1%} 落在 5%~10% WARN 带 (调小旋钮逼近)",
            UserWarning,
        )
    assert delta <= 0.10


@pytest.mark.parametrize("variant", EXEMPT_VARIANTS)
def test_internal_exempt_variants_get_warn_annotation(cfg, variant, capsys):
    """Structurally exempt variants emit WARN annotation, not the 5%/10% check."""
    p_hyper = count_conditioning_params(HyperMuZeroModel(cfg))
    p_v = count_conditioning_params(create_baseline(cfg, variant))
    # No 5%/10% assertion. Just emit the annotation and confirm the structural diff.
    print(f"[WARN structural-exemption: {variant}] p_hyper={p_hyper}, p_v={p_v}")
    assert p_v != p_hyper, (
        f"{variant} conditioned subsystem same param count as hyper — "
        "expected structural difference, see spec 07 §2.6 exemption table"
    )
```

Asserted invariants: (i) strict variants delta ≤ 0.10; (ii) WARN band (0.05 < delta ≤ 0.10) emits `UserWarning`; (iii) exempt variants confirm structural diff (`p_v != p_hyper`) without running the 5%/10% check.

### 7.2 `test_internal_input_variant_forward_under_2x_hyper`

`tests/baselines/internal/test_perf.py`. Parametrized over `variant ∈ ("input_wide", "input_deep")`. Pseudocode:

```python
import time
import pytest
import torch
from hyper_mve.configs import V4Config
from hyper_mve.baselines import create_baseline
from hyper_mve.models import HyperMuZeroModel


@pytest.fixture
def cfg():
    return V4Config.from_preset("easy")  # smallest preset for speed


def _median_transition_time(model, cfg, B=8):
    model.set_context_objective(torch.zeros(B))
    model.set_context_subjective(
        0, torch.zeros(B, 4),
        (torch.zeros(B), torch.zeros(B, cfg.env.N - 1, 2)),
    )
    obs = torch.randn(B, cfg.env.N, cfg.env.obs_dim)
    s = model.encode(obs)
    a = torch.zeros(B, cfg.env.N * cfg.env.A)
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model.transition(s, a)
    # Timed
    times = []
    for _ in range(100):
        with torch.no_grad():
            t0 = time.perf_counter()
            model.transition(s, a)
            times.append(time.perf_counter() - t0)
    times.sort()
    return times[50]   # median


@pytest.mark.parametrize("variant", ("input_wide", "input_deep"))
def test_internal_input_variant_forward_under_2x_hyper(cfg, variant):
    """C7-INT-PERF1: input variant transition() median walltime ≤ 2.0× hyper."""
    hyper = HyperMuZeroModel(cfg)
    model = create_baseline(cfg, variant)
    # 3 outer trials for variance reduction
    hyper_medians = [_median_transition_time(hyper, cfg) for _ in range(3)]
    variant_medians = [_median_transition_time(model, cfg) for _ in range(3)]
    hyper_medians.sort()
    variant_medians.sort()
    ratio = variant_medians[1] / hyper_medians[1]  # median of medians
    assert ratio <= 2.0, (
        f"{variant} transition() {variant_medians[1]*1e3:.2f}ms vs hyper "
        f"{hyper_medians[1]*1e3:.2f}ms = {ratio:.2f}x > 2.0x budget"
    )
```

Asserted invariants: median ratio ≤ 2.0×.

### 7.3 `test_external_lr_sweep_grid_has_at_least_3_lrs`

`tests/baselines/external/test_<variant>_lr_sweep.py` (one file per Tier-1 variant: `test_mappo_lr_sweep.py`, `test_qmix_lr_sweep.py`, `test_ma_muzero_gh_lr_sweep.py`). Parametrized over 3 Tier-1 variants. Pseudocode:

```python
import pytest
from hyper_mve.configs import V4Config


@pytest.mark.parametrize(
    "variant",
    ("external_mappo", "external_qmix", "external_ma_muzero_gh"),
)
def test_external_lr_sweep_grid_has_at_least_3_lrs(variant):
    """C7-EXT-FAIR1: cfg.baselines.external_lr_sweep_grid[variant] has ≥3 LRs."""
    cfg = V4Config.from_preset("easy")
    grid = cfg.baselines.external_lr_sweep_grid
    assert variant in grid, f"{variant} missing from external_lr_sweep_grid"
    assert len(grid[variant]) >= 3, (
        f"{variant} grid has {len(grid[variant])} LRs; C7-EXT-FAIR1 requires ≥3"
    )
    # Defensive: LRs must be strictly positive floats
    for lr in grid[variant]:
        assert isinstance(lr, float) and lr > 0
```

Asserted invariants: grid contains key, len ≥ 3, all LRs are positive floats.

### 7.4 `test_disclosure_table_schema_locked`

`tests/baselines/external/test_disclosure_table.py`. Given a mocked set of RunRegistry rows, the markdown emitted by `compare --disclose` must have exactly the 10 columns of §4.2 in the prescribed order. Pseudocode:

```python
from hyper_mve.experiments.compare import _render_disclosure_table


EXPECTED_COLUMNS = (
    "variant", "preset", "param_count", "walltime_to_converge_seconds",
    "lr_swept_best", "final_return_mean", "final_return_sem",
    "seeds_run", "smoke_pass", "sourced",
)


def test_disclosure_table_schema_locked():
    """§4.2 schema lock: exactly 10 columns in prescribed order."""
    rows = [
        # 1 row per (variant, preset) — minimum for schema test
        {"variant": "external_mappo", "preset": "easy", "param_count": 12345,
         "walltime_to_converge_seconds": 600.0, "lr_swept_best": 3e-4,
         "final_return_mean": 1.5, "final_return_sem": 0.1, "seeds_run": 5,
         "smoke_pass": True, "sourced": True},
    ]
    md = _render_disclosure_table(rows)
    # Markdown table header row is the second line (after "## Disclosure table")
    header = md.splitlines()[2]
    columns = [c.strip() for c in header.split("|") if c.strip()]
    assert tuple(columns) == EXPECTED_COLUMNS, (
        f"Disclosure table columns drifted: got {columns}, "
        f"expected {EXPECTED_COLUMNS}"
    )
```

Asserted invariants: column count = 10; column names + order match `EXPECTED_COLUMNS` tuple verbatim.

### 7.5 `test_disclosure_warns_when_seeds_run_lt_5`

`tests/baselines/external/test_disclosure_table.py`. Feeds 3-seed data; asserts `[WARN seeds_run=3<5]` appears verbatim in the rendered markdown. Pseudocode:

```python
def test_disclosure_warns_when_seeds_run_lt_5():
    """§4.5: rows with 3 <= seeds_run < 5 get a WARN footnote."""
    rows = [
        {"variant": "external_mappo", "preset": "easy", "param_count": 12345,
         "walltime_to_converge_seconds": 600.0, "lr_swept_best": 3e-4,
         "final_return_mean": 1.5, "final_return_sem": 0.1, "seeds_run": 3,
         "smoke_pass": True, "sourced": True},
    ]
    md = _render_disclosure_table(rows)
    assert "[WARN seeds_run=3<5]" in md
```

Asserted invariants: WARN string appears in output verbatim.

### 7.6 `test_disclosure_row_for_mamba_not_sourced_is_null_with_note`

`tests/baselines/external/test_disclosure_table.py`. Pseudocode:

```python
def test_disclosure_row_for_mamba_not_sourced_is_null_with_note():
    """§4.6: MAMBA row exists with null cells + footnote when IS_SOURCED=False."""
    rows = [
        {"variant": "external_mamba", "preset": "easy", "param_count": None,
         "walltime_to_converge_seconds": None, "lr_swept_best": None,
         "final_return_mean": None, "final_return_sem": None, "seeds_run": None,
         "smoke_pass": False, "sourced": False},
    ]
    md = _render_disclosure_table(rows)
    # Row must exist
    assert "external_mamba" in md
    # All numeric cells rendered as "-"
    assert md.count(" - ") >= 6   # 6 nullable numeric columns
    # Footnote must mention the sourcing log
    assert "see spec 06 §7.3 sourcing log" in md
```

Asserted invariants: row present with `-` cells for the 6 numeric columns; footnote string present verbatim.

### 7.7 `test_marie_ga_excluded_from_disclosure_table`

`tests/baselines/external/test_disclosure_table.py`. Pseudocode:

```python
def test_marie_ga_excluded_from_disclosure_table():
    """§4.7: MARIE / GA do not appear as rows; only as footnote."""
    rows = []   # No MARIE / GA rows fed by the stats reducer
    md = _render_disclosure_table(rows)
    # Neither variant name appears as a table row
    assert "external_marie |" not in md
    assert "external_ga |" not in md
    # But the reserved-CLI footnote does mention them
    assert "external_marie" in md   # in footnote
    assert "external_ga" in md      # in footnote
    assert "Reserved CLI names not in disclosure table" in md
```

Asserted invariants: no table rows for MARIE / GA; both names appear in the reserved-CLI footnote text verbatim.

### 7.8 Test count summary

- Spec 07 owns: **8 named tests** = `test_internal_param_fairness_5pct_warn_10pct_fail` + `test_internal_exempt_variants_get_warn_annotation` + `test_internal_input_variant_forward_under_2x_hyper` + `test_external_lr_sweep_grid_has_at_least_3_lrs` + `test_disclosure_table_schema_locked` + `test_disclosure_warns_when_seeds_run_lt_5` + `test_disclosure_row_for_mamba_not_sourced_is_null_with_note` + `test_marie_ga_excluded_from_disclosure_table`.
- Spec 02 owns: `test_shared_backbone_param_count_bitexact` (referenced here for C7-INT-FAIR2 closure).

Combined fairness test surface = **9 named functions** across 4 test files (8 owned by spec 07 + 1 owned by spec 02). Sweep-harness LR-sweep dispatch (per (variant, lr, seed)) is not a test — it's an integration runtime, owned by pkg-08 spec 05.

---

## 8. Integration hooks (cross-spec consumer table)

| Consumer | Consumed | Use |
|----------|----------|-----|
| **spec 02** (`02-shared-backbones-internal.md`) | `count_conditioning_params(model)` | §2.1 single source of truth for the conditioned-subsystem count; §2.5 bit-exact backbone test owner |
| **spec 03** (`03-internal-variants.md`) | 4 internal capacity-tuning knobs (`cfg.baselines.internal_wide_hidden_dim` etc.) | §2.4 knob consumption by 5 internal model classes |
| **spec 05** (`05-external-mappo.md`) | MAPPO disclosure feed (`param_count()`, `walltime_seconds`, LR-best from §9 sweep) | §4.4 grid `(1e-4, 3e-4, 1e-3)` + §5 disclosure pipeline |
| **spec 06** (`06-external-qmix-mamuzero-mamba.md`) | QMIX / MA-MuZero-GH / MAMBA disclosure feed + MARIE/GA stub footnote | §4.4 grids + §4.6 MAMBA row + §4.7 stub footnote |
| **spec 08** (`08-integration-contracts.md`) | `cfg.baselines.external_lr_sweep_grid: MappingProxyType[str, tuple[float, ...]]` declaration site + 4 capacity-tuning fields | Locks §2.4 + §4.4 as cross-spec contract; spec 08 §3 codifies the 5-field `BaselinesConfig` 穷举 |
| **pkg-08 spec 05** (`05-sweep-harness-and-run-registry.md`) | Sweep enumeration over `(variant, lr, seed)` from `cfg.baselines.external_lr_sweep_grid` | §4.4 grid drives 15 runs / (variant, preset); §4.3 convergence threshold computed against per-row eval-return time series |
| **pkg-08 spec 07** (stats reduction) | Per-(variant, preset, lr) mean+SEM reduction + per-(variant, preset) `lr_swept_best` argmax | §5 step 2 data flow |
| **pkg-08 spec 08** (RunRegistry row schema) | Each row carries `param_count: int`, `walltime_to_converge_seconds: float`, `lr: float`, plus the standard fields | §5 step 1 data flow |
| **`hyper_mve/experiments/compare.py`** (downstream consumer) | `--disclose` flag renders the §4.2 schema markdown | §5 step 3 emit site |

### 8.1 No new cfg fields declared by this spec

Spec 07 **does not** declare any new field on `cfg.baselines`. The 5 fields it consumes (`internal_wide_hidden_dim`, `internal_deep_layers`, `internal_ma_muzero_share_pred_head`, `internal_explicit_type_branches`, `external_lr_sweep_grid`) are all declared by spec 08 §3 (the canonical `BaselinesConfig` 5-field 穷举 site per design D10). Spec 07 cross-references spec 08 for each field; spec 08's grep test asserts no field appears in `cfg.baselines` namespace without a spec 07 cross-reference.

### 8.2 Downstream code patches (declared by other specs; counted honestly here)

- `hyper_mve/experiments/compare.py::_render_disclosure_table` — **new function**, ~80 lines (table rendering + footnote emission). Declared by spec 07; codified at implementation time. Not a "patch" — it's a new module-level function.
- `hyper_mve/configs/v4_config.py::BaselinesConfig` — declared by spec 08 §3; spec 07 only consumes. Patch line count = 0 [no-behaviour for spec 07].
- `hyper_mve/baselines/internal/*.py` 5 model classes — read `cfg.baselines.internal_*` knobs to size their conditioned subsystem. Declared by spec 03; spec 07 only consumes. Patch line count = 0 [no-behaviour for spec 07].
- `tests/baselines/internal/test_param_fairness.py` — **new test file**, ~80 lines. Declared by spec 07; codified at implementation time.
- `tests/baselines/internal/test_perf.py` — **new test file**, ~60 lines. Declared by spec 07.
- `tests/baselines/external/test_disclosure_table.py` — **new test file**, ~120 lines. Declared by spec 07.
- `tests/baselines/external/test_{mappo,qmix,ma_muzero_gh}_lr_sweep.py` — each ~30 lines. Declared by spec 07; the MAPPO file is partially declared by spec 05 §12.5 (one of the 5 named tests there is `test_mappo_lr_sweep_grid_is_at_least_3_lrs` — spec 07 §7.3's parametrized version supersedes the per-variant version).

Total new code surface declared by spec 07 = 1 production function (~80 lines) + 4 test files (~290 lines). Zero modifications to existing production code [no-behaviour].

---

## 9. Cross-references

### 9.1 Upstream anchors

- **design.md §4 D5** — 双层 fairness (Internal 严格继承 pkg-06 D5 + External 披露式). The authoritative locked decision; spec 07 inherits both halves.
- **design.md §4 D7** — Tier-1 selection + adapter contract. Source of the per-variant LR-sweep grid table (§4.4 below references this).
- **design.md §4 D10** — `cfg.baselines.external_lr_sweep_grid: Mapping[str, tuple[float, ...]]` + 4 internal capacity-tuning fields. Spec 07 §2.4 consumes the internal 4; §4.4 consumes the external 1.
- **README C7-INT-FAIR1** — Conditioned-subsystem ≤5% warn / ≤10% fail. Spec 07 §2 is the enforcement site.
- **README C7-INT-FAIR2** — RepNet/BeliefNet/TriCtx 跨 internal variant 参数量逐位相同. Cross-referenced for closure; spec 02 §5.1 is the enforcement site.
- **README C7-INT-PERF1** — Input baseline single-step forward ≤ 2.0× hyper. Spec 07 §3 is the enforcement site.
- **README C7-EXT-FAIR1** — External LR sweep ≥3 LR × ≥3 seeds, 结果披露. Spec 07 §4 is the enforcement site; the 5-seed strengthening is documented in §4.5.
- **pkg-06 C6-FAIR1** — The 5%/10% rule's original lock site; spec 07 §2 supersedes by absorbing verbatim.
- **pkg-06 C6-FAIR2** — Backbone bit-exact; refreshed to C7-INT-FAIR2 in pkg-07 (lock site = spec 02).
- **pkg-06 C6-PERF1** — Perf 2.0× budget; refreshed to C7-INT-PERF1 in pkg-07 (lock site = spec 07 §3).

### 9.2 Sibling spec anchors

- **spec 02** `02-shared-backbones-internal.md` §2.2 (`count_conditioning_params`) + §5.1 (`test_shared_backbone_param_count_bitexact`) — the single source of truth for both axes' computation; bit-exact lock site.
- **spec 03** `03-internal-variants.md` §3.x (5 internal variant capacity-tuning knob consumption) — the upstream of `cfg.baselines.internal_*` field reads.
- **spec 05** `05-external-mappo.md` §9 (LR-sweep contract) + §11 (per-impl tuning constants — explicitly NOT on `cfg.baselines`) + §12.5 (`test_mappo_lr_sweep_grid_is_at_least_3_lrs`) — MAPPO disclosure feed.
- **spec 06** `06-external-qmix-mamuzero-mamba.md` §2.8 (QMIX LR-sweep) + §3.8 (MA-MuZero-GH LR-sweep) + §4.7 (MAMBA LR-sweep iff sourced) + §5 (MARIE/GA stub footnote source) — non-MAPPO Tier-1 + Tier-2 + stubs disclosure feed.
- **spec 08** `08-integration-contracts.md` §3 (`BaselinesConfig` 5-field 穷举) + §6 (drift detector for §4.2 schema) — the cfg declaration site + drift enforcement.

### 9.3 Downstream pkg-08 anchors

- **pkg-08 spec 05** `05-sweep-harness-and-run-registry.md` §"external sweep enumeration" — consumes `cfg.baselines.external_lr_sweep_grid` to drive 15 runs / (variant, preset).
- **pkg-08 spec 07** (stats reduction) — reduces per-(variant, preset, lr, seed) RunRegistry rows to per-(variant, preset) disclosure-row payloads.
- **pkg-08 spec 08** (RunRegistry row schema) — each row carries `param_count: int`, `walltime_to_converge_seconds: float`, `lr: float` so the stats layer can compute the 6 numeric columns of §4.2 without out-of-band lookups.
- **pkg-08 spec 01** §6 (external delegation contract) — `EvalReport.return_mean` is the per-seed final-return source for the stats reduction.

### 9.4 Repo ground-truth anchors

- `hyper_mve/configs/v4_config.py:23-37` — `V4Config` dataclass: env / model / train / mup / eval / legacy sub-configs. **NO `baselines` field yet** (declared by spec 08 §3 at implementation time). Spec 07 §2.4 + §4.4 read assumes the field will be added.
- `hyper_mve/baselines/shared_backbones.py` (not yet implemented; declared by spec 02 §2.1) — host of `count_conditioning_params`.
- `hyper_mve/experiments/compare.py` (not yet implemented; declared by spec 07 §5 step 3) — host of `_render_disclosure_table`.

### 9.5 Supersede anchor

- **pkg-06 spec 07** `07-param-fairness-and-lr-sweep.md` — spec 07 of pkg-07 supersedes by absorbing §2 (5%/10% double threshold) and §3 (LR-sweep protocol) verbatim, extending §3 from "uniform LR grid for hyper + 5 internal" to "per-variant LR grid for external + disclosure-style fairness reporting". pkg-06 README banner is already marked supersede.

### 9.6 Ch5 paper anchor

- **Ch5 §5.6 step1** — Equal-param statistical protocol (only the conditioned subsystem is counted). Spec 07 §2.1 is the code-side proxy of this protocol. The 5%/10% double threshold is the operational instantiation.
- **Ch5 §5.6 step2** — LR sweep at each variant's best LR. Spec 07 §4 codifies this for external; internal LR sweep is at pkg-08 spec 05 (uniform grid).
- **Ch6 appendix** — Disclosure table is shipped verbatim as a copy-paste-ready markdown block.

---

## 10. Anchors (verbatim grep targets for spec 08 §6 drift detector)

- `pkg-07 spec 07 §1: two-tier fairness protocol — Internal strict 5%/10% + External disclosure-style`
- `pkg-07 spec 07 §2.1: count_conditioning_params is the single source of truth for the conditioned-subsystem count`
- `pkg-07 spec 07 §2.3: delta = abs(P_v - P_hyper) / P_hyper — PASS ≤ 5%, WARN ≤ 10%, FAIL > 10%`
- `pkg-07 spec 07 §2.4: 4 internal capacity-tuning knobs on cfg.baselines (wide_hidden_dim, deep_layers, ma_muzero_share_pred_head, explicit_type_branches)`
- `pkg-07 spec 07 §2.6: STRICT = (input_wide, input_deep, no_belief); EXEMPT = (ma_muzero, rewardhead_explicit_type)`
- `pkg-07 spec 07 §3: input baseline transition() median walltime ≤ 2.0× hyper on Easy preset, 100 inner + 3 outer trials`
- `pkg-07 spec 07 §4.2: disclosure table has exactly 10 columns in this order — variant / preset / param_count / walltime_to_converge_seconds / lr_swept_best / final_return_mean / final_return_sem / seeds_run / smoke_pass / sourced`
- `pkg-07 spec 07 §4.3: convergence = first env-step where mean(eval_returns[-5:]) >= 0.80 * final_return_mean`
- `pkg-07 spec 07 §4.4: cfg.baselines.external_lr_sweep_grid — MAPPO (1e-4, 3e-4, 1e-3) / QMIX (5e-4, 1e-3, 5e-3) / MA-MuZero-GH (3e-4, 1e-3, 3e-3)`
- `pkg-07 spec 07 §4.5: seeds_run < 5 emits [WARN seeds_run=<n><5]; seeds_run < 3 emits [FAIL seeds_run=<n><3]`
- `pkg-07 spec 07 §4.6: MAMBA row exists with null cells when IS_SOURCED=False; footnote points to spec 06 §7.3 sourcing log`
- `pkg-07 spec 07 §4.7: MARIE / GA NOT in disclosure table — only in reserved-CLI footnote`
- `pkg-07 spec 07 §6: C7-INT-FAIR1 + C7-INT-PERF1 + C7-EXT-FAIR1 owned by spec 07; C7-INT-FAIR2 owned by spec 02`
- `pkg-07 spec 07 §7: 8 named tests across tests/baselines/internal/ and tests/baselines/external/`
