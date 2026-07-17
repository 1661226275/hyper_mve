# Spec 07 — Statistics (Welch t / Holm-Bonferroni) + Compare CLI + Paper-Grade Comparison Plot

> **Parent docs**: [`../proposal.md`](../proposal.md) · [`../design.md`](../design.md) §4 D9 / §4 D10 · [`../README.md`](../README.md) C8-ABL-STAT1 + C8-ABL-PLOT1 + Ch6.2.3 (statistics protocol — Welch t + Holm-Bonferroni + 5-seed minimum, anchored via C8-ABL-STAT1 row) + §"v4 关键约束" Ablation 半 rows 7–8
> **Upstream consumed**: [`./01-unified-evaluator.md`](./01-unified-evaluator.md) §3 (`@dataclass(frozen=True) EvalReport` 32-field schema — slots `return_mean`, `return_sem`, `episodes_per_c`, `planner_prior_return_gap`, `direct_inference_return_mean`, `planner_full_return_mean`, `regret_per_c`, `oracle_ceiling_cache_hit`, `c_visible`, `eval_planner_mode`, `belief_c_mae`, `belief_c_calibration`) · [`./05-sweep-harness-and-run-registry.md`](./05-sweep-harness-and-run-registry.md) §4 (`RegistryRow` 23-key schema — `eval_report_path`, `variant`, `seed`, `config_hash`, `ablation_cell`, `return_mean`, `return_zero_shot_unseen`, `regret_mean`, `status`) + §10.5 (`pandas.read_json(...lines=True)` reader contract + lazy `eval_report_path.apply(_load_report)` join) · [`./03-direct-inference-toggle.md`](./03-direct-inference-toggle.md) §6 (paired-mode protocol; row emission contract for `(variant, seed, eval_planner_mode)` pairing) · [`./02-zero-shot-and-c-hidden.md`](./02-zero-shot-and-c-hidden.md) §3 (c_hidden paired protocol `cfg.env.c_visible` flag) + §4 (regret-vs-oracle-ceiling fields)
> **Cross-refs**: [`./06-ablation-cli-and-cells.md`](./06-ablation-cli-and-cells.md) §3 (5 canned YAMLs — ablation cells become `--methods` comparison subjects via `ablation_cell` groupby) · [`./08-integration-contracts.md`](./08-integration-contracts.md) §5 (downstream-file declarations for `stats.py` + `compare.py` codified canonically; reverse-consumed from this spec's §12 patch tags) · [`../../pkg-07-baselines/specs/07-fairness-protocol.md`](../../pkg-07-baselines/specs/07-fairness-protocol.md) §4.2 (10-column external-runner disclosure table schema — REVERSE-CONSUMED via the `--disclose` flag declared in §6.3 below) + §4.5 (5-seed minimum WARN-only policy inherited verbatim) + §4.6 (MAMBA not-sourced row stability) + §4.7 (MARIE/GA exclusion footnote) + §5 step 3 (compare CLI emits disclosure markdown — pkg-07 ↔ pkg-08 data-flow anchor)
> **Status**: SDD only — describes the contract for `hyper_mve/experiments/stats.py` + `hyper_mve/experiments/compare.py`. No production code is written here. The two files are NEW; no existing file is modified by this spec.

---

## ⚠️ Header — three hard locks

### Lock 1 — Welch t-test is THE pairwise comparison primitive; no Student t, no Mann-Whitney U fallback

The single pairwise comparison primitive consumed by spec 07 is `scipy.stats.ttest_ind(sample_a, sample_b, equal_var=False)` — the Welch unequal-variance t-test, returning `(t_statistic, p_value_two_sided)`. **Equal-variance Student t (`equal_var=True`) is forbidden**, and **Mann-Whitney U (`scipy.stats.mannwhitneyu`) is forbidden** as a fallback for low-N or non-normal samples.

The locked rationale: empirical observation across v4-opt 2026-06 sweeps shows the seed-distribution variance is **heterogeneous across variants** at convergence — hyper has noticeably lower per-seed variance (≈ 0.4 SEM at the 5-seed protocol) than `no_belief` (≈ 1.1 SEM) and lower than `external_mappo` (≈ 0.9 SEM). Under unequal variances, the equal-variance Student t-test **systematically over-rejects** the null hypothesis (a Type-I error inflation between roughly 1.5× and 3× the nominal alpha at the empirical variance ratios observed), which is the failure mode the paper Ch6.2.3 protocol was designed to avoid (Theory Audit §10.3 explicit). Mann-Whitney U is correct under non-normality but throws away the magnitude information that the Welch t preserves; given the paper reports a magnitude (mean return) alongside the test, the Welch t is the consistent choice.

A drift detector test `test_welch_t_against_scipy_reference` (§11) asserts the implementation calls `scipy.stats.ttest_ind` with `equal_var=False` literally (via a `monkeypatch` on the scipy function that intercepts the kwarg) — any future "let's also support Student t" PR is caught at CI time.

### Lock 2 — Holm-Bonferroni correction applies WHEN AND ONLY WHEN the comparison call produces ≥ 2 pairwise tests; the k ≥ 3 method-count gate is the multiple-comparisons trigger

For ≥ 3 methods compared in a single call (k ≥ 3), the compare CLI auto-detects the method count, performs **k − 1 pairwise Welch t-tests against a designated reference method**, and applies the **Holm-Bonferroni step-down correction** over the k − 1 raw p-values. For exactly 2 methods (k = 2, 1 pairwise test), **no correction** is applied: `adjusted_p == raw_p` and `correction = "none"`. This is the locked semantic; users who want to compare 2 methods without correction get the raw p-value, and users who compare 3+ methods always get the corrected p-values without an opt-in flag.

The correction trigger is **the number of pairwise tests in a single call**, not the number of methods in the registry. A user who calls `compare --a runs/hyper.csv --b runs/input_wide.csv` produces k = 2 → no correction. A user who calls `compare --methods runs/hyper.csv,runs/input_wide.csv,runs/no_belief.csv --reference hyper` produces k = 3 → 2 pairwise tests vs reference → Holm-Bonferroni applied. A user who runs 10 sequential 2-method calls **does not** get a "global" correction across the 10 calls — that is the reviewer's responsibility to disclose (the paper's Ch6 main table is single-call multi-method via `--methods`, so the locked semantic covers every reported number).

A drift detector test `test_holm_bonferroni_skipped_for_single_comparison` (§11) constructs a 2-method call and asserts `ComparisonResult.correction == "none"` AND `adjusted == raw` byte-identically; the symmetric positive test `test_holm_bonferroni_against_statsmodels` (§11) constructs a 4-method call and asserts the adjusted p-values match `statsmodels.stats.multitest.multipletests(method="holm")` to a relative tolerance of 1e-12.

### Lock 3 — matplotlib Agg backend is the ONLY backend; `matplotlib.use("Agg")` is set at module-import time; no `plt.show()` anywhere

Headless rendering only. The compare CLI runs under sweep harness (spec 05) subprocess isolation and under CI without an X server; `matplotlib.use("Agg")` is the only backend that guarantees both. The directive is issued at **module-import time** at the top of `hyper_mve/experiments/compare.py`, before any `import matplotlib.pyplot as plt` line, so any future import-order regression is caught at import rather than at first figure construction. **`plt.show()` is forbidden** anywhere in the file; every figure is `plt.savefig(...)` then `plt.close(fig)` to release Tk/Agg resources.

A named test `test_compare_plot_renders_headless` (§11) asserts `matplotlib.get_backend().lower() == "agg"` after importing the compare module and that an end-to-end plot call writes a file with `os.path.getsize(...) > 0` to a `tmp_path` without raising. The 300-DPI rendering target (§8.2) is documented but not asserted at CI time (CI machines occasionally produce slightly different antialiasing artifacts; bit-exact DPI is not the locked invariant).

---

## 1. Purpose

Spec 07 is the **stats reducer + comparison-plot emitter** that converts the multi-row `RunRegistry` (spec 05) into the paper-grade numbers and figures Ch6 needs. Three reviewer-facing motivations make this spec necessary.

**Motivation 1 — Ch6.2.3 mandates Welch t + 5 seeds + multiple-comparison correction.** The paper's statistical protocol is locked at Ch6.2.3 + Theory Audit §10.3: 5-seed minimum for every main-table cell, Welch t-test for every pairwise "method X is different from method Y" claim, and Holm-Bonferroni when ≥ 3 methods are jointly compared. Without a single canonical code path, reviewers reading "p < 0.05" in Ch6 would need to chase down whether each claim used the same definition. Spec 07 provides the code path: `welch_t` + `holm_bonferroni` + `compare_methods` are the three functions every Ch6 number flows through.

**Motivation 2 — CLI-driven inspection without ad-hoc pandas.** During implementation phase D-F (per pkg-08 README §"实施期"), the user runs `compare --a runs/<a>.csv --b runs/<b>.csv` repeatedly to inspect sweep cells before committing to a paper number. Without a CLI, every inspection is a pandas snippet that drifts from the eventual paper-grade pipeline. Spec 07's CLI is the **single inspection-and-paper path** — every number a reviewer sees in Ch6 was produced by the same `compare` invocation chain a developer used during inspection.

**Motivation 3 — external-runner disclosure table feed (pkg-07 spec 07 §4.2 + §5 step 3).** Pkg-07 spec 07 §4.2 declares a 10-column disclosure table for external Tier-1 runners (MAPPO / QMIX / MA-MuZero-GH / MAMBA-if-sourced). Pkg-07 spec 07 §5 step 3 specifies `python -m hyper_mve.experiments.compare --disclose` as the canonical emitter of the disclosure markdown — but the implementation lives in pkg-08 (because the RunRegistry is a pkg-08 artifact, and stats reduction is a pkg-08 responsibility). Spec 07 of pkg-08 therefore implements the `--disclose` flag and the disclosure-table renderer; pkg-07 spec 07 §4.2 + §5 is the reverse-consumption anchor.

The deliverable is **two new files**, declared in spec 08 §5 of pkg-08 as new code surfaces:

```
hyper_mve/experiments/stats.py    — Welch t + Holm-Bonferroni + 5 stats joins; pure-function module, no I/O at import time
hyper_mve/experiments/compare.py  — argparse CLI + markdown table emitter + matplotlib bar+errorbar plot + --disclose dispatcher
```

No existing file is modified by this spec. No new `cfg` field is contributed by this spec (per §10 below — stats parameters live as CLI flags / function args, not on `cfg.eval`). The 5-field `cfg` enumeration in pkg-08 README §"5 新 cfg 字段穷举" is unchanged.

---

## 2. `stats.py` public API

The public surface of `hyper_mve/experiments/stats.py` is **3 free functions + 1 frozen dataclass + 5 join helpers**. All free functions are pure (no I/O, no global state); the join helpers consume in-memory `RegistryRow` sequences materialised by the caller via `pandas.read_json("runs/registry.jsonl", lines=True)` per spec 05 §10.5.

```python
# hyper_mve/experiments/stats.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np


# === Welch t-test primitive (Lock 1) =================================

def welch_t(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
) -> tuple[float, float]:
    """Returns (t_statistic, p_value_two_sided).
    Thin wrapper over scipy.stats.ttest_ind(sample_a, sample_b, equal_var=False).
    
    Raises ValueError if either sample contains NaN (§3.2) or has length < 2 (§5.2).
    """


# === Holm-Bonferroni step-down (Lock 2) ==============================

def holm_bonferroni(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> tuple[list[bool], list[float]]:
    """Step-down Holm-Bonferroni correction. Returns
       (rejection_per_comparison_at_alpha, adjusted_p_values),
    both lists of length len(p_values), in the same input order.

    Implementation: hand-coded 6-line step-down per §4.1 (avoids statsmodels dep).
    Identical results to statsmodels.stats.multitest.multipletests(method="holm")
    to floating-point precision; tested at §11 test_holm_bonferroni_against_statsmodels.
    """


# === Multi-method comparison primitive (Lock 2 gate) =================

def compare_methods(
    method_returns: Mapping[str, np.ndarray],
    *,
    reference: str | None = None,
    alpha: float = 0.05,
) -> ComparisonResult:
    """Pairwise Welch t each method vs the reference (or all-pairs if reference is None).
    
    method_returns: {method_name: 1d array of per-seed returns}; len >= 2 per method (§5.2).
    reference: if not None, must be a key of method_returns; the comparison is
               (k - 1) pairwise tests against the reference.
               If None, all k * (k - 1) / 2 pairs are tested.
    alpha: significance threshold for the rejection decision; default 0.05.
    
    Holm-Bonferroni is auto-applied iff the resulting number of pairwise tests >= 2
    (Lock 2: k >= 3 in the reference mode, or k >= 3 in all-pairs mode).
    
    The result includes a per-method seed-count map; methods with n < 5 are flagged
    in the ComparisonResult.markdown property (§7) via the "[WARN insufficient seeds]"
    prefix per §5.1.
    """


@dataclass(frozen=True)
class ComparisonResult:
    """Frozen container returned by compare_methods. Hashable + JSON-serialisable."""
    methods: tuple[str, ...]                                # ordered list of compared methods (sorted by mean desc per §8.1)
    reference: str | None                                    # the reference method (if pairwise-vs-reference); None if all-pairs
    pairs: tuple[tuple[str, str], ...]                       # comparison pairs (e.g. (hyper, input_wide))
    t_statistics: Mapping[tuple[str, str], float]            # raw Welch t per pair
    raw_p_values: Mapping[tuple[str, str], float]            # raw two-sided p per pair
    adjusted_p_values: Mapping[tuple[str, str], float]       # post-Holm-Bonferroni; == raw iff len(pairs) == 1
    rejections: Mapping[tuple[str, str], bool]               # True iff adjusted_p < alpha
    n_per_method: Mapping[str, int]                          # seeds-per-method counts; surfaced in markdown (§7)
    mean_per_method: Mapping[str, float]                     # sample means (for the bar plot §8.1)
    sem_per_method: Mapping[str, float]                      # sample SEM = std(ddof=1) / sqrt(n) (for the errorbars §8.1)
    correction: Literal["none", "holm-bonferroni"]           # "none" iff len(pairs) == 1 per Lock 2
    alpha: float                                             # the alpha threshold used


# === Stats-layer joins (§9) ==========================================

def compute_planner_prior_gap(registry_rows: Sequence[dict]) -> Mapping[tuple[str, int], float]: ...
def compute_cross_mode_deltas(registry_rows: Sequence[dict]) -> Mapping[tuple[str, int, str, str], float]: ...
def compute_c_hidden_gap(registry_rows: Sequence[dict]) -> Mapping[tuple[str, int], float]: ...
def compute_regret_per_method(registry_rows: Sequence[dict]) -> Mapping[str, tuple[float, float]]: ...
def render_disclosure_table(registry_rows: Sequence[dict], preset: str) -> str: ...
```

The 4 free functions + 1 dataclass form the **primitive layer**; the 5 join helpers form the **registry-consumption layer** (§9 details). The CLI in `compare.py` (§6) drives both layers.

---

## 3. Welch t-test contract

### 3.1 Implementation

`welch_t(sample_a, sample_b)` is a thin wrapper:

```python
def welch_t(sample_a: np.ndarray, sample_b: np.ndarray) -> tuple[float, float]:
    """Lock 1: scipy.stats.ttest_ind with equal_var=False ONLY."""
    sample_a = np.asarray(sample_a, dtype=float)
    sample_b = np.asarray(sample_b, dtype=float)
    if np.isnan(sample_a).any() or np.isnan(sample_b).any():
        raise ValueError(
            "welch_t: NaN in sample. Caller must filter NaN rows before calling. "
            "See spec 07 §3.2 (NaN-handling) and §9.4 oracle-ceiling cache-miss "
            "filtering — regret_mean is NaN on cache miss per spec 02 §4.7."
        )
    if len(sample_a) < 2 or len(sample_b) < 2:
        raise ValueError(
            f"welch_t: Welch t undefined for n < 2. Got n_a={len(sample_a)}, "
            f"n_b={len(sample_b)}. See spec 07 §5.2."
        )
    from scipy.stats import ttest_ind
    result = ttest_ind(sample_a, sample_b, equal_var=False)
    return (float(result.statistic), float(result.pvalue))
```

The `equal_var=False` kwarg is the **single non-default scipy kwarg** the spec consumes; any future maintainer who tries `equal_var=True` is caught by `test_welch_t_against_scipy_reference` (§11.1) which asserts the kwarg literally via monkeypatch.

### 3.2 NaN-handling

NaN in either sample raises `ValueError` at `welch_t` entry — **silent drop is forbidden**. The locked rationale: NaN appears in `EvalReport.regret_per_c` on oracle-ceiling cache miss (spec 02 §4.7 — `regret_per_c={c: float("nan") for c in zero_shot_test_c}` on any miss) and in `EvalReport.belief_c_mae` on non-hyper variants (spec 01 §3.1 slot 30 — `None` serialised as `null` in JSON, materialised as `NaN` on pandas round-trip). Silently dropping NaN would mean a 4-of-5-seed comparison silently becomes a 3-of-5-seed comparison without the user's knowledge; raising forces the caller to explicitly filter (or to interpret the missing data).

The caller's expected filter idiom:

```python
# At the compare.py CLI level (§6) before calling welch_t:
a_clean = a_returns[~np.isnan(a_returns)]
b_clean = b_returns[~np.isnan(b_returns)]
if len(a_clean) < len(a_returns):
    print(f"[WARN] dropped {len(a_returns) - len(a_clean)} NaN rows from method A", file=sys.stderr)
if len(a_clean) < 2 or len(b_clean) < 2:
    print(f"[FAIL] insufficient seeds after NaN drop", file=sys.stderr)
    sys.exit(1)
t, p = welch_t(a_clean, b_clean)
```

A named test `test_welch_t_raises_on_nan` (§11) constructs a sample with one NaN and asserts `ValueError` is raised with the documented message.

### 3.3 Two-sided p-value

The returned p-value is **two-sided** (the default `scipy.stats.ttest_ind` behaviour). The paper's convention is the FAIR claim is "method X is different from method Y" (rejecting the null `mean_X == mean_Y`), not "method X is better than method Y" (rejecting `mean_X <= mean_Y`). The two-sided p is the consistent primitive; a one-sided test would require the spec to lock the directionality, which is a stronger commitment than Ch6.2.3 prose makes.

### 3.4 Effect size NOT computed

Cohen's d, Hedges' g, and other effect-size statistics are **out of scope** for this spec. Reviewers who want effect sizes can compute them post-hoc from the same `(mean, sem, n)` triple the spec surfaces (the formula is `d = (mean_a - mean_b) / pooled_std`, where `pooled_std = sqrt((sem_a**2 * n_a + sem_b**2 * n_b) * n / (n - 2))` for the relevant pooled denominator). The spec does not include the effect-size column in `ComparisonResult` to keep the primary contract minimal; future extension is straightforward.

---

## 4. Holm-Bonferroni step-down

### 4.1 Algorithm

Hand-coded step-down (6 lines core; not a statsmodels dep):

```python
def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> tuple[list[bool], list[float]]:
    m = len(p_values)
    if m == 0:
        return [], []
    if m == 1:
        # Lock 2: no correction for single comparison.
        return [p_values[0] < alpha], [float(p_values[0])]
    # Sort p ascending, remember original positions.
    order = sorted(range(m), key=lambda i: p_values[i])
    sorted_p = [p_values[i] for i in order]
    # Step-down: adjusted_p_i = max over j<=i of min(p_j * (m - j), 1.0)
    adjusted_sorted: list[float] = []
    running_max = 0.0
    for j, p in enumerate(sorted_p):
        candidate = min(p * (m - j), 1.0)
        running_max = max(running_max, candidate)
        adjusted_sorted.append(running_max)
    # Un-sort back to input order.
    adjusted = [0.0] * m
    for sorted_idx, original_idx in enumerate(order):
        adjusted[original_idx] = adjusted_sorted[sorted_idx]
    rejections = [p < alpha for p in adjusted]
    return rejections, adjusted
```

The formula `min(p_j * (m - j), 1.0)` with `j` zero-indexed (so the smallest p gets multiplier `m`, the next gets `m - 1`, etc.) matches the canonical Holm formulation; the `running_max` accumulator enforces the step-down monotonicity (adjusted p-values are non-decreasing when sorted by raw p). The result is byte-identical to `statsmodels.stats.multitest.multipletests(method="holm")` (asserted at §11.2).

### 4.2 Single-comparison short-circuit (Lock 2 codified)

The `if m == 1` branch is the Lock 2 codification: for a single comparison, `correction = "none"` and `adjusted == raw`. The `compare_methods` function (§2) sets `ComparisonResult.correction = "none"` when the resulting `len(pairs) == 1` and `"holm-bonferroni"` otherwise — even though `holm_bonferroni([single_p])` does the right thing on its own. The two paths are equivalent in numerical output; the `ComparisonResult.correction` field is a human-readable label for the rendered markdown (§7).

### 4.3 Why hand-coded, not statsmodels

The spec prefers the hand-coded 6-line step-down over `statsmodels.stats.multitest.multipletests` for two reasons:

1. **Dependency weight.** `statsmodels` carries a `pandas` + `patsy` + `scipy` + `numpy` dependency footprint of ~30 MB on disk; the spec needs only the Holm step-down (~6 lines of logic). Pkg-08 already depends on `scipy` (for the Welch t in §3) and on `pandas` (for the registry reader in §6.4); `statsmodels` would be a net-new dep just for one function.
2. **Audit transparency.** The hand-coded version is auditable by a reviewer in one screen; a `statsmodels` call routes through the library's correction infrastructure (which also offers BH, BY, Bonferroni, Hommel, etc.). The spec wants Holm specifically and only.

The §11.2 test asserts byte-identical output to `statsmodels` at all 4 of the comparison cardinalities tested (k = 2, 3, 4, 5); the dep is needed only at CI time and is **dev-only**, not a runtime dep.

---

## 5. 5-seed minimum policy

### 5.1 WARN at n < 5, no block

If **any** method in `method_returns` has `n_seeds < 5`, the `ComparisonResult.markdown` rendering (§7) prefixes the table with the literal string `[WARN insufficient seeds]` on a leading line. The `ComparisonResult` itself is returned normally (no raise, no exception). This policy inherits pkg-07 spec 07 §4.5 (the external-runner side of the same protocol) and matches pkg-08 README §"5 seeds 最低（< 5 警告但不阻塞）" verbatim.

The WARN policy is asymmetric: it applies even if **only one** method has `n < 5` and the others have `n >= 5`. The rationale is that under a paired t-test design (which Welch t is approximately under unequal variances), the smallest sample dictates the test's power, so a single 3-seed method poisons every comparison it participates in. The WARN surfaces this without blocking; the reviewer reading Ch6 sees the prefix and knows which numbers to discount.

### 5.2 FAIL (ValueError) at n < 2

`welch_t(sample_a, sample_b)` raises `ValueError` if either `len(sample_a) < 2` or `len(sample_b) < 2`. The locked rationale: the Welch t is mathematically undefined for n = 1 — `np.var(sample, ddof=1)` is NaN for a 1-element sample, and the t-statistic carries that NaN through. `scipy.stats.ttest_ind` silently returns `nan` in this case; the spec raises explicitly because a silent `nan` would propagate through `compare_methods` into `ComparisonResult.adjusted_p_values` and confuse downstream consumers.

A named test `test_compare_methods_raises_on_lt_2_seeds` (§11) constructs a 1-seed sample and asserts `ValueError`.

### 5.3 Markdown annotation

The rendered markdown (§7) includes a **per-method `n = K` annotation** in the comparison table's row header:

```
| Method A    | Method B    | n_a | n_b | mean_a   | mean_b   | t      | p_raw   | p_adj   | reject |
| hyper       | input_wide  | 5   | 5   | 18.3     | 14.1     | 4.21   | 0.0021  | 0.0042  | True   |
| hyper       | no_belief   | 5   | 3   | 18.3     | 11.8     | 5.04   | 0.0035  | 0.0042  | True   |  [WARN n_b<5]
```

The per-row `[WARN n_b<5]` suffix surfaces the seed-count gap **at the row level**, in addition to the top-of-table `[WARN insufficient seeds]` prefix. Both annotations are emitted; the row-level annotation lets reviewers locate which specific comparison is the under-sampled one.

---

## 6. `compare.py` CLI signature

### 6.1 Argparse surface

```python
# hyper_mve/experiments/compare.py
import argparse
import pathlib
import sys

# Lock 3: matplotlib backend set at import time, BEFORE pyplot import.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m hyper_mve.experiments.compare",
        description=(
            "Pairwise / multi-method comparison + paper-grade bar+errorbar plot.\n"
            "Three modes:\n"
            "  1. 2-method: --a runs/<a>.csv --b runs/<b>.csv\n"
            "  2. multi-method: --methods runs/<a>.csv,runs/<b>.csv,... --reference <name>\n"
            "  3. disclosure: --disclose --registry runs/registry.jsonl [--preset <name>]"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Mode 1 (2-method)
    p.add_argument("--a", type=pathlib.Path, help="CSV path for method A")
    p.add_argument("--b", type=pathlib.Path, help="CSV path for method B")
    # Mode 2 (multi-method)
    p.add_argument("--methods", type=str, help="Comma-separated list of CSV paths")
    p.add_argument("--reference", type=str, help="Method name to use as reference for pairwise tests")
    # Mode 3 (disclosure)
    p.add_argument("--disclose", action="store_true", help="Emit external-runner disclosure table")
    p.add_argument("--registry", type=pathlib.Path, default=pathlib.Path("runs/registry.jsonl"))
    p.add_argument("--preset", choices=("easy", "medium", "hard"), default=None,
                   help="Restrict disclosure table to one preset; default: emit one block per preset")
    # Common
    p.add_argument("--metric", default="return_mean",
                   help="EvalReport scalar field to compare; default return_mean")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("runs/_compare"),
                   help="Output dir for markdown + .png + sidecar .csv")
    p.add_argument("--no-plot", action="store_true",
                   help="Suppress the matplotlib bar+errorbar plot; emit markdown only")
    return p.parse_args(argv)
```

The three modes are **mutually exclusive at the CLI level**: providing both `--a/--b` and `--methods`, or providing both `--disclose` and `--a/--b`, raises `ValueError` at the dispatcher. The dispatcher logic:

```python
def dispatch(args) -> int:
    if args.disclose:
        if args.a or args.b or args.methods:
            raise ValueError("--disclose is mutually exclusive with --a/--b and --methods")
        return _run_disclose(args)
    if args.methods:
        if args.a or args.b:
            raise ValueError("--methods is mutually exclusive with --a/--b")
        return _run_multi_method(args)
    if args.a and args.b:
        return _run_two_method(args)
    raise ValueError("Must specify one of: --a + --b, --methods, --disclose")
```

### 6.2 CSV input format

Each `<method>.csv` is the result of `pandas.read_json("runs/registry.jsonl", lines=True).query("variant == '<method>'").to_csv(...)` — i.e., a flat CSV with one row per (variant, seed) tuple and columns matching the `EvalReport` schema (per spec 01 §3.1) augmented with the registry's metadata columns (per spec 05 §4):

```
variant,seed,config_hash,eval_planner_mode,c_visible,return_mean,return_sem,return_zero_shot_unseen,return_zero_shot_gap,planner_prior_return_gap,direct_inference_return_mean,planner_full_return_mean,regret_mean,oracle_ceiling_cache_hit_all,belief_c_mae,ablation_cell,...
hyper,0,abc123...,planner_full,True,18.3,0.41,15.7,2.6,2.5,15.8,18.3,3.1,True,,
hyper,1,abc123...,planner_full,True,18.7,0.39,16.0,2.7,2.4,16.3,18.7,2.9,True,,
hyper,2,abc123...,planner_full,True,17.9,0.43,15.4,2.5,2.6,15.3,17.9,3.3,True,,
hyper,3,abc123...,planner_full,True,18.5,0.40,15.8,2.7,2.5,16.0,18.5,3.0,True,,
hyper,4,abc123...,planner_full,True,18.1,0.42,15.5,2.6,2.6,15.5,18.1,3.2,True,,
```

The CLI reads the CSV with `pandas.read_csv(args.a)` then selects the `args.metric` column as the per-seed sample. The `--metric` flag's default `return_mean` is the headline; other legal values are any scalar field of `EvalReport` (regret_mean, return_zero_shot_unseen, planner_prior_return_gap, etc.). Mapping fields (like `return_per_c`) are not legal `--metric` values (the CLI errors with `ValueError("--metric must be a scalar field; got mapping-valued '<field>'")`). 

### 6.3 Disclosure mode (`--disclose`)

```
python -m hyper_mve.experiments.compare --disclose \
    --registry runs/registry.jsonl \
    [--preset easy|medium|hard] \
    [--out <dir>]
```

Reads the registry, filters to `status == "completed"` rows under the external Tier-1 + MAMBA variants (per pkg-07 spec 07 §4.2), computes the 10 columns (per pkg-07 spec 07 §4.2 schema), and emits markdown to `<out>/disclosure.md`. If `--preset` is not specified, one block per preset is emitted (easy, medium, hard in that order; rows with no data for a preset still produce the block header but no rows). The data flow is detailed in §9.5 (the `render_disclosure_table` join helper).

The 10-column schema is **mirrored verbatim from pkg-07 spec 07 §4.2**:

```
variant | preset | param_count | walltime_to_converge_seconds | lr_swept_best | final_return_mean | final_return_sem | seeds_run | smoke_pass | sourced
```

The test `test_disclosure_table_rendered_10_columns` (§11) asserts the column count and the column-name tuple match pkg-07 spec 07 §4.2 byte-identically; the assertion uses the `EXPECTED_COLUMNS` tuple from pkg-07 spec 07 §7.4 as the reference. Drift between pkg-07 spec 07 §4.2 and the implementation of `render_disclosure_table` is caught by the synchronous-edit invariant (pkg-07 spec 07 §4.2 last sentence: "any addition / removal / reordering of columns requires synchronous edits to (i) pkg-07 spec 07 §4.2, (ii) `hyper_mve/experiments/compare.py::_render_disclosure_table` (the markdown emitter), (iii) pkg-08 spec 07 stats reduction").

### 6.4 Registry reader idiom

The disclosure mode + multi-method mode both consume `runs/registry.jsonl` through the shared loader:

```python
def load_registry(registry_path: pathlib.Path) -> pd.DataFrame:
    """Per spec 05 §10.5: pandas.read_json(lines=True), group by run_id, take last,
    filter to status='completed'."""
    df = pd.read_json(registry_path, lines=True)
    df = df.sort_values("started_at_iso8601").groupby("run_id").last().reset_index()
    return df[df["status"] == "completed"]
```

The eager-load of `eval_report_path` per row is the spec 07 join idiom:

```python
def load_eval_reports(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy join: only opens eval_report.json files that are present in df."""
    df = df.copy()
    df["eval_report"] = df["eval_report_path"].apply(lambda p: json.loads(pathlib.Path(p).read_text()))
    return df
```

This idiom is reused by every join helper in §9.

---

## 7. Markdown comparison-table format

The `ComparisonResult.markdown` rendering (returned by the `__str__` or `render_markdown` property; the spec leaves the exact attribute name as an implementation choice but locks the content):

```
[WARN insufficient seeds]            # only if any n < 5
[compare_methods: k=3 → Holm-Bonferroni applied; reference=hyper; alpha=0.05]

| Method A    | Method B    | n_a | n_b | mean_a   | mean_b   | t       | p_raw    | p_adj    | reject |
|-------------|-------------|-----|-----|----------|----------|---------|----------|----------|--------|
| hyper       | input_wide  | 5   | 5   | 18.3     | 14.1     |  4.21   | 0.0021   | 0.0042   | True   |
| hyper       | no_belief   | 5   | 3   | 18.3     | 11.8     |  5.04   | 0.0035   | 0.0042   | True   |  [WARN n_b<5]
```

Row ordering: methods are sorted by sample mean **descending**; the reference (when present) is the first column of every row, the comparison method varies. The `reject` column is the boolean `adjusted_p < alpha`. The header comment line documents `k`, the correction, and the reference; this is grep-stable for reviewers reading the raw markdown.

The CLI's stdout is the markdown (so users see it inline); a copy is also written to `<out>/<run_id>_compare.md` for archival. The plot path (§8) is appended at the bottom of the markdown:

```
![compare_return_mean](./compare_return_mean.png)
```

### 7.1 Concrete example

For the call `compare --methods runs/hyper.csv,runs/input_wide.csv,runs/no_belief.csv --reference hyper`:

```
[compare_methods: k=3 → Holm-Bonferroni applied; reference=hyper; alpha=0.05]

| Method A | Method B    | n_a | n_b | mean_a | mean_b | t      | p_raw   | p_adj   | reject |
|----------|-------------|-----|-----|--------|--------|--------|---------|---------|--------|
| hyper    | input_wide  | 5   | 5   | 18.30  | 14.10  | 4.21   | 0.00210 | 0.00210 | True   |
| hyper    | no_belief   | 5   | 5   | 18.30  | 11.80  | 5.04   | 0.00035 | 0.00070 | True   |

![compare_return_mean](./compare_return_mean.png)
```

Both raw p-values are < 0.05, so both rejections are True. The Holm step-down applied: with 2 comparisons, the smallest raw p (0.00035) is multiplied by 2 to give adjusted 0.00070; the next raw p (0.00210) is multiplied by 1 to give adjusted 0.00210 — both adjusted < 0.05. Reviewers can trace `p_adj` back to `p_raw` via the §4.1 algorithm.

---

## 8. Comparison plot (Agg backend; paper-grade)

### 8.1 Plot structure

A single matplotlib figure per `compare_methods` call. **X-axis**: method names, sorted by sample mean descending (matching the markdown order). **Y-axis**: the chosen `--metric` (default `return_mean`). **Error bars**: ±1 SEM (the `sem_per_method` field of `ComparisonResult`). **Significance markers**: stars above bars on every non-reference method:

- `***` for `p_adj < 0.001`
- `**` for `p_adj < 0.01`
- `*` for `p_adj < 0.05`
- no marker for `p_adj >= 0.05` (or `correction=="none"` with `p_raw >= 0.05`)

The reference method (when present) has no marker (it is the comparison baseline; star semantics don't apply to itself). When the call is all-pairs (no reference), markers are placed above every pair where the comparison is significant; the labeling is `*` adjacent to the lower-mean method of the pair. This avoids the visual ambiguity of "which pair does this star describe" for k ≥ 4.

### 8.2 File outputs

```
<out_dir>/compare_<metric>.png       # 300 DPI rendering; bbox_inches='tight'
<out_dir>/compare_<metric>.csv       # sidecar: one row per method with columns
                                     #   method, n, mean, sem, t_vs_reference, p_raw, p_adj, reject
```

The sidecar CSV is the **single-source reproducibility file**: every number in the PNG is in the CSV, and the CSV is what spec 07 stats reduction reads when regenerating the plot from a future code revision. The 300 DPI is the target for paper-grade rendering; the test asserts only `os.path.getsize(...) > 0` per Lock 3 (DPI bit-exactness is not asserted at CI time because CI antialiasing varies).

### 8.3 Matplotlib backend hygiene

`matplotlib.use("Agg")` is the **first non-comment line** of `compare.py` after the module docstring. Any `import matplotlib.pyplot as plt` must be physically below the `use("Agg")` call (matplotlib raises a warning if the backend is changed after pyplot import). The order is asserted by `test_compare_plot_renders_headless` (§11) via the import-order check `matplotlib.get_backend().lower() == "agg"` after `import hyper_mve.experiments.compare`.

`plt.show()` is **never called**. Every figure is constructed via `fig, ax = plt.subplots(...)`, populated, saved via `fig.savefig(...)`, and disposed via `plt.close(fig)`. The explicit `close` releases the figure's memory (matplotlib otherwise accumulates figures into a list, leaking under the sweep harness which may call `compare` 100+ times in a single session). A named test `test_compare_plot_closes_figures` (§11) constructs 100 figures and asserts `len(plt.get_fignums()) == 0` at the end (no figure handle leaked).

### 8.4 Color palette

The matplotlib default `tab10` cycle (10 distinct hues) is the default palette. No custom palette is locked at SDD time — reviewers may extend with a paper-style palette at implementation time (e.g., a colorblind-safe palette like `seaborn`'s `colorblind` or matplotlib's `tab10` with explicit reordering). The locked invariant is "the reference method is the first bar" (per §8.1 sort order); the colors themselves are not.

---

## 9. Stats-layer joins (paired-mode + c_hidden + disclosure)

Five join helpers live in `stats.py` and consume `RegistryRow` sequences materialised by the CLI from `runs/registry.jsonl` via `pandas.read_json(lines=True)`. Each helper is pure (no I/O, no global state); the caller passes the row sequence in.

### 9.1 `compute_planner_prior_gap(registry_rows)` — planner_full row in-row gap

```python
def compute_planner_prior_gap(registry_rows: Sequence[dict]) -> Mapping[tuple[str, int], float]:
    """Reads planner_prior_return_gap directly from planner_full rows.
    
    Per spec 03 §5 (per-mode population table) + §6.2 (stats-layer join sketch — explicitly says 'NOT a cross-row join'): every row carries its own in-row gap = planner_full_return_mean
    - direct_inference_return_mean (sourced from run_eval's planner_prior_gap output).
    The gap is NOT computed as a cross-row join; it is read from the planner_full
    row's own EvalReport.
    
    Returns dict[(variant, seed) -> gap]. Skips:
      - external_* variants (no planner; gap is sentinel 0.0; see spec 03 §4.3)
      - rows with eval_planner_mode != "planner_full" (only canonical row used)
    """
    gaps: dict[tuple[str, int], float] = {}
    for row in registry_rows:
        if row["variant"].startswith("external_"):
            continue
        if row.get("eval_planner_mode") != "planner_full":
            continue
        report = _load_report(row["eval_report_path"])
        gaps[(row["variant"], row["seed"])] = float(report["planner_prior_return_gap"])
    return gaps
```

Spec 03 §6.2's verbatim sketch lives here in the implementation. The return type `Mapping[tuple[str, int], float]` is hashable and JSON-serialisable; downstream (`compare_methods` consumer) materialises it as a per-variant array.

### 9.2 `compute_cross_mode_deltas(registry_rows)` — cross-mode 4-mode comparison join

```python
def compute_cross_mode_deltas(
    registry_rows: Sequence[dict],
) -> Mapping[tuple[str, int, str, str], float]:
    """Joins (variant, seed, config_hash) tuples across eval_planner_mode literals.
    
    Returns dict[(variant, seed, mode_a, mode_b) -> mode_a.return_mean - mode_b.return_mean].
    Used for:
      - The 4-mode comparison plot (spec 03 §6.1, §6.2)
      - CRN-isolated eval-time delta (planner_full - planner_no_crn)
      - CoordDesc-isolated eval-time delta (planner_full - planner_no_coord_desc)
    """
    by_key: dict[tuple[str, int, str], dict[str, dict]] = {}
    for row in registry_rows:
        if row["variant"].startswith("external_"):
            continue
        key = (row["variant"], row["seed"], row["config_hash"])
        by_key.setdefault(key, {})[row["eval_planner_mode"]] = row
    deltas: dict[tuple[str, int, str, str], float] = {}
    for (variant, seed, _hash), modes in by_key.items():
        for mode_a in modes:
            for mode_b in modes:
                if mode_a == mode_b:
                    continue
                deltas[(variant, seed, mode_a, mode_b)] = float(
                    modes[mode_a]["return_mean"] - modes[mode_b]["return_mean"]
                )
    return deltas
```

The join is keyed on `(variant, seed, config_hash)` to guarantee that two rows compared share an identical training config — per spec 03 §6.1's row-emission contract, this is the canonical join key. Deltas are emitted for every ordered pair of modes present at that key, so a fully-populated 4-mode `(variant, seed)` produces 4 × 3 = 12 deltas; downstream consumers (compare CLI 4-mode plot) typically filter to the 3 deltas with `planner_full` as `mode_a`.

### 9.3 `compute_c_hidden_gap(registry_rows)` — paired c_visible=True/False join

```python
def compute_c_hidden_gap(registry_rows: Sequence[dict]) -> Mapping[tuple[str, int], float]:
    """Joins (variant, seed) tuples across cfg.env.c_visible boolean.
    
    Returns dict[(variant, seed) -> return_mean_visible - return_mean_hidden].
    
    Per spec 02 §3 c_hidden paired protocol: every (variant, seed) tuple has
    exactly two rows in the registry — one with c_visible=True and one with
    c_visible=False. The gap surfaces BeliefNet quality at c_hidden time
    (Theory Audit Q7).
    """
    by_key: dict[tuple[str, int], dict[bool, float]] = {}
    for row in registry_rows:
        report = _load_report(row["eval_report_path"])
        c_vis = bool(report["c_visible"])
        key = (row["variant"], row["seed"])
        by_key.setdefault(key, {})[c_vis] = float(report["return_mean"])
    gaps: dict[tuple[str, int], float] = {}
    for key, returns_by_vis in by_key.items():
        if True in returns_by_vis and False in returns_by_vis:
            gaps[key] = returns_by_vis[True] - returns_by_vis[False]
    return gaps
```

The Ch6 c_hidden table consumes this dict directly. A `(variant, seed)` tuple with only one of the two visibility modes (e.g., the sweep harness emitted only the `c_visible=True` row) is silently skipped — the consumer (compare CLI) emits a `[WARN: variant=<v> seed=<s> missing c_hidden=<bool> row]` stderr line for each skipped tuple.

### 9.4 `compute_regret_per_method(registry_rows)` — per-variant regret aggregate

```python
def compute_regret_per_method(
    registry_rows: Sequence[dict],
) -> Mapping[str, tuple[float, float]]:
    """Averages EvalReport.regret_mean across seeds per variant.
    
    Returns dict[variant -> (mean_regret, sem_regret)].
    
    Skipped per variant if ANY (variant, seed) row has oracle_ceiling_cache_hit
    containing False (per spec 02 Lock 3: regret is NaN on cache miss).
    """
    by_variant: dict[str, list[float]] = {}
    skip_variants: set[str] = set()
    for row in registry_rows:
        report = _load_report(row["eval_report_path"])
        if not all(report["oracle_ceiling_cache_hit"].values()):
            skip_variants.add(row["variant"])
            continue
        by_variant.setdefault(row["variant"], []).append(float(report["regret_mean"]))
    out: dict[str, tuple[float, float]] = {}
    for variant, values in by_variant.items():
        if variant in skip_variants:
            continue
        arr = np.asarray(values, dtype=float)
        mean = float(arr.mean())
        sem = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else float("nan")
        out[variant] = (mean, sem)
    return out
```

A variant with any cache-miss row is omitted entirely (not partially aggregated) — the conservative interpretation per spec 02 Lock 3 (`oracle_ceiling_cache_hit` containing False means regret is NaN and any partial aggregate would be misleading). The compare CLI's regret table emits a `[SKIP <variant> oracle_ceiling cache_miss]` stderr line for each skipped variant.

### 9.5 `render_disclosure_table(registry_rows, preset)` — 10-column external disclosure

```python
def render_disclosure_table(
    registry_rows: Sequence[dict],
    preset: str | None = None,
) -> str:
    """Emits the 10-column external-runner disclosure table markdown.
    
    Schema (verbatim from pkg-07 spec 07 §4.2):
      variant | preset | param_count | walltime_to_converge_seconds | lr_swept_best
            | final_return_mean | final_return_sem | seeds_run | smoke_pass | sourced
    
    Behavior:
      - Filters to status="completed" + variant.startswith("external_").
      - If preset is None, emits one block per preset (easy, medium, hard).
      - Computes lr_swept_best per (variant, preset) via argmax over LR.
      - Aggregates per (variant, preset, lr=lr_swept_best) for the 6 numeric columns.
      - Handles MAMBA not-sourced row per pkg-07 spec 07 §4.6 (all numeric cells = '-').
      - Emits MARIE/GA exclusion footnote per pkg-07 spec 07 §4.7.
      - 5-seed minimum WARN per pkg-07 spec 07 §4.5 (footnote, non-blocking).
    
    Returns the rendered markdown string (caller writes to disclosure.md).
    """
```

The implementation reverses the data flow per pkg-07 spec 07 §5 step 3: "compare CLI emits disclosure table markdown via `python -m hyper_mve.experiments.compare --disclose`. The CLI renders the markdown table per preset (one table block per preset, columns as in §4.2), preceded by a preset name header (`## Disclosure table — easy preset`, `## Disclosure table — medium preset`), and emits the footnotes block at the end of each preset block. The output is appended to `runs/<exp_id>/disclosure.md`."

The 5-seed WARN policy (pkg-07 spec 07 §4.5) and the MAMBA not-sourced row (pkg-07 spec 07 §4.6) and the MARIE/GA exclusion footnote (pkg-07 spec 07 §4.7) are all handled inside `render_disclosure_table` — no caller-side logic is required. The 10-column count is asserted by `test_disclosure_table_rendered_10_columns` (§11).

---

## 10. New cfg fields

**NONE.** Spec 07 contributes **zero** new cfg fields. The stats parameters (`alpha`, `n_seeds_minimum`) are exposed as **CLI flags or function arguments**, not as `cfg.eval.*` fields. The per-impl constant policy locked at pkg-07 spec 05 §11 ("per-impl tuning constants live with their implementation, not on cfg") and pkg-08 spec 04 §6 ("μP-self-check constants live in `mup_verification.py` per-impl, not on cfg.mup") applies here verbatim: stats hyperparameters are per-call inspection knobs, not training-time configuration.

The README §"5 新 cfg 字段穷举" list remains exactly the 5 fields contributed by specs 02 (`cfg.env.c_visible`), 03 (`cfg.eval.eval_planner_mode` + `cfg.eval.eval_use_planner_direct_inference`), and 06 (`cfg.train.randomize_order` + `cfg.train.mve_joint_enumerate`). Spec 07 does not append to that list.

The locked rationale: a `cfg.stats.alpha` field would shift the responsibility of "what alpha did Ch6 use" from the CLI invocation (which is logged with `argv` in TB / sweep stderr) to a config file (which is silently inherited from a preset). Reviewers can grep the sweep log for `--alpha` invocations; they cannot easily reconstruct a `cfg.stats.alpha` history from a single ckpt. CLI flags are the more auditable surface for analysis-time parameters.

---

## 11. Test contract — ≥ 8 named tests

Located under `tests/experiments/`. Each name below corresponds to a specific Lock or §3/§4/§5/§6/§8/§9 contract.

### 11.1 `test_welch_t_against_scipy_reference` — Lock 1 + §3.1

For 5 random-seeded sample pairs (with NumPy seeds 0..4), call `welch_t(a, b)` and assert the returned `(t, p)` is byte-identical to `scipy.stats.ttest_ind(a, b, equal_var=False)`. Drift detector: a monkeypatch on `scipy.stats.ttest_ind` that records the kwargs and asserts `equal_var=False` is the literal value:

```python
def test_welch_t_against_scipy_reference():
    import scipy.stats
    captured_kwargs = []
    orig = scipy.stats.ttest_ind
    def recorder(*a, **kw):
        captured_kwargs.append(kw)
        return orig(*a, **kw)
    monkeypatch.setattr(scipy.stats, "ttest_ind", recorder)
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = rng.normal(0, 1, size=8)
        b = rng.normal(0.3, 1.5, size=8)
        t, p = welch_t(a, b)
        # Reference computation directly
        ref = orig(a, b, equal_var=False)
        assert t == pytest.approx(float(ref.statistic))
        assert p == pytest.approx(float(ref.pvalue))
    assert all(kw == {"equal_var": False} for kw in captured_kwargs)
```

Asserted invariants: byte-identical `(t, p)`; the kwarg `equal_var=False` is literal in every call.

### 11.2 `test_holm_bonferroni_against_statsmodels` — Lock 2 + §4

Random-seeded comparison vs `statsmodels.stats.multitest.multipletests(method="holm")` over k ∈ {2, 3, 4, 5}:

```python
@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_holm_bonferroni_against_statsmodels(k):
    from statsmodels.stats.multitest import multipletests
    rng = np.random.default_rng(42)
    p_values = rng.uniform(0.001, 0.5, size=k).tolist()
    rejections, adjusted = holm_bonferroni(p_values, alpha=0.05)
    ref_rejections, ref_adjusted, _, _ = multipletests(p_values, alpha=0.05, method="holm")
    np.testing.assert_allclose(adjusted, ref_adjusted, rtol=1e-12)
    assert rejections == list(ref_rejections)
```

Asserted invariants: adjusted p-values byte-identical to `statsmodels` at rel=1e-12 over 4 cardinalities.

### 11.3 `test_holm_bonferroni_skipped_for_single_comparison` — Lock 2 short-circuit

```python
def test_holm_bonferroni_skipped_for_single_comparison():
    rejections, adjusted = holm_bonferroni([0.04], alpha=0.05)
    assert adjusted == [0.04]      # byte-identical to raw
    assert rejections == [True]
```

Asserted invariants: `adjusted == raw` byte-identically when `len(p_values) == 1`.

### 11.4 `test_compare_methods_pairwise_against_reference` — §2 compare_methods + §7

Known input: 3 methods, 5 seeds each, with means {hyper: 18.3, input_wide: 14.1, no_belief: 11.8}:

```python
def test_compare_methods_pairwise_against_reference():
    rng = np.random.default_rng(0)
    method_returns = {
        "hyper":      rng.normal(18.3, 0.4, size=5),
        "input_wide": rng.normal(14.1, 0.4, size=5),
        "no_belief":  rng.normal(11.8, 0.4, size=5),
    }
    result = compare_methods(method_returns, reference="hyper", alpha=0.05)
    assert result.reference == "hyper"
    assert set(result.pairs) == {("hyper", "input_wide"), ("hyper", "no_belief")}
    assert result.correction == "holm-bonferroni"
    assert all(result.rejections.values())
```

Asserted invariants: 2 pairs emitted (hyper vs input_wide; hyper vs no_belief); correction = `"holm-bonferroni"`; both rejected.

### 11.5 `test_compare_methods_warns_on_lt_5_seeds` — §5.1

```python
def test_compare_methods_warns_on_lt_5_seeds():
    rng = np.random.default_rng(0)
    method_returns = {
        "hyper":      rng.normal(18.3, 0.4, size=3),
        "input_wide": rng.normal(14.1, 0.4, size=5),
    }
    result = compare_methods(method_returns, alpha=0.05)
    md = result.render_markdown()
    assert "[WARN insufficient seeds]" in md
```

Asserted invariants: WARN prefix appears verbatim in the rendered markdown.

### 11.6 `test_compare_methods_raises_on_lt_2_seeds` — §5.2

```python
def test_compare_methods_raises_on_lt_2_seeds():
    method_returns = {
        "hyper":      np.asarray([18.3]),
        "input_wide": np.asarray([14.1, 14.0, 14.2, 14.05, 14.15]),
    }
    with pytest.raises(ValueError, match="Welch t undefined for n < 2"):
        compare_methods(method_returns)
```

Asserted invariants: `ValueError` raised; message matches the documented one.

### 11.7 `test_compare_plot_renders_headless` — Lock 3

```python
def test_compare_plot_renders_headless(tmp_path):
    import hyper_mve.experiments.compare as compare_module
    import matplotlib
    assert matplotlib.get_backend().lower() == "agg"
    rng = np.random.default_rng(0)
    method_returns = {
        "hyper":      rng.normal(18.3, 0.4, size=5),
        "input_wide": rng.normal(14.1, 0.4, size=5),
    }
    result = compare_methods(method_returns)
    out_path = tmp_path / "compare_return_mean.png"
    compare_module.render_plot(result, out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0
```

Asserted invariants: backend is `"agg"` after import; PNG file exists and is non-empty.

### 11.8 `test_disclosure_table_rendered_10_columns` — §6.3 + §9.5

```python
def test_disclosure_table_rendered_10_columns():
    EXPECTED_COLUMNS = (
        "variant", "preset", "param_count", "walltime_to_converge_seconds",
        "lr_swept_best", "final_return_mean", "final_return_sem",
        "seeds_run", "smoke_pass", "sourced",
    )
    mock_rows = [
        {"variant": "external_mappo", "preset": "easy", "status": "completed",
         "eval_report_path": "<stub>", "config_hash": "abc", "seed": 0,
         # ... full row fields per spec 05 §4 schema ...
        },
        # Additional rows for 5 seeds × 3 LRs ...
    ]
    md = render_disclosure_table(mock_rows, preset="easy")
    header = md.splitlines()[2]    # "## Disclosure table — easy preset" + table header
    columns = tuple(c.strip() for c in header.split("|") if c.strip())
    assert columns == EXPECTED_COLUMNS, f"Disclosure column drift: {columns}"
```

Asserted invariants: column tuple matches pkg-07 spec 07 §4.2 byte-identically.

### 11.9 `test_planner_prior_gap_join_per_seed` — §9.1

Synthetic `planner_full` rows; assert the dict[(variant, seed) -> gap] is correctly populated:

```python
def test_planner_prior_gap_join_per_seed(tmp_path):
    rows = []
    for seed in range(5):
        report_path = tmp_path / f"hyper_seed{seed}.json"
        report_path.write_text(json.dumps({
            "planner_prior_return_gap": 2.5 + 0.1 * seed,
            # ... other EvalReport fields ...
        }))
        rows.append({
            "variant": "hyper",
            "seed": seed,
            "eval_planner_mode": "planner_full",
            "eval_report_path": str(report_path),
        })
    gaps = compute_planner_prior_gap(rows)
    assert gaps == {("hyper", s): 2.5 + 0.1 * s for s in range(5)}
```

Asserted invariants: dict has exactly the expected 5 keys with the documented values.

### 11.10 `test_c_hidden_gap_join_per_seed` — §9.3

Synthetic `c_visible=True / False` row pairs; assert the gap dict is correctly populated:

```python
def test_c_hidden_gap_join_per_seed(tmp_path):
    rows = []
    for seed in range(5):
        for c_vis, ret in [(True, 18.3 + 0.1 * seed), (False, 14.1 + 0.1 * seed)]:
            report_path = tmp_path / f"hyper_seed{seed}_vis{c_vis}.json"
            report_path.write_text(json.dumps({
                "c_visible": c_vis,
                "return_mean": ret,
            }))
            rows.append({
                "variant": "hyper",
                "seed": seed,
                "eval_report_path": str(report_path),
            })
    gaps = compute_c_hidden_gap(rows)
    expected = {("hyper", s): (18.3 + 0.1 * s) - (14.1 + 0.1 * s) for s in range(5)}
    for key, gap in expected.items():
        assert gaps[key] == pytest.approx(gap, rel=1e-9)
```

Asserted invariants: dict has 5 keys with the documented gap values (≈ 4.2 each).

---

## 12. Downstream code patches (NEW files; declared in spec 08 §5)

This spec contributes **two NEW files** and **zero modifications** to existing files. Honest line-count accounting:

### 12.1 `hyper_mve/experiments/stats.py` — NEW file

- **Tag**: `[new-file]`
- **Estimated size**: ~250 lines (3 free functions × ~25 lines each + 1 dataclass × ~25 lines + 5 join helpers × ~30 lines each = ~225 lines + ~25 lines module preamble/docstrings).
- **Content**: §2 public API verbatim — `welch_t` + `holm_bonferroni` + `compare_methods` + `ComparisonResult` + 5 join helpers (`compute_planner_prior_gap`, `compute_cross_mode_deltas`, `compute_c_hidden_gap`, `compute_regret_per_method`, `render_disclosure_table`).
- **Dependencies added**: scipy (already present; used by `welch_t`); pandas (already present; used by the join helpers' caller, not by the helpers themselves). No new dependency.

### 12.2 `hyper_mve/experiments/compare.py` — NEW file

- **Tag**: `[new-file]` + `[downstream-cli]`
- **Estimated size**: ~200 lines (argparse setup ~40 lines + 3 dispatch routes × ~30 lines each + markdown emitter ~30 lines + plot renderer ~30 lines + disclosure dispatcher ~15 lines = ~205 lines).
- **Content**: §6 CLI surface + §7 markdown emission + §8 plot rendering + §9.5 disclosure table dispatcher.
- **Dependencies added**: matplotlib (already present in the project's dev deps; used by every plotting test); no new dependency.

### 12.3 NO modifications to existing files

Spec 07 does **not** modify any of:
- `hyper_mve/configs/*` — no new cfg fields (per §10).
- `hyper_mve/training/evaluation.py:run_eval` — unchanged (Lock 1 of spec 01 still holds; spec 07 consumes the `EvalReport` schema, never the in-training eval).
- `hyper_mve/eval/*` — unchanged; spec 07 consumes `EvalReport` (defined by spec 01).
- `hyper_mve/experiments/sweep.py` — unchanged; spec 07 consumes the registry (defined by spec 05).
- pkg-07 `hyper_mve/baselines/*` — unchanged; spec 07 consumes the registry rows whose `variant` field references pkg-07's `REGISTRY` keys.

The "no modifications" invariant is asserted by spec 08 §5's downstream-patch-list, which enumerates exactly the 3 patches contributed by specs 02 + 06 (Pkg-02 obs-mask + Pkg-05 planner flag + Pkg-05 CLI rename) and does NOT add a spec-07-contributed patch.

---

## 13. Integration hooks (cross-spec)

| Producer | Producing | Consumed by spec 07 at |
|----------|-----------|------------------------|
| **spec 01** unified evaluator | `EvalReport` schema — `return_mean`, `return_sem`, `episodes_per_c`, `planner_prior_return_gap`, `direct_inference_return_mean`, `planner_full_return_mean`, `regret_per_c`, `oracle_ceiling_cache_hit`, `c_visible`, `eval_planner_mode`, `belief_c_mae` | §6.2 CSV column source + §9.1–9.5 join helpers |
| **spec 02** zero-shot + c_hidden + regret | `EvalReport.regret_per_c` + `oracle_ceiling_cache_hit` + `c_visible` field | §9.3 c_hidden gap join + §9.4 regret-per-method aggregate (Lock 3 cache-miss skip) |
| **spec 03** four-mode planner dispatch | Paired-mode row emission (one row per `(variant, seed, eval_planner_mode)` triple) | §9.1 planner-prior-gap in-row read + §9.2 cross-mode deltas join |
| **spec 05** sweep harness + RunRegistry | `runs/registry.jsonl` JSONL append-only + 23-key row schema + `eval_report_path` lazy join | §6.4 registry reader + §9 join helper input |
| **spec 06** ablation CLI + cells | `ablation_cell` field on every ablation row | §6 `--methods` comparison consumer + §7 markdown grouping by `ablation_cell` |
| **spec 08** integration contracts | Final codification of spec 07's NEW-file declarations + cross-pkg consumption lock | §12 patch tags reverse-consumed |
| **pkg-07 spec 07** fairness protocol §4-§5 | 10-column disclosure-table schema + 5-seed WARN policy + MAMBA not-sourced behavior + MARIE/GA exclusion footnote | §6.3 `--disclose` dispatcher + §9.5 `render_disclosure_table` (REVERSE-CONSUMED) |

The pkg-07 reverse-consumption is the most consequential cross-pkg anchor: pkg-07 spec 07 §5 step 3 designates `compare --disclose` as the implementation site, and pkg-08 spec 07 (this spec) provides that implementation. Drift between pkg-07 spec 07 §4.2 (10-column schema) and pkg-08 spec 07 §9.5 (`render_disclosure_table`) is caught by the synchronous-edit invariant (pkg-07 spec 07 §4.2 last paragraph) and by the `test_disclosure_table_rendered_10_columns` test (§11.8).

---

## 14. Cross-references

### Upstream anchors

- **pkg-08 design §4 D9** — `--ablation` CLI canned YAML dispatch; spec 07 consumes the resulting `ablation_cell` field as the groupby key for ablation-table comparison plots.
- **pkg-08 design §4 D10** — 5 new cfg fields; spec 07 contributes ZERO new fields (per §10) but consumes the 5 declared.
- **pkg-08 README C8-ABL-STAT1** — Welch t-test correctness gate; spec 07 §3 + §11.1 enforces.
- **pkg-08 README C8-ABL-PLOT1** — paper-grade comparison plot gate; spec 07 §8 + §11.7 enforces.
- **pkg-08 README Ch6.2.3 (statistics protocol)** — 5-seed minimum + Welch t + Holm-Bonferroni; spec 07 §3 + §4 + §5 codifies.
- **pkg-08 spec 01 §3** — `EvalReport` schema; spec 07 consumes the 12 fields enumerated in §13 table row 1.
- **pkg-08 spec 02 §3 + §4** — c_hidden paired protocol + regret-vs-oracle-ceiling; spec 07 §9.3 + §9.4 consumes.
- **pkg-08 spec 03 §5 + §6.2** — per-mode `EvalReport` population + paired-mode row emission with the canonical "NOT a cross-row join" lock; spec 07 §9.1 + §9.2 consumes.
- **pkg-08 spec 05 §4 + §10.5** — `RegistryRow` 23-key schema + `pandas.read_json(lines=True)` reader contract; spec 07 §6.4 consumes verbatim.
- **pkg-08 spec 06 §3** — 5 canned YAMLs; the resulting `ablation_cell` field is the groupby axis for §6 multi-method comparisons.
- **pkg-08 spec 08 §5** — canonical "下游补丁声明" downstream-file list; spec 07 §12 contributes 2 new files with `[new-file]` tags.
- **pkg-07 spec 07 §4.2 + §4.5 + §4.6 + §4.7 + §5 step 3** — 10-column external disclosure-table schema + 5-seed WARN-only policy + MAMBA not-sourced row stability + MARIE/GA exclusion footnote + `compare --disclose` data-flow step; spec 07 §6.3 + §9.5 reverse-consumes.
- **Theory Audit §10.3** — Welch t + 5 seeds + multiple-comparison correction necessary; spec 07 §3 + §4 + §5 implements.

### Existing repo ground-truth anchors

- `hyper_mve/experiments/sweep.py` (declared by spec 05) — produces `runs/registry.jsonl`; spec 07 consumes via §6.4 reader.
- `hyper_mve/eval/eval_report.py` (declared by spec 01) — produces `EvalReport`; spec 07 consumes the JSON dump at `runs/<run_tag>/eval_report.json`.
- `scipy.stats.ttest_ind` (pre-existing dep) — wrapped by `welch_t` per §3.
- `matplotlib.pyplot` (pre-existing dep) — used by §8 plot renderer with `Agg` backend per Lock 3.

---

## Anchors (verbatim grep targets for spec 08 §6 drift detector)

- `pkg-08 spec 07 Lock 1: Welch t-test is THE pairwise comparison primitive; scipy.stats.ttest_ind(equal_var=False) ONLY`
- `pkg-08 spec 07 Lock 2: Holm-Bonferroni applies WHEN AND ONLY WHEN k >= 3 methods (>= 2 pairwise tests)`
- `pkg-08 spec 07 Lock 3: matplotlib Agg backend ONLY; matplotlib.use("Agg") at module-import time; no plt.show()`
- `pkg-08 spec 07 §2: stats.py public API — 3 free functions + 1 frozen dataclass + 5 join helpers`
- `pkg-08 spec 07 §3.2: welch_t raises ValueError on NaN; caller must filter NaN before calling`
- `pkg-08 spec 07 §4.1: holm_bonferroni hand-coded 6-line step-down (not statsmodels dep)`
- `pkg-08 spec 07 §5.1: [WARN insufficient seeds] prefix when any method has n < 5`
- `pkg-08 spec 07 §5.2: welch_t raises ValueError for n < 2 (Welch t undefined)`
- `pkg-08 spec 07 §6.3: --disclose dispatcher emits 10-column external runner table per pkg-07 spec 07 §4.2`
- `pkg-08 spec 07 §8.3: matplotlib.use("Agg") is the first non-comment line of compare.py; no plt.show() anywhere`
- `pkg-08 spec 07 §9.1: compute_planner_prior_gap reads in-row gap from planner_full rows (NOT cross-row join)`
- `pkg-08 spec 07 §9.5: render_disclosure_table mirrors pkg-07 spec 07 §4.2 10-column schema byte-identically`
- `pkg-08 spec 07 §10: ZERO new cfg fields; stats parameters are CLI flags / function args`
- `pkg-08 spec 07 §12.1: hyper_mve/experiments/stats.py — NEW file ~250 lines [new-file]`
- `pkg-08 spec 07 §12.2: hyper_mve/experiments/compare.py — NEW file ~200 lines [new-file] [downstream-cli]`
- `pkg-08 spec 07 §12.3: NO modifications to existing files`
- `pkg-08 spec 07 §13: reverse-consumed pkg-07 spec 07 §4.2 + §4.5 + §4.6 + §4.7 + §5 step 3 (disclosure table)`
