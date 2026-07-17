"""pkg-08 spec 07 — Statistics: Welch t + Holm-Bonferroni + 5 stats joins.

Three hard locks:

* **Lock 1** — Welch t-test (`scipy.stats.ttest_ind(equal_var=False)`) is THE
  pairwise comparison primitive; equal-variance Student t and Mann-Whitney U
  are forbidden.
* **Lock 2** — Holm-Bonferroni applies WHEN AND ONLY WHEN ``len(p_values) >= 2``;
  for a single comparison ``correction = "none"`` and ``adjusted == raw``.
* **Lock 3** (compare.py) — matplotlib Agg backend only; not enforced here.

Public surface (3 free functions + 1 frozen dataclass + 5 join helpers):

    welch_t(a, b) -> (t, p)
    holm_bonferroni(p_values, alpha=0.05) -> (rejections, adjusted_p_values)
    compare_methods(method_returns, *, reference, alpha=0.05) -> ComparisonResult

    compute_planner_prior_gap(rows) -> Mapping[(variant, seed), float]
    compute_cross_mode_deltas(rows) -> Mapping[(variant, seed, mode_a, mode_b), float]
    compute_c_hidden_gap(rows)    -> Mapping[(variant, seed), float]
    compute_regret_per_method(rows) -> Mapping[variant, (mean, sem)]
    render_disclosure_table(rows, preset) -> str
"""
from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Sequence

__all__ = [
    "welch_t",
    "holm_bonferroni",
    "compare_methods",
    "ComparisonResult",
    "compute_planner_prior_gap",
    "compute_cross_mode_deltas",
    "compute_c_hidden_gap",
    "compute_regret_per_method",
    "render_disclosure_table",
    "DISCLOSURE_COLUMNS",
]


# pkg-07 spec 07 §4.2 + pkg-08 spec 07 §6.3 / §11.8 — verbatim 10-tuple.
DISCLOSURE_COLUMNS: tuple[str, ...] = (
    "variant",
    "preset",
    "param_count",
    "walltime_to_converge_seconds",
    "lr_swept_best",
    "final_return_mean",
    "final_return_sem",
    "seeds_run",
    "smoke_pass",
    "sourced",
)


# ===== Welch t-test (Lock 1) =============================================

def welch_t(sample_a: Sequence[float], sample_b: Sequence[float]) -> tuple[float, float]:
    """Welch unequal-variance t-test (two-sided).

    Wraps ``scipy.stats.ttest_ind(a, b, equal_var=False)``. Raises
    ``ValueError`` on NaN or n<2 — silent drop is forbidden (spec 07 §3.2).
    """
    import numpy as np
    from scipy.stats import ttest_ind

    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("welch_t: samples must be 1-D")
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError(
            "welch_t: NaN in sample. Caller must filter NaN rows before "
            "calling. See pkg-08 spec 07 §3.2 (NaN-handling) and §9.4 "
            "(oracle-ceiling cache-miss filtering)."
        )
    if len(a) < 2 or len(b) < 2:
        raise ValueError(
            f"welch_t: Welch t undefined for n < 2. Got n_a={len(a)}, "
            f"n_b={len(b)}. See pkg-08 spec 07 §5.2."
        )
    result = ttest_ind(a, b, equal_var=False)
    return float(result.statistic), float(result.pvalue)


# ===== Holm-Bonferroni step-down (Lock 2) ================================

def holm_bonferroni(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> tuple[list[bool], list[float]]:
    """Step-down Holm-Bonferroni correction (hand-coded; not statsmodels dep).

    Returns ``(rejection_per_comparison, adjusted_p_values)`` — both lists of
    length ``len(p_values)``, in the same input order. For a single
    comparison (``len == 1``), no correction is applied (Lock 2).
    """
    m = len(p_values)
    if m == 0:
        return [], []
    if m == 1:
        return [float(p_values[0]) < alpha], [float(p_values[0])]
    order = sorted(range(m), key=lambda i: p_values[i])
    sorted_p = [float(p_values[i]) for i in order]
    adjusted_sorted: list[float] = []
    running_max = 0.0
    for j, p in enumerate(sorted_p):
        candidate = min(p * (m - j), 1.0)
        running_max = max(running_max, candidate)
        adjusted_sorted.append(running_max)
    adjusted = [0.0] * m
    for sorted_idx, original_idx in enumerate(order):
        adjusted[original_idx] = adjusted_sorted[sorted_idx]
    rejections = [p < alpha for p in adjusted]
    return rejections, adjusted


# ===== ComparisonResult dataclass (spec 07 §2) ===========================

@dataclass(frozen=True)
class ComparisonResult:
    methods: tuple[str, ...]
    reference: str | None
    pairs: tuple[tuple[str, str], ...]
    t_statistics: Mapping[tuple[str, str], float]
    raw_p_values: Mapping[tuple[str, str], float]
    adjusted_p_values: Mapping[tuple[str, str], float]
    rejections: Mapping[tuple[str, str], bool]
    n_per_method: Mapping[str, int]
    mean_per_method: Mapping[str, float]
    sem_per_method: Mapping[str, float]
    correction: Literal["none", "holm-bonferroni"]
    alpha: float

    def render_markdown(self) -> str:  # spec 07 §7
        lines: list[str] = []
        if any(n < 5 for n in self.n_per_method.values()):
            lines.append("[WARN insufficient seeds]")
        ref_clause = f"; reference={self.reference}" if self.reference else ""
        lines.append(
            f"[compare_methods: k={len(self.methods)} -> "
            f"{self.correction} applied{ref_clause}; alpha={self.alpha}]"
        )
        lines.append("")
        header = (
            "| Method A | Method B | n_a | n_b | mean_a | mean_b | "
            "t | p_raw | p_adj | reject |"
        )
        sep = "|" + "|".join(["---"] * 10) + "|"
        lines.append(header)
        lines.append(sep)
        for a, b in self.pairs:
            n_a = self.n_per_method.get(a, 0)
            n_b = self.n_per_method.get(b, 0)
            mean_a = self.mean_per_method.get(a, float("nan"))
            mean_b = self.mean_per_method.get(b, float("nan"))
            t_val = self.t_statistics.get((a, b), float("nan"))
            p_raw = self.raw_p_values.get((a, b), float("nan"))
            p_adj = self.adjusted_p_values.get((a, b), float("nan"))
            reject = self.rejections.get((a, b), False)
            warn_suffix = ""
            if n_a < 5:
                warn_suffix += "  [WARN n_a<5]"
            if n_b < 5:
                warn_suffix += "  [WARN n_b<5]"
            lines.append(
                f"| {a} | {b} | {n_a} | {n_b} | {mean_a:.2f} | {mean_b:.2f} | "
                f"{t_val:.2f} | {p_raw:.4f} | {p_adj:.4f} | {reject} |{warn_suffix}"
            )
        return "\n".join(lines)


# ===== compare_methods (spec 07 §2 / §3 / §5) ============================

def compare_methods(
    method_returns: Mapping[str, Sequence[float]],
    *,
    reference: str | None = None,
    alpha: float = 0.05,
) -> ComparisonResult:
    """Pairwise Welch t — each method vs reference (or all-pairs if ``reference is None``).

    Holm-Bonferroni is auto-applied iff the resulting number of pairwise tests
    is >= 2 (Lock 2: k >= 3 in reference mode, or k >= 3 in all-pairs).
    """
    import numpy as np

    if not method_returns:
        raise ValueError("compare_methods: method_returns is empty")
    if len(method_returns) < 2:
        raise ValueError("compare_methods: need at least 2 methods")

    # Compute per-method (n, mean, sem); validate n >= 2 (spec 07 §5.2).
    n_per: dict[str, int] = {}
    mean_per: dict[str, float] = {}
    sem_per: dict[str, float] = {}
    arrays: dict[str, "np.ndarray"] = {}
    for name, raw in method_returns.items():
        arr = np.asarray(list(raw), dtype=float)
        if arr.ndim != 1:
            raise ValueError(f"compare_methods: method {name!r}: must be 1-D")
        if np.isnan(arr).any():
            raise ValueError(
                f"compare_methods: method {name!r}: NaN in sample. "
                f"See pkg-08 spec 07 §3.2."
            )
        if len(arr) < 2:
            raise ValueError(
                f"compare_methods: method {name!r}: Welch t undefined for n < 2. "
                f"Got n={len(arr)}. See pkg-08 spec 07 §5.2."
            )
        arrays[name] = arr
        n_per[name] = int(len(arr))
        mean_per[name] = float(arr.mean())
        sem_per[name] = (
            float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else float("nan")
        )

    # Order methods: reference first if set, else by mean desc.
    ordered = sorted(method_returns.keys(), key=lambda k: -mean_per[k])
    if reference is not None:
        if reference not in method_returns:
            raise ValueError(
                f"compare_methods: reference={reference!r} not in method_returns "
                f"(keys: {sorted(method_returns)})"
            )
        ordered = [reference] + [m for m in ordered if m != reference]

    # Build comparison pairs.
    if reference is not None:
        pairs = tuple((reference, m) for m in ordered if m != reference)
    else:
        pairs = tuple(
            (a, b)
            for i, a in enumerate(ordered)
            for b in ordered[i + 1:]
        )

    # Per-pair Welch t.
    t_stats: dict[tuple[str, str], float] = {}
    raw_p: dict[tuple[str, str], float] = {}
    raw_p_list: list[float] = []
    for a, b in pairs:
        t_val, p_val = welch_t(arrays[a], arrays[b])
        t_stats[(a, b)] = t_val
        raw_p[(a, b)] = p_val
        raw_p_list.append(p_val)

    rejections_list, adjusted_list = holm_bonferroni(raw_p_list, alpha=alpha)
    correction: Literal["none", "holm-bonferroni"] = (
        "holm-bonferroni" if len(pairs) >= 2 else "none"
    )

    adjusted_p: dict[tuple[str, str], float] = {}
    rejections: dict[tuple[str, str], bool] = {}
    for (pair, adj, rej) in zip(pairs, adjusted_list, rejections_list):
        adjusted_p[pair] = float(adj)
        rejections[pair] = bool(rej)

    return ComparisonResult(
        methods=tuple(ordered),
        reference=reference,
        pairs=pairs,
        t_statistics=t_stats,
        raw_p_values=raw_p,
        adjusted_p_values=adjusted_p,
        rejections=rejections,
        n_per_method=n_per,
        mean_per_method=mean_per,
        sem_per_method=sem_per,
        correction=correction,
        alpha=alpha,
    )


# ===== Stats-layer joins (spec 07 §9) ====================================

def _load_report(path: str | pathlib.Path) -> dict[str, Any]:
    text = pathlib.Path(path).read_text(encoding="utf-8")
    return json.loads(text)


def compute_planner_prior_gap(
    registry_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], float]:
    """spec 07 §9.1 — read in-row gap from planner_full rows; NOT a cross-row join.

    Skips ``external_*`` variants (no planner) and rows whose
    ``eval_planner_mode != "planner_full"``.
    """
    gaps: dict[tuple[str, int], float] = {}
    for row in registry_rows:
        variant = row.get("variant", "")
        if variant.startswith("external_"):
            continue
        if row.get("eval_planner_mode") and row["eval_planner_mode"] != "planner_full":
            continue
        path = row.get("eval_report_path")
        if not path:
            continue
        try:
            report = _load_report(path)
        except (OSError, json.JSONDecodeError):
            continue
        # planner_full rows must populate eval_planner_mode = planner_full.
        if str(report.get("eval_planner_mode", "planner_full")) != "planner_full":
            continue
        gap = report.get("planner_prior_return_gap")
        if gap is None:
            continue
        gaps[(variant, int(row["seed"]))] = float(gap)
    return gaps


def compute_cross_mode_deltas(
    registry_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, str, str], float]:
    """spec 07 §9.2 — joins (variant, seed, config_hash) across eval_planner_mode literals.

    Returns ``{(variant, seed, mode_a, mode_b): mean_a - mean_b}``.
    """
    by_key: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = {}
    for row in registry_rows:
        variant = row.get("variant", "")
        if variant.startswith("external_"):
            continue
        path = row.get("eval_report_path")
        if not path:
            continue
        try:
            report = _load_report(path)
        except (OSError, json.JSONDecodeError):
            continue
        mode = str(report.get("eval_planner_mode") or row.get("eval_planner_mode") or "")
        if not mode:
            continue
        key = (variant, int(row["seed"]), str(row.get("config_hash", "")))
        by_key.setdefault(key, {})[mode] = report
    deltas: dict[tuple[str, int, str, str], float] = {}
    for (variant, seed, _h), modes in by_key.items():
        for mode_a, rep_a in modes.items():
            mean_a = rep_a.get("return_mean")
            if mean_a is None:
                continue
            for mode_b, rep_b in modes.items():
                if mode_a == mode_b:
                    continue
                mean_b = rep_b.get("return_mean")
                if mean_b is None:
                    continue
                deltas[(variant, seed, mode_a, mode_b)] = float(mean_a) - float(mean_b)
    return deltas


def compute_c_hidden_gap(
    registry_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], float]:
    """spec 07 §9.3 — paired ``c_visible=True/False`` join per (variant, seed)."""
    by_key: dict[tuple[str, int], dict[bool, float]] = {}
    for row in registry_rows:
        path = row.get("eval_report_path")
        if not path:
            continue
        try:
            report = _load_report(path)
        except (OSError, json.JSONDecodeError):
            continue
        if "c_visible" not in report or "return_mean" not in report:
            continue
        key = (row["variant"], int(row["seed"]))
        c_vis = bool(report["c_visible"])
        by_key.setdefault(key, {})[c_vis] = float(report["return_mean"])
    gaps: dict[tuple[str, int], float] = {}
    for key, by_vis in by_key.items():
        if True in by_vis and False in by_vis:
            gaps[key] = by_vis[True] - by_vis[False]
    return gaps


def compute_regret_per_method(
    registry_rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[float, float]]:
    """spec 07 §9.4 — per-variant (mean, sem) of ``EvalReport.regret_mean``.

    A variant with any cache-miss row (``oracle_ceiling_cache_hit`` containing
    ``False``) is omitted entirely (spec 02 Lock 3 conservative interpretation).
    """
    import numpy as np

    by_variant: dict[str, list[float]] = {}
    skip_variants: set[str] = set()
    for row in registry_rows:
        variant = row.get("variant", "")
        path = row.get("eval_report_path")
        if not path:
            continue
        try:
            report = _load_report(path)
        except (OSError, json.JSONDecodeError):
            continue
        cache_hit = report.get("oracle_ceiling_cache_hit")
        if isinstance(cache_hit, dict) and not all(bool(v) for v in cache_hit.values()):
            skip_variants.add(variant)
            continue
        if cache_hit is False:
            skip_variants.add(variant)
            continue
        regret = report.get("regret_mean")
        if regret is None or (isinstance(regret, float) and math.isnan(regret)):
            skip_variants.add(variant)
            continue
        by_variant.setdefault(variant, []).append(float(regret))
    out: dict[str, tuple[float, float]] = {}
    for variant, values in by_variant.items():
        if variant in skip_variants:
            continue
        arr = np.asarray(values, dtype=float)
        mean = float(arr.mean())
        sem = (
            float(arr.std(ddof=1) / math.sqrt(len(arr)))
            if len(arr) > 1 else float("nan")
        )
        out[variant] = (mean, sem)
    return out


def render_disclosure_table(
    registry_rows: Sequence[Mapping[str, Any]],
    preset: str | None = None,
) -> str:
    """spec 07 §6.3 + §9.5 — emit 10-column external-runner disclosure table.

    Schema (verbatim from pkg-07 spec 07 §4.2):
        variant | preset | param_count | walltime_to_converge_seconds |
        lr_swept_best | final_return_mean | final_return_sem |
        seeds_run | smoke_pass | sourced
    """
    import numpy as np

    completed = [
        r for r in registry_rows
        if r.get("status") == "completed"
        and str(r.get("variant", "")).startswith("external_")
    ]
    presets_to_emit: list[str] = (
        [preset] if preset is not None else ["rel_duo", "rel_duo_holdout"]
    )

    lines: list[str] = []
    for ps in presets_to_emit:
        rows_ps = [r for r in completed if _row_preset(r) == ps]
        lines.append(f"## Disclosure table — {ps} preset")
        lines.append("")
        lines.append("| " + " | ".join(DISCLOSURE_COLUMNS) + " |")
        lines.append("|" + "|".join(["---"] * len(DISCLOSURE_COLUMNS)) + "|")
        # Group by variant (filter to one block per preset).
        seen_variants: dict[str, list[dict[str, Any]]] = {}
        for r in rows_ps:
            seen_variants.setdefault(r["variant"], []).append(dict(r))
        for variant in sorted(seen_variants):
            agg = seen_variants[variant]
            returns = [
                float(r.get("return_mean")) for r in agg
                if r.get("return_mean") is not None
            ]
            seeds_run = len({int(r["seed"]) for r in agg})
            mean = float(np.mean(returns)) if returns else float("nan")
            sem = (
                float(np.std(returns, ddof=1) / math.sqrt(len(returns)))
                if len(returns) > 1 else float("nan")
            )
            walltime = max(
                (float(r.get("walltime_seconds") or 0.0) for r in agg),
                default=0.0,
            )
            param_count = (
                _first_field(agg, "param_count") or "-"
            )
            lr_best = _first_field(agg, "lr_swept_best") or "-"
            smoke_pass = _first_field(agg, "smoke_pass")
            sourced = _first_field(agg, "sourced", default=True)
            seeds_warn = "  [WARN seeds<5]" if seeds_run < 5 else ""
            lines.append(
                f"| {variant} | {ps} | {param_count} | "
                f"{walltime:.1f} | {lr_best} | "
                f"{mean:.2f} | {sem:.2f} | "
                f"{seeds_run}{seeds_warn} | "
                f"{('-' if smoke_pass is None else smoke_pass)} | "
                f"{('-' if sourced is None else sourced)} |"
            )
        lines.append("")
        lines.append(
            "_Footnote (pkg-07 spec 07 §4.5)_: 5-seed minimum is WARN-only "
            "(non-blocking)."
        )
        lines.append(
            "_Footnote (pkg-07 spec 07 §4.6)_: MAMBA not-sourced rows display "
            "'-' for all numeric cells."
        )
        lines.append(
            "_Footnote (pkg-07 spec 07 §4.7)_: external_marie / external_ga "
            "are excluded — Tier-2 stubs."
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _row_preset(row: Mapping[str, Any]) -> str | None:
    """Try multiple paths to recover the preset string from a registry row."""
    if "preset" in row:
        return str(row["preset"])
    snap = row.get("config_snapshot_path")
    if snap and pathlib.Path(snap).exists():
        try:
            cfg = json.loads(pathlib.Path(snap).read_text(encoding="utf-8"))
            return cfg.get("preset_name")
        except (OSError, json.JSONDecodeError):
            pass
    return None


def _first_field(
    rows: Iterable[Mapping[str, Any]], name: str, default: Any = None,
) -> Any:
    for r in rows:
        if name in r and r[name] is not None:
            return r[name]
    return default
