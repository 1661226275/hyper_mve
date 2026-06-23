"""Thesis table renderers (Table 6.1–6.7) → markdown.

Each renderer has the signature ``(suite_root, cells, out_dir) -> RenderOutcome``
where ``cells`` is the list of suite cells that declare this deliverable. They
reuse ``registry_io`` (per-variant metric samples) + ``stats.compare_methods``
(Welch-t + Holm-Bonferroni p-values vs Hyper) and degrade gracefully when a
cell hasn't run yet (``status="no_data"``) or is blocked (``status="blocked"``).
"""
from __future__ import annotations

import pathlib
from typing import Any

from . import registry_io
from .render_common import (
    RenderOutcome,
    WELFARE_METRICS,
    cell_rows,
    mean_sem,
    type_ratio_of_row,
)

REFERENCE = "hyper"


def _pvalues_vs_reference(rows: list[dict[str, Any]], metric: str, reference: str) -> dict[str, float]:
    """Holm-Bonferroni-adjusted Welch-t p-values of each variant vs ``reference``."""
    samples = {k: v for k, v in registry_io.method_samples(rows, metric).items() if len(v) >= 2}
    if reference not in samples or len(samples) < 2:
        return {}
    try:
        from hyper_mve.experiments import stats
        res = stats.compare_methods(samples, reference=reference)
    except Exception:  # numpy/scipy missing or degenerate
        return {}
    out: dict[str, float] = {}
    for (a, b), p in res.adjusted_p_values.items():
        other = b if a == reference else a
        out[other] = float(p)
    return out


def _fmt(mean: float, sem: float) -> str:
    if mean != mean:  # NaN
        return "—"
    if sem != sem:
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {sem:.3f}"


def render_methods_table(
    suite_root, cells, out_dir, *, deliverable: str,
    metrics=WELFARE_METRICS, reference=REFERENCE, title: str | None = None,
) -> RenderOutcome:
    """Per-variant mean ± sem across ``metrics`` + p(vs reference) on the headline."""
    rows: list[dict[str, Any]] = []
    for cell in cells:
        rows.extend(cell_rows(suite_root, cell))
    if not rows:
        return RenderOutcome(deliverable, "no_data", None, "no completed rows yet")

    prim = metrics[0][0]
    prim_samples = {k: v for k, v in registry_io.method_samples(rows, prim).items() if v}
    if not prim_samples:
        return RenderOutcome(deliverable, "no_data", None, f"metric {prim!r} absent")
    order = sorted(prim_samples, key=lambda k: -mean_sem(prim_samples[k])[0])
    pvals = _pvalues_vs_reference(rows, prim, reference)

    lines = [f"## {title or deliverable}", ""]
    header = "| variant | " + " | ".join(lbl for _, lbl in metrics) + f" | p(vs {reference}) |"
    lines.append(header)
    lines.append("|" + "|".join(["---"] * (len(metrics) + 2)) + "|")
    per_metric = {m: registry_io.method_samples(rows, m) for m, _ in metrics}
    for v in order:
        cells_md = []
        for m, _lbl in metrics:
            mean, sem = mean_sem(per_metric[m].get(v, []))
            cells_md.append(_fmt(mean, sem))
        p = pvals.get(v)
        p_md = "—" if (v == reference or p is None) else f"{p:.4f}"
        n = len(prim_samples.get(v, []))
        warn = "  ⚠n<5" if n < 5 else ""
        lines.append(f"| {v} ({n} seed{warn}) | " + " | ".join(cells_md) + f" | {p_md} |")
    lines.append("")
    lines.append("_Metrics: W_total=`return_mean` (subjective ΣR); W_phys/S/F/T from the §F eval extension. "
                 "p-values are Holm-Bonferroni-adjusted Welch-t vs Hyper on W_total._")
    md = "\n".join(lines)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{deliverable.replace(' ', '_').replace('.', '_')}.md"
    path.write_text(md + "\n", encoding="utf-8")
    return RenderOutcome(deliverable, "rendered", path, f"{len(order)} variants")


def render_table_6_1(suite_root, cells, out_dir) -> RenderOutcome:
    return render_methods_table(
        suite_root, cells, out_dir, deliverable="Table 6.1",
        title="Table 6.1 — 主对比 (Medium, 5 seeds): 5 metrics × 7 methods",
    )


def render_table_6_2(suite_root, cells, out_dir) -> RenderOutcome:
    return render_methods_table(
        suite_root, cells, out_dir, deliverable="Table 6.2",
        title="Table 6.2 — 消融1 条件化谱 (断言 B′)",
    )


def render_table_6_5(suite_root, cells, out_dir) -> RenderOutcome:
    return render_methods_table(
        suite_root, cells, out_dir, deliverable="Table 6.5",
        title="Table 6.5 — 消融4 CRN×CoordDesc 2×2 (断言 D)",
    )


def render_table_6_3(suite_root, cells, out_dir) -> RenderOutcome:
    """Abl2 context-paths — BLOCKED (only hyper + no_belief exist)."""
    rows: list[dict[str, Any]] = []
    for cell in cells:
        rows.extend(cell_rows(suite_root, cell))
    note = ("Table 6.3 (消融2 三联通路) is BLOCKED: no_type / no_cap / only_c variants "
            "are not implemented. Shipping the 2 existing variants (hyper, no_belief).")
    if rows:
        oc = render_methods_table(
            suite_root, cells, out_dir, deliverable="Table 6.3",
            title="Table 6.3 — 消融2 (PARTIAL: hyper + no_belief only)",
        )
        oc.status = "partial"
        oc.message = note
        return oc
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "Table_6_3.md"
    path.write_text(f"## Table 6.3 — 消融2 (BLOCKED)\n\n{note}\n", encoding="utf-8")
    return RenderOutcome("Table 6.3", "blocked", path, note)


def render_table_6_4(suite_root, cells, out_dir) -> RenderOutcome:
    """Abl3 bell curve as a table: return_mean per (n_alpha, n_beta) per variant."""
    rows: list[dict[str, Any]] = []
    for cell in cells:
        rows.extend(cell_rows(suite_root, cell))
    if not rows:
        return RenderOutcome("Table 6.4", "no_data", None, "no completed Abl3 rows yet")
    # group: (variant, (n_alpha, n_beta)) -> [return_mean]
    grouped: dict[str, dict[tuple[int, int], list[float]]] = {}
    for r in rows:
        tr = type_ratio_of_row(r)
        if tr is None:
            continue
        val = registry_io.metric_value(r, "return_mean")
        if val is None:
            continue
        grouped.setdefault(str(r.get("variant")), {}).setdefault(tr, []).append(val)
    if not grouped:
        return RenderOutcome("Table 6.4", "partial", None,
                             "rows present but type_assignment not recoverable from config snapshots")
    ratios = sorted({tr for d in grouped.values() for tr in d})
    lines = ["## Table 6.4 — 消融3 类型异质性钟形曲线 (W_total per ρ_β)", ""]
    lines.append("| variant | " + " | ".join(f"{a}α{b}β" for a, b in ratios) + " |")
    lines.append("|" + "|".join(["---"] * (len(ratios) + 1)) + "|")
    for v in sorted(grouped):
        cellvals = []
        for tr in ratios:
            xs = grouped[v].get(tr, [])
            mean, sem = mean_sem(xs)
            cellvals.append(_fmt(mean, sem))
        lines.append(f"| {v} | " + " | ".join(cellvals) + " |")
    lines.append("\n_See Fig 6.4 for the bell curve; peak advantage expected near 2α2β._")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "Table_6_4.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return RenderOutcome("Table 6.4", "rendered", path, f"{len(ratios)} ratios × {len(grouped)} variants")


def render_table_6_6(suite_root, cells, out_dir) -> RenderOutcome:
    """Abl6 Fehr-Schmidt 3×3 sensitivity (welfare metrics per variant — robustness)."""
    return render_methods_table(
        suite_root, cells, out_dir, deliverable="Table 6.6",
        title="Table 6.6 — 消融5 Fehr-Schmidt 稳健性 (aggregated over the 3×3 grid)",
    )


def render_table_6_7(suite_root, cells, out_dir) -> RenderOutcome:
    """Zero-shot retention: seen / unseen / gap per variant (Ch6.9)."""
    rows: list[dict[str, Any]] = []
    for cell in cells:
        rows.extend(cell_rows(suite_root, cell))
    if not rows:
        return RenderOutcome("Table 6.7", "no_data", None, "no completed zero-shot rows yet")
    seen = registry_io.method_samples(rows, "return_zero_shot_seen")
    unseen = registry_io.method_samples(rows, "return_zero_shot_unseen")
    gap = registry_io.method_samples(rows, "return_zero_shot_gap")
    variants = sorted(set(seen) | set(unseen),
                      key=lambda k: -(mean_sem(unseen.get(k, []))[0] if unseen.get(k) else 0))
    if not variants:
        return RenderOutcome("Table 6.7", "no_data", None, "zero-shot fields absent")
    lines = ["## Table 6.7 — 零样本泛化保留率 (seen / unseen / gap)", "",
             "| variant | seen c | unseen c | gap (seen−unseen) |", "|---|---|---|---|"]
    for v in variants:
        s = _fmt(*mean_sem(seen.get(v, [])))
        u = _fmt(*mean_sem(unseen.get(v, [])))
        g = _fmt(*mean_sem(gap.get(v, [])))
        lines.append(f"| {v} | {s} | {u} | {g} |")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "Table_6_7.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return RenderOutcome("Table 6.7", "rendered", path, f"{len(variants)} variants")


def render_gen_scope_sweep(suite_root, cells, out_dir) -> RenderOutcome:
    """Decision-Gate-0 gen_scope sweep: one row per cell (preset), welfare columns.

    The LoRA cells are all ``variant=hyper`` and differ only by preset (gen_scope
    mode), so a variant-grouped table collapses them. Here each cell is a row.
    """
    lines = ["## Table 6.2 — gen_scope 选型 (决策门 0): per-preset welfare", "",
             "| cell (gen_scope preset) | " + " | ".join(lbl for _, lbl in WELFARE_METRICS) + " | n |",
             "|" + "|".join(["---"] * (len(WELFARE_METRICS) + 2)) + "|"]
    any_rows = False
    for cell in cells:
        rows = cell_rows(suite_root, cell)
        if not rows:
            lines.append(f"| {cell.id} | " + " | ".join(["—"] * len(WELFARE_METRICS)) + " | 0 |")
            continue
        any_rows = True
        vals = []
        n = 0
        for m, _lbl in WELFARE_METRICS:
            xs = [x for s in registry_io.per_variant_metric(rows, m).values() for _, x in s]
            n = max(n, len(xs))
            vals.append(_fmt(*mean_sem(xs)))
        lines.append(f"| {cell.id} | " + " | ".join(vals) + f" | {n} |")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "Table_6_2_gen_scope_sweep.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return RenderOutcome("Table 6.2 (gen_scope sweep)",
                         "rendered" if any_rows else "no_data", path,
                         f"{len(cells)} gen_scope presets")


RENDERERS = {
    "Table 6.1": render_table_6_1,
    "Table 6.2": render_table_6_2,
    "Table 6.3": render_table_6_3,
    "Table 6.4": render_table_6_4,
    "Table 6.5": render_table_6_5,
    "Table 6.6": render_table_6_6,
    "Table 6.7": render_table_6_7,
}

# Exact-deliverable-string overrides (take precedence over the token map).
EXACT_RENDERERS = {
    "Table 6.2 (gen_scope sweep)": render_gen_scope_sweep,
}
