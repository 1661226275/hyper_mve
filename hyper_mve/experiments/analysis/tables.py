"""Thesis table renderers (v5 Pkg-09: Table 6.1 rel gate / 6.2 regime holdout) → markdown.

Each renderer has the signature ``(suite_root, cells, out_dir) -> RenderOutcome``
where ``cells`` is the list of suite cells that declare this deliverable. They
reuse ``registry_io`` (per-variant metric samples) + ``stats.compare_methods``
(Welch-t + Holm-Bonferroni p-values vs Hyper) and degrade gracefully when a
cell hasn't run yet (``status="no_data"``).

The v4 renderers (Table 6.3–6.7, gen_scope sweep — c segments, type-ratio
bell curve, Fehr-Schmidt robustness) went with the resource_commons design.
"""
from __future__ import annotations

from typing import Any

from . import registry_io
from .render_common import (
    RenderOutcome,
    WELFARE_METRICS,
    cell_rows,
    mean_sem,
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


def _per_regime_samples(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[float]]]:
    """``{variant: {regime_id(str): [return, ...]}}`` from rel-v1 ``return_per_regime``."""
    out: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        rep = registry_io.report_dict(r)
        if rep is None:
            continue
        rpr = rep.get("return_per_regime") or {}
        for g, val in rpr.items():
            out.setdefault(str(r.get("variant")), {}).setdefault(str(g), []).append(float(val))
    return out


def render_methods_table(
    suite_root, cells, out_dir, *, deliverable: str,
    metrics=WELFARE_METRICS, reference=REFERENCE, title: str | None = None,
    per_regime: bool = True,
) -> RenderOutcome:
    """Per-variant mean ± sem across ``metrics`` + p(vs reference) on the headline,
    plus (rel-v1) a per-regime return appendix."""
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
    lines.append("_Metrics: W_total=`return_mean` (subjective ΣR); W_phys/S/F/T from the eval "
                 "extension. p-values are Holm-Bonferroni-adjusted Welch-t vs Hyper on W_total._")

    # rel-v1 per-regime appendix
    if per_regime:
        seg = _per_regime_samples(rows)
        if seg:
            regimes = sorted({g for d in seg.values() for g in d}, key=lambda s: int(s))
            lines += ["", "### Per-regime W_total (`return_per_regime`)", "",
                      "| variant | " + " | ".join(f"g={g}" for g in regimes) + " |",
                      "|" + "|".join(["---"] * (len(regimes) + 1)) + "|"]
            for v in order:
                if v not in seg:
                    continue
                vals = [_fmt(*mean_sem(seg[v].get(g, []))) for g in regimes]
                lines.append(f"| {v} | " + " | ".join(vals) + " |")

    md = "\n".join(lines)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{deliverable.replace(' ', '_').replace('.', '_')}.md"
    path.write_text(md + "\n", encoding="utf-8")
    return RenderOutcome(deliverable, "rendered", path, f"{len(order)} variants")


def render_table_6_1(suite_root, cells, out_dir) -> RenderOutcome:
    return render_methods_table(
        suite_root, cells, out_dir, deliverable="Table 6.1",
        title="Table 6.1 — v5 关系博弈主对照 (rel_gate_duo, 混合隐 regime)",
    )


def render_table_6_2(suite_root, cells, out_dir) -> RenderOutcome:
    """Zero-shot regime holdout: seen / unseen / gap per variant (rel_zero_shot_duo)."""
    rows: list[dict[str, Any]] = []
    for cell in cells:
        rows.extend(cell_rows(suite_root, cell))
    if not rows:
        return RenderOutcome("Table 6.2", "no_data", None, "no completed zero-shot rows yet")
    seen = registry_io.method_samples(rows, "return_zero_shot_seen")
    unseen = registry_io.method_samples(rows, "return_zero_shot_unseen")
    gap = registry_io.method_samples(rows, "return_zero_shot_gap")
    variants = sorted(set(seen) | set(unseen),
                      key=lambda k: -(mean_sem(unseen.get(k, []))[0] if unseen.get(k) else 0))
    if not variants:
        return RenderOutcome("Table 6.2", "no_data", None, "zero-shot fields absent")
    lines = ["## Table 6.2 — 零样本 regime 泛化 (train {coop,comp,neutral} → eval 非对称)", "",
             "| variant | seen regimes | unseen regimes | gap (seen−unseen) |", "|---|---|---|---|"]
    for v in variants:
        s = _fmt(*mean_sem(seen.get(v, [])))
        u = _fmt(*mean_sem(unseen.get(v, [])))
        g = _fmt(*mean_sem(gap.get(v, [])))
        lines.append(f"| {v} | {s} | {u} | {g} |")
    lines.append("")
    lines.append("_seen = regimes in `train_regime_ids` (0,1,4); unseen = held-out asymmetric "
                 "pair (2,3). Small gap ⇒ the relationship-linear value decomposition "
                 "recombines across W(g)._")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "Table_6_2.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return RenderOutcome("Table 6.2", "rendered", path, f"{len(variants)} variants")


RENDERERS = {
    "Table 6.1": render_table_6_1,
    "Table 6.2": render_table_6_2,
}

# Exact-deliverable-string overrides (take precedence over the token map).
EXACT_RENDERERS: dict[str, Any] = {}
