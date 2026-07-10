"""Thesis figure renderers (v5 Pkg-09: Fig 6.1 per-regime returns / 6.2 seen-vs-unseen) → PNG.

Same ``(suite_root, cells, out_dir) -> RenderOutcome`` signature as the table
renderers. matplotlib is imported lazily (Agg backend) so this module loads
without it; a renderer returns ``status="error"`` with a clear message if a
plot is attempted without matplotlib. Renderers degrade to ``no_data`` when a
cell hasn't run yet.

The v4 renderers (training curves, c-segment bars, type-heterogeneity bell
curve, belief-ĉ MAE, ∂R/∂u placeholder) went with the resource_commons design;
TB training curves remain reachable via ``analyze_results.py --tb``.
"""
from __future__ import annotations

import pathlib
from typing import Any

from . import registry_io
from .render_common import RenderOutcome, cell_rows, mean_sem


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _all_rows(suite_root, cells) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in cells:
        rows.extend(cell_rows(suite_root, cell))
    return rows


def _save(fig, out_dir: pathlib.Path, name: str) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path


def render_fig_6_1(suite_root, cells, out_dir) -> RenderOutcome:
    """Per-regime W_total grouped bars (rel-v1 ``return_per_regime``), one group
    per regime id, one bar per variant — the headline mixed-motive figure."""
    rows = _all_rows(suite_root, cells)
    if not rows:
        return RenderOutcome("Fig 6.1", "no_data", None, "no completed rows yet")
    # variant -> regime(str) -> [value]
    seg: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        rep = registry_io.report_dict(r)
        if rep is None:
            continue
        rpr = rep.get("return_per_regime") or {}
        for g, val in rpr.items():
            seg.setdefault(str(r.get("variant")), {}).setdefault(str(g), []).append(float(val))
    if not seg:
        return RenderOutcome("Fig 6.1", "no_data", None, "return_per_regime empty")
    try:
        plt = _plt()
        import numpy as np
    except Exception as e:
        return RenderOutcome("Fig 6.1", "error", None, f"matplotlib/numpy unavailable: {e}")
    regimes = sorted({g for d in seg.values() for g in d}, key=lambda s: int(s))
    variants = sorted(seg)
    x = np.arange(len(regimes))
    n = max(1, len(variants))
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(max(5.0, 1.5 * len(regimes)), 4.5))
    for i, v in enumerate(variants):
        ys = [mean_sem(seg[v].get(g, []))[0] for g in regimes]
        es = [mean_sem(seg[v].get(g, []))[1] for g in regimes]
        es = [0.0 if e != e else e for e in es]
        ax.bar(x + (i - (n - 1) / 2) * w, ys, w, yerr=es, capsize=3, label=v)
    ax.set_xticks(x)
    ax.set_xticklabels([f"g={g}" for g in regimes])
    ax.set_xlabel("regime id (g2: 0 coop / 1 comp / 2-3 asym / 4 neutral)")
    ax.set_ylabel("W_total (return_per_regime)")
    ax.set_title("Fig 6.1 — 逐 regime 社会总福利")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = _save(fig, out_dir, "Fig_6_1_per_regime_returns.png")
    plt.close(fig)
    return RenderOutcome("Fig 6.1", "rendered", path,
                         f"{len(variants)} variants × {len(regimes)} regimes")


def render_fig_6_2(suite_root, cells, out_dir) -> RenderOutcome:
    """Zero-shot regime holdout: seen vs unseen grouped bars per variant."""
    rows = _all_rows(suite_root, cells)
    if not rows:
        return RenderOutcome("Fig 6.2", "no_data", None, "no completed zero-shot rows yet")
    seen = registry_io.method_samples(rows, "return_zero_shot_seen")
    unseen = registry_io.method_samples(rows, "return_zero_shot_unseen")
    variants = sorted(set(seen) | set(unseen))
    if not variants:
        return RenderOutcome("Fig 6.2", "no_data", None, "zero-shot fields absent")
    try:
        plt = _plt()
        import numpy as np
    except Exception as e:
        return RenderOutcome("Fig 6.2", "error", None, f"matplotlib/numpy unavailable: {e}")
    x = np.arange(len(variants))
    w = 0.38
    seen_m = [mean_sem(seen.get(v, []))[0] for v in variants]
    unseen_m = [mean_sem(unseen.get(v, []))[0] for v in variants]
    fig, ax = plt.subplots(figsize=(max(4.0, 1.4 * len(variants)), 4.5))
    ax.bar(x - w / 2, seen_m, w, label="seen regimes (0,1,4)", color="tab:green")
    ax.bar(x + w / 2, unseen_m, w, label="unseen regimes (2,3)", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=20, ha="right")
    ax.set_ylabel("W_total")
    ax.set_title("Fig 6.2 — 零样本 regime 泛化 seen vs unseen")
    ax.legend()
    fig.tight_layout()
    path = _save(fig, out_dir, "Fig_6_2_regime_holdout.png")
    plt.close(fig)
    return RenderOutcome("Fig 6.2", "rendered", path, f"{len(variants)} variants")


RENDERERS = {
    "Fig 6.1": render_fig_6_1,
    "Fig 6.2": render_fig_6_2,
}
