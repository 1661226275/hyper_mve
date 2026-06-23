"""Thesis figure renderers (Fig 6.1–6.7) → PNG (matplotlib Agg).

Same ``(suite_root, cells, out_dir) -> RenderOutcome`` signature as the table
renderers. matplotlib is imported lazily (Agg backend) so this module loads
without it; a renderer returns ``status="error"`` with a clear message if a
plot is attempted without matplotlib. Renderers degrade to ``no_data`` /
``blocked`` when a cell hasn't run or the feature isn't implemented.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

from . import registry_io, tb_scraper
from .render_common import RenderOutcome, cell_rows, mean_sem, type_ratio_of_row


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


def render_fig_6_2(suite_root, cells, out_dir) -> RenderOutcome:
    """Final-performance bar chart of W_total per variant (main comparison)."""
    rows = _all_rows(suite_root, cells)
    if not rows:
        return RenderOutcome("Fig 6.2", "no_data", None, "no completed rows yet")
    samples = {k: v for k, v in registry_io.method_samples(rows, "return_mean").items() if v}
    if not samples:
        return RenderOutcome("Fig 6.2", "no_data", None, "return_mean absent")
    order = sorted(samples, key=lambda k: -mean_sem(samples[k])[0])
    means = [mean_sem(samples[v])[0] for v in order]
    sems = [mean_sem(samples[v])[1] for v in order]
    sems = [0.0 if s != s else s for s in sems]
    try:
        plt = _plt()
    except Exception as e:
        return RenderOutcome("Fig 6.2", "error", None, f"matplotlib unavailable: {e}")
    fig, ax = plt.subplots(figsize=(max(4.0, 1.3 * len(order)), 4.5))
    ax.bar(range(len(order)), means, yerr=sems, capsize=4, color="tab:blue", edgecolor="black")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("W_total (return_mean)")
    ax.set_title("Fig 6.2 — 主对比最终性能")
    fig.tight_layout()
    path = _save(fig, out_dir, "Fig_6_2_final_performance.png")
    plt.close(fig)
    return RenderOutcome("Fig 6.2", "rendered", path, f"{len(order)} variants")


def render_fig_6_4(suite_root, cells, out_dir) -> RenderOutcome:
    """Bell curve: W_total vs β count, one line per variant (Hyper vs MA-MuZero)."""
    rows = _all_rows(suite_root, cells)
    if not rows:
        return RenderOutcome("Fig 6.4", "no_data", None, "no completed Abl3 rows yet")
    grouped: dict[str, dict[int, list[float]]] = {}
    for r in rows:
        tr = type_ratio_of_row(r)
        val = registry_io.metric_value(r, "return_mean")
        if tr is None or val is None:
            continue
        grouped.setdefault(str(r.get("variant")), {}).setdefault(tr[1], []).append(val)  # key = n_beta
    if not grouped:
        return RenderOutcome("Fig 6.4", "partial", None, "type_assignment not recoverable")
    try:
        plt = _plt()
    except Exception as e:
        return RenderOutcome("Fig 6.4", "error", None, f"matplotlib unavailable: {e}")
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for v in sorted(grouped):
        betas = sorted(grouped[v])
        ys = [mean_sem(grouped[v][b])[0] for b in betas]
        es = [mean_sem(grouped[v][b])[1] for b in betas]
        es = [0.0 if e != e else e for e in es]
        ax.errorbar(betas, ys, yerr=es, marker="o", capsize=4, label=v)
    ax.set_xlabel("β agents (类型异质度)")
    ax.set_ylabel("W_total (return_mean)")
    ax.set_title("Fig 6.4 — 类型异质性钟形曲线 (断言 A)")
    ax.legend()
    fig.tight_layout()
    path = _save(fig, out_dir, "Fig_6_4_bellcurve.png")
    plt.close(fig)
    return RenderOutcome("Fig 6.4", "rendered", path, f"{len(grouped)} variants")


def render_fig_6_5(suite_root, cells, out_dir) -> RenderOutcome:
    """Zero-shot retention: seen vs unseen grouped bars per variant."""
    rows = _all_rows(suite_root, cells)
    if not rows:
        return RenderOutcome("Fig 6.5", "no_data", None, "no completed zero-shot rows yet")
    seen = registry_io.method_samples(rows, "return_zero_shot_seen")
    unseen = registry_io.method_samples(rows, "return_zero_shot_unseen")
    variants = sorted(set(seen) | set(unseen))
    if not variants:
        return RenderOutcome("Fig 6.5", "no_data", None, "zero-shot fields absent")
    try:
        plt = _plt()
        import numpy as np
    except Exception as e:
        return RenderOutcome("Fig 6.5", "error", None, f"matplotlib/numpy unavailable: {e}")
    x = np.arange(len(variants))
    w = 0.38
    seen_m = [mean_sem(seen.get(v, []))[0] for v in variants]
    unseen_m = [mean_sem(unseen.get(v, []))[0] for v in variants]
    fig, ax = plt.subplots(figsize=(max(4.0, 1.4 * len(variants)), 4.5))
    ax.bar(x - w / 2, seen_m, w, label="seen c", color="tab:green")
    ax.bar(x + w / 2, unseen_m, w, label="unseen c", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=20, ha="right")
    ax.set_ylabel("W_total")
    ax.set_title("Fig 6.5 — 零样本泛化 seen vs unseen")
    ax.legend()
    fig.tight_layout()
    path = _save(fig, out_dir, "Fig_6_5_retention.png")
    plt.close(fig)
    return RenderOutcome("Fig 6.5", "rendered", path, f"{len(variants)} variants")


def render_fig_6_3(suite_root, cells, out_dir) -> RenderOutcome:
    """c-segment grouped bar from EvalReport.return_per_segment (per variant)."""
    rows = _all_rows(suite_root, cells)
    if not rows:
        return RenderOutcome("Fig 6.3", "no_data", None, "no completed rows yet")
    # variant -> segment(str) -> [value]
    seg: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        p = r.get("eval_report_path")
        if not p or not pathlib.Path(p).exists():
            continue
        try:
            rep = json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rps = rep.get("return_per_segment") or {}
        for k, val in rps.items():
            seg.setdefault(str(r.get("variant")), {}).setdefault(str(k), []).append(float(val))
    if not seg:
        return RenderOutcome("Fig 6.3", "no_data", None, "return_per_segment empty")
    try:
        plt = _plt()
        import numpy as np
    except Exception as e:
        return RenderOutcome("Fig 6.3", "error", None, f"matplotlib/numpy unavailable: {e}")
    segs = sorted({s for d in seg.values() for s in d})
    variants = sorted(seg)
    x = np.arange(len(segs))
    n = max(1, len(variants))
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(max(5.0, 1.5 * len(segs)), 4.5))
    for i, v in enumerate(variants):
        ys = [mean_sem(seg[v].get(s, []))[0] for s in segs]
        ax.bar(x + (i - (n - 1) / 2) * w, ys, w, label=v)
    ax.set_xticks(x)
    ax.set_xticklabels(segs, rotation=15, ha="right")
    ax.set_ylabel("W_total")
    ax.set_title("Fig 6.3 — c 分段社会福利")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = _save(fig, out_dir, "Fig_6_3_c_segments.png")
    plt.close(fig)
    return RenderOutcome("Fig 6.3", "rendered", path, f"{len(variants)} variants × {len(segs)} segments")


def render_fig_6_1(suite_root, cells, out_dir) -> RenderOutcome:
    """Training curves: scrape eval/planner/return_total across the cell's TB dirs."""
    import pathlib as _p
    series: dict[str, list[tuple[int, float]]] = {}
    for cell in cells:
        root = _p.Path(suite_root) / cell.runs_subdir
        if not root.exists():
            continue
        try:
            for tb in sorted(root.glob("**/tb")):
                run = str(tb.parent.relative_to(root)).replace("\\", "/")
                s = tb_scraper.scrape_tb(tb, ["eval/planner/return_total"]).get(
                    "eval/planner/return_total")
                if s:
                    series[f"{cell.id}/{run}"] = s
        except Exception as e:  # tensorboard missing, etc.
            return RenderOutcome("Fig 6.1", "error", None, f"TB scrape failed: {e}")
    if not series:
        return RenderOutcome("Fig 6.1", "no_data", None, "no TB eval/planner/return_total found")
    try:
        plt = _plt()
    except Exception as e:
        return RenderOutcome("Fig 6.1", "error", None, f"matplotlib unavailable: {e}")
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for name, s in series.items():
        xs = [p[0] for p in s]
        ys = [p[1] for p in s]
        ax.plot(xs, ys, label=name, linewidth=1.0)
    ax.set_xlabel("train step")
    ax.set_ylabel("eval planner W_total")
    ax.set_title("Fig 6.1 — 训练曲线")
    if len(series) <= 12:
        ax.legend(fontsize=7)
    fig.tight_layout()
    path = _save(fig, out_dir, "Fig_6_1_training_curves.png")
    plt.close(fig)
    return RenderOutcome("Fig 6.1", "rendered", path, f"{len(series)} curves")


def render_fig_6_6(suite_root, cells, out_dir) -> RenderOutcome:
    """Belief / curriculum: belief_c_mae per variant (best-effort)."""
    rows = _all_rows(suite_root, cells)
    if not rows:
        return RenderOutcome("Fig 6.6", "no_data", None, "no completed curriculum rows yet")
    mae = {k: v for k, v in registry_io.method_samples(rows, "belief_c_mae").items() if v}
    if not mae:
        return RenderOutcome("Fig 6.6", "partial", None,
                             "belief_c_mae only populated for hyper; needs per-phase TB for full Fig 6.6")
    try:
        plt = _plt()
    except Exception as e:
        return RenderOutcome("Fig 6.6", "error", None, f"matplotlib unavailable: {e}")
    variants = sorted(mae)
    means = [mean_sem(mae[v])[0] for v in variants]
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.bar(range(len(variants)), means, color="tab:purple", edgecolor="black")
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=20, ha="right")
    ax.set_ylabel("belief ĉ MAE")
    ax.set_title("Fig 6.6 — 信念推断质量 (best-effort)")
    fig.tight_layout()
    path = _save(fig, out_dir, "Fig_6_6_belief.png")
    plt.close(fig)
    return RenderOutcome("Fig 6.6", "partial", path, "ĉ MAE only; per-phase curves need TB")


def render_fig_6_7(suite_root, cells, out_dir) -> RenderOutcome:
    """∂R/∂u type-gradient visualisation — BLOCKED (needs an offline autograd probe)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "Fig_6_7_grad.md"
    note = ("Fig 6.7 (∂R/∂u type-gradient) is BLOCKED: needs an offline autograd probe "
            "(eval/grad_probe.py). Placeholder only.")
    path.write_text(f"## Fig 6.7 — BLOCKED\n\n{note}\n", encoding="utf-8")
    return RenderOutcome("Fig 6.7", "blocked", path, note)


RENDERERS = {
    "Fig 6.1": render_fig_6_1,
    "Fig 6.2": render_fig_6_2,
    "Fig 6.3": render_fig_6_3,
    "Fig 6.4": render_fig_6_4,
    "Fig 6.5": render_fig_6_5,
    "Fig 6.6": render_fig_6_6,
    "Fig 6.7": render_fig_6_7,
}
