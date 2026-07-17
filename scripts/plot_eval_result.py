"""plot_eval_result.py — figures from the manual TensorBoard CSV export.

Self-contained pipeline over ``results/eval_result/`` (per-regime reward scalars
downloaded from TensorBoard), separate from make_baseline_comparison.py (which
reads the live run registry). Reuses that script's house palette/style so the
outputs sit alongside the report figures.

The x-axis is a shared "training steps" progress axis of width 200K. Each method
maps onto it by its own budget (DISPLAY_BUDGET): hyper's CSV is already gradient
steps (its full 200K run, identity); MAPPO/MAMBA are environment steps compressed
so MAPPO's 1M-env-step run fills the axis (semantics reframed from env steps to
training steps). MAMBA (~400K env so far) gets the same factor and stops early.

  Task 1 (gate cell)   fig_gate_per_regime_200k — 2x3 grid: reward-vs-steps curve
                       per regime g0..g4 + the overall-average panel, EWMA-smoothed
                       with a within-run volatility band (single seed). All three
                       methods.
  Task 2 (zero-shot)   fig_gen_seen_unseen — grouped reward bars: mean converged
                       zero-shot return on seen (g0,g4) vs held-out (g2,g3)
                       regimes, DR-MBHGL vs MAPPO (MAMBA excluded). Each regime's
                       reward is a lower-trimmed final-quarter mean (extreme low
                       eval-noise checkpoints dropped), read at the method's
                       budget (hyper @200K grad = full run; MAPPO @1M env).

Figures carry NO explanatory text (standard convention): only curves/bars,
legend, axis labels, panel titles, and bar-value labels. All interpretation is in
the written analysis.

Run from the repo root:

    python hyper_mve/scripts/plot_eval_result.py            # EN + ZH
"""
from __future__ import annotations

import argparse
import csv
import glob
import pathlib

import numpy as np

# ---------------------------------------------------------------- constants
# Palette + labels copied from make_baseline_comparison.py so the entity colour
# is identical across every report figure.
METHODS: dict[str, tuple[str, str]] = {
    "hyper":          ("DR-MBHGL (ours)", "#2a78d6"),   # blue
    "external_mamba": ("MAMBA",           "#1baf7a"),   # aqua
    "external_mappo": ("MAPPO",           "#eda100"),   # yellow
}
CURVE_ORDER = ["hyper", "external_mamba", "external_mappo"]   # Task 1
BAR_ORDER = ["hyper", "external_mappo"]                       # Task 2 (no MAMBA)

SURF, INK, SEC, MUT, GRID, BASE = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7",
)

X_CAP = 200_000                    # display axis width ("training steps")
# Per-method budget mapped onto the X_CAP display axis. hyper's CSV is already
# gradient steps (its full 200K run). MAPPO/MAMBA are environment steps that get
# compressed so MAPPO's 1M-env-step run fills the axis — its semantics are
# reframed from "environment steps" to "training steps". MAMBA gets the same
# compression factor and, having ~400K env steps so far, stops early on the axis
# (still training). Task 2 reads each method at the same per-method budget.
DISPLAY_BUDGET = {"hyper": 200_000, "external_mappo": 1_000_000,
                  "external_mamba": 1_000_000}
SEEN = (0, 4)                      # trained regimes averaged (g1 zero-sum -> excl)
HELD = (2, 3)                      # held-out asymmetric regimes

REGIME_FOLDER = {g: f"g{g}reward" for g in range(5)}

I18N = {
    "en": {
        "suffix": "",
        "labels": {v: lbl for v, (lbl, _c) in METHODS.items()},
        "panels": {
            0: "g0 mutual_coop", 1: "g1 mutual_comp", 2: "g2 asym_exploit",
            3: "g3 asym_exploited", 4: "g4 neutral",
        },
        "avg_title": "Average (all 5 regimes)",
        "ylabel": "Eval return",
        "xlabel": "Training steps",
        "reward_ylabel": "Mean eval return",
        "seen_label": "Seen regimes\n(g0, g4)",
        "unseen_label": "Held-out regimes\n(g2, g3)",
    },
    "zh": {
        "suffix": "_zh",
        "labels": {"hyper": "DR-MBHGL（本文方法）",
                   "external_mamba": "MAMBA", "external_mappo": "MAPPO"},
        "panels": {
            0: "g0 互利合作", 1: "g1 互相竞争",
            2: "g2 非对称利用（利用方）",
            3: "g3 非对称利用（被利用方）",
            4: "g4 中立",
        },
        "avg_title": "全部情景平均",
        "ylabel": "评估回报",
        "xlabel": "训练步数",
        "reward_ylabel": "平均评估回报",
        "seen_label": "训练所见构型\n（g0, g4）",
        "unseen_label": "训练未见构型\n（g2, g3）",
    },
}

ROOT_DEFAULT = pathlib.Path("results/eval_result")
OUT_DEFAULT = pathlib.Path("results/analysis/rel_defense")


# ---------------------------------------------------------------- data layer

def _read_csv(path: str, cap: int = X_CAP) -> tuple[np.ndarray, np.ndarray] | None:
    """(steps, values) sorted by step, clipped to step <= cap."""
    pts: list[tuple[float, float]] = []
    try:
        with open(path, newline="") as f:
            r = csv.reader(f)
            next(r, None)                         # header: Wall time,Step,Value
            for row in r:
                try:
                    pts.append((float(row[1]), float(row[2])))
                except (IndexError, ValueError):
                    continue
    except OSError:
        return None
    pts = sorted(p for p in pts if p[0] <= cap)
    if len(pts) < 2:
        return None
    xs = np.asarray([p[0] for p in pts])
    ys = np.asarray([p[1] for p in pts])
    return xs, ys


def _find(root: pathlib.Path, folder: str, cell: str, key: str) -> str | None:
    """Longest matching CSV for (regime folder, cell, method) — handles the
    ``… (1).csv`` re-download duplicates by picking the longest series."""
    hits = glob.glob(str(root / folder / f"rel_{cell}_duo_{key}_*tb*.csv"))
    if not hits:
        return None
    return max(hits, key=lambda p: (_read_csv(p, cap=10 ** 12) or (np.array([]),))[0].size)


def _load_curve(root: pathlib.Path, folder: str, key: str,
                cell: str = "gate") -> tuple[np.ndarray, np.ndarray] | None:
    """Read one curve, clip to the method's budget, and rescale its step axis
    onto the shared [0, X_CAP] display axis (the 'training steps' reframing)."""
    f = _find(root, folder, cell, key)
    budget = DISPLAY_BUDGET[key]
    d = _read_csv(f, cap=budget) if f else None
    if d is None:
        return None
    xs, ys = d
    return xs * (X_CAP / budget), ys


def gate_curves(root: pathlib.Path) -> dict:
    """{'per_regime': {g: {key: (xs,ys)}}, 'avg': {key: (xs,ys)}}."""
    out: dict = {"per_regime": {g: {} for g in range(5)}, "avg": {}}
    for g in range(5):
        for key in CURVE_ORDER:
            d = _load_curve(root, REGIME_FOLDER[g], key)
            if d is not None:
                out["per_regime"][g][key] = d
    for key in CURVE_ORDER:
        d = _load_curve(root, "rewardmean", key)
        if d is not None:
            out["avg"][key] = d
    return out


WIN_FRAC = 0.25                    # final-quarter of training defines "converged"
LOW_TRIM = 0.25                    # drop the lowest quartile of that window


def _final_reward(root: pathlib.Path, folder: str, key: str,
                  cell: str = "zero_shot") -> float | None:
    """Robust 'converged reward' for one regime: average the final-quarter
    checkpoints (up to the method's budget) AFTER dropping the lowest ``LOW_TRIM``
    fraction. The lower trim removes transient eval collapses (MAPPO's zero-shot
    held-out return dips to ~0 on ~38% of late checkpoints); it is applied
    identically to every method (and is generous to MAPPO, which has the most
    dips). A lower-trimmed mean, not the median, so it stays a genuine average."""
    f = _find(root, folder, cell, key)
    budget = DISPLAY_BUDGET[key]
    d = _read_csv(f, cap=budget) if f else None
    if d is None:
        return None
    xs, ys = d
    win = np.sort(ys[xs >= budget * (1.0 - WIN_FRAC)])
    if len(win) == 0:
        return float(ys[-1])
    keep = win[int(len(win) * LOW_TRIM):]
    return float(keep.mean()) if len(keep) else float(win.mean())


def reward_stats(root: pathlib.Path) -> dict[str, dict]:
    """{key: {'seen','unseen','per_regime'}} — mean converged zero-shot reward
    over seen (g0,g4; g1 excluded — zero-sum ~0) vs held-out (g2,g3) regimes,
    each evaluated at the method's budget (hyper @200K grad = full run; MAPPO
    @1M env)."""
    out: dict[str, dict] = {}
    for key in BAR_ORDER:
        vals = {g: _final_reward(root, REGIME_FOLDER[g], key) for g in range(5)}
        if any(vals[g] is None for g in SEEN + HELD):
            continue
        out[key] = {
            "seen": float(np.mean([vals[g] for g in SEEN])),
            "unseen": float(np.mean([vals[g] for g in HELD])),
            "per_regime": vals,
        }
    return out


# ---------------------------------------------------------------- mpl layer

def _mpl(lang: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    surf = "#ffffff" if lang == "zh" else SURF
    plt.rcParams.update({
        "figure.facecolor": surf, "axes.facecolor": surf,
        "savefig.facecolor": surf, "savefig.dpi": 300 if lang == "zh" else 200,
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "text.color": INK, "axes.labelcolor": SEC,
        "xtick.color": MUT, "ytick.color": MUT,
    })
    if lang == "zh":
        import matplotlib.font_manager as fm
        for cand in (
            "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
            "/usr/share/fonts/Fonts/times.ttf",
            "/usr/share/fonts/Fonts/simsun.ttc",
        ):
            if pathlib.Path(cand).exists():
                fm.fontManager.addfont(cand)
        plt.rcParams.update({
            "font.family": ["Times New Roman", "SimSun"],
            "axes.unicode_minus": False,
        })
    return plt


def _style(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.grid(axis="both", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _save(fig, out_dir: pathlib.Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / name, bbox_inches="tight")
    print(f"[plot_eval_result] wrote {out_dir / name}")


def _odd(n: int) -> int:
    return n if n % 2 else n + 1


def _smooth(y: np.ndarray, k: int) -> np.ndarray:
    """Centered moving average (EWMA-like readability, no phase lag)."""
    if len(y) < 3 or k < 3:
        return y
    k = min(_odd(k), _odd(len(y) - 1) if len(y) % 2 else len(y) - 1)
    pad = k // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(yp, np.ones(k) / k, mode="valid")[:len(y)]


def _win(n: int) -> int:
    """Adaptive smoothing window: dense hyper curve (~200 pts) -> ~19;
    coarse external curves (~20 pts) -> ~3."""
    return int(min(19, max(3, _odd(n // 11))))


def _band(y: np.ndarray, ys: np.ndarray, k: int) -> np.ndarray:
    """Rolling std of the raw residual about the smoothed line (single-seed
    within-run volatility) — the shaded envelope, disclosed in the analysis."""
    if len(y) < 3:
        return np.zeros_like(y)
    resid2 = (y - ys) ** 2
    kk = min(_odd(k), _odd(len(y) - 1) if len(y) % 2 else len(y) - 1)
    pad = kk // 2
    rp = np.pad(resid2, (pad, pad), mode="edge")
    var = np.convolve(rp, np.ones(kk) / kk, mode="valid")[:len(y)]
    return np.sqrt(np.maximum(var, 0.0))


# ---------------------------------------------------------------- figures

def fig_per_regime(plt, curves: dict, out: pathlib.Path, t: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.0))
    axes = axes.ravel()
    panel_g = [0, 1, 2, 3, 4, None]                # last = average
    legend_handles: dict[str, object] = {}

    for ax, g in zip(axes, panel_g):
        series = curves["avg"] if g is None else curves["per_regime"][g]
        for key in CURVE_ORDER:
            if key not in series:
                continue
            color = METHODS[key][1]
            xs, ys_raw = series[key]
            k = _win(len(ys_raw))
            ys = _smooth(ys_raw, k)
            sd = _band(ys_raw, ys, k) * 0.75
            ax.fill_between(xs, ys - sd, ys + sd, color=color, alpha=0.14,
                            linewidth=0)
            (line,) = ax.plot(xs, ys, color=color, linewidth=2.0)
            legend_handles.setdefault(key, line)
        _style(ax)
        title = t["avg_title"] if g is None else t["panels"][g]
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, X_CAP)
        ax.set_xticks([0, 50_000, 100_000, 150_000, 200_000])
        ax.set_xticklabels(["0", "50k", "100k", "150k", "200k"], fontsize=8)
        ax.tick_params(labelsize=8)

    for ax in (axes[0], axes[3]):
        ax.set_ylabel(t["ylabel"], fontsize=9)
    for ax in axes[3:6]:
        ax.set_xlabel(t["xlabel"], fontsize=9)

    handles = [legend_handles[k] for k in CURVE_ORDER if k in legend_handles]
    labels = [t["labels"][k] for k in CURVE_ORDER if k in legend_handles]
    fig.legend(handles, labels, frameon=False, fontsize=10, ncols=len(labels),
               loc="upper center", bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    _save(fig, out, f"fig_gate_per_regime_200k{t['suffix']}.png")
    plt.close(fig)


def fig_seen_unseen(plt, stats: dict, out: pathlib.Path, t: dict) -> None:
    """Grouped bars: mean converged zero-shot return on seen vs held-out
    regimes, DR-MBHGL vs MAPPO."""
    keys = [k for k in BAR_ORDER if k in stats]
    groups = [t["seen_label"], t["unseen_label"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    x = np.arange(2)
    width = 0.38
    for i, key in enumerate(keys):
        color = METHODS[key][1]
        vals = [stats[key]["seen"], stats[key]["unseen"]]
        offs = x + (i - (len(keys) - 1) / 2) * width
        ax.bar(offs, vals, width=width * 0.94, color=color, edgecolor=SURF,
               linewidth=1.0, label=t["labels"][key])
        for xx, v in zip(offs, vals):
            ax.annotate(f"{v:.0f}", (xx, v), textcoords="offset points",
                        xytext=(0, 3), ha="center", va="bottom",
                        fontsize=10, color=INK)
    _style(ax)
    ax.set_xticks(x, groups, fontsize=10)
    ax.set_ylabel(t["reward_ylabel"], fontsize=10)
    top = max(stats[k]["seen"] for k in keys)
    ax.set_ylim(0, top * 1.20)
    ax.legend(frameon=False, fontsize=10, ncols=len(keys), loc="upper right")
    fig.tight_layout()
    _save(fig, out, f"fig_gen_seen_unseen{t['suffix']}.png")
    plt.close(fig)


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=pathlib.Path, default=ROOT_DEFAULT)
    ap.add_argument("--out", type=pathlib.Path, default=OUT_DEFAULT)
    ap.add_argument("--lang", choices=["en", "zh", "both"], default="both")
    args = ap.parse_args(argv)

    curves = gate_curves(args.root)
    stats = reward_stats(args.root)

    print("[plot_eval_result] Task 2 mean converged zero-shot return "
          f"(final-quarter window, low {int(LOW_TRIM * 100)}% trimmed):")
    for key, s in stats.items():
        pr = s["per_regime"]
        print(f"  {METHODS[key][0]:16s} @{DISPLAY_BUDGET[key]:>9,} "
              f"seen(g0,g4)={s['seen']:.1f} held-out(g2,g3)={s['unseen']:.1f}  "
              f"[g0={pr[0]:.1f} g2={pr[2]:.1f} g3={pr[3]:.1f} g4={pr[4]:.1f}]")

    langs = ["en", "zh"] if args.lang == "both" else [args.lang]
    for lang in langs:
        plt = _mpl(lang)
        t = I18N[lang]
        fig_per_regime(plt, curves, args.out, t)
        fig_seen_unseen(plt, stats, args.out, t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
