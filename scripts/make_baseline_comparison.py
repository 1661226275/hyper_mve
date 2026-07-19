"""make_baseline_comparison.py — comparison artifacts for the v5 formal grid.

Reads the FLAT ``results/registry.jsonl`` written by ``scripts/train.py`` and
assembles the three headline metrics of the phase-2 realignment, plus the
ablation-arm comparison:

  Fig 1  final return per method, grouped by env             [metric ② reward]
  Fig 2  per-regime return bars (relation)                   [metric ② reward]
  Fig 3  seen vs held-out generalization (relation_holdout)  [metric ③ general.]
  Fig 4  world-model reward fidelity, lower better           [metric ① fidelity]
  Fig 5  ablation arms vs the unablated method
  Fig 6  sample-efficiency curves — ONLY for runners carrying a periodic eval
         probe (mappo / mamba). The method, happo, mbom and m3w_adapted log a
         single final eval by design, so they appear as endpoint markers. The
         figure says so rather than implying the others lack data.
  Table 1 markdown summary (all three metrics, per env, mean ± sem over seeds).

Run from the repo root:

    python scripts/make_baseline_comparison.py \\
        --registry results/registry.jsonl \\
        --out results/analysis/v5_formal --assets docs/assets/interim_report

Partial-data safe: every artifact is built from whatever rows are completed, so
this is useful while the grid is still running; missing pieces are reported,
not fatal.

Superseded design (pre-2026-07-18): this script used to read per-cell
registries under ``runs/suite/<cell>/``, filter externals to a 2M env-step
budget, and project in-flight runs via saturating fits. The formal grid is
single-budget (200k) with every row run to completion, so the budget filter,
the projection machinery and the MAMBA checkpoint-series overlay are gone —
along with the pre-refactor variant vocabulary (``hyper`` / ``external_*``),
which matched none of the seven registry keys.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import numpy as np

# ---------------------------------------------------------------- constants

# Categorical slots 1-7 of the validated reference palette, IN ITS ORDER.
# The order is load-bearing: it is the adjacency the palette was validated on
# (`node scripts/validate_palette.js "<hexes>" --mode light` -> ALL CHECKS PASS,
# worst adjacent CVD ΔE 9.1 protan, worst normal-vision ΔE 19.6). Reordering
# these rows silently changes which pairs sit next to each other — re-run the
# validator if you do. An earlier draft ordered them ours/mamba/mappo/happo/...
# and FAILED at orange↔green ΔE 3.2 (protan red-green confusion).
#
# Colour follows the ENTITY: mazero_mixed/mappo/mamba keep the exact hues they
# had under the old names (hyper/external_mappo/external_mamba) so figures
# already in the report stay consistent.
METHODS: dict[str, tuple[str, str]] = {
    "mazero_mixed": ("DR-MBHGL (ours)", "#2a78d6"),   # slot 1 blue
    "happo":        ("HAPPO",           "#008300"),   # slot 2 green
    "mbom":         ("MBOM",            "#e87ba4"),   # slot 3 magenta
    "mappo":        ("MAPPO",           "#eda100"),   # slot 4 yellow
    "mamba":        ("MAMBA",           "#1baf7a"),   # slot 5 aqua
    "mbom_oracle":  ("MBOM-oracle",     "#eb6834"),   # slot 6 orange
    "m3w_adapted":  ("M3W-adapted",     "#4a3aa7"),   # slot 7 violet
}

# Ablation arms. The unablated reference keeps its method hue (entity rule);
# the arms take the next slots of the same validated order. Arms and baselines
# never share a figure, so the two maps are independent.
ABLATION_REF = "mazero_mixed"
ARMS: dict[str, tuple[str, str]] = {
    "mazero_mixed":                     ("unablated (full)",   "#2a78d6"),
    "mazero_mixed_point_estimate_leaf": ("point-estimate leaf", "#008300"),
    "mazero_mixed_joint_selection":     ("joint selection",     "#e87ba4"),
    "mazero_mixed_no_subjective":       ("no subjective",       "#eda100"),
    "mazero_mixed_moe_router":          ("MoE router",          "#1baf7a"),
    "mazero_mixed_film":                ("FiLM",                "#eb6834"),
}

# Runners with no world model, or a supplied one, have no fidelity hook:
# runner.predict_rewards() returns None and no fidelity.json is written.
# Listing them keeps "N/A by design" distinct from "data missing".
NO_FIDELITY = ("mappo", "happo", "mbom", "mbom_oracle")

ENVS: tuple[str, ...] = ("relation", "relation_holdout", "mpe_tag")

REGIME_LABELS = {
    0: "g0\nmutual_coop", 1: "g1\nmutual_comp", 2: "g2\nasym_exploit",
    3: "g3\nasym_exploited", 4: "g4\nneutral",
}

SURF, INK, SEC, MUT, GRID, BASE = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7",
)

I18N: dict[str, dict] = {
    "en": {
        "suffix": "",
        "surf": SURF,
        "method_labels": {v: lab for v, (lab, _c) in METHODS.items()},
        "arm_labels": {v: lab for v, (lab, _c) in ARMS.items()},
        "env_labels": {"relation": "relation\n(rel_duo)",
                       "relation_holdout": "relation_holdout\n(held-out split)",
                       "mpe_tag": "mpe_tag\n(simple_tag, N=4)"},
        "regime_labels": REGIME_LABELS,
        "n_fmt": "{label} (n={n})",
        "fig1_title": "Final eval return by environment",
        "fig1_ylabel": "Mean eval return (all regimes)",
        "fig2_title": "Global reward per hidden regime (relation)",
        "fig2_ylabel": "Eval return",
        "fig3_title": "Generalization to held-out regimes (relation_holdout)",
        "fig3_groups": ["Training regimes\n(seen)", "Held-out regimes\n(unseen)"],
        "fig3_ylabel": "Mean eval return",
        "fig4_title": "World-model reward fidelity (relation)",
        "fig4_ylabel": "Reward MAE (↓ better)",
        "fig4_na": "no world-model hook (N/A by design)",
        "fig5_title": "Ablation arms vs the unablated method (relation)",
        "fig5_ylabel": "Mean eval return",
        "fig6_title": "Sample efficiency — runners with a periodic eval probe",
        "fig6_xlabel": "Environment steps",
        "fig6_ylabel": "Mean eval return (all regimes)",
        "fig6_note": ("Lines: runners carrying a periodic eval probe. Markers: "
                      "runners that log a single final eval by design — an "
                      "endpoint, not a curve."),
        "seeds_note": "Error bars: ±1 s.e.m. over seeds.",
    },
    "zh": {
        "suffix": "_zh",
        "surf": "#ffffff",
        "method_labels": {
            "mazero_mixed": "DR-MBHGL（本文方法）", "happo": "HAPPO",
            "mbom": "MBOM", "mappo": "MAPPO", "mamba": "MAMBA",
            "mbom_oracle": "MBOM-oracle", "m3w_adapted": "M3W-adapted",
        },
        "arm_labels": {
            "mazero_mixed": "完整方法（未消融）",
            "mazero_mixed_point_estimate_leaf": "点估计叶节点",
            "mazero_mixed_joint_selection": "联合动作选择",
            "mazero_mixed_no_subjective": "去主观通路",
            "mazero_mixed_moe_router": "MoE 路由",
            "mazero_mixed_film": "FiLM 调制",
        },
        "env_labels": {"relation": "relation\n（rel_duo）",
                       "relation_holdout": "relation_holdout\n（留出划分）",
                       "mpe_tag": "mpe_tag\n（simple_tag, N=4）"},
        "regime_labels": {
            0: "g0\n互利合作", 1: "g1\n互相竞争",
            2: "g2\n非对称利用\n（利用方）", 3: "g3\n非对称利用\n（被利用方）",
            4: "g4\n中立",
        },
        "n_fmt": "{label}（n={n}）",
        "fig1_title": None,
        "fig1_ylabel": "全情景平均评估回报",
        "fig2_title": None,
        "fig2_ylabel": "评估回报",
        "fig3_title": None,
        "fig3_groups": ["训练所见构型", "训练未见构型"],
        "fig3_ylabel": "平均评估回报",
        "fig4_title": None,
        "fig4_ylabel": "奖励预测平均绝对误差（越低越好）",
        "fig4_na": "无世界模型接口（设计上不适用）",
        "fig5_title": None,
        "fig5_ylabel": "平均评估回报",
        "fig6_title": None,
        "fig6_xlabel": "环境交互步数",
        "fig6_ylabel": "全情景平均评估回报",
        "fig6_note": "实线：具备周期性评估探针的算法；标记点：仅记录最终评估的算法。",
        "seeds_note": "误差棒为跨随机种子的 ±1 标准误。",
    },
}


# ---------------------------------------------------------------- data layer

def _latest_rows(registry_path: pathlib.Path) -> list[dict]:
    """Latest row per ``run_id`` (train.py appends one row per finished run)."""
    if not registry_path.exists():
        return []
    latest: dict[str, dict] = {}
    for line in registry_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = row.get("run_id") or (
            f"{row.get('variant')}/{row.get('env')}/seed{row.get('seed')}")
        prev = latest.get(key)
        if prev is None or str(row.get("started_at_iso8601", "")) >= str(
                prev.get("started_at_iso8601", "")):
            latest[key] = row
    return list(latest.values())


def _load_json(path: str | None) -> dict | None:
    if not path:
        return None
    p = pathlib.Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _curve(row: dict) -> list[tuple[float, float]] | None:
    """(env_steps, eval return) for one row, or None if it logs no probe.

    UnifiedLogger writes every scalar against the canonical train-step x AND
    co-logs ``progress/env_steps`` at that same x, so the env-step axis is
    recovered by joining the two tags on the step key — no per-runner
    conversion factor, which is exactly what that co-logging is for.
    """
    from hyper_mve.utils.analysis.tb_scraper import scrape_tb

    tb = row.get("tensorboard_dir")
    if not tb or not pathlib.Path(tb).exists():
        return None
    try:
        series = scrape_tb(tb, ["eval/return_mean", "progress/env_steps"])
    except ImportError:
        return None
    ret = series.get("eval/return_mean") or []
    if len(ret) < 3:              # a lone final eval is an endpoint, not a curve
        return None
    env_at = {int(s): v for s, v in (series.get("progress/env_steps") or [])}
    out = [(float(env_at.get(int(s), s)), float(v)) for s, v in ret]
    out.sort(key=lambda p: p[0])
    return out or None


def _fidelity(row: dict) -> float | None:
    """Reward MAE for a row — registry field first, fidelity.json as fallback."""
    val = row.get("fidelity_reward_mae")
    if val is not None:
        return float(val)
    rd = row.get("run_dir")
    fid = _load_json(str(pathlib.Path(rd) / "fidelity.json")) if rd else None
    if fid and fid.get("reward_mae") is not None:
        return float(fid["reward_mae"])
    return None


def collect(registry_path: pathlib.Path) -> dict:
    """``{env: {variant: {rows, reports, curves, fidelity}}}`` for completed runs."""
    known = set(METHODS) | set(ARMS)
    out: dict = {env: {} for env in ENVS}
    for row in _latest_rows(registry_path):
        variant = row.get("variant")
        env = row.get("env")
        if variant not in known or env not in out:
            continue
        if row.get("status") != "completed":
            continue
        ent = out[env].setdefault(
            variant, {"rows": [], "reports": {}, "curves": {}, "fidelity": {}})
        seed = int(row.get("seed", 0))
        ent["rows"].append(row)
        rep = _load_json(row.get("eval_report_path"))
        if rep:
            ent["reports"][seed] = rep
        fid = _fidelity(row)
        if fid is not None:
            ent["fidelity"][seed] = fid
        cur = _curve(row)
        if cur:
            ent["curves"][seed] = cur
    return out


def _agg(values: list[float]) -> tuple[float, float]:
    """(mean, s.e.m.) — s.e.m. is 0 for a single seed."""
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1) / math.sqrt(arr.size))


def _metric(ent: dict, key: str) -> tuple[float, float, int]:
    """(mean, sem, n_seeds) of a scalar EvalReport field across seeds."""
    vals = [float(r[key]) for r in ent["reports"].values() if r.get(key) is not None]
    m, s = _agg(vals)
    return m, s, len(vals)


def _regime_metric(ent: dict, g: int) -> tuple[float, float, int]:
    vals = []
    for rep in ent["reports"].values():
        per = rep.get("return_per_regime") or {}
        v = per.get(str(g), per.get(g))
        if v is not None:
            vals.append(float(v))
    m, s = _agg(vals)
    return m, s, len(vals)


# ---------------------------------------------------------------- mpl layer

def _mpl(lang: str = "en"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    surf = "#ffffff" if lang == "zh" else SURF
    plt.rcParams.update({
        "figure.facecolor": surf, "axes.facecolor": surf,
        "savefig.facecolor": surf,
        "savefig.dpi": 300 if lang == "zh" else 200,
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "text.color": INK, "axes.labelcolor": SEC,
        "xtick.color": MUT, "ytick.color": MUT,
    })
    if lang == "zh":
        import matplotlib.font_manager as fm
        # System fontconfig resolves these aliases fine, but matplotlib's own
        # font cache often misses the underlying files — register explicitly.
        for candidate in (
            "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
            "/usr/share/fonts/Fonts/times.ttf",
            "/usr/share/fonts/Fonts/simsun.ttc",
        ):
            if pathlib.Path(candidate).exists():
                fm.fontManager.addfont(candidate)
        plt.rcParams.update({
            # A concrete font list (not the "serif" generic alias) so Agg's
            # per-glyph fallback actually walks Times New Roman -> SimSun.
            "font.family": ["Times New Roman", "SimSun"],
            "axes.unicode_minus": False,
        })
    return plt


def _style(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _save(fig, out_dir: pathlib.Path, assets: pathlib.Path | None, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / name, bbox_inches="tight")
    if assets is not None:
        assets.mkdir(parents=True, exist_ok=True)
        fig.savefig(assets / name, bbox_inches="tight")
    print(f"[make_baseline_comparison] wrote {out_dir / name}")


def _bar_labels(ax, bars, values, fmt="{:.1f}") -> None:
    """Direct value labels.

    Not decoration: three palette slots sit below 3:1 contrast on the light
    surface, so the validated palette ships under the relief rule — visible
    labels or a table view. Both are provided (see write_table).
    """
    span = max((abs(v) for v in values), default=1.0) or 1.0
    for bar, val in zip(bars, values):
        off = 0.02 * span
        ax.annotate(fmt.format(val),
                    (bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + (off if val >= 0 else -off)),
                    ha="center", va="bottom" if val >= 0 else "top",
                    fontsize=7, color=SEC)


def _grouped_bars(ax, groups, series, values, errs, colors, labels, t):
    """Grouped bars with a 2px surface gap between adjacent fills."""
    n = max(len(series), 1)
    width = 0.8 / n
    x = np.arange(len(groups), dtype=float)
    for i, v in enumerate(series):
        off = (i - (n - 1) / 2) * width
        vals = [values[g][i] for g in range(len(groups))]
        es = [errs[g][i] for g in range(len(groups))]
        bars = ax.bar(x + off, vals, width * 0.92, yerr=es, capsize=2,
                      color=colors[v], label=labels[v], zorder=3,
                      error_kw={"elinewidth": 0.9, "ecolor": SEC})
        if len(series) <= 4:
            _bar_labels(ax, bars, vals)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    _style(ax)


# ---------------------------------------------------------------- figures

def _present(data: dict, env: str) -> list[str]:
    """Methods (fixed order, never cycled) with completed rows for this env."""
    return [v for v in METHODS if data.get(env, {}).get(v, {}).get("reports")]


def fig_final_return(plt, data: dict, out, assets, t) -> bool:
    envs = [e for e in ENVS if _present(data, e)]
    if not envs:
        return False
    series = [v for v in METHODS if any(v in _present(data, e) for e in envs)]
    if not series:
        return False
    vals, errs = [], []
    for e in envs:
        row_v, row_e = [], []
        for v in series:
            ent = data[e].get(v)
            m, s, _ = _metric(ent, "return_mean") if ent and ent["reports"] else (np.nan, 0.0, 0)
            row_v.append(m)
            row_e.append(s)
        vals.append(row_v)
        errs.append(row_e)
    fig, ax = plt.subplots(figsize=(max(7, 2.6 * len(envs)), 4.2))
    _grouped_bars(ax, [t["env_labels"].get(e, e) for e in envs], series,
                  vals, errs, {v: METHODS[v][1] for v in series},
                  {v: t["method_labels"].get(v, v) for v in series}, t)
    ax.set_ylabel(t["fig1_ylabel"])
    if t["fig1_title"]:
        ax.set_title(t["fig1_title"], color=INK, pad=10)
    ax.legend(frameon=False, fontsize=8, ncol=min(4, len(series)),
              loc="upper center", bbox_to_anchor=(0.5, -0.13))
    ax.annotate(t["seeds_note"], (0, -0.30), xycoords="axes fraction",
                fontsize=7, color=MUT)
    _save(fig, out, assets, f"fig1_final_return{t['suffix']}.png")
    plt.close(fig)
    return True


def fig_per_regime(plt, data: dict, out, assets, t, env="relation") -> bool:
    series = _present(data, env)
    if not series:
        return False
    regimes = sorted({int(g) for v in series
                      for rep in data[env][v]["reports"].values()
                      for g in (rep.get("return_per_regime") or {})})
    if not regimes:
        return False
    vals, errs = [], []
    for g in regimes:
        rv, re_ = [], []
        for v in series:
            m, s, _ = _regime_metric(data[env][v], g)
            rv.append(m)
            re_.append(s)
        vals.append(rv)
        errs.append(re_)
    fig, ax = plt.subplots(figsize=(max(8, 1.7 * len(regimes)), 4.2))
    _grouped_bars(ax, [t["regime_labels"].get(g, f"g{g}") for g in regimes],
                  series, vals, errs, {v: METHODS[v][1] for v in series},
                  {v: t["method_labels"].get(v, v) for v in series}, t)
    ax.set_ylabel(t["fig2_ylabel"])
    if t["fig2_title"]:
        ax.set_title(t["fig2_title"], color=INK, pad=10)
    ax.legend(frameon=False, fontsize=8, ncol=min(4, len(series)),
              loc="upper center", bbox_to_anchor=(0.5, -0.13))
    _save(fig, out, assets, f"fig2_per_regime{t['suffix']}.png")
    plt.close(fig)
    return True


def fig_generalization(plt, data: dict, out, assets, t,
                       env="relation_holdout") -> bool:
    series = _present(data, env)
    if not series:
        return False
    vals, errs = [], []
    for key in ("return_zero_shot_seen", "return_zero_shot_unseen"):
        rv, re_ = [], []
        for v in series:
            m, s, _ = _metric(data[env][v], key)
            rv.append(m)
            re_.append(s)
        vals.append(rv)
        errs.append(re_)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    _grouped_bars(ax, t["fig3_groups"], series, vals, errs,
                  {v: METHODS[v][1] for v in series},
                  {v: t["method_labels"].get(v, v) for v in series}, t)
    ax.set_ylabel(t["fig3_ylabel"])
    if t["fig3_title"]:
        ax.set_title(t["fig3_title"], color=INK, pad=10)
    ax.legend(frameon=False, fontsize=8, ncol=min(4, len(series)),
              loc="upper center", bbox_to_anchor=(0.5, -0.13))
    _save(fig, out, assets, f"fig3_generalization{t['suffix']}.png")
    plt.close(fig)
    return True


def fig_fidelity(plt, data: dict, out, assets, t, env="relation") -> bool:
    """Headline metric ① — world-model reward fidelity (lower is better)."""
    series = [v for v in METHODS if data.get(env, {}).get(v, {}).get("fidelity")]
    if not series:
        return False
    means, sems = [], []
    for v in series:
        m, s = _agg(list(data[env][v]["fidelity"].values()))
        means.append(m)
        sems.append(s)
    fig, ax = plt.subplots(figsize=(max(5.5, 1.5 * len(series)), 4.0))
    x = np.arange(len(series), dtype=float)
    bars = ax.bar(x, means, 0.55, yerr=sems, capsize=3, zorder=3,
                  color=[METHODS[v][1] for v in series],
                  error_kw={"elinewidth": 0.9, "ecolor": SEC})
    _bar_labels(ax, bars, means, fmt="{:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels([t["method_labels"].get(v, v) for v in series],
                       fontsize=8)
    ax.set_ylabel(t["fig4_ylabel"])
    if t["fig4_title"]:
        ax.set_title(t["fig4_title"], color=INK, pad=10)
    _style(ax)
    na = [t["method_labels"].get(v, v) for v in NO_FIDELITY
          if v in _present(data, env)]
    if na:
        ax.annotate(f"{', '.join(na)}: {t['fig4_na']}",
                    (0, -0.22), xycoords="axes fraction", fontsize=7, color=MUT)
    _save(fig, out, assets, f"fig4_fidelity{t['suffix']}.png")
    plt.close(fig)
    return True


def fig_ablation(plt, data: dict, out, assets, t, env="relation") -> bool:
    series = [v for v in ARMS if data.get(env, {}).get(v, {}).get("reports")]
    if len(series) < 2:                    # a lone reference is not a comparison
        return False
    means, sems = [], []
    for v in series:
        m, s, _ = _metric(data[env][v], "return_mean")
        means.append(m)
        sems.append(s)
    fig, ax = plt.subplots(figsize=(max(6.5, 1.5 * len(series)), 4.2))
    x = np.arange(len(series), dtype=float)
    bars = ax.bar(x, means, 0.55, yerr=sems, capsize=3, zorder=3,
                  color=[ARMS[v][1] for v in series],
                  error_kw={"elinewidth": 0.9, "ecolor": SEC})
    _bar_labels(ax, bars, means)
    # the unablated reference as a rule, so each arm reads as a delta from it
    if ABLATION_REF in series:
        ref = means[series.index(ABLATION_REF)]
        ax.axhline(ref, color=METHODS[ABLATION_REF][1], linewidth=1.2,
                   linestyle=(0, (5, 3)), zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([t["arm_labels"].get(v, v) for v in series],
                       fontsize=8, rotation=15, ha="right")
    ax.set_ylabel(t["fig5_ylabel"])
    if t["fig5_title"]:
        ax.set_title(t["fig5_title"], color=INK, pad=10)
    _style(ax)
    _save(fig, out, assets, f"fig5_ablation{t['suffix']}.png")
    plt.close(fig)
    return True


def fig_sample_efficiency(plt, data: dict, out, assets, t, env="relation") -> bool:
    curved = [v for v in METHODS if data.get(env, {}).get(v, {}).get("curves")]
    endpoints = [v for v in _present(data, env) if v not in curved]
    if not curved:
        return False
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for v in curved:
        ent = data[env][v]
        # interpolate seeds onto a shared grid so the band is a real spread
        series = list(ent["curves"].values())
        xs = np.unique(np.concatenate([np.asarray([p[0] for p in s])
                                       for s in series]))
        stack = np.vstack([
            np.interp(xs, [p[0] for p in s], [p[1] for p in s]) for s in series])
        mean = stack.mean(axis=0)
        ax.plot(xs, mean, color=METHODS[v][1], linewidth=2,
                label=t["method_labels"].get(v, v), zorder=3)
        if stack.shape[0] > 1:
            sem = stack.std(axis=0, ddof=1) / math.sqrt(stack.shape[0])
            ax.fill_between(xs, mean - sem, mean + sem, color=METHODS[v][1],
                            alpha=0.15, linewidth=0, zorder=2)
    for v in endpoints:
        ent = data[env][v]
        m, s, _ = _metric(ent, "return_mean")
        budget = float(ent["rows"][0].get("total_env_steps", 0) or 0)
        if budget <= 0:
            continue
        ax.errorbar([budget], [m], yerr=[s], fmt="o", markersize=7,
                    color=METHODS[v][1], markeredgecolor=SURF,
                    markeredgewidth=1.5, capsize=3, zorder=4,
                    label=t["method_labels"].get(v, v))
    ax.set_xlabel(t["fig6_xlabel"])
    ax.set_ylabel(t["fig6_ylabel"])
    if t["fig6_title"]:
        ax.set_title(t["fig6_title"], color=INK, pad=10)
    _style(ax)
    ax.legend(frameon=False, fontsize=8, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax.annotate(t["fig6_note"], (0, -0.34), xycoords="axes fraction",
                fontsize=7, color=MUT, wrap=True)
    _save(fig, out, assets, f"fig6_sample_efficiency{t['suffix']}.png")
    plt.close(fig)
    return True


# ---------------------------------------------------------------- table

def write_table(data: dict, out: pathlib.Path) -> None:
    """Markdown summary — also the accessible table view the palette's
    contrast WARN obligates (three slots sit below 3:1 on the light surface)."""
    out.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Baseline comparison — v5 formal grid", "",
        "Mean ± s.e.m. over seeds. `fidelity` = world-model reward MAE "
        "(lower better); `—` = no world-model hook (N/A by design).", "",
    ]
    for env in ENVS:
        present = [v for v in METHODS if data.get(env, {}).get(v, {}).get("reports")]
        arms = [v for v in ARMS if v != ABLATION_REF
                and data.get(env, {}).get(v, {}).get("reports")]
        if not present and not arms:
            continue
        lines += [f"## {env}", "",
                  "| method | seeds | return | seen | unseen | gap | fidelity |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for v in present + arms:
            ent = data[env][v]
            label = (METHODS.get(v) or ARMS.get(v))[0]
            m, s, n = _metric(ent, "return_mean")
            sn, _, _ = _metric(ent, "return_zero_shot_seen")
            un, _, _ = _metric(ent, "return_zero_shot_unseen")
            gp, _, _ = _metric(ent, "return_zero_shot_gap")
            if ent["fidelity"]:
                fm_, fs = _agg(list(ent["fidelity"].values()))
                fid = f"{fm_:.4f} ± {fs:.4f}"
            else:
                fid = "—"
            lines.append(
                f"| {label} | {n} | {m:.2f} ± {s:.2f} | {sn:.2f} | {un:.2f} "
                f"| {gp:.2f} | {fid} |")
        lines.append("")
    path = out / "table1_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[make_baseline_comparison] wrote {path}")


# ---------------------------------------------------------------- entry

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--registry", type=pathlib.Path,
                    default=pathlib.Path("results/registry.jsonl"),
                    help="flat registry written by scripts/train.py")
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("results/analysis/v5_formal"))
    ap.add_argument("--assets", type=pathlib.Path,
                    default=pathlib.Path("docs/assets/interim_report"))
    ap.add_argument("--no-assets", action="store_true")
    ap.add_argument("--lang", choices=["en", "zh"], default="en",
                    help="figure text language + style (zh = publication style "
                         "for the Chinese report: no titles, serif, 300 dpi)")
    args = ap.parse_args(argv)
    assets = None if args.no_assets else args.assets
    t = I18N[args.lang]

    data = collect(args.registry)
    n_runs = sum(len(ent["rows"]) for env in data.values() for ent in env.values())
    if not n_runs:
        print(f"[make_baseline_comparison] no completed runs in {args.registry} "
              f"— nothing to plot yet")
        return 0
    print(f"[make_baseline_comparison] {n_runs} completed run(s) across "
          f"{sum(1 for e in data.values() if e)} env(s)")
    plt = _mpl(args.lang)

    made = {
        "fig1_final_return": fig_final_return(plt, data, args.out, assets, t),
        "fig2_per_regime": fig_per_regime(plt, data, args.out, assets, t),
        "fig3_generalization": fig_generalization(plt, data, args.out, assets, t),
        "fig4_fidelity": fig_fidelity(plt, data, args.out, assets, t),
        "fig5_ablation": fig_ablation(plt, data, args.out, assets, t),
        "fig6_sample_efficiency": fig_sample_efficiency(plt, data, args.out, assets, t),
    }
    write_table(data, args.out)
    missing = [k for k, ok in made.items() if not ok]
    if missing:
        print(f"[make_baseline_comparison] skipped (no data yet): "
              f"{', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
