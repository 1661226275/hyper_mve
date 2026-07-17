"""make_baseline_comparison.py — mid-term defense comparison artifacts (Part D).

Assembles the ours-vs-MAMBA-vs-MAPPO(-vs-QMIX) deliverables from real run
artifacts (registries + eval_report.json + TB scalars + game-metrics JSONs):

  Fig 1  sample-efficiency curves — eval return vs cumulative env steps, log-x.
         Internal rows (hyper) log eval at train-step cadence; their x-axis is
         converted via env_steps = train_step × (episodes_per_iter × T_max /
         train_steps_per_iter), read from each row's config snapshot. External
         rows' PeriodicEvalProbe tags are already keyed by env steps.
  Fig 2  per-regime return bars (gate cell, final EvalReport.return_per_regime).
  Fig 2b per-episode return distribution @2M (companion box plot to fig 2,
         a separate figure so it can be placed independently in the report).
  Fig 3  seen vs held-out regime generalization (holdout cell EvalReports).
  Fig 4  NashConv per regime (game_metrics_<variant>.json, lower bound).
  Table 1 markdown summary (budgets disclosed per method).

Run from the repo root:

    python hyper_mve/scripts/make_baseline_comparison.py \
        --out runs/_analysis/rel_defense --assets docs/assets/interim_report

Partial-data safe: every artifact is produced from whatever rows are completed
(curves also use still-running rows' TB); missing pieces are reported, not fatal.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

# ---------------------------------------------------------------- constants

# Fixed method order + categorical slots (validated palette; color follows the
# entity across every figure — see dataviz palette reference).
METHODS: dict[str, tuple[str, str]] = {
    "hyper":          ("DR-MBHGL (ours)", "#2a78d6"),   # slot 1 blue
    "external_mamba": ("MAMBA",           "#1baf7a"),   # slot 2 aqua
    "external_mappo": ("MAPPO",           "#eda100"),   # slot 3 yellow
    "external_qmix":  ("QMIX",            "#008300"),   # slot 4 green
}

REGIME_LABELS = {
    0: "g0\nmutual_coop", 1: "g1\nmutual_comp", 2: "g2\nasym_exploit",
    3: "g3\nasym_exploited", 4: "g4\nneutral",
}

SURF, INK, SEC, MUT, GRID, BASE = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7",
)

# Marks reserved for PROJECTED (fitted-extrapolation) values of an in-progress
# run — a dashed forward curve, a hollow endpoint, and hatched bars — so a
# projection can never be read as a measured result.
PROJ_DASH = (0, (5, 3))
PROJ_HATCH = "////"
# Acceptance gate for projecting a saturating fit (applied identically in the
# table and every figure so they never disagree). Two ways to pass:
#   * R² ≥ R2_MIN — a clear saturating trend the model explains; or
#   * mild extrapolation — |proj − last| ≤ 2×residual-RMSE. This covers the
#     plateaued case, where R² ≈ 0 by construction (no variance to explain)
#     yet extending the plateau is statistically safe.
# A fit that fails both (e.g. a steep rise conjured from noise) is NOT drawn.
R2_MIN = 0.6


def _fit_ok(fit: dict | None) -> bool:
    if not fit:
        return False
    if fit["r2"] >= R2_MIN:
        return True
    return abs(fit["proj"] - fit["last"]) <= 2.0 * fit["rmse"]

# MAPPO's per-checkpoint eval return is visibly noisier than the other two
# real series (no world model damping policy-driven variance) — give it a
# wider centered rolling-mean window in fig 2's line panels so the trend
# reads clearly. Disclosed in the report caption text, not silently applied.
PANEL_SMOOTH_K = {"external_mappo": 9}

# Sample-efficiency (fig 1) x-window: the first ~5k env steps are a warm-up with
# no evaluation (a flat artifact), and every method is compared over the same
# [5k, 2M] env-step budget — so hyper is shown to 2M here too (its full 200k-grad
# result stays the table/bar headline).
X_MIN_ENV = 5_000.0
X_MAX_ENV = 2_000_000.0

# Per-language strings. "zh" is the publication style for the Chinese report:
# no in-figure titles (the report caption carries them), Times/SimSun serif,
# white background, 300 dpi, "_zh" filename suffix.
I18N: dict[str, dict] = {
    "en": {
        "suffix": "",
        "surf": SURF,
        "method_labels": {v: label for v, (label, _c) in METHODS.items()},
        "regime_labels": REGIME_LABELS,
        "n_fmt": "{label} (n={n})",
        "fig1_title": "Sample efficiency — mixed hidden regimes (rel_gate_duo)",
        "fig1_xlabel": "Environment steps (log scale)",
        "fig1_ylabel": "Mean eval return (all regimes)",
        "fig2_title": "Global reward during training — per regime (rel_gate_duo)",
        "fig2_ylabel": "Eval return",
        "fig2_avg_title": "Average (all 5 regimes)",
        "fig2_box_title": "Return distribution @2M (30 ep)",
        "fig2_xlabel": "Environment steps",
        "fig2_panel_titles": {
            0: "g0 mutual_coop", 2: "g2 asym_exploit",
            3: "g3 asym_exploited", 4: "g4 neutral",
        },
        "box_progress_fmt": "{label} @{step} (in progress)",
        "fig3_title": "Generalization to held-out regimes — rel_zero_shot_duo",
        "fig3_groups": ["Training regimes\n(coop / comp / neutral)",
                        "Held-out regimes\n(asymmetric)"],
        "fig3_ylabel": "Mean eval return (seen / unseen regimes)",
        "fig4_title": "Approximate exploitability per regime (DQN best response)",
        "fig4_ylabel": "NashConv (lower bound; ↓ better)",
        "proj_tag": "(proj.)",
        "proj_note": ("Dashed = fitted extrapolation of MAMBA's accurate 30-episode "
                      "checkpoint evals to the full 2M env-step budget — provisional, "
                      "to be replaced by the measured result at completion."),
        "proj_bar_tag": " (proj. @2M)",
        "proj_note_bar": ("Dashed = fitted extrapolation of MAMBA's 30-episode "
                          "checkpoint evals to 2M env steps (provisional); hatched "
                          "box = measured distribution at MAMBA's latest checkpoint "
                          "(training in progress)."),
    },
    "zh": {
        "suffix": "_zh",
        "surf": "#ffffff",
        "method_labels": {
            "hyper":          "DR-MBHGL（本文方法）",
            "external_mamba": "MAMBA",
            "external_mappo": "MAPPO",
            "external_qmix":  "QMIX",
        },
        "regime_labels": {
            0: "g0\n互利合作", 1: "g1\n互相竞争",
            2: "g2\n非对称利用\n（利用方）", 3: "g3\n非对称利用\n（被利用方）",
            4: "g4\n中立",
        },
        "n_fmt": "{label}（n={n}）",
        "fig1_title": None,
        "fig1_xlabel": "环境交互步数（对数坐标）",
        "fig1_ylabel": "全情景平均评估回报",
        "fig2_title": None,
        "fig2_ylabel": "评估回报",
        "fig2_avg_title": "全部情景平均",
        "fig2_box_title": "2M 步回报分布（30 回合）",
        "fig2_xlabel": "环境交互步数",
        "fig2_panel_titles": {
            0: "g0 互利合作", 2: "g2 非对称利用（利用方）",
            3: "g3 非对称利用（被利用方）", 4: "g4 中立",
        },
        "box_progress_fmt": "{label} @{step}（训练中）",
        "fig3_title": None,
        "fig3_groups": ["训练所见构型\n（合作/竞争/中立）",
                        "训练未见构型\n（非对称）"],
        "fig3_ylabel": "所见/未见情景平均回报",
        "fig4_title": None,
        "fig4_ylabel": "NashConv（下界，越低越好）",
        "proj_tag": "（预测）",
        "proj_note": ("虚线为对 MAMBA 各中间检查点完整 30 回合评估"
                      "拟合外推至 200 万环境交互步的预测值（训练进行中，暂定，"
                      "后续以实测结果替换）。"),
        "proj_bar_tag": "（预测@2M）",
        "proj_note_bar": ("虚线为对 MAMBA 各中间检查点完整 30 回合评估拟合外推至 "
                          "200 万环境交互步的预测值（暂定）；斜纹箱线为 MAMBA "
                          "最新检查点的实测回报分布（训练进行中）。"),
    },
}

GATE_CELL, ZS_CELL = "rel_gate_duo", "rel_zero_shot_duo"


# ---------------------------------------------------------------- data layer

def _latest_rows(registry_path: pathlib.Path) -> list[dict]:
    """Latest state per (variant, seed, config_hash) — any status."""
    if not registry_path.exists():
        return []
    latest: dict[tuple, dict] = {}
    for line in registry_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        latest[(row.get("variant"), row.get("seed"), row.get("config_hash"))] = row
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


def _env_step_factor(row: dict) -> float:
    """Internal rows: env steps per train step from the row's config snapshot."""
    cfg = _load_json(row.get("config_snapshot_path")) or {}
    train = cfg.get("train", {})
    eppi = float(train.get("episodes_per_iter", 8))
    tspi = float(train.get("train_steps_per_iter", 8))
    tmax = float((cfg.get("env") or {}).get("T_max", 100))
    return eppi * tmax / max(tspi, 1.0)


def _row_budget_env_steps(row: dict) -> float:
    cfg = _load_json(row.get("config_snapshot_path")) or {}
    mts = float((cfg.get("train") or {}).get("max_train_steps", 0))
    if row["variant"] in METHODS and row["variant"] != "hyper":
        return mts                       # externals: env-step budget directly
    return mts * _env_step_factor(row)


def _curve(row: dict) -> list[tuple[float, float]] | None:
    """(env_steps, eval return) series for one row, internal or external."""
    from hyper_mve.experiments.analysis.tb_scraper import scrape_tb

    tb = row.get("tensorboard_dir")
    if not tb or not pathlib.Path(tb).exists():
        return None
    if row["variant"] == "hyper":
        tags = ["eval/planner/return_total"]
        factor = _env_step_factor(row)
    else:
        tags = ["eval/return_mean"]
        factor = 1.0
    try:
        series = scrape_tb(tb, tags).get(tags[0]) or []
    except ImportError:
        return None
    return [(step * factor, val) for step, val in series] or None


def _curves_per_regime(row: dict, regimes=(0, 1, 2, 3, 4)) -> dict[int, list]:
    """Per-regime (env_steps, return) training curves for one row.

    hyper: ``eval/planner/return_total_g{g}`` (train-step x → ×100 env-step
    factor). Externals: PeriodicEvalProbe ``eval/return_g{g}`` (already env-step
    keyed; 2-episode probe — endpoints get anchored to the measured eval)."""
    from hyper_mve.experiments.analysis.tb_scraper import scrape_tb

    tb = row.get("tensorboard_dir")
    if not tb or not pathlib.Path(tb).exists():
        return {}
    if row["variant"] == "hyper":
        tag_of = {g: f"eval/planner/return_total_g{g}" for g in regimes}
        factor = _env_step_factor(row)
    else:
        tag_of = {g: f"eval/return_g{g}" for g in regimes}
        factor = 1.0
    try:
        raw = scrape_tb(tb, list(tag_of.values()))
    except ImportError:
        return {}
    out: dict[int, list] = {}
    for g, tag in tag_of.items():
        ser = raw.get(tag) or []
        if ser:
            out[g] = [(float(s) * factor, float(v)) for s, v in ser]
    return out


def _fit_project(series, target_x: float) -> dict | None:
    """Fit a saturating learning curve ``y = a + b·e^(−k·x)`` to real
    (env_step, value) points and project the value at ``target_x``.

    Returns ``{'proj','fn','r2','last','last_x'}`` or ``None``. x is scaled to
    millions of env steps for numerical stability; ``k`` is bounded strictly
    positive so the model saturates (no runaway linear extrapolation). A low R²
    signals the trend is not yet a clean saturating curve — the caller decides
    whether to trust the projection or fall back to the last observed value.
    Every value produced here is an EXTRAPOLATION, replaced by the measured
    eval_report once training finishes.
    """
    if not series or len(series) < 5:
        return None
    pts = sorted(series)
    xs = np.asarray([p[0] for p in pts], float)
    ys = np.asarray([p[1] for p in pts], float)
    if float(xs[-1]) <= 0 or target_x <= xs[-1]:
        return None
    xn, tn = xs / 1e6, target_x / 1e6
    last = float(ys[-1])

    def model(x, a, b, k):
        return a + b * np.exp(-k * x)

    # Fit the denoised trend (eval curves are noisy) but score R² against the raw
    # points, so smoothing aids parameter estimation without inflating goodness.
    yf = _smooth(ys, 3)
    try:
        from scipy.optimize import curve_fit
        popt, _ = curve_fit(
            model, xn, yf, p0=[last, yf[0] - last, 3.0],
            bounds=([-1e4, -1e4, 0.1], [1e4, 1e4, 60.0]), maxfev=40000,
        )
    except Exception:
        return None
    resid = ys - model(xn, *popt)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2)) or 1.0
    r2 = 1.0 - ss_res / ss_tot
    return {
        "proj": float(model(tn, *popt)),
        "fn": (lambda xe: float(model(xe / 1e6, *popt))),
        "r2": r2, "last": last, "last_x": float(xs[-1]),
        "rmse": float(np.sqrt(ss_res / max(len(ys), 1))),
    }


def _series_to_curves(points: list[dict]) -> dict[str, list[tuple[float, float]]]:
    """Reshape accurate checkpoint-eval points into {tag: [(env_step, value)]}."""
    rm, seen, unseen = [], [], []
    per: dict[int, list] = {g: [] for g in range(8)}
    for p in points:
        x = float(p["env_step"])
        rm.append((x, float(p["return_mean"])))
        seen.append((x, float(p["seen"])))
        unseen.append((x, float(p["unseen"])))
        for g, v in (p.get("per_regime") or {}).items():
            per[int(g)].append((x, float(v)))
    out = {"eval/return_mean": rm, "eval/return_seen": seen, "eval/return_unseen": unseen}
    for g, ser in per.items():
        if ser:
            out[f"eval/return_g{g}"] = ser
    return out


def _projected_summary(ent: dict) -> dict | None:
    """Project an in-progress MAMBA run to its full budget from its accurate
    checkpoint-eval series (real 30-episode evals at intermediate checkpoints).

    Fits each metric (return_mean, per-regime g, seen, unseen) independently and
    returns the projected values, or ``None`` if there is no fittable series yet.
    All outputs are extrapolations — swapped for the measured eval_report once
    the run completes.
    """
    curves = ent.get("accurate") or {}
    target = ent.get("target")
    if not curves or not target:
        return None
    rm = _fit_project(curves.get("eval/return_mean"), target)
    if not _fit_ok(rm):
        return None                         # extrapolation not yet trustworthy
    per_regime = {}
    for g in range(8):
        f = _fit_project(curves.get(f"eval/return_g{g}"), target)
        if _fit_ok(f):
            per_regime[g] = f["proj"]
    seen = _fit_project(curves.get("eval/return_seen"), target)
    unseen = _fit_project(curves.get("eval/return_unseen"), target)
    return {
        "return_mean": rm["proj"], "per_regime": per_regime,
        "seen": seen["proj"] if _fit_ok(seen) else None,
        "unseen": unseen["proj"] if _fit_ok(unseen) else None,
        "r2": rm["r2"], "last_x": rm["last_x"], "target": float(target),
    }


_MAMBA_SERIES = {
    GATE_CELL: "mamba_ckpt_series_gate.json",
    ZS_CELL: "mamba_ckpt_series_zeroshot.json",
}


def collect(runs_root: pathlib.Path, external_budget: float = 2_000_000.0,
            mamba_series_dir: pathlib.Path | None = None,
            final_evals_dir: pathlib.Path | None = None,
            mamba_freeze_env_step: float | None = None) -> dict:
    """{cell: {variant: {"rows": [...], "reports": {seed: dict}, "curves": {seed: [...]}}}}

    External rows are restricted to those trained at ``external_budget`` env steps
    (±1%) so the comparison is at a single budget — this drops earlier runs at
    other budgets (e.g. a prior MAMBA@200k / MAPPO@1M) that share the variant key.
    ``hyper`` is never budget-filtered (its budget is the grad-step schedule),
    but its REPORT values are replaced by the 2M-standard checkpoint eval when
    ``final_evals_dir`` holds one (see below) — the whole comparison sits at one
    env-step budget.

    For an in-progress MAMBA row, ``mamba_ckpt_series_<cell>.json`` (accurate
    30-episode evals at intermediate checkpoints) is loaded into ``ent['accurate']``
    and used as its fig-1 curve + projection source — never the noisy 2-ep probe.
    That series file is written continuously by a separate, still-running eval
    loop; ``mamba_freeze_env_step`` (if given) drops any point past that env
    step so every figure/table built from one ``collect()`` call agrees with a
    single MAMBA snapshot instead of silently drifting mid-report as more
    checkpoints land — set it to whatever step the surrounding report text
    already quotes.

    ``final_evals_dir`` (collect_final_evals.py output) overlays, per variant:
      * hyper  — pseudo-report from its 2M checkpoint (step_20000.pt, 30 ep) +
        per-episode returns (``ent['episodes']``) + ``ent['budget_override']``.
      * mappo  — per-episode returns for the box plot (report already at 2M).
      * mamba  — per-episode returns at its LATEST checkpoint (``ent['box_step']``,
        in-progress; box plot only, never a report).
    """
    out: dict = {}
    for cell in (GATE_CELL, ZS_CELL):
        cell_data: dict = {}
        for row in _latest_rows(runs_root / cell / "registry.jsonl"):
            v = row.get("variant")
            if v not in METHODS:
                continue
            if v != "hyper" and external_budget > 0:
                b = _row_budget_env_steps(row)
                if b <= 0 or abs(b - external_budget) / external_budget > 0.01:
                    continue                 # not the requested external budget
            ent = cell_data.setdefault(v, {"rows": [], "reports": {}, "curves": {}})
            ent["rows"].append(row)
            if row.get("status") == "completed":
                rep = _load_json(row.get("eval_report_path"))
                if rep:
                    ent["reports"][int(row.get("seed", 0))] = rep
            # In-progress externals contribute NO probe-based curves — their
            # only curve source is the accurate checkpoint-eval series loaded
            # below (user decision: never present the noisy 2-ep probe).
            in_progress_ext = row.get("status") != "completed" and v != "hyper"
            if not in_progress_ext:
                c = _curve(row)
                if c:
                    prev = ent["curves"].get(int(row.get("seed", 0)))
                    if prev is None or len(c) > len(prev):
                        ent["curves"][int(row.get("seed", 0))] = c
                cg = _curves_per_regime(row)
                for g, ser in cg.items():
                    prev_g = ent.setdefault("curves_g", {}).get(g)
                    if prev_g is None or len(ser) > len(prev_g):
                        ent["curves_g"][g] = ser
            # Completed external: anchor the fig-1 curve's right end to the
            # measured eval_report return (the probe trace is noisy; bars + table
            # use this value, so the curve endpoint must match it).
            if row.get("status") == "completed" and v != "hyper":
                sd = int(row.get("seed", 0))
                rep2 = ent["reports"].get(sd)
                budget = _row_budget_env_steps(row)
                cur = ent["curves"].get(sd)
                if rep2 and budget > 0 and cur:
                    cur = [(x, y) for (x, y) in cur if x < budget]
                    cur.append((budget, float(rep2.get("return_mean", 0.0))))
                    ent["curves"][sd] = cur
            # In-progress external row: remember its full env-step budget as the
            # projection target.
            if row.get("status") != "completed" and v != "hyper":
                ent["target"] = _row_budget_env_steps(row)
        # In-progress MAMBA: accurate 30-ep checkpoint-eval series → fig-1 curve
        # + projection source (replaces the noisy 2-episode probe).
        ent = cell_data.get("external_mamba")
        if ent is not None and not ent.get("reports") and mamba_series_dir:
            series = _load_json(str(mamba_series_dir / _MAMBA_SERIES[cell]))
            pts = (series or {}).get("points") or []
            if mamba_freeze_env_step is not None:
                pts = [p for p in pts if p.get("env_step", 0) <= mamba_freeze_env_step]
            if len(pts) >= 3:
                curves = _series_to_curves(pts)
                ent["accurate"] = curves
                ent["target"] = float((series or {}).get("budget_target", external_budget))
                ent["curves"] = {0: curves["eval/return_mean"]}
                ent["curves_g"] = {g: curves[f"eval/return_g{g}"]
                                   for g in range(8) if f"eval/return_g{g}" in curves}
        # 2M-standard overlay (collect_final_evals.py output).
        if final_evals_dir:
            for v in METHODS:
                ent = cell_data.get(v)
                if ent is None:
                    continue
                fe = _load_json(str(final_evals_dir / f"{v}_{cell}.json"))
                if not fe:
                    continue
                per_ep = {int(g): list(vals)
                          for g, vals in (fe.get("per_regime_returns") or {}).items()}
                if v == "external_mamba":
                    ent["box_episodes"] = per_ep       # latest ckpt — box only
                    ent["box_step"] = int(fe.get("env_step", 0))
                    continue
                ent["episodes"] = per_ep
                # hyper + mappo: the collector eval (one uniform protocol — 30
                # deterministic episodes/regime at the 2M checkpoint) becomes
                # the canonical report, so bars/lines/box/table all quote the
                # same episode set for every method.
                ent["reports"] = {0: {
                    "return_mean": fe["return_mean"],
                    "return_per_regime": fe["per_regime_mean"],
                    "return_zero_shot_seen": fe["seen"],
                    "return_zero_shot_unseen": fe["unseen"],
                }}
                if v == "hyper":
                    ent["budget_override"] = float(fe.get("env_step", external_budget))
        out[cell] = cell_data
    return out


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
        "hatch.linewidth": 1.1,          # projected-bar hatch visibility
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
            # per-glyph fallback actually walks Times New Roman -> SimSun
            # instead of resolving "serif" to a single font up front.
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


# ---------------------------------------------------------------- figures

def _smooth(y: np.ndarray, k: int = 5) -> np.ndarray:
    """Centered rolling mean (curve readability; raw kept as faint line)."""
    if len(y) < 3 * k:
        return y
    pad = k // 2
    yp = np.pad(y, (pad, k - 1 - pad), mode="edge")
    return np.convolve(yp, np.ones(k) / k, mode="valid")


def fig_sample_efficiency(plt, data: dict, out: pathlib.Path, assets, t: dict) -> bool:
    cell = data.get(GATE_CELL, {})
    any_curve = False
    projected = False
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for v, (_label, color) in METHODS.items():
        label = t["method_labels"][v]
        ent = cell.get(v) or {}
        curves = ent.get("curves") or {}
        if not curves:
            continue
        seeds = sorted(curves)
        if len(seeds) == 1:
            xs, ys = map(np.asarray, zip(*curves[seeds[0]]))
            win = (xs >= X_MIN_ENV) & (xs <= X_MAX_ENV)   # drop warm-up; cap at 2M
            xs, ys = xs[win], ys[win]
            if len(xs) < 3:
                continue                     # too few points in-window to draw a curve
            any_curve = True
            # Endpoint pinned to the accurate 30-episode 2M checkpoint eval —
            # the same value fig2/box/table quote.
            anchor = _panel_anchor(ent, None)
            xs, ys = _apply_anchor(xs, ys, anchor)
            ax.plot(xs, ys, color=color, linewidth=0.9, alpha=0.25)
            ys_s = _smooth(ys)
            ax.plot(xs, ys_s, color=color, linewidth=2, label=label)
            end_x, end_y = xs[-1], (anchor if anchor is not None else ys_s[-1])
            # In-progress external → fitted extrapolation to its full budget.
            fit = (_fit_project(list(zip(xs.tolist(), ys.tolist())), ent["target"])
                   if not ent.get("reports") and ent.get("target") else None)
            if _fit_ok(fit):
                ext_x = np.geomspace(max(xs[-1], 1.0), ent["target"], 60)
                ax.plot(ext_x, [fit["fn"](x) for x in ext_x], color=color,
                        linewidth=1.7, ls=PROJ_DASH, alpha=0.9)
                ax.plot([ent["target"]], [fit["proj"]], marker="o", linestyle="",
                        mfc=t["surf"], mec=color, mew=1.7, ms=7)
                end_x, end_y, projected = ent["target"], fit["proj"], True
                ax.annotate(f"{end_y:.1f} " + t["proj_tag"], (end_x, end_y),
                            textcoords="offset points", xytext=(4, 0),
                            fontsize=8, color=color, fontstyle="italic")
                continue
        else:
            any_curve = True
            lo = max(min(x for x, _ in curves[s]) for s in seeds)
            hi = min(max(x for x, _ in curves[s]) for s in seeds)
            lo, hi = max(lo, X_MIN_ENV), min(hi, X_MAX_ENV)   # [5k, 2M] window
            grid_x = np.geomspace(max(lo, 1.0), max(hi, lo * 2), 200)
            mat = []
            for s in seeds:
                xs, ys = zip(*curves[s])
                mat.append(np.interp(grid_x, xs, ys))
            mat = np.asarray(mat)
            mean, sem = mat.mean(0), mat.std(0) / np.sqrt(mat.shape[0])
            mean_s = _smooth(mean)
            ax.plot(grid_x, mean, color=color, linewidth=0.9, alpha=0.25)
            ax.plot(grid_x, mean_s, color=color, linewidth=2,
                    label=t["n_fmt"].format(label=label, n=len(seeds)))
            ax.fill_between(grid_x, _smooth(mean - sem), _smooth(mean + sem),
                            color=color, alpha=0.18, linewidth=0)
            end_x, end_y = grid_x[-1], mean_s[-1]
        ax.annotate(f"{end_y:.1f}", (end_x, end_y), textcoords="offset points",
                    xytext=(4, 0), fontsize=8, color=INK)
    if not any_curve:
        plt.close(fig)
        return False
    ax.set_xscale("log")
    ax.set_xlim(X_MIN_ENV, X_MAX_ENV)
    _style(ax)
    ax.set_xlabel(t["fig1_xlabel"])
    ax.set_ylabel(t["fig1_ylabel"])
    if t["fig1_title"]:
        ax.set_title(t["fig1_title"])
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    if projected:
        fig.text(0.5, -0.02, t["proj_note"], ha="center", va="top",
                 fontsize=7.5, color=SEC, wrap=True)
    _save(fig, out, assets, f"fig1_sample_efficiency{t['suffix']}.png")
    plt.close(fig)
    return True


def _per_regime_stats(cell: dict, v: str, key: str = "return_per_regime"):
    reports = (cell.get(v) or {}).get("reports") or {}
    if not reports:
        return None
    per_seed = []
    for rep in reports.values():
        d = rep.get(key) or {}
        per_seed.append({int(k): float(val) for k, val in d.items()})
    regimes = sorted(set().union(*per_seed))
    mean = {g: float(np.mean([d.get(g, np.nan) for d in per_seed])) for g in regimes}
    sem = {
        g: (float(np.std([d[g] for d in per_seed]) / np.sqrt(len(per_seed)))
            if len(per_seed) > 1 else 0.0)
        for g in regimes
    }
    return regimes, mean, sem, len(per_seed)


def _panel_series(ent: dict, g: int | None) -> list[tuple[float, float]] | None:
    """One method's training curve for a fig-2 panel: per-regime (g) or the
    all-regime average (g=None); windowed to [X_MIN_ENV, X_MAX_ENV]."""
    if g is None:
        curves = ent.get("curves") or {}
        ser = curves.get(sorted(curves)[0]) if curves else None
    else:
        ser = (ent.get("curves_g") or {}).get(g)
    if not ser:
        return None
    ser = [(x, y) for (x, y) in ser if X_MIN_ENV <= x <= X_MAX_ENV]
    return ser if len(ser) >= 3 else None


def _panel_anchor(ent: dict, g: int | None) -> float | None:
    """Measured 2M value to anchor a completed method's curve endpoint."""
    reports = ent.get("reports") or {}
    if not reports:
        return None
    rep = reports[sorted(reports)[0]]
    if g is None:
        return float(rep.get("return_mean", 0.0))
    d = rep.get("return_per_regime") or {}
    val = d.get(g, d.get(str(g)))
    return float(val) if val is not None else None


def _apply_anchor(xs: np.ndarray, ys: np.ndarray, anchor: float | None):
    """Pin a curve's 2M endpoint to the accurate 30-episode checkpoint eval
    (the in-training trace uses few episodes; the anchored endpoint is the
    same number the bars/box/table quote)."""
    if anchor is None:
        return xs, ys
    if xs[-1] >= X_MAX_ENV * 0.999:
        ys = np.append(ys[:-1], anchor)
    else:
        xs = np.append(xs, X_MAX_ENV)
        ys = np.append(ys, anchor)
    return xs, ys


def fig_per_regime(plt, data: dict, out: pathlib.Path, assets, t: dict) -> bool:
    """M2 global reward — 2×3 grid (last cell blank): four regime line panels
    (competitive g1 excluded — ≈0 for every method, see 2.2.1a) plus the
    all-regime average panel. The per-episode return distribution box plot at
    the 2M standard is a separate figure — see fig_per_regime_box below."""
    cell = data.get(GATE_CELL, {})
    if not cell:
        return False
    panel_defs: list[tuple[int | None, str]] = [
        (0, t["fig2_panel_titles"][0]), (2, t["fig2_panel_titles"][2]),
        (3, t["fig2_panel_titles"][3]), (4, t["fig2_panel_titles"][4]),
        (None, t["fig2_avg_title"]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.8))
    axes = axes.ravel()
    drew_any = False

    for ax, (g, title) in zip(axes[:5], panel_defs):
        for v, (_lbl, color) in METHODS.items():
            ent = cell.get(v) or {}
            ser = _panel_series(ent, g)
            if not ser:
                continue
            drew_any = True
            xs, ys = map(np.asarray, zip(*ser))
            # Completed method → anchor the endpoint to the measured 2M value.
            xs, ys = _apply_anchor(xs, ys, _panel_anchor(ent, g))
            ax.plot(xs, ys, color=color, linewidth=0.9, alpha=0.25)
            ys_s = _smooth(ys, PANEL_SMOOTH_K.get(v, 5))
            ax.plot(xs, ys_s, color=color, linewidth=1.8,
                    label=t["method_labels"][v])
            # In-progress (MAMBA) → real curve stops at its last actual
            # checkpoint (above); a single hollow marker — not a drawn-out
            # line — flags the fitted projection to the 2M budget point so
            # it can't be mistaken for measured data. Disclosed in caption.
            if not ent.get("reports") and ent.get("target"):
                key = "eval/return_mean" if g is None else f"eval/return_g{g}"
                fit = _fit_project((ent.get("accurate") or {}).get(key),
                                   ent["target"])
                if _fit_ok(fit):
                    ax.plot([ent["target"]], [fit["proj"]], marker="o",
                            linestyle="", mfc=t["surf"], mec=color,
                            mew=1.5, ms=6)
        ax.set_xscale("log")
        ax.set_xlim(X_MIN_ENV, X_MAX_ENV)
        _style(ax)
        ax.set_title(title, fontsize=9.5)
        ax.tick_params(labelsize=8)

    axes[5].set_axis_off()

    if not drew_any:
        plt.close(fig)
        return False
    # Shared axis labels + one legend (from the average panel's handles).
    for ax in (axes[0], axes[3]):
        ax.set_ylabel(t["fig2_ylabel"], fontsize=9)
    for ax in axes[3:5]:
        ax.set_xlabel(t["fig2_xlabel"], fontsize=9)
    handles, labels = axes[4].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, fontsize=9,
                   ncols=len(labels), loc="upper center",
                   bbox_to_anchor=(0.5, 1.02))
    if t["fig2_title"]:
        fig.suptitle(t["fig2_title"], fontsize=11, y=1.06)
    fig.tight_layout()
    _save(fig, out, assets, f"fig2_per_regime{t['suffix']}.png")
    plt.close(fig)
    return True


def fig_per_regime_box(plt, data: dict, out: pathlib.Path, assets, t: dict) -> bool:
    """Per-episode return distribution at the 2M standardized budget (30
    episodes/regime) — companion box plot to fig_per_regime, saved as its
    own figure (report appendix) rather than a subplot of the main one."""
    cell = data.get(GATE_CELL, {})
    if not cell:
        return False
    box_regimes = [0, 2, 3, 4]
    box_entries: list[tuple[str, str, dict, bool, int]] = []
    for v, (_lbl, color) in METHODS.items():
        ent = cell.get(v) or {}
        if ent.get("episodes"):
            box_entries.append((v, color, ent["episodes"], False, 0))
        elif ent.get("box_episodes"):
            box_entries.append((v, color, ent["box_episodes"], True,
                                int(ent.get("box_step", 0))))
    if not box_entries:
        return False

    fig, bax = plt.subplots(figsize=(5.5, 4.2))
    n_m = len(box_entries)
    width = 0.8 / n_m
    for j, (v, color, eps, in_prog, step) in enumerate(box_entries):
        pos = [i - 0.4 + width * (j + 0.5) for i in range(len(box_regimes))]
        vals = [eps.get(g, eps.get(str(g), [])) for g in box_regimes]
        label = (t["box_progress_fmt"].format(
                    label=t["method_labels"][v], step=f"{step / 1000:.0f}k")
                 if in_prog else t["method_labels"][v])
        bp = bax.boxplot(
            vals, positions=pos, widths=width * 0.85, patch_artist=True,
            showfliers=False, medianprops={"color": INK, "linewidth": 1.2},
            whiskerprops={"color": SEC, "linewidth": 0.9},
            capprops={"color": SEC, "linewidth": 0.9},
            boxprops={"linewidth": 0.9},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.55 if not in_prog else 0.35)
            patch.set_edgecolor(color)
            if in_prog:
                patch.set_hatch(PROJ_HATCH)
        bax.plot([], [], color=color, linewidth=5, alpha=0.6, label=label)
    _style(bax)
    bax.axhline(0, color=BASE, linewidth=0.8)
    bax.set_xticks(range(len(box_regimes)),
                   [f"g{g}" for g in box_regimes], fontsize=8)
    bax.set_title(t["fig2_box_title"], fontsize=9.5)
    bax.tick_params(labelsize=8)
    bax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    _save(fig, out, assets, f"fig2b_per_regime_box{t['suffix']}.png")
    plt.close(fig)
    return True


def fig_generalization(plt, data: dict, out: pathlib.Path, assets, t: dict) -> bool:
    cell = data.get(ZS_CELL, {})
    entries = []              # (v, s_mean, u_mean, s_sem, u_sem, n, is_proj)
    for v in METHODS:
        reports = (cell.get(v) or {}).get("reports") or {}
        if reports:
            seen = [r.get("return_zero_shot_seen", 0.0) for r in reports.values()]
            unseen = [r.get("return_zero_shot_unseen", 0.0) for r in reports.values()]
            entries.append((v, float(np.mean(seen)), float(np.mean(unseen)),
                            float(np.std(seen) / np.sqrt(len(seen))) if len(seen) > 1 else 0.0,
                            float(np.std(unseen) / np.sqrt(len(unseen))) if len(unseen) > 1 else 0.0,
                            len(seen), False))
            continue
        ps = _projected_summary(cell.get(v) or {})
        if ps and ps["seen"] is not None and ps["unseen"] is not None:
            entries.append((v, ps["seen"], ps["unseen"], 0.0, 0.0, 1, True))
    if not entries:
        return False
    edge = t["surf"]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    groups = t["fig3_groups"]
    x = np.arange(2)
    n_m = len(entries)
    width = 0.7 / n_m
    for j, (v, s_mean, u_mean, s_sem, u_sem, n, is_proj) in enumerate(entries):
        color = METHODS[v][1]
        label = (t["method_labels"][v] + t["proj_bar_tag"] if is_proj
                 else t["n_fmt"].format(label=t["method_labels"][v], n=n))
        offs = x - 0.35 + width * (j + 0.5)
        vals, errs = [s_mean, u_mean], [s_sem, u_sem]
        ax.bar(offs, vals, width=width * 0.94, color=color, edgecolor=edge,
               linewidth=1.0, hatch=(PROJ_HATCH if is_proj else None),
               yerr=(errs if any(errs) else None), ecolor=SEC,
               capsize=2, label=label)
        for xx, val, err in zip(offs, vals, errs):
            tip = val + err if val >= 0 else val - err
            ax.annotate(f"{val:.1f}", (xx, tip), textcoords="offset points",
                        xytext=(0, 2 if val >= 0 else -2), ha="center",
                        va="bottom" if val >= 0 else "top",
                        fontsize=7.5, color=INK)
    _style(ax)
    ax.axhline(0, color=BASE, linewidth=0.8)
    ax.set_xticks(x, groups, fontsize=9)
    ax.set_ylabel(t["fig3_ylabel"])
    if t["fig3_title"]:
        ax.set_title(t["fig3_title"])
    ymax = ax.get_ylim()[1]
    ax.set_ylim(top=ymax * 1.22)          # headroom so the legend clears bar labels
    ax.legend(frameon=False, fontsize=8.5, ncols=3, loc="upper center")
    if any(e[6] for e in entries):
        fig.text(0.5, -0.02, t["proj_note_bar"], ha="center", va="top",
                 fontsize=7.5, color=SEC, wrap=True)
    _save(fig, out, assets, f"fig3_generalization{t['suffix']}.png")
    plt.close(fig)
    return True


def fig_nashconv(plt, gm_dir: pathlib.Path, out: pathlib.Path, assets, t: dict) -> bool:
    stats = {}
    for v in METHODS:
        body = _load_json(str(gm_dir / f"game_metrics_{v}.json"))
        if body and body.get("nashconv"):
            stats[v] = {int(k): float(val) for k, val in body["nashconv"].items()}
    if not stats:
        return False
    edge = t["surf"]
    regimes = sorted(set().union(*[set(s) for s in stats.values()]))
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    x = np.arange(len(regimes))
    n_m = len(stats)
    width = 0.8 / n_m
    for j, (v, nc) in enumerate(stats.items()):
        label, color = t["method_labels"][v], METHODS[v][1]
        offs = x - 0.4 + width * (j + 0.5)
        vals = [nc.get(g, 0.0) for g in regimes]
        ax.bar(offs, vals, width=width * 0.94, color=color, edgecolor=edge,
               linewidth=1.0, label=label)
        for xx, val in zip(offs, vals):
            ax.annotate(f"{val:.1f}", (xx, val), ha="center", va="bottom",
                        fontsize=7.5, color=INK)
    _style(ax)
    ax.set_xticks(x, [t["regime_labels"].get(g, f"g{g}") for g in regimes], fontsize=8)
    ax.set_ylabel(t["fig4_ylabel"])
    if t["fig4_title"]:
        ax.set_title(t["fig4_title"])
    ax.legend(frameon=False, fontsize=9)
    _save(fig, out, assets, f"fig4_nashconv{t['suffix']}.png")
    plt.close(fig)
    return True


# ---------------------------------------------------------------- table

def _mf_reference(cdata: dict) -> float | None:
    """Model-free reference return (MAPPO) for the world-model fidelity gap."""
    reports = (cdata.get("external_mappo") or {}).get("reports") or {}
    if not reports:
        return None
    return float(np.mean([r.get("return_mean", 0.0) for r in reports.values()]))


def write_table(data: dict, gm_dir: pathlib.Path, out: pathlib.Path) -> None:
    lines = [
        "# Table 1 — baseline comparison summary (rel_duo, N=2, |G|=5)",
        "",
        "| Method | Cell | Seeds | Env-step budget | Return (mean±SEM) | "
        "Δ vs model-free | Seen regimes | Held-out regimes | "
        "Zero-shot gap | NashConv (mean, LB) | Walltime/seed |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    any_proj = False
    for cell in (GATE_CELL, ZS_CELL):
        cdata = data.get(cell, {})
        mf_ref = _mf_reference(cdata)
        for v, (label, _c) in METHODS.items():
            ent = cdata.get(v)
            if not ent:
                continue
            gm = _load_json(str(gm_dir / f"game_metrics_{v}.json")) or {}
            nc = gm.get("nashconv") or {}
            nc_mean = (f"{np.mean([float(x) for x in nc.values()]):.2f}"
                       if (nc and cell == GATE_CELL) else "—")
            model_based = v != "external_mappo"
            if ent.get("reports"):
                reports = list(ent["reports"].values())
                n = len(reports)
                rm = [r.get("return_mean", 0.0) for r in reports]
                zs_s = [r.get("return_zero_shot_seen", 0.0) for r in reports]
                zs_u = [r.get("return_zero_shot_unseen", 0.0) for r in reports]
                rows_done = [r for r in ent["rows"] if r.get("status") == "completed"]
                budget = max((_row_budget_env_steps(r) for r in rows_done), default=0)
                wall = [r.get("walltime_seconds") for r in rows_done
                        if r.get("walltime_seconds")]
                if ent.get("budget_override"):
                    budget = float(ent["budget_override"])   # 2M-checkpoint eval
                    wall = []                                # row walltime ≠ this budget
                sem = float(np.std(rm) / np.sqrt(n)) if n > 1 else 0.0
                sem_txt = f" ± {sem:.2f}" if n > 1 else " (single seed)"
                gap_mf = (f"{np.mean(rm) - mf_ref:+.2f}"
                          if (model_based and mf_ref is not None) else "0 (ref)")
                unseen_txt = f"{np.mean(zs_u):.2f}" if cell == ZS_CELL else "—"
                seen_txt = f"{np.mean(zs_s):.2f}" if cell == ZS_CELL else "—"
                zs_gap = (f"{np.mean(zs_u) - np.mean(zs_s):+.2f}"
                          if cell == ZS_CELL else "—")
                wall_txt = f"{np.mean(wall) / 3600:.1f} h" if wall else "—"
                lines.append(
                    f"| {label} | {cell} | {n} | {budget:,.0f} | "
                    f"{np.mean(rm):.2f}{sem_txt} | {gap_mf} | "
                    f"{seen_txt} | {unseen_txt} | {zs_gap} | "
                    f"{nc_mean} | {wall_txt} |"
                )
                continue
            # In-progress row: projected outcome at the full budget.
            ps = _projected_summary(ent)
            if not ps:
                continue
            any_proj = True
            gap_mf = (f"{ps['return_mean'] - mf_ref:+.2f} *"
                      if (model_based and mf_ref is not None) else "—")
            unseen_txt = (f"{ps['unseen']:.2f} *" if cell == ZS_CELL
                          and ps["unseen"] is not None else "—")
            seen_txt = (f"{ps['seen']:.2f} *" if cell == ZS_CELL
                        and ps["seen"] is not None else "—")
            zs_gap = (f"{ps['unseen'] - ps['seen']:+.2f} *"
                      if cell == ZS_CELL and ps["seen"] is not None
                      and ps["unseen"] is not None else "—")
            lines.append(
                f"| {label} | {cell} | 1 (proj.) | {ps['target']:,.0f} * | "
                f"{ps['return_mean']:.2f} * | {gap_mf} | {seen_txt} | {unseen_txt} | "
                f"{zs_gap} | {nc_mean} | in progress |"
            )
    lines += [
        "",
        "Notes: all values are 30 deterministic episodes/regime evaluated at each "
        "method's 2M-env-step checkpoint (one uniform protocol; `hyper` = "
        "step_20000.pt — 20k gradient steps × 100 env steps/step). "
        "`Δ vs model-free`: model-free methods interact with the real environment "
        "directly and carry no world-model error, so at a matched env-step budget "
        "in a deterministic training scenario their return is the natural upper "
        "reference; a model-based method's gap to it reflects how faithfully its "
        "world model simulates the environment (smaller |Δ| = higher fidelity). "
        "`Zero-shot gap` = unseen − seen: positive means training experience "
        "TRANSFERS to regimes never seen in training (knowledge transfer), "
        "negative means performance collapses off the training distribution — "
        "the primary generalization criterion here, ahead of absolute return. "
        "NashConv values are lower bounds (approximate DQN best response, budget "
        "in game_metrics JSON). Single-seed rows carry no error estimate.",
    ]
    if any_proj:
        lines.append(
            "`*` / `(proj.)` = fitted saturating-curve extrapolation of MAMBA's "
            "accurate 30-episode evals at intermediate checkpoints to the full 2M "
            "env-step budget — provisional, replaced by the measured eval_report "
            "once training completes."
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "table1.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[make_baseline_comparison] wrote {out / 'table1.md'}")


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", type=pathlib.Path, default=pathlib.Path("runs/suite"))
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("runs/_analysis/rel_defense"))
    ap.add_argument("--assets", type=pathlib.Path,
                    default=pathlib.Path("docs/assets/interim_report"))
    ap.add_argument("--game-metrics", type=pathlib.Path, default=None,
                    help="dir holding game_metrics_<variant>.json (default: --out)")
    ap.add_argument("--no-assets", action="store_true")
    ap.add_argument("--external-budget", type=float, default=2_000_000.0,
                    help="env-step budget selecting which external rows to include "
                         "(±1%%); 0 disables the filter (default: 2e6)")
    ap.add_argument("--mamba-series-dir", type=pathlib.Path, default=None,
                    help="dir with mamba_ckpt_series_<cell>.json (accurate "
                         "checkpoint evals; default: --out)")
    ap.add_argument("--mamba-freeze-env-step", type=float, default=None,
                    help="drop MAMBA checkpoint-series points past this env "
                         "step (that series updates continuously from a "
                         "still-running eval loop; freeze it to whatever step "
                         "the report text already quotes so every figure/table "
                         "from this run reflects one consistent MAMBA snapshot "
                         "instead of drifting mid-report as training advances)")
    ap.add_argument("--final-evals-dir", type=pathlib.Path,
                    default=pathlib.Path("runs/_analysis/rel_defense/final_evals_2m"),
                    help="collect_final_evals.py output (2M-standard "
                         "checkpoint evals + per-episode returns)")
    ap.add_argument("--lang", choices=["en", "zh"], default="en",
                    help="figure text language + style (zh = publication style "
                         "for the Chinese report: no titles, serif, 300 dpi)")
    args = ap.parse_args(argv)
    gm_dir = args.game_metrics or args.out
    assets = None if args.no_assets else args.assets
    t = I18N[args.lang]

    data = collect(args.runs_root, args.external_budget,
                   args.mamba_series_dir or args.out,
                   args.final_evals_dir,
                   args.mamba_freeze_env_step)
    plt = _mpl(args.lang)

    made = {
        "fig1": fig_sample_efficiency(plt, data, args.out, assets, t),
        "fig2": fig_per_regime(plt, data, args.out, assets, t),
        "fig2b": fig_per_regime_box(plt, data, args.out, assets, t),
        "fig3": fig_generalization(plt, data, args.out, assets, t),
        "fig4": fig_nashconv(plt, gm_dir, args.out, assets, t),
    }
    write_table(data, gm_dir, args.out)
    missing = [k for k, ok in made.items() if not ok]
    if missing:
        print(f"[make_baseline_comparison] skipped (no data yet): {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
