#!/usr/bin/env python
"""export_exp_result_0729.py — re-cut v5_final TensorBoard scalars into a CSV tree
for plotting, plus the three ablation tables.

Supersedes ``export_exp_result_0728.py`` with the 0729 brief:

  * **Comparison 1 — value-equivalent modeling error.** Arms
    ``mazero_mixed_ref_bc_anneal_scaled_hardval_decoupled_big``, ``mamba_pm``,
    ``m3w_adapted``; **every** ``fidelity/*`` tag, not just ``reward_mae``.
  * **Comparison 2 — reward.** Arms
    ``mazero_mixed_ref_bc_anneal_scaled_hardval_decoupled``, ``mbom_pm``,
    ``happo_pm``, ``mamba_pm``, ``m3w_adapted``; ``eval/return_mean`` +
    ``eval/return_regime_<g>``, plus ``fidelity/reward_mae`` where it exists.
  * **Ablation curves.** ``mazero_mixed_ref_bc_anneal_scaled_no_subjective_decoupled``,
    ``eval/return_mean`` only (the method arm's curve is exported alongside it so the
    ablation figure can be drawn from one directory).
  * **Tables 1-3** transposed to the requested shape: one column per algorithm /
    ablation variant, one row per (metric, seed) with a ``mean`` row closing each
    metric block.

Read-only against ``results/v5_final`` and CPU-only: event files, the
``eval_diagnostics*.json`` / ``fidelity.json`` artifacts and ``registry.jsonl`` are
opened for reading only. Nothing is written into ``v5_final``, no checkpoint is
re-evaluated, no GPU is touched -- safe to run while the grid is still training.
Idempotent: re-run to refresh as runs finish.

Usage:
    python scripts/export_exp_result_0729.py
    python scripts/export_exp_result_0729.py --root results/v5_final --out results/exp_result_0728
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import pathlib
import statistics
import sys

REPO_ROOT = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ------------------------------------------------------------------ config

MAIN_ARM = "ref_bc_anneal_scaled_hardval_decoupled"
NOSUBJ_ARM = "ref_bc_anneal_scaled_no_subjective_decoupled"

MAIN_VARIANT = f"mazero_mixed_{MAIN_ARM}"
NOSUBJ_VARIANT = f"mazero_mixed_{NOSUBJ_ARM}"
BIG_VARIANT = f"{MAIN_VARIANT}_big"

# Comparison 1 (value-equivalent modeling error): fidelity only, all tags.
# NOTE: the brief says "mamba"; the only mamba arm in v5_final is `mamba_pm`
# (the periodic-metrics relaunch), so that is what is exported.
EXP1_ALGOS = [BIG_VARIANT, "mamba_pm", "m3w_adapted"]

# Comparison 2 (reward), in report order. Seeds are discovered on disk.
EXP2_ALGOS = [MAIN_VARIANT, "mbom_pm", "happo_pm", "mamba_pm", "m3w_adapted"]

# Ablation curve arms (eval/return_mean only, per the brief).
ABLATION_CURVE_ALGOS = [MAIN_VARIANT, NOSUBJ_VARIANT]

TAG_MEAN = "eval/return_mean"
TAG_FIDELITY = "fidelity/reward_mae"
TAG_REGIME = [f"eval/return_regime_{g}" for g in range(5)]
# Every fidelity tag written by hyper_mve (see algo/runner.py::_log_fidelity).
FIDELITY_TAGS = [TAG_FIDELITY] + [f"fidelity/reward_mae_regime_{g}" for g in range(5)]

ENV = "relation"

# `relation` runs the g2 family (N=2), utils/schemas/relation.py::build_g2.
# g1 (mutual_comp) is zero-sum: team return is ~0 there for ANY policy, so it is
# always the "worst" regime and carries no signal about a method's quality.
REGIME_NAMES = ["mutual_coop", "mutual_comp", "asym_exploit", "asym_exploited", "neutral"]

T1_DIR = "超网络条件世界模型消融"
T2_DIR = "规划模块消融"
T3_DIR = "全消融"

# Ablation variant columns: (key prefix in eval_diagnostics, source arm, label).
A_COLS = [
    ("A1", MAIN_VARIANT, "return_per_regime_planner", "A1 Bayes-avg (ours)"),
    ("A2", MAIN_VARIANT, "return_per_regime_planner_map", "A2 argmax / MAP head"),
    ("A3", MAIN_VARIANT, "return_per_regime_prior", "A3 no MCTS (distilled prior)"),
    ("A4", NOSUBJ_VARIANT, "return_per_regime_planner", "A4 plain MAZero, planner"),
    ("A5", NOSUBJ_VARIANT, "return_per_regime_prior", "A5 plain MAZero + no MCTS (prior)"),
    ("UB", MAIN_VARIANT, "return_per_regime_planner_oracle", "UB oracle (true g at deploy)"),
]
T2_COLS = ["A1", "A2", "A3"]
T3_COLS = ["A1", "A2", "A3", "A4", "A5", "UB"]


# ------------------------------------------------------------ TB extraction


def _read_all_series(tb_dir: pathlib.Path, tags):
    """``{tag: [(wall_time, step, value)]}`` for ``tags``, restart-purged and deduped.

    One ``EventAccumulator`` per tb dir, loaded once for all requested tags and
    dropped on return -- reloading per tag makes the export minutes-long on the
    big logs (mamba_pm writes ~426K progress events), and caching every run's
    scalars at once would hold the whole grid in RAM.

    TB dirs here are not clean monotonic logs: a crashed-and-relaunched run leaves
    several event files in one dir (steps restart at 0), and the final
    registry-protocol eval is written at the *same* step as the last periodic eval.
    So per tag:

      1. order by wall_time;
      2. on a STRICT step decrease, drop everything accumulated so far (the
         orphaned pre-restart segment).  Strict ``<`` matters -- ``<=`` would purge
         the whole history on the same-step final-eval duplicate;
      3. keep the last value written at each step (the final eval wins over the
         periodic one, reproducing registry.jsonl exactly);
      4. return sorted by step.
    """
    out = {t: [] for t in tags}
    if not tb_dir.is_dir():
        return out

    from hyper_mve.utils.analysis.tb_scraper import _event_accumulator

    acc = _event_accumulator()(str(tb_dir), size_guidance={"scalars": 0})
    acc.Reload()
    available = set(acc.Tags().get("scalars", []))
    for tag in tags:
        if tag not in available:
            continue
        events = sorted(acc.Scalars(tag), key=lambda e: e.wall_time)
        kept: list = []
        last_step = None
        for ev in events:
            if last_step is not None and ev.step < last_step:
                kept = []  # restart: everything before this point is orphaned
            kept.append(ev)
            last_step = ev.step
        dedup: dict[int, tuple[float, float]] = {}
        for ev in kept:
            dedup[int(ev.step)] = (float(ev.wall_time), float(ev.value))
        out[tag] = [(w, s, v) for s, (w, v) in sorted(dedup.items())]
    del acc
    return out


def _write_csv(path: pathlib.Path, rows, header=("Wall time", "Step", "Value")):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# --------------------------------------------------------------- discovery


def _seed_dirs(root: pathlib.Path, algo: str) -> dict[int, pathlib.Path]:
    base = root / algo / ENV
    if not base.is_dir():
        return {}
    out = {}
    for d in sorted(base.iterdir()):
        if d.is_dir() and d.name.startswith("seed") and d.name[4:].isdigit():
            out[int(d.name[4:])] = d
    return out


def _load_registry(root: pathlib.Path) -> dict[tuple[str, int], dict]:
    path = root / "registry.jsonl"
    reg: dict[tuple[str, int], dict] = {}
    if not path.exists():
        return reg
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        reg[(d.get("variant"), int(d.get("seed", -1)))] = d
    return reg


def _load_diag(run_dir: pathlib.Path):
    """Diagnostics for a run, preferring the reeval file when present.

    Runs trained before the oracle deploy pass landed have no
    ``return_per_regime_planner_oracle`` in their original eval_diagnostics.json;
    reeval_checkpoint.py backfills it under the same deterministic protocol.
    """
    for name in ("eval_diagnostics_reeval.json", "eval_diagnostics.json"):
        p = run_dir / name
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            d["_source"] = name
            return d
    return None


# ------------------------------------------------------------ table helpers


def _fmt(v, nd=3):
    return "" if v is None else f"{float(v):.{nd}f}"


def _mean_of(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _table_to_md(title, note_lines, header, rows):
    out = [f"# {title}", ""]
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in r) + " |")
    if note_lines:
        out.append("")
        for n in note_lines:
            out.append(f"- {n}")
    return "\n".join(out) + "\n"


def _write_table(dir_path: pathlib.Path, stem: str, title, notes, header, rows):
    dir_path.mkdir(parents=True, exist_ok=True)
    with (dir_path / f"{stem}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows([["" if c is None else c for c in r] for r in rows])
    (dir_path / f"{stem}.md").write_text(
        _table_to_md(title, notes, header, rows), encoding="utf-8")


def _metric_block(metric_label, seeds, per_col_values, marks=None):
    """Rows for one metric: seed0..seedN then `mean`, one column per algorithm.

    ``per_col_values`` is ``[[v_seed0, v_seed1, ...], ...]``, one list per column.
    ``marks`` (same shape) appends a suffix such as ``*`` to flag partial runs; a
    ``mean`` built from any marked cell inherits the mark.
    """
    rows = []
    for i, s in enumerate(seeds):
        cells = []
        for c, vals in enumerate(per_col_values):
            v = vals[i] if i < len(vals) else None
            mk = (marks[c][i] if marks and i < len(marks[c]) else "") if v is not None else ""
            cells.append(_fmt(v) + mk)
        rows.append([metric_label, f"seed{s}"] + cells)
    cells = []
    for c, vals in enumerate(per_col_values):
        agg = _mean_of(vals)
        mk = ""
        if marks and agg is not None and any(
                marks[c][i] and vals[i] is not None for i in range(min(len(vals), len(marks[c])))):
            mk = "*"
        cells.append(_fmt(agg) + mk)
    rows.append([metric_label, "mean"] + cells)
    return rows


# ------------------------------------------------------------------- main


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="results/v5_final")
    p.add_argument("--out", default="results/exp_result_0728")
    args = p.parse_args(argv)

    root = (REPO_ROOT / args.root) if not os.path.isabs(args.root) else pathlib.Path(args.root)
    out = (REPO_ROOT / args.out) if not os.path.isabs(args.out) else pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    registry = _load_registry(root)
    manifest: list[dict] = []
    problems: list[str] = []
    # (algo, seed) -> {value, step, status, n} for eval/return_mean
    finals: dict[tuple[str, int], dict] = {}
    # (algo, seed) -> last fidelity/reward_mae from TB
    fid_finals: dict[tuple[str, int], float] = {}

    def _status(algo, seed):
        reg = registry.get((algo, seed))
        return (reg or {}).get("status") or "running/incomplete"

    def _emit(rel, series, algo, seed, metric, header=("Wall time", "Step", "Value")):
        path = out / rel
        _write_csv(path, series, header)
        manifest.append({
            "experiment": rel.split("/")[0],
            "algo": algo, "seed": seed, "metric": metric,
            "csv_path": str(path.relative_to(out)),
            "n_points": len(series),
            "last_step": series[-1][1] if series else "",
            "last_value": f"{series[-1][2]:.6f}" if series else "",
            "run_status": _status(algo, seed),
        })

    # ------------------------------- comparison 1: value-equivalent modeling error
    for algo in EXP1_ALGOS:
        seeds = _seed_dirs(root, algo)
        if not seeds:
            problems.append(f"exp1: no seed dirs for {algo}")
            continue
        for seed, run_dir in sorted(seeds.items()):
            series = _read_all_series(run_dir / "tb", FIDELITY_TAGS)
            got = [t for t in FIDELITY_TAGS if series[t]]
            if not got:
                problems.append(f"exp1 {algo}/seed{seed}: no fidelity/* tags")
                continue
            for tag in got:
                sub = tag.split("/", 1)[1]           # reward_mae[_regime_<g>]
                _emit(f"comparison1_fidelity/{sub}/{algo}/seed{seed}.csv",
                      series[tag], algo, seed, tag)
            missing = [t for t in FIDELITY_TAGS if not series[t]]
            if missing:
                problems.append(
                    f"exp1 {algo}/seed{seed}: missing {', '.join(missing)}")

    # ---------------------------------------------- comparison 2: reward + fidelity
    for algo in EXP2_ALGOS:
        seeds = _seed_dirs(root, algo)
        if not seeds:
            problems.append(f"exp2: no seed dirs for {algo}")
            continue
        for seed, run_dir in sorted(seeds.items()):
            reg = registry.get((algo, seed))
            status = _status(algo, seed)
            series = _read_all_series(run_dir / "tb",
                                      [TAG_MEAN] + TAG_REGIME + [TAG_FIDELITY])

            mean_s = series[TAG_MEAN]
            if not mean_s:
                problems.append(f"exp2 {algo}/seed{seed}: no {TAG_MEAN}")
                continue

            # Registry cross-check: for a completed run the last exported point must
            # be the final registry-protocol eval. RELATIVE tolerance because TB
            # stores scalars as float32 while the registry keeps float64 -- at a
            # return of ~65 that quantization is ~3e-6 absolute, i.e. exact
            # agreement, not a mismatch.
            if reg and reg.get("status") == "completed" and reg.get("return_mean") is not None:
                expected = float(reg["return_mean"])
                if not math.isclose(expected, mean_s[-1][2], rel_tol=1e-6, abs_tol=1e-6):
                    problems.append(
                        f"REGISTRY MISMATCH {algo}/seed{seed}: "
                        f"tb={mean_s[-1][2]!r} registry={expected!r} "
                        f"(d={abs(expected - mean_s[-1][2]):g})")

            finals[(algo, seed)] = {
                "value": mean_s[-1][2], "step": mean_s[-1][1], "status": status,
                "n": len(mean_s),
            }
            _emit(f"comparison/reward/mean/{algo}/seed{seed}.csv",
                  mean_s, algo, seed, TAG_MEAN)

            steps_mean = [s for _, s, _ in mean_s]
            regime_series = []
            for g, tag in enumerate(TAG_REGIME):
                s = series[tag]
                if not s:
                    problems.append(f"exp2 {algo}/seed{seed}: no {tag}")
                    regime_series.append(None)
                    continue
                if [st for _, st, _ in s] != steps_mean:
                    # A live run can flush a regime scalar a moment before (or after)
                    # its return_mean partner, so a still-training run's tags can
                    # differ by the last point. Not an error: spread is computed over
                    # the steps common to all six series.
                    problems.append(
                        f"note {algo}/seed{seed}: {tag} has {len(s)} points vs "
                        f"{len(mean_s)} for {TAG_MEAN} (live-write skew); spread uses "
                        f"the common steps")
                regime_series.append(s)
                _emit(f"comparison/reward/regime{g}/{algo}/seed{seed}.csv",
                      s, algo, seed, tag)

            fid_s = series[TAG_FIDELITY]
            if fid_s:
                _emit(f"comparison/fidelity/{algo}/seed{seed}.csv",
                      fid_s, algo, seed, TAG_FIDELITY)
                fid_finals[(algo, seed)] = fid_s[-1][2]
            else:
                manifest.append({
                    "experiment": "comparison", "algo": algo, "seed": seed,
                    "metric": TAG_FIDELITY, "csv_path": "", "n_points": 0,
                    "last_step": "", "last_value": "",
                    "run_status": f"{status}; N/A (no world-model reward head)",
                })

            # derived: per-step cross-regime spread, over the steps present in all
            # five regime series (plus return_mean, for the wall-time column)
            if all(s is not None for s in regime_series):
                by_step = [{st: v for _, st, v in s} for s in regime_series]
                wall = {st: w for w, st, _ in mean_s}
                common = sorted(set(wall).intersection(*[set(d) for d in by_step]))
                rows = []
                for st in common:
                    vals = [by_step[g][st] for g in range(5)]
                    rows.append([
                        f"{wall[st]:.6f}", st,
                        f"{statistics.fmean(vals):.6f}",
                        f"{statistics.pstdev(vals):.6f}",
                        f"{min(vals):.6f}", f"{max(vals):.6f}",
                    ])
                path = out / f"comparison/spread/{algo}/seed{seed}.csv"
                _write_csv(path, rows,
                           ("Wall time", "Step", "Value", "std", "min", "max"))
                manifest.append({
                    "experiment": "comparison", "algo": algo, "seed": seed,
                    "metric": "derived/regime_spread",
                    "csv_path": str(path.relative_to(out)),
                    "n_points": len(rows),
                    "last_step": rows[-1][1] if rows else "",
                    "last_value": rows[-1][3] if rows else "",   # std
                    "run_status": status,
                })

    # ------------------------------------------- ablation curves (return_mean only)
    t1_dir = out / "ablation" / T1_DIR
    for algo in ABLATION_CURVE_ALGOS:
        for seed, run_dir in sorted(_seed_dirs(root, algo).items()):
            series = _read_all_series(run_dir / "tb", [TAG_MEAN] + TAG_REGIME)
            s_mean = series[TAG_MEAN]
            if not s_mean:
                problems.append(f"ablation {algo}/seed{seed}: no {TAG_MEAN}")
                continue
            _emit(f"ablation/{T1_DIR}/curves/{algo}/seed{seed}.csv",
                  s_mean, algo, seed, f"{TAG_MEAN} (ablation curve)")
            finals.setdefault((algo, seed), {
                "value": s_mean[-1][2], "step": s_mean[-1][1],
                "status": _status(algo, seed), "n": len(s_mean)})
            # spread at the final common step, for Table 1's generalization rows
            regs = [series[t] for t in TAG_REGIME]
            if all(regs):
                by_step = [{st: v for _, st, v in s} for s in regs]
                common = sorted(set(by_step[0]).intersection(*[set(d) for d in by_step[1:]]))
                if common:
                    vals = [d[common[-1]] for d in by_step]
                    sd = statistics.pstdev(vals)
                    mu = statistics.fmean(vals)
                    finals[(algo, seed)]["spread"] = sd
                    # Scale-free companion: a raw std grows with the return level, so
                    # it "rewards" a uniformly weaker model. std/|mean| does not.
                    finals[(algo, seed)]["rel_spread"] = sd / abs(mu) if mu else None
                    finals[(algo, seed)]["worst_regime"] = min(vals)

    # -------------------------------------------------------------- Table 1
    t1_algos = [MAIN_VARIANT, NOSUBJ_VARIANT]
    t1_labels = [f"{MAIN_VARIANT} (ours)",
                 f"{NOSUBJ_VARIANT} (control, Module-1 removed)"]
    t1_seeds = sorted({s for a in t1_algos for s in _seed_dirs(root, a)})

    def _fid_final(algo, seed):
        """Final fidelity for a run: fidelity.json (registry protocol) if present,
        else the last TB point. Returns None when the arm cannot produce it."""
        f = root / algo / ENV / f"seed{seed}" / "fidelity.json"
        if f.exists():
            try:
                return float(json.loads(f.read_text(encoding="utf-8"))["reward_mae"])
            except (KeyError, ValueError):
                pass
        return fid_finals.get((algo, seed))

    marks_by_algo = [[("*" if finals.get((a, s), {}).get("status") != "completed"
                       and (a, s) in finals else "") for s in t1_seeds] for a in t1_algos]

    header1 = ["metric", "seed"] + t1_labels
    rows1 = []
    rows1 += _metric_block(
        "final reward (eval/return_mean)", t1_seeds,
        [[finals.get((a, s), {}).get("value") for s in t1_seeds] for a in t1_algos],
        marks_by_algo)
    rows1 += _metric_block(
        "fidelity (reward MAE, lower=better)", t1_seeds,
        [[_fid_final(a, s) for s in t1_seeds] for a in t1_algos], marks_by_algo)
    rows1 += _metric_block(
        "generalization: cross-regime std (scale-dependent, read with the row below)",
        t1_seeds,
        [[finals.get((a, s), {}).get("spread") for s in t1_seeds] for a in t1_algos],
        marks_by_algo)
    rows1 += _metric_block(
        "generalization: relative spread std/|mean| (lower=better)", t1_seeds,
        [[finals.get((a, s), {}).get("rel_spread") for s in t1_seeds] for a in t1_algos],
        marks_by_algo)
    rows1 += _metric_block(
        "generalization: worst regime (always g1, zero-sum — see note)", t1_seeds,
        [[finals.get((a, s), {}).get("worst_regime") for s in t1_seeds] for a in t1_algos],
        marks_by_algo)

    incomplete1 = [f"{a}/seed{s} @ step {finals[(a, s)]['step']:,}"
                   for a in t1_algos for s in t1_seeds
                   if (a, s) in finals and finals[(a, s)]["status"] != "completed"]
    missing1 = [f"{a}/seed{s}" for a in t1_algos for s in t1_seeds if (a, s) not in finals]
    notes1 = [
        "Columns are algorithms; each metric occupies four rows — `seed0`, `seed1`, `seed2`, "
        "`mean` (mean over the seeds that exist).",
        "**Reward** is the last point of the deduped TensorBoard `eval/return_mean` series. For a "
        "**completed** run that point is the final registry-protocol eval (16 episodes/regime, "
        "`planner_full`) and matches `registry.jsonl`; for an **incomplete** run it is a periodic "
        "in-training eval — noisier, and at a smaller budget.",
        "**Fidelity** is one-step reward MAE from the run's `fidelity.json` (registry protocol, "
        "80 episodes / 8000 transitions). The control cell is **empty by construction, not "
        "unmeasured**: removing Module 1 removes `belief_net`, so "
        "`predict_rewards_from_model` returns `None` "
        "(hyper_mve/algo/mazero_mixed/core/test.py) and no world-model reward prediction exists "
        "to score. The `mean` row for the control is therefore blank as well.",
        "**Generalization** is *not* `eval/return_seen`: in the `relation` env all five regimes "
        "are marked seen, so `return_seen` is byte-identical to `return_mean` and "
        "`return_unseen` is 0. The reported cross-regime measures are computed from the same TB "
        "series as the reward row, at the final common step.",
        "**Do not read the raw std row on its own.** It is scale-dependent: a model that scores "
        "higher on every regime except the zero-sum one necessarily has a larger absolute spread, "
        "so the raw std makes the *stronger* arm look worse. `std/|mean|` is the scale-free "
        "comparison; both rows are given so the effect is visible rather than hidden.",
        "**The worst-regime row is uninformative here.** `relation` runs the g2 family "
        f"({', '.join(f'g{i}={n}' for i, n in enumerate(REGIME_NAMES))}); g1 is zero-sum, so team "
        "return is ~0 there for any policy and g1 is the minimum for every arm. The row is kept "
        "for completeness, not as a discriminating metric.",
        "Blank cells are runs that do not exist; they are excluded from `mean`.",
    ]
    if incomplete1:
        notes1.append(
            "`*` marks a cell read off a run that was **still training** at export time — a "
            "periodic in-training eval at a smaller budget, not a converged 1M result. A `mean` "
            "carrying `*` mixes budgets and must not be quoted as a converged number. "
            "Still training: " + "; ".join(incomplete1) + ".")
    if missing1:
        notes1.append("Not launched: " + ", ".join(missing1) + ".")
    _write_table(t1_dir, "table1",
                 f"Table 1 — {T1_DIR} (hypernetwork-conditioned world model)",
                 notes1, header1, rows1)

    # ------------------------------------------------- Tables 2 & 3 (diagnostics)
    # Per seed, per ablation column: the mean over regimes, and the per-regime values.
    ab_seeds = sorted({s for a in (MAIN_VARIANT, NOSUBJ_VARIANT) for s in _seed_dirs(root, a)})
    diag_cache: dict[tuple[str, int], dict] = {}
    for algo in (MAIN_VARIANT, NOSUBJ_VARIANT):
        for seed, run_dir in sorted(_seed_dirs(root, algo).items()):
            d = _load_diag(run_dir)
            if d is None:
                problems.append(f"ablation {algo}/seed{seed}: no eval_diagnostics*.json")
            else:
                diag_cache[(algo, seed)] = d

    def _cell(col_key, seed, regime=None):
        """Ablation cell: mean over regimes (regime=None) or one regime's value."""
        spec = next(c for c in A_COLS if c[0] == col_key)
        _, algo, field, _label = spec
        d = diag_cache.get((algo, seed))
        if not d:
            return None
        per = d.get(field)
        if not per:
            return None
        per = {int(g): float(v) for g, v in per.items()}
        if regime is None:
            return statistics.fmean(per.values())
        return per.get(regime)

    diag_src = sorted({d.get("_source") for d in diag_cache.values() if d.get("_source")})

    def _build(col_keys):
        labels = {k: lbl for k, _a, _f, lbl in A_COLS}
        header = ["metric", "seed"] + [labels[k] for k in col_keys]
        rows = _metric_block("return (mean over regimes)", ab_seeds,
                             [[_cell(k, s) for s in ab_seeds] for k in col_keys])
        for g in range(5):
            rows += _metric_block(f"return regime {g} ({REGIME_NAMES[g]})", ab_seeds,
                                  [[_cell(k, s, g) for s in ab_seeds] for k in col_keys])
        return header, rows

    common_notes = [
        "Columns are ablation variants; each metric occupies four rows — `seed0`, `seed1`, "
        "`seed2`, `mean`.",
        "Source: each checkpoint's `eval_diagnostics*.json` (registry eval protocol, "
        "16 episodes/regime). Files used: "
        f"{', '.join(diag_src) if diag_src else 'n/a'} "
        "(`eval_diagnostics_reeval.json` supersedes `eval_diagnostics.json` where present — it "
        "backfills the oracle pass on runs trained before it landed; A1/A2/A3 are unchanged).",
        "Every cell is the corresponding `return_per_regime_*` field: the `return (mean over "
        "regimes)` block averages the five regimes, the `return regime g` blocks give them "
        "individually. A1 therefore equals the headline `return_mean` for the same seed — the "
        "bridge between these tables and Table 1.",
        f"A1/A2/A3/UB come from `{MAIN_VARIANT}`; A4/A5 from the matched Module-1 control "
        f"`{NOSUBJ_VARIANT}`.",
        "Regimes are the g2 family: "
        + ", ".join(f"g{i} = {n}" for i, n in enumerate(REGIME_NAMES))
        + ". **g1 is zero-sum** — team return is ~0 there for any policy, so its row separates "
        "nothing and should not be averaged into a claim.",
        "Blank cells are runs that do not exist; they are excluded from `mean`.",
    ]

    def _delta_note(a, b):
        d = {s: (_cell(a, s) - _cell(b, s)) for s in ab_seeds
             if _cell(a, s) is not None and _cell(b, s) is not None}
        if not d:
            return None
        return (f"{a} − {b} per seed: "
                + ", ".join(f"seed{s} {v:+.2f}" for s, v in sorted(d.items()))
                + f" (mean {_mean_of(list(d.values())):+.2f}).")

    h2, r2 = _build(T2_COLS)
    notes2 = common_notes + [
        "Read: A1 − A2 = value of Bayes-averaging over hard MAP-head selection; "
        "A1 − A3 = value of search over the distilled prior.",
    ]
    notes2 += [n for n in (_delta_note("A1", "A2"), _delta_note("A1", "A3")) if n]
    _write_table(out / "ablation" / T2_DIR, "table2", f"Table 2 — {T2_DIR} (planning module)",
                 notes2, h2, r2)

    h3, r3 = _build(T3_COLS)
    notes3 = common_notes + [
        "**UB oracle is a privileged upper bound, not a method result**: it uses the true regime "
        "id `g` at deploy time. Read UB − A1 as the headroom lost to imperfect regime inference — "
        "UB ≈ A1 means belief accuracy is not the limiter (the value heads are); UB ≫ A1 means "
        "regime inference is.",
        "Read: A1 − A3 = value of search; A1 − A4 = value of Module 1 (the hypernetwork-"
        "conditioned world model); A4 − A5 = value of search without Module 1.",
    ]
    notes3 += [n for n in (_delta_note("UB", "A1"), _delta_note("A1", "A2"),
                           _delta_note("A1", "A3"), _delta_note("A1", "A4"),
                           _delta_note("A4", "A5")) if n]
    t3_dir = out / "ablation" / T3_DIR
    _write_table(t3_dir, "table3", f"Table 3 — {T3_DIR} (full ablation)", notes3, h3, r3)
    (t3_dir / "tables.json").write_text(json.dumps(
        {str(s): {k: _cell(k, s) for k in T3_COLS} for s in ab_seeds},
        indent=2, ensure_ascii=False), encoding="utf-8")

    # -------------------------------------------------------- manifest + README
    man_cols = ["experiment", "algo", "seed", "metric", "csv_path", "n_points",
                "last_step", "last_value", "run_status"]
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=man_cols)
        w.writeheader()
        w.writerows(manifest)

    status_lines = []
    for algo in dict.fromkeys(EXP1_ALGOS + EXP2_ALGOS + [NOSUBJ_VARIANT]):
        for seed in sorted(_seed_dirs(root, algo)):
            f = finals.get((algo, seed))
            if f:
                status_lines.append(
                    f"| `{algo}` | {seed} | {f['status']} | {f['step']:,} | {f['n']} | "
                    f"{f['value']:.3f} |")
            else:   # exp1-only arm: no reward series exported, report the registry row
                reg = registry.get((algo, seed)) or {}
                rm = reg.get("return_mean")
                st = reg.get("status") or "running/incomplete"
                status_lines.append(
                    f"| `{algo}` | {seed} | {st} | — | — | "
                    f"{'—' if rm is None else format(float(rm), '.3f')} |")

    (out / "README.md").write_text(f"""# exp_result_0728 — plotting data export

Exported {datetime.datetime.now():%Y-%m-%d %H:%M} from `{args.root}` by
`scripts/export_exp_result_0729.py`. **Snapshot of a live grid** — several runs were still
training; re-run the script to refresh.

## Layout

```
comparison1_fidelity/<tag>/<algo>/seed<i>.csv   Comparison 1: every fidelity/* tag
    tag ∈ reward_mae, reward_mae_regime_0..4
    algo ∈ {', '.join(EXP1_ALGOS)}

comparison/reward/mean/<algo>/seed<i>.csv       Comparison 2: eval/return_mean
comparison/reward/regime<g>/<algo>/seed<i>.csv  Comparison 2: eval/return_regime_<g> (g=0..4)
comparison/fidelity/<algo>/seed<i>.csv          Comparison 2: fidelity/reward_mae
comparison/spread/<algo>/seed<i>.csv            derived: mean/std/min/max over g0..g4
    algo ∈ {', '.join(EXP2_ALGOS)}

ablation/{T1_DIR}/curves/<algo>/seed<i>.csv    eval/return_mean
ablation/{T1_DIR}/table1.{{csv,md}}
ablation/{T2_DIR}/table2.{{csv,md}}
ablation/{T3_DIR}/table3.{{csv,md}} + tables.json
manifest.csv                                    one row per exported series
```

CSV header is `Wall time,Step,Value` — the native TensorBoard download schema, so
`scripts/plot_eval_result.py::_read_csv` parses these unchanged. `spread/` appends
`,std,min,max` after `Value` (there `Value` is the across-regime mean).

## Read this before plotting

1. **The x-axis is environment steps** for every series and every algorithm. No per-method
   rescaling is needed here — unlike the older `results/eval_result` CSVs, which mixed gradient
   steps and environment steps. Probe cadence differs by algorithm (200 steps for the mazero
   arms, 4000 for mamba/happo/mbom), so series have very different point counts on the same axis.
2. **Comparison 1 uses `{BIG_VARIANT}`** — the capacity-matched
   arm — while Comparison 2's reward figure uses `{MAIN_VARIANT}`.
   They are different checkpoints with different returns; do not read a fidelity number and a
   reward number off the same row of the two figures.
3. **`mamba` is exported as `mamba_pm`.** That is the only mamba arm present in `v5_final`.
4. **Final points are not all the same protocol.** For a *completed* run the last point is the
   final registry-protocol eval (16 episodes/regime, `planner_full`) and matches
   `registry.jsonl`. For an *incomplete* run it is a periodic in-training eval — noisier. Check
   `run_status` in `manifest.csv` before quoting a final number.
5. **`fidelity/*` does not exist for every arm.** `mbom_pm` and `happo_pm` are model-free (no
   world-model reward head) and the `no_subjective` arms have no `belief_net`, so
   `predict_rewards_from_model` returns `None` by construction. Those rows appear in
   `manifest.csv` with an empty `csv_path` and an `N/A` note.
6. **`eval/return_seen` was not exported**: in the `relation` env all five regimes are marked
   seen, so it is byte-identical to `eval/return_mean` (`return_unseen` = 0). Use
   `comparison/spread/` for the cross-regime generalization signal.
7. **Tables 2 and 3 use a different source from Table 1.** They come from each checkpoint's
   `eval_diagnostics*.json` (the registry eval protocol); Table 1's reward row reads the TB
   curves. A1 is the bridge and must equal Table 1's reward for the same seed.
8. Restarted runs (extra event files in one `tb/` dir) were restart-purged: on a strict step
   decrease everything before it is dropped, then the last value per step wins.

## Run status at export time

| algo | seed | status | last step | eval points | last `eval/return_mean` |
|---|---|---|---|---|---|
{chr(10).join(status_lines)}
""", encoding="utf-8")

    # ------------------------------------------------------------------ report
    n_csv = sum(1 for _ in out.rglob("*.csv")) - 1   # minus manifest.csv
    print(f"exported {len(manifest)} manifest rows; {n_csv} data CSVs")
    print(f"tables: {T1_DIR}/table1, {T2_DIR}/table2, {T3_DIR}/table3")
    notes = [p for p in problems if p.startswith("note ")]
    errs = [p for p in problems if not p.startswith("note ")]
    if errs:
        print(f"\n{len(errs)} issue(s):")
        for pr in errs:
            print(f"  - {pr}")
    else:
        print("no issues: registry cross-check passed for every completed run")
    if notes:
        print(f"\n{len(notes)} note(s) (expected on live runs, not errors):")
        for n in notes:
            print(f"  - {n[5:]}")
    return 1 if any(p.startswith("REGISTRY MISMATCH") for p in problems) else 0


if __name__ == "__main__":
    sys.exit(main())
