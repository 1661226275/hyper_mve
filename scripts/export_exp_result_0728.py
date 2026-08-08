#!/usr/bin/env python
"""export_exp_result_0728.py — re-cut v5_final TensorBoard scalars into a flat CSV
tree for plotting, plus the three ablation tables.

Read-only against ``results/v5_final`` and CPU-only: event files, the
``eval_diagnostics*.json`` artifacts and ``registry.jsonl`` are opened for
reading only. Nothing is written into ``v5_final``, no checkpoint is
re-evaluated, no GPU is touched -- so this is safe to run while the grid is
still training. Idempotent: re-run to refresh as runs finish.

Output layout (``results/exp_result_0728``)::

    comparison/reward/mean/<algo>/seed<i>.csv        eval/return_mean
    comparison/reward/regime<g>/<algo>/seed<i>.csv   eval/return_regime_<g>
    comparison/fidelity/<algo>/seed<i>.csv           fidelity/reward_mae
    comparison/generalization/<algo>/seed<i>.csv     eval/return_seen
    comparison/spread/<algo>/seed<i>.csv             derived std/min/max over g0..g4
    ablation/<name>/{curves,table*.csv,table*.md}
    manifest.csv, README.md

CSV header is ``Wall time,Step,Value`` -- the native TensorBoard UI download
schema, so scripts/plot_eval_result.py::_read_csv parses these unchanged.

Usage:
    python scripts/export_exp_result_0728.py
    python scripts/export_exp_result_0728.py --root results/v5_final --out results/exp_result_0728
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
import subprocess
import sys
import tempfile

REPO_ROOT = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ------------------------------------------------------------------ config

MAIN_ARM = "ref_bc_anneal_scaled_hardval_decoupled"
NOSUBJ_ARM = "ref_bc_anneal_scaled_no_subjective_decoupled"

MAIN_VARIANT = f"mazero_mixed_{MAIN_ARM}"
NOSUBJ_VARIANT = f"mazero_mixed_{NOSUBJ_ARM}"

# Comparison arms, in report order. Seeds are discovered on disk, not assumed.
COMPARISON_ALGOS = [
    MAIN_VARIANT,
    "mbom_pm",
    "happo_pm",
    "mamba_pm",
    "m3w_adapted",
]

# Ablation curve arms (eval/return_mean only, per the brief).
ABLATION_CURVE_ALGOS = [MAIN_VARIANT, NOSUBJ_VARIANT]

TAG_MEAN = "eval/return_mean"
TAG_SEEN = "eval/return_seen"
TAG_FIDELITY = "fidelity/reward_mae"
TAG_REGIME = [f"eval/return_regime_{g}" for g in range(5)]

ENV = "relation"

T1_DIR = "超网络条件世界模型消融"
T2_DIR = "规划模块消融"
T3_DIR = "全消融"

# Row labels for the ablation tables (assemble_ablation.py key -> label).
T2_ROWS = [
    ("A1_bayes_mean", "A1 Bayes-avg (ours)"),
    ("A2_argmax_mean", "A2 argmax / MAP head"),
    ("A3_prior_mean", "A3 no MCTS (distilled prior)"),
]
T3_ROWS = T2_ROWS + [
    ("A4_plainMAZero_mean", "A4 plain MAZero, planner"),
    ("A5_prior_mean", "A5 plain MAZero + no MCTS (prior)"),
    ("UB_oracle_mean", "UB oracle (true g at deploy) [not a method result]"),
]


# ------------------------------------------------------------ TB extraction


_ACC_CACHE: dict[str, object] = {}


def _accumulator(tb_dir: pathlib.Path):
    """One ``EventAccumulator`` per tb dir, reloaded once and reused.

    Each run is queried for ~8 tags; reloading the event files per tag makes the
    export minutes-long on the big logs (mamba_pm writes ~426K progress events).
    """
    from hyper_mve.utils.analysis.tb_scraper import _event_accumulator

    key = str(tb_dir)
    if key not in _ACC_CACHE:
        acc = _event_accumulator()(key, size_guidance={"scalars": 0})
        acc.Reload()
        _ACC_CACHE[key] = acc
    return _ACC_CACHE[key]


def _read_series(tb_dir: pathlib.Path, tag: str) -> list[tuple[float, int, float]]:
    """``[(wall_time, step, value)]`` for one tag, restart-purged and deduped.

    TB dirs here are not clean monotonic logs: a crashed-and-relaunched run
    leaves several event files in one dir (steps restart at 0), and the final
    registry-protocol eval is written at the *same* step as the last periodic
    eval. So:

      1. order by wall_time;
      2. on a STRICT step decrease, drop everything accumulated so far (the
         orphaned pre-restart segment).  Strict ``<`` matters -- ``<=`` would
         purge the whole history on the same-step final-eval duplicate;
      3. keep the last value written at each step (the final eval wins over the
         periodic one, reproducing registry.jsonl exactly);
      4. return sorted by step.
    """
    if not tb_dir.is_dir():
        return []
    acc = _accumulator(tb_dir)
    if tag not in acc.Tags().get("scalars", []):
        return []

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
    return [(w, s, v) for s, (w, v) in sorted(dedup.items())]


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


# ------------------------------------------------------------ table helpers


def _fmt(v, nd=3):
    return "" if v is None else f"{float(v):.{nd}f}"


def _mean_of(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _seed_cols(seeds):
    return [f"seed{s}" for s in seeds] + ["mean"]


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
    # (algo, seed) -> last eval/return_mean value + last step + status
    finals: dict[tuple[str, int], dict] = {}

    # ------------------------------------------------ comparison CSV export
    for algo in COMPARISON_ALGOS:
        seeds = _seed_dirs(root, algo)
        if not seeds:
            problems.append(f"no seed dirs found for {algo}")
            continue
        for seed, run_dir in sorted(seeds.items()):
            tb = run_dir / "tb"
            reg = registry.get((algo, seed))
            status = (reg or {}).get("status") or "running/incomplete"

            mean_s = _read_series(tb, TAG_MEAN)
            if not mean_s:
                problems.append(f"{algo}/seed{seed}: no {TAG_MEAN}")
                continue

            # Registry cross-check: for a completed run the last exported point
            # must be the final registry-protocol eval. Compared with a RELATIVE
            # tolerance because TB stores scalars as float32 while the registry
            # keeps float64 -- at a return of ~65 that quantization is ~3e-6
            # absolute, which is exact agreement, not a mismatch.
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

            def emit(rel: str, series, metric: str, header=("Wall time", "Step", "Value")):
                path = out / rel
                _write_csv(path, series, header)
                manifest.append({
                    "algo": algo, "seed": seed, "metric": metric,
                    "csv_path": str(path.relative_to(out)),
                    "n_points": len(series),
                    "last_step": series[-1][1] if series else "",
                    "last_value": f"{series[-1][2]:.6f}" if series else "",
                    "run_status": status,
                })

            emit(f"comparison/reward/mean/{algo}/seed{seed}.csv", mean_s, TAG_MEAN)

            # per-regime, and the derived spread (needs aligned step columns)
            steps_mean = [s for _, s, _ in mean_s]
            regime_series = []
            for g, tag in enumerate(TAG_REGIME):
                s = _read_series(tb, tag)
                if not s:
                    problems.append(f"{algo}/seed{seed}: no {tag}")
                    regime_series.append(None)
                    continue
                if [st for _, st, _ in s] != steps_mean:
                    # A live run can flush a regime scalar a moment before (or
                    # after) its return_mean partner, so a still-training run's
                    # tags can differ by the last point. Not an error: spread is
                    # computed over the steps common to all six series.
                    problems.append(
                        f"note {algo}/seed{seed}: {tag} has {len(s)} points vs "
                        f"{len(mean_s)} for {TAG_MEAN} (live-write skew); spread "
                        f"uses the common steps")
                regime_series.append(s)
                emit(f"comparison/reward/regime{g}/{algo}/seed{seed}.csv", s, tag)

            seen_s = _read_series(tb, TAG_SEEN)
            if seen_s:
                emit(f"comparison/generalization/{algo}/seed{seed}.csv", seen_s, TAG_SEEN)
            else:
                problems.append(f"{algo}/seed{seed}: no {TAG_SEEN}")

            fid_s = _read_series(tb, TAG_FIDELITY)
            if fid_s:
                emit(f"comparison/fidelity/{algo}/seed{seed}.csv", fid_s, TAG_FIDELITY)
            else:
                manifest.append({
                    "algo": algo, "seed": seed, "metric": TAG_FIDELITY,
                    "csv_path": "", "n_points": 0, "last_step": "", "last_value": "",
                    "run_status": f"{status}; N/A (no world-model reward head)",
                })

            # derived: per-step spread across the five regimes, over the steps
            # present in all five (plus return_mean, for the wall-time column)
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
                    "algo": algo, "seed": seed, "metric": "derived/regime_spread",
                    "csv_path": str(path.relative_to(out)),
                    "n_points": len(rows),
                    "last_step": rows[-1][1] if rows else "",
                    "last_value": rows[-1][3] if rows else "",   # std
                    "run_status": status,
                })

    # ------------------------------------------- ablation curves (return_mean)
    t1_dir = out / "ablation" / T1_DIR
    for algo in ABLATION_CURVE_ALGOS:
        for seed, run_dir in sorted(_seed_dirs(root, algo).items()):
            s = _read_series(run_dir / "tb", TAG_MEAN)
            if not s:
                continue
            reg = registry.get((algo, seed))
            status = (reg or {}).get("status") or "running/incomplete"
            path = t1_dir / "curves" / algo / f"seed{seed}.csv"
            _write_csv(path, s)
            manifest.append({
                "algo": algo, "seed": seed, "metric": f"{TAG_MEAN} (ablation curve)",
                "csv_path": str(path.relative_to(out)), "n_points": len(s),
                "last_step": s[-1][1], "last_value": f"{s[-1][2]:.6f}",
                "run_status": status,
            })
            finals.setdefault((algo, seed), {
                "value": s[-1][2], "step": s[-1][1], "status": status, "n": len(s)})

    # ------------------------------------------------------------- Table 1
    t1_algos = [MAIN_VARIANT, NOSUBJ_VARIANT]
    t1_seeds = sorted({s for a in t1_algos for s in _seed_dirs(root, a)})

    def _spread_last(algo, seed):
        f = out / f"comparison/spread/{algo}/seed{seed}.csv"
        if not f.exists():
            # ablation control is not in COMPARISON_ALGOS -> compute on the fly
            tb = root / algo / ENV / f"seed{seed}" / "tb"
            regs = [_read_series(tb, t) for t in TAG_REGIME]
            if not all(regs):
                return None
            by_step = [{st: v for _, st, v in s} for s in regs]
            common = sorted(set(by_step[0]).intersection(*[set(d) for d in by_step[1:]]))
            if not common:
                return None
            return statistics.pstdev([d[common[-1]] for d in by_step])
        with f.open(encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        return float(rows[-1][3]) if len(rows) > 1 else None

    header1 = ["variant"]
    for m in ("reward", "generalization", "spread"):
        header1 += [f"{m}_{c}" for c in _seed_cols(t1_seeds)]
    rows1 = []
    # A cell read off a still-training run is a periodic in-training eval at a
    # smaller budget, not a final 1M result -- mark it so nobody quotes a mixed
    # mean as a converged number.
    partial = [(algo, s) for algo in t1_algos for s in t1_seeds
               if (algo, s) in finals and finals[(algo, s)]["status"] != "completed"]
    for algo in t1_algos:
        label = f"{algo} (ours)" if algo == MAIN_VARIANT else f"{algo} (control)"
        row = [label]
        rew = [finals.get((algo, s), {}).get("value") for s in t1_seeds]
        spr = [_spread_last(algo, s) if (algo, s) in finals else None for s in t1_seeds]
        marks = ["*" if (algo, s) in partial else "" for s in t1_seeds]
        for vals in (rew, rew, spr):   # generalization == reward (return_seen ≡ return_mean)
            cells = [_fmt(v) + (m if v is not None else "") for v, m in zip(vals, marks)]
            agg = _fmt(_mean_of(vals))
            if agg and any(marks[i] and vals[i] is not None for i in range(len(vals))):
                agg += "*"
            row += cells + [agg]
        rows1.append(row)

    incomplete1 = [f"{a}/seed{s} @ step {finals[(a, s)]['step']:,}"
                   for a in t1_algos for s in t1_seeds
                   if (a, s) in finals and finals[(a, s)]["status"] != "completed"]
    missing1 = [f"{a}/seed{s}" for a in t1_algos for s in t1_seeds if (a, s) not in finals]
    notes1 = [
        "Source: last point of the deduped TensorBoard `eval/return_mean` series. For a "
        "**completed** run that point is the final registry-protocol eval (16 episodes/regime, "
        "`planner_full`); for an **incomplete** run it is a periodic in-training eval — noisier.",
        "`generalization` = `eval/return_seen`, which is byte-identical to `eval/return_mean` in "
        "this env (`relation` marks all 5 regimes seen, so `return_unseen` = 0). The column is "
        "kept for completeness; `spread` (population std across g0..g4 at the final step) is the "
        "informative cross-regime measure.",
        "**Fidelity is intentionally absent.** `predict_rewards_from_model` returns `None` when "
        "the model has no `belief_net` (hyper_mve/algo/mazero_mixed/core/test.py:27), so the "
        "no_subjective control cannot produce `fidelity/reward_mae` for any seed — the comparison "
        "would be undefined, not merely unmeasured.",
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
    _write_table(t1_dir, "table1", f"Table 1 — {T1_DIR} (hypernetwork-conditioned world model)",
                 notes1, header1, rows1)

    # -------------------------------------------- Tables 2 & 3 (diagnostics)
    with tempfile.TemporaryDirectory() as td:
        assembled_path = pathlib.Path(td) / "ablation_assembled.json"
        cmd = [sys.executable, str(REPO_ROOT / "scripts" / "assemble_ablation.py"),
               "--root", str(root), "--main-arm", MAIN_ARM, "--nosubj-arm", NOSUBJ_ARM,
               "--out", str(assembled_path)]
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
        if proc.returncode != 0:
            problems.append("assemble_ablation.py failed:\n" + proc.stdout + proc.stderr)
            assembled = {}
        else:
            assembled = json.loads(assembled_path.read_text(encoding="utf-8"))
    assembled = {int(k): v for k, v in assembled.items()}
    ab_seeds = sorted(assembled) or [0, 1, 2]

    def _build(rows_spec):
        header = ["setup"] + _seed_cols(ab_seeds)
        rows = []
        for key, label in rows_spec:
            vals = [assembled.get(s, {}).get(key) for s in ab_seeds]
            rows.append([label] + [_fmt(v) for v in vals] + [_fmt(_mean_of(vals))])
        return header, rows

    diag_src = sorted({v.get("diag_source") for v in assembled.values() if v.get("diag_source")})
    common_notes = [
        "Source: each checkpoint's `eval_diagnostics*.json` (registry eval protocol, "
        "16 episodes/regime), assembled by `scripts/assemble_ablation.py --main-arm "
        f"{MAIN_ARM} --nosubj-arm {NOSUBJ_ARM}`. Files used: "
        f"{', '.join(diag_src) if diag_src else 'n/a'} "
        "(`eval_diagnostics_reeval.json` supersedes `eval_diagnostics.json` where present — it "
        "backfills the oracle pass on runs trained before it landed; A1/A2/A3 are unchanged).",
        "Each cell is the mean over the five regimes of the corresponding "
        "`return_per_regime_*` field. A1 therefore equals the headline `return_mean` for the "
        "same seed — the bridge between these tables and Table 1.",
        "Blank cells are runs that do not exist; they are excluded from `mean`.",
    ]
    h2, r2 = _build(T2_ROWS)
    _write_table(out / "ablation" / T2_DIR, "table2", f"Table 2 — {T2_DIR} (planning module)",
                 common_notes + [
                     "A1 − A2 = value of Bayes-averaging over hard MAP head selection; "
                     "A1 − A3 = value of search over the distilled prior.",
                 ], h2, r2)

    h3, r3 = _build(T3_ROWS)
    notes3 = common_notes + [
        f"A4/A5 come from the matched Module-1 control `{NOSUBJ_VARIANT}`, which has only "
        "seed0 completed (seed1 still training, seed2 not launched).",
        "**UB oracle is a privileged upper bound, not a method result**: it uses the true regime "
        "id `g` at deploy time. Read UB − A1 as the headroom lost to imperfect regime inference — "
        "UB ≈ A1 means belief accuracy is not the limiter (the value heads are); UB ≫ A1 means "
        "regime inference is.",
        "Read: A1 − A3 = value of search; A1 − A4 = value of Module 1 (the hypernetwork-"
        "conditioned world model); A4 − A5 = value of search without Module 1.",
    ]
    ub = {s: assembled.get(s, {}).get("UB_minus_A1") for s in ab_seeds}
    ub = {s: v for s, v in ub.items() if v is not None}
    if ub:
        notes3.append("UB − A1 per seed: "
                      + ", ".join(f"seed{s} {v:+.2f}" for s, v in sorted(ub.items()))
                      + f" (mean {_mean_of(list(ub.values())):+.2f}).")
    hd = {s: assembled.get(s, {}).get("head_diversity_back_half") for s in ab_seeds}
    hd = {s: v for s, v in hd.items() if v is not None}
    if hd:
        notes3.append("`train/head_diversity` back-half (mechanism-active check on the selected "
                      "method): " + ", ".join(f"seed{s} {v:.3f}" for s, v in sorted(hd.items()))
                      + ".")
    t3_dir = out / "ablation" / T3_DIR
    _write_table(t3_dir, "table3", f"Table 3 — {T3_DIR} (full ablation)", notes3, h3, r3)
    (t3_dir / "tables.json").write_text(
        json.dumps({str(k): v for k, v in assembled.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    # ------------------------------------------------------ manifest + README
    man_cols = ["algo", "seed", "metric", "csv_path", "n_points", "last_step",
                "last_value", "run_status"]
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=man_cols)
        w.writeheader()
        w.writerows(manifest)

    status_lines = []
    for algo in COMPARISON_ALGOS + [NOSUBJ_VARIANT]:
        for seed in sorted(_seed_dirs(root, algo)):
            f = finals.get((algo, seed))
            if not f:
                continue
            status_lines.append(
                f"| `{algo}` | {seed} | {f['status']} | {f['step']:,} | {f['n']} | "
                f"{f['value']:.3f} |")

    (out / "README.md").write_text(f"""# exp_result_0728 — plotting data export

Exported {datetime.datetime.now():%Y-%m-%d %H:%M} from `{args.root}` by
`scripts/export_exp_result_0728.py`. **Snapshot of a live grid** — several runs were still
training; re-run the script to refresh.

## Layout

```
comparison/reward/mean/<algo>/seed<i>.csv        eval/return_mean
comparison/reward/regime<g>/<algo>/seed<i>.csv   eval/return_regime_<g>   (g = 0..4)
comparison/fidelity/<algo>/seed<i>.csv           fidelity/reward_mae
comparison/generalization/<algo>/seed<i>.csv     eval/return_seen
comparison/spread/<algo>/seed<i>.csv             derived: mean/std/min/max over g0..g4
ablation/{T1_DIR}/    curves/ + table1.{{csv,md}}
ablation/{T2_DIR}/                table2.{{csv,md}}
ablation/{T3_DIR}/                    table3.{{csv,md}} + tables.json
manifest.csv                                     one row per exported series
```

CSV header is `Wall time,Step,Value` — the native TensorBoard download schema, so
`scripts/plot_eval_result.py::_read_csv` parses these unchanged. `spread/` appends
`,std,min,max` after `Value` (there `Value` is the across-regime mean).

## Read this before plotting

1. **The x-axis is environment steps** for every series and every algorithm. No per-method
   rescaling is needed here — unlike the older `results/eval_result` CSVs, which mixed gradient
   steps and environment steps.
2. **Final points are not all the same protocol.** For a *completed* run the last point is the
   final registry-protocol eval (16 episodes/regime, `planner_full`) and matches
   `registry.jsonl`. For an *incomplete* run it is a periodic in-training eval — noisier. Check
   `run_status` in `manifest.csv` before quoting a final number.
3. **`generalization/` is a duplicate of `reward/mean/`.** `eval/return_seen` is byte-identical
   to `eval/return_mean` in every run, because the `relation` env marks all five regimes as seen
   (`return_unseen` = 0). Use `spread/` for the actual cross-regime generalization signal.
4. **`fidelity/reward_mae` does not exist for every arm.** `mbom_pm` and `happo_pm` are model-free
   (no world-model reward head), and the `no_subjective` arms have no `belief_net`, so
   `predict_rewards_from_model` returns `None` by construction. Those rows appear in
   `manifest.csv` with an empty `csv_path` and an `N/A` note.
5. **Tables 2 and 3 use a different source from Table 1.** They come from each checkpoint's
   `eval_diagnostics*.json` (the registry eval protocol); Table 1 reads the TB curves. A1 is the
   bridge and must equal Table 1's reward for the same seed.
6. Restarted runs (extra event files in one `tb/` dir) were restart-purged: on a strict step
   decrease everything before it is dropped, then the last value per step wins.

## Run status at export time

| algo | seed | status | last step | eval points | last `eval/return_mean` |
|---|---|---|---|---|---|
{chr(10).join(status_lines)}
""", encoding="utf-8")

    # ------------------------------------------------------------- report
    n_csv = sum(1 for _ in (out / "comparison").rglob("*.csv"))
    print(f"exported {len(manifest)} manifest rows; {n_csv} CSVs under comparison/")
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
