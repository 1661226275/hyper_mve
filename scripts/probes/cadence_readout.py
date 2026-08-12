#!/usr/bin/env python
"""Compare `--env-steps-per-grad` cells on the robust last-20% statistic.

The endpoint of a single run is not a usable selection basis here: late in
training these runs oscillate with sd ~8-12 (`late_training_instability.md`), so
a one-point comparison mostly measures where the oscillation happened to be. The
selection statistic is therefore the mean of the run's own periodic
`eval/return_mean` over the final 20% of training, reported with its sd and
point count.

Return is not the only thing that has to hold up. A cadence that wins on return
while collapsing the belief posterior to chance (0.200 under a 5-regime family)
would undercut the role-aware claim regardless of score -- that is exactly how
the centralized cell was rejected in `stageA_selection_2x2.md`. So
`train/head_diversity` and the final report's `regime_accuracy` are printed
beside the return and must be read together with it.

Usage::

    python scripts/probes/cadence_readout.py results_v7_cadence_r16 \
        results_v7_cadence_r4 results_v7_cadence_r2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys


def _load_curve(tb_dir: str, tag: str):
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    ea = EventAccumulator(tb_dir)
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return []
    return [(int(e.step), float(e.value)) for e in ea.Scalars(tag)]


def _last_fraction(curve, frac: float):
    """Points in the final `frac` of the run, measured on the x-axis."""
    if not curve:
        return []
    xs = [x for x, _ in curve]
    cutoff = max(xs) - frac * (max(xs) - min(xs))
    return [(x, v) for x, v in curve if x >= cutoff]


def _summarize(run_dir: str, frac: float) -> dict:
    tb = os.path.join(run_dir, "tb")
    ret = _load_curve(tb, "eval/return_mean")
    div = _load_curve(tb, "train/head_diversity")
    tail = _last_fraction(ret, frac)
    vals = [v for _, v in tail]
    div_tail = [v for _, v in _last_fraction(div, frac)]

    meta_p = os.path.join(run_dir, "meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    rep_p = os.path.join(run_dir, "eval_report.json")
    report = json.load(open(rep_p)) if os.path.exists(rep_p) else {}

    # meta.json is only written when the run finishes, so fall back to the
    # `..._r<N>` root name to keep the table readable (and ordered) mid-flight.
    ratio = meta.get("env_steps_per_grad")
    if not ratio:
        tail = run_dir.split(os.sep)[0].rsplit("_r", 1)
        ratio = int(tail[1]) if len(tail) == 2 and tail[1].isdigit() else "?"

    return {
        "run": run_dir,
        "ratio": ratio,
        "env_steps": meta.get("env_steps_logged"),
        "train_steps": meta.get("train_steps_logged"),
        "walltime_h": round((meta.get("train_walltime_s") or 0) / 3600, 2),
        "points_total": len(ret),
        "points_tail": len(vals),
        "robust": round(statistics.fmean(vals), 2) if vals else None,
        "sd": round(statistics.stdev(vals), 2) if len(vals) > 1 else None,
        "endpoint": round(ret[-1][1], 2) if ret else None,
        "head_diversity": (round(statistics.fmean(div_tail), 2)
                           if div_tail else None),
        "regime_accuracy": report.get("regime_accuracy"),
        "complete": bool(report),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("roots", nargs="+", help="one --out root per cell")
    p.add_argument("--frac", type=float, default=0.2,
                   help="tail fraction for the robust statistic (default 0.2)")
    args = p.parse_args(argv)

    rows = []
    for root in args.roots:
        hits = sorted(glob.glob(os.path.join(root, "*", "*", "seed*")))
        if not hits:
            print(f"  (no run under {root})", file=sys.stderr)
            continue
        for run_dir in hits:
            rows.append(_summarize(run_dir, args.frac))

    # One root per cadence cell was the original use, and there the ratio column
    # identified the row on its own. Pointed at a whole wave root instead, every
    # row shares a ratio and the table is unreadable without the run's name --
    # so the name is printed, trimmed of the shared root and the trailing seed.
    def _label(run_dir: str) -> str:
        parts = run_dir.split(os.sep)
        return "/".join(parts[1:]) if len(parts) > 1 else run_dir

    width = max([28] + [len(_label(r["run"])) for r in rows])
    hdr = (f"{'run':<{width}} {'ratio':>5} {'robust':>8} {'sd':>7} "
           f"{'endpoint':>9} {'pts':>5} {'div':>7} {'reg_acc':>8} "
           f"{'steps':>8} {'walltime':>9}  done")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda d: (-(d["ratio"] if isinstance(d["ratio"], int) else 0),
                                         -(d["robust"] or 0))):
        acc = r["regime_accuracy"]
        print(f"{_label(r['run']):<{width}} {str(r['ratio']):>5} "
              f"{str(r['robust']):>8} {str(r['sd']):>7} "
              f"{str(r['endpoint']):>9} {r['points_tail']:>2}/{r['points_total']:<2} "
              f"{str(r['head_diversity']):>7} "
              f"{(round(acc, 3) if isinstance(acc, float) else acc)!s:>8} "
              f"{str(r['env_steps']):>8} {r['walltime_h']:>8}h  "
              f"{'yes' if r['complete'] else 'RUNNING'}")

    print("\nSelection rule: robust (last-20% mean of eval/return_mean), NOT the")
    print("endpoint -- within-run late oscillation has sd ~8-12. Reject any cell")
    print("whose regime_accuracy sits at chance (0.200) however good its return.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
