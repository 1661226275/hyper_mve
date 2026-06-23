"""analyze_results.py — generic results analysis for the experiment suite.

Three modes (mutually exclusive):

    # 1. method comparison from a sweep cell's registry (Welch-t + Holm-Bonferroni
    #    table + bar/errorbar plot + per-variant CSVs the `compare` CLI can reuse)
    python hyper_mve/scripts/analyze_results.py --compare \
        --registry runs/fast_300k/registry.jsonl --metric return_mean \
        --reference hyper --out runs/_analysis

    # 2. TB-only runs (e.g. runs/lora_sweep): tabulate the last value of a scalar
    python hyper_mve/scripts/analyze_results.py --tb \
        --tb-root runs/lora_sweep --tag eval/planner/return_total --out runs/_analysis

    # 3. external-runner disclosure table from a registry
    python hyper_mve/scripts/analyze_results.py --disclose \
        --registry runs/fast_300k/registry.jsonl --preset medium --out runs/_analysis

Reuses ``experiments.stats`` (Welch-t / Holm-Bonferroni / disclosure),
``experiments.compare.render_plot`` (Agg bar chart), and
``experiments.analysis.{registry_io, tb_scraper}``. Built FIRST so it can be
validated against the *existing* ``runs/fast_300k`` and ``runs/lora_sweep``
before any new suite cell runs.
"""
from __future__ import annotations

import argparse
import math
import os
import pathlib
import sys
from typing import Sequence

# Allow `python hyper_mve/scripts/analyze_results.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hyper_mve.experiments import stats  # noqa: E402
from hyper_mve.experiments.analysis import registry_io, tb_scraper  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python hyper_mve/scripts/analyze_results.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--compare", action="store_true",
                      help="Welch-t method comparison from a registry.")
    mode.add_argument("--tb", action="store_true",
                      help="Tabulate a TB scalar across TB-only runs.")
    mode.add_argument("--disclose", action="store_true",
                      help="External-runner disclosure table from a registry.")
    # --compare / --disclose
    p.add_argument("--registry", type=pathlib.Path, default=None,
                   help="Path to a cell's registry.jsonl (compare/disclose).")
    p.add_argument("--metric", default="return_mean",
                   help="EvalReport / registry scalar field to compare (default return_mean).")
    p.add_argument("--reference", default="hyper",
                   help="Reference method for pairwise tests (default hyper); "
                        "use '' / 'none' for all-pairs.")
    p.add_argument("--preset", choices=("easy", "medium", "hard"), default=None,
                   help="Restrict disclosure table to one preset.")
    # --tb
    p.add_argument("--tb-root", dest="tb_root", type=pathlib.Path, default=None,
                   help="Root dir holding TB-only runs (e.g. runs/lora_sweep).")
    p.add_argument("--tag", default="eval/planner/return_total",
                   help="Scalar tag to tabulate (default eval/planner/return_total).")
    # common
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("runs/_analysis"),
                   help="Output dir for markdown / png / csv (default runs/_analysis).")
    p.add_argument("--no-plot", dest="no_plot", action="store_true",
                   help="Skip the matplotlib bar chart (compare mode).")
    return p.parse_args(argv)


def _fallback_mean_table(samples: dict[str, list[float]], metric: str) -> str:
    """Per-variant mean ± sem table for when Welch-t is undefined (n<2 or k<2)."""
    lines = [f"[compare: insufficient seeds for Welch-t; mean ± sem only | metric={metric}]", ""]
    lines.append("| Method | n | mean | sem |")
    lines.append("|---|---|---|---|")
    for name in sorted(samples, key=lambda k: -(sum(samples[k]) / len(samples[k]) if samples[k] else 0)):
        xs = samples[name]
        n = len(xs)
        mean = sum(xs) / n if n else float("nan")
        if n > 1:
            var = sum((x - mean) ** 2 for x in xs) / (n - 1)
            sem = math.sqrt(var) / math.sqrt(n)
        else:
            sem = float("nan")
        lines.append(f"| {name} | {n} | {mean:.4f} | {sem:.4f} |")
    return "\n".join(lines)


def _run_compare(args: argparse.Namespace) -> int:
    if args.registry is None:
        raise SystemExit("--compare needs --registry <path>")
    rows = registry_io.load_completed_rows(args.registry)
    if not rows:
        raise SystemExit(f"[compare] no completed rows in {args.registry}")
    samples = registry_io.method_samples(rows, args.metric)
    samples = {k: v for k, v in samples.items() if v}
    if not samples:
        raise SystemExit(f"[compare] metric {args.metric!r} not found in any completed row")

    args.out.mkdir(parents=True, exist_ok=True)
    csv_paths = registry_io.to_compare_csv(rows, args.metric, args.out)
    print(f"[compare] wrote {len(csv_paths)} per-variant CSV(s) to {args.out}")

    reference = args.reference if args.reference and args.reference.lower() != "none" else None
    if reference is not None and reference not in samples:
        print(f"[compare] reference {reference!r} absent; falling back to all-pairs")
        reference = None

    md_path = args.out / f"compare_{args.metric}.md"
    enough = len(samples) >= 2 and all(len(v) >= 2 for v in samples.values())
    if not enough:
        md = _fallback_mean_table(samples, args.metric)
        md_path.write_text(md + "\n", encoding="utf-8")
        print(md)
        return 0

    result = stats.compare_methods(samples, reference=reference, alpha=args.alpha)
    md = result.render_markdown()
    if not args.no_plot:
        try:
            from hyper_mve.experiments.compare import render_plot
            plot_path = render_plot(result, args.out / f"compare_{args.metric}.png", metric=args.metric)
            md += f"\n\n![compare_{args.metric}](./{plot_path.name})\n"
        except Exception as e:  # matplotlib missing / headless issue
            print(f"[compare] plot skipped: {type(e).__name__}: {e}", file=sys.stderr)
    md_path.write_text(md + "\n", encoding="utf-8")
    print(md)
    print(f"\n[compare] wrote {md_path}")
    return 0


def _run_tb(args: argparse.Namespace) -> int:
    if args.tb_root is None:
        raise SystemExit("--tb needs --tb-root <dir>")
    table = tb_scraper.scrape_runs_root(args.tb_root, args.tag)
    if not table:
        raise SystemExit(f"[tb] tag {args.tag!r} not found under {args.tb_root}")
    args.out.mkdir(parents=True, exist_ok=True)
    lines = [f"# TB scalar `{args.tag}` under {args.tb_root}", "", "| run | last_step | value |", "|---|---|---|"]
    for name in sorted(table, key=lambda k: -table[k][1]):
        step, val = table[name]
        lines.append(f"| {name} | {step} | {val:.4f} |")
    md = "\n".join(lines)
    safe_tag = args.tag.replace("/", "_")
    out_md = args.out / f"tb_{safe_tag}.md"
    out_md.write_text(md + "\n", encoding="utf-8")
    # sidecar CSV
    import csv
    out_csv = args.out / f"tb_{safe_tag}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "last_step", args.tag])
        for name in sorted(table):
            step, val = table[name]
            w.writerow([name, step, val])
    print(md)
    print(f"\n[tb] wrote {out_md} + {out_csv}")
    return 0


def _run_disclose(args: argparse.Namespace) -> int:
    if args.registry is None:
        raise SystemExit("--disclose needs --registry <path>")
    rows = registry_io.load_completed_rows(args.registry)
    md = stats.render_disclosure_table(rows, preset=args.preset)
    args.out.mkdir(parents=True, exist_ok=True)
    out_md = args.out / "disclosure.md"
    out_md.write_text(md, encoding="utf-8")
    print(md)
    print(f"\n[disclose] wrote {out_md}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.compare:
        return _run_compare(args)
    if args.tb:
        return _run_tb(args)
    if args.disclose:
        return _run_disclose(args)
    raise SystemExit("no mode selected")  # argparse required-group guards this


if __name__ == "__main__":
    raise SystemExit(main())
