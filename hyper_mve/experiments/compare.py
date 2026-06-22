"""pkg-08 spec 07 §6 / §8 — `compare` CLI: markdown table + bar+errorbar plot + disclosure.

**Lock 3** — matplotlib Agg backend ONLY. ``matplotlib.use("Agg")`` is the
first non-comment line after the module docstring; ``import matplotlib.pyplot
as plt`` follows. ``plt.show()`` is forbidden anywhere in this file.

Three CLI modes (mutually exclusive):

* ``--a CSV --b CSV`` — 2-method comparison (k=2, no correction)
* ``--methods a.csv,b.csv,...`` plus optional ``--reference NAME`` — k>=3
* ``--disclose --registry runs/registry.jsonl`` — 10-column external table
"""
from __future__ import annotations

# Lock 3: backend MUST be set before any pyplot import.
import matplotlib  # noqa: E402
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import argparse
import json
import pathlib
import sys
from typing import Any, Sequence

from .stats import (
    ComparisonResult,
    DISCLOSURE_COLUMNS,
    compare_methods,
    render_disclosure_table,
)


__all__ = [
    "parse_args",
    "main",
    "render_plot",
    "load_registry",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m hyper_mve.experiments.compare",
        description=(
            "Pairwise / multi-method comparison + bar+errorbar plot.\n"
            "Three modes:\n"
            "  1. 2-method:    --a runs/<a>.csv --b runs/<b>.csv\n"
            "  2. multi-method: --methods <a.csv>,<b.csv>,... [--reference NAME]\n"
            "  3. disclosure:  --disclose --registry runs/registry.jsonl [--preset NAME]"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Mode 1
    p.add_argument("--a", type=pathlib.Path, default=None,
                   help="CSV path for method A (mode 1)")
    p.add_argument("--b", type=pathlib.Path, default=None,
                   help="CSV path for method B (mode 1)")
    # Mode 2
    p.add_argument("--methods", type=str, default=None,
                   help="Comma-separated list of CSV paths (mode 2)")
    p.add_argument("--reference", type=str, default=None,
                   help="Method name to use as reference for pairwise tests")
    # Mode 3
    p.add_argument("--disclose", action="store_true",
                   help="Emit external-runner disclosure table")
    p.add_argument("--registry", type=pathlib.Path,
                   default=pathlib.Path("runs/registry.jsonl"))
    p.add_argument("--preset", choices=("easy", "medium", "hard"), default=None,
                   help="Restrict disclosure table to one preset; default: "
                        "emit one block per preset")
    # Common
    p.add_argument("--metric", default="return_mean",
                   help="Scalar EvalReport / registry field to compare; "
                        "default return_mean")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out", type=pathlib.Path,
                   default=pathlib.Path("runs/_compare"),
                   help="Output dir for markdown + .png + sidecar .csv")
    p.add_argument("--no-plot", dest="no_plot", action="store_true",
                   help="Suppress matplotlib bar+errorbar plot; "
                        "emit markdown only")
    return p.parse_args(argv)


# ===== Registry / CSV loaders ============================================

def load_registry(registry_path: pathlib.Path) -> list[dict[str, Any]]:
    """Read JSONL registry; return list of latest-per-run_id ``status="completed"`` rows."""
    if not registry_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(registry_path, "rb") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            rows.append(json.loads(line.decode("utf-8")))
    # Latest per run_id.
    by_id: dict[str, dict[str, Any]] = {}
    for r in rows:
        rid = r.get("run_id")
        if rid is None:
            continue
        if (
            rid not in by_id
            or r.get("started_at_iso8601", "") >= by_id[rid].get("started_at_iso8601", "")
        ):
            by_id[rid] = r
    return [r for r in by_id.values() if r.get("status") == "completed"]


def _load_csv_returns(path: pathlib.Path, metric: str) -> tuple[str, list[float]]:
    """Read a CSV produced from ``runs/registry.jsonl``; return (method_name, samples)."""
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError("pandas is required for compare CLI CSV ingestion") from e
    df = pd.read_csv(path)
    if "variant" not in df.columns:
        raise ValueError(
            f"{path}: CSV missing 'variant' column. "
            f"Expected pkg-08 spec 07 §6.2 schema."
        )
    if metric not in df.columns:
        raise ValueError(
            f"{path}: --metric={metric!r} not found in columns: {list(df.columns)}"
        )
    name = str(df["variant"].iloc[0])
    samples = [float(v) for v in df[metric].dropna().tolist()]
    return name, samples


# ===== Markdown + plot rendering =========================================

def render_plot(
    result: ComparisonResult,
    out_path: pathlib.Path,
    *,
    metric: str = "return_mean",
) -> pathlib.Path:
    """Bar + errorbar plot per spec 07 §8.1; saved at 300 DPI; closed after save."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    methods = list(result.methods)
    means = [result.mean_per_method[m] for m in methods]
    sems = [result.sem_per_method[m] for m in methods]

    fig, ax = plt.subplots(figsize=(max(4.0, 1.4 * len(methods)), 4.5))
    xs = list(range(len(methods)))
    ax.bar(xs, means, yerr=sems, capsize=4, color="tab:blue", edgecolor="black")
    ax.set_xticks(xs)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"compare ({result.correction})")

    # Star markers (spec 07 §8.1).
    for i, m in enumerate(methods):
        if result.reference is not None and m == result.reference:
            continue
        # Find the (ref, m) pair if reference is set.
        target_pair: tuple[str, str] | None = None
        if result.reference is not None:
            target_pair = (result.reference, m)
            if target_pair not in result.adjusted_p_values:
                target_pair = (m, result.reference)
        else:
            # All-pairs: pick the smallest p where m is one of the two.
            for pair in result.pairs:
                if m in pair:
                    if (
                        target_pair is None
                        or result.adjusted_p_values[pair] < result.adjusted_p_values[target_pair]
                    ):
                        target_pair = pair
        if target_pair is None or target_pair not in result.adjusted_p_values:
            continue
        p_adj = result.adjusted_p_values[target_pair]
        marker = ""
        if p_adj < 0.001:
            marker = "***"
        elif p_adj < 0.01:
            marker = "**"
        elif p_adj < 0.05:
            marker = "*"
        if marker:
            ax.text(
                i, means[i] + (sems[i] or 0) + 0.05 * max(abs(v) for v in means + [1.0]),
                marker, ha="center", va="bottom", fontsize=12,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _write_sidecar_csv(
    result: ComparisonResult,
    out_path: pathlib.Path,
    *,
    metric: str = "return_mean",
) -> pathlib.Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["method,n,mean,sem,t_vs_reference,p_raw,p_adj,reject"]
    ref = result.reference
    for m in result.methods:
        n = result.n_per_method.get(m, 0)
        mean = result.mean_per_method.get(m, float("nan"))
        sem = result.sem_per_method.get(m, float("nan"))
        if ref is None or m == ref:
            t_ = ""
            pr = ""
            pa = ""
            rj = ""
        else:
            pair = (ref, m) if (ref, m) in result.t_statistics else (m, ref)
            t_ = f"{result.t_statistics.get(pair, float('nan')):.6f}"
            pr = f"{result.raw_p_values.get(pair, float('nan')):.6e}"
            pa = f"{result.adjusted_p_values.get(pair, float('nan')):.6e}"
            rj = str(result.rejections.get(pair, False))
        lines.append(f"{m},{n},{mean:.6f},{sem:.6f},{t_},{pr},{pa},{rj}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# ===== Mode dispatchers ==================================================

def _emit_markdown(result: ComparisonResult, out_dir: pathlib.Path,
                   metric: str, plot_filename: str | None) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = result.render_markdown()
    if plot_filename is not None:
        md += f"\n\n![compare_{metric}](./{plot_filename})\n"
    out_path = out_dir / f"compare_{metric}.md"
    out_path.write_text(md, encoding="utf-8")
    print(md)
    return out_path


def _run_two_method(args: argparse.Namespace) -> int:
    name_a, samples_a = _load_csv_returns(args.a, args.metric)
    name_b, samples_b = _load_csv_returns(args.b, args.metric)
    result = compare_methods(
        {name_a: samples_a, name_b: samples_b},
        reference=None,
        alpha=args.alpha,
    )
    plot_filename: str | None = None
    if not args.no_plot:
        plot_path = render_plot(
            result, args.out / f"compare_{args.metric}.png", metric=args.metric,
        )
        plot_filename = plot_path.name
    _emit_markdown(result, args.out, args.metric, plot_filename)
    _write_sidecar_csv(result, args.out / f"compare_{args.metric}.csv", metric=args.metric)
    return 0


def _run_multi_method(args: argparse.Namespace) -> int:
    paths = [pathlib.Path(p) for p in str(args.methods).split(",") if p.strip()]
    if len(paths) < 2:
        raise ValueError("--methods: need at least 2 CSV paths")
    method_returns: dict[str, list[float]] = {}
    for p in paths:
        name, samples = _load_csv_returns(p, args.metric)
        method_returns[name] = samples
    result = compare_methods(
        method_returns, reference=args.reference, alpha=args.alpha,
    )
    plot_filename: str | None = None
    if not args.no_plot:
        plot_path = render_plot(
            result, args.out / f"compare_{args.metric}.png", metric=args.metric,
        )
        plot_filename = plot_path.name
    _emit_markdown(result, args.out, args.metric, plot_filename)
    _write_sidecar_csv(result, args.out / f"compare_{args.metric}.csv", metric=args.metric)
    return 0


def _run_disclose(args: argparse.Namespace) -> int:
    rows = load_registry(args.registry)
    md = render_disclosure_table(rows, preset=args.preset)
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "disclosure.md"
    out_path.write_text(md, encoding="utf-8")
    print(md)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.disclose:
        if args.a or args.b or args.methods:
            raise ValueError(
                "--disclose is mutually exclusive with --a/--b and --methods"
            )
        return _run_disclose(args)
    if args.methods:
        if args.a or args.b:
            raise ValueError("--methods is mutually exclusive with --a/--b")
        return _run_multi_method(args)
    if args.a and args.b:
        return _run_two_method(args)
    raise ValueError("Must specify one of: --a + --b, --methods, --disclose")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
