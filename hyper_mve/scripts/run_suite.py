"""run_suite.py — unified launcher for the modular thesis experiment suite.

One entry point over the declarative cells in ``experiments/suite/`` (+ the
in-place ``experiments/ablations/``). Reuses ``run_sweep`` per cell with
per-cell registry isolation, and supports **explicit selective re-run** down to
a single ``(variant, seed)`` row.

    # see every cell + row count + blocked status (no training)
    python hyper_mve/scripts/run_suite.py --list

    # run one cell (Easy 生死判官 gate), preview only
    python hyper_mve/scripts/run_suite.py --only abl3_easy_n2 --dry-run

    # run all must_have cells across a GPU pool
    python hyper_mve/scripts/run_suite.py --tier must_have --gpus 2,3,4,5 --slots-per-gpu 2

    # re-run ONLY one row after editing its code (force past the resume cache)
    python hyper_mve/scripts/run_suite.py --only main_comparison --variants hyper --seeds 0 --force

Selective re-run model (locked: explicit targeting):
  * ``run_sweep`` already skips rows whose ``(variant, seed, config_hash)`` is
    completed — so re-invoking the suite never retrains finished rows.
  * ``--only/--variants/--seeds/--size`` narrow *which* rows a cell enumerates.
  * ``--force`` appends a ``failed`` tombstone (append-only safe) for exactly the
    narrowed rows, then runs that cell with ``retry_failed=True`` — so only those
    rows re-execute. (Use after editing a cell's underlying code; the harness has
    no code-version invalidation, so re-run is explicit.)
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from typing import Sequence

# Allow `python hyper_mve/scripts/run_suite.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hyper_mve.experiments import sweep  # noqa: E402
from hyper_mve.experiments.run_registry import RegistryRow, RunRegistry  # noqa: E402
from hyper_mve.experiments.suite import SuiteCell, load_manifest  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python hyper_mve/scripts/run_suite.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--manifest", type=pathlib.Path, default=None,
                   help="manifest.yaml path (default experiments/suite/manifest.yaml)")
    p.add_argument("--list", dest="list_only", action="store_true",
                   help="print every cell + tier + deliverables + row count, then exit")
    p.add_argument("--only", default=None,
                   help="comma-separated cell ids to run (default: all cells)")
    p.add_argument("--tier", default=None,
                   choices=("must_have", "degradable", "cuttable"),
                   help="restrict to cells of this tier")
    p.add_argument("--size", default=None, choices=("easy", "medium", "hard"),
                   help="restrict to cells of this size (informational meta.size)")
    p.add_argument("--variants", default=None,
                   help="comma-separated variants to KEEP within each cell (intersection)")
    p.add_argument("--seeds", default=None,
                   help="comma-separated seeds to use within each cell (replaces the cell seeds)")
    p.add_argument("--max-steps", dest="max_steps", type=int, default=None,
                   help="override each cell's max training steps")
    p.add_argument("--force", action="store_true",
                   help="force re-run of the (narrowed) rows even if completed "
                        "(appends a failed tombstone + retry_failed)")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="enumerate + print per cell; do not spawn workers")
    p.add_argument("--runs-root", dest="runs_root", type=pathlib.Path,
                   default=pathlib.Path("runs/suite"),
                   help="output root; each cell lands under <runs-root>/<runs_subdir>/")
    p.add_argument("--gpus", default=None,
                   help="comma-separated physical CUDA device ids (default: cell n_gpus)")
    p.add_argument("--slots-per-gpu", dest="slots_per_gpu", type=int, default=1)
    p.add_argument("--max-parallel", dest="max_parallel", type=int, default=None)
    p.add_argument("--retry-failed", dest="retry_failed", action="store_true",
                   help="re-enqueue rows that previously failed (independent of --force)")
    p.add_argument("--mem-frac", dest="mem_frac", type=float, default=0.0,
                   help="per-process VRAM cap (0 disables)")
    p.add_argument("--cudnn-benchmark", dest="cudnn_benchmark",
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--matmul-precision", dest="matmul_precision",
                   choices=("highest", "high", "medium", "off"), default="high")
    p.add_argument("--mamz-num-simulations", dest="mamz_num_simulations", type=int, default=8,
                   help="MCTS sims/step for external_ma_muzero_gh (cost is linear; module default 8)")
    return p.parse_args(argv)


# ===== helpers ============================================================

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _select_cells(cells: list[SuiteCell], args: argparse.Namespace) -> list[SuiteCell]:
    out = cells
    if args.only:
        wanted = [c.strip() for c in args.only.split(",") if c.strip()]
        by_id = {c.id: c for c in cells}
        bad = [w for w in wanted if w not in by_id]
        if bad:
            raise SystemExit(
                f"[run_suite] unknown cell id(s) {bad}; valid: {sorted(by_id)}"
            )
        out = [by_id[w] for w in wanted]
    if args.tier:
        out = [c for c in out if c.tier == args.tier]
    if args.size:
        out = [c for c in out if c.size == args.size]
    return out


def _narrow(cell: SuiteCell, args: argparse.Namespace):
    """Return the cell's SweepConfig narrowed by --variants/--seeds/--max-steps."""
    cfg = cell.sweep_config
    warn = None
    if args.variants:
        keep = set(v.strip() for v in args.variants.split(",") if v.strip())
        kept = tuple(v for v in cfg.variants if v in keep)
        if not kept:
            warn = f"--variants {sorted(keep)} ∩ {list(cfg.variants)} is empty"
        cfg = replace(cfg, variants=kept)
    if args.seeds:
        seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
        cfg = replace(cfg, seeds=seeds)
    if args.max_steps is not None:
        cfg = replace(cfg, max_steps=int(args.max_steps))
    return cfg, warn


def _child_env(args: argparse.Namespace) -> dict[str, str]:
    env: dict[str, str] = {}
    if args.mem_frac and args.mem_frac > 0:
        env["HYPER_MVE_GPU_MEM_FRAC"] = f"{args.mem_frac:.4f}"
    if args.cudnn_benchmark:
        env["HYPER_MVE_CUDNN_BENCHMARK"] = "1"
    if args.matmul_precision and args.matmul_precision != "off":
        env["HYPER_MVE_MATMUL_PRECISION"] = args.matmul_precision
    if args.mamz_num_simulations and args.mamz_num_simulations > 0:
        env["HYPER_MVE_MAMZ_NUM_SIMULATIONS"] = str(int(args.mamz_num_simulations))
    return env


def _force_invalidate(cfg, registry_path: pathlib.Path) -> int:
    """Append a `failed` tombstone for every (narrowed) row so it re-runs.

    Append-only safe: a NEW run_id row with a current timestamp becomes the
    latest state for that ``(variant, seed, config_hash)`` triple; with
    ``retry_failed=True`` ``_resume_skip`` then re-enqueues exactly those rows.
    """
    reg = RunRegistry(registry_path)
    now = _now_iso()
    rows = sweep.enumerate_cartesian(cfg)
    n = 0
    for row in rows:
        ch = sweep.row_config_hash(cfg, row)
        reg.append_row(RegistryRow(
            run_id=uuid.uuid4().hex,
            variant=row.variant,
            seed=row.seed,
            config_hash=ch,
            ablation_cell=cfg.ablation_cell_id,
            sweep_row_index=row.sweep_row_index,
            git_sha="",
            git_dirty=False,
            started_at_iso8601=now,
            completed_at_iso8601=now,
            status="failed",
            failure_reason="forced re-run (--force)",
            gpu_id=None,
            walltime_seconds=None,
            peak_gpu_memory_mb=None,
            config_snapshot_path="",
            checkpoint_path=None,
            eval_report_path=None,
            tensorboard_dir="",
            return_mean=None,
            return_zero_shot_unseen=None,
            regret_mean=None,
        ))
        n += 1
    return n


def _print_list(cells: list[SuiteCell], args: argparse.Namespace) -> None:
    print(f"{'id':<26} {'tier':<11} {'size':<7} {'rows':>5}  deliverables / blocked_on")
    print("-" * 100)
    total = 0
    for cell in cells:
        cfg, warn = _narrow(cell, args)
        n = len(sweep.enumerate_cartesian(cfg))
        total += n
        tag = ""
        if cell.blocked:
            tag = f"  [BLOCKED: {', '.join(cell.blocked_on)}]"
        delivs = ", ".join(cell.deliverables)
        print(f"{cell.id:<26} {cell.tier:<11} {cell.size:<7} {n:>5}  {delivs}{tag}")
        if warn:
            print(f"{'':<26} {'':<11} {'':<7} {'':>5}  ⚠ {warn}")
    print("-" * 100)
    print(f"{len(cells)} cell(s), {total} total rows (after narrowing).")


# ===== main ==============================================================

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cells = load_manifest(args.manifest)
    selected = _select_cells(cells, args)
    if not selected:
        print("[run_suite] no cells selected.")
        return 0

    if args.list_only:
        _print_list(selected, args)
        return 0

    gpu_ids = None
    if args.gpus:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]

    # When a GPU pool is given without an explicit --max-parallel, fill it:
    # default concurrency = len(gpus) × slots_per_gpu (otherwise the cell's own
    # max_parallel — typically 2 — would bottleneck a larger pool).
    max_parallel = args.max_parallel
    if max_parallel is None and gpu_ids is not None:
        max_parallel = len(gpu_ids) * max(1, args.slots_per_gpu)

    n_failed_cells = 0
    for cell in selected:
        cfg, warn = _narrow(cell, args)
        if warn:
            print(f"[run_suite] skip {cell.id}: {warn}", file=sys.stderr)
            continue
        if not cfg.variants or not cfg.seeds:
            print(f"[run_suite] skip {cell.id}: no variants/seeds after narrowing", file=sys.stderr)
            continue

        runs_root_cell = args.runs_root / cell.runs_subdir
        runs_root_cell.mkdir(parents=True, exist_ok=True)
        registry_path = runs_root_cell / "registry.jsonl"

        blocked_note = f"  [BLOCKED: {', '.join(cell.blocked_on)}]" if cell.blocked else ""
        print(f"\n=== cell {cell.id} ({cell.tier}, {cell.size}){blocked_note} ===", flush=True)
        print(f"    variants={list(cfg.variants)} seeds={list(cfg.seeds)} "
              f"preset={cfg.preset} max_steps={cfg.max_steps} → {runs_root_cell}", flush=True)

        retry_failed = bool(args.retry_failed)
        if args.force and not args.dry_run:
            n_tomb = _force_invalidate(cfg, registry_path)
            retry_failed = True
            print(f"    [--force] wrote {n_tomb} failed-tombstone(s) → will re-run those rows", flush=True)

        rows = sweep.run_sweep(
            cfg,
            runs_root=runs_root_cell,
            registry_path=registry_path,
            max_parallel=max_parallel,
            n_gpus=(cfg.n_gpus if gpu_ids is None else None),
            gpu_ids=gpu_ids,
            slots_per_gpu=args.slots_per_gpu,
            child_env=_child_env(args),
            dry_run=args.dry_run,
            retry_failed=retry_failed,
        )
        if not args.dry_run:
            nf = sum(1 for r in rows if r.status == "failed")
            nc = sum(1 for r in rows if r.status == "completed")
            ns = sum(1 for r in rows if r.status == "skipped")
            print(f"    cell {cell.id}: completed={nc} failed={nf} skipped={ns}", flush=True)
            if nf:
                n_failed_cells += 1

    return 0 if n_failed_cells == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
