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

def _build_plan(selected, args) -> list[tuple]:
    """Narrow each cell + prepare its runs_root/registry → [(cell, cfg, root, reg)]."""
    plan: list[tuple] = []
    for cell in selected:
        cfg, warn = _narrow(cell, args)
        if warn:
            print(f"[run_suite] skip {cell.id}: {warn}", file=sys.stderr)
            continue
        if not cfg.variants or not cfg.seeds:
            print(f"[run_suite] skip {cell.id}: no variants/seeds after narrowing", file=sys.stderr)
            continue
        root = args.runs_root / cell.runs_subdir
        root.mkdir(parents=True, exist_ok=True)
        plan.append((cell, cfg, root, root / "registry.jsonl"))
    return plan


def _cell_summary(cell, rows) -> int:
    """Print a cell's completed/failed/skipped tally; return the failed count."""
    nf = sum(1 for r in rows if r.status == "failed")
    nc = sum(1 for r in rows if r.status == "completed")
    ns = sum(1 for r in rows if r.status == "skipped")
    print(f"    cell {cell.id}: completed={nc} failed={nf} skipped={ns}", flush=True)
    return nf


def _run_sequential(plan, gpu_ids, args, child_env, retry_failed) -> int:
    """One cell at a time (dry-run / single cell / no GPU pool)."""
    max_parallel = args.max_parallel
    if max_parallel is None and gpu_ids is not None:
        max_parallel = len(gpu_ids) * max(1, args.slots_per_gpu)
    n_failed = 0
    for cell, cfg, root, reg in plan:
        blocked = f"  [BLOCKED: {', '.join(cell.blocked_on)}]" if cell.blocked else ""
        print(f"\n=== cell {cell.id} ({cell.tier}, {cell.size}){blocked} ===", flush=True)
        print(f"    variants={list(cfg.variants)} seeds={list(cfg.seeds)} "
              f"preset={cfg.preset} max_steps={cfg.max_steps} → {root}", flush=True)
        rows = sweep.run_sweep(
            cfg, runs_root=root, registry_path=reg,
            max_parallel=max_parallel,
            n_gpus=(cfg.n_gpus if gpu_ids is None else None),
            gpu_ids=gpu_ids, slots_per_gpu=args.slots_per_gpu,
            child_env=child_env, dry_run=args.dry_run, retry_failed=retry_failed,
        )
        if not args.dry_run and _cell_summary(cell, rows):
            n_failed += 1
    return 0 if n_failed == 0 else 1


def _run_cross_cell(plan, gpu_ids, args, child_env, retry_failed) -> int:
    """Pool rows from ALL selected cells through ONE shared GPU semaphore.

    Every cell's run_sweep runs concurrently, sharing one
    MultiSlotGpuSemaphore(len(gpus) × slots_per_gpu) — so the pool fills with rows
    drawn ACROSS cells (e.g. six 3-row LoRA cells → 9 concurrent, not 3-at-a-time).
    Per-cell registries stay isolated (separate runs_subdir).
    """
    import concurrent.futures

    slots = max(1, args.slots_per_gpu)
    shared_sem = sweep.MultiSlotGpuSemaphore(gpu_ids, slots_per_gpu=slots)
    total_slots = shared_sem.n_gpus
    total_rows = sum(len(sweep.enumerate_cartesian(cfg)) for _c, cfg, _r, _g in plan)
    print(f"\n[run_suite] cross-cell pool: {len(plan)} cells, {total_rows} rows, "
          f"{total_slots} concurrent ({len(gpu_ids)} GPUs × {slots} slots).", flush=True)
    for cell, cfg, root, _reg in plan:
        b = f"  [BLOCKED: {', '.join(cell.blocked_on)}]" if cell.blocked else ""
        print(f"    {cell.id}: {len(sweep.enumerate_cartesian(cfg))} rows → {root}{b}", flush=True)

    def _one(item):
        cell, cfg, root, reg = item
        rows = sweep.run_sweep(
            cfg, runs_root=root, registry_path=reg,
            max_parallel=total_slots, gpu_ids=gpu_ids, slots_per_gpu=slots,
            shared_sem=shared_sem, child_env=child_env, retry_failed=retry_failed,
        )
        return cell, rows

    n_failed = 0
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(plan), thread_name_prefix="suite-cell",
    ) as pool:
        futs = {pool.submit(_one, item): item[0] for item in plan}
        for fut in concurrent.futures.as_completed(futs):
            cell = futs[fut]
            try:
                _cell, rows = fut.result()
            except Exception as e:  # noqa: BLE001 — one cell failing shouldn't sink the rest
                print(f"[run_suite] cell {cell.id} raised {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
                n_failed += 1
                continue
            if _cell_summary(cell, rows):
                n_failed += 1
    return 0 if n_failed == 0 else 1


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

    plan = _build_plan(selected, args)
    if not plan:
        print("[run_suite] nothing to run after narrowing.")
        return 0

    child_env = _child_env(args)

    # --force: append failed tombstones for every planned cell's (narrowed) rows,
    # then run with retry_failed so only those rows re-execute.
    if args.force and not args.dry_run:
        for cell, cfg, _root, reg in plan:
            n = _force_invalidate(cfg, reg)
            print(f"[run_suite] --force {cell.id}: {n} tombstone(s)", flush=True)
    retry_failed = bool(args.retry_failed or (args.force and not args.dry_run))

    # Cross-cell pool when a GPU pool is given and >1 cell runs live; otherwise
    # one cell at a time (single cell or no pool). Dry-run always enumerates
    # per-cell, then advertises the cross-cell pool the live run would use.
    cross = gpu_ids is not None and len(plan) > 1
    if args.dry_run:
        rc = _run_sequential(plan, gpu_ids, args, child_env, retry_failed)
        if cross:
            slots = max(1, args.slots_per_gpu)
            total = len(gpu_ids) * slots
            print(f"\n[run_suite] LIVE run pools all {len(plan)} cells through ONE "
                  f"{total}-slot pool ({len(gpu_ids)} GPUs × {slots}) → up to {total} "
                  f"concurrent ACROSS cells. (Per-cell 'max_parallel' above is the "
                  f"within-cell cap, not the live total.)", flush=True)
        return rc
    if cross:
        return _run_cross_cell(plan, gpu_ids, args, child_env, retry_failed)
    return _run_sequential(plan, gpu_ids, args, child_env, retry_failed)


if __name__ == "__main__":
    raise SystemExit(main())
