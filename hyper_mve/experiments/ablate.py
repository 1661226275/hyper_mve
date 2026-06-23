"""pkg-08 spec 06 §2 — Ablation CLI dispatcher.

`python -m hyper_mve.experiments.ablate --ablation <id>` is a thin wrapper:
loads the canned YAML at ``hyper_mve/experiments/ablations/<id>.yaml``,
materialises a :class:`SweepConfig`, optionally applies ``--preset`` /
``--seeds`` overrides, and dispatches to :func:`run_sweep`.

Five canned IDs (4 logical cells; Lock 1 5/4 asymmetry):

    abl1                  → abl1_gen_scope.yaml
    abl4_crn_joint        → abl4_crn_joint.yaml
    abl4_joint_easy_n2    → abl4_joint_easy_n2.yaml          (HARD-PIN preset=easy)
    abl6                  → abl6_fehr_schmidt.yaml
    abl7                  → abl7_curriculum.yaml
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys
import warnings
from typing import Sequence

from .suite.cell import load_cell
from .sweep import SweepConfig, run_sweep


# pkg-08 spec 06 §2.1 — locked verbatim.
ABLATION_IDS: tuple[str, ...] = (
    "abl1",
    "abl4_crn_joint",
    "abl4_joint_easy_n2",
    "abl6",
    "abl7",
)

# CLI-id → YAML-stem mapping (pkg-08 spec 06 §2.2 dispatch table).
_YAML_STEM: dict[str, str] = {
    "abl1": "abl1_gen_scope",
    "abl4_crn_joint": "abl4_crn_joint",
    "abl4_joint_easy_n2": "abl4_joint_easy_n2",
    "abl6": "abl6_fehr_schmidt",
    "abl7": "abl7_curriculum",
}

# IDs whose YAML hard-pins ``preset:`` (Lock 3) — ``--preset`` is logged-and-ignored.
_HARDPIN_PRESET: dict[str, str] = {
    "abl4_joint_easy_n2": "easy",
}

ABLATIONS_DIR: pathlib.Path = pathlib.Path(__file__).parent / "ablations"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m hyper_mve.experiments.ablate",
        description=(
            "Dispatch a canned ablation cell defined by a YAML under "
            "hyper_mve/experiments/ablations/. Five legal IDs (4 logical "
            "cells, with Abl4 split into two dispatch files per spec 06 "
            "Lock 1):\n"
            "  abl1                  — gen_scope 7-cell generalisation grid\n"
            "  abl4_crn_joint        — CRN x CoordDesc 2x2 matrix (Medium)\n"
            "  abl4_joint_easy_n2    — Easy N=2 exhaustive 36-action enum\n"
            "  abl6                  — Fehr-Schmidt alpha/beta 3x3 sensitivity scan\n"
            "  abl7                  — curriculum oracle_only/mixed/infer_only"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ablation", required=True, choices=ABLATION_IDS)
    p.add_argument("--preset", default=None, choices=("easy", "medium", "hard"),
                   help="Override the YAML's default preset; ignored when the "
                        "YAML hard-pins a preset (Lock 3).")
    p.add_argument("--seeds", type=int, default=None,
                   help="Override the YAML's default seed count "
                        "(n seeds → tuple(range(n))).")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="Print the materialised SweepConfig and exit 0 "
                        "without dispatching to run_sweep.")
    p.add_argument("--max-parallel", dest="max_parallel", type=int, default=None)
    p.add_argument("--n-gpus", dest="n_gpus", type=int, default=None)
    p.add_argument("--retry-failed", dest="retry_failed", action="store_true")
    p.add_argument("--runs-root", dest="runs_root", type=pathlib.Path,
                   default=pathlib.Path("runs"))
    p.add_argument("--registry-path", dest="registry_path", type=pathlib.Path, default=None)
    return p.parse_args(argv)


def materialise_sweep_config(args: argparse.Namespace) -> SweepConfig:
    """Load YAML + apply CLI overrides per Lock 3 hard-pin guard."""
    yaml_path = ABLATIONS_DIR / f"{_YAML_STEM[args.ablation]}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Canned ablation YAML not found: {yaml_path}. "
            f"Expected file for --ablation={args.ablation}."
        )
    # Ablation YAMLs are suite cells (carry a `meta:` block SweepConfig.from_yaml
    # would reject); load via the cell loader and take its SweepConfig.
    sweep_cfg = load_cell(yaml_path).sweep_config

    # Hard-pin guard (Lock 3).
    pinned = _HARDPIN_PRESET.get(args.ablation)
    if pinned is not None and args.preset is not None and args.preset != pinned:
        print(
            f"[WARN] --preset={args.preset} ignored: {args.ablation} hard-pins "
            f"preset={pinned} per spec 06 Lock 3.",
            file=sys.stderr,
        )

    # Apply CLI overrides (only when not hard-pinned).
    overrides: dict[str, object] = {}
    if pinned is None and args.preset is not None:
        overrides["preset"] = args.preset
    if args.seeds is not None and args.seeds > 0:
        overrides["seeds"] = tuple(range(int(args.seeds)))
    if args.max_parallel is not None:
        overrides["max_parallel"] = int(args.max_parallel)
    if args.n_gpus is not None:
        overrides["n_gpus"] = int(args.n_gpus)
    if overrides:
        sweep_cfg = dataclasses.replace(sweep_cfg, **overrides)
    return sweep_cfg


def _emit_dry_run(sweep_cfg: SweepConfig) -> None:
    """spec 06 §2.4 dry-run output — printed before exit 0."""
    n_v, n_s = len(sweep_cfg.variants), len(sweep_cfg.seeds)
    n_o = len(sweep_cfg.overrides) or 1
    total = n_v * n_s * n_o
    print(f"[dry-run] variants={sweep_cfg.variants}")
    print(f"[dry-run] seeds={sweep_cfg.seeds}, preset={sweep_cfg.preset!r}, "
          f"overrides={n_o}")
    print(f"[dry-run] eval_planner_mode={sweep_cfg.eval_planner_mode!r} "
          f"(per spec 03 Lock 1)")
    print(f"[dry-run] ablation_cell_id={sweep_cfg.ablation_cell_id!r}")
    print(f"[dry-run] Total rows would be: {n_v} x {n_s} x {n_o} = {total}")
    print("[dry-run] Exiting without calling run_sweep.")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    sweep_cfg = materialise_sweep_config(args)
    if args.dry_run:
        _emit_dry_run(sweep_cfg)
        return 0
    rows = run_sweep(
        sweep_cfg,
        runs_root=args.runs_root,
        registry_path=args.registry_path,
        max_parallel=args.max_parallel,
        n_gpus=args.n_gpus,
        retry_failed=args.retry_failed,
    )
    n_completed = sum(1 for r in rows if r.status == "completed")
    n_failed = sum(1 for r in rows if r.status == "failed")
    n_skipped = sum(1 for r in rows if r.status == "skipped")
    print(f"ablate {args.ablation} done: completed={n_completed} "
          f"failed={n_failed} skipped={n_skipped}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
