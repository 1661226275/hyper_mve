"""Fast-training launcher: 12-way concurrent sweep over the v4.7 ablation cell.

Thin wrapper around :func:`hyper_mve.experiments.sweep.run_sweep` that:

  * packs ``slots_per_gpu`` worker processes onto each physical GPU listed in
    ``--gpus`` (default 6 GPUs × 2 slots = 12 concurrent training runs),
  * caps every child's VRAM at ``--mem-frac`` via
    ``torch.cuda.set_per_process_memory_fraction`` (env var injected through
    the sweep harness, applied at the top of ``_sweep_worker.py::main``),
  * enables cuDNN benchmark + TF32 matmul-precision in each child (same
    mechanism),
  * defaults to a 2-seed × 7-variant sweep at 300K steps on ``duo_base_lora``
    (mirrors the runs/2agent_06c pilot with the user-requested baselines pulled
    in for the thesis methods table),
  * writes ``runs/fast_300k/sweep_manifest.json`` listing every row + its
    registry / TB / checkpoint pointers for post-hoc analysis.

CLI examples::

    # default 14-row sweep
    python hyper_mve/scripts/train_fast_sweep.py

    # smoke first (single internal variant, 100 steps)
    python hyper_mve/scripts/train_fast_sweep.py \\
        --variants hyper --seeds 0 --max-steps 100 \\
        --gpus 2 --slots-per-gpu 1 --runs-root runs/_fast_smoke

    # dry-run to see what would spawn
    python hyper_mve/scripts/train_fast_sweep.py --dry-run

Exit codes mirror :func:`run_sweep`: 0 if no row failed, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Sequence

# Repo bootstrap: allow execution as `python hyper_mve/scripts/train_fast_sweep.py`
# without `pip install`.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hyper_mve.experiments.sweep import (  # noqa: E402
    SweepConfig,
    run_sweep,
)
from hyper_mve.experiments.run_registry import RegistryRow  # noqa: E402

DEFAULT_VARIANTS: tuple[str, ...] = (
    "hyper",
    "baseline_input_wide",
    "baseline_input_deep",
    "no_belief",
    "external_mappo",
    "external_qmix",
    "external_ma_muzero_gh",
)
DEFAULT_SEEDS: tuple[int, ...] = (0, 1)
DEFAULT_GPUS: tuple[int, ...] = (2, 3, 4, 5, 6, 7)
DEFAULT_PRESET: str = "duo_base_lora"
DEFAULT_MAX_STEPS: int = 300_000
DEFAULT_SLOTS_PER_GPU: int = 2
DEFAULT_MEM_FRAC: float = 0.45
DEFAULT_RUNS_ROOT: pathlib.Path = _REPO_ROOT / "runs" / "fast_300k"
DEFAULT_ABLATION_CELL_ID: str = "fast_300k"


# ===== Config dataclass ====================================================

@dataclass(frozen=True)
class LauncherConfig:
    preset: str
    variants: tuple[str, ...]
    seeds: tuple[int, ...]
    max_steps: int
    gpus: tuple[int, ...]
    slots_per_gpu: int
    runs_root: pathlib.Path
    mem_frac: float
    max_parallel: int
    eval_planner_mode: str
    ablation_cell_id: str
    cudnn_benchmark: bool
    matmul_precision: str | None
    extra_overrides: tuple[str, ...]
    retry_failed: bool
    dry_run: bool


# ===== CLI =================================================================

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python hyper_mve/scripts/train_fast_sweep.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--preset", type=str, default=DEFAULT_PRESET,
                   help=f"V4Config preset name (default {DEFAULT_PRESET!r})")
    p.add_argument("--variants", type=str, nargs="+", default=list(DEFAULT_VARIANTS),
                   help="space-separated CLI variant names "
                        f"(default {' '.join(DEFAULT_VARIANTS)})")
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                   help=f"seeds (default {' '.join(str(s) for s in DEFAULT_SEEDS)})")
    p.add_argument("--max-steps", dest="max_steps", type=int,
                   default=DEFAULT_MAX_STEPS,
                   help=f"per-row max training steps (default {DEFAULT_MAX_STEPS})")
    p.add_argument("--gpus", type=int, nargs="+", default=list(DEFAULT_GPUS),
                   help=f"physical CUDA device indices to use "
                        f"(default {' '.join(str(g) for g in DEFAULT_GPUS)})")
    p.add_argument("--slots-per-gpu", dest="slots_per_gpu", type=int,
                   default=DEFAULT_SLOTS_PER_GPU,
                   help=f"workers per GPU (default {DEFAULT_SLOTS_PER_GPU})")
    p.add_argument("--runs-root", dest="runs_root", type=pathlib.Path,
                   default=DEFAULT_RUNS_ROOT,
                   help=f"output root (default {DEFAULT_RUNS_ROOT})")
    p.add_argument("--mem-frac", dest="mem_frac", type=float,
                   default=DEFAULT_MEM_FRAC,
                   help="per-process VRAM cap (0 disables); "
                        f"default {DEFAULT_MEM_FRAC}")
    p.add_argument("--max-parallel", dest="max_parallel", type=int, default=None,
                   help="cap on concurrent rows (default = len(gpus)*slots_per_gpu)")
    p.add_argument("--eval-planner-mode", dest="eval_planner_mode", type=str,
                   default="planner_full",
                   choices=("direct_inference", "planner_no_crn",
                            "planner_no_coord_desc", "planner_full"),
                   help="eval planner mode (default planner_full)")
    p.add_argument("--ablation-cell-id", dest="ablation_cell_id", type=str,
                   default=DEFAULT_ABLATION_CELL_ID,
                   help=f"manifest/registry ablation tag "
                        f"(default {DEFAULT_ABLATION_CELL_ID!r})")
    p.add_argument("--cudnn-benchmark", dest="cudnn_benchmark",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="enable torch.backends.cudnn.benchmark in every child")
    p.add_argument("--matmul-precision", dest="matmul_precision",
                   choices=("highest", "high", "medium", "off"),
                   default="high",
                   help="torch.set_float32_matmul_precision in every child")
    p.add_argument("--extra-override", dest="extra_overrides",
                   action="append", default=[],
                   help="extra train_main override, repeatable, e.g. "
                        '"train.batch_size=64". Forwarded verbatim to every row.')
    p.add_argument("--retry-failed", dest="retry_failed", action="store_true",
                   help="rerun rows that previously failed; pass-through to run_sweep")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="enumerate rows, print plan, do not spawn workers")
    return p.parse_args(argv)


# ===== Validation ==========================================================

def _validate_variants(variants: Sequence[str]) -> None:
    try:
        from hyper_mve.baselines import CLI_CHOICES
    except ImportError as e:
        raise SystemExit(f"[train_fast_sweep] cannot import CLI_CHOICES: {e}")
    bad = [v for v in variants if v not in CLI_CHOICES]
    if bad:
        raise SystemExit(
            f"[train_fast_sweep] unknown variant(s) {bad}; valid: {sorted(CLI_CHOICES)}\n"
            "See hyper_mve/baselines/__init__.py::CLI_CHOICES."
        )


def _validate_gpus(gpus: Sequence[int]) -> None:
    if not gpus:
        raise SystemExit("[train_fast_sweep] --gpus must be non-empty")
    if len(set(gpus)) != len(gpus):
        raise SystemExit(f"[train_fast_sweep] --gpus contains duplicates: {gpus}")
    try:
        import torch  # type: ignore
    except ImportError:
        print("[train_fast_sweep] torch not importable; skipping device-count check.",
              file=sys.stderr)
        return
    if not torch.cuda.is_available():
        print("[train_fast_sweep] CUDA not available; trainer will fall back to CPU.",
              file=sys.stderr)
        return
    n_devices = torch.cuda.device_count()
    bad = [g for g in gpus if g >= n_devices or g < 0]
    if bad:
        raise SystemExit(
            f"[train_fast_sweep] --gpus {bad} out of range "
            f"[0, {n_devices}); torch sees {n_devices} GPUs."
        )


# ===== Build config artefacts ==============================================

def build_launcher_config(args: argparse.Namespace) -> LauncherConfig:
    matmul_prec: str | None = args.matmul_precision
    if matmul_prec == "off":
        matmul_prec = None
    max_parallel = args.max_parallel
    if max_parallel is None:
        max_parallel = len(args.gpus) * args.slots_per_gpu
    return LauncherConfig(
        preset=args.preset,
        variants=tuple(args.variants),
        seeds=tuple(int(s) for s in args.seeds),
        max_steps=int(args.max_steps),
        gpus=tuple(int(g) for g in args.gpus),
        slots_per_gpu=int(args.slots_per_gpu),
        runs_root=pathlib.Path(args.runs_root),
        mem_frac=float(args.mem_frac),
        max_parallel=int(max_parallel),
        eval_planner_mode=args.eval_planner_mode,
        ablation_cell_id=args.ablation_cell_id,
        cudnn_benchmark=bool(args.cudnn_benchmark),
        matmul_precision=matmul_prec,
        extra_overrides=tuple(args.extra_overrides),
        retry_failed=bool(args.retry_failed),
        dry_run=bool(args.dry_run),
    )


def _parse_override(token: str) -> tuple[str, object]:
    if "=" not in token:
        raise SystemExit(
            f"[train_fast_sweep] bad --extra-override {token!r}; expected key=value"
        )
    key, raw = token.split("=", 1)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw  # treat as raw string
    return key.strip(), value


def build_sweep_config(lc: LauncherConfig) -> SweepConfig:
    overrides_dict: dict[str, object] = {}
    for tok in lc.extra_overrides:
        key, value = _parse_override(tok)
        overrides_dict[key] = value
    overrides_tuple: tuple = (overrides_dict,) if overrides_dict else ({},)
    return SweepConfig(
        variants=lc.variants,
        seeds=lc.seeds,
        overrides=overrides_tuple,
        preset=lc.preset,  # type: ignore[arg-type]
        max_steps=lc.max_steps,
        eval_planner_mode=lc.eval_planner_mode,  # type: ignore[arg-type]
        max_parallel=lc.max_parallel,
        n_gpus=len(lc.gpus),  # informational; gpu_ids drives the semaphore
        ablation_cell_id=lc.ablation_cell_id,
    )


def build_child_env(lc: LauncherConfig) -> dict[str, str]:
    env: dict[str, str] = {}
    if lc.mem_frac > 0:
        env["HYPER_MVE_GPU_MEM_FRAC"] = f"{lc.mem_frac:.4f}"
    if lc.cudnn_benchmark:
        env["HYPER_MVE_CUDNN_BENCHMARK"] = "1"
    if lc.matmul_precision:
        env["HYPER_MVE_MATMUL_PRECISION"] = lc.matmul_precision
    return env


# ===== Manifest ============================================================

def write_manifest(
    lc: LauncherConfig,
    rows: list[RegistryRow],
    manifest_path: pathlib.Path,
    *,
    started_at: str,
    completed_at: str,
    interrupted: bool = False,
) -> None:
    rows_serialised: list[dict] = []
    for r in rows:
        d = asdict(r)
        rows_serialised.append({
            "run_id": d.get("run_id"),
            "run_tag": d.get("run_id"),  # back-compat alias; full tag below
            "variant": d.get("variant"),
            "seed": d.get("seed"),
            "config_hash": d.get("config_hash"),
            "gpu_id": d.get("gpu_id"),
            "status": d.get("status"),
            "failure_reason": d.get("failure_reason"),
            "walltime_seconds": d.get("walltime_seconds"),
            "config_snapshot_path": d.get("config_snapshot_path"),
            "checkpoint_path": d.get("checkpoint_path"),
            "eval_report_path": d.get("eval_report_path"),
            "tensorboard_dir": d.get("tensorboard_dir"),
            "return_mean": d.get("return_mean"),
            "return_zero_shot_unseen": d.get("return_zero_shot_unseen"),
            "regret_mean": d.get("regret_mean"),
            "ablation_cell": d.get("ablation_cell"),
        })
    payload = {
        "launcher": "train_fast_sweep",
        "started_at_iso8601": started_at,
        "completed_at_iso8601": completed_at,
        "interrupted": interrupted,
        "preset": lc.preset,
        "variants": list(lc.variants),
        "seeds": list(lc.seeds),
        "max_steps": lc.max_steps,
        "gpus": list(lc.gpus),
        "slots_per_gpu": lc.slots_per_gpu,
        "max_parallel": lc.max_parallel,
        "mem_frac": lc.mem_frac,
        "cudnn_benchmark": lc.cudnn_benchmark,
        "matmul_precision": lc.matmul_precision,
        "eval_planner_mode": lc.eval_planner_mode,
        "ablation_cell_id": lc.ablation_cell_id,
        "extra_overrides": list(lc.extra_overrides),
        "runs_root": str(lc.runs_root),
        "registry_path": str(lc.runs_root / "registry.jsonl"),
        "n_rows_total": len(lc.variants) * len(lc.seeds),
        "n_rows_completed": sum(1 for r in rows if r.status == "completed"),
        "n_rows_failed": sum(1 for r in rows if r.status == "failed"),
        "n_rows_skipped": sum(1 for r in rows if r.status == "skipped"),
        "rows": rows_serialised,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ===== Main flow ===========================================================

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    lc = build_launcher_config(args)
    _validate_variants(lc.variants)
    _validate_gpus(lc.gpus)

    if lc.slots_per_gpu > 1:
        print(
            f"[train_fast_sweep] slots_per_gpu={lc.slots_per_gpu}: packing multiple workers "
            f"onto each GPU. If you see cuBLAS / cuDNN init errors or OOMs, drop to "
            f"--slots-per-gpu 1.",
            file=sys.stderr, flush=True,
        )

    sweep_cfg = build_sweep_config(lc)
    child_env = build_child_env(lc)
    lc.runs_root.mkdir(parents=True, exist_ok=True)

    print(f"[train_fast_sweep] preset={lc.preset!r} variants={lc.variants} seeds={lc.seeds}",
          flush=True)
    print(f"[train_fast_sweep] max_steps={lc.max_steps} eval_planner_mode={lc.eval_planner_mode!r}",
          flush=True)
    print(f"[train_fast_sweep] gpus={lc.gpus} slots_per_gpu={lc.slots_per_gpu} "
          f"max_parallel={lc.max_parallel}",
          flush=True)
    print(f"[train_fast_sweep] child_env={dict(child_env)}", flush=True)
    print(f"[train_fast_sweep] runs_root={lc.runs_root}", flush=True)

    started_at = _now_iso()
    rows: list[RegistryRow] = []
    interrupted = False
    try:
        rows = run_sweep(
            sweep_cfg,
            runs_root=lc.runs_root,
            registry_path=lc.runs_root / "registry.jsonl",
            max_parallel=lc.max_parallel,
            gpu_ids=lc.gpus,
            slots_per_gpu=lc.slots_per_gpu,
            child_env=child_env,
            dry_run=lc.dry_run,
            retry_failed=lc.retry_failed,
        )
    except KeyboardInterrupt:
        interrupted = True
        print("[train_fast_sweep] interrupted; writing partial manifest…",
              file=sys.stderr, flush=True)

    completed_at = _now_iso()
    if lc.dry_run:
        print("[train_fast_sweep] dry-run: no manifest written", flush=True)
        return 0

    manifest_path = lc.runs_root / "sweep_manifest.json"
    write_manifest(lc, rows, manifest_path,
                   started_at=started_at, completed_at=completed_at,
                   interrupted=interrupted)
    print(f"[train_fast_sweep] manifest: {manifest_path}", flush=True)
    n_failed = sum(1 for r in rows if r.status == "failed")
    n_completed = sum(1 for r in rows if r.status == "completed")
    n_skipped = sum(1 for r in rows if r.status == "skipped")
    print(f"[train_fast_sweep] completed={n_completed} failed={n_failed} skipped={n_skipped}",
          flush=True)
    if interrupted:
        return 130
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
