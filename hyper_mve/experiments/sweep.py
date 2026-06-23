"""pkg-08 spec 05 — Sweep harness (`SweepConfig` + `run_sweep` + cartesian + GpuSemaphore).

Three hard locks (mirrored from spec 05 header):

* **Lock 1** — subprocess-per-row is non-negotiable (C8-ABL-ISO1). Every
  sweep row spawns its own Python interpreter via
  ``subprocess.Popen([sys.executable, "-m", "hyper_mve.experiments._sweep_worker"], ...)``.
  No in-process row loop, no ``concurrent.futures.ProcessPoolExecutor``
  worker reuse.
* **Lock 2** — ``runs/registry.jsonl`` is JSONL append-only (C8-ABL-REG1). Every
  state transition (pending → running → completed/failed/skipped) is a NEW
  line. See ``run_registry.RunRegistry``.
* **Lock 3** — schema mirror against pkg-08 spec 08 §4 is byte-identical
  (23 keys: 22 schema-domain + 1 ``schema_version`` sentinel).

The sweep harness is a SCHEDULER, not a trainer — all training happens
inside ``_sweep_worker.py`` which delegates to
``hyper_mve.scripts.train_main.main(argv)``.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import pathlib
import subprocess
import sys
import threading
import uuid
import warnings
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Iterable, Literal, Mapping, Sequence

from .run_registry import RegistryRow, RunRegistry, SCHEMA_VERSION

__all__ = [
    "SweepConfig",
    "SweepRow",
    "GpuSemaphore",
    "MultiSlotGpuSemaphore",
    "compute_config_hash",
    "enumerate_cartesian",
    "load_yaml",
    "from_mapping",
    "row_config_hash",
    "run_sweep",
]


# ===== SweepConfig dataclass (spec 05 §3.1) ==============================

PlannerMode = Literal[
    "direct_inference",
    "planner_no_crn",
    "planner_no_coord_desc",
    "planner_full",
]
Preset = Literal["easy", "medium", "hard"]


@dataclass(frozen=True)
class SweepConfig:
    """Single source of truth for one sweep invocation (spec 05 §3.1).

    Materialised either from a YAML at
    ``hyper_mve/experiments/ablations/<id>.yaml`` (spec 06 dispatcher) or
    constructed in-process (μP self-check, spec 04).
    """

    # === Cartesian axes (3) ===
    variants: tuple[str, ...]
    seeds: tuple[int, ...]
    overrides: tuple[Mapping[str, object], ...]

    # === Single-valued scalars (3) ===
    preset: Preset = "medium"
    max_steps: int = 200000
    eval_planner_mode: PlannerMode = "planner_full"

    # === Sweep-level resource hints (3) ===
    max_parallel: int = 1
    n_gpus: int = 1
    ablation_cell_id: str | None = None

    # === Schema sentinel (1) — bumped only by synchronous spec 05 + spec 08 §4 edit ===
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def from_yaml(cls, path: pathlib.Path | str) -> "SweepConfig":
        """Load from a YAML file. Strict-mode load — unknown keys raise ``ValueError``."""
        return load_yaml(path)


@dataclass(frozen=True)
class SweepRow:
    """One row of an enumerated SweepConfig (spec 05 §3.3)."""

    variant: str
    seed: int
    overrides: Mapping[str, object]
    sweep_row_index: int
    ablation_cell: str | None
    config_hash: str = ""
    run_id: str = ""
    run_tag: str = ""


# ===== YAML loader (spec 05 §3.2) ========================================

_LEGAL_KEYS = {f.name for f in dataclasses.fields(SweepConfig)}


def from_mapping(raw: Mapping[str, Any], *, source: str = "<mapping>") -> SweepConfig:
    """Build a SweepConfig from an already-parsed mapping; strict unknown-key rejection.

    Single source of strictness for both ``load_yaml`` (file path) and the suite
    cell loader (which pops a non-SweepConfig ``meta:`` block first, then hands
    the rest here). ``source`` is only used in error messages.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: top-level node must be a mapping; got {type(raw).__name__}")
    extra = set(raw) - _LEGAL_KEYS
    if extra:
        raise ValueError(
            f"{source}: unknown SweepConfig keys {sorted(extra)}; legal keys: {sorted(_LEGAL_KEYS)}"
        )
    variants = tuple(raw.get("variants", ()) or ())
    seeds = tuple(int(s) for s in (raw.get("seeds", ()) or ()))
    overrides = tuple(dict(o or {}) for o in (raw.get("overrides", ()) or ()))
    kwargs: dict[str, Any] = dict(
        variants=variants,
        seeds=seeds,
        overrides=overrides,
    )
    for key in (
        "preset", "max_steps", "eval_planner_mode",
        "max_parallel", "n_gpus", "ablation_cell_id", "schema_version",
    ):
        if key in raw:
            kwargs[key] = raw[key]
    return SweepConfig(**kwargs)


def load_yaml(path: pathlib.Path | str) -> SweepConfig:
    """Load a SweepConfig YAML; strict-mode unknown-key rejection."""
    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        raise ImportError("pyyaml is required for SweepConfig.from_yaml()") from e
    p = pathlib.Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return from_mapping(raw, source=str(p))


# ===== config_hash (spec 05 §3.5) ========================================

def compute_config_hash(cfg_dict: Mapping[str, Any]) -> str:
    """16-byte (32 hex char) blake2b of canonical-JSON-serialised cfg.to_dict().

    Stable across Python process restarts, dict ordering (sort_keys=True),
    and tuple-vs-list (cfg.to_dict() coerces tuples to lists). Sensitive to
    every field of every sub-config.
    """
    canonical = json.dumps(cfg_dict, sort_keys=True, default=str)
    return hashlib.blake2b(canonical.encode("utf-8"), digest_size=16).hexdigest()


# ===== Cartesian enumeration (spec 05 §3.3) ==============================

def enumerate_cartesian(sweep: SweepConfig) -> list[SweepRow]:
    """Deterministic cartesian product of variants × seeds × overrides.

    Cardinality: ``|V| · |S| · max(1, |O|)``. Order: lexicographic over
    ``(variant_idx, seed_idx, override_idx)`` — stable across runs, so resume
    semantics work.

    Empty-overrides edge case: when ``sweep.overrides == ()``, the harness
    substitutes ``({},)`` (single empty dict) so the cardinality is
    ``|V| · |S| · 1``.
    """
    overrides = sweep.overrides if sweep.overrides else ({},)
    rows: list[SweepRow] = []
    idx = 0
    for v, s, o in itertools.product(sweep.variants, sweep.seeds, overrides):
        rows.append(
            SweepRow(
                variant=v,
                seed=int(s),
                overrides=dict(o),
                sweep_row_index=idx,
                ablation_cell=sweep.ablation_cell_id,
            )
        )
        idx += 1
    return rows


# ===== Per-row config hash (resume / --force key surrogate) ==============

def _row_hash_input(sweep_cfg: SweepConfig, row: SweepRow) -> dict[str, Any]:
    """Deterministic surrogate hashed into ``config_hash`` (no torch in the parent).

    Mirrors the worker's full-config hash inputs closely enough for resume +
    selective re-run: (variant, seed, preset, max_steps, eval_planner_mode,
    overrides, ablation_cell_id, schema_version).
    """
    return {
        "variant": row.variant,
        "seed": row.seed,
        "preset": sweep_cfg.preset,
        "max_steps": sweep_cfg.max_steps,
        "eval_planner_mode": sweep_cfg.eval_planner_mode,
        "overrides": dict(row.overrides),
        "ablation_cell_id": sweep_cfg.ablation_cell_id,
        "schema_version": sweep_cfg.schema_version,
    }


def row_config_hash(sweep_cfg: SweepConfig, row: SweepRow) -> str:
    """``config_hash`` for one row — the exact key ``run_sweep`` skips/resumes on.

    Exposed so ``run_suite --force`` can recompute the same triple
    ``(variant, seed, config_hash)`` it needs to invalidate.
    """
    return compute_config_hash(_row_hash_input(sweep_cfg, row))


# ===== GpuSemaphore (spec 05 §7) =========================================

class GpuSemaphore:
    """Token-queue over n_gpus integer tokens (spec 05 §7).

    ``acquire()`` blocks until a token is available, returns gpu_id.
    ``release(gpu_id)`` returns a token to the queue.
    """

    def __init__(self, n_gpus: int) -> None:
        if n_gpus < 1:
            raise ValueError(f"n_gpus must be >= 1; got {n_gpus}")
        self._n = int(n_gpus)
        self._available: "mp.Queue[int]" = mp.Queue(maxsize=self._n)
        for i in range(self._n):
            self._available.put(i)

    @property
    def n_gpus(self) -> int:
        return self._n

    def acquire(self) -> int:
        return int(self._available.get())

    def release(self, gpu_id: int) -> None:
        if not (0 <= int(gpu_id) < self._n):
            raise ValueError(f"release({gpu_id}): gpu_id out of range [0, {self._n})")
        self._available.put(int(gpu_id))


class MultiSlotGpuSemaphore:
    """Token-bag over ``len(gpu_ids) × slots_per_gpu`` GPU-id tokens.

    Each ``gpu_id`` appears ``slots_per_gpu`` times in the underlying queue, so
    up to ``slots_per_gpu`` workers may share one physical device by setting
    ``CUDA_VISIBLE_DEVICES=<gpu_id>`` in their environment.

    Differences vs :class:`GpuSemaphore`:
      * ``GpuSemaphore(n_gpus)`` holds tokens labelled ``0..n_gpus-1``.
      * ``MultiSlotGpuSemaphore([2,3,4,5,6,7], slots_per_gpu=2)`` holds 12
        tokens; each of ``{2,3,4,5,6,7}`` appears twice.

    Concurrency guarantees match :class:`GpuSemaphore` (both use ``mp.Queue``,
    which is thread-safe and process-safe).
    """

    def __init__(self, gpu_ids: Sequence[int], slots_per_gpu: int = 1) -> None:
        gpu_ids = tuple(int(g) for g in gpu_ids)
        if not gpu_ids:
            raise ValueError("gpu_ids must be non-empty")
        if slots_per_gpu < 1:
            raise ValueError(f"slots_per_gpu must be >= 1; got {slots_per_gpu}")
        self._gpu_ids: tuple[int, ...] = gpu_ids
        self._slots = int(slots_per_gpu)
        self._n_total = len(self._gpu_ids) * self._slots
        self._allowed: frozenset[int] = frozenset(self._gpu_ids)
        self._available: "mp.Queue[int]" = mp.Queue(maxsize=self._n_total)
        for g in self._gpu_ids:
            for _ in range(self._slots):
                self._available.put(int(g))

    @property
    def n_gpus(self) -> int:
        """Total slot count (matches the GpuSemaphore.n_gpus contract for clamps)."""
        return self._n_total

    @property
    def gpu_ids(self) -> tuple[int, ...]:
        return self._gpu_ids

    @property
    def slots_per_gpu(self) -> int:
        return self._slots

    def acquire(self) -> int:
        return int(self._available.get())

    def release(self, gpu_id: int) -> None:
        g = int(gpu_id)
        if g not in self._allowed:
            raise ValueError(
                f"release({gpu_id}): not one of gpu_ids={self._gpu_ids}"
            )
        self._available.put(g)


# ===== Git provenance helpers ============================================

def _git_sha(repo_root: pathlib.Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root),
            stderr=subprocess.DEVNULL, timeout=5,
        )
        return out.decode("ascii").strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unknown"


def _git_dirty(repo_root: pathlib.Path) -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(repo_root),
            stderr=subprocess.DEVNULL, timeout=5,
        )
        return bool(out.strip())
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return False


def _now_iso8601() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ===== Subprocess invocation (spec 05 §5.2) ==============================

def _build_run_tag(row: SweepRow) -> str:
    return f"{row.variant}_seed{row.seed}_row{row.sweep_row_index}_{row.config_hash[:8]}"


def _build_payload(
    row: SweepRow,
    sweep_cfg: SweepConfig,
    run_dir: pathlib.Path,
) -> dict[str, Any]:
    return {
        "variant": row.variant,
        "seed": row.seed,
        "overrides": dict(row.overrides),
        "preset": sweep_cfg.preset,
        "max_steps": sweep_cfg.max_steps,
        "eval_planner_mode": sweep_cfg.eval_planner_mode,
        "ablation_cell": row.ablation_cell,
        "run_tag": row.run_tag,
        "config_snapshot_path": str(run_dir / "config.yaml"),
        "eval_report_path": str(run_dir / "eval_report.json"),
        "checkpoint_path": str(run_dir / "ckpt_final.pt"),
        "tensorboard_dir": str(run_dir / "tb"),
    }


def _tail_bytes(path: pathlib.Path, n_bytes: int = 8192) -> bytes:
    """Return the last ``n_bytes`` of a file (for failure_reason extraction)."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            if size > n_bytes:
                f.seek(size - n_bytes)
            return f.read()
    except OSError:
        return b""


def _spawn_row(
    row: SweepRow,
    sweep_cfg: SweepConfig,
    run_dir: pathlib.Path,
    gpu_id: int | None,
    repo_root: pathlib.Path,
    *,
    extra_env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    # Unbuffered child so its prints land in train.log promptly (tail -f works).
    env.setdefault("PYTHONUNBUFFERED", "1")
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if extra_env:
        env.update({str(k): str(v) for k, v in extra_env.items()})
    payload = _build_payload(row, sweep_cfg, run_dir)
    payload_bytes = json.dumps(payload).encode("utf-8")
    # Stream the child's combined stdout+stderr to a per-run train.log (mirrors
    # the 2agent_06c layout) so a long run is observable via `tail -f`. The
    # in-memory PIPE approach hid all progress and lost the log on success.
    log_path = run_dir / "train.log"
    header = (
        f"# {row.variant} seed={row.seed} row={row.sweep_row_index} "
        f"gpu={gpu_id} preset={sweep_cfg.preset} max_steps={sweep_cfg.max_steps}\n"
    ).encode("utf-8")
    with open(log_path, "wb") as logf:
        logf.write(header)
        logf.flush()
        proc = subprocess.Popen(
            [sys.executable, "-m", "hyper_mve.experiments._sweep_worker"],
            stdin=subprocess.PIPE,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(repo_root),
        )
        # communicate(input=...) writes+closes stdin itself; stdout/stderr are
        # redirected to the file (not PIPEs), so it returns (None, None) for
        # them and just waits. Avoids the "flush of closed file" stdin bug.
        proc.communicate(input=payload_bytes)
    # Read the log tail for the registry failure_reason on non-zero exit.
    stderr_tail = _tail_bytes(log_path, 8192)
    return subprocess.CompletedProcess(
        args=proc.args, returncode=proc.returncode,
        stdout=b"", stderr=stderr_tail,
    )


def _exit_code_to_status(returncode: int) -> tuple[str, str | None]:
    """Map worker exit code → (status, failure_reason)."""
    if returncode == 0:
        return "completed", None
    if returncode == 1:
        return "failed", "exit_code=1 (Python exception; see stderr)"
    if returncode == 2:
        return "skipped", "exit_code=2 (NotImplementedError stub)"
    if returncode == 139:
        return "failed", "exit_code=139 (SIGSEGV)"
    if returncode < 0:
        return "failed", f"exit_code={returncode} (terminated by signal {-returncode})"
    return "failed", f"exit_code={returncode}"


def _last_stderr_line(stderr: bytes) -> str | None:
    if not stderr:
        return None
    text = stderr.decode("utf-8", errors="replace").strip()
    if not text:
        return None
    return text.splitlines()[-1][:200]


def _materialise_summary(payload: dict[str, Any]) -> dict[str, float | None]:
    """Read EvalReport from disk and extract the 3 summary fields."""
    out: dict[str, float | None] = {
        "return_mean": None,
        "return_zero_shot_unseen": None,
        "regret_mean": None,
    }
    p = pathlib.Path(payload.get("eval_report_path", ""))
    if not p.exists():
        return out
    try:
        report = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return out
    for key in out:
        v = report.get(key)
        if v is not None:
            try:
                out[key] = float(v)
            except (TypeError, ValueError):
                out[key] = None
    return out


# ===== Resume logic (spec 05 §8.2) =======================================

def _dedup_key(row: SweepRow) -> tuple[str, int, str]:
    return (row.variant, row.seed, row.config_hash)


def _resume_skip(
    row: SweepRow,
    latest_by_run: dict[str, dict],
    retry_failed: bool,
) -> tuple[bool, str]:
    """Return (skip, reason). Match by ``(variant, seed, config_hash)``."""
    matches = [
        r for r in latest_by_run.values()
        if r.get("variant") == row.variant
        and int(r.get("seed", -1)) == row.seed
        and r.get("config_hash") == row.config_hash
    ]
    if not matches:
        return False, ""
    matches.sort(key=lambda r: r.get("started_at_iso8601", ""), reverse=True)
    latest = matches[0]
    status = latest.get("status")
    if status == "completed":
        return True, f"completed at {latest.get('completed_at_iso8601')}"
    if status == "skipped":
        return True, "stub (NotImplementedError)"
    if status == "failed" and not retry_failed:
        return True, "failed (re-run with --retry-failed to re-enqueue)"
    return False, ""


# ===== run_sweep (spec 05 §2 / §5 / §7 / §8) =============================

def _resolve_repo_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    # hyper_mve/experiments/sweep.py → repo_root = parents[2]
    return here.parents[2]


def run_sweep(
    sweep_cfg: SweepConfig,
    *,
    runs_root: pathlib.Path | str = "runs",
    registry_path: pathlib.Path | str | None = None,
    max_parallel: int | None = None,
    n_gpus: int | None = None,
    gpu_ids: Sequence[int] | None = None,
    slots_per_gpu: int = 1,
    child_env: Mapping[str, str] | None = None,
    dry_run: bool = False,
    retry_failed: bool = False,
    repo_root: pathlib.Path | None = None,
    shared_sem: "GpuSemaphore | MultiSlotGpuSemaphore | None" = None,
) -> list[RegistryRow]:
    """Schedule and run a sweep.

    Returns the list of final-state RegistryRow objects (one per processed
    row). Skipped-because-resume rows return ``status="completed"`` (or the
    prior terminal status) verbatim from the registry without re-running.

    Args:
        gpu_ids: Optional explicit list of physical CUDA device indices to
            schedule onto. When given, overrides ``n_gpus``; rows are spawned
            with ``CUDA_VISIBLE_DEVICES=<gpu_id>``.
        slots_per_gpu: Workers per physical device (default 1 → identical to
            legacy ``GpuSemaphore`` behaviour). When ``> 1``, the harness
            uses :class:`MultiSlotGpuSemaphore` instead.
        child_env: Extra env vars merged into every spawned worker's
            environment. Use for runtime tuning knobs that the worker reads
            at startup (e.g. ``HYPER_MVE_GPU_MEM_FRAC``).
    """
    repo_root = pathlib.Path(repo_root) if repo_root is not None else _resolve_repo_root()
    runs_root = pathlib.Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    if registry_path is None:
        registry_path = runs_root / "registry.jsonl"
    registry = RunRegistry(registry_path)

    rows = enumerate_cartesian(sweep_cfg)
    if not rows:
        warnings.warn(
            "run_sweep: enumerate_cartesian returned 0 rows; nothing to do",
            stacklevel=2,
        )
        return []

    # Compute config_hash + run_tag per row. The full V4Config materialisation
    # for the hash is the worker's responsibility; row_config_hash() hashes a
    # deterministic surrogate so resume detection works without importing torch
    # in the parent (and run_suite --force can recompute the same key).
    rows = [
        replace(
            row,
            config_hash=row_config_hash(sweep_cfg, row),
            run_id=uuid.uuid4().hex,
        )
        for row in rows
    ]
    rows = [replace(row, run_tag=_build_run_tag(row)) for row in rows]

    # Resume: load existing registry once.
    existing = registry.read_all()
    from .run_registry import load_latest_per_run_id
    latest_by_run = load_latest_per_run_id(existing)

    # Resource clamps + semaphore selection.
    if slots_per_gpu < 1:
        raise ValueError(f"slots_per_gpu must be >= 1; got {slots_per_gpu}")
    if gpu_ids is not None:
        gpu_ids_tuple = tuple(int(g) for g in gpu_ids)
        if not gpu_ids_tuple:
            raise ValueError("gpu_ids must be non-empty when provided")
        n_total_slots = len(gpu_ids_tuple) * slots_per_gpu
    else:
        n_gpus_resolved = max(1, int(n_gpus if n_gpus is not None else sweep_cfg.n_gpus))
        gpu_ids_tuple = tuple(range(n_gpus_resolved))
        n_total_slots = n_gpus_resolved * slots_per_gpu
    if shared_sem is not None:
        # A caller-supplied semaphore is the GLOBAL concurrency cap — e.g.
        # run_suite cross-cell mode pools rows from many cells through ONE pool.
        # Size this cell's view to the shared pool so its rows queue on the
        # shared tokens (rather than each cell capping itself at its own slots).
        n_total_slots = shared_sem.n_gpus
    max_parallel = max(1, int(max_parallel if max_parallel is not None else sweep_cfg.max_parallel))
    max_parallel = min(max_parallel, n_total_slots, len(rows))

    if dry_run:
        skipped = sum(1 for row in rows if _resume_skip(row, latest_by_run, retry_failed)[0])
        return _dry_run_emit(
            sweep_cfg, rows, skipped, max_parallel,
            n_total_slots, gpu_ids_tuple, slots_per_gpu,
        )

    git_sha = _git_sha(repo_root)
    git_dirty = _git_dirty(repo_root)

    if shared_sem is not None:
        sem: GpuSemaphore | MultiSlotGpuSemaphore = shared_sem
    elif slots_per_gpu > 1 or gpu_ids is not None:
        sem = MultiSlotGpuSemaphore(
            gpu_ids_tuple, slots_per_gpu=slots_per_gpu,
        )
    else:
        sem = GpuSemaphore(len(gpu_ids_tuple))

    final_rows: list[RegistryRow] = []
    final_lock = threading.Lock()

    def _process_one_row(row: SweepRow) -> RegistryRow | None:
        """Spawn one subprocess for ``row``; append pending + terminal registry rows.

        Returns the terminal RegistryRow on success, or ``None`` if the row was
        skipped via resume. Exceptions inside the worker are mapped to a failed
        registry row; only catastrophic parent-side failures propagate.
        """
        skip, reason = _resume_skip(row, latest_by_run, retry_failed)
        if skip:
            print(
                f"[skip] row {row.sweep_row_index} ({row.variant}, seed={row.seed}): {reason}",
                flush=True,
            )
            return None

        run_dir = runs_root / row.run_tag
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "tb").mkdir(parents=True, exist_ok=True)
        started_at = _now_iso8601()

        gpu_id = sem.acquire()
        try:
            pending_row = _build_registry_row(
                row, sweep_cfg, run_dir, git_sha, git_dirty,
                gpu_id, started_at, status="pending", failure_reason=None,
                completed_at=None, walltime=None, peak_memory_mb=None,
                summary={"return_mean": None, "return_zero_shot_unseen": None, "regret_mean": None},
            )
            registry.append_row(pending_row)
            print(
                f"[start] row {row.sweep_row_index} ({row.variant}, seed={row.seed}) "
                f"gpu={gpu_id} | tail -f {run_dir / 'train.log'}",
                flush=True,
            )
            t0 = datetime.now(timezone.utc)
            try:
                proc = _spawn_row(
                    row, sweep_cfg, run_dir, gpu_id, repo_root,
                    extra_env=child_env,
                )
                returncode = proc.returncode
                stderr = proc.stderr or b""
            except Exception as e:  # parent-side spawn error
                returncode = 1
                stderr = f"FAIL parent spawn: {type(e).__name__}: {e}".encode("utf-8")
            walltime = (datetime.now(timezone.utc) - t0).total_seconds()
            status, default_reason = _exit_code_to_status(returncode)
            stderr_tail = _last_stderr_line(stderr)
            failure_reason = stderr_tail if (status != "completed" and stderr_tail) else default_reason
            summary = (
                _materialise_summary(_build_payload(row, sweep_cfg, run_dir))
                if status == "completed"
                else {"return_mean": None, "return_zero_shot_unseen": None, "regret_mean": None}
            )
            terminal_row = _build_registry_row(
                row, sweep_cfg, run_dir, git_sha, git_dirty,
                gpu_id, started_at, status=status, failure_reason=failure_reason,
                completed_at=_now_iso8601(), walltime=walltime, peak_memory_mb=None,
                summary=summary,
            )
            registry.append_row(terminal_row)
            with final_lock:
                final_rows.append(terminal_row)
            ret_str = (
                f" return_mean={summary['return_mean']:.3f}"
                if summary.get("return_mean") is not None else ""
            )
            reason_str = f" | {failure_reason}" if status != "completed" and failure_reason else ""
            print(
                f"[done]  row {row.sweep_row_index} ({row.variant}, seed={row.seed}) "
                f"{status} in {walltime:.1f}s{ret_str}{reason_str}",
                flush=True,
            )
            return terminal_row
        finally:
            sem.release(gpu_id)

    # Thread-pool drain: each worker thread acquires one GPU slot and spawns
    # its own subprocess. Lock 1 is preserved (one fresh Python interpreter
    # per row); Lock 2 is preserved (RunRegistry.append_row is OS-file-locked
    # and thread-safe; see tests/experiments/test_run_registry_jsonl_append_atomic.py).
    if max_parallel <= 1:
        for row in rows:
            _process_one_row(row)
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_parallel,
            thread_name_prefix="sweep",
        ) as pool:
            futures = [pool.submit(_process_one_row, row) for row in rows]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    fut.result()
                except Exception as e:  # pragma: no cover - parent-side dispatch fault
                    print(f"[sweep] worker thread raised {type(e).__name__}: {e}",
                          file=sys.stderr, flush=True)
    return final_rows


def _build_registry_row(
    row: SweepRow,
    sweep_cfg: SweepConfig,
    run_dir: pathlib.Path,
    git_sha: str,
    git_dirty: bool,
    gpu_id: int | None,
    started_at: str,
    *,
    status: str,
    failure_reason: str | None,
    completed_at: str | None,
    walltime: float | None,
    peak_memory_mb: float | None,
    summary: Mapping[str, float | None],
) -> RegistryRow:
    return RegistryRow(
        run_id=row.run_id,
        variant=row.variant,
        seed=row.seed,
        config_hash=row.config_hash,
        ablation_cell=row.ablation_cell,
        sweep_row_index=row.sweep_row_index,
        git_sha=git_sha,
        git_dirty=git_dirty,
        started_at_iso8601=started_at,
        completed_at_iso8601=completed_at,
        status=status,  # type: ignore[arg-type]
        failure_reason=failure_reason,
        gpu_id=gpu_id,
        walltime_seconds=walltime,
        peak_gpu_memory_mb=peak_memory_mb,
        config_snapshot_path=str(run_dir / "config.yaml"),
        checkpoint_path=str(run_dir / "ckpt_final.pt") if status == "completed" else None,
        eval_report_path=str(run_dir / "eval_report.json") if status == "completed" else None,
        tensorboard_dir=str(run_dir / "tb"),
        return_mean=summary.get("return_mean"),
        return_zero_shot_unseen=summary.get("return_zero_shot_unseen"),
        regret_mean=summary.get("regret_mean"),
    )


def _dry_run_emit(
    sweep_cfg: SweepConfig,
    rows: Sequence[SweepRow],
    skipped: int,
    max_parallel: int,
    n_total_slots: int,
    gpu_ids: Sequence[int],
    slots_per_gpu: int,
) -> list[RegistryRow]:
    print(
        f"[dry-run] SweepConfig: variants={len(sweep_cfg.variants)}, "
        f"seeds={len(sweep_cfg.seeds)}, "
        f"overrides={len(sweep_cfg.overrides) or 1}, total={len(rows)} rows"
    )
    print(f"[dry-run] eval_planner_mode={sweep_cfg.eval_planner_mode!r}, "
          f"preset={sweep_cfg.preset!r}, max_steps={sweep_cfg.max_steps}")
    print(f"[dry-run] ablation_cell_id={sweep_cfg.ablation_cell_id!r}")
    print(f"[dry-run] {len(rows) - skipped} rows would be spawned (skipping {skipped} resumed)")
    print(
        f"[dry-run] GPU pool: gpu_ids={tuple(gpu_ids)} slots_per_gpu={slots_per_gpu} "
        f"effective_concurrent={n_total_slots} max_parallel={max_parallel}"
    )
    return []


# ===== CLI (spec 05 §8.1) ================================================

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m hyper_mve.experiments.sweep",
        description="Run a sweep over (variant, seed, override) cartesian.",
    )
    p.add_argument("--config", type=pathlib.Path, required=True,
                   help="Path to a SweepConfig YAML.")
    p.add_argument("--max-parallel", dest="max_parallel", type=int, default=None)
    p.add_argument("--n-gpus", dest="n_gpus", type=int, default=None)
    p.add_argument("--dry-run", dest="dry_run", action="store_true")
    p.add_argument("--retry-failed", dest="retry_failed", action="store_true")
    p.add_argument("--registry-path", dest="registry_path", type=pathlib.Path,
                   default=None, help="Override registry JSONL location.")
    p.add_argument("--runs-root", dest="runs_root", type=pathlib.Path,
                   default=pathlib.Path("runs"))
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    sweep_cfg = SweepConfig.from_yaml(args.config)
    rows = run_sweep(
        sweep_cfg,
        runs_root=args.runs_root,
        registry_path=args.registry_path,
        max_parallel=args.max_parallel,
        n_gpus=args.n_gpus,
        dry_run=args.dry_run,
        retry_failed=args.retry_failed,
    )
    n_completed = sum(1 for r in rows if r.status == "completed")
    n_failed = sum(1 for r in rows if r.status == "failed")
    n_skipped = sum(1 for r in rows if r.status == "skipped")
    print(f"sweep done: completed={n_completed} failed={n_failed} skipped={n_skipped}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
