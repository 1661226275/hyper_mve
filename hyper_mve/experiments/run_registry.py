"""pkg-08 spec 05 §4 — RunRegistry: 23-key JSONL append-only run registry.

Schema mirrored byte-identically in pkg-08 spec 08 §4 (Lock 3); the 23-key
breakdown is **22 schema-domain fields + 1 schema_version sentinel**:

    Identity (5)        run_id / variant / seed / config_hash / ablation_cell
    Provenance (4)      sweep_row_index / git_sha / git_dirty / started_at_iso8601
    Lifecycle (3)       completed_at_iso8601 / status / failure_reason
    Resource (3)        gpu_id / walltime_seconds / peak_gpu_memory_mb
    Output ptrs (3)     config_snapshot_path / checkpoint_path / eval_report_path
    TB pointer (1)      tensorboard_dir
    Summary (3)         return_mean / return_zero_shot_unseen / regret_mean
    Sentinel (1)        schema_version  = "pkg08-spec05-v1"

The append-only contract (Lock 2): every state transition (pending → running →
completed/failed/skipped) is a NEW line appended to ``runs/registry.jsonl``;
the file is never rewritten or truncated. Concurrent appenders (parent +
each subprocess child) coordinate via OS file lock — ``msvcrt.locking`` on
Win32, ``fcntl.flock`` on POSIX — held on the JSONL file's own fd for the
duration of one ``write + flush + fsync`` cycle.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from dataclasses import asdict, dataclass, fields
from typing import Iterable, Literal, Sequence

__all__ = ["RegistryRow", "RunRegistry", "SCHEMA_VERSION"]


# pkg-08 spec 05 §3.1 + §4 / spec 08 §4 — synchronous-edit sentinel.
SCHEMA_VERSION: str = "pkg08-spec05-v1"


@dataclass(frozen=True)
class RegistryRow:
    """One line of ``runs/registry.jsonl``.

    Schema is mirrored byte-identically in pkg-08 spec 08 §4 (Lock 3); count
    the dataclass body line-by-line — 22 schema-domain fields + 1
    ``schema_version`` sentinel = 23 keys.
    """

    # === Identity (5) ===
    run_id: str
    variant: str
    seed: int
    config_hash: str
    ablation_cell: str | None

    # === Provenance (4) ===
    sweep_row_index: int
    git_sha: str
    git_dirty: bool
    started_at_iso8601: str

    # === Lifecycle (3) ===
    completed_at_iso8601: str | None
    status: Literal["pending", "running", "completed", "failed", "skipped"]
    failure_reason: str | None

    # === Resource (3) ===
    gpu_id: int | None
    walltime_seconds: float | None
    peak_gpu_memory_mb: float | None

    # === Output pointers (3) ===
    config_snapshot_path: str
    checkpoint_path: str | None
    eval_report_path: str | None

    # === TensorBoard pointer (1) ===
    tensorboard_dir: str

    # === Duplicated summary metrics (3) — populated iff status == "completed" ===
    return_mean: float | None
    return_zero_shot_unseen: float | None
    regret_mean: float | None

    # === Schema sentinel (1) — bumped on synchronous spec 05 §4 + spec 08 §4 edit ===
    schema_version: str = SCHEMA_VERSION


# === File-lock primitives (pkg-08 spec 05 §4.3) ==========================

if sys.platform == "win32":
    import msvcrt

    def _lock_exclusive(fd: int) -> None:
        # msvcrt has no truly-blocking append-friendly lock; poll with backoff
        # then fall back to LK_LOCK (which can spuriously raise on contention).
        for delay in (0.001, 0.005, 0.02, 0.1, 0.5):
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                time.sleep(delay)
        msvcrt.locking(fd, msvcrt.LK_LOCK, 1)

    def _unlock(fd: int) -> None:
        try:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
else:
    import fcntl  # type: ignore[import-not-found]

    def _lock_exclusive(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX)

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


def _serialise_row(row: RegistryRow) -> bytes:
    return (json.dumps(asdict(row), separators=(",", ":")) + "\n").encode("utf-8")


# === RunRegistry class (pkg-08 spec 05 §4.2 / §4.3) ======================

class RunRegistry:
    """Append-only JSONL handle.

    Cheap to construct — does not open the file at construction time. Every
    ``append_row`` call opens, locks, writes, flushes, fsyncs, unlocks, and
    closes the fd. The lock window is microseconds (one write + fsync).
    """

    def __init__(self, registry_path: pathlib.Path | str) -> None:
        self._path = pathlib.Path(registry_path)

    @property
    def path(self) -> pathlib.Path:
        return self._path

    def append_row(self, row: RegistryRow) -> None:
        """Atomically append one row to ``runs/registry.jsonl``.

        Race-safe across threads in a single process and across subprocess
        children of a sweep harness invocation. The 16-thread × 100-row
        stress test (see ``tests/experiments/test_run_registry_jsonl_append_atomic``)
        completes in well under 1 s with exactly 1600 newline bytes.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = _serialise_row(row)
        with open(self._path, "ab") as f:
            _lock_exclusive(f.fileno())
            try:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            finally:
                _unlock(f.fileno())

    def append_rows(self, rows: Iterable[RegistryRow]) -> None:
        for row in rows:
            self.append_row(row)

    def read_all(self) -> list[dict]:
        """Read every line, returning the parsed JSON dicts in file order.

        No groupby / status filter — callers that want "the latest state per
        run_id" should use :func:`load_latest_per_run_id`.
        """
        if not self._path.exists():
            return []
        out: list[dict] = []
        with open(self._path, "rb") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                out.append(json.loads(line.decode("utf-8")))
        return out


def load_latest_per_run_id(rows: Sequence[dict]) -> dict[str, dict]:
    """Reduce a JSONL row sequence to ``{run_id: latest_row}``.

    Per pkg-08 spec 05 §4.2: rows for one ``run_id`` form an append-only
    history (pending → running → completed/failed). The latest row by
    ``started_at_iso8601`` is the canonical state.
    """
    out: dict[str, dict] = {}
    for r in rows:
        rid = r["run_id"]
        if rid not in out or r.get("started_at_iso8601", "") >= out[rid].get("started_at_iso8601", ""):
            out[rid] = r
    return out


def field_names() -> tuple[str, ...]:
    """Return the 23-tuple of RegistryRow field names in definition order."""
    return tuple(f.name for f in fields(RegistryRow))
