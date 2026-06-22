"""C8-ABL-REG1 — JSONL append atomicity stress test (pkg-08 spec 05 §4.3 / §9)."""
from __future__ import annotations

import json
import threading

import pytest

from hyper_mve.experiments.run_registry import RegistryRow, RunRegistry, SCHEMA_VERSION


def _make_row(idx: int, run_id: str) -> RegistryRow:
    return RegistryRow(
        run_id=run_id,
        variant="hyper",
        seed=idx % 7,
        config_hash="0" * 32,
        ablation_cell=None,
        sweep_row_index=idx,
        git_sha="0" * 40,
        git_dirty=False,
        started_at_iso8601="2026-06-22T00:00:00Z",
        completed_at_iso8601=None,
        status="pending",
        failure_reason=None,
        gpu_id=None,
        walltime_seconds=None,
        peak_gpu_memory_mb=None,
        config_snapshot_path=f"runs/test_{idx}/config.yaml",
        checkpoint_path=None,
        eval_report_path=None,
        tensorboard_dir=f"runs/test_{idx}/tb",
        return_mean=None,
        return_zero_shot_unseen=None,
        regret_mean=None,
    )


def test_append_atomic_16_threads_x_100_rows(tmp_path):
    """C8-ABL-REG1 — exactly 1600 lines, no JSON parse errors, no interleave."""
    registry_path = tmp_path / "registry.jsonl"
    registry = RunRegistry(registry_path)

    n_threads = 16
    rows_per_thread = 100

    def worker(tid: int) -> None:
        for i in range(rows_per_thread):
            registry.append_row(_make_row(tid * rows_per_thread + i, f"r{tid}_{i}"))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    text = registry_path.read_bytes()
    assert text.count(b"\n") == n_threads * rows_per_thread, (
        f"newline count drift: got {text.count(chr(10).encode())}, "
        f"expected {n_threads * rows_per_thread}"
    )
    rows = []
    with open(registry_path, "rb") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            rows.append(json.loads(line.decode("utf-8")))
    assert len(rows) == n_threads * rows_per_thread
    for row in rows:
        assert row["schema_version"] == SCHEMA_VERSION
        assert "run_id" in row and "variant" in row
