"""C8-ABL-REG1 + Lock 3 — RegistryRow 23-key schema lock-test (pkg-08 spec 05 §4)."""
from __future__ import annotations

from dataclasses import fields

from hyper_mve.utils.analysis.run_registry import RegistryRow, SCHEMA_VERSION


# pkg-08 spec 05 §4 + spec 08 §4 byte-identical 23-tuple.
EXPECTED_FIELDS: tuple[str, ...] = (
    # Identity (5)
    "run_id", "variant", "seed", "config_hash", "ablation_cell",
    # Provenance (4)
    "sweep_row_index", "git_sha", "git_dirty", "started_at_iso8601",
    # Lifecycle (3)
    "completed_at_iso8601", "status", "failure_reason",
    # Resource (3)
    "gpu_id", "walltime_seconds", "peak_gpu_memory_mb",
    # Output pointers (3)
    "config_snapshot_path", "checkpoint_path", "eval_report_path",
    # TensorBoard pointer (1)
    "tensorboard_dir",
    # Duplicated summary metrics (3)
    "return_mean", "return_zero_shot_unseen", "regret_mean",
    # Schema sentinel (1)
    "schema_version",
)


def test_registry_row_has_23_fields():
    """C8-ABL-REG1 — 23 fields exactly."""
    actual = tuple(f.name for f in fields(RegistryRow))
    assert len(actual) == 23, f"RegistryRow drift: {len(actual)} fields, expected 23"


def test_registry_row_field_order_matches_spec():
    """Lock 3 — field-name tuple matches spec 05 §4 byte-identically."""
    actual = tuple(f.name for f in fields(RegistryRow))
    assert actual == EXPECTED_FIELDS


def test_schema_version_is_pkg08_spec05_v1():
    """schema_version sentinel is 'pkg08-spec05-v1' (spec 05 §3.1 + §4)."""
    assert SCHEMA_VERSION == "pkg08-spec05-v1"
    # Default value of the dataclass field also exposes the sentinel.
    sentinel_field = next(f for f in fields(RegistryRow) if f.name == "schema_version")
    assert sentinel_field.default == "pkg08-spec05-v1"
