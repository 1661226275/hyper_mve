"""Phase 6 integration drift detectors — cross-package byte-identity locks.

Runs without torch. Validates that pkg-07 + pkg-08 still agree on the four
spine invariants:

* `EvalReport` exposes 33 dataclass fields with `schema_version="pkg08-spec01-v1"`.
* `RegistryRow` exposes 23 dataclass fields with `schema_version="pkg08-spec05-v1"`.
* `REGISTRY` keys (11) match `CLI_CHOICES` (14, after curriculum-override removal).
* `cli_to_factory_arg` is total over `CLI_CHOICES` minus curriculum-overrides.
"""
from __future__ import annotations

from dataclasses import fields

import pytest


def test_eval_report_33_field_dataclass_lock():
    """pkg-08 spec 01 Lock 2 — 32 payload fields + 1 `schema_version` sentinel."""
    pytest.importorskip("torch")  # EvalReport pulls torch transitively
    from hyper_mve.eval import EvalReport
    fld = tuple(f.name for f in fields(EvalReport))
    assert len(fld) == 33, f"EvalReport drift: {len(fld)} fields (expect 33)"
    assert fld[-1] == "schema_version"
    sentinel_field = fields(EvalReport)[-1]
    assert sentinel_field.default == "pkg08-spec01-v1"


def test_registry_row_23_field_dataclass_lock():
    """pkg-08 spec 05 Lock 3 — 22 schema-domain + 1 `schema_version` sentinel."""
    from hyper_mve.experiments.run_registry import RegistryRow, SCHEMA_VERSION
    fld = tuple(f.name for f in fields(RegistryRow))
    assert len(fld) == 23, f"RegistryRow drift: {len(fld)} fields (expect 23)"
    assert fld[-1] == "schema_version"
    sentinel_field = fields(RegistryRow)[-1]
    assert sentinel_field.default == "pkg08-spec05-v1"
    assert SCHEMA_VERSION == "pkg08-spec05-v1"


def test_registry_keys_equal_cli_choices_minus_curriculum():
    """pkg-07 spec 01 §2 — REGISTRY keys ∪ curriculum-overrides == CLI_CHOICES."""
    from hyper_mve.baselines import CLI_CHOICES, REGISTRY
    curriculum = {"hyper", "oracle_only", "infer_only"}
    expected = set(CLI_CHOICES)
    assert set(REGISTRY) | curriculum == expected, (
        f"REGISTRY ∪ curriculum drift vs CLI_CHOICES: "
        f"extra={(set(REGISTRY) | curriculum) - expected}, "
        f"missing={expected - (set(REGISTRY) | curriculum)}"
    )
    assert len(REGISTRY) == 11
    assert len(CLI_CHOICES) == 14


def test_cli_to_factory_arg_total_over_non_curriculum_choices():
    """pkg-07 spec 01 §5 — cli_to_factory_arg is total over CLI minus curriculum."""
    from hyper_mve.baselines import CLI_CHOICES, REGISTRY, cli_to_factory_arg
    curriculum = {"hyper", "oracle_only", "infer_only"}
    for cli in CLI_CHOICES:
        if cli in curriculum:
            with pytest.raises(ValueError, match="curriculum-override"):
                cli_to_factory_arg(cli)
        else:
            arg = cli_to_factory_arg(cli)
            assert arg in REGISTRY, (
                f"cli_to_factory_arg({cli!r}) -> {arg!r} not in REGISTRY"
            )


def test_planner_mode_literal_4tuple():
    """pkg-08 spec 03 Lock 1 — exactly 4 planner modes."""
    from hyper_mve.experiments.sweep import PlannerMode
    import typing
    args = typing.get_args(PlannerMode)
    assert set(args) == {
        "direct_inference", "planner_no_crn",
        "planner_no_coord_desc", "planner_full",
    }
    assert len(args) == 4


def test_ablation_ids_5_tuple():
    """pkg-08 spec 06 Lock 1 — 5 canned IDs (4 logical cells)."""
    from hyper_mve.experiments.ablate import ABLATION_IDS
    assert ABLATION_IDS == (
        "abl1", "abl4_crn_joint", "abl4_joint_easy_n2", "abl6", "abl7",
    )


def test_disclosure_columns_10_tuple():
    """pkg-07 spec 07 §4.2 — 10 disclosure columns verbatim."""
    from hyper_mve.experiments.stats import DISCLOSURE_COLUMNS
    assert DISCLOSURE_COLUMNS == (
        "variant", "preset", "param_count", "walltime_to_converge_seconds",
        "lr_swept_best", "final_return_mean", "final_return_sem",
        "seeds_run", "smoke_pass", "sourced",
    )
    assert len(DISCLOSURE_COLUMNS) == 10
