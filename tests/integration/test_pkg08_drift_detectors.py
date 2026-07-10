"""Phase 6 integration drift detectors — cross-package byte-identity locks.

Runs without torch. Validates that pkg-07 + pkg-08 still agree on the four
spine invariants:

* `EvalReport` exposes 28 dataclass fields with `schema_version="rel-v1"`.
* `RegistryRow` exposes 23 dataclass fields with `schema_version="pkg08-spec05-v1"`.
* `REGISTRY` keys (10) match `CLI_CHOICES` (13, after curriculum-override removal).
* `cli_to_factory_arg` is total over `CLI_CHOICES` minus curriculum-overrides.
"""
from __future__ import annotations

from dataclasses import fields

import pytest


def test_eval_report_28_field_dataclass_lock():
    """rel-v1 (v5 Pkg-09) — 27 payload fields + 1 `schema_version` sentinel.

    The v4 per-c / c-segment / type-ratio / regret machinery went with c_t and
    the type system; per-regime breakdown + belief regime-quality replace them.
    """
    pytest.importorskip("torch")  # EvalReport pulls torch transitively
    from hyper_mve.eval import EvalReport
    fld = tuple(f.name for f in fields(EvalReport))
    assert len(fld) == 28, f"EvalReport drift: {len(fld)} fields (expect 28)"
    assert fld[-1] == "schema_version"
    for name in (
        "return_per_regime", "return_per_regime_sem", "episodes_per_regime",
        "regime_accuracy", "regime_nll",
        "welfare_physical_mean", "sustainability_mean",
        "fairness_mean", "tragedy_index_mean",
    ):
        assert name in fld, f"EvalReport missing field {name!r}"
    for gone in ("c_visible", "return_per_c", "regret_mean", "return_per_segment",
                 "return_per_type_ratio", "belief_c_mae"):
        assert gone not in fld, f"EvalReport regrew v4 field {gone!r}"
    sentinel_field = fields(EvalReport)[-1]
    assert sentinel_field.default == "rel-v1"


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
    """pkg-07 spec 01 §2 — every non-curriculum CLI choice resolves into REGISTRY.

    The naive ``set(REGISTRY) | curriculum == set(CLI_CHOICES)`` invariant is FALSE
    by design: per spec §2.2 the CLI uses ``baseline_*`` prefixes for internal
    "with-shared-backbone" variants (``baseline_input_wide``) while REGISTRY uses
    the bare factory arg (``input_wide``). The real invariant is that the bridge
    function :func:`cli_to_factory_arg` is total over ``CLI_CHOICES - curriculum``
    and lands in REGISTRY, with the cardinalities checked separately.
    """
    from hyper_mve.baselines import CLI_CHOICES, REGISTRY, cli_to_factory_arg
    curriculum = {"hyper", "oracle_only", "infer_only"}
    non_curriculum = set(CLI_CHOICES) - curriculum
    resolved = {cli_to_factory_arg(c) for c in non_curriculum}
    assert resolved == set(REGISTRY), (
        f"CLI → factory drift: extra={resolved - set(REGISTRY)}, "
        f"missing={set(REGISTRY) - resolved}"
    )
    assert len(REGISTRY) == 10   # v5: rewardhead_explicit_type deleted (Pkg-09)
    assert len(CLI_CHOICES) == 13


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
