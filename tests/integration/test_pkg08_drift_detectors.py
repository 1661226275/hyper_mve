"""Integration drift detectors — cross-module byte-identity locks (phase-2).

Runs without heavyweight imports where possible. Validates the spine
invariants of the realigned layout:

* `EvalReport` exposes 28 dataclass fields with `schema_version="rel-v1"`.
* `RegistryRow` exposes 23 dataclass fields with `schema_version="pkg08-spec05-v1"`.
* `hyper_mve.comparison.REGISTRY` is the lazy string registry with exactly the
  phase-registered keys (extended by realignment phases 4–6; end state 7 keys).
* The retired v5 namespaces stay deleted.
* Disclosure columns stay verbatim.
"""
from __future__ import annotations

from dataclasses import fields

import pytest


def test_eval_report_28_field_dataclass_lock():
    """rel-v1 (v5 Pkg-09) — 27 payload fields + 1 `schema_version` sentinel.

    Phase-2 note: world-model fidelity is a SEPARATE artifact (fidelity-v1
    JSON, phase 7) precisely so this lock never moves for it.
    """
    pytest.importorskip("torch")  # EvalReport pulls torch transitively
    from hyper_mve.utils.eval import EvalReport
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
    from hyper_mve.utils.analysis.run_registry import RegistryRow, SCHEMA_VERSION
    fld = tuple(f.name for f in fields(RegistryRow))
    assert len(fld) == 23, f"RegistryRow drift: {len(fld)} fields (expect 23)"
    assert fld[-1] == "schema_version"
    sentinel_field = fields(RegistryRow)[-1]
    assert sentinel_field.default == "pkg08-spec05-v1"
    assert SCHEMA_VERSION == "pkg08-spec05-v1"


def test_runner_registry_lazy_string_lock():
    """Phase-2 registry lock: lazy ``module:Class`` strings, no torch import.

    Registered keys per phase: 1-3 → {mazero_mixed, mappo, mamba};
    phase 4 adds happo; phase 5 adds mbom + mbom_oracle; phase 6 adds
    m3w_adapted (end state 7). Update this lock in the SAME commit as the
    registry change.
    """
    import sys

    from hyper_mve.comparison import REGISTRY

    assert sorted(REGISTRY) == ["happo", "mamba", "mappo", "mazero_mixed"]
    for key, target in REGISTRY.items():
        module_name, sep, class_name = target.partition(":")
        assert sep == ":", f"REGISTRY[{key!r}] not in module:Class form: {target!r}"
        assert module_name.startswith("hyper_mve.")
        assert class_name.isidentifier()
    # No retired namespaces in keys.
    assert not any(k.startswith("external_") for k in REGISTRY)
    # Lazy: enumerating the registry must not have imported torch-heavy
    # runner modules (mappo/mamba pull torch at module import).
    assert "hyper_mve.comparison.mappo" not in sys.modules or "torch" in sys.modules


def test_retired_v5_namespaces_stay_deleted():
    """Phase-2 cleanup lock — the v5 stack must not regrow."""
    import importlib.util

    for gone in (
        "hyper_mve.training",
        "hyper_mve.planning",
        "hyper_mve.baselines",
        "hyper_mve.experiments",
        "hyper_mve._legacy_v4_7",
        "hyper_mve.algo.modules.hyper_muzero_model",
        "hyper_mve.algo.modules.transition_net",
        "hyper_mve.algo.modules.representation_net",
    ):
        assert importlib.util.find_spec(gone) is None, f"{gone} regrew"


def test_disclosure_columns_10_tuple():
    """pkg-07 spec 07 §4.2 — 10 disclosure columns verbatim."""
    from hyper_mve.utils.analysis.stats import DISCLOSURE_COLUMNS
    assert DISCLOSURE_COLUMNS == (
        "variant", "preset", "param_count", "walltime_to_converge_seconds",
        "lr_swept_best", "final_return_mean", "final_return_sem",
        "seeds_run", "smoke_pass", "sourced",
    )
    assert len(DISCLOSURE_COLUMNS) == 10
