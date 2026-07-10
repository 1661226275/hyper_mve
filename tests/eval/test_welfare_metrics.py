"""Thesis welfare metrics (Ch3.8.3 / Table 6.1) added to the eval pipeline.

Covers the fairness formula and the EvalReport schema extension. The full
end-to-end populate (run_eval → unified_evaluator) needs torch + the env stack
and is exercised by the suite smoke (`run_suite --only rel_gate_duo
--max-steps 100`) in the runtime checklist.
"""
from __future__ import annotations

from dataclasses import fields

import math

import pytest


def test_phys_fairness_formula():
    """F = 1 − N·σ(w)/Σ(w) = 1 − coefficient-of-variation, with degenerate guards."""
    pytest.importorskip("torch")  # evaluation.py imports torch transitively
    from hyper_mve.training.evaluation import _phys_fairness

    # Perfectly equal harvest → σ=0 → F=1.
    assert _phys_fairness([1.0, 1.0, 1.0, 1.0]) == pytest.approx(1.0)
    # One agent takes everything (N=4): F = 1 − √3 ≈ -0.732 (allowed to be < 0).
    assert _phys_fairness([4.0, 0.0, 0.0, 0.0]) == pytest.approx(1.0 - math.sqrt(3.0), abs=1e-6)
    # Single agent → trivially fair.
    assert _phys_fairness([5.0]) == pytest.approx(1.0)
    # Nothing harvested (Σ≈0) → trivially fair (no welfare to divide).
    assert _phys_fairness([0.0, 0.0]) == pytest.approx(1.0)
    # Fairness is always ≤ 1.
    for w in ([2.0, 1.0, 0.5], [10.0, 0.0], [3.0, 3.0, 3.0]):
        assert _phys_fairness(w) <= 1.0 + 1e-9


def test_eval_report_welfare_fields_default_zero():
    """The 4 welfare fields exist with a 0.0 default (external runners use the default)."""
    pytest.importorskip("torch")  # EvalReport pulls torch transitively via the package
    from hyper_mve.eval import EvalReport

    defaults = {f.name: f.default for f in fields(EvalReport)}
    for name in (
        "welfare_physical_mean", "sustainability_mean",
        "fairness_mean", "tragedy_index_mean",
    ):
        assert name in defaults, f"EvalReport missing {name!r}"
        assert defaults[name] == 0.0, f"{name} default should be 0.0, got {defaults[name]!r}"
    assert defaults["schema_version"] == "rel-v1"
