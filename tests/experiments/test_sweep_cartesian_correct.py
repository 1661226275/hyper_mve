"""C8-ABL-SWEEP1 — cartesian semantics + edge cases (pkg-08 spec 05 §3.3)."""
from __future__ import annotations

import pytest

from hyper_mve.experiments.sweep import SweepConfig, enumerate_cartesian


@pytest.mark.parametrize("V,S,O,expected", [
    (1, 1, 0, 1),    # empty-overrides edge case → 1 empty-dict cell
    (1, 5, 0, 5),
    (3, 5, 4, 60),
    (14, 5, 0, 70),
])
def test_cartesian_cardinality(V, S, O, expected):
    sweep = SweepConfig(
        variants=tuple(f"v{i}" for i in range(V)),
        seeds=tuple(range(S)),
        overrides=tuple({"k": i} for i in range(O)),
        preset="medium",
        max_steps=100,
    )
    rows = enumerate_cartesian(sweep)
    assert len(rows) == expected


def test_cartesian_lexicographic_order():
    sweep = SweepConfig(
        variants=("a", "b"),
        seeds=(0, 1),
        overrides=({"k": 0}, {"k": 1}),
        preset="medium", max_steps=10,
    )
    rows = enumerate_cartesian(sweep)
    # Order: (a,0,0), (a,0,1), (a,1,0), (a,1,1), (b,0,0), ...
    assert [(r.variant, r.seed) for r in rows[:4]] == [("a", 0), ("a", 0), ("a", 1), ("a", 1)]
    assert [(r.variant, r.seed) for r in rows[4:]] == [("b", 0), ("b", 0), ("b", 1), ("b", 1)]
    # sweep_row_index is dense.
    assert [r.sweep_row_index for r in rows] == list(range(len(rows)))


def test_empty_overrides_collapses_to_single_empty_cell():
    sweep = SweepConfig(
        variants=("v",), seeds=(0,), overrides=(),
        preset="medium", max_steps=10,
    )
    rows = enumerate_cartesian(sweep)
    assert len(rows) == 1
    assert rows[0].overrides == {}


def test_ablation_cell_id_propagates_to_rows():
    sweep = SweepConfig(
        variants=("v",), seeds=(0, 1), overrides=({},),
        preset="medium", max_steps=10,
        ablation_cell_id="abl_test",
    )
    rows = enumerate_cartesian(sweep)
    assert all(r.ablation_cell == "abl_test" for r in rows)
