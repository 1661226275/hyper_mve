"""SuiteCell loader — meta/SweepConfig split + strict-parse preservation.

Runs without torch or pyyaml: ``cell_from_mapping`` takes an already-parsed
mapping, and ``sweep.from_mapping`` is pure-stdlib.
"""
from __future__ import annotations

import pytest

from hyper_mve.experiments.suite.cell import cell_from_mapping, SuiteCell, TIERS


def _body(**over):
    base = {
        "meta": {
            "id": "abl3_type_heterogeneity",
            "title": "type heterogeneity bell curve",
            "tier": "must_have",
            "size": "medium",
            "deliverables": ["Table 6.4", "Fig 6.4"],
            "assertion": "A",
            "runs_subdir": "abl3_type_het",
            "blocked_on": [],
        },
        "variants": ["hyper", "baseline_ma_muzero"],
        "seeds": [0, 1],
        "overrides": [
            {"env.N": 4, "env.type_assignment": [1, 1, 1, 1]},
            {"env.N": 4, "env.type_assignment": [0, 0, 0, 0]},
        ],
        "preset": "medium",
        "max_steps": 1000000,
        "ablation_cell_id": "abl3_type_heterogeneity",
    }
    base.update(over)
    return base


def test_valid_cell_roundtrip():
    cell = cell_from_mapping(_body(), source="abl3.yaml")
    assert isinstance(cell, SuiteCell)
    assert cell.id == "abl3_type_heterogeneity"
    assert cell.tier == "must_have" and cell.tier in TIERS
    assert cell.deliverables == ("Table 6.4", "Fig 6.4")
    assert cell.runs_subdir == "abl3_type_het"
    assert cell.blocked is False
    # SweepConfig body parsed strictly; overrides preserved verbatim.
    sc = cell.sweep_config
    assert sc.variants == ("hyper", "baseline_ma_muzero")
    assert sc.seeds == (0, 1)
    assert sc.preset == "medium" and sc.max_steps == 1000000
    assert dict(sc.overrides[0])["env.type_assignment"] == [1, 1, 1, 1]
    # 2 variants × 2 seeds × 2 overrides = 8 rows.
    assert cell.n_rows() == 8


def test_blocked_cell():
    body = _body()
    body["meta"]["blocked_on"] = ["variant:no_type", "metric:gini"]
    cell = cell_from_mapping(body, source="abl2.yaml")
    assert cell.blocked is True
    assert cell.blocked_on == ("variant:no_type", "metric:gini")


def test_meta_absent_uses_defaults():
    body = {"variants": ["hyper"], "seeds": [0], "preset": "easy"}
    cell = cell_from_mapping(body, source="x.yaml", default_id="my_cell")
    assert cell.id == "my_cell"
    assert cell.tier == "degradable"
    assert cell.size == "easy"  # falls back to preset
    assert cell.blocked is False


def test_unknown_meta_key_rejected():
    body = _body()
    body["meta"]["bogus"] = 1
    with pytest.raises(ValueError, match="unknown meta keys"):
        cell_from_mapping(body, source="bad.yaml")


def test_unknown_sweep_key_still_rejected():
    """meta is popped, but the SweepConfig body keeps strict unknown-key rejection."""
    body = _body()
    body["totally_not_a_sweep_key"] = 1
    with pytest.raises(ValueError, match="unknown SweepConfig keys"):
        cell_from_mapping(body, source="bad.yaml")


def test_bad_tier_rejected():
    body = _body()
    body["meta"]["tier"] = "p0"
    with pytest.raises(ValueError, match="meta.tier"):
        cell_from_mapping(body, source="bad.yaml")
