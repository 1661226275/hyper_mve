"""spec 07 §6.3 + §9.5 + §11.8 — disclosure table 10-column schema lock."""
from __future__ import annotations

from hyper_mve.experiments.stats import DISCLOSURE_COLUMNS, render_disclosure_table


# pkg-07 spec 07 §4.2 + pkg-08 spec 07 §11.8 verbatim 10-tuple.
EXPECTED_COLUMNS = (
    "variant", "preset", "param_count", "walltime_to_converge_seconds",
    "lr_swept_best", "final_return_mean", "final_return_sem",
    "seeds_run", "smoke_pass", "sourced",
)


def test_disclosure_columns_constant_locked():
    """spec 07 §11.8 — DISCLOSURE_COLUMNS is the verbatim 10-tuple."""
    assert DISCLOSURE_COLUMNS == EXPECTED_COLUMNS
    assert len(DISCLOSURE_COLUMNS) == 10


def test_disclosure_table_emits_per_preset_block_with_columns():
    """spec 07 §9.5 — markdown contains the 10 columns and per-preset header."""
    rows = [
        {
            "status": "completed", "variant": "external_mappo",
            "preset": "easy", "seed": s,
            "return_mean": 12.0 + 0.1 * s, "walltime_seconds": 60.0 + s,
        }
        for s in range(5)
    ]
    md = render_disclosure_table(rows, preset="easy")
    assert "## Disclosure table — easy preset" in md
    for col in EXPECTED_COLUMNS:
        assert f"| {col} |" in md or f"{col} " in md
    # Footnotes per pkg-07 spec 07 §4.5/§4.6/§4.7.
    assert "5-seed minimum" in md
    assert "MAMBA" in md
    assert "external_marie" in md or "external_ga" in md


def test_disclosure_table_filters_to_external_completed_only():
    """spec 07 §9.5 — only completed external_* rows are aggregated."""
    rows = [
        {"status": "completed", "variant": "hyper", "preset": "easy",
         "seed": 0, "return_mean": 18.0, "walltime_seconds": 0.0},
        {"status": "failed", "variant": "external_mappo", "preset": "easy",
         "seed": 0, "return_mean": None, "walltime_seconds": None},
        {"status": "completed", "variant": "external_qmix", "preset": "easy",
         "seed": 0, "return_mean": 10.0, "walltime_seconds": 30.0},
    ]
    md = render_disclosure_table(rows, preset="easy")
    # 'hyper' is internal → excluded.
    assert " hyper " not in md.replace("|", " ")
    # Only completed external row appears.
    assert " external_qmix " in md.replace("|", " ")
