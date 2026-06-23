"""run_suite selective-targeting + --force invalidation.

Torch-free and yaml-free: cells are built via ``cell_from_mapping`` (dict) and
the registry is the pure-stdlib ``RunRegistry``.
"""
from __future__ import annotations

import argparse
import pathlib

from hyper_mve.experiments import sweep
from hyper_mve.experiments.run_registry import RunRegistry, RegistryRow, load_latest_per_run_id
from hyper_mve.experiments.suite.cell import cell_from_mapping
from hyper_mve.scripts import run_suite


def _args(**over):
    base = dict(
        only=None, tier=None, size=None, variants=None, seeds=None, max_steps=None,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _cell(cid, *, tier="must_have", size="medium", variants=("hyper", "external_mappo"),
          seeds=(0, 1, 2)):
    return cell_from_mapping(
        {
            "meta": {"id": cid, "tier": tier, "size": size, "runs_subdir": cid},
            "variants": list(variants),
            "seeds": list(seeds),
            "preset": size if size in ("easy", "medium", "hard") else "medium",
            "ablation_cell_id": cid,
        },
        source=f"{cid}.yaml",
    )


def test_select_only_and_filters():
    cells = [_cell("a", tier="must_have", size="easy"),
             _cell("b", tier="degradable", size="medium"),
             _cell("c", tier="must_have", size="medium")]
    # --only
    assert [c.id for c in run_suite._select_cells(cells, _args(only="c,a"))] == ["c", "a"]
    # --tier
    assert [c.id for c in run_suite._select_cells(cells, _args(tier="must_have"))] == ["a", "c"]
    # --size
    assert [c.id for c in run_suite._select_cells(cells, _args(size="medium"))] == ["b", "c"]


def test_select_unknown_id_raises():
    cells = [_cell("a")]
    try:
        run_suite._select_cells(cells, _args(only="nope"))
        raise AssertionError("expected SystemExit")
    except SystemExit as e:
        assert "unknown cell id" in str(e)


def test_narrow_variants_seeds_maxsteps():
    cell = _cell("a", variants=("hyper", "external_mappo", "external_qmix"), seeds=(0, 1, 2))
    cfg, warn = run_suite._narrow(cell, _args(variants="hyper,external_qmix", seeds="0,5", max_steps=123))
    assert warn is None
    assert cfg.variants == ("hyper", "external_qmix")  # intersection, cell order
    assert cfg.seeds == (0, 5)
    assert cfg.max_steps == 123


def test_narrow_empty_intersection_warns():
    cell = _cell("a", variants=("hyper",))
    cfg, warn = run_suite._narrow(cell, _args(variants="external_marie"))
    assert warn is not None and cfg.variants == ()


def test_force_invalidate_reenables_only_targeted_rows(tmp_path: pathlib.Path):
    cell = _cell("a", variants=("hyper", "external_mappo"), seeds=(0, 1))
    reg_path = tmp_path / "registry.jsonl"
    reg = RunRegistry(reg_path)

    # Seed: mark EVERY enumerated row 'completed' (so all would normally skip).
    full_rows = sweep.enumerate_cartesian(cell.sweep_config)
    for r in full_rows:
        ch = sweep.row_config_hash(cell.sweep_config, r)
        reg.append_row(RegistryRow(
            run_id=f"orig-{r.variant}-{r.seed}", variant=r.variant, seed=r.seed,
            config_hash=ch, ablation_cell="a", sweep_row_index=r.sweep_row_index,
            git_sha="x", git_dirty=False, started_at_iso8601="2026-06-23T00:00:00Z",
            completed_at_iso8601="2026-06-23T00:00:10Z", status="completed", failure_reason=None,
            gpu_id=0, walltime_seconds=1.0, peak_gpu_memory_mb=None,
            config_snapshot_path="", checkpoint_path="c", eval_report_path="e",
            tensorboard_dir="", return_mean=1.0, return_zero_shot_unseen=0.0, regret_mean=0.0,
        ))

    # Narrow to a SINGLE row (hyper, seed 0) and force it.
    narrowed = run_suite._narrow(cell, _args(variants="hyper", seeds="0"))[0]
    n = run_suite._force_invalidate(narrowed, reg_path)
    assert n == 1  # exactly the one targeted row

    latest = load_latest_per_run_id(reg.read_all())
    # The forced (hyper, seed0) row: latest is now 'failed' → re-runs with retry_failed.
    forced_row = sweep.enumerate_cartesian(narrowed)[0]
    skip, _ = sweep._resume_skip(forced_row, latest, retry_failed=True)
    assert skip is False, "forced row must re-run"
    # A non-targeted row (external_mappo, seed1) stays completed → still skips.
    other = [r for r in full_rows if r.variant == "external_mappo" and r.seed == 1][0]
    other = sweep_replace_hash(other, cell)
    skip2, _ = sweep._resume_skip(other, latest, retry_failed=True)
    assert skip2 is True, "untargeted completed row must still skip"


def sweep_replace_hash(row, cell):
    """Attach the config_hash run_sweep would compute (resume matches on it)."""
    from dataclasses import replace as _r
    return _r(row, config_hash=sweep.row_config_hash(cell.sweep_config, row))
