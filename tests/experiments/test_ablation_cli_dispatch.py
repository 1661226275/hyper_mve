"""C8-ABL-CLI1 + Lock 1 + Lock 3 — ablate CLI dispatch (pkg-08 spec 06 §2 / §7)."""
from __future__ import annotations

import io
import pathlib
import sys

import pytest

from hyper_mve.experiments import ablate
from hyper_mve.experiments.ablate import (
    ABLATION_IDS,
    ABLATIONS_DIR,
    main as ablate_main,
    materialise_sweep_config,
    parse_args,
)


def test_ablation_ids_locked():
    """Lock 1 — five canned IDs verbatim."""
    assert ABLATION_IDS == (
        "abl1", "abl4_crn_joint", "abl4_joint_easy_n2", "abl6", "abl7",
    )


@pytest.mark.parametrize("ablation_id", ABLATION_IDS)
def test_ablation_cli_dispatch_all_5(ablation_id):
    """C8-ABL-CLI1 — each CLI ID materialises a SweepConfig."""
    args = parse_args(["--ablation", ablation_id, "--dry-run"])
    sweep_cfg = materialise_sweep_config(args)
    expected_cell = (
        "abl4_joint_easy_n2" if ablation_id == "abl4_joint_easy_n2"
        else None
    )
    if ablation_id != "abl4_joint_easy_n2":
        expected_cell = sweep_cfg.ablation_cell_id  # YAML supplies it
    assert sweep_cfg.ablation_cell_id is not None
    assert isinstance(sweep_cfg.variants, tuple)
    assert isinstance(sweep_cfg.seeds, tuple)
    # spec 03 Lock 1 — every YAML hard-pins planner_full.
    assert sweep_cfg.eval_planner_mode == "planner_full"


def test_ablation_cli_rejects_unknown_id():
    """spec 06 §2.3 — argparse rejects unknown IDs with SystemExit(2)."""
    with pytest.raises(SystemExit) as exc_info:
        parse_args(["--ablation", "abl9"])
    assert exc_info.value.code == 2


def test_ablation_cli_dry_run_does_not_call_run_sweep(monkeypatch, capsys):
    """spec 06 §2.4 — --dry-run short-circuits before run_sweep."""
    def _fail(*args, **kwargs):
        raise RuntimeError("run_sweep should not be called in dry-run")
    monkeypatch.setattr(ablate, "run_sweep", _fail)
    code = ablate_main(["--ablation", "abl1", "--dry-run"])
    assert code == 0
    out = capsys.readouterr().out
    assert "[dry-run]" in out
    assert "Total rows would be:" in out
    assert "Exiting without calling run_sweep" in out


def test_abl4_joint_easy_n2_hardpin_preset(capsys):
    """Lock 3 — abl4_joint_easy_n2 hard-pins preset=easy regardless of CLI override."""
    for cli_preset in ("easy", "medium", "hard"):
        args = parse_args([
            "--ablation", "abl4_joint_easy_n2",
            "--preset", cli_preset,
            "--dry-run",
        ])
        sweep_cfg = materialise_sweep_config(args)
        assert sweep_cfg.preset == "easy", (
            f"hardpin failed: --preset={cli_preset} → cfg.preset={sweep_cfg.preset!r}"
        )
    err = capsys.readouterr().err
    # Hard-pin warning fires for non-easy CLI overrides.
    assert "hard-pins preset=easy" in err


def test_abl4_joint_easy_n2_override_carries_mve_joint_enumerate():
    args = parse_args(["--ablation", "abl4_joint_easy_n2", "--dry-run"])
    sweep_cfg = materialise_sweep_config(args)
    assert len(sweep_cfg.overrides) == 1
    assert sweep_cfg.overrides[0].get("train.mve_joint_enumerate") is True


def test_canned_yaml_inventory_matches_design():
    """Lock 1 drift detector — five canned YAML stems."""
    found = {p.stem for p in ABLATIONS_DIR.glob("*.yaml")}
    expected = {
        "abl1_gen_scope", "abl4_crn_joint", "abl4_joint_easy_n2",
        "abl6_fehr_schmidt", "abl7_curriculum",
    }
    assert found == expected, (
        f"Canned YAML inventory drift: extra={found - expected}, "
        f"missing={expected - found}"
    )


def test_abl1_gen_scope_yaml_materialises_7_cells():
    """spec 06 §3.1 — 7 cells × 6 variants × 5 seeds = 210 rows."""
    args = parse_args(["--ablation", "abl1", "--dry-run"])
    sweep_cfg = materialise_sweep_config(args)
    assert len(sweep_cfg.overrides) == 7
    assert len(sweep_cfg.variants) == 6
    assert len(sweep_cfg.seeds) == 5
    tags = {o.get("legacy.ablation_cell_tag") for o in sweep_cfg.overrides}
    assert "within_distribution" in tags
    assert "c_hidden" in tags
    assert "both_axes_hidden" in tags
    assert len(tags) == 7


def test_abl4_crn_joint_yaml_materialises_2x2():
    """spec 06 §3.2 — 4 overrides; each carries use_crn + randomize_order."""
    args = parse_args(["--ablation", "abl4_crn_joint", "--dry-run"])
    sweep_cfg = materialise_sweep_config(args)
    assert len(sweep_cfg.overrides) == 4
    for override in sweep_cfg.overrides:
        assert "train.use_crn" in override
        assert "train.randomize_order" in override
    pairs = {(o["train.use_crn"], o["train.randomize_order"])
             for o in sweep_cfg.overrides}
    assert pairs == {(True, True), (False, True), (True, False), (False, False)}


def test_abl6_fehr_schmidt_yaml_materialises_3x3():
    """spec 06 §3.4 — 9 cells (3 alpha × 3 beta)."""
    args = parse_args(["--ablation", "abl6", "--dry-run"])
    sweep_cfg = materialise_sweep_config(args)
    assert len(sweep_cfg.overrides) == 9
    alpha_vals = {o["train.fehr_schmidt_alpha"] for o in sweep_cfg.overrides}
    beta_vals = {o["train.fehr_schmidt_beta"] for o in sweep_cfg.overrides}
    assert alpha_vals == {0.0, 0.5, 1.0}
    assert beta_vals == {0.0, 0.25, 0.5}


def test_abl7_curriculum_yaml_three_variants():
    """spec 06 §3.5 — variants order-preserving; overrides == ({},)."""
    args = parse_args(["--ablation", "abl7", "--dry-run"])
    sweep_cfg = materialise_sweep_config(args)
    assert sweep_cfg.variants == ("oracle_only", "hyper", "infer_only")
    assert sweep_cfg.overrides == ({},)
