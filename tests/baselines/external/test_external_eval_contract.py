"""C7-EXT-API1 + pkg-08 spec 01 §6 — external runners' ``evaluate`` returns
an :class:`EvalReport` with the spec-locked field-population matrix.

pkg-07 spec 05 §12.3 + spec 06 §10.1 / §10.2 — parametrised across the
three Tier-1 runners. The check is "shape-conformant": run a tiny
2-c × 2-episode eval against a randomly-initialised runner (no
``train()`` required) and assert every external-delegation field is
populated correctly per the §8.1 matrix.
"""
from __future__ import annotations

from types import MappingProxyType

import pytest


_TIER1_VARIANTS = ("external_mappo", "external_qmix", "external_ma_muzero_gh")


@pytest.mark.parametrize("variant", _TIER1_VARIANTS)
def test_evaluate_returns_evalreport_shape_conforms(variant):
    """Field-population matrix per pkg-07 spec 05 §8.1 / spec 06 §2.7 / §3.7.

    Runs against a randomly-initialised runner (no train()). Asserts:
      - return type is EvalReport
      - variant + eval_mode + eval_planner_mode are correct
      - external delegation: planner_prior_return_gap == 0,
        direct_inference_return_mean == return_mean == planner_full_return_mean
      - belief_c_mae / belief_c_calibration are None (no belief head)
      - info_gating_strict is True; set_context_subjective_oracle_leak is False
      - return_per_segment / return_per_type_ratio / regret_per_c are
        empty / zero (the unified evaluator post-aggregates)
      - return_per_c / return_per_c_sem / episodes_per_c are populated
        for every c in the grid
    """
    pytest.importorskip("torch")
    pytest.importorskip("pettingzoo")

    from hyper_mve.baselines import create_baseline
    from hyper_mve.configs import V4Config
    from hyper_mve.envs.adapters.pettingzoo_wrapper import ResourceCommonsPettingZooEnv
    from hyper_mve.eval.eval_report import EvalReport

    cfg = V4Config.from_preset("easy")
    runner = create_baseline(cfg, variant)
    env_fn = lambda: ResourceCommonsPettingZooEnv(
        cfg.env, oracle_mode=False, eval_info_mode=False,
    )
    c_grid = (0.0, 1.0)
    report = runner.evaluate(env_fn, c_grid=c_grid, episodes=2)

    assert isinstance(report, EvalReport)
    assert report.variant == variant
    assert report.eval_mode == "planner"
    assert report.eval_planner_mode == "planner_full"
    # External delegation field-population matrix.
    assert report.planner_prior_return_gap == 0.0
    assert report.direct_inference_return_mean == report.return_mean
    assert report.planner_full_return_mean == report.return_mean
    assert report.belief_c_mae is None
    assert report.belief_c_calibration is None
    assert report.info_gating_strict is True
    assert report.set_context_subjective_oracle_leak is False
    # Per-c populated for every c in grid.
    assert set(report.return_per_c) == set(c_grid)
    assert set(report.return_per_c_sem) == set(c_grid)
    assert set(report.episodes_per_c) == set(c_grid)
    for c in c_grid:
        assert report.episodes_per_c[c] == 2
    # Schema sentinel (v2: +4 welfare metrics, 2026-06).
    assert report.schema_version == "pkg08-spec01-v2"
    # External runners don't surface the thesis welfare metrics → defaults.
    assert report.welfare_physical_mean == 0.0
    assert report.sustainability_mean == 0.0
    assert report.fairness_mean == 0.0
    assert report.tragedy_index_mean == 0.0
