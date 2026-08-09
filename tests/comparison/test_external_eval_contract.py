"""C7-EXT-API1 + pkg-08 spec 01 §6 — external runners' ``evaluate`` returns
an :class:`EvalReport` with the spec-locked field-population matrix (rel-v1).

pkg-07 spec 05 §12.3 + spec 06 §10.1 / §10.2 — parametrised across the
three Tier-1 runners. The check is "shape-conformant": run a tiny
2-regime × 2-episode eval against a randomly-initialised runner (no
``train()`` required) and assert every external-delegation field is
populated correctly.
"""
from __future__ import annotations

import pytest


# Phase-2 registry names. "mazero_mixed" is excluded here: the method's
# evaluate() is a real prior-policy protocol (eval_mode="prior") with its own
# field-population assertions in test_mazero_mixed_smoke.py.
_TIER1_VARIANTS = (
    "mappo",
    "mamba",
)


@pytest.mark.parametrize("variant", _TIER1_VARIANTS)
def test_evaluate_returns_evalreport_shape_conforms(variant):
    """Field-population matrix (rel-v1).

    Runs against a randomly-initialised runner (no train()). Asserts:
      - return type is EvalReport, sentinel is "rel-v3"
      - variant + eval_mode + eval_planner_mode are correct
      - external delegation: planner_prior_return_gap == 0,
        direct_inference_return_mean == return_mean == planner_full_return_mean
      - regime_accuracy / regime_nll are None (no belief head)
      - info_gating_strict is True; set_context_subjective_oracle_leak is False
      - return_per_regime / _sem / episodes_per_regime are populated
        for every regime id in the grid
    """
    pytest.importorskip("torch")
    pytest.importorskip("pettingzoo")

    from hyper_mve.comparison import create_runner
    from hyper_mve.utils.configs import V4Config
    from hyper_mve.envs.adapters.pettingzoo_wrapper import RelationCommonsPettingZooEnv
    from hyper_mve.utils.eval.eval_report import EvalReport

    cfg = V4Config.from_preset("rel_duo")
    runner = create_runner(cfg, variant)
    env_fn = lambda: RelationCommonsPettingZooEnv(
        cfg.env, oracle_mode=False, eval_info_mode=False,
    )
    regime_grid = (0, 1)
    report = runner.evaluate(env_fn, regime_grid=regime_grid, episodes=2)

    assert isinstance(report, EvalReport)
    assert report.variant == variant
    assert report.eval_mode == "planner"
    assert report.eval_planner_mode == "planner_full"
    # External delegation field-population matrix.
    assert report.planner_prior_return_gap == 0.0
    assert report.direct_inference_return_mean == report.return_mean
    assert report.planner_full_return_mean == report.return_mean
    assert report.regime_accuracy is None
    assert report.regime_nll is None
    assert report.info_gating_strict is True
    assert report.set_context_subjective_oracle_leak is False
    # Per-regime populated for every id in the grid.
    assert set(report.return_per_regime) == set(regime_grid)
    assert set(report.return_per_regime_sem) == set(regime_grid)
    assert set(report.episodes_per_regime) == set(regime_grid)
    for g in regime_grid:
        assert report.episodes_per_regime[g] == 2
    # rel-v3: per-agent breakdown, one entry per agent per regime, and the
    # scalar per-regime return must be its sum. This is the invariant that
    # catches a mis-wired evaluate loop -- a dropped term or a permuted agent
    # order is invisible in the scalar alone.
    n_agents = int(cfg.env.N)
    assert set(report.return_per_regime_per_agent) == set(regime_grid)
    for g in regime_grid:
        vec = report.return_per_regime_per_agent[g]
        assert len(vec) == n_agents, f"regime {g}: {len(vec)} agents, expect {n_agents}"
        assert report.return_per_regime[g] == pytest.approx(sum(vec), abs=1e-6)
    # Schema sentinel (rel-v3).
    assert report.schema_version == "rel-v3"
    # rel-v2: the per-regime dicts above key on family-relative ids, so every
    # producer must say which family. Empty here would mean an id-keyed report
    # that cannot be interpreted once more than one family is in play.
    assert report.regime_names == (
        "mutual_coop", "mutual_comp", "asym_exploit", "asym_exploited", "neutral",
    )
    # zero-shot split: rel_duo has train_regime_ids=None ⇒ everything "seen".
    assert report.return_zero_shot_unseen == 0.0
    # External runners don't surface the thesis welfare metrics → defaults.
    assert report.welfare_physical_mean == 0.0
    assert report.sustainability_mean == 0.0
    assert report.fairness_mean == 0.0
    assert report.tragedy_index_mean == 0.0
