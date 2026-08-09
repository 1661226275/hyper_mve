"""Post-hoc game-theoretic metrics (v5 Pkg-09) — NashConv + empirical PoA.

Unit-level checks on the cheap pieces (ε schedule, report serialisation).
The former tiny end-to-end run was bound to the retired HyperMuZeroModel;
end-to-end game metrics for the new method run through
`scripts/eval_game_metrics.py` on frozen checkpoints (external act_fn path).
"""
from __future__ import annotations

import json

import pytest


def test_epsilon_schedule_endpoints():
    pytest.importorskip("torch")
    from hyper_mve.utils.eval.game_metrics import BRConfig, _epsilon

    br = BRConfig(eps_start=1.0, eps_end=0.05, eps_decay_steps=100)
    assert _epsilon(br, 0) == pytest.approx(1.0)
    assert _epsilon(br, 50) == pytest.approx(0.525)
    assert _epsilon(br, 100) == pytest.approx(0.05)
    # Clamps past the decay horizon.
    assert _epsilon(br, 10_000) == pytest.approx(0.05)


def test_report_to_dict_json_serialisable():
    """Regime ids become string keys; sentinel + lower-bound note present."""
    pytest.importorskip("torch")
    from hyper_mve.utils.eval.game_metrics import GameMetricsReport

    report = GameMetricsReport(
        variant="hyper", checkpoint="ckpt.pt", regime_ids=[0, 2],
        br_env_steps=20_000, eval_episodes=10,
    )
    report.v_pi[0] = [1.0, 2.0]
    report.nashconv[0] = 0.5
    report.welfare_physical[0] = 3.0
    report.efficiency[0] = 0.75

    body = json.loads(json.dumps(report.to_dict()))
    assert body["schema_version"] == "game-metrics-v2"
    assert body["nashconv"] == {"0": 0.5}
    assert body["v_pi"] == {"0": [1.0, 2.0]}
    assert "LOWER BOUNDS" in body["note"]
    assert body["coop_reference_welfare"] is None
