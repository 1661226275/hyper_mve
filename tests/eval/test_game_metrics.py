"""Post-hoc game-theoretic metrics (v5 Pkg-09) — NashConv + empirical PoA.

Unit-level checks on the cheap pieces (ε schedule, report serialisation) plus
one tiny end-to-end `compute_game_metrics` run on a shrunken rel_duo with a
random-init model — shape/sign conformance, not metric quality. The real-budget
protocol runs through `scripts/eval_game_metrics.py` on frozen checkpoints.
"""
from __future__ import annotations

import json

import pytest


def test_epsilon_schedule_endpoints():
    pytest.importorskip("torch")
    from hyper_mve.eval.game_metrics import BRConfig, _epsilon

    br = BRConfig(eps_start=1.0, eps_end=0.05, eps_decay_steps=100)
    assert _epsilon(br, 0) == pytest.approx(1.0)
    assert _epsilon(br, 50) == pytest.approx(0.525)
    assert _epsilon(br, 100) == pytest.approx(0.05)
    # Clamps past the decay horizon.
    assert _epsilon(br, 10_000) == pytest.approx(0.05)


def test_report_to_dict_json_serialisable():
    """Regime ids become string keys; sentinel + lower-bound note present."""
    pytest.importorskip("torch")
    from hyper_mve.eval.game_metrics import GameMetricsReport

    report = GameMetricsReport(
        variant="hyper", checkpoint="ckpt.pt", regime_ids=[0, 2],
        br_env_steps=20_000, eval_episodes=10,
    )
    report.v_pi[0] = [1.0, 2.0]
    report.nashconv[0] = 0.5
    report.welfare_physical[0] = 3.0
    report.efficiency[0] = 0.75

    body = json.loads(json.dumps(report.to_dict()))
    assert body["schema_version"] == "game-metrics-v1"
    assert body["nashconv"] == {"0": 0.5}
    assert body["v_pi"] == {"0": [1.0, 2.0]}
    assert "LOWER BOUNDS" in body["note"]
    assert body["coop_reference_welfare"] is None


def test_compute_game_metrics_tiny_end_to_end():
    """Shape/sign conformance on rel_duo with a random-init model.

    NashConv ≥ 0 by construction (per-agent max(0, ·)); efficiency populated
    exactly when a coop reference is supplied; JSON round-trips.
    """
    pytest.importorskip("torch")
    from dataclasses import replace

    from hyper_mve.configs import V4Config
    from hyper_mve.eval.game_metrics import BRConfig, compute_game_metrics
    from hyper_mve.models.hyper_muzero_model import HyperMuZeroModel

    cfg = V4Config.from_preset("rel_duo")
    cfg = replace(cfg, env=replace(cfg.env, T_max=10))
    model = HyperMuZeroModel(cfg)
    br = BRConfig(
        env_steps=24, warmup_steps=4, eps_decay_steps=10, batch_size=8,
        buffer_size=64, target_update_every=8, train_every=2, hidden=32,
    )
    report = compute_game_metrics(
        model, cfg, variant="hyper", checkpoint="random-init",
        regime_ids=[0], br=br, eval_episodes=1, seed=0,
        coop_reference_welfare=1.0, coop_reference_provenance="unit-test dummy",
    )

    N = cfg.env.N
    assert report.regime_ids == [0]
    assert len(report.v_pi[0]) == N
    assert len(report.v_br[0]) == N
    assert len(report.exploitability[0]) == N
    assert all(x >= 0.0 for x in report.exploitability[0])
    assert report.nashconv[0] >= 0.0
    assert report.nashconv[0] == pytest.approx(sum(report.exploitability[0]))
    assert report.welfare_physical[0] >= 0.0
    assert report.efficiency[0] == pytest.approx(report.welfare_physical[0] / 1.0)
    json.dumps(report.to_dict())  # must be serialisable end-to-end
