"""ref_bc_anneal_scaled must preserve the VALIDATED 74.7% anneal fraction
(28000/37500 training steps at the 600K budget where ref_bc was tuned), not
the raw absolute step count -- see hyper_mve/ablation/arms.py's
ref_bc_anneal_scaled comment and hyper_mve/algo/runner.py's special-case in
MAZeroMixedRunner._build_game_config.
"""
from __future__ import annotations

import pytest


def _parsed_args_for(total_env_steps, ablation, monkeypatch):
    """Build a game config under the given ablation/budget, capturing the
    argparse Namespace core.config.parse_args actually produced (cheaper and
    more direct than reading it back off the constructed GameConfig)."""
    from hyper_mve.comparison import create_runner
    from hyper_mve.utils.configs import V4Config
    import core.config as core_config

    captured = {}
    real_parse = core_config.parse_args

    def capturing_parse(argv):
        args = real_parse(argv)
        captured["args"] = args
        return args

    monkeypatch.setattr(core_config, "parse_args", capturing_parse)

    cfg = V4Config.from_preset("rel_duo")
    runner = create_runner(cfg, "mazero_mixed")
    runner.cfg = cfg
    runner._ablation = ablation
    runner._num_pmcts = 1
    runner._build_game_config(total_env_steps=total_env_steps, lr=0.02, seed=0)
    return captured["args"]


def test_600k_budget_keeps_the_validated_28000_step_anneal(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("pettingzoo")
    args = _parsed_args_for(600_000, "ref_bc_anneal_scaled", monkeypatch)
    assert args.reference_episode_anneal_steps == 28000


def test_1m_budget_scales_anneal_proportionally_not_28000(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("pettingzoo")
    args = _parsed_args_for(1_000_000, "ref_bc_anneal_scaled", monkeypatch)
    # training_steps = 1_000_000 // 16 = 62500; round(62500 * 28000 / 37500) = 46667
    assert args.reference_episode_anneal_steps == 46667
    assert args.reference_episode_anneal_steps != 28000


def test_plain_ref_bc_still_hardcodes_28000_at_1m(monkeypatch):
    """Confirms the bug this arm works around: unscaled ref_bc keeps the
    hardcoded value regardless of budget -- 44.8% of 1M's training_steps
    instead of the 600K-validated 74.7%."""
    pytest.importorskip("torch")
    pytest.importorskip("pettingzoo")
    args = _parsed_args_for(1_000_000, "ref_bc", monkeypatch)
    assert args.reference_episode_anneal_steps == 28000
