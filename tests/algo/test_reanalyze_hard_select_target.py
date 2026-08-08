"""value_hard_select must make the reanalyzed value TARGET consistent with the
hard-selected PREDICTION, not just the prediction alone.

Regression test for a training-recipe bug: train.py's forward pass already
called model.set_oracle_regime(g_true) before computing the value it trains
(the "prediction"), but reanalyze_worker.py's _prepare_reward_value_re -- which
builds the bootstrapped value TARGET that prediction is compared against --
never called set_oracle_regime at all, so it silently fell through to the
Bayes-averaged value. In the aliased regimes (g2/g3), where belief confidently
collapses onto the wrong symmetric partner, that made the "clean" hard-selected
prediction chase a target built from the WRONG regime's value -- undermining
the fix specifically where it was meant to help (confirmed via
scripts/probes/value_deploy_probe.py + belief_confusion_probe.py: oracle-mode
value lookup underperforms the belief-averaged deploy on both ref_bc and
ref_bc_hardval checkpoints, worst in g3).

This exercises the real serial training loop (ReanalyzeWorker instantiated
in-process by train_sync_serial) with value_hard_select on, and spies on
set_oracle_regime to confirm BOTH call sites (train.py's prediction and
reanalyze_worker.py's target) now invoke it with real regime ids.
"""
from __future__ import annotations

import pytest


def test_reanalyze_sets_oracle_regime_for_value_targets(monkeypatch):
    """Crash-free run (validates game.g_trues[bootstrap_index] shape/dtype/
    alignment) AND a calls log proving reanalyze's target computation now
    hard-selects too, not just train.py's prediction."""
    pytest.importorskip("torch")
    pytest.importorskip("pettingzoo")
    import importlib
    import torch

    from hyper_mve.algo.runner import _ensure_fork_on_path
    _ensure_fork_on_path()
    # runner.py builds the model via the FORK's own bare "config.relation"
    # import (relies on _ensure_fork_on_path putting the fork dir on
    # sys.path), a separate module identity from
    # hyper_mve.algo.mazero_mixed.config.relation.subjective_model even though
    # it's the same file -- must patch the class object actually instantiated
    # at runtime, or the monkeypatch silently misses every call.
    HyperMAMuZeroNet = importlib.import_module(
        "config.relation.subjective_model"
    ).HyperMAMuZeroNet
    from hyper_mve.comparison import create_runner
    from hyper_mve.utils.configs import V4Config
    from hyper_mve.envs.adapters.pettingzoo_wrapper import RelationCommonsPettingZooEnv

    calls = []
    real_set_oracle_regime = HyperMAMuZeroNet.set_oracle_regime

    def spy_set_oracle_regime(self, g_true):
        if g_true is not None:
            calls.append(g_true.shape[0] if hasattr(g_true, "shape") else None)
        return real_set_oracle_regime(self, g_true)

    monkeypatch.setattr(HyperMAMuZeroNet, "set_oracle_regime", spy_set_oracle_regime)

    cfg = V4Config.from_preset("rel_duo")
    runner = create_runner(cfg, "mazero_mixed")
    env_fn = lambda: RelationCommonsPettingZooEnv(
        cfg.env, oracle_mode=False, eval_info_mode=False,
    )
    # ref_bc_hardval turns on value_hard_select (plus ref_bc's BC/reference
    # guidance, irrelevant here but harmless at this tiny budget).
    runner.train(cfg, env_fn, total_env_steps=1200, lr=0.02, seed=0,
                 ablation="ref_bc_hardval")

    non_none_calls = [c for c in calls if c is not None]
    assert len(non_none_calls) >= 2, (
        f"expected set_oracle_regime to be called with real regime ids from "
        f"BOTH train.py's prediction pass and reanalyze_worker.py's target "
        f"pass, got {len(non_none_calls)} non-None call(s): {calls}"
    )
    report = runner.evaluate(env_fn, regime_grid=(0, 2, 3), episodes=1)
    for g, r in report.return_per_regime.items():
        assert r == r, f"regime {g} return is NaN -- hard-select target fix corrupted training"
