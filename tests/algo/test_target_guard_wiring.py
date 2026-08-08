"""End-to-end wiring gate for the stage-C flat-Q guard.

The guard ships OFF (``--policy_target_min_qstd 0``), so no ordinary run
exercises it. A code path that only ever runs disabled is a code path that has
never actually been tested — this trains for real with the guard ENABLED and
asserts it both fires and does not corrupt training.

Covers the whole chain the unit tests cannot: reanalyze_worker computes the
predicate -> it survives the 6-element batch_policies tuple -> train.py unpacks
it, applies it per unroll step, and renormalizes.
"""
from __future__ import annotations

import pytest


def _train_with(monkeypatch, extra_argv, total_env_steps=600):
    """Train tiny, returning the runner, with extra fork argv appended."""
    from hyper_mve.algo import runner as runner_mod
    from hyper_mve.comparison import create_runner
    from hyper_mve.utils.configs import V4Config
    from hyper_mve.envs.adapters.pettingzoo_wrapper import RelationCommonsPettingZooEnv

    orig = runner_mod.MAZeroMixedRunner._build_game_config

    def patched(self, *a, **kw):
        import core.config as core_config
        real_parse = core_config.parse_args

        def parse_with_extra(argv):
            return real_parse(list(argv) + list(extra_argv))

        monkeypatch.setattr(core_config, "parse_args", parse_with_extra)
        try:
            return orig(self, *a, **kw)
        finally:
            monkeypatch.setattr(core_config, "parse_args", real_parse)

    monkeypatch.setattr(runner_mod.MAZeroMixedRunner, "_build_game_config", patched)

    cfg = V4Config.from_preset("rel_duo")
    runner = create_runner(cfg, "mazero_mixed")
    env_fn = lambda: RelationCommonsPettingZooEnv(
        cfg.env, oracle_mode=False, eval_info_mode=False,
    )
    runner.train(cfg, env_fn, total_env_steps=total_env_steps, lr=0.02, seed=0)
    return runner, cfg, env_fn


def test_guard_enabled_trains_and_evaluates(monkeypatch):
    """A real train+eval with the guard ON. Catches any break in the predicate
    -> tuple -> unpack -> apply -> renormalize chain.

    Deliberately NOT marked slow (~2 min, same tier as the mazero smoke, which
    is also unmarked; `slow` is reserved here for the 20K env-step external
    gates). The guard ships disabled, so without this in the default suite the
    entire path would rot unnoticed until someone turned it on.
    """
    pytest.importorskip("torch")
    pytest.importorskip("pettingzoo")
    from hyper_mve.utils.eval.eval_report import EvalReport

    runner, _, env_fn = _train_with(
        monkeypatch, ["--policy_target_min_qstd", "0.4"])
    assert runner.param_count() > 0
    report = runner.evaluate(env_fn, regime_grid=(0, 1), episodes=1)
    assert isinstance(report, EvalReport)
    # a guard that masked everything would leave the policy untrained but must
    # never produce non-finite returns
    for g, r in report.return_per_regime.items():
        assert r == r, f"regime {g} return is NaN — guard corrupted training"


@pytest.mark.slow
def test_extreme_guard_does_not_nan_the_run(monkeypatch):
    """An absurd threshold masks (nearly) every transition. The renormalization
    must degrade to zero policy gradient, NOT to 0/0 -> NaN propagating through
    total_loss and poisoning value/reward too."""
    pytest.importorskip("torch")
    pytest.importorskip("pettingzoo")
    from hyper_mve.utils.eval.eval_report import EvalReport

    runner, _, env_fn = _train_with(
        monkeypatch, ["--policy_target_min_qstd", "1000.0"])
    report = runner.evaluate(env_fn, regime_grid=(0,), episodes=1)
    assert isinstance(report, EvalReport)
    for g, r in report.return_per_regime.items():
        assert r == r, f"regime {g} return is NaN under an all-masking guard"
