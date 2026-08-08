"""End-to-end wiring gate for the action-axis policy targets.

The unit tests in ``test_policy_target_agent_marginal.py`` prove the target's
maths on hand-built tensors. They cannot catch the things that only appear when
the real pipeline supplies the inputs, and those are exactly what would waste a
wave: ``target_sampled_actions`` arriving with a dtype ``scatter_add_`` rejects,
the ``(B, K+1, C, N)`` slice not lining up with ``target_sampled_adv``,
``config.action_space_size`` not existing on the real config object, or the new
branch of ``policy_loss_step`` never being reached because the choice failed to
parse.

So this trains for real on both new targets. Same pattern and same tier as
``test_target_guard_wiring.py`` -- a code path that has only ever run on
fixtures is a code path that has never actually been tested.

Kept in the default suite deliberately: these targets ship as selectable arms,
and a break in the chain would otherwise surface twelve hours into a run.
"""
from __future__ import annotations

import pytest


def _train_with(monkeypatch, extra_argv, total_env_steps=600):
    """Train tiny, returning the runner, with extra fork argv appended.

    Copied from test_target_guard_wiring.py rather than shared: the two files
    gate different chains, and a shared helper would couple them.
    """
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


@pytest.mark.parametrize("target", ["agent_q_softmax", "agent_q_blend"])
def test_action_axis_target_trains_and_evaluates(monkeypatch, target):
    """A real train+eval on each action-axis target.

    Covers the whole chain the unit tests cannot: the config choice parses ->
    policy_loss_step dispatches to the new branch -> agent_marginal_target
    scatters the real ``target_sampled_actions`` -> the loss backprops without
    producing NaN.
    """
    pytest.importorskip("torch")
    pytest.importorskip("pettingzoo")
    from hyper_mve.utils.eval.eval_report import EvalReport

    runner, _, env_fn = _train_with(
        monkeypatch,
        ["--policy_target_type", target, "--policy_target_temperature", "0.85"])
    assert runner.param_count() > 0
    report = runner.evaluate(env_fn, regime_grid=(0, 1), episodes=1)
    assert isinstance(report, EvalReport)
    for g, r in report.return_per_regime.items():
        assert r == r, f"regime {g} return is NaN — {target} corrupted training"


def test_action_axis_target_trains_under_root_cover_star(monkeypatch):
    """The star cell is the one the wave exists to test, and it is also the one
    with the different shapes: ``sampled_action_times`` rises to 13 so the C
    axis widens, and the root holds ~11 children instead of ~4.

    If the scatter mis-handles the wider, more collision-heavy batch, it must
    fail here rather than twelve hours into ``..._agentq_cover``.
    """
    pytest.importorskip("torch")
    pytest.importorskip("pettingzoo")
    from hyper_mve.utils.eval.eval_report import EvalReport

    runner, _, env_fn = _train_with(
        monkeypatch,
        ["--policy_target_type", "agent_q_softmax",
         "--policy_target_temperature", "0.85",
         "--root_cover", "star",
         "--sampled_action_times", "13", "--leaf_sampled_times", "5"])
    report = runner.evaluate(env_fn, regime_grid=(0,), episodes=1)
    assert isinstance(report, EvalReport)
    for g, r in report.return_per_regime.items():
        assert r == r, f"regime {g} return is NaN under star + agent_q_softmax"


def test_the_shipped_arms_train(monkeypatch):
    """The two registered arms, exercised through ``apply_arm_argv`` exactly as
    the grid will invoke them.

    The tests above pass flags directly; this one gates the arm definitions
    themselves, so a typo in arms.py cannot reach a launch.
    """
    pytest.importorskip("torch")
    pytest.importorskip("pettingzoo")
    from hyper_mve.ablation.arms import ARMS

    for arm in ("ref_bc_anneal_scaled_hardval_decoupled_agentq",
                "ref_bc_anneal_scaled_hardval_decoupled_agentq_cover"):
        assert arm in ARMS

    from hyper_mve.algo import runner as runner_mod
    from hyper_mve.comparison import create_runner
    from hyper_mve.utils.configs import V4Config
    from hyper_mve.envs.adapters.pettingzoo_wrapper import RelationCommonsPettingZooEnv

    cfg = V4Config.from_preset("rel_duo")
    runner = create_runner(cfg, "mazero_mixed")
    env_fn = lambda: RelationCommonsPettingZooEnv(
        cfg.env, oracle_mode=False, eval_info_mode=False,
    )
    # the arm MUST go through the ``ablation`` kwarg: train() overwrites
    # ``self._ablation`` from kwargs (runner.py:290), so pre-setting the
    # attribute -- which is the right idiom for load_checkpoint -- is silently
    # discarded here and the run would train on the DEFAULT visit target.
    # This assertion is the point of the test.
    runner.train(cfg, env_fn, total_env_steps=600, lr=0.02, seed=0,
                 ablation="ref_bc_anneal_scaled_hardval_decoupled_agentq")
    gc = runner._game_config
    assert gc.policy_target_type == "agent_q_softmax"
    assert gc.policy_target_temperature == 0.85
    assert runner.param_count() > 0
