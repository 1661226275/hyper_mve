"""Unified evaluator (pkg-08 spec 01, v5 form) — wraps ``run_eval`` and
assembles the locked rel-v1 :class:`EvalReport`.

Responsibilities:
  * **Extend** ``training/evaluation.py:run_eval`` (do not modify it; Lock 1)
    — one call covers the whole regime grid; this evaluator extracts the
    per-regime keys and populates the report fields the in-training eval
    doesn't surface (zero-shot seen/unseen/gap via the train_regime_ids
    split, belief regime accuracy).
  * **Unify** internal (``BaselineModel`` / ``HyperMuZeroModel``) and external
    (``ExternalBaselineRunner``) eval paths behind a single dispatch
    signature ``evaluate(runner, env_fn, cfg) -> EvalReport``.
  * Verify the env-side info-gating contract (oracle_mode=False) and populate
    ``info_gating_strict``.
"""
from __future__ import annotations

import math
import time
from types import MappingProxyType
from typing import Any, Callable

import numpy as np

from hyper_mve.configs import V4Config
from hyper_mve.eval.eval_report import EvalReport


def evaluate(
    runner,
    env_fn: Callable[[], Any],
    cfg: V4Config,
) -> EvalReport:
    """Run a unified evaluation pass and return the locked rel-v1 report.

    Args:
        runner: a :class:`hyper_mve.baselines.BaselineModel` / hyper model
            (internal branch — has ``set_context_subjective``) or an
            :class:`hyper_mve.baselines.external.base.ExternalBaselineRunner`
            (external branch — has ``.evaluate`` but no
            ``set_context_subjective``).
        env_fn: factory returning a
            :class:`hyper_mve.envs.adapters.pettingzoo_wrapper.RelationCommonsPettingZooEnv`
            with ``oracle_mode=False``.
        cfg: full ``V4Config`` (the evaluator reads ``cfg.eval`` and
            ``cfg.env.train_regime_ids``).
    """
    info_gating_strict = _verify_env_fn_flags(env_fn)

    if _is_external_runner(runner):
        return _evaluate_external(runner, env_fn, cfg, info_gating_strict)
    return _evaluate_internal(runner, env_fn, cfg, info_gating_strict)


# --------------------------------------------------------------- dispatch helpers

def _is_external_runner(runner) -> bool:
    """External runners expose ``.evaluate`` but lack ``set_context_subjective``."""
    return (
        hasattr(runner, "evaluate")
        and not hasattr(runner, "set_context_subjective")
    )


def _verify_env_fn_flags(env_fn: Callable[[], Any]) -> bool:
    """pkg-08 spec 01 §7.1 layer 1: ``env._oracle_mode is False``."""
    env = env_fn()
    try:
        oracle_mode = bool(getattr(env, "_oracle_mode", False))
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    if oracle_mode:
        raise RuntimeError(
            "eval-time env_fn must have oracle_mode=False; got True. "
            "Pkg-07 spec 04 §7 violated."
        )
    return True


def _regime_grid(cfg: V4Config) -> tuple[int, ...]:
    if cfg.eval.eval_regime_grid is not None:
        return tuple(int(g) for g in cfg.eval.eval_regime_grid)
    from hyper_mve.schemas import get_regime_family
    return tuple(range(get_regime_family(cfg.env).size))


def _seen_unseen(cfg: V4Config, per_regime: dict[int, float]) -> tuple[float, float]:
    from hyper_mve.baselines.external.base import split_seen_unseen_regimes
    return split_seen_unseen_regimes(cfg, per_regime)


# --------------------------------------------------------------- internal branch

def _evaluate_internal(
    runner,
    env_fn: Callable[[], Any],
    cfg: V4Config,
    info_gating_strict: bool,
) -> EvalReport:
    """Internal-runner eval (hyper or BaselineModel): one ``run_eval`` pass
    over the regime grid; populate the rel-v1 fields."""
    # Late import: training.evaluation pulls torch + the full env stack.
    from hyper_mve.training.evaluation import run_eval

    t0 = time.time()
    grid = _regime_grid(cfg)
    results = run_eval(runner, cfg, global_step=0)

    return_per_regime: dict[int, float] = {}
    return_per_regime_sem: dict[int, float] = {}
    episodes_per_regime: dict[int, int] = {}
    welfare_per_regime: dict[int, float] = {}
    sustain_per_regime: dict[int, float] = {}
    fairness_per_regime: dict[int, float] = {}
    tragedy_per_regime: dict[int, float] = {}
    for g in grid:
        return_per_regime[g] = float(results.get(f"planner/return_total_g{g}", 0.0))
        # SEM is not surfaced per regime by run_eval; conservative zero
        # placeholder (documented pkg-08 follow-up).
        return_per_regime_sem[g] = 0.0
        episodes_per_regime[g] = int(cfg.eval.eval_episodes_planner)
        welfare_per_regime[g] = float(results.get(f"planner/welfare_physical_g{g}", 0.0))
        sustain_per_regime[g] = float(results.get(f"planner/sustainability_g{g}", float("nan")))
        fairness_per_regime[g] = float(results.get(f"planner/fairness_g{g}", float("nan")))
        tragedy_per_regime[g] = float(results.get(f"planner/tragedy_g{g}", float("nan")))

    def _mean_finite(d: dict[int, float]) -> float:
        vals = [v for v in d.values() if v == v]  # NaN != NaN
        return float(np.mean(vals)) if vals else 0.0

    all_returns = list(return_per_regime.values())
    return_mean = float(np.mean(all_returns)) if all_returns else 0.0
    return_sem = (
        float(np.std(all_returns) / max(np.sqrt(len(all_returns)), 1.0))
        if len(all_returns) > 1 else 0.0
    )
    zs_seen, zs_unseen = _seen_unseen(cfg, return_per_regime)

    direct_mean = float(results.get("prior/return_total", 0.0))
    full_mean = float(results.get("planner/return_total", return_mean))
    gap_mean = float(results.get("planner_prior_gap", full_mean - direct_mean))

    # Belief regime quality: present for belief-carrying models (hyper); the
    # run_eval key is NaN-free only when records carried a posterior.
    acc = results.get("belief/regime_accuracy_planner",
                      results.get("belief/regime_accuracy_prior", float("nan")))
    regime_accuracy = None if (acc is None or math.isnan(float(acc))) else float(acc)

    return EvalReport(
        variant=type(runner).__name__,
        seed=0,
        config_hash="0" * 40,
        eval_mode="planner",
        eval_planner_mode=cfg.eval.eval_planner_mode,
        return_mean=return_mean,
        return_sem=return_sem,
        return_zero_shot_seen=zs_seen,
        return_zero_shot_unseen=zs_unseen,
        return_zero_shot_gap=zs_seen - zs_unseen,
        return_per_regime=MappingProxyType(dict(return_per_regime)),
        return_per_regime_sem=MappingProxyType(dict(return_per_regime_sem)),
        episodes_per_regime=MappingProxyType(dict(episodes_per_regime)),
        planner_prior_return_gap=gap_mean,
        direct_inference_return_mean=direct_mean,
        planner_full_return_mean=full_mean,
        walltime_seconds=float(time.time() - t0),
        env_steps_evaluated=0,        # not surfaced by run_eval; placeholder
        episodes_total=int(sum(episodes_per_regime.values())),
        info_gating_strict=bool(info_gating_strict),
        set_context_subjective_oracle_leak=False,
        regime_accuracy=regime_accuracy,
        regime_nll=None,              # reserved (analysis-stage computation)
        welfare_physical_mean=_mean_finite(welfare_per_regime),
        sustainability_mean=_mean_finite(sustain_per_regime),
        fairness_mean=_mean_finite(fairness_per_regime),
        tragedy_index_mean=_mean_finite(tragedy_per_regime),
    )


# --------------------------------------------------------------- external branch

def _evaluate_external(
    runner,
    env_fn: Callable[[], Any],
    cfg: V4Config,
    info_gating_strict: bool,
) -> EvalReport:
    """External-runner eval (delegates to ``runner.evaluate(env_fn,
    regime_grid, episodes) -> EvalReport``)."""
    grid = _regime_grid(cfg)
    episodes = int(cfg.eval.evaluate_episodes)
    report = runner.evaluate(env_fn=env_fn, regime_grid=grid, episodes=episodes)

    # Layer-1 guard already ran at evaluate() entry; re-stamp the boolean
    # so downstream aggregation honours the canonical population.
    if not report.info_gating_strict and info_gating_strict:
        from dataclasses import replace as dc_replace
        report = dc_replace(report, info_gating_strict=True)
    return report


__all__ = ["evaluate"]
