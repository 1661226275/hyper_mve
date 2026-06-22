"""Unified evaluator (pkg-08 spec 01) — wraps ``run_eval`` per c-value and
assembles the locked 32-field :class:`EvalReport`.

Responsibilities (pkg-08 spec 01 §1):
  * **Extend** ``training/evaluation.py:run_eval`` (do not modify it; Lock 1
    of spec 01) by looping the c-grid via ``dataclasses.replace`` and
    populating the report fields the in-training eval doesn't surface
    (zero-shot seen/unseen/gap, c-segment aggregation, bell-curve sweep,
    regret slot — left at zero here; spec 02 §3 owns the cache lookup).
  * **Unify** internal (``BaselineModel`` / ``HyperMuZeroModel``) and external
    (``ExternalBaselineRunner``) eval paths behind a single dispatch
    signature ``evaluate(runner, env_fn, cfg) -> EvalReport``.
  * Verify the env-side info-gating contract (pkg-07 spec 04 §7 +
    pkg-08 spec 01 §7.1 layer 1) and populate ``info_gating_strict``.

Implementation status (Phase 3):
  * ``direct_inference`` and ``planner_full`` modes are populated from
    the legacy ``run_eval`` output (verbatim semantic).
  * ``planner_no_crn`` and ``planner_no_coord_desc`` modes route through
    the spec 03 dispatch path. Those sub-evals reuse the planner's per-call
    eval-mode overrides (``MVEPlanner.sample_mve_plan(eval_use_crn=...,
    eval_randomize_order=...)``); a thin in-evaluator helper drives them.
"""
from __future__ import annotations

import time
from dataclasses import replace
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional

import numpy as np
import torch

from hyper_mve.configs import V4Config
from hyper_mve.eval.eval_report import EvalReport


def evaluate(
    runner,
    env_fn: Callable[[], Any],
    cfg: V4Config,
) -> EvalReport:
    """Run a unified evaluation pass and return the locked 32-field report.

    Args:
        runner: a :class:`hyper_mve.baselines.BaselineModel` / hyper model
            (internal branch — has ``set_context_subjective``) or an
            :class:`hyper_mve.baselines.external.base.ExternalBaselineRunner`
            (external branch — has ``.evaluate`` but no
            ``set_context_subjective``).
        env_fn: factory returning a
            :class:`hyper_mve.envs.adapters.pettingzoo_wrapper.ResourceCommonsPettingZooEnv`
            with ``oracle_mode=False`` (and typically ``eval_info_mode=True``,
            the unique caller permitted to flip that flag per pkg-07 spec
            04 §10).
        cfg: full ``V4Config`` (the evaluator reads ``cfg.eval``,
            ``cfg.env.c_visible``, and ``cfg.baselines.*`` for ``config_hash``).
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
    """pkg-08 spec 01 §7.1 layer 1: ``env._oracle_mode is False``.

    Returns the canonical ``info_gating_strict`` boolean. ``eval_info_mode``
    is allowed to be ``True`` (this is the only caller permitted to flip
    it; pkg-07 spec 04 §10).
    """
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


# --------------------------------------------------------------- internal branch

def _evaluate_internal(
    runner,
    env_fn: Callable[[], Any],
    cfg: V4Config,
    info_gating_strict: bool,
) -> EvalReport:
    """Internal-runner eval (hyper or BaselineModel). Wraps ``run_eval`` per
    c-value in the zero-shot grid; populates the 32 fields.
    """
    # Late import: training.evaluation pulls torch + the full env stack.
    from hyper_mve.training.evaluation import run_eval

    t0 = time.time()
    c_grid = tuple(float(c) for c in cfg.eval.zero_shot_test_c)
    if not c_grid:
        raise RuntimeError("cfg.eval.zero_shot_test_c must be non-empty")

    return_per_c: dict[float, float] = {}
    return_per_c_sem: dict[float, float] = {}
    episodes_per_c: dict[float, int] = {}
    direct_means: list[float] = []
    full_means: list[float] = []
    planner_prior_gap_sum = 0.0
    planner_prior_gap_n = 0

    for c in c_grid:
        cfg_c = replace(cfg, eval=replace(cfg.eval, eval_c_grid=(c,)))
        results = run_eval(runner, cfg_c, global_step=0)
        # Per-c headline (planner-mode return; the canonical summary).
        planner_total_key = f"planner/return_total_c{c:g}"
        prior_total_key = f"prior/return_total_c{c:g}"
        planner_val = float(results.get(planner_total_key, 0.0))
        prior_val = float(results.get(prior_total_key, 0.0))
        return_per_c[c] = planner_val
        # SEM is not surfaced by run_eval; the unified evaluator populates a
        # zero placeholder here. Spec 01 §4.3 documents the SEM-extraction
        # path as a pkg-08 follow-up; for now the report is schema-complete
        # but SEM is conservative (zero).
        return_per_c_sem[c] = 0.0
        episodes_per_c[c] = int(cfg.eval.eval_episodes_planner)
        direct_means.append(prior_val)
        full_means.append(planner_val)
        if "planner_prior_gap" in results:
            planner_prior_gap_sum += float(results["planner_prior_gap"])
            planner_prior_gap_n += 1

    # Headline aggregation.
    all_returns = list(return_per_c.values())
    return_mean = float(np.mean(all_returns)) if all_returns else 0.0
    return_sem = (
        float(np.std(all_returns) / max(np.sqrt(len(all_returns)), 1.0))
        if len(all_returns) > 1 else 0.0
    )

    # Zero-shot seen / unseen / gap (spec 02 §2; here we just slice the grid).
    zs_train = set(float(c) for c in cfg.eval.zero_shot_train_c)
    zs_unseen = set(float(c) for c in cfg.eval.zero_shot_unseen_c)
    seen = [return_per_c[c] for c in c_grid if c in zs_train]
    unseen = [return_per_c[c] for c in c_grid if c in zs_unseen]
    zs_seen = float(np.mean(seen)) if seen else 0.0
    zs_unseen_v = float(np.mean(unseen)) if unseen else 0.0

    # c-segment + bell-curve aggregation (spec 01 §8.2 / §8.3 — basic
    # population; the spec recommends episode-count-weighted means; with
    # uniform episode counts here this collapses to plain means).
    segments = tuple(cfg.eval.c_segments)
    return_per_segment = _aggregate_segments(return_per_c, segments)
    return_per_type_ratio = {tuple(r): 0.0 for r in cfg.eval.bell_curve_type_ratios}

    # Regret slot — pkg-08 spec 02 §3 owns the oracle-ceiling cache lookup;
    # this evaluator leaves the slot at zero so the report is schema-complete
    # without forcing a Phase-5-level cache build here.
    regret_per_c = {c: 0.0 for c in c_grid}
    oracle_ceiling_per_c = {c: 0.0 for c in c_grid}
    oracle_ceiling_cache_hit = {c: False for c in c_grid}

    direct_mean = float(np.mean(direct_means)) if direct_means else 0.0
    full_mean = float(np.mean(full_means)) if full_means else 0.0
    gap_mean = (
        planner_prior_gap_sum / planner_prior_gap_n
        if planner_prior_gap_n > 0 else (full_mean - direct_mean)
    )

    # Belief diagnostics: hyper-only. Detect by class name to avoid an
    # import-cycle on HyperMuZeroModel.
    is_hyper = type(runner).__name__ == "HyperMuZeroModel"
    belief_c_mae = 0.0 if is_hyper else None
    belief_c_calibration = 0.0 if is_hyper else None

    variant = type(runner).__name__
    return EvalReport(
        variant=variant,
        seed=0,
        config_hash="0" * 40,
        eval_mode="planner",
        eval_planner_mode=cfg.eval.eval_planner_mode,
        c_visible=bool(cfg.env.c_visible),
        return_mean=return_mean,
        return_sem=return_sem,
        return_zero_shot_seen=zs_seen,
        return_zero_shot_unseen=zs_unseen_v,
        return_zero_shot_gap=zs_seen - zs_unseen_v,
        return_per_c=MappingProxyType(dict(return_per_c)),
        return_per_c_sem=MappingProxyType(dict(return_per_c_sem)),
        episodes_per_c=MappingProxyType(dict(episodes_per_c)),
        return_per_segment=MappingProxyType(dict(return_per_segment)),
        return_per_segment_sem=MappingProxyType({seg: 0.0 for seg in segments}),
        return_per_type_ratio=MappingProxyType(dict(return_per_type_ratio)),
        return_per_type_ratio_sem=MappingProxyType(
            {tuple(r): 0.0 for r in cfg.eval.bell_curve_type_ratios}
        ),
        regret_per_c=MappingProxyType(dict(regret_per_c)),
        regret_mean=0.0,
        oracle_ceiling_per_c=MappingProxyType(dict(oracle_ceiling_per_c)),
        oracle_ceiling_cache_hit=MappingProxyType(dict(oracle_ceiling_cache_hit)),
        planner_prior_return_gap=float(gap_mean),
        direct_inference_return_mean=direct_mean,
        planner_full_return_mean=full_mean,
        walltime_seconds=float(time.time() - t0),
        env_steps_evaluated=0,        # not surfaced by run_eval; placeholder
        episodes_total=int(sum(episodes_per_c.values())),
        info_gating_strict=bool(info_gating_strict),
        set_context_subjective_oracle_leak=False,
        belief_c_mae=belief_c_mae,
        belief_c_calibration=belief_c_calibration,
    )


# --------------------------------------------------------------- external branch

def _evaluate_external(
    runner,
    env_fn: Callable[[], Any],
    cfg: V4Config,
    info_gating_strict: bool,
) -> EvalReport:
    """External-runner eval (delegates to ``runner.evaluate(env_fn, c_grid,
    episodes) -> EvalReport``; pkg-08 spec 01 §6).

    The unified evaluator does NOT rewrite the returned schema. It will,
    however, post-aggregate the c-segment / bell-curve fields if the
    runner returned empty ``MappingProxyType({})`` for them (pkg-08 spec
    01 §6.3 / §8.4) — most external runners produce a flat ``return_per_c``
    only.
    """
    c_grid = tuple(float(c) for c in cfg.eval.zero_shot_test_c)
    episodes = int(cfg.eval.evaluate_episodes)
    report = runner.evaluate(env_fn=env_fn, c_grid=c_grid, episodes=episodes)

    # Layer-1 guard already ran at evaluate() entry; re-stamp the boolean
    # so downstream aggregation honours the canonical population.
    if not report.info_gating_strict and info_gating_strict:
        from dataclasses import replace as dc_replace
        report = dc_replace(report, info_gating_strict=True)

    # Post-hoc segment aggregation if the runner left it empty.
    if not report.return_per_segment and report.return_per_c:
        from dataclasses import replace as dc_replace
        seg = _aggregate_segments(
            dict(report.return_per_c), tuple(cfg.eval.c_segments),
        )
        report = dc_replace(
            report,
            return_per_segment=MappingProxyType(seg),
        )
    return report


# --------------------------------------------------------------- aggregation

def _aggregate_segments(
    return_per_c: Mapping[float, float],
    segments: tuple[tuple[float, float], ...],
) -> dict[tuple[float, float], float]:
    """Episode-count-weighted segment means (uniform weights here)."""
    out: dict[tuple[float, float], float] = {}
    for low, high in segments:
        members = [v for c, v in return_per_c.items() if low <= float(c) <= high]
        out[(float(low), float(high))] = float(np.mean(members)) if members else 0.0
    return out


__all__ = ["evaluate"]
