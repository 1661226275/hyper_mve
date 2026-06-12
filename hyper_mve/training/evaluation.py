"""In-training periodic evaluation [v4-opt 2026-06] (2agent run diagnosis).

The optimization-phase runs logged only losses — no episode returns, no eval — so
"policy degrading" was indistinguishable from "policy improving while the commons
depletes". This module adds the missing measurement:

    run_eval(model, cfg)  ->  flat {tag: float} dict (TB family ``eval/*``)

Protocol (user decision 2026-06-11, dual-mode + c-grid):
    - **prior** mode: planner OFF — actions = argmax of the prediction net's own
      policy pi_hat. Measures the *distilled* policy the thesis ultimately ships.
    - **planner** mode: planner ON — actions = argmax of pi_mve. Measures the true
      acting agent (Alg 5.2). The planner-prior return gap is the distillation
      residual: large gap ⇒ pi_hat has not absorbed the planner's signal.
    - Both run deterministically (epsilon=0, argmax, no sampling RNG) on dedicated
      eval envs with ``c_mode="static"`` pinned to each c in ``eval_c_grid`` and
      fixed reset seeds, and the planner uses a constant CRN seed — so curves at
      different global_steps differ only through the model weights (CRN across
      evaluations).

All episodes of one mode run as a single vectorized batch through
``Worker.collect_episodes`` (B = len(c_grid) * episodes), so one eval costs about
two batched episode rollouts.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np
import torch

from hyper_mve.configs import V4Config
from hyper_mve.envs.resource_commons.env import ResourceCommonsEnv
from hyper_mve.planning.mve_planner import MVEPlanner
from hyper_mve.schemas import AgentType
from hyper_mve.training.worker import Worker

# Constant seeds: identical eval conditions at every call (CRN across evaluations).
_EVAL_ENV_SEED_BASE = 990_000
_EVAL_PLANNER_SEED = 20260611
# Argmax tie-break: see Worker.collect_episodes(tiebreak_rng=...) — without it,
# the planner's uniform-fallback rows would deterministically pick action 0
# (NOOP) and trivially underestimate the planner [v4-opt 2026-06b].
_EVAL_TIEBREAK_SEED = 1_234_567


def _type_masks(type_assignment) -> tuple[np.ndarray, np.ndarray]:
    types = np.array([int(t) for t in type_assignment])
    return types == int(AgentType.ALPHA), types == int(AgentType.BETA)


@torch.no_grad()
def run_eval(
    model,
    cfg: V4Config,
    global_step: int = 0,
) -> dict[str, float]:
    """Dual-mode deterministic evaluation on the static-c grid.

    Returns:
        Flat dict of scalars; keys are TB tags *without* the ``eval/`` prefix,
        e.g. ``prior/return_total``, ``planner/return_alpha_c0.5``,
        ``planner_prior_gap``. NaN-free as long as at least one mode runs.
    """
    ecfg = cfg.eval
    c_grid = tuple(ecfg.eval_c_grid)
    assert len(c_grid) > 0, "eval_c_grid must not be empty"
    env_cfg = replace(cfg.env, c_mode="static")   # pin c for comparability
    alpha_mask, beta_mask = _type_masks(cfg.env.type_assignment)

    model.eval()
    results: dict[str, float] = {}
    mode_totals: dict[str, float] = {}

    modes = (
        ("prior", int(ecfg.eval_episodes_prior), False),
        ("planner", int(ecfg.eval_episodes_planner), True),
    )
    for mode, n_eps, use_planner in modes:
        if n_eps <= 0:
            continue

        envs, seeds, options = [], [], []
        for ci, c_val in enumerate(c_grid):
            for e in range(n_eps):
                seed = _EVAL_ENV_SEED_BASE + ci * 100 + e
                envs.append(ResourceCommonsEnv(env_cfg, seed=seed))
                seeds.append(seed)
                options.append({"c": float(c_val)})

        planner = MVEPlanner(cfg)
        planner.crn_rng = np.random.default_rng(_EVAL_PLANNER_SEED)
        worker = Worker(cfg, model, envs=envs, planner=planner)

        outs = worker.collect_episodes(
            epsilon=0.0,
            use_planner=use_planner,
            deterministic=True,
            reset_seeds=seeds,
            reset_options=options,
            tiebreak_rng=np.random.default_rng(_EVAL_TIEBREAK_SEED),
        )

        rets = np.stack([o.returns for o in outs], axis=0)        # (B, N)
        for ci, c_val in enumerate(c_grid):
            block = rets[ci * n_eps:(ci + 1) * n_eps]             # (n_eps, N)
            suffix = f"_c{c_val:g}"
            results[f"{mode}/return_total{suffix}"] = float(block.sum(axis=1).mean())
            if alpha_mask.any():
                results[f"{mode}/return_alpha{suffix}"] = float(block[:, alpha_mask].sum(axis=1).mean())
            if beta_mask.any():
                results[f"{mode}/return_beta{suffix}"] = float(block[:, beta_mask].sum(axis=1).mean())

        results[f"{mode}/return_total"] = float(rets.sum(axis=1).mean())
        if alpha_mask.any():
            results[f"{mode}/return_alpha"] = float(rets[:, alpha_mask].sum(axis=1).mean())
        if beta_mask.any():
            results[f"{mode}/return_beta"] = float(rets[:, beta_mask].sum(axis=1).mean())
        results[f"{mode}/ep_len"] = float(np.mean([len(o.records) for o in outs]))
        if use_planner:
            results[f"{mode}/pi_mve_entropy"] = float(np.mean([o.pi_entropy_mean for o in outs]))
            results[f"{mode}/q_std"] = float(np.mean([o.q_std_mean for o in outs]))
            results[f"{mode}/uniform_frac"] = float(np.mean([o.uniform_frac for o in outs]))
        mode_totals[mode] = results[f"{mode}/return_total"]

    if "prior" in mode_totals and "planner" in mode_totals:
        # Distillation residual: planner return minus distilled-policy return.
        # We compute it from the per-c means (each c contributes equally) rather
        # than the raw episode means, so the asymmetric episode counts
        # (eval_episodes_prior vs eval_episodes_planner) don't bias the gap.
        prior_per_c = [results[f"prior/return_total_c{c:g}"] for c in c_grid]
        planner_per_c = [results[f"planner/return_total_c{c:g}"] for c in c_grid]
        results["planner_prior_gap"] = float(np.mean(planner_per_c) - np.mean(prior_per_c))

    return results
