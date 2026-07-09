"""In-training periodic evaluation (v5 per-regime form; base [v4-opt 2026-06]).

    run_eval(model, cfg)  ->  flat {tag: float} dict (TB family ``eval/*``)

Protocol (v5 amendment of the 2026-06-11 dual-mode design):
    - **prior** mode: planner OFF — actions = argmax of the prediction net's own
      policy pi_hat. Measures the *distilled* policy the thesis ultimately ships.
    - **planner** mode: planner ON — actions = argmax of pi_mve. Measures the true
      acting agent. The planner-prior return gap is the distillation residual.
    - Both run deterministically (epsilon=0, argmax, no sampling RNG) on dedicated
      eval envs pinned to each regime ``g`` in the eval grid via
      ``reset(options={"g": gid})`` (bypasses ``train_regime_ids`` — held-out
      regimes ARE evaluable, that is the zero-shot probe), fixed reset seeds, and
      a constant planner CRN seed — so curves at different global_steps differ
      only through the model weights.

v5-added metric: ``belief/regime_accuracy`` (+ per-regime variants) — the
step-mean argmax accuracy of the BeliefNet regime posterior stored in the
collected records. Reported per regime because own-row aliasing across
regimes makes the global chance level misleading.

All episodes of one mode run as a single vectorized batch through
``Worker.collect_episodes`` (B = len(grid) * episodes), so one eval costs
about two batched episode rollouts.
"""
from __future__ import annotations

import numpy as np
import torch

from hyper_mve.configs import V4Config
from hyper_mve.envs.relation_commons import RelationCommonsEnv
from hyper_mve.planning.mve_planner import MVEPlanner
from hyper_mve.schemas import get_regime_family
from hyper_mve.training.worker import Worker

# Constant seeds: identical eval conditions at every call (CRN across evaluations).
_EVAL_ENV_SEED_BASE = 990_000
_EVAL_PLANNER_SEED = 20260611
# Argmax tie-break: see Worker.collect_episodes(tiebreak_rng=...) — without it,
# the planner's uniform-fallback rows would deterministically pick action 0
# (NOOP) and trivially underestimate the planner [v4-opt 2026-06b].
_EVAL_TIEBREAK_SEED = 1_234_567

# [2026-06 thesis welfare] commons-collapse threshold for the tragedy indicator
# (T = 1[S < 0.2]).
_TRAGEDY_THRESHOLD = 0.2


def _phys_fairness(w: np.ndarray) -> float:
    """Thesis fairness ``F = 1 − N·σ(w)/Σ(w)`` on per-agent physical harvest ``w``.

    Equals ``1 − coefficient_of_variation``; ``F ≤ 1`` (can go negative for
    N ≥ 3 under extreme inequality). Degenerate ``Σ(w) ≈ 0`` or ``N ≤ 1`` → 1.0
    (nothing to share / a single agent ⇒ trivially fair).
    """
    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 1 or w.size <= 1 or not np.all(np.isfinite(w)):
        return 1.0 if (w.size <= 1) else float("nan")
    total = float(w.sum())
    if abs(total) < 1e-9:
        return 1.0
    return float(1.0 - w.size * float(w.std()) / total)


def _regime_accuracy(records) -> float:
    """Step-mean argmax accuracy of the stored regime posterior vs oracle g."""
    correct, total = 0, 0
    for rec in records:
        pred = np.asarray(rec.g_hat).argmax(axis=-1)     # (N,)
        correct += int((pred == rec.g).sum())
        total += pred.size
    return correct / total if total else float("nan")


@torch.no_grad()
def run_eval(
    model,
    cfg: V4Config,
    global_step: int = 0,
) -> dict[str, float]:
    """Dual-mode deterministic evaluation on the regime grid.

    Returns:
        Flat dict of scalars; keys are TB tags *without* the ``eval/`` prefix,
        e.g. ``prior/return_total``, ``planner/return_total_g2``,
        ``belief/regime_accuracy_g2``, ``planner_prior_gap``.
    """
    ecfg = cfg.eval
    family = get_regime_family(cfg.env)
    if ecfg.eval_regime_grid is not None:
        regime_grid = tuple(int(g) for g in ecfg.eval_regime_grid)
    else:
        regime_grid = tuple(range(family.size))
    assert len(regime_grid) > 0, "eval regime grid must not be empty"

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
        for gi, gid in enumerate(regime_grid):
            for e in range(n_eps):
                seed = _EVAL_ENV_SEED_BASE + gi * 100 + e
                envs.append(RelationCommonsEnv(cfg.env, seed=seed))
                seeds.append(seed)
                options.append({"g": int(gid)})

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

        rets = np.stack([o.returns for o in outs], axis=0)        # (B, N) ΣR (subjective)
        # [2026-06 thesis welfare] physical harvest Σu (B, N) + sustainability (B,).
        phys = np.stack([
            np.asarray(o.phys_returns, dtype=np.float64)
            if getattr(o, "phys_returns", None) is not None else np.full(rets.shape[1], np.nan)
            for o in outs
        ], axis=0)                                                # (B, N)
        sus = np.array([float(getattr(o, "sustainability", np.nan)) for o in outs], dtype=np.float64)
        acc = np.array([_regime_accuracy(o.records) for o in outs], dtype=np.float64)

        for gi, gid in enumerate(regime_grid):
            sl = slice(gi * n_eps, (gi + 1) * n_eps)
            block = rets[sl]                                      # (n_eps, N)
            block_phys = phys[sl]
            block_sus = sus[sl]
            block_acc = acc[sl]
            suffix = f"_g{gid}"
            results[f"{mode}/return_total{suffix}"] = float(block.sum(axis=1).mean())
            # --- welfare metric family (Table 6.1) ---
            results[f"{mode}/welfare_physical{suffix}"] = float(block_phys.sum(axis=1).mean())
            fair = [_phys_fairness(block_phys[e]) for e in range(block_phys.shape[0])]
            fair = [f for f in fair if np.isfinite(f)]
            results[f"{mode}/fairness{suffix}"] = float(np.mean(fair)) if fair else float("nan")
            valid_sus = block_sus[np.isfinite(block_sus)]
            results[f"{mode}/sustainability{suffix}"] = (
                float(valid_sus.mean()) if valid_sus.size else float("nan"))
            results[f"{mode}/tragedy{suffix}"] = (
                float((valid_sus < _TRAGEDY_THRESHOLD).mean()) if valid_sus.size else float("nan"))
            # --- v5 belief quality (per regime — own-row aliasing) ---
            valid_acc = block_acc[np.isfinite(block_acc)]
            results[f"belief/regime_accuracy{suffix}_{mode}"] = (
                float(valid_acc.mean()) if valid_acc.size else float("nan"))

        results[f"{mode}/return_total"] = float(rets.sum(axis=1).mean())
        # overall welfare family (across all regimes) — also surfaced to TB.
        results[f"{mode}/welfare_physical"] = float(phys.sum(axis=1).mean())
        _fair_all = [_phys_fairness(phys[i]) for i in range(phys.shape[0])]
        _fair_all = [f for f in _fair_all if np.isfinite(f)]
        results[f"{mode}/fairness"] = float(np.mean(_fair_all)) if _fair_all else float("nan")
        _valid_all = sus[np.isfinite(sus)]
        if _valid_all.size:
            results[f"{mode}/sustainability"] = float(_valid_all.mean())
            results[f"{mode}/tragedy"] = float((_valid_all < _TRAGEDY_THRESHOLD).mean())
        _valid_acc_all = acc[np.isfinite(acc)]
        results[f"belief/regime_accuracy_{mode}"] = (
            float(_valid_acc_all.mean()) if _valid_acc_all.size else float("nan"))
        results[f"{mode}/ep_len"] = float(np.mean([len(o.records) for o in outs]))
        if use_planner:
            results[f"{mode}/pi_mve_entropy"] = float(np.mean([o.pi_entropy_mean for o in outs]))
            results[f"{mode}/q_std"] = float(np.mean([o.q_std_mean for o in outs]))
            results[f"{mode}/uniform_frac"] = float(np.mean([o.uniform_frac for o in outs]))
        mode_totals[mode] = results[f"{mode}/return_total"]

    if "prior" in mode_totals and "planner" in mode_totals:
        # Distillation residual: planner return minus distilled-policy return.
        # Computed from the per-regime means (each regime contributes equally)
        # so asymmetric episode counts don't bias the gap.
        prior_per_g = [results[f"prior/return_total_g{g}"] for g in regime_grid]
        planner_per_g = [results[f"planner/return_total_g{g}"] for g in regime_grid]
        results["planner_prior_gap"] = float(np.mean(planner_per_g) - np.mean(prior_per_g))

    return results
