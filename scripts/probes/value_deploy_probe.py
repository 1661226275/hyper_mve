#!/usr/bin/env python
"""Deploy-time value-aggregation diagnostic (2026-07-23).

On a checkpoint whose per-regime value heads DIFFERENTIATED during training
(ref_bc_hardval; head_diversity > 0), the deployed planner still Bayes-averages
the heads against a ~0.58-acc belief. This probe asks: are the heads correct, and
is the Bayes-mix against a blind belief the damage? It re-runs the planner eval per
regime under several deploy-time value aggregations (NO retraining), all on the same
checkpoint, with identical env seeds + RNG so the ONLY difference is aggregation:

  bayes    v = sum_g v_g * belief_g                    (current deploy; baseline)
  oracle   v = v_{g_true}                              (cheating upper bound: heads right?)
  argmax   v = v_{argmax belief}                       (hard-select belief's top pick)
  temp-T   v = sum_g v_g * softmax(log belief / T)     (tempered posterior; T<1 sharpens)

Reads per-regime planner return for each mode. If oracle >> bayes and the win
localizes to the aliased regimes (g0/g2/g3), the heads are correct and the belief is
the bottleneck; whether argmax/temp recover it depends on belief CALIBRATION.

Usage:
  /opt/anaconda3/bin/python scripts/probes/value_deploy_probe.py \
    --run-dir results_competence_600K/mazero_mixed_ref_bc_hardval/relation/seed0 \
    --episodes 16
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

_REGIME_NAMES = ["g0 coop", "g1 comp", "g2 explt", "g3 explable", "g4 neutrl"]


def run_mode(runner, env_fn, grid, episodes, device, mode, temp=1.0):
    """One deploy mode over the whole grid. Fresh RNG seeded identically per mode so
    env resets (seed-deterministic in _rollout_planner) + search RNG match across
    modes; the sole variable is the value aggregation."""
    import numpy as np
    from gymnasium.utils import seeding

    model = runner._model
    per_regime_returns = {}
    rng, _ = seeding.np_random(0)
    for g in grid:
        if mode == "oracle":
            model.set_value_deploy("oracle", deploy_g=int(g))
        elif mode == "argmax":
            model.set_value_deploy("argmax")
        elif mode == "temp":
            model.set_value_deploy("temp", temp=temp)
        else:
            model.set_value_deploy("bayes")
        rets, *_ = runner._rollout_planner(env_fn, int(g), int(episodes), device, rng)
        per_regime_returns[int(g)] = [float(x) for x in rets]
    model.set_value_deploy("bayes")  # restore trained default
    return per_regime_returns


def main(argv=None):
    import numpy as np
    from belief_confusion_probe import _load_runner
    from hyper_mve.utils.schemas.relation import get_regime_family

    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--arm", default=None)
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--temps", type=float, nargs="*", default=[0.3, 0.5, 0.7])
    args = p.parse_args(argv)

    run_dir = pathlib.Path(args.run_dir)
    runner, cfg, env_fn, arm, env_id, seed = _load_runner(run_dir, args.arm)
    model = runner._model
    model.eval()
    device = next(model.parameters()).device
    G = get_regime_family(cfg.env).size
    grid = tuple(range(G))

    modes = [("bayes", None), ("oracle", None), ("argmax", None)]
    modes += [("temp", t) for t in args.temps]

    results = {}
    for mode, temp in modes:
        label = mode if temp is None else f"temp{temp}"
        results[label] = run_mode(runner, env_fn, grid, args.episodes, device,
                                  mode, temp or 1.0)
        allret = [x for g in grid for x in results[label][g]]
        print(f"[{label:9s}] mean={np.mean(allret):7.3f} "
              f"sem={np.std(allret)/np.sqrt(len(allret)):5.3f}", flush=True)

    def mean_g(per, g):
        return float(np.mean(per[g]))

    def overall(per):
        allret = [x for g in grid for x in per[g]]
        return float(np.mean(allret)), float(np.std(allret) / np.sqrt(len(allret)))

    print(f"\nDeploy-time value aggregation | {run_dir}")
    print(f"arm={arm}  episodes/regime={args.episodes}  (team return per regime)\n")
    hdr = f"{'mode':9s} " + " ".join(f"{_REGIME_NAMES[g]:>10s}" for g in grid) + f" {'MEAN':>9s}"
    print(hdr)
    print("-" * len(hdr))
    for label, per in results.items():
        m, s = overall(per)
        row = f"{label:9s} " + " ".join(f"{mean_g(per, g):10.2f}" for g in grid) + f" {m:9.2f}"
        print(row)

    base = results["bayes"]
    print("\nΔ vs bayes (per-regime; the aliased regimes are g0/g2/g3):")
    for label, per in results.items():
        if label == "bayes":
            continue
        md = float(np.mean([mean_g(per, g) - mean_g(base, g) for g in grid]))
        row = f"{label:9s} " + " ".join(f"{mean_g(per, g) - mean_g(base, g):+10.2f}"
                                        for g in grid) + f" {md:+9.2f}"
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
