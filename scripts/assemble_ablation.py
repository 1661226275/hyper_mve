#!/usr/bin/env python
"""Assemble the deploy-mode ablation table (A1-A5) + head_diversity back-half.

Reads a restart registry (default results/v5_final/registry.jsonl), pulls the
two mazero checkpoints per seed (main = subjective, and ref_bc_no_subjective),
and reads each run's eval_diagnostics.json to lay out the five deploy-mode
ablation points -- all from checkpoints already trained, no retraining:

  A1  Module2 = Bayes-avg (ours)     main   return_per_regime_planner      / return_mean
  A2  Module2 = argmax (MAP head)    main   return_per_regime_planner_map  / planner_map_return_mean
  A3  Module2 = no MCTS (prior)      main   return_per_regime_prior        / prior mean
  A4  Module1 removed (plain MAZero) nosubj return_per_regime_planner      / return_mean
  A5  Modules1&2 removed             nosubj return_per_regime_prior        / prior mean

For each subjective-mazero checkpoint it also reports the mean of
``train/head_diversity`` over the back half of training (env_step > budget/2),
read from the run's TensorBoard events -- the check that the per-regime value
heads keep diverging post-fix rather than re-collapsing toward 0.

Usage:
  python scripts/assemble_ablation.py --registry results/v5_final/registry.jsonl
  python scripts/assemble_ablation.py --root results/v5_final          # same default
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _mean(vals):
    vals = [float(v) for v in vals]
    return sum(vals) / len(vals) if vals else 0.0


def _load_diag(run_dir: pathlib.Path):
    p = run_dir / "eval_diagnostics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _regime_mean(d, key):
    """Mean over regimes of a {regime: return} dict field; None if absent."""
    m = d.get(key)
    if not m:
        return None, {}
    per = {int(g): float(v) for g, v in m.items()}
    return _mean(per.values()), per


def head_diversity_back_half(tb_dir: pathlib.Path, budget: int):
    """Mean of train/head_diversity for env_step > budget/2. None if unavailable."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except Exception:
        return None
    if not tb_dir.exists():
        return None
    acc = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    try:
        acc.Reload()
    except Exception:
        return None
    if "train/head_diversity" not in acc.Tags().get("scalars", []):
        return None
    half = budget / 2.0
    vals = [ev.value for ev in acc.Scalars("train/head_diversity") if ev.step > half]
    return _mean(vals) if vals else None


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--registry", default=None)
    p.add_argument("--root", default="results/v5_final")
    p.add_argument("--main-arm", default=None,
                   help="main subjective arm variant suffix, e.g. ref_bc or "
                        "ref_bc_anneal_scaled; auto-detected if omitted")
    p.add_argument("--out", default=None, help="write assembled JSON here")
    args = p.parse_args(argv)

    root = pathlib.Path(args.root)
    reg_path = pathlib.Path(args.registry) if args.registry else root / "registry.jsonl"
    if not reg_path.exists():
        raise SystemExit(f"registry not found: {reg_path}")

    rows = [json.loads(l) for l in reg_path.read_text().splitlines() if l.strip()]
    mazero = [r for r in rows if r.get("algo") == "mazero_mixed"
              and r.get("status") == "completed"]
    # main = subjective (variant not ending in _no_subjective); nosubj = the arm.
    by_seed: dict[int, dict] = {}
    for r in mazero:
        seed = int(r["seed"])
        variant = r.get("variant", "")
        role = "nosubj" if variant.endswith("no_subjective") else "main"
        by_seed.setdefault(seed, {})[role] = r

    assembled = {}
    for seed in sorted(by_seed):
        cell = by_seed[seed]
        out = {}
        main_r = cell.get("main")
        nos_r = cell.get("nosubj")
        if main_r:
            d = _load_diag(pathlib.Path(main_r["run_dir"]))
            if d:
                out["A1_bayes_mean"], out["A1_bayes_per_regime"] = _regime_mean(d, "return_per_regime_planner")
                out["A2_argmax_mean"], out["A2_argmax_per_regime"] = _regime_mean(d, "return_per_regime_planner_map")
                out["A3_prior_mean"], out["A3_prior_per_regime"] = _regime_mean(d, "return_per_regime_prior")
            budget = int(main_r.get("total_env_steps", 1_000_000))
            out["head_diversity_back_half"] = head_diversity_back_half(
                pathlib.Path(main_r["tensorboard_dir"]), budget)
            out["main_variant"] = main_r.get("variant")
        if nos_r:
            d = _load_diag(pathlib.Path(nos_r["run_dir"]))
            if d:
                out["A4_plainMAZero_mean"], out["A4_plainMAZero_per_regime"] = _regime_mean(d, "return_per_regime_planner")
                out["A5_prior_mean"], out["A5_prior_per_regime"] = _regime_mean(d, "return_per_regime_prior")
        assembled[seed] = out

    # print a compact table
    hdr = f"{'seed':>4} {'A1 bayes':>9} {'A2 argmax':>9} {'A3 prior':>9} {'A4 plain':>9} {'A5 prior':>9} {'head_div_bh':>11}"
    print(hdr)
    print("-" * len(hdr))
    for seed, o in assembled.items():
        def f(k):
            v = o.get(k)
            return f"{v:9.2f}" if isinstance(v, (int, float)) else f"{'--':>9}"
        hd = o.get("head_diversity_back_half")
        hd_s = f"{hd:11.5f}" if isinstance(hd, (int, float)) else f"{'--':>11}"
        print(f"{seed:>4} {f('A1_bayes_mean')} {f('A2_argmax_mean')} {f('A3_prior_mean')} "
              f"{f('A4_plainMAZero_mean')} {f('A5_prior_mean')} {hd_s}")

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(assembled, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
