#!/usr/bin/env python
"""Does more MCTS search at eval improve the return? (search-depth hypothesis)

Sweeps num_simulations on a FIXED checkpoint, no retraining. If planner return
climbs with simulations, deeper search is a real lever worth training with. If
it is flat, search depth is not the current bottleneck — consistent with a
model whose reward head cannot tell on-cell from off-cell, so rolling it forward
never reveals that walking pays.

The training/eval default is num_simulations=25; this sweeps around it.
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = "/home/data/zhengwenbo/hyper_mve"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--arm", default=None)
    ap.add_argument("--gpus", default="5")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--sims", default="10,25,50,100,200")
    args = ap.parse_args()

    from train import _set_gpus
    _set_gpus(args.gpus)

    import json, pathlib
    import numpy as np
    from train import build_cfg, make_env_fn
    from hyper_mve.comparison import create_runner
    from hyper_mve.utils.schemas.relation import get_regime_family

    run_dir = pathlib.Path(args.run_dir).resolve()
    meta = json.loads((run_dir / "meta.json").read_text())
    arm = args.arm or str(meta.get("ablation") or "none")
    env_id = str(meta.get("env", "relation"))
    cfg = build_cfg(env_id, arm)
    env_fn = make_env_fn(cfg, env_id)

    runner = create_runner(cfg, "mazero_mixed"); runner._ablation = arm
    runner.load_checkpoint(run_dir / "ckpt.pt")
    dev = runner._device_of(runner._model)
    grid = tuple(range(get_regime_family(cfg.env).size))

    print(f"checkpoint: {run_dir}  grad={meta.get('train_steps_logged')}  "
          f"episodes/regime={args.episodes}")
    print(f"reference: harvest_only=15.11  scripted_greedy=99.79  ceiling~186\n")
    print(f"{'num_sims':>9s} {'planner_return':>15s} {'a5_frac':>8s} {'visit_entropy':>14s}")

    base = None
    for S in (int(x) for x in args.sims.split(",")):
        runner._game_config.num_simulations = S
        np_random = np.random.RandomState(12345)
        rets, a_counts, v_ent = [], np.zeros(int(cfg.env.A)), []
        for g in grid:
            r, ac, _, _, st = runner._rollout_planner(
                env_fn, int(g), args.episodes, dev, np_random)
            rets.extend(r); a_counts += ac; v_ent.append(st["visit_entropy"])
        mean = float(np.mean(rets))
        a5 = a_counts[5] / max(a_counts.sum(), 1)
        if base is None:
            base = mean
        print(f"{S:9d} {mean:15.3f} {a5:8.3f} {float(np.mean(v_ent)):14.3f}"
              + (f"   ({mean-base:+.2f} vs sims=10)" if S != int(args.sims.split(',')[0]) else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
