#!/usr/bin/env python
"""A/B the root cover on a FIXED, already-collapsed checkpoint (no retraining).

Stage A of the prior-collapse fix widens the root candidate set. This measures
what that alone buys the ACTING policy, holding the network constant: same
weights, same seeds, same episode count, only ``root_cover_mode`` differs.

It isolates stage A from stage B by construction — the policy target never
enters, because nothing is trained here. If planner return does not move, the
root candidate set was not the binding constraint at eval time and any gain
seen after retraining must come from the target change instead.

Reference scale: noop=0.00 random=1.39 harvest_only=15.11
                 scripted_greedy=99.79 ceiling~186
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
    ap.add_argument("--gpus", default="6")
    ap.add_argument("--episodes", type=int, default=8)
    args = ap.parse_args()

    from train import _set_gpus
    _set_gpus(args.gpus)

    import json
    import pathlib
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

    runner = create_runner(cfg, "mazero_mixed")
    runner._ablation = arm
    runner.load_checkpoint(run_dir / "ckpt.pt")
    dev = runner._device_of(runner._model)
    grid = tuple(range(get_regime_family(cfg.env).size))

    print(f"checkpoint: {run_dir}  arm={arm}  grad={meta.get('train_steps_logged')}")
    print(f"{'root_cover':>12s} {'return':>9s} {'a5_frac':>8s} {'children':>9s} "
          f"{'coverage':>9s} {'pi_mass':>8s} {'visit_ent':>10s}")

    base = None
    for mode, label in ((0, "none"), (1, "star")):
        runner._game_config.root_cover_mode = mode
        # buffer width is irrelevant at eval (nothing is written), but the tree
        # allocates from it, so it must admit the cover.
        runner._game_config.sampled_action_times = 13 if mode else 5
        runner._game_config.leaf_sampled_times = 5
        np_random = np.random.RandomState(12345)
        rets, a_counts = [], np.zeros(int(cfg.env.A))
        ch, cov, mass, ent = [], [], [], []
        for g in grid:
            r, ac, _, _, st = runner._rollout_planner(
                env_fn, int(g), args.episodes, dev, np_random)
            rets.extend(r)
            a_counts += ac
            ch.append(st["root_child_count"])
            cov.append(st["root_action_coverage"])
            mass.append(st["root_policy_mass_covered"])
            ent.append(st["visit_entropy"])
        mean = float(np.mean(rets))
        if base is None:
            base = mean
        print(f"{label:>12s} {mean:9.3f} {a_counts[5] / max(a_counts.sum(), 1):8.3f} "
              f"{float(np.mean(ch)):9.2f} {float(np.mean(cov)):9.3f} "
              f"{float(np.mean(mass)):8.3f} {float(np.mean(ent)):10.3f}"
              + ("" if mode == 0 else f"   ({mean - base:+.2f})"))

    print("\ncoverage 1.0 => every action present for every agent at the root.")
    print("A return gain here is stage A alone; the target change needs training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
