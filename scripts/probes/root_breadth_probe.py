#!/usr/bin/env python
"""Pre-flight for the root-cover-enumeration plan: does a BROADER set of root
candidates improve the acting policy on an already-collapsed checkpoint?

The collapse hypothesis is that root children are sampled from the (collapsed)
prior, so `select_child_decoupled`'s `if (!present[a]) continue;` gate means
better actions are never even considered. If that is right, widening the root
candidate set must raise planner return on a FIXED checkpoint, with no
retraining and no C++ change.

Two existing knobs widen it without touching the tree:
  * sampled_tau  -- `batch_beta = batch_beta ** (1/tau)` (mcts_sampled.py:99),
                    so tau > 1 flattens the sampling distribution toward
                    uniform. tau -> inf approximates full enumeration.
  * sampled_action_times -- more joint draws => more distinct children. Free at
                    eval (nothing is written to the replay buffer here), so it
                    is not bounded by the training-time buffer width.

If return is FLAT across this sweep, the diagnosis is wrong: the search is
already considering the good actions and something else is the bottleneck.
Stop and re-derive before writing any C++.
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
    ap.add_argument("--taus", default="1,2,4,8")
    ap.add_argument("--samples", default="5,13,25")
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

    # _rollout_planner hardcodes add_noise=False / sampled_tau=1.0 at its
    # batch_search call, so sweep tau by patching the default in.
    from core.mcts import SampledMCTS
    orig = SampledMCTS.batch_search
    state = {"tau": 1.0}

    def patched(self, model, network_output, legal_actions_lst=None, device=None,
                add_noise=False, sampled_tau=1.0, sampled_actions_res=None):
        return orig(self, model, network_output, legal_actions_lst, device,
                    add_noise, state["tau"], sampled_actions_res)

    SampledMCTS.batch_search = patched

    print(f"checkpoint: {run_dir}  arm={arm}  grad={meta.get('train_steps_logged')}")
    print(f"reference: harvest_only=15.11  scripted_greedy=99.79  ceiling~186")
    print("tau=1 + samples=5 is the shipped configuration (the baseline row).\n")
    print(f"{'tau':>5s} {'samples':>8s} {'return':>9s} {'a5_frac':>8s} "
          f"{'visit_ent':>10s} {'vs base':>9s}")

    base = None
    try:
        for C in (int(x) for x in args.samples.split(",")):
            runner._game_config.sampled_action_times = C
            for tau in (float(x) for x in args.taus.split(",")):
                state["tau"] = tau
                np_random = np.random.RandomState(12345)
                rets, a_counts, v_ent = [], np.zeros(int(cfg.env.A)), []
                for g in grid:
                    r, ac, _, _, st = runner._rollout_planner(
                        env_fn, int(g), args.episodes, dev, np_random)
                    rets.extend(r)
                    a_counts += ac
                    v_ent.append(st["visit_entropy"])
                mean = float(np.mean(rets))
                a5 = a_counts[5] / max(a_counts.sum(), 1)
                if base is None:
                    base = mean
                print(f"{tau:5.1f} {C:8d} {mean:9.3f} {a5:8.3f} "
                      f"{float(np.mean(v_ent)):10.3f} {mean - base:+9.2f}")
    finally:
        SampledMCTS.batch_search = orig

    print("\nRead: if return rises with tau/samples, the root candidate set is "
          "the bottleneck and the plan is sound. If flat, STOP and re-derive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
