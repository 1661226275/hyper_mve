#!/usr/bin/env python
"""Can the LEARNED model, rolled forward, see that walking pays?

Search of any depth plans inside the world model: it composes dynamics + reward
+ bootstrap value over multi-step action sequences. So the ceiling on what
search can discover is what the model PREDICTS for those sequences. This rolls
two plans through the learned model from an off-cell start (agent 0, d steps west
of a full cell) and reports the model's own predicted discounted return:

    WALK:    RIGHT * d, then HARVEST forever   (true return ~90)
    CAMP:    HARVEST forever in place          (true return 0, off-cell)

If the model predicts WALK > CAMP, deeper search would find the walk and search
depth is the lever. If the model predicts WALK <= CAMP, no search depth helps —
the model cannot represent the payoff of moving onto a cell.
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = "/home/data/zhengwenbo/hyper_mve"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
HARVEST, RIGHT = 5, 4
GAMMA = 0.997


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--arm", default=None)
    ap.add_argument("--gpus", default="5")
    ap.add_argument("--horizon", type=int, default=40)
    args = ap.parse_args()

    from train import _set_gpus
    _set_gpus(args.gpus)

    import json, pathlib
    import numpy as np
    import torch
    from train import build_cfg
    from hyper_mve.comparison import create_runner
    from hyper_mve.envs.relation_commons import RelationCommonsEnv
    from hyper_mve.envs.relation_commons.observations import build_joint_observation

    run_dir = pathlib.Path(args.run_dir).resolve()
    meta = json.loads((run_dir / "meta.json").read_text())
    arm = args.arm or str(meta.get("ablation") or "none")
    cfg = build_cfg(str(meta.get("env", "relation")), arm)
    runner = create_runner(cfg, "mazero_mixed"); runner._ablation = arm
    runner.load_checkpoint(run_dir / "ckpt.pt")
    m = runner._model; m.eval(); dev = runner._device_of(m)
    N = m.num_agents

    def model_return(env, plan):
        """Discounted sum of the model's predicted rewards along an action plan,
        plus a bootstrap of its value at the end — exactly what MCTS backs up."""
        obs = build_joint_observation(env._state, env.L, env.T_max, env.K)
        ot = torch.from_numpy(obs).unsqueeze(0).float().to(dev)
        with torch.no_grad():
            bel = torch.zeros(1, N, env.family.size, device=dev)
            bel[:, :, env._state.g] = 1.0
            m.set_belief(bel)
            out = m.initial_inference(ot)
            h = out.hidden_state
            disc, g = 0.0, 1.0
            for a0 in plan:
                act = torch.tensor([[a0, HARVEST]], device=dev)
                nxt = m.recurrent_inference(h, act)
                r = float(np.asarray(nxt.reward).reshape(N, -1)[0, 0])
                disc += g * r
                g *= GAMMA
                h = nxt.hidden_state
            v_end = float(np.asarray(nxt.value).reshape(N, -1)[0, 0])
            return disc + g * v_end

    print(f"checkpoint: {run_dir}  grad={meta.get('train_steps_logged')}  horizon={args.horizon}\n")
    print(f"{'regime':10s} {'d':>2s} {'model WALK':>11s} {'model CAMP':>11s} {'prefers':>9s}"
          f"   {'true WALK':>10s} {'true CAMP':>10s}")
    fam = ["coop", "comp", "exploit", "exploited", "neutral"]
    for g in (4, 0):
        for d in (2, 4):
            env = RelationCommonsEnv(cfg.env, seed=1234)
            env.reset(seed=1234, options={"g": g})
            st = env._state
            st.resource_positions[0] = np.array([4, 1], dtype=np.int32)
            st.resource_stocks[:] = float(cfg.env.Q_max)
            st.agent_positions[0] = np.array([4 - d, 1], dtype=np.int32)
            st.agent_positions[1] = np.array([7, 6], dtype=np.int32)

            walk = [RIGHT] * d + [HARVEST] * (args.horizon - d)
            camp = [HARVEST] * args.horizon
            mw = model_return(env, walk)
            mc = model_return(env, camp)

            # ground truth by real-env rollout
            def truth(plan):
                e = __import__("copy").deepcopy(env); s = e._state; tot = 0.0; gg = 1.0
                for a0 in plan:
                    _, rew, done, _, _ = e.step(np.array([a0, HARVEST])); tot += gg * float(rew[0]); gg *= GAMMA
                    if done: break
                return tot
            tw, tc = truth(walk), truth(camp)
            pref = "WALK" if mw > mc else "CAMP"
            print(f"{fam[g]:10s} {d:>2d} {mw:11.3f} {mc:11.3f} {pref:>9s}   {tw:10.2f} {tc:10.2f}")
    print("\nmodel prefers CAMP where truth prefers WALK  =>  no search depth can find the")
    print("walk: the plan's predicted return is wrong at the source.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
