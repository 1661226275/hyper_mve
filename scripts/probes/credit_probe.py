#!/usr/bin/env python
"""Credit-assignment probe: does the learned value model see that walking to a
resource beats harvesting nothing in place?

Why this scenario. A cell regenerates at alpha*(Q_max - q), so continuous
harvesting settles at q* = 0.909 and pays ~0.909/step forever REGARDLESS of the
cell's starting stock. Camping is therefore near-optimal *once you stand on a
cell* (~93/episode/agent, and the scripted reference gets 182 total for N=2).
The entire task reduces to a one-time walk of <=7 steps to reach a cell — only
8 of 64 grid squares hold one, so a random spawn is off-cell 87% of the time.

So the sharp question is: standing OFF a cell, where HARVEST pays exactly 0,
does the model rank a move toward the cell above HARVEST? Ground truth is
overwhelming (walk-then-camp ~90 vs harvest-in-place 0). If the model gets this
wrong, the collapse is a value-model failure, not exploration and not the eval.

Belief is pinned to the true regime one-hot, so belief error is excluded as a
confound: this isolates the value/reward heads.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

REPO = "/home/data/zhengwenbo/hyper_mve"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

ACTIONS = ["NOOP", "UP", "DOWN", "LEFT", "RIGHT", "HARVEST"]
HARVEST = 5
GAMMA = 0.997


def build_state(env, d: int):
    """Agent 0 is d steps west of resource 0 and off every cell; agent 1 camps
    on resource 1 far away. All stocks full."""
    import numpy as np

    st = env._state
    K = st.resource_positions.shape[0]
    # Spread cells along a row well away from the agents' corridor.
    st.resource_positions[:] = np.array(
        [[2 + (k % 6), 6 if k % 2 else 7] for k in range(K)], dtype=np.int32)
    st.resource_positions[0] = np.array([4, 1], dtype=np.int32)
    st.resource_positions[1] = np.array([7, 6], dtype=np.int32)
    st.resource_stocks[:] = float(env.cfg.Q_max)
    st.agent_positions[0] = np.array([4 - d, 1], dtype=np.int32)   # d steps west
    st.agent_positions[1] = np.array([7, 6], dtype=np.int32)       # on resource 1
    assert not (st.agent_positions[0] == st.resource_positions).all(-1).any(), \
        "agent 0 must start off every cell"
    return st


def rollout_truth(env, d: int, g: int, agent0_policy: str) -> tuple[float, float]:
    """(undiscounted, discounted) return-to-go for agent 0. Agent 1 always camps."""
    e = copy.deepcopy(env)
    st = e._state
    total, disc = 0.0, 0.0
    for t in range(e.T_max - st.step_idx):
        if agent0_policy == "harvest":
            a0 = HARVEST
        else:  # walk east to resource 0, then camp
            dx = int(st.resource_positions[0][0] - st.agent_positions[0][0])
            dy = int(st.resource_positions[0][1] - st.agent_positions[0][1])
            a0 = 4 if dx > 0 else 3 if dx < 0 else 1 if dy > 0 else 2 if dy < 0 else HARVEST
        _, rew, done, _, _ = e.step(__import__("numpy").array([a0, HARVEST]))
        total += float(rew[0])
        disc += (GAMMA ** t) * float(rew[0])
        if done:
            break
    return total, disc


def model_q(runner, env, g: int):
    """Per-action Q_0 = r_0 + gamma * v_0(next), agent 1 held at HARVEST."""
    import numpy as np
    import torch
    from hyper_mve.envs.relation_commons.observations import build_joint_observation

    model = runner._model
    device = runner._device_of(model)
    n_regimes = env.family.size
    obs = build_joint_observation(env._state, env.L, env.T_max, env.K)
    obs_t = torch.from_numpy(obs).unsqueeze(0).float().to(device)

    with torch.no_grad():
        # True regime one-hot: excludes belief error from the measurement.
        belief = torch.zeros(1, env.N, n_regimes, device=device)
        belief[:, :, g] = 1.0
        model.set_belief(belief)
        out = model.initial_inference(obs_t)
        v_root = float(np.asarray(out.value).reshape(env.N, -1)[0, 0])

        qs, rs, vs = [], [], []
        for a0 in range(len(ACTIONS)):
            act = torch.tensor([[a0, HARVEST]], device=device)
            nxt = model.recurrent_inference(out.hidden_state, act)
            r = float(np.asarray(nxt.reward).reshape(env.N, -1)[0, 0])
            v = float(np.asarray(nxt.value).reshape(env.N, -1)[0, 0])
            rs.append(r); vs.append(v); qs.append(r + GAMMA * v)
    return v_root, rs, vs, qs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--arm", default=None)
    p.add_argument("--gpus", default="5")
    p.add_argument("--regimes", default="4,0")
    p.add_argument("--distances", default="1,2,3,5")
    args = p.parse_args()

    from train import _set_gpus
    _set_gpus(args.gpus)

    import json
    import pathlib
    import numpy as np
    from train import build_cfg
    from hyper_mve.comparison import create_runner
    from hyper_mve.envs.relation_commons import RelationCommonsEnv

    run_dir = pathlib.Path(args.run_dir).resolve()
    meta = json.loads((run_dir / "meta.json").read_text())
    arm = args.arm or str(meta.get("ablation") or "none")
    cfg = build_cfg(str(meta.get("env", "relation")), arm)

    runner = create_runner(cfg, "mazero_mixed")
    runner._ablation = arm
    runner.load_checkpoint(run_dir / "ckpt.pt")
    runner._model.eval()

    print(f"checkpoint : {run_dir}   arm={arm}  grad_steps={meta.get('train_steps_logged')}")
    print(f"gamma={GAMMA}; agent 0 starts OFF every cell, HARVEST there pays exactly 0\n")

    verdict_rows = []
    for g in (int(x) for x in args.regimes.split(",")):
        print(f"===== regime g={g} ({cfg.env.relation_family}) =====")
        for d in (int(x) for x in args.distances.split(",")):
            env = RelationCommonsEnv(cfg.env, seed=1234)
            env.reset(seed=1234, options={"g": g})
            build_state(env, d)

            v_root, rs, vs, qs = model_q(runner, env, g)
            t_harv_u, t_harv_d = rollout_truth(env, d, g, "harvest")
            t_walk_u, t_walk_d = rollout_truth(env, d, g, "walk")

            toward = 4  # RIGHT: resource 0 is due east
            best = int(np.argmax(qs))
            correct = qs[toward] > qs[HARVEST]
            verdict_rows.append(correct)

            print(f"  d={d}  truth: walk-then-camp {t_walk_u:7.2f} (disc {t_walk_d:6.2f})   "
                  f"harvest-in-place {t_harv_u:5.2f} (disc {t_harv_d:5.2f})")
            print(f"        model v(root)={v_root:7.3f}   "
                  f"Q[RIGHT]={qs[toward]:7.3f}  Q[HARVEST]={qs[HARVEST]:7.3f}  "
                  f"argmax={ACTIONS[best]}  {'OK' if correct else 'WRONG'}")
            print(f"        r̂: " + " ".join(f"{ACTIONS[i][:4]}={rs[i]:+.3f}" for i in range(6)))
            print(f"        v̂: " + " ".join(f"{ACTIONS[i][:4]}={vs[i]:+.3f}" for i in range(6)))
        print()

    n_ok = sum(verdict_rows)
    print(f"VERDICT: model prefers moving over harvesting-nothing in "
          f"{n_ok}/{len(verdict_rows)} probe states")
    if n_ok == 0:
        print("  -> value model NEVER sees the walk. Credit assignment has failed;")
        print("     more gradient steps is the plausible fix only if this improves with budget.")
    elif n_ok == len(verdict_rows):
        print("  -> value model DOES see the walk. The collapse is not a value-ranking")
        print("     failure at these states; look elsewhere (search depth / policy target).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
