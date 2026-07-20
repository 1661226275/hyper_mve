#!/usr/bin/env python
"""Is the learned dynamics network action-sensitive at all?

The credit probe found v_hat identical across all 6 actions to 3 decimals, so
Q ranking collapses onto the reward head alone. Two explanations:
  (a) the dynamics net learned to ignore the action one-hot, or
  (b) it responds but the value head is flat over that response.
This measures the hidden-state response directly, and compares the trained net
against a freshly-initialized one. If random init is action-sensitive and the
trained net is not, the model actively unlearned movement — which is what you
would expect from a behaviour policy that emits HARVEST 100% of the time and
therefore generates no movement data to fit.
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = "/home/data/zhengwenbo/hyper_mve"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

ACTIONS = ["NOOP", "UP", "DOWN", "LEFT", "RIGHT", "HARVEST"]
HARVEST = 5


def probe(model, env, device, g: int, label: str):
    import numpy as np
    import torch
    from hyper_mve.envs.relation_commons.observations import build_joint_observation

    obs = build_joint_observation(env._state, env.L, env.T_max, env.K)
    obs_t = torch.from_numpy(obs).unsqueeze(0).float().to(device)
    with torch.no_grad():
        belief = torch.zeros(1, env.N, env.family.size, device=device)
        belief[:, :, g] = 1.0
        model.set_belief(belief)
        out = model.initial_inference(obs_t)
        h0 = out.hidden_state
        ref = None
        rows = []
        for a0 in range(6):
            act = torch.tensor([[a0, HARVEST]], device=device)
            nxt = model.recurrent_inference(h0, act)
            hs = nxt.hidden_state.float()
            if ref is None:
                ref = hs
            d = float(torch.linalg.norm(hs - ref))
            v = float(np.asarray(nxt.value).reshape(env.N, -1)[0, 0])
            r = float(np.asarray(nxt.reward).reshape(env.N, -1)[0, 0])
            rows.append((ACTIONS[a0], d, v, r))
        h_scale = float(torch.linalg.norm(ref))

    print(f"  --- {label}   ||h_next(NOOP)|| = {h_scale:.4f}")
    print(f"      {'action':9s} {'||dh|| vs NOOP':>15s} {'rel':>9s} {'v_hat':>10s} {'r_hat':>9s}")
    for name, d, v, r in rows:
        print(f"      {name:9s} {d:15.6f} {d/max(h_scale,1e-9):9.5f} {v:10.4f} {r:9.4f}")
    spread = max(v for _, _, v, _ in rows) - min(v for _, _, v, _ in rows)
    print(f"      value spread across actions: {spread:.6f}")
    return spread, max(d for _, d, _, _ in rows) / max(h_scale, 1e-9)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--arm", default=None)
    p.add_argument("--gpus", default="5")
    p.add_argument("--regime", type=int, default=4)
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

    env = RelationCommonsEnv(cfg.env, seed=1234)
    env.reset(seed=1234, options={"g": args.regime})
    st = env._state
    st.resource_positions[0] = np.array([4, 1], dtype=np.int32)
    st.resource_stocks[:] = float(cfg.env.Q_max)
    st.agent_positions[0] = np.array([2, 1], dtype=np.int32)
    st.agent_positions[1] = np.array([7, 6], dtype=np.int32)

    print(f"checkpoint: {run_dir}  arm={arm}  grad={meta.get('train_steps_logged')}\n")

    trained = create_runner(cfg, "mazero_mixed")
    trained._ablation = arm
    trained.load_checkpoint(run_dir / "ckpt.pt")
    trained._model.eval()
    dev = trained._device_of(trained._model)
    s_tr, d_tr = probe(trained._model, env, dev, args.regime, "TRAINED")

    print()
    fresh = create_runner(cfg, "mazero_mixed")
    fresh._ablation = arm
    fresh._weights_source = "random-init (probe baseline)"
    fresh._lazy_model().eval()
    s_rn, d_rn = probe(fresh._model, env, dev, args.regime, "RANDOM INIT")

    print()
    print(f"value spread   trained {s_tr:.6f}   random {s_rn:.6f}   "
          f"ratio {s_tr/max(s_rn,1e-12):.4f}")
    print(f"hidden response trained {d_tr:.6f}   random {d_rn:.6f}   "
          f"ratio {d_tr/max(d_rn,1e-12):.4f}")
    if s_tr < 0.01 and s_rn > s_tr * 5:
        print("\n-> TRAINED net is action-blind where random init is not: the dynamics")
        print("   model was fit on data containing only HARVEST, so it never learned")
        print("   what moving does. More gradient steps on the same collapsed behaviour")
        print("   policy generate no movement data and cannot fix this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
