#!/usr/bin/env python
"""Does the trained value head credit movement on REAL env trajectories?

Rolls the oracle-free eval env under two reference behaviours (scripted_greedy,
which walks to resources and scores ~99.79, and harvest_only, which camps and
scores 15.11), feeding every visited state through the belief GRU +
initial_inference exactly as ``runner._rollout_prior`` does.  Per regime it
reports the mean predicted team value on each trajectory family and the
correlation between V_t and the realized discounted return-to-go G_t.
H-value is confirmed at the model level iff V(greedy states) is not above
V(camp states) or V_t is uncorrelated with G_t.
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = "/home/data/zhengwenbo/hyper_mve"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))


def rollout(model, env_fn, policy, g: int, episodes: int, device, gamma: float):
    import numpy as np
    import torch

    env = env_fn()
    agents = list(env.possible_agents)
    N = len(agents)
    values, rewards, ep_returns = [], [], []
    for ep in range(int(episodes)):
        obs_dict, _ = env.reset(seed=10_000 + 97 * int(g) + ep,
                                options={"g": int(g)})
        hidden = model.belief_net.init_hidden(1, N, device=device)
        ep_vals, ep_rews, done = [], [], False
        while not done:
            obs = np.stack([obs_dict[a] for a in agents]).astype(np.float32)
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                hidden, g_hat = model.belief_net.step(obs_t, hidden)
                model.set_belief(g_hat)
                out = model.initial_inference(obs_t)
            v_team = float(np.asarray(out.value).reshape(N, -1)[:, 0].sum())
            acts = {a: int(policy(obs[i], i)) for i, a in enumerate(agents)}
            obs_dict, rew, term, trunc, _ = env.step(acts)
            ep_vals.append(v_team)
            ep_rews.append(float(sum(rew.values())))
            done = bool(any(term.values()) or any(trunc.values()))
        g_to_go, acc = [], 0.0
        for r in reversed(ep_rews):
            acc = r + gamma * acc
            g_to_go.append(acc)
        g_to_go.reverse()
        values.extend(ep_vals)
        rewards.extend(g_to_go)
        ep_returns.append(sum(ep_rews))
    env.close()
    return np.asarray(values), np.asarray(rewards), ep_returns


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--gpus", default="5")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.997)
    args = p.parse_args()

    from train import _set_gpus
    _set_gpus(args.gpus)

    import json
    import pathlib
    import numpy as np
    from train import build_cfg, make_env_fn
    from hyper_mve.comparison import create_runner
    from hyper_mve.envs.relation_commons.reference_policies import (
        harvest_only_policy,
        make_scripted_greedy_policy,
    )

    run_dir = pathlib.Path(args.run_dir).resolve()
    meta = json.loads((run_dir / "meta.json").read_text())
    arm = str(meta.get("ablation") or "none")
    cfg = build_cfg(str(meta.get("env", "relation")), arm)
    env_fn = make_env_fn(cfg, str(meta.get("env", "relation")))

    runner = create_runner(cfg, "mazero_mixed")
    runner._ablation = arm
    runner.load_checkpoint(run_dir / "ckpt.pt")
    model = runner._model
    model.eval()
    device = runner._device_of(model)

    greedy = make_scripted_greedy_policy(int(cfg.env.N), int(cfg.env.K),
                                         distinct_targets=True)
    print(f"checkpoint: {run_dir}  arm={arm}  grad={meta.get('train_steps_logged')}"
          f"  episodes/regime={args.episodes}  gamma={args.gamma}\n")
    print(f"{'g':>2s}  {'policy':13s} {'ep_return':>10s} {'mean V':>8s}"
          f" {'mean G_t':>9s} {'corr(V,G)':>10s}")

    verdict_v, verdict_c = [], []
    from hyper_mve.utils.schemas.relation import get_regime_family

    grid = cfg.eval.eval_regime_grid
    if grid is None:
        grid = tuple(range(get_regime_family(cfg.env).size))
    for g in [int(x) for x in grid]:
        row = {}
        for name, pol in (("scripted_greedy", greedy),
                          ("harvest_only", harvest_only_policy)):
            V, G, rets = rollout(model, env_fn, pol, g, args.episodes,
                                 device, args.gamma)
            corr = float(np.corrcoef(V, G)[0, 1]) if V.std() > 1e-9 else float("nan")
            print(f"{g:2d}  {name:13s} {np.mean(rets):10.2f} {V.mean():8.3f}"
                  f" {G.mean():9.2f} {corr:10.3f}")
            row[name] = (V.mean(), corr)
        dv = row["scripted_greedy"][0] - row["harvest_only"][0]
        verdict_v.append(dv)
        verdict_c.append(row["scripted_greedy"][1])
        print(f"    V(greedy) - V(camp) = {dv:+.3f}")

    print(f"\nmean over regimes: V(greedy)-V(camp) = {np.mean(verdict_v):+.3f},"
          f"  corr(V, G_t) on greedy = {np.nanmean(verdict_c):.3f}")
    if np.mean(verdict_v) <= 0 or np.nanmean(verdict_c) < 0.3:
        print("-> H-value CONFIRMED: the value head does not credit the states a"
              " good policy visits;\n   search built on it cannot rank movement"
              " above camping regardless of budget.")
    else:
        print("-> value head does separate greedy from camping states; the"
              " bottleneck is more likely\n   distillation/search-side, not"
              " value learning.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
