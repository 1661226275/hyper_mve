#!/usr/bin/env python
"""Decisive test: global representation issue vs reward-head-specific issue.

Freezes a trained checkpoint's representation (encoder latents) and fits FRESH
heads on behaviorally DIVERSE data (scripted_greedy + random + harvest_only
across all regimes), with a raw-observation control (the obs provably contains
the signal — the scripted policy decodes positions/stocks from it).

Probes, from easiest to hardest:
  P-on   : binary — "agent i stands on a stocked resource cell" (the harvest
           affordance). Trivially decodable from raw obs by construction.
  P-rew  : regression — true per-agent reward r_i from (input, joint action).
  P-val  : regression — per-agent discounted return-to-go G_t from input.

Reading (user's decision rule):
  fresh head fits from LATENT as well as from RAW-OBS -> representation is
    fine; the shipped head failed by optimization/data imbalance -> reweight /
    raise reward-loss weight (path 1).
  raw-obs fits but latent does not -> representation bottleneck; head-side
    changes cannot recover it.
  raw-obs itself does not fit -> probe underpowered; no conclusion.
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = "/home/data/zhengwenbo/hyper_mve"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

TRAIN_EPS, VAL_EPS, TEST_EPS = 9, 1, 2  # per (regime, policy) cell


def collect(model, env_fn, policies, regimes, episodes, device, gamma, N, K):
    import numpy as np
    import torch
    from hyper_mve.envs.relation_commons.reference_policies import resource_view

    X_h, X_a, X_o, Y_r, Y_g, Y_on, R_hat, EP = [], [], [], [], [], [], [], []
    ep_id = 0
    for g in regimes:
        for pol in policies.values():
            for ep in range(episodes):
                env = env_fn()
                agents = list(env.possible_agents)
                obs_dict, _ = env.reset(seed=10_000 + 97 * int(g) + ep,
                                        options={"g": int(g)})
                belief = torch.zeros(1, N, model.n_regimes, device=device)
                belief[:, :, int(g)] = 1.0
                rews, done = [], False
                while not done:
                    obs = np.stack([obs_dict[a] for a in agents]).astype(np.float32)
                    obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
                    acts = [int(pol(obs[i], i)) for i in range(N)]
                    a_t = torch.tensor([acts], device=device)
                    with torch.no_grad():
                        model.set_belief(belief)
                        out = model.initial_inference(obs_t)
                        h = torch.as_tensor(out.hidden_state).to(device)
                        out2 = model.recurrent_inference(h, a_t)
                    on = np.zeros(N, np.float32)
                    for i in range(N):
                        rv = resource_view(obs[i], N, K)  # (K, 3): dx, dy, stock
                        on[i] = float(np.any(
                            (np.abs(rv[:, 0]) < 1e-6) & (np.abs(rv[:, 1]) < 1e-6)
                            & (rv[:, 2] > 0.05)))
                    obs_dict, rew, term, trunc, _ = env.step(
                        {a: acts[i] for i, a in enumerate(agents)})
                    X_h.append(np.asarray(h.detach().cpu()).reshape(-1))
                    X_o.append(obs.reshape(-1))
                    onehot = np.zeros(N * model.action_space_size, np.float32)
                    for i, a in enumerate(acts):
                        onehot[i * model.action_space_size + a] = 1.0
                    X_a.append(onehot)
                    Y_r.append(np.array([rew[a] for a in agents], np.float32))
                    Y_on.append(on)
                    R_hat.append(np.asarray(out2.reward).reshape(N))
                    EP.append(ep_id)
                    rews.append(np.array([rew[a] for a in agents], np.float32))
                    done = bool(any(term.values()) or any(trunc.values()))
                env.close()
                gt, acc = [], np.zeros(N, np.float32)
                for r in reversed(rews):
                    acc = r + gamma * acc
                    gt.append(acc.copy())
                gt.reverse()
                Y_g.extend(gt)
                ep_id += 1
    return (np.asarray(X_h, np.float32), np.asarray(X_a, np.float32),
            np.asarray(X_o, np.float32), np.asarray(Y_r, np.float32),
            np.asarray(Y_g, np.float32), np.asarray(Y_on, np.float32),
            np.asarray(R_hat, np.float64), np.asarray(EP))


def standardize(Xtr, *rest):
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    return [(Xtr - mu) / sd] + [(X - mu) / sd for X in rest]


def ridge(Xtr, ytr, Xva, yva, Xte, yte):
    """Closed-form ridge; lambda picked on the val split. Returns test R2."""
    import numpy as np

    Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)
    XtX = Xtr.T @ Xtr
    Xty = Xtr.T @ (ytr - ytr.mean())
    best = (None, -np.inf)
    for lam in (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4):
        w = np.linalg.solve(XtX + lam * np.eye(Xtr.shape[1]), Xty)
        pv = Xva @ w + ytr.mean()
        r2v = 1 - ((pv - yva) ** 2).sum() / max(((yva - yva.mean()) ** 2).sum(), 1e-12)
        if r2v > best[1]:
            best = (w, r2v)
    w = best[0]
    pt = Xte @ w + ytr.mean()
    return 1 - float(((pt - yte) ** 2).sum()) / max(
        float(((yte - yte.mean()) ** 2).sum()), 1e-12)


def mlp_fit(Xtr, ytr, Xva, yva, Xte, yte, device, binary=False,
            hidden=128, epochs=1500, lr=1e-3, wd=1e-4):
    """MLP with early stopping on val. Returns test R2 (or balanced acc)."""
    import numpy as np
    import torch
    import torch.nn as nn

    Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    Xtr_t, ytr_t = t(Xtr), t(ytr).unsqueeze(1)
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], hidden), nn.ReLU(),
                        nn.Linear(hidden, 1)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    lossf = (nn.BCEWithLogitsLoss() if binary else nn.MSELoss())
    best_val, best_state, patience = -np.inf, None, 0
    for e in range(epochs):
        opt.zero_grad()
        lossf(net(Xtr_t), ytr_t).backward()
        opt.step()
        if e % 20 == 0:
            with torch.no_grad():
                pv = net(t(Xva)).cpu().numpy().reshape(-1)
            if binary:
                sv = _balanced_acc(pv > 0, yva)
            else:
                sv = 1 - ((pv - yva) ** 2).sum() / max(
                    ((yva - yva.mean()) ** 2).sum(), 1e-12)
            if sv > best_val:
                best_val, patience = sv, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                patience += 1
                if patience > 15:
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    with torch.no_grad():
        pt = net(t(Xte)).cpu().numpy().reshape(-1)
    if binary:
        return _balanced_acc(pt > 0, yte)
    return 1 - float(((pt - yte) ** 2).sum()) / max(
        float(((yte - yte.mean()) ** 2).sum()), 1e-12)


def _balanced_acc(pred, y):
    import numpy as np

    pos, neg = y > 0.5, y <= 0.5
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    return 0.5 * (float(pred[pos].mean()) + float(1 - pred[neg].mean()))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--gpus", default="5")
    p.add_argument("--gamma", type=float, default=0.997)
    args = p.parse_args()

    from train import _set_gpus
    _set_gpus(args.gpus)

    import json
    import pathlib
    import numpy as np
    from train import build_cfg, make_env_fn
    from hyper_mve.comparison import create_runner
    from hyper_mve.utils.schemas.relation import get_regime_family
    from hyper_mve.envs.relation_commons.reference_policies import (
        harvest_only_policy,
        make_random_policy,
        make_scripted_greedy_policy,
    )

    run_dir = pathlib.Path(args.run_dir).resolve()
    meta = json.loads((run_dir / "meta.json").read_text())
    arm = str(meta.get("ablation") or "none")
    cfg = build_cfg(str(meta.get("env", "relation")), arm)
    env_fn = make_env_fn(cfg, str(meta.get("env", "relation")))
    regimes = list(range(get_regime_family(cfg.env).size))
    N, K = int(cfg.env.N), int(cfg.env.K)

    runner = create_runner(cfg, "mazero_mixed")
    runner._ablation = arm
    runner.load_checkpoint(run_dir / "ckpt.pt")
    model = runner._model
    model.eval()
    device = runner._device_of(model)

    policies = {
        "scripted_greedy": make_scripted_greedy_policy(N, K, distinct_targets=True),
        "random": make_random_policy(seed=0),
        "harvest_only": harvest_only_policy,
    }
    episodes = TRAIN_EPS + VAL_EPS + TEST_EPS
    print(f"checkpoint: {run_dir}  arm={arm}  grad={meta.get('train_steps_logged')}")
    X_h, X_a, X_o, Y_r, Y_g, Y_on, R_hat, EP = collect(
        model, env_fn, policies, regimes, episodes, device, args.gamma, N, K)

    mod = EP % episodes
    tr = mod < TRAIN_EPS
    va = mod == TRAIN_EPS
    te = mod >= TRAIN_EPS + VAL_EPS
    print(f"transitions: {len(X_h)} (train {tr.sum()} / val {va.sum()} /"
          f" test {te.sum()}); on-stocked-cell base rate {Y_on.mean():.3f}")

    inputs = {"latent": (np.concatenate([X_h, X_a], 1), X_h),
              "raw-obs": (np.concatenate([X_o, X_a], 1), X_o)}

    for i in range(N):
        frozen = float(np.abs(R_hat[te, i] - Y_r[te, i]).mean())
        print(f"\nagent{i}: frozen-ckpt reward test MAE {frozen:.4f} "
              f"(mean-pred {float(np.abs(Y_r[te, i] - Y_r[tr, i].mean()).mean()):.4f})")
        print(f"  {'probe':7s} {'input':8s} {'ridge':>7s} {'MLP':>7s}")
        for iname, (Xr, Xv) in inputs.items():
            on_m = mlp_fit(Xv[tr], Y_on[tr, i], Xv[va], Y_on[va, i],
                           Xv[te], Y_on[te, i], device, binary=True)
            r_r = ridge(Xr[tr], Y_r[tr, i], Xr[va], Y_r[va, i], Xr[te], Y_r[te, i])
            r_m = mlp_fit(Xr[tr], Y_r[tr, i], Xr[va], Y_r[va, i],
                          Xr[te], Y_r[te, i], device)
            v_r = ridge(Xv[tr], Y_g[tr, i], Xv[va], Y_g[va, i], Xv[te], Y_g[te, i])
            v_m = mlp_fit(Xv[tr], Y_g[tr, i], Xv[va], Y_g[va, i],
                          Xv[te], Y_g[te, i], device)
            print(f"  {'P-on':7s} {iname:8s} {'-':>7s} {on_m:7.3f}   (balanced acc)")
            print(f"  {'P-rew':7s} {iname:8s} {r_r:7.3f} {r_m:7.3f}   (test R2)")
            print(f"  {'P-val':7s} {iname:8s} {v_r:7.3f} {v_m:7.3f}   (test R2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
