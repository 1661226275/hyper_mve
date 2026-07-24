#!/usr/bin/env python
"""belief_confusion_probe.py — offline belief-net error-structure diagnostic.

Answers "WHERE does the ~0.53 belief accuracy fail, and does that error even
matter for the value the planner consumes?" on an existing checkpoint — no
retraining, no training grid. The loader is lifted from
``scripts/reeval_checkpoint.py`` (arm-correct checkpoint loading).

Per (episode, agent, timestep) on the distilled-prior rollout (which reproduces
``regime_accuracy`` bit-for-bit — the acting policy uses the inferred belief;
the oracle pass is a pure side-probe), it records the argmax posterior, its
confidence, and the VALUE SWING between belief-conditioned and oracle-conditioned
``initial_inference``. Outputs:

  * 5x5 confusion matrix + per-regime accuracy (the diagonal).
  * calibration: mean max-prob / entropy split by correct/wrong (confident-wrong
    poisons hard; uncertain-wrong self-hedges).
  * accuracy vs within-episode timestep (early-episode aliasing vs flat ceiling).
  * per-role split in g2/g3 (accuracy AND return by agent) — the exploited agent.
  * value-belief coupling: |V(belief) - V(oracle)| bucketed by correct/wrong x
    regime. Near-zero swing => belief is NOT the binding constraint.
  * distribution shift (--planner-dist): same belief net's accuracy under the
    de-collapsed PLANNER rollout vs the collapsed prior rollout.

    python scripts/probes/belief_confusion_probe.py --gpus 3 --episodes 16 \
        --run-dir results/mazero_mixed/relation/seed0 \
        --run-dir results_fixval/mazero_mixed/relation/seed0 \
        --run-dir results_fixval/mazero_mixed_mcts_fix_oracle/relation/seed0
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

_REGIME_NAMES = ["g0 coop", "g1 comp", "g2 exploit", "g3 exploited", "g4 neutral"]
_T_BUCKETS = [(0, 1), (1, 5), (5, 10), (10, 20), (20, 100_000)]   # [lo, hi)


def _load_runner(run_dir: pathlib.Path, arm_override):
    from reeval_checkpoint import _arm_from_run_dir     # noqa: E402
    from train import build_cfg, make_env_fn            # noqa: E402
    from hyper_mve.comparison import create_runner      # noqa: E402

    ckpt = run_dir / "ckpt.pt"
    if not ckpt.exists():
        raise SystemExit(f"{run_dir}: no ckpt.pt")
    arm, env_id, seed = _arm_from_run_dir(run_dir, arm_override)
    cfg = build_cfg(env_id, arm)
    env_fn = make_env_fn(cfg, env_id)
    runner = create_runner(cfg, "mazero_mixed")
    runner._ablation = arm
    runner.load_checkpoint(ckpt)
    return runner, cfg, env_fn, arm, env_id, seed


def prior_probe(runner, cfg, env_fn, grid, episodes):
    """Batch-1 distilled-prior rollout with per-step belief + value-swing records."""
    import numpy as np
    import torch
    from hyper_mve.utils.schemas.relation import get_regime_family

    model = runner._model
    model.eval()
    if not hasattr(model, "belief_net"):
        return None
    device = next(model.parameters()).device
    N = int(cfg.env.N)
    G = get_regime_family(cfg.env).size

    confusion = np.zeros((G, G), dtype=np.int64)
    records = []                                   # per (g, agent, t)
    returns_by_role = {}                           # (g, agent) -> [ep returns]

    for g in grid:
        env = env_fn()
        agents = list(env.possible_agents)
        for ep in range(int(episodes)):
            obs_dict, _ = env.reset(seed=10_000 + 97 * int(g) + ep,
                                    options={"g": int(g)})
            hidden = model.belief_net.init_hidden(1, N, device=device)
            ep_ret_agent = np.zeros(N, dtype=np.float64)
            t, done = 0, False
            while not done:
                obs = np.stack([obs_dict[a] for a in agents]).astype(np.float32)
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)

                hidden, g_hat = model.belief_net.step(obs_t, hidden)
                gh = g_hat.detach().float().cpu().numpy().reshape(N, G)
                pred = gh.argmax(-1)
                maxp = gh.max(-1)
                ent = -(gh * np.log(gh + 1e-12)).sum(-1)

                # value + policy under the INFERRED belief (this is what acts)
                model.set_belief(g_hat)
                out_b = model.initial_inference(obs_t)
                val_b = np.asarray(out_b.value).reshape(N)
                pol = np.asarray(out_b.policy_logits).reshape(N, -1)

                # value under the ORACLE one-hot (pure side-probe, does not act)
                onehot = torch.zeros(1, N, G, device=device)
                onehot[..., int(g)] = 1.0
                model.set_belief(onehot)
                out_o = model.initial_inference(obs_t)
                val_o = np.asarray(out_o.value).reshape(N)

                for i in range(N):
                    confusion[int(g), int(pred[i])] += 1
                    records.append((int(g), i, t, int(pred[i]),
                                    int(pred[i] == g), float(maxp[i]),
                                    float(ent[i]), float(val_b[i]),
                                    float(val_o[i]), abs(float(val_b[i] - val_o[i]))))

                acts = {a: int(pol[i].argmax()) for i, a in enumerate(agents)}
                obs_dict, rew, term, trunc, _ = env.step(acts)
                for i, a in enumerate(agents):
                    ep_ret_agent[i] += float(rew[a])
                t += 1
                done = bool(any(term.values()) or any(trunc.values()))
            for i in range(N):
                returns_by_role.setdefault((int(g), i), []).append(float(ep_ret_agent[i]))
        env.close()

    return {"confusion": confusion, "records": records,
            "returns_by_role": returns_by_role, "N": N, "G": G}


def planner_dist_accuracy(runner, env_fn, grid, episodes):
    """Belief accuracy under the DE-COLLAPSED planner rollout, via a set_belief
    wrapper — the distribution-shift signal against the prior-rollout accuracy."""
    import numpy as np
    model = runner._model
    device = next(model.parameters()).device
    orig = model.set_belief
    state = {"g": None, "hits": 0, "n": 0}

    def wrapped(g_hat, step=None):
        gh = g_hat.detach().float().cpu().numpy()
        pred = gh.argmax(-1)
        state["hits"] += int((pred == state["g"]).sum())
        state["n"] += int(pred.size)
        return orig(g_hat, step)

    model.set_belief = wrapped
    per_g = {}
    try:
        np_random = np.random.default_rng(0)
        for g in grid:
            state["g"] = int(g)
            h0, n0 = state["hits"], state["n"]
            runner._rollout_planner(env_fn, int(g), int(episodes), device, np_random)
            per_g[int(g)] = (state["hits"] - h0) / max(state["n"] - n0, 1)
    finally:
        model.set_belief = orig
    overall = state["hits"] / max(state["n"], 1)
    return per_g, overall


def summarize(res, label):
    import numpy as np
    conf = res["confusion"]
    G = res["G"]
    recs = np.array([r[:5] for r in res["records"]], dtype=np.float64)  # g,agent,t,pred,correct
    conf_f = res["records"]
    g_arr = recs[:, 0].astype(int)
    agent_arr = recs[:, 1].astype(int)
    t_arr = recs[:, 2].astype(int)
    correct = recs[:, 4].astype(bool)
    maxp = np.array([r[5] for r in conf_f])
    ent = np.array([r[6] for r in conf_f])
    vswing = np.array([r[9] for r in conf_f])

    row_sums = conf.sum(1)
    per_regime_acc = {g: (float(conf[g, g]) / row_sums[g] if row_sums[g] else None)
                      for g in range(G)}
    overall_acc = float(conf.trace()) / max(conf.sum(), 1)

    print("\n" + "=" * 74)
    print(f"{label}   overall belief accuracy = {overall_acc:.4f}  (chance {1/G:.2f})")
    print("-" * 74)
    print("confusion (rows = g_true, cols = argmax g_hat):")
    print("        " + " ".join(f"{'p'+str(c):>6s}" for c in range(G)))
    for g in range(G):
        acc = per_regime_acc[g]
        print(f"  g{g} {_REGIME_NAMES[g][3:11]:9s} " +
              " ".join(f"{conf[g, c]:6d}" for c in range(G)) +
              f"   acc={acc:.3f}" if acc is not None else "")

    # calibration + value swing, correct vs wrong
    print("\ncalibration & value-swing (mean over steps):")
    print(f"  {'bucket':16s} {'n':>7s} {'maxprob':>8s} {'entropy':>8s} {'|V_bel-V_ora|':>14s}")
    for name, mask in (("correct", correct), ("wrong", ~correct)):
        if mask.sum():
            print(f"  {name:16s} {int(mask.sum()):7d} {maxp[mask].mean():8.3f} "
                  f"{ent[mask].mean():8.3f} {vswing[mask].mean():14.3f}")

    # per-regime value swing
    print("\nvalue swing |V(belief) - V(oracle)| per regime (wrong-belief steps):")
    for g in range(G):
        m = (g_arr == g) & (~correct)
        mc = (g_arr == g) & correct
        sw = vswing[m].mean() if m.sum() else float("nan")
        swc = vswing[mc].mean() if mc.sum() else float("nan")
        print(f"  g{g} {_REGIME_NAMES[g][3:11]:9s} wrong={sw:7.3f} (n={int(m.sum()):4d})  "
              f"correct={swc:7.3f} (n={int(mc.sum()):4d})")

    # timestep curve
    print("\naccuracy vs within-episode timestep:")
    for lo, hi in _T_BUCKETS:
        m = (t_arr >= lo) & (t_arr < hi)
        if m.sum():
            print(f"  t in [{lo:3d},{hi if hi < 1000 else '.':>3}) : "
                  f"acc={correct[m].mean():.3f}  (n={int(m.sum())})")

    # per-role g2/g3
    print("\nper-role split (g2 exploit / g3 exploited) — belief acc & mean return:")
    rbr = res["returns_by_role"]
    for g in (2, 3):
        for i in range(res["N"]):
            m = (g_arr == g) & (agent_arr == i)
            acc = correct[m].mean() if m.sum() else float("nan")
            rets = rbr.get((g, i), [])
            mret = float(np.mean(rets)) if rets else float("nan")
            print(f"  g{g} agent{i}: belief_acc={acc:.3f}  mean_return={mret:7.2f}")

    return {"overall_acc": overall_acc, "per_regime_acc": per_regime_acc,
            "confusion": conf.tolist(),
            "vswing_wrong_mean": float(vswing[~correct].mean()) if (~correct).sum() else None,
            "vswing_correct_mean": float(vswing[correct].mean()) if correct.sum() else None,
            "maxprob_wrong_mean": float(maxp[~correct].mean()) if (~correct).sum() else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", action="append", required=True,
                    help="results/<algo>[_<arm>]/<env>/seed<k>; repeatable")
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--arm", default=None, help="override ablation arm from meta.json")
    ap.add_argument("--gpus", default="3")
    ap.add_argument("--planner-dist", action="store_true",
                    help="also measure belief accuracy under the planner rollout")
    args = ap.parse_args(argv)

    from train import _set_gpus
    _set_gpus(args.gpus)                       # before torch import

    from hyper_mve.utils.schemas.relation import get_regime_family

    out_all = {}
    for rd in args.run_dir:
        run_dir = pathlib.Path(rd).resolve()
        print(f"\n[probe] loading {run_dir}", flush=True)
        runner, cfg, env_fn, arm, env_id, seed = _load_runner(run_dir, args.arm)
        G = get_regime_family(cfg.env).size
        grid = tuple(range(G))

        res = prior_probe(runner, cfg, env_fn, grid, args.episodes)
        label = f"{arm}/{env_id}/seed{seed}"
        if res is None:
            print(f"  {label}: no belief_net (plain-MAZero arm) — skipped")
            continue
        summary = summarize(res, label)

        if args.planner_dist:
            try:
                per_g, overall = planner_dist_accuracy(runner, env_fn, grid, args.episodes)
                print(f"\ndistribution shift — belief accuracy by rollout policy:")
                print(f"  prior (collapsed) rollout : {summary['overall_acc']:.4f}")
                print(f"  planner (de-collapsed)    : {overall:.4f}  per-regime "
                      + " ".join(f"g{g}={per_g[g]:.2f}" for g in grid))
                summary["planner_dist_overall"] = overall
                summary["planner_dist_per_regime"] = per_g
            except Exception as e:            # noqa: BLE001
                print(f"  [planner-dist skipped: {type(e).__name__}: {e}]")

        out_all[label] = summary

    out_dir = pathlib.Path(REPO_ROOT) / "results" / "analysis" / "belief_ceiling"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "belief_confusion.json"
    out_path.write_text(json.dumps(out_all, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
