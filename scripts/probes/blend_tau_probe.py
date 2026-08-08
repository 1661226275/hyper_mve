#!/usr/bin/env python
"""Pick the ``visit_q_blend`` temperature bracket from measured logit scales.

Under ``--policy_target_type visit_q_blend`` the target logit is

    log(visit(c)) + adv_i(c) / tau

so tau is the ONLY knob trading the two terms off, and a bracket that does not
straddle the point where they are comparable measures nothing: every arm in it
is either the visit target or q_softmax under another name. Guessing that point
is exactly the mistake the lr sweep exists to prevent (0.02 vs 0.01 was 46.68 vs
23.78 -- the bracket matters more than the arm).

This measures both spans on a real checkpoint and reports the tau that equalizes
them:

    tau_balanced = span(adv) / span(log visit)

per regime and pooled. Cheap: one lockstep search batch per regime, same as
``adv_scale_probe.py``, which this deliberately mirrors so the two are
comparable.

Note the numbers here describe the CHECKPOINT's search, not the search a blend
run would produce -- the visit distribution co-evolves with the target. The
bracket should therefore be wide around the measured value, not pinned to it.
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = "/home/data/zhengwenbo/hyper_mve"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))


def collect(runner, env_fn, cfg, g, episodes, device, np_random):
    """One regime: per-root log-visit span and per-agent advantage span."""
    import numpy as np
    import torch
    from core.mcts import SampledMCTS
    from core.utils import select_action

    model = runner._model
    B = int(episodes)
    envs = [env_fn() for _ in range(B)]
    agents = list(envs[0].possible_agents)
    N = len(agents)
    A = int(cfg.env.A)
    subjective = hasattr(model, "belief_net")

    obs_dicts = [envs[i].reset(seed=10_000 + 97 * int(g) + i,
                               options={"g": int(g)})[0] for i in range(B)]
    hidden = (model.belief_net.init_hidden(B, N, device=device)
              if subjective else None)
    legal = np.ones((B, N, A), dtype=np.float32)
    mcts = SampledMCTS(runner._game_config, np_random)
    num_sims = int(runner._game_config.num_simulations)

    # RAW advantages are collected and normalized by the caller, because the
    # loss does not see raw advantages: reanalyze_worker divides by an adv_std
    # pooled over axes (0, 1) -- across the whole batch, per agent. Measuring
    # the span in raw units and calling the ratio "tau" would be wrong by
    # exactly that factor (0.37-0.67 here, i.e. ~2x).
    lv_spans, adv_raw, n_children, n_unvisited = [], [], [], 0
    done = False
    while not done:
        obs = np.stack([np.stack([od[a] for a in agents])
                        for od in obs_dicts]).astype(np.float32)
        obs_t = torch.from_numpy(obs).to(device)
        if subjective:
            hidden, g_hat = model.belief_net.step(obs_t, hidden)
            model.set_belief(g_hat)
        out = model.initial_inference(obs_t)
        search = mcts.batch_search(model, out, legal, device, False, 1.0)

        vvec = np.asarray(search.value_vec, dtype=np.float64)          # (B, N)
        for i in range(B):
            qv = np.asarray(search.sampled_qvalues_vec[i],
                            dtype=np.float64).reshape(-1, N)            # (K, N)
            vis = np.asarray(search.sampled_visit_count[i],
                             dtype=np.float64).reshape(-1)              # (K,)
            k = min(len(vis), qv.shape[0])
            vis, qv = vis[:k], qv[:k]
            n_children.append(k)
            n_unvisited += int((vis <= 0).sum())
            # the log-visit term as the loss actually builds it
            vp = vis / max(num_sims, 1)
            live = vp > 0
            if live.sum() >= 2:
                lv = np.log(vp[live])
                lv_spans.append(lv.max() - lv.min())
                adv_raw.append(qv[live] - vvec[i][None, :])             # (k', N)

        dones = np.zeros(B, dtype=bool)
        for i in range(B):
            pos, _ = select_action(search.sampled_visit_count[i], temperature=1,
                                   deterministic=True, np_random=np_random)
            joint = np.asarray(search.sampled_actions[i][pos]).reshape(-1)
            acts = {a: int(joint[k2]) for k2, a in enumerate(agents)}
            obs_dicts[i], _, term, trunc, _ = envs[i].step(acts)
            dones[i] = bool(any(term.values()) or any(trunc.values()))
        done = bool(dones.all())

    for e in envs:
        e.close()
    return (np.asarray(lv_spans), adv_raw, np.asarray(n_children), n_unvisited)


def main() -> int:
    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--gpus", default="0")
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
    arm = str(meta.get("ablation") or "none")
    env_key = str(meta.get("env", "relation_recip"))
    cfg = build_cfg(env_key, arm)
    env_fn = make_env_fn(cfg, env_key)

    runner = create_runner(cfg, "mazero_mixed")
    runner._ablation = arm
    runner.load_checkpoint(run_dir / "ckpt.pt")
    dev = runner._device_of(runner._model)
    np_random = np.random.RandomState(12345)
    # deliberately NOT overriding root_cover_mode/sampled_action_times the way
    # adv_scale_probe does: visits-per-child is the quantity under measurement
    # here, and it differs 5.0 vs 1.9 between root_cover none and star. Forcing
    # star would report every arm at star's budget.
    n_regimes = get_regime_family(cfg.env).size

    print(f"checkpoint: {run_dir}")
    print(f"arm={arm}  num_simulations={runner._game_config.num_simulations}  "
          f"sampled_action_times={runner._game_config.sampled_action_times}  "
          f"root_cover_mode={getattr(runner._game_config, 'root_cover_mode', '?')}")
    print()
    # pass 1: collect every regime, then normalize with ONE pooled adv_std, the
    # way reanalyze_worker does (axes (0,1) => over roots and children, per agent)
    per_g = {}
    for g in range(n_regimes):
        lv, adv_raw, nc, nun = collect(runner, env_fn, cfg, g, args.episodes,
                                       dev, np_random)
        if len(lv):
            per_g[g] = (lv, adv_raw, nc, nun)

    pooled = np.concatenate([np.concatenate(v[1], axis=0)
                             for v in per_g.values()], axis=0)      # (rows, N)
    adv_std = pooled.std(axis=0)                                    # (N,)
    print(f"pooled adv_std (reanalyze divides by this): "
          f"{np.round(adv_std, 4).tolist()}")
    print()
    print(" g   roots  children  log-visit span   adv span(norm)   tau_balanced")
    print("-" * 70)

    all_lv, all_sp = [], []
    for g, (lv, adv_raw, nc, nun) in per_g.items():
        sp = np.asarray([
            ((a.max(axis=0) - a.min(axis=0)) / np.maximum(adv_std, 1e-9)).mean()
            for a in adv_raw
        ])
        all_lv.append(lv)
        all_sp.append(sp)
        print(f"{g:2d}  {len(lv):6d}  {nc.mean():8.1f}  {lv.mean():14.3f}  "
              f"{sp.mean():15.3f}  {sp.mean() / max(lv.mean(), 1e-9):13.2f}")

    lv = np.concatenate(all_lv)
    sp = np.concatenate(all_sp)
    tau_bal = sp.mean() / max(lv.mean(), 1e-9)
    print("-" * 70)
    print(f"pooled: log-visit span {lv.mean():.3f}  "
          f"normalized adv span {sp.mean():.3f}")
    print(f"\ntau_balanced = {tau_bal:.2f}   "
          "(the two logit terms contribute equally here)")
    print(f"suggested bracket: {tau_bal/4:.2f}  {tau_bal/2:.2f}  "
          f"{tau_bal:.2f}  {tau_bal*2:.2f}")
    print("  low tau  -> advantage-dominated (approaches q_softmax)")
    print("  high tau -> visit-dominated (approaches the visit target)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
