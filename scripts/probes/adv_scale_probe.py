#!/usr/bin/env python
"""Per-regime advantage scale: is the Q-softmax target's effective temperature
regime-dependent?

``reanalyze_worker.py`` normalizes the per-agent advantage by statistics
reduced over axes ``(0, 1)`` — per-agent, but POOLED ACROSS REGIMES:

    batch_sampled_adv = (q - root_pred_value - adv_mean) / (adv_std + 1e-5)

Under ``--policy_target_type q_softmax`` that normalized advantage IS the
policy target's logit, so ``adv_std`` acts as an inverse temperature. If a
regime's own advantage spread differs from the pooled spread, that regime's
target is systematically sharpened (own_std > pooled) or flattened
(own_std < pooled) relative to what its own Q values justify.

The distortion factor reported here is ``own_std / pooled_std``:
    ~1.0  -> pooled normalization is fine for that regime
    <1    -> regime's target is FLATTENED (its logits shrink)
    >1    -> regime's target is SHARPENED

**A distortion != 1 is not automatically a bug.** If a regime genuinely offers
less action discrimination, a flatter target there is correct — and pooling is
what preserves that. The distinguishing evidence is the reward column: a
regime with a small advantage spread but a LARGE spread in realized per-action
return is a value head failing on that regime, not a regime where actions do
not matter. Both are printed side by side.

Running the same probe on the ``no_subjective`` arm (single value head, no
Bayes averaging over the regime family) isolates whether any distortion comes
from the Bayes average specifically.
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = "/home/data/zhengwenbo/hyper_mve"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))


def collect(runner, env_fn, cfg, g, episodes, device, np_random):
    """One regime: raw per-agent advantages over every root child, every step."""
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

    advs, rootv, qspread, rets = [], [], [], np.zeros(B)
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

        vvec = np.asarray(search.value_vec, dtype=np.float64)        # (B, N)
        for i in range(B):
            qv = np.asarray(search.sampled_qvalues_vec[i],
                            dtype=np.float64).reshape(-1, N)          # (K, N)
            advs.append(qv - vvec[i][None, :])                        # (K, N)
            qspread.append(qv.max(axis=0) - qv.min(axis=0))           # (N,)
        rootv.append(vvec)

        dones = np.zeros(B, dtype=bool)
        for i in range(B):
            pos, _ = select_action(search.sampled_visit_count[i], temperature=1,
                                   deterministic=True, np_random=np_random)
            joint = np.asarray(search.sampled_actions[i][pos]).reshape(-1)
            acts = {a: int(joint[k]) for k, a in enumerate(agents)}
            obs_dicts[i], rew, term, trunc, _ = envs[i].step(acts)
            rets[i] += float(sum(rew.values()))
            dones[i] = bool(any(term.values()) or any(trunc.values()))
        done = bool(dones.all())

    for e in envs:
        e.close()
    return (np.concatenate(advs, axis=0), np.concatenate(rootv, axis=0),
            np.stack(qspread, axis=0), rets)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--arm", default=None)
    ap.add_argument("--gpus", default="4")
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
    cfg = build_cfg(str(meta.get("env", "relation")), arm)
    env_fn = make_env_fn(cfg, str(meta.get("env", "relation")))

    runner = create_runner(cfg, "mazero_mixed")
    runner._ablation = arm
    runner.load_checkpoint(run_dir / "ckpt.pt")
    dev = runner._device_of(runner._model)
    # match the shipped training config so the numbers are the ones that matter
    runner._game_config.root_cover_mode = 1
    runner._game_config.sampled_action_times = 13
    runner._game_config.leaf_sampled_times = 5
    grid = tuple(range(get_regime_family(cfg.env).size))

    bayes = hasattr(runner._model, "belief_net")
    print(f"checkpoint: {run_dir}")
    print(f"arm={arm}  bayes_averaged_value_head={bayes}  "
          f"grad={meta.get('train_steps_logged')}\n")

    np_random = np.random.RandomState(12345)
    per_g = {}
    for g in grid:
        per_g[g] = collect(runner, env_fn, cfg, g, args.episodes, dev, np_random)

    all_adv = np.concatenate([per_g[g][0] for g in grid], axis=0)   # (rows, N)
    pooled_std = all_adv.std(axis=0)                                # (N,)
    print(f"pooled adv_std (what reanalyze_worker actually divides by): "
          f"{np.round(pooled_std, 4).tolist()}\n")

    hdr = (f"{'g':>2s} {'adv_std(a0)':>12s} {'adv_std(a1)':>12s} "
           f"{'distortion':>11s} {'root_v':>9s} {'q_spread':>9s} "
           f"{'return':>9s}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for g in grid:
        adv, rootv, qsp, rets = per_g[g]
        s = adv.std(axis=0)
        dist = float(np.mean(s / (pooled_std + 1e-12)))
        rows.append((g, dist, float(np.mean(qsp)), float(np.mean(rets))))
        print(f"{g:2d} {s[0]:12.4f} {s[1]:12.4f} {dist:11.3f} "
              f"{float(np.mean(rootv)):9.3f} {float(np.mean(qsp)):9.4f} "
              f"{float(np.mean(rets)):9.3f}")

    d = np.array([r[1] for r in rows])
    print(f"\ndistortion spread: min={d.min():.3f} max={d.max():.3f} "
          f"ratio={d.max() / max(d.min(), 1e-9):.2f}x")
    print("ratio <1.5x  -> pooling is harmless; leave normalization alone.")
    print("ratio >=1.5x -> check the q_spread/return columns before acting: a "
          "regime with small\n                adv_std AND small q_spread is "
          "signal (actions really don't\n                discriminate there); "
          "small adv_std with a large return spread is a\n                "
          "value head failing on that regime.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
