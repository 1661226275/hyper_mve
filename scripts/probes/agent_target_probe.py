#!/usr/bin/env python
"""Gate the action-axis policy target, and measure its temperature.

``agent_q_softmax`` differs from ``q_softmax`` in exactly one way: it
aggregates the root's children onto per-agent action slots by a visit-weighted
AVERAGE inside the exponent rather than a sum outside it. Both are per-agent
targets over the same ``(N, A)`` grid -- because the policy head is factorized,
every target is -- so the two coincide whenever each agent's child->action map
is injective, and differ only through COLLISIONS::

    q_softmax        W_i(a) = sum_{c: a_i^c=a} exp(adv_i(c) / tau)
    agent_q_softmax  T_i(a) prop. exp( [sum_{c: a_i^c=a} visit(c) adv_i(c)] /
                                       [sum_{c: a_i^c=a} visit(c)] / tau )

So there are exactly two ways to waste a wave here, and this probe measures
both before anything is launched.

**1. It could be a no-op.** If collisions are rare, ``TV(W_i, T_i) ~ 0`` and
the arm re-measures ``q_softmax`` under a new name. That is not hypothetical:
the proposal this target replaced -- feeding the C++ tree's
``marginal_visit_count`` to the loss -- turned out to be *exactly* the shipped
``visit`` target, for the same structural reason. TV is the decisive column.

**2. The temperature could confound it.** Averaging within a group shrinks the
advantage span, so at a fixed ``tau`` the action-axis target is systematically
FLATTER. Running the wave at ``tau=1.0`` would then compare "per-agent
marginalization" against "a flatter target" and could not tell them apart.
``span(qbar) / span(adv)`` is the correction factor. This is the same mistake
``blend_tau_probe.py`` was written to prevent, one level up.

Also reported, because they cost nothing and settle open questions:
per-agent action multiplicity and the target mass sitting on the most-repeated
action (the mechanism predicted for ``--root_cover star``, where the cover pins
each agent at one draw from its own noised prior across ``A`` of the ~``2A``
children); the realized visits-per-child; and the count of DISTINCT sampled
root actions per agent, which ``v6_reporting_protocol.md`` flags as the direct
test separating "collapsed action set" from "flat game".

Advantages are normalized by ONE pooled ``adv_std`` over all regimes, per
agent, exactly as ``reanalyze_worker`` does -- the loss never sees raw
advantages, and reporting a span in raw units is wrong by that factor (~2x).
Mirrors ``blend_tau_probe.py`` / ``adv_scale_probe.py`` so the three compare.
"""
from __future__ import annotations

import argparse
import os
import sys

# derived, not hardcoded, so the probe measures the checkout it ships in --
# a worktree copy must not silently import the main checkout's code.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))


def collect(runner, env_fn, cfg, g, episodes, device, np_random):
    """One regime: per-root (actions, visits, raw advantages) at the root."""
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

    roots = []
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

        vvec = np.asarray(search.value_vec, dtype=np.float64)           # (B, N)
        for i in range(B):
            qv = np.asarray(search.sampled_qvalues_vec[i],
                            dtype=np.float64).reshape(-1, N)            # (K, N)
            vis = np.asarray(search.sampled_visit_count[i],
                             dtype=np.float64).reshape(-1)              # (K,)
            act = np.asarray(search.sampled_actions[i],
                             dtype=np.int64).reshape(-1, N)             # (K, N)
            k = min(len(vis), qv.shape[0], act.shape[0])
            vis, qv, act = vis[:k], qv[:k], act[:k]
            live = vis > 0
            if live.sum() >= 2:
                roots.append(dict(
                    actions=act[live], visits=vis[live] / max(num_sims, 1),
                    adv_raw=qv[live] - vvec[i][None, :], n_children=k))

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
    return roots


def _targets(root, adv_std, A, tau_q, tau_agent):
    """Both action-axis targets for one root. Returns (W, T, qbar_span, adv_span,
    multiplicity, mass_on_max_multiplicity, n_distinct) per agent."""
    import numpy as np

    act, vis = root["actions"], root["visits"]
    adv = root["adv_raw"] / np.maximum(adv_std, 1e-9)[None, :]          # (K, N)
    K, N = adv.shape
    W = np.zeros((N, A))
    Tn = np.zeros((N, A))
    Ts = np.zeros((N, A))
    mult = np.zeros((N, A))
    for i in range(N):
        for c in range(K):
            a = int(act[c, i])
            W[i, a] += np.exp(adv[c, i] / tau_q)
            Tn[i, a] += vis[c] * adv[c, i]
            Ts[i, a] += vis[c]
            mult[i, a] += 1
    present = Ts > 0
    qbar = np.where(present, Tn / np.maximum(Ts, 1e-12), 0.0)
    T = np.where(present, np.exp(qbar / tau_agent), 0.0)
    # normalize both over the present set
    W = np.where(present, W, 0.0)
    W = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
    T = T / np.maximum(T.sum(axis=1, keepdims=True), 1e-12)

    out = []
    for i in range(N):
        p = present[i]
        if p.sum() < 2:
            continue
        tv = 0.5 * np.abs(W[i] - T[i]).sum()
        amax = int(np.argmax(mult[i]))
        out.append(dict(
            tv=tv,
            qbar_span=float(qbar[i][p].max() - qbar[i][p].min()),
            adv_span=float(adv[:, i].max() - adv[:, i].min()),
            max_mult=float(mult[i, amax]),
            mass_w=float(W[i, amax]),
            mass_t=float(T[i, amax]),
            n_distinct=int(p.sum()),
        ))
    return out


def main() -> int:
    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--tau", type=float, default=1.0,
                    help="temperature for BOTH targets in the TV comparison "
                         "(apples to apples); the span-matched tau for the "
                         "agent target is reported separately")
    ap.add_argument("--out", default=None, help="optional JSON dump")
    args = ap.parse_args()

    from train import _set_gpus
    _set_gpus(args.gpus)

    import json
    import pathlib
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
    A = int(cfg.env.A)
    num_sims = int(runner._game_config.num_simulations)
    n_regimes = get_regime_family(cfg.env).size

    print(f"checkpoint: {run_dir}")
    print(f"arm={arm}  num_simulations={num_sims}  "
          f"sampled_action_times={runner._game_config.sampled_action_times}  "
          f"root_cover_mode={getattr(runner._game_config, 'root_cover_mode', '?')}")
    print()

    per_g = {}
    for g in range(n_regimes):
        roots = collect(runner, env_fn, cfg, g, args.episodes, dev, np_random)
        if roots:
            per_g[g] = roots

    # ONE pooled adv_std over every root and child, per agent -- reanalyze_worker
    # normalizes over axes (0, 1), so a per-regime std would be the wrong unit.
    pooled = np.concatenate([r["adv_raw"] for rs in per_g.values() for r in rs],
                            axis=0)
    adv_std = pooled.std(axis=0)
    print(f"pooled adv_std (reanalyze divides by this): "
          f"{np.round(adv_std, 4).tolist()}")
    print()

    # pass A: span ratio at a common tau, to derive the span-matched tau
    stats0 = [s for rs in per_g.values() for r in rs
              for s in _targets(r, adv_std, A, args.tau, args.tau)]
    span_ratio = (np.mean([s["qbar_span"] for s in stats0])
                  / max(np.mean([s["adv_span"] for s in stats0]), 1e-9))
    tau_agent = args.tau * span_ratio

    print(f"span(qbar)/span(adv) = {span_ratio:.3f}   "
          f"=> span-matched tau_agent = {tau_agent:.3f}  (vs tau_q={args.tau})")
    print("  averaging within a group shrinks the advantage span; running the")
    print("  wave at tau=1.0 would confound marginalization with flatness.")
    print()

    hdr = (" g   roots   children  vis/child  distinct  max mult   "
           "mass_q  mass_ag   TV(tau)  TV(matched)")
    print(hdr)
    print("-" * len(hdr))
    rows = {}
    for g, rs in per_g.items():
        s_c = [s for r in rs for s in _targets(r, adv_std, A, args.tau, args.tau)]
        s_m = [s for r in rs for s in _targets(r, adv_std, A, args.tau, tau_agent)]
        nc = np.mean([r["n_children"] for r in rs])
        rows[g] = dict(
            roots=len(rs), children=float(nc),
            vis_per_child=float(num_sims / max(nc, 1e-9)),
            distinct=float(np.mean([s["n_distinct"] for s in s_c])),
            max_mult=float(np.mean([s["max_mult"] for s in s_c])),
            mass_q=float(np.mean([s["mass_w"] for s in s_c])),
            mass_agent=float(np.mean([s["mass_t"] for s in s_c])),
            tv=float(np.mean([s["tv"] for s in s_c])),
            tv_matched=float(np.mean([s["tv"] for s in s_m])),
        )
        r = rows[g]
        print(f"{g:2d}  {r['roots']:6d}  {r['children']:8.1f}  "
              f"{r['vis_per_child']:9.2f}  {r['distinct']:8.2f}  "
              f"{r['max_mult']:8.2f}  {r['mass_q']:7.3f}  {r['mass_agent']:7.3f}  "
              f"{r['tv']:8.4f}  {r['tv_matched']:10.4f}")

    tv_all = float(np.mean([rows[g]["tv"] for g in rows]))
    tv_m_all = float(np.mean([rows[g]["tv_matched"] for g in rows]))
    print("-" * len(hdr))
    print(f"pooled TV(tau={args.tau}) = {tv_all:.4f}    "
          f"TV(span-matched) = {tv_m_all:.4f}")
    print()
    if tv_all < 0.02:
        print("GATE: FAIL -- TV ~ 0. The action-axis target is a no-op against")
        print("  q_softmax on this checkpoint (collisions too rare to matter).")
        print("  Do NOT spend a wave on it; report the null and stop.")
    else:
        print(f"GATE: PASS -- the two targets differ materially (TV {tv_all:.3f}).")
        print(f"  Run the wave at --policy_target_temperature {tau_agent:.2f}, "
              "not 1.0.")

    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(dict(
            schema="agent-target-probe-v1", run_dir=str(run_dir), arm=arm,
            num_simulations=num_sims, episodes=args.episodes, tau_q=args.tau,
            span_ratio=float(span_ratio), tau_agent=float(tau_agent),
            adv_std=adv_std.tolist(), per_regime=rows,
            tv_pooled=tv_all, tv_pooled_matched=tv_m_all,
        ), indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
