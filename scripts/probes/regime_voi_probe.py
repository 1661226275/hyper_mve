#!/usr/bin/env python
"""regime_voi_probe.py — is the hidden relationship worth inferring at all?

The 3-seed ablations found the belief posterior worth ~nothing (`UB - A1 =
+0.67 +/- 2.88`) and the standing explanation was architectural (QMDP leaf
mixture + a belief frozen through the search). This probe tests the prior
question — whether there is anything *to* infer — with scripted controllers,
no checkpoint and no GPU.

It measures the value of information for agent 0's strategy choice:

    VoI = E_g[ max_t R_0(t | g) ]  -  max_t E_g[ R_0(t | g) ]

with ``g`` drawn from agent 0's posterior GIVEN ITS OWN ROW, the only thing it
observes (``w_01=+lam => g in {g0,g3}``; ``-lam => {g1,g2}``; ``0 => {g4}``).
VoI > 0 means knowing the rest of the regime would change what agent 0 should
do; VoI == 0 means the observed own row already determines the best response
and the belief channel has nothing to earn.

Three things make this measure what it claims:

* **Per-agent return, not the team sum.** Summing over agents cancels
  structurally (g1 == 0 exactly, g2/g3 == one agent's harvest), which would
  hide a regime-dependent best response.
* **Asymmetric strategies.** The two agents must be free to play differently,
  or "best response to the opponent" is not expressible.
* **Genuine abstention.** The strategy family is the harvest threshold, and an
  agent that cannot decline to harvest cannot express restraint. An earlier
  version fell back to `viable = ones` and `_step_toward(0,0) -> HARVEST`, so
  it always harvested something; that understated VoI everywhere and made
  K=1 look degenerate.

**Offline recompute.** ``W(g)`` enters only the reward, never the dynamics, and
the threshold policy below never reads the observation's row block — so for a
fixed ``(t0, t1)`` the trajectory, and hence ``(U_0, U_1, M_0)``, is IDENTICAL
across all five regimes. The grid is therefore rolled out ONCE and every
candidate reward coupling is evaluated in closed form from
``effective_coupling_matrix``. That is what makes sweeping the design surface
affordable. ``tests/envs/test_voi_probe.py`` pins the row-block independence;
if a probe policy ever reads the row, this shortcut is silently invalid.

    python scripts/probes/regime_voi_probe.py                       # rel_duo
    python scripts/probes/regime_voi_probe.py --preset rel_recip
    python scripts/probes/regime_voi_probe.py --preset rel_duo --K 2 --L 3 \
        --regrowth-law logistic --alpha 0.30

Writes results/analysis/belief_ceiling/regime_voi_<preset>.json.
Findings: results/analysis/regime_knowledge_ceiling.md
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import sys

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # pure-numpy refs; never grab a GPU

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hyper_mve.utils.configs import V4Config                                    # noqa: E402
from hyper_mve.envs.adapters.pettingzoo_wrapper import (                        # noqa: E402
    RelationCommonsPettingZooEnv,
)
from hyper_mve.envs.relation_commons.reference_policies import (                # noqa: E402
    HARVEST, NOOP, _nearest, _step_toward, neighbor_view, resource_view,
)
from hyper_mve.utils.schemas.relation import (                                  # noqa: E402
    effective_coupling_matrix, get_regime_family,
)

GRID = (0, 1, 2, 3, 4)
THRESHOLDS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
SEED_BASE = 10_000

# Agent 0's posterior over g given its own row, under the uniform regime prior.
POSTERIOR = {"+lam": (0, 3), "-lam": (1, 2), "0": (4,)}

# The comparison set reported alongside the preset's configured coupling.
DESIGNS: tuple[tuple[str, str, float], ...] = (
    ("own_row (v5)", "own_row", 0.0),
    ("levine lam=0.5", "levine", 0.5),
    ("levine lam=1.0", "levine", 1.0),
    ("levine lam=2.0", "levine", 2.0),
    ("levine lam=3.0", "levine", 3.0),
    ("reciprocal (v6)", "reciprocal", 0.0),
)


def threshold_action(obs_i, agent_idx, N, K, thresh):
    """Greedy walk-and-harvest that declines to harvest a cell below ``thresh``.

    Returns NOOP when nothing clears the threshold — genuine restraint, which
    is what a commons dilemma requires. MUST NOT read the observation's row
    block, or the offline recompute above becomes invalid.
    """
    cells = resource_view(obs_i, N, K)
    rel, stock = cells[:, :2], cells[:, 2]
    underfoot = (rel[:, 0] == 0) & (rel[:, 1] == 0)
    if underfoot.any() and stock[underfoot].max() > thresh:
        return HARVEST
    viable = stock > thresh
    if not viable.any():
        return NOOP
    d_self = np.abs(rel[:, 0]) + np.abs(rel[:, 1])
    if N == 2:
        nb = neighbor_view(obs_i, N, K)[0, :2]
        d_nb = np.abs(rel[:, 0] - nb[0]) + np.abs(rel[:, 1] - nb[1])
        wins = d_self <= d_nb if agent_idx == 0 else d_self < d_nb
        if (viable & wins).any():
            viable = viable & wins
    t = _nearest(viable, d_self, K)
    if d_self[t] == 0:
        return HARVEST if stock[t] > thresh else NOOP
    return _step_toward(float(rel[t, 0]), float(rel[t, 1]))


def rollout_grid(env_cfg, episodes):
    """``(U0, U1, M0)`` over the threshold grid. One pass covers every regime.

    Runs under regime g4 (``W = 0``), where the reward reduces to
    ``R_i = u_i - eps*moved_i``, so the physical harvests are read back exactly.
    """
    n = len(THRESHOLDS)
    eps = float(env_cfg.epsilon_move)
    U0, U1, M0 = (np.zeros((n, n)) for _ in range(3))
    for i, t0 in enumerate(THRESHOLDS):
        for j, t1 in enumerate(THRESHOLDS):
            env = RelationCommonsPettingZooEnv(
                env_cfg, oracle_mode=False, eval_info_mode=False)
            agents = list(env.possible_agents)
            u, moves = np.zeros(2), 0.0
            for ep in range(episodes):
                obs, _ = env.reset(seed=SEED_BASE + ep, options={"g": 4})
                done = False
                while not done:
                    acts = {
                        a: int(threshold_action(np.asarray(obs[a]), k,
                                                env_cfg.N, env_cfg.K, (t0, t1)[k]))
                        for k, a in enumerate(agents)
                    }
                    obs, rew, term, trunc, _ = env.step(acts)
                    for k, a in enumerate(agents):
                        moved = 1 <= acts[a] <= 4
                        u[k] += float(rew[a]) + (eps if moved else 0.0)
                        if k == 0 and moved:
                            moves += 1.0
                    done = bool(any(term.values()) or any(trunc.values()))
            env.close()
            U0[i, j], U1[i, j], M0[i, j] = u[0]/episodes, u[1]/episodes, moves/episodes
        print(f"      t0={t0:.1f} done", flush=True)
    return U0, U1, M0


def voi(weights, U0, U1, M0, eps):
    """VoI for agent 0, averaged over own-row cases. ``weights[g]`` = ŵ_01."""
    per_case = {}
    for name, gs in POSTERIOR.items():
        # R_0 = (U0 + w*U1)/(1+|w|) - eps*M0, linear in the harvests because the
        # coefficients are constant within an episode.
        R = {g: (U0 + weights[g] * U1) / (1.0 + abs(weights[g])) - eps * M0
             for g in gs}
        per_opp = []
        for j in range(len(THRESHOLDS)):
            informed = np.mean([R[g][:, j].max() for g in gs])
            uninformed = np.mean([R[g][:, j] for g in gs], axis=0).max()
            per_opp.append(float(informed - uninformed))
        per_case[name] = float(np.mean(per_opp))
    return float(np.mean(list(per_case.values()))), per_case


def coupling_weights(env_cfg, coupling, lam):
    """``ŵ_01`` per regime under a candidate coupling rule."""
    family = get_regime_family(env_cfg)
    return {g: float(effective_coupling_matrix(
        family.regimes[g].w_array(), coupling, lam)[0, 1]) for g in GRID}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="rel_duo")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--K", type=int, default=None, help="override resource cells")
    ap.add_argument("--L", type=int, default=None, help="override grid size")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--regrowth-law", choices=("constant", "logistic"), default=None)
    args = ap.parse_args(argv)

    env_cfg = V4Config.from_preset(args.preset).env
    over = {k: v for k, v in (("K", args.K), ("L", args.L), ("alpha", args.alpha),
                              ("regrowth_law", args.regrowth_law)) if v is not None}
    if over:
        env_cfg = dataclasses.replace(env_cfg, **over)

    print(f">>> preset={args.preset}  N={env_cfg.N} L={env_cfg.L} K={env_cfg.K} "
          f"alpha={env_cfg.alpha} law={env_cfg.regrowth_law} "
          f"coupling={env_cfg.reward_coupling}"
          + (f" lam={env_cfg.reciprocity_lambda}"
             if env_cfg.reward_coupling == "levine" else ""))
    if over:
        print(f"    overrides: {over}")
    print(f"    rolling out {len(THRESHOLDS)}x{len(THRESHOLDS)} thresholds "
          f"x {args.episodes} episodes (once; all regimes share it)", flush=True)

    U0, U1, M0 = rollout_grid(env_cfg, args.episodes)
    eps = float(env_cfg.epsilon_move)

    # Action coupling: agent 0's threshold must MOVE agent 1's harvest, or no
    # reward weighting can change any argmax. This localizes a null result to
    # geometry (this number ~0) vs reward (this number healthy, VoI still 0).
    c_t0, c_t1 = U1.mean(axis=1).std(), U1.mean(axis=0).std()
    ratio = c_t0 / max(c_t1, 1e-9)
    print(f"\n  U0 {U0.min():5.1f}-{U0.max():5.1f}   U1 {U1.min():5.1f}-{U1.max():5.1f}")
    print(f"  ACTION COUPLING  sd(U1|t0)={c_t0:6.2f}  sd(U1|t1)={c_t1:6.2f}"
          f"   ratio={ratio:.3f}")
    if ratio < 0.05:
        print("    -> agent 0's strategy does not move agent 1's harvest; the "
              "agents are decoupled and NO reward design can produce VoI here.")

    configured = (f"CONFIGURED ({env_cfg.reward_coupling})",
                  env_cfg.reward_coupling, float(env_cfg.reciprocity_lambda))
    print(f"\n  {'design':<22} {'w_01 per regime (g0..g4)':<32} {'VoI':>7}"
          f" {'+row':>7} {'-row':>7}")
    results = {}
    for label, coupling, lam in (configured,) + DESIGNS:
        w = coupling_weights(env_cfg, coupling, lam)
        v, per = voi(w, U0, U1, M0, eps)
        results[label] = {"coupling": coupling, "lambda": lam, "weights": w,
                          "voi": v, "per_own_row": per}
        wstr = " ".join(f"{w[g]:+.2f}" for g in GRID)
        print(f"  {label:<22} {wstr:<32} {v:7.2f} {per['+lam']:7.2f} {per['-lam']:7.2f}")

    print("\n  gate: VoI must clear the method's seed sd to be detectable at 3 "
          "seeds.\n  NOTE this is a lower bound — a 1-D scripted threshold "
          "family cannot\n  exploit everything a learned policy can.")

    out_dir = pathlib.Path(REPO_ROOT) / "results" / "analysis" / "belief_ceiling"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"regime_voi_{args.preset}.json"
    out_path.write_text(json.dumps({
        "preset": args.preset, "overrides": over, "episodes": args.episodes,
        "thresholds": THRESHOLDS, "regimes": GRID,
        "env": {"N": env_cfg.N, "L": env_cfg.L, "K": env_cfg.K,
                "alpha": env_cfg.alpha, "regrowth_law": env_cfg.regrowth_law,
                "reward_coupling": env_cfg.reward_coupling},
        "action_coupling": {"sd_U1_given_t0": c_t0, "sd_U1_given_t1": c_t1,
                            "ratio": ratio},
        "designs": results,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
