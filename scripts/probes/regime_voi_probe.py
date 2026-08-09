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
observes. That partition is **derived from the family** by ``own_row_posterior``,
never hardcoded — on ``g2`` it comes out as ``+lam => {g0,g3}``, ``-lam => {g1,g2}``,
``0 => {g4}``. VoI > 0 means knowing the rest of the regime would change what agent
0 should do; VoI == 0 means the observed own row already determines the best
response and the belief channel has nothing to earn. A bucket holding a single
regime therefore contributes exactly 0, which is how a family edit that collapses
one shows up here rather than silently.

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
across every regime in the family. The grid is therefore rolled out ONCE and every
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

THRESHOLDS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
SEED_BASE = 10_000


def own_row_posterior(family, grid, agent=0):
    """Agent ``agent``'s candidate set for each value of its OWN row, under the
    uniform regime prior — derived from the family, never hardcoded.

    This is where VoI comes from and the only place it can come from: the agent
    sees ``w_ij`` and must infer the hidden ``w_ji``, so a bucket holding a single
    regime leaves nothing to infer and contributes **exactly zero**. Deriving the
    partition rather than writing it down means a family edit that collapses a
    bucket shows up here instead of silently halving the headline number.
    """
    buckets = {}
    for g in grid:
        v = float(family.regimes[int(g)].row(agent)[0])
        buckets.setdefault(v, []).append(int(g))
    vals = sorted(buckets)
    signs = [(v > 0) - (v < 0) for v in vals]
    by_sign = len(set(signs)) == len(signs)      # at most one bucket per sign
    out = {}
    for v in vals:
        if by_sign:
            key = "+lam" if v > 0 else ("-lam" if v < 0 else "0")
        else:
            key = f"w01={v:+.2f}"
        out[key] = tuple(buckets[v])
    return out


def zero_coupling_regime(family):
    """The regime with no off-diagonal weight, where ``R_i = u_i - eps*moved_i``
    so the physical harvests read back exactly. ``rollout_grid`` pins it."""
    for r in family.regimes:
        W = r.w_array()
        if np.allclose(W[~np.eye(len(W), dtype=bool)], 0.0):
            return int(r.id)
    raise SystemExit(
        f"family {family.name!r} has no zero-coupling regime; rollout_grid needs "
        "one to read back physical harvests")

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

    Runs under the family's zero-coupling regime (``W = 0``), where the reward
    reduces to ``R_i = u_i - eps*moved_i``, so the physical harvests are read back
    exactly and every candidate coupling can be scored offline from one rollout.
    """
    from hyper_mve.utils.schemas.relation import get_regime_family
    g_zero = zero_coupling_regime(get_regime_family(env_cfg))
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
                obs, _ = env.reset(seed=SEED_BASE + ep, options={"g": g_zero})
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


def voi(weights, U0, U1, M0, eps, posterior):
    """VoI for agent 0, averaged over own-row cases. ``weights[g]`` = ŵ_01.

    The aggregate is an unweighted mean over the own-row cases in ``posterior``,
    NOT a prior-weighted expectation — so a case whose bucket holds one regime
    contributes a hard 0 and drags the headline down by its full share. That is
    the intended behaviour: it is what makes a collapsed bucket visible.
    """
    per_case = {}
    for name, gs in posterior.items():
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


def coupling_weights(env_cfg, coupling, lam, grid):
    """``ŵ_01`` per regime under a candidate coupling rule."""
    family = get_regime_family(env_cfg)
    return {int(g): float(effective_coupling_matrix(
        family.regimes[int(g)].w_array(), coupling, lam)[0, 1]) for g in grid}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="rel_duo")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--K", type=int, default=None, help="override resource cells")
    ap.add_argument("--L", type=int, default=None, help="override grid size")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--regrowth-law", choices=("constant", "logistic"), default=None)
    ap.add_argument("--regimes", type=int, nargs="*", default=None, metavar="G",
                    help="regime ids to score (default: the whole preset family). "
                         "Scoring a subset changes the own-row partition, so it can "
                         "change VoI on its own — use it to ask what a restricted "
                         "family would measure, not to filter noise.")
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
    family = get_regime_family(env_cfg)
    grid = tuple(int(g) for g in (args.regimes if args.regimes is not None
                                  else range(family.size)))
    bad = [g for g in grid if not 0 <= g < family.size]
    if bad:
        raise SystemExit(f"--regimes {bad} out of range for family "
                         f"{family.name!r} (|G|={family.size})")
    posterior = own_row_posterior(family, grid)

    print(f"    family={family.name} |G|={family.size} scoring {len(grid)} regimes: "
          + ", ".join(f"g{g}={family.names()[g]}" for g in grid))
    print("    agent 0's own-row partition (a 1-regime bucket has VoI 0 by "
          "construction):")
    for k, gs in posterior.items():
        mark = "  <-- SINGLETON, contributes 0" if len(gs) < 2 else ""
        print(f"      w_01 {k:>6s} -> " + ", ".join(family.names()[g] for g in gs)
              + mark)
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
    cases = list(posterior)
    gstr = f"g{grid[0]}..g{grid[-1]}" if grid else "-"
    print(f"\n  {'design':<22} {'w_01 per regime (' + gstr + ')':<32} {'VoI':>7}"
          + "".join(f" {c:>7}" for c in cases))
    results = {}
    for label, coupling, lam in (configured,) + DESIGNS:
        w = coupling_weights(env_cfg, coupling, lam, grid)
        v, per = voi(w, U0, U1, M0, eps, posterior)
        results[label] = {"coupling": coupling, "lambda": lam, "weights": w,
                          "voi": v, "per_own_row": per}
        wstr = " ".join(f"{w[g]:+.2f}" for g in grid)
        print(f"  {label:<22} {wstr:<32} {v:7.2f}"
              + "".join(f" {per[c]:7.2f}" for c in cases))

    print("\n  gate: VoI must clear the method's seed sd to be detectable at 3 "
          "seeds.\n  NOTE this is a lower bound — a 1-D scripted threshold "
          "family cannot\n  exploit everything a learned policy can.")

    out_dir = pathlib.Path(REPO_ROOT) / "results" / "analysis" / "belief_ceiling"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"regime_voi_{args.preset}.json"
    out_path.write_text(json.dumps({
        "preset": args.preset, "overrides": over, "episodes": args.episodes,
        "thresholds": THRESHOLDS, "regimes": list(grid),
        "family": family.name,
        "regime_names": [family.names()[g] for g in grid],
        "own_row_partition": {k: list(v) for k, v in posterior.items()},
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
