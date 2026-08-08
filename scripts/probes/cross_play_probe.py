#!/usr/bin/env python
"""cross_play_probe.py — does self-play here CYCLE, or does it improve?

NFSP-style policy averaging is a large change to a MuZero-style algorithm, and
this repo has none of the machinery (no opponent pool, no league, no averaged
policy — an exhaustive grep finds zero hits). It is worth building only if
self-play actually cycles, which has never been measured here. This measures it.

Method: play checkpoint *i* as agent 0 against checkpoint *j* as agent 1, over a
grid of checkpoints from one run, per regime. If later checkpoints beat earlier
ones consistently, training is **transitive** — monotone improvement, and
averaging buys nothing. Rock-paper-scissors among checkpoints is
**intransitive**, which is the failure NFSP exists to fix.

Scored on **per-agent** return, never the team sum. The summed metric cancels
structurally on this reward — in g1 the harvests cancel exactly, leaving only
``-eps*(moves)`` — so a cross-play matrix built on it would be all zeros in
precisely the zero-sum regime where cycling is most likely. See
``results/analysis/regime_knowledge_ceiling.md``.

Checkpoints come from the fork's periodic saves,
``<run>/mazero_mixed_fork/model/model_{step}.p`` (``--save_interval``, default
10000 gradient steps), or from a run's final ``ckpt.pt``.

    # matrix over one run's checkpoint series, zero-sum regime only
    python scripts/probes/cross_play_probe.py \\
        --run-dir results/v5_final/mazero_mixed_ref_bc_anneal_scaled_hardval_decoupled/relation/seed0 \\
        --regimes 1 --episodes 8

    # every regime, fewer checkpoints
    python scripts/probes/cross_play_probe.py --run-dir <run> --stride 2

Writes <run>/cross_play.json.

NOTE on what cross-play means for a CENTRALIZED controller: MAZero decides both
agents' actions from one joint search, so taking agent 0's component from A's
joint plan and agent 1's from B's means neither is executing its intended joint
action. That is inherent to cross-playing a centralized policy, and it is the
same substitution ``train_best_response`` already makes
(``utils/eval/game_metrics.py``: ``joint[agent_id] = a_i``).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import pathlib
import re
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hyper_mve.utils.configs import V4Config                                    # noqa: E402
from hyper_mve.envs.adapters.pettingzoo_wrapper import (                        # noqa: E402
    RelationCommonsPettingZooEnv,
)

SEED_BASE = 10_000


def find_checkpoints(run_dir: pathlib.Path, stride: int) -> list[tuple[int, pathlib.Path]]:
    """``(step, path)`` for the run's periodic saves, ascending, else ckpt.pt."""
    model_dir = run_dir / "mazero_mixed_fork" / "model"
    out: list[tuple[int, pathlib.Path]] = []
    if model_dir.is_dir():
        for p in model_dir.glob("model_*.p"):
            m = re.fullmatch(r"model_(\d+)\.p", p.name)
            if m:
                out.append((int(m.group(1)), p))
    out.sort()
    if stride > 1:
        out = out[::stride]
    if not out and (run_dir / "ckpt.pt").exists():
        out = [(-1, run_dir / "ckpt.pt")]
    return out


def load_act_fn(cfg, ckpt_path: pathlib.Path, ablation: str, mode: str):
    """An act_fn over the weights in ``ckpt_path``.

    Handles both checkpoint layouts: the runner's ``ckpt.pt``
    (``{"model_state_dict": ...}``) and the fork's periodic ``model_*.p``
    (a bare state dict).
    """
    import torch
    from hyper_mve.algo.runner import MAZeroMixedRunner

    runner = MAZeroMixedRunner(cfg)
    runner._ablation = ablation
    model = runner._lazy_model()
    blob = torch.load(str(ckpt_path), map_location=runner._device_of(model),
                      weights_only=False)
    state = blob["model_state_dict"] if isinstance(blob, dict) and \
        "model_state_dict" in blob else blob
    model.load_state_dict(state)
    model.eval()
    runner._weights_source = str(ckpt_path)
    return runner.make_act_fn(mode)


def play(cfg, act0, act1, g: int, episodes: int) -> np.ndarray:
    """Mean per-agent return with agent 0 on ``act0`` and agent 1 on ``act1``.

    Each policy is asked for a full joint action; agent k executes component k
    of policy k's plan.
    """
    env = RelationCommonsPettingZooEnv(cfg.env, oracle_mode=False,
                                       eval_info_mode=False)
    agents = list(env.possible_agents)
    acc = np.zeros((episodes, len(agents)))
    for ep in range(episodes):
        obs_dict, _ = env.reset(seed=SEED_BASE + 97 * int(g) + ep,
                                options={"g": int(g)})
        t, done = 0, False
        while not done:
            obs = np.stack([obs_dict[a] for a in agents]).astype(np.float32)
            joint = [int(act0(obs, t)[0]), int(act1(obs, t)[1])]
            acts = {a: joint[k] for k, a in enumerate(agents)}
            obs_dict, rew, term, trunc, _ = env.step(acts)
            for k, a in enumerate(agents):
                acc[ep, k] += float(rew[a])
            t += 1
            done = bool(any(term.values()) or any(trunc.values()))
    env.close()
    return acc.mean(axis=0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--preset", default=None,
                    help="default: read from the run's meta.json, else rel_duo")
    ap.add_argument("--ablation", default=None,
                    help="default: read from the run's meta.json, else none")
    ap.add_argument("--regimes", type=int, nargs="*", default=None)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--stride", type=int, default=1,
                    help="take every Nth checkpoint (the matrix is quadratic)")
    ap.add_argument("--mode", choices=("prior", "planner"), default="planner",
                    help="planner is what deploys; prior is much cheaper")
    args = ap.parse_args(argv)

    run_dir = pathlib.Path(args.run_dir)
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    preset = args.preset or {"relation": "rel_duo",
                             "relation_recip": "rel_recip"}.get(
                                 meta.get("env", ""), "rel_duo")
    ablation = args.ablation or meta.get("ablation") or "none"
    cfg = V4Config.from_preset(preset)

    ckpts = find_checkpoints(run_dir, args.stride)
    if len(ckpts) < 2:
        raise SystemExit(
            f"{run_dir}: found {len(ckpts)} checkpoint(s); a cross-play matrix "
            "needs at least 2. Periodic saves live under "
            "mazero_mixed_fork/model/ and require --save_interval during training."
        )
    from hyper_mve.utils.schemas import get_regime_family
    regimes = args.regimes if args.regimes is not None else list(
        range(get_regime_family(cfg.env).size))

    print(f">>> {run_dir.name}  preset={preset} ablation={ablation} "
          f"mode={args.mode}")
    print(f"    {len(ckpts)} checkpoints: {[s for s, _ in ckpts]}")

    act_fns = {}
    for step, path in ckpts:
        act_fns[step] = load_act_fn(cfg, path, ablation, args.mode)
        print(f"    loaded {path.name}", flush=True)

    results = {}
    for g in regimes:
        steps = [s for s, _ in ckpts]
        # M[i][j] = agent 0's own return when checkpoint i plays checkpoint j.
        M = np.zeros((len(steps), len(steps)))
        for i, j in itertools.product(range(len(steps)), repeat=2):
            M[i, j] = play(cfg, act_fns[steps[i]], act_fns[steps[j]],
                           g, args.episodes)[0]
        results[str(g)] = {"steps": steps, "matrix": M.tolist()}

        print(f"\n  === g{g}: agent 0's own return, row = its checkpoint ===")
        print("      " + " ".join(f"{s:>8d}" for s in steps))
        for i, s in enumerate(steps):
            print(f"  {s:>5d} " + " ".join(f"{M[i, j]:8.2f}" for j in range(len(steps))))

        # Transitivity: does a later checkpoint beat an earlier one against a
        # COMMON opponent? Intransitivity is the thing NFSP would fix.
        #
        # Ties are counted separately and reported first. A matrix whose rows
        # are identical means the policy never changed — on the distilled prior
        # that is the known collapse, every checkpoint playing constant-HARVEST.
        # Scoring that as "later rarely beats earlier" would read as cycling and
        # motivate NFSP for a policy that is simply frozen, so the degenerate
        # case has to be named rather than folded into the win rate.
        tol = 1e-9
        wins = losses = ties = 0
        for i, j in itertools.combinations(range(len(steps)), 2):
            for k in range(len(steps)):
                d = M[j, k] - M[i, k]                 # j is the later checkpoint
                if abs(d) <= tol:
                    ties += 1
                elif d > 0:
                    wins += 1
                else:
                    losses += 1
        decisive = wins + losses
        frac = wins / decisive if decisive else float("nan")
        res = results[str(g)]
        res.update(later_beats_earlier_frac=frac, wins=wins,
                   losses=losses, ties=ties,
                   row_spread=float(M.max(axis=0).max() - M.min(axis=0).min()))

        if decisive == 0:
            verdict = ("DEGENERATE — every checkpoint plays identically; "
                       "nothing to cycle. Not evidence for or against NFSP.")
        elif frac > 0.8:
            verdict = "transitive (monotone improvement)"
        elif frac < 0.65:
            verdict = "INTRANSITIVE — later does not reliably beat earlier"
        else:
            verdict = "mixed"
        print(f"    later vs earlier on a common opponent: "
              f"{wins}W/{losses}L/{ties}T"
              + (f"  frac={frac:.2f}" if decisive else "")
              + f"\n    -> {verdict}")

    out = run_dir / "cross_play.json"
    out.write_text(json.dumps(
        {"schema_version": "crossplay-v1", "run_dir": str(run_dir),
         "preset": preset, "ablation": ablation, "mode": args.mode,
         "episodes": args.episodes, "results": results}, indent=2),
        encoding="utf-8")
    print(f"\nwrote {out}")
    print("read: a high fraction means self-play improves monotonically and "
          "NFSP-style averaging is unmotivated; a low one means cycling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
