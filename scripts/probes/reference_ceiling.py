#!/usr/bin/env python
"""reference_ceiling.py — per-regime returns for the scripted reference policies.

The published table in ``hyper_mve/envs/relation_commons/reference_policies.py``
reports only g0/g4. The belief-ceiling investigation (2026-07-21) needs the
ASYMMETRIC regimes g2/g3, because ``scripted_greedy`` is regime-BLIND (it reads
only own-observation, never the regime id), so its g2/g3 return is the "competent
play WITHOUT knowing the regime" ceiling. Read against the trained per-regime
returns it adjudicates the user's game-theoretic hypothesis:

  * scripted_greedy WINS g3  -> g3 is winnable regime-blind; the trained policy's
    g3 collapse is a competence failure, not a game-theoretic bound.
  * scripted_greedy TANKS g3 -> g3 is structurally adversarial; the oracle's ~11
    is near-ceiling and the collapsed baseline's ~22 (defensive HARVEST) is the
    anomaly, so g3 should be reported as bounded, not chased.

No checkpoint, no GPU. Writes results/analysis/belief_ceiling/reference_ceiling.json.

    python scripts/probes/reference_ceiling.py --episodes 16
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # pure-numpy refs; never grab a GPU

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from train import build_cfg, make_env_fn  # noqa: E402


# runs whose eval_report.json we print alongside the reference rows, for scale.
_TRAINED_ROWS = {
    "PRE-FIX main (22.1)": "results/mazero_mixed/relation/seed0/eval_report.json",
    "A+B fix (13.3)": "results_fixval/mazero_mixed/relation/seed0/eval_report.json",
    "A+B+oracle (18.5)": "results_fixval/mazero_mixed_mcts_fix_oracle/relation/seed0/eval_report.json",
}


def _load_trained(path: str) -> dict | None:
    p = pathlib.Path(REPO_ROOT) / path
    if not p.exists():
        return None
    r = json.loads(p.read_text(encoding="utf-8"))
    return r.get("return_per_regime")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="relation")
    ap.add_argument("--preset", default=None,
                    help="V4Config preset to score (default: whatever --env maps to)")
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--seed-base", type=int, default=10_000,
                    help="matches runner._rollout_* episode seeding (10000+97*g+ep)")
    args = ap.parse_args(argv)

    from hyper_mve.envs.relation_commons.reference_policies import (
        evaluate_reference_policy, make_relational_greedy_policy,
        reference_policy_suite,
    )
    from hyper_mve.utils.schemas.relation import get_regime_family

    if args.preset:
        from hyper_mve.utils.configs import V4Config
        cfg = V4Config.from_preset(args.preset)
    else:
        cfg = build_cfg(args.env, "none")
    env_fn = make_env_fn(cfg, args.env)
    N, K = int(cfg.env.N), int(cfg.env.K)
    family = get_regime_family(cfg.env)
    G = family.size
    grid = tuple(range(G))

    suite = reference_policy_suite(N, K, seed=0)
    # The three information levels the redesign is measured against. They share
    # a controller and differ only in what they know about the hidden half of
    # the relationship. The coupling MUST come from the env: which entry of W
    # sets an agent's regard for its neighbour is exactly what v6 changes, and
    # a controller given the wrong rule optimises the wrong objective.
    # Under logistic regrowth the regime's value lives largely in the
    # conservation decision; a purely positional controller cannot express it
    # and reports a spurious null. Under constant regrowth restraint is
    # strictly harmful, so the threshold must stay 0 there.
    _conserve = 0.3 if cfg.env.regrowth_law == "logistic" else 0.0
    _cpl = dict(coupling=cfg.env.reward_coupling,
                reciprocity_lambda=cfg.env.reciprocity_lambda,
                conserve_threshold=_conserve)
    suite["relational_self_info"] = make_relational_greedy_policy(
        N, K, level="self_info", **_cpl)
    suite["relational_oracle"] = make_relational_greedy_policy(
        N, K, level="oracle", family=family, **_cpl)
    names = list(family.names())

    rows: dict[str, dict] = {}
    for name, policy in suite.items():
        res = evaluate_reference_policy(
            env_fn, policy, grid, args.episodes, seed_base=args.seed_base)
        rows[name] = {
            "return_mean": res["return_mean"],
            "return_per_regime": res["return_per_regime"],
            "action_fractions": res["action_fractions"],
        }

    # ---- print ----
    hdr = f"{'policy':28s} {'mean':>7s} " + " ".join(f"g{g}:{names[g][:6]:<6s}" for g in grid)
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, r in rows.items():
        pr = r["return_per_regime"]
        line = f"{name:28s} {r['return_mean']:7.2f} " + " ".join(
            f"{pr[str(g)] if str(g) in pr else pr.get(g, 0.0):9.2f}" for g in grid)
        print(line)

    print("\n--- trained policies (for scale) ---")
    for label, path in _TRAINED_ROWS.items():
        pr = _load_trained(path)
        if pr is None:
            print(f"{label:28s}  (no eval_report.json)")
            continue
        mean = sum(pr.values()) / len(pr)
        line = f"{label:28s} {mean:7.2f} " + " ".join(
            f"{pr.get(str(g), 0.0):9.2f}" for g in grid)
        print(line)

    # ---- the value of knowing the hidden half of the relationship ----
    # This MUST be unilateral: VoI is what ONE agent gains by deviating while
    # the other holds still. Handing both agents the oracle measures something
    # else — in a social dilemma it can lower the team return while raising
    # each agent's own objective, which is how an earlier version of this probe
    # reported -25.75 on exactly the regimes where VoI is largest.
    baseline = make_relational_greedy_policy(N, K, level="self_info", **_cpl)
    deviator = make_relational_greedy_policy(
        N, K, level="oracle", family=family, **_cpl)
    partner = make_relational_greedy_policy(N, K, level="self_info", **_cpl)

    uni_si = evaluate_reference_policy(
        env_fn, baseline, grid, args.episodes, seed_base=args.seed_base,
        policies=[baseline, partner])
    uni_or = evaluate_reference_policy(
        env_fn, deviator, grid, args.episodes, seed_base=args.seed_base,
        policies=[deviator, partner])

    # Agent 0's OWN return, not the team sum.
    gaps = {g: (uni_or["return_per_agent_per_regime"][g][0]
                - uni_si["return_per_agent_per_regime"][g][0]) for g in grid}
    gap_mean = sum(gaps.values()) / len(gaps)
    # The mirror prior is *correct* in every symmetric regime, so the whole
    # effect must live in the asymmetric ones; a nonzero gap elsewhere is
    # controller noise. Derived from the family, not written down: the ids are
    # (2, 3) under g2 but (1, 2, 3) under g2cm.
    asym = [g for g in grid if g in set(family.asymmetric_ids())]
    gap_asym = sum(gaps[g] for g in asym) / max(len(asym), 1)

    effect = sum(abs(gaps[g]) for g in asym) / max(len(asym), 1)

    print("\n--- effect of knowing the hidden half of W ---")
    print("    (agent 0 deviates to oracle; agent 1 held at self_info;")
    print("     scored on agent 0's OWN return, not the team sum)")
    print("  " + " ".join(f"g{g}:{names[g][:6]:<6s}" for g in grid))
    print("  " + " ".join(f"{gaps[g]:9.2f}" for g in grid))
    asym_label = "/".join(f"g{g}" for g in asym) or "(none)"
    print(f"  signed mean, all regimes : {gap_mean:+.2f}")
    print(f"  signed mean, {asym_label:<12s}: {gap_asym:+.2f}")
    print(f"  EFFECT SIZE |gap| {asym_label:<7s}: {effect:.2f}")
    print("  read: this is the difference between two FIXED HEURISTICS, so it")
    print("  can be negative where the oracle heuristic mis-responds — it is")
    print("  not a VoI and is not bounded below by 0. Use the effect size to")
    print("  see whether the regime moves the payoff at all, and take the")
    print("  non-negative optimised quantity from regime_voi_probe.py.")

    ceiling = {
        "per_regime": {str(g): gaps[g] for g in grid},
        "mean": gap_mean,
        "family": family.name,
        "regime_names": list(family.names()),
        "asymmetric_ids": list(asym),
        "asymmetric_mean": gap_asym,
        "asymmetric_effect_size": effect,
        "protocol": "unilateral deviation, agent 0's own return",
        "note": ("agent 0 oracle vs self_info with agent 1 held at self_info; the "
                 "two controllers differ only in the hidden half of W. This is a "
                 "difference of fixed heuristics, NOT a value of information: it "
                 "can be negative when the oracle heuristic responds badly to the "
                 "truth. regime_voi_probe.py optimises over a strategy family and "
                 "reports the non-negative VoI."),
    }

    # the falsifiable read on g3
    sg = rows.get("scripted_greedy_distinct", rows.get("scripted_greedy"))
    blind_ceiling = {}
    for g in asym:
        blind_ceiling[int(g)] = float(
            sg["return_per_regime"].get(str(g), sg["return_per_regime"].get(g, 0.0)))
    print("\nregime-blind scripted ceiling: "
          + "  ".join(f"g{g}({names[g][:9]})={v:.1f}"
                      for g, v in blind_ceiling.items()))
    print("  read: an asymmetric regime whose regime-BLIND ceiling far exceeds "
          "what the\n  oracle controller reaches is winnable without knowing the "
          "regime — a competence\n  gap, not a game-theoretic bound. One that "
          "sits at the oracle's level is bounded.")

    out_dir = pathlib.Path(REPO_ROOT) / "results" / "analysis" / "belief_ceiling"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.preset}" if args.preset else ""
    out_path = out_dir / f"reference_ceiling{suffix}.json"
    out_path.write_text(json.dumps(
        {"env": args.env, "episodes": args.episodes, "N": N, "K": K,
         "regime_names": names, "reference": rows,
         "regime_knowledge_ceiling": ceiling}, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
