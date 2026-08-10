"""Assemble the v7 headline readout: per-role returns, global reward, NashConv.

Why this script exists
----------------------
`return_mean` is the summed SUBJECTIVE reward. Wherever the two agents' rewards
oppose it cancels, so two completely different policies can report the same
number -- measured on the stage-1 happo run, g1 `asym_exploit` scored 69.89 from
[66.74, 3.15] while g2 `asym_exploited` scored 69.08 from [22.61, 46.47]. Nearly
identical sums, opposite distributions. It cannot express the individual
optimality the method claims, so v7 reports a per-role table plus NashConv
instead, and this script builds both from the artefacts a run already writes.

The two regime sets, which are NOT the same thing
-------------------------------------------------
* ``GLOBAL_REWARD = (0, 3, 4)`` -- a SCORING choice: the regimes whose summed
  return is informative, because both agents' harvests move it. Applies inside
  runs that train and test on all of g0..g4. Under g2cm, g3 `asym_exploit_mild`
  qualifies -- its reward is (u_0 + u_1) / 2 (see relation.py), so it responds
  to both agents.
* ``seen / held = (0,1,2) / (3,4)`` -- the train/holdout PARTITION, which
  belongs to the separate generalization retraining and is deliberately not
  applied here.

These coincided under the old g2 family, which is exactly how they came to be
conflated. Keep them apart.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

# Scoring choice, not a train/test split. See module docstring.
GLOBAL_REWARD = (0, 3, 4)


def load_reports(root: str, env: str, seed: int) -> dict:
    out = {}
    for p in sorted(glob.glob(f"{root}/*/{env}/seed{seed}/eval_report.json")):
        algo = os.path.relpath(p, root).split(os.sep)[0]
        with open(p) as fh:
            out[algo] = (json.load(fh), os.path.dirname(p))
    return out


def load_nashconv(rundir: str):
    """Yield (label, {regime: nashconv}, br_env_steps) per report in `rundir`.

    ``--frozen-mode both`` writes <out> (the distilled prior) and
    <out>.planner.json (the MCTS policy that actually deploys). Those are very
    different quantities -- only the prior is comparable to the model-free
    baselines -- so they are emitted as separate rows and never averaged.
    """
    for p in sorted(glob.glob(f"{rundir}/game_metrics*.json")):
        with open(p) as fh:
            r = json.load(fh)
        nc = {int(k): v for k, v in (r.get("nashconv") or {}).items()}
        if not nc:
            continue
        label = "planner" if p.endswith(".planner.json") else "prior"
        yield label, nc, r.get("br_env_steps")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="results_v7_coopmix")
    ap.add_argument("--env", default="relation_coopmix")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    reports = load_reports(args.root, args.env, args.seed)
    if not reports:
        print(f"no eval_report.json under {args.root}/*/{args.env}/seed{args.seed}/")
        return 1

    names = next(iter(reports.values()))[0]["regime_names"]
    n_reg = len(names)

    print("## Per-role return (agent 0 / agent 1)\n")
    print("| algo | " + " | ".join(f"g{g} {names[g]}" for g in range(n_reg)) + " |")
    print("|" + "---|" * (n_reg + 1))
    for algo, (r, _) in reports.items():
        pa = r["return_per_regime_per_agent"]
        cells = [" / ".join(f"{x:.1f}" for x in pa[str(g)]) for g in range(n_reg)]
        print(f"| {algo} | " + " | ".join(cells) + " |")

    print("\n## Global (summed) reward -- scored on g0/g3/g4 only\n")
    print("| algo | " + " | ".join(f"g{g}" for g in GLOBAL_REWARD) + " | mean |")
    print("|" + "---|" * (len(GLOBAL_REWARD) + 2))
    rows = []
    for algo, (r, _) in reports.items():
        vals = [r["return_per_regime"][str(g)] for g in GLOBAL_REWARD]
        rows.append((sum(vals) / len(vals), algo, vals))
    for m, algo, vals in sorted(rows, reverse=True):
        print(f"| {algo} | " + " | ".join(f"{v:.2f}" for v in vals) + f" | **{m:.2f}** |")

    print("\n## Role-swap consistency (g1 vs g2)\n")
    print(
        "g2 `asym_exploited` is g1 `asym_exploit` with the agent indices swapped\n"
        "(relation.py: W=((1,-l),(+l,1)) vs ((1,+l),(-l,1)); the docstring calls it\n"
        "'the mirror'). So a policy that treats the two SLOTS equivalently must give\n"
        "r[g1][0] ~ r[g2][1] and r[g1][1] ~ r[g2][0]. Deviation means the policy has\n"
        "learned slot-specific behaviour rather than role-specific behaviour -- which\n"
        "a summed metric cannot see at all. This is a DIAGNOSTIC, not a ranking: a\n"
        "uniformly weak policy is trivially symmetric.\n"
    )
    print("| algo | \\|r1[0]-r2[1]\\| | \\|r1[1]-r2[0]\\| | mean | % of scale |")
    print("|---|---|---|---|---|")
    for algo, (r, _) in reports.items():
        pa = r["return_per_regime_per_agent"]
        if "1" not in pa or "2" not in pa:
            continue
        r1, r2 = pa["1"], pa["2"]
        d0, d1 = abs(r1[0] - r2[1]), abs(r1[1] - r2[0])
        mean_d = (d0 + d1) / 2
        scale = (sum(r1) + sum(r2)) / 4  # mean per-agent return over the two regimes
        pct = 100.0 * mean_d / scale if scale else float("nan")
        print(f"| {algo} | {d0:.2f} | {d1:.2f} | {mean_d:.2f} | {pct:.1f}% |")

    print("\n## Schema / invariant check\n")
    bad = 0
    for algo, (r, _) in reports.items():
        pa, pr = r["return_per_regime_per_agent"], r["return_per_regime"]
        ok = all(abs(sum(pa[str(g)]) - pr[str(g)]) < 1e-6 for g in range(n_reg))
        bad += not ok
        print(f"  {algo:55s} {'OK' if ok else 'MISMATCH'}  schema={r['schema_version']}")

    print("\n## NashConv (lower is better)\n")
    print("| algo | frozen | " + " | ".join(f"g{g}" for g in range(n_reg)) + " | sum | br |")
    print("|" + "---|" * (n_reg + 4))
    found = False
    for algo, (_, d) in reports.items():
        for label, nc, br in load_nashconv(d):
            found = True
            cells = " | ".join(f"{nc.get(g, float('nan')):.2f}" for g in range(n_reg))
            print(f"| {algo} | {label} | {cells} | **{sum(nc.values()):.2f}** | {br} |")
    if not found:
        print("  (no game_metrics reports in the run dirs yet)")
    print(
        "\nNashConv from an approximate best response is a LOWER BOUND on true\n"
        "exploitability -- comparable across algorithms only at equal br_env_steps."
    )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
