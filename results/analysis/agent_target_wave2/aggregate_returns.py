"""Aggregate the agent-target 2x2 on the robust last-20% statistic.

Range is n-dependent, so the star comparison is reported two ways: sd (n=4 on
the new arms) and the range restricted to seeds 0-1, which is the only
like-for-like comparison against wave 1's n=2 rows.
"""
import glob
import json
import os
import statistics as st

from tensorboard.backend.event_processing import event_accumulator

NEW = "results_v6_agentq"
OLD = "/home/data/zhengwenbo/hyper_mve/results_v6_2x2"
PREFIX = "mazero_mixed_ref_bc_anneal_scaled_hardval_decoupled"

CELLS = [
    ("none/visit      (baseline)", OLD, "", [0, 1]),
    ("star/visit      (cover)", OLD, "_cover", [0, 1]),
    ("none/q_softmax  (qtarget)", OLD, "_qtarget", [0, 1]),
    ("star/q_softmax  (mctsfix)", OLD, "_mctsfix", [0, 1]),
    ("none/agent_q    (agentq)", NEW, "_agentq", [0, 1, 2, 3]),
    ("star/agent_q    (agentq_cover)", NEW, "_agentq_cover", [0, 1, 2, 3]),
]

TAGS = ["eval/return_mean"] + [f"eval/return_regime_{g}" for g in range(5)]


def last20(root, suffix, seed):
    d = f"{root}/{PREFIX}{suffix}/relation_recip/seed{seed}"
    ev = sorted(glob.glob(f"{d}/tb/events*"))
    if not ev:
        return None
    ea = event_accumulator.EventAccumulator(ev[-1], size_guidance={"scalars": 0})
    ea.Reload()
    have = ea.Tags()["scalars"]
    out = {}
    for t in TAGS:
        if t not in have:
            continue
        s = ea.Scalars(t)
        if len(s) < 5:
            continue
        k = max(1, int(round(0.2 * len(s))))
        tail = [x.value for x in s[-k:]]
        out[t] = (st.fmean(tail), st.pstdev(tail), len(s), k)
    fin = f"{d}/eval_report.json"
    if os.path.exists(fin):
        out["_final"] = json.load(open(fin))
    return out


rows = {}
for name, root, suffix, seeds in CELLS:
    per_seed = {}
    for s in seeds:
        r = last20(root, suffix, s)
        if r and "eval/return_mean" in r:
            per_seed[s] = r
    rows[name] = per_seed

print("=" * 92)
print("LAST-20% eval/return_mean  (robust statistic; final single point has sd 8-12)")
print("=" * 92)
print(f"{'cell':32s} {'per-seed last-20%':38s} {'mean':>7s} {'sd':>6s} {'rng01':>6s}")
for name, _, _, _ in CELLS:
    ps = rows[name]
    vals = {s: v["eval/return_mean"][0] for s, v in ps.items()}
    if not vals:
        print(f"{name:32s} (no data)")
        continue
    txt = " ".join(f"s{s}={v:6.2f}" for s, v in sorted(vals.items()))
    xs = list(vals.values())
    sd = st.pstdev(xs) if len(xs) > 1 else float("nan")
    v01 = [vals[s] for s in (0, 1) if s in vals]
    rng01 = (max(v01) - min(v01)) if len(v01) == 2 else float("nan")
    print(f"{name:32s} {txt:38s} {st.fmean(xs):7.2f} {sd:6.2f} {rng01:6.2f}")

print()
print("=" * 92)
print("LAST-20% per regime  (g0/g4 team return, g2/g3 per-agent-ish, g1 NOT meaningful as return)")
print("=" * 92)
hdr = f"{'cell':32s}" + "".join(f"{'g'+str(g):>10s}" for g in range(5))
print(hdr)
for name, _, _, _ in CELLS:
    ps = rows[name]
    if not ps:
        continue
    line = f"{name:32s}"
    for g in range(5):
        t = f"eval/return_regime_{g}"
        xs = [v[t][0] for v in ps.values() if t in v]
        line += f"{st.fmean(xs):10.2f}" if xs else f"{'--':>10s}"
    print(line)

print()
print("=" * 92)
print("FINAL-checkpoint return_per_regime from eval_report.json (128 eps) - for reference only")
print("=" * 92)
for name, _, _, _ in CELLS:
    ps = rows[name]
    for s, v in sorted(ps.items()):
        f = v.get("_final")
        if not f:
            continue
        rpr = f.get("return_per_regime", {})
        g = " ".join(f"g{k}={rpr[k]:7.2f}" for k in sorted(rpr))
        print(f"{name:32s} s{s}  mean={f['return_mean']:6.2f}  {g}")

print()
print("n_eval_points per run (sanity: all runs reached the same training length)")
for name, _, _, _ in CELLS:
    for s, v in sorted(rows[name].items()):
        n, k = v["eval/return_mean"][2], v["eval/return_mean"][3]
        print(f"  {name:32s} s{s}  points={n:4d}  tail_k={k}")
