"""Rescore wave 2 on the proposed g1-free family {g0, g2, g3, g4}.

Motivation: the ONLY significant effect in wave 2 was g1 NashConv (+17.61,
4.3 se), and g1 is the regime being removed. So the rejection of
agent_q_softmax rests entirely on a regime that leaves the scope. This
recomputes the return contrast over the retained regimes only.

Uses the unweighted mean of the four retained per-regime last-20% curves, NOT
eval/return_mean (which averages all five).
"""
import glob
import statistics as st

from tensorboard.backend.event_processing import event_accumulator

NEW = "results_v6_agentq"
OLD = "/home/data/zhengwenbo/hyper_mve/results_v6_2x2"
P = "mazero_mixed_ref_bc_anneal_scaled_hardval_decoupled"
KEEP = [0, 2, 3, 4]          # g1 dropped

CELLS = [
    ("none/q_softmax", OLD, "_qtarget", [0, 1]),
    ("star/q_softmax", OLD, "_mctsfix", [0, 1]),
    ("none/agent_q", NEW, "_agentq", [0, 1, 2, 3]),
    ("star/agent_q", NEW, "_agentq_cover", [0, 1, 2, 3]),
]


def run_value(root, suffix, seed):
    d = f"{root}/{P}{suffix}/relation_recip/seed{seed}"
    ev = sorted(glob.glob(f"{d}/tb/events*"))
    ea = event_accumulator.EventAccumulator(ev[-1], size_guidance={"scalars": 0})
    ea.Reload()
    per_regime = []
    for g in KEEP:
        s = ea.Scalars(f"eval/return_regime_{g}")
        k = max(1, int(round(0.2 * len(s))))
        per_regime.append(st.fmean(x.value for x in s[-k:]))
    return st.fmean(per_regime), per_regime


res = {}
print("last-20%, mean over RETAINED regimes {g0,g2,g3,g4} -- g1 excluded")
for name, root, suffix, seeds in CELLS:
    vals = {}
    for s in seeds:
        v, pr = run_value(root, suffix, s)
        vals[s] = v
    res[name] = vals
    xs = list(vals.values())
    sd = st.stdev(xs)
    txt = " ".join(f"s{s}={v:6.2f}" for s, v in sorted(vals.items()))
    print(f"  {name:16s} n={len(xs)}  {txt:40s} mean={st.fmean(xs):6.2f} "
          f"sd={sd:6.2f} sem={sd/len(xs)**0.5:5.2f}")

print()
print("Contrast agent_q - q_softmax on the retained family:")
for cov in ("none", "star"):
    a = list(res[f"{cov}/agent_q"].values())
    q = list(res[f"{cov}/q_softmax"].values())
    d = st.fmean(a) - st.fmean(q)
    se = ((st.stdev(a) / len(a) ** 0.5) ** 2 + (st.stdev(q) / len(q) ** 0.5) ** 2) ** 0.5
    verdict = "significant" if abs(d) > 2 * se else "NOT significant"
    print(f"  {cov:5s}  delta={d:+7.2f}  se={se:5.2f}  ({d/se:+5.2f} se) -> {verdict}")
