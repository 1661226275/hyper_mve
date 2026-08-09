"""g1 NashConv across the agent-target 2x2 -- the pre-registered instrument.

All 12 numbers were measured in ONE session at br_env_steps=20000, so the BR
side is matched; only the checkpoints differ in provenance (qtarget/mctsfix are
wave-1's). Wave-1's originally reported values are printed alongside so the
BR-noise floor on this instrument is visible rather than assumed.
"""
import json
import os
import statistics as st

D = os.path.dirname(os.path.abspath(__file__))

CELLS = [
    ("none/q_softmax  (qtarget)", "qtarget", [0, 1], {0: 4.90, 1: 5.96}),
    ("star/q_softmax  (mctsfix)", "mctsfix", [0, 1], {0: 3.72, 1: 22.71}),
    ("none/agent_q    (agentq)", "agentq", [0, 1, 2, 3], {}),
    ("star/agent_q    (agentqcov)", "agentqcov", [0, 1, 2, 3], {}),
]


def load(tag, s):
    with open(f"{D}/gm_{tag}_s{s}.json") as f:
        r = json.load(f)
    return (r["nashconv"]["1"], r["welfare_physical"]["1"],
            r["exploitability"]["1"], r["v_pi"]["1"])


print("=" * 100)
print("g1 NashConv, planner frozen, br_env_steps=20000, 16 eps  (LOWER = closer to equilibrium)")
print("=" * 100)
print(f"{'cell':30s} {'per-seed NashConv':40s} {'mean':>7s} {'sd':>6s} {'rng01':>7s}")
res = {}
for name, tag, seeds, _ in CELLS:
    vals = {s: load(tag, s)[0] for s in seeds}
    res[name] = vals
    txt = " ".join(f"s{s}={v:6.2f}" for s, v in sorted(vals.items()))
    xs = list(vals.values())
    sd = st.pstdev(xs) if len(xs) > 1 else float("nan")
    v01 = [vals[s] for s in (0, 1) if s in vals]
    rng01 = max(v01) - min(v01)
    print(f"{name:30s} {txt:40s} {st.fmean(xs):7.2f} {sd:6.2f} {rng01:7.2f}")

print()
print("Re-measurement vs wave-1's reported values (same ckpt, fresh BR):")
print(f"{'cell':30s} {'seed':>5s} {'wave-1':>8s} {'now':>8s} {'delta':>8s}")
for name, tag, seeds, old in CELLS:
    for s in sorted(old):
        now = res[name][s]
        print(f"{name:30s} {s:5d} {old[s]:8.2f} {now:8.2f} {now - old[s]:+8.2f}")

print()
print("=" * 100)
print("PRE-REGISTERED TEST: does agent_q collapse the star seed spread toward the none cells?")
print("=" * 100)
sq = res["star/q_softmax  (mctsfix)"]
sa = res["star/agent_q    (agentqcov)"]
nq = res["none/q_softmax  (qtarget)"]
na = res["none/agent_q    (agentq)"]


def rng(d, seeds):
    xs = [d[s] for s in seeds if s in d]
    return max(xs) - min(xs)


print(f"  matched seeds 0-1 range   star/q_softmax {rng(sq,[0,1]):6.2f}"
      f"  ->  star/agent_q {rng(sa,[0,1]):6.2f}")
print(f"  matched seeds 0-1 range   none/q_softmax {rng(nq,[0,1]):6.2f}"
      f"  ->  none/agent_q {rng(na,[0,1]):6.2f}")
print(f"  full-n sd                 star/agent_q (n=4) {st.pstdev(list(sa.values())):6.2f}"
      f"   none/agent_q (n=4) {st.pstdev(list(na.values())):6.2f}")
print(f"  star-minus-none sd gap    agent_q {st.pstdev(list(sa.values())) - st.pstdev(list(na.values())):+6.2f}"
      f"   (wave-1 q_softmax range gap {rng(sq,[0,1]) - rng(nq,[0,1]):+6.2f})")

print()
print("welfare_physical + per-agent exploitability / v_pi (g1)")
for name, tag, seeds, _ in CELLS:
    for s in seeds:
        nc, w, ex, vp = load(tag, s)
        print(f"  {name:30s} s{s}  nash={nc:6.2f}  welfare={w:7.2f}  "
              f"expl=({ex[0]:6.2f},{ex[1]:6.2f})  v_pi=({vp[0]:6.2f},{vp[1]:6.2f})")
