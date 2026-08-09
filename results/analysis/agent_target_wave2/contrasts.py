"""Sample sd / SEM on the return side, so the agent_q return win is stated with
its uncertainty rather than as a bare point gap."""
import statistics as st

# last-20% eval/return_mean, from agg.py
R = {
    "none/visit": [26.77, 29.10],
    "star/visit": [37.55, 13.95],
    "none/q_softmax": [28.95, 41.28],
    "star/q_softmax": [26.80, 42.50],
    "none/agent_q": [56.63, 31.72, 35.09, 38.73],
    "star/agent_q": [25.24, 72.86, 38.33, 26.30],
}
N = {  # g1 NashConv
    "none/q_softmax": [4.90, 5.96],
    "star/q_softmax": [3.72, 22.71],
    "none/agent_q": [19.19, 13.52, 28.76, 30.68],
    "star/agent_q": [12.44, 24.13, 26.49, 6.92],
}


def line(k, xs):
    n = len(xs)
    sd = st.stdev(xs)          # sample sd, not population
    sem = sd / n ** 0.5
    m = st.fmean(xs)
    return f"  {k:18s} n={n}  mean={m:7.2f}  sd={sd:6.2f}  sem={sem:6.2f}  95%CI~[{m-2*sem:6.2f},{m+2*sem:6.2f}]"


print("RETURN (last-20%)")
for k, xs in R.items():
    print(line(k, xs))
print()
print("g1 NASHCONV")
for k, xs in N.items():
    print(line(k, xs))

print()
print("Contrasts (agent_q - q_softmax), with pooled SEM:")
for cov in ("none", "star"):
    for label, D in (("return", R), ("nashconv", N)):
        a = D[f"{cov}/agent_q"]
        q = D[f"{cov}/q_softmax"]
        da = st.fmean(a) - st.fmean(q)
        sa = st.stdev(a) / len(a) ** 0.5
        sq = st.stdev(q) / len(q) ** 0.5
        se = (sa ** 2 + sq ** 2) ** 0.5
        sig = "significant" if abs(da) > 2 * se else "NOT significant"
        print(f"  {cov:5s} {label:9s} delta={da:+7.2f}  se={se:6.2f}  ({da/se:+5.2f} se) -> {sig}")
