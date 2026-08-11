# Module-1 (role awareness) on `rel_coopmix`: five measurements, one conclusion

**2026-08-11, seed 0, 500k env steps.** Five independent measurements agree that
the role-aware / belief module cannot be shown to do work on this environment —
and that this is substantially a property of the ENVIRONMENT, not only of the
module. Recorded together because no single one of them is decisive and the
agreement is the result.

## 1. The ablation costs nothing

Robust last-20% of each run's own `eval/return_mean` (the endpoint is not a
usable basis, `late_training_instability.md`):

| arm | robust | sd | endpoint | wall |
|---|---|---|---|---|
| method `ref_bc_anneal_scaled_hardval_decoupled` | 102.35 | 1.52 | 102.03 | 6.7 h |
| **`..._no_subjective_decoupled` (Module-1 removed)** | **102.04** | 1.56 | 101.97 | 4.6 h |

Gap **0.31** against within-run sds of ~1.5. Removing the subjective module —
the hypernet-generated per-agent heads and the belief posterior that drives them
— costs nothing measurable, while keeping it costs 2.1 h more wall-clock per run.

## 2. The environment's information is worth ~2 points, which is the noise floor

`scripts/probes/regime_voi_probe.py --preset rel_coopmix --episodes 200`:

**VoI = 2.256** for the configured (reciprocal) coupling. The probe prints its
own gate: *"VoI must clear the method's seed sd to be detectable at 3 seeds."*
The method's seed sd, from the 3-seed method of record, is **±2.16**.

So the entire benefit a PERFECT regime oracle could deliver is about the size of
the seed noise. No module, however good, can produce a clearly detectable effect
here. Note the g1 removal cost measurement power: VoI was **3.99** on
`rel_recip`, and is 2.256 on `rel_coopmix`. `hyper-mve-g1-removal-scope` flagged
exactly this risk and said to measure rather than trust it — measured, it fell
by 44%.

Per own-row breakdown: the value is concentrated entirely in `+lam` (6.176),
with `-lam` at 0.593 and the zero row at 0.000.

## 3. Perfect regime knowledge is worth +1.21, inside one standard error

`scripts/probes/value_deploy_probe.py`, 32 episodes/regime:

| deploy mode | mean | sem |
|---|---|---|
| bayes (the method's own) | 101.986 | 2.760 |
| **oracle (true regime given)** | **103.193** | 2.822 |
| argmax | 102.192 | 2.811 |
| temp 0.3 / 0.5 / 0.7 | 102.137 / 102.490 / 101.753 | ~2.8 |

Oracle − bayes = **+1.21 against sem ≈ 2.8** — indistinguishable from zero, and
below even the 2.256 VoI ceiling. This reproduces the 1M `rel_duo` result
(+0.67 ± 2.35) on the new environment and budget.

## 4. The posterior mostly never identifies the regime

`scripts/probes/belief_confusion_probe.py`, 64 episodes/regime. Settle time =
first step that is correct AND holds maxprob ≥ 0.50 for the rest of the episode:

| regime | never settles | median t* | \|V(belief) − V(oracle)\| |
|---|---|---|---|
| g0 `mutual_coop` | 3.1 % | 0 | 2.992 |
| g1 `asym_exploit` | 98.4 % | 97 | 1.857 |
| g2 `asym_exploited` | **100 %** | — | 1.747 |
| g3 `asym_exploit_mild` | 87.5 % | 69.5 | 2.681 |
| g4 `neutral` | 14.1 % | 2 | 0.953 |

At a 0.70 threshold every regime except g4 never settles. The value swing is
0.95–2.99 throughout, consistent with §2 and §3.

## 5. Return does not depend on getting the regime right

The per-role split of the asymmetric regimes is the sharpest single number here:

| regime / agent | belief accuracy | mean return |
|---|---|---|
| g1 agent 0 | 0.603 | 19.78 |
| g1 agent 1 | 0.014 | −0.26 |
| **g2 agent 0** | **0.001** | 0.33 |
| **g2 agent 1** | **0.000** | **19.78** |
| g3 agent 0 | 0.378 | 17.55 |
| g3 agent 1 | 0.055 | 0.06 |

In g1 the winning agent has belief accuracy 0.603 and earns 19.78. In g2 — which
is g1 with the agent indices swapped — the winning agent has accuracy **0.000**
and earns **exactly the same 19.78**. Whatever produces the return, it is not
regime inference.

## What this does and does not establish

**Established:** on `rel_coopmix` at 500k, seed 0, the role-aware module produces
no detectable return benefit, and the environment's own information content
(2.256) is at the noise floor (2.16), so the experiment could not have shown a
clear benefit even in principle.

**Not established:** that the module is worthless in general. Two readings remain
open and this data cannot separate them — the module genuinely contributes
nothing, or every test of it here is underpowered by construction. The
distinction matters for the writeup: the honest claim is about THIS environment.

**The constructive consequence** is that VoI is the quantity to design against.
An environment where regime knowledge is worth ~2 points cannot support a
role-awareness claim at 3 seeds regardless of the architecture. `rel_recip`
scored 3.99 and was dropped for scope reasons; raising VoI well above the seed
sd is the precondition for this line of experiments to be able to say anything.

**Caveats.** n=1 seed for §1; §3 and §4 are single-checkpoint probes. The
`regime_voi_probe` note applies to §2: it is a LOWER bound, since a 1-D scripted
threshold family cannot exploit everything a learned policy could.

## Related

Consistent with [[hyper-mve-bayes-averaging-negative-result]] (the posterior
contributes ~nothing, 3 measurements) and with `v7_cadence_probe.md`, where
`head_diversity` rose monotonically with more optimisation while return fell —
head specialisation not converting into return is the same phenomenon seen from
the training side.
