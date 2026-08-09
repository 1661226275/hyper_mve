# Removing g1: the `g2cm` family and the `rel_coopmix` preset (2026-08-09)

Supersedes `g1_removal_handoff.md`, whose Trap 4 was **wrong** — see "The
correction" below.

## The decision

Restrict the regime family so that a change of role relationship never changes
the *nature of the game*. `mutual_comp` was the only regime with both weights
negative, hence the only purely adversarial one, and its presence meant no single
training objective was correct across the family. Removing it makes every regime
compatible with individual optimality.

It was also empirically inert as a return regime: ≈ 0 for every algorithm, method
and baselines alike (`comparison_3seed_final.md:43-45`).

## What was built

New family **`g2cm`** ("cooperative + mixed") and new preset **`rel_coopmix`**
(env id `relation_coopmix`). `g2`, `rel_duo` and `rel_recip` are untouched and
stay frozen, so no archived run changes meaning.

| new id | name | (w01, w10) | old `g2` id | character |
|---|---|---|---|---|
| 0 | `mutual_coop` | (+λ, +λ) | 0 — unchanged | cooperative |
| 1 | `asym_exploit` | (−λ, +λ) | 2 | mixed |
| 2 | `asym_exploited` | (+λ, −λ) | 3 | mixed |
| 3 | `asym_exploit_mild` | (−λ, 0) | — **new** | mixed |
| 4 | `neutral` | (0, 0) | 4 — unchanged | independent |

`mutual_coop` keeps id 0 and `neutral` keeps id 4 deliberately: the empirical
price-of-anarchy denominator reads `welfare_physical["0"]`, and probes that need
`W = 0` to read back physical harvests pin the zero-coupling regime. Only ids
1–3 differ from `g2`.

`rel_coopmix_holdout` trains on `(0, 1, 4)` = one regime of each character
(cooperative / mixed / independent) and holds out `asym_exploited` +
`asym_exploit_mild`. **The literal `(0, 1, 4)` is byte-identical to
`rel_recip_holdout`'s value while naming different regimes** — id 1 was
`mutual_comp`, it is now `asym_exploit`. Tests assert on resolved names for
exactly this reason.

## The correction — why a regime had to be ADDED

`g1_removal_handoff.md:97-108` predicted VoI would survive or rise, reasoning
that `mutual_comp` is symmetric and so contributes ~nothing. That reasoning was
about g1's *own* contribution and missed g1's role as a **confusion partner**.

Under `reciprocal` coupling agent 0 observes its own row `w_01` and must infer
the hidden `w_10`. VoI comes only from own-row values consistent with more than
one regime; a singleton bucket contributes exactly 0 by construction, and
`regime_voi_probe.voi()` averages unweighted over the three buckets.

| agent 0 observes | `g2` | bare removal | `g2cm` |
|---|---|---|---|
| `w01 = +λ` | {coop, exploited} → 5.99 | unchanged → 5.99 | unchanged → 5.99 |
| `w01 = −λ` | {comp, exploit} → 5.99 | **{exploit} → 0** | {exploit, exploit_mild} → 0.79 |
| `w01 = 0` | {neutral} → 0 | 0 | 0 |
| **total** | **3.99** | **2.00** | **2.26** |

All three measured with `scripts/probes/regime_voi_probe.py --episodes 20`. The
`g2` figure reproduces the archived `3.9926904766343` exactly after the probe was
de-hardcoded, which is what licenses comparing the other two against it. The bare
removal was measured with the new `--regimes 0 2 3 4` flag and confirms the
closed-form prediction of 1.996 to three decimals.

### 2.26 is a ceiling, not a tuning choice

The recovery is real but weak — 57% of the `g2` value, and the `−λ` branch
recovers only 0.79 of the 5.99 it lost. That is structural, not a bad pick:

> Under reciprocal coupling `ŵ01 = w10`. For the `w01 = −λ` bucket to contain a
> *negative* `ŵ`, some regime would need `w01 = −λ` **and** `w10 < 0` — both
> weights negative, i.e. exactly the purely adversarial regime the scope decision
> removes. So `{+λ, 0}` is the widest weight spread that bucket can have, and
> **no 5-regime family satisfying the constraint can do better.**

The constraint caps the metric. Trying other replacement regimes is not a lever.

### The lever that does exist

The `w01 = 0` bucket is still a singleton `{neutral}`, contributing a hard 0 to
the three-way mean. A regime with `w01 = 0, w10 = ±λ` has at most one negative
weight, so it is admissible under the same criterion, and would make that bucket
ambiguous too. Extrapolating from the measured spread-1 branch (0.79), one such
regime gives ≈ 2.5 and a `(0, +λ)` / `(0, −λ)` pair gives ≈ 4.3 — above the `g2`
baseline. **Estimates, not measurements**, and |G| would grow to 6 or 7. Out of
scope for this change; recorded so the option is not lost.

## Reporting payoff

`v6_reporting_protocol.md`'s hardest rule — *"g1 is scored on NashConv,
everything else on return, never mix the tables"* — is **deleted**. All five
regimes score on return; NashConv is an optional diagnostic. The duplicate copy
in `per_agent_mcts_handoff.md` is banner-marked as superseded.

Agent-slots the summed metric can respond to went from **6 of 10** to **8 of 10**:
`mutual_comp` cancelled to `−ε·(moves)` regardless of policy, so it was
maximised by standing still. `asym_exploit_mild` sums to `(u_0+u_1)/2` and is
blind to nothing.

## What did NOT change

- **Chance regime accuracy stays 0.200.** |G| is still 5, so the standing
  collapsed-belief test ("`regime_acc = 0.200` = exactly chance for 5 regimes")
  in `method_of_record.md` and `stageA_selection_2x2.md` **remains valid**. This
  is a direct payoff of restoring a regime rather than shipping four.
- **`agent_q_softmax` stays parked.** Its only significant effect was g1
  NashConv (+17.61 ± 4.08); rescored without g1 it is +6.85 / +7.56 at 0.66 /
  0.44 se — unresolved and underpowered, *not* rejected. Needs n ≈ 30/cell.
- **Two g1-justified decisions are annotated, not retuned**: the `min_qstd`
  advantage-spread threshold (`core/utils.py`) and the `visit_q_blend` arm
  selection (`ablation/arms.py`). Both cite measurements from a regime that no
  longer exists; re-deriving them on a guess would be worse than flagging them.

## Schema

`rel-v1 → rel-v2`, `game-metrics-v1 → v2`, `evaldiag-v2 → v3`,
`fidelity-v1 → v2`, and a new `regime_names` field on every per-regime report.
Regime ids are family-relative — `g1` is `mutual_comp` under `g2` and
`asym_exploit` under `g2cm` — so an id-keyed report is not self-describing and
two of them must not be compared silently.

## Archived tables

Every per-regime table under `results/analysis/` predating this date uses **v5
`g2` ids**, where `g1` = `mutual_comp`. They are stamped, not edited.
