"""rel_coopmix preset — v6 RelationCommons on a family with no adversarial regime.

``rel_recip`` on the ``g2cm`` family instead of ``g2``. Every v6 knob is inherited
unchanged (``L=3``, ``K=2``, ``alpha=0.30``, ``regrowth_law="logistic"``,
``reward_coupling="reciprocal"``, ``regime_switch_prob=0.0``), so ``rel_recip`` is the
controlled one-knob comparison: the *only* difference is which regimes exist.

Why the family changed
----------------------
``g2`` spans cooperative through zero-sum. ``mutual_comp`` is the only regime with both
weights negative, so it is the only one where a change of role relationship changes the
**nature of the game** — no single training objective is correct across the family.
``g2cm`` drops it, which makes every regime compatible with individual optimality and
lets all five be scored on **return**. That deletes the hardest rule in the reporting
protocol ("g1 on NashConv, everything else on return, never mix the tables"); NashConv
becomes an optional diagnostic.

``mutual_comp`` was also empirically inert as a *return* regime — ≈ 0 for every
algorithm, method and baselines alike (``results/analysis/comparison_3seed_final.md``).

What replaced it, and why something had to
------------------------------------------
It was **not** inert for value of information. Under ``reciprocal`` coupling agent ``i``
weights ``u_j`` by the hidden ``w_ji`` while observing its own row ``w_ij``, so VoI comes
only from regimes sharing an observed row but differing in the hidden one. Deleting
``mutual_comp`` alone would strand the ``w_01 = -λ`` branch as a singleton, and a
singleton branch has VoI exactly 0 by construction — the measured number would halve,
3.99 -> 1.996 (derived in closed form from the archived
``results/analysis/belief_ceiling/regime_voi_rel_recip.json``, whose 3.9927 decomposes as
``mean(5.989, 5.989, 0.0)`` over the three own-row cases).

``asym_exploit_mild`` (``w_01 = -λ``, ``w_10 = 0``: agent 0 hostile, agent 1 indifferent)
restores that ambiguity with only one negative weight, so it is mixed — the same
character as ``asym_exploit``/``asym_exploited`` — not adversarial.

======  ==================  ===========  ============  =============
new id  name                (w01, w10)   old ``g2`` id  character
======  ==================  ===========  ============  =============
0       mutual_coop         (+λ, +λ)     0 (unchanged)  cooperative
1       asym_exploit        (-λ, +λ)     2              mixed
2       asym_exploited      (+λ, -λ)     3              mixed
3       asym_exploit_mild   (-λ,  0)     — (new)        mixed
4       neutral             ( 0,  0)     4 (unchanged)  independent
======  ==================  ===========  ============  =============

Only ids 1-3 differ from ``g2``. ``mutual_coop`` stays at 0 and ``neutral`` stays at 4
deliberately — see ``build_g2cm``'s docstring for the consumers that depend on those two
positions.
"""
from __future__ import annotations

from dataclasses import replace

from ..v4_config import V4Config


def build_rel_coopmix_config() -> V4Config:
    """v6 RelationCommons with no purely adversarial regime (family ``g2cm``)."""
    from .rel_recip import build_rel_recip_config

    base = build_rel_recip_config()
    env = replace(base.env, relation_family="g2cm")
    return replace(base, env=env, preset_name="rel_coopmix")


def build_rel_coopmix_holdout_config() -> V4Config:
    """rel_coopmix trained on ``(0, 1, 2)``, zero-shot tested on ``(3, 4)``.

    Trains on ``mutual_coop`` (cooperative, symmetric) plus **both** asymmetric
    mirrors, ``asym_exploit`` and ``asym_exploited``; holds out
    ``asym_exploit_mild`` and ``neutral``.

    .. warning::

       **The tuple is a false friend across families.** This preset used to hold
       ``(0, 1, 4)``, byte-identical to ``rel_recip_holdout``'s value but meaning
       something else: under ``g2`` id 1 is ``mutual_comp``, under ``g2cm`` it is
       ``asym_exploit``. Anyone diffing the two preset modules sees "no change" at
       exactly the line where the semantics inverted. Tests here assert on
       **resolved regime names**, not on the tuple, for this reason.

    What each held-out regime actually probes (``_w2(w01, w10)``, ``relation.py:218``;
    ``w01`` is the entry an agent sees in its own row):

    ==========================  ======  ======  ===================================
    regime                      w01     w10     status under ``(0, 1, 2)``
    ==========================  ======  ======  ===================================
    g0 ``mutual_coop``          +λ      +λ      trained
    g1 ``asym_exploit``         -λ      +λ      trained
    g2 ``asym_exploited``       +λ      -λ      trained
    g3 ``asym_exploit_mild``    -λ      0       **held out — unseen COMBINATION**
    g4 ``neutral``              0       0       **held out — unseen own-row VALUE**
    ==========================  ======  ======  ===================================

    The two therefore test different things and should not be reported as one
    number. Training covers own-row values ``{+λ, -λ}`` but never ``0``, so **g4 is
    the only regime presenting an own-row value the model has never trained on**.
    g3's ``-λ`` is already familiar from g1; what is new there is the *pairing*
    (partner row ``0`` rather than ``+λ``), so g3 is the milder, in-distribution
    generalization test and g4 the sharp out-of-distribution one.

    Expect held-out ``regime_accuracy`` near zero: the belief head has 5 classes and
    only 3 occur in training. Chance stays ``0.200`` — |G| is still 5, only the
    training support shrank. Measure rather than argue: run
    ``scripts/probes/belief_confusion_probe.py`` against a trained checkpoint. If the
    posterior degenerates entirely, the fallback is ``(0, 1, 2, 3)``, which keeps a
    partner-row ``0`` in training and holds out only ``neutral``.
    """
    base = build_rel_coopmix_config()
    env = replace(base.env, train_regime_ids=(0, 1, 2))
    return replace(base, env=env, preset_name="rel_coopmix_holdout")
