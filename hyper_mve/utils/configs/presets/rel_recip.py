"""rel_recip preset — v6 RelationCommons, where the hidden regime is worth inferring.

``rel_duo`` (v5) has a measured value of information of **exactly zero**: an
agent's reward weight on its neighbour's harvest is ``w_ij``, its **own** row,
which the observation already carries, so the best response cannot depend on the
hidden ``w_ji``. Confirmed three ways in
``results/analysis/regime_knowledge_ceiling.md`` — scripted oracle +0.81,
VoI +0.00, trained-model oracle gap +0.67 ± 2.88.

This preset changes exactly three things, each necessary and none sufficient
alone (measured VoI in the scripted threshold family, in brackets):

1. ``reward_coupling="reciprocal"`` — ``Ŵ = Wᵀ``, so agent ``i``'s weight on
   ``u_j`` is ``w_ji``: reciprocal altruism, and the weight is now *hidden*.
   Necessary because with ``own_row`` the VoI is zero by construction. The
   weight must also **flip sign** with the hidden row — non-flipping designs
   (Levine λ ≤ 1) measure 0.06–0.17 where ``reciprocal`` measures 3.99.
2. ``regrowth_law="logistic"`` with ``alpha=0.30`` — the v5 law regrows fastest
   when a cell is *empty*, so restraint is strictly harmful and there is no
   strategic dimension for the relationship to act on. Logistic growth makes
   the optimal harvest threshold interior (worth ~+116).
3. ``L=3, K=2`` — the binding constraint turned out to be geometry, not reward.
   On the 8×8 grid two agents simply forage apart: agent 0's strategy moved
   agent 1's harvest by *nothing* (sd 0.28 vs 20.31), and even maximal reward
   coupling gave VoI 0.00 there. Shrinking the grid forces interaction
   [K=2 L=8 → 0.26; L=4 → 2.33; **L=3 → 3.99**; L=2 → 5.28]. ``K ≤ N`` is
   required: at K=3 the agents separate again and coupling collapses to 0.07.

L=3 rather than L=2 is a deliberate trade: 3.99 is under the ≥5 gate as stated,
but VoI here is measured within a one-dimensional scripted threshold family and
is a **lower bound** on what a learned policy can exploit, and L=2 would leave a
2×2 grid — a matrix game with a token spatial dimension.

``rel_duo`` is left untouched and is the controlled λ=0 / constant-regrowth
comparison: same agents, same regimes, same everything but these three knobs.
"""
from __future__ import annotations

from dataclasses import replace

from ..env_config import EnvConfig
from ..v4_config import V4Config


def build_rel_recip_config() -> V4Config:
    """v6 RelationCommons: reciprocal reward + logistic commons on a 3×3 grid."""
    from .rel_duo import build_rel_duo_config

    base = build_rel_duo_config()
    env = EnvConfig(
        N=2,
        L=3,                            # forces interaction; see module docstring
        K=2,                            # K ≤ N, else the agents separate again
        T_max=base.env.T_max,
        relation_family="g2",
        relation_intensity=1.0,
        regime_switch_prob=0.0,         # research point 1: per-episode static
        regime_kernel="uniform",
        alpha=0.30,                     # logistic growth rate
        Q_max=base.env.Q_max,
        epsilon_move=base.env.epsilon_move,
        regrowth_law="logistic",
        reward_coupling="reciprocal",
    )
    return replace(base, env=env, preset_name="rel_recip")


def build_rel_recip_holdout_config() -> V4Config:
    """rel_recip restricted to the symmetric training regimes {coop, comp, neutral}.

    Mirrors ``rel_duo_holdout``: trains on regime ids (0, 1, 4) and evaluates on
    the held-out asymmetric regimes (2, 3). Under ``reciprocal`` coupling the
    held-out regimes are exactly the ones where ``w_ji ≠ w_ij``, i.e. the only
    ones a model cannot solve by reading its own row — so this is a sharper
    zero-shot test here than it was on ``rel_duo``.
    """
    base = build_rel_recip_config()
    env = replace(base.env, train_regime_ids=(0, 1, 4))
    return replace(base, env=env, preset_name="rel_recip_holdout")
