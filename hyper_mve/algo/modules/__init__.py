"""Subjective conditioning modules for the mazero_mixed method (v5 lineage).

Dual-path subjective context encoding (role + belief), regime belief
inference (BeliefNet GRU), hypernetwork parameter generation, functional
heads, and belief gradient gating. Consumed by
``hyper_mve/algo/mazero_mixed/config/relation/subjective_model.py`` and the
fork trainer (belief losses).
"""
from __future__ import annotations

from hyper_mve.algo.modules.role_encoder import RoleEncoder
from hyper_mve.algo.modules._belief_obs_encoder import BeliefObsEncoder
from hyper_mve.algo.modules.belief_net import BeliefNet
from hyper_mve.algo.modules.belief_losses import (
    l_regime,
    l_div,
    belief_loss,
    build_oracle_g_seq,
)
from hyper_mve.algo.modules.belief_encoder import BeliefEncoder
from hyper_mve.algo.modules.tri_context_encoder import TriContextEncoder
from hyper_mve.algo.modules.hyper_network import (
    DualHyperNetwork,
    HyperNetMLP,
    reward_diversity_loss,
)
from hyper_mve.algo.modules.functional_nets import (
    FunctionalRewardHead,
    FunctionalPredictionNet,
)
from hyper_mve.algo.modules.grad_gating import BeliefGradGating

__all__ = [
    # sub-encoders
    "RoleEncoder",
    "BeliefEncoder",
    "TriContextEncoder",
    # belief net
    "BeliefObsEncoder",
    "BeliefNet",
    # belief losses + oracle g
    "l_regime",
    "l_div",
    "belief_loss",
    "build_oracle_g_seq",
    # hypernet (subjective)
    "DualHyperNetwork",
    "HyperNetMLP",
    "reward_diversity_loss",
    # functional heads (subjective)
    "FunctionalRewardHead",
    "FunctionalPredictionNet",
    # grad gating
    "BeliefGradGating",
]
