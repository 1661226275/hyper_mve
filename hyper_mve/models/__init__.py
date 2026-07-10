"""Pkg-03/04 公开 API (v5, Pkg-09 amendment).

双路主观条件编码 (role + belief) + regime 信念推断网络 + 主观 hypernet +
普通共享 TransitionNet. 供 Pkg-05 trainer/worker 消费.
"""
from __future__ import annotations

from hyper_mve.models.role_encoder import RoleEncoder
from hyper_mve.models._belief_obs_encoder import BeliefObsEncoder
from hyper_mve.models.belief_net import BeliefNet
from hyper_mve.models.belief_losses import (
    l_regime,
    l_div,
    belief_loss,
    build_oracle_g_seq,
)
from hyper_mve.models.belief_encoder import BeliefEncoder
from hyper_mve.models.tri_context_encoder import TriContextEncoder

# Pkg-04: DualHyperNetwork + HyperMuZeroModel + functional nets + RepNet + grad gating
from hyper_mve.models.hyper_network import DualHyperNetwork, HyperNetMLP, reward_diversity_loss
from hyper_mve.models.functional_nets import (
    FunctionalRewardHead,
    FunctionalPredictionNet,
)
from hyper_mve.models.transition_net import TransitionNet
from hyper_mve.models.representation_net import (
    RepresentationNet,
    Projector,
    cosine_similarity_loss,
    negative_cosine_similarity,
)
from hyper_mve.models.grad_gating import BeliefGradGating
from hyper_mve.models.hyper_muzero_model import HyperMuZeroModel

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
    # functional nets (subjective) + plain transition
    "FunctionalRewardHead",
    "FunctionalPredictionNet",
    "TransitionNet",
    # representation net
    "RepresentationNet",
    "Projector",
    "cosine_similarity_loss",
    "negative_cosine_similarity",
    # grad gating + model
    "BeliefGradGating",
    "HyperMuZeroModel",
]
