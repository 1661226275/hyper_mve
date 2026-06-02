"""Pkg-03: TriContextEncoder + BeliefNet 公开 API.

三路条件编码 (Ch4.2.4) + 信念推断网络 (Ch4.2.3 + 4.5). 供 Pkg-04 DualHyperNetwork
v2 / Pkg-05 trainer/worker 消费.
"""
from __future__ import annotations

from hyper_mve.models.c_encoder import CEncoder
from hyper_mve.models.role_encoder import RoleEncoder
from hyper_mve.models.permutation_invariant_pool import (
    MeanPool,
    MaxPool,
    AttentionPool,
    make_pool,
)
from hyper_mve.models._belief_obs_encoder import BeliefObsEncoder
from hyper_mve.models._belief_id_emb import BeliefIdEmbedding
from hyper_mve.models.belief_net import BeliefNet
from hyper_mve.models.belief_losses import (
    l_c,
    l_opp,
    l_div,
    belief_loss,
    build_oracle_z_seq,
)
from hyper_mve.models.belief_encoder import BeliefEncoder
from hyper_mve.models.tri_context_encoder import TriContextEncoder

# Pkg-04: DualHyperNetwork v2 + HyperMuZeroModel + functional nets + RepNet + grad gating
from hyper_mve.models.hyper_network import DualHyperNetwork, HyperNetMLP, reward_diversity_loss
from hyper_mve.models.functional_nets import (
    FunctionalStateTransNet,
    FunctionalRewardHead,
    FunctionalPredictionNet,
)
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
    "CEncoder",
    "RoleEncoder",
    "BeliefEncoder",
    "TriContextEncoder",
    # pooling
    "MeanPool",
    "MaxPool",
    "AttentionPool",
    "make_pool",
    # belief net
    "BeliefObsEncoder",
    "BeliefIdEmbedding",
    "BeliefNet",
    # belief losses + oracle z
    "l_c",
    "l_opp",
    "l_div",
    "belief_loss",
    "build_oracle_z_seq",
    # Pkg-04: hypernet v2
    "DualHyperNetwork",
    "HyperNetMLP",
    "reward_diversity_loss",
    # Pkg-04: functional nets
    "FunctionalStateTransNet",
    "FunctionalRewardHead",
    "FunctionalPredictionNet",
    # Pkg-04: representation net
    "RepresentationNet",
    "Projector",
    "cosine_similarity_loss",
    "negative_cosine_similarity",
    # Pkg-04: grad gating + model
    "BeliefGradGating",
    "HyperMuZeroModel",
]
