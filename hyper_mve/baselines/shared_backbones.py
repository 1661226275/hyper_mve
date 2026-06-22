"""Shared-backbone factories for the 5 internal baseline variants (pkg-07 spec 02).

Each ``create_*`` function returns a fresh ``nn.Module`` instance whose
parameter count matches the corresponding backbone in :class:`hyper_mve.models.HyperMuZeroModel`
bit-for-bit (SB2 / C7-INT-FAIR2). Backbone classes are imported from
``hyper_mve.models`` (the pkg-03/04 published surface); this module is a thin
wrapper that guarantees the 5 internal variants + hyper all hit the *same*
construction path.

External baselines (``external_mappo`` / ``external_qmix`` /
``external_ma_muzero_gh`` / ``external_mamba`` / stubs) **MUST NOT** import
from this module (pkg-07 design §D4). They bring their own backbones via the
PettingZoo adapter (spec 04).
"""
from __future__ import annotations

import torch.nn as nn

from hyper_mve.configs import EnvConfig, ModelConfig
from hyper_mve.models import (
    BeliefNet,
    RepresentationNet,
    TriContextEncoder,
)


# The exact attribute names that 5 internal model classes + HyperMuZeroModel
# all use for the three shared backbones (SB5). ``count_conditioning_params``
# filters parameter names by these prefixes.
SHARED_BACKBONE_PREFIXES: tuple[str, ...] = (
    "rep_net", "belief_net", "tri_context_encoder",
)


class _RepNetCfgAdapter:
    """Thin wrapper that exposes ``cfg.env`` / ``cfg.model`` to RepresentationNet.

    The pkg-04 ``RepresentationNet(cfg)`` constructor accepts a ``V4Config``-shaped
    object (it reads ``cfg.env.N`` / ``cfg.env.K`` / ``cfg.model.latent_dim`` etc.).
    Wrapping ``(env_cfg, model_cfg)`` into a duck-typed config keeps the factory
    signature stable (pkg-07 spec 02 §2.2) without forcing a RepresentationNet
    signature change (pkg-04 spec 02 line 124 stays unchanged).
    """
    __slots__ = ("env", "model")

    def __init__(self, env_cfg: EnvConfig, model_cfg: ModelConfig) -> None:
        self.env = env_cfg
        self.model = model_cfg


def create_rep_net(env_cfg: EnvConfig, model_cfg: ModelConfig) -> RepresentationNet:
    """Build a RepresentationNet (pkg-04 spec 02 §3.4) — independent instance."""
    return RepresentationNet(_RepNetCfgAdapter(env_cfg, model_cfg))


def create_belief_net(env_cfg: EnvConfig, model_cfg: ModelConfig) -> BeliefNet:
    """Build a BeliefNet (pkg-03 spec 04) — independent instance.

    .. note:: ``no_belief`` variants still construct a full BeliefNet (SB2 /
       C7-INT-FAIR2). The belief path is zeroed inside ``set_context_subjective``
       at information-flow time, not at parameter time.
    """
    return BeliefNet(env_cfg, model_cfg)


def create_tri_context_encoder(env_cfg: EnvConfig, model_cfg: ModelConfig) -> TriContextEncoder:
    """Build a TriContextEncoder (pkg-03 spec 01) — independent instance."""
    return TriContextEncoder(env_cfg, model_cfg)


def count_conditioning_params(model: nn.Module) -> int:
    """Count parameters in the conditioning subsystem (pkg-07 design §D5).

    The conditioning subsystem is everything that is NOT a shared backbone —
    hypernet generators (when present), functional nets, RewardHead, ID
    embeddings, etc. Used by spec 07 5%/10% fairness check and by spec 01
    C7-INT-CFG1 to assert the four ``cfg.baselines.internal_*`` capacity knobs
    flow into actual params.
    """
    if not hasattr(model, "SHARED_BACKBONE_PREFIXES"):
        raise AttributeError(
            f"{type(model).__name__} must define SHARED_BACKBONE_PREFIXES "
            "to be counted by count_conditioning_params (pkg-07 spec 02 §3.3)."
        )
    prefixes = tuple(model.SHARED_BACKBONE_PREFIXES)
    total = 0
    for name, p in model.named_parameters():
        if any(name.startswith(prefix + ".") or name == prefix for prefix in prefixes):
            continue
        total += p.numel()
    return total


__all__ = [
    "SHARED_BACKBONE_PREFIXES",
    "create_rep_net",
    "create_belief_net",
    "create_tri_context_encoder",
    "count_conditioning_params",
]
