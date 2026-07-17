"""Regime-ified MPE simple_tag (phase-3 realignment).

``MPETagRegimeEnv`` wraps ``mpe2.simple_tag_v3`` behind the same
PettingZoo-parallel dict surface + two-flag info gate as
``RelationCommonsPettingZooEnv``, re-weighting the raw per-agent physical
tag rewards through the hidden relationship regime ``W(g)``
(family ``tag4``, :func:`hyper_mve.utils.schemas.relation.build_tag4`).
"""
from hyper_mve.envs.mpe_tag.env import MPETagRegimeEnv

__all__ = ["MPETagRegimeEnv"]
