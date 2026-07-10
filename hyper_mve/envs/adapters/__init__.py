"""Adapters for external (non-hyper_mve) MARL frameworks (pkg-07 G4, v5).

This subpackage exposes ``RelationCommonsPettingZooEnv`` — a
``pettingzoo.ParallelEnv`` wrapper around
:class:`hyper_mve.envs.relation_commons.RelationCommonsEnv` with two-flag
information gating (``oracle_mode`` + ``eval_info_mode``) per pkg-07 spec 04
§7 (v5: oracle fields are ``g_true`` / ``rows``).

``ResourceCommonsPettingZooEnv`` is a legacy alias (removed in Stage 6).
"""
from __future__ import annotations

from .pettingzoo_wrapper import (
    RelationCommonsPettingZooEnv,
    ResourceCommonsPettingZooEnv,
)

__all__ = ["RelationCommonsPettingZooEnv", "ResourceCommonsPettingZooEnv"]
