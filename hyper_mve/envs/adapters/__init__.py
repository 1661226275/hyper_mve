"""Adapters for external (non-hyper_mve) MARL frameworks (pkg-07 G4).

This subpackage exposes ``ResourceCommonsPettingZooEnv`` — a
``pettingzoo.ParallelEnv`` wrapper around :class:`hyper_mve.envs.resource_commons.env.ResourceCommonsEnv`
with two-flag information gating (``oracle_mode`` + ``eval_info_mode``) per
pkg-07 spec 04 §7.
"""
from __future__ import annotations

from .pettingzoo_wrapper import ResourceCommonsPettingZooEnv

__all__ = ["ResourceCommonsPettingZooEnv"]
