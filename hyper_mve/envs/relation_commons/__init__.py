"""RelationCommons environment package (v5, Pkg-09).

Public surface:

    >>> from hyper_mve.configs import V4Config
    >>> from hyper_mve.envs.relation_commons import RelationCommonsEnv
    >>> cfg = V4Config.from_preset("rel_duo")
    >>> env = RelationCommonsEnv(cfg.env, seed=42)
    >>> obs, info = env.reset()
"""
from __future__ import annotations

from .env import RelationCommonsEnv, make_relation_commons

__all__ = ["RelationCommonsEnv", "make_relation_commons"]
