"""ResourceCommons environment package (Pkg-02 output).

Public surface:

    >>> from hyper_mve.configs import V4Config
    >>> from hyper_mve.envs.resource_commons import ResourceCommonsEnv
    >>> cfg = V4Config.from_preset("medium")
    >>> env = ResourceCommonsEnv(cfg.env, seed=42)
    >>> obs, info = env.reset()
"""
from __future__ import annotations

from .env import ResourceCommonsEnv, make_resource_commons

__all__ = ["ResourceCommonsEnv", "make_resource_commons"]
