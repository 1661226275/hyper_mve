"""Comparison-runner registry (phase-2 realignment).

Lazy string registry: values are ``"module:Class"`` targets resolved on
demand, so listing algorithms never imports torch. ``"mazero_mixed"`` is THE
METHOD (lives in :mod:`hyper_mve.algo`); every other key is a comparison
baseline. Keys for HAPPO / MBOM / MBOM-oracle / M3W-adapted are appended by
realignment phases 4–6 (target end state: 7 keys).

The v5 internal-baseline namespace (input_wide / no_belief / …) and the
``external_*`` CLI prefix are retired with the v5 stack.
"""
from __future__ import annotations

import importlib
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

from hyper_mve.comparison.base import ExternalBaselineRunner, _FORBIDDEN_INFO_KEYS

if TYPE_CHECKING:  # pragma: no cover
    from hyper_mve.utils.configs import V4Config

REGISTRY: Mapping[str, str] = MappingProxyType({
    "mazero_mixed": "hyper_mve.algo.runner:MAZeroMixedRunner",
    "mappo":        "hyper_mve.comparison.mappo:MAPPOAlgorithm",
    "mamba":        "hyper_mve.comparison.mamba:MAMBAAlgorithm",
    "happo":        "hyper_mve.comparison.happo:HAPPORunner",
    "mbom":         "hyper_mve.comparison.mbom:MBOMRunner",
    "mbom_oracle":  "hyper_mve.comparison.mbom:MBOMOracleRunner",
    "m3w_adapted":  "hyper_mve.comparison.m3w_adapted.runner:M3WAdaptedRunner",
})


def create_runner(cfg: "V4Config", name: str) -> ExternalBaselineRunner:
    """Resolve ``name`` through REGISTRY and instantiate the runner with ``cfg``."""
    try:
        target = REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown runner {name!r}. Valid: {sorted(REGISTRY)}."
        ) from None
    module_name, _, class_name = target.partition(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(cfg)


__all__ = [
    "REGISTRY",
    "create_runner",
    "ExternalBaselineRunner",
    "_FORBIDDEN_INFO_KEYS",
]
