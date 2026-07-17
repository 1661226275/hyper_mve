"""pkg-07 baseline package — factory + 10-key REGISTRY (4 internal + 6 external).

Public surface (pkg-07 spec 01 §3.2):

    >>> from hyper_mve.baselines import create_baseline, REGISTRY, BaselineLike
    >>> runner = create_baseline(cfg, "input_wide")        # INTERNAL → BaselineModel
    >>> runner = create_baseline(cfg, "external_mappo")    # EXTERNAL → ExternalBaselineRunner

CLI translation (front-end ↔ factory arg) lives in :func:`cli_to_factory_arg`
because pkg-07 spec 01 §2.2 enforces a prefix asymmetry:

  * ``baseline_*`` CLI strings → factory args strip the ``baseline_`` prefix
    (``baseline_input_wide`` → ``"input_wide"``).
  * Bare-name internal variants pass through unchanged (``no_belief`` →
    ``"no_belief"``).
  * ``external_*`` CLI strings pass through unchanged.

The curriculum-override CLI strings (``hyper`` / ``oracle_only`` /
``infer_only``) **do not enter the factory** — they construct
``HyperMuZeroModel(cfg)`` directly with a ``curriculum_stage_1_end_frac``
override. The factory rejects them with ``ValueError`` so misrouting fails
loudly.
"""
from __future__ import annotations

import warnings
from types import MappingProxyType
from typing import Callable, Mapping, Union

from hyper_mve.configs import V4Config

# Internal variants (4; pkg-07 spec 03, v5).
from hyper_mve.baselines.internal.input_conditioned import (
    InputDeepBaselineModel,
    InputWideBaselineModel,
)
from hyper_mve.baselines.internal.ma_muzero import MAMuZeroBaselineModel
from hyper_mve.baselines.internal.no_belief import NoBeliefBaselineModel

# External runners (6; pkg-07 spec 05 + 06).
from hyper_mve.baselines.external.mappo import MAPPOAlgorithm
from hyper_mve.baselines.external.qmix import QMIXAlgorithm
from hyper_mve.baselines.external.ma_muzero_gh import MAMuZeroGHAlgorithm
from hyper_mve.baselines.external.mamba import MAMBAAlgorithm
from hyper_mve.baselines.external.mazero_mixed import MAZeroMixedRunner
from hyper_mve.baselines.external.stubs import MARIEStub, GAStub
from hyper_mve.baselines.external.base import ExternalBaselineRunner


# pkg-07 spec 01 §3.2 — Union return type. Forward-reference to
# ``BaselineModel`` (defined under ``hyper_mve.baselines.internal.base``)
# avoids a circular import.
BaselineLike = Union[
    "BaselineModel",
    ExternalBaselineRunner,
]


# Internal namespace: 4 keys (pkg-07 spec 01 §2.1; v5 deleted rewardhead_explicit_type
# — discrete-type reward branching is meaningless under continuous relationship rows).
INTERNAL_REGISTRY: Mapping[str, Callable[[V4Config], "BaselineModel"]] = MappingProxyType({
    "input_wide":                InputWideBaselineModel,
    "input_deep":                InputDeepBaselineModel,
    "ma_muzero":                 MAMuZeroBaselineModel,
    "no_belief":                 NoBeliefBaselineModel,
})

# External namespace: 6 keys. ``MAMBAAlgorithm`` resolves to a stub (raises
# ``NotImplementedError`` on ``__init__``) per pkg-07 design D8 — the sourcing
# window is intentionally not opened on this implementation pass.
EXTERNAL_REGISTRY: Mapping[str, Callable[[V4Config], ExternalBaselineRunner]] = MappingProxyType({
    "external_mappo":         MAPPOAlgorithm,
    "external_qmix":          QMIXAlgorithm,
    "external_ma_muzero_gh":  MAMuZeroGHAlgorithm,
    "external_mamba":         MAMBAAlgorithm,
    "external_marie":         MARIEStub,
    "external_ga":            GAStub,
    # THE METHOD (not a baseline): MAZero-fork mixed-game stack behind the
    # same runner contract so the sweep/eval spine drives it unchanged.
    "mazero_mixed":           MAZeroMixedRunner,
})

# Merged read-only view consumed by pkg-08 sweep harness (pkg-07 spec 01 §3.2).
REGISTRY: Mapping[str, Callable[[V4Config], BaselineLike]] = MappingProxyType(
    {**INTERNAL_REGISTRY, **EXTERNAL_REGISTRY}
)


# ----------------------------------------------------------------------- CLI

_CURRICULUM_OVERRIDE_VARIANTS: frozenset[str] = frozenset({
    "hyper", "oracle_only", "infer_only",
})

# CLI prefix-asymmetry sets (pkg-07 spec 01 §5.2).
_CLI_TO_FACTORY_INTERNAL_PREFIX: frozenset[str] = frozenset({
    "baseline_input_wide", "baseline_input_deep", "baseline_ma_muzero",
})
_CLI_BARE_INTERNAL: frozenset[str] = frozenset({
    "no_belief",
})

# CLI strings recognised by ``train_main.py --variant`` (pkg-07 spec 01 §2.1).
CLI_CHOICES: tuple[str, ...] = (
    "hyper", "oracle_only", "infer_only",
    "baseline_input_wide", "baseline_input_deep", "baseline_ma_muzero",
    "no_belief",
    "external_mappo", "external_qmix", "external_ma_muzero_gh",
    "external_mamba", "external_marie", "external_ga",
)


def cli_to_factory_arg(cli: str) -> str:
    """Single source of truth for CLI ↔ factory-arg conversion (pkg-07 spec 01 §5.2).

    Rules (per spec 01 §2.2 + design §3.3 footnote):
      - curriculum-override (``hyper`` / ``oracle_only`` / ``infer_only``):
        no factory; raises :class:`ValueError`.
      - internal-with-shared-backbone (``baseline_*``): strip the ``baseline_`` prefix.
      - internal-ablation-only (``no_belief``):
        pass-through.
      - external (``external_*``): pass-through (preserves prefix for
        factory-side disambiguation).
      - any other string: raises :class:`ValueError` with the full list of
        valid CLI choices.
    """
    if cli in _CURRICULUM_OVERRIDE_VARIANTS:
        raise ValueError(
            f"'{cli}' is a curriculum-override variant, not a factory variant "
            f"(see design.md §3.3 curriculum-override rows)."
        )
    if cli in _CLI_TO_FACTORY_INTERNAL_PREFIX:
        return cli[len("baseline_"):]
    if cli in _CLI_BARE_INTERNAL:
        return cli
    if cli.startswith("external_") and cli in EXTERNAL_REGISTRY:
        return cli
    raise ValueError(
        f"Unknown CLI variant {cli!r}. Valid CLI choices: "
        f"{tuple(sorted(_CURRICULUM_OVERRIDE_VARIANTS)) + tuple(sorted(REGISTRY))}."
    )


# ----------------------------------------------------------------------- factory

def create_baseline(cfg: V4Config, variant: str) -> BaselineLike:
    """11-key REGISTRY factory (pkg-07 spec 01 §3).

    Args:
        cfg: ``V4Config`` (typically from ``V4Config.from_preset(...)``).
            Internal variants consume ``cfg.baselines.internal_*`` capacity
            knobs (spec 01 §6.1); external runners consume
            ``cfg.baselines.external_lr_sweep_grid``.
        variant: factory arg — must be one of the 11 REGISTRY keys.
            Curriculum-override strings (``hyper``/``oracle_only``/
            ``infer_only``) are rejected here; route them to
            ``HyperMuZeroModel(cfg)`` directly.

    Returns:
        :data:`BaselineLike` — either a :class:`BaselineModel` (INTERNAL) or
        an :class:`ExternalBaselineRunner` (EXTERNAL).

    Raises:
        ValueError: when ``variant`` is a curriculum-override string, or when
            ``variant`` is not in REGISTRY.
        NotImplementedError: when ``variant`` resolves to a stub
            (``external_marie`` / ``external_ga``, permanent; or
            ``external_mamba`` whose sourcing failed, design D8).
    """
    if variant == "hyper":
        raise ValueError(
            "'hyper' uses HyperMuZeroModel directly, not this factory. "
            "Call HyperMuZeroModel(cfg) instead (pkg-07 spec 01 §2.1 / design §3.3)."
        )
    if variant in _CURRICULUM_OVERRIDE_VARIANTS:
        raise ValueError(
            f"'{variant}' uses HyperMuZeroModel(cfg) + curriculum_stage_1_end_frac "
            "override, not this factory (pkg-07 design §3.3 curriculum-override row)."
        )
    if variant in INTERNAL_REGISTRY:
        return INTERNAL_REGISTRY[variant](cfg)
    if variant in EXTERNAL_REGISTRY:
        # MAMBA / MARIE / GA stubs raise NotImplementedError from __init__ —
        # the exception propagates naturally.
        return EXTERNAL_REGISTRY[variant](cfg)
    raise ValueError(
        f"Unknown baseline variant {variant!r}. "
        f"Valid factory args: {sorted(REGISTRY)}. "
        f"(For 'hyper'/'oracle_only'/'infer_only' use HyperMuZeroModel + curriculum override.)"
    )


def create_baseline_model(cfg: V4Config, variant: str) -> BaselineLike:
    """DEPRECATED pkg-06 alias for :func:`create_baseline`.

    Removed in the next release. Emits ``DeprecationWarning`` on every call
    (pkg-07 spec 01 §3.2 declaration 1).
    """
    warnings.warn(
        "create_baseline_model is renamed to create_baseline (pkg-07 spec 01). "
        "This alias will be removed in the next release.",
        DeprecationWarning, stacklevel=2,
    )
    return create_baseline(cfg, variant)


# Forward-import of the base class for the BaselineLike Union. Done at
# module bottom so the union type alias resolves at runtime when
# ``create_baseline`` is called (pkg-07 spec 03 §10.1).
from hyper_mve.baselines.internal.base import BaselineModel  # noqa: E402,F401  (re-export)


__all__ = [
    "BaselineLike",
    "BaselineModel",
    "ExternalBaselineRunner",
    "INTERNAL_REGISTRY",
    "EXTERNAL_REGISTRY",
    "REGISTRY",
    "CLI_CHOICES",
    "cli_to_factory_arg",
    "create_baseline",
    "create_baseline_model",
]
