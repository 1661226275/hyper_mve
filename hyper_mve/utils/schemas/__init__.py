"""Public schema types shared across all v5 packages (Pkg-01/09 output).

Import from this module — sub-module paths are not part of the public API:

    >>> from hyper_mve.utils.schemas import (
    ...     RegimeFamily,
    ...     RelationObservationLayout,
    ...     TimeStepRecord,
    ...     compute_relational_rewards,
    ... )
"""
from __future__ import annotations

from . import _constants
from .buffer_record import TimeStepRecord
from .observation import (
    ObservationLayout,
    RelationObservationLayout,
    pad_neighbor_block,
    pad_resource_block,
    slice_relation_block,
)
from .relation import (
    Regime,
    RegimeFamily,
    build_g2,
    build_g4,
    build_g4_ext,
    compute_relational_rewards,
    get_regime_family,
    sample_initial_regime,
    step_regime,
)

__all__ = [
    # core schemas
    "TimeStepRecord",
    # v5 relationship regimes (Pkg-09)
    "Regime",
    "RegimeFamily",
    "RelationObservationLayout",
    "ObservationLayout",            # Stage-6 alias of RelationObservationLayout
    "slice_relation_block",
    "build_g2",
    "build_g4",
    "build_g4_ext",
    "compute_relational_rewards",
    "get_regime_family",
    "sample_initial_regime",
    "step_regime",
    # helpers
    "pad_neighbor_block",
    "pad_resource_block",
    # constants namespace
    "_constants",
]
