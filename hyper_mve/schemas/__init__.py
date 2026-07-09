"""Public schema types shared across all v4 packages (Pkg-01 output).

Import from this module — sub-module paths are not part of the public API:

    >>> from hyper_mve.schemas import (
    ...     AgentType,
    ...     CapabilityVector,
    ...     ObservationLayout,
    ...     TimeStepRecord,
    ...     ContextSchema,
    ... )
"""
from __future__ import annotations

from . import _constants
from .agent_type import (
    AgentType,
    count_in_assignment,
    from_index,
    from_str,
    one_hot,
    to_long_tensor,
)
from .buffer_record import TimeStepRecord
from .capability import (
    CapabilityVector,
    sample_default,
    sample_n,
    to_batch_tensor,
)
from .context import ContextSchema
from .observation import (
    ObservationBlockSpec,
    ObservationLayout,
    RelationObservationLayout,
    pad_neighbor_block,
    pad_resource_block,
    slice_block,
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
    "AgentType",
    "CapabilityVector",
    "ContextSchema",
    "ObservationBlockSpec",
    "ObservationLayout",
    "TimeStepRecord",
    # v5 relationship regimes (Pkg-09)
    "Regime",
    "RegimeFamily",
    "RelationObservationLayout",
    "slice_relation_block",
    "build_g2",
    "build_g4",
    "build_g4_ext",
    "compute_relational_rewards",
    "get_regime_family",
    "sample_initial_regime",
    "step_regime",
    # helpers
    "count_in_assignment",
    "from_index",
    "from_str",
    "one_hot",
    "to_long_tensor",
    "sample_default",
    "sample_n",
    "to_batch_tensor",
    "pad_neighbor_block",
    "pad_resource_block",
    "slice_block",
    # constants namespace
    "_constants",
]
