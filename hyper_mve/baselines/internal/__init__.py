"""Internal baseline variants (pkg-07 spec 03).

5 model classes (input_wide, input_deep, ma_muzero, no_belief,
no_belief), all implementing the v5 6-API +
stateful + Self-Info strict + BeliefGradGating contracts.
"""
from __future__ import annotations

from .base import BaselineModel
from .input_conditioned import InputDeepBaselineModel, InputWideBaselineModel
from .ma_muzero import MAMuZeroBaselineModel
from .no_belief import NoBeliefBaselineModel

__all__ = [
    "BaselineModel",
    "InputWideBaselineModel",
    "InputDeepBaselineModel",
    "MAMuZeroBaselineModel",
    "NoBeliefBaselineModel",
]
