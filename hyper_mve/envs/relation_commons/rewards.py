"""v5 relational reward — thin wrapper over the citable formula (Pkg-09).

The formula itself lives in :func:`hyper_mve.utils.schemas.relation.compute_relational_rewards`
(single source of truth for tests / docs / thesis Ch3):

    ``R_i = (u_i + Σ_{j≠i} w_ij·u_j) / (1 + Σ_{j≠i} |w_ij|) - ε·1[moved_i]``
"""
from __future__ import annotations

from hyper_mve.utils.schemas.relation import compute_relational_rewards

__all__ = ["compute_relational_rewards"]
