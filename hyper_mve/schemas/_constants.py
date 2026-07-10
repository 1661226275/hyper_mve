"""Global numeric constants shared across v5 schemas and configs.

Single source of truth for the environment scalars and the architecture
hard-constraint dimensions. Downstream packages MUST import from here rather
than redefining the values inline. Any change requires updating the relevant
design doc first (`sdd/pkg-09-dynamic-relations/design.md`), then this file,
then propagating via unit tests.

The v4 blocks (Fehr-Schmidt scalars, α(c_t) range, capability sampling ranges
and CAP_NORM endpoints, D_TYPE_EMB/D_CAP_EMB/D_C_CTX/D_BELIEF_PROJ, AgentType
values) were deleted with the type system in the Stage-6 cleanup.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Resource dynamics (fixed physics)
# ---------------------------------------------------------------------------
Q_MAX: float = 10.0                 # maximum stock per resource cell
EPSILON_MOVE: float = 0.01          # per-move cost in the relational reward

# ---------------------------------------------------------------------------
# v5 architecture dimensions (Pkg-09 hard constraints)
# role = Concat[id, row] exactly fills d_role with no padding;
# ctx_aug = [role | belief] = 64.
# ---------------------------------------------------------------------------
D_ID_EMB: int = 8
D_ROW_EMB: int = 24                 # own-row w_i· embedding
D_ROLE: int = D_ID_EMB + D_ROW_EMB  # = 32

D_BELIEF: int = 32                  # regime-posterior encoding (Linear(|G|, 32))

D_CTX_AUG: int = D_ROLE + D_BELIEF  # = 64
