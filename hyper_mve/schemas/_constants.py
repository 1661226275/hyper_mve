"""Global numeric constants shared across v4 schemas and configs.

Single source of truth for:

- Fehr-Schmidt / resource dynamics constants (Ch3.3, 3.5).
- v4 architecture hard-constraint dimensions (Ch4.2.2, 4.2.3, 4.2.4).

Downstream packages (Pkg-02..08) MUST import from here rather than
redefining the values inline. Any change requires updating the relevant
chapter in the design docs first (Ch3 for env constants, Ch4 for model
dimensions), then this file, then propagating via unit tests.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Resource dynamics (Ch3.3)
# ---------------------------------------------------------------------------
Q_MAX: float = 10.0                 # maximum stock per resource cell
ALPHA_MIN: float = 0.02             # lower bound of regen rate α(c_t)
ALPHA_MAX: float = 0.20             # upper bound of regen rate α(c_t)

# ---------------------------------------------------------------------------
# Fehr-Schmidt preferences (Ch3.5)
# ---------------------------------------------------------------------------
KAPPA: float = 0.5                  # φ(c) = κ (1 - 2c)
LAMBDA_DISADV: float = 2.0          # disadvantageous-inequity weight
LAMBDA_ADV: float = 0.6             # advantageous-inequity weight
EPSILON_MOVE: float = 0.01          # small per-move cost

# ---------------------------------------------------------------------------
# Capability sampling ranges (Ch3.6) — closed intervals
# ---------------------------------------------------------------------------
ETA_RANGE: tuple[float, float] = (0.5, 1.5)
PHI_FOV_RANGE: tuple[float, float] = (2.0, 4.0)
NU_RANGE: tuple[float, float] = (0.8, 1.0)
ZETA_RANGE: tuple[float, float] = (10.0, 30.0)

# ---------------------------------------------------------------------------
# v4 architecture dimensions (Ch4.2.2 + 4.2.3 + 4.2.4 hard constraints)
# ---------------------------------------------------------------------------
# role = Concat[id, type, cap] exactly fills d_role with no padding.
D_ID_EMB: int = 8
D_TYPE_EMB: int = 8                 # v4 critical: 8 (not 4)
D_CAP_EMB: int = 16
D_ROLE: int = D_ID_EMB + D_TYPE_EMB + D_CAP_EMB    # = 32 (Ch4.2.2)

# Belief subcomponents: ĉ scalar → Proj(16), pooled ẑ → Proj(16), concat → 32
D_C_CTX: int = 16                   # c_ctx path (Ch4.2.1)
D_BELIEF_PROJ: int = 16             # per-subcomponent projection (Ch4.2.3)
D_BELIEF: int = 2 * D_BELIEF_PROJ   # = 32 (Ch4.2.4 d_b^{proj·2})

# Total per-agent conditioning vector (Ch4.2.4)
D_CTX_AUG: int = D_C_CTX + D_ROLE + D_BELIEF        # = 80

# ---------------------------------------------------------------------------
# AgentType enum values (kept here to avoid circular imports with schemas)
# ---------------------------------------------------------------------------
AGENT_TYPE_ALPHA: int = 0
AGENT_TYPE_BETA: int = 1
NUM_AGENT_TYPES: int = 2
