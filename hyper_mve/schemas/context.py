"""ContextSchema (Ch4.2) — fixed dimensions of the 3-path conditioning vector.

The v4 hypernet consumes ``ctx_i = Concat[c_ctx_i, role_i, belief_i]``, with
each subcomponent dimension hard-pinned in `_constants`:

- ``c_ctx``  (d=16) — global context projection (Ch4.2.1)
- ``role``   (d=32) — Concat[id_emb(8), type_emb(8), cap_emb(16)] (Ch4.2.2)
- ``belief`` (d=32) — Concat[Proj(ĉ)(16), Proj(Pool(ẑ))(16)] (Ch4.2.3/2.4)

This module exposes the constants in a single typed namespace so downstream
packages (Pkg-03 BeliefNet, Pkg-04 Model) reference one place.
"""
from __future__ import annotations

from dataclasses import dataclass

from ._constants import (
    D_BELIEF,
    D_BELIEF_PROJ,
    D_CAP_EMB,
    D_C_CTX,
    D_CTX_AUG,
    D_ID_EMB,
    D_ROLE,
    D_TYPE_EMB,
)


@dataclass(frozen=True)
class ContextSchema:
    """Frozen view of the v4 conditioning vector dimensions.

    The instance fields mirror module-level constants so external code can
    either ``import ContextSchema`` and read ``ContextSchema().d_role`` or
    use the class attributes directly.
    """

    d_c: int = D_C_CTX
    d_role: int = D_ROLE
    d_belief: int = D_BELIEF

    # role internal decomposition
    d_id_emb: int = D_ID_EMB
    d_type_emb: int = D_TYPE_EMB
    d_cap_emb: int = D_CAP_EMB

    # belief internal decomposition
    d_belief_proj: int = D_BELIEF_PROJ

    @property
    def d_ctx_aug(self) -> int:
        """Total conditioning vector dim: ``d_c + d_role + d_belief = 80``."""
        return self.d_c + self.d_role + self.d_belief

    def __post_init__(self) -> None:
        if self.d_id_emb + self.d_type_emb + self.d_cap_emb != self.d_role:
            raise ValueError(
                f"ContextSchema: d_id({self.d_id_emb}) + d_type({self.d_type_emb}) "
                f"+ d_cap({self.d_cap_emb}) must equal d_role({self.d_role}); "
                "Ch4.2.2 requires exact fill, no padding."
            )
        if 2 * self.d_belief_proj != self.d_belief:
            raise ValueError(
                f"ContextSchema: 2 × d_belief_proj({self.d_belief_proj}) "
                f"must equal d_belief({self.d_belief}); Ch4.2.4 hard constraint."
            )
        if self.d_ctx_aug != D_CTX_AUG:
            raise ValueError(
                f"ContextSchema.d_ctx_aug={self.d_ctx_aug} != {D_CTX_AUG}; "
                "constants drift from Ch4.2.4 — update _constants.py first."
            )
