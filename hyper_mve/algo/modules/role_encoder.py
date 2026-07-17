"""RoleEncoder — v5 role pathway (Pkg-09; supersedes the v4 type+cap version).

Encodes agent i's (id, own relationship row ``w_i·``) into
``role_i ∈ ℝ^{d_role=32}``:

    role_i = Concat[id_emb_i (8), row_emb_i (24)]   (exact fill, no pad)

v5 key change (vs v4 ``id + type_emb + cap_emb``): the discrete AgentType and
the capability vector are gone; the agent's role IS its diagonal-free row of
the relationship matrix (values in [-1, 1], ordered ascending j skipping
self — the same convention as the observation ``row`` block and BeliefNet's
regime posterior). Row values are already normalized to [-1, 1], so no input
scaling is needed (the v4 CAP_NORM machinery is deleted).

Self-Info strictness (Pkg-09): role_i contains only agent i's OWN row;
others' rows are inferred by BeliefNet's regime head.

P7: no internal LayerNorm (LN responsibility stays in TriContextEncoder).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RoleEncoder(nn.Module):
    """v5 role pathway: ``(id, own row w_i·) -> role_i ∈ ℝ^32``.

    role = Concat[id_emb (8), row_emb (24)].
    """

    def __init__(
        self,
        N: int,
        d_id_emb: int = 8,
        d_row_emb: int = 24,
        row_hidden_dim: int = 16,
    ):
        super().__init__()
        self.N = N
        self.d_id_emb = d_id_emb
        self.d_row_emb = d_row_emb
        self.d_role = d_id_emb + d_row_emb

        assert self.d_role == 32, (
            f"d_role={self.d_role}, expected 32 (8+24). v5 requires exact fill, no pad."
        )

        # id_emb: agent_id ∈ {0, ..., N-1} -> d_id_emb
        self.id_emb = nn.Embedding(N, d_id_emb)

        # row_mlp: own diagonal-free row (N-1,) ∈ [-1, 1]^{N-1} -> d_row_emb
        # (no internal LN, P7)
        self.row_mlp = nn.Sequential(
            nn.Linear(N - 1, row_hidden_dim),
            nn.ReLU(),
            nn.Linear(row_hidden_dim, d_row_emb),
        )

    def forward(
        self,
        agent_ids: torch.Tensor,            # (B, N) int64
        rows: torch.Tensor,                 # (B, N, N-1) float32 own rows w_i·
    ) -> torch.Tensor:                      # (B, N, 32) float32
        """
        Args:
            agent_ids: (B, N) agent indices ∈ {0, ..., N-1}
            rows:      (B, N, N-1) each agent's OWN diagonal-free row
                       (ascending j, skipping self)
        Returns:
            role_i: (B, N, 32) = Concat[id_emb, row_emb]
        """
        B, N = agent_ids.shape
        assert rows.shape == (B, N, self.N - 1), (
            f"rows shape {tuple(rows.shape)} != ({B}, {N}, {self.N - 1})"
        )

        id_vec = self.id_emb(agent_ids)                # (B, N, 8)
        row_vec = self.row_mlp(rows)                   # (B, N, 24)
        return torch.cat([id_vec, row_vec], dim=-1)    # (B, N, 32)
