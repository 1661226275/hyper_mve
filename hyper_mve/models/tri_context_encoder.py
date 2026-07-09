"""SubjectiveContextEncoder (class name TriContextEncoder for continuity) — v5.

v5 change (Pkg-09): the objective c_ctx pathway is deleted with c_t. The
conditioning vector is now two subjective pathways:

    ctx_i = Concat[role_i (32), belief_i (32)] ∈ ℝ^{d_ctx_aug=64}
    slices: [0:32] role, [32:64] belief.

Responsibilities (P5 unchanged): call the two sub-encoders + per-path LN +
concat; projection logic lives in RoleEncoder / BeliefEncoder. Per-path
LayerNorm (D7 + P7): sub-encoders contain no LN.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from hyper_mve.configs import EnvConfig, ModelConfig
from hyper_mve.models.role_encoder import RoleEncoder
from hyper_mve.models.belief_encoder import BeliefEncoder


class TriContextEncoder(nn.Module):
    """v5 subjective conditioning encoder.

    Output ``ctx_i ∈ ℝ^64 = Concat[role (32), belief (32)]``.
    (The "Tri" name is kept for import/checkpoint continuity; the objective
    c path was removed in v5 — see Pkg-09 design.)
    """

    def __init__(self, env_cfg: EnvConfig, model_cfg: ModelConfig):
        super().__init__()
        self.env_cfg = env_cfg
        self.model_cfg = model_cfg

        self.N = env_cfg.N
        self.d_role = model_cfg.d_role                  # 32
        self.d_belief = model_cfg.d_belief              # 32

        self.role_encoder = RoleEncoder(
            N=self.N,
            d_id_emb=model_cfg.d_id_emb,        # 8
            d_row_emb=model_cfg.d_row_emb,      # 24
        )
        self.belief_encoder = BeliefEncoder(env_cfg, model_cfg)

        # Per-path LayerNorm (D7; P7: LN responsibility concentrated here)
        self.ln_role = nn.LayerNorm(self.d_role)
        self.ln_belief = nn.LayerNorm(self.d_belief)

    @property
    def d_ctx_aug(self) -> int:
        """ctx_i total dim = 32 + 32 = 64 (v5)."""
        return self.d_role + self.d_belief

    def forward(
        self,
        agent_ids: torch.Tensor,            # (B, N) int64
        rows: torch.Tensor,                 # (B, N, N-1) float32 own rows w_i·
        g_hat: torch.Tensor,                # (B, N, |G|) regime posterior
    ) -> torch.Tensor:                      # (B, N, 64) float32
        """Combine the two subjective pathways → ctx_i.

        Pathway semantics:
            role (32):   per-agent Self-Info (own id + own row w_i·)
            belief (32): per-agent inferred regime posterior ĝ_i
        """
        B, N = agent_ids.shape
        assert rows.shape[:2] == (B, N)
        assert g_hat.shape[:2] == (B, N)

        role = self.role_encoder(agent_ids, rows)         # (B, N, 32)
        role = self.ln_role(role)

        belief_vec = self.belief_encoder(g_hat)           # (B, N, 32)
        belief_vec = self.ln_belief(belief_vec)

        return torch.cat([role, belief_vec], dim=-1)      # (B, N, 64)
