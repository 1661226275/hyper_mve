"""BeliefEncoder — v5 belief-pathway projection (Pkg-09).

Projects BeliefNet's regime posterior ``ĝ_i ∈ Δ^{|G|}`` to
``belief_vec ∈ ℝ^{d_belief=32}``, symmetric with RoleEncoder as the second
subjective sub-encoder.

v5 simplification (vs v4): the (ĉ, pooled ẑ) 16+16 split is gone — the
regime IS the joint object, so a single ``Linear(|G|, 32) → ReLU`` projection
suffices; no permutation-invariant pooling, no per-opponent expansion.

No internal LayerNorm (P7: LN responsibility stays in TriContextEncoder).
Gradient gating (Ch4.6.5) detaches the raw ``g_hat`` upstream in the model;
this projection itself keeps training from the main loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from hyper_mve.configs import EnvConfig, ModelConfig
from hyper_mve.schemas import get_regime_family


class BeliefEncoder(nn.Module):
    """v5 belief pathway: ``ĝ (B, N, |G|) -> belief_vec (B, N, 32)``."""

    def __init__(self, env_cfg: EnvConfig, model_cfg: ModelConfig):
        super().__init__()
        self.env_cfg = env_cfg
        self.model_cfg = model_cfg

        self.d_belief = model_cfg.d_belief                      # 32
        self.n_regimes = get_regime_family(env_cfg).size        # |G|

        self.proj_g = nn.Sequential(
            nn.Linear(self.n_regimes, self.d_belief),
            nn.ReLU(),
        )

    def forward(
        self,
        g_hat: torch.Tensor,                # (B, N, |G|) softmax regime posterior
    ) -> torch.Tensor:                      # (B, N, 32) belief_vec
        B, N, G = g_hat.shape
        assert G == self.n_regimes, (
            f"g_hat last dim {G} != |G|={self.n_regimes}"
        )
        return self.proj_g(g_hat)
