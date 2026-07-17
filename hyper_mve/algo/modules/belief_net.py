"""BeliefNet — v5 GRU trunk + regime head (Pkg-09; supersedes v4 head_c/head_opp).

Encodes each agent's observation history into a hidden state
``b_i^t ∈ ℝ^128``, then infers the hidden relationship regime:

    head_regime -> ĝ_i^t (B, N, |G|) softmax    (oracle-g CE supervised)

v5 key changes (vs v4):
    - ``head_c`` deleted — c_t is gone (fixed physics), nothing to read back.
    - ``head_opp`` (per-opponent 2-class type inference with opp-id embedding
      expansion) replaced by a single |G|-way regime posterior per agent: the
      regime is the joint object; conditioning on the agent's own row comes
      for free through the observation's ``row`` block.

Retained properties:
    - Shared weights, independent hidden states: N agents share one GRU,
      each keeps its own ``b_i^t``.
    - uni-directional (causal — online RL inference).
    - Two APIs: ``step()`` single step (worker online) + ``forward()``
      sequence (trainer K-step unroll).
    - GRU output LayerNorm (Ch4.6.2 defence 2).

Belief gradient gating (Ch4.6.5) is applied in the consuming model
(HyperMuZeroModel / BaselineModel), not here.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from hyper_mve.utils.configs import EnvConfig, ModelConfig
from hyper_mve.utils.schemas import RelationObservationLayout, get_regime_family
from hyper_mve.algo.modules._belief_obs_encoder import BeliefObsEncoder


class BeliefNet(nn.Module):
    """v5 BeliefNet (Pkg-09).

    Architecture:

        obs (B, N, obs_dim)
          -> BeliefObsEncoder -> gru_input (B, N, 64)
          -> GRUCell(64, 128) -> b_i^t (B, N, 128) -> LN -> hidden
          -> head_regime (MLP 128 -> 64 -> |G|) -> softmax -> g_hat (B, N, |G|)
    """

    def __init__(self, env_cfg: EnvConfig, model_cfg: ModelConfig):
        super().__init__()
        self.env_cfg = env_cfg
        self.model_cfg = model_cfg

        self.N = env_cfg.N
        self.n_regimes = get_regime_family(env_cfg).size
        self.obs_dim = RelationObservationLayout.total_dim(env_cfg.N, env_cfg.K)
        self.gru_input_dim = 64                     # internal, not exposed in ModelConfig
        self.gru_hidden = model_cfg.belief_gru_hidden       # 128

        # ====== GRU trunk ======
        self.obs_encoder = BeliefObsEncoder(
            obs_dim=self.obs_dim,
            gru_input_dim=self.gru_input_dim,
            hidden_dim=128,
        )
        self.gru = nn.GRUCell(self.gru_input_dim, self.gru_hidden)
        self.ln_belief = nn.LayerNorm(self.gru_hidden)

        # ====== head_regime: |G|-way posterior (oracle-g CE supervised) ======
        head_hidden = 64
        self.head_regime = nn.Sequential(
            nn.Linear(self.gru_hidden, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, self.n_regimes),
        )
        # softmax applied in forward (probabilities enter the belief pathway).

    def init_hidden(
        self,
        batch_size: int,
        num_agents: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:                      # (B, N, 128) zeros
        """Initialize GRU hidden (independent state per (batch_idx, agent_idx))."""
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(batch_size, num_agents, self.gru_hidden, device=device)

    def _gru_step(
        self,
        obs_t: torch.Tensor,                # (B, N, obs_dim)
        prev_hidden: torch.Tensor,          # (B, N, 128)
    ) -> torch.Tensor:                      # new_hidden (B, N, 128), post-LN
        """Single GRU update (shared by step and forward)."""
        B, N, _ = obs_t.shape
        gru_input = self.obs_encoder(obs_t)             # (B, N, 64)
        gi_flat = gru_input.reshape(B * N, -1)
        h_flat = prev_hidden.reshape(B * N, -1)
        new_h_flat = self.gru(gi_flat, h_flat)
        new_h = new_h_flat.reshape(B, N, -1)
        return self.ln_belief(new_h)                    # Ch4.6.2 defence 2

    def _compute_g_hat(self, hidden: torch.Tensor) -> torch.Tensor:
        """head_regime forward: ``(..., 128) -> (..., |G|)`` softmax posterior."""
        logits = self.head_regime(hidden)
        return torch.softmax(logits, dim=-1)

    def step(
        self,
        obs_t: torch.Tensor,                # (B, N, obs_dim)
        prev_hidden: torch.Tensor,          # (B, N, 128)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-step inference (worker online).

        Returns:
            new_hidden: (B, N, 128)
            g_hat:      (B, N, |G|) softmax regime posterior
        """
        new_hidden = self._gru_step(obs_t, prev_hidden)
        return new_hidden, self._compute_g_hat(new_hidden)

    def forward(
        self,
        obs_seq: torch.Tensor,              # (B, T, N, obs_dim)
        init_hidden: Optional[torch.Tensor] = None,
                                            # (B, N, 128), default zeros
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sequence inference (trainer K-step unroll).

        Returns:
            hidden_seq: (B, T, N, 128)
            g_hat_seq:  (B, T, N, |G|)
        """
        B, T, N, _ = obs_seq.shape
        device = obs_seq.device

        if init_hidden is None:
            init_hidden = self.init_hidden(B, N, device=device)

        hidden = init_hidden
        hidden_list = []
        g_hat_list = []
        for t in range(T):
            hidden = self._gru_step(obs_seq[:, t], hidden)
            hidden_list.append(hidden)
            g_hat_list.append(self._compute_g_hat(hidden))

        return torch.stack(hidden_list, dim=1), torch.stack(g_hat_list, dim=1)

    def get_head_regime_predictions(
        self,
        hidden_seq: torch.Tensor,           # (B, T, N, 128)
    ) -> torch.Tensor:                      # (B, T, N, |G|)
        """Recompute head_regime posteriors from a stored hidden sequence."""
        B, T, N, _ = hidden_seq.shape
        return torch.stack(
            [self._compute_g_hat(hidden_seq[:, t]) for t in range(T)], dim=1,
        )
