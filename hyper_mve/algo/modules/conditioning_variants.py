"""Conditioning-swap ablation arms for the DualHyperNetwork (phase 7).

Direction-1 controlled mechanism comparison: same ctx_aug input, same
``forward_subjective(ctx) -> (theta_rew, theta_val)`` contract and the same
grouped-RMS output normalization (via the hypernet's own
``normalize_generated_output``), but the θ-GENERATION mechanism is swapped:

* ``MoERouterConditioner`` (arm ``moe_router``) — top-k routing over K
  learnable parameter-vector experts (the M3W-style conditioning mechanism,
  here at parameter level on the same ctx).
* ``FiLMConditioner`` (arm ``film``) — a single learnable base θ modulated
  per output group by ctx-generated scale/shift (parameter-space FiLM).

Both expose ``share_subjective_trunk = True`` so the subjective model's
per-regime θ_val path (Bayes leaves) routes through ``forward_subjective``,
and both mirror the DualHyperNetwork detach semantics: the θ_val pathway
detaches the ctx-derived modulation, so the value loss does not train the
conditioning trunk.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .hyper_network import normalize_generated_output


class MoERouterConditioner(nn.Module):
    share_subjective_trunk = True

    def __init__(
        self,
        ctx_aug_dim: int,
        rew_param_count: int,
        pred_param_count: int,
        *,
        n_experts: int = 4,
        top_k: int = 2,
        rew_output_scale_init: float = 0.1,
        pred_output_scale_init: float = 0.01,
        rew_output_groups=None,
        pred_output_groups=None,
    ) -> None:
        super().__init__()
        assert top_k <= n_experts
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.rew_output_groups = rew_output_groups
        self.pred_output_groups = pred_output_groups

        self.router = nn.Linear(ctx_aug_dim, n_experts)
        self.rew_experts = nn.Parameter(
            torch.randn(n_experts, rew_param_count) * 0.02)
        self.pred_experts = nn.Parameter(
            torch.randn(n_experts, pred_param_count) * 0.02)
        self.rew_output_scale = nn.Parameter(
            torch.tensor(float(rew_output_scale_init)))
        self.pred_output_scale = nn.Parameter(
            torch.tensor(float(pred_output_scale_init)))

    def _gates(self, ctx: torch.Tensor) -> torch.Tensor:
        logits = self.router(ctx)                              # (B, K)
        top_v, top_i = logits.topk(self.top_k, dim=-1)
        # softmax is on autocast's float32 list while `logits` may be half
        # under the fork's autocast() — align dtypes before scatter.
        sparse = torch.softmax(top_v, dim=-1).to(logits.dtype)
        gates = torch.zeros_like(logits)
        return gates.scatter(-1, top_i, sparse)

    def forward_subjective(self, ctx_aug: torch.Tensor):
        gates = self._gates(ctx_aug)
        theta_rew = normalize_generated_output(
            gates @ self.rew_experts, self.rew_output_scale,
            True, self.rew_output_groups)
        # value pathway: detach the routing (conditioning trunk untrained
        # by the value loss — DualHyperNetwork detach_pred_context analogue)
        theta_val = normalize_generated_output(
            gates.detach() @ self.pred_experts, self.pred_output_scale,
            True, self.pred_output_groups)
        return theta_rew, theta_val


class FiLMConditioner(nn.Module):
    share_subjective_trunk = True

    def __init__(
        self,
        ctx_aug_dim: int,
        rew_param_count: int,
        pred_param_count: int,
        *,
        hidden_dim: int = 128,
        rew_output_scale_init: float = 0.1,
        pred_output_scale_init: float = 0.01,
        rew_output_groups=None,
        pred_output_groups=None,
    ) -> None:
        super().__init__()
        self.rew_output_groups = rew_output_groups
        self.pred_output_groups = pred_output_groups
        self._rew_groups = (list(rew_output_groups) if rew_output_groups
                            else [int(rew_param_count)])
        self._pred_groups = (list(pred_output_groups) if pred_output_groups
                             else [int(pred_param_count)])

        self.rew_base = nn.Parameter(torch.randn(int(rew_param_count)))
        self.pred_base = nn.Parameter(torch.randn(int(pred_param_count)))
        self.rew_output_scale = nn.Parameter(
            torch.tensor(float(rew_output_scale_init)))
        self.pred_output_scale = nn.Parameter(
            torch.tensor(float(pred_output_scale_init)))
        n_seg = len(self._rew_groups) + len(self._pred_groups)
        self.film = nn.Sequential(
            nn.Linear(ctx_aug_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * n_seg),
        )
        # start at identity modulation (γ=0, β=0)
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

    @staticmethod
    def _modulate(base_norm, gamma, beta, groups, scale):
        """θ_seg = (1 + γ_seg)·base_seg + β_seg·scale, per output group."""
        segs, offset = [], 0
        for j, gsize in enumerate(groups):
            seg = base_norm[:, offset:offset + gsize]
            segs.append(seg * (1.0 + gamma[:, j:j + 1])
                        + beta[:, j:j + 1] * scale)
            offset += gsize
        return torch.cat(segs, dim=-1)

    def forward_subjective(self, ctx_aug: torch.Tensor):
        B = ctx_aug.shape[0]
        gb = self.film(ctx_aug)                                # (B, 2·n_seg)
        n_r, n_p = len(self._rew_groups), len(self._pred_groups)
        gamma_r, beta_r = gb[:, :n_r], gb[:, n_r:2 * n_r]
        gamma_p, beta_p = (gb[:, 2 * n_r:2 * n_r + n_p],
                           gb[:, 2 * n_r + n_p:])

        rew_base = normalize_generated_output(
            self.rew_base.unsqueeze(0).expand(B, -1), self.rew_output_scale,
            True, self.rew_output_groups)
        theta_rew = self._modulate(rew_base, gamma_r, beta_r,
                                   self._rew_groups, self.rew_output_scale)

        pred_base = normalize_generated_output(
            self.pred_base.unsqueeze(0).expand(B, -1), self.pred_output_scale,
            True, self.pred_output_groups)
        # value pathway: detach the ctx modulation (see MoE variant note)
        theta_val = self._modulate(pred_base, gamma_p.detach(),
                                   beta_p.detach(), self._pred_groups,
                                   self.pred_output_scale)
        return theta_rew, theta_val


def build_conditioner(kind: str, **kwargs) -> nn.Module:
    """Factory keyed by the ``--conditioning`` fork flag / ablation arm."""
    if kind == "moe_router":
        return MoERouterConditioner(**kwargs)
    if kind == "film":
        return FiLMConditioner(**kwargs)
    raise ValueError(f"unknown conditioning variant {kind!r}")
