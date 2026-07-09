"""BeliefNet losses + oracle g construction (v5, Pkg-09).

Two losses (v5; the v4 L_c / L_opp are superseded by L_regime):

    L_regime (CE):    per-agent regime-posterior cross-entropy against the
                      oracle regime id g_true (env info["g_true"]).
    L_div (hinge):    prevents b_i^t collapsing to a constant across the N
                      agents (unchanged from v4 — agents see different rows,
                      so their hidden states must stay distinguishable).

Total: L = λ_regime · L_regime + λ_div · L_div, default (1.0, 0.01).

build_oracle_g_seq: one-hot oracle posterior for the curriculum's Stage-1/2
oracle blend (``g_main = w · onehot(g) + (1 − w) · ĝ``), the direct analog of
the v4 build_oracle_z_seq.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def l_regime(
    g_hat_seq: torch.Tensor,            # (B, T, N, |G|) softmax posteriors
    g_true_seq: torch.Tensor,           # (B, T) int64 oracle regime ids
    mask: Optional[torch.Tensor] = None,
                                        # (B, T) bool, optional
) -> torch.Tensor:                      # scalar (with grad)
    """L_regime CE: ``-(1/(B·T·N)) Σ log ĝ[b,t,i, g_true[b,t]]``.

    The regime is common latent state — all N agents share the same target —
    but each agent's posterior is conditioned on its own history/row, so the
    loss broadcasts g_true over the N dimension (same convention the v4 L_c
    used for c_true).
    """
    B, T, N, G = g_hat_seq.shape
    assert g_true_seq.shape == (B, T), (
        f"g_true_seq shape {tuple(g_true_seq.shape)}, expected (B={B}, T={T}). "
        "g_true is the shared latent regime id — no N dim; broadcast internally."
    )
    eps = 1e-8
    log_probs = torch.log(g_hat_seq + eps)                       # (B, T, N, G)
    labels = g_true_seq.long().unsqueeze(-1).expand(B, T, N)     # (B, T, N)
    nll = -torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    # nll: (B, T, N)

    if mask is not None:
        mask_expanded = mask.unsqueeze(-1).expand(B, T, N).float()
        nll = nll * mask_expanded
        denom = mask_expanded.sum().clamp(min=1.0)
    else:
        denom = float(B * T * N)

    return nll.sum() / denom


def l_div(
    hidden_seq: torch.Tensor,           # (B, T, N, h_dim) GRU hidden state
    target_std: float = 0.1,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """L_div variance hinge: keeps b_i^t from collapsing across the N agents.

    ``loss = max(0, target_std² − Var_i(b_i^t))`` — variance over the N
    dimension per feature, mean over features and steps. Unchanged from v4.
    """
    B, T, N, h_dim = hidden_seq.shape
    target_var = target_std ** 2

    var_per_dim = hidden_seq.var(dim=2, unbiased=False)   # (B, T, h_dim)
    var_avg = var_per_dim.mean(dim=-1)                    # (B, T)
    hinge = (target_var - var_avg).clamp(min=0)           # (B, T)

    if mask is not None:
        hinge = hinge * mask.float()
        denom = mask.float().sum().clamp(min=1.0)
    else:
        denom = float(B * T)

    return hinge.sum() / denom


def belief_loss(
    g_hat_seq: torch.Tensor,            # (B, T, N, |G|) softmax
    hidden_seq: torch.Tensor,           # (B, T, N, 128)
    g_true_seq: torch.Tensor,           # (B, T) int64
    weights: tuple[float, float] = (1.0, 0.01),
                                        # (λ_regime, λ_div)
    div_target_std: float = 0.1,
    mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """v5 BeliefNet total loss.

    Returns:
        total: scalar (with grad), weighted.
        breakdown: dict with "l_regime", "l_div", "total" (detached, logging).
    """
    lam_regime, lam_div = weights

    loss_regime = l_regime(g_hat_seq, g_true_seq, mask=mask)
    loss_div = l_div(hidden_seq, target_std=div_target_std, mask=mask)

    total = lam_regime * loss_regime + lam_div * loss_div

    breakdown = {
        "l_regime": loss_regime.detach(),
        "l_div": loss_div.detach(),
        "total": total.detach(),
    }
    return total, breakdown


def build_oracle_g_seq(
    g_true_seq: torch.Tensor,           # (B, T) int64 oracle regime ids
    N: int,
    n_regimes: int,
) -> torch.Tensor:                      # (B, T, N, |G|) one-hot ∈ Δ^{|G|}
    """One-hot oracle regime posterior for the curriculum oracle blend.

    All agents receive the same one-hot (the regime is shared); shape mirrors
    ``BeliefNet.forward``'s g_hat_seq so the convex blend in
    ``compose_total_loss`` is well-defined.
    """
    B, T = g_true_seq.shape
    onehot = F.one_hot(g_true_seq.long(), num_classes=n_regimes).float()  # (B, T, G)
    return onehot.unsqueeze(2).expand(B, T, N, n_regimes).contiguous()
