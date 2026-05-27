"""Fehr-Schmidt reward — formulas 3.5-3.10 (Ch3.5 + Pkg-02 spec 03).

Pure functions, no module-level constants — coefficients are passed in by
the caller (``env.step``) from :class:`EnvConfig`.

Two key correctness contracts (Pkg-02 hard-gate tests):

* **Table 3.5.4** — type β φ·ψ values across the four (c, Δ) quadrants
  match {-0.3, -1.0, +0.3, +1.0} exactly (numpy).
* **Ch4.1.1 partial-derivative table** — autograd-extracted
  ∂R^β/∂u_i in the four quadrants matches {0.7, 2.0, 1.3, 0.0} exactly
  (torch). The torch variant lives below for testing.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from hyper_mve.schemas import AgentType

if TYPE_CHECKING:
    import torch


# ---------------------------------------------------------------------------
# Numpy core (production path)
# ---------------------------------------------------------------------------

def compute_phi(c_t: float, kappa: float) -> float:
    """``φ(c) = κ (1 - 2c)``  (Ch3.5 formula 3.7).

    ``c=0`` ⇒ ``+κ`` (lean season amplifies inequity aversion).
    ``c=0.5`` ⇒ ``0`` (β degenerates to selfish).
    ``c=1`` ⇒ ``-κ`` (abundance flips the sign — altruistic).
    """
    return kappa * (1.0 - 2.0 * c_t)


def compute_psi(
    delta: float,
    lambda_disadv: float,
    lambda_adv: float,
) -> float:
    """``ψ(Δ) = -[λ_disadv · max(0,-Δ) + λ_adv · max(0,Δ)]``  (formula 3.8).

    ``ψ ≤ 0`` always; equals 0 at Δ=0.
    """
    return -(lambda_disadv * max(0.0, -delta) + lambda_adv * max(0.0, delta))


def compute_psi_vec(
    deltas: np.ndarray,
    lambda_disadv: float,
    lambda_adv: float,
) -> np.ndarray:
    """Vectorised :func:`compute_psi`. Returns ``float32`` array."""
    disadv = lambda_disadv * np.maximum(0.0, -deltas)
    adv = lambda_adv * np.maximum(0.0, deltas)
    return (-(disadv + adv)).astype(np.float32)


def compute_delta(harvests: np.ndarray) -> np.ndarray:
    """``Δ_i = u_i - mean_{j≠i} u_j``  (formula 3.9), vectorised over all N agents.

    Computed for every agent regardless of type — see Pkg-02 Decision D3
    (buffer-schema consistency; ``TimeStepRecord.delta`` has shape ``(N,)``
    even when some agents are type α).
    """
    N = harvests.shape[0]
    total = float(harvests.sum())
    denom = max(N - 1, 1)
    mean_others = (total - harvests) / denom
    return (harvests - mean_others).astype(np.float32)


def compute_rewards(
    *,
    harvests: np.ndarray,
    moved_mask: np.ndarray,
    agent_types: np.ndarray,
    c_t: float,
    kappa: float,
    lambda_disadv: float,
    lambda_adv: float,
    epsilon_move: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-agent type-aware reward (formulas 3.5 / 3.6 / 3.10).

    ``R_i = u_i - ε · 1[moved_i]  +  1[τ_i = β] · φ(c) · ψ(Δ_i)``.

    Args:
        harvests:    ``(N,)`` per-agent u_i.
        moved_mask:  ``(N,) bool`` — physical move *intent* (see Pkg-02
                     plan deviation #6: cost is independent of execution
                     success).
        agent_types: ``(N,) int8`` (AgentType.value).
        c_t:         shared context ∈ [0, 1].
        kappa, lambda_disadv, lambda_adv, epsilon_move: Ch3.5 scalars.

    Returns:
        ``(rewards (N,) float32, deltas (N,) float32)``.
    """
    deltas = compute_delta(harvests)
    phi_c = compute_phi(c_t, kappa)
    psi = compute_psi_vec(deltas, lambda_disadv, lambda_adv)

    is_beta = (agent_types == int(AgentType.BETA)).astype(np.float32)
    physical = harvests.astype(np.float32) - epsilon_move * moved_mask.astype(np.float32)
    preference = phi_c * psi * is_beta
    rewards = (physical + preference).astype(np.float32)
    return rewards, deltas


# ---------------------------------------------------------------------------
# Torch mirror (autograd-friendly; test path only)
# ---------------------------------------------------------------------------

def compute_rewards_torch(
    harvests: "torch.Tensor",
    moved_mask: "torch.Tensor",
    agent_types: "torch.Tensor",
    c_t: float,
    *,
    kappa: float,
    lambda_disadv: float,
    lambda_adv: float,
    epsilon_move: float,
) -> "torch.Tensor":
    """PyTorch version of :func:`compute_rewards` — autograd-compatible.

    Used exclusively by ``test_rewards.py`` to verify the Ch4.1.1
    ∂R^β/∂u_i partial-derivative table.
    """
    import torch  # local import — torch is not a runtime dep of env.step

    N = harvests.shape[0]
    total = harvests.sum()
    denom = max(N - 1, 1)
    mean_others = (total - harvests) / denom
    deltas = harvests - mean_others

    phi_c = kappa * (1.0 - 2.0 * c_t)
    psi = -(
        lambda_disadv * torch.clamp(-deltas, min=0.0)
        + lambda_adv * torch.clamp(deltas, min=0.0)
    )

    is_beta = (agent_types == int(AgentType.BETA)).to(harvests.dtype)
    physical = harvests - epsilon_move * moved_mask.to(harvests.dtype)
    preference = phi_c * psi * is_beta
    return physical + preference
