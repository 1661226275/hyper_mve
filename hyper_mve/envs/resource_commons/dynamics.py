"""Resource dynamics — formulas 3.1-3.4 (Ch3.3 + Pkg-02 spec 02).

Pure functions. All physical constants come in via keyword arguments — there
are no module-level defaults — so the caller (``env.step``) is responsible
for sourcing them from :class:`EnvConfig`. This matches the Pkg-02 plan's
"no Pkg-01 constants drift" rule.
"""
from __future__ import annotations

import numpy as np

from .state import ResourceCommonsState


# ---------------------------------------------------------------------------
# Formula 3.4 — context-modulated regen rate
# ---------------------------------------------------------------------------

def compute_alpha(c_t: float, alpha_min: float, alpha_max: float) -> float:
    """``α(c_t) = α_min + (α_max - α_min) · c_t``  (Ch3.3 formula 3.4)."""
    if not (0.0 <= c_t <= 1.0):
        raise AssertionError(f"c_t={c_t} out of [0, 1]")
    return alpha_min + (alpha_max - alpha_min) * c_t


# ---------------------------------------------------------------------------
# Formula 3.3 — neighbor influence factor
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid (clips the exponent to ±50)."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def compute_neighbor_factor(
    resource_stocks: np.ndarray,
    resource_positions: np.ndarray,
    k: int,
    *,
    q_max: float,
    kappa_f: float,
    theta_f: float,
    d_nbr: int,
) -> float:
    """``f(neighbors of k) = σ(κ_f · mean_q_norm − θ_f)``  (Ch3.3 formula 3.3).

    Chebyshev distance ≤ ``d_nbr`` defines the neighborhood; ``k`` itself is
    excluded. With no neighbors the factor falls back to ``σ(-θ_f)`` (low but
    nonzero — solitary hotspots regrow slowly rather than not at all).
    """
    pos_k = resource_positions[k]
    distances = np.max(np.abs(resource_positions - pos_k), axis=-1)
    neighbor_mask = (distances <= d_nbr) & (distances > 0)

    if not neighbor_mask.any():
        return float(_sigmoid(-theta_f))

    mean_q_norm = float(np.mean(resource_stocks[neighbor_mask] / q_max))
    return float(_sigmoid(kappa_f * mean_q_norm - theta_f))


# ---------------------------------------------------------------------------
# Formula 3.1 — logistic regrowth step (mutates state.resource_stocks)
# ---------------------------------------------------------------------------

def step_dynamics(
    state: ResourceCommonsState,
    harvests_per_resource: np.ndarray,
    *,
    alpha_min: float,
    alpha_max: float,
    q_max: float,
    kappa_f: float,
    theta_f: float,
    d_nbr: int,
) -> None:
    """In-place: ``q_{k,t+1} = clip(q_{k,t} − h_k + α(c) · f · (Q_max − q_{k,t}), 0, Q_max)``.

    Neighbor factor ``f`` is computed against the **pre-update** stocks for
    every ``k``, so we snapshot to ``stocks_prev`` and write a fresh array.
    """
    K = state.resource_stocks.shape[0]
    alpha_c = compute_alpha(state.c_t, alpha_min, alpha_max)
    stocks_prev = state.resource_stocks
    new_stocks = stocks_prev.copy()

    for k in range(K):
        f_k = compute_neighbor_factor(
            stocks_prev, state.resource_positions, k,
            q_max=q_max, kappa_f=kappa_f, theta_f=theta_f, d_nbr=d_nbr,
        )
        regrowth = alpha_c * f_k * (q_max - stocks_prev[k])
        new_stocks[k] = stocks_prev[k] - harvests_per_resource[k] + regrowth

    np.clip(new_stocks, 0.0, q_max, out=new_stocks)
    state.resource_stocks[:] = new_stocks


# ---------------------------------------------------------------------------
# Formula 3.2 — fair-share harvest
# ---------------------------------------------------------------------------

def fair_share_harvest(
    state: ResourceCommonsState,
    agent_positions: np.ndarray,
    harvest_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``u_{i,k} = min(η_i, q_k / |H_k|)`` for ``i ∈ H_k`` (Ch3.3 formula 3.2).

    Returns:
        ``(harvests_per_agent (N,), harvests_per_resource (K,))`` both float32.
        Both arrays are fresh allocations; the env caches them on the side.
    """
    N = agent_positions.shape[0]
    K = state.resource_stocks.shape[0]
    harvests_per_agent = np.zeros(N, dtype=np.float32)
    harvests_per_resource = np.zeros(K, dtype=np.float32)

    for k in range(K):
        on_cell = np.all(agent_positions == state.resource_positions[k], axis=-1)
        H_k = np.where(on_cell & harvest_mask)[0]
        if H_k.size == 0:
            continue
        share = float(state.resource_stocks[k]) / float(H_k.size)
        for i in H_k:
            eta_i = float(state.agent_caps[int(i)].eta)
            u_i = min(eta_i, share)
            harvests_per_agent[i] = u_i
            harvests_per_resource[k] += u_i

    return harvests_per_agent, harvests_per_resource
