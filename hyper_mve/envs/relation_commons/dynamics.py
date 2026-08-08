"""v5 resource dynamics (Pkg-09) — constant-rate regrowth + fair-share harvest.

Adapted from the v4 ``resource_commons.dynamics``. Removed: the
context-modulated rate ``α(c_t)`` (v5 uses the constant ``EnvConfig.alpha``)
and the O(K²) neighbor-influence factor. Harvest keeps the fair-share rule
with a **homogeneous** per-step cap ``η = 1.0`` (capability heterogeneity is
gone in v5).

Pure functions over plain arrays; physical constants come in via keyword
arguments sourced from :class:`EnvConfig` by the caller (``env.step``).
"""
from __future__ import annotations

import numpy as np

ETA: float = 1.0     # homogeneous per-agent per-step harvest cap (v5)


def step_dynamics(
    resource_stocks: np.ndarray,
    harvests_per_resource: np.ndarray,
    *,
    alpha: float,
    q_max: float,
    law: str = "constant",
) -> None:
    """In-place resource update under one of two regrowth laws.

    ``constant`` (v5 default)
        ``q ← clip(q − h + α·(Q_max − q), 0, Q_max)``. Regrowth is **fastest
        when the cell is empty** and zero when it is full, so stripping a cell
        costs nothing and holding stock wastes regeneration. Restraint has
        *negative* option value — there is no commons dilemma, and the optimal
        harvest threshold is 0 in every regime.

    ``logistic`` (v6)
        ``q ← clip(q + α·q·(1 − q/Q_max) − h, 0, Q_max)``. Stock left in the
        ground compounds, so restraint pays and "will the other agent leave
        what I leave?" becomes a real question. Note ``q = 0`` is **absorbing**:
        a fully stripped cell never recovers. That is the dilemma, not a bug.

    See ``results/analysis/regime_knowledge_ceiling.md`` for why this became a
    choice rather than a constant.
    """
    if law == "constant":
        regrowth = alpha * (q_max - resource_stocks)
    elif law == "logistic":
        regrowth = alpha * resource_stocks * (1.0 - resource_stocks / q_max)
    else:
        raise ValueError(f"Unknown regrowth law: {law!r} (valid: constant, logistic)")
    resource_stocks += regrowth - harvests_per_resource
    np.clip(resource_stocks, 0.0, q_max, out=resource_stocks)


def fair_share_harvest(
    resource_positions: np.ndarray,
    resource_stocks: np.ndarray,
    agent_positions: np.ndarray,
    harvest_mask: np.ndarray,
    *,
    eta: float = ETA,
) -> tuple[np.ndarray, np.ndarray]:
    """``u_{i,k} = min(η, q_k / |H_k|)`` for ``i ∈ H_k`` (v4 formula 3.2, η homogeneous).

    Uses **pre-update** stocks, matching the v4 step ordering.

    Returns:
        ``(harvests_per_agent (N,), harvests_per_resource (K,))`` both float32,
        fresh allocations.
    """
    N = agent_positions.shape[0]
    K = resource_stocks.shape[0]
    harvests_per_agent = np.zeros(N, dtype=np.float32)
    harvests_per_resource = np.zeros(K, dtype=np.float32)

    for k in range(K):
        on_cell = np.all(agent_positions == resource_positions[k], axis=-1)
        H_k = np.where(on_cell & harvest_mask)[0]
        if H_k.size == 0:
            continue
        share = float(resource_stocks[k]) / float(H_k.size)
        u = min(eta, share)
        harvests_per_agent[H_k] = u
        harvests_per_resource[k] = u * H_k.size

    return harvests_per_agent, harvests_per_resource
