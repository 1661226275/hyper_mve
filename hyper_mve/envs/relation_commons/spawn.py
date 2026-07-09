"""Uniform resource spawning (Pkg-09).

v5 drops the v4 patchy-Gaussian / hotspot machinery: resource cells are a
uniform no-replacement draw over the grid, so the physical layout carries no
structure beyond what the relationship regime induces.
"""
from __future__ import annotations

import numpy as np


def spawn_uniform_resources(
    K: int,
    L: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample ``K`` distinct resource cells uniformly on the ``L×L`` grid.

    Returns:
        ``(K, 2) int32`` positions (x, y), all distinct.
    """
    if K > L * L:
        raise ValueError(f"K={K} exceeds grid capacity L*L={L * L}")
    flat = rng.choice(L * L, size=K, replace=False)
    return np.stack([flat % L, flat // L], axis=1).astype(np.int32)
