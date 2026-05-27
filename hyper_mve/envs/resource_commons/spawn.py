"""Patchy resource spawning — Ch3.3.2 + Pkg-02 spec 06.

Resources cluster around ``M`` hotspots, ``K`` total cells. Sampling is
``rng``-driven so :func:`spawn_patchy_resources(seed=...)` is fully
reproducible.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np


def spawn_patchy_resources(
    K: int,
    M: int,
    L: int,
    sigma_patch: float = 2.0,
    min_hotspot_distance: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate ``M`` hotspots and ``K`` resource points around them.

    Args:
        K: total resource cells.
        M: number of hotspots (``K`` is distributed as evenly as possible).
        L: grid side length.
        sigma_patch: Gaussian std-dev of resource scatter around each hotspot.
        min_hotspot_distance: minimum Euclidean distance between hotspot
            centers. Defaults to ``2 * sigma_patch``.
        rng: numpy ``Generator``; one is created if ``None``.

    Returns:
        ``(hotspot_centers, resource_positions)`` — both ``int32``,
        shapes ``(M, 2)`` and ``(K, 2)``. Resource positions may repeat
        (two cells at the same grid point), see spec 06 §3.3.
    """
    if rng is None:
        rng = np.random.default_rng()
    if min_hotspot_distance is None:
        min_hotspot_distance = 2.0 * sigma_patch

    hotspot_centers = _sample_hotspot_centers(M, L, min_hotspot_distance, rng)
    points_per_hotspot = _distribute_K_over_M(K, M)

    resource_positions = np.zeros((K, 2), dtype=np.int32)
    idx = 0
    for m in range(M):
        center = hotspot_centers[m]
        for _ in range(int(points_per_hotspot[m])):
            resource_positions[idx] = _sample_around_center(center, sigma_patch, L, rng)
            idx += 1
    return hotspot_centers, resource_positions


def _sample_hotspot_centers(
    M: int,
    L: int,
    min_distance: float,
    rng: np.random.Generator,
    max_retries: int = 100,
) -> np.ndarray:
    """Sample ``M`` centers with pairwise distance ≥ ``min_distance``.

    Falls back to the last candidate (and warns) if a slot cannot be placed
    within ``max_retries`` attempts — typically only triggers on degenerate
    configurations like ``M=5, L=8, min_distance=10``.
    """
    centers = np.zeros((M, 2), dtype=np.int32)
    last_candidate = np.zeros(2, dtype=np.int32)
    for m in range(M):
        placed = False
        for _ in range(max_retries):
            last_candidate = rng.integers(0, L, size=2).astype(np.int32)
            ok = True
            for k in range(m):
                if float(np.linalg.norm(last_candidate - centers[k])) < min_distance:
                    ok = False
                    break
            if ok:
                centers[m] = last_candidate
                placed = True
                break
        if not placed:
            warnings.warn(
                f"Hotspot {m} placement failed after {max_retries} retries; "
                "using last candidate. Lower min_hotspot_distance or M.",
                UserWarning,
                stacklevel=2,
            )
            centers[m] = last_candidate
    return centers


def _distribute_K_over_M(K: int, M: int) -> np.ndarray:
    """Spread ``K`` items as evenly as possible across ``M`` buckets.

    ``K=20, M=3 → [7, 7, 6]``; the remainder is assigned to the first
    buckets.
    """
    if M <= 0:
        raise ValueError(f"M must be ≥ 1, got {M}")
    base, remainder = divmod(K, M)
    counts = np.full(M, base, dtype=np.int32)
    counts[:remainder] += 1
    return counts


def _sample_around_center(
    center: np.ndarray,
    sigma: float,
    L: int,
    rng: np.random.Generator,
    max_retries: int = 20,
) -> np.ndarray:
    """Gaussian-sample a single grid cell near ``center``.

    Tries ``max_retries`` times to land strictly inside ``[0, L)``; otherwise
    clips to the grid.
    """
    for _ in range(max_retries):
        offset = rng.normal(0.0, sigma, size=2)
        point = np.round(center + offset).astype(np.int32)
        if np.all((point >= 0) & (point < L)):
            return point
    offset = rng.normal(0.0, sigma, size=2)
    point = np.round(center + offset).astype(np.int32)
    return np.clip(point, 0, L - 1)
