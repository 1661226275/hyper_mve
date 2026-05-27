"""Unit tests for patchy resource spawning (Pkg-02 spec 06 §5.1)."""
from __future__ import annotations

import numpy as np
import pytest

from hyper_mve.envs.resource_commons.spawn import (
    _distribute_K_over_M,
    _sample_hotspot_centers,
    spawn_patchy_resources,
)


def test_spawn_shape_medium():
    rng = np.random.default_rng(42)
    centers, positions = spawn_patchy_resources(K=20, M=3, L=16, rng=rng)
    assert centers.shape == (3, 2)
    assert positions.shape == (20, 2)
    assert centers.dtype == np.int32
    assert positions.dtype == np.int32


def test_spawn_in_grid():
    rng = np.random.default_rng(42)
    centers, positions = spawn_patchy_resources(K=20, M=3, L=16, rng=rng)
    assert np.all((centers >= 0) & (centers < 16))
    assert np.all((positions >= 0) & (positions < 16))


def test_distribute_K_over_M_balanced():
    counts = _distribute_K_over_M(K=20, M=3)
    assert list(counts) == [7, 7, 6]
    assert int(counts.sum()) == 20


def test_distribute_K_over_M_exact_divide():
    counts = _distribute_K_over_M(K=21, M=3)
    assert list(counts) == [7, 7, 7]


def test_distribute_K_zero_M_raises():
    with pytest.raises(ValueError):
        _distribute_K_over_M(K=10, M=0)


def test_hotspot_min_distance():
    rng = np.random.default_rng(42)
    centers = _sample_hotspot_centers(M=3, L=16, min_distance=4.0, rng=rng)
    for i in range(3):
        for j in range(i + 1, 3):
            dist = float(np.linalg.norm(centers[i] - centers[j]))
            assert dist >= 4.0 - 1e-6


def test_spawn_reproducible_same_seed():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    c1, p1 = spawn_patchy_resources(K=20, M=3, L=16, rng=rng1)
    c2, p2 = spawn_patchy_resources(K=20, M=3, L=16, rng=rng2)
    assert np.array_equal(c1, c2)
    assert np.array_equal(p1, p2)


def test_spawn_different_seeds_different():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(43)
    c1, _ = spawn_patchy_resources(K=20, M=3, L=16, rng=rng1)
    c2, _ = spawn_patchy_resources(K=20, M=3, L=16, rng=rng2)
    assert not np.array_equal(c1, c2)


def test_resources_clustered_near_hotspots():
    """With sigma_patch=2, most points should fall within 2σ of some hotspot."""
    rng = np.random.default_rng(42)
    centers, positions = spawn_patchy_resources(
        K=30, M=3, L=20, sigma_patch=2.0, rng=rng,
    )
    near = 0
    for p in positions:
        dists = np.linalg.norm(centers - p, axis=1)
        if float(dists.min()) < 4.0:
            near += 1
    assert near >= 27   # ~90% within 2σ


def test_spawn_easy_config():
    rng = np.random.default_rng(42)
    centers, positions = spawn_patchy_resources(K=8, M=1, L=8, rng=rng)
    assert centers.shape == (1, 2)
    assert positions.shape == (8, 2)


def test_spawn_hard_config():
    rng = np.random.default_rng(42)
    centers, positions = spawn_patchy_resources(K=40, M=5, L=24, rng=rng)
    assert centers.shape == (5, 2)
    assert positions.shape == (40, 2)


def test_min_distance_too_large_warns():
    rng = np.random.default_rng(42)
    with pytest.warns(UserWarning, match="placement failed"):
        _sample_hotspot_centers(M=5, L=8, min_distance=10.0, rng=rng)
