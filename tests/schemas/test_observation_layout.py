"""Unit tests for ``hyper_mve.schemas.observation``."""
from __future__ import annotations

import numpy as np
import pytest

from hyper_mve.schemas import (
    ObservationLayout,
    pad_neighbor_block,
    pad_resource_block,
    slice_block,
)


def test_total_dim_medium():
    """Medium config (N=4, K=20): 4 + 60 + 27 + 2 + 4 + 2 = 99."""
    assert ObservationLayout.total_dim(N=4, K=20) == 99


def test_total_dim_easy():
    """Easy config (N=2, K=8): 4 + 24 + 9 + 2 + 4 + 2 = 45."""
    assert ObservationLayout.total_dim(N=2, K=8) == 45


def test_total_dim_hard():
    """Hard config (N=8, K=40): 4 + 120 + 63 + 2 + 4 + 2 = 195."""
    assert ObservationLayout.total_dim(N=8, K=40) == 195


def test_block_dim_each():
    assert ObservationLayout.block_dim("self", 4, 20) == 4
    assert ObservationLayout.block_dim("resource", 4, 20) == 60
    assert ObservationLayout.block_dim("neighbor", 4, 20) == 27
    assert ObservationLayout.block_dim("global", 4, 20) == 2
    assert ObservationLayout.block_dim("capability", 4, 20) == 4
    assert ObservationLayout.block_dim("type", 4, 20) == 2


def test_block_dim_invalid():
    with pytest.raises(ValueError, match="Unknown block"):
        ObservationLayout.block_dim("foo", 4, 20)


def test_block_offset_type_is_last_two():
    start, end = ObservationLayout.block_offset("type", N=4, K=20)
    assert end == 99
    assert end - start == 2


def test_block_offset_self_is_first():
    start, end = ObservationLayout.block_offset("self", N=4, K=20)
    assert start == 0
    assert end == 4


def test_block_offsets_sum_to_total():
    N, K = 4, 20
    for block in ObservationLayout.BLOCK_ORDER:
        s, e = ObservationLayout.block_offset(block, N, K)
        assert e - s == ObservationLayout.block_dim(block, N, K)
    last_start, last_end = ObservationLayout.block_offset("type", N, K)
    assert last_end == ObservationLayout.total_dim(N, K)


def test_block_offset_invalid():
    with pytest.raises(ValueError, match="Unknown block"):
        ObservationLayout.block_offset("foo", 4, 20)


def test_pad_resource_block_empty():
    out = pad_resource_block([], K=20)
    assert out.shape == (60,)
    assert np.allclose(out, 0)


def test_pad_resource_block_partial():
    visible = [(0.1, 0.2, 0.5), (0.3, 0.4, 0.8)]
    out = pad_resource_block(visible, K=20)
    assert out.shape == (60,)
    assert np.allclose(out[:6], [0.1, 0.2, 0.5, 0.3, 0.4, 0.8])
    assert np.allclose(out[6:], 0)


def test_pad_resource_block_overflow():
    visible = [(0.1, 0.2, 0.5)] * 25
    out = pad_resource_block(visible, K=20)
    assert out.shape == (60,)
    # only first K entries retained
    assert np.allclose(out[:60], np.tile([0.1, 0.2, 0.5], 20))


def test_pad_neighbor_block_with_presence():
    visible = [
        (0.0, 0.1, np.array([1, 0, 0, 0, 0, 0], dtype=np.float32), True),
        (0.0, 0.0, np.array([0, 1, 0, 0, 0, 0], dtype=np.float32), False),
    ]
    out = pad_neighbor_block(visible, N=4)
    assert out.shape == (27,)
    assert out[0] == 0.0 and out[1] == 0.1
    assert np.allclose(out[2:8], [1, 0, 0, 0, 0, 0])
    assert out[8] == 1.0
    # slot 1: presence_flag=0
    assert out[17] == 0.0
    # slot 2 (padding): all zeros
    assert np.allclose(out[18:27], 0.0)


def test_pad_neighbor_block_empty():
    out = pad_neighbor_block([], N=4)
    assert out.shape == (27,)
    assert np.allclose(out, 0.0)


def test_slice_block_type():
    obs = np.arange(99, dtype=np.float32)
    type_block = slice_block(obs, "type", N=4, K=20)
    assert type_block.shape == (2,)
    assert np.allclose(type_block, [97.0, 98.0])


def test_self_info_principle():
    """Type block is always 2-dim (own one-hot only), independent of N."""
    assert ObservationLayout.TYPE_DIM == 2
    assert ObservationLayout.block_dim("type", N=8, K=40) == 2
    assert ObservationLayout.block_dim("type", N=2, K=8) == 2


def test_boundary_n1_k0():
    """Degenerate N=1, K=0 case: only self + global + capability + type."""
    assert ObservationLayout.total_dim(N=1, K=0) == 4 + 0 + 0 + 2 + 4 + 2
