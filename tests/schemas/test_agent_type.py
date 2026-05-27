"""Unit tests for ``hyper_mve.schemas.agent_type``."""
from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch

from hyper_mve.schemas import (
    AgentType,
    count_in_assignment,
    from_index,
    from_str,
    one_hot,
    to_long_tensor,
)


def test_enum_value_stable():
    assert AgentType.ALPHA.value == 0
    assert AgentType.BETA.value == 1


def test_one_hot_shape_and_dtype():
    arr = one_hot(AgentType.ALPHA)
    assert arr.shape == (2,)
    assert arr.dtype == np.float32
    assert np.allclose(arr, [1.0, 0.0])


def test_one_hot_beta():
    arr = one_hot(AgentType.BETA)
    assert np.allclose(arr, [0.0, 1.0])


def test_one_hot_dtype_override():
    arr = one_hot(AgentType.ALPHA, dtype=np.int8)
    assert arr.dtype == np.int8
    assert arr.tolist() == [1, 0]


def test_from_index_valid():
    assert from_index(0) == AgentType.ALPHA
    assert from_index(1) == AgentType.BETA


def test_from_index_invalid():
    with pytest.raises(ValueError, match=r"∉.*0.*1"):
        from_index(2)
    with pytest.raises(ValueError):
        from_index(-1)


def test_from_str_variants():
    assert from_str("alpha") == AgentType.ALPHA
    assert from_str("ALPHA") == AgentType.ALPHA
    assert from_str("α") == AgentType.ALPHA
    assert from_str("a") == AgentType.ALPHA
    assert from_str("beta") == AgentType.BETA
    assert from_str("BETA") == AgentType.BETA
    assert from_str("β") == AgentType.BETA
    assert from_str("b") == AgentType.BETA


def test_from_str_invalid():
    with pytest.raises(ValueError):
        from_str("gamma")


def test_count_in_assignment():
    types = [AgentType.ALPHA, AgentType.ALPHA, AgentType.BETA, AgentType.BETA]
    counts = count_in_assignment(types)
    assert counts[AgentType.ALPHA] == 2
    assert counts[AgentType.BETA] == 2


def test_count_in_assignment_empty():
    counts = count_in_assignment([])
    assert counts[AgentType.ALPHA] == 0
    assert counts[AgentType.BETA] == 0


def test_to_long_tensor():
    types = [AgentType.ALPHA, AgentType.BETA, AgentType.BETA]
    t = to_long_tensor(types)
    assert t.dtype == torch.int64
    assert t.tolist() == [0, 1, 1]


def test_to_long_tensor_empty():
    t = to_long_tensor([])
    assert t.shape == (0,)


def test_pickle_roundtrip():
    assert pickle.loads(pickle.dumps(AgentType.BETA)) == AgentType.BETA


def test_intenum_arithmetic():
    """IntEnum is directly usable as an int (nn.Embedding indexing)."""
    assert int(AgentType.ALPHA) + 0 == 0
    assert AgentType.BETA == 1


def test_numpy_array_packs_to_int():
    arr = np.array([AgentType.ALPHA, AgentType.BETA])
    assert arr.tolist() == [0, 1]
