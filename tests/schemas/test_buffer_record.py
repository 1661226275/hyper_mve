"""Unit tests for ``hyper_mve.schemas.buffer_record``."""
from __future__ import annotations

import pickle
from dataclasses import fields, replace

import numpy as np
import pytest

from hyper_mve.schemas import TimeStepRecord


def _make_valid_record(N: int = 4, A: int = 6, obs_dim: int = 99, t: int = 0) -> TimeStepRecord:
    return TimeStepRecord(
        o=np.zeros((N, obs_dim), dtype=np.float32),
        a=np.zeros(N, dtype=np.int64),
        r=np.zeros(N, dtype=np.float32),
        delta=np.zeros(N, dtype=np.float32),
        pi_mve=np.full((N, A), 1.0 / A, dtype=np.float32),
        v=np.zeros(N, dtype=np.float32),
        tau=np.array([0, 0, 1, 1][:N], dtype=np.int8),
        cap=np.zeros((N, 4), dtype=np.float32),
        c_hat=np.full(N, 0.5, dtype=np.float32),
        z_hat=np.full((N, N - 1, 2), 0.5, dtype=np.float32),
        t=t,
        done=False,
    )


def test_construct_valid():
    record = _make_valid_record()
    assert record.o.shape == (4, 99)
    assert record.tau.tolist() == [0, 0, 1, 1]


def test_n_dimension_mismatch():
    with pytest.raises(ValueError, match="N dimension mismatch"):
        TimeStepRecord(
            o=np.zeros((4, 99), dtype=np.float32),
            a=np.zeros(3, dtype=np.int64),       # WRONG: N=3 vs 4
            r=np.zeros(4, dtype=np.float32),
            delta=np.zeros(4, dtype=np.float32),
            pi_mve=np.zeros((4, 6), dtype=np.float32),
            v=np.zeros(4, dtype=np.float32),
            tau=np.zeros(4, dtype=np.int8),
            cap=np.zeros((4, 4), dtype=np.float32),
            c_hat=np.zeros(4, dtype=np.float32),
            z_hat=np.zeros((4, 3, 2), dtype=np.float32),
            t=0,
        )


def test_z_hat_dim_mismatch():
    with pytest.raises(ValueError, match="z_hat dim 1"):
        TimeStepRecord(
            o=np.zeros((4, 99), dtype=np.float32),
            a=np.zeros(4, dtype=np.int64),
            r=np.zeros(4, dtype=np.float32),
            delta=np.zeros(4, dtype=np.float32),
            pi_mve=np.zeros((4, 6), dtype=np.float32),
            v=np.zeros(4, dtype=np.float32),
            tau=np.zeros(4, dtype=np.int8),
            cap=np.zeros((4, 4), dtype=np.float32),
            c_hat=np.zeros(4, dtype=np.float32),
            z_hat=np.zeros((4, 4, 2), dtype=np.float32),  # WRONG: should be (4, 3, 2)
            t=0,
        )


def test_c_hat_is_scalar_per_agent():
    """v4 (Ch4.2.3 Head 1): c_hat is (N,) scalar, not (N, d_c)."""
    record = _make_valid_record()
    assert record.c_hat.shape == (4,)
    assert record.c_hat.dtype == np.float32


def test_z_hat_order_convention():
    """For agent i, z_hat[i, k] -> agent_id (k if k<i else k+1).

    Mis-ordering silently corrupts L_opp CE loss (Ch4.5.2).
    """
    N = 4
    z_hat = np.zeros((N, N - 1, 2), dtype=np.float32)
    for i in range(N):
        for k in range(N - 1):
            opp_id = k if k < i else k + 1
            z_hat[i, k, 0] = (opp_id + 1) / 10.0
            z_hat[i, k, 1] = 1.0 - z_hat[i, k, 0]
    # agent 2 sees opponents [0, 1, 3] → probs [0.1, 0.2, 0.4]
    assert np.isclose(z_hat[2, 0, 0], 0.1)
    assert np.isclose(z_hat[2, 1, 0], 0.2)
    assert np.isclose(z_hat[2, 2, 0], 0.4)


def test_to_from_arrays_roundtrip():
    record = _make_valid_record(t=42)
    arrays = record.to_arrays()
    assert "o" in arrays and "tau" in arrays and "z_hat" in arrays

    record2 = TimeStepRecord.from_arrays(arrays)
    assert record2.t == 42
    assert record2.done is False
    assert np.allclose(record.o, record2.o)
    assert np.array_equal(record.tau, record2.tau)
    assert np.allclose(record.z_hat, record2.z_hat)


def test_empty_belief_factory():
    record = TimeStepRecord.empty_belief(N=4, A=6, obs_dim=99, t=5)
    assert record.t == 5
    assert np.allclose(record.pi_mve, 1.0 / 6)
    assert np.allclose(record.z_hat, 0.5)
    assert record.c_hat.shape == (4,)


def test_done_terminal():
    base = _make_valid_record()
    record = replace(base, done=True)
    assert record.done is True


def test_v_nan_allowed():
    record = _make_valid_record()
    arrays = record.to_arrays()
    arrays["v"] = np.array([np.nan, 0, 0, 0], dtype=np.float32)
    record2 = TimeStepRecord.from_arrays(arrays)
    assert np.isnan(record2.v[0])


def test_pickle_roundtrip():
    record = _make_valid_record(t=10)
    record2 = pickle.loads(pickle.dumps(record))
    assert record.t == record2.t
    assert np.array_equal(record.tau, record2.tau)


def test_field_count():
    """10 data fields + t + done (Ch5.6.1)."""
    field_names = {f.name for f in fields(TimeStepRecord)}
    expected = {
        "o", "a", "r", "delta", "pi_mve", "v",
        "tau", "cap", "c_hat", "z_hat",
        "t", "done",
    }
    assert field_names == expected
