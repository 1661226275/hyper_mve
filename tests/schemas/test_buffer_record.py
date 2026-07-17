"""Unit tests for ``hyper_mve.utils.schemas.buffer_record`` (v5 TimeStepRecord)."""
from __future__ import annotations

import pickle
from dataclasses import fields, replace

import numpy as np
import pytest

from hyper_mve.utils.schemas import TimeStepRecord

_G = 5  # |G| for the g2 family


def _make_valid_record(N: int = 4, A: int = 6, obs_dim: int = 95, t: int = 0) -> TimeStepRecord:
    return TimeStepRecord(
        o=np.zeros((N, obs_dim), dtype=np.float32),
        a=np.zeros(N, dtype=np.int64),
        r=np.zeros(N, dtype=np.float32),
        pi_mve=np.full((N, A), 1.0 / A, dtype=np.float32),
        v=np.zeros(N, dtype=np.float32),
        row=np.zeros((N, N - 1), dtype=np.float32),
        g_hat=np.full((N, _G), 1.0 / _G, dtype=np.float32),
        g=2,
        t=t,
        done=False,
    )


def test_construct_valid():
    record = _make_valid_record()
    assert record.o.shape == (4, 95)
    assert record.g == 2
    assert record.row.shape == (4, 3)


def test_n_dimension_mismatch():
    with pytest.raises(ValueError, match="N dimension mismatch"):
        TimeStepRecord(
            o=np.zeros((4, 95), dtype=np.float32),
            a=np.zeros(3, dtype=np.int64),       # WRONG: N=3 vs 4
            r=np.zeros(4, dtype=np.float32),
            pi_mve=np.zeros((4, 6), dtype=np.float32),
            v=np.zeros(4, dtype=np.float32),
            row=np.zeros((4, 3), dtype=np.float32),
            g_hat=np.zeros((4, _G), dtype=np.float32),
            g=0,
            t=0,
        )


def test_row_dim_mismatch():
    with pytest.raises(ValueError, match="row dim 1"):
        TimeStepRecord(
            o=np.zeros((4, 95), dtype=np.float32),
            a=np.zeros(4, dtype=np.int64),
            r=np.zeros(4, dtype=np.float32),
            pi_mve=np.zeros((4, 6), dtype=np.float32),
            v=np.zeros(4, dtype=np.float32),
            row=np.zeros((4, 4), dtype=np.float32),   # WRONG: should be (4, 3)
            g_hat=np.zeros((4, _G), dtype=np.float32),
            g=0,
            t=0,
        )


def test_row_order_convention():
    """For agent i, row[i, k] -> w_{i,j} with j = (k if k<i else k+1).

    Must match the observation ``row`` block and ``Regime.row(i)`` —
    mis-ordering silently corrupts the conditioning.
    """
    from hyper_mve.utils.schemas import build_g2

    fam = build_g2(1.0)
    reg = fam.regimes[2]  # asym_exploit: W = [[1,-1],[1,1]]
    rows = np.stack([reg.row(i) for i in range(2)])
    record = TimeStepRecord(
        o=np.zeros((2, 39), dtype=np.float32),
        a=np.zeros(2, dtype=np.int64),
        r=np.zeros(2, dtype=np.float32),
        pi_mve=np.full((2, 6), 1.0 / 6, dtype=np.float32),
        v=np.zeros(2, dtype=np.float32),
        row=rows,
        g_hat=np.full((2, _G), 1.0 / _G, dtype=np.float32),
        g=2,
        t=0,
    )
    assert record.row[0, 0] == -1.0   # w_01
    assert record.row[1, 0] == 1.0    # w_10


def test_to_from_arrays_roundtrip():
    record = _make_valid_record(t=42)
    arrays = record.to_arrays()
    assert "o" in arrays and "row" in arrays and "g_hat" in arrays and "g" in arrays

    record2 = TimeStepRecord.from_arrays(arrays)
    assert record2.t == 42
    assert record2.g == 2
    assert record2.done is False
    assert np.allclose(record.o, record2.o)
    assert np.array_equal(record.row, record2.row)
    assert np.allclose(record.g_hat, record2.g_hat)


def test_empty_belief_factory():
    record = TimeStepRecord.empty_belief(N=4, A=6, obs_dim=95, n_regimes=_G, t=5)
    assert record.t == 5
    assert np.allclose(record.pi_mve, 1.0 / 6)
    assert np.allclose(record.g_hat, 1.0 / _G)
    assert record.row.shape == (4, 3)
    assert record.g == 0


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
    assert np.array_equal(record.row, record2.row)


def test_field_count():
    """7 data fields + g + t + done (v5)."""
    field_names = {f.name for f in fields(TimeStepRecord)}
    expected = {
        "o", "a", "r", "pi_mve", "v", "row", "g_hat",
        "g", "t", "done",
    }
    assert field_names == expected
