"""Unit tests for ``hyper_mve.schemas.capability``."""
from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch

from hyper_mve.schemas import (
    CapabilityVector,
    sample_default,
    sample_n,
    to_batch_tensor,
)


def test_construct_valid():
    cap = CapabilityVector(eta=1.0, phi_fov=3.0, nu=0.9, zeta=20.0)
    assert cap.eta == 1.0
    assert cap.fov_int == 3


def test_boundary_lower():
    cap = CapabilityVector(eta=0.5, phi_fov=2.0, nu=0.8, zeta=10.0)
    assert cap.fov_int == 2


def test_boundary_upper():
    cap = CapabilityVector(eta=1.5, phi_fov=4.0, nu=1.0, zeta=30.0)
    assert cap.fov_int == 4


def test_eta_out_of_range():
    with pytest.raises(ValueError, match=r"eta.*∉"):
        CapabilityVector(eta=2.0, phi_fov=3.0, nu=0.9, zeta=20.0)
    with pytest.raises(ValueError):
        CapabilityVector(eta=0.4, phi_fov=3.0, nu=0.9, zeta=20.0)


def test_phi_fov_out_of_range():
    with pytest.raises(ValueError, match=r"phi_fov.*∉"):
        CapabilityVector(eta=1.0, phi_fov=5.0, nu=0.9, zeta=20.0)


def test_nu_out_of_range():
    with pytest.raises(ValueError):
        CapabilityVector(eta=1.0, phi_fov=3.0, nu=0.5, zeta=20.0)


def test_zeta_out_of_range():
    with pytest.raises(ValueError):
        CapabilityVector(eta=1.0, phi_fov=3.0, nu=0.9, zeta=50.0)


def test_nan_rejected():
    with pytest.raises(ValueError):
        CapabilityVector(eta=float("nan"), phi_fov=3.0, nu=0.9, zeta=20.0)


def test_frozen():
    cap = CapabilityVector(eta=1.0, phi_fov=3.0, nu=0.9, zeta=20.0)
    with pytest.raises(Exception):  # FrozenInstanceError
        cap.eta = 0.5  # type: ignore[misc]


def test_fov_int_round():
    assert CapabilityVector(eta=1.0, phi_fov=2.4, nu=0.9, zeta=20.0).fov_int == 2
    assert CapabilityVector(eta=1.0, phi_fov=2.6, nu=0.9, zeta=20.0).fov_int == 3
    assert CapabilityVector(eta=1.0, phi_fov=3.5, nu=0.9, zeta=20.0).fov_int == 4


def test_to_array_dtype():
    cap = CapabilityVector(eta=1.0, phi_fov=3.0, nu=0.9, zeta=20.0)
    arr = cap.to_array()
    assert arr.shape == (4,)
    assert arr.dtype == np.float32
    assert np.allclose(arr, [1.0, 3.0, 0.9, 20.0])


def test_from_array_roundtrip():
    cap = CapabilityVector(eta=1.2, phi_fov=2.5, nu=0.85, zeta=15.0)
    arr = cap.to_array(dtype=np.float64)
    cap2 = CapabilityVector.from_array(arr)
    assert cap == cap2


def test_from_array_bad_shape():
    with pytest.raises(AssertionError):
        CapabilityVector.from_array(np.array([0.6, 3.0, 0.9, 20.0, 0.0]))


def test_sample_default_in_range():
    rng = np.random.default_rng(seed=42)
    for _ in range(1000):
        cap = sample_default(rng)
        assert 0.5 <= cap.eta <= 1.5
        assert 2.0 <= cap.phi_fov <= 4.0
        assert 0.8 <= cap.nu <= 1.0
        assert 10.0 <= cap.zeta <= 30.0


def test_sample_n_reproducible():
    caps_a = sample_n(4, np.random.default_rng(seed=42))
    caps_b = sample_n(4, np.random.default_rng(seed=42))
    assert caps_a == caps_b


def test_to_batch_tensor_shape():
    caps = sample_n(4, np.random.default_rng(seed=0))
    t = to_batch_tensor(caps)
    assert t.shape == (4, 4)
    assert t.dtype == torch.float32


def test_pickle_roundtrip():
    cap = CapabilityVector(eta=1.0, phi_fov=3.0, nu=0.9, zeta=20.0)
    assert pickle.loads(pickle.dumps(cap)) == cap


def test_field_order_matches_ch_3_6():
    """Ch3.6 order: eta, phi_fov, nu, zeta — fixed, not reorderable."""
    from dataclasses import fields
    names = [f.name for f in fields(CapabilityVector)]
    assert names == ["eta", "phi_fov", "nu", "zeta"]
