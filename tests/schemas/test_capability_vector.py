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


# ====== v4 修订: normalize() utility 单测 (A1, Pkg-01 spec 02 §5.1) ======

def test_capability_normalize_shape_and_dtype():
    """normalize 输出 (4,) float32."""
    cap = CapabilityVector(eta=1.0, phi_fov=3.0, nu=0.9, zeta=20.0)
    out = cap.normalize()
    assert out.shape == (4,)
    assert out.dtype == np.float32


def test_capability_normalize_lower_boundary():
    """每维下界 → 输出 = 0.0."""
    cap = CapabilityVector(eta=0.5, phi_fov=2.0, nu=0.8, zeta=10.0)
    out = cap.normalize()
    assert np.allclose(out, [0.0, 0.0, 0.0, 0.0])


def test_capability_normalize_upper_boundary():
    """每维上界 → 输出 = 1.0."""
    cap = CapabilityVector(eta=1.5, phi_fov=4.0, nu=1.0, zeta=30.0)
    out = cap.normalize()
    assert np.allclose(out, [1.0, 1.0, 1.0, 1.0])


def test_capability_normalize_midpoint():
    """每维中点 → 输出 = 0.5."""
    cap = CapabilityVector(eta=1.0, phi_fov=3.0, nu=0.9, zeta=20.0)
    out = cap.normalize()
    assert np.allclose(out, [0.5, 0.5, 0.5, 0.5])


def test_capability_normalize_eta_only():
    """单维度归一化正确性: eta 在范围内任意值."""
    # eta = 0.5 + 0.3*(1.5-0.5) = 0.8 → normalized = 0.3
    cap = CapabilityVector(eta=0.8, phi_fov=3.0, nu=0.9, zeta=20.0)
    out = cap.normalize()
    assert np.isclose(out[0], 0.3, atol=1e-5)


def test_capability_normalize_zeta_scale_handling():
    """zeta (10-30) 归一化解决量级问题."""
    cap_small = CapabilityVector(eta=1.0, phi_fov=3.0, nu=0.9, zeta=10.0)
    cap_large = CapabilityVector(eta=1.0, phi_fov=3.0, nu=0.9, zeta=30.0)
    # raw 量级差 3x; normalize 后差 1.0 (与其他维度同量级)
    assert np.isclose(cap_small.normalize()[3], 0.0)
    assert np.isclose(cap_large.normalize()[3], 1.0)


def test_capability_normalize_1000_samples_in_unit_range():
    """1000 次采样 cap, normalize 后每维 ∈ [0, 1]; 均值约 0.5."""
    rng = np.random.default_rng(seed=42)
    samples = np.stack([sample_default(rng).normalize() for _ in range(1000)])

    assert samples.shape == (1000, 4)
    assert (samples >= 0.0).all()
    assert (samples <= 1.0).all()

    means = samples.mean(axis=0)
    # uniform 分布均值 = 0.5, 1000 samples 容差 ~ 0.05
    assert np.allclose(means, [0.5, 0.5, 0.5, 0.5], atol=0.05)


def test_capability_normalize_does_not_modify_raw_fields():
    """normalize 是只读 utility, 不改 frozen 字段."""
    cap = CapabilityVector(eta=1.2, phi_fov=2.5, nu=0.85, zeta=15.0)
    _ = cap.normalize()
    # 原字段仍是 raw values
    assert cap.eta == 1.2
    assert cap.phi_fov == 2.5
    assert cap.nu == 0.85
    assert cap.zeta == 15.0


def test_cap_norm_constants_match_ch36():
    """CAP_NORM_LO/HI 必须与 Ch3.6 采样范围一致 (env/sample_default 同源)."""
    from hyper_mve.schemas.capability import CAP_NORM_LO, CAP_NORM_HI
    assert CAP_NORM_LO == (0.5, 2.0, 0.8, 10.0)
    assert CAP_NORM_HI == (1.5, 4.0, 1.0, 30.0)


def test_capability_normalize_custom_dtype():
    """normalize 接受 dtype 参数."""
    cap = CapabilityVector(eta=1.0, phi_fov=3.0, nu=0.9, zeta=20.0)
    out_f64 = cap.normalize(dtype=np.float64)
    assert out_f64.dtype == np.float64
    assert np.allclose(out_f64, [0.5, 0.5, 0.5, 0.5])
