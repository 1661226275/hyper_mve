"""Pkg-05 spec 06 acceptance: MVEPlanner (C5-P1 CRN determinism, v5 API)."""
import time
from dataclasses import replace

import numpy as np
import pytest
import torch

from hyper_mve.configs import V4Config
from hyper_mve.models import HyperMuZeroModel
from hyper_mve.planning.mve_planner import MVEPlanner

_G = 5


def _spy(monkeypatch, obj, name):
    """Lightweight mocker.spy replacement: record calls, delegate to original."""
    calls = []
    orig = getattr(obj, name)

    def wrapper(*args, **kwargs):
        calls.append((args, kwargs))
        return orig(*args, **kwargs)

    monkeypatch.setattr(obj, name, wrapper)
    return calls


@pytest.fixture
def cfg_duo():
    cfg = V4Config.from_preset("rel_duo")
    # small planner budget for CPU test speed
    return replace(cfg, train=replace(cfg.train, mve_samples=12, mve_depth=2))


@pytest.fixture
def model(cfg_duo):
    m = HyperMuZeroModel(cfg_duo)
    m.eval()
    return m


@pytest.fixture
def planner(cfg_duo):
    return MVEPlanner(cfg_duo)


def _make_inputs(cfg, B=1, device="cpu"):
    N = cfg.env.N
    return {
        "root_s": torch.randn(B, cfg.model.latent_dim, device=device),
        "row": {k: torch.rand(B, N - 1, device=device) * 2 - 1 for k in range(N)},
        "belief": {
            k: torch.softmax(torch.randn(B, _G, device=device), dim=-1)
            for k in range(N)
        },
    }


# ====== C5-P1: CRN determinism ======

def test_crn_step0_deterministic_same_seed(planner, model, cfg_duo):
    inputs = _make_inputs(cfg_duo)
    planner.crn_rng = np.random.default_rng(42)
    out1 = planner.sample_mve_plan(model, **inputs)
    planner.crn_rng = np.random.default_rng(42)
    out2 = planner.sample_mve_plan(model, **inputs)
    assert torch.allclose(out1, out2, atol=1e-5)


def test_crn_different_seed_different_output(cfg_duo, model):
    # Disable the qstd noise guard (an untrained model's flat candidate returns
    # would collapse both seeds to the identical uniform fallback).
    cfg = replace(cfg_duo, train=replace(cfg_duo.train, mve_qstd_floor=0.0))
    planner = MVEPlanner(cfg)
    inputs = _make_inputs(cfg)
    planner.crn_rng = np.random.default_rng(42)
    out1 = planner.sample_mve_plan(model, **inputs)
    planner.crn_rng = np.random.default_rng(999)
    out2 = planner.sample_mve_plan(model, **inputs)
    assert not torch.allclose(out1, out2, atol=1e-3)


# ====== v5 API: subjective-only context calls ======

def test_planner_uses_subjective_context_only(planner, model, cfg_duo, monkeypatch):
    subj_calls = _spy(monkeypatch, model, "set_context_subjective")
    planner.sample_mve_plan(model, **_make_inputs(cfg_duo))
    # θ-cache path: one set_context_subjective per agent at base batch
    assert len(subj_calls) >= cfg_duo.env.N
    assert not hasattr(model, "set_context_objective")


def test_planner_no_legacy_set_context(planner, model, cfg_duo, monkeypatch):
    if hasattr(model, "set_context"):
        calls = _spy(monkeypatch, model, "set_context")
        planner.sample_mve_plan(model, **_make_inputs(cfg_duo))
        assert len(calls) == 0


# ====== θ-cache fast path ≡ slow path ======

class _NoThetaCacheProxy:
    """Duck-typed model view that hides install_subjective_theta, forcing the
    planner onto the verbatim set_context_subjective slow path."""

    def __init__(self, m):
        object.__setattr__(self, "_m", m)

    def __getattr__(self, name):
        if name == "install_subjective_theta":
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "_m"), name)


def test_theta_cache_matches_slow_path(cfg_duo, model):
    cfg = replace(cfg_duo, train=replace(cfg_duo.train, mve_qstd_floor=0.0))
    inputs = _make_inputs(cfg, B=2)

    fast = MVEPlanner(cfg)
    fast.crn_rng = np.random.default_rng(7)
    out_fast = fast.sample_mve_plan(model, **inputs)

    slow = MVEPlanner(cfg)
    slow.crn_rng = np.random.default_rng(7)
    out_slow = slow.sample_mve_plan(_NoThetaCacheProxy(model), **inputs)

    # The two paths run the hypernet GEMMs at different batch sizes (B vs B*M),
    # so MKL/cuBLAS blocking gives ~1e-4-level float drift which the z-score
    # softmax amplifies slightly — equivalence is numerical, not bitwise.
    assert torch.allclose(out_fast, out_slow, atol=5e-3)
    assert (out_fast.argmax(-1) == out_slow.argmax(-1)).all()


# ====== output shape ======

def test_sample_mve_plan_output_shape(planner, model, cfg_duo):
    B = 2
    policies = planner.sample_mve_plan(model, **_make_inputs(cfg_duo, B=B))
    assert policies.shape == (B, cfg_duo.env.N, cfg_duo.env.A)
    assert torch.allclose(
        policies.sum(dim=-1), torch.ones(B, cfg_duo.env.N), atol=1e-5,
    )


def test_diagnostics_shapes(planner, model, cfg_duo):
    B = 2
    pi, diag = planner.sample_mve_plan(
        model, **_make_inputs(cfg_duo, B=B), return_diagnostics=True)
    N, A = cfg_duo.env.N, cfg_duo.env.A
    assert diag["returns_per_action"].shape == (B, N, A)
    assert diag["q_std"].shape == (B, N)
    assert diag["q_gap"].shape == (B, N)
    assert 0.0 <= float(diag["uniform_frac"]) <= 1.0


# ====== R5-10: row/belief shape guard ======

def test_planner_row_belief_shape(planner, model, cfg_duo):
    inputs = _make_inputs(cfg_duo)
    inputs["row"][0] = torch.rand(1, cfg_duo.env.N)   # full-W info leak
    with pytest.raises((AssertionError, RuntimeError, ValueError)):
        planner.sample_mve_plan(model, **inputs)

    inputs = _make_inputs(cfg_duo)
    inputs["belief"][0] = (torch.rand(1), torch.rand(1, 1, 2))   # v4 tuple form
    with pytest.raises((AssertionError, RuntimeError, ValueError, TypeError)):
        planner.sample_mve_plan(model, **inputs)


# ====== ablation switches (shape only; full ablation is Pkg-08) ======

def test_use_crn_disabled(model, cfg_duo):
    cfg = replace(cfg_duo, train=replace(cfg_duo.train, use_crn=False))
    p = MVEPlanner(cfg)
    out = p.sample_mve_plan(model, **_make_inputs(cfg))
    assert out.shape == (1, cfg.env.N, cfg.env.A)


def test_randomize_order_disabled(model, cfg_duo):
    cfg = replace(cfg_duo, train=replace(cfg_duo.train, randomize_order=False))
    p = MVEPlanner(cfg)
    out = p.sample_mve_plan(model, **_make_inputs(cfg))
    assert out.shape == (1, cfg.env.N, cfg.env.A)


# ====== performance ======

@pytest.mark.gpu
def test_sample_mve_plan_under_300ms(cfg_duo):
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    cfg = V4Config.from_preset("rel_duo")
    model = HyperMuZeroModel(cfg).cuda().eval()
    planner = MVEPlanner(cfg)
    inputs = _make_inputs(cfg, device="cuda")
    for _ in range(5):
        planner.sample_mve_plan(model, **inputs)
    torch.cuda.synchronize()
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        planner.sample_mve_plan(model, **inputs)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    mean_ms = sum(times) / len(times)
    assert mean_ms < 300.0, f"sample_mve_plan {mean_ms:.1f}ms (regression guard)"
