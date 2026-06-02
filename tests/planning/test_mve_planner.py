"""Pkg-05 spec 06 acceptance: MVEPlanner (C5-P1 CRN determinism + C5-P2 migration)."""
import time
from dataclasses import replace

import numpy as np
import pytest
import torch

from hyper_mve.configs import V4Config
from hyper_mve.models import HyperMuZeroModel
from hyper_mve.planning.mve_planner import MVEPlanner


@pytest.fixture
def cfg_medium():
    return V4Config.from_preset("medium")


@pytest.fixture
def model(cfg_medium):
    m = HyperMuZeroModel(cfg_medium)
    m.eval()
    return m


@pytest.fixture
def planner(cfg_medium):
    return MVEPlanner(cfg_medium)


def _make_inputs(cfg, B=1, device="cpu"):
    N = cfg.env.N
    return {
        "root_s": torch.randn(B, cfg.model.latent_dim, device=device),
        "cap": {k: torch.rand(B, 4, device=device) for k in range(N)},
        "belief": {
            k: (torch.rand(B, device=device),
                torch.softmax(torch.randn(B, N - 1, 2, device=device), dim=-1))
            for k in range(N)
        },
        "c_t": torch.full((B,), 0.5, device=device),
    }


# ====== C5-P1: CRN determinism ======

def test_crn_step0_deterministic_same_seed(planner, model, cfg_medium):
    inputs = _make_inputs(cfg_medium)
    planner.crn_rng = np.random.default_rng(42)
    out1 = planner.sample_mve_plan(model, **inputs)
    planner.crn_rng = np.random.default_rng(42)
    out2 = planner.sample_mve_plan(model, **inputs)
    assert torch.allclose(out1, out2, atol=1e-5)


def test_crn_different_seed_different_output(planner, model, cfg_medium):
    inputs = _make_inputs(cfg_medium)
    planner.crn_rng = np.random.default_rng(42)
    out1 = planner.sample_mve_plan(model, **inputs)
    planner.crn_rng = np.random.default_rng(999)
    out2 = planner.sample_mve_plan(model, **inputs)
    assert not torch.allclose(out1, out2, atol=1e-3)


# ====== C5-P2: 4 set_context call sites migrated to two-step API ======

def test_planner_4_set_context_migrated(planner, model, cfg_medium, mocker):
    spy_obj = mocker.spy(model, "set_context_objective")
    spy_subj = mocker.spy(model, "set_context_subjective")
    planner.sample_mve_plan(model, **_make_inputs(cfg_medium))
    assert spy_obj.call_count >= 1
    assert spy_subj.call_count >= cfg_medium.env.N


def test_planner_no_legacy_set_context(planner, model, cfg_medium, mocker):
    if hasattr(model, "set_context"):
        spy = mocker.spy(model, "set_context")
        planner.sample_mve_plan(model, **_make_inputs(cfg_medium))
        assert spy.call_count == 0


# ====== output shape ======

def test_sample_mve_plan_output_shape(planner, model, cfg_medium):
    B = 2
    policies = planner.sample_mve_plan(model, **_make_inputs(cfg_medium, B=B))
    assert policies.shape == (B, cfg_medium.env.N, cfg_medium.env.A)
    assert torch.allclose(
        policies.sum(dim=-1), torch.ones(B, cfg_medium.env.N), atol=1e-5,
    )


# ====== R5-10: cap/belief shape guard ======

def test_planner_cap_belief_shape(planner, model, cfg_medium):
    inputs = _make_inputs(cfg_medium)
    inputs["cap"][0] = torch.rand(1, 5)  # type-leak shape
    with pytest.raises((AssertionError, RuntimeError, ValueError)):
        planner.sample_mve_plan(model, **inputs)


# ====== ablation switches (shape only; full ablation is Pkg-08) ======

def test_use_crn_disabled(model, cfg_medium):
    cfg = replace(cfg_medium, train=replace(cfg_medium.train, use_crn=False))
    p = MVEPlanner(cfg)
    out = p.sample_mve_plan(model, **_make_inputs(cfg))
    assert out.shape == (1, cfg.env.N, cfg.env.A)


def test_use_coord_desc_disabled(model, cfg_medium):
    cfg = replace(cfg_medium, train=replace(cfg_medium.train, use_coord_desc=False))
    p = MVEPlanner(cfg)
    out = p.sample_mve_plan(model, **_make_inputs(cfg))
    assert out.shape == (1, cfg.env.N, cfg.env.A)


# ====== performance ======

@pytest.mark.gpu
def test_sample_mve_plan_under_50ms(cfg_medium):
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    model = HyperMuZeroModel(cfg_medium).cuda().eval()
    planner = MVEPlanner(cfg_medium)
    inputs = _make_inputs(cfg_medium, device="cuda")
    for _ in range(5):
        planner.sample_mve_plan(model, **inputs)
    torch.cuda.synchronize()
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        planner.sample_mve_plan(model, **inputs)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    assert sum(times) / len(times) < 50.0
