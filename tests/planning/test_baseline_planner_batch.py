"""Regression: the 5 internal baselines survive the MVE planner's batch expansion.

Root cause (Test-2 crash ``Sizes of tensors must match except in dimension 1.
Expected size 96 but got size 16``): the planner grows the rollout batch by
``repeat_interleave`` from ``B_spa`` to ``B*M = B_spa * A`` *between*
``set_context_subjective`` (Phase-1 action sampling) and the Phase-3
``transition``, refreshing the subjective context only before reward/predict.
``HyperMuZeroModel`` is immune (``set_context_objective`` regenerates
``theta_state`` at the expanded batch); the baselines defer all conditioning to
``set_context_subjective`` and so handed ``transition`` a stale-batch ctx/theta/
id tensor. ``BaselineModel._match_batch`` tiles the cached conditioning to the
operand batch, fixing every input/theta/id-conditioned variant uniformly.

These tests would have caught the crash: each runs a real ``sample_mve_plan``
(96 = B*M operands vs 16 = B_spa ctx is exactly the layout exercised here).
"""
import pytest
import torch

from hyper_mve.baselines import create_baseline
from hyper_mve.baselines.internal.base import BaselineModel
from hyper_mve.configs import V4Config
from hyper_mve.planning.mve_planner import MVEPlanner

# The 5 internal factory args (pkg-07 spec 01 §2.1). All defer conditioning to
# set_context_subjective, so all five hit the planner's transition-batch gap.
_INTERNAL_VARIANTS = [
    "input_wide",
    "input_deep",
    "ma_muzero",
    "no_belief",
    "rewardhead_explicit_type",
]


@pytest.fixture(scope="module")
def cfg():
    # medium: N=4, A=6, mve_samples=50 -> spa=8, so the planner expands
    # B_spa=B*8 to B*M=B*8*6 (factor A=6) — the exact ratio of the reported
    # crash (96/16=6).
    return V4Config.from_preset("medium")


def _make_inputs(cfg, B=2, device="cpu"):
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


@pytest.mark.parametrize("variant", _INTERNAL_VARIANTS)
def test_internal_baseline_survives_planner_batch_expansion(cfg, variant):
    """sample_mve_plan must run end-to-end and return a valid (B, N, A) policy."""
    model = create_baseline(cfg, variant)
    model.eval()
    planner = MVEPlanner(cfg)
    inputs = _make_inputs(cfg, B=2)

    pi = planner.sample_mve_plan(model, **inputs)  # must not raise (was the crash)

    B, N, A = 2, cfg.env.N, cfg.env.A
    assert pi.shape == (B, N, A), f"{variant}: bad pi_mve shape {tuple(pi.shape)}"
    assert torch.isfinite(pi).all(), f"{variant}: non-finite pi_mve"
    # Per-agent rows are softmax distributions.
    assert torch.allclose(pi.sum(dim=-1), torch.ones(B, N), atol=1e-4), (
        f"{variant}: pi_mve rows must sum to 1"
    )


@pytest.mark.parametrize("variant", _INTERNAL_VARIANTS)
def test_internal_baseline_planner_diagnostics_shapes(cfg, variant):
    """The diagnostics path (worker/eval use it) also survives the expansion."""
    model = create_baseline(cfg, variant)
    model.eval()
    planner = MVEPlanner(cfg)
    pi, diag = planner.sample_mve_plan(
        model, **_make_inputs(cfg, B=2), return_diagnostics=True,
    )
    B, N, A = 2, cfg.env.N, cfg.env.A
    assert pi.shape == (B, N, A)
    assert diag["returns_per_action"].shape == (B, N, A)
    assert torch.isfinite(diag["returns_per_action"]).all()


# ====== _match_batch unit semantics ======

def test_match_batch_identity_when_equal():
    cond = torch.randn(8, 5)
    ref = torch.zeros(8, 3)
    out = BaselineModel._match_batch(cond, ref)
    assert out is cond  # no copy on the common equal-batch path


def test_match_batch_repeat_interleave_layout():
    # Mirrors the planner: ctx at B_spa=2 -> operand at B*M=6 (factor A=3).
    cond = torch.tensor([[1.0, 1.0], [2.0, 2.0]])  # (2, 2)
    ref = torch.zeros(6, 9)
    out = BaselineModel._match_batch(cond, ref)
    assert out.shape == (6, 2)
    # repeat_interleave (NOT tile): [r0,r0,r0,r1,r1,r1] — matches
    # s_scenarios.repeat_interleave(A) in mve_planner Phase 2.
    expected = torch.tensor(
        [[1.0, 1.0]] * 3 + [[2.0, 2.0]] * 3
    )
    assert torch.equal(out, expected)


def test_match_batch_rejects_non_divisor():
    with pytest.raises(RuntimeError, match="integer"):
        BaselineModel._match_batch(torch.randn(5, 2), torch.zeros(12, 2))
