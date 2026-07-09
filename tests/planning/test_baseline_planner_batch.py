"""Regression: the internal baselines survive the MVE planner's batch expansion (v5).

Root cause (Test-2 crash ``Sizes of tensors must match except in dimension 1.
Expected size 96 but got size 16``): the planner grows the rollout batch by
``repeat_interleave`` from ``B_spa`` to ``B*M = B_spa * A`` *between*
``set_context_subjective`` (Phase-1 action sampling) and the Phase-3
``transition``, refreshing the subjective context only before reward/predict.
``HyperMuZeroModel`` is immune (v5: transition is a plain batch-agnostic
module); the baselines defer subjective conditioning to
``set_context_subjective`` and so handed reward/predict a stale-batch ctx/theta/
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

# The 4 internal factory args (pkg-07 spec 01 §2.1, v5). All defer subjective
# conditioning to set_context_subjective, so all hit the planner batch gap.
_INTERNAL_VARIANTS = [
    "input_wide",
    "input_deep",
    "ma_muzero",
    "no_belief",
]


@pytest.fixture(scope="module")
def cfg():
    # rel_duo: N=2, A=6, mve_samples=48 -> spa=8, so the planner expands
    # B_spa=B*8 to B*M=B*8*6 (factor A=6) — the exact ratio of the reported
    # crash (96/16=6).
    return V4Config.from_preset("rel_duo")


_G = 5


def _make_inputs(cfg, B=2, device="cpu"):
    N = cfg.env.N
    return {
        "root_s": torch.randn(B, cfg.model.latent_dim, device=device),
        "row": {k: torch.rand(B, N - 1, device=device) * 2 - 1 for k in range(N)},
        "belief": {
            k: torch.softmax(torch.randn(B, _G, device=device), dim=-1)
            for k in range(N)
        },
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
def test_transition_is_objective(cfg, variant):
    """transition() must work with NO context call at all (v5: unconditioned).

    The trainer (compose_total_loss) and planner both call transition before
    any per-agent set_context_subjective — transition is objective physics.
    """
    model = create_baseline(cfg, variant)
    model.eval()
    B = 3
    s = torch.randn(B, cfg.model.latent_dim)
    action = torch.zeros(B, cfg.env.N * cfg.env.A)
    s_next = model.transition(s, action)  # must NOT raise
    assert s_next.shape == (B, cfg.model.latent_dim)
    assert torch.isfinite(s_next).all()


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
