"""Pkg-04 spec 04 acceptance tests: belief gradient gating (unit level).

The former HyperMuZeroModel integration section was retired with the v5
stack; gating-inside-the-model coverage now lives in the mazero_mixed fork
(subjective_model.py applies apply_raw/apply_ctx, exercised by its smoke
gates).
"""
import pytest
import torch

from hyper_mve.algo.modules.grad_gating import BeliefGradGating

_G = 5


# ====== BeliefGradGating unit ======

def test_grad_gating_detach_before_warmup():
    gating = BeliefGradGating(num_warmup_steps=5000)
    g_hat = torch.softmax(torch.randn(1, _G, requires_grad=True), dim=-1)
    g_out = gating.apply(g_hat, step=1000)
    assert g_out.requires_grad is False


def test_grad_gating_passthrough_after_warmup():
    gating = BeliefGradGating(num_warmup_steps=5000)
    g_hat = torch.softmax(torch.randn(1, _G), dim=-1).requires_grad_(True)
    g_out = gating.apply(g_hat, step=10000)
    assert g_out.requires_grad is True


def test_grad_gating_boundary_step_eq_warmup():
    gating = BeliefGradGating(num_warmup_steps=5000)
    g_hat = torch.softmax(torch.randn(1, _G), dim=-1).requires_grad_(True)
    g_out = gating.apply(g_hat, step=5000)
    assert g_out.requires_grad is True


def test_apply_ctx_detaches_only_belief_slice():
    """v5 ctx_aug = [role (0:32) | belief (32:64)]."""
    gating = BeliefGradGating(num_warmup_steps=5000)
    ctx = torch.randn(2, 64, requires_grad=True)
    out = gating.apply_ctx(ctx, step=1000, belief_slice=(32, 64))
    (out[..., :32].sum() + out[..., 32:].sum()).backward()
    assert ctx.grad is not None
    assert ctx.grad[..., :32].abs().sum().item() > 0
    assert ctx.grad[..., 32:64].abs().sum().item() == 0.0


def test_apply_ctx_passthrough_after_warmup():
    gating = BeliefGradGating(num_warmup_steps=5000)
    ctx = torch.randn(2, 64, requires_grad=True)
    out = gating.apply_ctx(ctx, step=10000, belief_slice=(32, 64))
    out[..., 32:].sum().backward()
    assert ctx.grad[..., 32:64].abs().sum().item() > 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
