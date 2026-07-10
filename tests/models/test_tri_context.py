"""Unit tests for ``hyper_mve.models.tri_context_encoder`` + ``belief_encoder``
(v5 Pkg-09: role + belief dual pathway, ctx_aug = 64)."""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from hyper_mve.configs import V4Config
from hyper_mve.models import TriContextEncoder, BeliefEncoder

_G = 5


@pytest.fixture
def cfg_duo():
    return V4Config.from_preset("rel_duo")


@pytest.fixture
def cfg_quad(cfg_duo):
    env = replace(cfg_duo.env, N=4, K=20, relation_family="g4")
    return replace(cfg_duo, env=env)


def _inputs(B, N):
    agent_ids = torch.arange(N).unsqueeze(0).expand(B, N).contiguous()
    rows = torch.rand(B, N, N - 1) * 2 - 1
    g_hat = torch.softmax(torch.randn(B, N, _G), dim=-1)
    return agent_ids, rows, g_hat


def test_output_dim(cfg_quad):
    """ctx_i shape (B, N, 64) exact (v5)."""
    encoder = TriContextEncoder(cfg_quad.env, cfg_quad.model)
    ctx_i = encoder.forward(*_inputs(2, 4))
    assert ctx_i.shape == (2, 4, 64)


def test_d_ctx_aug_property(cfg_duo):
    """d_ctx_aug = 32 + 32 = 64 (c path removed in v5)."""
    encoder = TriContextEncoder(cfg_duo.env, cfg_duo.model)
    assert encoder.d_ctx_aug == 64
    assert encoder.d_role == 32
    assert encoder.d_belief == 32
    assert not hasattr(encoder, "c_encoder")
    assert not hasattr(encoder, "forward_c_ctx_only")


def test_role_dim_exact(cfg_duo):
    """role = id (8) + row (24) = 32 exact fill."""
    encoder = TriContextEncoder(cfg_duo.env, cfg_duo.model)
    assert cfg_duo.model.d_id_emb + cfg_duo.model.d_row_emb == 32
    assert encoder.role_encoder.d_role == 32


def test_gradient_flow(cfg_quad):
    encoder = TriContextEncoder(cfg_quad.env, cfg_quad.model)
    agent_ids, rows, _ = _inputs(2, 4)
    g_hat = torch.softmax(torch.randn(2, 4, _G, requires_grad=True), dim=-1)
    ctx_i = encoder.forward(agent_ids, rows, g_hat)
    (ctx_i ** 2).sum().backward()
    for name, p in encoder.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No grad for {name}"
            assert p.grad.norm() > 0, f"Zero grad for {name}"


def test_concat_order(cfg_quad):
    """ctx_i slices: [0:32] role, [32:64] belief."""
    encoder = TriContextEncoder(cfg_quad.env, cfg_quad.model)
    agent_ids, rows, g_hat = _inputs(2, 4)
    ctx_i = encoder.forward(agent_ids, rows, g_hat)

    role_alone = encoder.ln_role(encoder.role_encoder(agent_ids, rows))
    belief_alone = encoder.ln_belief(encoder.belief_encoder(g_hat))
    assert torch.allclose(ctx_i[..., :32], role_alone, atol=1e-6)
    assert torch.allclose(ctx_i[..., 32:], belief_alone, atol=1e-6)


def test_belief_change_leaves_role_segment(cfg_duo):
    """Varying the posterior only moves the belief slice."""
    encoder = TriContextEncoder(cfg_duo.env, cfg_duo.model)
    agent_ids, rows, g_hat = _inputs(1, 2)
    ctx_a = encoder.forward(agent_ids, rows, g_hat)
    ctx_b = encoder.forward(agent_ids, rows, torch.softmax(torch.randn(1, 2, _G), dim=-1))
    assert torch.allclose(ctx_a[..., :32], ctx_b[..., :32])
    assert not torch.allclose(ctx_a[..., 32:], ctx_b[..., 32:])


def test_n_variation(cfg_duo):
    """N=2 (g2) and N=4 (g4_ext, |G|=9) both run."""
    enc2 = TriContextEncoder(cfg_duo.env, cfg_duo.model)
    ctx2 = enc2.forward(*_inputs(1, 2))
    assert ctx2.shape == (1, 2, 64)

    env4 = replace(cfg_duo.env, N=4, K=20, relation_family="g4_ext")
    enc4 = TriContextEncoder(env4, cfg_duo.model)
    agent_ids = torch.arange(4).unsqueeze(0)
    rows = torch.rand(1, 4, 3)
    g_hat = torch.softmax(torch.randn(1, 4, 9), dim=-1)
    assert enc4.forward(agent_ids, rows, g_hat).shape == (1, 4, 64)


# ====== BeliefEncoder standalone ======

def test_belief_encoder_output_shape(cfg_quad):
    be = BeliefEncoder(cfg_quad.env, cfg_quad.model)
    g_hat = torch.softmax(torch.randn(2, 4, _G), dim=-1)
    assert be(g_hat).shape == (2, 4, 32)


def test_belief_encoder_rejects_wrong_g_width(cfg_duo):
    be = BeliefEncoder(cfg_duo.env, cfg_duo.model)
    with pytest.raises(AssertionError, match="g_hat last dim"):
        be(torch.softmax(torch.randn(1, 2, 3), dim=-1))


def test_belief_encoder_no_internal_ln(cfg_duo):
    """P7: BeliefEncoder contains no internal LayerNorm."""
    be = BeliefEncoder(cfg_duo.env, cfg_duo.model)
    has_ln = any(isinstance(m, torch.nn.LayerNorm) for m in be.modules())
    assert not has_ln


def test_belief_encoder_gradient_flow(cfg_duo):
    be = BeliefEncoder(cfg_duo.env, cfg_duo.model)
    g_hat = torch.softmax(torch.randn(2, 2, _G, requires_grad=True), dim=-1)
    (be(g_hat) ** 2).sum().backward()
    for name, p in be.named_parameters():
        assert p.grad is not None and p.grad.norm() > 0, f"No/zero grad for {name}"
