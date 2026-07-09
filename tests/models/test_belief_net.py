"""Unit tests for ``hyper_mve.models.belief_net`` (v5 Pkg-09: GRU trunk + head_regime)."""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from hyper_mve.configs import V4Config
from hyper_mve.models import BeliefNet
from hyper_mve.schemas import get_regime_family


@pytest.fixture
def cfg_duo():
    return V4Config.from_preset("rel_duo")


@pytest.fixture
def cfg_quad():
    base = V4Config.from_preset("rel_duo")
    env = replace(base.env, N=4, K=20, relation_family="g4",
                  type_assignment=base.env.type_assignment * 2)
    return replace(base, env=env)


# ====== GRU trunk ======

def test_init_hidden_zeros(cfg_quad):
    bn = BeliefNet(cfg_quad.env, cfg_quad.model)
    hidden = bn.init_hidden(batch_size=2, num_agents=4)
    assert hidden.shape == (2, 4, 128)
    assert torch.all(hidden == 0)


def test_obs_dim_from_relation_layout(cfg_duo):
    """BeliefNet sizes its obs encoder from RelationObservationLayout."""
    bn = BeliefNet(cfg_duo.env, cfg_duo.model)
    assert bn.obs_dim == 39            # N=2, K=8: 5 + 3K + 10(N-1)
    assert bn.gru_input_dim == 64
    assert bn.gru_hidden == cfg_duo.model.belief_gru_hidden
    assert bn.n_regimes == get_regime_family(cfg_duo.env).size == 5


def test_gru_step_seq_equiv(cfg_quad):
    """step-by-step ≈ forward(seq)."""
    bn = BeliefNet(cfg_quad.env, cfg_quad.model)
    bn.eval()

    B, T, N = 2, 5, 4
    obs_seq = torch.randn(B, T, N, bn.obs_dim)
    init_h = bn.init_hidden(B, N)

    hidden_seq, g_hat_seq = bn.forward(obs_seq, init_hidden=init_h)

    h = init_h
    hs, gs = [], []
    for t in range(T):
        h, g = bn.step(obs_seq[:, t], h)
        hs.append(h)
        gs.append(g)

    assert torch.allclose(hidden_seq, torch.stack(hs, dim=1), atol=1e-5)
    assert torch.allclose(g_hat_seq, torch.stack(gs, dim=1), atol=1e-5)


def test_hidden_state_evolution(cfg_quad):
    bn = BeliefNet(cfg_quad.env, cfg_quad.model)
    bn.eval()
    B, N = 1, 4
    obs_t = torch.randn(B, N, bn.obs_dim)
    h0 = bn.init_hidden(B, N)
    h1, _ = bn.step(obs_t, h0)
    assert not torch.allclose(h1, h0)


def test_shared_gru_weights(cfg_quad):
    """One GRUCell shared across agents; same (obs, hidden) ⇒ same output."""
    bn = BeliefNet(cfg_quad.env, cfg_quad.model)
    assert isinstance(bn.gru, torch.nn.GRUCell)
    B = 1
    same_obs = torch.randn(B, 1, bn.obs_dim)
    same_hidden = torch.randn(B, 1, 128)
    out_a, _ = bn.step(same_obs, same_hidden)
    obs_pair = torch.cat([same_obs, torch.randn(B, 1, bn.obs_dim)], dim=1)
    hidden_pair = torch.cat([same_hidden, torch.randn(B, 1, 128)], dim=1)
    out_pair, _ = bn.step(obs_pair, hidden_pair)
    assert torch.allclose(out_a[:, 0], out_pair[:, 0], atol=1e-5)


def test_layernorm_after_gru(cfg_duo):
    bn = BeliefNet(cfg_duo.env, cfg_duo.model)
    assert isinstance(bn.ln_belief, torch.nn.LayerNorm)


def test_causality(cfg_quad):
    """uni-directional: t=0 output independent of future obs."""
    bn = BeliefNet(cfg_quad.env, cfg_quad.model)
    B, N = 1, 4
    obs_a = torch.randn(B, 5, N, bn.obs_dim)
    obs_b = obs_a.clone()
    obs_b[:, 3:] = torch.randn(B, 2, N, bn.obs_dim)

    h0 = bn.init_hidden(B, N)
    h_a_t0, _ = bn.step(obs_a[:, 0], h0)
    h_b_t0, _ = bn.step(obs_b[:, 0], h0)
    assert torch.allclose(h_a_t0, h_b_t0)


# ====== head_regime ======

def test_head_regime_output_shape(cfg_duo):
    """head_regime outputs (B, N, |G|) softmax posterior."""
    bn = BeliefNet(cfg_duo.env, cfg_duo.model)
    B, N = 2, 2
    obs_t = torch.randn(B, N, bn.obs_dim)
    _, g_hat = bn.step(obs_t, bn.init_hidden(B, N))

    assert g_hat.shape == (B, N, 5)
    assert (g_hat >= 0).all()
    sums = g_hat.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_head_regime_size_follows_family(cfg_quad):
    """|G| output width follows the preset's regime family (g4 ⇒ 5; g4_ext ⇒ 9)."""
    bn4 = BeliefNet(cfg_quad.env, cfg_quad.model)
    assert bn4.n_regimes == 5
    env_ext = replace(cfg_quad.env, relation_family="g4_ext")
    bn9 = BeliefNet(env_ext, cfg_quad.model)
    _, g_hat = bn9.step(torch.randn(1, 4, bn9.obs_dim), bn9.init_hidden(1, 4))
    assert g_hat.shape == (1, 4, 9)


def test_get_head_regime_predictions_consistency(cfg_duo):
    """Recomputing posteriors from stored hiddens matches forward's output."""
    bn = BeliefNet(cfg_duo.env, cfg_duo.model)
    bn.eval()
    B, T, N = 2, 4, 2
    obs_seq = torch.randn(B, T, N, bn.obs_dim)
    hidden_seq, g_hat_seq = bn.forward(obs_seq)
    g_pred = bn.get_head_regime_predictions(hidden_seq)
    assert torch.allclose(g_pred, g_hat_seq, atol=1e-6)


def test_head_regime_gradient_flow(cfg_duo):
    bn = BeliefNet(cfg_duo.env, cfg_duo.model)
    obs_seq = torch.randn(1, 3, 2, bn.obs_dim)
    _, g_hat_seq = bn.forward(obs_seq)
    g_hat_seq.sum().backward()
    has_grad = any(p.grad is not None and p.grad.norm() > 0
                   for p in bn.head_regime.parameters())
    assert has_grad
