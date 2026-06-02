"""Pkg-05 spec 05 acceptance: compose_total_loss double-path (C5-L1/L2).

The two gradient-isolation tests monkeypatch ``oracle_z_mixing_weight -> 0`` so the
main-path belief input is the *predicted* z (carrying grad). This isolates the
mechanism under test — the belief gradient GATING threshold (cfg.train.belief_grad_gating_steps,
default 5000) — from the (separate) curriculum stage boundary. This is the
real-API reconciliation noted in the plan (R2): BeliefNet.forward has no mixing arg.
"""
import pytest
import torch

from hyper_mve.configs import V4Config
from hyper_mve.models import HyperMuZeroModel
from hyper_mve.schemas import ObservationLayout
from hyper_mve.training import MuZeroTrainer
from hyper_mve.training.loss_composition import compose_total_loss


@pytest.fixture
def cfg_medium():
    return V4Config.from_preset("medium")


@pytest.fixture
def model(cfg_medium):
    return HyperMuZeroModel(cfg_medium)


@pytest.fixture
def trainer(cfg_medium, model):
    return MuZeroTrainer(cfg_medium, model)


def _make_batch(cfg, B=4, device="cpu"):
    N, A, K = cfg.env.N, cfg.env.A, cfg.train.unroll_K
    obs_dim = ObservationLayout.total_dim(N, cfg.env.K)
    return {
        "obs": torch.randn(B, K + 1, N, obs_dim, device=device),
        "actions": torch.randint(0, A, (B, K + 1, N), device=device),
        "rewards": torch.randn(B, K + 1, N, device=device),
        "c_t": torch.rand(B, K + 1, device=device),
        "cap": torch.rand(B, K + 1, N, 4, device=device),
        "c_hat": torch.rand(B, K + 1, N, device=device),
        "z_hat": torch.softmax(torch.randn(B, K + 1, N, N - 1, 2, device=device), dim=-1),
        "tau": torch.randint(0, 2, (B, K + 1, N), dtype=torch.int8, device=device),
        "pi_mve": torch.softmax(torch.randn(B, K + 1, N, A, device=device), dim=-1),
        "v": torch.randn(B, K + 1, N, device=device),
        "delta": torch.randn(B, K + 1, N, device=device),
        "dones": torch.zeros(B, K + 1, dtype=torch.bool, device=device),
    }


def _belief_grad_norm(model):
    return sum(
        p.grad.abs().sum().item()
        for p in model.belief_net.parameters() if p.grad is not None
    )


def _zero_belief_grads(model):
    for p in model.belief_net.parameters():
        p.grad = None


# ====== dict structure / NaN ======

def test_loss_composition_returns_full_dict(trainer, cfg_medium, model):
    losses = compose_total_loss(model, _make_batch(cfg_medium), trainer, 100, cfg_medium)
    assert set(losses) == {
        "total", "main", "belief", "lambda_b",
        "policy", "value", "reward", "consist",
        "belief_c", "belief_opp", "belief_div",
    }


def test_loss_composition_no_nan(trainer, cfg_medium, model):
    losses = compose_total_loss(model, _make_batch(cfg_medium), trainer, 100, cfg_medium)
    for k, v in losses.items():
        if torch.is_tensor(v):
            assert not torch.isnan(v).any(), f"NaN in loss['{k}']"


# ====== C5-L1: lambda_b at loss level ======

def test_lambda_b_curve_matches_cfg(trainer, cfg_medium, model):
    batch = _make_batch(cfg_medium)
    for step in [0, 50_000, 100_000, 200_000]:
        losses = compose_total_loss(model, batch, trainer, step, cfg_medium)
        assert torch.allclose(losses["lambda_b"], torch.tensor(float(trainer.scheduler.lambda_b(step))))


def test_lambda_b_zero_disables_belief_weight(trainer, cfg_medium, model, monkeypatch):
    monkeypatch.setattr(trainer.scheduler, "lambda_b", lambda step: 0.0)
    losses = compose_total_loss(model, _make_batch(cfg_medium), trainer, 100, cfg_medium)
    assert torch.allclose(losses["total"], losses["main"], atol=1e-6)
    assert losses["belief"].item() > 0.0


# ====== C5-L2: belief gradient gating (double path) ======

def test_belief_gradient_isolation_pre_5k(trainer, cfg_medium, model, monkeypatch):
    monkeypatch.setattr(trainer.scheduler, "oracle_z_mixing_weight", lambda step: 0.0)
    step = 1000  # < belief_grad_gating_steps (5000): main path detached from BeliefNet
    trainer.model.update_step(step)
    losses = compose_total_loss(model, _make_batch(cfg_medium), trainer, step, cfg_medium)

    _zero_belief_grads(model)
    losses["belief"].backward(retain_graph=True)
    belief_only = _belief_grad_norm(model)
    assert belief_only > 0.0, "L_belief must train BeliefNet"

    _zero_belief_grads(model)
    losses["total"].backward()
    total = _belief_grad_norm(model)

    # gating ON -> main path contributes no BeliefNet grad -> total ~= belief-only
    assert abs(total - belief_only) / max(belief_only, 1e-8) < 0.05


def test_belief_gradient_both_sources_post_5k(trainer, cfg_medium, model, monkeypatch):
    monkeypatch.setattr(trainer.scheduler, "oracle_z_mixing_weight", lambda step: 0.0)
    step = 10_000  # >= 5000: gating off, main path also trains BeliefNet
    trainer.model.update_step(step)
    losses = compose_total_loss(model, _make_batch(cfg_medium), trainer, step, cfg_medium)

    _zero_belief_grads(model)
    losses["belief"].backward(retain_graph=True)
    belief_only = _belief_grad_norm(model)

    _zero_belief_grads(model)
    losses["total"].backward()
    total = _belief_grad_norm(model)

    assert total > belief_only * 1.05, (
        f"post-5k BeliefNet grad should include main+belief: total={total:.6f} "
        f"belief_only={belief_only:.6f}"
    )
