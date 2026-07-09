"""Pkg-05 spec 07 acceptance: EMA target + LR scheduler (C5-E1/E2)."""
from dataclasses import replace

import pytest
import torch

from hyper_mve.configs import V4Config
from hyper_mve.models import HyperMuZeroModel
from hyper_mve.schemas import RelationObservationLayout
from hyper_mve.training import MuZeroTrainer


@pytest.fixture
def cfg_duo():
    return V4Config.from_preset("rel_duo")


@pytest.fixture
def trainer(cfg_duo):
    return MuZeroTrainer(cfg_duo, HyperMuZeroModel(cfg_duo), device=torch.device("cpu"))


_G = 5


def _make_batch(cfg, B=4, device="cpu"):
    N, A, K = cfg.env.N, cfg.env.A, cfg.train.unroll_K
    obs_dim = RelationObservationLayout.total_dim(N, cfg.env.K)
    return {
        "obs": torch.randn(B, K + 1, N, obs_dim, device=device),
        "actions": torch.randint(0, A, (B, K + 1, N), device=device),
        "rewards": torch.randn(B, K + 1, N, device=device),
        "row": torch.rand(B, K + 1, N, N - 1, device=device) * 2 - 1,
        "g_hat": torch.softmax(torch.randn(B, K + 1, N, _G, device=device), dim=-1),
        "g": torch.randint(0, _G, (B, K + 1), device=device),
        "pi_mve": torch.softmax(torch.randn(B, K + 1, N, A, device=device), dim=-1),
        "v": torch.randn(B, K + 1, N, device=device),
        "dones": torch.zeros(B, K + 1, dtype=torch.bool, device=device),
    }


# ====== C5-E1: EMA tau=0.99 decay ======

def test_ema_decay_correctness_tau_099(trainer, cfg_duo):
    tau = cfg_duo.train.ema_tau  # 0.99
    for p in trainer.model.parameters():
        p.data.fill_(1.0)
    for p in trainer.target_model.parameters():
        p.data.fill_(0.0)
    trainer._update_ema_target()
    for p in trainer.target_model.parameters():
        assert torch.allclose(p, torch.full_like(p, 1.0 - tau), atol=1e-6)


def test_ema_converges_to_online(trainer):
    for p in trainer.model.parameters():
        p.data.fill_(1.0)
    for p in trainer.target_model.parameters():
        p.data.fill_(0.0)
    for _ in range(500):
        trainer._update_ema_target()
    for p in trainer.target_model.parameters():
        assert p.mean().item() > 0.99


def test_ema_does_not_modify_online(trainer):
    online_p0 = next(trainer.model.parameters()).detach().clone()
    trainer._update_ema_target()
    assert torch.allclose(online_p0, next(trainer.model.parameters()))


# ====== C5-E2: warmup_cosine LR curve (small max for a fast test) ======

def test_lr_warmup_then_cosine_anneal(cfg_duo):
    cfg = replace(cfg_duo, train=replace(cfg_duo.train, max_train_steps=20_000))
    trainer = MuZeroTrainer(cfg, HyperMuZeroModel(cfg), device=torch.device("cpu"))
    sched = trainer.lr_scheduler
    t = cfg.train

    for _ in range(t.lr_warmup_steps):
        sched.step()
    lr_after_warmup = trainer.optimizer.param_groups[0]["lr"]
    assert abs(lr_after_warmup - t.lr) / t.lr < 0.05

    for _ in range(t.max_train_steps - t.lr_warmup_steps):
        sched.step()
    lr_final = trainer.optimizer.param_groups[0]["lr"]
    assert abs(lr_final - t.lr_min) / t.lr_min < 0.05


def test_lr_monotonic_after_warmup(cfg_duo):
    cfg = replace(cfg_duo, train=replace(cfg_duo.train, max_train_steps=20_000))
    trainer = MuZeroTrainer(cfg, HyperMuZeroModel(cfg), device=torch.device("cpu"))
    sched = trainer.lr_scheduler
    for _ in range(cfg.train.lr_warmup_steps):
        sched.step()
    lrs = []
    for _ in range(100):
        sched.step()
        lrs.append(trainer.optimizer.param_groups[0]["lr"])
    for lr0, lr1 in zip(lrs[:-1], lrs[1:]):
        assert lr0 >= lr1 - 1e-12


# ====== target subjective context synced (spec 05 §3.3, v5) ======

def test_target_model_set_context_synced(trainer, cfg_duo):
    trainer.train_step(_make_batch(cfg_duo), global_step=100)
    # the V-bootstrap loop set the target model's subjective θ
    assert trainer.target_model._theta_rew is not None
    assert trainer.target_model._theta_rew.shape == trainer.model._theta_rew.shape
