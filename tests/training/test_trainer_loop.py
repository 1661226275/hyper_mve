"""Pkg-05 spec 01 acceptance: MuZeroTrainer single train_step (C5-T1/T2/T3 + R5-1).

Run from repo root so ``hyper_mve`` imports; needs torch (+ pytest-mock for spies).
"""
import time

import pytest
import torch

from hyper_mve.configs import V4Config
from hyper_mve.models import HyperMuZeroModel
from hyper_mve.schemas import RelationObservationLayout
from hyper_mve.training import MuZeroTrainer
from hyper_mve.training.curriculum import CurriculumScheduler


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
    return V4Config.from_preset("rel_duo")


@pytest.fixture
def model(cfg_duo):
    return HyperMuZeroModel(cfg_duo)


@pytest.fixture
def trainer(cfg_duo, model):
    return MuZeroTrainer(cfg_duo, model, device=torch.device("cpu"))


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


# ====== C5-T1/T3: v5 6-API call order ======

def test_trainer_calls_update_step_per_step(trainer, cfg_duo, monkeypatch):
    calls = _spy(monkeypatch, trainer.model, "update_step")
    trainer.train_step(_make_batch(cfg_duo), global_step=100)
    assert len(calls) == 1
    assert calls[0][0][0] == 100


def test_no_objective_api(trainer):
    """v5: set_context_objective no longer exists on the model."""
    assert not hasattr(trainer.model, "set_context_objective")


def test_subjective_called_per_agent(trainer, cfg_duo, monkeypatch):
    calls = _spy(monkeypatch, trainer.model, "set_context_subjective")
    trainer.train_step(_make_batch(cfg_duo), global_step=100)
    assert len(calls) >= cfg_duo.env.N


# ====== scheduler injection (review 修订 1) ======

def test_scheduler_injection_default(cfg_duo, model):
    t = MuZeroTrainer(cfg_duo, model, device=torch.device("cpu"))
    assert isinstance(t.scheduler, CurriculumScheduler)


def test_scheduler_injection_custom(cfg_duo, model):
    custom = CurriculumScheduler(cfg_duo)
    trainer = MuZeroTrainer(cfg_duo, model, scheduler=custom)
    assert trainer.scheduler is custom


# ====== EMA target update ======

def test_target_model_ema_update(trainer, cfg_duo):
    # Perturb online so it clearly differs from target, then one train_step's EMA
    # must move at least one target parameter (robust to tiny per-step deltas).
    with torch.no_grad():
        for p in trainer.model.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    target_before = [p.detach().clone() for p in trainer.target_model.parameters()]
    trainer.train_step(_make_batch(cfg_duo), global_step=100)
    moved = any(
        not torch.allclose(tb, ta)
        for tb, ta in zip(target_before, trainer.target_model.parameters())
    )
    assert moved, "EMA should move the target model toward online"


# ====== n-step return ======

def test_n_step_return_correctness(trainer, cfg_duo):
    B, K, N = 2, cfg_duo.train.unroll_K, cfg_duo.env.N
    rewards = torch.full((B, K + 1, N), 1.0)
    v_target = torch.full((B, K + 1, N), 10.0)
    dones = torch.zeros(B, K + 1, dtype=torch.bool)
    n = cfg_duo.train.n_step
    z = trainer.compute_n_step_return(rewards, v_target, dones, n)
    n_eff = min(n, K)  # t=0 bootstrap horizon within the K+1 window
    gamma = cfg_duo.train.gamma
    expected = sum(gamma ** i for i in range(n_eff)) + gamma ** n_eff * 10.0
    assert torch.allclose(z[0, 0, 0], torch.tensor(expected), atol=1e-4)


def test_n_step_return_done_truncates(trainer, cfg_duo):
    B, K, N = 1, cfg_duo.train.unroll_K, cfg_duo.env.N
    rewards = torch.full((B, K + 1, N), 1.0)
    v_target = torch.full((B, K + 1, N), 10.0)
    dones = torch.zeros(B, K + 1, dtype=torch.bool)
    dones[:, 0] = True  # done immediately after reward at t=0
    z = trainer.compute_n_step_return(rewards, v_target, dones, cfg_duo.train.n_step)
    # only r_0 counts, no bootstrap (done at t=0)
    assert torch.allclose(z[0, 0, 0], torch.tensor(1.0), atol=1e-4)


# ====== checkpoint ======

def test_save_load_checkpoint_roundtrip(trainer, cfg_duo, tmp_path):
    path = str(tmp_path / "ckpt.pt")
    trainer.global_step = 12345
    trainer.save_checkpoint(path)
    trainer2 = MuZeroTrainer(cfg_duo, HyperMuZeroModel(cfg_duo), device=torch.device("cpu"))
    assert trainer2.load_checkpoint(path) == 12345
    assert trainer2.global_step == 12345


def test_load_v47_checkpoint_raises(trainer, tmp_path):
    path = str(tmp_path / "v47.pt")
    torch.save({"version": "v4.7", "model_state": {}}, path)
    with pytest.raises(RuntimeError, match="v4 与 v4.7 不兼容"):
        trainer.load_checkpoint(path)


# ====== R5-1: train_step < 400 ms ======

@pytest.mark.gpu
def test_train_step_under_400ms(cfg_duo):
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    model = HyperMuZeroModel(cfg_duo)
    trainer = MuZeroTrainer(cfg_duo, model, device=torch.device("cuda"))
    batch = _make_batch(cfg_duo, B=cfg_duo.train.batch_size, device="cuda")
    for _ in range(5):
        trainer.train_step(batch, global_step=0)
    torch.cuda.synchronize()
    times = []
    for step in range(20):
        t0 = time.perf_counter()
        trainer.train_step(batch, global_step=step)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    mean_ms = sum(times) / len(times)
    assert mean_ms < 400.0, f"train_step {mean_ms:.2f}ms > 400ms"
