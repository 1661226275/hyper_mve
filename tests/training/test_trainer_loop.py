"""Pkg-05 spec 01 acceptance: MuZeroTrainer single train_step (C5-T1/T2/T3 + R5-1).

Run from repo root so ``hyper_mve`` imports; needs torch (+ pytest-mock for spies).
"""
import time

import pytest
import torch

from hyper_mve.configs import V4Config
from hyper_mve.models import HyperMuZeroModel
from hyper_mve.schemas import ObservationLayout
from hyper_mve.training import MuZeroTrainer
from hyper_mve.training.curriculum import CurriculumScheduler


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


# ====== C5-T1/T2/T3: 7-API call order ======

def test_trainer_calls_update_step_per_step(trainer, cfg_medium, mocker):
    spy = mocker.spy(trainer.model, "update_step")
    trainer.train_step(_make_batch(cfg_medium), global_step=100)
    assert spy.call_count == 1
    assert spy.call_args[0][0] == 100


def test_objective_called_once_per_unroll(trainer, cfg_medium, mocker):
    spy = mocker.spy(trainer.model, "set_context_objective")
    trainer.train_step(_make_batch(cfg_medium), global_step=100)
    assert spy.call_count == 1


def test_subjective_called_per_agent(trainer, cfg_medium, mocker):
    spy = mocker.spy(trainer.model, "set_context_subjective")
    trainer.train_step(_make_batch(cfg_medium), global_step=100)
    assert spy.call_count >= cfg_medium.env.N


# ====== scheduler injection (review 修订 1) ======

def test_scheduler_injection_default(cfg_medium, model):
    assert isinstance(MuZeroTrainer(cfg_medium, model).scheduler, CurriculumScheduler)


def test_scheduler_injection_custom(cfg_medium, model):
    custom = CurriculumScheduler(cfg_medium)
    trainer = MuZeroTrainer(cfg_medium, model, scheduler=custom)
    assert trainer.scheduler is custom


# ====== EMA target update ======

def test_target_model_ema_update(trainer, cfg_medium):
    online_p0 = next(trainer.model.parameters()).detach().clone()
    target_p0 = next(trainer.target_model.parameters()).detach().clone()
    trainer.train_step(_make_batch(cfg_medium), global_step=100)
    target_p1 = next(trainer.target_model.parameters()).detach().clone()
    # target moved (online changed via optimizer; EMA pulled target toward it)
    assert not torch.allclose(target_p0, target_p1)
    # and online was not just copied wholesale into target
    assert not torch.allclose(target_p1, next(trainer.model.parameters()))
    _ = online_p0  # online snapshot kept for clarity


# ====== n-step return ======

def test_n_step_return_correctness(trainer, cfg_medium):
    B, K, N = 2, cfg_medium.train.unroll_K, cfg_medium.env.N
    rewards = torch.full((B, K + 1, N), 1.0)
    v_target = torch.full((B, K + 1, N), 10.0)
    dones = torch.zeros(B, K + 1, dtype=torch.bool)
    n = cfg_medium.train.n_step
    z = trainer.compute_n_step_return(rewards, v_target, dones, n)
    n_eff = min(n, K)  # t=0 bootstrap horizon within the K+1 window
    gamma = cfg_medium.train.gamma
    expected = sum(gamma ** i for i in range(n_eff)) + gamma ** n_eff * 10.0
    assert torch.allclose(z[0, 0, 0], torch.tensor(expected), atol=1e-4)


def test_n_step_return_done_truncates(trainer, cfg_medium):
    B, K, N = 1, cfg_medium.train.unroll_K, cfg_medium.env.N
    rewards = torch.full((B, K + 1, N), 1.0)
    v_target = torch.full((B, K + 1, N), 10.0)
    dones = torch.zeros(B, K + 1, dtype=torch.bool)
    dones[:, 0] = True  # done immediately after reward at t=0
    z = trainer.compute_n_step_return(rewards, v_target, dones, cfg_medium.train.n_step)
    # only r_0 counts, no bootstrap (done at t=0)
    assert torch.allclose(z[0, 0, 0], torch.tensor(1.0), atol=1e-4)


# ====== checkpoint ======

def test_save_load_checkpoint_roundtrip(trainer, cfg_medium, tmp_path):
    path = str(tmp_path / "ckpt.pt")
    trainer.global_step = 12345
    trainer.save_checkpoint(path)
    trainer2 = MuZeroTrainer(cfg_medium, HyperMuZeroModel(cfg_medium))
    assert trainer2.load_checkpoint(path) == 12345
    assert trainer2.global_step == 12345


def test_load_v47_checkpoint_raises(trainer, tmp_path):
    path = str(tmp_path / "v47.pt")
    torch.save({"version": "v4.7", "model_state": {}}, path)
    with pytest.raises(RuntimeError, match="v4 与 v4.7 不兼容"):
        trainer.load_checkpoint(path)


# ====== R5-1: train_step < 400 ms ======

@pytest.mark.gpu
def test_train_step_under_400ms(cfg_medium):
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    model = HyperMuZeroModel(cfg_medium)
    trainer = MuZeroTrainer(cfg_medium, model, device=torch.device("cuda"))
    batch = _make_batch(cfg_medium, B=cfg_medium.train.batch_size, device="cuda")
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
