"""Pkg-05 spec 03 acceptance: EpisodeReplayBuffer (C5-B1/B2 + R5-2)."""
import pickle
import time
from dataclasses import replace

import numpy as np
import pytest
import torch

from hyper_mve.configs import V4Config
from hyper_mve.schemas import AgentType, ObservationLayout, TimeStepRecord
from hyper_mve.training import EpisodeReplayBuffer


@pytest.fixture
def cfg_medium():
    return V4Config.from_preset("medium")


@pytest.fixture
def dims(cfg_medium):
    N = cfg_medium.env.N
    A = cfg_medium.env.A
    obs_dim = ObservationLayout.total_dim(N, cfg_medium.env.K)
    return N, A, obs_dim


@pytest.fixture
def buffer(cfg_medium):
    return EpisodeReplayBuffer(cfg_medium)


def _types(N, kind="mixed"):
    if kind == "alpha":
        return [int(AgentType.ALPHA)] * N
    if kind == "beta":
        return [int(AgentType.BETA)] * N
    half = N // 2
    return [int(AgentType.ALPHA)] * half + [int(AgentType.BETA)] * (N - half)


def _make_record(N, A, obs_dim, t=0, types=None):
    types = types or _types(N)
    return TimeStepRecord(
        o=np.zeros((N, obs_dim), dtype=np.float32),
        a=np.zeros(N, dtype=np.int64),
        r=np.zeros(N, dtype=np.float32),
        delta=np.zeros(N, dtype=np.float32),
        pi_mve=np.full((N, A), 1.0 / A, dtype=np.float32),
        v=np.zeros(N, dtype=np.float32),
        tau=np.array(types, dtype=np.int8),
        cap=np.zeros((N, 4), dtype=np.float32),
        c_hat=np.full(N, 0.5, dtype=np.float32),
        z_hat=np.full((N, N - 1, 2), 0.5, dtype=np.float32),
        t=t, done=False,
    )


def _make_episode(dims, T=50, types=None):
    N, A, obs_dim = dims
    records = [_make_record(N, A, obs_dim, t=i, types=types) for i in range(T)]
    return records, torch.full((T,), 0.5)


# ====== store ======

def test_store_episode_basic(buffer, dims):
    buffer.store_episode(*_make_episode(dims, T=50))
    assert len(buffer) == 1


def test_store_episode_too_short_raises(buffer, dims, cfg_medium):
    T = cfg_medium.train.unroll_K + cfg_medium.train.n_step
    with pytest.raises(AssertionError, match="episode too short"):
        buffer.store_episode(*_make_episode(dims, T=T))


def test_store_episode_c_t_seq_length_mismatch(buffer, dims):
    records, _ = _make_episode(dims, T=50)
    with pytest.raises(AssertionError, match="c_t_seq len"):
        buffer.store_episode(records, torch.full((30,), 0.5))


# ====== sample shapes ======

def test_sample_batch_shape(buffer, dims, cfg_medium):
    N, A, obs_dim = dims
    for _ in range(40):
        buffer.store_episode(*_make_episode(dims, T=50))
    K = 5
    b = buffer.sample_batch(batch_size=16, unroll_K=K)
    assert b["obs"].shape == (16, K + 1, N, obs_dim)
    assert b["actions"].shape == (16, K + 1, N)
    assert b["rewards"].shape == (16, K + 1, N)
    assert b["delta"].shape == (16, K + 1, N)
    assert b["pi_mve"].shape == (16, K + 1, N, A)
    assert b["v"].shape == (16, K + 1, N)
    assert b["tau"].shape == (16, K + 1, N)
    assert b["cap"].shape == (16, K + 1, N, 4)
    assert b["c_hat"].shape == (16, K + 1, N)
    assert b["z_hat"].shape == (16, K + 1, N, N - 1, 2)
    assert b["t"].shape == (16, K + 1)
    assert b["dones"].shape == (16, K + 1)
    assert b["c_t"].shape == (16, K + 1)


# ====== C5-B1: z_hat order preserved end-to-end ======

def test_buffer_z_hat_order_e2e(buffer, dims):
    N, A, obs_dim = dims
    records = []
    for t in range(50):
        r = _make_record(N, A, obs_dim, t=t)
        for i in range(N):
            for k in range(N - 1):
                r.z_hat[i, k, 0] = i * 10 + k
        records.append(r)
    buffer.store_episode(records, torch.full((50,), 0.5))

    b = buffer.sample_batch(batch_size=1, unroll_K=5)
    z = b["z_hat"]  # (1, K+1, N, N-1, 2)
    for t in range(6):
        for i in range(N):
            for k in range(N - 1):
                assert z[0, t, i, k, 0].item() == i * 10 + k


# ====== C5-B2: stratified min-per-type fraction ======

def test_stratified_min_per_type_frac(buffer, dims):
    for _ in range(50):
        buffer.store_episode(*_make_episode(dims, T=50, types=_types(dims[0], "alpha")))
    for _ in range(50):
        buffer.store_episode(*_make_episode(dims, T=50, types=_types(dims[0], "beta")))

    b = buffer.sample_batch(batch_size=100, unroll_K=5)
    tau_t0 = b["tau"][:, 0, 0]
    alpha = int((tau_t0 == int(AgentType.ALPHA)).sum())
    beta = int((tau_t0 == int(AgentType.BETA)).sum())
    assert alpha >= int(100 * 0.3)
    assert beta >= int(100 * 0.3)


# ====== R5-2: sample_batch < 50 ms ======

def test_sample_batch_under_50ms(buffer, dims):
    for _ in range(300):  # enough variety for B=256 (sample_batch has no min-buffer gate)
        buffer.store_episode(*_make_episode(dims, T=50))
    for _ in range(5):
        buffer.sample_batch(batch_size=256, unroll_K=5)
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        buffer.sample_batch(batch_size=256, unroll_K=5)
        times.append((time.perf_counter() - t0) * 1000)
    mean_ms = sum(times) / len(times)
    assert mean_ms < 50.0, f"sample_batch {mean_ms:.2f}ms > 50ms"


# ====== v4.7 -> v4 field mapping ======

def test_v47_episodedata_v4_record_field_mapping(buffer, dims):
    buffer.store_episode(*_make_episode(dims, T=50))
    b = buffer.sample_batch(batch_size=4, unroll_K=5)
    for f in ("obs", "actions", "rewards", "pi_mve", "c_t", "dones"):  # v4.7 6 fields
        assert f in b
    for f in ("delta", "v", "tau", "cap", "c_hat", "z_hat", "t"):       # v4 new
        assert f in b


# ====== pickle roundtrip (R5-7) ======

def test_buffer_pickle_roundtrip(buffer, dims):
    buffer.store_episode(*_make_episode(dims, T=50))
    buffer2 = pickle.loads(pickle.dumps(buffer))
    assert len(buffer2) == 1
    assert buffer2.sample_batch(batch_size=1, unroll_K=5)["obs"].shape[0] == 1


# ====== FIFO eviction ======

def test_fifo_buffer_eviction(cfg_medium, dims):
    cfg_small = replace(cfg_medium, train=replace(cfg_medium.train, buffer_size=5))
    buffer = EpisodeReplayBuffer(cfg_small)
    for _ in range(10):
        buffer.store_episode(*_make_episode(dims, T=50))
    assert len(buffer) == 5
