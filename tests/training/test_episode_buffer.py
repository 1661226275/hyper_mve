"""Pkg-05 spec 03 acceptance: EpisodeReplayBuffer (C5-B1/B2 + R5-2, v5 Pkg-09)."""
import pickle
import time
from dataclasses import replace

import numpy as np
import pytest

from hyper_mve.configs import V4Config
from hyper_mve.schemas import RelationObservationLayout, TimeStepRecord
from hyper_mve.training import EpisodeReplayBuffer

_G = 5  # |G| for g2


@pytest.fixture
def cfg_duo():
    return V4Config.from_preset("rel_duo")


@pytest.fixture
def dims(cfg_duo):
    N = cfg_duo.env.N
    A = cfg_duo.env.A
    obs_dim = RelationObservationLayout.total_dim(N, cfg_duo.env.K)
    return N, A, obs_dim


@pytest.fixture
def buffer(cfg_duo):
    return EpisodeReplayBuffer(cfg_duo)


def _make_record(N, A, obs_dim, t=0, g=0):
    return TimeStepRecord(
        o=np.zeros((N, obs_dim), dtype=np.float32),
        a=np.zeros(N, dtype=np.int64),
        r=np.zeros(N, dtype=np.float32),
        pi_mve=np.full((N, A), 1.0 / A, dtype=np.float32),
        v=np.zeros(N, dtype=np.float32),
        row=np.zeros((N, N - 1), dtype=np.float32),
        g_hat=np.full((N, _G), 1.0 / _G, dtype=np.float32),
        g=g, t=t, done=False,
    )


def _make_episode(dims, T=50, g=0):
    N, A, obs_dim = dims
    return [_make_record(N, A, obs_dim, t=i, g=g) for i in range(T)]


# ====== store ======

def test_store_episode_basic(buffer, dims):
    buffer.store_episode(_make_episode(dims, T=50))
    assert len(buffer) == 1


def test_store_episode_too_short_raises(buffer, dims, cfg_duo):
    T = cfg_duo.train.unroll_K + cfg_duo.train.n_step
    with pytest.raises(AssertionError, match="episode too short"):
        buffer.store_episode(_make_episode(dims, T=T))


# ====== sample shapes (v5 batch keys) ======

def test_sample_batch_shape(buffer, dims):
    N, A, obs_dim = dims
    for _ in range(40):
        buffer.store_episode(_make_episode(dims, T=50))
    K = 5
    b = buffer.sample_batch(batch_size=16, unroll_K=K)
    assert b["obs"].shape == (16, K + 1, N, obs_dim)
    assert b["actions"].shape == (16, K + 1, N)
    assert b["rewards"].shape == (16, K + 1, N)
    assert b["pi_mve"].shape == (16, K + 1, N, A)
    assert b["v"].shape == (16, K + 1, N)
    assert b["row"].shape == (16, K + 1, N, N - 1)
    assert b["g_hat"].shape == (16, K + 1, N, _G)
    assert b["g"].shape == (16, K + 1)
    assert b["g"].dtype.is_floating_point is False
    assert b["t"].shape == (16, K + 1)
    assert b["dones"].shape == (16, K + 1)
    assert "c_t" not in b            # v5: c machinery removed
    assert b["planner_on"].shape == (16,)
    assert b["collected_at_step"].shape == (16,)


# ====== C5-B1: row/g_hat layout preserved end-to-end ======

def test_buffer_row_ghat_order_e2e(buffer, dims):
    N, A, obs_dim = dims
    records = []
    for t in range(50):
        r = _make_record(N, A, obs_dim, t=t, g=3)
        for i in range(N):
            r.row[i, :] = i * 10 + t % 7
            r.g_hat[i, :] = 0.0
            r.g_hat[i, i % _G] = 1.0
        records.append(r)
    buffer.store_episode(records)

    b = buffer.sample_batch(batch_size=1, unroll_K=5)
    assert (b["g"] == 3).all()
    for tt in range(6):
        for i in range(N):
            assert b["g_hat"][0, tt, i].argmax().item() == i % _G


# ====== C5-B2: regime-stratified sampling ======

def test_stratified_min_per_regime(buffer, dims):
    for g in (0, 1, 4):
        for _ in range(30):
            buffer.store_episode(_make_episode(dims, T=50, g=g))

    b = buffer.sample_batch(batch_size=90, unroll_K=5)
    g0 = b["g"][:, 0]
    # min_per_bucket = max(1, 90*0.3/3) = 9 from each present regime
    for g in (0, 1, 4):
        assert int((g0 == g).sum()) >= 9, f"regime {g} under-represented"


def test_stratified_many_buckets_trims_to_batch(cfg_duo, dims):
    buffer = EpisodeReplayBuffer(cfg_duo)
    for g in range(_G):
        buffer.store_episode(_make_episode(dims, T=50, g=g))
    b = buffer.sample_batch(batch_size=4, unroll_K=5)   # fewer than buckets
    assert b["obs"].shape[0] == 4


# ====== R5-2: sample_batch < 50 ms ======

def test_sample_batch_under_50ms(buffer, dims):
    for _ in range(300):
        buffer.store_episode(_make_episode(dims, T=50))
    for _ in range(5):
        buffer.sample_batch(batch_size=256, unroll_K=5)
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        buffer.sample_batch(batch_size=256, unroll_K=5)
        times.append((time.perf_counter() - t0) * 1000)
    mean_ms = sum(times) / len(times)
    assert mean_ms < 50.0, f"sample_batch {mean_ms:.2f}ms > 50ms"


# ====== trainer-facing key surface ======

def test_v5_batch_field_surface(buffer, dims):
    buffer.store_episode(_make_episode(dims, T=50))
    b = buffer.sample_batch(batch_size=4, unroll_K=5)
    for f in ("obs", "actions", "rewards", "pi_mve", "v", "row", "g_hat", "g",
              "t", "dones", "planner_on", "collected_at_step"):
        assert f in b
    for gone in ("delta", "tau", "cap", "c_hat", "z_hat", "c_t"):
        assert gone not in b


# ====== pickle roundtrip (R5-7) ======

def test_buffer_pickle_roundtrip(buffer, dims):
    buffer.store_episode(_make_episode(dims, T=50))
    buffer2 = pickle.loads(pickle.dumps(buffer))
    assert len(buffer2) == 1
    assert buffer2.sample_batch(batch_size=1, unroll_K=5)["obs"].shape[0] == 1


# ====== FIFO eviction ======

def test_fifo_buffer_eviction(cfg_duo, dims):
    cfg_small = replace(cfg_duo, train=replace(cfg_duo.train, buffer_size=5))
    buffer = EpisodeReplayBuffer(cfg_small)
    for _ in range(10):
        buffer.store_episode(_make_episode(dims, T=50))
    assert len(buffer) == 5
