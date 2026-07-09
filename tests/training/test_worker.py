"""Pkg-05 spec 02 acceptance: Worker collection (C5-W1/W2/W3, v5 Pkg-09).

Uses a reduced env T_max for speed (obs_dim is independent of T_max).
"""
import time
from dataclasses import replace

import numpy as np
import pytest
import torch

from hyper_mve.configs import V4Config
from hyper_mve.envs.relation_commons import RelationCommonsEnv
from hyper_mve.models import HyperMuZeroModel
from hyper_mve.planning.mve_planner import MVEPlanner
from hyper_mve.schemas import TimeStepRecord, get_regime_family
from hyper_mve.training import Worker


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
def cfg_fast():
    cfg = V4Config.from_preset("rel_duo")
    return replace(
        cfg,
        env=replace(cfg.env, T_max=30),
        train=replace(cfg.train, mve_samples=12, mve_depth=2),
    )


@pytest.fixture
def model(cfg_fast):
    return HyperMuZeroModel(cfg_fast)


@pytest.fixture
def env(cfg_fast):
    return RelationCommonsEnv(cfg_fast.env)


@pytest.fixture
def worker(cfg_fast, model, env):
    return Worker(cfg_fast, model, env)


# ====== C5-W1: worker never calls update_step ======

def test_worker_no_update_step(worker, monkeypatch):
    calls = _spy(monkeypatch, worker.model, "update_step")
    worker.collect_episode(epsilon=0.1, use_planner=False)
    assert len(calls) == 0


# ====== C5-W2: online BeliefNet.step ======

def test_worker_uses_belief_net_step(worker, monkeypatch):
    calls = _spy(monkeypatch, worker.model.belief_net, "step")
    records = worker.collect_episode(epsilon=0.1, use_planner=False)
    assert len(calls) == len(records)


# ====== C5-W3: record field validity (v5 TimeStepRecord) ======

def test_g_hat_shape_matches_family(worker, cfg_fast):
    records = worker.collect_episode(epsilon=0.1, use_planner=False)
    N = cfg_fast.env.N
    G = get_regime_family(cfg_fast.env).size
    assert records[-1].g_hat.shape == (N, G)


def test_collect_episode_record_fields_valid(worker, cfg_fast):
    records = worker.collect_episode(epsilon=0.1, use_planner=False)
    N, A = cfg_fast.env.N, cfg_fast.env.A
    G = get_regime_family(cfg_fast.env).size
    assert len(records) > 0
    for r in records:
        assert isinstance(r, TimeStepRecord)
        assert r.o.shape[0] == N
        assert r.a.shape == (N,) and r.a.dtype == np.int64
        assert r.r.shape == (N,) and r.r.dtype == np.float32
        assert r.pi_mve.shape == (N, A) and r.pi_mve.dtype == np.float32
        assert r.v.shape == (N,) and r.v.dtype == np.float32
        assert r.row.shape == (N, N - 1) and r.row.dtype == np.float32
        assert r.g_hat.shape == (N, G) and r.g_hat.dtype == np.float32
        assert isinstance(r.g, int) and 0 <= r.g < G
        assert isinstance(r.t, int)
        assert isinstance(r.done, bool)
    # p=0: the regime is static within the episode
    assert len({r.g for r in records}) == 1


def test_records_row_matches_regime(worker, cfg_fast):
    """The stored row must be the regime's diagonal-free row (oracle discipline)."""
    fam = get_regime_family(cfg_fast.env)
    records = worker.collect_episode(epsilon=0.1, use_planner=False)
    reg = fam.regimes[records[0].g]
    expected = np.stack([reg.row(i) for i in range(cfg_fast.env.N)])
    np.testing.assert_array_equal(records[0].row, expected)


# ====== review 修订 5: persistent / injectable planner ======

def test_worker_planner_persistent(worker):
    pid = id(worker.planner)
    worker.collect_episode(epsilon=0.1, use_planner=True)
    assert id(worker.planner) == pid


def test_worker_planner_custom_injection(cfg_fast, model, env):
    custom = MVEPlanner(cfg_fast)
    worker = Worker(cfg_fast, model, env, planner=custom)
    assert worker.planner is custom


# ====== Self-Info strictness: row-i-only-for-agent-i, no full-W leak ======

def test_worker_row_discipline(worker, cfg_fast, monkeypatch):
    calls = _spy(monkeypatch, worker.model, "set_context_subjective")
    worker.collect_episode(epsilon=0.1, use_planner=False)
    N = cfg_fast.env.N
    for args, kwargs in calls:
        row_i = kwargs.get("row_i", args[1] if len(args) > 1 else None)
        assert row_i is not None and row_i.shape[-1] == N - 1


# ====== R5-3: collect_episode < 5 s (use_planner=False) ======

@pytest.mark.gpu
def test_collect_episode_under_5s(cfg_fast):
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    cfg = replace(cfg_fast, env=replace(cfg_fast.env, T_max=100))
    model = HyperMuZeroModel(cfg).cuda()
    env = RelationCommonsEnv(cfg.env)
    worker = Worker(cfg, model, env)
    t0 = time.perf_counter()
    worker.collect_episode(epsilon=0.1, use_planner=False)
    assert time.perf_counter() - t0 < 5.0
