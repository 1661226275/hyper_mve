"""Pkg-05 spec 02 acceptance: Worker collection (C5-W1/W2/W3).

Uses a reduced env T_max for speed (obs_dim is independent of T_max).
"""
import time
from dataclasses import replace

import numpy as np
import pytest
import torch

from hyper_mve.configs import V4Config
from hyper_mve.envs.resource_commons.env import ResourceCommonsEnv
from hyper_mve.models import HyperMuZeroModel
from hyper_mve.planning.mve_planner import MVEPlanner
from hyper_mve.schemas import TimeStepRecord
from hyper_mve.training import Worker


@pytest.fixture
def cfg_fast():
    cfg = V4Config.from_preset("medium")
    return replace(cfg, env=replace(cfg.env, T_max=30))


@pytest.fixture
def model(cfg_fast):
    return HyperMuZeroModel(cfg_fast)


@pytest.fixture
def env(cfg_fast):
    return ResourceCommonsEnv(cfg_fast.env)


@pytest.fixture
def worker(cfg_fast, model, env):
    return Worker(cfg_fast, model, env)


# ====== C5-W1: worker never calls update_step ======

def test_worker_no_update_step(worker, mocker):
    spy = mocker.spy(worker.model, "update_step")
    worker.collect_episode(epsilon=0.1, use_planner=False)
    assert spy.call_count == 0


# ====== C5-W2: online BeliefNet.step ======

def test_worker_uses_belief_net_step(worker, mocker):
    spy = mocker.spy(worker.model.belief_net, "step")
    records, _ = worker.collect_episode(epsilon=0.1, use_planner=False)
    assert spy.call_count == len(records)


# ====== C5-W3: z_hat shape / record field validity ======

def test_z_hat_shape_matches_pkg01_spec04(worker, cfg_fast):
    records, _ = worker.collect_episode(epsilon=0.1, use_planner=False)
    N = cfg_fast.env.N
    assert records[-1].z_hat.shape == (N, N - 1, 2)


def test_collect_episode_record_fields_valid(worker, cfg_fast):
    records, c_t_seq = worker.collect_episode(epsilon=0.1, use_planner=False)
    N, A = cfg_fast.env.N, cfg_fast.env.A
    assert len(records) > 0
    for r in records:
        assert isinstance(r, TimeStepRecord)
        assert r.o.shape[0] == N
        assert r.a.shape == (N,) and r.a.dtype == np.int64
        assert r.r.shape == (N,) and r.r.dtype == np.float32
        assert r.delta.shape == (N,) and r.delta.dtype == np.float32
        assert r.pi_mve.shape == (N, A) and r.pi_mve.dtype == np.float32
        assert r.v.shape == (N,) and r.v.dtype == np.float32
        assert r.tau.shape == (N,) and r.tau.dtype == np.int8
        assert r.cap.shape == (N, 4) and r.cap.dtype == np.float32
        assert r.c_hat.shape == (N,) and r.c_hat.dtype == np.float32
        assert r.z_hat.shape == (N, N - 1, 2) and r.z_hat.dtype == np.float32
        assert isinstance(r.t, int)
        assert isinstance(r.done, bool)
    assert len(c_t_seq) == len(records)


# ====== review 修订 5: persistent / injectable planner ======

def test_worker_planner_persistent(worker):
    pid = id(worker.planner)
    worker.collect_episode(epsilon=0.1, use_planner=True)
    assert id(worker.planner) == pid


def test_worker_planner_custom_injection(cfg_fast, model, env):
    custom = MVEPlanner(cfg_fast)
    worker = Worker(cfg_fast, model, env, planner=custom)
    assert worker.planner is custom


# ====== Self-Info strictness: no type leak into set_context_subjective ======

def test_worker_no_oracle_types_leak(worker, mocker):
    spy = mocker.spy(worker.model, "set_context_subjective")
    worker.collect_episode(epsilon=0.1, use_planner=False)
    for call in spy.call_args_list:
        args, kwargs = call
        cap_i = kwargs.get("cap_i", args[1] if len(args) > 1 else None)
        assert cap_i is not None and cap_i.shape[-1] == 4


# ====== R5-3: collect_episode < 5 s (use_planner=False) ======

@pytest.mark.gpu
def test_collect_episode_under_5s(cfg_fast):
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    cfg = replace(cfg_fast, env=replace(cfg_fast.env, T_max=200))
    model = HyperMuZeroModel(cfg).cuda()
    env = ResourceCommonsEnv(cfg.env)
    worker = Worker(cfg, model, env)
    t0 = time.perf_counter()
    worker.collect_episode(epsilon=0.1, use_planner=False)
    assert time.perf_counter() - t0 < 5.0
