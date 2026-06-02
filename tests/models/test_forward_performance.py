"""Pkg-04 spec 07 acceptance tests: forward performance budget (3 档).

GPU-gated: 在无 CUDA 环境自动 skip. CPU 上仅 import/构造 sanity.
"""
import time

import pytest
import torch

from hyper_mve.configs import V4Config
from hyper_mve.models import HyperMuZeroModel


# ====== 档位 1: 单 forward call < 5ms ======

@pytest.mark.gpu
def test_single_call_under_5ms():
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    cfg = V4Config.from_preset("medium")
    model = HyperMuZeroModel(cfg).cuda().eval()
    B, N = 256, cfg.env.N
    obs_dim = model.rep_net.obs_dim

    obs = torch.randn(B, N, obs_dim, device='cuda')
    s = model.encode(obs)
    model.update_step(0)
    model.set_context_objective(torch.full((B,), 0.5, device='cuda'))
    cap = torch.rand(B, 4, device='cuda')
    belief = (torch.rand(B, device='cuda'),
              torch.softmax(torch.randn(B, N - 1, 2, device='cuda'), dim=-1))
    model.set_context_subjective(0, cap, belief)
    action = torch.zeros(B, N * cfg.env.A, device='cuda')
    action[:, 0] = 1.0

    def time_call(fn, n=30):
        for _ in range(10):
            _ = fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            _ = fn()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        return sum(times) / len(times)

    with torch.no_grad():
        assert time_call(lambda: model.encode(obs)) < 5.0
        assert time_call(lambda: model.transition(s, action)) < 5.0
        assert time_call(lambda: model.predict_reward(s, action)) < 5.0
        assert time_call(lambda: model.predict(s)) < 5.0
        assert time_call(lambda: model.set_context_objective(torch.full((B,), 0.5, device='cuda'))) < 5.0
        assert time_call(lambda: model.set_context_subjective(0, cap, belief)) < 5.0


# ====== 档位 2: 单 agent K=5 step unroll < 25ms ======

@pytest.mark.gpu
def test_single_agent_unroll_under_25ms():
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    cfg = V4Config.from_preset("medium")
    model = HyperMuZeroModel(cfg).cuda().eval()
    B, N = 256, cfg.env.N
    obs_dim = model.rep_net.obs_dim
    K = 5

    obs = torch.randn(B, N, obs_dim, device='cuda')
    cap = torch.rand(B, 4, device='cuda')
    belief = (torch.rand(B, device='cuda'),
              torch.softmax(torch.randn(B, N - 1, 2, device='cuda'), dim=-1))
    action = torch.zeros(B, N * cfg.env.A, device='cuda')
    action[:, 0] = 1.0

    with torch.no_grad():
        for _ in range(10):
            s = model.encode(obs)
            model.update_step(0)
            model.set_context_objective(torch.full((B,), 0.5, device='cuda'))
            model.set_context_subjective(0, cap, belief)
            for _ in range(K):
                s = model.transition(s, action)
                _ = model.predict_reward(s, action)
                _ = model.predict(s)
        torch.cuda.synchronize()

        times = []
        for _ in range(30):
            s = model.encode(obs)
            model.update_step(0)
            model.set_context_objective(torch.full((B,), 0.5, device='cuda'))

            t0 = time.perf_counter()
            model.set_context_subjective(0, cap, belief)
            for _ in range(K):
                s = model.transition(s, action)
                _ = model.predict_reward(s, action)
                _ = model.predict(s)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    mean_ms = sum(times) / len(times)
    assert mean_ms < 25.0, f"Single agent K-step unroll {mean_ms:.2f}ms > 25ms"


# ====== 档位 3: N=4 agents x K=5 step unroll < 100ms ======

def _full_step(model, obs, caps, c_hats, z_hats, action, N, K):
    B = obs.shape[0]
    s = model.encode(obs)
    model.update_step(0)
    model.set_context_objective(torch.full((B,), 0.5, device=obs.device))
    for k in range(N):
        model.set_context_subjective(k, caps[:, k], (c_hats[:, k], z_hats[:, k]))
        s_k = s
        for _ in range(K):
            s_k = model.transition(s_k, action)
            _ = model.predict_reward(s_k, action)
            _ = model.predict(s_k)
    return s


@pytest.mark.gpu
def test_full_step_under_100ms():
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    cfg = V4Config.from_preset("medium")
    model = HyperMuZeroModel(cfg).cuda().eval()
    B, N = 256, cfg.env.N
    obs_dim = model.rep_net.obs_dim
    K = 5

    obs = torch.randn(B, N, obs_dim, device='cuda')
    caps = torch.rand(B, N, 4, device='cuda')
    c_hats = torch.rand(B, N, device='cuda')
    z_hats = torch.softmax(torch.randn(B, N, N - 1, 2, device='cuda'), dim=-1)
    action = torch.zeros(B, N * cfg.env.A, device='cuda')
    action[:, 0] = 1.0

    with torch.no_grad():
        for _ in range(10):
            _ = _full_step(model, obs, caps, c_hats, z_hats, action, N, K)
        torch.cuda.synchronize()

        times = []
        for _ in range(30):
            t0 = time.perf_counter()
            _ = _full_step(model, obs, caps, c_hats, z_hats, action, N, K)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    mean_ms = sum(times) / len(times)
    assert mean_ms < 100.0, f"Full step (N x K) {mean_ms:.2f}ms > 100ms"


def test_forward_smoke_under_15ms():
    """旧名兼容: 实际由 test_full_step_under_100ms 覆盖 (review 修订 4)."""
    test_full_step_under_100ms()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
