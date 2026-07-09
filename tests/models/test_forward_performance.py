"""Pkg-04 spec 07 acceptance tests: forward performance budget (3 tiers, v5 API).

GPU-gated: auto-skips without CUDA. CPU gets import/construct sanity only.
"""
import time

import pytest
import torch

from hyper_mve.configs import V4Config
from hyper_mve.models import HyperMuZeroModel

_G = 5


def _subjective_inputs(B, N, device):
    row = torch.rand(B, N - 1, device=device) * 2 - 1
    g_hat = torch.softmax(torch.randn(B, _G, device=device), dim=-1)
    return row, g_hat


# ====== tier 1: single forward call < 5ms ======

@pytest.mark.gpu
def test_single_call_under_5ms():
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    cfg = V4Config.from_preset("rel_duo")
    model = HyperMuZeroModel(cfg).cuda().eval()
    B, N = 256, cfg.env.N
    obs_dim = model.rep_net.obs_dim

    obs = torch.randn(B, N, obs_dim, device='cuda')
    s = model.encode(obs)
    model.update_step(0)
    row, g_hat = _subjective_inputs(B, N, 'cuda')
    model.set_context_subjective(0, row, g_hat)
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
        assert time_call(lambda: model.set_context_subjective(0, row, g_hat)) < 5.0


# ====== tier 2: single-agent K=5 unroll < 25ms ======

@pytest.mark.gpu
def test_single_agent_unroll_under_25ms():
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    cfg = V4Config.from_preset("rel_duo")
    model = HyperMuZeroModel(cfg).cuda().eval()
    B, N = 256, cfg.env.N
    obs_dim = model.rep_net.obs_dim
    K = 5

    obs = torch.randn(B, N, obs_dim, device='cuda')
    row, g_hat = _subjective_inputs(B, N, 'cuda')
    action = torch.zeros(B, N * cfg.env.A, device='cuda')
    action[:, 0] = 1.0

    with torch.no_grad():
        for _ in range(10):
            s = model.encode(obs)
            model.update_step(0)
            model.set_context_subjective(0, row, g_hat)
            for _ in range(K):
                s = model.transition(s, action)
                _ = model.predict_reward(s, action)
                _ = model.predict(s)
        torch.cuda.synchronize()

        times = []
        for _ in range(30):
            s = model.encode(obs)
            model.update_step(0)

            t0 = time.perf_counter()
            model.set_context_subjective(0, row, g_hat)
            for _ in range(K):
                s = model.transition(s, action)
                _ = model.predict_reward(s, action)
                _ = model.predict(s)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    mean_ms = sum(times) / len(times)
    assert mean_ms < 25.0, f"Single agent K-step unroll {mean_ms:.2f}ms > 25ms"


# ====== tier 3: N agents x K=5 unroll < 100ms ======

def _full_step(model, obs, rows, g_hats, action, N, K):
    s = model.encode(obs)
    model.update_step(0)
    for k in range(N):
        model.set_context_subjective(k, rows[:, k], g_hats[:, k])
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
    cfg = V4Config.from_preset("rel_duo")
    model = HyperMuZeroModel(cfg).cuda().eval()
    B, N = 256, cfg.env.N
    obs_dim = model.rep_net.obs_dim
    K = 5

    obs = torch.randn(B, N, obs_dim, device='cuda')
    rows = torch.rand(B, N, N - 1, device='cuda') * 2 - 1
    g_hats = torch.softmax(torch.randn(B, N, _G, device='cuda'), dim=-1)
    action = torch.zeros(B, N * cfg.env.A, device='cuda')
    action[:, 0] = 1.0

    with torch.no_grad():
        for _ in range(10):
            _ = _full_step(model, obs, rows, g_hats, action, N, K)
        torch.cuda.synchronize()

        times = []
        for _ in range(30):
            t0 = time.perf_counter()
            _ = _full_step(model, obs, rows, g_hats, action, N, K)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    mean_ms = sum(times) / len(times)
    assert mean_ms < 100.0, f"Full step (N x K) {mean_ms:.2f}ms > 100ms"


def test_forward_smoke_under_15ms():
    """Legacy-name shim: covered by test_full_step_under_100ms (review 修订 4)."""
    test_full_step_under_100ms()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
