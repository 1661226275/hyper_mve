"""Forward smoke (migrated from scripts/test_hyper_model_forward.py).

A full (s, a) → (r, π, v) + transition pass through HyperMuZeroModel on Easy
must be NaN-free with rewards in range. The CUDA perf budget (档位 3) is a manual
long-run concern; this CI smoke is small + CPU-friendly.
"""
from __future__ import annotations

import pytest


def test_hyper_model_forward_no_nan_smoke():
    pytest.importorskip("torch")
    import torch
    from hyper_mve.configs import V4Config
    from hyper_mve.models import HyperMuZeroModel

    cfg = V4Config.from_preset("easy")
    model = HyperMuZeroModel(cfg)
    model.eval()
    N = cfg.env.N
    A = cfg.env.A
    obs_dim = model.rep_net.obs_dim
    B = 4
    with torch.no_grad():
        obs = torch.randn(B, N, obs_dim)
        c_t = torch.full((B,), 0.5)
        cap = torch.rand(B, N, 4)
        c_hat = torch.rand(B, N)
        z_hat = torch.softmax(torch.randn(B, N, N - 1, 2), dim=-1)
        action = torch.zeros(B, N * A)
        action[:, 0] = 1.0

        s = model.encode(obs)
        model.update_step(0)
        model.set_context_objective(c_t)
        for k in range(N):
            model.set_context_subjective(k, cap[:, k], (c_hat[:, k], z_hat[:, k]))
            r = model.predict_reward(s, action)
            pi, v = model.predict(s)
            assert not torch.isnan(r).any()
            assert (r > -10).all() and (r < 10).all()
            assert not torch.isnan(pi).any() and not torch.isnan(v).any()
        s_next = model.transition(s, action)
        assert not torch.isnan(s_next).any()
