"""Pkg-04 端到端 forward smoke test + 性能 profile (spec 07 §6.1).

Run (from D:\\RL\\hyper_mve):
    python hyper_mve/scripts/test_hyper_model_forward.py --preset medium --steps 100

Validates:
    - 100 步无 NaN / Inf
    - reward 数值合理 (|r| < 10)
    - 单步 (N agents x 1 transition) forward 性能 (CUDA: 档位 3 < 100ms)
    - 参数量分解 (打印; 范围核对见 NOTE)

NOTE (spec 07 §3.2/§3.3): v4 用 vanilla-MLP hypernet, 每个 hyper 输出层 ~256 x P
(P ~ 30K) => 单 hyper ~9M, 三 hyper 远超 [2.8M, 3.6M] 预算. 这是 SDD 已知缺口
(spec 07 §3.3 "当前 SDD 阶段不锁定精确数字, 实测后回填 + 触发 PR 评审"). 故本脚本
把参数量做成**警告打印**而非硬 assert, 以免阻断 NaN/perf 冒烟; 是否调整架构
(chunked hypernet / 缩小 hidden) 留待 PR 评审与 Ch4.3.4 对账.
"""
import os
import sys
import time
import argparse

# Allow `from hyper_mve...` when run from the outer hyper_mve/ working dir.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

from hyper_mve.configs import V4Config
from hyper_mve.models import HyperMuZeroModel


PARAM_BUDGET_LO = 2.8e6
PARAM_BUDGET_HI = 3.6e6


def benchmark_forward(model, cfg, B=256, num_steps=100, device='cuda'):
    model = model.to(device)
    model.eval()

    N = cfg.env.N
    obs_dim = model.rep_net.obs_dim

    times = []
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            obs = torch.randn(B, N, obs_dim, device=device)
            c_t = torch.full((B,), 0.5, device=device)
            cap = torch.rand(B, N, 4, device=device)
            c_hat = torch.rand(B, N, device=device)
            z_hat = torch.softmax(torch.randn(B, N, N - 1, 2, device=device), dim=-1)

            s = model.encode(obs)
            model.update_step(0)
            model.set_context_objective(c_t)
            for k in range(N):
                model.set_context_subjective(k, cap[:, k], (c_hat[:, k], z_hat[:, k]))
                _ = model.predict_reward(s, torch.zeros(B, N * cfg.env.A, device=device))
                _ = model.predict(s)

        if device == 'cuda':
            torch.cuda.synchronize(device)

        for step in range(num_steps):
            t0 = time.perf_counter()

            obs = torch.randn(B, N, obs_dim, device=device)
            c_t = torch.full((B,), 0.5, device=device)
            cap = torch.rand(B, N, 4, device=device)
            c_hat = torch.rand(B, N, device=device)
            z_hat = torch.softmax(torch.randn(B, N, N - 1, 2, device=device), dim=-1)
            action = torch.zeros(B, N * cfg.env.A, device=device)
            action[:, 0] = 1.0

            s = model.encode(obs)
            model.update_step(step)
            model.set_context_objective(c_t)

            for k in range(N):
                model.set_context_subjective(k, cap[:, k], (c_hat[:, k], z_hat[:, k]))
                r = model.predict_reward(s, action)
                pi, v = model.predict(s)
                assert not torch.isnan(r).any(), f"NaN in reward at step {step} agent {k}"
                assert (r > -10).all() and (r < 10).all(), f"reward out of range at step {step}"

            s_next = model.transition(s, action)
            assert not torch.isnan(s_next).any(), f"NaN in s_next at step {step}"

            if device == 'cuda':
                torch.cuda.synchronize(device)
            times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    return {
        "mean_ms": float(times.mean()),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "max_ms": float(times.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preset', default='medium')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    cfg = V4Config.from_preset(args.preset)
    model = HyperMuZeroModel(cfg)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params / 1e6:.2f}M")

    print("\nParam breakdown:")
    for name, sub in [("RepNet", model.rep_net),
                      ("BeliefNet", model.belief_net),
                      ("TriContextEncoder", model.tri_context_encoder),
                      ("DualHyperNetwork", model.hyper_net),
                      ("StateTransNet", model.state_trans_net),
                      ("RewardHead", model.reward_head),
                      ("PredictionNet", model.prediction_net)]:
        n = sum(p.numel() for p in sub.parameters())
        print(f"  {name}: {n / 1e6:.3f}M")

    # 参数量范围核对 (spec 07 §3.3: 警告而非硬 assert; 见文件头 NOTE).
    if not (PARAM_BUDGET_LO < total_params < PARAM_BUDGET_HI):
        print(
            f"\n[WARN] Total params {total_params / 1e6:.2f}M out of "
            f"[{PARAM_BUDGET_LO / 1e6:.1f}M, {PARAM_BUDGET_HI / 1e6:.1f}M] (spec 07 §3.3). "
            f"vanilla-MLP hypernet 输出层主导; 需 PR 评审 + 与 Ch4.3.4 对账 "
            f"(chunked hypernet / 缩小 hyper hidden)."
        )
    else:
        print(f"\n[OK] Total params within [{PARAM_BUDGET_LO/1e6:.1f}M, {PARAM_BUDGET_HI/1e6:.1f}M].")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = benchmark_forward(model, cfg, B=args.batch_size, num_steps=args.steps, device=device)

    print(f"\nForward performance ({args.steps} steps, B={args.batch_size}, device={device}):")
    print(f"  Mean:   {results['mean_ms']:.2f} ms")
    print(f"  Median: {results['median_ms']:.2f} ms")
    print(f"  P95:    {results['p95_ms']:.2f} ms")
    print(f"  Max:    {results['max_ms']:.2f} ms")

    if device == 'cuda':
        assert results['mean_ms'] < 100.0, (
            f"Mean full-step {results['mean_ms']:.2f}ms exceeds 100ms budget (档位 3)"
        )
        print(f"\n[PASS] 档位 3 (full step): {results['mean_ms']:.2f}ms < 100ms")
    else:
        print("\n[WARN] Running on CPU, performance budget not enforced.")

    print("\n[PASS] forward smoke: 100 steps, no NaN, reward in range.")


if __name__ == '__main__':
    main()
