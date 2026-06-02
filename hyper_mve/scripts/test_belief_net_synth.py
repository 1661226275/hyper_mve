"""Synthetic convergence test for BeliefNet (Pkg-03 spec 06 §5.2 / spec 01 §5.3).

Collects fixed-``c`` / fixed-``types`` trajectories from ``ResourceCommonsEnv``,
then trains ``BeliefNet`` with ``belief_loss`` and asserts the three belief
heads actually learn:

- ``head_c`` MSE < 0.05            (L_c works: ĉ tracks oracle c_t)
- ``head_opp`` accuracy > 0.80     (L_opp works: type 2-class CE from Oracle)
- GRU hidden per-agent variance >= 0.1  (L_div hinge works: no collapse)

Notes
-----
* ``c`` is fixed at reset (``options={"c": 0.7}``). It is part of the public
  context block in the observation, so ``head_c`` should converge quickly.
* ``types`` are fixed to ``[ALPHA, ALPHA, BETA, BETA]`` and never exposed to a
  given agent's own observation (Self-Info). ``head_opp`` must infer *other*
  agents' types from the trajectory — this is the hard signal that L_opp drives.

Run from the repository root::

    python hyper_mve/scripts/test_belief_net_synth.py
    python hyper_mve/scripts/test_belief_net_synth.py --steps 5000 --episodes 64
    python hyper_mve/scripts/test_belief_net_synth.py --quick   # fast smoke (small T/steps)

Prints ``[PASS]`` / ``[FAIL]`` per metric and exits non-zero on any failure.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

# torch is required; fail loudly with a clear message if it is missing.
try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - environment guard
    print(f"[FAIL] PyTorch is required to run this script: {exc}")
    sys.exit(2)

from hyper_mve.configs import V4Config
from hyper_mve.envs.resource_commons import ResourceCommonsEnv
from hyper_mve.models import BeliefNet
from hyper_mve.models.belief_losses import belief_loss, build_oracle_z_seq
from hyper_mve.schemas import AgentType


def collect_dataset(
    cfg: V4Config,
    n_episodes: int,
    seq_len: int,
    c_fixed: float,
    types_fixed: tuple,
    base_seed: int,
):
    """Roll ``n_episodes`` random-policy trajectories with fixed c / types.

    Returns torch tensors:
        obs_seq    (E, T, N, obs_dim) float32
        c_true_seq (E, T)             float32   (oracle c_t per step)
        types_seq  (E, T, N)          int64     (oracle types, constant over T)
    """
    env = ResourceCommonsEnv(cfg.env, seed=base_seed)
    rng = np.random.default_rng(base_seed)
    N = cfg.env.N

    obs_list, c_list, types_list = [], [], []
    for ep in range(n_episodes):
        obs, info = env.reset(
            seed=base_seed + ep,
            options={"c": c_fixed, "types": types_fixed},
        )
        ep_obs, ep_c, ep_types = [], [], []
        for _ in range(seq_len):
            ep_obs.append(np.asarray(obs, dtype=np.float32))          # (N, obs_dim)
            ep_c.append(float(info["c_true"]))
            ep_types.append(np.asarray(info["types"], dtype=np.int64))  # (N,)
            action = rng.integers(0, 6, size=N, dtype=np.int64)
            obs, _reward, done, _trunc, info = env.step(action)
            if done:
                # Re-pad by repeating the terminal observation to keep T uniform.
                while len(ep_obs) < seq_len:
                    ep_obs.append(np.asarray(obs, dtype=np.float32))
                    ep_c.append(float(info["c_true"]))
                    ep_types.append(np.asarray(info["types"], dtype=np.int64))
                break
        obs_list.append(np.stack(ep_obs[:seq_len]))     # (T, N, obs_dim)
        c_list.append(np.stack(ep_c[:seq_len]))         # (T,)
        types_list.append(np.stack(ep_types[:seq_len]))  # (T, N)

    obs_seq = torch.from_numpy(np.stack(obs_list)).float()       # (E, T, N, obs_dim)
    c_true_seq = torch.from_numpy(np.stack(c_list)).float()      # (E, T)
    types_seq = torch.from_numpy(np.stack(types_list)).long()    # (E, T, N)
    return obs_seq, c_true_seq, types_seq


def head_opp_accuracy(z_hat: torch.Tensor, types_true: torch.Tensor) -> float:
    """Fraction of (b, t, i, k) where argmax(z_hat) == oracle opponent type.

    Uses ``build_oracle_z_seq`` to obtain the one-hot oracle labels in the exact
    same (i, k) -> agent_id ordering as ``l_opp`` / ``head_opp``.
    """
    oracle = build_oracle_z_seq(types_true)             # (B, T, N, N-1, 2)
    pred_label = z_hat.argmax(dim=-1)                   # (B, T, N, N-1)
    true_label = oracle.argmax(dim=-1)                  # (B, T, N, N-1)
    return (pred_label == true_label).float().mean().item()


def per_agent_hidden_variance(hidden_seq: torch.Tensor) -> float:
    """Mean (over batch / step / dim) variance of hidden across the N agents."""
    # hidden_seq: (B, T, N, h) -> var over agent dim (=2), unbiased to match l_div
    var = hidden_seq.var(dim=2, unbiased=False)         # (B, T, h)
    return var.mean().item()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--steps", type=int, default=5000, help="optimizer steps")
    parser.add_argument("--episodes", type=int, default=64, help="dataset episodes")
    parser.add_argument("--seq-len", type=int, default=0,
                        help="trajectory length T (0 = cfg.env.T_max)")
    parser.add_argument("--batch", type=int, default=16, help="minibatch episodes")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true",
                        help="fast smoke: small T / steps / episodes")
    parser.add_argument("--mse-thresh", type=float, default=0.05)
    parser.add_argument("--acc-thresh", type=float, default=0.80)
    parser.add_argument("--var-thresh", type=float, default=0.1)
    args = parser.parse_args()

    if args.quick:
        args.steps = min(args.steps, 300)
        args.episodes = min(args.episodes, 16)
        if args.seq_len == 0:
            args.seq_len = 30

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = V4Config.from_preset(args.preset)
    N = cfg.env.N
    seq_len = args.seq_len if args.seq_len > 0 else cfg.env.T_max

    # Fixed types: first half ALPHA, second half BETA (Medium N=4 -> [a, a, b, b]).
    half = N // 2
    types_fixed = tuple([AgentType.ALPHA] * half + [AgentType.BETA] * (N - half))

    print(
        f"[setup] preset={args.preset} N={N} T={seq_len} "
        f"episodes={args.episodes} steps={args.steps} batch={args.batch} "
        f"c=0.7 types={[t.name for t in types_fixed]}"
    )

    obs_seq, c_true_seq, types_seq = collect_dataset(
        cfg, args.episodes, seq_len, c_fixed=0.7,
        types_fixed=types_fixed, base_seed=args.seed,
    )
    E = obs_seq.shape[0]
    print(f"[setup] dataset obs_seq={tuple(obs_seq.shape)} "
          f"c_true_seq={tuple(c_true_seq.shape)} types_seq={tuple(types_seq.shape)}")

    if torch.isnan(obs_seq).any() or torch.isinf(obs_seq).any():
        print("[FAIL] dataset obs contains NaN/Inf")
        return 1

    bn = BeliefNet(cfg.env, cfg.model)
    opt = torch.optim.Adam(bn.parameters(), lr=args.lr)

    bn.train()
    rng = np.random.default_rng(args.seed)
    for step in range(args.steps):
        idx = rng.integers(0, E, size=min(args.batch, E))
        obs_b = obs_seq[idx]                # (B, T, N, obs_dim)
        c_b = c_true_seq[idx]               # (B, T)
        types_b = types_seq[idx]            # (B, T, N)

        hidden_seq, c_hat_seq, z_hat_seq = bn.forward(obs_b)
        total, breakdown = belief_loss(
            c_hat_seq, z_hat_seq, hidden_seq, c_b, types_b,
            weights=(1.0, 0.5, 0.01),
        )

        opt.zero_grad()
        total.backward()
        opt.step()

        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            print(
                f"[step {step:5d}] total={total.item():.4f} "
                f"l_c={breakdown['l_c'].item():.4f} "
                f"l_opp={breakdown['l_opp'].item():.4f} "
                f"l_div={breakdown['l_div'].item():.5f}"
            )

    # ====== Final evaluation on the full dataset ======
    bn.eval()
    with torch.no_grad():
        hidden_seq, c_hat_seq, z_hat_seq = bn.forward(obs_seq)
        # head_c MSE vs oracle c (broadcast c_true (E,T) -> (E,T,N))
        c_mse = F.mse_loss(c_hat_seq, c_true_seq.unsqueeze(-1).expand_as(c_hat_seq)).item()
        acc = head_opp_accuracy(z_hat_seq, types_seq)
        hvar = per_agent_hidden_variance(hidden_seq)

    print("\n====== RESULTS ======")
    ok = True

    def check(name: str, value: float, thresh: float, op: str) -> None:
        nonlocal ok
        passed = (value < thresh) if op == "<" else (value >= thresh)
        ok = ok and passed
        tag = "[PASS]" if passed else "[FAIL]"
        print(f"{tag} {name}: {value:.4f} {op} {thresh}")

    check("head_c MSE", c_mse, args.mse_thresh, "<")
    check("head_opp accuracy", acc, args.acc_thresh, ">=")
    check("hidden per-agent variance", hvar, args.var_thresh, ">=")

    if torch.isnan(hidden_seq).any() or torch.isnan(c_hat_seq).any() or torch.isnan(z_hat_seq).any():
        print("[FAIL] BeliefNet outputs contain NaN")
        ok = False

    print("=====================")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
