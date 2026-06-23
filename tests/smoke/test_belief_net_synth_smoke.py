"""BeliefNet synthetic-training smoke (migrated from scripts/test_belief_net_synth.py).

A short BeliefNet training run on fixed-c / fixed-types Easy rollouts must execute
without NaN and produce finite losses. The strict convergence gate (head_c MSE <
0.05, head_opp acc > 0.80 over ~5000 steps) is a manual long-run check; this CI
smoke (``@slow`` → Tier 3) is a robust no-NaN end-to-end pass.
"""
from __future__ import annotations

import pytest


def _collect(cfg, n_episodes, seq_len, c_fixed, types_fixed, base_seed):
    import numpy as np
    import torch
    from hyper_mve.envs.resource_commons import ResourceCommonsEnv

    env = ResourceCommonsEnv(cfg.env, seed=base_seed)
    rng = np.random.default_rng(base_seed)
    N = cfg.env.N
    obs_list, c_list, types_list = [], [], []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=base_seed + ep, options={"c": c_fixed, "types": types_fixed})
        ep_obs, ep_c, ep_types = [], [], []
        for _ in range(seq_len):
            ep_obs.append(np.asarray(obs, dtype=np.float32))
            ep_c.append(float(info["c_true"]))
            ep_types.append(np.asarray(info["types"], dtype=np.int64))
            obs, _r, done, _t, info = env.step(rng.integers(0, 6, size=N, dtype=np.int64))
            if done:
                while len(ep_obs) < seq_len:
                    ep_obs.append(np.asarray(obs, dtype=np.float32))
                    ep_c.append(float(info["c_true"]))
                    ep_types.append(np.asarray(info["types"], dtype=np.int64))
                break
        obs_list.append(np.stack(ep_obs[:seq_len]))
        c_list.append(np.stack(ep_c[:seq_len]))
        types_list.append(np.stack(ep_types[:seq_len]))
    return (
        torch.from_numpy(np.stack(obs_list)).float(),
        torch.from_numpy(np.stack(c_list)).float(),
        torch.from_numpy(np.stack(types_list)).long(),
    )


@pytest.mark.slow
def test_belief_net_synth_smoke():
    pytest.importorskip("torch")
    import numpy as np
    import torch
    from hyper_mve.configs import V4Config
    from hyper_mve.models import BeliefNet
    from hyper_mve.models.belief_losses import belief_loss
    from hyper_mve.schemas import AgentType

    torch.manual_seed(0)
    np.random.seed(0)
    cfg = V4Config.from_preset("easy")
    N = cfg.env.N
    seq_len = min(30, cfg.env.T_max)
    half = N // 2
    types_fixed = tuple([AgentType.ALPHA] * half + [AgentType.BETA] * (N - half))

    obs_seq, c_true_seq, types_seq = _collect(cfg, 8, seq_len, 0.7, types_fixed, 0)
    assert not torch.isnan(obs_seq).any() and not torch.isinf(obs_seq).any()

    bn = BeliefNet(cfg.env, cfg.model)
    opt = torch.optim.Adam(bn.parameters(), lr=1e-3)
    bn.train()
    E = obs_seq.shape[0]
    rng = np.random.default_rng(0)
    last = None
    for _ in range(50):  # short smoke; not the strict convergence gate
        idx = rng.integers(0, E, size=min(4, E))
        hidden_seq, c_hat_seq, z_hat_seq = bn.forward(obs_seq[idx])
        total, _bd = belief_loss(c_hat_seq, z_hat_seq, hidden_seq,
                                 c_true_seq[idx], types_seq[idx], weights=(1.0, 0.5, 0.01))
        opt.zero_grad()
        total.backward()
        opt.step()
        assert torch.isfinite(total), "belief loss went non-finite"
        last = float(total.item())
    assert last is not None and last == last  # finite
    bn.eval()
    with torch.no_grad():
        h, c_hat, z_hat = bn.forward(obs_seq)
    assert not (torch.isnan(h).any() or torch.isnan(c_hat).any() or torch.isnan(z_hat).any())
