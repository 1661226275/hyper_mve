"""BeliefNet synthetic-training smoke (v5: regime inference on RelationCommons).

A short BeliefNet training run on regime-pinned rel_duo rollouts must execute
without NaN and produce finite, decreasing-ish losses. The strict convergence
gate (regime accuracy > chance over ~5000 steps) is a manual long-run check;
this CI smoke (``@slow`` → Tier 3) is a robust no-NaN end-to-end pass.
"""
from __future__ import annotations

import pytest


def _collect(cfg, n_episodes, seq_len, base_seed):
    import numpy as np
    import torch
    from hyper_mve.envs.relation_commons import RelationCommonsEnv

    env = RelationCommonsEnv(cfg.env, seed=base_seed)
    rng = np.random.default_rng(base_seed)
    N = cfg.env.N
    obs_list, g_list = [], []
    for ep in range(n_episodes):
        g_pin = ep % 5
        obs, info = env.reset(seed=base_seed + ep, options={"g": g_pin})
        ep_obs, ep_g = [], []
        for _ in range(seq_len):
            ep_obs.append(np.asarray(obs, dtype=np.float32))
            ep_g.append(int(info["g_true"]))
            obs, _r, done, _t, info = env.step(rng.integers(0, 6, size=N, dtype=np.int64))
            if done:
                while len(ep_obs) < seq_len:
                    ep_obs.append(np.asarray(obs, dtype=np.float32))
                    ep_g.append(int(info["g_true"]))
                break
        obs_list.append(np.stack(ep_obs[:seq_len]))
        g_list.append(np.array(ep_g[:seq_len], dtype=np.int64))
    return (
        torch.from_numpy(np.stack(obs_list)).float(),   # (E, T, N, obs_dim)
        torch.from_numpy(np.stack(g_list)).long(),      # (E, T)
    )


@pytest.mark.slow
def test_belief_net_synth_smoke():
    pytest.importorskip("torch")
    import numpy as np
    import torch
    from hyper_mve.utils.configs import V4Config
    from hyper_mve.algo.modules import BeliefNet
    from hyper_mve.algo.modules.belief_losses import belief_loss

    torch.manual_seed(0)
    np.random.seed(0)
    cfg = V4Config.from_preset("rel_duo")
    seq_len = min(30, cfg.env.T_max)

    obs_seq, g_seq = _collect(cfg, n_episodes=8, seq_len=seq_len, base_seed=123)

    bn = BeliefNet(cfg.env, cfg.model)
    opt = torch.optim.Adam(bn.parameters(), lr=1e-3)

    first_loss = last_loss = None
    for step in range(30):
        hidden_seq, g_hat_seq = bn(obs_seq)
        total, breakdown = belief_loss(g_hat_seq, hidden_seq, g_seq)
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
        val = float(total.detach())
        assert np.isfinite(val), f"non-finite belief loss at step {step}"
        if first_loss is None:
            first_loss = val
        last_loss = val

    # very soft learning signal: loss must not blow up (and typically drops)
    assert last_loss < first_loss * 1.5
