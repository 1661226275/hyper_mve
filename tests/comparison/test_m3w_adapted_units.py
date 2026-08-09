"""M3W-adapted unit gates (phase-6).

1. Vendor integrity — the MoE classes are imported from the UNMODIFIED
   clone (getsourcefile under vendor/m3w-marl + nested git worktree clean
   for that file).
2. Shape contract of RegimeCondWorldModel.
3. Reward MSE decreases when training on a synthetic regime-dependent
   reward mapping (the conditioning channel carries signal).
4. SparseMoE router top-k sparsity (≤ k experts active per sample, eval mode).
5. Given-ID conditioning changes predictions (the embedding is consumed).
"""
from __future__ import annotations

import inspect
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("einops")

import torch  # noqa: E402

from hyper_mve.comparison.m3w_adapted.vendor_import import (  # noqa: E402
    M3W_DIR,
    load_vendor_modules,
)
from hyper_mve.comparison.m3w_adapted.world_model import (  # noqa: E402
    RegimeCondWorldModel,
)

_DIMS = dict(obs_dim=39, n_agents=2, n_actions=6, n_regimes=5)


def _tiny_wm(**over) -> RegimeCondWorldModel:
    kw = dict(_DIMS, d_z=16, d_g=4, dyn_mlp_dims=(32,),
              n_dyn_experts=2, n_rew_experts=4, rew_top_k=2)
    kw.update(over)
    return RegimeCondWorldModel(**kw)


def test_vendor_modules_imported_unmodified():
    dyn_cls, rew_cls, router_cls = load_vendor_modules()
    # `vendor/` may be a SYMLINK to a shared clone: the vendored baselines are
    # gitignored, so a git worktree does not carry them and they are linked in
    # from the main checkout instead. `getsourcefile().resolve()` follows that
    # link, while M3W_DIR appends "vendor/m3w-marl" to an already-resolved
    # __file__ and so stays un-followed -- resolve both sides or they can never
    # match. The property under test is unchanged: the classes still have to
    # come from the vendored clone, and that clone file still has to be
    # unmodified in its nested git repo.
    m3w_root = M3W_DIR.resolve()
    for cls in (dyn_cls, rew_cls, router_cls):
        src = Path(inspect.getsourcefile(cls)).resolve()
        assert str(src).startswith(str(m3w_root)), (
            f"{cls.__name__} not sourced from the m3w-marl clone: {src}"
        )
    # the clone file itself is untouched in its nested git repo
    rel = Path(inspect.getsourcefile(dyn_cls)).resolve().relative_to(m3w_root)
    proc = subprocess.run(
        ["git", "-C", str(m3w_root), "status", "--porcelain", "--", str(rel)],
        capture_output=True, text=True, timeout=30,
    )
    if proc.returncode != 0:
        pytest.skip(f"nested git unavailable: {proc.stderr.strip()}")
    assert proc.stdout.strip() == "", (
        f"vendor file modified: {proc.stdout.strip()}"
    )


def test_world_model_shape_contract():
    wm = _tiny_wm()
    B, N = 7, _DIMS["n_agents"]
    obs = torch.randn(B, N, _DIMS["obs_dim"])
    g = torch.randint(0, _DIMS["n_regimes"], (B,))
    a = torch.randint(0, _DIMS["n_actions"], (B, N))

    zc = wm.encode(obs, g)
    assert zc.shape == (B, N, wm.d_zc)
    nxt = wm.predict_next(zc, a)
    assert nxt.shape == (B, N, wm.d_zc)
    r = wm.predict_rewards(zc, a)
    assert r.shape == (B, N)

    total, parts = wm.loss(obs, g, a, torch.randn(B, N), obs, g)
    assert total.dim() == 0 and torch.isfinite(total)
    assert set(parts) == {"wm_dyn_loss", "wm_rew_mse", "wm_bal_loss"}


def test_reward_mse_decreases_on_regime_dependent_target():
    torch.manual_seed(0)
    wm = _tiny_wm()
    opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
    B, N = 64, _DIMS["n_agents"]

    def batch():
        obs = torch.randn(B, N, _DIMS["obs_dim"])
        g = torch.randint(0, _DIMS["n_regimes"], (B,))
        a = torch.randint(0, _DIMS["n_actions"], (B, N))
        # reward depends ONLY on (g, a): regime-conditioned deterministic map
        r = (g.float().unsqueeze(1) - 2.0) * 0.5 + 0.1 * a.float()
        return obs, g, a, r

    def rew_mse() -> float:
        wm.eval()
        with torch.no_grad():
            obs, g, a, r = batch()
            mse = torch.nn.functional.mse_loss(
                wm.predict_rewards(wm.encode(obs, g), a), r)
        wm.train()
        return float(mse)

    torch.manual_seed(42)
    before = rew_mse()
    for _ in range(300):
        obs, g, a, r = batch()
        total, _ = wm.loss(obs, g, a, r, obs, g)
        opt.zero_grad()
        total.backward()
        # mirror the runner's optimization (the balancing loss can spike)
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 10.0)
        opt.step()
    torch.manual_seed(42)
    after = rew_mse()
    assert after < 0.5 * before, (before, after)


def test_router_top_k_sparsity():
    wm = _tiny_wm()
    wm.eval()  # deterministic gating
    B, N = 16, _DIMS["n_agents"]
    zc = wm.encode(torch.randn(B, N, _DIMS["obs_dim"]),
                   torch.randint(0, _DIMS["n_regimes"], (B,)))
    a = torch.nn.functional.one_hot(
        torch.randint(0, _DIMS["n_actions"], (B, N)),
        _DIMS["n_actions"]).float()
    _, aux = wm.reward(zc, a)
    gates = aux["gates"]
    assert gates.shape == (B, wm.reward.n_experts)
    active = (gates > 0).sum(dim=1)
    assert (active <= wm.rew_top_k).all(), active
    assert torch.allclose(gates.sum(dim=1), torch.ones(B), atol=1e-5)


def test_given_regime_id_conditions_predictions():
    torch.manual_seed(1)
    wm = _tiny_wm()
    wm.eval()
    B, N = 8, _DIMS["n_agents"]
    obs = torch.randn(B, N, _DIMS["obs_dim"])
    a = torch.randint(0, _DIMS["n_actions"], (B, N))
    g0 = torch.zeros(B, dtype=torch.long)
    g1 = torch.ones(B, dtype=torch.long)
    with torch.no_grad():
        r0 = wm.predict_rewards(wm.encode(obs, g0), a)
        r1 = wm.predict_rewards(wm.encode(obs, g1), a)
        n0 = wm.predict_next(wm.encode(obs, g0), a)
        n1 = wm.predict_next(wm.encode(obs, g1), a)
    assert not torch.allclose(r0, r1), "regime ID does not reach the reward MoE"
    assert not torch.allclose(n0, n1), "regime ID does not reach the dynamics MoE"
