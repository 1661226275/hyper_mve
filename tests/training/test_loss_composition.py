"""Pkg-05 spec 05 acceptance: compose_total_loss double-path (C5-L1/L2, v5 Pkg-09).

The two gradient-isolation tests monkeypatch ``oracle_g_mixing_weight -> 0`` so the
main-path belief input is the *predicted* posterior (carrying grad). This isolates
the mechanism under test — the belief gradient GATING threshold
(cfg.train.belief_grad_gating_steps) — from the (separate) curriculum stage boundary.
"""
import math
from dataclasses import replace

import pytest
import torch

from hyper_mve.configs import V4Config
from hyper_mve.models import HyperMuZeroModel
from hyper_mve.schemas import RelationObservationLayout
from hyper_mve.training import MuZeroTrainer
from hyper_mve.training.loss_composition import compose_total_loss

_G = 5


@pytest.fixture
def cfg_duo():
    cfg = V4Config.from_preset("rel_duo")
    # the gating tests need the pred path detached (rel_duo default is False)
    return replace(cfg, train=replace(cfg.train, detach_pred_context=True))


@pytest.fixture
def model(cfg_duo):
    return HyperMuZeroModel(cfg_duo)


@pytest.fixture
def trainer(cfg_duo, model):
    return MuZeroTrainer(cfg_duo, model, device=torch.device("cpu"))


def _make_batch(cfg, B=4, device="cpu"):
    N, A, K = cfg.env.N, cfg.env.A, cfg.train.unroll_K
    obs_dim = RelationObservationLayout.total_dim(N, cfg.env.K)
    return {
        "obs": torch.randn(B, K + 1, N, obs_dim, device=device),
        "actions": torch.randint(0, A, (B, K + 1, N), device=device),
        "rewards": torch.randn(B, K + 1, N, device=device),
        "row": torch.rand(B, K + 1, N, N - 1, device=device) * 2 - 1,
        "g_hat": torch.softmax(torch.randn(B, K + 1, N, _G, device=device), dim=-1),
        "g": torch.randint(0, _G, (B, K + 1), device=device),
        "pi_mve": torch.softmax(torch.randn(B, K + 1, N, A, device=device), dim=-1),
        "v": torch.randn(B, K + 1, N, device=device),
        "dones": torch.zeros(B, K + 1, dtype=torch.bool, device=device),
    }


def _belief_grad_norm(model):
    return sum(
        p.grad.abs().sum().item()
        for p in model.belief_net.parameters() if p.grad is not None
    )


def _zero_belief_grads(model):
    for p in model.belief_net.parameters():
        p.grad = None


# ====== dict structure / NaN ======

# Canonical (non-diagnostic) loss keys that compose_total_loss must always emit.
# diag_* keys are an open, append-only set — checked as a prefix family.
_REQUIRED_LOSS_KEYS = frozenset({
    "total", "main", "belief", "lambda_b",
    "policy", "value", "reward", "consist",
    "belief_regime", "belief_div",
    "L_policy_raw", "L_value_raw", "L_reward_raw", "L_consist_raw",
})


def test_loss_composition_returns_full_dict(trainer, cfg_duo, model):
    losses = compose_total_loss(model, _make_batch(cfg_duo), trainer, 100, cfg_duo)
    keys = set(losses)
    missing = _REQUIRED_LOSS_KEYS - keys
    assert not missing, f"compose_total_loss dropped required keys: {sorted(missing)}"
    extras = keys - _REQUIRED_LOSS_KEYS
    non_diag = {k for k in extras if not k.startswith("diag_")}
    assert not non_diag, f"unexpected non-diagnostic keys: {sorted(non_diag)}"
    assert {
        "diag_pi_mve_entropy", "diag_pi_pred_entropy",
        "diag_cos_pred_pair", "diag_cos_rew_pair",
        "diag_l2_rew_pair", "diag_norm_rew",
    } <= keys


def test_loss_composition_internal_baseline_compatible(cfg_duo):
    """The internal baselines train through the SAME compose_total_loss as hyper.

    They expose no ``current_subjective_thetas`` (no generated theta), so the
    hypernet pairwise-θ diagnostics fall back to NaN — but the canonical
    grad-carrying losses are finite and the dict is complete.
    """
    from hyper_mve.baselines import create_baseline
    model = create_baseline(cfg_duo, "input_wide")
    trainer = MuZeroTrainer(cfg_duo, model, device=torch.device("cpu"))
    losses = compose_total_loss(model, _make_batch(cfg_duo), trainer, 100, cfg_duo)

    missing = _REQUIRED_LOSS_KEYS - set(losses)
    assert not missing, f"baseline dropped required keys: {sorted(missing)}"
    for k in _REQUIRED_LOSS_KEYS:
        v = losses[k]
        if torch.is_tensor(v):
            assert not torch.isnan(v).any(), f"NaN in canonical loss '{k}' (baseline)"
    for k in ("diag_cos_pred_pair", "diag_cos_rew_pair",
              "diag_l2_rew_pair", "diag_norm_rew"):
        assert math.isnan(losses[k].item()), f"{k} should be NaN for a non-hypernet baseline"


def test_loss_composition_no_nan(trainer, cfg_duo, model):
    losses = compose_total_loss(model, _make_batch(cfg_duo), trainer, 100, cfg_duo)
    for k, v in losses.items():
        if torch.is_tensor(v):
            assert not torch.isnan(v).any(), f"NaN in loss['{k}']"


# ====== diagnostics: entropy + pairwise-θ ranges ======

def test_diagnostics_entropy_and_pairwise_ranges(trainer, cfg_duo, model):
    losses = compose_total_loss(model, _make_batch(cfg_duo), trainer, 100, cfg_duo)
    ln_A = math.log(cfg_duo.env.A)
    for key in ("diag_pi_mve_entropy", "diag_pi_pred_entropy"):
        v = losses[key].item()
        assert 0.0 <= v <= ln_A + 1e-4, f"{key}={v} out of [0, ln A={ln_A:.3f}]"
    for key in ("diag_cos_pred_pair", "diag_cos_rew_pair"):
        v = losses[key].item()
        assert not math.isnan(v), f"{key} should be defined for N=2"
        assert -1.0 - 1e-4 <= v <= 1.0 + 1e-4, f"{key}={v} out of [-1, 1]"
    assert losses["diag_l2_rew_pair"].item() >= 0.0
    assert losses["diag_norm_rew"].item() > 0.0


def test_full_gen_scope_compose_path():
    """The FULL gen_scope model keeps the full loss/diag dict NaN-free."""
    cfg = V4Config.from_preset("rel_duo")
    cfg = replace(cfg, model=replace(cfg.model, hyper_gen_scope="full"))
    model = HyperMuZeroModel(cfg)
    trainer = MuZeroTrainer(cfg, model, device=torch.device("cpu"))
    losses = compose_total_loss(model, _make_batch(cfg), trainer, 100, cfg)
    keys = set(losses)
    missing = _REQUIRED_LOSS_KEYS - keys
    assert not missing, f"full scope dropped required keys: {sorted(missing)}"
    for k, v in losses.items():
        if torch.is_tensor(v):
            assert not torch.isnan(v).any(), f"NaN in loss['{k}'] (full)"


# ====== C5-L1: lambda_b at loss level ======

def test_lambda_b_curve_matches_cfg(trainer, cfg_duo, model):
    batch = _make_batch(cfg_duo)
    for step in [0, 50_000, 100_000, 200_000]:
        losses = compose_total_loss(model, batch, trainer, step, cfg_duo)
        assert torch.allclose(losses["lambda_b"], torch.tensor(float(trainer.scheduler.lambda_b(step))))


def test_lambda_b_zero_disables_belief_weight(trainer, cfg_duo, model, monkeypatch):
    monkeypatch.setattr(trainer.scheduler, "lambda_b", lambda step: 0.0)
    losses = compose_total_loss(model, _make_batch(cfg_duo), trainer, 100, cfg_duo)
    assert torch.allclose(losses["total"], losses["main"], atol=1e-6)
    assert losses["belief"].item() > 0.0


# ====== C5-L2: belief gradient gating (double path) ======

def test_belief_gradient_isolation_pre_5k(trainer, cfg_duo, model, monkeypatch):
    # mixing=0 isolates the GATING threshold from the curriculum stage, so the
    # main-path belief is the *predicted* posterior (would carry grad if not gated).
    monkeypatch.setattr(trainer.scheduler, "oracle_g_mixing_weight", lambda step: 0.0)
    step = 1000  # < belief_grad_gating_steps (5000)
    trainer.model.update_step(step)
    losses = compose_total_loss(model, _make_batch(cfg_duo), trainer, step, cfg_duo)

    # L_belief always trains BeliefNet.
    _zero_belief_grads(model)
    losses["belief"].backward(retain_graph=True)
    assert _belief_grad_norm(model) > 0.0, "L_belief must train BeliefNet"

    # The main path, in isolation, must NOT reach BeliefNet pre-5k (double detach).
    _zero_belief_grads(model)
    losses["main"].backward()
    main_only = _belief_grad_norm(model)
    assert main_only == 0.0, (
        f"pre-5k: gating must detach the main path from BeliefNet, got {main_only}"
    )


def test_belief_gradient_both_sources_post_5k(trainer, cfg_duo, model, monkeypatch):
    monkeypatch.setattr(trainer.scheduler, "oracle_g_mixing_weight", lambda step: 0.0)
    step = 10_000  # >= 5000: gating off
    trainer.model.update_step(step)
    losses = compose_total_loss(model, _make_batch(cfg_duo), trainer, step, cfg_duo)

    # Once gating is off, the main path also backprops into BeliefNet — via the
    # reward head -> hyper_rew -> ctx_aug belief segment (detach_pred_context=True
    # severs the policy/value path in this fixture). Small but strictly nonzero.
    _zero_belief_grads(model)
    losses["main"].backward()
    main_only = _belief_grad_norm(model)
    assert main_only > 0.0, (
        "post-5k: main loss should backprop into BeliefNet via reward/hyper_rew"
    )


# ====== planner_on mask ======

def test_planner_off_episodes_masked_from_policy_loss(trainer, cfg_duo, model):
    batch = _make_batch(cfg_duo)
    batch["planner_on"] = torch.zeros(batch["obs"].shape[0], dtype=torch.bool)
    losses = compose_total_loss(model, batch, trainer, 100, cfg_duo)
    # all-masked batch ⇒ policy CE contributes 0 and pi_mve entropy is NaN
    assert losses["L_policy_raw"].item() == 0.0
    assert math.isnan(losses["diag_pi_mve_entropy"].item())
