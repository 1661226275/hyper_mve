"""Pkg-05 spec 05 acceptance: compose_total_loss double-path (C5-L1/L2).

The two gradient-isolation tests monkeypatch ``oracle_z_mixing_weight -> 0`` so the
main-path belief input is the *predicted* z (carrying grad). This isolates the
mechanism under test — the belief gradient GATING threshold (cfg.train.belief_grad_gating_steps,
default 5000) — from the (separate) curriculum stage boundary. This is the
real-API reconciliation noted in the plan (R2): BeliefNet.forward has no mixing arg.
"""
import math

import pytest
import torch

from hyper_mve.configs import V4Config
from hyper_mve.models import HyperMuZeroModel
from hyper_mve.schemas import ObservationLayout
from hyper_mve.training import MuZeroTrainer
from hyper_mve.training.loss_composition import compose_total_loss


@pytest.fixture
def cfg_medium():
    return V4Config.from_preset("medium")


@pytest.fixture
def model(cfg_medium):
    return HyperMuZeroModel(cfg_medium)


@pytest.fixture
def trainer(cfg_medium, model):
    return MuZeroTrainer(cfg_medium, model)


def _make_batch(cfg, B=4, device="cpu"):
    N, A, K = cfg.env.N, cfg.env.A, cfg.train.unroll_K
    obs_dim = ObservationLayout.total_dim(N, cfg.env.K)
    return {
        "obs": torch.randn(B, K + 1, N, obs_dim, device=device),
        "actions": torch.randint(0, A, (B, K + 1, N), device=device),
        "rewards": torch.randn(B, K + 1, N, device=device),
        "c_t": torch.rand(B, K + 1, device=device),
        "cap": torch.rand(B, K + 1, N, 4, device=device),
        "c_hat": torch.rand(B, K + 1, N, device=device),
        "z_hat": torch.softmax(torch.randn(B, K + 1, N, N - 1, 2, device=device), dim=-1),
        "tau": torch.randint(0, 2, (B, K + 1, N), dtype=torch.int8, device=device),
        "pi_mve": torch.softmax(torch.randn(B, K + 1, N, A, device=device), dim=-1),
        "v": torch.randn(B, K + 1, N, device=device),
        "delta": torch.randn(B, K + 1, N, device=device),
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
# diag_* keys are an open, append-only set (new probes are added across v4-opt
# phases) — they are checked as a prefix family, not an exact lock.
_REQUIRED_LOSS_KEYS = frozenset({
    "total", "main", "belief", "lambda_b",
    "policy", "value", "reward", "consist",
    "belief_c", "belief_opp", "belief_div",
    "L_policy_raw", "L_value_raw", "L_reward_raw", "L_consist_raw",
})


def test_loss_composition_returns_full_dict(trainer, cfg_medium, model):
    losses = compose_total_loss(model, _make_batch(cfg_medium), trainer, 100, cfg_medium)
    keys = set(losses)
    missing = _REQUIRED_LOSS_KEYS - keys
    assert not missing, f"compose_total_loss dropped required keys: {sorted(missing)}"
    # Every non-canonical key must be a diagnostic (diag_* prefix), so an
    # accidental typo'd or stray key is still caught.
    extras = keys - _REQUIRED_LOSS_KEYS
    non_diag = {k for k in extras if not k.startswith("diag_")}
    assert not non_diag, f"unexpected non-diagnostic keys: {sorted(non_diag)}"
    # The original canonical diagnostics must still be present.
    assert {
        "diag_pi_mve_entropy", "diag_pi_pred_entropy",
        "diag_cos_pred_cross", "diag_cos_pred_same",
        "diag_cos_rew_cross", "diag_cos_rew_same",
    } <= keys


def test_loss_composition_no_nan(trainer, cfg_medium, model):
    losses = compose_total_loss(model, _make_batch(cfg_medium), trainer, 100, cfg_medium)
    for k, v in losses.items():
        if torch.is_tensor(v):
            assert not torch.isnan(v).any(), f"NaN in loss['{k}']"
    # medium is 2α+2β → all role-cosine categories have pairs (none NaN)


# ====== diagnostics: entropy ranges + hypernet role-cosine ======

def test_diagnostics_entropy_and_cosine_ranges(trainer, cfg_medium, model):
    losses = compose_total_loss(model, _make_batch(cfg_medium), trainer, 100, cfg_medium)
    ln_A = math.log(cfg_medium.env.A)
    for key in ("diag_pi_mve_entropy", "diag_pi_pred_entropy"):
        v = losses[key].item()
        assert 0.0 <= v <= ln_A + 1e-4, f"{key}={v} out of [0, ln A={ln_A:.3f}]"
    # medium (2α+2β): both cross-type and same-type pairs exist → all defined in [-1, 1]
    for key in ("diag_cos_pred_cross", "diag_cos_pred_same",
                "diag_cos_rew_cross", "diag_cos_rew_same"):
        v = losses[key].item()
        assert not math.isnan(v), f"{key} should be defined for medium (2α+2β)"
        assert -1.0 - 1e-4 <= v <= 1.0 + 1e-4, f"{key}={v} out of [-1, 1]"


def test_diagnostics_duo_two_agent_cosine():
    """Duo (N=2, 1α+1β): cross-type cosine defined, same-type undefined (NaN).

    Also exercises the N=2 compose path end-to-end (z_hat opponent dim N-1=1).
    """
    cfg = V4Config.from_preset("duo")
    model = HyperMuZeroModel(cfg)
    trainer = MuZeroTrainer(cfg, model)
    losses = compose_total_loss(model, _make_batch(cfg), trainer, 100, cfg)

    assert not math.isnan(losses["diag_cos_pred_cross"].item())
    assert not math.isnan(losses["diag_cos_rew_cross"].item())
    assert math.isnan(losses["diag_cos_pred_same"].item()), "no same-type pair in duo"
    assert math.isnan(losses["diag_cos_rew_same"].item()), "no same-type pair in duo"


# ====== film_head partial-generation: compose path intact ======

def test_loss_composition_film_head_full_dict_and_no_nan():
    """film_head model (shared trunk + generated FiLM/head) keeps the full loss/diag
    dict and stays NaN-free. Medium (N=4=2a+2b) so all cosine categories are defined."""
    from dataclasses import replace
    cfg = V4Config.from_preset("medium")
    cfg = replace(cfg, model=replace(
        cfg.model, hyper_gen_scope="film_head",
        trans_output_scale_init=0.1, pred_output_scale_init=0.1,
    ))
    model = HyperMuZeroModel(cfg)
    trainer = MuZeroTrainer(cfg, model)
    losses = compose_total_loss(model, _make_batch(cfg), trainer, 100, cfg)
    keys = set(losses)
    missing = _REQUIRED_LOSS_KEYS - keys
    assert not missing, f"film_head dropped required keys: {sorted(missing)}"
    non_diag = {k for k in (keys - _REQUIRED_LOSS_KEYS) if not k.startswith("diag_")}
    assert not non_diag, f"unexpected non-diagnostic keys (film_head): {sorted(non_diag)}"
    for k, v in losses.items():
        if torch.is_tensor(v):
            assert not torch.isnan(v).any(), f"NaN in loss['{k}'] (film_head)"


# ====== C5-L1: lambda_b at loss level ======

def test_lambda_b_curve_matches_cfg(trainer, cfg_medium, model):
    batch = _make_batch(cfg_medium)
    for step in [0, 50_000, 100_000, 200_000]:
        losses = compose_total_loss(model, batch, trainer, step, cfg_medium)
        assert torch.allclose(losses["lambda_b"], torch.tensor(float(trainer.scheduler.lambda_b(step))))


def test_lambda_b_zero_disables_belief_weight(trainer, cfg_medium, model, monkeypatch):
    monkeypatch.setattr(trainer.scheduler, "lambda_b", lambda step: 0.0)
    losses = compose_total_loss(model, _make_batch(cfg_medium), trainer, 100, cfg_medium)
    assert torch.allclose(losses["total"], losses["main"], atol=1e-6)
    assert losses["belief"].item() > 0.0


# ====== C5-L2: belief gradient gating (double path) ======

def test_belief_gradient_isolation_pre_5k(trainer, cfg_medium, model, monkeypatch):
    # mixing=0 isolates the GATING threshold (5000) from the curriculum stage, so
    # the main-path belief is the *predicted* z (would carry grad if not gated).
    monkeypatch.setattr(trainer.scheduler, "oracle_z_mixing_weight", lambda step: 0.0)
    step = 1000  # < belief_grad_gating_steps (5000)
    trainer.model.update_step(step)
    losses = compose_total_loss(model, _make_batch(cfg_medium), trainer, step, cfg_medium)

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


def test_belief_gradient_both_sources_post_5k(trainer, cfg_medium, model, monkeypatch):
    monkeypatch.setattr(trainer.scheduler, "oracle_z_mixing_weight", lambda step: 0.0)
    step = 10_000  # >= 5000: gating off
    trainer.model.update_step(step)
    losses = compose_total_loss(model, _make_batch(cfg_medium), trainer, step, cfg_medium)

    # Once gating is off, the main path also backprops into BeliefNet — but ONLY
    # through the reward head -> hyper_rew -> ctx_aug belief segment, because
    # detach_pred_context=True severs the policy/value (hyper_pred) path. So it is
    # small but strictly nonzero. Measure the main path in isolation (robust:
    # avoids the fragile L1-norm-of-sum cancellation between the two gradient paths).
    _zero_belief_grads(model)
    losses["main"].backward()
    main_only = _belief_grad_norm(model)
    assert main_only > 0.0, (
        "post-5k: main loss should backprop into BeliefNet via reward/hyper_rew"
    )
