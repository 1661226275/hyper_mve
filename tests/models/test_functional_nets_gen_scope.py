"""gen_scope partial-generation tests for the three functional nets (FiLM + head).

FULL (default) has the HyperNet generate every layer's weights; ``film_head`` shares
fc1/fc2 as SGD ``nn.Linear`` and the HyperNet generates ONLY FiLM gamma/beta + the
output head. These tests pin the generated-param counts (the DualHyperNetwork is sized
by ``generated_param_count``), the norm groups, the film_head forward shapes + the
zero-params residual identity, and confirm FULL is structurally unchanged.

Numbers assume latent=64, hidden=128, A=6, N=2 (joint_action=12) — matching the plan.
"""
import pytest
import torch

from hyper_mve.models.functional_nets import (
    FunctionalPredictionNet,
    FunctionalRewardHead,
    FunctionalStateTransNet,
)

LATENT, HIDDEN, A = 64, 128, 6
JOINT = 2 * A  # N=2

# FULL totals (N=2, A=6): count_params_adaln over every layer (weight+bias+gamma+beta).
FULL = {"trans": 35136, "rew": 27009, "pred": 26247}
# film_head generated (N-independent): film 512 (= 2*128 gamma/beta * 2 hidden) + head.
FILM = {"trans": 8768, "rew": 641, "pred": 1415}
GROUPS = {"trans": [512, 8256], "rew": [512, 129], "pred": [512, 903]}
# base_gen generated (Option B): plain SGD fc1 base (nothing generated) + fc2 fully
# generated (weight+FiLM) + head. film 256 (fc2 gamma/beta) + weight (fc2 w+b + head).
BASE = {"trans": 25024, "rew": 16897, "pred": 17671}
BASE_GROUPS = {"trans": [256, 24768], "rew": [256, 16641], "pred": [256, 17415]}


def _build(net, gen_scope):
    if net == "trans":
        return FunctionalStateTransNet(LATENT, JOINT, HIDDEN, gen_scope)
    if net == "rew":
        return FunctionalRewardHead(LATENT, JOINT, HIDDEN, gen_scope)
    return FunctionalPredictionNet(LATENT, A, HIDDEN, gen_scope)


@pytest.mark.parametrize("net", ["trans", "rew", "pred"])
def test_full_generated_equals_total(net):
    m = _build(net, "full")
    assert m.generated_param_count == m.total_params == FULL[net]
    assert m.gen_groups is None
    assert m.gen_spec is None
    assert not hasattr(m, "fc1")  # weight-free in FULL


@pytest.mark.parametrize("net", ["trans", "rew", "pred"])
def test_film_head_generated_counts_and_groups(net):
    m = _build(net, "film_head")
    assert m.generated_param_count == FILM[net]
    assert m.generated_param_count < m.total_params
    assert m.gen_groups == GROUPS[net]
    assert sum(m.gen_groups) == m.generated_param_count
    # shared SGD trunk now exists and is trainable
    assert hasattr(m, "fc1") and hasattr(m, "fc2")
    assert m.fc1.weight.requires_grad and m.fc2.weight.requires_grad


def test_film_head_forward_shapes_no_nan():
    B = 4
    state = torch.randn(B, LATENT)
    action = torch.zeros(B, JOINT)
    action[:, 0] = 1.0

    trans = _build("trans", "film_head")
    rew = _build("rew", "film_head")
    pred = _build("pred", "film_head")

    s_next = trans(state, action, torch.randn(B, FILM["trans"]))
    r = rew(state, action, torch.randn(B, FILM["rew"]))
    pi, v = pred(state, torch.randn(B, FILM["pred"]))

    assert s_next.shape == (B, LATENT)
    assert r.shape == (B, 1)
    assert pi.shape == (B, A)
    assert v.shape == (B, 1)
    for t in (s_next, r, pi, v):
        assert not torch.isnan(t).any()


def test_film_head_state_trans_zero_params_is_residual_identity():
    # Zero generated params -> head W=b=0 -> delta_s (pre-LN) = 0 -> LN(0)=0 (affine
    # bias init 0) -> s_next = state + 0. Confirms the residual path is preserved.
    B = 3
    trans = _build("trans", "film_head")
    state = torch.randn(B, LATENT)
    action = torch.zeros(B, JOINT)
    action[:, 0] = 1.0
    s_next = trans(state, action, torch.zeros(B, FILM["trans"]))
    assert torch.allclose(s_next, state, atol=1e-5)


def test_full_mode_forward_unchanged_shapes():
    B = 2
    state = torch.randn(B, LATENT)
    action = torch.zeros(B, JOINT)
    action[:, 0] = 1.0
    pred = _build("pred", "full")
    pi, v = pred(state, torch.randn(B, FULL["pred"]))
    assert pi.shape == (B, A) and v.shape == (B, 1)


@pytest.mark.parametrize("net", ["trans", "rew", "pred"])
def test_base_gen_generated_counts_and_groups(net):
    m = _build(net, "base_gen")
    assert m.generated_param_count == BASE[net]
    assert m.generated_param_count < m.total_params
    assert m.gen_groups == BASE_GROUPS[net]
    assert sum(m.gen_groups) == m.generated_param_count


@pytest.mark.parametrize("net", ["trans", "rew", "pred"])
def test_base_gen_structure(net):
    m = _build(net, "base_gen")
    # plain SGD fc1 base (Linear + ln1) exists and trains; fc2 is GENERATED -> absent.
    assert hasattr(m, "fc1") and m.fc1.weight.requires_grad
    assert hasattr(m, "ln1") and isinstance(m.ln1, torch.nn.LayerNorm)
    assert not hasattr(m, "fc2")


def test_base_gen_forward_shapes_no_nan():
    B = 4
    state = torch.randn(B, LATENT)
    action = torch.zeros(B, JOINT)
    action[:, 0] = 1.0

    trans = _build("trans", "base_gen")
    rew = _build("rew", "base_gen")
    pred = _build("pred", "base_gen")

    s_next = trans(state, action, torch.randn(B, BASE["trans"]))
    r = rew(state, action, torch.randn(B, BASE["rew"]))
    pi, v = pred(state, torch.randn(B, BASE["pred"]))

    assert s_next.shape == (B, LATENT)
    assert r.shape == (B, 1)
    assert pi.shape == (B, A)
    assert v.shape == (B, 1)
    for t in (s_next, r, pi, v):
        assert not torch.isnan(t).any()


def test_base_gen_state_trans_zero_params_is_residual_identity():
    # Zero generated params -> head W=b=0 -> delta_s (pre-LN)=0 -> LN(0)=0 -> s_next =
    # state. Confirms the residual path survives the SGD-base + generated-fc2 split.
    B = 3
    trans = _build("trans", "base_gen")
    state = torch.randn(B, LATENT)
    action = torch.zeros(B, JOINT)
    action[:, 0] = 1.0
    s_next = trans(state, action, torch.zeros(B, BASE["trans"]))
    assert torch.allclose(s_next, state, atol=1e-5)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
