"""gen_scope partial-generation tests for the two subjective functional nets (FiLM + head).

v5 (Pkg-09): FunctionalStateTransNet is deleted — the transition is a plain shared
``TransitionNet`` (see tests below) and only reward/prediction remain functional.

FULL (default) has the HyperNet generate every layer's weights; ``film_head`` shares
fc1/fc2 as SGD ``nn.Linear`` and the HyperNet generates ONLY FiLM gamma/beta + the
output head. These tests pin the generated-param counts (the DualHyperNetwork is sized
by ``generated_param_count``), the norm groups, the film_head forward shapes + the
zero-params residual identity, and confirm FULL is structurally unchanged.

Numbers assume latent=64, hidden=128, A=6, N=2 (joint_action=12) — matching the plan.
"""
import pytest
import torch

import torch.nn as nn

from hyper_mve.models.functional_nets import (
    FunctionalPredictionNet,
    FunctionalRewardHead,
)
from hyper_mve.models.transition_net import TransitionNet

LATENT, HIDDEN, A = 64, 128, 6
JOINT = 2 * A  # N=2

# FULL totals (N=2, A=6): count_params_adaln over every layer (weight+bias+gamma+beta).
FULL = {"rew": 27009, "pred": 26247}
# film_head generated (N-independent): film 512 (= 2*128 gamma/beta * 2 hidden) + head.
FILM = {"rew": 641, "pred": 1415}
GROUPS = {"rew": [512, 129], "pred": [512, 903]}
# base_gen generated (Option B): plain SGD fc1 base (nothing generated) + fc2 fully
# generated (weight+FiLM) + head. film 256 (fc2 gamma/beta) + weight (fc2 w+b + head).
BASE = {"rew": 16897, "pred": 17671}
BASE_GROUPS = {"rew": [256, 16641], "pred": [256, 17415]}
# lora_fc2 generated (film_head + rank-r fc2 weight delta): film 512 + Af/Bf (256*r) + head.
LORA_R = 8
LORA = {"rew": 2689, "pred": 3463}            # = FILM + 256 * r
LORA_GROUPS = {"rew": [512, 2177], "pred": [512, 2951]}


def _build(net, gen_scope):
    if net == "rew":
        return FunctionalRewardHead(LATENT, JOINT, HIDDEN, gen_scope)
    return FunctionalPredictionNet(LATENT, A, HIDDEN, gen_scope)


def _build_lora(net, lora_rank):
    if net == "rew":
        return FunctionalRewardHead(LATENT, JOINT, HIDDEN, "lora_fc2", lora_rank)
    return FunctionalPredictionNet(LATENT, A, HIDDEN, "lora_fc2", lora_rank)


@pytest.mark.parametrize("net", ["rew", "pred"])
def test_full_generated_equals_total(net):
    m = _build(net, "full")
    assert m.generated_param_count == m.total_params == FULL[net]
    assert m.gen_groups is None
    assert m.gen_spec is None
    assert not hasattr(m, "fc1")  # weight-free in FULL


@pytest.mark.parametrize("net", ["rew", "pred"])
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

    rew = _build("rew", "film_head")
    pred = _build("pred", "film_head")

    r = rew(state, action, torch.randn(B, FILM["rew"]))
    pi, v = pred(state, torch.randn(B, FILM["pred"]))

    assert r.shape == (B, 1)
    assert pi.shape == (B, A)
    assert v.shape == (B, 1)
    for t in (r, pi, v):
        assert not torch.isnan(t).any()


def test_transition_net_plain_and_residual():
    """v5: TransitionNet is an ordinary SGD module with a residual output."""
    B = 3
    trans = TransitionNet(LATENT, JOINT, HIDDEN)
    assert all(isinstance(p, torch.Tensor) for p in trans.parameters())
    state = torch.randn(B, LATENT)
    action = torch.zeros(B, JOINT)
    action[:, 0] = 1.0
    s_next = trans(state, action)
    assert s_next.shape == (B, LATENT)
    assert not torch.isnan(s_next).any()
    # zeroing fc3 makes delta_s = LN(0) = 0 -> pure residual identity
    with torch.no_grad():
        trans.fc3.weight.zero_()
        trans.fc3.bias.zero_()
    assert torch.allclose(trans(state, action), state, atol=1e-5)


def test_transition_net_gradient_flow():
    trans = TransitionNet(LATENT, JOINT, HIDDEN)
    state = torch.randn(2, LATENT)
    action = torch.zeros(2, JOINT)
    action[:, 0] = 1.0
    (trans(state, action) ** 2).sum().backward()
    for name, p in trans.named_parameters():
        assert p.grad is not None, f"No grad for {name}"


def test_full_mode_forward_unchanged_shapes():
    B = 2
    state = torch.randn(B, LATENT)
    action = torch.zeros(B, JOINT)
    action[:, 0] = 1.0
    pred = _build("pred", "full")
    pi, v = pred(state, torch.randn(B, FULL["pred"]))
    assert pi.shape == (B, A) and v.shape == (B, 1)


@pytest.mark.parametrize("net", ["rew", "pred"])
def test_base_gen_generated_counts_and_groups(net):
    m = _build(net, "base_gen")
    assert m.generated_param_count == BASE[net]
    assert m.generated_param_count < m.total_params
    assert m.gen_groups == BASE_GROUPS[net]
    assert sum(m.gen_groups) == m.generated_param_count


@pytest.mark.parametrize("net", ["rew", "pred"])
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

    rew = _build("rew", "base_gen")
    pred = _build("pred", "base_gen")

    r = rew(state, action, torch.randn(B, BASE["rew"]))
    pi, v = pred(state, torch.randn(B, BASE["pred"]))

    assert r.shape == (B, 1)
    assert pi.shape == (B, A)
    assert v.shape == (B, 1)
    for t in (r, pi, v):
        assert not torch.isnan(t).any()


@pytest.mark.parametrize("net", ["rew", "pred"])
def test_lora_fc2_rank0_matches_film_head_count(net):
    # r=0 sentinel: no Af/Bf generated -> fc2 = gen_film only -> film_head counts.
    m0 = _build_lora(net, 0)
    assert m0.generated_param_count == FILM[net]


@pytest.mark.parametrize("net", ["rew", "pred"])
def test_lora_fc2_generated_counts_and_groups(net):
    m = _build_lora(net, LORA_R)
    assert m.generated_param_count == LORA[net]
    assert m.gen_groups == LORA_GROUPS[net]
    assert sum(m.gen_groups) == m.generated_param_count
    # a true middle rung: cheaper than full fc2 generation (base_gen), richer than film_head.
    assert FILM[net] < m.generated_param_count < BASE[net]
    # film-style fc1 + shared SGD fc2 base both present and trainable; no base_gen ln1.
    assert hasattr(m, "fc1") and hasattr(m, "fc2")
    assert m.fc1.weight.requires_grad and m.fc2.weight.requires_grad
    assert not hasattr(m, "ln1")


def test_lora_fc2_forward_shapes_no_nan():
    B = 4
    state = torch.randn(B, LATENT)
    action = torch.zeros(B, JOINT)
    action[:, 0] = 1.0

    rew = _build_lora("rew", LORA_R)
    pred = _build_lora("pred", LORA_R)

    r = rew(state, action, torch.randn(B, LORA["rew"]))
    pi, v = pred(state, torch.randn(B, LORA["pred"]))

    assert r.shape == (B, 1)
    assert pi.shape == (B, A)
    assert v.shape == (B, 1)
    for t in (r, pi, v):
        assert not torch.isnan(t).any()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
