"""The transition network must be able to emit signed deltas.

On 2026-07-19 the whole formal grid was traced back to this: ``mlp()`` in
``config/relation/model.py`` appends ``Linear -> ReLU -> LayerNorm`` after every
layer *including the output*, and ``fc_dynamic`` was the only head that did not
pass ``use_value_out=True`` to strip that trailing pair. Forcing a residual
transition's output non-negative and then renormalizing it let the branch
collapse: measured on two independently trained checkpoints, the output ReLU had
died to 1/128 live units (64/128 at init), the branch emitted a near-constant,
and ``state + hidden_state`` became a frozen-world identity map with

    max ||h'(a) - h'(NOOP)|| ~ 2e-7   for every action

at any action amplitude and any state scale. MCTS then saw identical values on
every branch, only the reward head could rank actions, and the policy distilled
to 100% HARVEST. Dead ReLUs get no gradient, so it was unrecoverable.

These tests pin the architectural invariant rather than a trained behaviour,
because the fix zero-initializes the final Linear — an untrained model is
action-invariant by construction, so only the *shape* is checkable up front.

Parametrized over both model paths: ``none`` (subjective HyperMAMuZeroNet /
ObjectiveDynamics) and ``no_subjective`` (plain MAMuZeroNet / DynamicsNetwork)
— the plain path carried the same vendor bug and is exercised by a real
ablation arm.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

ARMS = ["none", "no_subjective"]


def _model(arm: str):
    from hyper_mve.utils.configs import V4Config
    from hyper_mve.comparison import create_runner

    cfg = V4Config.from_preset("rel_duo")
    runner = create_runner(cfg, "mazero_mixed")
    runner._ablation = arm
    runner._weights_source = "test: architecture inspection only"
    return runner._lazy_model()


def _dynamics_module(arm: str):
    return _model(arm).dynamics_network


@pytest.mark.parametrize("arm", ARMS)
def test_fc_dynamic_output_is_a_bare_linear(arm):
    """No activation or normalization on the transition output.

    A residual transition has to produce signed, unbounded deltas. ReLU makes
    the output non-negative and, once dead, unrecoverable.
    """
    dyn = _dynamics_module(arm)
    last = dyn.fc_dynamic[-1]
    assert isinstance(last, nn.Linear), (
        f"[{arm}] fc_dynamic must end in a bare Linear, found "
        f"{type(last).__name__}. Pass use_value_out=True to mlp() — see this "
        "module's docstring."
    )
    for i, layer in enumerate(dyn.fc_dynamic):
        if isinstance(layer, (nn.ReLU, nn.Tanh, nn.LayerNorm)):
            assert i < len(dyn.fc_dynamic) - 1, "activation/norm on the output"


@pytest.mark.parametrize("arm", ARMS)
def test_every_prediction_head_strips_its_output_activation(arm):
    """fc_dynamic was the odd one out; keep all four heads consistent."""
    model = _model(arm)

    checked = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Sequential):
            continue
        if not name.endswith(("fc_dynamic", "fc_reward", "fc_value", "fc_policy")):
            continue
        assert isinstance(module[-1], nn.Linear), (
            f"[{arm}] {name} ends in {type(module[-1]).__name__}, not Linear"
        )
        checked += 1
    assert checked >= 1, "no prediction heads found — model layout changed"


@pytest.mark.parametrize("arm", ARMS)
def test_dynamics_is_action_sensitive_once_the_output_layer_is_nonzero(arm):
    """With a non-degenerate output layer, different actions must move the state.

    The fix zero-initializes the final Linear (standard residual practice), so
    this perturbs it first — otherwise the model is action-invariant by
    construction at init and the test would pass vacuously on a broken build.
    """
    dyn = _dynamics_module(arm)
    torch.manual_seed(0)
    last = dyn.fc_dynamic[-1]
    device = last.weight.device
    with torch.no_grad():
        last.weight.normal_(0.0, 0.05)

    B, N, H = 1, 2, last.out_features
    A = dyn.attention_stack[0].in_features - H
    hidden = torch.randn(B, N, H, device=device)

    def step(a0: int) -> torch.Tensor:
        onehot = torch.zeros(B, N, A, device=device)
        onehot[0, 0, a0] = 1.0
        onehot[0, 1, A - 1] = 1.0
        with torch.no_grad():
            out = dyn(hidden, onehot)
        # plain DynamicsNetwork returns (state, reward); ObjectiveDynamics
        # returns state only
        return out[0] if isinstance(out, tuple) else out

    ref = step(0)
    diffs = [torch.linalg.norm(step(a) - ref).item() for a in range(1, A)]
    scale = torch.linalg.norm(ref).item()
    assert max(diffs) / scale > 1e-3, (
        f"[{arm}] transition is action-blind: max relative response "
        f"{max(diffs)/scale:.2e}. The learned model cannot represent what "
        "actions do, so search ranks every branch identically and only the "
        "reward head differentiates."
    )
