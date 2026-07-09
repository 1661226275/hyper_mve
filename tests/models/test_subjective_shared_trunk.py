"""Shared subjective hypernet trunk (Idea 1) tests.

When share_subjective_trunk=True, hyper_rew + hyper_pred collapse into one shared trunk
(SubjectiveHyperNet) with two heads (theta_rew / theta_pred). the objective path no longer exists (v5). Each head keeps its own output_scale + grouped RMS. Detach (D5) under sharing
detaches the TRUNK OUTPUT for the pred head (stronger than the unshared input-detach):
value loss then trains only pred_head, not the shared trunk or the context (ctx_aug).
"""
import pytest
import torch

from hyper_mve.models.hyper_network import DualHyperNetwork, SubjectiveHyperNet

CTX_AUG = 64                      # v5: role (32) + belief (32)
REW_PC, PRED_PC = 16897, 17671    # base_gen counts (latent=64, hidden=128, A=6, N=2)
REW_GROUPS, PRED_GROUPS = [256, 16641], [256, 17415]


def _dual(share, detach=False):
    torch.manual_seed(0)
    return DualHyperNetwork(
        ctx_aug_dim=CTX_AUG,
        rew_param_count=REW_PC, pred_param_count=PRED_PC,
        rew_output_scale_init=0.1, pred_output_scale_init=0.1,
        detach_pred_context=detach,
        rew_output_groups=REW_GROUPS, pred_output_groups=PRED_GROUPS,
        share_subjective_trunk=share,
    )


def _sub(detach):
    torch.manual_seed(0)
    return SubjectiveHyperNet(
        ctx_aug_dim=CTX_AUG, rew_param_count=REW_PC, pred_param_count=PRED_PC,
        rew_output_scale_init=0.1, pred_output_scale_init=0.1,
        detach_pred_context=detach,
        rew_output_groups=REW_GROUPS, pred_output_groups=PRED_GROUPS,
    )


def _trunk_first_linear(sub):
    return [m for m in sub.trunk if isinstance(m, torch.nn.Linear)][0]


def test_shared_trunk_shapes_and_modules():
    net = _dual(share=True)
    assert hasattr(net, "subjective") and not hasattr(net, "hyper_rew")
    assert not hasattr(net, "hyper_pred")
    assert not hasattr(net, "hyper_trans")  # v5: objective path deleted
    theta_rew, theta_pred = net.forward_subjective(torch.randn(4, CTX_AUG))
    assert theta_rew.shape == (4, REW_PC)
    assert theta_pred.shape == (4, PRED_PC)
    assert not torch.isnan(theta_rew).any() and not torch.isnan(theta_pred).any()


def test_unshared_default_unchanged():
    net = _dual(share=False)
    assert hasattr(net, "hyper_rew") and hasattr(net, "hyper_pred")
    assert not hasattr(net, "subjective")


def test_per_head_scale_independent():
    sub = _sub(detach=False)
    assert sub.rew_output_scale.item() == pytest.approx(0.1)
    assert sub.pred_output_scale.item() == pytest.approx(0.1)
    assert sub.rew_output_scale is not sub.pred_output_scale  # distinct Parameters


def test_detach_true_blocks_trunk_and_ctx_grad_for_pred():
    sub = _sub(detach=True)
    ctx = torch.randn(4, CTX_AUG, requires_grad=True)
    _, theta_pred = sub(ctx)
    theta_pred.sum().backward()
    # value path trains pred_head but NOT the shared trunk or the context.
    assert sub.pred_head.weight.grad is not None
    assert _trunk_first_linear(sub).weight.grad is None
    assert ctx.grad is None or torch.count_nonzero(ctx.grad) == 0


def test_detach_true_rew_path_trains_trunk_and_ctx():
    sub = _sub(detach=True)
    ctx = torch.randn(4, CTX_AUG, requires_grad=True)
    theta_rew, _ = sub(ctx)
    theta_rew.sum().backward()
    assert _trunk_first_linear(sub).weight.grad is not None
    assert ctx.grad is not None and torch.count_nonzero(ctx.grad) > 0


def test_detach_false_trains_trunk_and_ctx_from_pred():
    sub = _sub(detach=False)
    ctx = torch.randn(4, CTX_AUG, requires_grad=True)
    _, theta_pred = sub(ctx)
    theta_pred.sum().backward()
    assert _trunk_first_linear(sub).weight.grad is not None
    assert ctx.grad is not None and torch.count_nonzero(ctx.grad) > 0


def test_shared_head_grouped_rms_meaningful_film():
    sub = _sub(detach=False)
    theta_rew, theta_pred = sub(torch.randn(16, CTX_AUG))
    # film segment = first 256 of each head's vector; gamma occupies [0:128].
    assert theta_rew[:, 0:128].abs().mean().item() >= 1e-2
    assert theta_pred[:, 0:128].abs().mean().item() >= 1e-2


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
