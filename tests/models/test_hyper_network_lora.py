"""Output-layer LoRA (hyper_output_rank) tests for HyperNetMLP + DualHyperNetwork.

When ``output_rank=r`` is set, the dense ``output_layer = Linear(prev, pc)`` is replaced by a
LoRA factorization ``Linear(prev, r, bias=False) -> Linear(r, pc, bias=True)`` (params
``prev*pc`` -> ``prev*r + r*pc + pc``). A=orthogonal, B=small_init (NOT zero, else the grouped
RMS divides by the 1e-8 floor -> step-0 gradient spike). ``output_rank=None`` is byte-identical
to the prior dense path. Applied uniformly to hyper_trans/hyper_rew/hyper_pred; the shared
SubjectiveHyperNet branch is not yet wired (raises NotImplementedError).
"""
import pytest
import torch

from hyper_mve.models.hyper_network import DualHyperNetwork, HyperNetMLP


def test_lora_factorization_modules_and_shape():
    r, pc = 32, 8768
    mlp = HyperNetMLP(input_dim=16, output_dim=pc, hidden_dims=(256, 256),
                      output_scale_init=0.1, output_rank=r)
    assert not hasattr(mlp, "output_layer")
    assert mlp.output_A.weight.shape == (r, 256) and mlp.output_A.bias is None
    assert mlp.output_B.weight.shape == (pc, r) and mlp.output_B.bias is not None
    out = mlp(torch.randn(4, 16))
    assert out.shape == (4, pc)
    assert torch.isfinite(out).all()


def test_lora_param_count_formula():
    r, pc, prev = 32, 8768, 256
    mlp = HyperNetMLP(input_dim=16, output_dim=pc, hidden_dims=(256, 256), output_rank=r)
    n = (mlp.output_A.weight.numel() + mlp.output_B.weight.numel()
         + mlp.output_B.bias.numel())
    assert n == prev * r + r * pc + pc   # 8192 + 280576 + 8768 = 297536


def test_lora_grouped_rms_keeps_film_meaningful():
    # film_head-like groups [512, 8256]; the FiLM segment must stay ~scale after grouped RMS.
    mlp = HyperNetMLP(input_dim=16, output_dim=8768, output_groups=[512, 8256],
                      output_scale_init=0.1, output_rank=32)
    out = mlp(torch.randn(8, 16))
    assert out[:, :512].abs().mean().item() >= 1e-2


def test_lora_both_factors_get_grad():
    mlp = HyperNetMLP(input_dim=16, output_dim=200, output_rank=8, output_scale_init=0.1)
    mlp(torch.randn(4, 16)).sum().backward()
    assert mlp.output_A.weight.grad is not None and mlp.output_A.weight.grad.abs().sum() > 0
    assert mlp.output_B.weight.grad is not None and mlp.output_B.weight.grad.abs().sum() > 0


def test_output_rank_none_is_dense():
    mlp = HyperNetMLP(input_dim=16, output_dim=100, output_scale_init=0.1, output_rank=None)
    assert hasattr(mlp, "output_layer")
    assert not hasattr(mlp, "output_A")
    assert mlp(torch.randn(2, 16)).shape == (2, 100)


def test_dual_threads_rank_to_all_three():
    net = DualHyperNetwork(
        c_ctx_dim=16, ctx_aug_dim=80,
        trans_param_count=8768, rew_param_count=641, pred_param_count=1415,
        trans_output_scale_init=0.1, rew_output_scale_init=0.1, pred_output_scale_init=0.1,
        hyper_output_rank=32,
    )
    for sub in (net.hyper_trans, net.hyper_rew, net.hyper_pred):
        assert sub.output_rank == 32
        assert hasattr(sub, "output_A") and hasattr(sub, "output_B")
    # forward still produces the right shapes under LoRA.
    theta_state = net.forward_trans(torch.randn(2, 16))
    theta_rew, theta_pred = net.forward_subjective(torch.randn(2, 80))
    assert theta_state.shape == (2, 8768)
    assert theta_rew.shape == (2, 641) and theta_pred.shape == (2, 1415)


def test_dual_share_plus_rank_raises():
    with pytest.raises(NotImplementedError, match="SubjectiveHyperNet"):
        DualHyperNetwork(
            c_ctx_dim=16, ctx_aug_dim=80,
            trans_param_count=8768, rew_param_count=641, pred_param_count=1415,
            share_subjective_trunk=True, hyper_output_rank=32,
        )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
