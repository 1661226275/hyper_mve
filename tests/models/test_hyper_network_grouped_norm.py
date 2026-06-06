"""Grouped RMS normalization (film_head) tests for HyperNetMLP.

ACCEPTANCE CONTRACT for the FiLM refactor: in film_head mode the generated FiLM gamma
must have a MEANINGFUL magnitude (mean|gamma| >= 1e-2), not the ~output_scale/sqrt(dim)
that whole-vector L2 norm pins it to (0.1/sqrt(512) ~ 4e-3 -> (1+gamma)~1 -> no
modulation). Per-group RMS normalization makes each element's magnitude ~ output_scale
regardless of segment dim. Whole-vector L2 (output_groups=None) must be unchanged.
"""
import pytest
import torch

from hyper_mve.models.hyper_network import HyperNetMLP

# pred-shaped film_head output: film 512 + head 903. Within the 512 film segment the
# split layout is [gamma1(128), beta1(128), gamma2(128), beta2(128)] -> gamma occupies
# [0:128] and [256:384].
FILM_TOTAL, HEAD_TOTAL = 512, 903
OUT = FILM_TOTAL + HEAD_TOTAL
SCALE = 0.1


def _mlp(output_groups):
    torch.manual_seed(0)
    return HyperNetMLP(
        input_dim=80, output_dim=OUT, hidden_dims=(64, 64),
        norm_output=True, output_scale_init=SCALE, output_groups=output_groups,
    )


def _gamma(out):
    return torch.cat([out[:, 0:128], out[:, 256:384]], dim=-1)


def test_film_gamma_magnitude_meaningful():
    """The headline guard: grouped RMS norm keeps |gamma| ~ output_scale (~0.08)."""
    out = _mlp([FILM_TOTAL, HEAD_TOTAL])(torch.randn(16, 80))
    mean_abs_gamma = _gamma(out).abs().mean().item()
    assert mean_abs_gamma >= 1e-2, (
        f"mean|gamma|={mean_abs_gamma:.4g} < 1e-2: grouped RMS norm is not giving FiLM a "
        "meaningful magnitude (whole-vector L2 would pin it ~output_scale/sqrt(dim))."
    )


def test_grouped_rms_head_near_output_scale():
    out = _mlp([FILM_TOTAL, HEAD_TOTAL])(torch.randn(16, 80))
    head_rms = out[:, FILM_TOTAL:].pow(2).mean(dim=-1).sqrt().mean().item()
    assert abs(head_rms - SCALE) < 0.03, f"head RMS {head_rms:.4g} far from {SCALE}"


def test_film_gamma_differentiates_context():
    mlp = _mlp([FILM_TOTAL, HEAD_TOTAL])
    a = mlp(torch.randn(1, 80))
    b = mlp(torch.randn(1, 80))
    assert (a - b).norm().item() > 1e-3


def test_whole_vector_l2_unchanged():
    # output_groups=None -> legacy path: ||out|| == output_scale for every sample.
    out = _mlp(None)(torch.randn(8, 80))
    norms = torch.linalg.norm(out, dim=-1)
    assert torch.allclose(norms, torch.full_like(norms, SCALE), atol=1e-5)


def test_output_groups_must_sum_to_output_dim():
    with pytest.raises(AssertionError):
        HyperNetMLP(input_dim=80, output_dim=OUT, output_groups=[FILM_TOTAL, HEAD_TOTAL + 1])


def test_base_gen_grouped_rms_film_and_weight():
    # base_gen trans-shaped output: film 256 (fc2 gamma/beta) + weight 24768 (fc2 w+b +
    # head). The weight group is far larger than the film group, so per-group RMS (not
    # whole-vector L2) is what keeps the small film group's gamma meaningful.
    film_total, weight_total = 256, 24768
    out_dim = film_total + weight_total
    torch.manual_seed(0)
    mlp = HyperNetMLP(
        input_dim=80, output_dim=out_dim, hidden_dims=(64, 64),
        norm_output=True, output_scale_init=SCALE,
        output_groups=[film_total, weight_total],
    )
    out = mlp(torch.randn(16, 80))
    # fc2 gamma occupies the first 128 of the 256-wide film segment.
    assert out[:, 0:128].abs().mean().item() >= 1e-2
    # weight segment RMS ~ output_scale (starts near fan-in standard init).
    weight_rms = out[:, film_total:].pow(2).mean(dim=-1).sqrt().mean().item()
    assert abs(weight_rms - SCALE) < 0.03


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
