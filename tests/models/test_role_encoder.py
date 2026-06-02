"""Unit tests for ``hyper_mve.models.role_encoder`` (Pkg-03 spec 03 §5.1)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hyper_mve.models.role_encoder import RoleEncoder


def test_output_shape():
    """role shape (B, N, 32)."""
    enc = RoleEncoder(N=4)
    B = 2
    agent_ids = torch.arange(4).unsqueeze(0).expand(B, 4)
    types = torch.tensor([[0, 0, 1, 1]] * B)
    caps = torch.rand(B, 4, 4)

    role = enc(agent_ids, types, caps)
    assert role.shape == (B, 4, 32)


def test_role_dim_exact_fill_v4():
    """v4 关键: d_role = d_id + d_type + d_cap = 8 + 8 + 16 = 32 精确, 无 pad."""
    enc = RoleEncoder(N=4, d_id_emb=8, d_type_emb=8, d_cap_emb=16)
    assert enc.d_role == 32
    assert enc.d_id_emb + enc.d_type_emb + enc.d_cap_emb == enc.d_role


def test_role_dim_assertion_on_mismatch():
    """d_id + d_type + d_cap != 32 应抛."""
    with pytest.raises(AssertionError, match="d_role"):
        RoleEncoder(N=4, d_id_emb=8, d_type_emb=4, d_cap_emb=16)  # 28 != 32


def test_type_emb_distinct_for_alpha_beta():
    """v4 关键: type α 与 type β 的 embedding 应不同 (训练后)."""
    enc = RoleEncoder(N=2)
    # 初始随机, 仅检查表大小
    assert enc.type_emb.num_embeddings == 2
    assert enc.type_emb.embedding_dim == 8

    # 同一 agent_id 不同 type 输出 role 不同
    agent_ids = torch.tensor([[0]])
    cap = torch.rand(1, 1, 4)
    role_alpha = enc(agent_ids, torch.tensor([[0]]), cap)
    role_beta = enc(agent_ids, torch.tensor([[1]]), cap)
    # type 子段 (8:16) 必然不同
    assert not torch.allclose(role_alpha[..., 8:16], role_beta[..., 8:16])


def test_id_emb_distinct():
    """不同 agent_id 的 id_emb 子段不同."""
    enc = RoleEncoder(N=4)
    types = torch.zeros(1, 4, dtype=torch.long)
    caps = torch.rand(1, 4, 4)

    role = enc(torch.tensor([[0, 1, 2, 3]]), types, caps)
    # id_emb 子段 (0:8) 在 4 个 agent 间不同
    for i in range(4):
        for j in range(i + 1, 4):
            assert not torch.allclose(role[0, i, :8], role[0, j, :8])


def test_self_info_severity_only_own_type():
    """Self-Info 严格性: 本模块仅看 own type, 不接受 (N, N) 全 agent type 矩阵."""
    enc = RoleEncoder(N=4)
    # 正确: types shape (B, N) - 每个 agent 一个 type (自己的)
    agent_ids = torch.tensor([[0, 1, 2, 3]])
    types = torch.tensor([[0, 0, 1, 1]])      # 每个 agent 自己的 type
    caps = torch.rand(1, 4, 4)

    role = enc(agent_ids, types, caps)
    assert role.shape == (1, 4, 32)

    # 错误用法 (会被 shape check 捕获):
    # 假设 trainer 误传 types shape (B, N, N) (全 N×N 类型矩阵)
    bad_types = torch.zeros(1, 4, 4, dtype=torch.long)
    with pytest.raises((AssertionError, RuntimeError)):
        enc(agent_ids, bad_types, caps)


def test_cap_normalization_handles_scale_diff():
    """C 修订: cap 4 维量级悬殊在内部 normalize 后被消除.

    验证: 输入 raw cap (含 ζ=30 vs η=0.5 差 60x), normalize 后所有维 ∈ [0, 1].
    """
    enc = RoleEncoder(N=4)
    agent_ids = torch.tensor([[0, 1, 2, 3]])
    types = torch.tensor([[0, 0, 1, 1]])
    # 极端 cap: η=0.5, φ_fov=4, ν=0.8, ζ=30 (规模差 60x)
    caps = torch.tensor([[
        [0.5, 4.0, 0.8, 30.0],
        [1.5, 2.0, 1.0, 10.0],
        [1.0, 3.0, 0.9, 20.0],
        [0.8, 4.0, 0.85, 25.0],
    ]], dtype=torch.float32)

    # 直接验证 _normalize_caps 输出 ∈ [0, 1]^4
    caps_normed = enc._normalize_caps(caps)
    assert (caps_normed >= 0.0).all()
    assert (caps_normed <= 1.0).all()
    # 边界: agent 0 (η=0.5) → 第 0 维 = 0.0; agent 1 (η=1.5) → 第 0 维 = 1.0
    assert torch.isclose(caps_normed[0, 0, 0], torch.tensor(0.0))
    assert torch.isclose(caps_normed[0, 1, 0], torch.tensor(1.0))

    # forward 输出无 NaN/Inf
    role = enc(agent_ids, types, caps)
    assert not torch.isnan(role).any()
    assert not torch.isinf(role).any()


def test_cap_normalization_matches_pkg01_normalize():
    """C 修订: _normalize_caps 与 Pkg-01 CapabilityVector.normalize() 数学等价."""
    from hyper_mve.schemas import CapabilityVector

    enc = RoleEncoder(N=4)
    cap_dataclass = CapabilityVector(eta=1.2, phi_fov=2.5, nu=0.85, zeta=15.0)

    # Pkg-01 path
    normed_pkg01 = cap_dataclass.normalize()  # (4,) numpy

    # Pkg-03 path (vectorized)
    cap_tensor = torch.tensor([[[1.2, 2.5, 0.85, 15.0]]], dtype=torch.float32)
    normed_pkg03 = enc._normalize_caps(cap_tensor)  # (1, 1, 4)

    # 数学等价 (容差 1e-6)
    assert np.allclose(normed_pkg03[0, 0].numpy(), normed_pkg01, atol=1e-6)


def test_cap_mlp_no_internal_layernorm():
    """P7 修订: cap_mlp 内部不含 LayerNorm (LN 责任在 TriContextEncoder)."""
    enc = RoleEncoder(N=4)
    has_ln_in_cap_mlp = any(
        isinstance(m, torch.nn.LayerNorm) for m in enc.cap_mlp.modules()
    )
    assert not has_ln_in_cap_mlp, (
        "cap_mlp should NOT contain LayerNorm (P7: LN 责任集中在 TriContextEncoder.ln_role)"
    )


def test_cap_norm_buffer_registered():
    """C 修订: _cap_lo / _cap_hi 注册为 buffer (随 device 迁移)."""
    enc = RoleEncoder(N=4)
    # buffer 名字在 named_buffers 中
    buffer_names = [name for name, _ in enc.named_buffers()]
    assert "_cap_lo" in buffer_names
    assert "_cap_hi" in buffer_names
    # 数值与 Pkg-01 常量一致
    from hyper_mve.schemas.capability import CAP_NORM_LO, CAP_NORM_HI
    assert torch.allclose(enc._cap_lo, torch.tensor(CAP_NORM_LO))
    assert torch.allclose(enc._cap_hi, torch.tensor(CAP_NORM_HI))


def test_cap_norm_buffer_device_migration():
    """C 修订: _cap_lo / _cap_hi 随 module .cuda() 迁移."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    enc = RoleEncoder(N=4).cuda()
    assert enc._cap_lo.device.type == "cuda"
    assert enc._cap_hi.device.type == "cuda"


def test_gradient_flow():
    """反向传播覆盖 id_emb + type_emb + cap_mlp 所有参数."""
    enc = RoleEncoder(N=4)
    agent_ids = torch.tensor([[0, 1, 2, 3]])
    types = torch.tensor([[0, 0, 1, 1]])
    caps = torch.rand(1, 4, 4)

    role = enc(agent_ids, types, caps)
    loss = (role ** 2).sum()
    loss.backward()

    for name, p in enc.named_parameters():
        assert p.grad is not None, f"No grad for {name}"
        assert p.grad.abs().sum() > 0, f"Zero grad for {name}"


def test_episode_invariance():
    """同一 (id, type, cap) 多次调用输出一致 (无随机性, episode 内可缓存)."""
    enc = RoleEncoder(N=4)
    enc.eval()  # 关闭 dropout (本模块无 dropout 但仍 eval)

    agent_ids = torch.tensor([[0, 1, 2, 3]])
    types = torch.tensor([[0, 0, 1, 1]])
    caps = torch.rand(1, 4, 4)

    role_1 = enc(agent_ids, types, caps)
    role_2 = enc(agent_ids, types, caps)
    assert torch.allclose(role_1, role_2)


def test_param_count_medium():
    """Medium preset (N=4) 参数量 ~432."""
    enc = RoleEncoder(N=4, d_id_emb=8, d_type_emb=8, d_cap_emb=16)
    total = sum(p.numel() for p in enc.parameters())
    assert 400 < total < 500


def test_n_scaling():
    """id_emb 表大小随 N 缩放."""
    enc2 = RoleEncoder(N=2)
    enc8 = RoleEncoder(N=8)
    assert enc2.id_emb.num_embeddings == 2
    assert enc8.id_emb.num_embeddings == 8
