"""Unit tests for ``hyper_mve.models.tri_context_encoder`` + ``belief_encoder``
(Pkg-03 spec 01 §5.1)."""
from __future__ import annotations

import pytest
import torch

from hyper_mve.configs import V4Config
from hyper_mve.models import TriContextEncoder, BeliefEncoder


@pytest.fixture
def cfg_medium():
    return V4Config.from_preset("medium")


def test_output_dim(cfg_medium):
    """ctx_i shape (B, N, 80) 精确."""
    encoder = TriContextEncoder(cfg_medium.env, cfg_medium.model)
    B, N = 2, 4
    c_t = torch.rand(B)
    agent_ids = torch.arange(N).unsqueeze(0).expand(B, N).contiguous()
    types = torch.zeros(B, N, dtype=torch.long)
    caps = torch.rand(B, N, 4)
    c_hat = torch.rand(B, N)
    z_hat = torch.softmax(torch.randn(B, N, N - 1, 2), dim=-1)

    ctx_i = encoder.forward(c_t, agent_ids, types, caps, (c_hat, z_hat))
    assert ctx_i.shape == (B, N, 80)


def test_d_ctx_aug_property(cfg_medium):
    """d_ctx_aug = 16 + 32 + 32 = 80."""
    encoder = TriContextEncoder(cfg_medium.env, cfg_medium.model)
    assert encoder.d_ctx_aug == 80
    assert encoder.d_c == 16
    assert encoder.d_role == 32
    assert encoder.d_belief == 32


def test_role_dim_exact(cfg_medium):
    """role 内部 8+8+16 = 32 精确填满, 无 pad."""
    encoder = TriContextEncoder(cfg_medium.env, cfg_medium.model)
    assert cfg_medium.model.d_id_emb + cfg_medium.model.d_type_emb + cfg_medium.model.d_cap_emb == 32
    assert encoder.role_encoder.d_role == 32


def test_gradient_flow(cfg_medium):
    """反向传播覆盖三路 + 三 head 所有可学参数."""
    encoder = TriContextEncoder(cfg_medium.env, cfg_medium.model)
    B, N = 2, 4
    c_t = torch.rand(B, requires_grad=False)
    agent_ids = torch.arange(N).unsqueeze(0).expand(B, N).contiguous()
    types = torch.zeros(B, N, dtype=torch.long)
    caps = torch.rand(B, N, 4)
    c_hat = torch.rand(B, N, requires_grad=True)
    z_hat = torch.softmax(torch.randn(B, N, N - 1, 2, requires_grad=True), dim=-1)

    ctx_i = encoder.forward(c_t, agent_ids, types, caps, (c_hat, z_hat))
    loss = (ctx_i ** 2).sum()
    loss.backward()

    # 三路 sub-encoder + LN + projections 所有可学参数有梯度
    for name, p in encoder.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No grad for {name}"
            assert p.grad.norm() > 0, f"Zero grad for {name}"


def test_concat_order(cfg_medium):
    """ctx_i 切片顺序: [0:16] c_ctx, [16:48] role, [48:80] belief."""
    encoder = TriContextEncoder(cfg_medium.env, cfg_medium.model)
    B, N = 2, 4
    c_t = torch.rand(B)
    agent_ids = torch.arange(N).unsqueeze(0).expand(B, N).contiguous()
    types = torch.zeros(B, N, dtype=torch.long)
    caps = torch.rand(B, N, 4)
    c_hat = torch.rand(B, N)
    z_hat = torch.softmax(torch.randn(B, N, N - 1, 2), dim=-1)

    ctx_i = encoder.forward(c_t, agent_ids, types, caps, (c_hat, z_hat))

    # c_ctx 子段应在 [0:16]
    c_ctx_only = encoder.forward_c_ctx_only(c_t)        # (B, 16)
    c_ctx_broadcast = c_ctx_only.unsqueeze(1).expand(B, N, 16)
    assert torch.allclose(ctx_i[..., :16], c_ctx_broadcast)


def test_permutation_invariance_belief_mean_pool(cfg_medium):
    """mean pool 下, 改变 z_hat 对手维度顺序, belief 子段不变."""
    cfg = cfg_medium
    encoder = TriContextEncoder(cfg.env, cfg.model)
    B, N = 2, 4

    c_t = torch.rand(B)
    agent_ids = torch.arange(N).unsqueeze(0).expand(B, N).contiguous()
    types = torch.zeros(B, N, dtype=torch.long)
    caps = torch.rand(B, N, 4)
    c_hat = torch.rand(B, N)
    z_hat = torch.softmax(torch.randn(B, N, N - 1, 2), dim=-1)

    ctx_i_orig = encoder.forward(c_t, agent_ids, types, caps, (c_hat, z_hat))

    # 打乱 z_hat 在 N-1 维度的顺序 (每个 agent 内独立打乱)
    z_hat_shuffled = z_hat.clone()
    perm = torch.randperm(N - 1)
    z_hat_shuffled = z_hat_shuffled[:, :, perm, :]

    ctx_i_shuf = encoder.forward(c_t, agent_ids, types, caps, (c_hat, z_hat_shuffled))

    # belief 子段 (ctx_i[..., 48:80]) 应不变 (mean pool 是 set-invariant)
    assert torch.allclose(
        ctx_i_orig[..., 48:80], ctx_i_shuf[..., 48:80], atol=1e-5,
    )


def test_c_ctx_only_dim(cfg_medium):
    encoder = TriContextEncoder(cfg_medium.env, cfg_medium.model)
    c_t = torch.rand(8)
    c_ctx = encoder.forward_c_ctx_only(c_t)
    assert c_ctx.shape == (8, 16)


def test_c_t_input_shapes(cfg_medium):
    """c_t 接受 (B,) 或 (B, 1) 两种形式."""
    encoder = TriContextEncoder(cfg_medium.env, cfg_medium.model)
    B, N = 2, 4
    agent_ids = torch.arange(N).unsqueeze(0).expand(B, N).contiguous()
    types = torch.zeros(B, N, dtype=torch.long)
    caps = torch.rand(B, N, 4)
    c_hat = torch.rand(B, N)
    z_hat = torch.softmax(torch.randn(B, N, N - 1, 2), dim=-1)

    # (B,)
    c_t_1d = torch.rand(B)
    ctx_1d = encoder.forward(c_t_1d, agent_ids, types, caps, (c_hat, z_hat))

    # (B, 1)
    c_t_2d = c_t_1d.unsqueeze(-1)
    ctx_2d = encoder.forward(c_t_2d, agent_ids, types, caps, (c_hat, z_hat))

    assert torch.allclose(ctx_1d, ctx_2d)


def test_n_variation(cfg_medium):
    """N=2 (Easy) 与 N=8 (Hard) 均应可运行."""
    from dataclasses import replace
    from hyper_mve.schemas import AgentType

    # N=2
    cfg2 = replace(cfg_medium, env=replace(cfg_medium.env, N=2,
        type_assignment=(AgentType.ALPHA, AgentType.BETA)))
    enc2 = TriContextEncoder(cfg2.env, cfg2.model)
    c_t = torch.rand(1)
    aid = torch.tensor([[0, 1]])
    typ = torch.tensor([[0, 1]])
    cap = torch.rand(1, 2, 4)
    c_h = torch.rand(1, 2)
    z_h = torch.softmax(torch.randn(1, 2, 1, 2), dim=-1)
    ctx2 = enc2.forward(c_t, aid, typ, cap, (c_h, z_h))
    assert ctx2.shape == (1, 2, 80)

    # N=8
    cfg8 = replace(cfg_medium, env=replace(cfg_medium.env, N=8,
        type_assignment=(AgentType.ALPHA,) * 4 + (AgentType.BETA,) * 4))
    enc8 = TriContextEncoder(cfg8.env, cfg8.model)
    aid8 = torch.arange(8).unsqueeze(0)
    typ8 = torch.zeros(1, 8, dtype=torch.long)
    cap8 = torch.rand(1, 8, 4)
    c_h8 = torch.rand(1, 8)
    z_h8 = torch.softmax(torch.randn(1, 8, 7, 2), dim=-1)
    ctx8 = enc8.forward(c_t, aid8, typ8, cap8, (c_h8, z_h8))
    assert ctx8.shape == (1, 8, 80)


# ====== BeliefEncoder 独立单测 (P5 拆出后可独立测试) ======

def test_belief_encoder_output_shape(cfg_medium):
    """BeliefEncoder 输出 (B, N, 32)."""
    be = BeliefEncoder(cfg_medium.env, cfg_medium.model)
    B, N = 2, 4
    c_hat = torch.rand(B, N)
    z_hat = torch.softmax(torch.randn(B, N, N - 1, 2), dim=-1)

    out = be(c_hat, z_hat)
    assert out.shape == (B, N, 32)


def test_belief_encoder_no_internal_ln(cfg_medium):
    """P7: BeliefEncoder 不含内部 LayerNorm."""
    be = BeliefEncoder(cfg_medium.env, cfg_medium.model)
    has_ln = any(isinstance(m, torch.nn.LayerNorm) for m in be.modules())
    assert not has_ln, "BeliefEncoder should NOT contain internal LayerNorm (P7)"


def test_belief_encoder_gradient_flow(cfg_medium):
    be = BeliefEncoder(cfg_medium.env, cfg_medium.model)
    B, N = 2, 4
    c_hat = torch.rand(B, N, requires_grad=True)
    z_hat = torch.softmax(torch.randn(B, N, N - 1, 2, requires_grad=True), dim=-1)

    out = be(c_hat, z_hat)
    loss = (out ** 2).sum()
    loss.backward()

    for name, p in be.named_parameters():
        assert p.grad is not None, f"No grad for {name}"
        assert p.grad.norm() > 0, f"Zero grad for {name}"


def test_belief_encoder_concat_order(cfg_medium):
    """belief_vec[..., :16] 来自 proj_c_hat, [..., 16:32] 来自 proj_z_pooled."""
    be = BeliefEncoder(cfg_medium.env, cfg_medium.model)
    B, N = 1, 4

    # 给一个固定输入
    c_hat = torch.rand(B, N)
    z_hat = torch.softmax(torch.randn(B, N, N - 1, 2), dim=-1)

    out = be(c_hat, z_hat)

    # 单独走 proj_c_hat 路径验证
    c_proj_alone = be.proj_c_hat(c_hat.unsqueeze(-1))
    assert torch.allclose(out[..., :16], c_proj_alone)
