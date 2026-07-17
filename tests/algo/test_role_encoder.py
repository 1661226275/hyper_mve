"""Unit tests for ``hyper_mve.algo.modules.role_encoder`` (v5 Pkg-09: id + own row)."""
from __future__ import annotations

import pytest
import torch

from hyper_mve.algo.modules.role_encoder import RoleEncoder


def test_output_shape():
    """role shape (B, N, 32)."""
    enc = RoleEncoder(N=4)
    B = 2
    agent_ids = torch.arange(4).unsqueeze(0).expand(B, 4)
    rows = torch.rand(B, 4, 3) * 2 - 1

    role = enc(agent_ids, rows)
    assert role.shape == (B, 4, 32)


def test_role_dim_exact_fill_v5():
    """v5: d_role = d_id + d_row = 8 + 24 = 32 exact, no pad."""
    enc = RoleEncoder(N=4, d_id_emb=8, d_row_emb=24)
    assert enc.d_role == 32
    assert enc.d_id_emb + enc.d_row_emb == enc.d_role


def test_role_dim_assertion_on_mismatch():
    with pytest.raises(AssertionError, match="d_role"):
        RoleEncoder(N=4, d_id_emb=8, d_row_emb=16)  # 24 != 32


def test_row_changes_role():
    """Different own rows ⇒ different role vectors (the v5 role IS the row)."""
    enc = RoleEncoder(N=2)
    agent_ids = torch.tensor([[0]])
    role_ally = enc(agent_ids, torch.tensor([[[1.0]]]))
    role_rival = enc(agent_ids, torch.tensor([[[-1.0]]]))
    # row sub-segment (8:32) must differ
    assert not torch.allclose(role_ally[..., 8:], role_rival[..., 8:])
    # id sub-segment (0:8) is row-independent
    assert torch.allclose(role_ally[..., :8], role_rival[..., :8])


def test_id_emb_distinct():
    """Different agent_ids get distinct id_emb sub-segments."""
    enc = RoleEncoder(N=4)
    rows = torch.rand(1, 4, 3)
    role = enc(torch.tensor([[0, 1, 2, 3]]), rows)
    for i in range(4):
        for j in range(i + 1, 4):
            assert not torch.allclose(role[0, i, :8], role[0, j, :8])


def test_self_info_row_width_strict():
    """Self-Info: rows must be (B, N, N-1) — a full (B, N, N) W is rejected."""
    enc = RoleEncoder(N=4)
    agent_ids = torch.tensor([[0, 1, 2, 3]])
    role = enc(agent_ids, torch.rand(1, 4, 3))
    assert role.shape == (1, 4, 32)

    with pytest.raises(AssertionError, match="rows shape"):
        enc(agent_ids, torch.rand(1, 4, 4))   # full W leak


def test_row_mlp_no_internal_layernorm():
    """P7: row_mlp contains no LayerNorm (LN centralised in TriContextEncoder)."""
    enc = RoleEncoder(N=4)
    has_ln = any(isinstance(m, torch.nn.LayerNorm) for m in enc.row_mlp.modules())
    assert not has_ln


def test_gradient_flow():
    """Backward reaches id_emb + row_mlp parameters."""
    enc = RoleEncoder(N=4)
    role = enc(torch.tensor([[0, 1, 2, 3]]), torch.rand(1, 4, 3))
    (role ** 2).sum().backward()
    for name, p in enc.named_parameters():
        assert p.grad is not None, f"No grad for {name}"
        assert p.grad.abs().sum() > 0, f"Zero grad for {name}"


def test_episode_invariance():
    """Same (id, row) input ⇒ identical output (cacheable within episode)."""
    enc = RoleEncoder(N=4)
    enc.eval()
    agent_ids = torch.tensor([[0, 1, 2, 3]])
    rows = torch.rand(1, 4, 3)
    assert torch.allclose(enc(agent_ids, rows), enc(agent_ids, rows))


def test_n_scaling():
    """id_emb table and row_mlp input scale with N."""
    enc2 = RoleEncoder(N=2)
    enc8 = RoleEncoder(N=8)
    assert enc2.id_emb.num_embeddings == 2
    assert enc8.id_emb.num_embeddings == 8
    assert enc2.row_mlp[0].in_features == 1
    assert enc8.row_mlp[0].in_features == 7
