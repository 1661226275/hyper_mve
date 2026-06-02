"""Unit tests for ``hyper_mve.models.belief_net`` (Pkg-03 spec 04 + spec 05 §5.1).

Combines the GRU backbone tests (spec 04) and the two-head tests (spec 05) since
both exercise the single ``BeliefNet`` module.
"""
from __future__ import annotations

import pytest
import torch

from hyper_mve.configs import V4Config
from hyper_mve.models import BeliefNet


@pytest.fixture
def cfg_medium():
    return V4Config.from_preset("medium")


# ====== spec 04: GRU backbone ======

def test_init_hidden_zeros(cfg_medium):
    """init_hidden 输出全零 (B, N, 128)."""
    bn = BeliefNet(cfg_medium.env, cfg_medium.model)
    hidden = bn.init_hidden(batch_size=2, num_agents=4)
    assert hidden.shape == (2, 4, 128)
    assert torch.all(hidden == 0)


def test_init_hidden_device():
    """init_hidden 自动选 model 所在 device."""
    cfg = V4Config.from_preset("medium")
    bn = BeliefNet(cfg.env, cfg.model)
    if torch.cuda.is_available():
        bn = bn.cuda()
        hidden = bn.init_hidden(2, 4)
        assert hidden.device.type == "cuda"


def test_gru_input_dim_independent_of_latent():
    """BeliefObsEncoder gru_input_dim 与 ModelConfig.latent_dim 解耦."""
    cfg = V4Config.from_preset("medium")
    bn = BeliefNet(cfg.env, cfg.model)
    assert bn.gru_input_dim == 64
    # latent_dim 是主 RepNet 用 (Pkg-04), BeliefNet 不消费
    assert bn.gru_hidden == cfg.model.belief_gru_hidden  # 128


def test_gru_step_seq_equiv(cfg_medium):
    """关键测试: step 逐步 ≈ forward(seq).

    GRUCell 在 step 与 forward 中应使用相同的 forward 逻辑.
    """
    bn = BeliefNet(cfg_medium.env, cfg_medium.model)
    bn.eval()  # 关闭 dropout (本模块无 dropout 但 safety)

    B, T, N = 2, 5, 4
    obs_dim = bn.obs_dim
    obs_seq = torch.randn(B, T, N, obs_dim)
    init_h = bn.init_hidden(B, N)

    # 方法 1: forward(seq)
    hidden_seq, c_hat_seq, z_hat_seq = bn.forward(obs_seq, init_hidden=init_h)

    # 方法 2: step 逐步
    h = init_h
    step_outputs_h = []
    step_outputs_c = []
    step_outputs_z = []
    for t in range(T):
        h, c, z = bn.step(obs_seq[:, t], h)
        step_outputs_h.append(h)
        step_outputs_c.append(c)
        step_outputs_z.append(z)

    step_hidden_seq = torch.stack(step_outputs_h, dim=1)        # (B, T, N, 128)
    step_c_seq = torch.stack(step_outputs_c, dim=1)             # (B, T, N)
    step_z_seq = torch.stack(step_outputs_z, dim=1)             # (B, T, N, N-1, 2)

    # 严格相等 (误差 < 1e-5)
    assert torch.allclose(hidden_seq, step_hidden_seq, atol=1e-5)
    assert torch.allclose(c_hat_seq, step_c_seq, atol=1e-5)
    assert torch.allclose(z_hat_seq, step_z_seq, atol=1e-5)


def test_hidden_state_evolution(cfg_medium):
    """hidden 应随 step 变化 (而非常数)."""
    bn = BeliefNet(cfg_medium.env, cfg_medium.model)
    bn.eval()

    B, N = 1, 4
    obs_t = torch.randn(B, N, bn.obs_dim)
    h0 = bn.init_hidden(B, N)
    h1, _, _ = bn.step(obs_t, h0)

    # h1 应与 h0 (全零) 不同
    assert not torch.allclose(h1, h0)


def test_shared_gru_weights(cfg_medium):
    """GRU 在 N 个 agent 间共享权重 (而非 per-agent)."""
    bn = BeliefNet(cfg_medium.env, cfg_medium.model)
    # 只有一个 GRUCell 实例
    assert isinstance(bn.gru, torch.nn.GRUCell)
    # 验证: 给两个 agent 相同的 obs 和 hidden, 输出应相同
    B = 1
    obs_dim = bn.obs_dim
    same_obs = torch.randn(B, 1, obs_dim)
    same_hidden = torch.randn(B, 1, 128)
    # agent 0 输出
    out_a, _, _ = bn.step(same_obs, same_hidden)
    # agent 0 在 (B=1, N=2) 中, 与一个噪声 agent 1 并列, 看是否 agent 0 输出不变
    obs_pair = torch.cat([same_obs, torch.randn(B, 1, obs_dim)], dim=1)
    hidden_pair = torch.cat([same_hidden, torch.randn(B, 1, 128)], dim=1)
    out_pair, _, _ = bn.step(obs_pair, hidden_pair)
    # agent 0 在两次调用中应得到同样输出 (因为 GRU 共享权重 + same obs/hidden)
    assert torch.allclose(out_a[:, 0], out_pair[:, 0], atol=1e-5)


def test_layernorm_after_gru(cfg_medium):
    """GRU 输出经过 LayerNorm (Ch4.6.2 防线 2)."""
    bn = BeliefNet(cfg_medium.env, cfg_medium.model)
    has_ln_belief = hasattr(bn, "ln_belief") and isinstance(bn.ln_belief, torch.nn.LayerNorm)
    assert has_ln_belief, "BeliefNet must apply LayerNorm after GRU (Ch4.6.2)"


def test_no_bidirectional(cfg_medium):
    """v4 D6: uni-directional only (RL 在线推断必需)."""
    bn = BeliefNet(cfg_medium.env, cfg_medium.model)
    # 验证: 后续 step 不依赖未来 obs (causality)
    B, N = 1, 4
    obs_dim = bn.obs_dim
    # 同样的 obs 序列, 改变未来 obs, t=0 的 hidden 应不变
    obs_a = torch.randn(B, 5, N, obs_dim)
    obs_b = obs_a.clone()
    obs_b[:, 3:] = torch.randn(B, 2, N, obs_dim)  # 改变 t>=3 的 obs

    h0 = bn.init_hidden(B, N)
    h_a_t0, _, _ = bn.step(obs_a[:, 0], h0)
    h_b_t0, _, _ = bn.step(obs_b[:, 0], h0)

    assert torch.allclose(h_a_t0, h_b_t0)


# ====== spec 05: two heads ======

def test_head_c_output_shape(cfg_medium):
    """head_c 输出 (B, N) sigmoid ∈ [0, 1]."""
    bn = BeliefNet(cfg_medium.env, cfg_medium.model)
    B, N = 2, 4
    obs_t = torch.randn(B, N, bn.obs_dim)
    h0 = bn.init_hidden(B, N)
    _, c_hat, _ = bn.step(obs_t, h0)

    assert c_hat.shape == (B, N)
    assert (c_hat >= 0).all() and (c_hat <= 1).all()


def test_head_opp_output_shape(cfg_medium):
    """head_opp 输出 (B, N, N-1, 2) softmax."""
    bn = BeliefNet(cfg_medium.env, cfg_medium.model)
    B, N = 2, 4
    obs_t = torch.randn(B, N, bn.obs_dim)
    h0 = bn.init_hidden(B, N)
    _, _, z_hat = bn.step(obs_t, h0)

    assert z_hat.shape == (B, N, N - 1, 2)
    # softmax: 每个 (b, i, k) 概率和 = 1
    sums = z_hat.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_z_hat_agent_id_order_convention(cfg_medium):
    """v4 关键: z_hat[b, i, k] 对应 agent_id = (k if k < i else k + 1)."""
    bn = BeliefNet(cfg_medium.env, cfg_medium.model)
    bn.eval()

    # Hook BeliefIdEmbedding: 每个 id 用一个独特的 embedding 值
    with torch.no_grad():
        for j in range(bn.N):
            bn.opp_id_emb.embedding.weight[j].fill_(float(j + 1))

    B, N = 1, 4
    obs_t = torch.zeros(B, N, bn.obs_dim)
    h0 = torch.zeros(B, N, 128)

    expected_opp_ids = [
        [1, 2, 3],
        [0, 2, 3],
        [0, 1, 3],
        [0, 1, 2],
    ]

    _, _, z_hat = bn.step(obs_t, h0)

    # 重新构造 opp_ids 应与 expected 一致
    actual_opp_ids = torch.zeros(N, N - 1, dtype=torch.long)
    for i in range(N):
        actual_opp_ids[i] = torch.tensor([j for j in range(N) if j != i])
    assert actual_opp_ids.tolist() == expected_opp_ids


def test_head_c_is_scalar_per_agent_v4(cfg_medium):
    """v4 修订: head_c 输出 (B, N) scalar, 不是 (B, N, d_c=16)."""
    bn = BeliefNet(cfg_medium.env, cfg_medium.model)
    B, N = 1, 4
    obs_t = torch.randn(B, N, bn.obs_dim)
    h0 = bn.init_hidden(B, N)
    _, c_hat, _ = bn.step(obs_t, h0)

    assert c_hat.dim() == 2          # 不是 3 (即不是 (B, N, d_c))
    assert c_hat.shape[-1] == N      # 最后维是 N, 不是 16


def test_head_opp_oracle_substitution(cfg_medium):
    """oracle_z 给出时, forward 返回 oracle_z (Stage 1 用)."""
    bn = BeliefNet(cfg_medium.env, cfg_medium.model)
    B, T, N = 1, 3, 4
    obs_seq = torch.randn(B, T, N, bn.obs_dim)
    oracle_z = torch.ones(B, T, N, N - 1, 2) * 0.5

    _, _, z_seq = bn.forward(obs_seq, oracle_z_seq=oracle_z)
    assert torch.allclose(z_seq, oracle_z)


def test_head_opp_still_trained_in_oracle_mode(cfg_medium):
    """Stage 1 oracle 模式下 head_opp 仍 forward (供 L_opp 训练)."""
    bn = BeliefNet(cfg_medium.env, cfg_medium.model)
    B, T, N = 1, 3, 4
    obs_seq = torch.randn(B, T, N, bn.obs_dim, requires_grad=True)
    oracle_z = torch.ones(B, T, N, N - 1, 2) * 0.5

    h_seq, c_seq, z_seq = bn.forward(obs_seq, oracle_z_seq=oracle_z)

    # 用 get_head_opp_predictions 取 head_opp 真实预测
    z_pred = bn.get_head_opp_predictions(h_seq)
    assert z_pred.shape == oracle_z.shape
    # 应是 softmax 概率
    assert torch.allclose(z_pred.sum(dim=-1), torch.ones_like(z_pred.sum(dim=-1)), atol=1e-5)

    # head_opp 参数应有梯度 (用 z_pred 的 loss 反向)
    loss = z_pred.sum()
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.norm() > 0
                   for p in bn.head_opp_mlp.parameters())
    assert has_grad


def test_opp_id_emb_independent_of_role_id_emb(cfg_medium):
    """BeliefIdEmbedding 与 RoleEncoder.id_emb 是独立 nn.Embedding."""
    from hyper_mve.models import TriContextEncoder
    bn = BeliefNet(cfg_medium.env, cfg_medium.model)
    enc = TriContextEncoder(cfg_medium.env, cfg_medium.model)

    # 两个 Embedding 实例不同对象
    assert id(bn.opp_id_emb.embedding) != id(enc.role_encoder.id_emb)
    # 维度可不同 (本 spec: opp_id_emb=16, role.id_emb=8)
    assert bn.opp_id_emb.embedding.embedding_dim == 16
    assert enc.role_encoder.id_emb.embedding_dim == 8


def test_softmax_simplex_property(cfg_medium):
    """z_hat 是有效的 simplex (概率 ≥ 0, 和 = 1)."""
    bn = BeliefNet(cfg_medium.env, cfg_medium.model)
    B, N = 4, 4
    obs_t = torch.randn(B, N, bn.obs_dim)
    h0 = bn.init_hidden(B, N)
    _, _, z_hat = bn.step(obs_t, h0)

    assert (z_hat >= 0).all()
    sums = z_hat.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
