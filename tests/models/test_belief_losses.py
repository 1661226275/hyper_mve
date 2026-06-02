"""Unit tests for ``hyper_mve.models.belief_losses`` (Pkg-03 spec 06 §5.1 + spec 08 §4.2)."""
from __future__ import annotations

import inspect
import math

import pytest
import torch

from hyper_mve.models.belief_losses import (
    l_c,
    l_opp,
    l_div,
    belief_loss,
    build_oracle_z_seq,
)


def test_l_c_zero_loss_at_perfect_pred():
    """c_hat == c_true 时 L_c = 0."""
    B, T, N = 2, 3, 4
    c_true = torch.tensor([[0.3, 0.4, 0.5], [0.6, 0.7, 0.8]])
    c_hat = c_true.unsqueeze(-1).expand(B, T, N).clone()
    loss = l_c(c_hat, c_true)
    assert loss.item() < 1e-6


def test_l_c_positive_when_err():
    """c_hat != c_true 时 L_c > 0."""
    B, T, N = 2, 3, 4
    c_true = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    c_hat = torch.zeros(B, T, N)  # all 0, vs true 0.5 -> MSE = 0.25
    loss = l_c(c_hat, c_true)
    assert torch.allclose(loss, torch.tensor(0.25), atol=1e-5)


def test_l_c_gradient_direction():
    """L_c 梯度方向: c_hat < c_true -> 梯度 < 0 (推 c_hat 增大)."""
    B, T, N = 1, 1, 1
    c_true = torch.tensor([[0.7]])
    c_hat = torch.tensor([[[0.3]]], requires_grad=True)
    loss = l_c(c_hat, c_true)
    loss.backward()
    # ∂L/∂c_hat = 2 * (c_hat - c_true) / (B*T*N) = 2 * (0.3 - 0.7) = -0.8
    assert c_hat.grad.item() < 0


def test_l_opp_oracle_ce(monkeypatch):
    """L_opp 是 Oracle CE: 标签是 int 类型 types_true, 不是动作."""
    B, T, N = 1, 1, 4
    # types_true: agent 0,1 是 α (0); agent 2,3 是 β (1)
    types_true = torch.tensor([[[0, 0, 1, 1]]], dtype=torch.long)

    # 构造完美预测的 z_hat: 每个 agent i 对所有对手 j 的预测正确
    z_hat = torch.zeros(B, T, N, N - 1, 2)
    for i in range(N):
        opp_list = [j for j in range(N) if j != i]
        for k, j in enumerate(opp_list):
            true_type = types_true[0, 0, j].item()  # 0 or 1
            z_hat[0, 0, i, k, true_type] = 1.0  # one-hot 完美预测

    loss = l_opp(z_hat, types_true)
    # 完美预测时 loss ≈ 0
    assert loss.item() < 1e-4


def test_l_opp_uniform_pred_baseline():
    """uniform softmax (0.5, 0.5) 预测时 L_opp = ln(2) ≈ 0.693."""
    B, T, N = 1, 1, 4
    types_true = torch.tensor([[[0, 0, 1, 1]]], dtype=torch.long)
    z_hat = torch.ones(B, T, N, N - 1, 2) * 0.5  # uniform

    loss = l_opp(z_hat, types_true)
    # -log(0.5) = ln(2) ≈ 0.6931
    assert torch.allclose(loss, torch.tensor(math.log(2)), atol=1e-4)


def test_l_opp_agent_id_order_consistency():
    """L_opp 与 z_hat 顺序约定一致 (agent_id 升序跳过 self)."""
    B, T, N = 1, 1, 4
    types_true = torch.tensor([[[0, 1, 0, 1]]], dtype=torch.long)  # α β α β
    z_hat = torch.ones(B, T, N, N - 1, 2) * 0.5

    # agent 2 对对手 0 (agent 0, α) 预测全错 (预测 β)
    z_hat[0, 0, 2, 0, 0] = 0.01  # P(α) very low
    z_hat[0, 0, 2, 0, 1] = 0.99  # P(β) high (但 true is α)

    loss = l_opp(z_hat, types_true)
    # 该错预测贡献 -log(0.01) ≈ 4.6 显著大于 -log(0.5) = 0.69
    # 整体 loss 应大于 0.7
    assert loss.item() > 0.7


def test_l_opp_index_mapping_correctness():
    """🔑 P3 关键单测: 验证 j → agent_id 映射精确正确, 防 silent bug."""
    B, T, N = 1, 1, 4
    # types: agent 0 = α (0), agent 1 = β (1), agent 2 = α (0), agent 3 = β (1)
    types_true = torch.tensor([[[0, 1, 0, 1]]], dtype=torch.long)

    # 构造完美预测:
    # 对每个 (i, k), 标签是 types_true[opp_agent_id(i, k)]
    # opp_agent_id(i, k) = k if k < i else k + 1
    z_hat = torch.zeros(B, T, N, N - 1, 2)
    for i in range(N):
        for k in range(N - 1):
            j = k if k < i else k + 1  # 正确映射
            true_type = types_true[0, 0, j].item()
            # one-hot 完美预测 (注: 用 0.999 避免 log(0))
            z_hat[0, 0, i, k, true_type] = 0.999
            z_hat[0, 0, i, k, 1 - true_type] = 0.001

    loss = l_opp(z_hat, types_true)
    # 完美预测 → loss ≈ -log(0.999) ≈ 1e-3
    assert loss.item() < 0.01, (
        f"L_opp = {loss.item():.4f} > 0.01; 映射可能错位. "
        f"若映射正确, 完美对角预测应 ≈ -log(0.999) ≈ 1e-3."
    )


def test_l_opp_index_mapping_specific_pairs():
    """🔑 P3: 验证 (i, k) → j 的具体映射规则."""
    B, T, N = 1, 1, 4
    # types: agent 0=α, agent 1=β, agent 2=α, agent 3=β
    types_true = torch.tensor([[[0, 1, 0, 1]]], dtype=torch.long)

    # 测试样本: (i, k, expected_opp_id)
    test_cases = [
        (0, 0, 1),  # agent 0 的对手 0 → agent 1
        (0, 1, 2),  # agent 0 的对手 1 → agent 2
        (0, 2, 3),  # agent 0 的对手 2 → agent 3
        (1, 0, 0),  # agent 1 的对手 0 → agent 0
        (1, 1, 2),  # agent 1 的对手 1 → agent 2
        (1, 2, 3),  # agent 1 的对手 2 → agent 3
        (2, 0, 0),  # agent 2 的对手 0 → agent 0
        (2, 1, 1),  # agent 2 的对手 1 → agent 1
        (2, 2, 3),  # agent 2 的对手 2 → agent 3
        (3, 0, 0),  # agent 3 的对手 0 → agent 0
        (3, 1, 1),  # agent 3 的对手 1 → agent 1
        (3, 2, 2),  # agent 3 的对手 2 → agent 2
    ]

    for i, k, expected_opp_id in test_cases:
        # 对其他位置均匀预测 (loss = log 2), 仅 (i, k) 错预测
        z_hat = torch.ones(B, T, N, N - 1, 2) * 0.5
        true_type_at_j = types_true[0, 0, expected_opp_id].item()
        wrong_type = 1 - true_type_at_j
        # 在 (i, k) 位置预测 wrong_type 满 prob
        z_hat[0, 0, i, k, true_type_at_j] = 0.001
        z_hat[0, 0, i, k, wrong_type] = 0.999

        loss = l_opp(z_hat, types_true)
        assert loss.item() > 0.8, (
            f"(i={i}, k={k}, expected_opp_id={expected_opp_id}): "
            f"L_opp={loss.item():.4f} should be > 0.8 (一处错预测应让 loss 显著上升). "
            f"若 loss < 0.8, 可能映射 (i,k)→j 错位."
        )


def test_l_c_broadcast_correctness():
    """P8: c_true_seq shape (B, T) 不含 N 维, 内部 broadcast 到 (B, T, N)."""
    B, T, N = 2, 3, 4
    # 固定 c_hat = 0.5 across all agents, c_true 不同
    c_hat_seq = torch.full((B, T, N), 0.5)
    c_true_seq = torch.tensor([
        [0.5, 0.5, 0.5],   # batch 0: 完美匹配 → contrib 0
        [0.0, 1.0, 0.5],   # batch 1: 错配 (0.5)² + (0.5)² + 0 per agent
    ])

    loss = l_c(c_hat_seq, c_true_seq)
    # 手算标量:
    # batch 0: 0 (3 step × N=4 agent, 全 0)
    # batch 1: t=0 → (0.5-0)² × 4 = 1.0; t=1 → (0.5-1)² × 4 = 1.0; t=2 → 0
    # 总 sq_err = 0 + 1.0 + 1.0 + 0 = 2.0
    # denom = B*T*N = 2*3*4 = 24
    # L_c = 2.0 / 24 ≈ 0.0833
    expected = 2.0 / 24
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-5)


def test_l_c_wrong_input_shape_rejected():
    """P8: c_true_seq shape (B, T, N) 应被拒绝 (与契约不符)."""
    B, T, N = 1, 2, 4
    c_hat_seq = torch.zeros(B, T, N)
    bad_c_true = torch.zeros(B, T, N)  # 错: 含 N 维
    with pytest.raises(AssertionError, match="c_true_seq shape"):
        l_c(c_hat_seq, bad_c_true)


def test_l_div_hinge_zero_when_high_variance():
    """variance >= target_var 时 loss = 0."""
    B, T, N, h = 1, 1, 4, 8
    # 制造高方差 hidden
    hidden = torch.randn(B, T, N, h) * 5  # std ~ 5, var ~ 25 >> 0.01
    loss = l_div(hidden, target_std=0.1)
    assert loss.item() < 1e-6


def test_l_div_hinge_positive_when_low_variance():
    """variance < target_var 时 loss > 0."""
    B, T, N, h = 1, 1, 4, 8
    # 制造低方差 hidden (所有 agent 几乎相同)
    base = torch.randn(B, T, 1, h)
    hidden = base.expand(B, T, N, h).clone() + torch.randn(B, T, N, h) * 0.001
    # var << 0.01
    loss = l_div(hidden, target_std=0.1)
    assert loss.item() > 0


def test_l_div_collapse_protection():
    """完全 collapse (var=0) 时 loss = target_var (上限)."""
    B, T, N, h = 1, 1, 4, 8
    hidden = torch.ones(B, T, N, h) * 0.5  # 所有 agent 全相同
    loss = l_div(hidden, target_std=0.1)
    # var = 0 → loss = target_var = 0.01
    assert torch.allclose(loss, torch.tensor(0.01), atol=1e-6)


def test_belief_loss_combination():
    """belief_loss 正确组合三 loss + 加权."""
    B, T, N = 2, 3, 4
    c_hat = torch.rand(B, T, N)
    z_hat = torch.softmax(torch.randn(B, T, N, N - 1, 2), dim=-1)
    hidden = torch.randn(B, T, N, 128)
    c_true = torch.rand(B, T)
    types_true = torch.randint(0, 2, (B, T, N))

    total, breakdown = belief_loss(
        c_hat, z_hat, hidden, c_true, types_true,
        weights=(1.0, 0.5, 0.01),
    )

    assert total.requires_grad
    assert "l_c" in breakdown
    assert "l_opp" in breakdown
    assert "l_div" in breakdown
    assert "total" in breakdown

    # breakdown 是 detach
    for k, v in breakdown.items():
        assert not v.requires_grad

    # 验证加权
    expected = 1.0 * breakdown["l_c"] + 0.5 * breakdown["l_opp"] + 0.01 * breakdown["l_div"]
    assert torch.allclose(total.detach(), expected, atol=1e-5)


def test_belief_loss_mask():
    """mask=True 的样本被纳入, mask=False 被忽略."""
    B, T, N = 2, 4, 4
    c_hat = torch.rand(B, T, N)
    z_hat = torch.softmax(torch.randn(B, T, N, N - 1, 2), dim=-1)
    hidden = torch.randn(B, T, N, 128)
    c_true = torch.rand(B, T)
    types_true = torch.randint(0, 2, (B, T, N))

    # mask: 第一个 batch 只前 2 step 有效, 第二个全有效
    mask = torch.tensor([[True, True, False, False], [True] * 4])

    total_mask, _ = belief_loss(c_hat, z_hat, hidden, c_true, types_true, mask=mask)
    total_full, _ = belief_loss(c_hat, z_hat, hidden, c_true, types_true)

    # mask 后 loss 可能不同 (排除 padding step)
    assert total_mask.requires_grad


def test_l_c_uses_oracle_field_name():
    """document: L_c 标签字段名是 c_true_seq (与 Pkg-02 env.info['c_true'] 对应)."""
    sig = inspect.signature(l_c)
    assert "c_true_seq" in sig.parameters


def test_l_opp_uses_types_true_field_name():
    """document: L_opp 标签字段名是 types_true (与 Pkg-02 env.info['types'] 对应)."""
    sig = inspect.signature(l_opp)
    assert "types_true" in sig.parameters


# ====== spec 08 §4.2: build_oracle_z_seq (与 l_opp 共用 opp_indices) ======

def test_build_oracle_z_seq_one_hot():
    """oracle_z 输出 one-hot."""
    types = torch.tensor([[[0, 1, 0, 1]]], dtype=torch.long)
    z = build_oracle_z_seq(types)
    assert z.shape == (1, 1, 4, 3, 2)
    # 每个 (b, t, i, k) 概率分布是 one-hot
    sums = z.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums))


def test_build_oracle_z_seq_matches_head_opp_order():
    """oracle_z 顺序与 l_opp gather 顺序严格一致 (端到端验证)."""
    types = torch.tensor([[[0, 1, 0, 1]]], dtype=torch.long)
    oracle_z = build_oracle_z_seq(types)
    # 若用 oracle_z 作 z_hat 喂 l_opp, 损失应 ≈ 0 (完美匹配)
    loss = l_opp(oracle_z * 0.999 + 0.001 / 2, types)  # 加 smoothing 避免 log(0)
    assert loss.item() < 0.01
