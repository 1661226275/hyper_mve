# Spec 04: BeliefNet GRU 主干

> 父文档：[`../proposal.md`](../proposal.md) §1.2 · [`../design.md`](../design.md) §3 D6 / §6.3

---

## 1. Purpose

按 Ch4.2.3 实现 BeliefNet 的 GRU 主干部分（不含两 head，head 见 spec 05）。负责把 agent $i$ 的观测历史 $h_i^{0:t}$ 编码为隐状态 $b_i^t \in \mathbb{R}^{128}$：

$$b_i^t = \text{GRU}_{\text{belief}}(\text{Encoder}(o_i^t),\;b_i^{t-1})$$

**关键性质**：
- **共享权重，独立隐状态**：所有 N 个 agent 使用同一 GRU 参数，但各自维护独立的 $b_i^t$
- **uni-directional**（D6）：causal，RL 在线推断必需
- **两套 API**：`step()` 单步（worker 在线）+ `forward()` 序列（trainer K-step unroll）
- **GRU 后接 LayerNorm**（**Ch4.6.2 防线 2 硬约束**，P6 修订显式标注）：稳定 b_i^t 演化，防止跨 step hidden 分布漂移
- **obs_encoder 架构钉死**（P6 修订）：`Linear(obs_dim, 128) → ReLU → Linear(128, 64) → LayerNorm(64)`，gru_input_dim=64

本模块只实现 GRU 主干 + 观测编码器 + hidden state 管理；两 head（head_c、head_opp）在 spec 05 实现，组合后构成完整 `BeliefNet` 类。

---

## 2. Interface

### 2.1 文件路径

`hyper_mve/models/belief_net.py`（含 spec 05 head；本 spec 仅描述 GRU 主干部分）
`hyper_mve/models/_belief_obs_encoder.py`（独立观测编码器）

### 2.2 _belief_obs_encoder.py 接口

```python
import torch
import torch.nn as nn


class BeliefObsEncoder(nn.Module):
    """BeliefNet 内部的观测编码器 (独立于 Pkg-04 主 RepNet, 避免循环依赖).
    
    将 (B, N, obs_dim) 编码为 (B, N, gru_input_dim).
    
    与主 RepNet 的差异:
        主 RepNet (Pkg-04): obs -> s ∈ ℝ^latent_dim (供 StateTransNet 用)
        本编码器:           obs -> gru_input ∈ ℝ^gru_input_dim (供 GRU 用)
    
    保持独立有 3 个理由:
        1. 避免 Pkg-03 ↔ Pkg-04 循环依赖
        2. BeliefNet 输入维度可独立调整 (gru_input_dim) 而不影响主 latent
        3. BeliefNet 训练梯度独立流动 (符合 Ch4.6.5 梯度门控设计)
    """
    
    def __init__(
        self,
        obs_dim: int,
        gru_input_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.gru_input_dim = gru_input_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, gru_input_dim),
            nn.LayerNorm(gru_input_dim),
        )
    
    def forward(
        self,
        obs: torch.Tensor,                  # (..., obs_dim) float32
    ) -> torch.Tensor:                      # (..., gru_input_dim) float32
        """支持任意前缀维度 (B,), (B, N), (B, T, N) 等."""
        return self.mlp(obs)
```

### 2.3 BeliefNet GRU 主干部分签名

完整 `BeliefNet` 类签名见 spec 05（包含两 head）。本 spec 列出 GRU 主干相关方法：

```python
import torch
import torch.nn as nn
from typing import Optional
from hyper_mve.configs import EnvConfig, ModelConfig
from hyper_mve.schemas import ObservationLayout
from hyper_mve.models._belief_obs_encoder import BeliefObsEncoder


class BeliefNet(nn.Module):
    """v4 BeliefNet (Ch4.2.3 + 4.5).
    
    本 spec 实现 GRU 主干 + hidden state 管理. 两 head 见 spec 05.
    
    架构 (D6: uni-directional GRU):
        obs (B, N, obs_dim)
          -> BeliefObsEncoder -> gru_input (B, N, 64)
          -> GRU(input=64, hidden=128) -> b (B, N, 128)
          -> head_c (b) -> c_hat (B, N) sigmoid             # spec 05
          -> head_opp (b, opp_id) -> z_hat (B, N, N-1, 2)  # spec 05
    
    GRU 共享权重 (所有 N 个 agent 同一 nn.GRUCell), 各自维护独立 hidden state.
    """
    
    def __init__(self, env_cfg: EnvConfig, model_cfg: ModelConfig):
        super().__init__()
        self.env_cfg = env_cfg
        self.model_cfg = model_cfg
        
        self.N = env_cfg.N
        self.obs_dim = ObservationLayout.total_dim(env_cfg.N, env_cfg.K)
        self.gru_input_dim = 64                     # 内部, 不暴露到 ModelConfig
        self.gru_hidden = model_cfg.belief_gru_hidden  # 128 (默认)
        
        # 观测编码器 (独立)
        self.obs_encoder = BeliefObsEncoder(
            obs_dim=self.obs_dim,
            gru_input_dim=self.gru_input_dim,
            hidden_dim=128,
        )
        
        # GRU 主干 (D6: uni-directional, 单 cell 复用)
        # 用 nn.GRUCell 而非 nn.GRU 因为 step API 更自然
        self.gru = nn.GRUCell(
            input_size=self.gru_input_dim,
            hidden_size=self.gru_hidden,
        )
        # forward(seq) 内部组装 RNN loop, 等价于 nn.GRU 但 step/seq API 共享 cell
        
        # GRU 输出 LayerNorm (Ch4.6.2 防线 2)
        self.ln_belief = nn.LayerNorm(self.gru_hidden)
        
        # 两 head 在 spec 05 中初始化
        # self.head_c = ...
        # self.head_opp = ...
    
    def init_hidden(
        self,
        batch_size: int,
        num_agents: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:                      # (B, N, 128) zeros
        """初始化 GRU hidden (每个 (batch_idx, agent_idx) 独立 state)."""
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(batch_size, num_agents, self.gru_hidden, device=device)
    
    def _gru_step(
        self,
        obs_t: torch.Tensor,                # (B, N, obs_dim)
        prev_hidden: torch.Tensor,          # (B, N, 128)
    ) -> torch.Tensor:                      # new_hidden (B, N, 128)
        """单步 GRU update (内部方法, step 与 forward 共享).
        
        将 (B, N, ...) reshape 为 (B*N, ...) 以适配 nn.GRUCell 期望的 (batch, dim).
        """
        B, N, _ = obs_t.shape
        gru_input = self.obs_encoder(obs_t)             # (B, N, 64)
        gru_input_flat = gru_input.reshape(B * N, -1)
        hidden_flat = prev_hidden.reshape(B * N, -1)
        new_hidden_flat = self.gru(gru_input_flat, hidden_flat)
        new_hidden = new_hidden_flat.reshape(B, N, -1)
        new_hidden = self.ln_belief(new_hidden)         # Ch4.6.2 防线 2
        return new_hidden
    
    # step() 与 forward() 完整签名见 spec 05 (含 head 调用)
```

### 2.4 典型用例

```python
# Worker 在线推断 (单步)
from hyper_mve.models import BeliefNet
belief_net = BeliefNet(cfg.env, cfg.model)

B, N, obs_dim = 1, 4, 99
prev_hidden = belief_net.init_hidden(B, N)              # (1, 4, 128)

for t in range(T):
    obs_t = env_obs[t]                                  # (1, 4, 99)
    new_hidden, c_hat, z_hat = belief_net.step(obs_t, prev_hidden)
    # ... 用 c_hat / z_hat 构造 TimeStepRecord
    prev_hidden = new_hidden

# Trainer 序列推断 (K-step unroll)
B, T, N = 256, 6, 4
obs_seq = batch_obs                                     # (256, 6, 4, 99)
hidden_seq, c_hat_seq, z_hat_seq = belief_net.forward(obs_seq)
# hidden_seq shape (256, 6, 4, 128)
# 后接 belief_loss(...) 计算
```

---

## 3. Implementation Notes

### 3.1 为何 nn.GRUCell 而非 nn.GRU

- nn.GRUCell 的 step API 更自然: `cell(input, hidden) -> new_hidden`，与 worker 单步推断模式直接对应
- nn.GRU 是序列 API (`gru(seq, init_hidden) -> (seq_output, final_hidden)`)，单步推断需要 reshape + slice，引入复杂度
- 用 GRUCell 后 forward(seq) 内部手动循环 T 步（每步调用 _gru_step），代码结构与 step() 完全对称
- 单测 `test_gru_step_seq_equiv` 严格验证 step 逐步 == forward(seq) 输出

### 3.2 (B, N) reshape 为 (B*N) 的并行化

- nn.GRUCell 期望输入 shape (batch, input_size)
- 把 (B, N, gru_input_dim) reshape 为 (B*N, gru_input_dim) 一次性 forward → 等价于 N 个独立 GRU 但权重共享
- 共享权重满足 Ch4.2.3 "BeliefNet 在所有 agent 间共享权重"
- 独立 hidden state 满足 "N 个 agent 各自维护独立的 b_i^t"

### 3.3 GRU 输出 LayerNorm 位置（Ch4.6.2 防线 2 硬约束，P6 修订）

- **Ch4.6.2 防线 2 硬约束**：'BeliefNet 的 GRU 后接 LayerNorm，稳定 b_i^t 的演化'
- 本模块在 `_gru_step` 末尾应用 `self.ln_belief = nn.LayerNorm(self.gru_hidden)`
- 这意味着输出的 hidden 是 LN 后的值（而非 raw GRU 输出）
- 注意：下次 step 时 prev_hidden 是 LN 后的，feed 回 GRUCell 也是 LN 后的 → 这是设计（防止 GRU 内部分布漂移）
- 单测 `test_layernorm_after_gru` 验证 `bn.ln_belief` 是 LayerNorm 实例

### 3.4 observation encoder 架构钉死（P6 修订）

**结构**：
```
Linear(obs_dim, 128) → ReLU → Linear(128, gru_input_dim=64) → LayerNorm(64)
```

**为什么 → 64 而非 → 128**（用户审阅 B 决定）：
- GRUCell 参数量 = 3 × input × hidden（reset/update/candidate 三个 gate）
- input=64, hidden=128 → 3 × 64 × 128 = 24576 params
- input=128, hidden=128 → 3 × 128 × 128 = 49152 params（翻倍）
- 64 足够编码 ResourceCommons obs（99 维原始观测），不影响收敛

**为什么独立 BeliefObsEncoder 而非复用 Pkg-04 主 RepNet**：
- v4.6 实现中 `gru_context_encoder` 复用主 RepNet；v4 改为独立 `BeliefObsEncoder`
- 独立的 3 个理由（见类 docstring）：避免 Pkg-03 ↔ Pkg-04 循环依赖、独立维度可调、独立梯度（与 Ch4.6.5 belief 梯度门控配合）
- 参数量代价：~12K（obs_dim=99 → 128 → 64 + LN），相对总 ~3.2M 可忽略
- v4 belief 梯度门控（Ch4.6.5）在 Pkg-04 model 中实现；本模块 forward 不参与门控逻辑

**末端 LayerNorm**：稳定 GRU 输入分布。与 GRU 输出后的 `ln_belief` 配合，保证 GRU 输入输出两端均归一。

### 3.5 hidden state 在 episode 边界

- worker 跨 episode 时必须重新调用 `init_hidden()`（不能跨 episode 累积 hidden）
- trainer K-step unroll 时 init_hidden=None 默认 zeros；若从 buffer 中 sample 中间 step 起始，需要传入 buffer 中存储的 hidden（当前默认仅 zeros，长期可扩展）
- 单测 `test_init_hidden_zeros` 验证 init_hidden 输出全零

### 3.6 性能预算

- step (B=1, N=4) < 1 ms (CPU/GPU)
- forward (B=256, T=6, N=4) < 10 ms (V100 GPU)
- 主要开销在 nn.Linear (obs_encoder MLP) 与 GRUCell 内部矩阵乘

---

## 4. Edge Cases

| 场景 | 行为 |
|------|------|
| init_hidden device 不匹配 | 用 `next(parameters()).device` 自动选 |
| step 时 prev_hidden shape 错 (如 (B, N, 64)) | GRUCell 内部 shape mismatch RuntimeError |
| forward(seq) 时 T=0 | 空 loop, 返回空张量 (实际不会发生) |
| obs 含 NaN | GRU 内部传播 NaN |
| B*N 极大 (B=256, N=8) → 2048 batch | 正常运行 (GRUCell 内部矩阵乘高效) |

---

## 5. Acceptance Criteria

### 5.1 单元测试（`tests/models/test_belief_net.py`，本 spec 部分）

```python
import pytest
import torch
from hyper_mve.configs import V4Config
from hyper_mve.models import BeliefNet


@pytest.fixture
def cfg_medium():
    return V4Config.from_preset("medium")


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
```

### 5.2 性能要求

- `step(obs_t, prev_hidden)` (B=1, N=4) < 1 ms (CPU)
- `forward(obs_seq)` (B=256, T=6, N=4) < 10 ms (V100 GPU)
- BeliefObsEncoder + GRUCell 总参数 < 50K (主要在 GRUCell ~50K = 3 × 64 × 128)

---

## 6. Cross-references

- Ch4.2.3 BeliefNet 主干（GRU + heads）
- Ch4.6.2 防线 2（GRU 输出 LayerNorm）
- Ch4.6.5 belief 梯度门控（Pkg-04 实现，本模块仅提供接口）
- `05-belief-heads.md`（两 head 实现，与本 spec 共同构成完整 BeliefNet）
- `08-integration-contracts.md`（worker step / trainer forward 调用约定）
- Pkg-01 `03-observation-layout.md`（obs_dim 计算）
- Pkg-01 `05-v4-config-structure.md` ModelConfig.belief_gru_hidden
- Pkg-04 spec `02-hyper-muzero-model-v2.md`（set_context 内调用 BeliefNet.step）
- Pkg-05 spec `04-trainer-loop-v2.md`（trainer K-step unroll 内调用 BeliefNet.forward）
