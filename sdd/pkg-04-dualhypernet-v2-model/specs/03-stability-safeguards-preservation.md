# Spec 03: 4 道稳定性防线保留方案 + v4.7→v4 行号映射

> 父文档：[`../proposal.md`](../proposal.md) §1.3 · [`../design.md`](../design.md) §3 D7
> **v4 关键**：v4.7 经验证有效的 4 道防线**逐行保留**至 v4，加入 v4 新增的防线 5（belief 梯度门控，spec 04 详述）。

---

## 1. Purpose

v4.7 经多版本迭代（v3.2 → v4.7）建立的 4 道稳定性防线是 Hyper-MuZero 训练能稳定收敛的关键。v4 重写时**必须逐行保留**，不允许"重新设计"——任何偏离都会引入未知风险。本 spec 提供：

1. 4 道防线的**精确技术内容**（公式 + 实现代码）
2. **v4.7 → v4 行号映射表**（保证迁移时不漏不错）
3. **数值不变性单测**（与 v4.7 数值回归对比，确保 v4 实现与 v4.7 等价）
4. v4 新增防线 5（belief 梯度门控）的引用（spec 04 详述）

---

## 2. 4 道防线技术清单

### 2.1 防线 1：权重生成的初始化（small_init + L2 norm + output_scale）

**Ch4.6.1 原文**："hypernetwork 输出层（生成 θ 的最后一层）的初始化必须谨慎，否则生成的功能网络权重在训练初期可能产生剧烈梯度，导致训练发散。"

**v4.7 实现** (`hyper_mve/models/hyper_network.py:61-82`):

```python
# 行 61: small_init 在 output_layer
small_init(self.output_layer, std=0.01)

# 行 63-65: L2 norm + learnable output_scale (v4.0 引入)
self.norm_output = norm_output
self.output_scale = nn.Parameter(torch.tensor(float(output_scale_init)))

# 行 67-82: forward 内归一化
def forward(self, context):
    h = self.trunk(context)
    raw = self.output_layer(h)
    if self.norm_output:
        norms = torch.linalg.norm(raw, dim=-1, keepdim=True)
        raw = raw / (norms + 1e-8)
    return raw * self.output_scale
```

**v4 迁移**：完全保留（spec 01 §2.2 `HyperNetMLP` 类，无任何修改）。

**关键不变量**：
- `small_init` 默认 std=0.01（不能改为 std=0.1）
- `norm_output=True` 默认开（关闭即退回 v3.2 失败状态）
- `output_scale_init` 三初值：trans=0.01, rew=**0.1**, pred=0.01（C8）

---

### 2.2 防线 2：LayerNorm + AdaLN with (1+γ) factor (v4.6 critical fix)

**Ch4.6.2 原文**："StateTransNet / RewardHead / PredictionNet 在每个隐藏层后用 GroupNorm（因为权重是 hypernetwork 生成的，BatchNorm 不适用）"。

**注**：v4.7 实际实现是 **AdaLN**（Adaptive LayerNorm with HyperNet-generated gamma/beta），不是 GroupNorm。Ch4.6.2 的描述与实现不一致——v4.7 实现更先进。spec 03 以 v4.7 实现为准。

**v4.7 实现** (`hyper_mve/models/functional_nets.py:43-77`):

```python
# 行 43-77: adaln_forward 函数
def adaln_forward(x, weight, bias, gamma, beta):
    # 行 68: Linear
    h = functional_linear(x, weight, bias)
    # 行 70-72: Instance-wise LayerNorm
    mean = h.mean(dim=-1, keepdim=True)
    var = h.var(dim=-1, keepdim=True, unbiased=False)
    h = (h - mean) / torch.sqrt(var + 1e-5)
    # 行 73-75: ★ Adaptive affine with residual modulation (v4.6 关键 fix)
    h = h * (1 + gamma) + beta   # ← (1 + ) factor 不能丢
    # 行 77: ReLU
    return F.relu(h)
```

**v4.6 关键 fix 原文（注释行 50-56）**：
- HyperNet output_scale 小（训练初期）时 gamma ≈ 0
- (1 + gamma) 让层 act as identity pass-through: h ≈ h_norm
- 没有 (1+) 时 gamma ≈ 0 crushes h_norm to zero，破坏所有输入信号（包括 action 信息），让功能网络输出常数

**v4 迁移**：`adaln_forward` 函数完全保留（无任何修改）。

**关键不变量**：
- `(1 + gamma) * x + beta`，**不能改为** `gamma * x + beta`（v3.2 失败状态）
- LayerNorm in instance-wise（per-sample over feature dim），不能改为 batch-wise
- ε=1e-5 在 sqrt(var + ε) 内

---

### 2.3 防线 3：StateTransNet Δs 残差（v4.7 经验）

**Ch4.6 未直接列出，但 v4.7 functional_nets.py 实现**

**v4.7 实现** (`hyper_mve/models/functional_nets.py:245-271`，FunctionalStateTransNet.forward):

```python
# 行 254: split params for layer specs
params = split_params_adaln(flat_params, self.layer_specs)
x = torch.cat([state, action_onehot], dim=-1)

# 行 257-263: FC1 + AdaLN + ReLU; FC2 + AdaLN + ReLU
w1, b1, g1, bt1 = params[0]
x = adaln_forward(x, w1, b1, g1, bt1)
w2, b2, g2, bt2 = params[1]
x = adaln_forward(x, w2, b2, g2, bt2)

# 行 265-267: FC3 plain output (输出 Δs)
w3, b3 = params[2]
delta_s = functional_linear(x, w3, b3)

# 行 269-271: ★ Fixed LayerNorm + Residual connection
delta_s = self.ln(delta_s)
return state + delta_s  # s' = s + Δs   ← 残差
```

**为什么残差**：
- StateTransNet 预测 Δs（增量）而非 s'（绝对值）
- 训练初期 hyper_trans 输出小 → Δs ≈ 0 → s' ≈ s（identity）
- 训练稳定（v3.2 直接预测 s' 时初期 collapse）

**v4 迁移**：`FunctionalStateTransNet` 类完全保留，仅修改 latent_dim / joint_action_dim 从 cfg 取（不再硬编码）。

**关键不变量**：
- `return state + delta_s` 必须包含 `state +`（残差）
- `delta_s = self.ln(delta_s)` 必须在 `state +` 之前（先 LN 后 add）
- `self.ln = nn.LayerNorm(latent_dim)`（fixed，不由 HyperNet 生成）

---

### 2.4 防线 4：一致性损失（BYOL-style，Ch5.8.2 详述）

**Ch4.6.4 原文**："详见 Chapter 5.8.2 节。这是 v3 验证有效的稳定化技术，v4 保留。"

**v4.7 实现**：在 `training/muzero_trainer.py` 中，BYOL projection + cosine similarity consistency loss（行号待 spec 05 Pkg-05 spec 详述）。

**v4 迁移责任**：本包**不实现**（属于 Pkg-05 trainer 范围）。本包仅提供 Projector 接口给 trainer 调用：

```python
# Pkg-04 不实现; Pkg-05 trainer 实现 BYOL loss
# 本包仅提供 cfg.model.proj_dim=64 配置字段供 Projector 用
```

**关键不变量**：proj_dim=64 默认（Pkg-01 ModelConfig.proj_dim 已锁定）。

---

### 2.5 防线 5 (v4 新增)：belief 梯度门控（spec 04 详述）

**Ch4.6.5 原文**："阶段 1 → 阶段 2（开始退火）与 阶段 2 → 阶段 3（完全使用 ẑ）的两次切换可能引入主任务 loss 的突跃。"

**v4 实现**：grad_gating.py + model.forward 内 step-conditional `.detach()`（详见 spec 04）。

**与防线 1-4 的关系**：防线 5 是 v4 新增的第 5 道防线，与防线 1-4 正交（不冲突）。本 spec 仅引用，详述见 spec 04。

---

## 3. v4.7 → v4 行号映射表

| 防线 | v4.7 文件 | v4.7 行号 | v4 迁移 spec | 修改 |
|------|-----------|-----------|--------------|------|
| 1: small_init + L2 norm + output_scale | `models/hyper_network.py` | L57-65 (init) + L67-82 (forward) | spec 01 §2.2 HyperNetMLP | 0（结构不变，仅 input/output dim 由 spec 01 DualHyperNetwork 控制） |
| 2: AdaLN (1+γ) | `models/functional_nets.py` | L43-77 (adaln_forward) | 本 spec §2.2 | 0（adaln_forward 函数完全保留） |
| 3: Δs 残差 | `models/functional_nets.py` | L245-271 (FunctionalStateTransNet.forward) | 本 spec §2.3 | latent_dim/action_dim 从 cfg 取（不影响残差逻辑） |
| 4: BYOL consistency | `training/muzero_trainer.py` | （Pkg-05 spec 03 行号） | Pkg-05 spec 03 | 本包仅提供 proj_dim cfg |
| 5: belief 梯度门控 | （v4 新增） | – | spec 04 | grad_gating.py 新增 |

---

## 4. functional_nets.py 修改清单（极小）

`hyper_mve/models/functional_nets.py` 修改极小（保留 4 道防线核心代码不动）：

| 修改 | 位置 | 改动 |
|------|------|------|
| obs_dim 配置化 | 各 Functional* 类 __init__ | 从 cfg.env.observation_space.shape 取 obs_dim（不再硬编码） |
| joint_action_dim 配置化 | 同上 | 从 cfg.env.N * cfg.env.A 派生 |
| latent_dim 配置化 | 同上 | cfg.model.latent_dim |

具体 diff 见 spec 03 §5 单测中的 v4.7 数值回归对比代码。

---

## 5. Acceptance Criteria

### 5.1 单元测试（`tests/models/test_stability_safeguards.py`）

```python
import pytest
import torch
import torch.nn.functional as F
from hyper_mve.configs import V4Config
from hyper_mve.models import HyperMuZeroModel
from hyper_mve.models.functional_nets import adaln_forward, FunctionalStateTransNet


# ====== 防线 1: small_init + L2 norm + output_scale ======

def test_small_init_output_layer_std():
    """small_init: output_layer weight std ≈ 0.01."""
    from hyper_mve.models.hyper_network import HyperNetMLP
    mlp = HyperNetMLP(input_dim=16, output_dim=1000)
    std = mlp.output_layer.weight.std().item()
    assert std < 0.05  # 容差 (small_init std=0.01)


def test_l2_norm_output_magnitude():
    """L2 norm: forward 输出 L2 norm == output_scale."""
    from hyper_mve.models.hyper_network import HyperNetMLP
    mlp = HyperNetMLP(input_dim=16, output_dim=100, norm_output=True, output_scale_init=0.1)
    x = torch.randn(4, 16)
    out = mlp(x)
    norms = torch.linalg.norm(out, dim=-1)
    assert torch.allclose(norms, torch.full_like(norms, 0.1), atol=1e-5)


def test_output_scale_inits_trans_rew_pred():
    """C8: trans=0.01, rew=0.1, pred=0.01."""
    cfg = V4Config.from_preset("medium")
    model = HyperMuZeroModel(cfg)
    assert torch.isclose(model.hyper_net.hyper_trans.output_scale, torch.tensor(0.01))
    assert torch.isclose(model.hyper_net.hyper_rew.output_scale, torch.tensor(0.1))
    assert torch.isclose(model.hyper_net.hyper_pred.output_scale, torch.tensor(0.01))


# ====== 防线 2: AdaLN (1+γ) factor ======

def test_adaln_one_plus_gamma_factor():
    """C7: gamma=0 时 h = (1+0)*x + beta = x + beta (identity + shift).
    
    若实现错误为 h = gamma*x + beta, gamma=0 时输出全 beta（破坏输入信号）.
    """
    B, in_dim, out_dim = 2, 16, 32
    x = torch.randn(B, in_dim)
    weight = torch.randn(B, out_dim, in_dim) * 0.01
    bias = torch.zeros(B, out_dim)
    gamma = torch.zeros(B, out_dim)         # ★ gamma = 0
    beta = torch.randn(B, out_dim) * 0.5
    
    out = adaln_forward(x, weight, bias, gamma, beta)
    # out = ReLU(LayerNorm(linear(x)) * (1+0) + beta)
    #     = ReLU(h_norm + beta)
    # 若错误实现 = ReLU(h_norm * 0 + beta) = ReLU(beta) → 完全破坏 h_norm 信号
    
    # 验证: out 应受 weight (即 x) 影响, 不应仅由 beta 决定
    # 通过对比 weight=0 vs weight=随机 输出差异显著来确认
    weight_zero = torch.zeros(B, out_dim, in_dim)
    out_no_weight = adaln_forward(x, weight_zero, bias, gamma, beta)
    
    # weight=随机 时 out 与 weight=0 时显著不同 → 证明 (1+γ) factor 保留了输入信号
    assert not torch.allclose(out, out_no_weight, atol=1e-3)


def test_adaln_gamma_nonzero():
    """gamma 非零时 (1+γ) factor 正确施加."""
    B, in_dim, out_dim = 1, 8, 16
    x = torch.randn(B, in_dim)
    weight = torch.eye(out_dim, in_dim).unsqueeze(0).expand(B, -1, -1).contiguous()
    weight = weight[:, :, :in_dim]  # (B, out_dim, in_dim)
    bias = torch.zeros(B, out_dim)
    gamma = torch.ones(B, out_dim) * 1.0    # (1 + 1) = 2x scaling
    beta = torch.zeros(B, out_dim)
    
    out = adaln_forward(x, weight, bias, gamma, beta)
    # 内部: h_norm = (linear(x) - mean) / sqrt(var), 然后 h = h_norm * 2 + 0
    # 不易直接断言数值; 仅断言无 NaN 且非常数
    assert not torch.isnan(out).any()
    assert out.std().item() > 0.01


# ====== 防线 3: Δs 残差 ======

def test_state_trans_delta_s_residual():
    """C6: StateTransNet forward 含 state + delta_s (残差)."""
    cfg = V4Config.from_preset("medium")
    stn = FunctionalStateTransNet(cfg)
    
    B = 2
    state = torch.randn(B, cfg.model.latent_dim)
    action = torch.zeros(B, cfg.env.N * cfg.env.A)
    action[:, 0] = 1.0
    flat_params = torch.zeros(B, stn.total_params)   # Δs = 0
    
    s_next = stn(state, action, flat_params)
    # flat_params 全 0 时 Δs ≈ 0 (LayerNorm 后), s_next ≈ state
    diff = (s_next - state).abs().mean().item()
    assert diff < 1.0  # state + delta_s, delta_s ≈ 0 时 s_next ≈ state


def test_state_trans_residual_not_identity_with_nonzero_params():
    """flat_params 非零时 s_next != state (Δs 有效)."""
    cfg = V4Config.from_preset("medium")
    stn = FunctionalStateTransNet(cfg)
    
    B = 2
    state = torch.randn(B, cfg.model.latent_dim)
    action = torch.zeros(B, cfg.env.N * cfg.env.A)
    action[:, 0] = 1.0
    flat_params = torch.randn(B, stn.total_params) * 0.5
    
    s_next = stn(state, action, flat_params)
    assert not torch.allclose(s_next, state, atol=1e-3)  # Δs 非零


def test_state_trans_uses_fixed_layernorm():
    """spec 03 §2.3: self.ln 是 fixed nn.LayerNorm (不由 HyperNet 生成)."""
    cfg = V4Config.from_preset("medium")
    stn = FunctionalStateTransNet(cfg)
    assert isinstance(stn.ln, torch.nn.LayerNorm)
    # 验证 self.ln 在 ModuleList 内 (其参数随主 optimizer 训练)
    has_ln_param = any(p is stn.ln.weight or p is stn.ln.bias for p in stn.parameters())
    assert has_ln_param


# ====== 数值回归对比 v4.7 (可选, 需 v4.7 archive 启用) ======

@pytest.mark.skip(reason="Requires v4.7 archive snapshot; enable in regression CI")
def test_adaln_matches_v47_numerical():
    """与 v4.7 archive 的 adaln_forward 输出数值回归对比 (误差 ≤ 1e-6)."""
    # 实施: import hyper_mve._legacy_v4_7.models.functional_nets as v47
    # v47_out = v47.adaln_forward(...)
    # v4_out = adaln_forward(...)  # 当前
    # assert torch.allclose(v47_out, v4_out, atol=1e-6)
    pass


# ====== 防线 5 (v4 新增): belief 梯度门控引用 ======

def test_grad_gating_referenced(cfg_medium):
    """防线 5 详细在 spec 04 test_grad_gating.py 测试; 本 spec 仅验证模型含 grad_gating 实例."""
    cfg = V4Config.from_preset("medium")
    model = HyperMuZeroModel(cfg)
    from hyper_mve.models.grad_gating import BeliefGradGating
    assert isinstance(model.grad_gating, BeliefGradGating)
```

### 5.2 v4.7 数值回归（可选 CI）

如未来需要严格回归对比，启用 `test_adaln_matches_v47_numerical` 等单测，要求与 `_legacy_v4_7/` 中 adaln_forward / FunctionalStateTransNet 输出数值误差 ≤ 1e-6。

### 5.3 性能要求

- adaln_forward 单次（B=256, out_dim=128）< 0.5 ms (V100)
- FunctionalStateTransNet.forward 单次 < 1 ms

---

## 6. v4 vs v4.7 防线对照表

| 防线 | v4.7 | v4 | 改动 |
|------|------|----|------|
| 1: small_init + L2 norm + output_scale | ✅ 已有 | ✅ 完整保留 | 0（HyperNetMLP 不动） |
| 2: AdaLN (1+γ) | ✅ v4.6 fix | ✅ 完整保留 | 0（adaln_forward 不动） |
| 3: Δs 残差 | ✅ 已有 | ✅ 完整保留 | obs_dim/latent_dim 配置化 |
| 4: BYOL consistency | ✅ Pkg-05 trainer | ✅ Pkg-05 trainer | 本包提供 proj_dim cfg |
| 5: belief 梯度门控 | – | ✅ **v4 新增**（spec 04） | grad_gating.py 新增 |

---

## 7. Cross-references

- Ch4.6 4 道稳定性防线
- Ch4.6.5 belief 梯度门控（v4 新增防线 5）
- `01-dualhypernet-v2-api.md`（防线 1 在 HyperNetMLP）
- `02-hyper-muzero-model-v2.md`（cfg.model.use_adaln / state_trans_residual / adaln_residual_one_plus）
- `04-belief-gradient-gating.md`（防线 5 详述）
- v4.7 `models/hyper_network.py:57-82` (防线 1)
- v4.7 `models/functional_nets.py:43-77` (防线 2 adaln_forward)
- v4.7 `models/functional_nets.py:245-271` (防线 3 Δs 残差)
- Pkg-01 spec 05 ModelConfig（use_adaln, state_trans_residual, adaln_residual_one_plus, proj_dim 等字段）
- Pkg-05 spec 03 BYOL consistency loss（防线 4 详述）

---

## B. [v4-opt 2026-06] 优化阶段修订(防线 1 扩展;与正文冲突处以本节为准)

> 实证动机:FULL 全量生成坍缩(cos_pred_cross 0.61→0.998)与整向量 L2 的 FiLM 稀释。实现提交 `079fcdf`/`dc5bbcd`;代码锚点 `hyper_mve/models/hyper_network.py:28-57,111-122`。

### B1. 防线 1 扩展:分组 RMS 归一(部分生成范围)

`gen_scope ∈ {film_head, base_gen, lora_fc2}` 时,HyperNetMLP 归一化从整向量 L2 改为**分组 RMS**(`output_groups=[film_total, weight_total]`,逐段 RMS 归一 × output_scale):
- 失效模式:整向量 L2 下单个 FiLM γ 幅值 ≈ scale/√dim(0.1/√512≈4e-3)⇒ (1+γ)≈1,AdaLN 调制名存实亡;
- 修复后:每生成元幅值 ≈ scale(维度无关);FULL 范围保持整向量 L2,逐字节兼容旧行为;
- 验收:`tests/models/test_hyper_network_grouped_norm.py`(|γ| ≥ 1e-2 守门)。

### B2. 防线 1 扩展:输出层 LoRA 初始化纪律

`hyper_output_rank=r` 时输出投影分解为 A(prev→r, 正交, 无 bias)→ B(r→pc, small_init std=0.01)。**B 禁止零初始化**:B=0 ⇒ raw=0 ⇒ 分组 RMS 除以 1e-8 下限 ⇒ step-0 约 1e4 梯度尖峰。验收:`tests/models/test_hyper_network_lora.py`。

### B3. 新增稳定性约束:ΔW 尺度律(lora_fc2)

分组 RMS 下每生成元 ≈ scale,故 fc2 低秩增量 ΔW = B_f A_f 的元素 RMS ≈ **scale²·√r**:
- scale=0.1, r=8 ⇒ ΔW≈0.028(kaiming fc2 基权 0.088 的 ~32%,有效);scale=0.01 ⇒ ΔW≈3e-4(死);
- 守门断言(`ModelConfig.__post_init__`):lora_fc2 要求三路 scale ≥ 0.05(推荐 0.1);base_gen 禁用 lora_fc2(fc2 已全生成,ΔW 冗余);r=0 与 film_head 逐字节等价(回归门);
- **未决风险**(复审 Q3,登记):output_scale 可学习且 ΔW 随其二次增长,训练后期 scale 上行可能使 ΔW 越过基权幅值——是否引入 scale clamp 待 sweep 的 scale 轨迹数据决定。

### B4. 防线 1 适用范围注记

原 §2.1 的"L2 norm + output_scale"描述仅适用于 FULL;§2.2 AdaLN(1+γ)与 §2.3 Δs 残差在全部 gen_scope 下保留不变(部分生成的 forward 路径均经 `adaln_modulate`/`adaln_forward` 维持 (1+γ) 形式)。

## 修订记录 (Changelog)

| 日期 | 修订 | 依据 |
|---|---|---|
| 2026-06-10 | B1-B4:分组 RMS、LoRA 初始化纪律、ΔW 尺度律(新增防线约束)、防线适用范围注记 | 提交 079fcdf/dc5bbcd;复审 `docs/Review_v4_TheoryAudit_2026-06.md` §3.3-3.4、Q3 |
