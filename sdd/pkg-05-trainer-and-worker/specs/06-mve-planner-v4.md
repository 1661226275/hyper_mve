# Spec 06: MVE Planner v4 — CRN 保留 + 4 处 set_context 迁移 + cap/belief 显式参数

> 父文档：[`../proposal.md`](../proposal.md) §2.1.4 · [`../design.md`](../design.md) §3 D7 · §6.8
> **D7 显式参数**：sample_mve_plan 接受 `cap` / `belief` / `c_t` 显式参数（trainer/worker 端准备）；**Pkg-04 spec 08 §3.1 已锁定 4 处行号**：L71/213/219/258。

---

## 1. Purpose

按 v4.7 mve_planner.py L80-290 完整保留 CRN 4 phase + coord descent 算法骨架，最小化迁移 4 处 set_context 调用点到 v4 两步分离 API：

| 改动 | 范围 |
|------|------|
| **保留**（不改）| CRN 4 phase 算法（L128-148 Phase 1 / L150-172 Phase 2 / L189-253 Phase 3 / L282-290 Phase 4）|
| **保留**（不改）| coord descent 随机 agent 顺序（L127, L141-143）|
| **迁移**（4 处）| set_context 调用点（L71/213/219/258 → set_context_objective + set_context_subjective）|
| **扩展**（新增）| sample_mve_plan 入口签名加 cap / belief / c_t 显式参数（D7）|
| **新增**（一次）| planner 入口处先调 set_context_objective（与 K-step unroll trainer 模式一致）|

---

## 2. Interface

### 2.1 文件路径

`hyper_mve/planning/mve_planner.py`（v4.7 同名文件 inplace 修改，D10）

### 2.2 sample_mve_plan 签名（D7 + Pkg-04 spec 08 §3.1）

```python
import torch
import numpy as np
from typing import Optional
from hyper_mve.configs import V4Config
from hyper_mve.models import HyperMuZeroModel


class MVEPlanner:
    """MVE Planner v4 - CRN + coord descent (Pkg-05 spec 06).
    
    v4 关键改动 (相对 v4.7 mve_planner.py):
        - CRN 4 phase 算法完整保留 (L80-290)
        - 4 处 set_context 调用点迁移 (Pkg-04 spec 08 §3.1)
        - sample_mve_plan 入口加 cap / belief / c_t 显式参数 (D7)
        - 持有 self.crn_rng_state 跨 episode 持续 (Pkg-05 spec 02 修订 5)
    """
    
    def __init__(self, cfg: V4Config):
        self.cfg = cfg
        self.mve_samples = cfg.train.mve_samples       # spa, 默认 50
        self.mve_depth = cfg.train.mve_depth           # K_mve, 默认 5
        self.mve_temperature = cfg.train.mve_temperature  # 默认 1.0
        self.use_crn = cfg.train.use_crn               # 默认 True (Pkg-08 ablation 1)
        self.use_coord_desc = cfg.train.use_coord_desc # 默认 True (Pkg-08 ablation 2)
        
        # CRN rng state (review 修订 5: 跨 episode 持续)
        self.crn_rng = np.random.default_rng(seed=cfg.train.epsilon_decay_steps)  # 任意 seed
    
    def sample_mve_plan(
        self,
        model: HyperMuZeroModel,
        root_s: torch.Tensor,              # (B, latent_dim) 根状态
        cap: dict[int, torch.Tensor],      # ← D7: agent_id → (B, 4) cap tensor
        belief: dict[int, tuple[torch.Tensor, torch.Tensor]],
                                           # ← D7: agent_id → (c_hat (B,), z_hat (B, N-1, 2))
        c_t: torch.Tensor,                 # ← D7: (B,) c_t scalar (planner 入口一次性 set_context_objective 用)
    ) -> torch.Tensor:                     # (B, N, A) MVE 输出策略
        """v4 MVE planner.
        
        D7 接口扩展: 接受 cap / belief / c_t 显式参数 (取代 v4.7 内部从 env 拉取).
        
        Args:
            model: HyperMuZeroModel 实例
            root_s: 根状态 (B, latent_dim)
            cap: {agent_id: cap_tensor (B, 4)} N agents 能力向量
            belief: {agent_id: (c_hat (B,), z_hat (B, N-1, 2))} N agents 信念
            c_t: (B,) 共享 c_t scalar
        
        Returns:
            policies: (B, N, A) per-agent MVE 策略分布
        """
        N = self.cfg.env.N
        A = self.cfg.env.A
        spa = self.mve_samples
        K = self.mve_depth
        B = root_s.shape[0]
        device = root_s.device
        
        # ============================================================
        # planner 入口: set_context_objective 一次性调用 (D4 缓存策略)
        # ============================================================
        # 复用 θ_state across N agents + K-step unroll
        model.set_context_objective(c_t)
        
        # ============================================================
        # Phase 1: 随机 agent 顺序 + 预采样其他 agent step 0 动作 (v4.7 L128-148 保留)
        # ============================================================
        if self.use_coord_desc:
            agent_order = self.crn_rng.permutation(N).tolist()
        else:
            agent_order = list(range(N))
        
        # 预采样其他 agent 在 root_s 下的 step 0 动作 (CRN)
        if self.use_crn:
            # CRN: 同一 seed 下其他 agent 动作固定 (减少方差)
            crn_seed = self.crn_rng.integers(0, 2**31)
        else:
            crn_seed = None
        
        # other_actions_step_0 shape (spa, N), 从 prior policy 采样
        other_actions = self._sample_other_actions_step_0(
            model, root_s, cap, belief, agent_order, spa, crn_seed,
        )
        
        # ============================================================
        # Phase 2-3: per-agent coord descent (v4.7 L189-253 保留)
        # ============================================================
        policies = torch.zeros(B, N, A, device=device)
        
        for idx_in_order, agent_id in enumerate(agent_order):
            # 当前 agent 视角下 K-step unroll
            
            # ★ v4 set_context_subjective 迁移点 1 (替代 v4.7 L71)
            model.set_context_subjective(
                agent_id=agent_id,
                cap_i=cap[agent_id],
                belief=belief[agent_id],
            )
            
            # Phase 2: 展开候选动作 (B, spa * A)
            curr_s = root_s.repeat_interleave(spa, dim=0)  # (B*spa, latent_dim)
            curr_s = curr_s.repeat_interleave(A, dim=0)    # (B*spa*A, latent_dim)
            
            # 累积 Q 值
            q_values = torch.zeros(B * spa * A, device=device)
            
            for k_step in range(K):
                if k_step == 0:
                    # 当前 agent 候选动作 (B*spa, A) → (B*spa*A,) 展开
                    own_action = torch.arange(A, device=device).repeat(B * spa)
                    # 其他 agent 用 Phase 1 的 CRN 采样
                    joint_action = self._compose_joint_action(
                        own_action, other_actions, agent_id, B, spa, A,
                    )
                else:
                    # k > 0: 所有 agent 从当前 policy 重新采样 (独立)
                    joint_action = self._sample_joint_action(
                        model, curr_s, cap, belief, agent_id, agent_order[:idx_in_order],
                        policies, B, spa, A,
                    )
                
                # ★ v4 set_context_subjective 迁移点 2 (替代 v4.7 L213, 目标态转移)
                # 转移: 客观, 不需要 set_context_subjective (用 set_context_objective 的 θ_state)
                joint_action_onehot = _joint_action_onehot(joint_action, A)  # (B*spa*A, N*A)
                curr_s = model.transition(curr_s, joint_action_onehot)
                
                # ★ v4 set_context_subjective 迁移点 3 (替代 v4.7 L219, 奖励预测)
                # 奖励: 主观 (per agent), 复用 set_context_subjective(agent_id)
                # 注: 上方 set_context_subjective(agent_id) 已设, 此处直接调 predict_reward
                r_k = model.predict_reward(curr_s, joint_action_onehot).squeeze(-1)  # (B*spa*A,)
                
                gamma_k = self.cfg.train.gamma ** k_step
                q_values = q_values + gamma_k * r_k
            
            # ★ v4 set_context_subjective 迁移点 4 (替代 v4.7 L258, 终值估计)
            # 终值: 主观 (per agent), 复用 set_context_subjective(agent_id)
            v_K = model.predict(curr_s)[1].squeeze(-1)  # (B*spa*A,)
            q_values = q_values + (self.cfg.train.gamma ** K) * v_K
            
            # ============================================================
            # Phase 4: 聚合 Q 值 → softmax policy (v4.7 L282-290 保留)
            # ============================================================
            q_reshape = q_values.reshape(B, spa, A)  # (B, spa, A)
            q_mean = q_reshape.mean(dim=1)           # (B, A) 在 spa 维度平均
            # 归一化 + softmax
            q_norm = (q_mean - q_mean.mean(dim=-1, keepdim=True)) / (q_mean.std(dim=-1, keepdim=True) + 1e-8)
            policies[:, agent_id] = torch.softmax(q_norm / self.mve_temperature, dim=-1)
        
        return policies
    
    # ====================================================================
    # 内部辅助 (CRN 4 phase 实施细节, v4.7 保留)
    # ====================================================================
    
    def _sample_other_actions_step_0(
        self, model, root_s, cap, belief, agent_order, spa, crn_seed,
    ) -> torch.Tensor:                  # (spa, N) int
        """Phase 1: 预采样其他 agent step 0 动作 (CRN)."""
        # v4.7 L128-148 算法保留
        # 关键: 用 crn_seed 控制采样, 同一 seed 下结果固定 (CRN 单测验证)
        pass
    
    def _compose_joint_action(
        self, own_action, other_actions, agent_id, B, spa, A,
    ) -> torch.Tensor:                  # (B*spa*A, N)
        """Phase 2: 组合 own_action + other_actions → joint."""
        # v4.7 L150-172 算法保留
        pass
    
    def _sample_joint_action(
        self, model, curr_s, cap, belief, current_agent, optimized_agents, policies, B, spa, A,
    ) -> torch.Tensor:                  # (B*spa*A, N)
        """Phase 3 k>0 步: 所有 agent 从当前 policy 重新采样.
        
        coord descent: 已优化的 agent 从 policies 采样, 未优化的从 prior.
        """
        # v4.7 L189-253 算法保留
        pass


def _joint_action_onehot(actions: torch.Tensor, A: int) -> torch.Tensor:
    """actions (B*spa*A, N) int → (B*spa*A, N*A) onehot flat."""
    import torch.nn.functional as F
    BS, N = actions.shape
    onehot = F.one_hot(actions, num_classes=A).float()  # (BS, N, A)
    return onehot.reshape(BS, N * A)


# NOTE (P1-3 / D10): v4 **不保留** v4.7 顶层 `sample_mve_plan(...)` 函数。
# 理由：顶层包装每次调用都 `MVEPlanner(cfg)` 新建实例 → 丢失 CRN rng 跨 episode
# 状态（见 §1 修订 5：worker 持有单一 planner 实例），并诱导 worker / 未来 baselines
# 走 "每次新建 planner" 反模式（spec 02 P1-2）。
# v4 唯一入口 = `MVEPlanner.sample_mve_plan(...)` 实例方法。
# v4.7 旧顶层签名 `sample_mve_plan(model, root_s, cfg, ...)` 已随 D10 inplace 重写
# 归档至 `_legacy_v4_7/`，所有调用方一律改用持有的 planner 实例（trainer / worker）。
```

---

## 3. Implementation Notes

### 3.1 v4.7 → v4 4 处 set_context 调用点迁移（Pkg-04 spec 08 §3.1 锁定）

| 行号 | v4.7 代码 | v4 迁移 | 调用模式 |
|------|----------|---------|---------|
| L71 | `model.set_context(rule_exp, id_i)` | `model.set_context_subjective(int(id_i[0]), cap[id_i[0]], belief[id_i[0]])` | per-agent 策略采样时 |
| L213 | `model.set_context(rule_exp, id_0)` | 同上模式 | 目标态转移（实际客观，无需 subjective）|
| L219 | `model.set_context(rule_exp, id_j)` | 同上模式 | 奖励预测 |
| L258 | `model.set_context(rule_exp, id_j)` | 同上模式 | 终值估计 |

**新增 planner 入口逻辑**（约在 sample_mve_plan 函数入口）：

```python
# v4 新增: planner 入口处一次性 set_context_objective
def sample_mve_plan(self, model, root_s, cap, belief, c_t):
    # planner 入口处先调一次 set_context_objective (C5-T2 一致)
    model.set_context_objective(c_t)
    
    # 后续循环内只调 set_context_subjective (已迁移 L71/213/219/258)
    ...
```

### 3.2 CRN 算法保留（C5-P1）

CRN (Common Random Numbers) 4 phase 实施完整保留（v4.7 mve_planner.py L128-290）：

- **Phase 1**（L128-148）：场景粒度预采样其他 agent step 0 动作
- **Phase 2**（L150-172）：展开到 (B*spa*A,) 布局
- **Phase 3**（L189-253）：K 步展开，step 0 用 CRN，step > 0 独立采样
- **Phase 4**（L282-290）：聚合 (B, spa, A) → (B, A)，Q 值归一化，softmax

**CRN 单测**：相同 crn_seed 下 step 0 其他 agent 动作必须一致：

```python
def test_crn_step0_deterministic_same_seed(planner, model, ...):
    # 用相同 crn_seed 调两次 sample_mve_plan
    # 验证 step 0 时 _sample_other_actions_step_0 输出一致
```

### 3.3 coord descent 算法保留（v4.7 L127, L141-143）

- 随机 agent 顺序：`agent_order = rng.permutation(N)`
- 已优化的 agent 从 policies 采样（spec 06 §2.2 _sample_joint_action 内）
- 未优化的从 prior policy 采样

**关闭 coord descent**（cfg.train.use_coord_desc=False）时退化为固定顺序 `[0, 1, ..., N-1]`，用于 Pkg-08 ablation 2。

### 3.4 planner 入口 set_context_objective 一次性调用

按 D4 缓存策略 + Pkg-04 spec 07 §2.2 性能优势：
- planner 入口 set_context_objective 一次 → 复用 θ_state across N agents × K-step
- 节省 N×K 次 hyper_trans forward（约 N×K × 2 ms = 40 ms / planning call）

实施层：sample_mve_plan 函数开头第一行就是 `model.set_context_objective(c_t)`。

### 3.5 worker 持有 planner 实例（review 修订 5）

spec 02 §2.2 已规定 worker.__init__ 接受 planner 参数，self.planner = planner or MVEPlanner(cfg)：
- CRN rng_state 跨 episode 持续（R5-3 NaN 稳定性 + 复现性）
- 单测 `test_worker_planner_persistent` 验证 id(self.planner) 跨 episode 不变

### 3.6 性能预算

| 子项 | 预算 | 备注 |
|------|------|------|
| 单 sample_mve_plan (B=1, mve_samples=50, K=5) | < 50 ms | v4.7 持平（CRN 算法不变）|
| set_context_objective | < 5 ms | Pkg-04 spec 07 档位 1 |
| N agents × set_context_subjective | < 20 ms (N=4) | Pkg-04 spec 07 档位 1 × 4 |
| K-step transition + reward + value | < 25 ms | Pkg-04 spec 07 档位 1 × 多次 |

**collect_episode (use_planner=True) 时**：T × sample_mve_plan = 200 × 50 = 10 s（spec 02 §3.6）。

### 3.7 cap / belief dict 取代 env 直接耦合（D7）

v4.7 mve_planner 内部从 env.cfg 或 env._state 直接拿 rule_exp / id_i：
- 问题：planner 与 env 耦合，难做 baseline 复用（Pkg-06）

v4 显式参数 dict：
- planner 不依赖 env
- baselines / eval / trainer 各自准备 cap / belief 传入
- 单测 `test_planner_cap_belief_shape` 入口 assert 防止 shape drift（R5-10）

---

## 4. Edge Cases

| 场景 | 行为 |
|------|------|
| cap dict 缺 agent_id 键 | KeyError 抛 |
| belief dict 缺 agent_id 键 | KeyError 抛 |
| cap[k] shape ≠ (B, 4) | Pkg-04 model.set_context_subjective 入口 assert 抛（C11 修订 2/澄清 2）|
| c_t shape ≠ (B,) | model.set_context_objective 入口校验 |
| use_crn=False 时 crn_rng 仍 advance | 允许（CRN 关闭仅影响 Phase 1 采样行为）|
| use_coord_desc=False 时 agent_order 固定 | OK，agent_order = [0, 1, ..., N-1] |
| mve_samples=0 | Q 值为空，softmax 退化为 uniform（极端 cfg 不推荐）|
| K_mve=0 | 仅终值估计，Q = γ⁰ · V(root_s) per agent |

---

## 5. Acceptance Criteria

### 5.1 单元测试（`tests/planning/test_mve_planner.py`）

```python
import pytest
import torch
import numpy as np
from hyper_mve.configs import V4Config
from hyper_mve.models import HyperMuZeroModel
from hyper_mve.planning.mve_planner import MVEPlanner  # P1-3: 顶层 sample_mve_plan 已删除


@pytest.fixture
def cfg_medium():
    return V4Config.from_preset("medium")


@pytest.fixture
def model(cfg_medium):
    return HyperMuZeroModel(cfg_medium)


@pytest.fixture
def planner(cfg_medium):
    return MVEPlanner(cfg_medium)


def _make_inputs(cfg, B=1, device="cpu"):
    N = cfg.env.N
    return {
        "root_s": torch.randn(B, cfg.model.latent_dim, device=device),
        "cap": {k: torch.rand(B, 4, device=device) for k in range(N)},
        "belief": {k: (torch.rand(B, device=device),
                        torch.softmax(torch.randn(B, N-1, 2, device=device), dim=-1))
                    for k in range(N)},
        "c_t": torch.full((B,), 0.5, device=device),
    }


# ====== C5-P1: CRN seed 相同 step 0 输出一致 ======

def test_crn_step0_deterministic_same_seed(planner, model, cfg_medium):
    """相同 crn_rng seed 下 sample_mve_plan 输出一致 (CRN 算法验证)."""
    inputs = _make_inputs(cfg_medium)
    
    # 第一次调用 (固定 seed)
    planner.crn_rng = np.random.default_rng(seed=42)
    out_1 = planner.sample_mve_plan(model, **inputs)
    
    # 第二次调用 (相同 seed)
    planner.crn_rng = np.random.default_rng(seed=42)
    out_2 = planner.sample_mve_plan(model, **inputs)
    
    assert torch.allclose(out_1, out_2, atol=1e-6), (
        "相同 crn_rng seed 下 sample_mve_plan 输出应一致 (CRN)"
    )


def test_crn_different_seed_different_output(planner, model, cfg_medium):
    """不同 crn_rng seed 下输出应不同."""
    inputs = _make_inputs(cfg_medium)
    
    planner.crn_rng = np.random.default_rng(seed=42)
    out_1 = planner.sample_mve_plan(model, **inputs)
    
    planner.crn_rng = np.random.default_rng(seed=999)
    out_2 = planner.sample_mve_plan(model, **inputs)
    
    assert not torch.allclose(out_1, out_2, atol=1e-3)


# ====== C5-P2: 4 处 set_context 调用点全部迁移 ======

def test_planner_4_set_context_migrated(planner, model, cfg_medium, mocker):
    """sample_mve_plan 内不应调旧 set_context(rule, id) 单参签名,
    应调 set_context_objective + set_context_subjective.
    """
    spy_obj = mocker.spy(model, "set_context_objective")
    spy_subj = mocker.spy(model, "set_context_subjective")
    
    inputs = _make_inputs(cfg_medium)
    planner.sample_mve_plan(model, **inputs)
    
    # 至少调用一次 set_context_objective (planner 入口)
    assert spy_obj.call_count >= 1
    # 至少调用 N 次 set_context_subjective (per agent)
    assert spy_subj.call_count >= cfg_medium.env.N


def test_planner_no_legacy_set_context(planner, model, cfg_medium, mocker):
    """sample_mve_plan 内不应调 model.set_context (旧单参签名).
    
    注: v4 HyperMuZeroModel 应已不存在 set_context 旧签名 (Pkg-04 已迁移).
    本测试用 mock 验证 planner 端不调.
    """
    # mock 一个不存在的方法以确认 planner 没调
    if hasattr(model, "set_context"):
        spy = mocker.spy(model, "set_context")
        inputs = _make_inputs(cfg_medium)
        planner.sample_mve_plan(model, **inputs)
        assert spy.call_count == 0, "planner 不应调 v4.7 旧 set_context 签名"


# ====== 输出 shape ======

def test_sample_mve_plan_output_shape(planner, model, cfg_medium):
    """输出 shape (B, N, A) 合规."""
    B = 2
    inputs = _make_inputs(cfg_medium, B=B)
    
    policies = planner.sample_mve_plan(model, **inputs)
    
    assert policies.shape == (B, cfg_medium.env.N, cfg_medium.env.A)
    # softmax 后行和为 1
    assert torch.allclose(policies.sum(dim=-1), torch.ones(B, cfg_medium.env.N), atol=1e-5)


# ====== 性能 ======

@pytest.mark.gpu
def test_sample_mve_plan_under_50ms(planner, model, cfg_medium):
    """单 sample_mve_plan (B=1, mve_samples=50, K=5) < 50 ms."""
    import time
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    
    model = model.cuda()
    planner = MVEPlanner(cfg_medium)
    inputs = _make_inputs(cfg_medium, device="cuda")
    
    # Warmup
    for _ in range(5):
        planner.sample_mve_plan(model, **inputs)
    torch.cuda.synchronize()
    
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        planner.sample_mve_plan(model, **inputs)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    
    mean_ms = sum(times) / len(times)
    assert mean_ms < 50.0, f"sample_mve_plan {mean_ms:.2f}ms > 50ms"


# ====== R5-10: cap/belief shape 入口 assert ======

def test_planner_cap_belief_shape(planner, model, cfg_medium):
    """cap[k] shape 必须 (B, 4); belief[k] 是 tuple of 2 tensors.
    
    入口 assert 验证 (R5-10 缓解).
    """
    inputs = _make_inputs(cfg_medium)
    
    # 故意传 cap shape (B, 5) (含 type leak 风险)
    inputs["cap"][0] = torch.rand(1, 5)
    
    with pytest.raises((AssertionError, RuntimeError, ValueError)):
        planner.sample_mve_plan(model, **inputs)


# ====== use_crn / use_coord_desc 开关 ======

def test_use_crn_disabled(planner, model, cfg_medium):
    """cfg.train.use_crn=False 时 planner 不用 CRN (Pkg-08 ablation 1)."""
    from dataclasses import replace
    cfg_no_crn = replace(cfg_medium, train=replace(cfg_medium.train, use_crn=False))
    planner_no_crn = MVEPlanner(cfg_no_crn)
    
    inputs = _make_inputs(cfg_no_crn)
    out = planner_no_crn.sample_mve_plan(model, **inputs)
    assert out.shape == (1, cfg_no_crn.env.N, cfg_no_crn.env.A)


def test_use_coord_desc_disabled(planner, model, cfg_medium):
    """cfg.train.use_coord_desc=False 时 agent_order = [0,1,...,N-1] (Pkg-08 ablation 2)."""
    from dataclasses import replace
    cfg_no_cd = replace(cfg_medium, train=replace(cfg_medium.train, use_coord_desc=False))
    planner_no_cd = MVEPlanner(cfg_no_cd)
    
    inputs = _make_inputs(cfg_no_cd)
    out = planner_no_cd.sample_mve_plan(model, **inputs)
    assert out.shape == (1, cfg_no_cd.env.N, cfg_no_cd.env.A)
```

### 5.2 集成测试

- worker.collect_episode (use_planner=True) 跑 1 集端到端 < 60 s（spec 02 §3.6）
- sample_mve_plan 输出 policy 行和为 1（softmax 验证）

### 5.3 性能要求

- 单 sample_mve_plan (B=1, mve_samples=50, K=5) < 50 ms（V100, Medium）
- 与 v4.7 持平（CRN 算法保留）

---

## 6. v4.7 → v4 对比

| 维度 | v4.7 | v4 | 差异 |
|------|------|----|------|
| CRN 4 phase 算法 | 完整 | 完整保留 | 无变化 |
| coord descent 随机顺序 | 已有 | 保留 + use_coord_desc 开关 | configurable |
| set_context 调用 | 4 处单参签名（L71/213/219/258）| 4 处两步分离 + planner 入口 set_context_objective | Pkg-04 spec 08 §3.1 迁移 |
| sample_mve_plan 入口 | 内部从 env 拉 rule | 显式参数 cap / belief / c_t（D7）| 接口扩展 |
| use_crn / use_coord_desc 开关 | hardcoded True | cfg.train.use_crn / use_coord_desc | configurable（Pkg-08 ablation 1+2 用）|
| 单次 planning 性能 | ~50 ms | ~50 ms | 持平 |

---

## 7. Cross-references

- v4.7 DESIGN_DOC_FINAL.md §4.1 + §5.8（CRN 重要性）
- `02-worker-collection.md`（worker 持有 planner，spec 02 修订 5）
- `08-integration-contracts.md` §2（4 处迁移 grep 验证 — 澄清 2 精确正则）
- Pkg-04 spec 02 §2.3 调用模板（planner 入口 set_context_objective + per agent set_context_subjective）
- Pkg-04 spec 07 §2.1 档位 2 单 agent K-step unroll < 25 ms（与 planner 性能预算配合）
- Pkg-04 spec 08 §3.1（v4.7 → v4 4 处行号迁移指引）
- Pkg-01 spec 05 TrainConfig（mve_samples / mve_depth / mve_temperature / use_crn / use_coord_desc）
- Pkg-08 ablation 1/2（CRN ✗ / coord_desc ✗ 对照实验）
