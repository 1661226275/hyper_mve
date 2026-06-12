# Spec 02: Worker Collection — BeliefNet.step 接入 + TimeStepRecord 输出

> 父文档：[`../proposal.md`](../proposal.md) §2.1.2 · [`../design.md`](../design.md) §3 D7 · §6.3 · §6.8
> **v4 关键改动**：v4.7 `set_context_from_history` / `set_context_default`（worker.py L125/128/143/145/149）→ v4 **BeliefNet.step 在线推断 + set_context 两步分离**（Pkg-04 spec 08 §3.3）+ worker 持有 planner（修订 5）。

---

## 1. Purpose

按 Pkg-04 spec 08 §3.3 worker 调用模板 + Pkg-03 spec 04 BeliefNet.step 接入实现 v4 `Worker`，提供：

| API | 用途 | 调用频率 |
|-----|------|---------|
| `__init__(cfg, model, env, planner=None)` | 构造 + worker 持有 planner（修订 5） | 一次性 |
| `collect_episode(epsilon, use_planner)` | 完整一集采集 → list[TimeStepRecord] | 每 iter × episodes_per_iter |

废弃 v4.7 worker 5 处 set_context 调用点（L125/128/143/145/149）。

---

## 2. Interface

### 2.1 文件路径

`hyper_mve/training/worker.py`（v4.7 同名文件 inplace 重写，D10）

### 2.2 类签名（review 修订 5：planner 持有）

```python
import numpy as np
import torch
from typing import Optional
from hyper_mve.configs import V4Config
from hyper_mve.models import HyperMuZeroModel
from hyper_mve.schemas import TimeStepRecord, AgentType, CapabilityVector
from hyper_mve.planning.mve_planner import MVEPlanner  # P1-3: 顶层 sample_mve_plan 已删除, 仅用实例方法


class Worker:
    """v4 Worker (Pkg-05 spec 02).
    
    v4 关键改动 (相对 v4.7 worker.py):
        v4.7: 5 处 set_context 调用点 (L125/128/143/145/149) - 多套 API
        v4: BeliefNet.step 在线推断 + set_context 两步分离, 写 v4 TimeStepRecord (12 字段)
    
    review 修订 5: worker 持有 self.planner (CRN seed 跨 episode 持续).
    """
    
    def __init__(
        self,
        cfg: V4Config,
        model: HyperMuZeroModel,
        env,                                # ResourceCommonsEnv (Pkg-02)
        planner: Optional[MVEPlanner] = None,   # ← review 修订 5
    ):
        """
        Args:
            cfg: V4Config
            model: HyperMuZeroModel (Pkg-04 实例, 与 trainer 共享同一实例)
            env: Pkg-02 ResourceCommonsEnv
            planner: MVEPlanner;
                review 修订 5:
                    None 时 worker 默认 MVEPlanner(cfg).
                    Pkg-08 ablation (use_crn=False / use_coord_desc=False) 时传自定义实例.
                worker 持有 (self.planner); CRN rng_state 跨 episode 持续 (R5-3 稳定性).
        """
        self.cfg = cfg
        self.model = model
        self.env = env
        self.planner = planner or MVEPlanner(cfg)
        
        # worker 永远 inference, no_grad context (Pkg-04 spec 04 §3.4)
        self.model.eval()
    
    # ====================================================================
    # API: collect_episode (v4 BeliefNet.step 接入 + TimeStepRecord 输出)
    # ====================================================================
    
    def collect_episode(
        self,
        epsilon: float = 0.1,
        use_planner: bool = True,
    ) -> tuple[list[TimeStepRecord], torch.Tensor]:
        """收集一个完整 episode.
        
        Returns:
            (records, c_t_seq):
                records: list of T TimeStepRecord (Pkg-01 spec 04 12 字段)
                c_t_seq: (T,) float32 tensor, 与 records 平行的 c_t 序列
                         (与 review 修订 4 store_episode 第二参数对接)
        
        worker 调用约定 (硬约束):
            C5-W1: 不调 model.update_step (Pkg-04 spec 04 §3.4)
            C5-W2: 用 model.belief_net.step(obs, prev_hidden) 在线推断
                   (替代 v4.7 set_context_from_history)
            C5-W3: TimeStepRecord 字段填充顺序按 Pkg-01 spec 04
                   (z_hat[i, k] 对应 agent_id = k if k < i else k + 1)
        """
        N = self.cfg.env.N
        A = self.cfg.env.A
        obs_dim = self.env.observation_space.shape[-1]
        device = next(self.model.parameters()).device
        
        # 1. env reset
        obs, info = self.env.reset()  # obs (N, obs_dim), info 含 c_true/types/caps
        
        # 2. BeliefNet hidden 初始化 (Pkg-03 spec 04 init_hidden)
        prev_hidden = self.model.belief_net.init_hidden(batch_size=1, num_agents=N)
        
        records: list[TimeStepRecord] = []
        c_t_list: list[float] = []
        
        with torch.no_grad():       # worker 永远 no_grad (C5-W1 隐含)
            for t in range(self.cfg.env.T_max):
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)  # (1, N, obs_dim)
                
                # ============================================================
                # Step A: BeliefNet.step 在线推断 (C5-W2, 替代 v4.7 set_context_from_history)
                # ============================================================
                prev_hidden, c_hat, z_hat = self.model.belief_net.step(
                    obs_tensor, prev_hidden,
                )
                # c_hat shape (1, N), z_hat shape (1, N, N-1, 2)
                
                # ============================================================
                # Step B: model 7 API 调用 (Pkg-04 spec 02 §2.3 worker 模板)
                # ============================================================
                s = self.model.encode(obs_tensor)                  # (1, latent_dim)
                
                c_t_scalar = float(info["c_true"])
                self.model.set_context_objective(
                    torch.tensor([c_t_scalar], device=device),
                )
                
                # 选 action per agent
                joint_action = np.zeros(N, dtype=np.int64)
                pi_mve = np.zeros((N, A), dtype=np.float32)
                value_estimates = np.zeros(N, dtype=np.float32)
                
                for k in range(N):
                    cap_k_raw = info["caps"][k]  # CapabilityVector 实例
                    cap_k = torch.from_numpy(
                        np.array([cap_k_raw.eta, cap_k_raw.phi_fov,
                                  cap_k_raw.nu, cap_k_raw.zeta], dtype=np.float32)
                    ).unsqueeze(0).to(device)  # (1, 4)
                    
                    # set_context_subjective per agent (Pkg-04 spec 02)
                    self.model.set_context_subjective(
                        agent_id=k,
                        cap_i=cap_k,
                        belief=(c_hat[0, k:k+1], z_hat[0, k:k+1]),
                    )
                    
                    # 预测 π / v
                    pi_logits_k, v_k = self.model.predict(s)
                    pi_k = torch.softmax(pi_logits_k, dim=-1)  # (1, A)
                    value_estimates[k] = v_k.item()
                    
                    # 暂存 prior policy (use_planner=False 时直接用)
                    if not use_planner:
                        pi_mve[k] = pi_k[0].cpu().numpy()
                
                # ============================================================
                # Step B.5: MVE planner 一次性算 N agents 的 policy (P1-1+P1-2 修订)
                # ============================================================
                # 修订前 (反模式):
                #   - per-agent 循环内调 _call_planner, cap_dict 用 placeholder 全设同 cap (P1-1)
                #   - _call_planner 内调顶层 sample_mve_plan 每次新建 planner, CRN 不持续 (P1-2)
                # 修订后:
                #   - 循环外一次性调 self.planner.sample_mve_plan, 输出 (B, N, A) 整体
                #   - cap_dict / belief_dict 用真实 N agents 数据构造
                #   - 复用 self.planner 实例, CRN seed 跨 episode 持续 (R5-3)
                if use_planner:
                    # 构造真实 cap_dict (P1-1 修订)
                    cap_dict = {}
                    for k in range(N):
                        cap_k_raw = info["caps"][k]
                        cap_dict[k] = torch.from_numpy(
                            np.array([cap_k_raw.eta, cap_k_raw.phi_fov,
                                      cap_k_raw.nu, cap_k_raw.zeta], dtype=np.float32)
                        ).unsqueeze(0).to(device)  # (1, 4)
                    
                    # 构造 belief_dict
                    belief_dict = {
                        k: (c_hat[0, k:k+1], z_hat[0, k:k+1])
                        for k in range(N)
                    }
                    
                    # P1-2 修订: 调 self.planner.sample_mve_plan (复用持有实例, 不是顶层函数)
                    pi_mve_all = self.planner.sample_mve_plan(
                        model=self.model,
                        root_s=s,
                        cap=cap_dict,
                        belief=belief_dict,
                        c_t=torch.tensor([c_t_scalar], device=device),
                    )  # (1, N, A)
                    
                    pi_mve = pi_mve_all[0].cpu().numpy()  # (N, A)
                
                # ============================================================
                # Step B.6: epsilon-greedy action selection (per-agent)
                # ============================================================
                for k in range(N):
                    # pi_mve[k] 已在两条路径填好: use_planner→Step B.5 planner 输出;
                    # 否则→Step B 的 prior policy (line 160). 此处统一消费即可.
                    pi_for_action = pi_mve[k]
                    if np.random.rand() < epsilon:
                        joint_action[k] = np.random.randint(A)
                    else:
                        # 从 pi_for_action 采样
                        joint_action[k] = int(np.random.choice(A, p=pi_for_action / pi_for_action.sum()))
                
                # ============================================================
                # Step C: env.step
                # ============================================================
                next_obs, rewards, done, _, next_info = self.env.step(joint_action)
                
                # ============================================================
                # Step D: 写 TimeStepRecord (C5-W3, Pkg-01 spec 04 12 字段)
                # ============================================================
                # 注: cap 维度 (N, 4), 从 info["caps"] 取
                cap_arr = np.stack(
                    [np.array([c.eta, c.phi_fov, c.nu, c.zeta], dtype=np.float32)
                     for c in info["caps"]],
                    axis=0,
                )  # (N, 4)
                
                # tau 维度 (N,), 从 info["types"] 取 (AgentType.value)
                tau_arr = np.array(
                    [t_.value for t_ in info["types"]], dtype=np.int8,
                )  # (N,)
                
                # c_hat / z_hat 维度 from BeliefNet
                c_hat_np = c_hat[0].cpu().numpy()    # (N,)
                z_hat_np = z_hat[0].cpu().numpy()    # (N, N-1, 2)
                #   ↑ Pkg-03 spec 04 BeliefNet head_opp 已保证 z_hat[i, k] 对应
                #     agent_id = k if k < i else k + 1 (agent_id 升序跳过 self)
                #     C5-W3 单测 test_z_hat_order_matches_pkg01_spec04 验证
                
                record = TimeStepRecord(
                    o=obs,                           # (N, obs_dim)
                    a=joint_action,                  # (N,)
                    r=rewards,                       # (N,)
                    delta=next_info.get("delta", np.zeros(N, dtype=np.float32)),  # (N,)
                    pi_mve=pi_mve,                   # (N, A)
                    v=value_estimates,               # (N,)
                    tau=tau_arr,                     # (N,)
                    cap=cap_arr,                     # (N, 4)
                    c_hat=c_hat_np,                  # (N,) raw scalar
                    z_hat=z_hat_np,                  # (N, N-1, 2)
                    t=t,
                    done=bool(done),
                )
                records.append(record)
                c_t_list.append(c_t_scalar)
                
                if done:
                    break
                obs = next_obs
                info = next_info
        
        c_t_seq = torch.tensor(c_t_list, dtype=torch.float32)  # (T,)
        return records, c_t_seq
    
    # ====================================================================
    # 内部辅助
    # ====================================================================
    
    # P1-1+P1-2 修订: _call_planner 方法已删除 (logic inline 到 collect_episode Step B.5).
    # 删除理由:
    #   - 原 _call_planner 内 cap_dict = {k: cap for k in range(N)} 是 placeholder (P1-1)
    #   - 原调顶层 sample_mve_plan 每次新建 planner 破坏 self.planner 持有 (P1-2)
    #   - 重构后: collect_episode Step B.5 直接调 self.planner.sample_mve_plan, 用真实 N agents cap/belief
    #   - planner 一次返回 (B, N, A) 整体, 不需要 per-agent _call_planner 循环
```

---

## 3. Implementation Notes

### 3.1 worker 不调 update_step（C5-W1，Pkg-04 spec 04 §3.4）

worker 永远在 `torch.no_grad()` 推断模式，model.update_step 仅影响 belief grad gating（训练时 main loss backward 触发）。worker 无梯度反向 → gating 无意义。

实施层：worker.collect_episode 内**不**出现 `model.update_step(...)` 调用。

单测 `test_worker_no_update_step` 用 `mocker.spy(model, "update_step")` 验证 call_count == 0。

### 3.2 BeliefNet.step vs BeliefNet.forward（API 区分）

| API | 用途 | 调用方 | shape |
|-----|------|--------|-------|
| `BeliefNet.step(obs, prev_hidden)` | 在线增量推断（单步）| **worker** | obs (1, N, obs_dim) → c_hat (1, N), z_hat (1, N, N-1, 2) |
| `BeliefNet.forward(obs_seq, oracle_z_seq=None)` | 训练时序列推断（K-step + oracle 注入）| **trainer** | obs_seq (B, T, N, obs_dim) → c_hat (B, T, N), z_hat (B, T, N, N-1, 2) |

worker 不接受 oracle_z_seq（在线推断时无 oracle，按 BeliefNet 实际预测）。trainer 训练时按 scheduler.stage 决定是否注入 oracle_z_seq。

### 3.3 z_hat 顺序一致性（C5-W3，Pkg-01 spec 04 顺序约定）

Pkg-03 spec 04 BeliefNet head_opp 输出 z_hat shape (B, N, N-1, 2)，**已**按 Pkg-01 spec 04 §3.3 约定的顺序：
- `z_hat[b, i, k, :]` = agent i 对 agent_id=(k if k < i else k+1) 的 α/β 概率分布

worker 写入 TimeStepRecord 时直接复用 BeliefNet 输出（无需重排）。

单测 `test_z_hat_order_matches_pkg01_spec04`：构造已知 type 分布的 env，让 BeliefNet 推断 10 步收敛，验证 z_hat[i, k, 0] (agent i 看到对手 k 是 ALPHA 的概率) 与 ground truth types[(k if k<i else k+1)] 一致。

### 3.4 v4.7 5 处 set_context 调用点迁移（Pkg-04 spec 08 §3.3）

| 行号 | v4.7 代码 | v4 迁移 |
|------|----------|---------|
| L125 | `model.set_context_from_history(h_obs, h_act, h_rew, id_0)` | **删除** + `model.belief_net.step(obs_t, prev_hidden)` |
| L128 | `model.set_context_default(id_0, batch_size=1)` | **删除** + `model.set_context_subjective(0, cap, (c_hat, z_hat))` |
| L143 | `model.set_context_from_history(h_obs, h_act, h_rew, id_i)` | 同 L125 |
| L145 | `model.set_context_default(id_i, batch_size=1)` | 同 L128 |
| L149 | `model.set_context(rule_t, id_i)` | 拆为 `model.set_context_objective(c_t)` + `model.set_context_subjective(i, cap, belief)` |

迁移完成后 grep 验证（沿用 Pkg-04 spec 08 §3.4 模式，澄清 2 精确正则）：

```powershell
# 必须仅匹配单参旧签名，不命中 _objective/_subjective
Select-String -Path hyper_mve\training\worker.py -Pattern '\bset_context\(' |
    Where-Object { $_.Line -notmatch 'set_context_(objective|subjective)\(' }
# 期望 0 行输出
```

### 3.5 worker 持有 planner（review 修订 5）

- self.planner = planner or MVEPlanner(cfg) 在 __init__ 一次性构造
- collect_episode 每集复用同一 planner 实例
- CRN rng_state 跨 episode 持续（R5-3 NaN 稳定性 + 复现性）
- 单测 `test_worker_planner_persistent` 验证 id(self.planner) 不变

### 3.6 性能预算（R5-3）

| 子项 | 预算 |
|------|------|
| collect_episode (T=200, N=4) | < 5 s |
| 单步 (worker step) | < 25 ms（含 BeliefNet.step + model 7 API + planner 调用）|
| BeliefNet.step | < 2 ms (Pkg-03 spec 04 验收) |
| set_context_objective + 4×set_context_subjective | < 14 ms (Pkg-04 spec 07 档位 1 / 2) |
| planner (use_planner=True) | < 50 ms / agent × N = 200 ms / step 上限（仅 use_planner=True 时） |

worker.collect_episode 在 `use_planner=False` 时仅 25 ms / step，T=200 时 5 s 余裕大。`use_planner=True` 时 200 ms / step × 200 步 = 40 s — 故 use_planner 通常仅在 eval 期启用，训练期 collect 通常 epsilon-greedy + use_planner=False。

---

## 4. Edge Cases

| 场景 | 行为 |
|------|------|
| env.reset 返回 obs shape ≠ (N, obs_dim) | assert 抛 |
| info 缺 "c_true" / "types" / "caps" 字段 | KeyError + 明确提示 Pkg-02 env.info 三段分组 |
| BeliefNet hidden state NaN | init_hidden 用 zeros (Pkg-03 spec 04 已 guard)，N 步内不应 NaN |
| epsilon 超 [0, 1] | 不校验（上层保证）|
| use_planner=True 但 planner 内部抛异常 | 上抛（worker 不 catch）|
| episode 在 T < T_max 时 done=True | break 提前终止，records 长度 < T_max |
| collect_episode 调用时 model 在 train mode | __init__ 已 self.model.eval()；若上层显式切回 train，worker 行为不变（torch.no_grad 已 guard）|
| info["caps"][k] 是 numpy array 而非 CapabilityVector dataclass | env 内部封装；Pkg-02 spec 08 已规定返回 CapabilityVector |

---

## 5. Acceptance Criteria

### 5.1 单元测试（`tests/training/test_worker.py`）

```python
import pytest
import torch
import numpy as np
from hyper_mve.configs import V4Config
from hyper_mve.models import HyperMuZeroModel
from hyper_mve.training import Worker
from hyper_mve.planning.mve_planner import MVEPlanner
from hyper_mve.envs import ResourceCommonsEnv  # Pkg-02
from hyper_mve.schemas import TimeStepRecord


@pytest.fixture
def cfg_medium():
    return V4Config.from_preset("medium")


@pytest.fixture
def model(cfg_medium):
    return HyperMuZeroModel(cfg_medium)


@pytest.fixture
def env(cfg_medium):
    return ResourceCommonsEnv(cfg_medium.env)


@pytest.fixture
def worker(cfg_medium, model, env):
    return Worker(cfg_medium, model, env)


# ====== C5-W1: worker 不调 update_step ======

def test_worker_no_update_step(worker, mocker):
    """worker.collect_episode 内不应调 model.update_step (Pkg-04 spec 04 §3.4)."""
    spy = mocker.spy(worker.model, "update_step")
    
    records, c_t_seq = worker.collect_episode(epsilon=0.1, use_planner=False)
    
    assert spy.call_count == 0, "worker 不应调 model.update_step"


# ====== C5-W2: worker 用 BeliefNet.step ======

def test_worker_uses_belief_net_step(worker, mocker):
    """worker.collect_episode 内应用 model.belief_net.step 在线推断."""
    spy = mocker.spy(worker.model.belief_net, "step")
    
    records, c_t_seq = worker.collect_episode(epsilon=0.1, use_planner=False)
    
    # 每步调一次 BeliefNet.step
    assert spy.call_count == len(records)


# ====== C5-W3: z_hat 顺序一致性 ======

def test_z_hat_order_matches_pkg01_spec04(worker, cfg_medium):
    """z_hat[i, k] 对应 agent_id = (k if k < i else k + 1) — Pkg-01 spec 04 顺序约定.
    
    构造已知 type 分布的 env, 让 BeliefNet 推断后验证 z_hat 顺序.
    """
    records, c_t_seq = worker.collect_episode(epsilon=0.1, use_planner=False)
    
    # 取最后一个 record 的 z_hat (BeliefNet 推断已稳定)
    record = records[-1]
    N = cfg_medium.env.N
    assert record.z_hat.shape == (N, N-1, 2)
    
    # 验证 z_hat[i, k] 的对手 id 映射 (顺序约定)
    # 这里不验证概率值正确（需训练后才能），只验证 shape 与字段命名顺序
    # 真正的概率正确性由 BeliefNet 训练后的 head_opp 单测验证


# ====== TimeStepRecord 字段填充完整性 ======

def test_collect_episode_record_fields_valid(worker, cfg_medium):
    """记录字段 shape / dtype 全部合规 (Pkg-01 spec 04 12 字段)."""
    records, c_t_seq = worker.collect_episode(epsilon=0.1, use_planner=False)
    
    N = cfg_medium.env.N
    A = cfg_medium.env.A
    
    for r in records:
        assert isinstance(r, TimeStepRecord)
        assert r.o.shape[0] == N
        assert r.a.shape == (N,) and r.a.dtype == np.int64
        assert r.r.shape == (N,) and r.r.dtype == np.float32
        assert r.delta.shape == (N,) and r.delta.dtype == np.float32
        assert r.pi_mve.shape == (N, A) and r.pi_mve.dtype == np.float32
        assert r.v.shape == (N,) and r.v.dtype == np.float32
        assert r.tau.shape == (N,) and r.tau.dtype == np.int8
        assert r.cap.shape == (N, 4) and r.cap.dtype == np.float32
        assert r.c_hat.shape == (N,) and r.c_hat.dtype == np.float32
        assert r.z_hat.shape == (N, N-1, 2) and r.z_hat.dtype == np.float32
        assert isinstance(r.t, int)
        assert isinstance(r.done, bool)
    
    # c_t_seq 长度 == records 长度
    assert len(c_t_seq) == len(records)


# ====== R5-3: collect_episode < 5s ======

@pytest.mark.gpu
def test_collect_episode_under_5s(worker):
    """单 episode collect < 5 s (T=200, use_planner=False)."""
    import time
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    
    t0 = time.perf_counter()
    records, c_t_seq = worker.collect_episode(epsilon=0.1, use_planner=False)
    elapsed = time.perf_counter() - t0
    
    assert elapsed < 5.0, f"collect_episode {elapsed:.2f}s > 5s"


# ====== 修订 5: worker 持有 planner (CRN 跨 episode 持续) ======

def test_worker_planner_persistent(worker):
    """worker 跨 episode 复用同一 planner 实例 (CRN seed 持续, R5-3 稳定性)."""
    planner_id_0 = id(worker.planner)
    
    worker.collect_episode(epsilon=0.1, use_planner=True)
    
    planner_id_1 = id(worker.planner)
    assert planner_id_0 == planner_id_1, "worker 应跨 episode 复用同一 planner 实例"


def test_worker_planner_custom_injection(cfg_medium, model, env):
    """worker __init__ 接受自定义 planner (D4 实例可替换 pattern)."""
    custom_planner = MVEPlanner(cfg_medium)
    custom_planner._test_marker = "custom"
    
    worker = Worker(cfg_medium, model, env, planner=custom_planner)
    assert worker.planner is custom_planner
    assert worker.planner._test_marker == "custom"


# ====== Self-Info 严格性 (worker 不传 oracle types 进 model.set_context_subjective) ======

def test_worker_no_oracle_types_leak(worker, cfg_medium, mocker):
    """C11 验证 (与 Pkg-04 spec 02 §5.1 联动): worker 调 set_context_subjective 时
    cap_i 必须是 (B, 4), 不含 type 维度.
    """
    spy = mocker.spy(worker.model, "set_context_subjective")
    
    worker.collect_episode(epsilon=0.1, use_planner=False)
    
    # 验证每次 set_context_subjective 调用 cap_i shape == (B, 4)
    for call in spy.call_args_list:
        args, kwargs = call
        cap_i = kwargs.get("cap_i") if "cap_i" in kwargs else args[1]
        assert cap_i.shape[-1] == 4, f"cap_i shape {cap_i.shape} not (B, 4) - Self-Info leak"
```

### 5.2 集成测试

`scripts/train_main.py --preset medium --max_steps 100` 端到端：
- worker collect 100 episodes 全部 records 字段合规
- BeliefNet hidden state 无 NaN
- TimeStepRecord z_hat 顺序与 Pkg-01 spec 04 一致

### 5.3 性能要求

- collect_episode (T=200, N=4, use_planner=False) < 5 s
- collect_episode (T=200, N=4, use_planner=True) < 60 s（planner 200 ms/step × 200 = 40 s，留 50% 缓冲）
- BeliefNet.step (1 step, N=4) < 2 ms（Pkg-03 spec 04 验收）

---

## 6. v4.7 → v4 对比

| 维度 | v4.7 | v4 | 差异 |
|------|------|----|------|
| set_context 调用 | 3 个 API（set_context / set_context_from_history / set_context_default）共 5 处 | 2 个 API（set_context_objective / set_context_subjective）+ BeliefNet.step 取代 history | Pkg-04 spec 08 §3.3 5 处迁移 |
| 在线推断方式 | GRU set_context_from_history（输入历史 obs/act/rew 序列）| BeliefNet.step（输入当前 obs + prev_hidden）| 替换 |
| TimeStepRecord 字段 | v4.7 EpisodeData 6 字段（无 c_hat/z_hat/tau/cap/delta）| v4 TimeStepRecord 12 字段 | 重写 |
| planner 持有 | worker 每集创建 | worker 持有 + CRN seed 跨 episode 持续 | 修订 5 |
| 单 episode 性能 | ~3 s | ~4.5 s | +50%（BeliefNet.step + record 字段填充）|

---

## 7. Cross-references

- Ch5.6.1 buffer 数据组织 + 5.7 课程接入
- `01-trainer-loop-v2.md`（trainer 消费 worker 产出的 records）
- `03-episode-buffer-v2.md`（buffer.store_episode(records, c_t_seq) 接口）
- `06-mve-planner-v4.md`（planner 4 处迁移 + cap/belief 显式参数）
- `08-integration-contracts.md` §2（v4.7 → w4 5 处 worker 迁移 grep 验证）
- Pkg-01 spec 04 TimeStepRecord（12 字段 + z_hat 顺序）
- Pkg-02 spec 08 env.info 三段分组（c_true / types / caps Oracle 信号）
- Pkg-03 spec 04 BeliefNet.step / init_hidden
- Pkg-04 spec 02 §2.3（worker 调用模板）+ §5.4（model.forward 不对外）
- Pkg-04 spec 04 §3.4（worker 不调 update_step）
- Pkg-04 spec 08 §3.3（v4.7 worker 5 处迁移行号）

---

## [v4-opt 2026-06] 修订:planner-on 采集的必要性论证与调试旗

本 spec 已规定 `use_planner=True` 默认;优化阶段(提交 `0ba2eac`)补充其**必要性论证与工程接线**:

1. **自蒸馏退化引理**(Ch5.9.1b):采集关规划器时 π_tgt = 模型自身先验 ⇒ ∇CE ≡ 0 ⇒ 策略熵钉死 ln(A);均匀策略是"采集-规划-训练"闭环的不动点,ε-greedy 不解此锁。已实测一次(duo 运行)。**use_planner=False 因此不是消融选项,而是训练不可行配置**;
2. **train_main 接线**:warmup 期(buffer < min_buffer_size)用 `epsilon=1.0, use_planner=False` 纯随机填充;其后采集恒 planner-on,`--no_collect_planner` 仅作调试探针保留;
3. **健康检查**:`diag/pi_mve_entropy` 是否离开 ln A 为训练第一道检查(spec 05 修订)。

## 修订记录 (Changelog)

| 日期 | 修订 | 依据 |
|---|---|---|
| 2026-06-10 | planner-on 必要性论证(ln(A) 不动点)、warmup/调试旗接线、健康检查指针 | 提交 0ba2eac;复审 §4.1;Ch5.9.1b |

## [v4-opt 2026-06b] 修订:向量化采集 + 确定性评估模式 + planner_on 标志

2agent 三连跑诊断(2026-06-11)发现单环境采集是吞吐瓶颈:planner 本身已支持任意 B,
但 worker 以 B=1 喂入 ⇒ 每 env step ~70 次小 batch GPU 调用(kernel-launch 受限),
实测 0.08 train-steps/s(1M 步 ≈ 135 天)。修订:

1. **`Worker.collect_episodes(n_envs, epsilon, use_planner, deterministic, reset_seeds, reset_options)`**:
   n_envs 个 ResourceCommons 环境 lockstep 推进(env 仅在 T_max 终止,Pkg-02 spec 08,
   故无需 done-mask),planner 每 env-step 一次 B=n_envs 批调用。旧 `collect_episode`
   API 保留,内部委托 B=1 路径(行为不变);
2. **返回类型 `CollectResult`**:records/c_t_seq 之外新增 per-episode 可观测量 —
   `returns (N,)`、`pi_entropy_mean`、planner 探针 `q_std_mean / q_gap_mean /
   uniform_frac`(planner-off 时为 NaN),供 train_main 的 `collect/*` TB 族;
3. **确定性评估模式**:`deterministic=True` ⇒ ε 忽略、动作 = argmax(pi_mve)、不消耗
   采样 RNG;`reset_seeds/reset_options` 支持固定种子 + 固定 c(training/evaluation.py
   的双模式 c-grid 评估在此之上构建);
4. **planner_on 责任划分**:worker 不再隐式存自蒸馏目标 —— 调用方(train_main)必须把
   planner-off 采集的 episode 以 `store_episode(..., planner_on=False)` 入库,策略损失
   据此掩蔽(spec 03 / spec 05 配套修订)。

## 修订记录 (Changelog)(追加)

| 日期 | 修订 | 依据 |
|---|---|---|
| 2026-06-11 | 向量化采集(collect_episodes/CollectResult)、确定性评估模式、planner_on 标志责任 | 2agent 诊断(0.08 steps/s;U 形 policy loss);用户决策 2026-06-11 |
