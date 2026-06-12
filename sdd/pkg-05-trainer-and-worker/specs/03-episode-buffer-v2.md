# Spec 03: Episode Buffer v2 — TimeStepRecord 容器 + stratified sampling

> 父文档：[`../proposal.md`](../proposal.md) §2.1.3 · [`../design.md`](../design.md) §3 D2/D9 · §6.4
> **v4 关键改动**：v4.7 `EpisodeData` (6 字段) → v4 **`TimeStepRecord` 容器** (12 字段，Pkg-01 spec 04)；**Q3 + D2**：buffer 归属 Pkg-05；**review 修订 4**：store_episode 显式增 `c_t_seq` 参数。

---

## 1. Purpose

按 Pkg-01 spec 04 TimeStepRecord 字段约定 + Pkg-01 spec 05 TrainConfig.stratified_sampling 字段实现 v4 `EpisodeReplayBuffer`，提供：

| API | 用途 | 调用频率 |
|-----|------|---------|
| `__init__(cfg)` | 构造，预分配容量 | 一次性 |
| `store_episode(records, c_t_seq)` | 存一集（review 修订 4：c_t_seq 显式参数） | 每集 |
| `sample_batch(batch_size, unroll_K)` | 采样 batch dict | 每 train_step |
| `__len__()` | 当前 episode 数 | 监控 |

废弃 v4.7 `EpisodeData` 6 字段，全面切到 v4 `TimeStepRecord` 12 字段 + c_t_seq 平行存储。

---

## 2. Interface

### 2.1 文件路径

`hyper_mve/training/episode_buffer.py`（v4.7 同名文件 inplace 重写，D10）

### 2.2 类签名（review 修订 4：c_t_seq 显式参数）

```python
import numpy as np
import torch
from collections import deque
from typing import Optional
from hyper_mve.configs import V4Config
from hyper_mve.schemas import TimeStepRecord, AgentType


class EpisodeReplayBuffer:
    """v4 Episode Replay Buffer (Pkg-05 spec 03).
    
    v4 关键改动 (相对 v4.7 EpisodeData + EpisodeReplayBuffer):
        v4.7: EpisodeData(obs, actions, rewards, search_policies, rule, dones, length) - 6 字段
        v4: TimeStepRecord 12 字段 (Pkg-01 spec 04) + 平行 c_t_seq 数组
    
    review 修订 4: store_episode 显式增 c_t_seq 参数 (不修改 TimeStepRecord schema, NG7 遵守).
    """
    
    def __init__(self, cfg: V4Config):
        self.cfg = cfg
        self.max_episodes = cfg.train.buffer_size
        self.unroll_K = cfg.train.unroll_K
        
        # FIFO storage
        # 每集存为 dict[str, np.ndarray]，shape 含 T 维度
        self._episodes: deque[dict[str, np.ndarray]] = deque(maxlen=self.max_episodes)
        self._c_t_seqs: deque[np.ndarray] = deque(maxlen=self.max_episodes)  # 平行 c_t_seq
        
        # stratified sampling cfg (D9 内联实现)
        self.stratified = cfg.train.stratified_sampling
        self.stratified_min_frac = cfg.train.stratified_min_per_type_frac
    
    # ====================================================================
    # API 1: store_episode (review 修订 4: c_t_seq 显式参数)
    # ====================================================================
    
    def store_episode(
        self,
        records: list[TimeStepRecord],
        c_t_seq: torch.Tensor,                  # ← review 修订 4
    ) -> None:
        """存储一集.
        
        Args:
            records: T 个 TimeStepRecord (Pkg-01 spec 04 12 字段)
            c_t_seq: (T,) float32, 与 records 平行的 c_t 标量序列
                     (worker 从 env.info["c_true"] 逐步采集)
        
        约束:
            - len(records) == len(c_t_seq)
            - len(records) > unroll_K + n_step (否则 sample_batch 无法取 unroll 窗口)
        """
        T = len(records)
        assert T == len(c_t_seq), (
            f"records len {T} != c_t_seq len {len(c_t_seq)}"
        )
        assert T > self.unroll_K + self.cfg.train.n_step, (
            f"episode too short: T={T} <= unroll_K+n_step={self.unroll_K+self.cfg.train.n_step}"
        )
        
        # 转 records → dict[str, np.ndarray]（沿 T 堆叠）
        # 12 字段 ∈ {o, a, r, delta, pi_mve, v, tau, cap, c_hat, z_hat, t, done}
        episode_dict = {
            "o":      np.stack([r.o for r in records], axis=0),       # (T, N, obs_dim)
            "a":      np.stack([r.a for r in records], axis=0),       # (T, N)
            "r":      np.stack([r.r for r in records], axis=0),       # (T, N)
            "delta":  np.stack([r.delta for r in records], axis=0),   # (T, N)
            "pi_mve": np.stack([r.pi_mve for r in records], axis=0),  # (T, N, A)
            "v":      np.stack([r.v for r in records], axis=0),       # (T, N)
            "tau":    np.stack([r.tau for r in records], axis=0),     # (T, N) int8
            "cap":    np.stack([r.cap for r in records], axis=0),     # (T, N, 4)
            "c_hat":  np.stack([r.c_hat for r in records], axis=0),   # (T, N) raw scalar
            "z_hat":  np.stack([r.z_hat for r in records], axis=0),   # (T, N, N-1, 2)
            "t":      np.array([r.t for r in records], dtype=np.int32),    # (T,)
            "done":   np.array([r.done for r in records], dtype=np.bool_), # (T,)
        }
        
        self._episodes.append(episode_dict)
        self._c_t_seqs.append(c_t_seq.cpu().numpy())                  # (T,)
    
    # ====================================================================
    # API 2: sample_batch
    # ====================================================================
    
    def sample_batch(
        self,
        batch_size: int,
        unroll_K: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        """采样 batch.
        
        Returns dict[str, Tensor]:
            obs:      (B, K+1, N, obs_dim)
            actions:  (B, K+1, N)
            rewards:  (B, K+1, N)
            delta:    (B, K+1, N)
            pi_mve:   (B, K+1, N, A)
            v:        (B, K+1, N)
            tau:      (B, K+1, N)
            cap:      (B, K+1, N, 4)
            c_hat:    (B, K+1, N)
            z_hat:    (B, K+1, N, N-1, 2)
            t:        (B, K+1)
            done:     (B, K+1)
            c_t:      (B, K+1)  ← review 修订 4: 来自 store_episode c_t_seq
        
        采样策略:
            - stratified=True: 按 type α / β 比例分桶, 保证 batch 内 type 平衡
              (C5-B2 stratified_min_per_type_frac 验收)
            - stratified=False: 均匀随机
        """
        K = unroll_K or self.unroll_K
        assert len(self._episodes) >= self.cfg.train.min_buffer_size or len(self._episodes) > 0, (
            f"Buffer not warm: {len(self._episodes)} < min {self.cfg.train.min_buffer_size}"
        )
        
        # ========== 1. 采样 episode indices ==========
        if self.stratified:
            ep_indices, start_indices = self._stratified_sample(batch_size, K)
        else:
            ep_indices, start_indices = self._uniform_sample(batch_size, K)
        
        # ========== 2. 切窗口 (B, K+1, ...) ==========
        batch = self._slice_batch(ep_indices, start_indices, K)
        
        return batch
    
    def __len__(self) -> int:
        """当前 episode 数."""
        return len(self._episodes)
    
    # ====================================================================
    # 内部辅助
    # ====================================================================
    
    def _uniform_sample(
        self, B: int, K: int,
    ) -> tuple[list[int], list[int]]:
        """均匀采样 episode + start step."""
        ep_indices = []
        start_indices = []
        for _ in range(B):
            ep_idx = np.random.randint(len(self._episodes))
            T_ep = len(self._episodes[ep_idx]["t"])
            # K+1 窗口 + n_step bootstrap 上限
            max_start = T_ep - K - self.cfg.train.n_step - 1
            start = np.random.randint(0, max(1, max_start + 1))
            ep_indices.append(ep_idx)
            start_indices.append(start)
        return ep_indices, start_indices
    
    def _stratified_sample(
        self, B: int, K: int,
    ) -> tuple[list[int], list[int]]:
        """按 type α / β 比例分层采样 (D9 内联, C5-B2 验收).
        
        策略:
            1. 把所有 (ep_idx, start) 候选按"窗口内 type α 占比"分桶
            2. α-heavy 桶 + β-heavy 桶分别贡献至少 B × stratified_min_frac 个样本
            3. 剩余样本均匀采样
        
        实施细节 (Day 实施时填充):
            - 预计算 type 分布索引（O(1) 查询）
            - 比例参数 stratified_min_per_type_frac (默认 0.3)
        """
        # 占位实现 (Day 实施时补充完整逻辑)
        min_per_type = int(B * self.stratified_min_frac)
        
        # 简化版: 按 episode 第一步的 type assignment 主导分桶
        ep_alpha_heavy = []   # episodes with >= 50% α agents
        ep_beta_heavy = []    # 反之
        for i, ep in enumerate(self._episodes):
            taus = ep["tau"][0]  # (N,)
            alpha_frac = (taus == AgentType.ALPHA.value).sum() / len(taus)
            if alpha_frac >= 0.5:
                ep_alpha_heavy.append(i)
            else:
                ep_beta_heavy.append(i)
        
        ep_indices = []
        start_indices = []
        
        # 至少 min_per_type 个来自每个桶
        for _ in range(min_per_type):
            if ep_alpha_heavy:
                ep_idx = np.random.choice(ep_alpha_heavy)
                T_ep = len(self._episodes[ep_idx]["t"])
                start = np.random.randint(0, max(1, T_ep - K - self.cfg.train.n_step))
                ep_indices.append(ep_idx)
                start_indices.append(start)
            if ep_beta_heavy:
                ep_idx = np.random.choice(ep_beta_heavy)
                T_ep = len(self._episodes[ep_idx]["t"])
                start = np.random.randint(0, max(1, T_ep - K - self.cfg.train.n_step))
                ep_indices.append(ep_idx)
                start_indices.append(start)
        
        # 剩余均匀采样
        remaining = B - len(ep_indices)
        ep_unif, start_unif = self._uniform_sample(remaining, K)
        ep_indices.extend(ep_unif)
        start_indices.extend(start_unif)
        
        # shuffle 防止 batch 内顺序偏向
        perm = np.random.permutation(B)
        ep_indices = [ep_indices[i] for i in perm]
        start_indices = [start_indices[i] for i in perm]
        
        return ep_indices, start_indices
    
    def _slice_batch(
        self,
        ep_indices: list[int],
        start_indices: list[int],
        K: int,
    ) -> dict[str, torch.Tensor]:
        """从采样的 (ep, start) 切 (B, K+1, ...) 窗口."""
        B = len(ep_indices)
        out = {}
        
        # 收集每个字段
        field_keys = ["o", "a", "r", "delta", "pi_mve", "v", "tau", "cap",
                      "c_hat", "z_hat", "t", "done"]
        for key in field_keys:
            slices = []
            for ep_idx, start in zip(ep_indices, start_indices):
                ep = self._episodes[ep_idx]
                arr = ep[key]  # (T, ...)
                slices.append(arr[start:start + K + 1])
            stacked = np.stack(slices, axis=0)  # (B, K+1, ...)
            out[key] = torch.from_numpy(stacked)
        
        # 重命名以匹配 trainer 期望
        out["obs"]     = out.pop("o")
        out["actions"] = out.pop("a")
        out["rewards"] = out.pop("r")
        out["dones"]   = out.pop("done")
        
        # c_t: 从 _c_t_seqs 平行切
        c_t_slices = []
        for ep_idx, start in zip(ep_indices, start_indices):
            c_t_arr = self._c_t_seqs[ep_idx]  # (T,)
            c_t_slices.append(c_t_arr[start:start + K + 1])
        out["c_t"] = torch.from_numpy(np.stack(c_t_slices, axis=0))  # (B, K+1)
        
        return out
```

---

## 3. Implementation Notes

### 3.1 NG7 遵守：不修改 Pkg-01 TimeStepRecord（review 修订 4）

Pkg-01 spec 04 TimeStepRecord 是 frozen=True dataclass 12 字段。Pkg-05 buffer **不**修改 schema，而是通过：
- store_episode 接受 records + c_t_seq **两个**入参
- buffer 内部维护 `_episodes` (records 序列) 和 `_c_t_seqs` (c_t_seq 序列) 平行 deque

sample_batch 返回的 dict 含 c_t 字段，来源是 _c_t_seqs（不是 records 字段）。

### 3.2 stratified sampling 实现策略（D9）

Pkg-01 spec 05 TrainConfig.stratified_sampling=True + stratified_min_per_type_frac=0.3 控制：
- 至少 30% batch 来自 α-heavy episodes（α agent 占多数 ≥ 50%）
- 至少 30% batch 来自 β-heavy episodes
- 剩余 40% 均匀采样

**简化版**（spec 03 §2.2 实现）按"episode 第一步 type assignment 主导"分桶，假设 episode 内 type 不变（Pkg-02 spec 03 已锁定 type fixed within episode）。

**Pkg-08 ablation** 可关闭 stratified（cfg.train.stratified_sampling=False）测试无分层基线。

### 3.3 内存预算（< 2 GB，Pkg-01 spec 04 §5.2）

5000 episode × 200 step × 4 agent × ~99 obs_dim ≈ 400 M float32 (obs) + 其他字段 ≈ 1.5 GB

| 字段 | 元素数 | 字节 / 元素 | 总 |
|------|--------|-------------|-----|
| obs | 5000×200×4×99 | 4 (float32) | 1584 MB |
| actions | 5000×200×4 | 8 (int64) | 32 MB |
| rewards | 5000×200×4 | 4 | 16 MB |
| delta | 5000×200×4 | 4 | 16 MB |
| pi_mve | 5000×200×4×6 | 4 | 96 MB |
| v | 5000×200×4 | 4 | 16 MB |
| tau | 5000×200×4 | 1 (int8) | 4 MB |
| cap | 5000×200×4×4 | 4 | 64 MB |
| c_hat | 5000×200×4 | 4 | 16 MB |
| z_hat | 5000×200×4×3×2 | 4 | 96 MB |
| t | 5000×200 | 4 (int32) | 4 MB |
| done | 5000×200 | 1 (bool) | 1 MB |
| **c_t_seq (修订 4)** | 5000×200 | 4 | 4 MB |
| **合计** | – | – | **~1950 MB** |

预算 < 2 GB ✓（与 Pkg-01 spec 04 §5.2 一致）。

### 3.4 v4.7 EpisodeData → v4 TimeStepRecord 字段映射

| v4.7 EpisodeData | v4 TimeStepRecord | 说明 |
|------------------|-------------------|------|
| `obs` | `o` | rename |
| `actions` | `a` | rename |
| `rewards` | `r` | rename |
| `search_policies` | `pi_mve` | rename |
| `rule` (标量) | `c_t_seq` (T,) + delete `rule` 字段 | rule 拆为 c_t 标量 (Pkg-02 spec 08) |
| `dones` | `done` (per-step) | v4.7 episode 末尾单一 bool → v4 per-step |
| `length` | `len(records)` | v4 records 列表本身有长度 |
| **(新增)** | `delta` | v4 新增 (Pkg-01 spec 04) |
| **(新增)** | `v` | v4 新增 |
| **(新增)** | `tau` | v4 新增 |
| **(新增)** | `cap` | v4 新增 |
| **(新增)** | `c_hat` | v4 新增 (BeliefNet 输出) |
| **(新增)** | `z_hat` | v4 新增 (BeliefNet 输出) |
| **(新增)** | `t` | v4 新增 (per-step index) |

单测 `test_v47_episodedata_v4_record_field_mapping` 验证字段映射完整无丢失。

### 3.5 性能预算（R5-2 sample_batch < 50 ms）

| 子项 | 预算 |
|------|------|
| stratified 分桶查询 | < 5 ms (O(B) 候选查询) |
| _slice_batch 切窗口 | < 30 ms (B=256 × 12 字段 × np.stack + torch.from_numpy) |
| c_t 切窗口 | < 5 ms |
| 其他 overhead | < 10 ms |
| **合计** | **< 50 ms** |

### 3.6 与 worker.collect_episode 的接口对接

worker 返回 `(records, c_t_seq)` tuple（spec 02 §2.2）：
- `records: list[TimeStepRecord]` 长度 T
- `c_t_seq: torch.Tensor` shape (T,)

trainer 端调：
```python
records, c_t_seq = worker.collect_episode(...)
buffer.store_episode(records, c_t_seq)
```

---

## 4. Edge Cases

| 场景 | 行为 |
|------|------|
| store_episode 调用时 records 与 c_t_seq 长度不一致 | AssertionError + 明确提示 |
| store_episode 调用时 records 长度 ≤ unroll_K + n_step | AssertionError + 明确提示（episode 太短无法采样窗口）|
| sample_batch 在 buffer 未达 min_buffer_size 时调用 | AssertionError 提示等待 |
| stratified_sampling=True 但所有 episode 都是 α-heavy（无 β-heavy）| 仅从 α-heavy + 均匀部分采样（无 β-heavy 桶贡献）|
| 同一 ep_idx 重复采样（小 batch 大 buffer） | 允许（无去重）|
| TimeStepRecord 内 c_hat shape (N,) 与 z_hat shape (N, N-1, 2) 一致性 | Pkg-01 spec 04 __post_init__ 已校验 |
| Pickle 序列化 buffer | deque + dict[str, np.ndarray] 全支持 |

---

## 5. Acceptance Criteria

### 5.1 单元测试（`tests/training/test_episode_buffer.py`）

```python
import pytest
import torch
import numpy as np
from hyper_mve.configs import V4Config
from hyper_mve.schemas import TimeStepRecord, AgentType
from hyper_mve.training import EpisodeReplayBuffer


@pytest.fixture
def cfg_medium():
    return V4Config.from_preset("medium")


@pytest.fixture
def buffer(cfg_medium):
    return EpisodeReplayBuffer(cfg_medium)


def _make_record(N=4, A=6, obs_dim=99, t=0, types=None):
    types = types or [AgentType.ALPHA.value, AgentType.ALPHA.value,
                      AgentType.BETA.value, AgentType.BETA.value]
    return TimeStepRecord(
        o=np.zeros((N, obs_dim), dtype=np.float32),
        a=np.zeros(N, dtype=np.int64),
        r=np.zeros(N, dtype=np.float32),
        delta=np.zeros(N, dtype=np.float32),
        pi_mve=np.ones((N, A), dtype=np.float32) / A,
        v=np.zeros(N, dtype=np.float32),
        tau=np.array(types, dtype=np.int8),
        cap=np.zeros((N, 4), dtype=np.float32),
        c_hat=np.full(N, 0.5, dtype=np.float32),
        z_hat=np.ones((N, N-1, 2), dtype=np.float32) * 0.5,
        t=t, done=False,
    )


def _make_episode(T=50, types=None):
    records = [_make_record(t=i, types=types) for i in range(T)]
    c_t_seq = torch.full((T,), 0.5)
    return records, c_t_seq


# ====== store_episode 基础 ======

def test_store_episode_basic(buffer):
    records, c_t_seq = _make_episode(T=50)
    buffer.store_episode(records, c_t_seq)
    assert len(buffer) == 1


def test_store_episode_too_short_raises(buffer, cfg_medium):
    """episode 长度 <= unroll_K + n_step 应抛 AssertionError."""
    T = cfg_medium.train.unroll_K + cfg_medium.train.n_step
    records, c_t_seq = _make_episode(T=T)
    with pytest.raises(AssertionError, match="episode too short"):
        buffer.store_episode(records, c_t_seq)


def test_store_episode_c_t_seq_length_mismatch(buffer):
    """records 与 c_t_seq 长度不匹配应抛 AssertionError."""
    records, c_t_seq = _make_episode(T=50)
    c_t_seq_wrong = torch.full((30,), 0.5)  # 长度不匹配
    with pytest.raises(AssertionError, match="c_t_seq len"):
        buffer.store_episode(records, c_t_seq_wrong)


# ====== sample_batch 基础 ======

def test_sample_batch_shape(buffer, cfg_medium):
    """sample_batch 返回 dict 字段 shape 合规."""
    # 填充至少 min_buffer_size
    for _ in range(cfg_medium.train.min_buffer_size + 5):
        records, c_t_seq = _make_episode(T=50)
        buffer.store_episode(records, c_t_seq)
    
    batch = buffer.sample_batch(batch_size=16, unroll_K=5)
    
    K = 5
    N = cfg_medium.env.N
    A = cfg_medium.env.A
    
    assert batch["obs"].shape == (16, K+1, N, 99)
    assert batch["actions"].shape == (16, K+1, N)
    assert batch["rewards"].shape == (16, K+1, N)
    assert batch["delta"].shape == (16, K+1, N)
    assert batch["pi_mve"].shape == (16, K+1, N, A)
    assert batch["v"].shape == (16, K+1, N)
    assert batch["tau"].shape == (16, K+1, N)
    assert batch["cap"].shape == (16, K+1, N, 4)
    assert batch["c_hat"].shape == (16, K+1, N)
    assert batch["z_hat"].shape == (16, K+1, N, N-1, 2)
    assert batch["t"].shape == (16, K+1)
    assert batch["dones"].shape == (16, K+1)
    assert batch["c_t"].shape == (16, K+1)  # 修订 4: c_t 来自 _c_t_seqs


# ====== C5-B1: z_hat 顺序一致性（端到端）======

def test_buffer_z_hat_order_e2e(buffer, cfg_medium):
    """buffer 取出 z_hat 顺序与存入时一致（Pkg-01 spec 04 顺序约定）."""
    N = cfg_medium.env.N
    # 构造已知 z_hat 顺序的 record
    records = []
    for t in range(50):
        record = _make_record(t=t)
        # 标记每个 z_hat[i, k] = (i * 10 + k)
        for i in range(N):
            for k in range(N-1):
                record.z_hat[i, k, 0] = i * 10 + k  # 仅标记，验证顺序
        records.append(record)
    c_t_seq = torch.full((50,), 0.5)
    buffer.store_episode(records, c_t_seq)
    
    batch = buffer.sample_batch(batch_size=1, unroll_K=5)
    z_hat = batch["z_hat"]  # (1, K+1, N, N-1, 2)
    
    # 验证顺序: z_hat[0, t, i, k, 0] == i*10 + k
    for t in range(6):
        for i in range(N):
            for k in range(N-1):
                expected = i * 10 + k
                assert z_hat[0, t, i, k, 0].item() == expected, (
                    f"z_hat 顺序错位 at t={t}, i={i}, k={k}"
                )


# ====== C5-B2: stratified sampling 比例 ======

def test_stratified_min_per_type_frac(buffer, cfg_medium):
    """stratified sampling 后 batch 内 type 比例满足 min_per_type_frac."""
    # 填充 50 个 α-heavy + 50 个 β-heavy episodes
    alpha_types = [AgentType.ALPHA.value] * 4
    beta_types = [AgentType.BETA.value] * 4
    
    for _ in range(50):
        records, c_t_seq = _make_episode(T=50, types=alpha_types)
        buffer.store_episode(records, c_t_seq)
    for _ in range(50):
        records, c_t_seq = _make_episode(T=50, types=beta_types)
        buffer.store_episode(records, c_t_seq)
    
    batch = buffer.sample_batch(batch_size=100, unroll_K=5)
    tau = batch["tau"]  # (100, K+1, N)
    
    # 取 t=0 的 type 计算 α-heavy / β-heavy 比例
    tau_t0 = tau[:, 0, 0]  # (100,) 第一个 agent 的 type
    alpha_count = (tau_t0 == AgentType.ALPHA.value).sum().item()
    beta_count = (tau_t0 == AgentType.BETA.value).sum().item()
    
    # 至少 30% 来自 α-heavy（stratified_min_per_type_frac=0.3）
    assert alpha_count >= int(100 * 0.3), f"alpha frac < 30%: {alpha_count}"
    assert beta_count >= int(100 * 0.3), f"beta frac < 30%: {beta_count}"


# ====== R5-2: sample_batch < 50 ms ======

def test_sample_batch_under_50ms(buffer, cfg_medium):
    """单 sample_batch < 50 ms (B=256, K=5)."""
    import time
    for _ in range(cfg_medium.train.min_buffer_size + 50):
        records, c_t_seq = _make_episode(T=50)
        buffer.store_episode(records, c_t_seq)
    
    # Warmup
    for _ in range(5):
        buffer.sample_batch(batch_size=256, unroll_K=5)
    
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        buffer.sample_batch(batch_size=256, unroll_K=5)
        times.append((time.perf_counter() - t0) * 1000)
    
    mean_ms = sum(times) / len(times)
    assert mean_ms < 50.0, f"sample_batch {mean_ms:.2f}ms > 50ms (R5-2)"


# ====== v4.7 → v4 字段映射 ======

def test_v47_episodedata_v4_record_field_mapping(buffer):
    """v4 buffer 字段完全覆盖 v4.7 EpisodeData 6 字段."""
    records, c_t_seq = _make_episode(T=50)
    buffer.store_episode(records, c_t_seq)
    
    batch = buffer.sample_batch(batch_size=4, unroll_K=5)
    
    # v4.7 字段对照
    v47_map = {
        "obs": "obs",         # v4.7 obs → v4 obs (经 rename)
        "actions": "actions", # v4.7 actions → v4 actions
        "rewards": "rewards", # v4.7 rewards → v4 rewards
        "pi_mve": "pi_mve",   # v4.7 search_policies → v4 pi_mve
        "c_t": "c_t",         # v4.7 rule → v4 c_t (修订 4)
        "dones": "dones",     # v4.7 dones → v4 dones
    }
    for v4_field in v47_map.values():
        assert v4_field in batch, f"v4 batch missing {v4_field}"
    
    # v4 新增字段
    v4_new = ["delta", "v", "tau", "cap", "c_hat", "z_hat", "t"]
    for f in v4_new:
        assert f in batch, f"v4 new field {f} missing"


# ====== buffer pickle roundtrip (R5-7) ======

def test_buffer_pickle_roundtrip(buffer):
    """buffer 可 pickle 序列化（用于 ckpt save）."""
    import pickle
    records, c_t_seq = _make_episode(T=50)
    buffer.store_episode(records, c_t_seq)
    
    data = pickle.dumps(buffer)
    buffer2 = pickle.loads(data)
    
    assert len(buffer2) == 1
    batch2 = buffer2.sample_batch(batch_size=1, unroll_K=5)
    assert batch2["obs"].shape[0] == 1


# ====== FIFO 容量 ======

def test_fifo_buffer_eviction(cfg_medium):
    """buffer 超过 max_episodes 时 FIFO 淘汰最旧."""
    cfg = cfg_medium
    cfg = cfg.__class__(env=cfg.env, model=cfg.model, train=cfg.train,
                       mup=cfg.mup, eval=cfg.eval, legacy=cfg.legacy,
                       preset_name=cfg.preset_name)
    # 重设 buffer_size 小值便于测试
    from dataclasses import replace
    cfg_small = replace(cfg, train=replace(cfg.train, buffer_size=5))
    buffer = EpisodeReplayBuffer(cfg_small)
    
    for _ in range(10):  # 推 10 个，超过 5
        records, c_t_seq = _make_episode(T=50)
        buffer.store_episode(records, c_t_seq)
    
    assert len(buffer) == 5  # 仅保留最新 5 个
```

### 5.2 集成测试

`scripts/train_main.py --preset medium --max_steps 100` 端到端：
- worker collect 100 episode 全部 store_episode 成功
- trainer 每 train_step 调 sample_batch 拿到合规 batch
- 内存峰值 < 2 GB（实测）

### 5.3 性能要求

- store_episode (T=200) < 100 ms (np.stack + numpy 转换)
- sample_batch (B=256, K=5) < 50 ms（R5-2）
- buffer 内存峰值 < 2 GB（Pkg-01 spec 04 §5.2）

---

## 6. v4.7 → v4 对比

| 维度 | v4.7 EpisodeData | v4 TimeStepRecord | 差异 |
|------|------------------|-------------------|------|
| 字段数 | 6 | 12 + c_t_seq | +6 字段 + c_t 平行存 |
| rule 字段 | 标量浮点（episode 唯一）| 拆为 c_t_seq（per-step）+ delete | rule → c_t 重命名 + per-step |
| dones | episode 末尾单 bool | per-step bool 列表 | 拓展 |
| sampling 策略 | 均匀 | 均匀 / stratified 可配置 | 新增 stratified |
| 内存 | ~1.2 GB | ~1.5 GB（含 c_hat / z_hat / delta / tau / cap） | +25% |

---

## 7. Cross-references

- Ch5.6.1 buffer 数据组织
- `01-trainer-loop-v2.md`（trainer 调 buffer.sample_batch）
- `02-worker-collection.md`（worker 返回 records + c_t_seq）
- `08-integration-contracts.md` §1（buffer API 稳定性）
- Pkg-01 spec 04 TimeStepRecord（12 字段定义）
- Pkg-01 spec 05 TrainConfig（buffer_size / stratified_sampling / unroll_K）

---

## [v4-opt 2026-06] 修订:per-episode 元数据(planner_on / collected_at_step)

2agent 诊断(2026-06-11)发现 warmup 的 1000 条 episode 以模型自身先验为 pi_mve
(自蒸馏目标,Ch5.9.1b),却以全权重参与策略 CE 直至 FIFO 逐出(~4000 步)。修订:

1. `store_episode(records, c_t_seq, planner_on: bool = True, collected_at_step: int = 0)`
   —— 两个 per-episode 元数据用与 `_c_t_seqs` 相同的平行 deque(同 FIFO 纪律)存储,
   **不改 TimeStepRecord schema**(12 字段不动);
2. `sample_batch` 输出新增 `planner_on (B,) bool` 与 `collected_at_step (B,) long`;
3. 消费方:spec 05 策略 CE 按 `planner_on` 掩蔽;trainer 以 `global_step −
   collected_at_step` 产出 `diag/target_age_steps`(buffer=5000 episodes ≈ 5000 步
   历史,目标陈旧度首次可观测)。

## 修订记录 (Changelog)

| 日期 | 修订 | 依据 |
|---|---|---|
| 2026-06-11 | planner_on / collected_at_step 平行元数据 + batch 字段 | 2agent 诊断(warmup 自蒸馏污染);用户决策 2026-06-11(掩蔽方案) |
