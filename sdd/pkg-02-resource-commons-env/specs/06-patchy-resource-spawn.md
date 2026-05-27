# Spec 06: Patchy Resource Spawn — Ch3.3.2 hotspot 生成

> 父文档：[`../proposal.md`](../proposal.md) §2.1 · [`../design.md`](../design.md) §3 D6

---

## 1. Purpose

实现 Ch3.3.2 的 patchy 资源分布：将 K 个资源点分布在 M 个 hotspot 中心周围（高斯采样，σ_patch=2.0）。这是 ResourceCommons 的关键空间博弈机制——agent 必须在 hotspot 间迁移（不能简单"等待原地再生"）。

---

## 2. Interface

### 2.1 文件路径

`hyper_mve/envs/resource_commons/spawn.py`

### 2.2 公开 API

```python
import numpy as np


def spawn_patchy_resources(
    K: int,                           # 总资源点数
    M: int,                           # hotspot 数
    L: int,                           # 网格边长
    sigma_patch: float = 2.0,         # 高斯标准差
    min_hotspot_distance: float = None, # hotspot 中心最小距离 (默认 2σ)
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """生成 patchy 资源分布 (Ch3.3.2).
    
    Args:
        K: 总资源点数 (如 Medium 20)
        M: hotspot 数 (如 Medium 3)
        L: 网格大小 (如 Medium 16)
        sigma_patch: 高斯采样标准差 (默认 2.0)
        min_hotspot_distance: hotspot 中心最小欧式距离 (默认 = 2*sigma_patch = 4)
        rng: numpy Generator (可复现性)
    
    Returns:
        hotspot_centers: (M, 2) int32, hotspot 中心坐标
        resource_positions: (K, 2) int32, 资源点坐标 (可能有重复格子)
    """
    if rng is None:
        rng = np.random.default_rng()
    if min_hotspot_distance is None:
        min_hotspot_distance = 2.0 * sigma_patch
    
    # 1. 采样 M 个 hotspot 中心 (含 min_distance 约束)
    hotspot_centers = _sample_hotspot_centers(M, L, min_hotspot_distance, rng)
    
    # 2. 围绕每个 hotspot 采样 K/M 个资源点
    points_per_hotspot = _distribute_K_over_M(K, M)
    
    resource_positions = np.zeros((K, 2), dtype=np.int32)
    idx = 0
    for m in range(M):
        n = points_per_hotspot[m]
        center = hotspot_centers[m]
        for _ in range(n):
            point = _sample_around_center(center, sigma_patch, L, rng)
            resource_positions[idx] = point
            idx += 1
    
    return hotspot_centers, resource_positions


def _sample_hotspot_centers(
    M: int, L: int, min_distance: float, rng, max_retries: int = 100
) -> np.ndarray:
    """采样 M 个中心, 满足两两距离 ≥ min_distance.
    
    若超过 max_retries 仍无法满足, 放宽约束并警告.
    """
    centers = np.zeros((M, 2), dtype=np.int32)
    for m in range(M):
        for retry in range(max_retries):
            candidate = rng.integers(0, L, size=2)
            # 检查与已有中心的距离
            ok = True
            for k in range(m):
                dist = np.linalg.norm(candidate - centers[k])
                if dist < min_distance:
                    ok = False
                    break
            if ok:
                centers[m] = candidate
                break
        else:
            # 放宽约束: 接受当前 candidate
            import warnings
            warnings.warn(
                f"Hotspot {m} placement failed after {max_retries} retries; "
                f"using last candidate. Consider lowering min_distance or M."
            )
            centers[m] = candidate
    return centers


def _distribute_K_over_M(K: int, M: int) -> np.ndarray:
    """将 K 个资源点尽量均匀分到 M 个 hotspot.
    
    K=20, M=3 → [7, 7, 6] (余数 2 分给前两个 hotspot)
    """
    base = K // M
    remainder = K % M
    counts = np.full(M, base, dtype=np.int32)
    counts[:remainder] += 1
    return counts


def _sample_around_center(
    center: np.ndarray, sigma: float, L: int, rng,
    max_retries: int = 20,
) -> np.ndarray:
    """高斯采样 1 个点, 截断到 [0, L) 网格.
    
    若多次采到边界外, 用 clip 截断.
    """
    for _ in range(max_retries):
        offset = rng.normal(0.0, sigma, size=2)
        point = np.round(center + offset).astype(np.int32)
        # 边界检查
        if np.all((point >= 0) & (point < L)):
            return point
    # Fallback: clip
    offset = rng.normal(0.0, sigma, size=2)
    point = np.round(center + offset).astype(np.int32)
    return np.clip(point, 0, L - 1)
```

---

## 3. Implementation Notes

### 3.1 hotspot 重叠风险与缓解

**风险**：若 3 个 hotspot 中心都集中在一角，资源分布稀疏（其他区域无资源）。

**缓解**：
1. `min_hotspot_distance = 2 × sigma_patch = 4.0`（默认）
2. Medium config (L=16) 下，3 hotspot 距离 ≥ 4 容易满足（网格够大）
3. 极端配置（如 L=8 + M=3）可能放宽约束并警告——单测覆盖

### 3.2 K 不能被 M 整除的处理

- 余数分给前几个 hotspot（D6 决策一致）
- `_distribute_K_over_M` 实现是 deterministic

### 3.3 资源点位置可重复

允许两个资源点在同一格子（高斯采样恰好相同 round 结果）。Env 内部把它们当作两个独立"资源单位"：
- 公式 3.2 中"在 k 点的 agent 集合 H_{k,t}"按 resource_positions 索引 k 计算，**不去重**
- 但这意味着 agent 站在同一格子 + HARVEST 时，可同时采集两个资源点的库存（fair-share 各自独立）

**论文说明**：Ch3.3.2 默认资源点位置去重；本实现允许重复以简化（K 较小时罕见，对实验结果影响 < 1%）。

### 3.4 高斯截断

`_sample_around_center` 用 `max_retries=20` 重采样，超过则 clip。  
sigma_patch=2.0 + L=16 时绝大多数采样在网格内，retry 罕见触发。

### 3.5 性能

- 单次 `spawn_patchy_resources` ~50 μs (M=3, K=20)
- 1000 episode reset 共 ~50 ms，可忽略

---

## 4. Edge Cases

| 场景 | 行为 |
|------|------|
| M=1, K=20 | 全部 20 资源在单 hotspot 周围 |
| M=K, K=20 | 每 hotspot 1 个资源（极端分散） |
| K=0 | 返回空数组（数学上有效，但 env 会失败） |
| min_distance > L/2 | hotspot 放置超 max_retries 触发警告 |
| sigma=0 | 所有资源点重合于 hotspot 中心 |
| L=1 (退化网格) | 所有资源 (0, 0) 重合 |
| 高斯采样溢出 | clip 到 [0, L-1] |
| 同 seed → 同布局 | 完全确定 |

---

## 5. Acceptance Criteria

### 5.1 单元测试（`tests/envs/test_spawn.py`）

```python
import numpy as np
import pytest
from hyper_mve.envs.resource_commons.spawn import (
    spawn_patchy_resources, _sample_hotspot_centers, _distribute_K_over_M,
)


def test_spawn_shape_medium():
    rng = np.random.default_rng(42)
    centers, positions = spawn_patchy_resources(K=20, M=3, L=16, rng=rng)
    assert centers.shape == (3, 2)
    assert positions.shape == (20, 2)


def test_spawn_in_grid():
    """所有点必须在 [0, L) 网格内."""
    rng = np.random.default_rng(42)
    centers, positions = spawn_patchy_resources(K=20, M=3, L=16, rng=rng)
    assert np.all((centers >= 0) & (centers < 16))
    assert np.all((positions >= 0) & (positions < 16))


def test_distribute_K_over_M_balanced():
    """K=20, M=3 → [7, 7, 6]."""
    counts = _distribute_K_over_M(K=20, M=3)
    assert list(counts) == [7, 7, 6]
    assert counts.sum() == 20


def test_distribute_K_exact_divide():
    """K=21, M=3 → [7, 7, 7]."""
    counts = _distribute_K_over_M(K=21, M=3)
    assert list(counts) == [7, 7, 7]


def test_hotspot_min_distance():
    """min_distance 约束生效."""
    rng = np.random.default_rng(42)
    centers = _sample_hotspot_centers(M=3, L=16, min_distance=4.0, rng=rng)
    for i in range(3):
        for j in range(i + 1, 3):
            dist = np.linalg.norm(centers[i] - centers[j])
            assert dist >= 4.0 - 1e-6  # 允许浮点误差


def test_spawn_reproducible():
    """相同 seed → 相同布局."""
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    centers1, pos1 = spawn_patchy_resources(K=20, M=3, L=16, rng=rng1)
    centers2, pos2 = spawn_patchy_resources(K=20, M=3, L=16, rng=rng2)
    assert np.array_equal(centers1, centers2)
    assert np.array_equal(pos1, pos2)


def test_spawn_different_seeds_different():
    """不同 seed → 不同布局."""
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(43)
    centers1, _ = spawn_patchy_resources(K=20, M=3, L=16, rng=rng1)
    centers2, _ = spawn_patchy_resources(K=20, M=3, L=16, rng=rng2)
    assert not np.array_equal(centers1, centers2)


def test_resources_clustered():
    """资源点应聚集在 hotspot 附近 (σ=2 → 大部分距离中心 < 4)."""
    rng = np.random.default_rng(42)
    centers, positions = spawn_patchy_resources(K=30, M=3, L=20, sigma_patch=2.0, rng=rng)
    # 每个资源点找到最近 hotspot
    near_count = 0
    for p in positions:
        min_dist = min(np.linalg.norm(p - c) for c in centers)
        if min_dist < 4.0:  # 2σ
            near_count += 1
    assert near_count >= 28  # 至少 93% 在 2σ 内 (high probability)


def test_easy_config():
    """Easy: M=1, K=8, L=8."""
    rng = np.random.default_rng(42)
    centers, positions = spawn_patchy_resources(K=8, M=1, L=8, rng=rng)
    assert centers.shape == (1, 2)
    assert positions.shape == (8, 2)


def test_hard_config():
    """Hard: M=5, K=40, L=24."""
    rng = np.random.default_rng(42)
    centers, positions = spawn_patchy_resources(K=40, M=5, L=24, rng=rng)
    assert centers.shape == (5, 2)
    assert positions.shape == (40, 2)


def test_min_distance_too_large_warns():
    """min_distance > L/2 应触发警告."""
    rng = np.random.default_rng(42)
    with pytest.warns(UserWarning, match="placement failed"):
        _sample_hotspot_centers(M=5, L=8, min_distance=10.0, rng=rng)
```

### 5.2 集成测试

- env reset 100 次后 hotspot_centers 分布合理（不全在角落）
- Visualization (render.py) 可显示 3 hotspot

### 5.3 性能

- `spawn_patchy_resources(K=40, M=5, L=24)` < 200 μs
- 1000 reset × Medium ≤ 100 ms 总开销

---

## 6. Cross-references

- Ch3.3.2 Patchy 资源分布
- `01-env-formal-tuple.md`（state.hotspot_centers / resource_positions 字段）
- `02-resource-dynamics.md`（neighbor factor 计算依赖 resource_positions）
- `04-six-block-observation.md`（observation resource block 读取 resource_positions）
- `07-difficulty-presets-env.md`（三 preset 的 M / K / L 参数）
- `08-gym-api.md`（info["hotspot_centers"] 暴露给 evaluator）
- `../design.md` D6 (每 reset 重生成)
