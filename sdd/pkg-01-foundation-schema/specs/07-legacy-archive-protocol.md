# Spec 07: v4.7 Legacy Archive Protocol

> 父文档：[`../proposal.md`](../proposal.md) §2.4 · [`../design.md`](../design.md) §3 D4

---

## 1. Purpose

定义 v4.7 代码归档到 `hyper_mve/_legacy_v4_7/` 的**完整操作协议**——保留 v4.7 工作版本的可复现性（论文 Ch6.12 补充材料 + Decision Point 1 失败时回滚预案）。

**关键原则**：
1. 归档**保留子目录结构**（envs/multiagent/models/等）
2. 归档后**仍可运行**：`python hyper_mve/_legacy_v4_7/scripts/test_env.py` 不报错
3. **明确触发 DeprecationWarning** 防止 v4 代码意外导入
4. **git tag v4.7-final** 锁定，提供单命令回滚路径

---

## 2. Interface

### 2.1 目录结构（归档后）

```
hyper_mve/
├── _legacy_v4_7/                     # ← 新增归档根目录
│   ├── __init__.py                   # 触发 DeprecationWarning
│   ├── README.md                     # 归档说明 + 命令对照 + 恢复指南
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── non_stationary_tag.py     # 从 hyper_mve/envs/ 移入
│   │   ├── ns_environment.py
│   │   └── make_env.py
│   ├── multiagent/                   # 整目录 (7 文件)
│   │   ├── __init__.py
│   │   ├── core.py
│   │   ├── environment.py
│   │   ├── multi_discrete.py
│   │   ├── rendering.py
│   │   └── scenario.py
│   ├── models/                       # v4.7 业务模型
│   │   ├── __init__.py
│   │   ├── baseline_model.py
│   │   ├── hyper_muzero_model.py
│   │   ├── infer_muzero_model.py
│   │   ├── context_encoder.py
│   │   └── gru_context_encoder.py
│   ├── models_advanced/              # 整目录 (Chunked variants)
│   │   ├── __init__.py
│   │   ├── oracle_v2.py
│   │   ├── infer_v2.py
│   │   └── chunked_hyper_network.py
│   ├── scripts/                      # v4.7 训练 + 测试脚本
│   │   ├── __init__.py
│   │   ├── train_baseline.py
│   │   ├── train_oracle.py
│   │   ├── train_oracle_v2.py
│   │   ├── train_infer.py
│   │   ├── train_infer_v2.py
│   │   ├── test_env.py
│   │   └── test_discrete_env.py
│   ├── training/                     # v4.7 训练遗留 (buffer.py v3, mve.py stub)
│   │   ├── buffer.py
│   │   └── mve.py
│   ├── config.py                     # v4.7 BaseConfig (副本; 主路径 config.py 替换为 shim)
│   └── DESIGN_DOC_FINAL.html         # 文档副本
│
├── envs/                             # ← v4 新空目录 (Pkg-02 填充)
├── models/                           # ← v4 保留: functional_nets.py, hyper_network.py 等
│   ├── functional_nets.py            # 保留 (Pkg-04 改)
│   ├── hyper_network.py              # 保留 (Pkg-04 重写)
│   ├── representation_net.py         # 保留 (Pkg-04 微调)
│   └── interfaces.py                 # 保留 (评估后保留或删除)
├── planning/
│   └── mve_planner.py                # 保留 (Pkg-05 适配接口)
├── training/
│   ├── muzero_trainer.py             # 保留 (Pkg-05 重写)
│   ├── episode_buffer.py             # 保留 (Pkg-05 扩展)
│   └── worker.py                     # 保留 (Pkg-05 适配)
├── utils/                            # 保留
│   ├── utils.py
│   └── evaluator.py
├── schemas/                          # ← Pkg-01 新增
├── configs/                          # ← Pkg-01 新增
├── config.py                         # ← backward-compat shim (本包替换)
└── DESIGN_DOC_FINAL.md               # 保留原位 (v4.7 spec, v4 doc 仍引用)
```

### 2.2 `_legacy_v4_7/__init__.py`

```python
"""v4.7 archive (frozen at git tag v4.7-final).

This module is deprecated. v4 code should NOT import from here.
For v4.7 reproducibility, use `git checkout v4.7-final` to restore
the original working tree.
"""
import warnings

warnings.warn(
    "Importing from hyper_mve._legacy_v4_7 is deprecated. "
    "v4.7 code is archived for paper Ch6.12 reproducibility and rollback only. "
    "Use git checkout v4.7-final for full v4.7 working tree.",
    DeprecationWarning,
    stacklevel=2,
)

# Add archive root to sys.path so v4.7 internal imports work
import sys
import os
_ARCHIVE_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ARCHIVE_ROOT not in sys.path:
    sys.path.insert(0, _ARCHIVE_ROOT)
```

### 2.3 `_legacy_v4_7/README.md`

```markdown
# v4.7 Legacy Archive

This directory contains the complete v4.7 working tree, archived at git tag `v4.7-final`.

## Why archived

v4 refactor introduces breaking changes (new environment, new model architecture, new training loop).
v4.7 is preserved for:

1. **Paper Ch6.12 reproducibility** — failure case analysis comparisons
2. **Rollback path** — if Decision Point 1 (Week 7, Roadmap §6.3) fails Assertion A
3. **Cross-version benchmarks** — supplementary material may include v4.7 vs v4 comparison

## Restoring v4.7 working tree

Two options:

**Option A: Direct file access (read-only)**
```bash
# Files are at hyper_mve/_legacy_v4_7/<module>/<file>.py
# Run v4.7 scripts via:
python -m hyper_mve._legacy_v4_7.scripts.train_baseline
```

**Option B: Full git restore**
```bash
git checkout v4.7-final          # detach to v4.7 commit
python hyper_mve/scripts/train_baseline.py   # original paths work
git checkout main                # return to v4 work
```

## Import path changes (v4.7 → v4)

| v4.7 | v4 (archived) |
|------|---------------|
| `from hyper_mve.envs.non_stationary_tag import NonStationaryTag` | `from hyper_mve._legacy_v4_7.envs.non_stationary_tag import NonStationaryTag` |
| `from hyper_mve.models.hyper_muzero_model import OracleHyperMuZeroModel` | `from hyper_mve._legacy_v4_7.models.hyper_muzero_model import OracleHyperMuZeroModel` |
| `python hyper_mve/scripts/train_oracle.py` | `python hyper_mve/_legacy_v4_7/scripts/train_oracle.py` |
| `from hyper_mve.config import BaseConfig` | `from hyper_mve._legacy_v4_7.config import BaseConfig` |

## DeprecationWarning

Every import from `_legacy_v4_7` triggers `DeprecationWarning`.
To suppress for legitimate use (e.g., running v4.7 in supplementary material):

```python
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from hyper_mve._legacy_v4_7.envs.non_stationary_tag import NonStationaryTag
```

## v4.7 ↔ v4 主要差异

| 项目 | v4.7 | v4 |
|------|------|-----|
| 环境 | NonStationaryTag (hunt/guard) | ResourceCommons (公地资源) |
| 偏好结构 | 派系重对齐 (rule∈[0,1]) | α/β + φ(c) Fehr-Schmidt |
| 上下文输入 | (rule, agent_id) 2 路 | (c_ctx, role, belief) 3 路 |
| 信念推断 | 仅 Infer 实验有 GRU | BeliefNet (GRU + head_c + head_opp) 全局 |
| 模型 | Oracle/Infer 两个分立 | 统一 HyperMuZeroModel |
| 训练 | 单阶段 | 课程学习 Oracle→Anneal→Pure 三阶段 |
| Buffer 字段 | 6 项 | 10 项 (+Δ +τ +cap +ẑ) |

## Audit

```bash
python scripts/audit_legacy.py
# 期望输出: "ALL FILES ACCOUNTED FOR: 20 archived, 8 in main path"
```

## 文件清单 (~20 个)

详见 [proposal.md §2.4](../sdd/pkg-01-foundation-schema/proposal.md).
```

### 2.4 `scripts/audit_legacy.py`

```python
"""审计 v4.7 → _legacy_v4_7/ 归档完整性 (Pkg-01 acceptance test).

期望输出: ALL FILES ACCOUNTED FOR: X archived, Y in main path
失败输出: MISSING FILES: [...] (with diagnosis)
"""
import sys
from pathlib import Path

# v4.7 应归档的文件清单 (相对 hyper_mve/_legacy_v4_7/)
ARCHIVED_FILES = [
    "envs/non_stationary_tag.py",
    "envs/ns_environment.py",
    "envs/make_env.py",
    "envs/__init__.py",
    "multiagent/__init__.py",
    "multiagent/core.py",
    "multiagent/environment.py",
    "multiagent/multi_discrete.py",
    "multiagent/rendering.py",
    "multiagent/scenario.py",
    "models/baseline_model.py",
    "models/hyper_muzero_model.py",
    "models/infer_muzero_model.py",
    "models/context_encoder.py",
    "models/gru_context_encoder.py",
    "models_advanced/__init__.py",
    "models_advanced/oracle_v2.py",
    "models_advanced/infer_v2.py",
    "models_advanced/chunked_hyper_network.py",
    "scripts/train_baseline.py",
    "scripts/train_oracle.py",
    "scripts/train_oracle_v2.py",
    "scripts/train_infer.py",
    "scripts/train_infer_v2.py",
    "scripts/test_env.py",
    "scripts/test_discrete_env.py",
    "training/buffer.py",
    "training/mve.py",
]

# 主路径应保留的文件 (Pkg-04/05 会重写但此刻仍存在)
MAIN_PATH_PRESERVED = [
    "models/functional_nets.py",
    "models/hyper_network.py",
    "models/representation_net.py",
    "models/interfaces.py",
    "planning/mve_planner.py",
    "planning/__init__.py",
    "training/muzero_trainer.py",
    "training/episode_buffer.py",
    "training/worker.py",
    "training/__init__.py",
    "utils/utils.py",
    "utils/evaluator.py",
    "utils/__init__.py",
]


def main():
    project_root = Path(__file__).parent.parent / "hyper_mve"
    archive_root = project_root / "_legacy_v4_7"
    
    missing_archived = []
    for f in ARCHIVED_FILES:
        if not (archive_root / f).exists():
            missing_archived.append(f)
    
    missing_main = []
    for f in MAIN_PATH_PRESERVED:
        if not (project_root / f).exists():
            missing_main.append(f)
    
    # 检查 _legacy_v4_7 必须存在的辅助文件
    aux_required = ["__init__.py", "README.md"]
    missing_aux = [f for f in aux_required if not (archive_root / f).exists()]
    
    # 检查不该在主路径但还在的文件 (forgot to archive)
    leaked_in_main = []
    for f in ARCHIVED_FILES:
        if (project_root / f).exists():
            leaked_in_main.append(f)
    
    if missing_archived or missing_main or missing_aux or leaked_in_main:
        print("ARCHIVE AUDIT FAILED:")
        if missing_archived:
            print(f"  Missing from archive: {missing_archived}")
        if missing_main:
            print(f"  Missing from main path: {missing_main}")
        if missing_aux:
            print(f"  Missing aux files in archive: {missing_aux}")
        if leaked_in_main:
            print(f"  Leaked (still in main, should be archived): {leaked_in_main}")
        sys.exit(1)
    
    print(f"ALL FILES ACCOUNTED FOR: {len(ARCHIVED_FILES)} archived, "
          f"{len(MAIN_PATH_PRESERVED)} in main path")
    sys.exit(0)


if __name__ == "__main__":
    main()
```

### 2.5 git 操作序列

```bash
# Step 1: 确保 working tree 干净
git status   # 应无未提交改动 (或 stash)
git diff --quiet || echo "WARNING: uncommitted changes"

# Step 2: 锁定 v4.7-final tag (在归档前)
git add -A
git commit -m "Pkg-01 prep: snapshot v4.7 working tree before refactor

This is the final v4.7 commit. Subsequent commits will:
1. Archive v4.7 files to hyper_mve/_legacy_v4_7/
2. Introduce v4 schemas and config restructure
3. Begin Pkg-02..08 v4 refactor

To restore v4.7 working tree:
    git checkout v4.7-final
"
git tag -a v4.7-final -m "v4.7 final working version (pre-v4 refactor)"

# Step 3: 创建 Pkg-01 分支
git checkout -b pkg-01/foundation-schema

# Step 4: 执行归档 (git mv 保留 history)
mkdir -p hyper_mve/_legacy_v4_7
git mv hyper_mve/envs/non_stationary_tag.py hyper_mve/_legacy_v4_7/envs/non_stationary_tag.py
# ... 重复对所有 ARCHIVED_FILES
# (脚本化, 见 scripts/perform_archive.sh)

# Step 5: 添加归档辅助文件
# 写 hyper_mve/_legacy_v4_7/__init__.py (DeprecationWarning)
# 写 hyper_mve/_legacy_v4_7/README.md
# 写 hyper_mve/_legacy_v4_7/<each subdir>/__init__.py

# Step 6: 替换主路径 config.py 为 shim
# (内容见 05-v4-config-structure.md §3.5)

# Step 7: 新增 v4 schemas/ + configs/
# (本包后续步骤)

# Step 8: 提交 + 推送 + 推送 tag
git add -A
git commit -m "Pkg-01: archive v4.7 + add v4 schemas/configs"
git push origin pkg-01/foundation-schema
git push origin v4.7-final
```

### 2.6 `scripts/perform_archive.sh`（自动化辅助）

```bash
#!/usr/bin/env bash
# perform_archive.sh - 自动化 v4.7 归档 (从 ARCHIVED_FILES 列表生成 git mv)
set -euo pipefail

ARCHIVED_FILES=(
    "envs/non_stationary_tag.py"
    "envs/ns_environment.py"
    "envs/make_env.py"
    # ... (与 audit_legacy.py 列表同步)
)

for f in "${ARCHIVED_FILES[@]}"; do
    src="hyper_mve/$f"
    dst="hyper_mve/_legacy_v4_7/$f"
    
    if [[ ! -f "$src" ]]; then
        echo "WARN: source $src does not exist, skipping"
        continue
    fi
    
    mkdir -p "$(dirname "$dst")"
    git mv "$src" "$dst"
    echo "Archived: $src -> $dst"
done

# 创建 _legacy_v4_7/__init__.py 链
find hyper_mve/_legacy_v4_7 -type d | while read dir; do
    if [[ ! -f "$dir/__init__.py" ]]; then
        touch "$dir/__init__.py"
    fi
done

echo "Archive complete. Run: python scripts/audit_legacy.py"
```

PowerShell 版本（Windows 主开发环境）：

```powershell
# perform_archive.ps1
$ARCHIVED = @(
    "envs/non_stationary_tag.py",
    "envs/ns_environment.py",
    # ... 完整列表
)

foreach ($f in $ARCHIVED) {
    $src = "hyper_mve/$f"
    $dst = "hyper_mve/_legacy_v4_7/$f"
    
    if (-not (Test-Path $src)) {
        Write-Warning "Source $src does not exist, skipping"
        continue
    }
    
    $dstDir = Split-Path $dst -Parent
    if (-not (Test-Path $dstDir)) {
        New-Item -ItemType Directory -Force $dstDir | Out-Null
    }
    
    git mv $src $dst
    Write-Host "Archived: $src -> $dst"
}

# 创建子目录 __init__.py
Get-ChildItem hyper_mve/_legacy_v4_7 -Recurse -Directory | ForEach-Object {
    $init = Join-Path $_.FullName "__init__.py"
    if (-not (Test-Path $init)) {
        New-Item -ItemType File $init | Out-Null
    }
}

Write-Host "Archive complete. Run: python scripts/audit_legacy.py"
```

---

## 3. Implementation Notes

### 3.1 `git mv` 而非 `mv`

`git mv` 保留文件历史（git blame / log 可追溯到归档前的修改）。直接 `mv` + `git rm + git add` 会失去历史关联（除非 git 检测到 rename，但大批量重命名时不稳定）。

### 3.2 v4.7 内部 import 路径如何工作

v4.7 文件相对 import 形如 `from envs.non_stationary_tag import NonStationaryTag`。归档后这种 import 在 `_legacy_v4_7/__init__.py` 中通过 `sys.path.insert(0, _ARCHIVE_ROOT)` 修复——`_ARCHIVE_ROOT = hyper_mve/_legacy_v4_7`，加入 sys.path 后 `from envs.non_stationary_tag` 自然找到 `_legacy_v4_7/envs/non_stationary_tag.py`。

### 3.3 v4.7 `config.py` 归档 vs shim

- v4.7 `BaseConfig` 类移到 `_legacy_v4_7/config.py`（原始副本）
- 主路径 `hyper_mve/config.py` 替换为 shim（见 05-v4-config-structure.md §3.5）
- v4.7 脚本通过 `_legacy_v4_7/scripts/train_*.py` 显式 import `from hyper_mve._legacy_v4_7.config import BaseConfig`，**不走 shim**

### 3.4 `DESIGN_DOC_FINAL.md` 保留主路径

虽然其内容是 v4.7 spec，但：
- v4 路线图（`D:\RL\docs\`）仍 cross-reference 此文档
- 用户可能在主路径直接阅读
- 移动会破坏当前的 markdown link
- **决策**：保留原位，归档仅复制 `.html` 版本（避免 markdown 引用断裂）

### 3.5 验证归档不破坏 v4.7

```bash
# 归档完成后必跑此测试
python hyper_mve/_legacy_v4_7/scripts/test_env.py
# 期望: 10 random episode 全部通过, 无 ImportError
```

如果失败，常见原因：
- v4.7 脚本中 `sys.path.insert(0, os.path.dirname(os.path.dirname(...)))` 路径计算偏移
- 缓解：脚本入口加 `sys.path.insert(0, str(Path(__file__).parent.parent))` 修正

---

## 4. Edge Cases

| 场景 | 处理 |
|------|------|
| 用户在归档过程中 Ctrl-C | git mv 是原子操作；中途中断时部分文件已移动，重跑 `perform_archive.sh` 会跳过已移动文件（"source does not exist"） |
| 主路径有未追踪文件（如 `*.pyc`） | git mv 不处理；建议归档前 `git clean -fdx` 清理或忽略 |
| v4.7 训练脚本调用 `import torch; torch.load("checkpoints/...")` 路径偏移 | v4.7 checkpoint 路径相对 `cwd`，归档后用户在 `D:\RL\hyper_mve\` 执行 `python -m hyper_mve._legacy_v4_7.scripts.train_baseline`，checkpoint 路径仍正确 |
| 用户希望 v4 与 v4.7 并行训练（同时跑） | 完全可行；v4.7 进程从 `_legacy_v4_7` import，v4 进程从主路径，两者不冲突 |
| Windows 路径分隔符 | git mv 在 Windows 上接受 `/` 或 `\`，PowerShell 脚本统一用 `/` |
| 大批量 git mv 触发性能问题 | 单次 git mv ~毫秒级；20 个文件 < 1 秒 |
| 用户希望删除归档（v4.7 不再需要） | `git rm -r hyper_mve/_legacy_v4_7/ && git commit`；tag 仍保留可恢复 |

---

## 5. Acceptance Criteria

### 5.1 单元测试（`tests/test_legacy_import_warning.py`）

```python
import warnings
import pytest


def test_import_archive_triggers_deprecation():
    """v4 代码意外 import _legacy_v4_7 必须触发 DeprecationWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from hyper_mve._legacy_v4_7 import envs  # noqa
        
        deprecation_warnings = [warning for warning in w 
                                if issubclass(warning.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1
        assert "deprecated" in str(deprecation_warnings[0].message).lower()


def test_archive_init_adds_to_sys_path():
    import sys
    from hyper_mve import _legacy_v4_7  # 触发 sys.path 注入
    
    archive_root = list(_legacy_v4_7.__path__)[0]
    assert archive_root in sys.path


def test_archived_module_importable():
    """归档后 v4.7 模块仍可 import (用于 Ch6.12 复现)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from hyper_mve._legacy_v4_7.envs.non_stationary_tag import NonStationaryTag
        from hyper_mve._legacy_v4_7.models.baseline_model import BaselineModel
    
    assert NonStationaryTag is not None
    assert BaselineModel is not None
```

### 5.2 集成测试

```powershell
# Test 1: audit 通过
python scripts/audit_legacy.py
# 期望 exit code 0, 输出 "ALL FILES ACCOUNTED FOR"

# Test 2: v4.7 test_env.py 仍可跑
python hyper_mve/_legacy_v4_7/scripts/test_env.py
# 期望 10 random episode 通过

# Test 3: git tag 存在
git tag -l | findstr v4.7-final
# 期望输出 "v4.7-final"

# Test 4: git history 保留 (v4.7 文件的 blame 可追溯)
git log --follow hyper_mve/_legacy_v4_7/envs/non_stationary_tag.py
# 期望显示 v4.7 时期的 commit history
```

### 5.3 端到端

```powershell
# 完整归档流程演练
git status   # 干净
git tag v4.7-final
git checkout -b pkg-01/foundation-schema
.\scripts\perform_archive.ps1
python scripts\audit_legacy.py   # PASS
python hyper_mve/_legacy_v4_7/scripts/test_env.py   # PASS
git add -A
git commit -m "Pkg-01: archive v4.7 working tree"
```

---

## 6. Cross-references

- `../proposal.md` §2.4（归档操作清单）
- `../design.md` D4（归档目录命名）
- `05-v4-config-structure.md` §3.5（backward-compat shim）
- Roadmap §2.4 代码基线锁定要求
- Plan File Part 6（前置检查清单）
- 论文 Ch6.12 失败案例分析（消费此归档）
