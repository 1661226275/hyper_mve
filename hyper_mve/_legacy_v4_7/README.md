# v4.7 Legacy Archive

This directory contains the complete v4.7 working tree, archived at git tag
`v4.7-final` as part of Pkg-01 (Foundation Schema & Config Restructure).

## Why archived

v4 refactor introduces breaking changes (new environment, new model
architecture, new training loop). v4.7 is preserved for:

1. **Paper Ch6.12 reproducibility** — failure case analysis comparisons.
2. **Rollback path** — if Decision Point 1 (Week 7, Roadmap §6.3) fails
   Assertion A, the codebase can fall back to v4.7.
3. **Cross-version benchmarks** — supplementary material may include v4.7 vs
   v4 comparisons.

## Restoring the v4.7 working tree

Two options:

**Option A: Direct file access (in-place)**

```powershell
# Run v4.7 scripts via the archived path:
python hyper_mve/_legacy_v4_7/scripts/train_baseline.py
python hyper_mve/_legacy_v4_7/scripts/test_env.py
```

Every such import emits a `DeprecationWarning` on first touch.

**Option B: Full git restore**

```powershell
git checkout v4.7-final          # detach HEAD to the v4.7 snapshot
python hyper_mve/scripts/train_baseline.py   # original paths work
git checkout main                # return to v4 work
```

## Import path changes (v4.7 → v4)

| v4.7 path | v4 (archived) path |
|-----------|--------------------|
| `from hyper_mve.envs.non_stationary_tag import NonStationaryTag` | `from hyper_mve._legacy_v4_7.envs.non_stationary_tag import NonStationaryTag` |
| `from hyper_mve.models.hyper_muzero_model import OracleHyperMuZeroModel` | `from hyper_mve._legacy_v4_7.models.hyper_muzero_model import OracleHyperMuZeroModel` |
| `python hyper_mve/scripts/train_oracle.py` | `python hyper_mve/_legacy_v4_7/scripts/train_oracle.py` |
| `from hyper_mve.config import BaseConfig` | `from hyper_mve._legacy_v4_7.config import BaseConfig` |

The main-path `hyper_mve/config.py` is now a backward-compat shim that
re-exports `V4Config.from_preset("medium")` as `BaseConfig` (with a
`DeprecationWarning`).

## Suppressing the DeprecationWarning

For legitimate use (e.g., reproducing v4.7 numbers for supplementary
material):

```python
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from hyper_mve._legacy_v4_7.envs.non_stationary_tag import NonStationaryTag
```

## v4.7 ↔ v4 main differences

| Aspect | v4.7 | v4 |
|--------|------|-----|
| Environment | NonStationaryTag (hunt/guard) | ResourceCommons (commons resource) |
| Preference structure | Faction re-alignment (rule ∈ [0,1]) | α/β + φ(c) Fehr-Schmidt |
| Context input | (rule, agent_id) — 2 paths | (c_ctx, role, belief) — 3 paths |
| Belief inference | GRU only in Infer-experiment | Global BeliefNet (GRU + head_c + head_opp) |
| Model | Oracle / Infer split | Unified `HyperMuZeroModel` |
| Training | Single stage | Curriculum: Oracle → Anneal → Pure, 3 stages |
| Buffer fields | 6 (o, a, r, π_mve, v, done) | 10 (+Δ, τ, cap, ĉ, ẑ) |

## Audit

Run from the repository root:

```powershell
python scripts/audit_legacy.py
# Expected: "ALL FILES ACCOUNTED FOR: 30 archived, 13 in main path"
```

The exit code is `0` on success and `1` on any missing / leaked file.

## File inventory

See `scripts/audit_legacy.py` for the authoritative list of archived files
and main-path files that must remain.
