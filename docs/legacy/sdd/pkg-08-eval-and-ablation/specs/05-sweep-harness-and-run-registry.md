# Spec 05 — Sweep Harness + Run Registry (`experiments/sweep.py` + `runs/registry.jsonl`)

> **Parent docs**: [`../proposal.md`](../proposal.md) §2.1.5 · [`../design.md`](../design.md) §3.4 / §3.5 / §4 D8 · [`../README.md`](../README.md) C8-ABL-SWEEP1 / C8-ABL-ISO1 / C8-ABL-REG1
> **Anchors**: this spec is the **spine of the Ablation half** of pkg-08. The SweepConfig cartesian + JSONL RunRegistry defined here is the surface that spec 06 (`--ablation` CLI canned YAMLs) dispatches into, that spec 07 (`stats.py` Welch t / `compare.py`) reads back out of, and that spec 08 (integration contracts) cross-locks against pkg-07 spec 01 §2 (REGISTRY 11 keys), pkg-07 spec 04 (N-parametric adapter `env_fn`), and pkg-07 spec 08 (`evaluate(env_fn, c_grid, episodes) -> EvalReport` signature).
> **Status**: SDD only — describes the contract for `hyper_mve/experiments/sweep.py` + `hyper_mve/experiments/run_registry.py` + `hyper_mve/experiments/_sweep_worker.py`, not the implementation.

---

## ⚠️ Header — three hard locks

### Lock 1 — Subprocess-per-row is non-negotiable (C8-ABL-ISO1)

Each sweep row spawns its own Python interpreter via `subprocess.Popen([sys.executable, "-m", "hyper_mve.experiments._sweep_worker"], ...)`. There is no in-process row loop, no `multiprocessing.Pool`, no `concurrent.futures.ProcessPoolExecutor` worker reuse. The five rationale points are itemised in §6 (CUDA OOM / segfault isolation, GPU driver-level cleanup, V4Config dataclass independence, per-GPU semaphore scheduling, graceful Ctrl-C). A named test `test_sweep_subprocess_isolation.py` (§9) injects a row that calls `os._exit(139)` (simulated SIGSEGV) and asserts the parent sweep continues, the registry records `status="failed"` with `failure_reason="exit_code=139"`, and the next row's subprocess starts on a clean CUDA context.

### Lock 2 — RunRegistry is JSONL append-only (C8-ABL-REG1)

`runs/registry.jsonl` is **never** rewritten, mutated, or truncated by the sweep harness. Every state transition (pending → running → completed/failed/skipped) is a new line appended to the file. Restarts append a new row with a fresh `run_id` (uuid4) but the same `(variant, seed, config_hash)` dedup tuple. Concurrent writers (the parent for `pending` rows + each child subprocess for `running`/`completed`/`failed` rows) coordinate via an OS-level file lock: `msvcrt.locking(LK_LOCK)` on Windows (the project's primary platform per `CLAUDE.md`), `fcntl.flock(LOCK_EX)` on POSIX. A named test `test_run_registry_concurrent_safe.py` (§9) spawns 16 threads each appending 100 rows and asserts the final line count is exactly 1 600 with no partial lines, no JSON parse errors, no interleaved bytes.

### Lock 3 — Schema mirror against spec 08 §4 is byte-identical

The RunRegistry row schema in §4 of this spec is mirrored verbatim in spec 08 §4 as the **integration contract lock**. Any addition, removal, or type change to a row field requires a synchronous edit of both §4 of this spec and §4 of spec 08. A drift detector test `test_run_registry_row_schema_matches_spec_08.py` (§9) parses both spec files for the `# === ... ===` field-group block fences and asserts the JSON-key sets are equal. **The 23-key count (22 schema-domain + 1 schema_version) is also asserted** (so a silent addition that happens to keep the JSON keys aligned still trips the count guard).

---

## 1. Purpose

The sweep harness + RunRegistry exist to convert single-run `EvalReport`s (spec 01) into the artefacts the paper Ch6 needs: the main table (hyper + 5 internal + 3 Tier-1 + MAMBA-if-sourced × 5 seeds × 2 presets), the four ablation tables (Abl1 gen_scope 7-cell / Abl4 CRN × Joint-CoordDesc / Abl6 Fehr-Schmidt 3×3 / Abl7 curriculum), the zero-shot table, and the comparison plots. Every downstream pkg-08 spec writes its results into this surface:

1. **Spec 02 (zero-shot + c_hidden + regret)** writes per-(variant, seed, c) `EvalReport`s with `regret_per_c` populated; the registry is the persistence path.
2. **Spec 03 (four-mode planner dispatch)** writes per-(variant, seed, eval_planner_mode) `EvalReport`s with the 4-mode taxonomy; the sweep harness enumerates the mode axis.
3. **Spec 04 (μP self-check 18-run)** is a fixed-shape sweep emitted via SweepConfig with no overrides (2 widths × 3 LRs × 3 seeds = 18 rows). The registry is the persistence path.
4. **Spec 06 (`--ablation` CLI)** is a thin dispatcher: each `--ablation <id>` reads a canned YAML at `experiments/ablations/<id>.yaml`, materialises a SweepConfig, and calls `run_sweep(sweep_cfg)`.
5. **Spec 07 (`stats.py` Welch t + `compare.py` markdown / bar+errorbar plot)** reads `runs/registry.jsonl` (via `pandas.read_json(..., lines=True)`), groups by `(variant, seed, ablation_cell)`, and reads `eval_report_path` to materialise raw per-episode returns for Welch t.

The sweep harness is therefore **the unique persistence boundary** between "single run produces EvalReport" (spec 01) and "Ch6 tables / plots / Welch t" (specs 07, 08). It is **the** spine.

The deliverable is three files (no others added in this spec):
- `hyper_mve/experiments/sweep.py` — `SweepConfig` dataclass + `run_sweep(SweepConfig) -> None` + cartesian enumerator + subprocess scheduler.
- `hyper_mve/experiments/run_registry.py` — `RunRegistry` JSONL append-only class + OS file-lock wrapper.
- `hyper_mve/experiments/_sweep_worker.py` — module-level `__main__` invoked as subprocess; reads one row spec via stdin JSON, calls `train_main.main(argv)`, writes `EvalReport` to disk, exits 0/non-0.

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│ SweepConfig YAML                                                     │
│   variants: [hyper, baseline_input_wide, external_mappo, ...]        │
│   seeds:    [0, 1, 2, 3, 4]                                          │
│   preset:   medium                                                   │
│   overrides: [{train.lr: 3e-4}, {env.N: 8}]                          │
│   max_steps: 200000                                                  │
│   eval_planner_mode: planner_full                                    │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼   (§3 enumerate_cartesian)
┌──────────────────────────────────────────────────────────────────────┐
│ rows: list[SweepRow]                                                 │
│   = variants × seeds × overrides                                     │
│   (cardinality = |V| · |S| · |O|, dedup'd by (variant,seed,hash))    │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼   (§5 SubprocessRunner)
┌──────────────────────────────────────────────────────────────────────┐
│ for row in rows:                                                     │
│     gpu_id = sem.acquire()              # §7 per-GPU semaphore       │
│     registry.append_pending(row, gpu_id)                             │
│     proc = subprocess.Popen([sys.executable, "-m",                   │
│         "hyper_mve.experiments._sweep_worker"],                      │
│         stdin=PIPE, env={CUDA_VISIBLE_DEVICES: str(gpu_id), ...})    │
│     proc.stdin.write(json.dumps(row_payload).encode())               │
│     proc.communicate()                                               │
│     registry.append_completed_or_failed(row, proc.returncode,        │
│         eval_report_path=row.eval_report_path)                       │
│     sem.release(gpu_id)                                              │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼   (§4 RunRegistry, JSONL append-only)
┌──────────────────────────────────────────────────────────────────────┐
│ runs/registry.jsonl  ←  one line per state transition               │
│ runs/<run_tag>/                                                      │
│   ├── config.yaml             (resolved V4Config snapshot)           │
│   ├── eval_report.json        (EvalReport JSON via asdict)           │
│   ├── eval_episodes.csv       (raw per-episode returns for Welch t)  │
│   ├── ckpt_final.pt           (final checkpoint)                     │
│   └── tb/                     (TensorBoard event files)              │
└──────────────────────────────────────────────────────────────────────┘
```

The harness is **a scheduler, not a trainer**. All training happens inside `_sweep_worker.py` which delegates to `train_main.main(argv)` (the existing entry point at `hyper_mve/scripts/train_main.py`); see §5.3 for the worker contract.

---

## 3. SweepConfig YAML schema

### 3.1 Dataclass

```python
# hyper_mve/experiments/sweep.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping


@dataclass(frozen=True)
class SweepConfig:
    """Single source of truth for one sweep invocation. Materialised either from
    a YAML at hyper_mve/experiments/ablations/<id>.yaml (spec 06 dispatcher) or
    constructed in-process by callers (μP self-check spec 04).
    """

    # === Cartesian axes (3) ===
    variants: tuple[str, ...]                       # subset of REGISTRY ∪ {hyper, oracle_only, infer_only}; pkg-07 spec 01 §2 14-row CLI strings
    seeds: tuple[int, ...]                          # >= 5 for Ch6 statistical protocol; <5 emits [WARN] (C8-ABL-STAT1 stays non-blocking)
    overrides: tuple[Mapping[str, object], ...]     # list of cfg override dicts; cartesian-multiplied with variants × seeds; empty tuple → single empty override

    # === Single-valued scalars (3) ===
    preset: Literal["easy", "medium", "hard"]       # V4Config.from_preset entry; restricted to the 3 paper presets (the 8 *_film_lora / *_base_lora presets are not in scope for Ch6 tables)
    max_steps: int                                  # train.max_train_steps; overrides preset default
    eval_planner_mode: Literal[                     # cfg.eval.eval_planner_mode (spec 03 D7); default "planner_full"
        "direct_inference",
        "planner_no_crn",
        "planner_no_coord_desc",
        "planner_full",
    ] = "planner_full"

    # === Sweep-level resource hints (3) ===
    max_parallel: int = 1                           # number of concurrent subprocesses; clamped to n_visible_gpus at run_sweep entry
    n_gpus: int = 1                                 # number of GPUs the harness may consume; subprocesses pinned via CUDA_VISIBLE_DEVICES
    ablation_cell_id: str | None = None             # parent ablation id (e.g. "abl4_crn_joint"); written into RunRegistry.ablation_cell for every row; None for non-ablation sweeps

    # === Schema sentinel (1) ===
    schema_version: str = "pkg08-spec05-v1"         # bumped only by synchronous spec 05 + spec 08 §4 edit
```

### 3.2 YAML serialisation

`SweepConfig` round-trips with `pyyaml`:

```yaml
# hyper_mve/experiments/ablations/abl4_crn_joint.yaml
variants:
  - hyper
seeds: [0, 1, 2, 3, 4]
preset: medium
overrides:
  - {train.use_crn: true,  train.randomize_order: true}
  - {train.use_crn: false, train.randomize_order: true}
  - {train.use_crn: true,  train.randomize_order: false}
  - {train.use_crn: false, train.randomize_order: false}
max_steps: 200000
eval_planner_mode: planner_full
max_parallel: 2
n_gpus: 2
ablation_cell_id: abl4_crn_joint
```

Loader: `SweepConfig.from_yaml(path: pathlib.Path) -> SweepConfig` uses `yaml.safe_load` + `dataclasses` field-by-field cast (tuples coerced from lists, `Mapping[str, object]` left as plain dict). Unknown YAML keys raise `ValueError` (strict-mode load — no silent ignore).

### 3.3 Cartesian semantics (C8-ABL-SWEEP1)

`enumerate_cartesian(sweep: SweepConfig) -> list[SweepRow]` produces the deterministic cartesian product:

```
rows = [
  SweepRow(variant=v, seed=s, overrides=o, sweep_row_index=i, ablation_cell=sweep.ablation_cell_id)
  for i, (v, s, o) in enumerate(
      itertools.product(sweep.variants, sweep.seeds, sweep.overrides)
  )
]
```

Cardinality: `|V| · |S| · |O|`. The order is **lexicographic over (variant, seed, override-index)** — stable across runs, so resume semantics (§8) work.

**Empty-overrides edge case**: when `sweep.overrides == ()` (the tuple is empty, not `({},)`), the harness substitutes `overrides=({},)` (single empty dict) so the cardinality is `|V| · |S| · 1`. A "bare baseline run" is therefore `variants=("baseline_input_wide",), seeds=(0,1,2,3,4), overrides=()` → 5 rows with no cfg overrides, all using the preset default. Test `test_sweep_cartesian_correct.py` (§9) parametrises `(|V|, |S|, |O|) ∈ {(1,1,0), (1,5,0), (3,5,4), (14,5,0)}` and asserts `len(rows) == max(1, |V|·|S|·max(1,|O|))`.

**SweepRow dataclass**:

```python
@dataclass(frozen=True)
class SweepRow:
    variant: str                          # CLI string; consumed as --variant by train_main
    seed: int                             # consumed as --seed
    overrides: Mapping[str, object]       # consumed as --override "key=value" for each item
    sweep_row_index: int                  # 0-indexed within parent SweepConfig
    ablation_cell: str | None             # mirrors sweep.ablation_cell_id

    # Derived fields (computed by harness, not in YAML):
    config_hash: str                      # see §3.5 — populated after V4Config materialisation, before subprocess spawn
    run_id: str                           # uuid4 hex; one per row execution (regenerated on retry; sweep_row_index stays stable)
    run_tag: str                          # f"{variant}_seed{seed}_row{sweep_row_index}_{config_hash[:8]}"
```

### 3.4 Override semantics

Each `overrides[i]` is a `Mapping[str, object]` whose keys are dotted `"section.field"` paths (matching `train_main.py:apply_overrides`); values are JSON literals (ints, floats, bools, strings, lists). The harness translates each key-value pair into a `--override "section.field=<json>"` CLI flag passed to `train_main.py` inside the subprocess (§5.3). The translation is **value-by-value `json.dumps`** so `True` → `"true"`, `3.14` → `"3.14"`, `["a", "b"]` → `"[\"a\", \"b\"]"`; this matches `train_main.py:_parse_value` which uses `json.loads`.

**Tuple semantics**: SweepConfig cartesian-multiplies `overrides`. If `overrides=({"train.lr": 3e-4}, {"train.lr": 1e-4})`, the harness emits 2 rows per (variant, seed). To run a single override that combines multiple keys, use one dict: `overrides=({"train.use_crn": True, "train.randomize_order": False},)` → 1 row per (variant, seed). The Abl4 CRN × Joint 2×2 cell in §3.2 demonstrates the latter idiom (4 dicts, each combining two flags).

### 3.5 `config_hash` computation

```python
# hyper_mve/experiments/sweep.py

import hashlib
import json
from dataclasses import asdict

def compute_config_hash(cfg: V4Config) -> str:
    """16-byte (32 hex char) blake2b of canonical-JSON-serialised V4Config.

    Stable across:
      - Python process restarts (no random salts)
      - dict ordering (sort_keys=True)
      - Tuple vs list (cfg.to_dict() coerces tuples to lists)

    Sensitive to (any of these changes the hash):
      - every field of V4Config.env / .model / .train / .mup / .eval / .legacy
      - V4Config.preset_name
    """
    canonical = json.dumps(cfg.to_dict(), sort_keys=True, default=str)
    return hashlib.blake2b(canonical.encode(), digest_size=16).hexdigest()
```

**16 bytes = 32 hex chars** is enough to make collision probability under realistic sweep cardinalities (≤ 10⁴ rows in this project's lifetime) negligible (birthday bound ~ 2¹⁶ ≈ 65 536). Truncating to `config_hash[:8]` in `run_tag` is for human readability only; the dedup key in the registry is the full 32-char hash.

**Sensitivity surface**: the hash covers *every* field of every sub-config. Two rows that differ only in `train.lr` have different hashes; two rows that differ only in `cfg.preset_name` also do. This is intentional — `config_hash` is the cache key for the oracle ceiling cache (spec 02 §3) and the dedup key for the registry, both of which need to discriminate any config delta that affects results.

**Why blake2b not SHA-256**: blake2b is faster on CPython, and 16-byte truncation is more conservative against the birthday bound than SHA-256 truncated to 16 bytes (negligible practical difference, but blake2b is the project's idiomatic choice).

**Pkg-07 cross-lock**: the oracle ceiling cache in spec 02 §3 uses the same `config_hash`. Drift in this hash function silently invalidates the cache (cache misses everywhere) but does not corrupt results — the next sweep just recomputes ceilings. A named test `test_config_hash_stable.py` (§9) asserts the hash of a fixed `V4Config.from_preset("easy")` is byte-identical across two consecutive calls (deterministic) and across two process invocations (no salt).

---

## 4. RunRegistry row schema (verbatim **22 schema-domain fields + 1 `schema_version` sentinel = 23 keys**, mirrored in spec 08 §4)

`runs/registry.jsonl` holds one JSON dict per line. Each line is a complete row; updates to a row are **new lines appended** (no in-place mutation). The 23-key schema:

```python
# hyper_mve/experiments/run_registry.py

@dataclass(frozen=True)
class RegistryRow:
    """One line of runs/registry.jsonl. Schema mirrored in spec 08 §4 byte-identically."""

    # === Identity (5) ===
    run_id: str                     # uuid4 hex, unique per row execution; regenerated on retry
    variant: str                    # CLI string (e.g. "hyper", "baseline_input_wide", "external_mappo"); 14 legal values from pkg-07 spec 01 §2
    seed: int                       # consumed as --seed by train_main
    config_hash: str                # 32-char blake2b hex of V4Config (§3.5); dedup key
    ablation_cell: str | None       # parent SweepConfig.ablation_cell_id (e.g. "abl4_crn_joint::row3"); None for main-table runs

    # === Provenance (4) ===
    sweep_row_index: int            # 0-indexed within parent SweepConfig
    git_sha: str                    # repo HEAD at row start; 40-char hex
    git_dirty: bool                 # True iff `git status --porcelain` is non-empty at row start
    started_at_iso8601: str         # row start time in UTC, e.g. "2026-06-20T14:32:11Z"

    # === Lifecycle (3) ===
    completed_at_iso8601: str | None    # row end time; None while status ∈ {pending, running}
    status: Literal[                    # state machine: pending → running → {completed, failed, skipped}
        "pending", "running", "completed", "failed", "skipped",
    ]
    failure_reason: str | None          # populated iff status ∈ {failed, skipped}; e.g. "NotImplementedError: stub external_marie", "exit_code=139 (SIGSEGV)"

    # === Resource (3) ===
    gpu_id: int | None              # CUDA device index, or None for CPU rows; pinned via CUDA_VISIBLE_DEVICES in subprocess env
    walltime_seconds: float | None  # row end - row start; None while running
    peak_gpu_memory_mb: float | None    # from torch.cuda.max_memory_allocated() at row end; None for CPU rows

    # === Output pointers (3) ===
    config_snapshot_path: str       # absolute path to runs/<run_tag>/config.yaml (resolved V4Config dump)
    checkpoint_path: str | None     # absolute path to runs/<run_tag>/ckpt_final.pt; None if training did not reach end
    eval_report_path: str | None    # absolute path to runs/<run_tag>/eval_report.json (EvalReport asdict + JSON); None if eval did not complete

    # NOTE: tensorboard_dir and the 3 summary metric fields are described in the
    # "duplicated summary metrics" block below. Authoritative count: count the
    # @dataclass body line-by-line — the bucket headers are navigational aids only.

    # === TensorBoard pointer (1) ===
    tensorboard_dir: str            # absolute path to runs/<run_tag>/tb/

    # === Duplicated summary metrics (3) — populated iff status == "completed" ===
    return_mean: float | None           # EvalReport.return_mean (mean across all eval episodes)
    return_zero_shot_unseen: float | None    # EvalReport.return_zero_shot_unseen (zero-shot headline)
    regret_mean: float | None           # EvalReport.regret_mean (mean regret across c-grid)
```

**Field count = 23 keys** (22 schema-domain + 1 `schema_version` sentinel = 23):
- Identity (5) + Provenance (4) + Lifecycle (3) + Resource (3) + Output pointers (3) + TensorBoard pointer (1) + Duplicated summary metrics (3) = **22 schema-domain fields**
- `schema_version` (1) sentinel below = **+1**
- **Total: 23 keys on every JSONL line.**

The schema sentinel is included as a field on every row:

```python
    schema_version: str = "pkg08-spec05-v1"     # bumped on synchronous spec 05 §4 + spec 08 §4 edit
```

So the on-the-wire JSON line has **23 keys total** (22 schema-domain + `schema_version`), enumerated in spec 08 §4 byte-identically. The test `test_run_registry_row_schema_matches_spec_08.py` (§9) asserts JSON-key sets are equal AND field count equals 23.

**Authoritative source for the count**: the `@dataclass` body above (count line-by-line). The bucket headers (Identity/Provenance/…) are navigational aids. The earlier prose claim of "18 fields" (now removed) was a drift inherited from design.md §3.4 (which is corrected synchronously alongside this spec).

### 4.1 Why these fields

| Group | Why these fields | Where consumed |
|-------|------------------|----------------|
| Identity (5) | Dedup tuple `(variant, seed, config_hash)` is the resume key (§8); `ablation_cell` lets spec 07 group by cell. `run_id` is per-execution uniqueness for the append-only invariant. | spec 06 ablation dispatcher; spec 07 stats groupby |
| Provenance (4) | `git_sha` + `git_dirty` lets a reviewer trace results back to source; `sweep_row_index` lets us reproduce a single row from a sweep config; `started_at` is the audit timestamp. | reviewer / Ch6 appendix reproducibility table |
| Lifecycle (3) | `status` state machine drives the resume logic (§8) and the `--retry-failed` flag. `failure_reason` is the human-readable string the dashboard surfaces. | resume logic; sweep harness console output |
| Resource (3) | `gpu_id` pins post-hoc analysis of GPU-correlated failures; `walltime_seconds` / `peak_gpu_memory_mb` are the cluster-accounting columns. | GPU-hour accounting; Ch6 cost table |
| Output pointers (3 + 1 TB) | Eager-load pointers: `pandas.read_json(...lines=True)` + `.assign(report=lambda d: d.eval_report_path.apply(read_eval_report))` reconstructs the full report ensemble for spec 07. | spec 07 stats; spec 08 integration |
| Duplicated summary metrics (3) | SQL-like filtering on the registry without opening every eval_report.json. `pandas.read_json(...).query("return_zero_shot_unseen > 5.0")` is the idiom. | spec 06 ablation dispatcher quick-look; spec 07 |

### 4.2 Append-only contract

State transitions are **always** new lines, never in-place mutation:

```
1. parent enqueues row → append RegistryRow(status="pending", run_id=R0, ...)
2. subprocess starts   → append RegistryRow(status="running", run_id=R0, ...) (same run_id; new line)
3. subprocess completes → append RegistryRow(status="completed", run_id=R0, completed_at=..., ...) (same run_id; new line)
```

Reads use `pandas.read_json("runs/registry.jsonl", lines=True).sort_values("started_at_iso8601").groupby("run_id").last()` to materialise the **latest state** per row.

**Failed/skipped rows stay**: status="failed" rows are NOT deleted. The `(variant, seed, config_hash)` dedup tuple becomes the resume key (§8); when the user reruns the sweep, the harness emits a *new* `run_id` but the same dedup tuple — the registry now has two history chains for the same logical row, the second one succeeding. Spec 07 stats reads `groupby((variant, seed, config_hash)).agg(last)` to use the latest attempt.

**Why not sqlite**: JSONL has no schema migration story but also no schema migration burden. Adding a field to RegistryRow (after a synchronous spec 05 + spec 08 §4 edit) is "fill `None` for old rows on read"; `pandas.read_json` handles missing columns natively. SQLite would require an `ALTER TABLE` and a one-shot data migration script — overkill for a research codebase with ~10⁴ lifetime rows.

**Why not CSV**: nested fields. `ablation_cell` is sometimes string, sometimes None; `overrides` (if we ever exposed it on the row) would be a dict. JSONL handles both natively. CSV would force string-encoding everywhere.

### 4.3 File-lock contract

Concurrent appenders (parent + each subprocess) use OS file locks:

```python
# hyper_mve/experiments/run_registry.py (sketch)

import sys
import time

if sys.platform == "win32":
    import msvcrt

    def _lock_exclusive(fd: int) -> None:
        # msvcrt.locking takes a length; we lock 1 byte at current file position.
        # Retry on LK_NBLCK until acquired (msvcrt has no LK_LOCK blocking semantics
        # on append; we poll with backoff).
        for delay in (0.001, 0.005, 0.02, 0.1, 0.5):
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                time.sleep(delay)
        msvcrt.locking(fd, msvcrt.LK_LOCK, 1)  # final blocking attempt

    def _unlock(fd: int) -> None:
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _lock_exclusive(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX)

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


def append_row(registry_path: pathlib.Path, row: RegistryRow) -> None:
    line = json.dumps(asdict(row)) + "\n"
    with open(registry_path, "ab") as f:
        _lock_exclusive(f.fileno())
        try:
            f.write(line.encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        finally:
            _unlock(f.fileno())
```

**Why fsync**: subprocess crashes immediately after write would otherwise lose the line. `os.fsync` is cheap on append-only JSONL (single page write).

**Why `"ab"` not `"a"`**: binary mode avoids platform newline translation (`\n` → `\r\n` on Windows in text mode); the test `test_run_registry_concurrent_safe.py` asserts the file has exactly `n` newline bytes for `n` rows.

**Per-row append granularity**: every state transition is one `append_row` call. The lock window is microseconds (one write + fsync). The 16-thread × 100-row stress test (§9) completes in < 1 s on commodity hardware.

---

## 5. SubprocessRunner contract

### 5.1 Why subprocesses

Five rationale points (echoed from design §3.5, mechanised here):

1. **CUDA OOM does not propagate**. v4-opt 2026-06 vectorised collection occasionally peaks GPU memory beyond `train.episodes_per_iter` budget; an in-process row loop would crash the parent. Subprocesses crash independently.
2. **GPU driver-level cleanup**. `torch.cuda.empty_cache()` releases the PyTorch allocator's pool but not the CUDA driver context; only process exit truly releases the GPU. Subsequent rows on the same GPU see a fresh context.
3. **V4Config dataclass independence**. `frozen=True` makes V4Config nominally immutable, but trainer state (optimiser, scheduler, EMA target) carries process-global side-effects (`torch.set_num_threads`, matplotlib backend, RNG generators). Subprocess isolation kills all of that.
4. **Per-GPU semaphore scheduling** (§7). Subprocesses are the natural unit for GPU pinning via `CUDA_VISIBLE_DEVICES`; in-process pinning requires `torch.cuda.set_device` which is process-global.
5. **Graceful Ctrl-C**. Parent's `signal.SIGINT` handler propagates SIGTERM to children, registry gets `status="failed", failure_reason="user_interrupt_SIGINT"` rows, sweep resumes cleanly on next invocation (§8).

### 5.2 Subprocess invocation

```python
# hyper_mve/experiments/sweep.py (sketch)

def _spawn_row(row: SweepRow, sweep_cfg: SweepConfig, gpu_id: int | None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Build the train_main argv inside the worker; we pass the row payload via stdin
    # rather than CLI so that override dicts with nested structures don't blow up
    # CLI quoting (Windows cmd.exe + JSON quoting is fragile).
    payload = {
        "variant": row.variant,
        "seed": row.seed,
        "overrides": row.overrides,
        "preset": sweep_cfg.preset,
        "max_steps": sweep_cfg.max_steps,
        "eval_planner_mode": sweep_cfg.eval_planner_mode,
        "run_tag": row.run_tag,
        "config_snapshot_path": str(run_dir / "config.yaml"),
        "eval_report_path":     str(run_dir / "eval_report.json"),
        "checkpoint_path":      str(run_dir / "ckpt_final.pt"),
        "tensorboard_dir":      str(run_dir / "tb"),
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "hyper_mve.experiments._sweep_worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=str(repo_root),                    # D:\RL\hyper_mve per CLAUDE.md "Running experiments"
    )
    proc.stdin.write(json.dumps(payload).encode("utf-8"))
    proc.stdin.close()
    stdout, stderr = proc.communicate()        # blocks until subprocess exits
    return subprocess.CompletedProcess(
        args=proc.args, returncode=proc.returncode, stdout=stdout, stderr=stderr,
    )
```

### 5.3 `_sweep_worker.py` contract

```python
# hyper_mve/experiments/_sweep_worker.py

"""Sweep worker: one process per sweep row.

Reads a SweepRow payload from stdin (JSON), constructs the train_main argv,
calls train_main.main(argv), then runs the eval and writes EvalReport.

Exit codes:
  0   → row completed successfully; EvalReport written to eval_report_path
  1   → row failed (Python exception); failure_reason in stderr last line
  2   → row skipped (NotImplementedError on stub variants); ditto
  >2  → row crashed below Python (CUDA segfault, etc.); failure_reason="exit_code=N"
"""

def main() -> None:
    payload = json.loads(sys.stdin.read())
    argv = [
        "--variant", payload["variant"],
        "--seed", str(payload["seed"]),
        "--preset", payload["preset"],
        "--max_steps", str(payload["max_steps"]),
        "--log_dir", payload["tensorboard_dir"],
        "--ckpt_dir", str(pathlib.Path(payload["checkpoint_path"]).parent),
    ]
    for key, value in payload["overrides"].items():
        argv += ["--override", f"{key}={json.dumps(value)}"]
    # Inject eval_planner_mode via an override (cfg.eval.eval_planner_mode landed
    # via design D10; pkg-01 spec 05 consumes).
    argv += ["--override", f"eval.eval_planner_mode={json.dumps(payload['eval_planner_mode'])}"]

    try:
        from hyper_mve.scripts.train_main import main as train_main
        train_main(argv)
    except NotImplementedError as e:
        # Stub external_marie / external_ga / unsourced external_mamba
        print(f"SKIP: {e}", file=sys.stderr, flush=True)
        sys.exit(2)
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    # train_main wrote ckpt + TB; now write resolved config + run unified evaluator.
    cfg = _materialise_cfg(payload)
    _write_config_snapshot(cfg, payload["config_snapshot_path"])
    runner = _load_runner_for_eval(cfg, payload["variant"], payload["checkpoint_path"])
    env_fn = _build_env_fn(cfg)
    from hyper_mve.eval.unified_evaluator import evaluate
    report = evaluate(runner, env_fn, cfg)
    _write_eval_report(report, payload["eval_report_path"])
    sys.exit(0)


if __name__ == "__main__":
    main()
```

**Exit code → registry mapping**:

| Exit code | RegistryRow.status | failure_reason example |
|-----------|--------------------|------------------------|
| 0         | `"completed"`      | None |
| 1         | `"failed"`         | `"FAIL: ValueError: train.lr=0 invalid"` (last stderr line) |
| 2         | `"skipped"`        | `"SKIP: NotImplementedError: external_marie stub"` |
| 139 (SIGSEGV) | `"failed"`     | `"exit_code=139 (SIGSEGV)"` |
| -SIGINT   | `"failed"`         | `"exit_code=-2 (SIGINT, parent interrupt)"` |

### 5.4 Stdout/stderr capture

Stdout (the `train_main.py` console output: `[train_main] preset=... variant=...`, warmup buffer fill progress, per-eval scalars) is **not** written to the registry. It is buffered in the parent and printed verbatim with a `[row N]` prefix on subprocess exit (success or fail). Stderr's *last line* is captured into `failure_reason` (per the mapping table above).

The full stdout / stderr is **not** persisted to disk — TensorBoard event files (in `tensorboard_dir`) carry the structured scalars; the registry row points there. If the user wants the raw text, they redirect on the sweep CLI:

```powershell
python -m hyper_mve.experiments.sweep --config sweep.yaml 2>&1 | Tee-Object sweep.log
```

---

## 6. Process-isolation rationale (verbatim from design §3.5)

Five points (already itemised in §5.1); this section anchors them as the SDD rationale-of-record.

| # | Reason | Without subprocess isolation | With subprocess isolation |
|---|--------|-------------------------------|---------------------------|
| 1 | CUDA OOM does not propagate | Row 1 OOM → CUDA context corrupt → rows 2..N all OOM | Row 1 OOM → SIGKILL → row 2 starts clean |
| 2 | GPU driver-level cleanup | `torch.cuda.empty_cache()` does not release driver context | subprocess exit releases everything |
| 3 | V4Config dataclass independence | `torch.set_num_threads(8)` from row 1 affects row 2 | each row starts with default thread count |
| 4 | Per-GPU semaphore scheduling | `torch.cuda.set_device(0)` race with concurrent row | `CUDA_VISIBLE_DEVICES=0` is process-local |
| 5 | Graceful Ctrl-C | parent crash kills everything; no resume state | parent SIGINT → children SIGTERM → registry has `status="failed"` rows; resume picks up |

**Counter-argument weighed**: "subprocess fork overhead is 100ms per row → for 1 000 rows, 100 s wasted". True; the alternative (CUDA OOM corrupts a row mid-sweep, wasting hours of GPU time) is far worse. We accept the overhead.

---

## 7. Per-GPU semaphore

```python
# hyper_mve/experiments/sweep.py (sketch)

import multiprocessing as mp

class GpuSemaphore:
    """Token bucket over n_gpus tokens; each token is the integer gpu_id ∈ [0, n_gpus).

    acquire() blocks until a token is available, returns gpu_id; release(gpu_id) returns it.
    """

    def __init__(self, n_gpus: int) -> None:
        self._n = n_gpus
        self._available: mp.Queue[int] = mp.Queue(maxsize=n_gpus)
        for i in range(n_gpus):
            self._available.put(i)

    def acquire(self) -> int:
        return self._available.get()      # blocks

    def release(self, gpu_id: int) -> None:
        self._available.put(gpu_id)
```

**Why a token queue, not `mp.Semaphore`**: `mp.Semaphore` gives binary block/unblock; we need to know **which** GPU the subprocess gets, so we can pass `CUDA_VISIBLE_DEVICES=<gpu_id>`. A token queue of integers gives both.

**`max_parallel` clamp**: `run_sweep` accepts `--max-parallel N` from the CLI (§8.1). The harness clamps `max_parallel = min(args.max_parallel, sweep_cfg.n_gpus, len(rows))` so it never spawns more subprocesses than GPUs. CPU-only rows (when `cuda.is_available() == False`) fall back to `max_parallel = 1` (no benefit to over-spawning).

**Lifecycle**:

```python
for row in rows:
    gpu_id = sem.acquire()                            # blocks until GPU free
    registry.append_pending(row, gpu_id=gpu_id)
    proc = _spawn_row(row, sweep_cfg, gpu_id)         # blocking; spawns and waits
    registry.append_completed_or_failed(row, proc, gpu_id=gpu_id)
    sem.release(gpu_id)
```

This is **sequential per GPU** but **parallel across GPUs**: for `n_gpus=2, max_parallel=2`, the harness alternates GPU 0 / GPU 1 row spawns. A `ThreadPoolExecutor(max_workers=max_parallel)` wraps the loop body so the parent thread does not block.

Test `test_gpu_semaphore.py` (§9) parametrises `n_gpus ∈ {1, 2, 4}` and asserts: (a) at most `n_gpus` subprocesses are alive simultaneously (introspected via `len(active_pids)`), (b) each gpu_id appears in `RegistryRow.gpu_id` exactly `ceil(n_rows / n_gpus)` times in steady state, (c) `release` of an unacquired gpu_id raises `ValueError`.

---

## 8. Resume / restart

### 8.1 CLI

```
python -m hyper_mve.experiments.sweep \
    --config <sweep_yaml> \
    [--max-parallel N] \
    [--n-gpus N] \
    [--dry-run] \
    [--retry-failed] \
    [--registry-path <path>]
```

| Flag | Default | Effect |
|------|---------|--------|
| `--config` | required | Path to SweepConfig YAML |
| `--max-parallel` | 1 | Clamp on concurrent subprocesses; clamped to `min(arg, n_gpus, len(rows))` |
| `--n-gpus` | from `nvidia-smi -L` or 1 | Number of GPUs the harness may consume |
| `--dry-run` | False | Print cartesian plan (variant × seed × override) and exit 0 without spawning subprocesses |
| `--retry-failed` | False | Re-enqueue rows whose latest registry state is `status="failed"`; without this flag, failed rows stay failed and the harness skips them |
| `--registry-path` | `runs/registry.jsonl` | Override registry location (used by tests to redirect to a tmp dir) |

### 8.2 Resume logic (C8-ABL-SWEEP1 acceptance criterion 5)

On `run_sweep(sweep_cfg)` entry:

1. Enumerate cartesian → `rows: list[SweepRow]` with `config_hash` computed per row.
2. Load existing registry: `existing = pandas.read_json(registry_path, lines=True).groupby("run_id").last()`.
3. For each row, look up latest state by `(variant, seed, config_hash)`:
   - If latest `status == "completed"` → **skip** (already done); log `[skip] row {i}: completed at {completed_at}`.
   - If latest `status == "skipped"` → **skip** (permanent stub failure); log `[skip] row {i}: stub`.
   - If latest `status == "failed"`:
     - With `--retry-failed`: enqueue with new `run_id`.
     - Without `--retry-failed`: skip; log `[skip] row {i}: failed (re-run with --retry-failed)`.
   - If latest `status ∈ {"pending", "running"}` (the previous sweep was interrupted mid-row) → enqueue with new `run_id`; the orphaned `pending`/`running` row stays in the registry as audit trail. Log `[orphan] row {i}: prior run_id={...} was {status}; re-enqueueing`.
   - If no latest state → enqueue with new `run_id`.
4. Process the enqueued rows under the per-GPU semaphore (§7).

Test `test_resume_skips_completed.py` (§9) constructs a registry with 5 `status="completed"` rows for a sweep with 10 rows, runs the harness, asserts only 5 new subprocesses are spawned, and the 5 completed rows are not re-spawned.

### 8.3 Dry-run

`--dry-run` short-circuits subprocess spawning:

```
[dry-run] SweepConfig: variants=4, seeds=5, overrides=2, total=40 rows
[dry-run] Existing registry: 12 completed, 3 failed, 25 missing
[dry-run] Without --retry-failed: 25 rows would be spawned
[dry-run] With --retry-failed: 28 rows would be spawned
[dry-run] Per-GPU semaphore: max_parallel=2, n_gpus=2
[dry-run] Estimated GPU-hours (per-row default 2 hr): 50.0
```

Test `test_dry_run_no_subprocess.py` (§9) asserts `subprocess.Popen` is never called when `--dry-run` is set (monkey-patches `subprocess.Popen` to raise).

---

## 9. Test contract (C8-ABL-* enforcement)

Each test is at `tests/experiments/test_*.py` and is **named** so that the CI grep matches the C8-ABL-* acceptance criterion in `../README.md`. Minimum 6 tests (one per acceptance criterion + the schema-drift detector):

| Test file | C8-ABL-* | Asserts |
|-----------|----------|---------|
| `test_sweep_cartesian_correct.py` | SWEEP1 | `len(enumerate_cartesian(sweep))` matches `\|V\|·\|S\|·max(1,\|O\|)` for 4 parametrised shapes; row order is lexicographic over (variant, seed, override-idx); empty `overrides=()` collapses to single empty-dict row. |
| `test_run_registry_jsonl_append_atomic.py` | REG1 | 16 threads × 100 row appends → exactly 1 600 lines, no partial lines, no JSON parse errors, no interleaved bytes; assertion uses `wc -l` and `pandas.read_json(lines=True)` roundtrip. |
| `test_run_registry_row_schema_matches_spec_08.py` | REG1 + Lock 3 | Parses §4 of this spec and §4 of spec 08 for the field-group block fences; asserts JSON-key sets are equal; asserts field count is **22 (schema-domain) + 1 (schema_version) = 23 keys**. |
| `test_sweep_subprocess_isolation.py` | ISO1 | Injects a row that calls `os._exit(139)` (simulated SIGSEGV); asserts parent sweep continues; asserts registry records `status="failed"` with `failure_reason="exit_code=139 (SIGSEGV)"`; asserts next row's subprocess starts on a clean CUDA context (PID differs from the dead row). |
| `test_config_hash_stable.py` | SWEEP1 | `compute_config_hash(V4Config.from_preset("easy"))` is byte-identical across two consecutive calls (deterministic) and across two subprocess invocations (no salt); changing any field of any sub-config changes the hash (parametrised over `(env.N, train.lr, model.latent_dim, eval.evaluate_episodes, mup.base_width, legacy.log_every_n)`). |
| `test_resume_skips_completed.py` | SWEEP1 | Constructs a registry with 5 `status="completed"` rows for a sweep with 10 rows; runs the harness; asserts only 5 new subprocesses are spawned; asserts the 5 completed rows are not re-spawned. Re-runs with `--retry-failed` after seeding 2 failed rows; asserts 7 new subprocesses are spawned. |
| `test_gpu_semaphore.py` | ISO1 | Parametrised over `n_gpus ∈ {1, 2, 4}`: (a) at most `n_gpus` subprocesses alive simultaneously, (b) each gpu_id appears `ceil(n_rows / n_gpus)` times in `RegistryRow.gpu_id` in steady state, (c) `release(unacquired_id)` raises `ValueError`. |
| `test_dry_run_no_subprocess.py` | SWEEP1 | Monkey-patches `subprocess.Popen` to raise; runs `run_sweep` with `--dry-run`; asserts the call succeeds with no subprocess spawn; asserts the dry-run plan stdout contains "would be spawned". |

**8 tests total** ≥ "minimum 6". Each test name maps 1-1 to a `C8-ABL-*` criterion (column 2).

---

## 10. Integration hooks

### 10.1 With pkg-08 spec 02 (zero-shot + c_hidden + regret)

Spec 02 is a **passive consumer**: its zero-shot eval, c_hidden eval, and regret computation all happen inside `_sweep_worker.py`'s call to `unified_evaluator.evaluate(runner, env_fn, cfg)` (spec 01 §2). The sweep harness has no special branch for spec 02 — the `cfg.env.c_visible` and `cfg.eval.zero_shot_*` fields are consumed by the unified evaluator, not by the harness.

Regret cache integration: spec 02 §3 writes to `runs/_oracle_ceilings/<config_hash>/<c>.json` using the same `config_hash` from §3.5 of this spec. Cache hits cross sweep invocations; the harness does not manage the cache (spec 02 owns it).

### 10.2 With pkg-08 spec 03 (four-mode planner dispatch)

Spec 03 introduces `cfg.eval.eval_planner_mode: Literal["direct_inference", "planner_no_crn", "planner_no_coord_desc", "planner_full"]`. The sweep harness consumes this via `SweepConfig.eval_planner_mode` (§3.1) and passes it to `train_main.py` via `--override eval.eval_planner_mode=<json>` (§5.3). To sweep the mode axis, the user writes 4 separate SweepConfig YAMLs (or a single YAML with 4 overrides), one per mode. The harness does not auto-enumerate the 4 modes — that is the YAML author's responsibility (avoids `n_modes × n_variants × n_seeds × n_overrides` explosion).

### 10.3 With pkg-08 spec 04 (μP self-check)

Spec 04's 18-run plan (2 widths × 3 LRs × 3 seeds) is materialised as a `SweepConfig` with:

```python
SweepConfig(
    variants=("hyper",),
    seeds=(0, 1, 2),
    overrides=tuple(
        {"model.latent_dim": w, "train.lr": lr}
        for w in (64, 128)
        for lr in (1e-4, 3e-4, 1e-3)
    ),
    preset="easy",
    max_steps=50000,
    eval_planner_mode="planner_full",
    ablation_cell_id="mup_verification",
)
```

→ `1 · 3 · 6 = 18` rows. The μP self-check spec does not need any new harness machinery.

### 10.4 With pkg-08 spec 06 (`--ablation` CLI canned YAMLs)

Spec 06 is a **thin dispatcher** over this harness:

```python
# hyper_mve/experiments/ablate.py (spec 06 sketch)

def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", required=True, choices=(
        "abl1", "abl4_crn_joint", "abl4_joint_easy_n2", "abl6", "abl7",
    ))
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--n-gpus", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    yaml_path = pathlib.Path("hyper_mve/experiments/ablations") / f"{args.ablation}.yaml"
    sweep_cfg = SweepConfig.from_yaml(yaml_path)
    run_sweep(sweep_cfg, max_parallel=args.max_parallel, n_gpus=args.n_gpus,
              dry_run=args.dry_run, retry_failed=args.retry_failed)
```

Spec 06 owns the YAML content; this spec owns `from_yaml` + `run_sweep`.

### 10.5 With pkg-08 spec 07 (`stats.py` Welch t / `compare.py`)

Spec 07's reading path:

```python
# hyper_mve/experiments/stats.py (spec 07 sketch)

def load_registry(registry_path: pathlib.Path) -> pandas.DataFrame:
    df = pandas.read_json(registry_path, lines=True)
    df = df.sort_values("started_at_iso8601").groupby("run_id").last().reset_index()
    df = df[df.status == "completed"]
    return df

def welch_t(df: pandas.DataFrame, a_variant: str, b_variant: str, c: float) -> tuple[float, float]:
    a_reports = df[df.variant == a_variant].eval_report_path.apply(_load_report)
    b_reports = df[df.variant == b_variant].eval_report_path.apply(_load_report)
    a_returns = [r.return_per_c[c] for r in a_reports]
    b_returns = [r.return_per_c[c] for r in b_reports]
    return scipy.stats.ttest_ind(a_returns, b_returns, equal_var=False)
```

The harness exposes `eval_report_path` per row (§4 output pointers) precisely so spec 07 can do the lazy `.apply(_load_report)` join.

### 10.6 With pkg-08 spec 08 (integration contracts)

Spec 08 §4 mirrors the RegistryRow schema (§4 of this spec) byte-identically. Spec 08 §5 declares the 3 downstream code patches that the harness's `_sweep_worker.py` indirectly consumes (Pkg-02 obs-mask + Pkg-05 planner flag + Pkg-05 CLI rename) — the harness does not touch the patch sites itself; the worker subprocess inherits the patched behaviour through the unchanged `train_main.py` entry point.

### 10.7 With pkg-07 (reverse-consumed contracts)

Three pkg-07 anchors are reverse-consumed by this spec:

1. **pkg-07 spec 01 §2 REGISTRY 11 keys** → SweepConfig.variants enumeration. Spec 06 ablation YAMLs reference variants by REGISTRY keys; this spec's `_sweep_worker.py` calls `train_main.py --variant <key>` which dispatches via `create_baseline(cfg, cli_to_factory_arg(<key>))`. Drift in REGISTRY → drift in SweepConfig variants → caught by spec 08 §6 drift detector.
2. **pkg-07 spec 04 N-parametric adapter `env_fn`** → `_sweep_worker.py`'s `_build_env_fn(cfg)` constructs `ResourceCommonsPettingZooEnv(cfg.env, oracle_mode=False, eval_info_mode=True)` (the unique caller permitted to flip `eval_info_mode=True` per pkg-07 spec 04 §10). The harness has no special branch — `env_fn` is opaque to the harness, owned by the worker.
3. **pkg-07 spec 08 `evaluate(env_fn, c_grid, episodes) -> EvalReport` signature** → `_sweep_worker.py` calls `unified_evaluator.evaluate(runner, env_fn, cfg)` (spec 01 dispatcher) which in turn calls the runner's `.evaluate(env_fn, c_grid, episodes) -> EvalReport`. The harness writes the returned `EvalReport` to disk at `eval_report_path` and records the path on the registry row.

---

## 11. Cross-references

### 11.1 Intra-pkg-08

- **spec 01 unified-evaluator**: `_sweep_worker.py` calls `unified_evaluator.evaluate(runner, env_fn, cfg) -> EvalReport`; the harness consumes the returned report's `.return_mean`, `.return_zero_shot_unseen`, `.regret_mean` for the duplicated summary metric fields on RegistryRow (§4).
- **spec 02 zero-shot + c_hidden + regret**: regret cache key is `config_hash` from §3.5 of this spec.
- **spec 03 four-mode planner dispatch**: `SweepConfig.eval_planner_mode` field; passed to worker via `--override eval.eval_planner_mode=<json>`.
- **spec 04 μP self-check**: 18-run plan materialised as a `SweepConfig` (§10.3); no new harness machinery.
- **spec 06 ablation CLI**: dispatches to `run_sweep(SweepConfig.from_yaml(<id>.yaml))`.
- **spec 07 stats / compare**: reads `runs/registry.jsonl` via `pandas.read_json(lines=True)`; uses `eval_report_path` for lazy report load.
- **spec 08 integration contracts**: mirrors RegistryRow schema (§4) byte-identically; declares 3 downstream code patches; cross-locks pkg-07 reverse consumption.

### 11.2 Pkg-07 reverse consumption

- **pkg-07 spec 01 §2 REGISTRY** → `SweepConfig.variants` enumeration (§3.1).
- **pkg-07 spec 01 §5 `cli_to_factory_arg`** → consumed transparently by `train_main.py` inside the worker (§5.3); the harness does not call it directly.
- **pkg-07 spec 04 N-parametric `env_fn`** → `_build_env_fn(cfg)` in worker (§5.3, §10.7).
- **pkg-07 spec 08 `evaluate()` signature** → harness writes returned `EvalReport` to `eval_report_path` (§4 output pointers).

### 11.3 Existing repo facts consumed

- **`hyper_mve/scripts/train_main.py`** (lines 55-76 `parse_args`, line 130 `V4Config.from_preset`, line 86 `apply_overrides`): worker constructs `argv` matching this CLI surface (§5.3); CLI flag rename `--use_coord_desc → --randomize_order` is declared in spec 06 + spec 08 as the third downstream code patch, but the harness consumes the renamed flag transparently (worker emits `--override train.randomize_order=<json>` rather than the CLI flag form).
- **`hyper_mve/configs/v4_config.py`** (lines 81-85 `to_dict`): used by §3.5 `compute_config_hash` for canonical JSON serialisation.
- **`hyper_mve/training/evaluation.py:run_eval`**: not directly consumed by the harness, but is the inner subroutine called by `unified_evaluator.evaluate` (spec 01 §4).

### 11.4 Theory Audit anchors

- **§10.3 Welch t / 5 seeds**: SweepConfig.seeds default ≥ 5 (recommended); < 5 emits `[WARN]` (passed through from spec 07).
- **M8 Abl4 redefinition**: SweepConfig override semantics (§3.4) support the 2×2 CRN × Joint-CoordDesc cell as 4 override dicts; spec 06 YAMLs make this concrete.
- **Q2 `use_coord_desc → randomize_order` rename**: harness writes `--override train.randomize_order=<json>`; CLI alias handled by `train_main.py` (spec 06 / spec 08 downstream patch).
- **Q7 c_hidden**: `cfg.env.c_visible` propagates via override; harness has no special branch.

---

## 12. Anchors (explicit)

- **C8-ABL-SWEEP1** — SweepConfig cartesian correctness: §3.3 (semantics) + §9 (`test_sweep_cartesian_correct.py`, `test_config_hash_stable.py`, `test_resume_skips_completed.py`, `test_dry_run_no_subprocess.py`).
- **C8-ABL-ISO1** — subprocess isolation: §5.1 + §5.2 + §6 + §9 (`test_sweep_subprocess_isolation.py`, `test_gpu_semaphore.py`).
- **C8-ABL-REG1** — JSONL append-only + concurrent file lock: §4 (schema) + §4.2 (append-only contract) + §4.3 (file lock) + §9 (`test_run_registry_jsonl_append_atomic.py`, `test_run_registry_row_schema_matches_spec_08.py`).
- **Schema mirror lock to spec 08 §4**: §4 of this spec + spec 08 §4 byte-identical; drift detector at `test_run_registry_row_schema_matches_spec_08.py`.
- **23-key row schema enumeration**: §4 (verbatim) — identity 5 + provenance 4 + lifecycle 3 + resource 3 + output pointers 3 + TB pointer 1 + duplicated summary metrics 3 = **22 schema-domain fields**; plus `schema_version` sentinel (1) = **23 keys on every JSONL line**. The earlier "18 fields" slogan (in spec 05 prose + design.md §3.4 + design.md HARD-GATE row 6 + §10 user checklist) was drift — corrected synchronously. The `@dataclass` body in §4 is the authoritative source; count it line-by-line.
- **Pkg-07 reverse consumption**: §10.7 + §11.2; drift detector cross-locks REGISTRY / `env_fn` / `evaluate()` signature.
- **Pkg-08 spec 06 dispatcher integration**: §10.4 (canned YAML → `SweepConfig.from_yaml` → `run_sweep`).
- **Pkg-08 spec 07 stats consumption**: §10.5 (registry read + `eval_report_path` lazy load).
- **Pkg-08 spec 08 integration lock**: §10.6 + §11.1 (schema mirror, 3 downstream patches inherited via worker).
