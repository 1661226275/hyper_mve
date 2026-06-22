# Spec 04 — μP Base-Shape LR-Doubling Verification (18-Run Self-Test)

> **Parent docs**: [`../proposal.md`](../proposal.md) §2.1 (G11 衍生目标) · [`../design.md`](../design.md) §2.2 G11 (μP base-shape LR-doubling self-test) + §4 D2 (`EvalReport.return_mean` consumer slot) + §4 D10 末段 (per-impl tuning constants pattern) · [`../README.md`](../README.md) C8-EVAL-MUP1 (acceptance criterion)
> **Upstream consumed**: [`./01-unified-evaluator.md`](./01-unified-evaluator.md) §3 (`EvalReport.return_mean` for the headline scalar; `EvalReport.config_hash` for cache key composition) · [`./05-sweep-harness-and-run-registry.md`](./05-sweep-harness-and-run-registry.md) §10.3 (the 18-row `SweepConfig` plan that materialises this spec's grid) + §3.1 (`SweepConfig.ablation_cell_id="mup_verification"` channel) + §4 (`RunRegistry` row schema, source of post-hoc verdict aggregation) · pkg-04 design §NG3 + Chapter 5 §5.10.2 + Chapter 6 §6.2.5 (the μP claim itself, which this self-test either substantiates or falsifies) · `hyper_mve/configs/mup_config.py` (the placeholder `MupConfig` dataclass on `V4Config.mup`)
> **Sibling specs cross-locked**: spec 01 (μP verdict does NOT extend `EvalReport`; it is a self-contained dataclass `MupSelftestVerdict` defined in this spec §3.4) · spec 05 (this spec is a passive consumer of the harness — no new harness machinery added) · spec 07 (this spec's headline figure goes through the stats layer's matplotlib-Agg headless path) · spec 08 (NO new cfg fields added by this spec — the 5 self-test parameters live as module-level constants per the pkg-07 spec 05 §11 per-impl-constants pattern)
> **Status**: SDD only — describes the contract for `hyper_mve/eval/mup_verification.py`, not the implementation. The implementation phase produces the 18-run output exactly once; on completion, the verdict is either committed to Ch6 §6.2.5 (PASS / WARN) or to the appendix as a disclosed failure (FAIL).

---

## Header — three hard locks

### Lock 1 — Self-contained 18-run experiment, fixed budget, no expansion

This spec is **self-contained** in the strict sense: it consumes only the unified evaluator's `EvalReport.return_mean` and the sweep harness's row-dispatch + RunRegistry write-path. It introduces no new harness machinery, no new ablation-cell type, and **no new cfg fields on `EvalConfig` / `EnvConfig` / `TrainConfig`**. The grid is fixed at **2 widths × 3 LRs × 3 seeds = 18 runs**, all on the Easy preset (`N=2`, `K=8`), each run trained for `50_000` env-steps. Wall-clock budget: **estimated** ≈ 2 GPU-hr per run × 18 = ≈ 36 GPU-hr at 2 GPU-hr/run; concrete figure to be reconciled with implementation-phase profiling and the README §0 / proposal §4.2 aggregate 850 GPU-hr allotment (the README inline-lists "μP 自检 18 runs" as one of the 850-hr aggregate's components but does NOT itemise a per-component budget; the 36-hr figure here is a planning estimate, not a contracted line-item). The grid **MUST NOT** be expanded inside this spec — any reviewer pushing for "more widths" or "more LRs" is referred to pkg-08 design §6.G11 and the README C8-EVAL-MUP1 acceptance criterion, both of which lock the 18-row size as the floor and the ceiling for this self-test. Expansion is permissible only as an appendix-disclosed follow-up after this spec's verdict is in.

### Lock 2 — PASS / WARN / FAIL is QUANTITATIVE, not visual

The LR-doubling self-test is decided by a **single quantitative rule** applied to the 18 runs' final held-out losses (see §3 below). No "by eye" overlap of LR-loss curves is permitted to override the rule. The rule is: for each seed, the LR achieving the best held-out loss must be **identical across widths** (within a tolerance of one step on the chosen log-spaced LR grid — see octave-semantics note below); aggregating across the 3 seeds yields the verdict:

- **PASS**: all 3 seeds have the same best-LR across both widths.
- **WARN**: best LRs differ by at most **one step on the grid** for ≥2 of 3 seeds.
- **FAIL**: any other outcome (i.e. best-LR differs by ≥2 steps on the grid for ≥2 seeds, **or** WARN-rule majority is not met).

**Octave semantics note.** This spec operationalises "one octave" as **one step on the chosen log-spaced LR grid (≈ factor 3.16 in raw LR units)**, NOT as the textbook factor-of-2 doubling. The grid `(3e-4, 1e-3, 3e-3)` is half-octave-spaced in the base-10-decade sense (each step is one half-decade ≈ 3.16×); the textbook factor-of-2 LR-doubling test would require a denser grid (e.g., `(2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3)`) which is **deferred to a follow-up appendix sweep** beyond this 18-row budget. The chosen operational octave (one grid-step) is the relaxed-but-clearly-defined criterion this self-test uses; the canonical "factor 2 doubling" test is acknowledged as a stricter follow-up.

The full truth table is enumerated in §3.3 and tested in §7. The rule is the only arbiter; no manual override at verdict time.

### Lock 3 — Explicit failure protocol; the paper drops the μP claim on FAIL

Upon **FAIL**, the paper's main-text μP claim in Chapter 5 §5.10.2 ("启用 Tensor Programs / μP for hidden_dim scaling") and Chapter 6 §6.2.5 (μP 学习率对齐协议) is **dropped** and replaced with an appendix subsection "μP self-test result (failed)" that discloses:
1. The verdict (FAIL),
2. The full 18-run raw losses table,
3. The headline figure (§5),
4. A one-paragraph commentary stating the alternate parametrization used (the documented Roadmap §2.3 降级方案: per-baseline independent LR sweep with no μP scaling).

Upon **WARN**, the main-text claim is **retained** but a footnote citing the appendix self-test (with the disclosed octave-delta and seed-majority) is added; the appendix carries the same disclosure as the FAIL case minus the "claim dropped" paragraph. Upon **PASS**, the main-text claim is retained as-is and a single sentence in Ch5 §5.10.2 + Ch6 §6.2.5 references this spec's verdict ("μP base-shape LR-doubling self-test passed at 2 widths × 3 LRs × 3 seeds; verdict logged at `runs/_mup_selftest/<config_hash>/mup_selftest.json`").

The failure protocol is **non-negotiable**: this spec is intentionally designed so that any of the three verdicts is publication-safe. The PASS path keeps the claim, the WARN path discloses the delta, and the FAIL path drops the claim with a clean alternate. No verdict produces an unpublishable result.

---

## 1. Purpose

Reviewers expect that an architectural claim ("our hypernetwork is μP-base-shape compatible — i.e., the same LR transfers across widths") is empirically backed by the standard μP self-test: the **LR-doubling curve**. At several widths, sweep the LR across an order-of-magnitude range; if μP holds, the loss-vs-LR curves at different widths align in the sense that the LR achieving the lowest loss is the same across widths. This is the "coordinate check" of Yang & Hu (2021) reduced to the simplest possible falsifiable form (one figure, three verdict bands).

The v4.7 architecture's μP claim is asserted in Chapter 1.5 contribution 4 加固协议 C ("启用 Tensor Programs / μP 完整版本"), Chapter 4 architecture §501 (`mup_config.py` 存在的位置), Chapter 5 §5.10.2 (步骤 1-4 协议), and Chapter 6 §6.2.5 (主对比报告同时给出 base LR 与 Medium 配置实际 LR). All four references treat the claim as an unconditional architectural property of the v4 DualHyperNetwork. A claim without a check is a claim auditors mistrust; the 18-run self-test fits the GPU budget (36 GPU-hr ≈ 4% of the 850 GPU-hr pkg-08 total per README) and is large enough to refute the hypothesis cleanly.

The deliverable is **one file**:

```
hyper_mve/eval/mup_verification.py     — entry point + verdict computation + figure generation
```

Plus three filesystem outputs after a successful run:

```
runs/_mup_selftest/<config_hash>/mup_selftest.png    — headline figure (LR vs final loss, per width)
runs/_mup_selftest/<config_hash>/mup_selftest.csv    — 18-row raw losses table
runs/_mup_selftest/<config_hash>/mup_selftest.json   — MupSelftestVerdict dataclass dump
```

The grid itself is materialised through the sweep harness (spec 05 §10.3) as a `SweepConfig` with `ablation_cell_id="mup_verification"`, so the 18 rows land on `runs/registry.jsonl` with that cell id and the post-hoc verdict step reads them through `pandas.read_json(lines=True)` filtered on the cell id (§3.5).

---

## 2. Experiment design

### 2.1 Grid composition (the locked 18-row plan)

The grid is the **outer product** of three small axes:

```
widths   = (128, 256)                    # 2 widths; "base" and "target" in μP terminology
LRs      = (3e-4, 1e-3, 3e-3)            # 3 LRs; half-octave log-spaced on the typical hyper range
seeds    = (0, 1, 2)                     # 3 seeds
cells    = widths × LRs × seeds = 2 × 3 × 3 = 18 rows
```

The placeholder width values `(128, 256)` are **proposed** for the LR-doubling test; the actual v4 default at `hyper_mve/configs/model_config.py:33` is `latent_dim: int = 64` (Easy and Medium presets inherit this; neither preset overrides `model.latent_dim`). **Implementation phase MUST confirm** whether to (a) override `model.latent_dim ∈ {128, 256}` per sweep row to exercise the proposed widths, or (b) re-anchor the grid to `(64, 128)` to match the as-shipped Easy default. The implementation choice is logged in `mup_selftest.json` provenance (`widths` field on `MupSelftestVerdict`, §3.4) so the verdict record is unambiguous. The factor-of-2 doubling between the two widths is the canonical LR-doubling test step regardless of which absolute values are chosen.

The LR grid `(3e-4, 1e-3, 3e-3)` is centred at `1e-3` — an order of magnitude above the codebase's MuZero-trainer default `lr: float = 1e-4` at `hyper_mve/configs/train_config.py:32`. The hyper-net families in v4.7 are typically run at higher LR than the encoder trunk per the v4-opt 2026-06 optimisation chronicle (`docs/Hyper_MuZero_v4_chronicle.md` if it exists, or the `BeliefGradGating` pre-5K freeze rationale in `pkg-04`). **Implementation phase MUST confirm** the actual hyper-net LR in production runs and re-centre the grid if it differs from `1e-3` by more than one grid step (≈ factor 3.16). The half-octave grid spacing ratio is `~3.16×` per LR step (one half-decade); on this grid, "one octave" is operationalised as one grid-step (per Lock 2 note above). The textbook factor-of-2 LR-doubling test is acknowledged as a stricter follow-up sweep beyond the 18-row budget.

The seed values `(0, 1, 2)` are deliberately small integers — these are the canonical first three seeds used everywhere in the codebase (worker collection, eval, regret ceiling cache) and using them keeps the cross-cache provenance traceable. Seed 3 and 4 (the rest of the regret-ceiling's 5-seed protocol) are intentionally **not** used here — the self-test wants the smallest robust SEM (3 seeds is sufficient for a verdict; 5 would inflate the budget by 67% without improving the per-seed best-LR identification).

### 2.2 Preset and training duration

The preset is **Easy** (`N=2`, `K=8`) — the smallest preset to fit the budget. Easy is also the only preset where a 50_000-env-step training run reliably reaches plateau on Hyper variants (Medium needs ~150_000 steps per the optimisation-phase chronicle; using Medium would 3× the budget and breach the 36 GPU-hr Lock 1). The training duration is fixed at:

```python
_TOTAL_ENV_STEPS = 50_000           # see §6 per-impl constants
```

Held-out loss evaluation cadence: every `5_000` env-steps, evaluate on a fixed eval set of 30 episodes at `c=0.5` (the centre of the typical c-grid; same value used in the existing `cfg.eval.eval_c_grid` per `eval_config.py`). The final held-out loss for verdict aggregation is the **mean over the last 3 eval points** (i.e. eval at steps 40_000 / 45_000 / 50_000), to smooth the noisy last-step fluctuation and reflect a near-converged regime. Cadence of 5K steps gives `50_000 / 5_000 = 10` eval points per run; the last 3 cover steps 40K-50K so the verdict ignores transient pre-convergence noise.

### 2.3 Variant pinned to `"hyper"`

The self-test runs only the `"hyper"` variant (the DualHyperNetwork v2 model defined at pkg-04 spec 02 + `hyper_mve/models/hyper_muzero_model.py`). Internal baselines (`input_wide` / `input_deep` / `ma_muzero` / `no_belief` / `rewardhead_explicit_type`) do **not** participate — they are not the subject of the μP claim, which is specifically about the DualHyperNetwork's hyper-net → functional-net path (pkg-04 design §NG3 explicitly defers μP integration to pkg-07; pkg-07 deferred it to the present spec at pkg-08 spec 04 because the experimental falsifier belongs in the eval-and-ablation package). External baselines (Tier-1 MAPPO / QMIX / MA-MuZero-GH + MAMBA) likewise do not participate — they are not hyper-net based and the μP claim is irrelevant to them.

### 2.4 Held-out loss metric (the verdict input)

The "held-out loss" used for verdict aggregation is the **mean episode return** at `c=0.5`, sign-flipped to convert it to a loss-like scalar (lower-is-better). Sourced as `-EvalReport.return_per_c[0.5]` from each row's `EvalReport`. The reason for using the return rather than the model's training loss (`loss_total`):

1. Training loss in v4 is a composite of policy / value / reward / projection losses with width-dependent normalisation (per `loss_composition.py` and the v4-opt 2026-06 vectorised collection); comparing training losses across widths confounds μP with loss-normalisation scaling.
2. Episode return at `c=0.5` is **width-agnostic** in unit and interpretable as "task performance"; the LR that maximises task return is the canonical "best LR" in the policy-learning literature.
3. The unified evaluator already produces `EvalReport.return_per_c[0.5]` for every row (spec 01 §3 schema slot 12), so no new metric plumbing is required.

A test in §7 asserts the metric source is `EvalReport.return_per_c[0.5]` and not the trainer's loss dict.

---

## 3. PASS / WARN / FAIL verdict (the quantitative rule)

### 3.1 Setup

Let the 18 rows produce a mapping:

```
losses[(seed, width, lr)] = -EvalReport.return_per_c[0.5]   # the held-out loss (lower=better)
```

with `seed in {0,1,2}`, `width in {128,256}`, `lr in {3e-4, 1e-3, 3e-3}`. For each (seed, width) pair, compute the **best LR** as:

```python
best_lr_per_seed_width = {
    (seed, width): min(lrs, key=lambda lr: losses[(seed, width, lr)])
    for seed in (0, 1, 2)
    for width in (128, 256)
}
```

This yields 6 (seed, width) → best_lr mappings. The verdict aggregates these.

### 3.2 Octave distance on the LR grid

The 3 LRs `(3e-4, 1e-3, 3e-3)` are half-octave-spaced (each step is a factor of ~3.16, close to one octave's factor of ~3). For the LR-doubling test, "one octave" is operationalised as **one step on the grid**:

```python
def grid_octave_distance(lr1: float, lr2: float) -> int:
    lrs_sorted = (3e-4, 1e-3, 3e-3)
    return abs(lrs_sorted.index(lr1) - lrs_sorted.index(lr2))
```

Distance `0` = identical best-LR. Distance `1` = one octave (factor ~3) different. Distance `2` = two octaves apart (the two extremes of the grid).

### 3.3 Verdict truth table

For each seed, compute the per-width best-LR pair `(best_lr@128, best_lr@256)` and the octave distance:

```python
delta_octaves_per_seed = {
    seed: grid_octave_distance(
        best_lr_per_seed_width[(seed, 128)],
        best_lr_per_seed_width[(seed, 256)],
    )
    for seed in (0, 1, 2)
}
```

The verdict aggregates the 3 per-seed deltas:

| Condition | Verdict |
|---|---|
| `all(delta == 0 for delta in delta_octaves_per_seed.values())` | **PASS** |
| `sum(1 for delta in delta_octaves_per_seed.values() if delta <= 1) >= 2` AND verdict is not PASS | **WARN** |
| otherwise (i.e. ≥2 seeds have delta >= 2, OR mixed pattern that does not hit the WARN majority) | **FAIL** |

The PASS rule is the strictest possible reading of LR-doubling: every seed agrees the same LR is best at both widths. The WARN rule relaxes to "at most one octave drift on a 2/3 majority of seeds" — this captures the case where two seeds confirm μP within one octave but one seed shows a larger drift (still publishable with a footnote). FAIL captures the case where the drift is larger than one octave on a majority of seeds, which is the empirical falsifier of the μP claim at this preset / width range.

### 3.4 The `MupSelftestVerdict` dataclass

The verdict is emitted as a `@dataclass(frozen=True) MupSelftestVerdict` (NOT an `EvalReport` — the μP self-test is structurally different from a single-config evaluation and the schema would not be a clean fit):

```python
# hyper_mve/eval/mup_verification.py

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping


@dataclass(frozen=True)
class MupSelftestVerdict:
    """Verdict for the μP base-shape LR-doubling self-test (spec 04).

    Emitted exactly once per (config_hash) run of the 18-row grid. The
    18 rows are sourced from runs/registry.jsonl filtered on
    ablation_cell_id="mup_verification".
    """

    # === Identity ===
    config_hash: str                                # SHA1 of (variant="hyper", env_cfg, model_cfg, train_cfg) at base_width; primary key
    widths: tuple[int, ...]                         # (128, 256) — locked by §2.1
    lrs: tuple[float, ...]                          # (3e-4, 1e-3, 3e-3) — locked by §2.1
    seeds: tuple[int, ...]                          # (0, 1, 2) — locked by §2.1
    preset: Literal["easy"]                         # always "easy" per §2.2

    # === Raw losses (the verdict input) ===
    raw_losses: Mapping[tuple[int, int, float], float]
                                                    # (seed, width, lr) -> held-out loss = -EvalReport.return_per_c[0.5]

    # === Derived (the verdict logic output) ===
    best_lr_per_seed_width: Mapping[tuple[int, int], float]
                                                    # (seed, width) -> argmin-loss LR
    delta_octaves_per_seed: Mapping[int, int]       # seed -> grid_octave_distance(best@128, best@256)
    verdict: Literal["PASS", "WARN", "FAIL"]

    # === Provenance ===
    plot_path: str                                  # absolute path to mup_selftest.png
    csv_path: str                                   # absolute path to mup_selftest.csv
    computed_at_iso8601: str
    git_sha: str                                    # 40-char full SHA at verdict computation time

    # === Schema sentinel ===
    schema_version: str = "pkg08-spec04-v1"         # bumped only on a synchronous spec 04 + spec 08 edit
```

The dataclass is **hashable + JSON-serialisable** via the same conventions as `EvalReport` (spec 01 §3.3): `Mapping[...]` wrapped in `MappingProxyType`, `tuple` keys for hashable mappings, `float`/`int` only for numeric fields. The dump path is `runs/_mup_selftest/<config_hash>/mup_selftest.json` (§1 deliverables list).

**Tuple-key JSON encoding (mandatory).** Standard `json.dump` cannot round-trip `tuple` keys (only `str`/`int`/`float`/`bool`/`None`). The implementation encodes tuple keys as `"|"`-joined string composites on dump and splits on `"|"` on load, mirroring the EvalReport convention noted in spec 01 §3.3:

- `raw_losses: Mapping[tuple[int, int, float], float]` — key `(seed, width, lr)` → JSON string `f"{seed}|{width}|{lr}"`; load splits and casts to `(int(seed), int(width), float(lr))`.
- `best_lr_per_seed_width: Mapping[tuple[int, int], float]` — key `(seed, width)` → JSON string `f"{seed}|{width}"`; load splits and casts to `(int(seed), int(width))`.
- `delta_octaves_per_seed: Mapping[int, int]` — int key, no encoding needed (JSON allows int keys via `str(int)` conversion on dump; load `int()` casts back).

The encoder lives in `hyper_mve/eval/_serde.py` (sibling module) as `dump_mup_verdict(verdict, path)` + `load_mup_verdict(path) -> MupSelftestVerdict`. A test in §7 round-trips a verdict through dump+load and asserts byte-stable re-emission.

### 3.5 Aggregation pipeline

```python
# hyper_mve/eval/mup_verification.py (sketch)

import pandas
from hyper_mve.eval.eval_report import EvalReport


def aggregate_verdict(
    registry_path: pathlib.Path = pathlib.Path("runs/registry.jsonl"),
    config_hash: str | None = None,
) -> MupSelftestVerdict:
    """Read the 18 mup_verification rows from registry.jsonl and emit the
    verdict. If config_hash is None, the latest mup_verification cell with
    18 completed rows is used.
    """
    df = pandas.read_json(registry_path, lines=True)
    df = df[df.ablation_cell == "mup_verification"]
    df = df[df.status == "completed"]
    if config_hash is None:
        # Order by start time per §9 edge case 2 — registry-write order is
        # NOT guaranteed to coincide with started_at_iso8601 under subprocess
        # concurrency (sweep harness writes rows as they complete).
        config_hash = df.sort_values("started_at_iso8601").config_hash.iloc[-1]
    df = df[df.config_hash == config_hash]
    assert len(df) == 18, f"Expected 18 rows; got {len(df)}"

    # Load each EvalReport from eval_report_path; extract return_per_c[0.5].
    raw_losses: dict[tuple[int, int, float], float] = {}
    for _, row in df.iterrows():
        report = _load_eval_report(row.eval_report_path)
        loss = -float(report.return_per_c[0.5])
        seed = int(row.seed)
        width = int(_decode_width_from_overrides(row))      # parses overrides["model.latent_dim"]
        lr = float(_decode_lr_from_overrides(row))          # parses overrides["train.lr"]
        raw_losses[(seed, width, lr)] = loss

    # Compute best LR per (seed, width).
    best_lr_per_seed_width: dict[tuple[int, int], float] = {
        (seed, width): min(_LRS, key=lambda lr: raw_losses[(seed, width, lr)])
        for seed in _SEEDS
        for width in _WIDTHS
    }

    # Compute octave deltas + verdict.
    delta_octaves_per_seed: dict[int, int] = {
        seed: grid_octave_distance(
            best_lr_per_seed_width[(seed, _WIDTHS[0])],
            best_lr_per_seed_width[(seed, _WIDTHS[1])],
        )
        for seed in _SEEDS
    }
    verdict = _decide_verdict(delta_octaves_per_seed)

    # Render figure + CSV.
    plot_path = _render_lr_vs_loss_figure(raw_losses, config_hash)
    csv_path = _write_csv(raw_losses, config_hash)

    return MupSelftestVerdict(
        config_hash=config_hash,
        widths=_WIDTHS,
        lrs=_LRS,
        seeds=_SEEDS,
        preset="easy",
        raw_losses=MappingProxyType(raw_losses),
        best_lr_per_seed_width=MappingProxyType(best_lr_per_seed_width),
        delta_octaves_per_seed=MappingProxyType(delta_octaves_per_seed),
        verdict=verdict,
        plot_path=str(plot_path),
        csv_path=str(csv_path),
        computed_at_iso8601=_now_iso(),
        git_sha=_git_sha_40char(),
    )
```

The aggregation function is **post-hoc and idempotent**: it reads from `runs/registry.jsonl` and produces `MupSelftestVerdict` deterministically given the 18 rows. Re-running the aggregation on the same 18 rows produces a byte-identical verdict (modulo `computed_at_iso8601` and `git_sha`).

---

## 4. Failure protocol (Lock 3 codified)

This section enumerates the publication consequences of each verdict, locking the editorial behaviour so the implementation does not become an arbiter of paper content.

### 4.1 PASS

- **Main text**: Chapter 5 §5.10.2 and Chapter 6 §6.2.5 retain the μP claim as written.
- **Appendix**: not modified.
- **Citation**: a single sentence appended to Ch5 §5.10.2 step-4 paragraph ("μP base-shape LR-doubling self-test passed at 2 widths × 3 LRs × 3 seeds; verdict logged at `runs/_mup_selftest/<config_hash>/mup_selftest.json`"). Same sentence appended to Ch6 §6.2.5 末段.
- **Figure**: `mup_selftest.png` is **optional** — main text shows it only if Ch6 §6.2.5 has the figure budget. Default = appendix-only.

### 4.2 WARN

- **Main text**: claim retained, but with a footnote citing the appendix self-test. Footnote template:
  > "$^N$ A formal μP base-shape LR-doubling self-test (2 widths × 3 LRs × 3 seeds, Easy preset) showed within-one-octave drift on the best-LR identification across widths for $K/3$ seeds (verdict: WARN). See Appendix §M.X for the full disclosure."
  Replace `K` with the actual seed count (1 or 2) that drifted by ≥1 octave, and `M.X` with the appendix subsection number.
- **Appendix**: subsection "μP self-test result (warn)" with the full 18-row CSV table + figure + 1-paragraph commentary explaining the within-octave drift.
- **Figure**: appendix-mandatory.

### 4.3 FAIL

- **Main text**: μP claim is **dropped** from Chapter 5 §5.10.2 ("启用 Tensor Programs / μP 完整版本") and Chapter 6 §6.2.5 ("μP 学习率对齐协议"). The replacement paragraph in Ch6 reads:
  > "A formal μP base-shape LR-doubling self-test (2 widths × 3 LRs × 3 seeds, Easy preset) showed > 1-octave best-LR drift across widths on a majority of seeds (verdict: FAIL). We therefore use per-baseline independent LR sweep (Roadmap §2.3 降级方案) instead of μP scaling for all main-table runs. See Appendix §M.X for the full self-test disclosure."
- **Appendix**: subsection "μP self-test result (failed)" with:
  - the full 18-row CSV,
  - the headline figure,
  - a 1-paragraph commentary on the alternate parametrization used (per-baseline LR sweep, with the actual LR values selected for each main-table variant cited),
  - the `MupSelftestVerdict` JSON dump as a fenced code block for reviewer audit.
- **Roadmap impact**: docs/Hyper_MuZero_v4_Roadmap.md §2.3 (the "μP 库可用性确认" + 降级方案 reference) is the canonical fallback documentation; the FAIL appendix points to it.

### 4.4 Editorial gate (one-shot)

The verdict is computed **exactly once** per pkg-08 implementation cycle. The 18 rows are run on a single sweep harness invocation, the verdict is emitted, and one of the three editorial paths is taken. **Re-running the 18 rows to "try again"** is explicitly disallowed by Lock 1 — if the verdict is FAIL, the failure is the result, not a signal to expand the grid. (An appendix-disclosed follow-up sweep with different widths or LRs is permissible after the present spec's verdict is in, but that is outside the contract of this spec.)

---

## 5. Output figure

### 5.1 Figure spec

A **single figure**, titled "LR vs final held-out loss, per width":

- **X-axis**: log-LR (`matplotlib` `xscale="log"`), tick labels at the 3 grid LRs.
- **Y-axis**: final held-out loss (`-EvalReport.return_per_c[0.5]` mean over last 3 eval points).
- **Lines**: one per width (`width=128` solid blue, `width=256` dashed red — matplotlib default style cycle, kept simple).
- **Error bands**: `±1 SEM` (filled, 0.2 alpha) drawn **per width-line**, computed from the 3-seed std at each LR divided by `sqrt(3)`. The visualisation choice is **one envelope per width** (aggregated over 3 seeds at each LR), NOT per seed — individual seeds are not plotted as separate lines.
- **Markers**: dots at each (width, lr) cell, sized proportional to the inverse seed-std (smaller = noisier).
- **Annotations**: a text box top-right with the verdict (`PASS` / `WARN` / `FAIL`) and the per-seed best-LR pairs.
- **Backend**: matplotlib `Agg` (headless; same backend the rest of pkg-08 uses per spec 07 §3 C8-ABL-PLOT1).

### 5.2 File paths

```
runs/_mup_selftest/<config_hash>/mup_selftest.png   # PNG, 300 DPI, paper-grade
runs/_mup_selftest/<config_hash>/mup_selftest.csv   # 18-row table; columns = (seed, width, lr, final_loss)
runs/_mup_selftest/<config_hash>/mup_selftest.json  # MupSelftestVerdict dump
```

The underscore-prefixed `_mup_selftest` directory parallels `_oracle_ceilings` (spec 02 §4.3) — both are content-addressed self-test caches that are not regular `runs/` rows.

### 5.3 Entry point

```bash
python -m hyper_mve.eval.mup_verification --run        # dispatches the 18-row SweepConfig
python -m hyper_mve.eval.mup_verification --aggregate  # reads registry.jsonl, emits verdict
python -m hyper_mve.eval.mup_verification              # auto: --run if no rows present, else --aggregate
```

The `__main__` block in `mup_verification.py` parses `--run` / `--aggregate` flags. The `--run` path constructs the `SweepConfig` per spec 05 §10.3 and calls `run_sweep(sweep_cfg)`. The `--aggregate` path calls `aggregate_verdict()` (§3.5). The auto-mode is convenience for the implementation phase.

---

## 6. Module-level constants (NOT new cfg fields)

### 6.1 Honest declaration of where the 5 parameters live

The README §"5 新 cfg 字段穷举" (and the design.md §4 D10 5-field穷举) lists exactly **5** new cfg fields for pkg-08: `c_visible` / `randomize_order` / `mve_joint_enumerate` / `eval_planner_mode` / `eval_use_planner_direct_inference`. None of them are μP-self-test parameters. This spec therefore **does NOT** add any new cfg fields. The 5 μP-self-test parameters (`widths`, `lrs`, `seeds`, `total_env_steps`, `eval_episodes_per_point`) live as **module-level constants** in `hyper_mve/eval/mup_verification.py`, mirroring the pkg-07 spec 05 §11 "per-impl tuning constants (NOT in cfg.baselines)" pattern. The rationale is the same: these constants are **impl-internal**, not cross-spec contract; they should not bloat `cfg.eval` because (i) no other spec consumes them, (ii) no sweep override changes them (Lock 1 forbids expansion), and (iii) `cfg.eval` is part of the V4Config schema and adding 5 self-test-only fields would violate the canonical 5-field穷убер.

### 6.2 The constants block (verbatim source code template)

```python
# hyper_mve/eval/mup_verification.py — module-level constants (NOT cfg fields)
# 
# Mirrors the per-impl-constants pattern from pkg-07 spec 05 §11.
# Per pkg-08 README "5 新 cfg 字段穷举", no new cfg fields are added by this spec.
# These constants are impl-internal; they are NOT cross-spec contract.

_WIDTHS: tuple[int, int] = (128, 256)               # base_width, target_width
_LRS: tuple[float, float, float] = (3e-4, 1e-3, 3e-3)   # half-octave log-spaced
_SEEDS: tuple[int, int, int] = (0, 1, 2)            # canonical first 3 seeds
_TOTAL_ENV_STEPS: int = 50_000                      # Easy preset training duration
_EVAL_EPISODES_PER_POINT: int = 30                  # episodes per held-out eval at c=0.5
_EVAL_CADENCE_ENV_STEPS: int = 5_000                # held-out eval every 5K env-steps
_LAST_N_EVAL_POINTS_FOR_LOSS: int = 3               # mean over last 3 eval points = final loss
_HELD_OUT_C: float = 0.5                            # fixed c for held-out eval
```

These constants are read by §3.5 `aggregate_verdict()` and by the `--run` entry point's `SweepConfig` construction. Tests in §7 import the constants directly and assert the grid math (`2 × 3 × 3 = 18`).

### 6.3 Alternative considered + rejected: spec-08 §5 cfg-block extension

A "mup_selftest_block" sub-dataclass on `cfg.eval` containing 5 new fields (`mup_selftest_widths` / `mup_selftest_lrs` / `mup_selftest_seeds` / `mup_selftest_total_env_steps` / `mup_selftest_eval_episodes_per_point`) was considered. It was **rejected** for three reasons:

1. **Canonical 5-field穷举 violation**: pkg-08 design D10 + README §"5 新 cfg 字段穷举" enumerate exactly 5 new cfg fields; adding 5 more would silently expand the contract and force pkg-01 spec 05 to register an extra block. Spec 08 §5 would need to declare them, breaking the "additive, never silent" SDD discipline.
2. **No cross-spec consumer**: only `mup_verification.py` reads these values. No other spec / sweep / training row touches them. Cfg fields are appropriate when they cross spec boundaries; module-level constants are appropriate when they do not.
3. **Lock 1 forbids overrides**: the grid is fixed by spec design (no per-run override is allowed). If they were cfg fields, sweep YAMLs could in principle override them, and a future contributor might expand the grid to "make μP look better" — defeating the falsifier protocol. Module-level constants are visible-in-source-code only and require an explicit spec edit to change.

The pattern matches pkg-07 spec 05 §11 (MAPPO's `_DEFAULT_BATCH_SIZE`, `_DEFAULT_K_EPOCHS`, etc. — 20 per-impl constants none of which are on `cfg.baselines`).

---

## 7. Test contract — C8-EVAL-MUP1 enforcement

Located under `tests/eval/`. Each test below corresponds to a specific guarantee of this spec.

### 7.1 `test_mup_selftest_emits_verdict_dataclass`

Smoke test that runs a **toy 2-run subset** (1 width × 2 LRs × 1 seed = 2 rows) through `aggregate_verdict` and asserts the returned object is a `MupSelftestVerdict` with `verdict in ("PASS", "WARN", "FAIL")`. Uses a stubbed `EvalReport` factory (no real training).

```python
def test_mup_selftest_emits_verdict_dataclass(tmp_path):
    from hyper_mve.eval.mup_verification import (
        MupSelftestVerdict,
        aggregate_verdict,
    )
    # Build a 2-row toy registry with stubbed EvalReports.
    # NOTE: rows are dict literals, NOT tuples — keyword-arguments inside
    # a tuple literal are a SyntaxError in Python.
    registry_path = _make_toy_registry(
        tmp_path,
        rows=[
            dict(seed=0, width=128, lr=1e-3, return_at_c0p5=10.0),
            dict(seed=0, width=128, lr=3e-3, return_at_c0p5= 8.0),
        ],
    )
    # Patch aggregate_verdict's row-count assertion to accept 2 rows for the toy case.
    verdict = aggregate_verdict(
        registry_path=registry_path,
        _expected_row_count=2,
    )
    assert isinstance(verdict, MupSelftestVerdict)
    assert verdict.verdict in ("PASS", "WARN", "FAIL")
    assert verdict.schema_version == "pkg08-spec04-v1"
```

### 7.2 `test_mup_selftest_pass_criterion`

Synthetic data where the best LR is identical across both widths for all 3 seeds → asserts verdict is PASS.

```python
def test_mup_selftest_pass_criterion(tmp_path):
    # Construct 18 synthetic losses where lr=1e-3 is the best at both widths for all seeds.
    raw_losses = {}
    for seed in (0, 1, 2):
        for width in (128, 256):
            raw_losses[(seed, width, 3e-4)] = 10.0
            raw_losses[(seed, width, 1e-3)] =  5.0     # best at both widths
            raw_losses[(seed, width, 3e-3)] = 10.0
    registry_path = _materialise_registry(tmp_path, raw_losses)
    verdict = aggregate_verdict(registry_path=registry_path)
    assert verdict.verdict == "PASS"
    assert all(d == 0 for d in verdict.delta_octaves_per_seed.values())
```

### 7.3 `test_mup_selftest_warn_criterion`

Synthetic data where the best LR differs by exactly one octave for ≥2 of 3 seeds → asserts verdict is WARN.

```python
def test_mup_selftest_warn_criterion(tmp_path):
    raw_losses = {}
    # Seeds 0 and 1: best LR at width 128 is 3e-4; at width 256 is 1e-3 → delta=1 octave.
    # Seed 2: best LR at both widths is 1e-3 → delta=0.
    for seed, best_lr_pair in [(0, (3e-4, 1e-3)), (1, (3e-4, 1e-3)), (2, (1e-3, 1e-3))]:
        for width, best_lr in zip((128, 256), best_lr_pair):
            for lr in (3e-4, 1e-3, 3e-3):
                raw_losses[(seed, width, lr)] = 5.0 if lr == best_lr else 10.0
    registry_path = _materialise_registry(tmp_path, raw_losses)
    verdict = aggregate_verdict(registry_path=registry_path)
    assert verdict.verdict == "WARN"
    # 2 seeds at delta=1, 1 seed at delta=0 → majority within 1 octave but not all-zero.
    assert verdict.delta_octaves_per_seed[0] == 1
    assert verdict.delta_octaves_per_seed[1] == 1
    assert verdict.delta_octaves_per_seed[2] == 0
```

### 7.4 `test_mup_selftest_fail_criterion`

Synthetic data where the best LR differs by 2 octaves for ≥2 of 3 seeds → asserts verdict is FAIL.

```python
def test_mup_selftest_fail_criterion(tmp_path):
    raw_losses = {}
    # Seeds 0 and 1: best LR at width 128 is 3e-4; at width 256 is 3e-3 → delta=2 octaves.
    # Seed 2: best LR at both widths is 1e-3 → delta=0.
    for seed, best_lr_pair in [(0, (3e-4, 3e-3)), (1, (3e-4, 3e-3)), (2, (1e-3, 1e-3))]:
        for width, best_lr in zip((128, 256), best_lr_pair):
            for lr in (3e-4, 1e-3, 3e-3):
                raw_losses[(seed, width, lr)] = 5.0 if lr == best_lr else 10.0
    registry_path = _materialise_registry(tmp_path, raw_losses)
    verdict = aggregate_verdict(registry_path=registry_path)
    assert verdict.verdict == "FAIL"
    assert verdict.delta_octaves_per_seed[0] == 2
    assert verdict.delta_octaves_per_seed[1] == 2
```

### 7.5 `test_mup_selftest_plot_renders_headless`

Asserts the figure rendering uses the matplotlib `Agg` backend (headless) and produces a non-empty PNG file at `tmp_path / "mup_selftest.png"`.

```python
def test_mup_selftest_plot_renders_headless(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    from hyper_mve.eval.mup_verification import _render_lr_vs_loss_figure
    raw_losses = {(s, w, lr): 5.0 + s for s in (0,1,2) for w in (128,256) for lr in (3e-4,1e-3,3e-3)}
    plot_path = _render_lr_vs_loss_figure(raw_losses, config_hash="test_hash", out_dir=tmp_path)
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
    import matplotlib
    assert matplotlib.get_backend().lower() == "agg"
```

### 7.6 `test_mup_selftest_csv_schema_locked`

Asserts the CSV columns are exactly `(seed, width, lr, final_loss)` and that the row count is 18.

```python
def test_mup_selftest_csv_schema_locked(tmp_path):
    from hyper_mve.eval.mup_verification import _write_csv
    raw_losses = {(s, w, lr): 5.0 for s in (0,1,2) for w in (128,256) for lr in (3e-4,1e-3,3e-3)}
    csv_path = _write_csv(raw_losses, config_hash="test_hash", out_dir=tmp_path)
    import csv
    with csv_path.open() as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = list(reader)
    assert header == ["seed", "width", "lr", "final_loss"]
    assert len(rows) == 18
```

### 7.7 `test_mup_selftest_uses_evalreport_return_per_c_metric`

Asserts the held-out loss is sourced from `EvalReport.return_per_c[0.5]` (sign-flipped), not from any trainer loss dict. This is the §2.4 guarantee.

```python
def test_mup_selftest_uses_evalreport_return_per_c_metric(tmp_path):
    from hyper_mve.eval.mup_verification import _extract_held_out_loss
    from hyper_mve.eval.eval_report import EvalReport
    report = _make_stub_evalreport(return_per_c={0.5: 12.0})
    loss = _extract_held_out_loss(report)
    assert loss == pytest.approx(-12.0)    # sign-flipped to lower-is-better
```

---

## 8. Integration hooks (cross-spec)

| Consumer | Consumed from this spec | Use |
|---|---|---|
| **spec 01 unified-evaluator** | `EvalReport.return_per_c[0.5]` per row | Each of the 18 rows produces an `EvalReport` via the standard unified evaluator path; the μP self-test reads only one field (the return at c=0.5). No new schema slot is added to `EvalReport`. |
| **spec 05 sweep-harness-and-run-registry** | `SweepConfig` materialised at §10.3 of spec 05 | The 18 rows land on `runs/registry.jsonl` with `ablation_cell="mup_verification"`. The harness needs no new machinery — this spec is a passive consumer of the existing sweep + register infrastructure. The grid is constructed inside `mup_verification.py --run` and passed to `run_sweep(sweep_cfg)`. |
| **spec 07 statistics-and-comparison** | matplotlib Agg backend convention | The figure-rendering subroutine follows the same headless-matplotlib convention as spec 07's `compare.py`. No data is passed through `stats.py`; the verdict computation is self-contained in `mup_verification.py`. |
| **spec 08 integration-contracts** | NO new cfg fields declared in this spec | Spec 08 §5 (5 新 cfg 字段穷举) does NOT list any μP-self-test fields — they live as module-level constants per §6. Spec 08 §6 drift detector greps the anchors in §10 of this spec to verify no cfg-block expansion has silently appeared. The `MupSelftestVerdict` dataclass IS a new schema, but it lives in `eval/mup_verification.py`, not in `eval_report.py`, and is not re-mirrored in spec 08 §3 (which is the `EvalReport` lock only). |
| **pkg-04 design §NG3 (μP base shape)** | The μP claim itself | Pkg-04 explicitly defers μP integration to pkg-07; pkg-07 deferred it again, and the falsification protocol lands here in pkg-08 spec 04. The verdict produced here is the empirical test of the pkg-04 design's μP-compatibility assertion. |
| **`hyper_mve/configs/mup_config.py`** | `MupConfig` dataclass | The placeholder dataclass on `V4Config.mup` (currently `enabled=False`, `base_shape_path=None`, `lr_scaling_enabled=False`) is the integration surface. The self-test does NOT require `mup.enabled=True` — the test runs against the v4 DualHyperNetwork's default initialisation (which is claimed to be μP-base-shape compatible per the architectural design, regardless of whether the explicit `mup` library is wired in). If the verdict is PASS, the μP claim survives without needing the `mup` library at all (the architecture is intrinsically μP-compatible). If the verdict is FAIL, dropping the claim removes the need for `MupConfig` altogether, but the dataclass is kept as a no-op placeholder for forward compatibility. |
| **Roadmap §2.3 降级方案** | The alternate parametrization on FAIL | Per-baseline independent LR sweep is the documented fallback; the FAIL appendix points to Roadmap §2.3 verbatim. |
| **Ch5 §5.10.2 / Ch6 §6.2.5** | The μP claim's main-text location | The PASS / WARN / FAIL editorial protocol (§4) directly governs what survives in these two sections. |

---

## 9. Edge cases

| Case | Behaviour |
|---|---|
| Registry has fewer than 18 `mup_verification` rows | `aggregate_verdict()` raises `AssertionError("Expected 18 rows; got N")`. The implementation phase MUST run the full 18 before aggregation. |
| Two `mup_verification` cells with different `config_hash` (e.g. the user re-ran with a different model config) | `aggregate_verdict()` filters by `config_hash`; if `config_hash=None`, the latest cell is used (per `started_at_iso8601` ordering on the registry). |
| One of the 18 rows has `status="failed"` | The row is excluded from aggregation; if fewer than 18 completed rows remain, the assertion in case 1 fires. The implementation must re-run the failed row before aggregation. |
| Tied best-LR within a (seed, width) cell (i.e. two LRs produce identical losses) | `min(_LRS, key=...)` returns the **first** matching LR (lowest LR by the tuple order). A test in §7 does not currently exercise this; a follow-up test `test_mup_selftest_tied_best_lr_picks_lowest` may be added during implementation if a real run shows ties. The verdict logic is unaffected because the next step compares LR-grid indices, not the LR values themselves. |
| LR grid is overridden by a future user (e.g. `_LRS = (1e-4, 1e-3, 1e-2)` to get a wider span) | The module-level constants are visible-in-source-code only; a future user editing them is making a spec-level change, not a runtime config tweak. The drift detector in spec 08 §6 greps for `_WIDTHS`, `_LRS`, `_SEEDS` literals and warns on mutation that is not accompanied by a spec edit. |
| `EvalReport.return_per_c` does not contain `0.5` (e.g. user's `cfg.eval.zero_shot_test_c` was overridden) | `_extract_held_out_loss` raises `KeyError`. The implementation phase must use the default `zero_shot_test_c` grid containing `0.5`; any override that removes `0.5` from the test grid invalidates the self-test. |
| Verdict is FAIL but the user wants to "try again with different seeds" | Disallowed by Lock 1 + §4.4 editorial gate. The verdict is one-shot. An appendix-disclosed follow-up sweep is permissible but is outside the contract of this spec. |
| The `mup` library (Microsoft mup) is not installed | Irrelevant — the self-test does not depend on the `mup` library. It tests the **architectural** μP-compatibility of the v4 DualHyperNetwork at the default initialisation. The `mup` library is only consumed by Ch5 §5.10.2 步骤 1 if the verdict is PASS and the implementation chooses to additionally wire in `mup` for production runs. The fallback per Roadmap §2.3 also does not depend on `mup`. |

---

## 10. Cross-references

### Upstream anchors

- **pkg-08 design.md §2.2 G11** — μP base-shape LR-doubling self-test (this spec is the SDD-level realisation of G11).
- **pkg-08 design.md §4 D2** — `EvalReport` schema; the μP self-test consumes only `EvalReport.return_per_c[0.5]` and does not extend the schema.
- **pkg-08 design.md §4 D10 末段** — per-impl tuning constants pattern (the rationale for §6 of this spec).
- **pkg-08 README C8-EVAL-MUP1** — acceptance criterion: "μP 自检 18 runs 跑出（2 widths × 3 LRs × 3 seeds, Easy preset）；若 LR-doubling 不成立 → 论文 drop μP 主张并 appendix 披露".
- **pkg-08 spec 01 §3.1** — `EvalReport.return_per_c` (slot 12); the μP self-test reads `return_per_c[0.5]` and sign-flips for loss.
- **pkg-08 spec 05 §10.3** — the 18-row `SweepConfig` materialisation that this spec consumes (`ablation_cell_id="mup_verification"`).
- **pkg-08 spec 05 §3.1** — `SweepConfig.ablation_cell_id` field; the cell-id channel for routing self-test rows.
- **pkg-08 spec 05 §4** — `RunRegistry` row schema; `ablation_cell` field is the filter key for `aggregate_verdict`.
- **pkg-07 spec 05 §11** — per-impl tuning constants pattern (the mirror reference for §6 of this spec).
- **pkg-04 design §NG3** — μP base shape integration is NOT implemented in pkg-04; deferred to pkg-07, deferred again to pkg-08 spec 04.
- **`hyper_mve/configs/mup_config.py`** — `MupConfig` placeholder dataclass on `V4Config.mup`; the integration surface (currently no-op).
- **`hyper_mve/configs/presets/easy.py`** — Easy preset (`N=2`, `K=8`) used by the self-test.
- **`hyper_mve/configs/train_config.py:32`** — `lr: float = 1e-4` (MuZero-trainer default; the §2.1 LR grid is centred an order of magnitude above this default per the hyper-net LR convention).
- **`hyper_mve/configs/model_config.py:33`** — `latent_dim: int = 64` (Easy / Medium presets inherit; the §2.1 width pin `(128, 256)` proposes overriding this per sweep row, subject to implementation-phase confirmation).
- **docs/Chapter5_Planner_Training_v4.md §5.10.2** — the μP claim's main-text location (subject to PASS/WARN/FAIL editorial protocol §4).
- **docs/Chapter6_Experiments_v4.md §6.2.5** — μP 学习率对齐协议 (same).
- **docs/Hyper_MuZero_v4_Roadmap.md §2.3** — μP 库可用性确认 + 降级方案 (the FAIL fallback).

### Downstream consumption (spec 04 → others)

- **spec 05** consumes the `SweepConfig` materialised by this spec's `--run` entry point.
- **spec 07** does NOT consume any field of `MupSelftestVerdict`; the figure rendering is self-contained in `mup_verification.py`.
- **spec 08** §6 drift detector greps the §11 anchors of this spec.
- **No new cfg field is contributed to spec 08 §5** (the 5 新 cfg 字段穷举 list is unchanged; this spec adds zero fields).

### Existing repo ground-truth anchors

- `hyper_mve/configs/mup_config.py:8-19` — `MupConfig @dataclass(frozen=True)` with 3 placeholder fields (`enabled=False`, `base_shape_path=None`, `lr_scaling_enabled=False`).
- `hyper_mve/configs/v4_config.py:12` — `from .mup_config import MupConfig` import surface.
- `hyper_mve/configs/v4_config.py:35` — `mup: MupConfig = field(default_factory=MupConfig)` on V4Config.
- `hyper_mve/configs/__init__.py:20` — `MupConfig` re-export.
- `hyper_mve/configs/presets/medium.py:14` + `medium.py:114` — `MupConfig` consumption (placeholder; no effect).
- `hyper_mve/training/evaluation.py:run_eval` — the inner eval loop the unified evaluator wraps; the μP self-test runs 18 invocations of this via the sweep harness, each producing one `EvalReport.return_per_c[0.5]` value.
- `hyper_mve/models/hyper_muzero_model.py` — the `HyperMuZeroModel` whose initialisation is the subject of the μP-compatibility claim.

---

## 11. Anchors (verbatim grep targets for spec 08 §6 drift detector)

- `pkg-08 spec 04 Lock 1: Self-contained 18-run experiment, fixed budget, no expansion (2 widths × 3 LRs × 3 seeds = 18 rows, Easy preset, 50_000 env-steps, 36 GPU-hr)`
- `pkg-08 spec 04 Lock 2: PASS / WARN / FAIL is QUANTITATIVE, not visual; grid-octave distance rule`
- `pkg-08 spec 04 Lock 3: Explicit failure protocol; the paper drops the μP claim on FAIL`
- `pkg-08 spec 04 §2.1: grid = widths (128, 256) × lrs (3e-4, 1e-3, 3e-3) × seeds (0, 1, 2) = 18 rows`
- `pkg-08 spec 04 §2.4: held-out loss = -EvalReport.return_per_c[0.5] (sign-flipped)`
- `pkg-08 spec 04 §3.4: MupSelftestVerdict @dataclass(frozen=True), schema_version="pkg08-spec04-v1"`
- `pkg-08 spec 04 §3.3: verdict truth table — PASS (all delta=0) / WARN (>=2 seeds delta<=1) / FAIL (otherwise)`
- `pkg-08 spec 04 §5.2: output paths runs/_mup_selftest/<config_hash>/{mup_selftest.png, .csv, .json}`
- `pkg-08 spec 04 §6: module-level constants pattern — NO new cfg fields added by this spec`
- `pkg-08 spec 04 §6.2: _WIDTHS / _LRS / _SEEDS / _TOTAL_ENV_STEPS / _EVAL_EPISODES_PER_POINT / _EVAL_CADENCE_ENV_STEPS / _LAST_N_EVAL_POINTS_FOR_LOSS / _HELD_OUT_C`
- `pkg-08 spec 04 §7: test contract — 7 named tests covering PASS / WARN / FAIL / headless plot / CSV schema / metric source`
- `pkg-08 spec 04 §10: ablation_cell="mup_verification" — registry row filter key`
