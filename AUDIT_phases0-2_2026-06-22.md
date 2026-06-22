# Spec-Conformance Audit — Phases 0–2 (pkg-07 baselines + pkg-08 eval)

**Date:** 2026-06-22 · **Status:** COMPLETE (all 22 findings adversarially verified)

## Provenance
- **Run 1** `wf_1c686705-aa5` (1.67M tok): 7 review surfaces + verify. **Hit session limit mid-Verify** — only 9 of 22 findings verified before the quota crash (reset 8:10pm Asia/Shanghai). `status:completed` was misleading.
- **Run 2** `wf_9848145c-842` (864k tok, clean): adversarially re-verified the 13 unverified findings + the 2 external "PASS" claims (14 total). Refute-by-default; claimed-high/blocker ran at `xhigh` with a forced code+spec+test triple-check.
- **Main agent**: independently re-checked factory-cli F1, CFG-01, and the mve_planner byproduct against the code.

**Authority caveat:** the audit treats `sdd/` specs as the locked authority (spec wins over code). For the CFG-01/02/03 cluster this matters — see "Judgment call" below. You own both code and spec; some "high" findings could equally be resolved by amending the spec.

## Surface verdicts
| Surface | Verdict | Real defects |
|---|---|---|
| registry byte-identity | ✅ clean | 0 |
| PettingZoo adapter | ✅ clean | 0 |
| EvalReport byte-identity | ✅ clean | 0 (1 cosmetic nit) |
| factory-cli | ⚠️ mixed | F1 (high), F3 (low) |
| configs | ⚠️ mixed | CFG-01/02/03 (high) |
| external-scaffold | ⚠️ mixed | F5, F6 (low); F1 nit (positive) |
| internal-variants | ❌ drifts | F1 (blocker), F2, F3 (high), F4 (nit) |

## CONFIRMED defects (real, verified by ≥2 independent passes)

### 🔴 Blocker (1)
- **internal-variants F1** — `baselines/internal/explicit_type_reward.py:54-57`. `explicit_type` variant builds plain `trans_net`/`reward_head`/`pred_net` MLPs; no `hyper_trans`/`hyper_pred` modules exist (grep confirms). Spec 03 §7.5 mandates keeping `hyper_trans`+`hyper_pred` (drop only `hyper_rew`). **`test_explicit_type_no_hyper_rew` fails** its two positive `any('hyper_trans'…)`/`any('hyper_pred'…)` asserts. Docstring openly admits "approximate via plain MLPs here."

### 🟠 High (6)
- **factory-cli F1** — `baselines/__init__.py:118-120`. `cli_to_factory_arg` only strips `baseline_`; never raises for curriculum-override/unknown CLI. Fails spec 01 §7.5 round-trip lock test. *(main-agent confirmed)*
- **CFG-01** — `configs/train_config.py:94`. `use_coord_desc` is a plain field, not the spec-locked `@property` shim; no DeprecationWarning on READ. Fails `test_use_coord_desc_alias_warns_read`. *(main-agent confirmed)*
- **CFG-02** — `train_config.py:143-149`. WRITE-path warning says "use_coord_desc is deprecated" but spec 06 §4.2 mandates "use_coord_desc **kwarg** is deprecated". `test_use_coord_desc_alias_warns_write` regex `match="…kwarg is deprecated"` fails.
- **CFG-03** — `train_config.py:94-156`. Missing the spec-locked synthetic `_use_coord_desc_compat: bool = True` sentinel (grep: zero matches in code). Code uses a value-mirror reconciliation instead of the spec's `@property`+sentinel mechanism.
- **internal-variants F2** — `baselines/internal/no_belief.py:37-48`. `no_belief` built as input-conditioned MLPs, not the mandated hypernet skeleton (spec 03 §7.4: "保留 hypernet 骨架 … 单点改动 = belief 路置零"). Breaks the §10.4 "isomorphic to hyper" equal-param premise behind Assertion C.
- **internal-variants F3** — `models/hyper_muzero_model.py`. `HyperMuZeroModel` never defines `SHARED_BACKBONE_PREFIXES` (grep: zero hits). Spec 02 §3.3 requires it on all 5 internal classes **+ HyperMuZeroModel**; `count_conditioning_params(hyper)` raises `AttributeError` and `test_shared_backbone_prefixes_defined` fails. *(Pkg-04 file, outside the assigned surface, but the spec-02 contract depends on it.)*

### 🟡 Low (4)
- **factory-cli F3** — unknown-CLI error message wording differs from spec §5.2; reconcile when F1 is fixed.
- **external F5** — `ExternalBaselineRunner` is an `abc.ABC` in `external/base.py` with 2 methods; every spec cites a 4-method Protocol in `_runner_protocol.py` (which doesn't exist). Location+shape drift; *downgraded medium→low* (scaffold-phase, internally consistent).
- **external F6** — `_FORBIDDEN_INFO_KEYS` not hoisted to `external/__init__.py` (absent repo-wide). Spec 06 §6.1 single-source-of-truth unmet; `test_runner_forbidden_info_keys_assertion_present` would fail. *Downgraded medium→low* (train bodies are Phase-4 stubs — no live CTDE leak yet).
- **external F3** — claimed-high zero-report violation **refuted as high, residual low**: the verifier found the spec text more permissive than the reviewer read; reconcile variant-naming/empty-MappingProxyType when the runners are implemented.

### ⚪ Nit / non-defects (3)
- **EVR-NIT-1** — `eval/eval_report.py` omits an unused `from types import MappingProxyType` import. No test impact.
- **external F1** — *positive observation*, not a defect: Tier-1 runners correctly construct and only `.train()` raises (acceptance MET).
- **internal-variants F4** — doc-only: spec 03 §6.1's illustrative `belief_grad_gating` import path is wrong; code correctly follows spec 02's package re-export.

### ➕ Byproduct (main-agent confirmed)
- **`mve_planner.py:84,160,161`** is an *unlisted* `use_coord_desc` reader. Spec 06 §8.2's planner rename was declared but never applied; `test_use_coord_desc_grep_codebase_clean` fails with `extra={mve_planner.py}`. (This is the true grep-clean violation that the refuted CFG-04 misattributed to `train_config.py`.)

## REFUTED — invalid (reviewer misread; do NOT act)
factory-cli **F2** (`_build_cli_variant_choices` "missing"), factory-cli **F4** (ValueError wording), **CFG-04** (grep-clean misattribution), **CFG-05** (`use_oracle_types` "missing"), external **F2** (positive — signature OK), external **F4** (stub message wording — corrected nit), external **F7** (dataclass-default fragility — invalid), internal **F5** (`apply_raw` is an exact alias of `apply`).

## Judgment call before fixing the configs cluster
CFG-01/02/03 + the mve_planner byproduct **all** stem from one decision: the spec locks a `@property`-shim + `_use_coord_desc_compat`-sentinel deprecation mechanism; the code implemented a simpler plain-field + value-mirror reconciliation that **preserves runtime behavior** but fails the locked lock-tests on mechanism/wording. Two valid resolutions:
1. **Conform code → spec** (4 small edits: `@property` shim, sentinel field, "kwarg" wording, planner rename). Restores the locked tests.
2. **Amend spec → code** if the plain-field design is preferred — then update specs 06/08 §4–7 + the named tests.

This is yours to decide; the audit can't, because it assumes the spec is authoritative by construction.

## Recommended action order
1. **internal-variants F1 (blocker)** — `explicit_type` must carry `hyper_trans`+`hyper_pred`. Biggest correctness gap; blocks the explicit-type comparison.
2. **internal-variants F2, F3 (high)** — restore `no_belief` hypernet skeleton; add `SHARED_BACKBONE_PREFIXES` to `HyperMuZeroModel`. These underpin the equal-param fairness story (Assertion C).
3. **factory-cli F1 (+F3)** — add the raise branches per spec 01 §5.2.
4. **configs cluster** — make the §"Judgment call" decision, then apply the chosen direction.
5. **external F5/F6/F3** — defer with the rest of the Phase-4 external-runner work; track so they aren't lost.
