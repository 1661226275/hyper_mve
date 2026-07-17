# PR: Pkg-07 (Baselines) SDD

> **状态**：Awaiting user final ack · **类型**：SDD only（不含代码实施） · **范围**：`sdd/pkg-07-baselines/` 11 文档 + `ref_matrix.csv` + `check_ref_matrix.ps1` · **Supersedes**：pkg-06（5 internal variant 吸收 + 工厂签名重命名 + alias-with-DeprecationWarning）

---

## Summary

- 完整 SDD 11 文档（README + proposal + design + specs 01-08 + ref_matrix.csv + check_ref_matrix.ps1 + PR_DESCRIPTION）
- 工厂从 pkg-06 `create_baseline_model(cfg, variant)` rename 为 `create_baseline(cfg, variant)`；旧名保留为 alias + `DeprecationWarning(stacklevel=2)`，一个 release 过渡窗口（design §D2）
- **5 个 internal variant**（`input_wide` / `input_deep` / `ma_muzero` / `no_belief` / `rewardhead_explicit_type`）通过 `create_baseline(cfg, variant)` 统一工厂；**3 个 curriculum-override**（`hyper` / `oracle_only` / `infer_only`）直接 `HyperMuZeroModel(cfg)` + curriculum cfg override 构造，**不进工厂**（design §D3 + §3.3）
- **3 个 Tier-1 外部范式 baseline**（MAPPO / QMIX / MA-MuZero-GH）通过 `pettingzoo.ParallelEnv` 适配器 `ResourceCommonsPettingZooEnv` 消费同一 `ResourceCommonsEnv`，各自带 trainer/buffer/loss（design §D7 + spec 04 / 05 / 06）
- **1 个 Tier-2 主动 sourcing**（MAMBA，2 日预算 + `IS_SOURCED` import-time class binding toggle + 失败 fallback 转 stub；spec 06 §3 + design §D8）；**2 个永久 stub**（MARIE / GA，`__init__` 抛 `NotImplementedError`；spec 06 §5 + design §D9）
- **11-key REGISTRY**（5 internal + 3 Tier-1 + MAMBA + 2 stubs，`MappingProxyType` 只读；spec 01 §2.1 + §4.2） + **14 CLI strings**（REGISTRY ∪ {`hyper`, `oracle_only`, `infer_only`}；spec 01 §2.3 量词 canonical）
- **双层 fairness 协议**：Internal 严格 5%/10% 双阈值 + ma_muzero/explicit_type 结构性豁免（C7-INT-FAIR1/FAIR2/PERF1 继承自 pkg-06）；External 披露式 10 列 disclosure table（params / walltime_to_converge / lr_swept_best / final_return_per_seed × 5 + 5 axes；C7-EXT-FAIR1；spec 07 §4 + design §D5）
- **N-parametric PettingZoo 适配器**（agent IDs = `agent_0..agent_{N-1}` where `N = env_cfg.N ∈ {2, 4, 8}`；spec 04 §6 + design §3.5） + **两 flag 信息门控**（`oracle_mode=False` AND `eval_info_mode=False` 默认，4 leak-surface 全覆盖 + schema-marker strip；C7-EXT-ADPT1/ADPT2；spec 04 §9）
- 落地 Day 1 HARD GATE **11 项** 验收清单 + 5 个 round-1 BLOCKER + 3 个 round-2 BLOCKER 修订全部内联（README §"Day 1 HARD GATE 验收清单"）
- 8-spec **引用矩阵机器实跑 [PASS]**（Day 8，`pwsh scripts/check_ref_matrix.ps1`，delta 全空；ref_matrix.csv）
- **20 项 C7-* 硬约束**单测全部映射到 spec：13 项 C7-INT-*（FACT1/FACT2/FAIR1/FAIR2/SELF1/API1/API2/REUSE1/GRAD1/STRUCT1/STRUCT2/CFG1/PERF1，从 pkg-06 C6-* 重号继承）+ 7 项 C7-EXT-*（FACT1/ADPT1/ADPT2/API1/SMOKE1/FAIR1/STUB1，pkg-07 新增）
- **4 处下游 patches** 全部 spec 08 §7.1 显式声明（Patch 1 cfg dataclass + Patch 2 adapter new-module + Patch 3 baselines/ tree new-module + Patch 4 train_main.py CLI +6 字串）；pkg-02 obs-mask + pkg-05 mve_joint_enumerate **不在 pkg-07 territory**，由 pkg-08 spec 02 §5.1 + pkg-08 spec 06 §5.1 owns
- **5 个 `cfg.baselines.*` 新 namespace 字段穷举**（4 internal 继承 pkg-06 D10 + 1 external sweep grid `Mapping[str, tuple[float, ...]]`；design §D10 + spec 07 §2.4）；挂 V4Config 顶层与 `env/model/train/mup/eval/legacy` 平级，**消费态、不改上游**

---

## Test plan

- [ ] 用户审阅 `design.md` §4 D1-D10（10 项 Decisions）+ §5 Decision Acceptance **11 项**（含 #11 = §3.3 CLI 三列表）
- [ ] 用户审阅 `spec 01` §2.1 14-行三列映射表（11 REGISTRY + 3 curriculum-override）+ §3.2 factory 签名 + §5.2 `cli_to_factory_arg(cli) -> str` 派生函数（前缀 asymmetry 单点处理）
- [ ] 用户审阅 `spec 02` §2 `count_conditioning_params` + §5.1 `test_shared_backbone_identical_param_count` parametrized over 5 variants × 3 backbones = 15 cells（bit-exact 等参）
- [ ] 用户审阅 `spec 04` §6 N-parametric `ResourceCommonsPettingZooEnv` + §9 `test_adapter_info_gating.py` 6 个 named test functions（两 flag 正交 × 4 leak-surface × {reset, step} × {Easy N=2, Medium N=4}）
- [ ] 用户审阅 `spec 05` §4 MAPPO port from `D:\RL\lzj\MAPPO\` + §5 `_FORBIDDEN_INFO_KEYS` 运行时 assert + §5.2 CTDE 合法 `concat([obs_i])` vs privileged `resource_state`/`hotspot_centers` 边界
- [ ] 用户审阅 `spec 06` §2 QMIX (PyMARL Apache 2.0 vendoring) + §3 MA-MuZero-GH (muzero-general MIT + thin MA wrapper) + §4 MAMBA Tier-2 sourcing 协议 (2 日预算 + `IS_SOURCED` toggle + fallback) + §5 MARIE/GA stub class
- [ ] 用户审阅 `spec 07` §2 Internal 严格（4 cfg 字段 bisection + 结构性豁免表 2.6）+ §4 External 披露式 10 列 disclosure table（5 axes × 2 presets）
- [ ] 用户审阅 `spec 08` §3 `BaselineLike` Union type + `evaluate(env_fn, c_grid, episodes) -> EvalReport` 统一签名 + §7.1 四处下游 patches 表 + §9 七项 spec-08-owned 单测
- [ ] 执行 `pwsh sdd/pkg-07-baselines/scripts/check_ref_matrix.ps1` 验证 8/8 spec delta 全空 → 期望 `[PASS]`
- [ ] ack 实施期 timeline（Phase A wk1-2 internal + Phase B wk2-3 adapter+MAPPO + Phase C wk3-4 QMIX+MA-MuZero-GH + Phase C' wk4 MAMBA sourcing，≈ 250 GPU-hr 总预算）

---

## 文档清单

```
sdd/pkg-07-baselines/
├── README.md                                    # 包索引 + 10 Decisions 速查表 + 13+7 C7-* 约束 + Day 1 HARD GATE 11 项 + 下游启动条件
├── proposal.md                                  # Why（4 断言 + 外部范式必要性 + 适配器 keystone + Tier 化）/ What Changes / Capabilities / Impact / R7-1..R7-12
├── design.md                                    # 10 Decisions D1-D10 + §3.3 14-row CLI 表 + §3.4 vendoring 来源 + §3.5 N-parametric adapter + §5 Decision Acceptance 11 项
├── ref_matrix.csv                               # Day 8 机器校验产出（8/8 delta 全空）
├── PR_DESCRIPTION.md                            # 本文档
├── scripts/
│   └── check_ref_matrix.ps1                     # 引用矩阵自动校验脚本（纯 ASCII，Day 8 实跑 [PASS]）
└── specs/
    ├── 01-baseline-registry-and-cli.md          # create_baseline + 11-key REGISTRY + 14-row CLI 表 + cli_to_factory_arg + supersede pkg-06/pkg-04 双 lock
    ├── 02-shared-backbones-internal.md          # create_rep_net/create_belief_net/create_tri_context_encoder + count_conditioning_params + bit-exact 共享契约（继承 pkg-06 spec 02）
    ├── 03-internal-variants.md                  # 5 internal model 类（input_wide/deep + ma_muzero + no_belief + rewardhead_explicit_type）+ 7-API + stateful + Self-Info + grad-gating（合并 pkg-06 specs 03+04+05）
    ├── 04-pettingzoo-adapter.md                 # ResourceCommonsPettingZooEnv（N-parametric, ParallelEnv, A=6）+ 两 flag 信息门控 + CTDE 合法/特权边界 + test_adapter_info_gating.py 6 testfn
    ├── 05-external-mappo.md                     # MAPPO Tier-1 (lzj port) + CTDE 集中 critic + _FORBIDDEN_INFO_KEYS + LR sweep + smoke gate
    ├── 06-external-qmix-mamuzero-mamba.md       # QMIX (PyMARL Apache 2.0) + MA-MuZero-GH (muzero-general MIT + MA wrapper) + MAMBA Tier-2 sourcing + MARIE/GA stub
    ├── 07-fairness-protocol.md                  # Internal 严格 5%/10% 双阈值 + bisection + 结构性豁免；External 披露式 10 列 disclosure table（5 axes × 2 presets）
    └── 08-integration-contracts.md              # BaselineLike Union type + evaluate() 统一签名 + REGISTRY 只读 + 4 下游 patches + drift detector 28 anchors + 7 spec-08-owned 单测
```

---

## Verification 结果

### 引用矩阵（Day 8 实跑 [PASS]）

| spec | actual | expected | delta |
|------|--------|----------|-------|
| 01-baseline-registry-and-cli | 02,03,04,05,06,07,08 | 02,03,04,05,06,08 | ✅ |
| 02-shared-backbones-internal | 01,03,04,05,07,08 | 03,08 | ✅ |
| 03-internal-variants | 01,02,04,05,06,07,08 | 02,07,08 | ✅ |
| 04-pettingzoo-adapter | 01,05,06,08 | 05,06,08 | ✅ |
| 05-external-mappo | 01,04,07,08 | 04,07,08 | ✅ |
| 06-external-qmix-mamuzero-mamba | 01,04,05,07,08 | 04,07,08 | ✅ |
| 07-fairness-protocol | 01,02,03,05,06,08 | 03,05,06,08 | ✅ |
| 08-integration-contracts | 01,02,03,04,05,06,07 | 01,02,03,04,05,06,07 | ✅ |

**8/8 spec delta 全空** ✅（脚本判据：expected ⊆ actual；额外的实际引用反映了 spec 之间的丰富交叉锚定，符合 design §8 预期表）

### C7-INT-* + C7-EXT-* 硬约束单测命名对账

#### Internal（13 项，从 pkg-06 C6-* 重号继承）

| # | 约束 | 单测 | spec | 状态 |
|---|------|------|------|------|
| C7-INT-FACT1 | 5 internal variant 均可经工厂实例化 | `test_factory_dispatches_11_keys` | spec 01 §7.1 | ✅ |
| C7-INT-FACT2 | 工厂收 `"hyper"` 抛 `ValueError` | `test_factory_rejects_hyper_with_value_error` | spec 01 §7.2 | ✅ |
| C7-INT-FAIR1 | 条件化消费子系统 ≤5% warn / ≤10% fail（3 strict + 2 exempt-annotated） | `test_internal_param_fairness_5pct_warn_10pct_fail` | spec 07 §2 | ✅ |
| C7-INT-FAIR2 | RepNet/BeliefNet/TriCtx 跨 5 internal variant 参数量逐位相同（5×3=15 cells） | `test_shared_backbone_param_count_bitexact` | spec 02 §5.1 | ✅ |
| C7-INT-SELF1 | 不泄漏 oracle types（每 variant；`cap_i.shape[-1]==4` 严格） | `test_internal_self_info` | spec 03 §9.3 | ✅ |
| C7-INT-API1 | 5 internal variant 实现完整 7-API（5×7=35 cells） | `test_7api_signatures` | spec 03 §9.1 | ✅ |
| C7-INT-API2 | stateful 用最后一次 subjective agent_id | `test_stateful_uses_last_subjective_agent_id` | spec 03 §9.2 | ✅ |
| C7-INT-REUSE1 | 同一 MuZeroTrainer/Worker/Buffer/compose_total_loss | `test_baseline_reuses_same_trainer_class` | spec 08 §9 + pkg-06 inheritance | ✅ |
| C7-INT-GRAD1 | pre-5K BeliefNet 梯度=0（每 variant，共用单一 `BeliefGradGating`） | `test_baseline_update_step_gates_belief_grad` | spec 03 §9.4 | ✅ |
| C7-INT-STRUCT1 | input-conditioned 不含 `DualHyperNetwork` | `test_input_baseline_no_hypernet` | spec 03 §9.5 | ✅ |
| C7-INT-STRUCT2 | ma_muzero 共享单一 `RewardHead` | `test_ma_muzero_shared_rewardhead` | spec 03 §9.5 | ✅ |
| C7-INT-CFG1 | 工厂消费 4 个 `cfg.baselines.internal_*` 字段无 hardcoded fallback | `test_factory_consumes_cfg_baselines_fields_no_hardcoded_fallback` | spec 01 §7.4 + spec 07 §2.4 | ✅ |
| C7-INT-PERF1 | input baseline 单步 forward ≤ 2.0× hyper（CPU, Easy N=2, 100×3 medians） | `test_internal_input_variant_forward_under_2x_hyper` | spec 07 §3.2 | ✅ |

#### External（7 项，pkg-07 新增）

| # | 约束 | 单测 | spec | 状态 |
|---|------|------|------|------|
| C7-EXT-FACT1 | 3 Tier-1 external variant 均可经工厂实例化 | `test_factory_dispatches_external_mappo` (+QMIX/MA-MuZero-GH 同模) | spec 05 §12.1 + spec 06 §12 | ✅ |
| C7-EXT-ADPT1 | 适配器默认 `oracle_mode=False` AND `eval_info_mode=False`，4 leak-surface (`c_true`/`types`/`hotspot_centers`/`resource_state`) 全 strip + schema-marker strip + 两 flag 正交 | `test_defaults_strip_all_leak_surface_and_markers` + `test_oracle_mode_exposes_only_oracle_fields` + `test_eval_info_mode_exposes_only_eval_fields` | spec 04 §9 (testfn 1+2+3) | ✅ |
| C7-EXT-ADPT2 | 适配器实现 `pettingzoo.ParallelEnv`，N-parametric agents `agent_0..agent_{N-1}`，离散 A=6 actions；smoke 跑 Easy (N=2) 与 Medium (N=4) | `test_agent_ids_are_N_parametric` + `test_action_space_is_discrete_6_per_agent` + `test_action_dict_missing_key_raises` | spec 04 §9 (testfn 4+5+6) | ✅ |
| C7-EXT-API1 | 3 Tier-1 external runner 实现统一 `.evaluate(env_fn, c_grid, episodes) -> EvalReport`（32-field schema） | `test_mappo_evaluate_returns_evalreport_with_correct_schema` + `test_evaluate_signature_uniform_across_baseline_types` | spec 05 §12.3 + spec 08 §9.4 | ✅ |
| C7-EXT-SMOKE1 | 3 Tier-1 external runner 在 Easy preset 上 return > random within 20K env steps（3 seeds, p<0.05） | `test_mappo_easy_smoke_return_beats_random_at_20k` (+QMIX/MA-MuZero-GH 同模) | spec 05 §10 + spec 06 §9 | ✅ |
| C7-EXT-FAIR1 | External LR sweep ≥3 LR × ≥3 seeds，10 列 disclosure table 披露 | `test_mappo_lr_sweep_grid_is_at_least_3_lrs` + `test_mappo_train_consumes_lr_kwarg` + `test_mappo_train_lr_changes_persisted_lr` | spec 05 §12.5 + spec 07 §4 | ✅ |
| C7-EXT-STUB1 | MARIE/GA/MAMBA-未 sourced 走 `NotImplementedError` | `test_stub_external_baselines` | spec 06 §5 + spec 01 §7.6 | ✅ |

**20/20 单测命名映射 PASS** ✅（13 INT + 7 EXT）

### Spec-08-owned 集成单测（7 项；spec 08 §9 + 4 处下游 patches 测试 gate）

- `test_factory_dispatches_11_keys`（spec 08 §9.1，mirror spec 01 §7.1）
- `test_factory_rejects_hyper_with_value_error`（spec 08 §9.2，mirror spec 01 §7.2）
- `test_create_baseline_model_alias_warns_deprecation`（spec 08 §9.3，mirror spec 01 §7.6）
- `test_evaluate_signature_uniform_across_baseline_types`（spec 08 §9.4，**新；** parametrized over 8 instantiable variants = 5 internal + 3 Tier-1）
- `test_baselines_config_has_5_fields_exact` + `test_v4config_has_baselines_subconfig`（spec 08 §9.5，**新；** 锁 `BaselinesConfig` 5 字段穷举 + V4Config 顶层挂载）
- `test_drift_detector_finds_all_28_anchors`（spec 08 §9.6，**新；** 验证 §6.3 grep 表 28 个跨 spec drift 锚点）
- `test_train_main_variant_cli_round_trip`（spec 08 §9.7，**新；** Patch 4 train_main.py +6 CLI 字串与 `REGISTRY` 派生一致）

---

## Downstream patches（4 处，spec 08 §7.1 显式声明）

| Patch # | 文件 | 估计 size | Tag | 声明在 spec § | 测试 gate |
|---------|------|-----------|-----|---------------|-----------|
| **1** | `hyper_mve/configs/v4_config.py` | +1 field on `V4Config` (`baselines: BaselinesConfig = field(default_factory=BaselinesConfig)`) + `BaselinesConfig` dataclass body (~25 lines) | `[config-additions]` | spec 01 §6 + spec 08 §5 | `test_cfg_baselines_5_fields_consumed_no_defaults` (spec 01 §7.4) + `test_v4config_has_baselines_subconfig` (spec 08 §9.5) |
| **2** | `hyper_mve/envs/adapters/__init__.py` + `hyper_mve/envs/adapters/pettingzoo_wrapper.py` | NEW directory（1 line `__init__.py`）+ `pettingzoo_wrapper.py` ~200 lines（class body + `_filter_info` + `reset` / `step` + observation/action space） | `[new-module]` | spec 04 §2-§8 | `test_adapter_info_gating.py`（spec 04 §9 — 6 testfn 覆盖 C7-EXT-ADPT1 + C7-EXT-ADPT2） |
| **3** | `hyper_mve/baselines/` directory tree | NEW：`__init__.py` (factory + REGISTRY ~30L) + `_runner_protocol.py` (~50L) + `shared_backbones.py` (~150L) + `internal/` (5 files × ~150L = ~750L) + `external/` (1 `__init__.py` + 3 Tier-1 × ~200L + 2 stubs × ~10L + MAMBA ~10–200L) — 总 ~2000 lines | `[new-module]` | spec 01 §3（factory + REGISTRY）+ spec 02 §2 + spec 03 §2 + spec 05 §4 + spec 06 §2/§3/§4/§5 | `test_factory_dispatches_11_keys` (spec 01 §7.1) + 10 sibling-spec tests |
| **4** | `hyper_mve/scripts/train_main.py` | +6 个 `--variant` CLI 字串（`external_mappo` / `external_qmix` / `external_ma_muzero_gh` / `external_mamba` / `external_marie` / `external_ga`）+ `_build_cli_variant_choices()` 派生函数 + `REGISTRY` import；diff ≈ +20 lines | `[downstream-cli]` | spec 01 §3.3 + §5.1 | `test_train_main_variant_cli_round_trip` (spec 08 §9.7) + `test_registry_keys_equal_cli_choices_and_factory_args` (spec 01 §7.5) |

> 实施总估计：~2000（Patch 3）+ ~200（Patch 2）+ ~25（Patch 1）+ ~20（Patch 4）≈ **2245 lines** 新代码 + ~1500-2000 lines tests under `tests/baselines/`（per README §"输出清单" tests subtree）。

### 明确不在 pkg-07 territory 的下游 patches（spec 08 §7.2）

两处潜在 patches 由 **pkg-08** 拥有，**不**进 pkg-07 4-patch 清单（避免 SDD 漂移）：

- **Excluded patch A — Pkg-02 obs-mask**：`hyper_mve/envs/resource_commons/observations.py` +3 lines `if not cfg.env.c_visible: obs[:, c_channel] = 0` — **pkg-08 spec 02 §5.1** owns（zero-shot c_hidden 评估通路） + **pkg-08 spec 08 §5** aggregates。
- **Excluded patch B — Pkg-05 mve_joint_enumerate**：`hyper_mve/planning/mve_planner.py` +1 line `if cfg.train.mve_joint_enumerate: candidates = list(itertools.product(...))` — **pkg-08 spec 06 §5.1** owns（Ablation 4 Joint cell 训练时 toggle） + **pkg-08 spec 08 §5** aggregates。

切分理由（spec 08 §7 锁定）：pkg-07 = "baselines + adapter（算法构造面）"，pkg-08 = "evaluation + c_hidden + ablation（实验 harness 面）"。Patch 归 pkg-07 iff 其目标文件在算法构造/工厂 dispatch 轴上（cfg dataclass / adapter / model class / CLI variant 注册）；归 pkg-08 iff 其目标文件调节评估时/消融时行为。

---

## 5 个 `cfg.baselines.*` 字段穷举（design §D10 dataclass）

| 字段 | 类型 | 默认 | 消费者 spec |
|------|------|------|-------------|
| `internal_wide_hidden_dim` | `int` | `512` | spec 03 §7.1 `InputWideBaselineModel.__init__` / spec 07 §2.4 bisection target 1 |
| `internal_deep_layers` | `int` | `8` | spec 03 §7.2 `InputDeepBaselineModel.__init__` / spec 07 §2.4 bisection target 2（层数离散 → 可能 [+5%, +10%] WARN） |
| `internal_ma_muzero_share_pred_head` | `bool` | `True` | spec 03 §7.3 `MAMuZeroBaselineModel` (default=`True` → 结构性豁免;`False` → 等参 attempt) |
| `internal_explicit_type_branches` | `int` | `2` (α / β) | spec 03 §7.5 `ExplicitTypeRewardBaselineModel`（匹配 env α/β capability split） |
| `external_lr_sweep_grid` | `Mapping[str, tuple[float, ...]]` (`MappingProxyType`) | `{"external_mappo": (1e-4, 3e-4, 1e-3), "external_qmix": (1e-4, 3e-4, 1e-3), "external_ma_muzero_gh": (1e-4, 3e-4, 1e-3)}` | spec 05 §12.5 MAPPO LR sweep + spec 06 §8 QMIX/MA-MuZero-GH LR sweep + spec 07 §4 披露式 fairness `lr_swept_best` 轴 + pkg-08 spec 05 sweep harness 枚举 |

> 4 internal 字段继承自 pkg-06 D10；1 external 字段（`external_lr_sweep_grid`）为 pkg-07 新增。**Per-impl tuning constants**（如 `external_mappo_share_policy` / `external_qmix_mixer_hidden_dim` / `external_ma_muzero_gh_simulations` / `external_smoke_max_env_steps`）**不进** `cfg.baselines`，留在 spec 05/06 内部 defaults（design §D10 末段：cfg.baselines 是「跨 spec 共享契约」而非「所有可调参数的字典」）。
>
> **挂载点**：V4Config 顶层新增 sub-config，与 `cfg.env/model/train/mup/eval/legacy` 平级；V4Config 无 `.v4` 中间层（verified `configs/v4_config.py:23-37`）。**纯消费态**：本包仅声明字段，实施期由 Pkg-01 spec05 同步消费（不改 Pkg-01..05 SDD）。

---

## Implementation timeline（从 Plan File）

实施期总预算 **4–5 周**，GPU 算力 **≈ 250 GPU-hr**（含 Tier-1 LR sweep + smoke 收敛性检查；不含 Pkg-08 全量主表）。

| Phase | 周次 | 任务 | 关键产出 |
|-------|------|------|----------|
| **Phase A** | wk 1–2 | 5 internal model classes + REGISTRY 工厂 + 11 internal tests | `baselines/__init__.py` + `baselines/internal/*.py` × 5 + `tests/baselines/internal/*.py` 全过；Patch 1 (cfg dataclass) Day 1 落地 |
| **Phase B** | wk 2–3 | PettingZoo adapter (Patch 2) + MAPPO Tier-1 (`baselines/external/mappo.py`) + smoke gate | `test_adapter_info_gating.py` (6 testfn) 全过 + `test_mappo_easy_smoke_return_beats_random_at_20k` 通过 + Patch 4 train_main.py CLI 扩 6 |
| **Phase C** | wk 3–4 | QMIX (PyMARL vendor Apache 2.0) + MA-MuZero-GH (muzero-general MIT + MA wrapper) | `_third_party_licenses/pymarl_LICENSE.txt` + `muzero_general_LICENSE.txt` 携带 + 两个 smoke gate 全过 + 9 列 Methods 主表数据齐全 |
| **Phase C'** | wk 4（并行） | MAMBA Tier-2 sourcing（2 日预算 + `IS_SOURCED` toggle 决议） | `external/mamba.py` 模块级 `IS_SOURCED: Final[bool]` + class binding；成功 → 主表 10 列；失败 → fallback stub + 搜索日志写入 spec 06 §3 |
| **Phase D-F** | 后续 | pkg-08 下游消费（unified evaluator + ablation harness + Methods 主表） | pkg-08 spec 01 `_evaluate_external` + pkg-08 spec 05 sweep harness over 11-key REGISTRY |

**响应式 SLA（SDD 期）**：Day 1 HARD GATE 不过 → 全流程顺延 1 天；每 spec 当天末交审，审阅往返不计入"天"。

---

## Risks（R7-1..R7-12，from proposal §5）

| # | 风险 | 缓解 |
|---|------|------|
| **R7-1** | `input_wide`/`deep` 与 hyper 参数对不齐 → 断言 B′ 失效 | 双阈值 warn ≤5% / fail ≤10% + bisection 调 `cfg.baselines.internal_wide_hidden_dim` / `internal_deep_layers`（spec 07 §2.4） |
| **R7-2** | shared backbone 跨 variant 参数量不一致 | spec 02 §2 强制同 `(env_cfg, model_cfg)` 同类构造 + `test_shared_backbone_param_count_bitexact` 15-cell parametrize |
| **R7-3** | External baseline 不收敛 / 在 ResourceCommons 上"平地起飞" | Easy preset smoke 收敛性检查（C7-EXT-SMOKE1, p<0.05 over 3 seeds）；失败降级 "appendix best-effort"（spec 05 §10） |
| **R7-4** | **PettingZoo 适配器泄漏 oracle / eval-only 信息** → external 偷看 | **两 flag `oracle_mode=False` AND `eval_info_mode=False` 默认**；适配器读 env schema-marker tuples（DRY 抗 env 演化）+ schema-marker 自身 strip + 强制单测 6 testfn × {Easy, Medium}（spec 04 §9） |
| **R7-5** | External runner 自带 trainer 引入新依赖（如 RLlib） | spec 05/06 头部 vendoring 政策：minimal port + 纯 PyTorch，不引 RLlib/SB3；vendoring 来源带 commit hash + LICENSE 文件 |
| **R7-6** | **MA-MuZero-GH vendoring 失败**（fork 不可移植 / 单 agent 仅） | spec 06 §3 三日检查点 + 失败 fallback "每 agent vanilla MuZero + 平均 reward"（弱版本） |
| **R7-7** | **MAMBA 找不到可用 source** | spec 06 §4 sourcing 协议 2 日 budget + 失败转 stub 保留搜索日志（不阻塞 SDD finalize；`IS_SOURCED=False` + class binding 在 import-time 切到 stub） |
| **R7-8** | External LR sweep 超 GPU 预算 | spec 07 §4 限制 ≥3 LR × ≥3 seeds；Easy 先 sweep 缩到 1 LR 再 Medium 跑 5 seed |
| **R7-9** | `randomize_order` / `use_coord_desc` 重命名碰撞 | alias-with-deprecation + 实施前 grep；pkg-08 spec 06 实施期检查 |
| **R7-10** | pkg-06 supersede 后旧 import 仍被消费 → 静默走旧契约 | spec 01 alias `create_baseline_model` + `DeprecationWarning(stacklevel=2)`；Day 6 grep 校验全仓单一 `create_baseline` 签名 |
| **R7-11** | External runner 公平地不公平（隐式假设 share-policy） | spec 05/06 显式声明 `share_policy=True/False`；adapter 暴露 per-agent obs dict 以允许 separate-policy |
| **R7-12** | Tier-1 任一 baseline smoke 失败 → 论文主表少一列 | smoke 早跑（Phase B/C 第 1 周）；失败立即换 alternative impl 或弱化为 "best-effort" 披露（spec 05 §3.10） |

---

## Review Checklist（用户最终勾选）

- [ ] `proposal.md` 读完，**Why** 充分（继承 pkg-06 4 条断言论证 + 新增"外部 baseline 必要性：论文需横向坐标对比 MARL 范式" + "PettingZoo 适配器单点封装防 oracle 泄漏" + "Tier 化以管理 GPU 与日历预算"）
- [ ] `design.md` 读完，**Day 1 HARD GATE 11 项验收清单**全过（10 Decision-anchored + #11 §3.3 CLI 三列表 14 行）
- [ ] §3.2 11-key REGISTRY 分层区别清晰（5 internal / 3 Tier-1 / 1 Tier-2 MAMBA / 2 永久 stubs；hyper/oracle_only/infer_only 不进工厂）
- [ ] §3.3 CLI ↔ factory-arg ↔ model-class **三列映射表 14 行** + 前缀约定脚注（`baseline_` strip / bare pass-through / `external_` preserve）可对账 `train_main.py:45-52 _DEFERRED_VARIANTS`
- [ ] §3.4 External vendoring 来源选定（MAPPO lzj-port + QMIX PyMARL Apache 2.0 + MA-MuZero-GH muzero-general MIT）+ 选源记录可追溯
- [ ] §3.5 **两 flag 信息门控**架构（`oracle_mode=False` AND `eval_info_mode=False` 默认，4 leak-surface (`c_true`/`types`/`hotspot_centers`/`resource_state`) 全覆盖，schema-marker strip，CTDE 合法 (`concat([obs_i])` ± action ± `caps`) / 特权 (`resource_state`/`hotspot_centers`) 边界，N-parametric agents `agent_0..agent_{N-1}`）认可
- [ ] §D5 **双层 fairness 协议**口径认可（Internal 严格 5%/10% 双阈值 + 结构性豁免 ma_muzero/explicit_type；External 披露式 10 列 disclosure table）
- [ ] §D7 mermaid 流程图（§7）正确反映 Internal vs External 训练流程区别（含 stubs `NotImplementedError`）+ 统一 Eval (`evaluate(env_fn, c_grid, episodes) -> EvalReport`)
- [ ] §D8 MAMBA sourcing 协议有失败 fallback（`IS_SOURCED` import-time class binding；不阻塞 SDD finalize）
- [ ] §D9 MARIE/GA stub 策略合理（`__init__` 抛 `NotImplementedError`；不进 main table；CLI 名保留供未来 pkg-07.5 promote）
- [ ] §D10 `cfg.baselines.*` **5 字段穷举**（4 internal 继承 pkg-06 D10 + 1 external `Mapping[str, tuple[float, ...]]`；挂 V4Config 顶层无 `.v4` 层；纯消费态）合理
- [ ] `ref_matrix.csv` 8/8 spec delta 全空 ✅；`scripts/check_ref_matrix.ps1` 实跑 `[PASS]`
- [ ] **20 项 C7-* 单测命名映射全部对账**（13 INT + 7 EXT，无悬空）
- [ ] **4 处下游 patches** 清单完整（Patch 1 cfg + Patch 2 adapter + Patch 3 baselines/ + Patch 4 CLI），且 pkg-02 obs-mask + pkg-05 mve_joint_enumerate **明确不在 pkg-07 territory**（由 pkg-08 spec 02/06 owns）
- [ ] 10 天 SDD 顺序合理（Day 1 HARD GATE 未过顺延 1 天）+ 实施期 4–5 周（≈ 250 GPU-hr）预算可接受
- [ ] 出包后可启动 Pkg-08（Eval + Ablation）—— `REGISTRY` 11 keys + 适配器 + `evaluate() -> EvalReport` + `BaselineLike Union type` + 量词 canonical 五件套已稳定

---

🤖 Generated as part of Hyper-MuZero v4 SDD writing workflow (pkg-07 Day 9)
