# Hyper-MuZero v4 — Pkg-01 / Pkg-02 / Pkg-03 测试清单

> 版本：2026-06-02 · 适用范围：`hyper_mve/` 已实现的 schemas + configs + ResourceCommons env + TriContextEncoder/BeliefNet
>
> 工作目录：**`D:\RL\hyper_mve\`**（所有 pytest / python 命令都在此目录下运行）。
>
> 本清单逐 spec 列出可执行的测试函数、硬约束（数值闸口）与必跑脚本。请按 §0 → §1 → §2 → §3 → §4 顺序勾选。

---

## §0 · 环境准备 & 一键自检

### 0.1 venv & 依赖

```powershell
# 在 D:\RL 下：
.\.venv\Scripts\Activate.ps1
cd D:\RL\hyper_mve
python -c "import torch, numpy, gym; print(torch.__version__, numpy.__version__, gym.__version__)"
```

- [ ] `import torch` 成功，且 `torch.cuda.is_available()` 与你期望一致（CPU 也可跑全部单测；BeliefNet 合成数据脚本建议 GPU）。
- [ ] `from hyper_mve.schemas import AgentType, CapabilityVector, ObservationLayout, TimeStepRecord, ContextSchema` 顶层 import 通过。
- [ ] `from hyper_mve.envs.resource_commons import ResourceCommonsEnv` 顶层 import 通过。
- [ ] `from hyper_mve.models import TriContextEncoder, BeliefNet, belief_loss` 顶层 import 通过（具体名以 `models/__init__.py` 暴露为准）。

### 0.2 现有测试目录速览（已存在文件）

| 目录 | 现有测试文件 |
|------|------|
| `tests/schemas/` | `test_agent_type.py`, `test_capability_vector.py`, `test_observation_layout.py`, `test_buffer_record.py`, `test_context_schema.py` |
| `tests/configs/` | `test_env_config.py`, `test_v4_config.py`, `test_presets.py`, `test_legacy_config.py` |
| `tests/envs/` | `test_state.py`, `test_dynamics.py`, `test_rewards.py`, `test_observations.py`, `test_context_evolution.py`, `test_spawn.py`, `test_presets_env.py`, `test_env_info_oracle.py`, `test_env_integration.py` |
| `tests/models/` | `test_tri_context.py`, `test_c_encoder.py`, `test_role_encoder.py`, `test_belief_net.py`, `test_belief_losses.py`, `test_permutation_pool.py` |
| `tests/` 根 | `test_legacy_import_warning.py` |

### 0.3 一次性全量自检（≈30 秒，CPU）

```powershell
pytest tests/ -q
```

- [ ] 全部测试通过（绿）。若个别 fail，到对应 §1/§2/§3 章节逐项调试。

---

## §1 · Pkg-01 Foundation Schema

> 目标：5 个 schema + 5 层 V4Config + 3 个 preset 字段 100% 与 Ch3.9 表对齐。

### 1.1 Spec 01 — AgentType（`tests/schemas/test_agent_type.py`）

```powershell
pytest tests/schemas/test_agent_type.py -v
```

- [ ] `test_enum_value_stable` — `ALPHA.value == 0`, `BETA.value == 1`
- [ ] `test_one_hot_shape_and_dtype` — `(2,)` float32, `[1,0]`
- [ ] `test_one_hot_beta` — `[0,1]`
- [ ] `test_from_index_valid` / `test_from_index_invalid` — `from_index(2)` 抛 `ValueError`
- [ ] `test_from_str_variants` — `'alpha'/'ALPHA'/'α'/'a'` ↔ ALPHA；`'gamma'` 抛错
- [ ] `test_count_in_assignment` — `[α,α,β,β]` 计数正确
- [ ] `test_to_long_tensor` — dtype int64
- [ ] `test_pickle_roundtrip` — IntEnum pickle 还原
- [ ] `test_intenum_arithmetic` — `int(ALPHA)+0 == 0`, `BETA == 1`

### 1.2 Spec 02 — CapabilityVector（`tests/schemas/test_capability_vector.py`）

```powershell
pytest tests/schemas/test_capability_vector.py -v
```

- [ ] `test_construct_valid` / `test_boundary_lower` / `test_boundary_upper`
- [ ] **范围拒绝**：`test_eta_out_of_range` / `test_phi_fov_out_of_range` / `test_nu_out_of_range` / `test_zeta_out_of_range`
- [ ] `test_nan_rejected` — η=NaN → ValueError
- [ ] `test_frozen` — 修改字段 → FrozenInstanceError
- [ ] `test_fov_int_round` — banker's rounding
- [ ] `test_to_array_dtype` / `test_from_array_roundtrip`
- [ ] `test_sample_default_in_range` — 1000 次采样全部范围内
- [ ] `test_sample_n_reproducible` — 同 seed 等价
- [ ] `test_to_batch_tensor_shape` — `(N, 4) float32`
- [ ] `test_pickle_roundtrip`
- [ ] **v4 normalize() 子套件**（关键）：
  - [ ] `test_capability_normalize_shape_and_dtype` — `(4,) float32`
  - [ ] `test_capability_normalize_lower_boundary` — 全 0
  - [ ] `test_capability_normalize_upper_boundary` — 全 1
  - [ ] `test_capability_normalize_midpoint` — 全 0.5
  - [ ] `test_capability_normalize_eta_only` — η=0.8 → 0.3
  - [ ] `test_capability_normalize_zeta_scale_handling` — ζ 量级解决
  - [ ] `test_capability_normalize_1000_samples_in_unit_range` — 均值≈0.5
  - [ ] `test_capability_normalize_does_not_modify_raw_fields`
  - [ ] `test_cap_norm_constants_match_ch36` — `CAP_NORM_LO=(0.5,2.0,0.8,10.0)`, `HI=(1.5,4.0,1.0,30.0)`
  - [ ] `test_capability_normalize_custom_dtype`

### 1.3 Spec 03 — ObservationLayout（`tests/schemas/test_observation_layout.py`）

```powershell
pytest tests/schemas/test_observation_layout.py -v
```

- [ ] **维度硬约束**：
  - [ ] `test_total_dim_easy` — N=2,K=8 → 45
  - [ ] `test_total_dim_medium` — N=4,K=20 → **99**
  - [ ] `test_total_dim_hard` — N=8,K=40 → 195
- [ ] `test_block_dim_each` — self=4, resource=60, neighbor=27, global=2, capability=4, type=2
- [ ] `test_block_dim_invalid` — 未知块 → ValueError
- [ ] `test_block_offset_type_is_last_two` — type 块在末尾
- [ ] `test_block_offset_self_is_first` — self 在开头
- [ ] `test_block_offsets_sum_to_total` — 6 块累加 == total_dim
- [ ] `test_pad_resource_block_empty` / `_partial` / `_overflow`
- [ ] `test_pad_neighbor_block_with_presence` — presence_flag=1 vs 0
- [ ] `test_slice_block_type` — 切 99 维 obs 末两维
- [ ] **`test_self_info_principle`** — `TYPE_DIM == 2`，N=8 时 type 块仍为 2 维（不是 16）

### 1.4 Spec 04 — TimeStepRecord（`tests/schemas/test_buffer_record.py`）

```powershell
pytest tests/schemas/test_buffer_record.py -v
```

- [ ] `test_construct_valid`
- [ ] `test_n_dimension_mismatch` — N 维不匹配 → ValueError
- [ ] `test_z_hat_dim_mismatch` — z_hat dim1 != N-1 → ValueError
- [ ] **`test_c_hat_is_scalar_per_agent`** — v4 关键：`c_hat.shape == (N,)`，**不是** `(N, d_c=16)`
- [ ] **`test_z_hat_order_convention`** — agent i 的 `z_hat[i, k]` 对应 `agent_id = (k if k<i else k+1)`（agent_id 升序跳过 self）
- [ ] `test_to_from_arrays_roundtrip`
- [ ] `test_empty_belief_factory`
- [ ] `test_done_terminal`
- [ ] `test_v_nan_allowed` — v 字段允许 NaN
- [ ] `test_pickle_roundtrip`
- [ ] **`test_field_count`** — 字段集合 == `{o, a, r, delta, pi_mve, v, tau, cap, c_hat, z_hat, t, done}`

### 1.5 Spec 05/06/07 — V4Config 5 层 + 3 preset + 归档

```powershell
pytest tests/configs/ -v
pytest tests/test_legacy_import_warning.py -v
pytest tests/schemas/test_context_schema.py -v
```

- [ ] `tests/configs/test_env_config.py` — `EnvConfig.__post_init__` 校验：`len(type_assignment)==N`、`c_mode∈{static,oscillate,random_walk}`、`alpha_min < alpha_max`
- [ ] `tests/configs/test_v4_config.py` — 5 sub-config 拼装、`from_preset('medium')` 可执行
- [ ] **`tests/configs/test_presets.py`**（hard gate）— 三 preset **字段值 100% 匹配 Ch3.9 Table**：
  - [ ] Easy: `N=2, L=8, K=8, M=1, T_max=100, c_mode=static`
  - [ ] Medium: `N=4, L=16, K=20, M=3, T_max=200, c_mode=static`，`type=(α,α,β,β)`
  - [ ] Hard: `N=8, L=24, K=40, M=5, T_max=300, **c_mode=oscillate**`
- [ ] `tests/configs/test_legacy_config.py` — `LegacyConfig` 默认实例可序列化
- [ ] **`tests/test_legacy_import_warning.py`** — `import hyper_mve._legacy_v4_7` 触发 `DeprecationWarning(stacklevel=2)`
- [ ] `tests/schemas/test_context_schema.py` — ContextSchema：
  - [ ] `d_id+d_type+d_cap == d_role == 32`（精确填满，无 pad）
  - [ ] `2*d_belief_proj == d_belief == 32`
  - [ ] `d_ctx_aug == 80 == 16+32+32`
  - [ ] 任一约束破坏 → ValueError

### 1.6 归档自检脚本

```powershell
python scripts/audit_legacy.py
```

- [ ] 输出 `ALL FILES ACCOUNTED FOR`（v4.7 旧文件全部进入 `_legacy_v4_7/`）。
- [ ] git tag 检查（手动）：`git tag --list v4.7-final` 返回非空。

---

## §2 · Pkg-02 ResourceCommons 环境

> 目标：① 公式 3.1-3.10 解析解 ±1e-4；② Table 3.5.4 + Ch4.1.1 偏导四象限 100% 对应；③ env.info 三 Oracle 字段（`c_true / types / caps`）齐全；④ 1000 episode 随机回合无 NaN/Inf。

### 2.1 一次性运行

```powershell
pytest tests/envs/ -v
```

- [ ] 全部 9 个测试文件通过（共 ≈105 个 test cases）。

### 2.2 Spec 01 — State 形式化（`test_state.py`）

- [ ] `test_state_construct_minimal` / `test_state_copy_independent`
- [ ] `test_state_dtype_strict` — positions int32, stocks float32, types int8, actions int64
- [ ] `test_state_caps_shared_by_reference` — tuple 不可变
- [ ] `test_state_c_history_size_t_max_plus_one`

### 2.3 Spec 02 — Resource Dynamics（`test_dynamics.py`，hard gate）

- [ ] **α(c) 端点**：`test_alpha_linear_endpoints` — α(0)=0.02, α(1)=0.20, α(0.5)=0.11
- [ ] `test_alpha_out_of_range` — c ∉ [0,1] → AssertionError
- [ ] **neighbor factor**：`test_neighbor_factor_full_neighbors` ≈ 0.997；`test_neighbor_factor_empty_neighbors_fallback` ≈ 0.426；`test_neighbor_factor_all_neighbors_depleted`
- [ ] **`test_logistic_regen_5steps_against_analytic`** — 5 步递推与解析解 ±1e-4
- [ ] 数值安全：`test_step_dynamics_clip_no_negative` / `_no_overflow`
- [ ] **公平分摊**：`test_fair_share_three_on_one_cell` (3 agent 同格 q=3 各得 1.0)、`_eta_caps_when_share_exceeds_eta`、`_off_cell_no_harvest`、`_harvest_mask_off`

### 2.4 Spec 03 — Fehr-Schmidt Reward（`test_rewards.py`，**两处 hard gate**）

- [ ] **Table 3.5.4 四象限**（4 个测试，数值精确）：
  - [ ] `test_table_3_5_4_lean_advantage` — c=0, Δ=+1, R-u = **−0.3**
  - [ ] `test_table_3_5_4_lean_disadvantage` — c=0, Δ=−1, R-u = **−1.0**
  - [ ] `test_table_3_5_4_abundance_advantage` — c=1, Δ=+1, R-u = **+0.3**
  - [ ] `test_table_3_5_4_abundance_disadvantage` — c=1, Δ=−1, R-u = **+1.0**
- [ ] **Ch4.1.1 偏导四象限**（autograd 提取）：
  - [ ] `test_beta_grad_four_quadrants` ∂R^β/∂u_i ∈ **{0.7, 2.0, 1.3, 0.0}** 对应四象限
- [ ] φ/ψ：`test_phi_endpoints` (0→0.5, 1→−0.5, 0.5→0)、`test_psi_zero_at_origin`、`test_psi_advantage_uses_lambda_adv`、`test_psi_disadvantage_uses_lambda_disadv`
- [ ] Δ 计算：`test_delta_zero_when_all_equal`、`test_delta_unit_advantage_four_agents`、`test_delta_single_agent_degenerate`
- [ ] 类型分流：`test_alpha_reward_has_no_preference_term`、`test_beta_reward_includes_preference_term`、`test_mixed_types_alpha_unaffected_beta_modulated`
- [ ] `test_move_cost_applied_to_all_types`、`test_returns_deltas_matching_compute_delta`
- [ ] **`test_alpha_grad_constant_one`** — α 类型 ∂R^α/∂u_i ≡ 1.0（100 个随机 seed autograd）

### 2.5 Spec 04 — Six-block Observation（`test_observations.py`）

- [ ] 形状/类型：`test_obs_total_dim_matches_layout`、`test_obs_dtype_float32`、`test_joint_observation_shape`
- [ ] **Self-Info 严格性**（v4 关键）：
  - [ ] `test_type_block_is_own_one_hot` — type 块=自身 2 维
  - [ ] `test_capability_block_is_own_cap` — cap 块=自身 4 维
  - [ ] **`test_other_agents_type_does_not_leak_into_obs`** — 改其他人 type，自己 obs 不变
  - [ ] **`test_other_agents_cap_does_not_leak_into_obs`** — 同上
- [ ] FOV：`test_resource_outside_fov_is_zero_padded`、`test_neighbor_presence_flag_distinguishes_visible_from_padded`
- [ ] `test_global_block_carries_c_t`

### 2.6 Spec 05 — Context Evolution（`test_context_evolution.py`）

- [ ] **Static**：`test_static_is_constant`（100 步不变）、`test_static_initial_in_unit_interval`
- [ ] **Oscillate**：`test_oscillate_step_zero_at_origin_and_period`（c(0)=c(25)=c(50)=0.5）、`test_oscillate_amplitude_in_unit_interval`、`test_oscillate_rejects_invalid_period`
- [ ] **RandomWalk**：`test_random_walk_clipped_to_unit_interval`、`test_random_walk_drifts_over_time`、`test_random_walk_shock_bounded`、`test_random_walk_reproducible_under_same_seed`、`test_random_walk_validates_inputs`
- [ ] **Factory**：`test_factory_returns_each_mode`、`test_factory_propagates_oscillate_period`、`test_factory_unknown_mode_raises`

### 2.7 Spec 06 — Patchy Spawn（`test_spawn.py`）

- [ ] 形状：`test_spawn_shape_medium`（hotspots (3,2), resources (20,2)）
- [ ] 范围：`test_spawn_in_grid`（∈ [0, L)）
- [ ] 分配：`test_distribute_K_over_M_balanced` (K=20,M=3 → [7,7,6])、`_exact_divide`、`_zero_M_raises`
- [ ] **`test_hotspot_min_distance`** — 两两 ≥ min_distance
- [ ] 复现：`test_spawn_reproducible_same_seed`、`test_spawn_different_seeds_different`
- [ ] **`test_resources_clustered_near_hotspots`** — ≥90% 落在某 hotspot 2σ 内
- [ ] preset：`test_spawn_easy_config`、`test_spawn_hard_config`、`test_min_distance_too_large_warns`

### 2.8 Spec 07 — Difficulty Presets（env 端，`test_presets_env.py`）

- [ ] `test_easy_env_dimensions` (obs_dim=45)、`test_medium_env_dimensions` (obs_dim=99)、`test_hard_env_dimensions` (obs_dim=195)
- [ ] `test_easy_type_assignment_1a_1b`、`test_medium_type_assignment_2a_2b`、`test_hard_type_assignment_4a_4b`
- [ ] **`test_hard_c_mode_is_oscillate`** — Hard 用 oscillate
- [ ] **`test_preset_switch_zero_branching`** — env 代码无 `if preset == ...`
- [ ] `test_no_hardcoded_preset_name_in_env_module`（源代码扫描）
- [ ] `test_custom_type_assignment_override_via_replace`

### 2.9 Spec 08 — gym API + info dict v4 Oracle（`test_env_info_oracle.py` + `test_env_integration.py`，hard gate）

- [ ] **info 三 Oracle 字段**：
  - [ ] `test_info_has_oracle_fields` — 含 `c_true: float`, `types: (N,) int8`, `caps: tuple[CapabilityVector,...]`
  - [ ] `test_info_oracle_matches_state` — `info["c_true"]==state.c_t`，`info["types"]==state.agent_types`
- [ ] eval-only：`test_info_eval_fields_present` — `hotspot_centers (M,2)`, `resource_state (K,3)`
- [ ] schema 版本：`test_info_schema_version_marker` — `_info_schema_version="v4.0"`、`_oracle_fields`、`_eval_only_fields`
- [ ] gym 契约：`test_step_returns_five_tuple_with_right_shapes`、`test_observation_space_matches_obs_shape`、`test_action_space_is_multidiscrete`
- [ ] reset：`test_reset_same_seed_same_obs`
- [ ] reset options：`test_reset_options_force_c`、**`test_reset_options_force_c_zero_regression`**（c=0.0 不被 falsy 短路）、`_c_out_of_range_rejected`、`_force_types`、`_none_falls_back_to_cfg`
- [ ] terminal：`test_done_at_t_max`
- [ ] render：`test_render_rgb_array (H,W,3) uint8`、`test_render_invalid_mode_raises`
- [ ] 工厂 & close：`test_make_resource_commons_factory`、`test_close_is_noop`
- [ ] **集成**：`tests/envs/test_env_integration.py` — 1000 episodes × 多步骤，shape/范围/types 不变性

### 2.10 烟囱脚本：1000 episode 随机回合

```powershell
python hyper_mve/scripts/test_resource_commons.py easy 1000
python hyper_mve/scripts/test_resource_commons.py medium 1000
python hyper_mve/scripts/test_resource_commons.py hard 1000
```

- [ ] 三难度都跑通，**无 NaN/Inf**。
- [ ] Medium 平均 step latency ≤ 1 ms（≥1000 step/s 性能要求）。
- [ ] 输出含 `mean episode return`、`steps/sec`，记录到测试日志。

### 2.11 可选：Proposition 3.1 实证

```powershell
python hyper_mve/scripts/proposition_3_1_validation.py
```

- [ ] 输出 Pareto-Nash gap 曲线（`docs/` 或 `runs/` 下生成图）。

---

## §3 · Pkg-03 TriContextEncoder & BeliefNet

> 目标：① 三路输出维度 80=16+32+32；② head_c 输出 `(B,N)` 标量、head_opp 输出 `(B,N,N-1,2)` softmax；③ z_hat 顺序约定（agent_id 升序跳过 self）三处一致；④ L_div hinge 阻止 collapse；⑤ 合成数据 5K 步收敛 gate。

### 3.1 一次性运行

```powershell
pytest tests/models/ -v
```

- [ ] 6 个测试文件全通过（≈80+ test cases）。

### 3.2 Spec 02 — CEncoder（`test_c_encoder.py`）

- [ ] `test_output_shape_default` — `(B, 16)`
- [ ] `test_output_shape_custom_d_c` — d_c 可配置
- [ ] **`test_input_shape_assertion`** — 拒绝 `(B,)`，必须 `(B,1)`
- [ ] `test_gradient_flow` — 全 MLP 参数有梯度
- [ ] `test_distinct_c_distinct_output` — 不同 c 输出不同
- [ ] `test_param_count` — ≈1.6K 参数
- [ ] `test_boundary_values` — c=0.0/1.0 无 NaN/Inf
- [ ] `test_no_internal_layernorm` — 内部无 LN（外置 P7）

### 3.3 Spec 03 — RoleEncoder（`test_role_encoder.py`，**v4 关键**）

- [ ] `test_output_shape` — `(B, N, 32)`
- [ ] **`test_role_dim_exact_fill_v4`** — d_id(8)+d_type(8)+d_cap(16)=32 精确填满，无 pad
- [ ] `test_role_dim_assertion_on_mismatch` — d_role≠32 → 抛错
- [ ] **`test_type_emb_distinct_for_alpha_beta`** — α/β 嵌入不同
- [ ] `test_id_emb_distinct` — 不同 agent_id 子段不同
- [ ] **`test_self_info_severity_only_own_type`** — 拒绝 (B,N,N) 全 type 矩阵
- [ ] **cap normalize**：
  - [ ] `test_cap_normalization_handles_scale_diff` — η=0.5 与 ζ=30 60× 量级差 → [0,1]^4
  - [ ] `test_cap_normalization_matches_pkg01_normalize` — 与 `CapabilityVector.normalize()` 数学等价
  - [ ] `test_cap_norm_buffer_registered` — `_cap_lo / _cap_hi` 为 buffer
  - [ ] `test_cap_norm_buffer_device_migration` — `.cuda()` 同步
- [ ] `test_cap_mlp_no_internal_layernorm`（P7）
- [ ] `test_gradient_flow`、`test_episode_invariance`、`test_param_count_medium`、`test_n_scaling`

### 3.4 Spec 04+05 — BeliefNet GRU + 两 Head（`test_belief_net.py`）

- [ ] **GRU 主干**：
  - [ ] `test_init_hidden_zeros` — `(B, N, 128)` 全零
  - [ ] `test_init_hidden_device` — 自动选 model device
  - [ ] `test_gru_input_dim_independent_of_latent` — gru_input_dim=64 独立于 latent_dim
  - [ ] **`test_gru_step_seq_equiv`** — step 与 forward(seq) 等价（atol 1e-5）
  - [ ] `test_hidden_state_evolution` — 隐状态演化
  - [ ] `test_shared_gru_weights` — 单 GRUCell 跨 N agent
  - [ ] **`test_layernorm_after_gru`** — `ln_belief` 必须存在（Ch4.6.2 Defense 2）
  - [ ] `test_no_bidirectional` — 单向因果
- [ ] **两 Head**：
  - [ ] `test_head_c_output_shape` — `(B, N)` sigmoid
  - [ ] **`test_head_c_is_scalar_per_agent_v4`** — `(B,N)` 标量，**不是** `(B,N,16)`
  - [ ] `test_head_opp_output_shape` — `(B, N, N-1, 2)` softmax
  - [ ] **`test_z_hat_agent_id_order_convention`** — `z_hat[b,i,k]` 对应 `agent_id=(k if k<i else k+1)`
  - [ ] `test_softmax_simplex_property` — 概率单纯形
  - [ ] **`test_head_opp_oracle_substitution`** — Stage 1 oracle_z 替换 → forward 返回 oracle_z
  - [ ] **`test_head_opp_still_trained_in_oracle_mode`** — Stage 1 仍然反向传播 head_opp（`get_head_opp_predictions`）
  - [ ] `test_opp_id_emb_independent_of_role_id_emb` — `BeliefIdEmbedding` 独立于 `RoleEncoder.id_emb`
  - [ ] `test_n_variation_heads` — N=2/N=8 都跑通

### 3.5 Spec 06 — BeliefNet Losses（`test_belief_losses.py`，**critical mapping**）

- [ ] **L_c**：
  - [ ] `test_l_c_zero_loss_at_perfect_pred`
  - [ ] `test_l_c_positive_when_err`
  - [ ] `test_l_c_gradient_direction`
  - [ ] **`test_l_c_broadcast_correctness`** — `c_true (B,T)` 自动广播到 `(B,T,N)`
  - [ ] `test_l_c_wrong_input_shape_rejected` — `(B,T,N)` 输入被拒（违 spec）
- [ ] **L_opp（Oracle CE）**：
  - [ ] `test_l_opp_oracle_ce` — Oracle CE，不是动作预测
  - [ ] `test_l_opp_uniform_pred_baseline` — 均匀 (0.5,0.5) → loss=ln 2≈0.693
  - [ ] `test_l_opp_agent_id_order_consistency` — 顺序 [0,1,3] 等
  - [ ] **`test_l_opp_index_mapping_correctness`** — 完美对角预测 → loss≈0（验证 (i,k)→j 映射）
  - [ ] **`test_l_opp_index_mapping_specific_pairs`** — 每对 (i,k) → 正确 opp agent_id
- [ ] **L_div（hinge variance, P1）**：
  - [ ] `test_l_div_hinge_zero_when_high_variance` — var≥σ² → 0
  - [ ] `test_l_div_hinge_positive_when_low_variance`
  - [ ] **`test_l_div_collapse_protection`** — var=0 → loss=σ²=0.01
- [ ] 组合：`test_belief_loss_combination`（λ_c=1, λ_opp=0.5, λ_div=0.01）、`test_belief_loss_mask`

### 3.6 Spec 07 — PermutationInvariantPool（`test_permutation_pool.py`）

- [ ] 形状：`test_mean_pool_output_shape` / `_max` / `_attention` — `(B,N,N-1,F) → (B,N,F)`
- [ ] **置换不变**：`test_mean_pool_set_invariant`、`test_max_pool_set_invariant`、`test_attention_pool_set_invariant`（共享 K/V）
- [ ] 数值：`test_mean_pool_computes_average`（(1+3+5)/3=3）、`test_max_pool_computes_max`
- [ ] 参数：`test_mean_pool_no_params`、`test_max_pool_no_params`、`test_attention_pool_has_params`
- [ ] 工厂：`test_make_pool_factory`、`test_make_pool_unknown_kind`
- [ ] 边界：**`test_pool_n_minus_one_equals_1`**（N=2，N-1=1 单元素）

### 3.7 Spec 01 + 08 — TriContextEncoder & 集成契约（`test_tri_context.py`）

- [ ] **维度**：
  - [ ] **`test_output_dim`** — ctx_i 形状 `(B, N, 80)`
  - [ ] `test_d_ctx_aug_property` — 80=16+32+32
  - [ ] `test_role_dim_exact` — 8+8+16=32
- [ ] **梯度流**：`test_gradient_flow` — 三路全部参数有梯度
- [ ] **顺序**：`test_concat_order` — `[0:16]=c_ctx`, `[16:48]=role`, `[48:80]=belief`
- [ ] `test_permutation_invariance_belief_mean_pool` — z_hat 行 perm 不变
- [ ] `test_c_ctx_only_dim` — `forward_c_ctx_only` → `(B,16)`
- [ ] `test_c_t_input_shapes` — `(B,)` 与 `(B,1)` 都接受
- [ ] `test_n_variation` — N=2 / N=8 形状正确
- [ ] BeliefEncoder：`test_belief_encoder_output_shape`、`test_belief_encoder_no_internal_ln`、`test_belief_encoder_gradient_flow`、`test_belief_encoder_concat_order`（前 16 来自 proj_c_hat，后 16 来自 proj_z_pooled）

### 3.8 合成数据收敛 gate（`scripts/test_belief_net_synth.py`）

```powershell
python hyper_mve/scripts/test_belief_net_synth.py
# 默认 5K-10K step；建议 GPU
```

- [ ] **`head_c MSE < 0.05`** 在 5K step 内（合成 c=0.7）
- [ ] **`head_opp accuracy > 80%`** 在 5K step 内（α/β 混合轨迹）
- [ ] **`Var(b_i^t) ≥ 0.1`** 在 10K step 后（L_div hinge 阻止 collapse）
- [ ] TensorBoard 输出：`runs/belief_synth/<run_tag>/` 三条 loss 曲线 + acc/MSE/var 标量
- [ ] 失败诊断：若 head_opp acc 卡在 50% → 检查 z_hat agent_id 顺序错位（spec 04 / 06 / 08 三处一致性）

### 3.9 命题 3.1 / 集成契约（手动审计）

- [ ] **Self-Info 严格性**（spec 03 + 08）— `RoleEncoder` 拒绝 `(B,N,N)` 全 type 矩阵；env obs `type` 块仅 2 维。
- [ ] **Oracle vs Self-Info 分离**（spec 08 contract 2）— `L_opp` 训练用 `info["types"]`；评估时不喂 oracle_z（Pkg-07 责任）。
- [ ] **课程 Stage 1/2/3** 设计就位：`build_oracle_z_seq` 与 head_opp 顺序映射严格一致（待 Pkg-05 trainer 联动验证）。
- [ ] **共享 BeliefNet 公平性**：7 个 baseline 共用同一 `BeliefNet` 类（Pkg-06a/b 责任，本期不测，但确认接口稳定）。

---

## §4 · 跨包 hard gate 速查（必通过）

| # | Gate | 测试位置 | 期望 |
|---|------|----------|------|
| H1 | Table 3.5.4 四象限 | `test_rewards.py` | R−u ∈ {−0.3, −1.0, +0.3, +1.0} |
| H2 | Ch4.1.1 偏导四象限 | `test_rewards.py` (autograd) | ∂R^β/∂u_i ∈ {0.7, 2.0, 1.3, 0.0} |
| H3 | logistic 5 步解析解 | `test_dynamics.py` | ±1e-4 |
| H4 | Self-Info 不泄漏 | `test_observations.py` + `test_role_encoder.py` | 改他人 type/cap，自己 obs / role 不变 |
| H5 | env.info Oracle 字段 | `test_env_info_oracle.py` | `c_true / types / caps` 全齐 |
| H6 | 1000 episode 无 NaN/Inf | `scripts/test_resource_commons.py` | 三难度都过 |
| H7 | TriContextEncoder 80 维 | `test_tri_context.py` | `(B,N,80)` |
| H8 | head_c scalar `(B,N)` | `test_belief_net.py` | v4 vs v3 关键差异 |
| H9 | head_opp `(B,N,N-1,2)` softmax | `test_belief_net.py` | v4 Oracle 2 分类 |
| H10 | z_hat 顺序约定（3 处一致） | `test_buffer_record.py` + `test_belief_net.py` + `test_belief_losses.py` | `(k if k<i else k+1)` |
| H11 | L_div hinge 阻 collapse | `test_belief_losses.py` + 合成脚本 | 10K 步后 var≥0.1 |
| H12 | 合成数据收敛 gate | `scripts/test_belief_net_synth.py` | head_c MSE<0.05；head_opp acc>80% |
| H13 | preset 字段 vs Ch3.9 Table | `tests/configs/test_presets.py` | 三 preset 100% 匹配 |

---

## §5 · 可选：性能 / mypy / 覆盖率

```powershell
# mypy 严格模式（schemas + envs + models）
pip install mypy
mypy --strict hyper_mve\schemas hyper_mve\configs hyper_mve\envs\resource_commons hyper_mve\models

# 覆盖率
pip install pytest-cov
pytest tests/ --cov=hyper_mve --cov-report=term-missing
```

- [ ] mypy 零错误（schemas + configs + envs + models）
- [ ] 覆盖率 ≥ 85%（schemas/configs ≥ 90%，envs/models ≥ 85%）
- [ ] step 性能（Medium，CPU 单线程）≥ 1000 step/s

---

## §6 · 失败排查 & 复测建议

| 症状 | 大概率原因 | 排查入口 |
|------|------------|---------|
| `test_capability_normalize_*` fail | `_constants.py` CAP_NORM_LO/HI 与 ETA_RANGE 不同步 | `schemas/_constants.py` |
| `test_table_3_5_4_*` 数值偏移 | KAPPA / λ_disadv / λ_adv 常量错 | `_constants.py` + `rewards.py` |
| `test_logistic_regen_*` 偏差 > 1e-4 | α(c) 公式或 dt 步长 | `dynamics.py` |
| `test_other_agents_type_does_not_leak_into_obs` 失败 | observation type 块写成 (N,2) 而非 (2,) | `observations.py` |
| `test_z_hat_*` 三处不一致 | head_opp / l_opp / build_oracle_z_seq 任一处 (i,k)→j 错位 | `belief_net.py`、`belief_losses.py` |
| `test_head_opp_still_trained_in_oracle_mode` fail | Stage 1 把 oracle_z 直接喂 forward 没有保留 head_opp 反向 | `belief_net.py.get_head_opp_predictions` |
| 合成脚本 head_opp acc 卡 50% | z_hat 顺序错位（H10）或 oracle types 取错 agent | spec 03/05/06 三处一致性 |
| `test_preset_switch_zero_branching` fail | env.py 出现 `if preset == "easy"` 等硬编码 | `envs/resource_commons/env.py` |

---

## §7 · 通过后的归档

- [ ] 完整通过 §1–§3 全部勾选项后，把本清单和 pytest 输出提交：
  ```powershell
  pytest tests/ -q --tb=short > TEST_REPORT_pkg1-3.txt
  ```
- [ ] 在 PR 描述里附上：
  - 各 spec hard gate 通过截图 / 文本片段
  - 1000 episode 烟囱脚本三难度的 `mean return / step latency`
  - 合成数据脚本的 head_c MSE / head_opp acc / b 方差三个最终值
- [ ] git tag：`pkg-03-passed`（如适用）。

> **下一步**：本清单 §1–§3 全部 PASS 是启动 Pkg-04 (DualHyperNetwork v2) 的前置条件。Pkg-04 直接消费 `TriContextEncoder.forward(...) → (B,N,80)` 与 `BeliefNet.step/forward(...)`，任何 §3 hard gate 不过都会让 Pkg-04 反向传播链断裂或拟合不上。
