Scratch files cleaned up. Now I'll emit the final structured report.

---

# TB Scalar Inventory & Run-Completion Report — `D:/RL/hyper_mve/runs/suite/`

Extraction method: `tensorboard.backend.event_processing.event_accumulator.EventAccumulator` via `D:/RL/.venv/Scripts/python.exe`, with `size_guidance={"scalars": 0}` (unbounded). Source TB binary at `D:/RL/.venv/Scripts/tensorboard.exe`. No torch / numpy / matplotlib imports.

All 18 run dirs across all 7 cells were processed successfully. Two `registry.jsonl` rows out of 18 unique `(cell, seed)` runs are present twice (pending → completed pattern) in `zero_shot_easy` and `abl4_joint_easy_n2`; the latest record per run_id is treated as authoritative below.

## TB scalar inventory

Each cell uses the same 101-scalar schema (collect/* eval/planner/* eval/prior/* eval/planner_prior_gap loss/* loss_raw/* diag/* perf/*). `lora/medium_base` is the only cell with an extended 104-tag schema (adds `diag/cos_pred_same`, `diag/cos_rew_same`, `diag/l2_rew_same`).

Per-cell schema summary (tags × seeds, with last_step and n_steps min/max across tags within run):

### Cell: `main_comparison_easy`  (union: 101 tags, all seeds identical schema)

| seed_run | n_tags | max_last_step | min_last_step | max_n_steps | min_n_steps |
|---|---|---|---|---|---|
| hyper_seed0_row0_776259c5 | 101 | 137400 | 137000 | 1374 | 138 |
| hyper_seed1_row1_31d800fe | 101 | 137300 | 137000 | 1373 | 138 |
| hyper_seed2_row2_f997c574 | 101 | 134000 | 134000 | 1340 | 135 |

(Eval-bus tags `eval/*` log at every ~1000 train-step iter → 134-138 samples; train-bus tags `loss/*`, `diag/*`, `collect/*`, `perf/*` log every iter → 1340-1374 samples.)

### Cell: `zero_shot_easy`  (union: 101 tags, all seeds identical schema)

| seed_run | n_tags | max_last_step | min_last_step | max_n_steps | min_n_steps |
|---|---|---|---|---|---|
| hyper_seed0_row0_04a30952 | 101 | 200000 | 200000 | 2000 | 201 |
| hyper_seed1_row1_23802fc5 | 101 | 200000 | 200000 | 2000 | 201 |
| hyper_seed2_row2_1e8b092c | 101 | 200000 | 200000 | 2000 | 201 |

All three runs ran to exactly 200000 env-steps (200001 eval samples, 2000 train samples) — clean completion.

### Cell: `abl4_joint_easy_n2`  (union: 101 tags, all seeds identical schema)

| seed_run | n_tags | max_last_step | min_last_step | max_n_steps | min_n_steps |
|---|---|---|---|---|---|
| hyper_seed0_row0_987aace8 | 101 | 100000 | 100000 | 1000 | 101 |
| hyper_seed1_row1_2c1abcc3 | 101 | 100000 | 100000 | 1000 | 101 |
| hyper_seed2_row2_b4dc5ade | 101 | 100000 | 100000 | 1000 | 101 |

All three runs ran to exactly 100000 env-steps (101 eval samples, 1000 train samples) — clean completion. Note this is **half** the env-steps of `zero_shot_easy` (likely a shorter ablation budget).

### Cell: `lora/duo_base`  (union: 101 tags, single seed)

| seed_run | n_tags | max_last_step | min_last_step | max_n_steps | min_n_steps |
|---|---|---|---|---|---|
| hyper_seed0_row0_74fd892a | 101 | 188800 | 188000 | 1888 | 189 |

### Cell: `lora/duo_film`  (union: 101 tags, all seeds identical schema)

| seed_run | n_tags | max_last_step | min_last_step | max_n_steps | min_n_steps |
|---|---|---|---|---|---|
| hyper_seed0_row0_3d4ff0f7 | 101 | 240400 | 240000 | 2404 | 241 |
| hyper_seed1_row1_636cc98d | 101 | 236100 | 236000 | 2361 | 237 |

### Cell: `lora/duo_film_fc2`  (union: 101 tags, all seeds identical schema)

| seed_run | n_tags | max_last_step | min_last_step | max_n_steps | min_n_steps |
|---|---|---|---|---|---|
| hyper_seed0_row0_1d8756ae | 101 | 239600 | 239000 | 2396 | 240 |
| hyper_seed1_row1_76f743bf | 101 | 211300 | 211000 | 2113 | 212 |
| hyper_seed2_row2_22d090b8 | 101 | 211400 | 211000 | 2114 | 212 |

### Cell: `lora/medium_base`  (union: **104** tags, all seeds identical schema)

Extra tags present only in this cell: `diag/cos_pred_same`, `diag/cos_rew_same`, `diag/l2_rew_same` (the in-context same-perspective diagnostic triplet, complementing the existing `*_cross` tags).

| seed_run | n_tags | max_last_step | min_last_step | max_n_steps | min_n_steps |
|---|---|---|---|---|---|
| hyper_seed0_row0_fd7ae6e2 | 104 | 66400 | 66000 | 664 | 67 |
| hyper_seed1_row1_f3bf2b6b | 104 | 66400 | 66000 | 664 | 67 |
| hyper_seed2_row2_23999baf | 104 | 66400 | 66000 | 664 | 67 |

All three runs reached only ~66400 train steps — far short of the lora/duo_film_fc2 sister cell (~211–240 k). See Issues.

### Full scalar tag list (101-tag schema, common to all non-medium_base cells)

```
collect/H_pi_mve_fresh, collect/epsilon, collect/q_gap, collect/q_std, collect/return_alpha, collect/return_beta, collect/return_total, collect/uniform_frac,
diag/cos_pred_cross, diag/cos_rew_cross, diag/kl_mve_to_pred, diag/l2_rew_cross, diag/mode_match_pred_mve, diag/norm_rew_alpha, diag/norm_rew_beta,
diag/pi_mve_entropy, diag/pi_mve_max_prob, diag/pi_pred_entropy, diag/pi_pred_max_prob, diag/target_age_steps,
eval/planner/{ep_len,fairness,fairness_c0.2,fairness_c0.5,fairness_c0.8,pi_mve_entropy,q_std,return_alpha,return_alpha_c0.2,return_alpha_c0.5,return_alpha_c0.8,return_beta,return_beta_c0.2,return_beta_c0.5,return_beta_c0.8,return_total,return_total_c0.2,return_total_c0.5,return_total_c0.8,sustainability,sustainability_c0.2,sustainability_c0.5,sustainability_c0.8,tragedy,tragedy_c0.2,tragedy_c0.5,tragedy_c0.8,uniform_frac,welfare_physical,welfare_physical_c0.2,welfare_physical_c0.5,welfare_physical_c0.8},
eval/planner_prior_gap,
eval/prior/{ep_len,fairness,fairness_c0.2,fairness_c0.5,fairness_c0.8,return_alpha,return_alpha_c0.2,return_alpha_c0.5,return_alpha_c0.8,return_beta,return_beta_c0.2,return_beta_c0.5,return_beta_c0.8,return_total,return_total_c0.2,return_total_c0.5,return_total_c0.8,sustainability,sustainability_c0.2,sustainability_c0.5,sustainability_c0.8,tragedy,tragedy_c0.2,tragedy_c0.5,tragedy_c0.8,welfare_physical,welfare_physical_c0.2,welfare_physical_c0.5,welfare_physical_c0.8},
loss/{belief,belief_c,belief_div,belief_opp,consist,lambda_b,lr,main,policy,reward,total,value},
loss_raw/{L_consist,L_policy,L_reward,L_value},
perf/{collect_sec_per_iter,env_steps_per_sec,train_sec_per_iter}
```

`lora/medium_base` adds `diag/cos_pred_same, diag/cos_rew_same, diag/l2_rew_same` (104 tags total).

## Run-completion status (from registry.jsonl)

`zero_shot_easy/registry.jsonl` and `abl4_joint_easy_n2/registry.jsonl` each contain two rows per run_id — an initial `pending` write at sweep dispatch time and a later `completed` overwrite. Other cells only ever contain a single `pending` row per run, even when TB data exists. Below uses the latest record per run_id.

| cell | run_id (short) | variant | seed | status | started_at | completed_at | walltime_s | return_mean | return_zero_shot_unseen | regret_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| main_comparison_easy | 062f4968 | hyper | 0 | **pending** | 2026-06-23T12:42:04Z | null | null | null | null | null |
| main_comparison_easy | 05f8e6ed | hyper | 1 | **pending** | 2026-06-23T12:42:04Z | null | null | null | null | null |
| main_comparison_easy | dd4104b7 | hyper | 2 | **pending** | 2026-06-23T12:42:04Z | null | null | null | null | null |
| zero_shot_easy | 5262270e | hyper | 0 | **completed** | 2026-06-23T12:42:04Z | 2026-06-27T23:22:44Z | 271403.49 | 138.2746 | 138.5066 | 0.0 |
| zero_shot_easy | b7c954c3 | hyper | 1 | **completed** | 2026-06-23T12:42:04Z | 2026-06-27T23:25:15Z | 271141.23 | 133.4332 | 134.5965 | 0.0 |
| zero_shot_easy | bc40781b | hyper | 2 | **completed** | 2026-06-23T12:42:04Z | 2026-06-28T01:15:39Z | 271198.07 | 120.8770 | 119.5246 | 0.0 |
| abl4_joint_easy_n2 | 21f26d88 | hyper | 0 | **completed** | 2026-06-23T12:42:04Z | 2026-06-24T19:59:21Z | 112636.66 | 120.8532 | 119.4724 | 0.0 |
| abl4_joint_easy_n2 | 857a2115 | hyper | 1 | **completed** | 2026-06-23T12:42:04Z | 2026-06-24T21:55:41Z | 119617.19 | 116.0313 | 117.3416 | 0.0 |
| abl4_joint_easy_n2 | bf57a626 | hyper | 2 | **completed** | 2026-06-23T12:42:04Z | 2026-06-24T20:06:14Z | 113050.19 | 116.6510 | 117.1935 | 0.0 |
| lora/duo_base | 4e2951dd | hyper | 0 | **pending** | 2026-06-23T12:36:33Z | null | null | null | null | null |
| lora/duo_film | ba92a816 | hyper | 0 | **pending** | 2026-06-23T12:36:33Z | null | null | null | null | null |
| lora/duo_film | 1bc4763a | hyper | 1 | **pending** | 2026-06-23T12:36:33Z | null | null | null | null | null |
| lora/duo_film_fc2 | 8a2bbbd0 | hyper | 0 | **pending** | 2026-06-23T12:36:33Z | null | null | null | null | null |
| lora/duo_film_fc2 | 0fd3bbdb | hyper | 1 | **pending** | 2026-06-23T12:36:33Z | null | null | null | null | null |
| lora/duo_film_fc2 | 8350fc93 | hyper | 2 | **pending** | 2026-06-23T12:36:33Z | null | null | null | null | null |
| lora/medium_base | 7b0b7a8d | hyper | 0 | **pending** | 2026-06-23T12:36:33Z | null | null | null | null | null |
| lora/medium_base | ffb96ae1 | hyper | 1 | **pending** | 2026-06-23T12:36:33Z | null | null | null | null | null |
| lora/medium_base | f901791d | hyper | 2 | **pending** | 2026-06-23T12:36:33Z | null | null | null | null | null |

Schema version: all rows are `pkg08-spec05-v1`. Git SHA: all rows pin `8ea7832b30a6a547ec6cd58072aa0c81c9342365` (clean tree). All variants are `hyper`.

Backfilled-metrics summary (return_mean / return_zero_shot_unseen / regret_mean):

- `zero_shot_easy` — all 3 seeds backfilled; mean return_mean ≈ 130.86; mean return_zero_shot_unseen ≈ 130.88; regret_mean = 0.0 across seeds.
- `abl4_joint_easy_n2` — all 3 seeds backfilled; mean return_mean ≈ 117.85; mean return_zero_shot_unseen ≈ 118.00; regret_mean = 0.0 across seeds.
- All five other cells: not backfilled (status still `pending`, metrics null), even though TB scalars exist and (for some) show long training horizons.

## Notable values  (final / mean-last-10% from TB)

Selected key scalars (eval @ last logged eval step; train @ last logged train step). All numbers rounded to 4 decimals.

### `main_comparison_easy`  (training ~134–137 k env-steps, status pending)

| seed | tag | last_step | final | mean_last_10% | min | max |
|---|---|---|---|---|---|---|
| seed0 | eval/planner/return_total | 137000 | 109.94 | 110.43 | 4.63 | 116.20 |
| seed0 | eval/prior/return_total | 137000 | 114.26 | 114.95 | -2.00 | 122.06 |
| seed0 | eval/planner/fairness | 137000 | 0.9070 | 0.8490 | 0.1021 | 0.9188 |
| seed0 | eval/planner/sustainability | 137000 | 0.7786 | 0.7757 | 0.7606 | 0.9967 |
| seed0 | eval/planner_prior_gap | 137000 | -4.32 | -4.51 | -17.72 | 50.21 |
| seed0 | loss/total | 137400 | 1.5149 | 1.5243 | 1.5077 | 2.2694 |
| seed0 | diag/mode_match_pred_mve | 137400 | 0.7793 | 0.7548 | 0.2978 | 0.7839 |
| seed1 | eval/planner/return_total | 137000 | 105.91 | 101.39 | 0.92 | 115.58 |
| seed1 | eval/prior/return_total | 137000 | 104.23 | 102.87 | -2.00 | 107.95 |
| seed1 | eval/planner/fairness | 137000 | 0.8682 | 0.8512 | 0.1979 | 0.9023 |
| seed1 | eval/planner_prior_gap | 137000 | 1.68 | -1.48 | -18.39 | 40.27 |
| seed1 | loss/total | 137300 | 1.5344 | 1.5276 | 1.5124 | 2.2375 |
| seed2 | eval/planner/return_total | 134000 | 100.45 | 98.69 | 4.63 | 108.51 |
| seed2 | eval/prior/return_total | 134000 | 99.60 | 96.38 | -1.97 | 110.30 |
| seed2 | eval/planner/fairness | 134000 | 0.7928 | 0.7924 | 0.1577 | 0.8862 |
| seed2 | eval/planner_prior_gap | 134000 | 0.86 | 2.31 | -32.48 | 49.27 |
| seed2 | loss/total | 134000 | 1.5246 | 1.5265 | 1.5101 | 2.2559 |

Across-seed (mean of mean_last_10%): `eval/planner/return_total` ≈ 103.50, `eval/prior/return_total` ≈ 104.73, `eval/planner_prior_gap` ≈ -1.23.

### `zero_shot_easy`  (200 k env-steps, status completed)

| seed | tag | last_step | final | mean_last_10% | min | max |
|---|---|---|---|---|---|---|
| seed0 | eval/planner/return_total | 200000 | 100.71 | 105.90 | 4.63 | 121.15 |
| seed0 | eval/prior/return_total | 200000 | 109.76 | 108.51 | -2.00 | 122.06 |
| seed0 | eval/planner/fairness | 200000 | 0.8747 | 0.8546 | 0.1021 | 0.9188 |
| seed0 | eval/planner_prior_gap | 200000 | -9.05 | -2.60 | -17.72 | 50.21 |
| seed0 | loss/total | 200000 | 1.5253 | 1.5221 | 1.5051 | 2.2694 |
| seed1 | eval/planner/return_total | 200000 | 103.83 | 101.83 | 0.92 | 115.58 |
| seed1 | eval/prior/return_total | 200000 | 102.93 | 102.84 | -2.00 | 110.80 |
| seed1 | eval/planner/fairness | 200000 | 0.8890 | 0.8905 | 0.1979 | 0.9203 |
| seed1 | eval/planner_prior_gap | 200000 | 0.90 | -1.01 | -18.39 | 40.27 |
| seed1 | loss/total | 200000 | 1.5317 | 1.5248 | 1.5064 | 2.2375 |
| seed2 | eval/planner/return_total | 200000 | 107.38 | 103.69 | 4.63 | 108.85 |
| seed2 | eval/prior/return_total | 200000 | 99.59 | 99.57 | -1.97 | 110.30 |
| seed2 | eval/planner/fairness | 200000 | 0.8325 | 0.8206 | 0.1577 | 0.8862 |
| seed2 | eval/planner_prior_gap | 200000 | 7.80 | 4.12 | -32.48 | 49.27 |
| seed2 | loss/total | 200000 | 1.5332 | 1.5263 | 1.5101 | 2.2559 |

Registry-backfilled `return_mean` (zero_shot_easy): seed0=138.27, seed1=133.43, seed2=120.88 — these are noticeably higher than the eval/planner/return_total mean_last_10% in TB (≈103.81), suggesting the registry `return_mean` is a separate end-of-training evaluation (likely from `eval_report.json`).

### `abl4_joint_easy_n2`  (100 k env-steps, status completed)

| seed | tag | last_step | final | mean_last_10% | min | max |
|---|---|---|---|---|---|---|
| seed0 | eval/planner/return_total | 100000 | 102.98 | 104.23 | 4.63 | 111.17 |
| seed0 | eval/prior/return_total | 100000 | 109.94 | 107.56 | -2.00 | 116.69 |
| seed0 | eval/planner/fairness | 100000 | 0.8785 | 0.8788 | 0.1021 | 0.9213 |
| seed0 | eval/planner_prior_gap | 100000 | -6.96 | -3.33 | -22.24 | 45.95 |
| seed1 | eval/planner/return_total | 100000 | 101.62 | 102.27 | 0.92 | 110.12 |
| seed1 | eval/prior/return_total | 100000 | 94.43 | 96.19 | -2.00 | 102.55 |
| seed1 | eval/planner_prior_gap | 100000 | 7.19 | 6.07 | -23.19 | 28.78 |
| seed2 | eval/planner/return_total | 100000 | 96.13 | 95.94 | 4.63 | 114.84 |
| seed2 | eval/prior/return_total | 100000 | 103.94 | 100.72 | -1.97 | 113.05 |
| seed2 | eval/planner_prior_gap | 100000 | -7.81 | -4.78 | -15.84 | 54.48 |

Registry-backfilled return_mean (abl4): seed0=120.85, seed1=116.03, seed2=116.65.

### `lora/duo_base`  (single seed, ~188 k env-steps, status pending)

| seed | tag | last_step | final | mean_last_10% | min | max |
|---|---|---|---|---|---|---|
| seed0 | eval/planner/return_total | 188000 | 262.50 | 261.68 | -0.35 | 293.12 |
| seed0 | eval/prior/return_total | 188000 | 263.08 | 257.16 | -2.00 | 290.84 |
| seed0 | eval/planner/fairness | 188000 | 0.9043 | 0.9015 | 0.1482 | 1.0000 |
| seed0 | eval/planner/sustainability | 188000 | 0.9085 | 0.9077 | 0.8986 | 1.0000 |
| seed0 | eval/planner_prior_gap | 188000 | -0.58 | 4.52 | -42.37 | 133.02 |
| seed0 | loss/total | 188800 | 1.0427 | 1.0199 | 0.4160 | 2.0595 |
| seed0 | diag/mode_match_pred_mve | 188800 | 0.7891 | 0.8029 | 0.6078 | 0.9980 |
| seed0 | diag/kl_mve_to_pred | 188800 | 0.2851 | 0.2605 | 0.0323 | 0.4673 |

### `lora/duo_film`  (2 seeds, ~236–240 k env-steps, status pending)

| seed | tag | last_step | final | mean_last_10% | min | max |
|---|---|---|---|---|---|---|
| seed0 | eval/planner/return_total | 240000 | 251.91 | 257.73 | -3.84 | 284.33 |
| seed0 | eval/prior/return_total | 240000 | 248.53 | 256.61 | -0.64 | 276.26 |
| seed0 | eval/planner/fairness | 240000 | 0.8698 | 0.8597 | 0.1206 | 1.0000 |
| seed0 | eval/planner_prior_gap | 240000 | 3.39 | 1.12 | -48.46 | 246.83 |
| seed0 | loss/total | 240400 | 1.0676 | 1.0446 | 0.3685 | 2.5113 |
| seed1 | eval/planner/return_total | 236000 | 248.62 | 218.82 | -2.41 | 263.92 |
| seed1 | eval/prior/return_total | 236000 | 153.75 | 117.67 | -3.48 | 172.46 |
| seed1 | eval/planner/fairness | 236000 | 0.8405 | 0.8375 | 0.1565 | 1.0000 |
| seed1 | eval/planner_prior_gap | 236000 | 94.87 | 101.14 | -28.07 | 221.36 |
| seed1 | loss/total | 236100 | 1.2140 | 1.2649 | 0.3774 | 2.2874 |

`lora/duo_film/seed1` shows a very large planner-vs-prior gap (~101) and visibly worse prior return — suggests a divergent prior in that seed.

### `lora/duo_film_fc2`  (3 seeds, ~211–240 k env-steps, status pending)

| seed | tag | last_step | final | mean_last_10% | min | max |
|---|---|---|---|---|---|---|
| seed0 | eval/planner/return_total | 239000 | 273.25 | 248.59 | -1.65 | 287.39 |
| seed0 | eval/prior/return_total | 239000 | 254.48 | 256.82 | -4.00 | 277.27 |
| seed0 | eval/planner_prior_gap | 239000 | 18.77 | -8.24 | -43.00 | 138.14 |
| seed0 | loss/total | 239600 | 1.0164 | 1.0299 | 0.3397 | 2.1558 |
| seed1 | eval/planner/return_total | 211000 | 253.29 | 233.88 | -2.41 | 276.45 |
| seed1 | eval/prior/return_total | 211000 | 240.01 | 213.90 | -4.00 | 256.49 |
| seed1 | eval/planner_prior_gap | 211000 | 13.28 | 19.98 | -46.38 | 208.64 |
| seed1 | loss/total | 211300 | 1.1302 | 1.1291 | 0.3426 | 2.1950 |
| seed2 | eval/planner/return_total | 211000 | 289.29 | 275.95 | -0.21 | 289.46 |
| seed2 | eval/prior/return_total | 211000 | 262.67 | 263.74 | -4.00 | 283.72 |
| seed2 | eval/planner_prior_gap | 211000 | 26.62 | 12.21 | -39.44 | 103.26 |
| seed2 | loss/total | 211400 | 1.0345 | 1.0186 | 0.3756 | 2.1965 |

### `lora/medium_base`  (3 seeds, only ~66 k env-steps, status pending)

| seed | tag | last_step | final | mean_last_10% | min | max |
|---|---|---|---|---|---|---|
| seed0 | eval/planner/return_total | 66000 | 56.89 | 42.24 | -6.74 | 210.16 |
| seed0 | eval/prior/return_total | 66000 | 155.59 | 109.51 | -3.68 | 155.59 |
| seed0 | eval/planner/fairness | 66000 | **-0.2719** | **-0.3043** | -0.6023 | 1.0000 |
| seed0 | eval/planner_prior_gap | 66000 | -98.70 | -67.28 | -98.70 | 179.92 |
| seed0 | loss/total | 66400 | 1.5484 | 1.5448 | 1.3163 | 2.2620 |
| seed1 | eval/planner/return_total | 66000 | 38.85 | 49.57 | -0.08 | 143.82 |
| seed1 | eval/prior/return_total | 66000 | 112.13 | 120.51 | -0.02 | 159.83 |
| seed1 | eval/planner/fairness | 66000 | **-0.4001** | **-0.3208** | -0.5126 | 0.7113 |
| seed1 | eval/planner_prior_gap | 66000 | -73.28 | -70.94 | -101.06 | 81.85 |
| seed1 | loss/total | 66400 | 1.5672 | 1.5857 | 1.3855 | 2.2732 |
| seed2 | eval/planner/return_total | 66000 | 107.36 | 160.25 | -2.24 | 216.03 |
| seed2 | eval/prior/return_total | 66000 | 115.12 | 130.91 | -1.02 | 220.65 |
| seed2 | eval/planner/fairness | 66000 | 0.1550 | 0.1823 | -0.6476 | 1.0000 |
| seed2 | eval/planner_prior_gap | 66000 | -7.76 | 29.34 | -143.44 | 115.85 |
| seed2 | loss/total | 66400 | 1.6644 | 1.6578 | 1.3492 | 2.3512 |

Across-seed (mean of mean_last_10%): `eval/planner/return_total` ≈ 84.02, `eval/prior/return_total` ≈ 120.31, `eval/planner_prior_gap` ≈ -36.29. Planner consistently underperforms prior by large margin in this cell.

## Issues / warnings

1. **Five cells have TB data but registry status still `pending`.** `main_comparison_easy` (3 seeds), `lora/duo_base` (1 seed), `lora/duo_film` (2 seeds), `lora/duo_film_fc2` (3 seeds), `lora/medium_base` (3 seeds) — 12 runs total — show no `completed_at_iso8601`, no `walltime_seconds`, no `checkpoint_path`, no `eval_report_path`, and no backfilled `return_mean` / `return_zero_shot_unseen` / `regret_mean`. Either the registry-finalization step never ran for those cells, or the runs are still in progress / were aborted without being marked.

2. **Likely incomplete training in `lora/medium_base`.** All three seeds halted at ~66 400 train steps vs ~211–240 k in the sister `lora/duo_film_fc2` cell. Combined with `eval/planner/fairness` going **negative** (seed0: -0.27; seed1: -0.40) and `eval/planner_prior_gap` deeply negative (~-67 to -71 mean_last_10% for seeds 0/1), this looks like a not-yet-converged or failed run. The 3 extra `_same` diagnostics (`diag/cos_pred_same`, `diag/cos_rew_same`, `diag/l2_rew_same`) only appear in this cell — possibly a newer code path with regressions, possibly just a different logging branch.

3. **`lora/duo_film` seed1 diverged.** `eval/planner_prior_gap` ≈ 101 (mean_last_10%), while seed0 ≈ 1. Prior return (~118) collapses well below planner return (~219). Worth a closer look before pooling seeds.

4. **Duplicate registry rows in 2 cells.** `zero_shot_easy/registry.jsonl` and `abl4_joint_easy_n2/registry.jsonl` each have 6 rows (3 runs × 2 records: pending then completed). The other 5 cells only have one row per run. The analysis pipeline should aggregate by `run_id` + latest record, not assume one-row-per-run.

5. **`return_mean` (registry) ≠ `eval/planner/return_total` final (TB) for `zero_shot_easy` / `abl4_joint_easy_n2`.** Registry values are 10–20 points higher than TB final/mean_last_10%. Likely from a separate end-of-training eval pass (over more episodes, different ctx grid) written into `eval_report.json`. Verify which is the authoritative number for downstream claims.

6. **`regret_mean = 0.0` for all six completed runs.** Either the scenario truly has zero regret at convergence, or the metric is not yet wired into the eval — worth confirming.

7. **`eval/planner/tragedy*` is identically 0.0 across all runs and all cells** — the tragedy-of-the-commons metric never triggers in the suite as evaluated. Either the environment never reaches a tragedy state under these checkpoints, or the metric is defined trivially in the easy variant. Not necessarily a bug, but flag for analysis design.

8. **`loss/belief` is effectively zero (≤1e-5) across all non-medium_base runs.** Only `lora/medium_base` shows a non-trivial belief loss (~1e-3). Either belief learning is mostly inactive in those configs or the belief loss is masked / weighted to ~0 — consistent with `loss/lambda_b = 1.0` constant.

9. **Schema mismatch only in `lora/medium_base`.** It logs 104 tags vs 101 elsewhere — the extra `_same` triplet. If your analysis joins by tag, mind the missing keys for other cells.

10. **`peak_gpu_memory_mb` is `null` everywhere.** Not collected for any run — if needed for the thesis, can no longer be backfilled retroactively.

Files inspected:
- 7 registry files: `D:/RL/hyper_mve/runs/suite/{main_comparison_easy,zero_shot_easy,abl4_joint_easy_n2,lora/duo_base,lora/duo_film,lora/duo_film_fc2,lora/medium_base}/registry.jsonl`
- 18 TB event-file directories: `D:/RL/hyper_mve/runs/suite/<cell>/<seed_run>/tb/events.out.tfevents.*`