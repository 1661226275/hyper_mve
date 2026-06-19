# Pkg-04/05 Test Checklist Report — 20260603_102839

## Environment
- **started_at**: 20260603_102839
- **host**: llabsrv01
- **platform**: Linux-5.15.0-168-generic-x86_64-with-glibc2.35
- **python**: 3.11.14
- **executable**: /home/zhengwenbo/.conda/envs/lightzero/bin/python
- **repo_root**: /home/data/zhengwenbo/hyper_mve
- **out_dir**: /home/data/zhengwenbo/hyper_mve/runs/test_checklist_pkg4_5/20260603_102839
- **args**: {'repo_root': None, 'out_dir': None, 'phases': '4,5', 'quick': False, 'skip_smoke': False, 'skip_e2e': False, 'keep_going': True, 'pytest_args': ''}

## Summary
- phases: 5/7 passed
- tests:  108/127 passed (18 failed, 1 skipped)
- hard gates: 24/32 passed
- total wall-clock: 144.7s

## Phases

| Phase | Status | Duration | Tests P/F/S | Notes |
|-------|--------|---------:|-------------|-------|
| §0 import sanity | PASS | 9.3s | 0/0/0 | rc=0 |
| phase4a_pkg04_models | PASS | 37.3s | 58/0/1 | rc=0 |
| §4b Pkg-04 forward smoke | PASS | 10.4s | 1/0/0 | rc=0 |
| phase5a_pkg05_training | FAIL | 53.1s | 40/15/0 | rc=1 |
| phase5b_pkg05_planning | FAIL | 18.9s | 6/3/0 | rc=1 |
| phase5c_pkg05_migration | PASS | 0.3s | 2/0/0 | rc=0 |
| §5d Pkg-05 e2e train pipeline | PASS | 15.4s | 1/0/0 | rc=0 |

## Hard Gates

| ID | Title | Status | Matched (P/F/T) | Required |
|----|-------|--------|-----------------|----------|
| C1 | hyper_trans uses c_ctx only (role-invariant) | PASS | 2/0/2 | ≥2 |
| C2 | subjective hyper input dim = 80 | PASS | 1/0/1 | ≥1 |
| C5d | ctx_aug dim == 80 (16+32+32) | PASS | 1/0/1 | ≥1 |
| C4 | belief grad gating (pre-5k detached / post-5k flows) | PASS | 2/0/2 | ≥2 |
| C6 | StateTransNet predicts delta-s (residual) | PASS | 1/0/1 | ≥1 |
| C7 | AdaLN (1 + gamma) residual modulation | PASS | 1/0/1 | ≥1 |
| C8 | output_scale inits (trans/rew/pred) | PASS | 1/0/1 | ≥1 |
| C9 | type-aware reward differentiation (assertion A) | PASS | 1/0/1 | ≥1 |
| C11 | Self-Info: no oracle types leak into set_context_subjective | PASS | 1/0/1 | ≥1 |
| API7 | model exposes exactly the 7 public APIs | PASS | 1/0/1 | ≥1 |
| R8 | model forward smoke < 15 ms (CPU-safe variant) | PASS | 1/0/1 | ≥1 |
| C5-T1 | trainer calls update_step once per train_step | FAIL | 0/1/1 | ≥1 |
| C5-T2 | set_context_objective once per unroll | FAIL | 0/1/1 | ≥1 |
| C5-T3 | set_context_subjective per agent | FAIL | 0/1/1 | ≥1 |
| C5-W1 | worker never calls update_step | FAIL | 0/1/1 | ≥1 |
| C5-W2 | worker uses BeliefNet.step online inference | FAIL | 0/1/1 | ≥1 |
| C5-W3 | TimeStepRecord z_hat shape / field validity | PASS | 2/0/2 | ≥2 |
| C5-B1 | buffer z_hat order preserved end-to-end | PASS | 1/0/1 | ≥1 |
| C5-B2 | stratified sampling min-per-type fraction | PASS | 1/0/1 | ≥1 |
| C5-S1 | curriculum 3-stage boundaries | PASS | 1/0/1 | ≥1 |
| C5-S2 | Stage 1 full oracle weight = 1.0 | PASS | 1/0/1 | ≥1 |
| C5-S3 | Stage 2 anneal monotone decreasing | PASS | 1/0/1 | ≥1 |
| C5-L1 | lambda_b curve matches cfg | FAIL | 0/1/1 | ≥1 |
| C5-L2 | belief gradient double-path (pre-5k isolated / post-5k both) | FAIL | 0/2/2 | ≥2 |
| C5-P1 | MVE CRN deterministic under same seed | PASS | 1/0/1 | ≥1 |
| C5-P2 | planner 4 set_context sites migrated | FAIL | 0/1/1 | ≥1 |
| C5-E1 | EMA tau=0.99 decay correctness | PASS | 1/0/1 | ≥1 |
| C5-E2 | warmup_cosine LR curve | PASS | 1/0/1 | ≥1 |
| C5-I1 | v4.7 -> v4 set_context migration grep (0 residual) | PASS | 2/0/2 | ≥2 |
| R5-2 | sample_batch < 50 ms | PASS | 1/0/1 | ≥1 |
| S04 | Pkg-04 forward smoke (100 steps no NaN + perf) | PASS | 1/0/1 | ≥1 |
| S05 | Pkg-05 e2e train pipeline (collect->store->sample->train_step) | PASS | 1/0/1 | ≥1 |

## Failures

- `tests.training.test_episode_buffer::test_sample_batch_shape` (phase5a_pkg05_training)
  - KeyError: 't'
- `tests.training.test_episode_buffer::test_v47_episodedata_v4_record_field_mapping` (phase5a_pkg05_training)
  - AssertionError: assert 't' in {'actions': tensor([[[0, 0, 0, 0],\n         [0, 0, 0, 0],\n         [0, 0, 0, 0],\n         [0, 0, 0, 0],\n         [0, 0, 0, 0],\n         [0, 0, 0, 0]],\n\n        [[0, 0, 0, 0],\n         [0, 0, 0, 0],\n         [0, 0, 0, 0],\n         [0, 0, 0, 0],\n         [0, 0, 0, 0],\n         [0, 0, 0, 0]],\n\n        [[0, 0, 0, 0],\n         [0, 0, 0, 0],\n         [0, 0, 0, 0],\n         [0, 0, 0, 0],\n         [0, 0, 0, 0],\n         [0, 0, 0, 0]],\n\n        [[0, 0, 0
- `tests.training.test_loss_composition::test_loss_composition_returns_full_dict` (phase5a_pkg05_training)
  - RuntimeError: Expected all tensors to be on the same device, but got mat2 is on cuda:0, different from other tensors on cpu (when checking argument in method wrapper_CUDA_mm)
- `tests.training.test_loss_composition::test_loss_composition_no_nan` (phase5a_pkg05_training)
  - RuntimeError: Expected all tensors to be on the same device, but got mat2 is on cuda:0, different from other tensors on cpu (when checking argument in method wrapper_CUDA_mm)
- `tests.training.test_loss_composition::test_lambda_b_curve_matches_cfg` (phase5a_pkg05_training)
  - RuntimeError: Expected all tensors to be on the same device, but got mat2 is on cuda:0, different from other tensors on cpu (when checking argument in method wrapper_CUDA_mm)
- `tests.training.test_loss_composition::test_lambda_b_zero_disables_belief_weight` (phase5a_pkg05_training)
  - RuntimeError: Expected all tensors to be on the same device, but got mat2 is on cuda:0, different from other tensors on cpu (when checking argument in method wrapper_CUDA_mm)
- `tests.training.test_loss_composition::test_belief_gradient_isolation_pre_5k` (phase5a_pkg05_training)
  - RuntimeError: Expected all tensors to be on the same device, but got mat2 is on cuda:0, different from other tensors on cpu (when checking argument in method wrapper_CUDA_mm)
- `tests.training.test_loss_composition::test_belief_gradient_both_sources_post_5k` (phase5a_pkg05_training)
  - RuntimeError: Expected all tensors to be on the same device, but got mat2 is on cuda:0, different from other tensors on cpu (when checking argument in method wrapper_CUDA_mm)
- `tests.training.test_trainer_loop::test_trainer_calls_update_step_per_step` (phase5a_pkg05_training)
  - failed on setup with "file /home/data/zhengwenbo/hyper_mve/tests/training/test_trainer_loop.py, line 53
  def test_trainer_calls_update_step_per_step(trainer, cfg_medium, mocker):
E       fixture 'mocker' not found
>       available fixtures: anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cfg_medium, doctest_namespace, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, model, monkeypatch, pyte
- `tests.training.test_trainer_loop::test_objective_called_once_per_unroll` (phase5a_pkg05_training)
  - failed on setup with "file /home/data/zhengwenbo/hyper_mve/tests/training/test_trainer_loop.py, line 60
  def test_objective_called_once_per_unroll(trainer, cfg_medium, mocker):
E       fixture 'mocker' not found
>       available fixtures: anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cfg_medium, doctest_namespace, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, model, monkeypatch, pytest
- `tests.training.test_trainer_loop::test_subjective_called_per_agent` (phase5a_pkg05_training)
  - failed on setup with "file /home/data/zhengwenbo/hyper_mve/tests/training/test_trainer_loop.py, line 66
  def test_subjective_called_per_agent(trainer, cfg_medium, mocker):
E       fixture 'mocker' not found
>       available fixtures: anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cfg_medium, doctest_namespace, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, model, monkeypatch, pytestconfi
- `tests.training.test_trainer_loop::test_target_model_ema_update` (phase5a_pkg05_training)
  - AssertionError: assert not True
 +  where True = <built-in method allclose of type object at 0x7ff38be08a60>(tensor([[-6.4409e-02, -5.1797e-02, -2.3584e-02,  ...,  2.2750e-02,\n          4.8818e-02, -1.9975e-02],\n        [ 1.9179e-02, -6.7196e-02, -1.0243e-01,  ..., -6.7275e-04,\n         -1.8419e-04, -2.2278e-05],\n        [-2.7445e-02, -2.3459e-02,  1.1652e-02,  ..., -1.7510e-02,\n         -6.7036e-02,  6.4205e-02],\n        ...,\n        [ 3.4354e-02,  2.4321e-02, -9.1227e-03,  ...,  6.4203e
- `tests.training.test_worker::test_worker_no_update_step` (phase5a_pkg05_training)
  - failed on setup with "file /home/data/zhengwenbo/hyper_mve/tests/training/test_worker.py, line 43
  def test_worker_no_update_step(worker, mocker):
E       fixture 'mocker' not found
>       available fixtures: anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cfg_fast, doctest_namespace, env, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, model, monkeypatch, pytestconfig, record_property, re
- `tests.training.test_worker::test_worker_uses_belief_net_step` (phase5a_pkg05_training)
  - failed on setup with "file /home/data/zhengwenbo/hyper_mve/tests/training/test_worker.py, line 51
  def test_worker_uses_belief_net_step(worker, mocker):
E       fixture 'mocker' not found
>       available fixtures: anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cfg_fast, doctest_namespace, env, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, model, monkeypatch, pytestconfig, record_proper
- `tests.training.test_worker::test_worker_no_oracle_types_leak` (phase5a_pkg05_training)
  - failed on setup with "file /home/data/zhengwenbo/hyper_mve/tests/training/test_worker.py, line 102
  def test_worker_no_oracle_types_leak(worker, mocker):
E       fixture 'mocker' not found
>       available fixtures: anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cfg_fast, doctest_namespace, env, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, model, monkeypatch, pytestconfig, record_prope
- `tests.planning.test_mve_planner::test_planner_4_set_context_migrated` (phase5b_pkg05_planning)
  - failed on setup with "file /home/data/zhengwenbo/hyper_mve/tests/planning/test_mve_planner.py, line 67
  def test_planner_4_set_context_migrated(planner, model, cfg_medium, mocker):
E       fixture 'mocker' not found
>       available fixtures: anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cfg_medium, doctest_namespace, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, model, monkeypatch, pl
- `tests.planning.test_mve_planner::test_planner_no_legacy_set_context` (phase5b_pkg05_planning)
  - failed on setup with "file /home/data/zhengwenbo/hyper_mve/tests/planning/test_mve_planner.py, line 75
  def test_planner_no_legacy_set_context(planner, model, cfg_medium, mocker):
E       fixture 'mocker' not found
>       available fixtures: anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cfg_medium, doctest_namespace, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, model, monkeypatch, pla
- `tests.planning.test_mve_planner::test_sample_mve_plan_under_50ms` (phase5b_pkg05_planning)
  - assert (3040.7846178859472 / 20) < 50.0
 +  where 3040.7846178859472 = sum([195.4997442662716, 194.9731968343258, 193.29542480409145, 140.6830232590437, 141.48932695388794, 137.62116618454456, ...])
 +  and   20 = len([195.4997442662716, 194.9731968343258, 193.29542480409145, 140.6830232590437, 141.48932695388794, 137.62116618454456, ...])
