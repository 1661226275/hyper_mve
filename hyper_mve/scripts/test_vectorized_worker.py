"""[v4-opt 2026-06] Vectorized-collection smoke test (2agent diagnosis follow-up).

Run (from the outer hyper_mve/ working dir, GPU env):
    python hyper_mve/scripts/test_vectorized_worker.py

Validates:
    1. Worker.collect_episodes(B=3) planner-OFF: shapes / lengths / pi simplex /
       finite returns; planner diagnostics are NaN (planner off).
    2. Worker.collect_episodes(B=3) planner-ON: same, plus finite q_std / q_gap and
       uniform_frac in [0, 1].
    3. Legacy single-env API collect_episode still returns (records, c_t_seq).
    4. Deterministic eval mode: same reset seeds + same planner CRN seed twice
       => bitwise-identical per-agent returns.
    5. Buffer round trip: planner_on / collected_at_step survive store -> sample.
    6. One full MuZeroTrainer.train_step on a mixed batch: masked policy loss is
       finite, diag_target_age present; an ALL-planner-off batch yields
       L_policy_raw == 0 (no self-distillation signal).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataclasses import replace

import numpy as np
import torch

from hyper_mve.configs import V4Config
from hyper_mve.envs.resource_commons.env import ResourceCommonsEnv
from hyper_mve.models import HyperMuZeroModel, Projector
from hyper_mve.planning.mve_planner import MVEPlanner
from hyper_mve.training import EpisodeReplayBuffer, MuZeroTrainer, Worker


def _small_cfg() -> V4Config:
    cfg = V4Config.from_preset("duo_film_lora")
    cfg = replace(
        cfg,
        env=replace(cfg.env, T_max=30),
        train=replace(cfg.train, mve_samples=12, mve_depth=3,
                      buffer_size=64, min_buffer_size=4, batch_size=16),
    )
    return cfg


def _check_results(tag, results, cfg, planner_on):
    N, A, T = cfg.env.N, cfg.env.A, cfg.env.T_max
    assert len(results) > 0
    for res in results:
        assert len(res.records) == T, f"{tag}: T={len(res.records)} != {T}"
        assert res.c_t_seq.shape == (T,)
        r0 = res.records[0]
        assert r0.o.shape[0] == N and r0.pi_mve.shape == (N, A)
        rows = np.stack([rec.pi_mve for rec in res.records])           # (T, N, A)
        assert np.all(np.isfinite(rows)) and np.allclose(rows.sum(-1), 1.0, atol=1e-4), \
            f"{tag}: pi_mve rows not on the simplex"
        assert np.all(np.isfinite(res.returns)) and res.returns.shape == (N,)
        assert np.isfinite(res.pi_entropy_mean)
        if planner_on:
            assert np.isfinite(res.q_std_mean) and np.isfinite(res.q_gap_mean), \
                f"{tag}: planner diagnostics not finite"
            assert 0.0 <= res.uniform_frac <= 1.0
        else:
            assert np.isnan(res.q_std_mean) and np.isnan(res.q_gap_mean)
    print(f"  PASS {tag}")


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = _small_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HyperMuZeroModel(cfg).to(device)
    model.update_step(0)

    B = 3
    envs = [ResourceCommonsEnv(cfg.env, seed=100 + i) for i in range(B)]
    worker = Worker(cfg, model, envs=envs)

    # 1/2. batched collection, planner off + on
    _check_results("collect_episodes planner=OFF", worker.collect_episodes(
        epsilon=0.5, use_planner=False), cfg, planner_on=False)
    _check_results("collect_episodes planner=ON", worker.collect_episodes(
        epsilon=0.5, use_planner=True), cfg, planner_on=True)

    # 3. legacy single-env API
    records, c_t_seq = worker.collect_episode(epsilon=0.5, use_planner=True)
    assert len(records) == cfg.env.T_max and c_t_seq.shape == (cfg.env.T_max,)
    print("  PASS legacy collect_episode API")

    # 4. deterministic eval repeatability (same env seeds + same planner CRN seed
    #    + same tie-break rng seed = CRN across calls)
    seeds = [900 + i for i in range(B)]
    options = [{"c": 0.5}] * B
    rets = []
    for _ in range(2):
        planner = MVEPlanner(cfg)
        planner.crn_rng = np.random.default_rng(777)
        det_worker = Worker(cfg, model, envs=[ResourceCommonsEnv(
            replace(cfg.env, c_mode="static"), seed=s) for s in seeds], planner=planner)
        outs = det_worker.collect_episodes(
            epsilon=0.0, use_planner=True, deterministic=True,
            reset_seeds=seeds, reset_options=options,
            tiebreak_rng=np.random.default_rng(42))
        rets.append(np.stack([o.returns for o in outs]))
    assert np.array_equal(rets[0], rets[1]), \
        f"deterministic eval not repeatable:\n{rets[0]}\nvs\n{rets[1]}"
    print("  PASS deterministic eval repeatability")

    # 4b. argmax tie-break: with a uniform pi_mve, action 0 = NOOP in
    # ResourceCommons — without tie-break, every action would collapse to NOOP.
    # Sanity check: ten random uniform rows shouldn't all argmax to action 0.
    rng = np.random.default_rng(7)
    uniform_pi = np.full((10, A), 1.0 / A, dtype=np.float32)
    chosen = np.array([int(rng.choice(np.flatnonzero(p >= p.max() - 1e-9))) for p in uniform_pi])
    assert len(set(chosen)) > 1, f"tie-break degenerate: chose only {set(chosen)}"
    print(f"  PASS argmax tie-break covers >1 action ({sorted(set(chosen))})")

    # 5. buffer round trip of the new metadata
    buffer = EpisodeReplayBuffer(cfg)
    for res in worker.collect_episodes(epsilon=1.0, use_planner=False):
        buffer.store_episode(res.records, res.c_t_seq, planner_on=False, collected_at_step=0)
    for res in worker.collect_episodes(epsilon=0.5, use_planner=True):
        buffer.store_episode(res.records, res.c_t_seq, planner_on=True, collected_at_step=50)
    batch = buffer.sample_batch(cfg.train.batch_size, cfg.train.unroll_K)
    assert batch["planner_on"].dtype == torch.bool and batch["planner_on"].shape == (cfg.train.batch_size,)
    assert batch["collected_at_step"].dtype == torch.long
    print("  PASS buffer planner_on / collected_at_step round trip")

    # 6. one full train_step with the masked policy loss
    projector = Projector(cfg.model.latent_dim, cfg.model.proj_dim)
    trainer = MuZeroTrainer(cfg, model, projector=projector, device=device)
    losses = trainer.train_step(batch, global_step=100)
    assert np.isfinite(losses["total"]), "total loss not finite"
    assert "diag_target_age_steps" in losses and losses["diag_target_age_steps"] >= 0.0
    print(f"  PASS train_step on mixed batch (total={losses['total']:.4f}, "
          f"target_age={losses['diag_target_age_steps']:.1f})")

    # all-planner-off batch => policy loss exactly 0 (no self-distillation signal)
    off_buffer = EpisodeReplayBuffer(cfg)
    for res in worker.collect_episodes(epsilon=1.0, use_planner=False):
        off_buffer.store_episode(res.records, res.c_t_seq, planner_on=False, collected_at_step=0)
    off_batch = off_buffer.sample_batch(cfg.train.batch_size, cfg.train.unroll_K)
    off_losses = trainer.train_step(off_batch, global_step=101)
    assert off_losses["L_policy_raw"] == 0.0, \
        f"planner-off batch should mask policy loss, got {off_losses['L_policy_raw']}"
    assert np.isnan(off_losses["diag_pi_mve_entropy"]), \
        "H_pi_mve should be NaN when the batch holds no planner-on sample"
    print("  PASS all-planner-off batch masks the policy loss")

    print("\nALL PASS — vectorized worker + planner_on mask")


if __name__ == "__main__":
    main()
