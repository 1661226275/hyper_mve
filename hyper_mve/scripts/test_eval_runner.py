"""[v4-opt 2026-06] In-training evaluation smoke test (2agent diagnosis follow-up).

Run (from the outer hyper_mve/ working dir, GPU env):
    python hyper_mve/scripts/test_eval_runner.py

Validates:
    1. The static-c eval env honours reset(options={"c": ...}) and keeps c fixed.
    2. run_eval returns the full dual-mode tag set, all finite.
    3. run_eval is deterministic: two calls on the same model weights are identical
       (fixed env seeds + constant planner CRN seed).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataclasses import replace

import numpy as np
import torch

from hyper_mve.configs import V4Config
from hyper_mve.envs.resource_commons.env import ResourceCommonsEnv
from hyper_mve.models import HyperMuZeroModel
from hyper_mve.training import run_eval


def _small_cfg() -> V4Config:
    cfg = V4Config.from_preset("duo_film_lora")
    return replace(
        cfg,
        env=replace(cfg.env, T_max=30),
        train=replace(cfg.train, mve_samples=12, mve_depth=3),
        eval=replace(cfg.eval, eval_c_grid=(0.2, 0.8),
                     eval_episodes_prior=2, eval_episodes_planner=1),
    )


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = _small_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HyperMuZeroModel(cfg).to(device)
    model.update_step(0)

    # 1. static-c env honours options["c"] and keeps it fixed across steps
    env = ResourceCommonsEnv(replace(cfg.env, c_mode="static"), seed=1)
    _, info = env.reset(options={"c": 0.7})
    assert abs(float(info["c_true"]) - 0.7) < 1e-9, f"c_0={info['c_true']} != 0.7"
    for _ in range(3):
        _, _, _, _, info = env.step(np.zeros(cfg.env.N, dtype=np.int64))
    assert abs(float(info["c_true"]) - 0.7) < 1e-9, "static c drifted"
    print("  PASS static-c env honours options['c']")

    # 2. full run_eval tag set, all finite
    metrics = run_eval(model, cfg, global_step=0)
    expected = ["prior/return_total", "prior/return_alpha", "prior/return_beta",
                "prior/ep_len",
                "planner/return_total", "planner/return_alpha", "planner/return_beta",
                "planner/ep_len", "planner/pi_mve_entropy", "planner/q_std",
                "planner/uniform_frac",
                "planner_prior_gap"]
    for c in cfg.eval.eval_c_grid:
        expected += [f"prior/return_total_c{c:g}", f"planner/return_total_c{c:g}"]
    missing = [k for k in expected if k not in metrics]
    assert not missing, f"missing eval tags: {missing}"
    bad = [k for k, v in metrics.items() if not np.isfinite(v)]
    assert not bad, f"non-finite eval metrics: {bad}"
    print(f"  PASS run_eval tag set ({len(metrics)} tags, "
          f"planner_total={metrics['planner/return_total']:.3f}, "
          f"gap={metrics['planner_prior_gap']:.3f})")

    # 3. determinism across calls (same weights => identical numbers)
    metrics2 = run_eval(model, cfg, global_step=0)
    diffs = {k: (metrics[k], metrics2[k]) for k in metrics if metrics[k] != metrics2[k]}
    assert not diffs, f"run_eval not deterministic: {diffs}"
    print("  PASS run_eval determinism (CRN across evaluations)")

    print("\nALL PASS — in-training evaluation runner")


if __name__ == "__main__":
    main()
