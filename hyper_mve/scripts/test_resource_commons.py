"""1000-episode random-policy smoke test for ResourceCommonsEnv.

Acceptance per Pkg-02 spec §5.2 / README:
- 1000 episodes complete with no NaN / Inf.
- Average step time ≤ 1 ms on Medium preset (single thread, CPU).

Run from the repository root::

    python hyper_mve/scripts/test_resource_commons.py [easy|medium|hard] [n_episodes]
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from hyper_mve.configs import V4Config
from hyper_mve.envs.resource_commons import ResourceCommonsEnv


def run_smoke(preset: str, n_episodes: int, base_seed: int = 0) -> int:
    cfg = V4Config.from_preset(preset)
    env = ResourceCommonsEnv(cfg.env, seed=base_seed)
    rng = np.random.default_rng(base_seed)

    total_steps = 0
    total_reward = 0.0
    nan_episodes = 0
    start = time.perf_counter()

    for ep in range(n_episodes):
        obs, info = env.reset(seed=base_seed + ep)
        ep_reward = 0.0
        for t in range(cfg.env.T_max):
            action = rng.integers(0, 6, size=cfg.env.N, dtype=np.int64)
            obs, reward, done, _, info = env.step(action)
            if np.isnan(obs).any() or np.isnan(reward).any():
                nan_episodes += 1
                break
            if np.isinf(obs).any() or np.isinf(reward).any():
                nan_episodes += 1
                break
            ep_reward += float(reward.sum())
            total_steps += 1
            if done:
                break
        total_reward += ep_reward
        if (ep + 1) % max(n_episodes // 10, 1) == 0:
            print(f"  ...{ep + 1}/{n_episodes} episodes done")

    elapsed = time.perf_counter() - start
    avg_step_ms = elapsed / max(total_steps, 1) * 1000.0
    print()
    print(f"preset           : {preset}")
    print(f"episodes         : {n_episodes}")
    print(f"total steps      : {total_steps}")
    print(f"nan/inf episodes : {nan_episodes}")
    print(f"avg step (ms)    : {avg_step_ms:.3f}")
    print(f"steps / sec      : {total_steps / max(elapsed, 1e-9):.0f}")
    print(f"mean episode ret : {total_reward / max(n_episodes, 1):.3f}")

    ok = nan_episodes == 0
    print()
    if ok:
        print(f"ALL {n_episodes} EPISODES PASS (no NaN/Inf)")
    else:
        print(f"FAILED: {nan_episodes} episodes contained NaN or Inf")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", nargs="?", default="medium",
                        choices=("easy", "medium", "hard"))
    parser.add_argument("n_episodes", nargs="?", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    return run_smoke(args.preset, args.n_episodes, args.seed)


if __name__ == "__main__":
    sys.exit(main())
