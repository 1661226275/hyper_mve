"""Proposition 3.1 empirical validation: Pareto-Nash gap vs c.

Hypothesis (Ch3 Prop 3.1): as ``c → 1`` (abundance), the Pareto-Nash gap
under the type β preference shrinks — type β agents' best-response set
becomes more cooperative-friendly so social welfare approaches the
Pareto frontier.

This script does **not** train any policy; it samples a fixed
policy-class proxy (uniform-random + greedy-harvest) and reports the
single-shot welfare gap as a function of c. Suitable for inclusion as a
paper figure once policies are trained (Pkg-05) — at that point swap in
the trained-policy welfare numbers.

Output: ``proposition_3_1.png`` in the current working directory.

Usage::

    python hyper_mve/scripts/proposition_3_1_validation.py [--episodes 30]
"""
from __future__ import annotations

import argparse
import sys
from typing import Sequence

import numpy as np

from hyper_mve.configs import V4Config
from hyper_mve.envs.resource_commons import ResourceCommonsEnv


# Simple proxy policies ------------------------------------------------------

def _random_policy(env: ResourceCommonsEnv, rng: np.random.Generator) -> np.ndarray:
    """Uniform random over the 6 discrete actions."""
    return rng.integers(0, 6, size=env.N, dtype=np.int64)


def _greedy_harvest_policy(env: ResourceCommonsEnv, rng: np.random.Generator) -> np.ndarray:
    """Always HARVEST — proxy for a 'self-interested' policy that stays put."""
    return np.full(env.N, 5, dtype=np.int64)


def _social_welfare(rewards_per_step: np.ndarray) -> float:
    """Total per-step reward summed across agents and time, averaged across episodes."""
    return float(rewards_per_step.sum())


def evaluate_at_c(
    preset: str, c: float, episodes: int, seed_base: int,
) -> tuple[float, float]:
    """Return (random_welfare, greedy_welfare) at fixed c."""
    cfg = V4Config.from_preset(preset)
    env = ResourceCommonsEnv(cfg.env, seed=seed_base)
    rng = np.random.default_rng(seed_base)

    rand_returns = []
    greedy_returns = []
    for ep in range(episodes):
        # Random
        obs, info = env.reset(seed=seed_base + ep, options={"c": c})
        total = 0.0
        for _ in range(cfg.env.T_max):
            obs, reward, done, _, _ = env.step(_random_policy(env, rng))
            total += float(reward.sum())
            if done:
                break
        rand_returns.append(total)

        # Greedy
        obs, info = env.reset(seed=seed_base + ep, options={"c": c})
        total = 0.0
        for _ in range(cfg.env.T_max):
            obs, reward, done, _, _ = env.step(_greedy_harvest_policy(env, rng))
            total += float(reward.sum())
            if done:
                break
        greedy_returns.append(total)

    return float(np.mean(rand_returns)), float(np.mean(greedy_returns))


def run(preset: str, episodes: int, seed: int, c_values: Sequence[float]) -> None:
    print(f"Proposition 3.1 — preset={preset}, episodes={episodes}")
    print(f"c values: {list(c_values)}")
    print()

    rand_curve, greedy_curve, gap_curve = [], [], []
    for c in c_values:
        r, g = evaluate_at_c(preset, c, episodes, seed)
        gap = g - r       # proxy for "self-interested vs uniform" gap
        rand_curve.append(r)
        greedy_curve.append(g)
        gap_curve.append(gap)
        print(f"  c={c:.2f}  random={r:8.2f}  greedy={g:8.2f}  gap={gap:+8.2f}")

    # Plot (best effort — skip silently if matplotlib not available).
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        ax.plot(c_values, rand_curve, "-o", label="random policy")
        ax.plot(c_values, greedy_curve, "-s", label="greedy HARVEST")
        ax.plot(c_values, gap_curve, "-^", label="gap (greedy - random)")
        ax.set_xlabel("c_t")
        ax.set_ylabel("episode social welfare")
        ax.set_title(f"Proposition 3.1 ({preset}, n={episodes})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out_path = f"proposition_3_1_{preset}.png"
        fig.savefig(out_path)
        print(f"\nFigure written to: {out_path}")
    except Exception as exc:           # pragma: no cover — env-dependent
        print(f"\n[warn] matplotlib unavailable, skipping plot: {exc}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="medium",
                        choices=("easy", "medium", "hard"))
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    c_values = (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0)
    run(args.preset, args.episodes, args.seed, c_values)
    return 0


if __name__ == "__main__":
    sys.exit(main())
