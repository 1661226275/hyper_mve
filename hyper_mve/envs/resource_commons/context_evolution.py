"""Context (c_t) evolution — Ch3.4.3 + Pkg-02 spec 05.

Three modes with a common :class:`ContextEvolution` ABC:

- :class:`StaticContext`       — c_t ≡ c_0 (default for training and most
  evaluations).
- :class:`OscillateContext`    — c_t = 0.5 + 0.5·sin(2π t / period).
- :class:`RandomWalkContext`   — Gaussian random walk with optional shocks.

All modes draw randomness from the env-owned :class:`numpy.random.Generator`
so ``env.reset(seed=...)`` fully determines the c_t time series (Ch3 RNS-MMG
constraint C3 — no agent-action input).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ContextEvolution(ABC):
    """Abstract base for c_t evolution policies."""

    @abstractmethod
    def initial_c(self, rng: np.random.Generator) -> float:
        """Return the initial c_0 (may consume randomness)."""

    @abstractmethod
    def step(
        self,
        c_prev: float,
        step_idx: int,
        rng: np.random.Generator,
    ) -> float:
        """Return c_{t+1} given c_t and step index (may consume randomness)."""


class StaticContext(ContextEvolution):
    """c_t ≡ c_0 — sampled once at reset, then constant."""

    def initial_c(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(0.0, 1.0))

    def step(self, c_prev: float, step_idx: int, rng: np.random.Generator) -> float:
        return float(c_prev)


class OscillateContext(ContextEvolution):
    """``c_t = 0.5 + 0.5·sin(2π t / period)`` — seasonal rhythm.

    Initial value is the midpoint 0.5 (``sin(0) = 0``) regardless of seed,
    so the period is anchored at ``step_idx = 0``.
    """

    def __init__(self, period: int = 50):
        if period <= 0:
            raise ValueError(f"period must be > 0, got {period}")
        self.period = int(period)

    def initial_c(self, rng: np.random.Generator) -> float:
        return 0.5

    def step(self, c_prev: float, step_idx: int, rng: np.random.Generator) -> float:
        return float(0.5 + 0.5 * np.sin(2.0 * np.pi * step_idx / self.period))


class RandomWalkContext(ContextEvolution):
    """``c_{t+1} = clip(c_t + N(0, σ²) + shock, 0, 1)``.

    Args:
        sigma: Gaussian step std-dev.
        shock_prob: probability of an additional uniform shock each step.
        shock_range: uniform shock support is ``[-shock_range, +shock_range]``.
    """

    def __init__(
        self,
        sigma: float = 0.05,
        shock_prob: float = 0.2,
        shock_range: float = 0.3,
    ):
        if sigma < 0.0:
            raise ValueError(f"sigma must be ≥ 0, got {sigma}")
        if not (0.0 <= shock_prob <= 1.0):
            raise ValueError(f"shock_prob must be in [0, 1], got {shock_prob}")
        if shock_range < 0.0:
            raise ValueError(f"shock_range must be ≥ 0, got {shock_range}")
        self.sigma = float(sigma)
        self.shock_prob = float(shock_prob)
        self.shock_range = float(shock_range)

    def initial_c(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(0.0, 1.0))

    def step(self, c_prev: float, step_idx: int, rng: np.random.Generator) -> float:
        c_new = c_prev + float(rng.normal(0.0, self.sigma))
        if rng.random() < self.shock_prob:
            c_new += float(rng.uniform(-self.shock_range, self.shock_range))
        return float(np.clip(c_new, 0.0, 1.0))


def build_context_evolution(c_mode: str, cfg) -> ContextEvolution:
    """Factory: select an evolution mode and pull its parameters from ``cfg``.

    ``cfg`` only needs the four attributes ``c_oscillate_period``,
    ``c_random_walk_sigma``, ``c_shock_prob``, ``c_shock_range`` — typically
    an :class:`EnvConfig`, but any duck-typed object works (the tests use a
    tiny dataclass stand-in).
    """
    if c_mode == "static":
        return StaticContext()
    if c_mode == "oscillate":
        return OscillateContext(period=int(cfg.c_oscillate_period))
    if c_mode == "random_walk":
        return RandomWalkContext(
            sigma=float(cfg.c_random_walk_sigma),
            shock_prob=float(cfg.c_shock_prob),
            shock_range=float(cfg.c_shock_range),
        )
    raise ValueError(f"Unknown c_mode: {c_mode!r}")
