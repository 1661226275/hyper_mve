"""Scripted reference policies for RelationCommons (v5).

These exist because the formal grid of 2026-07-18 spent ~7 GPU-hours per row
re-measuring a policy that had collapsed to constant HARVEST, and nothing in
the pipeline could say so. A learned return is only interpretable against a
scale, and the repo had none: ``docs/中期报告.md`` cites a random baseline of
~5.1 but no code produced it.

The scale on ``rel_duo`` (N=2, K=8, L=8, T_max=100), 16 episodes/regime under
the canonical eval seeding:

===========================  ======  ======  ======
policy                         mean      g0      g4
===========================  ======  ======  ======
``noop``                       0.00    0.00    0.00
``random``                     1.39    3.16    2.79
``harvest_only``              15.11   23.24   23.24
``scripted_greedy``           99.79  161.09  171.57
``scripted_greedy`` distinct  109.41  182.39  182.22
===========================  ======  ======  ======

``harvest_only`` is the degenerate attractor: HARVEST is the only
reward-generating action *and* it is free (only actions 1-4 pay the ``ε`` move
cost), so camping on a spawn cell pays without risk. It is worth ~14% of the
scripted policy and is not even a Nash equilibrium — unilateral deviation to
walk-and-harvest gains ~+48.5.

All policies here read **only the agent's own observation** — no oracle state.
The distinct-target rule coordinates two agents onto different cells without
communication, purely from the neighbor's relative position, which the
observation already carries. That keeps these usable as honest reference rows
rather than privileged upper bounds.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from hyper_mve.utils.schemas import RelationObservationLayout, slice_relation_block

from .env import HARVEST, NOOP

# Action encoding is fixed by env.py: 0=NOOP, 1=UP(+y), 2=DOWN(-y),
# 3=LEFT(-x), 4=RIGHT(+x), 5=HARVEST.
_RIGHT, _LEFT, _UP, _DOWN = 4, 3, 1, 2

# A cell below this normalized stock is not worth walking to; it regrows at
# ``alpha`` per step so it becomes a target again on its own.
_MIN_STOCK_NORM: float = 0.05

PolicyFn = Callable[[np.ndarray, int], int]
"""``(obs_i, agent_idx) -> action``. ``obs_i`` is one agent's flat observation."""


# --------------------------------------------------------------- obs decoding


def resource_view(obs_i: np.ndarray, N: int, K: int) -> np.ndarray:
    """``(K, 3)`` array of ``(rel_dx, rel_dy, q_norm)`` for every resource cell.

    ``rel_*`` are in raw grid units relative to this agent (``observations.py``
    stores the un-normalized displacement), so a cell is underfoot exactly when
    both are zero.
    """
    return slice_relation_block(obs_i, "resource", N, K).reshape(K, 3)


def neighbor_view(obs_i: np.ndarray, N: int, K: int) -> np.ndarray:
    """``(N-1, 9)`` neighbor block; columns 0,1 are the relative offsets."""
    return slice_relation_block(obs_i, "neighbor", N, K).reshape(
        N - 1, RelationObservationLayout.NEIGHBOR_PER_ITEM
    )


def _step_toward(rel_dx: float, rel_dy: float) -> int:
    """One greedy grid step toward a relative target; x axis first (stable)."""
    if rel_dx > 0:
        return _RIGHT
    if rel_dx < 0:
        return _LEFT
    if rel_dy > 0:
        return _UP
    if rel_dy < 0:
        return _DOWN
    return HARVEST


# ------------------------------------------------------------------ policies


def noop_policy(obs_i: np.ndarray, agent_idx: int) -> int:
    """Always NOOP. Floor reference: returns exactly 0 in every regime."""
    return NOOP


def harvest_only_policy(obs_i: np.ndarray, agent_idx: int) -> int:
    """Always HARVEST, never move — the collapse mode the grid trained into.

    Reproduces ``15.10890765041113`` on ``rel_duo`` bit-for-bit, which is what
    makes it usable as a regression assertion.
    """
    return HARVEST


def make_random_policy(seed: int = 0, A: int = 6) -> PolicyFn:
    """Uniform random over the action set, with an explicit RNG for repeatability."""
    rng = np.random.default_rng(seed)

    def _policy(obs_i: np.ndarray, agent_idx: int) -> int:
        return int(rng.integers(0, A))

    return _policy


def make_scripted_greedy_policy(
    N: int,
    K: int,
    *,
    distinct_targets: bool = True,
    min_stock_norm: float = _MIN_STOCK_NORM,
) -> PolicyFn:
    """Walk to a stocked resource cell, then HARVEST on it.

    Args:
        N, K: agent and resource counts, needed to slice the observation.
        distinct_targets: when True (and N == 2), an agent only claims cells it
            would reach no later than its neighbour (ties going to agent 0), and
            takes the nearest such cell. Two agents on one cell split the
            fair-share yield and leave a second cell unharvested, so separating
            them is worth ~+20 mean. Claiming by *advantage* rather than by
            "nearest cell I win" is worse than not coordinating at all — it
            sends an agent across the grid to a cell it merely wins by more.
        min_stock_norm: ignore cells below this normalized stock.
    """

    def _policy(obs_i: np.ndarray, agent_idx: int) -> int:
        cells = resource_view(obs_i, N, K)
        rel = cells[:, :2]
        stock = cells[:, 2]

        underfoot = (rel[:, 0] == 0) & (rel[:, 1] == 0) & (stock > 0.0)
        if underfoot.any():
            return HARVEST

        viable = stock > min_stock_norm
        if not viable.any():
            viable = np.ones(K, dtype=bool)

        d_self = np.abs(rel[:, 0]) + np.abs(rel[:, 1])
        claimed = viable

        if distinct_targets and N == 2:
            nb = neighbor_view(obs_i, N, K)[0, :2]
            # Cell position relative to the neighbour, from two relative
            # offsets that are both in this agent's own observation.
            d_nb = np.abs(rel[:, 0] - nb[0]) + np.abs(rel[:, 1] - nb[1])
            wins = d_self < d_nb if agent_idx != 0 else d_self <= d_nb
            if (viable & wins).any():
                claimed = viable & wins

        # Nearest claimed cell; index breaks exact distance ties so both
        # agents stay deterministic.
        order = np.lexsort((np.arange(K), d_self))
        target = int(order[np.argmax(claimed[order])])
        return _step_toward(float(rel[target, 0]), float(rel[target, 1]))

    return _policy


# ----------------------------------------------------------------- evaluation


def evaluate_reference_policy(
    env_fn: Callable[[], object],
    policy: PolicyFn,
    regime_grid,
    episodes: int,
    *,
    seed_base: int = 10_000,
) -> dict:
    """Per-regime returns for a scripted policy.

    Mirrors ``MAZeroMixedRunner.evaluate``'s loop exactly — same episode
    seeding (``seed_base + 97*g + ep``), same summed-over-agents return — so the
    numbers are directly comparable to an ``eval_report.json`` without any
    rescaling.

    Returns a dict with ``return_mean``, ``return_per_regime`` and
    ``action_histogram`` (counts over the full action set, all agents pooled).
    """
    env = env_fn()
    agents = list(env.possible_agents)
    N = len(agents)

    return_per_regime: dict[int, float] = {}
    all_returns: list[float] = []
    action_counts: Optional[np.ndarray] = None

    for g in regime_grid:
        g_returns: list[float] = []
        for ep in range(int(episodes)):
            obs_dict, _ = env.reset(
                seed=seed_base + 97 * int(g) + ep, options={"g": int(g)}
            )
            ep_ret, done = 0.0, False
            while not done:
                acts = {
                    a: int(policy(np.asarray(obs_dict[a]), i))
                    for i, a in enumerate(agents)
                }
                if action_counts is None:
                    action_counts = np.zeros(
                        int(env.action_space(agents[0]).n), dtype=np.int64
                    )
                for a in acts.values():
                    action_counts[a] += 1
                obs_dict, rew, term, trunc, _ = env.step(acts)
                ep_ret += float(sum(rew.values()))
                done = bool(any(term.values()) or any(trunc.values()))
            g_returns.append(ep_ret)
        return_per_regime[int(g)] = float(np.mean(g_returns)) if g_returns else 0.0
        all_returns.extend(g_returns)
    env.close()

    counts = action_counts if action_counts is not None else np.zeros(6, dtype=np.int64)
    return {
        "return_mean": float(np.mean(all_returns)) if all_returns else 0.0,
        "return_per_regime": return_per_regime,
        "episodes_per_regime": int(episodes),
        "action_histogram": counts.tolist(),
        "action_fractions": (counts / max(counts.sum(), 1)).round(6).tolist(),
    }


def reference_policy_suite(N: int, K: int, *, seed: int = 0) -> dict[str, PolicyFn]:
    """The standard reference row: name -> policy, in ascending strength."""
    return {
        "noop": noop_policy,
        "random": make_random_policy(seed=seed),
        "harvest_only": harvest_only_policy,
        "scripted_greedy": make_scripted_greedy_policy(N, K, distinct_targets=False),
        "scripted_greedy_distinct": make_scripted_greedy_policy(
            N, K, distinct_targets=True
        ),
    }
