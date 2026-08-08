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

from typing import Callable, Optional, Sequence

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


def own_row_view(obs_i: np.ndarray, N: int, K: int) -> np.ndarray:
    """``(N-1,)`` the agent's own relationship row ``w_i·``.

    Self-Info: this is in every agent's observation by construction
    (``observations.py`` block 5), so a policy reading it is *not* privileged.
    The opponent's row ``w_j·`` is the part that is never observed.
    """
    return slice_relation_block(obs_i, "row", N, K)


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


def _nearest(mask: np.ndarray, dist: np.ndarray, K: int) -> int:
    """Index of the nearest cell with ``mask`` set; index breaks distance ties.

    Caller must ensure ``mask.any()`` — ``argmax`` on an all-False mask silently
    returns the nearest cell instead.
    """
    order = np.lexsort((np.arange(K), dist))
    return int(order[np.argmax(mask[order])])


class RelationalGreedyPolicy:
    """Regime-reactive greedy controller (N == 2) at one of two information levels.

    Two signs drive the decision, and **which of them is observable depends on
    the environment's** ``reward_coupling``:

    * ``care_self`` = ``Ŵ[i,j]`` — how much agent ``i``'s reward weights the
      neighbour's harvest. Positive ⇒ yield them their cell; negative ⇒ contest
      it, because denying them pays the same as harvesting.
    * ``care_opp`` = ``Ŵ[j,i]`` — the same for the neighbour, used to predict
      whether they will separate or contest.

    Under ``own_row`` (v5) ``Ŵ = W``, so ``care_self = w_ij`` is read straight
    off the observation and only ``care_opp`` is hidden. Under ``reciprocal``
    (v6) ``Ŵ = Wᵀ`` and the two swap: the agent can predict the neighbour
    perfectly but does not know the sign of *its own* objective. That is the
    whole point of the v6 environment.

    The two levels differ only in which matrix they reason from:

    * ``level="self_info"`` sees only ``w_i·`` and assumes the neighbour mirrors
      it, i.e. it reasons from the symmetric matrix built from its own row.
      That assumption holds in g0/g1/g4 and fails in g2/g3 — exactly the
      60%-accurate prior reachable without learning anything about the opponent.
    * ``level="oracle"`` is told the regime through :meth:`set_regime` and
      reasons from the true ``W``.

    So ``oracle − self_info`` isolates the return that inferring the hidden half
    of the relationship is worth. Both are heuristics, so the gap is a **lower
    bound** on the value of knowing the regime, not the optimum.
    """

    def __init__(
        self,
        N: int,
        K: int,
        *,
        level: str = "self_info",
        family=None,
        coupling: str = "own_row",
        reciprocity_lambda: float = 0.0,
        conserve_threshold: float = 0.0,
        min_stock_norm: float = _MIN_STOCK_NORM,
    ):
        if level not in ("self_info", "oracle"):
            raise ValueError(f"level must be 'self_info' or 'oracle', got {level!r}")
        if level == "oracle" and family is None:
            raise ValueError("level='oracle' needs the RegimeFamily to read W from")
        self.N = N
        self.K = K
        self.level = level
        self.family = family
        self.coupling = coupling
        self.reciprocity_lambda = reciprocity_lambda
        self.conserve_threshold = conserve_threshold
        self.min_stock_norm = min_stock_norm
        self._w = None      # (N, N) true matrix of the current regime, oracle only

    def set_regime(self, g: int) -> None:
        """Called by :func:`evaluate_reference_policy` after each reset."""
        if self.level == "oracle":
            self._w = self.family.W_stack()[int(g)]

    def _care(self, agent_idx: int, s_i: float) -> tuple[float, float]:
        """``(care_self, care_opp)`` under the env's coupling rule.

        ``self_info`` substitutes the mirror prior ``w_ji := w_ij`` — the best
        guess available from the observation alone — and then applies the same
        coupling rule, so it is wrong exactly where the mirror prior is wrong.
        """
        from hyper_mve.utils.schemas.relation import effective_coupling_matrix

        opp = 1 - agent_idx
        if self.level == "oracle":
            if self._w is None:
                raise RuntimeError("oracle policy used before set_regime()")
            W = self._w
        else:
            W = np.eye(2, dtype=np.float32)
            W[agent_idx, opp] = s_i
            W[opp, agent_idx] = s_i         # mirror prior for the hidden half
        W_eff = effective_coupling_matrix(W, self.coupling, self.reciprocity_lambda)
        return float(W_eff[agent_idx, opp]), float(W_eff[opp, agent_idx])

    def _predict_opponent_target(
        self, viable, d_self, d_nb, care_opp: float, agent_idx: int,
    ) -> int:
        opp_idx = 1 - agent_idx
        if care_opp >= 0.0:
            # Separating neighbour: claims only cells it reaches no later than
            # me. Same tie convention as make_scripted_greedy_policy (agent 0
            # wins ties), applied from the neighbour's side.
            wins = d_nb <= d_self if opp_idx == 0 else d_nb < d_self
            claimed = viable & wins
            if not claimed.any():
                claimed = viable
            return _nearest(claimed, d_nb, self.K)
        # Contesting neighbour: comes for the cell I would take, when it can
        # arrive no later; otherwise falls back to its own nearest.
        my_best = _nearest(viable, d_self, self.K)
        opp_first = (
            d_nb[my_best] <= d_self[my_best] if opp_idx == 0
            else d_nb[my_best] < d_self[my_best]
        )
        return my_best if opp_first else _nearest(viable, d_nb, self.K)

    def __call__(self, obs_i: np.ndarray, agent_idx: int) -> int:
        cells = resource_view(obs_i, self.N, self.K)
        rel, stock = cells[:, :2], cells[:, 2]
        d_self = np.abs(rel[:, 0]) + np.abs(rel[:, 1])

        if self.N != 2:
            underfoot = (rel[:, 0] == 0) & (rel[:, 1] == 0) & (stock > 0.0)
            if underfoot.any():
                return HARVEST
            viable = stock > self.min_stock_norm
            if not viable.any():
                viable = np.ones(self.K, dtype=bool)
            return _step_toward(*rel[_nearest(viable, d_self, self.K)])

        s_i = float(own_row_view(obs_i, self.N, self.K)[0])
        nb = neighbor_view(obs_i, self.N, self.K)[0, :2]
        d_nb = np.abs(rel[:, 0] - nb[0]) + np.abs(rel[:, 1] - nb[1])
        care_self, care_opp = self._care(agent_idx, s_i)

        # Restraint. Under a compounding commons the neighbour's payoff also
        # depends on what I leave in the ground, so the regime's value lives
        # here as much as in which cell I target: conserve when their harvest
        # pays me, strip when it costs me. conserve_threshold=0 disables this
        # and reproduces the v5 controller exactly — correct there, because
        # under constant regrowth restraint is strictly harmful.
        floor = self.min_stock_norm
        if self.conserve_threshold > 0.0 and care_self > 0.0:
            floor = self.conserve_threshold

        underfoot = (rel[:, 0] == 0) & (rel[:, 1] == 0) & (stock > 0.0)
        if underfoot.any():
            if stock[underfoot].max() > floor:
                return HARVEST
            if self.conserve_threshold > 0.0:
                return NOOP         # let it regrow rather than strip it

        viable = stock > floor
        if not viable.any():
            if self.conserve_threshold > 0.0 and care_self > 0.0:
                return NOOP
            viable = np.ones(self.K, dtype=bool)

        opp_target = self._predict_opponent_target(
            viable, d_self, d_nb, care_opp, agent_idx)

        if care_self < 0.0:
            # Their harvest costs me, so denying a cell pays the same as taking
            # one. Contest their target when I arrive no later than they do.
            i_first = (
                d_self[opp_target] <= d_nb[opp_target] if agent_idx == 0
                else d_self[opp_target] < d_nb[opp_target]
            )
            target = opp_target if (viable[opp_target] and i_first) else _nearest(
                viable, d_self, self.K)
        else:
            # care_self > 0: their harvest pays me. == 0: sharing a cell still
            # costs me half its yield. Either way, leave them their target.
            avail = viable.copy()
            avail[opp_target] = False
            target = _nearest(avail if avail.any() else viable, d_self, self.K)

        return _step_toward(float(rel[target, 0]), float(rel[target, 1]))


def make_relational_greedy_policy(
    N: int,
    K: int,
    *,
    level: str = "self_info",
    family=None,
    coupling: str = "own_row",
    reciprocity_lambda: float = 0.0,
    conserve_threshold: float = 0.0,
    min_stock_norm: float = _MIN_STOCK_NORM,
) -> PolicyFn:
    """Factory for :class:`RelationalGreedyPolicy` (see its docstring).

    ``coupling`` must match the environment's ``EnvConfig.reward_coupling`` —
    the controller decides from the sign of its reward's weight on the
    neighbour's harvest, and that weight is a different entry of ``W`` under
    each rule. Passing the wrong one measures a controller optimising the
    wrong objective.

    ``conserve_threshold`` must be 0 under ``regrowth_law="constant"``, where
    restraint is strictly harmful, and positive under ``"logistic"``, where
    the regime's value largely lives in the conservation decision. A controller
    without it is blind to the v6 dilemma and will report a null.
    """
    return RelationalGreedyPolicy(
        N, K, level=level, family=family, coupling=coupling,
        reciprocity_lambda=reciprocity_lambda,
        conserve_threshold=conserve_threshold, min_stock_norm=min_stock_norm,
    )


# ----------------------------------------------------------------- evaluation


def evaluate_reference_policy(
    env_fn: Callable[[], object],
    policy: PolicyFn,
    regime_grid,
    episodes: int,
    *,
    seed_base: int = 10_000,
    policies: Optional[Sequence[PolicyFn]] = None,
) -> dict:
    """Per-regime returns for a scripted policy.

    Mirrors ``MAZeroMixedRunner.evaluate``'s loop exactly — same episode
    seeding (``seed_base + 97*g + ep``), same summed-over-agents return — so the
    numbers are directly comparable to an ``eval_report.json`` without any
    rescaling.

    Args:
        policies: optional per-agent controllers, overriding ``policy``. Needed
            for **unilateral** comparisons: the value of information is what one
            agent gains by deviating while the others hold still, so handing
            every agent the oracle at once measures something else entirely
            (in a social dilemma it can lower the team return while raising
            each agent's own objective).

    Returns ``return_mean`` / ``return_per_regime`` (summed over agents, the
    headline convention) plus ``return_per_agent_per_regime`` — use the latter
    for anything about an individual agent's objective.
    """
    env = env_fn()
    agents = list(env.possible_agents)
    N = len(agents)
    actors: Sequence[PolicyFn] = policies if policies is not None else [policy] * N
    if len(actors) != N:
        raise ValueError(f"policies has {len(actors)} entries, env has {N} agents")

    return_per_regime: dict[int, float] = {}
    per_agent_per_regime: dict[int, list[float]] = {}
    all_returns: list[float] = []
    action_counts: Optional[np.ndarray] = None

    for g in regime_grid:
        g_returns: list[float] = []
        g_agent: list[np.ndarray] = []
        for ep in range(int(episodes)):
            obs_dict, _ = env.reset(
                seed=seed_base + 97 * int(g) + ep, options={"g": int(g)}
            )
            # Oracle-level controllers declare set_regime; plain PolicyFns do
            # not. This is the only channel by which the regime id reaches a
            # policy, so a policy without it is regime-blind by construction.
            for p in actors:
                if hasattr(p, "set_regime"):
                    p.set_regime(int(g))
            ep_ret, done = 0.0, False
            ep_agent = np.zeros(N)
            while not done:
                acts = {
                    a: int(actors[i](np.asarray(obs_dict[a]), i))
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
                for i, a in enumerate(agents):
                    ep_agent[i] += float(rew[a])
                done = bool(any(term.values()) or any(trunc.values()))
            g_returns.append(ep_ret)
            g_agent.append(ep_agent)
        return_per_regime[int(g)] = float(np.mean(g_returns)) if g_returns else 0.0
        per_agent_per_regime[int(g)] = (
            np.mean(g_agent, axis=0).tolist() if g_agent else [0.0] * N
        )
        all_returns.extend(g_returns)
    env.close()

    counts = action_counts if action_counts is not None else np.zeros(6, dtype=np.int64)
    return {
        "return_mean": float(np.mean(all_returns)) if all_returns else 0.0,
        "return_per_regime": return_per_regime,
        "return_per_agent_per_regime": per_agent_per_regime,
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
