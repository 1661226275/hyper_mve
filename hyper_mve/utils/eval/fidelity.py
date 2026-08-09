"""World-model fidelity — headline metric ① (``fidelity-v1``, phase 7).

A SEPARATE artifact from the rel-v1 EvalReport (whose 28-field dataclass
lock never moves for this): ``scripts/train.py`` writes it next to
``eval_report.json`` as ``fidelity.json``.

Probe protocol
--------------
``collect_probe_set`` rolls held-out REAL trajectories under uniform-random
joint actions with a fixed seed, via the injected oracle-free ``env_fn`` —
so given (env preset, regime grid, episodes, seed) the probe dataset is
byte-identical for every method. The regime ID ``g`` is known because the
probe SETS it at reset; no oracle info is read.

Hook contract (duck-typed)
--------------------------
``runner.predict_rewards(episode) -> (T, N) ndarray | None`` — the model's
one-step per-agent reward prediction for each real transition
``(o_{≤t}, a_t)`` in the episode, oracle-free unless the runner's protocol
discloses otherwise (m3w_adapted conditions on the given regime ID).
NaN rows mark unscoreable steps (e.g. MAMBA's own convention cannot score a
sequence's first transition); the metric masks them and reports coverage.
Runners without the attribute (model-free: HAPPO/MAPPO) or returning None
(MBOM — its simulator is SUPPLIED, not learned) are N/A → no artifact.

Episode dict schema: ``{"g": int, "obs": (T+1, N, D) f32,
"actions": (T, N) i64, "rewards": (T, N) f32}``.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

# v2 (2026-08-09): added regime_names; per-regime keys are family-relative ids.
SCHEMA_VERSION = "fidelity-v2"


def collect_probe_set(
    env_fn: Callable[[], Any],
    regime_grid,
    *,
    episodes: int = 2,
    seed: int = 1234,
) -> list[dict]:
    """Held-out random-policy trajectories; identical across methods."""
    env = env_fn()
    agents = list(env.possible_agents)
    n_actions = int(env.action_space(agents[0]).n)
    rng = np.random.default_rng(int(seed))

    probe: list[dict] = []
    for g in regime_grid:
        for ep in range(int(episodes)):
            obs_dict, _ = env.reset(
                seed=int(seed) + 611 * int(g) + ep, options={"g": int(g)})
            obs_seq = [np.stack([np.asarray(obs_dict[a], dtype=np.float32)
                                 for a in agents])]
            act_seq, rew_seq = [], []
            done = False
            while not done:
                acts = rng.integers(0, n_actions, size=len(agents))
                obs_dict, rew, term, trunc, _ = env.step(
                    {a: int(acts[i]) for i, a in enumerate(agents)})
                obs_seq.append(np.stack(
                    [np.asarray(obs_dict[a], dtype=np.float32) for a in agents]))
                act_seq.append(acts.astype(np.int64))
                rew_seq.append(np.array([float(rew[a]) for a in agents],
                                        dtype=np.float32))
                done = bool(any(term.values()) or any(trunc.values()))
            probe.append({
                "g": int(g),
                "obs": np.stack(obs_seq),          # (T+1, N, D)
                "actions": np.stack(act_seq),      # (T, N)
                "rewards": np.stack(rew_seq),      # (T, N)
            })
    env.close()
    return probe


def compute_reward_fidelity(runner, probe_set: list[dict]) -> Optional[dict]:
    """One-step reward-prediction MAE over the probe set, or None (N/A)."""
    hook = getattr(runner, "predict_rewards", None)
    if not callable(hook):
        return None

    err_per_regime: dict[int, list[float]] = {}
    mag_per_regime: dict[int, list[float]] = {}
    err_per_agent: dict[int, list[float]] = {}
    scored, total = 0, 0

    for ep in probe_set:
        pred = hook(ep)
        if pred is None:
            return None
        pred = np.asarray(pred, dtype=np.float64)
        true = np.asarray(ep["rewards"], dtype=np.float64)
        assert pred.shape == true.shape, (
            f"predict_rewards shape {pred.shape} != rewards {true.shape}")
        g = int(ep["g"])
        mask = np.isfinite(pred).all(axis=1)          # (T,) scoreable steps
        total += true.shape[0]
        scored += int(mask.sum())
        if not mask.any():
            continue
        abs_err = np.abs(pred[mask] - true[mask])     # (t, N)
        err_per_regime.setdefault(g, []).extend(abs_err.mean(axis=1).tolist())
        mag_per_regime.setdefault(g, []).extend(
            np.abs(true[mask]).mean(axis=1).tolist())
        for i in range(true.shape[1]):
            err_per_agent.setdefault(i, []).extend(abs_err[:, i].tolist())

    if scored == 0:
        return None

    mae_per_regime = {g: float(np.mean(v)) for g, v in
                      sorted(err_per_regime.items())}
    rmae_per_regime = {
        g: float(np.mean(err_per_regime[g])
                 / (np.mean(mag_per_regime[g]) + 1e-8))
        for g in sorted(err_per_regime)
    }
    all_errs = [e for v in err_per_regime.values() for e in v]
    return {
        "schema_version": SCHEMA_VERSION,
        "variant": getattr(runner, "name", type(runner).__name__),
        "metric": "one_step_reward_mae",
        "reward_mae": float(np.mean(all_errs)),
        "reward_mae_per_regime": mae_per_regime,
        "reward_rmae_per_regime": rmae_per_regime,
        "reward_mae_per_agent": [
            float(np.mean(err_per_agent[i])) for i in sorted(err_per_agent)],
        "transitions_scored": int(scored),
        "transitions_total": int(total),
        "episodes": int(len(probe_set)),
    }


def compute_fidelity_report(
    runner,
    env_fn: Callable[[], Any],
    regime_grid,
    *,
    episodes: int = 2,
    seed: int = 1234,
) -> Optional[dict]:
    """Probe + score in one call; annotates the probe parameters."""
    if not callable(getattr(runner, "predict_rewards", None)):
        return None
    probe = collect_probe_set(env_fn, regime_grid,
                              episodes=episodes, seed=seed)
    report = compute_reward_fidelity(runner, probe)
    if report is not None:
        from hyper_mve.utils.eval.eval_report import regime_names_for
        report["probe"] = {
            "seed": int(seed),
            "episodes_per_regime": int(episodes),
            "regime_grid": [int(g) for g in regime_grid],
            # `runner` here is duck-typed on predict_rewards alone (see the
            # guard above), so it may carry no cfg -- getattr, not attribute
            # access. regime_names_for(None) yields () rather than raising.
            "regime_names": list(regime_names_for(getattr(runner, "cfg", None))),
        }
    return report
