"""PeriodicEvalProbe — in-training deterministic eval for EXTERNAL runners.

The internal variants (`hyper` + BaselineModel) already get per-regime eval
curves from ``training/evaluation.py`` (TB tags every ``cfg.eval.evaluate_freq``
train steps). External runners own their training loop and historically logged
nothing — so sample-efficiency curves (return vs env steps) could not be drawn
for MAPPO/QMIX/MAMBA. This probe closes that gap.

Contract:
  * built by the runner inside ``train()`` when the sweep harness supplies a
    ``tensorboard_dir`` (see ``_sweep_worker._train_runner``);
  * ``maybe_run(env_steps)`` is called after every collected episode; when the
    cumulative env-step counter crosses the cadence threshold it plays
    ``episodes_per_regime`` deterministic episodes pinned to every regime in
    the preset family (``reset(options={"g": g})``) on its OWN env instance —
    probe episodes never touch the training env nor count into the budget;
  * TB scalars, all keyed by **cumulative env steps** (the cross-method
    x-axis): ``eval/return_mean``, ``eval/return_g{g}``, ``eval/return_seen``,
    ``eval/return_unseen``.

The first ``maybe_run`` call always fires (near-init anchor point for log-x
curves); afterwards the cadence aligns to ``every_env_steps`` multiples, or to
``every_train_steps`` multiples of the caller-supplied ``train_steps`` count
when that is set (2026-07-20: the env-steps-per-train-step ratio is not fixed
for on-policy callers like MAPPO/MAMBA -- it tracks episode length, which
varies -- so a fixed env-step cadence drifts away from an even train-step
spacing over a run; passing ``train_steps`` keeps evals evenly spaced on the
axis TensorBoard curves are actually compared on).

``act_fn(obs, t, g)`` maps a stacked ``(N, obs_dim)`` float32 observation, the
in-episode step index ``t`` (``t == 0`` ⇒ new episode; recurrent runners reset
their hidden state on it), and the pinned regime id ``g`` for this episode to
an ``(N,)`` int action vector, deterministically. ``g`` is provided because
m3w_adapted's policy takes the regime id as an explicit argument (its
disclosed given-ID protocol, see fidelity.py); regime-blind callers just
ignore the third argument.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from hyper_mve.utils.configs import V4Config

__all__ = ["PeriodicEvalProbe"]


class PeriodicEvalProbe:
    def __init__(
        self,
        env_fn: Callable[[], Any],
        cfg: V4Config,
        writer,                      # torch.utils.tensorboard.SummaryWriter | None
        act_fn: Callable[[np.ndarray, int, int], np.ndarray],
        every_env_steps: int = 10_000,
        every_train_steps: Optional[int] = None,
        episodes_per_regime: int = 2,
        tag_prefix: str = "eval",
        fidelity_fn: Optional[Callable[[], Optional[dict]]] = None,
    ) -> None:
        self._env_fn = env_fn
        self._cfg = cfg
        self._writer = writer
        self._act_fn = act_fn
        self._every = int(every_env_steps)
        self._every_train = int(every_train_steps) if every_train_steps else None
        self._episodes = int(episodes_per_regime)
        self._prefix = tag_prefix
        # 2026-07-20: World Fidelity must log alongside training, not just
        # post-hoc (Model Generalization is fine post-training only; reward
        # is already this class's job). Zero-arg callable -- typically a
        # closure over compute_fidelity_report(runner, env_fn, grid, ...) --
        # called at the SAME cadence as the reward probe; same fire/skip
        # semantics, no separate cadence state, so the two curves share x.
        # None for runners without a learned world model (predict_rewards
        # absent or None -- see fidelity.py), which is most of them.
        self._fidelity_fn = fidelity_fn
        self._env = None
        self._next_at = 0            # first call always fires (near-init point)
        self._next_train_at = 0      # gates on train_steps instead when set

        if cfg.eval.eval_regime_grid is not None:
            self._grid = tuple(int(g) for g in cfg.eval.eval_regime_grid)
        else:
            from hyper_mve.utils.schemas import get_regime_family
            self._grid = tuple(range(get_regime_family(cfg.env).size))
        train_ids = cfg.env.train_regime_ids
        self._seen_ids = (
            set(self._grid) if train_ids is None else {int(i) for i in train_ids}
        )

    # ------------------------------------------------------------------ api

    def maybe_run(self, env_steps: int, train_steps: Optional[int] = None) -> None:
        if self._writer is None:
            return
        if self._every_train is not None and train_steps is not None:
            if train_steps < self._next_train_at:
                return
            self._next_train_at = max(
                self._every_train,
                (train_steps // self._every_train + 1) * self._every_train,
            )
        else:
            if env_steps < self._next_at:
                return
            self._next_at = max(self._every, (env_steps // self._every + 1) * self._every)
        self._run(int(env_steps))

    def close(self) -> None:
        if self._env is not None:
            close = getattr(self._env, "close", None)
            if callable(close):
                close()
            self._env = None

    # ------------------------------------------------------------- internals

    def _run(self, env_steps: int) -> None:
        if self._env is None:
            self._env = self._env_fn()
        n = self._cfg.env.N
        per_regime: dict[int, float] = {}
        for g in self._grid:
            rets = [self._one_episode(g, n) for _ in range(self._episodes)]
            per_regime[g] = float(np.mean(rets)) if rets else 0.0
            self._writer.add_scalar(f"{self._prefix}/return_g{g}", per_regime[g], env_steps)
        vals = list(per_regime.values())
        self._writer.add_scalar(f"{self._prefix}/return_mean",
                                float(np.mean(vals)) if vals else 0.0, env_steps)
        seen = [v for g, v in per_regime.items() if g in self._seen_ids]
        unseen = [v for g, v in per_regime.items() if g not in self._seen_ids]
        if seen:
            self._writer.add_scalar(f"{self._prefix}/return_seen",
                                    float(np.mean(seen)), env_steps)
        if unseen:
            self._writer.add_scalar(f"{self._prefix}/return_unseen",
                                    float(np.mean(unseen)), env_steps)
        if self._fidelity_fn is not None:
            report = self._fidelity_fn()
            if report is not None:
                # Same tag names as the post-hoc fidelity.json write in
                # scripts/train.py's run_one(), so the periodic curve and the
                # final point land on the same TB series.
                self._writer.add_scalar(
                    "fidelity/reward_mae", float(report["reward_mae"]), env_steps)
                for g, v in report["reward_mae_per_regime"].items():
                    self._writer.add_scalar(
                        f"fidelity/reward_mae_regime_{g}", float(v), env_steps)
        self._writer.flush()

    def _one_episode(self, g: int, n: int) -> float:
        obs_dict, _ = self._env.reset(options={"g": int(g)})
        total = 0.0
        t = 0
        while True:
            obs = np.stack(
                [obs_dict[f"agent_{i}"] for i in range(n)], axis=0,
            ).astype(np.float32)
            actions = self._act_fn(obs, t, g)
            action_dict = {f"agent_{i}": int(actions[i]) for i in range(n)}
            obs_dict, reward, term, trunc, _ = self._env.step(action_dict)
            total += float(sum(reward.values()))
            done = [
                bool(term[f"agent_{i}"]) or bool(trunc[f"agent_{i}"]) for i in range(n)
            ]
            t += 1
            if all(done):
                return total
