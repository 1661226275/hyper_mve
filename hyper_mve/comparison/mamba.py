"""MAMBAAlgorithm — external baseline, REAL PORT (pkg-07 spec 06 §4.4 amendment).

Sourcing window opened 2026-07-10 (mid-term baseline comparison): the design D8
stub is replaced by ``_RealMAMBA``, a port of jbr-ai-labs MAMBA @ ``2c97258``
(Egorov & Shpilman, "Scalable Multi-Agent Model-Based Reinforcement Learning",
AAMAS 2022). The DreamerV2-based algorithm core is vendored under ``_mamba/``
(see its ``__init__`` for the file-by-file edit log); this module is the
plumbing layer mirroring ``mappo.py``:

  * drives episode collection from ``RelationCommonsPettingZooEnv`` via the
    injected ``env_fn`` (replaces the upstream ray ``DreamerWorker`` /
    ``DreamerServer``; the buffer-dict schema — action one-hot / observation /
    per-agent reward / done / fake / all-ones avail_action / trailing ``last``
    flag — replicates ``DreamerController.dispatch_buffer``). The upstream
    absorbing-state double-append fires only on natural (non-timeout)
    termination, which RelationCommons never produces (fixed-T truncation), so
    it is not replicated;
  * ``train(cfg, env_fn)`` with the pkg-07 spec 05 §4.2 keyword contract plus
    the optional ``tensorboard_dir`` kwarg → ``PeriodicEvalProbe`` sample-
    efficiency curves;
  * ``evaluate(env_fn, regime_grid, episodes)`` → per-regime deterministic
    rollouts filling the locked rel-v1 ``EvalReport``;
  * checkpoints carry model+actor+critic+optimizers (+``in_dim`` so
    ``load_checkpoint`` can rebuild the learner standalone).

Comparability notes (disclosed in the thesis/report): imagination-phase reward
is averaged across agents (upstream cooperative assumption, kept); GAMMA
inherits ``cfg.train.gamma`` (=0.95, suite-wide); the runner-lr contract maps
to ACTOR_LR/VALUE_LR while MODEL_LR keeps the upstream 2e-4.
"""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Final, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from hyper_mve.comparison import _FORBIDDEN_INFO_KEYS
from hyper_mve.comparison.base import (
    ExternalBaselineRunner,
    split_seen_unseen_regimes,
)
from hyper_mve.utils.configs import V4Config
from hyper_mve.utils.eval.eval_report import EvalReport, regime_names_for

#: 2026-07-28 eval cadence in ENV steps (user-locked). Gating on gradient steps
#: gave this algorithm far too few points: mamba/happo logged every ~20000 env
#: steps (52 points) and mbom every ~80000 (14 points across a whole 1M run),
#: because each takes a different number of updates per env step. 4000 env steps
#: => 250 points. mazero and m3w_adapted keep their original schedules.
_PROBE_EVERY_ENV_STEPS: int = 4000


#: Module-level toggle (pkg-07 spec 06 §4.6 + spec 01 §3.2 line 139).
#: ``False`` → :class:`_MAMBAStub`; ``True`` → :class:`_RealMAMBA`.
IS_SOURCED: Final[bool] = True

#: Provenance marker for the audit trail (pkg-07 spec 05 §1).
_VENDORED_FROM: str = (
    "jbr-ai-labs/mamba @ 2c97258 (Egorov & Shpilman, AAMAS 2022) — "
    "vendored 2026-07-10 under baselines/external/_mamba/"
)

#: 2026-07-27 PARAMETER-MATCHED CAPACITY (disclosed deviation from upstream).
#: Upstream's SMAC-flavoured defaults give a ~8.4M-parameter network on this
#: env — 6.7x the method under comparison (1.26M) — so a win could be bought
#: with capacity rather than algorithm. These widths were chosen by measuring
#: DreamerModel+Actor+Critic parameter counts offline and picking the setting
#: closest to the method: 1,235,645 vs 1,260,293 (within 2%).
#:
#: Capacity is a RUNNER VARIANT, not a global switch: ``mamba`` keeps upstream's
#: own tuned widths (shrinking a baseline to match us would look like
#: handicapping it) and ``mamba_pm`` is the parameter-matched control. Both are
#: registered in hyper_mve.comparison, so the two rows coexist in one registry
#: without overwriting each other's run directories.
#: Upstream values (``mamba``): HIDDEN/MODEL_HIDDEN/EMBED/DETERMINISTIC = 256,
#: N_CATEGORICALS = N_CLASSES = 32 (STOCHASTIC = 1024), heads = 256.
_MATCHED_CAPACITY: dict = {
    "HIDDEN": 128, "MODEL_HIDDEN": 128, "EMBED": 128, "DETERMINISTIC": 128,
    "N_CATEGORICALS": 16, "N_CLASSES": 16,
    "VALUE_HIDDEN": 128, "ACTION_HIDDEN": 128,
    "REWARD_HIDDEN": 128, "PCONT_HIDDEN": 128,
}


def _apply_capacity(mcfg) -> None:
    """Apply the parameter-matched capacity in place.

    Derived fields (STOCHASTIC / FEAT / GLOBAL_FEAT) are recomputed here because
    MambaConfig.__init__ folds them in at construction time.
    """
    for k, v in _MATCHED_CAPACITY.items():
        setattr(mcfg, k, v)
    mcfg.STOCHASTIC = mcfg.N_CATEGORICALS * mcfg.N_CLASSES
    mcfg.FEAT = mcfg.STOCHASTIC + mcfg.DETERMINISTIC
    mcfg.GLOBAL_FEAT = mcfg.FEAT + mcfg.EMBED

_DEFAULT_LR: float = 3e-4        # actor/value LR (grid midpoint)


def _check_forbidden_info(info_dict: dict[str, dict]) -> None:
    """pkg-07 spec 05 §5.2 / spec 06 §6.2 runtime guard (same as mappo.py)."""
    for agent_key, per_agent_info in info_dict.items():
        leaked = _FORBIDDEN_INFO_KEYS & set(per_agent_info)
        assert not leaked, (
            f"MAMBAAlgorithm consumed forbidden info keys {leaked} at agent "
            f"{agent_key}. Either env_fn was constructed with oracle_mode=True "
            "(violates spec 04 §7) or the adapter is leaking."
        )


class _MAMBAStub(ExternalBaselineRunner):
    """Stub fallback kept for the ``IS_SOURCED=False`` binding path (§4.5)."""

    #: capacity variant flag; True only in MAMBAParamMatchedAlgorithm ("mamba_pm")
    PARAM_MATCHED: bool = False

    def __init__(self, cfg: V4Config) -> None:
        super().__init__(cfg)
        raise NotImplementedError(
            "MAMBA stub — IS_SOURCED is False. Flip "
            "hyper_mve.comparison.mamba.IS_SOURCED to use _RealMAMBA."
        )

    def train(self, cfg, env_fn, *, total_env_steps=0, lr=0.0, seed=0, **kwargs):
        raise NotImplementedError("MAMBA stub — see __init__.")

    def evaluate(self, env_fn, regime_grid, episodes):
        raise NotImplementedError("MAMBA stub — see __init__.")

    def save_checkpoint(self, path: Union[Path, str]) -> None:
        raise NotImplementedError("MAMBA stub — see __init__.")

    def load_checkpoint(self, path: Union[Path, str]) -> None:
        raise NotImplementedError("MAMBA stub — see __init__.")


class _RealMAMBA(ExternalBaselineRunner):
    """Vendored MAMBA impl (spec 06 §4.4), real port — see module docstring."""

    #: capacity variant flag; True only in MAMBAParamMatchedAlgorithm ("mamba_pm")
    PARAM_MATCHED: bool = False

    def __init__(self, cfg: V4Config, lr: Optional[float] = None) -> None:
        super().__init__(cfg)
        self._vendored_from: str = _VENDORED_FROM
        if lr is None:
            grid = cfg.baselines.external_lr_sweep_grid.get(
                "mamba", (_DEFAULT_LR,),
            )
            lr = grid[len(grid) // 2]
        self._lr: float = float(lr)
        self._learner = None
        self._mcfg = None
        self._tb = None                       # SummaryWriter | None
        self._env_steps: int = 0
        self._probe_prev: tuple = (None, None)
        self._last_episode_return: float = 0.0

    # ------------------------------------------------------------- building

    def _build(self, obs_dim: int) -> None:  # noqa: D401 (see _apply_capacity)
        from hyper_mve.comparison._mamba.config import MambaConfig
        from hyper_mve.comparison._mamba.learner import DreamerLearner

        mcfg = MambaConfig()
        mcfg.IN_DIM = int(obs_dim)
        mcfg.ACTION_SIZE = int(self.cfg.env.A)
        if self.PARAM_MATCHED:
            _apply_capacity(mcfg)
        # Comparability: inherit the suite-wide discount like mappo.py does.
        mcfg.GAMMA = float(self.cfg.train.gamma)
        mcfg.DISCOUNT = float(self.cfg.train.gamma)
        mcfg.ACTOR_LR = self._lr
        mcfg.VALUE_LR = self._lr
        self._mcfg = mcfg
        self._learner = DreamerLearner(mcfg, metrics_cb=self._metrics_cb)

    def _metrics_cb(self, metrics: dict) -> None:
        if self._tb is None:
            return
        for k, v in metrics.items():
            try:
                if hasattr(v, "detach"):
                    v = v.detach()
                self._tb.add_scalar(f"mamba/{k}", float(v), self._env_steps)
            except (TypeError, ValueError):
                pass

    # ------------------------------------------------------------ acting

    @torch.no_grad()
    def _select_actions(self, obs_t, prev_actions, prev_state, deterministic: bool):
        """One controller step (upstream DreamerController.step, in-process).

        obs_t: (1, N, in_dim) on the learner device. With all actions always
        available the upstream avail-mask resample is distribution-identical
        to the actor's own sample, so the stochastic path uses it directly.
        """
        state = self._learner.model(obs_t, prev_actions, prev_state)
        feats = state.get_features()
        action, pi = self._learner.actor(feats)
        if deterministic:
            idx = pi.argmax(-1)
            action = F.one_hot(idx, pi.shape[-1]).to(pi.dtype)
        return action, state

    def _probe_act(self, obs: np.ndarray, t: int, g: int) -> np.ndarray:
        """Deterministic act_fn for PeriodicEvalProbe (recurrent state per episode)."""
        del g  # regime-blind
        if t == 0:
            self._probe_prev = (None, None)
        device = self._mcfg.DEVICE
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action, state = self._select_actions(obs_t, *self._probe_prev, deterministic=True)
        self._probe_prev = (action, state)
        return action.squeeze(0).argmax(-1).cpu().numpy()

    # ----------------------------------------------------------- one-episode

    def _run_one_episode(
        self,
        env,
        deterministic: bool = False,
        g_override: Optional[int] = None,
        collect: bool = True,
    ) -> int:
        n = int(self.cfg.env.N)
        a_size = int(self._mcfg.ACTION_SIZE)
        device = self._mcfg.DEVICE
        reset_options = {"g": int(g_override)} if g_override is not None else None
        obs_dict, info = env.reset(options=reset_options)
        _check_forbidden_info(info)

        buffer = defaultdict(list) if collect else None
        prev_actions = None
        prev_state = None
        episode_return = 0.0
        steps = 0

        while True:
            obs_np = np.stack(
                [obs_dict[f"agent_{i}"] for i in range(n)], axis=0,
            ).astype(np.float32)
            obs_t = torch.as_tensor(obs_np, device=device).unsqueeze(0)
            action, state = self._select_actions(
                obs_t, prev_actions, prev_state, deterministic,
            )
            prev_actions, prev_state = action, state
            acts = action.squeeze(0).argmax(-1).cpu().numpy()
            action_dict = {f"agent_{i}": int(acts[i]) for i in range(n)}
            obs_dict, reward_dict, term_dict, trunc_dict, info = env.step(action_dict)
            _check_forbidden_info(info)
            r = np.array(
                [reward_dict[f"agent_{i}"] for i in range(n)], dtype=np.float32,
            )
            done = np.array(
                [
                    float(bool(term_dict[f"agent_{i}"]) or bool(trunc_dict[f"agent_{i}"]))
                    for i in range(n)
                ],
                dtype=np.float32,
            )
            episode_return += float(r.sum())
            steps += 1

            if collect:
                buffer["action"].append(action.squeeze(0).cpu().numpy())
                buffer["observation"].append(obs_np)
                buffer["reward"].append(r.reshape(n, 1))
                buffer["done"].append(done.reshape(n, 1))
                buffer["fake"].append(np.zeros((n, 1), dtype=np.float32))
                buffer["avail_action"].append(np.ones((n, a_size), dtype=np.float32))

            if bool(done.all()):
                break

        if collect:
            rollout = {k: np.asarray(v, dtype=np.float32) for k, v in buffer.items()}
            last = np.zeros_like(rollout["done"])
            last[-1] = 1.0
            rollout["last"] = last
            self._learner.step(rollout)

        self._last_episode_return = episode_return
        return steps

    # -------------------------------------------------------------- training

    def train(
        self,
        cfg: V4Config,
        env_fn: Callable[[], Any],
        *,
        total_env_steps: int = 0,
        lr: float = 0.0,
        seed: int = 0,
        max_train_steps: Optional[int] = None,
        tensorboard_dir: Optional[str] = None,
        checkpoint_every_env_steps: int = 25_000,
        **kwargs: Any,
    ) -> None:
        unified_logger = kwargs.pop("unified_logger", None)
        del kwargs  # forward-compat
        if total_env_steps > 0:
            budget = int(total_env_steps)
        elif max_train_steps is not None:
            budget = int(max_train_steps)
        else:
            budget = int(cfg.train.max_train_steps)
        if lr > 0:
            self._lr = float(lr)
        if seed:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))

        env = env_fn()
        obs_dict, info = env.reset(seed=int(seed) if seed else None)
        _check_forbidden_info(info)
        obs_dim = int(obs_dict["agent_0"].shape[-1])
        self._build(obs_dim)

        probe = None
        if tensorboard_dir or unified_logger is not None:
            from hyper_mve.comparison._probe import PeriodicEvalProbe

            if unified_logger is not None:
                # duck-types SummaryWriter (native_step_unit="env")
                self._tb = unified_logger
            else:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(tensorboard_dir)

            def _fidelity_fn():
                from hyper_mve.utils.eval.fidelity import compute_fidelity_report
                from hyper_mve.utils.schemas import get_regime_family

                grid = tuple(range(get_regime_family(cfg.env).size))
                return compute_fidelity_report(self, env_fn, grid,
                                               episodes=2, seed=1234)

            probe = PeriodicEvalProbe(
                env_fn, cfg, self._tb, act_fn=self._probe_act,
                every_env_steps=_PROBE_EVERY_ENV_STEPS, episodes_per_regime=8,
                fidelity_fn=_fidelity_fn,
            )

        # Periodic model checkpoints (env-step cadence) into the run dir, so a
        # long run that will not finish by a deadline can still be evaluated at
        # intermediate budgets with the real 30-episode eval (accurate points to
        # fit a sample-efficiency projection, unlike the 2-episode probe).
        ckpt_dir = None
        next_ckpt = 0
        if tensorboard_dir and checkpoint_every_env_steps > 0:
            ckpt_dir = Path(tensorboard_dir).parent
            next_ckpt = int(checkpoint_every_env_steps)

        self._env_steps = 0
        n_train_steps = 0  # one learner update round per collected episode
        while self._env_steps < budget:
            steps = self._run_one_episode(env, deterministic=False, collect=True)
            self._env_steps += steps
            n_train_steps += 1
            if unified_logger is not None:
                # one learner update round per collected episode (the vendored
                # DreamerLearner cadence) — proxy for the train_steps axis
                unified_logger.set_progress(env_steps=self._env_steps)
                unified_logger.advance(train_steps=1)
            if probe is not None:
                probe.maybe_run(self._env_steps, train_steps=n_train_steps)
            if ckpt_dir is not None and self._env_steps >= next_ckpt:
                self.save_checkpoint(ckpt_dir / f"step_{self._env_steps}.pt")
                next_ckpt += int(checkpoint_every_env_steps)
        env.close()
        if probe is not None:
            probe.close()
        if self._tb is not None:
            self._tb.flush()

    # -------------------------------------------------------------- evaluate

    def evaluate(
        self,
        env_fn: Callable[[], Any],
        regime_grid: tuple[int, ...],
        episodes: int,
    ) -> EvalReport:
        """Per-regime deterministic rollouts → locked rel-v1 EvalReport."""
        if self._learner is None:
            tmp_env = env_fn()
            obs_dict, _ = tmp_env.reset()
            obs_dim = int(obs_dict["agent_0"].shape[-1])
            tmp_env.close()
            self._build(obs_dim)

        t0 = time.time()
        env = env_fn()
        return_per_regime: dict[int, float] = {}
        return_per_regime_sem: dict[int, float] = {}
        episodes_per_regime: dict[int, int] = {}
        all_returns: list[float] = []
        env_steps_total = 0
        # Per-episode returns stashed for distribution views (box plots);
        # same episodes that produce the report means.
        self._eval_episode_returns: dict[int, list[float]] = {}

        for g in regime_grid:
            g_returns: list[float] = []
            for _ in range(int(episodes)):
                episode_steps = self._run_one_episode(
                    env, deterministic=True, g_override=int(g), collect=False,
                )
                g_returns.append(float(self._last_episode_return))
                env_steps_total += int(episode_steps)
            self._eval_episode_returns[int(g)] = list(g_returns)
            return_per_regime[int(g)] = float(np.mean(g_returns)) if g_returns else 0.0
            sem = (
                float(np.std(g_returns) / max(np.sqrt(len(g_returns)), 1.0))
                if len(g_returns) > 1 else 0.0
            )
            return_per_regime_sem[int(g)] = sem
            episodes_per_regime[int(g)] = int(len(g_returns))
            all_returns.extend(g_returns)
        env.close()

        zs_seen, zs_unseen_v = split_seen_unseen_regimes(self.cfg, return_per_regime)
        return_mean = float(np.mean(all_returns)) if all_returns else 0.0
        return_sem = (
            float(np.std(all_returns) / max(np.sqrt(len(all_returns)), 1.0))
            if len(all_returns) > 1 else 0.0
        )

        return EvalReport(
            variant="mamba",
            regime_names=regime_names_for(self.cfg),
            seed=0,
            config_hash="0" * 40,
            eval_mode="planner",
            eval_planner_mode=self.cfg.eval.eval_planner_mode,
            return_mean=return_mean,
            return_sem=return_sem,
            return_zero_shot_seen=zs_seen,
            return_zero_shot_unseen=zs_unseen_v,
            return_zero_shot_gap=zs_seen - zs_unseen_v,
            return_per_regime=MappingProxyType(return_per_regime),
            return_per_regime_sem=MappingProxyType(return_per_regime_sem),
            episodes_per_regime=MappingProxyType(episodes_per_regime),
            planner_prior_return_gap=0.0,
            direct_inference_return_mean=return_mean,
            planner_full_return_mean=return_mean,
            walltime_seconds=float(time.time() - t0),
            env_steps_evaluated=int(env_steps_total),
            episodes_total=int(len(all_returns)),
            info_gating_strict=True,
            set_context_subjective_oracle_leak=False,
        )

    # ----------------------------------------------------------- ckpt + diag

    def save_checkpoint(self, path: Union[Path, str]) -> None:
        if self._learner is None:
            return
        ckpt = {
            "model_state_dict": self._learner.model.state_dict(),
            "actor_state_dict": self._learner.actor.state_dict(),
            "critic_state_dict": self._learner.critic.state_dict(),
            "model_optimizer_state_dict": self._learner.model_optimizer.state_dict(),
            "actor_optimizer_state_dict": self._learner.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self._learner.critic_optimizer.state_dict(),
            "in_dim": int(self._mcfg.IN_DIM),
            "lr": self._lr,
            "vendored_from": _VENDORED_FROM,
        }
        torch.save(ckpt, path)

    def load_checkpoint(self, path: Union[Path, str]) -> None:
        device = None if self._mcfg is None else self._mcfg.DEVICE
        ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
        if self._learner is None:
            self._lr = float(ckpt.get("lr", self._lr))
            self._build(int(ckpt["in_dim"]))
        self._learner.model.load_state_dict(ckpt["model_state_dict"])
        self._learner.actor.load_state_dict(ckpt["actor_state_dict"])
        self._learner.critic.load_state_dict(ckpt["critic_state_dict"])
        self._learner.model_optimizer.load_state_dict(ckpt["model_optimizer_state_dict"])
        self._learner.actor_optimizer.load_state_dict(ckpt["actor_optimizer_state_dict"])
        self._learner.critic_optimizer.load_state_dict(ckpt["critic_optimizer_state_dict"])
        self._lr = float(ckpt.get("lr", self._lr))

    def param_count(self) -> int:
        if self._learner is None:
            return 0
        return (
            sum(p.numel() for p in self._learner.model.parameters())
            + sum(p.numel() for p in self._learner.actor.parameters())
            + sum(p.numel() for p in self._learner.critic.parameters())
        )

    # --------------------------------------------------------- fidelity hook
    def predict_rewards(self, episode) -> Optional[np.ndarray]:
        """fidelity-v1 hook: one-step per-agent reward predictions (T, N).

        Replicates the vendored ``loss.py:model_loss`` convention exactly:
        RSSM posterior rollout over the real (obs, action) sequence, then
        ``reward_model(cat([post.stoch, deters]))`` scores ``reward[1:]`` —
        MAMBA's own alignment cannot score a sequence's FIRST transition,
        so row 0 is NaN (masked by the metric, coverage disclosed).
        """
        if self._learner is None:
            return None
        from hyper_mve.comparison._mamba.rnns import rollout_representation

        model = self._learner.model
        device = self._mcfg.DEVICE
        obs_seq = np.asarray(episode["obs"][:-1], dtype=np.float32)  # (T,N,D)
        actions = np.asarray(episode["actions"], dtype=np.int64)     # (T,N)
        T, n = actions.shape
        a_size = int(self._mcfg.ACTION_SIZE)

        model.eval()
        with torch.no_grad():
            obs_t = torch.as_tensor(obs_seq, device=device).unsqueeze(1)
            action_t = F.one_hot(
                torch.as_tensor(actions, device=device), a_size
            ).float().unsqueeze(1)                                # (T,1,N,A)
            done_t = torch.zeros(T, 1, n, 1, device=device)
            embed = model.observation_encoder(
                obs_t.reshape(-1, n, obs_t.shape[-1]))
            embed = embed.reshape(T, 1, n, -1)
            prev_state = model.representation.initial_state(1, n, device=device)
            _, post, deters = rollout_representation(
                model.representation, T, embed, action_t, prev_state, done_t)
            feat = torch.cat([post.stoch, deters], -1)            # (T-1,1,N,F)
            r_hat = model.reward_model(feat)                      # (T-1,1,N,1)
        preds = np.full((T, n), np.nan, dtype=np.float64)
        preds[1:] = r_hat.reshape(T - 1, n).cpu().numpy()
        return preds


# pkg-07 spec 06 §4.6 byte-identical class-binding:
MAMBAAlgorithm = _RealMAMBA if IS_SOURCED else _MAMBAStub


class MAMBAParamMatchedAlgorithm(MAMBAAlgorithm):  # type: ignore[misc,valid-type]
    """MAMBA at the parameter-matched capacity (~1.24M vs upstream's ~8.4M).

    Registered as ``mamba_pm``. The unmodified ``mamba`` keeps upstream's own
    tuned widths so the baseline is never reported only in a handicapped form;
    this subclass is the capacity control for the like-for-like row.
    """

    PARAM_MATCHED = True


__all__ = ["MAMBAAlgorithm", "MAMBAParamMatchedAlgorithm", "IS_SOURCED"]
