"""mazero_mixed — the MAZero-based mixed-game METHOD behind the runner contract.

This is NOT a comparison baseline: it wraps the fork at
``hyper_mve/mazero_mixed/`` (MAZero ICLR-2024 lineage, stage 1–3 migration:
vectorized decoupled search, hypernet-generated per-agent subjective heads,
Bayes-averaged leaf values, belief curriculum) behind the same 4-method
``ExternalBaselineRunner`` interface so the pkg-08 sweep harness, rel-v1
EvalReports and game-metrics pipeline drive it unchanged.

Contract notes / disclosed deviations:
  * ``train`` ignores ``env_fn`` — the fork builds its own data-collection
    envs with ``oracle_mode=True`` (train-time-only privileged g for belief
    supervision, the method's CTDE design). The exact EnvConfig comes from the
    harness cfg via ``GameConfig.env_cfg_override``, so the env identity is
    identical to what ``env_fn`` would construct.
  * ``evaluate`` DOES consume ``env_fn`` (oracle-free) and executes the
    decentralized prior policy (belief GRU from own history + policy head) —
    the strict-CTDE execution mode. Search-at-evaluation is a separate,
    disclosed protocol and is not used here.
"""
from __future__ import annotations

import os
import sys
import time
import tempfile
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Union

import numpy as np

from hyper_mve.utils.configs import V4Config
from hyper_mve.utils.eval.eval_report import EvalReport
from hyper_mve.comparison.base import (
    ExternalBaselineRunner,
    split_seen_unseen_regimes,
)

_FORK_DIR = Path(__file__).resolve().parent / "mazero_mixed"

# grad-step pacing: ~1 training step per this many env transitions (matches
# the fork smoke ratio 5000/320 ≈ 16).
_ENV_STEPS_PER_GRAD = 16

# Number of MCTS trees searched in parallel by the selfplay worker. This is the
# inference BATCH SIZE on the search hot path: the fork evaluates all leaves of
# all trees in one forward per simulation, so num_pmcts=B turns B batch-1
# forwards into one batch-B forward.
#
# It does NOT change the env-steps-per-gradient-step ratio — core/train.py
# paces gradient steps as
#     target_steps = training_steps * transitions_collected / total_transitions
# so total env steps and total gradient steps are invariant to this value. It
# only makes collection chunkier (more transitions per collect round, hence
# more gradient steps between behaviour-policy refreshes), which the off-policy
# replay buffer + reanalyze worker are designed to absorb.
#
# Measured on this env (39-dim obs, 1.26M params, 25 sims/step, GPU 5):
#   num_pmcts   CUDA ms/step   CUDA ms/tree
#           1           60.8          60.81
#          16           62.5           3.91
#          32           63.5           1.99
# i.e. CUDA wall-time is essentially flat in B — the search is kernel-launch
# bound, not compute bound — so raising this is close to free throughput.
_DEFAULT_NUM_PMCTS = 1


def _ensure_fork_on_path() -> None:
    p = str(_FORK_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


class MAZeroMixedRunner(ExternalBaselineRunner):
    name = "mazero_mixed"

    def __init__(self, cfg: V4Config):
        self.cfg = cfg
        self._model = None            # torch nn.Module (HyperMAMuZeroNet)
        self._game_config = None
        self._ablation = "none"       # phase-7 arm name (fork-argv seam)
        self._num_pmcts = _DEFAULT_NUM_PMCTS
        self._eval_episode_returns: dict[int, list[float]] = {}

    @staticmethod
    def _device_of(model):
        """The device the model actually lives on (eval/fidelity follow it)."""
        try:
            return next(model.parameters()).device
        except StopIteration:  # pragma: no cover - parameterless model
            import torch

            return torch.device("cpu")

    # ------------------------------------------------------------ internals
    def _build_game_config(self, *, total_env_steps: int, lr: float, seed: int):
        _ensure_fork_on_path()
        import importlib
        core_config = importlib.import_module("core.config")

        total_env_steps = int(max(total_env_steps, 1))
        training_steps = max(1, total_env_steps // _ENV_STEPS_PER_GRAD)
        start_transition = int(min(256, max(64, total_env_steps // 4)))
        argv = [
            "--opr", "train_sync", "--case", "relation", "--env", "rel_duo",
            "--exp_name", "runner", "--seed", str(int(seed)),
            # all three workers on GPU: selfplay/reanalyze default to CPU in
            # the fork (core/config.py:40,132) and the adapter previously
            # passed only --train_on_gpu, so search AND the reanalyze
            # batch-refresh ran single-threaded on CPU. Both flags degrade
            # gracefully — core/config.py ANDs them with cuda.is_available().
            "--train_on_gpu", "--selfplay_on_gpu", "--reanalyze_on_gpu",
            "--data_actors", "1", "--num_pmcts", str(int(self._num_pmcts)),
            "--reanalyze_actors", "1",
            "--test_interval", str(10 * training_steps + 1),
            "--target_model_interval", "50",
            "--batch_size", "64", "--num_simulations", "25",
            "--sampled_action_times", "5",
            "--training_steps", str(training_steps), "--last_step", "0",
            "--lr", str(float(lr) if lr else 0.02), "--lr_adjust_func", "const",
            "--max_grad_norm", "10",
            "--total_transitions", str(total_env_steps),
            "--start_transition", str(start_transition),
            "--target_value_type", "pred-re", "--revisit_policy_search_rate", "1",
            "--use_off_correction", "--value_transform_type", "scalar",
            "--use_priority", "--use_max_priority",
            "--subjective_model", "--decoupled_selection",
        ]
        if self._ablation and self._ablation != "none":
            from hyper_mve.ablation.arms import apply_arm_argv

            argv = apply_arm_argv(argv, self._ablation)
        args = core_config.parse_args(argv)
        relation_pkg = importlib.import_module("config.relation")

        # env identity comes from the harness cfg, not the env_name preset
        game_config_cls = relation_pkg.GameConfig
        game_config_cls.env_cfg_override = self.cfg.env
        try:
            game_config = game_config_cls(args)
        finally:
            game_config_cls.env_cfg_override = None
        game_config.env_cfg_override = self.cfg.env
        return game_config

    def _lazy_model(self):
        if self._model is None:
            import torch

            self._game_config = self._build_game_config(
                total_env_steps=1024, lr=0.02, seed=0
            )
            model = self._game_config.get_uniform_network()
            # checkpoint-eval path (no train() in this process): put the model
            # on the GPU too, so evaluate()/predict_rewards() are not pinned to
            # the CPU just because training happened in another run.
            if torch.cuda.is_available():
                model = model.cuda()
            self._model = model
            self._model.eval()
        return self._model

    # -------------------------------------------------------------- train
    def train(self, cfg, env_fn, *, total_env_steps: int = 0, lr: float = 0.0,
              seed: int = 0, **kwargs) -> None:
        self.cfg = cfg
        self._ablation = str(kwargs.get("ablation") or "none")
        self._num_pmcts = int(kwargs.get("num_pmcts") or _DEFAULT_NUM_PMCTS)
        _ensure_fork_on_path()
        import torch
        from torch.utils.tensorboard import SummaryWriter
        from core.train import train_sync_serial

        game_config = self._build_game_config(
            total_env_steps=total_env_steps, lr=lr, seed=seed
        )
        tb_dir = kwargs.get("tensorboard_dir") or tempfile.mkdtemp(
            prefix="mazero_mixed_tb_"
        )
        # the fork's results dir (checkpoints/logs) lives under the fork tree;
        # model_dir/model_path are derived at config construction, so re-derive
        # them together with exp_path and create the directories the serial
        # trainer expects (main.py normally does this via make_results_dir).
        game_config.exp_path = os.path.join(
            str(_FORK_DIR), "results", "relation", "rel_duo", "runner",
            f"seed={seed}",
        )
        game_config.model_dir = os.path.join(game_config.exp_path, "model")
        game_config.model_path = os.path.join(game_config.exp_path, "model.p")
        os.makedirs(game_config.model_dir, exist_ok=True)
        os.makedirs(os.path.join(game_config.exp_path, "logs"), exist_ok=True)
        # UnifiedLogger duck-types SummaryWriter (native_step_unit="train":
        # the fork's core/log.py logs at gradient steps and its
        # train/transitions_collected scalar self-calibrates the env ratio).
        summary_writer = kwargs.get("unified_logger")
        if summary_writer is None:
            summary_writer = SummaryWriter(tb_dir, flush_secs=30)
        model, weights = train_sync_serial(game_config, summary_writer, None)
        model.set_weights(weights)
        model.eval()
        # keep the trained model on the device it was trained on — evaluate()
        # and predict_rewards() together run len(grid)*episodes*T_max forward
        # passes, which used to be forced onto the CPU by a .cpu() here.
        self._model = model
        self._game_config = game_config

    # ------------------------------------------------------------ evaluate
    def evaluate(self, env_fn: Callable[[], Any], regime_grid, episodes) -> EvalReport:
        import torch

        model = self._lazy_model()
        model.eval()
        device = self._device_of(model)
        t0 = time.time()

        env = env_fn()
        agents = list(env.possible_agents)
        N = len(agents)

        return_per_regime: dict[int, float] = {}
        return_per_regime_sem: dict[int, float] = {}
        episodes_per_regime: dict[int, int] = {}
        all_returns: list[float] = []
        env_steps_total = 0
        self._eval_episode_returns = {}

        with torch.no_grad():
            for g in regime_grid:
                g_returns: list[float] = []
                for ep in range(int(episodes)):
                    obs_dict, _ = env.reset(seed=10_000 + 97 * int(g) + ep,
                                            options={"g": int(g)})
                    # plain-MAZero ablation (no_subjective) has no belief net
                    subjective = hasattr(model, "belief_net")
                    hidden = (model.belief_net.init_hidden(1, N, device=device)
                              if subjective else None)
                    ep_ret, steps = 0.0, 0
                    done = False
                    while not done:
                        obs = np.stack([obs_dict[a] for a in agents]).astype(np.float32)
                        obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
                        if subjective:
                            hidden, g_hat = model.belief_net.step(obs_t, hidden)
                            model.set_belief(g_hat)
                        out = model.initial_inference(obs_t)
                        logits = np.asarray(out.policy_logits).reshape(N, -1)
                        acts = {a: int(np.argmax(logits[i])) for i, a in enumerate(agents)}
                        obs_dict, rew, term, trunc, _ = env.step(acts)
                        ep_ret += float(sum(rew.values()))
                        steps += 1
                        done = bool(any(term.values()) or any(trunc.values()))
                    g_returns.append(ep_ret)
                    env_steps_total += steps
                self._eval_episode_returns[int(g)] = list(g_returns)
                return_per_regime[int(g)] = float(np.mean(g_returns)) if g_returns else 0.0
                return_per_regime_sem[int(g)] = (
                    float(np.std(g_returns) / max(np.sqrt(len(g_returns)), 1.0))
                    if len(g_returns) > 1 else 0.0
                )
                episodes_per_regime[int(g)] = len(g_returns)
                all_returns.extend(g_returns)
        env.close()

        zs_seen, zs_unseen = split_seen_unseen_regimes(self.cfg, return_per_regime)
        return_mean = float(np.mean(all_returns)) if all_returns else 0.0
        return_sem = (
            float(np.std(all_returns) / max(np.sqrt(len(all_returns)), 1.0))
            if len(all_returns) > 1 else 0.0
        )
        return EvalReport(
            variant="mazero_mixed",
            seed=0,
            config_hash="0" * 40,
            eval_mode="prior",
            eval_planner_mode=self.cfg.eval.eval_planner_mode,
            return_mean=return_mean,
            return_sem=return_sem,
            return_zero_shot_seen=zs_seen,
            return_zero_shot_unseen=zs_unseen,
            return_zero_shot_gap=zs_seen - zs_unseen,
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

    # ---------------------------------------------------------------- ckpt
    def save_checkpoint(self, path: Union[Path, str]) -> None:
        import torch

        if self._model is None:
            return
        torch.save({"model_state_dict": self._model.state_dict()}, str(path))

    def load_checkpoint(self, path: Union[Path, str]) -> None:
        import torch

        model = self._lazy_model()
        ckpt = torch.load(str(path), map_location=self._device_of(model),
                          weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

    def param_count(self) -> int:
        model = self._lazy_model()
        return int(sum(p.numel() for p in model.parameters()))

    # --------------------------------------------------------- fidelity hook
    def predict_rewards(self, episode):
        """fidelity-v1 hook: one-step per-agent reward predictions (T, N).

        Oracle-free: the belief GRU runs over the real observation prefix
        (exactly the strict-CTDE eval path), then the learned reward head
        scores the realized joint action via ``recurrent_inference``.
        Plain-MAZero ablation models (no ``belief_net``) are N/A → None.
        """
        import torch

        model = self._lazy_model()
        if not hasattr(model, "belief_net"):
            return None
        model.eval()
        device = self._device_of(model)
        obs_seq = np.asarray(episode["obs"], dtype=np.float32)   # (T+1, N, D)
        actions = np.asarray(episode["actions"], dtype=np.int64)  # (T, N)
        T, N = actions.shape
        preds = np.zeros((T, N), dtype=np.float64)
        with torch.no_grad():
            hidden = model.belief_net.init_hidden(1, N, device=device)
            for t in range(T):
                obs_t = torch.from_numpy(obs_seq[t]).unsqueeze(0).to(device)
                hidden, g_hat = model.belief_net.step(obs_t, hidden)
                model.set_belief(g_hat)
                out = model.initial_inference(obs_t)
                a_t = torch.from_numpy(actions[t]).reshape(1, N).to(device)
                out2 = model.recurrent_inference(
                    torch.as_tensor(out.hidden_state).to(device), a_t)
                preds[t] = np.asarray(out2.reward).reshape(N)
        return preds
