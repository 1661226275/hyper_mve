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
  * ``evaluate`` DOES consume ``env_fn`` (oracle-free) and runs BOTH execution
    modes on identical initial conditions: the MCTS planner (the acting
    policy, and the headline ``return_mean``) and the decentralized prior
    policy (belief GRU from own history + policy head — the strict-CTDE
    deployable artifact). Their difference is reported as
    ``planner_prior_return_gap``.

    Until 2026-07-19 this method ran the prior alone while stamping the report
    ``eval_planner_mode="planner_full"``, and hardcoded the gap to 0.0. That
    hid a total policy collapse: four architecturally different arms all
    reported 15.10890765041113, which is exactly what a constant-HARVEST
    policy scores (see ``envs/relation_commons/reference_policies.py``).
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

# Episodes for the search pass. Defaults to matching the prior pass so the two
# modes run on identical seeds and the gap between them is a paired difference
# rather than a sampling artifact. Affordable because the planner batches all
# episodes of a regime into one search (see _rollout_planner): the measured
# table above shows wall-time is ~flat in batch size, so the whole search pass
# costs ~num_regimes * T_max * 63 ms ≈ 30 s.
_DEFAULT_PLANNER_EPISODES = None   # None -> match the prior episode count


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
        self._eval_diagnostics: dict[str, Any] = {}
        # None until train() or load_checkpoint() supplies weights. _lazy_model
        # silently builds a randomly-initialized net, so without this an eval
        # that forgot to load a checkpoint returns plausible numbers from noise.
        self._weights_source: str | None = None

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
            # In-training eval curve. This was `10 * training_steps + 1`, and
            # since `step_count % interval == 0` is true at step 0, the fork's
            # test() fired exactly once — on the untrained net — so a 7-hour run
            # produced a single eval point (test/mean_score, value 0.0) and the
            # policy collapse was invisible until the run ended. With
            # --use_mcts_test the curve tracks the ACTING policy.
            # 2026-07-20: pinned to a fixed 500 train steps (was
            # `training_steps // 25`, a value that only happened to equal 500
            # at this grid's specific total_env_steps budget and would drift
            # for any other budget) to match the fixed cadence now used by the
            # external runners' PeriodicEvalProbe (mappo.py/mamba.py,
            # every_train_steps=500) -- same 500-train-step spacing and same
            # 8-episode averaging (test_episodes) across every algo that has a
            # periodic curve at all. ~6 s per eval x 25 evals at this cadence
            # over a 200k-env-step/12.5k-train-step row is <0.1% of the run.
            # Caveat: test() averages over agents and does not pin the regime,
            # so test/mean_score is a trend line, not comparable to
            # eval/return_mean (which sums over agents, per pinned regime) --
            # unlike the external runners' probe, which IS eval/return_mean.
            "--test_interval", "200",
            "--test_episodes", "8", "--use_mcts_test",
            "--target_model_interval", "50",
            "--batch_size", "64", "--num_simulations", "25",
            "--sampled_action_times", "5",
            # The 2026-07-20 prior-collapse fix (root-cover enumeration +
            # Q-softmax policy target) is OPT-IN via the `mcts_fix` ablation
            # arm, NOT the default. A budget-matched seed-0 A/B showed it
            # de-collapses the prior but REDUCES return 22.1 -> 13.3: it walks
            # the policy off the defensive always-HARVEST fallback into the
            # value head's bad advice in the asymmetric regimes (g2/g3 drop
            # ~20 -> ~0; regime_acc ~0.50). The default stays byte-for-byte
            # upstream so existing results remain comparable. To reproduce or
            # extend the fix, use the arms in hyper_mve/ablation/arms.py
            # (`mcts_fix`, `mcts_fix_oracle`).
            "--training_steps", str(training_steps), "--last_step", "0",
            # Cosine decay, not the flat 0.02 that ran the whole 2026-07-18
            # grid: train/value_loss never converged there (rose to ~5, then
            # oscillated 1.25-6.5 for the entire run) while train/reward_loss
            # settled cleanly at ~0.1.
            "--lr", str(float(lr) if lr else 0.02), "--lr_adjust_func", "cos",
            # Exploration. Both of these were identically zero for all 12,400
            # steps of the collapsed runs (workers/temperature pinned at 1.0,
            # workers/greedy_epsilon at 0). In Sampled MuZero the prior seeds
            # its own action samples, so a peaked prior narrows the sample set,
            # which narrows the visit counts, which sharpens the target — a
            # self-reinforcing collapse with nothing opposing it. Epsilon is
            # the lever that breaks it: it forces off-prior actions into the
            # replay data, so the value model gets to see that moving and
            # harvesting beats camping. The temperature schedule
            # (1.0 -> 0.5 -> 0.25 over training) sharpens late-training data
            # quality; it is upstream's default and does not itself explore.
            "--use_change_temperature",
            "--eps_start", "0.25", "--eps_end", "0.02",
            "--eps_annealing_time", str(max(1, training_steps // 2)),
            "--max_grad_norm", "10",
            # 2026-07-20 harvest-collapse residual fix. Post fc_dynamic-repair
            # runs still camped ~85% of the time at 9.9k grad steps despite a
            # verifiably action-sensitive dynamics net and search exploring
            # ~25-30% non-HARVEST at every root. fresh_head_probe.py (with a
            # raw-obs decodability control, so this isn't an overfit artifact)
            # localized it to REPRESENTATION: a fresh head decodes "agent is
            # on a stocked resource cell" from raw obs at 0.93 balanced acc,
            # from the frozen encoder latent at only 0.53-0.61 (chance 0.5).
            # The encoder has no reconstruction loss — it keeps only what
            # reward/value/policy/consistency gradients demand — and a
            # camping-collapsed behaviour policy gives it almost no gradient
            # toward payoff-relevant state. Two levers, orthogonal to any
            # ablation arm:
            #   (a) reward_nonzero_upweight: 3x extra weight (4x total) on
            #       transitions with nonzero raw team reward, on top of a 2x
            #       reward_loss_coeff -- up to ~8x the old per-step reward
            #       gradient on informative (moved or harvested) steps,
            #       moderate enough not to destabilize value/policy loss.
            #   (b) reference_episode_prob: some fraction of fresh self-play
            #       episodes are driven by the scripted-greedy reference
            #       policy instead of the MCTS-selected action (MCTS search
            #       still runs every step, so priors/values/priorities train
            #       normally) -- guarantees walk-then-harvest transitions
            #       reach replay from the start, breaking the encoder-
            #       stagnation -> policy-collapse -> data-starvation loop.
            #       Starts high (40%) and decays to a light floor (8%) over
            #       the first half of training, mirroring the eps schedule.
            "--reward_loss_coeff", "2.0",
            "--reward_nonzero_upweight", "3.0", "--reward_nonzero_eps", "1e-6",
            "--reference_episode_prob_start", "0.4",
            "--reference_episode_prob_end", "0.08",
            "--reference_episode_anneal_steps", str(max(1, training_steps // 2)),
            "--total_transitions", str(total_env_steps),
            "--start_transition", str(start_transition),
            "--target_value_type", "pred-re", "--revisit_policy_search_rate", "1",
            "--use_off_correction", "--value_transform_type", "scalar",
            "--use_priority", "--use_max_priority",
            # Centralized MCTS deployment (2026-07-24, user-locked): one joint
            # tree searches + broadcasts the joint action (select_mode=0, the
            # original MAZero path) instead of per-agent decoupled selection.
            # --decoupled_selection is intentionally NOT passed here so every
            # mazero_mixed run (main + ablations) is centralized; the
            # `joint_selection` ablation arm (which removed it) is now a no-op.
            "--subjective_model",
        ]
        if self._ablation and self._ablation != "none":
            from hyper_mve.ablation.arms import apply_arm_argv

            argv = apply_arm_argv(argv, self._ablation)
        if self._ablation and "anneal_scaled" in self._ablation:
            # ref_bc's hardcoded 28000-step anneal validated at 600K
            # (training_steps=37500) is 74.7% of that budget; preserve the
            # FRACTION rather than the absolute count so longer budgets don't
            # spend a disproportionately larger share of training on the
            # unsupervised floor rate. Appended last so argparse's
            # last-occurrence-wins rule lets it override arms.py's placeholder.
            scaled_anneal = max(1, round(training_steps * 28000 / 37500))
            argv = argv + ["--reference_episode_anneal_steps", str(scaled_anneal)]
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
        tb_dir_kwarg = kwargs.get("tensorboard_dir")
        tb_dir = tb_dir_kwarg or tempfile.mkdtemp(prefix="mazero_mixed_tb_")
        # the fork's results dir (checkpoints/logs) lives under the fork tree;
        # model_dir/model_path are derived at config construction, so re-derive
        # them together with exp_path and create the directories the serial
        # trainer expects (main.py normally does this via make_results_dir).
        #
        # exp_path used to be keyed ONLY by seed (f"seed={seed}"), which is
        # always 0 across a grid's ablation arms/budgets: any two same-seed
        # runs launched concurrently (routine for a multi-GPU grid) wrote
        # into the identical model_dir and clobbered each other's mid-training
        # checkpoints (final checkpoints were safe -- save_checkpoint() writes
        # those to the caller's own per-arm-unique path). tensorboard_dir is
        # already unique per (algo+arm, env, seed) -- scripts/train.py sets it
        # to run_dir/"tb" -- so anchor exp_path off its parent instead of the
        # hardcoded shared path.
        if tb_dir_kwarg:
            run_dir = os.path.dirname(os.path.normpath(tb_dir_kwarg))
            game_config.exp_path = os.path.join(run_dir, "mazero_mixed_fork")
        else:
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
        # Seed the env->train ratio so the canonical env-steps x-axis is right
        # from the first log batch; train/transitions_collected refines it every
        # _log thereafter (UnifiedLogger self-calibration).
        if hasattr(summary_writer, "declare_ratio"):
            summary_writer.declare_ratio(_ENV_STEPS_PER_GRAD)
        model, weights = train_sync_serial(game_config, summary_writer, None)
        model.set_weights(weights)
        model.eval()
        # keep the trained model on the device it was trained on — evaluate()
        # and predict_rewards() together run len(grid)*episodes*T_max forward
        # passes, which used to be forced onto the CPU by a .cpu() here.
        self._model = model
        self._game_config = game_config
        self._weights_source = "train"

    # ------------------------------------------------------------ evaluate
    def _rollout_prior(self, env_fn, g: int, episodes: int, device):
        """Decentralized distilled policy: argmax of the prior, no search.

        Sequential (batch-1) on purpose — this reproduces the historical
        ``eval_mode="prior"`` numbers bit-for-bit, which is what lets archived
        runs stay comparable to new ones.
        """
        import torch

        model = self._model
        env = env_fn()
        agents = list(env.possible_agents)
        N = len(agents)
        subjective = hasattr(model, "belief_net")

        returns: list[float] = []
        action_counts = np.zeros(int(self.cfg.env.A), dtype=np.int64)
        steps_total = 0
        belief_hits, belief_n = 0, 0
        margins: list[float] = []

        for ep in range(int(episodes)):
            obs_dict, _ = env.reset(seed=10_000 + 97 * int(g) + ep,
                                    options={"g": int(g)})
            hidden = (model.belief_net.init_hidden(1, N, device=device)
                      if subjective else None)
            ep_ret, done = 0.0, False
            while not done:
                obs = np.stack([obs_dict[a] for a in agents]).astype(np.float32)
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
                if subjective:
                    hidden, g_hat = model.belief_net.step(obs_t, hidden)
                    model.set_belief(g_hat)
                    # regime_switch_prob == 0 on this preset, so the pinned g is
                    # ground truth for the whole episode.
                    pred = np.asarray(g_hat.detach().cpu()).reshape(N, -1)
                    belief_hits += int((pred.argmax(axis=-1) == g).sum())
                    belief_n += N
                out = model.initial_inference(obs_t)
                logits = np.asarray(out.policy_logits).reshape(N, -1)
                # top1 - top2: separates "argmax by a hair" (a tie-break
                # artifact) from saturated logits (a real optimization failure).
                srt = np.sort(logits, axis=-1)
                margins.append(float(np.mean(srt[:, -1] - srt[:, -2])))
                acts = {a: int(np.argmax(logits[i])) for i, a in enumerate(agents)}
                for a in acts.values():
                    action_counts[a] += 1
                obs_dict, rew, term, trunc, _ = env.step(acts)
                ep_ret += float(sum(rew.values()))
                steps_total += 1
                done = bool(any(term.values()) or any(trunc.values()))
            returns.append(ep_ret)
        env.close()
        stats = {
            "belief_hits": belief_hits,
            "belief_n": belief_n,
            "logit_margin": float(np.mean(margins)) if margins else 0.0,
        }
        return returns, action_counts, steps_total, stats

    def _rollout_planner(self, env_fn, g: int, episodes: int, device, np_random):
        """The acting policy: MCTS search over the learned model.

        Episodes for one regime run in lockstep as a single search batch. That
        is sound here because ``done`` is purely ``step_idx >= T_max`` (see
        ``envs/relation_commons/env.py``), so every episode in the batch ends on
        the same step and no ragged bookkeeping is needed. Batching matters:
        the search is kernel-launch bound, so B trees cost ~the same wall-time
        as one (see ``_DEFAULT_NUM_PMCTS`` for the measured table).
        """
        import torch
        from core.mcts import SampledMCTS
        from core.utils import select_action

        model = self._model
        cfg_fork = self._game_config
        B = int(episodes)
        envs = [env_fn() for _ in range(B)]
        agents = list(envs[0].possible_agents)
        N = len(agents)
        A = int(self.cfg.env.A)
        subjective = hasattr(model, "belief_net")

        obs_dicts = [
            envs[i].reset(seed=10_000 + 97 * int(g) + i, options={"g": int(g)})[0]
            for i in range(B)
        ]
        hidden = (model.belief_net.init_hidden(B, N, device=device)
                  if subjective else None)
        # No action masking in this env (core/game.py legal_actions is all-ones).
        legal = np.ones((B, N, A), dtype=np.float32)

        returns = np.zeros(B, dtype=np.float64)
        action_counts = np.zeros(A, dtype=np.int64)
        visit_counts = np.zeros((N, A), dtype=np.float64)
        visit_entropies: list[float] = []
        # Root-cover health (2026-07-20). child_counts tracks how many distinct
        # joints the root actually offered; coverage is the fraction of
        # (agent, action) pairs present among them -- 1.0 iff --root_cover is
        # doing its job. policy_mass is sum(pred_prob) over the cover: with
        # beta_hat == beta the target is the EXACT restriction of pi to the
        # cover, not an unbiased estimate of E_pi, so if this falls well below
        # 1 as the policy de-collapses the cover needs widening.
        child_counts: list[float] = []
        coverages: list[float] = []
        policy_mass: list[float] = []
        steps_total = 0
        mcts = SampledMCTS(cfg_fork, np_random)

        done = False
        while not done:
            obs = np.stack(
                [np.stack([od[a] for a in agents]) for od in obs_dicts]
            ).astype(np.float32)                              # (B, N, obs_dim)
            obs_t = torch.from_numpy(obs).to(device)
            if subjective:
                hidden, g_hat = model.belief_net.step(obs_t, hidden)
                model.set_belief(g_hat)
            out = model.initial_inference(obs_t)
            search = mcts.batch_search(model, out, legal, device, False, 1.0)

            # Root marginals: what the search actually wanted, before argmax.
            visit_counts += np.asarray(
                search.marginal_visit_count, dtype=np.float64
            ).reshape(B, N, A).sum(axis=0)

            dones = np.zeros(B, dtype=bool)
            for i in range(B):
                pos, ent = select_action(
                    search.sampled_visit_count[i], temperature=1,
                    deterministic=True, np_random=np_random,
                )
                visit_entropies.append(float(ent))
                root_acts = np.asarray(search.sampled_actions[i]).reshape(-1, N)
                child_counts.append(float(len(root_acts)))
                coverages.append(float(np.mean([
                    len(set(root_acts[:, k].tolist())) / float(A) for k in range(N)
                ])))
                policy_mass.append(
                    float(np.sum(np.asarray(search.sampled_pred_probs[i]))))
                joint = np.asarray(search.sampled_actions[i][pos]).reshape(-1)
                acts = {a: int(joint[k]) for k, a in enumerate(agents)}
                for a in acts.values():
                    action_counts[a] += 1
                obs_dicts[i], rew, term, trunc, _ = envs[i].step(acts)
                returns[i] += float(sum(rew.values()))
                steps_total += 1
                dones[i] = bool(any(term.values()) or any(trunc.values()))
            # Lockstep is only valid while every episode ends together; the
            # batched belief/search state has no per-episode masking.
            assert dones.all() or not dones.any(), (
                "planner batch desynchronized: episodes in one regime ended on "
                "different steps, which the lockstep search batch cannot represent"
            )
            done = bool(dones.all())

        for e in envs:
            e.close()
        stats = {
            "visit_entropy": (float(np.mean(visit_entropies))
                              if visit_entropies else 0.0),
            "root_child_count": float(np.mean(child_counts)) if child_counts else 0.0,
            "root_action_coverage": float(np.mean(coverages)) if coverages else 0.0,
            "root_policy_mass_covered": (float(np.mean(policy_mass))
                                         if policy_mass else 0.0),
        }
        return list(returns), action_counts, visit_counts, steps_total, stats

    def evaluate(self, env_fn: Callable[[], Any], regime_grid, episodes,
                 *, planner_episodes: int | None = None,
                 seed: int = 0, config_hash: str | None = None) -> EvalReport:
        """Dual-mode evaluation: the planner (headline) and the prior (disclosed).

        ``return_mean`` is the **planner** return — for a MAZero-family method
        the acting policy is the search, so that is the method's number. The
        distilled prior is reported alongside as ``direct_inference_return_mean``
        and the difference as ``planner_prior_return_gap``: a large gap means
        the search is working but distillation is not, which is exactly the
        failure the 2026-07-18 grid hid by reporting the prior alone.
        """
        import torch

        model = self._lazy_model()
        if self._weights_source is None:
            raise RuntimeError(
                "MAZeroMixedRunner.evaluate() called on a lazily-constructed, "
                "randomly-initialized model — call train() or load_checkpoint() "
                "first. (Evaluating noise returns plausible-looking numbers.)"
            )
        model.eval()
        device = self._device_of(model)
        t0 = time.time()

        # Search costs ~num_simulations forwards per step, so the planner pass
        # gets a smaller episode budget than the prior pass. Both counts are
        # recorded rather than implied.
        n_prior = int(episodes)
        n_planner = int(planner_episodes if planner_episodes is not None
                        else (_DEFAULT_PLANNER_EPISODES or n_prior))
        # Seeds the C-tree's own RNG (mcts_sampled.py: np_random.choice(256)),
        # which breaks visit-count ties — without this the planner number is
        # not reproducible.
        np_random = np.random.RandomState(12345)

        prior_per_regime: dict[int, float] = {}
        planner_per_regime: dict[int, float] = {}
        planner_per_regime_sem: dict[int, float] = {}
        episodes_per_regime: dict[int, int] = {}
        prior_returns_all: list[float] = []
        planner_returns_all: list[float] = []
        prior_actions = np.zeros(int(self.cfg.env.A), dtype=np.int64)
        planner_actions = np.zeros(int(self.cfg.env.A), dtype=np.int64)
        visit_total = None
        env_steps_total = 0
        self._eval_episode_returns = {}

        belief_hits = belief_n = 0
        margins: list[float] = []
        visit_ents: list[float] = []
        root_child_counts: list[float] = []
        root_coverages: list[float] = []
        root_policy_mass: list[float] = []

        with torch.no_grad():
            for g in regime_grid:
                g = int(g)
                p_ret, p_acts, p_steps, p_stats = self._rollout_prior(
                    env_fn, g, n_prior, device)
                prior_per_regime[g] = float(np.mean(p_ret)) if p_ret else 0.0
                prior_returns_all.extend(p_ret)
                prior_actions += p_acts
                env_steps_total += p_steps
                belief_hits += p_stats["belief_hits"]
                belief_n += p_stats["belief_n"]
                margins.append(p_stats["logit_margin"])

                s_ret, s_acts, s_visits, s_steps, s_stats = self._rollout_planner(
                    env_fn, g, n_planner, device, np_random)
                visit_ents.append(s_stats["visit_entropy"])
                root_child_counts.append(s_stats["root_child_count"])
                root_coverages.append(s_stats["root_action_coverage"])
                root_policy_mass.append(s_stats["root_policy_mass_covered"])
                planner_per_regime[g] = float(np.mean(s_ret)) if s_ret else 0.0
                planner_per_regime_sem[g] = (
                    float(np.std(s_ret) / max(np.sqrt(len(s_ret)), 1.0))
                    if len(s_ret) > 1 else 0.0
                )
                planner_returns_all.extend(s_ret)
                planner_actions += s_acts
                visit_total = s_visits if visit_total is None else visit_total + s_visits
                episodes_per_regime[g] = len(s_ret)
                env_steps_total += s_steps
                self._eval_episode_returns[g] = list(s_ret)

        # Module-2 ablation "argmax" deploy variant (A2): the SAME planner, but
        # leaf values use the single MAP-regime value head
        # (set_value_deploy("argmax")) instead of the Bayes average over the
        # belief posterior (A1, above). Subjective model only; a cheap extra
        # pass on the same checkpoint (no retraining) so the deploy-mode
        # ablation reads off one eval. Same RNG seed as the soft-planner pass so
        # env conditions match. A3 (no MCTS) = the prior pass above.
        planner_map_per_regime: dict[int, float] = {}
        planner_map_returns_all: list[float] = []
        if hasattr(model, "set_value_deploy"):
            np_random_map = np.random.RandomState(12345)
            try:
                model.set_value_deploy("argmax")
                with torch.no_grad():
                    for g in regime_grid:
                        g = int(g)
                        m_ret, _m_acts, _m_visits, _m_steps, _m_stats = (
                            self._rollout_planner(
                                env_fn, g, n_planner, device, np_random_map))
                        planner_map_per_regime[g] = (
                            float(np.mean(m_ret)) if m_ret else 0.0)
                        planner_map_returns_all.extend(m_ret)
            finally:
                model.set_value_deploy("bayes")  # restore trained default
        planner_map_mean = (float(np.mean(planner_map_returns_all))
                            if planner_map_returns_all else 0.0)

        regime_accuracy = (float(belief_hits) / belief_n) if belief_n else None
        prior_mean = float(np.mean(prior_returns_all)) if prior_returns_all else 0.0
        planner_mean = (float(np.mean(planner_returns_all))
                        if planner_returns_all else 0.0)
        zs_seen, zs_unseen = split_seen_unseen_regimes(self.cfg, planner_per_regime)
        return_sem = (
            float(np.std(planner_returns_all)
                  / max(np.sqrt(len(planner_returns_all)), 1.0))
            if len(planner_returns_all) > 1 else 0.0
        )

        # Diagnostics the EvalReport schema has no field for; scripts/train.py
        # writes them next to eval_report.json. An all-in-one-action histogram
        # here is the signature of a collapsed policy.
        self._eval_diagnostics = {
            "schema_version": "evaldiag-v2",
            "episodes_prior": n_prior,
            "episodes_planner": n_planner,
            "prior_action_histogram": prior_actions.tolist(),
            "prior_action_fractions":
                (prior_actions / max(prior_actions.sum(), 1)).round(6).tolist(),
            "planner_action_histogram": planner_actions.tolist(),
            "planner_action_fractions":
                (planner_actions / max(planner_actions.sum(), 1)).round(6).tolist(),
            "planner_root_visit_fractions": (
                (visit_total / max(visit_total.sum(), 1.0)).round(6).tolist()
                if visit_total is not None else None
            ),
            "return_per_regime_prior": dict(prior_per_regime),
            "return_per_regime_planner": dict(planner_per_regime),
            # deploy-mode ablation points on this checkpoint (evaldiag-v2):
            #   A1 Bayes-avg (ours)   = return_per_regime_planner / planner_mean
            #   A2 argmax (MAP head)  = return_per_regime_planner_map / *_map_mean
            #   A3 no-MCTS (prior)    = return_per_regime_prior / prior_mean
            "return_per_regime_planner_map": dict(planner_map_per_regime),
            "planner_map_return_mean": planner_map_mean,
            "planner_prior_return_gap": planner_mean - prior_mean,
            "prior_logit_margin": float(np.mean(margins)) if margins else 0.0,
            "planner_visit_entropy": float(np.mean(visit_ents)) if visit_ents else 0.0,
            # evaldiag-v2 (2026-07-20): root-cover health. coverage == 1.0 iff
            # every action is present for every agent at the root, which is the
            # precondition for the search being able to correct the prior at
            # all. Under the old prior-sampling path a collapsed prior gives
            # child_count 1 and coverage 1/A.
            "planner_root_child_count_mean": (
                float(np.mean(root_child_counts)) if root_child_counts else 0.0),
            "planner_root_action_coverage": (
                float(np.mean(root_coverages)) if root_coverages else 0.0),
            "planner_root_policy_mass_covered": (
                float(np.mean(root_policy_mass)) if root_policy_mass else 0.0),
            "regime_accuracy": regime_accuracy,
        }

        return EvalReport(
            variant="mazero_mixed",
            seed=int(seed),
            config_hash=str(config_hash) if config_hash else "0" * 40,
            eval_mode="dual",
            eval_planner_mode=self.cfg.eval.eval_planner_mode,
            return_mean=planner_mean,
            return_sem=return_sem,
            return_zero_shot_seen=zs_seen,
            return_zero_shot_unseen=zs_unseen,
            return_zero_shot_gap=zs_seen - zs_unseen,
            return_per_regime=MappingProxyType(planner_per_regime),
            return_per_regime_sem=MappingProxyType(planner_per_regime_sem),
            episodes_per_regime=MappingProxyType(episodes_per_regime),
            planner_prior_return_gap=planner_mean - prior_mean,
            direct_inference_return_mean=prior_mean,
            planner_full_return_mean=planner_mean,
            walltime_seconds=float(time.time() - t0),
            env_steps_evaluated=int(env_steps_total),
            episodes_total=int(len(planner_returns_all)),
            info_gating_strict=True,
            set_context_subjective_oracle_leak=False,
            regime_accuracy=regime_accuracy,
        )

    # ---------------------------------------------------------------- ckpt
    def save_checkpoint(self, path: Union[Path, str]) -> None:
        import torch

        if self._model is None:
            raise RuntimeError(
                f"save_checkpoint({path}) with no model — train() or "
                "load_checkpoint() must run first. This used to no-op silently, "
                "producing a run directory with no ckpt.pt and no error."
            )
        torch.save({"model_state_dict": self._model.state_dict()}, str(path))

    def load_checkpoint(self, path: Union[Path, str]) -> None:
        """Load weights into a model built for the CURRENT ``self._ablation``.

        Set ``_ablation`` before calling. ``point_estimate_leaf`` changes no
        module construction — only a branch in the leaf-value path — so its
        state dict loads cleanly into an unablated model and then evaluates
        with the wrong leaf rule, silently.
        """
        import torch

        model = self._lazy_model()
        ckpt = torch.load(str(path), map_location=self._device_of(model),
                          weights_only=False)
        state = ckpt["model_state_dict"]
        stale = [k for k in state if k.startswith("dynamics_network.fc_dynamic")
                 and k not in model.state_dict()]
        if stale:
            raise RuntimeError(
                f"{path} predates the 2026-07-19 dynamics fix: it carries "
                f"{sorted(stale)}, the output LayerNorm that made the transition "
                "action-blind. Checkpoints from before that change cannot be "
                "loaded or compared against ones after it — their world models "
                "have different architectures. Re-train, or read the archived "
                "eval_diagnostics_reeval.json for the old numbers."
            )
        model.load_state_dict(state)
        model.eval()
        self._weights_source = str(path)

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
        Body shared with the periodic in-training fidelity probe
        (core/train.py::train_sync_serial) via core.test.predict_rewards_from_model.
        """
        _ensure_fork_on_path()
        from core.test import predict_rewards_from_model

        model = self._lazy_model()
        device = self._device_of(model)
        return predict_rewards_from_model(model, episode, device)
