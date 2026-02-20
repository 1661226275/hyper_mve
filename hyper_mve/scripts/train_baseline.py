"""
Exp1 Baseline Training Script for Hyper-MuZero.

Flow:
    1. Warmup: collect episodes with random policy (no planner)
    2. Training loop:
        a. Collect episodes with current policy + MVE planner + epsilon-greedy
        b. Store in episode buffer
        c. Train MuZero unroll steps
        d. Decay epsilon
        e. Periodically evaluate
    3. Evaluation: fixed Rule, greedy policy, no exploration

Usage:
    cd D:/RL/hyper_mve
    python scripts/train_baseline.py
"""
import sys
import os
import copy
import time
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BaseConfig
from envs.make_env import make_ns_env
from models.baseline_model import BaselineModel
from training.episode_buffer import EpisodeReplayBuffer
from training.muzero_trainer import MuZeroTrainer
from training.worker import Worker
from models.representation_net import Projector
from utils.utils import get_device, get_active_agents
from utils.evaluator import run_eval_suite, log_results, print_summary, make_eval_result


def evaluate_baseline(env, rule, num_episodes, model, cfg, device):
    """
    Evaluate Baseline model with greedy policy (no exploration, no planner).

    Args:
        env:          NonStationaryMultiAgentEnv (shared eval env)
        rule:         float, forced rule value for this eval batch
        num_episodes: number of evaluation episodes
        model:        BaselineModel
        cfg:          config
        device:       torch device

    Returns:
        EvalResult dict
    """
    model.eval()

    all_rewards = []
    all_lengths = []

    for ep in range(num_episodes):
        obs_n, actual_rule = env.reset(options={'rule': rule})

        total_rewards = np.zeros(cfg.num_agents)

        for t in range(cfg.episode_limit):
            joint_obs = np.concatenate(obs_n)
            joint_obs_t = torch.tensor(joint_obs, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                s = model.encode(joint_obs_t)
                actions = []
                for i in range(cfg.num_agents):
                    id_i = torch.tensor([i], dtype=torch.long, device=device)
                    id_emb_i = model.get_id_emb(id_i)
                    logits_i, _ = model.predict(s, id_emb_i)
                    a_i = torch.argmax(logits_i, dim=-1).item()
                    actions.append(a_i)

            obs_n, reward_n, done_n, info_n = env.step(copy.deepcopy(actions))
            total_rewards += np.array(reward_n)

            if any(done_n):
                break

        all_rewards.append(total_rewards)
        all_lengths.append(t + 1)

    model.train()
    return make_eval_result(all_rewards, all_lengths)


def get_epsilon(step, cfg):
    """Linear epsilon decay."""
    frac = min(step / cfg.epsilon_decay_steps, 1.0)
    return cfg.epsilon_init + frac * (cfg.epsilon_min - cfg.epsilon_init)


def main():
    print("=" * 70)
    print("Hyper-MuZero Exp1: Baseline Training")
    print("=" * 70)

    # ── Config ──────────────────────────────────────────────────
    cfg = BaseConfig()
    device = get_device(cfg.device)
    print(f"Device: {device}")

    # ── Environment ─────────────────────────────────────────────
    env = make_ns_env(discrete=True)
    print(f"Environment: NonStationaryTag (discrete), {cfg.num_agents} agents, {cfg.num_actions} actions")

    # ── Model ───────────────────────────────────────────────────
    model = BaselineModel(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ── Projector (v4.0: shared for consistency loss) ────────────
    projector = Projector(latent_dim=cfg.latent_dim, proj_dim=cfg.latent_dim).to(device)
    proj_params = sum(p.numel() for p in projector.parameters())
    print(f"Projector parameters: {proj_params:,}")

    # ── Buffer & Trainer ────────────────────────────────────────
    buffer = EpisodeReplayBuffer(cfg, device)
    trainer = MuZeroTrainer(model, cfg, device, projector=projector)
    worker = Worker(env, model, cfg, device)

    # ── TensorBoard (optional) ──────────────────────────────────
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join(cfg.log_dir, 'baseline'))
        use_tb = True
        print("TensorBoard logging enabled")
    except ImportError:
        writer = None
        use_tb = False
        print("TensorBoard not available, using console logging only")

    # ── Warmup: collect initial episodes with random policy ─────
    warmup_target = getattr(cfg, 'min_buffer_size', cfg.batch_size)
    print(f"\n[Warmup] Collecting until buffer >= {warmup_target} valid episodes...")
    t_start = time.time()

    model.eval()
    collected = 0
    while len(buffer) < warmup_target:
        ep = worker.collect_episode(epsilon=1.0, use_planner=False)
        buffer.store_episode(ep)
        collected += 1
        if collected % 100 == 0:
            print(f"  Attempted {collected} episodes, buffer: {len(buffer)}/{warmup_target}"
                  f" (accept rate: {len(buffer)/collected:.1%})")

    t_warmup = time.time() - t_start
    print(f"  Warmup done: {collected} attempted, {len(buffer)} stored, {t_warmup:.1f}s")
    model.train()

    # ── Training Loop ───────────────────────────────────────────
    total_env_steps = 0
    total_train_steps = 0
    episodes_per_iter = getattr(cfg, 'episodes_per_iter', 32)
    train_steps_per_iter = getattr(cfg, 'train_steps_per_iter', 8)
    max_iterations = cfg.max_train_steps // train_steps_per_iter

    print(f"\n[Training] Starting main loop...")
    print(f"  episodes/iter: {episodes_per_iter}, train_steps/iter: {train_steps_per_iter}")
    print(f"  max iterations: {max_iterations}")
    print(f"  evaluate every: {cfg.evaluate_freq} train steps")
    print()

    t_train_start = time.time()

    for iteration in range(max_iterations):
        # ── Collect episodes ────────────────────────────────────
        epsilon = get_epsilon(total_train_steps, cfg)
        model.eval()

        # Use planner after some initial training
        use_planner = total_train_steps >= 500

        for _ in range(episodes_per_iter):
            ep = worker.collect_episode(epsilon=epsilon, use_planner=use_planner)
            buffer.store_episode(ep)
            total_env_steps += ep.length

        model.train()

        # ── Train ───────────────────────────────────────────────
        if not buffer.ready():
            continue

        prev_phase_name = None
        for _ in range(train_steps_per_iter):
            # Phased perspective sampling
            active_agents, phase_name = get_active_agents(total_train_steps, cfg)
            losses = trainer.train_step(buffer, active_agents=active_agents)
            total_train_steps += 1

            # Log phase transitions
            if phase_name != prev_phase_name and prev_phase_name is not None:
                print(f"--- Phase Switch @ step {total_train_steps}: "
                      f"Now training [{phase_name}] ---")
            prev_phase_name = phase_name

            # Logging
            if use_tb and total_train_steps % 100 == 0:
                for key, val in losses.items():
                    writer.add_scalar(f'train/{key}', val, total_train_steps)
                writer.add_scalar('train/epsilon', epsilon, total_train_steps)
                writer.add_scalar('train/buffer_size', len(buffer), total_train_steps)
                writer.add_scalar('train/active_phase',
                                  0 if phase_name in ('All', 'Warmup') else
                                  1 if phase_name == 'Hunters' else 2,
                                  total_train_steps)

        # ── Console logging ─────────────────────────────────────
        if iteration % 50 == 0:
            elapsed = time.time() - t_train_start
            print(f"[Iter {iteration:5d}] steps={total_train_steps:6d} "
                  f"eps={epsilon:.3f} buf={len(buffer):5d} "
                  f"phase={phase_name:>7s} "
                  f"loss={losses['loss_total']:.4f} "
                  f"l_pol={losses['l_pol']:.4f} "
                  f"l_val={losses['l_val']:.4f} "
                  f"l_rew={losses['l_rew']:.4f} "
                  f"l_con={losses['l_con']:.4f} "
                  f"({elapsed:.0f}s)")

        # ── Evaluation ──────────────────────────────────────────
        if iteration % cfg.evaluate_freq == 0 and total_train_steps > 0:
            eval_env = make_ns_env(discrete=True)
            eval_results = run_eval_suite(
                evaluate_baseline, eval_env,
                num_episodes=cfg.evaluate_episodes,
                model=model, cfg=cfg, device=device,
            )
            eval_env.close()

            print_summary(eval_results, step=total_train_steps)
            if use_tb:
                log_results(writer, eval_results, total_train_steps, prefix='eval')

            # Save checkpoint
            os.makedirs(cfg.save_dir, exist_ok=True)
            ckpt_path = os.path.join(cfg.save_dir, f'baseline_step{total_train_steps}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'projector_state_dict': projector.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'train_steps': total_train_steps,
                'eval_results': eval_results,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}\n")

    # ── Final ───────────────────────────────────────────────────
    total_time = time.time() - t_train_start
    print(f"\nTraining complete! Total time: {total_time:.0f}s")
    print(f"Total train steps: {total_train_steps}, env steps: {total_env_steps}")

    if use_tb:
        writer.close()

    env.close()


if __name__ == '__main__':
    main()
