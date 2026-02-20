"""
Exp3 v2: Infer-HyperMuZero with ChunkedHMLP Training Script (Phase 5).

Same as train_infer.py but uses InferHyperMuZeroModelV2
with ChunkedDualHyperNetwork (hypnettorch) for parameter efficiency.

Usage:
    cd D:/RL/hyper_mve
    python scripts/train_infer_v2.py
"""
import sys
import os
import copy
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BaseConfig
from envs.make_env import make_ns_env
from models_advanced.infer_v2 import InferHyperMuZeroModelV2
from training.episode_buffer import EpisodeReplayBuffer
from training.muzero_trainer import MuZeroTrainer
from training.worker import Worker
from models.representation_net import Projector
from utils.utils import get_device, actions_to_one_hot, get_active_agents
from utils.evaluator import run_eval_suite, log_results, print_summary, make_eval_result


def evaluate_infer_v2(env, rule, num_episodes, model, cfg, device):
    """
    Evaluate InferHyperMuZero v2 — Rule is NOT provided to the model.
    The model accumulates history within the episode to infer context.
    Rule is forced via env.reset(options={'rule': rule}) for consistent benchmarking.
    """
    model.eval()

    all_rewards = []
    all_lengths = []

    for ep in range(num_episodes):
        obs_n, actual_rule = env.reset(options={'rule': rule})

        total_rewards = np.zeros(cfg.num_agents)

        # History accumulator for GRU
        hist_obs_list = []
        hist_act_list = []
        hist_rew_list = []

        for t in range(cfg.episode_limit):
            joint_obs = np.concatenate(obs_n)
            joint_obs_t = torch.tensor(joint_obs, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                s = model.encode(joint_obs_t)

                if len(hist_obs_list) == 0:
                    # No history yet — use default embedding
                    actions = []
                    for i in range(cfg.num_agents):
                        id_i = torch.tensor([i], dtype=torch.long, device=device)
                        model.set_context_default(id_i, batch_size=1)
                        logits_i, _ = model.predict(s)
                        a_i = torch.argmax(logits_i, dim=-1).item()
                        actions.append(a_i)
                else:
                    # Build history tensors from accumulated data
                    W = min(len(hist_obs_list), cfg.trajectory_window)
                    h_obs = torch.tensor(
                        np.array(hist_obs_list[-W:]), dtype=torch.float32
                    ).unsqueeze(0).to(device)
                    h_act = torch.tensor(
                        np.array(hist_act_list[-W:]), dtype=torch.float32
                    ).unsqueeze(0).to(device)
                    h_rew = torch.tensor(
                        np.array(hist_rew_list[-W:]), dtype=torch.float32
                    ).unsqueeze(0).unsqueeze(-1).to(device)

                    actions = []
                    for i in range(cfg.num_agents):
                        id_i = torch.tensor([i], dtype=torch.long, device=device)
                        model.set_context_from_history(h_obs, h_act, h_rew, id_i)
                        logits_i, _ = model.predict(s)
                        a_i = torch.argmax(logits_i, dim=-1).item()
                        actions.append(a_i)

            obs_n, reward_n, done_n, info_n = env.step(copy.deepcopy(actions))
            total_rewards += np.array(reward_n)

            # Accumulate history
            hist_obs_list.append(joint_obs)
            act_onehot = np.zeros(cfg.num_agents * cfg.num_actions, dtype=np.float32)
            for i, a in enumerate(actions):
                act_onehot[i * cfg.num_actions + a] = 1.0
            hist_act_list.append(act_onehot)
            hist_rew_list.append(np.mean(reward_n))

            if any(done_n):
                break

        all_rewards.append(total_rewards)
        all_lengths.append(t + 1)

    model.train()
    return make_eval_result(all_rewards, all_lengths)


def get_epsilon(step, cfg):
    frac = min(step / cfg.epsilon_decay_steps, 1.0)
    return cfg.epsilon_init + frac * (cfg.epsilon_min - cfg.epsilon_init)


def main():
    print("=" * 70)
    print("Hyper-MuZero Exp3 v2: Infer-HyperMuZero + ChunkedHMLP (Phase 5)")
    print("=" * 70)

    cfg = BaseConfig()
    device = get_device(cfg.device)
    print(f"Device: {device}")

    env = make_ns_env(discrete=True)
    print(f"Environment: NonStationaryTag (discrete)")

    model = InferHyperMuZeroModelV2(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal model parameters: {total_params:,}")

    repr_params = sum(p.numel() for p in model.repr_net.parameters())
    ctx_params = sum(p.numel() for p in model.context_encoder.parameters())
    hyper_params = sum(p.numel() for p in model.hyper_net.parameters())
    print(f"  RepresentationNet:        {repr_params:,}")
    print(f"  InferContextEncoder (GRU): {ctx_params:,}")
    print(f"  ChunkedDualHyperNetwork:  {hyper_params:,}")
    print(f"  GRU input dim: {model.context_encoder.gru_inferrer.input_dim}")
    print(f"  GRU hidden: {cfg.gru_hidden_size}")
    print(f"  Trajectory window: {cfg.trajectory_window}")
    print(f"  Chunk config: alpha={cfg.chunk_alpha}, emb_size={cfg.chunk_emb_size}, "
          f"budget={cfg.chunk_budget_factor}, hyperfan={cfg.chunk_hyperfan_init}")

    # Projector (v4.0: shared for consistency loss)
    projector = Projector(latent_dim=cfg.latent_dim, proj_dim=cfg.latent_dim).to(device)
    proj_params = sum(p.numel() for p in projector.parameters())
    print(f"  Projector:                {proj_params:,}")

    buffer = EpisodeReplayBuffer(cfg, device)
    trainer = MuZeroTrainer(model, cfg, device, projector=projector)
    worker = Worker(env, model, cfg, device)

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join(cfg.log_dir, 'infer_v2'))
        use_tb = True
        print("TensorBoard logging enabled")
    except ImportError:
        writer = None
        use_tb = False

    # Warmup: collect until buffer >= min_buffer_size for initial sample diversity
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
    print(f"  Warmup done: {collected} attempted, {len(buffer)} stored, {time.time()-t_start:.1f}s")
    model.train()

    # Training
    total_env_steps = 0
    total_train_steps = 0
    episodes_per_iter = getattr(cfg, 'episodes_per_iter', 32)
    train_steps_per_iter = getattr(cfg, 'train_steps_per_iter', 8)
    max_iterations = cfg.max_train_steps // train_steps_per_iter

    print(f"\n[Training] Starting...")
    t_train_start = time.time()

    for iteration in range(max_iterations):
        epsilon = get_epsilon(total_train_steps, cfg)
        model.eval()
        use_planner = total_train_steps >= 500

        for _ in range(episodes_per_iter):
            ep = worker.collect_episode(epsilon=epsilon, use_planner=use_planner)
            buffer.store_episode(ep)
            total_env_steps += ep.length
        model.train()

        if not buffer.ready():
            continue

        prev_phase_name = None
        for _ in range(train_steps_per_iter):
            active_agents, phase_name = get_active_agents(total_train_steps, cfg)
            losses = trainer.train_step_infer(buffer, active_agents=active_agents)
            total_train_steps += 1

            if phase_name != prev_phase_name and prev_phase_name is not None:
                print(f"--- Phase Switch @ step {total_train_steps}: "
                      f"Now training [{phase_name}] ---")
            prev_phase_name = phase_name

            if use_tb and total_train_steps % 100 == 0:
                for key, val in losses.items():
                    writer.add_scalar(f'train/{key}', val, total_train_steps)
                writer.add_scalar('train/epsilon', epsilon, total_train_steps)
                writer.add_scalar('train/active_phase',
                                  0 if phase_name in ('All', 'Warmup') else
                                  1 if phase_name == 'Hunters' else 2,
                                  total_train_steps)

        if iteration % 50 == 0:
            elapsed = time.time() - t_train_start
            ctx_str = f"l_ctx={losses.get('loss_ctx', 0):.4f} " if 'loss_ctx' in losses else ""
            print(f"[Iter {iteration:5d}] steps={total_train_steps:6d} "
                  f"eps={epsilon:.3f} buf={len(buffer):5d} "
                  f"phase={phase_name:>7s} "
                  f"loss={losses['loss_total']:.4f} "
                  f"l_pol={losses['loss_policy']:.4f} "
                  f"l_pg={losses.get('loss_pg', 0):.4f} "
                  f"ce_g={losses.get('ce_gate', 0):.3f} "
                  f"l_val={losses['loss_value']:.4f} "
                  f"l_rew={losses['loss_reward']:.4f} "
                  f"l_con={losses['loss_consist']:.4f} "
                  f"{ctx_str}({elapsed:.0f}s)")

        if iteration % cfg.evaluate_freq == 0 and total_train_steps > 0:
            eval_env = make_ns_env(discrete=True)
            eval_results = run_eval_suite(
                evaluate_infer_v2, eval_env,
                num_episodes=cfg.evaluate_episodes,
                model=model, cfg=cfg, device=device,
            )
            eval_env.close()

            print_summary(eval_results, step=total_train_steps)
            if use_tb:
                log_results(writer, eval_results, total_train_steps, prefix='eval')

            os.makedirs(cfg.save_dir, exist_ok=True)
            ckpt_path = os.path.join(cfg.save_dir, f'infer_v2_step{total_train_steps}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'projector_state_dict': projector.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'train_steps': total_train_steps,
                'eval_results': eval_results,
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}\n")

    total_time = time.time() - t_train_start
    print(f"\nTraining complete! {total_time:.0f}s, {total_train_steps} steps")

    if use_tb:
        writer.close()
    env.close()


if __name__ == '__main__':
    main()