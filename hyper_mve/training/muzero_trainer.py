"""
MuZero-style Unrolled Trainer for Hyper-MuZero (v4.5).

Training loop:
    1. Sample (B, K+1) sequences from episode buffer
    2. Random perspective sampling: pick agent_id per sample
    3. Initial encoding: s_0 = RepNet(obs_0)
    4. K-step unroll with gradient half-life
    5. Compute losses per step: policy (CE+PG+entropy), value, reward, consistency
    6. Loss averaged over K steps (v4.0)
    7. Unified backward + optimizer step
    8. [v4.4] EMA update target network + CosineAnnealingLR step

Loss definitions (per step k) — v4.5:
    L_policy  = ce_gate * w_ce * CE(p_k, pi_mve) + w_pg * PG(p_k, adv) - w_ent * H(p_k)
    L_value   = MSE(v_k, h(z_{t+k}))      [z = N-step return via TARGET model, in scaled space]
    L_reward  = MSE(r_k, h(reward_{t+k}))  [in scaled space]
    L_consist = -CosineSim(Proj(s_pred), sg(Proj(RepNet(o_{t+k+1}))))  [v4.0]

v4.5 changes:
    - Policy gradient auxiliary loss (breaks planner deadlock)
    - Entropy-gated CE loss (prevents CE from fighting PG when π_mve is uniform)
    - Entropy bonus (prevents premature convergence)

v4.4 changes:
    - Target Network (EMA) for stable bootstrap in N-step return
    - CosineAnnealingLR replacing MultiStepLR
    - Adam eps=1e-5 for update stability
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import actions_to_one_hot, scalar_transform, inverse_scalar_transform
from planning.mve_planner import _is_hyper_model
from models.gru_context_encoder import context_variance_loss
from models.representation_net import negative_cosine_similarity


class MuZeroTrainer:
    """
    MuZero unrolled training manager.

    Handles:
        - Batch preparation from episode buffer
        - K-step unrolled loss computation
        - N-step return calculation (via target network, v4.4)
        - Gradient half-life trick
        - Unified optimizer step
        - EMA target network update (v4.4)
    """

    def __init__(self, model, cfg, device, projector=None):
        """
        Args:
            model:     BaselineModel (or HyperMuZeroModel)
            cfg:       config
            device:    torch device
            projector: Projector module for consistency loss (v4.0, shared across exps)
        """
        self.model = model
        self.cfg = cfg
        self.device = device
        self.projector = projector  # v4.0: Projector for Cosine Sim consistency

        # [v4.4] Target Network: EMA copy of the online model
        self.target_model = copy.deepcopy(model)
        self.target_model.requires_grad_(False)

        # Collect all trainable parameters: model + projector
        params = list(model.parameters())
        if projector is not None:
            params += list(projector.parameters())
        self.optimizer = torch.optim.Adam(
            params, lr=cfg.lr, eps=getattr(cfg, 'adam_eps', 1e-5)
        )

        # [v4.4] CosineAnnealingLR replacing MultiStepLR
        lr_min = getattr(cfg, 'lr_min', 1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.max_train_steps,
            eta_min=lr_min,
        )
        self.train_step_count = 0

    def _soft_update_target(self):
        """[v4.4] EMA update: target = tau * target + (1-tau) * online."""
        tau = getattr(self.cfg, 'ema_tau', 0.99)
        for tp, op in zip(self.target_model.parameters(), self.model.parameters()):
            tp.data.mul_(tau).add_(op.data, alpha=1.0 - tau)

    def compute_n_step_return(self, rewards_i, obs_seq, agent_ids, start_idx):
        """
        Compute real-time N-step return for a specific agent perspective.

        z_t = Σ_{k=0}^{n-1} γ^k * r_{t+k,i} + γ^n * V_target(s_{t+n}, i)

        [v4.4] Uses TARGET model for V bootstrap to prevent value overestimation
        feedback loops. The target model is an EMA-smoothed copy that changes
        slowly, providing stable bootstrap anchors even during phase transitions.

        Args:
            rewards_i:  (B, K) agent i's rewards for the sampled sequence
            obs_seq:    (B, K+1, joint_obs_dim) observation sequence
            agent_ids:  (B,) int tensor of sampled agent perspectives
            start_idx:  int, the step index within the K-step window to compute return for

        Returns:
            z: (B,) N-step return value (in ORIGINAL scale, NOT scaled)
        """
        B = rewards_i.shape[0]
        n = self.cfg.n_step
        K = self.cfg.unroll_K
        gamma = self.cfg.gamma

        z = torch.zeros(B, device=self.device)

        # Sum discounted rewards
        for k in range(n):
            idx = start_idx + k
            if idx < K:
                z += (gamma ** k) * rewards_i[:, idx]
            # If idx >= K, no more rewards available, bootstrap from there

        # [v4.4] Bootstrap with TARGET model's V at t+n (EMA-smoothed)
        bootstrap_idx = min(start_idx + n, K)  # obs index for bootstrap
        with torch.no_grad():
            s_boot = self.target_model.encode(obs_seq[:, bootstrap_idx])
            id_emb = self.target_model.get_id_emb(agent_ids)
            _, v_boot_scaled = self.target_model.predict(s_boot, id_emb)  # (B, 1)
            v_boot = inverse_scalar_transform(v_boot_scaled).squeeze(-1)  # (B,)

        z += (gamma ** min(n, K - start_idx)) * v_boot

        return z  # (B,) in original scale

    def _sample_agent_ids(self, B, active_agents=None):
        """
        Sample agent IDs for perspective sampling.

        Args:
            B: batch size
            active_agents: list of int or None.
                If None, sample from all agents [0, num_agents).
                If list, sample only from these agents (e.g. [0,1,2] or [3]).

        Returns:
            agent_ids: (B,) int tensor
        """
        if active_agents is None:
            return torch.randint(0, self.cfg.num_agents, (B,), device=self.device)
        else:
            agent_source = torch.tensor(active_agents, dtype=torch.long, device=self.device)
            indices = torch.randint(0, len(active_agents), (B,), device=self.device)
            return agent_source[indices]

    def train_step(self, buffer, active_agents=None):
        """
        One MuZero training step.

        Args:
            buffer: EpisodeReplayBuffer (must be ready)
            active_agents: list of int or None.
                Controls which agent perspectives are sampled for training.
                None = all agents. [0,1,2] = hunters only. [3] = prey only.

        Returns:
            dict of loss components for logging
        """
        cfg = self.cfg
        model = self.model
        K = cfg.unroll_K
        B = cfg.batch_size

        # 1. Sample batch
        batch = buffer.sample_batch()
        obs_seq = batch['obs']          # (B, K+1, joint_obs_dim)
        actions_seq = batch['actions']  # (B, K, num_agents)
        rewards_seq = batch['rewards']  # (B, K, num_agents)
        policies_seq = batch['policies']  # (B, K, num_agents, num_actions)
        # rules = batch['rules']        # (B,) — not used in Baseline
        # dones = batch['dones']        # (B, K) — could mask, but for now ignore

        # Check if HyperMuZero model
        is_hyper = _is_hyper_model(model)
        rules = batch['rules']  # (B,)

        # 2. Perspective sampling (optionally restricted to active_agents)
        agent_ids = self._sample_agent_ids(B, active_agents)
        id_emb = model.get_id_emb(agent_ids)  # (B, id_emb_dim)

        # For HyperMuZero: set context once (Rule + sampled agent_id)
        # This generates θ_state, θ_reward, θ_pred for the entire unroll
        if is_hyper:
            model.set_context(rules, agent_ids)
            # [v4.4] Set target model context for bootstrap
            with torch.no_grad():
                self.target_model.set_context(rules, agent_ids)

        # Extract per-agent rewards for the sampled perspective
        # rewards_seq: (B, K, N) -> rewards_i: (B, K)
        rewards_i = rewards_seq[torch.arange(B, device=self.device).unsqueeze(1),
                                torch.arange(K, device=self.device).unsqueeze(0),
                                agent_ids.unsqueeze(1).expand(B, K)]

        # 3. Initial encoding
        s = model.encode(obs_seq[:, 0])  # (B, latent_dim)

        # [v4.5] Pre-compute target model id_emb for PG advantage
        with torch.no_grad():
            id_emb_target = self.target_model.get_id_emb(agent_ids)

        # [v4.5] Policy loss hyperparams
        w_ce = getattr(cfg, 'w_ce', 1.0)
        w_pg = getattr(cfg, 'w_pg', 1.0)
        w_ent = getattr(cfg, 'w_entropy', 0.01)
        max_ent = math.log(cfg.num_actions)
        batch_idx = torch.arange(B, device=self.device)

        # 4. K-step unroll
        total_loss = torch.tensor(0.0, device=self.device)
        loss_policy_sum = 0.0
        loss_ce_sum = 0.0
        loss_pg_sum = 0.0
        entropy_sum = 0.0
        ce_gate_sum = 0.0
        loss_value_sum = 0.0
        loss_reward_sum = 0.0
        loss_consist_sum = 0.0

        for k in range(K):
            # 4.1 Prediction for current state (subjective)
            p_k, v_k = model.predict(s, id_emb)  # (B, A), (B, 1)

            # 4.2 Dynamics: state transition + reward
            # One-hot encode actions at step k
            action_k = actions_seq[:, k]  # (B, N) int
            action_onehot_k = actions_to_one_hot(action_k, cfg.num_actions)  # (B, N*A)

            s_next = model.transition(s, action_onehot_k)  # (B, latent_dim)
            r_k = model.predict_reward(s, action_onehot_k, id_emb)  # (B, 1) r=R(s,a)

            # 4.3 Targets
            # Policy target: search policy for this agent at step k
            # policies_seq: (B, K, N, A)
            target_pi = policies_seq[batch_idx, k, agent_ids]  # (B, A)

            # Reward target: real reward for this agent at step k (in scaled space)
            target_r = scalar_transform(rewards_i[:, k].unsqueeze(-1))  # (B, 1)

            # Value target: N-step return via TARGET model (v4.4, in scaled space)
            target_z_raw = self.compute_n_step_return(
                rewards_i, obs_seq, agent_ids, start_idx=k
            )  # (B,)
            target_z = scalar_transform(target_z_raw.unsqueeze(-1)).squeeze(-1)  # (B,)

            # 4.4 Consistency target — [v4.0] Projection + Cosine Similarity
            proj_pred = self.projector(s_next)  # (B, proj_dim)
            with torch.no_grad():
                s_target = model.encode(obs_seq[:, k + 1])  # (B, latent_dim)
                proj_target = self.projector(s_target)  # (B, proj_dim)

            # 4.5 Compute losses
            log_p = F.log_softmax(p_k, dim=-1)  # (B, A)

            # ── [v4.5] CE loss with entropy gating ──
            loss_ce = -(target_pi * log_p).sum(dim=-1).mean()
            with torch.no_grad():
                pi_ent = -(target_pi * (target_pi + 1e-8).log()).sum(dim=-1).mean()
                ce_gate = (1.0 - pi_ent / max_ent).clamp(min=0.0)

            # ── [v4.5] PG loss: advantage-weighted log-probability ──
            a_taken = actions_seq[batch_idx, k, agent_ids]  # (B,) actual action
            with torch.no_grad():
                s_curr_real = self.target_model.encode(obs_seq[:, k])
                s_next_real = self.target_model.encode(obs_seq[:, k + 1])
                _, v_curr_sc = self.target_model.predict(s_curr_real, id_emb_target)
                _, v_next_sc = self.target_model.predict(s_next_real, id_emb_target)
                v_curr_real = inverse_scalar_transform(v_curr_sc).squeeze(-1)
                v_next_real = inverse_scalar_transform(v_next_sc).squeeze(-1)
                adv = rewards_i[:, k] + cfg.gamma * v_next_real - v_curr_real
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # normalize

            log_prob_taken = log_p.gather(1, a_taken.unsqueeze(1).long()).squeeze(1)
            loss_pg = -(log_prob_taken * adv).mean()

            # ── [v4.5] Entropy bonus ──
            probs = F.softmax(p_k, dim=-1)
            entropy = -(probs * log_p).sum(dim=-1).mean()

            # ── [v4.5] Combined policy loss ──
            loss_policy = w_ce * ce_gate * loss_ce + w_pg * loss_pg - w_ent * entropy

            # Value loss: MSE in scaled space
            loss_value = F.mse_loss(v_k.squeeze(-1), target_z)

            # Reward loss: MSE in scaled space
            loss_reward = F.mse_loss(r_k, target_r)

            # Consistency loss: negative cosine similarity in projection space [v4.0]
            loss_consist = negative_cosine_similarity(proj_pred, proj_target)

            # Accumulate
            step_loss = (cfg.w_policy * loss_policy +
                         cfg.w_value * loss_value +
                         cfg.w_reward * loss_reward +
                         cfg.w_consist * loss_consist)
            total_loss = total_loss + step_loss

            loss_policy_sum += loss_policy.item()
            loss_ce_sum += loss_ce.item()
            loss_pg_sum += loss_pg.item()
            entropy_sum += entropy.item()
            ce_gate_sum += ce_gate.item()
            loss_value_sum += loss_value.item()
            loss_reward_sum += loss_reward.item()
            loss_consist_sum += loss_consist.item()

            # 4.6 Gradient half-life: prevent K-step gradient explosion
            s = 0.5 * s_next + 0.5 * s_next.detach()

        # 5. [v4.0] Loss averaging over K steps
        total_loss = total_loss / K

        # 6. Unified backward
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if self.projector is not None:
            nn.utils.clip_grad_norm_(self.projector.parameters(), cfg.grad_clip)
        self.optimizer.step()

        # 7. [v4.4] EMA update target network + LR schedule step
        self._soft_update_target()
        self.scheduler.step()

        self.train_step_count += 1

        return {
            'loss_total': total_loss.item(),
            'loss_policy': loss_policy_sum / K,
            'loss_ce': loss_ce_sum / K,
            'loss_pg': loss_pg_sum / K,
            'entropy': entropy_sum / K,
            'ce_gate': ce_gate_sum / K,
            'loss_value': loss_value_sum / K,
            'loss_reward': loss_reward_sum / K,
            'loss_consist': loss_consist_sum / K,
            'lr': self.scheduler.get_last_lr()[0],
        }

    def train_step_infer(self, buffer, active_agents=None):
        """
        One MuZero training step for Infer-HyperMuZero (Exp3).

        Same as train_step but:
            1. Samples batch WITH history window
            2. Uses GRU to infer rule_emb from history
            3. Adds context regularization loss
            4. [v4.4] Target model uses online GRU's rule_emb + target hypernetwork

        Args:
            buffer: EpisodeReplayBuffer (must be ready)
            active_agents: list of int or None.
                Controls which agent perspectives are sampled for training.
                None = all agents. [0,1,2] = hunters only. [3] = prey only.
        Returns:
            dict of loss components for logging
        """
        cfg = self.cfg
        model = self.model
        K = cfg.unroll_K
        B = cfg.batch_size

        # 1. Sample batch with history
        batch = buffer.sample_batch_with_history(
            trajectory_window=cfg.trajectory_window
        )
        obs_seq = batch['obs']
        actions_seq = batch['actions']
        rewards_seq = batch['rewards']
        policies_seq = batch['policies']

        hist_obs = batch['history_obs']               # (B, W, obs_dim)
        hist_actions = batch['history_actions_onehot'] # (B, W, act_dim)
        hist_rewards = batch['history_rewards']        # (B, W, 1)
        hist_mask = batch['history_mask']              # (B, W)

        # 2. Perspective sampling (optionally restricted to active_agents)
        agent_ids = self._sample_agent_ids(B, active_agents)

        # 3. Infer rule_emb from history via GRU, then set context
        # This call is differentiable — gradients flow back through GRU
        model.set_context_from_history(
            hist_obs, hist_actions, hist_rewards, agent_ids, hist_mask
        )

        # Get the inferred rule_emb for regularization
        inferred_rule_emb = model.get_current_rule_emb()  # (B, rule_emb_dim)

        # [v4.4] Set target model context: online GRU's rule_emb + target hypernetwork
        # GRU inference should update fast (online), but V prediction should be stable (target)
        with torch.no_grad():
            self.target_model.set_context(inferred_rule_emb.detach(), agent_ids)

        # Extract per-agent rewards
        rewards_i = rewards_seq[torch.arange(B, device=self.device).unsqueeze(1),
                                torch.arange(K, device=self.device).unsqueeze(0),
                                agent_ids.unsqueeze(1).expand(B, K)]

        # 4. Initial encoding
        s = model.encode(obs_seq[:, 0])

        # [v4.5] Pre-compute target model id_emb for PG advantage
        with torch.no_grad():
            id_emb_target = self.target_model.get_id_emb(agent_ids)

        # [v4.5] Policy loss hyperparams
        w_ce = getattr(cfg, 'w_ce', 1.0)
        w_pg = getattr(cfg, 'w_pg', 1.0)
        w_ent = getattr(cfg, 'w_entropy', 0.01)
        max_ent = math.log(cfg.num_actions)
        batch_idx = torch.arange(B, device=self.device)

        # 5. K-step unroll (same structure as train_step)
        total_loss = torch.tensor(0.0, device=self.device)
        loss_policy_sum = 0.0
        loss_ce_sum = 0.0
        loss_pg_sum = 0.0
        entropy_sum = 0.0
        ce_gate_sum = 0.0
        loss_value_sum = 0.0
        loss_reward_sum = 0.0
        loss_consist_sum = 0.0

        id_emb = model.get_id_emb(agent_ids)

        for k in range(K):
            p_k, v_k = model.predict(s, id_emb)

            action_k = actions_seq[:, k]
            action_onehot_k = actions_to_one_hot(action_k, cfg.num_actions)

            s_next = model.transition(s, action_onehot_k)
            r_k = model.predict_reward(s, action_onehot_k, id_emb)  # r=R(s,a)

            target_pi = policies_seq[batch_idx, k, agent_ids]
            target_r = scalar_transform(rewards_i[:, k].unsqueeze(-1))
            target_z_raw = self.compute_n_step_return(
                rewards_i, obs_seq, agent_ids, start_idx=k
            )
            target_z = scalar_transform(target_z_raw.unsqueeze(-1)).squeeze(-1)

            # [v4.0] Consistency: Projection + Cosine Similarity
            proj_pred = self.projector(s_next)  # (B, proj_dim)
            with torch.no_grad():
                s_target = model.encode(obs_seq[:, k + 1])
                proj_target = self.projector(s_target)  # (B, proj_dim)

            log_p = F.log_softmax(p_k, dim=-1)

            # ── [v4.5] CE loss with entropy gating ──
            loss_ce = -(target_pi * log_p).sum(dim=-1).mean()
            with torch.no_grad():
                pi_ent = -(target_pi * (target_pi + 1e-8).log()).sum(dim=-1).mean()
                ce_gate = (1.0 - pi_ent / max_ent).clamp(min=0.0)

            # ── [v4.5] PG loss: advantage-weighted log-probability ──
            a_taken = actions_seq[batch_idx, k, agent_ids]  # (B,)
            with torch.no_grad():
                s_curr_real = self.target_model.encode(obs_seq[:, k])
                s_next_real = self.target_model.encode(obs_seq[:, k + 1])
                _, v_curr_sc = self.target_model.predict(s_curr_real, id_emb_target)
                _, v_next_sc = self.target_model.predict(s_next_real, id_emb_target)
                v_curr_real = inverse_scalar_transform(v_curr_sc).squeeze(-1)
                v_next_real = inverse_scalar_transform(v_next_sc).squeeze(-1)
                adv = rewards_i[:, k] + cfg.gamma * v_next_real - v_curr_real
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            log_prob_taken = log_p.gather(1, a_taken.unsqueeze(1).long()).squeeze(1)
            loss_pg = -(log_prob_taken * adv).mean()

            # ── [v4.5] Entropy bonus ──
            probs = F.softmax(p_k, dim=-1)
            entropy = -(probs * log_p).sum(dim=-1).mean()

            # ── [v4.5] Combined policy loss ──
            loss_policy = w_ce * ce_gate * loss_ce + w_pg * loss_pg - w_ent * entropy

            loss_value = F.mse_loss(v_k.squeeze(-1), target_z)
            loss_reward = F.mse_loss(r_k, target_r)
            loss_consist = negative_cosine_similarity(proj_pred, proj_target)

            step_loss = (cfg.w_policy * loss_policy +
                         cfg.w_value * loss_value +
                         cfg.w_reward * loss_reward +
                         cfg.w_consist * loss_consist)
            total_loss = total_loss + step_loss

            loss_policy_sum += loss_policy.item()
            loss_ce_sum += loss_ce.item()
            loss_pg_sum += loss_pg.item()
            entropy_sum += entropy.item()
            ce_gate_sum += ce_gate.item()
            loss_value_sum += loss_value.item()
            loss_reward_sum += loss_reward.item()
            loss_consist_sum += loss_consist.item()

            s = 0.5 * s_next + 0.5 * s_next.detach()

        # 6. [v4.0] Loss averaging over K steps
        total_loss = total_loss / K

        # 7. [v4.0] Context regularization: Hinge Variance Loss (replaces 1/var)
        target_std = getattr(cfg, 'target_context_std', 0.1)
        loss_ctx = context_variance_loss(inferred_rule_emb, target_std=target_std)
        total_loss = total_loss + cfg.w_context * loss_ctx

        # 8. Unified backward
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if self.projector is not None:
            nn.utils.clip_grad_norm_(self.projector.parameters(), cfg.grad_clip)
        self.optimizer.step()

        # 9. [v4.4] EMA update target network + LR schedule step
        self._soft_update_target()
        self.scheduler.step()

        self.train_step_count += 1

        return {
            'loss_total': total_loss.item(),
            'loss_policy': loss_policy_sum / K,
            'loss_ce': loss_ce_sum / K,
            'loss_pg': loss_pg_sum / K,
            'entropy': entropy_sum / K,
            'ce_gate': ce_gate_sum / K,
            'loss_value': loss_value_sum / K,
            'loss_reward': loss_reward_sum / K,
            'loss_consist': loss_consist_sum / K,
            'loss_ctx': loss_ctx.item(),
            'lr': self.scheduler.get_last_lr()[0],
        }
