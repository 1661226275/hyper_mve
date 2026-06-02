"""
MuZero-style Unrolled Trainer for Hyper-MuZero (v4.6).

Training loop:
    1. Sample (B, K+1) sequences from episode buffer
    2. Random perspective sampling: pick agent_id per sample
    3. Initial encoding: s_0 = RepNet(obs_0)
    4. K-step unroll with gradient half-life
    5. Compute losses per step: policy (CE), value, reward, consistency
    6. Loss averaged over K steps (v4.0)
    7. Unified backward + optimizer step
    8. [v4.4] EMA update target network + CosineAnnealingLR step

Loss definitions (per step k) — v4.6:
    L_policy  = CE(p_k, pi_mve)
    L_value   = MSE(v_k, h(z_{t+k}))      [z = N-step return via TARGET model, in scaled space]
    L_reward  = MSE(r_k, h(reward_{t+k}))  [in scaled space]
    L_consist = -CosineSim(Proj(s_pred), sg(Proj(RepNet(o_{t+k+1}))))  [v4.0]

v4.6 changes:
    - Removed PG auxiliary loss (off-policy bias causes l_pg runaway with replay buffer)
    - Removed entropy-gated CE (no longer needed without PG)
    - Planner CRN fix provides non-uniform π_mve, standard CE is sufficient

v4.4 changes:
    - Target Network (EMA) for stable bootstrap in N-step return
    - CosineAnnealingLR replacing MultiStepLR
    - Adam eps=1e-5 for update stability
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import actions_to_one_hot, scalar_transform, inverse_scalar_transform
from planning.mve_planner import _is_hyper_model
from models.gru_context_encoder import context_variance_loss
from models.hyper_network import reward_diversity_loss
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

        # [v4.7] Configurable LR schedule: 'cosine' | 'multistep' | 'warmup_cosine'
        self.scheduler = self._build_scheduler(cfg)
        self.train_step_count = 0

    def _build_scheduler(self, cfg):
        """
        [v4.7] Build LR scheduler based on config.

        Supported schedules:
            'cosine':        CosineAnnealingLR (v4.4 default)
            'multistep':     MultiStepLR with configurable milestones
            'warmup_cosine': Linear warmup (lr_min -> lr) then CosineAnnealing

        Returns:
            LR scheduler instance
        """
        lr_min = getattr(cfg, 'lr_min', 1e-5)
        schedule = getattr(cfg, 'lr_schedule', 'cosine')

        if schedule == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.max_train_steps, eta_min=lr_min,
            )
        elif schedule == 'multistep':
            milestones = getattr(cfg, 'lr_milestones', [10000, 40000, 80000])
            gamma = getattr(cfg, 'lr_gamma', 0.3)
            return torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=milestones, gamma=gamma,
            )
        elif schedule == 'warmup_cosine':
            warmup_steps = getattr(cfg, 'lr_warmup_steps', 2000)
            cosine_steps = cfg.max_train_steps - warmup_steps

            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=lr_min / cfg.lr,  # start at lr_min
                end_factor=1.0,                 # ramp up to lr
                total_iters=warmup_steps,
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cosine_steps, eta_min=lr_min,
            )
            return torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )
        else:
            raise ValueError(f"Unknown lr_schedule: {schedule}. "
                             f"Choose from 'cosine', 'multistep', 'warmup_cosine'.")

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

        # Check if HyperMuZero model
        is_hyper = _is_hyper_model(model)
        rules = batch['rules']  # (B,)

        # 2. Perspective sampling (optionally restricted to active_agents)
        agent_ids = self._sample_agent_ids(B, active_agents)
        id_emb = model.get_id_emb(agent_ids)  # (B, id_emb_dim)

        # For HyperMuZero: set context once (Rule + sampled agent_id)
        # This generates θ_state, θ_reward, θ_pred for the entire unroll
        if is_hyper:
            # v4 migration deferred to Pkg-05 (Q2 折中); see Pkg-04 spec 08 §3.2.
            # v4.7 sites: model.set_context(rules, agent_ids) [orig L239] +
            #             self.target_model.set_context(rules, agent_ids) [orig L242].
            # v4 replaces with set_context_objective(c_t) once + per-agent
            # set_context_subjective(k, cap, belief); same for target_model.
            raise NotImplementedError(
                "v4 trainer set_context migration deferred to Pkg-05 (spec 08 §3.2)."
            )

        # Extract per-agent rewards for the sampled perspective
        # rewards_seq: (B, K, N) -> rewards_i: (B, K)
        rewards_i = rewards_seq[torch.arange(B, device=self.device).unsqueeze(1),
                                torch.arange(K, device=self.device).unsqueeze(0),
                                agent_ids.unsqueeze(1).expand(B, K)]

        # 3. Initial encoding
        s = model.encode(obs_seq[:, 0])  # (B, latent_dim)

        batch_idx = torch.arange(B, device=self.device)

        # 4. K-step unroll
        total_loss = torch.tensor(0.0, device=self.device)
        loss_policy_sum = 0.0
        loss_value_sum = 0.0
        loss_reward_sum = 0.0
        loss_consist_sum = 0.0

        # [Diagnostic] Per-component weighted loss tensors (for gradient attribution)
        total_pol_t = torch.tensor(0.0, device=self.device)
        total_val_t = torch.tensor(0.0, device=self.device)
        total_rew_t = torch.tensor(0.0, device=self.device)
        total_con_t = torch.tensor(0.0, device=self.device)

        # [Monitoring] Per-agent loss accumulators (gradient domination detection)
        num_agents = cfg.num_agents
        per_agent_pol = {i: 0.0 for i in range(num_agents)}
        per_agent_val = {i: 0.0 for i in range(num_agents)}
        per_agent_rew = {i: 0.0 for i in range(num_agents)}

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

            # 4.5 Compute losses (per-sample first, then batch average)
            # [v4.6] Plain CE — CRN-fixed planner provides meaningful π_mve targets
            loss_pol_per = -(target_pi * F.log_softmax(p_k, dim=-1)).sum(dim=-1)  # (B,)
            loss_policy = loss_pol_per.mean()

            # Value loss: MSE in scaled space
            loss_val_per = F.mse_loss(v_k.squeeze(-1), target_z, reduction='none')  # (B,)
            loss_value = loss_val_per.mean()

            # Reward loss: MSE in scaled space
            loss_rew_per = F.mse_loss(r_k, target_r, reduction='none').squeeze(-1)  # (B,)
            loss_reward = loss_rew_per.mean()

            # Consistency loss: negative cosine similarity in projection space [v4.0]
            loss_consist = negative_cosine_similarity(proj_pred, proj_target)

            # [Monitoring] Per-agent loss accumulation (detached, logging only)
            with torch.no_grad():
                for aid in range(num_agents):
                    mask = (agent_ids == aid)
                    if mask.any():
                        per_agent_pol[aid] += loss_pol_per[mask].mean().item()
                        per_agent_val[aid] += loss_val_per[mask].mean().item()
                        per_agent_rew[aid] += loss_rew_per[mask].mean().item()

            # Accumulate
            step_loss = (cfg.w_policy * loss_policy +
                         cfg.w_value * loss_value +
                         cfg.w_reward * loss_reward +
                         cfg.w_consist * loss_consist)
            total_loss = total_loss + step_loss

            # [Diagnostic] Accumulate weighted per-component losses (with grad)
            total_pol_t = total_pol_t + cfg.w_policy * loss_policy
            total_val_t = total_val_t + cfg.w_value * loss_value
            total_rew_t = total_rew_t + cfg.w_reward * loss_reward
            total_con_t = total_con_t + cfg.w_consist * loss_consist

            loss_policy_sum += loss_policy.item()
            loss_value_sum += loss_value.item()
            loss_reward_sum += loss_reward.item()
            loss_consist_sum += loss_consist.item()

            # 4.6 Gradient half-life: prevent K-step gradient explosion
            s = 0.5 * s_next + 0.5 * s_next.detach()

        # 5. [v4.0] Loss averaging over K steps
        total_loss = total_loss / K

        # 5.1 [v4.7] Reward diversity regularization (HyperMuZero only)
        loss_div = torch.tensor(0.0, device=self.device)
        if is_hyper and getattr(cfg, 'w_rew_diversity', 0) > 0:
            rule_emb = model._current_rule_emb  # cached from set_context()
            loss_div = reward_diversity_loss(
                hyper_rew=model.hyper_net.hyper_rew,
                rule_emb=rule_emb,
                id_embedding=model.context_encoder.id_embedding,
                num_agents=cfg.num_agents,
                target_cos=getattr(cfg, 'rew_diversity_target_cos', 0.3),
                skip_pairs=getattr(cfg, 'rew_diversity_skip_pairs', None),
            )
            total_loss = total_loss + cfg.w_rew_diversity * loss_div

        # 5.2 [Diagnostic] Per-loss gradient norms to context_encoder (every N steps)
        grad_diag = {}
        diag_interval = getattr(cfg, 'grad_diag_interval', 50)
        if is_hyper and self.train_step_count % diag_interval == 0:
            ctx_params = [p for p in model.context_encoder.parameters() if p.requires_grad]
            components = [
                ('pol', total_pol_t / K),
                ('val', total_val_t / K),
                ('rew', total_rew_t / K),
                ('con', total_con_t / K),
            ]
            for name, lc in components:
                self.optimizer.zero_grad()
                lc.backward(retain_graph=True)
                grads = [p.grad.flatten() for p in ctx_params if p.grad is not None]
                grad_diag[f'gd_ctx_{name}'] = torch.cat(grads).norm().item() if grads else 0.0

            gd_total = sum(grad_diag.values()) + 1e-10
            print(f"  [GradDiag] step={self.train_step_count}"
                  f"  ctx_encoder grad norms:"
                  f"  pol={grad_diag['gd_ctx_pol']:.6f} ({grad_diag['gd_ctx_pol']/gd_total:.0%}),"
                  f"  val={grad_diag['gd_ctx_val']:.6f} ({grad_diag['gd_ctx_val']/gd_total:.0%}),"
                  f"  rew={grad_diag['gd_ctx_rew']:.6f} ({grad_diag['gd_ctx_rew']/gd_total:.0%}),"
                  f"  con={grad_diag['gd_ctx_con']:.6f} ({grad_diag['gd_ctx_con']/gd_total:.0%})")

        # 6. Unified backward
        self.optimizer.zero_grad()
        total_loss.backward()

        # [v4.7] Monitor gradient norms before clipping
        grad_norm_model = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        grad_norm_proj = 0.0
        if self.projector is not None:
            grad_norm_proj = nn.utils.clip_grad_norm_(self.projector.parameters(), cfg.grad_clip)
        self.optimizer.step()

        # 7. [v4.4] EMA update target network + LR schedule step
        self._soft_update_target()
        self.scheduler.step()

        self.train_step_count += 1

        result = {
            'loss_total': total_loss.item(),
            'l_pol': loss_policy_sum / K,
            'l_val': loss_value_sum / K,
            'l_rew': loss_reward_sum / K,
            'l_con': loss_consist_sum / K,
            'l_div': loss_div.item(),
            'lr': self.scheduler.get_last_lr()[0],
            'grad_norm': float(grad_norm_model),
        }
        result.update(grad_diag)

        # [Monitoring] Per-agent losses (averaged over K steps)
        for aid in range(num_agents):
            if (agent_ids == aid).any():
                result[f'l_pol_a{aid}'] = per_agent_pol[aid] / K
                result[f'l_val_a{aid}'] = per_agent_val[aid] / K
                result[f'l_rew_a{aid}'] = per_agent_rew[aid] / K

        return result

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
        # v4 migration deferred to Pkg-05 (Q2 折中); see Pkg-04 spec 08 §3.2.
        # v4.7 sites: model.set_context_from_history(...) [orig L474] +
        #             self.target_model.set_context(inferred_rule_emb, agent_ids) [orig L484].
        # v4 deprecates Infer-mode GRU: BeliefNet (Pkg-03) supplies belief; trainer uses
        # set_context_objective + per-agent set_context_subjective (online belief).
        raise NotImplementedError(
            "v4 Infer-mode trainer migration deferred to Pkg-05 (spec 08 §3.2)."
        )

        # Extract per-agent rewards
        rewards_i = rewards_seq[torch.arange(B, device=self.device).unsqueeze(1),
                                torch.arange(K, device=self.device).unsqueeze(0),
                                agent_ids.unsqueeze(1).expand(B, K)]

        # 4. Initial encoding
        s = model.encode(obs_seq[:, 0])

        batch_idx = torch.arange(B, device=self.device)

        # 5. K-step unroll (same structure as train_step)
        total_loss = torch.tensor(0.0, device=self.device)
        loss_policy_sum = 0.0
        loss_value_sum = 0.0
        loss_reward_sum = 0.0
        loss_consist_sum = 0.0

        # [Monitoring] Per-agent loss accumulators (gradient domination detection)
        num_agents = cfg.num_agents
        per_agent_pol = {i: 0.0 for i in range(num_agents)}
        per_agent_val = {i: 0.0 for i in range(num_agents)}
        per_agent_rew = {i: 0.0 for i in range(num_agents)}

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

            # [v4.6] Plain CE (per-sample first, then batch average)
            loss_pol_per = -(target_pi * F.log_softmax(p_k, dim=-1)).sum(dim=-1)  # (B,)
            loss_policy = loss_pol_per.mean()

            loss_val_per = F.mse_loss(v_k.squeeze(-1), target_z, reduction='none')  # (B,)
            loss_value = loss_val_per.mean()

            loss_rew_per = F.mse_loss(r_k, target_r, reduction='none').squeeze(-1)  # (B,)
            loss_reward = loss_rew_per.mean()

            loss_consist = negative_cosine_similarity(proj_pred, proj_target)

            # [Monitoring] Per-agent loss accumulation (detached, logging only)
            with torch.no_grad():
                for aid in range(num_agents):
                    mask = (agent_ids == aid)
                    if mask.any():
                        per_agent_pol[aid] += loss_pol_per[mask].mean().item()
                        per_agent_val[aid] += loss_val_per[mask].mean().item()
                        per_agent_rew[aid] += loss_rew_per[mask].mean().item()

            step_loss = (cfg.w_policy * loss_policy +
                         cfg.w_value * loss_value +
                         cfg.w_reward * loss_reward +
                         cfg.w_consist * loss_consist)
            total_loss = total_loss + step_loss

            loss_policy_sum += loss_policy.item()
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

        # 7.1 [v4.7] Reward diversity regularization
        loss_div = torch.tensor(0.0, device=self.device)
        if getattr(cfg, 'w_rew_diversity', 0) > 0:
            # For Infer model, use the GRU-inferred rule_emb
            loss_div = reward_diversity_loss(
                hyper_rew=model.hyper_net.hyper_rew,
                rule_emb=inferred_rule_emb,
                id_embedding=model.context_encoder.id_embedding,
                num_agents=cfg.num_agents,
                target_cos=getattr(cfg, 'rew_diversity_target_cos', 0.3),
                skip_pairs=getattr(cfg, 'rew_diversity_skip_pairs', None),
            )
            total_loss = total_loss + cfg.w_rew_diversity * loss_div

        # 8. Unified backward
        self.optimizer.zero_grad()
        total_loss.backward()

        # [v4.7] Monitor gradient norms before clipping
        grad_norm_model = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        grad_norm_proj = 0.0
        if self.projector is not None:
            grad_norm_proj = nn.utils.clip_grad_norm_(self.projector.parameters(), cfg.grad_clip)
        self.optimizer.step()

        # 9. [v4.4] EMA update target network + LR schedule step
        self._soft_update_target()
        self.scheduler.step()

        self.train_step_count += 1

        result = {
            'loss_total': total_loss.item(),
            'l_pol': loss_policy_sum / K,
            'l_val': loss_value_sum / K,
            'l_rew': loss_reward_sum / K,
            'l_con': loss_consist_sum / K,
            'l_ctx': loss_ctx.item(),
            'l_div': loss_div.item(),
            'lr': self.scheduler.get_last_lr()[0],
            'grad_norm': float(grad_norm_model),
        }

        # [Monitoring] Per-agent losses (averaged over K steps)
        for aid in range(num_agents):
            if (agent_ids == aid).any():
                result[f'l_pol_a{aid}'] = per_agent_pol[aid] / K
                result[f'l_val_a{aid}'] = per_agent_val[aid] / K
                result[f'l_rew_a{aid}'] = per_agent_rew[aid] / K

        return result

    def compute_hypernet_diagnostics(self, rules=None):
        """
        Compute hypernetwork output differentiation metrics.

        Measures cosine similarity of generated parameters between agent pairs,
        indicating whether the hypernetwork produces sufficiently different
        parameters for different agents (negative transfer detection).

        Also measures ID embedding pairwise cosine similarity.

        Only applicable to HyperMuZero models.

        Args:
            rules: list of float rule values to test. Default [0.0, 0.5, 1.0].
                   Ignored for InferHyperMuZeroModel (uses default embedding).

        Returns:
            dict of metric_name -> float value
        """
        model = self.model
        if not _is_hyper_model(model):
            return {}

        from models.hyper_muzero_model import OracleHyperMuZeroModel
        from models.infer_muzero_model import InferHyperMuZeroModel

        is_oracle = isinstance(model, OracleHyperMuZeroModel)
        is_infer = isinstance(model, InferHyperMuZeroModel)

        if not (is_oracle or is_infer):
            return {}

        device = self.device
        num_agents = self.cfg.num_agents
        results = {}

        if rules is None:
            rules = [0.0, 0.5, 1.0]

        # For Infer model, rule-based theta comparison uses default embedding
        rule_list = rules if is_oracle else [0.0]

        for rule_val in rule_list:
            thetas_pred = {}
            thetas_rew = {}

            for aid in range(num_agents):
                id_t = torch.tensor([aid], dtype=torch.long, device=device)
                with torch.no_grad():
                    # v4 migration deferred to Pkg-05 (Q2 折中); see Pkg-04 spec 08 §3.2.
                    # v4.7 sites: model.set_context(rule_t, id_t) [orig L678] +
                    #             model.set_context_default(id_t, batch_size=1) [orig L680].
                    # v4 theta diagnostics: set_context_objective + set_context_subjective,
                    # read model._theta_rew (renamed from _theta_reward) / _theta_pred.
                    raise NotImplementedError(
                        "v4 theta-diagnostic set_context migration deferred to Pkg-05 "
                        "(spec 08 §3.2)."
                    )

            rule_tag = f'r{rule_val:.1f}' if is_oracle else 'default'
            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    cos_p = F.cosine_similarity(
                        thetas_pred[i], thetas_pred[j], dim=-1
                    ).item()
                    cos_r = F.cosine_similarity(
                        thetas_rew[i], thetas_rew[j], dim=-1
                    ).item()
                    results[f'hyper/{rule_tag}/cos_pred_{i}v{j}'] = cos_p
                    results[f'hyper/{rule_tag}/cos_rew_{i}v{j}'] = cos_r

        # ID embedding pairwise cosine similarity (rule-independent)
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                id_i = torch.tensor([i], dtype=torch.long, device=device)
                id_j = torch.tensor([j], dtype=torch.long, device=device)
                with torch.no_grad():
                    emb_i = model.get_id_emb(id_i)
                    emb_j = model.get_id_emb(id_j)
                    cos_id = F.cosine_similarity(emb_i, emb_j, dim=-1).item()
                results[f'hyper/cos_id_{i}v{j}'] = cos_id

        # [v4.7] Output scale monitoring (track initialization trap recovery)
        if hasattr(model, 'hyper_net'):
            hn = model.hyper_net
            results['hyper/output_scale_trans'] = hn.hyper_trans.output_scale.item()
            results['hyper/output_scale_rew'] = hn.hyper_rew.output_scale.item()
            results['hyper/output_scale_pred'] = hn.hyper_pred.output_scale.item()

        # Clean up: diagnostics modified model's context cache
        model.clear_context()

        return results
