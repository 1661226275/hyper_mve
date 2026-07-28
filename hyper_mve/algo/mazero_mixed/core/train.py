from collections import deque
import logging
import math
import os
import time

import numpy as np
import ray
import torch
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
import torch.optim as optim

from core.config import BaseConfig
from core.model import BaseNet
from core.log import _log, train_logger
from core.test import test, TestWorker, RemoteTestWorker
from core.replay_buffer import ReplayBuffer, RemoteReplayBuffer, RemotePriorityRefresher
from core.storage import SharedStorage, RemoteShareStorage, RemoteQueueStorage
from core.selfplay_worker import DataWorker, RemoteDataWorker
from core.reanalyze_worker import ReanalyzeWorker, RemoteReanalyzeWorker
from core.utils import Timer, remote_worker_handles

#: Uniform env-step cadence for the eval/* + fidelity/* curves, matching the
#: external baselines (see _PROBE_EVERY_ENV_STEPS in hyper_mve/comparison/*.py).
#: Gating these on gradient steps gave every algorithm a different curve
#: resolution (800 / 3200 / 20000 / 80000 env-steps per point) because each does
#: a different number of updates per env step. 5000 -> 200 points per 1M run for
#: EVERY algorithm, on the canonical env-step x-axis.
_PROBE_EVERY_ENV_STEPS = 5000


def reward_nonzero_weight(target_reward_step: torch.Tensor, upweight: float, eps: float) -> torch.Tensor:
    """2026-07-20 harvest-collapse fix: per-sample reward-loss weight that
    upweights transitions carrying nonzero raw team reward (see
    ``--reward_nonzero_upweight`` / ``--reward_nonzero_eps`` in config.py).

    ``target_reward_step`` is ``(batch, num_agents)`` raw (pre-transform)
    reward for one unroll step. A transition counts as nonzero if the
    team-summed absolute reward exceeds ``eps`` — this also catches the
    negative per-step movement cost, not just positive harvest payoff.
    Returns ``(batch,)``; all-ones when ``upweight <= 0`` (off).
    """
    if upweight <= 0:
        return torch.ones(target_reward_step.shape[0], device=target_reward_step.device)
    nz = (target_reward_step.abs().sum(dim=-1) > eps).float()
    return 1.0 + upweight * nz


def policy_target_weights(target_sampled_adv_step, sampled_action_mask_step, temperature):
    """Per-agent policy-target weights from the search's OWN advantage estimates.

    The upstream target is the normalized root visit count. Visits are
    allocated by UCB, whose prior term dominates the [0,1]-clipped value term
    when the prior is peaked, so under a collapsed policy the target mostly
    echoes the prior back at itself. This reads the target from what the search
    actually *evaluated* instead: a softmax over per-agent advantages
    (``qvalues - root_pred_value``), which no term of the prior enters.
    (The retired MVEPlanner's ``pi_mve = softmax(return_per_action / temp)``.)

    ``target_sampled_adv_step`` is ``(batch, C, num_agents)`` and
    ``sampled_action_mask_step`` is ``(batch, C)``. Returns ``(batch, C, N)``
    rows that sum to 1 over C, or exactly 0 for fully-masked rows.
    """
    m = sampled_action_mask_step.unsqueeze(-1)                      # (batch, C, 1)
    # fp32: under autocast the softmax would run in fp16, where the masking
    # sentinel below has to stay well inside the 65504 range.
    logits = target_sampled_adv_step.float() / temperature
    # -1e4, NOT -inf/-1e9: out-of-trajectory rows have an ALL-False mask
    # (reanalyze_worker.py zeroes them), and softmax over an all -inf row is
    # NaN -- which then survives `0 * NaN` and poisons total_loss even though
    # the row's weight is zero. -1e9 has the same effect via fp16 overflow.
    w = torch.softmax(logits.masked_fill(m < 0.5, -1e4), dim=1) * m
    # renormalize over the unmasked children; clamp_min keeps all-masked rows
    # at exactly 0 instead of 0/0.
    return w / w.sum(dim=1, keepdim=True).clamp_min(1e-8)


def policy_loss_step(config, per_agent_log_prob, sampled_actions_log_prob,
                     target_sampled_policies_step, target_sampled_adv_step,
                     sampled_action_mask_step):
    """One unroll step of the PG_type='none' policy loss. Returns ``(batch,)``."""
    if getattr(config, "policy_target_type", "visit") == "visit":
        return -(
            sampled_actions_log_prob
            * target_sampled_policies_step                      # visit count
            * sampled_action_mask_step                          # mask invalid actions
        ).sum(dim=1)
    w = policy_target_weights(
        target_sampled_adv_step, sampled_action_mask_step,
        config.policy_target_temperature)                       # (batch, C, N)
    return -(per_agent_log_prob * w.permute(0, 2, 1)).sum(dim=(1, 2))


def bc_loss_step(policy_logits, action_step, ref_flag, mask_step, weight_step):
    """One unroll step of the behavior-cloning loss (2026-07-21 competence fix).

    CE from the policy toward the EXECUTED action, active only on reference-
    episode steps. On a reference self-play episode the executed action is the
    scripted-greedy demonstration, but the reanalyze policy target is the
    (collapsible) MCTS visit distribution -- so without this the demonstrated
    action never becomes a policy target and the policy re-collapses to HARVEST.

        policy_logits (batch, N, A); action_step (batch, N); ref_flag (batch,);
        mask_step (batch,); weight_step (batch, N) per-agent reward weight
        (all-ones = plain uniform BC). Returns ``(batch,)``, exactly 0 on
        non-reference or out-of-trajectory rows. fp32-forced so the CE is
        autocast-safe. The per-agent weight is applied BEFORE the sum over agents
        (general-sum: agent i is weighted by its own return, 2026-07-22).
    """
    logp = policy_logits.float().log_softmax(dim=-1)                        # (b, N, A)
    ce = -logp.gather(dim=2, index=action_step.long().unsqueeze(-1)).squeeze(-1)  # (b, N)
    return (ce * weight_step).sum(dim=1) * ref_flag * mask_step             # (b,)


def bc_reward_weights(config, rtg, regime_id, ref_flag, mask, n_regimes):
    """Per-(regime, agent) reward weight for reward-weighted BC (2026-07-22).

    Weight each reference step's per-agent BC by how good the demonstrated action
    was — the agent's own full-episode return-to-go — so walk-toward-resource
    (high G) is upweighted and idle/low-value steps are not. Only above-own-average
    steps are upweighted (``clamp(Â, min=0)``); below-average steps get weight 0.

        rtg (B, K+1, N) per-agent return-to-go; regime_id (B,); ref_flag (B,);
        mask (B, K+1). Returns (B, K+1, N). All-ones when --bc_reward_weighting is
        off (⇒ bit-exact plain BC). Normalization is per-(regime, agent) because
        the return is per-agent and asymmetric regimes have per-role scales.
    """
    if not getattr(config, "bc_reward_weighting", False):
        return torch.ones_like(rtg)
    B, K1, N = rtg.shape
    # population for the per-group stats: reference AND in-trajectory steps
    pop = ((ref_flag.view(B, 1, 1) > 0.5) & (mask.view(B, K1, 1) > 0.5)).expand(B, K1, N)
    # UNIFORM FALLBACK: groups too sparse to normalize keep weight 1 (plain BC),
    # never 0 — the escape-HARVEST BC signal must not silently vanish just because
    # a (regime, agent) group has too few samples to standardize this batch.
    w = torch.ones_like(rtg)
    for g in range(int(n_regimes)):
        gsel = (regime_id == g)                                             # (B,)
        if not bool(gsel.any()):
            continue
        gcol = gsel.view(B, 1)                                              # (B,1)
        for i in range(N):
            valid = pop[:, :, i] & gcol                                     # (B,K1)
            if int(valid.sum()) < 2:
                continue                                                    # keep w=1
            vals = rtg[:, :, i][valid]
            mean, std = vals.mean(), vals.std()
            col = ((rtg[:, :, i] - mean) / (std + 1e-5)).clamp(min=0.0)     # positive-only
            w[:, :, i] = torch.where(gcol, col, w[:, :, i])
    cap = getattr(config, "bc_weight_cap", 0.0)
    if cap and cap > 0:
        w = w.clamp(max=float(cap))
    return w


def update_weights(config: BaseConfig, step_count: int, model: BaseNet, batch: tuple, optimizer: optim.Optimizer, scaler: GradScaler, device):
    """update models given a batch data
    Parameters
    ----------
    model: Any
        EfficientZero models
    batch: Any
        a batch data inlcudes [inputs_batch, targets_batch]
    scaler: Any
        scaler for torch amp
    """
    inputs_batch, targets_batch, info = batch
    obs_batch, action_batch, mask_batch, indices, weights_lst = inputs_batch[:5]
    reference_flag_b = inputs_batch[5]          # (B,) behavior-cloning flag
    returns_to_go_b = inputs_batch[6]           # (B, K+1, N) per-agent full-episode G_t
    regime_id_b = inputs_batch[7]               # (B,) regime id at the unroll start
    # subjective model extras: belief ctx at unroll start, oracle g, GRU hidden
    belief_ctx_b = g_true_b = belief_hidden_b = None
    if len(inputs_batch) > 8:
        belief_ctx_b, g_true_b, belief_hidden_b = inputs_batch[8:11]
    target_reward, target_value, target_policy = targets_batch
    (
        target_sampled_actions,
        target_sampled_policies,
        target_sampled_imp_ratio,
        target_sampled_adv,
        sampled_action_mask,
        target_policy_informative,
    ) = target_policy
    batch_future_return, batch_model_index, target_model_index = info

    if config.image_based:
        obs_batch = torch.from_numpy(np.array(obs_batch)).to(device).float() / 255.0
    else:
        obs_batch = torch.from_numpy(np.array(obs_batch)).to(device).float()

    # do augmentations
    if config.use_augmentation:
        obs_batch = config.augmentation_transform(obs_batch)

    # use GPU tensor
    action_batch = torch.from_numpy(np.array(action_batch)).to(device).long()
    mask_batch = torch.from_numpy(np.array(mask_batch)).to(device).float()
    weights = torch.from_numpy(np.array(weights_lst)).to(device).float()
    reference_flag_t = torch.from_numpy(np.array(reference_flag_b)).to(device).float()  # (B,)
    returns_to_go_t = torch.from_numpy(np.array(returns_to_go_b)).to(device).float()    # (B,K+1,N)
    regime_id_t = torch.from_numpy(np.array(regime_id_b)).to(device).long()             # (B,)
    # per-(regime,agent) reward-weight for BC (all-ones unless --bc_reward_weighting)
    bc_weight = bc_reward_weights(
        config, returns_to_go_t, regime_id_t, reference_flag_t, mask_batch,
        config.num_agents)                                                              # (B,K+1,N)

    target_reward = torch.from_numpy(np.array(target_reward)).to(device).float()
    target_value = torch.from_numpy(np.array(target_value)).to(device).float()
    # additional context for policy loss
    target_sampled_actions = torch.from_numpy(np.array(target_sampled_actions)).to(device).long()
    target_sampled_policies = torch.from_numpy(np.array(target_sampled_policies)).to(device).float()
    target_sampled_imp_ratio = torch.from_numpy(np.array(target_sampled_imp_ratio)).to(device).float()
    target_sampled_adv = torch.from_numpy(np.array(target_sampled_adv)).to(device).float()
    sampled_action_mask = torch.from_numpy(np.array(sampled_action_mask)).to(device).float()
    target_policy_informative = torch.from_numpy(np.array(target_policy_informative)).to(device).float()

    batch_size = obs_batch.size(0)
    obs_pad_size = config.image_channel * (config.stacked_observations + config.num_unroll_steps)
    # data shape check
    assert batch_size == config.batch_size
    assert obs_batch.shape == (batch_size, config.num_agents, obs_pad_size, *config.obs_shape[:-1])
    assert action_batch.shape == (batch_size, config.num_unroll_steps + 1, config.num_agents)
    assert mask_batch.shape == (batch_size, config.num_unroll_steps + 1)
    assert target_reward.shape == (batch_size, config.num_unroll_steps + 1, config.num_agents)
    assert target_value.shape == (batch_size, config.num_unroll_steps + 1, config.num_agents)

    assert target_sampled_actions.shape == (batch_size, config.num_unroll_steps + 1, config.sampled_action_times, config.num_agents)
    assert target_sampled_policies.shape == (batch_size, config.num_unroll_steps + 1, config.sampled_action_times)
    assert target_sampled_imp_ratio.shape == (batch_size, config.num_unroll_steps + 1, config.sampled_action_times)
    assert target_sampled_adv.shape == (batch_size, config.num_unroll_steps + 1, config.sampled_action_times, config.num_agents)
    assert sampled_action_mask.shape == (batch_size, config.num_unroll_steps + 1, config.sampled_action_times)
    assert target_policy_informative.shape == (batch_size, config.num_unroll_steps + 1)

    # transform targets to categorical representation
    target_reward_phi = config.reward_transform(target_reward)
    target_value_phi = config.value_transform(target_value)

    gradient_scale = 1 / config.num_unroll_steps

    with autocast():

        # init with the stacked observations:
        step_i, beg_index = 0, 0
        end_index = config.image_channel * config.stacked_observations
        if belief_ctx_b is not None:
            model.set_belief(
                torch.from_numpy(np.array(belief_ctx_b)).to(device).float(),
                step=step_count,
            )
        if (g_true_b is not None and getattr(config, "value_hard_select", False)
                and hasattr(model, "set_oracle_regime")):
            model.set_oracle_regime(
                torch.from_numpy(np.array(g_true_b)).to(device).long())
        network_output = model.initial_inference(obs_batch[:, :, beg_index:end_index])

        # calculate the new priorities for each transition (agent-mean of the
        # per-agent value errors)
        scaled_value = config.inverse_value_transform(network_output.value).squeeze(-1)
        value_priority = np.abs(
            scaled_value.detach().cpu().numpy() - target_value[:, step_i].detach().cpu().numpy()
        ).mean(-1) + config.prioritized_replay_eps
        new_priority_data = (indices, value_priority)

        # loss of the first step

        per_agent_log_prob = (
            network_output.policy_logits.log_softmax(dim=-1)        # (batch_size, num_agents, action_space_size)
            .gather(dim=2, index=target_sampled_actions[:, step_i].transpose(1, 2))  # index: (.., sampled_times, num_agents) -> (.., num_agents, sampled_times)
        )                       # (batch_size, num_agents, sampled_times)
        sampled_actions_log_prob = per_agent_log_prob.sum(dim=1)    # joint log-prob: (batch_size, sampled_times)

        if config.PG_type == "none":
            policy_loss = policy_loss_step(
                config, per_agent_log_prob, sampled_actions_log_prob,
                target_sampled_policies[:, step_i], target_sampled_adv[:, step_i],
                sampled_action_mask[:, step_i]) * target_policy_informative[:, step_i]
        else:
            # per-agent AWPO: agent i's factor is weighted by its OWN advantage
            if config.awac_lambda > 0:
                adv_weights = torch.exp(target_sampled_adv[:, step_i] / config.awac_lambda)
                '''Reference: https://github.com/Junyoungpark/Pytorch-AWAC/blob/main/src/Learner/AWAC.py#L77'''
            else:
                adv_weights = target_sampled_adv[:, step_i]

            if config.adv_clip > 0:
                adv_weights = torch.clamp(adv_weights, -config.adv_clip, config.adv_clip)
            adv_weights = adv_weights.permute(0, 2, 1)                 # (batch, C, N) -> (batch, N, C)

            if config.PG_type == "raw":
                policy_loss = -(
                    per_agent_log_prob
                    * adv_weights                                       # per-agent AWAC weight
                    * target_sampled_imp_ratio[:, step_i].unsqueeze(1)  # importance ratio
                    * sampled_action_mask[:, step_i].unsqueeze(1)       # mask invalid actions
                ).sum(dim=(1, 2))       # (batch, num_agents, sampled_times) -> (batch,)
            elif config.PG_type == "sharp":
                policy_loss = -(
                    per_agent_log_prob
                    * adv_weights                                       # per-agent AWAC weight
                    * target_sampled_policies[:, step_i].unsqueeze(1)   # visit count
                    * sampled_action_mask[:, step_i].unsqueeze(1)       # mask invalid actions
                ).sum(dim=(1, 2))       # (batch, num_agents, sampled_times) -> (batch,)
            else:
                raise NotImplementedError

        # behavior-cloning loss at step 0 (accumulated over the unroll below).
        bc_loss = bc_loss_step(network_output.policy_logits, action_batch[:, 0],
                               reference_flag_t, mask_batch[:, 0], bc_weight[:, 0])
        reward_loss = torch.zeros(batch_size, device=device)
        value_loss = config.value_loss(network_output.value, target_value_phi[:, 0])
        if config.consistency_coeff > 0:
            consistency_loss = torch.zeros(batch_size, device=device)

        # unroll with the dynamics function using actual executed action
        for step_i in range(1, config.num_unroll_steps + 1):
            beg_index = config.image_channel * step_i
            end_index = config.image_channel * (step_i + config.stacked_observations)
            network_output = model.recurrent_inference(network_output.hidden_state, action_batch[:, step_i - 1])

            # loss of the unrolled steps (k=1,...,K)

            per_agent_log_prob = (
                network_output.policy_logits.log_softmax(dim=-1)        # (batch_size, num_agents, action_space_size)
                .gather(dim=2, index=target_sampled_actions[:, step_i].transpose(1, 2))  # index: (.., sampled_times, num_agents) -> (.., num_agents, sampled_times)
            )                       # (batch_size, num_agents, sampled_times)
            sampled_actions_log_prob = per_agent_log_prob.sum(dim=1)    # joint log-prob: (batch_size, sampled_times)

            if config.PG_type == "none":
                policy_loss += policy_loss_step(
                    config, per_agent_log_prob, sampled_actions_log_prob,
                    target_sampled_policies[:, step_i], target_sampled_adv[:, step_i],
                    sampled_action_mask[:, step_i]) * target_policy_informative[:, step_i]
            else:
                # per-agent AWPO: agent i's factor is weighted by its OWN advantage
                if config.awac_lambda > 0:
                    adv_weights = torch.exp(target_sampled_adv[:, step_i] / config.awac_lambda)
                else:
                    adv_weights = target_sampled_adv[:, step_i]

                if config.adv_clip > 0:
                    adv_weights = torch.clamp(adv_weights, -config.adv_clip, config.adv_clip)
                adv_weights = adv_weights.permute(0, 2, 1)                 # (batch, C, N) -> (batch, N, C)

                if config.PG_type == "raw":
                    policy_loss += -(
                        per_agent_log_prob
                        * adv_weights                                       # per-agent AWAC weight
                        * target_sampled_imp_ratio[:, step_i].unsqueeze(1)  # importance ratio
                        * sampled_action_mask[:, step_i].unsqueeze(1)       # mask invalid actions
                    ).sum(dim=(1, 2))       # (batch, num_agents, sampled_times) -> (batch,)
                elif config.PG_type == "sharp":
                    policy_loss += -(
                        per_agent_log_prob
                        * adv_weights                                       # per-agent AWAC weight
                        * target_sampled_policies[:, step_i].unsqueeze(1)   # visit count
                        * sampled_action_mask[:, step_i].unsqueeze(1)       # mask invalid actions
                    ).sum(dim=(1, 2))       # (batch, num_agents, sampled_times) -> (batch,)
                else:
                    raise NotImplementedError

            bc_loss += bc_loss_step(network_output.policy_logits, action_batch[:, step_i],
                                    reference_flag_t, mask_batch[:, step_i], bc_weight[:, step_i])

            # 2026-07-20 harvest-collapse fix: zero-inflation counter. Under a
            # camping-heavy behaviour policy the overwhelming majority of
            # transitions carry ~0 team reward, so an unweighted reward loss
            # gives the encoder almost no gradient toward what "harvesting/
            # moving pays" looks like (confirmed by fresh_head_probe.py: the
            # frozen latent could not decode the on-resource-cell affordance
            # that raw obs decodes at 0.93 balanced accuracy). Upweight the
            # per-transition reward loss on the raw (pre-transform) reward so
            # informative steps count more than idle no-payoff steps.
            step_reward_weight = reward_nonzero_weight(
                target_reward[:, step_i - 1], config.reward_nonzero_upweight, config.reward_nonzero_eps)
            reward_loss += step_reward_weight * config.reward_loss(network_output.reward, target_reward_phi[:, step_i - 1])          # don't mask reward loss
            value_loss += config.value_loss(network_output.value, target_value_phi[:, step_i])                  # don't mask value loss
            if config.consistency_coeff > 0:
                # obtain the oracle hidden states from representation function
                representation = model.initial_inference(obs_batch[:, :, beg_index:end_index])
                # no grad for the presentation_state branch
                dynamic_proj = model.project(network_output.hidden_state, with_grad=True)    # P2(P1(s_{t,k}))
                represet_proj = model.project(representation.hidden_state, with_grad=False)  # sg(P1(s_{t+k,0}))
                consistency_loss += config.consistency_loss(dynamic_proj, represet_proj)                        # don't mask consistency loss ???
            # Follow MuZero, set half gradient
            network_output.hidden_state.register_hook(lambda grad: grad * 0.5)

        # Stage C renormalization. Masking transitions out of the policy loss
        # silently lowers the EFFECTIVE policy learning rate, because
        # total_loss averages over the full batch either way. Rescale by the
        # guarded-out fraction, as a global scalar so the per-sample (batch,)
        # structure the PER `weights` multiply into is preserved.
        #
        # `num` must be mask_batch.sum(), not numel(): out-of-trajectory rows
        # already contribute exactly 0 (reanalyze_worker zeroes their targets),
        # so counting them would silently deflate the scale on short
        # trajectories. When the guard never fires den == num -> scale == 1.0,
        # bit-exact with it disabled. den == 0 -> scale 0, no NaN and no
        # gradient. The cap bounds gradient-variance inflation on batches where
        # only a few transitions survive.
        policy_informative_frac = 1.0
        if getattr(config, "policy_target_min_qstd", 0.0) > 0:
            num = mask_batch.sum()
            den = (target_policy_informative * mask_batch).sum()
            policy_informative_frac = float((den / num.clamp_min(1.0)).item())
            scale = (torch.clamp(num / den, max=config.policy_target_renorm_cap)
                     if den > 0 else torch.zeros((), device=device))
            policy_loss = policy_loss * scale

        # weighted loss with masks (some invalid states which are out of trajectory.)
        loss = (
            config.reward_loss_coeff * reward_loss
            + config.policy_loss_coeff * policy_loss
            + config.value_loss_coeff * value_loss
            + getattr(config, "bc_loss_coeff", 0.0) * bc_loss
        )
        if config.consistency_coeff > 0:
            loss += config.consistency_coeff * consistency_loss
        total_loss = (weights * loss).mean()
        total_loss.register_hook(lambda grad: grad * gradient_scale)

        # BeliefNet supervision (subjective model): one-step truncated GRU
        # update on the sampled positions, CE against the oracle regime id
        # plus the anti-collapse diversity hinge. Trains BeliefNet through its
        # own independent loss path (the main loss is gated during warmup).
        belief_total = None
        if belief_hidden_b is not None and hasattr(model, "belief_net"):
            from hyper_mve.algo.modules.belief_losses import l_regime, l_div

            obs_t = obs_batch[:, :, 0:config.image_channel * config.stacked_observations]
            obs_t = obs_t.reshape(batch_size, config.num_agents, -1)
            prev_h = torch.from_numpy(np.array(belief_hidden_b)).to(device).float()
            g_true_t = torch.from_numpy(np.array(g_true_b)).to(device).long()
            valid = (g_true_t >= 0)
            if valid.any():
                new_h = model.belief_net._gru_step(obs_t, prev_h)
                g_hat_t = model.belief_net._compute_g_hat(new_h)
                bl_mask = valid.unsqueeze(1)                       # (B, T=1)
                bl_reg = l_regime(
                    g_hat_t.unsqueeze(1), g_true_t.clamp(min=0).unsqueeze(1), mask=bl_mask
                )
                bl_div = l_div(new_h.unsqueeze(1), mask=bl_mask)
                belief_total = bl_reg + 0.01 * bl_div
                total_loss = total_loss + config.belief_loss_coeff * belief_total

    # backward
    lr = config.adjust_lr(optimizer, step_count)
    optimizer.zero_grad()
    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    if hasattr(model, "set_oracle_regime"):
        model.set_oracle_regime(None)   # clear: reanalyze/eval/deploy Bayes-average

    # packing data for logging
    train_logs = {
        'total_loss': total_loss.item(),
        'reward_loss': (weights * reward_loss).mean().item(),
        'policy_loss': (weights * policy_loss).mean().item(),
        'value_loss': (weights * value_loss).mean().item(),
    }

    train_logs['bc_loss'] = (weights * bc_loss).mean().item()
    train_logs['reference_frac'] = reference_flag_t.mean().item()
    if getattr(model, "_head_diversity", None) is not None:
        train_logs['head_diversity'] = model._head_diversity.item()
    # reward-weighted-BC monitoring over reference & in-trajectory steps
    _pop = ((reference_flag_t.view(-1, 1, 1) > 0.5)
            & (mask_batch.unsqueeze(-1) > 0.5)).expand_as(returns_to_go_t)
    if bool(_pop.any()):
        train_logs['G_t_mean'] = returns_to_go_t[_pop].mean().item()
        train_logs['G_t_std'] = returns_to_go_t[_pop].std().item()
        train_logs['bc_weight_mean'] = bc_weight[_pop].mean().item()
        for g in range(5):
            gp = _pop & (regime_id_t == g).view(-1, 1, 1)
            if bool(gp.any()):
                train_logs[f'bc_weight_regime_{g}'] = bc_weight[gp].mean().item()
    if config.consistency_coeff > 0:
        train_logs['consistency_loss'] = (weights * consistency_loss).mean().item()
    if belief_total is not None:
        train_logs['belief_loss'] = belief_total.item()
    train_logs['lr'] = lr
    # Watch this: near 0 means the guard is starving the policy of gradient;
    # near 1 means it is inert and the threshold is too low to matter.
    train_logs['policy_target_informative_frac'] = policy_informative_frac
    train_logs['batch_future_return'] = batch_future_return
    train_logs['batch_model_diff'] = step_count - batch_model_index
    train_logs['target_model_diff'] = step_count - target_model_index

    return train_logs, new_priority_data


def _train(model, target_model, replay_buffer, shared_storage, batch_storage, config: BaseConfig, summary_writer):
    """training loop
    Parameters
    ----------
    model: Any
        EfficientZero models
    target_model: Any
        EfficientZero models for reanalyzing
    replay_buffer: Any
        replay buffer
    shared_storage: Any
        model storage
    batch_storage: Any
        batch storage (queue)
    summary_writer: Any
        logging for tensorboard
    """
    # ----------------------------------------------------------------------------------
    device = 'cuda' if (config.train_on_gpu and torch.cuda.is_available()) else 'cpu'
    model = model.to(device)
    target_model = target_model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr, eps=config.opti_eps,
                           weight_decay=config.weight_decay)

    scaler = GradScaler()

    model.train()
    target_model.eval()
    # ----------------------------------------------------------------------------------
    # set augmentation tools
    if config.use_augmentation:
        config.set_augmentation_transforms()

    # wait until collecting enough data to start
    while True:
        transitions_collected = ray.get(replay_buffer.transitions_collected.remote())
        train_logger.debug(f'ReplayBufferSize:{transitions_collected}/{config.start_transitions}')
        if transitions_collected >= config.start_transitions:
            break
        else:
            time.sleep(3)

    train_logger.info('Begin training...')
    # set signals for other workers
    shared_storage.set_start_signal.remote()

    step_count = 0
    # Note: the interval of the current model and the target model is between x and 2x. (x = target_model_interval)
    # recent_weights is the param of the target model
    recent_weights = model.get_weights()

    batch_start_time = time.time()
    train_start_time = batch_start_time
    batch_timecost, train_timecost = 0, 0

    # while loop
    while step_count < config.training_steps + config.last_steps:

        # obtain a batch
        batch = batch_storage.pop()
        if batch is None:
            time.sleep(0.5)
            continue
        current_time = time.time()
        batch_timecost += current_time - batch_start_time
        train_start_time = current_time
        shared_storage.incr_counter.remote()

        # update model for self-play
        if step_count % config.checkpoint_interval == 0:
            shared_storage.set_weights.remote(step_count, model.get_weights())

        # update model for reanalyzing
        if step_count % config.target_model_interval == 0:
            shared_storage.set_target_weights.remote(step_count, recent_weights)
            recent_weights = model.get_weights()

        train_logs, new_priority_data = update_weights(config, step_count, model, batch, optimizer, scaler, device)

        if config.use_priority and not config.use_priority_refresh:
            # update priority if no refresher
            indices, new_priority = new_priority_data
            replay_buffer.update_priorities.remote(indices, new_priority)

        current_time = time.time()
        train_timecost += current_time - train_start_time
        batch_start_time = current_time
        train_logs['Tp_perstep'] = batch_timecost / (step_count + 1)
        train_logs['Tu_perstep'] = train_timecost / (step_count + 1)

        if step_count % config.log_interval == 0:
            _log(config, step_count, train_logs, replay_buffer, shared_storage, summary_writer)

        # Chech queue capacity.
        if step_count >= 100 and step_count % 50 == 0:
            if batch_storage.get_len() == 0:
                train_logger.warn(f'#{step_count} Batch Queue is empty (Require more reanalyze actors Or actor fails).')
            elif batch_storage.get_len() == batch_storage.threshold:
                train_logger.warn(f'#{step_count} Batch Queue is excess (Reduce reanalyze actors).')

        step_count += 1

        # save models
        if step_count % config.save_interval == 0:
            model_path = os.path.join(config.model_dir, 'model_{}.p'.format(step_count))
            torch.save(model.state_dict(), model_path)

    shared_storage.set_weights.remote(step_count, model.get_weights())
    time.sleep(30)
    return model.get_weights()


def train(config: BaseConfig, summary_writer, model_path=None):
    """training process
    Parameters
    ----------
    summary_writer: Any
        logging for tensorboard
    model_path: str
        model path for resuming
        default: train from scratch
    """
    model = config.get_uniform_network()
    target_model = config.get_uniform_network()
    if model_path:
        train_logger.info('resume model from path: {}'.format(model_path))
        weights = torch.load(model_path)
        model.load_state_dict(weights)
        target_model.load_state_dict(weights)

    shared_storage = RemoteShareStorage.remote(model, target_model)

    # training-data flow: replay_buffer -> reanalyze_workers -> batch_storage -> _train
    replay_buffer = RemoteReplayBuffer.remote(config=config)
    batch_storage = RemoteQueueStorage(config.reanalyze_actors, math.ceil(config.reanalyze_actors * 1.5))

    # parallel tasks
    tasks = []

    availabel_gpus = ray.available_resources().get('GPU', 0)
    num_gpu_workers = 2  # train + test
    if config.selfplay_on_gpu:
        num_gpu_workers += config.data_actors
    if config.reanalyze_on_gpu:
        num_gpu_workers += (config.reanalyze_actors + config.reanalyze_update_actors)
    if config.use_priority_refresh:
        num_gpu_workers += config.refresh_actors
    num_gpus_per_worker = 1 / math.ceil(num_gpu_workers / availabel_gpus)

    # self-play workers
    data_workers = [
        RemoteDataWorker.options(
            num_gpus=num_gpus_per_worker if config.selfplay_on_gpu else 0,
        ).remote(
            rank, config, replay_buffer, shared_storage,
        ) for rank in range(config.data_actors)
    ]
    tasks += [worker.run_loop.remote() for worker in data_workers]

    # test workers
    test_worker = RemoteTestWorker.options(
        num_gpus=num_gpus_per_worker if config.selfplay_on_gpu else 0,
    ).remote(config, shared_storage)
    tasks += [test_worker.run_loop.remote()]

    # priority-refresh workers
    if config.use_priority_refresh:
        refresh_workers = [
            RemotePriorityRefresher.options(
                num_gpus=num_gpus_per_worker
            ).remote(config, replay_buffer, shared_storage) for _ in range(config.refresh_actors)
        ]
        tasks += [worker.run_loop.remote() for worker in refresh_workers]

    # reanalyze workers
    reanalyze_workers = [
        RemoteReanalyzeWorker.options(
            num_gpus=num_gpus_per_worker if config.reanalyze_on_gpu else 0,
        ).remote(
            idx, config, shared_storage, replay_buffer, batch_storage,
        ) for idx in range(config.reanalyze_actors + config.reanalyze_update_actors)
    ]
    tasks += [worker.run_loop.remote() for worker in reanalyze_workers[:config.reanalyze_actors]]
    tasks += [worker.update_loop.remote() for worker in reanalyze_workers[config.reanalyze_actors:]]

    # add handles to global variables
    global remote_worker_handles
    remote_worker_handles += data_workers
    remote_worker_handles.append(test_worker)

    # training loop
    final_weights = _train(model, target_model, replay_buffer, shared_storage, batch_storage, config, summary_writer)

    ray.wait(tasks)
    train_logger.info('Training over...')

    return model, final_weights


def train_sync_serial(config: BaseConfig, summary_writer, model_path=None):
    assert config.data_actors == 1, 'Sync training only support 1 data collector!'

    ''' initialize model '''

    model = config.get_uniform_network()
    target_model = config.get_uniform_network()
    recent_weights = model.get_weights()
    target_model.set_weights(recent_weights)
    if model_path:
        train_logger.info('resume model from path: {}'.format(model_path))
        weights = torch.load(model_path)
        model.load_state_dict(weights)
        target_model.load_state_dict(weights)

    ''' initialize workers '''

    shared_storage = SharedStorage(model, target_model)
    replay_buffer = ReplayBuffer(config)

    data_worker = DataWorker(0, config, replay_buffer, shared_storage)
    data_worker.update_model(0, model.get_weights())

    reanalyze_worker = ReanalyzeWorker(0, config)
    reanalyze_worker.update_model(0, recent_weights)

    test_worker = TestWorker(config)

    ''' initialize training utils '''

    device = 'cuda' if (config.train_on_gpu and torch.cuda.is_available()) else 'cpu'
    model = model.to(device)
    target_model = target_model.to(device)
    model.train()
    target_model.eval()
    optimizer = optim.Adam(model.parameters(), lr=config.lr, eps=config.opti_eps,
                           weight_decay=config.weight_decay)
    scaler = GradScaler()

    ''' fidelity-v1: periodic in-training world-model fidelity (2026-07-20) '''
    # Same test_interval cadence as test_worker below, so the reward and
    # fidelity curves share an x-axis. Guarded by case=="relation" rather
    # than imported unconditionally: this file is the generic vendored
    # fork's training loop (shared with non-relation env families upstream),
    # and fidelity.py / RelationCommonsPettingZooEnv are hyper_mve-specific —
    # same layering DataWorker already uses for its own relation-only
    # reference-episode-injection feature. None (a no-op) for any other
    # case, or if env_cfg_override wasn't set (evaluate()-only construction
    # paths, e.g. _lazy_model()).
    run_fidelity_probe = None
    if getattr(config, "case", None) == "relation":
        env_cfg = getattr(config, "env_cfg_override", None)
        if env_cfg is not None:
            from hyper_mve.envs.adapters.pettingzoo_wrapper import (
                RelationCommonsPettingZooEnv,
            )
            from hyper_mve.utils.eval.fidelity import compute_fidelity_report
            from hyper_mve.utils.schemas import get_regime_family
            from core.test import predict_rewards_from_model

            fidelity_grid = tuple(range(get_regime_family(env_cfg).size))

            def _fidelity_env_fn():
                return RelationCommonsPettingZooEnv(
                    env_cfg, oracle_mode=False, eval_info_mode=False)

            class _ModelPredictRewards:
                """Duck-types compute_fidelity_report's runner arg (just
                needs .predict_rewards) around the LIVE model -- a plain
                function wouldn't satisfy that contract."""

                def predict_rewards(self, episode):
                    return predict_rewards_from_model(
                        model, episode, next(model.parameters()).device)

            _fidelity_hook = _ModelPredictRewards()

            def run_fidelity_probe():
                report = compute_fidelity_report(
                    _fidelity_hook, _fidelity_env_fn, fidelity_grid,
                    episodes=2, seed=1234)
                if report is not None:
                    summary_writer.add_scalar(
                        "fidelity/reward_mae", float(report["reward_mae"]),
                        step_count)
                    for g, v in report["reward_mae_per_regime"].items():
                        summary_writer.add_scalar(
                            f"fidelity/reward_mae_regime_{g}", float(v),
                            step_count)

    ''' per-regime periodic reward probe (2026-07-24): the fork's test_worker
    logs only a pooled, agent-AVERAGED test/mean_score -- not comparable to the
    external baselines' probe, which logs per-regime, agent-SUMMED
    eval/return_mean + eval/return_regime_{g}. This probe closes that gap so the
    mazero reward curve overlays the baselines on the shared (env-steps) axis.
    Same cadence + guard as the fidelity probe. One mixed-regime batch search
    (reward_episodes per regime) keeps the eval cheap. '''
    run_reward_probe = None
    if getattr(config, "case", None) == "relation":
        env_cfg = getattr(config, "env_cfg_override", None)
        if env_cfg is not None:
            from hyper_mve.utils.schemas import get_regime_family as _grf

            reward_grid = tuple(range(_grf(env_cfg).size))
            _train_ids = getattr(env_cfg, "train_regime_ids", None)
            reward_seen_ids = (set(reward_grid) if _train_ids is None
                               else {int(i) for i in _train_ids})
            reward_probe_rng = np.random.RandomState(2024)

            def run_reward_probe(reward_episodes=2):
                # test() sets model.eval() + model.to(device); pass the LIVE
                # device so it stays on GPU, and restore train mode after.
                was_training = model.training
                probe_device = next(model.parameters()).device
                regimes_per_env = [g for g in reward_grid
                                   for _ in range(reward_episodes)]
                try:
                    tlog, _ = test(
                        config, model, step_count, len(regimes_per_env),
                        np_random=reward_probe_rng, pin_g=regimes_per_env,
                        sum_agents=True, verbose=False, device=probe_device)
                finally:
                    if was_training:
                        model.train()
                scores = tlog['scores']
                per_regime = {}
                for idx, g in enumerate(regimes_per_env):
                    per_regime.setdefault(int(g), []).append(float(scores[idx]))
                per_regime = {g: float(np.mean(v)) for g, v in per_regime.items()}
                for g, v in per_regime.items():
                    summary_writer.add_scalar(f"eval/return_regime_{g}", v, step_count)
                vals = list(per_regime.values())
                summary_writer.add_scalar(
                    "eval/return_mean", float(np.mean(vals)) if vals else 0.0,
                    step_count)
                seen = [v for g, v in per_regime.items() if g in reward_seen_ids]
                unseen = [v for g, v in per_regime.items() if g not in reward_seen_ids]
                if seen:
                    summary_writer.add_scalar(
                        "eval/return_seen", float(np.mean(seen)), step_count)
                if unseen:
                    summary_writer.add_scalar(
                        "eval/return_unseen", float(np.mean(unseen)), step_count)

    ''' training loop '''

    transitions_collected = 0
    start_training = False
    step_count = 0
    next_probe_env = 0
    timer = Timer()

    while step_count < config.training_steps + config.last_steps:

        # Run for a whole episode at a time
        if transitions_collected < config.total_transitions:
            timer.start('collect')
            data_worker.update_model(step_count, model.get_weights())
            transitions_collected += data_worker.run(start_training, step_count)
            timer.stop('collect')

        if replay_buffer.can_sample(config.batch_size):
            if not start_training:
                start_training = True
                train_logger.info('Begin training...')

            # compute training steps under current transitions_collected
            if step_count < config.training_steps:
                target_steps = int(config.training_steps * transitions_collected / config.total_transitions)
            else:
                target_steps = config.training_steps + config.last_steps

            while step_count < target_steps:

                # obtain a batch
                timer.start('prepare')
                beta = reanalyze_worker.beta_schedule.value(step_count)
                buffer_context = replay_buffer.prepare_batch_context(config.batch_size, beta)
                batch_context = reanalyze_worker.make_batch(buffer_context, transitions_collected)
                timer.stop('prepare')

                timer.start('update')
                train_logs, new_priority_data = update_weights(config, step_count, model, batch_context, optimizer, scaler, device)

                # update priority if no refresher
                if config.use_priority:
                    indices, new_priority = new_priority_data
                    replay_buffer.update_priorities(indices, new_priority)

                # update model for reanalyzing
                if step_count % config.target_model_interval == 0:
                    reanalyze_worker.update_model(step_count, recent_weights)
                    recent_weights = model.get_weights()

                # save models
                if step_count % config.save_interval == 0:
                    model_path = os.path.join(config.model_dir, 'model_{}.p'.format(step_count))
                    torch.save(model.state_dict(), model_path)
                timer.stop('update')

                # evaluation
                timer.start('eval')
                if step_count % config.test_interval == 0:
                    test_worker.update_model(step_count, model.get_weights())
                    test_log, eval_steps = test_worker.run()
                    shared_storage.add_test_logs(test_log)
                # 2026-07-28: the eval/* + fidelity/* curves fire on ENV steps,
                # not gradient steps. Gating them on step_count gave every
                # algorithm a different curve resolution (m3w 800 env-steps per
                # point, this fork 3200, mamba/happo 20000, mbom 80000) because
                # each does a different number of updates per env step -- and
                # the canonical x-axis is env steps. A fixed env-step cadence
                # puts every algorithm on the SAME grid (5000 -> 200 points).
                if transitions_collected >= next_probe_env:
                    next_probe_env = ((transitions_collected // _PROBE_EVERY_ENV_STEPS)
                                      + 1) * _PROBE_EVERY_ENV_STEPS
                    if run_fidelity_probe is not None:
                        run_fidelity_probe()
                    if run_reward_probe is not None:
                        run_reward_probe()
                timer.stop('eval')

                train_logs['Tc_perstep'] = timer.sum('collect') / (step_count + 1)
                train_logs['Tp_perstep'] = timer.sum('prepare') / (step_count + 1)
                train_logs['Tu_perstep'] = timer.sum('update') / (step_count + 1)
                train_logs['Te_perstep'] = timer.sum('eval') / (step_count + 1)

                # logging
                if step_count % config.log_interval == 0:
                    _log(config, step_count, train_logs, replay_buffer, shared_storage, summary_writer)

                step_count += 1
        else:
            train_logger.debug(f'ReplayBufferSize:{transitions_collected}/{config.batch_size}')

    train_logger.info('Training over...')
    data_worker.close()
    test_worker.close()
    model.eval()
    test_log, eval_steps = test(config, model, step_count, config.test_episodes)
    test_msg = "#{:<10} Test Mean Score of {}: {:<10} (max: {:<10}, min:{:<10}, std: {:<10})" \
               "".format(step_count, config.env_name, test_log["mean_score"], test_log["max_score"], test_log["min_score"], test_log["std_score"])
    logging.getLogger("train_test").info(test_msg)

    return model, model.get_weights()


def train_sync_parallel(config: BaseConfig, summary_writer, model_path=None):
    assert config.data_actors == 1, 'Sync training only support 1 data collector!'

    ''' initialize model '''

    model = config.get_uniform_network()
    target_model = config.get_uniform_network()
    recent_weights = model.get_weights()
    target_model.set_weights(recent_weights)
    if model_path:
        train_logger.info('resume model from path: {}'.format(model_path))
        weights = torch.load(model_path)
        model.load_state_dict(weights)
        target_model.load_state_dict(weights)

    ''' initialize workers '''

    assert ray.is_initialized(), "Must invoke ray.init when parallel"
    availabel_gpus = ray.available_resources().get('GPU', 0)
    num_gpu_workers = 2  # train + test
    if config.selfplay_on_gpu:
        num_gpu_workers += config.data_actors
    if config.reanalyze_on_gpu:
        num_gpu_workers += config.reanalyze_actors
    num_gpus_per_worker = 1 / math.ceil(num_gpu_workers / availabel_gpus)

    # shared_storage = RemoteShareStorage.remote(model, target_model)
    # replay_buffer = RemoteReplayBuffer.remote(config)

    # data_worker = RemoteDataWorker.options(
    #     num_gpus=num_gpus_per_worker if config.selfplay_on_gpu else 0
    # ).remote(0, config, replay_buffer, shared_storage)
    # data_worker.update_model.remote(0, model.get_weights())
    # data_handle = data_worker.run.remote(False, 0)

    shared_storage = SharedStorage(model, target_model)
    replay_buffer = ReplayBuffer(config)

    data_worker = DataWorker(0, config, replay_buffer, shared_storage)
    data_worker.update_model(0, model.get_weights())

    reanalyze_workers = [
        RemoteReanalyzeWorker.options(
            num_gpus=num_gpus_per_worker if config.reanalyze_on_gpu else 0
        ).remote(
            idx, config, None, None, None,  # manually control data sync
        ) for idx in range(config.reanalyze_actors)
    ]
    for worker in reanalyze_workers:
        worker.update_model.remote(0, recent_weights)

    test_worker = RemoteTestWorker.options(num_gpus=num_gpus_per_worker).remote(config, shared_storage)
    test_worker.update_model.remote(0, model.get_weights())
    test_handle = test_worker.run.remote()

    global remote_worker_handles
    # remote_worker_handles.append(data_worker)
    remote_worker_handles.append(test_worker)

    ''' initialize training utils '''

    device = 'cuda' if (config.train_on_gpu and torch.cuda.is_available()) else 'cpu'
    model = model.to(device)
    target_model = target_model.to(device)
    model.train()
    target_model.eval()
    optimizer = optim.Adam(model.parameters(), lr=config.lr, eps=config.opti_eps, weight_decay=config.weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-4)
    scaler = GradScaler()

    ''' training loop '''

    transitions_collected = 0
    start_training = False
    step_count = 0
    timer = Timer()

    while step_count < config.training_steps + config.last_steps:

        # Run for a whole episode at a time
        if transitions_collected < config.total_transitions:
            timer.start('collect')
            # transitions_collected += ray.get(data_handle)
            # data_worker.update_model.remote(step_count, model.get_weights())
            # data_handle = data_worker.run.remote(start_training, step_count)
            data_worker.update_model(step_count, model.get_weights())
            transitions_collected += data_worker.run(start_training, step_count)
            timer.stop('collect')

        # if ray.get(replay_buffer.can_sample.remote(config.batch_size)):
        if replay_buffer.can_sample(config.batch_size):
            if not start_training:
                start_training = True
                train_logger.info('Begin training...')

            # compute training steps under current transitions_collected
            if step_count < config.training_steps:
                target_steps = int(config.training_steps * transitions_collected / config.total_transitions)
            else:
                target_steps = config.training_steps + config.last_steps

            # assign target batch to reanalyze-workers
            beta_lst = ray.get([reanalyze_workers[i % config.reanalyze_actors].get_beta.remote(step_count + i) for i in range(target_steps - step_count)])
            # buffer_context_deque = deque([replay_buffer.prepare_batch_context.remote(config.batch_size, beta) for beta in beta_lst])
            buffer_context_deque = deque([replay_buffer.prepare_batch_context(config.batch_size, beta) for beta in beta_lst])
            batch_context_deque = deque()
            for idle_idx, _ in zip(range(config.reanalyze_actors), beta_lst):
                buffer_context = buffer_context_deque.popleft()
                batch_context_deque.append((idle_idx, reanalyze_workers[idle_idx].make_batch.remote(buffer_context, transitions_collected)))

            while step_count < target_steps:

                # obtain a batch
                timer.start('prepare')
                idle_idx, batch_handle = batch_context_deque.popleft()
                if len(buffer_context_deque) > 0:
                    buffer_context = buffer_context_deque.popleft()
                    batch_context_deque.append((idle_idx, reanalyze_workers[idle_idx].make_batch.remote(buffer_context, transitions_collected)))
                batch_context = ray.get(batch_handle)
                timer.stop('prepare')

                timer.start('update')
                train_logs, new_priority_data = update_weights(config, step_count, model, batch_context, optimizer, scaler, device)

                # update priority if no refresher
                if config.use_priority:
                    indices, new_priority = new_priority_data
                    # replay_buffer.update_priorities.remote(indices, new_priority)
                    replay_buffer.update_priorities(indices, new_priority)

                # update model for reanalyzing
                if step_count % config.target_model_interval == 0:
                    for worker in reanalyze_workers:
                        worker.update_model.remote(step_count, recent_weights)
                    recent_weights = model.get_weights()

                # save models
                if step_count % config.save_interval == 0:
                    model_path = os.path.join(config.model_dir, 'model_{}.p'.format(step_count))
                    torch.save(model.state_dict(), model_path)
                timer.stop('update')

                # evaluation
                timer.start('eval')
                if step_count % config.test_interval == 0 and step_count > 0:
                    test_log, eval_steps = ray.get(test_handle)
                    # shared_storage.add_test_logs.remote(test_log)
                    shared_storage.add_test_logs(test_log)
                    test_worker.update_model.remote(step_count, model.get_weights())
                    test_handle = test_worker.run.remote()
                timer.stop('eval')

                train_logs['Tc_perstep'] = timer.sum('collect') / (step_count + 1)
                train_logs['Tp_perstep'] = timer.sum('prepare') / (step_count + 1)
                train_logs['Tu_perstep'] = timer.sum('update') / (step_count + 1)
                train_logs['Te_perstep'] = timer.sum('eval') / (step_count + 1)

                # logging
                if step_count % config.log_interval == 0:
                    _log(config, step_count, train_logs, replay_buffer, shared_storage, summary_writer)

                step_count += 1
        else:
            train_logger.debug(f'ReplayBufferSize:{transitions_collected}/{config.batch_size}')

    train_logger.info('Training over...')
    # data_worker.close.remote()
    data_worker.close()
    test_worker.close.remote()
    model.eval()
    test_logs, eval_steps = test(config, model, step_count, config.test_episodes)
    test_msg = '#{:<10} Test Mean Score of {}: {:<10} (max: {:<10}, min:{:<10}, std: {:<10})' \
               ''.format(test_logs['test_counter'], config.env_name, test_logs["mean_score"], test_logs["max_score"], test_logs["min_score"], test_logs["std_score"])
    if 'win_rate' in test_logs:
        test_msg += ' | WinRate: {:.2f}'.format(test_logs['win_rate'])
    logging.getLogger("train_test").info(test_msg)

    return model, model.get_weights()
