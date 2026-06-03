"""compose_total_loss — main + L_belief double-path assembly (Pkg-05 spec 05, Ch5.6/5.8).

D5: loss assembly lives outside the model (Pkg-04 clarification 1: the model exposes
no ``compute_losses``) and outside the trainer (kept small). Two autograd paths share
one BeliefNet forward:

    BeliefNet.forward(obs)  ->  (hidden, c_hat_pred, z_hat_pred)   [grad]
        |                              |
        |  path A: L_belief            |  path B: main loss
        |  (always trains BeliefNet)   |  set_context_subjective(belief)
        v                              v  -> model.grad_gating detaches when
    belief_loss(predicted)             v     step < belief_grad_gating_steps
                                       v  -> hyper_rew / hyper_pred -> heads

Key reconciliations with the *real* Pkg-03 API (SDD pseudo-code differed):
  * ``BeliefNet.forward`` returns ``(hidden, c_hat, z_hat)`` (hidden first) and has
    **no** ``oracle_mixing_weight`` arg — giving ``oracle_z_seq`` fully replaces z.
    So we forward **without** oracle (predicted z, with grad) for L_belief, and do
    the Stage-2 soft anneal here:  ``z_main = w*oracle_z + (1-w)*z_pred`` (convex on
    the simplex). L_belief always uses the *predicted* z so head_opp keeps training
    even in Stage 1 (Oracle).
  * ``belief_loss`` returns ``(total, breakdown)`` and takes a ``weights`` tuple.
  * RewardHead / value head output **scaled** space (MuZero ``scalar_transform``);
    targets are scaled before MSE.

The v4 path trains **all N agents every step** (set_context_subjective per agent),
unlike v4.7 which sampled one perspective per batch element.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from hyper_mve.configs import V4Config
from hyper_mve.models import HyperMuZeroModel
from hyper_mve.models.belief_losses import belief_loss
from hyper_mve.models.representation_net import negative_cosine_similarity
from hyper_mve.utils.utils import actions_to_one_hot, inverse_scalar_transform, scalar_transform

if TYPE_CHECKING:  # avoid import cycle (trainer imports this module)
    from hyper_mve.training.muzero_trainer import MuZeroTrainer


def _role_cosine(thetas, type_assignment):
    """Mean pairwise cosine of per-agent hypernet-generated params, by role pairing.

    Diagnostic for "can the hypernet distinguish roles?": low cross-type cosine ⇒
    α/β get well-separated parameters; cross ≈ same ≈ 1 ⇒ role collapse.

    Args:
        thetas: list of N tensors (B, P) — generated params per agent (detached).
        type_assignment: length-N sequence of AgentType/int; agents with equal value
            form 'same-type' pairs, others 'cross-type'.
    Returns:
        (cross, same): scalar tensors (mean cosine over pairs then over batch). NaN
        when a category has no pairs (e.g. duo N=2 has no same-type pair).
    """
    N = len(thetas)
    nan = torch.tensor(float("nan"), device=thetas[0].device)
    cross, same = [], []
    for i in range(N):
        for j in range(i + 1, N):
            cs = F.cosine_similarity(thetas[i], thetas[j], dim=-1).mean()
            (same if type_assignment[i] == type_assignment[j] else cross).append(cs)
    cross_v = torch.stack(cross).mean() if cross else nan
    same_v = torch.stack(same).mean() if same else nan
    return cross_v, same_v


def compose_total_loss(
    model: HyperMuZeroModel,
    batch: dict[str, torch.Tensor],
    trainer: "MuZeroTrainer",
    global_step: int,
    cfg: V4Config,
) -> dict[str, torch.Tensor]:
    """Assemble total = main + lambda_b * L_belief (one backward, two paths).

    Args:
        model: online HyperMuZeroModel.
        batch: from ``EpisodeReplayBuffer.sample_batch`` (already moved to device by
            the trainer). Shapes: obs (B,K+1,N,obs_dim), actions/rewards/v (B,K+1,N),
            cap (B,K+1,N,4), pi_mve (B,K+1,N,A), c_t (B,K+1), tau (B,K+1,N) int8,
            dones (B,K+1) bool.
        trainer: provides ``.scheduler`` / ``.target_model`` / ``.projector`` /
            ``.compute_n_step_return``.
        global_step: for curriculum + gating.
        cfg: V4Config.

    Returns:
        dict of loss components; ``total`` / ``main`` / ``belief`` carry grad, the
        rest are detached scalars for logging.
    """
    sched = trainer.scheduler
    target_model = trainer.target_model
    projector = trainer.projector

    K = cfg.train.unroll_K
    N = cfg.env.N
    A = cfg.env.A
    # Make compose device-robust for any caller (train_step already moves the
    # batch; direct callers / tests may pass a CPU batch against a CUDA model).
    device = next(model.parameters()).device
    batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}

    obs = batch["obs"]                       # (B, K+1, N, obs_dim)
    actions = batch["actions"]               # (B, K+1, N) int64
    rewards = batch["rewards"]               # (B, K+1, N)
    cap = batch["cap"]                       # (B, K+1, N, 4)
    pi_mve = batch["pi_mve"]                 # (B, K+1, N, A)
    c_t = batch["c_t"]                       # (B, K+1)
    types_true = batch["tau"].long()         # (B, K+1, N) int64
    dones = batch["dones"]                   # (B, K+1) bool

    # ================================================================
    # Step 1: one BeliefNet forward (no oracle) -> predicted c_hat / z_hat (grad)
    # ================================================================
    hidden_seq, c_hat_pred, z_hat_pred = model.belief_net(obs)
    # hidden_seq (B,K+1,N,128), c_hat_pred (B,K+1,N), z_hat_pred (B,K+1,N,N-1,2)

    # ================================================================
    # Step 2: L_belief from the *predicted* tensors (original graph, no detach)
    # ================================================================
    L_belief, belief_breakdown = belief_loss(
        c_hat_pred, z_hat_pred, hidden_seq,
        c_true_seq=c_t,
        types_true=types_true,
        weights=(cfg.train.w_belief_c, cfg.train.w_belief_opp, cfg.train.w_belief_div),
        div_target_std=cfg.train.belief_div_target_std,
    )

    # ================================================================
    # Step 3: oracle blend for the main path (Stage 2 soft anneal done here)
    # ================================================================
    mixing_w = sched.oracle_z_mixing_weight(global_step)
    if mixing_w > 0.0:
        oracle_z = sched.build_oracle_z_seq(types_true)          # (B,K+1,N,N-1,2)
        z_main = mixing_w * oracle_z + (1.0 - mixing_w) * z_hat_pred
    else:
        z_main = z_hat_pred
    c_main = c_hat_pred  # c never oracle-injected (head_c is MSE-supervised by L_c)

    # ================================================================
    # Step 3.5: target-model V bootstrap (no_grad) for all (step, agent)
    # ================================================================
    c_root = c_t[:, 0]
    model.set_context_objective(c_root)
    target_model.set_context_objective(c_root)

    v_target = torch.zeros(obs.shape[0], K + 1, N, device=device)
    with torch.no_grad():
        for kk in range(K + 1):
            s_tgt = target_model.encode(obs[:, kk])
            for agent in range(N):
                target_model.set_context_subjective(
                    agent, cap[:, kk, agent],
                    (c_main[:, kk, agent].detach(), z_main[:, kk, agent].detach()),
                )
                _, v_s = target_model.predict(s_tgt)             # (B, 1) scaled
                v_target[:, kk, agent] = inverse_scalar_transform(v_s).squeeze(-1)

    # n-step return in ORIGINAL scale, (B, K+1, N)
    z_return = trainer.compute_n_step_return(rewards, v_target, dones, cfg.train.n_step)

    # ================================================================
    # Step 4: main K-step unroll over ALL N agents
    # ================================================================
    s_pred = model.encode(obs[:, 0])                             # (B, latent_dim)

    L_policy = torch.zeros((), device=device)
    L_value = torch.zeros((), device=device)
    L_reward = torch.zeros((), device=device)
    L_consist = torch.zeros((), device=device)

    # --- diagnostics (detached; logging only, no effect on the backward graph) ---
    H_pi_pred = torch.zeros((), device=device)            # predict-net policy entropy
    theta_pred_per_agent: list[torch.Tensor] = []         # k=0 per-agent generated params
    theta_rew_per_agent: list[torch.Tensor] = []

    for k in range(K):
        action_onehot = actions_to_one_hot(actions[:, k], A)     # (B, N*A)
        s_next = model.transition(s_pred, action_onehot)         # objective (theta_state)

        # consistency (objective, once per step): BYOL negative cosine similarity
        if projector is not None:
            proj_pred = projector(s_next)
            with torch.no_grad():
                proj_target = projector(model.encode(obs[:, k + 1]))
            L_consist = L_consist + negative_cosine_similarity(proj_pred, proj_target)

        for agent in range(N):
            # main path belief: predicted c + blended z (grad); model internally
            # detaches when global_step < belief_grad_gating_steps (Pkg-04 spec 04).
            model.set_context_subjective(
                agent, cap[:, k, agent],
                (c_main[:, k, agent], z_main[:, k, agent]),
            )
            p_k, v_k = model.predict(s_pred)                     # (B, A), (B, 1) scaled
            r_k = model.predict_reward(s_pred, action_onehot)    # (B, 1) scaled

            target_pi = pi_mve[:, k, agent]                      # (B, A)
            L_policy = L_policy + -(target_pi * F.log_softmax(p_k, dim=-1)).sum(-1).mean()

            target_r = scalar_transform(rewards[:, k, agent].unsqueeze(-1))      # (B, 1)
            L_reward = L_reward + F.mse_loss(r_k, target_r)

            target_z = scalar_transform(z_return[:, k, agent].unsqueeze(-1)).squeeze(-1)
            L_value = L_value + F.mse_loss(v_k.squeeze(-1), target_z)

            # --- diagnostics (detached): predict-net entropy + k=0 generated params ---
            probs_pred = F.softmax(p_k.detach(), dim=-1)
            H_pi_pred = H_pi_pred + -(probs_pred * torch.log(probs_pred + 1e-9)).sum(-1).mean()
            if k == 0:
                th_rew, th_pred = model.current_subjective_thetas()
                theta_rew_per_agent.append(th_rew.detach())
                theta_pred_per_agent.append(th_pred.detach())

        # gradient half-life: dampen K-step gradient growth (v4.6)
        s_pred = 0.5 * s_next + 0.5 * s_next.detach()

    KN = float(K * N)
    L_policy = L_policy / KN
    L_value = L_value / KN
    L_reward = L_reward / KN
    L_consist = L_consist / float(K) if projector is not None else L_consist

    # --- finalize diagnostics (all detached) ---
    H_pi_pred = (H_pi_pred / KN).detach()
    # planner (pi_mve) entropy over the unrolled window (data target; no grad).
    # ≈ ln(A) ⇒ planner not differentiating; lower ⇒ it favours specific actions.
    pim = pi_mve[:, :K].clamp_min(1e-9)                          # (B, K, N, A)
    H_pi_mve = -(pim * pim.log()).sum(-1).mean().detach()
    # hypernet role discrimination (k=0 generated params): cross-type vs same-type cosine.
    cos_pred_cross, cos_pred_same = _role_cosine(theta_pred_per_agent, cfg.env.type_assignment)
    cos_rew_cross, cos_rew_same = _role_cosine(theta_rew_per_agent, cfg.env.type_assignment)

    L_main = (
        cfg.train.w_policy * L_policy
        + cfg.train.w_value * L_value
        + cfg.train.w_reward * L_reward
        + cfg.train.w_consist * L_consist
    )

    # ================================================================
    # Step 5: total = main + lambda_b * L_belief  (single backward, two paths)
    # ================================================================
    lambda_b = sched.lambda_b(global_step)
    L_total = L_main + lambda_b * L_belief

    return {
        "total": L_total,
        "main": L_main,
        "belief": L_belief,
        "lambda_b": torch.tensor(float(lambda_b)),
        "policy": (cfg.train.w_policy * L_policy).detach(),
        "value": (cfg.train.w_value * L_value).detach(),
        "reward": (cfg.train.w_reward * L_reward).detach(),
        "consist": (cfg.train.w_consist * L_consist).detach(),
        "belief_c": belief_breakdown["l_c"],
        "belief_opp": belief_breakdown["l_opp"],
        "belief_div": belief_breakdown["l_div"],
        # --- raw (unweighted) loss magnitudes (the above are pre-multiplied by w_*) ---
        "L_policy_raw": L_policy.detach(),
        "L_value_raw": L_value.detach(),
        "L_reward_raw": L_reward.detach(),
        "L_consist_raw": L_consist.detach(),
        # --- action-distribution diagnostics ---
        "diag_pi_mve_entropy": H_pi_mve,        # planner differentiation (target ≈ ln A ⇒ uniform)
        "diag_pi_pred_entropy": H_pi_pred,      # predict-net sharpness
        # --- hypernet role-discrimination cosine (lower cross ⇒ roles separated) ---
        "diag_cos_pred_cross": cos_pred_cross,
        "diag_cos_pred_same": cos_pred_same,
        "diag_cos_rew_cross": cos_rew_cross,
        "diag_cos_rew_same": cos_rew_same,
    }
