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


def _role_l2(thetas, type_assignment):
    """[v4-opt 2026-06c] P2.2: pairwise L2 distance of per-agent hypernet params.

    Disambiguates 'rising cos_rew_cross' (which can mean either 'genuine direction
    convergence = role collapse' or 'same direction with growing magnitude offsets')
    by exposing the raw L2 norm of the difference. Combined with the cross-type
    cosine: high cosine + rising L2 = different magnitudes, same direction (head
    sharing structurally OK); high cosine + flat L2 = role collapse.

    Same NaN semantics as :func:`_role_cosine` (NaN when a category has no pairs).
    """
    N = len(thetas)
    nan = torch.tensor(float("nan"), device=thetas[0].device)
    cross, same = [], []
    for i in range(N):
        for j in range(i + 1, N):
            d = (thetas[i] - thetas[j]).norm(dim=-1).mean()
            (same if type_assignment[i] == type_assignment[j] else cross).append(d)
    cross_v = torch.stack(cross).mean() if cross else nan
    same_v = torch.stack(same).mean() if same else nan
    return cross_v, same_v


def _per_type_norm(thetas, type_assignment, target_type_int: int):
    """[v4-opt 2026-06c] P2.2: mean L2 norm of generated params over agents of one type.

    Returns a scalar tensor (mean over matching agents then over batch) or NaN if
    no agent matches ``target_type_int`` (e.g. an all-ALPHA configuration probed
    for BETA).
    """
    matching = [
        thetas[i].norm(dim=-1).mean()
        for i in range(len(thetas))
        if int(type_assignment[i]) == target_type_int
    ]
    if not matching:
        return torch.tensor(float("nan"), device=thetas[0].device)
    return torch.stack(matching).mean()


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

    # [v4-opt 2026-06] policy-target mask: planner_on=False episodes carry
    # self-distillation pi_mve (model's own prior, e.g. buffer warmup) — training
    # the policy CE on them is the Ch5.9.1b zero-information fixed point, so they
    # are excluded from L_policy (value/reward/consistency still train on them).
    # Absent key (direct callers / legacy tests) ⇒ all-ones mask (old behaviour).
    planner_on = batch.get("planner_on")
    if planner_on is None:
        planner_mask = torch.ones(obs.shape[0], device=device)
    else:
        planner_mask = planner_on.to(device).float()             # (B,)
    n_planner = planner_mask.sum().clamp(min=1.0)

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

    # [v4-opt 2026-06c] P2.1: distillation diagnostics (KL(π_mve ‖ π_pred) — the
    # direction the policy CE loss already minimises — plus argmax match rate and
    # separate sharpness probes for both distributions). All masked by planner_on
    # for consistency with the policy loss; NaN if the batch contains no planner-on
    # sample.  These pin down whether pi_mve's entropy plateau is a *target* problem
    # (mve_max_prob low) or a *prior* problem (mve_max_prob high but pred fails to
    # follow), and whether stalled KL is "two equally soft distributions" (kl ≈ const
    # but mve_max_prob low) or "real distillation".
    KL_mve_to_pred = torch.zeros((), device=device)
    mode_match = torch.zeros((), device=device)
    pi_mve_max_prob = torch.zeros((), device=device)
    pi_pred_max_prob = torch.zeros((), device=device)

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
            ce = -(target_pi * F.log_softmax(p_k, dim=-1)).sum(-1)        # (B,)
            # masked mean over planner-on samples [v4-opt 2026-06]
            L_policy = L_policy + (ce * planner_mask).sum() / n_planner

            target_r = scalar_transform(rewards[:, k, agent].unsqueeze(-1))      # (B, 1)
            L_reward = L_reward + F.mse_loss(r_k, target_r)

            target_z = scalar_transform(z_return[:, k, agent].unsqueeze(-1)).squeeze(-1)
            L_value = L_value + F.mse_loss(v_k.squeeze(-1), target_z)

            # --- diagnostics (detached): predict-net entropy + k=0 generated params ---
            probs_pred = F.softmax(p_k.detach(), dim=-1)
            H_pi_pred = H_pi_pred + -(probs_pred * torch.log(probs_pred + 1e-9)).sum(-1).mean()

            # [v4-opt 2026-06c] P2.1: per-(k, agent) distillation diagnostics, masked
            # to planner_on rows so warmup self-distillation entries (target_pi == pred)
            # don't bias the KL/mode-match estimate. Direction: KL(π_mve ‖ π_pred) —
            # the same direction the policy CE loss gradient walks (L_policy =
            # H(π_mve) + KL(π_mve ‖ π_pred); falls to 0 iff π_pred == π_mve). This
            # also matches the candidate distillation auxiliary in P3.2, so a future
            # KL-loss experiment can read its own loss off this scalar.
            with torch.no_grad():
                pi_mve_detached = target_pi.detach()
                # add tiny floors before log to avoid -inf where either distribution
                # has a perfectly zero entry (rare in practice).
                log_mve = (pi_mve_detached + 1e-9).log()
                log_pred = probs_pred.clamp_min(1e-9).log()
                kl_row = (pi_mve_detached * (log_mve - log_pred)).sum(-1)  # (B,)
                mm_row = (probs_pred.argmax(-1) == pi_mve_detached.argmax(-1)).float()
                pmve_row = pi_mve_detached.max(-1).values                 # (B,)
                ppred_row = probs_pred.max(-1).values                     # (B,)
                KL_mve_to_pred = KL_mve_to_pred + (kl_row * planner_mask).sum() / n_planner
                mode_match = mode_match + (mm_row * planner_mask).sum() / n_planner
                pi_mve_max_prob = pi_mve_max_prob + (pmve_row * planner_mask).sum() / n_planner
                pi_pred_max_prob = pi_pred_max_prob + (ppred_row * planner_mask).sum() / n_planner

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
    # [v4-opt 2026-06] masked to planner-on samples (warmup priors would skew it);
    # NaN when the batch holds no planner-on sample (TB writer skips NaN).
    pim = pi_mve[:, :K].clamp_min(1e-9)                          # (B, K, N, A)
    H_per_sample = -(pim * pim.log()).sum(-1).mean(dim=(1, 2))   # (B,)
    if float(planner_mask.sum()) > 0:
        H_pi_mve = ((H_per_sample * planner_mask).sum() / planner_mask.sum()).detach()
    else:
        H_pi_mve = torch.tensor(float("nan"), device=device)
    # hypernet role discrimination (k=0 generated params): cross-type vs same-type cosine.
    cos_pred_cross, cos_pred_same = _role_cosine(theta_pred_per_agent, cfg.env.type_assignment)
    cos_rew_cross, cos_rew_same = _role_cosine(theta_rew_per_agent, cfg.env.type_assignment)

    # [v4-opt 2026-06c] P2.2: reward-hypernet L2 distance + per-type norms. Cosine
    # alone cannot distinguish "same direction, different magnitude" (acceptable —
    # shared head structurally OK with α/β reward magnitude offsets) from "genuine
    # role collapse" (failure mode). L2 disambiguates.
    l2_rew_cross, l2_rew_same = _role_l2(theta_rew_per_agent, cfg.env.type_assignment)
    # AgentType.ALPHA = 0, AgentType.BETA = 1 (see schemas/_constants.py).
    norm_rew_alpha = _per_type_norm(theta_rew_per_agent, cfg.env.type_assignment, 0)
    norm_rew_beta = _per_type_norm(theta_rew_per_agent, cfg.env.type_assignment, 1)

    # [v4-opt 2026-06c] P2.1: finalize distillation diagnostics (KN-mean).
    if float(planner_mask.sum()) > 0:
        diag_kl_mve_to_pred = (KL_mve_to_pred / KN).detach()
        diag_mode_match = (mode_match / KN).detach()
        diag_pi_mve_max_prob = (pi_mve_max_prob / KN).detach()
        diag_pi_pred_max_prob = (pi_pred_max_prob / KN).detach()
    else:
        nan = torch.tensor(float("nan"), device=device)
        diag_kl_mve_to_pred = nan
        diag_mode_match = nan
        diag_pi_mve_max_prob = nan
        diag_pi_pred_max_prob = nan

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
        # [v4-opt 2026-06c] P2.2: reward-hypernet L2 + per-type norms — disambiguates
        # 'same direction, different magnitude' (acceptable) from genuine role
        # collapse (failure). NaN when a type has no agent or only one (e.g. duo).
        "diag_l2_rew_cross": l2_rew_cross,
        "diag_l2_rew_same": l2_rew_same,
        "diag_norm_rew_alpha": norm_rew_alpha,
        "diag_norm_rew_beta": norm_rew_beta,
        # [v4-opt 2026-06c] P2.1: distillation health — KL(π_mve ‖ π_pred) (the
        # direction the policy CE loss minimises), argmax match rate, and *separate*
        # mode-mass (max prob) probes for both distributions so a stalled KL can be
        # disambiguated as either 'two equally soft distributions' (both max-probs
        # low) or 'real distillation closing' (mode-match high, max-probs both rise).
        "diag_kl_mve_to_pred": diag_kl_mve_to_pred,
        "diag_mode_match_pred_mve": diag_mode_match,
        "diag_pi_mve_max_prob": diag_pi_mve_max_prob,
        "diag_pi_pred_max_prob": diag_pi_pred_max_prob,
    }
