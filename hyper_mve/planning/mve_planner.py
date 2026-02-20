"""
Per-Agent Coordinate Descent MVE Planner (v4.6).

For each agent in randomised order (coordinate descent):
    1. Pre-sample other agents' step-0 actions once per scenario (CRN)
    2. Enumerate all A candidate first-actions for this agent
    3. Replicate other agents' step-0 actions across all A candidates (noise cancels)
    4. For step>0: all agents sample from policy network (independent)
    5. Rollout K steps, accumulate agent j's discounted returns + terminal value
    6. π_mve[j] = softmax(mean_return_per_action / temperature)

v4.6 key fix — Common Random Numbers (CRN):
    When evaluating A candidate actions for agent j, other agents' step-0 actions
    are sampled ONCE per scenario and SHARED across all A candidates. This eliminates
    the dominant noise source (other agents' random actions) from the return difference
    between candidates, allowing the planner to detect the true signal.

    Without CRN: SNR ≈ 0.02 → softmax → uniform distribution
    With CRN:    noise cancels → SNR → ∞ for step 0

    Data layout change:
        Before (v4.5): candidate outer, sample inner → (B, A, spa)
        After  (v4.6): scenario outer, candidate inner → (B, spa, A)

Supports BaselineModel, OracleHyperMuZeroModel, InferHyperMuZeroModel via duck-typing.
For HyperMuZero: set_context() is called before each agent's perspective.
"""
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from utils.utils import actions_to_one_hot, inverse_scalar_transform


def _is_hyper_model(model):
    """Check if model is a HyperMuZero model (has set_context method)."""
    return hasattr(model, 'set_context')


def _expand_rule(rule, repeats):
    """Expand rule tensor for parallel sampling, handling both scalar and embedding forms."""
    if rule is None:
        return None
    # Works for both (B,) scalar and (B, rule_emb_dim) embedding
    return rule.repeat_interleave(repeats, dim=0)


def _sample_policy_action(model, curr_s, agent_idx, batch_size, device, is_hyper, rule_exp):
    """Sample action for agent_idx from model policy.

    Args:
        model:      model instance
        curr_s:     (batch_size, latent_dim) current state
        agent_idx:  int, which agent
        batch_size: int, leading dimension of curr_s
        device:     torch device
        is_hyper:   bool, whether model uses set_context
        rule_exp:   expanded rule tensor (batch_size, ...) or None

    Returns:
        a_i: (batch_size,) sampled actions
    """
    id_i = torch.full((batch_size,), agent_idx, dtype=torch.long, device=device)
    if is_hyper:
        model.set_context(rule_exp, id_i)
        logits_i, _ = model.predict(curr_s)
    else:
        id_emb_i = model.get_id_emb(id_i)
        logits_i, _ = model.predict(curr_s, id_emb_i)
    return Categorical(logits=logits_i).sample()


@torch.no_grad()
def sample_mve_plan(model, root_s, cfg, rule=None):
    """
    Per-agent coordinate descent MVE planning with Common Random Numbers (CRN).

    For each agent j (in random order):
        Phase 1: Pre-sample other agents' step-0 actions (once per scenario)
        Phase 2: Expand to (B*M,) — replicate across A candidates
        Phase 3: Rollout K steps with CRN at step 0, independent at step>0
        Phase 4: Aggregate: (B, spa, A) → mean over spa → softmax

    Args:
        model:   BaselineModel or OracleHyperMuZeroModel or InferHyperMuZeroModel
        root_s:  (B, latent_dim) - current latent state
        cfg:     config with mve_samples, mve_depth, num_agents, num_actions, gamma
        rule:    (B,) float or (B, rule_emb_dim) — required for HyperMuZero, ignored for Baseline

    Returns:
        pi_mve: (B, num_agents, num_actions) - search policy distribution per agent
    """
    B = root_s.shape[0]
    N = cfg.num_agents
    A = cfg.num_actions
    K = cfg.mve_depth
    S = cfg.mve_samples
    gamma = cfg.gamma
    temperature = getattr(cfg, 'mve_temperature', 1.0)
    device = root_s.device
    is_hyper = _is_hyper_model(model)

    spa = S // A          # samples per candidate action (scenarios)
    M = A * spa           # total trajectories per batch element per agent

    # Random agent ordering for coordinate descent
    agent_order = torch.randperm(N).tolist()

    # Output: per-agent search policy
    pi_mve = torch.zeros(B, N, A, device=device)
    optimised = set()

    for j in agent_order:
        # ── Phase 1: Pre-sample other agents' step-0 actions (CRN) ─────
        # Sample once per scenario at (B*spa,) granularity.
        # All scenarios start from the same root_s (per batch element),
        # so policy distributions are identical — pre-sampling is valid.
        s_scenarios = root_s.repeat_interleave(spa, dim=0)  # (B*spa, latent)
        rule_scenarios = _expand_rule(rule, spa) if is_hyper else None
        B_spa = B * spa

        step0_actions_per_scenario = {}  # agent_i -> (B*spa,) actions
        for i in range(N):
            if i == j:
                continue  # agent j will be enumerated
            if i in optimised:
                # Already-optimised: sample from its π_mve, once per scenario
                probs_i = pi_mve[:, i].repeat_interleave(spa, dim=0)  # (B*spa, A)
                step0_actions_per_scenario[i] = Categorical(probs=probs_i).sample()
            else:
                # Not-yet-optimised: sample from policy network, once per scenario
                step0_actions_per_scenario[i] = _sample_policy_action(
                    model, s_scenarios, i, B_spa, device, is_hyper, rule_scenarios
                )

        # ── Phase 2: Expand to (B*M,) = (B*spa*A,) ────────────────────
        # Layout: [b0_sc0_a0, b0_sc0_a1, ..., b0_sc0_a4,
        #          b0_sc1_a0, ..., b0_sc1_a4,
        #          ...,
        #          b0_sc9_a0, ..., b0_sc9_a4,
        #          b1_sc0_a0, ...]
        # scenario outer, candidate inner
        s_exp = s_scenarios.repeat_interleave(A, dim=0)  # (B*M, latent)
        rule_exp = _expand_rule(rule_scenarios, A) if is_hyper else None
        BM = B * M

        cum_return_j = torch.zeros(BM, device=device)
        discount = 1.0
        curr_s = s_exp

        # Agent j's candidate first-actions: [0,1,2,3,4, 0,1,2,3,4, ...]
        # Each group of A corresponds to one scenario, all sharing same other-agent actions
        first_action_j = torch.arange(A, device=device).repeat(B * spa)  # (B*M,)

        # Expand other agents' step-0 actions: replicate each scenario action A times
        step0_actions_expanded = {}
        for i, a_scenario in step0_actions_per_scenario.items():
            step0_actions_expanded[i] = a_scenario.repeat_interleave(A, dim=0)  # (B*M,)

        # ── Phase 3: Rollout K steps ──────────────────────────────────
        for step in range(K):
            all_actions = []
            for i in range(N):
                if step == 0 and i == j:
                    # Current agent, step 0: enumerated candidate (deterministic)
                    a_i = first_action_j
                elif step == 0 and i in step0_actions_expanded:
                    # Other agent, step 0: CRN — same action within each scenario
                    a_i = step0_actions_expanded[i]
                else:
                    # Step > 0: independent sampling (states have diverged)
                    a_i = _sample_policy_action(
                        model, curr_s, i, BM, device, is_hyper, rule_exp
                    )
                all_actions.append(a_i)

            joint_actions = torch.stack(all_actions, dim=-1)        # (B*M, N)
            action_onehot = actions_to_one_hot(joint_actions, A)    # (B*M, N*A)

            # ── Objective state transition ──────────────────────────
            if is_hyper:
                # θ_state depends only on rule (objective), any agent_id works
                id_0 = torch.zeros(BM, dtype=torch.long, device=device)
                model.set_context(rule_exp, id_0)
            s_next = model.transition(curr_s, action_onehot)

            # ── Subjective reward for agent j only ──────────────────
            id_j = torch.full((BM,), j, dtype=torch.long, device=device)
            if is_hyper:
                model.set_context(rule_exp, id_j)
                r_j_scaled = model.predict_reward(curr_s, action_onehot)
            else:
                id_emb_j = model.get_id_emb(id_j)
                r_j_scaled = model.predict_reward(curr_s, action_onehot, id_emb_j)

            r_j = inverse_scalar_transform(r_j_scaled).squeeze(-1)  # (B*M,)
            cum_return_j += discount * r_j

            curr_s = s_next
            discount *= gamma

        # ── Phase 3b: Terminal value for agent j ───────────────────
        id_j = torch.full((BM,), j, dtype=torch.long, device=device)
        if is_hyper:
            model.set_context(rule_exp, id_j)
            _, v_j_scaled = model.predict(curr_s)
        else:
            id_emb_j = model.get_id_emb(id_j)
            _, v_j_scaled = model.predict(curr_s, id_emb_j)

        v_j = inverse_scalar_transform(v_j_scaled).squeeze(-1)  # (B*M,)
        cum_return_j += discount * v_j

        # ── Phase 4: Aggregate with CRN layout ────────────────────
        # CRN layout: (B*M,) → (B, spa, A) → mean over scenarios → (B, A)
        returns_per_action = cum_return_j.view(B, spa, A).mean(dim=1)
        pi_mve[:, j] = F.softmax(returns_per_action / temperature, dim=-1)

        optimised.add(j)

    return pi_mve  # (B, N, A) soft probability distribution
