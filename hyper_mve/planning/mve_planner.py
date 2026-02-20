"""
Per-Agent Coordinate Descent MVE Planner (v4.5).

For each agent in randomised order (coordinate descent):
    1. Enumerate all A candidate first-actions for this agent
    2. For already-optimised agents: sample step-0 action from their π_mve
    3. For remaining agents and all step>0: sample from policy network
    4. Rollout K steps, accumulate agent j's discounted returns + terminal value
    5. π_mve[j] = softmax(mean_return_per_action / temperature)

Key improvement over v4.4:
    - Searches per-agent action space (A=5) instead of joint space (A^N=625)
    - S samples / A actions → meaningful per-candidate signal
    - Coordinate descent: later agents benefit from earlier agents' improvements

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
    Per-agent coordinate descent MVE planning from a root latent state.

    For each agent j (in random order):
        - Enumerate A candidate first-actions (deterministic)
        - Already-optimised agents use their π_mve at step 0
        - All other actions sampled from policy network
        - Accumulate agent j's discounted return over K steps + terminal value
        - π_mve[j] = softmax(mean_return_per_candidate / τ)

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

    spa = S // A          # samples per candidate action
    M = A * spa           # total trajectories per batch element per agent

    # Random agent ordering for coordinate descent
    agent_order = torch.randperm(N).tolist()

    # Output: per-agent search policy
    pi_mve = torch.zeros(B, N, A, device=device)
    optimised = set()

    # Pre-compute candidate first-action indices (reused for every agent j)
    # Layout: for batch b, candidate a, sample s → index b*M + a*spa + s, action = a
    # Shape: (M,) repeated B times → (B*M,)
    candidate_actions_template = torch.arange(A, device=device).repeat_interleave(spa)  # (M,)

    for j in agent_order:
        # ── Expand root state and rule for this agent's search ────────
        s_exp = root_s.repeat_interleave(M, dim=0)         # (B*M, latent)
        rule_exp = _expand_rule(rule, M) if is_hyper else None
        BM = B * M

        cum_return_j = torch.zeros(BM, device=device)
        discount = 1.0
        curr_s = s_exp

        # Agent j's deterministic first action: (B*M,)
        first_action_j = candidate_actions_template.repeat(B)

        for step in range(K):
            # ── 1. Determine actions for all agents ───────────────────
            all_actions = []
            for i in range(N):
                if step == 0 and i == j:
                    # Current agent, step 0: enumerated candidate (deterministic)
                    a_i = first_action_j
                elif step == 0 and i in optimised:
                    # Already-optimised agent, step 0: sample from its π_mve
                    # pi_mve[:, i] is (B, A) → expand to (B*M, A) → sample
                    probs_i = pi_mve[:, i].repeat_interleave(M, dim=0)  # (B*M, A)
                    a_i = Categorical(probs=probs_i).sample()
                else:
                    # Not-yet-optimised agent or step > 0: sample from policy
                    a_i = _sample_policy_action(
                        model, curr_s, i, BM, device, is_hyper, rule_exp
                    )
                all_actions.append(a_i)

            joint_actions = torch.stack(all_actions, dim=-1)        # (B*M, N)
            action_onehot = actions_to_one_hot(joint_actions, A)    # (B*M, N*A)

            # ── 2. Objective state transition ─────────────────────────
            if is_hyper:
                # θ_state depends only on rule (objective), any agent_id works
                id_0 = torch.zeros(BM, dtype=torch.long, device=device)
                model.set_context(rule_exp, id_0)
            s_next = model.transition(curr_s, action_onehot)

            # ── 3. Subjective reward for agent j only ─────────────────
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

        # ── 4. Terminal value for agent j ─────────────────────────────
        id_j = torch.full((BM,), j, dtype=torch.long, device=device)
        if is_hyper:
            model.set_context(rule_exp, id_j)
            _, v_j_scaled = model.predict(curr_s)
        else:
            id_emb_j = model.get_id_emb(id_j)
            _, v_j_scaled = model.predict(curr_s, id_emb_j)

        v_j = inverse_scalar_transform(v_j_scaled).squeeze(-1)  # (B*M,)
        cum_return_j += discount * v_j

        # ── 5. Aggregate: mean return per candidate action → softmax ──
        # cum_return_j: (B*M,) → (B, A, spa) → mean over spa → (B, A)
        returns_per_action = cum_return_j.view(B, A, spa).mean(dim=2)
        pi_mve[:, j] = F.softmax(returns_per_action / temperature, dim=-1)

        optimised.add(j)

    return pi_mve  # (B, N, A) soft probability distribution
