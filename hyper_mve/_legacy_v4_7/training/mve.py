"""
Model Value Expansion (MVE) target computation.

Computes k-step target: y_t = sum_{i=0}^{k-1} gamma^i * r_i + gamma^k * V(s_k)
Uses the DynamicNet to roll out k steps in latent space.
"""
import torch


@torch.no_grad()
def compute_mve_target(model, state, action, gamma, k):
    """
    Compute MVE k-step target using model rollout in latent space.

    Args:
        model:  BaselineModel (or any model with predict_next, predict_value, select_action)
        state:  (batch_size, latent_dim) - current latent state
        action: (batch_size, joint_action_dim) - action taken at current step
        gamma:  float - discount factor
        k:      int - number of rollout steps

    Returns:
        target: (batch_size, 1) - MVE target value
    """
    cumulative_reward = torch.zeros(state.shape[0], 1, device=state.device)
    discount = 1.0
    s = state

    for i in range(k):
        if i == 0:
            a = action
        else:
            a = model.select_action(s)

        s_next, r = model.predict_next(s, a)
        cumulative_reward += discount * r
        discount *= gamma
        s = s_next

    # Terminal value
    v_k = model.predict_value(s)
    target = cumulative_reward + discount * v_k

    return target
