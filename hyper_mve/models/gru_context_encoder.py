"""
GRU-based Context Encoder for Infer-HyperMuZero (Exp3).

Replaces the explicit RuleEncoder with a GRU that infers rule_emb
from a sliding window of historical (obs, action, reward) tuples.

Key design:
    - GRU processes a sequence of (joint_obs, joint_action_onehot, reward_scalar)
    - Final hidden state is projected to rule_emb_dim
    - At episode start (no history), outputs a learned default embedding
    - Context regularization: L2 norm + variance penalty
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import orthogonal_init


class GRUContextInferrer(nn.Module):
    """
    Infer rule_emb from historical trajectory using GRU.

    Input:  history sequence of (obs, action_onehot, reward) tuples
    Output: rule_emb (B, rule_emb_dim)

    The GRU input at each timestep is: [joint_obs, joint_action_onehot, reward_scalar]
    where reward_scalar is the mean reward across all agents (to avoid perspective bias).
    """

    def __init__(self, joint_obs_dim, joint_action_dim, rule_emb_dim=16,
                 gru_hidden_size=128):
        super().__init__()
        # GRU input: obs + action_onehot + 1 (mean reward)
        self.input_dim = joint_obs_dim + joint_action_dim + 1
        self.gru_hidden_size = gru_hidden_size
        self.rule_emb_dim = rule_emb_dim

        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # Project GRU hidden state to rule_emb
        self.proj = nn.Linear(gru_hidden_size, rule_emb_dim)
        orthogonal_init(self.proj)

        # Learned default embedding for episode start (no history)
        self.default_emb = nn.Parameter(torch.zeros(1, rule_emb_dim))
        nn.init.normal_(self.default_emb, mean=0.0, std=0.01)

    def forward(self, history_obs, history_actions_onehot, history_rewards, mask=None):
        """
        Infer rule embedding from history.

        Args:
            history_obs:            (B, W, joint_obs_dim)  float
            history_actions_onehot: (B, W, joint_action_dim) float
            history_rewards:        (B, W, 1) float — mean reward across agents
            mask:                   (B, W) bool — True for valid steps, False for padding

        Returns:
            rule_emb: (B, rule_emb_dim)
        """
        B, W, _ = history_obs.shape

        # Concatenate inputs: (B, W, input_dim)
        gru_input = torch.cat([history_obs, history_actions_onehot, history_rewards], dim=-1)

        if mask is not None and not mask.all():
            # Pack padded sequences for efficient GRU processing
            lengths = mask.sum(dim=1).long().clamp(min=1)  # (B,)
            packed = nn.utils.rnn.pack_padded_sequence(
                gru_input, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, h_n = self.gru(packed)  # h_n: (1, B, hidden)
        else:
            _, h_n = self.gru(gru_input)  # h_n: (1, B, hidden)

        # h_n: (1, B, hidden) -> (B, hidden)
        h = h_n.squeeze(0)

        # Project to rule_emb
        rule_emb = self.proj(h)  # (B, rule_emb_dim)

        return rule_emb

    def get_default_emb(self, batch_size, device):
        """
        Get default rule embedding for episode start (no history).

        Args:
            batch_size: int
            device: torch device
        Returns:
            rule_emb: (B, rule_emb_dim)
        """
        return self.default_emb.expand(batch_size, -1).to(device)


class InferContextEncoder(nn.Module):
    """
    Context encoder for Infer-HyperMuZero (Exp3).

    Replaces AugmentedContextEncoder's RuleEncoder with GRUContextInferrer.
    ID Embedding remains the same.

    Produces:
        - rule_emb:    GRU-inferred (B, rule_emb_dim) — for hyper_trans
        - aug_context: [rule_emb, id_emb] — for hyper_rew/hyper_pred
    """

    def __init__(self, num_agents, joint_obs_dim, joint_action_dim,
                 rule_emb_dim=16, id_emb_dim=16, gru_hidden_size=128):
        super().__init__()
        self.gru_inferrer = GRUContextInferrer(
            joint_obs_dim=joint_obs_dim,
            joint_action_dim=joint_action_dim,
            rule_emb_dim=rule_emb_dim,
            gru_hidden_size=gru_hidden_size,
        )
        self.id_embedding = nn.Embedding(num_agents, id_emb_dim)
        self.rule_emb_dim = rule_emb_dim
        self.id_emb_dim = id_emb_dim
        self.aug_dim = rule_emb_dim + id_emb_dim

    def forward(self, history_obs, history_actions_onehot, history_rewards,
                agent_ids, mask=None):
        """
        Args:
            history_obs:            (B, W, joint_obs_dim)
            history_actions_onehot: (B, W, joint_action_dim)
            history_rewards:        (B, W, 1) — mean reward
            agent_ids:              (B,) int
            mask:                   (B, W) bool, optional
        Returns:
            rule_emb:    (B, rule_emb_dim)
            id_emb:      (B, id_emb_dim)
            aug_context: (B, rule_emb_dim + id_emb_dim)
        """
        rule_emb = self.gru_inferrer(
            history_obs, history_actions_onehot, history_rewards, mask
        )
        id_emb = self.id_embedding(agent_ids)
        aug_context = torch.cat([rule_emb, id_emb], dim=-1)
        return rule_emb, id_emb, aug_context

    def infer_rule(self, history_obs, history_actions_onehot, history_rewards, mask=None):
        """Infer rule embedding only."""
        return self.gru_inferrer(
            history_obs, history_actions_onehot, history_rewards, mask
        )

    def encode_id(self, agent_ids):
        """Encode agent_id only."""
        return self.id_embedding(agent_ids)

    def get_default_rule_emb(self, batch_size, device):
        """Get default rule embedding for episode start."""
        return self.gru_inferrer.get_default_emb(batch_size, device)


def context_regularization_loss(rule_emb, beta=0.01, gamma_reg=0.01):
    """
    [DEPRECATED] Legacy context regularization with 1/var penalty.
    Kept for backward compatibility. Use context_variance_loss() instead.
    """
    l2_loss = beta * (rule_emb ** 2).mean()
    var_per_dim = rule_emb.var(dim=0)
    var_penalty = gamma_reg * (1.0 / (var_per_dim.mean() + 1e-6))
    return l2_loss + var_penalty


def context_variance_loss(rule_emb, target_std=0.1):
    """
    Hinge Variance context regularization loss for Exp3 (v4.0).

    Encourages batch-wise standard deviation of rule_emb to stay above
    target_std, preventing embedding collapse. Replaces the unstable
    1/var penalty from v3.2.

    Properties:
        - No division by variance -> no explosion risk when var -> 0
        - Linear penalty (ReLU) -> smooth gradient
        - Zero loss when std >= target_std -> no interference when healthy

    Args:
        rule_emb:   (B, rule_emb_dim) inferred rule embeddings
        target_std: float, minimum desired per-dim standard deviation
    Returns:
        loss: scalar
    """
    # Per-dimension std across batch, then mean over dims
    batch_std = torch.std(rule_emb, dim=0).mean()  # scalar
    return torch.relu(target_std - batch_std)
