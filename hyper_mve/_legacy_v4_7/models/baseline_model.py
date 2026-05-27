"""
Baseline Model (Exp1) — v3.1

Standard Model-Based RL without HyperNetworks.
Fixed-weight networks with agent_id embedding concatenated as input.

Architecture (matches Exp2/3 structure, minus HyperNet):
    - RepresentationNet:  o_joint -> s                  (shared, objective)
    - StateTransNet:      (s, A_joint_onehot) -> s'     (fixed weights, objective, NO agent_id)
    - RewardHead:         (s, A_joint_onehot, id_emb) -> r_i   (fixed weights, subjective, r=R(s,a))
    - PredictionNet:      (s, id_emb) -> p_i[5], v_i[1]        (fixed weights, subjective)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.representation_net import RepresentationNet
from utils.utils import orthogonal_init, actions_to_one_hot, scalar_transform


class StateTransNet(nn.Module):
    """
    Objective state transition: (s, A_joint_onehot) -> s'

    No agent_id input — physics is objective and agent-independent.

    Input:  s (B, latent_dim), A_joint_onehot (B, num_agents * num_actions)
    Output: s' (B, latent_dim)
    """

    def __init__(self, latent_dim, joint_action_dim, hidden_dim=128):
        super().__init__()
        input_dim = latent_dim + joint_action_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)
        self.ln = nn.LayerNorm(latent_dim)

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)

    def forward(self, state, action_onehot):
        """
        Args:
            state:         (batch_size, latent_dim)
            action_onehot: (batch_size, joint_action_dim)  # already one-hot encoded
        Returns:
            next_state: (batch_size, latent_dim)
        """
        x = torch.cat([state, action_onehot], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.ln(x)
        return x


class RewardHead(nn.Module):
    """
    Subjective reward prediction: (s, A_joint_onehot, id_emb) -> r_i

    Uses current state s (not s'), following MDP causality: r = R(s, a).
    Includes agent_id embedding — reward is perspective-dependent.

    Input:  s (B, latent_dim), A_joint_onehot (B, joint_action_dim), id_emb (B, id_emb_dim)
    Output: r_i (B, 1)
    """

    def __init__(self, latent_dim, joint_action_dim, id_emb_dim, hidden_dim=128):
        super().__init__()
        input_dim = latent_dim + joint_action_dim + id_emb_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)

    def forward(self, state, action_onehot, id_emb):
        """
        Args:
            state:         (batch_size, latent_dim)
            action_onehot: (batch_size, joint_action_dim)
            id_emb:        (batch_size, id_emb_dim)
        Returns:
            reward: (batch_size, 1)  # in scaled space
        """
        x = torch.cat([state, action_onehot, id_emb], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PredictionNet(nn.Module):
    """
    Subjective policy + value prediction: (s, id_emb) -> p_i[5], v_i[1]

    Includes agent_id embedding — policy and value are perspective-dependent.

    Input:  s (B, latent_dim), id_emb (B, id_emb_dim)
    Output: policy_logits (B, num_actions), value (B, 1)
    """

    def __init__(self, latent_dim, id_emb_dim, num_actions=5, hidden_dim=128):
        super().__init__()
        input_dim = latent_dim + id_emb_dim

        # Shared trunk
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Policy head
        self.policy_head = nn.Linear(hidden_dim, num_actions)

        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.policy_head)
        orthogonal_init(self.value_head)

    def forward(self, state, id_emb):
        """
        Args:
            state:  (batch_size, latent_dim)
            id_emb: (batch_size, id_emb_dim)
        Returns:
            policy_logits: (batch_size, num_actions) — raw logits, NOT softmax
            value:         (batch_size, 1)           — in scaled space
        """
        x = torch.cat([state, id_emb], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value


class BaselineModel(nn.Module):
    """
    Complete Baseline model (Exp1) for Hyper-MuZero.

    No HyperNetwork. Uses fixed weights with agent_id_embedding as input feature.
    Matches the structure of Exp2/Exp3 for fair comparison.

    Components:
        - repr_net:       RepresentationNet (objective, shared)
        - state_trans:    StateTransNet (objective, no agent_id)
        - reward_head:    RewardHead (subjective, with agent_id)
        - pred_net:       PredictionNet (subjective, with agent_id)
        - id_embedding:   nn.Embedding for agent identity
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Agent ID embedding
        self.id_embedding = nn.Embedding(cfg.num_agents, cfg.id_emb_dim)

        # Representation Network (objective)
        self.repr_net = RepresentationNet(
            joint_obs_dim=cfg.joint_obs_dim,
            hidden_dim=256,
            latent_dim=cfg.latent_dim,
        )

        # State Transition Network (objective, no agent_id)
        self.state_trans = StateTransNet(
            latent_dim=cfg.latent_dim,
            joint_action_dim=cfg.joint_action_dim,  # num_agents * num_actions = 20
            hidden_dim=cfg.hidden_dim,
        )

        # Reward Head (subjective, with agent_id)
        self.reward_head = RewardHead(
            latent_dim=cfg.latent_dim,
            joint_action_dim=cfg.joint_action_dim,
            id_emb_dim=cfg.id_emb_dim,
            hidden_dim=cfg.hidden_dim,
        )

        # Prediction Network (subjective, with agent_id)
        self.pred_net = PredictionNet(
            latent_dim=cfg.latent_dim,
            id_emb_dim=cfg.id_emb_dim,
            num_actions=cfg.num_actions,
            hidden_dim=cfg.hidden_dim,
        )

    def encode(self, joint_obs):
        """
        Encode joint observation into latent state.

        Args:
            joint_obs: (B, joint_obs_dim)
        Returns:
            s: (B, latent_dim)
        """
        return self.repr_net(joint_obs)

    def get_id_emb(self, agent_ids):
        """
        Get agent identity embeddings.

        Args:
            agent_ids: (B,) int tensor, values in [0, num_agents)
        Returns:
            id_emb: (B, id_emb_dim)
        """
        return self.id_embedding(agent_ids)

    def transition(self, state, action_onehot):
        """
        Predict next state (objective).

        Args:
            state:         (B, latent_dim)
            action_onehot: (B, joint_action_dim)
        Returns:
            next_state: (B, latent_dim)
        """
        return self.state_trans(state, action_onehot)

    def predict_reward(self, state, action_onehot, id_emb):
        """
        Predict per-agent reward (subjective).

        Args:
            state:         (B, latent_dim)
            action_onehot: (B, joint_action_dim)
            id_emb:        (B, id_emb_dim)
        Returns:
            reward: (B, 1) in scaled space
        """
        return self.reward_head(state, action_onehot, id_emb)

    def predict(self, state, id_emb):
        """
        Predict policy and value (subjective).

        Args:
            state:  (B, latent_dim)
            id_emb: (B, id_emb_dim)
        Returns:
            policy_logits: (B, num_actions)
            value:         (B, 1) in scaled space
        """
        return self.pred_net(state, id_emb)

    def initial_inference(self, joint_obs, agent_ids):
        """
        Full initial inference: encode obs, then predict policy+value.

        Args:
            joint_obs: (B, joint_obs_dim)
            agent_ids: (B,) int tensor
        Returns:
            s:             (B, latent_dim)
            policy_logits: (B, num_actions)
            value:         (B, 1)
        """
        s = self.encode(joint_obs)
        id_emb = self.get_id_emb(agent_ids)
        policy_logits, value = self.predict(s, id_emb)
        return s, policy_logits, value

    def recurrent_inference(self, state, action_onehot, agent_ids):
        """
        One step of recurrent inference: transition + predict.

        Args:
            state:         (B, latent_dim)
            action_onehot: (B, joint_action_dim)
            agent_ids:     (B,) int tensor
        Returns:
            next_state:    (B, latent_dim)
            reward:        (B, 1)
            policy_logits: (B, num_actions)
            value:         (B, 1)
        """
        id_emb = self.get_id_emb(agent_ids)
        next_state = self.transition(state, action_onehot)
        reward = self.predict_reward(state, action_onehot, id_emb)  # r=R(s,a)
        policy_logits, value = self.predict(next_state, id_emb)
        return next_state, reward, policy_logits, value
