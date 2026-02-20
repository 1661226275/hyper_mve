"""
Infer-HyperMuZero Model (Exp3) — Core Contribution.

Uses GRU to infer rule_emb from historical trajectory,
then feeds it to the same DualHyperNetwork as Exp2.

Key differences from Exp2 (Oracle):
    - Rule is NEVER provided (not even during training)
    - GRU infers rule_emb from sliding window of (obs, action, reward)
    - Additional context regularization loss
    - At episode start, uses a learned default embedding

Same interface as BaselineModel / OracleHyperMuZeroModel.
"""
import torch
import torch.nn as nn

from models.representation_net import RepresentationNet
from models.gru_context_encoder import InferContextEncoder
from models.hyper_network import DualHyperNetwork
from models.functional_nets import (
    FunctionalStateTransNet,
    FunctionalRewardHead,
    FunctionalPredictionNet,
)
from utils.utils import actions_to_one_hot


class InferHyperMuZeroModel(nn.Module):
    """
    Infer-HyperMuZero model (Exp3).

    Architecture:
        RepresentationNet (fixed params)  -> s
        GRU(history) -> rule_emb          (inferred, no explicit Rule)
        ID Embedding -> id_emb
        DualHyperNetwork                  -> θ_state, θ_reward, θ_pred
        FunctionalStateTransNet(θ_state)  -> s'
        FunctionalRewardHead(θ_reward)    -> r_i
        FunctionalPredictionNet(θ_pred)   -> p_i, v_i

    set_context_from_history() replaces set_context(rule, agent_ids)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # ── Representation Network (fixed params, objective) ────
        self.repr_net = RepresentationNet(
            joint_obs_dim=cfg.joint_obs_dim,
            hidden_dim=256,
            latent_dim=cfg.latent_dim,
        )

        # ── Infer Context Encoder (GRU + ID Embedding) ─────────
        self.context_encoder = InferContextEncoder(
            num_agents=cfg.num_agents,
            joint_obs_dim=cfg.joint_obs_dim,
            joint_action_dim=cfg.joint_action_dim,
            rule_emb_dim=cfg.rule_emb_dim,
            id_emb_dim=cfg.id_emb_dim,
            gru_hidden_size=cfg.gru_hidden_size,
        )

        # ── Functional Networks (weight-free skeletons) ─────────
        self.func_state_trans = FunctionalStateTransNet(
            latent_dim=cfg.latent_dim,
            joint_action_dim=cfg.joint_action_dim,
            hidden_dim=cfg.hidden_dim,
        )
        self.func_reward_head = FunctionalRewardHead(
            latent_dim=cfg.latent_dim,
            joint_action_dim=cfg.joint_action_dim,
            hidden_dim=cfg.hidden_dim,
        )
        self.func_pred_net = FunctionalPredictionNet(
            latent_dim=cfg.latent_dim,
            num_actions=cfg.num_actions,
            hidden_dim=cfg.hidden_dim,
        )

        # ── DualHyperNetwork ───────────────────────────────────
        self.hyper_net = DualHyperNetwork(
            rule_emb_dim=cfg.rule_emb_dim,
            id_emb_dim=cfg.id_emb_dim,
            trans_param_count=self.func_state_trans.total_params,
            rew_param_count=self.func_reward_head.total_params,
            pred_param_count=self.func_pred_net.total_params,
            hidden_dims=cfg.hyper_hidden_dims,
        )

        # ── Internal context cache ──────────────────────────────
        self._theta_state = None
        self._theta_reward = None
        self._theta_pred = None
        self._current_rule_emb = None  # For regularization loss

    def set_context(self, rule_emb_or_rule, agent_ids):
        """
        Set context from pre-computed rule_emb and agent_ids.

        This is called by the trainer after GRU inference.
        Also called by MVE planner (which detects is_hyper via hasattr).

        Args:
            rule_emb_or_rule: (B, rule_emb_dim) tensor — pre-inferred rule embedding
            agent_ids:        (B,) int tensor
        """
        rule_emb = rule_emb_or_rule
        id_emb = self.context_encoder.encode_id(agent_ids)
        theta_state, theta_reward, theta_pred = self.hyper_net(rule_emb, id_emb)
        self._theta_state = theta_state
        self._theta_reward = theta_reward
        self._theta_pred = theta_pred
        self._current_rule_emb = rule_emb

    def set_context_from_history(self, history_obs, history_actions_onehot,
                                  history_rewards, agent_ids, mask=None):
        """
        Infer rule_emb from history, then set context.

        Args:
            history_obs:            (B, W, joint_obs_dim)
            history_actions_onehot: (B, W, joint_action_dim)
            history_rewards:        (B, W, 1)
            agent_ids:              (B,) int
            mask:                   (B, W) bool, optional
        """
        rule_emb, id_emb, aug_context = self.context_encoder(
            history_obs, history_actions_onehot, history_rewards, agent_ids, mask
        )
        theta_state, theta_reward, theta_pred = self.hyper_net(rule_emb, id_emb)
        self._theta_state = theta_state
        self._theta_reward = theta_reward
        self._theta_pred = theta_pred
        self._current_rule_emb = rule_emb

    def set_context_default(self, agent_ids, batch_size):
        """
        Set context with default rule embedding (episode start, no history).

        Args:
            agent_ids:  (B,) int tensor
            batch_size: int
        """
        device = next(self.parameters()).device
        rule_emb = self.context_encoder.get_default_rule_emb(batch_size, device)
        self.set_context(rule_emb, agent_ids)

    def encode(self, joint_obs):
        """Encode joint observation into latent state."""
        return self.repr_net(joint_obs)

    def get_id_emb(self, agent_ids):
        """Get agent identity embeddings."""
        return self.context_encoder.encode_id(agent_ids)

    def transition(self, state, action_onehot):
        """Predict next state using HyperNet-generated weights (objective)."""
        assert self._theta_state is not None, "Call set_context first"
        return self.func_state_trans(state, action_onehot, self._theta_state)

    def predict_reward(self, state, action_onehot, id_emb=None):
        """Predict per-agent reward (subjective). id_emb ignored."""
        assert self._theta_reward is not None, "Call set_context first"
        return self.func_reward_head(state, action_onehot, self._theta_reward)

    def predict(self, state, id_emb=None):
        """Predict policy and value (subjective). id_emb ignored."""
        assert self._theta_pred is not None, "Call set_context first"
        return self.func_pred_net(state, self._theta_pred)

    def get_current_rule_emb(self):
        """Get the most recently inferred rule_emb (for regularization loss)."""
        return self._current_rule_emb

    def clear_context(self):
        """Clear cached context parameters."""
        self._theta_state = None
        self._theta_reward = None
        self._theta_pred = None
        self._current_rule_emb = None
