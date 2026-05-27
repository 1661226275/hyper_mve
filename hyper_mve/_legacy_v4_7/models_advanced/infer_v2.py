"""
Infer-HyperMuZero v2 Model (Phase 5).

Same as InferHyperMuZeroModel but uses ChunkedDualHyperNetwork
(hypnettorch ChunkedHMLP) instead of simple MLP HyperNet.

Interface is identical — drop-in replacement for Exp3.
"""
import torch
import torch.nn as nn

from models.representation_net import RepresentationNet
from models.gru_context_encoder import InferContextEncoder
from models_advanced.chunked_hyper_network import ChunkedDualHyperNetwork
from models.functional_nets import (
    FunctionalStateTransNet,
    FunctionalRewardHead,
    FunctionalPredictionNet,
)
from utils.utils import actions_to_one_hot


class InferHyperMuZeroModelV2(nn.Module):
    """
    Infer-HyperMuZero v2 — ChunkedHMLP edition.

    Architecture identical to InferHyperMuZeroModel, but HyperNet is
    replaced by ChunkedDualHyperNetwork for parameter efficiency.

    Same interface: set_context_from_history / set_context / set_context_default.
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

        # ── ChunkedDualHyperNetwork (Phase 5 upgrade) ──────────
        self.hyper_net = ChunkedDualHyperNetwork(
            rule_emb_dim=cfg.rule_emb_dim,
            id_emb_dim=cfg.id_emb_dim,
            trans_shapes=self.func_state_trans.param_shapes,
            trans_param_count=self.func_state_trans.total_params,
            rew_shapes=self.func_reward_head.param_shapes,
            rew_param_count=self.func_reward_head.total_params,
            pred_shapes=self.func_pred_net.param_shapes,
            pred_param_count=self.func_pred_net.total_params,
            chunk_alpha=cfg.chunk_alpha,
            chunk_emb_size=cfg.chunk_emb_size,
            budget_factor=cfg.chunk_budget_factor,
            do_hyperfan_init=cfg.chunk_hyperfan_init,
            verbose=True,
        )

        # ── Internal context cache ──────────────────────────────
        self._theta_state = None
        self._theta_reward = None
        self._theta_pred = None
        self._current_rule_emb = None

    def set_context(self, rule_emb_or_rule, agent_ids):
        """
        Set context from pre-computed rule_emb and agent_ids.

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