"""
Oracle-HyperMuZero v2 Model (Phase 5).

Same as OracleHyperMuZeroModel but uses ChunkedDualHyperNetwork
(hypnettorch ChunkedHMLP) instead of simple MLP HyperNet.

Interface is identical — drop-in replacement for Exp2.
"""
import torch
import torch.nn as nn

from models.representation_net import RepresentationNet
from models.context_encoder import AugmentedContextEncoder
from models_advanced.chunked_hyper_network import ChunkedDualHyperNetwork
from models.functional_nets import (
    FunctionalStateTransNet,
    FunctionalRewardHead,
    FunctionalPredictionNet,
)


class OracleHyperMuZeroModelV2(nn.Module):
    """
    Oracle-HyperMuZero v2 — ChunkedHMLP edition.

    Architecture identical to OracleHyperMuZeroModel, but HyperNet is
    replaced by ChunkedDualHyperNetwork for parameter efficiency.

    Same interface: set_context(rule, agent_ids) -> transition/predict_reward/predict.
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

        # ── Context Encoder ─────────────────────────────────────
        self.context_encoder = AugmentedContextEncoder(
            num_agents=cfg.num_agents,
            rule_emb_dim=cfg.rule_emb_dim,
            id_emb_dim=cfg.id_emb_dim,
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
        self._current_rule = None
        self._current_rule_emb = None

    def set_context(self, rule, agent_ids):
        """
        Set context (Rule + Agent IDs) and generate all params.

        Args:
            rule:      (B,) float
            agent_ids: (B,) int
        """
        rule_emb, id_emb, aug_context = self.context_encoder(rule, agent_ids)
        theta_state, theta_reward, theta_pred = self.hyper_net(rule_emb, id_emb)
        self._theta_state = theta_state
        self._theta_reward = theta_reward
        self._theta_pred = theta_pred
        self._current_rule = rule
        self._current_rule_emb = rule_emb

    def encode(self, joint_obs):
        """Encode joint observation into latent state. (B, joint_obs_dim) -> (B, latent_dim)"""
        return self.repr_net(joint_obs)

    def get_id_emb(self, agent_ids):
        """Get agent identity embeddings. (B,) -> (B, id_emb_dim)"""
        return self.context_encoder.encode_id(agent_ids)

    def transition(self, state, action_onehot):
        """Predict next state (objective). Requires set_context()."""
        assert self._theta_state is not None, "Call set_context() before transition()"
        return self.func_state_trans(state, action_onehot, self._theta_state)

    def predict_reward(self, state, action_onehot, id_emb=None):
        """Predict per-agent reward (subjective). id_emb ignored."""
        assert self._theta_reward is not None, "Call set_context() before predict_reward()"
        return self.func_reward_head(state, action_onehot, self._theta_reward)

    def predict(self, state, id_emb=None):
        """Predict policy and value (subjective). id_emb ignored."""
        assert self._theta_pred is not None, "Call set_context() before predict()"
        return self.func_pred_net(state, self._theta_pred)

    def set_context_for_agent(self, rule, agent_id_int, batch_size):
        """Set context for a single agent across a batch."""
        device = next(self.parameters()).device
        if isinstance(rule, (int, float)):
            rule = torch.full((batch_size,), rule, dtype=torch.float32, device=device)
        agent_ids = torch.full((batch_size,), agent_id_int, dtype=torch.long, device=device)
        self.set_context(rule, agent_ids)

    def clear_context(self):
        """Clear cached context parameters."""
        self._theta_state = None
        self._theta_reward = None
        self._theta_pred = None
        self._current_rule = None
        self._current_rule_emb = None