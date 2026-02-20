"""
Oracle-HyperMuZero Model (Exp2).

Uses DualHyperNetwork to dynamically generate weights based on
explicitly provided Rule + Agent ID.

Exposes the SAME interface as BaselineModel so that MuZeroTrainer,
Worker, and MVE Planner can be reused without modification.

Key difference from Baseline:
    - Baseline: fixed weights, agent_id concatenated as input feature
    - Oracle:   HyperNet-generated weights, agent_id encoded in weights themselves

The model maintains an internal "context state" that must be set before
calling transition/predict_reward/predict. This is handled automatically
by set_context() or by the higher-level methods.
"""
import torch
import torch.nn as nn

from models.representation_net import RepresentationNet
from models.context_encoder import AugmentedContextEncoder
from models.hyper_network import DualHyperNetwork
from models.functional_nets import (
    FunctionalStateTransNet,
    FunctionalRewardHead,
    FunctionalPredictionNet,
)


class OracleHyperMuZeroModel(nn.Module):
    """
    Oracle-HyperMuZero model (Exp2).

    Architecture:
        RepresentationNet (fixed params)  -> s
        ContextEncoder (Rule + ID)        -> rule_emb, id_emb
        DualHyperNetwork                  -> θ_state, θ_reward, θ_pred
        FunctionalStateTransNet(θ_state)  -> s'
        FunctionalRewardHead(θ_reward)    -> r_i
        FunctionalPredictionNet(θ_pred)   -> p_i, v_i

    Interface (same as BaselineModel):
        encode(joint_obs) -> s
        get_id_emb(agent_ids) -> id_emb
        transition(state, action_onehot) -> next_state
        predict_reward(state, action_onehot, id_emb) -> reward
        predict(state, id_emb) -> policy_logits, value
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
        # These are set by set_context() and used by transition/predict_reward/predict
        self._theta_state = None
        self._theta_reward = None
        self._theta_pred = None
        self._current_rule = None
        self._current_rule_emb = None

    def set_context(self, rule, agent_ids):
        """
        Set the current context (Rule + Agent IDs) and generate all params.

        Must be called before transition/predict_reward/predict
        when using those methods directly (e.g., from MuZeroTrainer).

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

        Compatible interface with BaselineModel.
        For HyperMuZero, this is used by the MVE planner and worker
        which call predict() with different agent_ids per agent.

        Args:
            agent_ids: (B,) int tensor
        Returns:
            id_emb: (B, id_emb_dim)
        """
        return self.context_encoder.encode_id(agent_ids)

    def transition(self, state, action_onehot):
        """
        Predict next state using HyperNet-generated weights (objective).

        Requires set_context() to have been called first.

        Args:
            state:         (B, latent_dim)
            action_onehot: (B, joint_action_dim)
        Returns:
            next_state: (B, latent_dim)
        """
        assert self._theta_state is not None, "Call set_context() before transition()"
        return self.func_state_trans(state, action_onehot, self._theta_state)

    def predict_reward(self, state, action_onehot, id_emb=None):
        """
        Predict per-agent reward using HyperNet-generated weights (subjective).

        Requires set_context() to have been called first.
        id_emb argument is accepted for interface compatibility but ignored
        (identity is already encoded in θ_reward via HyperNet).

        Args:
            state:         (B, latent_dim)
            action_onehot: (B, joint_action_dim)
            id_emb:        ignored (kept for interface compatibility)
        Returns:
            reward: (B, 1) in scaled space
        """
        assert self._theta_reward is not None, "Call set_context() before predict_reward()"
        return self.func_reward_head(state, action_onehot, self._theta_reward)

    def predict(self, state, id_emb=None):
        """
        Predict policy and value using HyperNet-generated weights (subjective).

        Requires set_context() to have been called first.
        id_emb argument is accepted for interface compatibility but ignored.

        Args:
            state:  (B, latent_dim)
            id_emb: ignored (kept for interface compatibility)
        Returns:
            policy_logits: (B, num_actions)
            value:         (B, 1) in scaled space
        """
        assert self._theta_pred is not None, "Call set_context() before predict()"
        return self.func_pred_net(state, self._theta_pred)

    # ── Convenience methods for MVE Planner and Worker ──────────

    def set_context_for_agent(self, rule, agent_id_int, batch_size):
        """
        Set context for a single agent across a batch.
        Useful in MVE planner which loops over agents.

        Args:
            rule:          (B,) float or scalar
            agent_id_int:  int — single agent id
            batch_size:    int
        """
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
