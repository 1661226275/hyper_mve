"""
Context Encoder for Hyper-MuZero.

Encodes the augmented context C_aug = [rule_emb, id_emb]:
    - RuleEncoder: scalar Rule -> rule_emb (MLP)
    - ID Embedding: int agent_id -> id_emb (nn.Embedding)

Exp2 (Oracle): Rule is provided explicitly.
Exp3 (Infer):  Rule embedding is inferred by GRU (replaces RuleEncoder).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import orthogonal_init


class RuleEncoder(nn.Module):
    """
    Encode scalar Rule value into a dense embedding.

    Input:  rule (B,) or (B, 1) float in [0, 1]
    Output: rule_emb (B, rule_emb_dim)
    """

    def __init__(self, rule_emb_dim=16, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, rule_emb_dim)

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)

    def forward(self, rule):
        """
        Args:
            rule: (B,) or (B, 1) float
        Returns:
            rule_emb: (B, rule_emb_dim)
        """
        if rule.dim() == 1:
            rule = rule.unsqueeze(-1)  # (B,) -> (B, 1)
        x = F.relu(self.fc1(rule))
        return self.fc2(x)


class AugmentedContextEncoder(nn.Module):
    """
    Full augmented context encoder for Oracle-HyperMuZero (Exp2).

    Produces:
        - rule_emb:    for hyper_trans (objective)
        - aug_context: [rule_emb, id_emb] for hyper_rew/hyper_pred (subjective)

    Components:
        - RuleEncoder: rule (scalar) -> rule_emb (rule_emb_dim)
        - ID Embedding: agent_id (int) -> id_emb (id_emb_dim)
    """

    def __init__(self, num_agents, rule_emb_dim=16, id_emb_dim=16, hidden_dim=64):
        super().__init__()
        self.rule_encoder = RuleEncoder(rule_emb_dim, hidden_dim)
        self.id_embedding = nn.Embedding(num_agents, id_emb_dim)
        self.rule_emb_dim = rule_emb_dim
        self.id_emb_dim = id_emb_dim
        self.aug_dim = rule_emb_dim + id_emb_dim

    def forward(self, rule, agent_ids):
        """
        Args:
            rule:      (B,) float in [0, 1]
            agent_ids: (B,) int in [0, num_agents)
        Returns:
            rule_emb:    (B, rule_emb_dim)
            id_emb:      (B, id_emb_dim)
            aug_context: (B, rule_emb_dim + id_emb_dim)
        """
        rule_emb = self.rule_encoder(rule)      # (B, rule_emb_dim)
        id_emb = self.id_embedding(agent_ids)   # (B, id_emb_dim)
        aug_context = torch.cat([rule_emb, id_emb], dim=-1)  # (B, aug_dim)
        return rule_emb, id_emb, aug_context

    def encode_rule(self, rule):
        """Encode rule only (for hyper_trans which doesn't need agent_id)."""
        return self.rule_encoder(rule)

    def encode_id(self, agent_ids):
        """Encode agent_id only."""
        return self.id_embedding(agent_ids)
