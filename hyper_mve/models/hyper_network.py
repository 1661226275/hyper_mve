"""
DualHyperNetwork for Hyper-MuZero.

Three-way hypernetwork that generates weights for:
    1. hyper_trans(rule_emb)           -> θ_state   (objective, Rule only)
    2. hyper_rew(rule_emb ⊕ id_emb)   -> θ_reward  (subjective, Rule + ID)
    3. hyper_pred(rule_emb ⊕ id_emb)  -> θ_pred    (subjective, Rule + ID)

Key engineering details:
    - Output layer uses small_init (std=0.01) to prevent initial weight explosion
    - LayerNorm between hidden layers for stability
    - Gradient clipping handled by the trainer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import orthogonal_init, small_init


class HyperNetMLP(nn.Module):
    """
    MLP that generates flat parameter vectors from context embeddings.

    Architecture: context -> FC1 -> LN -> ReLU -> FC2 -> LN -> ReLU -> FC_out -> flat_params

    Uses LayerNorm between layers and small initialization on the output layer
    to prevent the generated weights from being too large initially.
    """

    def __init__(self, input_dim, output_dim, hidden_dims=None, norm_output=True,
                 output_scale_init=0.01):
        """
        Args:
            input_dim:         dimension of input context
            output_dim:        total number of parameters to generate
            hidden_dims:       list of hidden layer dimensions (default: [256, 256])
            norm_output:       whether to apply L2 normalization to output (v4.0 stability trick)
            output_scale_init: [v4.7] initial value for learnable output_scale
                               (default 0.01; hyper_rew uses 0.1 to avoid initialization trap)
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        self.trunk = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)

        # Initialize trunk with orthogonal, output with small weights
        for module in self.trunk:
            if isinstance(module, nn.Linear):
                orthogonal_init(module)
        small_init(self.output_layer, std=0.01)

        # [v4.0] L2 Norm + Scale stability trick
        self.norm_output = norm_output
        self.output_scale = nn.Parameter(torch.tensor(float(output_scale_init)))

    def forward(self, context):
        """
        Args:
            context: (B, input_dim) — context embedding
        Returns:
            flat_params: (B, output_dim) — generated parameters
        """
        h = self.trunk(context)
        raw = self.output_layer(h)

        # [v4.0] L2 normalization: fix direction distribution, magnitude = 1.0
        if self.norm_output:
            norms = torch.linalg.norm(raw, dim=-1, keepdim=True)
            raw = raw / (norms + 1e-8)
        
        return raw * self.output_scale


class DualHyperNetwork(nn.Module):
    """
    Three-way hypernetwork for Hyper-MuZero.

    Generates weights for three functional networks:
        - θ_state  = hyper_trans(rule_emb)           — objective physics
        - θ_reward = hyper_rew(rule_emb ⊕ id_emb)   — subjective reward
        - θ_pred   = hyper_pred(rule_emb ⊕ id_emb)  — subjective policy+value

    Design principle: "Perspective as Context"
        - Rule determines objective physics → hyper_trans only sees rule_emb
        - Rule + Agent ID determines subjective evaluation → hyper_rew/pred see both
    """

    def __init__(self, rule_emb_dim, id_emb_dim, trans_param_count, rew_param_count,
                 pred_param_count, hidden_dims=None, rew_hidden_dims=None, norm_output=True,
                 rew_output_scale_init=0.01):
        """
        Args:
            rule_emb_dim:          dimension of rule embedding
            id_emb_dim:            dimension of agent id embedding
            trans_param_count:     total params for FunctionalStateTransNet
            rew_param_count:       total params for FunctionalRewardHead
            pred_param_count:      total params for FunctionalPredictionNet
            hidden_dims:           hidden layer dims for HyperNet MLPs (default for all branches)
            rew_hidden_dims:       [v4.7] optional separate hidden dims for hyper_rew (deeper for better differentiation)
            norm_output:           enable L2 norm + scale (default: True)
            rew_output_scale_init: [v4.7] initial output_scale for hyper_rew (default 0.01;
                                   0.1 recommended to avoid initialization trap where tiny
                                   AdaLN modulation prevents reward differentiation)
        """
        super().__init__()

        aug_dim = rule_emb_dim + id_emb_dim
        rew_hdims = rew_hidden_dims if rew_hidden_dims is not None else hidden_dims

        # Objective: Rule only
        self.hyper_trans = HyperNetMLP(rule_emb_dim, trans_param_count, hidden_dims, norm_output=norm_output)
        # Subjective: Rule + Agent ID
        self.hyper_rew = HyperNetMLP(aug_dim, rew_param_count, rew_hdims, norm_output=norm_output,
                                     output_scale_init=rew_output_scale_init)
        self.hyper_pred = HyperNetMLP(aug_dim, pred_param_count, hidden_dims, norm_output=norm_output)

        self.trans_param_count = trans_param_count
        self.rew_param_count = rew_param_count
        self.pred_param_count = pred_param_count

    def forward(self, rule_emb, id_emb):
        """
        Generate all three sets of parameters.

        Args:
            rule_emb: (B, rule_emb_dim)
            id_emb:   (B, id_emb_dim)
        Returns:
            θ_state:  (B, trans_param_count)
            θ_reward: (B, rew_param_count)
            θ_pred:   (B, pred_param_count)
        """
        aug_context = torch.cat([rule_emb, id_emb], dim=-1)

        theta_state = self.hyper_trans(rule_emb)
        theta_reward = self.hyper_rew(aug_context)
        theta_pred = self.hyper_pred(aug_context)

        return theta_state, theta_reward, theta_pred

    def generate_trans_params(self, rule_emb):
        """Generate state transition params only (objective, no agent_id)."""
        return self.hyper_trans(rule_emb)

    def generate_subjective_params(self, rule_emb, id_emb):
        """Generate reward + prediction params (subjective, needs agent_id)."""
        aug_context = torch.cat([rule_emb, id_emb], dim=-1)
        theta_reward = self.hyper_rew(aug_context)
        theta_pred = self.hyper_pred(aug_context)
        return theta_reward, theta_pred


def reward_diversity_loss(hyper_rew, rule_emb, id_embedding, num_agents,
                          target_cos=0.3, skip_pairs=None):
    """
    [v4.7] Diversity regularization for hyper_rew outputs.

    Penalizes high cosine similarity between θ_reward vectors generated
    for different agent identities under the same rule context. This prevents
    the hypernetwork from collapsing to a single "average" parameter set
    when reward MSE gradients are too weak to drive natural differentiation.

    Uses hinge loss: only penalizes when cos_sim > target_cos, so the loss
    naturally goes to zero once sufficient separation is achieved.

    Args:
        hyper_rew:    HyperNetMLP that generates θ_reward
        rule_emb:     (B, rule_emb_dim) — current batch's rule embeddings
        id_embedding: nn.Embedding for agent IDs
        num_agents:   int — total number of agents
        target_cos:   float — max allowed cosine similarity (hinge threshold)
        skip_pairs:   list of (i, j) tuples — agent pairs to skip (e.g., same-role agents)

    Returns:
        loss: scalar tensor — hinge diversity penalty (≥ 0)
    """
    B = rule_emb.shape[0]
    device = rule_emb.device

    # Generate θ_reward for each agent
    thetas = {}
    for aid in range(num_agents):
        id_t = torch.full((B,), aid, dtype=torch.long, device=device)
        id_emb = id_embedding(id_t)
        aug_ctx = torch.cat([rule_emb, id_emb], dim=-1)
        thetas[aid] = hyper_rew(aug_ctx)  # (B, rew_param_count)

    loss = torch.tensor(0.0, device=device)
    count = 0
    skip_set = set(skip_pairs) if skip_pairs else set()
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            if (i, j) in skip_set or (j, i) in skip_set:
                continue
            cos_sim = F.cosine_similarity(thetas[i], thetas[j], dim=-1)  # (B,)
            loss = loss + F.relu(cos_sim - target_cos).mean()
            count += 1

    return loss / max(count, 1)