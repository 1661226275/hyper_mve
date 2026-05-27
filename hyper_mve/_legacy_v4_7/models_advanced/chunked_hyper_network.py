"""
ChunkedDualHyperNetwork for Hyper-MuZero (Phase 5).

Replaces the simple HyperNetMLP with hypnettorch's ChunkedHMLP for
parameter-efficient weight generation.

Key advantages over simple MLP HyperNet:
    - Parameter efficiency: HyperNet params ≈ target net params (instead of >>)
    - Principled initialization: hyperfan_init designed specifically for hypernetworks
    - Chunk-based generation: weight vector is split into chunks, each generated
      by a shared small network + per-chunk embedding

Architecture is the same DualHyperNetwork pattern:
    - hyper_trans(rule_emb)           -> θ_state   (objective, Rule only)
    - hyper_rew(rule_emb ⊕ id_emb)   -> θ_reward  (subjective, Rule + ID)
    - hyper_pred(rule_emb ⊕ id_emb)  -> θ_pred    (subjective, Rule + ID)

Requires: pip install hypnettorch==0.0.4
"""
import numpy as np
import torch
import torch.nn as nn

from hypnettorch.hnets.chunked_mlp_hnet import ChunkedHMLP


def _get_hidden_size(num_weights_mnet, alpha, ctx_size, inp_embeddings,
                     return_size=False):
    """
    Calculate the optimal hidden size for ChunkedHMLP to match parameter budget.

    Finds hidden layer sizes such that the hypernetwork has approximately
    the same number of parameters as the target network.

    Reference: context-conditioned-world-model-and-mcts/dynamics_hypernetwork.py

    Args:
        num_weights_mnet: Target number of weights (multiplied by budget factor)
        alpha: Chunk alpha parameter (chunk_size = alpha * hidden_size)
        ctx_size: Context input size (cond_in_size)
        inp_embeddings: Chunk embedding size (chunk_emb_size)
        return_size: Whether to return the total parameter count

    Returns:
        Tuple of (chunk_size, hidden_layers, [total_params])
    """
    def getabc(good_h):
        num_chunks = np.ceil(num_weights_mnet / alpha / good_h)
        a = 1 + alpha
        b = (ctx_size + 1) + 1 + alpha + inp_embeddings
        c = -num_weights_mnet + inp_embeddings * num_chunks
        return a, b, c

    def _count_params(good_h):
        a, b, c = getabc(good_h)
        return a * good_h ** 2 + b * good_h + (c + num_weights_mnet)

    good_h = 1
    while True:
        a, b, c = getabc(good_h)

        def roots():
            discriminant = b ** 2 - 4 * a * c
            if discriminant < 0:
                return (0, 0)
            f1 = np.sqrt(discriminant)
            return int((-b - f1) / (2 * a)), int((-b + f1) / (2 * a))

        rs = roots()
        new_good_h = rs[-1]
        if new_good_h <= good_h or _count_params(new_good_h) > num_weights_mnet:
            break
        good_h = new_good_h

    if return_size:
        return alpha * good_h, [good_h, good_h], _count_params(good_h)
    return alpha * good_h, [good_h, good_h]


class ChunkedHyperNetWrapper(nn.Module):
    """
    Wrapper around ChunkedHMLP that outputs flattened parameter vectors.

    Provides the same interface as HyperNetMLP:
        forward(context) -> flat_params (B, output_dim)

    Internally uses ChunkedHMLP with ret_format="flattened" for batched output.

    Args:
        input_dim:      context embedding dimension
        target_shapes:  list of torch.Size for target network parameters
        total_params:   total number of parameters to generate
        chunk_alpha:    chunk size = alpha * hidden_size (default: 10)
        chunk_emb_size: per-chunk embedding dimension (default: 8)
        budget_factor:  HyperNet params ≈ budget_factor * target params (default: 1.0)
        do_hyperfan_init: use principled hyperfan initialization (default: True)
        verbose:        print construction info (default: False)
    """

    def __init__(self, input_dim, target_shapes, total_params,
                 chunk_alpha=10, chunk_emb_size=8, budget_factor=1.0,
                 do_hyperfan_init=True, norm_output=True, verbose=False):
        super().__init__()
        self.input_dim = input_dim
        self.total_params = total_params

        # Calculate optimal chunk_size and hidden layers to match parameter budget
        chunk_size, layers, hnet_params = _get_hidden_size(
            total_params * budget_factor,
            chunk_alpha,
            input_dim,
            chunk_emb_size,
            return_size=True,
        )

        # Handle edge case: if layers are too small
        if layers == [1, 1] and hnet_params >= total_params * budget_factor:
            layers = [1]

        # Ensure minimum hidden size for stability
        layers = [max(l, 4) for l in layers]

        if verbose:
            print(f"  ChunkedHyperNetWrapper: input={input_dim}, target_params={total_params}, "
                  f"chunk_size={chunk_size}, layers={layers}, hnet_params≈{hnet_params}")

        self.hnet = ChunkedHMLP(
            target_shapes=target_shapes,
            chunk_size=chunk_size,
            cond_in_size=input_dim,
            layers=tuple(layers),
            num_cond_embs=0,
            no_cond_weights=True,
            chunk_emb_size=chunk_emb_size,
            activation_fn=nn.ReLU(),
            verbose=verbose,
        )

        if do_hyperfan_init:
            self.hnet.apply_chunked_hyperfan_init()

        # --- Stability: L2 Norm + learnable scalar output scale (v4.0) ---
        # L2 Norm decouples direction from magnitude, preventing weight norm drift.
        # output_scale init=0.01: after L2 Norm (||raw||=1), actual magnitude = 0.01
        # This ensures gamma/beta ≈ 0.01 -> AdaLN output ≈ LayerNorm output -> "quiet start"
        self.norm_output = norm_output
        self.output_scale = nn.Parameter(torch.tensor(0.01))

        # Store actual hnet parameter count for reference
        self._hnet_num_params = self.hnet.num_params

    def forward(self, context):
        """
        Generate flat parameter vector from context.

        Pipeline: ChunkedHMLP -> L2 Norm (optional) -> scalar scale (init=0.01)

        Args:
            context: (B, input_dim) context embedding
        Returns:
            flat_params: (B, total_params)
        """
        raw = self.hnet.forward(cond_input=context, ret_format="flattened")
        # [v4.0] L2 normalization: fix direction distribution, magnitude = 1.0
        if self.norm_output:
            norms = torch.linalg.norm(raw, dim=-1, keepdim=True)
            raw = raw / (norms + 1e-8)
        return raw * self.output_scale  # (B, total_params) initially magnitude = 0.01


class ChunkedDualHyperNetwork(nn.Module):
    """
    Three-way hypernetwork using ChunkedHMLP for parameter efficiency.

    Same interface as DualHyperNetwork but uses ChunkedHMLP internally.

    Generates weights for three functional networks:
        - θ_state  = hyper_trans(rule_emb)           — objective physics
        - θ_reward = hyper_rew(rule_emb ⊕ id_emb)   — subjective reward
        - θ_pred   = hyper_pred(rule_emb ⊕ id_emb)  — subjective policy+value

    Args:
        rule_emb_dim:      dimension of rule embedding
        id_emb_dim:        dimension of agent id embedding
        trans_shapes:      target_shapes for FunctionalStateTransNet
        trans_param_count: total params for FunctionalStateTransNet
        rew_shapes:        target_shapes for FunctionalRewardHead
        rew_param_count:   total params for FunctionalRewardHead
        pred_shapes:       target_shapes for FunctionalPredictionNet
        pred_param_count:  total params for FunctionalPredictionNet
        chunk_alpha:       chunk alpha (default: 10)
        chunk_emb_size:    chunk embedding size (default: 8)
        budget_factor:     parameter budget ratio (default: 1.0)
        do_hyperfan_init:  use hyperfan initialization (default: True)
        verbose:           print construction info (default: True)
    """

    def __init__(self, rule_emb_dim, id_emb_dim,
                 trans_shapes, trans_param_count,
                 rew_shapes, rew_param_count,
                 pred_shapes, pred_param_count,
                 chunk_alpha=10, chunk_emb_size=8,
                 budget_factor=1.0, do_hyperfan_init=True,
                 verbose=True):
        super().__init__()

        aug_dim = rule_emb_dim + id_emb_dim

        if verbose:
            print(f"ChunkedDualHyperNetwork:")
            print(f"  rule_emb_dim={rule_emb_dim}, id_emb_dim={id_emb_dim}")
            print(f"  trans_params={trans_param_count}, rew_params={rew_param_count}, "
                  f"pred_params={pred_param_count}")

        # Objective: Rule only -> θ_state
        self.hyper_trans = ChunkedHyperNetWrapper(
            input_dim=rule_emb_dim,
            target_shapes=trans_shapes,
            total_params=trans_param_count,
            chunk_alpha=chunk_alpha,
            chunk_emb_size=chunk_emb_size,
            budget_factor=budget_factor,
            do_hyperfan_init=do_hyperfan_init,
            verbose=verbose,
        )

        # Subjective: Rule + Agent ID -> θ_reward
        self.hyper_rew = ChunkedHyperNetWrapper(
            input_dim=aug_dim,
            target_shapes=rew_shapes,
            total_params=rew_param_count,
            chunk_alpha=chunk_alpha,
            chunk_emb_size=chunk_emb_size,
            budget_factor=budget_factor,
            do_hyperfan_init=do_hyperfan_init,
            verbose=verbose,
        )

        # Subjective: Rule + Agent ID -> θ_pred
        self.hyper_pred = ChunkedHyperNetWrapper(
            input_dim=aug_dim,
            target_shapes=pred_shapes,
            total_params=pred_param_count,
            chunk_alpha=chunk_alpha,
            chunk_emb_size=chunk_emb_size,
            budget_factor=budget_factor,
            do_hyperfan_init=do_hyperfan_init,
            verbose=verbose,
        )

        self.trans_param_count = trans_param_count
        self.rew_param_count = rew_param_count
        self.pred_param_count = pred_param_count

        if verbose:
            total_hnet = (self.hyper_trans._hnet_num_params +
                         self.hyper_rew._hnet_num_params +
                         self.hyper_pred._hnet_num_params)
            total_target = trans_param_count + rew_param_count + pred_param_count
            print(f"  Total HyperNet params: {total_hnet:,} "
                  f"(target: {total_target:,}, ratio: {total_hnet/total_target:.2f}x)")

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