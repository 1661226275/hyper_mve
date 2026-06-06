"""
Functional Networks for Hyper-MuZero (v3.2).

These networks use EXTERNALLY provided weights (from HyperNetwork)
instead of internal nn.Parameter. This enables dynamic weight generation.

v3.2 Changes:
    - AdaLN (Adaptive LayerNorm) on hidden layers: HyperNet generates
      gamma/beta per layer to cut the "weight explosion -> activation explosion" chain
    - Residual connection in StateTransNet: s' = s + net(s, a; θ)
    - Output layers (FC3/PolicyHead/ValueHead) do NOT use AdaLN

Design:
    - functional_linear: single layer with external weight/bias
    - adaln_forward: linear + LayerNorm + adaptive gamma/beta + ReLU
    - FunctionalStateTransNet: (s, A_joint) -> s' with AdaLN + residual
    - FunctionalRewardHead: (s, A_joint) -> r_i with AdaLN
    - FunctionalPredictionNet: s -> (p_i[5], v_i[1]) with AdaLN in trunk

Note: These nets do NOT have agent_id as input (unlike Baseline).
      Agent identity is encoded in the HyperNet-generated weights.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def functional_linear(x, weight, bias):
    """
    Functional linear layer: y = x @ W^T + b

    Args:
        x:      (B, in_features)
        weight: (B, out_features, in_features) — batched weights
        bias:   (B, out_features) — batched bias
    Returns:
        y:      (B, out_features)
    """
    # (B, 1, in) @ (B, in, out) -> (B, 1, out) -> (B, out)
    return torch.bmm(x.unsqueeze(1), weight.transpose(1, 2)).squeeze(1) + bias


def adaln_forward(x, weight, bias, gamma, beta):
    """
    Adaptive LayerNorm forward: Linear -> LayerNorm -> (1+gamma)*x + beta -> ReLU

    Cuts the "weight explosion -> activation explosion" chain by normalizing
    activations before applying the HyperNet-generated affine transform.

    v4.6 fix — Residual modulation (1 + gamma):
        When HyperNet output_scale is small (start of training), gamma ≈ 0,
        so the layer acts as an identity pass-through: h ≈ h_norm.
        Without this, gamma ≈ 0 crushes h_norm to zero, destroying all input
        signal (including action information) and making the functional network
        output a constant regardless of input.
        This is standard in DiT, StyleGAN, FiLM-style conditional normalization.

    Args:
        x:      (B, in_features) — input
        weight: (B, out_features, in_features) — batched weights
        bias:   (B, out_features) — batched bias
        gamma:  (B, out_features) — AdaLN scale (from HyperNet)
        beta:   (B, out_features) — AdaLN shift (from HyperNet)
    Returns:
        h:      (B, out_features) — activated output
    """
    # Linear
    h = functional_linear(x, weight, bias)
    # Instance-wise LayerNorm (per-sample normalization over feature dim)
    mean = h.mean(dim=-1, keepdim=True)
    var = h.var(dim=-1, keepdim=True, unbiased=False)
    h = (h - mean) / torch.sqrt(var + 1e-5)
    # Adaptive affine transform with residual modulation (v4.6)
    # (1 + gamma) ensures identity pass-through when gamma ≈ 0
    h = h * (1 + gamma) + beta
    # Activation
    return F.relu(h)


def adaln_modulate(h, gamma, beta):
    """FiLM-style modulation for the ``film_head`` gen_scope.

    Same math as ``adaln_forward``'s tail (instance LayerNorm -> (1+gamma)*h + beta
    -> ReLU), but the linear that produced ``h`` is a SHARED SGD ``nn.Linear`` rather
    than a HyperNet-generated weight. Only gamma/beta are generated here.

    The normalization is the inline, PARAMETER-FREE instance LayerNorm (mean/var over
    the feature dim) — NOT a parametric nn.LayerNorm — to match the FULL-mode math and
    avoid stacking a second affine on top of (1 + gamma) + beta.

    Args:
        h:     (B, out_features) — output of a shared nn.Linear (pre-norm)
        gamma: (B, out_features) — AdaLN scale (from HyperNet)
        beta:  (B, out_features) — AdaLN shift (from HyperNet)
    Returns:
        (B, out_features) — activated, modulated output
    """
    mean = h.mean(dim=-1, keepdim=True)
    var = h.var(dim=-1, keepdim=True, unbiased=False)
    h = (h - mean) / torch.sqrt(var + 1e-5)
    h = h * (1 + gamma) + beta
    return F.relu(h)


def count_params_adaln(layer_specs):
    """
    Count total parameters for layers with AdaLN support.

    Args:
        layer_specs: list of (in_dim, out_dim, use_adaln) tuples
            use_adaln=True: weight + bias + gamma + beta
            use_adaln=False: weight + bias only
    Returns:
        total parameter count, list of param counts per layer
    """
    counts = []
    for in_dim, out_dim, use_adaln in layer_specs:
        count = out_dim * in_dim + out_dim  # weight + bias
        if use_adaln:
            count += out_dim + out_dim  # gamma + beta
        counts.append(count)
    return sum(counts), counts


def split_params_adaln(flat_params, layer_specs):
    """
    Split a flat parameter vector into per-layer parameter tuples.

    For AdaLN layers: returns (weight, bias, gamma, beta)
    For plain layers: returns (weight, bias)

    Args:
        flat_params: (B, total_params) flat parameter tensor
        layer_specs: list of (in_dim, out_dim, use_adaln) tuples
    Returns:
        list of tuples:
            AdaLN layer:  (weight, bias, gamma, beta)
            Plain layer:  (weight, bias)
    """
    B = flat_params.shape[0]
    params = []
    offset = 0
    for in_dim, out_dim, use_adaln in layer_specs:
        # Weight
        w_size = out_dim * in_dim
        weight = flat_params[:, offset:offset + w_size].view(B, out_dim, in_dim)
        offset += w_size
        # Bias
        bias = flat_params[:, offset:offset + out_dim]
        offset += out_dim
        if use_adaln:
            # Gamma
            gamma = flat_params[:, offset:offset + out_dim]
            offset += out_dim
            # Beta
            beta = flat_params[:, offset:offset + out_dim]
            offset += out_dim
            params.append((weight, bias, gamma, beta))
        else:
            params.append((weight, bias))
    return params


def layer_specs_to_param_shapes(layer_specs):
    """
    Convert layer_specs to param_shapes list compatible with hypnettorch target_shapes.

    For AdaLN layers: weight, bias, gamma, beta
    For plain layers: weight, bias

    Args:
        layer_specs: list of (in_dim, out_dim, use_adaln) tuples
    Returns:
        list of torch.Size
    """
    shapes = []
    for in_dim, out_dim, use_adaln in layer_specs:
        shapes.append(torch.Size([out_dim, in_dim]))  # weight
        shapes.append(torch.Size([out_dim]))           # bias
        if use_adaln:
            shapes.append(torch.Size([out_dim]))       # gamma
            shapes.append(torch.Size([out_dim]))       # beta
    return shapes


# ============================================================
# Unified descriptor-driven partial-generation helpers
# (gen_scope in {"film_head", "base_gen"}; "full" stays on the legacy
#  count_params_adaln / split_params_adaln path with gen_groups=None)
# ============================================================

def plan_generated_layers(layer_specs, gen_scope):
    """Map ``(in, out, use_adaln)`` specs + ``gen_scope`` -> per-layer generation kinds.

    Kinds (per layer):
        "sgd_base"  net owns a plain SGD ``nn.Linear`` (+ norm); NOTHING generated.
        "gen_film"  HyperNet generates gamma/beta only; the weight is a shared SGD
                    ``nn.Linear`` (film_head hidden layers).
        "gen_full"  HyperNet generates weight + bias + gamma + beta, applied via
                    ``adaln_forward`` (full hidden layers; base_gen fc2).
        "gen_head"  HyperNet generates weight + bias only (plain output head).

    Mapping (AdaLN layers = the ``use_adaln=True`` hidden layers, in order):
        full:      every AdaLN layer -> gen_full;              head -> gen_head
        film_head: every AdaLN layer -> gen_film;              head -> gen_head
        base_gen:  FIRST AdaLN -> sgd_base, REST -> gen_full;  head -> gen_head
    """
    plan = []
    adaln_idx = 0
    for in_dim, out_dim, use_adaln in layer_specs:
        if use_adaln:
            if gen_scope == "film_head":
                kind = "gen_film"
            elif gen_scope == "base_gen":
                kind = "sgd_base" if adaln_idx == 0 else "gen_full"
            elif gen_scope == "full":
                kind = "gen_full"
            else:
                raise ValueError(
                    f"plan_generated_layers: unknown gen_scope {gen_scope!r}"
                )
            adaln_idx += 1
        else:
            kind = "gen_head"
        plan.append((in_dim, out_dim, kind))
    return plan


def count_generated(layer_specs, gen_scope):
    """Count HyperNet-generated params and the role-grouped norm groups.

    The generated vector is role-grouped and contiguous::

        [ all FiLM gamma/beta (layer order, gamma then beta)
        | all generated weights+biases (layer order, weight then bias) ]

    so ``gen_groups = [film_total, weight_total]`` (zeros dropped) feeds HyperNetMLP's
    contiguous per-group RMS norm. For ``film_head`` the weight segment is head-only,
    reproducing the legacy ``[film_total, head_total]`` groups + per-element layout
    exactly. For ``base_gen`` the gen_full fc2 contributes gamma/beta to the FiLM group
    and weight/bias to the weight group.

    Returns ``(total, plan, groups)`` where ``plan`` is the ``plan_generated_layers``
    output (stored as the net's ``gen_spec`` and consumed by ``split_generated``).
    """
    plan = plan_generated_layers(layer_specs, gen_scope)
    film_total = 0
    weight_total = 0
    for in_dim, out_dim, kind in plan:
        if kind == "gen_film":
            film_total += 2 * out_dim
        elif kind == "gen_full":
            film_total += 2 * out_dim
            weight_total += out_dim * in_dim + out_dim
        elif kind == "gen_head":
            weight_total += out_dim * in_dim + out_dim
        # "sgd_base": nothing generated
    total = film_total + weight_total
    groups = [g for g in (film_total, weight_total) if g > 0]
    return total, plan, groups


def split_generated(flat_params, plan):
    """Split a role-grouped flat vector into per-layer tuples, in ``plan`` order.

    The vector is ``[ FiLM region | weight region ]``; a ``gen_full`` layer draws its
    gamma/beta from the FiLM region AND its weight/bias from the weight region (two
    independent cursors). Per-layer outputs (unpacked positionally by the net forward):

        "sgd_base" -> None
        "gen_film" -> (gamma, beta)
        "gen_full" -> (weight, bias, gamma, beta)
        "gen_head" -> (weight, bias)
    """
    B = flat_params.shape[0]
    film_len = sum(
        2 * out_dim for (_, out_dim, kind) in plan if kind in ("gen_film", "gen_full")
    )
    f_off = 0
    w_off = film_len
    out = []
    for in_dim, out_dim, kind in plan:
        if kind == "sgd_base":
            out.append(None)
        elif kind == "gen_film":
            gamma = flat_params[:, f_off:f_off + out_dim]
            f_off += out_dim
            beta = flat_params[:, f_off:f_off + out_dim]
            f_off += out_dim
            out.append((gamma, beta))
        elif kind == "gen_full":
            gamma = flat_params[:, f_off:f_off + out_dim]
            f_off += out_dim
            beta = flat_params[:, f_off:f_off + out_dim]
            f_off += out_dim
            w_size = out_dim * in_dim
            weight = flat_params[:, w_off:w_off + w_size].view(B, out_dim, in_dim)
            w_off += w_size
            bias = flat_params[:, w_off:w_off + out_dim]
            w_off += out_dim
            out.append((weight, bias, gamma, beta))
        else:  # "gen_head"
            w_size = out_dim * in_dim
            weight = flat_params[:, w_off:w_off + w_size].view(B, out_dim, in_dim)
            w_off += w_size
            bias = flat_params[:, w_off:w_off + out_dim]
            w_off += out_dim
            out.append((weight, bias))
    return out


def count_generated_film_head(layer_specs):
    """[deprecated] Thin delegator to ``count_generated(layer_specs, "film_head")``."""
    return count_generated(layer_specs, "film_head")


def split_generated_film_head(flat_params, plan):
    """[deprecated] Thin delegator to ``split_generated(flat_params, plan)``."""
    return split_generated(flat_params, plan)


# ============================================================
# Legacy helpers (for Baseline models that don't use AdaLN)
# ============================================================

def count_params(layer_dims):
    """
    Count total parameters for a list of (in_dim, out_dim) layer specs.
    Each layer has weight (out * in) + bias (out) parameters.
    """
    counts = []
    for in_dim, out_dim in layer_dims:
        counts.append(out_dim * in_dim + out_dim)
    return sum(counts), counts


def split_params(flat_params, layer_dims):
    """
    Split a flat parameter vector into per-layer (weight, bias) pairs.
    """
    B = flat_params.shape[0]
    params = []
    offset = 0
    for in_dim, out_dim in layer_dims:
        w_size = out_dim * in_dim
        b_size = out_dim
        weight = flat_params[:, offset:offset + w_size].view(B, out_dim, in_dim)
        offset += w_size
        bias = flat_params[:, offset:offset + b_size]
        offset += b_size
        params.append((weight, bias))
    return params


def layer_dims_to_param_shapes(layer_dims):
    """
    Convert layer_dims to param_shapes list (legacy, no AdaLN).
    """
    shapes = []
    for in_dim, out_dim in layer_dims:
        shapes.append(torch.Size([out_dim, in_dim]))
        shapes.append(torch.Size([out_dim]))
    return shapes


# ============================================================
# Functional Networks (v3.2 with AdaLN + Residual)
# ============================================================

class FunctionalStateTransNet(nn.Module):
    """
    Functional state transition network with AdaLN and residual connection.

    Architecture:
        [s, A_joint] -> FC1 -> AdaLN -> ReLU
                      -> FC2 -> AdaLN -> ReLU
                      -> FC3 (plain) -> LayerNorm -> delta_s
        s' = s + delta_s  (residual)

    AdaLN on FC1/FC2: HyperNet generates weight, bias, gamma, beta
    FC3 (output): HyperNet generates weight, bias only (no AdaLN)
    LayerNorm on output: fixed (not generated), normalizes latent state

    Input:  s (B, latent_dim), A_joint_onehot (B, joint_action_dim)
    Params: flat_params (B, total_params) from hyper_trans
    Output: s' (B, latent_dim)
    """

    def __init__(self, latent_dim, joint_action_dim, hidden_dim=128, gen_scope="full"):
        super().__init__()
        self.latent_dim = latent_dim
        self.gen_scope = gen_scope
        input_dim = latent_dim + joint_action_dim

        # (in_dim, out_dim, use_adaln)
        self.layer_specs = [
            (input_dim, hidden_dim, True),    # fc1 + AdaLN
            (hidden_dim, hidden_dim, True),   # fc2 + AdaLN
            (hidden_dim, latent_dim, False),  # fc3 (output, no AdaLN)
        ]
        self.total_params, self.param_counts = count_params_adaln(self.layer_specs)
        self.param_shapes = layer_specs_to_param_shapes(self.layer_specs)

        # Fixed LayerNorm for output normalization (not generated)
        self.ln = nn.LayerNorm(latent_dim)

        if gen_scope in ("film_head", "base_gen"):
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            if gen_scope == "film_head":
                # Shared SGD trunk; HyperNet generates only FiLM gamma/beta + output head.
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            else:  # "base_gen": fc1 is a plain SGD base (Linear+LN+ReLU, no FiLM);
                # fc2 weight is fully HyperNet-generated (AdaLN), head generated.
                self.ln1 = nn.LayerNorm(hidden_dim)
                assert sum(int(a) for _, _, a in self.layer_specs) >= 2, (
                    "base_gen requires >=2 AdaLN layers (fc1 SGD base + fc2 generated)"
                )
            self.generated_param_count, self.gen_spec, self.gen_groups = \
                count_generated(self.layer_specs, gen_scope)
        else:  # "full": HyperNet generates every layer's weights (legacy default)
            self.generated_param_count = self.total_params
            self.gen_spec = None
            self.gen_groups = None

    def forward(self, state, action_onehot, flat_params):
        """
        Args:
            state:         (B, latent_dim)
            action_onehot: (B, joint_action_dim)
            flat_params:   (B, generated_param_count)
        Returns:
            next_state: (B, latent_dim)
        """
        if self.gen_scope == "film_head":
            (g1, bt1), (g2, bt2), (w3, b3) = split_generated(flat_params, self.gen_spec)
            x = torch.cat([state, action_onehot], dim=-1)
            x = adaln_modulate(self.fc1(x), g1, bt1)   # fc1 shared SGD; gamma/beta generated
            x = adaln_modulate(self.fc2(x), g2, bt2)   # fc2 shared SGD; gamma/beta generated
            delta_s = functional_linear(x, w3, b3)     # generated head
            delta_s = self.ln(delta_s)                 # fixed output LayerNorm
            return state + delta_s                     # s' = s + Δs

        if self.gen_scope == "base_gen":
            _, (w2, b2, g2, bt2), (w3, b3) = split_generated(flat_params, self.gen_spec)
            x = torch.cat([state, action_onehot], dim=-1)
            x = F.relu(self.ln1(self.fc1(x)))          # fc1 plain SGD base (Linear+LN+ReLU)
            x = adaln_forward(x, w2, b2, g2, bt2)      # fc2 fully generated (AdaLN)
            delta_s = functional_linear(x, w3, b3)     # generated head
            delta_s = self.ln(delta_s)                 # fixed output LayerNorm
            return state + delta_s                     # s' = s + Δs

        params = split_params_adaln(flat_params, self.layer_specs)
        x = torch.cat([state, action_onehot], dim=-1)

        # FC1 + AdaLN + ReLU
        w1, b1, g1, bt1 = params[0]
        x = adaln_forward(x, w1, b1, g1, bt1)

        # FC2 + AdaLN + ReLU
        w2, b2, g2, bt2 = params[1]
        x = adaln_forward(x, w2, b2, g2, bt2)

        # FC3 (plain output)
        w3, b3 = params[2]
        delta_s = functional_linear(x, w3, b3)

        # Fixed LayerNorm + Residual connection
        delta_s = self.ln(delta_s)
        return state + delta_s  # s' = s + Δs


class FunctionalRewardHead(nn.Module):
    """
    Functional reward prediction network with AdaLN.

    Architecture:
        [s, A_joint] -> FC1 -> AdaLN -> ReLU
                      -> FC2 -> AdaLN -> ReLU
                      -> FC3 (plain) -> r_i

    AdaLN on FC1/FC2, plain FC3 output.
    Uses current state s (not s'), per MDP causality r = R(s, a).

    Input:  s (B, latent_dim), A_joint_onehot (B, joint_action_dim)
    Params: flat_params (B, total_params) from hyper_rew
    Output: r_i (B, 1)
    """

    def __init__(self, latent_dim, joint_action_dim, hidden_dim=128, gen_scope="full"):
        super().__init__()
        self.gen_scope = gen_scope
        input_dim = latent_dim + joint_action_dim

        # (in_dim, out_dim, use_adaln)
        self.layer_specs = [
            (input_dim, hidden_dim, True),    # fc1 + AdaLN
            (hidden_dim, hidden_dim, True),   # fc2 + AdaLN
            (hidden_dim, 1, False),           # fc3 (output, no AdaLN)
        ]
        self.total_params, self.param_counts = count_params_adaln(self.layer_specs)
        self.param_shapes = layer_specs_to_param_shapes(self.layer_specs)

        if gen_scope in ("film_head", "base_gen"):
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            if gen_scope == "film_head":
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            else:  # "base_gen": plain SGD fc1 base + generated fc2 (AdaLN)
                self.ln1 = nn.LayerNorm(hidden_dim)
                assert sum(int(a) for _, _, a in self.layer_specs) >= 2, (
                    "base_gen requires >=2 AdaLN layers (fc1 SGD base + fc2 generated)"
                )
            self.generated_param_count, self.gen_spec, self.gen_groups = \
                count_generated(self.layer_specs, gen_scope)
        else:
            self.generated_param_count = self.total_params
            self.gen_spec = None
            self.gen_groups = None

    def forward(self, state, action_onehot, flat_params):
        """
        Args:
            state:         (B, latent_dim)
            action_onehot: (B, joint_action_dim)
            flat_params:   (B, generated_param_count)
        Returns:
            reward: (B, 1) in scaled space
        """
        if self.gen_scope == "film_head":
            (g1, bt1), (g2, bt2), (w3, b3) = split_generated(flat_params, self.gen_spec)
            x = torch.cat([state, action_onehot], dim=-1)
            x = adaln_modulate(self.fc1(x), g1, bt1)
            x = adaln_modulate(self.fc2(x), g2, bt2)
            return functional_linear(x, w3, b3)

        if self.gen_scope == "base_gen":
            _, (w2, b2, g2, bt2), (w3, b3) = split_generated(flat_params, self.gen_spec)
            x = torch.cat([state, action_onehot], dim=-1)
            x = F.relu(self.ln1(self.fc1(x)))          # fc1 plain SGD base (Linear+LN+ReLU)
            x = adaln_forward(x, w2, b2, g2, bt2)      # fc2 fully generated (AdaLN)
            return functional_linear(x, w3, b3)

        params = split_params_adaln(flat_params, self.layer_specs)
        x = torch.cat([state, action_onehot], dim=-1)

        # FC1 + AdaLN + ReLU
        w1, b1, g1, bt1 = params[0]
        x = adaln_forward(x, w1, b1, g1, bt1)

        # FC2 + AdaLN + ReLU
        w2, b2, g2, bt2 = params[1]
        x = adaln_forward(x, w2, b2, g2, bt2)

        # FC3 (plain output)
        w3, b3 = params[2]
        x = functional_linear(x, w3, b3)

        return x


class FunctionalPredictionNet(nn.Module):
    """
    Functional policy + value prediction network with AdaLN.

    Architecture:
        s -> FC1 -> AdaLN -> ReLU -> FC2 -> AdaLN -> ReLU -> shared_features
        shared_features -> PolicyHead (plain) -> logits [num_actions]
        shared_features -> ValueHead  (plain) -> value  [1]

    AdaLN on shared trunk FC1/FC2, plain PolicyHead/ValueHead.

    Input:  s (B, latent_dim)
    Params: flat_params (B, total_params) from hyper_pred
    Output: policy_logits (B, num_actions), value (B, 1)
    """

    def __init__(self, latent_dim, num_actions=5, hidden_dim=128, gen_scope="full"):
        super().__init__()
        self.num_actions = num_actions
        self.gen_scope = gen_scope

        # (in_dim, out_dim, use_adaln)
        self.layer_specs = [
            (latent_dim, hidden_dim, True),       # fc1 (trunk) + AdaLN
            (hidden_dim, hidden_dim, True),       # fc2 (trunk) + AdaLN
            (hidden_dim, num_actions, False),     # policy_head (plain)
            (hidden_dim, 1, False),               # value_head (plain)
        ]
        self.total_params, self.param_counts = count_params_adaln(self.layer_specs)
        self.param_shapes = layer_specs_to_param_shapes(self.layer_specs)

        if gen_scope in ("film_head", "base_gen"):
            self.fc1 = nn.Linear(latent_dim, hidden_dim)
            if gen_scope == "film_head":
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            else:  # "base_gen": plain SGD fc1 base + generated fc2 (AdaLN)
                self.ln1 = nn.LayerNorm(hidden_dim)
                assert sum(int(a) for _, _, a in self.layer_specs) >= 2, (
                    "base_gen requires >=2 AdaLN layers (fc1 SGD base + fc2 generated)"
                )
            self.generated_param_count, self.gen_spec, self.gen_groups = \
                count_generated(self.layer_specs, gen_scope)
        else:
            self.generated_param_count = self.total_params
            self.gen_spec = None
            self.gen_groups = None

    def forward(self, state, flat_params):
        """
        Args:
            state:       (B, latent_dim)
            flat_params: (B, generated_param_count)
        Returns:
            policy_logits: (B, num_actions)
            value:         (B, 1) in scaled space
        """
        if self.gen_scope == "film_head":
            gen = split_generated(flat_params, self.gen_spec)
            (g1, bt1), (g2, bt2), (w_p, b_p), (w_v, b_v) = gen
            x = adaln_modulate(self.fc1(state), g1, bt1)   # shared trunk fc1
            x = adaln_modulate(self.fc2(x), g2, bt2)       # shared trunk fc2
            policy_logits = functional_linear(x, w_p, b_p)  # generated head
            value = functional_linear(x, w_v, b_v)          # generated head
            return policy_logits, value

        if self.gen_scope == "base_gen":
            gen = split_generated(flat_params, self.gen_spec)
            _, (w2, b2, g2, bt2), (w_p, b_p), (w_v, b_v) = gen
            x = F.relu(self.ln1(self.fc1(state)))          # fc1 plain SGD base (Linear+LN+ReLU)
            x = adaln_forward(x, w2, b2, g2, bt2)          # fc2 fully generated (AdaLN)
            policy_logits = functional_linear(x, w_p, b_p)  # generated head
            value = functional_linear(x, w_v, b_v)          # generated head
            return policy_logits, value

        params = split_params_adaln(flat_params, self.layer_specs)

        # Shared trunk: FC1 + AdaLN + ReLU
        w1, b1, g1, bt1 = params[0]
        x = adaln_forward(state, w1, b1, g1, bt1)

        # Shared trunk: FC2 + AdaLN + ReLU
        w2, b2, g2, bt2 = params[1]
        x = adaln_forward(x, w2, b2, g2, bt2)

        # Policy head (plain)
        w_p, b_p = params[2]
        policy_logits = functional_linear(x, w_p, b_p)

        # Value head (plain)
        w_v, b_v = params[3]
        value = functional_linear(x, w_v, b_v)

        return policy_logits, value
