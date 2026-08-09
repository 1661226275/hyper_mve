"""HyperMAMuZeroNet — stage-3 subjective model (fork 4b).

MAZero six-function architecture with the subjective pathway swapped in:

  objective (inherited, shared SGD): representation h, attention communication
      e + individual dynamics g (residual), SimSiam projection.
  subjective (hypernet-generated, per agent):
      R_i : FunctionalRewardHead over the GLOBAL concat [all latents, joint
            action]; θ_rew^i = hyper_rew(ctx_i) from the posterior context.
      V_i : FunctionalValueHead over the GLOBAL concat of latents;
            **Bayes-averaged leaf evaluation** — enumerate g ∈ G, generate
            θ_val^{i,g} = hyper_pred(ctx_i(g)) with the belief slot replaced
            by onehot(g), evaluate V^{i,g}, combine Σ_g b_i(g)·V^{i,g}.
            (--belief_point_estimate switches to a single posterior-ctx head:
            the D.3-i ablation arm.)
      P_i : shared SGD policy head on [own latent ‖ ctx_i] — input
            conditioning with the belief-blended point estimate (the
            deliberate conditioning asymmetry of report §2.2.2).

Context protocol (v5 6-API analog): ``set_belief(g_hat, step)`` must be called
before ``initial_inference`` at every REAL step; the context and generated
parameters are (re)built inside ``initial_inference`` (rows are extracted from
the observation's trailing row block) and stay FROZEN through all subsequent
``recurrent_inference`` calls (search / unroll).

Scalar-transform mode only for now (support size 1); the vector/categorical
support extension is deferred (v5 functional heads emit 1-dim outputs).
"""
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import BaseNet, NetworkOutput, Action, HiddenState

from .model import mlp, RepresentationNetwork, ProjectionNetwork
from .attention import AttentionEncoder

from hyper_mve.algo.modules.tri_context_encoder import TriContextEncoder
from hyper_mve.algo.modules.belief_net import BeliefNet
from hyper_mve.algo.modules.hyper_network import DualHyperNetwork
from hyper_mve.algo.modules.functional_nets import (
    FunctionalRewardHead,
    count_generated,
    split_generated,
    adaln_modulate,
    functional_linear,
)
from hyper_mve.algo.modules.grad_gating import BeliefGradGating


class FunctionalValueHead(nn.Module):
    """Per-agent value head with hypernet-generated FiLM + output layer.

    Mirrors FunctionalRewardHead's film_head structure without the action
    input: global_state -> fc1(SGD)+FiLM -> fc2(SGD)+FiLM -> head(generated).
    """

    def __init__(self, global_dim, hidden_dim=128):
        super().__init__()
        self.layer_specs = [
            (global_dim, hidden_dim, True),
            (hidden_dim, hidden_dim, True),
            (hidden_dim, 1, False),
        ]
        self.fc1 = nn.Linear(global_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.generated_param_count, self.gen_spec, self.gen_groups = count_generated(
            self.layer_specs, "film_head"
        )

    def forward(self, global_state, flat_params):
        (g1, b1), (g2, b2), (w3, bias3) = split_generated(flat_params, self.gen_spec)
        x = adaln_modulate(self.fc1(global_state), g1, b1)
        x = adaln_modulate(self.fc2(x), g2, b2)
        return functional_linear(x, w3, bias3)      # (B*, 1)

    def forward_multi(self, global_state, flat_params_stacked):
        """Evaluate |G| parameter sets against ONE shared ``global_state``.

        Numerically identical to looping :meth:`forward` over the regime axis
        and stacking, but issues a constant number of kernels instead of one
        set per regime — the Bayes-averaged leaf evaluation is on the MCTS hot
        path (every ``initial_inference`` / ``recurrent_inference``), so the
        per-launch overhead dominated the batch-1 search cost.

        ``fc1`` is regime-independent (it consumes only ``global_state``), so
        it is evaluated once and broadcast; only the FiLM modulation and the
        generated output layer are genuinely per-regime.

        Args:
            global_state:        (B*, global_dim)
            flat_params_stacked: (G, B*, P)
        Returns:
            (B*, 1, G) — the regime axis LAST, ready to multiply by beliefs.
        """
        G = flat_params_stacked.shape[0]
        Bn = global_state.shape[0]
        flat = flat_params_stacked.reshape(G * Bn, -1)
        (g1, b1), (g2, b2), (w3, bias3) = split_generated(flat, self.gen_spec)
        # (B*, hid) -> (G*B*, hid), tiled to match flat's [g0 rows; g1 rows; …]
        h1 = self.fc1(global_state).repeat(G, 1)
        x = adaln_modulate(h1, g1, b1)
        x = adaln_modulate(self.fc2(x), g2, b2)
        out = functional_linear(x, w3, bias3)           # (G*B*, 1)
        return out.reshape(G, Bn, 1).permute(1, 2, 0)   # (B*, 1, G)


class ObjectiveDynamics(nn.Module):
    """Inherited objective transition: attention communication + residual
    per-agent dynamics (MAZero's e + g, reward head removed)."""

    def __init__(self, hidden_state_size, action_space_size, fc_dynamic_layers):
        super().__init__()
        self.attention_stack = nn.Sequential(
            nn.Linear(hidden_state_size + action_space_size, hidden_state_size),
            nn.ReLU(),
            AttentionEncoder(3, hidden_state_size, hidden_state_size, dropout=0.1),
        )
        # use_value_out=True strips the trailing ReLU + LayerNorm that mlp()
        # otherwise appends after EVERY layer, including the output.
        #
        # Without it this branch is the only head in the model whose output is
        # forced non-negative and then renormalized — fc_reward, fc_value and
        # fc_policy all pass use_value_out=True. A residual transition has to
        # emit SIGNED deltas, and measured on two independently trained
        # checkpoints (2026-07-19) that output ReLU had died to 1/128 live
        # units, versus 64/128 at initialization. The branch then emitted a
        # near-constant, so `state + hidden_state` became a frozen-world
        # identity map: ‖h'(a) − h'(NOOP)‖ ≈ 2e-7 for every action, at any
        # action amplitude and any state scale. MCTS saw identical values on
        # every branch, only the reward head could rank actions, and the policy
        # distilled to 100% HARVEST. Dead ReLUs get no gradient, so this is an
        # absorbing state that neither budget nor exploration can escape.
        #
        # Disclosed vendor deviation: upstream MAZero has the same latent
        # issue (vendor/MAZero/config/{smac,matrix}/model.py build fc_dynamic
        # without use_value_out). Zero-init of the final Linear that comes with
        # this flag is standard residual practice — it starts the transition at
        # identity, but with a live gradient path, unlike a dead ReLU.
        self.fc_dynamic = mlp(
            hidden_state_size + action_space_size + hidden_state_size,
            fc_dynamic_layers, hidden_state_size, use_value_out=True,
        )

    def forward(self, hidden_state, action):
        # hidden_state (B, N, H), action one-hot (B, N, A)
        B, N = hidden_state.size(0), hidden_state.size(1)
        attn = self.attention_stack(torch.cat([hidden_state, action], dim=2))
        concat = torch.cat([hidden_state, action, attn], dim=2)
        state = self.fc_dynamic(concat.reshape(B * N, -1)).reshape(B, N, -1)
        return state + hidden_state


class HyperMAMuZeroNet(BaseNet):
    def __init__(
        self,
        num_agents: int,
        observation_shape: Tuple[int, ...],
        action_space_size: int,
        hidden_state_size: int,
        fc_representation_layers: List[int],
        fc_dynamic_layers: List[int],
        fc_policy_layers: List[int],
        inverse_value_transform,
        inverse_reward_transform,
        env_cfg=None,                    # v5 EnvConfig (regime family, N, K)
        model_cfg=None,                  # v5 ModelConfig (ctx dims, gru hidden)
        # None => derive from the regime family. A hardcoded default silently
        # disagrees with the family the moment one is added or resized, and the
        # disagreement surfaces as a shape error deep in the belief encoder.
        n_regimes: int = None,
        belief_point_estimate: bool = False,
        belief_blind: bool = False,
        value_hard_select: bool = False,
        belief_grad_gating_steps: int = 5000,
        conditioning: str = "hyper",

        proj_hid: int = 256, proj_out: int = 256,
        pred_hid: int = 64, pred_out: int = 256,
        use_feature_norm: bool = True,
        **kwargs,
    ):
        super().__init__(inverse_value_transform, inverse_reward_transform)
        self.num_agents = num_agents
        self.obs_size = int(np.prod(observation_shape))
        self.action_space_size = action_space_size
        self.hidden_state_size = hidden_state_size
        if n_regimes is None:
            from hyper_mve.utils.schemas.relation import get_regime_family
            n_regimes = get_regime_family(env_cfg).size
        self.n_regimes = n_regimes
        self.belief_point_estimate = belief_point_estimate
        self.belief_blind = belief_blind
        self.value_hard_select = value_hard_select

        # ---- objective pathway (inherited h / e+g / projection) ----
        self.representation_network = RepresentationNetwork(
            self.obs_size, hidden_state_size, fc_representation_layers, use_feature_norm
        )
        self.dynamics_network = ObjectiveDynamics(
            hidden_state_size, action_space_size, fc_dynamic_layers
        )
        self.projection_network = ProjectionNetwork(
            hidden_state_size * num_agents, proj_hid, proj_out, pred_hid, pred_out
        )

        # ---- subjective pathway (v5 modules, imported directly) ----
        self.belief_net = BeliefNet(env_cfg, model_cfg)
        self.ctx_encoder = TriContextEncoder(env_cfg, model_cfg)
        global_dim = hidden_state_size * num_agents
        joint_action_dim = num_agents * action_space_size
        self.reward_head = FunctionalRewardHead(
            latent_dim=global_dim, joint_action_dim=joint_action_dim,
            hidden_dim=128, gen_scope="film_head",
        )
        self.value_head = FunctionalValueHead(global_dim, hidden_dim=128)
        if conditioning == "hyper":
            self.hyper = DualHyperNetwork(
                ctx_aug_dim=self.ctx_encoder.d_ctx_aug,
                rew_param_count=self.reward_head.generated_param_count,
                pred_param_count=self.value_head.generated_param_count,
                rew_output_groups=self.reward_head.gen_groups,
                pred_output_groups=self.value_head.gen_groups,
            )
        else:
            # phase-7 conditioning-swap ablation arms (moe_router / film):
            # same ctx input + forward_subjective contract, θ-generation
            # mechanism swapped (controlled Direction-1 comparison).
            from hyper_mve.algo.modules.conditioning_variants import (
                build_conditioner,
            )
            self.hyper = build_conditioner(
                conditioning,
                ctx_aug_dim=self.ctx_encoder.d_ctx_aug,
                rew_param_count=self.reward_head.generated_param_count,
                pred_param_count=self.value_head.generated_param_count,
                rew_output_groups=self.reward_head.gen_groups,
                pred_output_groups=self.value_head.gen_groups,
            )
        self.conditioning = conditioning
        self.gating = BeliefGradGating(belief_grad_gating_steps)

        # policy: shared SGD head on [own latent ‖ ctx_i] (point-estimate ctx)
        self.fc_policy = mlp(
            hidden_state_size + self.ctx_encoder.d_ctx_aug,
            fc_policy_layers, action_space_size, use_value_out=True,
        )

        # cached context state (set per real step, frozen during search/unroll)
        self._belief = None          # (B, N, |G|) posterior/blend, gated
        self._ctx = None             # (B, N, 64)
        self._theta_rew = None       # (B*N, ·)
        self._theta_val = None       # point-estimate θ_val (B*N, ·)
        self._theta_val_G = None     # per-regime θ_val list len |G| of (B*N, ·)
        self._head_diversity = None  # across-regime variance of v_g (monitor)
        self._g_true = None          # oracle regime for hard-select value (train only)
        self._value_deploy_mode = "bayes"  # deploy-time value aggregation (diagnostic)
        self._value_deploy_temp = 1.0
        self._deploy_g = None        # oracle regime for the oracle-deploy diagnostic
        self._step = 0

    # ------------------------------------------------------------- context
    def set_belief(self, g_hat: torch.Tensor, step: int = None):
        """Cache the per-agent regime posterior (or curriculum blend) used to
        build the subjective context at the NEXT initial_inference call."""
        if step is not None:
            self._step = int(step)
        self._belief = g_hat

    def set_oracle_regime(self, g_true):
        """Cache the oracle regime id (B,) for hard-select value training, or None
        to disable it. Set only during the training forward and cleared after, so
        eval / reanalyze / deploy always Bayes-average regardless of train/eval mode."""
        self._g_true = g_true

    def set_value_deploy(self, mode="bayes", temp=1.0, deploy_g=None):
        """Deploy-time value aggregation over the per-regime heads (diagnostic).
        Default 'bayes' is bit-exact the trained behaviour. 'argmax' hard-selects
        the belief's top regime; 'temp' uses the tempered posterior w ∝ belief**(1/T)
        (T=1 -> Bayes, T->0 -> argmax); 'oracle' hard-selects deploy_g (cheating,
        diagnostic only). Affects only the Bayes/deploy path, never train hard-select."""
        self._value_deploy_mode = mode
        self._value_deploy_temp = float(temp)
        self._deploy_g = deploy_g

    def _extract_rows(self, obs_flat: torch.Tensor) -> torch.Tensor:
        # obs_flat (B, N, obs_size); own row w_i· = trailing (N-1) dims
        return obs_flat[..., -(self.num_agents - 1):]

    def _build_context(self, obs_flat: torch.Tensor):
        B, N, _ = obs_flat.shape
        device = obs_flat.device
        if self.belief_blind or self._belief is None or self._belief.shape[0] != B:
            # belief_blind (capacity-matched control) forces the uniform posterior;
            # same path as the cold fallback (workers set_belief per step otherwise).
            belief = torch.full((B, N, self.n_regimes), 1.0 / self.n_regimes, device=device)
        else:
            belief = self._belief.to(device)
        belief = self.gating.apply_raw(belief, self._step)

        rows = self._extract_rows(obs_flat)
        agent_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, N)

        ctx = self.ctx_encoder(agent_ids, rows, belief)          # (B, N, 64)
        ctx = self.gating.apply_ctx(ctx, self._step)
        self._ctx = ctx

        ctx_flat = ctx.reshape(B * N, -1)
        theta_rew, theta_val = self.hyper.forward_subjective(ctx_flat)
        self._theta_rew, self._theta_val = theta_rew, theta_val

        if not self.belief_point_estimate:
            # per-regime value heads for the Bayes-averaged leaf evaluation
            thetas = []
            for g in range(self.n_regimes):
                onehot = torch.zeros(B, N, self.n_regimes, device=device)
                onehot[..., g] = 1.0
                ctx_g = self.ctx_encoder(agent_ids, rows, onehot)
                if self.hyper.share_subjective_trunk:
                    _, th = self.hyper.forward_subjective(ctx_g.reshape(B * N, -1))
                else:
                    th = self.hyper.hyper_pred(ctx_g.reshape(B * N, -1))
                thetas.append(th)
            self._theta_val_G = thetas
        self._belief_probs = belief                              # (B, N, |G|)

    # ------------------------------------------------------------- heads
    def _global_state(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # hidden_state (B, N*H) -> per-agent replicated global input (B*N, N*H)
        B = hidden_state.shape[0]
        return hidden_state.unsqueeze(1).expand(B, self.num_agents, -1).reshape(
            B * self.num_agents, -1
        )

    def _predict_value(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Bayes-averaged (or point-estimate) per-agent values (B, N, 1)."""
        B = hidden_state.shape[0]
        gs = self._global_state(hidden_state)
        if self.belief_point_estimate:
            v = self.value_head(gs, self._theta_val)             # (B*N, 1)
            return v.reshape(B, self.num_agents, 1)
        # one batched pass over the regime family (see forward_multi) — the
        # per-regime Python loop this replaces was ~38% of search step time
        v_g = self.value_head.forward_multi(
            gs, torch.stack(self._theta_val_G, dim=0))           # (B*N, 1, |G|)
        # head-diversity monitor (belief_blind caveat): across-regime variance of
        # the per-regime value heads. Near 0 => the hypernet collapsed to identical
        # heads, so uniform averaging changes nothing and a belief_blind tie is
        # uninformative.
        self._head_diversity = v_g.detach().float().var(dim=-1).mean()
        if self.value_hard_select and self._g_true is not None:
            # gradient-diffusion fix (train only): route the value gradient to the
            # TRUE-regime head only (v = v_{g_true}) so each head gets a clean
            # per-regime signal instead of the belief-diffused blend. self._g_true
            # is set solely during the training forward and cleared after, so
            # eval / reanalyze / deploy fall through to the Bayes average below.
            idx = (self._g_true.view(B, 1).expand(B, self.num_agents)
                   .reshape(B * self.num_agents).clamp(min=0))
            v = v_g.gather(dim=-1, index=idx.view(-1, 1, 1)).squeeze(-1)   # (B*N, 1)
            return v.reshape(B, self.num_agents, 1)
        # _belief_probs was built at the ROOT batch size by _build_context. This
        # reshape is only valid because the search expands exactly one leaf per
        # tree per simulation, so every recurrent_inference batch matches the
        # root batch. Silent corruption if that ever changes.
        assert self._belief_probs.shape[0] == B, (
            f"belief batch {self._belief_probs.shape[0]} != hidden-state batch {B}; "
            "set_belief/initial_inference must run at the same batch size as the "
            "search that follows"
        )
        b = self._belief_probs.reshape(B * self.num_agents, 1, self.n_regimes)
        mode = self._value_deploy_mode
        if mode == "oracle" and self._deploy_g is not None:
            # cheating diagnostic: hard-select the TRUE regime head at deploy.
            v = v_g[..., int(self._deploy_g)]                    # (B*N, 1)
        elif mode == "argmax":
            idx = b.argmax(dim=-1, keepdim=True)                 # (B*N, 1, 1)
            v = v_g.gather(dim=-1, index=idx).squeeze(-1)        # (B*N, 1)
        elif mode == "temp":
            # tempered posterior w ∝ belief**(1/T); T=1 -> Bayes, T->0 -> argmax.
            w = torch.softmax(torch.log(b.clamp_min(1e-9)) / self._value_deploy_temp, dim=-1)
            v = (v_g * w).sum(-1)                                # (B*N, 1)
        else:  # bayes (default, bit-exact the trained behaviour)
            v = (v_g * b).sum(-1)                                # (B*N, 1)
        return v.reshape(B, self.num_agents, 1)

    def _predict_reward(self, hidden_state: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        # action_onehot (B, N, A) -> joint (B, N*A) replicated per agent
        B = hidden_state.shape[0]
        gs = self._global_state(hidden_state)
        ja = action_onehot.reshape(B, -1).unsqueeze(1).expand(
            B, self.num_agents, -1
        ).reshape(B * self.num_agents, -1)
        r = self.reward_head(gs, ja, self._theta_rew)            # (B*N, 1)
        return r.reshape(B, self.num_agents, 1)

    def prediction(self, hidden_state: torch.Tensor):
        B = hidden_state.shape[0]
        value = self._predict_value(hidden_state)
        s = hidden_state.reshape(B, self.num_agents, self.hidden_state_size)
        pol_in = torch.cat([s, self._ctx], dim=-1)
        policy_logit = self.fc_policy(
            pol_in.reshape(B * self.num_agents, -1)
        ).reshape(B, self.num_agents, self.action_space_size)
        return policy_logit, value

    def representation(self, observation: torch.Tensor) -> torch.Tensor:
        batch_size = observation.shape[0]
        hidden_state = self.representation_network(
            observation.reshape(batch_size * self.num_agents, self.obs_size)
        ).reshape(batch_size, self.num_agents * self.hidden_state_size)
        return hidden_state

    def dynamics(self, hidden_state: torch.Tensor, action: torch.Tensor):
        batch_size = hidden_state.shape[0]
        action_onehot = (
            torch.zeros((batch_size * self.num_agents, self.action_space_size))
            .to(action.device).float()
        )
        action_onehot.scatter_(1, action.reshape(batch_size * self.num_agents, 1).long(), 1.0)
        action_onehot = action_onehot.reshape(batch_size, self.num_agents, self.action_space_size)

        next_hidden = self.dynamics_network(
            hidden_state.reshape(batch_size, self.num_agents, self.hidden_state_size),
            action_onehot,
        ).reshape(batch_size, self.num_agents * self.hidden_state_size)
        # r = R(s, a): reward from the CURRENT state (v5 causality), global input
        reward = self._predict_reward(hidden_state, action_onehot)
        return next_hidden, reward

    def project(self, hidden_state: torch.Tensor, with_grad: bool = True) -> torch.Tensor:
        proj = self.projection_network.project(hidden_state)
        if with_grad:
            return self.projection_network.predict(proj)
        return proj.detach()

    def initial_inference(self, observation: torch.Tensor) -> NetworkOutput:
        batch_size = observation.size(0)
        obs_flat = observation.reshape(batch_size, self.num_agents, self.obs_size)
        self._build_context(obs_flat)

        hidden_state = self.representation(observation)
        policy_logit, value = self.prediction(hidden_state)

        if not self.training:
            value = self.inverse_value_transform(value).detach().cpu().numpy()
            policy_logit = policy_logit.detach().cpu().numpy()

        return NetworkOutput(
            hidden_state, np.zeros((batch_size, self.num_agents, 1)), value, policy_logit
        )

    def recurrent_inference(self, hidden_state: HiddenState, action: Action) -> NetworkOutput:
        next_hidden_state, reward = self.dynamics(hidden_state, action)
        policy_logit, value = self.prediction(next_hidden_state)

        if not self.training:
            reward = self.inverse_reward_transform(reward).detach().cpu().numpy()
            value = self.inverse_value_transform(value).detach().cpu().numpy()
            policy_logit = policy_logit.detach().cpu().numpy()

        return NetworkOutput(next_hidden_state, reward, value, policy_logit)
