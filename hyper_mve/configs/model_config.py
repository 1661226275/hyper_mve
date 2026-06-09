"""ModelConfig — network dimensions and hypernet configuration (Ch4.2-4.6, v4 Pass 2)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from hyper_mve.schemas._constants import (
    D_BELIEF,
    D_BELIEF_PROJ,
    D_CAP_EMB,
    D_C_CTX,
    D_ID_EMB,
    D_ROLE,
    D_TYPE_EMB,
)


_VALID_BELIEF_POOL: tuple[str, ...] = ("mean", "max", "attention")
_VALID_GEN_SCOPE: tuple[str, ...] = ("full", "film_head", "base_gen", "lora_fc2")


@dataclass(frozen=True)
class ModelConfig:
    """v4 model architecture configuration.

    Dimension fields default to the Ch4.2.2-4.2.4 hard constraints (id=8,
    type=8, cap=16 exactly fills role=32, with no padding; belief=32 =
    2 × proj=16). ``__post_init__`` re-checks the constraints so a user who
    accidentally overrides one number gets a clear failure.
    """

    # Representation
    latent_dim: int = 64
    hidden_dim: int = 128

    # Context-path dims (Ch4.2.4 hard constraints)
    d_c: int = D_C_CTX
    d_role: int = D_ROLE
    d_belief: int = D_BELIEF

    # role decomposition (Ch4.2.2)
    d_id_emb: int = D_ID_EMB
    d_type_emb: int = D_TYPE_EMB
    d_cap_emb: int = D_CAP_EMB

    # belief decomposition (Ch4.2.3)
    d_belief_proj: int = D_BELIEF_PROJ

    # HyperNet (Ch4.6 stability)
    hyper_hidden_dims: tuple[int, ...] = (256, 256)
    hyper_rew_hidden_dims: tuple[int, ...] = (256, 256, 256)

    # output_scale initialisation (Ch4.6 defence line 1)
    trans_output_scale_init: float = 0.01
    rew_output_scale_init: float = 0.1     # v4.7-tuned; bigger than trans/pred
    pred_output_scale_init: float = 0.01

    # HyperNet generation scope: "full" = generate every functional-net weight (legacy
    # default); "film_head" = shared SGD fc1/fc2 trunk + generate only FiLM gamma/beta +
    # output head (partial generation, grouped-RMS-normed); "base_gen" = plain SGD fc1
    # base (Linear+LN+ReLU, no FiLM) + fully generate fc2 (weight + FiLM gamma/beta) +
    # output head (CCWM fc_dynamics_1/fc_dynamics_2 style; more capacity than film_head,
    # more stable than full). See duo / duo_basegen presets.
    hyper_gen_scope: str = "full"

    # Share the subjective hypernet trunk: when True, hyper_rew + hyper_pred collapse to
    # one shared trunk over ctx_aug (depth = hyper_rew_hidden_dims) with two output heads
    # (theta_rew, theta_pred), each keeping its own output_scale + RMS groups. hyper_trans
    # (objective) stays separate. False (default) = two independent MLPs (legacy).
    share_subjective_trunk: bool = False

    # [LoRA] Output-layer low-rank factorization applied UNIFORMLY to all three hypernets
    # (hyper_trans / hyper_rew / hyper_pred): when set to r, each HyperNetMLP.output_layer
    # Linear(prev, pc) becomes Linear(prev, r, bias=False) -> Linear(r, pc). None (default)
    # = dense (legacy). NOTE: not yet wired for the shared SubjectiveHyperNet
    # (share_subjective_trunk=True); combining the two raises NotImplementedError.
    hyper_output_rank: Optional[int] = None

    # [lora_fc2] film_head-exclusive expressivity booster: per-context rank-r weight delta on
    # the shared fc2 (W2_eff = W2_base + Bf @ Af). Only takes effect when
    # hyper_gen_scope == "lora_fc2"; forbidden on base_gen (which generates fc2 fully).
    # Requires *_output_scale_init >= 0.05 so Delta_W stays expressive (see __post_init__).
    lora_fc2_rank: Optional[int] = None

    # AdaLN (Ch4.6 defence line 2)
    use_adaln: bool = True
    adaln_residual_one_plus: bool = True   # h × (1 + γ) + β

    # Δs residual in StateTransNet (Ch4.6)
    state_trans_residual: bool = True

    # BeliefNet (Ch4.2.3 / 4.5)
    belief_gru_hidden: int = 128
    belief_pool: str = "mean"

    # Projector (Ch5.8.2 BYOL consistency)
    proj_dim: int = 64

    def __post_init__(self) -> None:
        expected_role = self.d_id_emb + self.d_type_emb + self.d_cap_emb
        if expected_role != self.d_role:
            raise ValueError(
                f"d_role mismatch: d_id({self.d_id_emb}) + d_type({self.d_type_emb}) "
                f"+ d_cap({self.d_cap_emb}) = {expected_role}, but d_role={self.d_role}. "
                "Ch4.2.2 requires exact fill, no padding."
            )
        if self.d_belief != 2 * self.d_belief_proj:
            raise ValueError(
                f"d_belief mismatch: 2 × d_belief_proj({self.d_belief_proj}) "
                f"= {2 * self.d_belief_proj}, but d_belief={self.d_belief}. "
                "Ch4.2.4 requires d_belief = 2 × d_belief_proj."
            )
        if self.belief_pool not in _VALID_BELIEF_POOL:
            raise ValueError(
                f"Unknown belief_pool: {self.belief_pool!r} (valid: {_VALID_BELIEF_POOL})"
            )
        if self.hyper_gen_scope not in _VALID_GEN_SCOPE:
            raise ValueError(
                f"Unknown hyper_gen_scope: {self.hyper_gen_scope!r} "
                f"(valid: {_VALID_GEN_SCOPE})"
            )

        # [lora_fc2] film_head-exclusive: base_gen already generates fc2 fully, so layering a
        # rank-r Delta_W on top is mathematically redundant.
        if self.hyper_gen_scope == "base_gen":
            assert self.lora_fc2_rank in (None, 0), (
                "lora_fc2_rank must be None or 0 when gen_scope='base_gen' "
                "(base_gen already generates fc2 fully; Delta_W is redundant)."
            )

        # [lora_fc2] scale guardrail: under grouped RMS norm each generated element starts at
        # ~output_scale, so Delta_W = Bf @ Af ~ output_scale^2 * sqrt(r). At the default 0.01
        # that collapses to ~3e-4 (dead vs the kaiming fc2 base ~0.088); >= 0.05 (recommend
        # 0.1) keeps it expressive (~0.028 at scale 0.1, r=8).
        if self.lora_fc2_rank and self.lora_fc2_rank > 0:
            assert all(s >= 0.05 for s in (
                self.trans_output_scale_init,
                self.rew_output_scale_init,
                self.pred_output_scale_init,
            )), (
                "lora_fc2 requires output_scale_init >= 0.05 on all three hypernets "
                "(recommend 0.1) to keep Delta_W expressive."
            )

    @property
    def d_ctx_aug(self) -> int:
        """Total conditioning vector dim (Ch4.2.4): ``d_c + d_role + d_belief``."""
        return self.d_c + self.d_role + self.d_belief
