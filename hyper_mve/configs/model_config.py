"""ModelConfig — network dimensions and hypernet configuration (Ch4.2-4.6, v4 Pass 2)."""
from __future__ import annotations

from dataclasses import dataclass

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
_VALID_GEN_SCOPE: tuple[str, ...] = ("full", "film_head")


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
    # output head (partial generation, grouped-RMS-normed). See duo preset.
    hyper_gen_scope: str = "full"

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

    @property
    def d_ctx_aug(self) -> int:
        """Total conditioning vector dim (Ch4.2.4): ``d_c + d_role + d_belief``."""
        return self.d_c + self.d_role + self.d_belief
