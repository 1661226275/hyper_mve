"""Medium preset (Ch3.9 main reference configuration).

N=4, L=16, K=20, T_max=200, 2α+2β. Corresponds to the primary comparison
tables and figures in Chapter 6. All other presets derive from this one.
"""
from __future__ import annotations

from hyper_mve.schemas import AgentType

from ..env_config import EnvConfig
from ..eval_config import EvalConfig
from ..legacy_config import LegacyConfig
from ..model_config import ModelConfig
from ..mup_config import MupConfig
from ..train_config import TrainConfig
from ..v4_config import V4Config


def build_medium_config() -> V4Config:
    """Construct the Medium-difficulty V4Config (Ch3.9 main comparison)."""
    env = EnvConfig(
        N=4,
        L=16,
        K=20,
        M=3,
        T_max=200,
        type_assignment=(
            AgentType.ALPHA, AgentType.ALPHA,
            AgentType.BETA, AgentType.BETA,
        ),
        c_mode="static",
        # Resource dynamics (Ch3.3)
        Q_max=10.0,
        alpha_min=0.02,
        alpha_max=0.20,
        kappa_f=6.0,
        theta_f=0.3,
        d_nbr=3,
        sigma_patch=2.0,
        # Type mechanism (Ch3.5)
        kappa=0.5,
        lambda_disadv=2.0,
        lambda_adv=0.6,
        epsilon_move=0.01,
    )

    model = ModelConfig(
        latent_dim=64,
        hidden_dim=128,
        d_c=16,
        d_role=32,
        d_belief=32,            # v4: 2 × d_belief_proj
        d_belief_proj=16,
        d_id_emb=8,
        d_type_emb=8,           # v4: 8 (not 4)
        d_cap_emb=16,
        hyper_hidden_dims=(256, 256),
        hyper_rew_hidden_dims=(256, 256, 256),
        trans_output_scale_init=0.01,
        rew_output_scale_init=0.1,
        pred_output_scale_init=0.01,
        belief_gru_hidden=128,
        belief_pool="mean",
        proj_dim=64,
    )

    train = TrainConfig(
        max_train_steps=1_000_000,
        batch_size=256,
        buffer_size=5000,
        min_buffer_size=1000,
        episodes_per_iter=8,
        train_steps_per_iter=8,
        unroll_K=5,
        n_step=5,
        gamma=0.95,
        lr=1e-4,
        lr_min=5e-6,
        lr_warmup_steps=5000,
        # Loss weights
        w_policy=1.0,
        w_value=0.25,
        w_reward=3.0,
        w_consist=0.5,
        w_belief=1.0,
        w_belief_c=1.0,
        w_belief_opp=0.5,
        w_belief_div=0.01,
        # Curriculum
        curriculum_stage_1_end_frac=0.3,
        curriculum_stage_2_end_frac=0.7,
        belief_grad_gating_steps=5000,
        # EMA
        ema_tau=0.99,
        # Exploration
        epsilon_init=1.0,
        epsilon_min=0.05,
        epsilon_decay_steps=28_000,
        # MVE planner
        mve_samples=50,
        mve_depth=5,
        mve_temperature=1.0,
        # Defaults: all on (DPower)
        use_crn=True,
        use_coord_desc=True,
        stratified_sampling=True,
        stratified_min_per_type_frac=0.3,
    )

    return V4Config(
        env=env,
        model=model,
        train=train,
        mup=MupConfig(),
        eval=EvalConfig(),
        legacy=LegacyConfig(),
        preset_name="medium",
    )
