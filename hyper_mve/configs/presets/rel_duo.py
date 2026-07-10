"""rel_duo preset — v5 RelationCommons N=2 reference configuration (Pkg-09).

The research-point-1 workhorse: 2 agents on an 8×8 grid, 8 uniformly spawned
resource cells, T_max=100, regime family ``g2`` (|G|=5) with p=0 (regime
resampled per episode, static within). 200K-step budget.

Carried-over tuned knobs (env-agnostic duo learnings):
    - ``hyper_gen_scope="film_head"`` + ``rew/pred_output_scale_init=0.1``
      (grouped-RMS FiLM actually modulates; targets the cos collapse seen
      under FULL generation).
    - ``detach_pred_context=False`` (let policy/value gradient reach the
      role/belief encoders — the trunk is a stable shared SGD net).
    - ``mve_temperature=0.5`` (z-score aggregation entropy floor).
    - ``buffer_size=1500`` (target staleness).

Deliberately NOT carried over: duo's ``belief_grad_gating_steps=1e9`` —
under the v5 hidden regime the belief path is load-bearing (the posterior is
the only channel to the opponents' side of W), so the default 5000-step
gate applies.
"""
from __future__ import annotations

from dataclasses import replace

from ..env_config import EnvConfig
from ..eval_config import EvalConfig
from ..legacy_config import LegacyConfig
from ..model_config import ModelConfig
from ..mup_config import MupConfig
from ..train_config import TrainConfig
from ..v4_config import V4Config


def build_rel_duo_config() -> V4Config:
    """Construct the v5 RelationCommons N=2 V4Config."""
    env = EnvConfig(
        N=2,
        L=8,
        K=8,
        T_max=100,
        # v5 relationship regimes (Pkg-09)
        relation_family="g2",
        relation_intensity=1.0,
        regime_switch_prob=0.0,     # research point 1: per-episode static
        regime_kernel="uniform",
        alpha=0.10,                 # constant regen rate
        Q_max=10.0,
        epsilon_move=0.01,
    )

    model = ModelConfig(
        latent_dim=64,
        hidden_dim=128,
        d_role=32,
        d_belief=32,
        d_id_emb=8,
        d_row_emb=24,
        hyper_hidden_dims=(256, 256),
        hyper_rew_hidden_dims=(256, 256, 256),
        hyper_gen_scope="film_head",
        rew_output_scale_init=0.1,
        pred_output_scale_init=0.1,
        belief_gru_hidden=128,
        proj_dim=64,
    )

    train = TrainConfig(
        max_train_steps=200_000,
        batch_size=256,
        buffer_size=1500,
        min_buffer_size=500,
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
        w_belief_regime=1.0,
        w_belief_div=0.01,
        # Curriculum (oracle g -> anneal -> pure inference)
        curriculum_stage_1_end_frac=0.3,
        curriculum_stage_2_end_frac=0.7,
        belief_grad_gating_steps=5000,
        detach_pred_context=False,
        # EMA
        ema_tau=0.99,
        # Exploration
        epsilon_init=1.0,
        epsilon_min=0.05,
        epsilon_decay_steps=28_000,
        # MVE planner
        mve_samples=48,             # spa = 48/6 = 8 scenarios per candidate
        mve_depth=5,
        mve_temperature=0.5,
        use_crn=True,
        randomize_order=True,
        stratified_sampling=True,
        stratified_min_per_type_frac=0.3,   # per-regime min batch fraction (v5)
    )

    return V4Config(
        env=env,
        model=model,
        train=train,
        mup=MupConfig(),
        eval=EvalConfig(),
        legacy=LegacyConfig(),
        preset_name="rel_duo",
    )


def build_rel_duo_holdout_config() -> V4Config:
    """rel_duo restricted to the symmetric training regimes {coop, comp, neutral}.

    The zero-shot cell trains on regime ids (0, 1, 4) and evaluates on the
    held-out asymmetric regimes (2, 3) via ``reset(options={"g": ...})``.
    """
    base = build_rel_duo_config()
    env = replace(base.env, train_regime_ids=(0, 1, 4))
    return replace(base, env=env, preset_name="rel_duo_holdout")
