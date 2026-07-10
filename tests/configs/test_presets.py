"""Unit tests for the v5 RelationCommons presets (rel_duo / rel_duo_holdout)."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from hyper_mve.configs import V4Config


def test_rel_duo_preset_table():
    cfg = V4Config.from_preset("rel_duo")

    # Environment dimensions
    assert cfg.env.N == 2
    assert cfg.env.L == 8
    assert cfg.env.K == 8
    assert cfg.env.T_max == 100
    assert cfg.env.A == 6

    # Fixed physics
    assert cfg.env.Q_max == 10.0
    assert cfg.env.alpha == 0.10
    assert cfg.env.epsilon_move == 0.01

    # Relationship regimes (research point 1)
    assert cfg.env.relation_family == "g2"
    assert cfg.env.relation_intensity == 1.0
    assert cfg.env.regime_switch_prob == 0.0
    assert cfg.env.regime_kernel == "uniform"
    assert cfg.env.train_regime_ids is None
    assert cfg.env.regime_prior is None

    # Training budget + carried env-agnostic tuned knobs
    assert cfg.train.max_train_steps == 200_000
    assert cfg.model.hyper_gen_scope == "film_head"
    assert cfg.model.rew_output_scale_init == 0.1
    assert cfg.model.pred_output_scale_init == 0.1
    assert cfg.train.mve_temperature == 0.5

    # Belief is load-bearing in v5 (hidden-regime inference): the gate must
    # NOT be duo's never-open 1e9 — default 5000 warmup.
    assert cfg.train.belief_grad_gating_steps == 5000

    # ctx_aug = role(32) + belief(32)
    assert cfg.model.d_ctx_aug == 64

    assert cfg.preset_name == "rel_duo"


def test_rel_duo_holdout_differs_only_in_train_regime_ids():
    base = V4Config.from_preset("rel_duo")
    hold = V4Config.from_preset("rel_duo_holdout")

    # Symmetric regimes seen in training; asymmetric pair (2, 3) held out.
    assert hold.env.train_regime_ids == (0, 1, 4)
    assert hold.env == replace(base.env, train_regime_ids=(0, 1, 4))
    assert hold.model == base.model
    assert hold.train == base.train
    assert hold.preset_name == "rel_duo_holdout"


def test_base_gen_forbids_lora_fc2_rank():
    """ModelConfig.__post_init__ rejects lora_fc2_rank on base_gen (Delta_W redundant)."""
    from hyper_mve.configs.model_config import ModelConfig
    with pytest.raises(AssertionError, match="base_gen"):
        ModelConfig(hyper_gen_scope="base_gen", lora_fc2_rank=8)


def test_lora_fc2_requires_expressive_output_scale():
    """lora_fc2 with the default 0.01 pred scale fails the >= 0.05 guardrail."""
    from hyper_mve.configs.model_config import ModelConfig
    with pytest.raises(AssertionError, match="output_scale_init"):
        ModelConfig(hyper_gen_scope="lora_fc2", lora_fc2_rank=8)


def test_all_presets_construct_without_error():
    for name in ("rel_duo", "rel_duo_holdout"):
        V4Config.from_preset(name)


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        V4Config.from_preset("medium")   # v4 preset deleted in Stage 6


def test_preset_to_dict_serialisable():
    for name in ("rel_duo", "rel_duo_holdout"):
        cfg = V4Config.from_preset(name)
        s = json.dumps(cfg.to_dict())
        assert len(s) > 500


def test_replace_does_not_mutate_original():
    cfg = V4Config.from_preset("rel_duo")
    original_N = cfg.env.N
    _ = replace(cfg, env=replace(cfg.env, N=4, relation_family="g4"))
    assert cfg.env.N == original_N
