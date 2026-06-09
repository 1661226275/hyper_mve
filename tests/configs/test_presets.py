"""Unit tests for the three Ch3.9 difficulty presets."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from hyper_mve.configs import V4Config
from hyper_mve.schemas import AgentType


def test_medium_preset_ch_3_9_table():
    cfg = V4Config.from_preset("medium")

    # Environment dimensions
    assert cfg.env.N == 4
    assert cfg.env.L == 16
    assert cfg.env.K == 20
    assert cfg.env.M == 3
    assert cfg.env.T_max == 200
    assert cfg.env.c_mode == "static"

    # 2α+2β
    assert cfg.env.type_assignment == (
        AgentType.ALPHA, AgentType.ALPHA,
        AgentType.BETA, AgentType.BETA,
    )

    # Resource dynamics
    assert cfg.env.Q_max == 10.0
    assert cfg.env.alpha_min == 0.02
    assert cfg.env.alpha_max == 0.20
    assert cfg.env.kappa_f == 6.0
    assert cfg.env.theta_f == 0.3

    # Fehr-Schmidt
    assert cfg.env.kappa == 0.5
    assert cfg.env.lambda_disadv == 2.0
    assert cfg.env.lambda_adv == 0.6


def test_easy_preset_ch_3_9_table():
    cfg = V4Config.from_preset("easy")
    assert cfg.env.N == 2
    assert cfg.env.L == 8
    assert cfg.env.K == 8
    assert cfg.env.M == 1
    assert cfg.env.T_max == 100
    assert cfg.env.type_assignment == (AgentType.ALPHA, AgentType.BETA)
    assert cfg.train.max_train_steps == 200_000


def test_hard_preset_ch_3_9_table():
    cfg = V4Config.from_preset("hard")
    assert cfg.env.N == 8
    assert cfg.env.L == 24
    assert cfg.env.K == 40
    assert cfg.env.M == 5
    assert cfg.env.T_max == 300
    assert len(cfg.env.type_assignment) == 8
    assert sum(1 for t in cfg.env.type_assignment if t == AgentType.ALPHA) == 4
    assert sum(1 for t in cfg.env.type_assignment if t == AgentType.BETA) == 4
    assert cfg.env.c_mode == "oscillate"
    assert cfg.train.max_train_steps == 2_000_000


def test_duo_preset_two_agent_drift():
    cfg = V4Config.from_preset("duo")
    # 2-agent (1α+1β) drifting context
    assert cfg.env.N == 2
    assert cfg.env.type_assignment == (AgentType.ALPHA, AgentType.BETA)
    assert cfg.env.c_mode == "random_walk"
    assert cfg.preset_name == "duo"

    # Medium-scale env retained: only N + context differ from medium (Step 2 rationale).
    medium = V4Config.from_preset("medium")
    assert cfg.env.L == medium.env.L
    assert cfg.env.K == medium.env.K
    assert cfg.env.M == medium.env.M
    assert cfg.env.T_max == medium.env.T_max

    # film_head partial-generation + companion knobs (refactor Step 5).
    assert cfg.model.hyper_gen_scope == "film_head"
    assert cfg.model.trans_output_scale_init == 0.1
    assert cfg.model.pred_output_scale_init == 0.1
    assert cfg.model.rew_output_scale_init == 0.1  # medium default, unchanged
    assert cfg.train.detach_pred_context is False
    # ONLY those fields differ from medium — verify nothing else drifted.
    assert cfg.model == replace(
        medium.model, hyper_gen_scope="film_head",
        trans_output_scale_init=0.1, pred_output_scale_init=0.1,
    )
    assert cfg.train == replace(medium.train, detach_pred_context=False)


def test_duo_basegen_preset():
    cfg = V4Config.from_preset("duo_basegen")
    # same duo env (2-agent 1α+1β drift)
    assert cfg.env.N == 2
    assert cfg.env.type_assignment == (AgentType.ALPHA, AgentType.BETA)
    assert cfg.env.c_mode == "random_walk"
    assert cfg.preset_name == "duo_basegen"
    assert cfg.model.d_ctx_aug == 80

    # base_gen + shared subjective trunk + companion knobs (Idea 1 + Idea 2).
    assert cfg.model.hyper_gen_scope == "base_gen"
    assert cfg.model.share_subjective_trunk is True
    assert cfg.model.trans_output_scale_init == 0.1
    assert cfg.model.rew_output_scale_init == 0.1
    assert cfg.model.pred_output_scale_init == 0.1
    assert cfg.train.detach_pred_context is False

    # ONLY those fields differ from medium — verify nothing else drifted.
    medium = V4Config.from_preset("medium")
    assert cfg.model == replace(
        medium.model, hyper_gen_scope="base_gen", share_subjective_trunk=True,
        trans_output_scale_init=0.1, pred_output_scale_init=0.1,
    )
    assert cfg.train == replace(medium.train, detach_pred_context=False)


def test_duo_film_lora_preset():
    cfg = V4Config.from_preset("duo_film_lora")
    assert cfg.env.N == 2
    assert cfg.env.type_assignment == (AgentType.ALPHA, AgentType.BETA)
    assert cfg.env.c_mode == "random_walk"
    assert cfg.preset_name == "duo_film_lora"
    # film_head + 3-way output-layer LoRA(r=32), no lora_fc2, separate hypernets.
    assert cfg.model.hyper_gen_scope == "film_head"
    assert cfg.model.hyper_output_rank == 32
    assert cfg.model.lora_fc2_rank is None
    assert cfg.model.share_subjective_trunk is False
    assert cfg.model.trans_output_scale_init == 0.1
    assert cfg.model.rew_output_scale_init == 0.1
    assert cfg.model.pred_output_scale_init == 0.1
    assert cfg.train.detach_pred_context is False
    medium = V4Config.from_preset("medium")
    assert cfg.model == replace(
        medium.model, hyper_gen_scope="film_head", hyper_output_rank=32,
        trans_output_scale_init=0.1, pred_output_scale_init=0.1,
    )
    assert cfg.train == replace(medium.train, detach_pred_context=False)


def test_duo_film_lora_fc2_preset():
    cfg = V4Config.from_preset("duo_film_lora_fc2")
    assert cfg.env.N == 2
    assert cfg.preset_name == "duo_film_lora_fc2"
    # lora_fc2 (film_head + per-context rank-8 fc2 delta) + 3-way output-layer LoRA(r=32).
    assert cfg.model.hyper_gen_scope == "lora_fc2"
    assert cfg.model.lora_fc2_rank == 8
    assert cfg.model.hyper_output_rank == 32
    assert cfg.model.share_subjective_trunk is False
    assert cfg.model.trans_output_scale_init == 0.1
    assert cfg.model.pred_output_scale_init == 0.1
    assert cfg.train.detach_pred_context is False
    medium = V4Config.from_preset("medium")
    assert cfg.model == replace(
        medium.model, hyper_gen_scope="lora_fc2", lora_fc2_rank=8, hyper_output_rank=32,
        trans_output_scale_init=0.1, pred_output_scale_init=0.1,
    )


def test_duo_base_lora_preset():
    cfg = V4Config.from_preset("duo_base_lora")
    assert cfg.env.N == 2
    assert cfg.preset_name == "duo_base_lora"
    # base_gen + 3-way output-layer LoRA(r=32); lora_fc2 forbidden -> stays None.
    assert cfg.model.hyper_gen_scope == "base_gen"
    assert cfg.model.hyper_output_rank == 32
    assert cfg.model.lora_fc2_rank is None
    assert cfg.model.share_subjective_trunk is False
    medium = V4Config.from_preset("medium")
    assert cfg.model == replace(
        medium.model, hyper_gen_scope="base_gen", hyper_output_rank=32,
        trans_output_scale_init=0.1, pred_output_scale_init=0.1,
    )


def test_medium_film_lora_preset():
    cfg = V4Config.from_preset("medium_film_lora")
    medium = V4Config.from_preset("medium")
    assert cfg.env == medium.env          # N=4 (2a+2b, static) env untouched
    assert cfg.preset_name == "medium_film_lora"
    assert cfg.model.hyper_gen_scope == "film_head"
    assert cfg.model.hyper_output_rank == 32
    assert cfg.model.lora_fc2_rank is None
    assert cfg.model.share_subjective_trunk is False
    assert cfg.train.detach_pred_context is False
    assert cfg.model == replace(
        medium.model, hyper_gen_scope="film_head", hyper_output_rank=32,
        trans_output_scale_init=0.1, pred_output_scale_init=0.1,
    )
    assert cfg.train == replace(medium.train, detach_pred_context=False)


def test_medium_film_lora_fc2_preset():
    cfg = V4Config.from_preset("medium_film_lora_fc2")
    medium = V4Config.from_preset("medium")
    assert cfg.env == medium.env
    assert cfg.preset_name == "medium_film_lora_fc2"
    assert cfg.model.hyper_gen_scope == "lora_fc2"
    assert cfg.model.lora_fc2_rank == 8
    assert cfg.model.hyper_output_rank == 32
    assert cfg.model.share_subjective_trunk is False
    assert cfg.model == replace(
        medium.model, hyper_gen_scope="lora_fc2", lora_fc2_rank=8, hyper_output_rank=32,
        trans_output_scale_init=0.1, pred_output_scale_init=0.1,
    )


def test_medium_base_lora_preset():
    cfg = V4Config.from_preset("medium_base_lora")
    medium = V4Config.from_preset("medium")
    assert cfg.env == medium.env
    assert cfg.preset_name == "medium_base_lora"
    assert cfg.model.hyper_gen_scope == "base_gen"
    assert cfg.model.hyper_output_rank == 32
    assert cfg.model.lora_fc2_rank is None
    assert cfg.model.share_subjective_trunk is False
    assert cfg.model == replace(
        medium.model, hyper_gen_scope="base_gen", hyper_output_rank=32,
        trans_output_scale_init=0.1, pred_output_scale_init=0.1,
    )


def test_base_gen_forbids_lora_fc2_rank():
    """ModelConfig.__post_init__ rejects lora_fc2_rank on base_gen (Delta_W redundant)."""
    from hyper_mve.configs.model_config import ModelConfig
    with pytest.raises(AssertionError, match="base_gen"):
        ModelConfig(hyper_gen_scope="base_gen", lora_fc2_rank=8)


def test_lora_fc2_requires_expressive_output_scale():
    """lora_fc2 with the default 0.01 trans/pred scales fails the >= 0.05 guardrail."""
    from hyper_mve.configs.model_config import ModelConfig
    with pytest.raises(AssertionError, match="output_scale_init"):
        ModelConfig(hyper_gen_scope="lora_fc2", lora_fc2_rank=8)


def test_preset_name_field():
    assert V4Config.from_preset("easy").preset_name == "easy"
    assert V4Config.from_preset("medium").preset_name == "medium"
    assert V4Config.from_preset("hard").preset_name == "hard"
    assert V4Config.from_preset("duo").preset_name == "duo"
    assert V4Config.from_preset("duo_basegen").preset_name == "duo_basegen"
    assert V4Config.from_preset("duo_film_lora").preset_name == "duo_film_lora"
    assert V4Config.from_preset("duo_film_lora_fc2").preset_name == "duo_film_lora_fc2"
    assert V4Config.from_preset("duo_base_lora").preset_name == "duo_base_lora"
    assert V4Config.from_preset("medium_film_lora").preset_name == "medium_film_lora"
    assert V4Config.from_preset("medium_film_lora_fc2").preset_name == "medium_film_lora_fc2"
    assert V4Config.from_preset("medium_base_lora").preset_name == "medium_base_lora"


def test_easy_inherits_from_medium():
    medium = V4Config.from_preset("medium")
    easy = V4Config.from_preset("easy")

    # Same model
    assert easy.model == medium.model

    # Same train except max_train_steps
    assert easy.train.lr == medium.train.lr
    assert easy.train.batch_size == medium.train.batch_size
    assert easy.train.curriculum_stage_1_end_frac == 0.3
    assert easy.train.max_train_steps != medium.train.max_train_steps


def test_hard_inherits_from_medium_except_env():
    medium = V4Config.from_preset("medium")
    hard = V4Config.from_preset("hard")

    assert hard.model == medium.model
    # Hard only overrides max_train_steps in train
    assert hard.train.lr == medium.train.lr
    assert hard.train.batch_size == medium.train.batch_size


def test_all_presets_construct_without_error():
    for name in ("easy", "medium", "hard", "duo", "duo_basegen",
                 "duo_film_lora", "duo_film_lora_fc2", "duo_base_lora",
                 "medium_film_lora", "medium_film_lora_fc2", "medium_base_lora"):
        V4Config.from_preset(name)


def test_preset_to_dict_serialisable():
    for name in ("easy", "medium", "hard", "duo", "duo_basegen",
                 "duo_film_lora", "duo_film_lora_fc2", "duo_base_lora",
                 "medium_film_lora", "medium_film_lora_fc2", "medium_base_lora"):
        cfg = V4Config.from_preset(name)
        s = json.dumps(cfg.to_dict())
        assert len(s) > 500


def test_replace_does_not_mutate_original():
    cfg = V4Config.from_preset("medium")
    original_N = cfg.env.N
    _ = replace(cfg, env=replace(cfg.env, N=8, type_assignment=(AgentType.ALPHA,) * 8))
    assert cfg.env.N == original_N
