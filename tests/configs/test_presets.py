"""Unit tests for the three Ch3.9 difficulty presets."""
from __future__ import annotations

import json
from dataclasses import replace

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


def test_preset_name_field():
    assert V4Config.from_preset("easy").preset_name == "easy"
    assert V4Config.from_preset("medium").preset_name == "medium"
    assert V4Config.from_preset("hard").preset_name == "hard"


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
    for name in ("easy", "medium", "hard"):
        V4Config.from_preset(name)


def test_preset_to_dict_serialisable():
    for name in ("easy", "medium", "hard"):
        cfg = V4Config.from_preset(name)
        s = json.dumps(cfg.to_dict())
        assert len(s) > 500


def test_replace_does_not_mutate_original():
    cfg = V4Config.from_preset("medium")
    original_N = cfg.env.N
    _ = replace(cfg, env=replace(cfg.env, N=8, type_assignment=(AgentType.ALPHA,) * 8))
    assert cfg.env.N == original_N
