"""Unit tests for ``hyper_mve.configs.v4_config`` (v5 Pkg-09)."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from hyper_mve.configs import ModelConfig, TrainConfig, V4Config


def test_from_preset_rel_duo():
    cfg = V4Config.from_preset("rel_duo")
    assert cfg.env.N == 2
    assert cfg.env.K == 8
    assert cfg.env.T_max == 100
    assert cfg.preset_name == "rel_duo"


def test_from_preset_invalid():
    with pytest.raises(ValueError, match="Unknown preset"):
        V4Config.from_preset("ultra")
    with pytest.raises(ValueError, match="Unknown preset"):
        V4Config.from_preset("medium")   # v4 presets deleted in Stage 6


def test_replace_propagation():
    cfg = V4Config.from_preset("rel_duo")
    cfg_new = replace(cfg, train=replace(cfg.train, lr=3e-4))
    assert cfg.train.lr == 1e-4              # original unchanged
    assert cfg_new.train.lr == 3e-4
    assert cfg_new.env.N == 2                # other fields preserved


def test_to_dict_json_serializable():
    cfg = V4Config.from_preset("rel_duo")
    d = cfg.to_dict()
    s = json.dumps(d)
    assert len(s) > 500
    # v5: relation fields serialise plainly; no AgentType coercion needed.
    assert d["env"]["relation_family"] == "g2"
    assert "type_assignment" not in d["env"]


def test_train_config_curriculum_boundaries():
    with pytest.raises(ValueError, match="Curriculum"):
        TrainConfig(
            curriculum_stage_1_end_frac=0.5,
            curriculum_stage_2_end_frac=0.4,
        )


def test_train_config_invalid_lr_schedule():
    with pytest.raises(ValueError, match="lr_schedule"):
        TrainConfig(lr_schedule="adagrad")


def test_model_config_role_dim_exact_fill():
    """v5 (Pkg-09): role = id_emb (8) + row_emb (24) exact fill."""
    cfg = ModelConfig()
    assert cfg.d_role == 32
    assert cfg.d_id_emb + cfg.d_row_emb == cfg.d_role

    with pytest.raises(ValueError, match="d_role mismatch"):
        ModelConfig(d_role=32, d_id_emb=8, d_row_emb=16)


def test_model_config_total_ctx_aug():
    """v5 (Pkg-09): ctx_aug = role (32) + belief (32) = 64 (c path removed)."""
    cfg = ModelConfig()
    assert cfg.d_ctx_aug == 64


def test_model_config_v4_fields_gone():
    """Stage-6 lock: the deprecated v4 dims must not regrow."""
    from dataclasses import fields
    names = {f.name for f in fields(ModelConfig)}
    for gone in ("d_c", "d_type_emb", "d_cap_emb", "d_belief_proj",
                 "trans_output_scale_init"):
        assert gone not in names, f"ModelConfig regrew v4 field {gone!r}"


def test_v4_config_frozen():
    cfg = V4Config.from_preset("rel_duo")
    with pytest.raises(Exception):
        cfg.preset_name = "tampered"  # type: ignore[misc]
