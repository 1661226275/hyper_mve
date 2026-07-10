"""Unit tests for ``hyper_mve.configs.legacy_config``."""
from __future__ import annotations

from hyper_mve.configs import LegacyConfig, V4Config


def test_default_disabled():
    lc = LegacyConfig()
    assert lc.freeze_enabled is False


def test_v4_deleted_fields_absent():
    """v4 Pass 2 removed these fields from LegacyConfig (architecture made them unnecessary)."""
    lc = LegacyConfig()
    for name in (
        "w_rew_diversity",
        "rew_diversity_target_cos",
        "rew_diversity_skip_pairs",
        "detach_pred_context",
    ):
        assert not hasattr(lc, name), f"LegacyConfig must not expose {name}"


def test_legacy_in_v4_config_default_disabled():
    cfg = V4Config.from_preset("rel_duo")
    assert cfg.legacy.freeze_enabled is False


def test_freeze_fields_present_and_typed():
    lc = LegacyConfig()
    assert lc.freeze_warmup_steps == 4000
    assert lc.freeze_phase_steps == 4000
    assert lc.freeze_hunter_agents == (0, 1, 2)
    assert lc.freeze_prey_agents == (3,)


def test_chunk_fields_present():
    lc = LegacyConfig()
    assert lc.chunk_alpha == 10
    assert lc.chunk_emb_size == 8
    assert lc.chunk_budget_factor == 1.0
    assert lc.chunk_hyperfan_init is True
