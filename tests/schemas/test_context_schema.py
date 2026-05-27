"""Unit tests for ``hyper_mve.schemas.context``."""
from __future__ import annotations

from dataclasses import replace

import pytest

from hyper_mve.schemas import ContextSchema


def test_default_dims_match_ch_4_2():
    cs = ContextSchema()
    assert cs.d_c == 16
    assert cs.d_role == 32
    assert cs.d_belief == 32
    assert cs.d_id_emb == 8
    assert cs.d_type_emb == 8
    assert cs.d_cap_emb == 16
    assert cs.d_belief_proj == 16


def test_d_ctx_aug_sums():
    cs = ContextSchema()
    assert cs.d_ctx_aug == 16 + 32 + 32
    assert cs.d_ctx_aug == 80


def test_role_exact_fill_invariant():
    cs = ContextSchema()
    assert cs.d_id_emb + cs.d_type_emb + cs.d_cap_emb == cs.d_role


def test_role_mismatch_rejected():
    with pytest.raises(ValueError, match="d_id"):
        replace(ContextSchema(), d_type_emb=4)  # 8+4+16=28 != 32


def test_belief_mismatch_rejected():
    with pytest.raises(ValueError, match="d_belief"):
        replace(ContextSchema(), d_belief=48)   # 48 != 2*16


def test_total_drift_rejected():
    """If the sum drifts away from the global D_CTX_AUG constant, fail loud."""
    with pytest.raises(ValueError):
        replace(ContextSchema(), d_c=24)        # 24+32+32 = 88 != 80
