"""C8-ABL-RENAME1 — use_coord_desc → randomize_order alias-with-deprecation (pkg-08 spec 06 §4)."""
from __future__ import annotations

import dataclasses
import warnings

import pytest

from hyper_mve.configs import V4Config


def test_use_coord_desc_alias_warns_read():
    """spec 06 §4.1 / §7.5 — READ via @property emits DeprecationWarning."""
    cfg = V4Config.from_preset("easy")
    with pytest.warns(DeprecationWarning, match="use_coord_desc is deprecated"):
        value = cfg.train.use_coord_desc
    assert value == cfg.train.randomize_order


def test_use_coord_desc_alias_warns_write():
    """spec 06 §4.2 / §7.5 — WRITE via dataclasses.replace emits DeprecationWarning."""
    cfg = V4Config.from_preset("easy")
    # Pick a non-default value so the reconciliation triggers (otherwise compat
    # equals randomize_order's default and the warning is suppressed).
    new_value = not cfg.train.randomize_order
    with pytest.warns(DeprecationWarning, match="use_coord_desc kwarg is deprecated"):
        new_train = dataclasses.replace(cfg.train, use_coord_desc=new_value)
    assert new_train.randomize_order is new_value


def test_use_coord_desc_compat_is_internal():
    """spec 06 §6 — _use_coord_desc_compat is leading-underscore (internal)."""
    cfg = V4Config.from_preset("easy")
    public_field_names = {
        f.name for f in dataclasses.fields(cfg.train) if not f.name.startswith("_")
    }
    assert "randomize_order" in public_field_names
    assert "_use_coord_desc_compat" not in public_field_names
    assert "use_coord_desc" not in public_field_names  # only the @property exposes it


def test_default_construction_does_not_warn():
    """spec 06 §4.4 — default V4Config construction emits no DeprecationWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        cfg = V4Config.from_preset("easy")
        assert cfg.train.randomize_order is True
