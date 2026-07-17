"""Stub runners raise ``NotImplementedError`` on ``__init__``.

Phase-2: MARIE / GA stubs are kept as classes (documented never-sourced
baselines) but are NOT registry keys — instantiate the classes directly.
``MAMBAAlgorithm`` binds to the real port when ``IS_SOURCED`` (spec 06 §4.6);
the stub branch is exercised only if sourcing is ever rolled back.
"""
from __future__ import annotations

import pytest

from hyper_mve.comparison.mamba import IS_SOURCED, MAMBAAlgorithm
from hyper_mve.comparison.stubs import GAStub, MARIEStub
from hyper_mve.utils.configs import V4Config


_STUB_CLASSES = [MARIEStub, GAStub] + ([MAMBAAlgorithm] if not IS_SOURCED else [])


@pytest.mark.parametrize("stub_cls", _STUB_CLASSES)
def test_stub_baselines_raise_notimplementederror_at_init(stub_cls):
    cfg = V4Config.from_preset("rel_duo")
    with pytest.raises(NotImplementedError):
        stub_cls(cfg)


def test_mamba_is_sourced_and_class_binding_pattern_present():
    """spec 06 §4.6 anchor — class binding at import time:
    ``MAMBAAlgorithm = _RealMAMBA if IS_SOURCED else _MAMBAStub``."""
    from pathlib import Path

    assert IS_SOURCED is True
    src = (
        Path(__file__).resolve().parents[2]
        / "hyper_mve" / "comparison" / "mamba.py"
    ).read_text(encoding="utf-8")
    assert "_MAMBAStub" in src
    assert "_RealMAMBA" in src
    assert "MAZeroMixedRunner" not in src  # separation: the method is not here
    assert "MAMBAAlgorithm = _RealMAMBA if IS_SOURCED else _MAMBAStub" in src, (
        "spec 06 §4.6 byte-identical anchor missing from mamba.py"
    )
