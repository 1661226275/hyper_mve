"""C7-EXT-STUB1 — stubs raise ``NotImplementedError`` on ``__init__``.

pkg-07 spec 06 §5.4 — parametrised over MARIE / GA (always stubs) +
external_mamba (when ``IS_SOURCED is False``).
"""
from __future__ import annotations

import pytest

from hyper_mve.baselines import create_baseline
from hyper_mve.baselines.external.mamba import IS_SOURCED
from hyper_mve.configs import V4Config


_STUB_VARIANTS = ("external_marie", "external_ga") + \
    (("external_mamba",) if not IS_SOURCED else ())


@pytest.mark.parametrize("variant", _STUB_VARIANTS)
def test_stub_baselines_raise_notimplementederror_at_init(variant):
    """The factory itself does NOT raise; the stub class's __init__ does."""
    cfg = V4Config.from_preset("easy")
    with pytest.raises(NotImplementedError):
        create_baseline(cfg, variant)


def test_mamba_class_binding_pattern_present():
    """spec 06 §4.6 anchor — class binding at import time:
    ``MAMBAAlgorithm = _RealMAMBA if IS_SOURCED else _MAMBAStub``."""
    from pathlib import Path
    src = (
        Path(__file__).resolve().parents[3]
        / "hyper_mve" / "baselines" / "external" / "mamba.py"
    ).read_text(encoding="utf-8")
    assert "_MAMBAStub" in src
    assert "_RealMAMBA" in src
    assert "IS_SOURCED" in src
    # The conditional binding line itself.
    assert "MAMBAAlgorithm = _RealMAMBA if IS_SOURCED else _MAMBAStub" in src, (
        "spec 06 §4.6 byte-identical anchor missing from mamba.py"
    )
