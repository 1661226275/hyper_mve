"""C7-EXT-FACT1 — factory dispatches all 6 external CLI strings.

pkg-07 spec 05 §12.1 + spec 06 §10.1 / §10.2 — assert
``isinstance(create_baseline(cfg, ...), <expected runner class>)`` for
each Tier-1 runner; assert ``ExternalBaselineRunner`` protocol surface
present (the four-method ``train`` / ``evaluate`` / ``save_checkpoint`` /
``load_checkpoint``); assert NOT a ``BaselineModel`` (no internal 7-API).

Stubs (``external_mamba`` when ``IS_SOURCED is False``, ``external_marie``,
``external_ga``) are tested in :mod:`test_stub_external_baselines`.
"""
from __future__ import annotations

import pytest

from hyper_mve.baselines import create_baseline
from hyper_mve.baselines.external import ExternalBaselineRunner
from hyper_mve.baselines.external.ma_muzero_gh import MAMuZeroGHAlgorithm
from hyper_mve.baselines.external.mappo import MAPPOAlgorithm
from hyper_mve.baselines.external.qmix import QMIXAlgorithm
from hyper_mve.configs import V4Config


_TIER1_VARIANTS = [
    ("external_mappo", MAPPOAlgorithm),
    ("external_qmix", QMIXAlgorithm),
    ("external_ma_muzero_gh", MAMuZeroGHAlgorithm),
]


@pytest.mark.parametrize("variant,expected_cls", _TIER1_VARIANTS)
def test_factory_dispatches_tier1(variant, expected_cls):
    cfg = V4Config.from_preset("easy")
    runner = create_baseline(cfg, variant)
    assert isinstance(runner, expected_cls), (
        f"create_baseline({variant!r}) returned {type(runner).__name__}, "
        f"expected {expected_cls.__name__}"
    )
    assert isinstance(runner, ExternalBaselineRunner)
    # Four-method protocol surface (pkg-07 spec 05 §4 / spec 06 §2.3).
    for attr in ("train", "evaluate", "save_checkpoint", "load_checkpoint"):
        assert callable(getattr(runner, attr, None)), (
            f"{type(runner).__name__} missing {attr!r}"
        )
    # NOT a BaselineModel — no internal 7-API.
    assert not hasattr(runner, "set_context_subjective"), (
        f"{type(runner).__name__} accidentally exposes set_context_subjective"
    )


@pytest.mark.parametrize("variant,_cls", _TIER1_VARIANTS)
def test_tier1_train_signature_has_keyword_only_args(variant, _cls):
    """pkg-07 spec 05 §4.2 / spec 06 §2.3: ``train`` must have
    ``total_env_steps`` / ``lr`` / ``seed`` as keyword-only params."""
    import inspect
    cfg = V4Config.from_preset("easy")
    runner = create_baseline(cfg, variant)
    sig = inspect.signature(runner.train)
    for name in ("total_env_steps", "lr", "seed"):
        assert name in sig.parameters, f"{variant}.train missing {name!r}"
        assert sig.parameters[name].kind == inspect.Parameter.KEYWORD_ONLY, (
            f"{variant}.train: {name!r} is not keyword-only"
        )


def test_param_count_method_present_on_tier1():
    """spec 06 §8.4 disclosure feed — every Tier-1 runner exposes ``param_count``."""
    cfg = V4Config.from_preset("easy")
    for variant, _ in _TIER1_VARIANTS:
        runner = create_baseline(cfg, variant)
        assert hasattr(runner, "param_count")
        assert callable(runner.param_count)
        # Value pre-build is 0; post-build it's positive — but we don't
        # build here (would require torch + an env). 0 is fine.
        assert runner.param_count() == 0
