"""Registry contract — lazy 3-key registry dispatches every runner (phase-2).

Asserts ``isinstance(create_runner(cfg, ...), <expected class>)`` for each
registered key; the ``ExternalBaselineRunner`` four-method protocol surface
(``train`` / ``evaluate`` / ``save_checkpoint`` / ``load_checkpoint``); and
that no runner exposes the retired internal 7-API. HAPPO / MBOM / M3W keys
are appended by realignment phases 4–6 and extend ``_VARIANTS`` here.
"""
from __future__ import annotations

import pytest

from hyper_mve.comparison import REGISTRY, create_runner
from hyper_mve.comparison import ExternalBaselineRunner
from hyper_mve.utils.configs import V4Config


_VARIANTS = [
    ("mazero_mixed", "hyper_mve.algo.runner", "MAZeroMixedRunner"),
    ("mappo", "hyper_mve.comparison.mappo", "MAPPOAlgorithm"),
    ("mamba", "hyper_mve.comparison.mamba", "MAMBAAlgorithm"),
]


def test_registry_is_lazy_string_map():
    """Values are ``module:Class`` strings — listing never imports torch."""
    assert sorted(REGISTRY) == sorted(v for v, _, _ in _VARIANTS)
    for key, module_name, class_name in _VARIANTS:
        assert REGISTRY[key] == f"{module_name}:{class_name}"


@pytest.mark.parametrize("variant,module_name,class_name", _VARIANTS)
def test_factory_dispatches_registered_variants(variant, module_name, class_name):
    import importlib

    cfg = V4Config.from_preset("rel_duo")
    runner = create_runner(cfg, variant)
    expected_cls = getattr(importlib.import_module(module_name), class_name)
    assert isinstance(runner, expected_cls), (
        f"create_runner({variant!r}) returned {type(runner).__name__}, "
        f"expected {class_name}"
    )
    assert isinstance(runner, ExternalBaselineRunner)
    # Four-method protocol surface (pkg-07 spec 05 §4 / spec 06 §2.3).
    for attr in ("train", "evaluate", "save_checkpoint", "load_checkpoint"):
        assert callable(getattr(runner, attr, None)), (
            f"{type(runner).__name__} missing {attr!r}"
        )
    # NOT the retired internal 7-API.
    assert not hasattr(runner, "set_context_subjective"), (
        f"{type(runner).__name__} accidentally exposes set_context_subjective"
    )


def test_factory_rejects_unknown_and_retired_names():
    cfg = V4Config.from_preset("rel_duo")
    for bad in ("external_mappo", "external_qmix", "no_belief", "hyper", "nope"):
        with pytest.raises(ValueError):
            create_runner(cfg, bad)


@pytest.mark.parametrize("variant,_m,_c", _VARIANTS)
def test_train_signature_has_keyword_only_args(variant, _m, _c):
    """``train`` must have ``total_env_steps`` / ``lr`` / ``seed`` keyword-only."""
    import inspect

    cfg = V4Config.from_preset("rel_duo")
    runner = create_runner(cfg, variant)
    sig = inspect.signature(runner.train)
    for name in ("total_env_steps", "lr", "seed"):
        assert name in sig.parameters, f"{variant}.train missing {name!r}"
        assert sig.parameters[name].kind == inspect.Parameter.KEYWORD_ONLY, (
            f"{variant}.train: {name!r} is not keyword-only"
        )


def test_param_count_method_present():
    """Disclosure feed — every runner exposes ``param_count``."""
    cfg = V4Config.from_preset("rel_duo")
    for variant, _, _ in _VARIANTS:
        runner = create_runner(cfg, variant)
        assert callable(getattr(runner, "param_count", None))
        if variant == "mazero_mixed":
            # lazily builds the network to count — must be positive
            assert runner.param_count() > 0
        else:
            # pre-build (no train()) the port-based adapters report 0
            assert runner.param_count() == 0
