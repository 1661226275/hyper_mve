"""Tests covering the v4.7 archive import behaviour (DeprecationWarning + sys.path)."""
from __future__ import annotations

import importlib
import sys
import warnings


def _purge_legacy_modules() -> None:
    """Remove ``hyper_mve._legacy_v4_7`` and submodules from ``sys.modules``.

    Required because ``warnings.warn`` only fires once per source location per
    interpreter — re-importing without purging will not re-trigger the warning.
    """
    to_drop = [name for name in sys.modules if name.startswith("hyper_mve._legacy_v4_7")]
    for name in to_drop:
        del sys.modules[name]


def test_import_archive_triggers_deprecation():
    _purge_legacy_modules()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module("hyper_mve._legacy_v4_7")

        deprecations = [w for w in caught
                        if issubclass(w.category, DeprecationWarning)]
        assert deprecations, "Expected DeprecationWarning from _legacy_v4_7 import"
        assert "deprecated" in str(deprecations[0].message).lower()


def test_archive_init_adds_to_sys_path():
    _purge_legacy_modules()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = importlib.import_module("hyper_mve._legacy_v4_7")
    archive_root = list(legacy.__path__)[0]
    assert archive_root in sys.path


def test_archived_config_module_importable():
    """The v4.7 BaseConfig class stays accessible via the archive."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        mod = importlib.import_module("hyper_mve._legacy_v4_7.config")
    assert hasattr(mod, "BaseConfig")
    assert mod.BaseConfig.num_agents == 4
