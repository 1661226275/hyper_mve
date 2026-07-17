"""C7-EXT-ADPT* — external runners consume the adapter, never instantiate
the env directly; the runtime ``_FORBIDDEN_INFO_KEYS`` guard is coded.

pkg-07 spec 04 §10 + spec 05 §12.2 + spec 06 §6.3 grep-level lints,
parametrised over the three Tier-1 runner modules.
"""
from __future__ import annotations

from pathlib import Path

import pytest


_RUNNER_MODULES = ("mappo.py", "qmix.py", "ma_muzero_gh.py", "mamba.py")
_RUNNER_DIR = Path(__file__).resolve().parents[3] \
    / "hyper_mve" / "baselines" / "external"


@pytest.mark.parametrize("runner_module", _RUNNER_MODULES)
def test_runner_does_not_construct_resource_commons_env(runner_module):
    """spec 04 §10: external runners MUST NEVER call
    ``ResourceCommonsEnv(...)`` directly — the ``env_fn`` factory is the
    only legal env construction site."""
    src = (_RUNNER_DIR / runner_module).read_text(encoding="utf-8")
    bad_patterns = ["ResourceCommonsEnv(", "ResourceCommonsEnv ("]
    for pat in bad_patterns:
        assert pat not in src, (
            f"{runner_module} contains {pat!r} — spec 04 §10 lint violation."
        )


@pytest.mark.parametrize("runner_module", _RUNNER_MODULES)
def test_runner_imports_forbidden_info_keys(runner_module):
    """spec 05 §5.2 / spec 06 §6.2: every Tier-1 runner consumes the hoisted
    ``_FORBIDDEN_INFO_KEYS`` constant."""
    src = (_RUNNER_DIR / runner_module).read_text(encoding="utf-8")
    assert "_FORBIDDEN_INFO_KEYS" in src, (
        f"{runner_module} does not reference _FORBIDDEN_INFO_KEYS — spec "
        "06 §6.2 runtime guard absent."
    )


def test_forbidden_info_keys_hoisted_to_external_init():
    """spec 06 §6.1 + Lock 3: the literal frozenset({...}) must live ONLY
    in ``hyper_mve/baselines/external/__init__.py`` — runners import it."""
    init_src = (_RUNNER_DIR / "__init__.py").read_text(encoding="utf-8")
    assert "frozenset({" in init_src
    for k in ("c_true", "types", "resource_state", "hotspot_centers"):
        assert k in init_src, f"__init__.py missing {k!r} in the frozenset."
