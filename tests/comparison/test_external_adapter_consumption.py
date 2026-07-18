"""C7-EXT-ADPT* — comparison runners consume the adapter, never instantiate
the env directly; the runtime ``_FORBIDDEN_INFO_KEYS`` guard is coded.

Grep-level lints over the phase-2 runner modules
(``hyper_mve/comparison/{mappo,mamba}.py``). New baseline adapters
(happo/mbom/m3w) extend ``_RUNNER_FILES`` in their landing phase.
"""
from __future__ import annotations

from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNNER_FILES = (
    _REPO_ROOT / "hyper_mve" / "comparison" / "mappo.py",
    _REPO_ROOT / "hyper_mve" / "comparison" / "mamba.py",
    _REPO_ROOT / "hyper_mve" / "comparison" / "happo.py",
    _REPO_ROOT / "hyper_mve" / "comparison" / "mbom.py",
    _REPO_ROOT / "hyper_mve" / "comparison" / "m3w_adapted" / "runner.py",
)
_BASE_PY = _REPO_ROOT / "hyper_mve" / "comparison" / "base.py"


@pytest.mark.parametrize("runner_file", _RUNNER_FILES, ids=lambda p: p.name)
def test_runner_does_not_construct_env_directly(runner_file):
    """spec 04 §10: runners MUST NEVER construct the env class directly —
    the ``env_fn`` factory is the only legal env construction site."""
    src = runner_file.read_text(encoding="utf-8")
    for pat in ("ResourceCommonsEnv(", "ResourceCommonsEnv (",
                "RelationCommonsEnv(", "RelationCommonsEnv ("):
        assert pat not in src, (
            f"{runner_file.name} contains {pat!r} — spec 04 §10 lint violation."
        )


@pytest.mark.parametrize("runner_file", _RUNNER_FILES, ids=lambda p: p.name)
def test_runner_imports_forbidden_info_keys(runner_file):
    """spec 05 §5.2 / spec 06 §6.2: every runner consumes the hoisted
    ``_FORBIDDEN_INFO_KEYS`` constant."""
    src = runner_file.read_text(encoding="utf-8")
    assert "_FORBIDDEN_INFO_KEYS" in src, (
        f"{runner_file.name} does not reference _FORBIDDEN_INFO_KEYS — "
        "spec 06 §6.2 runtime guard absent."
    )


def test_forbidden_info_keys_hoisted_to_comparison_base():
    """spec 06 §6.1 + Lock 3 (phase-2 relocation): the literal
    frozenset({...}) lives ONLY in ``hyper_mve/comparison/base.py``;
    the package ``__init__`` re-exports it and runners import it."""
    base_src = _BASE_PY.read_text(encoding="utf-8")
    assert "frozenset({" in base_src
    for k in ("g_true", "rows", "c_true", "types", "resource_state",
              "hotspot_centers"):
        assert k in base_src, f"base.py missing {k!r} in the frozenset."
    init_src = (_BASE_PY.parent / "__init__.py").read_text(encoding="utf-8")
    assert "_FORBIDDEN_INFO_KEYS" in init_src, (
        "comparison/__init__.py must re-export _FORBIDDEN_INFO_KEYS"
    )
    # single source of truth: no second literal definition in runner files
    for runner_file in _RUNNER_FILES:
        src = runner_file.read_text(encoding="utf-8")
        assert 'frozenset({\n    "g_true"' not in src, (
            f"{runner_file.name} redefines the forbidden-keys literal"
        )
