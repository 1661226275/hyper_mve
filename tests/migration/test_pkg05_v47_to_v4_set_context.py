"""Pkg-05 spec 08 §2.3 (C5-I1): v4.7 -> v4 set_context migration grep.

Cross-platform pathlib scan (not subprocess grep) of planning/ + training/:
  * no legacy single-arg ``set_context(`` (only ``set_context_objective(`` /
    ``set_context_subjective(`` are allowed),
  * no legacy ``set_context_from_history(`` / ``set_context_default(``.
"""
import re
from pathlib import Path

# tests/migration/ -> repo root is parents[2] / "hyper_mve"
_PKG_ROOT = Path(__file__).resolve().parents[2] / "hyper_mve"
_SCAN_DIRS = [_PKG_ROOT / "planning", _PKG_ROOT / "training"]

_LEGACY_SET_CONTEXT = re.compile(r"\bset_context\(")
_V4_SET_CONTEXT = re.compile(r"set_context_(objective|subjective)\(")
_LEGACY_BELIEF_API = re.compile(r"set_context_(from_history|default)\(")


def _py_files():
    for d in _SCAN_DIRS:
        for p in sorted(d.glob("*.py")):
            yield p


def test_no_legacy_set_context_calls():
    offenders = []
    for path in _py_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _LEGACY_SET_CONTEXT.search(line) and not _V4_SET_CONTEXT.search(line):
                offenders.append(f"{path.name}:{lineno}: {line.strip()}")
    assert not offenders, "Legacy set_context( calls remain:\n" + "\n".join(offenders[:10])


def test_no_legacy_belief_apis():
    offenders = []
    for path in _py_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _LEGACY_BELIEF_API.search(line):
                offenders.append(f"{path.name}:{lineno}: {line.strip()}")
    assert not offenders, "Legacy belief APIs remain:\n" + "\n".join(offenders[:10])


# ====== v5 (Pkg-09) API amendment locks ======

_V4_OBJECTIVE = re.compile(r"set_context_objective\(")


def test_no_v4_objective_context_calls():
    """v5 amendment: set_context_objective is deleted — no caller may remain
    in planning/ or training/ (transition is a plain shared module)."""
    offenders = []
    for path in _py_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _V4_OBJECTIVE.search(line):
                offenders.append(f"{path.name}:{lineno}: {line.strip()}")
    assert not offenders, "v4 set_context_objective calls remain:\n" + "\n".join(offenders[:10])


def test_v5_set_context_subjective_signature_lock():
    """Lock the v5 signature: (agent_id, row_i, belief). A drift here breaks
    the worker/planner/loss call sites silently — fail loudly instead."""
    import inspect
    from hyper_mve.models import HyperMuZeroModel
    from hyper_mve.baselines.internal.base import BaselineModel

    for cls in (HyperMuZeroModel, BaselineModel):
        params = list(inspect.signature(cls.set_context_subjective).parameters)
        assert params == ["self", "agent_id", "row_i", "belief"], (
            f"{cls.__name__}.set_context_subjective signature drifted: {params}"
        )
        assert not hasattr(cls, "set_context_objective"), (
            f"{cls.__name__} regrew set_context_objective (deleted in v5)"
        )
