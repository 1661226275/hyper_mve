"""External baseline runners (pkg-07 spec 05 + spec 06).

Each runner implements :class:`ExternalBaselineRunner` (the abstract base
class declared in :mod:`hyper_mve.baselines.external.base`) and consumes
``RelationCommonsPettingZooEnv`` via an ``env_fn`` factory.

Implementation status:
  * ``MAPPOAlgorithm`` — Tier-1 real port from ``D:\\RL\\lzj\\MAPPO``
    (vendored under ``_lzj_mappo/``).
  * ``QMIXAlgorithm`` — Tier-1 clean-room port (Rashid et al. 2018, ICML);
    PyMARL upstream not present locally, so the algorithm is hand-authored
    to the paper rather than vendored byte-for-byte.
  * ``MAMuZeroGHAlgorithm`` — Tier-1 real port. Per pkg-07 spec 06 §3.10,
    ships the **per-agent vanilla MuZero + averaged cooperative reward**
    weak version (the spec-approved fallback when the joint world-model
    fails to converge inside the smoke budget; muzero-general upstream not
    present locally).
  * ``MAMBAAlgorithm`` — Tier-2 stub via class binding at import time
    (pkg-07 spec 06 §4.6: ``MAMBAAlgorithm = _RealMAMBA if IS_SOURCED else
    _MAMBAStub``).
  * ``MARIEStub`` / ``GAStub`` — permanent stubs (design D9).

CTDE legitimacy boundary (pkg-07 spec 06 Lock 3 + §6.1)
-------------------------------------------------------

``_FORBIDDEN_INFO_KEYS`` is hoisted to this module so that all four runner
modules (``mappo.py``, ``qmix.py``, ``ma_muzero_gh.py``, and a future
sourced ``mamba.py``) share **one** source of truth. A future env change
that adds a field to its ``_oracle_fields`` / ``_eval_only_fields``
markers requires a synchronous edit HERE, not in four separate runner
files.
"""
from __future__ import annotations

from typing import Final

from .base import ExternalBaselineRunner

# pkg-07 spec 06 §6.1 + Lock 3 — single source of truth for the four
# runner modules. The literal frozenset({...}) lives ONLY here; runners
# import it via ``from hyper_mve.baselines.external import _FORBIDDEN_INFO_KEYS``.
# v5 (Pkg-09): oracle fields are the regime id + all-agent rows; the legacy
# c_true/types names are kept in the guard until Stage-6 cleanup (harmless —
# the v5 env never emits them).
_FORBIDDEN_INFO_KEYS: Final[frozenset[str]] = frozenset({
    "g_true", "rows",
    "resource_state",
    "c_true", "types", "hotspot_centers",   # legacy v4 names (defence in depth)
})

# Defer concrete-class imports until after _FORBIDDEN_INFO_KEYS is bound, so
# the runner modules can import it from this package safely (avoids a partial-
# initialisation ImportError).
from .ma_muzero_gh import MAMuZeroGHAlgorithm  # noqa: E402
from .mamba import MAMBAAlgorithm              # noqa: E402  (class-bound at import)
from .mappo import MAPPOAlgorithm              # noqa: E402
from .qmix import QMIXAlgorithm                # noqa: E402
from .stubs import GAStub, MARIEStub           # noqa: E402

__all__ = [
    "ExternalBaselineRunner",
    "MAPPOAlgorithm",
    "QMIXAlgorithm",
    "MAMuZeroGHAlgorithm",
    "MAMBAAlgorithm",
    "MARIEStub",
    "GAStub",
    "_FORBIDDEN_INFO_KEYS",
]
