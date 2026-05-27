"""v4.7 archive (frozen at git tag v4.7-final).

This module is deprecated. v4 code should NOT import from here.
For v4.7 reproducibility, use ``git checkout v4.7-final`` to restore
the original working tree.

Importing any submodule of this package triggers ``DeprecationWarning``.

This package also injects its own directory into ``sys.path`` so that the
archived v4.7 source files (which use unqualified imports such as
``from envs.make_env import ...``) continue to resolve after the move.
"""
import os
import sys
import warnings

warnings.warn(
    "Importing from hyper_mve._legacy_v4_7 is deprecated. "
    "v4.7 code is archived for paper Ch6.12 reproducibility and rollback only. "
    "Use 'git checkout v4.7-final' for the full v4.7 working tree.",
    DeprecationWarning,
    stacklevel=2,
)

_ARCHIVE_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ARCHIVE_ROOT not in sys.path:
    sys.path.insert(0, _ARCHIVE_ROOT)
