"""Vendored muzero-general kernel — minimal subset (pkg-07 spec 06 §3.1).

Source: ``werner-duvaud/muzero-general`` @ commit
        ``0825bd544fc172a2e2dcc96d43711123222c4a2f``
        License: MIT (see ``../_third_party_licenses/muzero_general_LICENSE.txt``).

Files vendored from upstream:

  * ``models.py``     ← upstream root ``models.py`` (byte-faithful — provides
                        ``MuZeroFullyConnectedNetwork``, ``support_to_scalar``,
                        ``scalar_to_support``, ``mlp``).
  * ``self_play.py``  ← upstream root ``self_play.py`` with TWO minimal edits
                        documented inline at the top of that file:
                        (1) ``import models`` → ``from . import models`` so
                        the file imports inside this subpackage; (2) ``import
                        ray`` and the ``@ray.remote`` decorator on
                        ``SelfPlay`` are removed (the runner only uses
                        ``MCTS`` / ``Node`` / ``MinMaxStats`` / ``GameHistory``;
                        ``SelfPlay`` is a Ray orchestrator we replace with
                        an in-process loop, and never instantiate, so the
                        residual ``ray.get(...)`` calls in its method
                        bodies are dead code).

NOT vendored (deliberately): ``muzero.py`` (Ray orchestrator),
``trainer.py`` (reanalyze loop), ``replay_buffer.py`` (PER buffer).
The pkg-07 runner re-implements these as a single-process train loop
matching the §3.10 per-agent weak-fallback architecture.
"""
from __future__ import annotations

#: Provenance marker (pkg-07 spec 06 §3.1; mirrored on the runner module).
_VENDORED_FROM: str = "muzero-general @ 0825bd544fc172a2e2dcc96d43711123222c4a2f"

__all__ = ["_VENDORED_FROM"]
