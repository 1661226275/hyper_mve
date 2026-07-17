"""THE METHOD — MAZero-lineage mixed-game stack (phase-2 layout).

Contents:
- ``hyper_mve/algo/mazero_mixed/`` — the MAZero fork (GPL-3): vectorized
  decoupled Sampled-MCTS, hypernet-generated per-agent subjective heads,
  Bayes-averaged leaf values, belief curriculum. CWD-rooted internal imports
  (``from core...``); driven via :mod:`hyper_mve.algo.runner`.
- ``hyper_mve/algo/modules/`` — the subjective conditioning modules
  (belief / hypernet / functional heads) imported by the fork.
- ``hyper_mve/algo/runner.py`` — ``MAZeroMixedRunner``: the
  ``ExternalBaselineRunner``-contract bridge the unified train/eval spine
  drives ("mazero_mixed" registry key).

No eager imports here: the runner pulls in torch; import it explicitly or
resolve it lazily through ``hyper_mve.comparison.REGISTRY``.
"""
