"""Vendored PyMARL kernel — QMIX algorithm (pkg-07 spec 06 §2.1).

Source: ``oxwhirl/pymarl`` @ commit ``c971afdceb34635d31b778021b0ef90d7af51e86``
        License: Apache-2.0 (see ``../_third_party_licenses/pymarl_LICENSE.txt``).

Files vendored byte-faithful from upstream:

  * ``rnn_agent.py``       ← ``src/modules/agents/rnn_agent.py``
  * ``qmix_mixer.py``      ← ``src/modules/mixers/qmix.py``  (renamed to avoid
                              clashing with ``hyper_mve/baselines/external/qmix.py``
                              on case-insensitive filesystems)
  * ``q_learner.py``       ← ``src/learners/q_learner.py``   (algorithm body
                              consumed by ``hyper_mve.baselines.external.qmix``;
                              upstream imports reference pymarl-internal
                              EpisodeBatch / MAC / logger which the runner
                              re-implements rather than vendoring)
  * ``action_selectors.py`` ← ``src/components/action_selectors.py``  (idem;
                              the runner uses the ``epsilon_greedy`` branch
                              with a hand-built schedule).

Per pkg-07 spec 06 §2.2: "the port keeps the **algorithm logic** intact and
**rewrites the surrounding plumbing**". The algorithm-relevant modules
(``RNNAgent`` + ``QMixer``) are imported and instantiated unmodified; the
rest of pymarl (Sacred, YAML configs, SC2 multiprocess workers, ``EpisodeBatch``
tensor layout) is replaced by the runner's own bridging layer.
"""
from __future__ import annotations

#: Provenance marker (pkg-07 spec 06 §2.1; mirrored on the runner module).
_VENDORED_FROM: str = "pymarl @ c971afdceb34635d31b778021b0ef90d7af51e86"

# We deliberately do NOT re-export RNNAgent / QMixer here — runners import
# them via the explicit submodule path so the vendor boundary stays grep-able.

__all__ = ["_VENDORED_FROM"]
