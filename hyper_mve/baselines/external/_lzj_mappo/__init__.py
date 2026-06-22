"""Vendored MAPPO source — algorithm logic intact (lzj-mappo).

Source: ``D:\\RL\\lzj\\MAPPO\\`` (4 files, mtime 2025-04-01).
md5(``mappo.py``) = ``960d27648d56315a95cba837efd844dd``.

This subpackage holds the unmodified ``MAPPO_MPE`` algorithm class +
``ReplayBuffer`` + ``Normalization``/``RewardScaling`` helpers. The plumbing
that adapted these to the lzj/MPE entry-point lives in the parent module
``hyper_mve.baselines.external.mappo``, which builds the runtime ``args``
namespace from a ``V4Config`` and drives the training loop against
``ResourceCommonsPettingZooEnv``.
"""
