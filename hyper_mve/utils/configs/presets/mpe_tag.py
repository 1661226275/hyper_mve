"""mpe_tag presets — regime-ified MPE simple_tag (phase-3 realignment).

Two configurations over ``MPETagRegimeEnv`` (N=4: predators 0–2, prey 3;
family ``tag4``, |G| = 5; obs = 16 padded + own row 3 = 19; Discrete(5);
``epsilon_move=0`` so the W = I regime reproduces raw simple_tag rewards):

* ``mpe_tag`` — the main hidden-regime config: regime resampled per episode
  from the seen set ``train_regime_ids=(0, 1, 2)``
  (pred_full_coalition / pred_pair_coalition / all_solo); regimes 3–4
  (pred_rivalry / prey_sympathizer) are the zero-shot holdout.
* ``mpe_tag_fixed`` — fixed-role calibration: ``fixed_regime=2`` (W = I)
  ⇒ standard simple_tag, comparable against published numbers.

Model/train/eval sections are inherited from the rel_duo preset (they are
env-size-agnostic: row/belief dims derive from N and |G| at build time).
"""
from __future__ import annotations

from dataclasses import replace

from ..env_config import EnvConfig
from ..v4_config import V4Config
from .rel_duo import build_rel_duo_config


def _tag_env(**overrides) -> EnvConfig:
    base = dict(
        N=4,
        L=1,                    # unused by MPE (RelationCommons grid size)
        K=0,                    # unused by MPE (resource cells)
        T_max=25,               # standard simple_tag episode length
        A=5,                    # no_action + 4 directions
        env_kind="mpe_tag",
        relation_family="tag4",
        relation_intensity=1.0,
        regime_switch_prob=0.0,  # research point 1: per-episode static
        regime_kernel="uniform",
        epsilon_move=0.0,        # keep W=I ⇒ raw simple_tag rewards exact
    )
    base.update(overrides)
    return EnvConfig(**base)


def build_mpe_tag_config() -> V4Config:
    cfg = build_rel_duo_config()
    return replace(
        cfg,
        env=_tag_env(train_regime_ids=(0, 1, 2)),
        preset_name="mpe_tag",
    )


def build_mpe_tag_fixed_config() -> V4Config:
    cfg = build_rel_duo_config()
    return replace(
        cfg,
        env=_tag_env(fixed_regime=2),
        preset_name="mpe_tag_fixed",
    )
