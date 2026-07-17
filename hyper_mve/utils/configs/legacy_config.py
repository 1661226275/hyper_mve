"""LegacyConfig — v4.7 leftover knobs kept for rollback experiments.

v4 Pass 2 deletions (architecture made them unnecessary):
    - ``w_rew_diversity`` / ``rew_diversity_*``  — type_emb routed through
      the role path (Ch4.2.2 + 4.3.3) eliminates the RewardHead collapse
      that motivated the diversity regulariser.
    - ``detach_pred_context``  — replaced by BeliefNet gradient gating
      (Ch4.6.5), which is a different mechanism.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LegacyConfig:
    """v4.7-only fields, default-disabled.

    Adversarial Freezing (v4.2) and ChunkedHMLP (v4.5) are orthogonal to v4
    and remain available as fallback experiment switches.
    """

    # Adversarial Freezing (v4.2)
    freeze_enabled: bool = False
    freeze_warmup_steps: int = 4000
    freeze_phase_steps: int = 4000
    freeze_hunter_agents: tuple[int, ...] = (0, 1, 2)
    freeze_prey_agents: tuple[int, ...] = (3,)

    # ChunkedHMLP (v4.5, deprecated fallback)
    chunk_alpha: int = 10
    chunk_emb_size: int = 8
    chunk_budget_factor: float = 1.0
    chunk_hyperfan_init: bool = True
