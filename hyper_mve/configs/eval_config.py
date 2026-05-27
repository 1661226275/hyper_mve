"""EvalConfig — evaluation protocol parameters (Pkg-07 implements)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation cadence and scan points (Ch6.2 / 6.6 / 6.9)."""

    evaluate_freq: int = 500            # evaluate every N train steps
    evaluate_episodes: int = 30

    # c-segment evaluation (Ch6.2.4)
    c_segments: tuple[tuple[float, float], ...] = (
        (0.0, 0.3),
        (0.3, 0.7),
        (0.7, 1.0),
    )

    # Zero-shot c generalisation (Ch6.9)
    zero_shot_train_c: tuple[float, ...] = (0.2, 0.5, 0.8)
    zero_shot_test_c: tuple[float, ...] = (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)
    zero_shot_unseen_c: tuple[float, ...] = (0.0, 0.35, 0.65, 1.0)

    # Bell curve over type ratios (Ch6.6): (n_alpha, n_beta) pairs
    bell_curve_type_ratios: tuple[tuple[int, int], ...] = (
        (0, 4), (1, 3), (2, 2), (3, 1), (4, 0), (1, 3),
    )
