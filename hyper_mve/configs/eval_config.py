"""EvalConfig — evaluation protocol parameters (Pkg-07 implements).

[v4-opt 2026-06] The in-training periodic evaluation (training/evaluation.py,
wired into train_main) consumes ``evaluate_freq`` + the ``eval_c_grid`` /
``eval_episodes_*`` fields below. The remaining scan-point fields stay reserved
for the full Pkg-07 protocol.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation cadence and scan points (Ch6.2 / 6.6 / 6.9)."""

    # [v4-opt 2026-06] 500 -> 1000: in-training eval cadence (after buffer warmup).
    evaluate_freq: int = 1000           # evaluate every N train steps (0 disables)
    evaluate_episodes: int = 30         # reserved for the full Pkg-07 suite

    # [v4-opt 2026-06] in-training dual-mode eval (training/evaluation.py):
    # deterministic (argmax, epsilon=0) episodes on static-c eval envs, per c value.
    # "prior" = planner OFF (the distilled policy pi_hat); "planner" = planner ON
    # (the true acting agent). The planner-prior return gap measures distillation.
    eval_c_grid: tuple[float, ...] = (0.2, 0.5, 0.8)   # DEPRECATED (v5: per-regime eval)
    eval_episodes_prior: int = 4
    eval_episodes_planner: int = 2

    # [v5 Pkg-09] per-regime eval grid: regime ids pinned via reset options
    # {"g": gid}. None = every regime in the preset's family (including any
    # train_regime_ids holdout — that IS the zero-shot probe).
    eval_regime_grid: tuple[int, ...] | None = None

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

    # [pkg-08 spec 03 / design D7 + D10] Four-mode planner eval dispatch literal.
    # Default ``planner_full`` reproduces legacy `eval_mode="planner"` semantics.
    eval_planner_mode: Literal[
        "direct_inference",
        "planner_no_crn",
        "planner_no_coord_desc",
        "planner_full",
    ] = "planner_full"

    # [pkg-08 spec 03 / design D7] Mode short-circuit: when True the unified
    # evaluator forces ``eval_planner_mode="direct_inference"`` regardless of
    # the field above (CLI fast-toggle).
    eval_use_planner_direct_inference: bool = False
