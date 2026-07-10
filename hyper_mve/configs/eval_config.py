"""EvalConfig — evaluation protocol parameters (Pkg-07 implements).

[v5 Pkg-09] The in-training periodic evaluation (training/evaluation.py,
wired into train_main) consumes ``evaluate_freq`` + the ``eval_regime_grid`` /
``eval_episodes_*`` fields below. Zero-shot regime generalisation needs no
extra fields here: holdout is declared at the env level
(``EnvConfig.train_regime_ids``) and probing it just means evaluating the
full family grid (``eval_regime_grid=None``).
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
    # deterministic (argmax, epsilon=0) episodes on regime-pinned eval envs.
    # "prior" = planner OFF (the distilled policy pi_hat); "planner" = planner ON
    # (the true acting agent). The planner-prior return gap measures distillation.
    eval_episodes_prior: int = 4
    eval_episodes_planner: int = 2

    # [v5 Pkg-09] per-regime eval grid: regime ids pinned via reset options
    # {"g": gid}. None = every regime in the preset's family (including any
    # train_regime_ids holdout — that IS the zero-shot probe).
    eval_regime_grid: tuple[int, ...] | None = None

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
