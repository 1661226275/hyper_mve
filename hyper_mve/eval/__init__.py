"""Evaluation package (pkg-08).

Public surface:

    >>> from hyper_mve.eval import EvalReport, evaluate
    >>> report = evaluate(runner, env_fn, cfg)

``EvalReport`` is the unique evaluator output schema (pkg-08 spec 01 Lock 2);
``evaluate`` is the unified dispatch over internal + external runners.
"""
from __future__ import annotations

from .eval_report import EvalReport
from .unified_evaluator import evaluate

__all__ = ["EvalReport", "evaluate"]
