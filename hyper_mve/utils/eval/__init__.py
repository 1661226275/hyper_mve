"""Evaluation contracts (phase-2 layout).

Public surface:

    >>> from hyper_mve.utils.eval import EvalReport

``EvalReport`` (schema rel-v1) is the unique evaluator output; every runner's
``evaluate`` returns one. Offline metrics live beside it:
``game_metrics.py`` (NashConv / efficiency, schema game-metrics-v1) and
``fidelity.py`` (world-model reward fidelity, schema fidelity-v1 — phase 7).
The v5 ``unified_evaluator.evaluate`` dispatcher is retired; the unified
train script calls ``runner.evaluate`` directly.
"""
from __future__ import annotations

from .eval_report import EvalReport

__all__ = ["EvalReport"]
