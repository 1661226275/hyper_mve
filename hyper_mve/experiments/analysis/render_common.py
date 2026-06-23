"""Shared helpers for the thesis table/figure renderers.

Pure-stdlib (no matplotlib/torch) so it loads anywhere. Provides:
  * deliverable-id normalisation (``"Table 6.1 (Easy gate)"`` → ``"Table 6.1"``),
  * per-cell registry/EvalReport loading (``cell_rows``),
  * the thesis 5-metric column spec,
  * a per-(variant) mean ± sem helper and an Abl3 type-ratio key extractor.
"""
from __future__ import annotations

import math
import pathlib
import re
from dataclasses import dataclass
from typing import Any

from . import registry_io

__all__ = [
    "RenderOutcome",
    "deliverable_token",
    "cell_registry_path",
    "cell_rows",
    "WELFARE_METRICS",
    "mean_sem",
    "type_ratio_of_row",
]


@dataclass
class RenderOutcome:
    deliverable: str
    status: str            # rendered | partial | blocked | no_data | error
    path: pathlib.Path | None = None
    message: str = ""


_TOKEN_RE = re.compile(r"((?:Table|Fig)\s*6\.\d+)")


def deliverable_token(s: str) -> str:
    """Canonical ``"Table 6.N"`` / ``"Fig 6.N"`` token, dropping any suffix."""
    m = _TOKEN_RE.search(s.replace("Fig.", "Fig"))
    return m.group(1) if m else s.strip()


def cell_registry_path(suite_root: pathlib.Path | str, cell) -> pathlib.Path:
    return pathlib.Path(suite_root) / cell.runs_subdir / "registry.jsonl"


def cell_rows(suite_root: pathlib.Path | str, cell) -> list[dict[str, Any]]:
    """Latest-per-run completed registry rows for one cell (``[]`` if none yet)."""
    return registry_io.load_completed_rows(cell_registry_path(suite_root, cell))


# Thesis Table 6.1 columns (Ch3.8.3). ``return_mean`` is social TOTAL welfare;
# the other four come from the §F eval extension.
WELFARE_METRICS: list[tuple[str, str]] = [
    ("return_mean", "W_total 社会总福利"),
    ("welfare_physical_mean", "W_phys 社会物理福利"),
    ("sustainability_mean", "S 可持续性"),
    ("fairness_mean", "F 公平性"),
    ("tragedy_index_mean", "T 悲剧指数"),
]


def mean_sem(xs: list[float]) -> tuple[float, float]:
    """(mean, standard-error-of-mean); sem is NaN for n<2."""
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(xs) / n
    if n < 2:
        return mean, float("nan")
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return mean, math.sqrt(var) / math.sqrt(n)


def type_ratio_of_row(row: dict[str, Any]) -> tuple[int, int] | None:
    """(n_alpha, n_beta) from an Abl3 row's ``env.type_assignment`` override.

    Reads the row's config snapshot if present; falls back to parsing the
    ablation_cell / run_tag is not reliable, so returns ``None`` when the
    assignment can't be recovered. ALPHA=0, BETA=1.
    """
    snap = row.get("config_snapshot_path")
    if snap and pathlib.Path(snap).exists():
        import json
        try:
            cfg = json.loads(pathlib.Path(snap).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        ta = (cfg.get("env") or {}).get("type_assignment")
        if ta is not None:
            vals = [int(x) for x in ta]
            return (sum(1 for v in vals if v == 0), sum(1 for v in vals if v == 1))
    return None
