"""Registry / EvalReport IO for the analysis layer.

Reads a sweep cell's ``runs/<cell>/registry.jsonl``, reduces to the latest
``status="completed"`` row per ``run_id``, and exposes per-variant metric
samples + per-variant CSV emission that the ``compare`` CLI
(:func:`hyper_mve.experiments.compare._load_csv_returns`) can consume.

Pure-stdlib (json / pathlib / csv) — no torch, numpy or yaml — so it loads in
any environment. The 3 metrics the sweep duplicates into the registry summary
(``return_mean`` / ``return_zero_shot_unseen`` / ``regret_mean``; see
``run_registry.RegistryRow``) are read straight from the JSONL; every other
``EvalReport`` field is read on demand from the row's ``eval_report_path`` JSON
(cached per file).
"""
from __future__ import annotations

import csv
import json
import pathlib
from typing import Any

__all__ = [
    "load_completed_rows",
    "metric_value",
    "per_variant_metric",
    "method_samples",
    "report_dict",
    "to_compare_csv",
    "REGISTRY_SUMMARY_METRICS",
]

# Duplicated into RegistryRow by the sweep harness (run_registry.py); readable
# without opening the per-run EvalReport JSON.
REGISTRY_SUMMARY_METRICS: tuple[str, ...] = (
    "return_mean",
    "return_zero_shot_unseen",
    "regret_mean",
)


def load_completed_rows(registry_path: pathlib.Path | str) -> list[dict[str, Any]]:
    """Latest-per-``run_id``, ``status="completed"`` rows from a JSONL registry.

    Mirrors the reduction in ``compare.load_registry`` but is import-light
    (no matplotlib). Returns ``[]`` if the registry file does not exist.
    """
    p = pathlib.Path(registry_path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(p, "rb") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line.decode("utf-8")))
            except json.JSONDecodeError:
                continue
    by_id: dict[str, dict[str, Any]] = {}
    for r in rows:
        rid = r.get("run_id")
        if rid is None:
            continue
        if (
            rid not in by_id
            or r.get("started_at_iso8601", "") >= by_id[rid].get("started_at_iso8601", "")
        ):
            by_id[rid] = r
    return [r for r in by_id.values() if r.get("status") == "completed"]


_REPORT_CACHE: dict[str, dict[str, Any] | None] = {}


def _load_report(path: str) -> dict[str, Any] | None:
    if path in _REPORT_CACHE:
        return _REPORT_CACHE[path]
    p = pathlib.Path(path)
    report: dict[str, Any] | None
    if not p.exists():
        report = None
    else:
        try:
            report = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            report = None
    _REPORT_CACHE[path] = report
    return report


def report_dict(row: dict[str, Any]) -> dict[str, Any] | None:
    """The row's full EvalReport JSON (cached), or ``None``.

    For non-scalar rel-v1 fields (``return_per_regime`` etc.) that
    :func:`metric_value` cannot deliver.
    """
    path = row.get("eval_report_path")
    if not path:
        return None
    return _load_report(str(path))


def metric_value(row: dict[str, Any], metric: str) -> float | None:
    """Scalar value of ``metric`` for one registry row, or ``None``.

    Fast path: the 3 registry-summary metrics. Otherwise the row's
    ``eval_report_path`` JSON is opened (cached) and ``metric`` read from it.
    """
    if metric in REGISTRY_SUMMARY_METRICS and row.get(metric) is not None:
        try:
            return float(row[metric])
        except (TypeError, ValueError):
            return None
    path = row.get("eval_report_path")
    if not path:
        v = row.get(metric)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
    report = _load_report(str(path))
    if report is None:
        return None
    v = report.get(metric)
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def per_variant_metric(
    rows: list[dict[str, Any]], metric: str
) -> dict[str, list[tuple[int, float]]]:
    """``{variant: [(seed, value), ...]}`` sorted by seed; rows missing the metric are dropped."""
    out: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        variant = r.get("variant")
        if variant is None:
            continue
        v = metric_value(r, metric)
        if v is None:
            continue
        out.setdefault(str(variant), []).append((int(r.get("seed", -1)), v))
    for k in out:
        out[k].sort()
    return out


def method_samples(rows: list[dict[str, Any]], metric: str) -> dict[str, list[float]]:
    """``{variant: [value, ...]}`` (one per seed) — the shape ``stats.compare_methods`` wants."""
    return {v: [val for _seed, val in s] for v, s in per_variant_metric(rows, metric).items()}


def to_compare_csv(
    rows: list[dict[str, Any]], metric: str, out_dir: pathlib.Path | str
) -> dict[str, pathlib.Path]:
    """Emit one ``<variant>.csv`` (cols ``variant,seed,<metric>``) per variant.

    The schema matches ``compare._load_csv_returns`` (needs a ``variant`` column
    + the metric column, one row per seed), so the emitted files can be fed back
    into ``python -m hyper_mve.experiments.compare --methods a.csv,b.csv ...``.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, pathlib.Path] = {}
    for variant, samples in per_variant_metric(rows, metric).items():
        path = out_dir / f"{variant}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["variant", "seed", metric])
            for seed, val in samples:
                w.writerow([variant, seed, val])
        paths[variant] = path
    return paths
