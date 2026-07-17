"""TensorBoard event-file scraper for TB-only runs.

Some launchers (the LoRA gen-scope cells; the legacy ``run_lora_experiments``)
write only TB scalars + ``train.log`` — no registry / EvalReport — so any
structured comparison across those runs must read the event files directly.

Uses TensorBoard's ``EventAccumulator`` (lazy import so importing this module
never forces a tensorboard dependency); raises a clear ImportError only when a
scrape is actually attempted without tensorboard installed.
"""
from __future__ import annotations

import pathlib

__all__ = ["scrape_tb", "last_scalar", "scrape_runs_root"]


def _event_accumulator():
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except Exception as e:  # pragma: no cover - env-dependent
        raise ImportError(
            "TB scraping needs tensorboard: `pip install tensorboard`. "
            "(The training/eval envs already have it; this is only needed for "
            "analysing TB-only runs such as runs/lora_sweep.)"
        ) from e
    return EventAccumulator


def scrape_tb(
    tb_dir: pathlib.Path | str, tags: list[str] | None = None
) -> dict[str, list[tuple[int, float]]]:
    """Return ``{scalar_tag: [(step, value), ...]}`` for one run's ``tb/`` dir.

    ``tags=None`` returns every scalar tag present; otherwise only the requested
    tags that exist are returned (silently skipping absent ones).
    """
    ea = _event_accumulator()
    acc = ea(str(tb_dir))
    acc.Reload()
    available = list(acc.Tags().get("scalars", []))
    chosen = available if tags is None else [t for t in tags if t in available]
    out: dict[str, list[tuple[int, float]]] = {}
    for t in chosen:
        out[t] = [(int(ev.step), float(ev.value)) for ev in acc.Scalars(t)]
    return out


def last_scalar(tb_dir: pathlib.Path | str, tag: str) -> tuple[int, float] | None:
    """``(last_step, last_value)`` for ``tag`` in one run, or ``None`` if absent."""
    series = scrape_tb(tb_dir, [tag]).get(tag) or []
    return series[-1] if series else None


def scrape_runs_root(
    root: pathlib.Path | str, tag: str, *, glob: str = "**/tb"
) -> dict[str, tuple[int, float]]:
    """Find every TB run under ``root`` and return ``{run_name: (last_step, last_value)}``.

    A "run" is any directory matching ``glob`` (default ``**/tb``). The run name
    is the run dir's parent path relative to ``root`` (forward-slashed), e.g.
    ``2agent/film_head/off`` for ``runs/lora_sweep/2agent/film_head/off/tb``.
    """
    root = pathlib.Path(root)
    out: dict[str, tuple[int, float]] = {}
    for tb_dir in sorted(root.glob(glob)):
        if not tb_dir.is_dir():
            continue
        last = last_scalar(tb_dir, tag)
        if last is None:
            continue
        name = str(tb_dir.parent.relative_to(root)).replace("\\", "/")
        out[name] = last
    return out
