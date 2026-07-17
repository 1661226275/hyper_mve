"""Analysis layer for the thesis experiment suite.

Two halves:

* **IO** — :mod:`registry_io` (sweep registry / EvalReport → per-variant metric
  samples + CSV) and :mod:`tb_scraper` (TB-only runs → scalar tables).
* **Renderers** — :mod:`tables` / :mod:`figures` produce the v5 thesis tables
  (6.1 rel gate / 6.2 regime holdout) and figures (6.1 per-regime returns /
  6.2 seen-vs-unseen); :data:`RENDERERS` maps a deliverable id
  (e.g. ``"Table 6.1"``) to its renderer for ``make_thesis_artifacts``.

The renderer registry is imported lazily so this package stays importable in
environments without matplotlib (the IO half is pure-stdlib).
"""
from __future__ import annotations

from . import registry_io, tb_scraper

__all__ = ["registry_io", "tb_scraper", "get_renderers"]


def get_renderers() -> dict:
    """Return the ``{deliverable_id: renderer_fn}`` map (lazy; needs matplotlib)."""
    from . import tables, figures  # noqa: F401  (populates RENDERERS)

    renderers: dict = {}
    renderers.update(getattr(tables, "RENDERERS", {}))
    renderers.update(getattr(figures, "RENDERERS", {}))
    return renderers
