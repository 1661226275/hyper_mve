"""SuiteCell — one thesis experiment = a SweepConfig + thesis metadata.

A *cell YAML* is an ordinary strict SweepConfig YAML plus an extra top-level
``meta:`` block carrying thesis bookkeeping that ``SweepConfig.from_yaml``
would reject:

    meta:
      id: rel_gate_duo
      title: "v5 relation gate — rel_duo N=2 混合 regime 主对照"
      tier: must_have            # must_have | degradable | cuttable
      size: medium               # easy | medium | hard (informational)
      deliverables: ["Table 6.4", "Fig 6.4"]
      assertion: "A — 类型梯度撕裂"
      runs_subdir: abl3_type_het # runs_root/<runs_subdir>/{registry.jsonl, <run_tag>/}
      blocked_on: []             # e.g. ["variant:no_type", "metric:gini", "external_mamba"]
      code_deps: []              # informational provenance only
    variants: [...]              # <- everything below is a literal SweepConfig key
    seeds: [...]
    ...

``load_cell`` pops ``meta`` first, then hands the remaining mapping to
``sweep.from_mapping`` (the single strict-parse path, shared with ``load_yaml``).
The pure ``cell_from_mapping`` is split out so the loader is testable without
pyyaml.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any, Mapping

from hyper_mve.experiments import sweep
from hyper_mve.experiments.sweep import SweepConfig

__all__ = [
    "SuiteCell",
    "cell_from_mapping",
    "load_cell",
    "load_manifest",
    "default_manifest_path",
    "TIERS",
]

TIERS = ("must_have", "degradable", "cuttable")

_META_KEYS = {
    "id", "title", "tier", "size", "deliverables",
    "assertion", "runs_subdir", "blocked_on", "code_deps", "notes",
}


@dataclass(frozen=True)
class SuiteCell:
    """A declarative experiment cell: thesis metadata + its SweepConfig."""

    id: str
    title: str
    tier: str
    size: str
    deliverables: tuple[str, ...]
    assertion: str
    runs_subdir: str
    blocked_on: tuple[str, ...]
    code_deps: tuple[str, ...]
    notes: str
    sweep_config: SweepConfig
    path: pathlib.Path | None = None

    @property
    def blocked(self) -> bool:
        """A cell is blocked while any unmet dependency is declared in ``blocked_on``."""
        return bool(self.blocked_on)

    def n_rows(self) -> int:
        """Enumerated row count |variants| × |seeds| × max(1, |overrides|)."""
        return len(sweep.enumerate_cartesian(self.sweep_config))


def cell_from_mapping(
    raw: Mapping[str, Any],
    *,
    source: str = "<mapping>",
    default_id: str | None = None,
    path: pathlib.Path | None = None,
) -> SuiteCell:
    """Build a :class:`SuiteCell` from an already-parsed mapping (no file IO).

    Pops the ``meta`` block (validated against :data:`_META_KEYS`), then strict-
    parses the remainder as a SweepConfig. ``default_id`` is the fallback cell id
    when ``meta.id`` is absent (the loader passes the YAML stem).
    """
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: cell top-level must be a mapping; got {type(raw).__name__}")
    body = dict(raw)
    meta = dict(body.pop("meta", {}) or {})
    unknown = set(meta) - _META_KEYS
    if unknown:
        raise ValueError(
            f"{source}: unknown meta keys {sorted(unknown)}; legal: {sorted(_META_KEYS)}"
        )
    sweep_cfg = sweep.from_mapping(body, source=source)

    cid = str(meta.get("id") or default_id or "cell")
    tier = str(meta.get("tier", "degradable"))
    if tier not in TIERS:
        raise ValueError(f"{source}: meta.tier={tier!r} not in {TIERS}")
    return SuiteCell(
        id=cid,
        title=str(meta.get("title", cid)),
        tier=tier,
        size=str(meta.get("size", sweep_cfg.preset)),
        deliverables=tuple(str(d) for d in (meta.get("deliverables", []) or ())),
        assertion=str(meta.get("assertion", "")),
        runs_subdir=str(meta.get("runs_subdir") or cid),
        blocked_on=tuple(str(b) for b in (meta.get("blocked_on", []) or ())),
        code_deps=tuple(str(c) for c in (meta.get("code_deps", []) or ())),
        notes=str(meta.get("notes", "")),
        sweep_config=sweep_cfg,
        path=path,
    )


def _read_yaml(path: pathlib.Path) -> dict:
    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        raise ImportError("pyyaml is required to load suite cells / manifest") from e
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_cell(path: pathlib.Path | str) -> SuiteCell:
    """Load one cell YAML (meta + SweepConfig) from disk."""
    p = pathlib.Path(path)
    raw = _read_yaml(p)
    return cell_from_mapping(raw, source=str(p), default_id=p.stem, path=p)


def default_manifest_path() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "manifest.yaml"


def load_manifest(path: pathlib.Path | str | None = None) -> list[SuiteCell]:
    """Load every cell listed in the suite manifest, in declared order.

    ``manifest.yaml`` lists cell paths under a ``cells:`` key, each resolved
    relative to the manifest's own directory (``cells/*.yaml``).
    """
    manifest_path = pathlib.Path(path) if path is not None else default_manifest_path()
    raw = _read_yaml(manifest_path)
    entries = raw.get("cells", []) or []
    base = manifest_path.parent
    out: list[SuiteCell] = []
    for entry in entries:
        rel = entry if isinstance(entry, str) else entry.get("path")
        if not rel:
            continue
        out.append(load_cell((base / rel).resolve()))
    return out
