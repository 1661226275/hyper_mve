"""make_thesis_artifacts.py — render every thesis table/figure from suite runs.

Iterates the suite manifest, maps each cell's declared deliverables to a renderer
(exact-string override first, else the canonical ``Table 6.N`` / ``Fig 6.N``
token), and writes ``results/{tables,figures}/`` plus ``results/results_index.md``
with a status per deliverable (rendered / partial / blocked / no_data / error /
no_renderer).

    python hyper_mve/scripts/make_thesis_artifacts.py \
        --suite-root runs/suite --out results

Renderers degrade gracefully: a deliverable whose cell hasn't run yet is
``no_data``. Figure renderers need matplotlib; without it they report
``error`` but tables still render.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
from collections import OrderedDict
from typing import Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hyper_mve.experiments.suite import load_manifest  # noqa: E402
from hyper_mve.experiments.analysis import tables, figures  # noqa: E402
from hyper_mve.experiments.analysis.render_common import RenderOutcome, deliverable_token  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python hyper_mve/scripts/make_thesis_artifacts.py",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--suite-root", dest="suite_root", type=pathlib.Path,
                   default=pathlib.Path("runs/suite"),
                   help="root holding the suite's per-cell run dirs (default runs/suite)")
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results"),
                   help="output dir for tables/ figures/ + results_index.md (default results)")
    p.add_argument("--manifest", type=pathlib.Path, default=None)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cells = load_manifest(args.manifest)

    token_renderers = {**tables.RENDERERS, **figures.RENDERERS}
    exact_renderers = {
        **getattr(tables, "EXACT_RENDERERS", {}),
        **getattr(figures, "EXACT_RENDERERS", {}),
    }

    # Exact deliverable string → cells declaring it (manifest order preserved).
    by_deliv: "OrderedDict[str, list]" = OrderedDict()
    for cell in cells:
        for d in cell.deliverables:
            by_deliv.setdefault(d, []).append(cell)

    tables_dir = args.out / "tables"
    figs_dir = args.out / "figures"
    index = [
        "# Thesis artifacts index", "",
        f"_suite root: `{args.suite_root}`_", "",
        "| deliverable | status | output | cells | note |",
        "|---|---|---|---|---|",
    ]
    n_rendered = 0
    for deliv, dcells in by_deliv.items():
        tok = deliverable_token(deliv)
        renderer = exact_renderers.get(deliv) or token_renderers.get(tok)
        # Per-deliverable subdir: several exact strings (e.g. "Table 6.4" and
        # "Table 6.4 (Easy gate)") map to the same renderer, which writes a fixed
        # filename — isolate them so they don't overwrite each other.
        safe = (deliv.replace(" ", "_").replace("(", "").replace(")", "")
                .replace("/", "-").replace(".", "_"))
        out_dir = (figs_dir if tok.startswith("Fig") else tables_dir) / safe
        if renderer is None:
            index.append(f"| {deliv} | no_renderer | — | "
                         f"{', '.join(c.id for c in dcells)} | (no renderer mapped) |")
            print(f"[no_renderer] {deliv}")
            continue
        try:
            oc = renderer(args.suite_root, dcells, out_dir)
        except Exception as e:  # noqa: BLE001 — a bad renderer shouldn't sink the run
            oc = RenderOutcome(deliv, "error", None, f"{type(e).__name__}: {e}")
        rel = None
        if oc.path is not None:
            try:
                rel = oc.path.relative_to(args.out)
            except ValueError:
                rel = oc.path
        if oc.status == "rendered":
            n_rendered += 1
        index.append(f"| {deliv} | {oc.status} | {rel or '—'} | "
                     f"{', '.join(c.id for c in dcells)} | {oc.message} |")
        print(f"[{oc.status:>10}] {deliv:<34} {oc.message}")

    args.out.mkdir(parents=True, exist_ok=True)
    idx_path = args.out / "results_index.md"
    idx_path.write_text("\n".join(index) + "\n", encoding="utf-8")
    print(f"\n{n_rendered}/{len(by_deliv)} deliverables rendered → {idx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
