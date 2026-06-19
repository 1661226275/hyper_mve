"""Convert a TensorBoard event file to CSVs (one per scalar tag) + a wide-form combined CSV.

Usage:
    python tb_to_csv.py <event_file_or_dir> [--out <out_dir>]

If a directory is given, all events.out.tfevents.* files inside it are merged.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalars(path: str) -> dict[str, pd.DataFrame]:
    ea = EventAccumulator(path, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    out: dict[str, pd.DataFrame] = {}
    for tag in tags:
        events = ea.Scalars(tag)
        df = pd.DataFrame(
            {
                "wall_time": [e.wall_time for e in events],
                "step": [e.step for e in events],
                "value": [e.value for e in events],
            }
        )
        out[tag] = df
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Event file or directory containing event files")
    ap.add_argument("--out", default=None, help="Output directory (default: <path>_csv)")
    args = ap.parse_args()

    src = Path(args.path)
    if not src.exists():
        print(f"Path not found: {src}", file=sys.stderr)
        return 1

    out_dir = Path(args.out) if args.out else src.parent / (src.stem + "_csv" if src.is_file() else src.name + "_csv")
    out_dir.mkdir(parents=True, exist_ok=True)

    scalars = load_scalars(str(src))
    if not scalars:
        print("No scalar tags found in event file.", file=sys.stderr)
        return 2

    print(f"Found {len(scalars)} scalar tags. Writing to {out_dir}")

    summary_rows = []
    wide_frames: list[pd.DataFrame] = []
    for tag, df in scalars.items():
        safe = tag.replace("/", "__").replace("\\", "__")
        per_tag_path = out_dir / f"{safe}.csv"
        df.to_csv(per_tag_path, index=False)
        summary_rows.append(
            {
                "tag": tag,
                "n_points": len(df),
                "step_min": int(df["step"].min()) if len(df) else None,
                "step_max": int(df["step"].max()) if len(df) else None,
                "value_min": float(df["value"].min()) if len(df) else None,
                "value_max": float(df["value"].max()) if len(df) else None,
                "value_first": float(df["value"].iloc[0]) if len(df) else None,
                "value_last": float(df["value"].iloc[-1]) if len(df) else None,
            }
        )
        wide_frames.append(df[["step", "value"]].rename(columns={"value": tag}).set_index("step"))

    summary = pd.DataFrame(summary_rows).sort_values("tag")
    summary.to_csv(out_dir / "_summary.csv", index=False)

    if wide_frames:
        wide = pd.concat(wide_frames, axis=1).sort_index()
        wide.to_csv(out_dir / "_all_scalars_by_step.csv")

    print("Done.")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
