"""Plot loss/value, loss/reward, loss/consist, loss/policy curves from the wide-form CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default=r"D:\RL\hyper_mve\events.out.tfevents.1780500354.llabsrv01.125514_csv\_all_scalars_by_step.csv",
    )
    ap.add_argument("--out", default=None, help="Output PNG path (default: alongside csv)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)

    tags = ["loss/policy", "loss/value", "loss/reward", "loss/consist"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes = axes.flatten()

    for ax, tag in zip(axes, tags):
        ax.plot(df["step"], df[tag], marker="o", markersize=3, linewidth=1.2, color="#1f77b4")
        ax.set_title(tag)
        ax.set_xlabel("step")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.3)
        if tag in {"loss/value", "loss/reward", "loss/consist"}:
            ax.set_yscale("log")

    fig.suptitle(
        f"Training losses ({csv_path.parent.name})", fontsize=12
    )
    out_path = Path(args.out) if args.out else csv_path.parent / "losses_overview.png"
    fig.savefig(out_path, dpi=140)
    print(f"saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
