"""eval_mamba_checkpoints.py — accurate 30-episode eval of MAMBA periodic checkpoints.

The 2M MAMBA run does not finish by the mid-term deadline; ``mamba.py:train``
drops a model checkpoint (``step_<env_steps>.pt``) every 25k env steps. This
script evaluates each checkpoint with the real per-regime 30-episode protocol
(``_RealMAMBA.evaluate``) — unlike the 2-episode in-training probe — producing
accurate ``(env_step -> return)`` points to fit a sample-efficiency projection
to the full 2M budget.

Resumable: re-running skips checkpoints already present in ``--out``, so it can
be re-invoked as more checkpoints accrue.

Run from the repo root:

    python hyper_mve/scripts/eval_mamba_checkpoints.py \
        --run-dir runs/suite/rel_gate_duo/external_mamba_seed0_row0_be184580 \
        --preset rel_duo --episodes 30 \
        --out runs/_analysis/rel_defense/mamba_ckpt_series_gate.json

Output JSON::

    {"cell","preset","episodes","budget_target",
     "points":[{"env_step","return_mean","per_regime":{g:v},"seen","unseen"}]}
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_STEP_RE = re.compile(r"step_(\d+)\.pt$")


def _checkpoints(run_dir: pathlib.Path, stride: int) -> list[tuple[int, pathlib.Path]]:
    """(env_step, path) for every ``step_*.pt`` in ``run_dir``, sorted, strided."""
    found: list[tuple[int, pathlib.Path]] = []
    for p in run_dir.glob("step_*.pt"):
        m = _STEP_RE.search(p.name)
        if m:
            found.append((int(m.group(1)), p))
    found.sort()
    if stride > 1:
        found = found[::stride]
    return found


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=pathlib.Path, required=True,
                    help="MAMBA run dir holding step_*.pt checkpoints")
    ap.add_argument("--preset", default="rel_duo",
                    help="rel_duo (gate) | rel_duo_holdout (zero-shot)")
    ap.add_argument("--episodes", type=int, default=30,
                    help="episodes/regime (matches the real eval contract)")
    ap.add_argument("--regimes", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--budget-target", type=float, default=2_000_000.0)
    ap.add_argument("--stride", type=int, default=1,
                    help="evaluate every Nth checkpoint (thin the series)")
    ap.add_argument("--include-final", action="store_true",
                    help="also evaluate ckpt_final.pt if present")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--device", default=None, help='"cpu"/"cuda" (default: auto)')
    args = ap.parse_args(argv)

    from hyper_mve.baselines import create_baseline
    from hyper_mve.configs import V4Config
    from hyper_mve.envs.adapters.pettingzoo_wrapper import RelationCommonsPettingZooEnv

    cfg = V4Config.from_preset(args.preset)
    env_fn = lambda: RelationCommonsPettingZooEnv(
        cfg.env, oracle_mode=False, eval_info_mode=False,
    )
    grid = tuple(int(g) for g in args.regimes)

    # Resume: keep points already computed (keyed by env_step).
    prev: dict[int, dict] = {}
    if args.out.exists():
        try:
            for pt in json.loads(args.out.read_text()).get("points", []):
                prev[int(pt["env_step"])] = pt
        except (json.JSONDecodeError, OSError, KeyError):
            prev = {}

    ckpts = _checkpoints(args.run_dir, max(args.stride, 1))
    if args.include_final and (args.run_dir / "ckpt_final.pt").exists():
        # place final at its true budget if the run actually completed
        ckpts.append((int(args.budget_target), args.run_dir / "ckpt_final.pt"))

    points: dict[int, dict] = dict(prev)
    for step, path in ckpts:
        if step in points:
            continue                      # already evaluated (resume)
        runner = create_baseline(cfg, "external_mamba")
        try:
            runner.load_checkpoint(str(path))
            rep = runner.evaluate(env_fn, regime_grid=grid, episodes=args.episodes)
        except Exception as exc:          # truncated/partial ckpt → try next time
            print(f"[eval_mamba_checkpoints] skip {path.name}: {exc}", file=sys.stderr)
            continue
        points[step] = {
            "env_step": int(step),
            "return_mean": float(rep.return_mean),
            "per_regime": {int(k): float(v) for k, v in rep.return_per_regime.items()},
            "seen": float(rep.return_zero_shot_seen),
            "unseen": float(rep.return_zero_shot_unseen),
        }
        print(f"[eval_mamba_checkpoints] step={step:>8} return_mean={rep.return_mean:6.2f} "
              f"seen={rep.return_zero_shot_seen:6.2f} unseen={rep.return_zero_shot_unseen:6.2f}")

    ordered = [points[k] for k in sorted(points)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "cell": args.run_dir.parent.name if args.run_dir.parent else "",
        "run_dir": str(args.run_dir),
        "preset": args.preset,
        "episodes": int(args.episodes),
        "budget_target": float(args.budget_target),
        "points": ordered,
    }, indent=2), encoding="utf-8")
    print(f"[eval_mamba_checkpoints] wrote {args.out} ({len(ordered)} points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
