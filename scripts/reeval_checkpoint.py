#!/usr/bin/env python
"""reeval_checkpoint.py — re-evaluate a trained mazero_mixed checkpoint.

Runs the dual-mode protocol (MCTS planner + distilled prior) plus the action /
visit diagnostics on an existing ``ckpt.pt``, without retraining. This is the
discriminator for "did the policy collapse, or did the eval just never run the
search": if the planner return is far above the prior return, the search is
healthy and distillation failed; if both are equally degenerate, the collapse
reaches into the learned model itself.

    python scripts/reeval_checkpoint.py --gpus 3 \\
        --run-dir results/mazero_mixed/relation/seed0 \\
        --run-dir results/mazero_mixed_moe_router/relation/seed0

Writes ``eval_report_reeval.json`` + ``eval_diagnostics_reeval.json`` into each
run dir. It never overwrites the original artifacts and never appends to
``registry.jsonl``.

The ablation arm is read from each run's ``meta.json`` and applied to the
runner **before** the checkpoint is loaded. That matters: ``point_estimate_leaf``
changes no module construction, so its weights load cleanly into an unablated
model and then evaluate with the wrong leaf-value rule, with no error.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import ENV_PRESETS, _set_gpus, build_cfg, make_env_fn  # noqa: E402


def _arm_from_run_dir(run_dir: pathlib.Path, override: str | None) -> tuple[str, str, int]:
    """``(algo_arm, env_id, seed)`` from meta.json, cross-checked against the path.

    Layout is ``results/<algo>[_<arm>]/<env>/seed<k>/`` (scripts/train.py).
    """
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        if override is None:
            raise SystemExit(
                f"{run_dir}: no meta.json and no --arm given. Refusing to guess "
                "the ablation arm — loading a checkpoint under the wrong arm can "
                "succeed silently and evaluate the wrong model."
            )
        meta = {}
    else:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    arm = override or str(meta.get("ablation") or "none")
    env_id = str(meta.get("env") or run_dir.parent.name)
    seed = int(meta.get("seed", int(run_dir.name.replace("seed", "") or 0)))

    # The directory name encodes the arm; disagreement means the run dir and
    # its metadata describe different experiments.
    dir_arm = run_dir.parent.parent.name
    expected = "mazero_mixed" if arm == "none" else f"mazero_mixed_{arm}"
    if override is None and dir_arm != expected:
        raise SystemExit(
            f"{run_dir}: meta.json says ablation={arm!r} (-> {expected!r}) but the "
            f"directory says {dir_arm!r}. Pass --arm explicitly to override."
        )
    if env_id not in ENV_PRESETS:
        raise SystemExit(f"{run_dir}: unknown env id {env_id!r}")
    return arm, env_id, seed


def reeval_one(run_dir: pathlib.Path, *, episodes: int,
               planner_episodes: int | None, arm_override: str | None) -> dict:
    from hyper_mve.comparison import create_runner
    from hyper_mve.utils.schemas.relation import get_regime_family

    ckpt = run_dir / "ckpt.pt"
    if not ckpt.exists():
        raise SystemExit(f"{run_dir}: no ckpt.pt (run incomplete?)")

    arm, env_id, seed = _arm_from_run_dir(run_dir, arm_override)
    cfg = build_cfg(env_id, arm)
    env_fn = make_env_fn(cfg, env_id)

    runner = create_runner(cfg, "mazero_mixed")
    runner._ablation = arm          # must precede _lazy_model / load_checkpoint
    runner.load_checkpoint(ckpt)

    grid = cfg.eval.eval_regime_grid
    if grid is None:
        grid = tuple(range(get_regime_family(cfg.env).size))
    grid = tuple(int(g) for g in grid)

    report = runner.evaluate(
        env_fn, grid, int(episodes),
        planner_episodes=planner_episodes, seed=seed,
    )
    diagnostics = dict(getattr(runner, "_eval_diagnostics", {}))
    diagnostics["run_dir"] = str(run_dir)
    diagnostics["ablation"] = arm

    (run_dir / "eval_report_reeval.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    (run_dir / "eval_diagnostics_reeval.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8")
    return {"report": report, "diagnostics": diagnostics, "arm": arm,
            "env": env_id, "seed": seed}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", action="append", required=True,
                   help="results/<algo>[_<arm>]/<env>/seed<k>; repeatable")
    p.add_argument("--episodes", type=int, default=16,
                   help="episodes per regime for the prior pass (default 16)")
    p.add_argument("--planner-episodes", type=int, default=None,
                   help="episodes per regime for the search pass "
                        "(default: match --episodes, giving a paired comparison)")
    p.add_argument("--arm", default=None,
                   help="override the ablation arm from meta.json")
    p.add_argument("--gpus", default="3", help=f"subset of GPUs (default 3)")
    args = p.parse_args(argv)

    _set_gpus(args.gpus)   # before any torch import

    rows = []
    for rd in args.run_dir:
        run_dir = pathlib.Path(rd).resolve()
        print(f"[reeval] {run_dir}", flush=True)
        rows.append(reeval_one(
            run_dir, episodes=args.episodes,
            planner_episodes=args.planner_episodes, arm_override=args.arm,
        ))

    print()
    print(f"{'run':52s} {'prior':>9s} {'planner':>9s} {'gap':>8s} "
          f"{'a5_prior':>9s} {'a5_plan':>8s} {'belief':>7s}")
    for r in rows:
        rep, d = r["report"], r["diagnostics"]
        acc = d.get("regime_accuracy")
        print(
            f"{r['arm'] + '/' + r['env'] + '/seed' + str(r['seed']):52s} "
            f"{rep.direct_inference_return_mean:9.3f} "
            f"{rep.planner_full_return_mean:9.3f} "
            f"{rep.planner_prior_return_gap:8.3f} "
            f"{d['prior_action_fractions'][5]:9.3f} "
            f"{d['planner_action_fractions'][5]:8.3f} "
            f"{(f'{acc:.3f}' if acc is not None else '   n/a'):>7s}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
