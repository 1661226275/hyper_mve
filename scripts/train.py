#!/usr/bin/env python
"""train.py — unified training entry point (phase-2 realignment).

One command trains ANY registered algorithm on ANY registered environment,
writes checkpoints + a rel-v1 eval report + unified TensorBoard logs under
``results/``, and appends a flat row to ``results/registry.jsonl``.

    # single run (GPU ids must be a subset of {3,4,5}):
    python scripts/train.py --algo mazero_mixed --env relation \\
        --seed 0 --total-env-steps 200000 --gpus 4

    # list what is registered (no training, no GPU):
    python scripts/train.py --list

    # sequential grid (one subprocess per row):
    python scripts/train.py --grid scripts/grids/rel_gate.yaml --gpus 3

Hard constraint (user-locked 2026-07-17): training runs ONLY on GPUs 3, 4, 5.
``--gpus`` is validated and exported as ``CUDA_VISIBLE_DEVICES`` **before**
torch is imported; anything else exits non-zero.

Outputs per run: ``results/<algo>/<env>/seed<k>/`` containing ``ckpt.pt``,
``eval_report.json``, ``meta.json``, ``tb/`` (TensorBoard, canonical
train_steps x-axis via :class:`hyper_mve.utils.unified_logger.UnifiedLogger`),
and — from realignment phase 7 — ``fidelity.json`` for model-based methods.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

ALLOWED_GPUS = (3, 4, 5)

# env id → V4Config preset name. mpe_tag / mpe_tag_fixed land in phase 3.
ENV_PRESETS = {
    "relation": "rel_duo",
    "relation_holdout": "rel_duo_holdout",
    "mpe_tag": "mpe_tag",
    "mpe_tag_fixed": "mpe_tag_fixed",
}

# How each algorithm's verbatim writer calls count steps (UnifiedLogger
# native_step_unit). The fork logs gradient steps; the ported baselines and
# their eval probes log cumulative env steps.
NATIVE_STEP_UNIT = {
    "mazero_mixed": "train",
    "mappo": "env",
    "mamba": "env",
    "happo": "env",         # phase 4
    "mbom": "env",          # phase 5
    "mbom_oracle": "env",   # phase 5
    "m3w_adapted": "env",   # phase 6
}


def _set_gpus(spec: str) -> None:
    """Validate --gpus ⊆ {3,4,5} and export CUDA_VISIBLE_DEVICES.

    MUST run before any torch import (torch reads the env var once).
    """
    try:
        ids = tuple(int(x) for x in spec.split(",") if x.strip() != "")
    except ValueError:
        raise SystemExit(f"--gpus {spec!r}: expected comma-separated integers")
    if not ids:
        raise SystemExit("--gpus: at least one GPU id required")
    bad = sorted(set(ids) - set(ALLOWED_GPUS))
    if bad:
        raise SystemExit(
            f"--gpus {spec!r}: GPU(s) {bad} not allowed — training is "
            f"restricted to GPUs {list(ALLOWED_GPUS)} (user-locked constraint)."
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in ids)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Unified train/eval entry (phase-2)")
    p.add_argument("--algo", default=None, help="registry key (see --list)")
    p.add_argument("--env", default=None, choices=tuple(ENV_PRESETS),
                   help="environment id")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-env-steps", type=int, default=200_000)
    p.add_argument("--lr", type=float, default=0.0,
                   help="0 = the runner's own default")
    p.add_argument("--episodes", type=int, default=16,
                   help="eval episodes per regime for the final EvalReport")
    p.add_argument("--gpus", default="3",
                   help=f"comma-separated GPU ids ⊆ {list(ALLOWED_GPUS)}")
    p.add_argument("--out", type=pathlib.Path,
                   default=pathlib.Path(REPO_ROOT) / "results",
                   help="results root (default: <repo>/results)")
    p.add_argument("--tb-dir", type=pathlib.Path, default=None,
                   help="TensorBoard dir (default: <run_dir>/tb)")
    p.add_argument("--ablation", default="none",
                   help="ablation arm name (phase 7; 'none' = main method)")
    p.add_argument("--list", action="store_true",
                   help="print registered algorithms/envs and exit")
    p.add_argument("--grid", type=pathlib.Path, default=None,
                   help="YAML grid: {algos:[...], envs:[...], seeds:[...], "
                        "total_env_steps: int, episodes: int} — sequential "
                        "subprocess per (algo, env, seed) row")
    return p.parse_args(argv)


def _print_list() -> int:
    from hyper_mve.comparison import REGISTRY  # torch-free (lazy string map)

    print("algorithms (hyper_mve.comparison.REGISTRY):")
    for key in sorted(REGISTRY):
        print(f"  {key:14s} -> {REGISTRY[key]}")
    print("environments:")
    for env_id, preset in ENV_PRESETS.items():
        note = "" if env_id.startswith("relation") else "  [lands in phase 3]"
        print(f"  {env_id:16s} -> preset {preset}{note}")
    print(f"allowed GPUs: {list(ALLOWED_GPUS)}")
    return 0


def build_cfg(env_id: str, ablation: str):
    from hyper_mve.utils.configs import V4Config

    cfg = V4Config.from_preset(ENV_PRESETS[env_id])
    if ablation and ablation != "none":
        from hyper_mve.ablation.arms import apply_arm  # lands in phase 7

        cfg = apply_arm(cfg, ablation)
    return cfg


def make_env_fn(cfg, env_id: str):
    if env_id.startswith("relation"):
        from hyper_mve.envs.adapters.pettingzoo_wrapper import (
            RelationCommonsPettingZooEnv,
        )

        return lambda: RelationCommonsPettingZooEnv(
            cfg.env, oracle_mode=False, eval_info_mode=False,
        )
    if env_id.startswith("mpe_tag"):
        from hyper_mve.envs.mpe_tag.env import MPETagRegimeEnv  # phase 3

        fixed = cfg.env.fixed_regime if env_id == "mpe_tag_fixed" else None
        return lambda: MPETagRegimeEnv(
            cfg.env, oracle_mode=False, eval_info_mode=False, fixed_regime=fixed,
        )
    raise SystemExit(f"unknown env id {env_id!r}")


def run_one(*, algo: str, env_id: str, seed: int, total_env_steps: int,
            lr: float, episodes: int, out_root: pathlib.Path,
            tb_dir, ablation: str) -> pathlib.Path:
    from hyper_mve.comparison import REGISTRY, create_runner
    from hyper_mve.utils.schemas.relation import get_regime_family
    from hyper_mve.utils.unified_logger import UnifiedLogger

    if algo not in REGISTRY:
        raise SystemExit(f"--algo {algo!r} not registered; valid: {sorted(REGISTRY)}")

    arm_suffix = "" if ablation in ("", "none") else f"_{ablation}"
    run_dir = out_root / f"{algo}{arm_suffix}" / env_id / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = pathlib.Path(tb_dir) if tb_dir else run_dir / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(env_id, ablation)
    env_fn = make_env_fn(cfg, env_id)
    logger = UnifiedLogger(
        tb_dir, algo=algo, env_id=env_id, seed=seed,
        native_step_unit=NATIVE_STEP_UNIT.get(algo, "env"),
    )

    t0 = time.time()
    runner = create_runner(cfg, algo)
    runner.train(
        cfg, env_fn,
        total_env_steps=int(total_env_steps), lr=float(lr), seed=int(seed),
        tensorboard_dir=str(tb_dir), unified_logger=logger,
    )
    train_walltime = time.time() - t0

    ckpt_path = run_dir / "ckpt.pt"
    runner.save_checkpoint(ckpt_path)

    grid = cfg.eval.eval_regime_grid
    if grid is None:
        grid = tuple(range(get_regime_family(cfg.env).size))
    report = runner.evaluate(env_fn, tuple(int(g) for g in grid), int(episodes))
    logger.log_eval_report(report)
    logger.close()

    (run_dir / "eval_report.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    meta = {
        "algo": algo, "env": env_id, "seed": int(seed),
        "ablation": ablation,
        "total_env_steps": int(total_env_steps), "lr": float(lr),
        "episodes_per_regime": int(episodes),
        "train_walltime_s": round(train_walltime, 1),
        "train_steps_logged": logger.train_steps,
        "env_steps_logged": logger.env_steps,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2),
                                       encoding="utf-8")
    row = dict(meta)
    row.update({
        "return_mean": report.return_mean,
        "return_zero_shot_seen": report.return_zero_shot_seen,
        "return_zero_shot_unseen": report.return_zero_shot_unseen,
        "return_zero_shot_gap": report.return_zero_shot_gap,
        "run_dir": str(run_dir),
    })
    registry_path = out_root / "registry.jsonl"
    with registry_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")

    print(f"[train.py] {algo}/{env_id}/seed{seed}: "
          f"return_mean={report.return_mean:.3f} "
          f"(seen={report.return_zero_shot_seen:.3f} "
          f"unseen={report.return_zero_shot_unseen:.3f}) -> {run_dir}")
    return run_dir


def _run_grid(grid_path: pathlib.Path, args) -> int:
    import yaml

    spec = yaml.safe_load(grid_path.read_text(encoding="utf-8"))
    algos = spec.get("algos") or [args.algo]
    envs = spec.get("envs") or [args.env]
    seeds = spec.get("seeds") or [args.seed]
    total = int(spec.get("total_env_steps", args.total_env_steps))
    episodes = int(spec.get("episodes", args.episodes))
    failures = 0
    for algo in algos:
        for env_id in envs:
            for seed in seeds:
                cmd = [
                    sys.executable, os.path.abspath(__file__),
                    "--algo", str(algo), "--env", str(env_id),
                    "--seed", str(seed),
                    "--total-env-steps", str(total),
                    "--episodes", str(episodes),
                    "--gpus", args.gpus,
                    "--out", str(args.out),
                ]
                print(f"[train.py grid] {' '.join(cmd[1:])}", flush=True)
                rc = subprocess.call(cmd)
                if rc != 0:
                    failures += 1
                    print(f"[train.py grid] row FAILED (rc={rc}) — continuing",
                          flush=True)
    return 1 if failures else 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list:
        return _print_list()
    _set_gpus(args.gpus)  # BEFORE any torch import
    if args.grid is not None:
        return _run_grid(args.grid, args)
    if not args.algo or not args.env:
        raise SystemExit("--algo and --env are required (or use --list/--grid)")
    run_one(
        algo=args.algo, env_id=args.env, seed=args.seed,
        total_env_steps=args.total_env_steps, lr=args.lr,
        episodes=args.episodes, out_root=args.out,
        tb_dir=args.tb_dir, ablation=args.ablation,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
