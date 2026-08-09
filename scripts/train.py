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
import hashlib
import inspect
import json
import os
import pathlib
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# User-locked allocation. Was {3,4,5} from the 2026-07-17 realignment; GPU 6
# was secured on 2026-07-18 and authorised for the formal experiment grid;
# GPUs 7 and 8 were secured on 2026-07-27; GPUs 0-2 on 2026-07-28.
# GPU 9 remains policy-forbidden (it still carries another user's work).
ALLOWED_GPUS = (0, 1, 2, 3, 4, 5, 6, 7, 8)

# env id → V4Config preset name. mpe_tag / mpe_tag_fixed land in phase 3.
ENV_PRESETS = {
    "relation": "rel_duo",
    "relation_holdout": "rel_duo_holdout",
    # v6: the environment in which the hidden regime is actually worth
    # inferring (VoI 3.99 vs 0.00 on rel_duo). `relation` stays the frozen
    # control — see results/analysis/regime_knowledge_ceiling.md.
    "relation_recip": "rel_recip",
    "relation_recip_holdout": "rel_recip_holdout",
    # v6 scope change: same physics as `relation_recip`, on the `g2cm` family,
    # which carries no purely adversarial regime. `relation_recip` stays frozen
    # because runs are archived against it — see results/analysis/g1_removal.md.
    "relation_coopmix": "rel_coopmix",
    "relation_coopmix_holdout": "rel_coopmix_holdout",
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
    "mamba_pm": "env",      # 2026-07-27 parameter-matched capacity variants
    "happo_pm": "env",
    "mbom_pm": "env",
}


def _set_gpus(spec: str) -> None:
    """Validate --gpus ⊆ ALLOWED_GPUS and export CUDA_VISIBLE_DEVICES.

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
    p.add_argument("--num-pmcts", type=int, default=1,
                   help="mazero_mixed only: MCTS trees searched in parallel = "
                        "the search-time inference batch size. Does NOT change "
                        "the env-steps-per-gradient-step ratio (core/train.py "
                        "paces gradient steps off transitions_collected); it "
                        "trades behaviour-policy freshness for throughput. "
                        "CUDA search wall-time is ~flat in this value.")
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
            tb_dir, ablation: str, num_pmcts: int = 1) -> pathlib.Path:
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
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    runner = create_runner(cfg, algo)
    runner.train(
        cfg, env_fn,
        total_env_steps=int(total_env_steps), lr=float(lr), seed=int(seed),
        tensorboard_dir=str(tb_dir), unified_logger=logger,
        ablation=ablation, num_pmcts=int(num_pmcts),
    )
    train_walltime = time.time() - t0

    ckpt_path = run_dir / "ckpt.pt"
    runner.save_checkpoint(ckpt_path)

    grid = cfg.eval.eval_regime_grid
    if grid is None:
        grid = tuple(range(get_regime_family(cfg.env).size))
    grid = tuple(int(g) for g in grid)
    # Identity for the report. Both fields used to be hardcoded inside the
    # runner ("seed": 0, config_hash all-zeros), so every seed's report claimed
    # seed 0 and no report could be traced back to its config.
    #
    # The env PHYSICS must be in here. It was not, so a run under the v6 reward
    # or regrowth law hashed identically to an archived v5 run at the same
    # (algo, env, ablation, steps, lr) — two incomparable experiments sharing an
    # identity, with nothing in the report to tell them apart. Only fields that
    # change the MDP belong; presentation-only settings would churn the hash and
    # break comparability with archived runs for no reason.
    physics = {
        f: getattr(cfg.env, f) for f in (
            "N", "L", "K", "T_max", "A", "Q_max", "alpha", "epsilon_move",
            "regrowth_law", "reward_coupling", "reciprocity_lambda",
            "relation_family", "relation_intensity", "regime_prior",
            "regime_switch_prob", "regime_kernel", "train_regime_ids",
            "env_kind", "fixed_regime",
        )
    }
    config_hash = hashlib.sha1(
        json.dumps({"algo": algo, "env": env_id, "ablation": ablation,
                    "total_env_steps": int(total_env_steps),
                    "lr": float(lr), "env_physics": physics},
                   sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    # Baseline runners keep the 3-arg signature; only mazero_mixed accepts the
    # identity kwargs. Dispatch on the signature rather than catching TypeError,
    # which would also swallow a TypeError raised inside evaluate() itself.
    eval_params = inspect.signature(runner.evaluate).parameters
    eval_kwargs = {k: v for k, v in
                   (("seed", int(seed)), ("config_hash", config_hash))
                   if k in eval_params}
    report = runner.evaluate(env_fn, grid, int(episodes), **eval_kwargs)
    logger.log_eval_report(report)

    # Action/visit diagnostics (evaldiag-v1). Only mazero_mixed produces these;
    # they are what makes a collapsed policy visible without reading weights.
    diagnostics = getattr(runner, "_eval_diagnostics", None)
    if diagnostics:
        (run_dir / "eval_diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2), encoding="utf-8")
        for mode in ("prior", "planner"):
            for a, frac in enumerate(diagnostics[f"{mode}_action_fractions"]):
                logger.log_scalar(f"eval/action_frac_{mode}_a{a}", float(frac),
                                  train_step=logger.train_steps)
        for key in ("planner_prior_return_gap", "prior_logit_margin",
                    "planner_visit_entropy"):
            logger.log_scalar(f"eval/{key}", float(diagnostics[key]),
                              train_step=logger.train_steps)

    # metric ① — world-model fidelity (fidelity-v1, SEPARATE artifact from
    # the rel-v1 EvalReport). None for model-free / supplied-model runners.
    from hyper_mve.utils.eval.fidelity import compute_fidelity_report

    fidelity = compute_fidelity_report(
        runner, env_fn, grid, episodes=max(2, int(episodes)), seed=1234)
    if fidelity is not None:
        (run_dir / "fidelity.json").write_text(
            json.dumps(fidelity, indent=2), encoding="utf-8")
        logger.log_scalar("fidelity/reward_mae",
                          float(fidelity["reward_mae"]),
                          train_step=logger.train_steps)
        for g, v in fidelity["reward_mae_per_regime"].items():
            logger.log_scalar(f"fidelity/reward_mae_regime_{g}", float(v),
                              train_step=logger.train_steps)
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
    # The row is a SUPERSET: the flat realignment keys (above) plus the
    # identity/lifecycle keys that utils/analysis/registry_io.py requires —
    # it drops any row without `run_id` and keeps only status=="completed",
    # so omitting these makes `analyze_results.py --compare/--disclose`
    # silently report zero runs.
    variant = f"{algo}{arm_suffix}"
    run_id = f"{variant}/{env_id}/seed{seed}"
    # config_hash computed above, before evaluate(), so the report and the
    # registry row carry the same value.
    row = dict(meta)
    row.update({
        "run_id": run_id,
        "variant": variant,
        "config_hash": config_hash,
        "ablation_cell": None if ablation in ("", "none") else ablation,
        "status": "completed",
        "started_at_iso8601": started_at,
        "completed_at_iso8601": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tensorboard_dir": str(tb_dir),
        "checkpoint_path": str(ckpt_path),
        "eval_report_path": str(run_dir / "eval_report.json"),
        "walltime_seconds": round(train_walltime, 1),
        "return_mean": report.return_mean,
        "return_zero_shot_seen": report.return_zero_shot_seen,
        "return_zero_shot_unseen": report.return_zero_shot_unseen,
        "return_zero_shot_gap": report.return_zero_shot_gap,
        "fidelity_reward_mae": (float(fidelity["reward_mae"])
                                if fidelity is not None else None),
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
    # ablation arms are a full grid dimension: without this the subprocess
    # inherits the default 'none' and every arm row retrains the main method
    # into the SAME results/<algo>/ dir, overwriting the previous row.
    ablations = spec.get("ablations") or [args.ablation]
    total = int(spec.get("total_env_steps", args.total_env_steps))
    episodes = int(spec.get("episodes", args.episodes))
    lr = float(spec.get("lr", args.lr))
    num_pmcts = int(spec.get("num_pmcts", args.num_pmcts))
    failures = 0
    for algo in algos:
        for env_id in envs:
            for ablation in ablations:
                arm = str(ablation or "none")
                if arm != "none" and algo != "mazero_mixed":
                    print(f"[train.py grid] SKIP {algo}/{arm}: ablation arms "
                          f"are defined on mazero_mixed only", flush=True)
                    continue
                for seed in seeds:
                    # resume: a multi-day grid that dies on row k should not
                    # redo rows 0..k-1. A row counts as done only when its
                    # eval_report.json exists (run_one writes it after
                    # training AND evaluation succeed).
                    suffix = "" if arm == "none" else f"_{arm}"
                    done_marker = (pathlib.Path(args.out) / f"{algo}{suffix}"
                                   / str(env_id) / f"seed{seed}"
                                   / "eval_report.json")
                    if done_marker.exists():
                        print(f"[train.py grid] SKIP {algo}{suffix}/{env_id}/"
                              f"seed{seed}: already complete", flush=True)
                        continue
                    cmd = [
                        sys.executable, os.path.abspath(__file__),
                        "--algo", str(algo), "--env", str(env_id),
                        "--seed", str(seed),
                        "--total-env-steps", str(total),
                        "--episodes", str(episodes),
                        "--lr", str(lr),
                        "--ablation", arm,
                        "--num-pmcts", str(num_pmcts),
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
        num_pmcts=args.num_pmcts,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
