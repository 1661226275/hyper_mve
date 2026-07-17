"""collect_final_evals.py — 2M-standard checkpoint evals with per-episode returns.

Standardizes the baseline comparison at one env-step budget (2M): evaluates each
method's checkpoint AT that budget with the full 30-episode-per-regime protocol
and records the PER-EPISODE returns (sum over agents — the same definition
``run_eval`` / the external runners aggregate into their report means), so the
distribution views (box plots) draw from exactly the episodes behind the means.

  * hyper           — ``step_20000.pt`` (20k grad × 100 = 2M env steps), planner
                      mode, replicating ``training.evaluation.run_eval``'s
                      fixed-seed episode construction (episodes overridden to 30;
                      the in-training eval used only 2/regime).
  * external_mappo  — ``ckpt_final.pt`` of the 2M run (measured budget endpoint).
  * external_mamba  — latest ``step_*.pt`` of the in-progress 2M run (labeled
                      by its true env step; provisional until the run finishes).

Run from the repo root (GPU via CUDA_VISIBLE_DEVICES):

    python hyper_mve/scripts/collect_final_evals.py \
        --out-dir runs/_analysis/rel_defense/final_evals_2m

Output: one JSON per (variant, cell):
    {variant, cell, preset, ckpt, env_step, episodes, per_regime_returns:{g:[...]},
     per_regime_mean:{g:...}, return_mean, seen, unseen}
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

GATE = ("rel_gate_duo", "rel_duo")
ZS = ("rel_zero_shot_duo", "rel_duo_holdout")

RUN_DIRS = {
    ("hyper", "rel_gate_duo"): "runs/suite/rel_gate_duo/hyper_seed0_row0_4c442f29",
    ("hyper", "rel_zero_shot_duo"): "runs/suite/rel_zero_shot_duo/hyper_seed0_row0_a26259a8",
    ("external_mappo", "rel_gate_duo"): "runs/suite/rel_gate_duo/external_mappo_seed0_row0_9acbbaf5",
    ("external_mappo", "rel_zero_shot_duo"): "runs/suite/rel_zero_shot_duo/external_mappo_seed0_row0_210f9489",
    ("external_mamba", "rel_gate_duo"): "runs/suite/rel_gate_duo/external_mamba_seed0_row0_be184580",
    ("external_mamba", "rel_zero_shot_duo"): "runs/suite/rel_zero_shot_duo/external_mamba_seed0_row0_31eb304a",
}

_STEP_RE = re.compile(r"step_(\d+)\.pt$")


def _latest_step_ckpt(run_dir: pathlib.Path) -> tuple[int, pathlib.Path] | None:
    best = None
    for p in run_dir.glob("step_*.pt"):
        m = _STEP_RE.search(p.name)
        if m:
            s = int(m.group(1))
            if best is None or s > best[0]:
                best = (s, p)
    return best


def _eval_hyper(cfg, ckpt_path: pathlib.Path, episodes: int) -> dict[int, list[float]]:
    """Planner-mode fixed-seed eval of a hyper checkpoint, per-episode capture.

    Replicates ``training.evaluation.run_eval``'s planner block (same seed
    scheme, planner CRN, tie-break RNG) with ``episodes`` per regime, returning
    {g: [episode return (sum over agents), ...]}.
    """
    import torch

    from hyper_mve.envs.relation_commons import RelationCommonsEnv
    from hyper_mve.models.hyper_muzero_model import HyperMuZeroModel
    from hyper_mve.planning.mve_planner import MVEPlanner
    from hyper_mve.schemas import get_regime_family
    from hyper_mve.training.evaluation import (
        _EVAL_ENV_SEED_BASE,
        _EVAL_PLANNER_SEED,
        _EVAL_TIEBREAK_SEED,
    )
    from hyper_mve.training.worker import Worker

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = HyperMuZeroModel(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    family = get_regime_family(cfg.env)
    regime_grid = tuple(range(family.size))

    envs, seeds, options = [], [], []
    for gi, gid in enumerate(regime_grid):
        for e in range(episodes):
            seed = _EVAL_ENV_SEED_BASE + gi * 100 + e
            envs.append(RelationCommonsEnv(cfg.env, seed=seed))
            seeds.append(seed)
            options.append({"g": int(gid)})

    planner = MVEPlanner(cfg)
    planner.crn_rng = np.random.default_rng(_EVAL_PLANNER_SEED)
    worker = Worker(cfg, model, envs=envs, planner=planner)
    outs = worker.collect_episodes(
        epsilon=0.0,
        use_planner=True,
        deterministic=True,
        reset_seeds=seeds,
        reset_options=options,
        tiebreak_rng=np.random.default_rng(_EVAL_TIEBREAK_SEED),
    )
    rets = np.stack([o.returns for o in outs], axis=0)   # (B, N)
    per_episode = rets.sum(axis=1)                       # sum over agents
    out: dict[int, list[float]] = {}
    for gi, gid in enumerate(regime_grid):
        sl = slice(gi * episodes, (gi + 1) * episodes)
        out[int(gid)] = [float(v) for v in per_episode[sl]]
    return out


def _eval_external(cfg, variant: str, ckpt_path: pathlib.Path,
                   episodes: int) -> dict[int, list[float]]:
    """Full evaluate() of an external runner checkpoint; per-episode capture
    via the runner's ``_eval_episode_returns`` stash (same episodes as means)."""
    from hyper_mve.baselines import create_baseline
    from hyper_mve.envs.adapters.pettingzoo_wrapper import RelationCommonsPettingZooEnv
    from hyper_mve.schemas import get_regime_family

    env_fn = lambda: RelationCommonsPettingZooEnv(
        cfg.env, oracle_mode=False, eval_info_mode=False,
    )
    runner = create_baseline(cfg, variant)
    if variant == "external_mappo":
        runner.evaluate(env_fn, regime_grid=(0,), episodes=0)   # builder path
    runner.load_checkpoint(str(ckpt_path))
    grid = tuple(range(get_regime_family(cfg.env).size))
    runner.evaluate(env_fn, regime_grid=grid, episodes=episodes)
    return {int(g): list(v) for g, v in runner._eval_episode_returns.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=pathlib.Path,
                    default=pathlib.Path("runs/_analysis/rel_defense/final_evals_2m"))
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--variants", nargs="*",
                    default=["hyper", "external_mappo", "external_mamba"])
    ap.add_argument("--budget", type=float, default=2_000_000.0)
    args = ap.parse_args(argv)

    from dataclasses import replace

    from hyper_mve.configs import V4Config

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for cell, preset in (GATE, ZS):
        cfg = V4Config.from_preset(preset)
        cfg = replace(cfg, eval=replace(
            cfg.eval, eval_episodes_planner=args.episodes, eval_episodes_prior=0,
        ))
        for variant in args.variants:
            run_dir = pathlib.Path(RUN_DIRS[(variant, cell)])
            if variant == "hyper":
                ckpt = run_dir / "step_20000.pt"       # 20k grad × 100 = 2M env
                env_step = int(args.budget)
            elif variant == "external_mappo":
                ckpt = run_dir / "ckpt_final.pt"       # measured 2M endpoint
                env_step = int(args.budget)
            else:                                       # external_mamba: latest
                latest = _latest_step_ckpt(run_dir)
                if latest is None:
                    print(f"[collect_final_evals] {variant}/{cell}: no step ckpt yet — skip")
                    continue
                env_step, ckpt = latest
            if not ckpt.exists():
                print(f"[collect_final_evals] missing {ckpt} — skip")
                continue

            if variant == "hyper":
                per_ep = _eval_hyper(cfg, ckpt, args.episodes)
            else:
                per_ep = _eval_external(cfg, variant, ckpt, args.episodes)

            per_mean = {g: float(np.mean(v)) for g, v in per_ep.items()}
            from hyper_mve.baselines.external.base import split_seen_unseen_regimes
            seen, unseen = split_seen_unseen_regimes(cfg, per_mean)
            body = {
                "variant": variant, "cell": cell, "preset": preset,
                "ckpt": str(ckpt), "env_step": int(env_step),
                "episodes": int(args.episodes),
                "per_regime_returns": {str(g): v for g, v in per_ep.items()},
                "per_regime_mean": {str(g): per_mean[g] for g in per_mean},
                "return_mean": float(np.mean([v for vs in per_ep.values() for v in vs])),
                "seen": float(seen), "unseen": float(unseen),
            }
            out = args.out_dir / f"{variant}_{cell}.json"
            out.write_text(json.dumps(body, indent=2), encoding="utf-8")
            print(f"[collect_final_evals] {variant}/{cell} @env_step={env_step}: "
                  f"return_mean={body['return_mean']:.2f} "
                  f"per_regime={{{', '.join(f'{g}:{per_mean[g]:.1f}' for g in sorted(per_mean))}}} "
                  f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
