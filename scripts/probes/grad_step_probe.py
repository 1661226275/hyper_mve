#!/usr/bin/env python
"""Measure the TRUE ``optimizer.step()`` count per algorithm.

Why this exists
---------------
``progress/train_steps`` is not the same quantity across algorithms. For
``mazero_mixed`` and ``m3w_adapted`` it increments once per gradient step; for
``mamba``, ``happo`` and ``mbom`` it increments once per multi-epoch update
*round*, each of which contains many gradient steps:

* mamba  -- ``DreamerLearner.step`` runs ``MODEL_EPOCHS`` model updates, then
  ``train_agent``, which loops ``PPO_EPOCHS`` times over minibatches of 2000
  imagined states, updating actor AND critic per minibatch.
* happo  -- one HARL rollout drives ``ppo_epoch`` actor updates per agent plus
  ``critic_epoch`` critic updates.
* mbom   -- one epoch calls ``learn()`` per agent, each running
  ``a_update_times`` + ``v_update_times`` updates.

The v7 stage-1 cadence table therefore compared incommensurable units, in the
direction that made the model-based baselines look starved. This probe settles
it by measurement rather than by reading configs: it wraps
``torch.optim.Optimizer.step`` with a counter, so it needs no edit to any
vendored file, and counts whatever actually runs.

Usage (one algo per process -- CUDA_VISIBLE_DEVICES must be set before torch
is imported, which is why this is not a single multi-algo loop)::

    python scripts/probes/grad_step_probe.py --algo mamba --gpus 0 \
        --env-steps 10000 --out results/analysis/cadence/mamba.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algo", required=True)
    p.add_argument("--env", default="relation_coopmix")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env-steps", type=int, default=10_000)
    p.add_argument("--gpus", default="0")
    p.add_argument("--ablation", default="none")
    p.add_argument("--num-pmcts", type=int, default=1)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    import train as train_mod  # scripts/train.py

    train_mod._set_gpus(args.gpus)          # MUST precede the torch import

    import torch

    # ---- the counter -------------------------------------------------------
    # Patch every CONCRETE optimizer class that defines its own ``step``.
    # Wrapping only ``Optimizer.step`` counts nothing: Adam/SGD/... each define
    # ``step`` on the subclass, so dispatch never reaches the base method (the
    # first version of this probe reported 0 for happo because of exactly that).
    # Subclasses that do NOT define their own ``step`` inherit an already
    # wrapped one, so nothing is double counted.
    # Attribution matters as much as the total. MBOM's ``choose_action`` runs
    # imagined opponent-model fine-tuning as part of ACTING, so a raw count
    # mixes outer-loop policy training with inner-loop adaptation on a throwaway
    # copy -- two very different things for a "how much training" comparison.
    # Bucketing by optimizer instance (labelled with its parameter count)
    # separates them without needing to know each algorithm's internals.
    counts = {"n": 0}
    per_class: dict = {}
    per_instance: dict = {}
    patched = []

    def _make_counting(cls_name, orig):
        def wrapper(self, *a, **kw):
            counts["n"] += 1
            per_class[cls_name] = per_class.get(cls_name, 0) + 1
            key = id(self)
            rec = per_instance.get(key)
            if rec is None:
                n_par = sum(
                    int(p.numel())
                    for grp in getattr(self, "param_groups", [])
                    for p in grp.get("params", [])
                )
                rec = {"class": cls_name, "n_params": n_par, "steps": 0}
                per_instance[key] = rec
            rec["steps"] += 1
            return orig(self, *a, **kw)
        return wrapper

    for _name in dir(torch.optim):
        _obj = getattr(torch.optim, _name)
        if not isinstance(_obj, type) or not issubclass(_obj, torch.optim.Optimizer):
            continue
        if _obj is torch.optim.Optimizer or "step" not in _obj.__dict__:
            continue
        setattr(_obj, "step", _make_counting(_name, _obj.__dict__["step"]))
        patched.append(_name)

    from hyper_mve.comparison import create_runner
    from hyper_mve.utils.unified_logger import UnifiedLogger

    cfg = train_mod.build_cfg(args.env, args.ablation)
    env_fn = train_mod.make_env_fn(cfg, args.env)

    tmp = pathlib.Path(tempfile.mkdtemp(prefix=f"gradprobe_{args.algo}_"))
    logger = UnifiedLogger(
        tmp / "tb", algo=args.algo, env_id=args.env, seed=args.seed,
        native_step_unit=train_mod.NATIVE_STEP_UNIT.get(args.algo, "env"),
    )

    runner = create_runner(cfg, args.algo)
    t0 = time.time()
    runner.train(
        cfg, env_fn,
        total_env_steps=int(args.env_steps), lr=0.0, seed=int(args.seed),
        tensorboard_dir=str(tmp / "tb"), unified_logger=logger,
        ablation=args.ablation, num_pmcts=int(args.num_pmcts),
    )
    wall = time.time() - t0

    env_steps = int(getattr(logger, "env_steps", 0)) or int(args.env_steps)
    logged = int(getattr(logger, "train_steps", 0))
    result = {
        "algo": args.algo,
        "env": args.env,
        "env_steps_requested": int(args.env_steps),
        "env_steps_logged": env_steps,
        "train_steps_logged": logged,
        "true_grad_steps": counts["n"],
        "true_grad_per_1k_env": round(counts["n"] / max(env_steps, 1) * 1000, 1),
        "logged_per_1k_env": round(logged / max(env_steps, 1) * 1000, 1),
        "expansion_factor": round(counts["n"] / max(logged, 1), 1),
        "walltime_s": round(wall, 1),
        "per_optimizer_class": per_class,
        # One row per distinct optimizer object, biggest consumer first. A long
        # tail of short-lived instances with identical n_params is the signature
        # of inner-loop adaptation (a fresh optimizer per adaptation), not of
        # policy training (a handful of long-lived optimizers).
        "per_optimizer_instance": sorted(
            per_instance.values(), key=lambda r: -r["steps"])[:12],
        "distinct_optimizers": len(per_instance),
        "patched_classes": patched,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
