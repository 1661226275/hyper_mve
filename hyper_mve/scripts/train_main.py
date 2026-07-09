"""train_main.py — unified v5 training entry (Pkg-09; base Pkg-05 spec 08 §6).

One entry point; variants are expressed via cfg overrides. Run from the repo
root so ``hyper_mve`` imports.

    python hyper_mve/scripts/train_main.py --preset rel_duo --variant hyper --max_steps 1000
    python hyper_mve/scripts/train_main.py --preset rel_duo --override "train.lr=3e-4" --seed 0

CLI choices (pkg-07 spec 01 §2.1, v5) — 13 entries:

    Curriculum-overrides (3): hyper / oracle_only / infer_only
    Internal baselines (4):   baseline_input_wide / baseline_input_deep /
                              baseline_ma_muzero / no_belief
    External baselines (6):   external_mappo / external_qmix / external_ma_muzero_gh /
                              external_mamba / external_marie / external_ga

Internal baselines and external runners are constructed via
``hyper_mve.baselines.create_baseline``; this entry point can train ``hyper``
and the curriculum-overrides directly via the MuZeroTrainer path. For the
internal baselines + external runners (which carry their own
``.train()``/``.evaluate()``), the recommended driver is the pkg-08 sweep
harness; this entry point delegates with a clear message.

Stub variants (per pkg-07 design D9 + spec 06 §4.6) raise on
construction; the CLI surfaces this via ``_DEFERRED_VARIANTS``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from collections import deque
from dataclasses import replace

# Allow `from hyper_mve...` when launched as `python hyper_mve/scripts/train_main.py`
# from the repo root (the package is not pip-installed; mirrors hyper_mve/scripts/*).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Quiet TensorFlow's CUDA-plugin re-registration noise: when --log_dir is set,
# tensorboard pulls TensorFlow into the env, and TF/PyTorch sharing the same CUDA
# libs print harmless "cuFFT/cuDNN/cuBLAS factory already registered" / oneDNN lines.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import torch

from hyper_mve.baselines import CLI_CHOICES, REGISTRY, cli_to_factory_arg, create_baseline
from hyper_mve.configs import V4Config
from hyper_mve.models import HyperMuZeroModel, Projector
from hyper_mve.envs.relation_commons import RelationCommonsEnv
from hyper_mve.training import EpisodeReplayBuffer, MuZeroTrainer, Worker, run_eval

_SUB_CONFIGS = ("env", "model", "train", "mup", "eval", "legacy")

# pkg-07 spec 06 §4.6 + design D9: 3 stub-CLIs that raise on construction.
# Other variants (baseline_*, no_belief, rewardhead_explicit_type, external_mappo,
# external_qmix, external_ma_muzero_gh) are reachable but route through the
# pkg-08 sweep harness — train_main.py prints a delegation message rather
# than driving them through MuZeroTrainer.
_STUB_VARIANTS: dict[str, str] = {
    "external_mamba":  "Tier-2 stub (pkg-07 design D8); IS_SOURCED=False — see spec 06 §4.6.",
    "external_marie":  "permanent stub (pkg-07 design D9); see spec 06 §5.",
    "external_ga":     "permanent stub (pkg-07 design D9); see spec 06 §5.",
}

# Variants whose ``.train()`` / ``.evaluate()`` is owned by the runner itself
# (pkg-08 spec 05 sweep harness drives them); train_main.py prints a delegation
# message and exits gracefully.
_RUNNER_OWNED_VARIANTS: frozenset[str] = frozenset({
    "baseline_input_wide", "baseline_input_deep", "baseline_ma_muzero",
    "no_belief",
    "external_mappo", "external_qmix", "external_ma_muzero_gh",
})

# Curriculum-override variants (Pkg-05 spec 04 K1): need TrainConfig overrides
# only spec 08 fully wires; train_main.py supports `hyper` natively.
_CURRICULUM_OVERRIDE_DEFERRED: dict[str, str] = {
    "oracle_only": "needs curriculum_stage_*_end_frac=1.0 (TrainConfig forbids); pkg-08.",
    "infer_only":  "needs curriculum_stage_*_end_frac=0.0 (TrainConfig forbids); pkg-08.",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Hyper-MuZero v5 unified trainer")
    p.add_argument("--preset", default="rel_duo", choices=(
        "rel_duo", "rel_duo_holdout",
        "easy", "medium", "hard", "duo", "duo_basegen",
        "duo_film_lora", "duo_film_lora_fc2", "duo_base_lora",
        "medium_film_lora", "medium_film_lora_fc2", "medium_base_lora",
    ))
    p.add_argument("--variant", default="hyper", choices=CLI_CHOICES)
    p.add_argument("--max_steps", type=int, default=None, help="override train.max_train_steps")
    p.add_argument("--override", action="append", default=[], help='"section.field=value" (repeatable)')
    p.add_argument("--resume_from", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_dir", default=None)
    p.add_argument("--ckpt_dir", default="checkpoints/v4")
    # pkg-08 spec 06 §4 + design D10 — coordinate-descent agent ordering toggle.
    # The legacy ``--use_coord_desc`` flag is preserved as a deprecation alias.
    p.add_argument(
        "--randomize_order", dest="randomize_order",
        action=argparse.BooleanOptionalAction, default=None,
        help="MVE planner coordinate-descent agent ordering on/off (pkg-08 spec 06 §4).",
    )
    p.add_argument(
        "--use_coord_desc", dest="use_coord_desc",
        action=argparse.BooleanOptionalAction, default=None,
        help="DEPRECATED alias for --randomize_order; emits DeprecationWarning.",
    )
    p.add_argument(
        "--no_collect_planner", action="store_true",
        help="disable the MVE planner during training collection (debug only; default ON). "
             "With the planner OFF, pi_mve = the model's own prior → policy target = itself → "
             "uniform is a zero-gradient fixed point and the policy never improves.",
    )
    return p.parse_args(argv)


def _parse_value(raw: str):
    """JSON literal if possible (ints/floats/bools/lists), else raw string."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def apply_overrides(cfg: V4Config, overrides: list[str]) -> V4Config:
    """Apply ``--override "section.field=value"`` items via dataclasses.replace."""
    for item in overrides:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise ValueError(f"--override must be 'section.field=value', got: {item!r}")
        path, raw = item.split("=", 1)
        section, field = path.split(".", 1)
        if section not in _SUB_CONFIGS:
            raise ValueError(f"unknown config section {section!r} (valid: {_SUB_CONFIGS})")
        value = _parse_value(raw)
        sub = getattr(cfg, section)
        cfg = replace(cfg, **{section: replace(sub, **{field: value})})
    return cfg


def _resolve_randomize_order(args) -> bool | None:
    """Resolve --randomize_order vs --use_coord_desc deprecation alias."""
    if args.use_coord_desc is not None:
        warnings.warn(
            "--use_coord_desc is deprecated (pkg-08 spec 06 §4); use "
            "--randomize_order / --no-randomize_order instead.",
            DeprecationWarning, stacklevel=2,
        )
        if args.randomize_order is not None and args.randomize_order != args.use_coord_desc:
            raise ValueError(
                "--randomize_order and --use_coord_desc disagree; pass only one."
            )
        return bool(args.use_coord_desc)
    return args.randomize_order


def apply_variant(cfg: V4Config, variant: str) -> V4Config:
    """Resolve --variant against the 13-CLI surface (pkg-07 spec 01 §2.1, v5).

    Raises NotImplementedError for stubs (pkg-07 design D9 + spec 06 §4.6)
    and for curriculum-override variants that need TrainConfig invariants
    only pkg-08 fully wires.

    Returns the (possibly mutated) cfg for the directly-supported branches:
    ``hyper`` and the runner-owned variants (which delegate to the
    pkg-08 sweep harness — train_main.py exits gracefully).
    """
    if variant == "hyper":
        return cfg
    if variant in _STUB_VARIANTS:
        raise NotImplementedError(
            f"--variant {variant!r}: {_STUB_VARIANTS[variant]}"
        )
    if variant in _CURRICULUM_OVERRIDE_DEFERRED:
        raise NotImplementedError(
            f"--variant {variant!r}: {_CURRICULUM_OVERRIDE_DEFERRED[variant]}"
        )
    if variant in _RUNNER_OWNED_VARIANTS:
        # Sanity-check: factory string is reachable.
        _ = cli_to_factory_arg(variant)   # raises ValueError if unknown
        return cfg
    raise ValueError(f"unknown --variant {variant!r}")


def build_model(cfg: V4Config, variant: str):
    """Construct the model for variants train_main.py drives directly.

    ``hyper`` → ``HyperMuZeroModel(cfg)`` via the existing MuZeroTrainer path.
    All other variants are runner-owned (or stubs / deferred) — see
    :func:`apply_variant`.
    """
    if variant == "hyper":
        return HyperMuZeroModel(cfg)
    raise NotImplementedError(
        f"--variant {variant!r} is not driven by train_main.py. "
        "Internal baselines + external runners are owned by the pkg-08 "
        "sweep harness (pkg-07 spec 08 §7.1)."
    )


def _epsilon(cfg: V4Config, step: int) -> float:
    t = cfg.train
    frac = min(1.0, step / max(1, t.epsilon_decay_steps))
    return float(t.epsilon_min + (t.epsilon_init - t.epsilon_min) * (1.0 - frac))


def run_training(
    cfg: V4Config,
    model,
    *,
    seed: int = 0,
    ckpt_dir: str = "checkpoints/v4",
    log_dir: str | None = None,
    resume_from: str | None = None,
    collect_planner: bool = True,
    preset: str = "?",
    variant: str = "?",
) -> int:
    """Shared MuZeroTrainer training loop for any v5 6-API model.

    Drives ``hyper`` AND the 4 internal ``BaselineModel`` variants (input_wide /
    input_deep / ma_muzero / no_belief) — they all
    expose the same 6-API and the same shared backbones, so the unroll/loss/eval
    machinery is identical. Extracted from :func:`main` so the pkg-08 sweep
    worker can train the runner-owned *internal* baselines in-process and then
    evaluate the *same trained object* (closes the "eval on a fresh model" gap;
    pkg-07 spec 08 §7.1). External runners own their own ``.train()`` and never
    use this path.

    Returns the final ``global_step``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    projector = Projector(cfg.model.latent_dim, cfg.model.proj_dim)
    # [v4-opt 2026-06] vectorized collection: episodes_per_iter envs stepped in
    # lockstep through the (already batched) MVE planner — the old B=1 path was
    # kernel-launch bound (~0.08 train-steps/s on GPU).
    n_envs = max(1, cfg.train.episodes_per_iter)
    envs = [RelationCommonsEnv(cfg.env, seed=seed * 1000 + i) for i in range(n_envs)]

    trainer = MuZeroTrainer(cfg, model, projector=projector, device=device)
    worker = Worker(cfg, model, envs=envs)
    buffer = EpisodeReplayBuffer(cfg)

    start_step = trainer.load_checkpoint(resume_from) if resume_from else 0

    writer = None
    if log_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir)
        except ImportError:
            print("[train_main] tensorboard unavailable; logging to stdout only.")

    os.makedirs(ckpt_dir, exist_ok=True)
    max_steps = cfg.train.max_train_steps
    global_step = start_step

    print(f"[train_main] preset={preset} variant={variant} device={device} "
          f"max_steps={max_steps} | warming up buffer to "
          f"min_buffer_size={cfg.train.min_buffer_size} episodes "
          f"(T_max={cfg.env.T_max}; silent collection, can take a while)...", flush=True)

    # Warm up the buffer (fast batched collection without the planner). The stored
    # pi_mve is the model's own prior — a self-distillation target — so the episodes
    # are flagged planner_on=False and the policy loss masks them [v4-opt 2026-06].
    log_every = max(1, cfg.train.min_buffer_size // 20)
    last_logged = 0
    while len(buffer) < cfg.train.min_buffer_size:
        for res in worker.collect_episodes(epsilon=1.0, use_planner=False):
            buffer.store_episode(res.records,
                                 planner_on=False, collected_at_step=start_step)
        if len(buffer) - last_logged >= log_every:
            last_logged = len(buffer)
            print(f"[warmup] buffer {len(buffer)}/{cfg.train.min_buffer_size}", flush=True)

    # Training collection MUST carry the planning signal: with use_planner=False the
    # buffer's pi_mve = the model's own prior, so the policy target is itself and the
    # uniform distribution is a zero-gradient fixed point (policy never learns). The
    # MVE plan is an *improved* target the prediction net can learn toward.
    print(f"[train_main] warmup done; starting training loop "
          f"(collection planner={'ON' if collect_planner else 'OFF'}; "
          f"n_envs={n_envs}; eval every {cfg.eval.evaluate_freq} steps).", flush=True)

    # --- periodic deterministic evaluation (dual mode + regime grid, v5) ---
    best_eval_return = -float("inf")

    def _run_and_log_eval(step: int) -> None:
        nonlocal best_eval_return
        metrics = run_eval(model, cfg, global_step=step)
        planner_total = metrics.get("planner/return_total", float("nan"))
        prior_total = metrics.get("prior/return_total", float("nan"))
        gap = metrics.get("planner_prior_gap", float("nan"))
        print(f"[eval] step {step}  planner_total={planner_total:.3f} "
              f"prior_total={prior_total:.3f} gap={gap:.3f}", flush=True)
        if writer is not None:
            for k, v in metrics.items():
                if isinstance(v, float) and math.isnan(v):
                    continue
                writer.add_scalar(f"eval/{k}", v, step)
        ref = planner_total if not math.isnan(planner_total) else prior_total
        if not math.isnan(ref) and ref > best_eval_return:
            best_eval_return = ref
            trainer.save_checkpoint(os.path.join(ckpt_dir, "best.pt"))

    if cfg.eval.evaluate_freq > 0:
        _run_and_log_eval(global_step)   # post-warmup baseline

    collect_window: deque = deque(maxlen=32)   # recent CollectResult for collect/* stats
    collect_sec = train_sec = env_steps_per_sec = 0.0

    while global_step < max_steps:
        eps = _epsilon(cfg, global_step)
        t0 = time.perf_counter()
        results = worker.collect_episodes(epsilon=eps, use_planner=collect_planner)
        for res in results:
            buffer.store_episode(res.records,
                                 planner_on=collect_planner, collected_at_step=global_step)
            collect_window.append(res)
        collect_sec = time.perf_counter() - t0
        env_steps_per_sec = sum(len(r.records) for r in results) / max(collect_sec, 1e-9)

        t1 = time.perf_counter()
        for _ in range(cfg.train.train_steps_per_iter):
            batch = buffer.sample_batch(cfg.train.batch_size, cfg.train.unroll_K)
            losses = trainer.train_step(batch, global_step)
            global_step += 1

            if global_step % 100 == 0:
                rets = np.stack([r.returns for r in collect_window], axis=0)   # (n, N)
                ret_total = float(rets.sum(axis=1).mean())
                msg = (
                    f"step {global_step}  total={losses['total']:.4f} "
                    f"main={losses['main']:.4f} | "
                    f"pi={losses['policy']:.4f} v={losses['value']:.4f} "
                    f"r={losses['reward']:.4f} cons={losses['consist']:.4f} "
                    f"belief={losses['belief']:.4f} | "
                    f"H_pi_mve={losses['diag_pi_mve_entropy']:.3f} "
                    f"cos_pred_pair={losses['diag_cos_pred_pair']:.3f} "
                    f"ret={ret_total:.2f} eps={eps:.2f} "
                    f"lr={losses['lr']:.2e}"
                )
                print(msg, flush=True)
                if writer is not None:
                    for k, v in losses.items():
                        # NaN tags (e.g. cos_*_same with N=2: no same-type pair) are skipped.
                        if isinstance(v, float) and math.isnan(v):
                            continue
                        # route: diag_* -> diag/, *_raw -> loss_raw/, else loss/
                        if k.startswith("diag_"):
                            tag = f"diag/{k[len('diag_'):]}"
                        elif k.endswith("_raw"):
                            tag = f"loss_raw/{k[:-len('_raw')]}"
                        else:
                            tag = f"loss/{k}"
                        writer.add_scalar(tag, v, global_step)
                    # --- collect/* observability [v4-opt 2026-06] ---
                    writer.add_scalar("collect/return_total", ret_total, global_step)
                    for agent_i in range(rets.shape[1]):
                        writer.add_scalar(f"collect/return_agent{agent_i}",
                                          float(rets[:, agent_i].mean()), global_step)
                    writer.add_scalar("collect/epsilon", eps, global_step)
                    writer.add_scalar("collect/H_pi_mve_fresh",
                                      float(np.mean([r.pi_entropy_mean for r in collect_window])),
                                      global_step)
                    qstds = [r.q_std_mean for r in collect_window if not math.isnan(r.q_std_mean)]
                    if qstds:
                        writer.add_scalar("collect/q_std", float(np.mean(qstds)), global_step)
                        writer.add_scalar("collect/q_gap", float(np.mean(
                            [r.q_gap_mean for r in collect_window if not math.isnan(r.q_gap_mean)]
                        )), global_step)
                        writer.add_scalar("collect/uniform_frac", float(np.mean(
                            [r.uniform_frac for r in collect_window if not math.isnan(r.uniform_frac)]
                        )), global_step)
                    # --- perf/* throughput probes [v4-opt 2026-06] ---
                    writer.add_scalar("perf/collect_sec_per_iter", collect_sec, global_step)
                    writer.add_scalar("perf/train_sec_per_iter", train_sec, global_step)
                    writer.add_scalar("perf/env_steps_per_sec", env_steps_per_sec, global_step)

            if cfg.eval.evaluate_freq > 0 and global_step % cfg.eval.evaluate_freq == 0:
                _run_and_log_eval(global_step)
            if global_step % 10_000 == 0:
                trainer.save_checkpoint(os.path.join(ckpt_dir, f"step_{global_step}.pt"))
            if global_step >= max_steps:
                break
        train_sec = time.perf_counter() - t1

    trainer.save_checkpoint(os.path.join(ckpt_dir, f"step_{global_step}.pt"))
    if writer is not None:
        writer.close()
    return global_step


def main(argv=None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = V4Config.from_preset(args.preset)
    cfg = apply_overrides(cfg, args.override)
    cfg = apply_variant(cfg, args.variant)
    if args.max_steps is not None:
        cfg = replace(cfg, train=replace(cfg.train, max_train_steps=args.max_steps))

    randomize_order = _resolve_randomize_order(args)
    if randomize_order is not None:
        cfg = replace(cfg, train=replace(cfg.train, randomize_order=bool(randomize_order)))

    # Runner-owned variants: train_main.py is not the driver. Print a
    # delegation message and exit successfully (pkg-08 spec 05 owns the
    # sweep harness call site — _sweep_worker.py trains them in-process via
    # run_training (internal) or runner.train (external)).
    if args.variant in _RUNNER_OWNED_VARIANTS:
        factory_arg = cli_to_factory_arg(args.variant)
        print(
            f"[train_main] --variant {args.variant!r} is runner-owned. "
            "Construct via hyper_mve.baselines.create_baseline + the "
            "pkg-08 sweep harness:\n"
            f"    >>> from hyper_mve.baselines import create_baseline\n"
            f"    >>> runner = create_baseline(cfg, {factory_arg!r})\n"
            f"    >>> runner.train(cfg, env_fn, total_env_steps=...)\n"
            "(pkg-07 spec 08 §7.1 — train_main.py CLI surface preserved; "
            "no MuZeroTrainer path for this variant.)",
            flush=True,
        )
        return

    model = build_model(cfg, args.variant)
    run_training(
        cfg, model,
        seed=args.seed,
        ckpt_dir=args.ckpt_dir,
        log_dir=args.log_dir,
        resume_from=args.resume_from,
        collect_planner=not args.no_collect_planner,
        preset=args.preset,
        variant=args.variant,
    )


if __name__ == "__main__":
    main()
