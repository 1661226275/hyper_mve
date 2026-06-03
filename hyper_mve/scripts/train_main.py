"""train_main.py — unified v4 training entry (Pkg-05 spec 08 §6, Q4).

One entry point; variants are expressed via cfg overrides (Oracle/Infer are no longer
model classes). Run from the repo root (D:\\RL\\hyper_mve) so ``hyper_mve`` imports.

    python hyper_mve/scripts/train_main.py --preset medium --variant hyper --max_steps 1000
    python hyper_mve/scripts/train_main.py --preset medium --override "train.lr=3e-4" \
        --override "env.N=8" --seed 0

K1 (spec 04): oracle_only / infer_only need curriculum fracs of 1.0 / 0.0 which
TrainConfig rejects (0<s1<s2<1); those variants are wired in Pkg-08, not here.
baseline_* / no_belief need the Pkg-06 model factory. Pkg-05 fully supports --variant hyper.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace

# Allow `from hyper_mve...` when launched as `python hyper_mve/scripts/train_main.py`
# from the repo root (the package is not pip-installed; mirrors hyper_mve/scripts/*).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

from hyper_mve.configs import V4Config
from hyper_mve.models import HyperMuZeroModel, Projector
from hyper_mve.envs.resource_commons.env import ResourceCommonsEnv
from hyper_mve.schemas import AgentType
from hyper_mve.training import EpisodeReplayBuffer, MuZeroTrainer, Worker

_SUB_CONFIGS = ("env", "model", "train", "mup", "eval", "legacy")
_DEFERRED_VARIANTS = {
    "oracle_only": "needs curriculum_stage_*_end_frac=1.0 (TrainConfig forbids); Pkg-08",
    "infer_only": "needs curriculum_stage_*_end_frac=0.0 (TrainConfig forbids); Pkg-08",
    "baseline_input_wide": "needs Pkg-06 shared_backbones model factory",
    "baseline_input_deep": "needs Pkg-06 shared_backbones model factory",
    "baseline_ma_muzero": "needs Pkg-06 shared_backbones model factory",
    "no_belief": "needs Pkg-06 model factory (Ablation 7)",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Hyper-MuZero v4 unified trainer")
    p.add_argument("--preset", default="medium", choices=("easy", "medium", "hard"))
    p.add_argument("--variant", default="hyper")
    p.add_argument("--max_steps", type=int, default=None, help="override train.max_train_steps")
    p.add_argument("--override", action="append", default=[], help='"section.field=value" (repeatable)')
    p.add_argument("--resume_from", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_dir", default=None)
    p.add_argument("--ckpt_dir", default="checkpoints/v4")
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
        if section == "env" and field == "type_assignment":
            value = tuple(AgentType(int(x)) for x in value)
        sub = getattr(cfg, section)
        cfg = replace(cfg, **{section: replace(sub, **{field: value})})
    return cfg


def apply_variant(cfg: V4Config, variant: str) -> V4Config:
    if variant == "hyper":
        return cfg
    if variant in _DEFERRED_VARIANTS:
        raise NotImplementedError(
            f"--variant {variant!r} not supported in Pkg-05: {_DEFERRED_VARIANTS[variant]}."
        )
    raise ValueError(f"unknown --variant {variant!r}")


def build_model(cfg: V4Config, variant: str) -> HyperMuZeroModel:
    if variant == "hyper":
        return HyperMuZeroModel(cfg)
    raise NotImplementedError(f"model factory for --variant {variant!r} lands in Pkg-06")


def _epsilon(cfg: V4Config, step: int) -> float:
    t = cfg.train
    frac = min(1.0, step / max(1, t.epsilon_decay_steps))
    return float(t.epsilon_min + (t.epsilon_init - t.epsilon_min) * (1.0 - frac))


def main(argv=None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = V4Config.from_preset(args.preset)
    cfg = apply_overrides(cfg, args.override)
    cfg = apply_variant(cfg, args.variant)
    if args.max_steps is not None:
        cfg = replace(cfg, train=replace(cfg.train, max_train_steps=args.max_steps))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, args.variant)
    projector = Projector(cfg.model.latent_dim, cfg.model.proj_dim)
    env = ResourceCommonsEnv(cfg.env, seed=args.seed)

    trainer = MuZeroTrainer(cfg, model, projector=projector, device=device)
    worker = Worker(cfg, model, env)
    buffer = EpisodeReplayBuffer(cfg)

    start_step = trainer.load_checkpoint(args.resume_from) if args.resume_from else 0

    writer = None
    if args.log_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(args.log_dir)
        except ImportError:
            print("[train_main] tensorboard unavailable; logging to stdout only.")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    max_steps = cfg.train.max_train_steps
    global_step = start_step

    # Warm up the buffer (fast collection without the planner).
    while len(buffer) < cfg.train.min_buffer_size:
        records, c_t_seq = worker.collect_episode(epsilon=1.0, use_planner=False)
        buffer.store_episode(records, c_t_seq)

    while global_step < max_steps:
        eps = _epsilon(cfg, global_step)
        for _ in range(cfg.train.episodes_per_iter):
            records, c_t_seq = worker.collect_episode(epsilon=eps, use_planner=False)
            buffer.store_episode(records, c_t_seq)

        for _ in range(cfg.train.train_steps_per_iter):
            batch = buffer.sample_batch(cfg.train.batch_size, cfg.train.unroll_K)
            losses = trainer.train_step(batch, global_step)
            global_step += 1

            if global_step % 100 == 0:
                msg = (f"step {global_step}  total={losses['total']:.4f} "
                       f"main={losses['main']:.4f} belief={losses['belief']:.4f} "
                       f"lr={losses['lr']:.2e}")
                print(msg)
                if writer is not None:
                    for k, v in losses.items():
                        writer.add_scalar(f"loss/{k}", v, global_step)
            if global_step % 10_000 == 0:
                trainer.save_checkpoint(os.path.join(args.ckpt_dir, f"step_{global_step}.pt"))
            if global_step >= max_steps:
                break

    trainer.save_checkpoint(os.path.join(args.ckpt_dir, f"step_{global_step}.pt"))
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
