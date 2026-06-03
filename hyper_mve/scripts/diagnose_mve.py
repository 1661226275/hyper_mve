"""diagnose_mve.py — offline probe for the MVE planner (Ch5.6 / 5.8).

Loads a config (and optionally a checkpoint), runs a few env steps, and prints —
per agent, per step — the planner's **expected return for each candidate first
action**, the resulting search policy ``pi_mve``, and the prediction net's own
prior. This answers the two diagnostic questions directly, by eye:

    1. Is the MVE planner differentiating?  → look at the per-action return spread
       and H(pi_mve). Flat returns / H(pi_mve)≈ln(A) ⇒ no signal.
    2. Is the prediction net learning toward the plan?  → compare the planner's
       top action vs the prior's top action, and CE(pi_mve, prior).

Run from the repo root (so ``hyper_mve`` imports without a pip install):

    python hyper_mve/scripts/diagnose_mve.py --preset duo --seed 0 --steps 3
    python hyper_mve/scripts/diagnose_mve.py --preset duo \
        --ckpt checkpoints/duo_smoke/step_300.pt --steps 5 --greedy
    # the --preset/--override must match the architecture the checkpoint was trained with.
"""
from __future__ import annotations

import argparse
import os
import sys

# Allow `from hyper_mve...` when launched as `python hyper_mve/scripts/diagnose_mve.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import torch
import torch.nn.functional as F

from hyper_mve.configs import V4Config
from hyper_mve.envs.resource_commons.env import ResourceCommonsEnv
from hyper_mve.models import HyperMuZeroModel
from hyper_mve.planning.mve_planner import MVEPlanner
from hyper_mve.scripts.train_main import apply_overrides, apply_variant
from hyper_mve.utils.utils import inverse_scalar_transform

_ACTION_NAMES = ["NOOP", "UP", "DOWN", "LEFT", "RIGHT", "HARVEST"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Offline MVE planner probe")
    p.add_argument("--preset", default="duo", choices=("easy", "medium", "hard", "duo"))
    p.add_argument("--variant", default="hyper")
    p.add_argument("--override", action="append", default=[], help='"section.field=value" (repeatable)')
    p.add_argument("--ckpt", default=None, help="optional v4 checkpoint (model_state loaded strict)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=3, help="how many env steps to probe")
    p.add_argument("--greedy", action="store_true", help="advance env with argmax(pi_mve) (else sample)")
    return p.parse_args(argv)


def _action_labels(A):
    return _ACTION_NAMES if A == len(_ACTION_NAMES) else [str(i) for i in range(A)]


def _entropy(p):
    """Shannon entropy of a 1-D probability vector (numpy)."""
    p = np.clip(p, 1e-9, 1.0)
    return float(-(p * np.log(p)).sum())


def _fmt(vec, labels, highlight=None, sign=False):
    """Format a per-action vector with action labels; mark ``highlight`` with '*'."""
    out = []
    for i, v in enumerate(vec):
        mark = "*" if i == highlight else " "
        out.append(f"{mark}{labels[i]}={v:+.3f}" if sign else f"{mark}{labels[i]}={v:.3f}")
    return " ".join(out)


@torch.no_grad()
def probe(cfg: V4Config, args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, A = cfg.env.N, cfg.env.A
    labels = _action_labels(A)
    ln_A = float(np.log(A))

    model = HyperMuZeroModel(cfg).to(device)
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device)
        if ckpt.get("version") != "v4":
            print(f"[warn] checkpoint version={ckpt.get('version')!r} (expected 'v4').")
        model.load_state_dict(ckpt["model_state"])
        print(f"[probe] loaded {args.ckpt} (global_step={ckpt.get('global_step')}).")
    else:
        print("[probe] no --ckpt: probing a FRESH (untrained) model.")
    model.eval()

    env = ResourceCommonsEnv(cfg.env, seed=args.seed)
    planner = MVEPlanner(cfg)
    roles = [cfg.env.type_assignment[k].name for k in range(N)]

    print(f"[probe] preset={cfg.preset_name} N={N} A={A} c_mode={cfg.env.c_mode} "
          f"mve_samples={cfg.train.mve_samples} depth={cfg.train.mve_depth} "
          f"temp={cfg.train.mve_temperature} | uniform H=ln(A)={ln_A:.3f} | device={device}")

    obs, info = env.reset()
    prev_hidden = model.belief_net.init_hidden(1, N, device=device)

    for t in range(args.steps):
        obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(device)
        prev_hidden, c_hat, z_hat = model.belief_net.step(obs_t, prev_hidden)
        s = model.encode(obs_t)
        c_true = float(info["c_true"])
        c_t = torch.tensor([c_true], dtype=torch.float32, device=device)
        model.set_context_objective(c_t)

        cap_dict, belief_dict = {}, {}
        prior = np.zeros((N, A), dtype=np.float32)
        values = np.zeros(N, dtype=np.float32)
        theta_pred = []
        for k in range(N):
            cap_k = torch.from_numpy(
                np.asarray(info["caps"][k].to_array(), dtype=np.float32)
            ).unsqueeze(0).to(device)
            cap_dict[k] = cap_k
            belief_dict[k] = (c_hat[0, k:k + 1], z_hat[0, k:k + 1])
            model.set_context_subjective(k, cap_k, belief_dict[k])
            logits_k, v_k = model.predict(s)
            prior[k] = F.softmax(logits_k, dim=-1)[0].cpu().numpy()
            values[k] = float(inverse_scalar_transform(v_k).item())
            theta_pred.append(model.current_subjective_thetas()[1])  # theta_pred (1, P)

        pi_mve, diag = planner.sample_mve_plan(
            model, s, cap_dict, belief_dict, c_t, return_diagnostics=True,
        )
        pi_mve = pi_mve[0].cpu().numpy()                          # (N, A)
        returns = diag["returns_per_action"][0].cpu().numpy()    # (N, A)

        print(f"\n===== step {t}  c_true={c_true:.3f} =====")
        for k in range(N):
            ret_best = int(np.argmax(returns[k]))
            mve_best = int(np.argmax(pi_mve[k]))
            pri_best = int(np.argmax(prior[k]))
            spread = float(returns[k].max() - returns[k].min())
            ce = float(-(pi_mve[k] * np.log(np.clip(prior[k], 1e-9, 1.0))).sum())
            agree = "YES" if mve_best == pri_best else "no"
            print(f" agent{k}[{roles[k]:>5}] V={values[k]:+.3f}  return_spread={spread:.3f}  "
                  f"plan_best={labels[ret_best]} prior_best={labels[pri_best]} agree={agree}")
            print(f"   returns : {_fmt(returns[k], labels, ret_best, sign=True)}")
            print(f"   pi_mve  : {_fmt(pi_mve[k], labels, mve_best)}   H={_entropy(pi_mve[k]):.3f}")
            print(f"   prior   : {_fmt(prior[k], labels, pri_best)}   H={_entropy(prior[k]):.3f}  CE(mve,prior)={ce:.3f}")

        # role discrimination: mean cosine of theta_pred across different-type agent pairs
        cross = [F.cosine_similarity(theta_pred[i], theta_pred[j], dim=-1).mean().item()
                 for i in range(N) for j in range(i + 1, N)
                 if cfg.env.type_assignment[i] != cfg.env.type_assignment[j]]
        if cross:
            print(f"   cos(theta_pred, cross-type) = {sum(cross) / len(cross):+.3f}  "
                  f"(lower -> roles better separated)")

        # advance the env
        if args.greedy:
            joint = np.array([int(np.argmax(pi_mve[k])) for k in range(N)], dtype=np.int64)
        else:
            joint = np.array([int(np.random.choice(A, p=pi_mve[k] / pi_mve[k].sum()))
                              for k in range(N)], dtype=np.int64)
        obs, _r, done, _trunc, info = env.step(joint)
        if done:
            print("[probe] episode done; stopping early.")
            break


def main(argv=None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = V4Config.from_preset(args.preset)
    cfg = apply_overrides(cfg, args.override)
    cfg = apply_variant(cfg, args.variant)
    probe(cfg, args)


if __name__ == "__main__":
    main()
