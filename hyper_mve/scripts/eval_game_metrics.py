"""eval_game_metrics.py — post-hoc NashConv + empirical PoA on a frozen checkpoint.

Run from the repo root:

    # welfare only (fast; also used to obtain the coop reference Ŵ*):
    python hyper_mve/scripts/eval_game_metrics.py \\
        --ckpt checkpoints/v4/best.pt --preset rel_duo --welfare-only \\
        --regimes 0 --episodes 20 --out runs/_analysis/coop_ref.json

    # full NashConv + efficiency against a coop reference:
    python hyper_mve/scripts/eval_game_metrics.py \\
        --ckpt checkpoints/v4/best.pt --preset rel_duo \\
        --br-steps 20000 --episodes 10 \\
        --coop-ref runs/_analysis/coop_ref.json \\
        --out runs/_analysis/game_metrics_hyper.json

The JSON deliverable (schema ``game-metrics-v1``) is consumed by the analysis
stage; it is intentionally NOT part of the frozen per-run EvalReport (BR
training is too expensive to run at every eval).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="v5 game-theoretic metrics (NashConv + PoA)")
    p.add_argument("--ckpt", required=True,
                   help="checkpoint (.pt): MuZeroTrainer ckpt (internal, default) "
                        "or an external runner ckpt when --external is given")
    p.add_argument("--preset", default="rel_duo")
    p.add_argument("--variant", default="hyper", help="label recorded in the report")
    p.add_argument("--external", default=None,
                   choices=("external_mappo", "external_mamba"),
                   help="[M3 extension 2026-07-10] evaluate an EXTERNAL baseline "
                        "checkpoint instead of an internal model; --ckpt then "
                        "points at the runner's save_checkpoint output")
    p.add_argument("--regimes", type=int, nargs="*", default=None,
                   help="regime ids (default: all in the preset family)")
    p.add_argument("--br-steps", type=int, default=20_000,
                   help="BR DQN env-step budget per (agent, regime)")
    p.add_argument("--episodes", type=int, default=10,
                   help="deterministic eval episodes per rollout")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--coop-ref", default=None,
                   help="coop reference: a float, or a game-metrics JSON whose "
                        "welfare_physical['0'] (all-coop regime) is used as Ŵ*")
    p.add_argument("--welfare-only", action="store_true",
                   help="skip BR training; report per-regime welfare only")
    p.add_argument("--device", default=None, help='"cpu" / "cuda" (default: auto)')
    p.add_argument("--out", required=True, help="output JSON path")
    return p.parse_args(argv)


def _load_model(ckpt_path: str, cfg):
    import torch
    from hyper_mve.models import HyperMuZeroModel

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = HyperMuZeroModel(cfg)
    model.load_state_dict(ckpt["model_state"])
    return model


def _load_external_frozen(variant: str, ckpt_path: str, cfg, device):
    """Build an external runner from its checkpoint and wrap it as a
    FrozenExternalPolicy (M3 extension, 2026-07-10)."""
    import numpy as np

    from hyper_mve.baselines import create_baseline
    from hyper_mve.envs.adapters.pettingzoo_wrapper import RelationCommonsPettingZooEnv
    from hyper_mve.eval.game_metrics import FrozenExternalPolicy

    runner = create_baseline(cfg, variant)
    if variant == "external_mappo":
        # MAPPO's load_checkpoint needs the agent built first (env-derived
        # obs_dim); evaluate() with episodes=0 is the sanctioned builder path.
        env_fn = lambda: RelationCommonsPettingZooEnv(
            cfg.env, oracle_mode=False, eval_info_mode=False,
        )
        runner.evaluate(env_fn, regime_grid=(0,), episodes=0)
        runner.load_checkpoint(ckpt_path)

        def act_fn(obs, t):
            del t  # MLP policy — stateless
            a_n, _ = runner._agent.choose_action(obs, evaluate=True)
            return np.asarray(a_n)

    else:  # external_mamba — load_checkpoint rebuilds the learner standalone
        runner.load_checkpoint(ckpt_path)
        act_fn = runner._probe_act

    return FrozenExternalPolicy(act_fn, cfg, device=device)


def _resolve_coop_ref(raw: str | None) -> tuple[float | None, str]:
    if raw is None:
        return None, ""
    try:
        return float(raw), f"literal:{raw}"
    except ValueError:
        pass
    body = json.loads(pathlib.Path(raw).read_text(encoding="utf-8"))
    welfare = body.get("welfare_physical", {})
    if "0" not in welfare:
        raise SystemExit(
            f"--coop-ref {raw}: no welfare_physical['0'] (all-coop regime) in JSON"
        )
    return float(welfare["0"]), f"json:{raw} (regime 0 welfare)"


def main(argv=None) -> None:
    args = parse_args(argv)

    import torch
    from hyper_mve.configs import V4Config
    from hyper_mve.eval.game_metrics import BRConfig, compute_game_metrics

    cfg = V4Config.from_preset(args.preset)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    frozen = None
    model = None
    if args.external:
        frozen = _load_external_frozen(args.external, args.ckpt, cfg, device)
        if args.variant == "hyper":       # default label follows the runner
            args.variant = args.external
    else:
        model = _load_model(args.ckpt, cfg).to(device)

    coop_ref, provenance = _resolve_coop_ref(args.coop_ref)
    br = BRConfig(env_steps=0 if args.welfare_only else args.br_steps)

    if args.welfare_only:
        # welfare-only fast path: monkey-cheap BR skip (env_steps=0 would still
        # loop agents), so short-circuit via regime loop without BR.
        from hyper_mve.eval.game_metrics import FrozenPriorPolicy, GameMetricsReport, _rollout
        from hyper_mve.schemas import get_regime_family

        family = get_regime_family(cfg.env)
        regime_ids = args.regimes if args.regimes is not None else list(range(family.size))
        if frozen is None:
            frozen = FrozenPriorPolicy(model, cfg, device=device)
        report = GameMetricsReport(
            variant=args.variant, checkpoint=args.ckpt,
            regime_ids=list(regime_ids), br_env_steps=0,
            eval_episodes=args.episodes,
            coop_reference_welfare=coop_ref,
            coop_reference_provenance=provenance,
        )
        for g in regime_ids:
            base_returns, welfare = _rollout(cfg, frozen, g, args.episodes, args.seed)
            report.v_pi[g] = [float(x) for x in base_returns]
            report.welfare_physical[g] = welfare
            if coop_ref and coop_ref > 0:
                report.efficiency[g] = welfare / coop_ref
    else:
        report = compute_game_metrics(
            model, cfg,
            variant=args.variant,
            checkpoint=args.ckpt,
            regime_ids=args.regimes,
            br=br,
            eval_episodes=args.episodes,
            seed=args.seed,
            coop_reference_welfare=coop_ref,
            coop_reference_provenance=provenance,
            device=device,
            frozen=frozen,
        )

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(f"[eval_game_metrics] wrote {out}")
    for g in report.regime_ids:
        nc = report.nashconv.get(g, float("nan"))
        eff = report.efficiency.get(g, float("nan"))
        print(f"  regime {g}: welfare={report.welfare_physical.get(g, float('nan')):.2f} "
              f"nashconv={nc:.3f} efficiency={eff:.3f}")


if __name__ == "__main__":
    main()
