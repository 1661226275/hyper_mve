"""eval_game_metrics.py — post-hoc NashConv + empirical PoA on a frozen checkpoint.

Run from the repo root:

    # welfare only (fast; also used to obtain the coop reference Ŵ*):
    python scripts/eval_game_metrics.py \\
        --ckpt <run_dir>/ckpt.pt --external mappo --preset rel_duo \\
        --welfare-only --regimes 0 --episodes 20 \\
        --out results/analysis/coop_ref.json

    # full NashConv + efficiency against a coop reference:
    python scripts/eval_game_metrics.py \\
        --ckpt <run_dir>/ckpt.pt --external mamba --preset rel_duo \\
        --br-steps 20000 --episodes 10 \\
        --coop-ref results/analysis/coop_ref.json \\
        --out results/analysis/game_metrics_mamba.json

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (scripts/ is top-level)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="v5 game-theoretic metrics (NashConv + PoA)")
    p.add_argument("--ckpt", required=True,
                   help="external runner checkpoint (.pt) written by save_checkpoint")
    p.add_argument("--preset", default="rel_duo")
    p.add_argument("--variant", default=None,
                   help="label recorded in the report (default: --external value)")
    p.add_argument("--external", required=True,
                   choices=("mappo", "mamba", "mamba_pm", "mazero_mixed"),
                   help="which runner produced --ckpt. Must be the REGISTRY key, "
                        "not just the family: the _pm variants are wider, so "
                        "loading a mamba_pm checkpoint as 'mamba' fails on a "
                        "shape mismatch. happo / mbom / m3w_adapted expose their "
                        "act function as a local closure rather than a method, "
                        "so they have no frozen-policy adapter yet.")
    p.add_argument("--frozen-mode", choices=("prior", "planner", "both"),
                   default="prior",
                   help="mazero_mixed only: which policy to freeze. 'prior' is "
                        "the distilled prediction net (what the model-free "
                        "baselines expose, so the comparable choice); 'planner' "
                        "is the MCTS policy that actually deploys. They are very "
                        "different — on v5, A3 prior 14.28 vs A1 planner 61.94 — "
                        "so this changes what the NashConv means. 'both' writes "
                        "<out> and <out>.planner.json and prints the gap.")
    p.add_argument("--regimes", type=int, nargs="*", default=None,
                   help="regime ids (default: all in the preset family)")
    p.add_argument("--br-steps", type=int, default=20_000,
                   help="BR DQN env-step budget per (agent, regime)")
    p.add_argument("--episodes", type=int, default=10,
                   help="deterministic eval episodes per rollout")
    p.add_argument("--seed", type=int, default=10_000,
                   help="episode seeds are seed + 97*g + ep, so the default "
                        "reproduces MAZeroMixedRunner.evaluate's initial "
                        "conditions and v_pi lines up with eval_report.json's "
                        "per-regime returns. (Archived game_metrics_*.json "
                        "predate both this offset and this default.)")
    p.add_argument("--coop-ref", default=None,
                   help="coop reference: a float, or a game-metrics JSON whose "
                        "welfare_physical['0'] (all-coop regime) is used as Ŵ*")
    p.add_argument("--welfare-only", action="store_true",
                   help="skip BR training; report per-regime welfare only")
    p.add_argument("--device", default=None, help='"cpu" / "cuda" (default: auto)')
    p.add_argument("--out", required=True, help="output JSON path")
    return p.parse_args(argv)


def _load_external_frozen(variant: str, ckpt_path: str, cfg, device,
                          frozen_mode: str = "prior"):
    """Build a runner from its checkpoint and wrap it as a FrozenExternalPolicy.

    ``FrozenExternalPolicy.joint_actions`` calls ``act_fn(obs, t)`` with two
    arguments, so every runner's act function is adapted to that arity here.
    """
    import numpy as np

    from hyper_mve.comparison import create_runner
    from hyper_mve.envs.adapters.pettingzoo_wrapper import RelationCommonsPettingZooEnv
    from hyper_mve.utils.eval.game_metrics import FrozenExternalPolicy

    if variant == "mazero_mixed":
        from hyper_mve.algo.runner import MAZeroMixedRunner

        runner = MAZeroMixedRunner(cfg)
        runner.load_checkpoint(ckpt_path)
        act_fn = runner.make_act_fn(frozen_mode, device=device)
        return FrozenExternalPolicy(act_fn, cfg, device=device)

    runner = create_runner(cfg, variant)
    if variant.startswith("mappo"):
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

    else:  # mamba / mamba_pm — load_checkpoint rebuilds the learner standalone
        runner.load_checkpoint(ckpt_path)
        # _probe_act is (obs, t, g) for the PeriodicEvalProbe contract; the
        # frozen-policy contract is (obs, t). Adapt rather than call directly —
        # passing the 3-arg method straight through raised TypeError, which is
        # why this path was broken. g is unused (mamba is regime-blind).
        _probe = runner._probe_act

        def act_fn(obs, t):
            return _probe(obs, t, -1)

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
    from hyper_mve.utils.configs import V4Config
    from hyper_mve.utils.eval.game_metrics import BRConfig

    cfg = V4Config.from_preset(args.preset)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    if args.variant is None:
        args.variant = args.external
    if args.frozen_mode != "prior" and args.external != "mazero_mixed":
        raise SystemExit(
            f"--frozen-mode is mazero_mixed-only (got --external {args.external}); "
            "the model-free baselines expose one policy."
        )

    coop_ref, provenance = _resolve_coop_ref(args.coop_ref)
    br = BRConfig(env_steps=0 if args.welfare_only else args.br_steps)

    modes = ("prior", "planner") if args.frozen_mode == "both" else (args.frozen_mode,)
    reports = {}
    for mode in modes:
        reports[mode] = _one_mode(args, cfg, device, br, coop_ref, provenance, mode)

    for mode, report in reports.items():
        suffix = "" if mode == modes[0] else f".{mode}"
        out = pathlib.Path(args.out)
        out = out if not suffix else out.with_suffix(f"{suffix}{out.suffix}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"[eval_game_metrics] wrote {out}  (frozen={mode})")
        for g in report.regime_ids:
            nc = report.nashconv.get(g, float("nan"))
            eff = report.efficiency.get(g, float("nan"))
            print(f"  regime {g}: welfare="
                  f"{report.welfare_physical.get(g, float('nan')):.2f} "
                  f"nashconv={nc:.3f} efficiency={eff:.3f}")

    if len(reports) == 2:
        print("\n  planner − prior, per regime "
              "(how much the search changes exploitability):")
        for g in reports["prior"].regime_ids:
            a = reports["prior"].nashconv.get(g, float("nan"))
            b = reports["planner"].nashconv.get(g, float("nan"))
            print(f"    g{g}: nashconv {a:7.3f} -> {b:7.3f}   ({b - a:+7.3f})")


def _one_mode(args, cfg, device, br, coop_ref, provenance, frozen_mode):
    from hyper_mve.utils.eval.game_metrics import compute_game_metrics

    model = None            # the retired internal HyperMuZeroModel path is gone;
                            # every runner now arrives through an act_fn.
    frozen = _load_external_frozen(
        args.external, args.ckpt, cfg, device, frozen_mode)

    if args.welfare_only:
        # welfare-only fast path: monkey-cheap BR skip (env_steps=0 would still
        # loop agents), so short-circuit via regime loop without BR.
        from hyper_mve.utils.eval.game_metrics import FrozenPriorPolicy, GameMetricsReport, _rollout
        from hyper_mve.utils.schemas import get_regime_family

        family = get_regime_family(cfg.env)
        regime_ids = args.regimes if args.regimes is not None else list(range(family.size))
        if frozen is None:
            frozen = FrozenPriorPolicy(model, cfg, device=device)
        report = GameMetricsReport(
            variant=f"{args.variant}:{frozen_mode}", checkpoint=args.ckpt,
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
            variant=f"{args.variant}:{frozen_mode}",
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
    return report


if __name__ == "__main__":
    main()
