"""pkg-08 spec 05 §5.3 — sweep worker subprocess entry.

One process per sweep row. Reads a SweepRow payload from stdin (JSON), builds a
single ``V4Config`` (preset + overrides + ``eval_planner_mode`` + ``max_steps``),
constructs the runner, **trains it in-process**, then evaluates the *same trained
object* via the unified evaluator and writes ``EvalReport`` JSON.

Training is wired per runner family (pkg-07 spec 08 §7.1):
  * ``hyper`` + the 4 internal ``BaselineModel`` variants share the v5 6-API, so
    they train through the shared ``train_main.run_training`` MuZeroTrainer loop.
  * external runners own their algorithm loop and train via ``runner.train``.

This closes the earlier gap where the worker called the (runner-owned no-op)
``train_main`` and then evaluated a *freshly-initialised* model — every variant
(``hyper`` included) was scored at random init.

Exit codes:
  0   → row completed; EvalReport written
  1   → Python exception (failed)
  2   → NotImplementedError (stub variant; skipped)
  >2  → CUDA segfault, OOM, etc.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
from dataclasses import asdict, replace
from typing import Any


def _apply_runtime_tuning() -> None:
    """Read launcher-provided env knobs and tune the worker runtime.

    Must run BEFORE any project import that creates a CUDA context.

    Env vars consumed:
        HYPER_MVE_GPU_MEM_FRAC      "0.45" → set_per_process_memory_fraction(0.45, 0)
        HYPER_MVE_CUDNN_BENCHMARK   "1"    → torch.backends.cudnn.benchmark = True
        HYPER_MVE_MATMUL_PRECISION  "high" → torch.set_float32_matmul_precision("high")

    All failures degrade silently (one-line stderr warning). The default
    behaviour without any env var set is identical to legacy sweep harness.
    """
    mem_frac = os.environ.get("HYPER_MVE_GPU_MEM_FRAC")
    cudnn_bench = os.environ.get("HYPER_MVE_CUDNN_BENCHMARK") == "1"
    matmul_prec = os.environ.get("HYPER_MVE_MATMUL_PRECISION")
    if not (mem_frac or cudnn_bench or matmul_prec):
        return
    try:
        import torch
    except ImportError:
        return
    if mem_frac and torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(float(mem_frac), 0)
        except (RuntimeError, ValueError) as e:
            print(
                f"[worker] set_per_process_memory_fraction({mem_frac!r}) failed: {e}",
                file=sys.stderr, flush=True,
            )
    if cudnn_bench:
        try:
            torch.backends.cudnn.benchmark = True
        except Exception as e:  # noqa: BLE001
            print(f"[worker] enable cudnn.benchmark failed: {e}",
                  file=sys.stderr, flush=True)
    if matmul_prec in ("highest", "high", "medium"):
        try:
            torch.set_float32_matmul_precision(matmul_prec)
        except Exception as e:  # noqa: BLE001
            print(
                f"[worker] set_float32_matmul_precision({matmul_prec!r}) failed: {e}",
                file=sys.stderr, flush=True,
            )
    if sys.platform != "win32":
        try:
            import torch.multiprocessing as _tmp
            _tmp.set_sharing_strategy("file_system")
        except Exception:  # noqa: BLE001
            pass


def _build_cfg(payload: dict[str, Any]):
    """One V4Config: preset + payload overrides + eval_planner_mode + max_steps.

    Mirrors the argv that used to be handed to ``train_main`` but builds the cfg
    directly so the *same* cfg drives training and eval. (The old eval path
    rebuilt cfg from the bare preset and silently dropped the overrides +
    ``eval_planner_mode``.)
    """
    from hyper_mve.configs import V4Config
    from hyper_mve.scripts.train_main import apply_overrides

    cfg = V4Config.from_preset(payload["preset"])
    override_items = [
        f"{k}={json.dumps(v)}" for k, v in (payload.get("overrides") or {}).items()
    ]
    override_items.append(
        f"eval.eval_planner_mode={json.dumps(payload['eval_planner_mode'])}"
    )
    cfg = apply_overrides(cfg, override_items)
    if payload.get("max_steps"):
        cfg = replace(
            cfg, train=replace(cfg.train, max_train_steps=int(payload["max_steps"]))
        )
    return cfg


def _write_eval_report(report: Any, out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(report, "to_dict"):
        body = report.to_dict()
    else:
        body = asdict(report)
    out_path.write_text(json.dumps(body, default=str), encoding="utf-8")


def _write_config_snapshot(cfg: Any, out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(cfg, "to_dict"):
        body = cfg.to_dict()
    else:
        body = asdict(cfg)
    out_path.write_text(json.dumps(body, default=str, indent=2), encoding="utf-8")


def _build_runner(cfg, variant: str):
    """Construct the runner for a variant (hyper / internal baseline / external).

    Stub / deferred variants raise ``NotImplementedError`` from construction; the
    caller maps that to exit code 2 (SKIP).
    """
    from hyper_mve.baselines import REGISTRY, cli_to_factory_arg, create_baseline

    if variant in {"hyper", "oracle_only", "infer_only"}:
        # oracle_only / infer_only need curriculum overrides for their full
        # semantics (deferred, pkg-08); they are not in the default sweep. Here
        # they construct as plain hyper so the path stays total.
        from hyper_mve.models.hyper_muzero_model import HyperMuZeroModel
        return HyperMuZeroModel(cfg)
    factory_arg = cli_to_factory_arg(variant) if variant not in REGISTRY else variant
    return create_baseline(cfg, factory_arg)


def _train_runner(cfg, runner, payload: dict[str, Any], seed: int) -> None:
    """Train ``runner`` in-process: shared MuZeroTrainer loop for the 7-API models
    (hyper + internal baselines), the runner's own loop for external baselines."""
    from hyper_mve.baselines.external.base import ExternalBaselineRunner

    if isinstance(runner, ExternalBaselineRunner):
        from hyper_mve.envs.adapters.pettingzoo_wrapper import (
            RelationCommonsPettingZooEnv,
        )

        def env_fn() -> "RelationCommonsPettingZooEnv":
            return RelationCommonsPettingZooEnv(
                cfg.env, oracle_mode=False, eval_info_mode=False,
            )

        runner.train(
            cfg, env_fn,
            total_env_steps=int(cfg.train.max_train_steps),
            seed=seed,
            # Optional kwarg (base-contract **kwargs slack): runners that
            # support the PeriodicEvalProbe write sample-efficiency TB curves
            # into the row's tb/ dir; others swallow it.
            tensorboard_dir=payload["tensorboard_dir"],
        )
        runner.save_checkpoint(pathlib.Path(payload["checkpoint_path"]))
    else:
        from hyper_mve.scripts.train_main import run_training

        run_training(
            cfg, runner,
            seed=seed,
            ckpt_dir=str(pathlib.Path(payload["checkpoint_path"]).parent),
            log_dir=payload["tensorboard_dir"],
            collect_planner=True,
            preset=payload["preset"],
            variant=payload["variant"],
        )


def main() -> None:
    _apply_runtime_tuning()
    payload = json.loads(sys.stdin.read())
    variant = payload["variant"]
    seed = int(payload["seed"])

    try:
        import numpy as np
        import torch
        from hyper_mve.eval import evaluate
        from hyper_mve.envs.adapters.pettingzoo_wrapper import (
            RelationCommonsPettingZooEnv,
        )
    except ImportError as e:
        print(f"FAIL: worker imports: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    torch.manual_seed(seed)
    np.random.seed(seed)

    try:
        cfg = _build_cfg(payload)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: cfg build: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
    _write_config_snapshot(cfg, pathlib.Path(payload["config_snapshot_path"]))

    def env_fn() -> RelationCommonsPettingZooEnv:
        # eval_info_mode=False is the locked external-runner eval contract.
        # With eval_info_mode=True the env exposes the eval-only diagnostic
        # field ('resource_state'), which trips every external runner's
        # per-step _check_forbidden_info CTDE guard — and nothing in the
        # current eval path consumes it (the internal branch ignores env_fn
        # entirely; run_eval builds its own envs). oracle_mode stays False so
        # _verify_env_fn_flags passes.
        return RelationCommonsPettingZooEnv(
            cfg.env, oracle_mode=False, eval_info_mode=False,
        )

    # --- construct runner (stub/deferred variants → SKIP) ---
    try:
        runner = _build_runner(cfg, variant)
    except NotImplementedError as e:
        print(f"SKIP: {e}", file=sys.stderr, flush=True)
        sys.exit(2)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: build runner: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    # --- TRAIN (the gap this closes: runner-owned variants were never trained) ---
    try:
        _train_runner(cfg, runner, payload, seed)
    except NotImplementedError as e:
        print(f"SKIP: {e}", file=sys.stderr, flush=True)
        sys.exit(2)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: train: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    # --- EVAL the SAME trained object ---
    try:
        report = evaluate(runner, env_fn, cfg, variant=variant)
        _write_eval_report(report, pathlib.Path(payload["eval_report_path"]))
    except NotImplementedError as e:
        print(f"SKIP: {e}", file=sys.stderr, flush=True)
        sys.exit(2)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: eval: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":  # pragma: no cover
    main()
