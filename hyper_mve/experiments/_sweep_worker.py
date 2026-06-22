"""pkg-08 spec 05 §5.3 — sweep worker subprocess entry.

One process per sweep row. Reads a SweepRow payload from stdin (JSON),
constructs the train_main argv, calls ``train_main.main(argv)``, runs the
unified evaluator, then writes ``EvalReport`` JSON to disk.

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
from dataclasses import asdict
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


def _emit_argv(payload: dict[str, Any]) -> list[str]:
    argv: list[str] = [
        "--variant", payload["variant"],
        "--seed", str(payload["seed"]),
        "--preset", payload["preset"],
        "--max_steps", str(payload["max_steps"]),
        "--log_dir", payload["tensorboard_dir"],
        "--ckpt_dir", str(pathlib.Path(payload["checkpoint_path"]).parent),
    ]
    overrides = payload.get("overrides") or {}
    for key, value in overrides.items():
        argv += ["--override", f"{key}={json.dumps(value)}"]
    # Inject eval_planner_mode (pkg-08 spec 03 — eval-time mode lives on cfg.eval).
    argv += [
        "--override",
        f"eval.eval_planner_mode={json.dumps(payload['eval_planner_mode'])}",
    ]
    return argv


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


def main() -> None:
    _apply_runtime_tuning()
    payload = json.loads(sys.stdin.read())

    argv = _emit_argv(payload)
    try:
        from hyper_mve.scripts.train_main import main as train_main
    except ImportError as e:
        print(f"FAIL: train_main import: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    try:
        train_main(argv)
    except NotImplementedError as e:
        print(f"SKIP: {e}", file=sys.stderr, flush=True)
        sys.exit(2)
    except SystemExit as e:
        # train_main may sys.exit on success; treat 0 as completion, anything else as fail.
        code = e.code if isinstance(e.code, int) else 1
        if code == 0:
            pass  # fall through to eval phase
        else:
            print(f"FAIL: train_main SystemExit({code})", file=sys.stderr, flush=True)
            sys.exit(1)
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    # Eval phase — runs the unified evaluator on the trained model.
    try:
        from hyper_mve.configs import V4Config
        from hyper_mve.eval import evaluate
        from hyper_mve.envs.adapters.pettingzoo_wrapper import (
            ResourceCommonsPettingZooEnv,
        )
    except ImportError as e:
        print(f"FAIL: eval phase imports: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    cfg = V4Config.from_preset(payload["preset"])
    _write_config_snapshot(cfg, pathlib.Path(payload["config_snapshot_path"]))

    def env_fn() -> ResourceCommonsPettingZooEnv:
        return ResourceCommonsPettingZooEnv(
            cfg.env, oracle_mode=False, eval_info_mode=True,
        )

    # Worker-side runner reconstruction. Internal baselines + curriculum-overrides
    # use the factory; external runners follow the same path. The key invariant
    # is that ``evaluate`` accepts both BaselineModel and ExternalBaselineRunner.
    try:
        from hyper_mve.baselines import REGISTRY, cli_to_factory_arg, create_baseline
        variant = payload["variant"]
        if variant in {"hyper", "oracle_only", "infer_only"}:
            from hyper_mve.models.hyper_muzero_model import HyperMuZeroModel
            runner = HyperMuZeroModel(cfg)
        else:
            factory_arg = cli_to_factory_arg(variant) if variant not in REGISTRY else variant
            runner = create_baseline(cfg, factory_arg)
        report = evaluate(runner, env_fn, cfg)
        _write_eval_report(report, pathlib.Path(payload["eval_report_path"]))
    except NotImplementedError as e:
        print(f"SKIP: {e}", file=sys.stderr, flush=True)
        sys.exit(2)
    except Exception as e:
        print(f"FAIL: eval: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":  # pragma: no cover
    main()
