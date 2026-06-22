"""Tiered pytest orchestrator for the full hyper_mve test suite.

Runs ``tests/`` in 4 tiers, each in its own ``subprocess.run`` so the parent
process never imports torch and a tier crash never aborts later tiers:

    Tier 1 — pure-Python (no torch).  Fastest; validates configs, schemas,
             experiments registry, drift detectors.
    Tier 2 — torch CPU.  Env, models, training, planning, internal baselines.
    Tier 3 — slow external baselines smokes (mappo / qmix / ma_muzero_gh).
    Tier 4 — GPU performance gates (mve planner, forward perf, worker, trainer).

Outputs per invocation under ``runs/test_reports/<utc-stamp>/``:
    summary.json        rollup + per-tier counts / commands / durations
    summary.md          markdown table (paste-ready for thesis appendix)
    env.json            python / platform / torch / cuda detection
    collect_only.txt    raw ``pytest --collect-only`` output
    tier_<n>.txt        captured stdout
    tier_<n>.stderr.txt captured stderr
    tier_<n>.cmd.txt    exact argv that was run

CLI examples::

    python hyper_mve/scripts/run_full_tests.py --mode fast
    python hyper_mve/scripts/run_full_tests.py --mode all
    python hyper_mve/scripts/run_full_tests.py --tiers "1,4" --verbose
    python hyper_mve/scripts/run_full_tests.py --mode gpu --tier-4-timeout 1200

Exit codes:
    0   every selected tier passed (or had no tests collected)
    1   one or more tiers failed / errored / timed out
    2   pre-flight ``--collect-only`` health check failed
    130 interrupted by Ctrl+C (partial summary written)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence

REPO_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_REPORT_ROOT: pathlib.Path = REPO_ROOT / "runs" / "test_reports"
MAX_STDOUT_BYTES = 10 * 1024 * 1024  # 10 MB per tier capture cap


# ===== Tier definitions ====================================================

@dataclass(frozen=True)
class Tier:
    name: str                    # "tier_1"
    label: str                   # "pure-Python"
    paths: tuple[str, ...]       # relative to REPO_ROOT
    marker_expr: str             # e.g. "not gpu and not slow"
    timeout_s: int
    cpu_only: bool               # if True, force CUDA_VISIBLE_DEVICES=""
    requires_gpu: bool           # if True, inherit parent CUDA env


@dataclass
class TierResult:
    tier: Tier
    command: list[str]
    returncode: int | None
    status: str                  # passed|failed|errored|timeout|empty|interrupted
    counts: dict[str, int] = field(default_factory=dict)
    duration_s: float = 0.0
    stdout_path: pathlib.Path | None = None
    stderr_path: pathlib.Path | None = None
    notes: str | None = None


def _tier_1_paths() -> tuple[str, ...]:
    return (
        "tests/configs/",
        "tests/schemas/",
        "tests/migration/",
        "tests/test_legacy_import_warning.py",
        "tests/experiments/test_run_registry_row_schema.py",
        "tests/experiments/test_run_registry_jsonl_append_atomic.py",
        "tests/experiments/test_config_hash_stable.py",
        "tests/experiments/test_gpu_semaphore.py",
        "tests/experiments/test_multi_slot_gpu_semaphore.py",
        "tests/experiments/test_sweep_cartesian_correct.py",
        "tests/experiments/test_use_coord_desc_rename.py",
        "tests/experiments/test_stats_holm_bonferroni.py",
        "tests/experiments/test_compare_plot_renders_headless.py",
        "tests/experiments/test_disclosure_table_schema.py",
        "tests/experiments/test_ablation_cli_dispatch.py",
        "tests/integration/test_pkg08_drift_detectors.py",
    )


def _tier_2_paths() -> tuple[str, ...]:
    return (
        "tests/envs/",
        "tests/models/",
        "tests/training/",
        "tests/planning/",
        "tests/baselines/external/test_registry.py",
        "tests/baselines/external/test_external_adapter_consumption.py",
        "tests/baselines/external/test_external_eval_contract.py",
        "tests/baselines/external/test_stub_external_baselines.py",
    )


def _tier_3_paths() -> tuple[str, ...]:
    return (
        "tests/baselines/external/test_mappo_smoke.py",
        "tests/baselines/external/test_qmix_smoke.py",
        "tests/baselines/external/test_ma_muzero_gh_smoke.py",
    )


def _tier_4_paths() -> tuple[str, ...]:
    return (
        "tests/planning/test_mve_planner.py",
        "tests/models/test_forward_performance.py",
        "tests/training/test_worker.py",
        "tests/training/test_trainer_loop.py",
    )


def build_tiers(args: argparse.Namespace) -> list[Tier]:
    selected = _resolve_tier_selection(args)
    all_tiers: dict[int, Tier] = {
        1: Tier(
            name="tier_1", label="pure-Python",
            paths=_tier_1_paths(),
            marker_expr="not gpu and not slow",
            timeout_s=args.tier_1_timeout,
            cpu_only=True, requires_gpu=False,
        ),
        2: Tier(
            name="tier_2", label="torch CPU",
            paths=_tier_2_paths(),
            marker_expr="not gpu and not slow",
            timeout_s=args.tier_2_timeout,
            cpu_only=True, requires_gpu=False,
        ),
        3: Tier(
            name="tier_3", label="slow baselines",
            paths=_tier_3_paths(),
            marker_expr="slow",
            timeout_s=args.tier_3_timeout,
            cpu_only=False, requires_gpu=False,
        ),
        4: Tier(
            name="tier_4", label="GPU perf gates",
            paths=_tier_4_paths(),
            marker_expr="gpu",
            timeout_s=args.tier_4_timeout,
            cpu_only=False, requires_gpu=True,
        ),
    }
    return [all_tiers[i] for i in selected if i in all_tiers]


def _resolve_tier_selection(args: argparse.Namespace) -> list[int]:
    if args.tiers:
        out: list[int] = []
        for tok in args.tiers.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                n = int(tok)
            except ValueError as e:
                raise SystemExit(f"--tiers: bad token {tok!r}") from e
            if n not in (1, 2, 3, 4):
                raise SystemExit(f"--tiers: {n} not in {{1,2,3,4}}")
            if n not in out:
                out.append(n)
        return out
    mode = args.mode
    return {
        "fast": [1, 2],
        "full": [1, 2, 3],
        "gpu":  [1, 2, 4],
        "slow": [1, 3],
        "all":  [1, 2, 3, 4],
    }[mode]


# ===== pytest output parsing ===============================================

_SUMMARY_LINE_RE = re.compile(r"=+\s+.*=+\s*$")
_KIND_RES: dict[str, re.Pattern[str]] = {
    "passed":   re.compile(r"(\d+)\s+passed"),
    "failed":   re.compile(r"(\d+)\s+failed"),
    "errored":  re.compile(r"(\d+)\s+error"),  # "1 error" or "3 errors"
    "skipped":  re.compile(r"(\d+)\s+skipped"),
    "xfailed":  re.compile(r"(\d+)\s+xfailed"),
    "warnings": re.compile(r"(\d+)\s+warning"),
}


def parse_pytest_summary(stdout: str) -> dict[str, int]:
    """Parse the pytest summary line out of captured stdout."""
    counts = {k: 0 for k in _KIND_RES}
    lines = stdout.splitlines()[-50:]
    summary_line = None
    for ln in reversed(lines):
        if _SUMMARY_LINE_RE.match(ln) and ("passed" in ln or "failed" in ln
                                            or "error" in ln or "skipped" in ln):
            summary_line = ln
            break
    if summary_line is None:
        return counts
    for kind, pat in _KIND_RES.items():
        m = pat.search(summary_line)
        if m:
            counts[kind] = int(m.group(1))
    return counts


def _returncode_to_status(returncode: int | None, counts: dict[str, int]) -> str:
    if returncode is None:
        return "timeout"
    if returncode == 0:
        # Pytest returns 0 when nothing was collected only if --co is also passed;
        # for real runs, 0 always means tests passed.
        return "passed"
    if returncode == 1:
        return "failed"
    if returncode == 2:
        return "errored"
    if returncode == 5:
        return "empty"
    if returncode < 0:
        return "errored"
    return "errored"


# ===== Pre-flight collect-only =============================================

def _build_env(tier: Tier) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    env["TF_ENABLE_ONEDNN_OPTS"] = "0"
    if tier.cpu_only:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def run_collect_only(report_dir: pathlib.Path, timeout_s: int) -> tuple[str, float, pathlib.Path]:
    cmd = [
        sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q",
        "-W", "ignore::pytest.PytestUnknownMarkWarning",
    ]
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["CUDA_VISIBLE_DEVICES"] = ""
    out_path = report_dir / "collect_only.txt"
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd, env=env, cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=timeout_s, check=False,
        )
        elapsed = time.monotonic() - t0
        body = (result.stdout or "") + "\n--- stderr ---\n" + (result.stderr or "")
        out_path.write_text(body, encoding="utf-8")
        status = "passed" if result.returncode == 0 else "failed"
        return status, elapsed, out_path
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        out_path.write_text(f"collect-only timed out after {timeout_s} s\n", encoding="utf-8")
        return "timeout", elapsed, out_path


# ===== Tier run ============================================================

def run_tier(
    tier: Tier,
    report_dir: pathlib.Path,
    *,
    verbose: bool,
    pytest_extra: list[str],
    use_pytest_timeout: bool,
) -> TierResult:
    stdout_path = report_dir / f"{tier.name}.txt"
    stderr_path = report_dir / f"{tier.name}.stderr.txt"
    cmd_path = report_dir / f"{tier.name}.cmd.txt"
    paths = [p for p in tier.paths if (REPO_ROOT / p).exists()]
    missing = [p for p in tier.paths if not (REPO_ROOT / p).exists()]
    cmd = [
        sys.executable, "-m", "pytest",
        *paths,
        "-m", tier.marker_expr,
        "-W", "ignore::pytest.PytestUnknownMarkWarning",
        "-W", "ignore::DeprecationWarning",
        "-W", "ignore::pytest.PytestDeprecationWarning",
        "-r", "fEsxX",
    ]
    if use_pytest_timeout and tier.timeout_s:
        cmd += [f"--timeout={tier.timeout_s}"]
    cmd += pytest_extra
    cmd_path.write_text("\n".join(cmd) + "\n", encoding="utf-8")
    env = _build_env(tier)

    if not paths:
        return TierResult(
            tier=tier, command=cmd, returncode=5, status="empty",
            counts={k: 0 for k in _KIND_RES}, duration_s=0.0,
            stdout_path=stdout_path, stderr_path=stderr_path,
            notes=f"all paths missing: {missing}" if missing else "no paths",
        )

    print(f"[{tier.name}] {tier.label}: launching ({len(paths)} paths, marker={tier.marker_expr!r})",
          flush=True)
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd, env=env, cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=tier.timeout_s, check=False,
        )
        elapsed = time.monotonic() - t0
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        if verbose:
            print(stdout, flush=True)
            if stderr:
                print(stderr, file=sys.stderr, flush=True)
        _write_capped(stdout_path, stdout)
        _write_capped(stderr_path, stderr)
        counts = parse_pytest_summary(stdout)
        status = _returncode_to_status(result.returncode, counts)
        notes = None
        if missing:
            notes = f"skipped missing paths: {missing}"
        return TierResult(
            tier=tier, command=cmd, returncode=result.returncode,
            status=status, counts=counts, duration_s=elapsed,
            stdout_path=stdout_path, stderr_path=stderr_path,
            notes=notes,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - t0
        stdout = exc.stdout.decode("utf-8", "replace") if exc.stdout else ""
        stderr = exc.stderr.decode("utf-8", "replace") if exc.stderr else ""
        _write_capped(stdout_path, stdout + f"\n--- TIMEOUT after {tier.timeout_s} s ---\n")
        _write_capped(stderr_path, stderr)
        counts = parse_pytest_summary(stdout)
        return TierResult(
            tier=tier, command=cmd, returncode=None, status="timeout",
            counts=counts, duration_s=elapsed,
            stdout_path=stdout_path, stderr_path=stderr_path,
            notes=f"per-tier subprocess timeout {tier.timeout_s} s reached",
        )


def _write_capped(path: pathlib.Path, body: str) -> None:
    encoded = body.encode("utf-8")
    if len(encoded) > MAX_STDOUT_BYTES:
        head = encoded[: MAX_STDOUT_BYTES // 2]
        tail = encoded[-MAX_STDOUT_BYTES // 2:]
        sep = b"\n\n--- TRUNCATED at 10 MB ---\n\n"
        body_bytes = head + sep + tail
        path.write_bytes(body_bytes)
    else:
        path.write_text(body, encoding="utf-8")


# ===== Summary writers =====================================================

def write_summary_json(
    report_dir: pathlib.Path,
    results: list[TierResult],
    meta: dict,
    collect_meta: dict,
    interrupted: bool,
) -> None:
    rollup_counts = {k: 0 for k in _KIND_RES}
    tiers_failed: list[str] = []
    tiers_errored: list[str] = []
    tiers_timed_out: list[str] = []
    for r in results:
        for k in rollup_counts:
            rollup_counts[k] += r.counts.get(k, 0)
        if r.status == "failed":
            tiers_failed.append(r.tier.name)
        elif r.status == "errored":
            tiers_errored.append(r.tier.name)
        elif r.status == "timeout":
            tiers_timed_out.append(r.tier.name)
    if interrupted:
        overall = "interrupted"
    elif tiers_failed or tiers_errored or tiers_timed_out:
        overall = "failed"
    else:
        overall = "passed"
    payload: dict = {
        **meta,
        "collect_only": collect_meta,
        "tiers": {
            r.tier.name: {
                "label": r.tier.label,
                "paths": list(r.tier.paths),
                "marker_expr": r.tier.marker_expr,
                "command": r.command,
                "returncode": r.returncode,
                "status": r.status,
                "counts": r.counts,
                "duration_s": round(r.duration_s, 3),
                "stdout_path": str(r.stdout_path) if r.stdout_path else None,
                "stderr_path": str(r.stderr_path) if r.stderr_path else None,
                "notes": r.notes,
            }
            for r in results
        },
        "rollup": {
            **rollup_counts,
            "tiers_failed": tiers_failed,
            "tiers_errored": tiers_errored,
            "tiers_timed_out": tiers_timed_out,
            "overall_status": overall,
        },
    }
    (report_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8",
    )


def write_summary_md(report_dir: pathlib.Path, results: list[TierResult]) -> None:
    lines = [
        "| Tier | Label | Status | Passed | Failed | Errored | Skipped | Duration |",
        "|------|-------|--------|--------|--------|---------|---------|----------|",
    ]
    for r in results:
        c = r.counts
        lines.append(
            f"| {r.tier.name} | {r.tier.label} | {r.status.upper()} | "
            f"{c.get('passed', 0)} | {c.get('failed', 0)} | {c.get('errored', 0)} | "
            f"{c.get('skipped', 0)} | {r.duration_s:.1f} s |"
        )
    (report_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_console_table(results: list[TierResult]) -> str:
    rows = [
        ("Tier", "Label", "Status", "P", "F", "E", "S", "Duration"),
        ("----", "-----", "------", "-", "-", "-", "-", "--------"),
    ]
    for r in results:
        c = r.counts
        rows.append((
            r.tier.name, r.tier.label, r.status.upper(),
            str(c.get("passed", 0)), str(c.get("failed", 0)),
            str(c.get("errored", 0)), str(c.get("skipped", 0)),
            f"{r.duration_s:.1f}s",
        ))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    out_lines = [
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        for row in rows
    ]
    return "\n".join(out_lines)


def write_env_json(report_dir: pathlib.Path) -> None:
    info: dict = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "executable": sys.executable,
        "repo_root": str(REPO_ROOT),
    }
    try:
        import torch  # type: ignore
        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        info["torch_version"] = None
        info["cuda_available"] = False
        info["cuda_device_count"] = 0
    (report_dir / "env.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


# ===== Report dir resolution ===============================================

def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _resolve_report_dir(arg_value: pathlib.Path | None) -> pathlib.Path:
    if arg_value is None:
        d = DEFAULT_REPORT_ROOT / _utc_stamp()
    else:
        d = arg_value
    if d.exists() and any(d.iterdir()):
        i = 1
        while True:
            sibling = d.with_name(f"{d.name}_{i:03d}")
            if not sibling.exists():
                d = sibling
                break
            i += 1
    d.mkdir(parents=True, exist_ok=True)
    return d


# ===== CLI =================================================================

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python hyper_mve/scripts/run_full_tests.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", choices=("fast", "full", "gpu", "slow", "all"),
                   default="all", help="default tier selection (overridden by --tiers)")
    p.add_argument("--report-dir", type=pathlib.Path, default=None,
                   help="output dir; default runs/test_reports/<utc-stamp>")
    p.add_argument("--tier-1-timeout", type=int, default=180,
                   help="seconds (default 180)")
    p.add_argument("--tier-2-timeout", type=int, default=1800,
                   help="seconds (default 1800)")
    p.add_argument("--tier-3-timeout", type=int, default=4800,
                   help="seconds (default 4800)")
    p.add_argument("--tier-4-timeout", type=int, default=1200,
                   help="seconds (default 1200)")
    p.add_argument("--collect-only-timeout", type=int, default=120,
                   help="seconds (default 120)")
    p.add_argument("--skip-collect-check", action="store_true",
                   help="skip the pre-flight pytest --collect-only health check")
    p.add_argument("--pytest-extra", default="",
                   help="extra tokens forwarded to every tier (space-separated)")
    p.add_argument("--tiers", default="",
                   help='comma-separated tier ids, e.g. "1,3"; overrides --mode')
    p.add_argument("--fail-fast", action="store_true",
                   help="abort remaining tiers on the first failed tier")
    p.add_argument("--verbose", action="store_true",
                   help="mirror pytest output to console live")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report_dir = _resolve_report_dir(args.report_dir)
    print(f"[run_full_tests] report dir: {report_dir}", flush=True)
    write_env_json(report_dir)
    pytest_timeout_available = importlib.util.find_spec("pytest_timeout") is not None
    if not pytest_timeout_available:
        (report_dir / "notes.txt").write_text(
            "pytest-timeout not installed; per-test timeouts disabled.\n"
            "Per-tier subprocess timeouts are still enforced.\n",
            encoding="utf-8",
        )
    pytest_extra = [t for t in args.pytest_extra.split() if t]
    tiers = build_tiers(args)
    if not tiers:
        print("[run_full_tests] no tiers selected; nothing to do.", file=sys.stderr)
        return 1

    meta = {
        "report_dir": str(report_dir),
        "mode": args.mode,
        "tiers_selected": [int(t.name.split("_")[1]) for t in tiers],
        "started_at_iso8601": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pytest_timeout_available": pytest_timeout_available,
        "pytest_extra": pytest_extra,
    }

    # Pre-flight collect-only health check.
    collect_meta: dict = {"status": "skipped", "duration_s": 0.0, "stdout_path": None}
    if not args.skip_collect_check:
        c_status, c_dur, c_path = run_collect_only(report_dir, args.collect_only_timeout)
        collect_meta = {"status": c_status, "duration_s": round(c_dur, 3),
                        "stdout_path": str(c_path)}
        print(f"[collect-only] {c_status} in {c_dur:.1f} s", flush=True)
        if c_status != "passed":
            meta["completed_at_iso8601"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            meta["total_duration_s"] = round(c_dur, 3)
            write_summary_json(report_dir, [], meta, collect_meta, interrupted=False)
            print(f"[run_full_tests] collect-only failed; see {c_path}", file=sys.stderr)
            return 2

    results: list[TierResult] = []
    interrupted = False
    t_total = time.monotonic()
    try:
        for tier in tiers:
            res = run_tier(
                tier, report_dir,
                verbose=args.verbose, pytest_extra=pytest_extra,
                use_pytest_timeout=pytest_timeout_available,
            )
            results.append(res)
            print(f"[{tier.name}] {res.status.upper()} "
                  f"(passed={res.counts.get('passed', 0)}, "
                  f"failed={res.counts.get('failed', 0)}, "
                  f"errored={res.counts.get('errored', 0)}, "
                  f"skipped={res.counts.get('skipped', 0)}) "
                  f"in {res.duration_s:.1f}s", flush=True)
            if args.fail_fast and res.status not in ("passed", "empty"):
                print(f"[run_full_tests] --fail-fast: stopping after {tier.name}", flush=True)
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\n[run_full_tests] interrupted; writing partial summary…", file=sys.stderr)

    meta["completed_at_iso8601"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta["total_duration_s"] = round(time.monotonic() - t_total, 3)
    write_summary_json(report_dir, results, meta, collect_meta, interrupted=interrupted)
    write_summary_md(report_dir, results)
    print()
    print(render_console_table(results))
    print(f"\n[run_full_tests] summary: {report_dir / 'summary.json'}", flush=True)

    if interrupted:
        return 130
    if any(r.status not in ("passed", "empty") for r in results):
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
