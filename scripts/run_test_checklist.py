#!/usr/bin/env python3
"""Automated runner for the Pkg-01..03 test checklist (Linux server).

Executes every phase listed in ``TEST_CHECKLIST_pkg1-3.md`` (§0..§3) by shelling
out to pytest and the smoke / synthetic scripts that already live in
``hyper_mve/scripts/``. Per-test results come from JUnit XML so we can identify
which "hard gates" (H1..H13) passed without re-implementing the assertions.

Designed for headless Linux runs:

- Resolves the repo root by walking up from this file (so it works regardless of
  where you invoke it from). Override with ``--repo-root``.
- Picks up the active ``python`` (i.e. ``sys.executable``); honour any venv that
  was activated before launching this script.
- Streams pytest / smoke output live AND captures it to per-phase logs in
  ``--out-dir`` (defaults to ``runs/test_checklist/<timestamp>/``).
- Writes ``report.json`` (machine-readable) and ``report.md`` (human-readable)
  at the end. Exit code = 0 iff every required phase + every hard gate passed.

Typical use::

    # Full run on the server (≈3-10 min, GPU optional)
    python scripts/run_test_checklist.py

    # Fast smoke pass (cuts smoke to 50 ep, synth to 500 steps; CPU-friendly)
    python scripts/run_test_checklist.py --quick

    # Skip the synthetic-convergence gate (it's the slow one)
    python scripts/run_test_checklist.py --skip-synth

    # Only run schemas + configs + envs unit tests, no scripts
    python scripts/run_test_checklist.py --phases 0,1,2 --skip-smoke

    # CI mode: keep going even if a phase fails, but exit non-zero at the end
    python scripts/run_test_checklist.py --keep-going
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Hard gates (H1..H13) per checklist §4 — each maps to one or more JUnit
# (classname, name) patterns. Substring match keeps it robust to pytest's
# parametrize suffixes.
# ---------------------------------------------------------------------------
HARD_GATES: list[dict] = [
    {
        "id": "H1",
        "title": "Table 3.5.4 four quadrants",
        "patterns": [("test_rewards", "test_table_3_5_4_")],
        "min_matches": 4,
    },
    {
        "id": "H2",
        "title": "Ch4.1.1 partial-derivative four quadrants (autograd)",
        "patterns": [("test_rewards", "test_beta_grad_four_quadrants")],
        "min_matches": 1,
    },
    {
        "id": "H3",
        "title": "Logistic regen 5-step analytic match (±1e-4)",
        "patterns": [("test_dynamics", "test_logistic_regen_5steps_against_analytic")],
        "min_matches": 1,
    },
    {
        "id": "H4",
        "title": "Self-Info: type/cap don't leak into obs / role",
        "patterns": [
            ("test_observations", "test_other_agents_type_does_not_leak_into_obs"),
            ("test_observations", "test_other_agents_cap_does_not_leak_into_obs"),
            ("test_role_encoder", "test_self_info_severity_only_own_type"),
        ],
        "min_matches": 3,
    },
    {
        "id": "H5",
        "title": "env.info Oracle fields (c_true / types / caps)",
        "patterns": [
            ("test_env_info_oracle", "test_info_has_oracle_fields"),
            ("test_env_info_oracle", "test_info_oracle_matches_state"),
        ],
        "min_matches": 2,
    },
    {
        "id": "H6",
        "title": "1000-episode random rollout: no NaN/Inf",
        "patterns": [("smoke_rollout", "smoke_rollout_")],   # synthesized below
        "min_matches": 1,
    },
    {
        "id": "H7",
        "title": "TriContextEncoder output dim = 80 (16+32+32)",
        "patterns": [
            ("test_tri_context", "test_output_dim"),
            ("test_tri_context", "test_d_ctx_aug_property"),
        ],
        "min_matches": 2,
    },
    {
        "id": "H8",
        "title": "head_c is scalar (B, N) — v4 vs v3 critical",
        "patterns": [("test_belief_net", "test_head_c_is_scalar_per_agent_v4")],
        "min_matches": 1,
    },
    {
        "id": "H9",
        "title": "head_opp shape (B, N, N-1, 2) softmax",
        "patterns": [
            ("test_belief_net", "test_head_opp_output_shape"),
            ("test_belief_net", "test_softmax_simplex_property"),
        ],
        "min_matches": 2,
    },
    {
        "id": "H10",
        "title": "z_hat ordering convention (3 sites consistent)",
        "patterns": [
            ("test_buffer_record", "test_z_hat_order_convention"),
            ("test_belief_net", "test_z_hat_agent_id_order_convention"),
            ("test_belief_losses", "test_l_opp_index_mapping_correctness"),
            ("test_belief_losses", "test_l_opp_index_mapping_specific_pairs"),
        ],
        "min_matches": 4,
    },
    {
        "id": "H11",
        "title": "L_div hinge prevents collapse",
        "patterns": [
            ("test_belief_losses", "test_l_div_hinge_zero_when_high_variance"),
            ("test_belief_losses", "test_l_div_hinge_positive_when_low_variance"),
            ("test_belief_losses", "test_l_div_collapse_protection"),
        ],
        "min_matches": 3,
    },
    {
        "id": "H12",
        "title": "Synthetic convergence: head_c MSE / head_opp acc / Var",
        "patterns": [("synth_belief", "synth_belief_")],     # synthesized below
        "min_matches": 1,
    },
    {
        "id": "H13",
        "title": "Presets vs Ch3.9 Table (Easy / Medium / Hard)",
        "patterns": [("test_presets", None)],     # all tests in test_presets.py
        "min_matches": 1,
    },
]


@dataclass
class TestCase:
    classname: str
    name: str
    time: float
    status: str       # "passed" | "failed" | "error" | "skipped"
    message: str = ""


@dataclass
class PhaseResult:
    name: str
    command: list[str]
    returncode: int
    duration_s: float
    log_path: str
    junit_path: Optional[str] = None
    tests: list[TestCase] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""
    extras: dict = field(default_factory=dict)   # smoke / synth metrics

    @property
    def passed(self) -> bool:
        if self.skipped:
            return True
        return self.returncode == 0


# ---------------------------------------------------------------------------
# Repo / venv resolution
# ---------------------------------------------------------------------------
def find_repo_root(explicit: Optional[str]) -> Path:
    """Locate the ``hyper_mve`` repo root.

    Strategy: prefer the ``--repo-root`` override; otherwise walk up from this
    file looking for ``hyper_mve/__init__.py`` and ``tests/``.
    """
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not (p / "hyper_mve" / "__init__.py").exists():
            sys.exit(f"[fatal] --repo-root {p} doesn't contain hyper_mve/__init__.py")
        return p
    here = Path(__file__).resolve()
    for cand in [here.parent, *here.parents]:
        if (cand / "hyper_mve" / "__init__.py").exists() and (cand / "tests").exists():
            return cand
    sys.exit(
        "[fatal] cannot locate repo root (looked for hyper_mve/__init__.py + tests/ "
        "above this script). Pass --repo-root."
    )


# ---------------------------------------------------------------------------
# Subprocess helper — streams stdout/stderr to console AND captures to a file.
# ---------------------------------------------------------------------------
def run_command(
    cmd: list[str],
    cwd: Path,
    log_path: Path,
    env: dict,
    quiet: bool = False,
) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pretty = " ".join(shlex.quote(c) for c in cmd)
    if not quiet:
        print(f"\n$ {pretty}")
        print(f"  cwd : {cwd}")
        print(f"  log : {log_path}")
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(f"# command: {pretty}\n# cwd: {cwd}\n# started: {datetime.now().isoformat()}\n\n")
        logf.flush()
        # Merge stderr into stdout so the log preserves chronological order.
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            logf.write(line)
            if not quiet:
                sys.stdout.write(line)
        rc = proc.wait()
    duration = time.perf_counter() - start
    if not quiet:
        print(f"  -> exit {rc} in {duration:.1f}s")
    return rc, duration


# ---------------------------------------------------------------------------
# JUnit parsing — pytest's ``--junitxml`` schema.
# ---------------------------------------------------------------------------
def parse_junit(xml_path: Path) -> list[TestCase]:
    if not xml_path.exists():
        return []
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return []
    cases: list[TestCase] = []
    for tc in tree.iter("testcase"):
        classname = tc.attrib.get("classname", "")
        name = tc.attrib.get("name", "")
        try:
            t = float(tc.attrib.get("time", "0") or 0.0)
        except ValueError:
            t = 0.0
        status = "passed"
        message = ""
        for tag in ("failure", "error"):
            child = tc.find(tag)
            if child is not None:
                status = "failed" if tag == "failure" else "error"
                message = (child.attrib.get("message", "") or "")[:500]
                break
        else:
            if tc.find("skipped") is not None:
                status = "skipped"
        cases.append(TestCase(classname, name, t, status, message))
    return cases


# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------
def phase_import_sanity(repo: Path, env: dict, out_dir: Path) -> PhaseResult:
    """§0.1 — verify the package imports cleanly before running any tests."""
    code = (
        "import torch, numpy, hyper_mve;"
        "from hyper_mve.schemas import AgentType, CapabilityVector, ObservationLayout,"
        " TimeStepRecord, ContextSchema;"
        "from hyper_mve.envs.resource_commons import ResourceCommonsEnv;"
        "from hyper_mve.models.tri_context_encoder import TriContextEncoder;"
        "from hyper_mve.models.belief_net import BeliefNet;"
        "from hyper_mve.models.belief_losses import belief_loss;"
        "print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available());"
        "print('imports OK')"
    )
    cmd = [sys.executable, "-c", code]
    log = out_dir / "phase0_imports.log"
    rc, dt = run_command(cmd, repo, log, env)
    return PhaseResult(
        name="§0 import sanity",
        command=cmd,
        returncode=rc,
        duration_s=dt,
        log_path=str(log),
    )


def phase_pytest(
    name: str,
    target: str,
    repo: Path,
    env: dict,
    out_dir: Path,
    extra_args: Optional[list[str]] = None,
) -> PhaseResult:
    junit = out_dir / f"{name}.junit.xml"
    log = out_dir / f"{name}.log"
    cmd = [
        sys.executable, "-m", "pytest", target,
        "-v",
        "--tb=short",
        f"--junitxml={junit}",
    ]
    if extra_args:
        cmd.extend(extra_args)
    rc, dt = run_command(cmd, repo, log, env)
    cases = parse_junit(junit)
    return PhaseResult(
        name=name,
        command=cmd,
        returncode=rc,
        duration_s=dt,
        log_path=str(log),
        junit_path=str(junit),
        tests=cases,
    )


def phase_smoke(
    preset: str,
    n_episodes: int,
    repo: Path,
    env: dict,
    out_dir: Path,
) -> PhaseResult:
    name = f"§2.10 smoke {preset} {n_episodes}ep"
    log = out_dir / f"smoke_{preset}.log"
    cmd = [
        sys.executable, "hyper_mve/scripts/test_resource_commons.py",
        preset, str(n_episodes),
    ]
    rc, dt = run_command(cmd, repo, log, env)
    extras = parse_smoke_log(log)
    # Synthesize a TestCase so hard gate H6 can match it.
    synth = TestCase(
        classname="smoke_rollout",
        name=f"smoke_rollout_{preset}",
        time=dt,
        status="passed" if rc == 0 else "failed",
        message="" if rc == 0 else "non-zero exit",
    )
    return PhaseResult(
        name=name,
        command=cmd,
        returncode=rc,
        duration_s=dt,
        log_path=str(log),
        tests=[synth],
        extras=extras,
    )


def parse_smoke_log(log_path: Path) -> dict:
    """Pull mean step latency / steps-per-sec / mean return from the smoke log."""
    metrics: dict = {}
    if not log_path.exists():
        return metrics
    for raw in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("avg step (ms)"):
            metrics["avg_step_ms"] = _parse_float_after_colon(line)
        elif line.startswith("steps / sec"):
            metrics["steps_per_sec"] = _parse_float_after_colon(line)
        elif line.startswith("mean episode ret"):
            metrics["mean_episode_return"] = _parse_float_after_colon(line)
        elif line.startswith("nan/inf episodes"):
            metrics["nan_inf_episodes"] = _parse_float_after_colon(line)
        elif line.startswith("total steps"):
            metrics["total_steps"] = _parse_float_after_colon(line)
    return metrics


def _parse_float_after_colon(line: str) -> float:
    try:
        return float(line.split(":", 1)[1].strip().split()[0])
    except (IndexError, ValueError):
        return float("nan")


def phase_synth(
    repo: Path,
    env: dict,
    out_dir: Path,
    quick: bool,
    extra_args: Optional[list[str]] = None,
) -> PhaseResult:
    name = "§3.8 BeliefNet synthetic convergence"
    log = out_dir / "synth_belief.log"
    cmd = [
        sys.executable, "hyper_mve/scripts/test_belief_net_synth.py",
    ]
    if quick:
        cmd.append("--quick")
    if extra_args:
        cmd.extend(extra_args)
    rc, dt = run_command(cmd, repo, log, env)
    extras = parse_synth_log(log)
    synth = TestCase(
        classname="synth_belief",
        name="synth_belief_run",
        time=dt,
        status="passed" if rc == 0 else "failed",
        message="" if rc == 0 else "non-zero exit",
    )
    return PhaseResult(
        name=name,
        command=cmd,
        returncode=rc,
        duration_s=dt,
        log_path=str(log),
        tests=[synth],
        extras=extras,
    )


def parse_synth_log(log_path: Path) -> dict:
    """Pull head_c MSE / head_opp accuracy / hidden variance from the synth log.

    The script prints lines like ``[PASS] head_c_mse: 0.0234 < 0.0500`` — we
    capture the leading scalar of each metric, plus the final RESULT line.
    """
    metrics: dict = {}
    if not log_path.exists():
        return metrics
    for raw in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("[PASS]") or s.startswith("[FAIL]"):
            tag = s[:6]
            body = s[6:].strip()
            if ":" in body:
                metric_name, rest = body.split(":", 1)
                value = _first_float(rest)
                metrics[metric_name.strip()] = {"value": value, "tag": tag.strip("[]")}
        elif s.startswith("RESULT:"):
            metrics["overall"] = s.split(":", 1)[1].strip()
    return metrics


def _first_float(s: str) -> float:
    for tok in s.replace(",", " ").split():
        try:
            return float(tok)
        except ValueError:
            continue
    return float("nan")


# ---------------------------------------------------------------------------
# Hard-gate evaluation
# ---------------------------------------------------------------------------
def evaluate_gates(all_tests: list[TestCase]) -> list[dict]:
    """For each gate, find matching tests and decide pass/fail."""
    gates: list[dict] = []
    for spec in HARD_GATES:
        matched: list[TestCase] = []
        for tc in all_tests:
            for cls_substr, name_substr in spec["patterns"]:
                cls_ok = (cls_substr is None) or (cls_substr in tc.classname)
                name_ok = (name_substr is None) or (name_substr in tc.name)
                if cls_ok and name_ok:
                    matched.append(tc)
                    break
        passed_matches = [m for m in matched if m.status == "passed"]
        failed = [m for m in matched if m.status in ("failed", "error")]
        ok = (len(passed_matches) >= spec["min_matches"]) and not failed
        gates.append({
            "id": spec["id"],
            "title": spec["title"],
            "min_matches": spec["min_matches"],
            "matched_total": len(matched),
            "matched_passed": len(passed_matches),
            "matched_failed": len(failed),
            "failed_tests": [
                {"classname": m.classname, "name": m.name, "message": m.message}
                for m in failed
            ],
            "passed": ok,
        })
    return gates


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def write_reports(
    out_dir: Path,
    phases: list[PhaseResult],
    gates: list[dict],
    meta: dict,
) -> tuple[Path, Path]:
    """Write report.json + report.md side by side."""
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"

    payload = {
        "meta": meta,
        "phases": [
            {
                **{k: v for k, v in dataclasses.asdict(p).items() if k != "tests"},
                "passed": p.passed,
                "tests": [dataclasses.asdict(tc) for tc in p.tests],
            }
            for p in phases
        ],
        "hard_gates": gates,
        "summary": summarize(phases, gates),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines: list[str] = []
    md_lines.append(f"# Test Checklist Report — {meta['started_at']}")
    md_lines.append("")
    md_lines.append("## Environment")
    for k, v in meta.items():
        md_lines.append(f"- **{k}**: {v}")
    md_lines.append("")

    summary = payload["summary"]
    md_lines.append("## Summary")
    md_lines.append(
        f"- phases: {summary['phases_passed']}/{summary['phases_total']} passed"
    )
    md_lines.append(
        f"- tests:  {summary['tests_passed']}/{summary['tests_total']} passed "
        f"({summary['tests_failed']} failed, {summary['tests_skipped']} skipped)"
    )
    md_lines.append(
        f"- hard gates: {summary['gates_passed']}/{summary['gates_total']} passed"
    )
    md_lines.append(f"- total wall-clock: {summary['total_seconds']:.1f}s")
    md_lines.append("")

    md_lines.append("## Phases")
    md_lines.append("")
    md_lines.append("| Phase | Status | Duration | Tests P/F/S | Notes |")
    md_lines.append("|-------|--------|---------:|-------------|-------|")
    for p in phases:
        if p.skipped:
            status = "SKIPPED"
            notes = p.skip_reason or "—"
            counts = "—"
        else:
            status = "PASS" if p.passed else "FAIL"
            counts_p = sum(1 for t in p.tests if t.status == "passed")
            counts_f = sum(1 for t in p.tests if t.status in ("failed", "error"))
            counts_s = sum(1 for t in p.tests if t.status == "skipped")
            counts = f"{counts_p}/{counts_f}/{counts_s}"
            notes = ", ".join(f"{k}={v}" for k, v in p.extras.items()) or f"rc={p.returncode}"
        md_lines.append(
            f"| {p.name} | {status} | {p.duration_s:.1f}s | {counts} | {notes} |"
        )
    md_lines.append("")

    md_lines.append("## Hard Gates")
    md_lines.append("")
    md_lines.append("| ID | Title | Status | Matched (P/F/T) | Required |")
    md_lines.append("|----|-------|--------|-----------------|----------|")
    for g in gates:
        st = "PASS" if g["passed"] else "FAIL"
        counts = f"{g['matched_passed']}/{g['matched_failed']}/{g['matched_total']}"
        md_lines.append(
            f"| {g['id']} | {g['title']} | {st} | {counts} | ≥{g['min_matches']} |"
        )
    md_lines.append("")

    failed_tests = [
        (p, t) for p in phases for t in p.tests
        if t.status in ("failed", "error")
    ]
    if failed_tests:
        md_lines.append("## Failures")
        md_lines.append("")
        for p, t in failed_tests:
            md_lines.append(f"- `{t.classname}::{t.name}` ({p.name})")
            if t.message:
                md_lines.append(f"  - {t.message}")
        md_lines.append("")

    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return json_path, md_path


def summarize(phases: list[PhaseResult], gates: list[dict]) -> dict:
    tests_total = sum(len(p.tests) for p in phases)
    tests_passed = sum(1 for p in phases for t in p.tests if t.status == "passed")
    tests_failed = sum(1 for p in phases for t in p.tests if t.status in ("failed", "error"))
    tests_skipped = sum(1 for p in phases for t in p.tests if t.status == "skipped")
    return {
        "phases_total": len(phases),
        "phases_passed": sum(1 for p in phases if p.passed),
        "tests_total": tests_total,
        "tests_passed": tests_passed,
        "tests_failed": tests_failed,
        "tests_skipped": tests_skipped,
        "gates_total": len(gates),
        "gates_passed": sum(1 for g in gates if g["passed"]),
        "total_seconds": sum(p.duration_s for p in phases),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-root", default=None,
        help="Path to hyper_mve repo root (default: auto-detect from this script).",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Where to write logs + reports (default: runs/test_checklist/<timestamp>/).",
    )
    parser.add_argument(
        "--phases", default="0,1,2,3",
        help="Comma-separated phases to run (subset of 0,1,2,3).",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Cut smoke episodes + synth steps for a fast smoke pass.",
    )
    parser.add_argument(
        "--smoke-presets", default="easy,medium,hard",
        help="Smoke presets to run in §2.10 (comma-separated).",
    )
    parser.add_argument(
        "--smoke-episodes", type=int, default=1000,
        help="Episodes per preset for §2.10 smoke (default 1000; --quick → 50).",
    )
    parser.add_argument(
        "--skip-smoke", action="store_true",
        help="Skip §2.10 smoke rollouts.",
    )
    parser.add_argument(
        "--skip-synth", action="store_true",
        help="Skip §3.8 BeliefNet synthetic convergence.",
    )
    parser.add_argument(
        "--keep-going", action="store_true",
        help="Run every requested phase even if an earlier one fails.",
    )
    parser.add_argument(
        "--pytest-args", default="",
        help="Extra args appended to every pytest invocation (quote them).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = find_repo_root(args.repo_root)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = repo / "runs" / "test_checklist" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Inherit the parent env, but make sure ``hyper_mve`` is importable from cwd.
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(repo), env.get("PYTHONPATH", "")]))

    requested = {p.strip() for p in args.phases.split(",") if p.strip()}
    extra_pytest = shlex.split(args.pytest_args) if args.pytest_args else []
    smoke_presets = [p.strip() for p in args.smoke_presets.split(",") if p.strip()]
    smoke_episodes = 50 if args.quick else args.smoke_episodes

    print("=" * 70)
    print(f"Test checklist runner — {timestamp}")
    print(f"  repo     : {repo}")
    print(f"  out      : {out_dir}")
    print(f"  python   : {sys.executable}")
    print(f"  phases   : {sorted(requested)}")
    print(f"  quick    : {args.quick}")
    print(f"  smoke    : {'skipped' if args.skip_smoke else f'{smoke_presets} x {smoke_episodes}ep'}")
    print(f"  synth    : {'skipped' if args.skip_synth else 'enabled'}")
    print("=" * 70)

    phases: list[PhaseResult] = []
    aborted = False

    def gate(p: PhaseResult, *, required: bool = True) -> bool:
        """Append phase, return True if we should continue to the next."""
        phases.append(p)
        if p.passed or args.keep_going or not required:
            return True
        nonlocal aborted
        aborted = True
        print(f"\n[abort] {p.name} failed (rc={p.returncode}); pass --keep-going to continue.")
        return False

    if "0" in requested and not aborted:
        if not gate(phase_import_sanity(repo, env, out_dir)):
            pass

    if "1" in requested and not aborted:
        if not gate(phase_pytest("phase1_schemas", "tests/schemas/", repo, env, out_dir, extra_pytest)):
            pass
        if not aborted and not gate(phase_pytest("phase1_configs", "tests/configs/", repo, env, out_dir, extra_pytest)):
            pass
        if not aborted and not gate(phase_pytest("phase1_legacy_warning", "tests/test_legacy_import_warning.py", repo, env, out_dir, extra_pytest), required=False):
            pass

    if "2" in requested and not aborted:
        if not gate(phase_pytest("phase2_envs", "tests/envs/", repo, env, out_dir, extra_pytest)):
            pass
        if not aborted and not args.skip_smoke:
            for preset in smoke_presets:
                if not gate(phase_smoke(preset, smoke_episodes, repo, env, out_dir)):
                    break

    if "3" in requested and not aborted:
        if not gate(phase_pytest("phase3_models", "tests/models/", repo, env, out_dir, extra_pytest)):
            pass
        if not aborted and not args.skip_synth:
            gate(phase_synth(repo, env, out_dir, args.quick))

    # Mark anything we never reached as skipped (so the report is honest).
    if aborted:
        print("\n[note] some phases were skipped due to earlier failure.")

    all_tests = [t for p in phases for t in p.tests]
    gates = evaluate_gates(all_tests)

    meta = {
        "started_at": timestamp,
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "repo_root": str(repo),
        "out_dir": str(out_dir),
        "args": vars(args),
    }
    json_path, md_path = write_reports(out_dir, phases, gates, meta)

    summary = summarize(phases, gates)
    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"  phases  : {summary['phases_passed']}/{summary['phases_total']}")
    print(f"  tests   : {summary['tests_passed']} passed, "
          f"{summary['tests_failed']} failed, {summary['tests_skipped']} skipped "
          f"({summary['tests_total']} total)")
    print(f"  gates   : {summary['gates_passed']}/{summary['gates_total']}")
    print(f"  time    : {summary['total_seconds']:.1f}s wall-clock")
    print(f"  report  : {md_path}")
    print(f"  json    : {json_path}")

    failed_gates = [g for g in gates if not g["passed"]]
    if failed_gates:
        print("\nFailed gates:")
        for g in failed_gates:
            print(f"  - {g['id']} {g['title']}: "
                  f"{g['matched_passed']}/{g['matched_total']} matched, "
                  f"{g['matched_failed']} failed")

    overall_ok = (
        all(p.passed for p in phases)
        and not failed_gates
        and not aborted
    )
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
