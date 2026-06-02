#!/usr/bin/env python3
"""Automated runner for the Pkg-04 + Pkg-05 test checklist (Linux server).

Companion to ``scripts/run_test_checklist.py`` (Pkg-01..03). Runs the model
(Pkg-04) and trainer/worker (Pkg-05) pytest suites plus two smoke phases, parses
JUnit XML, evaluates the documented hard gates (Pkg-04 C1..R8 + Pkg-05 C5-* / R5-2),
and writes ``report.json`` / ``report.md``.

Phases:
    §0  import sanity        — torch + the 7-API model + the 5 Pkg-05 entry points
    §4a Pkg-04 model pytest  — tests/models/test_{hyper_network_v2,hyper_muzero_model,
                               grad_gating,stability_safeguards,forward_performance}.py
    §4b Pkg-04 forward smoke — hyper_mve/scripts/test_hyper_model_forward.py (no NaN + perf)
    §5a Pkg-05 training      — tests/training/
    §5b Pkg-05 planning      — tests/planning/
    §5c Pkg-05 migration     — tests/migration/ (set_context grep gate)
    §5d Pkg-05 e2e smoke     — scripts/train_main.py with tiny overrides (full
                               collect -> store -> sample -> train_step pipeline)

Designed for headless Linux: resolves the repo root by walking up, uses the active
``python`` (``sys.executable``; honour your venv), streams + captures output.

Typical use::

    python scripts/run_test_checklist_pkg4_5.py                 # full run (GPU optional)
    python scripts/run_test_checklist_pkg4_5.py --quick         # fast CPU smoke
    python scripts/run_test_checklist_pkg4_5.py --pytest-args "-m 'not gpu'"
    python scripts/run_test_checklist_pkg4_5.py --phases 5 --skip-e2e
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Hard gates — each maps a documented constraint to (classname, name) substring
# patterns. classname is pytest's dotted path (so the test FILE name is a robust
# substring); name=None matches every test in the file.
# ---------------------------------------------------------------------------
HARD_GATES: list[dict] = [
    # ---- Pkg-04 (DualHyperNetwork v2 + HyperMuZeroModel) ----
    {"id": "C1", "title": "hyper_trans uses c_ctx only (role-invariant)",
     "patterns": [("test_hyper_network_v2", "test_hyper_trans_only_c_ctx"),
                  ("test_hyper_network_v2", "test_hyper_trans_invariant_under_role_change")],
     "min_matches": 2},
    {"id": "C2", "title": "subjective hyper input dim = 80",
     "patterns": [("test_hyper_network_v2", "test_subjective_input_dim_80")], "min_matches": 1},
    {"id": "C5d", "title": "ctx_aug dim == 80 (16+32+32)",
     "patterns": [("test_hyper_network_v2", "test_ctx_aug_dim_eq_80")], "min_matches": 1},
    {"id": "C4", "title": "belief grad gating (pre-5k detached / post-5k flows)",
     "patterns": [("test_grad_gating", "test_pre_5k_belief_detached"),
                  ("test_grad_gating", "test_post_5k_belief_grad_flow")], "min_matches": 2},
    {"id": "C6", "title": "StateTransNet predicts delta-s (residual)",
     "patterns": [("test_stability_safeguards", "test_state_trans_delta_s_residual")], "min_matches": 1},
    {"id": "C7", "title": "AdaLN (1 + gamma) residual modulation",
     "patterns": [("test_stability_safeguards", "test_adaln_one_plus_gamma_factor")], "min_matches": 1},
    {"id": "C8", "title": "output_scale inits (trans/rew/pred)",
     "patterns": [("test_stability_safeguards", "test_output_scale_inits_trans_rew_pred")], "min_matches": 1},
    {"id": "C9", "title": "type-aware reward differentiation (assertion A)",
     "patterns": [("test_hyper_muzero_model", "test_type_aware_reward_differentiation")], "min_matches": 1},
    {"id": "C11", "title": "Self-Info: no oracle types leak into set_context_subjective",
     "patterns": [("test_hyper_muzero_model", "test_set_context_subjective_no_oracle_types_leak")], "min_matches": 1},
    {"id": "API7", "title": "model exposes exactly the 7 public APIs",
     "patterns": [("test_hyper_muzero_model", "test_model_has_7_public_apis")], "min_matches": 1},
    {"id": "R8", "title": "model forward smoke < 15 ms (CPU-safe variant)",
     "patterns": [("test_forward_performance", "test_forward_smoke_under_15ms")], "min_matches": 1},

    # ---- Pkg-05 (Trainer / Worker / Buffer / Curriculum / Loss / Planner) ----
    {"id": "C5-T1", "title": "trainer calls update_step once per train_step",
     "patterns": [("test_trainer_loop", "test_trainer_calls_update_step_per_step")], "min_matches": 1},
    {"id": "C5-T2", "title": "set_context_objective once per unroll",
     "patterns": [("test_trainer_loop", "test_objective_called_once_per_unroll")], "min_matches": 1},
    {"id": "C5-T3", "title": "set_context_subjective per agent",
     "patterns": [("test_trainer_loop", "test_subjective_called_per_agent")], "min_matches": 1},
    {"id": "C5-W1", "title": "worker never calls update_step",
     "patterns": [("test_worker", "test_worker_no_update_step")], "min_matches": 1},
    {"id": "C5-W2", "title": "worker uses BeliefNet.step online inference",
     "patterns": [("test_worker", "test_worker_uses_belief_net_step")], "min_matches": 1},
    {"id": "C5-W3", "title": "TimeStepRecord z_hat shape / field validity",
     "patterns": [("test_worker", "test_z_hat_shape_matches_pkg01_spec04"),
                  ("test_worker", "test_collect_episode_record_fields_valid")], "min_matches": 2},
    {"id": "C5-B1", "title": "buffer z_hat order preserved end-to-end",
     "patterns": [("test_episode_buffer", "test_buffer_z_hat_order_e2e")], "min_matches": 1},
    {"id": "C5-B2", "title": "stratified sampling min-per-type fraction",
     "patterns": [("test_episode_buffer", "test_stratified_min_per_type_frac")], "min_matches": 1},
    {"id": "C5-S1", "title": "curriculum 3-stage boundaries",
     "patterns": [("test_curriculum", "test_curriculum_stage_boundaries")], "min_matches": 1},
    {"id": "C5-S2", "title": "Stage 1 full oracle weight = 1.0",
     "patterns": [("test_curriculum", "test_stage_1_full_oracle")], "min_matches": 1},
    {"id": "C5-S3", "title": "Stage 2 anneal monotone decreasing",
     "patterns": [("test_curriculum", "test_oracle_mixing_anneal_monotonic")], "min_matches": 1},
    {"id": "C5-L1", "title": "lambda_b curve matches cfg",
     "patterns": [("test_loss_composition", "test_lambda_b_curve_matches_cfg")], "min_matches": 1},
    {"id": "C5-L2", "title": "belief gradient double-path (pre-5k isolated / post-5k both)",
     "patterns": [("test_loss_composition", "test_belief_gradient_isolation_pre_5k"),
                  ("test_loss_composition", "test_belief_gradient_both_sources_post_5k")], "min_matches": 2},
    {"id": "C5-P1", "title": "MVE CRN deterministic under same seed",
     "patterns": [("test_mve_planner", "test_crn_step0_deterministic_same_seed")], "min_matches": 1},
    {"id": "C5-P2", "title": "planner 4 set_context sites migrated",
     "patterns": [("test_mve_planner", "test_planner_4_set_context_migrated")], "min_matches": 1},
    {"id": "C5-E1", "title": "EMA tau=0.99 decay correctness",
     "patterns": [("test_ema_scheduler", "test_ema_decay_correctness_tau_099")], "min_matches": 1},
    {"id": "C5-E2", "title": "warmup_cosine LR curve",
     "patterns": [("test_ema_scheduler", "test_lr_warmup_then_cosine_anneal")], "min_matches": 1},
    {"id": "C5-I1", "title": "v4.7 -> v4 set_context migration grep (0 residual)",
     "patterns": [("test_pkg05_v47_to_v4_set_context", "test_no_legacy_set_context_calls"),
                  ("test_pkg05_v47_to_v4_set_context", "test_no_legacy_belief_apis")], "min_matches": 2},
    {"id": "R5-2", "title": "sample_batch < 50 ms",
     "patterns": [("test_episode_buffer", "test_sample_batch_under_50ms")], "min_matches": 1},

    # ---- synthesized smoke gates ----
    {"id": "S04", "title": "Pkg-04 forward smoke (100 steps no NaN + perf)",
     "patterns": [("smoke_hyper", "smoke_hyper_")], "min_matches": 1},
    {"id": "S05", "title": "Pkg-05 e2e train pipeline (collect->store->sample->train_step)",
     "patterns": [("e2e_train", "e2e_train_")], "min_matches": 1},
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
    extras: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        if self.skipped:
            return True
        return self.returncode == 0


# ---------------------------------------------------------------------------
# Repo resolution + subprocess + JUnit parsing (shared with the pkg1-3 runner).
# ---------------------------------------------------------------------------
def find_repo_root(explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not (p / "hyper_mve" / "__init__.py").exists():
            sys.exit(f"[fatal] --repo-root {p} doesn't contain hyper_mve/__init__.py")
        return p
    here = Path(__file__).resolve()
    for cand in [here.parent, *here.parents]:
        if (cand / "hyper_mve" / "__init__.py").exists() and (cand / "tests").exists():
            return cand
    sys.exit("[fatal] cannot locate repo root (hyper_mve/__init__.py + tests/). Pass --repo-root.")


def run_command(cmd: list[str], cwd: Path, log_path: Path, env: dict, quiet: bool = False) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pretty = " ".join(shlex.quote(c) for c in cmd)
    if not quiet:
        print(f"\n$ {pretty}\n  cwd : {cwd}\n  log : {log_path}")
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(f"# command: {pretty}\n# cwd: {cwd}\n# started: {datetime.now().isoformat()}\n\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
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
        status, message = "passed", ""
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
# Phases
# ---------------------------------------------------------------------------
def phase_import_sanity(repo: Path, env: dict, out_dir: Path) -> PhaseResult:
    code = (
        "import torch, numpy, hyper_mve;"
        "from hyper_mve.models import HyperMuZeroModel;"
        "from hyper_mve.training import MuZeroTrainer, Worker, EpisodeReplayBuffer;"
        "from hyper_mve.training.curriculum import CurriculumScheduler;"
        "from hyper_mve.training.loss_composition import compose_total_loss;"
        "from hyper_mve.planning.mve_planner import MVEPlanner;"
        "print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available());"
        "print('pkg04+05 imports OK')"
    )
    cmd = [sys.executable, "-c", code]
    log = out_dir / "phase0_imports.log"
    rc, dt = run_command(cmd, repo, log, env)
    return PhaseResult("§0 import sanity", cmd, rc, dt, str(log))


def phase_pytest(name: str, targets: list[str], repo: Path, env: dict, out_dir: Path,
                 extra_args: Optional[list[str]] = None) -> PhaseResult:
    junit = out_dir / f"{name}.junit.xml"
    log = out_dir / f"{name}.log"
    cmd = [sys.executable, "-m", "pytest", *targets, "-v", "--tb=short", f"--junitxml={junit}"]
    if extra_args:
        cmd.extend(extra_args)
    rc, dt = run_command(cmd, repo, log, env)
    return PhaseResult(name, cmd, rc, dt, str(log), junit_path=str(junit), tests=parse_junit(junit))


def phase_hyper_smoke(repo: Path, env: dict, out_dir: Path, quick: bool) -> PhaseResult:
    name = "§4b Pkg-04 forward smoke"
    log = out_dir / "smoke_hyper.log"
    steps = "20" if quick else "100"
    cmd = [sys.executable, "hyper_mve/scripts/test_hyper_model_forward.py",
           "--preset", "medium", "--steps", steps, "--batch_size", "64"]
    rc, dt = run_command(cmd, repo, log, env)
    synth = TestCase("smoke_hyper", "smoke_hyper_medium", dt,
                     "passed" if rc == 0 else "failed", "" if rc == 0 else "non-zero exit")
    return PhaseResult(name, cmd, rc, dt, str(log), tests=[synth])


def phase_e2e_train(repo: Path, env: dict, out_dir: Path, quick: bool) -> PhaseResult:
    name = "§5d Pkg-05 e2e train pipeline"
    log = out_dir / "e2e_train.log"
    steps = "4" if quick else "10"
    cmd = [sys.executable, "hyper_mve/scripts/train_main.py",
           "--preset", "medium", "--variant", "hyper", "--max_steps", steps,
           "--override", "train.min_buffer_size=2",
           "--override", "train.episodes_per_iter=1",
           "--override", "train.train_steps_per_iter=2",
           "--override", "train.batch_size=8",
           "--override", "env.T_max=40",
           "--ckpt_dir", str(out_dir / "e2e_ckpt"), "--seed", "0"]
    rc, dt = run_command(cmd, repo, log, env)
    synth = TestCase("e2e_train", "e2e_train_medium", dt,
                     "passed" if rc == 0 else "failed", "" if rc == 0 else "non-zero exit")
    return PhaseResult(name, cmd, rc, dt, str(log), tests=[synth])


# ---------------------------------------------------------------------------
# Gate evaluation + reports (same logic as the pkg1-3 runner).
# ---------------------------------------------------------------------------
def evaluate_gates(all_tests: list[TestCase]) -> list[dict]:
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
            "id": spec["id"], "title": spec["title"], "min_matches": spec["min_matches"],
            "matched_total": len(matched), "matched_passed": len(passed_matches),
            "matched_failed": len(failed),
            "failed_tests": [{"classname": m.classname, "name": m.name, "message": m.message} for m in failed],
            "passed": ok,
        })
    return gates


def summarize(phases: list[PhaseResult], gates: list[dict]) -> dict:
    return {
        "phases_total": len(phases),
        "phases_passed": sum(1 for p in phases if p.passed),
        "tests_total": sum(len(p.tests) for p in phases),
        "tests_passed": sum(1 for p in phases for t in p.tests if t.status == "passed"),
        "tests_failed": sum(1 for p in phases for t in p.tests if t.status in ("failed", "error")),
        "tests_skipped": sum(1 for p in phases for t in p.tests if t.status == "skipped"),
        "gates_total": len(gates),
        "gates_passed": sum(1 for g in gates if g["passed"]),
        "total_seconds": sum(p.duration_s for p in phases),
    }


def write_reports(out_dir: Path, phases: list[PhaseResult], gates: list[dict], meta: dict) -> tuple[Path, Path]:
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    summary = summarize(phases, gates)
    payload = {
        "meta": meta,
        "phases": [{**{k: v for k, v in dataclasses.asdict(p).items() if k != "tests"},
                    "passed": p.passed, "tests": [dataclasses.asdict(tc) for tc in p.tests]} for p in phases],
        "hard_gates": gates,
        "summary": summary,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [f"# Pkg-04/05 Test Checklist Report — {meta['started_at']}", "", "## Environment"]
    md += [f"- **{k}**: {v}" for k, v in meta.items()]
    md += ["", "## Summary",
           f"- phases: {summary['phases_passed']}/{summary['phases_total']} passed",
           f"- tests:  {summary['tests_passed']}/{summary['tests_total']} passed "
           f"({summary['tests_failed']} failed, {summary['tests_skipped']} skipped)",
           f"- hard gates: {summary['gates_passed']}/{summary['gates_total']} passed",
           f"- total wall-clock: {summary['total_seconds']:.1f}s", "",
           "## Phases", "", "| Phase | Status | Duration | Tests P/F/S | Notes |",
           "|-------|--------|---------:|-------------|-------|"]
    for p in phases:
        if p.skipped:
            md.append(f"| {p.name} | SKIPPED | — | — | {p.skip_reason or '—'} |")
        else:
            cp = sum(1 for t in p.tests if t.status == "passed")
            cf = sum(1 for t in p.tests if t.status in ("failed", "error"))
            cs = sum(1 for t in p.tests if t.status == "skipped")
            notes = ", ".join(f"{k}={v}" for k, v in p.extras.items()) or f"rc={p.returncode}"
            md.append(f"| {p.name} | {'PASS' if p.passed else 'FAIL'} | {p.duration_s:.1f}s | {cp}/{cf}/{cs} | {notes} |")
    md += ["", "## Hard Gates", "", "| ID | Title | Status | Matched (P/F/T) | Required |",
           "|----|-------|--------|-----------------|----------|"]
    for g in gates:
        st = "N/A" if g["matched_total"] == 0 else ("PASS" if g["passed"] else "FAIL")
        md.append(f"| {g['id']} | {g['title']} | {st} | "
                  f"{g['matched_passed']}/{g['matched_failed']}/{g['matched_total']} | ≥{g['min_matches']} |")
    md.append("")
    failures = [(p, t) for p in phases for t in p.tests if t.status in ("failed", "error")]
    if failures:
        md += ["## Failures", ""]
        for p, t in failures:
            md.append(f"- `{t.classname}::{t.name}` ({p.name})")
            if t.message:
                md.append(f"  - {t.message}")
        md.append("")
    md_path.write_text("\n".join(md), encoding="utf-8")
    return json_path, md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", default=None, help="hyper_mve repo root (default: auto-detect).")
    parser.add_argument("--out-dir", default=None, help="logs + reports dir (default: runs/test_checklist_pkg4_5/<ts>/).")
    parser.add_argument("--phases", default="4,5", help="Comma-separated phases (subset of 4,5).")
    parser.add_argument("--quick", action="store_true", help="Cut smoke/e2e steps for a fast CPU pass.")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip §4b Pkg-04 forward smoke.")
    parser.add_argument("--skip-e2e", action="store_true", help="Skip §5d Pkg-05 e2e train pipeline.")
    parser.add_argument("--keep-going", action="store_true", default=True,
                        help="Continue when a phase fails (default True; --strict to disable).")
    parser.add_argument("--strict", dest="keep_going", action="store_false", help="Abort on first phase failure.")
    parser.add_argument("--pytest-args", default="", help="Extra args appended to each pytest call (quote them).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = find_repo_root(args.repo_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir \
        else repo / "runs" / "test_checklist_pkg4_5" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(repo), env.get("PYTHONPATH", "")]))
    requested = {p.strip() for p in args.phases.split(",") if p.strip()}
    extra_pytest = shlex.split(args.pytest_args) if args.pytest_args else []

    print("=" * 70)
    print(f"Pkg-04/05 test checklist runner — {timestamp}")
    print(f"  repo   : {repo}\n  out    : {out_dir}\n  python : {sys.executable}")
    print(f"  phases : {sorted(requested)}  quick={args.quick}")
    print("=" * 70)

    phases: list[PhaseResult] = []
    aborted = False

    def gate(p: PhaseResult, *, required: bool = True) -> None:
        nonlocal aborted
        phases.append(p)
        if not p.passed and required and not args.keep_going:
            aborted = True
            print(f"\n[abort] {p.name} failed (rc={p.returncode}); pass --keep-going to continue.")

    # §0 import sanity always runs first and always aborts on failure.
    p_import = phase_import_sanity(repo, env, out_dir)
    phases.append(p_import)
    if not p_import.passed:
        aborted = True
        print(f"\n[abort] import sanity failed (rc={p_import.returncode}); fix the env first.")

    pkg04_targets = [f"tests/models/{f}.py" for f in (
        "test_hyper_network_v2", "test_hyper_muzero_model", "test_grad_gating",
        "test_stability_safeguards", "test_forward_performance",
    )]

    if "4" in requested and not aborted:
        gate(phase_pytest("phase4a_pkg04_models", pkg04_targets, repo, env, out_dir, extra_pytest))
        if not aborted and not args.skip_smoke:
            gate(phase_hyper_smoke(repo, env, out_dir, args.quick))

    if "5" in requested and not aborted:
        gate(phase_pytest("phase5a_pkg05_training", ["tests/training/"], repo, env, out_dir, extra_pytest))
        if not aborted:
            gate(phase_pytest("phase5b_pkg05_planning", ["tests/planning/"], repo, env, out_dir, extra_pytest))
        if not aborted:
            gate(phase_pytest("phase5c_pkg05_migration", ["tests/migration/"], repo, env, out_dir, extra_pytest))
        if not aborted and not args.skip_e2e:
            gate(phase_e2e_train(repo, env, out_dir, args.quick))

    all_tests = [t for p in phases for t in p.tests]
    gates = evaluate_gates(all_tests)
    meta = {
        "started_at": timestamp, "host": platform.node(), "platform": platform.platform(),
        "python": sys.version.split()[0], "executable": sys.executable,
        "repo_root": str(repo), "out_dir": str(out_dir), "args": vars(args),
    }
    json_path, md_path = write_reports(out_dir, phases, gates, meta)

    s = summarize(phases, gates)
    print("\n" + "=" * 70 + "\nRESULT\n" + "=" * 70)
    print(f"  phases : {s['phases_passed']}/{s['phases_total']}")
    print(f"  tests  : {s['tests_passed']} passed, {s['tests_failed']} failed, "
          f"{s['tests_skipped']} skipped ({s['tests_total']} total)")
    print(f"  gates  : {s['gates_passed']}/{s['gates_total']}")
    print(f"  time   : {s['total_seconds']:.1f}s\n  report : {md_path}\n  json   : {json_path}")

    failed_gates = [g for g in gates if g["matched_total"] > 0 and not g["passed"]]
    skipped_gates = [g for g in gates if g["matched_total"] == 0]
    if failed_gates:
        print("\nFailed gates:")
        for g in failed_gates:
            print(f"  - {g['id']} {g['title']}: {g['matched_passed']}/{g['matched_total']} matched, "
                  f"{g['matched_failed']} failed")
    if skipped_gates:
        print(f"\nGates with no matching tests run ({len(skipped_gates)} — phase skipped/not reached):")
        for g in skipped_gates:
            print(f"  - {g['id']} {g['title']}")

    overall_ok = all(p.passed for p in phases) and not failed_gates and not aborted
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
