#!/usr/bin/env python
"""GPU-scheduling job queue: run a list of training jobs as GPUs free up.

Why this exists
---------------
A wave is more jobs than GPUs, and the GPUs do not free up together — a happo
run finishes in under an hour while mamba holds its card for ~19. Launching by
hand either idles cards or oversubscribes them. This keeps a candidate GPU pool
saturated: poll, find genuinely idle cards, start the next pending job, repeat
until the queue drains.

It deliberately decides idleness from **nvidia-smi compute apps**, not from its
own bookkeeping alone, so it also absorbs cards freed by runs it did not launch
(e.g. a separately-launched tier-1 wave). Two guards stop it from
double-booking a card: a cooldown after each launch (a fresh process takes tens
of seconds to allocate, and would otherwise still look idle on the next poll),
and its own live PIDs are always treated as busy.

Crash-safety: state is written to disk after every transition, so the queue can
be inspected while running and resumed after a restart — jobs already `done`
are never re-run.

Usage::

    # write a queue file (list of job dicts), then:
    setsid nohup python scripts/train_queue.py \
        --queue scripts/grids/v7_tier2_queue.json \
        --gpus 5,6,7,8,0,1,2,3,4 \
        --state results_v7_500k/queue_state.json \
        --logdir /path/to/logs > /path/to/queue.log 2>&1 &

    # check progress at any time:
    python scripts/train_queue.py --state results_v7_500k/queue_state.json --status
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PYTHON = sys.executable


def _gpu_index_by_uuid() -> dict:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True, text=True, check=True).stdout
    m = {}
    for line in out.strip().splitlines():
        idx, uuid = [p.strip() for p in line.split(",", 1)]
        m[uuid] = int(idx)
    return m


def busy_gpus() -> set:
    """GPU indices with at least one live compute process."""
    by_uuid = _gpu_index_by_uuid()
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
         "--format=csv,noheader"],
        capture_output=True, text=True, check=True).stdout
    busy = set()
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        uuid = line.split(",", 1)[0].strip()
        if uuid in by_uuid:
            busy.add(by_uuid[uuid])
    return busy


def job_expect(job: dict) -> str | None:
    """Path whose existence means the job really produced its artefact.

    Exit code alone is not enough for the training jobs (a run that dies after
    training but before eval exits 0 from some shells), and for a restarted
    scheduler there is no exit code at all -- the Popen handle died with the
    previous process. Explicit for ``cmd`` jobs; defaults to the eval report for
    train jobs.
    """
    if "expect" in job:
        return job["expect"] or None
    if "cmd" in job:
        return None
    return os.path.join(job["out"], _run_rel(job), "eval_report.json")


def job_ready(job: dict) -> bool:
    """False while an input this job needs has not been produced yet.

    Lets a NashConv job sit in the queue behind the training run whose
    checkpoint it consumes, instead of being launched by hand once that run
    happens to finish.
    """
    for path in job.get("requires") or ():
        if not os.path.exists(path):
            return False
    return True


def build_cmd(job: dict, gpu: int) -> list:
    if "cmd" in job:
        # `{gpu}` lets a job that takes its own GPU flag place it; if it uses no
        # placeholder the card is handed over via CUDA_VISIBLE_DEVICES instead
        # (see build_env), so both conventions work unchanged.
        cmd = [str(tok).format(gpu=gpu) for tok in job["cmd"]]
        # A bare "python" would resolve against PATH, which is not the conda env
        # this queue runs under -- the failure mode is a torch-less interpreter
        # several hours into a wave. Pin it to our own.
        if cmd and cmd[0] == "python":
            cmd[0] = PYTHON
        return cmd
    cmd = [
        PYTHON, os.path.join(REPO_ROOT, "scripts", "train.py"),
        "--algo", str(job["algo"]),
        "--env", str(job.get("env", "relation_coopmix")),
        "--seed", str(int(job.get("seed", 0))),
        "--total-env-steps", str(int(job.get("total_env_steps", 500_000))),
        "--episodes", str(int(job.get("episodes", 128))),
        "--gpus", str(gpu),
        "--out", str(job["out"]),
    ]
    if job.get("ablation") and job["ablation"] != "none":
        cmd += ["--ablation", str(job["ablation"])]
    if job.get("num_pmcts"):
        cmd += ["--num-pmcts", str(int(job["num_pmcts"]))]
    if job.get("env_steps_per_grad"):
        cmd += ["--env-steps-per-grad", str(int(job["env_steps_per_grad"]))]
    return cmd


def _lock_path(lockdir: pathlib.Path, gpu: int) -> pathlib.Path:
    return lockdir / f"gpu{gpu}.lock"


def acquire_gpu(lockdir: pathlib.Path, gpu: int) -> bool:
    """Claim `gpu` across schedulers, or return False.

    nvidia-smi idleness is not enough once more than one queue runs: a fresh
    process takes tens of seconds to allocate, so two schedulers polling the
    same second both see the card free and both launch on it. The per-GPU
    cooldown only protects a scheduler from itself.

    The lock holds the PID of the job that owns the card. A lock whose PID is
    gone is stale -- a scheduler killed mid-wave would otherwise strand its
    cards forever -- so it is stolen rather than respected.
    """
    lockdir.mkdir(parents=True, exist_ok=True)
    path = _lock_path(lockdir, gpu)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, b"pending")
        os.close(fd)
        return True
    except FileExistsError:
        try:
            holder = path.read_text().strip()
        except OSError:
            return False
        if holder and holder != "pending" and not alive(holder):
            try:
                path.unlink()
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, b"pending")
                os.close(fd)
                return True
            except (OSError, FileExistsError):
                return False
        return False


def set_gpu_owner(lockdir: pathlib.Path, gpu: int, pid: int) -> None:
    try:
        _lock_path(lockdir, gpu).write_text(str(int(pid)))
    except OSError:
        pass


def release_gpu(lockdir: pathlib.Path, gpu) -> None:
    if gpu is None:
        return
    try:
        _lock_path(lockdir, gpu).unlink()
    except OSError:
        pass


def build_env(job: dict, gpu: int) -> dict:
    env = dict(os.environ)
    if "cmd" in job and not any("{gpu}" in str(t) for t in job["cmd"]):
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def load_state(path: pathlib.Path, queue: list) -> dict:
    if path.exists():
        state = json.loads(path.read_text())
        known = {j["name"] for j in state["jobs"]}
        for j in queue:                      # allow appending to a live queue
            if j["name"] not in known:
                state["jobs"].append({**j, "status": "pending", "pid": None,
                                      "gpu": None, "started": None,
                                      "finished": None, "returncode": None})
        return state
    return {"jobs": [{**j, "status": "pending", "pid": None, "gpu": None,
                      "started": None, "finished": None, "returncode": None}
                     for j in queue]}


def save_state(path: pathlib.Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(path)


def alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def print_status(state: dict) -> int:
    order = {"running": 0, "pending": 1, "done": 2, "failed": 3}
    rows = sorted(state["jobs"], key=lambda j: (order.get(j["status"], 9),
                                                j["name"]))
    print(f"{'status':>8}  {'gpu':>3}  {'name':<52} {'elapsed':>9}")
    print("-" * 80)
    now = time.time()
    for j in rows:
        el = ""
        if j.get("started"):
            end = j.get("finished") or now
            el = f"{(end - j['started']) / 3600:.2f}h"
        # `or '-'` would print GPU 0 as '-': 0 is falsy.
        gpu = j.get("gpu")
        gpu_s = "-" if gpu is None else str(gpu)
        status = j["status"]
        if status == "pending" and not job_ready(j):
            status = "blocked"
        print(f"{status:>8}  {gpu_s:>3}  {j['name']:<52} {el:>9}")
    n = {k: sum(1 for j in state["jobs"] if j["status"] == k)
         for k in ("pending", "running", "done", "failed")}
    print(f"\npending {n['pending']}  running {n['running']}  "
          f"done {n['done']}  failed {n['failed']}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue", type=pathlib.Path, default=None,
                   help="JSON list of job dicts")
    p.add_argument("--state", type=pathlib.Path, required=True)
    p.add_argument("--gpus", default="5,6,7,8",
                   help="candidate GPU pool, in preference order")
    p.add_argument("--logdir", type=pathlib.Path, default=None)
    p.add_argument("--poll", type=int, default=60, help="seconds between polls")
    p.add_argument("--cooldown", type=int, default=300,
                   help="seconds a just-launched GPU is treated as busy")
    p.add_argument("--status", action="store_true",
                   help="print the state file and exit")
    p.add_argument("--lockdir", type=pathlib.Path, default=None,
                   help="cross-scheduler GPU locks (default: <state>/../.gpu_locks). "
                        "All concurrent queues MUST share one lockdir.")
    args = p.parse_args(argv)

    if args.status:
        return print_status(json.loads(args.state.read_text()))

    queue = json.loads(args.queue.read_text()) if args.queue else []
    state = load_state(args.state, queue)
    save_state(args.state, state)

    pool = [int(x) for x in args.gpus.split(",") if x.strip()]
    logdir = args.logdir or (args.state.parent / "logs")
    logdir.mkdir(parents=True, exist_ok=True)
    just_launched: dict = {}                 # gpu -> monotonic launch time
    handles: dict = {}                       # pid -> Popen, for exit codes
    lockdir = args.lockdir or (args.state.parent / ".gpu_locks")

    print(f"[queue] {len(state['jobs'])} jobs, pool {pool}, "
          f"poll {args.poll}s, cooldown {args.cooldown}s", flush=True)

    while True:
        # ---- reap finished jobs
        for j in state["jobs"]:
            if j["status"] == "running" and not alive(j["pid"]):
                j["finished"] = time.time()
                proc = handles.pop(j["pid"], None)
                rc = proc.poll() if proc is not None else None
                expect = job_expect(j)
                produced = expect is None or os.path.exists(expect)
                # An exit code is authoritative when we still have the handle;
                # after a scheduler restart there is none, so fall back to the
                # artefact. Both must agree when both are available.
                ok = produced and (rc in (0, None))
                j["returncode"] = rc
                j["status"] = "done" if ok else "failed"
                release_gpu(lockdir, j.get("gpu"))
                why = "" if ok else (
                    f" (rc={rc}, missing {expect})" if not produced
                    else f" (rc={rc})")
                print(f"[queue] {j['status'].upper()} {j['name']} "
                      f"(gpu {j['gpu']}){why}", flush=True)
                save_state(args.state, state)

        pending = [j for j in state["jobs"] if j["status"] == "pending"]
        running = [j for j in state["jobs"] if j["status"] == "running"]
        if not pending and not running:
            print("[queue] all jobs finished", flush=True)
            return 0

        # A job whose inputs do not exist yet is not launchable, but it is also
        # not skipped -- it is simply passed over until the run it depends on
        # produces them. Blocked jobs must not consume the GPU slot.
        blocked = [j for j in pending if not job_ready(j)]
        pending = [j for j in pending if job_ready(j)]
        if blocked and not running and not pending:
            print("[queue] DEADLOCK: only blocked jobs remain -- "
                  + ", ".join(f"{j['name']} needs {j.get('requires')}"
                              for j in blocked), flush=True)
            return 1

        # ---- find launchable GPUs
        if pending:
            now = time.monotonic()
            busy = busy_gpus()
            busy |= {j["gpu"] for j in running if j.get("gpu") is not None}
            busy |= {g for g, t in just_launched.items()
                     if now - t < args.cooldown}
            for gpu in pool:
                if not pending:
                    break
                if gpu in busy:
                    continue
                if not acquire_gpu(lockdir, gpu):
                    continue          # another scheduler got there first
                job = pending.pop(0)
                cmd = build_cmd(job, gpu)
                log = logdir / f"{job['name']}.log"
                with open(log, "ab") as fh:
                    proc = subprocess.Popen(
                        cmd, stdout=fh, stderr=subprocess.STDOUT,
                        cwd=REPO_ROOT, start_new_session=True,
                        env=build_env(job, gpu))
                handles[proc.pid] = proc
                set_gpu_owner(lockdir, gpu, proc.pid)
                job.update(status="running", pid=proc.pid, gpu=gpu,
                           started=time.time())
                just_launched[gpu] = time.monotonic()
                busy.add(gpu)
                print(f"[queue] START {job['name']} on gpu {gpu} "
                      f"pid {proc.pid}\n         {shlex.join(cmd)}", flush=True)
                save_state(args.state, state)

        time.sleep(args.poll)


def _run_rel(job: dict) -> str:
    """`<algo><_arm>/<env>/seed<k>` — mirrors scripts/train.py's run_dir."""
    arm = job.get("ablation") or "none"
    suffix = "" if arm in ("", "none") else f"_{arm}"
    return os.path.join(f"{job['algo']}{suffix}",
                        str(job.get("env", "relation_coopmix")),
                        f"seed{int(job.get('seed', 0))}")


if __name__ == "__main__":
    raise SystemExit(main())
