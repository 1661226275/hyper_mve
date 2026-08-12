"""Liveness and lock-ownership invariants for the GPU job scheduler.

Both invariants below were violated in the v7 wave and both failures were
*silent* — no crash, no error line, just cards sitting idle:

1. ``scripts/train_queue.py`` launches its jobs with ``subprocess.Popen`` and
   never waits on them, so a finished child stays in the process table as a
   zombie. ``os.kill(pid, 0)`` **succeeds** on a zombie, so the original
   liveness test reported every one of the scheduler's own finished jobs as
   still running. The queue stalled permanently the moment its first job
   exited, and only ever advanced because the scheduler was being restarted by
   hand (which orphans the children to init, where they are reaped for real).

2. ``release_gpu`` unlinked the lock file unconditionally. With two queues
   overlapping in time, a scheduler restarted after one of its jobs had ended
   would reap that job and delete a lock another scheduler had since taken for
   a different run on the same card — re-opening exactly the double-booking
   window the lock exists to close.

These tests use real processes rather than mocks: the bug was in what the OS
does with an unwaited child, which a mocked pid cannot reproduce.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import train_queue                                                      # noqa: E402


def _wait_zombie(proc, timeout=10.0):
    """Let `proc` exit *without* waiting on it, so it becomes a real zombie."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with open(f"/proc/{proc.pid}/stat", "rb") as fh:
            state = fh.read().rsplit(b")", 1)[1].split()[0]
        if state == b"Z":
            return
        time.sleep(0.05)
    pytest.fail("child never became a zombie")


@pytest.mark.skipif(not os.path.isdir("/proc"), reason="needs procfs")
def test_alive_reports_a_zombie_as_dead():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        _wait_zombie(proc)
        # The pid still exists, so the naive os.kill(pid, 0) probe passes...
        os.kill(proc.pid, 0)
        # ...but the job behind it is over, and alive() must say so.
        assert train_queue.alive(proc.pid) is False
    finally:
        proc.wait()


def test_alive_reports_a_running_process_as_alive():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert train_queue.alive(proc.pid) is True
    finally:
        proc.kill()
        proc.wait()


def test_alive_is_false_for_missing_and_empty_pids():
    assert train_queue.alive(None) is False
    assert train_queue.alive(0) is False
    assert train_queue.alive(2 ** 22 - 1) is False        # above pid_max


def test_release_gpu_leaves_a_lock_another_job_now_owns(tmp_path):
    lockdir = tmp_path / "locks"
    assert train_queue.acquire_gpu(lockdir, 3)
    train_queue.set_gpu_owner(lockdir, 3, 4242)            # current owner

    train_queue.release_gpu(lockdir, 3, 1111)              # a stale reaper
    assert train_queue._lock_path(lockdir, 3).exists(), (
        "a finished job released a card that had already been re-locked "
        "by a different run")

    train_queue.release_gpu(lockdir, 3, 4242)              # the true owner
    assert not train_queue._lock_path(lockdir, 3).exists()


def test_release_gpu_without_a_pid_still_forces_the_unlink(tmp_path):
    lockdir = tmp_path / "locks"
    assert train_queue.acquire_gpu(lockdir, 5)
    train_queue.set_gpu_owner(lockdir, 5, 4242)
    train_queue.release_gpu(lockdir, 5)                    # shutdown path
    assert not train_queue._lock_path(lockdir, 5).exists()


def test_acquire_gpu_is_exclusive_then_steals_a_dead_holder(tmp_path):
    lockdir = tmp_path / "locks"
    assert train_queue.acquire_gpu(lockdir, 7) is True
    assert train_queue.acquire_gpu(lockdir, 7) is False    # second queue loses

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()                                            # pid now truly gone
    train_queue.set_gpu_owner(lockdir, 7, proc.pid)
    assert train_queue.acquire_gpu(lockdir, 7) is True, (
        "a scheduler killed mid-wave would strand its cards forever")


def test_queue_advances_past_its_own_finished_jobs(tmp_path):
    """End-to-end: the stall reproducer, on fake GPU indices.

    Two slots, four jobs. Before the zombie fix the scheduler launched the
    first two and then polled forever, because it was the parent of both.
    """
    out = tmp_path / "out"
    out.mkdir()
    jobs = [{"name": f"j{i}",
             "cmd": ["bash", "-c", f"echo gpu={{gpu}} > {out}/j{i}.txt"],
             "expect": f"{out}/j{i}.txt"} for i in range(4)]
    qpath = tmp_path / "q.json"
    qpath.write_text(json.dumps(jobs))
    state = tmp_path / "state.json"

    # GPU indices that cannot exist, so busy_gpus() never reports them busy and
    # the test can never collide with a real training run on this host.
    proc = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "scripts", "train_queue.py"),
         "--queue", str(qpath), "--state", str(state),
         "--gpus", "97,98", "--lockdir", str(tmp_path / "locks"),
         "--logdir", str(tmp_path / "logs"), "--poll", "1", "--cooldown", "0"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=90)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "all jobs finished" in proc.stdout
    done = [j for j in json.loads(state.read_text())["jobs"]
            if j["status"] == "done"]
    assert len(done) == 4, proc.stdout
    for i in range(4):
        assert (out / f"j{i}.txt").exists()
