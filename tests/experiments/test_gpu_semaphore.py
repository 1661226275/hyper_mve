"""C8-ABL-ISO1 — GpuSemaphore (pkg-08 spec 05 §7)."""
from __future__ import annotations

import threading
import time

import pytest

from hyper_mve.experiments.sweep import GpuSemaphore


def test_gpu_semaphore_acquire_release_round_trip():
    sem = GpuSemaphore(2)
    a = sem.acquire()
    b = sem.acquire()
    assert {a, b} == {0, 1}
    sem.release(a)
    sem.release(b)


def test_gpu_semaphore_release_out_of_range_raises():
    sem = GpuSemaphore(2)
    with pytest.raises(ValueError):
        sem.release(99)
    with pytest.raises(ValueError):
        sem.release(-1)


def test_gpu_semaphore_blocks_until_release():
    """At most n_gpus tokens are held simultaneously."""
    sem = GpuSemaphore(2)
    held: list[int] = []
    blocked = threading.Event()
    released = threading.Event()

    def hold_then_release(idx: int) -> None:
        gid = sem.acquire()
        held.append(gid)
        if len(held) == 2:
            blocked.set()
        released.wait(timeout=2.0)
        sem.release(gid)

    threads = [threading.Thread(target=hold_then_release, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    assert blocked.wait(timeout=2.0)
    # Now all tokens are held; a third acquire should block.
    third_done = threading.Event()

    def third() -> None:
        gid = sem.acquire()
        third_done.set()
        sem.release(gid)

    third_thread = threading.Thread(target=third)
    third_thread.start()
    time.sleep(0.05)
    assert not third_done.is_set(), "third acquire should be blocked"
    released.set()
    for t in threads:
        t.join(timeout=2.0)
    third_thread.join(timeout=2.0)
    assert third_done.is_set()


def test_gpu_semaphore_invalid_n_gpus():
    with pytest.raises(ValueError):
        GpuSemaphore(0)
