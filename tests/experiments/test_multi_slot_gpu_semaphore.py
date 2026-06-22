"""C8-ABL-ISO1 extension — MultiSlotGpuSemaphore (pkg-08 spec 05 §7 + fast_300k launcher)."""
from __future__ import annotations

import threading
import time
from collections import Counter

import pytest

from hyper_mve.experiments.sweep import MultiSlotGpuSemaphore


def test_multi_slot_round_trip():
    sem = MultiSlotGpuSemaphore([2, 3, 4], slots_per_gpu=2)
    assert sem.n_gpus == 6
    assert sem.slots_per_gpu == 2
    assert sem.gpu_ids == (2, 3, 4)
    acquired: list[int] = [sem.acquire() for _ in range(6)]
    assert Counter(acquired) == Counter({2: 2, 3: 2, 4: 2})
    for g in acquired:
        sem.release(g)


def test_multi_slot_release_unknown_id_raises():
    sem = MultiSlotGpuSemaphore([2, 3], slots_per_gpu=1)
    g = sem.acquire()
    sem.release(g)
    with pytest.raises(ValueError):
        sem.release(0)
    with pytest.raises(ValueError):
        sem.release(99)


def test_multi_slot_invalid_construction():
    with pytest.raises(ValueError):
        MultiSlotGpuSemaphore([], slots_per_gpu=2)
    with pytest.raises(ValueError):
        MultiSlotGpuSemaphore([2, 3], slots_per_gpu=0)


def test_multi_slot_blocks_when_exhausted():
    sem = MultiSlotGpuSemaphore([5], slots_per_gpu=2)
    a = sem.acquire()
    b = sem.acquire()
    assert {a, b} == {5}  # both tokens are gpu 5
    third_done = threading.Event()
    seen: list[int] = []

    def third() -> None:
        gid = sem.acquire()
        seen.append(gid)
        third_done.set()
        sem.release(gid)

    t = threading.Thread(target=third)
    t.start()
    time.sleep(0.05)
    assert not third_done.is_set()
    sem.release(a)
    t.join(timeout=2.0)
    assert third_done.is_set()
    assert seen == [5]
    sem.release(b)


def test_multi_slot_slots_per_gpu_one_matches_single_token_semantics():
    """With slots_per_gpu=1, MultiSlot acquires each id exactly once."""
    sem = MultiSlotGpuSemaphore([4, 5, 6], slots_per_gpu=1)
    assert sem.n_gpus == 3
    a, b, c = sem.acquire(), sem.acquire(), sem.acquire()
    assert {a, b, c} == {4, 5, 6}
    for g in (a, b, c):
        sem.release(g)
