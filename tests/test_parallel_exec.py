"""Tests for parallel execution helpers."""

import threading

from src.analysis.parallel_exec import default_worker_count, pool_chunksize, run_parallel_map


def test_default_worker_count_positive():
    assert default_worker_count() >= 1


def test_pool_chunksize_scales_with_tasks():
    small = pool_chunksize(10, workers=4)
    large = pool_chunksize(10_000, workers=16)
    assert small >= 1
    assert large >= small


def test_run_parallel_map_honours_cancel_event():
    cancel = threading.Event()
    tasks = list(range(20))
    seen: list[int] = []

    for i, value in enumerate(
        run_parallel_map(
            lambda x: x * 2,
            tasks,
            use_parallel=False,
            cancel_event=cancel,
        )
    ):
        seen.append(value)
        if i >= 2:
            cancel.set()

    assert len(seen) < 20
    assert len(seen) >= 3
