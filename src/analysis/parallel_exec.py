"""Shared multiprocessing helpers for analysis workloads."""

from __future__ import annotations

import multiprocessing
import os
import threading
from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar

from src.services.stock_config import stock_config

T = TypeVar("T")
R = TypeVar("R")


def default_worker_count() -> int:
    cfg = stock_config()
    try:
        cap = max(1, int(cfg.worker_count))
    except (TypeError, ValueError):
        cap = 4
    cpus = os.cpu_count() or 4
    return max(1, min(cap, cpus))


def pool_chunksize(task_count: int, workers: int | None = None) -> int:
    """Balance scheduling overhead vs load balancing for large universes."""
    w = workers or default_worker_count()
    if task_count <= w * 4:
        return 1
    return max(1, task_count // (w * 8))


def run_parallel_map(
    func: Callable[[T], R],
    tasks: Iterable[T],
    *,
    use_parallel: bool,
    workers: int | None = None,
    chunksize: int | None = None,
    initializer: Callable[..., None] | None = None,
    initargs: tuple = (),
    cancel_event: threading.Event | None = None,
) -> Iterator[R]:
    """Run func over tasks with optional multiprocessing pool."""
    task_list = list(tasks)
    if not task_list:
        return iter(())
    if not use_parallel or len(task_list) == 1:
        for task in task_list:
            if cancel_event is not None and cancel_event.is_set():
                return
            yield func(task)
        return

    w = workers or default_worker_count()
    cs = chunksize if chunksize is not None else pool_chunksize(len(task_list), w)
    pool = multiprocessing.Pool(
        processes=w,
        initializer=initializer,
        initargs=initargs,
    )
    cancelled = False
    try:
        for item in pool.imap_unordered(func, task_list, chunksize=cs):
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            yield item
    finally:
        if cancelled:
            pool.terminate()
        else:
            pool.close()
        pool.join()
