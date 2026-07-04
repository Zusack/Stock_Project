"""
Helpers for consuming LLM streams with cancel/timeout support.
Uses a worker thread so cancellation can interrupt slow backends.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Iterator, Optional

_SENTINEL_END = object()
_SENTINEL_ABORT = object()

DEFAULT_CHUNK_TIMEOUT = 0.5
DEFAULT_JOB_TIMEOUT_SEC = 3600


class JobTimeoutError(Exception):
    """Raised when a single prompt/job exceeds the allowed runtime."""


class ErrorCallInterrupted(Exception):
    """Raised when the user cancels mid-stream."""


def iterate_stream(
    response_stream,
    *,
    cancel_event: Optional[threading.Event] = None,
    chunk_timeout: float = DEFAULT_CHUNK_TIMEOUT,
    job_timeout_sec: Optional[float] = None,
) -> Iterator:
    """
    Yield chunks from ``response_stream`` with optional cancel and job timeout.

    When ``cancel_event`` is set, the stream is cancelled and ``InterruptedError``
    is raised.
    """
    job_timeout = job_timeout_sec if job_timeout_sec is not None else DEFAULT_JOB_TIMEOUT_SEC
    job_start = time.monotonic()
    q: queue.Queue = queue.Queue()

    def worker():
        try:
            for ch in response_stream:
                q.put(ch)
            q.put(_SENTINEL_END)
        except Exception as e:
            q.put((_SENTINEL_ABORT, e))

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    while True:
        if cancel_event is not None and cancel_event.is_set():
            try:
                if hasattr(response_stream, "cancel"):
                    response_stream.cancel()
            except Exception:
                pass
            raise InterruptedError("Stopped by user")

        try:
            wait_sec = min(chunk_timeout, max(0.1, job_timeout - (time.monotonic() - job_start)))
            if wait_sec <= 0:
                _cancel_stream(response_stream)
                raise JobTimeoutError(
                    f"Job exceeded maximum runtime ({job_timeout}s)."
                )
            x = q.get(timeout=wait_sec)
        except queue.Empty:
            elapsed = time.monotonic() - job_start
            if elapsed >= job_timeout:
                _cancel_stream(response_stream)
                raise JobTimeoutError(
                    f"Job exceeded maximum runtime ({job_timeout}s)."
                )
            if cancel_event is not None and cancel_event.is_set():
                _cancel_stream(response_stream)
                raise InterruptedError("Stopped by user")
            continue

        if x is _SENTINEL_END:
            break
        if isinstance(x, tuple) and len(x) == 2 and x[0] is _SENTINEL_ABORT:
            raise x[1]
        if x is _SENTINEL_ABORT:
            raise RuntimeError("Stream aborted")

        if cancel_event is not None and cancel_event.is_set():
            _cancel_stream(response_stream)
            raise InterruptedError("Stopped by user")

        if (time.monotonic() - job_start) >= job_timeout:
            _cancel_stream(response_stream)
            raise JobTimeoutError(
                f"Job exceeded maximum runtime ({job_timeout}s)."
            )

        yield x


def _cancel_stream(response_stream) -> None:
    try:
        if hasattr(response_stream, "cancel"):
            response_stream.cancel()
    except Exception:
        pass
