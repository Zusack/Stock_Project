"""
Helpers for consuming LLM streams with thermal and control checks during long blocking inference.
Uses a worker thread and a timeout on the main side so we can run _check_control_flags (and thus
react to thermal pause/stop) even when the backend is slow to produce the next chunk.
"""
import queue
import threading
import time

_SENTINEL_END = object()
_SENTINEL_ABORT = object()

# Default timeout (seconds) when waiting for the next chunk; on timeout we check thermals/control.
# Reduced from 1.5s to 0.5s for more responsive hard stop detection
DEFAULT_CHUNK_TIMEOUT = 0.5

# Default max time (seconds) for a single job; after this we raise JobTimeoutError.
# Hardcoded for now; can be wired to a user setting later.
DEFAULT_JOB_TIMEOUT_SEC = 3600  # 1 hour


class JobTimeoutError(Exception):
    """Raised when a single prompt/job exceeds the allowed runtime."""
    pass


class ErrorCallInterrupted(Exception):
    """Raised when user pressed Error Call - save partial output as error and move on."""
    pass


def iterate_stream_with_thermal_checks(controller, response_stream, chunk_timeout=DEFAULT_CHUNK_TIMEOUT, job_timeout_sec=None):
    """
    Yield chunks from `response_stream`. While waiting for the next chunk, if a timeout occurs,
    run `controller._check_control_flags(True)`. On STOP/SKIP, cancel the stream and raise
    InterruptedError. On stream errors, re-raise the worker's exception.
    
    If job_timeout_sec is set, raise JobTimeoutError if no completion within that many seconds
    from the start of iteration (handles hanging backends that never return or never finish).

    CRITICAL: Also checks for hard stop after EACH chunk is yielded to ensure immediate
    response to thermal/power hard stops, not just when waiting for the next chunk.

    Use this instead of `for chunk in response_stream` so thermal/control checks happen even
    when the model is slow (e.g. long chain-of-thought) and no chunk arrives for a while.
    """
    job_timeout = job_timeout_sec if job_timeout_sec is not None else DEFAULT_JOB_TIMEOUT_SEC
    job_start = time.monotonic()
    q = queue.Queue()

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
        try:
            wait_sec = min(chunk_timeout, max(0.1, job_timeout - (time.monotonic() - job_start)))
            if wait_sec <= 0:
                try:
                    if hasattr(response_stream, "cancel"):
                        response_stream.cancel()
                except Exception:
                    pass
                raise JobTimeoutError(
                    f"Job exceeded maximum runtime ({job_timeout}s). No completion from backend in time."
                )
            x = q.get(timeout=wait_sec)
        except queue.Empty:
            # Timeout: check job timeout and control flags while waiting for next chunk
            elapsed = time.monotonic() - job_start
            if elapsed >= job_timeout:
                try:
                    if hasattr(response_stream, "cancel"):
                        response_stream.cancel()
                except Exception:
                    pass
                raise JobTimeoutError(
                    f"Job exceeded maximum runtime ({job_timeout}s). No completion from backend in time."
                )
            sig = controller._check_control_flags(True)
            if sig == "ERROR_CALL":
                try:
                    if hasattr(response_stream, "cancel"):
                        response_stream.cancel()
                except Exception:
                    pass
                raise ErrorCallInterrupted("Error call by user")
            if sig in ("STOP", "SKIP"):
                try:
                    if hasattr(response_stream, "cancel"):
                        response_stream.cancel()
                except Exception:
                    pass
                raise InterruptedError("Stopped by user")
            continue

        if x is _SENTINEL_END:
            break
        if isinstance(x, tuple) and len(x) == 2 and x[0] is _SENTINEL_ABORT:
            raise x[1]
        if x is _SENTINEL_ABORT:
            raise RuntimeError("Stream aborted")
        
        # CRITICAL: Check for hard stop AFTER each chunk to ensure immediate response
        sig = controller._check_control_flags(True)
        if sig == "ERROR_CALL":
            try:
                if hasattr(response_stream, "cancel"):
                    response_stream.cancel()
            except Exception:
                pass
            raise ErrorCallInterrupted("Error call by user")
        if sig in ("STOP", "SKIP"):
            try:
                if hasattr(response_stream, "cancel"):
                    response_stream.cancel()
            except Exception:
                pass
            raise InterruptedError("Stopped by user")
        # Enforce job timeout after each chunk as well
        if (time.monotonic() - job_start) >= job_timeout:
            try:
                if hasattr(response_stream, "cancel"):
                    response_stream.cancel()
            except Exception:
                pass
            raise JobTimeoutError(
                f"Job exceeded maximum runtime ({job_timeout}s). No completion from LM Studio in time."
            )
        
        yield x
