"""Database operation telemetry, retries, and SLA targets."""

from __future__ import annotations

import sqlite3
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, TypeVar

T = TypeVar("T")

# Target P95 ceilings (seconds) for regression checks in perf_benchmark.py
OPERATION_SLAS: dict[str, float] = {
    "bulk_remove_dead_no_history": 30.0,
    "bulk_skip_dead_tickers": 5.0,
    "count_summary": 2.0,
    "get_db_summary": 3.0,
    "dashboard_fetch": 8.0,
    "load_prices_bulk": 5.0,
    "ingest_per_ticker": 8.0,
}

_MAX_RECENT = 200
_recent_ops: deque[dict[str, Any]] = deque(maxlen=_MAX_RECENT)
_stats_lock = threading.Lock()
_slow_threshold_ms = 500.0


def is_db_locked_error(exc: BaseException) -> bool:
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    msg = str(exc).lower()
    return "locked" in msg or "busy" in msg


@dataclass
class DbReadResult:
    """Result wrapper for read paths that must distinguish lock vs empty."""

    data: Any = None
    locked: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return not self.locked and self.error is None


@dataclass
class DbRunSummary:
    """Standard post-run summary for UI and logs."""

    operation: str
    duration_sec: float = 0.0
    items_processed: int = 0
    items_total: int = 0
    success_count: int = 0
    failure_count: int = 0
    lock_wait_ms: float = 0.0
    retry_count: int = 0
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "duration_sec": round(self.duration_sec, 3),
            "items_processed": self.items_processed,
            "items_total": self.items_total,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "lock_wait_ms": round(self.lock_wait_ms, 1),
            "retry_count": self.retry_count,
            "notes": self.notes,
            **self.extra,
        }

    def format_short(self) -> str:
        parts = [
            f"{self.operation}: {self.duration_sec:.1f}s",
            f"{self.items_processed}/{self.items_total} processed",
        ]
        if self.lock_wait_ms > 0:
            parts.append(f"lock wait {self.lock_wait_ms:.0f}ms")
        if self.retry_count:
            parts.append(f"{self.retry_count} retries")
        if self.notes:
            parts.append(self.notes[:80])
        return " · ".join(parts)


def record_db_op(
    operation: str,
    *,
    duration_ms: float,
    rows: int | None = None,
    retries: int = 0,
    wait_ms: float = 0.0,
    caller: str | None = None,
    result: str = "ok",
) -> None:
    entry = {
        "operation": operation,
        "duration_ms": round(duration_ms, 2),
        "rows": rows,
        "retries": retries,
        "wait_ms": round(wait_ms, 2),
        "caller": caller,
        "result": result,
        "ts": time.time(),
    }
    with _stats_lock:
        _recent_ops.append(entry)

    if duration_ms >= _slow_threshold_ms:
        try:
            from src.utils.logger_utils import app_logger

            app_logger.log(
                "DB_PERF",
                f"Slow DB op: {operation}",
                level="WARN",
                duration_ms=round(duration_ms, 1),
                rows=rows,
                retries=retries,
                caller=caller,
                result=result,
            )
        except Exception:
            pass


@contextmanager
def track_db_op(operation: str, *, caller: str | None = None) -> Iterator[dict[str, Any]]:
    """Context manager that records duration and allows attaching row counts."""
    meta: dict[str, Any] = {"rows": None, "retries": 0, "wait_ms": 0.0, "result": "ok"}
    start = time.perf_counter()
    try:
        yield meta
    except Exception:
        meta["result"] = "error"
        raise
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        record_db_op(
            operation,
            duration_ms=duration_ms,
            rows=meta.get("rows"),
            retries=int(meta.get("retries") or 0),
            wait_ms=float(meta.get("wait_ms") or 0.0),
            caller=caller,
            result=str(meta.get("result") or "ok"),
        )


def execute_with_retry(
    operation: str,
    fn: Callable[[], T],
    *,
    retries: int = 3,
    retry_delay: float = 0.5,
    caller: str | None = None,
) -> T:
    """Run *fn* with retries on SQLITE_BUSY / locked errors."""
    last_error: Exception | None = None
    total_wait = 0.0
    for attempt in range(retries):
        try:
            start = time.perf_counter()
            result = fn()
            duration_ms = (time.perf_counter() - start) * 1000
            record_db_op(
                operation,
                duration_ms=duration_ms,
                retries=attempt,
                wait_ms=total_wait,
                caller=caller,
            )
            return result
        except sqlite3.OperationalError as exc:
            last_error = exc
            if not is_db_locked_error(exc) or attempt >= retries - 1:
                record_db_op(
                    operation,
                    duration_ms=0,
                    retries=attempt + 1,
                    wait_ms=total_wait,
                    caller=caller,
                    result="error",
                )
                raise
            delay = retry_delay * (attempt + 1)
            total_wait += delay * 1000
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def get_recent_db_ops(limit: int = 50) -> list[dict[str, Any]]:
    with _stats_lock:
        return list(_recent_ops)[-limit:]


def check_sla(operation: str, duration_sec: float) -> bool:
    """Return True if duration is within SLA ceiling."""
    ceiling = OPERATION_SLAS.get(operation)
    if ceiling is None:
        return True
    return duration_sec <= ceiling
