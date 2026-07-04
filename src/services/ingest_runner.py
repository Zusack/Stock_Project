"""Orchestration helpers for universe ingest (extracted from Data Management view)."""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from typing import Any

from src.analysis.ingest import ingest_stock_data
from src.utils.progress_format import format_eta_remaining

ProgressCallback = Callable[[float, str, dict | None, str | None], None]


def estimate_ingest_eta_seconds(
    *,
    elapsed_sec: float,
    finished_total: int,
    finished_baseline: int,
    remaining: int,
    min_session_samples: int = 1,
) -> float | None:
    """Estimate seconds left from throughput in the current UI session only."""
    if remaining <= 0:
        return 0.0
    if elapsed_sec < 0 or not (elapsed_sec == elapsed_sec):
        return None
    session_finished = max(0, int(finished_total) - int(finished_baseline))
    if session_finished < min_session_samples:
        return None
    return (elapsed_sec / session_finished) * remaining


def format_ingest_eta_label(
    *,
    elapsed_sec: float,
    finished_total: int,
    finished_baseline: int,
    remaining: int,
    total: int = 0,
    min_session_samples: int = 1,
) -> str:
    """User-facing ETA text for ingest progress."""
    if total <= 0 and remaining <= 0 and finished_total <= 0:
        return "No tickers to ingest"
    if remaining <= 0:
        return "Finishing up…"
    eta_sec = estimate_ingest_eta_seconds(
        elapsed_sec=elapsed_sec,
        finished_total=finished_total,
        finished_baseline=finished_baseline,
        remaining=remaining,
        min_session_samples=min_session_samples,
    )
    if eta_sec is None:
        return "Estimating time remaining…"
    return format_eta_remaining(eta_sec)


def run_universe_ingest(
    *,
    db_path: str,
    use_parallel: bool,
    progress_callback: ProgressCallback,
    mode: str,
    resume: bool = False,
    scope: str = "universe",
    tickers: list[str] | None = None,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Run ingest and return summary dict."""
    return ingest_stock_data(
        db_path=db_path,
        tickers=tickers,
        mode=mode,
        use_parallel=use_parallel,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
        resume=resume,
        scope=scope,
    )


def parse_ingest_total_from_status(status_msg: str, ingest_total: int) -> int:
    """Extract total ticker count from ingest status messages."""
    if not status_msg:
        return ingest_total
    m = re.search(r"(\d+)/(\d+)\s+finished", status_msg)
    if m:
        return max(ingest_total, int(m.group(2)))
    m2 = re.search(r"(\d+)\s+tickers", status_msg, re.IGNORECASE)
    if m2:
        return max(ingest_total, int(m2.group(1)))
    return ingest_total
