"""Leaderboard scoring orchestration (extracted from LeaderboardView)."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import pandas as pd

from src.analysis.leaderboard import LeaderboardSegment, build_leaderboard

LeaderboardProgressCallback = Callable[..., None]


def run_leaderboard_build(
    db_path: str,
    market_ticker: str,
    *,
    tickers: list[str] | None = None,
    segment: LeaderboardSegment | None = None,
    limit: int = 100,
    progress_callback: LeaderboardProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
    use_parallel: bool = True,
    workers: int | None = None,
    bulk_preload: bool = True,
    score_version: str | None = None,
) -> pd.DataFrame:
    """Score and rank the universe; thin wrapper over build_leaderboard."""
    return build_leaderboard(
        db_path,
        market_ticker,
        tickers=tickers,
        segment=segment,
        limit=limit,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
        use_parallel=use_parallel,
        workers=workers,
        bulk_preload=bulk_preload,
        score_version=score_version,
    )


def run_leaderboard_scoring(
    db_path: str,
    market_ticker: str,
    *,
    segment: LeaderboardSegment | None = None,
    tickers: list[str] | None = None,
    limit: int = 100,
    progress_callback: LeaderboardProgressCallback | None = None,
    **kwargs: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score leaderboard for a segment; returns (dataframe, metadata)."""
    seg = segment or LeaderboardSegment.CANDIDATES
    if progress_callback:
        progress_callback(0.05, "Loading universe…")

    df = run_leaderboard_build(
        db_path,
        market_ticker,
        tickers=tickers,
        segment=seg,
        limit=limit,
        progress_callback=progress_callback,
        **kwargs,
    )
    meta = {
        "segment": seg.value if hasattr(seg, "value") else str(seg),
        "symbol_count": len(df) if df is not None else 0,
    }
    if progress_callback:
        progress_callback(1.0, "Scoring complete.")
    return df, meta
